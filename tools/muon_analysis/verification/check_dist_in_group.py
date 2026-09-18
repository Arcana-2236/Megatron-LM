# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Equivalence gate + guards + timing window for stage-5 / phase-2 `gtp-distribute-in-group-g2`.

The candidate adds a DISTRIBUTION, not a schedule: at `fuse_sub_g<g>_dist...` the g ranks of
a duplication subgroup each keep their own slab of a TALL matrix and cooperate through one
Gram all-reduce per Newton-Schulz step, instead of each orthogonalizing the whole matrix.
Wide-once-gathered shapes are untouched and stay replicated in the subgroup.

Two rows are therefore load-bearing and they pull in opposite directions:

  W  UNTOUCHED means BITWISE. For a shape the arm does NOT distribute, `fuse_sub_g<g>_dist`
     must be bit-for-bit `fuse_sub_g<g>` at the same suffix and the same deal
     (`balanced=False`, where both arms deal round-robin and therefore give every matrix to
     the same subgroup). Anything else means the split of the exchange perturbed a shape it
     was not supposed to touch.
  N  DISTRIBUTED means NUMERIC, and the gate is the spec's atol = rtol = 1e-3. Exactly two
     things deviate, both of them the deviations `tp_mode="distributed"` already has against
     `"duplicated"`: the Frobenius norm is a collective sum of per-shard sums of squares
     (taken in fp64 here) rather than one reduction over the whole matrix, and the Gram is
     summed over g partial products rather than formed in one. stage-2 measured
     3.66e-4...9.77e-4 for FULL-WORLD (64-way) `distributed`; at g = 2 the Gram is a sum of
     two terms, not 64, so this must land well under that -- but that is a measurement, not
     an argument, which is why this window runs before the 16-node job.

Rows:
  A  each dist arm vs the per-matrix `duplicated` lineage        (tolerance)
  B  each dist arm vs the REPLICATED arm at the same g and suffix (tolerance; the pure
     distribution deviation, isolated from the lineage's own composition difference)
  W  wide shapes only, dist vs replicated at balanced=False       (BITWISE, required)
  C  matrix 0 vs the frozen phase-0 reference tensor              (tolerance -- the gate)
  D  deal-invariance: dist at balanced=False vs balanced=True     (tolerance)
  E  guards: g = 1, non-pipelined, and an all-wide region must RAISE. A silent fallback
     would let the replicated distribution be scored under this candidate's name.

At world 4 the only legal distribute-in-group column is g = 2 (g = 1 has nothing to
distribute, g = 4 is `groups == 1`), which is exactly the column the candidate adds to the
64-rank job. The TIMING half is NOT a result: at world 4 the shard sizes, the deal and the
transport all differ; only the 16-node job scores.
"""

import argparse
import csv
import json
import os
import statistics
import sys
import zlib

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "tools", "muon_analysis"))

from bench_ns_strategies import (  # noqa: E402
    HAVE_FUSED_NS,
    newton_schulz_tp_subgroup_fused,
)
from emerging_optimizers.orthogonalized_optimizers.muon_utils import (  # noqa: E402
    newton_schulz_tp,
)


def build_stack(shape, count, world, rank, seed):
    """Matrix 0 is the frozen Phase-0 input; the rest are distinct, so a deal or staging bug
    that permutes matrices across subgroups, chunks or shapes cannot pass by coincidence."""
    m, n = shape
    local_rows = m // world
    stack = torch.empty((count, local_rows, n), device="cuda", dtype=torch.float32)
    gen = torch.Generator(device="cuda")
    tag = zlib.crc32(f"gtp_{m}x{n}".encode()) % 100000
    for j in range(count):
        gen.manual_seed(seed + tag if j == 0 else seed + 7919 * j + tag)
        fj = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=gen)
        stack[j] = fj[rank * local_rows : (rank + 1) * local_rows]
        del fj
    torch.cuda.empty_cache()
    return stack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--timing-out", default="")
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--shapes", nargs="+", required=True)
    ap.add_argument("--counts", nargs="+", type=int, required=True)
    ap.add_argument("--subgroup-size", type=int, default=2)
    ap.add_argument("--pipe-chunks", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--coefficient-type", default="polar_express")
    ap.add_argument("--fp32-matmul-prec", default="medium")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()

    assert len(args.shapes) == len(args.counts), "--shapes and --counts must pair up"
    assert len(args.shapes) >= 2, "the fused exchange is cross-shape; give >= 2 shapes"

    # phase-3 lost a probe to omitting this: the benchmark sets it, so the window must too
    # or the fused kernel's precondition and the lineage's GEMMs disagree.
    torch.set_float32_matmul_precision(args.fp32_matmul_prec)
    assert torch.get_float32_matmul_precision() == args.fp32_matmul_prec

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl")
    group = torch.distributed.distributed_c10d._get_default_group()
    world, rank = group.size(), group.rank()
    g = args.subgroup_size

    def log(message):
        if rank == 0:
            print(message, flush=True)

    # D:997 records a false green from a silent fused-NS fallback; assert, never assume.
    assert HAVE_FUSED_NS, (
        "kernels.fused_ns is unavailable; the in-group chain lives in it and must never be "
        "gated against a silent fallback."
    )
    assert world % g == 0 and world // g > 1, (
        f"subgroup size {g} is not a legal distribute-in-group column at world {world}"
    )

    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes]
    for (m, n) in shapes:
        assert m % world == 0, f"{m}x{n}: rows must divide the {world}-rank group"
    tall = [i for i, (m, n) in enumerate(shapes) if m > n]
    wide = [i for i, (m, n) in enumerate(shapes) if m <= n]
    assert tall, "give at least one shape that is TALL once gathered -- the arm distributes"
    assert wide, "give at least one WIDE shape -- the W row proves it is left untouched"
    log(f"world={world} g={g} shapes={args.shapes} counts={args.counts} "
        f"tall={[args.shapes[i] for i in tall]} wide={[args.shapes[i] for i in wide]}")

    rows_out = []

    def record(scope, arm, a, b, must_be_bitwise=False):
        d = (a.float() - b.float()).abs()
        denom = b.float().abs().clamp_min(1e-12)
        max_abs = d.max().item()
        bitwise = bool(torch.equal(a, b))
        ok = max_abs <= args.atol and (bitwise or not must_be_bitwise)
        row = {
            "axis": "gtp", "scope": scope, "arm": arm, "world": world,
            "max_abs_diff": f"{max_abs:.6e}",
            "mean_abs_diff": f"{d.mean().item():.6e}",
            "max_rel_diff": f"{(d / denom).max().item():.6e}",
            "bitwise_equal": bitwise, "bitwise_required": must_be_bitwise,
            "atol": args.atol, "rtol": args.rtol,
            "pass": "PASS" if ok else "FAIL",
        }
        rows_out.append(row)
        log(f"  {scope:>14} {arm:<74} max_abs={row['max_abs_diff']} "
            f"bitwise={bitwise} {row['pass']}")

    stacks = [
        build_stack(shape, count, world, rank, args.seed)
        for shape, count in zip(shapes, args.counts)
    ]

    lineage = [
        torch.stack([
            newton_schulz_tp(
                stacks[i][j], steps=args.steps, coefficient_type=args.coefficient_type,
                tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
            )
            for j in range(args.counts[i])
        ])
        for i in range(len(shapes))
    ]
    torch.cuda.empty_cache()

    references = []
    for (m, n) in shapes:
        path = os.path.join(args.reference_dir, f"gtp_{m}x{n}.pt")
        if os.path.exists(path):
            local_rows = m // world
            references.append(
                torch.load(path, map_location="cpu")[
                    rank * local_rows : (rank + 1) * local_rows
                ].clone().cuda()
            )
        else:
            log(f"  no frozen reference at {path}")
            references.append(None)

    def fused(gg, balanced, pc, dist):
        return newton_schulz_tp_subgroup_fused(
            stacks, steps=args.steps, coefficient_type=args.coefficient_type,
            tp_group=group, use_syrk=True, subgroup_size=gg, balanced=balanced,
            pipelined=bool(pc), pipe_chunks=pc, dist_in_group=dist,
        )

    def tag_of(gg, balanced, pc, dist):
        return (f"fuse_sub_g{gg}" + ("_dist" if dist else "")
                + ("_bal" if balanced else "") + (f"_pipe{pc}" if pc else ""))

    # ---- A / B / W / C -------------------------------------------------------------------
    keep = {}
    for balanced in (False, True):
        for pc in args.pipe_chunks:
            rep = fused(g, balanced, pc, False)
            dist = fused(g, balanced, pc, True)
            dtag = tag_of(g, balanced, pc, True)
            rtag = tag_of(g, balanced, pc, False)
            for i, (m, n) in enumerate(shapes):
                assert dist[i].shape == stacks[i].shape, (
                    f"{dtag}: shape {i} returned {tuple(dist[i].shape)}, not the input "
                    "shard shape -- the region must return this rank's slice of every "
                    "matrix in INPUT order"
                )
                assert dist[i].dtype is torch.float32, f"{dtag}: shape {i} not fp32"
                kind = "tall/distributed" if i in tall else "wide/replicated"
                record(f"{m}x{n}", f"A: {dtag} vs duplicated lineage [{kind}]",
                       dist[i], lineage[i])
                record(f"{m}x{n}", f"B: {dtag} vs {rtag} [{kind}]", dist[i], rep[i],
                       must_be_bitwise=(i in wide and not balanced))
                if references[i] is not None:
                    record(f"{m}x{n}", f"C: {dtag} matrix 0 vs phase0-reference",
                           dist[i][0], references[i])
            if pc == args.pipe_chunks[0]:
                keep[balanced] = dist
            del rep, dist
            torch.cuda.empty_cache()

    # ---- D) deal-invariance --------------------------------------------------------------
    pc0 = args.pipe_chunks[0]
    for i, (m, n) in enumerate(shapes):
        record(f"{m}x{n}",
               f"D: {tag_of(g, True, pc0, True)} vs {tag_of(g, False, pc0, True)} "
               "(deal-invariance)",
               keep[True][i], keep[False][i])
    del keep
    torch.cuda.empty_cache()

    # ---- E) guards: preconditions must RAISE ----------------------------------------------
    def guard(label, fn):
        raised = ""
        try:
            fn()
        except ValueError as error:
            raised = f"ValueError: {error}"
        rows_out.append({
            "axis": "gtp", "scope": "guard", "arm": f"E: {label}", "world": world,
            "max_abs_diff": "", "mean_abs_diff": "", "max_rel_diff": "",
            "bitwise_equal": "", "bitwise_required": "", "atol": args.atol,
            "rtol": args.rtol, "pass": "PASS" if raised else "FAIL",
        })
        log(f"  {'guard':>14} {('E: ' + label):<74} raised={raised or 'NOTHING'} "
            f"{'PASS' if raised else 'FAIL'}")

    guard("g = 1 has nothing to distribute inside", lambda: fused(1, True, pc0, True))
    guard("dist_in_group needs the pipelined exchange", lambda: fused(g, True, 0, True))
    guard(
        "an all-WIDE region has nothing to distribute",
        lambda: newton_schulz_tp_subgroup_fused(
            [stacks[i] for i in wide] * (1 if len(wide) >= 2 else 2),
            steps=args.steps, coefficient_type=args.coefficient_type, tp_group=group,
            use_syrk=True, subgroup_size=g, balanced=True, pipelined=True,
            pipe_chunks=pc0, dist_in_group=True,
        ),
    )

    # ---- timing (NOT a result; world-4 shard sizes, deal and transport differ) -------------
    timing = {"world": world, "shapes": args.shapes, "counts": args.counts, "arms": {}}
    if not args.skip_timing:
        def time_call(fn):
            for _ in range(args.warmup):
                fn()
            torch.cuda.synchronize()
            torch.distributed.barrier(group=group)
            out = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(args.iters):
                torch.distributed.barrier(group=group)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                out.append(start.elapsed_time(end))
            return statistics.median(out)

        for dist in (False, True):
            for balanced in (False, True):
                for pc in args.pipe_chunks:
                    tag = tag_of(g, balanced, pc, dist)
                    timing["arms"][tag] = {
                        "ms": time_call(
                            lambda b=balanced, p=pc, d=dist: fused(g, b, p, d)
                        )
                    }
                    torch.cuda.empty_cache()
        # g = 1 is the banked column; timed here only so the window's own numbers are
        # readable beside it. It is not a result either.
        for balanced in (False, True):
            for pc in args.pipe_chunks:
                tag = tag_of(1, balanced, pc, False)
                timing["arms"][tag] = {
                    "ms": time_call(lambda b=balanced, p=pc: fused(1, b, p, False))
                }
                torch.cuda.empty_cache()
        log("  timing: " + ", ".join(
            f"{k}={v['ms']:.3f}" for k, v in timing["arms"].items()
        ))

    if rank == 0:
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)

        def worst(pred):
            return max((float(r["max_abs_diff"]) for r in rows_out
                        if r["max_abs_diff"] and pred(r)), default=0.0)

        summary = {
            "gate": "equivalence", "atol": args.atol, "rtol": args.rtol, "world": world,
            "subgroup_size": g, "shapes": args.shapes, "counts": args.counts,
            "pipe_chunks": args.pipe_chunks, "rows": len(rows_out),
            "all_pass": all(r["pass"] == "PASS" for r in rows_out),
            "untouched_wide_all_bitwise": all(
                r["bitwise_equal"] is True for r in rows_out if r["bitwise_required"] is True
            ),
            "guards_all_raised": all(
                r["pass"] == "PASS" for r in rows_out if r["scope"] == "guard"
            ),
            "max_abs_diff": worst(lambda r: True),
            "max_abs_diff_reference_row": worst(lambda r: r["arm"].startswith("C: ")),
            "max_abs_diff_distribution_row": worst(lambda r: r["arm"].startswith("B: ")),
        }
        print("EQUIVALENCE_SUMMARY=" + json.dumps(summary), flush=True)
        if args.timing_out:
            with open(args.timing_out, "w") as fh:
                json.dump(timing, fh, indent=2)
            print("TIMING=" + json.dumps(timing), flush=True)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
