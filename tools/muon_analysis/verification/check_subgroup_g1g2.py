# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Equivalence gate + guards + timing window for stage-5 / phase-1 `gtp-subgroup-size-g1g2`.

The candidate adds NO code path. It widens `bench_ns_gtp.sbatch`'s `SUBGROUP_SIZES` from
"4 8 16" to "1 2 4 8 16", i.e. it adds two more columns to a sweep whose per-shape and
per-profile argmin already picks the winner. The only thing that can therefore go wrong is
NUMERIC: if the deal, the exchange or the return ordering were a function of g, a
g-selected arm would not be the same computation as the banked g = 4 one.

So the load-bearing row of this window is g-INVARIANCE, and it is required to be BITWISE:

  G  fuse_sub_g1<suffix> vs fuse_sub_g2<suffix>, same suffix, same tensors. Every matrix is
     all-gathered in RANK order and orthogonalized WHOLE by the same newton_schulz, the
     same 5 steps and the same polar_express coefficients whatever g is; g only decides
     WHICH subgroup owns a matrix and how the two all_to_all splits are cut. Any difference
     at all is a staging/ordering bug, not a numeric one.
  A  each g in {1, 2} arm vs the per-matrix `duplicated` lineage -- also BITWISE, which is
     the same claim `--fuse-shape-exchanges` makes for g = 4/8/16 and which phase-3's D
     control already observed for fuse_sub_g1_pipe2 at world 4.
  C  matrix 0 vs the frozen phase-0 reference tensor (tolerance gate, atol = rtol = 1e-3):
     the reference was banked through a different composition, so it is not a bitwise row.
  E  guards: g == world (groups == 1) and g not dividing world must RAISE. A silent
     fallback would let a scored column time a different design under this candidate's name.

At world 4 the legal g are exactly {1, 2} -- g = 4 is `groups == 1` -- so this window can
prove the two NEW columns and nothing else. That is the right coverage: g = 4/8/16 are the
banked columns and are unchanged by this diff. The TIMING half is not a result (the 64-rank
region has different shard sizes, redundancy and transport); only the 16-node job scores.
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
    that permutes matrices across subgroups, chunks or shapes cannot pass by coincidence.
    Distinctness is what makes the G row load-bearing: g = 1 and g = 2 deal the SAME set to
    DIFFERENT owners, so identical output can only mean order-independent placement."""
    m, n = shape
    local_rows = m // world
    stack = torch.empty((count, local_rows, n), device="cuda", dtype=torch.float32)
    gen = torch.Generator(device="cuda")
    tag = zlib.crc32(f"gtp_{m}x{n}".encode()) % 100000
    gen.manual_seed(seed + tag)
    full0 = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=gen)
    stack[0] = full0[rank * local_rows : (rank + 1) * local_rows]
    del full0
    for j in range(1, count):
        gen.manual_seed(seed + 7919 * j + tag)
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
    ap.add_argument("--subgroup-sizes", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--pipe-chunks", nargs="+", type=int, default=[0, 2, 4])
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

    def log(message):
        if rank == 0:
            print(message, flush=True)

    # D:997 records a false green from a silent fused-NS fallback; assert, never assume.
    assert HAVE_FUSED_NS, (
        "kernels.fused_ns is unavailable; the pipelined fused arms batch the per-matrix "
        "Newton-Schulz through it and must never be gated against a silent fallback."
    )

    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes]
    for (m, n) in shapes:
        assert m % world == 0, f"{m}x{n}: rows must divide the {world}-rank group"

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
        log(f"  {scope:>14} {arm:<78} max_abs={row['max_abs_diff']} "
            f"bitwise={bitwise} {row['pass']}")

    stacks = [
        build_stack(shape, count, world, rank, args.seed)
        for shape, count in zip(shapes, args.counts)
    ]
    log(f"world={world} shapes={args.shapes} counts={args.counts} "
        f"g={args.subgroup_sizes} pipe_chunks={args.pipe_chunks}")

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

    def fused(g, balanced, pc, steps=None):
        return newton_schulz_tp_subgroup_fused(
            stacks, steps=args.steps if steps is None else steps,
            coefficient_type=args.coefficient_type, tp_group=group, use_syrk=True,
            subgroup_size=g, balanced=balanced, pipelined=bool(pc), pipe_chunks=pc,
        )

    def tag_of(g, balanced, pc):
        return f"fuse_sub_g{g}" + ("_bal" if balanced else "") + (f"_pipe{pc}" if pc else "")

    legal = [g for g in args.subgroup_sizes if world % g == 0 and world // g > 1]
    for g in args.subgroup_sizes:
        if g not in legal:
            log(f"skipping g={g}: does not divide world {world}, or groups == 1")
    assert len(legal) >= 1, f"no legal g in {args.subgroup_sizes} at world {world}"

    # ---- A) every new-g arm vs the per-matrix duplicated lineage, BITWISE ---------------
    # Also cache the outputs keyed by (balanced, pc) so the G row can compare g against g
    # on the SAME suffix without recomputing.
    by_suffix = {}
    for g in legal:
        log(f"g={g} (k={world // g}):")
        for balanced in (False, True):
            for pc in args.pipe_chunks:
                tag = tag_of(g, balanced, pc)
                out = fused(g, balanced, pc)
                for i, (m, n) in enumerate(shapes):
                    assert out[i].shape == stacks[i].shape, (
                        f"{tag}: shape {i} returned {tuple(out[i].shape)}, not the input "
                        "shard shape -- the region must return this rank's slice of every "
                        "matrix in INPUT order"
                    )
                    assert out[i].dtype is torch.float32, f"{tag}: shape {i} not fp32"
                    record(f"{m}x{n}", f"A: {tag} vs duplicated lineage", out[i],
                           lineage[i], must_be_bitwise=True)
                    if references[i] is not None:
                        record(f"{m}x{n}", f"C: {tag} matrix 0 vs phase0-reference",
                               out[i][0], references[i])
                by_suffix.setdefault((balanced, pc), {})[g] = out
                torch.cuda.empty_cache()

    # ---- G) g-invariance: the load-bearing row ------------------------------------------
    base_g = legal[0]
    for (balanced, pc), per_g in by_suffix.items():
        for g in legal[1:]:
            for i, (m, n) in enumerate(shapes):
                record(f"{m}x{n}",
                       f"G: {tag_of(g, balanced, pc)} vs {tag_of(base_g, balanced, pc)} "
                       "(g-invariance)",
                       per_g[g][i], per_g[base_g][i], must_be_bitwise=True)
    del by_suffix
    torch.cuda.empty_cache()

    # ---- E) guards: preconditions must RAISE --------------------------------------------
    guards = [
        (f"g == world ({world}) is plain 'duplicated' (groups == 1)", world),
        (f"g = {world + 1} does not divide world {world}", world + 1),
    ]
    for label, g_bad in guards:
        raised = ""
        try:
            fused(g_bad, True, 2)
        except ValueError as error:
            raised = f"ValueError: {error}"
        rows_out.append({
            "axis": "gtp", "scope": "guard", "arm": f"E: {label}", "world": world,
            "max_abs_diff": "", "mean_abs_diff": "", "max_rel_diff": "",
            "bitwise_equal": "", "bitwise_required": "", "atol": args.atol,
            "rtol": args.rtol, "pass": "PASS" if raised else "FAIL",
        })
        log(f"  {'guard':>14} {('E: ' + label):<78} raised={raised or 'NOTHING'} "
            f"{'PASS' if raised else 'FAIL'}")

    # ---- timing (NOT a result; world-4 shard sizes and transport differ) -----------------
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

        for g in legal:
            for balanced in (False, True):
                for pc in args.pipe_chunks:
                    tag = tag_of(g, balanced, pc)
                    timing["arms"][tag] = {
                        "ms": time_call(lambda gg=g, b=balanced, p=pc: fused(gg, b, p))
                    }
                    torch.cuda.empty_cache()
            log(f"  timing g={g}: " + ", ".join(
                f"{tag_of(g, b, pc)}={timing['arms'][tag_of(g, b, pc)]['ms']:.3f}"
                for b in (False, True) for pc in args.pipe_chunks
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
            "shapes": args.shapes, "counts": args.counts,
            "subgroup_sizes_legal_here": legal, "pipe_chunks": args.pipe_chunks,
            "rows": len(rows_out),
            "all_pass": all(r["pass"] == "PASS" for r in rows_out),
            "g_invariance_all_bitwise": all(
                r["bitwise_equal"] is True for r in rows_out if r["arm"].startswith("G: ")
            ),
            "lineage_all_bitwise": all(
                r["bitwise_equal"] is True for r in rows_out if r["arm"].startswith("A: ")
            ),
            "guards_all_raised": all(
                r["pass"] == "PASS" for r in rows_out if r["scope"] == "guard"
            ),
            "max_abs_diff": worst(lambda r: True),
            "max_abs_diff_reference_row": worst(lambda r: r["arm"].startswith("C: ")),
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
