# Equivalence gate + precondition guards + a real-count timing window for the
# stage-4 / phase-2 candidate `gtp-exchange-pipeline`.
#
# spec.toml sets workload_mode = "inference", so the gate is EQUIVALENCE, not convergence.
# The candidate restructures `newton_schulz_tp_subgroup_fused` -- the LAST hot region still
# running the pre-stage-3 shape -- into the accepted EGTP form: async chunked exchange,
# staging-loop elision, and one batched+fused Newton-Schulz per shape per chunk instead of
# a per-matrix loop. The deal, the matrices, the step count, the coefficients and the SYRK
# path are all untouched; what changes is how the two collectives are staged and scheduled
# and how the identical 5-step chain is batched.
#
# That split matters for the gate, because the two mechanism families have DIFFERENT
# expected exactness and must be separated rather than averaged:
#
#   F) EXCHANGE EXACTNESS, at --steps 0. With zero Newton-Schulz steps both arms reduce to
#      the same elementwise `normalize -> bf16 -> fp32`, which no batching or grouping can
#      perturb. So any difference at steps = 0 is a DATA-MOVEMENT bug: a wrong window, a
#      wrong peer, a wrong transpose, a wrong scatter. This row must be BITWISE, and it is
#      the load-bearing row for the restructure itself.
#   A) THE SUBSTITUTED ARM, at the real --steps. Pipelined vs the monolithic fused arm of
#      the same (g, balanced). Expected within tolerance, not bitwise: the per-matrix
#      `newton_schulz` loop becomes one batched fused call, which is the same
#      bf16-tiling-level difference stage-3's already-accepted `_fused` and batched arms
#      show (a different cuBLAS tile for `baddbmm` at a different batch size).
#   B) the per-matrix `duplicated` lineage every subgroup arm has been gated against since
#      stage 2.
#   C) matrix 0 of each shape against the frozen Phase-0 reference tensor.
#   D) CONTROL: the monolithic fused arm vs the same lineage. Any residual in B that also
#      shows in D is pre-existing and not introduced here.
#   E) GUARDS: the preconditions are hard errors, never a silent fallback to the monolithic
#      path. A silent fallback would let the arm be scored while running the very code it
#      claims to replace.
#
# The timing leg then measures, at the real owned counts, every pipeline depth against the
# monolithic arm it substitutes. At a 4-rank world it is NOT a result -- group size, shard
# size and redundancy all differ from the modelled workload; only the 16-node job scores.

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
    newton_schulz_tp_subgroup,
    newton_schulz_tp_subgroup_fused,
)
from emerging_optimizers.orthogonalized_optimizers.muon_utils import (  # noqa: E402
    newton_schulz_tp,
)


def build_stack(shape, count, world, rank, seed):
    """Matrix 0 is the frozen Phase-0 input; the rest are distinct so a deal or window bug
    that permutes matrices across subgroups, across CHUNKS or across SHAPES cannot pass by
    coincidence."""
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
    ap.add_argument("--out", required=True, help="equivalence.csv to write")
    ap.add_argument("--timing-out", default="", help="optional window_timing.json")
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--shapes", nargs="+", required=True)
    ap.add_argument("--counts", nargs="+", type=int, required=True,
                    help="Owned count per shape; deliberately UNEQUAL so the chunk windows "
                         "cut the per-shape splits at different places.")
    ap.add_argument("--subgroup-sizes", nargs="+", type=int, required=True)
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
    assert len(args.shapes) >= 2, "the fusion is cross-shape; give at least two shapes"

    torch.set_float32_matmul_precision(args.fp32_matmul_prec)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl")
    group = torch.distributed.distributed_c10d._get_default_group()
    world, rank = group.size(), group.rank()

    def log(message):
        if rank == 0:
            print(message, flush=True)

    assert HAVE_FUSED_NS, (
        "kernels.fused_ns is unavailable; the pipelined arm batches the per-matrix "
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
            "axis": "gtp",
            "scope": scope,
            "arm": arm,
            "world": world,
            "max_abs_diff": f"{max_abs:.6e}",
            "mean_abs_diff": f"{d.mean().item():.6e}",
            "max_rel_diff": f"{(d / denom).max().item():.6e}",
            "bitwise_equal": bitwise,
            "bitwise_required": must_be_bitwise,
            "atol": args.atol,
            "rtol": args.rtol,
            "pass": "PASS" if ok else "FAIL",
        }
        rows_out.append(row)
        log(f"  {scope:>14} {arm:<62} max_abs={row['max_abs_diff']} "
            f"bitwise={bitwise} {row['pass']}")

    stacks = [
        build_stack(shape, count, world, rank, args.seed)
        for shape, count in zip(shapes, args.counts)
    ]
    log(f"world={world} shapes={args.shapes} counts={args.counts} "
        f"pipe_chunks={args.pipe_chunks}")

    # ---- B/D reference: the per-matrix `duplicated` lineage ----------------------------
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

    timing = {"world": world, "shapes": args.shapes, "counts": args.counts, "arms": {}}

    def fused(balanced, g, pc, steps):
        return newton_schulz_tp_subgroup_fused(
            stacks, steps=steps, coefficient_type=args.coefficient_type,
            tp_group=group, use_syrk=True, subgroup_size=g, balanced=balanced,
            pipelined=bool(pc), pipe_chunks=pc,
        )

    for g in args.subgroup_sizes:
        if world % g or g >= world:
            log(f"skipping g={g}: does not divide {world} or is not smaller than it")
            continue
        log(f"g={g} (k={world // g}):")
        per_shape = [
            newton_schulz_tp_subgroup(
                stacks[i], steps=args.steps, coefficient_type=args.coefficient_type,
                tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
                subgroup_size=g,
            )
            for i in range(len(shapes))
        ]

        for balanced in (False, True):
            base_tag = f"fuse_sub_g{g}" + ("_bal" if balanced else "")
            mono = fused(balanced, g, 0, args.steps)
            for i, (m, n) in enumerate(shapes):
                record(f"{m}x{n}", f"D control: {base_tag} vs duplicated lineage",
                       mono[i], lineage[i])
            # steps = 0 isolates the DATA MOVEMENT: both arms collapse to the same
            # elementwise normalize -> bf16 -> fp32, so batching cannot perturb it. If the
            # library refuses a 0-step chain the row is recorded as a hard FAIL rather than
            # skipped, so the discriminator can never be lost silently.
            try:
                mono0 = fused(balanced, g, 0, 0)
            except Exception as error:  # noqa: BLE001 - recorded, never swallowed
                mono0 = None
                rows_out.append({
                    "axis": "gtp", "scope": "exchange-only", "world": world,
                    "arm": f"F exchange-only(steps=0): {base_tag} reference unavailable: "
                           f"{type(error).__name__}: {error}",
                    "max_abs_diff": "", "mean_abs_diff": "", "max_rel_diff": "",
                    "bitwise_equal": "", "bitwise_required": True, "atol": args.atol,
                    "rtol": args.rtol, "pass": "FAIL",
                })
                log(f"  F exchange-only reference at steps=0 FAILED: {error}")

            for pc in args.pipe_chunks:
                tag = f"{base_tag}_pipe{pc}"
                if mono0 is not None:
                    pipe0 = fused(balanced, g, pc, 0)
                    for i, (m, n) in enumerate(shapes):
                        record(f"{m}x{n}",
                               f"F exchange-only(steps=0): {tag} vs {base_tag}",
                               pipe0[i], mono0[i], must_be_bitwise=True)
                    del pipe0
                got = fused(balanced, g, pc, args.steps)
                assert len(got) == len(shapes)
                for i, (m, n) in enumerate(shapes):
                    assert got[i].shape == stacks[i].shape, (
                        f"{tag}: shape {i} returned {tuple(got[i].shape)}, "
                        f"expected {tuple(stacks[i].shape)}"
                    )
                    record(f"{m}x{n}", f"A: {tag} vs the arm it substitutes ({base_tag})",
                           got[i], mono[i])
                    record(f"{m}x{n}", f"B: {tag} vs duplicated lineage", got[i], lineage[i])
                    if references[i] is not None:
                        record(f"{m}x{n}", f"C: {tag} matrix 0 vs phase0-reference",
                               got[i][0], references[i])
                del got
                torch.cuda.empty_cache()
            del mono, mono0
            torch.cuda.empty_cache()

        # ---- timing: every depth against the monolithic arm it substitutes -------------
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

            for balanced in (False, True):
                base_tag = f"fuse_sub_g{g}" + ("_bal" if balanced else "")
                for pc in [0] + list(args.pipe_chunks):
                    tag = base_tag + (f"_pipe{pc}" if pc else "")
                    timing["arms"].setdefault(tag, {})["ms"] = time_call(
                        lambda b=balanced, p=pc: fused(b, g, p, args.steps)
                    )
                    torch.cuda.empty_cache()
                log(f"  timing g={g} balanced={balanced}: " + ", ".join(
                    f"{base_tag + (f'_pipe{p}' if p else '')}="
                    f"{timing['arms'][base_tag + (f'_pipe{p}' if p else '')]['ms']:.3f}"
                    for p in [0] + list(args.pipe_chunks)
                ))

        del per_shape
        torch.cuda.empty_cache()

    # ---- E) guards: every precondition must RAISE, not fall back -----------------------
    g_ok = next((g for g in args.subgroup_sizes if world % g == 0 and g < world), world // 2)
    guards = [
        ("pipelined with pipe_chunks = 0", dict(stacks=stacks, subgroup_size=g_ok, pc=0)),
        ("pipelined with pipe_chunks < 0", dict(stacks=stacks, subgroup_size=g_ok, pc=-1)),
        ("pipelined single stack (not cross-shape)",
         dict(stacks=stacks[:1], subgroup_size=g_ok, pc=2)),
        ("pipelined subgroup_size == world", dict(stacks=stacks, subgroup_size=world, pc=2)),
        ("pipelined subgroup_size does not divide world",
         dict(stacks=stacks, subgroup_size=world + 1, pc=2)),
    ]
    for label, kwargs in guards:
        raised = ""
        try:
            newton_schulz_tp_subgroup_fused(
                kwargs["stacks"], steps=args.steps,
                coefficient_type=args.coefficient_type, tp_group=group, use_syrk=True,
                subgroup_size=kwargs["subgroup_size"], balanced=True,
                pipelined=True, pipe_chunks=kwargs["pc"],
            )
        except ValueError as error:
            raised = f"ValueError: {error}"
        rows_out.append({
            "axis": "gtp", "scope": "guard", "arm": f"E: {label}", "world": world,
            "max_abs_diff": "", "mean_abs_diff": "", "max_rel_diff": "",
            "bitwise_equal": "", "bitwise_required": "", "atol": args.atol,
            "rtol": args.rtol, "pass": "PASS" if raised else "FAIL",
        })
        log(f"  {'guard':>14} {('E: ' + label):<62} raised={raised or 'NOTHING'} "
            f"{'PASS' if raised else 'FAIL'}")

    if rank == 0:
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
        summary = {
            "gate": "equivalence",
            "atol": args.atol,
            "rtol": args.rtol,
            "world": world,
            "shapes": args.shapes,
            "counts": args.counts,
            "subgroup_sizes": args.subgroup_sizes,
            "pipe_chunks": args.pipe_chunks,
            "all_pass": all(r["pass"] == "PASS" for r in rows_out),
            "f_arm_all_bitwise": all(
                r["bitwise_equal"] is True for r in rows_out if r["arm"].startswith("F ")
            ),
            "max_abs_diff": max(
                (float(r["max_abs_diff"]) for r in rows_out if r["max_abs_diff"]), default=0.0
            ),
            "max_abs_diff_a_arm": max(
                (float(r["max_abs_diff"]) for r in rows_out if r["arm"].startswith("A: ")),
                default=0.0,
            ),
            "rows": len(rows_out),
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
