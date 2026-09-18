# Equivalence gate + precondition guards + a real-count timing window for the
# stage-3 / phase-4 candidate `gtp-fuse-shape-exchanges`.
#
# spec.toml sets workload_mode = "inference", so the gate is EQUIVALENCE, not convergence.
# The candidate replaces N per-shape subgroup regions (one pair of all_to_all_single each)
# with ONE region carrying one pair of exchanges over all of them, optionally with the
# ownership deal pooled across shapes. Which subgroup owns which matrix and how the two
# collectives are packed are both INTERNAL: each shape's return value is still this rank's
# slice of every matrix, in input order. So the gate that matters is exactness against the
# per-shape arm the candidate substitutes.
#
# Four comparisons, all on the same tensors in the same job:
#
#   A) fused vs THE ARM IT SUBSTITUTES -- looping `newton_schulz_tp_subgroup(stack_i, g)`
#      once per shape. Expected BITWISE identical for the round-robin deal (`balanced=
#      False`), which deals every shape exactly as the per-shape arm does; for the pooled
#      deal (`balanced=True`) it is still bitwise, because a matrix's own arithmetic does
#      not depend on which subgroup runs it.
#   B) fused vs the per-matrix `duplicated` lineage -- `newton_schulz_tp(stack_i[j],
#      tp_mode="duplicated")`. This is the lineage every subgroup arm has been gated
#      against since stage 2.
#   C) fused matrix 0 of each shape vs the frozen Phase-0 reference tensor, which ties the
#      candidate to the Phase-0 lineage rather than only to a sibling arm.
#   D) CONTROL: the per-shape subgroup arm vs the same lineage (B's reference). Any
#      residual in B that also appears in D is pre-existing and not introduced here.
#
#   E) GUARDS -- the preconditions are hard errors, never a silent fallback to the
#      per-shape path. A silent fallback would let the arm be scored while running the
#      very code it claims to replace.
#
# The timing leg then measures, at the real owned counts, the fused region against the sum
# of the per-shape regions it replaces, for every g and both deals -- which is what makes a
# negative outcome diagnosable: an exchange-only win shows on the round-robin arm, a deal
# win only on the pooled one.

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
    fused_subgroup_deal,
    newton_schulz_tp_subgroup,
    newton_schulz_tp_subgroup_fused,
)
from emerging_optimizers.orthogonalized_optimizers.muon_utils import (  # noqa: E402
    newton_schulz_tp,
)


def build_stack(shape, count, world, rank, seed):
    """Matrix 0 is the frozen Phase-0 input; the rest are distinct so a deal bug that
    permutes matrices across subgroups or across SHAPES cannot pass by coincidence."""
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
    ap.add_argument("--shapes", nargs="+", required=True,
                    help="Full (all-gathered) GTP shapes to fuse, e.g. 10240x8192 8192x8192")
    ap.add_argument("--counts", nargs="+", type=int, required=True,
                    help="Owned count per shape; deliberately UNEQUAL so the fused packing "
                         "has to carry per-shape splits of different sizes.")
    ap.add_argument("--subgroup-sizes", nargs="+", type=int, required=True)
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

    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes]
    for (m, n) in shapes:
        assert m % world == 0, f"{m}x{n}: rows must divide the {world}-rank group"

    rows_out = []

    def record(scope, arm, a, b):
        d = (a.float() - b.float()).abs()
        denom = b.float().abs().clamp_min(1e-12)
        max_abs = d.max().item()
        row = {
            "axis": "gtp",
            "scope": scope,
            "arm": arm,
            "world": world,
            "max_abs_diff": f"{max_abs:.6e}",
            "mean_abs_diff": f"{d.mean().item():.6e}",
            "max_rel_diff": f"{(d / denom).max().item():.6e}",
            "bitwise_equal": bool(torch.equal(a, b)),
            "atol": args.atol,
            "rtol": args.rtol,
            "pass": "PASS" if max_abs <= args.atol else "FAIL",
        }
        rows_out.append(row)
        log(f"  {scope:>14} {arm:<52} max_abs={row['max_abs_diff']} "
            f"bitwise={row['bitwise_equal']} {row['pass']}")

    stacks = [
        build_stack(shape, count, world, rank, args.seed)
        for shape, count in zip(shapes, args.counts)
    ]
    log(f"world={world} shapes={args.shapes} counts={args.counts}")

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

    for g in args.subgroup_sizes:
        if world % g or g >= world:
            log(f"skipping g={g}: does not divide {world} or is not smaller than it")
            continue
        k = world // g
        log(f"g={g} (k={k}):")
        # A-arm reference: the per-shape subgroup path, one region and one pair of
        # exchanges per shape -- exactly what the candidate replaces.
        per_shape = [
            newton_schulz_tp_subgroup(
                stacks[i], steps=args.steps, coefficient_type=args.coefficient_type,
                tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
                subgroup_size=g,
            )
            for i in range(len(shapes))
        ]
        for i, (m, n) in enumerate(shapes):
            record(f"{m}x{n}", f"D control: per_shape_sub_g{g} vs duplicated lineage",
                   per_shape[i], lineage[i])

        for balanced in (False, True):
            tag = f"fuse_sub_g{g}" + ("_bal" if balanced else "")
            got = newton_schulz_tp_subgroup_fused(
                stacks, steps=args.steps, coefficient_type=args.coefficient_type,
                tp_group=group, use_syrk=True, subgroup_size=g, balanced=balanced,
            )
            assert len(got) == len(shapes)
            for i, (m, n) in enumerate(shapes):
                assert got[i].shape == stacks[i].shape, (
                    f"{tag}: shape {i} returned {tuple(got[i].shape)}, "
                    f"expected {tuple(stacks[i].shape)}"
                )
                record(f"{m}x{n}", f"A: {tag} vs the per-shape arm it substitutes",
                       got[i], per_shape[i])
                record(f"{m}x{n}", f"B: {tag} vs duplicated lineage", got[i], lineage[i])
                if references[i] is not None:
                    record(f"{m}x{n}", f"C: {tag} matrix 0 vs phase0-reference",
                           got[i][0], references[i])
            del got
            torch.cuda.empty_cache()

            # The deal itself: every matrix of every shape owned by exactly one subgroup,
            # and (balanced) the pooled load spread strictly better than round-robin.
            owners = fused_subgroup_deal(
                [(m, n) for (m, n) in shapes], list(args.counts), k, balanced
            )
            assert all(0 <= s < k for per in owners for s in per), f"{tag}: owner out of range"
            loads = [0] * k
            for i, (m, n) in enumerate(shapes):
                big, small = max(m, n), min(m, n)
                for j in range(args.counts[i]):
                    loads[owners[i][j]] += big * small * small
            timing["arms"].setdefault(tag, {})["deal_max_over_mean_load"] = (
                max(loads) / (sum(loads) / k) if sum(loads) else 0.0
            )

        # ---- timing: fused region vs the SUM of the per-shape regions it replaces -------
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

            def per_shape_sum():
                for i in range(len(shapes)):
                    newton_schulz_tp_subgroup(
                        stacks[i], steps=args.steps,
                        coefficient_type=args.coefficient_type, tp_group=group,
                        partition_dim=0, tp_mode="duplicated", use_syrk=True,
                        subgroup_size=g,
                    )

            timing["arms"].setdefault(f"per_shape_sub_g{g}", {})["ms"] = time_call(per_shape_sum)
            for balanced in (False, True):
                tag = f"fuse_sub_g{g}" + ("_bal" if balanced else "")
                timing["arms"].setdefault(tag, {})["ms"] = time_call(
                    lambda b=balanced: newton_schulz_tp_subgroup_fused(
                        stacks, steps=args.steps,
                        coefficient_type=args.coefficient_type, tp_group=group,
                        use_syrk=True, subgroup_size=g, balanced=b,
                    )
                )
            log(f"  timing g={g}: per-shape sum="
                f"{timing['arms'][f'per_shape_sub_g{g}']['ms']:.3f} ms, "
                f"fused={timing['arms'][f'fuse_sub_g{g}']['ms']:.3f} ms, "
                f"fused_bal={timing['arms'][f'fuse_sub_g{g}_bal']['ms']:.3f} ms")

        del per_shape
        torch.cuda.empty_cache()

    # ---- E) guards: every precondition must RAISE, not fall back -----------------------
    guards = [
        ("single stack (not a cross-shape fusion)", dict(stacks=stacks[:1], subgroup_size=world // 2)),
        ("subgroup_size == world (plain duplicated)", dict(stacks=stacks, subgroup_size=world)),
        ("subgroup_size does not divide world", dict(stacks=stacks, subgroup_size=world + 1)),
        ("subgroup_size <= 0", dict(stacks=stacks, subgroup_size=0)),
    ]
    for label, kwargs in guards:
        raised = ""
        try:
            newton_schulz_tp_subgroup_fused(
                kwargs["stacks"], steps=args.steps,
                coefficient_type=args.coefficient_type, tp_group=group, use_syrk=True,
                subgroup_size=kwargs["subgroup_size"],
            )
        except ValueError as error:
            raised = f"ValueError: {error}"
        rows_out.append({
            "axis": "gtp", "scope": "guard", "arm": f"E: {label}", "world": world,
            "max_abs_diff": "", "mean_abs_diff": "", "max_rel_diff": "",
            "bitwise_equal": "", "atol": args.atol, "rtol": args.rtol,
            "pass": "PASS" if raised else "FAIL",
        })
        log(f"  {'guard':>14} {('E: ' + label):<52} raised={raised or 'NOTHING'} "
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
            "all_pass": all(r["pass"] == "PASS" for r in rows_out),
            "a_arm_all_bitwise": all(
                r["bitwise_equal"] is True for r in rows_out if r["arm"].startswith("A: ")
            ),
            "max_abs_diff": max(
                (float(r["max_abs_diff"]) for r in rows_out if r["max_abs_diff"]), default=0.0
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
