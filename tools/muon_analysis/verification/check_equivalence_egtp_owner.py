# Equivalence gate for the `egtp-owner-computes` candidate (stage 2, phase 4).
#
# spec.toml sets workload_mode = "inference", so the gate is EQUIVALENCE, not convergence.
#
# The candidate is `newton_schulz_tp_subgroup(..., subgroup_size=1, batched=True)` on the
# 2-rank EGTP axis: g == 1 makes the duplication group a single rank, so the owned set is
# dealt round-robin across the two ranks, each matrix is orthogonalized WHOLE by exactly
# one of them, and the result slice the other rank needs is MOVED by an all_to_all_single
# rather than recomputed. Redundancy 2.0 -> 1.0.
#
# Four comparisons, all on the SAME tensors in the SAME job:
#
#   A) candidate vs the per-matrix `duplicated` loop -- the phase-0/stage-1 mechanism and
#      the lineage every earlier number was taken under.
#   B) candidate vs `newton_schulz_tp_batched` -- the ACCEPTED state (phase 1) and the
#      reference arm the candidate's delta is scored against. B is the comparison that
#      matters for "did the exchange move the right bytes", because it is the arm the
#      candidate replaces.
#   C) candidate matrix 0 vs the frozen Phase-0 reference tensor, which ties the candidate
#      to the lineage rather than only to a sibling arm.
#   D) the UNBATCHED subgroup arm (`batched=False`) vs the same duplicated loop -- the
#      control that separates "owner-computes moved the wrong slice" from "the batched
#      dispatch changed the numerics".
#
# Counts are chosen to exercise every deal branch at world=2, k=2:
#   count=1 -> owned = [1, 0]: the ZERO-OWNER rank, which still takes part in both
#              all_to_all_single calls with all-zero output-side splits and an empty
#              compute list. This is the branch a naive implementation deadlocks on.
#   count=4 -> owned = [2, 2] (even deal)
#   count=5 -> owned = [3, 2] (uneven deal, unequal all_to_all splits rather than padding)
#
# Matrix 0 of every stack is the identically seeded Phase-0 input; matrices 1..n-1 are
# distinct random fill, so a bug that permutes matrices across ranks cannot pass by
# coincidence.

import argparse
import csv
import json
import os
import sys
import zlib

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "tools", "muon_analysis"))

from bench_ns_strategies import (  # noqa: E402
    newton_schulz_tp_batched,
    newton_schulz_tp_subgroup,
)
from emerging_optimizers.orthogonalized_optimizers.muon_utils import (  # noqa: E402
    newton_schulz_tp,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="equivalence.csv to write")
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--shapes", nargs="+", default=["5120x2048", "2048x5120"],
                    help="Full (all-gathered) EGTP shapes to check.")
    ap.add_argument("--counts", nargs="+", type=int, default=[1, 4, 5])
    ap.add_argument("--subgroup-size", type=int, default=1)
    ap.add_argument("--batch-chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--coefficient-type", default="polar_express")
    ap.add_argument("--fp32-matmul-prec", default="medium")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    torch.set_float32_matmul_precision(args.fp32_matmul_prec)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl")
    group = torch.distributed.distributed_c10d._get_default_group()
    world, rank = group.size(), group.rank()

    def log(message):
        if rank == 0:
            print(message, flush=True)

    rows = []

    def record(shape, count, arm, a, b):
        # max_abs_diff is the gate column (spec.toml gates.equivalence.atol). max_rel_diff
        # is reported but NOT gated: its denominator is an entry of an orthogonalized
        # matrix and goes to ~0, so the ratio is unbounded and uninformative there.
        d = (a.float() - b.float()).abs()
        denom = b.float().abs().clamp_min(1e-12)
        max_abs = d.max().item() if d.numel() else 0.0
        row = {
            "axis": "egtp",
            "shape": shape,
            "count": count,
            "arm": arm,
            "world": world,
            "subgroup_size": args.subgroup_size,
            "max_abs_diff": f"{max_abs:.6e}",
            "mean_abs_diff": f"{(d.mean().item() if d.numel() else 0.0):.6e}",
            "max_rel_diff": f"{((d / denom).max().item() if d.numel() else 0.0):.6e}",
            "atol": args.atol,
            "rtol": args.rtol,
            "pass": "PASS" if max_abs <= args.atol else "FAIL",
        }
        rows.append(row)
        log(f"  {shape:>11} n={count} {arm:<44} max_abs={row['max_abs_diff']} {row['pass']}")

    for shape in args.shapes:
        m, n = (int(v) for v in shape.split("x"))
        assert m % world == 0, f"{shape}: rows must divide the {world}-rank group"
        local_rows = m // world
        tag = f"egtp_{m}x{n}"

        ref_path = os.path.join(args.reference_dir, f"{tag}.pt")
        reference = None
        if os.path.exists(ref_path):
            reference = torch.load(ref_path, map_location="cpu")[
                rank * local_rows : (rank + 1) * local_rows
            ].clone().cuda()
        else:
            log(f"  {shape}: no reference tensor at {ref_path}")

        for count in args.counts:
            stack = torch.empty((count, local_rows, n), device="cuda", dtype=torch.float32)
            gen = torch.Generator(device="cuda")
            # Matrix 0: byte-identical to capture_reference.py's frozen input.
            gen.manual_seed(args.seed + zlib.crc32(tag.encode()) % 100000)
            full0 = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=gen)
            stack[0] = full0[rank * local_rows : (rank + 1) * local_rows]
            del full0
            for j in range(1, count):
                gen.manual_seed(args.seed + 7919 * j + zlib.crc32(tag.encode()) % 100000)
                fj = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=gen)
                stack[j] = fj[rank * local_rows : (rank + 1) * local_rows]
                del fj
            torch.cuda.empty_cache()

            log(f"{shape}: count={count}, local {local_rows}x{n}, world={world}, "
                f"g={args.subgroup_size}")

            duplicated = torch.stack([
                newton_schulz_tp(
                    stack[j], steps=args.steps, coefficient_type=args.coefficient_type,
                    tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
                )
                for j in range(count)
            ])
            batched = newton_schulz_tp_batched(
                stack, steps=args.steps, coefficient_type=args.coefficient_type,
                tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
            )
            owner_batched = newton_schulz_tp_subgroup(
                stack, steps=args.steps, coefficient_type=args.coefficient_type,
                tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
                subgroup_size=args.subgroup_size, batched=True,
                batch_chunk=args.batch_chunk,
            )
            owner_unbatched = newton_schulz_tp_subgroup(
                stack, steps=args.steps, coefficient_type=args.coefficient_type,
                tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
                subgroup_size=args.subgroup_size, batched=False,
            )

            record(shape, count, "A owner_batched vs duplicated-loop", owner_batched, duplicated)
            record(shape, count, "B owner_batched vs batched (accepted)", owner_batched, batched)
            record(shape, count, "D owner_unbatched vs duplicated-loop", owner_unbatched,
                   duplicated)
            record(shape, count, "  batched vs duplicated-loop (context)", batched, duplicated)
            if reference is not None:
                record(shape, count, "C owner_batched[0] vs phase0-reference",
                       owner_batched[0], reference)
                record(shape, count, "  duplicated[0] vs phase0-reference (context)",
                       duplicated[0], reference)

            del stack, duplicated, batched, owner_batched, owner_unbatched
            torch.cuda.empty_cache()
        del reference
        torch.cuda.empty_cache()

    if rank == 0:
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        gated = [r for r in rows if r["arm"].startswith(("A", "B", "C", "D"))]
        summary = {
            "gate": "equivalence",
            "atol": args.atol,
            "rtol": args.rtol,
            "world": world,
            "counts": args.counts,
            "subgroup_size": args.subgroup_size,
            "shapes": args.shapes,
            "all_pass": all(r["pass"] == "PASS" for r in rows),
            "gated_rows": len(gated),
            "rows": len(rows),
            "max_abs_diff": max(float(r["max_abs_diff"]) for r in rows),
            "max_abs_diff_gated": max(float(r["max_abs_diff"]) for r in gated),
        }
        print("EQUIVALENCE_SUMMARY=" + json.dumps(summary), flush=True)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
