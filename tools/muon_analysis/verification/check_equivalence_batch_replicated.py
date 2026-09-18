# Equivalence gate for the `gtp-batch-set-in-argmin` candidate (stage-3 rank 1).
#
# spec.toml sets workload_mode = "inference", so the gate is EQUIVALENCE, not convergence.
#
# The candidate lets the per-shape argmin bank the already-timed `duplicated_batch` column
# on the shard_count == 1 (replicated) GTP shapes. What it therefore has to prove is one
# thing: on a replicated shape, orthogonalizing the owned group as ONE batched call is the
# same computation as looping the per-matrix call over that group.
#
# Three comparisons, same tensors, same job:
#
#   A) candidate vs the mechanism baseline it replaces -- `newton_schulz_tp_batched(stack,
#      partition_dim=None)` chunked exactly as `time_group` chunks it, against looping
#      `newton_schulz_tp(stack[j], partition_dim=None, tp_mode="duplicated")`. Expected to
#      agree tightly: at partition_dim is None the batched entry point short-circuits to
#      plain `newton_schulz` on the stack, so the only difference is baddbmm/batched-SYRK
#      over a leading batch dim instead of addmm/SYRK per matrix -- no reassociation of any
#      reduction, same coefficients, same step count, same SYRK path.
#   B) candidate vs the frozen Phase-0 reference, on matrix 0 (the identically seeded input
#      capture_reference.py froze), so the candidate is tied to the Phase-0 lineage and not
#      only to a sibling arm.
#   C) the baseline itself vs the Phase-0 reference, so a reference/lineage drift cannot be
#      misread as a candidate regression.
#
# The stack holds `--count` DISTINCT matrices, so a bug that broadcast one matrix over the
# batch -- the exact hazard `distributed_normalize_p2` would introduce if the batched call
# ever reached the collective route -- cannot pass by coincidence: it would corrupt every
# slice but one.
#
# Replicated means no collective, so this gate is rank-local and every rank checks the same
# thing independently; the recorded row is rank 0's.

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

from bench_ns_strategies import newton_schulz_tp_batched  # noqa: E402
from emerging_optimizers.orthogonalized_optimizers.muon_utils import (  # noqa: E402
    newton_schulz_tp,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="equivalence.csv to write")
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--shapes", nargs="+", required=True,
                    help="Replicated (shard_count == 1) GTP shapes, e.g. 8192x2048.")
    ap.add_argument("--batch-chunks", nargs="+", type=int, default=[4],
                    help="batch_chunk values to check; must include the one the scored "
                         "sbatch passes, since that is the arm being banked.")
    ap.add_argument("--count", type=int, default=6, help="matrices per stack")
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

    def record(shape, arm, a, b):
        d = (a.float() - b.float()).abs()
        denom = b.float().abs().clamp_min(1e-12)
        max_abs = d.max().item()
        row = {
            "axis": "gtp",
            "shape": shape,
            "arm": arm,
            "world": world,
            "count": args.count,
            "max_abs_diff": f"{max_abs:.6e}",
            "mean_abs_diff": f"{d.mean().item():.6e}",
            "max_rel_diff": f"{(d / denom).max().item():.6e}",
            "atol": args.atol,
            "rtol": args.rtol,
            "pass": "PASS" if max_abs <= args.atol else "FAIL",
        }
        rows.append(row)
        log(f"  {shape:>12} {arm:<40} max_abs={row['max_abs_diff']} {row['pass']}")

    for shape in args.shapes:
        m, n = (int(v) for v in shape.split("x"))
        # shard_count == 1: every rank holds the WHOLE matrix, so local == full and
        # partition_dim is None. That is the invariant the candidate's gate rests on.
        stack = torch.empty((args.count, m, n), device="cuda", dtype=torch.float32)
        gen = torch.Generator(device="cuda")
        # Matrix 0 is the frozen Phase-0 input, seeded exactly as capture_reference.py did;
        # the rest are distinct fill.
        gen.manual_seed(args.seed + zlib.crc32(f"gtp_{m}x{n}".encode()) % 100000)
        stack[0] = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=gen)
        for j in range(1, args.count):
            gen.manual_seed(args.seed + 7919 * j + zlib.crc32(f"gtp_{m}x{n}".encode()) % 100000)
            stack[j] = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=gen)
        torch.cuda.empty_cache()

        log(f"{shape}: count={args.count}, replicated {m}x{n} (partition_dim=None), world={world}")

        baseline = torch.stack([
            newton_schulz_tp(
                stack[j], steps=args.steps, coefficient_type=args.coefficient_type,
                tp_group=group, partition_dim=None, tp_mode="duplicated", use_syrk=True,
            )
            for j in range(args.count)
        ])

        ref_path = os.path.join(args.reference_dir, f"gtp_{m}x{n}.pt")
        reference = None
        if os.path.exists(ref_path):
            reference = torch.load(ref_path, map_location="cpu").clone().cuda()
        else:
            log(f"  {shape}: no reference tensor at {ref_path}")
        if reference is not None:
            record(shape, "per-matrix duplicated vs phase0-reference", baseline[0], reference)

        for chunk in args.batch_chunks:
            chunk = chunk if chunk > 0 else args.count
            got = torch.cat([
                newton_schulz_tp_batched(
                    stack[s : s + chunk], steps=args.steps,
                    coefficient_type=args.coefficient_type, tp_group=group,
                    partition_dim=None, tp_mode="duplicated", use_syrk=True,
                )
                for s in range(0, args.count, chunk)
            ])
            record(shape, f"duplicated_batch[chunk={chunk}] vs per-matrix", got, baseline)
            if reference is not None:
                record(
                    shape, f"duplicated_batch[chunk={chunk}] vs phase0-reference",
                    got[0], reference,
                )
            del got
            torch.cuda.empty_cache()
        del stack, baseline, reference
        torch.cuda.empty_cache()

    if rank == 0:
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "gate": "equivalence",
            "atol": args.atol,
            "rtol": args.rtol,
            "world": world,
            "count": args.count,
            "shapes": args.shapes,
            "batch_chunks": args.batch_chunks,
            "all_pass": all(r["pass"] == "PASS" for r in rows),
            "max_abs_diff": max(float(r["max_abs_diff"]) for r in rows),
            "rows": len(rows),
        }
        print("EQUIVALENCE_SUMMARY=" + json.dumps(summary), flush=True)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
