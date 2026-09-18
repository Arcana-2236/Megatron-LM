# Equivalence gate for the `gtp-subgroup-dup-16384` candidate.
#
# spec.toml sets workload_mode = "inference", so the gate is EQUIVALENCE, not convergence.
# Two comparisons are made, on the same tensors, in the same job:
#
#   A) candidate vs mechanism baseline. For a stack of `--count` distinct matrices of one
#      shape, compare `newton_schulz_tp_subgroup(stack, subgroup_size=g)` against looping
#      `newton_schulz_tp(stack[j], tp_mode="duplicated", partition_dim=0)`. This is the
#      claim the candidate actually makes: the same matrices, orthogonalized whole by the
#      same call, on fewer ranks each, with the result slice MOVED rather than recomputed.
#
#   B) candidate vs the frozen Phase-0 reference. Matrix 0 of every stack is the
#      identically seeded input `capture_reference.py` used, so this rank's output slice is
#      compared against the corresponding rows of `reference_outputs/gtp_<M>x<N>.pt`. This
#      is what ties the candidate to the Phase-0 lineage rather than only to a sibling arm.
#
# Every eligible g is checked, and `--count` is chosen so that at least one g divides the
# count evenly and at least one does not -- the uneven `owned[]` path is where an index
# bug would hide.
#
# `--baseline-modes` (phase 3, `gtp-subgroup-dup-distributed`): the A-arm above compares
# against the mode the candidate REPLACES. For the 8192x16384 shape that mode is
# `duplicated`, and the comparison is bitwise. For the four shard_count=64 shapes this
# candidate targets (8192x10240, 36864x8192, 10240x8192, 8192x8192) the mode being
# replaced is `distributed`, and the comparison is NOT expected to be bitwise:
# `distributed` computes a sharded `A @ A` plus an all-reduce, which is a different
# reduction chain from the single local GEMM the duplicated/subgroup path runs. The claim
# there is equivalence at atol/rtol 1e-3 (spec.toml gate), and BOTH arms are additionally
# compared against the frozen Phase-0 reference so the candidate is tied to the lineage
# rather than only to a sibling. Passing `--baseline-modes duplicated distributed` records
# all of it in one job.

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

from bench_ns_strategies import newton_schulz_tp_subgroup  # noqa: E402
from emerging_optimizers.orthogonalized_optimizers.muon_utils import (  # noqa: E402
    newton_schulz_tp,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="equivalence.csv to write")
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--shapes", nargs="+", default=["8192x16384"],
                    help="Full (all-gathered) GTP shapes to check.")
    ap.add_argument("--subgroup-sizes", nargs="+", type=int, required=True)
    ap.add_argument("--baseline-modes", nargs="+", default=["duplicated"],
                    choices=["duplicated", "distributed"],
                    help="tp_mode(s) the candidate is compared against. The subgroup arm "
                         "is recorded against each of them, and each is itself recorded "
                         "against the Phase-0 reference.")
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
        log(f"  {shape:>12} {arm:<28} max_abs={row['max_abs_diff']} {row['pass']}")

    for shape in args.shapes:
        m, n = (int(v) for v in shape.split("x"))
        assert m % world == 0, f"{shape}: rows must divide the {world}-rank group"
        local_rows = m // world

        # Matrix 0 is the frozen Phase-0 input; the rest are distinct fill so a bug that
        # permutes matrices across subgroups cannot pass by coincidence.
        stack = torch.empty((args.count, local_rows, n), device="cuda", dtype=torch.float32)
        gen = torch.Generator(device="cuda")
        gen.manual_seed(args.seed + zlib.crc32(f"gtp_{m}x{n}".encode()) % 100000)
        full0 = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=gen)
        stack[0] = full0[rank * local_rows : (rank + 1) * local_rows]
        del full0
        for j in range(1, args.count):
            gen.manual_seed(args.seed + 7919 * j + zlib.crc32(f"gtp_{m}x{n}".encode()) % 100000)
            fj = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=gen)
            stack[j] = fj[rank * local_rows : (rank + 1) * local_rows]
            del fj
        torch.cuda.empty_cache()

        # A-arm references: the unmodified per-matrix path of every mode this candidate
        # may replace on this shape.
        baselines = {
            mode: torch.stack([
                newton_schulz_tp(
                    stack[j], steps=args.steps, coefficient_type=args.coefficient_type,
                    tp_group=group, partition_dim=0, tp_mode=mode, use_syrk=True,
                )
                for j in range(args.count)
            ])
            for mode in args.baseline_modes
        }

        ref_path = os.path.join(args.reference_dir, f"gtp_{m}x{n}.pt")
        reference = None
        if os.path.exists(ref_path):
            # Load on host and move only this rank's slice: the full reference is up to
            # 1.2 GB and a CUDA slice would keep all of it resident as a view.
            reference = torch.load(ref_path, map_location="cpu")[
                rank * local_rows : (rank + 1) * local_rows
            ].clone().cuda()
        else:
            log(f"  {shape}: no reference tensor at {ref_path}")
        if reference is not None:
            for mode in args.baseline_modes:
                record(shape, f"{mode} vs phase0-reference", baselines[mode][0], reference)

        log(f"{shape}: count={args.count}, local {local_rows}x{n}, world={world}")
        for g in args.subgroup_sizes:
            if world % g or g >= world:
                log(f"  skipping g={g}: does not divide {world} or is not smaller than it")
                continue
            got = newton_schulz_tp_subgroup(
                stack, steps=args.steps, coefficient_type=args.coefficient_type,
                tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
                subgroup_size=g,
            )
            k = world // g
            for mode in args.baseline_modes:
                record(shape, f"subgroup_g{g}(k={k}) vs {mode}", got, baselines[mode])
            if reference is not None:
                record(shape, f"subgroup_g{g}(k={k}) vs phase0-reference", got[0], reference)
            del got
            torch.cuda.empty_cache()
        del stack, baselines, reference
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
            "subgroup_sizes": args.subgroup_sizes,
            "baseline_modes": args.baseline_modes,
            "all_pass": all(r["pass"] == "PASS" for r in rows),
            "max_abs_diff": max(float(r["max_abs_diff"]) for r in rows),
            "rows": len(rows),
        }
        print("EQUIVALENCE_SUMMARY=" + json.dumps(summary), flush=True)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
