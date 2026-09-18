# Phase-0 reference capture for the dist_muon Newton-Schulz benchmark.
#
# The benchmark driver (bench_ns_strategies.py) is a pure latency benchmark: it seeds
# nothing and emits no numerical output. This script freezes the input batch (fixed seed)
# and captures the Newton-Schulz OUTPUT for every distinct matrix shape the modelled
# workload orthogonalizes, so later attempts have something to run the equivalence gate
# against.
#
# The full (all-gathered) matrix is used with partition_dim=None: that is exactly the
# tensor `duplicated` mode orthogonalizes on every rank after its all-gather, so the
# single-GPU result is a faithful reference for the mode both axes currently pick.
#
# Runs the whole capture TWICE with the identical seed so the run-to-run output
# nondeterminism floor can be computed from the same job.

import argparse
import json
import os
import sys
import zlib

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "tools", "muon_analysis"))

from bench_ns_strategies import build_model_matrices, ns_cost  # noqa: E402
from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp  # noqa: E402


class Cfg:
    tp, gtp, ep, etp, egtp = 1, 64, 64, 1, 2
    modelled_world_size = 128
    shard_latent_proj = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--coefficient-type", default="polar_express")
    ap.add_argument("--fp32-matmul-prec", default="medium")
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()

    torch.set_float32_matmul_precision(args.fp32_matmul_prec)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.distributed.init_process_group(backend="nccl")
    group = torch.distributed.distributed_c10d._get_default_group()

    dense, expert = build_model_matrices(Cfg)
    shapes = []
    for axis, mats in (("gtp", dense), ("egtp", expert)):
        for (rows, cols), shard in sorted(set(mats), key=lambda e: -ns_cost(e)):
            shapes.append((axis, rows * shard, cols))
    seen, uniq = set(), []
    for s in shapes:
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for p in range(args.passes):
        subdir = os.path.join(args.out, f"pass{p}")
        os.makedirs(subdir, exist_ok=True)
        for axis, m, n in uniq:
            # Frozen input: per-shape generator seeded identically in every pass.
            gen = torch.Generator(device="cuda")
            # zlib.crc32, not hash(): PYTHONHASHSEED randomizes hash() per process, which
            # would make the "frozen" batch differ between jobs.
            gen.manual_seed(args.seed + zlib.crc32(f"{axis}_{m}x{n}".encode()) % 100000)
            x = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=gen)
            y = newton_schulz_tp(
                x,
                steps=args.steps,
                coefficient_type=args.coefficient_type,
                tp_group=group,
                partition_dim=None,
                tp_mode="duplicated",
                use_syrk=True,
            )
            name = f"{axis}_{m}x{n}.pt"
            torch.save(y.detach().cpu(), os.path.join(subdir, name))
            if p == 0:
                manifest.append(
                    {"axis": axis, "shape": [m, n], "dtype": "float32", "file": name}
                )
            del x, y
            torch.cuda.empty_cache()
        print(f"[capture] pass {p} done ({len(uniq)} shapes)", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(
            {
                "seed": args.seed,
                "steps": args.steps,
                "coefficient_type": args.coefficient_type,
                "fp32_matmul_precision": args.fp32_matmul_prec,
                "use_syrk": True,
                "tp_mode": "duplicated",
                "partition_dim": None,
                "tensors": manifest,
            },
            fh,
            indent=2,
        )

    # Output nondeterminism floor from the two identical-seed passes.
    if args.passes >= 2:
        rows = []
        for t in manifest:
            a = torch.load(os.path.join(args.out, "pass0", t["file"]))
            b = torch.load(os.path.join(args.out, "pass1", t["file"]))
            d = (a - b).abs()
            denom = a.abs().clamp_min(1e-12)
            rows.append(
                {
                    "file": t["file"],
                    "max_abs_diff": d.max().item(),
                    "mean_abs_diff": d.mean().item(),
                    "max_rel_diff": (d / denom).max().item(),
                    "mean_rel_diff": (d / denom).mean().item(),
                }
            )
            print(f"[floor] {t['file']}: {rows[-1]}", flush=True)
        with open(os.path.join(args.out, "output_floor.json"), "w") as fh:
            json.dump(
                {
                    "per_tensor": rows,
                    "max_abs_diff": max(r["max_abs_diff"] for r in rows),
                    "max_rel_diff": max(r["max_rel_diff"] for r in rows),
                    "mean_abs_diff": sum(r["mean_abs_diff"] for r in rows) / len(rows),
                    "mean_rel_diff": sum(r["mean_rel_diff"] for r in rows) / len(rows),
                },
                fh,
                indent=2,
            )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
