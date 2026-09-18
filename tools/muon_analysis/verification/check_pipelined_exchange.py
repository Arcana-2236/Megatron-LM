# Functional + timing window for the `egtp-exchange-restructure` candidate (stage 3,
# phase 3).
#
# spec.toml sets workload_mode = "inference", so the gate is EQUIVALENCE, not convergence.
#
# The candidate is `newton_schulz_tp_subgroup(..., subgroup_size=1, batched=True,
# pipelined=True)`: the same owner-computes deal on the 2-rank EGTP axis, with the two
# subgroup exchanges restructured. Block ownership (matrix j -> subgroup j // mine, instead
# of j % k) makes the send side of the input exchange and the receive side of the output
# exchange exact VIEWS of the caller's `stack` and of `result`, so two of the four
# full-size staging copies stop existing; the per-chunk exchanges are then issued with
# async_op=True so the wire overlaps a different chunk's compute.
#
# Ownership is INTERNAL: the return value is still this rank's slice of every matrix in
# input order, so the change must be bit-for-bit invisible. Four comparisons, on the SAME
# tensors in the SAME job:
#
#   A) pipelined vs the ACCEPTED arm it substitutes -- monolithic subgroup, batched, with
#      the same fused/unfused compute path. This is the arm the delta is scored against and
#      the one the substitution assert in bench_ns_strategies.py enforces.
#   B) pipelined vs the per-matrix `duplicated` loop -- the phase-0/stage-1 mechanism and
#      the lineage every earlier number was taken under.
#   C) pipelined matrix 0 vs the frozen Phase-0 reference tensor.
#   D) the ACCEPTED arm vs the same duplicated loop -- the control that separates "the
#      restructured exchange moved the wrong bytes" from "this arm was already off".
#
# Counts are even multiples of the world size, which is the only case the block deal
# admits; `--expect-reject` additionally asserts that an indivisible count and a g > 1 deal
# are HARD ERRORS rather than a silent fallback to the monolithic path, because a silent
# fallback would let the arm be scored while running the code it claims to replace.
#
# The timing half runs both arms at the REAL region count (192 matrices, 96 owned per rank)
# so the phase has a cheap read on the direction and rough size of the delta before the
# scored job is spent. It is a window, not the scored number: the scored number comes from
# bench_ns_strategies.py under --set-timing in the same job as its reference arm.

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


def build_stack(tag, m, n, count, world, rank, seed):
    """Matrix 0 is capture_reference.py's frozen input; 1..n-1 are distinct random fill."""
    local_rows = m // world
    stack = torch.empty((count, local_rows, n), device="cuda", dtype=torch.float32)
    gen = torch.Generator(device="cuda")
    base = seed + zlib.crc32(tag.encode()) % 100000
    for j in range(count):
        gen.manual_seed(base if j == 0 else seed + 7919 * j + zlib.crc32(tag.encode()) % 100000)
        full = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=gen)
        stack[j] = full[rank * local_rows : (rank + 1) * local_rows]
        del full
    torch.cuda.empty_cache()
    return stack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="equivalence.csv to write")
    ap.add_argument("--timing-out", default=None, help="optional timing json to write")
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--shapes", nargs="+", default=["5120x2048", "2048x5120"])
    ap.add_argument("--counts", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--timing-count", type=int, default=192)
    ap.add_argument("--batch-chunk", type=int, default=64)
    ap.add_argument("--pipe-chunks", nargs="+", type=int, default=[4])
    ap.add_argument("--fused", action="store_true",
                    help="Run both arms on the fused Newton-Schulz compute path, which is "
                         "the accepted state on this axis. The pipelined arm must never "
                         "switch this relative to its reference.")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
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

    def arm(stack, pipelined, pipe_chunks=4, subgroup_size=1, batched=True):
        return newton_schulz_tp_subgroup(
            stack, steps=args.steps, coefficient_type=args.coefficient_type,
            tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
            subgroup_size=subgroup_size, batched=batched, batch_chunk=args.batch_chunk,
            fused=args.fused, pipelined=pipelined, pipe_chunks=pipe_chunks,
        )

    rows = []

    def record(shape, count, name, a, b, note=""):
        d = (a.float() - b.float()).abs()
        max_abs = d.max().item() if d.numel() else 0.0
        row = {
            "axis": "egtp",
            "shape": shape,
            "count": count,
            "arm": name,
            "world": world,
            "fused": int(args.fused),
            "max_abs_diff": f"{max_abs:.6e}",
            "mean_abs_diff": f"{(d.mean().item() if d.numel() else 0.0):.6e}",
            "bitwise_equal": bool(torch.equal(a, b)),
            "atol": args.atol,
            "rtol": args.rtol,
            "pass": "PASS" if max_abs <= args.atol else "FAIL",
            "note": note,
        }
        rows.append(row)
        log(f"  {shape:>11} n={count} {name:<46} max_abs={row['max_abs_diff']} "
            f"bitwise={row['bitwise_equal']} {row['pass']}")

    # ---- equivalence -------------------------------------------------------------------
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
            stack = build_stack(tag, m, n, count, world, rank, args.seed)
            log(f"{shape}: count={count}, local {local_rows}x{n}, world={world}, "
                f"fused={args.fused}")
            duplicated = torch.stack([
                newton_schulz_tp(
                    stack[j], steps=args.steps, coefficient_type=args.coefficient_type,
                    tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
                )
                for j in range(count)
            ])
            accepted = arm(stack, pipelined=False)
            for pc in args.pipe_chunks:
                piped = arm(stack, pipelined=True, pipe_chunks=pc)
                record(shape, count, f"A pipelined(pc={pc}) vs accepted arm", piped,
                       accepted, note="substituted arm")
                record(shape, count, f"B pipelined(pc={pc}) vs duplicated-loop", piped,
                       duplicated, note="lineage mechanism")
                if reference is not None:
                    record(shape, count, f"C pipelined(pc={pc})[0] vs phase0-reference",
                           piped[0], reference, note="phase-0 lineage")
                del piped
            record(shape, count, "D accepted arm vs duplicated-loop", accepted, duplicated,
                   note="control")
            del stack, duplicated, accepted
            torch.cuda.empty_cache()
        del reference
        torch.cuda.empty_cache()

    # ---- the preconditions must be HARD errors, not silent fallbacks -------------------
    guard_shape = args.shapes[0]
    m, n = (int(v) for v in guard_shape.split("x"))
    guard = build_stack(f"egtp_{m}x{n}", m, n, world + 1, world, rank, args.seed)
    guards = []
    for name, kwargs in (
        ("indivisible count", dict(pipelined=True)),
        ("subgroup_size > 1", dict(pipelined=True, subgroup_size=world)),
        ("unbatched", dict(pipelined=True, batched=False)),
    ):
        try:
            arm(guard, **kwargs)
            guards.append((name, "NO-RAISE"))
        except (ValueError, RuntimeError) as exc:
            guards.append((name, f"raised {type(exc).__name__}"))
    del guard
    torch.cuda.empty_cache()
    log("precondition guards: " + ", ".join(f"{k}={v}" for k, v in guards))
    for name, outcome in guards:
        rows.append({
            "axis": "egtp", "shape": guard_shape, "count": world + 1,
            "arm": f"E guard: {name}", "world": world, "fused": int(args.fused),
            "max_abs_diff": "0.000000e+00", "mean_abs_diff": "0.000000e+00",
            "bitwise_equal": True, "atol": args.atol, "rtol": args.rtol,
            "pass": "PASS" if outcome != "NO-RAISE" else "FAIL",
            "note": outcome,
        })

    # ---- timing window at the real region count ---------------------------------------
    timing = {}
    if args.timing_count:
        for shape in args.shapes:
            m, n = (int(v) for v in shape.split("x"))
            stack = build_stack(f"egtp_{m}x{n}", m, n, args.timing_count, world, rank,
                                args.seed)

            def bench(fn):
                for _ in range(args.warmup):
                    fn()
                torch.cuda.synchronize()
                torch.distributed.barrier()
                samples = []
                for _ in range(args.iters):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    fn()
                    end.record()
                    torch.cuda.synchronize()
                    samples.append(start.elapsed_time(end))
                samples.sort()
                return samples[len(samples) // 2]

            ms_ref = bench(lambda: arm(stack, pipelined=False))
            entry = {"accepted": ms_ref}
            for pc in args.pipe_chunks:
                entry[f"pipelined_pc{pc}"] = bench(
                    lambda pc=pc: arm(stack, pipelined=True, pipe_chunks=pc)
                )
            # The step is the max over ranks, so report this rank's numbers and the max.
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, entry)
            timing[shape] = {
                "per_rank": gathered,
                "max": {k: max(g[k] for g in gathered) for k in entry},
            }
            log(f"TIMING {shape} (max over ranks, count={args.timing_count}): "
                + ", ".join(f"{k}={v:.3f}ms" for k, v in timing[shape]["max"].items()))
            del stack
            torch.cuda.empty_cache()

    if rank == 0:
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        gated = [r for r in rows if r["arm"].startswith(("A", "B", "C", "D", "E"))]
        summary = {
            "gate": "equivalence",
            "atol": args.atol,
            "rtol": args.rtol,
            "world": world,
            "counts": args.counts,
            "pipe_chunks": args.pipe_chunks,
            "fused": args.fused,
            "shapes": args.shapes,
            "all_pass": all(r["pass"] == "PASS" for r in rows),
            "all_bitwise": all(r["bitwise_equal"] for r in rows if r["arm"].startswith("A")),
            "gated_rows": len(gated),
            "rows": len(rows),
            "max_abs_diff": max(float(r["max_abs_diff"]) for r in rows),
        }
        print("EQUIVALENCE_SUMMARY=" + json.dumps(summary), flush=True)
        if timing:
            print("TIMING_SUMMARY=" + json.dumps(timing), flush=True)
            if args.timing_out:
                with open(args.timing_out, "w") as fh:
                    json.dump(timing, fh, indent=2)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
