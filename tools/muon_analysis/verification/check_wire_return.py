# Functional/timing window for the `egtp-wire-bf16-return` candidate (stage 4, phase 4).
#
# spec.toml sets workload_mode = "inference", so the gate is EQUIVALENCE, not convergence.
#
# The candidate sends the OUTPUT leg of the pipelined g == 1 subgroup exchange as bf16 and
# widens it into the fp32 `result` inside the timed region. Its claim is BITWISE, not
# tolerance: `fused_newton_schulz_batched` holds X in bf16 through all five polar_express
# steps and widens only at its `_cast_into` epilogue, so every value on that wire is
# exactly bf16-representable. stage-4/phase-3's pre-flight already measured that
# (`output_bf16_exact_frac = 1.000000`, all seven hot shapes, both axes); this script
# tests the claim THROUGH THE REAL COLLECTIVE, which is the thing the pre-flight could not
# do on one GPU.
#
# Arms, on the ACCEPTED state (`newton_schulz_tp_subgroup(..., subgroup_size=1,
# batched=True, fused=True, pipelined=True, elide_self=True)`, banked in stage-4 phase 1
# as `per_shape_batch_sub_elide_set`):
#
#   A) wired vs the ACCEPTED elided arm it substitutes -- must be BIT-IDENTICAL. This is
#      the load-bearing row: the whole candidate is "a bf16 round trip of a value that is
#      already bf16-exact is the identity".
#   A0) the same comparison at steps = 0. At zero Newton-Schulz steps the fused kernel
#      collapses to `normalize -> bf16 -> fp32`, so this row isolates the DATA MOVEMENT
#      from the five-step chain. If A were bitwise only by luck of the chain, A0 would
#      still have to be bitwise for the transport change to be sound.
#   B) wired vs the per-matrix `duplicated` loop -- the lineage mechanism every earlier
#      number on this axis was taken under.
#   C) wired matrix 0 vs the frozen Phase-0 reference tensor.
#   D) accepted elided arm vs the same duplicated loop -- the control that separates "the
#      wire moved the wrong bytes" from "this arm was already off".
#   E) the un-elided pipelined arm, wired vs unwired -- the wire must be bitwise on BOTH
#      self-block treatments, since the scored job times both columns.
#   F) exactness census -- the fraction of the region's OWN output that is exactly
#      bf16-representable, measured on the real arm's return value rather than assumed.
#   G) guards -- a wire mode without `pipelined=True`, on the unfused compute path, or an
#      unknown mode name must be a HARD ERROR, not a silent fp32 fallback, because a
#      silent fallback would let the arm be scored for a mechanism it did not run.
#
# The timing half runs both arms at the REAL region count (192 matrices, 96 owned per
# rank) so the phase has a cheap read on direction and size before a scored job is spent.
# It is a window, not the scored number.

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


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="equivalence.csv to write")
    ap.add_argument("--timing-out", default=None)
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--shapes", nargs="+", default=["5120x2048", "2048x5120"])
    ap.add_argument("--counts", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--timing-count", type=int, default=192)
    ap.add_argument("--batch-chunk", type=int, default=64)
    ap.add_argument("--pipe-chunks", nargs="+", type=int, default=[4])
    ap.add_argument("--wire", default="bf16out")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--coefficient-type", default="polar_express")
    ap.add_argument("--fp32-matmul-prec", default="medium")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    # kernels.fused_ns._supported makes this a PRECONDITION of the fused path. stage-4
    # phase-3 lost a probe to omitting it (job 2893445 silently measured the library
    # fallback and read a false green), so it is set here and asserted below.
    torch.set_float32_matmul_precision(args.fp32_matmul_prec)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl")
    group = torch.distributed.distributed_c10d._get_default_group()
    world, rank = group.size(), group.rank()

    def log(message):
        if rank == 0:
            print(message, flush=True)

    # The arm must run the FUSED path -- the bitwise property comes from the fused
    # kernel's bf16 X. A silent fallback here would make every row below meaningless.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kernels import fused_ns  # noqa: E402

    probe = torch.zeros((1, 8, 8), device="cuda", dtype=torch.float32)
    assert fused_ns._supported(probe, args.coefficient_type, True), (
        "kernels.fused_ns._supported is False: the fused compute path would silently fall "
        "back to the library newton_schulz and this window would measure the wrong code"
    )
    del probe

    def arm(stack, wire="", elide=True, pipelined=True, pipe_chunks=4, subgroup_size=1,
            batched=True, fused=True, steps=None):
        return newton_schulz_tp_subgroup(
            stack, steps=args.steps if steps is None else steps,
            coefficient_type=args.coefficient_type,
            tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
            subgroup_size=subgroup_size, batched=batched, batch_chunk=args.batch_chunk,
            fused=fused, pipelined=pipelined, pipe_chunks=pipe_chunks,
            elide_self=elide, wire=wire,
        )

    rows = []

    def record(shape, count, name, a, b, note=""):
        d = (a.float() - b.float()).abs()
        max_abs = d.max().item() if d.numel() else 0.0
        row = {
            "axis": "egtp", "shape": shape, "count": count, "arm": name, "world": world,
            "wire": args.wire,
            "max_abs_diff": f"{max_abs:.6e}",
            "mean_abs_diff": f"{(d.mean().item() if d.numel() else 0.0):.6e}",
            "bitwise_equal": bool(torch.equal(a, b)),
            "atol": args.atol, "rtol": args.rtol,
            "pass": "PASS" if max_abs <= args.atol else "FAIL",
            "note": note,
        }
        rows.append(row)
        log(f"  {shape:>11} n={count} {name:<52} max_abs={row['max_abs_diff']} "
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
                f"wire={args.wire}")
            duplicated = torch.stack([
                newton_schulz_tp(
                    stack[j], steps=args.steps, coefficient_type=args.coefficient_type,
                    tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
                )
                for j in range(count)
            ])
            for pc in args.pipe_chunks:
                accepted = arm(stack, wire="", elide=True, pipe_chunks=pc)
                wired = arm(stack, wire=args.wire, elide=True, pipe_chunks=pc)
                record(shape, count, f"A wired(pc={pc}) vs accepted elided arm",
                       wired, accepted, note="substituted arm; must be bitwise")
                record(shape, count, f"B wired(pc={pc}) vs duplicated-loop", wired,
                       duplicated, note="lineage mechanism")
                if reference is not None:
                    record(shape, count, f"C wired(pc={pc})[0] vs phase0-reference",
                           wired[0], reference, note="phase-0 lineage")
                record(shape, count, f"D accepted(pc={pc}) vs duplicated-loop", accepted,
                       duplicated, note="control")

                # A0: data movement isolated from the five-step chain.
                zero_ref = arm(stack, wire="", elide=True, pipe_chunks=pc, steps=0)
                zero_wired = arm(stack, wire=args.wire, elide=True, pipe_chunks=pc, steps=0)
                record(shape, count, f"A0 wired(pc={pc},steps=0) vs accepted(steps=0)",
                       zero_wired, zero_ref,
                       note="data movement only; must be bitwise")

                # E: the same wire on the UN-elided pipelined column, which the scored
                # job also times.
                un_ref = arm(stack, wire="", elide=False, pipe_chunks=pc)
                un_wired = arm(stack, wire=args.wire, elide=False, pipe_chunks=pc)
                record(shape, count, f"E wired(pc={pc},no-elide) vs pipelined(no-elide)",
                       un_wired, un_ref, note="wire on the un-elided column; bitwise")

                # F: the exactness census the bitwise claim rests on, measured on THIS
                # region's own output rather than inherited from the pre-flight.
                exact = torch.equal(
                    accepted, accepted.to(torch.bfloat16).to(torch.float32))
                frac = (
                    accepted == accepted.to(torch.bfloat16).to(torch.float32)
                ).float().mean().item()
                rows.append({
                    "axis": "egtp", "shape": shape, "count": count,
                    "arm": f"F bf16-exactness census(pc={pc})", "world": world,
                    "wire": args.wire, "max_abs_diff": "0.000000e+00",
                    "mean_abs_diff": "0.000000e+00", "bitwise_equal": bool(exact),
                    "atol": args.atol, "rtol": args.rtol,
                    "pass": "PASS" if exact else "FAIL",
                    "note": f"output_bf16_exact_frac={frac:.6f}",
                })
                log(f"  {shape:>11} n={count} F bf16-exact frac={frac:.6f} "
                    f"all_exact={exact}")
                del wired, accepted, zero_ref, zero_wired, un_ref, un_wired
            del stack, duplicated
            torch.cuda.empty_cache()
        del reference
        torch.cuda.empty_cache()

    # ---- the preconditions must be HARD errors, not silent fp32 fallbacks --------------
    guard_shape = args.shapes[0]
    m, n = (int(v) for v in guard_shape.split("x"))
    guard = build_stack(f"egtp_{m}x{n}", m, n, 2 * world, world, rank, args.seed)
    guards = []
    for name, kwargs in (
        ("wire without pipelined", dict(wire=args.wire, pipelined=False)),
        ("wire on the unfused path", dict(wire=args.wire, fused=False)),
        ("unknown wire mode", dict(wire="fp8out")),
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
            "axis": "egtp", "shape": guard_shape, "count": 2 * world,
            "arm": f"G guard: {name}", "world": world, "wire": args.wire,
            "max_abs_diff": "0.000000e+00", "mean_abs_diff": "0.000000e+00",
            "bitwise_equal": True, "atol": args.atol, "rtol": args.rtol,
            "pass": "PASS" if outcome != "NO-RAISE" else "FAIL", "note": outcome,
        })

    # ---- timing window at the real region count ---------------------------------------
    timing = {}
    if args.timing_count:
        for shape in args.shapes:
            m, n = (int(v) for v in shape.split("x"))
            stack = build_stack(f"egtp_{m}x{n}", m, n, args.timing_count, world, rank,
                                args.seed)
            entry = {}
            for pc in args.pipe_chunks:
                entry[f"accepted_pc{pc}"] = bench(
                    lambda pc=pc: arm(stack, wire="", elide=True, pipe_chunks=pc),
                    args.warmup, args.iters)
                entry[f"wired_pc{pc}"] = bench(
                    lambda pc=pc: arm(stack, wire=args.wire, elide=True, pipe_chunks=pc),
                    args.warmup, args.iters)
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, entry)
            mx = {k: max(g[k] for g in gathered) for k in entry}
            timing[shape] = {
                "per_rank": gathered,
                "max": mx,
                "delta_ms": {f"pc{pc}": mx[f"wired_pc{pc}"] - mx[f"accepted_pc{pc}"]
                             for pc in args.pipe_chunks},
            }
            log(f"TIMING {shape} (max over ranks, count={args.timing_count}): "
                + ", ".join(f"{k}={v:.3f}ms" for k, v in mx.items())
                + " | delta " + ", ".join(f"{k}={v:+.3f}ms"
                                          for k, v in timing[shape]["delta_ms"].items()))
            del stack
            torch.cuda.empty_cache()

    if rank == 0:
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        bitwise_arms = ("A ", "A0", "E ")
        summary = {
            "gate": "equivalence", "atol": args.atol, "rtol": args.rtol, "world": world,
            "counts": args.counts, "pipe_chunks": args.pipe_chunks, "wire": args.wire,
            "shapes": args.shapes,
            "all_pass": all(r["pass"] == "PASS" for r in rows),
            "all_bitwise": all(
                r["bitwise_equal"] for r in rows
                if r["arm"].startswith(bitwise_arms)
            ),
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
