# Functional/timing window for `egtp-wire-24bit-input-leg` (stage 5, phase 3).
#
# spec.toml sets workload_mode = "inference", so the gate is EQUIVALENCE, not convergence.
#
# The candidate keeps the accepted bf16 OUTPUT leg and additionally carries the INPUT leg
# of the pipelined g == 1 subgroup exchange as a 24-bit split (bf16 high half + the next 8
# mantissa bits, `kernels/wire24.py`), reassembled into the fp32 gather buffer inside the
# timed region. 3 bytes per element instead of 4 on the only leg still at full width.
#
# Unlike its stage-4 predecessor this claim is NOT bitwise: the codec clears the low 8 of
# fp32's 24 mantissa bits after a round-to-nearest-even, so every entry entering the
# Newton-Schulz prologue is perturbed by <= 2^-16 relative. That is exactly the mechanism
# plain bf16 (8 mantissa bits) was refuted on -- max_abs 1.221e-3 against the 1e-3 gate,
# stage-4 phase-5 -- so the load-bearing question here is a MEASURED one and cannot be
# answered by per-entry ULP arithmetic: decisions.md:567 records a 1-ULP perturbation of
# 260 of 83.9M entries amplifying to 1.95e-3 by step 3.
#
# Rows, on the ACCEPTED state (`newton_schulz_tp_subgroup(..., subgroup_size=1,
# batched=True, fused=True, pipelined=True, elide_self=True, wire="bf16out")`):
#
#   A) 24-bit arm vs the ACCEPTED `bf16out` arm it substitutes -- the isolated deviation of
#      this candidate and nothing else. TOLERANCE row (atol = rtol = 1e-3).
#   A0) the same comparison at steps = 0, which collapses the fused kernel to
#      `normalize -> bf16 -> fp32`. This isolates the TRANSPORT error from the five-step
#      chain's amplification of it, so a FAIL on A with a clean A0 is diagnosable.
#   B) 24-bit arm vs the per-matrix `duplicated` loop -- the lineage mechanism every
#      earlier number on this axis was taken under.
#   C) 24-bit arm, matrix 0, vs the frozen phase-0 reference tensor -- THE GATE.
#   D) accepted `bf16out` arm vs the same duplicated loop -- the control that separates
#      "the codec moved the value" from "this arm was already off".
#   E) the same codec on the UN-elided pipelined column (where the self block travels the
#      wire too), which the scored job also times.
#   F) codec unit rows, on the region's own tensors: max relative round-trip error against
#      the 2^-16 bound the mechanism claims, and the identity property on a bf16-valued
#      input (a value with 8 mantissa bits must round-trip BITWISE through a 16-mantissa-
#      bit wire, so the accepted output leg's payload would be untouched by this codec).
#   G) guards -- a wire mode without `pipelined=True`, on the unfused compute path, or an
#      unknown mode name must be a HARD ERROR, never a silent full-width fallback, because
#      a silent fallback would let the arm be scored for a mechanism it did not run.
#
# Every equivalence row is run at THREE SEEDS, because a lossy row whose predecessor failed
# at 1.22x the gate must not rest on one draw (the same discipline stage-4 phase-5 applied
# to the bf16 pre-flight it withdrew on).
#
# The timing half runs both arms at the REAL region count (192 matrices, 96 owned per rank)
# so the phase has a cheap read on direction and size before a scored job is spent. It is a
# window, not the scored number.

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
    ap.add_argument("--seeds", nargs="+", type=int, default=[1234, 2025, 77])
    # build_stack() makes MATRIX 0 the frozen phase-0 input, and that input is a function of
    # the seed -- so row C (against the frozen reference tensor) is only meaningful at the
    # seed capture_reference.py used. At any other seed it would compare the region's answer
    # for one input against a reference computed from a DIFFERENT input, which is a
    # mismatched pair, not a failed gate. The extra seeds exist to re-draw rows A/B/E; C is
    # recorded as SKIPPED under them.
    ap.add_argument("--reference-seed", type=int, default=1234)
    ap.add_argument("--timing-count", type=int, default=192)
    ap.add_argument("--batch-chunk", type=int, default=64)
    ap.add_argument("--pipe-chunks", nargs="+", type=int, default=[4])
    ap.add_argument("--wire", default="bf16out24in", help="the candidate arm's mode")
    ap.add_argument("--baseline-wire", default="bf16out", help="the ACCEPTED arm's mode")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
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

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kernels import fused_ns  # noqa: E402
    from kernels.wire24 import WIRE24_AVAILABLE, roundtrip24  # noqa: E402

    probe = torch.zeros((1, 8, 8), device="cuda", dtype=torch.float32)
    assert fused_ns._supported(probe, args.coefficient_type, True), (
        "kernels.fused_ns._supported is False: the fused compute path would silently fall "
        "back to the library newton_schulz and this window would measure the wrong code"
    )
    del probe
    assert WIRE24_AVAILABLE, (
        "kernels.wire24 reports the Triton codec unavailable; the arm would raise rather "
        "than run, and a window that cannot run the arm proves nothing"
    )

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

    def record(shape, count, seed, name, a, b, bitwise_required=False, note=""):
        d = (a.float() - b.float()).abs()
        max_abs = d.max().item() if d.numel() else 0.0
        denom = b.float().abs().clamp_min(1e-30)
        max_rel = (d / denom).max().item() if d.numel() else 0.0
        bitwise = bool(torch.equal(a, b))
        ok = max_abs <= args.atol and (bitwise or not bitwise_required)
        row = {
            "axis": "egtp", "scope": shape, "arm": name, "world": world,
            "count": count, "seed": seed, "wire": args.wire,
            "max_abs_diff": f"{max_abs:.6e}",
            "mean_abs_diff": f"{(d.mean().item() if d.numel() else 0.0):.6e}",
            "max_rel_diff": f"{max_rel:.6e}",
            "bitwise_equal": bitwise,
            "bitwise_required": bitwise_required,
            "atol": args.atol, "rtol": args.rtol,
            "pass": "PASS" if ok else "FAIL",
            "note": note,
        }
        rows.append(row)
        log(f"  {shape:>11} n={count} s={seed} {name:<50} max_abs={row['max_abs_diff']} "
            f"bitwise={bitwise} {row['pass']}")

    def note_row(shape, count, seed, name, ok, note, max_abs=0.0):
        rows.append({
            "axis": "egtp", "scope": shape, "arm": name, "world": world,
            "count": count, "seed": seed, "wire": args.wire,
            "max_abs_diff": f"{max_abs:.6e}", "mean_abs_diff": "0.000000e+00",
            "max_rel_diff": "0.000000e+00", "bitwise_equal": bool(ok),
            "bitwise_required": False, "atol": args.atol, "rtol": args.rtol,
            "pass": "PASS" if ok else "FAIL", "note": note,
        })
        log(f"  {shape:>11} n={count} s={seed} {name:<50} {note} "
            f"{'PASS' if ok else 'FAIL'}")

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

        for seed in args.seeds:
            for count in args.counts:
                stack = build_stack(tag, m, n, count, world, rank, seed)
                log(f"{shape}: count={count}, seed={seed}, local {local_rows}x{n}, "
                    f"world={world}, wire={args.wire}")
                duplicated = torch.stack([
                    newton_schulz_tp(
                        stack[j], steps=args.steps,
                        coefficient_type=args.coefficient_type,
                        tp_group=group, partition_dim=0, tp_mode="duplicated",
                        use_syrk=True,
                    )
                    for j in range(count)
                ])
                for pc in args.pipe_chunks:
                    accepted = arm(stack, wire=args.baseline_wire, elide=True,
                                   pipe_chunks=pc)
                    wired = arm(stack, wire=args.wire, elide=True, pipe_chunks=pc)
                    record(shape, count, seed, f"A wire24(pc={pc}) vs accepted bf16out",
                           wired, accepted,
                           note="isolated candidate deviation; tolerance row")
                    record(shape, count, seed, f"B wire24(pc={pc}) vs duplicated-loop",
                           wired, duplicated, note="lineage mechanism")
                    if reference is not None and seed == args.reference_seed:
                        record(shape, count, seed,
                               f"C wire24(pc={pc})[0] vs phase0-reference",
                               wired[0], reference, note="THE GATE")
                    elif reference is not None:
                        note_row(shape, count, seed,
                                 f"C wire24(pc={pc})[0] vs phase0-reference", True,
                                 f"SKIPPED: reference tensor is frozen at seed "
                                 f"{args.reference_seed}; matrix 0 differs at this seed")
                    record(shape, count, seed,
                           f"D accepted(pc={pc}) vs duplicated-loop", accepted,
                           duplicated, note="control")

                    # A0: transport error isolated from the five-step amplification.
                    zero_ref = arm(stack, wire=args.baseline_wire, elide=True,
                                   pipe_chunks=pc, steps=0)
                    zero_wired = arm(stack, wire=args.wire, elide=True, pipe_chunks=pc,
                                     steps=0)
                    record(shape, count, seed,
                           f"A0 wire24(pc={pc},steps=0) vs accepted(steps=0)",
                           zero_wired, zero_ref, note="transport only, pre-chain")

                    # E: the same codec on the UN-elided column, where the self block is on
                    # the wire too. The scored job times that column as well.
                    un_ref = arm(stack, wire=args.baseline_wire, elide=False,
                                 pipe_chunks=pc)
                    un_wired = arm(stack, wire=args.wire, elide=False, pipe_chunks=pc)
                    record(shape, count, seed,
                           f"E wire24(pc={pc},no-elide) vs bf16out(no-elide)",
                           un_wired, un_ref, note="self block on the wire too")
                    del wired, accepted, zero_ref, zero_wired, un_ref, un_wired

                # F: codec unit rows on this region's own tensors.
                rt = roundtrip24(stack)
                err = (rt - stack).abs()
                rel = (err / stack.abs().clamp_min(1e-30)).max().item()
                note_row(shape, count, seed, "F codec rel err vs 2^-16 bound",
                         rel <= 2.0 ** -16 * 1.001, f"max_rel={rel:.6e} bound={2.0 ** -16:.6e}",
                         max_abs=err.max().item())
                bf = stack.to(torch.bfloat16).to(torch.float32)
                note_row(shape, count, seed, "F codec identity on bf16-valued input",
                         bool(torch.equal(roundtrip24(bf), bf)),
                         "a 16-mantissa-bit wire is the identity on an 8-bit value")
                del rt, err, bf
                del stack, duplicated
                torch.cuda.empty_cache()
        del reference
        torch.cuda.empty_cache()

    # ---- the preconditions must be HARD errors, not silent full-width fallbacks --------
    guard_shape = args.shapes[0]
    m, n = (int(v) for v in guard_shape.split("x"))
    guard = build_stack(f"egtp_{m}x{n}", m, n, 2 * world, world, rank, args.seeds[0])
    guards = []
    for name, kwargs in (
        ("wire24 without pipelined", dict(wire=args.wire, pipelined=False)),
        ("wire24 on the unfused path", dict(wire=args.wire, fused=False)),
        ("unknown wire mode", dict(wire="fp8out24in")),
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
        note_row(guard_shape, 2 * world, args.seeds[0], f"G guard: {name}",
                 outcome != "NO-RAISE", outcome)

    # ---- timing window at the real region count ---------------------------------------
    timing = {}
    if args.timing_count:
        for shape in args.shapes:
            m, n = (int(v) for v in shape.split("x"))
            stack = build_stack(f"egtp_{m}x{n}", m, n, args.timing_count, world, rank,
                                args.seeds[0])
            entry = {}
            for pc in args.pipe_chunks:
                entry[f"accepted_pc{pc}"] = bench(
                    lambda pc=pc: arm(stack, wire=args.baseline_wire, elide=True,
                                      pipe_chunks=pc),
                    args.warmup, args.iters)
                entry[f"wire24_pc{pc}"] = bench(
                    lambda pc=pc: arm(stack, wire=args.wire, elide=True, pipe_chunks=pc),
                    args.warmup, args.iters)
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, entry)
            mx = {k: max(g[k] for g in gathered) for k in entry}
            timing[shape] = {
                "per_rank": gathered,
                "max": mx,
                "delta_ms": {f"pc{pc}": mx[f"wire24_pc{pc}"] - mx[f"accepted_pc{pc}"]
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
        gate_rows = [r for r in rows if r["arm"].startswith("C ")]
        summary = {
            "gate": "equivalence", "atol": args.atol, "rtol": args.rtol, "world": world,
            "counts": args.counts, "seeds": args.seeds, "pipe_chunks": args.pipe_chunks,
            "wire": args.wire, "baseline_wire": args.baseline_wire,
            "shapes": args.shapes,
            "all_pass": all(r["pass"] == "PASS" for r in rows),
            "gate_rows": len(gate_rows),
            "gate_max_abs_diff": max(
                [float(r["max_abs_diff"]) for r in gate_rows] or [0.0]),
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
