# Equivalence gate + precondition guards + a real-count timing window for the
# stage-4 / phase-3 candidate `gtp-wire-bf16`.
#
# spec.toml sets workload_mode = "inference", so the gate is EQUIVALENCE, atol = rtol = 1e-3.
# The candidate changes the TRANSPORT ENCODING of the two exchange legs of
# `_fused_exchange_pipelined` (the accepted stage-4 phase-2 state) from fp32 to bf16. It
# changes nothing else: the deal, the matrices, the step count, the polar_express
# coefficients and the SYRK path are identical, the gather buffer handed to the fused
# Newton-Schulz kernel is still fp32 (so `fused_ns._supported` still holds and the fusion
# cannot silently un-bank), and the region still returns fp32.
#
# The two legs have DIFFERENT expected exactness and are therefore separate arms, never
# averaged:
#
#   bf16out  OUTPUT leg only. `fused_newton_schulz_batched` keeps X in bf16 through all 5
#            steps and only widens at `_cast_into`, so every value it returns is exactly
#            bf16-representable -- measured, not asserted: `preflight_wire_bf16.csv` reports
#            output_bf16_exact_frac = 1.000000 on all five GTP shapes and both EGTP shapes.
#            This arm must therefore be BITWISE, and that is the load-bearing row for the
#            output leg: any difference at all is a data-movement bug.
#   bf16in   INPUT leg only. This perturbs EVERY entry of the momentum before the
#            Newton-Schulz prologue, so it is gated at tolerance and nothing stronger.
#            `wire-bf16-preflight` (scripts/preflight_wire_bf16.py, job 2893451) measured
#            the post-5-step divergence at 4.9e-4 .. 9.8e-4 on the GTP shapes against the
#            1e-3 gate -- a PASS with ~2 % margin, and a FAIL (1.221e-3) on the EGTP shapes.
#            The margin is thin by measurement, so this arm is gated here at the real
#            gathered shapes, not argued from the preflight.
#   bf16     both legs.
#
# Rows: F (steps = 0, isolates data movement), A (vs the fp32 pipelined arm each mode
# suffixes), B (vs the per-matrix `duplicated` lineage), C (vs the frozen phase-0
# reference), D (control: the fp32 pipelined arm vs the lineage), E (guards: preconditions
# must RAISE, never silently fall back to the fp32 wire).

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
    WIRE_MODES,
    newton_schulz_tp_subgroup_fused,
)
from emerging_optimizers.orthogonalized_optimizers.muon_utils import (  # noqa: E402
    newton_schulz_tp,
)

# Modes whose A row is required to be BITWISE against the fp32 wire arm. Only the output
# leg qualifies, and only because the preflight MEASURED the region output to be entirely
# bf16-representable.
BITWISE_MODES = {"bf16out"}


def build_stack(shape, count, world, rank, seed):
    """Matrix 0 is the frozen Phase-0 input; the rest are distinct, so a staging or window
    bug that permutes matrices across subgroups, chunks or shapes cannot pass by
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--timing-out", default="")
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--shapes", nargs="+", required=True)
    ap.add_argument("--counts", nargs="+", type=int, required=True)
    ap.add_argument("--subgroup-sizes", nargs="+", type=int, required=True)
    ap.add_argument("--pipe-chunks", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--wire-modes", nargs="+", default=["bf16", "bf16in", "bf16out"])
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
    for w in args.wire_modes:
        assert w in WIRE_MODES, f"unknown wire mode {w}; known {sorted(WIRE_MODES)}"

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
            "axis": "gtp", "scope": scope, "arm": arm, "world": world,
            "max_abs_diff": f"{max_abs:.6e}",
            "mean_abs_diff": f"{d.mean().item():.6e}",
            "max_rel_diff": f"{(d / denom).max().item():.6e}",
            "bitwise_equal": bitwise, "bitwise_required": must_be_bitwise,
            "atol": args.atol, "rtol": args.rtol,
            "pass": "PASS" if ok else "FAIL",
        }
        rows_out.append(row)
        log(f"  {scope:>14} {arm:<74} max_abs={row['max_abs_diff']} "
            f"bitwise={bitwise} {row['pass']}")

    stacks = [
        build_stack(shape, count, world, rank, args.seed)
        for shape, count in zip(shapes, args.counts)
    ]
    log(f"world={world} shapes={args.shapes} counts={args.counts} "
        f"pipe_chunks={args.pipe_chunks} wire_modes={args.wire_modes}")

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

    def fused(balanced, g, pc, steps, wire=""):
        return newton_schulz_tp_subgroup_fused(
            stacks, steps=steps, coefficient_type=args.coefficient_type,
            tp_group=group, use_syrk=True, subgroup_size=g, balanced=balanced,
            pipelined=bool(pc), pipe_chunks=pc, wire=wire,
        )

    for g in args.subgroup_sizes:
        if world % g or g >= world:
            log(f"skipping g={g}: does not divide {world} or is not smaller than it")
            continue
        log(f"g={g} (k={world // g}):")
        for balanced in (False, True):
            base_tag = f"fuse_sub_g{g}" + ("_bal" if balanced else "")
            for pc in args.pipe_chunks:
                ref_tag = f"{base_tag}_pipe{pc}"
                # The arm every wire mode suffixes: the ACCEPTED fp32-wire pipelined state.
                ref = fused(balanced, g, pc, args.steps)
                ref0 = fused(balanced, g, pc, 0)
                for i, (m, n) in enumerate(shapes):
                    record(f"{m}x{n}", f"D control: {ref_tag} (fp32 wire) vs duplicated "
                                       f"lineage", ref[i], lineage[i])
                for wire in args.wire_modes:
                    tag = f"{ref_tag}_w{wire}"
                    must_bitwise = wire in BITWISE_MODES
                    # F) steps = 0 isolates the DATA MOVEMENT from the 5-step chain.
                    got0 = fused(balanced, g, pc, 0, wire)
                    for i, (m, n) in enumerate(shapes):
                        record(f"{m}x{n}",
                               f"F exchange-only(steps=0): {tag} vs {ref_tag}",
                               got0[i], ref0[i], must_be_bitwise=must_bitwise)
                    del got0
                    got = fused(balanced, g, pc, args.steps, wire)
                    for i, (m, n) in enumerate(shapes):
                        assert got[i].shape == stacks[i].shape, (
                            f"{tag}: shape {i} returned {tuple(got[i].shape)}"
                        )
                        assert got[i].dtype is torch.float32, (
                            f"{tag}: shape {i} returned {got[i].dtype}, not fp32 -- the "
                            "wire modes change TRANSPORT only"
                        )
                        record(f"{m}x{n}",
                               f"A: {tag} vs the arm it substitutes ({ref_tag})",
                               got[i], ref[i], must_be_bitwise=must_bitwise)
                        record(f"{m}x{n}", f"B: {tag} vs duplicated lineage",
                               got[i], lineage[i])
                        if references[i] is not None:
                            record(f"{m}x{n}", f"C: {tag} matrix 0 vs phase0-reference",
                                   got[i][0], references[i])
                    del got
                    torch.cuda.empty_cache()
                del ref, ref0
                torch.cuda.empty_cache()

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
                for pc in args.pipe_chunks:
                    for wire in [""] + list(args.wire_modes):
                        tag = f"{base_tag}_pipe{pc}" + (f"_w{wire}" if wire else "")
                        timing["arms"].setdefault(tag, {})["ms"] = time_call(
                            lambda b=balanced, p=pc, w=wire: fused(b, g, p, args.steps, w)
                        )
                        torch.cuda.empty_cache()
                    log(f"  timing g={g} balanced={balanced} pipe{pc}: " + ", ".join(
                        f"{w or 'fp32'}="
                        f"{timing['arms'][f'{base_tag}_pipe{pc}' + (f'_w{w}' if w else '')]['ms']:.3f}"
                        for w in [""] + list(args.wire_modes)
                    ))

    # ---- E) guards: every precondition must RAISE, not fall back to the fp32 wire -------
    g_ok = next((g for g in args.subgroup_sizes if world % g == 0 and g < world), world // 2)
    guards = [
        ("wire mode on the MONOLITHIC fused exchange", dict(pc=0, wire="bf16")),
        ("unknown wire mode", dict(pc=2, wire="fp8_e4m3_nonexistent")),
    ]
    for label, kwargs in guards:
        raised = ""
        try:
            newton_schulz_tp_subgroup_fused(
                stacks, steps=args.steps, coefficient_type=args.coefficient_type,
                tp_group=group, use_syrk=True, subgroup_size=g_ok, balanced=True,
                pipelined=bool(kwargs["pc"]), pipe_chunks=kwargs["pc"],
                wire=kwargs["wire"],
            )
        except ValueError as error:
            raised = f"ValueError: {error}"
        rows_out.append({
            "axis": "gtp", "scope": "guard", "arm": f"E: {label}", "world": world,
            "max_abs_diff": "", "mean_abs_diff": "", "max_rel_diff": "",
            "bitwise_equal": "", "bitwise_required": "", "atol": args.atol,
            "rtol": args.rtol, "pass": "PASS" if raised else "FAIL",
        })
        log(f"  {'guard':>14} {('E: ' + label):<74} raised={raised or 'NOTHING'} "
            f"{'PASS' if raised else 'FAIL'}")

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
            "subgroup_sizes": args.subgroup_sizes, "pipe_chunks": args.pipe_chunks,
            "wire_modes": args.wire_modes,
            "all_pass": all(r["pass"] == "PASS" for r in rows_out),
            "per_mode_all_pass": {
                w: all(r["pass"] == "PASS" for r in rows_out if f"_w{w} " in r["arm"] + " "
                       or f"_w{w}" in r["arm"])
                for w in args.wire_modes
            },
            "bf16out_A_all_bitwise": all(
                r["bitwise_equal"] is True for r in rows_out
                if r["arm"].startswith("A: ") and "_wbf16out" in r["arm"]
            ),
            "max_abs_diff": worst(lambda r: True),
            "max_abs_diff_a_arm": worst(lambda r: r["arm"].startswith("A: ")),
            "max_abs_diff_a_bf16in": worst(
                lambda r: r["arm"].startswith("A: ") and "_wbf16in" in r["arm"]
            ),
            "max_abs_diff_a_bf16": worst(
                lambda r: r["arm"].startswith("A: ") and r["arm"].split(" vs ")[0].endswith("_wbf16")
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
