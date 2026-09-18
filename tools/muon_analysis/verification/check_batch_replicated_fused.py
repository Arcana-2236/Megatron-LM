# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cheap functional + timing window for stage-4 rank 7 `gtp-replicated-fused-batched-ns`.

ONE GPU, no collectives -- and that is not a simplification, it is the region: the two
scored rows are `shard_count == 1` (replicated) GTP shapes, and at `partition_dim is None`
``newton_schulz_tp_batched`` short-circuits BEFORE any collective. The candidate routes
exactly that short-circuit through ``kernels.fused_ns.fused_newton_schulz_batched``, so the
whole claim -- numerics and delta -- is settleable on a single rank.

Arms (all at the real region shape: 24 x 8192x2048 and 12 x 512x8192, --batch-chunk 4):

  A  fused batched vs the UNFUSED batched arm it substitutes -- the accepted state's
     ``duplicated_batch`` column. Per chunk, so a tail-chunk bug cannot hide in an
     aggregate.
  B  fused batched vs the per-matrix ``newton_schulz_tp`` loop (``duplicated_set``), the
     lineage mechanism both batched arms descend from.
  C  fused batched, on the FROZEN phase-0 input, vs the frozen phase-0 reference tensor.
  D  control: the UNFUSED batched arm vs the same per-matrix loop. Bounds how much of A's
     and B's difference is the batching that the reference already banked.
  E  guards: with ``fused=True`` the fused entry point's silent fallback must be an ERROR,
     never a no-op, or the scored column would time the reference arm under the
     candidate's name.

Timing: the two batched arms under ONE composition, same tensors, same chunking, so the
delta is the prologue/epilogue fusion and nothing else.

Writes equivalence.csv and perf_window.json.
"""

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
WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools", "muon_analysis"))
sys.path.insert(0, WORK)

from bench_ns_strategies import newton_schulz_tp_batched  # noqa: E402
from emerging_optimizers.orthogonalized_optimizers.muon_utils import (  # noqa: E402
    newton_schulz_tp,
)
from kernels.fused_ns import FUSED_NS_AVAILABLE  # noqa: E402


def _time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end))
    return statistics.median(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="+", default=["8192x2048", "512x8192"])
    p.add_argument("--counts", nargs="+", type=int, default=[24, 12])
    p.add_argument("--batch-chunk", type=int, default=4)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--coefficient-type", type=str, default="polar_express")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--reference-dir", type=str,
                   default=os.path.join(WORK, "reference_outputs"))
    p.add_argument("--out-csv", type=str, required=True)
    p.add_argument("--out-json", type=str, required=True)
    args = p.parse_args()
    assert len(args.shapes) == len(args.counts)

    assert FUSED_NS_AVAILABLE, (
        "fused NS kernel unavailable (needs Triton >= 3.4 + batched_tsyrk_ex)"
    )
    # The benchmark sets this from --fp32-matmul-prec (default "medium", matching
    # Megatron's muon_fp32_matmul_prec). kernels.fused_ns._supported REQUIRES it, and
    # phase-3 lost a probe to omitting it -- so set it and assert it.
    torch.set_float32_matmul_precision("medium")
    assert torch.get_float32_matmul_precision() == "medium"
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    # A real (1-rank) group, not None: the replicated path never touches it, but the
    # per-matrix lineage arm goes through the library's ``newton_schulz_tp``, and passing
    # it the same object the benchmark passes keeps the two call sites identical.
    torch.distributed.init_process_group(backend="nccl")
    group = torch.distributed.distributed_c10d._get_default_group()
    torch.manual_seed(args.seed)

    rows_out = []
    timing = {}
    ok = True

    def record(shape, count, arm, got, ref, note, require_pass=True):
        nonlocal ok
        diff = (got - ref).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        max_rel = (diff / ref.abs().clamp_min(1e-12)).max().item()
        passed = bool(torch.allclose(got, ref, atol=args.atol, rtol=args.rtol))
        if require_pass:
            ok = ok and passed
        rows_out.append({
            "axis": "gtp", "shape": shape, "count": count, "arm": arm, "world": 1,
            "max_abs_diff": f"{max_abs:.6e}", "mean_abs_diff": f"{mean_abs:.6e}",
            "max_rel_diff": f"{max_rel:.6e}",
            "bitwise_equal": bool(max_abs == 0.0),
            "atol": args.atol, "rtol": args.rtol,
            "pass": "PASS" if passed else "FAIL", "note": note,
        })

    for shape, count in zip(args.shapes, args.counts):
        rows, cols = (int(v) for v in shape.split("x"))
        chunk = args.batch_chunk
        x = torch.randn((count, rows, cols), device="cuda", dtype=torch.float32)

        def batched(fused):
            # EXACTLY time_group's batched arm: chunked calls into
            # newton_schulz_tp_batched at partition_dim=None (shard_count == 1).
            out = []
            for a in range(0, count, chunk):
                out.append(newton_schulz_tp_batched(
                    x[a : a + chunk], steps=args.steps,
                    coefficient_type=args.coefficient_type, tp_group=group,
                    partition_dim=None, tp_mode="duplicated", use_syrk=True,
                    fused=fused,
                ))
            return torch.cat(out, dim=0)

        def per_matrix():
            return torch.stack([
                newton_schulz_tp(
                    x[i], steps=args.steps, coefficient_type=args.coefficient_type,
                    tp_group=group, partition_dim=None, tp_mode="duplicated",
                    use_syrk=True,
                )
                for i in range(count)
            ])

        ref_unfused = batched(False)
        got_fused = batched(True)
        ref_loop = per_matrix()

        for a in range(0, count, chunk):
            n = min(chunk, count - a)
            record(shape, n, f"A fused vs unfused batched [chunk={a}:{a + n}]",
                   got_fused[a : a + n], ref_unfused[a : a + n],
                   "substituted arm (accepted duplicated_batch column)")
        record(shape, count, "B fused batched vs per-matrix duplicated loop",
               got_fused, ref_loop, "lineage mechanism")
        record(shape, count, "D unfused batched vs per-matrix duplicated loop (control)",
               ref_unfused, ref_loop, "control: batching the reference already banked")

        m = min(rows, cols)
        probe_r = ref_unfused[0] if rows <= cols else ref_unfused[0].mT
        probe_g = got_fused[0] if rows <= cols else got_fused[0].mT
        eye = torch.eye(m, device="cuda", dtype=torch.float32)
        dev_r = ((probe_r @ probe_r.mT) - eye).abs().max().item()
        dev_g = ((probe_g @ probe_g.mT) - eye).abs().max().item()

        del ref_unfused, got_fused, ref_loop
        torch.cuda.empty_cache()

        # ---- C: frozen phase-0 lineage -------------------------------------------------
        ref_path = os.path.join(args.reference_dir, f"gtp_{rows}x{cols}.pt")
        if os.path.exists(ref_path):
            gen = torch.Generator(device="cuda")
            gen.manual_seed(args.seed + zlib.crc32(f"gtp_{rows}x{cols}".encode()) % 100000)
            frozen = torch.randn((rows, cols), device="cuda", dtype=torch.float32,
                                 generator=gen)
            stack = frozen.unsqueeze(0).expand(chunk, rows, cols).contiguous()
            y = newton_schulz_tp_batched(
                stack, steps=args.steps, coefficient_type=args.coefficient_type,
                tp_group=group, partition_dim=None, tp_mode="duplicated", use_syrk=True,
                fused=True,
            )
            frozen_ref = torch.load(ref_path).cuda()
            record(shape, 1, "C fused batched[0] on frozen input vs phase-0 reference",
                   y[0], frozen_ref, "phase-0 lineage")
            del frozen, stack, y, frozen_ref
            torch.cuda.empty_cache()
        else:
            rows_out.append({
                "axis": "gtp", "shape": shape, "count": 1,
                "arm": "C phase-0 reference", "world": 1, "max_abs_diff": "",
                "mean_abs_diff": "", "max_rel_diff": "", "bitwise_equal": "",
                "atol": args.atol, "rtol": args.rtol, "pass": "SKIP",
                "note": f"no frozen reference at {ref_path}",
            })

        # ---- timing window -------------------------------------------------------------
        unfused_ms = _time(lambda: batched(False), args.iters, args.warmup)
        fused_ms = _time(lambda: batched(True), args.iters, args.warmup)
        timing[shape] = {
            "count": count, "batch_chunk": chunk,
            "unfused_batched_ms": unfused_ms, "fused_batched_ms": fused_ms,
            "delta_ms": fused_ms - unfused_ms,
            "orthogonality_dev_unfused": dev_r,
            "orthogonality_dev_fused": dev_g,
        }
        del x
        torch.cuda.empty_cache()

    # ---- E: guards -- the fused entry point's silent fallback must be an ERROR ---------
    guard_x = torch.randn((2, 512, 8192), device="cuda", dtype=torch.float32)

    def guard(name, fn):
        try:
            fn()
        except RuntimeError as exc:
            status, note = "PASS", f"raised RuntimeError: {str(exc)[:90]}"
        else:
            status, note = "FAIL", "no error raised -- silent fallback possible"
        rows_out.append({
            "axis": "gtp", "shape": "guard", "count": 0, "arm": f"E guard: {name}",
            "world": 1, "max_abs_diff": "", "mean_abs_diff": "", "max_rel_diff": "",
            "bitwise_equal": "", "atol": args.atol, "rtol": args.rtol,
            "pass": status, "note": note,
        })
        return status == "PASS"

    def _call(t, prec="medium"):
        old = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision(prec)
        try:
            return newton_schulz_tp_batched(
                t, steps=args.steps, coefficient_type=args.coefficient_type,
                tp_group=group, partition_dim=None, tp_mode="duplicated", use_syrk=True,
                fused=True,
            )
        finally:
            torch.set_float32_matmul_precision(old)

    guards_ok = True
    guards_ok &= guard("fp32_matmul_precision != medium", lambda: _call(guard_x, "high"))
    guards_ok &= guard("non-contiguous input", lambda: _call(guard_x.transpose(1, 2)))
    guards_ok &= guard("use_syrk=False", lambda: newton_schulz_tp_batched(
        guard_x, steps=args.steps, coefficient_type=args.coefficient_type,
        tp_group=group, partition_dim=None, tp_mode="duplicated", use_syrk=False,
        fused=True))
    ok = ok and guards_ok
    del guard_x
    torch.cuda.empty_cache()

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    timing["total_delta_ms"] = sum(
        v["delta_ms"] for v in timing.values() if isinstance(v, dict)
    )
    timing["all_pass"] = ok
    timing["guards_pass"] = bool(guards_ok)
    timing["rows"] = len(rows_out)
    timing["max_abs_diff"] = max(
        (float(r["max_abs_diff"]) for r in rows_out if r["max_abs_diff"]), default=0.0
    )
    with open(args.out_json, "w") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2), flush=True)
    for r in rows_out:
        print(r, flush=True)
    print(f"EQUIVALENCE_SUMMARY all_pass={ok} rows={len(rows_out)} "
          f"max_abs_diff={timing['max_abs_diff']:.6e}", flush=True)
    if not ok:
        raise SystemExit("EQUIVALENCE FAILED")


if __name__ == "__main__":
    main()
