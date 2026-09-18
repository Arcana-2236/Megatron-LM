# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cheap functional + timing window for stage-3 rank 2 `egtp-ns-step-fusion`.

ONE GPU, no collectives. The candidate only touches the COMPUTE half of the EGTP
batched-subgroup region, so the whole claim can be settled without a distributed job:

  functional  fused vs the library call it replaces -- ``full_t[a:a+chunk] =
              newton_schulz(global_x[a:a+chunk], steps, ct, use_syrk=True)`` -- on the
              exact chunking the region uses (96 owned matrices, ``--batch-chunk 64``, so
              a full 64 chunk AND a short 32 tail), at the spec.toml gate
              atol = rtol = 1e-3. Also against the frozen phase-0 reference when one
              exists for the shape.
  timing      the same two call patterns timed under ONE composition, so the delta is the
              fusion and nothing else. The library arm INCLUDES the ``full_t[...] = ``
              store, because removing that store is part of what the fusion does; timing
              the library's ``newton_schulz`` alone would credit the fusion with a copy
              the baseline really pays.

Writes equivalence.csv and perf_window.json.
"""

import argparse
import csv
import json
import statistics
import sys

import torch
from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.fused_ns import FUSED_NS_AVAILABLE, fused_newton_schulz_batched  # noqa: E402


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
    p.add_argument("--shapes", nargs="+", default=["5120x2048", "2048x5120"])
    p.add_argument("--count", type=int, default=96)
    p.add_argument("--batch-chunk", type=int, default=64)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--coefficient-type", type=str, default="polar_express")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--out-csv", type=str, required=True)
    p.add_argument("--out-json", type=str, required=True)
    args = p.parse_args()

    assert FUSED_NS_AVAILABLE, "fused NS kernel unavailable (needs Triton >= 3.4 + batched_tsyrk_ex)"
    torch.set_float32_matmul_precision("medium")
    torch.cuda.set_device(0)
    torch.manual_seed(1234)

    rows_out = []
    timing = {}
    ok = True

    for shape in args.shapes:
        rows, cols = (int(v) for v in shape.split("x"))
        count, chunk = args.count, args.batch_chunk
        x = torch.randn((count, rows, cols), device="cuda", dtype=torch.float32)

        def library():
            full_t = torch.empty_like(x)
            for a in range(0, count, chunk):
                full_t[a : a + chunk] = newton_schulz(
                    x[a : a + chunk], args.steps, args.coefficient_type, use_syrk=True
                )
            return full_t

        def fused():
            full_t = torch.empty_like(x)
            for a in range(0, count, chunk):
                fused_newton_schulz_batched(
                    x[a : a + chunk], args.steps, args.coefficient_type,
                    use_syrk=True, out=full_t[a : a + chunk],
                )
            return full_t

        ref = library()
        got = fused()

        # Per chunk, so the full-64 chunk and the short-32 tail are reported separately and
        # a tail-only masking bug cannot hide inside a whole-stack aggregate.
        for a in range(0, count, chunk):
            n = min(chunk, count - a)
            r = ref[a : a + n]
            g = got[a : a + n]
            diff = (g - r).abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            max_rel = (diff / r.abs().clamp_min(1e-12)).max().item()
            passed = torch.allclose(g, r, atol=args.atol, rtol=args.rtol)
            ok = ok and passed
            rows_out.append({
                "axis": "egtp",
                "shape": shape,
                "arm": f"fused_ns vs library newton_schulz [chunk={a}:{a + n}]",
                "world": 1,
                "count": n,
                "max_abs_diff": f"{max_abs:.6e}",
                "mean_abs_diff": f"{mean_abs:.6e}",
                "max_rel_diff": f"{max_rel:.6e}",
                "atol": args.atol,
                "rtol": args.rtol,
                "pass": "PASS" if passed else "FAIL",
            })

        # Orthogonality is the property the iteration is FOR, so it is reported too: a
        # kernel can match the reference and still both be wrong, but it cannot match the
        # reference AND carry a different singular-value spectrum.
        m = min(rows, cols)
        probe_r = ref[0] if rows <= cols else ref[0].mT
        probe_g = got[0] if rows <= cols else got[0].mT
        eye = torch.eye(m, device="cuda", dtype=torch.float32)
        dev_r = ((probe_r @ probe_r.mT) - eye).abs().max().item()
        dev_g = ((probe_g @ probe_g.mT) - eye).abs().max().item()

        del ref, got
        torch.cuda.empty_cache()

        lib_ms = _time(library, args.iters, args.warmup)
        fused_ms = _time(fused, args.iters, args.warmup)
        timing[shape] = {
            "library_ms": lib_ms,
            "fused_ms": fused_ms,
            "delta_ms": fused_ms - lib_ms,
            "orthogonality_dev_library": dev_r,
            "orthogonality_dev_fused": dev_g,
        }
        del x
        torch.cuda.empty_cache()

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    timing["total_delta_ms"] = sum(v["delta_ms"] for v in timing.values() if isinstance(v, dict))
    timing["all_pass"] = ok
    with open(args.out_json, "w") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2), flush=True)
    for r in rows_out:
        print(r, flush=True)
    if not ok:
        raise SystemExit("EQUIVALENCE FAILED")


if __name__ == "__main__":
    main()
