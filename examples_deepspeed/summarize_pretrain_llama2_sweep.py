#!/usr/bin/env python3
"""Summarize llama2 sweep logs into CSV + markdown table text.

Example:
  python3 ./examples_deepspeed/summarize_pretrain_llama2_sweep.py \
      --log-dir ./.logging/0414 --date-tag 0414
"""

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MEM_RE = re.compile(
    r"\(after\s+(\d+)\s+iterations\)\s+memory \(MB\)\s+\|\s+allocated:\s+([0-9.]+)\s+\|\s+max allocated:\s+([0-9.]+)\s+\|\s+reserved:\s+([0-9.]+)\s+\|\s+max reserved:\s+([0-9.]+)"
)
CPU_RE = re.compile(r"CPU Virtual Memory:\s+used =\s*([0-9.]+)\s*GB")
ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+\s+\|.*elapsed time per iteration \(ms\):\s*([0-9.]+)\s+\|.*tokens per gpu per second \(tgs\):\s*([0-9.]+)"
)
OPT_STEP_RE = re.compile(r"optimizer_step:\s*([0-9.]+)")
BWD_RE = re.compile(r"\|\s*bwd:\s*([0-9.]+)")

# Supports:
#   0414_FR_llama1B_DP1_..._ZeRO0_..._nooffload.log
#   0403_DDP_FR_llama1B_DP1_..._ZeRO0_..._nooffload.log
LOG_NAME_RE = re.compile(
    r"^(?P<date>\d{4})_(?:(?P<label>[A-Za-z0-9]+)_)?(?P<model>FR|CoLA)_llama1B_DP(?P<dp>\d+)_TP\d+_PP\d+_ZeRO(?P<zero>\d+)_seq\d+_mbz\d+_gbz\d+_(?P<offload>nooffload|cpuoffload)\.log$"
)

HEADER = [
    "Model",
    "DP",
    "ZeRO",
    "Offload",
    "Peak GPU Alloc (GB)",
    "Peak GPU Reserved (GB)",
    "CPU Mem (GB)",
    "Iter Time (ms)",
    "Tokens/s/GPU",
    "Optimizer Step (ms)",
    "Backward (ms)",
    "Peak Gap (GB)",
]


def stable_values(values: List[float]) -> List[float]:
    if len(values) <= 1:
        return values
    return values[1:]


def fmt_float(v: Optional[float], ndigits: int) -> str:
    if v is None:
        return ""
    return f"{v:.{ndigits}f}"


def parse_metrics(log_path: Path) -> Dict[str, Optional[float]]:
    mem_rows: List[Tuple[int, float, float]] = []
    cpu_vals: List[float] = []
    iter_times: List[float] = []
    tgs_vals: List[float] = []
    opt_steps: List[float] = []
    bwd_vals: List[float] = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = MEM_RE.search(line)
            if m:
                iter_idx = int(m.group(1))
                max_alloc_mb = float(m.group(3))
                max_res_mb = float(m.group(5))
                mem_rows.append((iter_idx, max_alloc_mb, max_res_mb))
                continue

            m = CPU_RE.search(line)
            if m:
                cpu_vals.append(float(m.group(1)))
                continue

            m = ITER_RE.search(line)
            if m:
                iter_idx = int(m.group(1))
                if iter_idx >= 2:
                    iter_times.append(float(m.group(2)))
                    tgs_vals.append(float(m.group(3)))
                continue

            m = OPT_STEP_RE.search(line)
            if m:
                opt_steps.append(float(m.group(1)))
                continue

            m = BWD_RE.search(line)
            if m:
                bwd_vals.append(float(m.group(1)))

    stable_mem = [(a, r) for i, a, r in mem_rows if i >= 2]
    if not stable_mem:
        stable_mem = [(a, r) for _, a, r in mem_rows]
    alloc_vals = [a for a, _ in stable_mem]
    res_vals = [r for _, r in stable_mem]

    peak_alloc_gb = statistics.median(alloc_vals) / 1024.0 if alloc_vals else None
    peak_res_gb = statistics.median(res_vals) / 1024.0 if res_vals else None
    cpu_mem_gb = max(cpu_vals) if cpu_vals else None
    iter_time_ms = statistics.median(iter_times) if iter_times else None
    tgs = int(round(statistics.median(tgs_vals))) if tgs_vals else None

    stable_opt = stable_values(opt_steps)
    stable_bwd = stable_values(bwd_vals)
    opt_ms = statistics.median(stable_opt) if stable_opt else None
    bwd_ms = statistics.median(stable_bwd) if stable_bwd else None

    peak_gap_gb = None
    if peak_alloc_gb is not None and peak_res_gb is not None:
        peak_gap_gb = peak_res_gb - peak_alloc_gb

    return {
        "Peak GPU Alloc (GB)": peak_alloc_gb,
        "Peak GPU Reserved (GB)": peak_res_gb,
        "CPU Mem (GB)": cpu_mem_gb,
        "Iter Time (ms)": iter_time_ms,
        "Tokens/s/GPU": tgs,
        "Optimizer Step (ms)": opt_ms,
        "Backward (ms)": bwd_ms,
        "Peak Gap (GB)": peak_gap_gb,
    }


def key_order(key: Tuple[str, int, int, str]) -> Tuple[int, int, int, int]:
    model, dp, zero, offload = key
    model_rank = 0 if model == "FR" else 1
    offload_rank = 0 if offload == "No" else 1
    return (dp, zero, offload_rank, model_rank)


def expected_12_keys() -> List[Tuple[str, int, int, str]]:
    keys: List[Tuple[str, int, int, str]] = []
    for dp in (1, 4):
        keys.append(("FR", dp, 0, "No"))
        keys.append(("CoLA", dp, 0, "No"))
    for offload in ("No", "Yes"):
        for dp in (1, 4):
            keys.append(("FR", dp, 1, offload))
            keys.append(("CoLA", dp, 1, offload))
    return keys


def discover_logs(log_dir: Path, date_tag: str) -> Dict[Tuple[str, int, int, str], Path]:
    selected: Dict[Tuple[str, int, int, str], Path] = {}
    for log_path in sorted(log_dir.glob(f"{date_tag}_*.log")):
        m = LOG_NAME_RE.match(log_path.name)
        if not m:
            continue
        model = m.group("model")
        dp = int(m.group("dp"))
        zero = int(m.group("zero"))
        offload = "Yes" if m.group("offload") == "cpuoffload" else "No"
        key = (model, dp, zero, offload)
        # If duplicate key exists, keep the latest modified log.
        if key in selected:
            if log_path.stat().st_mtime > selected[key].stat().st_mtime:
                selected[key] = log_path
        else:
            selected[key] = log_path
    return selected


def write_csv(out_csv: Path, rows: List[Dict[str, object]]) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for r in rows:
            writer.writerow(
                [
                    r["Model"],
                    r["DP"],
                    r["ZeRO"],
                    r["Offload"],
                    fmt_float(r["Peak GPU Alloc (GB)"], 2),  # type: ignore[arg-type]
                    fmt_float(r["Peak GPU Reserved (GB)"], 2),  # type: ignore[arg-type]
                    fmt_float(r["CPU Mem (GB)"], 2),  # type: ignore[arg-type]
                    fmt_float(r["Iter Time (ms)"], 1),  # type: ignore[arg-type]
                    "" if r["Tokens/s/GPU"] is None else str(r["Tokens/s/GPU"]),
                    fmt_float(r["Optimizer Step (ms)"], 1),  # type: ignore[arg-type]
                    fmt_float(r["Backward (ms)"], 1),  # type: ignore[arg-type]
                    fmt_float(r["Peak Gap (GB)"], 2),  # type: ignore[arg-type]
                ]
            )


def write_txt(out_txt: Path, rows: List[Dict[str, object]]) -> None:
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(
            "| Model | DP | ZeRO | Offload | Peak GPU Alloc (GB) | Peak GPU Reserved (GB) | CPU Mem (GB) | Iter Time (ms) | Tokens/s/GPU | Optimizer Step (ms) | Backward (ms) | Peak Gap (GB) |\n"
        )
        f.write(
            "| ----- | -: | ---: | ------: | ------------------: | ---------------------: | -----------: | -------------: | -----------: | ------------------: | ------------: | ------------: |\n"
        )
        for r in rows:
            f.write(
                "| {Model} | {DP} | {ZeRO} | {Offload} | {alloc} | {res} | {cpu} | {iter_ms} | {tgs} | {opt} | {bwd} | {gap} |\n".format(
                    Model=r["Model"],
                    DP=r["DP"],
                    ZeRO=r["ZeRO"],
                    Offload=r["Offload"],
                    alloc=fmt_float(r["Peak GPU Alloc (GB)"], 2),  # type: ignore[arg-type]
                    res=fmt_float(r["Peak GPU Reserved (GB)"], 2),  # type: ignore[arg-type]
                    cpu=fmt_float(r["CPU Mem (GB)"], 2),  # type: ignore[arg-type]
                    iter_ms=fmt_float(r["Iter Time (ms)"], 1),  # type: ignore[arg-type]
                    tgs="" if r["Tokens/s/GPU"] is None else str(r["Tokens/s/GPU"]),
                    opt=fmt_float(r["Optimizer Step (ms)"], 1),  # type: ignore[arg-type]
                    bwd=fmt_float(r["Backward (ms)"], 1),  # type: ignore[arg-type]
                    gap=fmt_float(r["Peak Gap (GB)"], 2),  # type: ignore[arg-type]
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sweep logs into csv/txt.")
    parser.add_argument("--log-dir", required=True, help="Directory containing run .log files.")
    parser.add_argument("--date-tag", required=True, help="Date tag prefix, e.g. 0403 or 0414.")
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output filename prefix. Default: <date-tag>_12exp_summary under --log-dir.",
    )
    parser.add_argument(
        "--strict-12",
        action="store_true",
        help="Emit exactly the 12 expected rows (fill missing rows with blanks).",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    if not log_dir.exists():
        raise FileNotFoundError(f"log directory not found: {log_dir}")

    out_prefix = args.out_prefix or f"{args.date_tag}_12exp_summary"
    out_csv = log_dir / f"{out_prefix}.csv"
    out_txt = log_dir / f"{out_prefix}.txt"

    logs_by_key = discover_logs(log_dir, args.date_tag)

    if args.strict_12:
        ordered_keys = expected_12_keys()
    else:
        ordered_keys = sorted(logs_by_key.keys(), key=key_order)

    rows: List[Dict[str, object]] = []
    for model, dp, zero, offload in ordered_keys:
        key = (model, dp, zero, offload)
        if key in logs_by_key:
            metrics = parse_metrics(logs_by_key[key])
        else:
            metrics = {
                "Peak GPU Alloc (GB)": None,
                "Peak GPU Reserved (GB)": None,
                "CPU Mem (GB)": None,
                "Iter Time (ms)": None,
                "Tokens/s/GPU": None,
                "Optimizer Step (ms)": None,
                "Backward (ms)": None,
                "Peak Gap (GB)": None,
            }
        rows.append({"Model": model, "DP": dp, "ZeRO": zero, "Offload": offload, **metrics})

    write_csv(out_csv, rows)
    write_txt(out_txt, rows)

    print(f"[summary] logs dir: {log_dir}")
    print(f"[summary] matched runs: {len(logs_by_key)}")
    print(f"[summary] wrote csv: {out_csv}")
    print(f"[summary] wrote txt: {out_txt}")


if __name__ == "__main__":
    main()
