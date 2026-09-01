#!/bin/bash
# Runs both dist_muon Newton-Schulz benchmark axes (dense/GTP + expert/EGTP) as one
# blocking command and reports a single combined step-time metric.
#
# Written for PerfBot autopilot: the two sbatch drivers in this directory each model
# one half of a real dist_muon optimizer step (see README.md), so an apples-to-apples
# "how long does one optimizer step take" number needs both.
#
# --use-syrk is forced on for every attempt (USE_SYRK=1), matching the always-on
# Triton triangular-kernel path this workload is locked to. Blockwise NS is never
# requested (the sbatch scripts' --modes default is duplicated+distributed).
#
# Usage: bash tools/muon_analysis/run_dist_muon_bench.sh
# Can be invoked from anywhere; it resolves its own paths.
#
# Expect this to be SLOW end-to-end. The 16-node GTP job has been observed queuing
# ~10h on gb300, and that wait is paid per attempt by design (see MAX_WAIT_SECONDS).

set -uo pipefail  # deliberately NOT -e: job failures are handled explicitly below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT_DIR="$(cd "${REPO_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# Must match ROOT_DIR/NAME in bench_ns_{gtp,egtp}.sbatch, which write their logs to
# ${ROOT_DIR}/runs/ns_bench/<axis>/logs -- i.e. beside the repo, NOT inside it. If
# those scripts' ROOT_DIR changes, this has to change with it.
GTP_LOG_DIR="${ROOT_DIR}/runs/ns_bench/gtp${TAG:+_$TAG}/logs"
EGTP_LOG_DIR="${ROOT_DIR}/runs/ns_bench/egtp${TAG:+_$TAG}/logs"

POLL_SECONDS=${POLL_SECONDS:-60}
# Deep gb300 queue: a 16-node job was measured starting ~10h after submission, so a
# short ceiling would abort healthy attempts. 24h by default.
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-86400}

submitted_ids=()
cleanup() {
    if [[ ${#submitted_ids[@]} -gt 0 ]]; then
        echo "[run_dist_muon_bench] interrupted -- cancelling ${submitted_ids[*]}" >&2
        scancel "${submitted_ids[@]}" 2>/dev/null
    fi
}
trap cleanup INT TERM

submit_axis() {  # $1 = sbatch file relative to REPO_DIR; echoes the jobid
    sbatch --parsable --export=ALL,USE_SYRK=1 "$1"
}

# Poll each id separately: `squeue -j a,b` errors out when one id has already been
# purged, which would look like "both finished" while the other is still running.
job_active() {
    squeue -h -j "$1" -o "%T" 2>/dev/null \
        | grep -qE 'PENDING|RUNNING|COMPLETING|CONFIGURING|SUSPENDED'
}

# slurmdbd can lag behind squeue: the job leaves the queue before its accounting
# record lands, and an empty state would read as "not COMPLETED" and fail a healthy
# run. Settle for up to 60s before believing the answer.
job_state() {
    local s i
    for ((i = 0; i < 12; i++)); do
        s=$(sacct -j "$1" -X -n -o State 2>/dev/null | head -n1 | tr -d ' ')
        [[ -n "${s}" ]] && { echo "${s}"; return 0; }
        sleep 5
    done
    echo "UNKNOWN_SACCT_LAG"
}

wait_for_jobs() {  # $@ = jobids; 0 = all left the queue, 1 = hit the ceiling
    local elapsed=0 id active
    while true; do
        active=0
        for id in "$@"; do job_active "${id}" && active=1; done
        [[ ${active} -eq 0 ]] && return 0
        if [[ ${elapsed} -ge ${MAX_WAIT_SECONDS} ]]; then
            echo "[run_dist_muon_bench] FAIL: still queued/running after ${MAX_WAIT_SECONDS}s" >&2
            scancel "$@" 2>/dev/null
            return 1
        fi
        sleep "${POLL_SECONDS}"
        elapsed=$((elapsed + POLL_SECONDS))
    done
}

echo "[run_dist_muon_bench] submitting bench_ns_gtp.sbatch (dense/GTP=64, NVLink)..."
gtp_id=$(submit_axis tools/muon_analysis/bench_ns_gtp.sbatch) || {
    echo "[run_dist_muon_bench] FAIL: gtp submission rejected" >&2; exit 1; }
submitted_ids+=("${gtp_id}")

echo "[run_dist_muon_bench] submitting bench_ns_egtp.sbatch (expert/EGTP=2, network)..."
egtp_id=$(submit_axis tools/muon_analysis/bench_ns_egtp.sbatch) || {
    echo "[run_dist_muon_bench] FAIL: egtp submission rejected" >&2; scancel "${gtp_id}"; exit 1; }
submitted_ids+=("${egtp_id}")

echo "[run_dist_muon_bench] submitted gtp=${gtp_id} egtp=${egtp_id}; waiting..."
wait_for_jobs "${gtp_id}" "${egtp_id}" || exit 1

# NODE_FAIL is an infra fault, not a result -- observed on 1 of 3 jobs on this
# partition. Resubmit that axis exactly once so a bad node is not scored as a failed
# optimization attempt. Never retry FAILED/CANCELLED/TIMEOUT: those are real.
retry_ids=()
if [[ "$(job_state "${gtp_id}")" == "NODE_FAIL" ]]; then
    echo "[run_dist_muon_bench] gtp ${gtp_id} hit NODE_FAIL -- resubmitting once" >&2
    new_id=$(submit_axis tools/muon_analysis/bench_ns_gtp.sbatch) || {
        echo "[run_dist_muon_bench] FAIL: gtp resubmission rejected" >&2; exit 1; }
    gtp_id="${new_id}"; submitted_ids+=("${gtp_id}"); retry_ids+=("${gtp_id}")
fi
if [[ "$(job_state "${egtp_id}")" == "NODE_FAIL" ]]; then
    echo "[run_dist_muon_bench] egtp ${egtp_id} hit NODE_FAIL -- resubmitting once" >&2
    new_id=$(submit_axis tools/muon_analysis/bench_ns_egtp.sbatch) || {
        echo "[run_dist_muon_bench] FAIL: egtp resubmission rejected" >&2; exit 1; }
    egtp_id="${new_id}"; submitted_ids+=("${egtp_id}"); retry_ids+=("${egtp_id}")
fi
if [[ ${#retry_ids[@]} -gt 0 ]]; then
    echo "[run_dist_muon_bench] waiting on retry ${retry_ids[*]} (queue wait restarts)..."
    wait_for_jobs "${retry_ids[@]}" || exit 1
fi
trap - INT TERM

gtp_state=$(job_state "${gtp_id}")
egtp_state=$(job_state "${egtp_id}")

if [[ "${gtp_state}" != "COMPLETED" || "${egtp_state}" != "COMPLETED" ]]; then
    echo "[run_dist_muon_bench] FAIL: gtp(${gtp_id})=${gtp_state} egtp(${egtp_id})=${egtp_state}" >&2
    # Pair each dir with its own axis id, so a log is not tailed once per directory.
    for pair in "${GTP_LOG_DIR}:${gtp_id}" "${EGTP_LOG_DIR}:${egtp_id}"; do
        for f in "${pair%%:*}"/*_"${pair##*:}"_*.log; do
            [[ -f "${f}" ]] && { echo "--- tail ${f} ---" >&2; tail -n 30 "${f}" >&2; }
        done
    done
    exit 1
fi

# Locate each log by its jobid (the sbatch --output pattern is %x_%j_${DATETIME}.log),
# which is exact -- unlike picking the newest file, which races concurrent runs.
gtp_log=$(ls -1 "${GTP_LOG_DIR}"/*_"${gtp_id}"_*.log 2>/dev/null | head -n1)
egtp_log=$(ls -1 "${EGTP_LOG_DIR}"/*_"${egtp_id}"_*.log 2>/dev/null | head -n1)

if [[ ! -f "${gtp_log}" || ! -f "${egtp_log}" ]]; then
    echo "[run_dist_muon_bench] FAIL: log not found (gtp='${gtp_log}' egtp='${egtp_log}')" >&2
    echo "  looked in ${GTP_LOG_DIR} and ${EGTP_LOG_DIR}" >&2
    exit 1
fi

# Guard the measurement: a run with syrk silently off is a different workload, and a
# fast-but-wrong number is worse than no number.
for f in "${gtp_log}" "${egtp_log}"; do
    if ! grep -q "use_syrk=True" "${f}"; then
        echo "[run_dist_muon_bench] FAIL: --use-syrk did not take effect in ${f}" >&2
        exit 1
    fi
done

parse_ms()   { grep -oP 'fastest step: \S+ \(\K[0-9.]+(?= ms\))' "$1" | tail -n1; }
parse_mode() { grep -oP 'fastest step: \K\S+' "$1" | tail -n1; }

gtp_ms=$(parse_ms "${gtp_log}")
egtp_ms=$(parse_ms "${egtp_log}")

if [[ -z "${gtp_ms}" || -z "${egtp_ms}" ]]; then
    echo "[run_dist_muon_bench] FAIL: could not parse 'fastest step' from logs" >&2
    echo "  gtp log:  ${gtp_log}" >&2
    echo "  egtp log: ${egtp_log}" >&2
    exit 1
fi

total_ms=$(python3 -c "print(f'{${gtp_ms} + ${egtp_ms}:.3f}')")

echo "[run_dist_muon_bench] gtp  fastest step: ${gtp_ms} ms ($(parse_mode "${gtp_log}"))  log: ${gtp_log}"
echo "[run_dist_muon_bench] egtp fastest step: ${egtp_ms} ms ($(parse_mode "${egtp_log}"))  log: ${egtp_log}"
echo "[run_dist_muon_bench] TOTAL_STEP_MS=${total_ms}"
