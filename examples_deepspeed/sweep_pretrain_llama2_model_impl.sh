#!/bin/bash
# bash ./examples_deepspeed/sweep_pretrain_llama2_model_impl.sh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RUN_SCRIPT="${SCRIPT_DIR}/pretrain_llama2_distributed.sh"

MODEL_IMPLS=${MODEL_IMPLS:-"baseline cola"}  # baseline=FR
DP_VALUES=${DP_VALUES:-"1 4"}
ZERO_STAGES=${ZERO_STAGES:-"0 1"}
DATE_TAG=${DATE_TAG:-$(date +%m%d)}
LOG_DIR=${LOG_DIR:-"${REPO_ROOT}/.logging/${DATE_TAG}"}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "${LOG_DIR}"

echo "[sweep] Model impls: ${MODEL_IMPLS}"
echo "[sweep] DP values: ${DP_VALUES}"
echo "[sweep] ZeRO stages: ${ZERO_STAGES}"
echo "[sweep] Log dir: ${LOG_DIR}"
echo "[sweep] Dry run: ${DRY_RUN}"

TOTAL_RUNS=0

for DP in ${DP_VALUES}; do
  for ZERO_STAGE in ${ZERO_STAGES}; do
    OFFLOAD_VALUES="0 1"
    if [ "${ZERO_STAGE}" -eq 0 ]; then
      OFFLOAD_VALUES="0"
    fi

    for OFFLOAD_OPTIMIZER in ${OFFLOAD_VALUES}; do
      for MODEL_IMPL in ${MODEL_IMPLS}; do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))

        MODEL_TAG="FR"
        if [ "${MODEL_IMPL}" = "cola" ]; then
          MODEL_TAG="CoLA"
        fi

        OFFLOAD_TAG="nooffload"
        if [ "${OFFLOAD_OPTIMIZER}" -eq 1 ]; then
          OFFLOAD_TAG="cpuoffload"
        fi

        LOG_FILE="${LOG_DIR}/${DATE_TAG}_${MODEL_TAG}_llama1B_DP${DP}_TP${TP:-1}_PP${PP:-1}_ZeRO${ZERO_STAGE}_seq${SEQ_LENGTH:-1024}_mbz${MICRO_BATCH_SIZE:-1}_gbz${GLOBAL_BATCH_SIZE:-4}_${OFFLOAD_TAG}.log"
        echo "[sweep][${TOTAL_RUNS}] MODEL_IMPL=${MODEL_IMPL} DP=${DP} ZeRO=${ZERO_STAGE} OFFLOAD=${OFFLOAD_OPTIMIZER}"
        echo "[sweep][${TOTAL_RUNS}] Logging to ${LOG_FILE}"

        if [ "${DRY_RUN}" -eq 1 ]; then
          continue
        fi

        (
          cd "${REPO_ROOT}"
          GPUS_PER_NODE="${DP}" \
          MODEL_IMPL="${MODEL_IMPL}" \
          ZERO_STAGE="${ZERO_STAGE}" \
          OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER}" \
          CPU_OPTIMIZER="${OFFLOAD_OPTIMIZER}" \
          bash "${RUN_SCRIPT}"
        ) 2>&1 | tee "${LOG_FILE}"
      done
    done
  done
done

echo "[sweep] Completed ${TOTAL_RUNS} runs."
[ "${DRY_RUN}" -eq 1 ] || python3 "${SCRIPT_DIR}/summarize_pretrain_llama2_sweep.py" --log-dir "${LOG_DIR}" --date-tag "${DATE_TAG}" --strict-12
