#!/bin/bash

# Cheap functional + timing window for `egtp-ns-step-fusion` (stage-3 rank 2).
# ONE node, ONE rank: the candidate is compute-only, so no collective is needed to settle
# either half and the job schedules in minutes.

#SBATCH -p gb300
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 00:25:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=coreai_dlalgo_llm-dist_muon.window_fused_ns
#SBATCH --comment='{"APS":{}}'

# Layout, derived -- never a hardcoded absolute path (see ../README.md). ROOT_DIR is the
# directory CONTAINING the Megatron-LM checkout; it defaults to the parent of wherever
# sbatch was invoked, which is right when submitting from the repo root.
ROOT_DIR="${ROOT_DIR:-$(cd "${SLURM_SUBMIT_DIR:-${PWD}}/.." && pwd)}"
REPO_DIR="${ROOT_DIR}/Megatron-LM"
MA_DIR="${REPO_DIR}/tools/muon_analysis"
VERIF_DIR="${MA_DIR}/verification"
# Frozen phase-0 reference tensors. Not in git (8.3 GB); regenerate with
# capture_reference.sbatch, which seeds per shape from --seed (default 1234).
REF_DIR="${REF_DIR:-${ROOT_DIR}/runs/verification/reference_outputs}"
if [[ ! -f "${MA_DIR}/bench_ns_strategies.py" ]]; then
    echo "ERROR: no benchmark at ${MA_DIR}; set ROOT_DIR to the dir containing Megatron-LM" >&2
    exit 1
fi
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/verification/phase-2-egtp-ns-step-fusion}"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/training.sqsh}"
LOGS_DIR="${OUT_DIR}/logs"
mkdir -p "${LOGS_DIR}"
DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

cmd="python -u ${VERIF_DIR}/check_fused_ns.py \
    --shapes 5120x2048 2048x5120 \
    --count 96 --batch-chunk 64 --steps 5 \
    --coefficient-type polar_express \
    --iters 20 --warmup 5 --atol 1e-3 --rtol 1e-3 \
    --out-csv ${OUT_DIR}/equivalence.csv \
    --out-json ${OUT_DIR}/perf_window.json"

srun -l --mpi=none \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/window_%j_${DATETIME}.log" \
    bash -c "${cmd}"
