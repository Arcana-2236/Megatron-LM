#!/bin/bash

# Cheap functional + timing window for stage-3 / phase-3 `egtp-exchange-restructure`.
#
# 1 node, 2 ranks -- the REAL EGTP degree (egtp=2), so this is not a scaled-down window on
# the parallelism, only on the iteration count. NCCL P2P/SHM/NVLS are disabled to match
# bench_ns_egtp.sbatch's network-path emulation, which is what makes the wire term this
# candidate overlaps the same wire the scored job measures.
#
# It answers the three questions that must be settled before the scored job is spent:
#   1. FUNCTIONAL -- is the restructured exchange bit-for-bit invisible? Block ownership
#      and the async per-chunk exchanges are internal, so the answer must be yes against
#      BOTH the accepted arm it substitutes and the per-matrix duplicated lineage.
#   2. GUARDS -- are the preconditions hard errors rather than a silent fallback to the
#      monolithic path? A silent fallback would let the arm be scored while running the
#      very code it claims to replace.
#   3. TIMING -- at the real region count (192 matrices, 96 owned per rank) and the real
#      chunking, does the pipelined arm move in the expected direction, and at which
#      pipeline depth?
#
# The harness leg then runs bench_ns_strategies.py itself with --pipelined-exchange at
# reduced --iters, to prove the new column, policy, audit and substitution assert all wire
# up before the scored job pays for them.

#SBATCH -p gb300
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=2
#SBATCH --job-name=coreai_dlalgo_llm-dist_muon.window_pipe_exchange
#SBATCH --comment='{"APS":{}}'

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

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
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/verification/phase-3-egtp-exchange-restructure}"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/training.sqsh}"
LOGS_DIR="${OUT_DIR}/logs"
mkdir -p "${LOGS_DIR}"
DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29523

# PYTHONPATH so `kernels.fused_ns` (the accepted compute path both arms run) is importable.
export PYTHONPATH="${MA_DIR}:${PYTHONPATH}"

check_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${VERIF_DIR}/check_pipelined_exchange.py \
    --out ${OUT_DIR}/equivalence.csv \
    --timing-out ${OUT_DIR}/window_timing.json \
    --reference-dir ${REF_DIR} \
    --shapes 5120x2048 2048x5120 \
    --counts ${WINDOW_COUNTS:-2 4} \
    --timing-count ${WINDOW_TIMING_COUNT:-192} \
    --batch-chunk 64 \
    --pipe-chunks ${WINDOW_PIPE_CHUNKS:-2 4 8} \
    --fused \
    --iters 5 --warmup 2 \
    --steps 5 --coefficient-type polar_express"

bench_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${REPO_DIR}/tools/muon_analysis/bench_ns_strategies.py \
    --group egtp --modelled-world-size 128 \
    --tp 1 --gtp 64 --ep 64 --etp 1 --egtp 2 \
    --num-ns-steps 5 --coefficient-type polar_express \
    --iters 5 --warmup 2 \
    --set-timing --batch-chunk 64 \
    --subgroup-sizes 1 --subgroup-batched \
    --fused-ns-kernel --pipelined-exchange --pipe-chunks ${EGTP_PIPE_CHUNKS:-4} \
    --use-syrk"

srun -l --mpi=none \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/check_%j_${DATETIME}.log" \
    bash -c "${check_cmd}"

if [[ "${RUN_BENCH:-1}" == "1" ]]; then
srun -l --mpi=none \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/bench_%j_${DATETIME}.log" \
    bash -c "${bench_cmd}"
fi
