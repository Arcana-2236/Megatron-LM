#!/bin/bash

# Cheap functional + timing window for stage-4 / phase-3 `gtp-wire-bf16`.
#
# 1 node / 4 ranks instead of the 16 nodes the scored GTP job needs, so it schedules in
# minutes. It settles three things before a 16-node job is spent:
#
#   1. FUNCTIONAL -- the bf16 OUTPUT leg must be BITWISE against the fp32-wire pipelined
#      arm it suffixes (the preflight measured the region output to be 100 %
#      bf16-representable, so anything else is a data-movement bug); the bf16 INPUT leg
#      must sit inside atol = 1e-3 against that same arm, the duplicated lineage and the
#      frozen phase-0 reference, at the REAL gathered shapes.
#   2. GUARDS -- a wire mode on the monolithic exchange, and an unknown mode, must RAISE.
#      A silent fp32 fallback would let a bf16 arm be scored while running the fp32 wire.
#   3. WIRE-UP -- the harness leg runs bench_ns_strategies.py itself with
#      --fused-wire-modes at reduced --iters, to prove the extra arm columns, the
#      fused-arm argmin and the all-arm table wire up before the 16-node job pays for them.
#
# The timing half is NOT a result: at GTP=4 the group size, shard sizes and redundancy all
# differ from the modelled workload, and the wire is intra-node NVLink at 4 ranks rather
# than the 64-rank domain. Only the 16-node job scores.

#SBATCH -p gb300
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 00:40:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=coreai_dlalgo_llm-dist_muon.window_fused_wire
#SBATCH --comment='{"APS":{}}'

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
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/verification/phase-3-gtp-wire-bf16}"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/training.sqsh}"
LOGS_DIR="${OUT_DIR}/logs"
mkdir -p "${LOGS_DIR}"
DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29537

SUBGROUP_SIZES=${WINDOW_SUBGROUP_SIZES:-"1 2"}
PIPE_CHUNKS=${WINDOW_PIPE_CHUNKS:-"2"}
WIRE_MODES=${WINDOW_WIRE_MODES:-"bf16 bf16in bf16out"}

check_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${VERIF_DIR}/check_fused_wire.py \
    --out ${OUT_DIR}/equivalence.csv \
    --timing-out ${OUT_DIR}/window_timing.json \
    --reference-dir ${REF_DIR} \
    --shapes 10240x8192 8192x8192 8192x16384 \
    --counts 3 5 2 \
    --subgroup-sizes ${SUBGROUP_SIZES} \
    --pipe-chunks ${PIPE_CHUNKS} \
    --wire-modes ${WIRE_MODES} \
    --iters 5 --warmup 2 \
    --steps 5 --coefficient-type polar_express"

bench_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${REPO_DIR}/tools/muon_analysis/bench_ns_strategies.py \
    --group gtp --modelled-world-size 128 \
    --tp 1 --gtp 4 --ep 64 --etp 1 --egtp 2 \
    --num-ns-steps 5 --coefficient-type polar_express \
    --iters 2 --warmup 1 \
    --set-timing --batch-chunk 4 \
    --subgroup-sizes ${SUBGROUP_SIZES} \
    --subgroup-all-shapes --batch-replicated-shapes --fuse-shape-exchanges \
    --pipeline-fused-exchange --fused-pipe-chunks ${PIPE_CHUNKS} \
    --fused-wire-modes ${WIRE_MODES} \
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
