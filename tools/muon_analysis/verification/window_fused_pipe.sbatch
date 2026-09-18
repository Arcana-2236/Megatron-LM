#!/bin/bash

# Cheap functional + timing window for stage-4 / phase-2 `gtp-exchange-pipeline`.
#
# 1 node / 4 ranks instead of the 16 nodes the scored GTP job needs, so it schedules in
# minutes rather than hours. It settles the three questions that must be answered before a
# 16-node job is spent:
#
#   1. FUNCTIONAL -- at --steps 0 the restructured region must be BITWISE against the
#      monolithic one (both collapse to the same elementwise normalize -> bf16 -> fp32, so
#      only data movement can differ); at the real 5 steps it must sit inside the spec's
#      atol = rtol = 1e-3 against the arm it substitutes, the duplicated lineage and the
#      frozen phase-0 reference. Run under BOTH deals, at UNEQUAL per-shape counts (3/5/2,
#      so the chunk windows cut the per-shape splits at different places) and at two
#      pipeline depths.
#   2. GUARDS -- the preconditions are hard errors, not a silent fallback to the monolithic
#      path. A silent fallback would let the arm be scored while running the code it claims
#      to replace.
#   3. WIRE-UP -- the harness leg runs bench_ns_strategies.py itself with
#      --pipeline-fused-exchange at reduced --iters, to prove the extra arm columns, the
#      fused-arm argmin, the per-profile audit and the all-arm table all wire up before the
#      16-node job pays for them.
#
# The timing half here is NOT a result. At GTP=4 the group size, the shard sizes and the
# redundancy are all different from the modelled workload; only the 16-node job scores.

#SBATCH -p gb300
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 00:40:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=coreai_dlalgo_llm-dist_muon.window_fused_pipe
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
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/verification/phase-2-gtp-exchange-pipeline}"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/training.sqsh}"
LOGS_DIR="${OUT_DIR}/logs"
mkdir -p "${LOGS_DIR}"
DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29533

# g must divide the 4-rank world and be smaller than it: 2 -> k=2, 1 -> k=4 (owner-computes).
SUBGROUP_SIZES=${WINDOW_SUBGROUP_SIZES:-"1 2"}
PIPE_CHUNKS=${WINDOW_PIPE_CHUNKS:-"2 4"}

check_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${VERIF_DIR}/check_fused_pipe.py \
    --out ${OUT_DIR}/equivalence.csv \
    --timing-out ${OUT_DIR}/window_timing.json \
    --reference-dir ${REF_DIR} \
    --shapes 10240x8192 8192x8192 8192x16384 \
    --counts 3 5 2 \
    --subgroup-sizes ${SUBGROUP_SIZES} \
    --pipe-chunks ${PIPE_CHUNKS} \
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
