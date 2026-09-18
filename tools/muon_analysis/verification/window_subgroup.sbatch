#!/bin/bash

# Cheap functional + timing window for the `gtp-subgroup-dup-16384` candidate.
#
# 2 nodes / 8 ranks instead of the 16 nodes the scored GTP job needs, so it schedules in
# minutes rather than hours. It answers the two questions that must be settled before a
# 16-node job is spent:
#
#   1. FUNCTIONAL -- does newton_schulz_tp_subgroup return, for every rank and every
#      matrix, exactly what the per-matrix `duplicated` path returns, at both an even
#      (g=4 -> k=2, 3+3 matrices) and an UNEVEN (g=2 -> k=4, 2+2+1+1) deal? And does it
#      still match the frozen Phase-0 reference tensor?
#   2. TIMING -- does the harness run end to end with --set-timing and the new policy, and
#      does the subgroup arm move in the expected direction at a group size where it
#      cannot possibly pay off as much as at 64?
#
# The timing half here is NOT a result. At GTP=8 the shapes, the shard sizes and the
# redundancy are all different from the modelled workload; only the 16-node job scores.

#SBATCH -p gb300
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH -t 00:25:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --segment=2
#SBATCH --job-name=coreai_dlalgo_llm-dist_muon.window_subgroup
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
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/verification/phase-2-gtp-subgroup-dup-16384}"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/training.sqsh}"

LOGS_DIR="${ROOT_DIR}/runs/verification/phase-2-gtp-subgroup-dup-16384/logs"
mkdir -p "${LOGS_DIR}"
DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29517

# g must divide the 8-rank world and be smaller than it. 4 -> k=2 (even 6/2), 2 -> k=4
# (uneven 2,2,1,1). Both are exercised by the equivalence half.
SUBGROUP_SIZES="2 4"

equiv_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${VERIF_DIR}/check_equivalence_subgroup.py \
    --out ${OUT_DIR}/equivalence.csv \
    --reference-dir ${REF_DIR} \
    --shapes 8192x16384 36864x8192 \
    --subgroup-sizes ${SUBGROUP_SIZES} \
    --count 6 --steps 5 --coefficient-type polar_express"

bench_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${REPO_DIR}/tools/muon_analysis/bench_ns_strategies.py \
    --group gtp --modelled-world-size 128 \
    --tp 1 --gtp 8 --ep 64 --etp 1 --egtp 2 \
    --num-ns-steps 5 --coefficient-type polar_express \
    --iters 5 --warmup 2 \
    --set-timing --batch-chunk 4 --subgroup-sizes ${SUBGROUP_SIZES} --use-syrk"

srun -l --mpi=none \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/equiv_%j_${DATETIME}.log" \
    bash -c "${equiv_cmd}"

srun -l --mpi=none \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/bench_%j_${DATETIME}.log" \
    bash -c "${bench_cmd}"
