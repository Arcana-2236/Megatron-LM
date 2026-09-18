#!/bin/bash

# Cheap functional + timing window for `gtp-batch-set-in-argmin` (stage-3 rank 1).
#
# 2 nodes / 8 ranks instead of the 16 nodes the scored GTP job needs, so it schedules in
# minutes rather than hours. It settles the two things that must hold before a 16-node job
# is spent on this candidate:
#
#   1. FUNCTIONAL -- on the three shard_count == 1 (replicated) GTP shapes, does the
#      batched call agree with the per-matrix loop it would replace, and with the frozen
#      Phase-0 reference, at the spec.toml gate atol = rtol = 1e-3? Two batch_chunk values
#      are exercised: 4 (the value the scored sbatch passes, and one that does NOT divide
#      count=6, so the short final chunk is covered) and 6 (the whole group at once).
#   2. TIMING/SCOPE -- does the harness run end to end with --batch-replicated-shapes, does
#      the new `per_shape_sub_all_batch_set` column appear, and does the per-shape audit
#      show the widened argmin firing on shard_count == 1 shapes AND NOWHERE ELSE? The
#      in-script assert would abort the job if `duplicated_batch` were ever banked on a
#      sharded shape, so a scoping regression cannot be silently timed.
#
# The timing half is NOT a result. At GTP=8 the shapes, shard sizes and redundancy all
# differ from the modelled workload; only the 16-node job scores.

#SBATCH -p gb300
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH -t 00:35:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --segment=2
#SBATCH --job-name=coreai_dlalgo_llm-dist_muon.window_batch_replicated
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
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/verification/phase-1-gtp-batch-set-in-argmin}"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/training.sqsh}"

LOGS_DIR="${OUT_DIR}/logs"
mkdir -p "${LOGS_DIR}"
DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29521

# The three shapes build_model_matrices emits with shard_degree 1 -- the router
# (512x8192) and the two unsharded MoE latent projections (2048x8192, 8192x2048). These
# and only these are the shapes this candidate's argmin widens over. The sharded shapes
# are deliberately absent: on them the batched column is a different computation and the
# harness refuses to bank it.
SHAPES="8192x2048 2048x8192 512x8192"

# g must divide the 8-rank world and be smaller than it; carried over so the reference arm
# `per_shape_sub_all_set` this candidate is scored against is actually formed.
SUBGROUP_SIZES="2 4"

equiv_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${VERIF_DIR}/check_equivalence_batch_replicated.py \
    --out ${OUT_DIR}/equivalence.csv \
    --reference-dir ${REF_DIR} \
    --shapes ${SHAPES} \
    --batch-chunks 4 6 \
    --count 6 --steps 5 --coefficient-type polar_express"

bench_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${REPO_DIR}/tools/muon_analysis/bench_ns_strategies.py \
    --group gtp --modelled-world-size 128 \
    --tp 1 --gtp 8 --ep 64 --etp 1 --egtp 2 \
    --num-ns-steps 5 --coefficient-type polar_express \
    --iters 5 --warmup 2 \
    --set-timing --batch-chunk 4 --subgroup-sizes ${SUBGROUP_SIZES} \
    --subgroup-all-shapes --batch-replicated-shapes --use-syrk"

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
