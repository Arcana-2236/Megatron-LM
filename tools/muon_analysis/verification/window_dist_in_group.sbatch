#!/bin/bash

# Cheap functional + timing window for stage-5 / phase-2 `gtp-distribute-in-group-g2`.
#
# 1 node / 4 ranks instead of the 16 nodes the scored GTP job needs, so it schedules in
# minutes. At world 4 the only legal distribute-in-group column is g = 2 (g = 1 has nothing
# to distribute inside a one-rank subgroup, g = 4 is `groups == 1`), which is exactly the
# column this candidate adds to the 64-rank job -- so the window covers 100 % of the new
# surface and none of the banked one.
#
# It settles four things before a 16-node job is spent:
#
#   1. UNTOUCHED-IS-BITWISE. A shape that is WIDE once gathered is not distributed; under
#      the round-robin deal it must come back bit-for-bit identical to the replicated arm.
#   2. DISTRIBUTED-IS-IN-TOLERANCE. The tall shapes carry exactly the two deviations
#      `tp_mode="distributed"` already has (split Frobenius reduction, Gram summed over g
#      partials) and are gated at the spec's atol = rtol = 1e-3 against the per-matrix
#      `duplicated` lineage AND against the frozen phase-0 reference.
#   3. GUARDS. g = 1, a non-pipelined call, and an all-wide region must RAISE; a silent
#      fallback would let the replicated distribution be scored under this arm's name.
#   4. WIRE-UP. The harness leg runs bench_ns_strategies.py itself at --gtp 4 with the NEW
#      `--fused-dist-in-group-sizes` default sourced out of bench_ns_gtp.sbatch, at reduced
#      --iters, to prove the extra columns and the fused-arm argmin wire up before the
#      16-node job pays for them.
#
# The timing half is NOT a result: at GTP=4 the shard sizes, the deal and the transport all
# differ from the 64-rank region. Only the 16-node job scores.

#SBATCH -p gb300
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 00:40:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=coreai_dlalgo_llm-dist_muon.window_dist_in_group
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
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/verification/phase-2-gtp-distribute-in-group-g2}"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/training.sqsh}"
LOGS_DIR="${OUT_DIR}/logs"
mkdir -p "${LOGS_DIR}"
DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29543

# Two TALL shapes (distributed by the arm) and two WIDE ones (left replicated), with uneven
# counts, so the window exercises both halves of the split in one region -- including the
# 36864x8192 shape that owns the 64-rank region's LPT critical path.
SHAPES=${WINDOW_SHAPES:-"36864x8192 10240x8192 8192x8192 8192x16384"}
COUNTS=${WINDOW_COUNTS:-"2 3 4 2"}
PIPE_CHUNKS=${WINDOW_PIPE_CHUNKS:-"2 4"}

check_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${VERIF_DIR}/check_dist_in_group.py \
    --out ${OUT_DIR}/equivalence.csv \
    --timing-out ${OUT_DIR}/window_timing.json \
    --reference-dir ${REF_DIR} \
    --shapes ${SHAPES} \
    --counts ${COUNTS} \
    --subgroup-size 2 \
    --pipe-chunks ${PIPE_CHUNKS} \
    --iters 5 --warmup 2 \
    --steps 5 --coefficient-type polar_express"

# The harness leg SOURCES the new default out of bench_ns_gtp.sbatch rather than restating
# it, so a typo in the banked default fails here and not on 16 nodes. g must divide the
# group and leave more than one group, so at --gtp 4 the default filters to {2}.
BENCH_DEFAULT=$(sed -n 's/^FUSED_DIST_IN_GROUP_SIZES=\${FUSED_DIST_IN_GROUP_SIZES-\(.*\)}$/\1/p' \
    "${REPO_DIR}/tools/muon_analysis/bench_ns_gtp.sbatch")
echo "[window] bench_ns_gtp.sbatch FUSED_DIST_IN_GROUP_SIZES default = '${BENCH_DEFAULT}'"
BENCH_DIST_SIZES=""
for g in ${BENCH_DEFAULT}; do
    if (( 4 % g == 0 && g >= 2 && 4 / g > 1 )); then BENCH_DIST_SIZES="${BENCH_DIST_SIZES}${g} "; fi
done
echo "[window] legal at world 4: '${BENCH_DIST_SIZES}'"
if [[ "${BENCH_DIST_SIZES}" != "2 " ]]; then
    echo "[window] FAIL: banked default does not contain the new column g=2" >&2
    exit 1
fi

bench_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${REPO_DIR}/tools/muon_analysis/bench_ns_strategies.py \
    --group gtp --modelled-world-size 128 \
    --tp 1 --gtp 4 --ep 64 --etp 1 --egtp 2 \
    --num-ns-steps 5 --coefficient-type polar_express \
    --iters 2 --warmup 1 \
    --set-timing --batch-chunk 4 \
    --subgroup-sizes 1 2 \
    --subgroup-all-shapes --batch-replicated-shapes --batch-replicated-fused \
    --fuse-shape-exchanges \
    --pipeline-fused-exchange --fused-pipe-chunks 2 4 \
    --fused-dist-in-group-sizes ${BENCH_DIST_SIZES} \
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
