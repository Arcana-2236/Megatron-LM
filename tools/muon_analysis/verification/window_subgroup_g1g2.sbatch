#!/bin/bash

# Cheap functional + timing window for stage-5 / phase-1 `gtp-subgroup-size-g1g2`.
#
# 1 node / 4 ranks instead of the 16 nodes the scored GTP job needs, so it schedules in
# minutes. At world 4 the legal duplication-group sizes are exactly g in {1, 2} (g = 4 is
# `groups == 1`), which is precisely the pair of columns this candidate adds -- so this
# window covers 100 % of the new surface and none of the banked one.
#
# It settles three things before a 16-node job is spent:
#
#   1. FUNCTIONAL -- g-INVARIANCE. fuse_sub_g2<suffix> must be BITWISE against
#      fuse_sub_g1<suffix> and both must be BITWISE against the per-matrix `duplicated`
#      lineage, on every (balanced, pipe) suffix. g decides only which subgroup owns a
#      matrix and how the two all_to_all splits are cut; it must not touch the arithmetic.
#   2. GUARDS -- g == world and a g that does not divide world must RAISE. A silent
#      fallback would let a scored g-column time a different design under this name.
#   3. WIRE-UP -- the harness leg runs bench_ns_strategies.py itself at --gtp 4 with the
#      NEW default `--subgroup-sizes 1 2 4 8 16` (filtered to what is legal at world 4 by
#      the same code the 64-rank job uses), at reduced --iters, to prove the extra columns,
#      the per-shape subgroup argmin and the fused all-arm table wire up before the 16-node
#      job pays for them.
#
# The timing half is NOT a result: at GTP=4 the shard sizes, the redundancy factor and the
# transport (intra-node NVLink at 4 ranks vs the 64-rank domain) all differ. Only the
# 16-node job scores.

#SBATCH -p gb300
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 00:40:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=coreai_dlalgo_llm-dist_muon.window_subgroup_g1g2
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
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/verification/phase-1-gtp-subgroup-size-g1g2}"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/training.sqsh}"
LOGS_DIR="${OUT_DIR}/logs"
mkdir -p "${LOGS_DIR}"
DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29541

# Counts chosen so the deal is UNEVEN and, at g = 1 (k = 4 groups), at least one shape has
# fewer matrices than groups (10240x8192 at 3 over 4 groups leaves one group idle). That is
# the exact regime the 64-rank job enters at small g -- the region floors at one whole
# matrix -- so the window exercises it rather than hiding it behind a full deal.
SUBGROUP_SIZES=${WINDOW_SUBGROUP_SIZES:-"1 2"}
PIPE_CHUNKS=${WINDOW_PIPE_CHUNKS:-"0 2 4"}

check_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${VERIF_DIR}/check_subgroup_g1g2.py \
    --out ${OUT_DIR}/equivalence.csv \
    --timing-out ${OUT_DIR}/window_timing.json \
    --reference-dir ${REF_DIR} \
    --shapes 10240x8192 8192x8192 8192x16384 \
    --counts 3 5 2 \
    --subgroup-sizes ${SUBGROUP_SIZES} \
    --pipe-chunks ${PIPE_CHUNKS} \
    --iters 5 --warmup 2 \
    --steps 5 --coefficient-type polar_express"

# The harness leg SOURCES the new default out of bench_ns_gtp.sbatch rather than restating
# it, so a typo in the banked default fails here and not on 16 nodes. bench_ns_strategies.py
# hard-asserts that every g divides the group and is smaller than it (line ~2028), so at
# --gtp 4 the default must be filtered to what is legal at world 4 -- {1, 2}, i.e. exactly
# the two NEW columns. The filter also PROVES the two new values are in the banked default:
# it fails loudly if either is missing.
BENCH_DEFAULT=$(sed -n 's/^SUBGROUP_SIZES=\${SUBGROUP_SIZES:-"\(.*\)"}$/\1/p' \
    "${REPO_DIR}/tools/muon_analysis/bench_ns_gtp.sbatch")
echo "[window] bench_ns_gtp.sbatch SUBGROUP_SIZES default = '${BENCH_DEFAULT}'"
BENCH_SUBGROUP_SIZES=""
for g in ${BENCH_DEFAULT}; do
    if (( 4 % g == 0 && g < 4 )); then BENCH_SUBGROUP_SIZES="${BENCH_SUBGROUP_SIZES}${g} "; fi
done
echo "[window] legal at world 4: '${BENCH_SUBGROUP_SIZES}'"
if [[ "${BENCH_SUBGROUP_SIZES}" != "1 2 " ]]; then
    echo "[window] FAIL: banked default does not contain both new columns g=1 and g=2" >&2
    exit 1
fi

bench_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${REPO_DIR}/tools/muon_analysis/bench_ns_strategies.py \
    --group gtp --modelled-world-size 128 \
    --tp 1 --gtp 4 --ep 64 --etp 1 --egtp 2 \
    --num-ns-steps 5 --coefficient-type polar_express \
    --iters 2 --warmup 1 \
    --set-timing --batch-chunk 4 \
    --subgroup-sizes ${BENCH_SUBGROUP_SIZES} \
    --subgroup-all-shapes --batch-replicated-shapes --batch-replicated-fused \
    --fuse-shape-exchanges \
    --pipeline-fused-exchange --fused-pipe-chunks 2 4 \
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
