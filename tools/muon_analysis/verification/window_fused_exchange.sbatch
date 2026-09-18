#!/bin/bash

# Cheap functional + timing window for stage-3 / phase-4 `gtp-fuse-shape-exchanges`.
#
# 1 node / 4 ranks instead of the 16 nodes the scored GTP job needs, so it schedules in
# minutes rather than hours. It answers the three questions that must be settled before a
# 16-node job is spent:
#
#   1. FUNCTIONAL -- does the fused cross-shape region return, for every rank, every shape
#      and every matrix, exactly what looping newton_schulz_tp_subgroup once per shape
#      returns? Under BOTH deals (round-robin, which must be bitwise trivially, and the
#      pooled LPT deal, which must be bitwise because a matrix's arithmetic does not
#      depend on which subgroup runs it), at UNEQUAL per-shape counts (3 / 5 / 2, so the
#      per-shape splits inside the one exchange differ in size) and at an even (g=2 ->
#      k=2) and a fully owner-computes (g=1 -> k=4) deal.
#   2. GUARDS -- are the preconditions hard errors rather than a silent fallback to the
#      per-shape path? A silent fallback would let the arm be scored while running the
#      very code it claims to replace.
#   3. TIMING -- at the real owned counts, does the fused region move against the SUM of
#      the per-shape regions it replaces, and does the movement come from the exchange
#      (round-robin arm) or from the pooled deal ('_bal' arm)?
#
# The harness leg then runs bench_ns_strategies.py itself with --fuse-shape-exchanges at
# reduced --iters, to prove the per-profile fused rows, the scored policy, the audit and
# the substitution asserts all wire up before the 16-node job pays for them.
#
# The timing half here is NOT a result. At GTP=4 the group size, the shard sizes and the
# redundancy are all different from the modelled workload; only the 16-node job scores.

#SBATCH -p gb300
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=coreai_dlalgo_llm-dist_muon.window_fused_exchange
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
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/verification/phase-4-gtp-fuse-shape-exchanges}"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/training.sqsh}"
LOGS_DIR="${OUT_DIR}/logs"
mkdir -p "${LOGS_DIR}"
DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29531

# g must divide the 4-rank world and be smaller than it: 2 -> k=2, 1 -> k=4 (owner-computes).
SUBGROUP_SIZES=${WINDOW_SUBGROUP_SIZES:-"1 2"}

check_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${VERIF_DIR}/check_fused_exchange.py \
    --out ${OUT_DIR}/equivalence.csv \
    --timing-out ${OUT_DIR}/window_timing.json \
    --reference-dir ${REF_DIR} \
    --shapes 10240x8192 8192x8192 8192x16384 \
    --counts 3 5 2 \
    --subgroup-sizes ${SUBGROUP_SIZES} \
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
