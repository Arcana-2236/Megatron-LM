#!/bin/bash

# Benchmark Newton-Schulz distribution strategies over the GTP axis.
#
# GTP shards dense weights. At the production degree of 64 the group fits inside one
# GB200/GB300 NVLink domain, so NVLink is left enabled here: that is the transport a
# real run uses. On Lyris (4 GPUs/node) a 64-rank group needs 16 nodes, and --segment=16
# keeps them inside a single NVLink domain.
#
# GTP degree and node count must agree: nodes * 4 (GPUs/node) == GTP. To try a smaller
# degree first, override both, e.g.
#   sbatch --nodes=2 --segment=2 --export=ALL,GTP=8 tools/muon_analysis/bench_ns_gtp.sbatch

#SBATCH -p gb300
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH --mem=0
# No --gpus-per-node: Lyris rejects it as an invalid gres spec. --exclusive gives all
# 4 GPUs on the node.
#SBATCH --ntasks-per-node=4
#SBATCH --segment=16
#SBATCH --job-name=coreai_dlalgo_llm-dist_muon.bench_ns_gtp
#SBATCH --comment='{"APS":{}}'

export NCCL_DEBUG=WARN

GTP=${GTP:-64}

ROOT_DIR="/lustre/fsw/coreai_dlalgo_llm/zhengywang/dist_muon_proxy_ootb"
REPO_DIR="${ROOT_DIR}/Megatron-LM"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/training.sqsh}"

# TAG separates runs on different hardware, e.g. sbatch -p gb200 --export=ALL,TAG=gb200
NAME="runs/ns_bench/gtp${TAG:+_$TAG}"
LOGS_DIR="${ROOT_DIR}/${NAME}/logs"
mkdir -p "${LOGS_DIR}"

DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

# torchrun is not used, so the rendezvous env has to come from Slurm explicitly.
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29512

# USE_SYRK=1 forwards --use-syrk to newton_schulz_tp (Triton triangular kernel).
# sbatch --export=ALL,USE_SYRK=1 bench_ns_gtp.sbatch
SYRK_OPTION=""
if [[ "${USE_SYRK:-0}" == "1" ]]; then SYRK_OPTION="--use-syrk"; fi

# SUBGROUP_SIZES: duplication-group sizes g for the subgroup-duplication arms. Each must
# divide GTP and be smaller than it, so a reduced-degree smoke run must override this too,
# e.g. sbatch --nodes=2 --segment=2 --export=ALL,GTP=8,SUBGROUP_SIZES="2 4" ...
SUBGROUP_SIZES=${SUBGROUP_SIZES:-"4 8 16"}

# --subgroup-all-shapes: scores a SECOND subgroup policy, 'per_shape_sub_all_set'. It is
# the same substitution rule as the already-accepted 'per_shape_sub_set' but applied on
# every shape that has a subgroup column, including the four shard_count=64 shapes whose
# set-composed winner is 'distributed_set' (8192x10240, 36864x8192, 10240x8192, 8192x8192).
# Costs ZERO extra GPU time: the per-g columns those shapes need are already timed for the
# printed table; only the argmin that forms the policy widens. The shard_count == 1 shapes
# (8192x2048, 512x8192) have no subgroup column at all and are untouched, which is exactly
# the plan's scope -- subgrouping them would introduce a collective where there is none.
# Three arms are therefore printed from ONE job: 'per_shape_set' (B_set), 'per_shape_sub_set'
# (the accepted state) and 'per_shape_sub_all_set' (this candidate).
#
# --set-timing: this axis owns 12-27 copies of each dense shape, so subgroup duplication --
# which deals the owned SET across duplication groups of g < 64 -- is a property of the set
# and is unmeasurable under the per-matrix composition. Set composition times the whole
# owned group of a shape in one region and scores both arms from THIS job: the unmodified
# 'per_shape_set' reference and the 'per_shape_sub_set' candidate. The isolated per-matrix
# columns are still printed, unscored, for continuity with every earlier GTP number.
#
# --batch-chunk 4 (not the 64 the EGTP axis uses): the batched arm's all-gather buffer is
# chunk x 64 x the local shard, and the dense shards here are up to 576x8192 fp32, so 64
# would stage ~14.5 GB before Newton-Schulz workspace. 4 bounds it to ~4.8 GB. The batched
# arm is not this attempt's candidate; it is printed because --set-timing produces it.
# --batch-replicated-shapes: scores a THIRD set-composed policy,
# 'per_shape_sub_all_batch_set'. It is 'per_shape_sub_all_set' with the already-timed
# 'duplicated_batch' column added to the per-shape argmin, on the shard_count == 1
# (replicated) shapes ONLY -- on this axis 8192x2048 (24 owned) and 512x8192 (12 owned),
# which have no subgroup column at all and whose set-composed winner is a per-matrix loop
# of 24 / 12 serial Newton-Schulz chains. Costs ZERO extra GPU time: 'duplicated_batch' is
# already timed for every shape for the printed table; only the argmin widens.
# The shard_count == 1 gate is the safety property and is asserted in the script: at
# partition_dim is None the batched call short-circuits to plain newton_schulz, so the
# 'distributed' route -- whose distributed_normalize_p2 would share one Frobenius norm
# across a 3-D stack -- is never reached. Sharded shapes keep their existing arms.
options=" \
    --group gtp \
    --modelled-world-size 128 \
    --tp 1 --gtp ${GTP} --ep 64 --etp 1 --egtp 2 \
    --num-ns-steps 5 \
    --coefficient-type polar_express \
    --iters 20 \
    --warmup 5 \
    --set-timing \
    --batch-chunk 4 \
    --subgroup-sizes ${SUBGROUP_SIZES} \
    --subgroup-all-shapes \
    --batch-replicated-shapes \
    ${SYRK_OPTION} "

run_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${REPO_DIR}/tools/muon_analysis/bench_ns_strategies.py ${options}"

srun -l \
    --mpi=none \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    bash -c "${run_cmd}"
