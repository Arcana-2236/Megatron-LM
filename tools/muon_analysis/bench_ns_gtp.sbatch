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
#
# --fuse-shape-exchanges: scores a FOURTH set-composed policy, 'per_shape_fuse_sub_set'.
# Every shape the accepted state banked on a subgroup arm today pays its OWN pair of
# all_to_all_single AND its own independent deal, and the deal is what costs: a shape with
# 3 owned matrices dealt over k = world/g subgroups leaves k-3 subgroups idle for a whole
# region, so the region costs one WHOLE matrix of Newton-Schulz however few matrices it
# carries (10240x8192 at count 3 measured 10.857 ms against a 10.260 ms single-matrix
# duplicated time, and was FLAT in g at 10.857 / 10.738 / 10.910 for g = 4/8/16, which is
# the signature of exactly one matrix per busy subgroup at every g). This policy runs those
# shapes as ONE region with ONE pair of exchanges, and times two deals side by side so the
# delta is attributable: 'fuse_sub_g<g>' keeps the per-shape round-robin ownership (the
# fused exchange in isolation) and 'fuse_sub_g<g>_bal' also pools the deal across shapes by
# greedy LPT on ns_cost (the stranded matrices get absorbed). Numerics are untouched under
# both: every matrix is still all-gathered in RANK order and orthogonalized WHOLE by the
# same newton_schulz, same 5 steps, same polar_express coefficients, same SYRK path, and
# each shape still returns this rank's slice of every matrix in INPUT order. Shapes the
# accepted state did NOT bank on a subgroup arm (the shard_count == 1 shapes, which run
# 'duplicated_batch') are untouched and asserted so. Because which shapes a fused region
# spans is a property of the RANK PROFILE, the fused rows are reported per profile and
# never re-attributed to one shape.
# --pipeline-fused-exchange: adds the stage-4 phase-2 candidate `gtp-exchange-pipeline` as
# EXTRA COLUMNS of the fused-exchange region, one per pipeline depth in
# --fused-pipe-chunks, timed in the SAME job over the SAME tensors under the SAME
# composition as the monolithic arms and fed to the SAME fused-arm argmin. It is the port
# of the accepted EGTP restructure onto `newton_schulz_tp_subgroup_fused`, the last hot
# region still running the pre-stage-3 shape: async chunked exchange (the wire overlaps a
# different window's compute instead of bracketing all of it), staging-loop elision (the
# per-destination send buffer -- which at g = 4 holds every byte FOUR times -- becomes one
# per-SUBGROUP block handed to the list form of all_to_all, and the cat/stack python loops
# collapse to one strided copy per shape per chunk), and one batched call into
# kernels.fused_ns per shape per chunk in place of the per-matrix newton_schulz loop.
# Identical math: same deal, same matrices, same 5 steps, same polar_express coefficients,
# same SYRK path, and each shape still returns this rank's slice of every matrix in INPUT
# order. Off by default in the benchmark, so the flag is what turns it on here.
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
    --fuse-shape-exchanges \
    --pipeline-fused-exchange \
    --fused-pipe-chunks ${FUSED_PIPE_CHUNKS:-2 4} \
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
