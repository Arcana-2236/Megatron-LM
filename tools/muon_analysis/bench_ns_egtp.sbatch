#!/bin/bash

# Benchmark Newton-Schulz distribution strategies over the EGTP axis.
#
# EGTP=4 fits on one GB300 node (4 GPUs), but in a real job the group can span the
# scale-out fabric, InfiniBand or RoCE depending on the cluster. NVLink, SHM and NVLS are
# disabled below so the collectives take the network path and the timings reflect that.

#SBATCH -p gb300
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 00:20:00
#SBATCH --mem=0
# No --gpus-per-node: Lyris rejects it as an invalid gres spec. --exclusive gives all
# 4 GPUs on the node.
#SBATCH --ntasks-per-node=2
#SBATCH --job-name=coreai_dlalgo_llm-dist_muon.bench_ns_egtp
#SBATCH --comment='{"APS":{}}'

# Force the EGTP group off NVLink so it measures the network path.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

ROOT_DIR="/lustre/fsw/coreai_dlalgo_llm/zhengywang/dist_muon_proxy_ootb"
REPO_DIR="${ROOT_DIR}/Megatron-LM"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/training.sqsh}"

# TAG separates runs on different hardware, e.g. sbatch -p gb200 --export=ALL,TAG=gb200
NAME="runs/ns_bench/egtp${TAG:+_$TAG}"
LOGS_DIR="${ROOT_DIR}/${NAME}/logs"
mkdir -p "${LOGS_DIR}"

DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

# torchrun is not used, so the rendezvous env has to come from Slurm explicitly.
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29511

# USE_SYRK=1 forwards --use-syrk to newton_schulz_tp (Triton triangular kernel).
# sbatch --export=ALL,USE_SYRK=1 bench_ns_egtp.sbatch
SYRK_OPTION=""
if [[ "${USE_SYRK:-0}" == "1" ]]; then SYRK_OPTION="--use-syrk"; fi

# --subgroup-sizes 1 --subgroup-batched: OWNER-COMPUTES on this 2-rank axis. g=1 means the
# duplication group is a single rank, so the 192 matrices of a shape are dealt round-robin
# across the two ranks and each matrix is orthogonalized WHOLE by exactly one of them;
# redundancy falls 2.0 -> 1.0. Two all_to_all_single exchanges realize it: the input side
# gives each rank every shard of the matrices it owns (half the ingress duplicated's
# all_gather moves, not more), and the output side hands the other rank its row-slice of
# each result -- the result is MOVED rather than recomputed. --subgroup-batched runs the
# owned half as ONE batched Newton-Schulz per --batch-chunk, so owner-computes composes
# with the accepted batched arm instead of competing with it, and the scored pair is
# 'per_shape_batch_sub_set' (candidate) vs 'per_shape_batch_set' (the accepted state,
# i.e. 'duplicated_batch' here). Both arms come from THIS one job under ONE composition.
# The unbatched 'duplicated_sub_g1' column is timed alongside as the control that shows
# what batching contributes on top of owner-computes. The g list is overridable with
# EGTP_SUBGROUP_SIZES -- deliberately NOT the GTP axis's SUBGROUP_SIZES, because the
# driver submits both axes with --export=ALL and one shared name would leak a GTP
# sweep list (4 8 16) onto a 2-rank group where none of those values is legal.
#
# The script sweeps every rank profile itself and reports the slowest, so nothing to pick.
#
# --fused-ns-kernel: times each batched-subgroup column a SECOND time with the fused
# Newton-Schulz entry point from dist_muon_opt/kernels/fused_ns.py, and scores
# 'per_shape_batch_sub_fused_set' against the accepted 'per_shape_batch_sub_set'. Identical
# math -- same 5 steps, same polar_express coefficients, same batched Triton SYRK path --
# with the fp32 Frobenius-normalize + bf16 cast prologue and the bf16->fp32 cast + store
# epilogue that wrap the chain collapsed into two Triton passes each. Measured composition
# of the unfused region on this hardware (dist_muon_opt/stage-3/phase-2-egtp-ns-step-fusion,
# job 2882597): of 34.7-37.7 ms of compute per region, 28.1 ms is the GEMM chain, 6.6 ms is
# exactly those prologue/epilogue passes, and the tall shape pays a further ~3.0 ms because
# its store is strided. Both arms come from THIS one job over the SAME tensors under the
# SAME composition.
#
# --pipelined-exchange: times each g == 1 batched-subgroup column a THIRD time with the
# restructured exchange ('duplicated_batch_sub_g1_fused_pipe'), and scores
# 'per_shape_batch_sub_pipe_set' against the accepted 'per_shape_batch_sub_fused_set'.
# Identical deal, identical math, identical fused compute path -- what changes is that
# BLOCK ownership (matrix j -> subgroup j // mine instead of j % k) makes the input
# exchange's send buffer and the output exchange's receive buffer exact VIEWS of the
# caller's tensors, so two of the four full-size staging copies stop existing, and the
# per-chunk exchanges are issued async_op=True on the process group's stream so the wire
# overlaps a DIFFERENT chunk's compute. --pipe-chunks sets the pipeline depth (4 windows of
# 24 owned matrices here); it raises the collective count per region from 2 to 8, which at
# the measured alpha_ib = 75.681 us costs +0.45 ms/region. All three arms come from THIS
# one job over the SAME tensors under the SAME composition.
#
# --set-timing: this axis owns 192 copies each of two expert shapes, so its cost is a
# property of the SET, not of one matrix. It times the whole owned group inside one region
# under two arms -- unbatched ('<mode>_set', the baseline) and batched ('duplicated_batch',
# one all-gather + one batched Newton-Schulz per chunk) -- and scores 'fastest step' over
# the set-composed policies. Both arms come from this one job, so the batched arm is never
# credited with the composition change. --batch-chunk bounds the all-gather buffer.
options=" \
    --group egtp \
    --modelled-world-size 128 \
    --tp 1 --gtp 64 --ep 64 --etp 1 --egtp 2 \
    --num-ns-steps 5 \
    --coefficient-type polar_express \
    --iters 20 \
    --warmup 5 \
    --set-timing \
    --batch-chunk 64 \
    --subgroup-sizes ${EGTP_SUBGROUP_SIZES:-1} \
    --subgroup-batched \
    --fused-ns-kernel \
    --pipelined-exchange \
    --pipe-chunks ${EGTP_PIPE_CHUNKS:-4} \
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
