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
# --elide-self-block: times each g == 1 PIPELINED column a FOURTH time with the rank's OWN
# block dropped from both exchanges ('duplicated_batch_sub_g1_fused_pipe_selfelide'), and
# scores 'per_shape_batch_sub_elide_set' against the accepted 'per_shape_batch_sub_pipe_set'.
# Identical deal, identical math, identical fused compute path, identical pipeline depth --
# what changes is that the s == rank entry of each per-chunk exchange leaves the collective
# (which becomes a peer-only batch_isend_irecv over the world - 1 real peers) and is served
# by a local device write into the buffer that CONSUMES it: into the gather buffer `gx` on
# the way in, into `result` on the way out. Both of those buffers were going to be written
# by the transpose the pipelined path already pays, so no HBM traffic is added; what is
# removed is the round trip of the own block through `recv`/`sendout`. The measured
# transport probe (job 2892006, dist_muon_opt/stage-4/phase-1-egtp-selfblock-elide/
# transport_probe.json) shows the self block was NOT on the wire at the region's real
# payload (all_to_all_single 3.611 ms vs peer-only p2p 3.594 ms, ratio 1.005 against 2.0 if
# it had been), so the win is the staging round trip, not wire bytes. The same job's
# real-arm window at the region's real count of 192 measured -4.328 ms (5120x2048) and
# -5.076 ms (2048x5120), bit-identical output. All four arms come from THIS one job over
# the SAME tensors under the SAME composition.
#
# --subgroup-wire-modes: adds the stage-4 phase-4 candidate `egtp-wire-bf16-return` as
# FURTHER COLUMNS of the same region -- one per (self-block treatment, wire mode) -- timed
# in the SAME job over the SAME tensors under the SAME composition and scored as
# 'per_shape_batch_sub_wire_set' against the accepted 'per_shape_batch_sub_elide_set'.
# What changes is the TRANSPORT encoding of the OUTPUT leg only: the reply that carries
# each peer's row block of the orthogonalized result travels as bf16 and is widened into
# the fp32 `result` inside the timed region (one strided pass per pipeline window, drained
# at a lag of two so it overlaps the next window's exchange rather than trailing it).
# COMPUTE precision is untouched: `gx` is still fp32 so kernels.fused_ns._supported still
# holds and the accepted fusion cannot silently un-bank, the 5 polar_express steps, the
# SYRK path, the deal and the pipeline depth are identical, and the region still returns
# fp32 in INPUT order.
#
# BITWISE, and measured rather than argued: fused_newton_schulz_batched holds X in bf16
# through all five steps and widens only at its _cast_into epilogue, so every value on
# that wire is exactly bf16-representable -- stage-4's preflight_wire_bf16.csv reports
# output_bf16_exact_frac = 1.000000 on all seven hot shapes, both axes.
#
# Only the OUTPUT leg is offered. The input leg (plan rank 5, `egtp-wire-bf16-input`) was
# refuted at zero scored cost by that same pre-flight: EGTP max_abs_diff 1.221e-3 against
# the 1e-3 equivalence gate.
#
# ACCEPTED and BANKED (job 2894899, 07:46). The argmin selected
# `duplicated_batch_sub_g1_fused_pipe_selfelide_wbf16out` on BOTH shapes and the EGTP axis
# went 118.128 -> 99.012 ms, per_shape_batch_sub_elide_set -> per_shape_batch_sub_wire_set
# = -19.117 ms, from the SAME job over the SAME tensors under the SAME composition. Per
# region: 2048x5120 59.049 -> 49.410 (-9.639), 5120x2048 59.079 -> 49.602 (-9.477). That
# clears the 5.17 ms EGTP axis nondeterminism floor by 3.70x, and it is 2.4x the -8.0 ms
# the candidate was priced at -- the pricing charged +4.3 ms for a receive-side upcast the
# pipeline turns out to absorb, because the widening is drained at a lag of two windows and
# so overlaps the next window's exchange instead of trailing the last one.
#
# `bf16out` is therefore the DEFAULT here. Set EGTP_SUBGROUP_WIRE_MODES="" to score the
# pre-phase-4 state, or to a longer mode list to re-open the argmin:
#   sbatch --export=ALL,USE_SYRK=1,EGTP_SUBGROUP_WIRE_MODES="" \
#          tools/muon_analysis/bench_ns_egtp.sbatch
# An unknown mode, a mode without --pipelined-exchange, or one on the unfused compute path
# is a hard ValueError in the benchmark, never a silent fp32 fallback.
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
    --elide-self-block \
    --subgroup-wire-modes ${EGTP_SUBGROUP_WIRE_MODES-bf16out} \
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
