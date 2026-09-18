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
#SBATCH -t 00:25:00
#SBATCH --mem=0
# No --gpus-per-node: Lyris rejects it as an invalid gres spec. --exclusive gives all
# 4 GPUs on the node.
#SBATCH --ntasks-per-node=4
#SBATCH --segment=16
#SBATCH --job-name=coreai_dlalgo_llm-dist_muon.bench_ns_gtp
#SBATCH --comment='{"APS":{}}'

export NCCL_DEBUG=WARN

GTP=${GTP:-64}

# ROOT_DIR is the directory CONTAINING the Megatron-LM checkout; run logs land in
# ${ROOT_DIR}/runs/. It defaults to the parent of wherever sbatch was invoked, which is
# correct when the job is submitted from the repo root -- what run_dist_muon_bench.sh
# does. Any other layout must say so explicitly:
#   sbatch --export=ALL,ROOT_DIR=/path/that/contains/Megatron-LM tools/muon_analysis/...
# It is deliberately NOT a hardcoded absolute path: one clone writing into another
# clone's runs/ directory is silent, and the wrapper then parses the wrong log.
ROOT_DIR="${ROOT_DIR:-$(cd "${SLURM_SUBMIT_DIR:-${PWD}}/.." && pwd)}"
REPO_DIR="${ROOT_DIR}/Megatron-LM"
if [[ ! -f "${REPO_DIR}/tools/muon_analysis/bench_ns_strategies.py" ]]; then
    echo "ERROR: no benchmark at ${REPO_DIR}/tools/muon_analysis/bench_ns_strategies.py" >&2
    echo "  ROOT_DIR=${ROOT_DIR} -- set it to the directory containing your checkout" >&2
    exit 1
fi

# The container must provide torch, triton and emerging_optimizers; see README.md.
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/training.sqsh}"
if [[ ! -e "${IMAGE_PATH}" ]]; then
    echo "ERROR: container image not found at ${IMAGE_PATH}" >&2
    echo "  set IMAGE_PATH to an image providing torch + triton + emerging_optimizers" >&2
    exit 1
fi

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
#
# g = 1 and g = 2 added (stage-5 phase-1 `gtp-subgroup-size-g1g2`). The sweep had only ever
# run g in {4, 8, 16}, yet g is a REPLICATION factor: a duplication group of g ranks all
# orthogonalize the whole matrix, so the compute term of every subgroup arm grows ~linearly
# in g while the wire term shrinks only slightly. The measured fused-region table is
# monotone in exactly that direction and never reached the small end -- profile 1 read
# fuse_sub_g16_bal_pipe2 = 221.830, g8 = 115.904, g4 = 64.660 (job 2896176), i.e. ~2x per
# halving of g, with the banked winner sitting at the SMALLEST g ever measured.
# g = 1 makes each rank its own group (owner-computes, the design the EGTP axis already
# runs); g = 2 halves the replication while keeping two ranks per group so the greedy-LPT
# deal still has slack. Both are legal: newton_schulz_tp_subgroup{,_fused} reject only
# groups == 1 (g == world), and the world-4 window has run fuse_sub_g1_pipe2 BITWISE against
# the per-matrix duplicated lineage since phase-3.
# Zero new code path: these are two more columns of the SAME sweep, fed to the SAME
# per-shape / per-profile argmin, so a column that loses cannot change the scored policy.
# The physical risk is the other way -- at small g the deal has fewer matrices per group
# than groups, so the region floors at ONE whole matrix and LPT imbalance can eat the win;
# that is why g = 2 is swept beside g = 1 rather than instead of it.
SUBGROUP_SIZES=${SUBGROUP_SIZES:-"1 2 4 8 16"}

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
# --fused-wire-modes: adds the stage-4 phase-3 candidate `gtp-wire-bf16` as further EXTRA
# COLUMNS of the fused-exchange region -- one per (pipeline depth, wire mode) -- timed in
# the SAME job over the SAME tensors under the SAME composition and fed to the SAME
# fused-arm argmin as every other fused column. The region moves fp32 on both exchange legs
# today (--dtype defaults to float32) while every GEMM inside it runs bf16; these arms move
# bf16 instead. COMPUTE precision is untouched in every mode: the gather buffer handed to
# the fused Newton-Schulz kernel is still fp32 (so kernels.fused_ns._supported still holds
# and the fusion cannot silently un-bank), the 5 polar_express steps, the SYRK path and the
# deal are identical, and the region still returns fp32 in INPUT order. Three of the four
# width changes ride copies the pipelined path already pays; the scatter into `results` is
# the one that costs an extra pass, which is why the legs are timed as SEPARABLE arms:
#   bf16out  output leg only -- BITWISE. The fused kernel keeps X in bf16 through all five
#            steps and only widens at its epilogue, so every value it returns is exactly
#            bf16-representable. Measured, not argued: preflight_wire_bf16.csv reports
#            output_bf16_exact_frac = 1.000000 on all five GTP shapes.
#   bf16in   input leg only -- carries 4/5 of the region's wire bytes at subgroup_size = 4
#            (per rank the input leg receives world * my_block, the output leg groups *
#            my_block, i.e. 64 : 16). Perturbs every entry before the Newton-Schulz
#            prologue, so it is gated on the equivalence check.
#   bf16     both legs.
# Off by default in the benchmark AND empty by default here, so the scored policy set is
# byte-identical to the accepted state unless FUSED_WIRE_MODES is set. It is left wired
# rather than removed because the phase-3 measurement is what settled the question and must
# stay reproducible in one command:
#   sbatch --export=ALL,USE_SYRK=1,FUSED_WIRE_MODES="bf16 bf16in bf16out" \
#          tools/muon_analysis/bench_ns_gtp.sbatch
# MEASURED AND NOT BANKED (job 2894159, 54 fused arms, 18:41). At the real 64-rank region
# the whole win is -1.750 ms on a 64.357 ms region (fuse_sub_g4_bal_pipe2 64.357 ->
# _wbf16 62.607) and -1.493 ms on the GTP axis (82.070 -> 80.577), against the 2.70 ms
# GTP conservative nondeterminism floor -- it does not clear it, so the arm is not adopted.
# The per-leg split also refutes the byte model the candidate was priced on: bf16in carries
# 4/5 of the wire bytes and bought -0.605 ms, while bf16out carries 1/5 AND pays an extra
# HBM pass yet bought -0.813 ms. Halving the wire is not what this region is waiting on.
# Cost reference for any future sweep: ~8.5 s of GPU time per extra arm (jobs 2892041 =
# 6 fused arms / 12:50 vs 2892579 = 18 fused arms / 14:32).
# --batch-replicated-fused: scores 'per_shape_sub_all_batch_fused_set' -- the accepted
# 'per_shape_sub_all_batch_set' with ONE substitution, on the two shard_count == 1
# (replicated) shapes only (8192x2048 at 24 owned, 512x8192 at 12): their banked
# 'duplicated_batch' column is routed through kernels.fused_ns.fused_newton_schulz_batched
# instead of the library newton_schulz. Those two rows never received stage-3's accepted
# NS-step fusion, because at partition_dim is None newton_schulz_tp_batched short-circuits
# to a local 3-D newton_schulz -- so they still pay the fp32 normalize + bf16 cast prologue
# and the bf16 -> fp32 cast + store epilogue as separate HBM passes, which is why they are
# the two worst '% of SOL' rows on this axis. Identical math: same batched call, same
# --batch-chunk, same 5 steps, same polar_express coefficients, same batched SYRK path, no
# collective anywhere near it (a replicated shape has nothing to gather). The candidate
# column is timed BESIDE the arm it substitutes in this same job, over the same tensors,
# under the same composition, so the delta is never a between-jobs one. The fused entry
# point's silent fallback is a hard error on this path, so the column can never measure the
# reference arm under the candidate's name. Set BATCH_REPLICATED_FUSED=0 to score the
# pre-phase-6 state.
# FUSED_DIST_IN_GROUP_SIZES: duplication-group sizes g that ALSO get a distribute-in-group
# column of the fused-exchange region (stage-5 phase-2 `gtp-distribute-in-group-g2`).
#
# g is a REPLICATION factor: at `fuse_sub_g<g>` all g ranks of a duplication subgroup
# orthogonalize the WHOLE matrix. Phase 1 drove g down to 1 for exactly that reason and the
# region landed at 49.604 ms -- but the measured table is FLAT between g = 1 (49.604) and
# g = 2 (49.967) even though g = 2 does twice the arithmetic, which is the signature of a
# region bound by its LPT critical path (one 36864x8192 matrix) rather than by aggregate
# throughput: at both g the largest subgroup load is that single matrix, and 51 of 64 ranks
# wait on the rank that owns it.
#
# `fuse_sub_g<g>_dist...` breaks that floor instead of trading against it: inside a subgroup
# each rank keeps its OWN slab of a tall matrix and the g ranks cooperate -- one Gram
# all-reduce per Newton-Schulz step -- so the biggest matrix's critical path shards g ways
# and the balanced deal (which charges a distributed shape its per-rank fraction) can
# actually balance. Wire is unchanged from g = 1: a distributed shape's shard goes to ONE
# rank of each subgroup, not to all g. Wide-once-gathered shapes stay replicated in the
# subgroup, so their exchange, staging and compute are the pre-existing path byte for byte.
# Identical math otherwise: same 5 steps, same polar_express coefficients, same SYRK path,
# same whiten-the-smaller-dimension orientation, and each shape still returns this rank's
# slice of every matrix in INPUT order. The two numeric deviations -- the Frobenius norm
# summed over shards (in fp64) and the Gram summed over g partial products -- are the ones
# `tp_mode="distributed"` already has and are what the world-4 window's equivalence gate
# measures (scripts/check_dist_in_group.py).
# Empty this to score the pre-phase-2 state exactly:
#   sbatch --export=ALL,USE_SYRK=1,FUSED_DIST_IN_GROUP_SIZES= ...
# g = 2 and g = 4 are both swept: the win is a balance win, and it is not monotone in g.
# Distributing shrinks a TALL shape's per-rank arithmetic by ~(3/g + M/N)/(3 + M/N) but a
# WIDE shape is still replicated g ways, so the wide half of the region grows linearly in g
# while the tall half shrinks -- g = 2 is the modelled optimum and g = 4 is swept beside it
# so the argmin, not the model, decides. ~8.5 s of GPU time per extra arm.
FUSED_DIST_IN_GROUP_SIZES=${FUSED_DIST_IN_GROUP_SIZES-2 4}

BATCH_REPLICATED_FUSED_OPTION=""
if [[ "${BATCH_REPLICATED_FUSED:-1}" == "1" ]]; then
    BATCH_REPLICATED_FUSED_OPTION="--batch-replicated-fused"
fi

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
    --fused-wire-modes ${FUSED_WIRE_MODES:-} \
    --fused-dist-in-group-sizes ${FUSED_DIST_IN_GROUP_SIZES} \
    ${BATCH_REPLICATED_FUSED_OPTION} \
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
