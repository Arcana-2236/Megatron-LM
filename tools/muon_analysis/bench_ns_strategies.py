# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Benchmark Newton-Schulz distribution strategies on a fixed set of problem sizes.

``TensorParallelMuon`` can orthogonalize a sharded weight three ways, selected by
``--muon-tp-mode``:

  duplicated   all-gather the shards, run Newton-Schulz on the whole matrix on every
               rank, then keep this rank's slice. One all-gather, redundant compute.
  distributed  keep the shard, run Newton-Schulz across the group by all-reducing the
               small Gram matrix each step. ``steps`` all-reduces, but the replicated
               ``A @ A`` term does not shrink with the group.
  blockwise    run Newton-Schulz on the local shard, no collective at all. Off by default:
               it orthogonalizes each block rather than the matrix, so it changes the
               update rather than distributing the same one. Pass ``--modes`` to include it.

Which wins depends on the matrix shape, the group size, and whether the group sits
inside an NVLink domain. This measures that directly, on the shapes a given rank
actually owns.

``duplicated`` and ``distributed`` are also the two corners of one continuous knob -- how
many ranks redundantly orthogonalize the same matrix -- and ``--subgroup-sizes`` (under
``--set-timing``) measures the interior: the owned set of a shape dealt across
``world_size / g`` disjoint duplication subgroups. See ``newton_schulz_tp_subgroup``.

Two axes are worth measuring separately:

  --group gtp    dense weights, sharded ``GTP`` ways. On GB200 a 64-rank GTP group fits
                 inside one NVLink domain, so launch across enough nodes to fill it.
  --group egtp   routed-expert weights, sharded ``EGTP`` ways. Small enough to sit on a
                 single node, but in a real job the group can span the scale-out fabric,
                 InfiniBand or RoCE depending on the cluster. To emulate that, run on one
                 node with NVLink disabled::

                     NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NVLS_ENABLE=0

The layer-wise optimizer assigns whole matrices to shards, so ranks own different shapes:
one rank may hold a single 10240x24576 attention matrix while another holds two dozen
3072x10240 expert matrices. Every rank orthogonalizes concurrently and the step finishes
when the slowest does, so the step cost is the *max* over ranks, not the mean.

Sweeping every rank is free. Ranks collapse into a handful of distinct profiles over a
handful of distinct shapes, so each shape is timed once and profile totals are composed
from those numbers. The world size of this job only has to match the sharding degree of
``--group``, not the modelled job.

Example, 4-rank EGTP group on one node with NVLink off::

    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NVLS_ENABLE=0 \\
    torchrun --nproc-per-node 4 tools/muon_analysis/bench_ns_strategies.py \\
        --group egtp --num-ns-steps 16
"""

import argparse
import os
import statistics
import sys
from collections import Counter
from typing import Dict, List, Tuple

import torch

try:
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
        newton_schulz,
        newton_schulz_tp,
    )

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False

# Optional: the fused batched Newton-Schulz kernel from the optimization workspace
# (``dist_muon_opt/kernels/fused_ns.py``). It is a drop-in for the ``newton_schulz`` call
# the batched-subgroup compute loop makes, with the fp32 normalize/cast prologue and the
# bf16->fp32 epilogue-plus-store fused into two Triton passes each. The 5-step chain
# itself is untouched (same coefficients, same batched SYRK path, same step count); only
# the surrounding full-size fp32 traffic changes. Imported by path so the benchmark keeps
# running unchanged in a tree that has no workspace beside it.
_FUSED_NS_PATH = os.environ.get(
    "FUSED_NS_DIR",
    "/lustre/fsw/coreai_dlalgo_llm/zhengywang/dist_muon_proxy_ootb/dist_muon_opt",
)
try:
    if _FUSED_NS_PATH not in sys.path:
        sys.path.insert(0, _FUSED_NS_PATH)
    from kernels.fused_ns import FUSED_NS_AVAILABLE, fused_newton_schulz_batched
    # ``fused_newton_schulz_batched`` falls back to the library ``newton_schulz`` whenever
    # its preconditions do not hold, which makes it safe to enable unconditionally but ALSO
    # makes a silent un-banking possible. A caller that SCORES the fusion as its own column
    # imports the predicate and asserts it rather than assuming it.
    from kernels.fused_ns import _supported as fused_ns_supported

    HAVE_FUSED_NS = FUSED_NS_AVAILABLE
except Exception:  # pragma: no cover - absence is a supported configuration
    HAVE_FUSED_NS = False
    fused_newton_schulz_batched = None
    fused_ns_supported = None

# --------------------------------------------------------------------------------------
# Weight shapes of the modelled workload: a 54-layer hybrid Mamba-MoE.
#
# These constants DEFINE the benchmark. Do not change them: they are the problem to be
# solved, not a parameter to be tuned. Timings across runs are only comparable at these
# values.
# --------------------------------------------------------------------------------------

HIDDEN = 8192
FFN = 5120                 # routed-expert intermediate
MOE_LATENT = 2048
NUM_EXPERTS = 512
SHARED_EXPERT = 10240
KV_CHANNELS = 128
NUM_HEADS = 64
NUM_QUERY_GROUPS = 8
MAMBA_HEADS = 256
MAMBA_HEAD_DIM = 64
D_INNER = MAMBA_HEADS * MAMBA_HEAD_DIM
# in_proj packs z|x|B|C|dt = 16384 + 16384 + 1024 + 1024 + 256 = 35072, stored padded to
# 36864 by the GTP weight-remat allocator (alignment is lcm(64, bucket_divisor/dp), not
# GTP divisibility -- 35072 already divides 64). Newton-Schulz sees the padded rows, so
# the padded value is the one to model. Bucket alignment itself is NOT modelled here.
MAMBA_IN_PROJ = 36864
# 6 blocks of MEMEMEM*E: 4 mamba, 4 moe, 1 attention each.
NUM_MAMBA_LAYERS = 24
NUM_MOE_LAYERS = 24
NUM_ATTN_LAYERS = 6


def build_model_matrices(config) -> Tuple[List, List]:
    """Return (dense, expert) matrices as ``(local_shape, shard_degree)`` pairs.

    ``local_shape`` is what one rank stores; Newton-Schulz sees ``local_shape[0] *
    shard_degree`` rows once the group has all-gathered.
    """
    tp, gtp, etp, egtp = config.tp, config.gtp, config.etp, config.egtp
    local_experts = NUM_EXPERTS // config.ep
    dense, expert = [], []

    dense += [((MAMBA_IN_PROJ // gtp, HIDDEN), gtp)] * NUM_MAMBA_LAYERS
    dense += [((HIDDEN // gtp, D_INNER // tp), gtp)] * NUM_MAMBA_LAYERS
    qkv_out = (NUM_HEADS + 2 * NUM_QUERY_GROUPS) * KV_CHANNELS
    dense += [((qkv_out // tp // gtp, HIDDEN), gtp)] * NUM_ATTN_LAYERS
    dense += [((HIDDEN // gtp, NUM_HEADS * KV_CHANNELS // tp), gtp)] * NUM_ATTN_LAYERS
    for _ in range(NUM_MOE_LAYERS):
        dense.append(((NUM_EXPERTS, HIDDEN), 1))  # router: never sharded
        if config.shard_latent_proj:
            dense.append(((MOE_LATENT // gtp, HIDDEN), gtp))
            dense.append(((HIDDEN // gtp, MOE_LATENT), gtp))
        else:
            dense.append(((MOE_LATENT, HIDDEN), 1))
            dense.append(((HIDDEN, MOE_LATENT), 1))
        dense.append(((SHARED_EXPERT // tp // gtp, HIDDEN), gtp))
        dense.append(((HIDDEN // gtp, SHARED_EXPERT // tp), gtp))
        for _ in range(local_experts):
            expert.append(((FFN // etp // egtp, MOE_LATENT), egtp))
            expert.append(((MOE_LATENT // etp // egtp, FFN), egtp))
    return dense, expert


def ns_cost(matrix) -> int:
    """Newton-Schulz cost of the full post-all-gather matrix, as the optimizer models it."""
    (rows, cols), shard_count = matrix
    rows *= shard_count
    big, small = max(rows, cols), min(rows, cols)
    return big * small * small


def owned_matrices(matrices: List, dp_size: int, dp_rank: int) -> List:
    """Return the matrices assigned to *dp_rank* under compute-balanced greedy LPT.

    Mirrors ``_emit_bucket``'s ordering so the benchmark measures what a rank really
    orthogonalizes. Bucketing is ignored: it changes which matrices land together, not
    the set of shapes, and the per-shape timings are what matter here.
    """
    loads = [0] * dp_size
    owned = [[] for _ in range(dp_size)]
    for matrix in sorted(matrices, key=lambda e: -ns_cost(e)):
        shard = min(range(dp_size), key=lambda s: loads[s])
        loads[shard] += ns_cost(matrix)
        owned[shard].append(matrix)
    return owned[dp_rank]


# --------------------------------------------------------------------------------------
# FLOP model
# --------------------------------------------------------------------------------------


def ns_step_flops(m: int, n: int, use_syrk: bool = False) -> float:
    """FLOPs for one Newton-Schulz step on an (m, n) matrix with m <= n.

    Per ``newton_schulz_step``: ``A = X @ X.mT`` costs 2*m^2*n, ``B = A @ A`` costs 2*m^3,
    and ``X = B @ X`` costs 2*m^2*n.

    With ``use_syrk`` the two symmetric products run as triangular kernels at half the
    FLOPs -- ``A = X @ X.mT`` costs m^2*n and ``B = A @ A`` costs m^3 -- while ``X = B @ X``
    stays a general GEMM. That is 25% fewer FLOPs when m << n, rising to 33% at m == n.
    """
    if use_syrk:
        return 3.0 * m * m * n + 1.0 * m * m * m
    return 4.0 * m * m * n + 2.0 * m * m * m


def flop_model(
    matrix, mode: str, steps: int, group_size: int, use_syrk: bool = False
) -> Tuple[float, float]:
    """Return (issued, useful) FLOPs on one GPU for one orthogonalization.

    ``useful`` is the irreducible share: orthogonalizing the full matrix once, divided
    evenly across the group. ``issued`` is what the mode actually executes on one GPU.
    Their ratio is the redundancy the mode pays.

    blockwise issues *less* than useful, because it orthogonalizes each block rather than
    the matrix. That is a different and cheaper computation, not a faster route to the same
    answer, so its throughput is not comparable to the other two on equal terms.

    ``shard_count`` is how many ranks this particular weight is split across, which is not
    always ``group_size``: the MoE router is replicated, so every rank holds the whole
    thing. A replicated weight is orthogonalized identically and redundantly on every rank,
    with no collective and nothing to divide.
    """
    (rows, cols), shard_count = matrix
    if shard_count == 1:
        issued = ns_step_flops(min(rows, cols), max(rows, cols), use_syrk) * steps
        return issued, issued

    full_rows, full_cols = rows * shard_count, cols
    fm, fn = min(full_rows, full_cols), max(full_rows, full_cols)
    useful = ns_step_flops(fm, fn, use_syrk) * steps / shard_count

    if mode == "blockwise":
        issued = ns_step_flops(min(rows, cols), max(rows, cols), use_syrk) * steps
    elif mode == "duplicated":
        # Every rank recomputes the whole matrix, so issued is shard_count x useful.
        issued = ns_step_flops(fm, fn, use_syrk) * steps
    else:
        # distributed shards the two m^2*n GEMMs across the group; A @ A is replicated.
        # When the sharded dimension is the SHORTER one, newton_schulz runs along the long
        # dimension instead, the waste newton_schulz_tp's docstring warns about.
        if full_rows < full_cols:
            m, n = fn, fm
        else:
            m, n = fm, fn
        # SYRK halves the two symmetric products, so the sharded term goes 4 -> 3 and the
        # replicated one 2 -> 1.
        sharded_coeff, replicated_coeff = (3.0, 1.0) if use_syrk else (4.0, 2.0)
        issued = (
            sharded_coeff * m * m * n / shard_count + replicated_coeff * m * m * m
        ) * steps
    return issued, useful


# --------------------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------------------


def time_strategy(
    local_shard, group, mode, steps, coefficient_type, iters, warmup, shard_count,
    use_syrk=False,
) -> float:
    """Return the median wall-clock milliseconds of one orthogonalization.

    A weight with ``shard_count == 1`` is replicated rather than sharded, so there is
    nothing to gather and every mode collapses to a local Newton-Schulz. Passing
    partition_dim=0 for one would all-gather ``group_size`` identical copies and
    orthogonalize a matrix that does not exist in the model.
    """
    # blockwise passes partition_dim=None, which is newton_schulz_tp's non-TP fallback:
    # Newton-Schulz on the local block with no collective.
    partition_dim = None if (mode == "blockwise" or shard_count == 1) else 0
    tp_mode = "duplicated" if mode == "blockwise" else mode

    def once():
        newton_schulz_tp(
            local_shard,
            steps=steps,
            coefficient_type=coefficient_type,
            tp_group=group,
            partition_dim=partition_dim,
            tp_mode=tp_mode,
            use_syrk=use_syrk,
        )

    for _ in range(warmup):
        once()
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)

    timings = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.distributed.barrier(group=group)
        start.record()
        once()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


# --------------------------------------------------------------------------------------
# Set-composed timing (--set-timing)
#
# ``time_strategy`` above times ONE matrix with a fully drained stream (barrier, record,
# synchronize per call) and the profile total multiplies that median by the owned count.
# That composition is exact for a per-matrix cost, but it makes any cross-matrix
# optimization unmeasurable by construction, and it charges every matrix a full
# host-dispatch bubble that back-to-back issue would hide.
#
# ``--set-timing`` times the WHOLE owned group of one shape inside a single timed region
# and reports that region directly as the group's contribution to the profile total. Two
# arms are timed under that identical composition so a cross-matrix optimization is scored
# against a like-for-like baseline and never credited with the composition change itself:
#
#   <mode>_set       the unmodified algorithm, one ``newton_schulz_tp`` call per matrix,
#                    issued back-to-back. This is the set-composed BASELINE arm.
#   duplicated_batch the same math with the group stacked into one 3-D tensor: ONE
#                    all-gather for the whole stack and ONE batched Newton-Schulz chain.
#
# Both arms run over the identical tensors (the unbatched arm iterates the slices of the
# same stack the batched arm consumes whole), so data, dtype and resident footprint are
# equal and only the issue pattern differs.
# --------------------------------------------------------------------------------------

SET_SUFFIX = "_set"
BATCH_POLICY = "duplicated_batch"


def newton_schulz_tp_batched(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str,
    tp_group,
    partition_dim: int | None,
    tp_mode: str,
    use_syrk: bool = False,
    fused: bool = False,
) -> torch.Tensor:
    """Batched ``newton_schulz_tp`` over a stack of identically shaped local shards.

    ``x`` is ``(B, rows, cols)``: B matrices of the same shape, each sharded the same way
    across ``tp_group``. ``duplicated`` mode then needs ONE all-gather for the whole stack
    instead of B, and ``emerging_optimizers.newton_schulz`` already accepts a 3-D input and
    dispatches to ``batched_newton_schulz_step_tsyrk`` / ``batched_tsyrk_ex``, so the
    arithmetic is the same per-matrix Newton-Schulz -- ``baddbmm``/batched-SYRK over a
    leading batch dim rather than ``addmm``/SYRK once per matrix. No reassociation, same
    coefficients, same step count, same SYRK path.

    ``distributed`` is deliberately NOT batched. Its collective route runs
    ``distributed_normalize_p2``, which reduces ``(x*x).sum()`` over the whole tensor; on a
    3-D stack that would normalize all B matrices by one shared Frobenius norm, which is a
    different computation, not a faster route to the same one.

    ``fused`` routes the REPLICATED (``partition_dim is None``) branch through
    ``kernels.fused_ns.fused_newton_schulz_batched`` instead of the library
    ``newton_schulz``. Same 5-step chain, same polar_express coefficients, same batched
    SYRK path, same batch dim -- the only difference is that the fp32 normalize + bf16 cast
    prologue and the bf16 -> fp32 cast + store epilogue stop being separate HBM passes.
    It is offered on the replicated branch ONLY: the collective branch's ``global_x`` is a
    ``permute``d ``reshape`` and the fused entry point requires a contiguous input, so
    enabling it there would silently fall back to the library call it is meant to replace.
    """
    if partition_dim is None:
        # Replicated weight: nothing to gather, same non-TP fallback newton_schulz_tp takes.
        if not fused:
            return newton_schulz(x, steps, coefficient_type, use_syrk=use_syrk)
        # Assert the fused preconditions rather than assume them: the entry point falls
        # back to ``newton_schulz`` silently, which would time the reference arm under the
        # candidate's name. This arm is scored, so the fallback must be an error here.
        if not HAVE_FUSED_NS:
            raise RuntimeError(
                "fused Newton-Schulz requested but kernels.fused_ns is unavailable"
            )
        if not fused_ns_supported(x, coefficient_type, use_syrk):
            raise RuntimeError(
                "fused Newton-Schulz preconditions do not hold for the replicated batched "
                f"path (shape={tuple(x.shape)}, dtype={x.dtype}, "
                f"contiguous={x.is_contiguous()}, use_syrk={use_syrk}, "
                f"fp32_matmul_precision={torch.get_float32_matmul_precision()}); the entry "
                "point would silently fall back to newton_schulz and the column would not "
                "measure the fusion"
            )
        return fused_newton_schulz_batched(
            x, steps, coefficient_type, use_syrk=use_syrk,
        )
    if tp_mode != "duplicated":
        raise ValueError(f"batched Newton-Schulz supports tp_mode='duplicated' only, got {tp_mode!r}")
    if partition_dim not in (0, 1):
        raise ValueError(f"Invalid partition_dim: {partition_dim}")

    x = x.contiguous()
    world = tp_group.size()
    batch, rows, cols = x.shape
    # ONE aggregated, CONTIGUOUS all-gather for the whole stack. ``all_gather_into_tensor``
    # gives NCCL a single flat destination, so the whole chunk moves in one ncclAllGather of
    # ``batch`` x the per-matrix payload. The per-matrix path's list-based ``all_gather``
    # instead gathers into a flat buffer AND copies out into ``world`` separate tensors,
    # which ``torch.cat`` then copies a second time -- so this is one collective and one
    # copy where the unbatched arm pays ``batch`` collectives and two copies each.
    gathered = torch.empty((world, batch, rows, cols), dtype=x.dtype, device=x.device)
    torch.distributed.all_gather_into_tensor(gathered, x, group=tp_group)
    # Rank-major (W, B, r, c) -> the per-matrix concatenation along partition_dim, per
    # matrix. +1 on the dim indices for the leading batch dim.
    if partition_dim == 0:
        global_x = gathered.permute(1, 0, 2, 3).reshape(batch, world * rows, cols)
    else:
        global_x = gathered.permute(1, 2, 0, 3).reshape(batch, rows, world * cols)
    orthogonalized = newton_schulz(global_x, steps, coefficient_type, use_syrk=use_syrk)
    return orthogonalized.chunk(world, dim=partition_dim + 1)[tp_group.rank()]


# --------------------------------------------------------------------------------------
# Subgroup duplication (--subgroup-sizes)
#
# ``duplicated`` and ``distributed`` are the two CORNERS of one knob: how many ranks
# redundantly orthogonalize the same matrix. ``duplicated`` sets that duplication group to
# the whole TP group (all ``world`` ranks recompute every matrix, redundancy = world);
# ``distributed`` sets it to 1 but pays ``steps`` all-reduces and keeps a replicated
# ``A @ A`` term. Nothing forces the group to be one of those two values.
#
# With g < world the ``world`` ranks split into ``k = world / g`` subgroups and the owned
# same-shape SET is dealt round-robin across them: subgroup s orthogonalizes matrices
# ``s, s+k, s+2k, ...`` and no other. Per rank the arithmetic falls by ~k because it runs
# ~count/k matrices instead of count -- each of them still whole, by exactly the same
# ``newton_schulz`` call ``duplicated`` makes, so the numerics are unchanged.
#
# Two collectives realize it, both over the FULL group (the shards live on all ``world``
# ranks no matter how the work is partitioned, so a subgroup-local collective could not
# reach them). Both are ``all_to_all_single``, which is the shape of this exchange:
#
#   input   every rank sends its own shard of the matrices that belong to the RECEIVER's
#           subgroup. Per-rank ingress is ``count/k`` gathered matrices rather than
#           ``count`` -- a k-fold reduction of the volume ``duplicated``'s per-matrix
#           all_gather moves, not a k-fold increase. Volume, written down before coding:
#           egress = ingress = (world-1)/world x count/k x world x shard_bytes.
#   output  each rank of a subgroup is the designated sender for ``k`` of the ``world``
#           destinations (``d % g == rank % g``), and ships them their row-slice of the
#           subgroup's matrices. This is the only NEW traffic: count x shard_bytes per
#           rank each way, ~1/63 of what the input side saves.
#
# This is a POLICY over the existing kernels, not a new kernel: same ``newton_schulz``,
# same 2-D call per matrix, same coefficients, same step count, same SYRK path.
# --------------------------------------------------------------------------------------

SUBGROUP_PREFIX = "duplicated_sub_g"
# Owner-computes INSIDE the batched path (--subgroup-batched): the same subgroup deal as
# ``duplicated_sub_g<g>``, but the owned matrices are orthogonalized as ONE batched
# Newton-Schulz per chunk instead of one 2-D call each. At g == 1 on a 2-rank axis this is
# literally owner-computes: each rank orthogonalizes half the matrices whole and the two
# exchange result slices, so ``redundancy`` falls from 2.0 to 1.0.
BATCH_SUBGROUP_PREFIX = "duplicated_batch_sub_g"
# Reference arm for the batched-subgroup candidate: the ACCEPTED state, i.e. per shape the
# better of the set-composed winner and the batched arm. Costs no extra timing -- both
# columns it selects between are already measured.
PER_SHAPE_BATCH_POLICY = "per_shape_batch_set"
# The candidate: the same per-shape argmin with the batched-subgroup arms added.
PER_SHAPE_BATCH_SUBGROUP_POLICY = "per_shape_batch_sub_set"
PER_SHAPE_SUBGROUP_POLICY = "per_shape_sub_set"
# Same substitution rule, wider scope: also on the shapes whose set-composed winner is
# ``distributed_set``. See the ``--subgroup-all-shapes`` block near the scored policies.
PER_SHAPE_SUBGROUP_ALL_POLICY = "per_shape_sub_all_set"
# The replicated-batch candidate: ``per_shape_sub_all_set`` with the ALREADY-TIMED
# ``duplicated_batch`` column added to the per-shape argmin, admitted on shard_count == 1
# shapes ONLY. Those shapes get no subgroup column (subgrouping a replicated weight is a
# no-op) and their set-composed winner is a per-matrix loop over the whole owned group, so
# the batched column is the only arm that changes their cost -- and it is free to score,
# being measured for the printed table already. The shard_count == 1 gate is not a
# convenience: at partition_dim is None ``newton_schulz_tp_batched`` short-circuits to
# plain ``newton_schulz`` on the stack, so the ``distributed`` collective route -- and with
# it ``distributed_normalize_p2``, which would share ONE Frobenius norm across the B
# matrices of a 3-D stack -- is never reached. On a sharded shape that hazard is real,
# which is why the gate is asserted below rather than left to the argmin.
PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY = "per_shape_sub_all_batch_set"
# The NS-step-fusion candidate (--fused-ns-kernel). Same subgroup deal, same batched
# Newton-Schulz, same chunking -- the ONLY difference is that the per-chunk call goes to
# ``kernels.fused_ns.fused_newton_schulz_batched`` instead of the library
# ``newton_schulz``, which fuses the fp32 normalize + bf16 cast prologue and the bf16->fp32
# cast + destination store epilogue that surround the identical 5-step chain. Timed as its
# own column beside the unfused one, from the SAME job over the SAME tensors under the SAME
# composition, so the pair is like-for-like and neither arm is credited with a composition
# change. Suffix, not prefix, so ``policy_mode``'s BATCH_SUBGROUP_PREFIX test still fires.
FUSED_SUFFIX = "_fused"
# The scored pair: reference is the accepted state PER_SHAPE_BATCH_SUBGROUP_POLICY.
PER_SHAPE_BATCH_SUBGROUP_FUSED_POLICY = "per_shape_batch_sub_fused_set"
# The replicated-batch FUSION candidate (--batch-replicated-fused). The shard_count == 1
# shapes are the only ones whose banked arm is the plain ``duplicated_batch`` column, and
# that column never received the fusion above: at partition_dim is None
# ``newton_schulz_tp_batched`` short-circuits to the library ``newton_schulz`` on the 3-D
# stack, so it still pays the fp32 normalize + bf16 cast prologue and the bf16 -> fp32 cast
# + store epilogue as separate HBM passes. This column is the SAME batched call with the
# same chunking, same 5 steps, same polar_express coefficients and the same batched SYRK
# path -- only the per-chunk entry point differs, exactly as ``FUSED_SUFFIX`` means
# everywhere else. Timed as its own column beside the unfused one, from the SAME job over
# the SAME tensors under the SAME composition. Written as a suffix on BATCH_POLICY so
# ``policy_mode`` strips it to ``duplicated`` and the FLOP model is unchanged.
BATCH_FUSED_POLICY = BATCH_POLICY + FUSED_SUFFIX
# The scored pair: reference is the accepted state PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY.
PER_SHAPE_SUBGROUP_ALL_BATCH_FUSED_POLICY = "per_shape_sub_all_batch_fused_set"
# The exchange-restructure candidate (--pipelined-exchange). Same subgroup deal, same
# batched Newton-Schulz, same coefficients -- the ONLY difference is HOW the two subgroup
# exchanges are staged and scheduled: block ownership makes the send/receive buffers views
# of the caller's own tensors (two of the four full-size staging copies stop existing), and
# the per-chunk exchanges are issued async so the wire overlaps the compute of a different
# chunk. Timed as its own column beside the arm it substitutes, from the SAME job over the
# SAME tensors under the SAME composition. Suffix goes AFTER FUSED_SUFFIX so a pipelined
# fused arm reads ``duplicated_batch_sub_g1_fused_pipe``; policy_mode strips both.
PIPE_SUFFIX = "_pipe"
# The scored pair: reference is whichever of the two accepted arms the run has enabled --
# PER_SHAPE_BATCH_SUBGROUP_FUSED_POLICY when --fused-ns-kernel is on, else
# PER_SHAPE_BATCH_SUBGROUP_POLICY.
PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY = "per_shape_batch_sub_pipe_set"
# The self-block-elision candidate (--elide-self-block). Same subgroup deal, same batched
# Newton-Schulz, same coefficients, same block ownership, same pipeline depth -- the ONLY
# difference is that the ``s == rank`` entry of each per-chunk exchange is dropped from the
# collective and served by a local device write into the buffer that consumes it. Timed as
# its own column beside the pipelined arm it substitutes, from the SAME job over the SAME
# tensors under the SAME composition. Suffix goes AFTER PIPE_SUFFIX so an elided fused
# pipelined arm reads ``duplicated_batch_sub_g1_fused_pipe_selfelide``; policy_mode strips
# all three.
ELIDE_SUFFIX = "_selfelide"
# The scored pair: reference is PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY, the state stage-3
# phase 3 banked.
PER_SHAPE_BATCH_SUBGROUP_ELIDE_POLICY = "per_shape_batch_sub_elide_set"
# The return-wire-precision candidate (--subgroup-wire-modes). Same subgroup deal, same
# batched Newton-Schulz, same coefficients, same block ownership, same pipeline depth,
# same self-block treatment -- the ONLY difference is the TRANSPORT ENCODING of the
# OUTPUT leg: the reply that carries each peer's row block of the orthogonalized result
# travels as bf16 and is widened into the fp32 ``result`` inside the timed region.
#
# This is BITWISE, not a tolerance argument. ``fused_newton_schulz_batched`` keeps ``X``
# in bf16 through all five polar_express steps and widens only at its ``_cast_into``
# epilogue (kernels/fused_ns.py), so EVERY value it returns is exactly bf16-representable
# -- measured, not asserted: stage-4's ``preflight_wire_bf16.csv`` reports
# ``output_bf16_exact_frac = 1.000000`` on all seven hot shapes, both axes. Narrowing that
# value to bf16 and widening it back is the identity, so the returned tensor must be
# BIT-IDENTICAL to the arm this one suffixes.
#
# Only the OUTPUT leg is offered. The input leg was refuted at zero scored cost by the
# same pre-flight (EGTP max_abs_diff 1.221e-3 against the 1e-3 equivalence gate), and it
# is not bitwise: it would perturb every entry before the Newton-Schulz prologue.
#
# Suffix goes LAST, after ELIDE_SUFFIX, so an arm reads
# ``duplicated_batch_sub_g1_fused_pipe_selfelide_wbf16out``; ``policy_mode`` strips all
# four. Off by default, so the scored policy set is byte-identical to the accepted state
# unless the flag is passed.
#
# mode -> dtype of the output leg. Deliberately NOT ``WIRE_MODES`` (the GTP
# fused-exchange dict, which also offers input-leg modes): an input-leg mode on this
# exchange is a hard ValueError here rather than a silently-fp32 no-op.
SUBGROUP_WIRE_MODES = {"bf16out": torch.bfloat16}
# The scored pair: reference is whichever of the accepted arms the run has enabled --
# PER_SHAPE_BATCH_SUBGROUP_ELIDE_POLICY when --elide-self-block is on, else the pipelined
# policy.
PER_SHAPE_BATCH_SUBGROUP_WIRE_POLICY = "per_shape_batch_sub_wire_set"


def subgroup_policy(subgroup_size: int) -> str:
    return f"{SUBGROUP_PREFIX}{subgroup_size}"


def batch_subgroup_policy(
    subgroup_size: int, fused: bool = False, pipelined: bool = False,
    elide_self: bool = False, wire: str = "",
) -> str:
    return (
        f"{BATCH_SUBGROUP_PREFIX}{subgroup_size}"
        + (FUSED_SUFFIX if fused else "")
        + (PIPE_SUFFIX if pipelined else "")
        + (ELIDE_SUFFIX if elide_self else "")
        + (f"{WIRE_SUFFIX}{wire}" if wire else "")
    )


# --------------------------------------------------------------------------------------
# Pipelined, staging-free variant of the g == 1 subgroup exchange (--pipelined-exchange).
#
# ``newton_schulz_tp_subgroup``'s monolithic path is strictly serial -- ONE
# ``all_to_all_single`` over the whole owned group, then every compute chunk, then ONE
# ``all_to_all_single`` back -- and it stages four FULL-SIZE copies through HBM to get
# there: the ``cat`` of strided slices on the way in, the ``permute -> reshape`` after the
# input exchange, a ``stack``-per-destination python loop plus ``cat`` on the way out, and
# a strided scatter into ``result``.
#
# Two mechanisms, one diff:
#
#   1. BLOCK ownership instead of round-robin. Matrix ``j`` is owned by subgroup
#      ``j // mine`` rather than ``j % k``. Which subgroup owns which matrix is internal to
#      this function -- the return value is still this rank's slice of EVERY matrix in
#      INPUT order -- but with contiguous blocks the send side of the input exchange and
#      the receive side of the output exchange become exact VIEWS of ``stack`` and
#      ``result``, so two of the four staging copies stop existing rather than being made
#      cheaper. This is only legal at ``subgroup_size == 1``, where each rank is its own
#      duplication group and so appears exactly ONCE in the destination list; at g > 1 the
#      same block goes to g destinations and the per-peer views would not be disjoint.
#   2. The LIST form of ``all_to_all`` (one tensor per peer instead of one packed buffer),
#      which is what lets those views be handed to NCCL directly, issued with
#      ``async_op=True`` so the exchange runs on the process group's own stream while the
#      compute stream runs the batched Newton-Schulz of a DIFFERENT chunk. The loop is a
#      two-deep software pipeline: the input exchange of chunk c+1 and the output exchange
#      of chunk c-1 are both in flight while chunk c computes.
#
# The two copies that remain are the ``(peer, matrix)`` transposes no exchange layout can
# avoid -- a batched Newton-Schulz needs the matrix index outermost, an all-to-all needs
# the peer index outermost -- and they are now INSIDE the pipeline, so they overlap the
# wire instead of adding to it. The per-destination ``stack``/``cat`` python loop (96
# iterations per region on the EGTP axis) collapses to one strided ``copy_`` per chunk.
#
# The arithmetic is untouched: each owned matrix is still assembled with its row blocks in
# RANK order -- exactly what ``duplicated``'s ``all_gather`` + ``cat`` produces -- and run
# through the same batched ``newton_schulz`` (or the same fused entry point), same step
# count, same coefficients, same SYRK path. Only the batch a chunk carries changes, from
# ``batch_chunk`` to ``ceil(mine / pipe_chunks)``, and every matrix in it is independent of
# its neighbours in the batch.
#
# Collective count per region goes from 2 to ``2 * pipe_chunks``; at the measured
# alpha_ib = 75.681 us that is +0.45 ms per region at pipe_chunks = 4, which is why the
# chunk count is deliberately small.
# --------------------------------------------------------------------------------------


def _subgroup_pipelined(
    stack: torch.Tensor,
    steps: int,
    coefficient_type: str,
    tp_group,
    use_syrk: bool,
    fused: bool,
    pipe_chunks: int,
    elide_self: bool = False,
    wire_out=None,
) -> torch.Tensor:
    """``newton_schulz_tp_subgroup`` at ``subgroup_size == 1``, pipelined and staging-free.

    Preconditions, all checked by the caller before dispatch here: ``partition_dim == 0``,
    ``tp_mode == "duplicated"``, ``subgroup_size == 1``, ``batched=True`` and
    ``count % world == 0``. Returns the same ``(count, rows, cols)`` this rank's-slice
    tensor, in input order, that the monolithic path returns.
    """
    world = tp_group.size()
    rank = tp_group.rank()
    count, rows, cols = stack.shape
    mine = count // world

    result = torch.empty((count, rows, cols), dtype=stack.dtype, device=stack.device)
    if mine == 0:
        return result

    n_chunks = max(1, min(int(pipe_chunks), mine))
    chunk = (mine + n_chunks - 1) // n_chunks
    windows = [(a, min(a + chunk, mine)) for a in range(0, mine, chunk)]

    # Peer-major landing buffer for the input exchange, matrix-major buffers for compute,
    # peer-major again for the output exchange. Only ``recv``/``sendout`` are full size --
    # they hold the in-flight chunks -- while the compute buffers are one chunk each.
    recv = torch.empty((world, mine, rows, cols), dtype=stack.dtype, device=stack.device)
    # Transport encoding of the OUTPUT leg only. ``stack.dtype`` (fp32) is the COMPUTE
    # dtype and never changes: ``gx`` stays fp32 so ``fused_ns._supported`` still holds and
    # the accepted fusion cannot silently un-bank, and ``result`` stays fp32 so the region
    # returns exactly the tensor type it always did. Bitwise, because the fused kernel's
    # output is exactly bf16-representable -- see SUBGROUP_WIRE_MODES.
    out_wire = wire_out or stack.dtype
    sendout = torch.empty((world, mine, rows, cols), dtype=out_wire, device=stack.device)
    # Landing buffer for a narrowed output leg. The reply cannot land directly in
    # ``result`` when the wire is narrower than it, so this is the one genuinely new
    # full-size pass the arm pays, and it is INSIDE the timed region by construction.
    recvout = (
        None if out_wire is stack.dtype
        else torch.empty((world, mine, rows, cols), dtype=out_wire, device=stack.device)
    )
    gx = torch.empty((chunk, world * rows, cols), dtype=stack.dtype, device=stack.device)
    tmp = (
        torch.empty((chunk, world * rows, cols), dtype=stack.dtype, device=stack.device)
        if fused else None
    )

    # ``elide_self``: the s == rank entry of both exchanges is this rank's OWN block --
    # ``stack[rank*mine+a : rank*mine+b]`` on the way in, row block ``rank`` of the result
    # on the way out. It is already in this device's HBM, so handing it to the collective
    # only asks NCCL to move it back to where it is. Under
    # ``NCCL_P2P_DISABLE=1``/``NCCL_SHM_DISABLE=1``/``NCCL_NVLS_ENABLE=0`` -- the settings
    # that make the 2-rank proxy take the network path -- whether that self connection is
    # still serviced locally is exactly the question ``probe_selfblock_transport.sbatch``
    # answers. With the flag on it cannot be on the wire either way: the collective becomes
    # a peer-only ``batch_isend_irecv`` over the (world - 1) real peers, and the own block
    # is written straight into the buffer that consumes it -- into ``gx`` on the way in and
    # into ``result`` on the way out, so NOT ONE extra byte of HBM traffic is added
    # relative to the transposes the monolithic path already pays.
    #
    # Peers are addressed by GLOBAL rank via ``get_global_rank(tp_group, s)``, which is what
    # ``P2POp`` takes; at ``egtp = 2`` this is one isend + one irecv per chunk per leg.
    # Arithmetic, ordering and the returned tensor are unchanged: the same row block from
    # the same source lands in the same slot, so the result must be BIT-IDENTICAL.
    peers = [s for s in range(world) if s != rank] if elide_self else list(range(world))
    global_peer = (
        {s: torch.distributed.get_global_rank(tp_group, s) for s in peers}
        if elide_self else {}
    )

    def issue_input(ci):
        a, b = windows[ci]
        # Send peer d the block of matrices d owns, window [a, b): a contiguous view of
        # ``stack`` -- no packing copy. Receive from peer s that peer's shard of MY
        # matrices, window [a, b), landing peer-major in ``recv``.
        if elide_self:
            if not peers:
                return []
            ops = []
            for s in peers:
                ops.append(torch.distributed.P2POp(
                    torch.distributed.irecv, recv[s, a:b], global_peer[s], tp_group))
                ops.append(torch.distributed.P2POp(
                    torch.distributed.isend,
                    stack[s * mine + a : s * mine + b], global_peer[s], tp_group))
            return torch.distributed.batch_isend_irecv(ops)
        return [torch.distributed.all_to_all(
            [recv[s, a:b] for s in range(world)],
            [stack[d * mine + a : d * mine + b] for d in range(world)],
            group=tp_group,
            async_op=True,
        )]

    in_works = [None] * len(windows)
    in_works[0] = issue_input(0)
    # (works, a, b) per in-flight output window. The window bounds are carried because a
    # narrowed wire needs a widening pass once that window has landed.
    out_works: List[Tuple[list, int, int]] = []

    def drain_output(pending):
        """Wait one output window and, on a narrowed wire, widen it into ``result``."""
        works, a, b = pending
        for work in works:
            work.wait()
        if recvout is None:
            return
        # The upcast. One strided pass per window over the peer blocks only -- the self
        # block (under ``elide_self``) was written into ``result`` in fp32 already and is
        # not on the wire. Kept per-window rather than one pass at the end so it overlaps
        # the next window's exchange instead of trailing it.
        for s in peers:
            result[s * mine + a : s * mine + b].copy_(recvout[s, a:b])

    for ci, (a, b) in enumerate(windows):
        # Prefetch: chunk ci+1's input exchange goes on the wire BEFORE this chunk's
        # compute is issued, so it overlaps it. Every rank issues the same sequence.
        if ci + 1 < len(windows):
            in_works[ci + 1] = issue_input(ci + 1)
        # Retire output windows at a lag of two, matching the input pipeline's depth: the
        # newest one was issued at the end of the previous iteration and is left in
        # flight, so nothing here waits on a wire that has only just been handed to NCCL.
        while len(out_works) > 1:
            drain_output(out_works.pop(0))
        for work in in_works[ci]:
            work.wait()
        in_works[ci] = None
        n = b - a
        # (peer, matrix, rows, cols) -> (matrix, peer * rows, cols): per matrix the shards
        # concatenated in RANK order, identical to duplicated's all_gather + cat(dim=0).
        gx_c = gx[:n]
        if elide_self:
            # Same transpose, one peer slice at a time, with the self slice taken from
            # ``stack`` instead of from ``recv``. Same bytes copied, same RANK order.
            gx_v = gx_c.view(n, world, rows, cols)
            for s in range(world):
                if s == rank:
                    gx_v[:, s].copy_(stack[rank * mine + a : rank * mine + b])
                else:
                    gx_v[:, s].copy_(recv[s, a:b])
        else:
            gx_c.view(n, world, rows, cols).copy_(recv[:, a:b].transpose(0, 1))

        if fused:
            fused_newton_schulz_batched(
                gx_c, steps, coefficient_type, use_syrk=use_syrk, out=tmp[:n],
            )
            y = tmp[:n]
        else:
            y = newton_schulz(gx_c, steps, coefficient_type, use_syrk=use_syrk)

        # Inverse transpose: row block d of every result goes to peer d, peer-major.
        if elide_self:
            # Row block ``rank`` is this rank's own answer for its own matrices: write it
            # straight into ``result`` rather than staging it for a self-send.
            y_v = y.view(n, world, rows, cols)
            for d in range(world):
                if d == rank:
                    result[rank * mine + a : rank * mine + b].copy_(y_v[:, d])
                else:
                    sendout[d, a:b].copy_(y_v[:, d])
        else:
            sendout[:, a:b].copy_(y.view(n, world, rows, cols).transpose(0, 1))
        del y
        # Peer s owns matrices [s * mine, (s + 1) * mine); its reply for window [a, b)
        # lands DIRECTLY in the output slice -- no scatter copy. On a narrowed wire it
        # lands in ``recvout`` instead and ``drain_output`` widens it into that same slice.
        def out_dst(s):
            return (
                result[s * mine + a : s * mine + b] if recvout is None
                else recvout[s, a:b]
            )

        if elide_self:
            ops = []
            for s in peers:
                ops.append(torch.distributed.P2POp(
                    torch.distributed.irecv, out_dst(s), global_peer[s], tp_group))
                ops.append(torch.distributed.P2POp(
                    torch.distributed.isend, sendout[s, a:b], global_peer[s], tp_group))
            if ops:
                out_works.append((torch.distributed.batch_isend_irecv(ops), a, b))
        else:
            out_works.append((
                [torch.distributed.all_to_all(
                    [out_dst(s) for s in range(world)],
                    [sendout[d, a:b] for d in range(world)],
                    group=tp_group,
                    async_op=True,
                )],
                a,
                b,
            ))

    while out_works:
        drain_output(out_works.pop(0))
    return result


def newton_schulz_tp_subgroup(
    stack: torch.Tensor,
    steps: int,
    coefficient_type: str,
    tp_group,
    partition_dim: int | None,
    tp_mode: str,
    use_syrk: bool = False,
    subgroup_size: int = 0,
    batched: bool = False,
    batch_chunk: int = 0,
    fused: bool = False,
    pipelined: bool = False,
    pipe_chunks: int = 4,
    elide_self: bool = False,
    wire: str = "",
) -> torch.Tensor:
    """``duplicated`` over ``world/subgroup_size`` disjoint subgroups of the owned set.

    ``stack`` is ``(count, rows, cols)``: the whole owned group of ONE shape, each matrix
    sharded the same way across ``tp_group``. Returns ``(count, rows, cols)``: this rank's
    slice of every matrix, in the input order -- i.e. exactly what looping
    ``newton_schulz_tp(stack[j], tp_mode="duplicated")`` returns, stacked.

    Only ``partition_dim == 0`` is supported, which is the axis this benchmark shards on.
    """
    if tp_mode != "duplicated":
        raise ValueError(f"subgroup duplication is a 'duplicated' variant, got {tp_mode!r}")
    if partition_dim is None:
        # Replicated weight: no shards to redistribute, so there is no duplication group to
        # shrink. Same non-TP fallback newton_schulz_tp takes.
        return torch.stack([
            newton_schulz(stack[j], steps, coefficient_type, use_syrk=use_syrk)
            for j in range(stack.size(0))
        ])
    if partition_dim != 0:
        raise ValueError(f"subgroup duplication supports partition_dim=0, got {partition_dim}")

    world = tp_group.size()
    rank = tp_group.rank()
    if subgroup_size <= 0 or world % subgroup_size != 0:
        raise ValueError(f"subgroup_size {subgroup_size} must divide world {world}")
    groups = world // subgroup_size          # k
    if groups == 1:
        raise ValueError("subgroup_size == world is plain 'duplicated'")

    stack = stack.contiguous()
    count, rows, cols = stack.shape

    if elide_self and not pipelined:
        # Elision is a property of the pipelined (list-form, block-ownership) exchange
        # only; a silent no-op on the monolithic path would let the arm be scored while
        # running code that never elided anything.
        raise ValueError("self-block elision requires the pipelined subgroup exchange")

    if wire and not pipelined:
        # A wire encoding is a property of the pipelined (list-form) exchange only. A
        # silent fp32 no-op on the monolithic path would let the arm be scored while
        # running code that never narrowed anything -- the same hazard the elision guard
        # above closes.
        raise ValueError("subgroup wire encoding requires the pipelined subgroup exchange")
    if wire and wire not in SUBGROUP_WIRE_MODES:
        raise ValueError(
            f"unknown subgroup wire mode {wire!r}; known: "
            f"{sorted(SUBGROUP_WIRE_MODES)}"
        )
    if wire and not fused:
        # The bitwise property is a property of the FUSED kernel's output (every value it
        # returns is exactly bf16-representable because it holds X in bf16 across all five
        # steps). The library path's fp32 output carries no such guarantee, so narrowing
        # its return leg would be a tolerance change wearing a bitwise arm's name.
        raise ValueError(
            "subgroup wire encoding is defined on the fused compute path only: the "
            "bitwise property comes from fused_newton_schulz_batched's bf16 X"
        )

    if pipelined:
        # Same deal, same math, restructured exchange -- see _subgroup_pipelined. Every
        # precondition is a hard error rather than a silent fallback, so the arm can never
        # be scored while quietly running the monolithic path.
        if not batched:
            raise ValueError("pipelined subgroup exchange requires the batched path")
        if subgroup_size != 1:
            raise ValueError(
                "pipelined subgroup exchange is defined at subgroup_size == 1 only, "
                f"got {subgroup_size}"
            )
        if count % world != 0:
            raise ValueError(
                f"pipelined subgroup exchange needs count ({count}) divisible by world "
                f"({world}): the block deal has no uneven case"
            )
        if fused and not HAVE_FUSED_NS:
            raise RuntimeError(
                "fused Newton-Schulz requested but kernels.fused_ns is unavailable"
            )
        if elide_self and world < 2:
            raise ValueError("self-block elision needs at least one real peer")
        return _subgroup_pipelined(
            stack, steps, coefficient_type, tp_group, use_syrk, fused, pipe_chunks,
            elide_self=elide_self,
            wire_out=SUBGROUP_WIRE_MODES[wire] if wire else None,
        )

    my_sub = rank // subgroup_size           # s_r
    my_slot = rank % subgroup_size           # i_r, this rank's send-duty slot
    # Matrix j belongs to subgroup j % k, so subgroup s owns stack[s::k].
    owned = [(count - s + groups - 1) // groups for s in range(groups)]
    mine = owned[my_sub]

    # ---- input exchange: give each subgroup every shard of the matrices it owns --------
    send = torch.cat(
        [stack[d // subgroup_size :: groups].reshape(-1, cols) for d in range(world)]
    )
    recv = torch.empty((world * mine * rows, cols), dtype=stack.dtype, device=stack.device)
    torch.distributed.all_to_all_single(
        recv,
        send,
        output_split_sizes=[mine * rows] * world,
        input_split_sizes=[owned[d // subgroup_size] * rows for d in range(world)],
        group=tp_group,
    )
    del send
    # (src, matrix, rows, cols) -> per matrix, the shards concatenated in rank order, which
    # is what duplicated's ``all_gather`` + ``cat(dim=partition_dim)`` produces.
    global_x = recv.view(world, mine, rows, cols).permute(1, 0, 2, 3).reshape(
        mine, world * rows, cols
    )

    # ---- compute: the SAME newton_schulz duplicated runs, on 1/k as many matrices -------
    # ``batched=False`` is the phase-2/3 path, untouched: one 2-D call per owned matrix.
    # ``batched=True`` stacks the owned matrices into the leading dim of ONE call per
    # chunk -- the same dispatch ``newton_schulz_tp_batched`` already uses (3-D input ->
    # ``batched_newton_schulz_step_tsyrk`` / ``batched_tsyrk_ex``), same coefficients, same
    # step count, same SYRK path, no reassociation across matrices. It is what makes
    # owner-computes composable with the accepted batched arm: the batch simply becomes
    # ``count/k`` matrices per rank instead of ``count``.
    if batched:
        chunk = batch_chunk if (batch_chunk and batch_chunk > 0) else max(mine, 1)
        full_t = torch.empty_like(global_x)
        if fused:
            # Same chain, same coefficients, same batched SYRK path -- the fused entry
            # point only removes the fp32 prologue/epilogue passes AND this loop's own
            # ``full_t[a:a+chunk] = `` store, by writing the fp32 result straight into the
            # destination slice (un-transposed for the tall shapes, which is where that
            # store is strided today). It falls back to the library call itself if any of
            # its preconditions do not hold, so the arm is never silently skipped.
            if not HAVE_FUSED_NS:
                raise RuntimeError(
                    "fused Newton-Schulz requested but kernels.fused_ns is unavailable"
                )
            for a in range(0, mine, chunk):
                fused_newton_schulz_batched(
                    global_x[a : a + chunk], steps, coefficient_type,
                    use_syrk=use_syrk, out=full_t[a : a + chunk],
                )
        else:
            for a in range(0, mine, chunk):
                full_t[a : a + chunk] = newton_schulz(
                    global_x[a : a + chunk], steps, coefficient_type, use_syrk=use_syrk
                )
        # Views, not copies: the output exchange below indexes 2-D matrices either way.
        full = [full_t[m] for m in range(mine)]
    else:
        full = [
            newton_schulz(global_x[m], steps, coefficient_type, use_syrk=use_syrk)
            for m in range(mine)
        ]
        full_t = None
    del recv, global_x

    # ---- output exchange: hand every rank its row-slice of every matrix ----------------
    # This rank serves destinations ``d`` with ``d % subgroup_size == my_slot``: k of them,
    # one per (destination subgroup), each getting ``mine`` slices of ``rows`` rows.
    if full:
        send_out = torch.cat(
            [
                torch.stack([y[d * rows : (d + 1) * rows] for y in full]).reshape(-1, cols)
                for d in range(my_slot, world, subgroup_size)
            ]
        )
    else:
        send_out = torch.empty((0, cols), dtype=stack.dtype, device=stack.device)
    del full, full_t
    out = torch.empty((count * rows, cols), dtype=stack.dtype, device=stack.device)
    torch.distributed.all_to_all_single(
        out,
        send_out,
        # Received from src = s * subgroup_size + my_slot for s = 0..k-1, ascending, so the
        # blocks arrive in subgroup order: owned[0] matrices, then owned[1], ...
        output_split_sizes=[
            owned[src // subgroup_size] * rows if src % subgroup_size == my_slot else 0
            for src in range(world)
        ],
        input_split_sizes=[
            mine * rows if d % subgroup_size == my_slot else 0 for d in range(world)
        ],
        group=tp_group,
    )
    del send_out
    result = torch.empty((count, rows, cols), dtype=stack.dtype, device=stack.device)
    offset = 0
    for s in range(groups):
        block = out[offset : offset + owned[s] * rows].view(owned[s], rows, cols)
        result[s::groups] = block
        offset += owned[s] * rows
    return result


# --------------------------------------------------------------------------------------
# Fused, CROSS-SHAPE subgroup exchange (--fuse-shape-exchanges).
#
# ``newton_schulz_tp_subgroup`` deals ONE shape's owned group across the k = world/g
# subgroups and pays its own pair of ``all_to_all_single`` for it. A rank that owns five
# distinct dense shapes therefore runs five independent subgroup regions back to back,
# each with its own exchange and -- more importantly -- its own INDEPENDENT deal.
#
# This function fuses all of them into ONE region with ONE pair of exchanges. Two things
# change, and they are deliberately separable so the measurement can attribute the delta:
#
#   1. THE EXCHANGE IS FUSED. Every shape's payload for a given peer is concatenated into
#      that peer's slot of a single ``all_to_all_single``, so the region pays one pair of
#      collectives instead of one pair per shape. Per-shape splits differ in size (and in
#      g, since each shape may have banked a different duplication degree), which is
#      exactly what ``input_split_sizes`` / ``output_split_sizes`` express. With
#      ``balanced=False`` the OWNERSHIP is bit-for-bit the round-robin
#      ``newton_schulz_tp_subgroup`` already uses (matrix j of shape i -> subgroup j % k),
#      so this arm isolates the fused exchange and nothing else.
#   2. WITH ``balanced=True``, THE DEAL IS ALSO POOLED ACROSS SHAPES. A per-shape deal
#      strands work: a shape with 3 owned matrices dealt over k = 16 subgroups leaves 13
#      subgroups idle for the whole region, and since the closing collective is a barrier
#      in effect, the region costs one WHOLE matrix of Newton-Schulz no matter how few
#      matrices it carries. That is what the measured data shows -- 10240x8192 at count 3
#      costs 10.857 ms against a 10.260 ms single-matrix ``duplicated`` time, and is flat
#      in g (10.857 / 10.738 / 10.910 at g = 4/8/16) precisely because every g leaves
#      exactly one matrix per busy subgroup. Fusing the region makes those 3 stranded
#      matrices absorbable: the pooled set is dealt by greedy LPT on ``ns_cost``, the same
#      cost model and the same tie-breaking ``owned_matrices`` uses for the outer
#      data-parallel assignment, so the region's makespan approaches the pooled mean
#      instead of the sum of five independent maxima.
#
# NUMERICS ARE UNTOUCHED under both deals. Which subgroup owns which matrix is internal:
# every matrix is still all-gathered in RANK order and orthogonalized WHOLE by exactly the
# same ``newton_schulz`` call, same step count, same ``polar_express`` coefficients, same
# SYRK path, and each shape's return value is still this rank's slice of every matrix in
# INPUT order. Only the (subgroup -> matrix) assignment and the packing of the two
# collectives change.
# --------------------------------------------------------------------------------------

FUSED_EXCHANGE_PREFIX = "fuse_sub_g"
# Suffix for the pooled-LPT deal; without it the arm is the round-robin deal, i.e. the
# fused exchange in isolation.
BALANCED_SUFFIX = "_bal"
# The scored policy: the accepted state with the subgroup-banked shapes' per-shape regions
# replaced by ONE fused region. Shapes the reference did not bank on a subgroup arm are
# untouched, so the policy can never re-bank anything the reference already chose.
PER_SHAPE_FUSED_EXCHANGE_POLICY = "per_shape_fuse_sub_set"

# Wire precision of the two fused-exchange legs (--fused-wire-modes). The suffix goes LAST,
# after the pipeline depth, so an arm reads ``fuse_sub_g4_bal_pipe2_wbf16``; it is a
# TRANSPORT encoding only -- the deal, the matrices, the step count, the coefficients and
# the SYRK path are identical to the arm it suffixes, and every widening/narrowing rides a
# copy the pipelined path already pays. ``bf16in`` is the input leg alone, which carries
# 4/5 of the region's wire bytes at subgroup_size = 4 (see WIRE_MODES below).
WIRE_SUFFIX = "_w"
# mode -> (input-leg dtype, output-leg dtype); "" is the unmodified fp32 wire.
WIRE_MODES = {
    "bf16": (torch.bfloat16, torch.bfloat16),
    "bf16in": (torch.bfloat16, None),
    "bf16out": (None, torch.bfloat16),
}


def fused_exchange_policy(
    subgroup_size: int, balanced: bool = False, pipe_chunks: int = 0, wire: str = ""
) -> str:
    return (
        f"{FUSED_EXCHANGE_PREFIX}{subgroup_size}"
        + (BALANCED_SUFFIX if balanced else "")
        + (f"{PIPE_SUFFIX}{pipe_chunks}" if pipe_chunks else "")
        + (f"{WIRE_SUFFIX}{wire}" if wire else "")
    )


def fused_subgroup_deal(full_shapes, counts, groups: int, balanced: bool):
    """``owners[i][j]`` = subgroup owning matrix ``j`` of shape ``i``.

    Pure arithmetic over shapes and counts, so it is IDENTICAL on every rank of the group
    -- which is what keeps the two collectives in lockstep.
    """
    if not balanced:
        # Exactly what newton_schulz_tp_subgroup does per shape today.
        return [[j % groups for j in range(count)] for count in counts]
    loads = [0] * groups
    owners = [[0] * count for count in counts]
    items = []
    for index, ((rows, cols), count) in enumerate(zip(full_shapes, counts)):
        big, small = max(rows, cols), min(rows, cols)
        cost = big * small * small          # ns_cost of the full, all-gathered matrix
        for j in range(count):
            items.append((cost, index, j))
    # Greedy LPT, ties broken by (shape, matrix) then by lowest subgroup id: deterministic.
    for cost, index, j in sorted(items, key=lambda t: (-t[0], t[1], t[2])):
        pick = min(range(groups), key=lambda s: (loads[s], s))
        loads[pick] += cost
        owners[index][j] = pick
    return owners


# --------------------------------------------------------------------------------------
# Pipelined, staging-lean variant of the CROSS-SHAPE fused subgroup exchange
# (--pipeline-fused-exchange). This is the port of the accepted EGTP restructure
# (``_subgroup_pipelined``, stage-3 phase 3) onto ``newton_schulz_tp_subgroup_fused``,
# which is the last hot region still running the pre-stage-3 shape: ONE
# ``all_to_all_single`` in, every matrix computed, ONE ``all_to_all_single`` out, with a
# python ``cat``/``stack`` staging loop on each side and a PER-MATRIX ``newton_schulz``
# loop in between.
#
# Three mechanisms, one diff -- all three already measured on the EGTP axis:
#
#   1. ASYNC CHUNKED EXCHANGE. The subgroup's pooled share is split into ``pipe_chunks``
#      windows of its flat (shape-major) item list and each window gets its own pair of
#      exchanges, issued ``async_op=True``. Chunk c+1's input exchange goes on the wire
#      before chunk c's compute is issued and every output exchange is left in flight, so
#      the wire overlaps a different chunk's Newton-Schulz instead of bracketing all of it.
#      The window boundaries are pure arithmetic over ``owned``, which every rank computes
#      identically, so the two collectives stay in lockstep.
#
#   2. STAGING-LOOP ELISION. The monolithic send buffer is built per DESTINATION RANK, but
#      what a destination needs depends only on its SUBGROUP -- so at ``subgroup_size = g``
#      the buffer holds every byte ``g`` times, materialised by a
#      ``groups * n_shapes * g``-iteration python gather loop feeding one giant
#      ``torch.cat``. The LIST form of ``all_to_all`` takes one tensor per peer, so the
#      per-subgroup block is staged ONCE and handed to all ``g`` of its ranks: the send
#      buffer shrinks by ``g`` (4x on the GTP axis) and the ``cat`` disappears, replaced by
#      ``index_select(..., out=)`` writing straight into its final slot. On the way out the
#      ``groups * n_shapes`` ``torch.stack`` python loop plus ``cat`` collapses to ONE
#      strided ``copy_`` per shape per chunk, because the destinations this rank serves are
#      the arithmetic progression ``my_slot :: subgroup_size`` -- a pure view of the result.
#      The block-ownership VIEW trick of ``_subgroup_pipelined`` is NOT reused: it is only
#      legal at ``g == 1``, where a block has exactly one destination. Here the same block
#      goes to ``g`` destinations, so what is elided is the REPLICATION, not the copy.
#
#   3. BATCHED + FUSED NEWTON-SCHULZ. The monolithic path runs ``newton_schulz`` once per
#      owned matrix; here every matrix of one shape inside a chunk is one batched call into
#      ``kernels.fused_ns.fused_newton_schulz_batched``, which fuses the fp32
#      normalize/cast prologue and the bf16->fp32 cast-and-store epilogue around the
#      IDENTICAL 5-step chain. The gather buffer stays fp32 and contiguous so
#      ``fused_ns._supported`` holds and the fusion cannot silently un-bank; if it does not
#      hold, that entry point falls back to the very same library ``newton_schulz``.
#
# The arithmetic is untouched. Every matrix is still all-gathered in RANK order, still
# orthogonalized WHOLE, same 5 steps, same polar_express coefficients, same SYRK path, same
# deal (``fused_subgroup_deal`` is called once, before the branch), and each shape still
# returns this rank's slice of every matrix in INPUT order. Only how the two collectives
# are staged, scheduled and chunked changes.
#
# Collective count per region goes from 2 to ``2 * pipe_chunks``, and the scatter into
# ``results`` goes from ``groups * n_shapes`` writes to ``pipe_chunks`` times that, which
# is why the chunk count is small and is swept as its own arm rather than assumed.
# --------------------------------------------------------------------------------------


def _fused_exchange_pipelined(
    stacks: List[torch.Tensor],
    steps: int,
    coefficient_type: str,
    tp_group,
    use_syrk: bool,
    subgroup_size: int,
    groups: int,
    my_sub: int,
    my_slot: int,
    owned,
    counts,
    dims,
    elems,
    dtype,
    device,
    pipe_chunks: int,
    wire_in=None,
    wire_out=None,
) -> List[torch.Tensor]:
    """``newton_schulz_tp_subgroup_fused`` with the exchange chunked, async and unstaged.

    Every argument after ``use_syrk`` is the setup the caller already computed, passed in
    rather than recomputed so the DEAL is provably the same object both arms run.
    """
    world = tp_group.size()
    n_shapes = len(stacks)
    # Transport encoding of each leg, independent of the COMPUTE dtype (``dtype``), which
    # never changes: ``gx`` is fp32 so ``fused_ns._supported`` still holds, and ``results``
    # is fp32 so the region's return value has the same type it always had.
    wire_in = wire_in or dtype
    wire_out = wire_out or dtype

    # ---- windows: contiguous slices of each subgroup's flat, SHAPE-MAJOR item list ------
    # Shape-major is not a choice: it is the order the monolithic path's send buffer and
    # receive parse already use, so a window of it maps to a contiguous slice of
    # ``owned[i][s]`` for every shape at once.
    item_counts = [sum(len(owned[i][s]) for i in range(n_shapes)) for s in range(groups)]
    n_chunks = max(1, min(int(pipe_chunks), max(item_counts) if item_counts else 1))

    # slices[s][c][i] = (lo, hi) into owned[i][s] -- identical on every rank.
    slices = []
    for s in range(groups):
        cnt = item_counts[s]
        size = (cnt + n_chunks - 1) // n_chunks if cnt else 0
        wins = [(a, min(a + size, cnt)) for a in range(0, cnt, size)] if size else []
        wins += [(cnt, cnt)] * (n_chunks - len(wins))
        per_chunk = []
        for p, q in wins:
            base, cuts = 0, []
            for i in range(n_shapes):
                n_i = len(owned[i][s])
                cuts.append((min(max(p - base, 0), n_i), min(max(q - base, 0), n_i)))
                base += n_i
            per_chunk.append(cuts)
        slices.append(per_chunk)

    def blk(s: int, c: int) -> int:
        """Elements subgroup ``s``'s window ``c`` occupies per rank, across all shapes."""
        return sum((hi - lo) * elems[i] for i, (lo, hi) in enumerate(slices[s][c]))

    # ---- one device index tensor per shape, holding owned[i][0] ++ owned[i][1] ++ ... ---
    # Every per-(subgroup, chunk) index list is then a VIEW of it, so the gather and the
    # scatter cost 2 * n_shapes host-to-device copies per region, not 2 * groups *
    # n_shapes * n_chunks.
    perm, runbase = [], []
    for i in range(n_shapes):
        flat, bases, base = [], [], 0
        for s in range(groups):
            bases.append(base)
            flat.extend(owned[i][s])
            base += len(owned[i][s])
        perm.append(torch.tensor(flat, dtype=torch.long, device=device))
        runbase.append(bases)

    results = [
        torch.empty((counts[i], dims[i][0], dims[i][1]), dtype=dtype, device=device)
        for i in range(n_shapes)
    ]
    empty = torch.empty(0, dtype=wire_out, device=device)

    # Send-side narrowing for the input leg. ``index_select(..., out=)`` needs a matching
    # dtype, so the cast cannot ride the gather; it is done ONCE per shape per region here
    # rather than once per (subgroup, chunk) gather, which is the same number of elements
    # touched either way (every matrix is staged exactly once) but one allocation and one
    # pass instead of many. Freed as soon as the last window is on the wire.
    wire_stacks = list(stacks)
    if wire_in is not dtype:
        wire_stacks = [stack.to(wire_in) for stack in stacks]

    def issue_input(c: int):
        """Stage window ``c`` ONCE per destination SUBGROUP and put it on the wire."""
        sizes = [blk(s, c) for s in range(groups)]
        offs, total = [], 0
        for s in range(groups):
            offs.append(total)
            total += sizes[s]
        stage = torch.empty(total, dtype=wire_in, device=device)
        cursor = 0
        for s in range(groups):
            for i in range(n_shapes):
                lo, hi = slices[s][c][i]
                n = hi - lo
                if n == 0:
                    continue
                rows, cols = dims[i]
                span = n * elems[i]
                torch.index_select(
                    wire_stacks[i], 0,
                    perm[i][runbase[i][s] + lo : runbase[i][s] + hi],
                    out=stage[cursor : cursor + span].view(n, rows, cols),
                )
                cursor += span
        recv_n = blk(my_sub, c)
        recv = torch.empty(world * recv_n, dtype=wire_in, device=device)
        recv_v = recv.view(world, recv_n)
        # Destination d needs its SUBGROUP's block; the g ranks of a subgroup are handed
        # the SAME staged tensor rather than g copies of it.
        work = torch.distributed.all_to_all(
            [recv_v[src] for src in range(world)],
            [stage[offs[d // subgroup_size] : offs[d // subgroup_size] + sizes[d // subgroup_size]]
             for d in range(world)],
            group=tp_group,
            async_op=True,
        )
        return work, recv, recv_n, stage

    def compute_and_send(c: int, recv: torch.Tensor, recv_n: int):
        recv_v = recv.view(world, recv_n)
        ys, offset = [], 0
        for i in range(n_shapes):
            lo, hi = slices[my_sub][c][i]
            n = hi - lo
            if n == 0:
                continue
            rows, cols = dims[i]
            span = n * elems[i]
            # (src, matrix, rows, cols) -> (matrix, src * rows, cols): per matrix the
            # shards concatenated in RANK order, which is what duplicated's all_gather +
            # cat(dim=partition_dim) produces. ``unflatten`` keeps it a view, so this is
            # ONE strided copy, the same one the monolithic ``permute -> reshape`` pays.
            src = recv_v[:, offset : offset + span].unflatten(1, (n, rows, cols))
            offset += span
            gx = torch.empty((n, world * rows, cols), dtype=dtype, device=device)
            # ``gx`` is fp32 whatever the wire is: ``fused_ns._supported`` rejects a bf16
            # input outright, and the widening rides THIS copy -- the transposing pass the
            # pipeline already pays -- whose READ side halves when the wire is bf16.
            gx.view(n, world, rows, cols).copy_(src.permute(1, 0, 2, 3))
            y = fused_newton_schulz_batched(
                gx, steps, coefficient_type, use_syrk=use_syrk,
                out=torch.empty_like(gx),
            )
            del gx
            ys.append((i, n, y))

        # Narrowing for the output leg rides the strided ``copy_`` below, which the
        # pipelined path already pays; only its WRITE side changes width.
        sendout = torch.empty(groups * recv_n, dtype=wire_out, device=device)
        send_v = sendout.view(groups, recv_n)
        offset = 0
        for i, n, y in ys:
            rows, cols = dims[i]
            span = n * elems[i]
            # This rank serves destinations my_slot, my_slot + g, my_slot + 2g, ... -- an
            # arithmetic progression, so row block selection is a pure VIEW and the whole
            # per-destination ``stack``/``cat`` python loop is one strided copy.
            send_v[:, offset : offset + span].unflatten(1, (n, rows, cols)).copy_(
                y.view(n, world, rows, cols)[:, my_slot::subgroup_size].permute(1, 0, 2, 3)
            )
            offset += span
        del ys

        sizes = [blk(s, c) for s in range(groups)]
        offs, total = [], 0
        for s in range(groups):
            offs.append(total)
            total += sizes[s]
        outbuf = torch.empty(total, dtype=wire_out, device=device)
        out_list, in_list = [], []
        for r in range(world):
            if r % subgroup_size == my_slot:
                s = r // subgroup_size
                out_list.append(outbuf[offs[s] : offs[s] + sizes[s]])
                in_list.append(send_v[s])
            else:
                out_list.append(empty)
                in_list.append(empty)
        work = torch.distributed.all_to_all(
            out_list, in_list, group=tp_group, async_op=True
        )
        return work, outbuf, offs, sendout

    pending_in = [None] * n_chunks
    pending_in[0] = issue_input(0)
    pending_out = []
    for c in range(n_chunks):
        # Prefetch: window c+1's input exchange is on the wire BEFORE window c's compute
        # is issued, so it overlaps it. Every rank issues the same sequence.
        if c + 1 < n_chunks:
            pending_in[c + 1] = issue_input(c + 1)
        else:
            # Every window is staged; drop the bf16 send-side copy before compute peaks.
            wire_stacks = None
        work, recv, recv_n, stage = pending_in[c]
        pending_in[c] = None
        work.wait()
        del stage
        pending_out.append(compute_and_send(c, recv, recv_n))
        del recv, work

    for work, _outbuf, _offs, _sendout in pending_out:
        work.wait()
    # Scatter last: subgroup-major then shape-major, exactly the cursor walk the monolithic
    # path does, so ``results[i]`` is filled in INPUT order either way.
    for c, (_work, outbuf, offs, _sendout) in enumerate(pending_out):
        for s in range(groups):
            cursor = offs[s]
            for i in range(n_shapes):
                lo, hi = slices[s][c][i]
                n = hi - lo
                if n == 0:
                    continue
                rows, cols = dims[i]
                span = n * elems[i]
                block = outbuf[cursor : cursor + span].view(n, rows, cols)
                if block.dtype is not dtype:
                    # ``index_copy_`` requires self and source to share a dtype, so unlike
                    # the other three widenings this one cannot ride an existing copy: the
                    # output leg buys half its wire bytes at the price of one extra pass
                    # over the received block. That is why ``bf16in`` is a separate arm.
                    block = block.to(dtype)
                results[i].index_copy_(
                    0,
                    perm[i][runbase[i][s] + lo : runbase[i][s] + hi],
                    block,
                )
                cursor += span
    return results


def newton_schulz_tp_subgroup_fused(
    stacks: List[torch.Tensor],
    steps: int,
    coefficient_type: str,
    tp_group,
    use_syrk: bool = False,
    subgroup_size: int = 0,
    balanced: bool = False,
    pipelined: bool = False,
    pipe_chunks: int = 0,
    wire: str = "",
) -> List[torch.Tensor]:
    """Subgroup duplication over SEVERAL shapes at once, with one pair of exchanges.

    ``stacks[i]`` is ``(count_i, rows_i, cols_i)``: the whole owned group of shape ``i``,
    every matrix sharded on dim 0 across ``tp_group``. Returns a list of
    ``(count_i, rows_i, cols_i)`` tensors -- this rank's slice of every matrix, in input
    order -- i.e. exactly what calling ``newton_schulz_tp_subgroup`` once per shape
    returns, only dealt and exchanged together.

    Every precondition is a hard error rather than a silent fallback: an arm that quietly
    ran the per-shape path would be scored for a mechanism it did not use.
    """
    world = tp_group.size()
    rank = tp_group.rank()
    if len(stacks) < 2:
        raise ValueError(
            "fused subgroup exchange is a CROSS-SHAPE fusion and needs >= 2 stacks, got "
            f"{len(stacks)}"
        )
    if subgroup_size <= 0 or world % subgroup_size != 0:
        raise ValueError(f"subgroup_size {subgroup_size} must divide world {world}")
    groups = world // subgroup_size          # k
    if groups == 1:
        raise ValueError("subgroup_size == world is plain 'duplicated'")
    dtype, device = stacks[0].dtype, stacks[0].device
    for stack in stacks:
        if stack.dim() != 3:
            raise ValueError(f"each stack must be (count, rows, cols), got {tuple(stack.shape)}")
        if stack.dtype is not dtype or stack.device != device:
            raise ValueError("all stacks must share one dtype and device")
    stacks = [stack.contiguous() for stack in stacks]

    n_shapes = len(stacks)
    counts = [stack.size(0) for stack in stacks]
    dims = [(stack.size(1), stack.size(2)) for stack in stacks]
    elems = [rows * cols for rows, cols in dims]
    owners = fused_subgroup_deal(
        [(rows * world, cols) for rows, cols in dims], counts, groups, balanced
    )
    my_sub = rank // subgroup_size           # s_r
    my_slot = rank % subgroup_size           # i_r, this rank's send-duty slot
    # owned[i][s]: the matrix indices of shape i that subgroup s owns, ascending.
    owned = [
        [[j for j in range(counts[i]) if owners[i][j] == s] for s in range(groups)]
        for i in range(n_shapes)
    ]
    mine = [len(owned[i][my_sub]) for i in range(n_shapes)]

    def sub_elems(sub: int) -> int:
        """Elements one subgroup's whole (cross-shape) share occupies, per rank."""
        return sum(len(owned[i][sub]) * elems[i] for i in range(n_shapes))

    my_block = sub_elems(my_sub)

    if wire and wire not in WIRE_MODES:
        raise ValueError(f"unknown fused wire mode {wire!r}; known: {sorted(WIRE_MODES)}")
    if wire and not pipelined:
        # The four widening/narrowing sites the wire modes ride are all inside
        # ``_fused_exchange_pipelined``; the monolithic path stages through ``cat``/``stack``
        # and would have to pay a NEW pass for each. A silent fp32 fallback would let a
        # bf16-wire arm be scored while running the fp32 wire, so this is a hard error.
        raise ValueError("fused wire modes require the pipelined fused exchange")

    if pipelined:
        # Same deal (``owners`` above), same math, restructured exchange -- see
        # ``_fused_exchange_pipelined``. A silent fallback here would let the arm be scored
        # while running the code it claims to replace, so the precondition is a hard error.
        if pipe_chunks <= 0:
            raise ValueError(
                f"pipelined fused exchange needs pipe_chunks >= 1, got {pipe_chunks}"
            )
        if not HAVE_FUSED_NS:
            raise ValueError(
                "pipelined fused exchange requires kernels.fused_ns; it batches the "
                "per-matrix Newton-Schulz loop through the fused entry point"
            )
        wire_in, wire_out = WIRE_MODES.get(wire, (None, None)) if wire else (None, None)
        return _fused_exchange_pipelined(
            stacks, steps, coefficient_type, tp_group, use_syrk,
            subgroup_size, groups, my_sub, my_slot, owned, counts, dims, elems,
            dtype, device, pipe_chunks, wire_in, wire_out,
        )

    # ---- ONE input exchange over every shape ------------------------------------------
    # Peer-major outer, shape-major inner: destination d gets, for each shape in order,
    # this rank's shard of the matrices d's subgroup owns.
    pieces = []
    for d in range(world):
        sub_d = d // subgroup_size
        for i in range(n_shapes):
            idx = owned[i][sub_d]
            if idx:
                pieces.append(stacks[i][idx].reshape(-1))
    send = torch.cat(pieces) if pieces else torch.empty(0, dtype=dtype, device=device)
    del pieces
    recv = torch.empty(world * my_block, dtype=dtype, device=device)
    torch.distributed.all_to_all_single(
        recv,
        send,
        output_split_sizes=[my_block] * world,
        input_split_sizes=[sub_elems(d // subgroup_size) for d in range(world)],
        group=tp_group,
    )
    del send

    # ---- compute: the SAME per-matrix newton_schulz, on this subgroup's pooled share ---
    recv_2d = recv.view(world, my_block)
    results = [
        torch.empty((counts[i], dims[i][0], dims[i][1]), dtype=dtype, device=device)
        for i in range(n_shapes)
    ]
    full: List[List[torch.Tensor]] = []
    offset = 0
    for i in range(n_shapes):
        rows, cols = dims[i]
        span = mine[i] * elems[i]
        if span == 0:
            full.append([])
            continue
        # (src, matrix, rows, cols) -> per matrix the shards concatenated in RANK order,
        # which is what duplicated's all_gather + cat(dim=partition_dim) produces.
        global_x = (
            recv_2d[:, offset : offset + span]
            .reshape(world, mine[i], rows, cols)
            .permute(1, 0, 2, 3)
            .reshape(mine[i], world * rows, cols)
        )
        offset += span
        full.append([
            newton_schulz(global_x[m], steps, coefficient_type, use_syrk=use_syrk)
            for m in range(mine[i])
        ])
        del global_x
    del recv_2d, recv

    # ---- ONE output exchange over every shape -----------------------------------------
    # This rank serves destinations d with d % subgroup_size == my_slot: k of them, each
    # getting every owned matrix's row block d, for every shape, in the same order the
    # receiver parses.
    pieces = []
    for d in range(world):
        if d % subgroup_size != my_slot:
            continue
        for i in range(n_shapes):
            if not full[i]:
                continue
            rows = dims[i][0]
            pieces.append(
                torch.stack([y[d * rows : (d + 1) * rows] for y in full[i]]).reshape(-1)
            )
    send_out = torch.cat(pieces) if pieces else torch.empty(0, dtype=dtype, device=device)
    del pieces, full
    out = torch.empty(
        sum(counts[i] * elems[i] for i in range(n_shapes)), dtype=dtype, device=device
    )
    torch.distributed.all_to_all_single(
        out,
        send_out,
        # Received from src = s * subgroup_size + my_slot for s = 0..k-1, ascending, so the
        # blocks arrive in subgroup order and, inside each, in shape order.
        output_split_sizes=[
            sub_elems(src // subgroup_size) if src % subgroup_size == my_slot else 0
            for src in range(world)
        ],
        input_split_sizes=[
            my_block if d % subgroup_size == my_slot else 0 for d in range(world)
        ],
        group=tp_group,
    )
    del send_out
    cursor = 0
    for s in range(groups):
        for i in range(n_shapes):
            idx = owned[i][s]
            if not idx:
                continue
            rows, cols = dims[i]
            span = len(idx) * elems[i]
            results[i][idx] = out[cursor : cursor + span].view(len(idx), rows, cols)
            cursor += span
    return results


def time_fused_group(
    stacks, group, steps, coefficient_type, iters, warmup, use_syrk=False,
    subgroup_size=0, balanced=False, pipelined=False, pipe_chunks=0, wire="",
) -> float:
    """Median wall-clock ms for ONE fused region spanning several owned shape groups.

    The return value is the whole region's total. It replaces the SUM of the per-shape
    regions it fuses, which is why the composition change is explicit: a fused row is
    reported per RANK PROFILE, never re-attributed to one shape.
    """

    def once():
        newton_schulz_tp_subgroup_fused(
            stacks,
            steps=steps,
            coefficient_type=coefficient_type,
            tp_group=group,
            use_syrk=use_syrk,
            subgroup_size=subgroup_size,
            balanced=balanced,
            pipelined=pipelined,
            pipe_chunks=pipe_chunks,
            wire=wire,
        )

    for _ in range(warmup):
        once()
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)

    timings = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.distributed.barrier(group=group)
        start.record()
        once()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)



def time_group(
    stack, group, mode, steps, coefficient_type, iters, warmup, shard_count,
    use_syrk=False, batched=False, batch_chunk=0, subgroup_size=0, fused=False,
    pipelined=False, pipe_chunks=4, elide_self=False, wire="",
) -> float:
    """Median wall-clock ms to orthogonalize a WHOLE same-shape group in one timed region.

    ``stack`` is ``(count, rows, cols)``. The return value is the group's total, not a
    per-matrix cost: it enters the profile total directly, so the composition stays
    ``sum over owned groups`` exactly as the per-matrix path's ``median * count`` does.
    """
    partition_dim = None if shard_count == 1 else 0
    count = stack.size(0)
    chunk = batch_chunk if (batch_chunk and batch_chunk > 0) else count

    if subgroup_size:
        def once():
            newton_schulz_tp_subgroup(
                stack,
                steps=steps,
                coefficient_type=coefficient_type,
                tp_group=group,
                partition_dim=partition_dim,
                tp_mode=mode,
                use_syrk=use_syrk,
                subgroup_size=subgroup_size,
                batched=batched,
                batch_chunk=chunk,
                fused=fused,
                pipelined=pipelined,
                pipe_chunks=pipe_chunks,
                elide_self=elide_self,
                wire=wire,
            )
    elif batched:
        def once():
            for start_index in range(0, count, chunk):
                newton_schulz_tp_batched(
                    stack[start_index : start_index + chunk],
                    steps=steps,
                    coefficient_type=coefficient_type,
                    tp_group=group,
                    partition_dim=partition_dim,
                    tp_mode=mode,
                    use_syrk=use_syrk,
                    fused=fused,
                )
    else:
        def once():
            for index in range(count):
                newton_schulz_tp(
                    stack[index],
                    steps=steps,
                    coefficient_type=coefficient_type,
                    tp_group=group,
                    partition_dim=partition_dim,
                    tp_mode=mode,
                    use_syrk=use_syrk,
                )

    for _ in range(warmup):
        once()
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)

    timings = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.distributed.barrier(group=group)
        start.record()
        once()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


def main() -> None:
    """Time every strategy on every distinct shape, then report per-profile and max."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--group",
        choices=["gtp", "egtp"],
        required=True,
        help="Which sharding axis this job's process group stands in for.",
    )
    parser.add_argument("--modelled-world-size", type=int, default=12288)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gtp", type=int, default=64)
    parser.add_argument("--ep", type=int, default=64)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--egtp", type=int, default=4)
    parser.add_argument("--shard-latent-proj", action="store_true")
    parser.add_argument("--num-ns-steps", type=int, default=16)
    parser.add_argument("--coefficient-type", type=str, default="polar_express")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    # blockwise is not used in practice: it orthogonalizes each block rather than the
    # matrix, which changes the update rather than distributing the same one. Pass it
    # explicitly (--modes blockwise duplicated distributed) if you want it as a floor.
    parser.add_argument("--modes", nargs="+", default=["duplicated", "distributed"])
    # Forwarded to newton_schulz_tp. SYRK replaces the two symmetric products with
    # half-FLOP triangular kernels; it only takes effect at --fp32-matmul-prec medium and
    # needs both dims to be multiples of 8. ns_step_flops/flop_model follow the flag, so
    # the FLOP columns stay self-consistent -- but GF is then on a different cost model
    # than a GEMM-path run and must not be compared across the two.
    parser.add_argument("--use-syrk", action="store_true")
    # Set-composed timing. Off by default so the legacy per-matrix composition -- and the
    # numbers taken under it -- keep reproducing byte-for-byte.
    parser.add_argument(
        "--set-timing",
        action="store_true",
        help="Additionally time each owned same-shape GROUP inside one timed region: the "
             "unbatched '<mode>_set' baseline arm and the batched 'duplicated_batch' arm. "
             "When set, 'fastest step' is reported over the set-composed policies.",
    )
    parser.add_argument(
        "--subgroup-sizes",
        nargs="*",
        type=int,
        default=[],
        help="Under --set-timing, also time subgroup duplication at these duplication-"
             "group sizes g (each must divide the world size and be < it). The owned "
             "same-shape set is dealt round-robin over world/g subgroups; see "
             "newton_schulz_tp_subgroup. Empty (the default) leaves it off entirely.",
    )
    parser.add_argument(
        "--subgroup-batched",
        action="store_true",
        help="Under --set-timing with --subgroup-sizes, ALSO time a batched variant of "
             "each subgroup column ('duplicated_batch_sub_g<g>'): the same subgroup deal "
             "with the owned matrices orthogonalized as one batched Newton-Schulz per "
             "--batch-chunk instead of one 2-D call each. Adds two scored policies, "
             f"'{PER_SHAPE_BATCH_POLICY}' (the accepted-state reference: per shape the "
             f"better of the set winner and 'duplicated_batch') and "
             f"'{PER_SHAPE_BATCH_SUBGROUP_POLICY}' (the same argmin with the batched-"
             "subgroup arms added). Off by default, so without it every column, policy "
             "and scored number is byte-identical to before.",
    )
    parser.add_argument(
        "--subgroup-all-shapes",
        action="store_true",
        help="Additionally score 'per_shape_sub_all_set': the same subgroup substitution "
             "as 'per_shape_sub_set' but on EVERY shape that has a subgroup column, "
             "including the ones whose set-composed winner is 'distributed_set'. Off by "
             "default, so without it the scored policy set is byte-identical to before.",
    )
    parser.add_argument(
        "--fused-ns-kernel",
        action="store_true",
        help="Under --subgroup-batched, ALSO time each batched-subgroup column with the "
             "fused Newton-Schulz entry point from dist_muon_opt/kernels/fused_ns.py "
             f"('{BATCH_SUBGROUP_PREFIX}<g>{FUSED_SUFFIX}') and score "
             f"'{PER_SHAPE_BATCH_SUBGROUP_FUSED_POLICY}' against the accepted state "
             f"'{PER_SHAPE_BATCH_SUBGROUP_POLICY}'. Identical math -- same 5 steps, same "
             "polar_express coefficients, same batched SYRK path -- with the fp32 "
             "normalize/cast prologue and the bf16->fp32 cast-and-store epilogue fused. "
             "Off by default, so without it the scored policy set is byte-identical to "
             "before.",
    )
    parser.add_argument(
        "--pipelined-exchange",
        action="store_true",
        help="Under --subgroup-batched at g == 1, ALSO time each batched-subgroup column "
             f"with the restructured exchange ('...{PIPE_SUFFIX}') and score "
             f"'{PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY}' against the accepted state. Same "
             "deal, same batched Newton-Schulz, same 5 steps / polar_express coefficients "
             "/ SYRK path -- what changes is that block ownership turns the exchange "
             "buffers into views of the caller's tensors (two of the four full-size "
             "staging copies stop existing) and the per-chunk exchanges are issued "
             "async_op=True so the wire overlaps a different chunk's compute. Off by "
             "default, so without it the scored policy set is byte-identical to before.",
    )
    parser.add_argument(
        "--elide-self-block",
        action="store_true",
        help="Under --pipelined-exchange, ALSO time each pipelined column with the "
             f"rank's OWN block dropped from both exchanges ('...{ELIDE_SUFFIX}') and "
             f"score '{PER_SHAPE_BATCH_SUBGROUP_ELIDE_POLICY}' against the accepted "
             "pipelined state. With shard_count = 2 on the egtp axis, HALF of every "
             "exchange payload is the rank's own block; this drops it from the collective "
             "and writes it locally into the buffer that consumes it (into the gather "
             "buffer on the way in, into the result on the way out), which adds no HBM "
             "traffic over the transposes the pipelined path already pays. Same deal, "
             "same batched Newton-Schulz, same 5 steps / coefficients / SYRK path, same "
             "pipeline depth -- the result is BIT-IDENTICAL. Off by default, so without "
             "it the scored policy set is byte-identical to before.",
    )
    parser.add_argument(
        "--subgroup-wire-modes",
        nargs="*",
        default=[],
        metavar="MODE",
        help="Under --pipelined-exchange, ALSO time each pipelined (and, with "
             f"--elide-self-block, each elided) column with the OUTPUT leg of the "
             f"exchange encoded at the given wire precision ('...{'_w'}<mode>', modes: "
             f"{', '.join(sorted(SUBGROUP_WIRE_MODES))}) and score "
             f"'{PER_SHAPE_BATCH_SUBGROUP_WIRE_POLICY}' against the accepted state. The "
             "reply that carries each peer's row block of the orthogonalized result "
             "travels as bf16 and is widened into the fp32 result INSIDE the timed "
             "region. Same deal, same batched Newton-Schulz, same 5 steps / coefficients "
             "/ SYRK path, same pipeline depth, same self-block treatment; the compute "
             "dtype and the returned dtype are untouched. BITWISE: the fused kernel holds "
             "X in bf16 across all five steps and widens only at its epilogue, so every "
             "value on that wire is exactly bf16-representable. Empty by default, so "
             "without it the scored policy set is byte-identical to before; an unknown "
             "mode, a wire mode without --pipelined-exchange, or one on the unfused "
             "compute path is a hard ValueError, never a silent fp32 fallback.",
    )
    parser.add_argument(
        "--pipe-chunks",
        type=int,
        default=4,
        help="Pipeline depth for --pipelined-exchange: the owned set is split into this "
             "many windows, each with its own pair of exchanges. Raises the collective "
             "count per region from 2 to 2*N (at alpha_ib = 75.681 us, +0.45 ms/region at "
             "N = 4), so keep it small; N = 1 degenerates to no overlap.",
    )
    parser.add_argument(
        "--fuse-shape-exchanges",
        action="store_true",
        help="Under --set-timing with --subgroup-sizes, ALSO time the owned shapes whose "
             "banked arm is a subgroup column as ONE fused region with ONE pair of "
             f"all_to_all_single ('{FUSED_EXCHANGE_PREFIX}<g>' = the same round-robin deal, "
             f"exchange fused only; '{FUSED_EXCHANGE_PREFIX}<g>{BALANCED_SUFFIX}' = the "
             "deal also pooled across shapes by greedy LPT on ns_cost), and score "
             f"'{PER_SHAPE_FUSED_EXCHANGE_POLICY}' against the accepted state. Identical "
             "math -- every matrix is still all-gathered in rank order and orthogonalized "
             "whole by the same newton_schulz, same 5 steps, same coefficients, same SYRK "
             "path; only which subgroup owns which matrix and how the two collectives are "
             "packed change. The fused region is timed and reported PER RANK PROFILE, "
             "since which shapes it spans is a property of the profile. Off by default, "
             "so without it the scored policy set is byte-identical to before.",
    )
    parser.add_argument(
        "--pipeline-fused-exchange",
        action="store_true",
        help="Under --fuse-shape-exchanges, ALSO time each fused-exchange arm with the "
             f"restructured exchange ('{FUSED_EXCHANGE_PREFIX}<g>[{BALANCED_SUFFIX}]"
             f"{PIPE_SUFFIX}<N>') and let the fused-arm argmin see it. Same deal, same "
             "matrices, same 5 steps / polar_express coefficients / SYRK path -- what "
             "changes is that the pooled share is split into N windows whose exchanges are "
             "issued async so the wire overlaps a different window's compute, that the "
             "per-destination send staging is replaced by one per-SUBGROUP block handed to "
             "the list form of all_to_all (the buffer shrinks by g and the cat/stack python "
             "loops collapse to one strided copy), and that the per-matrix newton_schulz "
             "loop becomes one batched call into the fused kernel. Off by default, so "
             "without it the scored policy set is byte-identical to before.",
    )
    parser.add_argument(
        "--fused-pipe-chunks",
        type=int,
        nargs="+",
        default=[2, 4],
        help="Pipeline depths N to time under --pipeline-fused-exchange, each its own arm. "
             "The GTP fused region's pooled share is only 3-4 matrices per subgroup, so N "
             "is bounded by that; N = 1 degenerates to no overlap. Each N raises the "
             "collective count per region to 2*N and the results scatter to N times "
             "groups*shapes writes, so the sweep is what prices the tradeoff.",
    )
    parser.add_argument(
        "--fused-wire-modes",
        type=str,
        nargs="*",
        default=[],
        choices=sorted(WIRE_MODES),
        help="Under --pipeline-fused-exchange, ALSO time each pipelined fused column with "
             f"the named TRANSPORT encoding ('...{WIRE_SUFFIX}<mode>'), in the SAME job "
             "over the SAME tensors under the SAME composition, fed to the SAME fused-arm "
             "argmin. 'bf16' puts both exchange legs on the wire in bfloat16, 'bf16in' the "
             "input leg only (which carries 4/5 of the region's wire bytes at "
             "subgroup_size = 4) and 'bf16out' the output leg only. COMPUTE dtype is "
             "unchanged in every mode: the gather buffer stays fp32 so the fused kernel's "
             "preconditions hold, and the region still returns fp32. Three of the four "
             "width changes ride copies the pipelined path already pays (the send gather, "
             "the transposing receive copy, the strided send-out copy); the fourth, the "
             "scatter into results, costs one extra pass, which is why the legs are "
             "separable arms. The OUTPUT leg is bitwise -- the fused kernel's epilogue "
             "upcasts a bf16 X, so its values are exactly bf16-representable -- while the "
             "INPUT leg perturbs every entry before the Newton-Schulz prologue and is "
             "gated on the equivalence check, not assumed. Off by default.",
    )
    parser.add_argument(
        "--batch-replicated-shapes",
        action="store_true",
        help="Additionally score '" + PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY + "': "
             "'" + PER_SHAPE_SUBGROUP_ALL_POLICY + "' with the already-timed '"
             + BATCH_POLICY + "' column added to the per-shape argmin, on shard_count == 1 "
             "(replicated) shapes ONLY. Costs no extra timing -- both columns it selects "
             "between are already measured for the printed table. Off by default, so "
             "without it the scored policy set is byte-identical to before.",
    )
    parser.add_argument(
        "--batch-replicated-fused",
        action="store_true",
        help="Additionally time the '" + BATCH_FUSED_POLICY + "' column on shard_count == 1 "
             "(replicated) shapes -- the same batched Newton-Schulz with the same chunking, "
             "routed through kernels.fused_ns.fused_newton_schulz_batched so the fp32 "
             "normalize/cast prologue and the cast/store epilogue stop being separate HBM "
             "passes -- and score '" + PER_SHAPE_SUBGROUP_ALL_BATCH_FUSED_POLICY + "', i.e. '"
             + PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY + "' with that column substituted ONLY on "
             "the shapes whose reference arm is already '" + BATCH_POLICY + "'. Requires "
             "--batch-replicated-shapes. The fused entry point's silent fallback is an "
             "ERROR on this path, not a no-op, so the column can never measure the "
             "reference arm under the candidate's name. Off by default.",
    )
    parser.add_argument(
        "--batch-chunk",
        type=int,
        default=64,
        help="Matrices per batched Newton-Schulz call under --set-timing (0 = the whole "
             "group at once). Bounds the resident all-gather buffer.",
    )
    # Newton-Schulz requires fp32: it runs on Muon's momentum, which the optimizer keeps
    # in fp32 regardless of the parameter dtype. bf16 raises ValueError.
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    # newton_schulz reads the GLOBAL torch float32 matmul precision, and at "medium" it casts
    # to bf16 for the GEMMs and back. Megatron defaults muon_fp32_matmul_prec to "medium", so
    # leaving this unset would measure a different precision than a real run.
    parser.add_argument(
        "--fp32-matmul-prec",
        type=str,
        default="medium",
        choices=["medium", "high", "highest"],
        help="medium=bf16 compute (Megatron default), high=tf32, highest=true fp32.",
    )
    config = parser.parse_args()

    assert HAVE_EMERGING_OPTIMIZERS, "emerging_optimizers is required; pip install it first."

    torch.set_float32_matmul_precision(config.fp32_matmul_prec)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl")
    group = torch.distributed.distributed_c10d._get_default_group()
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    def log(message: str) -> None:
        if rank == 0:
            print(message, flush=True)

    dense, expert = build_model_matrices(config)
    if config.group == "gtp":
        matrices, group_size = dense, config.gtp
        dp_size = config.modelled_world_size // (config.tp * config.gtp)
    else:
        matrices, group_size = expert, config.egtp
        dp_size = config.modelled_world_size // (config.etp * config.egtp * config.ep)

    assert world == group_size, (
        f"--group {config.group} models a {group_size}-way sharded axis, so launch with "
        f"{group_size} ranks; got {world}."
    )

    # Validated here, before any timing runs: a bad g is a typo in the sbatch, and finding
    # out about it after the per-shape sweep wastes the job.
    subgroup_sizes = sorted({g for g in config.subgroup_sizes if g > 0}) if config.set_timing else []
    for g in subgroup_sizes:
        assert group_size % g == 0 and g < group_size, (
            f"--subgroup-sizes {g} must divide the {group_size}-rank group and be smaller "
            f"than it (g == group size is plain 'duplicated')."
        )
    if config.batch_replicated_fused:
        assert config.batch_replicated_shapes, (
            "--batch-replicated-fused requires --batch-replicated-shapes: the column it "
            "substitutes is the replicated batched arm, which only enters a scored policy "
            "there."
        )
        assert HAVE_FUSED_NS, (
            "--batch-replicated-fused needs kernels.fused_ns (FUSED_NS_DIR); the arm routes "
            "the replicated batched Newton-Schulz through the fused entry point and must "
            "never silently score the unfused code it claims to replace."
        )
    if config.pipeline_fused_exchange:
        assert config.fuse_shape_exchanges, (
            "--pipeline-fused-exchange requires --fuse-shape-exchanges: it restructures "
            "the fused cross-shape exchange, so there is nothing to restructure without it."
        )
        assert all(n >= 1 for n in config.fused_pipe_chunks), (
            f"--fused-pipe-chunks must all be >= 1, got {config.fused_pipe_chunks}"
        )
        assert HAVE_FUSED_NS, (
            "--pipeline-fused-exchange needs kernels.fused_ns (FUSED_NS_DIR); the arm "
            "batches the per-matrix Newton-Schulz loop through the fused entry point and "
            "must never silently score the unfused code it claims to replace."
        )
    if config.fused_wire_modes:
        assert config.pipeline_fused_exchange, (
            "--fused-wire-modes requires --pipeline-fused-exchange: the widenings the "
            "modes ride only exist in the pipelined fused exchange."
        )
        assert config.dtype == "float32", (
            "--fused-wire-modes changes the TRANSPORT encoding of an fp32 region; it is "
            f"undefined at --dtype {config.dtype}."
        )
    if config.subgroup_wire_modes:
        assert config.pipelined_exchange, (
            "--subgroup-wire-modes requires --pipelined-exchange: the exchange whose "
            "output leg is re-encoded only exists in the pipelined subgroup path."
        )
        assert config.fused_ns_kernel, (
            "--subgroup-wire-modes requires --fused-ns-kernel: the bitwise property comes "
            "from fused_newton_schulz_batched holding X in bf16 across all five steps."
        )
        unknown = [w for w in config.subgroup_wire_modes if w not in SUBGROUP_WIRE_MODES]
        assert not unknown, (
            f"unknown --subgroup-wire-modes {unknown}; known: "
            f"{sorted(SUBGROUP_WIRE_MODES)}"
        )
        assert config.dtype == "float32", (
            "--subgroup-wire-modes changes the TRANSPORT encoding of an fp32 region; it "
            f"is undefined at --dtype {config.dtype}."
        )
    if config.subgroup_batched:
        assert config.set_timing and subgroup_sizes, (
            "--subgroup-batched needs --set-timing and a non-empty --subgroup-sizes."
        )
    if subgroup_sizes:
        assert "duplicated" in config.modes, (
            "subgroup duplication is a 'duplicated' variant; keep duplicated in --modes."
        )

    # Collapse the dp_size ranks into distinct ownership profiles.
    profiles: Dict[Tuple, List[int]] = {}
    for dp_rank in range(dp_size):
        signature = tuple(sorted(Counter(owned_matrices(matrices, dp_size, dp_rank)).items()))
        profiles.setdefault(signature, []).append(dp_rank)

    distinct = sorted({matrix for sig in profiles for matrix, _ in sig}, key=lambda e: -ns_cost(e))
    log(f"group={config.group} group_size={group_size} dp_size={dp_size}")
    log(f"{len(profiles)} distinct rank profiles over {len(distinct)} distinct shapes")
    log(
        f"ns_steps={config.num_ns_steps} coefficient={config.coefficient_type} "
        f"dtype={config.dtype} iters={config.iters}"
    )
    log(
        f"fp32_matmul_precision={torch.get_float32_matmul_precision()} "
        f"(medium casts the GEMMs to bf16)"
    )
    if config.use_syrk:
        log("use_syrk=True: FLOP columns use the SYRK cost model (3m^2n + m^3 per step),")
        log("so GF is NOT comparable to a GEMM-path run; ms and TF/s are.")
    if config.set_timing:
        log(
            f"set_timing=True: owned same-shape groups are also timed as a SET "
            f"(batch_chunk={config.batch_chunk}). 'fastest step' is over the set-composed "
            f"policies only."
        )
    if subgroup_sizes:
        log(
            f"subgroup_sizes={subgroup_sizes}: subgroup duplication is timed at "
            f"these g and enters the scored policy '{PER_SHAPE_SUBGROUP_POLICY}', which "
            f"substitutes it ONLY on shapes whose unscoped set-composed winner is already "
            f"'duplicated{SET_SUFFIX}'."
        )
    if subgroup_sizes and config.subgroup_batched:
        log(
            f"subgroup_batched=True: also timing '{BATCH_SUBGROUP_PREFIX}<g>' (owner-"
            f"computes inside the batched path) and scoring "
            f"'{PER_SHAPE_BATCH_SUBGROUP_POLICY}' against its accepted-state reference "
            f"'{PER_SHAPE_BATCH_POLICY}'. At g=1 on this axis each matrix is "
            f"orthogonalized by exactly ONE rank, so redundancy falls to 1.0."
        )
    if subgroup_sizes and config.subgroup_all_shapes:
        log(
            f"subgroup_all_shapes=True: also scoring '{PER_SHAPE_SUBGROUP_ALL_POLICY}', "
            f"the same substitution on EVERY shape that has a subgroup column, including "
            f"the ones whose winner is 'distributed{SET_SUFFIX}'. Costs no extra timing: "
            f"those columns are already measured for the table."
        )
    if config.batch_replicated_fused:
        log(
            f"batch_replicated_fused=True: also timing '{BATCH_FUSED_POLICY}' on "
            f"shard_count == 1 shapes and scoring "
            f"'{PER_SHAPE_SUBGROUP_ALL_BATCH_FUSED_POLICY}', i.e. "
            f"'{PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY}' with that column substituted only "
            f"where the reference already banked '{BATCH_POLICY}'. Same batched call, same "
            f"chunking, same 5 steps, same coefficients, same batched SYRK path -- only "
            f"the per-chunk entry point differs."
        )
    if config.batch_replicated_shapes:
        log(
            f"batch_replicated_shapes=True: also scoring "
            f"'{PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY}', i.e. "
            f"'{PER_SHAPE_SUBGROUP_ALL_POLICY}' with the '{BATCH_POLICY}' column added to "
            f"the per-shape argmin on shard_count == 1 shapes only. Costs no extra timing: "
            f"that column is already measured for the table."
        )
    log("")

    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
    errors: Dict[str, List[str]] = {}
    per_shape: Dict[Tuple, Dict[str, float]] = {}

    log("PER-SHAPE  (issued = FLOPs this GPU executes; useful = its share of orthogonalizing")
    log("            the full matrix once; redundancy = issued / useful)")
    header = (
        f"{'local shard':>13}{'all-gathered':>14}{'mode':>13}{'issued GF':>11}"
        f"{'useful GF':>11}{'redund':>8}{'ms':>9}{'issued TF/s':>13}{'useful TF/s':>13}"
    )
    log(header)
    log("-" * len(header))
    for matrix in distinct:
        (rows, cols), shard_count = matrix
        # The all-gathered shape is what duplicated and distributed orthogonalize. It
        # equals the local shard for a replicated weight, where shard_count is 1.
        gathered = f"{rows * shard_count}x{cols}"
        shard = torch.randn((rows, cols), device="cuda", dtype=dtype)
        timings: Dict[str, float] = {}
        for mode in config.modes:
            issued, useful = flop_model(
                matrix, mode, config.num_ns_steps, group_size, config.use_syrk
            )
            try:
                ms = time_strategy(
                    shard,
                    group,
                    mode,
                    config.num_ns_steps,
                    config.coefficient_type,
                    config.iters,
                    config.warmup,
                    shard_count,
                    config.use_syrk,
                )
                timings[mode] = ms
                log(
                    f"{f'{rows}x{cols}':>13}{gathered:>14}{mode:>13}{issued / 1e9:>11.1f}"
                    f"{useful / 1e9:>11.1f}{issued / useful:>7.2f}x{ms:>9.3f}"
                    f"{issued / 1e9 / ms:>13.0f}{useful / 1e9 / ms:>13.0f}"
                )
            except Exception as error:  # noqa: BLE001 - report and keep going
                log(f"{f'{rows}x{cols}':>13}{gathered:>14}{mode:>13}{type(error).__name__:>56}")
                errors.setdefault(f"{type(error).__name__}: {error}", []).append(mode)
        per_shape[matrix] = timings

    if errors:
        log("\nerrors:")
        for message, modes in errors.items():
            log(f"  {','.join(sorted(set(modes)))}: {message}")
        torch.distributed.destroy_process_group()
        return

    # ----------------------------------------------------------------------------------
    # Per-shape mode selection.
    #
    # The optimizer picks ``--muon-tp-mode`` per *buffer*, not per axis, so nothing forces
    # every weight on an axis onto the same mode. Which mode wins is a property of the
    # SHAPE (duplicated's redundant compute grows with the matrix; distributed's replicated
    # ``A @ A`` term does not shrink with the group), and the winner genuinely flips within
    # a single axis. ``per_shape`` is therefore reported as a policy alongside the two
    # single-mode columns: each distinct shape takes whichever of ``--modes`` measured
    # faster, and the step cost is recomputed as the max over rank profiles UNDER THAT
    # SELECTION -- not by patching the previous single-mode winner, since the slowest
    # profile can change once the per-shape costs do.
    #
    # Taking the per-shape argmin minimizes every profile's total simultaneously, so it
    # also minimizes their max: the policy is optimal over per-shape mode assignments.
    #
    # Every rank of the group holds the same set of matrices, so the selection is identical
    # on every rank and the collectives inside newton_schulz_tp stay in lockstep. The
    # selection changes only WHICH already-benchmarked mode runs for a shape; no new
    # numerics are introduced.
    # ----------------------------------------------------------------------------------
    per_shape_policy = "per_shape"
    best_mode_for_shape: Dict[Tuple, str] = {}
    if len(config.modes) > 1:
        for matrix in distinct:
            best_mode_for_shape[matrix] = min(config.modes, key=lambda m: per_shape[matrix][m])

    policies = list(config.modes)
    if best_mode_for_shape:
        policies.append(per_shape_policy)

    # ----------------------------------------------------------------------------------
    # Set-composed arms (--set-timing). See the block above ``time_group``.
    # ----------------------------------------------------------------------------------
    set_modes = [f"{mode}{SET_SUFFIX}" for mode in config.modes]
    per_shape_set_policy = per_shape_policy + SET_SUFFIX
    # group_ms[(matrix, count)][policy] is the WHOLE group's time, not a per-matrix cost.
    group_ms: Dict[Tuple, Dict[str, float]] = {}
    set_policies: List[str] = []
    best_set_mode_for_shape: Dict[Tuple, str] = {}
    best_subgroup_for_shape: Dict[Tuple, str] = {}
    best_subgroup_all_for_shape: Dict[Tuple, str] = {}
    best_subgroup_all_batch_for_shape: Dict[Tuple, str] = {}
    best_batch_base_for_shape: Dict[Tuple, str] = {}
    best_batch_subgroup_for_shape: Dict[Tuple, str] = {}
    best_batch_subgroup_fused_for_shape: Dict[Tuple, str] = {}
    best_batch_subgroup_pipe_for_shape: Dict[Tuple, str] = {}
    pipe_reference_policy = PER_SHAPE_BATCH_SUBGROUP_POLICY
    best_batch_subgroup_elide_for_shape: Dict[Tuple, str] = {}
    elide_reference_policy = PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY
    best_batch_subgroup_wire_for_shape: Dict[Tuple, str] = {}
    wire_reference_policy = PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY
    # Cross-shape exchange fusion (--fuse-shape-exchanges). Unlike every other arm this is
    # a PER-PROFILE region, not a per-shape column, so it lives in its own maps.
    fused_region_ms: Dict[Tuple, Dict[str, float]] = {}
    fuse_shapes: Dict[Tuple, List[Tuple]] = {}
    fuse_reference_policy = None
    fuse_arm = None
    fused_audit: Dict[Tuple, Tuple[str, float, str, float]] = {}
    pipe_audit: Dict[Tuple, Tuple[str, float, str, float]] = {}
    elide_audit: Dict[Tuple, Tuple[str, float, str, float]] = {}
    wire_audit: Dict[Tuple, Tuple[str, float, str, float]] = {}
    subgroup_audit: Dict[Tuple, Tuple[str, float, str, float]] = {}
    batch_subgroup_audit: Dict[Tuple, Tuple[str, float, str, float]] = {}
    batch_replicated_audit: Dict[Tuple, Tuple[str, float, str, float]] = {}
    batch_replicated_fused_audit: Dict[Tuple, Tuple[str, float, str, float]] = {}
    best_subgroup_all_batch_fused_for_shape: Dict[Tuple, str] = {}
    if config.set_timing:
        # Every (shape, owned count) pair that appears in any rank profile, timed once.
        needed: Dict[Tuple, set] = {}
        for signature in profiles:
            for matrix, count in signature:
                needed.setdefault(matrix, set()).add(count)

        log("\nSET-COMPOSED  (whole owned group of one shape inside ONE timed region;")
        log("               ms is the GROUP total, per-matrix = ms / count)")
        set_header = f"{'local shard':>13}{'count':>7}{'policy':>19}{'group ms':>11}{'per-matrix ms':>15}"
        log(set_header)
        log("-" * len(set_header))
        for matrix in distinct:
            (rows, cols), shard_count = matrix
            for count in sorted(needed.get(matrix, ())):
                # One stack shared by both arms: the unbatched arm iterates its slices,
                # the batched arm consumes it whole. Same data, same resident footprint.
                stack = torch.randn((count, rows, cols), device="cuda", dtype=dtype)
                entry: Dict[str, float] = {}
                for mode in config.modes:
                    entry[f"{mode}{SET_SUFFIX}"] = time_group(
                        stack, group, mode, config.num_ns_steps, config.coefficient_type,
                        config.iters, config.warmup, shard_count, config.use_syrk,
                        batched=False,
                    )
                # Only ``duplicated`` is batched; see newton_schulz_tp_batched's docstring.
                if "duplicated" in config.modes:
                    entry[BATCH_POLICY] = time_group(
                        stack, group, "duplicated", config.num_ns_steps,
                        config.coefficient_type, config.iters, config.warmup, shard_count,
                        config.use_syrk, batched=True, batch_chunk=config.batch_chunk,
                    )
                    if config.batch_replicated_fused and shard_count == 1:
                        # Same arm, same tensors, same chunking, same composition -- only
                        # the per-chunk Newton-Schulz entry point differs. shard_count == 1
                        # only: at partition_dim is None the batched call short-circuits to
                        # a local 3-D Newton-Schulz, which is the call the fusion replaces.
                        # On a sharded shape the compute runs on a permuted reshape of the
                        # all-gather buffer, which is not contiguous, so the fused entry
                        # point's preconditions would not hold there.
                        entry[BATCH_FUSED_POLICY] = time_group(
                            stack, group, "duplicated", config.num_ns_steps,
                            config.coefficient_type, config.iters, config.warmup,
                            shard_count, config.use_syrk, batched=True,
                            batch_chunk=config.batch_chunk, fused=True,
                        )
                # Subgroup duplication, one column per g. Timed on every shape so the
                # column is readable, but SCORED only where noted below.
                for g in subgroup_sizes:
                    if shard_count == 1:
                        # Replicated weight: no shards to redistribute, so subgrouping is
                        # a no-op that would only re-time ``duplicated``. Skipped, not
                        # timed, so no column implies a win that is not there.
                        continue
                    entry[subgroup_policy(g)] = time_group(
                        stack, group, "duplicated", config.num_ns_steps,
                        config.coefficient_type, config.iters, config.warmup, shard_count,
                        config.use_syrk, subgroup_size=g,
                    )
                    if config.subgroup_batched:
                        entry[batch_subgroup_policy(g)] = time_group(
                            stack, group, "duplicated", config.num_ns_steps,
                            config.coefficient_type, config.iters, config.warmup,
                            shard_count, config.use_syrk, subgroup_size=g, batched=True,
                            batch_chunk=config.batch_chunk,
                        )
                        if config.fused_ns_kernel:
                            # Same arm, same tensors, same composition -- only the
                            # per-chunk Newton-Schulz entry point differs.
                            entry[batch_subgroup_policy(g, fused=True)] = time_group(
                                stack, group, "duplicated", config.num_ns_steps,
                                config.coefficient_type, config.iters, config.warmup,
                                shard_count, config.use_syrk, subgroup_size=g,
                                batched=True, batch_chunk=config.batch_chunk, fused=True,
                            )
                        if config.pipelined_exchange and g == 1:
                            # Same arm, same tensors, same composition -- only the staging
                            # and the scheduling of the two exchanges differ. Timed on the
                            # SAME fused/unfused compute path the reference arm banked, so
                            # the delta is the exchange and nothing else. g == 1 only: the
                            # block deal needs each rank to be its own duplication group.
                            entry[batch_subgroup_policy(
                                g, fused=config.fused_ns_kernel, pipelined=True
                            )] = time_group(
                                stack, group, "duplicated", config.num_ns_steps,
                                config.coefficient_type, config.iters, config.warmup,
                                shard_count, config.use_syrk, subgroup_size=g,
                                batched=True, batch_chunk=config.batch_chunk,
                                fused=config.fused_ns_kernel, pipelined=True,
                                pipe_chunks=config.pipe_chunks,
                            )
                            if config.elide_self_block:
                                # Same arm, same tensors, same composition, same pipeline
                                # depth -- only the self entry of each exchange differs.
                                entry[batch_subgroup_policy(
                                    g, fused=config.fused_ns_kernel, pipelined=True,
                                    elide_self=True,
                                )] = time_group(
                                    stack, group, "duplicated", config.num_ns_steps,
                                    config.coefficient_type, config.iters, config.warmup,
                                    shard_count, config.use_syrk, subgroup_size=g,
                                    batched=True, batch_chunk=config.batch_chunk,
                                    fused=config.fused_ns_kernel, pipelined=True,
                                    pipe_chunks=config.pipe_chunks, elide_self=True,
                                )
                            # Wire-encoding columns: one per (self-block treatment, mode).
                            # Same arm, same tensors, same composition, same pipeline
                            # depth, same self-block treatment -- only the TRANSPORT
                            # encoding of the output leg differs. Timed against BOTH the
                            # pipelined and the elided column so the wire delta is
                            # separable from the elision whichever one the reference
                            # banked.
                            for w in config.subgroup_wire_modes:
                                for es in (
                                    (False, True) if config.elide_self_block else (False,)
                                ):
                                    entry[batch_subgroup_policy(
                                        g, fused=config.fused_ns_kernel, pipelined=True,
                                        elide_self=es, wire=w,
                                    )] = time_group(
                                        stack, group, "duplicated", config.num_ns_steps,
                                        config.coefficient_type, config.iters,
                                        config.warmup, shard_count, config.use_syrk,
                                        subgroup_size=g, batched=True,
                                        batch_chunk=config.batch_chunk,
                                        fused=config.fused_ns_kernel, pipelined=True,
                                        pipe_chunks=config.pipe_chunks, elide_self=es,
                                        wire=w,
                                    )
                group_ms[(matrix, count)] = entry
                for policy, ms in entry.items():
                    log(
                        f"{f'{rows}x{cols}':>13}{count:>7}{policy:>19}{ms:>11.3f}"
                        f"{ms / count:>15.3f}"
                    )
                del stack
                torch.cuda.empty_cache()

        set_policies = list(set_modes)
        if BATCH_POLICY in next(iter(group_ms.values()), {}):
            set_policies.append(BATCH_POLICY)
        if len(set_modes) > 1:
            for matrix in distinct:
                counts = sorted(needed.get(matrix, ()))
                # Argmin on the per-matrix cost; identical across counts in practice, and
                # the largest owned count is the one that dominates the profile total.
                biggest = counts[-1]
                best_set_mode_for_shape[matrix] = min(
                    set_modes, key=lambda m: group_ms[(matrix, biggest)][m]
                )
            set_policies.append(per_shape_set_policy)
        # ------------------------------------------------------------------------------
        # Scored subgroup policy.
        #
        # The per-g columns are NOT scored on their own: applied blindly to every shape
        # they would also replace ``distributed`` on the shapes where ``distributed``
        # wins, which is a different question (a different duplication-group knob on a
        # different mode) and would be scored here without being asked. This policy is
        # exactly ``per_shape_set`` with subgroup duplication substituted on the shapes
        # whose unscoped set-composed winner is ALREADY ``duplicated_set`` -- so the only
        # thing it changes relative to its own reference arm is the size of a duplication
        # group that the reference already chose to use.
        # ------------------------------------------------------------------------------
        if subgroup_sizes and best_set_mode_for_shape:
            duplicated_set = f"duplicated{SET_SUFFIX}"
            for matrix in distinct:
                base = best_set_mode_for_shape[matrix]
                biggest = sorted(needed.get(matrix, ()))[-1]
                entry = group_ms[(matrix, biggest)]
                arms = [g for g in subgroup_sizes if subgroup_policy(g) in entry]
                if not arms:
                    # shard_count == 1: nothing to redistribute, no column was timed.
                    best_subgroup_for_shape[matrix] = base
                    best_subgroup_all_for_shape[matrix] = base
                    continue
                best_g = min(arms, key=lambda g: entry[subgroup_policy(g)])
                best_arm = subgroup_policy(best_g)
                subgroup_audit[matrix] = (base, entry[base], best_arm, entry[best_arm])
                # Widest scope (--subgroup-all-shapes): substitute wherever the subgroup
                # arm wins, INCLUDING the shapes whose winner is ``distributed_set``.
                best_subgroup_all_for_shape[matrix] = min(
                    (base, best_arm), key=lambda p: entry[p]
                )
                # Narrow scope (the phase-2 policy, unchanged): only where the reference
                # already chose a duplication group, i.e. base == duplicated_set.
                best_subgroup_for_shape[matrix] = (
                    best_subgroup_all_for_shape[matrix] if base == duplicated_set else base
                )
            set_policies.append(PER_SHAPE_SUBGROUP_POLICY)
            if config.subgroup_all_shapes:
                set_policies.append(PER_SHAPE_SUBGROUP_ALL_POLICY)
        # ------------------------------------------------------------------------------
        # Scored replicated-batch policy (--batch-replicated-shapes).
        #
        # The reference is the accepted state ``per_shape_sub_all_set``; the ONLY thing
        # this policy changes relative to it is that shard_count == 1 shapes may bank the
        # ``duplicated_batch`` column instead of their per-matrix loop. Those shapes have
        # no subgroup column at all (none is timed for them), so the reference banked their
        # set-composed winner unchanged and there is nothing here to re-bank.
        #
        # The shard_count == 1 gate is the safety property, and it is enforced twice: the
        # loop only ever looks at BATCH_POLICY under ``shard_count == 1``, and the result
        # is asserted afterwards. Reason: only at partition_dim is None does
        # ``newton_schulz_tp_batched`` short-circuit to plain ``newton_schulz``, keeping the
        # 3-D stack away from ``distributed_normalize_p2``'s whole-tensor
        # ``(x*x).sum()``. On a sharded shape the batched column is NOT a per-matrix-
        # equivalent arm and must never enter this argmin.
        #
        # Costs zero extra GPU time: BATCH_POLICY is timed for every shape already (see the
        # SET-COMPOSED block above); only the argmin that forms the policy widens.
        # ------------------------------------------------------------------------------
        if config.batch_replicated_shapes:
            if PER_SHAPE_SUBGROUP_ALL_POLICY not in set_policies:
                log(
                    f"\nbatch_replicated_shapes=True but '{PER_SHAPE_SUBGROUP_ALL_POLICY}' "
                    "was not scored (needs --subgroup-all-shapes and >1 mode); skipping "
                    f"'{PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY}' rather than scoring it "
                    "against an ambiguous reference."
                )
            else:
                for matrix in distinct:
                    (_, _), shard_count = matrix
                    base = best_subgroup_all_for_shape[matrix]
                    best_subgroup_all_batch_for_shape[matrix] = base
                    if shard_count != 1:
                        # Sharded: the batched column is a different computation on the
                        # collective route. Never a candidate here.
                        continue
                    biggest = sorted(needed.get(matrix, ()))[-1]
                    entry = group_ms[(matrix, biggest)]
                    if BATCH_POLICY not in entry:
                        continue
                    batch_replicated_audit[matrix] = (
                        base, entry[base], BATCH_POLICY, entry[BATCH_POLICY],
                    )
                    best_subgroup_all_batch_for_shape[matrix] = min(
                        (base, BATCH_POLICY), key=lambda p: entry[p]
                    )
                banked = [
                    m for m, p in best_subgroup_all_batch_for_shape.items()
                    if p == BATCH_POLICY
                ]
                assert all(m[1] == 1 for m in banked), (
                    f"{BATCH_POLICY} banked on a sharded shape: "
                    f"{[m for m in banked if m[1] != 1]}"
                )
                set_policies.append(PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY)
                # --------------------------------------------------------------------
                # Scored replicated-batch FUSION policy (--batch-replicated-fused).
                #
                # Reference is the ACCEPTED state PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY.
                # The ONLY thing this policy changes relative to it is that a shape whose
                # reference arm is exactly ``duplicated_batch`` may bank the FUSED variant
                # of that SAME column. It never re-opens the mode argmin, never re-opens
                # the subgroup argmin and never touches a sharded shape, so it cannot be
                # credited with any win the reference already made -- the delta is the
                # prologue/epilogue fusion and nothing else.
                # --------------------------------------------------------------------
                if config.batch_replicated_fused:
                    for matrix in distinct:
                        banked_arm = best_subgroup_all_batch_for_shape[matrix]
                        best_subgroup_all_batch_fused_for_shape[matrix] = banked_arm
                        if banked_arm != BATCH_POLICY:
                            # The reference did not bank the replicated batched column for
                            # this shape, so there is no fused counterpart to substitute.
                            continue
                        biggest = sorted(needed.get(matrix, ()))[-1]
                        entry = group_ms[(matrix, biggest)]
                        if BATCH_FUSED_POLICY not in entry:
                            continue
                        batch_replicated_fused_audit[matrix] = (
                            banked_arm, entry[banked_arm],
                            BATCH_FUSED_POLICY, entry[BATCH_FUSED_POLICY],
                        )
                        best_subgroup_all_batch_fused_for_shape[matrix] = min(
                            (banked_arm, BATCH_FUSED_POLICY), key=lambda p: entry[p]
                        )
                    fused_banked = [
                        m for m, p in best_subgroup_all_batch_fused_for_shape.items()
                        if p == BATCH_FUSED_POLICY
                    ]
                    assert all(m[1] == 1 for m in fused_banked), (
                        f"{BATCH_FUSED_POLICY} banked on a sharded shape: "
                        f"{[m for m in fused_banked if m[1] != 1]}"
                    )
                    set_policies.append(PER_SHAPE_SUBGROUP_ALL_BATCH_FUSED_POLICY)
        # ------------------------------------------------------------------------------
        # Scored batched-subgroup pair (--subgroup-batched).
        #
        # The reference is the ACCEPTED state, not B_set: per shape, the better of the
        # set-composed winner and the batched arm -- which on the EGTP axis is exactly the
        # 'duplicated_batch' that phase 1 banked. Charging this candidate against B_set
        # would re-bank phase 1's win. Both arms are formed here, from the same job, over
        # the same tensors, under the same composition, so the pair can never be assembled
        # from mismatched runs.
        # ------------------------------------------------------------------------------
        if config.subgroup_batched and best_set_mode_for_shape:
            for matrix in distinct:
                biggest = sorted(needed.get(matrix, ()))[-1]
                entry = group_ms[(matrix, biggest)]
                base_arms = [best_set_mode_for_shape[matrix]]
                if BATCH_POLICY in entry:
                    base_arms.append(BATCH_POLICY)
                base = min(base_arms, key=lambda p: entry[p])
                best_batch_base_for_shape[matrix] = base
                arms = [
                    g for g in subgroup_sizes if batch_subgroup_policy(g) in entry
                ]
                if not arms:
                    # shard_count == 1: nothing to redistribute, no column was timed.
                    best_batch_subgroup_for_shape[matrix] = base
                    continue
                best_g = min(arms, key=lambda g: entry[batch_subgroup_policy(g)])
                best_arm = batch_subgroup_policy(best_g)
                batch_subgroup_audit[matrix] = (base, entry[base], best_arm, entry[best_arm])
                best_batch_subgroup_for_shape[matrix] = min(
                    (base, best_arm), key=lambda p: entry[p]
                )
            set_policies.append(PER_SHAPE_BATCH_POLICY)
            set_policies.append(PER_SHAPE_BATCH_SUBGROUP_POLICY)
            # --------------------------------------------------------------------------
            # Scored NS-step-fusion policy (--fused-ns-kernel).
            #
            # Reference is the ACCEPTED state PER_SHAPE_BATCH_SUBGROUP_POLICY, and the only
            # thing this policy changes relative to it is that a shape whose banked arm is
            # a batched-subgroup column may bank the FUSED variant of that SAME g instead.
            # It never re-opens the g argmin and never re-banks a base arm, so it cannot be
            # credited with any win the reference already made -- the delta is the fusion
            # and nothing else.
            # --------------------------------------------------------------------------
            if config.fused_ns_kernel:
                for matrix in distinct:
                    banked = best_batch_subgroup_for_shape[matrix]
                    best_batch_subgroup_fused_for_shape[matrix] = banked
                    if not banked.startswith(BATCH_SUBGROUP_PREFIX):
                        # The reference did not bank a batched-subgroup arm for this shape,
                        # so there is no fused counterpart to substitute.
                        continue
                    biggest = sorted(needed.get(matrix, ()))[-1]
                    entry = group_ms[(matrix, biggest)]
                    fused_arm = banked + FUSED_SUFFIX
                    if fused_arm not in entry:
                        continue
                    fused_audit[matrix] = (
                        banked, entry[banked], fused_arm, entry[fused_arm],
                    )
                    best_batch_subgroup_fused_for_shape[matrix] = min(
                        (banked, fused_arm), key=lambda p: entry[p]
                    )
                set_policies.append(PER_SHAPE_BATCH_SUBGROUP_FUSED_POLICY)
            # --------------------------------------------------------------------------
            # Scored exchange-restructure policy (--pipelined-exchange).
            #
            # Reference is the ACCEPTED state: the fused policy when --fused-ns-kernel is
            # on, else the batched-subgroup policy. The only thing this policy changes
            # relative to that reference is that a shape whose banked arm is a
            # batched-subgroup column may bank the PIPELINED variant of that SAME arm. It
            # never re-opens the g argmin, never re-banks a base arm and never switches the
            # fused/unfused compute path, so it cannot be credited with any win the
            # reference already made -- the delta is the exchange and nothing else.
            # --------------------------------------------------------------------------
            if config.pipelined_exchange:
                pipe_reference_policy = (
                    PER_SHAPE_BATCH_SUBGROUP_FUSED_POLICY if config.fused_ns_kernel
                    else PER_SHAPE_BATCH_SUBGROUP_POLICY
                )
                reference_selection = (
                    best_batch_subgroup_fused_for_shape if config.fused_ns_kernel
                    else best_batch_subgroup_for_shape
                )
                for matrix in distinct:
                    banked = reference_selection[matrix]
                    best_batch_subgroup_pipe_for_shape[matrix] = banked
                    if not banked.startswith(BATCH_SUBGROUP_PREFIX):
                        # The reference did not bank a batched-subgroup arm for this shape,
                        # so there is no pipelined counterpart to substitute.
                        continue
                    biggest = sorted(needed.get(matrix, ()))[-1]
                    entry = group_ms[(matrix, biggest)]
                    pipe_arm = banked + PIPE_SUFFIX
                    if pipe_arm not in entry:
                        # g != 1: no pipelined column was timed for this arm.
                        continue
                    pipe_audit[matrix] = (
                        banked, entry[banked], pipe_arm, entry[pipe_arm],
                    )
                    best_batch_subgroup_pipe_for_shape[matrix] = min(
                        (banked, pipe_arm), key=lambda p: entry[p]
                    )
                set_policies.append(PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY)
            # --------------------------------------------------------------------------
            # Scored self-block-elision policy (--elide-self-block).
            #
            # Reference is the ACCEPTED state after stage-3 phase 3:
            # PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY. The only thing this policy changes
            # relative to that reference is that a shape whose banked arm is a PIPELINED
            # column may bank the ELIDED variant of that SAME arm. It never re-opens the g
            # argmin, never un-pipelines, and never switches the fused/unfused compute
            # path, so it cannot be credited with any win the reference already made --
            # the delta is the self block and nothing else.
            # --------------------------------------------------------------------------
            if config.elide_self_block:
                if not config.pipelined_exchange:
                    raise ValueError(
                        "--elide-self-block requires --pipelined-exchange: elision is "
                        "defined on the pipelined exchange only"
                    )
                for matrix in distinct:
                    banked = best_batch_subgroup_pipe_for_shape[matrix]
                    best_batch_subgroup_elide_for_shape[matrix] = banked
                    if not banked.endswith(PIPE_SUFFIX):
                        # The reference did not bank a pipelined arm for this shape, so
                        # there is no elided counterpart to substitute.
                        continue
                    biggest = sorted(needed.get(matrix, ()))[-1]
                    entry = group_ms[(matrix, biggest)]
                    elide_arm = banked + ELIDE_SUFFIX
                    if elide_arm not in entry:
                        continue
                    elide_audit[matrix] = (
                        banked, entry[banked], elide_arm, entry[elide_arm],
                    )
                    best_batch_subgroup_elide_for_shape[matrix] = min(
                        (banked, elide_arm), key=lambda p: entry[p]
                    )
                set_policies.append(PER_SHAPE_BATCH_SUBGROUP_ELIDE_POLICY)
            # --------------------------------------------------------------------------
            # Scored return-wire-precision policy (--subgroup-wire-modes).
            #
            # Reference is the ACCEPTED state: the elision policy when --elide-self-block
            # is on, else the pipelined policy. The only thing this policy changes
            # relative to that reference is that a shape whose banked arm is a pipelined
            # (optionally elided) column may bank the WIRE-ENCODED variant of that SAME
            # arm. It never re-opens the g argmin, never un-pipelines, never changes the
            # self-block treatment and never switches the fused/unfused compute path, so
            # the delta is the output leg's encoding and nothing else.
            # --------------------------------------------------------------------------
            if config.subgroup_wire_modes:
                wire_reference_policy = (
                    PER_SHAPE_BATCH_SUBGROUP_ELIDE_POLICY if config.elide_self_block
                    else PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY
                )
                reference_selection = (
                    best_batch_subgroup_elide_for_shape if config.elide_self_block
                    else best_batch_subgroup_pipe_for_shape
                )
                for matrix in distinct:
                    banked = reference_selection[matrix]
                    best_batch_subgroup_wire_for_shape[matrix] = banked
                    if not (banked.endswith(PIPE_SUFFIX)
                            or banked.endswith(ELIDE_SUFFIX)):
                        # The reference did not bank a pipelined arm for this shape, so
                        # there is no wire-encoded counterpart to substitute.
                        continue
                    biggest = sorted(needed.get(matrix, ()))[-1]
                    entry = group_ms[(matrix, biggest)]
                    arms = [banked] + [
                        banked + WIRE_SUFFIX + w
                        for w in config.subgroup_wire_modes
                        if banked + WIRE_SUFFIX + w in entry
                    ]
                    if len(arms) < 2:
                        continue
                    best_wire = min(arms[1:], key=lambda p: entry[p])
                    wire_audit[matrix] = (
                        banked, entry[banked], best_wire, entry[best_wire],
                    )
                    best_batch_subgroup_wire_for_shape[matrix] = min(
                        arms, key=lambda p: entry[p]
                    )
                set_policies.append(PER_SHAPE_BATCH_SUBGROUP_WIRE_POLICY)
        policies += set_policies

    def resolve(matrix, policy: str) -> str:
        """Mode a policy runs for one shape."""
        if policy == per_shape_policy:
            return best_mode_for_shape[matrix]
        if policy == per_shape_set_policy:
            return best_set_mode_for_shape[matrix]
        if policy == PER_SHAPE_SUBGROUP_POLICY:
            return best_subgroup_for_shape[matrix]
        if policy == PER_SHAPE_SUBGROUP_ALL_POLICY:
            return best_subgroup_all_for_shape[matrix]
        if policy == PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY:
            return best_subgroup_all_batch_for_shape[matrix]
        if policy == PER_SHAPE_SUBGROUP_ALL_BATCH_FUSED_POLICY:
            return best_subgroup_all_batch_fused_for_shape[matrix]
        if policy == PER_SHAPE_BATCH_POLICY:
            return best_batch_base_for_shape[matrix]
        if policy == PER_SHAPE_BATCH_SUBGROUP_POLICY:
            return best_batch_subgroup_for_shape[matrix]
        if policy == PER_SHAPE_BATCH_SUBGROUP_FUSED_POLICY:
            return best_batch_subgroup_fused_for_shape[matrix]
        if policy == PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY:
            return best_batch_subgroup_pipe_for_shape[matrix]
        if policy == PER_SHAPE_BATCH_SUBGROUP_ELIDE_POLICY:
            return best_batch_subgroup_elide_for_shape[matrix]
        if policy == PER_SHAPE_BATCH_SUBGROUP_WIRE_POLICY:
            return best_batch_subgroup_wire_for_shape[matrix]
        if policy == PER_SHAPE_FUSED_EXCHANGE_POLICY:
            # The fusion never changes WHICH arm a shape runs -- it only merges the
            # regions of the shapes the reference already banked on a subgroup arm -- so
            # per shape it resolves to exactly the reference's arm. That is also what
            # keeps the FLOP model identical between the two.
            return resolve(matrix, fuse_reference_policy)
        return policy

    def policy_mode(policy: str) -> str:
        """The underlying tp_mode a policy runs, for the FLOP model."""
        # The fused and pipelined arms issue identical arithmetic to the arm they suffix,
        # so they must map to the same tp_mode and be charged the same FLOPs.
        for _w in set(WIRE_MODES) | set(SUBGROUP_WIRE_MODES):
            if policy.endswith(f"{WIRE_SUFFIX}{_w}"):
                policy = policy[: -len(_w) - len(WIRE_SUFFIX)]
                break
        if policy.endswith(ELIDE_SUFFIX):
            policy = policy[: -len(ELIDE_SUFFIX)]
        if policy.endswith(PIPE_SUFFIX):
            policy = policy[: -len(PIPE_SUFFIX)]
        if policy.endswith(FUSED_SUFFIX):
            policy = policy[: -len(FUSED_SUFFIX)]
        if (
            policy == BATCH_POLICY
            or policy.startswith(SUBGROUP_PREFIX)
            or policy.startswith(BATCH_SUBGROUP_PREFIX)
        ):
            return "duplicated"
        return policy[: -len(SET_SUFFIX)] if policy.endswith(SET_SUFFIX) else policy

    def group_total(signature, policy: str) -> float:
        """Profile total for one policy: the composition is 'sum over owned groups'."""
        if policy == PER_SHAPE_FUSED_EXCHANGE_POLICY:
            # Composition: ONE fused region for the shapes it spans, plus the reference's
            # own regions for every shape it does not. Never re-attributed to a shape.
            fused = fuse_shapes.get(signature, [])
            fused_ms = fused_region_ms.get(signature, {}).get(fuse_arm)
            if len(fused) < 2 or fused_ms is None:
                return group_total(signature, fuse_reference_policy)
            spanned = set(fused)
            return fused_ms + sum(
                group_ms[(e, n)][resolve(e, fuse_reference_policy)]
                for e, n in signature
                if (e, n) not in spanned
            )
        if policy in set_policies:
            return sum(group_ms[(e, n)][resolve(e, policy)] for e, n in signature)
        return sum(per_shape[e][resolve(e, policy)] * n for e, n in signature)


    # ----------------------------------------------------------------------------------
    # Fused cross-shape subgroup region (--fuse-shape-exchanges).
    #
    # This block runs AFTER the per-shape selection because WHICH shapes it fuses is
    # defined by the reference: exactly the shapes whose banked arm is a subgroup column.
    # A shape the reference banked on anything else (duplicated_batch, distributed_set,
    # ...) is untouched, so the policy cannot be credited with a win the reference already
    # made -- what it changes is the number of exchanges and the deal, nothing else.
    #
    # The region is per RANK PROFILE, not per shape: a profile's fused set is the set of
    # shapes IT owns. Both are timed by all ranks in the same job, over the same tensors,
    # under the same composition, and the SET-COMPOSED table gains its own fused rows
    # rather than any time being re-attributed to one shape.
    # ----------------------------------------------------------------------------------
    if config.set_timing and config.fuse_shape_exchanges:
        for _candidate in (
            PER_SHAPE_SUBGROUP_ALL_BATCH_FUSED_POLICY,
            PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY,
            PER_SHAPE_SUBGROUP_ALL_POLICY,
            PER_SHAPE_SUBGROUP_POLICY,
        ):
            if _candidate in set_policies:
                fuse_reference_policy = _candidate
                break
        if fuse_reference_policy is None:
            log(
                "\nfuse_shape_exchanges=True but no subgroup policy was scored (needs "
                "--set-timing and --subgroup-sizes); skipping "
                f"'{PER_SHAPE_FUSED_EXCHANGE_POLICY}' rather than scoring it against an "
                "ambiguous reference."
            )
        else:
            for signature in profiles:
                fuse_shapes[signature] = [
                    (e, n) for e, n in signature
                    if resolve(e, fuse_reference_policy).startswith(SUBGROUP_PREFIX)
                ]
            if not any(len(v) >= 2 for v in fuse_shapes.values()):
                log(
                    "\nfuse_shape_exchanges=True but no rank profile banks a subgroup arm "
                    f"on 2+ shapes under '{fuse_reference_policy}'; there is nothing to "
                    f"fuse, so '{PER_SHAPE_FUSED_EXCHANGE_POLICY}' is not scored."
                )
                fuse_shapes = {}
            else:
                log(
                    f"\nFUSED-EXCHANGE  (per RANK PROFILE: the shapes '{fuse_reference_policy}'"
                    " banked on a subgroup"
                )
                log(
                    "                 arm, run as ONE region with ONE pair of "
                    "all_to_all_single. '_bal' also"
                )
                log(
                    "                 pools the DEAL across those shapes by greedy LPT on "
                    "ns_cost.)"
                )
                fused_header = (
                    f"  {'profile':>8}{'shapes':>8}{'matrices':>10}{'arm':>30}"
                    f"{'fused ms':>11}{'per-shape sum ms':>18}{'delta ms':>11}"
                )
                log(fused_header)
                log("  " + "-" * (len(fused_header) - 2))
                for p_index, signature in enumerate(
                    sorted(profiles, key=lambda s: -len(profiles[s]))
                ):
                    fused = fuse_shapes[signature]
                    if len(fused) < 2:
                        log(f"  {p_index:>8}{len(fused):>8}  <2 fusable shapes; not fused")
                        continue
                    reference_sum = sum(
                        group_ms[(e, n)][resolve(e, fuse_reference_policy)] for e, n in fused
                    )
                    stacks = [
                        torch.randn((n, e[0][0], e[0][1]), device="cuda", dtype=dtype)
                        for e, n in fused
                    ]
                    entry: Dict[str, float] = {}
                    for g in subgroup_sizes:
                        for balanced in (False, True):
                            # (pipe_chunks = 0) is the monolithic arm; the rest are the
                            # restructured ones, timed in the SAME job over the SAME
                            # tensors under the SAME composition.
                            depths = [0] + (
                                list(config.fused_pipe_chunks)
                                if config.pipeline_fused_exchange else []
                            )
                            # (pc, wire): wire = "" is the fp32 wire this axis has always
                            # run; the rest are transport encodings of the SAME arm, timed
                            # here so the argmin ranges over them under one composition.
                            variants = [
                                (pc, w)
                                for pc in depths
                                for w in ([""] + list(config.fused_wire_modes) if pc else [""])
                            ]
                            for pc, wire in variants:
                                arm = fused_exchange_policy(g, balanced, pc, wire)
                                entry[arm] = time_fused_group(
                                    stacks, group, config.num_ns_steps,
                                    config.coefficient_type, config.iters, config.warmup,
                                    config.use_syrk, subgroup_size=g, balanced=balanced,
                                    pipelined=bool(pc), pipe_chunks=pc, wire=wire,
                                )
                                log(
                                    f"  {p_index:>8}{len(fused):>8}"
                                    f"{sum(n for _, n in fused):>10}{arm:>30}"
                                    f"{entry[arm]:>11.3f}{reference_sum:>18.3f}"
                                    f"{entry[arm] - reference_sum:>+11.3f}"
                                )
                                torch.cuda.empty_cache()
                    fused_region_ms[signature] = entry
                    del stacks
                    torch.cuda.empty_cache()
                # ONE global arm for the policy -- never a per-profile cherry-pick -- chosen
                # by the same criterion the step cost uses: the max over rank profiles.
                arm_names = sorted(
                    set.intersection(*(set(v) for v in fused_region_ms.values()))
                )
                scored = {}
                for arm in arm_names:
                    fuse_arm = arm
                    scored[arm] = max(
                        group_total(sig, PER_SHAPE_FUSED_EXCHANGE_POLICY) for sig in profiles
                    )
                fuse_arm = min(arm_names, key=lambda a: scored[a])
                log(
                    f"  fused arm scored: "
                    + ", ".join(f"{a}={scored[a]:.3f}" for a in arm_names)
                    + f"  -> selected {fuse_arm}"
                )
                # Substitution discipline, asserted rather than assumed: the fused region
                # may only span shapes the reference banked on a subgroup arm, and every
                # shape it does NOT span must keep the reference's arm exactly.
                for signature in profiles:
                    spanned = set(fuse_shapes.get(signature, []))
                    for e, n in signature:
                        ref_arm = resolve(e, fuse_reference_policy)
                        if (e, n) in spanned:
                            assert ref_arm.startswith(SUBGROUP_PREFIX), (
                                f"fused region spans {e} whose reference arm is {ref_arm}"
                            )
                        else:
                            assert resolve(e, PER_SHAPE_FUSED_EXCHANGE_POLICY) == ref_arm, (
                                f"unfused shape {e} re-banked away from {ref_arm}"
                            )
                set_policies.append(PER_SHAPE_FUSED_EXCHANGE_POLICY)
                policies.append(PER_SHAPE_FUSED_EXCHANGE_POLICY)

    if best_mode_for_shape:
        log("\nPER-SHAPE MODE SELECTION  (policy 'per_shape' runs this mode for this shape)")
        selection_header = f"{'local shard':>13}{'all-gathered':>14}{'selected':>13}{'ms':>9}"
        for mode in config.modes:
            selection_header += f"{mode:>13}"
        log(selection_header)
        log("-" * len(selection_header))
        for matrix in distinct:
            (rows, cols), shard_count = matrix
            chosen = best_mode_for_shape[matrix]
            line = (
                f"{f'{rows}x{cols}':>13}{f'{rows * shard_count}x{cols}':>14}"
                f"{chosen:>13}{per_shape[matrix][chosen]:>9.3f}"
            )
            for mode in config.modes:
                line += f"{per_shape[matrix][mode]:>13.3f}"
            log(line)

    log("\nPER-PROFILE  (sum over the matrices that profile owns)")
    profile_header = f"{'ranks':>6}  {'owns':<38}"
    for policy in policies:
        profile_header += f"{policy:>18}{'TF/s':>8}"
    log(profile_header)
    log("-" * len(profile_header))
    totals: Dict[str, List[float]] = {policy: [] for policy in policies}
    useful_totals: Dict[str, List[float]] = {policy: [] for policy in policies}
    for signature, dp_ranks in sorted(profiles.items(), key=lambda kv: -len(kv[1])):
        owns = " + ".join(f"{n}x[{e[0][0]}x{e[0][1]}]" for e, n in signature)
        line = f"{len(dp_ranks):>6}  {owns:<38}"
        for policy in policies:
            total_ms = group_total(signature, policy)
            useful = sum(
                flop_model(
                    e, policy_mode(resolve(e, policy)), config.num_ns_steps, group_size,
                    config.use_syrk,
                )[1]
                * n
                for e, n in signature
            )
            totals[policy].append(total_ms)
            useful_totals[policy].append(useful)
            line += f"{total_ms:>17.3f}m{useful / 1e9 / total_ms:>8.0f}"
        log(line)

    log("-" * len(profile_header))
    step = f"{'':>6}  {'STEP COST = slowest profile':<38}"
    for policy in policies:
        slowest = max(range(len(totals[policy])), key=lambda i: totals[policy][i])
        step += (
            f"{totals[policy][slowest]:>17.3f}m"
            f"{useful_totals[policy][slowest] / 1e9 / totals[policy][slowest]:>8.0f}"
        )
    log(step)
    imbalance = f"{'':>6}  {'imbalance (max / mean)':<38}"
    for policy in policies:
        mean = sum(t * len(r) for t, r in zip(totals[policy], profiles.values())) / dp_size
        imbalance += f"{max(totals[policy]) / mean:>17.2f}x{'':>8}"
    log(imbalance)

    # Under --set-timing the two compositions are NOT interchangeable, so the reported
    # step is taken over the set-composed policies only and the unbatched set-composed
    # baseline is printed beside it. That is what keeps the batched arm from ever being
    # credited with the composition change: both numbers come from the same job, over the
    # same tensors, under the same composition.
    if config.set_timing:
        candidates = set_policies
        # The reference arm must contain no candidate mechanism: neither the batched arm
        # nor any subgroup arm. What is left is the unmodified algorithm under the same
        # composition, which is what a candidate delta is subtracted from.
        baseline_pool = [
            p for p in set_policies
            if p not in (BATCH_POLICY, PER_SHAPE_SUBGROUP_POLICY,
                         PER_SHAPE_SUBGROUP_ALL_POLICY,
                         PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY,
                         PER_SHAPE_SUBGROUP_ALL_BATCH_FUSED_POLICY,
                         PER_SHAPE_BATCH_POLICY,
                         PER_SHAPE_BATCH_SUBGROUP_POLICY,
                         PER_SHAPE_BATCH_SUBGROUP_FUSED_POLICY,
                         PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY,
                         PER_SHAPE_BATCH_SUBGROUP_ELIDE_POLICY,
                         PER_SHAPE_BATCH_SUBGROUP_WIRE_POLICY,
                         PER_SHAPE_FUSED_EXCHANGE_POLICY)
            and not p.startswith(SUBGROUP_PREFIX)
            and not p.startswith(BATCH_SUBGROUP_PREFIX)
            and not p.startswith(FUSED_EXCHANGE_PREFIX)
            and not p.endswith(FUSED_SUFFIX)
            and not p.endswith(PIPE_SUFFIX)
            and not p.endswith(ELIDE_SUFFIX)
            and not p.endswith(BALANCED_SUFFIX)
        ]
        baseline = min(baseline_pool, key=lambda p: max(totals[p]))
        log(
            f"\nset-composed baseline (unbatched): {baseline} "
            f"({max(totals[baseline]):.3f} ms)"
        )
        if PER_SHAPE_SUBGROUP_POLICY in set_policies:
            # Printed explicitly so the scored pair is one grep away and can never be
            # formed from arms measured in different jobs or under different compositions.
            log(
                f"subgroup candidate arm: {PER_SHAPE_SUBGROUP_POLICY} "
                f"({max(totals[PER_SHAPE_SUBGROUP_POLICY]):.3f} ms), "
                f"delta vs {baseline} = "
                f"{max(totals[PER_SHAPE_SUBGROUP_POLICY]) - max(totals[baseline]):+.3f} ms"
            )
            log(
                "  subgroup selection: "
                + ", ".join(
                    f"{e[0][0] * e[1]}x{e[0][1]}={best_subgroup_for_shape[e]}"
                    for e in distinct
                )
            )
        if PER_SHAPE_SUBGROUP_ALL_POLICY in set_policies:
            # This candidate's scored pair. Its INCREMENTAL reference is the accepted
            # state (per_shape_sub_set), because that policy already banked the
            # duplicated_set shapes; what is new here is the substitution on the shapes
            # whose winner is distributed_set. Both deltas are printed so neither the
            # incremental nor the cumulative number has to be reconstructed by hand.
            wide = max(totals[PER_SHAPE_SUBGROUP_ALL_POLICY])
            narrow = max(totals[PER_SHAPE_SUBGROUP_POLICY])
            log(
                f"subgroup-all candidate arm: {PER_SHAPE_SUBGROUP_ALL_POLICY} "
                f"({wide:.3f} ms), delta vs {PER_SHAPE_SUBGROUP_POLICY} = "
                f"{wide - narrow:+.3f} ms, delta vs {baseline} = "
                f"{wide - max(totals[baseline]):+.3f} ms"
            )
            log(
                "  subgroup-all selection: "
                + ", ".join(
                    f"{e[0][0] * e[1]}x{e[0][1]}={best_subgroup_all_for_shape[e]}"
                    for e in distinct
                )
            )
            # Per-shape audit: the substitution rule is an argmin over measured medians,
            # so print the pair it chose between, on the largest owned count, for every
            # shape that HAS a subgroup column. Without this the group delta cannot be
            # attributed to a region.
            audit_header = (
                f"  {'all-gathered':>14}{'base policy':>19}{'base ms':>10}"
                f"{'best subgroup':>19}{'subgroup ms':>13}{'chosen':>19}"
            )
            log("  subgroup-all per-shape audit (largest owned count):")
            log(audit_header)
            for e in distinct:
                if e not in subgroup_audit:
                    continue
                base_p, base_ms, arm_p, arm_ms = subgroup_audit[e]
                log(
                    f"  {f'{e[0][0] * e[1]}x{e[0][1]}':>14}{base_p:>19}{base_ms:>10.3f}"
                    f"{arm_p:>19}{arm_ms:>13.3f}{best_subgroup_all_for_shape[e]:>19}"
                )
        if PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY in set_policies:
            # This candidate's scored pair. Its INCREMENTAL reference is the accepted
            # state (per_shape_sub_all_set), which already banked every subgroup
            # substitution; what is new here is the batched column on the replicated
            # (shard_count == 1) shapes. Both deltas are printed so neither the incremental
            # nor the cumulative number has to be reconstructed by hand.
            cand = max(totals[PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY])
            ref = max(totals[PER_SHAPE_SUBGROUP_ALL_POLICY])
            log(
                f"batch-replicated reference arm (accepted state): "
                f"{PER_SHAPE_SUBGROUP_ALL_POLICY} ({ref:.3f} ms)"
            )
            log(
                f"batch-replicated candidate arm: "
                f"{PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY} ({cand:.3f} ms), delta vs "
                f"{PER_SHAPE_SUBGROUP_ALL_POLICY} = {cand - ref:+.3f} ms, delta vs "
                f"{baseline} = {cand - max(totals[baseline]):+.3f} ms"
            )
            log(
                "  batch-replicated selection: "
                + ", ".join(
                    f"{e[0][0] * e[1]}x{e[0][1]}={best_subgroup_all_batch_for_shape[e]}"
                    for e in distinct
                )
            )
            # Per-shape audit over the shapes the argmin actually widened, i.e. the
            # replicated ones. Without it the group delta cannot be attributed to a region.
            audit_header = (
                f"  {'all-gathered':>14}{'shard_count':>13}{'base policy':>25}"
                f"{'base ms':>10}{'batched arm':>19}{'batched ms':>12}{'chosen':>25}"
            )
            log("  batch-replicated per-shape audit (largest owned count, "
                "shard_count == 1 shapes only):")
            log(audit_header)
            for e in distinct:
                if e not in batch_replicated_audit:
                    continue
                base_p, base_ms, arm_p, arm_ms = batch_replicated_audit[e]
                log(
                    f"  {f'{e[0][0] * e[1]}x{e[0][1]}':>14}{e[1]:>13}{base_p:>25}"
                    f"{base_ms:>10.3f}{arm_p:>19}{arm_ms:>12.3f}"
                    f"{best_subgroup_all_batch_for_shape[e]:>25}"
                )
        if PER_SHAPE_SUBGROUP_ALL_BATCH_FUSED_POLICY in set_policies:
            # This candidate's scored pair. Its INCREMENTAL reference is the accepted state
            # (per_shape_sub_all_batch_set), which already banked the replicated batched
            # column; what is new here is routing THAT column through the fused entry
            # point. Both arms come from this job, over the same tensors, under the same
            # composition, so the pair can never be assembled from mismatched runs -- and
            # because the shapes it touches are exactly the ones the cross-shape fused
            # exchange never spans, the delta is additive onto the composed axis total.
            cand = max(totals[PER_SHAPE_SUBGROUP_ALL_BATCH_FUSED_POLICY])
            ref = max(totals[PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY])
            log(
                f"batch-replicated-fused reference arm (accepted state): "
                f"{PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY} ({ref:.3f} ms)"
            )
            log(
                f"batch-replicated-fused candidate arm: "
                f"{PER_SHAPE_SUBGROUP_ALL_BATCH_FUSED_POLICY} ({cand:.3f} ms), delta vs "
                f"{PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY} = {cand - ref:+.3f} ms, delta vs "
                f"{baseline} = {cand - max(totals[baseline]):+.3f} ms"
            )
            log(
                "  batch-replicated-fused selection: "
                + ", ".join(
                    f"{e[0][0] * e[1]}x{e[0][1]}="
                    f"{best_subgroup_all_batch_fused_for_shape[e]}"
                    for e in distinct
                )
            )
            audit_header = (
                f"  {'all-gathered':>14}{'shard_count':>13}{'base policy':>25}"
                f"{'base ms':>10}{'fused arm':>25}{'fused ms':>12}{'chosen':>25}"
            )
            log("  batch-replicated-fused per-shape audit (largest owned count, shapes "
                f"whose reference arm is '{BATCH_POLICY}'):")
            log(audit_header)
            for e in distinct:
                if e not in batch_replicated_fused_audit:
                    continue
                base_p, base_ms, arm_p, arm_ms = batch_replicated_fused_audit[e]
                log(
                    f"  {f'{e[0][0] * e[1]}x{e[0][1]}':>14}{e[1]:>13}{base_p:>25}"
                    f"{base_ms:>10.3f}{arm_p:>25}{arm_ms:>12.3f}"
                    f"{best_subgroup_all_batch_fused_for_shape[e]:>25}"
                )
        if PER_SHAPE_BATCH_SUBGROUP_POLICY in set_policies:
            # This candidate's scored pair, printed so it is one grep away and can never be
            # formed from arms measured in different jobs or under different compositions.
            cand = max(totals[PER_SHAPE_BATCH_SUBGROUP_POLICY])
            ref = max(totals[PER_SHAPE_BATCH_POLICY])
            log(
                f"batch-subgroup reference arm (accepted state): "
                f"{PER_SHAPE_BATCH_POLICY} ({ref:.3f} ms)"
            )
            log(
                f"batch-subgroup candidate arm: {PER_SHAPE_BATCH_SUBGROUP_POLICY} "
                f"({cand:.3f} ms), delta vs {PER_SHAPE_BATCH_POLICY} = "
                f"{cand - ref:+.3f} ms, delta vs {baseline} = "
                f"{cand - max(totals[baseline]):+.3f} ms"
            )
            log(
                "  batch-subgroup selection: "
                + ", ".join(
                    f"{e[0][0] * e[1]}x{e[0][1]}={best_batch_subgroup_for_shape[e]}"
                    for e in distinct
                )
            )
            audit_header = (
                f"  {'all-gathered':>14}{'base policy':>25}{'base ms':>10}"
                f"{'best batch subgroup':>25}{'subgroup ms':>13}{'chosen':>25}"
            )
            log("  batch-subgroup per-shape audit (largest owned count):")
            log(audit_header)
            for e in distinct:
                if e not in batch_subgroup_audit:
                    continue
                base_p, base_ms, arm_p, arm_ms = batch_subgroup_audit[e]
                log(
                    f"  {f'{e[0][0] * e[1]}x{e[0][1]}':>14}{base_p:>25}{base_ms:>10.3f}"
                    f"{arm_p:>25}{arm_ms:>13.3f}"
                    f"{best_batch_subgroup_for_shape[e]:>25}"
                )
        if PER_SHAPE_BATCH_SUBGROUP_FUSED_POLICY in totals:
            ref = max(totals[PER_SHAPE_BATCH_SUBGROUP_POLICY])
            cand = max(totals[PER_SHAPE_BATCH_SUBGROUP_FUSED_POLICY])
            log(
                f"\nns-step-fusion reference arm: {PER_SHAPE_BATCH_SUBGROUP_POLICY} "
                f"({ref:.3f} ms)  <- the ACCEPTED state"
            )
            log(
                f"ns-step-fusion candidate arm: {PER_SHAPE_BATCH_SUBGROUP_FUSED_POLICY} "
                f"({cand:.3f} ms), delta vs {PER_SHAPE_BATCH_SUBGROUP_POLICY} = "
                f"{cand - ref:+.3f} ms, delta vs {baseline} = "
                f"{cand - max(totals[baseline]):+.3f} ms"
            )
            log("  ns-step-fusion per-shape audit (largest owned count):")
            log(
                f"  {'all-gathered':>14}{'banked arm':>28}{'banked ms':>11}"
                f"{'fused arm':>34}{'fused ms':>11}{'chosen':>34}"
            )
            for e in distinct:
                if e not in fused_audit:
                    continue
                base_p, base_ms, arm_p, arm_ms = fused_audit[e]
                log(
                    f"  {f'{e[0][0] * e[1]}x{e[0][1]}':>14}{base_p:>28}{base_ms:>11.3f}"
                    f"{arm_p:>34}{arm_ms:>11.3f}"
                    f"{best_batch_subgroup_fused_for_shape[e]:>34}"
                )
            # A fused arm must never be banked where the reference did not already bank the
            # very same subgroup arm unfused: the fusion is a substitution, not a new arm.
            for e, p_sel in best_batch_subgroup_fused_for_shape.items():
                if p_sel.endswith(FUSED_SUFFIX):
                    assert p_sel[: -len(FUSED_SUFFIX)] == best_batch_subgroup_for_shape[e], (
                        f"fused arm {p_sel} banked on {e} whose reference arm is "
                        f"{best_batch_subgroup_for_shape[e]}"
                    )
        if PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY in totals:
            ref = max(totals[pipe_reference_policy])
            cand = max(totals[PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY])
            log(
                f"\nexchange-restructure reference arm: {pipe_reference_policy} "
                f"({ref:.3f} ms)  <- the ACCEPTED state"
            )
            log(
                f"exchange-restructure candidate arm: "
                f"{PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY} ({cand:.3f} ms), delta vs "
                f"{pipe_reference_policy} = {cand - ref:+.3f} ms, delta vs {baseline} = "
                f"{cand - max(totals[baseline]):+.3f} ms"
            )
            log("  exchange-restructure per-shape audit (largest owned count):")
            log(
                f"  {'all-gathered':>14}{'banked arm':>34}{'banked ms':>11}"
                f"{'pipelined arm':>39}{'pipe ms':>11}{'chosen':>39}"
            )
            for e in distinct:
                if e not in pipe_audit:
                    continue
                base_p, base_ms, arm_p, arm_ms = pipe_audit[e]
                log(
                    f"  {f'{e[0][0] * e[1]}x{e[0][1]}':>14}{base_p:>34}{base_ms:>11.3f}"
                    f"{arm_p:>39}{arm_ms:>11.3f}"
                    f"{best_batch_subgroup_pipe_for_shape[e]:>39}"
                )
            # A pipelined arm must never be banked where the reference did not already bank
            # the very same arm unpipelined: the restructure is a substitution, not a new
            # arm, and it must not switch the fused/unfused compute path either.
            for e, p_sel in best_batch_subgroup_pipe_for_shape.items():
                if p_sel.endswith(PIPE_SUFFIX):
                    assert p_sel[: -len(PIPE_SUFFIX)] == resolve(e, pipe_reference_policy), (
                        f"pipelined arm {p_sel} banked on {e} whose reference arm is "
                        f"{resolve(e, pipe_reference_policy)}"
                    )
        if PER_SHAPE_BATCH_SUBGROUP_ELIDE_POLICY in totals:
            ref = max(totals[elide_reference_policy])
            cand = max(totals[PER_SHAPE_BATCH_SUBGROUP_ELIDE_POLICY])
            log(
                f"\nself-block-elision reference arm: {elide_reference_policy} "
                f"({ref:.3f} ms)  <- the ACCEPTED state"
            )
            log(
                f"self-block-elision candidate arm: "
                f"{PER_SHAPE_BATCH_SUBGROUP_ELIDE_POLICY} ({cand:.3f} ms), delta vs "
                f"{elide_reference_policy} = {cand - ref:+.3f} ms, delta vs {baseline} = "
                f"{cand - max(totals[baseline]):+.3f} ms"
            )
            log("  self-block-elision per-shape audit (largest owned count):")
            log(
                f"  {'all-gathered':>14}{'banked arm':>39}{'banked ms':>11}"
                f"{'elided arm':>49}{'elide ms':>11}{'chosen':>49}"
            )
            for e in distinct:
                if e not in elide_audit:
                    continue
                base_p, base_ms, arm_p, arm_ms = elide_audit[e]
                log(
                    f"  {f'{e[0][0] * e[1]}x{e[0][1]}':>14}{base_p:>39}{base_ms:>11.3f}"
                    f"{arm_p:>49}{arm_ms:>11.3f}"
                    f"{best_batch_subgroup_elide_for_shape[e]:>49}"
                )
            # An elided arm must never be banked where the reference did not already bank
            # the very same arm unelided: the elision is a substitution, not a new arm.
            for e, p_sel in best_batch_subgroup_elide_for_shape.items():
                if p_sel.endswith(ELIDE_SUFFIX):
                    assert p_sel[: -len(ELIDE_SUFFIX)] == resolve(
                        e, elide_reference_policy), (
                        f"elided arm {p_sel} banked on {e} whose reference arm is "
                        f"{resolve(e, elide_reference_policy)}"
                    )
        if PER_SHAPE_BATCH_SUBGROUP_WIRE_POLICY in totals:
            ref = max(totals[wire_reference_policy])
            cand = max(totals[PER_SHAPE_BATCH_SUBGROUP_WIRE_POLICY])
            log(
                f"\nreturn-wire reference arm: {wire_reference_policy} "
                f"({ref:.3f} ms)  <- the ACCEPTED state"
            )
            log(
                f"return-wire candidate arm: "
                f"{PER_SHAPE_BATCH_SUBGROUP_WIRE_POLICY} ({cand:.3f} ms), delta vs "
                f"{wire_reference_policy} = {cand - ref:+.3f} ms, delta vs {baseline} = "
                f"{cand - max(totals[baseline]):+.3f} ms"
            )
            log("  return-wire per-shape audit (largest owned count):")
            log(
                f"  {'all-gathered':>14}{'banked arm':>49}{'banked ms':>11}"
                f"{'wire arm':>59}{'wire ms':>11}{'chosen':>59}"
            )
            for e in distinct:
                if e not in wire_audit:
                    continue
                base_p, base_ms, arm_p, arm_ms = wire_audit[e]
                log(
                    f"  {f'{e[0][0] * e[1]}x{e[0][1]}':>14}{base_p:>49}{base_ms:>11.3f}"
                    f"{arm_p:>59}{arm_ms:>11.3f}"
                    f"{best_batch_subgroup_wire_for_shape[e]:>59}"
                )
            # Every wire arm on every shape, so a negative outcome is diagnosable and the
            # per-mode split is visible rather than collapsed into the argmin.
            log("  return-wire all-arm table (group ms at the largest owned count):")
            for e in distinct:
                biggest = sorted(needed.get(e, ()))[-1]
                entry = group_ms[(e, biggest)]
                arms = sorted(
                    a for a in entry
                    if any(WIRE_SUFFIX + w in a for w in SUBGROUP_WIRE_MODES)
                )
                if not arms:
                    continue
                log(
                    f"    {f'{e[0][0] * e[1]}x{e[0][1]}':>14}: "
                    + ", ".join(f"{a}={entry[a]:.3f}" for a in arms)
                )
            # A wire-encoded arm must never be banked where the reference did not already
            # bank the very same arm at fp32: the encoding is a substitution, not a new
            # arm, and it must not change the self-block treatment or the compute path.
            for e, p_sel in best_batch_subgroup_wire_for_shape.items():
                for w in SUBGROUP_WIRE_MODES:
                    suffix = WIRE_SUFFIX + w
                    if p_sel.endswith(suffix):
                        assert p_sel[: -len(suffix)] == resolve(
                            e, wire_reference_policy), (
                            f"wire arm {p_sel} banked on {e} whose reference arm is "
                            f"{resolve(e, wire_reference_policy)}"
                        )

        if PER_SHAPE_FUSED_EXCHANGE_POLICY in totals:
            ref = max(totals[fuse_reference_policy])
            cand = max(totals[PER_SHAPE_FUSED_EXCHANGE_POLICY])
            log(
                f"\nexchange-fusion reference arm: {fuse_reference_policy} "
                f"({ref:.3f} ms)  <- the ACCEPTED state"
            )
            log(
                f"exchange-fusion candidate arm: {PER_SHAPE_FUSED_EXCHANGE_POLICY} "
                f"({cand:.3f} ms) at {fuse_arm}, delta vs {fuse_reference_policy} = "
                f"{cand - ref:+.3f} ms, delta vs {baseline} = "
                f"{cand - max(totals[baseline]):+.3f} ms"
            )
            # BOTH rank profiles are printed, with the fused region broken out against the
            # per-shape sum it replaces, so a negative outcome is diagnosable rather than
            # merely negative: an exchange-only win shows up on the non-'_bal' arms, a
            # deal win only on the '_bal' ones.
            log(
                "  exchange-fusion per-profile audit (fused region vs the per-shape "
                "regions it replaces):"
            )
            log(
                f"  {'profile':>8}{'ranks':>7}{'fused shapes':>50}"
                f"{'per-shape sum':>15}{'fused':>10}{'delta':>10}{'profile total':>15}"
            )
            for p_index, signature in enumerate(
                sorted(profiles, key=lambda s: -len(profiles[s]))
            ):
                fused = fuse_shapes.get(signature, [])
                spanned = ", ".join(f"{n}x[{e[0][0] * e[1]}x{e[0][1]}]" for e, n in fused)
                fused_ms = fused_region_ms.get(signature, {}).get(fuse_arm)
                reference_sum = sum(
                    group_ms[(e, n)][resolve(e, fuse_reference_policy)] for e, n in fused
                )
                total_ms = group_total(signature, PER_SHAPE_FUSED_EXCHANGE_POLICY)
                if fused_ms is None or len(fused) < 2:
                    log(
                        f"  {p_index:>8}{len(profiles[signature]):>7}{spanned:>50}"
                        f"{'not fused':>15}{'-':>10}{'-':>10}{total_ms:>15.3f}"
                    )
                    continue
                log(
                    f"  {p_index:>8}{len(profiles[signature]):>7}{spanned:>50}"
                    f"{reference_sum:>15.3f}{fused_ms:>10.3f}"
                    f"{fused_ms - reference_sum:>+10.3f}{total_ms:>15.3f}"
                )
            # Every arm, every profile: the full table the plan asked the verifier to see.
            log("  exchange-fusion all-arm table (fused region ms):")
            for p_index, signature in enumerate(
                sorted(profiles, key=lambda s: -len(profiles[s]))
            ):
                entry = fused_region_ms.get(signature)
                if not entry:
                    continue
                log(
                    f"    profile {p_index}: "
                    + ", ".join(f"{a}={entry[a]:.3f}" for a in sorted(entry))
                )
    else:
        candidates = policies

    best = min(candidates, key=lambda p: max(totals[p]))
    log(f"\nfastest step: {best} ({max(totals[best]):.3f} ms)")
    if best in (per_shape_policy, per_shape_set_policy, PER_SHAPE_SUBGROUP_POLICY,
                PER_SHAPE_SUBGROUP_ALL_POLICY, PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY,
                PER_SHAPE_BATCH_POLICY, PER_SHAPE_BATCH_SUBGROUP_POLICY,
                PER_SHAPE_BATCH_SUBGROUP_FUSED_POLICY,
                PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY,
                PER_SHAPE_BATCH_SUBGROUP_ELIDE_POLICY,
                PER_SHAPE_BATCH_SUBGROUP_WIRE_POLICY,
                PER_SHAPE_FUSED_EXCHANGE_POLICY):
        selection = {
            per_shape_policy: best_mode_for_shape,
            per_shape_set_policy: best_set_mode_for_shape,
            PER_SHAPE_SUBGROUP_POLICY: best_subgroup_for_shape,
            PER_SHAPE_SUBGROUP_ALL_POLICY: best_subgroup_all_for_shape,
            PER_SHAPE_SUBGROUP_ALL_BATCH_POLICY: best_subgroup_all_batch_for_shape,
            PER_SHAPE_BATCH_POLICY: best_batch_base_for_shape,
            PER_SHAPE_BATCH_SUBGROUP_POLICY: best_batch_subgroup_for_shape,
            PER_SHAPE_BATCH_SUBGROUP_FUSED_POLICY: best_batch_subgroup_fused_for_shape,
            PER_SHAPE_BATCH_SUBGROUP_PIPE_POLICY: best_batch_subgroup_pipe_for_shape,
            PER_SHAPE_BATCH_SUBGROUP_ELIDE_POLICY: best_batch_subgroup_elide_for_shape,
            PER_SHAPE_BATCH_SUBGROUP_WIRE_POLICY: best_batch_subgroup_wire_for_shape,
            # The fusion changes no per-shape arm; it merges regions. Reported as the
            # reference's own selection plus the fused arm, printed below.
            PER_SHAPE_FUSED_EXCHANGE_POLICY: {
                e: resolve(e, PER_SHAPE_FUSED_EXCHANGE_POLICY) for e in distinct
            } if fuse_reference_policy else {},
        }[best]
        log(
            f"  {best} selection: "
            + ", ".join(f"{e[0][0] * e[1]}x{e[0][1]}={selection[e]}" for e in distinct)
        )
        if best == PER_SHAPE_FUSED_EXCHANGE_POLICY:
            log(
                f"  {best} fused arm: {fuse_arm}; the listed per-shape arms are the "
                f"reference's ({fuse_reference_policy}) and are unchanged -- the shapes "
                "marked with a subgroup arm run inside ONE fused region per profile."
            )
    if config.set_timing:
        log(
            f"  set-composed policies scored: {', '.join(candidates)}; "
            f"batch_chunk={config.batch_chunk}. Isolated-composition columns "
            f"({', '.join(policies[: len(policies) - len(set_policies)])}) are printed for "
            "continuity and are NOT scored."
        )
    log("notes:")
    log("  useful TF/s already discounts redundancy. duplicated recomputes the whole matrix")
    log("  on every rank, so its useful TF/s is its issued TF/s divided by the group size")
    log(f"  ({group_size} here); do not apply that factor a second time.")
    log("  blockwise issues less than useful because it orthogonalizes blocks rather than the")
    log("  matrix, a cheaper and different computation, so its TF/s is not comparable to the")
    log("  other two on equal terms.")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
