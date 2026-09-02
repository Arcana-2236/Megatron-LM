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
    """
    if partition_dim is None:
        # Replicated weight: nothing to gather, same non-TP fallback newton_schulz_tp takes.
        return newton_schulz(x, steps, coefficient_type, use_syrk=use_syrk)
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
PER_SHAPE_SUBGROUP_POLICY = "per_shape_sub_set"


def subgroup_policy(subgroup_size: int) -> str:
    return f"{SUBGROUP_PREFIX}{subgroup_size}"


def newton_schulz_tp_subgroup(
    stack: torch.Tensor,
    steps: int,
    coefficient_type: str,
    tp_group,
    partition_dim: int | None,
    tp_mode: str,
    use_syrk: bool = False,
    subgroup_size: int = 0,
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

    # ---- compute: the SAME 2-D newton_schulz duplicated runs, on 1/k as many matrices ---
    full = [
        newton_schulz(global_x[m], steps, coefficient_type, use_syrk=use_syrk)
        for m in range(mine)
    ]
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
    del full
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


def time_group(
    stack, group, mode, steps, coefficient_type, iters, warmup, shard_count,
    use_syrk=False, batched=False, batch_chunk=0, subgroup_size=0,
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
                if base != duplicated_set or not arms:
                    best_subgroup_for_shape[matrix] = base
                    continue
                best_g = min(arms, key=lambda g: entry[subgroup_policy(g)])
                best_subgroup_for_shape[matrix] = min(
                    (base, subgroup_policy(best_g)), key=lambda p: entry[p]
                )
            set_policies.append(PER_SHAPE_SUBGROUP_POLICY)
        policies += set_policies

    def resolve(matrix, policy: str) -> str:
        """Mode a policy runs for one shape."""
        if policy == per_shape_policy:
            return best_mode_for_shape[matrix]
        if policy == per_shape_set_policy:
            return best_set_mode_for_shape[matrix]
        if policy == PER_SHAPE_SUBGROUP_POLICY:
            return best_subgroup_for_shape[matrix]
        return policy

    def policy_mode(policy: str) -> str:
        """The underlying tp_mode a policy runs, for the FLOP model."""
        if policy == BATCH_POLICY or policy.startswith(SUBGROUP_PREFIX):
            return "duplicated"
        return policy[: -len(SET_SUFFIX)] if policy.endswith(SET_SUFFIX) else policy

    def group_total(signature, policy: str) -> float:
        """Profile total for one policy: the composition is 'sum over owned groups'."""
        if policy in set_policies:
            return sum(group_ms[(e, n)][resolve(e, policy)] for e, n in signature)
        return sum(per_shape[e][resolve(e, policy)] * n for e, n in signature)

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
            if p != BATCH_POLICY and p != PER_SHAPE_SUBGROUP_POLICY
            and not p.startswith(SUBGROUP_PREFIX)
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
    else:
        candidates = policies

    best = min(candidates, key=lambda p: max(totals[p]))
    log(f"\nfastest step: {best} ({max(totals[best]):.3f} ms)")
    if best in (per_shape_policy, per_shape_set_policy, PER_SHAPE_SUBGROUP_POLICY):
        selection = {
            per_shape_policy: best_mode_for_shape,
            per_shape_set_policy: best_set_mode_for_shape,
            PER_SHAPE_SUBGROUP_POLICY: best_subgroup_for_shape,
        }[best]
        log(
            f"  {best} selection: "
            + ", ".join(f"{e[0][0] * e[1]}x{e[0][1]}={selection[e]}" for e in distinct)
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
