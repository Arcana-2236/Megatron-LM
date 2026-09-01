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
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp

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

    def resolve(matrix, policy: str) -> str:
        """Mode a policy runs for one shape."""
        return best_mode_for_shape[matrix] if policy == per_shape_policy else policy

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
        profile_header += f"{policy:>12}{'TF/s':>8}"
    log(profile_header)
    log("-" * len(profile_header))
    totals: Dict[str, List[float]] = {policy: [] for policy in policies}
    useful_totals: Dict[str, List[float]] = {policy: [] for policy in policies}
    for signature, dp_ranks in sorted(profiles.items(), key=lambda kv: -len(kv[1])):
        owns = " + ".join(f"{n}x[{e[0][0]}x{e[0][1]}]" for e, n in signature)
        line = f"{len(dp_ranks):>6}  {owns:<38}"
        for policy in policies:
            total_ms = sum(per_shape[e][resolve(e, policy)] * n for e, n in signature)
            useful = sum(
                flop_model(
                    e, resolve(e, policy), config.num_ns_steps, group_size, config.use_syrk
                )[1]
                * n
                for e, n in signature
            )
            totals[policy].append(total_ms)
            useful_totals[policy].append(useful)
            line += f"{total_ms:>11.3f}m{useful / 1e9 / total_ms:>8.0f}"
        log(line)

    log("-" * len(profile_header))
    step = f"{'':>6}  {'STEP COST = slowest profile':<38}"
    for policy in policies:
        slowest = max(range(len(totals[policy])), key=lambda i: totals[policy][i])
        step += (
            f"{totals[policy][slowest]:>11.3f}m"
            f"{useful_totals[policy][slowest] / 1e9 / totals[policy][slowest]:>8.0f}"
        )
    log(step)
    imbalance = f"{'':>6}  {'imbalance (max / mean)':<38}"
    for policy in policies:
        mean = sum(t * len(r) for t, r in zip(totals[policy], profiles.values())) / dp_size
        imbalance += f"{max(totals[policy]) / mean:>11.2f}x{'':>8}"
    log(imbalance)

    best = min(policies, key=lambda p: max(totals[p]))
    log(f"\nfastest step: {best} ({max(totals[best]):.3f} ms)")
    if best == per_shape_policy:
        log(
            "  per_shape selection: "
            + ", ".join(
                f"{e[0][0] * e[1]}x{e[0][1]}={best_mode_for_shape[e]}" for e in distinct
            )
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
