# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused batched Newton-Schulz for the two EGTP expert shapes.

stage-3 / plan.md rank 2, ``egtp-ns-step-fusion``.

What this replaces
------------------
``emerging_optimizers.orthogonalized_optimizers.muon_utils.newton_schulz`` on a 3-D
stack, as called from the batched-subgroup compute loop of
``bench_ns_strategies.newton_schulz_tp_subgroup``::

    full_t[a : a + chunk] = newton_schulz(global_x[a : a + chunk], steps, ct, use_syrk)

Around the 5-step chain the library pays, per region, three full-size fp32 passes that
carry no arithmetic at all:

  prologue   ``F.normalize(x, p=2, dim=(-2,-1))``  -- a reduction pass over the fp32
             stack, then a second pass that reads fp32 and writes fp32 -- followed by
             ``X.to(bfloat16)``, a third pass reading fp32 and writing bf16.
  epilogue   ``X.to(torch.float32)`` materialises an fp32 temporary, and the caller's
             ``full_t[a:a+chunk] = ...`` then copies that temporary again -- and for the
             ``5120x2048`` shape the copy is *strided*, because ``newton_schulz`` returns
             ``X.mT`` after whitening the smaller dimension.

Counting only compulsory DRAM traffic at the EGTP region size (96 matrices of
``2048 x 5120``; ``B*M*N = 1.007e9`` elements, 4.03 GB in fp32, 2.01 GB in bf16):

  prologue, library   4.03 (norm reduce) + 4.03 + 4.03 (divide) + 4.03 + 2.01 (cast)
                      = 18.1 GB
  prologue, fused     4.03 (norm reduce, still torch's) + 4.03 + 2.01 (scale+cast in
                      ONE pass) = 10.1 GB
  epilogue, library   2.01 + 4.03 (to fp32) + 4.03 + 4.03 (copy into full_t) = 14.1 GB
  epilogue, fused     2.01 + 4.03 (cast straight into full_t's slice) = 6.0 GB

i.e. 32.2 GB -> 16.1 GB per region.  Nothing in the 5-step chain itself is touched: the
steps still run ``triton_kernels.batched_tsyrk_ex`` and ``torch.baddbmm`` in exactly the
order and with exactly the coefficients ``batched_newton_schulz_step_tsyrk`` uses, and the
coefficient schedule is imported from the library rather than restated here, so
``--num-ns-steps 5`` and ``--coefficient-type polar_express`` cannot drift.

Math preservation
-----------------
Same coefficients, same step count, same SYRK path, same per-matrix algorithm.  The Frobenius
norm is still taken by the library's own ``linalg.vector_norm`` on the same view, so the
prologue is bit-exact.  The one place numerics can differ at all is that the ``bfloat16``
X handed to the first step is *contiguous* here, where for the ``5120x2048`` shape the
library leaves it in the transposed layout ``x.mT`` produces -- which can select a
different cuBLAS kernel for the ``baddbmm``.  That is a bf16-tiling-level difference of the
same class the already-accepted batched arm shows against the per-matrix loop;
``scripts/check_fused_ns.py`` gates it at the spec's atol = rtol = 1e-3 and reports the
orthogonality deviation of both arms beside it.

Sample inputs (the only shapes this is used on, both EGTP expert weights at EGTP=2,
EP=64, 24 MoE layers, owner-computes so 96 of the 192 per rank, ``--batch-chunk 64``):

  x: (64, 5120, 2048) float32 cuda   -> out (64, 5120, 2048) float32   (transpose=True)
  x: (32, 5120, 2048) float32 cuda   -> the short tail chunk
  x: (64, 2048, 5120) float32 cuda   -> out (64, 2048, 5120) float32   (transpose=False)
  x: (32, 2048, 5120) float32 cuda
  steps=5, coefficient_type="polar_express", use_syrk=True
"""

import torch
import triton
import triton.language as tl

from emerging_optimizers import triton_kernels
from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
    _COEFFICIENT_SETS,
    get_coefficient_iterator,
    newton_schulz,
)

__all__ = [
    "fused_newton_schulz_batched",
    "fused_newton_schulz_batched_in_group",
    "in_group_shard_factor",
    "FUSED_NS_AVAILABLE",
]

FUSED_NS_AVAILABLE = bool(getattr(triton_kernels, "HAS_TRITON_340", False)) and hasattr(
    triton_kernels, "batched_tsyrk_ex"
)

# ``newton_schulz`` picks "repeat_last" for these and "cycle" otherwise. Kept as a lookup of
# the library's own constant list so the schedule cannot drift from the library's.
_REPEAT_LAST_TYPES = ("polar_express", "cans", "deepseekv4")


# --------------------------------------------------------------------------------------
# Prologue: Frobenius sum-of-squares (pass 1) then scale + cast to bf16 (pass 2).
#
# Pass 1 reads the source flat, because the sum of squares is layout-independent -- which
# also makes it perfectly coalesced for both shapes.  Pass 2 is the only pass that has to
# know about the whiten-the-smaller-dimension transpose, and it does the transpose on the
# read side of a tile so the bf16 store stays contiguous.
# --------------------------------------------------------------------------------------


@triton.jit
def _scale_cast_kernel(
    x_ptr, out_ptr, denom_ptr,
    M, N, s_mat, s_row, s_col,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """out[b, m, n] (contiguous bf16) = x[b, ...] / denom[b], with x read via strides.

    ``s_row`` / ``s_col`` are the strides of the *logical* (M, N) view, so a transposed
    read is expressed by swapping them at the call site rather than by a separate kernel.
    """
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    src = (
        x_ptr
        + b.to(tl.int64) * s_mat
        + offs_m[:, None].to(tl.int64) * s_row
        + offs_n[None, :].to(tl.int64) * s_col
    )
    v = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    denom = tl.load(denom_ptr + b)
    # ``div_rn``, not ``/``: Triton's default fp32 divide is a reciprocal-multiply that is
    # 1 ULP off on a small fraction of inputs, and the diagnostic (scripts/diag_fused_ns.py,
    # job 2882678) traced the whole divergence from the library to exactly that -- 260 of
    # 83.9M prologue entries rounding to a different bf16, which the 5-step iteration then
    # amplifies to ~1e-3 by step 3. Round-to-nearest division makes the prologue bit-exact
    # against ``F.normalize(...).to(bfloat16)``.
    v = tl.math.div_rn(v, denom)
    dst = out_ptr + b.to(tl.int64) * (M * N) + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    tl.store(dst, v.to(tl.bfloat16), mask=mask)


@triton.jit
def _cast_store_kernel(
    x_ptr, out_ptr,
    M, N, s_mat, s_row, s_col,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """out[b, ...] (fp32, written via strides) = x[b, m, n] (contiguous bf16), upcast.

    Writing through ``s_row`` / ``s_col`` is what lets the caller drop both the fp32
    temporary ``X.to(torch.float32)`` and the ``full_t[a:a+chunk] = ...`` copy: the result
    lands directly in the destination slice, already un-transposed.
    """
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    src = x_ptr + b.to(tl.int64) * (M * N) + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    v = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    dst = (
        out_ptr
        + b.to(tl.int64) * s_mat
        + offs_m[:, None].to(tl.int64) * s_row
        + offs_n[None, :].to(tl.int64) * s_col
    )
    tl.store(dst, v, mask=mask)


def _tile(transpose: bool):
    """Tile shape for the two elementwise passes.

    Untransposed, the strided side of the copy is the fully contiguous inner dimension, so
    a wide-and-short tile keeps both the load and the store on long contiguous runs.
    Transposed, one of the two sides is strided whatever the tile, so a square tile is used
    -- the standard tiled-transpose shape, which coalesces the strided side across the
    tile's other axis instead of losing it per element.
    """
    return (64, 64, 4) if transpose else (32, 256, 8)


def _normalize_cast_bf16(
    x: torch.Tensor, transpose: bool, eps: float, denom: torch.Tensor | None = None
) -> torch.Tensor:
    """``F.normalize(x, p=2, dim=(-2,-1)).to(bfloat16)``, optionally transposed, in 2 passes.

    ``denom`` overrides the locally-computed Frobenius norm. That is what lets a rank
    holding one SHARD of a matrix normalize by the norm of the WHOLE matrix (which it can
    only know through a collective) while paying exactly the same two passes.
    """
    batch = x.size(0)
    rows, cols = x.size(1), x.size(2)
    M, N = (cols, rows) if transpose else (rows, cols)

    # The Frobenius norm is taken by the LIBRARY's own reduction, on the same view
    # ``newton_schulz`` reduces (``x.mT`` when the smaller dimension is whitened), not by a
    # Triton kernel of our own. Reason: a different reduction order changes the fp32 norm
    # in its last bits, which then flips the last bit of roughly half the bf16 entries of
    # the very first step and shows up as ~1 ULP of drift all the way through the chain.
    # Reusing ``linalg.vector_norm`` costs nothing -- it is the same single read pass over
    # the fp32 stack that a hand-written reduction would be -- and makes the prologue
    # BIT-EXACT against ``F.normalize(...).to(bfloat16)``: same norm, same fp32 divide,
    # same round-to-bf16, in that order.
    xv = x.mT if transpose else x
    if denom is None:
        denom = torch.linalg.vector_norm(xv, dim=(-2, -1)).clamp_min(eps)

    s_mat, s_r, s_c = x.stride()
    s_row, s_col = (s_c, s_r) if transpose else (s_r, s_c)

    out = torch.empty((batch, M, N), device=x.device, dtype=torch.bfloat16)
    BLOCK_M, BLOCK_N, warps = _tile(transpose)
    grid = (batch, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _scale_cast_kernel[grid](
        x, out, denom, M, N, s_mat, s_row, s_col,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=warps,
    )
    return out


def _cast_into(x_bf16: torch.Tensor, out: torch.Tensor, transpose: bool) -> None:
    """Upcast contiguous bf16 ``(B, M, N)`` straight into ``out``, un-transposing if needed."""
    batch, M, N = x_bf16.shape
    s_mat, s_r, s_c = out.stride()
    s_row, s_col = (s_c, s_r) if transpose else (s_r, s_c)
    BLOCK_M, BLOCK_N, warps = _tile(transpose)
    grid = (batch, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _cast_store_kernel[grid](
        x_bf16, out, M, N, s_mat, s_row, s_col,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=warps,
    )


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def _supported(x: torch.Tensor, coefficient_type: str, use_syrk: bool) -> bool:
    if not FUSED_NS_AVAILABLE or not use_syrk:
        return False
    if x.ndim != 3 or x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous():
        return False
    if torch.get_float32_matmul_precision() != "medium":
        return False
    if coefficient_type not in _COEFFICIENT_SETS:
        return False
    # SYRK needs both GEMM dims 16-byte aligned for TMA, same guard newton_schulz applies.
    if x.size(-1) % 8 != 0 or x.size(-2) % 8 != 0:
        return False
    return True


def fused_newton_schulz_batched(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str,
    use_syrk: bool = True,
    eps: float = 1e-7,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched Newton-Schulz with the fp32 prologue/epilogue passes fused.

    Falls back to the library ``newton_schulz`` (writing into ``out`` if given) whenever
    the fused path's preconditions do not hold, so a caller can enable it unconditionally.

    Arguments:
        x: ``(B, rows, cols)`` float32, contiguous, cuda.
        steps: Newton-Schulz steps -- 5 for this workload, locked by spec.toml.
        coefficient_type: ``polar_express`` for this workload, locked by spec.toml.
        use_syrk: must be True; the SYRK path is forced on for every run by spec.toml.
        eps: Frobenius-norm clamp, matching ``newton_schulz``'s default.
        out: optional ``(B, rows, cols)`` float32 destination; when given the result is
            written straight into it and no fp32 temporary is materialised.

    Returns:
        ``out`` if given, else a freshly allocated ``(B, rows, cols)`` float32 tensor.
    """
    if not _supported(x, coefficient_type, use_syrk):
        result = newton_schulz(x, steps, coefficient_type, use_syrk=use_syrk, eps=eps)
        if out is None:
            return result
        out.copy_(result)
        return out

    transpose = x.size(-2) > x.size(-1)
    X = _normalize_cast_bf16(x, transpose, eps)

    iter_mode = "repeat_last" if coefficient_type in _REPEAT_LAST_TYPES else "cycle"
    for a, b, c in get_coefficient_iterator(
        steps, _COEFFICIENT_SETS[coefficient_type], mode=iter_mode
    ):
        # Identical to muon_utils.batched_newton_schulz_step_tsyrk, tp_group=None.
        A = triton_kernels.batched_tsyrk_ex(X)
        if c != 0.0:
            Bm = triton_kernels.batched_tsyrk_ex(A, A, alpha=c, beta=b)
            X = torch.baddbmm(X, Bm, X, alpha=1.0, beta=a)
        else:
            X = torch.baddbmm(X, A, X, alpha=b, beta=a)

    if out is None:
        out = torch.empty_like(x)
    _cast_into(X, out, transpose)
    return out


# --------------------------------------------------------------------------------------
# Distribute-in-group: the SAME 5-step chain, with the matrix kept SHARDED across a small
# process group instead of replicated on every rank of it.
#
# stage-5 / plan.md rank 2, ``gtp-distribute-in-group-g2``.
#
# ``fused_newton_schulz_batched`` above is the *duplicated* corner of the distribution
# knob: every rank of a duplication subgroup holds the whole all-gathered matrix and runs
# the whole chain on it, so the subgroup does g copies of one matrix's arithmetic. This is
# the *distributed* corner at subgroup granularity: the g ranks each keep their own slab
# and cooperate, so the subgroup does one copy.
#
# Which axis the slab lies on decides what a step costs, and only ONE of the two is worth
# distributing. ``newton_schulz`` whitens the SMALLER dimension: it works on
# ``X = x.mT if rows > cols else x``, so ``X`` is ``(M, N)`` with ``M = min`` and
# ``N = max``, and one step is
#
#     A  = X @ X.T                (SYRK, M x M)
#     Bm = c * A @ A + b * A      (SYRK, M x M)
#     X  = a * X + Bm @ X         (GEMM, M x N)
#
# * The full matrix is TALL (``rows > cols``), so ``X = full.mT`` and the rank shard --
#   a row slab of ``full`` -- is a COLUMN slab of ``X``: a shard of ``N``. Then
#   ``A = sum_r X_r @ X_r.T`` is one all-reduce of the small ``M x M`` Gram, ``A @ A`` is
#   replicated, and ``Bm @ X_r`` shards. Per-rank arithmetic goes from ``3*M^2*N + M^3``
#   to ``3*M^2*N/g + M^3`` -- and the only wire is ``g`` ranks' worth of ``M x M`` per
#   step, which is the smallest object in the iteration. This is the case this function
#   implements, and it is exactly ``newton_schulz_tp(tp_mode="distributed")`` restricted to
#   a subgroup and batched.
# * The full matrix is WIDE, so ``X = full`` and the rank shard is a ROW slab of ``X``: a
#   shard of ``M``. Then ``A_r = X_r @ X.T`` needs the whole ``X`` gathered EVERY step
#   (``M x N``, the largest object), and neither ``A_r`` nor ``A_r @ A`` can use the
#   symmetric SYRK kernel, so two of the three products cost the same as the replicated
#   ones. That corner buys ~25 % of the arithmetic for ~5x the wire and is NOT implemented:
#   the caller keeps those shapes replicated in the subgroup.
#
# Math preservation: same coefficients, same step count, same SYRK path on the terms that
# still have one, and the same whiten-the-smaller-dimension orientation. Two things deviate
# in the last bits and nothing else does:
#   (1) the Frobenius norm is a collective sum of per-shard sums of squares rather than one
#       ``linalg.vector_norm`` over the whole matrix (taken in fp64 here to keep the split
#       reduction at least as accurate as the unsplit one), and
#   (2) the Gram is summed over ``g`` partial products instead of formed in one.
# Both are the deviations ``tp_mode="distributed"`` already has against ``"duplicated"``;
# they are what the equivalence gate measures.
#
# Row ORDER inside the shard is deliberately not constrained. Permuting the rows of the
# full matrix by P sends A -> P A P.T, Bm -> P Bm P.T and X -> P X, i.e. the whole
# iteration is equivariant, so the caller may hand this function any slab of rows in any
# order and get the matching slab of the answer back. That is what lets the exchange
# deliver a rank the STRIDED set of source blocks it must return, with no permutation
# anywhere.
# --------------------------------------------------------------------------------------


def in_group_shard_factor(rows_total: int, cols: int, subgroup_size: int) -> float:
    """Per-rank arithmetic of the distributed form, as a fraction of the replicated form.

    ``(3*M^2*N/g + M^3) / (3*M^2*N + M^3)`` with ``M = min``, ``N = max`` -- the step cost
    above, counting the replicated ``A @ A`` term at full price. Used by the caller's deal
    so greedy LPT balances what a rank ACTUALLY does, not what a replicated rank would.
    Returns 1.0 for a shape this function does not distribute (a wide matrix).
    """
    if subgroup_size <= 1 or rows_total <= cols:
        return 1.0
    M, N = min(rows_total, cols), max(rows_total, cols)
    full = 3.0 * M * M * N + float(M) ** 3
    return (3.0 * M * M * N / subgroup_size + float(M) ** 3) / full


def fused_newton_schulz_batched_in_group(
    x_local: torch.Tensor,
    rows_total: int,
    group,
    steps: int,
    coefficient_type: str,
    use_syrk: bool = True,
    eps: float = 1e-7,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched Newton-Schulz on matrices ROW-SHARDED across ``group``.

    Arguments:
        x_local: ``(B, rows_local, cols)`` float32, contiguous, cuda -- this rank's slab of
            ``B`` matrices whose full shape is ``(rows_total, cols)``. The slab's rows may
            be any subset of the full row set, in any order (see the equivariance note
            above), as long as the ``g`` ranks of ``group`` partition the rows between
            them and every rank holds the same number.
        rows_total: rows of the FULL matrix; must equal ``rows_local * group.size()``.
        group: the process group the matrix is sharded across.
        steps / coefficient_type / use_syrk / eps: as ``fused_newton_schulz_batched``.
        out: optional ``(B, rows_local, cols)`` float32 destination.

    Returns:
        ``out`` if given, else a fresh tensor -- this rank's slab of the orthogonalized
        full matrix, row-for-row in the order it handed its slab in.
    """
    g = group.size()
    batch, rows_local, cols = x_local.shape
    if rows_total != rows_local * g:
        raise ValueError(
            f"rows_total {rows_total} != rows_local {rows_local} * group size {g}"
        )
    if rows_total <= cols:
        # The wide corner is the expensive one (see the header); refusing it is what keeps
        # a caller from silently buying 5x the wire for a quarter of the arithmetic.
        raise ValueError(
            f"distribute-in-group needs a TALL matrix, got {rows_total}x{cols}; the caller "
            "must keep wide shapes replicated in the subgroup"
        )
    if not _supported(x_local, coefficient_type, use_syrk):
        raise ValueError(
            "distribute-in-group requires the fused preconditions (fp32, contiguous, cuda, "
            "use_syrk, matmul precision 'medium'); a fallback here would silently run a "
            "different distribution"
        )
    # X = full.mT is (M, N) = (cols, rows_total): the shard is on N, whose local extent
    # must stay 16-byte aligned for the SYRK/TMA path, same guard ``_supported`` applies.
    if rows_local % 8 != 0:
        raise ValueError(f"local rows {rows_local} must be a multiple of 8 for SYRK")

    # ---- global Frobenius norm: one collective over per-shard sums of squares -----------
    # fp64 for the local reduction and the sum, so splitting the reduction cannot be LESS
    # accurate than the single ``linalg.vector_norm`` it replaces.
    ssq = torch.linalg.vector_norm(x_local, dim=(-2, -1), dtype=torch.float64) ** 2
    torch.distributed.all_reduce(ssq, group=group)
    denom = ssq.sqrt().clamp_min(eps).to(torch.float32)

    # (B, M, N_local) contiguous bf16, transposed on the read side exactly as the
    # replicated prologue does, but divided by the WHOLE matrix's norm.
    X = _normalize_cast_bf16(x_local, True, eps, denom=denom)

    iter_mode = "repeat_last" if coefficient_type in _REPEAT_LAST_TYPES else "cycle"
    for a, b, c in get_coefficient_iterator(
        steps, _COEFFICIENT_SETS[coefficient_type], mode=iter_mode
    ):
        # A is the FULL M x M Gram: each rank forms its columns' partial product with the
        # same symmetric kernel the replicated path uses, and the group sums them.
        A = triton_kernels.batched_tsyrk_ex(X).contiguous()
        torch.distributed.all_reduce(A, group=group)
        # From here every rank holds the same A, so Bm is formed identically on all of
        # them -- the one replicated term -- and only the M x N product shards.
        if c != 0.0:
            Bm = triton_kernels.batched_tsyrk_ex(A, A, alpha=c, beta=b)
            X = torch.baddbmm(X, Bm, X, alpha=1.0, beta=a)
        else:
            X = torch.baddbmm(X, A, X, alpha=b, beta=a)

    if out is None:
        out = torch.empty_like(x_local)
    _cast_into(X.contiguous(), out, True)
    return out


# --------------------------------------------------------------------------------------
# torch.library registration -- lets the op be called, traced and overridden by name.
# --------------------------------------------------------------------------------------

_LIB = torch.library.Library("dist_muon_opt", "FRAGMENT")
_LIB.define(
    "fused_newton_schulz_batched(Tensor x, int steps, str coefficient_type, "
    "bool use_syrk=True, float eps=1e-7) -> Tensor"
)


def _op_impl(x, steps, coefficient_type, use_syrk=True, eps=1e-7):
    return fused_newton_schulz_batched(x, steps, coefficient_type, use_syrk, eps)


_LIB.impl("fused_newton_schulz_batched", _op_impl, "CUDA")


@torch.library.register_fake("dist_muon_opt::fused_newton_schulz_batched")
def _op_meta(x, steps, coefficient_type, use_syrk=True, eps=1e-7):
    return torch.empty_like(x)
