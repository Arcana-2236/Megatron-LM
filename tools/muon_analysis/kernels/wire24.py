# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""24-bit split transport codec for the EGTP Newton-Schulz INPUT leg.

stage-5 / plan.md rank 4, ``egtp-wire-24bit-input-leg``.

What this is for
----------------
The EGTP region (``ns:egtp:duplicated_batch_sub_g1_fused_pipe_selfelide_wbf16out``) is
**hard comm-bound**: 29.769 ms of wire against 14.701 ms of compute per region, 91.2 % of
its state-aware SOL.  Efficiency there is exhausted, so the only lever left is BYTES.  The
output leg already travels as bf16 (stage-4, bitwise).  The **input** leg -- the shard of
the un-orthogonalized matrix each rank ships to the owner -- is still fp32, 2.013 of the
3.02 GB/rank/region.

Plain bf16 on that leg was **refuted on numerics**: it perturbs every entry before the
5-step polar_express chain and measured ``max_abs_diff`` 1.221e-3 against the workload's
1e-3 equivalence gate (stage-4 phase-5).  This codec keeps **16 of fp32's 24 mantissa
bits** instead of bf16's 8 -- 3 bytes on the wire instead of 4 (a 25 % cut) and ~2^-16
relative error after round-to-nearest-even instead of bf16's ~2^-9, i.e. ~256x more
headroom against the same gate.

Wire format
-----------
Per fp32 element, the 32-bit word ``u`` is rounded to nearest-even at bit 8 and split::

    hi : bits 31..16 of the rounded word -- shipped in a ``bfloat16`` buffer (bit-reused
         as a 16-bit container; NCCL has no 16-bit integer datatype, and bf16 is exactly
         the top half of an fp32 word, so the container is the natural one)
    lo : bits 15..8  of the rounded word -- shipped in a ``uint8`` buffer

and reassembled as ``(hi << 16) | (lo << 8)``, i.e. the original word with its low 8
mantissa bits cleared.  The value is therefore fp32 with a 16-bit mantissa:

  * exact for anything already representable in 16 mantissa bits (in particular this is
    the identity on any bf16-valued input),
  * otherwise a relative error <= 2^-16 -- the half-ulp of a 16-bit mantissa is 2^-17 of
    the significand, which is at worst 2^-16 of the value because the significand of a
    normal fp32 lies in [1, 2),
  * exact and total on zeros, subnormals, infinities and NaNs -- the rounding step is
    skipped when the exponent field is all ones so a NaN can never be rounded into an
    infinity, and the split/reassemble is a pure bit operation everywhere else.

This is **lossy**, deliberately and by a bounded amount; it is gated by the workload's
equivalence gate (atol = rtol = 1e-3) measured through the real Newton-Schulz chain, not
by a per-entry ULP argument -- see ``scripts/preflight_wire24.py``.

Sample inputs (the EGTP input leg at EGTP=2, EP=64, 24 MoE layers, 192 matrices per shape
per rank, ``--pipe-chunks 4`` so a window is 24 matrices of one peer block)::

  src: (24, 2560, 5120) float32 cuda contiguous -> hi (24, 2560, 5120) bfloat16
                                                   lo (24, 2560, 5120) uint8
  src: (24, 1024, 2048) float32 cuda contiguous  (the 5120x2048 shape's rank shard)
  unpack destination: a (n, rows, cols) float32 view with an arbitrary OUTER stride
  (``gx.view(n, world, rows, cols)[:, s]``) and contiguous ``rows x cols`` blocks.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only on a Triton-less install
    HAVE_TRITON = False

__all__ = ["pack24", "unpack24", "WIRE24_AVAILABLE"]


if HAVE_TRITON:

    @triton.jit
    def _pack24_kernel(
        src_ptr, hi_ptr, lo_ptr, bs, s_src, s_hi, s_lo, BLOCK: tl.constexpr,
    ):
        """hi/lo[b, i] = top 16 / next 8 bits of RNE(src[b, i]) at bit 8.

        Grid is ``(n_blocks, cdiv(bs, BLOCK))``: one program row per contiguous block, so a
        strided source (a peer slice of a matrix-major buffer) costs one index multiply
        rather than a per-element divide.
        """
        b = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < bs
        x = tl.load(src_ptr + b * s_src + offs, mask=mask, other=0.0)
        # Bit-level, in int32: two's-complement addition on the raw word increments the
        # MAGNITUDE for both signs, so one rounding expression covers positives and
        # negatives. Carries out of the mantissa propagate into the exponent exactly as
        # IEEE rounding requires.
        u = x.to(tl.int32, bitcast=True)
        # Round to nearest, ties to even, at the bit-8 boundary.
        rne = u + (0x7F + ((u >> 8) & 1))
        # ... except on inf/NaN, where the carry would turn a NaN into an infinity.
        u = tl.where((u & 0x7F800000) == 0x7F800000, u, rne)
        # ``>> 16`` is arithmetic here, but the narrowing cast keeps only the low 16 bits,
        # which are exactly bits 31..16 of the word. ``hi_ptr`` is the bf16 wire buffer
        # RE-VIEWED as int16 by the wrapper, so the container's interpretation never enters
        # the kernel and no bitcast is needed inside it.
        tl.store(hi_ptr + b * s_hi + offs, (u >> 16).to(tl.int16), mask=mask)
        tl.store(
            lo_ptr + b * s_lo + offs,
            ((u >> 8) & 0xFF).to(tl.uint8),
            mask=mask,
        )

    @triton.jit
    def _unpack24_kernel(
        hi_ptr, lo_ptr, dst_ptr, bs, s_hi, s_lo, s_dst, BLOCK: tl.constexpr,
    ):
        """dst[b, i] = bitcast((hi << 16) | (lo << 8)) -- the exact inverse of the split."""
        b = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < bs
        # ``hi_ptr`` is the bf16 wire buffer re-viewed as int16 by the wrapper.
        hi = tl.load(hi_ptr + b * s_hi + offs, mask=mask)
        lo = tl.load(lo_ptr + b * s_lo + offs, mask=mask)
        # int16 sign-extends into int32; the sign bits are shifted straight back out by the
        # ``<< 16``, so the reassembled word is bit-exact for negative values too.
        w = (hi.to(tl.int32) << 16) | (lo.to(tl.int32) << 8)
        tl.store(dst_ptr + b * s_dst + offs, w.to(tl.float32, bitcast=True), mask=mask)


WIRE24_AVAILABLE = HAVE_TRITON

_BLOCK = 1024


def _blocked(t: torch.Tensor):
    """(n_blocks, block_size, block_stride) for a tensor whose trailing dims are dense.

    The two call sites are a contiguous ``(n, rows, cols)`` slice and a peer slice
    ``gx.view(n, world, rows, cols)[:, s]``; both have dense ``rows x cols`` blocks and an
    arbitrary outer stride, which is exactly what the kernels' 2-D grid consumes.
    """
    n = t.size(0)
    bs = t.numel() // n if n else 0
    stride = t.stride(0)
    assert t.ndim >= 1
    assert t[0].is_contiguous(), "wire24 needs dense blocks after the outer dim"
    return n, bs, stride


def _pack24_torch(src: torch.Tensor, hi: torch.Tensor, lo: torch.Tensor) -> None:
    u = src.contiguous().view(torch.int32)
    special = (u & 0x7F800000) == 0x7F800000
    r = torch.where(special, u, u + (0x7F + ((u >> 8) & 1)))
    hi.copy_((r >> 16).to(torch.int16).view(torch.bfloat16))
    lo.copy_(((r >> 8) & 0xFF).to(torch.uint8))


def _unpack24_torch(hi: torch.Tensor, lo: torch.Tensor, dst: torch.Tensor) -> None:
    w = (hi.contiguous().view(torch.int16).to(torch.int32) << 16) | (
        lo.to(torch.int32) << 8
    )
    dst.copy_(w.view(torch.float32))


def pack24(src: torch.Tensor, hi: torch.Tensor, lo: torch.Tensor) -> None:
    """Split fp32 ``src`` into the 3-byte wire pair ``(hi bf16, lo uint8)``, in place."""
    assert src.dtype == torch.float32, src.dtype
    assert hi.dtype == torch.bfloat16 and lo.dtype == torch.uint8
    assert hi.shape == src.shape == lo.shape, (src.shape, hi.shape, lo.shape)
    if src.numel() == 0:
        return
    if not (WIRE24_AVAILABLE and src.is_cuda):
        _pack24_torch(src, hi, lo)
        return
    n, bs, s_src = _blocked(src)
    _, _, s_hi = _blocked(hi)
    _, _, s_lo = _blocked(lo)
    _pack24_kernel[(n, triton.cdiv(bs, _BLOCK))](
        src, hi.view(torch.int16), lo, bs, s_src, s_hi, s_lo, BLOCK=_BLOCK, num_warps=4,
    )


def unpack24(hi: torch.Tensor, lo: torch.Tensor, dst: torch.Tensor) -> None:
    """Reassemble the wire pair into fp32 ``dst`` (which may be outer-strided), in place."""
    assert dst.dtype == torch.float32, dst.dtype
    assert hi.dtype == torch.bfloat16 and lo.dtype == torch.uint8
    assert hi.shape == dst.shape == lo.shape, (hi.shape, lo.shape, dst.shape)
    if dst.numel() == 0:
        return
    if not (WIRE24_AVAILABLE and dst.is_cuda):
        _unpack24_torch(hi, lo, dst)
        return
    n, bs, s_hi = _blocked(hi)
    _, _, s_lo = _blocked(lo)
    _, _, s_dst = _blocked(dst)
    _unpack24_kernel[(n, triton.cdiv(bs, _BLOCK))](
        hi.view(torch.int16), lo, dst, bs, s_hi, s_lo, s_dst, BLOCK=_BLOCK, num_warps=4,
    )


def roundtrip24(x: torch.Tensor) -> torch.Tensor:
    """``unpack24(pack24(x))`` -- the value a 24-bit-split wire delivers. For gates."""
    hi = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    lo = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    out = torch.empty_like(x)
    pack24(x.contiguous(), hi, lo)
    unpack24(hi, lo, out)
    return out


# --------------------------------------------------------------------------------------
# torch.library registration -- lets the codec be called, traced and overridden by name.
# Both ops MUTATE their destinations (the wire buffers / the gather buffer), so they are
# declared with ``(a!)`` alias annotations and return nothing.
# --------------------------------------------------------------------------------------

_LIB = torch.library.Library("dist_muon_opt", "FRAGMENT")
_LIB.define("wire24_pack(Tensor src, Tensor(a!) hi, Tensor(b!) lo) -> ()")
_LIB.define("wire24_unpack(Tensor hi, Tensor lo, Tensor(a!) dst) -> ()")
_LIB.impl("wire24_pack", pack24, "CUDA")
_LIB.impl("wire24_unpack", unpack24, "CUDA")


@torch.library.register_fake("dist_muon_opt::wire24_pack")
def _pack24_meta(src, hi, lo):
    return None


@torch.library.register_fake("dist_muon_opt::wire24_unpack")
def _unpack24_meta(hi, lo, dst):
    return None
