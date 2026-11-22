# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 W8A16 dense linear for Inkling2 (opt-in; azjhf-dev fork addition).

Weights are quantized to ``float8_e4m3fn`` with one scale per output row
(W8); activations stay in bf16 (A16). The Triton kernels convert each FP8
weight tile to bf16 in registers and accumulate in fp32, so no FP8 tensor
cores are required and the same path runs on SM90 / SM100 / SM120. A pure
torch reference path keeps the module usable off CUDA (numerics tests,
CPU-only environments).

This is our own implementation of the "small W8A16 Triton kernel" described
(but not shared) in the r/LocalLLaMA Inkling-2-Mini-on-4090 thread. It does
NOT reproduce that poster's numbers; treat their ~190-220 tok/s claim as
unverified until measured on hardware with ``benchmarks/inkling2/bench_5090.py``.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

# e4m3fn max normal value.
FP8_MAX = 448.0

# Decode regime: at most this many rows use the GEMV-style kernel, mirroring
# gemv.py's sm12x path (tl.dot needs M >= 16, so rows are zero-padded to MP).
GEMV_MAX_M = 16
_GEMV_BN, _GEMV_BK = 32, 256

# Prefill / verify regime: plain tiled GEMM.
_GEMM_BM, _GEMM_BN, _GEMM_BK = 64, 64, 64


def quant_weight_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row absmax quantization of a 2D weight to float8_e4m3fn.

    Returns (weight_q [N, K] float8_e4m3fn, scale [N] float32) such that
    ``dequant_weight_fp8(weight_q, scale)`` approximates ``weight``.
    """
    assert weight.ndim == 2
    w = weight.detach().to(torch.float32)
    scale = w.abs().amax(dim=1).clamp(min=1e-12) / FP8_MAX
    weight_q = (w / scale[:, None]).to(torch.float8_e4m3fn)
    return weight_q, scale


def dequant_weight_fp8(
    weight_q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference dequantization; FP8 values are exactly representable in bf16."""
    return (weight_q.to(torch.bfloat16).to(torch.float32) * scale[:, None]).to(dtype)


@triton.jit
def _fp8_w8a16_gemv_kernel(
    X,
    W,
    S,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0.0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(x, w.to(x.dtype))
    s = tl.load(S + cols, cols < N, other=0.0)
    acc = acc * s[None, :]
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _fp8_w8a16_gemm_kernel(
    X,
    W,
    S,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[:, None] < K),
            other=0.0,
        )
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(x, w.to(x.dtype), acc)
    s = tl.load(S + cols, cols < N, other=0.0)
    acc = acc * s[None, :]
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def fp8_w8a16_linear(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``x @ dequant(weight_q, scale).T`` with fp32 accumulation.

    x: [..., K] bf16 (or fp16); weight_q: [N, K] float8_e4m3fn; scale: [N]
    fp32. Uses the Triton kernels on CUDA and an exact same-order torch
    reference elsewhere. Returns ``out_dtype`` (default: x.dtype).
    """
    assert weight_q.ndim == 2 and weight_q.dtype == torch.float8_e4m3fn
    n, k = weight_q.shape
    assert x.shape[-1] == k and scale.shape == (n,)
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    out_dtype = out_dtype or x.dtype

    if x2.is_cuda:
        y = torch.empty((m, n), device=x2.device, dtype=out_dtype)
        if m <= GEMV_MAX_M:
            grid = (triton.cdiv(n, _GEMV_BN),)
            _fp8_w8a16_gemv_kernel[grid](
                x2, weight_q, scale, bias, y, m, n, k,
                x2.stride(0), weight_q.stride(0), y.stride(0),
                HAS_BIAS=bias is not None,
                BN=_GEMV_BN, BK=_GEMV_BK, MP=GEMV_MAX_M,
            )
        else:
            grid = (triton.cdiv(m, _GEMM_BM), triton.cdiv(n, _GEMM_BN))
            _fp8_w8a16_gemm_kernel[grid](
                x2, weight_q, scale, bias, y, m, n, k,
                x2.stride(0), weight_q.stride(0), y.stride(0),
                HAS_BIAS=bias is not None,
                BM=_GEMM_BM, BN=_GEMM_BN, BK=_GEMM_BK,
            )
        return y.view(*lead, n)

    # Reference path: same operation order as the kernels (bf16 product terms,
    # fp32 accumulation, per-row scale applied afterwards).
    y = torch.matmul(x2.to(torch.float32), weight_q.to(torch.float32).t())
    y = y * scale.to(torch.float32)
    if bias is not None:
        y = y + bias.to(torch.float32)
    return y.to(out_dtype).view(*lead, n)


class Fp8W8A16Linear:
    """Quantized replacement for a dense linear's weight.

    Deliberately NOT an nn.Module: it is attached as a plain attribute (e.g.
    ``lm_head.fp8_w8a16``) after weight loading, so it must not appear in
    ``named_parameters`` / module traversals that vLLM and the DSpark drafter
    run over the model.
    """

    def __init__(
        self,
        weight_q: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        self.weight_q = weight_q
        self.scale = scale
        self.bias = bias

    @classmethod
    def from_weight(
        cls, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> "Fp8W8A16Linear":
        weight_q, scale = quant_weight_fp8(weight)
        return cls(weight_q, scale, bias)

    @property
    def device(self) -> torch.device:
        return self.weight_q.device

    def rows(self, start: int, end: int) -> "Fp8W8A16Linear":
        """View over a contiguous run of output rows (views, no copies).

        Used for fused projections such as Inkling2's qkvr, where the caller
        matmuls against row slices of the weight."""
        return Fp8W8A16Linear(self.weight_q[start:end], self.scale[start:end])

    def __call__(
        self, x: torch.Tensor, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        return fp8_w8a16_linear(x, self.weight_q, self.scale, self.bias, out_dtype)
