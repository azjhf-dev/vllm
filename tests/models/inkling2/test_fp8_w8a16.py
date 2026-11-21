# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the FP8 W8A16 dense path (azjhf-dev fork).

The quantization/reference-linear tests run on CPU; the Triton kernel tests
require a CUDA device and are skipped otherwise. No model weights needed.
"""

import os

import pytest
import torch

from vllm.models.inkling2.nvidia.ops.fp8_w8a16 import (
    FP8_MAX,
    Fp8W8A16Linear,
    dequant_weight_fp8,
    fp8_w8a16_linear,
    quant_weight_fp8,
)

DTYPE = torch.bfloat16


def _randn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(DTYPE)


class TestQuantWeightFp8:
    def test_scales_cover_absmax(self):
        w = _randn(64, 128)
        q, s = quant_weight_fp8(w)
        assert q.dtype == torch.float8_e4m3fn
        assert s.dtype == torch.float32 and s.shape == (64,)
        assert (q.to(torch.float32).abs() <= FP8_MAX).all()
        # Zero rows must not produce zero/NaN scales.
        w2 = w.clone()
        w2[7] = 0
        _, s2 = quant_weight_fp8(w2)
        assert s2[7] > 0 and torch.isfinite(s2).all()

    def test_dequant_relative_error(self):
        w = _randn(256, 512)
        q, s = quant_weight_fp8(w)
        w_hat = dequant_weight_fp8(q, s, dtype=torch.float32)
        rel = (w_hat - w.float()).norm() / w.float().norm()
        # e4m3 has 3 mantissa bits; per-row scaling keeps the global relative
        # error around 2^-4..2^-5 for Gaussian weights.
        assert rel < 0.08


class TestFp8W8A16LinearReference:
    """CPU reference path (mirrors the Triton kernels' operation order)."""

    @pytest.mark.parametrize("m", [1, 4, 16, 33, 128])
    def test_matches_bf16_matmul(self, m):
        n, k = 96, 256
        x = _randn(m, k, seed=m)
        w = _randn(n, k, seed=m + 1)
        q, s = quant_weight_fp8(w)
        y = fp8_w8a16_linear(x, q, s)
        assert y.dtype == DTYPE
        y_ref = torch.matmul(x.float(), w.float().t()).to(DTYPE)
        # Compare against the bf16 baseline: the FP8 result should be much
        # closer to the exact matmul than half the weight's own norm, and the
        # relative error should sit in the e4m3 quantization range.
        rel = (y.float() - y_ref.float()).norm() / y_ref.float().norm()
        assert rel < 0.10

    def test_bias_and_out_dtype(self):
        m, n, k = 3, 64, 128
        x = _randn(m, k)
        w = _randn(n, k)
        b = _randn(n)
        q, s = quant_weight_fp8(w)
        y = fp8_w8a16_linear(x, q, s, bias=b, out_dtype=torch.float32)
        assert y.dtype == torch.float32
        y_nb = fp8_w8a16_linear(x, q, s, out_dtype=torch.float32)
        assert torch.allclose(y, y_nb + b.float(), atol=1e-3)

    def test_module_wrapper(self):
        n, k = 32, 64
        w = _randn(n, k)
        head = Fp8W8A16Linear.from_weight(w)
        x = _randn(2, k)
        y = head(x)
        assert y.shape == (2, n)
        # Plain object: must not register as an nn.Module attribute.
        parent = torch.nn.Module()
        parent.head = head  # type: ignore[assignment]
        assert "head" not in parent._modules
        assert not any(p.dtype == torch.float8_e4m3fn for p in parent.parameters())


class TestEnvParsing:
    def test_groups(self, monkeypatch):
        from vllm.models.inkling2.nvidia.fp8_dense import fp8_dense_groups

        monkeypatch.delenv("INKLING2_FP8_DENSE", raising=False)
        assert fp8_dense_groups() == frozenset()
        monkeypatch.setenv("INKLING2_FP8_DENSE", "lm_head")
        assert fp8_dense_groups() == frozenset({"lm_head"})
        monkeypatch.setenv("INKLING2_FP8_DENSE", " lm_head , attn ")
        assert fp8_dense_groups() == frozenset({"lm_head", "attn"})
        monkeypatch.setenv("INKLING2_FP8_DENSE", "bogus")
        with pytest.raises(ValueError, match="unknown group"):
            fp8_dense_groups()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestTritonKernels:
    @pytest.mark.parametrize("m", [1, 8, 16, 17, 64, 200])
    def test_kernel_vs_reference(self, m):
        n, k = 512, 2304  # Inkling 2 Mini hidden size
        x = _randn(m, k, seed=m).cuda()
        w = _randn(n, k, seed=m + 1).cuda()
        q, s = quant_weight_fp8(w)
        y_gpu = fp8_w8a16_linear(x, q, s)
        y_ref = torch.matmul(x.float(), q.to(torch.float32).t()) * s
        rel = (y_gpu.float() - y_ref).norm() / y_ref.norm()
        assert rel < 2e-2  # bf16 product terms vs fp32 reference

    def test_lm_head_shape(self):
        # Inkling 2 Mini padded vocab x hidden, sliced to keep CI fast.
        n, k = 201024 // 8, 2304
        x = _randn(1, k).cuda()
        w = _randn(n, k).cuda()
        q, s = quant_weight_fp8(w)
        y = fp8_w8a16_linear(x, q, s, out_dtype=torch.float32)
        assert y.shape == (1, n) and y.dtype == torch.float32
