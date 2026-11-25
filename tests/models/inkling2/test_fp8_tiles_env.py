# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the INKLING2_FP8_* tile-size env overrides.

Loads fp8_w8a16 with a stubbed vllm.triton_utils so it runs anywhere torch
runs, including a CPU-only sandbox. Tile parsing happens at import time,
so each case reloads the module.
"""

import importlib.util
import sys
import types

import pytest
import torch  # noqa: F401 - the module under test needs it

MODULE_PATH = ("vllm/models/inkling2/nvidia/ops/fp8_w8a16.py")


def _load(env: dict[str, str], monkeypatch):
    for key in ("INKLING2_FP8_GEMV_BN", "INKLING2_FP8_GEMV_BK",
                "INKLING2_FP8_GEMM_BM", "INKLING2_FP8_GEMM_BN",
                "INKLING2_FP8_GEMM_BK"):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)

    tu = types.ModuleType("vllm.triton_utils")

    class _Triton:
        @staticmethod
        def jit(fn):
            return fn

        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b

    tu.triton = _Triton()
    tu.tl = types.ModuleType("tl")
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", tu)

    spec = importlib.util.spec_from_file_location("fp8_w8a16_testmod", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTileEnv:
    def test_defaults(self, monkeypatch):
        mod = _load({}, monkeypatch)
        assert (mod._GEMV_BN, mod._GEMV_BK) == (32, 256)
        assert (mod._GEMM_BM, mod._GEMM_BN, mod._GEMM_BK) == (64, 64, 64)

    def test_valid_override(self, monkeypatch):
        mod = _load({"INKLING2_FP8_GEMV_BN": "128",
                     "INKLING2_FP8_GEMM_BK": "32"}, monkeypatch)
        assert mod._GEMV_BN == 128
        assert mod._GEMM_BK == 32
        assert mod._GEMV_BK == 256  # untouched default

    def test_rejects_non_power_of_two(self, monkeypatch):
        with pytest.raises(ValueError, match="power of two"):
            _load({"INKLING2_FP8_GEMV_BK": "96"}, monkeypatch)

    def test_rejects_below_dot_minimum(self, monkeypatch):
        with pytest.raises(ValueError, match=">= 16"):
            _load({"INKLING2_FP8_GEMM_BM": "8"}, monkeypatch)

    def test_rejects_garbage(self, monkeypatch):
        with pytest.raises(ValueError, match="integer"):
            _load({"INKLING2_FP8_GEMM_BN": "wide"}, monkeypatch)
