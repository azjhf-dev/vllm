# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep FP8 W8A16 tile sizes on the target GPU (azjhf-dev fork).

The Triton kernels in vllm/models/inkling2/nvidia/ops/fp8_w8a16.py use
conservative fixed tile sizes shared across SM90/SM100/SM120. This script
benchmarks candidate tiles for the real Inkling 2 Mini shapes on YOUR GPU
and prints the INKLING2_FP8_* env values to set if a candidate beats the
defaults. Requires CUDA and a few GB of free VRAM. Nothing here is tuned
yet: the defaults are untested on hardware, so running this once on the
5090 is the first real data point.

  python benchmarks/inkling2/tune_fp8_tiles.py
  python benchmarks/inkling2/tune_fp8_tiles.py --shape 4608x2304 --shape 2304x2304

Shapes default to the lm_head (201024x2304). Add qkvr row-slice and wo_ud
shapes with --shape NxK (N = output rows, K = input cols). M rows sweep the
decode regime (GEMV kernel) and a prefill point (GEMM kernel).
"""

from __future__ import annotations

import argparse

# Candidate tiles. tl.dot needs every edge >= 16; keep lists small so the
# sweep finishes in minutes (each config JIT-compiles on first use).
GEMV_BN_CANDIDATES = (16, 32, 64, 128)
GEMV_BK_CANDIDATES = (64, 128, 256, 512)
GEMM_BM_CANDIDATES = (32, 64, 128)
GEMM_BN_CANDIDATES = (32, 64, 128)
GEMM_BK_CANDIDATES = (32, 64, 128)

DEFAULT_SHAPES = [(201024, 2304)]  # lm_head
GEMV_MS = (1, 2, 4, 8, 16)   # decode
GEMM_MS = (64, 256)          # prefill / verify


def parse_shape(text: str) -> tuple[int, int]:
    n, _, k = text.lower().partition("x")
    return int(n), int(k)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", action="append", default=[],
                   help="extra NxK weight shape, repeatable")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--iters", type=int, default=50)
    return p.parse_args()


def bench(fn, iters: int) -> float:
    """Median milliseconds via triton.testing.do_bench (clears L2 per rep)."""
    import triton  # noqa: PLC0415
    return triton.testing.do_bench(fn, warmup=10, rep=iters)


def sweep_gemv(x, weight_q, scale, ms, iters):
    from vllm.models.inkling2.nvidia.ops.fp8_w8a16 import (  # noqa: PLC0415
        _fp8_w8a16_gemv_kernel, GEMV_MAX_M)
    import triton  # noqa: PLC0415
    n, k = weight_q.shape
    results = {}
    for m in ms:
        best = None
        for bn in GEMV_BN_CANDIDATES:
            for bk in GEMV_BK_CANDIDATES:
                grid = (triton.cdiv(n, bn),)
                fn = lambda: _fp8_w8a16_gemv_kernel[grid](  # noqa: B023
                    x[:m], weight_q, scale, None, out_buf[:m], m, n, k,
                    x.stride(0), weight_q.stride(0), out_buf.stride(0),
                    HAS_BIAS=False, BN=bn, BK=bk, MP=GEMV_MAX_M)
                t = bench(fn, iters)
                if best is None or t < best[0]:
                    best = (t, bn, bk)
        results[m] = best
    return results


def sweep_gemm(x, weight_q, scale, ms, iters):
    from vllm.models.inkling2.nvidia.ops.fp8_w8a16 import (  # noqa: PLC0415
        _fp8_w8a16_gemm_kernel)
    import triton  # noqa: PLC0415
    n, k = weight_q.shape
    results = {}
    for m in ms:
        best = None
        for bm in GEMM_BM_CANDIDATES:
            for bn in GEMM_BN_CANDIDATES:
                for bk in GEMM_BK_CANDIDATES:
                    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
                    fn = lambda: _fp8_w8a16_gemm_kernel[grid](  # noqa: B023
                        x[:m], weight_q, scale, None, out_buf[:m], m, n, k,
                        x.stride(0), weight_q.stride(0), out_buf.stride(0),
                        HAS_BIAS=False, BM=bm, BN=bn, BK=bk)
                    t = bench(fn, iters)
                    if best is None or t < best[0]:
                        best = (t, bm, bn, bk)
        results[m] = best
    return results


def main() -> None:
    global out_buf
    args = parse_args()
    import torch  # noqa: PLC0415
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU; this script tunes real hardware")
    from vllm.models.inkling2.nvidia.ops.fp8_w8a16 import (  # noqa: PLC0415
        quant_weight_fp8)

    dtype = getattr(torch, args.dtype)
    shapes = DEFAULT_SHAPES + [parse_shape(s) for s in args.shape]
    max_m = max(max(GEMV_MS), max(GEMM_MS))

    for n, k in shapes:
        print(f"\n=== weight shape N={n} K={k} ({dtype}) ===")
        x = torch.randn(max_m, k, device="cuda", dtype=dtype)
        w = torch.randn(n, k, device="cuda", dtype=dtype)
        weight_q, scale = quant_weight_fp8(w)
        del w  # fp8 copy only from here on
        out_buf = torch.empty(max_m, n, device="cuda", dtype=dtype)

        gemv = sweep_gemv(x, weight_q, scale, GEMV_MS, args.iters)
        gemm = sweep_gemm(x, weight_q, scale, GEMM_MS, args.iters)

        bytes_per = n * k  # fp8 weight read dominates; x/out are tiny for m<=16
        for m, (t, bn, bk) in sorted(gemv.items()):
            gbps = bytes_per / (t * 1e-3) / 1e9
            print(f"GEMV m={m:<3} best BN={bn:<4} BK={bk:<4} "
                  f"{t * 1e3:8.1f} us  ({gbps:7.1f} GB/s weight read)")
        for m, (t, bm, bn, bk) in sorted(gemm.items()):
            gbps = bytes_per / (t * 1e-3) / 1e9
            print(f"GEMM m={m:<3} best BM={bm:<4} BN={bn:<4} BK={bk:<4} "
                  f"{t * 1e3:8.1f} us  ({gbps:7.1f} GB/s weight read)")
        del x, weight_q, scale, out_buf
        torch.cuda.empty_cache()

    print("\nIf the winners differ from the defaults (GEMV BN=32 BK=256, "
          "GEMM BM=64 BN=64 BK=64), set them before starting vLLM, e.g.:")
    print("  INKLING2_FP8_GEMV_BN=64 INKLING2_FP8_GEMV_BK=128 \\")
    print("      python benchmarks/inkling2/bench_5090.py --fp8-dense lm_head ...")
    print("Then paste the table into the fork thread so the defaults can be "
          "updated for sm_120.")


if __name__ == "__main__":
    main()
