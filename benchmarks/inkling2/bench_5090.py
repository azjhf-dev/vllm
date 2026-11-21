# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 Mini RTX 5090 (sm_120) benchmark — azjhf-dev vLLM fork.

Measures end-to-end generation throughput with the levers the r/LocalLLaMA
~200 tok/s claim rests on, independently switchable:

  --fp8-dense lm_head   our FP8 W8A16 Triton path for the LM head
                        (INKLING2_FP8_DENSE; default off)
  --dspark              DSpark speculative decoding (method "dspark", draft
                        auto-derived from the checkpoint's mtp_config)

Example (single 5090, 35k context like the Reddit post):

  python benchmarks/inkling2/bench_5090.py \
      --model thinkingmachines/Inkling-2-Mini \
      --fp8-dense lm_head --dspark --num-speculative-tokens 7 \
      --max-model-len 35840 --input-len 512 --output-len 256 --num-prompts 8

Run each configuration as a separate process invocation (vLLM caches engine
state). This script does not claim to reproduce anyone else's numbers; it
produces OUR numbers on YOUR hardware. Report acceptance lengths from the
printed spec-decode metrics when --dspark is on.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="thinkingmachines/Inkling-2-Mini")
    p.add_argument("--fp8-dense", default="",
                   help="value for INKLING2_FP8_DENSE, e.g. 'lm_head' (empty = off)")
    p.add_argument("--dspark", action="store_true",
                   help="enable DSpark speculative decoding")
    p.add_argument("--num-speculative-tokens", type=int, default=7)
    p.add_argument("--max-model-len", type=int, default=35840)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default="",
                   help="optional path to write the results dict")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Must be set before vLLM (and the model code) is imported.
    if args.fp8_dense:
        os.environ["INKLING2_FP8_DENSE"] = args.fp8_dense

    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise SystemExit("bench_5090.py needs a CUDA GPU (RTX 5090 target)")
    cap = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()}  capability sm_{cap[0]}{cap[1]}")

    speculative_config = None
    if args.dspark:
        speculative_config = {
            "method": "dspark",
            "model": args.model,  # draft auto-derives from mtp_config
            "num_speculative_tokens": args.num_speculative_tokens,
        }

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
        enforce_eager=False,
        seed=args.seed,
    )

    # Deterministic pseudo-random prompts of the requested length.
    g = torch.Generator().manual_seed(args.seed)
    prompt_ids = torch.randint(
        1000, 200000, (args.num_prompts, args.input_len), generator=g
    ).tolist()
    prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.output_len,
        ignore_eos=True,
    )

    # Warmup (also captures CUDA graphs / Triton autotune out of the timing).
    llm.generate(prompts[:1], SamplingParams(temperature=0.0, max_tokens=8))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sampling)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    mem = torch.cuda.max_memory_allocated() / 2**30

    result = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(),
        "sm": f"{cap[0]}{cap[1]}",
        "fp8_dense": args.fp8_dense or None,
        "dspark": args.dspark,
        "num_speculative_tokens": args.num_speculative_tokens if args.dspark else 0,
        "max_model_len": args.max_model_len,
        "num_prompts": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "temperature": args.temperature,
        "wall_s": round(dt, 3),
        "output_tokens": out_tokens,
        "agg_output_tok_s": round(out_tokens / dt, 2),
        "per_request_tok_s": round(out_tokens / dt / args.num_prompts, 2),
        "peak_mem_gib": round(mem, 2),
    }
    print(json.dumps(result, indent=2))

    # Spec-decode acceptance stats, when available (v1 metrics).
    try:
        for m in llm.get_metrics():  # type: ignore[attr-defined]
            name = getattr(m, "name", "")
            if "spec_decod" in name or "acceptance" in name:
                print("metric:", name, getattr(m, "value", m))
    except Exception as e:  # metrics API differs across versions; non-fatal
        print(f"(spec-decode metrics unavailable: {e})")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
