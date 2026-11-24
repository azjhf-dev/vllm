# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 Mini FP8 quality gate — azjhf-dev vLLM fork.

Measures the perplexity cost of the FP8 W8A16 dense path on real text, so
the r/LocalLLaMA "0.4% perplexity increase" claim can be checked on your
own hardware. The OP's number is unverified; this script produces OUR
delta, it does not reproduce theirs.

Method: teacher-forced perplexity over a HF dataset via vLLM
prompt_logprobs. Each evaluation window is fed as a prompt, and the mean
negative log-likelihood of tokens `window_overlap`..end is summed. This
runs the exact same model code (and the same INKLING2_FP8_DENSE switch) as
generation, so the number reflects the deployed path.

Run the baseline and the FP8 variant as separate process invocations and
compare:

  python benchmarks/inkling2/quality_gate.py --json-out ppl_bf16.json
  python benchmarks/inkling2/quality_gate.py --fp8-dense lm_head \
      --json-out ppl_fp8_lmhead.json
  python benchmarks/inkling2/quality_gate.py --compare ppl_bf16.json ppl_fp8_lmhead.json

Requires a GPU; this script is not meaningful in a CPU sandbox. Use
--max-windows for a quick smoke run (fewer windows = wider error bars).
Window placement is deterministic (fixed seed, sequential stride), so two
runs with identical flags evaluate identical text.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass


@dataclass
class GateResult:
    model: str
    fp8_dense: str
    dataset: str
    split: str
    window: int
    stride: int
    seed: int
    num_windows: int
    total_scored_tokens: int
    mean_nll: float
    perplexity: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="thinkingmachines/Inkling-2-Mini")
    p.add_argument("--fp8-dense", default="",
                   help="value for INKLING2_FP8_DENSE, e.g. 'lm_head' or 'all' "
                        "(empty = bf16 baseline)")
    p.add_argument("--dataset", default="Salesforce/wikitext",
                   help="HF dataset with a text column")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="test")
    p.add_argument("--text-field", default="text")
    p.add_argument("--window", type=int, default=2048,
                   help="tokens scored per window")
    p.add_argument("--stride", type=int, default=2048,
                   help="window advance; < window re-uses context, > window skips text")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-windows", type=int, default=0,
                   help="cap windows (0 = all that fit)")
    p.add_argument("--max-model-len", type=int, default=0,
                   help="override vLLM max model len (0 = window size)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--json-out", default="", help="write the result dict here")
    p.add_argument("--compare", nargs=2, metavar=("BASELINE_JSON", "VARIANT_JSON"),
                   help="skip evaluation; print the delta between two saved runs")
    return p.parse_args()


def load_text(args: argparse.Namespace) -> str:
    from datasets import load_dataset  # noqa: PLC0415 - optional heavy dep

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    # wikitext rows are lines; join preserves paragraph breaks.
    return "\n".join(ds[args.text_field])


def make_windows(token_ids: list[int], window: int, stride: int,
                 seed: int, max_windows: int) -> list[tuple[int, int]]:
    """Deterministic (start, end) token spans of length `window`."""
    spans: list[tuple[int, int]] = []
    starts = list(range(0, max(len(token_ids) - window + 1, 0), stride))
    random.Random(seed).shuffle(starts)
    for s in sorted(starts[:max_windows or None]):
        spans.append((s, s + window))
    return spans


def run(args: argparse.Namespace) -> GateResult:
    if args.fp8_dense:
        os.environ["INKLING2_FP8_DENSE"] = args.fp8_dense

    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    text = load_text(args)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(text, return_attention_mask=False)["input_ids"]

    spans = make_windows(ids, args.window, args.stride, args.seed, args.max_windows)
    if not spans:
        raise SystemExit(
            f"dataset too short: {len(ids)} tokens < window {args.window}")

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len or args.window,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1,
                            prompt_logprobs=0, detokenize=False)

    prompts = [ids[s:e] for s, e in spans]
    outs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts], params, use_tqdm=True)

    nll_sum = 0.0
    scored = 0
    for out in outs:
        pls = out.prompt_logprobs
        assert pls is not None
        for pos, tok_id in enumerate(out.prompt_token_ids):
            if pos == 0:
                continue  # first token has no context
            entry = pls[pos]
            if entry is None or tok_id not in entry:
                continue  # position not returned (e.g. chunked-prefill edge)
            nll_sum -= entry[tok_id].logprob
            scored += 1

    mean_nll = nll_sum / max(scored, 1)
    res = GateResult(
        model=args.model, fp8_dense=args.fp8_dense or "off",
        dataset=f"{args.dataset}/{args.dataset_config}", split=args.split,
        window=args.window, stride=args.stride, seed=args.seed,
        num_windows=len(spans), total_scored_tokens=scored,
        mean_nll=mean_nll, perplexity=math.exp(mean_nll))
    return res


def compare(a_path: str, b_path: str) -> None:
    with open(a_path) as f:
        a = json.load(f)
    with open(b_path) as f:
        b = json.load(f)
    same = (a["dataset"], a["window"], a["stride"], a["seed"]) == \
           (b["dataset"], b["window"], b["stride"], b["seed"])
    delta = (b["perplexity"] - a["perplexity"]) / a["perplexity"] * 100
    print(f"baseline  ppl {a['perplexity']:.4f}  ({a['fp8_dense']}, "
          f"{a['num_windows']} windows, {a['total_scored_tokens']} tokens)")
    print(f"variant   ppl {b['perplexity']:.4f}  ({b['fp8_dense']})")
    print(f"delta     {delta:+.3f}%")
    if not same:
        print("WARNING: runs used different dataset/window/stride/seed; "
              "delta is not apples-to-apples")
    print("note: the Reddit OP's +0.4% claim is unverified; this delta is "
          "measured on your hardware with this fork.")


def main() -> None:
    args = parse_args()
    if args.compare:
        compare(*args.compare)
        return
    res = run(args)
    print(json.dumps(asdict(res), indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(asdict(res), f, indent=2)


if __name__ == "__main__":
    main()
