# Inkling 2 Mini on RTX 5090 — FP8 W8A16 + DSpark (azjhf-dev fork)

Branch: `inkling2-mini-sm120-fp8`, rebased onto upstream vLLM `main` at
`3146a9df4` (upstream `main` as of 2026-11-24; base pinned here — rebases
are paused because our push token lacks the `workflow` scope GitHub
requires when upstream commits touch `.github/workflows/`). That base
includes the merged Inkling 2 Mini support (PR
[#69028](https://github.com/vllm-project/vllm/pull/69028), merged
2026-11-22 00:19 SGT as squash commit `ab18334a4f`). The merged version adds
review fixes over the PR head plus a new sm_120 `qkv_bmm` split-projection
path (`ops/ll_gemm_sm120.py`); our FP8 path takes precedence when enabled,
the bf16 `qkv_bmm` path is untouched otherwise.

## Why this fork exists

A r/LocalLLaMA thread (2026-11-21,
[post](https://www.reddit.com/r/LocalLLaMA/comments/1y1n4k7/inkling_2_mini_on_rtx_4090_200_toks_with_vllm_fp8/))
reports ~190-220 tok/s for Inkling 2 Mini on an RTX 4090 using:

1. a shared 7-edit sm89 enablement patch (in the thread, packaged by Claude,
   untested by anyone in the thread);
2. an **unshared** "bigger patch": FP8 W8A16 attention / lm_head / embeddings
   via a small Triton kernel, plus DSpark speculative drafting.

This fork re-implements part (2) ourselves for the **RTX 5090 (sm_120)**,
from primary sources only: PR #69028, the
[model card](https://thinkingmachines.ai/model-card/inkling-2-mini/), and the
[checkpoint config](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
We do not use or copy the Reddit poster's unshared code, and we do **not**
claim to reproduce their numbers. Their tok/s figures, 0.4% perplexity-loss
figure, and 24 GB / 35k-context fit are self-reported and unverified.

Note on the 4090 patch: the sm89 edits are irrelevant on a 5090. PR #69028
already supports SM120/SM121 natively (capability-family 120 gates), so the
5090 needs no enablement patch; only the FP8 work matters here.

## What is in this branch

| Piece | Status | Notes |
|---|---|---|
| Base: PR #69028 (model + DSpark drafter, NVFP4 experts, FP8 KV) | inherited, upstream-tested on GB200 | sm_120 covered by the family-120 gates |
| `nvidia/ops/fp8_w8a16.py`: FP8 W8A16 Triton GEMV (M<=16) + tiled GEMM, per-row scales, fp32 accumulation | added; CPU-reference tested in sandbox; **Triton kernels compile-tested only, not run on a GPU** | no FP8 tensor cores needed; bf16 math; `rows()` views for fused projections |
| `nvidia/fp8_dense.py` + `INKLING2_FP8_DENSE` env gating, wired into `load_weights` and `Inkling2LogitsProcessor._apply_head` | added; groups `lm_head`, `attn`, `embed` implemented | bf16 lm_head weight is kept: the DSpark drafter aliases it and asserts bf16 |
| FP8 for attention projections (qkvr / wo_ud) | **implemented (iteration 2)** | FP8 copies attached as `attn.qkvr_fp8` / `attn.wo_ud_fp8`; `_project` and the wo_ud call site use them when present (row slices via `rows()`); bf16 weights kept, LoRA skips conversion entirely; wo_ud pays one extra device copy into `out` for now |
| FP8 input embeddings | **implemented (iteration 2, EXPERIMENTAL)** | `Fp8Embedding` swap at TP=1 only; memory play (0.93 GB -> 0.46 GB); quality impact unmeasured, off unless requested |
| `tests/models/inkling2/test_fp8_w8a16.py` | added | CPU tests run in CI-style env; CUDA tests skip without a GPU |
| `benchmarks/inkling2/bench_5090.py` | added, not yet run on hardware | configurable: fp8 on/off, DSpark on/off, context, batch, lens |
| `benchmarks/inkling2/fp8_weight_probe.py` | added; **run on the real checkpoint weights** | range-downloads one tensor from the HF safetensors file and measures FP8 error; `--from-file` for local dumps |

## Measured FP8 error on the real weights (CPU, 2026-11-23)

`fp8_weight_probe.py` on `model.llm.embed.weight` and
`model.llm.unembed.weight` (201024 x 2304 bf16 each), per-row e4m3 absmax:

| Tensor | global rel. Frobenius err | row err median / p99 / max | dot-product sim |
|---|---|---|---|
| embed | 2.64% | 2.64% / 2.76% / 2.88% | 2.64% |
| unembed (lm_head) | 2.64% | 2.64% / 2.76% / 2.94% | 2.63% |

Tight distribution, ~2^-5.2, as expected for e4m3 with per-row scales.
What this means: lm_head FP8 perturbs every logit by ~2.6% relative, which
can flip greedy argmax when the top tokens are close; embed FP8 injects
~2.6% relative error per token directly into the residual stream (kept
EXPERIMENTAL for that reason). Whether either lands near the Reddit OP's
claimed ~0.4% perplexity loss can only be answered by running the model -
the probe bounds the weight-space error, not the task metric.

## Usage (on a 5090)

```bash
# stock PR behaviour (bf16 head, DSpark off)
python benchmarks/inkling2/bench_5090.py --model thinkingmachines/Inkling-2-Mini

# our FP8 W8A16 lm_head + attention + DSpark, 35k context like the Reddit setup
python benchmarks/inkling2/bench_5090.py \
    --model thinkingmachines/Inkling-2-Mini \
    --fp8-dense lm_head,attn --dspark --num-speculative-tokens 7 \
    --max-model-len 35840 --input-len 512 --output-len 256 --num-prompts 8
```

`INKLING2_FP8_DENSE` is a comma list of `lm_head`, `attn`, `embed`
(`embed` is experimental). Unknown values raise; nothing is on by default.

## Quality gate (2026-11-24)

`benchmarks/inkling2/quality_gate.py` measures the perplexity cost of the
FP8 path so the Reddit "+0.4% perplexity" claim can be checked on your own
hardware. Teacher-forced perplexity over WikiText-2 (configurable) via vLLM
`prompt_logprobs`, same model code and the same `INKLING2_FP8_DENSE` switch
as generation:

```bash
python benchmarks/inkling2/quality_gate.py --json-out ppl_bf16.json
python benchmarks/inkling2/quality_gate.py --fp8-dense lm_head --json-out ppl_fp8.json
python benchmarks/inkling2/quality_gate.py --compare ppl_bf16.json ppl_fp8.json
```

Window placement is deterministic (fixed seed), so two runs with identical
flags score identical text and the delta is apples-to-apples. CPU-tested:
window construction and the compare logic (6 unit tests). The evaluation
loop itself needs a GPU and has not run yet; the OP's 0.4% stays
unverified until someone runs it.

## Triton tile tuning (2026-11-25, needs one 5090 run)

The FP8 kernels ship conservative fixed tile sizes (GEMV BN=32 BK=256,
GEMM BM=64 BN=64 BK=64) shared across SM90/SM100/SM120. They are
untested on hardware. `benchmarks/inkling2/tune_fp8_tiles.py` sweeps
candidate tiles for the real Inkling 2 Mini shapes on your GPU and prints
the winners as env settings. Override without recompiling:

```bash
INKLING2_FP8_GEMV_BN=64 INKLING2_FP8_GEMV_BK=128 \
    python benchmarks/inkling2/bench_5090.py --fp8-dense lm_head ...
```

Env values must be powers of two >= 16 (tl.dot minimum); invalid values
raise at import. Defaults stay until a sweep on the 5090 says otherwise;
paste the table into the fork thread and the sm_120 defaults get updated.

## Expected effect (reasoning, not measurement)

Decode is bandwidth-bound. Per generated token the dense reads are dominated
by the LM head (201024 x 2304 bf16 = ~0.93 GB, TP1). FP8 W8A16 halves that
read; qkvr/wo_ud/attn_res are tens of MB per token and secondary. The
experts are already NVFP4 and the KV cache already FP8 in PR #69028. DSpark
(with its acceptance rate) multiplies whatever the verify step costs.
Net: expect a meaningful but single-digit-to-tens-of-percent gain from the
lm_head FP8 path alone, plus the DSpark multiplier. Measure, don't trust.

## Honesty / gaps

- No GPU in the authoring sandbox: the Triton kernels are compile-checked
  and mirrored by a CPU reference (unit-tested); they have **not** executed
  on hardware. First 5090 run may surface bugs (Triton fp8 load conversion
  on sm_120 is the most likely spot).
- The DSpark enablement in `bench_5090.py` (`speculative_config` with
  `method: dspark`, draft auto-derived from the checkpoint `mtp_config`)
  follows vLLM's MTP conventions and the PR's `Inkling2DSparkSpeculatorConfig`
  but has not been run; exact required keys may differ.
- `llm.get_metrics()` acceptance-rate reporting is best-effort across vLLM
  versions.
- LoRA + FP8 lm_head is rejected (LoRA paths read the raw weight).

## Licenses

- vLLM and this fork's code: Apache-2.0 (SPDX headers kept).
- Inkling 2 Mini weights: Apache-2.0 per the model card, plus the Thinking
  Machines
  [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/).
- The Reddit sm89 patch is not included anywhere in this branch.
