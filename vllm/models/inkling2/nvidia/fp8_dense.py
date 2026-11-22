# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in FP8 W8A16 conversion of Inkling2 dense weights (azjhf-dev fork).

Enabled with the ``INKLING2_FP8_DENSE`` environment variable, a comma list of
groups:

- ``lm_head``: quantize the LM head (201024 x 2304 for Inkling 2 Mini) to
  FP8 W8A16. The target logits path in ``Inkling2LogitsProcessor`` then reads
  ~0.46 GB instead of ~0.93 GB per step, the largest dense weight read at
  decode time. The bf16 weight is KEPT: the DSpark drafter aliases
  ``lm_head.weight`` (draft logits + Markov fold) and asserts bf16.
- ``attn``  : (planned, not yet implemented) qkvr / wo_ud projections.
- ``embed`` : (planned, not yet implemented) input embeddings. Experimental:
  per-row FP8 error feeds the residual stream directly; quality impact must
  be measured before this is recommended.

Skipped with a warning when LoRA is enabled (the LoRA logits path and
``add_linear_lora`` read the raw ``.weight``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

from .ops.fp8_w8a16 import Fp8W8A16Linear, quant_weight_fp8

if TYPE_CHECKING:
    from .model import Inkling2ForConditionalGeneration

logger = init_logger(__name__)

ENV_VAR = "INKLING2_FP8_DENSE"
IMPLEMENTED_GROUPS = ("lm_head", "attn", "embed")
PLANNED_GROUPS: tuple[str, ...] = ()
VALID_GROUPS = IMPLEMENTED_GROUPS + PLANNED_GROUPS


def fp8_dense_groups() -> frozenset[str]:
    raw = os.environ.get(ENV_VAR, "")
    groups = {g.strip().lower() for g in raw.split(",") if g.strip()}
    unknown = groups - set(VALID_GROUPS)
    if unknown:
        raise ValueError(
            f"{ENV_VAR}: unknown group(s) {sorted(unknown)}; "
            f"valid groups are {list(VALID_GROUPS)}"
        )
    return frozenset(groups)


def apply_fp8_dense(model: "Inkling2ForConditionalGeneration") -> list[str]:
    """Quantize the requested dense groups in place. Called once at the end
    of ``Inkling2ForConditionalGeneration.load_weights``. Returns the groups
    actually applied."""
    groups = fp8_dense_groups()
    applied: list[str] = []
    if not groups:
        return applied

    for planned in set(groups) & set(PLANNED_GROUPS):
        logger.warning(
            "%s=%s is not implemented yet in this fork; skipping. "
            "Track progress in INKLING2_5090.md.",
            ENV_VAR,
            planned,
        )

    # Local import to avoid a cycle: model.py imports this module lazily.
    from .model import Inkling2LogitsProcessor

    if not isinstance(model.logits_processor, Inkling2LogitsProcessor):
        if groups:
            logger.warning(
                "%s=%s skipped: LoRA is enabled and the LoRA paths read the "
                "raw bf16 weights.",
                ENV_VAR,
                ",".join(sorted(groups)),
            )
        return applied

    if "attn" in groups:
        n = _convert_attn(model)
        if n:
            applied.append("attn")
            logger.info(
                "%s: qkvr/wo_ud of %d attention layers quantized to FP8 W8A16 "
                "(bf16 weights kept; add_linear_lora is a no-op without LoRA).",
                ENV_VAR,
                n,
            )

    if "embed" in groups:
        if _convert_embed(model):
            applied.append("embed")

    if "lm_head" not in groups:
        return applied

    weight = model.lm_head.weight
    if weight.dtype != torch.bfloat16:
        logger.warning(
            "%s=lm_head skipped: expected a bf16 lm_head weight, got %s.",
            ENV_VAR,
            weight.dtype,
        )
        return applied

    model.lm_head.fp8_w8a16 = Fp8W8A16Linear.from_weight(weight)
    applied.append("lm_head")
    logger.info(
        "%s: lm_head quantized to FP8 W8A16 (%s -> %s bytes for the weight). "
        "The bf16 weight is kept for the DSpark drafter.",
        ENV_VAR,
        weight.numel() * weight.element_size(),
        model.lm_head.fp8_w8a16.weight_q.numel(),
    )
    return applied


def _convert_attn(model: "Inkling2ForConditionalGeneration") -> int:
    """Attach FP8 copies of every attention layer's qkvr / wo_ud weights.

    The bf16 weights are kept (small: ~54 MB per layer pair on Inkling 2
    Mini); ``Inkling2Attention._project`` and the wo_ud call site read the
    FP8 copy when present. Returns the number of layers converted.
    """
    converted = 0
    for layer in model.model.layers:
        attn = getattr(layer, "attn", None)
        if attn is None or getattr(attn, "qkvr_fp8", None) is not None:
            continue
        attn.qkvr_fp8 = Fp8W8A16Linear.from_weight(attn.qkvr.weight)
        attn.wo_ud_fp8 = Fp8W8A16Linear.from_weight(attn.wo_ud.weight)
        converted += 1
    return converted


class Fp8Embedding(torch.nn.Module):
    """FP8 W8A16 drop-in for VocabParallelEmbedding at TP=1 (EXPERIMENTAL).

    Per-row FP8 gather + dequant. The win is memory (0.93 GB -> 0.46 GB for
    Inkling 2 Mini's 201024 x 2304 table), not speed: decode gathers one row
    per token. Quality impact is unmeasured - per-row FP8 error feeds the
    residual stream directly. Not used by default.
    """

    def __init__(self, orig: torch.nn.Module) -> None:
        super().__init__()
        weight_q, scale = quant_weight_fp8(orig.weight)
        # Buffers so .to()/device moves keep working; not nn.Parameters, so
        # checkpoint-loading traversals never see them (conversion runs
        # after load_weights).
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("scale", scale)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        w = self.weight_q[input_]
        s = self.scale[input_]
        return (w.to(torch.bfloat16).to(torch.float32) * s[..., None]).to(
            torch.bfloat16
        )


def _convert_embed(model: "Inkling2ForConditionalGeneration") -> bool:
    from vllm.distributed import get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() != 1:
        logger.warning(
            "%s=embed skipped: only TP=1 is supported (vocab shards need "
            "masked gathers).",
            ENV_VAR,
        )
        return False
    embed = model.model.embed_tokens
    if isinstance(embed, Fp8Embedding):
        return False
    if embed.weight.dtype != torch.bfloat16:
        logger.warning(
            "%s=embed skipped: expected a bf16 embedding, got %s.",
            ENV_VAR,
            embed.weight.dtype,
        )
        return False
    model.model.embed_tokens = Fp8Embedding(embed)
    logger.info(
        "%s: input embeddings quantized to FP8 W8A16 (EXPERIMENTAL, memory "
        "play only; quality impact unmeasured).",
        ENV_VAR,
    )
    return True
