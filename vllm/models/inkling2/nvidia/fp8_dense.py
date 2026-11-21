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

from .ops.fp8_w8a16 import Fp8W8A16Linear

if TYPE_CHECKING:
    from .model import Inkling2ForConditionalGeneration

logger = init_logger(__name__)

ENV_VAR = "INKLING2_FP8_DENSE"
IMPLEMENTED_GROUPS = ("lm_head",)
PLANNED_GROUPS = ("attn", "embed")
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

    if "lm_head" not in groups:
        return applied

    # Local import to avoid a cycle: model.py imports this module lazily.
    from .model import Inkling2LogitsProcessor

    if not isinstance(model.logits_processor, Inkling2LogitsProcessor):
        logger.warning(
            "%s=lm_head skipped: LoRA is enabled and the LoRA logits path "
            "reads the raw bf16 weight.",
            ENV_VAR,
        )
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
