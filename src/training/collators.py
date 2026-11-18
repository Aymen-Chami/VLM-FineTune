"""Batch collators turning shared records into model inputs for training."""
from __future__ import annotations

from typing import Sequence

import torch

from src.models import paligemma_prompts


class PaliGemmaTrainCollator:
    """prompt = "<image>answer en {question}", target = first gold answer.

    The PaliGemma processor builds `labels` from `suffix` and masks the image + prompt tokens with
    -100; we additionally mask padding so it never contributes to the loss.
    """

    def __init__(self, processor, dtype: torch.dtype = torch.bfloat16):
        self.processor = processor
        self.dtype = dtype

    def __call__(self, records: Sequence[dict]) -> dict:
        self.processor.tokenizer.padding_side = "right"
        batch = self.processor(
            text=paligemma_prompts(records),
            images=[r["image"] for r in records],
            suffix=[r["answers"][0] for r in records],
            return_tensors="pt",
            padding="longest",
        )
        labels = batch["labels"]
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        batch["pixel_values"] = batch["pixel_values"].to(self.dtype)
        return dict(batch)
