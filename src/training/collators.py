"""Batch collators / losses turning shared records into model inputs for training."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from src.models import paligemma_prompts


class PaliGemmaTrainCollator:
    """prompt = "<image>answer en {question}", target = first gold answer.

    The PaliGemma processor builds `labels` from `suffix` and masks the image + prompt tokens with
    -100.  Memory trick: we pad on the LEFT (as at inference), so every answer sits at the end of its
    sequence, and ask the model for logits of the last K positions only (`logits_to_keep`).  Full
    logits would be batch x ~1050 tokens x 257k vocab (~4 GB in fp32 for batch 4) although only the
    ~10 answer tokens per sample are supervised.  `labels` is sliced to the same last K positions and
    the loss is computed by `answer_only_loss` below.
    """

    def __init__(self, processor, dtype: torch.dtype = torch.bfloat16):
        self.processor = processor
        self.dtype = dtype

    def __call__(self, records: Sequence[dict]) -> dict:
        self.processor.tokenizer.padding_side = "left"
        batch = self.processor(
            text=paligemma_prompts(records),
            images=[r["image"] for r in records],
            suffix=[r["answers"][0] for r in records],
            return_tensors="pt",
            padding="longest",
        )
        labels = batch["labels"]
        labels[batch["attention_mask"] == 0] = -100
        # +1: the logit that predicts the first answer token sits one position before it
        keep = int((labels != -100).sum(dim=1).max().item()) + 1
        batch = dict(batch)
        batch["labels"] = labels[:, -keep:].contiguous()
        batch["logits_to_keep"] = keep
        batch["pixel_values"] = batch["pixel_values"].to(self.dtype)
        return batch


def answer_only_loss(outputs, labels, num_items_in_batch=None):
    """Causal-LM cross-entropy on the last-K logits returned with `logits_to_keep`.

    logits[:, t] predicts labels[:, t + 1].  Summed and divided by the number of target tokens in
    the whole accumulated batch (the Trainer passes `num_items_in_batch`), so gradient
    accumulation gives the same result as one big batch.
    """
    logits = outputs.logits[:, :-1, :].float()
    targets = labels[:, 1:].to(logits.device)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                           ignore_index=-100, reduction="sum")
    denom = num_items_in_batch if num_items_in_batch is not None else (targets != -100).sum()
    return loss / denom
