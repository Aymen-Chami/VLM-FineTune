"""Model wrappers with a common interface: `predict(records) -> list[str]`.

The prompt formats are fixed by PLAN.md (identical for both datasets, so the demo app does not
need to know where a question comes from).  Do not change them without telling the other person.
"""
from __future__ import annotations

from typing import Sequence

import torch

from src.evaluation.metrics import postprocess_prediction

PALIGEMMA_PROMPT = "<image>answer en {question}"
FLORENCE_PROMPT = "<VQA>{question}"

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def paligemma_prompts(records: Sequence[dict]) -> list[str]:
    return [PALIGEMMA_PROMPT.format(question=r["question"]) for r in records]


def florence_prompts(records: Sequence[dict]) -> list[str]:
    return [FLORENCE_PROMPT.format(question=r["question"]) for r in records]


class PaliGemmaVQA:
    family = "paligemma2"

    def __init__(self, base: str = "google/paligemma2-3b-pt-448", adapter: str | None = None,
                 dtype: str = "bf16", attn_implementation: str | None = None, device: str = "cuda"):
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self.device, self.dtype = device, DTYPES[dtype]
        self.processor = AutoProcessor.from_pretrained(base)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            base, dtype=self.dtype, attn_implementation=attn_implementation)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(device).eval()

    @torch.inference_mode()
    def predict(self, records: Sequence[dict], max_new_tokens: int = 64) -> list[str]:
        tok = self.processor.tokenizer
        tok.padding_side = "left"  # decoder-only batched generation needs left padding
        inputs = self.processor(text=paligemma_prompts(records), images=[r["image"] for r in records],
                                return_tensors="pt", padding="longest")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return [postprocess_prediction(t) for t in texts]


class FlorenceVQA:
    family = "florence2"

    def __init__(self, base: str = "florence-community/Florence-2-base-ft", adapter: str | None = None,
                 dtype: str = "fp16", device: str = "cuda", **_):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self.device, self.dtype = device, DTYPES[dtype]
        self.processor = AutoProcessor.from_pretrained(base)
        # For Florence-2 the "adapter" is a full fine-tuned checkpoint (path or Hub id).
        model = Florence2ForConditionalGeneration.from_pretrained(adapter or base, dtype=self.dtype)
        self.model = model.to(device).eval()

    @torch.inference_mode()
    def predict(self, records: Sequence[dict], max_new_tokens: int = 64) -> list[str]:
        inputs = self.processor(text=florence_prompts(records), images=[r["image"] for r in records],
                                return_tensors="pt", padding="longest")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        out = self.model.generate(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                                  pixel_values=inputs["pixel_values"], max_new_tokens=max_new_tokens,
                                  do_sample=False, num_beams=1)
        texts = self.processor.batch_decode(out, skip_special_tokens=True)
        return [postprocess_prediction(t) for t in texts]


def load_model(family: str, **kwargs):
    if family == "paligemma2":
        return PaliGemmaVQA(**kwargs)
    if family == "florence2":
        return FlorenceVQA(**kwargs)
    raise ValueError(f"unknown model family {family!r}")
