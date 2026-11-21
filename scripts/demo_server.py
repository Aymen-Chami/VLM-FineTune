"""GPU model server for the web demo: answers each question with the base and the fine-tuned model.

    python scripts/demo_server.py                     # listens on 127.0.0.1:8000, POST /infer
    python scripts/demo_server.py --pg-adapter <hub id or dir> --florence-ft <hub id or dir>

The website (app/) forwards requests here when INFERENCE_URL=http://127.0.0.1:8000/infer.
Prompts are the ones used in training/evaluation (src/models.py).
"""
from __future__ import annotations

import argparse
import io
import sys
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import postprocess_prediction  # noqa: E402
from src.models import FLORENCE_PROMPT, PALIGEMMA_PROMPT  # noqa: E402

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_NEW_TOKENS = 64
DTYPE = torch.bfloat16


class PaliGemmaPair:
    """One copy of the base weights; the LoRA adapter is switched off for the base answer."""

    def __init__(self, base: str, adapter: str):
        from peft import PeftModel
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(base)
        model = PaliGemmaForConditionalGeneration.from_pretrained(base, dtype=DTYPE, attn_implementation="eager")
        self.model = PeftModel.from_pretrained(model, adapter).to("cuda").eval()

    def prompt(self, template: str, text: str) -> str:
        return "<image>caption en" if template == "caption" else PALIGEMMA_PROMPT.format(question=text)

    @torch.inference_mode()
    def answer(self, image: Image.Image, prompt: str, finetuned: bool) -> str:
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)
        kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
        if finetuned:
            out = self.model.generate(**inputs, **kwargs)
        else:
            with self.model.disable_adapter():
                out = self.model.generate(**inputs, **kwargs)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return postprocess_prediction(text)


class FlorencePair:
    """Florence-2 was fully fine-tuned, so base and fine-tuned are two small separate models."""

    def __init__(self, base: str, finetuned: str):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(base)
        self.models = {
            ft: Florence2ForConditionalGeneration.from_pretrained(path, dtype=DTYPE).to("cuda").eval()
            for ft, path in ((False, base), (True, finetuned))
        }

    def prompt(self, template: str, text: str) -> str:
        return "<CAPTION>" if template == "caption" else FLORENCE_PROMPT.format(question=text)

    @torch.inference_mode()
    def answer(self, image: Image.Image, prompt: str, finetuned: bool) -> str:
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
        out = self.models[finetuned].generate(
            input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs["pixel_values"].to(DTYPE), max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
        return postprocess_prediction(self.processor.batch_decode(out, skip_special_tokens=True)[0])


def build_app(models: dict) -> FastAPI:
    app = FastAPI(title="VLM demo model server")
    gpu = threading.Lock()  # one request on the GPU at a time

    @app.get("/health")
    def health():
        return {"ok": True, "models": list(models)}

    @app.post("/infer")
    def infer(image: UploadFile = File(...), prompt: str = Form(""), question: str = Form(""),
              template: str = Form("vqa"), model: str = Form(...)):
        pair = models.get(model)
        if pair is None:
            raise HTTPException(400, f"unknown model {model!r}")
        data = image.file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "image too large")
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(400, "could not read the image")
        full_prompt = pair.prompt(template, prompt or question)

        if not gpu.acquire(timeout=120):
            raise HTTPException(503, "the model is busy, try again in a moment")
        try:
            result = {}
            for key, finetuned in (("base", False), ("finetuned", True)):
                t0 = time.perf_counter()
                answer = pair.answer(img, full_prompt, finetuned)
                result[key] = {"answer": answer, "latencyMs": round((time.perf_counter() - t0) * 1000)}
            return result
        finally:
            gpu.release()

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pg-base", default="google/paligemma2-3b-pt-448")
    ap.add_argument("--pg-adapter", default="outputs/pg2_448/checkpoint-4000")
    ap.add_argument("--florence-base", default="florence-community/Florence-2-base-ft")
    ap.add_argument("--florence-ft", default="outputs/florence2_base/checkpoint-3000")
    args = ap.parse_args()

    print("loading PaliGemma 2 + LoRA ...", flush=True)
    models = {"paligemma2": PaliGemmaPair(args.pg_base, args.pg_adapter)}
    print("loading Florence-2 base + fine-tuned ...", flush=True)
    models["florence2"] = FlorencePair(args.florence_base, args.florence_ft)
    print(f"ready, GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    uvicorn.run(build_app(models), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
