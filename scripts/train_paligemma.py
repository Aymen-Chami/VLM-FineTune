"""LoRA fine-tuning of PaliGemma 2 on DocVQA + ChartQA (mixed).

    python scripts/train_paligemma.py --config configs/paligemma2_448.yaml
    # overfit sanity check (PLAN.md: must reach ~0 loss and reproduce the answers)
    python scripts/train_paligemma.py --config configs/paligemma2_448.yaml --run-name overfit \
        --limit-train 64 --epochs 15 --save-steps 0 --no-eval
    # resume after an interruption
    python scripts/train_paligemma.py --config configs/paligemma2_448.yaml --resume

Outputs: <output_dir>/checkpoint-*/ (LoRA adapters), <output_dir>/final/, train_ids.json (when
--limit-train is used), and results/training/<run_name>.json (loss curves, config, timing).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loaders import REPO_ROOT, MixedDataset, VQADataset, read_ids  # noqa: E402
from src.training.collators import PaliGemmaTrainCollator, answer_only_loss  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--limit-train", type=int, default=None, help="total train samples (split evenly)")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=None, help="0 = only save the final adapter")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    t, d = cfg["train"], cfg["data"]
    run_name = args.run_name or cfg["run_name"]
    out_dir = Path(args.output_dir or (cfg["output_dir"] if not args.run_name else f"outputs/{run_name}"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in {"epochs": args.epochs, "max_steps": args.max_steps, "lr": args.lr,
                 "per_device_batch_size": args.batch_size, "grad_accum": args.grad_accum,
                 "save_steps": args.save_steps}.items():
        if v is not None:
            t[k] = v

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoProcessor, PaliGemmaForConditionalGeneration, Trainer,
                              TrainingArguments)

    # ------------------------------------------------------------------ data
    dev_ids = read_ids("docvqa_dev_ids.json")
    doc_lim, chart_lim = d["docvqa_limit"], d["chartqa_limit"]
    if args.limit_train:
        doc_lim = chart_lim = args.limit_train // 2
    train_doc = VQADataset("docvqa", "train", exclude_ids=dev_ids, limit=doc_lim)
    train_chart = VQADataset("chartqa", "train", limit=chart_lim)
    train_ds = MixedDataset([train_doc, train_chart])
    if args.limit_train:
        (out_dir / "train_ids.json").write_text(json.dumps(train_doc.ids() + train_chart.ids()))
    eval_ds = None
    if not args.no_eval:
        n = d["eval_loss_samples"]
        eval_ds = MixedDataset([
            VQADataset("docvqa", "train", ids=read_ids("docvqa_dev500_ids.json")[:n]),
            VQADataset("chartqa", "val", ids=read_ids("chartqa_val500_ids.json")[:n]),
        ])
    print(f"[train] {run_name}: train={len(train_ds)} (docvqa {len(train_doc)}, chartqa {len(train_chart)}), "
          f"eval_loss={len(eval_ds) if eval_ds else 0}", flush=True)

    # ------------------------------------------------------------------ model
    processor = AutoProcessor.from_pretrained(cfg["base"])
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        cfg["base"], dtype=torch.bfloat16, attn_implementation=cfg.get("attn_implementation"))
    model.config.use_cache = False
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_regex"], bias="none"))
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable params {n_trainable / 1e6:.1f}M / {n_total / 1e9:.2f}B "
          f"({100 * n_trainable / n_total:.2f}%)", flush=True)

    # ------------------------------------------------------------------ trainer
    eff_bs = t["per_device_batch_size"] * t["grad_accum"]
    steps_per_epoch = math.ceil(len(train_ds) / eff_bs)
    total_steps = t["max_steps"] if t["max_steps"] and t["max_steps"] > 0 else math.ceil(steps_per_epoch * t["epochs"])
    save_steps = t["save_steps"]
    targs = TrainingArguments(
        output_dir=str(out_dir),
        run_name=run_name,
        num_train_epochs=t["epochs"],
        max_steps=t["max_steps"] if t["max_steps"] else -1,
        per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=t["per_device_batch_size"] * 2,
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"],
        lr_scheduler_type=t["scheduler"],
        warmup_steps=int(t["warmup_ratio"] * total_steps),
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t["logging_steps"],
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=t["eval_steps"],
        dataloader_num_workers=t["num_workers"],
        dataloader_persistent_workers=t["num_workers"] > 0,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["tensorboard"],
        prediction_loss_only=True,
        seed=t["seed"],
        optim="adamw_torch_fused",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=PaliGemmaTrainCollator(processor), compute_loss_func=answer_only_loss)

    t0 = time.time()
    resume = args.resume and any(out_dir.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    train_time = time.time() - t0
    trainer.save_model(str(out_dir / "final"))

    # ------------------------------------------------------------------ log for the report
    log = {
        "run_name": run_name, "config": cfg, "train_samples": len(train_ds),
        "train_docvqa": len(train_doc), "train_chartqa": len(train_chart),
        "effective_batch": eff_bs, "total_steps": total_steps,
        "trainable_params": n_trainable, "total_params": n_total,
        "train_time_s": train_time, "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu": torch.cuda.get_device_name(0), "log_history": trainer.state.log_history,
    }
    res_dir = REPO_ROOT / "results" / "training"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{run_name}.json").write_text(json.dumps(log, indent=1, default=str))
    print(f"[train] done in {train_time / 3600:.2f} h, peak mem {log['peak_gpu_mem_gb']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
