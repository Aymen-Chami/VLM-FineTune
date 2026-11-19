"""Fine-tuning Florence-2 on DocVQA + ChartQA (mixed), built for Colab / Kaggle session limits.

    python scripts/train_florence.py --config configs/florence2_base.yaml
    # overfit sanity check first (must reach ~0 loss, then evaluate on outputs/overfit_florence/*_ids.json)
    python scripts/train_florence.py --config configs/florence2_base.yaml --run-name overfit_florence \
        --limit-train 64 --epochs 15 --lr 1e-5 --save-steps 0 --no-eval
    # Colab/Kaggle: push every checkpoint (incl. optimizer state) to a private Hub repo ...
    python scripts/train_florence.py --config configs/florence2_base.yaml --hub-repo florence2-base-docchart
    # ... and continue in a NEW session from the last pushed checkpoint
    python scripts/train_florence.py --config configs/florence2_base.yaml --hub-repo florence2-base-docchart --resume-from-hub

Outputs: <output_dir>/checkpoint-*/ (full model checkpoints), <output_dir>/final/,
results/training/<run_name>.json (loss curves, config, timing).
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
from src.training.collators import FlorenceTrainCollator  # noqa: E402


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
    ap.add_argument("--save-steps", type=int, default=None, help="0 = only save the final model")
    ap.add_argument("--precision", default=None, choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--resume", action="store_true", help="resume from the latest local checkpoint")
    ap.add_argument("--hub-repo", default=None, help="private Hub repo to push checkpoints to (name or user/name)")
    ap.add_argument("--resume-from-hub", action="store_true", help="download last-checkpoint from --hub-repo and resume")
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
    precision = args.precision or cfg.get("precision", "fp16")

    from transformers import AutoProcessor, Florence2ForConditionalGeneration, Trainer, TrainingArguments

    # ------------------------------------------------------------------ data (identical to PaliGemma)
    dev_ids = read_ids("docvqa_dev_ids.json")
    doc_lim, chart_lim = d["docvqa_limit"], d["chartqa_limit"]
    if args.limit_train:
        doc_lim = chart_lim = args.limit_train // 2
    train_doc = VQADataset("docvqa", "train", exclude_ids=dev_ids, limit=doc_lim)
    train_chart = VQADataset("chartqa", "train", limit=chart_lim)
    train_ds = MixedDataset([train_doc, train_chart])
    if args.limit_train:
        (out_dir / "docvqa_ids.json").write_text(json.dumps(train_doc.ids()))
        (out_dir / "chartqa_ids.json").write_text(json.dumps(train_chart.ids()))
    eval_ds = None
    if not args.no_eval:
        n = d["eval_loss_samples"]
        eval_ds = MixedDataset([
            VQADataset("docvqa", "train", ids=read_ids("docvqa_dev500_ids.json")[:n]),
            VQADataset("chartqa", "val", ids=read_ids("chartqa_val500_ids.json")[:n]),
        ])
    print(f"[train] {run_name}: train={len(train_ds)} (docvqa {len(train_doc)}, chartqa {len(train_chart)}), "
          f"eval_loss={len(eval_ds) if eval_ds else 0}, precision={precision}", flush=True)

    # ------------------------------------------------------------------ model
    processor = AutoProcessor.from_pretrained(cfg["base"])
    # fp32 master weights; fp16/bf16 only through autocast (Trainer's fp16/bf16 flags)
    model = Florence2ForConditionalGeneration.from_pretrained(cfg["base"], dtype=torch.float32)
    if cfg.get("freeze_vision", True):
        for p in model.model.vision_tower.parameters():
            p.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable params {n_trainable / 1e6:.1f}M / {n_total / 1e6:.1f}M", flush=True)

    # ------------------------------------------------------------------ trainer
    eff_bs = t["per_device_batch_size"] * t["grad_accum"]
    steps_per_epoch = math.ceil(len(train_ds) / eff_bs)
    total_steps = t["max_steps"] if t["max_steps"] and t["max_steps"] > 0 else math.ceil(steps_per_epoch * t["epochs"])
    save_steps = t["save_steps"]
    hub = {}
    if args.hub_repo:
        from huggingface_hub import HfApi
        repo_id = args.hub_repo if "/" in args.hub_repo else f"{HfApi().whoami()['name']}/{args.hub_repo}"
        # "checkpoint": every save is pushed, and the latest one (with optimizer state) also goes to
        # last-checkpoint/ so a new Colab/Kaggle session can resume from it.
        hub = dict(push_to_hub=True, hub_model_id=repo_id, hub_strategy="checkpoint", hub_private_repo=True)
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
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
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
        prediction_loss_only=True,
        report_to=["tensorboard"],
        seed=t["seed"],
        **hub,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=FlorenceTrainCollator(processor))

    resume = None
    if args.resume_from_hub:
        assert args.hub_repo, "--resume-from-hub needs --hub-repo"
        from huggingface_hub import snapshot_download
        snapshot_download(hub["hub_model_id"], allow_patterns=["last-checkpoint/*"], local_dir=str(out_dir))
        resume = str(out_dir / "last-checkpoint")
        print(f"[train] resuming from {resume}", flush=True)
    elif args.resume and any(out_dir.glob("checkpoint-*")):
        resume = True

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    train_time = time.time() - t0
    trainer.save_model(str(out_dir / "final"))
    if args.hub_repo:
        trainer.push_to_hub(commit_message="final model")

    log = {
        "run_name": run_name, "config": cfg, "precision": precision, "train_samples": len(train_ds),
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
