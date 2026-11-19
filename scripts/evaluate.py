"""Evaluate one model on one dataset split with the shared protocol (PLAN.md, section 3).

Examples
    # zero-shot baseline on the quick dev subsets
    python scripts/evaluate.py --family paligemma2 --stage zeroshot --dataset docvqa --split train \
        --ids docvqa_dev500_ids.json
    # fine-tuned LoRA on the full DocVQA validation set (our test set)
    python scripts/evaluate.py --family paligemma2 --stage ft448 --adapter outputs/pg2_448/final \
        --dataset docvqa --split validation

Writes
    results/preds/<name>.jsonl     one line per sample: id, question, prediction, answers, score, meta
    results/summary/<name>.json    overall score, per-category scores, n, latency, GPU, ...
where <name> = <family>_<stage>_<dataset>_<split>[_<tag>].
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loaders import REPO_ROOT, VQADataset, read_ids  # noqa: E402
from src.evaluation.metrics import METRIC_NAME, mean, score_single  # noqa: E402
from src.models import load_model  # noqa: E402

DEFAULT_BASE = {"paligemma2": "google/paligemma2-3b-pt-448",
                "florence2": "florence-community/Florence-2-base-ft"}
DEFAULT_DTYPE = {"paligemma2": "bf16", "florence2": "fp16"}
# Gemma 2 uses attention logit soft-capping, which only the eager implementation applies.
DEFAULT_ATTN = {"paligemma2": "eager", "florence2": None}


def category_scores(source: str, rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if source == "docvqa":
            for t in r["meta"].get("question_types") or ["(none)"]:
                groups[t].append(r["score"])
        else:
            groups[r["meta"]["question_type"]].append(r["score"])
    return {k: {"score": mean(v), "n": len(v)} for k, v in sorted(groups.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["paligemma2", "florence2"])
    ap.add_argument("--base", default=None, help="base checkpoint (default depends on family)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter (PaliGemma) or fine-tuned checkpoint (Florence)")
    ap.add_argument("--stage", required=True, help="e.g. zeroshot, ft448, ft896, mix")
    ap.add_argument("--dataset", required=True, choices=["docvqa", "chartqa"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--ids", default=None, help="json list of record ids (file in configs/splits/ or path)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--attn", default=None, help="attn_implementation (e.g. eager, sdpa)")
    ap.add_argument("--tag", default=None, help="suffix for output file names (e.g. ckpt-1500)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    base = args.base or DEFAULT_BASE[args.family]
    dtype = args.dtype or DEFAULT_DTYPE[args.family]
    attn = args.attn or DEFAULT_ATTN[args.family]
    ids = read_ids(args.ids) if args.ids else None
    ds = VQADataset(args.dataset, args.split, ids=ids, limit=args.limit)

    name = f"{args.family}_{args.stage}_{args.dataset}_{ds.split}" + (f"_{args.tag}" if args.tag else "")
    if args.ids and not args.tag:
        name += "_" + Path(args.ids).stem.replace("_ids", "")
    out_dir = Path(args.out_dir)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "preds" / f"{name}.jsonl"
    print(f"[eval] {name}: {len(ds)} samples, base={base}, adapter={args.adapter}, dtype={dtype}", flush=True)

    model = load_model(args.family, base=base, adapter=args.adapter, dtype=dtype, attn_implementation=attn)
    torch.cuda.reset_peak_memory_stats()

    rows: list[dict] = []
    gen_seconds = 0.0
    t_start = time.time()
    with pred_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(ds), args.batch_size):
            batch = [ds[i] for i in range(start, min(start + args.batch_size, len(ds)))]
            torch.cuda.synchronize()
            t0 = time.time()
            preds = model.predict(batch, max_new_tokens=args.max_new_tokens)
            torch.cuda.synchronize()
            gen_seconds += time.time() - t0
            for r, p in zip(batch, preds):
                row = {"id": r["id"], "question": r["question"], "prediction": p, "answers": r["answers"],
                       "score": score_single(args.dataset, p, r["answers"]), "meta": r["meta"]}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (start // args.batch_size) % 20 == 0:
                print(f"  {len(rows)}/{len(ds)}  running {METRIC_NAME[args.dataset]}="
                      f"{mean(r['score'] for r in rows):.4f}  ({time.time() - t_start:.0f}s)", flush=True)

    summary = {
        "name": name, "family": args.family, "stage": args.stage, "base": base, "adapter": args.adapter,
        "dataset": args.dataset, "split": ds.split, "ids_file": args.ids, "tag": args.tag, "n": len(rows),
        "metric": METRIC_NAME[args.dataset], "score": mean(r["score"] for r in rows),
        "by_category": category_scores(args.dataset, rows),
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens,
                       "batch_size": args.batch_size, "dtype": dtype, "attn": attn},
        "latency_s_per_sample": gen_seconds / max(len(rows), 1),
        "wall_time_s": time.time() - t_start,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu": torch.cuda.get_device_name(0),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "summary" / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(f"[eval] {name}: {summary['metric']} = {summary['score']:.4f}  (n={len(rows)})")
    for k, v in summary["by_category"].items():
        print(f"    {k:<22} {v['score']:.4f}  (n={v['n']})")


if __name__ == "__main__":
    main()
