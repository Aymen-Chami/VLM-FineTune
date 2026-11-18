"""Download both datasets and write the fixed split-ID files to configs/splits/.

    python scripts/prepare_data.py

Creates (seed 42, committed to git so both of us use the exact same samples):
    docvqa_dev_ids.json      ~1000 DocVQA *train* questions held out by docId (model selection)
    docvqa_dev500_ids.json   500 of those (quick checks during training)
    chartqa_val500_ids.json  500 ChartQA val questions (quick checks during training)
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loaders import SPLITS_DIR, VQADataset  # noqa: E402

SEED = 42
DEV_MIN_QUESTIONS = 1000


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- DocVQA: hold out whole documents from train
    train = VQADataset("docvqa", "train")
    ids, doc_ids = train.ids(), train.column("docId")
    by_doc: dict[int, list[str]] = {}
    for rid, d in zip(ids, doc_ids):
        by_doc.setdefault(d, []).append(rid)
    docs = sorted(by_doc)
    random.Random(SEED).shuffle(docs)
    dev: list[str] = []
    dev_docs: list[int] = []
    for d in docs:
        if len(dev) >= DEV_MIN_QUESTIONS:
            break
        dev.extend(by_doc[d])
        dev_docs.append(d)
    dev500 = random.Random(SEED).sample(dev, 500)
    (SPLITS_DIR / "docvqa_dev_ids.json").write_text(json.dumps(dev, indent=0))
    (SPLITS_DIR / "docvqa_dev500_ids.json").write_text(json.dumps(dev500, indent=0))
    print(f"DocVQA train: {len(ids)} questions / {len(by_doc)} docs -> dev hold-out "
          f"{len(dev)} questions from {len(dev_docs)} docs")

    val = VQADataset("docvqa", "validation")
    types = Counter(t for ts in val.column("question_types") for t in ts)
    print(f"DocVQA validation (our test set): {len(val)} questions; question types: {dict(types)}")

    # ---- ChartQA
    for split in ("train", "val", "test"):
        ds = VQADataset("chartqa", split)
        hm = Counter(ds.column("human_or_machine"))
        print(f"ChartQA {split}: {len(ds)} (human_or_machine counts: {dict(hm)}); "
              f"features: {ds.ds.features['human_or_machine']}")
    cval = VQADataset("chartqa", "val")
    val500 = random.Random(SEED).sample(cval.ids(), 500)
    (SPLITS_DIR / "chartqa_val500_ids.json").write_text(json.dumps(val500, indent=0))

    # sanity: one decoded record per dataset
    for ds in (train, cval):
        r = ds[0]
        print({k: (v if k != "image" else v.size) for k, v in r.items()})


if __name__ == "__main__":
    main()
