"""Unified DocVQA / ChartQA loaders.

Every sample is returned in the shared record format (PLAN.md, section 1):

    {"id": str, "source": "docvqa" | "chartqa", "image": PIL.Image (RGB),
     "question": str, "answers": list[str], "meta": dict}

IDs are stable and used everywhere (split files, prediction files):
    DocVQA : "docvqa-<questionId>"
    ChartQA: "chartqa-<split>-<row index>"   (ChartQA has no native id)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from datasets import load_dataset

DOCVQA_REPO = "HuggingFaceM4/DocumentVQA"
CHARTQA_REPO = "HuggingFaceM4/ChartQA"

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "configs" / "splits"

# Split names on the Hub.  DocVQA test answers are hidden -> we use "validation" as our test set.
HUB_SPLITS = {
    "docvqa": {"train": "train", "validation": "validation", "val": "validation", "test": "test"},
    "chartqa": {"train": "train", "val": "val", "validation": "val", "test": "test"},
}

# Very large scans (up to ~7000 px wide) are shrunk before the processor sees them; the models
# resize to 448/768/896 anyway, this only saves CPU time and RAM in the data loader.
MAX_IMAGE_SIDE = 2048


def _prepare_image(img):
    img = img.convert("RGB")
    if max(img.size) > MAX_IMAGE_SIDE:
        img = img.copy()
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return img


def load_raw(source: str, split: str):
    hub_split = HUB_SPLITS[source][split]
    repo = DOCVQA_REPO if source == "docvqa" else CHARTQA_REPO
    return load_dataset(repo, split=hub_split)


def read_ids(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = SPLITS_DIR / path
    return json.loads(path.read_text())


class VQADataset:
    """Map-style dataset (works with torch DataLoader / HF Trainer) yielding shared records.

    ids:          keep only these record ids (in this order)
    exclude_ids:  drop these record ids (used to remove the DocVQA dev hold-out from train)
    """

    def __init__(self, source: str, split: str, ids: Sequence[str] | None = None,
                 exclude_ids: Sequence[str] | None = None, limit: int | None = None):
        assert source in ("docvqa", "chartqa"), source
        self.source, self.split = source, HUB_SPLITS[source][split]
        self.ds = load_raw(source, split)
        all_ids = self._all_ids()
        if ids is not None:
            pos = {rid: i for i, rid in enumerate(all_ids)}
            missing = [i for i in ids if i not in pos]
            if missing:
                raise KeyError(f"{len(missing)} ids not found in {source}/{split}, e.g. {missing[:3]}")
            self.indices = [pos[i] for i in ids]
        else:
            self.indices = list(range(len(all_ids)))
        if exclude_ids:
            excl = set(exclude_ids)
            self.indices = [i for i in self.indices if all_ids[i] not in excl]
        if limit is not None:
            self.indices = self.indices[:limit]
        self._all = all_ids

    # ids are computed from cheap columns only (no image decoding)
    def _all_ids(self) -> list[str]:
        if self.source == "docvqa":
            qids = self.ds.data.column("questionId").to_pylist()
            return [f"docvqa-{q}" for q in qids]
        return [f"chartqa-{self.split}-{i}" for i in range(len(self.ds))]

    def column(self, name: str) -> list:
        """A raw column for the selected rows, without decoding images."""
        col = self.ds.data.column(name).to_pylist()
        return [col[i] for i in self.indices]

    def ids(self) -> list[str]:
        return [self._all[i] for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        ex = self.ds[idx]
        if self.source == "docvqa":
            return {
                "id": self._all[idx], "source": "docvqa", "image": _prepare_image(ex["image"]),
                "question": ex["question"].strip(), "answers": [a.strip() for a in ex["answers"]],
                "meta": {"question_types": list(ex.get("question_types") or []), "docId": ex["docId"]},
            }
        return {
            "id": self._all[idx], "source": "chartqa", "image": _prepare_image(ex["image"]),
            "question": ex["query"].strip(), "answers": [str(a).strip() for a in ex["label"]],
            # ClassLabel: 0 = human-written question, 1 = machine-generated (augmented)
            "meta": {"question_type": "human" if ex["human_or_machine"] == 0 else "augmented"},
        }


class MixedDataset:
    """Concatenation of several VQADatasets (the trainer's sampler does the shuffling)."""

    def __init__(self, parts: Sequence[VQADataset]):
        self.parts = list(parts)
        self.offsets, total = [], 0
        for p in self.parts:
            self.offsets.append(total)
            total += len(p)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, i: int) -> dict:
        for p, off in zip(reversed(self.parts), reversed(self.offsets)):
            if i >= off:
                return p[i - off]
        raise IndexError(i)
