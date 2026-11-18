"""Evaluation metrics shared by every model (see PLAN.md, section 3).

- DocVQA  -> ANLS (Average Normalized Levenshtein Similarity, threshold 0.5)
- ChartQA -> relaxed accuracy (numeric answers within 5% relative error, otherwise exact match)

Do not change these without telling the other person: every reported number depends on them.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence


# --------------------------------------------------------------------------- common
def postprocess_prediction(text: str) -> str:
    """Applied to every raw model output before scoring: strip whitespace and one trailing '.'."""
    text = text.strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    return text


def _normalize_text(s: str) -> str:
    # Same normalisation as the official DocVQA evaluation: lowercase, trim, collapse whitespace.
    return " ".join(s.strip().lower().split())


# --------------------------------------------------------------------------- ANLS
def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert / delete / substitute, all cost 1)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1,            # deletion
                               current[j - 1] + 1,         # insertion
                               previous[j - 1] + (ca != cb)))  # substitution
        previous = current
    return previous[-1]


def anls_single(prediction: str, gold_answers: Sequence[str], threshold: float = 0.5) -> float:
    """ANLS score of one prediction against all its gold answers (max over answers)."""
    pred = _normalize_text(prediction)
    best = 0.0
    for gold in gold_answers:
        g = _normalize_text(gold)
        if not pred and not g:
            return 1.0
        nl = levenshtein(pred, g) / max(len(pred), len(g))
        score = 1.0 - nl if nl < threshold else 0.0
        best = max(best, score)
    return best


def anls(predictions: Sequence[str], golds: Sequence[Sequence[str]], threshold: float = 0.5) -> float:
    assert len(predictions) == len(golds), "predictions and golds must have the same length"
    if not predictions:
        return 0.0
    return sum(anls_single(p, g, threshold) for p, g in zip(predictions, golds)) / len(predictions)


# --------------------------------------------------------------------------- relaxed accuracy
_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _to_float(text: str) -> float | None:
    """Parse '1,234.5', '45%', '$12', ' 3.0 ' as numbers; return None for anything else."""
    t = text.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    if not _NUM_RE.match(t):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def relaxed_match(prediction: str, gold: str, max_relative_error: float = 0.05) -> bool:
    p_num, g_num = _to_float(prediction), _to_float(gold)
    if p_num is not None and g_num is not None:
        if g_num == 0:
            return p_num == 0
        return abs(p_num - g_num) / abs(g_num) <= max_relative_error
    return _normalize_text(prediction) == _normalize_text(gold)


def relaxed_accuracy_single(prediction: str, gold_answers: Sequence[str],
                            max_relative_error: float = 0.05) -> float:
    return float(any(relaxed_match(prediction, g, max_relative_error) for g in gold_answers))


def relaxed_accuracy(predictions: Sequence[str], golds: Sequence[Sequence[str]],
                     max_relative_error: float = 0.05) -> float:
    assert len(predictions) == len(golds), "predictions and golds must have the same length"
    if not predictions:
        return 0.0
    return sum(relaxed_accuracy_single(p, g, max_relative_error)
               for p, g in zip(predictions, golds)) / len(predictions)


# --------------------------------------------------------------------------- per-dataset dispatch
METRIC_NAME = {"docvqa": "anls", "chartqa": "relaxed_accuracy"}


def score_single(source: str, prediction: str, gold_answers: Sequence[str]) -> float:
    if source == "docvqa":
        return anls_single(prediction, gold_answers)
    if source == "chartqa":
        return relaxed_accuracy_single(prediction, gold_answers)
    raise ValueError(f"unknown source {source!r}")


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
