"""Export results/ into the JSON files the web app reads (app/data/).

    python scripts/export_web_data.py

Re-run after new evaluation runs. metrics.json gives the mock latencies; examples.json feeds the
Reference samples (images in app/public/samples/<id>.jpg) and their recorded answers.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "app" / "data"

FAMILIES = ["paligemma2", "florence2"]
STAGES = {"zeroshot": "base", "ft448": "finetuned", "ft": "finetuned"}
FINAL = {("docvqa", "validation"), ("chartqa", "test")}

EXAMPLES = [
    "docvqa-57376", "docvqa-57395", "docvqa-16429",
    "chartqa-test-58", "chartqa-test-114", "chartqa-test-1",
]


def r(x: float, n: int = 6) -> float:
    return round(x, n)


def export_metrics() -> dict:
    fams: dict = {}
    type_counts: dict = {}
    for p in sorted((RESULTS / "summary").glob("*.json")):
        s = json.loads(p.read_text())
        if (s["dataset"], s["split"]) not in FINAL or s.get("ids_file") or s["stage"] not in STAGES:
            continue
        m = fams.setdefault(s["family"], {}).setdefault(STAGES[s["stage"]], {"latency": []})
        m["latency"].append(s["latency_s_per_sample"])
        if s["dataset"] == "docvqa":
            m["docvqa"] = r(s["score"])
            m["docvqaByType"] = {k: r(v["score"]) for k, v in s["by_category"].items()}
            type_counts = {k: v["n"] for k, v in s["by_category"].items()}
        else:
            m["chartqa"] = r(s["score"])
            m["chartqaHuman"] = r(s["by_category"]["human"]["score"])
            m["chartqaAugmented"] = r(s["by_category"]["augmented"]["score"])
    for stages in fams.values():
        for m in stages.values():
            lat = m.pop("latency")
            m["latencyS"] = r(sum(lat) / len(lat), 3)
    return {"families": fams, "docvqaTypeCounts": type_counts}


def export_examples() -> list[dict]:
    preds: dict = {}
    for fam in FAMILIES:
        for stage, key in STAGES.items():
            for ds, split in FINAL:
                f = RESULTS / "preds" / f"{fam}_{stage}_{ds}_{split}.jsonl"
                if not f.exists():
                    continue
                for line in f.read_text(encoding="utf-8").splitlines():
                    rec = json.loads(line)
                    if rec["id"] in EXAMPLES:
                        preds.setdefault(rec["id"], {"rec": rec, "p": {}})
                        preds[rec["id"]]["p"].setdefault(fam, {})[key] = {
                            "answer": rec["prediction"], "score": r(rec["score"], 3)}
    out = []
    for i in EXAMPLES:
        rec, p = preds[i]["rec"], preds[i]["p"]
        meta = rec["meta"]
        out.append({
            "id": i,
            "dataset": "docvqa" if i.startswith("docvqa") else "chartqa",
            "category": (meta.get("question_types") or [meta.get("question_type")])[0],
            "image": f"/samples/{i}.jpg",
            "question": rec["question"],
            "answers": rec["answers"],
            "predictions": p,
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in [("metrics", export_metrics()), ("examples", export_examples())]:
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", OUT / f"{name}.json")


if __name__ == "__main__":
    main()
