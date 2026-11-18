# Project plan — VLM fine-tuning on DocVQA & ChartQA

Two people, two compute setups, one shared evaluation protocol.

| | **Person A (AWS)** | **Person B (Colab / Kaggle)** |
|---|---|---|
| Main model | **PaliGemma 2 3B** (`google/paligemma2-3b-pt-448`), LoRA | **Florence-2** (`florence-community/Florence-2-base-ft` → `-large-ft`), full fine-tune with frozen vision encoder |
| Optional 2nd run | PaliGemma 2 at **896 px** (resolution ablation) | Florence-2-large if base was done first |
| Shared code owned | `src/data/` (loaders, splits), `scripts/evaluate.py` (eval runner), `app/` | `src/evaluation/metrics.py` + tests |
| Why this split | 3B model needs bf16 + ~24 GB GPU → AWS | Florence-2 is 0.23B / 0.77B → fits a T4 (16 GB) |

> **Assumption:** the AWS instance has a 24 GB GPU (g5.xlarge / A10G or g6.xlarge / L4). If it's a T4 (g4dn), switch PaliGemma to QLoRA (4-bit) and fp16, and skip the 896 px run.

---

## 1. Datasets

We train **one model per architecture on DocVQA + ChartQA mixed together**, then evaluate it on each dataset separately. That halves the number of runs, and the demo app doesn't need to know which dataset a question comes from.

### DocVQA — `HuggingFaceM4/DocumentVQA`
- Splits: train 39.5k / validation 5.35k / test 5.19k. **The test answers are hidden** (scored only by the official server), so we treat **validation as our test set**.
- Fields: `questionId, question, question_types, image, docId, answers (list)`.
- **Model-selection split:** hold out ~1,000 questions from train, **split by `docId`** so no document shows up in both train and dev. Save the IDs to `configs/splits/docvqa_dev_ids.json` and commit the file.
- Images can be up to ~7,000 px wide. Pre-resize them (e.g. longest side 2,048) when caching so data loading stays fast.

### ChartQA — `HuggingFaceM4/ChartQA`
- Splits: train 28.3k / val 1.92k / test 2.5k.
- Fields: `image, query, label (list of 1 string), human_or_machine` (0 = human-written, 1 = augmented).
- Use val for model selection and test for the final numbers. **Report human, augmented, and the overall average.** Human questions are much harder, and that's where the interesting gap shows up.

### Shared record format (A writes `src/data/`, everyone uses it)
```python
{"id": "docvqa-12345", "source": "docvqa" | "chartqa", "image": PIL.Image,
 "question": str, "answers": list[str], "meta": {"question_types": [...] | "human_or_machine": 0/1}}
```

### Fixed eval subsets
- **Quick dev subsets** (used during training): 500 DocVQA dev-holdout + 500 ChartQA val, seed 42. Commit the IDs to `configs/splits/`.
- **Final eval:** full DocVQA validation (5,349) + full ChartQA test (2,500).

---

## 2. Models — exact setup

### Prompt format (fixed, identical for both datasets)
| Model | Input | Target |
|---|---|---|
| Florence-2 | `"<VQA>" + question` | answer |
| PaliGemma 2 | `"<image>answer en " + question` | answer (passed via `suffix=` to the processor, which masks the prompt out of the labels) |

Training target = **the first answer** in `answers`. Evaluation compares against **all** gold answers.

### Florence-2 (Person B)
- Load with `Florence2ForConditionalGeneration` / `AutoProcessor` (native in transformers v5, so no `trust_remote_code`).
- Step 1: `Florence-2-base-ft` (0.23B). It's fast, so use it to debug the whole pipeline. Step 2: `Florence-2-large-ft` (0.77B) if time allows.
- **Freeze the vision encoder** (DaViT). Train the rest fully.
- Precision: T4 and P100 have **no bf16** → fp32 weights + fp16 autocast with GradScaler. Fall back to pure fp32 if the loss goes NaN.
- Optimizer: AdamW, **lr 1e-6 to start** (Florence-2 is sensitive; try 5e-6 if learning is too slow), 5% warmup, linear decay, weight decay 0.01.
- Batch: 2–4 per step + gradient accumulation → effective batch 16. Turn on gradient checkpointing for large.
- Input resolution is fixed at 768×768, so small text in documents gets blurred. Expect weaker DocVQA results than PaliGemma; this is a point for the analysis.
- **Session limits:** Kaggle allows 12 h per session and ~30 GPU-h per week, and Colab free tier disconnects. Save a checkpoint every ~1,000 steps and **push it to a private HF Hub repo** so a run can resume in a new session.

### PaliGemma 2 (Person A)
- `google/paligemma2-3b-pt-448`. **Gated:** both of you need to accept the license on the model page, and you need an HF token.
- Use the **`pt`** checkpoint, **not `mix`**. The mix checkpoints were already fine-tuned on downstream tasks, so they would make our "before vs after" comparison meaningless.
- Load in **bf16**. LoRA on the Gemma 2 decoder: `r=16, alpha=32, dropout=0.05`, targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. Vision tower (SigLIP) and projector stay frozen.
- Optimizer: AdamW, lr 1e-4, 3% warmup, cosine decay. Batch 4 × grad accumulation 4 = 16. Gradient checkpointing on. 1–2 epochs over the mixed train set.
- If Gemma 2 prints attention warnings during training, use `attn_implementation="eager"`.
- **Optional 2nd run:** `paligemma2-3b-pt-896` with the same recipe, batch 1 × accumulation 16 (QLoRA if memory runs out). This tests whether resolution matters for documents, which is a good research question for the report.
  - Alternative, if you'd rather compare a third model family: a recent small general VLM (e.g. Qwen2.5-VL-3B). Its pretraining already includes DocVQA and ChartQA, though, so report it only as a reference point, not as a fair before/after comparison.

### Before running any full training (both of you)
1. **Overfit 50–100 samples.** Loss should go close to 0 and the model should reproduce the answers. If it can't, there's a bug in labels, masking, or the prompt.
2. Measure throughput (samples/s), then choose how much data fits your budget. If Florence is too slow on Colab/Kaggle, subsample to e.g. 20k DocVQA + 20k ChartQA.

---

## 3. Evaluation protocol (the key part: it must be identical for every model)

### Metrics (Person B writes `src/evaluation/metrics.py`, with unit tests)
- **DocVQA → ANLS.** For each question: lowercase and strip both strings, compute `NL = levenshtein(pred, gold) / max(len(pred), len(gold))`, score `1 - NL` if `NL < 0.5` else `0`. Take the max over gold answers, then the mean over questions.
- **ChartQA → relaxed accuracy.** Strip `%`, `,` and `$`. If both prediction and gold parse as numbers, it's correct when `|pred - gold| / |gold| ≤ 0.05` (exact match if gold is 0). Otherwise, compare lowercased strings for an exact match.
- Unit tests should cover: exact match, case and whitespace differences, the 0.5 threshold, multiple gold answers, 4.9% vs 5.1% numeric error, percentages, and gold = 0.

### Generation settings (fixed)
Greedy decoding (`num_beams=1`, `do_sample=False`), `max_new_tokens=64`. Post-processing: strip whitespace and a trailing `.`.

### Eval runner (Person A writes `scripts/evaluate.py`)
```
python scripts/evaluate.py --model paligemma2 --checkpoint <hub id or path> \
    --dataset docvqa --split val [--limit 500] [--ids configs/splits/...json]
```
Writes:
- `results/preds/{model}_{stage}_{dataset}.jsonl`, one line per sample: `{"id", "question", "prediction", "answers", "meta"}`
- `results/summary/{model}_{stage}_{dataset}.json`: score, per-category scores, n, latency, GPU, date

Commit `results/` to git (it's small); checkpoints stay in `outputs/` and on the HF Hub.

### Models to evaluate
| Stage | Florence-2 | PaliGemma 2 |
|---|---|---|
| **Zero-shot baseline** (before) | `-base-ft` / `-large-ft` with the same `<VQA>` prompt | `pt-448` with the same `answer en` prompt |
| **Fine-tuned** (after) | our checkpoint(s) | our LoRA (448, and 896 if done) |
| Reference only (optional) | — | `paligemma2-3b-mix-448` (already trained on these tasks: an upper-bound reference) |

Zero-shot scores for `pt` models will be very low. That's expected and still the honest "before" number.

### What to report
1. **Main table:** DocVQA ANLS, ChartQA relaxed accuracy (human / augmented / overall) for every row above.
2. **Breakdowns:** DocVQA by `question_types` (form, table/list, handwritten, figure, layout, yes/no…). ChartQA human vs augmented.
3. **Efficiency:** total / trainable params, peak GPU memory, training time, inference latency per sample. **Run the final eval of all models on the same AWS GPU** so latency numbers are comparable.
4. **Error analysis:** manually label ~50 failures per model (OCR misread, wrong region, numeric reasoning, format mismatch such as "5%" vs "5", gold-label issue) and show qualitative examples.
5. **Training curves:** train loss and dev score per checkpoint. Use a shared W&B project (free for students) or at least log to JSON.

---

## 4. Timeline (assumes ~5 weeks; compress phases if your deadline is sooner)

| Phase | Person A (AWS) | Person B (Colab/Kaggle) | Done when… |
|---|---|---|---|
| **0. Setup** (days 1–3) | Launch instance (Deep Learning AMI, ~150 GB EBS, `HF_HOME` on that volume), billing alert. Accept the PaliGemma license. **Pin exact package versions** in `requirements.txt` (at least `transformers`, `peft`, `torch`). | HF account + token as a Colab/Kaggle secret. Accept the PaliGemma license. Notebook template: clone repo → `pip install -r requirements.txt` → import `src`. | Both environments run the same `transformers` version and can load their model. |
| **1. Pipeline** (week 1) | `src/data/` loaders + split ID files. `scripts/evaluate.py`. | `metrics.py` + tests (CPU only, can start day 1). Florence-2 collate/training loop. | Zero-shot baselines on the 500+500 dev subsets for both models, results in `results/`. |
| **2. Training** (weeks 2–3) | Overfit test → full PaliGemma 448 LoRA run (one night or so). | Overfit test → Florence-2-base full run → large. | Each model clearly beats its baseline on the dev subsets. |
| **3. Final eval + ablation** (week 3–4) | 896 px run (optional). **Final eval of all checkpoints** (including Florence's, pulled from the Hub) on the full splits. | Pick best Florence checkpoint using dev-set scores. Start error analysis. | Main table complete. |
| **4. Analysis + app + report** (weeks 4–5) | Gradio app in `app/` serving the best model (Florence-2 can run on CPU; PaliGemma needs a GPU). | Breakdowns, error analysis, figures. | Report + demo. |

---

## 5. Collaboration rules
- **Day 1 contract:** agree on the record format, prompts, prediction JSONL format and metric code above *before* writing training code. Changing them later means re-running evals.
- Nobody changes `metrics.py`, prompts, or split ID files without telling the other person. Same for generation settings.
- Work on branches, open small PRs, and let the other person review the shared code (`src/data`, `src/evaluation`).
- Checkpoints and adapters go to **private HF Hub repos** (e.g. `<user>/florence2-docchart`, `<user>/paligemma2-docchart-lora`) so either of you can evaluate or demo any model.
- AWS costs: run training in `tmux`, **stop the instance when idle**, and use spot instances only if checkpoint-resume works.

## 6. Risks & fallbacks
| Risk | Fallback |
|---|---|
| Colab/Kaggle session dies mid-training | Checkpoint to the Hub every ~1k steps and resume from the last one. |
| PaliGemma out of memory | Batch size 1 + more accumulation → QLoRA 4-bit → stay at 448 px. |
| Florence-2-large too slow on a T4 | Stick with base and train on a data subsample. |
| Fine-tuned model outputs long sentences instead of short answers | Check that labels mask the prompt, and that training targets are the short gold answers. |
| Time runs short | Drop the 896 ablation and Florence-large. The core deliverable is 2 models × 2 datasets × before/after. |
