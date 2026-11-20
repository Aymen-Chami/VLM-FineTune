# Florence-2 on Colab / Kaggle: the same steps as the PaliGemma run

This is the exact recipe used for PaliGemma 2 on AWS (branch `aws-paligemma`), adapted to Florence-2 and to
Colab/Kaggle. Same data, same splits, same metrics and the same evaluation script, so our numbers are directly
comparable. Every command is meant to be run from the repo root, in a notebook cell with `!` in front.

The PaliGemma and Florence-2 results (both run on AWS with this pipeline) are in [results/RESULTS.md](../results/RESULTS.md).

---

## 0. What's already done (don't redo)

| Done | Where |
|---|---|
| Metrics (ANLS, ChartQA relaxed accuracy) + 12 unit tests | `src/evaluation/metrics.py`, `tests/` |
| Data loaders (shared record format) | `src/data/loaders.py` |
| **Fixed split IDs** (DocVQA dev hold-out by document, 500-question dev subsets) | `configs/splits/` (committed: use these, don't regenerate) |
| Prompts + generation for both models (`<VQA>{question}` for Florence) | `src/models.py` |
| Evaluation script (writes `results/preds/*.jsonl` and `results/summary/*.json`) | `scripts/evaluate.py` |
| Florence-2 training script + config: **run end to end on the AWS A10G** (bf16, `configs/florence2_base_aws.yaml`, 1 epoch in 48 min). The results are in `results/RESULTS.md`, and the weights are in the private HF repo `asdsssss27/florence2-base-docchart-ft`. On Colab/Kaggle, only the precision (fp16) and `num_workers` differ. | `scripts/train_florence.py`, `configs/florence2_base.yaml` |

Please review `src/evaluation/metrics.py`: it was planned as your part, and I wrote it to unblock the baseline.

---

## 1. Environment

```python
# Colab: Runtime -> Change runtime type -> T4 GPU.   Kaggle: Settings -> Accelerator -> GPU T4 x2, Internet on.
!git clone -b aws-paligemma https://github.com/Aymen-Chami/VLM-FineTune.git
%cd VLM-FineTune
!pip install -q transformers==5.23.0 peft==0.21.3 accelerate==1.16.0 datasets==5.2.0 huggingface_hub==2.4.0 pyyaml tensorboard pytest
!python -m pytest -q tests          # 12 passed
```

- **Keep the torch that Colab/Kaggle ships.** Do *not* `pip install -r requirements.txt` there: it pins
  `torch==2.15.0`, whose default wheel is built for CUDA 13 and needs a newer NVIDIA driver than Colab/Kaggle
  may have. Check with `import torch; torch.cuda.is_available()`.
- **HF token:** add a secret `HF_TOKEN` (Colab: key icon on the left; Kaggle: Add-ons → Secrets) with *write*
  access (needed to push checkpoints), then:
  ```python
  import os
  from google.colab import userdata; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")      # Colab
  # from kaggle_secrets import UserSecretsClient; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")  # Kaggle
  ```
  Florence-2 is not gated; the token is for pushing your checkpoints.
- **Disk:** the DocVQA cache takes **~34 GB** and ChartQA ~1.4 GB. Colab's T4 runtime has enough space. On Kaggle,
  run `!df -h` first. Download into the big temporary disk, not `/kaggle/working` (20 GB limit):
  `os.environ["HF_HOME"] = "/kaggle/tmp/hf"` (or `/tmp/hf`) **before** importing `datasets`.

## 2. Download the data (~10 min)

```python
!python scripts/prepare_data.py
!git status configs/splits      # must show NO changes: the split files are identical to the committed ones
```

If `git status` shows the split files changed, stop and tell me. Our dev sets would then differ.

## 3. Baseline ("before") — zero-shot Florence-2

```python
for ds, split, ids in [("docvqa", "train", "--ids docvqa_dev500_ids.json"),
                       ("chartqa", "val", "--ids chartqa_val500_ids.json"),
                       ("docvqa", "validation", ""), ("chartqa", "test", "")]:
    !python scripts/evaluate.py --family florence2 --stage zeroshot --dataset {ds} --split {split} {ids}
```

Expect near-zero scores. The base model has no VQA task token, so it outputs things like `"vQA"`. That is still our
honest "before" number. (PaliGemma pt-448 scored 3.0 DocVQA ANLS and 4.8 ChartQA before fine-tuning.)

## 4. Sanity check: overfit 64 samples (~5 min). Don't skip this.

```python
!python scripts/train_florence.py --config configs/florence2_base.yaml --run-name overfit_florence \
    --limit-train 64 --epochs 15 --lr 1e-5 --save-steps 0 --no-eval
!python scripts/evaluate.py --family florence2 --stage overfit --adapter outputs/overfit_florence/final \
    --dataset docvqa --split train --ids outputs/overfit_florence/docvqa_ids.json --out-dir outputs/overfit_florence/results
!python scripts/evaluate.py --family florence2 --stage overfit --adapter outputs/overfit_florence/final \
    --dataset chartqa --split train --ids outputs/overfit_florence/chartqa_ids.json --out-dir outputs/overfit_florence/results
```

The loss should go close to 0, and both scores should be close to 1.0 (PaliGemma got 0.999 / 1.0). If not, something
is wrong in the labels or prompt. Fix it before spending GPU hours.

## 5. Full training

```python
!python scripts/train_florence.py --config configs/florence2_base.yaml --hub-repo florence2-base-docchart
```

- **Data:** 66,762 training questions (all of DocVQA train *minus the dev hold-out*, plus all of ChartQA
  train), mixed, 1 epoch, effective batch 16 → 4,173 steps. This is exactly the PaliGemma setup.
- **Recipe:**
  - Vision tower (DaViT, 90M) frozen; projector + language model (141M) fully trained.
  - AdamW, lr 5e-6, linear decay, 3% warmup.
  - fp16 mixed precision (T4/P100 have no bf16). If the loss becomes `nan`, rerun with `--precision fp32`.
- **During training:** the loss on held-out questions is logged every 500 steps, and a checkpoint is saved every 1,000 steps.
  The training loss drops fast in the first ~200 steps, then slowly. For PaliGemma it went 1.0 → 0.6 → 0.5,
  and the held-out loss went 0.56 → 0.48 by mid-training.
- **Session limits:** Kaggle allows 12 h per session, and Colab disconnects. `--hub-repo` pushes every checkpoint, *including the
  optimizer state*, to a private repo `<your-user>/florence2-base-docchart`. In a new session, redo steps 1–2,
  then continue exactly where it stopped:
  ```python
  !python scripts/train_florence.py --config configs/florence2_base.yaml --hub-repo florence2-base-docchart --resume-from-hub
  ```
- **Before the long run:** check speed on 30 steps (`--max-steps 30 --save-steps 0 --no-eval`). Multiply the
  s/step you see by 4,173 to get the total time. If it doesn't fit your GPU budget, reduce the data with
  `--limit-train N` (e.g. 40000), and use the same N for any comparison.
- **Florence-2-large** (if time allows): copy the config, set `base: florence-community/Florence-2-large-ft`,
  `gradient_checkpointing: true`, batch 4 × accumulation 4, `run_name`/`output_dir` → `florence2_large`.

## 6. Pick the best checkpoint (on the dev subsets, never on the test sets)

```python
import glob
for ck in sorted(glob.glob("outputs/florence2_base/checkpoint-*")) + ["outputs/florence2_base/final"]:
    tag = ck.split("/")[-1]
    !python scripts/evaluate.py --family florence2 --stage ft --adapter {ck} --tag {tag} --dataset docvqa --split train --ids docvqa_dev500_ids.json
    !python scripts/evaluate.py --family florence2 --stage ft --adapter {ck} --tag {tag} --dataset chartqa --split val --ids chartqa_val500_ids.json
```

Pick the checkpoint with the best **average of dev ANLS and dev relaxed accuracy**. That's the rule used for PaliGemma.

## 7. Final evaluation (once, on the best checkpoint)

```python
BEST = "outputs/florence2_base/checkpoint-XXXX"
!python scripts/evaluate.py --family florence2 --stage ft --adapter {BEST} --dataset docvqa --split validation
!python scripts/evaluate.py --family florence2 --stage ft --adapter {BEST} --dataset chartqa --split test
!python scripts/push_to_hub.py --folder {BEST} --repo florence2-base-docchart-best
```

Also send me the Hub id of the best checkpoint. The plan is to re-run the final evaluation of *all* models on
the same AWS GPU so the latency numbers are comparable. The accuracy should match yours.

## 8. Share the results

```python
!python scripts/make_results_table.py
!git add results && git commit -m "Florence-2 base: baseline, training log, results" && git push
```

`results/preds/*.jsonl` holds every prediction. We'll use those for the error analysis.

---

## Pitfalls we hit on the PaliGemma run (so you don't)

| Problem | What happened / fix |
|---|---|
| Out of memory at batch 4 | Full logits (batch × 1,050 tokens × 257k vocab) were 4 GB. Fixed by computing logits only for the answer tokens. **Florence doesn't have this problem**: its decoder only sees the answer. |
| Batched generation must give the same answers as one-by-one | We checked it on 8 samples: same outputs. Already handled in `src/models.py`. |
| Session / connection drops | All long jobs ran detached and resumable. On Colab/Kaggle that means `--hub-repo` + `--resume-from-hub`. |
| Bigger batch ≠ faster | When the GPU is already at 100%, batch 8 was no faster than batch 4. Keep effective batch 16 and pick the per-device size that fits. |
| Dev vs test | Dev subsets for choosing checkpoints, the full DocVQA validation / ChartQA test only for the final number. DocVQA test answers are hidden. |
