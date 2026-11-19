# VLM-FineTune

Fine-tuning Florence-2 and PaliGemma 2 on document and chart question answering (DocVQA, ChartQA), comparing them before and after training, and deploying the best one as a web app.

See [PLAN.md](PLAN.md) for the work split, models, datasets and the shared evaluation protocol, and
[docs/FLORENCE_GUIDE.md](docs/FLORENCE_GUIDE.md) for the step-by-step Florence-2 recipe on Colab/Kaggle.

## Structure

```
configs/          training configs per model; configs/splits/ = fixed dev-subset ids (committed)
scripts/          entry points: prepare_data, evaluate, train_paligemma, run_pipeline, push_to_hub
src/data/         unified DocVQA / ChartQA loaders (shared record format)
src/models.py     model wrappers with the fixed prompts (PaliGemma 2, Florence-2)
src/training/     collators / losses
src/evaluation/   metrics (ANLS for DocVQA, relaxed accuracy for ChartQA) + tests in tests/
results/          predictions, summaries, training logs, RESULTS.md (committed, small)
app/              web demo
outputs/          checkpoints (git-ignored; adapters go to private HF Hub repos)
```

## Setup

Python 3.10+ (tested with 3.12). Install torch first, then the pinned requirements:

```
pip install torch==2.15.0 torchvision==0.29.2
pip install -r requirements.txt
```

On Colab / Kaggle: clone the repo, `pip install -r requirements.txt`, and set `HF_TOKEN` as a secret
(PaliGemma 2 is gated: accept the license on its model page first). On AWS (plain Amazon Linux 2023):
`sudo bash scripts/setup_aws.sh` then `bash scripts/install_env.sh`.

## Usage

```
python scripts/prepare_data.py                      # download datasets (split ids are already committed)
python -m pytest tests                              # metric unit tests

# evaluate any model with the shared protocol (writes results/preds/*.jsonl + results/summary/*.json)
python scripts/evaluate.py --family florence2 --stage zeroshot --dataset chartqa --split val --ids chartqa_val500_ids.json
python scripts/evaluate.py --family florence2 --stage ft --adapter <your checkpoint dir or hub id> \
    --dataset docvqa --split validation

# PaliGemma 2 LoRA training / full AWS pipeline (baseline -> train -> select -> final eval -> Hub)
python scripts/train_paligemma.py --config configs/paligemma2_448.yaml
bash scripts/run_pipeline.sh configs/paligemma2_448.yaml ft448

python scripts/make_results_table.py                # results/RESULTS.md
```

Final numbers use DocVQA `validation` (its test answers are hidden) and ChartQA `test`; the quick dev
subsets (`docvqa_dev500_ids.json` = DocVQA train questions held out by document, `chartqa_val500_ids.json`)
are for checkpoint selection. **When training, exclude the DocVQA dev hold-out from train:**
`VQADataset("docvqa", "train", exclude_ids=read_ids("docvqa_dev_ids.json"))`.
