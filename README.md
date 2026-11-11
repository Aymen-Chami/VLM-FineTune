# VLM-FineTune

Fine-tuning Florence-2 and PaliGemma 2 on document and chart question answering (DocVQA, ChartQA), comparing them before and after training, and deploying the best one as a web app.

## Structure

```
configs/          training/eval configs per model
data/             downloaded datasets (git-ignored)
notebooks/        exploration and quick experiments
scripts/          entry points (train, evaluate, download data)
src/data/         dataset loading and preprocessing
src/training/     fine-tuning code
src/evaluation/   metrics (ANLS for DocVQA, relaxed accuracy for ChartQA)
app/              web demo
outputs/          checkpoints, logs, results (git-ignored)
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
