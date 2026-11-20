#!/usr/bin/env bash
# Full AWS pipeline for one run (PaliGemma 2 or Florence-2): baseline -> training -> checkpoint selection -> final eval -> Hub.
# The model family comes from the config's `family:` key (default paligemma2).
# Designed to run unattended inside tmux:
#     tmux new -d -s pipe 'bash scripts/run_pipeline.sh configs/paligemma2_448.yaml ft448 2>&1 | tee -a ~/work/logs/pipeline_ft448.log'
# Every step is skipped if its output already exists, so the script can simply be re-run after a crash.
set -uo pipefail
source /home/ec2-user/work/env.sh
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/paligemma2_448.yaml}
STAGE=${2:-ft448}
BASE=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['base'])")
OUT=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
FAMILY=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG')).get('family', 'paligemma2'))")
case "$FAMILY" in
  paligemma2) TRAIN_SCRIPT=scripts/train_paligemma.py; DEFAULT_HUB=paligemma2-3b-docchart-lora-${STAGE#ft} ;;
  florence2)  TRAIN_SCRIPT=scripts/train_florence.py;  DEFAULT_HUB=$(basename "$OUT" | tr _ -)-docchart-${STAGE} ;;
  *) echo "unknown family $FAMILY"; exit 1 ;;
esac
HUB_REPO=${HUB_REPO:-$DEFAULT_HUB}
EVAL_BS=${EVAL_BS:-8}
SKIP_BASELINE=${SKIP_BASELINE:-0}

log() { echo "[pipeline $(date -u +%H:%M:%S)] $*"; }
ev() {  # ev <stage> <dataset> <split> [extra args...]  (skips if summary exists)
  local stage=$1 ds=$2 split=$3; shift 3
  python scripts/evaluate.py --family "$FAMILY" --base "$BASE" --stage "$stage" --dataset "$ds" \
      --split "$split" --batch-size "$EVAL_BS" "$@" || { log "EVAL FAILED: $stage $ds $split $*"; return 1; }
}
have() { [ -f "results/summary/$1.json" ]; }

# ---------------------------------------------------------------- 1. zero-shot baseline ("before")
if [ "$SKIP_BASELINE" != 1 ]; then
  have ${FAMILY}_zeroshot_docvqa_train_docvqa_dev500   || ev zeroshot docvqa  train      --ids docvqa_dev500_ids.json
  have ${FAMILY}_zeroshot_chartqa_val_chartqa_val500   || ev zeroshot chartqa val        --ids chartqa_val500_ids.json
  have ${FAMILY}_zeroshot_docvqa_validation            || ev zeroshot docvqa  validation
  have ${FAMILY}_zeroshot_chartqa_test                 || ev zeroshot chartqa test
fi

# ---------------------------------------------------------------- 2. training
if [ ! -d "$OUT/final" ]; then
  log "training $CONFIG"
  python "$TRAIN_SCRIPT" --config "$CONFIG" --resume || { log "TRAINING FAILED"; exit 1; }
fi

# ---------------------------------------------------------------- 3. checkpoint selection on quick dev subsets
for ck in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n) "$OUT/final"; do
  tag=$(basename "$ck")
  have "${FAMILY}_${STAGE}_docvqa_train_${tag}"  || ev "$STAGE" docvqa  train --ids docvqa_dev500_ids.json  --adapter "$ck" --tag "$tag"
  have "${FAMILY}_${STAGE}_chartqa_val_${tag}"   || ev "$STAGE" chartqa val   --ids chartqa_val500_ids.json --adapter "$ck" --tag "$tag"
done
BEST=$(python - "$STAGE" "$OUT" "$FAMILY" <<'EOF'
import json, sys, glob, os
stage, out, family = sys.argv[1], sys.argv[2], sys.argv[3]
best, best_score = None, -1
for ck in glob.glob(f"{out}/checkpoint-*") + [f"{out}/final"]:
    tag = os.path.basename(ck)
    try:
        d = json.load(open(f"results/summary/{family}_{stage}_docvqa_train_{tag}.json"))["score"]
        c = json.load(open(f"results/summary/{family}_{stage}_chartqa_val_{tag}.json"))["score"]
    except FileNotFoundError:
        continue
    print(f"  {tag:<18} dev ANLS {d:.4f}  dev RA {c:.4f}  mean {(d + c) / 2:.4f}", file=sys.stderr)
    if (d + c) / 2 > best_score:
        best, best_score = ck, (d + c) / 2
print(best)
EOF
)
log "best checkpoint: $BEST"
echo "$BEST" > "$OUT/BEST"

# ---------------------------------------------------------------- 4. final evaluation of the best checkpoint
have "${FAMILY}_${STAGE}_docvqa_validation" || ev "$STAGE" docvqa  validation --adapter "$BEST"
have "${FAMILY}_${STAGE}_chartqa_test"      || ev "$STAGE" chartqa test       --adapter "$BEST"

# ---------------------------------------------------------------- 5. upload adapter + tables
python scripts/push_to_hub.py --folder "$BEST" --repo "$HUB_REPO" --message "best checkpoint $(basename "$BEST") ($STAGE)" \
  || log "HUB UPLOAD FAILED"
python scripts/make_results_table.py > /dev/null
log "PIPELINE DONE ($STAGE)"
