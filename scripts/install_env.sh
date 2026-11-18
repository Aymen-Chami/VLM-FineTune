#!/usr/bin/env bash
# User-level environment for the AWS instance (run as ec2-user after setup_aws.sh).
# Creates ~/work/{venv,hf,tmp}, an env.sh sourced from ~/.bashrc, and installs requirements.
# Put the Hugging Face token in ~/work/hf/token (chmod 600) before running training/eval.
set -euxo pipefail

WORK=/home/ec2-user/work
mkdir -p "$WORK/hf" "$WORK/tmp" "$WORK/.cache/pip"

cat > "$WORK/env.sh" <<'EOS'
export WORK=/home/ec2-user/work
export HF_HOME=$WORK/hf
export PIP_CACHE_DIR=$WORK/.cache/pip
export TMPDIR=$WORK/tmp
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
if [ -f $WORK/venv/bin/activate ]; then source $WORK/venv/bin/activate; fi
EOS
grep -q "$WORK/env.sh" ~/.bashrc || echo "source $WORK/env.sh" >> ~/.bashrc
source "$WORK/env.sh"

python3.12 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install -U pip wheel
pip install torch torchvision
pip install -r "$(dirname "$0")/../requirements.txt"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
