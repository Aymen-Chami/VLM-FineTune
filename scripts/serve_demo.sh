#!/usr/bin/env bash
# Run the shareable web demo on the AWS instance (as ec2-user, after setup_aws.sh + install_env.sh):
# GPU model server on 127.0.0.1:8000 (private) + website on port 3000 (public), both in tmux session "demo".
#
#   bash scripts/serve_demo.sh        # (re)start; installs Node and rebuilds the site
#   tmux attach -t demo               # logs (Ctrl-b n = next window, Ctrl-b d = detach)
#   tmux kill-session -t demo         # stop and free the GPU
#
# Open TCP 3000 in the instance's security group first. The public IP changes every time the instance
# is stopped and started.
set -euo pipefail
source /home/ec2-user/work/env.sh
cd "$(dirname "$0")/.."
REPO=$(pwd)
NODE_VERSION=v24.11.1
NODE_BIN=$WORK/node-$NODE_VERSION-linux-x64/bin
mkdir -p "$WORK/logs"

if [ ! -x "$NODE_BIN/node" ]; then
  echo "installing Node $NODE_VERSION into $WORK"
  curl -fsSL "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-x64.tar.xz" | tar -xJ -C "$WORK"
fi
export PATH=$NODE_BIN:$PATH

pip install -q -r requirements.txt
(cd app && npm ci --no-audit --no-fund && npm run build)

tmux kill-session -t demo 2>/dev/null || true
tmux new-session -d -s demo -n model \
  "source $WORK/env.sh && cd $REPO && python scripts/demo_server.py 2>&1 | tee -a $WORK/logs/demo_model.log"
tmux new-window -t demo -n web \
  "export PATH=$NODE_BIN:\$PATH && cd $REPO/app && INFERENCE_URL=http://127.0.0.1:8000/infer npx next start -p 3000 2>&1 | tee -a $WORK/logs/demo_web.log"

echo -n "waiting for the models to load"
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  echo -n "."
  sleep 5
done
echo
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "model server not up yet, check: tmux attach -t demo"; exit 1; }
echo "demo running: http://$(curl -s https://checkip.amazonaws.com):3000"
