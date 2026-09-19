#!/bin/zsh
# Daily Automated Shadow Sync & Model Retraining Workflow

set -e
DATE_STR="${1:-$(date +%Y-%m-%d)}"
ROUTER_LOGS="/Users/chenyc/Documents/study/jev-cliproxy-router/logs"
DATA_FILE="/Users/chenyc/Documents/study/NanoJev/data/harvested_shadow_data.jsonl"
REMOTE_HOST="chenyc@192.168.123.88"

echo "=== [1/4] Running Daily Shadow Log Analysis for ${DATE_STR} ==="
cd /Users/chenyc/Documents/study/NanoJev
source .venv/bin/activate
PYTHONPATH=scripts python3 scripts/shadow_pipeline.py \
  --logs-dir "${ROUTER_LOGS}" \
  --date "${DATE_STR}" \
  --harvest-out "${DATA_FILE}"

# Check if there are newly harvested hard cases to sync and retrain
if [[ -f "${DATA_FILE}" ]] && [[ $(wc -l < "${DATA_FILE}") -ge 10 ]]; then
  echo "=== [2/4] Accumulated enough hard cases, syncing to 88 GPU Server ==="
  rtk scp "${DATA_FILE}" "${REMOTE_HOST}:~/work/NanoJev/data/harvested_shadow_data.jsonl"

  echo "=== [3/4] Triggering Incremental Retraining on M1 Mac Studio ==="
  ssh "${REMOTE_HOST}" "
    cd ~/work/NanoJev
    source .venv/bin/activate
    # Combine baseline dataset with newly harvested production cases
    cat data/master_realistic_dataset.jsonl data/harvested_shadow_data.jsonl > data/auto_evolved_dataset.jsonl
    nohup python3 scripts/train_deep_heads.py \
      --input data/auto_evolved_dataset.jsonl \
      --output-dir checkpoints/router_realistic_mlp \
      --base-checkpoint checkpoints/NanoJev \
      --epochs 20 \
      --lr 1e-3 \
      --batch-questions 16 \
      --max-length 4096 > train_auto.log 2>&1 &
    echo 'Retraining background PID: '\$!
  "

  echo "=== [4/4] Auto-retraining started in background! ==="
else
  echo "=== [2/2] Daily analysis complete. Hard cases count is under threshold (10), continuing accumulation. ==="
fi
