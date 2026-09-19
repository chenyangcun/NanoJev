#!/bin/zsh
# Hourly Incremental Shadow Sync & Model Auto-Evolve Workflow

set -e
DATE_STR="${1:-$(date +%Y-%m-%d)}"
ROUTER_LOGS="/Users/chenyc/Documents/study/jev-cliproxy-router/logs"
DATA_FILE="/Users/chenyc/Documents/study/NanoJev/data/harvested_shadow_data.jsonl"
STATE_FILE="/Users/chenyc/Documents/study/NanoJev/data/.processed_shadow_ids.json"
REMOTE_HOST="chenyc@192.168.123.88"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Hourly Incremental Shadow Scan for ${DATE_STR}..."
cd /Users/chenyc/Documents/study/NanoJev

PYTHONPATH=scripts .venv/bin/python3 scripts/shadow_pipeline.py \
  --logs-dir "${ROUTER_LOGS}" \
  --date "${DATE_STR}" \
  --harvest-out "${DATA_FILE}" \
  --state-file "${STATE_FILE}"

# If accumulated hard cases reach threshold (>= 10), trigger remote retraining
if [[ -f "${DATA_FILE}" ]] && [[ $(wc -l < "${DATA_FILE}") -ge 10 ]]; then
  echo "=== Accumulated $(wc -l < "${DATA_FILE}") hard cases (>=10). Triggering remote MLX retraining on 88 ==="
  rtk scp "${DATA_FILE}" "${REMOTE_HOST}:~/work/NanoJev/data/harvested_shadow_data.jsonl"

  ssh "${REMOTE_HOST}" "
    cd ~/work/NanoJev
    source .venv/bin/activate
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
  echo "=== Remote retraining triggered successfully! ==="
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scan complete. Accumulated samples under threshold, no retraining needed."
fi
