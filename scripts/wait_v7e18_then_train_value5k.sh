#!/usr/bin/env bash
set -euo pipefail

cd /home/yydd/spirecomm

EPOCH_FILE="models/v3_combat_transformer_v5_dual_action_binding_v7_local.pt.epochs/epoch_018.pt"
V7_PGID="${V7_PGID:-1436027}"
V7_PID="${V7_PID:-1436039}"

echo "[$(date +%F_%T)] waiting for ${EPOCH_FILE}"
while [ ! -s "${EPOCH_FILE}" ]; do
  if ! ps -p "${V7_PID}" >/dev/null 2>&1; then
    echo "[$(date +%F_%T)] v7 pid ${V7_PID} exited before epoch_018.pt appeared"
    break
  fi
  sleep 15
done

if [ -s "${EPOCH_FILE}" ]; then
  echo "[$(date +%F_%T)] epoch_018 checkpoint detected; stopping v7 process group -${V7_PGID}"
  kill -TERM -"${V7_PGID}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! ps -p "${V7_PID}" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  if ps -p "${V7_PID}" >/dev/null 2>&1; then
    echo "[$(date +%F_%T)] v7 still alive after TERM; sending KILL"
    kill -KILL -"${V7_PGID}" 2>/dev/null || true
  fi
fi

echo "[$(date +%F_%T)] starting value model training"
mkdir -p logs/run_value_v1 models/run_value_v1/value_5k_baseline data/run_value_v1/value_5k_baseline

CUDA_MODULE_LOADING=LAZY \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
/home/yydd/miniforge3/envs/spirecomm-rl/bin/python -u scripts/run_value/train_run_value_model.py \
  --input-dir data/run_value_v1/value_2k_baseline/iter00_rollout \
  --cache-dir data/run_value_v1/value_5k_baseline/tensor_cache_iter00 \
  --output models/run_value_v1/value_5k_baseline/run_value_iter00.pt \
  --summary models/run_value_v1/value_5k_baseline/run_value_iter00.pt.summary.json \
  --rebuild-cache \
  --chunk-size 100000 \
  --cache-workers 8 \
  --epochs 30 \
  --batch-size 4096 \
  --learning-rate 2e-4 \
  --min-learning-rate 2e-5 \
  --weight-decay 1e-4 \
  --hidden-dim 384 \
  --depth 3 \
  --dropout 0.05 \
  --seed 11 \
  --device cuda \
  --progress-interval 50

echo "[$(date +%F_%T)] value model training finished"
