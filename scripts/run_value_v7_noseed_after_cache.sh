#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/yydd/miniforge3/envs/spirecomm-rl/bin/python}"
LOG_DIR="${LOG_DIR:-logs/run_value_v1}"
INPUT_DIR="${INPUT_DIR:-data/run_value_v1/value_maprng_v3_1k/iter00_rollout}"
CACHE_DIR="${CACHE_DIR:-data/run_value_v1/value_v2_aftermix_residual_aux_noseedrng/tensor_cache_iter00_2336}"
CACHE_PID_FILE="${CACHE_PID_FILE:-logs/run_value_v1/value_v2_noseed_cache_rebuild.pid}"
mkdir -p "$LOG_DIR" analysis_outputs/run_value models/run_value_v1

if [[ -f "$CACHE_PID_FILE" ]]; then
  cache_pid="$(cat "$CACHE_PID_FILE")"
  if [[ -n "$cache_pid" ]] && kill -0 "$cache_pid" 2>/dev/null; then
    echo "waiting cache rebuild pid=$cache_pid"
    while kill -0 "$cache_pid" 2>/dev/null; do
      sleep 60
    done
  fi
fi

if [[ ! -f "$CACHE_DIR/manifest.json" ]]; then
  echo "missing cache manifest: $CACHE_DIR/manifest.json" >&2
  exit 1
fi

if [[ -f models/run_value_v1/value_v2_aftermix_residual_aux_noseedrng/run_value_iter00.pt ]]; then
  "$PY" evaluate_run_value_model.py \
    --checkpoint models/run_value_v1/value_v2_aftermix_residual_aux_noseedrng/run_value_iter00.pt \
    --cache-dir "$CACHE_DIR" \
    --output analysis_outputs/run_value/value_v2_rebuilt_cache_eval.json \
    --device cuda \
    --batch-size 8192 \
    > "$LOG_DIR/value_v2_rebuilt_cache_eval.log" 2>&1
fi

OUT_DIR="models/run_value_v1/value_v7_noseed_beta2"
mkdir -p "$OUT_DIR"
"$PY" -u train_run_value_model.py \
  --input-dir "$INPUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --output "$OUT_DIR/run_value_iter00.pt" \
  --summary "$OUT_DIR/run_value_iter00.pt.summary.json" \
  --sample-mode before_after \
  --feature-variant no_seed_structrng \
  --record-weight-mode balanced_v2 \
  --before-weight 0.4 \
  --after-weight 0.6 \
  --cache-feature-dtype float16 \
  --cache-direct-write \
  --device cuda \
  --epochs 24 \
  --architecture mlp \
  --model-input-dim 2336 \
  --hidden-dim 384 \
  --depth 3 \
  --dropout 0.05 \
  --batch-size 2048 \
  --learning-rate 1e-4 \
  --min-learning-rate 1e-5 \
  --weight-decay 1e-4 \
  --survival-bins 12 \
  --survival-weight 0.05 \
  --survival-value-weight 0.0 \
  --final-floor-bins 8 \
  --final-floor-weight 0.02 \
  --final-floor-value-weight 0.0 \
  --final-floor-readout none \
  --final-loss-weight 0.05 \
  --act-bce-weight 0.05 \
  --death-bce-weight 0.10 \
  --row-weight-mode precomputed \
  --regression-loss smooth_l1 \
  --smooth-l1-beta 2.0 \
  --progress-interval 100 \
  --early-stop-patience 6 \
  --early-stop-min-delta 0.001 \
  --save-each-epoch \
  --seed 67 \
  > "$LOG_DIR/value_v7_noseed_beta2.log" 2>&1

"$PY" evaluate_run_value_model.py \
  --checkpoint "$OUT_DIR/run_value_iter00.pt" \
  --cache-dir "$CACHE_DIR" \
  --output analysis_outputs/run_value/value_v7_noseed_beta2_eval.json \
  --device cuda \
  --batch-size 8192 \
  > "$LOG_DIR/value_v7_noseed_beta2_eval.log" 2>&1

echo "value_v7_noseed_beta2 complete"
