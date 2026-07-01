#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/yydd/miniforge3/envs/spirecomm-rl/bin/python}"
INPUT_DIR="${INPUT_DIR:-data/run_value_v1/value_maprng_v3_1k/iter00_rollout}"
CACHE_DIR="${CACHE_DIR:-data/run_value_v1/value_v5_augmented/tensor_cache_iter00}"
ANALYSIS_DIR="${ANALYSIS_DIR:-analysis_outputs/run_value}"
LOG_DIR="${LOG_DIR:-logs/run_value_v1}"
TARGET_MAE="${TARGET_MAE:-3.0}"
CURRENT_PID_FILE="${CURRENT_PID_FILE:-logs/run_value_v1/value_v5_augmented_v2config.pid}"

mkdir -p "$ANALYSIS_DIR" "$LOG_DIR" models/run_value_v1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

wait_for_current() {
  if [[ -f "$CURRENT_PID_FILE" ]]; then
    local pid
    pid="$(cat "$CURRENT_PID_FILE")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "waiting current run-value job pid=$pid"
      while kill -0 "$pid" 2>/dev/null; do
        sleep 60
      done
    fi
  fi
}

eval_checkpoint() {
  local name="$1"
  local ckpt="$2"
  local output="$ANALYSIS_DIR/${name}_eval.json"
  "$PY" scripts/run_value/evaluate_run_value_model.py \
    --checkpoint "$ckpt" \
    --cache-dir "$CACHE_DIR" \
    --output "$output" \
    --device cuda \
    --batch-size 4096 \
    > "$LOG_DIR/${name}_eval.log" 2>&1
  "$PY" - "$output" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
print(data["metrics"]["remaining_mae"])
PY
}

target_reached() {
  local mae="$1"
  "$PY" - "$mae" "$TARGET_MAE" <<'PY'
import sys
mae = float(sys.argv[1])
target = float(sys.argv[2])
raise SystemExit(0 if mae <= target else 1)
PY
}

train_variant() {
  local name="$1"
  shift
  local out_dir="models/run_value_v1/${name}"
  mkdir -p "$out_dir"
  "$PY" -u scripts/run_value/train_run_value_model.py \
    --input-dir "$INPUT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output "$out_dir/run_value_iter00.pt" \
    --summary "$out_dir/run_value_iter00.pt.summary.json" \
    --sample-mode before_after \
    --feature-variant no_seed_structrng_explicit_aug \
    --record-weight-mode balanced_v2 \
    --before-weight 0.4 \
    --after-weight 0.6 \
    --cache-feature-dtype float16 \
    --cache-direct-write \
    --device cuda \
    --epochs 30 \
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
    --residual-floor-baseline \
    --residual-baseline-key floor \
    --progress-interval 100 \
    --early-stop-patience 6 \
    --early-stop-min-delta 0.001 \
    --save-each-epoch \
    "$@" \
    > "$LOG_DIR/${name}.log" 2>&1
}

wait_for_current

if [[ -f models/run_value_v1/value_v5_augmented_v2config/run_value_iter00.pt ]]; then
  mae="$(eval_checkpoint value_v5_augmented_v2config models/run_value_v1/value_v5_augmented_v2config/run_value_iter00.pt)"
  echo "value_v5_augmented_v2config eval_mae=$mae"
  if target_reached "$mae"; then
    echo "target reached by value_v5_augmented_v2config"
    exit 0
  fi
fi

train_variant value_v6_aug_resmlp_d010 \
  --architecture res_mlp \
  --hidden-dim 384 \
  --depth 3 \
  --dropout 0.10 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --min-learning-rate 1e-5 \
  --seed 61
mae="$(eval_checkpoint value_v6_aug_resmlp_d010 models/run_value_v1/value_v6_aug_resmlp_d010/run_value_iter00.pt)"
echo "value_v6_aug_resmlp_d010 eval_mae=$mae"
if target_reached "$mae"; then
  echo "target reached by value_v6_aug_resmlp_d010"
  exit 0
fi

train_variant value_v6_aug_mlp_d015 \
  --architecture mlp \
  --hidden-dim 384 \
  --depth 3 \
  --dropout 0.15 \
  --batch-size 2048 \
  --learning-rate 1e-4 \
  --min-learning-rate 1e-5 \
  --seed 62
mae="$(eval_checkpoint value_v6_aug_mlp_d015 models/run_value_v1/value_v6_aug_mlp_d015/run_value_iter00.pt)"
echo "value_v6_aug_mlp_d015 eval_mae=$mae"
if target_reached "$mae"; then
  echo "target reached by value_v6_aug_mlp_d015"
  exit 0
fi

train_variant value_v6_aug_mlp_l1 \
  --architecture mlp \
  --hidden-dim 384 \
  --depth 3 \
  --dropout 0.05 \
  --batch-size 2048 \
  --learning-rate 1e-4 \
  --min-learning-rate 1e-5 \
  --regression-loss l1 \
  --seed 63
mae="$(eval_checkpoint value_v6_aug_mlp_l1 models/run_value_v1/value_v6_aug_mlp_l1/run_value_iter00.pt)"
echo "value_v6_aug_mlp_l1 eval_mae=$mae"
if target_reached "$mae"; then
  echo "target reached by value_v6_aug_mlp_l1"
  exit 0
fi

echo "all configured variants finished without reaching target_mae=$TARGET_MAE"
