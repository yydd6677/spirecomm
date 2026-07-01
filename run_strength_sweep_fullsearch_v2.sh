#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/yydd/miniforge3/envs/spirecomm-rl/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE

WORKERS="${WORKERS:-8}"
OUT_DIR="${OUT_DIR:-teacher_sweep_runs/v3_teacher_strength_fullsearch_v2}"
LOG_PATH="${LOG_PATH:-logs/v3_teacher_strength_fullsearch_v2.log}"
PID_PATH="${PID_PATH:-logs/v3_teacher_strength_fullsearch_v2.pid}"

STRENGTH_GRID="${STRENGTH_GRID:-{\"player_power_weights.Strength\":[0.0,2.0,4.486054133555118,6.0,7.0,8.0,9.0,9.5,10.0,10.3,10.6,11.0,11.5,12.0,13.0,14.0,16.0,18.0,22.0]}}"
PARAM_RANGE_JSON="${PARAM_RANGE_JSON:-{\"player_power_weights.Strength\":[0.0,22.0]}}"

ROUND1_COUNT="${ROUND1_COUNT:-100}"
ROUND2_TOP="${ROUND2_TOP:-10}"
ROUND2_COUNT="${ROUND2_COUNT:-200}"
ROUND3_TOP="${ROUND3_TOP:-6}"
ROUND3_COUNT="${ROUND3_COUNT:-300}"
ROUND4_TOP="${ROUND4_TOP:-3}"
ROUND4_COUNT="${ROUND4_COUNT:-600}"
STOP_AFTER="${STOP_AFTER:-round4}"
START_AT="${START_AT:-round1}"

if [ "${SMOKE:-0}" = "1" ]; then
  WORKERS=1
  OUT_DIR="teacher_sweep_runs/_smoke_strength_fullsearch_v2"
  LOG_PATH="logs/_smoke_strength_fullsearch_v2.log"
  PID_PATH="logs/_smoke_strength_fullsearch_v2.pid"
  STRENGTH_GRID='{"player_power_weights.Strength":[4.486054133555118]}'
  ROUND1_COUNT=1
  ROUND2_TOP=1
  ROUND2_COUNT=1
  ROUND3_TOP=1
  ROUND3_COUNT=1
  ROUND4_TOP=1
  ROUND4_COUNT=1
  STOP_AFTER="round1"
fi

mkdir -p "$(dirname "$LOG_PATH")" "$OUT_DIR"

CMD=(
  "$PYTHON" -u run_v3_teacher_config_sweep_fast.py
  --output-dir "$OUT_DIR"
  --param-ranges-json "$PARAM_RANGE_JSON"
  --round1-grid-json "$STRENGTH_GRID"
  --seed-start 1
  --workers "$WORKERS"
  --torch-threads 1
  --blas-threads 1
  --metrics-mode floor
  --summary-interval 20
  --progress-interval-tasks 10
  --round0-count 0
  --round1-size 0
  --round1-count "$ROUND1_COUNT"
  --round1-stage-counts "$ROUND1_COUNT"
  --round1-stage-keeps "$ROUND2_TOP"
  --round2-top "$ROUND2_TOP"
  --round2-count "$ROUND2_COUNT"
  --round3-top "$ROUND3_TOP"
  --round3-count "$ROUND3_COUNT"
  --round4-top "$ROUND4_TOP"
  --round4-count "$ROUND4_COUNT"
  --round1-proxy-beam-width 0
  --round1-proxy-node-budget 0
  --round1-proxy-max-depth 0
  --round2-proxy-beam-width 0
  --round2-proxy-node-budget 0
  --round2-proxy-max-depth 0
  --round3-proxy-beam-width 0
  --round3-proxy-node-budget 0
  --round3-proxy-max-depth 0
  --round4-proxy-beam-width 0
  --round4-proxy-node-budget 0
  --round4-proxy-max-depth 0
  --start-at "$START_AT"
  --stop-after "$STOP_AFTER"
  --resume
)

printf '%q ' "${CMD[@]}" > "${LOG_PATH%.log}.command"
printf '\n' >> "${LOG_PATH%.log}.command"

echo "python=$PYTHON"
echo "strength grid: $STRENGTH_GRID"
echo "output_dir=$OUT_DIR"
echo "log_path=$LOG_PATH"
echo "workers=$WORKERS"
echo "rounds: round1=${ROUND1_COUNT}, round2=${ROUND2_COUNT}, round3=${ROUND3_COUNT}, round4=${ROUND4_COUNT}"
echo "promotion: round2_top=${ROUND2_TOP}, round3_top=${ROUND3_TOP}, round4_top=${ROUND4_TOP}"
echo "search: full teacher defaults in every round"
echo "command file=${LOG_PATH%.log}.command"

if [ "${FOREGROUND:-0}" = "1" ] || [ "${SMOKE:-0}" = "1" ]; then
  "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
else
  setsid "${CMD[@]}" >> "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "started pid=$(cat "$PID_PATH")"
  echo "tail -f $LOG_PATH"
fi
