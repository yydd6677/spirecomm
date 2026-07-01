#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "smoke" ]]; then
  export RUN_NAME="${RUN_NAME:-smoke}"
  export MAIN_COUNT="${MAIN_COUNT:-2}"
  export EVAL_COUNT="${EVAL_COUNT:-2}"
  export VALUE_EPOCHS="${VALUE_EPOCHS:-1}"
  export POLICY_EPOCHS="${POLICY_EPOCHS:-1}"
  export WORKERS="${WORKERS:-1}"
  export CLEAN="${CLEAN:-1}"
  export VAL_MOD="${VAL_MOD:-2}"
  export VAL_REM="${VAL_REM:-0}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-/home/yydd/miniforge3/envs/spirecomm-rl/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

RUN_NAME="${RUN_NAME:-iter00}"
ITERATIONS="${ITERATIONS:-1}"
MAIN_SEED_START="${MAIN_SEED_START:-1}"
MAIN_COUNT="${MAIN_COUNT:-3000}"
EVAL_SEED_START="${EVAL_SEED_START:-1}"
EVAL_COUNT="${EVAL_COUNT:-300}"
WORKERS="${WORKERS:-8}"
ASCENSION="${ASCENSION:-0}"
MAX_STEPS="${MAX_STEPS:-1500}"
MAX_FLOOR="${MAX_FLOOR:-60}"
RESUME="${RESUME:-1}"
CLEAN="${CLEAN:-0}"

TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
VALUE_INFER_DEVICE="${VALUE_INFER_DEVICE:-cpu}"
COMBAT_DEVICE="${COMBAT_DEVICE:-cpu}"
SELECTOR_DEVICE="${SELECTOR_DEVICE:-cpu}"
TORCH_THREADS="${TORCH_THREADS:-1}"

VALUE_EPOCHS="${VALUE_EPOCHS:-20}"
VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-1024}"
VALUE_LR="${VALUE_LR:-2e-4}"
VALUE_HIDDEN_DIM="${VALUE_HIDDEN_DIM:-384}"
VALUE_DEPTH="${VALUE_DEPTH:-3}"
VALUE_CHUNK_SIZE="${VALUE_CHUNK_SIZE:-50000}"

POLICY_EPOCHS="${POLICY_EPOCHS:-15}"
POLICY_BATCH_ROOTS="${POLICY_BATCH_ROOTS:-256}"
POLICY_LR="${POLICY_LR:-2e-4}"
POLICY_HIDDEN_DIM="${POLICY_HIDDEN_DIM:-512}"
POLICY_DEPTH="${POLICY_DEPTH:-3}"
POLICY_CHUNK_SIZE_ROOTS="${POLICY_CHUNK_SIZE_ROOTS:-1000}"
TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-1.0}"
POLICY_CACHE_PROGRESS_INTERVAL="${POLICY_CACHE_PROGRESS_INTERVAL:-1000}"

VAL_MOD="${VAL_MOD:-10}"
VAL_REM="${VAL_REM:-0}"
SUMMARY_INTERVAL="${SUMMARY_INTERVAL:-25}"
RERANK_ALPHA="${RERANK_ALPHA:-0.25}"
RERANK_MIN_MARGIN="${RERANK_MIN_MARGIN:-0.0}"
RERANK_PHASES="${RERANK_PHASES:-COMBAT}"
QENV_PHASES="${QENV_PHASES:-COMBAT,CARD_REWARD}"
RECORD_LEGAL_ACTIONS="${RECORD_LEGAL_ACTIONS:-0}"
STRICT_GATES="${STRICT_GATES:-0}"
ALLOW_WARN_PROMOTION="${ALLOW_WARN_PROMOTION:-0}"

V3_COMBAT_MODEL="${V3_COMBAT_MODEL:-models/cache/download8_corrected_vocab/v5_dual_semantic_legacy_gate.pt}"
CARD_REWARD_MODEL="${CARD_REWARD_MODEL:-models/card_reward.pt}"
SHOP_CHOICE_MODEL="${SHOP_CHOICE_MODEL:-models/shop_choice_prior_delta.pt}"

ROOT_DATA_DIR="${ROOT_DATA_DIR:-data/run_value_v1/${RUN_NAME}}"
CACHE_ROOT="${CACHE_ROOT:-_cache/run_value_v1/${RUN_NAME}}"
MODEL_ROOT="${MODEL_ROOT:-models/run_value_v1/${RUN_NAME}}"
EVAL_ROOT="${EVAL_ROOT:-eval_runs/run_value_v1/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-logs/run_value_v1}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

mkdir -p "$ROOT_DATA_DIR" "$CACHE_ROOT" "$MODEL_ROOT" "$EVAL_ROOT" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

resume_flag=()
if [[ "$RESUME" == "1" ]]; then
  resume_flag=(--resume)
fi

legal_flag=(--no-record-legal-actions)
if [[ "$RECORD_LEGAL_ACTIONS" == "1" ]]; then
  legal_flag=(--record-legal-actions)
fi

safe_clean_dir() {
  local target="$1"
  case "$target" in
    data/run_value_v1/smoke*|_cache/run_value_v1/smoke*|models/run_value_v1/smoke*|eval_runs/run_value_v1/smoke*|logs/run_value_v1/smoke*)
      rm -rf "$target"
      ;;
    *)
      echo "refusing CLEAN=1 outside smoke path: $target" >&2
      return 2
      ;;
  esac
}

if [[ "$CLEAN" == "1" ]]; then
  safe_clean_dir "$ROOT_DATA_DIR"
  safe_clean_dir "$CACHE_ROOT"
  safe_clean_dir "$MODEL_ROOT"
  safe_clean_dir "$EVAL_ROOT"
  mkdir -p "$ROOT_DATA_DIR" "$CACHE_ROOT" "$MODEL_ROOT" "$EVAL_ROOT"
fi

run_cmd() {
  echo
  echo "[$(date '+%F %T')] $*"
  "$@"
}

check_value_checkpoint() {
  local checkpoint="$1"
  run_cmd "$PYTHON_BIN" - "$checkpoint" <<'PY'
import sys
from spirecomm.ai.run_value import STATE_FEATURE_DIM, load_run_value_checkpoint

path = sys.argv[1]
model, checkpoint = load_run_value_checkpoint(path, device="cpu")
schema = checkpoint.get("feature_schema") or {}
if int(schema.get("state_feature_dim") or -1) != STATE_FEATURE_DIM:
    raise SystemExit(f"bad state_feature_dim: {schema}")
if int(model.input_dim) != STATE_FEATURE_DIM:
    raise SystemExit(f"bad model input_dim: {model.input_dim}")
print(f"value checkpoint ok: {path}")
PY
}

check_policy_checkpoint() {
  local checkpoint="$1"
  run_cmd "$PYTHON_BIN" - "$checkpoint" <<'PY'
import sys
from spirecomm.ai.run_value import ACTION_CANDIDATE_FEATURE_DIM, load_run_action_policy_checkpoint

path = sys.argv[1]
model, checkpoint = load_run_action_policy_checkpoint(path, device="cpu")
schema = checkpoint.get("feature_schema") or {}
if int(schema.get("candidate_feature_dim") or -1) != ACTION_CANDIDATE_FEATURE_DIM:
    raise SystemExit(f"bad candidate_feature_dim: {schema}")
if int(model.input_dim) != ACTION_CANDIDATE_FEATURE_DIM:
    raise SystemExit(f"bad model input_dim: {model.input_dim}")
print(f"policy checkpoint ok: {path}")
PY
}

print_summary() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "${label}: missing $path"
    return
  fi
  run_cmd "$PYTHON_BIN" - "$label" "$path" <<'PY'
import json
import sys

label, path = sys.argv[1], sys.argv[2]
data = json.load(open(path, encoding="utf-8"))
keys = ["count", "mean_floor", "win_count", "death_count", "truncated_count", "error_count", "root_count", "candidate_count"]
parts = [f"{key}={data[key]}" for key in keys if key in data]
print(f"{label}: " + " ".join(parts))
PY
}

gate_rollout() {
  local label="$1"
  local baseline_summary="$2"
  local candidate_summary="$3"
  if [[ ! -f "$baseline_summary" || ! -f "$candidate_summary" ]]; then
    echo "gate ${label}: missing summary"
    return
  fi
  echo
  echo "[$(date '+%F %T')] $PYTHON_BIN - $label $baseline_summary $candidate_summary $STRICT_GATES"
  "$PYTHON_BIN" - "$label" "$baseline_summary" "$candidate_summary" "$STRICT_GATES" <<'PY'
import json
import sys

label, baseline_path, candidate_path, strict = sys.argv[1:5]
baseline = json.load(open(baseline_path, encoding="utf-8"))
candidate = json.load(open(candidate_path, encoding="utf-8"))
base_floor = float(baseline.get("mean_floor") or 0.0)
cand_floor = float(candidate.get("mean_floor") or 0.0)
errors = int(candidate.get("error_count") or 0)
truncated = int(candidate.get("truncated_count") or 0)
delta = cand_floor - base_floor
passed = errors == 0 and truncated == 0 and delta >= -1.0
print(
    f"gate {label}: baseline={base_floor:.3f} candidate={cand_floor:.3f} "
    f"delta={delta:+.3f} errors={errors} truncated={truncated} status={'PASS' if passed else 'WARN'}"
)
if strict == "1" and not passed:
    raise SystemExit(2)
if not passed:
    raise SystemExit(1)
PY
}

collect_rollout() {
  local output_dir="$1"
  local mode="$2"
  local count="$3"
  local seed_start="$4"
  shift 4
  if [[ "$RESUME" == "1" && -f "${output_dir}/summary.json" ]]; then
    echo "skip existing rollout: ${output_dir}"
    print_summary "$output_dir" "${output_dir}/summary.json"
    return
  fi
  run_cmd "$PYTHON_BIN" collect_run_value_trajectories.py \
    --output-dir "$output_dir" \
    --seed-start "$seed_start" \
    --count "$count" \
    --workers "$WORKERS" \
    --ascension "$ASCENSION" \
    --max-steps "$MAX_STEPS" \
    --max-floor "$MAX_FLOOR" \
    --mode "$mode" \
    --rerank-phases "$RERANK_PHASES" \
    --rerank-alpha "$RERANK_ALPHA" \
    --rerank-min-margin "$RERANK_MIN_MARGIN" \
    --device "$SELECTOR_DEVICE" \
    --combat-device "$COMBAT_DEVICE" \
    --v3-combat-model "$V3_COMBAT_MODEL" \
    --card-reward-model "$CARD_REWARD_MODEL" \
    --shop-choice-model "$SHOP_CHOICE_MODEL" \
    --value-device "$VALUE_INFER_DEVICE" \
    --torch-threads "$TORCH_THREADS" \
    --summary-interval "$SUMMARY_INTERVAL" \
    "${legal_flag[@]}" \
    --no-record-baseline-scores \
    "${resume_flag[@]}" \
    "$@"
  print_summary "$output_dir" "${output_dir}/summary.json"
}

train_value() {
  local input_dir="$1"
  local cache_dir="$2"
  local output="$3"
  if [[ "$RESUME" == "1" && -f "$output" && -f "${output}.summary.json" ]]; then
    echo "skip existing value model: ${output}"
    check_value_checkpoint "$output"
    return
  fi
  run_cmd "$PYTHON_BIN" train_run_value_model.py \
    --input-dir "$input_dir" \
    --cache-dir "$cache_dir" \
    --output "$output" \
    --rebuild-cache \
    --chunk-size "$VALUE_CHUNK_SIZE" \
    --val-mod "$VAL_MOD" \
    --val-rem "$VAL_REM" \
    --device "$TRAIN_DEVICE" \
    --epochs "$VALUE_EPOCHS" \
    --batch-size "$VALUE_BATCH_SIZE" \
    --learning-rate "$VALUE_LR" \
    --hidden-dim "$VALUE_HIDDEN_DIM" \
    --depth "$VALUE_DEPTH" \
    --progress-interval "$SUMMARY_INTERVAL"
  check_value_checkpoint "$output"
}

build_qenv() {
  local output_dir="$1"
  local value_model="$2"
  local rollout_mode="$3"
  shift 3
  if [[ "$RESUME" == "1" && -f "${output_dir}/summary.json" ]]; then
    echo "skip existing qenv: ${output_dir}"
    print_summary "$output_dir" "${output_dir}/summary.json"
    return
  fi
  run_cmd "$PYTHON_BIN" build_run_action_policy_dataset.py \
    --output-dir "$output_dir" \
    --seed-start "$MAIN_SEED_START" \
    --count "$MAIN_COUNT" \
    --workers "$WORKERS" \
    --ascension "$ASCENSION" \
    --max-steps "$MAX_STEPS" \
    --max-floor "$MAX_FLOOR" \
    --value-model "$value_model" \
    --value-device "$VALUE_INFER_DEVICE" \
    --rollout-mode "$rollout_mode" \
    --qenv-phases "$QENV_PHASES" \
    --rerank-phases "$RERANK_PHASES" \
    --rerank-alpha "$RERANK_ALPHA" \
    --rerank-min-margin "$RERANK_MIN_MARGIN" \
    --device "$SELECTOR_DEVICE" \
    --combat-device "$COMBAT_DEVICE" \
    --v3-combat-model "$V3_COMBAT_MODEL" \
    --card-reward-model "$CARD_REWARD_MODEL" \
    --shop-choice-model "$SHOP_CHOICE_MODEL" \
    --torch-threads "$TORCH_THREADS" \
    --summary-interval "$SUMMARY_INTERVAL" \
    "${resume_flag[@]}" \
    "$@"
  print_summary "$output_dir" "${output_dir}/summary.json"
}

train_policy() {
  local input_dir="$1"
  local cache_dir="$2"
  local output="$3"
  if [[ "$RESUME" == "1" && -f "$output" && -f "${output}.summary.json" ]]; then
    echo "skip existing policy model: ${output}"
    check_policy_checkpoint "$output"
    return
  fi
  run_cmd "$PYTHON_BIN" train_run_action_policy.py \
    --input-dir "$input_dir" \
    --cache-dir "$cache_dir" \
    --output "$output" \
    --rebuild-cache \
    --chunk-size-roots "$POLICY_CHUNK_SIZE_ROOTS" \
    --val-mod "$VAL_MOD" \
    --val-rem "$VAL_REM" \
    --device "$TRAIN_DEVICE" \
    --epochs "$POLICY_EPOCHS" \
    --batch-roots "$POLICY_BATCH_ROOTS" \
    --learning-rate "$POLICY_LR" \
    --hidden-dim "$POLICY_HIDDEN_DIM" \
    --depth "$POLICY_DEPTH" \
    --target-temperature "$TARGET_TEMPERATURE" \
    --cache-progress-interval "$POLICY_CACHE_PROGRESS_INTERVAL" \
    --progress-interval "$SUMMARY_INTERVAL"
  check_policy_checkpoint "$output"
}

previous_policy=""
for iteration in $(seq 0 $((ITERATIONS - 1))); do
  iter_id=$(printf "iter%02d" "$iteration")
  rollout_dir="${ROOT_DATA_DIR}/${iter_id}_rollout"
  value_cache="${CACHE_ROOT}/${iter_id}_value_cache"
  value_model="${MODEL_ROOT}/run_value_${iter_id}.pt"
  shadow_dir="${EVAL_ROOT}/${iter_id}_shadow"
  rerank_dir="${EVAL_ROOT}/${iter_id}_rerank"
  qenv_dir="${ROOT_DATA_DIR}/${iter_id}_qenv"
  policy_cache="${CACHE_ROOT}/${iter_id}_policy_cache"
  policy_model="${MODEL_ROOT}/run_action_policy_${iter_id}.pt"
  policy_eval_dir="${EVAL_ROOT}/${iter_id}_policy_eval"

  rollout_mode="baseline"
  rollout_extra=()
  if [[ "$iteration" -gt 0 && -n "$previous_policy" ]]; then
    rollout_mode="${ONPOLICY_ROLLOUT_MODE:-policy}"
    rollout_extra=(--action-policy-model "$previous_policy")
  fi

  collect_rollout "$rollout_dir" "$rollout_mode" "$MAIN_COUNT" "$MAIN_SEED_START" "${rollout_extra[@]}"
  train_value "$rollout_dir" "$value_cache" "$value_model"

  collect_rollout "$shadow_dir" shadow "$EVAL_COUNT" "$EVAL_SEED_START" --value-model "$value_model" --record-candidate-scores
  collect_rollout "$rerank_dir" rerank "$EVAL_COUNT" "$EVAL_SEED_START" --value-model "$value_model"
  if gate_rollout "${iter_id}_rerank" "${rollout_dir}/summary.json" "${rerank_dir}/summary.json"; then
    :
  else
    gate_status=$?
    if [[ "$gate_status" == "2" ]]; then
      exit 2
    fi
  fi

  build_qenv "$qenv_dir" "$value_model" "$rollout_mode" "${rollout_extra[@]}"
  train_policy "$qenv_dir" "$policy_cache" "$policy_model"
  collect_rollout "$policy_eval_dir" policy "$EVAL_COUNT" "$EVAL_SEED_START" --action-policy-model "$policy_model"
  policy_gate_pass=0
  if gate_rollout "${iter_id}_policy" "${rollout_dir}/summary.json" "${policy_eval_dir}/summary.json"; then
    policy_gate_pass=1
  else
    gate_status=$?
    if [[ "$gate_status" == "2" ]]; then
      exit 2
    fi
  fi
  if [[ "$policy_gate_pass" == "1" || "$ALLOW_WARN_PROMOTION" == "1" ]]; then
    previous_policy="$policy_model"
  else
    previous_policy=""
    echo "policy gate did not pass; next iteration will fall back to baseline rollout unless ALLOW_WARN_PROMOTION=1"
  fi
done

echo
echo "run-value iteration workflow complete: ${RUN_NAME}"
echo "log: ${LOG_FILE}"
