#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("/home/yydd/下载/aliyun_teacher_tempo_power_sweep_v1")

MODEL_FILES = (
    "models/combat.pt",
    "models/v3_combat_scorer.pt",
    "models/card_reward.pt",
    "models/boss_relic.pt",
    "models/campfire.pt",
    "models/event_choice.pt",
    "models/map_choice.pt",
    "models/potion_use.pt",
    "models/purge_target.pt",
    "models/shop_choice.pt",
    "models/shop_choice_prior_delta.pt",
    "models/upgrade_target.pt",
)

TOP_LEVEL_FILES = (
    "build_v3_first_combat_snapshots.py",
    "evaluate_v3_rollout_batch.py",
    "run_v3_teacher_config_sweep.py",
    "run_v3_teacher_config_sweep_fast.py",
    "watch_teacher_sweep_progress.py",
    "setup.py",
)

TEMPO_POWER_RANGES = (
    '{"power_card_constant":[0,20],'
    '"skill_power_turn_constant":[4,16],'
    '"turn_order_decay_per_card":[0,0.4],'
    '"play_card_constant":[-0.8,1.2],'
    '"energy_spent_weight":[-0.6,0.3]}'
)


def ignore_patterns(_path: str, names: list[str]) -> set[str]:
    ignored = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
    }
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo")))
    return ignored


def copy_path(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=ignore_patterns)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_text(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def script_common() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PYTHON:-python3}"
fi
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

sanitize_thread_var() {
  local name="$1"
  local fallback="$2"
  local value="${!name:-$fallback}"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 1 ]]; then
    value="$fallback"
  fi
  export "$name=$value"
}

sanitize_thread_var OMP_NUM_THREADS 1
sanitize_thread_var MKL_NUM_THREADS 1
sanitize_thread_var OPENBLAS_NUM_THREADS 1
sanitize_thread_var NUMEXPR_NUM_THREADS 1
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"

export SPIRECOMM_V3_TEACHER_FAST_STATE="${SPIRECOMM_V3_TEACHER_FAST_STATE:-1}"
export SPIRECOMM_V3_TEACHER_FAST_LEGAL_ACTIONS="${SPIRECOMM_V3_TEACHER_FAST_LEGAL_ACTIONS:-1}"
export SPIRECOMM_V3_TEACHER_COMBAT_STEP_SOURCE="${SPIRECOMM_V3_TEACHER_COMBAT_STEP_SOURCE:-1}"
export SPIRECOMM_V3_TEACHER_ENGINE_STEP_SOURCE="${SPIRECOMM_V3_TEACHER_ENGINE_STEP_SOURCE:-1}"
export SPIRECOMM_V3_TEACHER_FAST_BRANCH_SYNC="${SPIRECOMM_V3_TEACHER_FAST_BRANCH_SYNC:-1}"
export SPIRECOMM_V3_TEACHER_SLIM_BRANCH_CLONE="${SPIRECOMM_V3_TEACHER_SLIM_BRANCH_CLONE:-1}"
export SPIRECOMM_V3_TEACHER_DEDUPE_EQUIVALENT_CARD_ACTIONS="${SPIRECOMM_V3_TEACHER_DEDUPE_EQUIVALENT_CARD_ACTIONS:-1}"
export SPIRECOMM_V3_TEACHER_DEDUPE_EQUIVALENT_TARGETS="${SPIRECOMM_V3_TEACHER_DEDUPE_EQUIVALENT_TARGETS:-1}"
export SPIRECOMM_V3_TEACHER_CANONICALIZE_HAND_ORDER="${SPIRECOMM_V3_TEACHER_CANONICALIZE_HAND_ORDER:-1}"
export SPIRECOMM_V3_TEACHER_CANONICALIZE_MONSTER_ORDER="${SPIRECOMM_V3_TEACHER_CANONICALIZE_MONSTER_ORDER:-1}"
export SPIRECOMM_V3_TEACHER_SEMANTIC_TRANSPOSITION_CACHE="${SPIRECOMM_V3_TEACHER_SEMANTIC_TRANSPOSITION_CACHE:-1}"
export SPIRECOMM_V3_TEACHER_BATCH_POTION_BASELINE_CACHE="${SPIRECOMM_V3_TEACHER_BATCH_POTION_BASELINE_CACHE:-1}"
export SPIRECOMM_V3_TEACHER_SHARED_SURVIVAL_GUARD_CACHE="${SPIRECOMM_V3_TEACHER_SHARED_SURVIVAL_GUARD_CACHE:-1}"
export SPIRECOMM_V3_TEACHER_ROOT_CONTINUATION_CACHE="${SPIRECOMM_V3_TEACHER_ROOT_CONTINUATION_CACHE:-1}"
export SPIRECOMM_V3_TEACHER_NON_POTION_ROOT_CACHE="${SPIRECOMM_V3_TEACHER_NON_POTION_ROOT_CACHE:-1}"
export SPIRECOMM_V3_TEACHER_VECTOR_REWARD_VALUES="${SPIRECOMM_V3_TEACHER_VECTOR_REWARD_VALUES:-1}"
export SPIRECOMM_V3_TEACHER_FAST_ENGINE_CLONE="${SPIRECOMM_V3_TEACHER_FAST_ENGINE_CLONE:-0}"
export SPIRECOMM_TEACHER_SWEEP_BATCH_NON_POTION_ROOTS="${SPIRECOMM_TEACHER_SWEEP_BATCH_NON_POTION_ROOTS:-1}"
export SPIRECOMM_TEACHER_SWEEP_GROUP_SAME_SEED="${SPIRECOMM_TEACHER_SWEEP_GROUP_SAME_SEED:-1}"
export SPIRECOMM_TEACHER_SWEEP_FAST_COMBAT_SPLIT_STEP="${SPIRECOMM_TEACHER_SWEEP_FAST_COMBAT_SPLIT_STEP:-0}"
export SPIRECOMM_TEACHER_SWEEP_BATCH_MIN_CANDIDATES="${SPIRECOMM_TEACHER_SWEEP_BATCH_MIN_CANDIDATES:-2}"
export SPIRECOMM_TEACHER_SWEEP_FULL_SEED_GROUPS="${SPIRECOMM_TEACHER_SWEEP_FULL_SEED_GROUPS:-1}"
export SPIRECOMM_TEACHER_SWEEP_FULL_GROUP_MIN_WORKER_RATIO="${SPIRECOMM_TEACHER_SWEEP_FULL_GROUP_MIN_WORKER_RATIO:-0}"
export SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_SIZE="${SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_SIZE:-1}"
export SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_MIN_CANDIDATES="${SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_MIN_CANDIDATES:-999999}"
export SPIRECOMM_TEACHER_SWEEP_MIN_BATCHES_PER_WORKER="${SPIRECOMM_TEACHER_SWEEP_MIN_BATCHES_PER_WORKER:-0}"
export SPIRECOMM_TEACHER_SWEEP_MAX_SEED_GROUP_CANDIDATES="${SPIRECOMM_TEACHER_SWEEP_MAX_SEED_GROUP_CANDIDATES:-0}"
export SPIRECOMM_TEACHER_SWEEP_PROGRESS_HEARTBEAT_SECONDS="${SPIRECOMM_TEACHER_SWEEP_PROGRESS_HEARTBEAT_SECONDS:-60}"
export SPIRECOMM_FAST_DISABLE_GC="${SPIRECOMM_FAST_DISABLE_GC:-1}"

# This cache is not enabled by default: it matched in small tests but was slower
# in the latest potion sweep benchmark.
export SPIRECOMM_V3_TEACHER_STEP_BRANCH_CACHE="${SPIRECOMM_V3_TEACHER_STEP_BRANCH_CACHE:-0}"

default_workers() {
  local cores
  cores="$(nproc 2>/dev/null || echo 1)"
  local percent="${SWEEP_WORKER_PERCENT:-150}"
  if [[ ! "$percent" =~ ^[0-9]+$ ]] || [[ "$percent" -lt 1 ]]; then percent=150; fi
  local workers=$(( (cores * percent + 99) / 100 ))
  if [[ "$workers" -lt 1 ]]; then workers=1; fi
  echo "$workers"
}
"""


def run_sweep_script() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/_common.sh"

WORKERS="${{WORKERS:-$(default_workers)}}"
START_AT="${{START_AT:-round0}}"
STOP_AFTER="${{STOP_AFTER:-round4}}"
OUT_DIR="${{OUT_DIR:-teacher_sweep_runs/v3_teacher_tempo_power_sweep_v1}}"
LOG_PATH="${{LOG_PATH:-logs/v3_teacher_tempo_power_sweep_v1.log}}"
PID_PATH="${{PID_PATH:-${{LOG_PATH%.log}}.pid}}"
SCHEDULE_ORDER="${{SWEEP_SCHEDULE_ORDER:-seed-major}}"
SEED_MAJOR_CANDIDATE_CHUNK_SIZE="${{SWEEP_SEED_MAJOR_CANDIDATE_CHUNK_SIZE:-0}}"
EXECUTOR_BACKEND="${{SWEEP_EXECUTOR_BACKEND:-mp-pool}}"
TASK_BATCH_SIZE="${{SWEEP_TASK_BATCH_SIZE:-0}}"
MAX_PENDING_FUTURES="${{SWEEP_MAX_PENDING_FUTURES:-0}}"
RESULT_FLUSH_INTERVAL="${{SWEEP_RESULT_FLUSH_INTERVAL:-64}}"
PROGRESS_INTERVAL_TASKS="${{SWEEP_PROGRESS_INTERVAL_TASKS:-25}}"
ROUND1_STAGE_COUNTS="${{ROUND1_STAGE_COUNTS:-100}}"
ROUND1_STAGE_KEEPS="${{ROUND1_STAGE_KEEPS:-32}}"
ROUND0_MAX_FLOOR="${{ROUND0_MAX_FLOOR:-0}}"
ROUND1_MAX_FLOOR="${{ROUND1_MAX_FLOOR:-0}}"
ROUND2_MAX_FLOOR="${{ROUND2_MAX_FLOOR:-0}}"
ROUND3_MAX_FLOOR="${{ROUND3_MAX_FLOOR:-0}}"
ROUND4_MAX_FLOOR="${{ROUND4_MAX_FLOOR:-0}}"
ROUND0_PROXY_BEAM_WIDTH="${{ROUND0_PROXY_BEAM_WIDTH:-0}}"
ROUND0_PROXY_NODE_BUDGET="${{ROUND0_PROXY_NODE_BUDGET:-0}}"
ROUND0_PROXY_MAX_DEPTH="${{ROUND0_PROXY_MAX_DEPTH:-0}}"
ROUND0_PROXY_CONTINUATION_ACTION_CAP="${{ROUND0_PROXY_CONTINUATION_ACTION_CAP:-0}}"
ROUND0_PROXY_ROOT_ONLY="${{ROUND0_PROXY_ROOT_ONLY:-0}}"
ROUND1_PROXY_BEAM_WIDTH="${{ROUND1_PROXY_BEAM_WIDTH:-0}}"
ROUND1_PROXY_NODE_BUDGET="${{ROUND1_PROXY_NODE_BUDGET:-0}}"
ROUND1_PROXY_MAX_DEPTH="${{ROUND1_PROXY_MAX_DEPTH:-0}}"
ROUND1_PROXY_CONTINUATION_ACTION_CAP="${{ROUND1_PROXY_CONTINUATION_ACTION_CAP:-0}}"
ROUND1_PROXY_ROOT_ONLY="${{ROUND1_PROXY_ROOT_ONLY:-0}}"
ROUND2_PROXY_BEAM_WIDTH="${{ROUND2_PROXY_BEAM_WIDTH:-0}}"
ROUND2_PROXY_NODE_BUDGET="${{ROUND2_PROXY_NODE_BUDGET:-0}}"
ROUND2_PROXY_MAX_DEPTH="${{ROUND2_PROXY_MAX_DEPTH:-0}}"
ROUND2_PROXY_CONTINUATION_ACTION_CAP="${{ROUND2_PROXY_CONTINUATION_ACTION_CAP:-0}}"
ROUND3_PROXY_BEAM_WIDTH="${{ROUND3_PROXY_BEAM_WIDTH:-0}}"
ROUND3_PROXY_NODE_BUDGET="${{ROUND3_PROXY_NODE_BUDGET:-0}}"
ROUND3_PROXY_MAX_DEPTH="${{ROUND3_PROXY_MAX_DEPTH:-0}}"
ROUND3_PROXY_CONTINUATION_ACTION_CAP="${{ROUND3_PROXY_CONTINUATION_ACTION_CAP:-0}}"
ROUND4_PROXY_BEAM_WIDTH="${{ROUND4_PROXY_BEAM_WIDTH:-0}}"
ROUND4_PROXY_NODE_BUDGET="${{ROUND4_PROXY_NODE_BUDGET:-0}}"
ROUND4_PROXY_MAX_DEPTH="${{ROUND4_PROXY_MAX_DEPTH:-0}}"
ROUND4_PROXY_CONTINUATION_ACTION_CAP="${{ROUND4_PROXY_CONTINUATION_ACTION_CAP:-0}}"
mkdir -p "$ROOT/logs" "$ROOT/teacher_sweep_runs" "$OUT_DIR"

export SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON='{TEMPO_POWER_RANGES}'

CMD=(
  "$PY" -u "$ROOT/run_v3_teacher_config_sweep_fast.py"
  --output-dir "$OUT_DIR"
  --param-ranges-json "$SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON"
  --seed-start 1
  --workers "$WORKERS"
  --torch-threads 1
  --blas-threads 1
  --metrics-mode floor
  --summary-interval 0
  --progress-interval-tasks "$PROGRESS_INTERVAL_TASKS"
  --round0-count 600
  --round0-max-floor "$ROUND0_MAX_FLOOR"
  --round1-size 128
  --round1-count 100
  --round1-stage-counts "$ROUND1_STAGE_COUNTS"
  --round1-stage-keeps "$ROUND1_STAGE_KEEPS"
  --round1-max-floor "$ROUND1_MAX_FLOOR"
  --round0-proxy-beam-width "$ROUND0_PROXY_BEAM_WIDTH"
  --round0-proxy-node-budget "$ROUND0_PROXY_NODE_BUDGET"
  --round0-proxy-max-depth "$ROUND0_PROXY_MAX_DEPTH"
  --round0-proxy-continuation-action-cap "$ROUND0_PROXY_CONTINUATION_ACTION_CAP"
  --round2-top 32
  --round2-count 200
  --round2-max-floor "$ROUND2_MAX_FLOOR"
  --round3-top 16
  --round3-count 300
  --round3-max-floor "$ROUND3_MAX_FLOOR"
  --round4-top 8
  --round4-count 600
  --round4-max-floor "$ROUND4_MAX_FLOOR"
  --round1-proxy-beam-width "$ROUND1_PROXY_BEAM_WIDTH"
  --round1-proxy-node-budget "$ROUND1_PROXY_NODE_BUDGET"
  --round1-proxy-max-depth "$ROUND1_PROXY_MAX_DEPTH"
  --round1-proxy-continuation-action-cap "$ROUND1_PROXY_CONTINUATION_ACTION_CAP"
  --round2-proxy-beam-width "$ROUND2_PROXY_BEAM_WIDTH"
  --round2-proxy-node-budget "$ROUND2_PROXY_NODE_BUDGET"
  --round2-proxy-max-depth "$ROUND2_PROXY_MAX_DEPTH"
  --round2-proxy-continuation-action-cap "$ROUND2_PROXY_CONTINUATION_ACTION_CAP"
  --round3-proxy-beam-width "$ROUND3_PROXY_BEAM_WIDTH"
  --round3-proxy-node-budget "$ROUND3_PROXY_NODE_BUDGET"
  --round3-proxy-max-depth "$ROUND3_PROXY_MAX_DEPTH"
  --round3-proxy-continuation-action-cap "$ROUND3_PROXY_CONTINUATION_ACTION_CAP"
  --round4-proxy-beam-width "$ROUND4_PROXY_BEAM_WIDTH"
  --round4-proxy-node-budget "$ROUND4_PROXY_NODE_BUDGET"
  --round4-proxy-max-depth "$ROUND4_PROXY_MAX_DEPTH"
  --round4-proxy-continuation-action-cap "$ROUND4_PROXY_CONTINUATION_ACTION_CAP"
  --executor-backend "$EXECUTOR_BACKEND"
  --worker-crash-retries 4
  --worker-crash-retry-worker-scale 0.90
  --schedule-order "$SCHEDULE_ORDER"
  --seed-major-candidate-chunk-size "$SEED_MAJOR_CANDIDATE_CHUNK_SIZE"
  --task-batch-size "$TASK_BATCH_SIZE"
  --max-pending-futures "$MAX_PENDING_FUTURES"
  --result-flush-interval "$RESULT_FLUSH_INTERVAL"
  --result-storage round-aggregate
  --no-write-candidate-configs
  --no-write-candidate-summaries
  --no-write-sweep-results
  --include-default-in-round1
  --start-at "$START_AT"
  --stop-after "$STOP_AFTER"
  --resume
)

if [[ "$ROUND0_PROXY_ROOT_ONLY" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  CMD+=(--round0-proxy-root-only)
fi
if [[ "$ROUND1_PROXY_ROOT_ONLY" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  CMD+=(--round1-proxy-root-only)
fi

printf '%q ' "${{CMD[@]}}" > "${{LOG_PATH%.log}}.command"
printf '\\n' >> "${{LOG_PATH%.log}}.command"

echo "python=$PY"
"$PY" check_bundle.py
echo "output_dir=$OUT_DIR"
echo "log_path=$LOG_PATH"
echo "workers=$WORKERS"
echo "schedule_order=$SCHEDULE_ORDER"
echo "executor_backend=$EXECUTOR_BACKEND"
echo "ranges=$SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON"
echo "round0=default seed1-600"
echo "round1=128 + default seed1-100 stage_counts=$ROUND1_STAGE_COUNTS stage_keeps=$ROUND1_STAGE_KEEPS; round2=top32 seed1-200; round3=top16 seed1-300; round4=top8 seed1-600"
echo "round_max_floor r0=$ROUND0_MAX_FLOOR r1=$ROUND1_MAX_FLOOR r2=$ROUND2_MAX_FLOOR r3=$ROUND3_MAX_FLOOR r4=$ROUND4_MAX_FLOOR"
echo "proxy_round0=$ROUND0_PROXY_BEAM_WIDTH/$ROUND0_PROXY_NODE_BUDGET/$ROUND0_PROXY_MAX_DEPTH cap=$ROUND0_PROXY_CONTINUATION_ACTION_CAP"
echo "proxy_round1=$ROUND1_PROXY_BEAM_WIDTH/$ROUND1_PROXY_NODE_BUDGET/$ROUND1_PROXY_MAX_DEPTH cap=$ROUND1_PROXY_CONTINUATION_ACTION_CAP"
echo "proxy_round2=$ROUND2_PROXY_BEAM_WIDTH/$ROUND2_PROXY_NODE_BUDGET/$ROUND2_PROXY_MAX_DEPTH cap=$ROUND2_PROXY_CONTINUATION_ACTION_CAP"
echo "proxy_round3=$ROUND3_PROXY_BEAM_WIDTH/$ROUND3_PROXY_NODE_BUDGET/$ROUND3_PROXY_MAX_DEPTH cap=$ROUND3_PROXY_CONTINUATION_ACTION_CAP"
echo "proxy_round4=$ROUND4_PROXY_BEAM_WIDTH/$ROUND4_PROXY_NODE_BUDGET/$ROUND4_PROXY_MAX_DEPTH cap=$ROUND4_PROXY_CONTINUATION_ACTION_CAP"
echo "continuation_action_cap=${{SPIRECOMM_V3_TEACHER_CONTINUATION_ACTION_CAP:-0}}"

if [[ "${{FOREGROUND:-0}}" = "1" ]]; then
  "${{CMD[@]}}" 2>&1 | tee "$LOG_PATH"
else
  setsid "${{CMD[@]}}" >> "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "started pid=$(cat "$PID_PATH")"
  echo "tail -f $LOG_PATH"
fi
"""


def run_background_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/_common.sh"
mkdir -p "$ROOT/logs"
WORKERS="${WORKERS:-$(default_workers)}"
log="$ROOT/logs/v3_teacher_tempo_power_sweep_v1.log"
echo "start tempo/power sweep workers=$WORKERS log=$log"
(cd "$ROOT" && WORKERS="$WORKERS" bash scripts/run_tempo_power_sweep.sh)
"""


def run_turbo_background_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/_common.sh"
mkdir -p "$ROOT/logs"

export OUT_DIR="${OUT_DIR:-teacher_sweep_runs/v3_teacher_tempo_power_sweep_turbo_v1}"
export LOG_PATH="${LOG_PATH:-logs/v3_teacher_tempo_power_sweep_turbo_v1.log}"
export PID_PATH="${PID_PATH:-logs/v3_teacher_tempo_power_sweep_turbo_v1.pid}"
export ROUND1_STAGE_COUNTS="${ROUND1_STAGE_COUNTS:-10,25,50,100}"
export ROUND1_STAGE_KEEPS="${ROUND1_STAGE_KEEPS:-64,48,32,32}"
export ROUND0_MAX_FLOOR="${ROUND0_MAX_FLOOR:-0}"
export ROUND1_MAX_FLOOR="${ROUND1_MAX_FLOOR:-0}"
export ROUND2_MAX_FLOOR="${ROUND2_MAX_FLOOR:-0}"
export ROUND3_MAX_FLOOR="${ROUND3_MAX_FLOOR:-0}"
export ROUND4_MAX_FLOOR="${ROUND4_MAX_FLOOR:-0}"
export SPIRECOMM_TEACHER_SWEEP_MERGE_SPLIT_STATES="${SPIRECOMM_TEACHER_SWEEP_MERGE_SPLIT_STATES:-1}"
export SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_SIZE="${SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_SIZE:-1}"
export SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_MIN_CANDIDATES="${SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_MIN_CANDIDATES:-999999}"
export SPIRECOMM_V3_TEACHER_CONTINUATION_ACTION_CAP="${SPIRECOMM_V3_TEACHER_CONTINUATION_ACTION_CAP:-0}"

# Do not reduce teacher search budget in turbo mode.  Approximation is limited
# to equivalence/merge optimizations such as non-adjacent equivalent cards and
# same-state split merging.
export ROUND0_PROXY_BEAM_WIDTH="${ROUND0_PROXY_BEAM_WIDTH:-0}"
export ROUND0_PROXY_NODE_BUDGET="${ROUND0_PROXY_NODE_BUDGET:-0}"
export ROUND0_PROXY_MAX_DEPTH="${ROUND0_PROXY_MAX_DEPTH:-0}"
export ROUND0_PROXY_CONTINUATION_ACTION_CAP="${ROUND0_PROXY_CONTINUATION_ACTION_CAP:-0}"
export ROUND1_PROXY_BEAM_WIDTH="${ROUND1_PROXY_BEAM_WIDTH:-0}"
export ROUND1_PROXY_NODE_BUDGET="${ROUND1_PROXY_NODE_BUDGET:-0}"
export ROUND1_PROXY_MAX_DEPTH="${ROUND1_PROXY_MAX_DEPTH:-0}"
export ROUND1_PROXY_CONTINUATION_ACTION_CAP="${ROUND1_PROXY_CONTINUATION_ACTION_CAP:-0}"
export ROUND1_PROXY_ROOT_ONLY="${ROUND1_PROXY_ROOT_ONLY:-0}"
export ROUND2_PROXY_BEAM_WIDTH="${ROUND2_PROXY_BEAM_WIDTH:-0}"
export ROUND2_PROXY_NODE_BUDGET="${ROUND2_PROXY_NODE_BUDGET:-0}"
export ROUND2_PROXY_MAX_DEPTH="${ROUND2_PROXY_MAX_DEPTH:-0}"
export ROUND2_PROXY_CONTINUATION_ACTION_CAP="${ROUND2_PROXY_CONTINUATION_ACTION_CAP:-0}"
export ROUND3_PROXY_BEAM_WIDTH="${ROUND3_PROXY_BEAM_WIDTH:-0}"
export ROUND3_PROXY_NODE_BUDGET="${ROUND3_PROXY_NODE_BUDGET:-0}"
export ROUND3_PROXY_MAX_DEPTH="${ROUND3_PROXY_MAX_DEPTH:-0}"
export ROUND3_PROXY_CONTINUATION_ACTION_CAP="${ROUND3_PROXY_CONTINUATION_ACTION_CAP:-0}"
export ROUND4_PROXY_BEAM_WIDTH="${ROUND4_PROXY_BEAM_WIDTH:-0}"
export ROUND4_PROXY_NODE_BUDGET="${ROUND4_PROXY_NODE_BUDGET:-0}"
export ROUND4_PROXY_MAX_DEPTH="${ROUND4_PROXY_MAX_DEPTH:-0}"
export ROUND4_PROXY_CONTINUATION_ACTION_CAP="${ROUND4_PROXY_CONTINUATION_ACTION_CAP:-0}"

WORKERS="${WORKERS:-$(default_workers)}"
echo "start TURBO tempo/power sweep workers=$WORKERS log=$LOG_PATH out=$OUT_DIR"
echo "turbo round1 stages=$ROUND1_STAGE_COUNTS keeps=$ROUND1_STAGE_KEEPS"
echo "turbo max_floor r0=$ROUND0_MAX_FLOOR r1=$ROUND1_MAX_FLOOR r2=$ROUND2_MAX_FLOOR r3=$ROUND3_MAX_FLOOR r4=$ROUND4_MAX_FLOOR"
echo "turbo search budget unchanged: beam/node/depth remain teacher defaults unless explicitly overridden"
echo "turbo candidate_policy_cluster size=$SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_SIZE min_candidates=$SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_MIN_CANDIDATES"
(cd "$ROOT" && WORKERS="$WORKERS" bash scripts/run_tempo_power_sweep.sh)
"""


def status_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/teacher_sweep_runs/v3_teacher_tempo_power_sweep_v1}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/v3_teacher_tempo_power_sweep_v1.log}"
echo "== process =="
pgrep -af "run_v3_teacher_config_sweep_fast.py.*$(basename "$OUT_DIR")" || true
echo
echo "== latest log =="
tail -80 "$LOG_PATH" 2>/dev/null || true
echo
echo "== progress =="
PY="${PY:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY="${PYTHON:-python3}"; fi
"$PY" "$ROOT/watch_teacher_sweep_progress.py" --output-dir "$OUT_DIR" --once --top 12 2>/dev/null || true
echo
echo "== baseline =="
"$PY" "$ROOT/scripts/summarize_tempo_power_sweep.py" --output-dir "$OUT_DIR" 2>/dev/null || true
"""


def stop_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_PATH="${PID_PATH:-$ROOT/logs/v3_teacher_tempo_power_sweep_v1.pid}"
if [[ ! -f "$PID_PATH" ]]; then
  echo "no pid file: $PID_PATH"
  exit 0
fi
pid="$(cat "$PID_PATH")"
pgid="$(ps -p "$pid" -o pgid= | tr -d ' ' || true)"
if [[ -z "$pgid" ]]; then
  echo "process not running: pid=$pid"
  exit 0
fi
kill -TERM "-$pgid"
echo "sent TERM to pgid=$pgid pid=$pid"
"""


def install_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_bundle.py
"""


def smoke_script() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/_common.sh"

TMP_ROOT="${{TMPDIR:-/tmp}}/aliyun_teacher_tempo_power_smoke_$$"
rm -rf "$TMP_ROOT"
mkdir -p "$(dirname "$TMP_ROOT")"

export SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON='{TEMPO_POWER_RANGES}'

"$PY" -u "$ROOT/run_v3_teacher_config_sweep_fast.py" \\
  --output-dir "$TMP_ROOT" \\
  --param-ranges-json "$SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON" \\
  --seed-start 1 --workers 1 --torch-threads 1 --blas-threads 1 \\
  --max-floor 4 --max-steps 300 \\
  --metrics-mode floor --summary-interval 1 --progress-interval-tasks 1 \\
  --round0-count 2 \\
  --round1-size 2 --round1-count 1 --round1-stage-counts 1 --round1-stage-keeps 2 \\
  --round2-top 2 --round2-count 1 \\
  --round3-top 1 --round3-count 1 \\
  --round4-top 1 --round4-count 1 \\
  --round1-proxy-beam-width 0 --round1-proxy-node-budget 0 --round1-proxy-max-depth 0 \\
  --round2-proxy-beam-width 0 --round2-proxy-node-budget 0 --round2-proxy-max-depth 0 \\
  --round3-proxy-beam-width 0 --round3-proxy-node-budget 0 --round3-proxy-max-depth 0 \\
  --round4-proxy-beam-width 0 --round4-proxy-node-budget 0 --round4-proxy-max-depth 0 \\
  --include-default-in-round1 \\
  --start-at round0 --stop-after round1 --resume

"$PY" - "$TMP_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "sweep_config.json").read_text(encoding="utf-8"))
expected = {{
    "power_card_constant": [0.0, 20.0],
    "skill_power_turn_constant": [4.0, 16.0],
    "turn_order_decay_per_card": [0.0, 0.4],
    "play_card_constant": [-0.8, 1.2],
    "energy_spent_weight": [-0.6, 0.3],
}}
actual = {{k: list(v) for k, v in (config.get("param_ranges") or {{}}).items()}}
if actual != expected:
    raise SystemExit(f"smoke failed: wrong ranges {{actual!r}}")
if (config.get("rounds") or {{}}).get("round0_default_seed_count") != 2:
    raise SystemExit("smoke failed: round0 count not recorded as 2")
for key in ("round1_proxy_search", "round2_proxy_search", "round3_proxy_search", "round4_proxy_search"):
    if config.get(key):
        raise SystemExit(f"smoke failed: {{key}} is enabled")
leaderboard = root / "leaderboard_round1_seed1.json"
rows = json.loads(leaderboard.read_text(encoding="utf-8"))
if len(rows) < 2:
    raise SystemExit("smoke failed: too few round1 rows")
errors = []
for path in sorted((root / "evals" / "round1_seed1").glob("*/summary.json")):
    summary = json.loads(path.read_text(encoding="utf-8"))
    if int(summary.get("error_count") or 0):
        errors.append((path.parent.name, summary.get("errors") or []))
if errors:
    raise SystemExit("smoke failed: rollout errors found: " + repr(errors[:3]))
print("smoke_ok", "round1_rows", len(rows), "default_done", (root / "leaderboard_round0_default.json").exists())
PY

rm -rf "$TMP_ROOT"
"""


def turbo_smoke_script() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/_common.sh"

TMP_ROOT="${{TMPDIR:-/tmp}}/aliyun_teacher_tempo_power_turbo_smoke_$$"
rm -rf "$TMP_ROOT"
mkdir -p "$(dirname "$TMP_ROOT")"

export SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON='{TEMPO_POWER_RANGES}'
export SPIRECOMM_V3_TEACHER_CONTINUATION_ACTION_CAP="${{SPIRECOMM_V3_TEACHER_CONTINUATION_ACTION_CAP:-0}}"
export SPIRECOMM_V3_TEACHER_CONTINUATION_ALWAYS_KEEP_END="${{SPIRECOMM_V3_TEACHER_CONTINUATION_ALWAYS_KEEP_END:-1}}"

"$PY" -u "$ROOT/run_v3_teacher_config_sweep_fast.py" \\
  --output-dir "$TMP_ROOT" \\
  --param-ranges-json "$SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON" \\
  --seed-start 1 --workers 1 --torch-threads 1 --blas-threads 1 \\
  --max-floor 3 --max-steps 300 \\
  --metrics-mode floor --summary-interval 1 --progress-interval-tasks 1 \\
  --round0-count 2 \\
  --round0-max-floor 4 \\
  --round0-proxy-beam-width 0 --round0-proxy-node-budget 0 --round0-proxy-max-depth 0 --round0-proxy-continuation-action-cap 0 \\
  --round1-size 4 --round1-count 2 --round1-stage-counts 2 --round1-stage-keeps 2 \\
  --round1-max-floor 4 \\
  --round2-top 2 --round2-count 1 \\
  --round3-top 1 --round3-count 1 \\
  --round4-top 1 --round4-count 1 \\
  --round1-proxy-beam-width 0 --round1-proxy-node-budget 0 --round1-proxy-max-depth 0 --round1-proxy-continuation-action-cap 0 \\
  --round2-proxy-beam-width 0 --round2-proxy-node-budget 0 --round2-proxy-max-depth 0 --round2-proxy-continuation-action-cap 0 \\
  --round3-proxy-beam-width 0 --round3-proxy-node-budget 0 --round3-proxy-max-depth 0 --round3-proxy-continuation-action-cap 0 \\
  --round4-proxy-beam-width 0 --round4-proxy-node-budget 0 --round4-proxy-max-depth 0 --round4-proxy-continuation-action-cap 0 \\
  --include-default-in-round1 \\
  --start-at round0 --stop-after round1 --resume

"$PY" - "$TMP_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "sweep_config.json").read_text(encoding="utf-8"))
if config.get("round1_proxy_search"):
    raise SystemExit("turbo smoke failed: round1 proxy search should not be enabled")
if int(config.get("teacher_sweep_candidate_policy_cluster_size") or 0) != 1:
    raise SystemExit("turbo smoke failed: candidate policy clustering should be disabled by default")
leaderboard = root / "leaderboard_round1_seed2.json"
rows = json.loads(leaderboard.read_text(encoding="utf-8"))
if len(rows) < 2:
    raise SystemExit("turbo smoke failed: too few round1 rows")
errors = []
for path in sorted((root / "evals" / "round1_seed2").glob("*/summary.json")):
    summary = json.loads(path.read_text(encoding="utf-8"))
    if int(summary.get("error_count") or 0):
        errors.append((path.parent.name, summary.get("errors") or []))
if errors:
    raise SystemExit("turbo smoke failed: rollout errors found: " + repr(errors[:3]))
print("turbo_smoke_ok", "round1_rows", len(rows), "proxy", config.get("round1_proxy_search"))
PY

rm -rf "$TMP_ROOT"
"""


def summarize_script() -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_aggregate(path: Path, candidate_id: str) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("_candidate_id") or record.get("candidate_id") or "") != candidate_id:
                continue
            rows[int(record["seed"])] = record
    return rows


def load_candidate(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows[int(record["seed"])] = record
    return rows


def summarize(rows: dict[int, dict], max_seed: int) -> dict[str, float | int]:
    selected = [rows[seed] for seed in sorted(rows) if seed <= max_seed]
    if not selected:
        return {"count": 0}
    floors = [float(row.get("floor") or 0.0) for row in selected]
    return {
        "count": len(selected),
        "mean_floor": sum(floors) / len(floors),
        "win_count": sum(1 for row in selected if bool(row.get("won"))),
        "death_count": sum(1 for row in selected if bool(row.get("dead"))),
        "error_count": sum(1 for row in selected if row.get("error")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("teacher_sweep_runs/v3_teacher_tempo_power_sweep_v1"))
    args = parser.parse_args()
    aggregate_path = args.output_dir / "evals" / "round0_default" / "results.jsonl"
    candidate_path = args.output_dir / "evals" / "round0_default" / "default" / "results.jsonl"
    rows = load_aggregate(aggregate_path, "default")
    if not rows:
        rows = load_candidate(candidate_path)
    print("baseline_aggregate_path", aggregate_path)
    print("baseline_candidate_path", candidate_path)
    print("baseline_seed1_300", json.dumps(summarize(rows, 300), ensure_ascii=False, sort_keys=True))
    print("baseline_seed1_600", json.dumps(summarize(rows, 600), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
"""


def check_bundle() -> str:
    return f"""from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RANGES = json.loads('{TEMPO_POWER_RANGES}')


def main() -> None:
    import torch

    from spirecomm.ai.v3_combat_teacher import (
        TEACHER_VERSION,
        _teacher_batch_linear_reward_compatible,
        default_teacher_config,
        teacher_config_from_mapping,
    )
    from spirecomm.native_sim_v3.content.characters import starting_profile

    required = [
        "models/combat.pt",
        "models/v3_combat_scorer.pt",
        "models/card_reward.pt",
        "models/boss_relic.pt",
        "models/campfire.pt",
        "models/event_choice.pt",
        "models/map_choice.pt",
        "models/potion_use.pt",
        "models/purge_target.pt",
        "models/shop_choice.pt",
        "models/shop_choice_prior_delta.pt",
        "models/upgrade_target.pt",
        "spirecomm/native_sim_v3/reference/decompiled_sts/com/megacrit/cardcrawl/characters/Ironclad.java",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("missing required files: " + ", ".join(missing))

    cfg = default_teacher_config()
    if (cfg.beam_width, cfg.node_budget_per_root, cfg.max_depth) != (24, 768, 20):
        raise SystemExit(f"unexpected search defaults: {{cfg.beam_width}}/{{cfg.node_budget_per_root}}/{{cfg.max_depth}}")
    expected_defaults = {{
        "power_card_constant": 10.0,
        "skill_power_turn_constant": 10.0,
        "turn_order_decay_per_card": 0.2,
        "play_card_constant": 0.0,
        "energy_spent_weight": 0.0,
    }}
    for name, expected in expected_defaults.items():
        actual = float(getattr(cfg, name))
        if abs(actual - expected) > 1e-9:
            raise SystemExit(f"unexpected default {{name}}={{actual}} expected={{expected}}")
    override = teacher_config_from_mapping({{"teacher_config": {{name: values[0] for name, values in RANGES.items()}}}})
    for name, values in RANGES.items():
        if abs(float(getattr(override, name)) - float(values[0])) > 1e-9:
            raise SystemExit(f"override failed for {{name}}")
    high = teacher_config_from_mapping({{"teacher_config": {{name: values[1] for name, values in RANGES.items()}}}})
    if not _teacher_batch_linear_reward_compatible([override, cfg, high]):
        raise SystemExit("tempo/power configs unexpectedly cannot use linear many-config reward cache")
    profile = starting_profile("IRONCLAD")
    print(f"root={{ROOT}}")
    print(f"torch={{torch.__version__}}")
    print(f"teacher_version={{TEACHER_VERSION}}")
    print(f"search_defaults=beam:{{cfg.beam_width}} node:{{cfg.node_budget_per_root}} depth:{{cfg.max_depth}}")
    print(f"lethal={{cfg.lethal_check_node_budget}}/{{cfg.lethal_block_suppression_factor}}")
    print(f"tempo_power_defaults={{expected_defaults}}")
    print(f"tempo_power_ranges={{RANGES}}")
    print(f"ironclad_hp={{profile.current_hp}}/{{profile.max_hp}}")
    print("bundle_check=ok")


if __name__ == "__main__":
    main()
"""


def readme() -> str:
    return """# Aliyun Tempo/Power Teacher Sweep v1

This bundle sweeps five combat teacher tempo/power coefficients:

```json
{"power_card_constant":[0,20],"skill_power_turn_constant":[4,16],"turn_order_decay_per_card":[0,0.4],"play_card_constant":[-0.8,1.2],"energy_spent_weight":[-0.6,0.3]}
```

The copied code includes the current accepted defaults, including:

- teacher search budget `beam_width=24`, `node_budget_per_root=768`, `max_depth=20`
- lethal guard `lethal_check_node_budget=32`, `lethal_block_suppression_factor=0.75`
- potion defaults currently in code, including elite factor `1.3` and potion cost scale `1.1`
- equivalent-card dedupe, shared survival guard cache, full same-seed candidate grouping, and current non-potion batch/caching speedups
- tempo/power candidates are checked to be compatible with the linear many-config reward cache

## Install

```bash
cd aliyun_teacher_tempo_power_sweep_v1
bash scripts/install_deps.sh
```

## Smoke

```bash
bash scripts/smoke_tempo_power_sweep_seed1.sh
```

## Run

Background:

```bash
WORKERS=$(nproc) bash run_tempo_power_sweep_background.sh
```

Foreground:

```bash
FOREGROUND=1 WORKERS=$(nproc) bash scripts/run_tempo_power_sweep.sh
```

Resume from a later round:

```bash
START_AT=round3 STOP_AFTER=round4 WORKERS=$(nproc) bash run_tempo_power_sweep_background.sh
```

Equivalence-merge turbo mode:

```bash
WORKERS=$(nproc) bash run_tempo_power_sweep_turbo_background.sh
```

Turbo mode writes to `teacher_sweep_runs/v3_teacher_tempo_power_sweep_turbo_v1`
by default, so it does not mix with the exact run. It keeps the teacher search
budget unchanged (`beam_width=24`, `node_budget_per_root=768`, `max_depth=20`).
The only default approximations are equivalence/merge optimizations:

- non-adjacent equivalent-card action dedupe, except known hand-order/RNG-sensitive hands
- split-state merging after candidate divergence
- candidate policy clustering is disabled by default; opt in only by setting
  `SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_SIZE>1`
- round1 staged seed pruning: seed `10,25,50,100`, keep `64,48,32,32`

Do not set `ROUND*_PROXY_*`, `ROUND*_MAX_FLOOR`, or
`SPIRECOMM_V3_TEACHER_CONTINUATION_ACTION_CAP` unless you intentionally want to
reduce search/evaluation budget.

## Monitor

```bash
bash scripts/status.sh
tail -f logs/v3_teacher_tempo_power_sweep_v1.log
python watch_teacher_sweep_progress.py --output-dir teacher_sweep_runs/v3_teacher_tempo_power_sweep_v1 --once --top 12
python scripts/summarize_tempo_power_sweep.py --output-dir teacher_sweep_runs/v3_teacher_tempo_power_sweep_v1
```

## Search Plan

- `round0`: current default baseline, seed1-600.
- `round1`: 128 Latin-hypercube candidates plus default anchor, seed1-100, no early-stage pruning, promote top32.
- `round2`: top32, seed1-200, promote top16.
- `round3`: top16, seed1-300, promote top8.
- `round4`: top8, seed1-600.
- No proxy search overrides are used in any round.

Expected candidate-seed tasks, excluding already resumable work:

- round0: 600
- round1: 12,900 including default anchor
- round2: 6,400
- round3: 4,800
- round4: 4,800
- total: 29,500

The default scheduler is `seed-major` because this bucket changes combat scoring
for all candidates and can reuse same-seed branching work. If it is slower on
your server, switch at launch time without changing results:

```bash
SWEEP_SCHEDULE_ORDER=candidate-major WORKERS=$(nproc) bash run_tempo_power_sweep_background.sh
```
"""


def build(out_dir: Path) -> Path:
    out_dir = out_dir.expanduser().resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    copy_path(ROOT / "spirecomm", out_dir / "spirecomm")
    for rel in TOP_LEVEL_FILES:
        copy_path(ROOT / rel, out_dir / rel)
    for rel in MODEL_FILES:
        copy_path(ROOT / rel, out_dir / rel)

    write_text(out_dir / "requirements.txt", "numpy\ntorch\n")
    write_text(out_dir / "README.md", readme())
    write_text(out_dir / "check_bundle.py", check_bundle())
    write_text(out_dir / "scripts" / "_common.sh", script_common(), executable=True)
    write_text(out_dir / "scripts" / "install_deps.sh", install_script(), executable=True)
    write_text(out_dir / "scripts" / "run_tempo_power_sweep.sh", run_sweep_script(), executable=True)
    write_text(out_dir / "scripts" / "smoke_tempo_power_sweep_seed1.sh", smoke_script(), executable=True)
    write_text(out_dir / "scripts" / "smoke_tempo_power_sweep_turbo_seed1.sh", turbo_smoke_script(), executable=True)
    write_text(out_dir / "scripts" / "status.sh", status_script(), executable=True)
    write_text(out_dir / "scripts" / "stop_tempo_power_sweep.sh", stop_script(), executable=True)
    write_text(out_dir / "scripts" / "summarize_tempo_power_sweep.py", summarize_script(), executable=True)
    write_text(out_dir / "run_tempo_power_sweep_background.sh", run_background_script(), executable=True)
    write_text(out_dir / "run_tempo_power_sweep_turbo_background.sh", run_turbo_background_script(), executable=True)

    tar_path = out_dir.with_suffix(".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return tar_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Aliyun bundle for tempo/power teacher coefficient sweep.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    tar_path = build(args.out_dir)
    print(f"bundle_dir={args.out_dir.expanduser().resolve()}")
    print(f"tar_path={tar_path}")


if __name__ == "__main__":
    main()
