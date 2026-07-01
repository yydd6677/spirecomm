#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("/home/yydd/下载/aliyun_potion_sweep_v1")

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
    "scripts/v3_combat/build_v3_first_combat_snapshots.py",
    "scripts/v3_combat/evaluate_v3_rollout_batch.py",
    "scripts/v3_combat/run_v3_teacher_config_sweep.py",
    "scripts/v3_combat/run_v3_teacher_config_sweep_fast.py",
    "scripts/v3_combat/watch_teacher_sweep_progress.py",
    "setup.py",
    "README.md",
    "LICENSE",
)

POTION_RANGES = (
    '{"potion_monster_room_reward_factor":[0.15,0.55],'
    '"potion_elite_room_reward_factor":[0.9,1.6],'
    '"potion_boss_room_reward_factor":[1.5,2.8],'
    '"potion_cost_scale":[0.7,1.3]}'
)


def ignore_patterns(_path: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".pytest_cache", ".mypy_cache"}
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo")))
    return ignored


def copy_path(src: Path, dst: Path) -> None:
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
    return r'''#!/usr/bin/env bash
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

default_workers() {
  local cores
  cores="$(nproc 2>/dev/null || echo 1)"
  local percent="${SWEEP_WORKER_PERCENT:-150}"
  if [[ ! "$percent" =~ ^[0-9]+$ ]] || [[ "$percent" -lt 1 ]]; then percent=150; fi
  local workers=$(( (cores * percent + 99) / 100 ))
  if [[ "$workers" -lt 1 ]]; then workers=1; fi
  echo "$workers"
}
'''


def run_potion_script() -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/_common.sh"

WORKERS="${{WORKERS:-$(default_workers)}}"
SEED_MAJOR_CANDIDATE_CHUNK_SIZE="${{SWEEP_SEED_MAJOR_CANDIDATE_CHUNK_SIZE:-0}}"
SCHEDULE_ORDER="${{SWEEP_SCHEDULE_ORDER:-candidate-major}}"
EXECUTOR_BACKEND="${{SWEEP_EXECUTOR_BACKEND:-mp-pool}}"
TASK_BATCH_SIZE="${{SWEEP_TASK_BATCH_SIZE:-0}}"
MAX_PENDING_FUTURES="${{SWEEP_MAX_PENDING_FUTURES:-0}}"
RESULT_FLUSH_INTERVAL="${{SWEEP_RESULT_FLUSH_INTERVAL:-64}}"
PROGRESS_INTERVAL_TASKS="${{SWEEP_PROGRESS_INTERVAL_TASKS:-25}}"
START_AT="${{START_AT:-round0}}"
STOP_AFTER="${{STOP_AFTER:-round4}}"
OUT_DIR="${{OUT_DIR:-teacher_sweep_runs/v3_teacher_potion_sweep_v1}}"
LOG_PATH="${{LOG_PATH:-logs/v3_teacher_potion_sweep_v1.log}}"
mkdir -p "$ROOT/logs" "$ROOT/teacher_sweep_runs" "$OUT_DIR"

export SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON='{POTION_RANGES}'

CMD=(
  "$PY" -u "$ROOT/scripts/v3_combat/run_v3_teacher_config_sweep_fast.py"
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
  --round1-size 64
  --round1-count 100
  --round1-stage-counts 100
  --round1-stage-keeps 16
  --round2-top 16
  --round2-count 200
  --round3-top 10
  --round3-count 300
  --round4-top 6
  --round4-count 600
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

printf '%q ' "${{CMD[@]}}" > "${{LOG_PATH%.log}}.command"
printf '\\n' >> "${{LOG_PATH%.log}}.command"

echo "python=$PY"
"$PY" check_bundle.py
echo "output_dir=$OUT_DIR"
echo "log_path=$LOG_PATH"
echo "workers=$WORKERS"
echo "schedule_order=$SCHEDULE_ORDER"
echo "seed_major_candidate_chunk_size=$SEED_MAJOR_CANDIDATE_CHUNK_SIZE"
echo "executor_backend=$EXECUTOR_BACKEND"
echo "task_batch_size=$TASK_BATCH_SIZE"
echo "max_pending_futures=$MAX_PENDING_FUTURES"
echo "result_flush_interval=$RESULT_FLUSH_INTERVAL"
echo "progress_interval_tasks=$PROGRESS_INTERVAL_TASKS"
echo "round0 baseline=seed1-600"
echo "round1=64 candidates + default anchor seed1-100; round2=top16 seed1-200; round3=top10 seed1-300; round4=top6 seed1-600"
echo "potion ranges=$SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON"
echo "full teacher search defaults are used in every round"

if [[ "${{FOREGROUND:-0}}" = "1" ]]; then
  "${{CMD[@]}}" 2>&1 | tee "$LOG_PATH"
else
  setsid "${{CMD[@]}}" >> "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > logs/v3_teacher_potion_sweep_v1.pid
  echo "started pid=$(cat logs/v3_teacher_potion_sweep_v1.pid)"
  echo "tail -f $LOG_PATH"
fi
'''


def run_background_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/_common.sh"
mkdir -p "$ROOT/logs"
WORKERS="${WORKERS:-$(default_workers)}"
log="$ROOT/logs/v3_teacher_potion_sweep_v1.log"
echo "start potion sweep workers=$WORKERS log=$log"
(cd "$ROOT" && WORKERS="$WORKERS" bash scripts/run_potion_sweep.sh)
'''


def status_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "== process =="
pgrep -af 'scripts/v3_combat/run_v3_teacher_config_sweep_fast.py.*v3_teacher_potion_sweep_v1' || true
echo
echo "== latest log =="
tail -80 "$ROOT/logs/v3_teacher_potion_sweep_v1.log" 2>/dev/null || true
echo
echo "== progress =="
PY="${PY:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY="${PYTHON:-python3}"; fi
"$PY" "$ROOT/scripts/v3_combat/watch_teacher_sweep_progress.py" --output-dir "$ROOT/teacher_sweep_runs/v3_teacher_potion_sweep_v1" --once --top 12 2>/dev/null || true
echo
echo "== baseline summary =="
"$PY" "$ROOT/scripts/summarize_potion_sweep.py" --output-dir "$ROOT/teacher_sweep_runs/v3_teacher_potion_sweep_v1" 2>/dev/null || true
'''


def stop_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_PATH="$ROOT/logs/v3_teacher_potion_sweep_v1.pid"
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
'''


def install_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python3 check_bundle.py
'''


def smoke_script() -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/_common.sh"

TMP_ROOT="${{TMPDIR:-/tmp}}/aliyun_potion_sweep_smoke_$$"
rm -rf "$TMP_ROOT"
mkdir -p "$(dirname "$TMP_ROOT")"

export SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON='{POTION_RANGES}'

"$PY" -u "$ROOT/scripts/v3_combat/run_v3_teacher_config_sweep_fast.py" \\
  --output-dir "$TMP_ROOT" \\
  --param-ranges-json "$SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON" \\
  --seed-start 1 --workers 1 --torch-threads 1 --blas-threads 1 \\
  --max-floor 3 --max-steps 300 \\
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

"$PY" "$ROOT/scripts/summarize_potion_sweep.py" --output-dir "$TMP_ROOT"
"$PY" - "$TMP_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "sweep_config.json").read_text(encoding="utf-8"))
if (config.get("rounds") or dict()).get("round0_default_seed_count") != 2:
    raise SystemExit("smoke failed: round0 count not recorded as 2")
if config.get("round1_proxy_search") or config.get("round2_proxy_search"):
    raise SystemExit("smoke failed: proxy search is enabled")
leaderboard = root / "leaderboard_round1_seed1.json"
rows = json.loads(leaderboard.read_text(encoding="utf-8"))
if len(rows) < 2:
    raise SystemExit("smoke failed: too few round1 rows")
print("smoke_ok", "round1_rows", len(rows))
PY

rm -rf "$TMP_ROOT"
'''


def benchmark_scheduler_script() -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/_common.sh"

OUT_DIR="${{OUT_DIR:-teacher_sweep_runs/v3_teacher_potion_sweep_v1}}"
BENCH_ROOT="${{BENCH_ROOT:-${{TMPDIR:-/tmp}}/potion_sweep_scheduler_bench_$$}}"
BENCH_WORKERS="${{BENCH_WORKERS:-${{WORKERS:-32}}}}"
BENCH_TIMEOUT_SECONDS="${{BENCH_TIMEOUT_SECONDS:-180}}"
BENCH_PROGRESS_INTERVAL="${{BENCH_PROGRESS_INTERVAL:-10}}"
BENCH_ROUND4_COUNT="${{BENCH_ROUND4_COUNT:-600}}"
mkdir -p "$BENCH_ROOT"

if [[ ! -f "$OUT_DIR/leaderboard_round3_seed300.json" ]]; then
  echo "missing $OUT_DIR/leaderboard_round3_seed300.json" >&2
  exit 2
fi

export SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON='{POTION_RANGES}'

run_case() {{
  local name="$1"; shift
  local case_dir="$BENCH_ROOT/$name"
  rm -rf "$case_dir"
  mkdir -p "$case_dir"
  cp "$OUT_DIR/leaderboard_round3_seed300.json" "$case_dir/leaderboard_round3_seed300.json"
  if [[ -f "$OUT_DIR/evals/round4_seed600/results.jsonl" ]]; then
    mkdir -p "$case_dir/evals/round4_seed600"
    cp "$OUT_DIR/evals/round4_seed600/results.jsonl" "$case_dir/evals/round4_seed600/results.jsonl"
  fi
  echo "== $name =="
  echo "case_dir=$case_dir"
  set +e
  timeout -s TERM "$BENCH_TIMEOUT_SECONDS" "$PY" -u "$ROOT/scripts/v3_combat/run_v3_teacher_config_sweep_fast.py" \\
    --output-dir "$case_dir" \\
    --param-ranges-json "$SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON" \\
    --seed-start 1 \\
    --workers "$BENCH_WORKERS" \\
    --torch-threads 1 \\
    --blas-threads 1 \\
    --metrics-mode floor \\
    --summary-interval 0 \\
    --progress-interval-tasks "$BENCH_PROGRESS_INTERVAL" \\
    --round0-count 600 \\
    --round1-size 64 \\
    --round1-count 100 \\
    --round1-stage-counts 100 \\
    --round1-stage-keeps 16 \\
    --round2-top 16 \\
    --round2-count 200 \\
    --round3-top 10 \\
    --round3-count 300 \\
    --round4-top 6 \\
    --round4-count "$BENCH_ROUND4_COUNT" \\
    --round1-proxy-beam-width 0 --round1-proxy-node-budget 0 --round1-proxy-max-depth 0 \\
    --round2-proxy-beam-width 0 --round2-proxy-node-budget 0 --round2-proxy-max-depth 0 \\
    --round3-proxy-beam-width 0 --round3-proxy-node-budget 0 --round3-proxy-max-depth 0 \\
    --round4-proxy-beam-width 0 --round4-proxy-node-budget 0 --round4-proxy-max-depth 0 \\
    --worker-crash-retries 0 \\
    --result-storage round-aggregate \\
    --no-write-candidate-configs \\
    --no-write-candidate-summaries \\
    --no-write-sweep-results \\
    --include-default-in-round1 \\
    --start-at round4 \\
    --stop-after round4 \\
    --resume "$@" 2>&1 | tee "$BENCH_ROOT/$name.log"
  local code="${{PIPESTATUS[0]}}"
  set -e
  echo "exit_code=$code"
  grep -E 'scheduler|completed_tasks|worker_pool_failure|Traceback|error' "$BENCH_ROOT/$name.log" | tail -20 || true
  echo
}}

echo "bench_root=$BENCH_ROOT"
echo "bench_workers=$BENCH_WORKERS timeout=${{BENCH_TIMEOUT_SECONDS}}s"
run_case candidate_mp --executor-backend mp-pool --schedule-order candidate-major --seed-major-candidate-chunk-size 0 --task-batch-size 0 --max-pending-futures 0 --result-flush-interval 64
run_case candidate_proc --executor-backend process-pool --schedule-order candidate-major --seed-major-candidate-chunk-size 0 --task-batch-size 0 --max-pending-futures 0 --result-flush-interval 64
run_case seed_chunk1_mp --executor-backend mp-pool --schedule-order seed-major --seed-major-candidate-chunk-size 1 --task-batch-size 0 --max-pending-futures 0 --result-flush-interval 64
run_case seed_auto_mp --executor-backend mp-pool --schedule-order seed-major --seed-major-candidate-chunk-size 0 --task-batch-size 0 --max-pending-futures 0 --result-flush-interval 64

echo "summary:"
for log in "$BENCH_ROOT"/*.log; do
  printf '%s ' "$(basename "$log" .log)"
  grep -Eo 'new_rate=[0-9.]+/s' "$log" | tail -1 || true
done
'''


def summarize_script() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(path: Path) -> dict[int, dict]:
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
    parser.add_argument("--output-dir", type=Path, default=Path("teacher_sweep_runs/v3_teacher_potion_sweep_v1"))
    args = parser.parse_args()
    result_path = args.output_dir / "evals" / "round0_default" / "default" / "results.jsonl"
    rows = load_results(result_path)
    print("baseline_result_path", result_path)
    print("baseline_seed1_300", json.dumps(summarize(rows, 300), ensure_ascii=False, sort_keys=True))
    print("baseline_seed1_600", json.dumps(summarize(rows, 600), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def check_bundle() -> str:
    return f'''from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    import torch

    from spirecomm.ai.v3_combat_teacher import TEACHER_VERSION, default_teacher_config, teacher_config_from_mapping
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
        "potion_monster_room_reward_factor": 0.3,
        "potion_elite_room_reward_factor": 1.2,
        "potion_boss_room_reward_factor": 2.0,
        "potion_cost_scale": 1.0,
        "potion_buff_adjustment_scale": 1.0,
        "potion_generation_adjustment_scale": 1.0,
    }}
    for name, expected in expected_defaults.items():
        actual = float(getattr(cfg, name))
        if abs(actual - expected) > 1e-9:
            raise SystemExit(f"unexpected default {{name}}={{actual}} expected={{expected}}")
    ranges = json.loads('{POTION_RANGES}')
    override = teacher_config_from_mapping({{"teacher_config": {{name: values[0] for name, values in ranges.items()}}}})
    for name, values in ranges.items():
        if abs(float(getattr(override, name)) - float(values[0])) > 1e-9:
            raise SystemExit(f"override failed for {{name}}")
    profile = starting_profile("IRONCLAD")
    print(f"root={{ROOT}}")
    print(f"torch={{torch.__version__}}")
    print(f"teacher_version={{TEACHER_VERSION}}")
    print(f"search_defaults=beam:{{cfg.beam_width}} node:{{cfg.node_budget_per_root}} depth:{{cfg.max_depth}}")
    print(f"lethal={{cfg.lethal_check_node_budget}}/{{cfg.lethal_block_suppression_factor}}")
    print(f"power.Strength={{cfg.player_power_weights['Strength']}}")
    print(f"potion_defaults={{expected_defaults}}")
    print(f"potion_ranges={{ranges}}")
    print(f"ironclad_hp={{profile.current_hp}}/{{profile.max_hp}}")
    print("bundle_check=ok")


if __name__ == "__main__":
    main()
'''


def readme() -> str:
    return """# Aliyun Potion Teacher Sweep v1

This bundle sweeps four potion teacher coefficients with all currently accepted best defaults already baked into the copied repo code:

- search budget default: `beam_width=24`, `node_budget_per_root=768`, `max_depth=20`
- lethal default: `lethal_check_node_budget=32`, `lethal_block_suppression_factor=0.75`
- base reward and player-power sweep-best dictionaries are the current code defaults

Swept ranges:

```json
{"potion_monster_room_reward_factor":[0.15,0.55],"potion_elite_room_reward_factor":[0.9,1.6],"potion_boss_room_reward_factor":[1.5,2.8],"potion_cost_scale":[0.7,1.3]}
```

## Install

```bash
cd aliyun_potion_sweep_v1
bash scripts/install_deps.sh
```

## Smoke

```bash
bash scripts/smoke_potion_sweep_seed1.sh
```

## Run

```bash
WORKERS=$(nproc) bash run_potion_sweep_background.sh
```

`SWEEP_SCHEDULE_ORDER` defaults to `candidate-major`. The alternatives remain
available through environment variables because the fastest setting depends on
the remaining round4 task mix:

```bash
SWEEP_SCHEDULE_ORDER=seed-major SWEEP_SEED_MAJOR_CANDIDATE_CHUNK_SIZE=1 ...
SWEEP_EXECUTOR_BACKEND=process-pool ...
```

If speed regresses, run a non-mutating scheduler benchmark against a copy of the
current round4 progress:

```bash
BENCH_WORKERS=64 BENCH_TIMEOUT_SECONDS=180 bash scripts/benchmark_potion_sweep_schedulers.sh
```

Foreground:

```bash
FOREGROUND=1 WORKERS=$(nproc) bash scripts/run_potion_sweep.sh
```

## Monitor

```bash
bash scripts/status.sh
tail -f logs/v3_teacher_potion_sweep_v1.log
python3 scripts/v3_combat/watch_teacher_sweep_progress.py --output-dir teacher_sweep_runs/v3_teacher_potion_sweep_v1 --once --top 12
python3 scripts/summarize_potion_sweep.py --output-dir teacher_sweep_runs/v3_teacher_potion_sweep_v1
```

The baseline is intentionally seed1-600. The summarizer prints both seed1-300 and seed1-600 baseline metrics.

## Search Plan

- `round0`: current default baseline, seed1-600.
- `round1`: 64 Latin-hypercube candidates plus default anchor, seed1-100, promote top16.
- `round2`: top16, seed1-200, promote top10.
- `round3`: top10, seed1-300, promote top6.
- `round4`: top6, seed1-600.
- No proxy search overrides are used in any round.
- `--resume` is enabled.

## Outputs To Download

At minimum:

- `teacher_sweep_runs/v3_teacher_potion_sweep_v1/final_leaderboard.json`
- `teacher_sweep_runs/v3_teacher_potion_sweep_v1/leaderboard_round*.json`
- `teacher_sweep_runs/v3_teacher_potion_sweep_v1/sweep_config.json`
- `logs/v3_teacher_potion_sweep_v1.log`

Download the whole `teacher_sweep_runs/v3_teacher_potion_sweep_v1` folder for exact paired seed analysis.
"""


def build(out_dir: Path) -> Path:
    out_dir = out_dir.expanduser().resolve()
    preserved_sweep_runs: Path | None = None
    existing_sweep_runs = out_dir / "teacher_sweep_runs"
    if existing_sweep_runs.exists():
        preserved_root = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}_preserve_", dir=str(out_dir.parent)))
        preserved_sweep_runs = preserved_root / "teacher_sweep_runs"
        shutil.copytree(existing_sweep_runs, preserved_sweep_runs, ignore=ignore_patterns)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    copy_path(ROOT / "spirecomm", out_dir / "spirecomm")
    for rel in TOP_LEVEL_FILES:
        src = ROOT / rel
        if src.exists():
            copy_path(src, out_dir / rel)
    for rel in MODEL_FILES:
        copy_path(ROOT / rel, out_dir / rel)

    write_text(out_dir / "requirements.txt", "numpy\ntorch\n")
    write_text(out_dir / "README_SERVER.md", readme())
    write_text(out_dir / "check_bundle.py", check_bundle())
    write_text(out_dir / "scripts" / "_common.sh", script_common(), executable=True)
    write_text(out_dir / "scripts" / "install_deps.sh", install_script(), executable=True)
    write_text(out_dir / "scripts" / "run_potion_sweep.sh", run_potion_script(), executable=True)
    write_text(out_dir / "scripts" / "smoke_potion_sweep_seed1.sh", smoke_script(), executable=True)
    write_text(out_dir / "scripts" / "benchmark_potion_sweep_schedulers.sh", benchmark_scheduler_script(), executable=True)
    write_text(out_dir / "scripts" / "status.sh", status_script(), executable=True)
    write_text(out_dir / "scripts" / "stop_potion_sweep.sh", stop_script(), executable=True)
    write_text(out_dir / "scripts" / "summarize_potion_sweep.py", summarize_script(), executable=True)
    write_text(out_dir / "run_potion_sweep_background.sh", run_background_script(), executable=True)
    if preserved_sweep_runs is not None and preserved_sweep_runs.exists():
        shutil.copytree(preserved_sweep_runs, out_dir / "teacher_sweep_runs", ignore=ignore_patterns)
        shutil.rmtree(preserved_sweep_runs.parent, ignore_errors=True)

    tar_path = out_dir.with_suffix(".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return tar_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Aliyun bundle for potion teacher coefficient sweep.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    tar_path = build(args.out_dir)
    print(f"bundle_dir={args.out_dir.expanduser().resolve()}")
    print(f"tar_path={tar_path}")


if __name__ == "__main__":
    main()
