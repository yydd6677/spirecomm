#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("/home/yydd/下载/aliyun_strength_fullsearch_v2")

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
    "evaluate_v3_rollout_batch.py",
    "run_v3_teacher_config_sweep.py",
    "run_v3_teacher_config_sweep_fast.py",
    "watch_teacher_sweep_progress.py",
    "setup.py",
    "README.md",
    "LICENSE",
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


def run_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

sanitize_thread_var() {
  local name="$1"
  local fallback="$2"
  local value="${!name:-$fallback}"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 1 ]]; then
    value="$fallback"
  fi
  export "$name=$value"
}

default_workers() {
  local cores
  cores="$(nproc 2>/dev/null || echo 1)"
  local percent="${SWEEP_WORKER_PERCENT:-150}"
  if [[ ! "$percent" =~ ^[0-9]+$ ]] || [[ "$percent" -lt 1 ]]; then percent=150; fi
  local workers=$(( (cores * percent + 99) / 100 ))
  if [[ "$workers" -lt 1 ]]; then workers=1; fi
  echo "$workers"
}

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
sanitize_thread_var OMP_NUM_THREADS 1
sanitize_thread_var MKL_NUM_THREADS 1
sanitize_thread_var OPENBLAS_NUM_THREADS 1
sanitize_thread_var NUMEXPR_NUM_THREADS 1
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"

WORKERS="${WORKERS:-$(default_workers)}"
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

echo "python=$PYTHON"
"$PYTHON" check_bundle.py

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
  --summary-interval 0
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
'''


def check_bundle() -> str:
    return r'''from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    import torch

    from spirecomm.ai.v3_combat_teacher import TEACHER_VERSION, default_teacher_config
    from spirecomm.native_sim_v3.content.characters import starting_profile

    required_models = [
        "boss_relic.pt",
        "campfire.pt",
        "card_reward.pt",
        "combat.pt",
        "event_choice.pt",
        "map_choice.pt",
        "potion_use.pt",
        "purge_target.pt",
        "shop_choice_prior_delta.pt",
        "upgrade_target.pt",
        "v3_combat_scorer.pt",
    ]
    missing = [name for name in required_models if not (ROOT / "models" / name).exists()]
    if missing:
        raise SystemExit(f"missing required model files: {missing}")

    cfg = default_teacher_config()
    profile = starting_profile("IRONCLAD")
    script = (ROOT / "run_strength_sweep.sh").read_text(encoding="utf-8")
    required_snippets = [
        "--round1-count \"$ROUND1_COUNT\"",
        "--round2-count \"$ROUND2_COUNT\"",
        "--round3-count \"$ROUND3_COUNT\"",
        "--round4-count \"$ROUND4_COUNT\"",
        "--round1-proxy-beam-width 0",
        "--round2-proxy-node-budget 0",
        "--round3-proxy-max-depth 0",
    ]
    missing_snippets = [snippet for snippet in required_snippets if snippet not in script]
    if missing_snippets:
        raise SystemExit("run_strength_sweep.sh is not full-search v2: " + json.dumps(missing_snippets))

    print(f"root={ROOT}")
    print(f"torch={torch.__version__}")
    print(f"teacher_version={TEACHER_VERSION}")
    print(f"search_defaults=beam:{cfg.beam_width} node:{cfg.node_budget_per_root} depth:{cfg.max_depth}")
    print(f"lethal_budget={cfg.lethal_check_node_budget}")
    print(f"lethal_block_suppression_factor={cfg.lethal_block_suppression_factor}")
    print(f"power.Strength={cfg.player_power_weights['Strength']}")
    print(f"ironclad_hp={profile.current_hp}/{profile.max_hp}")
    print("bundle_check=ok")


if __name__ == "__main__":
    main()
'''


def install_deps() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_bundle.py
'''


def watch_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi
OUT_DIR="${OUT_DIR:-teacher_sweep_runs/v3_teacher_strength_fullsearch_v2}"
INTERVAL="${INTERVAL:-30}"
"$PYTHON" watch_teacher_sweep_progress.py --output-dir "$OUT_DIR" --interval "$INTERVAL" "$@"
'''


def status_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
echo "== process =="
if [ -f logs/v3_teacher_strength_fullsearch_v2.pid ]; then
  pid="$(cat logs/v3_teacher_strength_fullsearch_v2.pid)"
  ps -p "$pid" -o pid,ppid,pgid,stat,%cpu,%mem,etime,cmd || true
fi
pgrep -af 'run_v3_teacher_config_sweep_fast.py.*v3_teacher_strength_fullsearch_v2' || true
echo
echo "== latest log =="
tail -40 logs/v3_teacher_strength_fullsearch_v2.log 2>/dev/null || true
echo
echo "== progress summary =="
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi
"$PYTHON" watch_teacher_sweep_progress.py --output-dir teacher_sweep_runs/v3_teacher_strength_fullsearch_v2 --once --top 10 2>/dev/null || true
'''


def stop_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PID_PATH="${PID_PATH:-logs/v3_teacher_strength_fullsearch_v2.pid}"
if [ ! -f "$PID_PATH" ]; then
  echo "no pid file: $PID_PATH"
  exit 0
fi
pid="$(cat "$PID_PATH")"
pgid="$(ps -p "$pid" -o pgid= | tr -d ' ' || true)"
if [ -z "$pgid" ]; then
  echo "process not running: pid=$pid"
  exit 0
fi
kill -TERM "-$pgid"
echo "sent TERM to pgid=$pgid pid=$pid"
'''


def readme() -> str:
    return """# Aliyun Strength Full-Search Sweep v2

This bundle runs the Strength teacher sweep with no proxy stages:

- round1: all 19 Strength values, seed1-100
- round2: top 10, seed1-200
- round3: top 6, seed1-300
- round4: top 3, seed1-600
- every round uses the current default full teacher search, not beam=1/node=4/depth=3 proxy search
- `--resume` is enabled by default

## Install

```bash
cd aliyun_strength_fullsearch_v2
bash install_deps.sh
```

## Smoke Test

Run this before the long sweep:

```bash
SMOKE=1 bash run_strength_sweep.sh
```

A normal smoke can take around 1-3 minutes because it intentionally uses full teacher search.

## Run

```bash
WORKERS=14 bash run_strength_sweep.sh
```

The script defaults to about 80% of available CPU cores if `WORKERS` is not set.

## Monitor

```bash
bash status_strength_sweep.sh
tail -f logs/v3_teacher_strength_fullsearch_v2.log
bash watch_strength_sweep.sh --once
```

## Stop / Resume

```bash
bash stop_strength_sweep.sh
WORKERS=14 bash run_strength_sweep.sh
```

Resume uses the existing `teacher_sweep_runs/v3_teacher_strength_fullsearch_v2` directory.

## Outputs To Download

At minimum, download:

- `teacher_sweep_runs/v3_teacher_strength_fullsearch_v2/final_leaderboard.json`
- `teacher_sweep_runs/v3_teacher_strength_fullsearch_v2/leaderboard_round*_seed*.json`
- `teacher_sweep_runs/v3_teacher_strength_fullsearch_v2/sweep_config.json`
- `logs/v3_teacher_strength_fullsearch_v2.log`

If you want exact per-seed analysis, download the whole `teacher_sweep_runs/v3_teacher_strength_fullsearch_v2` folder.
"""


def build(out_dir: Path) -> Path:
    out_dir = out_dir.expanduser().resolve()
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
    write_text(out_dir / "install_deps.sh", install_deps(), executable=True)
    write_text(out_dir / "run_strength_sweep.sh", run_script(), executable=True)
    write_text(out_dir / "watch_strength_sweep.sh", watch_script(), executable=True)
    write_text(out_dir / "status_strength_sweep.sh", status_script(), executable=True)
    write_text(out_dir / "stop_strength_sweep.sh", stop_script(), executable=True)

    tar_path = out_dir.with_suffix(".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return tar_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Aliyun bundle for Strength full-search v2 sweep.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    tar_path = build(args.out_dir)
    print(f"bundle_dir={args.out_dir.expanduser().resolve()}")
    print(f"tar_path={tar_path}")


if __name__ == "__main__":
    main()
