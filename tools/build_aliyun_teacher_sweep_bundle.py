#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("/home/yydd/下载/aliyun_teacher_sweeps_v1")

SWEEP_DIRS = (
    "v3_teacher_power_A_sweep_v1",
    "v3_teacher_power_B_sweep_v1",
    "v3_teacher_power_C_sweep_v1",
    "v3_teacher_lethal_sweep_v1",
)

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

TOP_LEVEL_PY = (
    "scripts/v3_combat/evaluate_v3_rollout_batch.py",
    "scripts/v3_combat/run_v3_teacher_config_sweep.py",
    "scripts/v3_combat/run_v3_teacher_config_sweep_fast.py",
    "scripts/v3_combat/watch_teacher_sweep_progress.py",
    "setup.py",
)

POWER_A_RANGES = (
    '{"player_power_weights.Strength":[1.0,9.0],'
    '"player_power_weights.Flex":[-8.0,2.0],'
    '"player_power_weights.Demon Form":[2.0,36.0],'
    '"player_power_weights.Combust":[0.0,18.0],'
    '"player_power_weights.Rupture":[0.0,20.0],'
    '"player_power_weights.Fire Breathing":[0.0,18.0],'
    '"player_power_weights.Juggernaut":[0.0,30.0],'
    '"player_power_weights.Double Tap":[0.0,20.0]}'
)
POWER_B_RANGES = (
    '{"player_power_weights.Rage":[0.0,10.0],'
    '"player_power_weights.Metallicize":[0.0,22.0],'
    '"player_power_weights.Flame Barrier":[0.0,10.0],'
    '"player_power_weights.Barricade":[0.0,36.0],'
    '"player_power_weights.Feel No Pain":[0.0,20.0],'
    '"player_power_weights.IntangiblePlayer":[0.0,26.0],'
    '"player_power_weights.Artifact":[0.0,14.0]}'
)
POWER_C_RANGES = (
    '{"player_power_weights.Dark Embrace":[0.0,30.0],'
    '"player_power_weights.Evolve":[0.0,20.0],'
    '"player_power_weights.Brutality":[-4.0,14.0],'
    '"player_power_weights.Corruption":[0.0,40.0],'
    '"player_power_weights.Berserk":[-4.0,24.0],'
    '"player_power_weights.No Draw":[-24.0,0.0],'
    '"player_power_weights.Vulnerable":[-10.0,4.0]}'
)
LETHAL_RANGES = (
    '{"lethal_check_node_budget":[32,256],'
    '"lethal_block_suppression_factor":[-2,1]}'
)
LETHAL_GRID = (
    '{"lethal_check_node_budget":[32,64,96,160,256],'
    '"lethal_block_suppression_factor":[-2,-1.5,-1,-0.5,0,0.25,0.5,0.75,1]}'
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
"""


def power_script(name: str, ranges: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/_common.sh"

WORKERS="${{WORKERS:-$(default_workers)}}"
START_AT="${{START_AT:-round2}}"
STOP_AFTER="${{STOP_AFTER:-round4}}"
OUT_DIR="$ROOT/teacher_sweep_runs/v3_teacher_power_{name}_sweep_v1"
mkdir -p "$ROOT/logs" "$ROOT/teacher_sweep_runs"

export SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON='{ranges}'

exec "$PY" -u "$ROOT/scripts/v3_combat/run_v3_teacher_config_sweep_fast.py" \\
  --output-dir "$OUT_DIR" \\
  --seed-start 1 --workers "$WORKERS" --torch-threads 1 --blas-threads 1 \\
  --metrics-mode floor --summary-interval 0 --progress-interval-tasks 25 \\
  --round0-count 0 \\
  --round1-size 256 --round1-count 60 --round1-stage-counts 10,30,60 --round1-stage-keeps 128,64,32 \\
  --round2-top 32 --round2-count 100 \\
  --round3-top 16 --round3-count 300 \\
  --round4-top 6 --round4-count 600 \\
  --round1-proxy-beam-width 1 --round1-proxy-node-budget 4 --round1-proxy-max-depth 3 \\
  --round2-proxy-beam-width 2 --round2-proxy-node-budget 16 --round2-proxy-max-depth 6 \\
  --round3-proxy-beam-width 4 --round3-proxy-node-budget 48 --round3-proxy-max-depth 8 \\
  --round4-proxy-beam-width 0 --round4-proxy-node-budget 0 --round4-proxy-max-depth 0 \\
  --include-default-in-round1 --start-at "$START_AT" --stop-after "$STOP_AFTER" --resume
"""


def lethal_script() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/_common.sh"

WORKERS="${{WORKERS:-$(default_workers)}}"
START_AT="${{START_AT:-round3}}"
STOP_AFTER="${{STOP_AFTER:-round4}}"
OUT_DIR="$ROOT/teacher_sweep_runs/v3_teacher_lethal_sweep_v1"
mkdir -p "$ROOT/logs" "$ROOT/teacher_sweep_runs"

export SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON='{LETHAL_RANGES}'
export SPIRECOMM_TEACHER_SWEEP_ROUND1_GRID_JSON='{LETHAL_GRID}'

exec "$PY" -u "$ROOT/scripts/v3_combat/run_v3_teacher_config_sweep_fast.py" \\
  --output-dir "$OUT_DIR" \\
  --seed-start 1 --workers "$WORKERS" --torch-threads 1 --blas-threads 1 \\
  --metrics-mode floor --summary-interval 0 --progress-interval-tasks 25 \\
  --round0-count 0 \\
  --round1-count 60 --round1-stage-counts 10,30,60 --round1-stage-keeps 32,20,16 \\
  --round2-top 16 --round2-count 100 \\
  --round3-top 8 --round3-count 300 \\
  --round4-top 4 --round4-count 600 \\
  --round1-proxy-beam-width 1 --round1-proxy-node-budget 4 --round1-proxy-max-depth 3 \\
  --round2-proxy-beam-width 2 --round2-proxy-node-budget 16 --round2-proxy-max-depth 6 \\
  --round3-proxy-beam-width 4 --round3-proxy-node-budget 48 --round3-proxy-max-depth 8 \\
  --round4-proxy-beam-width 0 --round4-proxy-node-budget 0 --round4-proxy-max-depth 0 \\
  --start-at "$START_AT" --stop-after "$STOP_AFTER" --resume
"""


def run_all_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/logs"
cores="$(nproc 2>/dev/null || echo 1)"
percent="${SWEEP_WORKER_PERCENT:-150}"
if [[ ! "$percent" =~ ^[0-9]+$ ]] || [[ "$percent" -lt 1 ]]; then percent=150; fi
default_each=$(( (cores * percent + 99) / 100 / 4 ))
if [[ "$default_each" -lt 1 ]]; then default_each=1; fi
WORKERS_PER_SWEEP="${WORKERS_PER_SWEEP:-$default_each}"

start_one() {
  local name="$1"
  local script="$2"
  local log="$ROOT/logs/server_${name}.log"
  echo "start $name workers=$WORKERS_PER_SWEEP log=$log"
  (cd "$ROOT" && WORKERS="$WORKERS_PER_SWEEP" nohup bash "$script" > "$log" 2>&1 & echo $! > "logs/server_${name}.pid")
}

start_one power_A scripts/run_power_A.sh
start_one power_B scripts/run_power_B.sh
start_one power_C scripts/run_power_C.sh
start_one lethal scripts/run_lethal.sh
echo "started. use: bash scripts/status.sh"
"""


def status_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "== processes =="
pgrep -af 'scripts/v3_combat/run_v3_teacher_config_sweep_fast.py.*v3_teacher_(power_[ABC]|lethal)_sweep_v1' || true
echo
echo "== logs =="
for f in "$ROOT"/logs/server_*.log "$ROOT"/logs/v3_teacher_*_sweep_v1.log; do
  [[ -f "$f" ]] || continue
  echo "--- $f"
  tail -8 "$f"
done
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
python3 check_bundle.py
"""


def smoke_lethal_script() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/_common.sh"

TMP_ROOT="${{TMPDIR:-/tmp}}/aliyun_teacher_lethal_smoke_$$"
rm -rf "$TMP_ROOT"
mkdir -p "$(dirname "$TMP_ROOT")"
cp -a "$ROOT/teacher_sweep_runs/v3_teacher_lethal_sweep_v1" "$TMP_ROOT"

export SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON='{LETHAL_RANGES}'
export SPIRECOMM_TEACHER_SWEEP_ROUND1_GRID_JSON='{LETHAL_GRID}'

"$PY" -u "$ROOT/scripts/v3_combat/run_v3_teacher_config_sweep_fast.py" \\
  --output-dir "$TMP_ROOT" \\
  --seed-start 1 --workers 1 --torch-threads 1 --blas-threads 1 \\
  --metrics-mode floor --summary-interval 1 --progress-interval-tasks 1 \\
  --round0-count 0 \\
  --round1-count 60 --round1-stage-counts 10,30,60 --round1-stage-keeps 32,20,16 \\
  --round2-top 16 --round2-count 100 \\
  --round3-top 2 --round3-count 1 \\
  --round4-top 1 --round4-count 1 \\
  --round1-proxy-beam-width 1 --round1-proxy-node-budget 4 --round1-proxy-max-depth 3 \\
  --round2-proxy-beam-width 2 --round2-proxy-node-budget 16 --round2-proxy-max-depth 6 \\
  --round3-proxy-beam-width 4 --round3-proxy-node-budget 48 --round3-proxy-max-depth 8 \\
  --round4-proxy-beam-width 0 --round4-proxy-node-budget 0 --round4-proxy-max-depth 0 \\
  --start-at round3 --stop-after round3 --resume

"$PY" - "$TMP_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_paths = sorted((root / "evals" / "round3_seed300").glob("*/summary.json"))
if not summary_paths:
    raise SystemExit(f"smoke failed: no summary files under {{root}}")

errors = []
floors = []
for path in summary_paths:
    summary = json.loads(path.read_text(encoding="utf-8"))
    error_count = int(summary.get("error_count") or 0)
    floors.append(float(summary.get("mean_floor") or 0.0))
    if error_count:
        errors.append((path.parent.name, error_count, summary.get("errors") or []))

if errors:
    raise SystemExit("smoke failed: rollout errors found: " + repr(errors[:3]))
if not floors or max(floors) <= 1.0:
    raise SystemExit(f"smoke failed: suspicious floors={{floors}}")

print("smoke_ok", "candidates", len(summary_paths), "mean_floors", ",".join(f"{{value:.2f}}" for value in floors))
PY

rm -rf "$TMP_ROOT"
"""


def check_bundle() -> str:
    return """from __future__ import annotations

from pathlib import Path

root = Path(__file__).resolve().parent
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
    "teacher_sweep_runs/v3_teacher_lethal_sweep_v1/leaderboard_round2_seed100.json",
]
missing = [path for path in required if not (root / path).exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))

from spirecomm.native_sim_v3.content.characters import starting_profile
from spirecomm.ai.v3_combat_teacher import teacher_config_from_mapping

profile = starting_profile("IRONCLAD")
cfg = teacher_config_from_mapping({"teacher_config": {"player_power_weights.Strength": 7.0}})
print("bundle_ok", "starter_cards", len(profile.starter_deck_ids), "strength_weight", cfg.player_power_weights["Strength"])
"""


def readme() -> str:
    return """# Aliyun Teacher Sweep Bundle

This bundle is self-contained for CPU teacher sweeps. It includes the Python code, bundled STS decompiled/reference data, small model files, and sweep state directories.

## Install

```bash
cd aliyun_teacher_sweeps_v1
bash scripts/install_deps.sh
```

Optional full runtime smoke test before starting a long sweep:

```bash
bash scripts/smoke_lethal_seed1.sh
```

## Run One Sweep

Power A/B/C scripts default to `START_AT=round2`, intended for use after local `round1_seed60` has completed and the refreshed `teacher_sweep_runs/v3_teacher_power_*_sweep_v1` directories have been uploaded.

```bash
WORKERS=14 bash scripts/run_power_A.sh
WORKERS=14 bash scripts/run_power_B.sh
WORKERS=14 bash scripts/run_power_C.sh
```

Lethal script defaults to `START_AT=round3`. The local partial `round3_seed300` was intentionally discarded before packaging, so it resumes from the completed `round2_seed100` leaderboard.

```bash
WORKERS=14 bash scripts/run_lethal.sh
```

To run all four on one server:

```bash
WORKERS_PER_SWEEP=4 bash run_all_parallel.sh
bash scripts/status.sh
```

## Important

- If a Power A/B/C state directory does not yet contain `leaderboard_round1_seed60.json`, do not run its default script. Either upload a refreshed state after local round1 completes, or run with `START_AT=round1`.
- Results are written under `teacher_sweep_runs/`.
- Logs from `run_all_parallel.sh` are written under `logs/server_*.log`.
"""


def build(out_dir: Path, *, include_states: bool) -> Path:
    out_dir = out_dir.expanduser().resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    copy_path(ROOT / "spirecomm", out_dir / "spirecomm")
    for rel in TOP_LEVEL_PY:
        copy_path(ROOT / rel, out_dir / rel)
    for rel in MODEL_FILES:
        copy_path(ROOT / rel, out_dir / rel)
    if include_states:
        for name in SWEEP_DIRS:
            src = ROOT / "teacher_sweep_runs" / name
            if src.exists():
                copy_path(src, out_dir / "teacher_sweep_runs" / name)

    write_text(out_dir / "requirements.txt", "numpy\ntorch\n", executable=False)
    write_text(out_dir / "README.md", readme())
    write_text(out_dir / "check_bundle.py", check_bundle())
    write_text(out_dir / "scripts" / "_common.sh", script_common(), executable=True)
    write_text(out_dir / "scripts" / "install_deps.sh", install_script(), executable=True)
    write_text(out_dir / "scripts" / "run_power_A.sh", power_script("A", POWER_A_RANGES), executable=True)
    write_text(out_dir / "scripts" / "run_power_B.sh", power_script("B", POWER_B_RANGES), executable=True)
    write_text(out_dir / "scripts" / "run_power_C.sh", power_script("C", POWER_C_RANGES), executable=True)
    write_text(out_dir / "scripts" / "run_lethal.sh", lethal_script(), executable=True)
    write_text(out_dir / "scripts" / "smoke_lethal_seed1.sh", smoke_lethal_script(), executable=True)
    write_text(out_dir / "scripts" / "status.sh", status_script(), executable=True)
    write_text(out_dir / "run_all_parallel.sh", run_all_script(), executable=True)

    tar_path = out_dir.with_suffix(".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return tar_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-states", action="store_true")
    args = parser.parse_args()
    tar_path = build(args.out_dir, include_states=not args.no_states)
    print(f"bundle_dir={args.out_dir.expanduser().resolve()}")
    print(f"tar_path={tar_path}")


if __name__ == "__main__":
    main()
