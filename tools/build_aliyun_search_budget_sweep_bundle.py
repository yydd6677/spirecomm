#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("/home/yydd/下载/aliyun_search_budget_beam_sweep_v1")

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

SEARCH_BUDGET_GRID = '{"beam_width":[8,10,12,14,16,20]}'

SEARCH_BUDGET_RANGES = '{"beam_width":[8,20]}'


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


def run_search_budget_script() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/_common.sh"

WORKERS="${{WORKERS:-$(default_workers)}}"
START_AT="${{START_AT:-round0}}"
STOP_AFTER="${{STOP_AFTER:-round4}}"
OUT_DIR="$ROOT/teacher_sweep_runs/v3_teacher_beam_width_sweep_v1"
mkdir -p "$ROOT/logs" "$ROOT/teacher_sweep_runs"

export SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON='{SEARCH_BUDGET_RANGES}'
export SPIRECOMM_TEACHER_SWEEP_ROUND1_GRID_JSON='{SEARCH_BUDGET_GRID}'

exec "$PY" -u "$ROOT/scripts/v3_combat/run_v3_teacher_config_sweep_fast.py" \\
  --output-dir "$OUT_DIR" \\
  --seed-start 1 --workers "$WORKERS" --torch-threads 1 --blas-threads 1 \\
  --metrics-mode floor --summary-interval 0 --progress-interval-tasks 25 \\
  --round0-count 300 \\
  --round1-count 100 --round1-stage-counts 100 --round1-stage-keeps 6 \\
  --round2-top 6 --round2-count 200 \\
  --round3-top 4 --round3-count 300 \\
  --round4-top 3 --round4-count 600 \\
  --round1-proxy-beam-width 0 --round1-proxy-node-budget 0 --round1-proxy-max-depth 0 \\
  --round2-proxy-beam-width 0 --round2-proxy-node-budget 0 --round2-proxy-max-depth 0 \\
  --round3-proxy-beam-width 0 --round3-proxy-node-budget 0 --round3-proxy-max-depth 0 \\
  --round4-proxy-beam-width 0 --round4-proxy-node-budget 0 --round4-proxy-max-depth 0 \\
  --start-at "$START_AT" --stop-after "$STOP_AFTER" --resume
"""


def run_background_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/logs"
WORKERS="${WORKERS:-}"
if [[ -z "$WORKERS" ]]; then
  WORKERS="$(default_workers)"
fi
log="$ROOT/logs/search_budget.log"
echo "start search_budget workers=$WORKERS log=$log"
(cd "$ROOT" && WORKERS="$WORKERS" nohup bash scripts/run_search_budget.sh > "$log" 2>&1 & echo $! > logs/search_budget.pid)
echo "started pid=$(cat "$ROOT/logs/search_budget.pid")"
echo "status: bash scripts/status.sh"
"""


def status_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "== process =="
pgrep -af 'scripts/v3_combat/run_v3_teacher_config_sweep_fast.py.*v3_teacher_beam_width_sweep_v1' || true
echo
echo "== latest log =="
if [[ -f "$ROOT/logs/search_budget.log" ]]; then
  tail -40 "$ROOT/logs/search_budget.log"
else
  echo "missing $ROOT/logs/search_budget.log"
fi
echo
echo "== leaderboards =="
find "$ROOT/teacher_sweep_runs/v3_teacher_beam_width_sweep_v1" -maxdepth 1 -name 'leaderboard_*.json' -printf '%TY-%Tm-%Td %TH:%TM %p\\n' 2>/dev/null | sort || true
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


def smoke_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

TMP_ROOT="${TMPDIR:-/tmp}/aliyun_search_budget_smoke_$$"
rm -rf "$TMP_ROOT"
mkdir -p "$(dirname "$TMP_ROOT")"

export SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON='{"beam_width":[8,12]}'
export SPIRECOMM_TEACHER_SWEEP_ROUND1_GRID_JSON='{"beam_width":[8,12]}'

"$PY" -u "$ROOT/scripts/v3_combat/run_v3_teacher_config_sweep_fast.py" \\
  --output-dir "$TMP_ROOT" \\
  --seed-start 1 --workers 1 --torch-threads 1 --blas-threads 1 \\
  --max-floor 3 --max-steps 300 \\
  --metrics-mode floor --summary-interval 1 --progress-interval-tasks 1 \\
  --round0-count 1 \\
  --round1-count 1 --round1-stage-counts 1 --round1-stage-keeps 2 \\
  --round2-top 2 --round2-count 1 \\
  --round3-top 1 --round3-count 1 \\
  --round4-top 1 --round4-count 1 \\
  --round1-proxy-beam-width 0 --round1-proxy-node-budget 0 --round1-proxy-max-depth 0 \\
  --round2-proxy-beam-width 0 --round2-proxy-node-budget 0 --round2-proxy-max-depth 0 \\
  --round3-proxy-beam-width 0 --round3-proxy-node-budget 0 --round3-proxy-max-depth 0 \\
  --round4-proxy-beam-width 0 --round4-proxy-node-budget 0 --round4-proxy-max-depth 0 \\
  --start-at round0 --stop-after round1 --resume

"$PY" - "$TMP_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "sweep_config.json").read_text(encoding="utf-8"))
if config["round1_proxy_search"] or config["round2_proxy_search"] or config["round3_proxy_search"] or config["round4_proxy_search"]:
    raise SystemExit("smoke failed: proxy search overrides are not disabled")
leaderboard = root / "leaderboard_round1_seed1.json"
if not leaderboard.exists():
    raise SystemExit(f"smoke failed: missing {leaderboard}")
rows = json.loads(leaderboard.read_text(encoding="utf-8"))
if not rows:
    raise SystemExit("smoke failed: empty round1 leaderboard")
errors = []
for path in sorted((root / "evals" / "round1_seed1").glob("*/summary.json")):
    summary = json.loads(path.read_text(encoding="utf-8"))
    if int(summary.get("error_count") or 0):
        errors.append((path.parent.name, summary.get("errors") or []))
if errors:
    raise SystemExit("smoke failed: rollout errors found: " + repr(errors[:3]))
print("smoke_ok", "round1_rows", len(rows), "default_first_done", (root / "leaderboard_round0_default.json").exists())
PY

rm -rf "$TMP_ROOT"
"""


def check_bundle() -> str:
    return """from __future__ import annotations

import json
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
    "spirecomm/native_sim_v3/reference/decompiled_sts/com/megacrit/cardcrawl/characters/Ironclad.java",
]
missing = [path for path in required if not (root / path).exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))

from spirecomm.native_sim_v3.content.characters import starting_profile
from spirecomm.ai.v3_combat_teacher import teacher_config_from_mapping

profile = starting_profile("IRONCLAD")
cfg = teacher_config_from_mapping({"teacher_config": {"beam_width": 8, "node_budget_per_root": 128, "max_depth": 12}})
assert cfg.beam_width == 8
assert cfg.node_budget_per_root == 128
assert cfg.max_depth == 12
grid = json.loads('""" + SEARCH_BUDGET_GRID + """')
candidate_count = 1
for values in grid.values():
    candidate_count *= len(values)
print("bundle_ok", "starter_cards", len(profile.starter_deck_ids), "beam_width_candidates", candidate_count)
"""


def readme() -> str:
    return """# Aliyun Beam Width Sweep Bundle

This bundle sweeps only the v3 teacher combat search `beam_width`.

The current default `(beam_width=12, node_budget_per_root=256, max_depth=20)` is evaluated first as `round0_default`.
Round1 then evaluates only:

```text
8, 10, 12, 14, 16, 20
```

All round proxy search overrides are disabled, so candidate values are not overwritten by the sweep scheduler.

## Install

```bash
cd aliyun_search_budget_beam_sweep_v1
bash scripts/install_deps.sh
```

Optional smoke test:

```bash
bash scripts/smoke_search_budget_seed1.sh
```

## Run

Foreground:

```bash
WORKERS=14 bash scripts/run_search_budget.sh
```

Background:

```bash
WORKERS=14 bash run_search_budget_background.sh
bash scripts/status.sh
```

## Search Plan

- `round0`: default, seed1-300.
- `round1`: 6 beam candidates, seed1-100, keep all 6.
- `round2`: 6 candidates, seed1-200, promote top4.
- `round3`: top4, seed1-300, promote top3.
- `round4`: top3, seed1-600.

The grid is:

```json
{"beam_width":[8,10,12,14,16,20]}
```

Results are written under `teacher_sweep_runs/v3_teacher_beam_width_sweep_v1/`.
"""


def build(out_dir: Path) -> Path:
    out_dir = out_dir.expanduser().resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    copy_path(ROOT / "spirecomm", out_dir / "spirecomm")
    for rel in TOP_LEVEL_PY:
        copy_path(ROOT / rel, out_dir / rel)
    for rel in MODEL_FILES:
        copy_path(ROOT / rel, out_dir / rel)

    write_text(out_dir / "requirements.txt", "numpy\ntorch\n", executable=False)
    write_text(out_dir / "README.md", readme())
    write_text(out_dir / "check_bundle.py", check_bundle())
    write_text(out_dir / "scripts" / "_common.sh", script_common(), executable=True)
    write_text(out_dir / "scripts" / "install_deps.sh", install_script(), executable=True)
    write_text(out_dir / "scripts" / "run_search_budget.sh", run_search_budget_script(), executable=True)
    write_text(out_dir / "scripts" / "smoke_search_budget_seed1.sh", smoke_script(), executable=True)
    write_text(out_dir / "scripts" / "status.sh", status_script(), executable=True)
    write_text(out_dir / "run_search_budget_background.sh", run_background_script(), executable=True)

    tar_path = out_dir.with_suffix(".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return tar_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    tar_path = build(args.out_dir)
    print(f"bundle_dir={args.out_dir.expanduser().resolve()}")
    print(f"tar_path={tar_path}")


if __name__ == "__main__":
    main()
