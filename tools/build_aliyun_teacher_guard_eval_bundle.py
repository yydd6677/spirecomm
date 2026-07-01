#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("/home/yydd/下载/aliyun_teacher_guard_eval_v12")

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
    "run_native_run.py",
    "setup.py",
    "README.md",
)

BASELINE_CONFIG = {
    "teacher_config": {
        "potion_elite_room_reward_factor": 1.3,
        "potion_cost_scale": 1.1,
    }
}

GUARD_CONFIG = {
    "teacher_config": {
        "potion_elite_room_reward_factor": 1.3,
        "potion_cost_scale": 1.1,
        "teacher_survival_guard_enabled": True,
        "teacher_survival_guard_restrict_safe": True,
        "teacher_survival_guard_score_margin": 10000.0,
        "lethal_block_low_hp_protection": True,
        "lethal_block_low_hp_max": 8,
        "lethal_block_low_hp_suppression_factor": 1.0,
        "lethal_block_low_hp_requires_facing_lethal": True,
    }
}


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

default_total_workers() {
  local cores
  cores="$(nproc 2>/dev/null || echo 1)"
  local percent="${EVAL_WORKER_PERCENT:-100}"
  if [[ ! "$percent" =~ ^[0-9]+$ ]] || [[ "$percent" -lt 1 ]]; then percent=100; fi
  local workers=$(( (cores * percent + 99) / 100 ))
  if [[ "$workers" -lt 1 ]]; then workers=1; fi
  echo "$workers"
}
'''


def run_one_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="${1:-}"
CONFIG_PATH="${2:-}"
OUT_DIR="${3:-}"
if [[ -z "$NAME" || -z "$CONFIG_PATH" || -z "$OUT_DIR" ]]; then
  echo "usage: bash scripts/run_one_eval.sh <name> <config_path> <output_dir>" >&2
  exit 2
fi

WORKERS="${WORKERS:-$(default_total_workers)}"
SEED_START="${SEED_START:-1}"
COUNT="${COUNT:-600}"
MAX_STEPS="${MAX_STEPS:-1500}"
SUMMARY_INTERVAL="${SUMMARY_INTERVAL:-10}"
RESULT_FLUSH_INTERVAL="${RESULT_FLUSH_INTERVAL:-16}"
TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-1}"
mkdir -p "$ROOT/logs" "$ROOT/eval_runs"

exec "$PY" -u "$ROOT/evaluate_v3_rollout_batch.py" \
  --output-dir "$OUT_DIR" \
  --seed-start "$SEED_START" \
  --count "$COUNT" \
  --workers "$WORKERS" \
  --combat-selector v3-teacher \
  --teacher-config-path "$CONFIG_PATH" \
  --mean-floor-only \
  --max-steps "$MAX_STEPS" \
  --summary-interval "$SUMMARY_INTERVAL" \
  --result-flush-interval "$RESULT_FLUSH_INTERVAL" \
  --task-batch-size "$TASK_BATCH_SIZE" \
  --torch-threads 1 \
  --resume \
  --rerun-timeouts
'''


def run_pair_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_common.sh"

TOTAL_WORKERS="${WORKERS:-$(default_total_workers)}"
RUN_MODE="${RUN_MODE:-parallel}"
SEED_START="${SEED_START:-1}"
COUNT="${COUNT:-600}"
mkdir -p "$ROOT/logs" "$ROOT/eval_runs"

if [[ "$RUN_MODE" = "parallel" ]]; then
  WORKERS_PER_EVAL="${WORKERS_PER_EVAL:-$(( TOTAL_WORKERS / 2 ))}"
  if [[ "$WORKERS_PER_EVAL" -lt 1 ]]; then WORKERS_PER_EVAL=1; fi
  echo "start baseline+guard parallel total_workers=$TOTAL_WORKERS workers_per_eval=$WORKERS_PER_EVAL seeds=$SEED_START..$((SEED_START + COUNT - 1))"
  (
    cd "$ROOT"
    WORKERS="$WORKERS_PER_EVAL" SEED_START="$SEED_START" COUNT="$COUNT" \
      bash scripts/run_one_eval.sh baseline configs/teacher_baseline.json eval_runs/teacher_v12_baseline_seed1_600 \
      > logs/teacher_v12_baseline.log 2>&1 &
    echo $! > logs/teacher_v12_baseline.pid
    WORKERS="$WORKERS_PER_EVAL" SEED_START="$SEED_START" COUNT="$COUNT" \
      bash scripts/run_one_eval.sh guard configs/teacher_guard.json eval_runs/teacher_v12_guard_seed1_600 \
      > logs/teacher_v12_guard.log 2>&1 &
    echo $! > logs/teacher_v12_guard.pid
  )
else
  echo "start baseline then guard sequential workers=$TOTAL_WORKERS seeds=$SEED_START..$((SEED_START + COUNT - 1))"
  WORKERS="$TOTAL_WORKERS" SEED_START="$SEED_START" COUNT="$COUNT" \
    bash "$ROOT/scripts/run_one_eval.sh" baseline "$ROOT/configs/teacher_baseline.json" "$ROOT/eval_runs/teacher_v12_baseline_seed1_600" \
    2>&1 | tee "$ROOT/logs/teacher_v12_baseline.log"
  WORKERS="$TOTAL_WORKERS" SEED_START="$SEED_START" COUNT="$COUNT" \
    bash "$ROOT/scripts/run_one_eval.sh" guard "$ROOT/configs/teacher_guard.json" "$ROOT/eval_runs/teacher_v12_guard_seed1_600" \
    2>&1 | tee "$ROOT/logs/teacher_v12_guard.log"
fi

echo "started. monitor: bash scripts/status.sh"
'''


def smoke_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

rm -rf "$ROOT/_smoke_teacher_baseline" "$ROOT/_smoke_teacher_guard"
WORKERS=1 SEED_START=1 COUNT=1 MAX_STEPS=300 \
  "$PY" -u "$ROOT/evaluate_v3_rollout_batch.py" \
  --output-dir "$ROOT/_smoke_teacher_baseline" \
  --seeds 1 --workers 1 --combat-selector v3-teacher \
  --teacher-config-path "$ROOT/configs/teacher_baseline.json" \
  --mean-floor-only --max-floor 1 --summary-interval 1 --result-flush-interval 1 --task-batch-size 1 --torch-threads 1
WORKERS=1 SEED_START=1 COUNT=1 MAX_STEPS=300 \
  "$PY" -u "$ROOT/evaluate_v3_rollout_batch.py" \
  --output-dir "$ROOT/_smoke_teacher_guard" \
  --seeds 1 --workers 1 --combat-selector v3-teacher \
  --teacher-config-path "$ROOT/configs/teacher_guard.json" \
  --mean-floor-only --max-floor 1 --summary-interval 1 --result-flush-interval 1 --task-batch-size 1 --torch-threads 1
rm -rf "$ROOT/_smoke_teacher_baseline" "$ROOT/_smoke_teacher_guard"
echo "smoke=ok"
'''


def status_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "== processes =="
pgrep -af 'evaluate_v3_rollout_batch.py.*teacher_v12_(baseline|guard)|run_one_eval.sh (baseline|guard)' || true
echo
for name in baseline guard; do
  echo "== $name log =="
  tail -20 "$ROOT/logs/teacher_v12_${name}.log" 2>/dev/null || true
  echo
  echo "== $name partial =="
  sed -n '1,120p' "$ROOT/eval_runs/teacher_v12_${name}_seed1_600/summary_partial.json" 2>/dev/null || true
  echo
done
echo "== paired summary =="
PY="${PY:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY="${PYTHON:-python3}"; fi
"$PY" "$ROOT/scripts/summarize_pair.py" 2>/dev/null || true
'''


def stop_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for pidfile in "$ROOT"/logs/teacher_v12_*.pid; do
  [[ -f "$pidfile" ]] || continue
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "TERM pid=$pid from $pidfile"
    kill "$pid" 2>/dev/null || true
  fi
done
sleep 2
pgrep -f 'evaluate_v3_rollout_batch.py.*teacher_v12_(baseline|guard)|run_one_eval.sh (baseline|guard)' | xargs -r kill 2>/dev/null || true
'''


def summarize_pair_script() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["seed"])] = row
    return rows


def summarize(rows: dict[int, dict], limit: int) -> dict[str, float | int]:
    selected = [rows[seed] for seed in range(1, limit + 1) if seed in rows]
    if not selected:
        return {"count": 0}
    return {
        "count": len(selected),
        "mean_floor": sum(int(row.get("floor") or 0) for row in selected) / len(selected),
        "win_count": sum(1 for row in selected if bool(row.get("won"))),
        "timeout_count": sum(1 for row in selected if bool(row.get("timed_out"))),
        "error_count": sum(1 for row in selected if row.get("error")),
    }


def paired(base: dict[int, dict], guard: dict[int, dict], limit: int) -> dict[str, float | int]:
    seeds = [seed for seed in range(1, limit + 1) if seed in base and seed in guard]
    if not seeds:
        return {"count": 0}
    deltas = [int(guard[seed].get("floor") or 0) - int(base[seed].get("floor") or 0) for seed in seeds]
    return {
        "count": len(seeds),
        "mean_delta": sum(deltas) / len(deltas),
        "sum_delta": sum(deltas),
        "up": sum(1 for delta in deltas if delta > 0),
        "down": sum(1 for delta in deltas if delta < 0),
        "tie": sum(1 for delta in deltas if delta == 0),
        "win_gain": sum(1 for seed in seeds if bool(guard[seed].get("won")) and not bool(base[seed].get("won"))),
        "win_loss": sum(1 for seed in seeds if bool(base[seed].get("won")) and not bool(guard[seed].get("won"))),
    }


def main() -> None:
    base = load_rows(ROOT / "eval_runs" / "teacher_v12_baseline_seed1_600" / "results.jsonl")
    guard = load_rows(ROOT / "eval_runs" / "teacher_v12_guard_seed1_600" / "results.jsonl")
    payload = {
        "baseline_1_300": summarize(base, 300),
        "baseline_1_600": summarize(base, 600),
        "guard_1_300": summarize(guard, 300),
        "guard_1_600": summarize(guard, 600),
        "paired_guard_minus_baseline_1_300": paired(base, guard, 300),
        "paired_guard_minus_baseline_1_600": paired(base, guard, 600),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
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
python check_bundle.py
'''


def check_bundle() -> str:
    return r'''from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    import torch

    from spirecomm.ai.v3_combat_teacher import TEACHER_VERSION, default_teacher_config, teacher_config_from_mapping
    from spirecomm.native_sim_v3.content.characters import starting_profile

    required = [
        "models/card_reward.pt",
        "models/event_choice.pt",
        "models/potion_use.pt",
        "models/shop_choice_prior_delta.pt",
        "models/upgrade_target.pt",
        "spirecomm/native_sim_v3/reference/decompiled_sts/com/megacrit/cardcrawl/characters/Ironclad.java",
        "configs/teacher_baseline.json",
        "configs/teacher_guard.json",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("missing required files: " + ", ".join(missing))

    cfg = default_teacher_config()
    if (cfg.beam_width, cfg.node_budget_per_root, cfg.max_depth) != (24, 768, 20):
        raise SystemExit(f"unexpected search defaults: {cfg.beam_width}/{cfg.node_budget_per_root}/{cfg.max_depth}")
    if abs(float(cfg.potion_elite_room_reward_factor) - 1.3) > 1e-9:
        raise SystemExit(f"unexpected potion elite default: {cfg.potion_elite_room_reward_factor}")
    if abs(float(cfg.potion_cost_scale) - 1.1) > 1e-9:
        raise SystemExit(f"unexpected potion cost default: {cfg.potion_cost_scale}")
    guard_cfg = teacher_config_from_mapping({"teacher_config": {
        "teacher_survival_guard_enabled": True,
        "lethal_block_low_hp_protection": True,
    }})
    if not guard_cfg.teacher_survival_guard_enabled or not guard_cfg.lethal_block_low_hp_protection:
        raise SystemExit("guard config parsing failed")
    profile = starting_profile("IRONCLAD")
    print(f"root={ROOT}")
    print(f"torch={torch.__version__}")
    print(f"teacher_version={TEACHER_VERSION}")
    print(f"search_defaults=beam:{cfg.beam_width} node:{cfg.node_budget_per_root} depth:{cfg.max_depth}")
    print(f"potion_defaults=elite:{cfg.potion_elite_room_reward_factor} cost:{cfg.potion_cost_scale}")
    print(f"ironclad_hp={profile.current_hp}/{profile.max_hp}")
    print("bundle_check=ok")


if __name__ == "__main__":
    main()
'''


def readme() -> str:
    return """# Aliyun Teacher v12 Baseline vs Guard Eval

This bundle evaluates pure `v3-teacher` decisions on seed1-600.

Current baked defaults:

- `potion_elite_room_reward_factor = 1.3`
- `potion_cost_scale = 1.1`
- search budget `beam_width=24`, `node_budget_per_root=768`, `max_depth=20`

Compared configs:

- `configs/teacher_baseline.json`: current v12 teacher defaults.
- `configs/teacher_guard.json`: baseline plus survival/suicidal teacher guard and low-HP block protection.

Run:

```bash
cd aliyun_teacher_guard_eval_v12
bash scripts/install_deps.sh
bash scripts/smoke_seed1.sh
WORKERS=$(nproc) RUN_MODE=parallel bash run_teacher_eval_pair.sh
```

For lower memory pressure:

```bash
WORKERS=80 RUN_MODE=parallel bash run_teacher_eval_pair.sh
```

For maximum per-eval speed but sequential execution:

```bash
WORKERS=$(nproc) RUN_MODE=sequential bash run_teacher_eval_pair.sh
```

Monitor:

```bash
bash scripts/status.sh
tail -f logs/teacher_v12_baseline.log
tail -f logs/teacher_v12_guard.log
```

Stop:

```bash
bash scripts/stop.sh
```

Summarize:

```bash
python scripts/summarize_pair.py
```

Download after completion:

- `eval_runs/teacher_v12_baseline_seed1_600/results.jsonl`
- `eval_runs/teacher_v12_baseline_seed1_600/summary.json`
- `eval_runs/teacher_v12_guard_seed1_600/results.jsonl`
- `eval_runs/teacher_v12_guard_seed1_600/summary.json`
- `logs/teacher_v12_baseline.log`
- `logs/teacher_v12_guard.log`
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
    write_text(out_dir / "configs" / "teacher_baseline.json", json.dumps(BASELINE_CONFIG, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "configs" / "teacher_guard.json", json.dumps(GUARD_CONFIG, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "scripts" / "_common.sh", script_common(), executable=True)
    write_text(out_dir / "scripts" / "install_deps.sh", install_script(), executable=True)
    write_text(out_dir / "scripts" / "run_one_eval.sh", run_one_script(), executable=True)
    write_text(out_dir / "scripts" / "smoke_seed1.sh", smoke_script(), executable=True)
    write_text(out_dir / "scripts" / "status.sh", status_script(), executable=True)
    write_text(out_dir / "scripts" / "stop.sh", stop_script(), executable=True)
    write_text(out_dir / "scripts" / "summarize_pair.py", summarize_pair_script(), executable=True)
    write_text(out_dir / "run_teacher_eval_pair.sh", run_pair_script(), executable=True)

    tar_path = out_dir.with_suffix(".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return tar_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    tar_path = build(args.output_dir)
    print(f"bundle_dir={args.output_dir.expanduser().resolve()}")
    print(f"tar_path={tar_path}")


if __name__ == "__main__":
    main()
