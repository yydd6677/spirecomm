#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("/home/yydd/下载/aliyun_map_dp_sweep_v4")

TOP_LEVEL_FILES = (
    "evaluate_v3_rollout_batch.py",
    "run_map_dp_sweep.py",
    "run_shared_prefix_sweep.py",
    "setup.py",
    "README.md",
    "LICENSE",
)

MODEL_FILES = (
    "models/combat.pt",
    "models/v3_combat_scorer.pt",
    "models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt",
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


def common_script() -> str:
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
  local percent="${SWEEP_WORKER_PERCENT:-100}"
  if [[ ! "$percent" =~ ^[0-9]+$ ]] || [[ "$percent" -lt 1 ]]; then percent=100; fi
  local workers=$(( (cores * percent + 99) / 100 ))
  if [[ "$workers" -lt 1 ]]; then workers=1; fi
  echo "$workers"
}
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


def run_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

WORKERS="${WORKERS:-$(default_workers)}"
OUT_DIR="${OUT_DIR:-map_sweep_runs/v5_104_map_dp_sweep_v4}"
LOG_PATH="${LOG_PATH:-logs/v5_104_map_dp_sweep_v4.log}"
START_AT="${START_AT:-round0_default}"
STOP_AFTER="${STOP_AFTER:-all}"
COMBAT_DEVICE="${COMBAT_DEVICE:-cpu}"
mkdir -p "$ROOT/logs" "$ROOT/map_sweep_runs" "$OUT_DIR"

CMD=(
  "$PY" -u "$ROOT/run_map_dp_sweep.py"
  --output-dir "$OUT_DIR"
  --seed-start 1
  --workers "$WORKERS"
  --torch-threads 1
  --v3-combat-model models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt
  --device cpu
  --combat-device "$COMBAT_DEVICE"
  --preload-selectors always
  --max-floor 60
  --max-steps 1500
  --combat-stall-limit "${COMBAT_STALL_LIMIT:-250}"
  --summary-interval 50
  --task-batch-size 1
  --result-flush-interval 16
  --round3-source current_best_local
  --round0-count "${ROUND0_COUNT:-600}"
  --round1-size "${ROUND1_SIZE:-64}"
  --round1-count "${ROUND1_COUNT:-100}"
  --round2-groups "${ROUND2_GROUPS:-32}"
  --round2-count "${ROUND2_COUNT:-200}"
  --round3-groups "${ROUND3_GROUPS:-32}"
  --round3-count "${ROUND3_COUNT:-300}"
  --round4-top "${ROUND4_TOP:-8}"
  --round4-count "${ROUND4_COUNT:-600}"
  --start-at "$START_AT"
  --stop-after "$STOP_AFTER"
  --resume
)

printf '%q ' "${CMD[@]}" > "${LOG_PATH%.log}.command"
printf '\n' >> "${LOG_PATH%.log}.command"

echo "python=$PY"
"$PY" check_bundle.py
echo "output_dir=$OUT_DIR"
echo "log_path=$LOG_PATH"
echo "workers=$WORKERS"
echo "combat_device=$COMBAT_DEVICE"
echo "baseline default center seed1-600"
echo "round3 current-best local 32 groups seed1-300"
echo "round4 8 groups seed1-600"
echo "resume enabled; set START_AT/STOP_AFTER to resume a specific round"

if [[ "${FOREGROUND:-0}" = "1" ]]; then
  "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
else
  setsid "${CMD[@]}" >> "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > logs/v5_104_map_dp_sweep_v4.pid
  echo "started pid=$(cat logs/v5_104_map_dp_sweep_v4.pid)"
  echo "tail -f $LOG_PATH"
fi
'''


def run_background_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/_common.sh"
WORKERS="${WORKERS:-$(default_workers)}"
echo "start map dp sweep v4 workers=$WORKERS"
(cd "$ROOT" && WORKERS="$WORKERS" bash scripts/run_map_dp_sweep.sh)
'''


def smoke_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT_DIR="${TMPDIR:-/tmp}/map_dp_sweep_smoke_$$"
rm -rf "$OUT_DIR"
"$PY" -u "$ROOT/run_map_dp_sweep.py" \
  --output-dir "$OUT_DIR" \
  --seed-start 1 \
  --workers "${WORKERS:-1}" \
  --torch-threads 1 \
  --v3-combat-model models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt \
  --device cpu \
  --combat-device cpu \
  --preload-selectors never \
  --round3-source current_best_local \
  --start-at round0_default \
  --round0-count 1 \
  --round3-groups 4 \
  --round3-count 1 \
  --round4-top 2 \
  --round4-count 1 \
  --combat-stall-limit "${COMBAT_STALL_LIMIT:-250}" \
  --summary-interval 1 \
  --stop-after all
echo "smoke output=$OUT_DIR"
cat "$OUT_DIR/final_leaderboard.json"
'''


def status_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/map_sweep_runs/v5_104_map_dp_sweep_v4}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/v5_104_map_dp_sweep_v4.log}"
echo "== process =="
pgrep -af 'run_map_dp_sweep.py.*v5_104_map_dp_sweep_v4' || true
echo
echo "== latest log =="
tail -100 "$LOG_PATH" 2>/dev/null || true
echo
echo "== leaderboards =="
for f in "$OUT_DIR"/round*_results.json "$OUT_DIR"/final_leaderboard.json; do
  [[ -f "$f" ]] || continue
  echo "-- $f"
  python3 - "$f" <<'PY'
import json, sys
rows=json.load(open(sys.argv[1]))
if isinstance(rows, list):
    for row in rows[:8]:
        print({k: row.get(k) for k in ("stage","rank","name","kind","mean_floor","win_count","timeout_count","monster_value","rest_value","elite_base_value","green_elite_penalty","shop_gold_unit_value","shop_purgeable_curse_bonus","shop_purgeable_curse_urgency_bonus","shop_purgeable_curse_gold_threshold","winged_offpath_penalty")})
PY
done
'''


def stop_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_PATH="$ROOT/logs/v5_104_map_dp_sweep_v4.pid"
if [[ ! -f "$PID_PATH" ]]; then
  echo "no pid file: $PID_PATH"
  pgrep -af 'run_map_dp_sweep.py.*v5_104_map_dp_sweep_v4' || true
  exit 0
fi
pid="$(cat "$PID_PATH")"
pgid="$(ps -p "$pid" -o pgid= | tr -d ' ' || true)"
if [[ -n "$pgid" ]]; then
  kill -TERM "-$pgid" 2>/dev/null || true
  sleep 3
  kill -KILL "-$pgid" 2>/dev/null || true
  echo "stopped process group $pgid"
else
  echo "pid not running: $pid"
fi
'''


def check_bundle() -> str:
    return r'''#!/usr/bin/env python3
from pathlib import Path
import torch

required = [
    "evaluate_v3_rollout_batch.py",
    "run_map_dp_sweep.py",
    "run_shared_prefix_sweep.py",
    "spirecomm/ai/runtime_decision.py",
    "spirecomm/native_sim_v3/reference/decompiled_sts/com/megacrit/cardcrawl/characters/Ironclad.java",
    "models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt",
    "models/card_reward.pt",
    "models/shop_choice_prior_delta.pt",
]
missing = [path for path in required if not Path(path).exists()]
if missing:
    raise SystemExit("missing files: " + ", ".join(missing))
ckpt = torch.load("models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt", map_location="cpu", weights_only=False)
vocab = ckpt.get("entity_vocab") or []
if len(vocab) < 400:
    raise SystemExit(f"bad transformer entity_vocab length: {len(vocab)}")
print("bundle_ok torch=", torch.__version__, "entity_vocab=", len(vocab))
'''


def readme() -> str:
    return r'''# Aliyun Map DP Sweep v4

This bundle evaluates a narrowed map-DP coefficient search with the current best v5.104 runtime policy.

Base combat model:

`models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt`

This v4 bundle runs a seed1-600 default baseline first, skips round1/round2, then runs the round3/round4-sized local search around the current map default:

- center/default: `monster=-10`, `rest=70`, `elite=25`, `shop_gold=35`, `curse_bonus=60`
- local ranges:
  - `SPIRECOMM_MAP_DP_MONSTER_VALUE`: -16 to -4
  - `SPIRECOMM_MAP_DP_REST_VALUE`: 58 to 82
  - `SPIRECOMM_MAP_DP_ELITE_BASE`: 13 to 37
  - `SPIRECOMM_MAP_DP_SHOP_GOLD_UNIT_VALUE`: 23 to 47
  - `SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_BONUS`: 30 to 90

The swept variables are:

- `SPIRECOMM_MAP_DP_MONSTER_VALUE`
- `SPIRECOMM_MAP_DP_REST_VALUE`
- `SPIRECOMM_MAP_DP_ELITE_BASE`
- `SPIRECOMM_MAP_DP_SHOP_GOLD_UNIT_VALUE`
- `SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_BONUS`

Fixed defaults in this v4 bundle:

- `SPIRECOMM_MAP_DP_GREEN_ELITE_PENALTY`: 40
- `SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_BONUS`: 50
- `SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_GOLD_THRESHOLD`: 125
- `SPIRECOMM_MAP_DP_WINGED_OFFPATH_PENALTY`: 20

Round design:

- baseline: default center, seed1-600
- round3: 32 groups, seed1-300, includes default center
- round4: 8 groups, seed1-600

Setup:

```bash
bash scripts/install_deps.sh
bash scripts/smoke_map_dp_sweep.sh
```

Run:

```bash
WORKERS=$(nproc) bash run_map_dp_sweep_background.sh
```

Combat stall handling:

```bash
COMBAT_STALL_LIMIT=250 WORKERS=$(nproc) bash run_map_dp_sweep_background.sh
```

`COMBAT_STALL_LIMIT` stops a combat as timeout after N consecutive combat decisions with unchanged player/monster HP. This prevents policy loops from consuming the full 1500-step budget; set it to `0` to disable.

Status:

```bash
bash scripts/status.sh
tail -f logs/v5_104_map_dp_sweep_v4.log
```

Resume examples:

```bash
START_AT=round4_seed600 STOP_AFTER=round4_seed600 WORKERS=$(nproc) bash run_map_dp_sweep_background.sh
```

Stop:

```bash
bash scripts/stop_map_dp_sweep.sh
```

Download results:

- `map_sweep_runs/v5_104_map_dp_sweep_v4/final_leaderboard.json`
- `map_sweep_runs/v5_104_map_dp_sweep_v4/round*_results.json`
- `map_sweep_runs/v5_104_map_dp_sweep_v4/sweep_config.json`
- `logs/v5_104_map_dp_sweep_v4.log`

For exact paired seed analysis, download the whole `map_sweep_runs/v5_104_map_dp_sweep_v4` directory.
'''


def build(out_dir: Path) -> Path:
    out_dir = out_dir.expanduser().resolve()
    preserved_runs: Path | None = None
    existing_runs = out_dir / "map_sweep_runs"
    if existing_runs.exists():
        preserved_root = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}_preserve_", dir=str(out_dir.parent)))
        preserved_runs = preserved_root / "map_sweep_runs"
        shutil.copytree(existing_runs, preserved_runs, ignore=ignore_patterns)
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
    write_text(out_dir / "check_bundle.py", check_bundle(), executable=True)
    write_text(out_dir / "scripts" / "_common.sh", common_script(), executable=True)
    write_text(out_dir / "scripts" / "install_deps.sh", install_script(), executable=True)
    write_text(out_dir / "scripts" / "run_map_dp_sweep.sh", run_script(), executable=True)
    write_text(out_dir / "scripts" / "smoke_map_dp_sweep.sh", smoke_script(), executable=True)
    write_text(out_dir / "scripts" / "status.sh", status_script(), executable=True)
    write_text(out_dir / "scripts" / "stop_map_dp_sweep.sh", stop_script(), executable=True)
    write_text(out_dir / "run_map_dp_sweep_background.sh", run_background_script(), executable=True)
    if preserved_runs is not None and preserved_runs.exists():
        shutil.copytree(preserved_runs, out_dir / "map_sweep_runs", ignore=ignore_patterns)
        shutil.rmtree(preserved_runs.parent, ignore_errors=True)

    tar_path = out_dir.with_suffix(".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return tar_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Aliyun bundle for map DP rollout sweep.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    tar_path = build(args.out_dir)
    print(f"bundle_dir={args.out_dir.expanduser().resolve()}")
    print(f"tar_path={tar_path}")


if __name__ == "__main__":
    main()
