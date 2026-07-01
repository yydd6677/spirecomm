#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUT = Path("/home/yydd/下载/aliyun_v5_best_teacher_v12_200k_roots")

BEST_COMBAT_MODEL = "models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt"

MODEL_FILES = (
    "models/combat.pt",
    "models/v3_combat_scorer.pt",
    BEST_COMBAT_MODEL,
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
    "generate_v3_combat_teacher_dataset.py",
    "run_native_run.py",
    "setup.py",
    "README.md",
)


def ignore_patterns(_path: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".pytest_cache", ".mypy_cache", ".venv"}
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


def teacher_config_payload() -> dict:
    from spirecomm.ai.v3_combat_teacher import default_teacher_config

    return {"teacher_config": asdict(default_teacher_config())}


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

default_total_workers() {
  local cores
  cores="$(nproc 2>/dev/null || echo 1)"
  local reserve="${WORKER_RESERVE_CORES:-1}"
  if [[ ! "$reserve" =~ ^[0-9]+$ ]]; then reserve=1; fi
  local workers=$(( cores - reserve ))
  if [[ "$workers" -lt 1 ]]; then workers=1; fi
  echo "$workers"
}
'''


def run_generate_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

OUT_DIR="${OUT_DIR:-$ROOT/data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

WORKERS="${WORKERS:-$(default_total_workers)}"
COLLECT_WORKERS="${COLLECT_WORKERS:-$WORKERS}"
SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-6000}"
TARGET_ROOTS="${TARGET_ROOTS:-200000}"
SHARD_SIZE="${SHARD_SIZE:-256}"
LABEL_BATCH_SHARDS="${LABEL_BATCH_SHARDS:-4}"
SHARD_WRITE_WORKERS="${SHARD_WRITE_WORKERS:-4}"
LABEL_PIPELINE_BATCHES="${LABEL_PIPELINE_BATCHES:-0}"
MAX_STEPS_PER_SEED="${MAX_STEPS_PER_SEED:-1500}"
PER_SEED_ROOT_CAP="${PER_SEED_ROOT_CAP:-}"
DEVICE="${DEVICE:-cpu}"

APPEND_ARGS=()
if compgen -G "$OUT_DIR/shard_*.pt" > /dev/null; then
  APPEND_ARGS+=(--append-output)
fi
PER_SEED_ARGS=()
if [[ -n "$PER_SEED_ROOT_CAP" ]]; then
  PER_SEED_ARGS+=(--per-seed-root-cap "$PER_SEED_ROOT_CAP")
fi

export SPIRECOMM_LABEL_ROOT_TASK_BATCH_SIZE="${SPIRECOMM_LABEL_ROOT_TASK_BATCH_SIZE:-8}"
export SPIRECOMM_LABEL_ROOT_PROGRESS_INTERVAL="${SPIRECOMM_LABEL_ROOT_PROGRESS_INTERVAL:-256}"
export SPIRECOMM_V3_TEACHER_NON_POTION_ROOT_CACHE_SIZE="${SPIRECOMM_V3_TEACHER_NON_POTION_ROOT_CACHE_SIZE:-4096}"
export SPIRECOMM_V3_TEACHER_ROOT_CONTINUATION_CACHE="${SPIRECOMM_V3_TEACHER_ROOT_CONTINUATION_CACHE:-1}"
export SPIRECOMM_FAST_DISABLE_GC="${SPIRECOMM_FAST_DISABLE_GC:-1}"

echo "seedgen start out_dir=$OUT_DIR target_roots=$TARGET_ROOTS seeds=${SEED_START}-${SEED_END} collect_workers=$COLLECT_WORKERS label_workers=$WORKERS random_action_rate=0.0"
echo "model=models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt"
echo "teacher_config=configs/teacher_v12_default.json"

exec "$PY" -u "$ROOT/generate_v3_combat_teacher_dataset.py" \
  --source exploratory \
  --output-dir "$OUT_DIR" \
  "${APPEND_ARGS[@]}" \
  --seed-start "$SEED_START" \
  --seed-end "$SEED_END" \
  --target-roots "$TARGET_ROOTS" \
  --shard-size "$SHARD_SIZE" \
  --workers "$WORKERS" \
  --collect-workers "$COLLECT_WORKERS" \
  --label-batch-shards "$LABEL_BATCH_SHARDS" \
  --shard-write-workers "$SHARD_WRITE_WORKERS" \
  --label-pipeline-batches "$LABEL_PIPELINE_BATCHES" \
  --device "$DEVICE" \
  --combat-selector v3-candidate \
  --v3-combat-model "$ROOT/models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt" \
  --teacher-config-path "$ROOT/configs/teacher_v12_default.json" \
  --beam-width 24 \
  --node-budget 768 \
  --max-depth 20 \
  --random-action-rate 0.0 \
  "${PER_SEED_ARGS[@]}" \
  --max-steps-per-seed "$MAX_STEPS_PER_SEED" \
  --memory-log "$OUT_DIR/memory_log.jsonl"
'''


def run_background_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/logs"
LOG_PATH="${LOG_PATH:-$ROOT/logs/v5_best_teacher_v12_200k_roots.log}"
PID_PATH="${PID_PATH:-$ROOT/logs/v5_best_teacher_v12_200k_roots.pid}"
if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
  echo "already running pid=$(cat "$PID_PATH") log=$LOG_PATH"
  exit 0
fi
(
  cd "$ROOT"
  bash scripts/run_generate_roots.sh
) > "$LOG_PATH" 2>&1 &
echo $! > "$PID_PATH"
echo "started pid=$(cat "$PID_PATH") log=$LOG_PATH"
'''


def status_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_PATH="${PID_PATH:-$ROOT/logs/v5_best_teacher_v12_200k_roots.pid}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/v5_best_teacher_v12_200k_roots.log}"
OUT_DIR="${OUT_DIR:-$ROOT/data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k}"
echo "== process =="
if [[ -f "$PID_PATH" ]]; then
  pid="$(cat "$PID_PATH")"
  ps -fp "$pid" || true
else
  echo "no pid file"
fi
echo
echo "== latest log =="
tail -80 "$LOG_PATH" 2>/dev/null || true
echo
echo "== shards =="
count="$(find "$OUT_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l | tr -d ' ')"
echo "shards=$count approx_roots=$(( count * 256 )) out_dir=$OUT_DIR"
if [[ -f "$OUT_DIR/summary.json" ]]; then
  echo
  echo "== summary =="
  PY_BIN="$ROOT/.venv/bin/python"
  if [[ ! -x "$PY_BIN" ]]; then PY_BIN="${PYTHON:-python3}"; fi
  "$PY_BIN" - "$OUT_DIR/summary.json" <<'PY' 2>/dev/null || true
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
print(json.dumps({
    "roots": d.get("roots"),
    "exploratory_roots": d.get("exploratory_roots"),
    "processed_seeds": d.get("processed_seeds"),
    "teacher_version": d.get("teacher_version"),
    "root_stats": d.get("root_stats", {}),
}, ensure_ascii=False, indent=2)[:4000])
PY
fi
'''


def stop_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_PATH="${PID_PATH:-$ROOT/logs/v5_best_teacher_v12_200k_roots.pid}"
if [[ -f "$PID_PATH" ]]; then
  pid="$(cat "$PID_PATH")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "TERM pid=$pid"
    kill "$pid" 2>/dev/null || true
  fi
fi
sleep 2
pgrep -f 'generate_v3_combat_teacher_dataset.py.*v5_best_teacher_v12_norandom_200k' | xargs -r kill 2>/dev/null || true
'''


def summarize_script() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spirecomm.ai.v3_combat_dataset import load_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k"))
    parser.add_argument("--sample-shards", type=int, default=4)
    args = parser.parse_args()
    out = args.output_dir
    shards = sorted(out.glob("shard_*.pt"))
    total_roots = 0
    total_candidates = 0
    top_kind_counts: dict[str, int] = {}
    nonfinite = 0
    metadata_samples = []
    for shard in shards[: max(0, int(args.sample_shards))]:
        payload = load_shard(shard)
        metadata = payload.get("metadata") or {}
        metadata_samples.append({
            "path": str(shard),
            "root_count": metadata.get("root_count"),
            "random_action_rate": metadata.get("random_action_rate"),
            "combat_selector": metadata.get("combat_selector"),
            "collect_workers": metadata.get("collect_workers"),
            "teacher_version": metadata.get("teacher_version"),
            "beam_width": (metadata.get("teacher_config") or {}).get("beam_width"),
            "node_budget_per_root": (metadata.get("teacher_config") or {}).get("node_budget_per_root"),
            "teacher_survival_guard_enabled": (metadata.get("teacher_config") or {}).get("teacher_survival_guard_enabled"),
            "lethal_block_low_hp_protection": (metadata.get("teacher_config") or {}).get("lethal_block_low_hp_protection"),
            "potion_elite_room_reward_factor": (metadata.get("teacher_config") or {}).get("potion_elite_room_reward_factor"),
            "potion_cost_scale": (metadata.get("teacher_config") or {}).get("potion_cost_scale"),
        })
        for labeled in payload.get("roots") or []:
            total_roots += 1
            candidates = list(getattr(labeled, "candidates", []) or [])
            total_candidates += len(candidates)
            if candidates:
                best = max(candidates, key=lambda c: float(getattr(c, "teacher_q", 0.0)))
                kind = str((getattr(best, "action", {}) or {}).get("kind") or "")
                top_kind_counts[kind] = top_kind_counts.get(kind, 0) + 1
            for cand in candidates:
                q = float(getattr(cand, "teacher_q", 0.0))
                if q != q or q in {float("inf"), float("-inf")}:
                    nonfinite += 1
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    print(json.dumps({
        "output_dir": str(out),
        "shard_count": len(shards),
        "summary_roots": summary.get("roots"),
        "summary_exploratory_roots": summary.get("exploratory_roots"),
        "sampled_roots": total_roots,
        "sampled_candidates": total_candidates,
        "sampled_nonfinite_teacher_q": nonfinite,
        "sampled_top_kind_counts": top_kind_counts,
        "metadata_samples": metadata_samples,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''


def check_bundle_script() -> str:
    return r'''from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    import torch

    from spirecomm.ai.runtime_decision import build_runtime_selectors
    from spirecomm.ai.v3_combat_teacher import TEACHER_VERSION, default_teacher_config
    from spirecomm.native_sim_v3.content.characters import starting_profile

    required = [
        "generate_v3_combat_teacher_dataset.py",
        "models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt",
        "models/card_reward.pt",
        "models/event_choice.pt",
        "models/potion_use.pt",
        "models/shop_choice_prior_delta.pt",
        "models/upgrade_target.pt",
        "spirecomm/native_sim_v3/reference/decompiled_sts/com/megacrit/cardcrawl/characters/Ironclad.java",
        "configs/teacher_v12_default.json",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("missing required files: " + ", ".join(missing))

    cfg = default_teacher_config()
    if (cfg.beam_width, cfg.node_budget_per_root, cfg.max_depth) != (24, 768, 20):
        raise SystemExit(f"unexpected search defaults: {cfg.beam_width}/{cfg.node_budget_per_root}/{cfg.max_depth}")
    if not cfg.teacher_survival_guard_enabled or not cfg.lethal_block_low_hp_protection:
        raise SystemExit("teacher guard defaults are not enabled")
    if abs(float(cfg.potion_elite_room_reward_factor) - 1.3) > 1e-9:
        raise SystemExit(f"unexpected potion elite default: {cfg.potion_elite_room_reward_factor}")
    if abs(float(cfg.potion_cost_scale) - 1.1) > 1e-9:
        raise SystemExit(f"unexpected potion cost default: {cfg.potion_cost_scale}")
    frozen = json.loads((ROOT / "configs" / "teacher_v12_default.json").read_text(encoding="utf-8"))
    frozen_cfg = frozen.get("teacher_config") or {}
    if frozen_cfg.get("teacher_survival_guard_enabled") is not True:
        raise SystemExit("frozen config missing teacher_survival_guard_enabled=true")
    selectors = build_runtime_selectors(
        repo_root=ROOT,
        device="cpu",
        combat_selector="v3-candidate",
        v3_combat_model=ROOT / "models" / "v3_combat_transformer_v5_18_epoch003_rollout_best.pt",
    )
    unavailable = [name for name, selector in selectors.items() if name != "map" and not getattr(selector, "available", True)]
    if unavailable:
        raise SystemExit("unavailable selectors: " + ", ".join(unavailable))
    profile = starting_profile("IRONCLAD")
    print(f"root={ROOT}")
    print(f"torch={torch.__version__}")
    print(f"teacher_version={TEACHER_VERSION}")
    print(f"search_defaults=beam:{cfg.beam_width} node:{cfg.node_budget_per_root} depth:{cfg.max_depth}")
    print(f"teacher_guard={cfg.teacher_survival_guard_enabled}/{cfg.lethal_block_low_hp_protection}")
    print(f"potion_defaults=elite:{cfg.potion_elite_room_reward_factor} cost:{cfg.potion_cost_scale}")
    print(f"ironclad_hp={profile.current_hp}/{profile.max_hp}")
    print("bundle_check=ok")


if __name__ == "__main__":
    main()
'''


def smoke_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

SMOKE_DIR="${SMOKE_DIR:-$ROOT/_smoke_v5_best_teacher_v12_roots}"
rm -rf "$SMOKE_DIR"
OUT_DIR="$SMOKE_DIR" \
WORKERS="${WORKERS:-2}" \
SEED_START=1 \
SEED_END=20 \
TARGET_ROOTS=4 \
SHARD_SIZE=2 \
LABEL_BATCH_SHARDS=1 \
SHARD_WRITE_WORKERS=1 \
LABEL_PIPELINE_BATCHES=0 \
PER_SEED_ROOT_CAP=4 \
MAX_STEPS_PER_SEED=300 \
bash "$ROOT/scripts/run_generate_roots.sh"

"$PY" "$ROOT/scripts/summarize_roots.py" --output-dir "$SMOKE_DIR" --sample-shards 8
"$PY" - "$SMOKE_DIR" <<'PY'
import json
import sys
from pathlib import Path
from spirecomm.ai.v3_combat_dataset import load_shard

out = Path(sys.argv[1])
summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
assert summary["roots"] == 4, summary
assert summary["exploratory_roots"] == 4, summary
shards = sorted(out.glob("shard_*.pt"))
assert shards, "no smoke shards"
seen = set()
for shard in shards:
    payload = load_shard(shard)
    metadata = payload.get("metadata") or {}
    assert metadata.get("random_action_rate") == 0.0, metadata.get("random_action_rate")
    assert metadata.get("combat_selector") == "v3-candidate", metadata.get("combat_selector")
    cfg = metadata.get("teacher_config") or {}
    assert cfg.get("teacher_survival_guard_enabled") is True, cfg.get("teacher_survival_guard_enabled")
    assert cfg.get("lethal_block_low_hp_protection") is True, cfg.get("lethal_block_low_hp_protection")
    assert abs(float(cfg.get("potion_elite_room_reward_factor")) - 1.3) < 1e-9
    assert abs(float(cfg.get("potion_cost_scale")) - 1.1) < 1e-9
    roots = payload.get("roots") or []
    assert roots, "empty shard"
    for labeled in roots:
        rid = str(labeled.root.root_id)
        assert rid not in seen, rid
        seen.add(rid)
        candidates = list(labeled.candidates or [])
        assert len(candidates) == len(labeled.root.actions) >= 2
        ranks = sorted(int(c.teacher_rank) for c in candidates)
        assert ranks == list(range(len(candidates))), (rid, ranks)
        for cand in candidates:
            q = float(cand.teacher_q)
            assert q == q and q not in {float("inf"), float("-inf")}, (rid, q)
print("smoke_generation_check=ok")
PY
rm -rf "$SMOKE_DIR"
echo "smoke=ok"
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


def readme() -> str:
    return """# V5 Best Teacher v12 No-Random 200k Root Generator

Purpose: generate `200000` v3 combat teacher roots on server for retraining.

Baked policy:

- rollout model: `models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt`
- runtime guards: code defaults from the bundled repo
- random exploration: disabled, `random_action_rate = 0.0`
- teacher: frozen `configs/teacher_v12_default.json`
- teacher search budget: `beam_width=24`, `node_budget_per_root=768`, `max_depth=20`
- accepted teacher defaults include `potion_elite_room_reward_factor=1.3`, `potion_cost_scale=1.1`, `teacher_survival_guard_enabled=true`, `lethal_block_low_hp_protection=true`
- parallelism: `COLLECT_WORKERS` parallelizes rollout/root collection, `WORKERS` parallelizes teacher labeling. By default `COLLECT_WORKERS=$WORKERS`.

Install and smoke:

```bash
cd aliyun_v5_best_teacher_v12_200k_roots
bash scripts/install_deps.sh
bash scripts/smoke_generate.sh
```

Start generation:

```bash
WORKERS=125 COLLECT_WORKERS=125 bash scripts/run_generate_background.sh
```

If you want to reserve a few cores:

```bash
WORKERS=120 COLLECT_WORKERS=120 bash scripts/run_generate_background.sh
```

Monitor:

```bash
bash scripts/status.sh
tail -f logs/v5_best_teacher_v12_200k_roots.log
```

Stop:

```bash
bash scripts/stop.sh
```

Resume:

```bash
WORKERS=$(nproc) bash scripts/run_generate_background.sh
```

The generator automatically adds `--append-output` if existing `shard_*.pt` files are present, and it deduplicates root ids while resuming.

Check generated shards:

```bash
python scripts/summarize_roots.py --output-dir data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k
```

Download after completion:

- `data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k/shard_*.pt`
- `data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k/summary.json`
- `data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k/memory_log.jsonl`
- `logs/v5_best_teacher_v12_200k_roots.log`
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
    write_text(out_dir / "README_SERVER.md", readme())
    write_text(out_dir / "configs" / "teacher_v12_default.json", json.dumps(teacher_config_payload(), ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "check_bundle.py", check_bundle_script())
    write_text(out_dir / "scripts" / "_common.sh", common_script(), executable=True)
    write_text(out_dir / "scripts" / "install_deps.sh", install_script(), executable=True)
    write_text(out_dir / "scripts" / "run_generate_roots.sh", run_generate_script(), executable=True)
    write_text(out_dir / "scripts" / "run_generate_background.sh", run_background_script(), executable=True)
    write_text(out_dir / "scripts" / "status.sh", status_script(), executable=True)
    write_text(out_dir / "scripts" / "stop.sh", stop_script(), executable=True)
    write_text(out_dir / "scripts" / "summarize_roots.py", summarize_script(), executable=True)
    write_text(out_dir / "scripts" / "smoke_generate.sh", smoke_script(), executable=True)

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
