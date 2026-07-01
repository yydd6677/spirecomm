#!/usr/bin/env python3
from __future__ import annotations
# Allow this CLI to run directly from its workflow subdirectory.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import pickle
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from scripts.run_value.collect_noncombat_hard_roots import (
    _action_match,
    _candidate_actions_for_source,
    _compact_action,
    _cuda_available,
    _resolve_combat_device,
)
from spirecomm.ai.runtime_decision import build_runtime_selectors, choose_model_required_action


TERMINAL_PHASES = {"GAME_OVER", "COMPLETE", "VICTORY"}
_SELECTORS: dict[str, Any] | None = None
_CONFIG: dict[str, Any] = {}


def _init_worker(config: dict[str, Any]) -> None:
    global _CONFIG, _SELECTORS
    _CONFIG = dict(config)
    for key, value in (_CONFIG.get("env") or {}).items():
        if value is not None:
            os.environ[str(key)] = str(value)
    try:
        import gc

        gc.disable()
    except Exception:
        pass
    try:
        from spirecomm.ai.torch_compat import torch

        if torch is not None:
            torch.set_num_threads(max(1, int(_CONFIG.get("torch_threads") or 1)))
    except Exception:
        pass
    _SELECTORS = build_runtime_selectors(
        repo_root=Path(_CONFIG["repo_root"]),
        device=str(_CONFIG["device"]),
        combat_device=str(_CONFIG["combat_device"]),
        combat_model=Path(_CONFIG["combat_model"]),
        combat_selector=str(_CONFIG["combat_selector"]),
        v3_combat_model=Path(_CONFIG["v3_combat_model"]),
        card_reward_model=Path(_CONFIG["card_reward_model"]),
        shop_model=Path(_CONFIG["shop_choice_model"]),
    )


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_env_blob(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def _clone_env(env: Any) -> Any:
    return pickle.loads(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL))


def _current_candidates_for_record(env: Any, record: dict[str, Any]) -> list[dict[str, Any]]:
    actions = [dict(action) for action in env.legal_actions()]
    source = str(record.get("source") or "")
    score_count = int(record.get("score_count") or 0)
    scores = [0.0] * score_count
    return _candidate_actions_for_source(source, actions, scores) if score_count else actions


def _candidate_key(action: dict[str, Any]) -> str:
    compact = _compact_action(action)
    material = json.dumps(compact, ensure_ascii=False, sort_keys=True)
    return material


def _branch_score(result: dict[str, Any]) -> float:
    # This is only a sortable proxy for diagnostics. Training should consume the
    # raw branch metrics and can choose a different target without recollecting.
    floor = int(result.get("floor") or 0)
    hp = int(result.get("hp") or 0)
    max_hp = max(1, int(result.get("max_hp") or 1))
    score = floor * 100.0 + 10.0 * hp / max_hp
    if result.get("won"):
        score += 1000.0
    if result.get("dead"):
        score -= 250.0
    if result.get("timed_out"):
        score -= 50.0
    return float(score)


def _continue_policy(env: Any, *, start_floor: int) -> dict[str, Any]:
    assert _SELECTORS is not None
    max_steps = int(_CONFIG["branch_max_steps"])
    horizon_floors = int(_CONFIG["horizon_floors"])
    max_floor = int(_CONFIG["max_floor"])
    source_counts: Counter[str] = Counter()
    error: str | None = None
    for step in range(max_steps):
        if env.phase in TERMINAL_PHASES:
            break
        if int(getattr(env, "floor", 0) or 0) > max_floor:
            break
        if horizon_floors > 0 and int(getattr(env, "floor", 0) or 0) >= start_floor + horizon_floors:
            break
        try:
            phase = str(getattr(env, "phase", ""))
            action, _scores, source = choose_model_required_action(env, _SELECTORS, return_scores=(phase != "COMBAT"))
            source_counts[str(source)] += 1
            env.step(action)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
    result = {
        "phase": str(getattr(env, "phase", "")),
        "floor": int(getattr(env, "floor", 0) or 0),
        "hp": int(getattr(getattr(env, "player", None), "current_hp", 0) or 0),
        "max_hp": int(getattr(getattr(env, "player", None), "max_hp", 0) or 0),
        "gold": int(getattr(env, "gold", 0) or 0),
        "deck_size": len(list(getattr(env, "deck", []) or [])),
        "relic_count": len(list(getattr(env, "relics", []) or [])),
        "potion_count": len(list(getattr(env, "potions", []) or [])),
        "steps": int(step + 1 if "step" in locals() else 0),
        "won": str(getattr(env, "phase", "")) in {"COMPLETE", "VICTORY"},
        "dead": str(getattr(env, "phase", "")) == "GAME_OVER",
        "timed_out": str(getattr(env, "phase", "")) not in TERMINAL_PHASES
        and not error
        and int(step + 1 if "step" in locals() else 0) >= max_steps,
        "error": error,
        "source_counts": dict(source_counts),
    }
    result["branch_score"] = _branch_score(result)
    return result


def _label_one(record: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    input_root = Path(_CONFIG["input_root"])
    blob_path = input_root / str(record["env_blob_path"])
    env = _load_env_blob(blob_path)
    start_floor = int(getattr(env, "floor", 0) or record.get("floor") or 0)
    candidates = _current_candidates_for_record(env, record)
    max_candidates = int(_CONFIG["max_candidates_per_root"])
    if max_candidates > 0 and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    branches: list[dict[str, Any]] = []
    error_traceback: str | None = None
    for index, candidate in enumerate(candidates):
        branch_env = _clone_env(env)
        try:
            branch_env.step(dict(candidate))
            result = _continue_policy(branch_env, start_floor=start_floor)
            branch_error = None
        except Exception as exc:
            result = {
                "phase": str(getattr(branch_env, "phase", "")),
                "floor": int(getattr(branch_env, "floor", 0) or 0),
                "hp": int(getattr(getattr(branch_env, "player", None), "current_hp", 0) or 0),
                "max_hp": int(getattr(getattr(branch_env, "player", None), "max_hp", 0) or 0),
                "won": False,
                "dead": False,
                "timed_out": False,
                "error": f"{type(exc).__name__}: {exc}",
                "branch_score": -999999.0,
            }
            branch_error = traceback.format_exc()
            error_traceback = branch_error if error_traceback is None else error_traceback + "\n" + branch_error
        score = None
        if index < len(record.get("scores") or []):
            try:
                score = float(record["scores"][index])
            except Exception:
                score = None
        branches.append(
            {
                "candidate_index": index,
                "candidate": _compact_action(candidate),
                "model_score": score,
                "was_chosen": _action_match(record.get("action") or {}, candidate),
                "result": result,
            }
        )
    if branches:
        best_index = max(range(len(branches)), key=lambda idx: float(branches[idx]["result"].get("branch_score") or -999999.0))
        chosen_index = next((idx for idx, branch in enumerate(branches) if branch["was_chosen"]), None)
    else:
        best_index = None
        chosen_index = None
    chosen_score = (
        float(branches[chosen_index]["result"].get("branch_score") or 0.0)
        if chosen_index is not None
        else None
    )
    best_score = (
        float(branches[best_index]["result"].get("branch_score") or 0.0)
        if best_index is not None
        else None
    )
    chosen_regret = None if chosen_score is None or best_score is None else best_score - chosen_score
    chosen_is_best = None if chosen_regret is None else chosen_regret <= 1e-6
    return {
        "root_id": record.get("root_id"),
        "seed": record.get("seed"),
        "source": record.get("source"),
        "floor": record.get("floor"),
        "phase": record.get("phase"),
        "env_blob_path": record.get("env_blob_path"),
        "chosen_index": chosen_index,
        "branch_best_index": best_index,
        "branch_chosen_is_best": chosen_is_best,
        "branch_chosen_regret": chosen_regret,
        "branch_count": len(branches),
        "branches": branches,
        "seconds": time.time() - started,
        "error_traceback": error_traceback,
    }


def _load_existing_labels(output_dir: Path) -> set[str]:
    path = output_dir / "branch_labels.jsonl"
    if not path.exists():
        return set()
    root_ids: set[str] = set()
    for row in _iter_jsonl(path):
        if row.get("root_id"):
            root_ids.add(str(row["root_id"]))
    return root_ids


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            clean = dict(row)
            clean.pop("error_traceback", None)
            handle.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")


def _summarize(labels: list[dict[str, Any]], output_dir: Path, started: float) -> dict[str, Any]:
    regrets = [
        float(row["branch_chosen_regret"])
        for row in labels
        if row.get("branch_chosen_regret") is not None and math.isfinite(float(row["branch_chosen_regret"]))
    ]
    source_counts = Counter(str(row.get("source") or "") for row in labels)
    source_regrets: dict[str, list[float]] = {}
    for row in labels:
        regret = row.get("branch_chosen_regret")
        if regret is None:
            continue
        source_regrets.setdefault(str(row.get("source") or ""), []).append(float(regret))
    return {
        "count": len(labels),
        "output_dir": str(output_dir),
        "seconds": time.time() - started,
        "source_counts": dict(source_counts.most_common()),
        "branch_best_agreement": (
            sum(1 for row in labels if row.get("branch_chosen_is_best")) / max(1, len(labels))
        ),
        "mean_branch_regret": mean(regrets) if regrets else None,
        "positive_regret_count": sum(1 for value in regrets if value > 1e-6),
        "source_mean_regret": {
            source: mean(values) for source, values in sorted(source_regrets.items()) if values
        },
    }


def _parse_env_pairs(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--env expects NAME=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Label saved non-combat roots by branch rollouts.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources", default="card_reward,card_reward_skip,upgrade_target")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summary-interval", type=int, default=10)
    parser.add_argument("--horizon-floors", type=int, default=6)
    parser.add_argument("--branch-max-steps", type=int, default=450)
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-candidates-per-root", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default=os.environ.get("SPIRECOMM_EVAL_COMBAT_DEVICE", "auto"))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--combat-selector", choices=["legacy-slot", "v3-candidate", "v3-teacher"], default="v3-candidate")
    parser.add_argument("--combat-model", type=Path, default=Path("models/combat.pt"))
    parser.add_argument("--v3-combat-model", type=Path, default=Path("models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt"))
    parser.add_argument("--card-reward-model", type=Path, default=Path("models/card_reward.pt"))
    parser.add_argument("--shop-choice-model", type=Path, default=Path(os.environ.get("SPIRECOMM_SHOP_CHOICE_MODEL_PATH", "models/shop_choice_prior_delta.pt")))
    parser.add_argument("--shop-policy", choices=["model", "value"], default=os.environ.get("SPIRECOMM_SHOP_POLICY", "value"))
    parser.add_argument("--fast-disable-runtime-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--env", action="append", default=[], help="Extra environment override NAME=VALUE.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {token.strip() for token in args.sources.split(",") if token.strip()}
    rows = [
        row
        for row in _iter_jsonl(args.input_root / "hard_decisions.jsonl")
        if row.get("env_blob_path") and str(row.get("source") or "") in sources
    ]
    if args.limit > 0:
        rows = rows[: int(args.limit)]
    if args.resume:
        existing = _load_existing_labels(output_dir)
        rows = [row for row in rows if str(row.get("root_id")) not in existing]
    else:
        existing = set()

    env = _parse_env_pairs(args.env)
    env["SPIRECOMM_SHOP_POLICY"] = str(args.shop_policy)
    if args.fast_disable_runtime_search:
        env["SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOPK"] = "0"
        env["SPIRECOMM_MAP_ROLLOUT_RERANK_TOPK"] = "0"

    args.combat_device = _resolve_combat_device(args.device, args.combat_device)
    config = {
        "repo_root": str(_REPO_ROOT),
        "input_root": str(args.input_root),
        "output_dir": str(output_dir),
        "horizon_floors": int(args.horizon_floors),
        "branch_max_steps": int(args.branch_max_steps),
        "max_floor": int(args.max_floor),
        "max_candidates_per_root": int(args.max_candidates_per_root),
        "device": str(args.device),
        "combat_device": str(args.combat_device),
        "torch_threads": int(args.torch_threads),
        "combat_selector": str(args.combat_selector),
        "combat_model": str(args.combat_model),
        "v3_combat_model": str(args.v3_combat_model),
        "card_reward_model": str(args.card_reward_model),
        "shop_choice_model": str(args.shop_choice_model),
        "env": env,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"labeling noncombat branches: pending={len(rows)} existing={len(existing)} workers={args.workers} "
        f"horizon_floors={args.horizon_floors} output={output_dir}",
        flush=True,
    )

    started = time.time()
    labels: list[dict[str, Any]] = []
    label_path = output_dir / "branch_labels.jsonl"
    if args.resume and label_path.exists():
        labels.extend(_iter_jsonl(label_path))
    completed = 0
    with ProcessPoolExecutor(max_workers=int(args.workers), initializer=_init_worker, initargs=(config,)) as executor:
        futures = {executor.submit(_label_one, row): row.get("root_id") for row in rows}
        for future in as_completed(futures):
            label = future.result()
            labels.append(label)
            _write_jsonl(label_path, [label], append=True)
            if label.get("error_traceback"):
                error_dir = output_dir / "errors"
                error_dir.mkdir(parents=True, exist_ok=True)
                (error_dir / f"{label.get('root_id')}.txt").write_text(str(label["error_traceback"]), encoding="utf-8")
            completed += 1
            if completed == 1 or (args.summary_interval and completed % int(args.summary_interval) == 0):
                summary = _summarize(labels, output_dir, started)
                (output_dir / "summary_partial.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(
                    f"completed={completed}/{len(rows)} labels={len(labels)} "
                    f"agreement={summary['branch_best_agreement']:.3f} "
                    f"mean_regret={summary['mean_branch_regret']} elapsed={summary['seconds']:.1f}s",
                    flush=True,
                )

    summary = _summarize(labels, output_dir, started)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    if "fork" in mp.get_all_start_methods():
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass
    main()
