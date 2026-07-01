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
from statistics import mean, median
from typing import Any

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from spirecomm.ai.runtime_decision import build_runtime_selectors, choose_model_required_action
from spirecomm.native_sim_v3 import NativeRunEnv


TERMINAL_PHASES = {"GAME_OVER", "COMPLETE", "VICTORY"}
DEFAULT_TARGET_SOURCES = (
    "card_reward",
    "card_reward_skip",
    "upgrade_target",
    "campfire",
    "campfire_low_hp_rest_guard",
    "shop",
    "shop_value",
    "shop_value_leave",
    "event",
    "event_dead_adventurer_macro",
    "event_dead_adventurer_lowhp_cap",
    "event_golden_idol_max_hp_over_damage",
    "event_scrap_ooze_lowhp_cap",
    "event_lowhp_cost_guard",
    "event_chosen_lowhp_cost_guard",
    "map_dp",
    "map_dp_rollout_rerank",
)

_SELECTORS: dict[str, Any] | None = None
_CONFIG: dict[str, Any] = {}


def _disable_gc_if_requested() -> None:
    if str(os.environ.get("SPIRECOMM_FAST_DISABLE_GC", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return
    import gc

    gc.disable()


def _cuda_available() -> bool:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in {"", "-1"}:
        return False
    if Path("/proc/driver/nvidia/gpus").exists():
        return True
    return False


def _resolve_combat_device(device: str, combat_device: str) -> str:
    token = str(combat_device or "auto").strip().lower()
    if token in {"", "auto"}:
        return "cuda" if _cuda_available() else str(device or "cpu")
    return str(combat_device)


def _prewarm_native_content_caches() -> None:
    try:
        from spirecomm.native_sim_v3.content.cards import card_catalog, card_pools, starter_deck
        from spirecomm.native_sim_v3.content.encounters import encounter_catalog
        from spirecomm.native_sim_v3.content.events import event_catalog
        from spirecomm.native_sim_v3.content.map_rules import map_rules
        from spirecomm.native_sim_v3.content.potions import potion_pool
        from spirecomm.native_sim_v3.content.relics import relic_catalog, relic_pools, starter_relics
        from spirecomm.native_sim_v3.content.shop import shop_rules

        card_catalog()
        card_pools("IRONCLAD")
        starter_deck("IRONCLAD")
        relic_catalog()
        relic_pools("IRONCLAD")
        starter_relics("IRONCLAD")
        potion_pool("IRONCLAD")
        encounter_catalog()
        event_catalog()
        map_rules()
        shop_rules()
    except Exception:
        pass


def _init_worker(config: dict[str, Any]) -> None:
    global _CONFIG, _SELECTORS
    _CONFIG = dict(config)
    _disable_gc_if_requested()
    for key, value in (_CONFIG.get("env") or {}).items():
        if value is not None:
            os.environ[str(key)] = str(value)
    try:
        from spirecomm.ai.torch_compat import torch

        if torch is not None:
            torch.set_num_threads(max(1, int(_CONFIG.get("torch_threads") or 1)))
    except Exception:
        pass
    _prewarm_native_content_caches()
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


def _compact_card(card: Any) -> dict[str, Any]:
    if not isinstance(card, dict):
        return {"name": str(card)}
    keys = (
        "card_id",
        "id",
        "name",
        "type",
        "rarity",
        "cost",
        "base_cost",
        "cost_for_turn",
        "upgrades",
        "misc",
        "exhausts",
        "ethereal",
        "has_target",
    )
    return {key: card.get(key) for key in keys if key in card}


def _compact_relic(relic: Any) -> dict[str, Any]:
    if not isinstance(relic, dict):
        return {"name": str(relic)}
    return {
        key: relic.get(key)
        for key in ("relic_id", "id", "name", "tier", "counter", "used_up")
        if key in relic
    }


def _compact_potion(potion: Any) -> dict[str, Any]:
    if not isinstance(potion, dict):
        return {"name": str(potion)}
    return {
        key: potion.get(key)
        for key in ("potion_id", "id", "name", "requires_target", "can_use", "can_discard")
        if key in potion
    }


def _compact_action(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {"value": str(action)}
    keys = (
        "kind",
        "name",
        "label",
        "choice_index",
        "node_id",
        "symbol",
        "item_kind",
        "item_id",
        "price",
        "card_id",
        "potion_id",
        "relic_id",
        "card_index",
        "source_index",
        "target_index",
        "potion_index",
        "amount",
        "mode",
        "bonus",
        "drawback",
    )
    payload = {key: action.get(key) for key in keys if key in action}
    if isinstance(action.get("card"), dict):
        payload["card"] = _compact_card(action["card"])
    return payload


def _state_summary(state: dict[str, Any], *, full_state: bool = False) -> dict[str, Any]:
    if full_state:
        return state
    map_state = state.get("map_state") if isinstance(state.get("map_state"), dict) else {}
    current_node = map_state.get("current_node") if isinstance(map_state, dict) else None
    return {
        "seed": state.get("seed"),
        "ascension_level": state.get("ascension_level"),
        "act": state.get("act"),
        "dungeon_id": state.get("dungeon_id"),
        "floor": state.get("floor"),
        "phase": state.get("phase"),
        "screen": state.get("screen"),
        "room_type": state.get("room_type"),
        "event_id": state.get("event_id"),
        "current_hp": state.get("current_hp"),
        "max_hp": state.get("max_hp"),
        "gold": state.get("gold"),
        "has_ruby_key": state.get("has_ruby_key"),
        "has_emerald_key": state.get("has_emerald_key"),
        "has_sapphire_key": state.get("has_sapphire_key"),
        "act_boss": state.get("act_boss"),
        "deck": [_compact_card(card) for card in list(state.get("deck") or [])],
        "relics": [_compact_relic(relic) for relic in list(state.get("relics") or [])],
        "potions": [_compact_potion(potion) for potion in list(state.get("potions") or [])],
        "map_current_node": current_node,
        "choice_list": [_compact_action(action) for action in list(state.get("choice_list") or [])],
    }


def _candidate_actions_for_source(source: str, actions: list[dict[str, Any]], scores: list[float]) -> list[dict[str, Any]]:
    if source in {"card_reward", "card_reward_skip"}:
        reward_actions = [action for action in actions if action.get("kind") == "card_reward"]
        skip_action = next((action for action in actions if action.get("kind") in {"skip", "proceed"}), None)
        candidates = list(reward_actions)
        if skip_action is not None:
            candidates.append(skip_action)
        if len(candidates) == len(scores):
            return candidates
    if len(actions) == len(scores):
        return list(actions)
    return list(actions)


def _action_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left is right:
        return True
    keys = (
        "kind",
        "choice_index",
        "node_id",
        "item_kind",
        "item_id",
        "price",
        "card_id",
        "potion_id",
        "relic_id",
        "card_index",
        "source_index",
        "target_index",
        "potion_index",
        "name",
        "label",
    )
    shared = [key for key in keys if key in left and key in right]
    if shared and all(left.get(key) == right.get(key) for key in shared):
        return True
    left_card = left.get("card") if isinstance(left.get("card"), dict) else {}
    right_card = right.get("card") if isinstance(right.get("card"), dict) else {}
    if left.get("kind") == right.get("kind") and left_card and right_card:
        return (
            (left_card.get("card_id") or left_card.get("id") or left_card.get("name"))
            == (right_card.get("card_id") or right_card.get("id") or right_card.get("name"))
            and int(left_card.get("upgrades") or 0) == int(right_card.get("upgrades") or 0)
        )
    return False


def _score_metrics(
    *,
    action: dict[str, Any],
    actions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    scores: list[float],
) -> dict[str, Any]:
    numeric_scores = [float(score) for score in scores if score is not None]
    metrics: dict[str, Any] = {
        "score_count": len(scores),
        "candidate_count": len(candidates),
        "legal_action_count": len(actions),
        "scores_aligned": len(scores) == len(candidates) and len(scores) > 0,
        "chosen_index": None,
        "best_index": None,
        "chosen_score": None,
        "best_score": None,
        "second_score": None,
        "best_margin": None,
        "chosen_gap_to_best": None,
        "chosen_is_score_best": None,
    }
    if not scores or len(scores) != len(candidates):
        return metrics
    chosen_index = next((index for index, candidate in enumerate(candidates) if _action_match(action, candidate)), None)
    best_index = max(range(len(scores)), key=lambda index: float(scores[index]))
    ordered = sorted(((float(score), index) for index, score in enumerate(scores)), reverse=True)
    best_score = float(scores[best_index])
    second_score = float(ordered[1][0]) if len(ordered) > 1 else None
    chosen_score = float(scores[chosen_index]) if chosen_index is not None else None
    metrics.update(
        {
            "chosen_index": chosen_index,
            "best_index": best_index,
            "chosen_score": chosen_score,
            "best_score": best_score,
            "second_score": second_score,
            "best_margin": None if second_score is None else best_score - second_score,
            "chosen_gap_to_best": None if chosen_score is None else chosen_score - best_score,
            "chosen_is_score_best": None if chosen_index is None else chosen_index == best_index,
            "score_min": min(numeric_scores) if numeric_scores else None,
            "score_max": max(numeric_scores) if numeric_scores else None,
            "score_mean": mean(numeric_scores) if numeric_scores else None,
        }
    )
    return metrics


def _is_target_decision(source: str, phase: str, target_sources: set[str], target_phases: set[str]) -> bool:
    if source in target_sources:
        return True
    if phase in target_phases:
        return True
    return False


def _hard_flags(record: dict[str, Any], *, config: dict[str, Any]) -> dict[str, bool]:
    margin = record.get("best_margin")
    chosen_gap = record.get("chosen_gap_to_best")
    final_floor = int(record.get("final_floor") or 0)
    decision_floor = int(record.get("floor") or 0)
    hard_margin_max = float(config["hard_margin_max"])
    hard_final_floor_max = int(config["hard_final_floor_max"])
    death_window = int(config["hard_death_floor_window"])
    low_margin = margin is not None and math.isfinite(float(margin)) and float(margin) <= hard_margin_max
    chosen_not_best = chosen_gap is not None and math.isfinite(float(chosen_gap)) and float(chosen_gap) < -1e-6
    bad_final = final_floor <= hard_final_floor_max
    died_soon = bool(record.get("final_dead")) and final_floor <= decision_floor + death_window
    missing_scores = not bool(record.get("scores_aligned"))
    return {
        "hard_low_margin": bool(low_margin),
        "hard_chosen_not_best": bool(chosen_not_best),
        "hard_bad_final": bool(bad_final),
        "hard_died_soon": bool(died_soon),
        "hard_missing_scores": bool(missing_scores),
    }


def _write_env_blob(env: NativeRunEnv, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=1) as handle:
        pickle.dump(env, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _run_seed(seed: int) -> dict[str, Any]:
    assert _SELECTORS is not None
    started = time.time()
    config = _CONFIG
    target_sources = set(config["target_sources"])
    target_phases = set(config["target_phases"])
    blob_sources = set(config["blob_sources"])
    output_dir = Path(config["output_dir"])
    blob_dir = output_dir / "env_blobs"

    env = NativeRunEnv(seed=seed, ascension_level=int(config["ascension"]), enable_neow=True)
    decisions: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    error: str | None = None
    error_traceback: str | None = None
    max_steps = int(config["max_steps"])
    max_floor = int(config["max_floor"])
    save_env_blobs = bool(config["save_env_blobs"])
    max_blobs_per_seed = int(config["max_env_blobs_per_seed"])
    blob_margin_max = float(config["blob_score_margin_max"])
    full_state = bool(config["record_full_state"])
    blob_count = 0
    step_count = 0

    for step_index in range(max_steps):
        if env.phase in TERMINAL_PHASES or int(env.floor) > max_floor:
            break
        pre_phase = str(env.phase)
        try:
            pre_state = env.state()
            actions = [dict(action) for action in env.legal_actions()]
            want_scores = pre_phase != "COMBAT"
            action, scores, source = choose_model_required_action(env, _SELECTORS, return_scores=want_scores)
            scores = [float(score) for score in list(scores or [])]
            source = str(source)
            source_counts[source] += 1
            if _is_target_decision(source, pre_phase, target_sources, target_phases):
                candidates = _candidate_actions_for_source(source, actions, scores)
                metrics = _score_metrics(action=dict(action), actions=actions, candidates=candidates, scores=scores)
                root_id = f"seed{seed:06d}_step{step_index:04d}_idx{len(decisions):04d}"
                record: dict[str, Any] = {
                    "root_id": root_id,
                    "seed": int(seed),
                    "step": int(step_index),
                    "phase": pre_phase,
                    "source": source,
                    "floor": int(pre_state.get("floor") or 0),
                    "act": int(pre_state.get("act") or 0),
                    "room_type": str(pre_state.get("room_type") or ""),
                    "event_id": pre_state.get("event_id"),
                    "hp": int(pre_state.get("current_hp") or 0),
                    "max_hp": int(pre_state.get("max_hp") or 0),
                    "gold": int(pre_state.get("gold") or 0),
                    "deck_size": len(list(pre_state.get("deck") or [])),
                    "relic_count": len(list(pre_state.get("relics") or [])),
                    "potion_count": len(list(pre_state.get("potions") or [])),
                    "action": _compact_action(action),
                    "legal_actions": [_compact_action(candidate) for candidate in actions],
                    "candidates": [_compact_action(candidate) for candidate in candidates],
                    "scores": scores,
                    "state": _state_summary(pre_state, full_state=full_state),
                    **metrics,
                }
                if save_env_blobs and blob_count < max_blobs_per_seed and source in blob_sources:
                    margin = record.get("best_margin")
                    should_save = blob_margin_max < 0.0 or (
                        margin is not None and math.isfinite(float(margin)) and float(margin) <= blob_margin_max
                    )
                    if should_save:
                        blob_path = blob_dir / f"{root_id}.pkl.gz"
                        _write_env_blob(env, blob_path)
                        record["env_blob_path"] = str(blob_path.relative_to(output_dir))
                        blob_count += 1
                decisions.append(record)
            env.step(action)
            step_count += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            error_traceback = traceback.format_exc()
            break

    result = {
        "seed": int(seed),
        "ascension": int(config["ascension"]),
        "phase": str(env.phase),
        "floor": int(env.floor),
        "hp": int(env.player.current_hp),
        "max_hp": int(env.player.max_hp),
        "gold": int(env.gold),
        "deck_size": len(env.deck),
        "relic_count": len(env.relics),
        "potion_count": len(env.potions),
        "steps": int(step_count),
        "won": str(env.phase) in {"COMPLETE", "VICTORY"},
        "dead": str(env.phase) == "GAME_OVER",
        "timed_out": str(env.phase) not in TERMINAL_PHASES and error is None,
        "error": error,
        "seconds": time.time() - started,
        "source_counts": dict(source_counts),
        "decision_count": len(decisions),
        "env_blob_count": blob_count,
    }
    for record in decisions:
        record.update(
            {
                "final_phase": result["phase"],
                "final_floor": result["floor"],
                "final_hp": result["hp"],
                "final_max_hp": result["max_hp"],
                "final_gold": result["gold"],
                "final_steps": result["steps"],
                "final_won": result["won"],
                "final_dead": result["dead"],
                "final_timed_out": result["timed_out"],
                "future_floor_delta": int(result["floor"]) - int(record.get("floor") or 0),
            }
        )
        flags = _hard_flags(record, config=config)
        record.update(flags)
        record["is_hard"] = any(flags.values())
    return {"result": result, "decisions": decisions, "error_traceback": error_traceback}


def _load_existing_results(output_dir: Path) -> dict[int, dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    path = output_dir / "results.jsonl"
    if not path.exists():
        return by_seed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                by_seed[int(result["seed"])] = result
            except Exception:
                continue
    return by_seed


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_decision_rows(decision_dir: Path, seeds: set[int] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not decision_dir.exists():
        return rows
    for path in sorted(decision_dir.glob("seed_*.jsonl")):
        if seeds is not None:
            try:
                seed = int(path.stem.split("_", 1)[1])
            except Exception:
                continue
            if seed not in seeds:
                continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _summarize(results: list[dict[str, Any]], decisions: list[dict[str, Any]], output_dir: Path, started: float) -> dict[str, Any]:
    floors = [int(result.get("floor") or 0) for result in results]
    errors = [result for result in results if result.get("error")]
    source_counts: Counter[str] = Counter()
    for result in results:
        source_counts.update(result.get("source_counts") or {})
    decision_sources = Counter(str(row.get("source") or "") for row in decisions)
    hard_sources = Counter(str(row.get("source") or "") for row in decisions if row.get("is_hard"))
    margins = [
        float(row["best_margin"])
        for row in decisions
        if row.get("best_margin") is not None and math.isfinite(float(row["best_margin"]))
    ]
    chosen_gaps = [
        float(row["chosen_gap_to_best"])
        for row in decisions
        if row.get("chosen_gap_to_best") is not None and math.isfinite(float(row["chosen_gap_to_best"]))
    ]
    return {
        "count": len(results),
        "output_dir": str(output_dir),
        "seconds": time.time() - started,
        "mean_floor": mean(floors) if floors else 0.0,
        "median_floor": median(floors) if floors else 0.0,
        "win_count": sum(1 for result in results if result.get("won")),
        "death_count": sum(1 for result in results if result.get("dead")),
        "timeout_count": sum(1 for result in results if result.get("timed_out")),
        "error_count": len(errors),
        "decision_count": len(decisions),
        "hard_decision_count": sum(1 for row in decisions if row.get("is_hard")),
        "env_blob_count": sum(int(result.get("env_blob_count") or 0) for result in results),
        "source_counts": dict(source_counts.most_common()),
        "decision_source_counts": dict(decision_sources.most_common()),
        "hard_source_counts": dict(hard_sources.most_common()),
        "mean_best_margin": mean(margins) if margins else None,
        "median_best_margin": median(margins) if margins else None,
        "mean_chosen_gap_to_best": mean(chosen_gaps) if chosen_gaps else None,
        "chosen_not_best_count": sum(1 for row in decisions if row.get("hard_chosen_not_best")),
        "low_margin_count": sum(1 for row in decisions if row.get("hard_low_margin")),
        "bad_final_count": sum(1 for row in decisions if row.get("hard_bad_final")),
        "died_soon_count": sum(1 for row in decisions if row.get("hard_died_soon")),
        "missing_scores_count": sum(1 for row in decisions if row.get("hard_missing_scores")),
        "errors": [
            {"seed": result.get("seed"), "floor": result.get("floor"), "phase": result.get("phase"), "error": result.get("error")}
            for result in errors[:20]
        ],
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
    parser = argparse.ArgumentParser(description="Collect compact on-policy non-combat hard decision roots.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-interval", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default=os.environ.get("SPIRECOMM_EVAL_COMBAT_DEVICE", "auto"))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--combat-selector", choices=["legacy-slot", "v3-candidate", "v3-teacher"], default="v3-candidate")
    parser.add_argument("--combat-model", type=Path, default=Path("models/combat.pt"))
    parser.add_argument("--v3-combat-model", type=Path, default=Path("models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt"))
    parser.add_argument("--card-reward-model", type=Path, default=Path("models/card_reward.pt"))
    parser.add_argument("--shop-choice-model", type=Path, default=Path(os.environ.get("SPIRECOMM_SHOP_CHOICE_MODEL_PATH", "models/shop_choice_prior_delta.pt")))
    parser.add_argument("--shop-policy", choices=["model", "value"], default=os.environ.get("SPIRECOMM_SHOP_POLICY", "value"))
    parser.add_argument("--shop-value-price-cost", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_PRICE_COST", "0.044348003822393976")))
    parser.add_argument("--shop-value-reserve-shortfall-cost", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_RESERVE_SHORTFALL_COST", "0.043490245962190935")))
    parser.add_argument("--shop-value-future-shop-reserve", type=int, default=int(os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE", "120")))
    parser.add_argument("--shop-value-future-shop-horizon", type=int, default=int(os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_HORIZON", "5")))
    parser.add_argument("--shop-value-card-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_SCALE", "4.6262945279949435")))
    parser.add_argument("--shop-value-potion-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_POTION_SCALE", "0.5084989138155764")))
    parser.add_argument("--shop-value-relic-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_RELIC_SCALE", "0.8")))
    parser.add_argument("--shop-value-threshold", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_THRESHOLD", "0.0")))
    parser.add_argument("--shop-prior-weight-override", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_PRIOR_WEIGHT_OVERRIDE", "0.8")))
    parser.add_argument("--v3-normal-room-potion-penalty", type=float, default=float(os.environ.get("SPIRECOMM_V3_NORMAL_ROOM_POTION_PENALTY", "1.5")))
    parser.add_argument("--fast-disable-runtime-search", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-sources", default=",".join(DEFAULT_TARGET_SOURCES))
    parser.add_argument("--target-phases", default="")
    parser.add_argument("--hard-margin-max", type=float, default=0.75)
    parser.add_argument("--hard-final-floor-max", type=int, default=16)
    parser.add_argument("--hard-death-floor-window", type=int, default=6)
    parser.add_argument("--record-full-state", action="store_true")
    parser.add_argument("--save-env-blobs", action="store_true")
    parser.add_argument("--blob-sources", default="card_reward,card_reward_skip,upgrade_target,campfire,shop_value,shop_value_leave,event,map_dp,map_dp_rollout_rerank")
    parser.add_argument("--blob-score-margin-max", type=float, default=0.75, help="Use -1 to save all target-source blobs.")
    parser.add_argument("--max-env-blobs-per-seed", type=int, default=12)
    parser.add_argument("--env", action="append", default=[], help="Extra environment override NAME=VALUE.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.seeds:
        seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    else:
        seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.count)))

    env = _parse_env_pairs(args.env)
    env.update(
        {
            "SPIRECOMM_SHOP_POLICY": str(args.shop_policy),
            "SPIRECOMM_SHOP_VALUE_PRICE_COST": str(args.shop_value_price_cost),
            "SPIRECOMM_SHOP_VALUE_RESERVE_SHORTFALL_COST": str(args.shop_value_reserve_shortfall_cost),
            "SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE": str(args.shop_value_future_shop_reserve),
            "SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_HORIZON": str(args.shop_value_future_shop_horizon),
            "SPIRECOMM_SHOP_VALUE_CARD_SCALE": str(args.shop_value_card_scale),
            "SPIRECOMM_SHOP_VALUE_POTION_SCALE": str(args.shop_value_potion_scale),
            "SPIRECOMM_SHOP_VALUE_RELIC_SCALE": str(args.shop_value_relic_scale),
            "SPIRECOMM_SHOP_VALUE_THRESHOLD": str(args.shop_value_threshold),
            "SPIRECOMM_SHOP_PRIOR_WEIGHT_OVERRIDE": str(args.shop_prior_weight_override),
            "SPIRECOMM_V3_NORMAL_ROOM_POTION_PENALTY": str(max(0.0, float(args.v3_normal_room_potion_penalty))),
        }
    )
    if args.fast_disable_runtime_search:
        env["SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOPK"] = "0"
        env["SPIRECOMM_MAP_ROLLOUT_RERANK_TOPK"] = "0"

    args.combat_device = _resolve_combat_device(args.device, args.combat_device)
    target_sources = tuple(token.strip() for token in args.target_sources.split(",") if token.strip())
    target_phases = tuple(token.strip() for token in args.target_phases.split(",") if token.strip())
    blob_sources = tuple(token.strip() for token in args.blob_sources.split(",") if token.strip())
    config = {
        "repo_root": str(args.repo_root.resolve()),
        "output_dir": str(output_dir),
        "ascension": int(args.ascension),
        "max_floor": int(args.max_floor),
        "max_steps": int(args.max_steps),
        "device": str(args.device),
        "combat_device": str(args.combat_device),
        "torch_threads": int(args.torch_threads),
        "combat_selector": str(args.combat_selector),
        "combat_model": str(args.combat_model),
        "v3_combat_model": str(args.v3_combat_model),
        "card_reward_model": str(args.card_reward_model),
        "shop_choice_model": str(args.shop_choice_model),
        "target_sources": target_sources,
        "target_phases": target_phases,
        "hard_margin_max": float(args.hard_margin_max),
        "hard_final_floor_max": int(args.hard_final_floor_max),
        "hard_death_floor_window": int(args.hard_death_floor_window),
        "record_full_state": bool(args.record_full_state),
        "save_env_blobs": bool(args.save_env_blobs),
        "blob_sources": blob_sources,
        "blob_score_margin_max": float(args.blob_score_margin_max),
        "max_env_blobs_per_seed": int(args.max_env_blobs_per_seed),
        "env": env,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    existing = _load_existing_results(output_dir) if args.resume else {}
    results: list[dict[str, Any]] = [existing[seed] for seed in seeds if seed in existing]
    pending = [seed for seed in seeds if seed not in existing]
    print(
        f"collecting noncombat hard roots: total={len(seeds)} existing={len(results)} pending={len(pending)} "
        f"workers={args.workers} output={output_dir}",
        flush=True,
    )

    started = time.time()
    decision_dir = output_dir / "decisions"
    result_path = output_dir / "results.jsonl"
    errors_dir = output_dir / "errors"
    completed_new = 0
    if pending:
        with ProcessPoolExecutor(max_workers=int(args.workers), initializer=_init_worker, initargs=(config,)) as executor:
            futures = {executor.submit(_run_seed, seed): seed for seed in pending}
            for future in as_completed(futures):
                seed = futures[future]
                payload = future.result()
                result = dict(payload["result"])
                decisions = list(payload["decisions"])
                results.append(result)
                _write_jsonl(result_path, [result], append=True)
                _write_jsonl(decision_dir / f"seed_{seed:06d}.jsonl", decisions, append=False)
                if payload.get("error_traceback"):
                    errors_dir.mkdir(parents=True, exist_ok=True)
                    (errors_dir / f"seed_{seed:06d}.txt").write_text(str(payload["error_traceback"]), encoding="utf-8")
                completed_new += 1
                if completed_new == 1 or (args.summary_interval and completed_new % int(args.summary_interval) == 0):
                    current_seeds = {int(result.get("seed")) for result in results}
                    decisions_now = _read_decision_rows(decision_dir, current_seeds)
                    summary = _summarize(results, decisions_now, output_dir, started)
                    (output_dir / "summary_partial.json").write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    print(
                        f"completed_new={completed_new}/{len(pending)} total={len(results)}/{len(seeds)} "
                        f"mean_floor={summary['mean_floor']:.2f} decisions={summary['decision_count']} "
                        f"hard={summary['hard_decision_count']} elapsed={summary['seconds']:.1f}s",
                        flush=True,
                    )

    seed_set = {int(result.get("seed")) for result in results}
    all_decisions = _read_decision_rows(decision_dir, seed_set)
    hard_decisions = [row for row in all_decisions if row.get("is_hard")]
    _write_jsonl(output_dir / "hard_decisions.jsonl", hard_decisions, append=False)
    summary = _summarize(results, all_decisions, output_dir, started)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    if "fork" in mp.get_all_start_methods():
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass
    main()
