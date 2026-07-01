#!/usr/bin/env python3
from __future__ import annotations
# Allow this CLI to run directly from its workflow subdirectory.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import os
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scripts.v3_combat.evaluate_v3_rollout_batch import TERMINAL_PHASES, _jsonable_action
from spirecomm.ai.runtime_decision import build_runtime_selectors, choose_model_required_action
from spirecomm.ai.v3_combat_dataset import save_shard
from spirecomm.ai.v3_combat_features import action_key
from spirecomm.ai.v3_combat_teacher import label_env, teacher_config_from_env
from spirecomm.native_sim_v3 import NativeRunEnv


_CONFIG: dict[str, Any] = {}
_SELECTORS: dict[str, Any] | None = None
_TEACHER_CONFIG: Any | None = None


def _set_runtime_env(config: dict[str, Any]) -> None:
    for env_name, config_key in (
        ("SPIRECOMM_SHOP_POLICY", "shop_policy"),
        ("SPIRECOMM_SHOP_VALUE_PRICE_COST", "shop_value_price_cost"),
        ("SPIRECOMM_SHOP_VALUE_RESERVE_SHORTFALL_COST", "shop_value_reserve_shortfall_cost"),
        ("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE", "shop_value_future_shop_reserve"),
        ("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_HORIZON", "shop_value_future_shop_horizon"),
        ("SPIRECOMM_SHOP_VALUE_CARD_SCALE", "shop_value_card_scale"),
        ("SPIRECOMM_SHOP_VALUE_CARD_REFERENCE_PRICE", "shop_value_card_reference_price"),
        ("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MIN", "shop_value_card_price_factor_min"),
        ("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MAX", "shop_value_card_price_factor_max"),
        ("SPIRECOMM_SHOP_VALUE_POTION_SCALE", "shop_value_potion_scale"),
        ("SPIRECOMM_SHOP_VALUE_RELIC_SCALE", "shop_value_relic_scale"),
        ("SPIRECOMM_SHOP_VALUE_ITEM_SCALE", "shop_value_item_scale"),
        ("SPIRECOMM_SHOP_VALUE_THRESHOLD", "shop_value_threshold"),
        ("SPIRECOMM_SHOP_PRIOR_WEIGHT_OVERRIDE", "shop_prior_weight_override"),
        ("SPIRECOMM_V3_NORMAL_ROOM_POTION_PENALTY", "v3_normal_room_potion_penalty"),
        ("SPIRECOMM_V3_TEACHER_CONFIG_JSON", "teacher_config_json"),
        ("SPIRECOMM_V3_TEACHER_CONFIG_PATH", "teacher_config_path"),
    ):
        value = config.get(config_key)
        if value is not None and str(value) != "":
            os.environ[env_name] = str(value)


def _init_worker(config: dict[str, Any]) -> None:
    global _CONFIG, _SELECTORS, _TEACHER_CONFIG
    _CONFIG = dict(config)
    _set_runtime_env(_CONFIG)
    torch_threads = int(_CONFIG.get("torch_threads") or 0)
    if torch_threads > 0:
        try:
            from spirecomm.ai.torch_compat import torch

            if torch is not None:
                torch.set_num_threads(torch_threads)
                torch.set_num_interop_threads(1)
        except Exception:
            pass
    _SELECTORS = build_runtime_selectors(
        repo_root=Path(_CONFIG["repo_root"]),
        device=str(_CONFIG["device"]),
        combat_device=str(_CONFIG["combat_device"]),
        combat_selector="v3-candidate",
        v3_combat_model=Path(_CONFIG["model"]),
        card_reward_model=Path(_CONFIG["card_reward_model"]),
        shop_model=Path(_CONFIG["shop_choice_model"]),
    )
    _TEACHER_CONFIG = teacher_config_from_env()


def _top_margin(scores: list[float]) -> float:
    if len(scores) < 2:
        return float("inf")
    ordered = sorted((float(value) for value in scores), reverse=True)
    return ordered[0] - ordered[1]


def _action_kind(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return "none"
    return str(action.get("kind") or "other")


def _passes_guard_filter(guard_names: list[str], config: dict[str, Any]) -> bool:
    include = set(config.get("include_guards") or [])
    exclude = set(config.get("exclude_guards") or [])
    guards = set(guard_names)
    if include and not guards.intersection(include):
        return False
    if exclude and guards.intersection(exclude):
        return False
    return True


def _annotate_and_optionally_boost(
    labeled: Any,
    *,
    before_state: dict[str, Any],
    chosen: dict[str, Any],
    pre_top_index: int,
    final_top_index: int,
    pre_scores: list[float],
    final_scores: list[float],
    guard_names: list[str],
    seed: int,
    step: int,
    floor: int,
    room_type: str,
    pseudo_margin: float,
) -> None:
    chosen_key = action_key(chosen, before_state)
    labeled.root.chosen_action_key = chosen_key
    metadata = dict(getattr(labeled.root, "metadata", {}) or {})
    metadata["guard_delta"] = {
        "seed": int(seed),
        "step": int(step),
        "floor": int(floor),
        "room_type": room_type,
        "guard_names": list(guard_names),
        "pre_top_index": int(pre_top_index),
        "final_top_index": int(final_top_index),
        "pre_top_score": float(pre_scores[pre_top_index]) if 0 <= pre_top_index < len(pre_scores) else None,
        "final_top_score": float(final_scores[final_top_index]) if 0 <= final_top_index < len(final_scores) else None,
        "pre_margin": _top_margin(pre_scores),
        "final_margin": _top_margin(final_scores),
        "pre_top_action": _jsonable_action(labeled.root.actions[pre_top_index])
        if 0 <= pre_top_index < len(labeled.root.actions)
        else None,
        "final_top_action": _jsonable_action(chosen),
    }
    labeled.root.metadata = metadata
    teacher_config = dict(getattr(labeled, "teacher_config", {}) or {})
    teacher_config["guard_delta"] = metadata["guard_delta"]
    labeled.teacher_config = teacher_config
    for candidate in labeled.candidates:
        candidate.is_chosen = tuple(candidate.action_key) == tuple(chosen_key)
    if float(pseudo_margin) <= 0.0:
        return
    chosen_candidates = [candidate for candidate in labeled.candidates if tuple(candidate.action_key) == tuple(chosen_key)]
    if not chosen_candidates:
        return
    chosen_candidate = chosen_candidates[0]
    max_q = max((float(candidate.teacher_q) for candidate in labeled.candidates), default=0.0)
    target_q = max(max_q + float(pseudo_margin), float(chosen_candidate.teacher_q))
    delta = float(target_q - float(chosen_candidate.teacher_q))
    chosen_candidate.teacher_q = target_q
    components = dict(getattr(chosen_candidate, "reward_components", {}) or {})
    components["teacher_q"] = target_q
    components["guard_distill_pseudo_boost"] = delta
    chosen_candidate.reward_components = components
    ranked = sorted(range(len(labeled.candidates)), key=lambda index: labeled.candidates[index].teacher_q, reverse=True)
    for rank, index in enumerate(ranked):
        labeled.candidates[index].teacher_rank = rank


def _run_seed(seed: int) -> dict[str, Any]:
    assert _SELECTORS is not None
    combat_selector = _SELECTORS.get("combat")
    if not getattr(combat_selector, "available", False):
        raise RuntimeError(f"combat selector unavailable: {getattr(combat_selector, 'last_error', '')}")
    env = NativeRunEnv(seed=seed, ascension_level=int(_CONFIG["ascension"]), enable_neow=True)
    max_steps = int(_CONFIG["max_steps"])
    max_floor = int(_CONFIG["max_floor"])
    max_export_roots = int(_CONFIG.get("max_export_roots_per_seed") or 0)
    pseudo_margin = float(_CONFIG.get("pseudo_margin") or 0.0)
    hard_labeled_roots: list[Any] = []
    guard_records: list[dict[str, Any]] = []
    guard_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    errors: list[str] = []
    step_count = 0
    started = time.time()
    for step_index in range(max_steps):
        if env.phase in TERMINAL_PHASES or int(env.floor) > max_floor:
            break
        phase = str(env.phase)
        try:
            if phase == "COMBAT":
                legal_actions = env.legal_actions()
                if len(legal_actions) > 1:
                    before_state = env.state()
                    chosen, scores = combat_selector.choose_env(env, return_scores=True, legal_actions=legal_actions)
                    if chosen is None:
                        raise RuntimeError(f"combat_selector_returned_no_choice:{getattr(combat_selector, 'last_error', '')}")
                    pre_scores = [float(value) for value in getattr(combat_selector, "last_pre_guard_scores", [])]
                    final_scores = [float(value) for value in getattr(combat_selector, "last_final_scores", scores)]
                    pre_top_index = getattr(combat_selector, "last_pre_guard_top_index", None)
                    final_top_index = getattr(combat_selector, "last_final_top_index", None)
                    guard_names = [str(value) for value in getattr(combat_selector, "last_guard_names", [])]
                    if (
                        guard_names
                        and pre_top_index is not None
                        and final_top_index is not None
                        and int(pre_top_index) != int(final_top_index)
                        and _passes_guard_filter(guard_names, _CONFIG)
                    ):
                        pre_action = legal_actions[int(pre_top_index)]
                        final_action = legal_actions[int(final_top_index)]
                        for guard in guard_names:
                            guard_counts.update([guard])
                        transition = f"{'+'.join(guard_names)}:{_action_kind(pre_action)}->{_action_kind(final_action)}"
                        transition_counts.update([transition])
                        room_type = str(before_state.get("room_type") or getattr(env, "current_room_type", "") or "")
                        record = {
                            "seed": int(seed),
                            "step": int(step_index),
                            "floor": int(env.floor),
                            "room_type": room_type,
                            "guard_names": guard_names,
                            "pre_top_index": int(pre_top_index),
                            "final_top_index": int(final_top_index),
                            "pre_top_action": _jsonable_action(pre_action),
                            "final_top_action": _jsonable_action(final_action),
                            "pre_margin": _top_margin(pre_scores),
                            "final_margin": _top_margin(final_scores),
                            "pre_top_score": float(pre_scores[int(pre_top_index)]) if pre_scores else None,
                            "final_top_score": float(final_scores[int(final_top_index)]) if final_scores else None,
                            "candidate_count": len(legal_actions),
                        }
                        guard_records.append(record)
                        if max_export_roots <= 0 or len(hard_labeled_roots) < max_export_roots:
                            labeled = label_env(
                                env,
                                root_id=f"guard_delta:{seed}:{step_index}:{len(hard_labeled_roots)}",
                                source="guard_delta",
                                config=_TEACHER_CONFIG,
                                legal_actions=legal_actions,
                                validate_action_keys=False,
                            )
                            if labeled is not None and getattr(labeled, "candidates", None):
                                _annotate_and_optionally_boost(
                                    labeled,
                                    before_state=before_state,
                                    chosen=final_action,
                                    pre_top_index=int(pre_top_index),
                                    final_top_index=int(final_top_index),
                                    pre_scores=pre_scores,
                                    final_scores=final_scores,
                                    guard_names=guard_names,
                                    seed=seed,
                                    step=step_index,
                                    floor=int(env.floor),
                                    room_type=room_type,
                                    pseudo_margin=pseudo_margin,
                                )
                                hard_labeled_roots.append(labeled)
                    env.step(chosen)
                else:
                    action, _scores, _source = choose_model_required_action(env, _SELECTORS, return_scores=False)
                    env.step(action)
            else:
                action, _scores, _source = choose_model_required_action(env, _SELECTORS, return_scores=False)
                env.step(action)
            step_count += 1
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            break
    shard_path = ""
    if hard_labeled_roots and str(_CONFIG.get("output_shard_dir") or ""):
        shard_dir = Path(str(_CONFIG["output_shard_dir"]))
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = str(shard_dir / f"guard_delta_seed_{seed:05d}.pt")
        save_shard(
            Path(shard_path),
            hard_labeled_roots,
            metadata={
                "schema": "v3_combat_guard_delta_roots_v1",
                "seed": int(seed),
                "source_model": str(_CONFIG.get("model") or ""),
                "root_count": len(hard_labeled_roots),
                "pseudo_margin": float(pseudo_margin),
                "include_guards": list(_CONFIG.get("include_guards") or []),
                "exclude_guards": list(_CONFIG.get("exclude_guards") or []),
            },
        )
    return {
        "seed": int(seed),
        "phase": str(env.phase),
        "floor": int(env.floor),
        "hp": int(env.player.current_hp),
        "max_hp": int(env.player.max_hp),
        "gold": int(env.gold),
        "steps": int(step_count),
        "won": str(env.phase) in {"COMPLETE", "VICTORY"},
        "dead": str(env.phase) == "GAME_OVER",
        "seconds": time.time() - started,
        "errors": errors[:5],
        "guard_records": guard_records,
        "guard_counts": dict(guard_counts),
        "transition_counts": dict(transition_counts),
        "shard_path": shard_path,
        "exported_roots": len(hard_labeled_roots),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect combat roots where runtime guards change the model top action.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--model", type=Path, default=Path("models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-shard-dir", type=Path, default=None)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--max-export-roots-per-seed", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--card-reward-model", type=Path, default=Path("models/card_reward.pt"))
    parser.add_argument("--shop-choice-model", type=Path, default=Path("models/shop_choice_prior_delta.pt"))
    parser.add_argument("--shop-policy", choices=["model", "value"], default=os.environ.get("SPIRECOMM_SHOP_POLICY", "value"))
    parser.add_argument("--shop-value-price-cost", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_PRICE_COST", "0.044348003822393976")))
    parser.add_argument("--shop-value-reserve-shortfall-cost", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_RESERVE_SHORTFALL_COST", "0.043490245962190935")))
    parser.add_argument("--shop-value-future-shop-reserve", type=int, default=int(os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE", "120")))
    parser.add_argument("--shop-value-future-shop-horizon", type=int, default=int(os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_HORIZON", "5")))
    parser.add_argument("--shop-value-card-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_SCALE", "4.6262945279949435")))
    parser.add_argument("--shop-value-card-reference-price", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_REFERENCE_PRICE", "60.0")))
    parser.add_argument("--shop-value-card-price-factor-min", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MIN", "0.65")))
    parser.add_argument("--shop-value-card-price-factor-max", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MAX", "1.35")))
    parser.add_argument("--shop-value-potion-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_POTION_SCALE", "0.5084989138155764")))
    parser.add_argument("--shop-value-relic-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_RELIC_SCALE", "0.8")))
    parser.add_argument("--shop-value-item-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_ITEM_SCALE", "1.0")))
    parser.add_argument("--shop-value-threshold", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_THRESHOLD", "0.0")))
    parser.add_argument("--shop-prior-weight-override", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_PRIOR_WEIGHT_OVERRIDE", "0.8")))
    parser.add_argument("--v3-normal-room-potion-penalty", type=float, default=float(os.environ.get("SPIRECOMM_V3_NORMAL_ROOM_POTION_PENALTY", "1.5")))
    parser.add_argument("--teacher-config-json", default=os.environ.get("SPIRECOMM_V3_TEACHER_CONFIG_JSON", ""))
    parser.add_argument("--teacher-config-path", default=os.environ.get("SPIRECOMM_V3_TEACHER_CONFIG_PATH", ""))
    parser.add_argument("--include-guards", default="", help="Comma-separated guard allowlist. Empty keeps all guard deltas.")
    parser.add_argument("--exclude-guards", default="teacher_fallback,teacher_blend,branch_advisor,rescue", help="Comma-separated guard denylist.")
    parser.add_argument("--pseudo-margin", type=float, default=0.0, help="If >0, boost guard-chosen candidate above teacher top by this margin in exported shards.")
    parser.add_argument("--summary-interval", type=int, default=5)
    args = parser.parse_args()

    seeds = (
        [int(token.strip()) for token in str(args.seeds).split(",") if token.strip()]
        if str(args.seeds).strip()
        else list(range(int(args.seed_start), int(args.seed_start) + int(args.count)))
    )
    config = {
        "repo_root": str(args.repo_root.resolve()),
        "model": str(args.model),
        "output_dir": str(args.output_dir),
        "output_shard_dir": "" if args.output_shard_dir is None else str(args.output_shard_dir),
        "seeds": seeds,
        "ascension": int(args.ascension),
        "max_floor": int(args.max_floor),
        "max_steps": int(args.max_steps),
        "max_export_roots_per_seed": int(args.max_export_roots_per_seed),
        "device": str(args.device),
        "combat_device": str(args.combat_device),
        "torch_threads": int(args.torch_threads),
        "card_reward_model": str(args.card_reward_model),
        "shop_choice_model": str(args.shop_choice_model),
        "shop_policy": str(args.shop_policy),
        "shop_value_price_cost": float(args.shop_value_price_cost),
        "shop_value_reserve_shortfall_cost": float(args.shop_value_reserve_shortfall_cost),
        "shop_value_future_shop_reserve": int(args.shop_value_future_shop_reserve),
        "shop_value_future_shop_horizon": int(args.shop_value_future_shop_horizon),
        "shop_value_card_scale": float(args.shop_value_card_scale),
        "shop_value_card_reference_price": float(args.shop_value_card_reference_price),
        "shop_value_card_price_factor_min": float(args.shop_value_card_price_factor_min),
        "shop_value_card_price_factor_max": float(args.shop_value_card_price_factor_max),
        "shop_value_potion_scale": float(args.shop_value_potion_scale),
        "shop_value_relic_scale": float(args.shop_value_relic_scale),
        "shop_value_item_scale": float(args.shop_value_item_scale),
        "shop_value_threshold": float(args.shop_value_threshold),
        "shop_prior_weight_override": float(args.shop_prior_weight_override),
        "v3_normal_room_potion_penalty": max(0.0, float(args.v3_normal_room_potion_penalty)),
        "teacher_config_json": str(args.teacher_config_json or ""),
        "teacher_config_path": str(args.teacher_config_path or ""),
        "include_guards": [value.strip() for value in str(args.include_guards or "").split(",") if value.strip()],
        "exclude_guards": [value.strip() for value in str(args.exclude_guards or "").split(",") if value.strip()],
        "pseudo_margin": float(args.pseudo_margin),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_shard_dir is not None:
        args.output_shard_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    started = time.time()
    seed_results: list[dict[str, Any]] = []
    guard_records: list[dict[str, Any]] = []
    guard_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    shard_records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers)), initializer=_init_worker, initargs=(config,)) as executor:
        futures = {executor.submit(_run_seed, seed): seed for seed in seeds}
        for future in as_completed(futures):
            result = future.result()
            seed_results.append(
                {
                    key: result.get(key)
                    for key in ("seed", "phase", "floor", "hp", "max_hp", "gold", "steps", "won", "dead", "seconds", "errors", "exported_roots")
                }
            )
            guard_records.extend(result.get("guard_records") or [])
            guard_counts.update(result.get("guard_counts") or {})
            transition_counts.update(result.get("transition_counts") or {})
            if result.get("shard_path"):
                shard_records.append(
                    {
                        "seed": int(result["seed"]),
                        "path": str(result["shard_path"]),
                        "root_count": int(result.get("exported_roots") or 0),
                    }
                )
            completed = len(seed_results)
            if completed == 1 or completed % max(1, int(args.summary_interval)) == 0 or completed == len(seeds):
                elapsed = max(1.0e-6, time.time() - started)
                floors = [int(item["floor"]) for item in seed_results]
                partial = {
                    "schema": "v3_combat_guard_delta_partial_v1",
                    "completed_seeds": completed,
                    "seed_count": len(seeds),
                    "mean_floor": sum(floors) / max(1, len(floors)),
                    "guard_delta_count": len(guard_records),
                    "guard_counts": dict(guard_counts.most_common()),
                    "transition_counts": dict(transition_counts.most_common(50)),
                    "exported_root_count": sum(item["root_count"] for item in shard_records),
                    "seconds": elapsed,
                }
                (args.output_dir / "summary_partial.json").write_text(
                    json.dumps(partial, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    "[guard-delta] "
                    f"seeds={completed}/{len(seeds)} mean_floor={sum(floors) / max(1, len(floors)):.2f} "
                    f"guard_deltas={len(guard_records)} exported={sum(item['root_count'] for item in shard_records)} "
                    f"seeds/s={completed / elapsed:.3f}",
                    flush=True,
                )
    seed_results.sort(key=lambda item: int(item["seed"]))
    shard_records.sort(key=lambda item: int(item["seed"]))
    floors = [int(item["floor"]) for item in seed_results]
    wins = sum(1 for item in seed_results if item.get("won"))
    errors = [item for item in seed_results if item.get("errors")]
    summary = {
        "schema": "v3_combat_guard_delta_summary_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": str(args.model),
        "seed_count": len(seed_results),
        "mean_floor": sum(floors) / max(1, len(floors)),
        "win_count": wins,
        "win_rate": wins / max(1, len(seed_results)),
        "error_count": len(errors),
        "guard_delta_count": len(guard_records),
        "guard_counts": dict(guard_counts.most_common()),
        "transition_counts": dict(transition_counts.most_common()),
        "exported_shard_count": len(shard_records),
        "exported_root_count": sum(int(item.get("root_count") or 0) for item in shard_records),
        "seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "shards.json").write_text(json.dumps({"shards": shard_records}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(args.output_dir / "seed_results.jsonl", seed_results)
    _write_jsonl(args.output_dir / "guard_deltas.jsonl", guard_records)
    print(f"[guard-delta] wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
