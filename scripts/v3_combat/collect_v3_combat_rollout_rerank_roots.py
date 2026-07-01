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
from spirecomm.ai.runtime_decision import (
    _combat_action_index,
    _maybe_rerank_combat_with_rollout,
    build_runtime_selectors,
    choose_model_required_action,
)
from spirecomm.ai.v3_combat_dataset import save_shard
from spirecomm.ai.v3_combat_features import action_key
from spirecomm.ai.v3_combat_teacher import label_env, teacher_config_from_env
from spirecomm.native_sim_v3 import NativeRunEnv


_CONFIG: dict[str, Any] = {}
_SELECTORS: dict[str, Any] | None = None
_TEACHER_CONFIG: Any | None = None


def _set_runtime_env(config: dict[str, Any]) -> None:
    for env_name, config_key in (
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOPK", "topk"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MAX_STEPS", "max_rollout_steps"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MARGIN_MAX", "margin_max"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MIN_ADVANTAGE", "min_advantage"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_ROOM_TYPES", "room_types"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_DANGER", "require_danger"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_DANGER_HP_RATIO_MAX", "danger_hp_ratio_max"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_KINDS", "top_kinds"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_CARD_TYPES", "top_card_types"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_SCORE_MAX", "top_score_max"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_CANDIDATE_KINDS", "candidate_kinds"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_SAME_TURN_ONLY", "same_turn_only"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_LIGHT_POLICY", "light_policy"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_FLOOR_MAX", "floor_max"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_CHOSEN_NOT_CLEAR", "require_chosen_not_clear"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_CHOSEN_CLEAR_FULL_POLICY", "chosen_clear_full_policy"),
        ("SPIRECOMM_SHOP_POLICY", "shop_policy"),
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


def _action_label(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return "none"
    kind = str(action.get("kind") or "other")
    if kind == "card":
        return f"card:{action.get('card_id') or action.get('name') or ''}"
    if kind == "potion":
        return f"potion:{action.get('potion_id') or action.get('name') or ''}"
    return f"{kind}:{action.get('name') or ''}"


def _annotate_root(
    labeled: Any,
    *,
    before_state: dict[str, Any],
    model_action: dict[str, Any],
    rerank_action: dict[str, Any],
    model_index: int,
    rerank_index: int,
    scores: list[float],
    seed: int,
    step: int,
    floor: int,
    room_type: str,
    pseudo_margin: float,
) -> None:
    chosen_key = action_key(rerank_action, before_state)
    labeled.root.chosen_action_key = chosen_key
    metadata = dict(getattr(labeled.root, "metadata", {}) or {})
    metadata["rollout_rerank_delta"] = {
        "seed": int(seed),
        "step": int(step),
        "floor": int(floor),
        "room_type": str(room_type),
        "model_index": int(model_index),
        "rerank_index": int(rerank_index),
        "model_score": float(scores[model_index]) if 0 <= model_index < len(scores) else None,
        "rerank_score": float(scores[rerank_index]) if 0 <= rerank_index < len(scores) else None,
        "model_margin": _top_margin(scores),
        "model_action": _jsonable_action(model_action),
        "rerank_action": _jsonable_action(rerank_action),
    }
    labeled.root.metadata = metadata
    teacher_config = dict(getattr(labeled, "teacher_config", {}) or {})
    teacher_config["rollout_rerank_delta"] = metadata["rollout_rerank_delta"]
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
    delta = target_q - float(chosen_candidate.teacher_q)
    chosen_candidate.teacher_q = target_q
    components = dict(getattr(chosen_candidate, "reward_components", {}) or {})
    components["teacher_q"] = target_q
    components["rollout_rerank_pseudo_boost"] = float(delta)
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
    step_policy = str(_CONFIG.get("step_policy") or "rerank")
    hard_labeled_roots: list[Any] = []
    records: list[dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()
    errors: list[str] = []
    started = time.time()
    step_count = 0
    for step_index in range(max_steps):
        if env.phase in TERMINAL_PHASES or int(env.floor) > max_floor:
            break
        phase = str(env.phase)
        try:
            if phase == "COMBAT":
                legal_actions_env = getattr(combat_selector, "legal_actions_env", None)
                legal_actions = legal_actions_env(env) if callable(legal_actions_env) else env.legal_actions()
                if len(legal_actions) > 1:
                    before_state = env.state()
                    if hasattr(combat_selector, "choose_env"):
                        model_action, scores = combat_selector.choose_env(env, return_scores=True, legal_actions=legal_actions)
                    else:
                        model_action, scores = combat_selector.choose(before_state, legal_actions)
                    if model_action is None:
                        raise RuntimeError(f"combat_selector_returned_no_choice:{getattr(combat_selector, 'last_error', '')}")
                    scores = [float(value) for value in scores]
                    model_index = _combat_action_index(legal_actions, model_action)
                    rerank_action, changed = _maybe_rerank_combat_with_rollout(
                        env,
                        legal_actions,
                        scores,
                        model_action,
                        _SELECTORS,
                    )
                    if changed:
                        rerank_index = _combat_action_index(legal_actions, rerank_action)
                        if rerank_index is None:
                            raise RuntimeError("rollout rerank returned action outside legal actions")
                        room_type = str(before_state.get("room_type") or getattr(env, "current_room_type", "") or "")
                        record = {
                            "seed": int(seed),
                            "step": int(step_index),
                            "floor": int(env.floor),
                            "room_type": room_type,
                            "model_index": int(model_index if model_index is not None else -1),
                            "rerank_index": int(rerank_index),
                            "model_action": _jsonable_action(model_action),
                            "rerank_action": _jsonable_action(rerank_action),
                            "model_score": float(scores[model_index]) if model_index is not None else None,
                            "rerank_score": float(scores[rerank_index]),
                            "model_margin": _top_margin(scores),
                            "candidate_count": len(legal_actions),
                        }
                        records.append(record)
                        transition_counts.update([f"{_action_label(model_action)}->{_action_label(rerank_action)}"])
                        if max_export_roots <= 0 or len(hard_labeled_roots) < max_export_roots:
                            labeled = label_env(
                                env,
                                root_id=f"rollout_rerank:{seed}:{step_index}:{len(hard_labeled_roots)}",
                                source="rollout_rerank_delta",
                                config=_TEACHER_CONFIG,
                                legal_actions=legal_actions,
                                validate_action_keys=False,
                            )
                            if labeled is not None and getattr(labeled, "candidates", None):
                                _annotate_root(
                                    labeled,
                                    before_state=before_state,
                                    model_action=model_action,
                                    rerank_action=rerank_action,
                                    model_index=int(model_index if model_index is not None else -1),
                                    rerank_index=int(rerank_index),
                                    scores=scores,
                                    seed=seed,
                                    step=step_index,
                                    floor=int(env.floor),
                                    room_type=room_type,
                                    pseudo_margin=pseudo_margin,
                                )
                                hard_labeled_roots.append(labeled)
                    chosen = rerank_action if changed and step_policy == "rerank" else model_action
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
        shard_path = str(shard_dir / f"rollout_rerank_seed_{seed:05d}.pt")
        save_shard(
            Path(shard_path),
            hard_labeled_roots,
            metadata={
                "schema": "v3_combat_rollout_rerank_delta_roots_v1",
                "seed": int(seed),
                "source_model": str(_CONFIG.get("model") or ""),
                "root_count": len(hard_labeled_roots),
                "pseudo_margin": float(pseudo_margin),
                "step_policy": step_policy,
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
        "rerank_records": records,
        "transition_counts": dict(transition_counts),
        "shard_path": shard_path,
        "exported_roots": len(hard_labeled_roots),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect combat roots where combat rollout-rerank changes the model action.")
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
    parser.add_argument("--v3-normal-room-potion-penalty", type=float, default=float(os.environ.get("SPIRECOMM_V3_NORMAL_ROOM_POTION_PENALTY", "1.5")))
    parser.add_argument("--teacher-config-json", default=os.environ.get("SPIRECOMM_V3_TEACHER_CONFIG_JSON", ""))
    parser.add_argument("--teacher-config-path", default=os.environ.get("SPIRECOMM_V3_TEACHER_CONFIG_PATH", ""))
    parser.add_argument("--topk", type=int, default=int(os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOPK", "5")))
    parser.add_argument("--max-rollout-steps", type=int, default=int(os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MAX_STEPS", "80")))
    parser.add_argument("--margin-max", type=float, default=float(os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MARGIN_MAX", "8.0")))
    parser.add_argument("--min-advantage", type=float, default=float(os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MIN_ADVANTAGE", "1000.0")))
    parser.add_argument("--room-types", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_ROOM_TYPES", "MonsterRoomBoss"))
    parser.add_argument("--require-danger", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_DANGER", ""))
    parser.add_argument("--danger-hp-ratio-max", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_DANGER_HP_RATIO_MAX", ""))
    parser.add_argument("--top-kinds", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_KINDS", "card"))
    parser.add_argument("--top-card-types", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_CARD_TYPES", "ATTACK"))
    parser.add_argument("--top-score-max", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_SCORE_MAX", "0.0"))
    parser.add_argument("--candidate-kinds", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_CANDIDATE_KINDS", "card,potion"))
    parser.add_argument("--same-turn-only", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_SAME_TURN_ONLY", "0"))
    parser.add_argument("--light-policy", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_LIGHT_POLICY", "1"))
    parser.add_argument("--floor-max", type=int, default=int(os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_FLOOR_MAX", "16")))
    parser.add_argument("--require-chosen-not-clear", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_CHOSEN_NOT_CLEAR", "1"))
    parser.add_argument("--chosen-clear-full-policy", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_CHOSEN_CLEAR_FULL_POLICY", "1"))
    parser.add_argument("--step-policy", choices=["model", "rerank"], default="rerank")
    parser.add_argument("--pseudo-margin", type=float, default=0.0)
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
        "v3_normal_room_potion_penalty": max(0.0, float(args.v3_normal_room_potion_penalty)),
        "teacher_config_json": str(args.teacher_config_json or ""),
        "teacher_config_path": str(args.teacher_config_path or ""),
        "topk": int(args.topk),
        "max_rollout_steps": int(args.max_rollout_steps),
        "margin_max": float(args.margin_max),
        "min_advantage": float(args.min_advantage),
        "room_types": str(args.room_types or ""),
        "require_danger": str(args.require_danger or ""),
        "danger_hp_ratio_max": str(args.danger_hp_ratio_max or ""),
        "top_kinds": str(args.top_kinds or ""),
        "top_card_types": str(args.top_card_types or ""),
        "top_score_max": str(args.top_score_max or ""),
        "candidate_kinds": str(args.candidate_kinds or ""),
        "same_turn_only": str(args.same_turn_only or ""),
        "light_policy": str(args.light_policy or ""),
        "floor_max": int(args.floor_max),
        "require_chosen_not_clear": str(args.require_chosen_not_clear or ""),
        "chosen_clear_full_policy": str(args.chosen_clear_full_policy or ""),
        "step_policy": str(args.step_policy),
        "pseudo_margin": float(args.pseudo_margin),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_shard_dir is not None:
        args.output_shard_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    started = time.time()
    seed_results: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
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
            records.extend(result.get("rerank_records") or [])
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
                    "schema": "v3_combat_rollout_rerank_delta_partial_v1",
                    "completed_seeds": completed,
                    "seed_count": len(seeds),
                    "mean_floor": sum(floors) / max(1, len(floors)),
                    "rerank_delta_count": len(records),
                    "transition_counts": dict(transition_counts.most_common(50)),
                    "exported_root_count": sum(item["root_count"] for item in shard_records),
                    "seconds": elapsed,
                }
                (args.output_dir / "summary_partial.json").write_text(
                    json.dumps(partial, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    "[rollout-rerank-delta] "
                    f"seeds={completed}/{len(seeds)} mean_floor={sum(floors) / max(1, len(floors)):.2f} "
                    f"deltas={len(records)} exported={sum(item['root_count'] for item in shard_records)} "
                    f"seeds/s={completed / elapsed:.3f}",
                    flush=True,
                )
    seed_results.sort(key=lambda item: int(item["seed"]))
    shard_records.sort(key=lambda item: int(item["seed"]))
    floors = [int(item["floor"]) for item in seed_results]
    wins = sum(1 for item in seed_results if item.get("won"))
    errors = [item for item in seed_results if item.get("errors")]
    summary = {
        "schema": "v3_combat_rollout_rerank_delta_summary_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": str(args.model),
        "seed_count": len(seed_results),
        "mean_floor": sum(floors) / max(1, len(floors)),
        "win_count": wins,
        "win_rate": wins / max(1, len(seed_results)),
        "error_count": len(errors),
        "rerank_delta_count": len(records),
        "transition_counts": dict(transition_counts.most_common()),
        "exported_shard_count": len(shard_records),
        "exported_root_count": sum(int(item.get("root_count") or 0) for item in shard_records),
        "seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "shards.json").write_text(json.dumps({"shards": shard_records}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(args.output_dir / "seed_results.jsonl", seed_results)
    _write_jsonl(args.output_dir / "rollout_rerank_deltas.jsonl", records)
    print(f"[rollout-rerank-delta] wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
