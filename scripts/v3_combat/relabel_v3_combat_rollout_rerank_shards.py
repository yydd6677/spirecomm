#!/usr/bin/env python3
from __future__ import annotations
# Allow this CLI to run directly from its workflow subdirectory.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import copy
import json
import os
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scripts.v3_combat.evaluate_v3_rollout_batch import _jsonable_action
from spirecomm.ai.runtime_decision import (
    _combat_action_index,
    _maybe_rerank_combat_with_rollout,
    build_runtime_selectors,
)
from spirecomm.ai.v3_combat_dataset import load_shard, save_shard
from spirecomm.ai.v3_combat_features import action_key


_CONFIG: dict[str, Any] = {}
_SELECTORS: dict[str, Any] | None = None


def _set_runtime_env(config: dict[str, Any]) -> None:
    for env_name, config_key in (
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOPK", "topk"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MAX_STEPS", "max_rollout_steps"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MARGIN_MAX", "margin_max"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MIN_ADVANTAGE", "min_advantage"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_ROOM_TYPES", "room_types"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_DANGER", "require_danger"),
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
    ):
        value = config.get(config_key)
        if value is not None and str(value) != "":
            os.environ[env_name] = str(value)


def _init_worker(config: dict[str, Any]) -> None:
    global _CONFIG, _SELECTORS
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


def _action_label(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return "none"
    kind = str(action.get("kind") or "other")
    if kind == "card":
        return f"card:{action.get('card_id') or action.get('name') or ''}"
    if kind == "potion":
        return f"potion:{action.get('potion_id') or action.get('name') or ''}"
    return f"{kind}:{action.get('name') or ''}"


def _top_margin(scores: list[float]) -> float:
    if len(scores) < 2:
        return float("inf")
    ordered = sorted((float(value) for value in scores), reverse=True)
    return ordered[0] - ordered[1]


def _root_matches(labeled: Any, *, allowed_rooms: set[str], floor_max: int, floor_min: int) -> bool:
    before_state = getattr(getattr(labeled, "root", None), "visible_before", None)
    if not isinstance(before_state, dict):
        return False
    if allowed_rooms:
        room_type = str(before_state.get("room_type") or "")
        if room_type not in allowed_rooms:
            return False
    try:
        floor = int(before_state.get("floor") or 0)
    except (TypeError, ValueError):
        return False
    if floor_min > 0 and floor < floor_min:
        return False
    if floor_max > 0 and floor > floor_max:
        return False
    return True


def _annotate_and_boost(
    labeled: Any,
    *,
    model_action: dict[str, Any],
    rerank_action: dict[str, Any],
    model_index: int,
    rerank_index: int,
    scores: list[float],
    source_shard: str,
    root_index: int,
    pseudo_margin: float,
) -> bool:
    before_state = labeled.root.visible_before
    chosen_key = action_key(rerank_action, before_state)
    labeled.root.chosen_action_key = chosen_key
    metadata = dict(getattr(labeled.root, "metadata", {}) or {})
    metadata["offline_rollout_rerank_delta"] = {
        "source_shard": str(source_shard),
        "root_index": int(root_index),
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
    teacher_config["offline_rollout_rerank_delta"] = metadata["offline_rollout_rerank_delta"]
    labeled.teacher_config = teacher_config
    matched = False
    chosen_candidate = None
    for candidate in labeled.candidates:
        candidate.is_chosen = tuple(candidate.action_key) == tuple(chosen_key)
        if candidate.is_chosen:
            matched = True
            chosen_candidate = candidate
    if not matched:
        return False
    if float(pseudo_margin) > 0.0 and chosen_candidate is not None:
        max_q = max((float(candidate.teacher_q) for candidate in labeled.candidates), default=0.0)
        target_q = max(max_q + float(pseudo_margin), float(chosen_candidate.teacher_q))
        delta = target_q - float(chosen_candidate.teacher_q)
        chosen_candidate.teacher_q = target_q
        components = dict(getattr(chosen_candidate, "reward_components", {}) or {})
        components["teacher_q"] = target_q
        components["offline_rollout_rerank_pseudo_boost"] = float(delta)
        chosen_candidate.reward_components = components
        ranked = sorted(range(len(labeled.candidates)), key=lambda index: labeled.candidates[index].teacher_q, reverse=True)
        for rank, index in enumerate(ranked):
            labeled.candidates[index].teacher_rank = rank
    return True


def _process_shard(task: tuple[int, str]) -> dict[str, Any]:
    assert _SELECTORS is not None
    shard_index, shard_path_raw = task
    shard_path = Path(shard_path_raw)
    output_dir = Path(_CONFIG["output_dir"])
    output_path = output_dir / f"offline_rollout_rerank_{shard_index:05d}.pt"
    if bool(_CONFIG.get("resume")) and output_path.exists():
        try:
            payload = load_shard(output_path)
            roots = payload.get("roots") or []
            return {
                "shard_index": shard_index,
                "source_shard": str(shard_path),
                "output_path": str(output_path),
                "changed_roots": len(roots),
                "matched_roots": int(payload.get("metadata", {}).get("matched_roots") or 0),
                "resume": True,
                "errors": [],
                "transition_counts": dict(payload.get("metadata", {}).get("transition_counts") or {}),
                "teacher_rank_counts": dict(payload.get("metadata", {}).get("teacher_rank_counts") or {}),
            }
        except Exception:
            pass

    started = time.time()
    combat_selector = _SELECTORS.get("combat")
    if not getattr(combat_selector, "available", False):
        raise RuntimeError(f"combat selector unavailable: {getattr(combat_selector, 'last_error', '')}")
    allowed_rooms = set(_CONFIG.get("allowed_rooms") or [])
    floor_max = int(_CONFIG.get("root_floor_max") or 0)
    floor_min = int(_CONFIG.get("root_floor_min") or 0)
    max_changed = int(_CONFIG.get("max_changed_roots_per_shard") or 0)
    pseudo_margin = float(_CONFIG.get("pseudo_margin") or 0.0)
    payload = load_shard(shard_path)
    changed_roots: list[Any] = []
    transition_counts: Counter[str] = Counter()
    teacher_rank_counts: Counter[str] = Counter()
    matched_roots = 0
    checked_roots = 0
    errors: list[str] = []
    for root_index, labeled in enumerate(payload.get("roots") or []):
        if max_changed > 0 and len(changed_roots) >= max_changed:
            break
        if not _root_matches(labeled, allowed_rooms=allowed_rooms, floor_max=floor_max, floor_min=floor_min):
            continue
        checked_roots += 1
        try:
            env = labeled.root.load_env()
            actions = list(getattr(labeled.root, "actions", []) or [])
            if len(actions) <= 1:
                continue
            if hasattr(combat_selector, "choose_env"):
                model_action, scores = combat_selector.choose_env(env, return_scores=True, legal_actions=actions)
            else:
                model_action, scores = combat_selector.choose(labeled.root.visible_before, actions)
            if model_action is None:
                continue
            scores = [float(value) for value in scores]
            model_index = _combat_action_index(actions, model_action)
            rerank_action, changed = _maybe_rerank_combat_with_rollout(env, actions, scores, model_action, _SELECTORS)
            if not changed:
                continue
            rerank_index = _combat_action_index(actions, rerank_action)
            if model_index is None or rerank_index is None:
                continue
            cloned = copy.deepcopy(labeled)
            if not _annotate_and_boost(
                cloned,
                model_action=model_action,
                rerank_action=rerank_action,
                model_index=int(model_index),
                rerank_index=int(rerank_index),
                scores=scores,
                source_shard=str(shard_path),
                root_index=int(root_index),
                pseudo_margin=pseudo_margin,
            ):
                continue
            matched_roots += 1
            chosen_key = tuple(cloned.root.chosen_action_key or ())
            chosen_candidates = [candidate for candidate in cloned.candidates if tuple(candidate.action_key) == chosen_key]
            if chosen_candidates:
                teacher_rank_counts.update([str(chosen_candidates[0].teacher_rank)])
            transition_counts.update([f"{_action_label(model_action)}->{_action_label(rerank_action)}"])
            changed_roots.append(cloned)
        except Exception as exc:
            errors.append(f"root_index={root_index}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            if len(errors) >= 3:
                break
    if changed_roots:
        save_shard(
            output_path,
            changed_roots,
            metadata={
                "schema": "v3_combat_offline_rollout_rerank_delta_roots_v1",
                "source_shard": str(shard_path),
                "checked_roots": int(checked_roots),
                "matched_roots": int(matched_roots),
                "pseudo_margin": float(pseudo_margin),
                "transition_counts": dict(transition_counts),
                "teacher_rank_counts": dict(teacher_rank_counts),
            },
        )
    return {
        "shard_index": shard_index,
        "source_shard": str(shard_path),
        "output_path": str(output_path) if changed_roots else "",
        "checked_roots": int(checked_roots),
        "changed_roots": len(changed_roots),
        "matched_roots": int(matched_roots),
        "seconds": time.time() - started,
        "errors": errors,
        "transition_counts": dict(transition_counts),
        "teacher_rank_counts": dict(teacher_rank_counts),
    }


def _read_shards(args: argparse.Namespace) -> list[Path]:
    shards = [Path(path) for path in args.shards]
    if args.shards_file is not None:
        shards.extend(
            Path(line.strip())
            for line in args.shards_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if args.shards_dir is not None:
        shards.extend(sorted(args.shards_dir.glob("*.pt")))
    shards = sorted(dict.fromkeys(shards))
    if int(args.limit_shards) > 0:
        shards = shards[: int(args.limit_shards)]
    if not shards:
        raise SystemExit("no shards")
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline relabel existing combat roots with combat rollout-rerank deltas.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--shards", nargs="*", type=Path, default=[])
    parser.add_argument("--shards-file", type=Path, default=None)
    parser.add_argument("--shards-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--max-changed-roots-per-shard", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--card-reward-model", type=Path, default=Path("models/card_reward.pt"))
    parser.add_argument("--shop-choice-model", type=Path, default=Path("models/shop_choice_prior_delta.pt"))
    parser.add_argument("--shop-policy", choices=["model", "value"], default=os.environ.get("SPIRECOMM_SHOP_POLICY", "value"))
    parser.add_argument("--v3-normal-room-potion-penalty", type=float, default=float(os.environ.get("SPIRECOMM_V3_NORMAL_ROOM_POTION_PENALTY", "1.5")))
    parser.add_argument("--root-room-types", default="MonsterRoomBoss")
    parser.add_argument("--root-floor-min", type=int, default=0)
    parser.add_argument("--root-floor-max", type=int, default=16)
    parser.add_argument("--topk", type=int, default=int(os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOPK", "5")))
    parser.add_argument("--max-rollout-steps", type=int, default=int(os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MAX_STEPS", "80")))
    parser.add_argument("--margin-max", type=float, default=float(os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MARGIN_MAX", "8.0")))
    parser.add_argument("--min-advantage", type=float, default=float(os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MIN_ADVANTAGE", "1000.0")))
    parser.add_argument("--room-types", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_ROOM_TYPES", "MonsterRoomBoss"))
    parser.add_argument("--require-danger", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_DANGER", ""))
    parser.add_argument("--top-kinds", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_KINDS", "card"))
    parser.add_argument("--top-card-types", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_CARD_TYPES", "ATTACK"))
    parser.add_argument("--top-score-max", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_SCORE_MAX", "0.0"))
    parser.add_argument("--candidate-kinds", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_CANDIDATE_KINDS", "card,potion"))
    parser.add_argument("--same-turn-only", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_SAME_TURN_ONLY", "0"))
    parser.add_argument("--light-policy", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_LIGHT_POLICY", "1"))
    parser.add_argument("--floor-max", type=int, default=int(os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_FLOOR_MAX", "16")))
    parser.add_argument("--require-chosen-not-clear", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_CHOSEN_NOT_CLEAR", "1"))
    parser.add_argument("--chosen-clear-full-policy", default=os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_CHOSEN_CLEAR_FULL_POLICY", "1"))
    parser.add_argument("--pseudo-margin", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-interval", type=int, default=10)
    args = parser.parse_args()

    shards = _read_shards(args)
    allowed_rooms = [part.strip() for part in str(args.root_room_types or "").split(",") if part.strip()]
    config = {
        "repo_root": str(args.repo_root.resolve()),
        "model": str(args.model),
        "output_dir": str(args.output_dir),
        "device": str(args.device),
        "combat_device": str(args.combat_device),
        "torch_threads": int(args.torch_threads),
        "card_reward_model": str(args.card_reward_model),
        "shop_choice_model": str(args.shop_choice_model),
        "shop_policy": str(args.shop_policy),
        "v3_normal_room_potion_penalty": max(0.0, float(args.v3_normal_room_potion_penalty)),
        "allowed_rooms": allowed_rooms,
        "root_floor_min": int(args.root_floor_min),
        "root_floor_max": int(args.root_floor_max),
        "max_changed_roots_per_shard": int(args.max_changed_roots_per_shard),
        "topk": int(args.topk),
        "max_rollout_steps": int(args.max_rollout_steps),
        "margin_max": float(args.margin_max),
        "min_advantage": float(args.min_advantage),
        "room_types": str(args.room_types or ""),
        "require_danger": str(args.require_danger or ""),
        "top_kinds": str(args.top_kinds or ""),
        "top_card_types": str(args.top_card_types or ""),
        "top_score_max": str(args.top_score_max or ""),
        "candidate_kinds": str(args.candidate_kinds or ""),
        "same_turn_only": str(args.same_turn_only or ""),
        "light_policy": str(args.light_policy or ""),
        "floor_max": int(args.floor_max),
        "require_chosen_not_clear": str(args.require_chosen_not_clear or ""),
        "chosen_clear_full_policy": str(args.chosen_clear_full_policy or ""),
        "pseudo_margin": float(args.pseudo_margin),
        "resume": bool(args.resume),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps({"config": config, "shard_count": len(shards), "shards": [str(path) for path in shards]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    started = time.time()
    results: list[dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()
    teacher_rank_counts: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers)), initializer=_init_worker, initargs=(config,)) as executor:
        futures = {executor.submit(_process_shard, (index, str(path))): path for index, path in enumerate(shards)}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            transition_counts.update(result.get("transition_counts") or {})
            teacher_rank_counts.update(result.get("teacher_rank_counts") or {})
            completed = len(results)
            if completed == 1 or completed % max(1, int(args.summary_interval)) == 0 or completed == len(shards):
                elapsed = max(1.0e-6, time.time() - started)
                changed = sum(int(item.get("changed_roots") or 0) for item in results)
                checked = sum(int(item.get("checked_roots") or 0) for item in results)
                partial = {
                    "schema": "v3_combat_offline_rollout_rerank_partial_v1",
                    "completed_shards": completed,
                    "shard_count": len(shards),
                    "checked_roots": checked,
                    "changed_roots": changed,
                    "transition_counts": dict(transition_counts.most_common(50)),
                    "teacher_rank_counts": dict(teacher_rank_counts),
                    "seconds": elapsed,
                }
                (args.output_dir / "summary_partial.json").write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
                print(
                    "[offline-rerank] "
                    f"shards={completed}/{len(shards)} checked={checked} changed={changed} "
                    f"changed/s={changed / elapsed:.3f} shards/s={completed / elapsed:.3f}",
                    flush=True,
                )
    results.sort(key=lambda item: int(item["shard_index"]))
    changed = sum(int(item.get("changed_roots") or 0) for item in results)
    checked = sum(int(item.get("checked_roots") or 0) for item in results)
    summary = {
        "schema": "v3_combat_offline_rollout_rerank_summary_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": str(args.model),
        "shard_count": len(shards),
        "checked_roots": checked,
        "changed_roots": changed,
        "output_shard_count": sum(1 for item in results if item.get("output_path")),
        "transition_counts": dict(transition_counts.most_common()),
        "teacher_rank_counts": dict(teacher_rank_counts),
        "error_count": sum(1 for item in results if item.get("errors")),
        "seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "shards.json").write_text(
        json.dumps(
            {
                "shards": [
                    {
                        "source_shard": item.get("source_shard"),
                        "path": item.get("output_path"),
                        "root_count": int(item.get("changed_roots") or 0),
                    }
                    for item in results
                    if item.get("output_path")
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"[offline-rerank] wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
