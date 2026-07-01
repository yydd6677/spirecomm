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

from scripts.v3_combat.diagnose_v3_hard_validation_v0 import (
    BucketStats,
    ROOM_TYPE_NAMES,
    _best_index,
    _bucket_key,
    _format_brief,
    _merge_stats,
    _new_buckets,
    _rank_desc,
    _root_categories,
    _top2_gap,
    _update_bucket,
)
from scripts.v3_combat.evaluate_v3_rollout_batch import TERMINAL_PHASES, _jsonable_action
from spirecomm.ai.runtime_decision import build_runtime_selectors, choose_model_required_action
from spirecomm.ai.v3_combat_dataset import save_shard
from spirecomm.ai.v3_combat_selector import V3CandidateCombatSelector
from spirecomm.ai.v3_combat_teacher import label_env, teacher_config_from_env
from spirecomm.native_sim_v3 import NativeRunEnv


_CONFIG: dict[str, Any] = {}
_SELECTORS: dict[str, Any] | None = None
_BASELINE_COMBAT: V3CandidateCombatSelector | None = None
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
    global _CONFIG, _SELECTORS, _BASELINE_COMBAT, _TEACHER_CONFIG
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
    baseline_model = str(_CONFIG.get("baseline_model") or "")
    _BASELINE_COMBAT = (
        V3CandidateCombatSelector(Path(baseline_model), device=str(_CONFIG["combat_device"]))
        if baseline_model
        else None
    )
    _TEACHER_CONFIG = teacher_config_from_env()


def _action_kind(action: dict[str, Any]) -> str:
    kind = str(action.get("kind") or "")
    if kind == "potion":
        return "potion"
    if kind == "card":
        return "card"
    if kind == "end":
        return "end"
    return kind or "other"


def _room_name_from_state(state: dict[str, Any]) -> str:
    room_type = str(state.get("room_type") or "")
    if room_type:
        return room_type
    room_id = state.get("room_type_id")
    if isinstance(room_id, int):
        return ROOM_TYPE_NAMES.get(room_id, f"room_{room_id}")
    return "Unknown"


def _record_onpolicy_root(
    *,
    labeled: Any,
    pred_values: list[float],
    baseline_values: list[float] | None,
    root_id: str,
    seed: int,
    step: int,
    floor: int,
    room_type: str,
    high_conf_margin: float,
    gap_thresholds: list[float],
) -> tuple[dict[str, BucketStats], dict[str, Any], list[str]]:
    candidates = list(labeled.candidates)
    teacher_values = [float(candidate.teacher_q) for candidate in candidates]
    actions = [dict(candidate.action) for candidate in candidates]
    local_kinds = [_action_kind(action) for action in actions]
    local_potion_flags = [kind == "potion" for kind in local_kinds]
    if len(pred_values) != len(teacher_values):
        raise ValueError(f"pred/teacher candidate mismatch: {len(pred_values)} != {len(teacher_values)}")
    if baseline_values is not None and len(baseline_values) != len(teacher_values):
        raise ValueError(f"baseline/teacher candidate mismatch: {len(baseline_values)} != {len(teacher_values)}")

    teacher_top, teacher_gap, _teacher_second = _top2_gap(teacher_values)
    pred_top, pred_margin, _pred_second = _top2_gap(pred_values)
    top1 = pred_top == teacher_top
    teacher_top_q = float(teacher_values[teacher_top])
    pred_choice_teacher_q = float(teacher_values[pred_top])
    max_abs_teacher_q = max((abs(float(value)) for value in teacher_values), default=0.0)
    max_abs_pred_q = max((abs(float(value)) for value in pred_values), default=0.0)
    regret = max(0.0, teacher_top_q - pred_choice_teacher_q)
    teacher_kind = local_kinds[teacher_top]
    pred_kind = local_kinds[pred_top]
    high_conf_disagreement = (not top1) and pred_margin >= float(high_conf_margin)

    potion_indices = [index for index, flag in enumerate(local_potion_flags) if flag]
    non_potion_indices = [index for index, flag in enumerate(local_potion_flags) if not flag]
    potion_pair = bool(potion_indices and non_potion_indices)
    potion_pair_sign_correct = False
    teacher_prefers_potion = False
    pred_prefers_potion = False
    if potion_pair:
        teacher_best_potion = _best_index(potion_indices, teacher_values)
        teacher_best_non_potion = _best_index(non_potion_indices, teacher_values)
        pred_best_potion = _best_index(potion_indices, pred_values)
        pred_best_non_potion = _best_index(non_potion_indices, pred_values)
        teacher_potion_gap = teacher_values[teacher_best_potion] - teacher_values[teacher_best_non_potion]
        pred_potion_gap = pred_values[pred_best_potion] - pred_values[pred_best_non_potion]
        teacher_prefers_potion = teacher_potion_gap > 0.0
        pred_prefers_potion = pred_potion_gap > 0.0
        potion_pair_sign_correct = (teacher_potion_gap == 0.0 and pred_potion_gap == 0.0) or (
            teacher_potion_gap > 0.0
        ) == (pred_potion_gap > 0.0)

    baseline_top1 = None
    if baseline_values is not None:
        baseline_top = max(range(len(baseline_values)), key=baseline_values.__getitem__)
        baseline_top1 = baseline_top == teacher_top

    categories = _root_categories(
        top1=top1,
        regret=regret,
        teacher_gap=teacher_gap,
        pred_margin=pred_margin,
        high_conf_disagreement=high_conf_disagreement,
        teacher_kind=teacher_kind,
        pred_kind=pred_kind,
        potion_pair=potion_pair,
        teacher_prefers_potion=teacher_prefers_potion,
        pred_prefers_potion=pred_prefers_potion,
        baseline_top1=baseline_top1,
    )
    record = {
        "candidate_count": len(candidates),
        "top1": top1,
        "regret": regret,
        "teacher_gap": teacher_gap,
        "pred_margin": pred_margin,
        "high_conf_disagreement": high_conf_disagreement,
        "teacher_top_q": teacher_top_q,
        "pred_choice_teacher_q": pred_choice_teacher_q,
        "teacher_kind": teacher_kind,
        "pred_kind": pred_kind,
        "potion_pair": potion_pair,
        "potion_pair_sign_correct": potion_pair_sign_correct,
        "teacher_prefers_potion": teacher_prefers_potion,
        "pred_prefers_potion": pred_prefers_potion,
        "baseline_top1": baseline_top1,
    }
    buckets = _new_buckets()
    root_bucket_keys = [
        "overall",
        _bucket_key("room", room_type),
        _bucket_key("teacher_top_kind", teacher_kind),
        _bucket_key("pred_top_kind", pred_kind),
        _bucket_key("candidate_count", min(len(candidates), 10)),
    ]
    if potion_pair:
        root_bucket_keys.append("potion_pair")
        root_bucket_keys.append(
            "potion_pair:teacher_prefers_potion" if teacher_prefers_potion else "potion_pair:teacher_prefers_non_potion"
        )
    if any(local_potion_flags):
        root_bucket_keys.append("has_potion_candidate")
    for threshold in gap_thresholds:
        if teacher_gap >= threshold:
            root_bucket_keys.append(_bucket_key("teacher_gap_ge", threshold))
    if high_conf_disagreement:
        root_bucket_keys.append("high_conf_disagreement")
    if baseline_top1 is not None:
        root_bucket_keys.append("baseline_correct" if baseline_top1 else "baseline_wrong")
    for key in root_bucket_keys:
        _update_bucket(buckets, key, record)

    root_record = {
        "regret": regret,
        "teacher_gap": teacher_gap,
        "pred_margin": pred_margin,
        "teacher_rank_of_pred": _rank_desc(teacher_values, pred_top),
        "pred_rank_of_teacher": _rank_desc(pred_values, teacher_top),
        "root_id": root_id,
        "source": "onpolicy",
        "seed": int(seed),
        "step": int(step),
        "floor": int(floor),
        "room_type": room_type,
        "candidate_count": len(candidates),
        "teacher_top_index": teacher_top,
        "pred_top_index": pred_top,
        "teacher_top_action": _jsonable_action(actions[teacher_top]),
        "pred_top_action": _jsonable_action(actions[pred_top]),
        "teacher_top_kind": teacher_kind,
        "pred_top_kind": pred_kind,
        "teacher_top_q": teacher_top_q,
        "pred_choice_teacher_q": pred_choice_teacher_q,
        "max_abs_teacher_q": max_abs_teacher_q,
        "max_abs_pred_q": max_abs_pred_q,
        "top1": top1,
        "high_conf_disagreement": high_conf_disagreement,
        "has_potion_candidate": any(local_potion_flags),
        "potion_pair": potion_pair,
        "teacher_prefers_potion": teacher_prefers_potion if potion_pair else None,
        "pred_prefers_potion": pred_prefers_potion if potion_pair else None,
        "baseline_top1": baseline_top1,
        "categories": categories,
    }
    return buckets, root_record, categories


def _hard_record_passes_export_filter(root_record: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    max_abs_teacher_q = float(root_record.get("max_abs_teacher_q") or 0.0)
    max_abs_teacher_q_limit = float(config.get("hard_shard_max_abs_teacher_q") or 0.0)
    if max_abs_teacher_q_limit > 0.0 and max_abs_teacher_q > max_abs_teacher_q_limit:
        return False, "max_abs_teacher_q"

    regret = float(root_record.get("regret") or 0.0)
    max_regret = float(config.get("hard_shard_max_regret") or 0.0)
    if max_regret > 0.0 and regret > max_regret:
        return False, "max_regret"

    min_regret = float(config.get("hard_shard_min_regret") or 0.0)
    if min_regret > 0.0 and regret < min_regret and not bool(root_record.get("high_conf_disagreement")):
        return False, "min_regret"

    include_categories = set(config.get("hard_shard_include_categories") or [])
    if include_categories and not include_categories.intersection(root_record.get("categories") or []):
        return False, "include_categories"

    return True, ""


def _annotate_hard_labeled_root(labeled: Any, root_record: dict[str, Any]) -> None:
    metadata = dict(getattr(labeled.root, "metadata", {}) or {})
    metadata["onpolicy_hard_validation"] = {
        "root_id": root_record.get("root_id"),
        "seed": root_record.get("seed"),
        "step": root_record.get("step"),
        "floor": root_record.get("floor"),
        "room_type": root_record.get("room_type"),
        "regret": root_record.get("regret"),
        "teacher_gap": root_record.get("teacher_gap"),
        "pred_margin": root_record.get("pred_margin"),
        "teacher_top_index": root_record.get("teacher_top_index"),
        "pred_top_index": root_record.get("pred_top_index"),
        "teacher_top_kind": root_record.get("teacher_top_kind"),
        "pred_top_kind": root_record.get("pred_top_kind"),
        "baseline_top1": root_record.get("baseline_top1"),
        "categories": list(root_record.get("categories") or []),
    }
    labeled.root.metadata = metadata
    teacher_config = dict(getattr(labeled, "teacher_config", {}) or {})
    teacher_config["onpolicy_hard_validation"] = metadata["onpolicy_hard_validation"]
    labeled.teacher_config = teacher_config


def _run_seed(seed: int) -> dict[str, Any]:
    assert _SELECTORS is not None
    combat_selector = _SELECTORS.get("combat")
    if not getattr(combat_selector, "available", False):
        raise RuntimeError(f"combat selector unavailable: {getattr(combat_selector, 'last_error', '')}")
    env = NativeRunEnv(seed=seed, ascension_level=int(_CONFIG["ascension"]), enable_neow=True)
    max_steps = int(_CONFIG["max_steps"])
    max_floor = int(_CONFIG["max_floor"])
    max_roots_per_seed = int(_CONFIG["max_roots_per_seed"])
    high_conf_margin = float(_CONFIG["high_conf_margin"])
    gap_thresholds = [float(value) for value in _CONFIG["gap_thresholds"]]
    buckets = _new_buckets()
    hard_records: list[dict[str, Any]] = []
    worst_records: list[dict[str, Any]] = []
    hard_labeled_roots: list[Any] = []
    category_counts: Counter[str] = Counter()
    transition_category_counts: Counter[str] = Counter()
    hard_shard_filter_counts: Counter[str] = Counter()
    errors: list[str] = []
    root_count = 0
    step_count = 0
    started = time.time()
    for step_index in range(max_steps):
        if env.phase in TERMINAL_PHASES or int(env.floor) > max_floor:
            break
        phase = str(env.phase)
        try:
            if phase == "COMBAT":
                legal_actions = env.legal_actions()
                if len(legal_actions) > 1 and (max_roots_per_seed <= 0 or root_count < max_roots_per_seed):
                    before_state = env.state()
                    root_id = f"onpolicy:{seed}:{step_index}:{root_count}"
                    labeled = label_env(
                        env,
                        root_id=root_id,
                        source="onpolicy",
                        config=_TEACHER_CONFIG,
                        legal_actions=legal_actions,
                        validate_action_keys=False,
                    )
                    chosen, pred_scores = combat_selector.choose_env(env, return_scores=True, legal_actions=legal_actions)
                    if chosen is None:
                        raise RuntimeError(f"combat_selector_returned_no_choice:{getattr(combat_selector, 'last_error', '')}")
                    baseline_scores = None
                    if _BASELINE_COMBAT is not None:
                        _baseline_action, baseline_scores = _BASELINE_COMBAT.choose_env(
                            env,
                            return_scores=True,
                            legal_actions=legal_actions,
                        )
                    if labeled is not None and labeled.candidates:
                        room_type = _room_name_from_state(before_state)
                        root_buckets, root_record, categories = _record_onpolicy_root(
                            labeled=labeled,
                            pred_values=[float(value) for value in pred_scores],
                            baseline_values=None if baseline_scores is None else [float(value) for value in baseline_scores],
                            root_id=root_id,
                            seed=seed,
                            step=step_index,
                            floor=int(env.floor),
                            room_type=room_type,
                            high_conf_margin=high_conf_margin,
                            gap_thresholds=gap_thresholds,
                        )
                        _merge_stats(buckets, root_buckets)
                        category_counts.update(categories)
                        transition_category_counts.update(
                            [f"{root_record['teacher_top_kind']}->{root_record['pred_top_kind']}:{category}" for category in categories]
                        )
                        if (not root_record["top1"]) or root_record["high_conf_disagreement"]:
                            hard_records.append(root_record)
                            hard_shard_dir = str(_CONFIG.get("hard_shard_dir") or "")
                            max_hard_shard_roots = int(_CONFIG.get("max_hard_shard_roots_per_seed") or 0)
                            if hard_shard_dir and (max_hard_shard_roots <= 0 or len(hard_labeled_roots) < max_hard_shard_roots):
                                passes_export, filter_reason = _hard_record_passes_export_filter(root_record, _CONFIG)
                                if passes_export:
                                    _annotate_hard_labeled_root(labeled, root_record)
                                    hard_labeled_roots.append(labeled)
                                else:
                                    hard_shard_filter_counts.update([filter_reason])
                        if float(root_record["regret"]) > 0.0 or root_record["high_conf_disagreement"]:
                            worst_records.append(root_record)
                        root_count += 1
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
    hard_shard_path = ""
    if hard_labeled_roots and str(_CONFIG.get("hard_shard_dir") or ""):
        hard_shard_dir = Path(str(_CONFIG["hard_shard_dir"]))
        hard_shard_path = str(hard_shard_dir / f"onpolicy_hard_seed_{seed:05d}.pt")
        save_shard(
            Path(hard_shard_path),
            hard_labeled_roots,
            metadata={
                "schema": "v3_combat_onpolicy_hard_roots_v1",
                "seed": int(seed),
                "source_model": str(_CONFIG.get("model") or ""),
                "baseline_model": str(_CONFIG.get("baseline_model") or ""),
                "max_abs_teacher_q": float(_CONFIG.get("hard_shard_max_abs_teacher_q") or 0.0),
                "max_regret": float(_CONFIG.get("hard_shard_max_regret") or 0.0),
                "min_regret": float(_CONFIG.get("hard_shard_min_regret") or 0.0),
                "include_categories": list(_CONFIG.get("hard_shard_include_categories") or []),
                "root_count": len(hard_labeled_roots),
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
        "root_count": int(root_count),
        "seconds": time.time() - started,
        "errors": errors[:5],
        "buckets": buckets,
        "hard_records": hard_records,
        "worst_records": worst_records,
        "hard_shard_path": hard_shard_path,
        "hard_shard_roots": len(hard_labeled_roots),
        "hard_shard_filter_counts": dict(hard_shard_filter_counts),
        "category_counts": dict(category_counts),
        "transition_category_counts": dict(transition_category_counts),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="On-policy hard validation for v3 combat candidate policies.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--model", type=Path, default=Path("models/cache/download8_corrected_vocab/v5_dual_semantic_legacy_gate.pt"))
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=Path("models/cache/v3_combat_transformer_stage5_v8_potion_pair_200k_actionset_best_epoch011.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/onpolicy_hard_validation/v5_dual_gate_seed1_60"))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--max-roots-per-seed", type=int, default=0, help="<=0 means all combat roots reached by the policy.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--card-reward-model", type=Path, default=Path("models/card_reward.pt"))
    parser.add_argument("--shop-choice-model", type=Path, default=Path("models/shop_choice_prior_delta.pt"))
    parser.add_argument("--shop-policy", choices=["model", "value"], default=os.environ.get("SPIRECOMM_SHOP_POLICY", "value"))
    parser.add_argument("--shop-value-price-cost", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_PRICE_COST", "0.044348003822393976")))
    parser.add_argument(
        "--shop-value-reserve-shortfall-cost",
        type=float,
        default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_RESERVE_SHORTFALL_COST", "0.043490245962190935")),
    )
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
    parser.add_argument("--gap-thresholds", default="0.5,1.0,2.0,5.0")
    parser.add_argument("--high-conf-margin", type=float, default=1.0)
    parser.add_argument("--max-worst-records", type=int, default=1000)
    parser.add_argument("--max-hard-records", type=int, default=50000)
    parser.add_argument(
        "--hard-shard-dir",
        type=Path,
        default=None,
        help="If set, export trainable hard roots as one labeled shard per seed.",
    )
    parser.add_argument(
        "--hard-shard-max-abs-teacher-q",
        type=float,
        default=1_000_000.0,
        help="Skip exported hard roots whose max absolute teacher_q exceeds this value; <=0 disables.",
    )
    parser.add_argument(
        "--hard-shard-max-regret",
        type=float,
        default=1_000_000.0,
        help="Skip exported hard roots whose regret exceeds this value; <=0 disables.",
    )
    parser.add_argument(
        "--hard-shard-min-regret",
        type=float,
        default=0.0,
        help="Only export hard roots with at least this regret, unless they are high-confidence disagreements.",
    )
    parser.add_argument(
        "--hard-shard-include-categories",
        default="",
        help="Comma-separated category allowlist for exported hard roots. Empty exports all hard roots passing numeric filters.",
    )
    parser.add_argument(
        "--max-hard-shard-roots-per-seed",
        type=int,
        default=0,
        help="Cap exported hard roots per seed; <=0 means no cap.",
    )
    parser.add_argument("--summary-interval", type=int, default=5)
    args = parser.parse_args()

    seeds = (
        [int(token.strip()) for token in str(args.seeds).split(",") if token.strip()]
        if str(args.seeds).strip()
        else list(range(int(args.seed_start), int(args.seed_start) + int(args.count)))
    )
    gap_thresholds = [float(value.strip()) for value in str(args.gap_thresholds).split(",") if value.strip()]
    hard_shard_include_categories = [
        value.strip() for value in str(args.hard_shard_include_categories or "").split(",") if value.strip()
    ]
    config = {
        "repo_root": str(args.repo_root.resolve()),
        "model": str(args.model),
        "baseline_model": "" if args.baseline_model is None else str(args.baseline_model),
        "output_dir": str(args.output_dir),
        "seeds": seeds,
        "ascension": int(args.ascension),
        "max_floor": int(args.max_floor),
        "max_steps": int(args.max_steps),
        "max_roots_per_seed": int(args.max_roots_per_seed),
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
        "gap_thresholds": gap_thresholds,
        "high_conf_margin": float(args.high_conf_margin),
        "hard_shard_dir": "" if args.hard_shard_dir is None else str(args.hard_shard_dir),
        "hard_shard_max_abs_teacher_q": float(args.hard_shard_max_abs_teacher_q),
        "hard_shard_max_regret": float(args.hard_shard_max_regret),
        "hard_shard_min_regret": float(args.hard_shard_min_regret),
        "hard_shard_include_categories": hard_shard_include_categories,
        "max_hard_shard_roots_per_seed": int(args.max_hard_shard_roots_per_seed),
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.hard_shard_dir is not None:
        args.hard_shard_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    started = time.time()
    buckets = _new_buckets()
    category_counts: Counter[str] = Counter()
    transition_category_counts: Counter[str] = Counter()
    seed_results: list[dict[str, Any]] = []
    hard_records: list[dict[str, Any]] = []
    worst_records: list[dict[str, Any]] = []
    hard_shards: list[dict[str, Any]] = []
    hard_shard_filter_counts: Counter[str] = Counter()
    processed_roots = 0

    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_init_worker,
        initargs=(config,),
    ) as executor:
        futures = {executor.submit(_run_seed, seed): seed for seed in seeds}
        for future in as_completed(futures):
            seed = futures[future]
            result = future.result()
            seed_result = {
                key: result.get(key)
                for key in ("seed", "phase", "floor", "hp", "max_hp", "gold", "steps", "won", "dead", "root_count", "seconds", "errors")
            }
            seed_results.append(seed_result)
            processed_roots += int(result.get("root_count") or 0)
            _merge_stats(buckets, result["buckets"])
            category_counts.update(result.get("category_counts") or {})
            transition_category_counts.update(result.get("transition_category_counts") or {})
            hard_records.extend(result.get("hard_records") or [])
            worst_records.extend(result.get("worst_records") or [])
            hard_shard_path = str(result.get("hard_shard_path") or "")
            if hard_shard_path:
                hard_shards.append(
                    {
                        "seed": int(result["seed"]),
                        "path": hard_shard_path,
                        "root_count": int(result.get("hard_shard_roots") or 0),
                    }
                )
            hard_shard_filter_counts.update(result.get("hard_shard_filter_counts") or {})
            if len(hard_records) > int(args.max_hard_records) * 2:
                hard_records.sort(key=lambda item: (str(item["room_type"]), str(item["teacher_top_kind"]), -float(item["regret"])))
                del hard_records[int(args.max_hard_records) :]
            if len(worst_records) > int(args.max_worst_records) * 4:
                worst_records.sort(key=lambda item: (float(item["regret"]), float(item["teacher_gap"])), reverse=True)
                del worst_records[int(args.max_worst_records) :]
            completed = len(seed_results)
            if completed == 1 or completed % max(1, int(args.summary_interval)) == 0 or completed == len(seeds):
                overall = buckets["overall"].finalize()
                elapsed = max(1e-6, time.time() - started)
                print(
                    "[onpolicy-hard-val] "
                    f"seeds={completed}/{len(seeds)} roots={processed_roots} roots/s={processed_roots / elapsed:.2f} "
                    f"{_format_brief(overall)}",
                    flush=True,
                )
                partial = {
                    "schema": "v3_combat_onpolicy_hard_validation_partial",
                    "completed_seeds": completed,
                    "processed_roots": processed_roots,
                    "metrics": {key: stats.finalize() for key, stats in sorted(buckets.items())},
                }
                (output_dir / "summary_partial.json").write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")

    seed_results.sort(key=lambda item: int(item["seed"]))
    hard_records.sort(key=lambda item: (str(item["room_type"]), str(item["teacher_top_kind"]), -float(item["regret"])))
    hard_records = hard_records[: int(args.max_hard_records)]
    hard_shards.sort(key=lambda item: int(item["seed"]))
    worst_records.sort(key=lambda item: (float(item["regret"]), float(item["teacher_gap"])), reverse=True)
    worst_records = worst_records[: int(args.max_worst_records)]
    finalized = {key: stats.finalize() for key, stats in sorted(buckets.items())}
    floors = [int(item["floor"]) for item in seed_results]
    wins = sum(1 for item in seed_results if item.get("won"))
    errors = [item for item in seed_results if item.get("errors")]
    summary = {
        "schema": "v3_combat_onpolicy_hard_validation_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": str(args.model),
        "baseline_model": None if args.baseline_model is None else str(args.baseline_model),
        "seed_count": len(seed_results),
        "mean_floor": sum(floors) / max(1, len(floors)),
        "win_count": wins,
        "win_rate": wins / max(1, len(seed_results)),
        "error_count": len(errors),
        "processed_roots": processed_roots,
        "exported_hard_shard_count": len(hard_shards),
        "exported_hard_root_count": sum(int(item.get("root_count") or 0) for item in hard_shards),
        "seconds": time.time() - started,
        "metrics": finalized,
    }
    hard_summary = {
        "schema": "v3_combat_onpolicy_hard_validation_hard_root_summary",
        "hard_root_count": len(hard_records),
        "exported_hard_shard_count": len(hard_shards),
        "exported_hard_root_count": sum(int(item.get("root_count") or 0) for item in hard_shards),
        "hard_shard_filter_counts": dict(hard_shard_filter_counts),
        "hard_shards": hard_shards,
        "category_counts": dict(category_counts),
        "transition_category_counts": dict(transition_category_counts),
        "top_transition_categories": dict(transition_category_counts.most_common(50)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "hard_root_summary.json").write_text(json.dumps(hard_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(output_dir / "seed_results.jsonl", seed_results)
    _write_jsonl(output_dir / "hard_roots.jsonl", hard_records)
    _write_jsonl(output_dir / "worst_roots.jsonl", worst_records)
    print(f"[onpolicy-hard-val] wrote {output_dir / 'summary.json'}", flush=True)
    print(f"[onpolicy-hard-val] wrote {output_dir / 'hard_root_summary.json'}", flush=True)
    print(f"[onpolicy-hard-val] overall {_format_brief(finalized['overall'])}", flush=True)


if __name__ == "__main__":
    main()
