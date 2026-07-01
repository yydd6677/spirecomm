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
import multiprocessing as mp
import os
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from typing import Any

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from spirecomm.ai.run_value import (
    branch_after_state,
    encode_action_candidate,
    load_run_action_policy_checkpoint,
    load_run_value_checkpoint,
    open_jsonl,
    predict_remaining_floor,
)
from spirecomm.ai.runtime_decision import (
    build_runtime_selectors,
    choose_model_required_action,
    validate_model_required_selectors,
)
from spirecomm.ai.torch_compat import require_torch, torch
from spirecomm.native_sim_v3 import NativeRunEnv


TERMINAL_PHASES = {"GAME_OVER", "COMPLETE", "VICTORY"}
_CONFIG: dict[str, Any] = {}
_SELECTORS: dict[str, Any] | None = None
_VALUE_MODEL: Any | None = None
_ACTION_POLICY: Any | None = None


def _apply_runtime_env(config: dict[str, Any]) -> None:
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
    ):
        if config.get(config_key) is not None:
            os.environ[env_name] = str(config[config_key])


def _init_worker(config: dict[str, Any]) -> None:
    global _CONFIG, _SELECTORS, _VALUE_MODEL, _ACTION_POLICY
    _CONFIG = dict(config)
    _apply_runtime_env(_CONFIG)
    torch_threads = int(_CONFIG.get("torch_threads") or 0)
    if torch_threads > 0:
        try:
            torch.set_num_threads(torch_threads)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    _SELECTORS = build_runtime_selectors(
        repo_root=Path(_CONFIG["repo_root"]),
        device=str(_CONFIG["device"]),
        combat_device=str(_CONFIG["combat_device"]),
        combat_selector=str(_CONFIG["combat_selector"]),
        combat_model=Path(_CONFIG["combat_model"]),
        v3_combat_model=Path(_CONFIG["v3_combat_model"]),
        card_reward_model=Path(_CONFIG["card_reward_model"]),
        shop_model=Path(_CONFIG["shop_choice_model"]),
    )
    validate_model_required_selectors(_SELECTORS)
    value_model_path = str(_CONFIG.get("value_model") or "")
    if value_model_path:
        _VALUE_MODEL, _ = load_run_value_checkpoint(value_model_path, device=str(_CONFIG["value_device"]))
        _VALUE_MODEL.eval()
    action_policy_path = str(_CONFIG.get("action_policy_model") or "")
    if action_policy_path:
        _ACTION_POLICY, _ = load_run_action_policy_checkpoint(action_policy_path, device=str(_CONFIG["value_device"]))
        _ACTION_POLICY.eval()


def _action_signature(action: dict[str, Any]) -> str:
    compact = {
        key: action.get(key)
        for key in (
            "kind",
            "name",
            "card_id",
            "relic_id",
            "potion_id",
            "item_kind",
            "item_id",
            "choice_index",
            "target_index",
            "card_index",
            "price",
            "symbol",
            "mode",
        )
        if key in action
    }
    return json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _terminal_labels(env: NativeRunEnv, *, truncated: bool = False) -> dict[str, Any]:
    final_floor = int(env.floor)
    won = str(env.phase) in {"COMPLETE", "VICTORY"}
    return {
        "final_floor": final_floor,
        "won": bool(won),
        "dead": str(env.phase) == "GAME_OVER",
        "truncated": bool(truncated),
        "act1_clear": final_floor >= 17,
        "act2_clear": final_floor >= 34,
        "act3_clear": bool(won) or final_floor >= 50,
    }


def _death_next(records: list[dict[str, Any]], index: int, horizon: int) -> bool:
    current_floor = int(records[index].get("floor") or 0)
    final = records[index].get("terminal") or {}
    if not bool(final.get("dead")):
        return False
    death_floor = int(final.get("final_floor") or current_floor)
    return death_floor <= current_floor + int(horizon)


def _value_for_state(state: dict[str, Any]) -> float:
    assert _VALUE_MODEL is not None
    require_torch()
    device = str(_CONFIG["value_device"])
    remaining = predict_remaining_floor(_VALUE_MODEL, state, device=device)
    return float(int(state.get("floor") or 0) + remaining)


def _policy_score(before_state: dict[str, Any], action: dict[str, Any], after_state: dict[str, Any]) -> float:
    assert _ACTION_POLICY is not None
    require_torch()
    device = str(_CONFIG["value_device"])
    feature_variant = str(getattr(_ACTION_POLICY, "feature_variant", "current") or "current")
    vector = encode_action_candidate(before_state, action, after_state, feature_variant=feature_variant)
    input_dim = int(getattr(_ACTION_POLICY, "input_dim", len(vector)) or len(vector))
    if len(vector) > input_dim:
        vector = vector[:input_dim]
    elif len(vector) < input_dim:
        vector = vector + [0.0] * (input_dim - len(vector))
    features = torch.tensor(
        [vector],
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        score = _ACTION_POLICY(features)
    return float(score[0].detach().cpu().item())


def _phase_allowed(phase: str, allow: set[str]) -> bool:
    if not allow:
        return False
    return phase in allow or "*" in allow


def _candidate_scores(env: NativeRunEnv, actions: list[dict[str, Any]], *, scorer: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()
    before_state = env.state()
    rows: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        after_state, error = branch_after_state(env, action)
        row: dict[str, Any] = {
            "index": int(index),
            "action_signature": _action_signature(action),
            "kind": str(action.get("kind") or ""),
            "name": str(action.get("name") or action.get("card_id") or action.get("item_id") or ""),
            "branch_error": error,
        }
        if after_state is not None:
            stats["branch_ok"] += 1
            row["after_floor"] = int(after_state.get("floor") or 0)
            if scorer in {"value", "both"} and _VALUE_MODEL is not None:
                try:
                    row["value_score"] = _value_for_state(after_state)
                    stats["value_ok"] += 1
                except Exception as exc:
                    row["value_error"] = f"{type(exc).__name__}: {exc}"
                    stats["value_error"] += 1
            if scorer in {"policy", "both"} and _ACTION_POLICY is not None:
                try:
                    row["policy_score"] = _policy_score(before_state, action, after_state)
                    stats["policy_ok"] += 1
                except Exception as exc:
                    row["policy_error"] = f"{type(exc).__name__}: {exc}"
                    stats["policy_error"] += 1
        else:
            stats["branch_error"] += 1
        rows.append(row)
    return rows, stats


def _choose_with_run_value(
    env: NativeRunEnv,
    baseline_action: dict[str, Any],
    baseline_scores: list[float],
    *,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase = str(env.phase)
    actions = list(env.legal_actions())
    allow = set(str(_CONFIG.get("rerank_phases") or "").split(","))
    allow = {token.strip() for token in allow if token.strip()}
    if mode == "baseline" or not _phase_allowed(phase, allow):
        return baseline_action, {"applied": False, "reason": "mode_or_phase"}
    scorer = "policy" if mode == "policy" else "value"
    candidate_rows, stats = _candidate_scores(env, actions, scorer=scorer)
    score_key = "policy_score" if mode == "policy" else "value_score"
    scored = [row for row in candidate_rows if score_key in row]
    if not scored:
        return baseline_action, {"applied": False, "reason": "no_scored_candidates", "candidate_scores": candidate_rows, "branch_stats": dict(stats)}
    scored.sort(key=lambda row: float(row[score_key]), reverse=True)
    top = scored[0]
    second = scored[1] if len(scored) > 1 else None
    margin = float(top[score_key]) - (float(second[score_key]) if second is not None else -1.0e9)
    baseline_sig = _action_signature(baseline_action)
    top_index = int(top["index"])
    top_action = dict(actions[top_index])
    min_margin = float(_CONFIG.get("rerank_min_margin") or 0.0)
    if mode == "shadow":
        return baseline_action, {
            "applied": False,
            "reason": "shadow",
            "baseline_signature": baseline_sig,
            "top_signature": top["action_signature"],
            "top_index": top_index,
            "top_score": float(top[score_key]),
            "margin": margin,
            "candidate_scores": candidate_rows,
            "branch_stats": dict(stats),
        }
    if top["action_signature"] == baseline_sig:
        return baseline_action, {
            "applied": False,
            "reason": "top_is_baseline",
            "top_score": float(top[score_key]),
            "margin": margin,
            "candidate_scores": candidate_rows if bool(_CONFIG.get("record_candidate_scores")) else [],
            "branch_stats": dict(stats),
        }
    if margin < min_margin:
        return baseline_action, {
            "applied": False,
            "reason": "margin_below_threshold",
            "top_score": float(top[score_key]),
            "margin": margin,
            "candidate_scores": candidate_rows if bool(_CONFIG.get("record_candidate_scores")) else [],
            "branch_stats": dict(stats),
        }
    if baseline_scores and len(baseline_scores) == len(actions) and mode in {"rerank", "policy"}:
        alpha = float(_CONFIG.get("rerank_alpha") or 0.25)
        base_values = [float(value) for value in baseline_scores]
        value_by_index = {int(row["index"]): float(row[score_key]) for row in scored}
        value_values = [value_by_index.get(index, min(value_by_index.values())) for index in range(len(actions))]
        base_center = sum(base_values) / max(1, len(base_values))
        value_center = sum(value_values) / max(1, len(value_values))
        base_scale = max(1.0e-6, max(abs(value - base_center) for value in base_values))
        value_scale = max(1.0e-6, max(abs(value - value_center) for value in value_values))
        combined = [
            (base_values[index] - base_center) / base_scale
            + alpha * ((value_values[index] - value_center) / value_scale)
            for index in range(len(actions))
        ]
        combined_index = max(range(len(combined)), key=lambda idx: combined[idx])
        if combined_index != top_index:
            top_index = combined_index
            top_action = dict(actions[top_index])
            top["action_signature"] = _action_signature(top_action)
    return top_action, {
        "applied": True,
        "reason": mode,
        "baseline_signature": baseline_sig,
        "top_signature": top["action_signature"],
        "top_index": top_index,
        "top_score": float(top.get(score_key, 0.0)),
        "margin": margin,
        "candidate_scores": candidate_rows if bool(_CONFIG.get("record_candidate_scores")) else [],
        "branch_stats": dict(stats),
    }


def _finalize_records(records: list[dict[str, Any]], terminal: dict[str, Any]) -> list[dict[str, Any]]:
    for index, record in enumerate(records):
        floor = int(record.get("floor") or 0)
        record["terminal"] = dict(terminal)
        record["remaining_floor_gain"] = int(terminal["final_floor"]) - floor
        record["death_next_3"] = _death_next(records, index, 3)
        record["death_next_6"] = _death_next(records, index, 6)
    return records


def _run_seed(seed: int) -> dict[str, Any]:
    assert _SELECTORS is not None
    started = time.time()
    output_dir = Path(_CONFIG["output_dir"])
    decision_dir = output_dir / "decisions"
    decision_dir.mkdir(parents=True, exist_ok=True)
    decision_path = decision_dir / f"seed_{int(seed)}.jsonl.gz"
    env = NativeRunEnv(seed=int(seed), ascension_level=int(_CONFIG["ascension"]), enable_neow=True)
    records: list[dict[str, Any]] = []
    sources: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    rerank_reasons: Counter[str] = Counter()
    branch_stats: Counter[str] = Counter()
    error = None
    error_traceback = None
    for step in range(int(_CONFIG["max_steps"])):
        if str(env.phase) in TERMINAL_PHASES or int(env.floor) > int(_CONFIG["max_floor"]):
            break
        before_state = env.state()
        legal_actions = list(env.legal_actions())
        pre_phase = str(env.phase)
        try:
            baseline_action, baseline_scores, source = choose_model_required_action(env, _SELECTORS, return_scores=True)
            action, rerank_info = _choose_with_run_value(
                env,
                baseline_action,
                baseline_scores,
                mode=str(_CONFIG["mode"]),
            )
            after_state, branch_error = branch_after_state(env, action)
            env.step(action)
            applied_after_state = env.state()
            if after_state is None:
                after_state = applied_after_state
            record = {
                "schema": "run_value_decision_v1",
                "seed": int(seed),
                "step": int(step),
                "phase": pre_phase,
                "floor": int(before_state.get("floor") or 0),
                "room_type": str(before_state.get("room_type") or ""),
                "source": str(source),
                "state_before": before_state,
                "legal_actions": legal_actions if bool(_CONFIG.get("record_legal_actions")) else [],
                "chosen_action": dict(action),
                "baseline_action": dict(baseline_action),
                "baseline_scores": [float(value) for value in baseline_scores] if bool(_CONFIG.get("record_baseline_scores")) else [],
                "state_after": after_state,
                "applied_state_after": applied_after_state if bool(_CONFIG.get("record_applied_state_after")) else {},
                "branch_error": branch_error,
                "rerank": rerank_info,
            }
            records.append(record)
            sources[str(source)] += 1
            phases[pre_phase] += 1
            rerank_reasons[str(rerank_info.get("reason") or "")] += 1
            branch_stats.update(rerank_info.get("branch_stats") or {})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            error_traceback = traceback.format_exc()
            break
    truncated = str(env.phase) not in TERMINAL_PHASES and int(env.floor) <= int(_CONFIG["max_floor"]) and error is None
    terminal = _terminal_labels(env, truncated=truncated)
    records = _finalize_records(records, terminal)
    with open_jsonl(decision_path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "seed": int(seed),
        "decision_path": str(decision_path),
        "floor": int(env.floor),
        "phase": str(env.phase),
        "hp": int(env.player.current_hp),
        "gold": int(env.gold),
        "won": bool(terminal["won"]),
        "dead": bool(terminal["dead"]),
        "truncated": bool(terminal["truncated"]),
        "steps": len(records),
        "error": error,
        "error_traceback": error_traceback,
        "sources": dict(sources),
        "phases": dict(phases),
        "rerank_reasons": dict(rerank_reasons),
        "branch_stats": dict(branch_stats),
        "seconds": time.time() - started,
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize(results: list[dict[str, Any]], started: float) -> dict[str, Any]:
    floors = [int(row.get("floor") or 0) for row in results]
    sources: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    rerank_reasons: Counter[str] = Counter()
    branch_stats: Counter[str] = Counter()
    for row in results:
        sources.update(row.get("sources") or {})
        phases.update(row.get("phases") or {})
        rerank_reasons.update(row.get("rerank_reasons") or {})
        branch_stats.update(row.get("branch_stats") or {})
    return {
        "count": len(results),
        "mean_floor": mean(floors) if floors else 0.0,
        "median_floor": median(floors) if floors else 0.0,
        "max_floor": max(floors) if floors else 0,
        "min_floor": min(floors) if floors else 0,
        "win_count": sum(1 for row in results if row.get("won")),
        "death_count": sum(1 for row in results if row.get("dead")),
        "truncated_count": sum(1 for row in results if row.get("truncated")),
        "error_count": sum(1 for row in results if row.get("error")),
        "mean_seconds": mean([float(row.get("seconds") or 0.0) for row in results]) if results else 0.0,
        "elapsed_seconds": time.time() - started,
        "sources": dict(sources.most_common()),
        "phases": dict(phases.most_common()),
        "rerank_reasons": dict(rerank_reasons.most_common()),
        "branch_stats": dict(branch_stats.most_common()),
    }


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(token.strip()) for token in str(args.seeds).split(",") if token.strip()]
    return list(range(int(args.seed_start), int(args.seed_start) + int(args.count)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect full-run trajectories for run-value training and rerank evaluation.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--mode", choices=["baseline", "shadow", "rerank", "policy"], default="baseline")
    parser.add_argument("--rerank-phases", default="COMBAT,CARD_REWARD")
    parser.add_argument("--rerank-alpha", type=float, default=0.25)
    parser.add_argument("--rerank-min-margin", type=float, default=0.0)
    parser.add_argument("--value-model", type=Path, default=None)
    parser.add_argument("--action-policy-model", type=Path, default=None)
    parser.add_argument("--value-device", default="cpu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default=os.environ.get("SPIRECOMM_EVAL_COMBAT_DEVICE", "auto"))
    parser.add_argument("--combat-selector", choices=["legacy-slot", "v3-candidate", "v3-teacher"], default="v3-candidate")
    parser.add_argument("--combat-model", type=Path, default=Path("models/combat.pt"))
    parser.add_argument("--v3-combat-model", type=Path, default=Path("models/cache/download8_corrected_vocab/v5_dual_semantic_legacy_gate.pt"))
    parser.add_argument("--card-reward-model", type=Path, default=Path("models/card_reward.pt"))
    parser.add_argument("--shop-choice-model", type=Path, default=Path(os.environ.get("SPIRECOMM_SHOP_CHOICE_MODEL_PATH", "models/shop_choice_prior_delta.pt")))
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
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--record-candidate-scores", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record-legal-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-baseline-scores", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record-applied-state-after", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-interval", type=int, default=25)
    args = parser.parse_args()

    if args.combat_device == "auto":
        args.combat_device = "cuda" if torch is not None and torch.cuda.is_available() else str(args.device)
    if args.mode in {"shadow", "rerank"} and args.value_model is None:
        raise SystemExit(f"--mode {args.mode} requires --value-model")
    if args.mode == "policy" and args.action_policy_model is None:
        raise SystemExit("--mode policy requires --action-policy-model")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args)
    config = {
        **vars(args),
        "repo_root": str(args.repo_root.resolve()),
        "output_dir": str(output_dir),
        "v3_combat_model": str(args.v3_combat_model),
        "combat_model": str(args.combat_model),
        "card_reward_model": str(args.card_reward_model),
        "shop_choice_model": str(args.shop_choice_model),
        "value_model": "" if args.value_model is None else str(args.value_model),
        "action_policy_model": "" if args.action_policy_model is None else str(args.action_policy_model),
        "seeds": seeds,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    results_path = output_dir / "results.jsonl"
    existing: dict[int, dict[str, Any]] = {}
    if args.resume and results_path.exists():
        original_results_text = results_path.read_text(encoding="utf-8")
        kept_lines: list[str] = []
        retry_error_count = 0
        for line in original_results_text.splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("error"):
                    retry_error_count += 1
                    continue
                existing[int(row["seed"])] = row
                kept_lines.append(json.dumps(row, ensure_ascii=False))
        if retry_error_count:
            backup_path = results_path.with_name(f"{results_path.name}.retry_errors_{int(time.time())}.bak")
            backup_path.write_text(original_results_text, encoding="utf-8")
            results_path.write_text(("\n".join(kept_lines) + "\n") if kept_lines else "", encoding="utf-8")
            print(
                f"resume: retrying {retry_error_count} error rows; backed up original results to {backup_path}",
                flush=True,
            )
    elif results_path.exists():
        results_path.unlink()
    pending = [seed for seed in seeds if seed not in existing]
    results = [existing[seed] for seed in seeds if seed in existing]
    started = time.time()
    if pending:
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(1, int(args.workers)),
            initializer=_init_worker,
            initargs=(config,),
            mp_context=mp_context,
        ) as executor:
            futures = {executor.submit(_run_seed, seed): seed for seed in pending}
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                _append_jsonl(results_path, row)
                completed = len(results)
                if completed == 1 or completed == len(seeds) or (int(args.summary_interval) > 0 and completed % int(args.summary_interval) == 0):
                    summary = _summarize(results, started)
                    (output_dir / "summary_partial.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(
                        f"completed {completed}/{len(seeds)} mean_floor={summary['mean_floor']:.2f} "
                        f"wins={summary['win_count']} errors={summary['error_count']}",
                        flush=True,
                    )
    results.sort(key=lambda item: int(item["seed"]))
    summary = _summarize(results, started)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
