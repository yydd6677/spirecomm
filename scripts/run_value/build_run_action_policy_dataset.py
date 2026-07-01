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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from scripts.run_value import collect_run_value_trajectories as collect
from spirecomm.ai.run_value import branch_after_state, open_jsonl
from spirecomm.ai.runtime_decision import choose_model_required_action
from spirecomm.native_sim_v3 import NativeRunEnv


TERMINAL_PHASES = {"GAME_OVER", "COMPLETE", "VICTORY"}


def _phase_allowed(phase: str, allow: set[str]) -> bool:
    return "*" in allow or phase in allow


def _write_root(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _run_seed(seed: int) -> dict[str, Any]:
    assert collect._SELECTORS is not None
    started = time.time()
    config = collect._CONFIG
    output_dir = Path(config["output_dir"])
    root_dir = output_dir / "roots"
    root_dir.mkdir(parents=True, exist_ok=True)
    root_path = root_dir / f"seed_{int(seed)}.jsonl.gz"
    allow = {token.strip() for token in str(config.get("qenv_phases") or "COMBAT,CARD_REWARD").split(",") if token.strip()}
    env = NativeRunEnv(seed=int(seed), ascension_level=int(config["ascension"]), enable_neow=True)
    root_count = 0
    candidate_count = 0
    error = None
    error_traceback = None
    with open_jsonl(root_path, "wt") as handle:
        for step in range(int(config["max_steps"])):
            if str(env.phase) in TERMINAL_PHASES or int(env.floor) > int(config["max_floor"]):
                break
            phase = str(env.phase)
            before_state = env.state()
            legal_actions = list(env.legal_actions())
            try:
                baseline_action, baseline_scores, source = choose_model_required_action(
                    env,
                    collect._SELECTORS,
                    return_scores=True,
                )
                if _phase_allowed(phase, allow):
                    candidates: list[dict[str, Any]] = []
                    for index, action in enumerate(legal_actions):
                        after_state, branch_error = branch_after_state(env, action)
                        if after_state is None:
                            candidates.append(
                                {
                                    "index": int(index),
                                    "action": dict(action),
                                    "branch_error": branch_error,
                                }
                            )
                            continue
                        try:
                            value_score = collect._value_for_state(after_state)
                        except Exception as exc:
                            candidates.append(
                                {
                                    "index": int(index),
                                    "action": dict(action),
                                    "after_state": after_state,
                                    "branch_error": None,
                                    "value_error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            continue
                        candidates.append(
                            {
                                "index": int(index),
                                "action": dict(action),
                                "after_state": after_state,
                                "branch_error": None,
                                "q_env": float(value_score),
                            }
                        )
                    valid = [candidate for candidate in candidates if "q_env" in candidate]
                    if len(valid) >= 2:
                        _write_root(
                            handle,
                            {
                                "schema": "run_action_qenv_root_v1",
                                "seed": int(seed),
                                "step": int(step),
                                "phase": phase,
                                "floor": int(before_state.get("floor") or 0),
                                "room_type": str(before_state.get("room_type") or ""),
                                "source": str(source),
                                "state_before": before_state,
                                "baseline_action": dict(baseline_action),
                                "baseline_scores": [float(value) for value in baseline_scores],
                                "candidates": candidates,
                            },
                        )
                        root_count += 1
                        candidate_count += len(valid)
                action, _rerank_info = collect._choose_with_run_value(
                    env,
                    baseline_action,
                    baseline_scores,
                    mode=str(config.get("rollout_mode") or "baseline"),
                )
                env.step(action)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                error_traceback = traceback.format_exc()
                break
    return {
        "seed": int(seed),
        "root_path": str(root_path),
        "floor": int(env.floor),
        "phase": str(env.phase),
        "won": str(env.phase) in {"COMPLETE", "VICTORY"},
        "dead": str(env.phase) == "GAME_OVER",
        "roots": int(root_count),
        "candidates": int(candidate_count),
        "error": error,
        "error_traceback": error_traceback,
        "seconds": time.time() - started,
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize(rows: list[dict[str, Any]], started: float) -> dict[str, Any]:
    floors = [int(row.get("floor") or 0) for row in rows]
    return {
        "count": len(rows),
        "mean_floor": mean(floors) if floors else 0.0,
        "win_count": sum(1 for row in rows if row.get("won")),
        "death_count": sum(1 for row in rows if row.get("dead")),
        "error_count": sum(1 for row in rows if row.get("error")),
        "root_count": sum(int(row.get("roots") or 0) for row in rows),
        "candidate_count": sum(int(row.get("candidates") or 0) for row in rows),
        "elapsed_seconds": time.time() - started,
        "mean_seconds": mean([float(row.get("seconds") or 0.0) for row in rows]) if rows else 0.0,
    }


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(token.strip()) for token in str(args.seeds).split(",") if token.strip()]
    return list(range(int(args.seed_start), int(args.seed_start) + int(args.count)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build q_env roots for unified run action policy training.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--value-model", type=Path, required=True)
    parser.add_argument("--value-device", default="cpu")
    parser.add_argument("--rollout-mode", choices=["baseline", "shadow", "rerank", "policy"], default="baseline")
    parser.add_argument("--qenv-phases", default="COMBAT,CARD_REWARD")
    parser.add_argument("--rerank-phases", default="COMBAT,CARD_REWARD")
    parser.add_argument("--rerank-alpha", type=float, default=0.25)
    parser.add_argument("--rerank-min-margin", type=float, default=0.0)
    parser.add_argument("--action-policy-model", type=Path, default=None)
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-interval", type=int, default=25)
    args = parser.parse_args()
    if args.combat_device == "auto":
        from spirecomm.ai.torch_compat import torch

        args.combat_device = "cuda" if torch is not None and torch.cuda.is_available() else str(args.device)
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
        "value_model": str(args.value_model),
        "action_policy_model": "" if args.action_policy_model is None else str(args.action_policy_model),
        "mode": str(args.rollout_mode),
        "record_candidate_scores": False,
        "record_legal_actions": False,
        "record_baseline_scores": False,
        "record_applied_state_after": False,
        "seeds": seeds,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    results_path = output_dir / "results.jsonl"
    existing: dict[int, dict[str, Any]] = {}
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[int(row["seed"])] = row
    elif results_path.exists():
        results_path.unlink()
    pending = [seed for seed in seeds if seed not in existing]
    results = [existing[seed] for seed in seeds if seed in existing]
    started = time.time()
    if pending:
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(1, int(args.workers)),
            initializer=collect._init_worker,
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
                        f"completed {completed}/{len(seeds)} roots={summary['root_count']} "
                        f"candidates={summary['candidate_count']} mean_floor={summary['mean_floor']:.2f} errors={summary['error_count']}",
                        flush=True,
                    )
    summary = _summarize(results, started)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
