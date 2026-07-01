#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from spirecomm.ai.runtime_decision import build_runtime_selectors, choose_model_required_action
from spirecomm.ai.v3_combat_dataset import (
    DATASET_SCHEMA_VERSION,
    V3CombatCandidateExample,
    V3CombatLabeledRoot,
    V3CombatRootSample,
    save_shard,
)
from spirecomm.ai.v3_combat_features import (
    FEATURE_SCHEMA_VERSION,
    action_key,
    action_keys_are_unique,
    clone_env_blob,
    encode_candidate_with_before_summary,
    encode_state_summary,
    incoming_damage,
    root_combat_actions,
    schema,
    step_branch_from_blob,
)
from spirecomm.native_sim_v3 import NativeRunEnv


_CONFIG: dict[str, Any] = {}
_SELECTORS: dict[str, Any] | None = None

TERMINAL_PHASES = {"GAME_OVER", "COMPLETE", "VICTORY"}


def _init_worker(config: dict[str, Any]) -> None:
    global _CONFIG, _SELECTORS
    _CONFIG = dict(config)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(max(1, int(_CONFIG.get("torch_threads") or 1)))
    try:
        from spirecomm.ai.torch_compat import torch

        if torch is not None:
            torch.set_num_threads(max(1, int(_CONFIG.get("torch_threads") or 1)))
            torch.set_num_interop_threads(1)
    except Exception:
        pass
    _SELECTORS = build_runtime_selectors(
        repo_root=Path(_CONFIG["repo_root"]),
        device=str(_CONFIG["device"]),
        combat_device=str(_CONFIG.get("combat_device") or _CONFIG["device"]),
        combat_selector=str(_CONFIG["combat_selector"]),
        v3_combat_model=Path(_CONFIG["v3_combat_model"]),
        card_reward_model=Path(_CONFIG["card_reward_model"]),
        shop_model=Path(_CONFIG["shop_choice_model"]),
    )


def _combat_state(state: dict[str, Any]) -> dict[str, Any]:
    combat = state.get("combat_state")
    return combat if isinstance(combat, dict) else state


def _player_hp(env: Any, state: dict[str, Any] | None = None) -> int:
    if state is None:
        try:
            state = env.state()
        except Exception:
            state = {}
    combat = _combat_state(state)
    player = combat.get("player")
    if isinstance(player, dict):
        try:
            return int(player.get("current_hp") or 0)
        except (TypeError, ValueError):
            pass
    try:
        return int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _player_block(state: dict[str, Any]) -> int:
    player = _combat_state(state).get("player")
    if not isinstance(player, dict):
        return 0
    try:
        return int(player.get("block") or 0)
    except (TypeError, ValueError):
        return 0


def _live_monster_stats(state: dict[str, Any]) -> tuple[int, int]:
    monsters = _combat_state(state).get("monsters")
    if not isinstance(monsters, list):
        return 0, 0
    live_count = 0
    live_hp = 0
    for monster in monsters:
        if not isinstance(monster, dict):
            continue
        if bool(monster.get("is_gone") or monster.get("half_dead")):
            continue
        try:
            hp = int(monster.get("current_hp") or 0)
        except (TypeError, ValueError):
            hp = 0
        if hp > 0:
            live_count += 1
            live_hp += hp
    return live_count, live_hp


def _branch_terminal_score(branch: Any, *, root_hp: int, root_live_hp: int, steps: int, max_steps_reached: bool) -> float:
    state = branch.state()
    phase = str(getattr(branch, "phase", state.get("phase", "")) or "")
    hp = _player_hp(branch, state)
    if phase == "GAME_OVER" or hp <= 0:
        return -1000.0 - float(steps)
    if phase in {"COMPLETE", "VICTORY"}:
        return 2000.0 + float(hp) * 12.0 - float(steps) * 0.2
    if phase != "COMBAT":
        # Combat ended; stop before reward/map policy adds unrelated noise.
        return 1000.0 + float(hp) * 12.0 + float(max(0, hp - root_hp)) * 2.0 - float(steps) * 0.2
    live_count, live_hp = _live_monster_stats(state)
    progress = max(0.0, float(root_live_hp - live_hp))
    try:
        inc = int(incoming_damage(state))
    except Exception:
        inc = 0
    block = _player_block(state)
    uncovered = max(0, inc - max(0, block))
    score = (
        float(hp) * 10.0
        + progress * 3.0
        - float(live_hp) * 2.0
        - float(live_count) * 8.0
        + float(min(block, inc)) * 1.5
        - float(uncovered) * 12.0
        - float(steps) * 0.4
    )
    if max_steps_reached:
        score -= 25.0
    return float(score)


def _score_branch_to_combat_end(
    root_blob: bytes,
    action: dict[str, Any],
    *,
    root_hp: int,
    root_live_hp: int,
    max_branch_steps: int,
) -> tuple[float, int, str]:
    assert _SELECTORS is not None
    steps = 1
    try:
        branch = step_branch_from_blob(root_blob, action, strip_debug_history=True)
        while steps < max_branch_steps:
            phase = str(getattr(branch, "phase", "") or "")
            if phase in TERMINAL_PHASES or phase != "COMBAT" or _player_hp(branch) <= 0:
                break
            next_action, _scores, _source = choose_model_required_action(branch, _SELECTORS, return_scores=False)
            branch.step(next_action)
            steps += 1
        phase = str(getattr(branch, "phase", "") or "")
        score = _branch_terminal_score(
            branch,
            root_hp=root_hp,
            root_live_hp=root_live_hp,
            steps=steps,
            max_steps_reached=steps >= max_branch_steps and phase == "COMBAT",
        )
        return score, steps, phase
    except Exception as exc:
        return float("-inf"), steps, f"ERROR:{type(exc).__name__}:{exc}"


def _selected_candidate_indices(actions: list[dict[str, Any]], model_scores: list[float], *, topk: int) -> list[int]:
    if topk <= 0 or topk >= len(actions):
        return list(range(len(actions)))
    finite_scores = [
        (index, float(score) if index < len(model_scores) and math.isfinite(float(model_scores[index])) else float("-inf"))
        for index, score in enumerate(model_scores[: len(actions)])
    ]
    ranked = [index for index, _score in sorted(finite_scores, key=lambda item: item[1], reverse=True)]
    selected = set(ranked[:topk])
    for index, action in enumerate(actions):
        if str(action.get("kind") or "") in {"potion", "end"}:
            selected.add(index)
    return sorted(selected)


def _make_labeled_branch_root(env: Any, *, seed: int, step_index: int, chosen_action: dict[str, Any], scores: list[float]) -> V3CombatLabeledRoot | None:
    actions = root_combat_actions(env)
    before_state = env.state()
    if len(actions) <= 1 or not action_keys_are_unique(actions, before_state):
        return None
    chosen_key = action_key(chosen_action, before_state)
    candidate_indices = _selected_candidate_indices(actions, scores, topk=int(_CONFIG["candidate_topk"]))
    candidate_actions = [actions[index] for index in candidate_indices]
    if chosen_key not in {action_key(action, before_state) for action in candidate_actions}:
        candidate_actions.append(chosen_action)
    if len(candidate_actions) <= 1 or not action_keys_are_unique(candidate_actions, before_state):
        return None

    before_summary = encode_state_summary(before_state)
    root_blob = clone_env_blob(env, strip_debug_history=True)
    root_hp = _player_hp(env, before_state)
    _live_count, root_live_hp = _live_monster_stats(before_state)
    candidates: list[V3CombatCandidateExample] = []
    branch_steps: list[int] = []
    branch_phases: list[str] = []
    for action in candidate_actions:
        branch = step_branch_from_blob(root_blob, action, strip_debug_history=True)
        visible_after = branch.state()
        features = encode_candidate_with_before_summary(before_state, before_summary, action, visible_after)
        q_value, steps, phase = _score_branch_to_combat_end(
            root_blob,
            action,
            root_hp=root_hp,
            root_live_hp=root_live_hp,
            max_branch_steps=int(_CONFIG["branch_max_steps"]),
        )
        branch_steps.append(int(steps))
        branch_phases.append(str(phase))
        key = action_key(action, before_state)
        candidates.append(
            V3CombatCandidateExample(
                action=dict(action),
                action_key=key,
                visible_after=visible_after,
                delta_features=features[-schema().delta_dim :],
                candidate_features=features,
                teacher_q=float(q_value),
                is_chosen=key == chosen_key,
            )
        )
    if not any(candidate.is_chosen for candidate in candidates):
        return None
    root = V3CombatRootSample(
        root_id=f"explore:{int(seed)}:{int(step_index)}:branch_rollout",
        source="branch_rollout",
        env_blob=root_blob,
        visible_before=before_state,
        actions=[dict(action) for action in candidate_actions],
        action_keys=[action_key(action, before_state) for action in candidate_actions],
        chosen_action_key=chosen_key,
        metadata={
            "seed": int(seed),
            "step": int(step_index),
            "branch_max_steps": int(_CONFIG["branch_max_steps"]),
            "candidate_topk": int(_CONFIG["candidate_topk"]),
            "branch_steps": branch_steps,
            "branch_phases": branch_phases,
        },
    )
    return V3CombatLabeledRoot(root=root, candidates=candidates, teacher_config={"label_source": "branch_rollout_v1"})


def _run_seed(seed: int) -> dict[str, Any]:
    assert _SELECTORS is not None
    output_dir = Path(_CONFIG["output_dir"])
    shard_path = output_dir / f"seed_{int(seed):06d}.pkl"
    if bool(_CONFIG.get("resume")) and shard_path.exists():
        try:
            with shard_path.open("rb") as handle:
                payload = pickle.load(handle)
            roots = payload.get("roots") or []
            return {"seed": seed, "roots": len(roots), "resumed": True, "path": str(shard_path), "error": None}
        except Exception:
            pass

    roots: list[V3CombatLabeledRoot] = []
    env = NativeRunEnv(seed=int(seed), ascension_level=int(_CONFIG["ascension"]), enable_neow=True)
    max_steps = int(_CONFIG["max_steps"])
    max_floor = int(_CONFIG["max_floor"])
    max_roots = int(_CONFIG["roots_per_seed"])
    error = None
    started = time.time()
    for step_index in range(max_steps):
        if str(getattr(env, "phase", "")) in TERMINAL_PHASES or int(getattr(env, "floor", 0)) > max_floor:
            break
        try:
            if str(getattr(env, "phase", "")) == "COMBAT":
                actions = root_combat_actions(env)
                if len(actions) > 1:
                    combat_selector = _SELECTORS.get("combat")
                    chosen, scores = combat_selector.choose_env(env, return_scores=True, legal_actions=actions)
                    if chosen is None:
                        raise RuntimeError(f"combat selector returned no action: {getattr(combat_selector, 'last_error', '')}")
                    if len(roots) < max_roots:
                        labeled = _make_labeled_branch_root(
                            env,
                            seed=seed,
                            step_index=step_index,
                            chosen_action=chosen,
                            scores=scores,
                        )
                        if labeled is not None:
                            roots.append(labeled)
                    env.step(chosen)
                    continue
            action, _scores, _source = choose_model_required_action(env, _SELECTORS, return_scores=False)
            env.step(action)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
    save_shard(
        shard_path,
        roots,
        metadata={
            "dataset_schema": DATASET_SCHEMA_VERSION,
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "source": "branch_rollout",
            "seed": int(seed),
            "root_count": len(roots),
            "candidate_count": sum(len(root.candidates) for root in roots),
            "ascension": int(_CONFIG["ascension"]),
            "combat_model": str(_CONFIG["v3_combat_model"]),
            "branch_max_steps": int(_CONFIG["branch_max_steps"]),
            "candidate_topk": int(_CONFIG["candidate_topk"]),
            "error": error,
        },
    )
    return {
        "seed": seed,
        "roots": len(roots),
        "candidates": sum(len(root.candidates) for root in roots),
        "floor": int(getattr(env, "floor", 0)),
        "phase": str(getattr(env, "phase", "")),
        "won": str(getattr(env, "phase", "")) in {"COMPLETE", "VICTORY"},
        "dead": str(getattr(env, "phase", "")) == "GAME_OVER",
        "seconds": time.time() - started,
        "resumed": False,
        "path": str(shard_path),
        "error": error,
    }


def _write_summary(output_dir: Path, results: list[dict[str, Any]], *, started: float, config: dict[str, Any]) -> None:
    done = [result for result in results if result.get("error") is None]
    summary = {
        "count": len(results),
        "done": len(done),
        "root_count": sum(int(result.get("roots") or 0) for result in results),
        "candidate_count": sum(int(result.get("candidates") or 0) for result in results),
        "mean_roots_per_seed": mean([int(result.get("roots") or 0) for result in results]) if results else 0.0,
        "mean_floor": mean([int(result.get("floor") or 0) for result in results if "floor" in result]) if results else 0.0,
        "wins": sum(1 for result in results if result.get("won")),
        "deaths": sum(1 for result in results if result.get("dead")),
        "errors": [result for result in results if result.get("error")],
        "seconds": time.time() - started,
        "config": config,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect v3 combat roots labeled by branch rollout to combat end.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--roots-per-seed", type=int, default=64)
    parser.add_argument("--candidate-topk", type=int, default=5, help="Top model candidates to branch-label; <=0 labels all legal actions.")
    parser.add_argument("--branch-max-steps", type=int, default=80)
    parser.add_argument("--combat-selector", default="v3-candidate")
    parser.add_argument("--v3-combat-model", type=Path, default=Path("models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt"))
    parser.add_argument("--card-reward-model", type=Path, default=Path("models/card_reward.pt"))
    parser.add_argument("--shop-choice-model", type=Path, default=Path("models/shop_choice_prior_delta.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summary-interval", type=int, default=10)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "repo_root": str(args.repo_root.resolve()),
        "output_dir": str(output_dir),
        "ascension": int(args.ascension),
        "max_floor": int(args.max_floor),
        "max_steps": int(args.max_steps),
        "roots_per_seed": int(args.roots_per_seed),
        "candidate_topk": int(args.candidate_topk),
        "branch_max_steps": int(args.branch_max_steps),
        "combat_selector": str(args.combat_selector),
        "v3_combat_model": str(args.v3_combat_model),
        "card_reward_model": str(args.card_reward_model),
        "shop_choice_model": str(args.shop_choice_model),
        "device": str(args.device),
        "combat_device": str(args.combat_device),
        "torch_threads": int(args.torch_threads),
        "resume": bool(args.resume),
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.count)))
    started = time.time()
    results: list[dict[str, Any]] = []
    results_path = output_dir / "results.jsonl"
    print(
        f"branch_rollout seed_range={seeds[0]}-{seeds[-1] if seeds else seeds[0]} "
        f"workers={args.workers} roots_per_seed={args.roots_per_seed} topk={args.candidate_topk} "
        f"branch_max_steps={args.branch_max_steps}",
        flush=True,
    )
    if int(args.workers) <= 1:
        _init_worker(config)
        for seed in seeds:
            try:
                result = _run_seed(seed)
            except Exception as exc:
                result = {"seed": seed, "roots": 0, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            results.append(result)
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            if len(results) % max(1, int(args.summary_interval)) == 0:
                _write_summary(output_dir, results, started=started, config=config)
                print(
                    f"completed {len(results)}/{len(seeds)} roots={sum(int(r.get('roots') or 0) for r in results)} "
                    f"mean_roots={mean([int(r.get('roots') or 0) for r in results]):.2f}",
                    flush=True,
                )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(args.workers), mp_context=ctx, initializer=_init_worker, initargs=(config,)) as pool:
            futures = {pool.submit(_run_seed, seed): seed for seed in seeds}
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"seed": seed, "roots": 0, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                results.append(result)
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                if len(results) % max(1, int(args.summary_interval)) == 0:
                    _write_summary(output_dir, results, started=started, config=config)
                    print(
                        f"completed {len(results)}/{len(seeds)} roots={sum(int(r.get('roots') or 0) for r in results)} "
                        f"mean_roots={mean([int(r.get('roots') or 0) for r in results]):.2f}",
                        flush=True,
                    )
    _write_summary(output_dir, results, started=started, config=config)
    print(json.dumps(json.loads((output_dir / "summary.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
