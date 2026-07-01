#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Any

from spirecomm.ai.runtime_decision import build_runtime_selectors, choose_model_required_action
from spirecomm.ai.v3_combat_features import clone_env_blob
from spirecomm.native_sim_v3 import NativeRunEnv


TERMINAL_PHASES = {"GAME_OVER", "COMPLETE", "VICTORY"}
_CONFIG: dict[str, Any] = {}
_SELECTORS: dict[str, Any] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _thread_env(thread_count: int) -> dict[str, str]:
    value = str(max(1, int(thread_count)))
    return {
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "OMP_DYNAMIC": "FALSE",
    }


def _base_config(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    return {
        "repo_root": str(root.resolve()),
        "ascension": int(args.ascension),
        "max_steps": int(args.max_steps),
        "torch_threads": int(args.torch_threads),
        "device": "cpu",
        "combat_device": "cpu",
        "combat_selector": "v3-teacher",
        "combat_model": str(root / "models" / "combat.pt"),
        "v3_combat_model": str(root / "models" / "v3_combat_scorer.pt"),
        "card_reward_model": str(root / "models" / "card_reward.pt"),
        "shop_choice_model": str(root / os.environ.get("SPIRECOMM_SHOP_CHOICE_MODEL_PATH", "models/shop_choice_prior_delta.pt")),
    }


def _init_worker(config: dict[str, Any]) -> None:
    global _CONFIG, _SELECTORS
    _CONFIG = dict(config)
    os.environ.update(_thread_env(int(config.get("torch_threads") or 1)))
    try:
        from spirecomm.ai.torch_compat import torch

        if torch is not None:
            torch.set_num_threads(int(config.get("torch_threads") or 1))
            torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        from evaluate_v3_rollout_batch import _prewarm_native_content_caches

        _prewarm_native_content_caches()
    except Exception:
        pass
    if bool(_CONFIG.get("selectors_preloaded")) and _SELECTORS is not None:
        return
    _SELECTORS = build_runtime_selectors(
        repo_root=Path(config["repo_root"]),
        device=str(config["device"]),
        combat_device=str(config["combat_device"]),
        combat_selector=str(config["combat_selector"]),
        combat_model=Path(config["combat_model"]),
        v3_combat_model=Path(config["v3_combat_model"]),
        card_reward_model=Path(config["card_reward_model"]),
        shop_model=Path(config["shop_choice_model"]),
    )


def _terminal_result(env: NativeRunEnv, *, seed: int, steps: int, seconds: float) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "ascension": int(_CONFIG["ascension"]),
        "phase": str(env.phase),
        "floor": int(env.floor),
        "hp": int(env.player.current_hp),
        "max_hp": int(env.player.max_hp),
        "gold": int(env.gold),
        "deck_size": len(env.deck),
        "relic_count": len(env.relics),
        "potion_count": len(env.potions),
        "steps": int(steps),
        "won": str(env.phase) in {"COMPLETE", "VICTORY"},
        "dead": str(env.phase) == "GAME_OVER",
        "timed_out": False,
        "max_floor_stopped": False,
        "error": None,
        "trace_path": None,
        "seconds": float(seconds),
    }


def _build_seed(seed: int) -> dict[str, Any]:
    assert _SELECTORS is not None
    started = time.time()
    env = NativeRunEnv(seed=int(seed), ascension_level=int(_CONFIG["ascension"]), enable_neow=True)
    sources: Counter[str] = Counter()
    steps = 0
    error = None
    max_steps = int(_CONFIG["max_steps"])
    try:
        while str(env.phase) != "COMBAT" and str(env.phase) not in TERMINAL_PHASES and steps < max_steps:
            action, _scores, source = choose_model_required_action(env, _SELECTORS, return_scores=False)
            sources[str(source)] += 1
            env.step(action)
            steps += 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    snapshot: dict[str, Any] = {
        "seed": int(seed),
        "steps": int(steps),
        "phase": str(env.phase),
        "floor": int(env.floor),
        "prefix_metrics": {"sources": dict(sources)},
        "seconds": time.time() - started,
        "error": error,
    }
    if error is None and str(env.phase) in TERMINAL_PHASES:
        snapshot["terminal_result"] = _terminal_result(env, seed=seed, steps=steps, seconds=time.time() - started)
    elif error is None and str(env.phase) == "COMBAT":
        snapshot["env_blob"] = clone_env_blob(env, strip_debug_history=True)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v3 run snapshots at the first combat decision for exact sweep prefix reuse.")
    parser.add_argument("--output-dir", type=Path, default=Path("_cache/v3_first_combat_snapshots"))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--preload-selectors-before-fork",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = _base_config(args)
    (args.output_dir / "config.json").write_text(json.dumps(config | {"seed_start": args.seed_start, "count": args.count}, indent=2), encoding="utf-8")
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.count)))
    pending = []
    for seed in seeds:
        path = args.output_dir / f"seed_{seed}.pkl"
        if args.force and path.exists():
            path.unlink()
        if not args.resume or not path.exists():
            pending.append(seed)
    print(f"snapshots output={args.output_dir} resume={len(seeds) - len(pending)}/{len(seeds)} pending={len(pending)} workers={args.workers}", flush=True)
    started = time.time()
    completed = len(seeds) - len(pending)
    context = None
    if bool(args.preload_selectors_before_fork) and "fork" in mp.get_all_start_methods() and pending:
        preload_config = dict(config)
        preload_config["selectors_preloaded"] = False
        _init_worker(preload_config)
        config["selectors_preloaded"] = True
        context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers)), initializer=_init_worker, initargs=(config,), mp_context=context) as executor:
        futures = {executor.submit(_build_seed, seed): seed for seed in pending}
        for future in as_completed(futures):
            seed = futures[future]
            snapshot = future.result()
            snapshot_path = args.output_dir / f"seed_{seed}.pkl"
            tmp_path = args.output_dir / f".seed_{seed}.{os.getpid()}.tmp"
            with tmp_path.open("wb") as handle:
                pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(snapshot_path)
            completed += 1
            if completed == len(seeds) or completed % 25 == 0:
                elapsed = max(1e-6, time.time() - started)
                rate = max(0, completed - (len(seeds) - len(pending))) / elapsed
                remaining = len(seeds) - completed
                eta = remaining / rate if rate > 1e-9 else 0.0
                print(f"completed={completed}/{len(seeds)} new_rate={rate:.3f}/s eta={eta / 60.0:.1f}m", flush=True)


if __name__ == "__main__":
    main()
