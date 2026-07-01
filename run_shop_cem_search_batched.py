#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import shutil
import time
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any

import evaluate_v3_rollout_batch as rollout_eval


PARAM_RANGES: dict[str, tuple[float, float]] = {
    "w": (0.0, 6.0),
    "card_scale": (2.0, 6.5),
    "relic_scale": (0.5, 2.2),
    "potion_scale": (0.15, 0.85),
    "price_cost": (0.015, 0.065),
    "reserve_cost": (0.015, 0.13),
    "future_shop_reserve": (0.0, 260.0),
    "future_shop_horizon": (0.0, 10.0),
    "shop_value_threshold": (0.0, 5.0),
}

INTEGER_PARAMS = {"future_shop_reserve", "future_shop_horizon"}

ANCHOR_PARAMS: list[tuple[str, dict[str, float]]] = [
    (
        "current_best",
        {
            "w": 3.9894704869881643,
            "card_scale": 3.269107435576244,
            "relic_scale": 1.2020539153538472,
            "potion_scale": 0.46952272967127556,
            "price_cost": 0.037396152383869063,
            "reserve_cost": 0.0436283294282452,
            "future_shop_reserve": 120,
            "future_shop_horizon": 5,
            "shop_value_threshold": 0.0,
        },
    ),
    (
        "current_best_strong_reserve",
        {
            "w": 3.9894704869881643,
            "card_scale": 3.269107435576244,
            "relic_scale": 1.2020539153538472,
            "potion_scale": 0.46952272967127556,
            "price_cost": 0.037396152383869063,
            "reserve_cost": 0.0700,
            "future_shop_reserve": 180,
            "future_shop_horizon": 7,
            "shop_value_threshold": 1.0,
        },
    ),
    (
        "current_second",
        {
            "w": 1.6416131593344885,
            "card_scale": 3.7429453610210013,
            "relic_scale": 0.9289922603489507,
            "potion_scale": 0.5301823247890755,
            "price_cost": 0.037383006639073554,
            "reserve_cost": 0.08218041191324241,
            "future_shop_reserve": 120,
            "future_shop_horizon": 5,
            "shop_value_threshold": 0.0,
        },
    ),
    (
        "low_relic_high_potion",
        {
            "w": 1.2044740476555527,
            "card_scale": 4.6262945279949435,
            "relic_scale": 0.8,
            "potion_scale": 0.5084989138155764,
            "price_cost": 0.044348003822393976,
            "reserve_cost": 0.043490245962190935,
            "future_shop_reserve": 120,
            "future_shop_horizon": 5,
            "shop_value_threshold": 0.0,
        },
    ),
    (
        "baseline",
        {
            "w": 1.0,
            "card_scale": 4.0,
            "relic_scale": 1.25,
            "potion_scale": 0.6,
            "price_cost": 0.045,
            "reserve_cost": 0.05,
            "future_shop_reserve": 120,
            "future_shop_horizon": 5,
            "shop_value_threshold": 0.0,
        },
    ),
    (
        "strong_reserve",
        {
            "w": 2.5,
            "card_scale": 3.8,
            "relic_scale": 1.1,
            "potion_scale": 0.45,
            "price_cost": 0.045,
            "reserve_cost": 0.10,
            "future_shop_reserve": 220,
            "future_shop_horizon": 8,
            "shop_value_threshold": 1.5,
        },
    ),
    (
        "aggressive_spend",
        {
            "w": 2.0,
            "card_scale": 4.8,
            "relic_scale": 1.0,
            "potion_scale": 0.55,
            "price_cost": 0.025,
            "reserve_cost": 0.02,
            "future_shop_reserve": 0,
            "future_shop_horizon": 0,
            "shop_value_threshold": 0.0,
        },
    ),
    (
        "high_relic_low_potion",
        {
            "w": 3.0,
            "card_scale": 3.5,
            "relic_scale": 1.8,
            "potion_scale": 0.30,
            "price_cost": 0.040,
            "reserve_cost": 0.060,
            "future_shop_reserve": 160,
            "future_shop_horizon": 6,
            "shop_value_threshold": 0.8,
        },
    ),
    (
        "conservative_buy",
        {
            "w": 4.0,
            "card_scale": 3.2,
            "relic_scale": 1.3,
            "potion_scale": 0.35,
            "price_cost": 0.050,
            "reserve_cost": 0.09,
            "future_shop_reserve": 220,
            "future_shop_horizon": 8,
            "shop_value_threshold": 3.0,
        },
    ),
]


def _clamp(name: str, value: float) -> float | int:
    lo, hi = PARAM_RANGES[name]
    clipped = max(lo, min(hi, float(value)))
    if name in INTEGER_PARAMS:
        return int(round(clipped))
    return clipped


def _normalize_params(params: dict[str, float]) -> dict[str, float | int]:
    return {name: _clamp(name, params[name]) for name in PARAM_RANGES}


def _midpoint_params() -> dict[str, float | int]:
    return {name: _clamp(name, (lo + hi) * 0.5) for name, (lo, hi) in PARAM_RANGES.items()}


def _sample_uniform(rng: random.Random) -> dict[str, float | int]:
    return {name: _clamp(name, rng.uniform(lo, hi)) for name, (lo, hi) in PARAM_RANGES.items()}


def _sample_normal(rng: random.Random, mean: dict[str, float], sigma: dict[str, float]) -> dict[str, float | int]:
    return {name: _clamp(name, rng.gauss(mean[name], sigma[name])) for name in PARAM_RANGES}


def _trial_sort_key(record: dict[str, Any]) -> tuple[float, int, int, int]:
    return (
        float(record.get("mean_floor", 0.0) or 0.0),
        int(record.get("win_count", 0) or 0),
        -int(record.get("timeout_count", 0) or 0),
        -int(record.get("error_count", 0) or 0),
    )


def _base_config(
    args: argparse.Namespace,
    output_dir: Path,
    params: dict[str, Any],
    *,
    metrics_mode: str,
) -> dict[str, Any]:
    return {
        "repo_root": str(Path("/home/yydd/spirecomm")),
        "output_dir": str(output_dir),
        "ascension": int(args.ascension),
        "max_floor": int(args.max_floor),
        "max_steps": int(args.max_steps),
        "no_progress_limit": 0,
        "trace_mode": "none",
        "metrics_mode": str(metrics_mode),
        "torch_threads": int(args.torch_threads),
        "device": str(args.device),
        "combat_device": args.combat_device,
        "combat_selector": str(args.combat_selector),
        "combat_model": str(args.combat_model),
        "v3_combat_model": str(args.v3_combat_model),
        "card_reward_model": str(args.card_reward_model),
        "shop_choice_model": str(args.shop_choice_model),
        "shop_policy": "value",
        "shop_value_price_cost": float(params["price_cost"]),
        "shop_value_reserve_shortfall_cost": float(params["reserve_cost"]),
        "shop_value_future_shop_reserve": int(params["future_shop_reserve"]),
        "shop_value_future_shop_horizon": int(params["future_shop_horizon"]),
        "shop_value_card_scale": float(params["card_scale"]),
        "shop_value_card_reference_price": float(args.shop_value_card_reference_price),
        "shop_value_card_price_factor_min": float(args.shop_value_card_price_factor_min),
        "shop_value_card_price_factor_max": float(args.shop_value_card_price_factor_max),
        "shop_value_potion_scale": float(params["potion_scale"]),
        "shop_value_relic_scale": float(params["relic_scale"]),
        "shop_value_item_scale": float(args.shop_value_item_scale),
        "shop_value_threshold": float(params["shop_value_threshold"]),
        "shop_prior_weight_override": float(params["w"]),
        "write_results_json": bool(args.write_results_json),
        "start_snapshot_cache_entries": int(args.start_snapshot_cache_entries),
        "selectors_preloaded": True,
    }


def _worker_init(config: dict[str, Any]) -> None:
    rollout_eval._init_worker(config)


def _snapshot_metrics_payload(
    sources: Counter[str],
    potion_actions_by_id: Counter[str],
    potion_actions_by_room_type: Counter[str],
    shop_actions_by_item_kind: Counter[str],
    shop_spend_by_item_kind: Counter[str],
) -> dict[str, Any]:
    return {
        "sources": dict(sources),
        "potion_action_count": sum(potion_actions_by_id.values()),
        "potion_actions_by_id": dict(potion_actions_by_id),
        "potion_actions_by_room_type": dict(potion_actions_by_room_type),
        "shop_action_count": sum(shop_actions_by_item_kind.values()),
        "shop_actions_by_item_kind": dict(shop_actions_by_item_kind),
        "shop_spend_by_item_kind": dict(shop_spend_by_item_kind),
        "shop_spend_total": sum(shop_spend_by_item_kind.values()),
    }


def _snapshot_terminal_result(
    seed: int,
    env: Any,
    steps: int,
    started: float,
    error: str | None,
    prefix_metrics: dict[str, Any],
) -> dict[str, Any]:
    phase = str(getattr(env, "phase", ""))
    max_steps = int(_CONFIG_SNAPSHOT.get("max_steps", 1500))
    max_floor = int(_CONFIG_SNAPSHOT.get("max_floor", 60))
    floor = int(getattr(env, "floor", 0))
    return {
        "seed": int(seed),
        "ascension": int(getattr(env, "ascension_level", 0)),
        "phase": phase,
        "floor": floor,
        "hp": int(getattr(getattr(env, "player", None), "current_hp", 0)),
        "max_hp": int(getattr(getattr(env, "player", None), "max_hp", 0)),
        "gold": int(getattr(env, "gold", 0)),
        "deck_size": len(list(getattr(env, "deck", []) or [])),
        "relic_count": len(list(getattr(env, "relics", []) or [])),
        "potion_count": len(list(getattr(env, "potions", []) or [])),
        "steps": int(steps),
        "won": phase in {"COMPLETE", "VICTORY"},
        "dead": phase == "GAME_OVER",
        "timed_out": phase not in rollout_eval.TERMINAL_PHASES and error is None and int(steps) >= max_steps,
        "max_floor_stopped": floor > max_floor and phase not in rollout_eval.TERMINAL_PHASES,
        "error": error,
        "trace_path": None,
        "seconds": time.time() - started,
    } | prefix_metrics


_CONFIG_SNAPSHOT: dict[str, Any] = {}


def _build_first_shop_snapshot(config: dict[str, Any], cache_dir: str, seed: int) -> str:
    global _CONFIG_SNAPSHOT
    _CONFIG_SNAPSHOT = dict(config)
    rollout_eval._init_worker(config)
    assert rollout_eval._SELECTORS is not None
    cache_path = Path(cache_dir) / f"seed_{int(seed)}.pkl"
    if cache_path.exists():
        return str(cache_path)

    started = time.time()
    env = rollout_eval.NativeRunEnv(seed=int(seed), ascension_level=int(config["ascension"]), enable_neow=True)
    steps = 0
    error: str | None = None
    sources: Counter[str] = Counter()
    potion_actions_by_id: Counter[str] = Counter()
    potion_actions_by_room_type: Counter[str] = Counter()
    shop_actions_by_item_kind: Counter[str] = Counter()
    shop_spend_by_item_kind: Counter[str] = Counter()
    max_steps = int(config["max_steps"])
    max_floor = int(config["max_floor"])
    for _ in range(max_steps):
        phase = str(getattr(env, "phase", ""))
        if phase == "SHOP" or phase in rollout_eval.TERMINAL_PHASES or int(getattr(env, "floor", 0)) > max_floor:
            break
        pre_room_type = str(getattr(env, "current_room_type", "") or phase or "UNKNOWN")
        try:
            action, _, source = rollout_eval.choose_model_required_action(env, rollout_eval._SELECTORS, return_scores=False)
            if str(action.get("kind") or "") == "potion":
                potion_id = str(action.get("potion_id") or action.get("name") or "UNKNOWN")
                potion_actions_by_id[potion_id] += 1
                potion_actions_by_room_type[pre_room_type or "UNKNOWN"] += 1
            if str(action.get("kind") or "") == "shop":
                item_kind = str(action.get("item_kind") or "UNKNOWN")
                shop_actions_by_item_kind[item_kind] += 1
                try:
                    price = max(0, int(action.get("price") or 0))
                except (TypeError, ValueError):
                    price = 0
                shop_spend_by_item_kind[item_kind] += price
            env.step(action)
            sources[str(source)] += 1
            steps += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break

    phase = str(getattr(env, "phase", ""))
    prefix_metrics = _snapshot_metrics_payload(
        sources,
        potion_actions_by_id,
        potion_actions_by_room_type,
        shop_actions_by_item_kind,
        shop_spend_by_item_kind,
    )
    if phase == "SHOP" and error is None:
        payload = {
            "version": 2,
            "kind": "first_shop",
            "seed": int(seed),
            "steps": int(steps),
            "phase": phase,
            "floor": int(getattr(env, "floor", 0)),
            "prefix_metrics": prefix_metrics,
            "env_blob": pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL),
        }
    else:
        payload = {
            "version": 2,
            "kind": "first_shop_terminal",
            "seed": int(seed),
            "steps": int(steps),
            "phase": phase,
            "floor": int(getattr(env, "floor", 0)),
            "prefix_metrics": prefix_metrics,
            "terminal_result": _snapshot_terminal_result(int(seed), env, steps, started, error, prefix_metrics),
        }
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)
    return str(cache_path)


def _run_seed_chunk(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    rollout_eval._init_worker(config)
    return [rollout_eval._run_seed(seed) for seed in seeds]


def _chunks(values: list[int], chunk_size: int) -> list[list[int]]:
    chunk_size = max(1, int(chunk_size))
    return [values[start:start + chunk_size] for start in range(0, len(values), chunk_size)]


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_label(seed_start: int, count: int) -> str:
    return f"seed{int(seed_start)}_{int(seed_start) + int(count) - 1}"


def _record_from_summary(
    trial_id: str,
    params: dict[str, Any],
    count: int,
    output_dir: Path,
    summary: dict[str, Any],
    *,
    seed_start: int,
) -> dict[str, Any]:
    shop = summary.get("shop_actions_by_item_kind") or {}
    return {
        "trial_id": trial_id,
        "count": int(count),
        "seed_start": int(seed_start),
        "seed_end": int(seed_start) + int(count) - 1,
        "params": dict(params),
        "mean_floor": float(summary.get("mean_floor", 0.0) or 0.0),
        "win_count": int(summary.get("win_count", 0) or 0),
        "timeout_count": int(summary.get("timeout_count", 0) or 0),
        "error_count": int(summary.get("error_count", 0) or 0),
        "shop_actions_by_item_kind": shop,
        "potion_relic_ratio": float(shop.get("potion", 0) or 0) / max(1.0, float(shop.get("relic", 0) or 0)) if shop else None,
        "seconds": float(summary.get("seconds", 0.0) or 0.0),
        "output_dir": str(output_dir),
    }


def _write_trial_outputs(output_dir: Path, config: dict[str, Any], seeds: list[int], results: list[dict[str, Any]], started: float) -> dict[str, Any]:
    results.sort(key=lambda item: int(item["seed"]))
    (output_dir / "config.json").write_text(
        json.dumps(config | {"seeds": seeds}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    if bool(config.get("write_results_json")):
        (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    summary = rollout_eval._summarize(results, started=started, output_dir=output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _evaluate_trial(
    executor: ProcessPoolExecutor,
    args: argparse.Namespace,
    trial_id: str,
    params: dict[str, Any],
    count: int,
    *,
    seed_start: int = 1,
    metrics_mode: str | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir / f"{trial_id}_{_seed_label(seed_start, count)}"
    summary_path = output_dir / "summary.json"
    existing = _load_summary(summary_path)
    if existing is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        seeds = list(range(int(seed_start), int(seed_start) + int(count)))
        selected_metrics_mode = str(metrics_mode or args.promoted_metrics_mode)
        config = _base_config(args, output_dir, params, metrics_mode=selected_metrics_mode)
        start_snapshot_dir = _prepare_first_shop_cache(executor, args, seeds, selected_metrics_mode)
        if start_snapshot_dir is not None:
            config["start_snapshot_dir"] = str(start_snapshot_dir)
        started = time.time()
        futures = [
            executor.submit(_run_seed_chunk, config, chunk)
            for chunk in _chunks(seeds, int(args.seed_chunk_size))
        ]
        results: list[dict[str, Any]] = []
        for future in as_completed(futures):
            results.extend(future.result())
        summary = _write_trial_outputs(output_dir, config, seeds, results, started)
    else:
        summary = existing

    record = _record_from_summary(trial_id, params, count, output_dir, summary, seed_start=seed_start)
    if existing is not None:
        record["_resumed"] = True
    else:
        with (args.output_dir / "trials.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _evaluate_trials_batch(
    executor: ProcessPoolExecutor,
    args: argparse.Namespace,
    specs: list[tuple[str, dict[str, Any]]],
    count: int,
    *,
    seed_start: int = 1,
    metrics_mode: str | None = None,
) -> list[dict[str, Any]]:
    records_by_id: dict[str, dict[str, Any]] = {}
    pending: dict[Any, str] = {}
    trial_state: dict[str, dict[str, Any]] = {}
    task_specs: list[tuple[str, dict[str, Any], list[int]]] = []
    seeds = list(range(int(seed_start), int(seed_start) + int(count)))
    selected_metrics_mode = str(metrics_mode or args.promoted_metrics_mode)
    start_snapshot_dir = _prepare_first_shop_cache(executor, args, seeds, selected_metrics_mode)
    max_pending = int(args.max_pending_chunks)
    if max_pending <= 0:
        max_pending = max(1, int(args.workers) * 4)

    for trial_id, params in specs:
        output_dir = args.output_dir / f"{trial_id}_{_seed_label(seed_start, count)}"
        existing = _load_summary(output_dir / "summary.json")
        if existing is not None:
            record = _record_from_summary(trial_id, params, count, output_dir, existing, seed_start=seed_start)
            record["_resumed"] = True
            records_by_id[trial_id] = record
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(args, output_dir, params, metrics_mode=selected_metrics_mode)
        if start_snapshot_dir is not None:
            config["start_snapshot_dir"] = str(start_snapshot_dir)
        chunks = _chunks(seeds, int(args.seed_chunk_size))
        trial_state[trial_id] = {
            "params": params,
            "output_dir": output_dir,
            "config": config,
            "started": time.time(),
            "remaining": len(chunks),
            "results": [],
        }
        for chunk in chunks:
            task_specs.append((trial_id, config, chunk))

    next_task_index = 0

    def submit_until_capacity() -> None:
        nonlocal next_task_index
        while next_task_index < len(task_specs) and len(pending) < max_pending:
            trial_id, config, chunk = task_specs[next_task_index]
            pending[executor.submit(_run_seed_chunk, config, chunk)] = trial_id
            next_task_index += 1

    submit_until_capacity()

    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            trial_id = pending.pop(future)
            state = trial_state[trial_id]
            try:
                chunk_results = future.result()
            except Exception as exc:
                raise RuntimeError(f"seed chunk failed while evaluating {trial_id}_{_seed_label(seed_start, count)}") from exc
            state["results"].extend(chunk_results)
            state["remaining"] -= 1
            if state["remaining"] == 0:
                summary = _write_trial_outputs(
                    state["output_dir"],
                    state["config"],
                    seeds,
                    state["results"],
                    float(state["started"]),
                )
                records_by_id[trial_id] = _record_from_summary(
                    trial_id,
                    state["params"],
                    count,
                    state["output_dir"],
                    summary,
                    seed_start=seed_start,
                )
        submit_until_capacity()

    for trial_id, state in trial_state.items():
        if trial_id in records_by_id:
            continue
        state = trial_state[trial_id]
        raise RuntimeError(
            f"trial {trial_id}_{_seed_label(seed_start, count)} did not finish: "
            f"remaining={state['remaining']} results={len(state['results'])}"
        )

    ordered = [records_by_id[trial_id] for trial_id, _ in specs]
    with (args.output_dir / "trials.jsonl").open("a", encoding="utf-8") as handle:
        for record in ordered:
            if record.get("_resumed"):
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return ordered


def _prepare_first_shop_cache(executor: ProcessPoolExecutor, args: argparse.Namespace, seeds: list[int], metrics_mode: str) -> Path | None:
    if not bool(args.early_first_shop_cache):
        return None
    if str(metrics_mode) not in {"floor", "full"} or not seeds:
        return None
    cache_root = args.output_dir / "_first_shop_cache_v2"
    cache_dir = cache_root / "shared"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        target = cache_dir / f"seed_{int(seed)}.pkl"
        if target.exists():
            continue
        for source in cache_root.glob(f"seed*/seed_{int(seed)}.pkl"):
            if not source.is_file():
                continue
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            break
    missing = [seed for seed in seeds if not (cache_dir / f"seed_{seed}.pkl").exists()]
    if not missing:
        return cache_dir
    params = _midpoint_params()
    config = _base_config(args, cache_dir, params, metrics_mode=str(metrics_mode))
    futures = [executor.submit(_build_first_shop_snapshot, config, str(cache_dir), seed) for seed in missing]
    for future in as_completed(futures):
        future.result()
    return cache_dir


def _print_record(prefix: str, record: dict[str, Any]) -> None:
    if record.get("_resumed"):
        return
    params = record["params"]
    ratio = record.get("potion_relic_ratio")
    ratio_text = "NA" if ratio is None else f"{float(ratio):.3f}"
    print(
        f"{prefix} {record['trial_id']} seed{record.get('seed_start', 1)}-{record.get('seed_end', record['count'])} "
        f"n={record['count']} mean={record['mean_floor']:.3f} "
        f"wins={record['win_count']} to={record['timeout_count']} err={record['error_count']} "
        f"p/r={ratio_text} shop={record['shop_actions_by_item_kind']} "
        f"params=w={params['w']:.3f},card={params['card_scale']:.3f},relic={params['relic_scale']:.3f},"
        f"potion={params['potion_scale']:.3f},price={params['price_cost']:.4f},reserve_cost={params['reserve_cost']:.4f},"
        f"future_reserve={int(params['future_shop_reserve'])},horizon={int(params['future_shop_horizon'])},"
        f"threshold={params['shop_value_threshold']:.3f}",
        flush=True,
    )


def _top(records: list[dict[str, Any]], count: int, limit: int, *, seed_start: int | None = None) -> list[dict[str, Any]]:
    candidates = (record for record in records if int(record["count"]) == int(count))
    if seed_start is not None:
        candidates = (record for record in candidates if int(record.get("seed_start", 1)) == int(seed_start))
    return sorted(candidates, key=_trial_sort_key, reverse=True)[:limit]


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not str(key).startswith("_")}


def _public_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_public_record(record) for record in records]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CEM shop parameter search using one persistent rollout worker pool.")
    parser.add_argument("--output-dir", type=Path, default=Path("_cache/shop_cem_prior_delta_mean_floor_batched"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed-chunk-size", type=int, default=2)
    parser.add_argument(
        "--max-pending-chunks",
        type=int,
        default=0,
        help="Limit submitted seed chunks per round batch. 0 means workers*4.",
    )
    parser.add_argument("--round-batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-first-shop-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-metrics-mode", choices=["full", "floor"], default="floor")
    parser.add_argument("--promoted-metrics-mode", choices=["full", "floor"], default="full")
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--round1", type=int, default=192)
    parser.add_argument("--round2", type=int, default=144)
    parser.add_argument("--elite", type=int, default=32)
    parser.add_argument("--promote-round3", type=int, default=36)
    parser.add_argument("--promote-final", type=int, default=12)
    parser.add_argument("--promote-robust", type=int, default=6)
    parser.add_argument("--early-count", type=int, default=80)
    parser.add_argument("--round3-count", type=int, default=180)
    parser.add_argument("--final-count", type=int, default=300)
    parser.add_argument("--robust-seed-start", type=int, default=301)
    parser.add_argument("--robust-count", type=int, default=300)
    parser.add_argument("--min-sigma-frac", type=float, default=0.08)
    parser.add_argument("--write-results-json", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--start-snapshot-cache-entries", type=int, default=512)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default=None)
    parser.add_argument("--combat-selector", choices=["legacy-slot", "v3-candidate"], default="v3-candidate")
    parser.add_argument("--combat-model", type=Path, default=Path("models/combat.pt"))
    parser.add_argument("--v3-combat-model", type=Path, default=Path("models/v3_combat_scorer_potion_stage5.pt"))
    parser.add_argument("--card-reward-model", type=Path, default=Path("models/card_reward.pt"))
    parser.add_argument("--shop-choice-model", type=Path, default=Path("models/shop_choice_prior_delta.pt"))
    parser.add_argument("--shop-value-future-shop-reserve", type=int, default=120)
    parser.add_argument("--shop-value-future-shop-horizon", type=int, default=5)
    parser.add_argument("--shop-value-card-reference-price", type=float, default=60.0)
    parser.add_argument("--shop-value-card-price-factor-min", type=float, default=0.65)
    parser.add_argument("--shop-value-card-price-factor-max", type=float, default=1.35)
    parser.add_argument("--shop-value-item-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))
    started = time.time()
    all_records: list[dict[str, Any]] = []

    preload_config = _base_config(
        args,
        args.output_dir / "_preload",
        _midpoint_params(),
        metrics_mode="floor",
    )
    rollout_eval._init_worker(preload_config)
    mp_context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_worker_init,
        initargs=(preload_config,),
        mp_context=mp_context,
    ) as executor:
        print(f"output_dir={args.output_dir}", flush=True)
        print(
            f"round1: {args.round1} trials, anchors={len(ANCHOR_PARAMS)}, seed1-{args.early_count}",
            flush=True,
        )
        anchor_specs = [
            (f"a{index:02d}_{name}", _normalize_params(params))
            for index, (name, params) in enumerate(ANCHOR_PARAMS, 1)
        ][: max(0, int(args.round1))]
        random_count = max(0, int(args.round1) - len(anchor_specs))
        round1_specs = [
            *anchor_specs,
            *[(f"r1_{index:03d}", _sample_uniform(rng)) for index in range(1, random_count + 1)],
        ]
        round1_records = (
            _evaluate_trials_batch(
                executor,
                args,
                round1_specs,
                int(args.early_count),
                seed_start=1,
                metrics_mode=str(args.early_metrics_mode),
            )
            if args.round_batch
            else [
                _evaluate_trial(
                    executor,
                    args,
                    trial_id,
                    params,
                    int(args.early_count),
                    seed_start=1,
                    metrics_mode=str(args.early_metrics_mode),
                )
                for trial_id, params in round1_specs
            ]
        )
        for record in round1_records:
            all_records.append(record)
            _print_record("R1", record)

        elite = _top(all_records, int(args.early_count), int(args.elite), seed_start=1)
        mean = {name: sum(record["params"][name] for record in elite) / len(elite) for name in PARAM_RANGES}
        sigma = {}
        for name, (lo, hi) in PARAM_RANGES.items():
            variance = sum((record["params"][name] - mean[name]) ** 2 for record in elite) / max(1, len(elite) - 1)
            sigma[name] = max(math.sqrt(variance), (hi - lo) * float(args.min_sigma_frac))
        print("round1_elite_mean", json.dumps(mean, ensure_ascii=False), flush=True)
        print("round1_elite_sigma", json.dumps(sigma, ensure_ascii=False), flush=True)

        print(f"round2: {args.round2} CEM trials, seed1-{args.early_count}", flush=True)
        round2_specs = [(f"r2_{index:03d}", _sample_normal(rng, mean, sigma)) for index in range(1, int(args.round2) + 1)]
        round2_records = (
            _evaluate_trials_batch(
                executor,
                args,
                round2_specs,
                int(args.early_count),
                seed_start=1,
                metrics_mode=str(args.early_metrics_mode),
            )
            if args.round_batch
            else [
                _evaluate_trial(
                    executor,
                    args,
                    trial_id,
                    params,
                    int(args.early_count),
                    seed_start=1,
                    metrics_mode=str(args.early_metrics_mode),
                )
                for trial_id, params in round2_specs
            ]
        )
        for record in round2_records:
            all_records.append(record)
            _print_record("R2", record)

        print(f"promote top {args.promote_round3} to seed1-{args.round3_count}", flush=True)
        promoted_round3_specs = [
            (f"p{int(args.round3_count)}_{index:03d}_{base_record['trial_id']}", base_record["params"])
            for index, base_record in enumerate(_top(all_records, int(args.early_count), int(args.promote_round3), seed_start=1), 1)
        ]
        promoted_round3 = (
            _evaluate_trials_batch(
                executor,
                args,
                promoted_round3_specs,
                int(args.round3_count),
                seed_start=1,
                metrics_mode=str(args.promoted_metrics_mode),
            )
            if args.round_batch
            else [
                _evaluate_trial(
                    executor,
                    args,
                    trial_id,
                    params,
                    int(args.round3_count),
                    seed_start=1,
                    metrics_mode=str(args.promoted_metrics_mode),
                )
                for trial_id, params in promoted_round3_specs
            ]
        )
        for record in promoted_round3:
            all_records.append(record)
            _print_record("P3", record)

        print(f"promote top {args.promote_final} to seed1-{args.final_count}", flush=True)
        promoted_final_specs = [
            (f"p{int(args.final_count)}_{index:03d}_{base_record['trial_id']}", base_record["params"])
            for index, base_record in enumerate(
                sorted(promoted_round3, key=_trial_sort_key, reverse=True)[: int(args.promote_final)],
                1,
            )
        ]
        promoted_final = (
            _evaluate_trials_batch(
                executor,
                args,
                promoted_final_specs,
                int(args.final_count),
                seed_start=1,
                metrics_mode=str(args.promoted_metrics_mode),
            )
            if args.round_batch
            else [
                _evaluate_trial(
                    executor,
                    args,
                    trial_id,
                    params,
                    int(args.final_count),
                    seed_start=1,
                    metrics_mode=str(args.promoted_metrics_mode),
                )
                for trial_id, params in promoted_final_specs
            ]
        )
        for record in promoted_final:
            all_records.append(record)
            _print_record("PF", record)

        print(
            f"robust top {args.promote_robust} to seed{args.robust_seed_start}-"
            f"{int(args.robust_seed_start) + int(args.robust_count) - 1}",
            flush=True,
        )
        robust_specs = [
            (
                f"robust_{index:03d}_{base_record['trial_id']}",
                base_record["params"],
            )
            for index, base_record in enumerate(
                sorted(promoted_final, key=_trial_sort_key, reverse=True)[: int(args.promote_robust)],
                1,
            )
        ]
        robust_records = (
            _evaluate_trials_batch(
                executor,
                args,
                robust_specs,
                int(args.robust_count),
                seed_start=int(args.robust_seed_start),
                metrics_mode=str(args.promoted_metrics_mode),
            )
            if args.round_batch
            else [
                _evaluate_trial(
                    executor,
                    args,
                    trial_id,
                    params,
                    int(args.robust_count),
                    seed_start=int(args.robust_seed_start),
                    metrics_mode=str(args.promoted_metrics_mode),
                )
                for trial_id, params in robust_specs
            ]
        )
        for record in robust_records:
            all_records.append(record)
            _print_record("RB", record)

    summary = {
        "output_dir": str(args.output_dir),
        "objective": "mean_floor",
        "param_ranges": PARAM_RANGES,
        "integer_params": sorted(INTEGER_PARAMS),
        "anchors": [{"name": name, "params": _normalize_params(params)} for name, params in ANCHOR_PARAMS],
        "total_wall_seconds": time.time() - started,
        "best_early": _public_records(_top(all_records, int(args.early_count), 20, seed_start=1)),
        "best_round3": _public_records(sorted(
            (
                record
                for record in all_records
                if int(record["count"]) == int(args.round3_count) and int(record.get("seed_start", 1)) == 1
            ),
            key=_trial_sort_key,
            reverse=True,
        )),
        "best_final": _public_records(sorted(
            (
                record
                for record in all_records
                if int(record["count"]) == int(args.final_count) and int(record.get("seed_start", 1)) == 1
            ),
            key=_trial_sort_key,
            reverse=True,
        )),
        "best_robust": _public_records(sorted(
            (
                record
                for record in all_records
                if int(record["count"]) == int(args.robust_count)
                and int(record.get("seed_start", 1)) == int(args.robust_seed_start)
            ),
            key=_trial_sort_key,
            reverse=True,
        )),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "DONE",
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "best_final": summary["best_final"][: int(args.promote_robust)],
                "best_robust": summary["best_robust"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        Path("logs").mkdir(parents=True, exist_ok=True)
        with Path("logs/shop_cem_prior_delta_search_batched.error.log").open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
            handle.write(traceback.format_exc())
        raise
