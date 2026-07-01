#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import pickle
import run_v3_teacher_config_sweep as sweep_base
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from queue import Empty
from typing import Any

from run_v3_teacher_config_sweep import (
    PARAM_RANGES,
    PARAM_BUCKETS,
    _candidate_payload,
    _corner_sample,
    _default_teacher_config,
    _latin_hypercube_sample,
    _rank_results,
    param_ranges_for_bucket,
)


_FAST_BASE_CONFIG: dict[str, Any] = {}
_FAST_CANDIDATE_PARAMS: dict[str, dict[str, float]] = {}
_FAST_SEARCH_OVERRIDES: dict[str, Any] = {}
_FAST_TEACHER_CONFIGS: dict[str, Any] = {}
_FAST_LAST_CONFIG_KEY: str | None = None
_FAST_CANDIDATES_ARE_POTION_ONLY: bool = False
_FAST_GROUP_ALL_COMBAT: bool = False
_FAST_BATCH_NON_POTION_ROOTS: bool = False
_FAST_PROGRESS_QUEUE: Any | None = None
_FAST_ACTIVE_CANDIDATE_ID: str | None = None
_PREPARED_SNAPSHOT_COUNTS: dict[str, int] = {}
_CLEARED_AGGREGATE_RESULT_FILES: set[str] = set()
_CLEARED_CANDIDATE_DIRS: set[str] = set()
_POTION_SWEEP_PARAM_NAMES = {
    "potion_monster_room_reward_factor",
    "potion_elite_room_reward_factor",
    "potion_boss_room_reward_factor",
    "potion_cost_scale",
    "potion_buff_adjustment_scale",
    "potion_generation_adjustment_scale",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _parse_int_csv(raw: str, *, fallback: list[int]) -> list[int]:
    if not raw.strip():
        return list(fallback)
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    return values or list(fallback)


def _grid_sample(raw: str) -> list[dict[str, float]]:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("--round1-grid-json must be a non-empty JSON object")
    names: list[str] = []
    values_by_name: list[list[float]] = []
    for name, raw_values in payload.items():
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"Invalid grid for {name!r}: expected a non-empty list")
        names.append(str(name))
        values_by_name.append([float(value) for value in raw_values])
    return [
        {name: float(value) for name, value in zip(names, values, strict=True)}
        for values in itertools.product(*values_by_name)
    ]


def _thread_env(thread_count: int) -> dict[str, str]:
    value = str(max(1, int(thread_count)))
    return {
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "OMP_DYNAMIC": "FALSE",
    }


def _fast_bool_env(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _reuse_existing_result(result: dict[str, Any]) -> bool:
    if _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_RERUN_ERROR_RESULTS", True) and result.get("error"):
        return False
    if _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_RERUN_TIMED_OUT_RESULTS", False) and result.get("timed_out"):
        return False
    return True


def _load_existing_results(output_dir: Path, seeds: set[int]) -> dict[int, dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for path in (output_dir / "results.jsonl", output_dir / "results.json"):
        if not path.exists():
            continue
        try:
            if path.suffix == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        result = json.loads(line)
                        seed = int(result["seed"])
                        if seed in seeds and _reuse_existing_result(result):
                            by_seed[seed] = result
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    for result in payload:
                        seed = int(result["seed"])
                        if seed in seeds and _reuse_existing_result(result):
                            by_seed[seed] = result
        except Exception:
            continue
    return by_seed


def _load_round_aggregate_results(path: Path, seeds: set[int]) -> dict[str, dict[int, dict[str, Any]]]:
    by_candidate: dict[str, dict[int, dict[str, Any]]] = {}
    if not path.exists():
        return by_candidate
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                result = json.loads(line)
                candidate_id = str(result.pop("_candidate_id", result.pop("candidate_id", "")) or "")
                if not candidate_id:
                    continue
                seed = int(result["seed"])
                if seed not in seeds:
                    continue
                if not _reuse_existing_result(result):
                    continue
                by_candidate.setdefault(candidate_id, {})[seed] = result
    except Exception:
        return {}
    return by_candidate


def _load_complete_summary(eval_dir: Path, target_count: int) -> dict[str, Any] | None:
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(summary, dict):
        return None
    try:
        count = int(summary.get("count") or 0)
    except (TypeError, ValueError):
        count = 0
    if count < int(target_count):
        return None
    return summary


def _append_result_jsonl(path: Path, result: dict[str, Any]) -> None:
    _append_results_jsonl(path, [result])


def _append_results_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for result in results:
            payload = {key: value for key, value in result.items() if key != "_candidate_id"} if "_candidate_id" in result else result
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_aggregate_results_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def _write_json_fast(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def _save_leaderboard_fast(output_dir: Path, round_name: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = _rank_results(results)
    compact = [
        {
            "rank": index + 1,
            "candidate_id": result["candidate_id"],
            "mean_floor": result["summary"].get("mean_floor"),
            "win_count": result["summary"].get("win_count"),
            "death_count": result["summary"].get("death_count"),
            "p25_floor": result["summary"].get("p25_floor"),
            "p50_floor": result["summary"].get("p50_floor"),
            "p75_floor": result["summary"].get("p75_floor"),
            "error_count": result["summary"].get("error_count"),
            "params": result["params"],
            "eval_dir": result["eval_dir"],
        }
        for index, result in enumerate(ranked)
    ]
    _write_json_fast(output_dir / f"leaderboard_{round_name}.json", compact)
    _write_json_fast(output_dir / "leaderboard_latest.json", compact)
    return ranked


def _summarize(results: list[dict[str, Any]], *, started: float, output_dir: Path) -> dict[str, Any]:
    from evaluate_v3_rollout_batch import _summarize as summarize_rollouts

    return summarize_rollouts(results, started=started, output_dir=output_dir)


def _base_eval_config(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    return {
        "repo_root": str(root.resolve()),
        "output_dir": str(args.output_dir / "_task_pool_worker"),
        "ascension": 0,
        "max_floor": int(args.max_floor),
        "max_steps": int(args.max_steps),
        "no_progress_limit": 0,
        "trace_mode": "none",
        "metrics_mode": str(args.metrics_mode),
        "compact_floor_results": str(args.metrics_mode) == "floor",
        "torch_threads": int(args.torch_threads),
        "device": "cpu",
        "combat_device": "cpu",
        "combat_selector": "v3-teacher",
        "fast_teacher_combat_direct": bool(args.fast_teacher_combat_direct),
        "v3_teacher_safe_single_action_inplace": _fast_bool_env(
            "SPIRECOMM_V3_TEACHER_SAFE_SINGLE_ACTION_INPLACE",
            True,
        ),
        "v3_teacher_combat_branch_only": _fast_bool_env(
            "SPIRECOMM_V3_TEACHER_COMBAT_BRANCH_ONLY",
            False,
        ),
        "v3_teacher_dedupe_equivalent_card_actions": _fast_bool_env(
            "SPIRECOMM_V3_TEACHER_DEDUPE_EQUIVALENT_CARD_ACTIONS",
            True,
        ),
        "combat_model": str(root / "models" / "combat.pt"),
        "v3_combat_model": str(root / "models" / "v3_combat_scorer.pt"),
        "teacher_config_json": "",
        "teacher_config_path": "",
        "card_reward_model": str(root / "models" / "card_reward.pt"),
        "shop_choice_model": str(root / os.environ.get("SPIRECOMM_SHOP_CHOICE_MODEL_PATH", "models/shop_choice_prior_delta.pt")),
        "shop_policy": os.environ.get("SPIRECOMM_SHOP_POLICY", "value"),
        "shop_value_price_cost": float(os.environ.get("SPIRECOMM_SHOP_VALUE_PRICE_COST", "0.044348003822393976")),
        "shop_value_reserve_shortfall_cost": float(os.environ.get("SPIRECOMM_SHOP_VALUE_RESERVE_SHORTFALL_COST", "0.043490245962190935")),
        "shop_value_future_shop_reserve": int(os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE", "120")),
        "shop_value_future_shop_horizon": int(os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_HORIZON", "5")),
        "shop_value_card_scale": float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_SCALE", "4.6262945279949435")),
        "shop_value_card_reference_price": float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_REFERENCE_PRICE", "60.0")),
        "shop_value_card_price_factor_min": float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MIN", "0.65")),
        "shop_value_card_price_factor_max": float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MAX", "1.35")),
        "shop_value_potion_scale": float(os.environ.get("SPIRECOMM_SHOP_VALUE_POTION_SCALE", "0.5084989138155764")),
        "shop_value_relic_scale": float(os.environ.get("SPIRECOMM_SHOP_VALUE_RELIC_SCALE", "0.8")),
        "shop_value_item_scale": float(os.environ.get("SPIRECOMM_SHOP_VALUE_ITEM_SCALE", "1.0")),
        "shop_value_threshold": float(os.environ.get("SPIRECOMM_SHOP_VALUE_THRESHOLD", "0.0")),
        "shop_prior_weight_override": float(os.environ.get("SPIRECOMM_SHOP_PRIOR_WEIGHT_OVERRIDE", "0.8")),
        "v3_normal_room_potion_penalty": max(0.0, float(os.environ.get("SPIRECOMM_V3_NORMAL_ROOM_POTION_PENALTY", "1.5"))),
        "write_results_json": False,
        "start_snapshot_dir": str(args.start_snapshot_dir or ""),
        "start_snapshot_cache_entries": int(args.start_snapshot_cache_entries),
        "selectors_preloaded": False,
        "blas_threads": int(args.blas_threads),
    }


def _file_fingerprint(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "missing": True}
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _first_combat_snapshot_fingerprint(args: argparse.Namespace) -> str:
    root = _repo_root()
    env_names = [
        "SPIRECOMM_NEOW_FIXED_CHOICE_INDEX",
        "SPIRECOMM_MAP_DP_MONSTER_VALUE",
        "SPIRECOMM_MAP_DP_REST_VALUE",
        "SPIRECOMM_MAP_DP_ELITE_BASE",
        "SPIRECOMM_MAP_DP_GREEN_ELITE_PENALTY",
        "SPIRECOMM_MAP_DP_WINGED_OFFPATH_PENALTY",
        "SPIRECOMM_MAP_DP_SHOP_GOLD_UNIT_VALUE",
        "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_BONUS",
        "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_BONUS",
        "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_HORIZON",
        "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_GOLD_THRESHOLD",
        "SPIRECOMM_SHOP_CHOICE_MODEL_PATH",
    ]
    payload = {
        "version": "first_combat_snapshot_v2",
        "ascension": 0,
        "seed_start": int(args.seed_start),
        "max_steps": int(args.first_combat_snapshot_max_steps),
        "env": {name: os.environ.get(name, "") for name in env_names},
        "files": [
            _file_fingerprint(root / "build_v3_first_combat_snapshots.py"),
            _file_fingerprint(root / "spirecomm" / "ai" / "runtime_decision.py"),
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _prepare_first_combat_snapshots(args: argparse.Namespace, target_count: int) -> Path | None:
    if args.start_snapshot_dir:
        return Path(args.start_snapshot_dir)
    if not bool(args.auto_first_combat_snapshots):
        return None
    count = max(0, int(target_count), int(args.first_combat_snapshot_count or 0))
    if count <= 0:
        return None
    if str(args.first_combat_snapshot_scope) == "global":
        fingerprint = _first_combat_snapshot_fingerprint(args)
        snapshot_dir = (
            Path(args.first_combat_snapshot_global_root)
            / fingerprint
            / f"seed{int(args.seed_start)}"
        )
    else:
        snapshot_dir = args.output_dir / "_first_combat_snapshots" / f"seed{int(args.seed_start)}"
    cache_key = str(snapshot_dir.resolve())
    if _PREPARED_SNAPSHOT_COUNTS.get(cache_key, 0) >= count:
        return snapshot_dir
    seeds = range(int(args.seed_start), int(args.seed_start) + count)
    missing = [seed for seed in seeds if not (snapshot_dir / f"seed_{seed}.pkl").exists()]
    if not missing:
        _PREPARED_SNAPSHOT_COUNTS[cache_key] = count
        return snapshot_dir
    workers = int(args.first_combat_snapshot_workers)
    if workers <= 0:
        workers = int(args.workers)
    workers = max(1, min(workers, len(missing)))
    command = [
        sys.executable,
        "-u",
        str(_repo_root() / "build_v3_first_combat_snapshots.py"),
        "--output-dir",
        str(snapshot_dir),
        "--seed-start",
        str(int(args.seed_start)),
        "--count",
        str(count),
        "--max-steps",
        str(int(args.first_combat_snapshot_max_steps)),
        "--workers",
        str(workers),
        "--torch-threads",
        str(int(args.torch_threads)),
        "--resume",
    ]
    print(
        f"[snapshots] preparing first-combat snapshots dir={snapshot_dir} "
        f"missing={len(missing)}/{count} workers={workers}",
        flush=True,
    )
    env = os.environ.copy()
    env.update(_thread_env(int(args.blas_threads)))
    subprocess.run(command, cwd=str(_repo_root()), env=env, check=True)
    _PREPARED_SNAPSHOT_COUNTS[cache_key] = count
    return snapshot_dir


def _fast_worker_init(
    base_config: dict[str, Any],
    candidate_params: dict[str, dict[str, float]] | None = None,
    search_overrides: dict[str, Any] | None = None,
    progress_queue: Any | None = None,
) -> None:
    global _FAST_BASE_CONFIG, _FAST_CANDIDATE_PARAMS, _FAST_SEARCH_OVERRIDES, _FAST_TEACHER_CONFIGS
    global _FAST_LAST_CONFIG_KEY
    global _FAST_CANDIDATES_ARE_POTION_ONLY, _FAST_GROUP_ALL_COMBAT, _FAST_BATCH_NON_POTION_ROOTS
    global _FAST_PROGRESS_QUEUE
    _FAST_BASE_CONFIG = dict(base_config)
    _FAST_CANDIDATE_PARAMS = {str(key): dict(value) for key, value in (candidate_params or {}).items()}
    _FAST_SEARCH_OVERRIDES = dict(search_overrides or {})
    _FAST_LAST_CONFIG_KEY = None
    _FAST_CANDIDATES_ARE_POTION_ONLY = bool(_FAST_CANDIDATE_PARAMS) and all(
        set(params).issubset(_POTION_SWEEP_PARAM_NAMES)
        for params in _FAST_CANDIDATE_PARAMS.values()
    )
    _FAST_GROUP_ALL_COMBAT = _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_GROUP_ALL_COMBAT", False)
    _FAST_BATCH_NON_POTION_ROOTS = _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_BATCH_NON_POTION_ROOTS", False)
    _FAST_PROGRESS_QUEUE = progress_queue
    os.environ.update(_thread_env(int(base_config.get("blas_threads") or 1)))
    if _FAST_GROUP_ALL_COMBAT and len(_FAST_CANDIDATE_PARAMS) > 1:
        os.environ.setdefault("SPIRECOMM_V3_TEACHER_STEP_BRANCH_CACHE", "1")
        os.environ.setdefault("SPIRECOMM_V3_TEACHER_STEP_BRANCH_CACHE_SIZE", "2048")
    import evaluate_v3_rollout_batch as eval_batch
    from spirecomm.ai.v3_combat_teacher import teacher_config_from_mapping

    _FAST_TEACHER_CONFIGS = {}
    for candidate_id, params in _FAST_CANDIDATE_PARAMS.items():
        teacher_config = dict(params)
        teacher_config.update(_FAST_SEARCH_OVERRIDES)
        teacher_payload = {"teacher_config": teacher_config}
        _FAST_TEACHER_CONFIGS[candidate_id] = teacher_config_from_mapping(teacher_payload)

    eval_batch._RUN_PROGRESS_HOOK = _eval_run_progress_hook if progress_queue is not None else None
    eval_batch._init_worker(dict(base_config))


def _activate_fast_candidate_config(eval_batch: Any, candidate_id: str) -> None:
    global _FAST_LAST_CONFIG_KEY, _FAST_ACTIVE_CANDIDATE_ID
    _FAST_ACTIVE_CANDIDATE_ID = str(candidate_id)
    config_key = str(candidate_id)
    if config_key == _FAST_LAST_CONFIG_KEY and eval_batch._SELECTORS is not None:
        return
    if eval_batch._SELECTORS is None:
        eval_batch._init_worker(dict(_FAST_BASE_CONFIG))
    combat = (eval_batch._SELECTORS or {}).get("combat")
    if combat is not None and hasattr(combat, "config"):
        combat.config = _FAST_TEACHER_CONFIGS[str(candidate_id)]
    _FAST_LAST_CONFIG_KEY = config_key


def _eval_run_progress_hook(**payload: Any) -> None:
    seed = int(payload.get("seed") or 0)
    step_delta = int(payload.get("step_delta") or 0)
    _worker_live_progress_work(path="plain_step", seed=seed, units=max(1, step_delta))


def _preload_selectors_for_fork(
    base_config: dict[str, Any],
    *,
    preload_start_snapshots: bool = False,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    if "fork" not in mp.get_all_start_methods():
        return dict(base_config)
    config = dict(base_config)
    os.environ.update(_thread_env(int(config.get("blas_threads") or 1)))
    import evaluate_v3_rollout_batch as eval_batch

    if eval_batch._SELECTORS is None:
        preload_config = dict(config)
        preload_config["selectors_preloaded"] = False
        eval_batch._init_worker(preload_config)
    if preload_start_snapshots and str(config.get("start_snapshot_dir") or ""):
        loaded = eval_batch._preload_start_snapshot_cache(
            str(config.get("start_snapshot_dir") or ""),
            seeds,
            max_entries=int(config.get("start_snapshot_cache_entries") or 0),
        )
        if loaded > 0:
            print(f"[snapshots] prefork_loaded={loaded}", flush=True)
    config["selectors_preloaded"] = True
    return config


def _run_seed_task(task: tuple[str, int]) -> tuple[str, dict[str, Any]]:
    candidate_id, seed = task
    import evaluate_v3_rollout_batch as eval_batch

    _activate_fast_candidate_config(eval_batch, str(candidate_id))
    result = eval_batch._run_seed(int(seed))
    return candidate_id, result


def _worker_live_progress_tick(
    *,
    path: str,
    candidate_id: str,
    seed: int,
    elapsed: float,
) -> None:
    queue_obj = _FAST_PROGRESS_QUEUE
    if queue_obj is None:
        return
    try:
        queue_obj.put_nowait(
            {
                "kind": "task_done",
                "pid": os.getpid(),
                "path": str(path),
                "candidate_id": str(candidate_id),
                "seed": int(seed),
                "elapsed": float(elapsed),
                "time": time.time(),
            }
        )
    except Exception:
        # Progress telemetry must never affect sweep correctness.
        return


def _worker_live_progress_work(
    *,
    path: str,
    seed: int,
    units: int,
) -> None:
    if units <= 0:
        return
    queue_obj = _FAST_PROGRESS_QUEUE
    if queue_obj is None:
        return
    try:
        queue_obj.put_nowait(
            {
                "kind": "work_done",
                "pid": os.getpid(),
                "path": str(path),
                "seed": int(seed),
                "units": int(units),
                "time": time.time(),
            }
        )
    except Exception:
        # Progress telemetry must never affect sweep correctness.
        return


def _worker_live_progress_result(
    *,
    path: str,
    candidate_id: str,
    seed: int,
    result: dict[str, Any],
) -> None:
    queue_obj = _FAST_PROGRESS_QUEUE
    if queue_obj is None:
        return
    try:
        queue_obj.put_nowait(
            {
                "kind": "result_done",
                "pid": os.getpid(),
                "path": str(path),
                "candidate_id": str(candidate_id),
                "seed": int(seed),
                "result": dict(result),
                "elapsed": 0.0,
                "time": time.time(),
            }
        )
    except Exception:
        # Progress telemetry must never affect sweep correctness.
        return


def _worker_live_branch(branch: dict[str, Any]) -> bool:
    queue_obj = _FAST_PROGRESS_QUEUE
    if queue_obj is None:
        return False
    try:
        queue_obj.put_nowait(
            {
                "kind": "branch_task",
                "pid": os.getpid(),
                "branch": branch,
                "time": time.time(),
            }
        )
        return True
    except Exception:
        return False


def _finish_seed_task_batch(
    tasks: list[tuple[str, int]],
    results: list[tuple[str, dict[str, Any]]],
    *,
    path: str,
    grouped_attempted: bool,
    started: float,
    fallback_error: str | None = None,
    branches: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "results": results,
        "branches": list(branches or []),
        "meta": {
            "task_count": len(tasks),
            "result_count": len(results),
            "seed_count": len({int(seed) for _candidate_id, seed in tasks}),
            "candidate_count": len({str(candidate_id) for candidate_id, _seed in tasks}),
            "path": path,
            "grouped_attempted": bool(grouped_attempted),
            "fallback_error": fallback_error or "",
            "worker_elapsed": max(1e-9, time.perf_counter() - started),
        },
    }


def _is_branch_task_batch(tasks: Any) -> bool:
    return isinstance(tasks, dict) and str(tasks.get("_task_type") or "") == "same_seed_branch"


def _branch_task_rows(task: dict[str, Any]) -> list[tuple[str, int]]:
    seed = int(task["seed"])
    return [(str(candidate_id), seed) for candidate_id in task.get("candidate_ids", [])]


def _serialize_same_seed_branch(
    seed: int,
    candidate_ids: list[str],
    env: Any,
    steps: int,
    *,
    env_blob: bytes | None = None,
) -> dict[str, Any]:
    return {
        "_task_type": "same_seed_branch",
        "seed": int(seed),
        "candidate_ids": [str(candidate_id) for candidate_id in candidate_ids],
        "steps": int(steps),
        "env_blob": env_blob if env_blob is not None else pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL),
    }


def _run_seed_task_batch(
    tasks: list[tuple[str, int]] | dict[str, Any],
) -> dict[str, Any]:
    import evaluate_v3_rollout_batch as eval_batch

    started = time.perf_counter()
    if _is_branch_task_batch(tasks):
        rows = _branch_task_rows(tasks)
        try:
            results, branches = _run_same_seed_branch_task(
                eval_batch,
                tasks,
                offload_splits=_fast_bool_env("SPIRECOMM_TEACHER_SWEEP_OFFLOAD_SPLITS", False),
            )
            return _finish_seed_task_batch(
                rows,
                results,
                path="same_seed_branch",
                grouped_attempted=True,
                started=started,
                branches=branches,
            )
        except Exception as exc:
            # A branch is already mid-rollout, so there is no safe cheap way to
            # restart only this branch with the old seed-level path. Surface the
            # failure instead of silently changing semantics.
            raise RuntimeError(f"same_seed_branch_failed:{type(exc).__name__}:{exc}") from exc

    grouped_attempted = _can_run_same_seed_candidate_batch(eval_batch, tasks)
    fallback_error: str | None = None
    if grouped_attempted:
        try:
            if _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_OFFLOAD_SPLITS", False):
                results, branches = _run_same_seed_candidate_batch_with_branches(
                    eval_batch,
                    tasks,
                    offload_splits=True,
                )
                return _finish_seed_task_batch(
                    tasks,
                    results,
                    path="same_seed_offload",
                    grouped_attempted=True,
                    started=started,
                    branches=branches,
                )
            return _finish_seed_task_batch(
                tasks,
                _run_same_seed_candidate_batch(eval_batch, tasks),
                path="same_seed",
                grouped_attempted=True,
                started=started,
            )
        except Exception as exc:
            # Fall back to the battle-tested per-candidate path. This keeps the
            # grouped rollout an exact speed optimization rather than a new
            # correctness dependency.
            fallback_error = f"{type(exc).__name__}:{exc}"

    results: list[tuple[str, dict[str, Any]]] = []
    for candidate_id, seed in tasks:
        _activate_fast_candidate_config(eval_batch, str(candidate_id))
        result = eval_batch._run_seed(int(seed))
        results.append((candidate_id, result))
        _worker_live_progress_result(
            path="fallback_stream" if grouped_attempted else "plain_stream",
            candidate_id=str(candidate_id),
            seed=int(seed),
            result=result,
        )
    return _finish_seed_task_batch(
        tasks,
        results,
        path="fallback" if grouped_attempted else "plain",
        grouped_attempted=grouped_attempted,
        started=started,
        fallback_error=fallback_error,
    )


def _can_run_same_seed_candidate_batch(eval_batch: Any, tasks: list[tuple[str, int]]) -> bool:
    if len(tasks) <= 1 or not _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_GROUP_SAME_SEED", True):
        return False
    if not (_FAST_CANDIDATES_ARE_POTION_ONLY or _FAST_GROUP_ALL_COMBAT or _FAST_BATCH_NON_POTION_ROOTS):
        return False
    seeds = {int(seed) for _candidate_id, seed in tasks}
    if len(seeds) != 1:
        return False
    config = getattr(eval_batch, "_CONFIG", {}) or {}
    return (
        str(config.get("metrics_mode") or "") == "floor"
        and bool(config.get("compact_floor_results"))
        and str(config.get("trace_mode") or "") == "none"
        and str(config.get("combat_selector") or "") in {"v3-teacher", "teacher"}
        and bool(config.get("fast_teacher_combat_direct"))
        and int(config.get("no_progress_limit") or 0) <= 0
    )


def _compact_rollout_result(env: Any, *, seed: int, steps: int, error: str | None = None) -> dict[str, Any]:
    config = _FAST_BASE_CONFIG
    terminal_phases = {"GAME_OVER", "COMPLETE", "VICTORY"}
    phase = str(getattr(env, "phase", ""))
    max_floor = int(config.get("max_floor") or 60)
    max_steps = int(config.get("max_steps") or 1500)
    floor = int(getattr(env, "floor", 0) or 0)
    total_steps = int(steps)
    return {
        "seed": int(seed),
        "ascension": int(config.get("ascension") or 0),
        "phase": phase,
        "floor": floor,
        "hp": int(getattr(getattr(env, "player", None), "current_hp", 0) or 0),
        "max_hp": int(getattr(getattr(env, "player", None), "max_hp", 0) or 0),
        "gold": int(getattr(env, "gold", 0) or 0),
        "steps": total_steps,
        "won": phase in {"COMPLETE", "VICTORY"},
        "dead": phase == "GAME_OVER",
        "timed_out": phase not in terminal_phases and not error and total_steps >= max_steps,
        "max_floor_stopped": floor > max_floor and phase not in terminal_phases,
        "error": error,
    }


def _snapshot_env_for_seed(eval_batch: Any, seed: int) -> tuple[Any, int, dict[str, Any] | None]:
    from spirecomm.native_sim_v3 import NativeRunEnv

    config = getattr(eval_batch, "_CONFIG", {}) or {}
    start_snapshot_dir = str(config.get("start_snapshot_dir") or "")
    if start_snapshot_dir:
        snapshot = eval_batch._load_start_snapshot(start_snapshot_dir, int(seed))
        if snapshot is not None:
            terminal_result = snapshot.get("terminal_result")
            if isinstance(terminal_result, dict):
                return None, int(snapshot.get("steps") or terminal_result.get("steps") or 0), dict(terminal_result)
            env_blob = snapshot.get("env_blob")
            if isinstance(env_blob, bytes):
                return pickle.loads(env_blob), int(snapshot.get("steps") or 0), None
    env = NativeRunEnv(seed=int(seed), ascension_level=int(config.get("ascension") or 0), enable_neow=True)
    return env, 0, None


def _group_action_key(action: dict[str, Any]) -> Any:
    items = tuple(sorted(action.items()))
    try:
        hash(items)
        return items
    except TypeError:
        return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _choose_group_action(
    eval_batch: Any,
    env: Any,
    candidate_id: str,
    *,
    legal_actions: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None, bool]:
    from spirecomm.ai.runtime_decision import choose_model_required_action

    phase = str(getattr(env, "phase", ""))
    if phase == "COMBAT":
        _activate_fast_candidate_config(eval_batch, str(candidate_id))
        combat_selector = (eval_batch._SELECTORS or {}).get("combat")
        if legal_actions is None:
            legal_actions_env = getattr(combat_selector, "legal_actions_env", None)
            actions = legal_actions_env(env) if callable(legal_actions_env) else env.legal_actions()
        else:
            actions = legal_actions
        has_potion = any((action.get("kind") or "") == "potion" for action in actions)
        if len(actions) == 1:
            return dict(actions[0]), actions, has_potion
        chosen, _scores = combat_selector.choose_env(env, return_scores=False, legal_actions=actions)
        if chosen is None:
            selector_error = str(getattr(combat_selector, "last_error", "") or "")
            raise RuntimeError(f"combat_selector_returned_no_choice:{selector_error}")
        return dict(chosen), actions, has_potion
    action, _scores, _source = choose_model_required_action(env, eval_batch._SELECTORS, return_scores=False)
    return dict(action), None, False


def _step_group_env(env: Any, action: dict[str, Any]) -> Any:
    env.step(dict(action))
    return env


def _run_shell_blob_without_combat(run_env: Any) -> bytes:
    had_combat = hasattr(run_env, "combat")
    old_combat = getattr(run_env, "combat", None)
    try:
        run_env.combat = None
        return pickle.dumps(run_env, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        try:
            if had_combat:
                run_env.combat = old_combat
            else:
                delattr(run_env, "combat")
        except Exception:
            pass


def _step_group_env_from_combat_source_exact(
    env: Any,
    source: Any,
    action: dict[str, Any],
    *,
    run_shell_blob: bytes | None = None,
) -> Any:
    """Step a combat branch using the small combat-engine payload but full run sync."""

    from spirecomm.ai.v3_combat_teacher import (
        _TeacherCombatStepSource,
        _copy_combat_shell_for_teacher_step,
        _copy_run_shell_for_teacher_step,
    )

    if not isinstance(source, _TeacherCombatStepSource):
        raise TypeError("expected _TeacherCombatStepSource")
    payload = pickle.loads(source.payload_blob)
    if source.payload_kind == "engine" and source.combat_env is not None:
        combat_branch = _copy_combat_shell_for_teacher_step(source.combat_env, payload)
    else:
        combat_branch = payload
    try:
        setattr(combat_branch, "_teacher_fast_combat_sync", False)
        engine = getattr(combat_branch, "engine", None)
        if engine is not None:
            setattr(engine, "_teacher_fast_step_refresh", False)
    except Exception:
        pass
    if source.run_env is None:
        combat_branch.step(dict(action))
        return combat_branch
    if run_shell_blob is not None:
        branch = pickle.loads(run_shell_blob)
        branch.combat = combat_branch
        branch.player = getattr(combat_branch, "player", getattr(branch, "player", None))
        branch.randoms = getattr(combat_branch, "randoms", getattr(branch, "randoms", None))
        combat_engine = getattr(combat_branch, "engine", None)
        if combat_engine is not None:
            branch.deck = list(getattr(combat_engine, "master_deck", getattr(branch, "deck", [])) or [])
            branch.relics = list(getattr(combat_engine, "relics", getattr(branch, "relics", [])) or [])
            branch.potions = list(getattr(combat_engine, "potions", getattr(branch, "potions", [])) or [])
            branch.gold = int(getattr(combat_engine, "gold", getattr(branch, "gold", 0)) or 0)
    else:
        branch = _copy_run_shell_for_teacher_step(source.run_env, combat_branch)
    try:
        setattr(branch, "_teacher_fast_combat_sync", False)
    except Exception:
        pass
    if getattr(branch, "combat", None) is not None and str(getattr(branch, "phase", "") or "") in {
        "COMBAT",
        "CARD_SELECT",
        "CARD_REWARD",
    }:
        branch._step_combat(dict(action))
    else:
        branch.step(dict(action))
    return branch


def _split_group_by_candidate_actions(
    eval_batch: Any,
    env: Any,
    candidate_ids: list[str],
    *,
    first_action: dict[str, Any] | None = None,
) -> dict[str, tuple[dict[str, Any], list[str]]]:
    chosen_by_key: dict[str, tuple[dict[str, Any], list[str]]] = {}
    start_index = 0
    if first_action is not None and candidate_ids:
        chosen_by_key[_group_action_key(first_action)] = (first_action, [candidate_ids[0]])
        start_index = 1
    for candidate_id in candidate_ids[start_index:]:
        candidate_action, _candidate_actions, _candidate_has_potion = _choose_group_action(
            eval_batch,
            env,
            candidate_id,
        )
        key = _group_action_key(candidate_action)
        if key not in chosen_by_key:
            chosen_by_key[key] = (candidate_action, [])
        chosen_by_key[key][1].append(candidate_id)
    return chosen_by_key


def _split_group_by_candidate_actions_batched(
    eval_batch: Any,
    env: Any,
    candidate_ids: list[str],
    *,
    legal_actions: list[dict[str, Any]] | None = None,
) -> dict[str, tuple[dict[str, Any], list[str]]] | None:
    if len(candidate_ids) <= 1 or not _FAST_BATCH_NON_POTION_ROOTS:
        return None
    if str(getattr(env, "phase", "")) != "COMBAT":
        return None
    combat_selector = (eval_batch._SELECTORS or {}).get("combat")
    if combat_selector is None:
        return None
    if legal_actions is None:
        legal_actions_env = getattr(combat_selector, "legal_actions_env", None)
        actions = legal_actions_env(env) if callable(legal_actions_env) else env.legal_actions()
    else:
        actions = legal_actions
    if len(actions) <= 1:
        return None
    try:
        from spirecomm.ai.v3_combat_teacher import best_teacher_actions_env_many_configs
    except Exception:
        return None

    chosen_by_key: dict[str, tuple[dict[str, Any], list[str]]] = {}
    policy_clusters = _candidate_policy_clusters(candidate_ids)
    config_groups: dict[tuple[Any, ...], list[str]] = {}
    representative_ids = [cluster[0] for cluster in policy_clusters]
    cluster_members_by_rep = {cluster[0]: list(cluster) for cluster in policy_clusters}
    for candidate_id in representative_ids:
        cfg = _FAST_TEACHER_CONFIGS[str(candidate_id)]
        group_key = (
            int(getattr(cfg, "beam_width", 0)),
            int(getattr(cfg, "node_budget_per_root", 0)),
            int(getattr(cfg, "max_depth", 0)),
            int(getattr(cfg, "continuation_action_cap", 0)),
            int(getattr(cfg, "lethal_check_node_budget", 0)),
        )
        config_groups.setdefault(group_key, []).append(str(candidate_id))

    for grouped_candidate_ids in config_groups.values():
        chosen_actions: list[dict[str, Any] | None] | None = None
        if len(grouped_candidate_ids) > 1:
            try:
                configs = [_FAST_TEACHER_CONFIGS[str(candidate_id)] for candidate_id in grouped_candidate_ids]
                chosen_actions = best_teacher_actions_env_many_configs(env, configs, legal_actions=actions)
            except Exception:
                chosen_actions = None
        if chosen_actions is None or len(chosen_actions) != len(grouped_candidate_ids):
            chosen_actions = []
            for candidate_id in grouped_candidate_ids:
                action, _candidate_actions, _candidate_has_potion = _choose_group_action(
                    eval_batch,
                    env,
                    candidate_id,
                    legal_actions=actions,
                )
                chosen_actions.append(action)
        for candidate_id, action in zip(grouped_candidate_ids, chosen_actions, strict=True):
            if action is None:
                return None
            key = _group_action_key(action)
            if key not in chosen_by_key:
                chosen_by_key[key] = (dict(action), [])
            chosen_by_key[key][1].extend(cluster_members_by_rep.get(str(candidate_id), [str(candidate_id)]))
    return chosen_by_key


def _candidate_policy_clusters(candidate_ids: list[str]) -> list[list[str]]:
    cluster_size = max(1, int(os.environ.get("SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_SIZE", "1") or 1))
    min_candidates = max(1, int(os.environ.get("SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_MIN_CANDIDATES", "999999") or 999999))
    if cluster_size <= 1 or len(candidate_ids) < min_candidates:
        return [[str(candidate_id)] for candidate_id in candidate_ids]
    default_ids = [str(candidate_id) for candidate_id in candidate_ids if str(candidate_id).endswith("default")]
    non_default_ids = [str(candidate_id) for candidate_id in candidate_ids if str(candidate_id) not in set(default_ids)]

    def sort_key(candidate_id: str) -> tuple[Any, ...]:
        params = _FAST_CANDIDATE_PARAMS.get(str(candidate_id), {})
        return tuple((str(key), round(float(value), 6)) for key, value in sorted(params.items()))

    ordered = sorted(non_default_ids, key=sort_key)
    clusters = [[candidate_id] for candidate_id in default_ids]
    for index in range(0, len(ordered), cluster_size):
        clusters.append(ordered[index : index + cluster_size])
    return [cluster for cluster in clusters if cluster]


def _stepped_split_groups(
    env: Any,
    steps: int,
    chosen_by_key: dict[str, tuple[dict[str, Any], list[str]]],
) -> list[tuple[list[str], Any, int]]:
    split_items = sorted(chosen_by_key.values(), key=lambda item: len(item[1]), reverse=True)
    if len(split_items) == 1:
        only_action, only_candidate_ids = split_items[0]
        _step_group_env(env, only_action)
        return [(only_candidate_ids, env, steps + 1)]
    merge_after_step = _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_MERGE_SPLIT_STATES", False)
    merged_by_state: dict[bytes, tuple[list[str], Any]] = {}
    stepped: list[tuple[list[str], Any, int]] = []

    def append_stepped(split_env: Any, split_candidate_ids: list[str]) -> None:
        if not merge_after_step:
            stepped.append((split_candidate_ids, split_env, steps + 1))
            return
        try:
            branch_blob = pickle.dumps(split_env, protocol=pickle.HIGHEST_PROTOCOL)
            digest = hashlib.blake2b(branch_blob, digest_size=20).digest()
        except Exception:
            stepped.append((split_candidate_ids, split_env, steps + 1))
            return
        previous = merged_by_state.get(digest)
        if previous is None:
            merged_by_state[digest] = (list(split_candidate_ids), split_env)
        else:
            previous[0].extend(split_candidate_ids)

    if (
        str(getattr(env, "phase", "")) == "COMBAT"
        and _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_FAST_COMBAT_SPLIT_STEP", False)
    ):
        try:
            from spirecomm.ai.v3_combat_teacher import _clone_step_source_or_none

            split_source = _clone_step_source_or_none(env)
            if split_source is not None:
                split_run_shell_blob = (
                    _run_shell_blob_without_combat(env)
                    if getattr(split_source, "run_env", None) is not None
                    else None
                )
                for split_action, split_candidate_ids in split_items:
                    split_env = _step_group_env_from_combat_source_exact(
                        env,
                        split_source,
                        split_action,
                        run_shell_blob=split_run_shell_blob,
                    )
                    append_stepped(split_env, split_candidate_ids)
                if merge_after_step:
                    for merged_candidate_ids, merged_env in merged_by_state.values():
                        stepped.append((merged_candidate_ids, merged_env, steps + 1))
                return stepped
        except Exception:
            pass

    env_blob = pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
    first = True
    for split_action, split_candidate_ids in split_items:
        split_env = env if first else pickle.loads(env_blob)
        first = False
        _step_group_env(split_env, split_action)
        append_stepped(split_env, split_candidate_ids)
    if merge_after_step:
        for merged_candidate_ids, merged_env in merged_by_state.values():
            stepped.append((merged_candidate_ids, merged_env, steps + 1))
    return stepped


def _append_split_groups(
    groups: list[tuple[list[str], Any, int]],
    env: Any,
    steps: int,
    chosen_by_key: dict[str, tuple[dict[str, Any], list[str]]],
) -> None:
    groups.extend(_stepped_split_groups(env, steps, chosen_by_key))


def _card_select_may_depend_on_candidate_config(env: Any) -> bool:
    if not _fast_bool_env("SPIRECOMM_TRUE_GRIT_TARGET_COMBAT_SEARCH", False):
        return False
    current_card_select = getattr(env, "current_card_select", None) or {}
    mode = str(current_card_select.get("mode") or "")
    if not mode:
        actions = env.legal_actions()
        if actions:
            mode = str(actions[0].get("mode") or "")
    if not mode:
        engine = getattr(getattr(env, "combat", None), "engine", None)
        pending = getattr(engine, "pending_card_select", None) or {}
        mode = str(pending.get("mode") or "")
    return mode.strip().lower() == "true_grit"


def _run_same_seed_candidate_batch(
    eval_batch: Any,
    tasks: list[tuple[str, int]],
) -> list[tuple[str, dict[str, Any]]]:
    results, _branches = _run_same_seed_candidate_batch_with_branches(
        eval_batch,
        tasks,
        offload_splits=False,
    )
    return results


def _run_same_seed_branch_task(
    eval_batch: Any,
    task: dict[str, Any],
    *,
    offload_splits: bool,
) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    seed = int(task["seed"])
    candidate_ids = [str(candidate_id) for candidate_id in task.get("candidate_ids", [])]
    env = pickle.loads(task["env_blob"])
    return _run_same_seed_candidate_group_from_env(
        eval_batch,
        seed=seed,
        candidate_ids=candidate_ids,
        env=env,
        prefix_steps=int(task.get("steps") or 0),
        offload_splits=offload_splits,
    )


def _run_same_seed_candidate_batch_with_branches(
    eval_batch: Any,
    tasks: list[tuple[str, int]],
    *,
    offload_splits: bool,
) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    seed = int(tasks[0][1])
    env, prefix_steps, terminal_result = _snapshot_env_for_seed(eval_batch, seed)
    if terminal_result is not None:
        result = {
            key: terminal_result[key]
            for key in (
                "seed",
                "ascension",
                "phase",
                "floor",
                "hp",
                "max_hp",
                "gold",
                "steps",
                "won",
                "dead",
                "timed_out",
                "max_floor_stopped",
                "error",
            )
            if key in terminal_result
        }
        return [(str(candidate_id), dict(result)) for candidate_id, _seed in tasks], []
    if env is None:
        raise RuntimeError("same_seed_batch_missing_env")
    return _run_same_seed_candidate_group_from_env(
        eval_batch,
        seed=seed,
        candidate_ids=[str(candidate_id) for candidate_id, _seed in tasks],
        env=env,
        prefix_steps=int(prefix_steps),
        offload_splits=offload_splits,
    )


def _run_same_seed_candidate_group_from_env(
    eval_batch: Any,
    *,
    seed: int,
    candidate_ids: list[str],
    env: Any,
    prefix_steps: int,
    offload_splits: bool,
) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    terminal_phases = {"GAME_OVER", "COMPLETE", "VICTORY"}
    max_steps = int(_FAST_BASE_CONFIG.get("max_steps") or 1500)
    max_floor = int(_FAST_BASE_CONFIG.get("max_floor") or 60)
    results: dict[str, dict[str, Any]] = {}
    branches: list[dict[str, Any]] = []
    groups: list[tuple[list[str], Any, int]] = [(list(candidate_ids), env, int(prefix_steps))]
    work_units_pending = 0
    work_progress_seconds = max(
        0.25,
        float(os.environ.get("SPIRECOMM_TEACHER_SWEEP_SAME_SEED_PROGRESS_SECONDS", "2") or "2"),
    )
    last_work_progress = time.perf_counter()

    def flush_work_progress(*, force: bool = False) -> None:
        nonlocal work_units_pending, last_work_progress
        if work_units_pending <= 0:
            return
        now = time.perf_counter()
        if not force and (now - last_work_progress) < work_progress_seconds:
            return
        _worker_live_progress_work(path="same_seed_step", seed=seed, units=work_units_pending)
        work_units_pending = 0
        last_work_progress = now

    def handle_split(
        split_env: Any,
        split_steps: int,
        chosen_by_key: dict[str, tuple[dict[str, Any], list[str]]],
    ) -> None:
        split_groups = _stepped_split_groups(split_env, split_steps, chosen_by_key)
        if offload_splits and len(split_groups) > 1:
            keep_local = _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_OFFLOAD_KEEP_LOCAL", False)
            singleton_splits = _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_OFFLOAD_SINGLETON_SPLITS", False)
            offload_groups = split_groups[1:] if keep_local else split_groups
            for split_candidate_ids, branch_env, branch_steps in offload_groups:
                if singleton_splits and len(split_candidate_ids) > 1:
                    branch_env_blob = pickle.dumps(branch_env, protocol=pickle.HIGHEST_PROTOCOL)
                    for split_candidate_id in split_candidate_ids:
                        branch = _serialize_same_seed_branch(
                            seed,
                            [str(split_candidate_id)],
                            branch_env,
                            branch_steps,
                            env_blob=branch_env_blob,
                        )
                        if not _worker_live_branch(branch):
                            branches.append(branch)
                    continue
                branch = _serialize_same_seed_branch(seed, split_candidate_ids, branch_env, branch_steps)
                if not _worker_live_branch(branch):
                    branches.append(branch)
            if keep_local:
                groups.append(split_groups[0])
            return
        groups.extend(split_groups)

    while groups:
        candidate_ids, group_env, steps = groups.pop()
        phase = str(getattr(group_env, "phase", ""))
        if phase in terminal_phases or int(getattr(group_env, "floor", 0) or 0) > max_floor or steps >= max_steps:
            result = _compact_rollout_result(group_env, seed=seed, steps=steps)
            for candidate_id in candidate_ids:
                results[candidate_id] = dict(result)
                _worker_live_progress_result(
                    path="same_seed_stream",
                    candidate_id=str(candidate_id),
                    seed=seed,
                    result=result,
                )
            continue
        work_units_pending += len(candidate_ids)
        flush_work_progress()

        representative_id = candidate_ids[0]
        can_batch_combat = bool(
            phase == "COMBAT"
            and len(candidate_ids) > 1
            and _FAST_BATCH_NON_POTION_ROOTS
        )
        must_split_combat = False
        actions = None
        has_potion = False
        if can_batch_combat:
            combat_selector = (eval_batch._SELECTORS or {}).get("combat")
            legal_actions_env = getattr(combat_selector, "legal_actions_env", None)
            actions = legal_actions_env(group_env) if callable(legal_actions_env) else group_env.legal_actions()
            has_potion = any((item.get("kind") or "") == "potion" for item in actions)
            must_split_combat = bool(len(actions) > 1 and ((not _FAST_CANDIDATES_ARE_POTION_ONLY) or has_potion))
        if must_split_combat:
            batched_actions = _split_group_by_candidate_actions_batched(
                eval_batch,
                group_env,
                candidate_ids,
                legal_actions=actions,
            )
            if batched_actions is not None:
                handle_split(group_env, steps, batched_actions)
                continue
        action, actions, has_potion = _choose_group_action(
            eval_batch,
            group_env,
            representative_id,
            legal_actions=actions,
        )
        if (
            phase == "CARD_SELECT"
            and len(candidate_ids) > 1
            and _card_select_may_depend_on_candidate_config(group_env)
        ):
            handle_split(
                group_env,
                steps,
                _split_group_by_candidate_actions(
                    eval_batch,
                    group_env,
                    candidate_ids,
                    first_action=action,
                ),
            )
            continue
        split_all_combat = bool(_FAST_GROUP_ALL_COMBAT and not _FAST_CANDIDATES_ARE_POTION_ONLY)
        if (
            phase != "COMBAT"
            or ((not split_all_combat and not must_split_combat) and not has_potion)
            or len(candidate_ids) == 1
        ):
            _step_group_env(group_env, action)
            groups.append((candidate_ids, group_env, steps + 1))
            continue

        handle_split(
            group_env,
            steps,
            _split_group_by_candidate_actions(
                eval_batch,
                group_env,
                candidate_ids,
                first_action=action,
            ),
        )

    flush_work_progress(force=True)
    return [(str(candidate_id), results[str(candidate_id)]) for candidate_id in candidate_ids if str(candidate_id) in results], branches


def _cleanup_candidate_dir(eval_dir: Path) -> None:
    for name in ("results.jsonl", "results.json", "summary.json", "summary_partial.json", "config.json"):
        path = eval_dir / name
        if path.exists():
            path.unlink()


def _write_candidate_config(
    *,
    output_dir: Path,
    eval_dir: Path,
    candidate_id: str,
    params: dict[str, float],
    eval_round_name: str,
    leaderboard_round_name: str,
    target_count: int,
    search_overrides: dict[str, Any],
    seeds: list[int],
    start_snapshot_dir: Path | None,
    result_storage: str,
) -> None:
    config_path = output_dir / "configs" / f"{candidate_id}.json"
    _write_json_fast(config_path, _candidate_payload(candidate_id, params, leaderboard_round_name))
    _write_json_fast(
        eval_dir / "config.json",
        {
            "candidate_id": candidate_id,
            "eval_round": eval_round_name,
            "leaderboard_round": leaderboard_round_name,
            "seed_count": target_count,
            "seeds": seeds,
            "params": params,
            "search_overrides": search_overrides,
            "start_snapshot_dir": str(start_snapshot_dir or ""),
            "scheduler": "task_pool",
            "result_storage": str(result_storage),
        },
    )


def _round_eval_dir(output_dir: Path, eval_round_name: str, candidate_id: str) -> Path:
    return output_dir / "evals" / eval_round_name / candidate_id


def _effective_task_batch_size(args: argparse.Namespace, search_overrides: dict[str, Any]) -> int:
    raw = int(args.task_batch_size)
    if raw > 0:
        return raw
    node_budget = search_overrides.get("node_budget_per_root")
    beam_width = search_overrides.get("beam_width")
    max_depth = search_overrides.get("max_depth")
    if node_budget is not None:
        node_budget_int = int(node_budget)
        if node_budget_int <= 4:
            return 16
        if node_budget_int <= 16:
            return 8
        if node_budget_int <= 64:
            return 4
        if node_budget_int <= 128:
            return 2
    if beam_width is not None and int(beam_width) <= 2:
        return 8
    if max_depth is not None and int(max_depth) <= 3:
        return 16
    if max_depth is not None and int(max_depth) <= 6:
        return 8
    return 1


def _bounded_task_batch_size(args: argparse.Namespace, search_overrides: dict[str, Any], pending_count: int) -> int:
    batch_size = _effective_task_batch_size(args, search_overrides)
    if batch_size <= 1 or pending_count <= 1:
        return 1
    target_batches = max(1, min(int(pending_count), max(1, int(args.workers)) * 2))
    max_batch_size = max(1, math.ceil(int(pending_count) / target_batches))
    return max(1, min(batch_size, max_batch_size))


def _seed_major_candidate_chunk_size(
    args: argparse.Namespace,
    search_overrides: dict[str, Any],
    *,
    candidate_count: int,
    seed_count: int,
    prefer_same_seed_candidate_groups: bool = False,
) -> int:
    raw = int(args.seed_major_candidate_chunk_size)
    if raw > 0:
        return max(1, min(raw, max(1, int(candidate_count))))
    node_budget = search_overrides.get("node_budget_per_root")
    max_depth = search_overrides.get("max_depth")
    if prefer_same_seed_candidate_groups:
        chunk_size = int(candidate_count)
    elif node_budget is not None:
        node_budget_int = int(node_budget)
        if node_budget_int <= 4:
            chunk_size = 32
        elif node_budget_int <= 16:
            chunk_size = 16
        elif node_budget_int <= 64:
            chunk_size = 8
        elif node_budget_int <= 128:
            chunk_size = 4
        else:
            chunk_size = 2
    elif max_depth is not None and int(max_depth) <= 6:
        chunk_size = 8
    else:
        # Full-budget sweeps are dominated by per-run teacher search, not
        # parent scheduling. Keep batches single-candidate for better tail
        # balance across workers.
        chunk_size = 1
    chunk_size = max(1, min(int(candidate_count), chunk_size))
    # In same-seed grouped mode, larger candidate chunks increase state sharing
    # inside each worker. Keep at least one wave of work for CPU saturation, but
    # do not force the usual 2x-worker tail-balancing split that fragments
    # candidate groups and repeats teacher search at identical states.
    if prefer_same_seed_candidate_groups and _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_FULL_SEED_GROUPS", True):
        try:
            min_full_group_ratio = float(os.environ.get("SPIRECOMM_TEACHER_SWEEP_FULL_GROUP_MIN_WORKER_RATIO", "0.75"))
        except (TypeError, ValueError):
            min_full_group_ratio = 0.75
        min_full_group_batches = max(1, int(math.ceil(max(1, int(args.workers)) * max(0.0, min_full_group_ratio))))
        if int(seed_count) >= min_full_group_batches:
            return chunk_size
    min_batches = max(1, int(args.workers) if prefer_same_seed_candidate_groups else int(args.workers) * 2)
    if seed_count * math.ceil(max(1, int(candidate_count)) / chunk_size) < min_batches:
        max_chunk_for_parallelism = max(1, math.ceil(max(1, int(candidate_count)) * max(1, int(seed_count)) / min_batches))
        chunk_size = max(1, min(chunk_size, max_chunk_for_parallelism))
    return max(1, min(chunk_size, max(1, int(candidate_count))))


def _build_pending_batches(
    *,
    args: argparse.Namespace,
    pending: list[tuple[str, int]],
    search_overrides: dict[str, Any],
    candidate_ids: list[str],
    seeds: list[int],
    prefer_same_seed_candidate_groups: bool = False,
) -> tuple[list[list[tuple[str, int]]], str, int]:
    schedule_order = str(args.schedule_order).strip().lower().replace("_", "-")
    if schedule_order != "seed-major":
        task_batch_size = _bounded_task_batch_size(args, search_overrides, len(pending))
        return (
            [pending[index : index + task_batch_size] for index in range(0, len(pending), task_batch_size)],
            "candidate-major",
            task_batch_size,
        )

    pending_by_seed_candidate: dict[tuple[int, str], tuple[str, int]] = {
        (int(seed), str(candidate_id)): (str(candidate_id), int(seed))
        for candidate_id, seed in pending
    }
    candidate_chunk_size = _seed_major_candidate_chunk_size(
        args,
        search_overrides,
        candidate_count=len(candidate_ids),
        seed_count=len(seeds),
        prefer_same_seed_candidate_groups=prefer_same_seed_candidate_groups,
    )
    if prefer_same_seed_candidate_groups:
        try:
            min_batches_per_worker = float(os.environ.get("SPIRECOMM_TEACHER_SWEEP_MIN_BATCHES_PER_WORKER", "0"))
        except (TypeError, ValueError):
            min_batches_per_worker = 1.5
        if min_batches_per_worker > 0.0 and pending:
            target_batches = max(1, int(math.ceil(max(1, int(args.workers)) * min_batches_per_worker)))
            active_seed_count = max(1, sum(
                1
                for seed in seeds
                if any((int(seed), str(candidate_id)) in pending_by_seed_candidate for candidate_id in candidate_ids)
            ))
            chunks_per_seed = max(1, int(math.ceil(target_batches / active_seed_count)))
            max_pending_candidate_count = max(
                1,
                max(
                    sum(1 for candidate_id in candidate_ids if (int(seed), str(candidate_id)) in pending_by_seed_candidate)
                    for seed in seeds
                ),
            )
            balanced_chunk_size = max(1, int(math.ceil(max_pending_candidate_count / chunks_per_seed)))
            candidate_chunk_size = max(1, min(candidate_chunk_size, balanced_chunk_size))
        try:
            max_seed_group_candidates = int(os.environ.get("SPIRECOMM_TEACHER_SWEEP_MAX_SEED_GROUP_CANDIDATES", "0"))
        except (TypeError, ValueError):
            max_seed_group_candidates = 24
        if max_seed_group_candidates > 0 and len(candidate_ids) > max_seed_group_candidates:
            candidate_chunk_size = max(1, min(candidate_chunk_size, max_seed_group_candidates))
    batches: list[list[tuple[str, int]]] = []
    for seed in seeds:
        seed_tasks = [
            pending_by_seed_candidate[(int(seed), str(candidate_id))]
            for candidate_id in candidate_ids
            if (int(seed), str(candidate_id)) in pending_by_seed_candidate
        ]
        for index in range(0, len(seed_tasks), candidate_chunk_size):
            batch = seed_tasks[index : index + candidate_chunk_size]
            if batch:
                batches.append(batch)
    return batches, "seed-major", candidate_chunk_size


def _effective_max_in_flight(
    args: argparse.Namespace,
    batch_count: int,
    *,
    worker_count: int | None = None,
) -> int:
    if batch_count <= 0:
        return 0
    raw = int(args.max_pending_futures)
    if raw < 0:
        return int(batch_count)
    if raw > 0:
        return min(int(batch_count), max(1, raw))
    workers = max(1, int(worker_count if worker_count is not None else args.workers))
    # Keep enough queued work to saturate workers without materializing very
    # large sweeps as thousands of Future objects in the parent process.
    auto_limit = max(64, workers * 8, workers + 16)
    return min(int(batch_count), auto_limit)


def _is_worker_pool_failure(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenProcessPool, BrokenPipeError, EOFError, ConnectionResetError)):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "terminated abruptly",
            "process pool",
            "broken pipe",
            "connection reset",
            "connection refused",
            "worker exited",
            "pool not running",
        )
    )


def _scaled_retry_workers(worker_count: int, scale: float) -> int:
    worker_count = max(1, int(worker_count))
    if worker_count <= 1:
        return 1
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        scale = 1.0
    next_workers = max(1, int(math.ceil(worker_count * float(scale))))
    if float(scale) < 1.0 and next_workers >= worker_count:
        next_workers = worker_count - 1
    return max(1, next_workers)


def _candidate_params_are_potion_only(candidates: list[tuple[str, dict[str, float]]]) -> bool:
    return bool(candidates) and all(
        set(params).issubset(_POTION_SWEEP_PARAM_NAMES)
        for _candidate_id, params in candidates
    )


def _run_candidate_stage(
    *,
    args: argparse.Namespace,
    eval_round_name: str,
    leaderboard_round_name: str,
    candidates: list[tuple[str, dict[str, float]]],
    target_count: int,
    search_overrides: dict[str, Any] | None = None,
    max_floor_override: int | None = None,
) -> list[dict[str, Any]]:
    search_overrides = dict(search_overrides or {})
    use_aggregate_results = str(args.result_storage) == "round-aggregate"
    start_snapshot_dir = _prepare_first_combat_snapshots(args, int(target_count))
    base_config = _base_eval_config(args)
    if max_floor_override is not None and int(max_floor_override) > 0:
        base_config["max_floor"] = int(max_floor_override)
    if start_snapshot_dir is not None:
        base_config["start_snapshot_dir"] = str(start_snapshot_dir)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(target_count)))
    seed_set = set(seeds)
    round_eval_root = args.output_dir / "evals" / eval_round_name
    aggregate_results_path = round_eval_root / "results.jsonl"
    if use_aggregate_results and args.force:
        aggregate_key = str(aggregate_results_path.resolve())
        if aggregate_key not in _CLEARED_AGGREGATE_RESULT_FILES:
            if aggregate_results_path.exists():
                aggregate_results_path.unlink()
            _CLEARED_AGGREGATE_RESULT_FILES.add(aggregate_key)
    allow_existing_results = bool(args.resume or (args.force and args.reuse_stage_prefix_results))
    aggregate_existing = (
        _load_round_aggregate_results(aggregate_results_path, seed_set)
        if use_aggregate_results and allow_existing_results
        else {}
    )
    started = time.time()
    states: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, int]] = []
    candidate_params_by_id = {candidate_id: dict(params) for candidate_id, params in candidates}
    write_buffers: dict[str, list[dict[str, Any]]] = {}
    aggregate_write_buffer: list[dict[str, Any]] = []

    for candidate_id, params in candidates:
        eval_dir = _round_eval_dir(args.output_dir, eval_round_name, candidate_id)
        if (
            bool(args.write_candidate_configs)
            or bool(args.write_candidate_summaries)
            or not use_aggregate_results
        ):
            eval_dir.mkdir(parents=True, exist_ok=True)
        if args.force:
            candidate_dir_key = str(eval_dir.resolve())
            if candidate_dir_key not in _CLEARED_CANDIDATE_DIRS:
                _cleanup_candidate_dir(eval_dir)
                _CLEARED_CANDIDATE_DIRS.add(candidate_dir_key)
        complete_summary = _load_complete_summary(eval_dir, len(seeds)) if allow_existing_results and eval_dir.exists() else None
        if bool(args.write_candidate_configs) and (
            complete_summary is None or not (eval_dir / "config.json").exists()
        ):
            _write_candidate_config(
                output_dir=args.output_dir,
                eval_dir=eval_dir,
                candidate_id=candidate_id,
                params=params,
                eval_round_name=eval_round_name,
                leaderboard_round_name=leaderboard_round_name,
                target_count=target_count,
                search_overrides=search_overrides,
                seeds=seeds,
                start_snapshot_dir=start_snapshot_dir,
                result_storage=str(args.result_storage),
            )
        if complete_summary is not None:
            existing = {}
        elif use_aggregate_results:
            existing = dict(aggregate_existing.get(candidate_id, {}))
            if allow_existing_results:
                # Backward compatibility with older per-candidate result files.
                # Merge rather than fallback-only: a resumed optimized run may
                # contain old per-candidate rows plus newer round-aggregate rows.
                legacy_existing = _load_existing_results(eval_dir, seed_set)
                if legacy_existing:
                    existing = {**legacy_existing, **existing}
        else:
            existing = _load_existing_results(eval_dir, seed_set) if allow_existing_results else {}
        results = {seed: existing[seed] for seed in seeds if seed in existing}
        states[candidate_id] = {
            "params": dict(params),
            "eval_dir": eval_dir,
            "results": results,
            "summary": complete_summary,
            "completed_count": len(seeds) if complete_summary is not None else len(results),
        }
        write_buffers[candidate_id] = []
        if complete_summary is None:
            for seed in seeds:
                if seed not in results:
                    pending.append((candidate_id, seed))

    complete_existing = sum(int(state.get("completed_count") or 0) for state in states.values())
    total = len(candidates) * len(seeds)
    print(
        f"[{leaderboard_round_name}] task_pool candidates={len(candidates)} seeds={len(seeds)} "
        f"resume={complete_existing}/{total} pending={len(pending)} workers={args.workers} overrides={search_overrides}",
        flush=True,
    )

    completed_new = 0
    def flush_aggregate_result_buffer() -> None:
        if not aggregate_write_buffer:
            return
        _append_aggregate_results_jsonl(aggregate_results_path, aggregate_write_buffer)
        aggregate_write_buffer.clear()

    def flush_result_buffer(candidate_id: str) -> None:
        if use_aggregate_results:
            return
        buffer = write_buffers.get(candidate_id)
        if not buffer:
            return
        state = states[candidate_id]
        _append_results_jsonl(Path(state["eval_dir"]) / "results.jsonl", buffer)
        buffer.clear()

    def flush_all_result_buffers() -> None:
        flush_aggregate_result_buffer()
        for candidate_id in list(write_buffers):
            flush_result_buffer(candidate_id)

    if pending:
        context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        if bool(args.preload_selectors_before_fork):
            base_config = _preload_selectors_for_fork(
                base_config,
                preload_start_snapshots=bool(args.preload_start_snapshots_before_fork),
                seeds=seeds,
            )
        result_flush_interval = max(1, int(args.result_flush_interval))
        fine_progress_enabled = _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_FINE_PROGRESS", True)
        try:
            progress_heartbeat_seconds = float(
                os.environ.get(
                    "SPIRECOMM_TEACHER_SWEEP_PROGRESS_HEARTBEAT_SECONDS",
                    "2" if fine_progress_enabled else "60",
                )
            )
        except (TypeError, ValueError):
            progress_heartbeat_seconds = 2.0 if fine_progress_enabled else 60.0
        progress_heartbeat_seconds = max(0.5 if fine_progress_enabled else 5.0, progress_heartbeat_seconds)
        candidates_are_potion_only = _candidate_params_are_potion_only(candidates)
        batch_min_candidates = int(os.environ.get("SPIRECOMM_TEACHER_SWEEP_BATCH_MIN_CANDIDATES", "2"))
        prefer_same_seed_candidate_groups = _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_GROUP_SAME_SEED", True) and (
            candidates_are_potion_only
            or _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_GROUP_ALL_COMBAT", False)
            or (
                _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_BATCH_NON_POTION_ROOTS", False)
                and len(candidates) >= max(2, batch_min_candidates)
            )
        )
        last_progress_completed = 0
        progress_batches_done = 0
        progress_batches_total = 0
        progress_worker_window_rate = 0.0
        progress_est_full_rate = 0.0
        speed_window: deque[tuple[int, float]] = deque(
            maxlen=max(1, int(os.environ.get("SPIRECOMM_TEACHER_SWEEP_SPEED_WINDOW_BATCHES", "20")))
        )
        batch_speed_log = _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_BATCH_SPEED_LOG", True)
        fine_progress_manager = mp.Manager() if fine_progress_enabled else None
        fine_progress_queue = (
            fine_progress_manager.Queue(maxsize=max(1000, int(os.environ.get("SPIRECOMM_TEACHER_SWEEP_FINE_PROGRESS_QUEUE_SIZE", "100000"))))
            if fine_progress_manager is not None
            else None
        )
        dynamic_branch_queue: deque[dict[str, Any]] | None = None
        dynamic_branch_batches_added = 0
        try:
            fine_progress_print_seconds = float(os.environ.get("SPIRECOMM_TEACHER_SWEEP_FINE_PROGRESS_PRINT_SECONDS", "2"))
        except (TypeError, ValueError):
            fine_progress_print_seconds = 2.0
        try:
            fine_progress_window_seconds = float(os.environ.get("SPIRECOMM_TEACHER_SWEEP_FINE_PROGRESS_WINDOW_SECONDS", "120"))
        except (TypeError, ValueError):
            fine_progress_window_seconds = 120.0
        fine_progress_print_seconds = max(0.5, fine_progress_print_seconds)
        fine_progress_window_seconds = max(5.0, fine_progress_window_seconds)
        live_total_tasks = 0
        live_total_work_units = 0
        live_last_print = 0.0
        live_window: deque[
            tuple[float, int, int, float, dict[str, int], dict[str, int]]
        ] = deque()

        def drain_fine_progress(*, force: bool = False) -> None:
            nonlocal live_total_tasks, live_total_work_units, live_last_print
            nonlocal dynamic_branch_batches_added
            if fine_progress_queue is None:
                return
            now = time.time()
            drained = 0
            drained_tasks = 0
            drained_work_units = 0
            elapsed_sum = 0.0
            paths: dict[str, int] = {}
            work_paths: dict[str, int] = {}
            streamed_results: list[tuple[str, dict[str, Any]]] = []
            while True:
                try:
                    item = fine_progress_queue.get_nowait()
                except Empty:
                    break
                except Exception:
                    break
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or "")
                if kind == "result_done":
                    result = item.get("result")
                    candidate_id = str(item.get("candidate_id") or "")
                    if candidate_id and isinstance(result, dict) and "seed" in result:
                        drained += 1
                        drained_tasks += 1
                        elapsed_sum += float(item.get("elapsed") or 0.0)
                        path = str(item.get("path") or "?")
                        paths[path] = paths.get(path, 0) + 1
                        streamed_results.append((candidate_id, dict(result)))
                    continue
                if kind == "task_done":
                    drained += 1
                    drained_tasks += 1
                    elapsed_sum += float(item.get("elapsed") or 0.0)
                    path = str(item.get("path") or "?")
                    paths[path] = paths.get(path, 0) + 1
                    continue
                if kind == "work_done":
                    units = int(item.get("units") or 0)
                    if units <= 0:
                        continue
                    drained += 1
                    drained_work_units += units
                    path = str(item.get("path") or "?")
                    work_paths[path] = work_paths.get(path, 0) + units
                    continue
                if kind == "branch_task":
                    branch = item.get("branch")
                    if dynamic_branch_queue is not None and isinstance(branch, dict):
                        dynamic_branch_queue.append(branch)
                        dynamic_branch_batches_added += 1
                        drained += 1
                    continue
                continue
            if streamed_results:
                consume_batch_results(streamed_results)
                if _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_STREAM_FLUSH_RESULTS", True):
                    flush_all_result_buffers()
            if drained_tasks > 0 or drained_work_units > 0:
                live_total_tasks += drained_tasks
                live_total_work_units += drained_work_units
                live_window.append((now, drained_tasks, drained_work_units, elapsed_sum, paths, work_paths))
            cutoff = now - fine_progress_window_seconds
            while live_window and live_window[0][0] < cutoff:
                live_window.popleft()
            should_print = force or drained > 0
            should_print = should_print and (now - live_last_print >= fine_progress_print_seconds)
            if not should_print:
                return
            window_count = sum(item[1] for item in live_window)
            window_work_units = sum(item[2] for item in live_window)
            window_elapsed = sum(item[3] for item in live_window)
            window_paths: dict[str, int] = {}
            window_work_paths: dict[str, int] = {}
            for _ts, _count, _work_units, _elapsed, item_paths, item_work_paths in live_window:
                for path, count in item_paths.items():
                    window_paths[path] = window_paths.get(path, 0) + int(count)
                for path, count in item_work_paths.items():
                    window_work_paths[path] = window_work_paths.get(path, 0) + int(count)
            window_age = (
                max(fine_progress_print_seconds, min(fine_progress_window_seconds, now - live_window[0][0]))
                if live_window
                else 0.0
            )
            live_window_rate = float(window_count) / window_age if window_age > 1e-9 else 0.0
            live_work_window_rate = float(window_work_units) / window_age if window_age > 1e-9 else 0.0
            live_total_rate = float(live_total_tasks) / max(1e-6, now - started)
            live_work_total_rate = float(live_total_work_units) / max(1e-6, now - started)
            avg_task_seconds = float(window_elapsed) / float(window_count) if window_count > 0 else 0.0
            paths_text = ",".join(f"{key}:{value}" for key, value in sorted(window_paths.items())) or "-"
            work_paths_text = ",".join(f"{key}:{value}" for key, value in sorted(window_work_paths.items())) or "-"
            print(
                f"[{leaderboard_round_name}] live_speed "
                f"live_tasks={live_total_tasks} live_work_units={live_total_work_units} "
                f"drained={drained} drained_tasks={drained_tasks} drained_work_units={drained_work_units} "
                f"live_rate={live_total_rate:.3f}/s "
                f"live_work_rate={live_work_total_rate:.3f}/s "
                f"live_window_rate={live_window_rate:.3f}/s "
                f"live_work_window_rate={live_work_window_rate:.3f}/s "
                f"avg_task={avg_task_seconds:.3f}s "
                f"paths={paths_text} work_paths={work_paths_text}",
                flush=True,
            )
            live_last_print = now

        def maybe_print_progress(*, force: bool = False) -> None:
            nonlocal last_progress_completed
            progress_interval = max(1, int(args.progress_interval_tasks))
            should_print = force or completed_new == 1 or completed_new == len(pending)
            should_print = should_print or (completed_new - last_progress_completed >= progress_interval)
            if not should_print:
                return
            elapsed = max(1e-6, time.time() - started)
            done_total = complete_existing + completed_new
            rate = completed_new / elapsed
            remaining = max(0, total - done_total)
            eta_seconds = remaining / rate if rate > 1e-9 else 0.0
            print(
                f"[{leaderboard_round_name}] completed_tasks={done_total}/{total} "
                f"new_rate={rate:.3f}/s "
                f"batch_worker_rate={progress_worker_window_rate:.3f}/s "
                f"batch_est_full_rate={progress_est_full_rate:.3f}/s "
                f"batches={progress_batches_done}/{progress_batches_total} "
                f"eta={eta_seconds / 60.0:.1f}m",
                flush=True,
            )
            last_progress_completed = completed_new

        def unpack_batch_payload(payload: Any) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
            if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                meta = payload.get("meta")
                return payload["results"], dict(meta) if isinstance(meta, dict) else {}
            return payload, {}

        def record_batch_speed(
            *,
            batch_meta: dict[str, Any],
            batch_new: int,
            completed_batches: int,
            total_batches: int,
            worker_count_current: int,
        ) -> None:
            nonlocal progress_batches_done, progress_batches_total
            nonlocal progress_worker_window_rate, progress_est_full_rate
            progress_batches_done = completed_batches
            progress_batches_total = total_batches
            worker_elapsed = float(batch_meta.get("worker_elapsed") or 0.0)
            task_count = int(batch_meta.get("task_count") or len(batch_meta.get("results") or []) or batch_new)
            task_count = max(0, task_count)
            if worker_elapsed > 0.0 and task_count > 0:
                speed_window.append((task_count, worker_elapsed))
            window_tasks = sum(item[0] for item in speed_window)
            window_worker_seconds = sum(item[1] for item in speed_window)
            progress_worker_window_rate = (
                float(window_tasks) / window_worker_seconds if window_worker_seconds > 1e-9 else 0.0
            )
            progress_est_full_rate = progress_worker_window_rate * max(1, int(worker_count_current))
            if not batch_speed_log:
                return
            batch_worker_rate = float(task_count) / worker_elapsed if worker_elapsed > 1e-9 else 0.0
            print(
                f"[{leaderboard_round_name}] batch_speed "
                f"batch={completed_batches}/{total_batches} "
                f"tasks={task_count} new={batch_new} "
                f"worker_elapsed={worker_elapsed:.3f}s "
                f"worker_rate={batch_worker_rate:.3f}/s "
                f"window_worker_rate={progress_worker_window_rate:.3f}/s "
                f"est_full_rate={progress_est_full_rate:.3f}/s "
                f"path={batch_meta.get('path', '?')} "
                f"fallback_error={str(batch_meta.get('fallback_error') or '')[:160]} "
                f"seeds={batch_meta.get('seed_count', '?')} "
                f"candidates={batch_meta.get('candidate_count', '?')}",
                flush=True,
            )

        def consume_batch_results(
            batch_results: list[tuple[str, dict[str, Any]]],
            *,
            batch_meta: dict[str, Any] | None = None,
            completed_batches: int = 0,
            total_batches: int = 0,
            worker_count_current: int = 1,
        ) -> None:
            nonlocal completed_new
            before_completed = completed_new
            for candidate_id, result in batch_results:
                seed = int(result["seed"])
                state = states[candidate_id]
                was_new = False
                if seed not in state["results"]:
                    state["results"][seed] = result
                    state["completed_count"] = int(state.get("completed_count") or 0) + 1
                    was_new = True
                    if use_aggregate_results:
                        aggregate_write_buffer.append({"_candidate_id": candidate_id, **result})
                        if len(aggregate_write_buffer) >= result_flush_interval:
                            flush_aggregate_result_buffer()
                    else:
                        write_buffers[candidate_id].append(result)
                        if len(write_buffers[candidate_id]) >= result_flush_interval:
                            flush_result_buffer(candidate_id)
                if was_new:
                    completed_new += 1
                candidate_count = len(state["results"])
                summary_interval = int(args.summary_interval)
                write_first_partial = bool(args.write_first_partial_summary) and candidate_count == 1
                if write_first_partial or (summary_interval > 0 and candidate_count % summary_interval == 0):
                    flush_aggregate_result_buffer()
                    flush_result_buffer(candidate_id)
                    if bool(args.write_candidate_summaries):
                        partial = _summarize(
                            [state["results"][item] for item in seeds if item in state["results"]],
                            started=started,
                            output_dir=Path(state["eval_dir"]),
                        )
                        _write_json_fast(Path(state["eval_dir"]) / "summary_partial.json", partial)
                maybe_print_progress()
            if batch_meta is not None:
                record_batch_speed(
                    batch_meta=batch_meta,
                    batch_new=completed_new - before_completed,
                    completed_batches=completed_batches,
                    total_batches=total_batches,
                    worker_count_current=worker_count_current,
                )

        def remaining_pending_tasks() -> list[tuple[str, int]]:
            current: list[tuple[str, int]] = []
            for candidate_id, _params in candidates:
                state = states[candidate_id]
                if state.get("summary") is not None:
                    continue
                results = state["results"]
                for seed in seeds:
                    if seed not in results:
                        current.append((candidate_id, seed))
            return current

        attempt = 0
        worker_count = max(1, int(args.workers))
        max_retries = max(0, int(args.worker_crash_retries))
        retry_scale = float(args.worker_crash_retry_worker_scale)
        while True:
            current_pending = remaining_pending_tasks()
            if not current_pending:
                break
            attempt += 1
            pending_batches, schedule_order, task_batch_size = _build_pending_batches(
                args=args,
                pending=current_pending,
                search_overrides=search_overrides,
                candidate_ids=[candidate_id for candidate_id, _params in candidates],
                seeds=seeds,
                prefer_same_seed_candidate_groups=prefer_same_seed_candidate_groups,
            )
            max_in_flight = _effective_max_in_flight(
                args,
                len(pending_batches),
                worker_count=worker_count,
            )
            print(
                f"[{leaderboard_round_name}] scheduler backend={args.executor_backend} "
                f"attempt={attempt} workers={worker_count} max_in_flight={max_in_flight} "
                f"pending_tasks={len(current_pending)} batches={len(pending_batches)} "
                f"order={schedule_order} task_batch_size={task_batch_size} "
                f"result_flush_interval={result_flush_interval}",
                flush=True,
            )
            try:
                if str(args.executor_backend) == "mp-pool":
                    pool_context = context if context is not None else mp
                    with pool_context.Pool(
                        processes=worker_count,
                        initializer=_fast_worker_init,
                        initargs=(base_config, candidate_params_by_id, search_overrides, fine_progress_queue),
                    ) as pool:
                        result_iter = pool.imap_unordered(_run_seed_task_batch, pending_batches, chunksize=1)
                        completed_batches = 0
                        while completed_batches < len(pending_batches):
                            try:
                                batch_payload = result_iter.next(timeout=progress_heartbeat_seconds)
                            except mp.TimeoutError:
                                drain_fine_progress(force=True)
                                maybe_print_progress(force=True)
                                continue
                            completed_batches += 1
                            drain_fine_progress(force=True)
                            if isinstance(batch_payload, dict) and batch_payload.get("branches"):
                                raise RuntimeError("SPIRECOMM_TEACHER_SWEEP_OFFLOAD_SPLITS requires --executor-backend process-pool")
                            batch_results, batch_meta = unpack_batch_payload(batch_payload)
                            consume_batch_results(
                                batch_results,
                                batch_meta=batch_meta,
                                completed_batches=completed_batches,
                                total_batches=len(pending_batches),
                                worker_count_current=worker_count,
                            )
                else:
                    branch_offload_enabled = bool(
                        prefer_same_seed_candidate_groups
                        and _fast_bool_env("SPIRECOMM_TEACHER_SWEEP_OFFLOAD_SPLITS", False)
                    )
                    with ProcessPoolExecutor(
                        max_workers=worker_count,
                        initializer=_fast_worker_init,
                        initargs=(base_config, candidate_params_by_id, search_overrides, fine_progress_queue),
                        mp_context=context,
                    ) as executor:
                        pending_iter = iter(pending_batches)
                        branch_queue: deque[dict[str, Any]] = deque()
                        dynamic_branch_queue = branch_queue if branch_offload_enabled else None
                        futures: dict[Any, Any] = {}
                        completed_batches = 0
                        total_batches_dynamic = len(pending_batches)
                        if branch_offload_enabled:
                            try:
                                root_in_flight_limit = int(
                                    os.environ.get(
                                        "SPIRECOMM_TEACHER_SWEEP_OFFLOAD_ROOT_IN_FLIGHT",
                                        str(max(1, int(worker_count) // 3)),
                                    )
                                )
                            except (TypeError, ValueError):
                                root_in_flight_limit = max(1, int(worker_count) // 3)
                            root_in_flight_limit = max(1, min(int(worker_count), int(root_in_flight_limit)))
                        else:
                            root_in_flight_limit = int(worker_count)

                        def submit_until_capacity() -> None:
                            while len(futures) < max_in_flight:
                                if branch_queue:
                                    task_batch = branch_queue.pop()
                                else:
                                    if branch_offload_enabled:
                                        root_in_flight = sum(
                                            1 for value in futures.values() if not _is_branch_task_batch(value)
                                        )
                                        if root_in_flight >= root_in_flight_limit:
                                            return
                                    try:
                                        task_batch = next(pending_iter)
                                    except StopIteration:
                                        return
                                futures[executor.submit(_run_seed_task_batch, task_batch)] = task_batch

                        submit_until_capacity()
                        while futures or branch_queue:
                            submit_until_capacity()
                            if not futures:
                                continue
                            done, _ = wait(futures, timeout=progress_heartbeat_seconds, return_when=FIRST_COMPLETED)
                            if not done:
                                drain_fine_progress(force=True)
                                total_batches_dynamic = max(
                                    total_batches_dynamic,
                                    completed_batches + len(futures) + len(branch_queue),
                                )
                                maybe_print_progress(force=True)
                                continue
                            for future in done:
                                futures.pop(future)
                                drain_fine_progress(force=True)
                                total_batches_dynamic = max(
                                    total_batches_dynamic,
                                    completed_batches + len(futures) + len(branch_queue) + 1,
                                )
                                payload = future.result()
                                if branch_offload_enabled and isinstance(payload, dict):
                                    new_branches = [
                                        branch
                                        for branch in payload.get("branches", [])
                                        if isinstance(branch, dict)
                                    ]
                                    if new_branches:
                                        branch_queue.extend(new_branches)
                                        total_batches_dynamic += len(new_branches)
                                batch_results, batch_meta = unpack_batch_payload(payload)
                                completed_batches += 1
                                consume_batch_results(
                                    batch_results,
                                    batch_meta=batch_meta,
                                    completed_batches=completed_batches,
                                    total_batches=total_batches_dynamic,
                                    worker_count_current=worker_count,
                                )
                            submit_until_capacity()
                        dynamic_branch_queue = None
                if last_progress_completed != completed_new:
                    drain_fine_progress(force=True)
                    maybe_print_progress(force=True)
                break
            except Exception as exc:
                flush_all_result_buffers()
                remaining = len(remaining_pending_tasks())
                if not _is_worker_pool_failure(exc) or attempt > max_retries or remaining <= 0:
                    raise
                next_workers = _scaled_retry_workers(worker_count, retry_scale)
                print(
                    f"[{leaderboard_round_name}] worker_pool_failure retry={attempt}/{max_retries} "
                    f"error={type(exc).__name__}: {exc} remaining={remaining} "
                    f"workers {worker_count}->{next_workers}",
                    flush=True,
                )
                worker_count = next_workers
                time.sleep(min(30.0, max(1.0, 2.0 * attempt)))
            finally:
                flush_all_result_buffers()
        if fine_progress_manager is not None:
            try:
                fine_progress_manager.shutdown()
            except Exception:
                pass
    else:
        flush_all_result_buffers()

    results_by_candidate: list[dict[str, Any]] = []
    sweep_result_lines: list[str] = []
    for candidate_id, params in candidates:
        state = states[candidate_id]
        summary = state.get("summary")
        if not isinstance(summary, dict):
            ordered_results = [state["results"][seed] for seed in seeds if seed in state["results"]]
            summary = _summarize(ordered_results, started=started, output_dir=Path(state["eval_dir"]))
            if bool(args.write_candidate_summaries):
                summary_path = Path(state["eval_dir"]) / ("summary.json" if len(ordered_results) == len(seeds) else "summary_partial.json")
                _write_json_fast(summary_path, summary)
        result = {
            "round": leaderboard_round_name,
            "candidate_id": candidate_id,
            "seed_start": int(args.seed_start),
            "seed_count": int(target_count),
            "params": dict(params),
            "search_overrides": dict(search_overrides),
            "eval_dir": str(state["eval_dir"]),
            "config_path": str(args.output_dir / "configs" / f"{candidate_id}.json")
            if bool(args.write_candidate_configs)
            else "",
            "summary": summary,
        }
        results_by_candidate.append(result)
        sweep_result_lines.append(json.dumps(result, ensure_ascii=False) + "\n")
    if bool(args.write_sweep_results) and sweep_result_lines and (bool(pending) or not bool(args.resume)):
        with (args.output_dir / "sweep_results.jsonl").open("a", encoding="utf-8") as handle:
            handle.writelines(sweep_result_lines)

    ranked = _save_leaderboard_fast(args.output_dir, leaderboard_round_name, results_by_candidate)
    best = ranked[0] if ranked else None
    if best is not None:
        print(
            f"[{leaderboard_round_name}] best={best['candidate_id']} "
            f"mean_floor={float(best['summary'].get('mean_floor') or 0.0):.2f} "
            f"wins={int(best['summary'].get('win_count') or 0)}",
            flush=True,
        )
    return ranked


def _candidate_list_from_ranked(ranked: list[dict[str, Any]], limit: int) -> list[tuple[str, dict[str, float]]]:
    return [(item["candidate_id"], dict(item["params"])) for item in ranked[: max(0, int(limit))]]


def _load_ranked_leaderboard(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"cannot start from {label}: missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"cannot start from {label}: empty leaderboard {path}")
    for item in payload:
        if not isinstance(item, dict) or "candidate_id" not in item or "params" not in item:
            raise SystemExit(f"cannot start from {label}: invalid leaderboard row in {path}")
    return payload


def _load_ranked_leaderboard_first(paths: list[Path], *, label: str) -> list[dict[str, Any]]:
    for path in paths:
        if path.exists():
            return _load_ranked_leaderboard(path, label=label)
    joined = ", ".join(str(path) for path in paths)
    raise SystemExit(f"cannot start from {label}: missing all candidate leaderboards: {joined}")


def _round1_search_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return _search_overrides(args, "round1")


def _search_overrides(args: argparse.Namespace, prefix: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    beam_width = int(getattr(args, f"{prefix}_proxy_beam_width"))
    node_budget = int(getattr(args, f"{prefix}_proxy_node_budget"))
    max_depth = int(getattr(args, f"{prefix}_proxy_max_depth"))
    continuation_action_cap = int(getattr(args, f"{prefix}_proxy_continuation_action_cap", 0))
    root_only = bool(getattr(args, f"{prefix}_proxy_root_only", False))
    if beam_width > 0:
        overrides["beam_width"] = beam_width
    if node_budget > 0:
        overrides["node_budget_per_root"] = node_budget
    if root_only:
        overrides["max_depth"] = 0
    elif max_depth > 0:
        overrides["max_depth"] = max_depth
    if continuation_action_cap > 0:
        overrides["continuation_action_cap"] = continuation_action_cap
    return overrides


def _stage_max_floor(args: argparse.Namespace, prefix: str) -> int | None:
    value = int(getattr(args, f"{prefix}_max_floor", 0) or 0)
    return value if value > 0 else None


def _validate_runtime() -> None:
    from spirecomm.ai.torch_compat import torch

    if torch is None:
        raise RuntimeError(
            "torch is unavailable in this Python interpreter. Use "
            "/home/yydd/miniforge3/envs/spirecomm-rl/bin/python for teacher sweeps."
        )


def _run_round1(args: argparse.Namespace, candidates: list[tuple[str, dict[str, float]]]) -> list[dict[str, Any]]:
    stage_counts = sorted(set(count for count in _parse_int_csv(args.round1_stage_counts, fallback=[args.round1_count]) if count > 0))
    if int(args.round1_count) not in stage_counts:
        stage_counts.append(int(args.round1_count))
        stage_counts = sorted(set(stage_counts))
    stage_counts = [min(int(args.round1_count), count) for count in stage_counts if count <= int(args.round1_count)]
    stage_keeps = _parse_int_csv(args.round1_stage_keeps, fallback=[len(candidates)])
    active = list(candidates)
    ranked: list[dict[str, Any]] = []
    overrides = _round1_search_overrides(args)
    for stage_index, seed_count in enumerate(stage_counts):
        leaderboard_round_name = f"round1_seed{seed_count}"
        ranked = _run_candidate_stage(
            args=args,
            eval_round_name=f"round1_seed{int(args.round1_count)}",
            leaderboard_round_name=leaderboard_round_name,
            candidates=active,
            target_count=seed_count,
            search_overrides=overrides,
            max_floor_override=_stage_max_floor(args, "round1"),
        )
        keep_default = int(args.round2_top) if seed_count == int(args.round1_count) else len(active)
        keep = stage_keeps[min(stage_index, len(stage_keeps) - 1)] if stage_keeps else keep_default
        if seed_count == int(args.round1_count):
            keep = int(args.round2_top)
        keep = min(len(ranked), max(1, int(keep)))
        active = _candidate_list_from_ranked(ranked, keep)
        print(f"[{leaderboard_round_name}] promote={len(active)}", flush=True)
    return ranked[: int(args.round2_top)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast task-pool sweep for v3 teacher transition-reward coefficients.")
    parser.add_argument("--output-dir", type=Path, default=Path("teacher_sweep_runs/v3_teacher_coeff_sweep_v2_fast"))
    parser.add_argument("--param-bucket", choices=sorted(PARAM_BUCKETS), default="base8")
    parser.add_argument("--param-ranges-json", default=os.environ.get("SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON", ""))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--start-snapshot-dir", default="")
    parser.add_argument("--start-snapshot-cache-entries", type=int, default=512)
    parser.add_argument(
        "--auto-first-combat-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache each seed at its first combat decision and reuse that exact prefix across all sweep candidates.",
    )
    parser.add_argument(
        "--preload-start-snapshots-before-fork",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using fork workers, load start snapshots in the parent so workers inherit them via copy-on-write.",
    )
    parser.add_argument(
        "--first-combat-snapshot-count",
        type=int,
        default=0,
        help="Prepare this many first-combat snapshots up front. 0 auto-fills from the largest enabled round seed count.",
    )
    parser.add_argument(
        "--first-combat-snapshot-scope",
        choices=["global", "output"],
        default="global",
        help="global reuses first-combat snapshots across compatible sweeps; output keeps the old per-output-dir cache.",
    )
    parser.add_argument(
        "--first-combat-snapshot-global-root",
        type=Path,
        default=Path("_cache/v3_first_combat_snapshots"),
    )
    parser.add_argument("--first-combat-snapshot-workers", type=int, default=0)
    parser.add_argument("--first-combat-snapshot-max-steps", type=int, default=120)
    parser.add_argument(
        "--preload-selectors-before-fork",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On Linux fork, preload runtime selectors in the parent so workers inherit model memory.",
    )
    parser.add_argument("--metrics-mode", choices=["full", "floor"], default="floor")
    parser.add_argument(
        "--fast-teacher-combat-direct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In floor-only teacher sweeps, dispatch COMBAT decisions directly to the teacher selector.",
    )
    parser.add_argument("--summary-interval", type=int, default=0)
    parser.add_argument(
        "--write-first-partial-summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write summary_partial.json after a candidate's first finished seed.",
    )
    parser.add_argument("--progress-interval-tasks", type=int, default=25)
    parser.add_argument(
        "--max-pending-futures",
        type=int,
        default=0,
        help="Cap submitted-but-unfinished task batches. 0 auto-keeps a full worker queue; -1 submits all batches.",
    )
    parser.add_argument(
        "--task-batch-size",
        type=int,
        default=0,
        help="Run this many candidate-seed tasks per process-pool future. 0 auto-batches cheap proxy rounds.",
    )
    parser.add_argument(
        "--executor-backend",
        choices=["mp-pool", "process-pool"],
        default="mp-pool",
        help="mp-pool streams task batches with lower parent-process overhead; process-pool keeps the older futures backend.",
    )
    parser.add_argument(
        "--worker-crash-retries",
        type=int,
        default=4,
        help="Retry a stage this many times if the worker pool is killed externally; completed rows are kept.",
    )
    parser.add_argument(
        "--worker-crash-retry-worker-scale",
        type=float,
        default=0.90,
        help="Multiply worker count by this factor after each worker-pool crash retry.",
    )
    parser.add_argument(
        "--schedule-order",
        choices=["seed-major", "candidate-major"],
        default="seed-major",
        help="seed-major groups candidates for the same seed in one worker batch to reuse snapshot/cache state.",
    )
    parser.add_argument(
        "--seed-major-candidate-chunk-size",
        type=int,
        default=0,
        help="Candidates per seed-major task batch. 0 chooses from the active search budget and worker count.",
    )
    parser.add_argument(
        "--result-flush-interval",
        type=int,
        default=64,
        help="Buffer this many per-candidate seed results before appending results.jsonl.",
    )
    parser.add_argument(
        "--result-storage",
        choices=["round-aggregate", "per-candidate"],
        default="round-aggregate",
        help="round-aggregate writes one results.jsonl per round; per-candidate keeps the older candidate-local files.",
    )
    parser.add_argument(
        "--write-candidate-configs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write legacy per-candidate config files. Leaderboards already store params, so default is off for speed/cleanliness.",
    )
    parser.add_argument(
        "--write-candidate-summaries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-candidate summary.json files. Default off because leaderboard contains the same summaries.",
    )
    parser.add_argument(
        "--write-sweep-results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write legacy append-only sweep_results.jsonl. Default off because leaderboards contain the same round summaries.",
    )
    parser.add_argument("--random-seed", type=int, default=20260518)
    parser.add_argument("--round1-sampler", choices=["latin-hypercube", "corners"], default="latin-hypercube")
    parser.add_argument(
        "--round1-grid-json",
        default=os.environ.get("SPIRECOMM_TEACHER_SWEEP_ROUND1_GRID_JSON", ""),
        help="Explicit first-round grid. Overrides --round1-sampler and ignores --round1-size.",
    )
    parser.add_argument("--round0-count", type=int, default=300)
    parser.add_argument("--round1-size", type=int, default=256)
    parser.add_argument(
        "--include-default-in-round1",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include current default values for the swept params as a round1 anchor candidate.",
    )
    parser.add_argument("--round0-proxy-beam-width", type=int, default=0)
    parser.add_argument("--round0-proxy-node-budget", type=int, default=0)
    parser.add_argument("--round0-proxy-max-depth", type=int, default=0)
    parser.add_argument("--round0-proxy-continuation-action-cap", type=int, default=0)
    parser.add_argument("--round0-proxy-root-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--round0-max-floor", type=int, default=0)
    parser.add_argument("--round1-count", type=int, default=60)
    parser.add_argument("--round1-stage-counts", default="5,10,20,40,60")
    parser.add_argument("--round1-stage-keeps", default="224,160,96,64,32")
    parser.add_argument("--round1-max-floor", type=int, default=0)
    parser.add_argument("--round1-proxy-beam-width", type=int, default=0)
    parser.add_argument("--round1-proxy-node-budget", type=int, default=0)
    parser.add_argument("--round1-proxy-max-depth", type=int, default=0)
    parser.add_argument("--round1-proxy-continuation-action-cap", type=int, default=0)
    parser.add_argument("--round1-proxy-root-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--round2-top", type=int, default=32)
    parser.add_argument("--round2-count", type=int, default=100)
    parser.add_argument("--round2-max-floor", type=int, default=0)
    parser.add_argument("--round2-proxy-beam-width", type=int, default=0)
    parser.add_argument("--round2-proxy-node-budget", type=int, default=0)
    parser.add_argument("--round2-proxy-max-depth", type=int, default=0)
    parser.add_argument("--round2-proxy-continuation-action-cap", type=int, default=0)
    parser.add_argument("--round2-proxy-root-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--round3-top", type=int, default=16)
    parser.add_argument("--round3-count", type=int, default=300)
    parser.add_argument("--round3-max-floor", type=int, default=0)
    parser.add_argument("--round3-proxy-beam-width", type=int, default=0)
    parser.add_argument("--round3-proxy-node-budget", type=int, default=0)
    parser.add_argument("--round3-proxy-max-depth", type=int, default=0)
    parser.add_argument("--round3-proxy-continuation-action-cap", type=int, default=0)
    parser.add_argument("--round3-proxy-root-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--round4-top", type=int, default=6)
    parser.add_argument("--round4-count", type=int, default=600)
    parser.add_argument("--round4-max-floor", type=int, default=0)
    parser.add_argument("--round4-proxy-beam-width", type=int, default=0)
    parser.add_argument("--round4-proxy-node-budget", type=int, default=0)
    parser.add_argument("--round4-proxy-max-depth", type=int, default=0)
    parser.add_argument("--round4-proxy-continuation-action-cap", type=int, default=0)
    parser.add_argument("--round4-proxy-root-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Ignore existing fast-sweep result files in the output dir.")
    parser.add_argument(
        "--reuse-stage-prefix-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --force, still reuse seed results generated by earlier stages in the same run.",
    )
    parser.add_argument(
        "--stop-after",
        choices=["round0", "round1", "round2", "round3", "round4"],
        default="round4",
        help="Stop after this round; useful for proxy-speed probes and staged manual promotion.",
    )
    parser.add_argument(
        "--start-at",
        choices=["round0", "round1", "round2", "round3", "round4"],
        default="round0",
        help="Resume from an existing leaderboard and start evaluating this round.",
    )
    args = parser.parse_args()

    if args.force:
        args.resume = False
    global PARAM_RANGES
    PARAM_RANGES = param_ranges_for_bucket(args.param_bucket, args.param_ranges_json)
    sweep_base.PARAM_RANGES = PARAM_RANGES
    round_order = {"round0": 0, "round1": 1, "round2": 2, "round3": 3, "round4": 4}
    if round_order[args.stop_after] < round_order[args.start_at]:
        raise SystemExit(f"--stop-after={args.stop_after} is before --start-at={args.start_at}")
    enabled_round_counts = []
    if round_order[args.start_at] <= round_order["round0"] <= round_order[args.stop_after]:
        enabled_round_counts.append(int(args.round0_count))
    if round_order[args.start_at] <= round_order["round1"] <= round_order[args.stop_after]:
        enabled_round_counts.append(int(args.round1_count))
    if round_order[args.start_at] <= round_order["round2"] <= round_order[args.stop_after]:
        enabled_round_counts.append(int(args.round2_count))
    if round_order[args.start_at] <= round_order["round3"] <= round_order[args.stop_after]:
        enabled_round_counts.append(int(args.round3_count))
    if round_order[args.start_at] <= round_order["round4"] <= round_order[args.stop_after]:
        enabled_round_counts.append(int(args.round4_count))
    max_enabled_seed_count = max([0, *enabled_round_counts])
    if bool(args.auto_first_combat_snapshots) and int(args.first_combat_snapshot_count or 0) <= 0:
        args.first_combat_snapshot_count = max_enabled_seed_count
    if bool(args.preload_start_snapshots_before_fork) and int(args.start_snapshot_cache_entries) < int(
        args.first_combat_snapshot_count or 0
    ):
        args.start_snapshot_cache_entries = int(args.first_combat_snapshot_count or 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(_repo_root()))
    _validate_runtime()
    _write_json_fast(
        args.output_dir / "sweep_config.json",
        {
            "version": "v3_teacher_coeff_sweep_v2_fast_task_pool",
            "param_bucket": args.param_bucket,
            "param_ranges": PARAM_RANGES,
            "seed_start": args.seed_start,
            "workers": args.workers,
            "max_floor": args.max_floor,
            "max_steps": args.max_steps,
            "torch_threads": args.torch_threads,
            "blas_threads": args.blas_threads,
            "start_snapshot_dir": str(args.start_snapshot_dir or ""),
            "start_snapshot_cache_entries": int(args.start_snapshot_cache_entries),
            "auto_first_combat_snapshots": bool(args.auto_first_combat_snapshots),
            "preload_start_snapshots_before_fork": bool(args.preload_start_snapshots_before_fork),
            "first_combat_snapshot_count": int(args.first_combat_snapshot_count or 0),
            "first_combat_snapshot_scope": str(args.first_combat_snapshot_scope),
            "first_combat_snapshot_global_root": str(args.first_combat_snapshot_global_root),
            "preload_selectors_before_fork": bool(args.preload_selectors_before_fork),
            "fast_teacher_combat_direct": bool(args.fast_teacher_combat_direct),
            "v3_teacher_safe_single_action_inplace": _fast_bool_env(
                "SPIRECOMM_V3_TEACHER_SAFE_SINGLE_ACTION_INPLACE",
                True,
            ),
            "v3_teacher_combat_branch_only": _fast_bool_env(
                "SPIRECOMM_V3_TEACHER_COMBAT_BRANCH_ONLY",
                False,
            ),
            "v3_teacher_dedupe_equivalent_card_actions": _fast_bool_env(
                "SPIRECOMM_V3_TEACHER_DEDUPE_EQUIVALENT_CARD_ACTIONS",
                True,
            ),
            "teacher_sweep_batch_combat_roots": _fast_bool_env(
                "SPIRECOMM_TEACHER_SWEEP_BATCH_NON_POTION_ROOTS",
                False,
            ),
            "teacher_sweep_batch_min_candidates": int(
                os.environ.get("SPIRECOMM_TEACHER_SWEEP_BATCH_MIN_CANDIDATES", "2")
            ),
            "teacher_sweep_merge_split_states": _fast_bool_env(
                "SPIRECOMM_TEACHER_SWEEP_MERGE_SPLIT_STATES",
                False,
            ),
            "teacher_sweep_candidate_policy_cluster_size": int(
                os.environ.get("SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_SIZE", "1") or 1
            ),
            "teacher_sweep_candidate_policy_cluster_min_candidates": int(
                os.environ.get("SPIRECOMM_TEACHER_SWEEP_CANDIDATE_POLICY_CLUSTER_MIN_CANDIDATES", "999999") or 999999
            ),
            "scheduler": "task_pool",
            "schedule_order": str(args.schedule_order),
            "seed_major_candidate_chunk_size": int(args.seed_major_candidate_chunk_size),
            "round1_stage_counts": _parse_int_csv(args.round1_stage_counts, fallback=[args.round1_count]),
            "round1_stage_keeps": _parse_int_csv(args.round1_stage_keeps, fallback=[args.round2_top]),
            "round1_grid_json": str(args.round1_grid_json or ""),
            "include_default_in_round1": bool(args.include_default_in_round1),
            "reuse_stage_prefix_results": bool(args.reuse_stage_prefix_results),
            "write_first_partial_summary": bool(args.write_first_partial_summary),
            "max_pending_futures": int(args.max_pending_futures),
            "effective_max_pending_futures": (
                "all" if int(args.max_pending_futures) < 0 else ("auto" if int(args.max_pending_futures) == 0 else int(args.max_pending_futures))
            ),
            "task_batch_size": int(args.task_batch_size),
            "executor_backend": str(args.executor_backend),
            "result_flush_interval": int(args.result_flush_interval),
            "result_storage": str(args.result_storage),
            "write_candidate_configs": bool(args.write_candidate_configs),
            "write_candidate_summaries": bool(args.write_candidate_summaries),
            "write_sweep_results": bool(args.write_sweep_results),
            "round1_proxy_search": _round1_search_overrides(args),
            "round0_proxy_search": _search_overrides(args, "round0"),
            "round2_proxy_search": _search_overrides(args, "round2"),
            "round3_proxy_search": _search_overrides(args, "round3"),
            "round4_proxy_search": _search_overrides(args, "round4"),
            "round_max_floor": {
                "round0": int(args.round0_max_floor),
                "round1": int(args.round1_max_floor),
                "round2": int(args.round2_max_floor),
                "round3": int(args.round3_max_floor),
                "round4": int(args.round4_max_floor),
            },
            "start_at": args.start_at,
            "rounds": {
                "round0_default_seed_count": args.round0_count,
                "round1_candidates": args.round1_size,
                "round1_seed_count": args.round1_count,
                "round2_top": args.round2_top,
                "round2_seed_count": args.round2_count,
                "round3_top": args.round3_top,
                "round3_seed_count": args.round3_count,
                "round4_top": args.round4_top,
                "round4_seed_count": args.round4_count,
            },
        },
    )

    default_params = _default_teacher_config()
    if round_order[args.start_at] <= round_order["round0"] and int(args.round0_count) > 0:
        _run_candidate_stage(
            args=args,
            eval_round_name="round0_default",
            leaderboard_round_name="round0_default",
            candidates=[("default", default_params)],
            target_count=int(args.round0_count),
            search_overrides=_search_overrides(args, "round0"),
            max_floor_override=_stage_max_floor(args, "round0"),
        )
    if args.stop_after == "round0":
        return

    top_round1: list[dict[str, Any]] = []
    if round_order[args.start_at] <= round_order["round1"]:
        if str(args.round1_grid_json or "").strip():
            sampled = _grid_sample(str(args.round1_grid_json))
            print(
                f"[round1] grid sampler produced {len(sampled)} candidates; "
                f"ignoring --round1-size={args.round1_size}",
                flush=True,
            )
        elif args.round1_sampler == "corners":
            sampled = _corner_sample(PARAM_RANGES)
        else:
            sampled = _latin_hypercube_sample(n=int(args.round1_size), ranges=PARAM_RANGES, seed=int(args.random_seed))
        round1_candidates = [(f"r1_{index:03d}", params) for index, params in enumerate(sampled)]
        if args.include_default_in_round1:
            round1_candidates.insert(0, ("r1_default", _default_teacher_config()))
        _write_json_fast(
            args.output_dir / "round1_candidates.json",
            [{"candidate_id": candidate_id, "params": params} for candidate_id, params in round1_candidates],
        )
        top_round1 = _run_round1(args, round1_candidates)
    elif round_order[args.start_at] <= round_order["round2"]:
        top_round1 = _load_ranked_leaderboard(
            args.output_dir / f"leaderboard_round1_seed{int(args.round1_count)}.json",
            label="round1",
        )[: int(args.round2_top)]
    if args.stop_after == "round1":
        return

    round2_name = f"round2_seed{int(args.round2_count)}"
    round3_name = f"round3_seed{int(args.round3_count)}"
    round4_name = f"round4_seed{int(args.round4_count)}"

    top_round2: list[dict[str, Any]] = []
    if round_order[args.start_at] <= round_order["round2"]:
        round2 = _run_candidate_stage(
            args=args,
            eval_round_name=round2_name,
            leaderboard_round_name=round2_name,
            candidates=_candidate_list_from_ranked(top_round1, int(args.round2_top)),
            target_count=int(args.round2_count),
            search_overrides=_search_overrides(args, "round2"),
            max_floor_override=_stage_max_floor(args, "round2"),
        )
        top_round2 = round2[: int(args.round3_top)]
    elif round_order[args.start_at] <= round_order["round3"]:
        top_round2 = _load_ranked_leaderboard_first(
            [
                args.output_dir / f"leaderboard_{round2_name}.json",
                args.output_dir / "leaderboard_round2_seed100.json",
            ],
            label="round2",
        )[: int(args.round3_top)]
    if args.stop_after == "round2":
        return

    top_round3: list[dict[str, Any]]
    if round_order[args.start_at] <= round_order["round3"]:
        round3 = _run_candidate_stage(
            args=args,
            eval_round_name=round3_name,
            leaderboard_round_name=round3_name,
            candidates=_candidate_list_from_ranked(top_round2, int(args.round3_top)),
            target_count=int(args.round3_count),
            search_overrides=_search_overrides(args, "round3"),
            max_floor_override=_stage_max_floor(args, "round3"),
        )
        top_round3 = round3[: int(args.round4_top)]
    else:
        top_round3 = _load_ranked_leaderboard_first(
            [
                args.output_dir / f"leaderboard_{round3_name}.json",
                args.output_dir / "leaderboard_round3_seed300.json",
            ],
            label="round3",
        )[: int(args.round4_top)]
    if args.stop_after == "round3":
        return

    round4 = _run_candidate_stage(
        args=args,
        eval_round_name=round4_name,
        leaderboard_round_name=round4_name,
        candidates=_candidate_list_from_ranked(top_round3, int(args.round4_top)),
        target_count=int(args.round4_count),
        search_overrides=_search_overrides(args, "round4"),
        max_floor_override=_stage_max_floor(args, "round4"),
    )
    final_path = args.output_dir / f"leaderboard_{round4_name}.json"
    if final_path.exists():
        _write_json_fast(args.output_dir / "final_leaderboard.json", json.loads(final_path.read_text(encoding="utf-8")))
    if round4:
        print(json.dumps(round4[0], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
