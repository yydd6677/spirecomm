#!/usr/bin/env python3
from __future__ import annotations
# Allow this CLI to run directly from its workflow subdirectory.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PARAMS: tuple[dict[str, Any], ...] = (
    {"key": "monster_value", "env": "SPIRECOMM_MAP_DP_MONSTER_VALUE", "current": -10, "low": -30, "high": 5},
    {"key": "rest_value", "env": "SPIRECOMM_MAP_DP_REST_VALUE", "current": 70, "low": 20, "high": 100},
    {"key": "elite_base_value", "env": "SPIRECOMM_MAP_DP_ELITE_BASE", "current": 25, "low": -10, "high": 70},
    {"key": "green_elite_penalty", "env": "SPIRECOMM_MAP_DP_GREEN_ELITE_PENALTY", "current": 40, "low": 10, "high": 90},
    {"key": "shop_gold_unit_value", "env": "SPIRECOMM_MAP_DP_SHOP_GOLD_UNIT_VALUE", "current": 35, "low": 0, "high": 60},
    {
        "key": "shop_purgeable_curse_bonus",
        "env": "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_BONUS",
        "current": 60,
        "low": 0,
        "high": 140,
    },
    {
        "key": "shop_purgeable_curse_urgency_bonus",
        "env": "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_BONUS",
        "current": 50,
        "low": 0,
        "high": 140,
    },
    {
        "key": "shop_purgeable_curse_gold_threshold",
        "env": "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_GOLD_THRESHOLD",
        "current": 125,
        "low": 75,
        "high": 200,
    },
    {
        "key": "winged_offpath_penalty",
        "env": "SPIRECOMM_MAP_DP_WINGED_OFFPATH_PENALTY",
        "current": 20,
        "low": 0,
        "high": 50,
    },
)

PARAM_KEYS = tuple(str(item["key"]) for item in PARAMS)
SWEEP_PARAM_KEYS = (
    "monster_value",
    "rest_value",
    "elite_base_value",
    "shop_gold_unit_value",
    "shop_purgeable_curse_bonus",
)
SWEEP_PARAMS = tuple(item for item in PARAMS if str(item["key"]) in SWEEP_PARAM_KEYS)

NARROW_MEAN_V3_CENTER = {
    "monster_value": -11,
    "rest_value": 72,
    "elite_base_value": 26,
    "shop_gold_unit_value": 35,
    "shop_purgeable_curse_bonus": 60,
}

NARROW_MEAN_V3_RANGES = {
    "monster_value": (-17, -5),
    "rest_value": (60, 84),
    "elite_base_value": (14, 38),
    "shop_gold_unit_value": (23, 47),
    "shop_purgeable_curse_bonus": (30, 90),
}

NARROW_MEAN_V3_ANCHORS = (
    ("round3_mean", {"monster_value": -11, "rest_value": 68, "elite_base_value": 26, "shop_gold_unit_value": 31, "shop_purgeable_curse_bonus": 67}),
    ("round4_mean", {"monster_value": -11, "rest_value": 76, "elite_base_value": 26, "shop_gold_unit_value": 39, "shop_purgeable_curse_bonus": 53}),
    ("round34_mean", NARROW_MEAN_V3_CENTER),
)

CURRENT_BEST_LOCAL_CENTER = {
    "monster_value": -10,
    "rest_value": 70,
    "elite_base_value": 25,
    "shop_gold_unit_value": 35,
    "shop_purgeable_curse_bonus": 60,
}

CURRENT_BEST_LOCAL_RANGES = {
    "monster_value": (-16, -4),
    "rest_value": (58, 82),
    "elite_base_value": (13, 37),
    "shop_gold_unit_value": (23, 47),
    "shop_purgeable_curse_bonus": (30, 90),
}

CURRENT_BEST_LOCAL_ANCHORS = (
    ("current_best_center", CURRENT_BEST_LOCAL_CENTER),
    ("low_monster_high_rest", {"monster_value": -14, "rest_value": 78, "elite_base_value": 25, "shop_gold_unit_value": 35, "shop_purgeable_curse_bonus": 60}),
    ("higher_elite_shop", {"monster_value": -10, "rest_value": 70, "elite_base_value": 31, "shop_gold_unit_value": 41, "shop_purgeable_curse_bonus": 60}),
    ("curse_shop_bias", {"monster_value": -10, "rest_value": 70, "elite_base_value": 25, "shop_gold_unit_value": 35, "shop_purgeable_curse_bonus": 78}),
)


def current_params() -> dict[str, int]:
    return {str(item["key"]): int(item["current"]) for item in PARAMS}


def clamp_params(params: dict[str, float | int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in PARAMS:
        key = str(item["key"])
        low = int(item["low"])
        high = int(item["high"])
        result[key] = int(min(high, max(low, round(float(params[key])))))
    return result


def signature(params: dict[str, int]) -> tuple[int, ...]:
    return tuple(int(params[key]) for key in PARAM_KEYS)


def add_group(
    groups: list[dict[str, Any]],
    seen: set[tuple[int, ...]],
    name: str,
    params: dict[str, float | int],
    kind: str,
) -> None:
    clamped = clamp_params(params)
    sig = signature(clamped)
    if sig in seen:
        return
    seen.add(sig)
    groups.append({"name": name, "params": clamped, "kind": kind})


def build_round1_groups(group_count: int) -> list[dict[str, Any]]:
    rng = random.Random(20260528)
    groups: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    default = current_params()
    add_group(groups, seen, "default", default, "default")

    for item in SWEEP_PARAMS:
        key = str(item["key"])
        low_params = dict(default)
        low_params[key] = int(item["low"])
        add_group(groups, seen, f"{key}_low", low_params, "axis_low")
        high_params = dict(default)
        high_params[key] = int(item["high"])
        add_group(groups, seen, f"{key}_high", high_params, "axis_high")

    anchors = [
        ("safe_rest", {"rest_value": 90, "elite_base_value": 0}),
        ("elite_hunter", {"rest_value": 35, "elite_base_value": 60}),
        ("shop_curse_high", {"shop_purgeable_curse_bonus": 120}),
        ("shop_gold_high", {"shop_gold_unit_value": 55}),
        ("monster_avoid", {"monster_value": -28, "rest_value": 75}),
        ("monster_accept", {"monster_value": 3, "elite_base_value": 45}),
        ("shop_route", {"shop_gold_unit_value": 45, "shop_purgeable_curse_bonus": 100}),
        ("balanced_high_value", {"monster_value": -5, "rest_value": 70, "elite_base_value": 45, "shop_gold_unit_value": 35}),
    ]
    for name, override in anchors:
        params = dict(default)
        params.update(override)
        add_group(groups, seen, name, params, "anchor")

    sample_count = max(1, group_count - len(groups))
    columns: dict[str, list[int]] = {}
    for item in SWEEP_PARAMS:
        key = str(item["key"])
        low = int(item["low"])
        high = int(item["high"])
        values = [int(round(low + ((index + rng.random()) / sample_count) * (high - low))) for index in range(sample_count)]
        rng.shuffle(values)
        columns[key] = values
    for index in range(sample_count):
        params = dict(default)
        params.update({key: columns[key][index] for key in SWEEP_PARAM_KEYS})
        add_group(groups, seen, f"lhs_{index + 1:03d}", params, "lhs")

    fill_index = 0
    while len(groups) < group_count:
        fill_index += 1
        add_group(
            groups,
            seen,
            f"random_{fill_index:03d}",
            {
                **default,
                **{str(item["key"]): rng.randint(int(item["low"]), int(item["high"])) for item in SWEEP_PARAMS},
            },
            "random",
        )
    return groups[:group_count]


def build_narrow_mean_v3_round3_groups(group_count: int) -> list[dict[str, Any]]:
    rng = random.Random(20260531)
    groups: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    default = current_params()
    add_group(groups, seen, "default", default, "default")
    for name, override in NARROW_MEAN_V3_ANCHORS:
        params = dict(default)
        params.update(override)
        add_group(groups, seen, name, params, "mean_anchor")

    center = dict(default)
    center.update(NARROW_MEAN_V3_CENTER)
    for key in SWEEP_PARAM_KEYS:
        low, high = NARROW_MEAN_V3_RANGES[key]
        low_params = dict(center)
        low_params[key] = low
        add_group(groups, seen, f"{key}_local_low", low_params, "local_axis")
        high_params = dict(center)
        high_params[key] = high
        add_group(groups, seen, f"{key}_local_high", high_params, "local_axis")

    sample_count = max(1, group_count - len(groups))
    columns: dict[str, list[int]] = {}
    for key in SWEEP_PARAM_KEYS:
        low, high = NARROW_MEAN_V3_RANGES[key]
        values = [int(round(low + ((index + rng.random()) / sample_count) * (high - low))) for index in range(sample_count)]
        rng.shuffle(values)
        columns[key] = values
    for index in range(sample_count):
        params = dict(default)
        params.update({key: columns[key][index] for key in SWEEP_PARAM_KEYS})
        add_group(groups, seen, f"narrow_lhs_{index + 1:03d}", params, "narrow_lhs")

    fill_index = 0
    while len(groups) < group_count:
        fill_index += 1
        params = dict(default)
        params.update({key: rng.randint(*NARROW_MEAN_V3_RANGES[key]) for key in SWEEP_PARAM_KEYS})
        add_group(groups, seen, f"narrow_random_{fill_index:03d}", params, "narrow_random")
    return groups[:group_count]


def build_current_best_local_round3_groups(group_count: int) -> list[dict[str, Any]]:
    rng = random.Random(20260601)
    groups: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    default = current_params()
    add_group(groups, seen, "default", default, "default")
    for name, override in CURRENT_BEST_LOCAL_ANCHORS:
        params = dict(default)
        params.update(override)
        add_group(groups, seen, name, params, "current_anchor")

    center = dict(default)
    center.update(CURRENT_BEST_LOCAL_CENTER)
    for key in SWEEP_PARAM_KEYS:
        low, high = CURRENT_BEST_LOCAL_RANGES[key]
        low_params = dict(center)
        low_params[key] = low
        add_group(groups, seen, f"{key}_local_low", low_params, "local_axis")
        high_params = dict(center)
        high_params[key] = high
        add_group(groups, seen, f"{key}_local_high", high_params, "local_axis")

    sample_count = max(1, group_count - len(groups))
    columns: dict[str, list[int]] = {}
    for key in SWEEP_PARAM_KEYS:
        low, high = CURRENT_BEST_LOCAL_RANGES[key]
        values = [int(round(low + ((index + rng.random()) / sample_count) * (high - low))) for index in range(sample_count)]
        rng.shuffle(values)
        columns[key] = values
    for index in range(sample_count):
        params = dict(default)
        params.update({key: columns[key][index] for key in SWEEP_PARAM_KEYS})
        add_group(groups, seen, f"current_lhs_{index + 1:03d}", params, "current_lhs")

    fill_index = 0
    while len(groups) < group_count:
        fill_index += 1
        params = dict(default)
        params.update({key: rng.randint(*CURRENT_BEST_LOCAL_RANGES[key]) for key in SWEEP_PARAM_KEYS})
        add_group(groups, seen, f"current_random_{fill_index:03d}", params, "current_random")
    return groups[:group_count]


def sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda item: float(item.get("mean_floor") or -1.0), reverse=True)


def weighted_center(rows: list[dict[str, Any]], top_n: int) -> tuple[dict[str, int], dict[str, float]]:
    top = sorted_rows(rows)[: max(1, top_n)]
    weights = [1.0 / float(index + 1) for index in range(len(top))]
    total_weight = sum(weights)
    center_raw: dict[str, float] = {}
    sigma: dict[str, float] = {}
    for item in SWEEP_PARAMS:
        key = str(item["key"])
        mean = sum(float(row[key]) * weight for row, weight in zip(top, weights)) / total_weight
        variance = sum(((float(row[key]) - mean) ** 2) * weight for row, weight in zip(top, weights)) / total_weight
        span = float(int(item["high"]) - int(item["low"]))
        center_raw[key] = mean
        sigma[key] = max(math.sqrt(variance), span * 0.06)
    center = current_params()
    center.update(clamp_params({**center, **center_raw}))
    return center, sigma


def row_params(row: dict[str, Any]) -> dict[str, int]:
    return clamp_params({key: int(row[key]) for key in PARAM_KEYS})


def build_local_groups(
    rows: list[dict[str, Any]],
    *,
    group_count: int,
    top_anchor_count: int,
    center_top_n: int,
    rng_seed: int,
    sigma_scale: float,
    axis_scales: tuple[float, ...],
    kind_prefix: str,
) -> list[dict[str, Any]]:
    ranked = sorted_rows(rows)
    groups: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    add_group(groups, seen, "default", current_params(), "default")
    for rank, row in enumerate(ranked[:top_anchor_count], start=1):
        add_group(groups, seen, f"top{rank:02d}_{row.get('name', 'candidate')}", row_params(row), "top_anchor")
    center, sigma = weighted_center(ranked, top_n=center_top_n)
    add_group(groups, seen, "weighted_center", center, "center")
    for key in SWEEP_PARAM_KEYS:
        for scale in axis_scales:
            up = dict(center)
            up[key] = int(round(float(center[key]) + sigma[key] * scale))
            add_group(groups, seen, f"center_{key}_up_{scale:.2f}", up, "axis")
            down = dict(center)
            down[key] = int(round(float(center[key]) - sigma[key] * scale))
            add_group(groups, seen, f"center_{key}_down_{scale:.2f}", down, "axis")

    rng = random.Random(rng_seed)
    sample_index = 0
    while len(groups) < group_count:
        sample_index += 1
        params = dict(center)
        params.update({key: center[key] + rng.gauss(0.0, sigma[key] * sigma_scale) for key in SWEEP_PARAM_KEYS})
        add_group(groups, seen, f"{kind_prefix}_{sample_index:03d}", params, "local_normal")
    return groups[:group_count]


def build_final_groups(rows: list[dict[str, Any]], *, group_count: int) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    add_group(groups, seen, "default", current_params(), "default")
    for row in sorted_rows(rows):
        add_group(groups, seen, f"final_top{len(groups):02d}_{row.get('name', 'candidate')}", row_params(row), "finalist")
        if len(groups) >= group_count:
            break
    return groups[:group_count]


def group_dir_name(index: int, group: dict[str, Any]) -> str:
    params = group["params"]
    short = (
        f"m{params['monster_value']}_r{params['rest_value']}_e{params['elite_base_value']}"
        f"_g{params['green_elite_penalty']}_sg{params['shop_gold_unit_value']}"
        f"_cb{params['shop_purgeable_curse_bonus']}_cu{params['shop_purgeable_curse_urgency_bonus']}"
        f"_ct{params['shop_purgeable_curse_gold_threshold']}_w{params['winged_offpath_penalty']}"
    )
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(group["name"]))[:60]
    return f"g{index:03d}_{safe_name}_{short}"


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def existing_count(summary: dict[str, Any] | None) -> int:
    if not summary:
        return 0
    try:
        return int(summary.get("count") or 0)
    except (TypeError, ValueError):
        return 0


def write_results(output_dir: Path, stage_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted_rows(rows)
    (output_dir / f"{stage_name}_results.json").write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "rank",
        "group",
        "name",
        "kind",
        *PARAM_KEYS,
        "mean_floor",
        "win_count",
        "timeout_count",
        "error_count",
        "seconds",
        "output_dir",
    ]
    with (output_dir / f"{stage_name}_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            writer.writerow({"rank": rank, **row})
    return ranked


def load_stage_results(output_dir: Path, stage_name: str) -> list[dict[str, Any]]:
    path = output_dir / f"{stage_name}_results.json"
    data = load_json(path)
    if not isinstance(data, list):
        raise SystemExit(f"missing stage results: {path}")
    return data


def run_eval_group(
    *,
    repo_root: Path,
    group_dir: Path,
    params: dict[str, int],
    args: argparse.Namespace,
    count: int,
) -> dict[str, Any] | None:
    summary_path = group_dir / "summary.json"
    summary = load_json(summary_path)
    if existing_count(summary) >= count and int((summary or {}).get("error_count") or 0) == 0:
        return summary

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    for item in PARAMS:
        env[str(item["env"])] = str(int(params[str(item["key"])]))

    cmd = [
        sys.executable,
        "-u",
        str(repo_root / "scripts/v3_combat/evaluate_v3_rollout_batch.py"),
        "--output-dir",
        str(group_dir),
        "--seed-start",
        str(int(args.seed_start)),
        "--count",
        str(int(count)),
        "--workers",
        str(int(args.workers)),
        "--combat-selector",
        str(args.combat_selector),
        "--v3-combat-model",
        str(args.v3_combat_model),
        "--device",
        str(args.device),
        "--combat-device",
        str(args.combat_device),
        "--torch-threads",
        str(int(args.torch_threads)),
        "--max-floor",
        str(int(args.max_floor)),
        "--max-steps",
        str(int(args.max_steps)),
        "--combat-stall-limit",
        str(int(args.combat_stall_limit)),
        "--mean-floor-only",
        "--summary-interval",
        str(int(args.summary_interval)),
        "--task-batch-size",
        str(int(args.task_batch_size)),
        "--preload-selectors",
        str(args.preload_selectors),
        "--result-flush-interval",
        str(int(args.result_flush_interval)),
    ]
    if bool(args.resume):
        cmd.append("--resume")
    if bool(args.rerun_timeouts):
        cmd.append("--rerun-timeouts")
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "map_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    (group_dir / "command.json").write_text(json.dumps({"cmd": cmd, "env": {item["env"]: env[str(item["env"])] for item in PARAMS}}, indent=2), encoding="utf-8")
    log_path = group_dir / "run.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n# start {time.strftime('%Y-%m-%d %H:%M:%S')} target_count={count} params={params}\n")
        log.flush()
        if args.dry_run:
            print("DRY-RUN", " ".join(cmd), flush=True)
            return None
        result = subprocess.run(cmd, cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise SystemExit(f"eval failed for {group_dir}; see {log_path}")
    return load_json(summary_path)


def candidate_env_for_params(params: dict[str, int]) -> dict[str, str]:
    return {str(item["env"]): str(int(params[str(item["key"])])) for item in PARAMS}


def run_stage_shared_prefix(
    *,
    repo_root: Path,
    output_dir: Path,
    stage_name: str,
    groups: list[dict[str, Any]],
    args: argparse.Namespace,
    count: int,
) -> list[dict[str, Any]]:
    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        {
            "name": str(group["name"]),
            "kind": str(group["kind"]),
            "env": candidate_env_for_params(group["params"]),
            "params": {key: int(group["params"][key]) for key in PARAM_KEYS},
        }
        for group in groups
    ]
    candidate_json = stage_dir / "shared_prefix_candidates.json"
    candidate_json.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    shared_dir = stage_dir / "_shared_prefix"
    cmd = [
        sys.executable,
        "-u",
        str(repo_root / "scripts/v3_combat/run_shared_prefix_sweep.py"),
        "--candidate-json",
        str(candidate_json),
        "--output-dir",
        str(shared_dir),
        "--affected-phases",
        "MAP",
        "--seed-start",
        str(int(args.seed_start)),
        "--count",
        str(int(count)),
        "--workers",
        str(int(args.workers)),
        "--task-batch-size",
        str(int(args.task_batch_size)),
        "--combat-selector",
        str(args.combat_selector),
        "--v3-combat-model",
        str(args.v3_combat_model),
        "--device",
        str(args.device),
        "--combat-device",
        str(args.combat_device),
        "--torch-threads",
        str(int(args.torch_threads)),
        "--max-floor",
        str(int(args.max_floor)),
        "--max-steps",
        str(int(args.max_steps)),
        "--combat-stall-limit",
        str(int(args.combat_stall_limit)),
        "--summary-interval",
        str(int(args.summary_interval)),
        "--result-flush-interval",
        str(int(args.result_flush_interval)),
        "--preload-selectors",
        str(args.preload_selectors),
    ]
    if bool(args.resume):
        cmd.append("--resume")
    if bool(getattr(args, "racing", False)):
        cmd.extend(
            [
                "--racing",
                "--racing-min-seeds",
                str(int(args.racing_min_seeds)),
                "--racing-wave-size",
                str(int(args.racing_wave_size)),
                "--racing-z",
                str(float(args.racing_z)),
                "--racing-margin",
                str(float(args.racing_margin)),
                "--racing-keep-min",
                str(int(args.racing_keep_min)),
            ]
        )
    log_path = stage_dir / "shared_prefix.log"
    (stage_dir / "groups.json").write_text(
        json.dumps({"params": list(PARAMS), "seed_start": args.seed_start, "count": count, "groups": groups}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (stage_dir / "shared_prefix_command.json").write_text(json.dumps({"cmd": cmd}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{stage_name}] shared-prefix run groups={len(groups)} seeds={count} output={shared_dir}", flush=True)
    if args.dry_run:
        print("DRY-RUN", " ".join(cmd), flush=True)
        return []
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n# start {time.strftime('%Y-%m-%d %H:%M:%S')} stage={stage_name} groups={len(groups)} seeds={count}\n")
        log.flush()
        result = subprocess.run(cmd, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise SystemExit(f"shared-prefix eval failed for {stage_name}; see {log_path}")
    candidate_results = load_json(shared_dir / "candidate_results.json")
    if not isinstance(candidate_results, list):
        raise SystemExit(f"missing shared-prefix candidate results: {shared_dir / 'candidate_results.json'}")
    by_name = {str(row.get("name")): row for row in candidate_results}
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        summary = by_name.get(str(group["name"]))
        if not isinstance(summary, dict):
            raise SystemExit(f"shared-prefix result missing candidate {group['name']}")
        params = group["params"]
        rows.append(
            {
                "group": index,
                "name": group["name"],
                "kind": group["kind"],
                **{key: int(params[key]) for key in PARAM_KEYS},
                "mean_floor": summary.get("mean_floor"),
                "win_count": summary.get("win_count"),
                "timeout_count": summary.get("timeout_count"),
                "error_count": summary.get("error_count"),
                "seconds": None,
                "output_dir": str(shared_dir),
            }
        )
    ranked = write_results(output_dir, stage_name, rows)
    if ranked:
        best = ranked[0]
        shared_summary = load_json(shared_dir / "summary.json") or {}
        dag_stats = shared_summary.get("dag_stats") if isinstance(shared_summary, dict) else {}
        compression = float((dag_stats or {}).get("action_eval_compression") or 0.0)
        print(
            f"[{stage_name}] shared-prefix done best={float(best.get('mean_floor') or 0.0):.3f} "
            f"best={best.get('name')} compression={compression:.1f}x",
            flush=True,
        )
    return ranked


def stage_row(index: int, group: dict[str, Any], summary: dict[str, Any], group_dir: Path) -> dict[str, Any]:
    params = group["params"]
    return {
        "group": index,
        "name": group["name"],
        "kind": group["kind"],
        **{key: int(params[key]) for key in PARAM_KEYS},
        "mean_floor": summary.get("mean_floor"),
        "win_count": summary.get("win_count"),
        "timeout_count": summary.get("timeout_count"),
        "error_count": summary.get("error_count"),
        "seconds": summary.get("seconds"),
        "output_dir": str(group_dir),
    }


def run_stage(
    *,
    repo_root: Path,
    output_dir: Path,
    stage_name: str,
    groups: list[dict[str, Any]],
    args: argparse.Namespace,
    count: int,
) -> list[dict[str, Any]]:
    if bool(getattr(args, "shared_prefix", False)):
        return run_stage_shared_prefix(
            repo_root=repo_root,
            output_dir=output_dir,
            stage_name=stage_name,
            groups=groups,
            args=args,
            count=count,
        )
    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "groups.json").write_text(
        json.dumps({"params": list(PARAMS), "seed_start": args.seed_start, "count": count, "groups": groups}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, group in enumerate(groups, start=1):
        params = group["params"]
        group_dir = stage_dir / group_dir_name(index, group)
        summary = load_json(group_dir / "summary.json")
        verb = "skip" if existing_count(summary) >= count else "run"
        print(
            f"[{stage_name} {index:03d}/{len(groups)}] {verb} {group['name']} target={count} "
            + " ".join(f"{key}={params[key]}" for key in PARAM_KEYS),
            flush=True,
        )
        summary = run_eval_group(repo_root=repo_root, group_dir=group_dir, params=params, args=args, count=count)
        if summary is None:
            continue
        row = stage_row(index, group, summary, group_dir)
        rows.append(row)
        ranked = write_results(output_dir, stage_name, rows)
        best = ranked[0]
        rate = len(rows) / max(1e-9, time.time() - started)
        remaining = len(groups) - len(rows)
        eta = remaining / max(rate, 1e-9)
        print(
            f"[{stage_name} {index:03d}/{len(groups)}] done mean={float(row.get('mean_floor') or 0.0):.3f} "
            f"wins={row.get('win_count')} best={float(best.get('mean_floor') or 0.0):.3f} "
            f"best={best.get('name')} done_rate={rate:.4f}/s eta_min={eta / 60.0:.1f}",
            flush=True,
        )
    return write_results(output_dir, stage_name, rows)


def write_final(output_dir: Path, rows_by_stage: dict[str, list[dict[str, Any]]]) -> None:
    all_rows: list[dict[str, Any]] = []
    for stage, rows in rows_by_stage.items():
        for row in rows:
            all_rows.append({"stage": stage, **row})
    ranked = sorted_rows(all_rows)
    (output_dir / "final_leaderboard.json").write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")
    if ranked:
        print("final best", json.dumps(ranked[0], ensure_ascii=False, sort_keys=True), flush=True)


def should_run(start_at: str, stage_name: str) -> bool:
    order = ["round0_default", "round1_seed100", "round2_seed200", "round3_seed300", "round4_seed600"]
    return order.index(stage_name) >= order.index(start_at)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-round rollout sweep for map DP coefficients.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("map_sweep_runs/v5_104_map_dp_sweep_v2"))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--round0-count", type=int, default=600)
    parser.add_argument("--round1-size", type=int, default=64)
    parser.add_argument("--round1-count", type=int, default=100)
    parser.add_argument("--round2-groups", type=int, default=32)
    parser.add_argument("--round2-count", type=int, default=200)
    parser.add_argument("--round3-groups", type=int, default=16)
    parser.add_argument("--round3-count", type=int, default=300)
    parser.add_argument("--round4-top", type=int, default=8)
    parser.add_argument("--round4-count", type=int, default=600)
    parser.add_argument(
        "--round3-source",
        choices=["prior", "narrow_mean_v3", "current_best_local"],
        default="prior",
        help="Build round3 from prior rounds or from the v2 round3/round4 mean-centered local range.",
    )
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SPIRECOMM_MAP_SWEEP_WORKERS", "12")))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--v3-combat-model", type=Path, default=Path("models/v3_combat_transformer_v5_18_epoch003_rollout_best.pt"))
    parser.add_argument("--combat-selector", default="v3-candidate", choices=["legacy-slot", "v3-candidate", "v3-teacher"])
    parser.add_argument("--device", default=os.environ.get("SPIRECOMM_EVAL_DEVICE", "cpu"))
    parser.add_argument("--combat-device", default=os.environ.get("SPIRECOMM_EVAL_COMBAT_DEVICE", "auto"))
    parser.add_argument(
        "--auto-cuda-worker-max",
        type=int,
        default=int(os.environ.get("SPIRECOMM_MAP_SWEEP_AUTO_CUDA_WORKER_MAX", "16")),
        help="When --combat-device auto, use CPU instead of CUDA above this worker count to avoid many processes contending one GPU.",
    )
    parser.add_argument("--preload-selectors", choices=["auto", "always", "never"], default="always")
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--combat-stall-limit",
        type=int,
        default=int(os.environ.get("SPIRECOMM_COMBAT_STALL_LIMIT", "250")),
        help="Forwarded to scripts/v3_combat/evaluate_v3_rollout_batch.py; use 0 to disable combat no-HP-progress timeout.",
    )
    parser.add_argument("--summary-interval", type=int, default=50)
    parser.add_argument("--task-batch-size", type=int, default=1)
    parser.add_argument("--result-flush-interval", type=int, default=16)
    parser.add_argument(
        "--shared-prefix",
        action=argparse.BooleanOptionalAction,
        default=str(os.environ.get("SPIRECOMM_MAP_SWEEP_SHARED_PREFIX", "1")).strip().lower() not in {"0", "false", "no", "off"},
        help="Evaluate each stage with scripts/v3_combat/run_shared_prefix_sweep.py so candidates share unchanged per-seed trajectory prefixes.",
    )
    parser.add_argument(
        "--racing",
        action="store_true",
        default=str(os.environ.get("SPIRECOMM_MAP_SWEEP_RACING", "")).strip().lower() in {"1", "true", "yes", "on"},
        help="With --shared-prefix, stop clearly losing candidates from receiving later seeds.",
    )
    parser.add_argument("--racing-min-seeds", type=int, default=int(os.environ.get("SPIRECOMM_MAP_SWEEP_RACING_MIN_SEEDS", "60")))
    parser.add_argument("--racing-wave-size", type=int, default=int(os.environ.get("SPIRECOMM_MAP_SWEEP_RACING_WAVE_SIZE", "20")))
    parser.add_argument("--racing-z", type=float, default=float(os.environ.get("SPIRECOMM_MAP_SWEEP_RACING_Z", "2.5")))
    parser.add_argument("--racing-margin", type=float, default=float(os.environ.get("SPIRECOMM_MAP_SWEEP_RACING_MARGIN", "0.0")))
    parser.add_argument("--racing-keep-min", type=int, default=int(os.environ.get("SPIRECOMM_MAP_SWEEP_RACING_KEEP_MIN", "4")))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--rerun-timeouts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--start-at",
        choices=["round0_default", "round1_seed100", "round2_seed200", "round3_seed300", "round4_seed600"],
        default="round0_default",
    )
    parser.add_argument(
        "--stop-after",
        choices=["round0_default", "round1_seed100", "round2_seed200", "round3_seed300", "round4_seed600", "all"],
        default="all",
    )
    args = parser.parse_args()

    if str(args.combat_device).strip().lower() == "auto" and int(args.workers) > int(args.auto_cuda_worker_max):
        args.combat_device = "cpu"

    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sweep_config.json").write_text(
        json.dumps(
            {
                "version": (
                    "v5_104_map_dp_sweep_v4_current_best_local"
                    if args.round3_source == "current_best_local"
                    else ("v5_104_map_dp_sweep_v3" if args.round3_source == "narrow_mean_v3" else "v5_104_map_dp_sweep_v2")
                ),
                "params": list(PARAMS),
                "swept_param_keys": list(SWEEP_PARAM_KEYS),
                "default_params": current_params(),
                "round3_source": str(args.round3_source),
                "narrow_mean_v3_center": NARROW_MEAN_V3_CENTER,
                "narrow_mean_v3_ranges": NARROW_MEAN_V3_RANGES,
                "current_best_local_center": CURRENT_BEST_LOCAL_CENTER,
                "current_best_local_ranges": CURRENT_BEST_LOCAL_RANGES,
                "args": {key: str(value) for key, value in vars(args).items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows_by_stage: dict[str, list[dict[str, Any]]] = {}
    skip_round1_round2_for_direct_round3 = args.round3_source != "prior"

    stage = "round0_default"
    if should_run(args.start_at, stage):
        rows_by_stage[stage] = run_stage(
            repo_root=repo_root,
            output_dir=output_dir,
            stage_name=stage,
            groups=[{"name": "default", "params": current_params(), "kind": "default"}],
            args=args,
            count=int(args.round0_count),
        )
    else:
        rows_by_stage[stage] = load_stage_results(output_dir, stage)
    if args.stop_after == stage:
        write_final(output_dir, rows_by_stage)
        return

    stage = "round1_seed100"
    if skip_round1_round2_for_direct_round3:
        rows_by_stage[stage] = []
    elif should_run(args.start_at, stage):
        rows_by_stage[stage] = run_stage(
            repo_root=repo_root,
            output_dir=output_dir,
            stage_name=stage,
            groups=build_round1_groups(int(args.round1_size)),
            args=args,
            count=int(args.round1_count),
        )
    else:
        rows_by_stage[stage] = load_stage_results(output_dir, stage)
    if args.stop_after == stage:
        write_final(output_dir, rows_by_stage)
        return

    stage = "round2_seed200"
    if skip_round1_round2_for_direct_round3:
        rows_by_stage[stage] = []
    elif should_run(args.start_at, stage):
        rows_by_stage[stage] = run_stage(
            repo_root=repo_root,
            output_dir=output_dir,
            stage_name=stage,
            groups=build_local_groups(
                rows_by_stage["round1_seed100"],
                group_count=int(args.round2_groups),
                top_anchor_count=20,
                center_top_n=20,
                rng_seed=20260529,
                sigma_scale=1.20,
                axis_scales=(0.5, 1.0, 1.5),
                kind_prefix="round2_local",
            ),
            args=args,
            count=int(args.round2_count),
        )
    else:
        rows_by_stage[stage] = load_stage_results(output_dir, stage)
    if args.stop_after == stage:
        write_final(output_dir, rows_by_stage)
        return

    stage = "round3_seed300"
    if should_run(args.start_at, stage):
        if args.round3_source == "narrow_mean_v3":
            groups = build_narrow_mean_v3_round3_groups(int(args.round3_groups))
        elif args.round3_source == "current_best_local":
            groups = build_current_best_local_round3_groups(int(args.round3_groups))
        else:
            groups = build_local_groups(
                rows_by_stage["round1_seed100"] + rows_by_stage["round2_seed200"],
                group_count=int(args.round3_groups),
                top_anchor_count=16,
                center_top_n=24,
                rng_seed=20260530,
                sigma_scale=0.85,
                axis_scales=(0.5, 1.0),
                kind_prefix="round3_local",
            )
        rows_by_stage[stage] = run_stage(
            repo_root=repo_root,
            output_dir=output_dir,
            stage_name=stage,
            groups=groups,
            args=args,
            count=int(args.round3_count),
        )
    else:
        rows_by_stage[stage] = load_stage_results(output_dir, stage)
    if args.stop_after == stage:
        write_final(output_dir, rows_by_stage)
        return

    stage = "round4_seed600"
    if should_run(args.start_at, stage):
        rows_by_stage[stage] = run_stage(
            repo_root=repo_root,
            output_dir=output_dir,
            stage_name=stage,
            groups=build_final_groups(
                rows_by_stage["round1_seed100"] + rows_by_stage["round2_seed200"] + rows_by_stage["round3_seed300"],
                group_count=int(args.round4_top),
            ),
            args=args,
            count=int(args.round4_count),
        )
    else:
        rows_by_stage[stage] = load_stage_results(output_dir, stage)
    write_final(output_dir, rows_by_stage)


if __name__ == "__main__":
    main()
