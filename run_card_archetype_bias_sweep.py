#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BIAS_KEYS = ("strength", "block", "aoe", "exhaust")
ENV_KEYS = {
    "strength": "SPIRECOMM_CARD_ARCHETYPE_STRENGTH_BIAS",
    "block": "SPIRECOMM_CARD_ARCHETYPE_BLOCK_BIAS",
    "aoe": "SPIRECOMM_CARD_ARCHETYPE_AOE_BIAS",
    "exhaust": "SPIRECOMM_CARD_ARCHETYPE_EXHAUST_BIAS",
}
RANGES = {
    "strength": (0.0, 0.8),
    "block": (0.0, 1.0),
    "aoe": (0.0, 0.9),
    "exhaust": (0.0, 1.0),
}


def _round_value(value: float) -> float:
    return round(float(value) + 0.0, 2)


def _clamp_group(group: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in BIAS_KEYS:
        low, high = RANGES[key]
        result[key] = _round_value(min(high, max(low, float(group[key]))))
    return result


def _signature(params: dict[str, float]) -> tuple[float, ...]:
    return tuple(float(params[key]) for key in BIAS_KEYS)


def build_round1_groups() -> list[dict[str, Any]]:
    anchors: list[tuple[str, dict[str, float]]] = [
        ("weak_only_all_zero", {"strength": 0.0, "block": 0.0, "aoe": 0.0, "exhaust": 0.0}),
        ("all_025", {"strength": 0.25, "block": 0.25, "aoe": 0.25, "exhaust": 0.25}),
        ("all_050", {"strength": 0.50, "block": 0.50, "aoe": 0.50, "exhaust": 0.50}),
        ("code_default", {"strength": 0.35, "block": 0.60, "aoe": 0.55, "exhaust": 0.60}),
        ("high_strength", {"strength": 0.80, "block": 0.20, "aoe": 0.25, "exhaust": 0.25}),
        ("high_block", {"strength": 0.20, "block": 1.00, "aoe": 0.25, "exhaust": 0.35}),
        ("high_aoe", {"strength": 0.20, "block": 0.25, "aoe": 0.90, "exhaust": 0.25}),
        ("high_exhaust", {"strength": 0.20, "block": 0.35, "aoe": 0.25, "exhaust": 1.00}),
    ]
    groups: list[dict[str, Any]] = [
        {"name": name, "params": _clamp_group(params), "kind": "anchor"} for name, params in anchors
    ]
    seen = {_signature(group["params"]) for group in groups}

    rng = random.Random(20260522)
    sample_count = 24
    columns: dict[str, list[float]] = {}
    for key in BIAS_KEYS:
        low, high = RANGES[key]
        values = [_round_value(low + ((index + rng.random()) / sample_count) * (high - low)) for index in range(sample_count)]
        rng.shuffle(values)
        columns[key] = values

    sample_index = 0
    for row in range(sample_count):
        params = _clamp_group({key: columns[key][row] for key in BIAS_KEYS})
        signature = _signature(params)
        if signature in seen:
            continue
        seen.add(signature)
        sample_index += 1
        groups.append({"name": f"lhs_{sample_index:02d}", "params": params, "kind": "lhs"})

    while len(groups) < 32:
        params = _clamp_group({key: rng.uniform(*RANGES[key]) for key in BIAS_KEYS})
        signature = _signature(params)
        if signature in seen:
            continue
        seen.add(signature)
        sample_index += 1
        groups.append({"name": f"lhs_{sample_index:02d}", "params": params, "kind": "lhs"})
    return groups[:32]


def _weighted_center(rows: list[dict[str, Any]], top_n: int) -> tuple[dict[str, float], dict[str, float]]:
    top = rows[: max(1, top_n)]
    weights = [1.0 / float(index + 1) for index in range(len(top))]
    total_weight = sum(weights)
    center: dict[str, float] = {}
    sigma: dict[str, float] = {}
    for key in BIAS_KEYS:
        mean = sum(float(row[key]) * weight for row, weight in zip(top, weights)) / total_weight
        variance = sum(((float(row[key]) - mean) ** 2) * weight for row, weight in zip(top, weights)) / total_weight
        low, high = RANGES[key]
        center[key] = _round_value(mean)
        sigma[key] = max(math.sqrt(variance), (high - low) * 0.08)
    return center, sigma


def build_round2_groups(round1_rows: list[dict[str, Any]], *, group_count: int) -> list[dict[str, Any]]:
    rows = sorted(round1_rows, key=lambda item: float(item.get("mean_floor") or -1.0), reverse=True)
    center, sigma = _weighted_center(rows, top_n=8)
    groups: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()

    def add(name: str, params: dict[str, float], kind: str) -> None:
        clamped = _clamp_group(params)
        signature = _signature(clamped)
        if signature in seen:
            return
        seen.add(signature)
        groups.append({"name": name, "params": clamped, "kind": kind})

    for rank, row in enumerate(rows[:6], start=1):
        add(f"round1_top{rank:02d}", {key: float(row[key]) for key in BIAS_KEYS}, "anchor")
    add("top_center", center, "center")
    for key in BIAS_KEYS:
        add(f"center_{key}_up", {k: center[k] + (sigma[k] if k == key else 0.0) for k in BIAS_KEYS}, "axis")
        add(f"center_{key}_down", {k: center[k] - (sigma[k] if k == key else 0.0) for k in BIAS_KEYS}, "axis")

    rng = random.Random(20260523)
    sample_index = 0
    while len(groups) < group_count:
        params = {
            key: center[key] + rng.gauss(0.0, sigma[key] * 1.25)
            for key in BIAS_KEYS
        }
        sample_index += 1
        add(f"local_{sample_index:02d}", params, "local_normal")
    return groups[:group_count]


def build_round3_groups(rows: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda item: float(item.get("mean_floor") or -1.0), reverse=True)
    groups: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for row in sorted_rows:
        params = _clamp_group({key: float(row[key]) for key in BIAS_KEYS})
        signature = _signature(params)
        if signature in seen:
            continue
        seen.add(signature)
        groups.append({"name": f"top{len(groups) + 1:02d}_{row.get('name', 'candidate')}", "params": params, "kind": "finalist"})
        if len(groups) >= top_n:
            break
    return groups


def _group_dir_name(index: int, group: dict[str, Any]) -> str:
    params = group["params"]
    return (
        f"g{index:02d}_{group['name']}"
        f"_s{params['strength']:.2f}_b{params['block']:.2f}"
        f"_a{params['aoe']:.2f}_e{params['exhaust']:.2f}"
    )


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_results(output_dir: Path, filename_prefix: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_sorted = sorted(rows, key=lambda item: float(item.get("mean_floor") or -1.0), reverse=True)
    (output_dir / f"{filename_prefix}_results.json").write_text(
        json.dumps(rows_sorted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / f"{filename_prefix}_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "group",
                "name",
                "kind",
                "strength",
                "block",
                "aoe",
                "exhaust",
                "mean_floor",
                "win_count",
                "timeout_count",
                "error_count",
                "seconds",
                "output_dir",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows_sorted, start=1):
            writer.writerow({"rank": rank, **row})
    return rows_sorted


def _existing_count(summary: dict[str, Any] | None) -> int:
    if not summary:
        return 0
    try:
        return int(summary.get("count") or 0)
    except (TypeError, ValueError):
        return 0


def _run_eval_group(
    *,
    repo_root: Path,
    group_dir: Path,
    params: dict[str, float],
    args: argparse.Namespace,
    count: int,
    stage_name: str,
) -> dict[str, Any] | None:
    summary_path = group_dir / "summary.json"
    summary = _load_summary(summary_path)
    if _existing_count(summary) >= count and int((summary or {}).get("error_count") or 0) == 0:
        return summary

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["SPIRECOMM_CARD_SCORE_BIAS_ENABLED"] = "1"
    env["SPIRECOMM_CARD_ARCHETYPE_BIAS_ENABLED"] = "1"
    for key, env_key in ENV_KEYS.items():
        env[env_key] = f"{params[key]:.2f}"

    cmd = [
        sys.executable,
        "-u",
        str(repo_root / "evaluate_v3_rollout_batch.py"),
        "--output-dir",
        str(group_dir),
        "--seed-start",
        str(int(args.seed_start)),
        "--count",
        str(int(count)),
        "--workers",
        str(int(args.workers)),
        "--mean-floor-only",
        "--combat-selector",
        str(args.combat_selector),
        "--v3-combat-model",
        str(args.v3_combat_model),
        "--summary-interval",
        str(int(args.summary_interval)),
    ]
    if args.resume:
        cmd.append("--resume")
    if str(args.combat_device):
        cmd.extend(["--combat-device", str(args.combat_device)])
    if str(args.preload_selectors):
        cmd.extend(["--preload-selectors", str(args.preload_selectors)])

    group_dir.mkdir(parents=True, exist_ok=True)
    log_path = group_dir / f"{stage_name}.log"
    if args.dry_run:
        print(" ".join(cmd), flush=True)
        return summary
    with log_path.open("a", encoding="utf-8") as log_handle:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise SystemExit(f"{stage_name} failed in {group_dir}; see {log_path}")
    return _load_summary(summary_path)


def run_stage(
    *,
    repo_root: Path,
    output_dir: Path,
    stage_dir: Path,
    stage_name: str,
    groups: list[dict[str, Any]],
    args: argparse.Namespace,
    count: int,
) -> list[dict[str, Any]]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / f"{stage_name}_params.json").write_text(
        json.dumps(
            {
                "bias_keys": list(BIAS_KEYS),
                "ranges": RANGES,
                "seed_start": int(args.seed_start),
                "count": int(count),
                "workers": int(args.workers),
                "combat_device": str(args.combat_device),
                "preload_selectors": str(args.preload_selectors),
                "v3_combat_model": str(args.v3_combat_model),
                "groups": groups,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, group in enumerate(groups, start=1):
        params = group["params"]
        group_dir = stage_dir / _group_dir_name(index, group)
        summary = _load_summary(group_dir / "summary.json")
        verb = "skip completed" if _existing_count(summary) >= count else "run"
        print(
            f"[{stage_name} {index:02d}/{len(groups)}] {verb} {group_dir.name} "
            f"target_count={count} s={params['strength']:.2f} b={params['block']:.2f} "
            f"a={params['aoe']:.2f} e={params['exhaust']:.2f}",
            flush=True,
        )
        summary = _run_eval_group(
            repo_root=repo_root,
            group_dir=group_dir,
            params=params,
            args=args,
            count=count,
            stage_name=stage_name,
        )
        if summary is None:
            continue
        row = {
            "group": index,
            "name": group["name"],
            "kind": group["kind"],
            "strength": params["strength"],
            "block": params["block"],
            "aoe": params["aoe"],
            "exhaust": params["exhaust"],
            "mean_floor": summary.get("mean_floor"),
            "win_count": summary.get("win_count"),
            "timeout_count": summary.get("timeout_count"),
            "error_count": summary.get("error_count"),
            "seconds": summary.get("seconds"),
            "output_dir": str(group_dir),
        }
        rows.append(row)
        ranked = _write_results(output_dir, stage_name, rows)
        best = ranked[0] if ranked else row
        print(
            f"[{stage_name} {index:02d}/{len(groups)}] done mean={float(row.get('mean_floor') or 0.0):.2f} "
            f"wins={row.get('win_count')} best={float(best.get('mean_floor') or 0.0):.2f} "
            f"elapsed_total={time.time() - started:.1f}s",
            flush=True,
        )
    return _write_results(output_dir, stage_name, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-round sweep for card archetype bias coefficients.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_runs/card_archetype_bias_sweep_v5"))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--round1-count", type=int, default=100)
    parser.add_argument("--round1-extend-count", type=int, default=300)
    parser.add_argument("--round2-count", type=int, default=300)
    parser.add_argument("--round2-groups", type=int, default=24)
    parser.add_argument("--round3-count", type=int, default=600)
    parser.add_argument("--round3-top", type=int, default=6)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SPIRECOMM_ARCHETYPE_SWEEP_WORKERS", "10")))
    parser.add_argument("--v3-combat-model", type=Path, default=Path("models/cache/download8_corrected_vocab/v5_dual_semantic_legacy_gate.pt"))
    parser.add_argument("--combat-selector", default="v3-candidate", choices=["legacy-slot", "v3-candidate", "v3-teacher"])
    parser.add_argument("--combat-device", default=os.environ.get("SPIRECOMM_EVAL_COMBAT_DEVICE", "auto"))
    parser.add_argument("--preload-selectors", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--summary-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after", choices=["round1_100", "round1_300", "round2_300", "round3_600", "all"], default="all")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    round1_groups = build_round1_groups()
    legacy_params = output_dir / "round1_params.json"
    if not legacy_params.exists():
        legacy_params.write_text(
            json.dumps({"bias_keys": list(BIAS_KEYS), "ranges": RANGES, "groups": round1_groups}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Backward-compatible layout: round1 lives directly under output_dir so
    # earlier interrupted runs can resume without copying data.
    round1_100 = run_stage(
        repo_root=repo_root,
        output_dir=output_dir,
        stage_dir=output_dir,
        stage_name="round1_100",
        groups=round1_groups,
        args=args,
        count=int(args.round1_count),
    )
    if args.stop_after == "round1_100":
        return

    round1_300 = run_stage(
        repo_root=repo_root,
        output_dir=output_dir,
        stage_dir=output_dir,
        stage_name="round1_300",
        groups=round1_groups,
        args=args,
        count=int(args.round1_extend_count),
    )
    if args.stop_after == "round1_300":
        return

    round2_groups = build_round2_groups(round1_300, group_count=int(args.round2_groups))
    round2_300 = run_stage(
        repo_root=repo_root,
        output_dir=output_dir,
        stage_dir=output_dir / "round2",
        stage_name="round2_300",
        groups=round2_groups,
        args=args,
        count=int(args.round2_count),
    )
    if args.stop_after == "round2_300":
        return

    round3_groups = build_round3_groups(round1_300 + round2_300, top_n=int(args.round3_top))
    run_stage(
        repo_root=repo_root,
        output_dir=output_dir,
        stage_dir=output_dir / "round3",
        stage_name="round3_600",
        groups=round3_groups,
        args=args,
        count=int(args.round3_count),
    )


if __name__ == "__main__":
    main()
