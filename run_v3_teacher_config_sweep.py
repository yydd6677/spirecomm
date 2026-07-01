#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


PARAM_RANGES: dict[str, tuple[float, float]] = {
    "hp_damage_weight": (0.5, 1.3),
    "monster_kill_weight": (5.0, 18.0),
    "combat_win_weight": (10.0, 35.0),
    "death_weight": (-80.0, -30.0),
    "hp_loss_weight": (-4.0, -0.8),
    "effective_block_weight": (0.4, 1.8),
    "raw_incoming_damage_reduction_weight": (0.2, 1.8),
    "playable_hand_count_delta_weight": (1.0, 9.0),
}

PARAM_BUCKETS: dict[str, dict[str, tuple[float, float]]] = {
    "base8": PARAM_RANGES,
    "A": {
        "play_card_constant": (-2.0, 2.0),
        "energy_spent_weight": (-1.2, 0.6),
        "skill_power_turn_constant": (4.0, 18.0),
        "turn_order_decay_per_card": (0.0, 0.5),
    },
    "C": {
        "monster_vulnerable_weight": (2.0, 8.0),
        "monster_weakened_weight": (0.5, 5.0),
        "monster_strength_weight": (2.0, 8.0),
        "monster_shackled_weight": (0.0, 2.0),
    },
    "D": {
        "potion_monster_room_reward_factor": (0.1, 0.8),
        "potion_elite_room_reward_factor": (0.8, 1.8),
        "potion_boss_room_reward_factor": (1.2, 3.0),
        "potion_cost_scale": (0.5, 1.5),
        "potion_buff_adjustment_scale": (0.5, 1.6),
        "potion_generation_adjustment_scale": (0.5, 1.8),
    },
}


def param_ranges_for_bucket(bucket: str = "base8", ranges_json: str = "") -> dict[str, tuple[float, float]]:
    if ranges_json.strip():
        payload = json.loads(ranges_json)
        if not isinstance(payload, dict):
            raise ValueError("--param-ranges-json must be a JSON object")
        ranges: dict[str, tuple[float, float]] = {}
        for name, raw_range in payload.items():
            if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                raise ValueError(f"Invalid range for {name!r}: expected [low, high]")
            low = float(raw_range[0])
            high = float(raw_range[1])
            if high < low:
                raise ValueError(f"Invalid range for {name!r}: high < low")
            ranges[str(name)] = (low, high)
        return ranges
    normalized = str(bucket or "base8").strip()
    for key, ranges in PARAM_BUCKETS.items():
        if normalized.lower() == key.lower():
            return dict(ranges)
    raise ValueError(f"Unknown param bucket {bucket!r}; choices: {', '.join(PARAM_BUCKETS)}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _worker_thread_env(thread_count: int) -> dict[str, str]:
    value = str(max(1, int(thread_count)))
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = value
    env.setdefault("OMP_DYNAMIC", "FALSE")
    return env


def _default_teacher_config() -> dict[str, Any]:
    sys.path.insert(0, str(_repo_root()))
    from spirecomm.ai.v3_combat_teacher import TeacherConfig

    defaults = asdict(TeacherConfig())
    values: dict[str, Any] = {}
    for name in PARAM_RANGES:
        current: Any = defaults
        for part in str(name).split("."):
            if not isinstance(current, dict) or part not in current:
                raise KeyError(f"Unknown TeacherConfig default key: {name}")
            current = current[part]
        values[name] = float(current)
    return values


def _latin_hypercube_sample(
    *,
    n: int,
    ranges: dict[str, tuple[float, float]],
    seed: int,
) -> list[dict[str, float]]:
    rng = random.Random(seed)
    columns: dict[str, list[float]] = {}
    for name, (low, high) in ranges.items():
        width = high - low
        values = [low + ((index + rng.random()) / n) * width for index in range(n)]
        rng.shuffle(values)
        columns[name] = values
    return [{name: columns[name][index] for name in ranges} for index in range(n)]


def _corner_sample(ranges: dict[str, tuple[float, float]]) -> list[dict[str, float]]:
    names = list(ranges)
    values_by_name = [[ranges[name][0], ranges[name][1]] for name in names]
    return [
        {name: float(value) for name, value in zip(names, values, strict=True)}
        for values in itertools.product(*values_by_name)
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_summary(path: Path) -> dict[str, Any] | None:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _score_key(result: dict[str, Any]) -> tuple[float, int, float, float]:
    summary = result["summary"]
    return (
        float(summary.get("mean_floor") or 0.0),
        int(summary.get("win_count") or 0),
        float(summary.get("p25_floor") or 0.0),
        -float(summary.get("error_count") or 0.0),
    )


def _rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=_score_key, reverse=True)


def _candidate_payload(candidate_id: str, params: dict[str, float], round_name: str) -> dict[str, Any]:
    return {
        "version": "v3_teacher_coeff_sweep_v1",
        "id": candidate_id,
        "round": round_name,
        "teacher_config": dict(params),
    }


def _run_eval(
    *,
    args: argparse.Namespace,
    round_name: str,
    candidate_id: str,
    params: dict[str, float],
    seed_count: int,
) -> dict[str, Any]:
    root = _repo_root()
    config_path = args.output_dir / "configs" / f"{candidate_id}.json"
    eval_dir = args.output_dir / "evals" / round_name / candidate_id
    _write_json(config_path, _candidate_payload(candidate_id, params, round_name))
    existing = _load_summary(eval_dir) if args.resume else None
    if existing is not None:
        print(
            f"[{round_name}] reuse {candidate_id} seeds={seed_count} "
            f"mean_floor={float(existing.get('mean_floor') or 0.0):.2f} wins={int(existing.get('win_count') or 0)}",
            flush=True,
        )
        summary = existing
    else:
        eval_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(root / "evaluate_v3_rollout_batch.py"),
            "--repo-root",
            str(root),
            "--output-dir",
            str(eval_dir),
            "--seed-start",
            str(args.seed_start),
            "--count",
            str(seed_count),
            "--workers",
            str(args.workers),
            "--combat-selector",
            "v3-teacher",
            "--teacher-config-path",
            str(config_path),
            "--device",
            "cpu",
            "--combat-device",
            "cpu",
            "--torch-threads",
            str(args.torch_threads),
            "--trace-mode",
            "none",
            "--no-progress-limit",
            "0",
            "--metrics-mode",
            args.metrics_mode,
            "--summary-interval",
            str(args.summary_interval),
            "--preload-selectors",
            "never",
            "--no-write-results-json",
        ]
        if args.resume:
            cmd.append("--resume")
        env = _worker_thread_env(args.blas_threads)
        started = time.time()
        print(f"[{round_name}] run {candidate_id} seeds={seed_count} workers={args.workers}", flush=True)
        subprocess.run(cmd, cwd=root, env=env, check=True)
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
        elapsed = time.time() - started
        print(
            f"[{round_name}] done {candidate_id} mean_floor={float(summary.get('mean_floor') or 0.0):.2f} "
            f"wins={int(summary.get('win_count') or 0)} seconds={elapsed:.1f}",
            flush=True,
        )
    result = {
        "round": round_name,
        "candidate_id": candidate_id,
        "seed_start": int(args.seed_start),
        "seed_count": int(seed_count),
        "params": dict(params),
        "eval_dir": str(eval_dir),
        "config_path": str(config_path),
        "summary": summary,
    }
    with (args.output_dir / "sweep_results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def _save_leaderboard(output_dir: Path, round_name: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    _write_json(output_dir / f"leaderboard_{round_name}.json", compact)
    _write_json(output_dir / "leaderboard_latest.json", compact)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only sweep for v3 teacher transition-reward coefficients.")
    parser.add_argument("--output-dir", type=Path, default=Path("teacher_sweep_runs/v3_teacher_coeff_sweep_v1"))
    parser.add_argument("--param-bucket", choices=sorted(PARAM_BUCKETS), default="base8")
    parser.add_argument("--param-ranges-json", default=os.environ.get("SPIRECOMM_TEACHER_SWEEP_PARAM_RANGES_JSON", ""))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--metrics-mode", choices=["full", "floor"], default="floor")
    parser.add_argument("--summary-interval", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260518)
    parser.add_argument(
        "--round1-sampler",
        choices=["latin-hypercube", "corners"],
        default="latin-hypercube",
        help="latin-hypercube covers interiors; corners evaluates the 2^8 low/high factorial design.",
    )
    parser.add_argument("--round0-count", type=int, default=300)
    parser.add_argument("--round1-size", type=int, default=256)
    parser.add_argument("--round1-count", type=int, default=60)
    parser.add_argument("--round2-top", type=int, default=32)
    parser.add_argument("--round2-count", type=int, default=100)
    parser.add_argument("--round3-top", type=int, default=16)
    parser.add_argument("--round3-count", type=int, default=300)
    parser.add_argument("--round4-top", type=int, default=6)
    parser.add_argument("--round4-count", type=int, default=600)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", help="Ignore existing summary.json files and re-run candidates.")
    args = parser.parse_args()
    if args.force:
        args.resume = False
    global PARAM_RANGES
    PARAM_RANGES = param_ranges_for_bucket(args.param_bucket, args.param_ranges_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output_dir / "sweep_config.json",
        {
            "param_ranges": PARAM_RANGES,
            "seed_start": args.seed_start,
            "workers": args.workers,
            "torch_threads": args.torch_threads,
            "blas_threads": args.blas_threads,
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
    round0 = [
        _run_eval(
            args=args,
            round_name="round0_default",
            candidate_id="default",
            params=default_params,
            seed_count=args.round0_count,
        )
    ]
    _save_leaderboard(args.output_dir, "round0_default", round0)

    if args.round1_sampler == "corners":
        round1_candidates = _corner_sample(PARAM_RANGES)
        if len(round1_candidates) != int(args.round1_size):
            print(
                f"[round1] corners sampler produced {len(round1_candidates)} candidates; "
                f"ignoring --round1-size={args.round1_size}",
                flush=True,
            )
    else:
        round1_candidates = _latin_hypercube_sample(n=args.round1_size, ranges=PARAM_RANGES, seed=args.random_seed)
    _write_json(
        args.output_dir / "round1_candidates.json",
        [
            {"candidate_id": f"r1_{index:03d}", "params": params}
            for index, params in enumerate(round1_candidates)
        ],
    )
    round1 = [
        _run_eval(
            args=args,
            round_name="round1_seed60",
            candidate_id=f"r1_{index:03d}",
            params=params,
            seed_count=args.round1_count,
        )
        for index, params in enumerate(round1_candidates)
    ]
    top_round1 = _save_leaderboard(args.output_dir, "round1_seed60", round1)[: args.round2_top]

    round2 = [
        _run_eval(
            args=args,
            round_name="round2_seed100",
            candidate_id=result["candidate_id"],
            params=result["params"],
            seed_count=args.round2_count,
        )
        for result in top_round1
    ]
    top_round2 = _save_leaderboard(args.output_dir, "round2_seed100", round2)[: args.round3_top]

    round3 = [
        _run_eval(
            args=args,
            round_name="round3_seed300",
            candidate_id=result["candidate_id"],
            params=result["params"],
            seed_count=args.round3_count,
        )
        for result in top_round2
    ]
    top_round3 = _save_leaderboard(args.output_dir, "round3_seed300", round3)[: args.round4_top]

    round4 = [
        _run_eval(
            args=args,
            round_name="round4_seed600",
            candidate_id=result["candidate_id"],
            params=result["params"],
            seed_count=args.round4_count,
        )
        for result in top_round3
    ]
    final = _save_leaderboard(args.output_dir, "round4_seed600", round4)
    _write_json(args.output_dir / "final_leaderboard.json", json.loads((args.output_dir / "leaderboard_round4_seed600.json").read_text(encoding="utf-8")))
    print(json.dumps(final[0], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
