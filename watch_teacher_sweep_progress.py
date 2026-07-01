#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROUND_ORDER = (
    "round0_default",
    "round1_seed60",
    "round1_seed100",
    "round2_seed100",
    "round2_seed200",
    "round3_seed300",
    "round4_seed600",
)


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _format_summary(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "no summary yet"
    return (
        f"count={summary.get('count')} mean_floor={float(summary.get('mean_floor') or 0.0):.2f} "
        f"wins={summary.get('win_count')} deaths={summary.get('death_count')} errors={summary.get('error_count')} "
        f"seconds={float(summary.get('seconds') or 0.0):.1f}"
    )


def _summarize_results_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    count = 0
    floor_sum = 0.0
    wins = 0
    deaths = 0
    errors = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                count += 1
                floor_sum += float(record.get("floor") or 0.0)
                wins += int(bool(record.get("won")))
                deaths += int(bool(record.get("dead")))
                errors += int(bool(record.get("error")))
    except Exception:
        return None
    if count <= 0:
        return None
    return {
        "count": count,
        "mean_floor": floor_sum / count,
        "win_count": wins,
        "death_count": deaths,
        "error_count": errors,
        "seconds": 0.0,
    }


def _best_candidate_summary(candidate_dir: Path) -> dict[str, Any] | None:
    from_results = _summarize_results_jsonl(candidate_dir / "results.jsonl")
    from_partial = _load_json(candidate_dir / "summary_partial.json")
    from_complete = _load_json(candidate_dir / "summary.json")
    summaries = [
        item for item in (from_complete, from_results, from_partial)
        if isinstance(item, dict)
    ]
    if not summaries:
        return None
    return max(summaries, key=lambda item: int(item.get("count") or 0))


def _aggregate_counts(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    candidate_ids: set[str] = set()
    rows = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                rows += 1
                candidate_id = str(record.get("_candidate_id") or record.get("candidate_id") or "")
                if candidate_id:
                    candidate_ids.add(candidate_id)
    except Exception:
        return 0, 0
    return rows, len(candidate_ids)


def _round_dirs(output_dir: Path) -> list[Path]:
    evals = output_dir / "evals"
    if not evals.exists():
        return []
    ordered = [evals / name for name in ROUND_ORDER if (evals / name).exists()]
    extras = sorted(path for path in evals.iterdir() if path.is_dir() and path.name not in ROUND_ORDER)
    return ordered + extras


def _render(output_dir: Path, top: int) -> str:
    lines: list[str] = []
    lines.append(time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append(f"output_dir={output_dir}")
    for round_dir in _round_dirs(output_dir):
        candidates = sorted(path for path in round_dir.iterdir() if path.is_dir())
        complete = sum(1 for path in candidates if (path / "summary.json").exists())
        partials = [
            path for path in candidates
            if (
                (path / "summary_partial.json").exists()
                or (path / "results.jsonl").exists()
            ) and not (path / "summary.json").exists()
        ]
        aggregate_rows, aggregate_candidates = _aggregate_counts(round_dir / "results.jsonl")
        suffix = (
            f" aggregate_rows={aggregate_rows} aggregate_candidates={aggregate_candidates}"
            if aggregate_rows
            else ""
        )
        lines.append(
            f"{round_dir.name}: complete={complete}/{len(candidates)} "
            f"running_or_partial={len(partials)}{suffix}"
        )
        for path in partials[:3]:
            lines.append(f"  partial {path.name}: {_format_summary(_best_candidate_summary(path))}")
    leaderboard = _load_json(output_dir / "leaderboard_latest.json")
    if isinstance(leaderboard, list) and leaderboard:
        lines.append(f"latest leaderboard top{min(top, len(leaderboard))}:")
        for item in leaderboard[:top]:
            lines.append(
                f"  #{item.get('rank')} {item.get('candidate_id')} "
                f"mean_floor={float(item.get('mean_floor') or 0.0):.2f} "
                f"wins={item.get('win_count')} p25={item.get('p25_floor')}"
            )
    else:
        lines.append("latest leaderboard: not available yet")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print or watch v3 teacher coefficient sweep progress.")
    parser.add_argument("--output-dir", type=Path, default=Path("teacher_sweep_runs/v3_teacher_coeff_sweep_v1"))
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        print(_render(args.output_dir, max(1, int(args.top))), flush=True)
        if args.once:
            return
        time.sleep(max(1.0, float(args.interval)))


if __name__ == "__main__":
    main()
