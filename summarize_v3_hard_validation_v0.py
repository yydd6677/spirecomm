#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "model",
    "top1",
    "mean_regret",
    "rmse_regret",
    "high_conf_bad",
    "potion_top1",
    "miss_teacher_potion",
    "false_potion_top",
    "baseline_break",
    "baseline_recover",
    "regret_ge_25",
    "gap5_miss",
    "card_mismatch",
    "hard_roots",
]


def _load_run(path: Path) -> dict[str, Any] | None:
    summary_path = path / "summary.json"
    hard_summary_path = path / "hard_root_summary.json"
    if not summary_path.exists() or not hard_summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hard_summary = json.loads(hard_summary_path.read_text(encoding="utf-8"))
    metrics = dict(summary.get("metrics") or {})
    overall = dict(metrics.get("overall") or {})
    potion = dict(metrics.get("teacher_top_kind:potion") or {})
    baseline_correct = dict(metrics.get("baseline_correct") or {})
    baseline_wrong = dict(metrics.get("baseline_wrong") or {})
    category_counts = dict(hard_summary.get("category_counts") or {})
    return {
        "model": path.name,
        "top1": overall.get("top1_accuracy"),
        "mean_regret": overall.get("mean_regret"),
        "rmse_regret": overall.get("rmse_regret"),
        "high_conf_bad": overall.get("high_conf_disagreement_rate"),
        "potion_top1": potion.get("top1_accuracy"),
        "miss_teacher_potion": potion.get("missed_teacher_potion_rate"),
        "false_potion_top": overall.get("false_potion_top_rate"),
        "baseline_break": baseline_correct.get("model_breaks_baseline_correct_rate"),
        "baseline_recover": baseline_wrong.get("model_recovers_baseline_wrong_rate"),
        "regret_ge_25": category_counts.get("regret_ge_25", 0),
        "gap5_miss": category_counts.get("high_teacher_gap_miss_ge_5", 0),
        "card_mismatch": category_counts.get("card_to_card_mismatch", 0),
        "hard_roots": hard_summary.get("hard_root_count", 0),
    }


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_tsv(rows: list[dict[str, Any]], output: Path | None) -> str:
    lines = ["\t".join(FIELDS)]
    for row in rows:
        lines.append("\t".join(_format_value(row.get(field)) for field in FIELDS))
    text = "\n".join(lines) + "\n"
    if output is not None:
        output.write_text(text, encoding="utf-8")
    return text


def _write_markdown(rows: list[dict[str, Any]], output: Path | None) -> str:
    lines = ["| " + " | ".join(FIELDS) + " |", "| " + " | ".join(["---"] * len(FIELDS)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(field)) for field in FIELDS) + " |")
    text = "\n".join(lines) + "\n"
    if output is not None:
        output.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize hard validation v0 output directories.")
    parser.add_argument("paths", nargs="+", type=Path, help="Run dirs or parent dirs containing run subdirectories.")
    parser.add_argument("--sort-by", default="mean_regret", choices=FIELDS)
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.paths:
        if (path / "summary.json").exists():
            row = _load_run(path)
            if row is not None:
                rows.append(row)
            continue
        for child in sorted(path.iterdir() if path.exists() else []):
            if child.is_dir():
                row = _load_run(child)
                if row is not None:
                    rows.append(row)
    rows.sort(key=lambda row: (row.get(args.sort_by) is None, row.get(args.sort_by)), reverse=bool(args.descending))
    if args.format == "markdown":
        text = _write_markdown(rows, args.output)
    else:
        text = _write_tsv(rows, args.output)
    print(text, end="")


if __name__ == "__main__":
    main()
