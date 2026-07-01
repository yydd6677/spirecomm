#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from spirecomm.ai.card_reward_model import (
    DEFAULT_STRONG_CARD_BIAS_BOOST,
    DEFAULT_WEAK_CARD_BIAS_TIER1,
    DEFAULT_WEAK_CARD_BIAS_TIER2,
    canonical_card_key,
)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _candidate_card(action: dict[str, Any]) -> dict[str, Any] | None:
    card = action.get("card")
    if isinstance(card, dict):
        return card
    if str(action.get("kind") or "") in {"skip", "proceed"}:
        return None
    payload = {
        key: action.get(key)
        for key in ("card_id", "id", "name", "type", "rarity", "cost", "base_cost", "upgrades")
        if key in action
    }
    return payload or None


def _branch_score(branch: dict[str, Any]) -> float | None:
    result = branch.get("result") if isinstance(branch.get("result"), dict) else {}
    value = result.get("branch_score")
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or score <= -999000.0:
        return None
    return score


def _default_biases() -> dict[str, float]:
    biases = {key: -0.6 for key in DEFAULT_WEAK_CARD_BIAS_TIER1}
    biases.update({key: -1.0 for key in DEFAULT_WEAK_CARD_BIAS_TIER2})
    biases.update({key: 0.6 for key in DEFAULT_STRONG_CARD_BIAS_BOOST})
    return biases


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small static card-score bias from non-combat branch labels.")
    parser.add_argument("--branch-label-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--sources", default="card_reward,card_reward_skip")
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--score-unit", type=float, default=100.0)
    parser.add_argument("--shrink-k", type=float, default=16.0)
    parser.add_argument("--min-count", type=int, default=6)
    parser.add_argument("--clamp", type=float, default=0.30)
    parser.add_argument("--include-default-biases", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    sources = {token.strip() for token in str(args.sources).split(",") if token.strip()}
    values: dict[str, list[float]] = defaultdict(list)
    root_count = 0
    for label in _iter_jsonl(args.branch_label_dir / "branch_labels.jsonl"):
        if str(label.get("source") or "") not in sources:
            continue
        branches = list(label.get("branches") or [])
        scored: list[tuple[dict[str, Any], float]] = []
        for branch in branches:
            candidate = branch.get("candidate") if isinstance(branch.get("candidate"), dict) else {}
            score = _branch_score(branch)
            if score is not None:
                scored.append((candidate, score))
        if len(scored) < 2:
            continue
        root_mean = mean(score for _candidate, score in scored)
        root_count += 1
        for candidate, score in scored:
            card = _candidate_card(candidate)
            if not card:
                continue
            key = canonical_card_key(card)
            if not key:
                continue
            values[key].append((score - root_mean) / max(1e-6, float(args.score_unit)))

    learned: dict[str, float] = {}
    details: list[dict[str, Any]] = []
    for key, deltas in sorted(values.items()):
        count = len(deltas)
        if count < int(args.min_count):
            continue
        raw = mean(deltas)
        shrink = count / (count + max(0.0, float(args.shrink_k)))
        bias = raw * float(args.scale) * shrink
        clamp = abs(float(args.clamp))
        bias = max(-clamp, min(clamp, bias))
        if abs(bias) < 0.01:
            continue
        learned[key] = round(float(bias), 4)
        details.append(
            {
                "card": key,
                "count": count,
                "mean_delta_floor_units": raw,
                "shrink": shrink,
                "learned_bias": learned[key],
            }
        )

    final = _default_biases() if bool(args.include_default_biases) else {}
    for key, value in learned.items():
        final[key] = round(float(final.get(key, 0.0)) + float(value), 4)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "branch_label_dir": str(args.branch_label_dir),
        "root_count": root_count,
        "learned_count": len(learned),
        "final_count": len(final),
        "scale": float(args.scale),
        "score_unit": float(args.score_unit),
        "shrink_k": float(args.shrink_k),
        "min_count": int(args.min_count),
        "clamp": float(args.clamp),
        "learned": sorted(details, key=lambda item: abs(float(item["learned_bias"])), reverse=True),
    }
    summary_path = args.output_summary or args.output_json.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
