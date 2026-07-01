#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _iter_decisions(root: Path):
    hard_path = root / "hard_decisions.jsonl"
    if hard_path.exists():
        yield from _iter_jsonl(hard_path)
        return
    decision_dir = root / "decisions"
    for path in sorted(decision_dir.glob("seed_*.jsonl")):
        yield from _iter_jsonl(path)


def _card_key_from_action(action: dict[str, Any]) -> str:
    card = action.get("card") if isinstance(action.get("card"), dict) else {}
    return str(
        action.get("card_id")
        or card.get("card_id")
        or card.get("id")
        or action.get("name")
        or card.get("name")
        or "UNKNOWN"
    )


def _bucket_floor(floor: int) -> str:
    if floor <= 16:
        return "act1"
    if floor <= 33:
        return "act2"
    if floor <= 50:
        return "act3"
    return "win_or_act4"


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize collected non-combat hard roots.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = list(_iter_decisions(args.root))
    hard_rows = [row for row in rows if row.get("is_hard")]
    source_counts = Counter(str(row.get("source") or "") for row in rows)
    hard_source_counts = Counter(str(row.get("source") or "") for row in hard_rows)
    final_floor_buckets = Counter(_bucket_floor(int(row.get("final_floor") or 0)) for row in hard_rows)
    source_floor_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in hard_rows:
        source_floor_buckets[str(row.get("source") or "")][_bucket_floor(int(row.get("final_floor") or 0))] += 1

    margins_by_source: dict[str, list[float]] = defaultdict(list)
    for row in hard_rows:
        margin = _finite_float(row.get("best_margin"))
        if margin is not None:
            margins_by_source[str(row.get("source") or "")].append(margin)

    chosen_cards = Counter()
    candidate_cards = Counter()
    hard_card_rewards: list[dict[str, Any]] = []
    hard_upgrades: list[dict[str, Any]] = []
    for row in hard_rows:
        source = str(row.get("source") or "")
        action = row.get("action") if isinstance(row.get("action"), dict) else {}
        if source in {"card_reward", "card_reward_skip"}:
            if source == "card_reward":
                chosen_cards[_card_key_from_action(action)] += 1
            else:
                chosen_cards["SKIP"] += 1
            for candidate in row.get("candidates") or []:
                if isinstance(candidate, dict) and candidate.get("kind") == "card_reward":
                    candidate_cards[_card_key_from_action(candidate)] += 1
                elif isinstance(candidate, dict) and candidate.get("kind") in {"skip", "proceed"}:
                    candidate_cards["SKIP"] += 1
            hard_card_rewards.append(row)
        if source == "upgrade_target":
            chosen_cards[f"UPGRADE:{_card_key_from_action(action)}"] += 1
            hard_upgrades.append(row)

    source_margin_summary = {
        source: {
            "count": len(values),
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }
        for source, values in sorted(margins_by_source.items())
        if values
    }

    def row_brief(row: dict[str, Any]) -> dict[str, Any]:
        action = row.get("action") if isinstance(row.get("action"), dict) else {}
        return {
            "root_id": row.get("root_id"),
            "seed": row.get("seed"),
            "floor": row.get("floor"),
            "source": row.get("source"),
            "action": _card_key_from_action(action) if row.get("source") in {"card_reward", "upgrade_target"} else action.get("name"),
            "best_margin": row.get("best_margin"),
            "future_floor_delta": row.get("future_floor_delta"),
            "final_floor": row.get("final_floor"),
            "final_dead": row.get("final_dead"),
            "env_blob_path": row.get("env_blob_path"),
            "flags": {
                key: bool(row.get(key))
                for key in (
                    "hard_low_margin",
                    "hard_chosen_not_best",
                    "hard_bad_final",
                    "hard_died_soon",
                    "hard_missing_scores",
                )
            },
        }

    low_margin_examples = sorted(
        (row for row in hard_rows if _finite_float(row.get("best_margin")) is not None),
        key=lambda row: (_finite_float(row.get("best_margin")) or 999999.0, int(row.get("final_floor") or 999)),
    )[: args.top]
    bad_final_examples = sorted(
        hard_rows,
        key=lambda row: (int(row.get("final_floor") or 999), int(row.get("floor") or 999)),
    )[: args.top]
    blob_rows = [row for row in hard_rows if row.get("env_blob_path")]

    summary = {
        "root": str(args.root),
        "decision_rows": len(rows),
        "hard_rows": len(hard_rows),
        "source_counts": dict(source_counts.most_common()),
        "hard_source_counts": dict(hard_source_counts.most_common()),
        "hard_final_floor_buckets": dict(final_floor_buckets.most_common()),
        "hard_source_floor_buckets": {
            source: dict(counter.most_common()) for source, counter in sorted(source_floor_buckets.items())
        },
        "source_margin_summary": source_margin_summary,
        "chosen_card_counts": dict(chosen_cards.most_common(args.top)),
        "candidate_card_counts": dict(candidate_cards.most_common(args.top)),
        "card_reward_hard_count": len(hard_card_rewards),
        "upgrade_hard_count": len(hard_upgrades),
        "env_blob_hard_count": len(blob_rows),
        "low_margin_examples": [row_brief(row) for row in low_margin_examples],
        "bad_final_examples": [row_brief(row) for row in bad_final_examples],
        "blob_examples": [row_brief(row) for row in blob_rows[: args.top]],
    }
    output_path = args.output or (args.root / "hard_analysis.json")
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
