#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_results(path: Path) -> dict[int, dict[str, Any]]:
    if path.is_dir():
        jsonl = path / "results.jsonl"
        pretty = path / "results.json"
    else:
        jsonl = path
        pretty = path
    by_seed: dict[int, dict[str, Any]] = {}
    if jsonl.name.endswith(".jsonl") or jsonl.exists():
        if jsonl.exists():
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                by_seed[int(row["seed"])] = row
            if by_seed:
                return by_seed
    if pretty.exists():
        payload = json.loads(pretty.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("results", [])
        for row in rows:
            by_seed[int(row["seed"])] = row
    return by_seed


def _floor(row: dict[str, Any]) -> int:
    return int(row.get("floor") or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired floor deltas between two eval result dirs/files.")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--changed-limit", type=int, default=80)
    parser.add_argument("--source", default="", help="Optional source key to count in candidate per-seed sources.")
    args = parser.parse_args()

    base = _load_results(args.base)
    cand = _load_results(args.candidate)
    common = sorted(set(base) & set(cand))
    if not common:
        print("no common seeds")
        return
    deltas = [(seed, _floor(base[seed]), _floor(cand[seed])) for seed in common]
    total = sum(cand_floor - base_floor for _, base_floor, cand_floor in deltas)
    up = sum(cand_floor > base_floor for _, base_floor, cand_floor in deltas)
    down = sum(cand_floor < base_floor for _, base_floor, cand_floor in deltas)
    same = len(deltas) - up - down
    base_mean = sum(base_floor for _, base_floor, _ in deltas) / len(deltas)
    cand_mean = sum(cand_floor for _, _, cand_floor in deltas) / len(deltas)
    print(
        f"n={len(deltas)} base={base_mean:.4f} candidate={cand_mean:.4f} "
        f"delta={total} mean_delta={total / len(deltas):+.4f} up/down/same={up}/{down}/{same}"
    )
    changed = [
        (seed, base_floor, cand_floor, cand_floor - base_floor)
        for seed, base_floor, cand_floor in deltas
        if cand_floor != base_floor
    ]
    if args.source:
        source_count = 0
        source_seeds: list[int] = []
        for seed in common:
            sources = cand[seed].get("sources") or {}
            if int(sources.get(args.source, 0) or 0) > 0:
                source_count += int(sources.get(args.source, 0) or 0)
                source_seeds.append(seed)
        print(f"source={args.source} source_count={source_count} seeds={source_seeds[: args.changed_limit]}")
    print(f"changed_count={len(changed)}")
    for row in changed[: max(0, int(args.changed_limit))]:
        print(row)


if __name__ == "__main__":
    main()
