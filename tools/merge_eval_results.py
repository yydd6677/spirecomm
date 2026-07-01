#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[int, dict[str, Any]]:
    source = path / "results.jsonl" if path.is_dir() else path
    rows: dict[int, dict[str, Any]] = {}
    if source.exists() and source.suffix == ".jsonl":
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[int(row["seed"])] = row
        return rows
    source = path / "results.json" if path.is_dir() else path
    if source.exists():
        payload = json.loads(source.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else payload.get("results", [])
        for row in items:
            rows[int(row["seed"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge eval result dirs/files into one results.jsonl directory.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    merged: dict[int, dict[str, Any]] = {}
    for path in args.inputs:
        merged.update(_load(path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "results.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for seed in sorted(merged):
            handle.write(json.dumps(merged[seed], ensure_ascii=False) + "\n")
    floors = [int(row.get("floor") or 0) for row in merged.values()]
    summary = {
        "count": len(merged),
        "mean_floor": (sum(floors) / len(floors)) if floors else 0.0,
        "min_seed": min(merged) if merged else None,
        "max_seed": max(merged) if merged else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
