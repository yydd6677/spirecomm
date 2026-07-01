#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge non-combat hard-root collection directories.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("input_roots", type=Path, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "decisions").mkdir(parents=True, exist_ok=True)
    (output_root / "env_blobs").mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    hard_rows: list[dict] = []
    configs: list[dict] = []
    for input_root in args.input_roots:
        input_root = input_root.resolve()
        config_path = input_root / "config.json"
        if config_path.exists():
            configs.append({"input_root": str(input_root), "config": json.loads(config_path.read_text(encoding="utf-8"))})
        for row in _iter_jsonl(input_root / "results.jsonl") or []:
            results.append(row)
        for path in sorted((input_root / "decisions").glob("seed_*.jsonl")):
            destination = output_root / "decisions" / path.name
            if destination.exists():
                raise FileExistsError(f"duplicate decision file: {destination}")
            shutil.copy2(path, destination)
        for path in sorted((input_root / "env_blobs").glob("*.pkl.gz")):
            destination = output_root / "env_blobs" / path.name
            if destination.exists():
                raise FileExistsError(f"duplicate env blob: {destination}")
            shutil.copy2(path, destination)
        for row in _iter_jsonl(input_root / "hard_decisions.jsonl") or []:
            hard_rows.append(row)

    results.sort(key=lambda row: int(row.get("seed") or 0))
    hard_rows.sort(key=lambda row: str(row.get("root_id") or ""))
    _write_jsonl(output_root / "results.jsonl", results)
    _write_jsonl(output_root / "hard_decisions.jsonl", hard_rows)
    summary = {
        "input_roots": [str(path) for path in args.input_roots],
        "result_count": len(results),
        "hard_decision_count": len(hard_rows),
        "env_blob_count": len(list((output_root / "env_blobs").glob("*.pkl.gz"))),
        "configs": configs,
    }
    (output_root / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
