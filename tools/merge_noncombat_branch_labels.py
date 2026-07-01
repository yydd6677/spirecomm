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


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge non-combat branch-label directories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("input_dirs", type=Path, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels: list[dict] = []
    configs: list[dict] = []
    seen: set[str] = set()
    for input_dir in args.input_dirs:
        config_path = input_dir / "config.json"
        if config_path.exists():
            configs.append({"input_dir": str(input_dir), "config": json.loads(config_path.read_text(encoding="utf-8"))})
        for row in _iter_jsonl(input_dir / "branch_labels.jsonl") or []:
            root_id = str(row.get("root_id") or "")
            if root_id in seen:
                raise ValueError(f"duplicate root_id: {root_id}")
            seen.add(root_id)
            labels.append(row)
    labels.sort(key=lambda row: str(row.get("root_id") or ""))
    with (output_dir / "branch_labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in labels:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "input_dirs": [str(path) for path in args.input_dirs],
        "label_count": len(labels),
        "configs": configs,
    }
    (output_dir / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
