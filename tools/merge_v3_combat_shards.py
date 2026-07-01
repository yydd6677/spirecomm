#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spirecomm.ai.v3_combat_dataset import V3CombatLabeledRoot, load_shard, save_shard


def _source_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for directory in args.input_dir or []:
        paths.extend(sorted(Path(directory).glob("*.pt")))
    for list_path in args.input_list or []:
        paths.extend(
            Path(line.strip())
            for line in Path(list_path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    paths.extend(Path(path) for path in args.input or [])
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge many small v3 combat labeled shards into larger shards.")
    parser.add_argument("--input-dir", action="append", type=Path, default=[])
    parser.add_argument("--input-list", action="append", type=Path, default=[])
    parser.add_argument("--input", nargs="*", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roots-per-shard", type=int, default=512)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--source", default="merged")
    args = parser.parse_args()

    paths = _source_paths(args)
    if not paths:
        raise SystemExit("no input shards")
    roots_per_shard = max(1, int(args.roots_per_shard))
    if args.clear_output and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[str] = []
    buffer: list[V3CombatLabeledRoot] = []
    total_roots = 0
    shard_index = 0
    source_metadata: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buffer, shard_index
        if not buffer:
            return
        out = args.output_dir / f"shard_{shard_index:05d}.pt"
        save_shard(
            out,
            buffer,
            metadata={
                "schema": "v3_combat_merged_shards_v1",
                "source": str(args.source),
                "root_count": len(buffer),
                "shard_index": int(shard_index),
                "input_shard_count": len(paths),
            },
        )
        output_paths.append(str(out))
        shard_index += 1
        buffer = []

    for path in paths:
        payload = load_shard(path)
        roots = list(payload.get("roots") or [])
        source_metadata.append(
            {
                "path": str(path),
                "root_count": len(roots),
                "metadata": payload.get("metadata") or {},
            }
        )
        for root in roots:
            buffer.append(root)
            total_roots += 1
            if len(buffer) >= roots_per_shard:
                flush()
    flush()
    summary = {
        "schema": "v3_combat_merge_summary_v1",
        "input_shard_count": len(paths),
        "output_shard_count": len(output_paths),
        "root_count": total_roots,
        "roots_per_shard": roots_per_shard,
        "output_shards": output_paths,
        "sources": source_metadata,
    }
    (args.output_dir / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"merged input_shards={len(paths)} roots={total_roots} output_shards={len(output_paths)} dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
