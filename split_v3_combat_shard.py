#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spirecomm.ai.v3_combat_dataset import load_shard, save_shard


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a v3 combat labeled shard into smaller shards.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="shard")
    parser.add_argument("--roots-per-shard", type=int, default=256)
    parser.add_argument("--clear-output", action="store_true")
    args = parser.parse_args()

    roots_per_shard = max(1, int(args.roots_per_shard))
    payload = load_shard(args.input)
    roots = list(payload.get("roots") or [])
    if not roots:
        raise SystemExit(f"no roots in {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_output:
        for old_path in args.output_dir.glob(f"{args.prefix}_*.pt"):
            old_path.unlink()
        for old_path in args.output_dir.glob("summary.json"):
            old_path.unlink()

    source_metadata = dict(payload.get("metadata") or {})
    paths: list[str] = []
    for index, start in enumerate(range(0, len(roots), roots_per_shard)):
        chunk_roots = roots[start : start + roots_per_shard]
        out_path = args.output_dir / f"{args.prefix}_{index:05d}.pt"
        metadata: dict[str, Any] = {
            "schema": "v3_combat_split_shard_v1",
            "source": str(args.input),
            "source_metadata": source_metadata,
            "start": start,
            "end": start + len(chunk_roots),
            "root_count": len(chunk_roots),
            "roots_per_shard": roots_per_shard,
        }
        save_shard(out_path, chunk_roots, metadata=metadata)
        paths.append(str(out_path))

    summary = {
        "schema": "v3_combat_split_shard_summary_v1",
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "root_count": len(roots),
        "roots_per_shard": roots_per_shard,
        "shard_count": len(paths),
        "paths": paths,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
