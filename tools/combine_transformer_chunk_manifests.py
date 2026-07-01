#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spirecomm.ai.torch_compat import require_torch


def _load(path: Path) -> dict[str, Any]:
    torch = require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("chunks"), list):
        raise SystemExit(f"not a chunked transformer tensor manifest: {path}")
    return payload


def _parse_source(raw: str) -> tuple[Path, int]:
    if ":" not in raw:
        return Path(raw), 1
    path_raw, repeat_raw = raw.rsplit(":", 1)
    try:
        repeat = int(repeat_raw)
    except ValueError:
        return Path(raw), 1
    return Path(path_raw), max(0, repeat)


def _assert_compatible(base: dict[str, Any], current: dict[str, Any], path: Path) -> None:
    keys = [
        "tensor_dataset_schema",
        "feature_schema",
        "feature_dims",
        "token_schema_version",
        "token_schema",
        "entity_vocab",
    ]
    for key in keys:
        if base.get(key) != current.get(key):
            raise SystemExit(f"incompatible {key} in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine chunked transformer tensor manifests without recaching existing chunks.")
    parser.add_argument("--source", action="append", required=True, help="Manifest path, optionally suffixed with :repeat.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-note", default="")
    args = parser.parse_args()

    sources = [_parse_source(raw) for raw in args.source]
    loaded: list[tuple[Path, int, dict[str, Any]]] = []
    for path, repeat in sources:
        payload = _load(path)
        if loaded:
            _assert_compatible(loaded[0][2], payload, path)
        loaded.append((path, repeat, payload))
    if not loaded:
        raise SystemExit("no sources")

    base_payload = loaded[0][2]
    chunks: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    root_count = 0
    candidate_count = 0
    shard_count = 0
    for path, repeat, payload in loaded:
        payload_chunks = [dict(chunk) for chunk in payload.get("chunks") or []]
        payload_roots = sum(int(chunk.get("root_count") or 0) for chunk in payload_chunks)
        payload_candidates = sum(int(chunk.get("candidate_count") or 0) for chunk in payload_chunks)
        for _ in range(max(0, repeat)):
            chunks.extend(dict(chunk) for chunk in payload_chunks)
            root_count += payload_roots
            candidate_count += payload_candidates
            shard_count += len(payload_chunks)
        source_summaries.append(
            {
                "path": str(path),
                "repeat": int(repeat),
                "source_root_count": int(payload_roots),
                "source_candidate_count": int(payload_candidates),
                "source_chunk_count": len(payload_chunks),
            }
        )

    output = {
        key: base_payload.get(key)
        for key in (
            "tensor_dataset_schema",
            "feature_schema",
            "feature_dims",
            "token_schema_version",
            "token_schema",
            "entity_vocab",
        )
    }
    metadata = dict(base_payload.get("metadata") or {})
    metadata.update(
        {
            "root_count": int(root_count),
            "candidate_count": int(candidate_count),
            "shard_count": int(shard_count),
            "chunk_count": len(chunks),
            "combined_from": source_summaries,
            "metadata_note": str(args.metadata_note or ""),
        }
    )
    output["metadata"] = metadata
    output["chunks"] = chunks
    output["source_shards"] = [str(chunk.get("source_shard") or "") for chunk in chunks]
    output["root_ids"] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch = require_torch()
    torch.save(output, args.output)
    (args.output.with_suffix(args.output.suffix + ".summary.json")).write_text(
        json.dumps({"metadata": metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"combined manifests -> {args.output} roots={root_count} candidates={candidate_count} chunks={len(chunks)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
