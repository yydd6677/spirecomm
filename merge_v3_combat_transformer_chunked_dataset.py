#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from spirecomm.ai.torch_compat import require_torch


def _load(path: Path) -> dict[str, Any]:
    torch = require_torch()
    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")


def _schema_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    metadata = dict(payload.get("metadata") or {})
    return (
        payload.get("tensor_dataset_schema"),
        payload.get("feature_schema"),
        payload.get("token_schema_version"),
        json.dumps(payload.get("feature_dims") or {}, sort_keys=True),
        tuple(payload.get("entity_vocab") or []),
        metadata.get("sequence_length"),
        metadata.get("scalar_dim"),
        metadata.get("entity_vocab_size"),
        metadata.get("max_actions"),
        metadata.get("action_segment_width"),
    )


def _write_validation_sources(
    *,
    payload: dict[str, Any],
    output: Path,
    seed: int,
    validation_fraction: float,
) -> None:
    chunks = list(payload.get("chunks") or [])
    rng = random.Random(int(seed))
    rng.shuffle(chunks)
    val_count = int(len(chunks) * float(validation_fraction))
    validation_sources = [str(chunk.get("source_shard") or "") for chunk in chunks[:val_count]]
    validation_sources = [source for source in validation_sources if source]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(validation_sources) + ("\n" if validation_sources else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge compatible chunked v3 combat transformer tensor manifests.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--validation-source-shards-output",
        type=Path,
        default=None,
        help="Optional file with validation source shards sampled from one input before merging.",
    )
    parser.add_argument("--validation-from-input-index", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    payloads = [_load(path) for path in args.inputs]
    if not payloads:
        raise SystemExit("no inputs")
    first_key = _schema_key(payloads[0])
    for path, payload in zip(args.inputs, payloads, strict=True):
        if _schema_key(payload) != first_key:
            raise SystemExit(f"incompatible tensor dataset schema: {path}")
        if not payload.get("chunks"):
            raise SystemExit(f"input has no chunks: {path}")

    if args.validation_source_shards_output is not None:
        source_index = int(args.validation_from_input_index)
        if source_index < 0 or source_index >= len(payloads):
            raise SystemExit(f"--validation-from-input-index out of range: {source_index}")
        _write_validation_sources(
            payload=payloads[source_index],
            output=args.validation_source_shards_output,
            seed=int(args.seed),
            validation_fraction=float(args.validation_fraction),
        )

    merged = dict(payloads[0])
    chunks: list[dict[str, Any]] = []
    root_ids: list[str] = []
    source_shards: list[str] = []
    root_count = 0
    candidate_count = 0
    source_manifests: list[dict[str, Any]] = []
    for path, payload in zip(args.inputs, payloads, strict=True):
        payload_chunks = [dict(chunk) for chunk in payload.get("chunks") or []]
        chunks.extend(payload_chunks)
        root_ids.extend(str(root_id) for root_id in payload.get("root_ids") or [])
        source_shards.extend(str(source_shard) for source_shard in payload.get("source_shards") or [])
        metadata = dict(payload.get("metadata") or {})
        root_count += int(metadata.get("root_count") or sum(int(chunk.get("root_count") or 0) for chunk in payload_chunks))
        candidate_count += int(
            metadata.get("candidate_count") or sum(int(chunk.get("candidate_count") or 0) for chunk in payload_chunks)
        )
        source_manifests.append(
            {
                "path": str(path),
                "root_count": int(metadata.get("root_count") or 0),
                "candidate_count": int(metadata.get("candidate_count") or 0),
                "chunk_count": int(metadata.get("chunk_count") or len(payload_chunks)),
            }
        )

    metadata = dict(merged.get("metadata") or {})
    metadata.update(
        {
            "root_count": root_count,
            "candidate_count": candidate_count,
            "shard_count": len(set(source_shards)),
            "chunk_count": len(chunks),
            "merged_from": source_manifests,
        }
    )
    merged["chunks"] = chunks
    merged["root_ids"] = root_ids
    merged["source_shards"] = source_shards
    merged["metadata"] = metadata

    args.output.parent.mkdir(parents=True, exist_ok=True)
    require_torch().save(merged, args.output)
    summary = {
        "output": str(args.output),
        "root_count": root_count,
        "candidate_count": candidate_count,
        "chunk_count": len(chunks),
        "source_manifest_count": len(source_manifests),
        "source_manifests": source_manifests,
        "validation_source_shards_output": str(args.validation_source_shards_output)
        if args.validation_source_shards_output is not None
        else None,
    }
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
