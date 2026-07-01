#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cache_v3_combat_transformer_dataset import (
    CHUNKED_ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA,
    CHUNKED_TRANSFORMER_TENSOR_DATASET_SCHEMA,
    _cache_shard_chunk_worker,
    _root_seed,
    _token_schema_payload,
)
from spirecomm.ai.torch_compat import require_torch
from spirecomm.ai.v3_combat_dataset import REWARD_COMPONENT_NAMES
from spirecomm.ai.v3_combat_features import FEATURE_SCHEMA_VERSION, schema
from spirecomm.ai.v3_combat_transformer import (
    ROOT_TOKEN_SCHEMA_VERSION,
    TOKEN_SCHEMA_VERSION,
    root_token_spec,
    token_spec,
    transformer_entity_ids,
)


def _chunk_index(path: Path) -> int | None:
    match = re.fullmatch(r"chunk_(\d+)\.pt", path.name)
    if not match:
        return None
    return int(match.group(1))


def _read_shards(args: argparse.Namespace) -> list[Path]:
    shards = list(args.shards)
    if args.shards_file is not None:
        shards.extend(
            Path(line.strip())
            for line in args.shards_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    shards = sorted(shards)
    if not shards:
        raise SystemExit("No shards provided. Pass --shards and/or --shards-file.")
    return shards


def _load_chunk_summary(path: Path, expected_source_shard: Path) -> dict[str, Any] | None:
    torch = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"invalid chunk removed path={path} error={exc}", flush=True)
        path.unlink(missing_ok=True)
        return None
    metadata = dict(payload.get("metadata") or {})
    source_shard = str(metadata.get("source_shard") or "")
    if source_shard != str(expected_source_shard):
        print(
            "stale chunk removed "
            f"path={path} expected={expected_source_shard} got={source_shard}",
            flush=True,
        )
        path.unlink(missing_ok=True)
        return None
    root_ids = list(payload.get("root_ids") or [])
    source_shards = list(payload.get("source_shards") or [])
    if int(metadata.get("root_count") or 0) != len(root_ids):
        print(f"corrupt chunk removed path={path} reason=root_count_mismatch", flush=True)
        path.unlink(missing_ok=True)
        return None
    seed_values = [seed for rid in root_ids if (seed := _root_seed(str(rid))) is not None]
    return {
        "chunk": {
            "path": str(path),
            "root_count": int(metadata.get("root_count") or len(root_ids)),
            "candidate_count": int(metadata.get("candidate_count") or 0),
            "source_shard": str(expected_source_shard),
        },
        "root_ids": root_ids,
        "source_shards": source_shards,
        "seed_values": seed_values,
        "root_count": int(metadata.get("root_count") or len(root_ids)),
        "candidate_count": int(metadata.get("candidate_count") or 0),
    }


def _existing_results(chunk_dir: Path, shards: list[Path]) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    if not chunk_dir.exists():
        return results
    for path in sorted(chunk_dir.glob("chunk_*.pt")):
        index = _chunk_index(path)
        if index is None or index < 0 or index >= len(shards):
            print(f"orphan chunk ignored path={path}", flush=True)
            continue
        summary = _load_chunk_summary(path, shards[index])
        if summary is not None:
            results[index] = summary
    return results


def _write_manifest(
    *,
    output: Path,
    summary_path: Path,
    shards: list[Path],
    results: dict[int, dict[str, Any]],
    dtype: str,
    root_action_set: bool,
    token_schema_version: str,
) -> None:
    torch = require_torch()
    spec = root_token_spec() if root_action_set else token_spec(version=token_schema_version)
    chunks: list[dict[str, Any]] = []
    root_ids: list[str] = []
    source_shards: list[str] = []
    seed_values: list[int] = []
    root_count = 0
    candidate_count = 0
    missing = [index for index in range(len(shards)) if index not in results]
    if missing:
        raise SystemExit(f"cannot write manifest; missing chunk summaries for indices: {missing[:20]}")
    for index in range(len(shards)):
        result = results[index]
        chunk = result.get("chunk")
        if chunk is None:
            continue
        chunks.append(chunk)
        root_ids.extend(result["root_ids"])
        source_shards.extend(result["source_shards"])
        seed_values.extend(result["seed_values"])
        root_count += int(result["root_count"])
        candidate_count += int(result["candidate_count"])
    if not chunks:
        raise SystemExit("No transformer candidates found in shards.")
    output_payload = {
        "tensor_dataset_schema": CHUNKED_ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA
        if root_action_set
        else CHUNKED_TRANSFORMER_TENSOR_DATASET_SCHEMA,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_dims": schema().__dict__,
        "token_schema_version": ROOT_TOKEN_SCHEMA_VERSION if root_action_set else spec.version,
        "token_schema": _token_schema_payload(spec),
        "entity_vocab": list(transformer_entity_ids()),
        "chunks": chunks,
        "root_ids": root_ids,
        "source_shards": source_shards,
        "metadata": {
            "root_count": root_count,
            "candidate_count": candidate_count,
            "shard_count": len(shards),
            "chunk_count": len(chunks),
            "seed_min": min(seed_values) if seed_values else None,
            "seed_max": max(seed_values) if seed_values else None,
            "unique_seed_count": len(set(seed_values)),
            "dtype": dtype,
            "limit_roots": None,
            "sequence_length": spec.max_sequence_length,
            "scalar_dim": spec.scalar_dim,
            "entity_vocab_size": len(transformer_entity_ids()),
            "max_actions": getattr(spec, "max_actions", None),
            "action_segment_width": getattr(spec, "action_segment_width", None),
            "uses_legacy_token": bool(getattr(spec, "uses_legacy_token", False)),
            "reward_component_names": list(REWARD_COMPONENT_NAMES),
            "resume_cache": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output)
    summary = dict(output_payload["metadata"])
    summary.update(
        {
            "output": str(output),
            "chunk_dir": str(output.with_suffix(output.suffix + ".chunks")),
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "feature_dim": schema().candidate_dim,
            "token_schema": ROOT_TOKEN_SCHEMA_VERSION if root_action_set else spec.version,
            "root_action_set": bool(root_action_set),
            "token_shape": [
                root_count if root_action_set else candidate_count,
                spec.max_sequence_length,
                spec.scalar_dim,
            ],
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume chunked transformer tensor caching without deleting existing valid chunks."
    )
    parser.add_argument("--shards", nargs="*", type=Path, default=[])
    parser.add_argument("--shards-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--root-action-set", action="store_true")
    parser.add_argument("--derive-reward-components", action="store_true")
    parser.add_argument("--token-schema-version", default=TOKEN_SCHEMA_VERSION)
    args = parser.parse_args()

    shards = _read_shards(args)
    spec = root_token_spec() if args.root_action_set else token_spec(version=args.token_schema_version)
    chunk_dir = args.output.with_suffix(args.output.suffix + ".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")

    results = _existing_results(chunk_dir, shards)
    missing_indices = [index for index in range(len(shards)) if index not in results]
    print(
        "resume cache "
        f"shards={len(shards)} existing={len(results)} missing={len(missing_indices)} "
        f"workers={max(1, int(args.workers))}",
        flush=True,
    )
    if missing_indices:
        tasks = [
            (
                index,
                str(shards[index]),
                str(chunk_dir / f"chunk_{index:05d}.pt"),
                args.dtype,
                bool(args.root_action_set),
                str(spec.version),
                bool(args.derive_reward_components),
            )
            for index in missing_indices
        ]
        with mp.Pool(processes=max(1, int(args.workers))) as pool:
            for completed, result in enumerate(pool.imap_unordered(_cache_shard_chunk_worker, tasks), start=1):
                index = int(result["shard_index"])
                results[index] = result
                print(
                    f"cached missing {completed}/{len(tasks)} "
                    f"index={index} source={result['source_shard']} roots={result['root_count']}",
                    flush=True,
                )

    _write_manifest(
        output=args.output,
        summary_path=summary_path,
        shards=shards,
        results=results,
        dtype=args.dtype,
        root_action_set=bool(args.root_action_set),
        token_schema_version=str(spec.version),
    )


if __name__ == "__main__":
    main()
