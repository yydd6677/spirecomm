#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from spirecomm.ai.torch_compat import require_torch
from spirecomm.ai.v3_combat_dataset import load_shard
from spirecomm.ai.v3_combat_features import FEATURE_SCHEMA_VERSION, schema
from spirecomm.ai.v3_combat_transformer import (
    TOKEN_SCHEMA_VERSION,
    TRANSFORMER_TENSOR_DATASET_SCHEMA,
    encode_transformer_candidate,
    token_spec,
    transformer_entity_ids,
)
from cache_v3_combat_transformer_dataset import CHUNKED_TRANSFORMER_TENSOR_DATASET_SCHEMA


def _root_seed(root_id: str) -> int | None:
    match = re.match(r"explore:(-?\d+):", str(root_id))
    return int(match.group(1)) if match else None


def _root_id(labeled: Any) -> str:
    root = getattr(labeled, "root", None)
    return str(getattr(root, "root_id", ""))


def _load_manifest(path: Path, torch: Any) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("tensor_dataset_schema") != CHUNKED_TRANSFORMER_TENSOR_DATASET_SCHEMA:
        raise ValueError(f"unsupported manifest schema in {path}: {payload.get('tensor_dataset_schema')}")
    if payload.get("feature_schema") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"feature schema mismatch in {path}: {payload.get('feature_schema')} != {FEATURE_SCHEMA_VERSION}")
    if payload.get("token_schema_version") != TOKEN_SCHEMA_VERSION:
        raise ValueError(f"token schema mismatch in {path}: {payload.get('token_schema_version')} != {TOKEN_SCHEMA_VERSION}")
    return payload


def _source_shards(source_dirs: list[Path], *, active_source_dirs: set[Path], stable_lag_shards: int) -> list[Path]:
    shards: list[Path] = []
    active_resolved = {path.resolve() for path in active_source_dirs}
    for source_dir in source_dirs:
        source_dir = source_dir.resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"source dir not found: {source_dir}")
        current = sorted(source_dir.glob("shard_*.pt"))
        if source_dir in active_resolved and stable_lag_shards > 0:
            current = current[: max(0, len(current) - int(stable_lag_shards))]
        shards.extend(current)
    return shards


def _save_manifest(
    *,
    output: Path,
    chunks: list[dict[str, Any]],
    root_ids: list[str],
    source_shards: list[str],
    dtype: str,
    limit_roots: int,
) -> dict[str, Any]:
    torch = require_torch()
    spec = token_spec()
    seed_values = [seed for rid in root_ids if (seed := _root_seed(rid)) is not None]
    root_count = sum(int(chunk.get("root_count") or 0) for chunk in chunks)
    candidate_count = sum(int(chunk.get("candidate_count") or 0) for chunk in chunks)
    payload = {
        "tensor_dataset_schema": CHUNKED_TRANSFORMER_TENSOR_DATASET_SCHEMA,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_dims": schema().__dict__,
        "token_schema_version": TOKEN_SCHEMA_VERSION,
        "token_schema": spec.__dict__,
        "entity_vocab": list(transformer_entity_ids()),
        "chunks": chunks,
        "root_ids": root_ids,
        "source_shards": source_shards,
        "metadata": {
            "root_count": root_count,
            "candidate_count": candidate_count,
            "shard_count": len({chunk.get("source_shard") for chunk in chunks}),
            "chunk_count": len(chunks),
            "seed_min": min(seed_values) if seed_values else None,
            "seed_max": max(seed_values) if seed_values else None,
            "unique_seed_count": len(set(seed_values)),
            "dtype": dtype,
            "limit_roots": limit_roots or None,
            "sequence_length": spec.max_sequence_length,
            "scalar_dim": spec.scalar_dim,
            "entity_vocab_size": len(transformer_entity_ids()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    summary = dict(payload["metadata"])
    summary.update(
        {
            "output": str(output),
            "chunk_dir": str(output.with_suffix(output.suffix + ".chunks")),
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "feature_dim": schema().candidate_dim,
            "token_schema": TOKEN_SCHEMA_VERSION,
            "token_shape": [candidate_count, spec.max_sequence_length, spec.scalar_dim],
        }
    )
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _write_validation_sources(path: Path, chunks: list[dict[str, Any]], *, fraction: float, seed: int) -> None:
    if path.exists():
        return
    rng = random.Random(seed)
    source_shards = sorted({str(chunk.get("source_shard") or "") for chunk in chunks if chunk.get("source_shard")})
    rng.shuffle(source_shards)
    count = max(1, int(len(source_shards) * max(0.0, min(1.0, fraction))))
    selected = sorted(source_shards[:count])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(selected) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally cache v3 combat teacher shards as transformer chunks.")
    parser.add_argument("--source-dir", action="append", type=Path, required=True)
    parser.add_argument("--active-source-dir", action="append", type=Path, default=[])
    parser.add_argument("--stable-lag-shards", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--limit-roots", type=int, default=0)
    parser.add_argument("--max-new-shards", type=int, default=0)
    parser.add_argument("--save-every-shards", type=int, default=16)
    parser.add_argument("--validation-source-shards-file", type=Path, default=None)
    parser.add_argument("--create-validation-if-missing", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    torch = require_torch()
    spec = token_spec()
    float_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    chunk_dir = args.output.with_suffix(args.output.suffix + ".chunks")
    manifest = _load_manifest(args.output, torch)
    if manifest is None:
        chunks: list[dict[str, Any]] = []
        root_ids: list[str] = []
        source_shards: list[str] = []
    else:
        chunks = list(manifest.get("chunks") or [])
        root_ids = list(manifest.get("root_ids") or [])
        source_shards = list(manifest.get("source_shards") or [])

    processed = {str(chunk.get("source_shard") or "") for chunk in chunks}
    root_count = sum(int(chunk.get("root_count") or 0) for chunk in chunks)
    limit_roots = max(0, int(args.limit_roots))
    candidates = [
        path
        for path in _source_shards(
            list(args.source_dir),
            active_source_dirs=set(args.active_source_dir or []),
            stable_lag_shards=max(0, int(args.stable_lag_shards)),
        )
        if str(path) not in processed
    ]
    if args.max_new_shards > 0:
        candidates = candidates[: int(args.max_new_shards)]

    new_shards = 0
    for shard_path in candidates:
        if limit_roots and root_count >= limit_roots:
            break
        payload = load_shard(shard_path)
        roots = payload.get("roots") or []
        shard_token_scalars = []
        shard_token_types = []
        shard_entities = []
        shard_slots = []
        shard_masks = []
        shard_legacy = []
        shard_teacher_q = []
        shard_sample_ids = []
        shard_chosen = []
        shard_candidate_offsets = [0]
        shard_root_ids: list[str] = []
        shard_source_shards: list[str] = []
        shard_candidate_count = 0
        shard_root_count = 0
        for labeled in roots:
            if limit_roots and root_count >= limit_roots:
                break
            before_state = labeled.root.visible_before
            rid = _root_id(labeled)
            root_ids.append(rid)
            source_shards.append(str(shard_path))
            shard_root_ids.append(rid)
            shard_source_shards.append(str(shard_path))
            for candidate in labeled.candidates:
                record = encode_transformer_candidate(
                    before_state,
                    candidate.action,
                    candidate.visible_after,
                    candidate_features=candidate.candidate_features,
                    spec=spec,
                )
                shard_token_scalars.append(record["token_scalar_features"])
                shard_token_types.append(record["token_type_ids"])
                shard_entities.append(record["entity_ids"])
                shard_slots.append(record["slot_ids"])
                shard_masks.append(record["attention_mask"])
                shard_legacy.append(record["candidate_features"])
                shard_teacher_q.append(float(candidate.teacher_q))
                shard_sample_ids.append(shard_root_count)
                shard_chosen.append(bool(candidate.is_chosen))
                shard_candidate_count += 1
            root_count += 1
            shard_root_count += 1
            shard_candidate_offsets.append(shard_candidate_count)
        if shard_token_scalars:
            chunk_payload = {
                "tensor_dataset_schema": TRANSFORMER_TENSOR_DATASET_SCHEMA,
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "feature_dims": schema().__dict__,
                "token_schema_version": TOKEN_SCHEMA_VERSION,
                "token_schema": spec.__dict__,
                "entity_vocab": list(transformer_entity_ids()),
                "token_scalar_features": torch.tensor(shard_token_scalars, dtype=float_dtype).contiguous(),
                "token_type_ids": torch.tensor(shard_token_types, dtype=torch.uint8).contiguous(),
                "entity_ids": torch.tensor(shard_entities, dtype=torch.int32).contiguous(),
                "slot_ids": torch.tensor(shard_slots, dtype=torch.int16).contiguous(),
                "attention_mask": torch.tensor(shard_masks, dtype=torch.bool).contiguous(),
                "features": torch.tensor(shard_legacy, dtype=float_dtype).contiguous(),
                "teacher_q": torch.tensor(shard_teacher_q, dtype=torch.float32).contiguous(),
                "sample_ids": torch.tensor(shard_sample_ids, dtype=torch.long).contiguous(),
                "chosen": torch.tensor(shard_chosen, dtype=torch.bool).contiguous(),
                "candidate_offsets": torch.tensor(shard_candidate_offsets, dtype=torch.long),
                "root_ids": shard_root_ids,
                "source_shards": shard_source_shards,
                "metadata": {
                    "root_count": shard_root_count,
                    "candidate_count": shard_candidate_count,
                    "source_shard": str(shard_path),
                    "dtype": args.dtype,
                    "sequence_length": spec.max_sequence_length,
                    "scalar_dim": spec.scalar_dim,
                    "entity_vocab_size": len(transformer_entity_ids()),
                },
            }
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = chunk_dir / f"chunk_{len(chunks):05d}.pt"
            torch.save(chunk_payload, chunk_path)
            chunks.append(
                {
                    "path": str(chunk_path),
                    "root_count": shard_root_count,
                    "candidate_count": shard_candidate_count,
                    "source_shard": str(shard_path),
                }
            )
            new_shards += 1
        print(f"cached {shard_path} roots={shard_root_count} chunks={len(chunks)}", flush=True)
        if new_shards and new_shards % max(1, int(args.save_every_shards)) == 0:
            _save_manifest(
                output=args.output,
                chunks=chunks,
                root_ids=root_ids,
                source_shards=source_shards,
                dtype=args.dtype,
                limit_roots=limit_roots,
            )

    summary = _save_manifest(
        output=args.output,
        chunks=chunks,
        root_ids=root_ids,
        source_shards=source_shards,
        dtype=args.dtype,
        limit_roots=limit_roots,
    )
    if args.validation_source_shards_file is not None and args.create_validation_if_missing:
        _write_validation_sources(
            args.validation_source_shards_file,
            chunks,
            fraction=float(args.validation_fraction),
            seed=int(args.seed),
        )
    summary["new_shards"] = new_shards
    summary["available_new_shards"] = len(candidates)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
