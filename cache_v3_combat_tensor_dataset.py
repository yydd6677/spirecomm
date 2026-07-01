#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from spirecomm.ai.torch_compat import require_torch
from spirecomm.ai.v3_combat_dataset import load_shard
from spirecomm.ai.v3_combat_features import FEATURE_SCHEMA_VERSION, schema


def _root_seed(root_id: str) -> int | None:
    match = re.match(r"explore:(-?\d+):", str(root_id))
    if not match:
        return None
    return int(match.group(1))


def _root_id(labeled: Any) -> str:
    root = getattr(labeled, "root", None)
    return str(getattr(root, "root_id", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert v3 combat teacher shards into a compact tensor dataset.")
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()

    torch = require_torch()
    feature_tensors = []
    teacher_tensors = []
    sample_id_tensors = []
    chosen_tensors = []
    candidate_offsets = [0]
    root_ids: list[str] = []
    source_shards: list[str] = []
    seed_values: list[int] = []
    candidate_count = 0
    root_count = 0

    float_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    shards = sorted(args.shards)
    for shard_index, shard_path in enumerate(shards):
        payload = load_shard(shard_path)
        roots = payload.get("roots") or []
        shard_features: list[list[float]] = []
        shard_teacher_q: list[float] = []
        shard_sample_ids: list[int] = []
        shard_chosen: list[bool] = []
        for labeled in roots:
            rid = _root_id(labeled)
            root_ids.append(rid)
            source_shards.append(str(shard_path))
            seed = _root_seed(rid)
            if seed is not None:
                seed_values.append(seed)
            for candidate in labeled.candidates:
                shard_features.append(candidate.candidate_features)
                shard_teacher_q.append(float(candidate.teacher_q))
                shard_sample_ids.append(root_count)
                shard_chosen.append(bool(candidate.is_chosen))
                candidate_count += 1
            root_count += 1
            candidate_offsets.append(candidate_count)
        if shard_features:
            feature_tensors.append(torch.tensor(shard_features, dtype=float_dtype))
            teacher_tensors.append(torch.tensor(shard_teacher_q, dtype=torch.float32))
            sample_id_tensors.append(torch.tensor(shard_sample_ids, dtype=torch.long))
            chosen_tensors.append(torch.tensor(shard_chosen, dtype=torch.bool))
        print(f"cached shard {shard_index + 1}/{len(shards)} roots={len(roots)} total_roots={root_count}", flush=True)

    if not feature_tensors:
        raise SystemExit("No candidates found in shards.")

    output = {
        "tensor_dataset_schema": "v3_combat_tensor_dataset_v1",
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_dims": schema().__dict__,
        "features": torch.cat(feature_tensors, dim=0).contiguous(),
        "teacher_q": torch.cat(teacher_tensors, dim=0).contiguous(),
        "sample_ids": torch.cat(sample_id_tensors, dim=0).contiguous(),
        "chosen": torch.cat(chosen_tensors, dim=0).contiguous(),
        "candidate_offsets": torch.tensor(candidate_offsets, dtype=torch.long),
        "root_ids": root_ids,
        "source_shards": source_shards,
        "metadata": {
            "root_count": root_count,
            "candidate_count": candidate_count,
            "shard_count": len(shards),
            "seed_min": min(seed_values) if seed_values else None,
            "seed_max": max(seed_values) if seed_values else None,
            "unique_seed_count": len(set(seed_values)),
            "dtype": args.dtype,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)

    summary = dict(output["metadata"])
    summary.update(
        {
            "output": str(args.output),
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "feature_dim": int(output["features"].shape[1]),
        }
    )
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
