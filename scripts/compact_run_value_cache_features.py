#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spirecomm.ai.torch_compat import require_torch, torch


def _load_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a run-value tensor cache with feature tensors sliced/padded to a fixed width.")
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()

    require_torch()
    feature_dim = int(args.feature_dim)
    if feature_dim <= 0:
        raise SystemExit(f"invalid feature dim: {feature_dim}")
    input_cache = args.input_cache
    output_cache = args.output_cache
    if output_cache.exists():
        if not bool(args.overwrite):
            raise SystemExit(f"output cache exists: {output_cache}")
        shutil.rmtree(output_cache)
    output_cache.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(input_cache)
    chunks = list(manifest.get("chunks") or [])
    if not chunks:
        raise SystemExit(f"cache has no chunks: {input_cache}")

    new_chunks: list[dict[str, Any]] = []
    total_rows = 0
    for index, chunk in enumerate(chunks):
        source_path = Path(str(chunk["path"]))
        if not source_path.is_absolute() and not source_path.exists():
            source_path = input_cache / Path(str(chunk["path"])).name
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        features = payload["features"]
        current_dim = int(features.shape[1])
        if current_dim > feature_dim:
            payload["features"] = features[:, :feature_dim].contiguous()
        elif current_dim < feature_dim:
            pad = torch.zeros((int(features.shape[0]), feature_dim - current_dim), dtype=features.dtype)
            payload["features"] = torch.cat([features, pad], dim=1).contiguous()
        else:
            payload["features"] = features.contiguous()
        target_path = output_cache / f"chunk_{index:05d}.pt"
        torch.save(payload, target_path)
        count = int(payload["features"].shape[0])
        total_rows += count
        new_chunks.append({"path": str(target_path), "count": count})
        if index == 0 or index + 1 == len(chunks) or (index + 1) % int(args.progress_interval) == 0:
            print(f"compact cache chunks={index + 1}/{len(chunks)} rows={total_rows} dim={feature_dim}", flush=True)

    new_manifest = dict(manifest)
    new_manifest["cache_dir"] = str(output_cache)
    new_manifest["chunks"] = new_chunks
    new_manifest["state_feature_dim"] = feature_dim
    new_manifest["original_state_feature_dim"] = int(manifest.get("state_feature_dim") or 0)
    new_manifest["compacted_from_cache_dir"] = str(input_cache)
    new_manifest["count"] = total_rows
    (output_cache / "manifest.json").write_text(json.dumps(new_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"compact cache complete rows={total_rows} dim={feature_dim} output={output_cache}", flush=True)


if __name__ == "__main__":
    main()
