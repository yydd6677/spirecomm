#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from spirecomm.ai.torch_compat import require_torch, torch


def _load_manifest(cache_dir: Path) -> dict[str, Any]:
    return json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))


def _floor_bucket(floor: int, width: int = 5) -> str:
    start = int(floor) // int(width) * int(width)
    return f"{start:02d}-{start + int(width) - 1:02d}"


def _phase(payload: dict[str, Any], index: int) -> str:
    phases = payload.get("phases") or [""] * int(payload["targets"].shape[0])
    return str(phases[int(index)])


def _key(payload: dict[str, Any], index: int) -> tuple[str, str]:
    return (_floor_bucket(int(payload["floors"][int(index)].item())), _phase(payload, int(index)))


def _reservoir_add(bucket: dict[str, Any], *, key: tuple[str, str], feature: Any, target: float, max_count: int, rng: random.Random) -> None:
    item = bucket.setdefault(key, {"seen": 0, "features": [], "targets": []})
    item["seen"] += 1
    seen = int(item["seen"])
    if len(item["features"]) < int(max_count):
        item["features"].append(feature.detach().cpu().clone())
        item["targets"].append(float(target))
        return
    replace = rng.randrange(seen)
    if replace < int(max_count):
        item["features"][replace] = feature.detach().cpu().clone()
        item["targets"][replace] = float(target)


def _collect_samples(
    chunks: list[dict[str, Any]],
    *,
    max_train_per_group: int,
    max_val_per_group: int,
    seed: int,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    train: dict[tuple[str, str], dict[str, Any]] = {}
    valid: dict[tuple[str, str], dict[str, Any]] = {}
    rng = random.Random(int(seed))
    counts = {"train_rows": 0, "validation_rows": 0, "groups": Counter()}
    for chunk_index, chunk in enumerate(chunks, start=1):
        payload = torch.load(Path(chunk["path"]), map_location="cpu", weights_only=False)
        features = payload["features"]
        targets = payload["targets"][:, 0].float()
        is_validation = payload["is_validation"].bool()
        for index in range(int(features.shape[0])):
            key = _key(payload, index)
            counts["groups"][f"{key[0]}|{key[1]}"] += 1
            if bool(is_validation[index].item()):
                counts["validation_rows"] += 1
                _reservoir_add(
                    valid,
                    key=key,
                    feature=features[index],
                    target=float(targets[index].item()),
                    max_count=int(max_val_per_group),
                    rng=rng,
                )
            else:
                counts["train_rows"] += 1
                _reservoir_add(
                    train,
                    key=key,
                    feature=features[index],
                    target=float(targets[index].item()),
                    max_count=int(max_train_per_group),
                    rng=rng,
                )
        if chunk_index == 1 or chunk_index == len(chunks) or chunk_index % 25 == 0:
            print(
                f"sample chunks={chunk_index}/{len(chunks)} train_rows={counts['train_rows']} val_rows={counts['validation_rows']}",
                flush=True,
            )
    return train, valid, counts


def _project(features: Any, projection: Any) -> Any:
    return features.float().matmul(projection)


def _knn_group(
    train_features: Any,
    train_targets: Any,
    val_features: Any,
    val_targets: Any,
    *,
    projection: Any,
    k_values: list[int],
    batch_size: int,
    device: str,
) -> dict[int, list[float]]:
    train_proj = _project(train_features.to(device), projection)
    val_proj = _project(val_features.to(device), projection)
    mean_vec = train_proj.mean(dim=0, keepdim=True)
    std_vec = train_proj.std(dim=0, keepdim=True).clamp_min(1.0e-4)
    train_proj = (train_proj - mean_vec) / std_vec
    val_proj = (val_proj - mean_vec) / std_vec
    train_targets = train_targets.float().to(device)
    max_k = min(max(k_values), int(train_targets.shape[0]))
    errors = {int(k): [] for k in k_values if int(k) <= int(train_targets.shape[0])}
    if not errors:
        return errors
    for start in range(0, int(val_proj.shape[0]), int(batch_size)):
        batch = val_proj[start : start + int(batch_size)]
        dists = torch.cdist(batch, train_proj)
        indexes = torch.topk(dists, k=max_k, largest=False, dim=1).indices
        true = val_targets[start : start + int(batch.shape[0])].float().to(device)
        for k in errors:
            pred = train_targets.index_select(0, indexes[:, : int(k)].reshape(-1)).view(int(batch.shape[0]), int(k)).median(dim=1).values
            errors[int(k)].extend(torch.abs(pred - true).detach().cpu().tolist())
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Approximate same-floor/phase KNN oracle for run-value tensor caches.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-per-group", type=int, default=2000)
    parser.add_argument("--max-val-per-group", type=int, default=200)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--k", type=str, default="25,50")
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-chunks", type=int, default=0)
    args = parser.parse_args()

    require_torch()
    manifest = _load_manifest(args.cache_dir)
    chunks = list(manifest.get("chunks") or [])
    if int(args.max_chunks) > 0:
        chunks = chunks[: int(args.max_chunks)]
    if not chunks:
        raise SystemExit(f"cache has no chunks: {args.cache_dir}")
    train, valid, counts = _collect_samples(
        chunks,
        max_train_per_group=int(args.max_train_per_group),
        max_val_per_group=int(args.max_val_per_group),
        seed=int(args.seed),
    )
    input_dim = int(manifest.get("state_feature_dim") or 0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(args.seed))
    projection = torch.randn((input_dim, int(args.projection_dim)), generator=gen, dtype=torch.float32) / max(1.0, float(args.projection_dim) ** 0.5)
    projection = projection.to(str(args.device))
    k_values = [int(value) for value in str(args.k).split(",") if str(value).strip()]
    global_errors: dict[int, list[float]] = {k: [] for k in k_values}
    floor_errors: dict[int, dict[str, list[float]]] = {k: defaultdict(list) for k in k_values}
    phase_errors: dict[int, dict[str, list[float]]] = {k: defaultdict(list) for k in k_values}
    group_summary = {}
    used_groups = 0
    for group_index, key in enumerate(sorted(valid), start=1):
        if key not in train:
            continue
        train_item = train[key]
        val_item = valid[key]
        if not train_item["features"] or not val_item["features"]:
            continue
        train_features = torch.stack(train_item["features"])
        val_features = torch.stack(val_item["features"])
        train_targets = torch.tensor(train_item["targets"], dtype=torch.float32)
        val_targets = torch.tensor(val_item["targets"], dtype=torch.float32)
        errors = _knn_group(
            train_features,
            train_targets,
            val_features,
            val_targets,
            projection=projection,
            k_values=k_values,
            batch_size=int(args.batch_size),
            device=str(args.device),
        )
        if not errors:
            continue
        used_groups += 1
        group_key = f"{key[0]}|{key[1]}"
        group_summary[group_key] = {
            "train_sample": int(train_features.shape[0]),
            "validation_sample": int(val_features.shape[0]),
            "mae": {str(k): float(mean(values)) for k, values in errors.items() if values},
        }
        for k, values in errors.items():
            global_errors[int(k)].extend(values)
            floor_errors[int(k)][key[0]].extend(values)
            phase_errors[int(k)][key[1]].extend(values)
        if group_index == 1 or group_index == len(valid) or group_index % 20 == 0:
            print(f"knn groups={group_index}/{len(valid)} used={used_groups}", flush=True)
    result = {
        "cache_dir": str(args.cache_dir),
        "manifest": {
            key: manifest.get(key)
            for key in ("count", "train_count", "validation_count", "state_feature_dim", "sample_mode", "feature_variant", "record_weight_mode")
        },
        "max_train_per_group": int(args.max_train_per_group),
        "max_val_per_group": int(args.max_val_per_group),
        "projection_dim": int(args.projection_dim),
        "k": k_values,
        "sample_counts": {
            "train_rows": int(counts["train_rows"]),
            "validation_rows": int(counts["validation_rows"]),
            "groups": dict(counts["groups"].most_common(100)),
        },
        "used_groups": int(used_groups),
        "metrics": {
            str(k): {
                "mae": float(mean(values)) if values else 0.0,
                "count": len(values),
                "floor_mae": {floor: float(mean(items)) for floor, items in sorted(floor_errors[k].items())},
                "phase_mae": {phase: float(mean(items)) for phase, items in sorted(phase_errors[k].items())},
            }
            for k, values in global_errors.items()
        },
        "group_summary": group_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "group_summary"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
