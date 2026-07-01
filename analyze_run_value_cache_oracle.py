#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from spirecomm.ai.torch_compat import require_torch, torch


def _floor_bucket(floor: int, width: int) -> str:
    start = int(floor) // int(width) * int(width)
    return f"{start:02d}-{start + int(width) - 1:02d}"


def _load_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "manifest.json"
    if not path.exists():
        raise SystemExit(f"missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _chunk_paths(manifest: dict[str, Any]) -> list[Path]:
    return [Path(row["path"]) for row in manifest.get("chunks") or []]


def _row_value(payload: dict[str, Any], key: str, index: int) -> Any:
    if key == "floor":
        return int(payload["floors"][index].item())
    if key == "floor_bucket":
        return _floor_bucket(int(payload["floors"][index].item()), 5)
    if key == "floor_bucket10":
        return _floor_bucket(int(payload["floors"][index].item()), 10)
    if key == "phase":
        values = payload.get("phases") or [""] * int(payload["targets"].shape[0])
        return str(values[index])
    if key == "sample_kind":
        values = payload.get("sample_kinds") or ["before"] * int(payload["targets"].shape[0])
        return str(values[index])
    if key == "source":
        values = payload.get("sources") or [""] * int(payload["targets"].shape[0])
        return str(values[index])
    raise KeyError(key)


def _fit_all_group_medians(
    chunks: list[Path],
    level_defs: list[tuple[str, tuple[str, ...]]],
    *,
    min_train_count: int,
) -> tuple[list[tuple[str, tuple[str, ...], dict[tuple[Any, ...], float]]], float]:
    values_by_level: list[dict[tuple[Any, ...], list[float]]] = [defaultdict(list) for _ in level_defs]
    global_values: list[float] = []
    for path in chunks:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        mask = ~payload["is_validation"].bool()
        indexes = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        targets = payload["targets"][:, 0].float()
        for index in indexes:
            target = float(targets[int(index)].item())
            for level_index, (_, fields) in enumerate(level_defs):
                key = tuple(_row_value(payload, field, int(index)) for field in fields)
                values_by_level[level_index][key].append(target)
            global_values.append(target)
    fitted: list[tuple[str, tuple[str, ...], dict[tuple[Any, ...], float]]] = []
    for (name, fields), values in zip(level_defs, values_by_level):
        medians = {
            key: float(median(items))
            for key, items in values.items()
            if len(items) >= int(min_train_count)
        }
        fitted.append((name, fields, medians))
    return fitted, float(median(global_values)) if global_values else 0.0


def _eval_oracle(
    chunks: list[Path],
    levels: list[tuple[str, tuple[str, ...], dict[tuple[Any, ...], float]]],
    *,
    global_median: float,
) -> dict[str, Any]:
    errors: list[float] = []
    seed_errors: dict[int, list[float]] = defaultdict(list)
    floor_errors: dict[str, list[float]] = defaultdict(list)
    phase_errors: dict[str, list[float]] = defaultdict(list)
    used = Counter()
    for path in chunks:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        mask = payload["is_validation"].bool()
        indexes = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        targets = payload["targets"][:, 0].float()
        for index in indexes:
            pred = float(global_median)
            used_level = "global"
            for name, fields, table in reversed(levels):
                key = tuple(_row_value(payload, field, int(index)) for field in fields)
                if key in table:
                    pred = float(table[key])
                    used_level = name
                    break
            target = float(targets[int(index)].item())
            err = abs(pred - target)
            errors.append(err)
            seed = int(payload["seeds"][int(index)].item())
            seed_errors[seed].append(err)
            floor_errors[_floor_bucket(int(payload["floors"][int(index)].item()), 5)].append(err)
            phase_errors[str(_row_value(payload, "phase", int(index)))].append(err)
            used[used_level] += 1
    return {
        "mae": float(mean(errors)) if errors else 0.0,
        "seed_balanced_mae": float(mean(mean(items) for items in seed_errors.values())) if seed_errors else 0.0,
        "count": len(errors),
        "used": dict(used.most_common()),
        "floor_mae": {key: float(mean(items)) for key, items in sorted(floor_errors.items())},
        "phase_mae": {
            key: {"count": len(items), "mae": float(mean(items))}
            for key, items in sorted(phase_errors.items())
        },
    }


def _eval_all_oracles(
    chunks: list[Path],
    fitted: list[tuple[str, tuple[str, ...], dict[tuple[Any, ...], float]]],
    *,
    global_median: float,
) -> dict[str, Any]:
    accum = []
    for name, _, _ in fitted:
        accum.append(
            {
                "name": name,
                "errors": [],
                "seed_errors": defaultdict(list),
                "floor_errors": defaultdict(list),
                "phase_errors": defaultdict(list),
                "used": Counter(),
            }
        )
    for path in chunks:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        mask = payload["is_validation"].bool()
        indexes = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        targets = payload["targets"][:, 0].float()
        for index in indexes:
            target = float(targets[int(index)].item())
            seed = int(payload["seeds"][int(index)].item())
            floor_bucket = _floor_bucket(int(payload["floors"][int(index)].item()), 5)
            phase = str(_row_value(payload, "phase", int(index)))
            active_levels: list[tuple[str, tuple[str, ...], dict[tuple[Any, ...], float]]] = []
            for level_index, level in enumerate(fitted):
                active_levels.append(level)
                pred = float(global_median)
                used_level = "global"
                for candidate_name, fields, table in reversed(active_levels):
                    key = tuple(_row_value(payload, field, int(index)) for field in fields)
                    if key in table:
                        pred = float(table[key])
                        used_level = candidate_name
                        break
                err = abs(pred - target)
                row = accum[level_index]
                row["errors"].append(err)
                row["seed_errors"][seed].append(err)
                row["floor_errors"][floor_bucket].append(err)
                row["phase_errors"][phase].append(err)
                row["used"][used_level] += 1
    results = {}
    for row in accum:
        errors = row["errors"]
        seed_errors = row["seed_errors"]
        floor_errors = row["floor_errors"]
        phase_errors = row["phase_errors"]
        results[str(row["name"])] = {
            "mae": float(mean(errors)) if errors else 0.0,
            "seed_balanced_mae": float(mean(mean(items) for items in seed_errors.values())) if seed_errors else 0.0,
            "count": len(errors),
            "used": dict(row["used"].most_common()),
            "floor_mae": {key: float(mean(items)) for key, items in sorted(floor_errors.items())},
            "phase_mae": {
                key: {"count": len(items), "mae": float(mean(items))}
                for key, items in sorted(phase_errors.items())
            },
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped median oracle for an existing run-value tensor cache.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train-count", type=int, default=30)
    args = parser.parse_args()

    require_torch()
    manifest = _load_manifest(args.cache_dir)
    chunks = _chunk_paths(manifest)
    level_defs: list[tuple[str, tuple[str, ...]]] = [
        ("G0_floor", ("floor",)),
        ("G1_floor_phase", ("floor", "phase")),
        ("G2_floor_phase_kind", ("floor", "phase", "sample_kind")),
        ("G3_floor_phase_source", ("floor", "phase", "source")),
        ("G4_floor_bucket_phase_source_kind", ("floor_bucket", "phase", "source", "sample_kind")),
    ]
    fitted, global_median = _fit_all_group_medians(
        chunks,
        level_defs,
        min_train_count=int(args.min_train_count),
    )
    results = _eval_all_oracles(chunks, fitted, global_median=global_median)
    summary = {
        "cache_dir": str(args.cache_dir),
        "manifest": {
            key: manifest.get(key)
            for key in (
                "count",
                "train_count",
                "validation_count",
                "state_feature_dim",
                "sample_mode",
                "feature_variant",
                "record_weight_mode",
            )
        },
        "min_train_count": int(args.min_train_count),
        "oracle": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
