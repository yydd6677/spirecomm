#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from spirecomm.ai.run_value import load_run_value_checkpoint, save_run_value_checkpoint
from spirecomm.ai.torch_compat import require_torch, torch
from train_run_value_model import (
    _clamp_remaining,
    _model_features,
    _iter_chunk_batches,
    _pred_remaining,
    _residual_baseline_for_batch,
    load_manifest,
)


def _floor_bucket(floor: int, width: int = 5) -> str:
    start = int(floor) // int(width) * int(width)
    return f"{start:02d}-{start + int(width) - 1:02d}"


def _group_key(batch: dict[str, Any], index: int, fields: tuple[str, ...]) -> tuple[Any, ...]:
    values: list[Any] = []
    for field in fields:
        if field == "floor":
            values.append(int(batch["floors"][index].item()))
        elif field == "floor_bucket":
            values.append(_floor_bucket(int(batch["floors"][index].item())))
        elif field == "phase":
            values.append(str(batch.get("phases", [""])[index]))
        elif field == "sample_kind":
            values.append(str(batch.get("sample_kinds", ["before"])[index]))
        elif field == "source":
            values.append(str(batch.get("sources", [""])[index]))
        else:
            raise KeyError(field)
    return tuple(values)


def _remaining_predictions(
    model: Any,
    batch: dict[str, Any],
    *,
    survival_bins: int,
    use_survival: bool,
    final_floor_bins: int,
    final_floor_readout: str,
    residual_floor_baseline: dict[str, Any] | None,
    device: str,
) -> tuple[Any, Any]:
    outputs = model(_model_features(model, batch["features"]))
    pred = _pred_remaining(
        outputs,
        batch["floors"],
        survival_bins=int(survival_bins),
        use_survival=bool(use_survival),
        final_floor_bins=int(final_floor_bins),
        final_floor_readout=str(final_floor_readout),
    ).detach().float().cpu()
    if residual_floor_baseline:
        pred += _residual_baseline_for_batch(
            batch["floors"],
            residual_floor_baseline=residual_floor_baseline,
            device="cpu",
            phases=batch.get("phases"),
            sources=batch.get("sources"),
            sample_kinds=batch.get("sample_kinds"),
        )
    pred = _clamp_remaining(pred, batch["floors"].detach().cpu())
    true = batch["targets"][:, 0].detach().float().cpu()
    return pred, true


def _fit_bias_tables(
    model: Any,
    chunks: list[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
    fields: tuple[str, ...],
    parent_fields: tuple[str, ...],
    min_count: int,
    shrink_count: float,
    survival_bins: int,
    use_survival: bool,
    final_floor_bins: int,
    final_floor_readout: str,
    residual_floor_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    rng = random.Random(2468)
    group_errors: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    parent_errors: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    with torch.inference_mode():
        for batch in _iter_chunk_batches(chunks, batch_size=batch_size, device=device, train=True, rng=rng):
            pred, true = _remaining_predictions(
                model,
                batch,
                survival_bins=survival_bins,
                use_survival=use_survival,
                final_floor_bins=final_floor_bins,
                final_floor_readout=final_floor_readout,
                residual_floor_baseline=residual_floor_baseline,
                device=device,
            )
            for index in range(int(true.shape[0])):
                err = float(true[index] - pred[index])
                group_errors[_group_key(batch, index, fields)].append(err)
                parent_errors[_group_key(batch, index, parent_fields)].append(err)
    parent_bias = {key: float(mean(values)) for key, values in parent_errors.items() if values}
    group_bias = {}
    for key, values in group_errors.items():
        if len(values) < int(min_count):
            continue
        parent_key = key[: len(parent_fields)]
        raw = float(mean(values))
        parent = float(parent_bias.get(parent_key, 0.0))
        n = float(len(values))
        group_bias[key] = (n / (n + float(shrink_count))) * raw + (float(shrink_count) / (n + float(shrink_count))) * parent
    return {
        "fields": list(fields),
        "parent_fields": list(parent_fields),
        "min_count": int(min_count),
        "shrink_count": float(shrink_count),
        "parent_bias": {"|".join(map(str, key)): value for key, value in parent_bias.items()},
        "group_bias": {"|".join(map(str, key)): value for key, value in group_bias.items()},
    }


def _eval_calibration(
    model: Any,
    chunks: list[dict[str, Any]],
    table: dict[str, Any],
    *,
    batch_size: int,
    device: str,
    survival_bins: int,
    use_survival: bool,
    final_floor_bins: int,
    final_floor_readout: str,
    residual_floor_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    fields = tuple(str(value) for value in table["fields"])
    parent_fields = tuple(str(value) for value in table["parent_fields"])
    group_bias = dict(table.get("group_bias") or {})
    parent_bias = dict(table.get("parent_bias") or {})
    rng = random.Random(1357)
    raw_abs = 0.0
    cal_abs = 0.0
    count = 0
    floor_errors: dict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for batch in _iter_chunk_batches(chunks, batch_size=batch_size, device=device, train=False, rng=rng):
            pred, true = _remaining_predictions(
                model,
                batch,
                survival_bins=survival_bins,
                use_survival=use_survival,
                final_floor_bins=final_floor_bins,
                final_floor_readout=final_floor_readout,
                residual_floor_baseline=residual_floor_baseline,
                device=device,
            )
            for index in range(int(true.shape[0])):
                group_key = "|".join(map(str, _group_key(batch, index, fields)))
                parent_key = "|".join(map(str, _group_key(batch, index, parent_fields)))
                bias = float(group_bias.get(group_key, parent_bias.get(parent_key, 0.0)))
                raw_err = abs(float(pred[index]) - float(true[index]))
                cal_err = abs(float(pred[index]) + bias - float(true[index]))
                raw_abs += raw_err
                cal_abs += cal_err
                floor_errors[_floor_bucket(int(batch["floors"][index].item()))].append(cal_err)
                count += 1
    return {
        "count": int(count),
        "raw_mae": raw_abs / max(1, count),
        "calibrated_mae": cal_abs / max(1, count),
        "floor_calibrated_mae": {key: float(mean(values)) for key, values in sorted(floor_errors.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit post-hoc group bias calibration for a run value model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--min-count", type=int, default=200)
    parser.add_argument("--shrink-count", type=float, default=1000.0)
    parser.add_argument(
        "--calibrated-checkpoint",
        type=Path,
        default=None,
        help="Optional output checkpoint with the best calibration table embedded in metadata.",
    )
    args = parser.parse_args()

    require_torch()
    model, checkpoint = load_run_value_checkpoint(args.checkpoint, device=str(args.device))
    metadata = checkpoint.get("metadata") if isinstance(checkpoint.get("metadata"), dict) else {}
    chunks = list(load_manifest(args.cache_dir).get("chunks") or [])
    if not chunks:
        raise SystemExit("cache has no chunks")
    survival_bins = int(metadata.get("survival_bins") or getattr(model, "survival_bins", 0) or 0)
    final_floor_bins = int(metadata.get("final_floor_bins") or getattr(model, "final_floor_bins", 0) or 0)
    final_floor_readout = str(metadata.get("final_floor_readout") or "none")
    residual_floor_baseline = metadata.get("residual_floor_baseline")
    configs = [
        ("floor", ("floor",), ("floor",)),
        ("floor_phase", ("floor", "phase"), ("floor",)),
        ("floor_phase_kind", ("floor", "phase", "sample_kind"), ("floor",)),
        ("floor_phase_source_kind", ("floor", "phase", "source", "sample_kind"), ("floor",)),
    ]
    results = {}
    tables = {}
    for name, fields, parent_fields in configs:
        table = _fit_bias_tables(
            model,
            chunks,
            batch_size=int(args.batch_size),
            device=str(args.device),
            fields=fields,
            parent_fields=parent_fields,
            min_count=int(args.min_count),
            shrink_count=float(args.shrink_count),
            survival_bins=survival_bins,
            use_survival=False,
            final_floor_bins=final_floor_bins,
            final_floor_readout=final_floor_readout,
            residual_floor_baseline=residual_floor_baseline,
        )
        tables[name] = table
        results[name] = _eval_calibration(
            model,
            chunks,
            table,
            batch_size=int(args.batch_size),
            device=str(args.device),
            survival_bins=survival_bins,
            use_survival=False,
            final_floor_bins=final_floor_bins,
            final_floor_readout=final_floor_readout,
            residual_floor_baseline=residual_floor_baseline,
        )
        print(f"{name}: raw={results[name]['raw_mae']:.4f} calibrated={results[name]['calibrated_mae']:.4f}", flush=True)
    best_name = min(results, key=lambda key: float(results[key]["calibrated_mae"])) if results else ""
    summary = {
        "checkpoint": str(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "min_count": int(args.min_count),
        "shrink_count": float(args.shrink_count),
        "best_table": best_name,
        "results": results,
        "tables": tables,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.calibrated_checkpoint and best_name:
        new_metadata = dict(metadata)
        new_metadata["value_calibration"] = tables[best_name]
        new_metadata["value_calibration_name"] = best_name
        new_metadata["value_calibration_metrics"] = results[best_name]
        save_run_value_checkpoint(args.calibrated_checkpoint, model, metadata=new_metadata)
    print(json.dumps({key: value for key, value in summary.items() if key != "tables"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
