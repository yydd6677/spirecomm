#!/usr/bin/env python3
from __future__ import annotations
# Allow this CLI to run directly from its workflow subdirectory.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, median
from typing import Any

from scripts.run_value.analyze_run_value_dataset import _row_from_state
from spirecomm.ai.run_value import iter_jsonl
from scripts.run_value.train_run_value_model import (
    _decision_files,
    _error_seeds,
    _seed_is_validation,
    _state_with_run_context,
)


LEVELS: list[tuple[str, tuple[str, ...]]] = [
    ("G0_floor_bucket", ("floor_bucket",)),
    ("G1_floor_phase", ("floor_bucket", "phase")),
    (
        "G2_basic_state",
        (
            "floor",
            "phase",
            "room_type",
            "act_boss",
            "hp_ratio_bin",
            "gold_bin",
            "deck_size_bin",
            "relic_count_bin",
            "potion_count",
            "act",
        ),
    ),
    (
        "G3_raw_signature",
        (
            "floor",
            "phase",
            "room_type",
            "act_boss",
            "hp_ratio_bin",
            "gold_bin",
            "deck_hash",
            "relic_hash",
            "potion_hash",
            "node_x",
            "node_y",
            "map_prefix_hash",
        ),
    ),
]


def _key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


_WORKER_CONFIG: dict[str, Any] = {}
_WORKER_EXCLUDED_ERROR_SEEDS: set[int] = set()
_WORKER_FITTED: list[tuple[str, tuple[str, ...], dict[tuple[Any, ...], float]]] = []


def _paths(args: argparse.Namespace) -> list[Path]:
    paths = _decision_files(args.input_dir)
    if int(args.max_files) > 0:
        paths = paths[: int(args.max_files)]
    return paths


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sample_mode": str(args.sample_mode),
        "feature_variant": str(args.feature_variant),
        "val_mod": int(args.val_mod),
        "val_rem": int(args.val_rem),
    }


def _init_worker(config: dict[str, Any], excluded_error_seeds: set[int], fitted: list[tuple[str, tuple[str, ...], dict[tuple[Any, ...], float]]] | None = None) -> None:
    global _WORKER_CONFIG, _WORKER_EXCLUDED_ERROR_SEEDS, _WORKER_FITTED
    _WORKER_CONFIG = dict(config)
    _WORKER_EXCLUDED_ERROR_SEEDS = set(excluded_error_seeds)
    _WORKER_FITTED = list(fitted or [])


def _iter_rows_from_path(path: str | Path, config: dict[str, Any], excluded_error_seeds: set[int]):
    context: dict[str, Any] = {}
    for record in iter_jsonl(path):
        seed = int(record.get("seed") or 0)
        if seed in excluded_error_seeds:
            continue
        terminal = record.get("terminal") if isinstance(record.get("terminal"), dict) else {}
        if bool(terminal.get("truncated")):
            continue
        before = _state_with_run_context(record, context, state_key="state_before", update_context=True)
        if isinstance(before, dict):
            yield _row_from_state(
                record,
                before,
                sample_kind="before",
                feature_variant=str(config["feature_variant"]),
                include_feature_signature=False,
                feature_quant_step=0.25,
            )
        if str(config["sample_mode"]) == "before_after":
            after = _state_with_run_context(record, context, state_key="state_after", update_context=False)
            if isinstance(after, dict):
                yield _row_from_state(
                    record,
                    after,
                    sample_kind="after",
                    feature_variant=str(config["feature_variant"]),
                    include_feature_signature=False,
                    feature_quant_step=0.25,
                )


def _empty_counts() -> dict[str, Any]:
    return {
        "train_rows": 0,
        "validation_rows": 0,
        "train_seeds": set(),
        "validation_seeds": set(),
        "phase": Counter(),
        "source": Counter(),
        "floor_bucket": Counter(),
        "sample_kind": Counter(),
    }


def _merge_counts(dest: dict[str, Any], src: dict[str, Any]) -> None:
    for key in ("train_rows", "validation_rows"):
        dest[key] += int(src[key])
    for key in ("train_seeds", "validation_seeds"):
        dest[key].update(src[key])
    for key in ("phase", "source", "floor_bucket", "sample_kind"):
        dest[key].update(src[key])


def _fit_file_worker(path: str) -> dict[str, Any]:
    values_by_level: list[dict[tuple[Any, ...], list[float]]] = [defaultdict(list) for _ in LEVELS]
    counts = _empty_counts()
    for row in _iter_rows_from_path(path, _WORKER_CONFIG, _WORKER_EXCLUDED_ERROR_SEEDS):
        seed = int(row["seed"])
        is_validation = _seed_is_validation(seed, val_mod=int(_WORKER_CONFIG["val_mod"]), val_rem=int(_WORKER_CONFIG["val_rem"]))
        if is_validation:
            counts["validation_rows"] += 1
            counts["validation_seeds"].add(seed)
            continue
        counts["train_rows"] += 1
        counts["train_seeds"].add(seed)
        for field in ("phase", "source", "floor_bucket", "sample_kind"):
            counts[field][str(row.get(field) or "")] += 1
        target = float(row["target_remaining"])
        for level_index, (_, fields) in enumerate(LEVELS):
            values_by_level[level_index][_key(row, fields)].append(target)
    return {"values": [dict(values) for values in values_by_level], "counts": counts}


def _empty_eval_accum() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "abs_sum": 0.0,
            "count": 0,
            "seed_errors": {},
            "floor_errors": {},
            "phase_errors": {},
            "used": Counter(),
        }
        for name, _ in LEVELS
    ]


def _add_error(bucket: dict[Any, list[Any]], key: Any, value: float) -> None:
    item = bucket.setdefault(key, [0.0, 0])
    item[0] += float(value)
    item[1] += 1


def _merge_eval_accum(dest: list[dict[str, Any]], src: list[dict[str, Any]]) -> None:
    for target, source in zip(dest, src):
        target["abs_sum"] += float(source["abs_sum"])
        target["count"] += int(source["count"])
        target["used"].update(source["used"])
        for field in ("seed_errors", "floor_errors", "phase_errors"):
            for key, value in source[field].items():
                item = target[field].setdefault(key, [0.0, 0])
                item[0] += float(value[0])
                item[1] += int(value[1])


def _eval_file_worker(path: str) -> list[dict[str, Any]]:
    accum = _empty_eval_accum()
    for row in _iter_rows_from_path(path, _WORKER_CONFIG, _WORKER_EXCLUDED_ERROR_SEEDS):
        seed = int(row["seed"])
        if not _seed_is_validation(seed, val_mod=int(_WORKER_CONFIG["val_mod"]), val_rem=int(_WORKER_CONFIG["val_rem"])):
            continue
        target = float(row["target_remaining"])
        for level_index in range(len(_WORKER_FITTED)):
            pred = 0.0
            used = "global_zero"
            for name, fields, table in reversed(_WORKER_FITTED[: level_index + 1]):
                key = _key(row, fields)
                if key in table:
                    pred = float(table[key])
                    used = name
                    break
            err = abs(pred - target)
            item = accum[level_index]
            item["abs_sum"] += err
            item["count"] += 1
            _add_error(item["seed_errors"], seed, err)
            _add_error(item["floor_errors"], str(row.get("floor_bucket") or ""), err)
            _add_error(item["phase_errors"], str(row.get("phase") or ""), err)
            item["used"][used] += 1
    return accum


def _finalize_eval_accum(accum: list[dict[str, Any]]) -> dict[str, Any]:
    results = {}
    for item in accum:
        count = int(item["count"])
        seed_errors = item["seed_errors"]
        results[str(item["name"])] = {
            "mae": float(item["abs_sum"]) / max(1, count),
            "seed_balanced_mae": float(mean(float(value[0]) / max(1, int(value[1])) for value in seed_errors.values())) if seed_errors else 0.0,
            "count": count,
            "used": dict(item["used"].most_common()),
            "floor_mae": {key: float(value[0]) / max(1, int(value[1])) for key, value in sorted(item["floor_errors"].items())},
            "phase_mae": {
                key: {"count": int(value[1]), "mae": float(value[0]) / max(1, int(value[1]))}
                for key, value in sorted(item["phase_errors"].items())
            },
        }
    return results


def _iter_rows(args: argparse.Namespace):
    paths = _paths(args)
    excluded_error_seeds = _error_seeds(args.input_dir)
    config = _config(args)
    total_paths = len(paths)
    for file_index, path in enumerate(paths, start=1):
        yield from _iter_rows_from_path(path, config, excluded_error_seeds)
        if file_index == 1 or file_index == total_paths or file_index % int(args.progress_interval) == 0:
            print(f"scanned files={file_index}/{total_paths}", flush=True)


def _fit_tables(args: argparse.Namespace) -> tuple[list[tuple[str, tuple[str, ...], dict[tuple[Any, ...], float]]], dict[str, Any]]:
    values_by_level: list[dict[tuple[Any, ...], list[float]]] = [defaultdict(list) for _ in LEVELS]
    counts = _empty_counts()
    paths = _paths(args)
    if int(args.workers) > 1:
        excluded_error_seeds = _error_seeds(args.input_dir)
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_init_worker,
            initargs=(_config(args), excluded_error_seeds, None),
        ) as executor:
            for file_index, result in enumerate(
                executor.map(_fit_file_worker, [str(path) for path in paths], chunksize=int(args.worker_chunksize)),
                start=1,
            ):
                _merge_counts(counts, result["counts"])
                for level_index, groups in enumerate(result["values"]):
                    target_groups = values_by_level[level_index]
                    for key, values in groups.items():
                        target_groups[key].extend(values)
                if file_index == 1 or file_index == len(paths) or file_index % int(args.progress_interval) == 0:
                    print(f"fit files={file_index}/{len(paths)} train_rows={counts['train_rows']} val_rows={counts['validation_rows']}", flush=True)
    else:
        for row in _iter_rows(args):
            seed = int(row["seed"])
            is_validation = _seed_is_validation(seed, val_mod=int(args.val_mod), val_rem=int(args.val_rem))
            if is_validation:
                counts["validation_rows"] += 1
                counts["validation_seeds"].add(seed)
                continue
            counts["train_rows"] += 1
            counts["train_seeds"].add(seed)
            for field in ("phase", "source", "floor_bucket", "sample_kind"):
                counts[field][str(row.get(field) or "")] += 1
            target = float(row["target_remaining"])
            for level_index, (_, fields) in enumerate(LEVELS):
                values_by_level[level_index][_key(row, fields)].append(target)
    fitted = []
    for (name, fields), groups in zip(LEVELS, values_by_level):
        table = {
            key: float(median(values))
            for key, values in groups.items()
            if len(values) >= int(args.min_train_count)
        }
        fitted.append((name, fields, table))
        print(f"fitted {name}: groups={len(table)}/{len(groups)}", flush=True)
    serial_counts = {
        "train_rows": int(counts["train_rows"]),
        "validation_rows": int(counts["validation_rows"]),
        "unique_train_seeds": len(counts["train_seeds"]),
        "unique_validation_seeds": len(counts["validation_seeds"]),
        "phase": dict(counts["phase"].most_common(100)),
        "source": dict(counts["source"].most_common(100)),
        "floor_bucket": dict(counts["floor_bucket"].most_common(100)),
        "sample_kind": dict(counts["sample_kind"].most_common(100)),
    }
    return fitted, serial_counts


def _eval_tables(args: argparse.Namespace, fitted: list[tuple[str, tuple[str, ...], dict[tuple[Any, ...], float]]]) -> dict[str, Any]:
    accum = _empty_eval_accum()
    paths = _paths(args)
    if int(args.workers) > 1:
        excluded_error_seeds = _error_seeds(args.input_dir)
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_init_worker,
            initargs=(_config(args), excluded_error_seeds, fitted),
        ) as executor:
            for file_index, result in enumerate(
                executor.map(_eval_file_worker, [str(path) for path in paths], chunksize=int(args.worker_chunksize)),
                start=1,
            ):
                _merge_eval_accum(accum, result)
                if file_index == 1 or file_index == len(paths) or file_index % int(args.progress_interval) == 0:
                    print(f"eval files={file_index}/{len(paths)} val_rows={accum[0]['count']}", flush=True)
    else:
        _init_worker(_config(args), _error_seeds(args.input_dir), fitted)
        for file_index, path in enumerate(paths, start=1):
            _merge_eval_accum(accum, _eval_file_worker(str(path)))
            if file_index == 1 or file_index == len(paths) or file_index % int(args.progress_interval) == 0:
                print(f"eval files={file_index}/{len(paths)} val_rows={accum[0]['count']}", flush=True)
    return _finalize_eval_accum(accum)


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming grouped oracle for run-value decision logs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-mode", choices=["before", "before_after"], default="before_after")
    parser.add_argument("--feature-variant", default="no_seed_structrng")
    parser.add_argument("--val-mod", type=int, default=10)
    parser.add_argument("--val-rem", type=int, default=0)
    parser.add_argument("--min-train-count", type=int, default=30)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-chunksize", type=int, default=4)
    args = parser.parse_args()

    fitted, counts = _fit_tables(args)
    results = _eval_tables(args, fitted)
    summary = {
        "input_dir": str(args.input_dir),
        "sample_mode": str(args.sample_mode),
        "feature_variant": str(args.feature_variant),
        "min_train_count": int(args.min_train_count),
        "counts": counts,
        "oracle": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
