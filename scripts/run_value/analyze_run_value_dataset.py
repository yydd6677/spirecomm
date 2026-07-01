#!/usr/bin/env python3
from __future__ import annotations
# Allow this CLI to run directly from its workflow subdirectory.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from spirecomm.ai.card_reward_model import normalize_token
from spirecomm.ai.run_value import encode_run_state, iter_jsonl
from scripts.run_value.train_run_value_model import (
    _decision_files,
    _error_seeds,
    _floor_bucket,
    _seed_is_validation,
    _state_with_run_context,
    _target_from_record,
)


def _bucket(value: float, width: float) -> int:
    if width <= 0.0:
        return int(value)
    return int(float(value) // float(width))


def _list_hash(values: list[Any]) -> str:
    normalized = [normalize_token(value) for value in values if normalize_token(value)]
    blob = json.dumps(sorted(normalized), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _card_multiset_hash(cards: list[dict[str, Any]]) -> str:
    values = []
    for card in cards:
        key = normalize_token(card.get("card_id") or card.get("id") or card.get("name") or "")
        if key:
            upgrades = int(card.get("upgrades") or 0)
            values.append(f"{key}+{upgrades}")
    return _list_hash(values)


def _id_set_hash(items: list[dict[str, Any]], *keys: str) -> str:
    values = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in keys:
            value = item.get(key)
            if value:
                values.append(str(value))
                break
    return _list_hash(values)


def _safe_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, (list, tuple)) else []


def _map_prefix_hash(state: dict[str, Any]) -> str:
    map_state = state.get("map_state") if isinstance(state.get("map_state"), dict) else {}
    current = map_state.get("current_node") if isinstance(map_state.get("current_node"), dict) else {}
    current_y = int(current.get("y") if current.get("y") is not None else (int(state.get("floor") or 0) - 1) % 17)
    nodes = [node for node in _safe_list(map_state.get("nodes")) if isinstance(node, dict)]
    values = []
    for node in nodes:
        y = int(node.get("y") or 0)
        if current_y < y <= current_y + 3:
            values.append(f"{y}:{node.get('x')}:{node.get('symbol')}:{bool(node.get('emerald'))}")
    return _list_hash(values)


def _feature_signature(state: dict[str, Any], *, feature_variant: str, step: float) -> str:
    values = encode_run_state(state, feature_variant=feature_variant)
    quantized = [int(round(float(value) / float(step))) for value in values]
    blob = ",".join(str(value) for value in quantized)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:20]


def _row_from_state(
    record: dict[str, Any],
    state: dict[str, Any],
    *,
    sample_kind: str,
    feature_variant: str,
    include_feature_signature: bool,
    feature_quant_step: float,
) -> dict[str, Any]:
    target = _target_from_record(record, state)
    floor = int(state.get("floor") or record.get("floor") or 0)
    hp = int(state.get("current_hp") or 0)
    max_hp = max(1, int(state.get("max_hp") or 1))
    deck = [card for card in _safe_list(state.get("deck")) if isinstance(card, dict)]
    relics = [relic for relic in _safe_list(state.get("relics")) if isinstance(relic, dict)]
    potions = [potion for potion in _safe_list(state.get("potions")) if isinstance(potion, dict)]
    map_state = state.get("map_state") if isinstance(state.get("map_state"), dict) else {}
    current_node = map_state.get("current_node") if isinstance(map_state.get("current_node"), dict) else {}
    row = {
        "seed": int(record.get("seed") or 0),
        "sample_kind": str(sample_kind),
        "phase": str(record.get("phase") or state.get("phase") or ""),
        "source": str(record.get("source") or ""),
        "floor": floor,
        "floor_bucket": _floor_bucket(floor),
        "target_remaining": float(target[0]),
        "target_final": float(target[1]),
        "room_type": str(record.get("room_type") or state.get("room_type") or ""),
        "act": int(state.get("act") or 0),
        "act_boss": normalize_token(state.get("act_boss") or ""),
        "hp_ratio_bin": _bucket(hp / max_hp, 0.1),
        "gold_bin": _bucket(float(state.get("gold") or 0), 50.0),
        "deck_size_bin": _bucket(len(deck), 5.0),
        "relic_count_bin": _bucket(len(relics), 3.0),
        "potion_count": sum(1 for potion in potions if normalize_token(potion.get("potion_id") or potion.get("id") or potion.get("name") or "") not in {"", "potionslot"}),
        "deck_hash": _card_multiset_hash(deck),
        "relic_hash": _id_set_hash(relics, "relic_id", "id", "name"),
        "potion_hash": _id_set_hash(potions, "potion_id", "id", "name"),
        "node_x": int(current_node.get("x") or -1),
        "node_y": int(current_node.get("y") or -1),
        "map_prefix_hash": _map_prefix_hash(state),
    }
    if include_feature_signature:
        row["feature_sig"] = _feature_signature(
            state,
            feature_variant=feature_variant,
            step=float(feature_quant_step),
        )
    return row


def _load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = _decision_files(args.input_dir)
    if args.max_files > 0:
        paths = paths[: int(args.max_files)]
    excluded_error_seeds = _error_seeds(args.input_dir)
    for file_index, path in enumerate(paths, start=1):
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
                rows.append(
                    _row_from_state(
                        record,
                        before,
                        sample_kind="before",
                        feature_variant=str(args.feature_variant),
                        include_feature_signature=bool(args.include_feature_signature),
                        feature_quant_step=float(args.feature_quant_step),
                    )
                )
            if str(args.sample_mode) == "before_after":
                after = _state_with_run_context(record, context, state_key="state_after", update_context=False)
                if isinstance(after, dict):
                    rows.append(
                        _row_from_state(
                            record,
                            after,
                            sample_kind="after",
                            feature_variant=str(args.feature_variant),
                            include_feature_signature=bool(args.include_feature_signature),
                            feature_quant_step=float(args.feature_quant_step),
                        )
                    )
            if args.max_rows > 0 and len(rows) >= int(args.max_rows):
                return rows[: int(args.max_rows)]
        if file_index == 1 or file_index == len(paths) or file_index % 250 == 0:
            print(f"loaded files={file_index}/{len(paths)} rows={len(rows)}", flush=True)
    return rows


def _split_rows(rows: list[dict[str, Any]], *, val_mod: int, val_rem: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = []
    valid = []
    for row in rows:
        if _seed_is_validation(int(row["seed"]), val_mod=val_mod, val_rem=val_rem):
            valid.append(row)
        else:
            train.append(row)
    return train, valid


def _key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def _fit_medians(rows: list[dict[str, Any]], fields: tuple[str, ...], *, min_train_count: int) -> dict[tuple[Any, ...], float]:
    values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        values[_key(row, fields)].append(float(row["target_remaining"]))
    return {
        key: float(median(targets))
        for key, targets in values.items()
        if len(targets) >= int(min_train_count)
    }


def _oracle(
    train: list[dict[str, Any]],
    valid: list[dict[str, Any]],
    levels: list[tuple[str, tuple[str, ...]]],
    *,
    min_train_count: int,
) -> dict[str, Any]:
    medians = [(name, fields, _fit_medians(train, fields, min_train_count=min_train_count)) for name, fields in levels]
    global_median = float(median([float(row["target_remaining"]) for row in train])) if train else 0.0
    errors = []
    seed_errors: dict[int, list[float]] = defaultdict(list)
    floor_errors: dict[str, list[float]] = defaultdict(list)
    phase_errors: dict[str, list[float]] = defaultdict(list)
    used_counts: Counter[str] = Counter()
    for row in valid:
        pred = global_median
        used = "global"
        for name, fields, table in reversed(medians):
            key = _key(row, fields)
            if key in table:
                pred = table[key]
                used = name
                break
        error = abs(pred - float(row["target_remaining"]))
        errors.append(error)
        seed_errors[int(row["seed"])].append(error)
        floor_errors[str(row["floor_bucket"])].append(error)
        phase_errors[str(row["phase"])].append(error)
        used_counts[used] += 1
    return {
        "mae": float(mean(errors)) if errors else 0.0,
        "seed_balanced_mae": float(mean([mean(values) for values in seed_errors.values()])) if seed_errors else 0.0,
        "count": len(valid),
        "used": dict(used_counts.most_common()),
        "floor_mae": {key: float(mean(values)) for key, values in sorted(floor_errors.items())},
        "phase_mae": {key: {"count": len(values), "mae": float(mean(values))} for key, values in sorted(phase_errors.items())},
    }


def _counts(rows: list[dict[str, Any]], field: str, *, top: int = 100) -> dict[str, int]:
    counter = Counter(str(row.get(field) or "") for row in rows)
    return dict(counter.most_common(top))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze run value dataset balance and grouped-oracle lower bounds.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-mode", choices=["before", "before_after"], default="before")
    parser.add_argument(
        "--feature-variant",
        choices=[
            "current",
            "no_seed_structrng",
            "no_seed_no_rng",
            "current_explicit",
            "no_seed_structrng_explicit",
            "no_seed_no_rng_explicit",
            "current_explicit_aug",
            "no_seed_structrng_explicit_aug",
            "no_seed_no_rng_explicit_aug",
        ],
        default="no_seed_structrng",
    )
    parser.add_argument("--include-feature-signature", action="store_true")
    parser.add_argument("--feature-quant-step", type=float, default=0.25)
    parser.add_argument("--val-mod", type=int, default=10)
    parser.add_argument("--val-rem", type=int, default=0)
    parser.add_argument("--min-train-count", type=int, default=30)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    rows = _load_rows(args)
    train, valid = _split_rows(rows, val_mod=int(args.val_mod), val_rem=int(args.val_rem))
    levels: list[tuple[str, tuple[str, ...]]] = [
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
    if bool(args.include_feature_signature):
        levels.append(("G4_feature_signature", ("feature_sig",)))
    oracle_results = {}
    for index in range(len(levels)):
        name = levels[index][0]
        oracle_results[name] = _oracle(train, valid, levels[: index + 1], min_train_count=int(args.min_train_count))

    summary = {
        "input_dir": str(args.input_dir),
        "sample_mode": str(args.sample_mode),
        "feature_variant": str(args.feature_variant),
        "include_feature_signature": bool(args.include_feature_signature),
        "rows": len(rows),
        "train_rows": len(train),
        "validation_rows": len(valid),
        "unique_train_seeds": len({int(row["seed"]) for row in train}),
        "unique_validation_seeds": len({int(row["seed"]) for row in valid}),
        "counts": {
            "phase": _counts(rows, "phase"),
            "source": _counts(rows, "source"),
            "floor_bucket": _counts(rows, "floor_bucket"),
            "sample_kind": _counts(rows, "sample_kind"),
        },
        "oracle": oracle_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
