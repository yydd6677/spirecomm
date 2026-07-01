#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any

from spirecomm.ai.run_value import (
    RUN_VALUE_OUTPUTS,
    STATE_FEATURE_DIM,
    RunValueNetwork,
    encode_run_state,
    iter_jsonl,
    save_run_value_checkpoint,
)
from spirecomm.ai.torch_compat import F, require_torch, torch


TARGET_DIM = len(RUN_VALUE_OUTPUTS)
SAMPLE_KIND_ORDER = ("before", "after")
PHASE_TARGET_SHARE = {
    "COMBAT": 0.35,
    "CARD_REWARD": 0.18,
    "MAP": 0.12,
    "CARD_SELECT": 0.10,
    "SHOP": 0.08,
    "EVENT": 0.06,
    "CAMPFIRE": 0.04,
    "BOSS_RELIC": 0.03,
    "TREASURE": 0.02,
    "NEOW": 0.02,
}
COARSE_SURVIVAL_THRESHOLDS = (8, 12, 16, 20, 24, 28, 32, 34, 38, 42, 46, 50)
COARSE_FINAL_FLOOR_BINS = (
    (0, 6, 3.0),
    (7, 12, 9.5),
    (13, 16, 14.5),
    (17, 23, 20.0),
    (24, 33, 28.5),
    (34, 40, 37.0),
    (41, 49, 45.0),
    (50, 50, 50.0),
)
RUN_CONTEXT_KEYS = (
    "seed",
    "act",
    "act_boss",
    "ascension_level",
    "dungeon_id",
    "boss_relic_pool",
    "colorless_card_pool",
    "common_card_pool",
    "uncommon_card_pool",
    "rare_card_pool",
    "curse_card_pool",
    "src_colorless_card_pool",
    "src_common_card_pool",
    "src_uncommon_card_pool",
    "src_rare_card_pool",
    "src_curse_card_pool",
    "common_relic_pool",
    "uncommon_relic_pool",
    "rare_relic_pool",
    "shop_relic_pool",
    "has_emerald_key",
    "has_ruby_key",
    "has_sapphire_key",
    "map_state",
    "rng_state",
)


def _decision_files(input_dir: Path) -> list[Path]:
    decision_dir = input_dir / "decisions"
    if decision_dir.exists():
        paths = sorted(decision_dir.glob("*.jsonl")) + sorted(decision_dir.glob("*.jsonl.gz"))
    else:
        paths = sorted(input_dir.glob("*.jsonl")) + sorted(input_dir.glob("*.jsonl.gz"))
    return [path for path in paths if path.is_file()]


def _error_seeds(input_dir: Path) -> set[int]:
    results_path = input_dir / "results.jsonl"
    if not results_path.exists():
        return set()
    seeds: set[int] = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("error"):
            seeds.add(int(row.get("seed") or 0))
    seeds.discard(0)
    return seeds


def _target_from_record(record: dict[str, Any], state: dict[str, Any] | None = None) -> list[float]:
    terminal = dict(record.get("terminal") or {})
    floor = int((state or {}).get("floor") if isinstance(state, dict) else record.get("floor") or 0)
    final_floor = float(terminal.get("final_floor") if terminal.get("final_floor") is not None else floor)
    return [
        float(final_floor - floor),
        float(final_floor),
        float(bool(terminal.get("won"))),
        float(bool(terminal.get("act1_clear"))),
        float(bool(terminal.get("act2_clear"))),
        float(bool(terminal.get("act3_clear"))),
        float(bool(record.get("death_next_3"))),
        float(bool(record.get("death_next_6"))),
    ]


def _seed_is_validation(seed: int, *, val_mod: int, val_rem: int) -> bool:
    return int(val_mod) > 0 and int(seed) % int(val_mod) == int(val_rem)


def _context_value_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, str)):
        return bool(value)
    return True


def _update_run_context(context: dict[str, Any], state: dict[str, Any]) -> None:
    for key in RUN_CONTEXT_KEYS:
        if key in state and _context_value_is_present(state.get(key)):
            context[key] = state[key]


def _state_with_run_context(
    record: dict[str, Any],
    context: dict[str, Any],
    *,
    state_key: str = "state_before",
    update_context: bool = True,
) -> dict[str, Any] | None:
    raw_state = record.get(state_key)
    if not isinstance(raw_state, dict):
        return None
    if update_context:
        _update_run_context(context, raw_state)
    state = dict(raw_state)
    for key, value in context.items():
        if not _context_value_is_present(state.get(key)):
            state[key] = value
    # Combat states are emitted by NativeCombatEnv and can lack run-level fields.
    # The top-level record still carries these stable identifiers.
    for key in ("seed", "floor", "phase", "room_type"):
        if not _context_value_is_present(state.get(key)) and record.get(key) is not None:
            state[key] = record.get(key)
    return state


def _stable_record_signature(record: dict[str, Any]) -> str:
    visible = {
        "phase": record.get("phase"),
        "floor": record.get("floor"),
        "room_type": record.get("room_type"),
        "source": record.get("source"),
        "choice_list": (record.get("state_before") or {}).get("choice_list") if isinstance(record.get("state_before"), dict) else None,
        "screen_state": (record.get("state_before") or {}).get("screen_state") if isinstance(record.get("state_before"), dict) else None,
        "legal_actions": record.get("legal_actions") or [],
    }
    blob = json.dumps(visible, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _floor_bucket(floor: int) -> str:
    start = int(floor) // 5 * 5
    return f"{start:02d}-{start + 4:02d}"


def _record_raw_weights(records: list[dict[str, Any]], *, record_weight_mode: str) -> list[float]:
    if str(record_weight_mode) == "none" or not records:
        return [1.0] * len(records)
    cap_groups: dict[tuple[Any, ...], int] = defaultdict(int)
    phase_counts: dict[str, int] = defaultdict(int)
    floor_counts: dict[str, int] = defaultdict(int)
    for record in records:
        seed = int(record.get("seed") or 0)
        floor = int(record.get("floor") or 0)
        phase = str(record.get("phase") or "")
        source = str(record.get("source") or "")
        room_type = str(record.get("room_type") or "")
        if phase == "COMBAT":
            cap_key = ("combat", seed, floor, phase, room_type, source)
        elif phase in {"CARD_REWARD", "SHOP", "EVENT", "BOSS_RELIC", "CARD_SELECT", "CAMPFIRE"}:
            cap_key = ("screen", seed, floor, phase, source, _stable_record_signature(record))
        else:
            cap_key = ("other", seed, floor, phase, source)
        cap_groups[cap_key] += 1
        phase_counts[phase] += 1
        floor_counts[_floor_bucket(floor)] += 1

    observed_phase_share = {
        phase: count / max(1, len(records))
        for phase, count in phase_counts.items()
    }
    observed_floor_share = {
        bucket: count / max(1, len(records))
        for bucket, count in floor_counts.items()
    }
    uniform_floor_share = 1.0 / max(1, len(floor_counts))

    weights: list[float] = []
    for record in records:
        seed = int(record.get("seed") or 0)
        floor = int(record.get("floor") or 0)
        phase = str(record.get("phase") or "")
        source = str(record.get("source") or "")
        room_type = str(record.get("room_type") or "")
        if phase == "COMBAT":
            cap_key = ("combat", seed, floor, phase, room_type, source)
            cap = min(1.0, 8.0 / max(1, cap_groups[cap_key]))
        elif phase in {"CARD_REWARD", "SHOP", "EVENT", "BOSS_RELIC", "CARD_SELECT", "CAMPFIRE"}:
            cap_key = ("screen", seed, floor, phase, source, _stable_record_signature(record))
            cap = min(1.0, 4.0 / max(1, cap_groups[cap_key]))
        else:
            cap = 1.0
        if source == "forced_single":
            cap *= 0.10

        target_phase = PHASE_TARGET_SHARE.get(phase, 0.02)
        phase_multiplier = target_phase / max(1.0e-8, observed_phase_share.get(phase, 1.0))
        phase_multiplier = max(0.25, min(4.0, phase_multiplier))

        floor_multiplier = math.sqrt(uniform_floor_share / max(1.0e-8, observed_floor_share.get(_floor_bucket(floor), uniform_floor_share)))
        floor_multiplier = max(0.50, min(2.00, floor_multiplier))
        weights.append(float(cap * phase_multiplier * floor_multiplier))
    return weights


def _records_from_decision_file(path: str, excluded_error_seeds: set[int], worker_config: dict[str, Any] | None = None) -> dict[str, Any]:
    worker_config = dict(worker_config or {})
    sample_mode = str(worker_config.get("sample_mode") or "before")
    feature_variant = str(worker_config.get("feature_variant") or "current")
    record_weight_mode = str(worker_config.get("record_weight_mode") or "none")
    before_weight = float(worker_config.get("before_weight") or 0.4)
    after_weight = float(worker_config.get("after_weight") or 0.6)
    features: list[list[float]] = []
    targets: list[list[float]] = []
    seeds: list[int] = []
    floors: list[int] = []
    phases: list[str] = []
    sources: list[str] = []
    sample_kinds: list[str] = []
    row_weights: list[float] = []
    skipped_error_seed_records = 0
    context: dict[str, Any] = {}
    usable_records: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]] = []
    for record in iter_jsonl(Path(path)):
        seed = int(record.get("seed") or 0)
        if seed in excluded_error_seeds:
            skipped_error_seed_records += 1
            continue
        terminal = record.get("terminal") if isinstance(record.get("terminal"), dict) else {}
        if bool(terminal.get("truncated")):
            continue
        before_state = _state_with_run_context(record, context, state_key="state_before", update_context=True)
        if not isinstance(before_state, dict):
            continue
        after_state = (
            _state_with_run_context(record, context, state_key="state_after", update_context=False)
            if sample_mode == "before_after"
            else None
        )
        usable_records.append((record, before_state, after_state))

    record_weights = _record_raw_weights([record for record, _, _ in usable_records], record_weight_mode=record_weight_mode)
    if str(record_weight_mode) != "none":
        total = sum(record_weights)
        if total > 0.0:
            # Keep each run/seed at roughly one unit of total record weight.
            record_weights = [weight / total for weight in record_weights]
    for (record, before_state, after_state), record_weight in zip(usable_records, record_weights):
        seed = int(record.get("seed") or 0)
        source = str(record.get("source") or "")
        phase = str(record.get("phase") or "")
        features.append(encode_run_state(before_state, feature_variant=feature_variant))
        targets.append(_target_from_record(record, before_state))
        seeds.append(seed)
        floors.append(int(before_state.get("floor") or record.get("floor") or 0))
        phases.append(phase)
        sources.append(source)
        sample_kinds.append("before")
        row_weights.append(float(record_weight) * (before_weight if sample_mode == "before_after" else 1.0))
        if sample_mode == "before_after" and isinstance(after_state, dict):
            features.append(encode_run_state(after_state, feature_variant=feature_variant))
            targets.append(_target_from_record(record, after_state))
            seeds.append(seed)
            floors.append(int(after_state.get("floor") or record.get("floor") or 0))
            phases.append(phase)
            sources.append(source)
            sample_kinds.append("after")
            row_weights.append(float(record_weight) * after_weight)
    return {
        "path": path,
        "features": features,
        "targets": targets,
        "seeds": seeds,
        "floors": floors,
        "phases": phases,
        "sources": sources,
        "sample_kinds": sample_kinds,
        "row_weights": row_weights,
        "count": len(features),
        "skipped_error_seed_records": skipped_error_seed_records,
    }


def _result_to_tensor_payload(
    result: dict[str, Any],
    *,
    val_mod: int,
    val_rem: int,
    feature_dtype: Any,
) -> dict[str, Any]:
    seed_tensor = torch.tensor(result["seeds"], dtype=torch.long)
    is_validation = torch.tensor(
        [_seed_is_validation(int(seed), val_mod=val_mod, val_rem=val_rem) for seed in result["seeds"]],
        dtype=torch.bool,
    )
    return {
        "features": torch.tensor(result["features"], dtype=feature_dtype),
        "targets": torch.tensor(result["targets"], dtype=torch.float32),
        "seeds": seed_tensor,
        "floors": torch.tensor(result["floors"], dtype=torch.long),
        "is_validation": is_validation,
        "phases": list(result["phases"]),
        "sources": list(result["sources"]),
        "sample_kinds": list(result["sample_kinds"]),
        "row_weights": torch.tensor(result["row_weights"], dtype=torch.float32),
    }


def _cache_decision_file_direct(args: tuple[int, str, set[int], dict[str, Any], str, int, int, str]) -> dict[str, Any]:
    index, path, excluded_error_seeds, worker_config, temp_dir, val_mod, val_rem, cache_feature_dtype = args
    feature_dtype = torch.float16 if str(cache_feature_dtype) == "float16" else torch.float32
    result = _records_from_decision_file(path, excluded_error_seeds, worker_config)
    payload = _result_to_tensor_payload(result, val_mod=int(val_mod), val_rem=int(val_rem), feature_dtype=feature_dtype)
    temp_path = Path(temp_dir) / f"part_{int(index):05d}.pt"
    torch.save(payload, temp_path)
    is_validation = payload["is_validation"].bool()
    return {
        "path": result["path"],
        "temp_path": str(temp_path),
        "count": int(payload["features"].shape[0]),
        "train_count": int((~is_validation).sum().item()),
        "validation_count": int(is_validation.sum().item()),
        "skipped_error_seed_records": int(result.get("skipped_error_seed_records") or 0),
    }


def build_cache(
    *,
    input_dir: Path,
    cache_dir: Path,
    chunk_size: int,
    val_mod: int,
    val_rem: int,
    max_records: int,
    cache_workers: int = 1,
    sample_mode: str = "before",
    feature_variant: str = "current",
    record_weight_mode: str = "none",
    before_weight: float = 0.4,
    after_weight: float = 0.6,
    cache_feature_dtype: str = "float32",
    cache_direct_write: bool = False,
) -> dict[str, Any]:
    require_torch()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stale in cache_dir.glob("chunk_*.pt"):
        stale.unlink()
    paths = _decision_files(input_dir)
    chunks = []
    features: list[list[float]] = []
    targets: list[list[float]] = []
    seeds: list[int] = []
    floors: list[int] = []
    phases: list[str] = []
    sources: list[str] = []
    sample_kinds: list[str] = []
    row_weights: list[float] = []
    pending_payloads: list[dict[str, Any]] = []
    pending_count = 0
    count = 0
    train_count = 0
    val_count = 0
    source_files = []
    excluded_error_seeds = _error_seeds(input_dir)
    skipped_error_seed_records = 0
    feature_dtype = torch.float16 if str(cache_feature_dtype) == "float16" else torch.float32

    def flush() -> None:
        nonlocal features, targets, seeds, floors, phases, sources, sample_kinds, row_weights, train_count, val_count
        if not features:
            return
        chunk_index = len(chunks)
        seed_tensor = torch.tensor(seeds, dtype=torch.long)
        is_validation = torch.tensor(
            [_seed_is_validation(int(seed), val_mod=val_mod, val_rem=val_rem) for seed in seeds],
            dtype=torch.bool,
        )
        payload = {
            "features": torch.tensor(features, dtype=feature_dtype),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "seeds": seed_tensor,
            "floors": torch.tensor(floors, dtype=torch.long),
            "is_validation": is_validation,
            "phases": list(phases),
            "sources": list(sources),
            "sample_kinds": list(sample_kinds),
            "row_weights": torch.tensor(row_weights, dtype=torch.float32),
        }
        chunk_path = cache_dir / f"chunk_{chunk_index:05d}.pt"
        torch.save(payload, chunk_path)
        train_count += int((~is_validation).sum().item())
        val_count += int(is_validation.sum().item())
        chunks.append({"path": str(chunk_path), "count": len(features)})
        features = []
        targets = []
        seeds = []
        floors = []
        phases = []
        sources = []
        sample_kinds = []
        row_weights = []

    def flush_pending(force: bool = False) -> None:
        nonlocal pending_payloads, pending_count, train_count, val_count
        while pending_payloads and (pending_count >= int(chunk_size) or force):
            tensors: dict[str, Any] = {}
            for key in ("features", "targets", "seeds", "floors", "is_validation", "row_weights"):
                tensors[key] = torch.cat([payload[key] for payload in pending_payloads], dim=0)
            string_fields: dict[str, list[str]] = {
                key: [value for payload in pending_payloads for value in payload[key]]
                for key in ("phases", "sources", "sample_kinds")
            }
            take = min(int(chunk_size), int(tensors["features"].shape[0]))
            chunk_index = len(chunks)
            chunk_path = cache_dir / f"chunk_{chunk_index:05d}.pt"
            chunk_payload = {
                "features": tensors["features"][:take],
                "targets": tensors["targets"][:take],
                "seeds": tensors["seeds"][:take],
                "floors": tensors["floors"][:take],
                "is_validation": tensors["is_validation"][:take],
                "row_weights": tensors["row_weights"][:take],
                "phases": string_fields["phases"][:take],
                "sources": string_fields["sources"][:take],
                "sample_kinds": string_fields["sample_kinds"][:take],
            }
            torch.save(chunk_payload, chunk_path)
            is_validation = chunk_payload["is_validation"].bool()
            train_count += int((~is_validation).sum().item())
            val_count += int(is_validation.sum().item())
            chunks.append({"path": str(chunk_path), "count": take})
            remaining = int(tensors["features"].shape[0]) - take
            if remaining > 0:
                pending_payloads = [
                    {
                        "features": tensors["features"][take:],
                        "targets": tensors["targets"][take:],
                        "seeds": tensors["seeds"][take:],
                        "floors": tensors["floors"][take:],
                        "is_validation": tensors["is_validation"][take:],
                        "row_weights": tensors["row_weights"][take:],
                        "phases": string_fields["phases"][take:],
                        "sources": string_fields["sources"][take:],
                        "sample_kinds": string_fields["sample_kinds"][take:],
                    }
                ]
                pending_count = remaining
            else:
                pending_payloads = []
                pending_count = 0
            if not force and pending_count < int(chunk_size):
                break

    def append_result(result: dict[str, Any]) -> None:
        nonlocal count, skipped_error_seed_records
        source_files.append(str(result["path"]))
        skipped_error_seed_records += int(result.get("skipped_error_seed_records") or 0)
        for feature, target, seed, floor, phase, source, sample_kind, row_weight in zip(
            result["features"],
            result["targets"],
            result["seeds"],
            result["floors"],
            result["phases"],
            result["sources"],
            result["sample_kinds"],
            result["row_weights"],
        ):
            features.append(feature)
            targets.append(target)
            seeds.append(int(seed))
            floors.append(int(floor))
            phases.append(str(phase))
            sources.append(str(source))
            sample_kinds.append(str(sample_kind))
            row_weights.append(float(row_weight))
            count += 1
            if len(features) >= int(chunk_size):
                flush()
            if max_records > 0 and count >= max_records:
                break

    def append_direct_payload(meta: dict[str, Any]) -> None:
        nonlocal count, skipped_error_seed_records, pending_count
        source_files.append(str(meta["path"]))
        skipped_error_seed_records += int(meta.get("skipped_error_seed_records") or 0)
        payload = torch.load(Path(meta["temp_path"]), map_location="cpu", weights_only=False)
        pending_payloads.append(payload)
        pending_count += int(payload["features"].shape[0])
        count += int(payload["features"].shape[0])
        try:
            Path(meta["temp_path"]).unlink()
        except OSError:
            pass
        flush_pending(force=False)

    total_paths = len(paths)
    worker_config = {
        "sample_mode": str(sample_mode),
        "feature_variant": str(feature_variant),
        "record_weight_mode": str(record_weight_mode),
        "before_weight": float(before_weight),
        "after_weight": float(after_weight),
        "cache_feature_dtype": str(cache_feature_dtype),
    }
    if bool(cache_direct_write) and int(cache_workers) > 1 and int(max_records) <= 0:
        temp_dir = cache_dir / "_direct_parts"
        temp_dir.mkdir(parents=True, exist_ok=True)
        for stale in temp_dir.glob("part_*.pt"):
            stale.unlink()
        tasks = [
            (index, str(path), excluded_error_seeds, worker_config, str(temp_dir), int(val_mod), int(val_rem), str(cache_feature_dtype))
            for index, path in enumerate(paths)
        ]
        with ProcessPoolExecutor(max_workers=int(cache_workers)) as executor:
            for index, meta in enumerate(executor.map(_cache_decision_file_direct, tasks, chunksize=1), start=1):
                append_direct_payload(meta)
                if index == 1 or index == total_paths or index % 50 == 0:
                    print(
                        f"cache build files={index}/{total_paths} records={count} "
                        f"chunks={len(chunks)} skipped_error_seed_records={skipped_error_seed_records}",
                        flush=True,
                    )
        flush_pending(force=True)
        try:
            temp_dir.rmdir()
        except OSError:
            pass
    elif int(cache_workers) > 1 and int(max_records) <= 0:
        with ProcessPoolExecutor(max_workers=int(cache_workers)) as executor:
            for index, result in enumerate(
                executor.map(
                    _records_from_decision_file,
                    [str(path) for path in paths],
                    [excluded_error_seeds] * len(paths),
                    [worker_config] * len(paths),
                    chunksize=1,
                ),
                start=1,
            ):
                append_result(result)
                if index == 1 or index == total_paths or index % 50 == 0:
                    print(
                        f"cache build files={index}/{total_paths} records={count} "
                        f"chunks={len(chunks)} skipped_error_seed_records={skipped_error_seed_records}",
                        flush=True,
                    )
    else:
        for index, path in enumerate(paths, start=1):
            append_result(_records_from_decision_file(str(path), excluded_error_seeds, worker_config))
            if index == 1 or index == total_paths or index % 50 == 0:
                print(
                    f"cache build files={index}/{total_paths} records={count} "
                    f"chunks={len(chunks)} skipped_error_seed_records={skipped_error_seed_records}",
                    flush=True,
                )
            if max_records > 0 and count >= max_records:
                break
    flush()
    flush_pending(force=True)
    manifest = {
        "schema": "run_value_tensor_cache_v2",
        "input_dir": str(input_dir),
        "cache_dir": str(cache_dir),
        "source_files": source_files,
        "chunks": chunks,
        "count": count,
        "train_count": train_count,
        "validation_count": val_count,
        "state_feature_dim": STATE_FEATURE_DIM,
        "target_names": list(RUN_VALUE_OUTPUTS),
        "val_mod": int(val_mod),
        "val_rem": int(val_rem),
        "excluded_error_seeds": sorted(excluded_error_seeds),
        "skipped_error_seed_records": int(skipped_error_seed_records),
        "cache_workers": int(cache_workers),
        "sample_mode": str(sample_mode),
        "feature_variant": str(feature_variant),
        "record_weight_mode": str(record_weight_mode),
        "before_weight": float(before_weight),
        "after_weight": float(after_weight),
        "cache_feature_dtype": str(cache_feature_dtype),
        "cache_direct_write": bool(cache_direct_write),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_manifest(cache_dir: Path) -> dict[str, Any]:
    return json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))


def _train_seed_counts(chunks: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for chunk in chunks:
        payload = torch.load(Path(chunk["path"]), map_location="cpu", weights_only=False)
        mask = ~payload["is_validation"].bool()
        seeds = payload["seeds"].long()
        for seed in seeds[mask].tolist():
            counts[int(seed)] += 1
    return dict(counts)


def _seed_weight_scale(seed_counts: dict[int, int], mode: str) -> float:
    if not seed_counts or mode == "none":
        return 1.0
    total_rows = sum(max(0, int(count)) for count in seed_counts.values())
    if total_rows <= 0:
        return 1.0
    raw_sum = 0.0
    for count in seed_counts.values():
        count = max(1, int(count))
        if mode == "inverse_seed_count":
            raw = 1.0 / float(count)
        elif mode == "sqrt_inverse_seed_count":
            raw = 1.0 / math.sqrt(float(count))
        else:
            raw = 1.0
        raw_sum += raw * count
    mean_raw = raw_sum / max(1, total_rows)
    return 1.0 / max(1.0e-8, mean_raw)


def _seed_mode_from_row_weight_mode(mode: str) -> str:
    mode = str(mode)
    if mode == "precomputed":
        return "none"
    if mode.startswith("precomputed_"):
        return mode.removeprefix("precomputed_")
    return mode


def _row_weights_for_seeds(
    seeds: Any,
    *,
    seed_counts: dict[int, int] | None,
    mode: str,
    scale: float,
    min_weight: float,
    max_weight: float,
    device: str,
) -> Any | None:
    if mode == "none" or not seed_counts:
        return None
    weights: list[float] = []
    for seed in seeds.tolist():
        count = max(1, int(seed_counts.get(int(seed), 1)))
        if mode == "inverse_seed_count":
            weight = 1.0 / float(count)
        elif mode == "sqrt_inverse_seed_count":
            weight = 1.0 / math.sqrt(float(count))
        else:
            weight = 1.0
        weight *= float(scale)
        if float(min_weight) > 0.0:
            weight = max(float(min_weight), weight)
        if float(max_weight) > 0.0:
            weight = min(float(max_weight), weight)
        weights.append(float(weight))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _batch_row_weights(
    payload: dict[str, Any],
    batch_indexes: Any,
    batch_seeds: Any,
    *,
    train: bool,
    row_weight_mode: str,
    seed_counts: dict[int, int] | None,
    seed_weight_scale: float,
    min_row_weight: float,
    max_row_weight: float,
    device: str,
) -> Any | None:
    if not train:
        return None
    mode = str(row_weight_mode)
    weights = None
    if mode.startswith("precomputed") and "row_weights" in payload:
        weights = payload["row_weights"].index_select(0, batch_indexes).float().to(device)
    seed_mode = _seed_mode_from_row_weight_mode(mode)
    seed_weights = _row_weights_for_seeds(
        batch_seeds,
        seed_counts=seed_counts,
        mode=seed_mode,
        scale=float(seed_weight_scale),
        min_weight=0.0,
        max_weight=0.0,
        device=device,
    )
    if weights is None:
        weights = seed_weights
    elif seed_weights is not None:
        weights = weights * seed_weights
    if weights is None:
        return None
    if float(min_row_weight) > 0.0:
        weights = weights.clamp_min(float(min_row_weight))
    if float(max_row_weight) > 0.0:
        weights = weights.clamp_max(float(max_row_weight))
    return weights


def _iter_chunk_batches(
    chunks: list[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
    train: bool,
    rng: random.Random,
    row_weight_mode: str = "none",
    seed_counts: dict[int, int] | None = None,
    seed_weight_scale: float = 1.0,
    min_row_weight: float = 0.0,
    max_row_weight: float = 0.0,
):
    ordered = list(chunks)
    rng.shuffle(ordered)
    def payloads():
        if not ordered:
            return
        with ThreadPoolExecutor(max_workers=1) as executor:
            iterator = iter(ordered)
            first = next(iterator, None)
            if first is None:
                return
            future = executor.submit(torch.load, Path(first["path"]), map_location="cpu", weights_only=False)
            for chunk in iterator:
                payload = future.result()
                future = executor.submit(torch.load, Path(chunk["path"]), map_location="cpu", weights_only=False)
                yield payload
            yield future.result()

    for payload in payloads():
        mask = ~payload["is_validation"] if train else payload["is_validation"]
        indexes = torch.nonzero(mask, as_tuple=False).flatten()
        if int(indexes.numel()) == 0:
            continue
        order = indexes.tolist()
        rng.shuffle(order)
        for start in range(0, len(order), int(batch_size)):
            batch_indexes = torch.tensor(order[start : start + int(batch_size)], dtype=torch.long)
            batch_seeds = payload["seeds"].index_select(0, batch_indexes)
            yield {
                "features": payload["features"].index_select(0, batch_indexes).to(device).float(),
                "targets": payload["targets"].index_select(0, batch_indexes).to(device),
                "floors": payload["floors"].index_select(0, batch_indexes),
                "seeds": batch_seeds,
                "weights": _batch_row_weights(
                    payload,
                    batch_indexes,
                    batch_seeds,
                    train=train,
                    seed_counts=seed_counts if train else None,
                    row_weight_mode=str(row_weight_mode) if train else "none",
                    seed_weight_scale=float(seed_weight_scale),
                    min_row_weight=float(min_row_weight),
                    max_row_weight=float(max_row_weight),
                    device=device,
                ),
                "phases": [payload["phases"][int(index)] for index in batch_indexes.tolist()],
                "sources": [payload.get("sources", [""] * int(payload["targets"].shape[0]))[int(index)] for index in batch_indexes.tolist()],
                "sample_kinds": [payload.get("sample_kinds", ["before"] * int(payload["targets"].shape[0]))[int(index)] for index in batch_indexes.tolist()],
            }


def _fit_batch_features(features: Any, input_dim: int) -> Any:
    input_dim = int(input_dim or 0)
    if input_dim <= 0 or int(features.shape[1]) == input_dim:
        return features
    if int(features.shape[1]) > input_dim:
        return features[:, :input_dim]
    pad = torch.zeros(
        (int(features.shape[0]), input_dim - int(features.shape[1])),
        dtype=features.dtype,
        device=features.device,
    )
    return torch.cat([features, pad], dim=1)


def _model_features(model: RunValueNetwork, features: Any) -> Any:
    return _fit_batch_features(features, int(getattr(model, "input_dim", 0) or 0))


def _survival_thresholds(survival_bins: int, *, device: str) -> Any:
    bins = int(survival_bins)
    if bins == len(COARSE_SURVIVAL_THRESHOLDS):
        return torch.tensor(COARSE_SURVIVAL_THRESHOLDS, dtype=torch.float32, device=device)
    return torch.arange(1, bins + 1, dtype=torch.float32, device=device)


def _survival_final(outputs: Any, *, survival_bins: int) -> Any | None:
    bins = int(survival_bins)
    if bins <= 0:
        return None
    survival_logits = outputs[:, TARGET_DIM : TARGET_DIM + bins].float()
    if bins == len(COARSE_SURVIVAL_THRESHOLDS):
        # Coarse ordinal bins are auxiliary-only. Approximate expectation from
        # interval widths when explicitly requested for diagnostics.
        thresholds = list(COARSE_SURVIVAL_THRESHOLDS)
        widths = [thresholds[0] - 1]
        widths.extend(max(1, thresholds[index] - thresholds[index - 1]) for index in range(1, len(thresholds)))
        weights = torch.tensor(widths, dtype=torch.float32, device=outputs.device).view(1, bins)
        return (torch.sigmoid(survival_logits) * weights).sum(dim=1)
    return torch.sigmoid(survival_logits).sum(dim=1)


def _categorical_floor_values(final_floor_bins: int, *, device: str) -> Any:
    bins = int(final_floor_bins)
    if bins == len(COARSE_FINAL_FLOOR_BINS):
        return torch.tensor([center for _, _, center in COARSE_FINAL_FLOOR_BINS], dtype=torch.float32, device=device).view(1, bins)
    return torch.arange(1, bins + 1, dtype=torch.float32, device=device).view(1, bins)


def _final_floor_class_labels(final_floors: Any, *, final_floor_bins: int, device: str) -> Any:
    bins = int(final_floor_bins)
    final_floors = final_floors.long().to(device)
    if bins == len(COARSE_FINAL_FLOOR_BINS):
        labels = torch.zeros_like(final_floors)
        for index, (lo, hi, _) in enumerate(COARSE_FINAL_FLOOR_BINS):
            labels = torch.where((final_floors >= int(lo)) & (final_floors <= int(hi)), torch.full_like(labels, index), labels)
        return labels.clamp(0, bins - 1)
    return (final_floors - 1).clamp(0, bins - 1)


def _categorical_final(outputs: Any, *, survival_bins: int, final_floor_bins: int, readout: str) -> Any | None:
    bins = int(final_floor_bins)
    if bins <= 0 or str(readout) == "none":
        return None
    start = TARGET_DIM + int(survival_bins)
    logits = outputs[:, start : start + bins].float()
    if logits.shape[1] < bins:
        return None
    if str(readout) == "mode":
        values = _categorical_floor_values(bins, device=outputs.device).view(-1)
        return values.index_select(0, torch.argmax(logits, dim=1))
    probs = torch.softmax(logits, dim=1)
    floors = _categorical_floor_values(bins, device=outputs.device)
    if str(readout) == "median":
        cdf = torch.cumsum(probs, dim=1)
        values = floors.view(-1)
        return values.index_select(0, torch.argmax((cdf >= 0.5).float(), dim=1))
    return (probs * floors).sum(dim=1)


def _pred_remaining(
    outputs: Any,
    floors: Any,
    *,
    survival_bins: int,
    use_survival: bool,
    final_floor_bins: int = 0,
    final_floor_readout: str = "none",
) -> Any:
    categorical_final = _categorical_final(
        outputs,
        survival_bins=int(survival_bins),
        final_floor_bins=int(final_floor_bins),
        readout=str(final_floor_readout),
    )
    if categorical_final is not None:
        return categorical_final - floors.float().to(outputs.device)
    survival_final = _survival_final(outputs, survival_bins=survival_bins)
    if bool(use_survival) and survival_final is not None:
        return survival_final - floors.float().to(outputs.device)
    return outputs[:, 0].float()


def _clamp_remaining(pred_remaining: Any, floors: Any) -> Any:
    """Clamp to the run's physically possible remaining floor range."""
    floors = floors.float().to(pred_remaining.device)
    return pred_remaining.float().clamp_min(0.0).minimum((50.0 - floors).clamp_min(0.0))


def _loss(
    outputs: Any,
    targets: Any,
    *,
    floors: Any,
    weights: Any | None = None,
    regression_loss: str = "smooth_l1",
    smooth_l1_beta: float = 1.0,
    survival_bins: int = 0,
    survival_weight: float = 0.0,
    survival_value_weight: float = 0.0,
    final_floor_bins: int = 0,
    final_floor_weight: float = 0.0,
    final_floor_value_weight: float = 0.0,
    final_loss_weight: float = 0.25,
    act_bce_weight: float = 1.0 / 3.0,
    death_bce_weight: float = 1.0 / 6.0,
) -> tuple[Any, dict[str, float]]:
    base_outputs = outputs[:, :TARGET_DIM].float()
    row_weights = weights.float().to(outputs.device) if weights is not None else None

    def weighted_mean(values: Any) -> Any:
        values = values.float()
        if row_weights is None:
            return values.mean()
        return (values * row_weights).sum() / row_weights.sum().clamp_min(1.0e-8)

    def regression_rows(pred: Any, true: Any) -> Any:
        loss_name = str(regression_loss or "smooth_l1")
        pred = pred.float()
        true = true.float()
        if loss_name == "l1":
            return torch.abs(pred - true)
        if loss_name == "mse":
            return F.mse_loss(pred, true, reduction="none")
        if loss_name == "log_cosh":
            diff = pred - true
            return torch.log(torch.cosh(diff.clamp(-20.0, 20.0)))
        try:
            return F.smooth_l1_loss(pred, true, reduction="none", beta=float(smooth_l1_beta))
        except TypeError:
            return F.smooth_l1_loss(pred, true, reduction="none")

    remaining_loss = weighted_mean(regression_rows(base_outputs[:, 0], targets[:, 0]))
    final_loss = weighted_mean(regression_rows(base_outputs[:, 1], targets[:, 1]))
    act_bce_rows = F.binary_cross_entropy_with_logits(base_outputs[:, 2:6], targets[:, 2:6].float(), reduction="none").mean(dim=1)
    death_bce_rows = F.binary_cross_entropy_with_logits(base_outputs[:, 6:8], targets[:, 6:8].float(), reduction="none").mean(dim=1)
    act_bce_loss = weighted_mean(act_bce_rows)
    death_bce_loss = weighted_mean(death_bce_rows)
    bce_loss = 0.5 * (act_bce_loss + death_bce_loss)
    total = (
        remaining_loss
        + float(final_loss_weight) * final_loss
        + float(act_bce_weight) * act_bce_loss
        + float(death_bce_weight) * death_bce_loss
    )
    metrics = {
        "loss": float(total.detach().cpu().item()),
        "remaining_loss": float(remaining_loss.detach().cpu().item()),
        "final_loss": float(final_loss.detach().cpu().item()),
        "bce_loss": float(bce_loss.detach().cpu().item()),
        "act_bce_loss": float(act_bce_loss.detach().cpu().item()),
        "death_bce_loss": float(death_bce_loss.detach().cpu().item()),
    }
    bins = int(survival_bins)
    if bins > 0:
        thresholds = _survival_thresholds(bins, device=outputs.device).view(1, bins)
        labels = (targets[:, 1].float().to(outputs.device).view(-1, 1) >= thresholds).float()
        survival_logits = outputs[:, TARGET_DIM : TARGET_DIM + bins].float()
        survival_rows = F.binary_cross_entropy_with_logits(survival_logits, labels, reduction="none").mean(dim=1)
        survival_loss = weighted_mean(survival_rows)
        survival_final = torch.sigmoid(survival_logits).sum(dim=1)
        survival_remaining = survival_final - floors.float().to(outputs.device)
        survival_value_loss = weighted_mean(regression_rows(survival_remaining, targets[:, 0]))
        total = total + float(survival_weight) * survival_loss + float(survival_value_weight) * survival_value_loss
        metrics.update(
            {
                "loss": float(total.detach().cpu().item()),
                "survival_loss": float(survival_loss.detach().cpu().item()),
                "survival_value_loss": float(survival_value_loss.detach().cpu().item()),
            }
        )
    final_bins = int(final_floor_bins)
    if final_bins > 0:
        start = TARGET_DIM + int(survival_bins)
        logits = outputs[:, start : start + final_bins].float()
        labels = _final_floor_class_labels(targets[:, 1], final_floor_bins=final_bins, device=outputs.device)
        ce_rows = F.cross_entropy(logits, labels, reduction="none")
        final_floor_loss = weighted_mean(ce_rows)
        probs = torch.softmax(logits, dim=1)
        floor_values = _categorical_floor_values(final_bins, device=outputs.device)
        categorical_final = (probs * floor_values).sum(dim=1)
        categorical_remaining = categorical_final - floors.float().to(outputs.device)
        categorical_value_loss = weighted_mean(regression_rows(categorical_remaining, targets[:, 0]))
        total = total + float(final_floor_weight) * final_floor_loss + float(final_floor_value_weight) * categorical_value_loss
        metrics.update(
            {
                "loss": float(total.detach().cpu().item()),
                "final_floor_ce_loss": float(final_floor_loss.detach().cpu().item()),
                "final_floor_value_loss": float(categorical_value_loss.detach().cpu().item()),
            }
        )
    return total, metrics


def _eval_model(
    model: RunValueNetwork,
    chunks: list[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
    survival_bins: int = 0,
    use_survival: bool = False,
    survival_weight: float = 0.0,
    survival_value_weight: float = 0.0,
    final_floor_bins: int = 0,
    final_floor_readout: str = "none",
    final_floor_weight: float = 0.0,
    final_floor_value_weight: float = 0.0,
    residual_floor_baseline: dict[str, Any] | None = None,
    regression_loss: str = "smooth_l1",
    smooth_l1_beta: float = 1.0,
    final_loss_weight: float = 0.25,
    act_bce_weight: float = 1.0 / 3.0,
    death_bce_weight: float = 1.0 / 6.0,
    value_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    rng = random.Random(12345)
    total = 0
    abs_remaining = 0.0
    abs_final = 0.0
    weighted: dict[str, float] = defaultdict(float)
    floor_buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    phase_buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    sample_kind_buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    source_buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with torch.inference_mode():
        for batch in _iter_chunk_batches(chunks, batch_size=batch_size, device=device, train=False, rng=rng):
            features = _model_features(model, batch["features"])
            outputs = model(features)
            loss_targets = _targets_for_loss(
                batch["targets"],
                batch["floors"],
                residual_floor_baseline=residual_floor_baseline,
                device=device,
                phases=batch.get("phases"),
                sources=batch.get("sources"),
                sample_kinds=batch.get("sample_kinds"),
            )
            loss, metrics = _loss(
                outputs,
                loss_targets,
                floors=batch["floors"],
                weights=batch.get("weights"),
                survival_bins=int(survival_bins),
                regression_loss=str(regression_loss),
                smooth_l1_beta=float(smooth_l1_beta),
                survival_weight=float(survival_weight),
                survival_value_weight=float(survival_value_weight),
                final_floor_bins=int(final_floor_bins),
                final_floor_weight=float(final_floor_weight),
                final_floor_value_weight=float(final_floor_value_weight),
                final_loss_weight=float(final_loss_weight),
                act_bce_weight=float(act_bce_weight),
                death_bce_weight=float(death_bce_weight),
            )
            n = int(batch["targets"].shape[0])
            total += n
            for key, value in metrics.items():
                weighted[key] += float(value) * n
            pred_remaining = _pred_remaining(
                outputs,
                batch["floors"],
                survival_bins=int(survival_bins),
                use_survival=bool(use_survival),
                final_floor_bins=int(final_floor_bins),
                final_floor_readout=str(final_floor_readout),
            ).detach().float().cpu()
            uses_absolute_final_readout = str(final_floor_readout) != "none" and int(final_floor_bins) > 0
            uses_absolute_survival_readout = bool(use_survival) and int(survival_bins) > 0
            if residual_floor_baseline and not uses_absolute_final_readout and not uses_absolute_survival_readout:
                pred_remaining += _residual_baseline_for_batch(
                    batch["floors"],
                    residual_floor_baseline=residual_floor_baseline,
                    device="cpu",
                    phases=batch.get("phases"),
                    sources=batch.get("sources"),
                    sample_kinds=batch.get("sample_kinds"),
                )
            if isinstance(value_calibration, dict):
                pred_remaining += _calibration_bias_for_batch(batch, value_calibration, device="cpu")
            pred_remaining = _clamp_remaining(pred_remaining, batch["floors"].detach().cpu())
            true_remaining = batch["targets"][:, 0].detach().float().cpu()
            categorical_final = _categorical_final(
                outputs,
                survival_bins=int(survival_bins),
                final_floor_bins=int(final_floor_bins),
                readout=str(final_floor_readout),
            )
            survival_final = _survival_final(outputs, survival_bins=int(survival_bins))
            pred_final = (
                categorical_final.detach().float().cpu()
                if categorical_final is not None
                else
                survival_final.detach().float().cpu()
                if bool(use_survival) and survival_final is not None
                else outputs[:, 1].detach().float().cpu()
            )
            true_final = batch["targets"][:, 1].detach().float().cpu()
            abs_remaining += float(torch.abs(pred_remaining - true_remaining).sum().item())
            abs_final += float(torch.abs(pred_final - true_final).sum().item())
            for index, floor in enumerate(batch["floors"].tolist()):
                bucket = f"{int(floor) // 5 * 5:02d}-{int(floor) // 5 * 5 + 4:02d}"
                floor_buckets[bucket].append((float(pred_remaining[index]), float(true_remaining[index])))
            for index, phase in enumerate(batch["phases"]):
                phase_buckets[str(phase)].append((float(pred_remaining[index]), float(true_remaining[index])))
            for index, sample_kind in enumerate(batch.get("sample_kinds") or []):
                sample_kind_buckets[str(sample_kind)].append((float(pred_remaining[index]), float(true_remaining[index])))
            for index, source in enumerate(batch.get("sources") or []):
                source_buckets[str(source)].append((float(pred_remaining[index]), float(true_remaining[index])))
    metrics = {key: value / max(1, total) for key, value in weighted.items()}
    metrics["remaining_mae"] = abs_remaining / max(1, total)
    metrics["final_mae"] = abs_final / max(1, total)
    metrics["count"] = total
    metrics["floor_calibration"] = {
        key: {
            "count": len(values),
            "pred_mean": mean([value[0] for value in values]),
            "true_mean": mean([value[1] for value in values]),
        }
        for key, values in sorted(floor_buckets.items())
        if values
    }
    metrics["phase_calibration"] = {
        key: {
            "count": len(values),
            "pred_mean": mean([value[0] for value in values]),
            "true_mean": mean([value[1] for value in values]),
        }
        for key, values in sorted(phase_buckets.items())
        if values
    }
    metrics["sample_kind_calibration"] = {
        key: {
            "count": len(values),
            "pred_mean": mean([value[0] for value in values]),
            "true_mean": mean([value[1] for value in values]),
            "mae": mean([abs(value[0] - value[1]) for value in values]),
        }
        for key, values in sorted(sample_kind_buckets.items())
        if values
    }
    metrics["source_calibration"] = {
        key: {
            "count": len(values),
            "pred_mean": mean([value[0] for value in values]),
            "true_mean": mean([value[1] for value in values]),
            "mae": mean([abs(value[0] - value[1]) for value in values]),
        }
        for key, values in sorted(source_buckets.items())
        if values
    }
    return metrics


def _calibration_value_for_batch(batch: dict[str, Any], index: int, field: str) -> str:
    field = str(field)
    if field == "floor":
        return str(int(batch["floors"][index].item()))
    if field == "floor_bucket":
        floor = int(batch["floors"][index].item())
        start = floor // 5 * 5
        return f"{start:02d}-{start + 4:02d}"
    if field == "phase":
        return str(batch.get("phases", [""])[index])
    if field == "source":
        return str(batch.get("sources", [""])[index])
    if field == "sample_kind":
        return str(batch.get("sample_kinds", ["before"])[index])
    return ""


def _calibration_bias_for_batch(batch: dict[str, Any], calibration: dict[str, Any], *, device: str) -> Any:
    fields = [str(field) for field in calibration.get("fields") or []]
    parent_fields = [str(field) for field in calibration.get("parent_fields") or []]
    group_bias = calibration.get("group_bias") or {}
    parent_bias = calibration.get("parent_bias") or {}
    values: list[float] = []
    count = int(batch["floors"].shape[0])
    for index in range(count):
        group_key = "|".join(_calibration_value_for_batch(batch, index, field) for field in fields)
        parent_key = "|".join(_calibration_value_for_batch(batch, index, field) for field in parent_fields)
        values.append(float(group_bias.get(group_key, parent_bias.get(parent_key, 0.0))))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _baseline_group_key(
    floor: int,
    *,
    phase: str = "",
    source: str = "",
    sample_kind: str = "",
    mode: str = "floor",
) -> str:
    floor_key = str(int(floor))
    mode = str(mode or "floor")
    if mode == "floor":
        return floor_key
    if mode == "floor_phase":
        return f"{floor_key}|{phase}"
    if mode == "floor_phase_source":
        return f"{floor_key}|{phase}|{source}"
    if mode == "floor_phase_source_kind":
        return f"{floor_key}|{phase}|{source}|{sample_kind}"
    return floor_key


def _baseline_mae(
    chunks: list[dict[str, Any]],
    *,
    key_mode: str = "floor",
    shrink_count: float = 0.0,
) -> dict[str, Any]:
    # Baseline: train-set mean remaining floor per selected group, fallback floor/global.
    key_mode = str(key_mode or "floor")
    group_sums: dict[str, float] = defaultdict(float)
    group_counts: dict[str, int] = defaultdict(int)
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    global_sum = 0.0
    global_count = 0
    val_rows: list[tuple[int, str, str, str, float]] = []
    for chunk in chunks:
        payload = torch.load(Path(chunk["path"]), map_location="cpu", weights_only=False)
        targets = payload["targets"][:, 0].float()
        floors = payload["floors"].long()
        mask = payload["is_validation"].bool()
        phases = list(payload.get("phases") or [""] * int(targets.shape[0]))
        sources = list(payload.get("sources") or [""] * int(targets.shape[0]))
        sample_kinds = list(payload.get("sample_kinds") or [""] * int(targets.shape[0]))
        for floor, phase, source, sample_kind, target, is_val in zip(
            floors.tolist(),
            phases,
            sources,
            sample_kinds,
            targets.tolist(),
            mask.tolist(),
        ):
            floor = int(floor)
            key = _baseline_group_key(
                floor,
                phase=str(phase),
                source=str(source),
                sample_kind=str(sample_kind),
                mode=key_mode,
            )
            if is_val:
                val_rows.append((floor, str(phase), str(source), str(sample_kind), float(target)))
            else:
                group_sums[key] += float(target)
                group_counts[key] += 1
                sums[floor] += float(target)
                counts[floor] += 1
                global_sum += float(target)
                global_count += 1
    global_mean = global_sum / max(1, global_count)
    floor_means = {str(floor): sums[floor] / counts[floor] for floor in sorted(counts)}

    def group_mean(key: str) -> float:
        raw_count = max(0, int(group_counts.get(key, 0)))
        if raw_count <= 0:
            floor_key = key.split("|", 1)[0]
            return float(floor_means.get(floor_key, global_mean))
        raw_mean = group_sums[key] / raw_count
        shrink = max(0.0, float(shrink_count))
        if shrink <= 0.0:
            return float(raw_mean)
        floor_key = key.split("|", 1)[0]
        parent = float(floor_means.get(floor_key, global_mean))
        return float((group_sums[key] + shrink * parent) / (raw_count + shrink))

    group_means = {key: group_mean(key) for key in sorted(group_counts)}
    abs_err = 0.0
    for floor, phase, source, sample_kind, target in val_rows:
        key = _baseline_group_key(
            floor,
            phase=phase,
            source=source,
            sample_kind=sample_kind,
            mode=key_mode,
        )
        if group_counts.get(key, 0):
            pred = group_means[key]
        elif counts.get(floor, 0):
            pred = sums[floor] / counts[floor]
        else:
            pred = global_mean
        abs_err += abs(pred - target)
    return {
        "baseline": f"train_mean_remaining_by_{key_mode}",
        "key_mode": key_mode,
        "shrink_count": float(shrink_count),
        "remaining_mae": abs_err / max(1, len(val_rows)),
        "validation_count": len(val_rows),
        "global_mean_remaining": global_mean,
        "floor_means": floor_means,
        "group_means": group_means,
    }


def _residual_baseline_for_batch(
    floors: Any,
    *,
    residual_floor_baseline: dict[str, Any] | None,
    device: str,
    phases: list[str] | None = None,
    sources: list[str] | None = None,
    sample_kinds: list[str] | None = None,
) -> Any:
    baseline = residual_floor_baseline or {}
    key_mode = str(baseline.get("key_mode") or "floor")
    group_table = baseline.get("group_means") or {}
    floor_table = baseline.get("floor_means") or {}
    global_mean = float((residual_floor_baseline or {}).get("global_mean_remaining") or 0.0)
    phase_values = phases or [""] * len(floors)
    source_values = sources or [""] * len(floors)
    sample_kind_values = sample_kinds or [""] * len(floors)
    values: list[float] = []
    for floor, phase, source, sample_kind in zip(floors.tolist(), phase_values, source_values, sample_kind_values):
        floor_key = str(int(floor))
        group_key = _baseline_group_key(
            int(floor),
            phase=str(phase),
            source=str(source),
            sample_kind=str(sample_kind),
            mode=key_mode,
        )
        values.append(float(group_table.get(group_key, floor_table.get(floor_key, global_mean))))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _targets_for_loss(
    targets: Any,
    floors: Any,
    *,
    residual_floor_baseline: dict[str, Any] | None,
    device: str,
    phases: list[str] | None = None,
    sources: list[str] | None = None,
    sample_kinds: list[str] | None = None,
) -> Any:
    if not residual_floor_baseline:
        return targets
    adjusted = targets.clone()
    adjusted[:, 0] = adjusted[:, 0].float() - _residual_baseline_for_batch(
        floors,
        residual_floor_baseline=residual_floor_baseline,
        device=device,
        phases=phases,
        sources=sources,
        sample_kinds=sample_kinds,
    )
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Train run-level value model from run-value decision logs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--cache-workers", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=0)
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
        default="current",
    )
    parser.add_argument("--record-weight-mode", choices=["none", "balanced_v2"], default="none")
    parser.add_argument("--before-weight", type=float, default=0.4)
    parser.add_argument("--after-weight", type=float, default=0.6)
    parser.add_argument("--cache-feature-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--cache-direct-write",
        action="store_true",
        help="Let cache workers write tensor parts directly, avoiding large Python-list IPC during cache build.",
    )
    parser.add_argument("--val-mod", type=int, default=10)
    parser.add_argument("--val-rem", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--model-input-dim",
        type=int,
        default=0,
        help="Optional model input width. Wider cache features are sliced and narrower cache features are zero-padded.",
    )
    parser.add_argument("--architecture", choices=["mlp", "late_fusion", "res_mlp"], default="mlp")
    parser.add_argument("--group-dim", type=int, default=64)
    parser.add_argument("--survival-bins", type=int, default=0)
    parser.add_argument("--survival-weight", type=float, default=0.0)
    parser.add_argument("--survival-value-weight", type=float, default=0.0)
    parser.add_argument("--use-survival-for-mae", action="store_true")
    parser.add_argument("--final-floor-bins", type=int, default=0)
    parser.add_argument("--final-floor-weight", type=float, default=0.0)
    parser.add_argument("--final-floor-value-weight", type=float, default=0.0)
    parser.add_argument("--final-floor-readout", choices=["none", "expected", "median", "mode"], default="none")
    parser.add_argument(
        "--row-weight-mode",
        choices=[
            "none",
            "sqrt_inverse_seed_count",
            "inverse_seed_count",
            "precomputed",
            "precomputed_sqrt_inverse_seed_count",
            "precomputed_inverse_seed_count",
        ],
        default="none",
        help="Training-only row weights. Inverse modes reduce long-run dominance by balancing each seed/run.",
    )
    parser.add_argument(
        "--residual-floor-baseline",
        action="store_true",
        help="Train output[0] as remaining_floor minus the train-set mean remaining for the current floor.",
    )
    parser.add_argument(
        "--residual-baseline-key",
        choices=["floor", "floor_phase", "floor_phase_source", "floor_phase_source_kind"],
        default="floor",
        help="Grouping key for the residual baseline. Runtime use is safest with floor or floor_phase.",
    )
    parser.add_argument(
        "--residual-baseline-shrink-count",
        type=float,
        default=0.0,
        help="Shrink grouped residual baseline means toward the per-floor mean by this pseudo-count.",
    )
    parser.add_argument("--final-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--regression-loss",
        choices=["smooth_l1", "l1", "mse", "log_cosh"],
        default="smooth_l1",
        help="Loss used for scalar remaining/final value regression.",
    )
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--act-bce-weight", type=float, default=1.0 / 3.0)
    parser.add_argument("--death-bce-weight", type=float, default=1.0 / 6.0)
    parser.add_argument("--min-row-weight", type=float, default=0.0)
    parser.add_argument("--max-row-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--save-each-epoch",
        action="store_true",
        help="Write an epoch_XXX checkpoint beside --output after each validation pass.",
    )
    args = parser.parse_args()

    require_torch()
    if args.rebuild_cache or not (args.cache_dir / "manifest.json").exists():
        manifest = build_cache(
            input_dir=args.input_dir,
            cache_dir=args.cache_dir,
            chunk_size=int(args.chunk_size),
            val_mod=int(args.val_mod),
            val_rem=int(args.val_rem),
            max_records=int(args.max_records),
            cache_workers=int(args.cache_workers),
            sample_mode=str(args.sample_mode),
            feature_variant=str(args.feature_variant),
            record_weight_mode=str(args.record_weight_mode),
            before_weight=float(args.before_weight),
            after_weight=float(args.after_weight),
            cache_feature_dtype=str(args.cache_feature_dtype),
            cache_direct_write=bool(args.cache_direct_write),
        )
    else:
        manifest = load_manifest(args.cache_dir)
        cached_dim = int(manifest.get("state_feature_dim") or 0)
        if cached_dim <= 0:
            raise SystemExit(f"cache feature dim missing or invalid: cache={cached_dim}")
        expected_cache_config = {
            "sample_mode": str(args.sample_mode),
            "feature_variant": str(args.feature_variant),
            "record_weight_mode": str(args.record_weight_mode),
            "before_weight": float(args.before_weight),
            "after_weight": float(args.after_weight),
            "cache_feature_dtype": str(args.cache_feature_dtype),
            "cache_direct_write": bool(args.cache_direct_write),
        }
        cached_cache_config = {
            "sample_mode": str(manifest.get("sample_mode") or "before"),
            "feature_variant": str(manifest.get("feature_variant") or "current"),
            "record_weight_mode": str(manifest.get("record_weight_mode") or "none"),
            "before_weight": float(manifest.get("before_weight") if manifest.get("before_weight") is not None else 0.4),
            "after_weight": float(manifest.get("after_weight") if manifest.get("after_weight") is not None else 0.6),
            "cache_feature_dtype": str(manifest.get("cache_feature_dtype") or "float32"),
            "cache_direct_write": bool(manifest.get("cache_direct_write", False)),
        }
        if cached_cache_config != expected_cache_config:
            raise SystemExit(
                f"cache config mismatch: cache={cached_cache_config} expected={expected_cache_config}; "
                "rerun with --rebuild-cache or use a matching cache dir"
            )
    chunks = list(manifest.get("chunks") or [])
    if not chunks:
        raise SystemExit("run value cache has no chunks")
    if int(manifest.get("validation_count") or 0) <= 0:
        raise SystemExit("run value cache has no validation rows")
    cache_input_dim = int(manifest.get("state_feature_dim") or STATE_FEATURE_DIM)
    model_input_dim = int(args.model_input_dim or cache_input_dim)
    if model_input_dim <= 0:
        raise SystemExit(f"invalid model input dim: {model_input_dim}")
    if model_input_dim != cache_input_dim:
        print(f"using model_input_dim={model_input_dim} with cache_feature_dim={cache_input_dim}", flush=True)
    model = RunValueNetwork(
        input_dim=model_input_dim,
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
        survival_bins=int(args.survival_bins),
        final_floor_bins=int(args.final_floor_bins),
        architecture=str(args.architecture),
        group_dim=int(args.group_dim),
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)), eta_min=float(args.min_learning_rate))
    rng = random.Random(int(args.seed))
    best = None
    best_epoch = 0
    best_state = None
    train_history = []
    baseline = _baseline_mae(
        chunks,
        key_mode=str(args.residual_baseline_key),
        shrink_count=float(args.residual_baseline_shrink_count),
    )
    residual_floor_baseline = baseline if bool(args.residual_floor_baseline) else None
    seed_counts: dict[int, int] | None = None
    seed_weight_scale = 1.0
    seed_weight_mode = _seed_mode_from_row_weight_mode(str(args.row_weight_mode))
    if seed_weight_mode != "none":
        seed_counts = _train_seed_counts(chunks)
        seed_weight_scale = _seed_weight_scale(seed_counts, seed_weight_mode)
    if seed_counts:
        counts = list(seed_counts.values())
        print(
            f"loaded run-value cache rows={manifest['count']} train={manifest['train_count']} "
            f"val={manifest['validation_count']} baseline_mae={baseline['remaining_mae']:.3f} "
            f"row_weight_mode={args.row_weight_mode} seeds={len(counts)} "
            f"seed_rows_min={min(counts)} seed_rows_mean={mean(counts):.1f} seed_rows_max={max(counts)} "
            f"weight_scale={seed_weight_scale:.4f}",
            flush=True,
        )
    else:
        print(f"loaded run-value cache rows={manifest['count']} train={manifest['train_count']} val={manifest['validation_count']} baseline_mae={baseline['remaining_mae']:.3f}", flush=True)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        weighted: dict[str, float] = defaultdict(float)
        seen = 0
        batches = 0
        for batch in _iter_chunk_batches(
            chunks,
            batch_size=int(args.batch_size),
            device=str(args.device),
            train=True,
            rng=rng,
            row_weight_mode=str(args.row_weight_mode),
            seed_counts=seed_counts,
            seed_weight_scale=float(seed_weight_scale),
            min_row_weight=float(args.min_row_weight),
            max_row_weight=float(args.max_row_weight),
        ):
            features = _model_features(model, batch["features"])
            outputs = model(features)
            loss_targets = _targets_for_loss(
                batch["targets"],
                batch["floors"],
                residual_floor_baseline=residual_floor_baseline,
                device=str(args.device),
                phases=batch.get("phases"),
                sources=batch.get("sources"),
                sample_kinds=batch.get("sample_kinds"),
            )
            loss, metrics = _loss(
                outputs,
                loss_targets,
                floors=batch["floors"],
                weights=batch.get("weights"),
                regression_loss=str(args.regression_loss),
                smooth_l1_beta=float(args.smooth_l1_beta),
                survival_bins=int(args.survival_bins),
                survival_weight=float(args.survival_weight),
                survival_value_weight=float(args.survival_value_weight),
                final_floor_bins=int(args.final_floor_bins),
                final_floor_weight=float(args.final_floor_weight),
                final_floor_value_weight=float(args.final_floor_value_weight),
                final_loss_weight=float(args.final_loss_weight),
                act_bce_weight=float(args.act_bce_weight),
                death_bce_weight=float(args.death_bce_weight),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = int(batch["targets"].shape[0])
            seen += n
            batches += 1
            for key, value in metrics.items():
                weighted[key] += float(value) * n
            if int(args.progress_interval) > 0 and batches % int(args.progress_interval) == 0:
                print(f"epoch {epoch:03d} batch={batches} rows={seen} loss={metrics['loss']:.4f}", flush=True)
        scheduler.step()
        train_metrics = {key: value / max(1, seen) for key, value in weighted.items()}
        valid_metrics = _eval_model(
            model,
            chunks,
            batch_size=int(args.batch_size),
            device=str(args.device),
            survival_bins=int(args.survival_bins),
            use_survival=bool(args.use_survival_for_mae),
            regression_loss=str(args.regression_loss),
            smooth_l1_beta=float(args.smooth_l1_beta),
            survival_weight=float(args.survival_weight),
            survival_value_weight=float(args.survival_value_weight),
            final_floor_bins=int(args.final_floor_bins),
            final_floor_readout=str(args.final_floor_readout),
            final_floor_weight=float(args.final_floor_weight),
            final_floor_value_weight=float(args.final_floor_value_weight),
            residual_floor_baseline=residual_floor_baseline,
            final_loss_weight=float(args.final_loss_weight),
            act_bce_weight=float(args.act_bce_weight),
            death_bce_weight=float(args.death_bce_weight),
        )
        train_history.append({"epoch": epoch, "train": train_metrics, "validation": valid_metrics})
        current = float(valid_metrics["remaining_mae"])
        if best is None or current < best:
            best = current
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if bool(args.save_each_epoch):
            epoch_metadata = {
                "manifest": manifest,
                "baseline": baseline,
                "current_epoch": int(epoch),
                "model_input_dim": int(model_input_dim),
                "cache_input_dim": int(cache_input_dim),
                "best_epoch": int(best_epoch),
                "best_validation_remaining_mae": float(best or 0.0),
                "survival_bins": int(args.survival_bins),
                "use_survival_for_mae": bool(args.use_survival_for_mae),
                "final_floor_bins": int(args.final_floor_bins),
                "final_floor_readout": str(args.final_floor_readout),
                "final_floor_weight": float(args.final_floor_weight),
                "final_floor_value_weight": float(args.final_floor_value_weight),
                "final_loss_weight": float(args.final_loss_weight),
                "regression_loss": str(args.regression_loss),
                "smooth_l1_beta": float(args.smooth_l1_beta),
                "act_bce_weight": float(args.act_bce_weight),
                "death_bce_weight": float(args.death_bce_weight),
                "row_weight_mode": str(args.row_weight_mode),
                "residual_floor_baseline": residual_floor_baseline,
                "residual_baseline_key": str(args.residual_baseline_key),
                "residual_baseline_shrink_count": float(args.residual_baseline_shrink_count),
                "min_row_weight": float(args.min_row_weight),
                "max_row_weight": float(args.max_row_weight),
                "seed_weight_scale": float(seed_weight_scale),
                "history": list(train_history),
            }
            epoch_path = args.output.parent / f"{args.output.stem}.epoch_{epoch:03d}{args.output.suffix}"
            save_run_value_checkpoint(epoch_path, model, metadata=epoch_metadata)
        print(
            f"epoch {epoch:03d} train_loss={train_metrics.get('loss', 0.0):.4f} "
            f"valid_mae={valid_metrics['remaining_mae']:.3f} final_mae={valid_metrics['final_mae']:.3f} best={best:.3f}@{best_epoch}",
            flush=True,
        )
        if int(args.early_stop_patience) > 0:
            stale_epochs = epoch - int(best_epoch)
            if stale_epochs >= int(args.early_stop_patience) and current > float(best or 0.0) - float(args.early_stop_min_delta):
                print(
                    f"early stop at epoch {epoch:03d}: best_mae={best:.3f}@{best_epoch}, stale_epochs={stale_epochs}",
                    flush=True,
                )
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "manifest": manifest,
        "baseline": baseline,
        "model_input_dim": int(model_input_dim),
        "cache_input_dim": int(cache_input_dim),
        "best_epoch": int(best_epoch),
        "best_validation_remaining_mae": float(best or 0.0),
        "survival_bins": int(args.survival_bins),
        "use_survival_for_mae": bool(args.use_survival_for_mae),
        "final_floor_bins": int(args.final_floor_bins),
        "final_floor_readout": str(args.final_floor_readout),
        "final_floor_weight": float(args.final_floor_weight),
        "final_floor_value_weight": float(args.final_floor_value_weight),
        "final_loss_weight": float(args.final_loss_weight),
        "regression_loss": str(args.regression_loss),
        "smooth_l1_beta": float(args.smooth_l1_beta),
        "act_bce_weight": float(args.act_bce_weight),
        "death_bce_weight": float(args.death_bce_weight),
        "row_weight_mode": str(args.row_weight_mode),
        "residual_floor_baseline": residual_floor_baseline,
        "residual_baseline_key": str(args.residual_baseline_key),
        "residual_baseline_shrink_count": float(args.residual_baseline_shrink_count),
        "min_row_weight": float(args.min_row_weight),
        "max_row_weight": float(args.max_row_weight),
        "seed_weight_scale": float(seed_weight_scale),
        "history": train_history,
    }
    save_run_value_checkpoint(args.output, model, metadata=metadata)
    final_valid = _eval_model(
        model,
        chunks,
        batch_size=int(args.batch_size),
        device=str(args.device),
        survival_bins=int(args.survival_bins),
        use_survival=bool(args.use_survival_for_mae),
        regression_loss=str(args.regression_loss),
        smooth_l1_beta=float(args.smooth_l1_beta),
        survival_weight=float(args.survival_weight),
        survival_value_weight=float(args.survival_value_weight),
        final_floor_bins=int(args.final_floor_bins),
        final_floor_readout=str(args.final_floor_readout),
        final_floor_weight=float(args.final_floor_weight),
        final_floor_value_weight=float(args.final_floor_value_weight),
        residual_floor_baseline=residual_floor_baseline,
        final_loss_weight=float(args.final_loss_weight),
        act_bce_weight=float(args.act_bce_weight),
        death_bce_weight=float(args.death_bce_weight),
    )
    summary = {
        "output": str(args.output),
        "baseline": baseline,
        "best_epoch": best_epoch,
        "best_validation_remaining_mae": best,
        "final_validation": final_valid,
        "manifest": manifest,
    }
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = dict(summary)
    printable_manifest = dict(printable.get("manifest") or {})
    if "source_files" in printable_manifest:
        printable_manifest["source_files"] = f"<{len(printable_manifest.get('source_files') or [])} files>"
    printable["manifest"] = printable_manifest
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
