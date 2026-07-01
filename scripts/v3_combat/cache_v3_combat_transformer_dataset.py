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
import multiprocessing as mp
import re
from pathlib import Path
from typing import Any

from spirecomm.ai.torch_compat import require_torch
from spirecomm.ai.v3_combat_dataset import REWARD_COMPONENT_DIM, REWARD_COMPONENT_NAMES, load_shard, reward_component_vector
from spirecomm.ai.v3_combat_features import COMBAT_ROOM_TYPES, FEATURE_SCHEMA_VERSION, schema
from spirecomm.ai.v3_combat_transformer import (
    ROOT_TOKEN_SCHEMA_VERSION,
    ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA,
    TOKEN_SCHEMA_VERSION,
    TRANSFORMER_TENSOR_DATASET_SCHEMA,
    encode_root_transformer_actions,
    encode_transformer_candidate,
    root_token_spec,
    token_spec,
    transformer_entity_ids,
)


CHUNKED_TRANSFORMER_TENSOR_DATASET_SCHEMA = "v3_combat_transformer_tensor_dataset_chunked_v1"
CHUNKED_ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA = "v3_combat_root_transformer_tensor_dataset_chunked_v1"
ROOM_TYPE_IDS = {room_type: index + 1 for index, room_type in enumerate(COMBAT_ROOM_TYPES)}


def _candidate_reward_components(candidate: Any) -> list[float] | None:
    components = getattr(candidate, "reward_components", None)
    if isinstance(components, dict) and components:
        return reward_component_vector(components)
    return None


def _derive_root_reward_components(labeled: Any) -> list[list[float]] | None:
    try:
        from spirecomm.ai.v3_combat_teacher import (
            _clone_env_blob_or_none,
            _combat_turn,
            _is_terminal_from_state,
            _phase_from_state,
            _root_turn_lethal_available_from_branches,
            _state_from_env,
            _step_branch_with_blob,
            default_teacher_config,
            transition_reward_components,
        )
    except Exception:
        return None
    try:
        cfg = default_teacher_config()
        root_env = labeled.root.load_env()
        before_state = getattr(labeled.root, "visible_before", None) or _state_from_env(root_env)
        root_turn = _combat_turn(root_env)
        root_env_blob = _clone_env_blob_or_none(root_env)
        expanded = []
        for candidate in list(getattr(labeled, "candidates", []) or []):
            branch = _step_branch_with_blob(root_env, root_env_blob, candidate.action)
            visible_after = _state_from_env(branch)
            expanded.append((candidate, branch, visible_after))
        root_lethal_available = _root_turn_lethal_available_from_branches(
            [(candidate.action, branch, visible_after) for candidate, branch, visible_after in expanded],
            root_turn=root_turn,
            config=cfg,
        )
        vectors: list[list[float]] = []
        for candidate, branch, visible_after in expanded:
            after_phase = _phase_from_state(branch, visible_after)
            after_terminal = _is_terminal_from_state(branch, visible_after, after_phase)
            components = transition_reward_components(
                root_env,
                before_state,
                candidate.action,
                branch,
                visible_after,
                cfg,
                lethal_available=root_lethal_available,
                block_reward_base_state=before_state,
                after_phase=after_phase,
                after_terminal=after_terminal,
            )
            immediate_total = float(components.get("immediate_total", 0.0))
            teacher_q = float(getattr(candidate, "teacher_q", 0.0))
            continuation_adjusted = float(teacher_q - immediate_total)
            components["continuation_raw"] = continuation_adjusted
            components["continuation_adjusted"] = continuation_adjusted
            components["teacher_q"] = teacher_q
            vectors.append(reward_component_vector(components))
        return vectors
    except Exception:
        return None


def _token_schema_payload(spec: Any) -> dict[str, Any]:
    payload = dict(getattr(spec, "__dict__", {}) or {})
    payload["max_sequence_length"] = int(spec.max_sequence_length)
    if hasattr(spec, "action_segment_width"):
        payload["action_segment_width"] = int(spec.action_segment_width)
    if hasattr(spec, "uses_legacy_token"):
        payload["uses_legacy_token"] = bool(spec.uses_legacy_token)
    return payload


def _root_seed(root_id: str) -> int | None:
    match = re.match(r"explore:(-?\d+):", str(root_id))
    if not match:
        return None
    return int(match.group(1))


def _root_id(labeled: Any) -> str:
    root = getattr(labeled, "root", None)
    return str(getattr(root, "root_id", ""))


def _cache_shard_chunk_worker(args: tuple[int, str, str, str, bool, str, bool, bool]) -> dict[str, Any]:
    (
        shard_index,
        shard_path_raw,
        chunk_path_raw,
        dtype,
        root_action_set,
        token_schema_version,
        derive_reward_components,
        resume_chunks,
    ) = args
    torch = require_torch()
    spec = root_token_spec() if root_action_set else token_spec(version=token_schema_version)
    float_dtype = torch.float16 if dtype == "float16" else torch.float32
    shard_path = Path(shard_path_raw)
    chunk_path = Path(chunk_path_raw)
    if resume_chunks and chunk_path.exists():
        try:
            chunk_payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
            metadata = dict(chunk_payload.get("metadata") or {})
            if str(metadata.get("source_shard") or "") == str(shard_path):
                root_ids = list(chunk_payload.get("root_ids") or [])
                source_shards = list(chunk_payload.get("source_shards") or [])
                seed_values = [seed for seed in (_root_seed(root_id) for root_id in root_ids) if seed is not None]
                return {
                    "shard_index": shard_index,
                    "chunk": {
                        "path": str(chunk_path),
                        "root_count": int(metadata.get("root_count") or len(root_ids)),
                        "candidate_count": int(metadata.get("candidate_count") or 0),
                        "source_shard": str(shard_path),
                    },
                    "root_ids": root_ids,
                    "source_shards": source_shards,
                    "seed_values": seed_values,
                    "root_count": int(metadata.get("root_count") or len(root_ids)),
                    "candidate_count": int(metadata.get("candidate_count") or 0),
                    "source_shard": str(shard_path),
                    "resumed": True,
                }
        except Exception:
            pass
    payload = load_shard(shard_path)
    roots = payload.get("roots") or []
    shard_token_scalars = []
    shard_token_types = []
    shard_entities = []
    shard_slots = []
    shard_masks = []
    shard_before_summary = []
    shard_action_positions = []
    shard_after_positions = []
    shard_delta_positions = []
    shard_legacy_positions = []
    shard_candidate_masks = []
    shard_legacy = []
    shard_teacher_q = []
    shard_reward_components = []
    shard_sample_ids = []
    shard_chosen = []
    shard_action_is_potion = []
    shard_room_type_ids = []
    shard_candidate_offsets = [0]
    shard_root_ids: list[str] = []
    shard_source_shards: list[str] = []
    shard_candidate_count = 0
    seed_values: list[int] = []
    reward_component_count = 0

    for shard_root_count, labeled in enumerate(roots):
        before_state = labeled.root.visible_before
        room_type_id = ROOM_TYPE_IDS.get(str(before_state.get("room_type") or ""), 0)
        rid = _root_id(labeled)
        shard_root_ids.append(rid)
        shard_source_shards.append(str(shard_path))
        seed = _root_seed(rid)
        if seed is not None:
            seed_values.append(seed)
        derived_reward_components = None
        if derive_reward_components:
            derived_reward_components = _derive_root_reward_components(labeled)
        if root_action_set:
            record = encode_root_transformer_actions(
                before_state,
                [candidate.action for candidate in labeled.candidates],
                [candidate.visible_after for candidate in labeled.candidates],
                candidate_features=[candidate.candidate_features for candidate in labeled.candidates],
                spec=spec,
            )
            shard_token_scalars.append(record["token_scalar_features"])
            shard_token_types.append(record["token_type_ids"])
            shard_entities.append(record["entity_ids"])
            shard_slots.append(record["slot_ids"])
            shard_masks.append(record["attention_mask"])
            shard_before_summary.append(record["before_summary"])
            shard_action_positions.append(record["action_token_positions"])
            shard_after_positions.append(record["after_token_positions"])
            shard_delta_positions.append(record["delta_token_positions"])
            shard_legacy_positions.append(record["legacy_token_positions"])
            shard_candidate_masks.append(record["candidate_mask"])
            for candidate_index, (candidate, features) in enumerate(zip(labeled.candidates, record["candidate_features"], strict=False)):
                component_vector = _candidate_reward_components(candidate)
                if component_vector is None and derived_reward_components is not None and candidate_index < len(derived_reward_components):
                    component_vector = derived_reward_components[candidate_index]
                if component_vector is None:
                    component_vector = [0.0] * REWARD_COMPONENT_DIM
                else:
                    reward_component_count += 1
                shard_legacy.append(features)
                shard_teacher_q.append(float(candidate.teacher_q))
                shard_reward_components.append(component_vector)
                shard_sample_ids.append(shard_root_count)
                shard_chosen.append(bool(candidate.is_chosen))
                shard_action_is_potion.append(str(candidate.action.get("kind") or "") == "potion")
                shard_room_type_ids.append(room_type_id)
                shard_candidate_count += 1
        else:
            for candidate in labeled.candidates:
                component_vector = _candidate_reward_components(candidate)
                if component_vector is None and derived_reward_components is not None and len(derived_reward_components) > (shard_candidate_count - shard_candidate_offsets[-1]):
                    component_vector = derived_reward_components[shard_candidate_count - shard_candidate_offsets[-1]]
                if component_vector is None:
                    component_vector = [0.0] * REWARD_COMPONENT_DIM
                else:
                    reward_component_count += 1
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
                shard_reward_components.append(component_vector)
                shard_sample_ids.append(shard_root_count)
                shard_chosen.append(bool(candidate.is_chosen))
                shard_action_is_potion.append(str(candidate.action.get("kind") or "") == "potion")
                shard_room_type_ids.append(room_type_id)
                shard_candidate_count += 1
        shard_candidate_offsets.append(shard_candidate_count)

    if not shard_token_scalars:
        return {
            "shard_index": shard_index,
            "chunk": None,
            "root_ids": [],
            "source_shards": [],
            "seed_values": [],
            "root_count": 0,
            "candidate_count": 0,
            "source_shard": str(shard_path),
            "resumed": False,
        }

    chunk_payload = {
        "tensor_dataset_schema": ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA if root_action_set else TRANSFORMER_TENSOR_DATASET_SCHEMA,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_dims": schema().__dict__,
        "token_schema_version": ROOT_TOKEN_SCHEMA_VERSION if root_action_set else spec.version,
        "token_schema": _token_schema_payload(spec),
        "entity_vocab": list(transformer_entity_ids()),
        "token_scalar_features": torch.tensor(shard_token_scalars, dtype=float_dtype).contiguous(),
        "token_type_ids": torch.tensor(shard_token_types, dtype=torch.uint8).contiguous(),
        "entity_ids": torch.tensor(shard_entities, dtype=torch.int32).contiguous(),
        "slot_ids": torch.tensor(shard_slots, dtype=torch.int16).contiguous(),
        "attention_mask": torch.tensor(shard_masks, dtype=torch.bool).contiguous(),
        "features": torch.tensor(shard_legacy, dtype=float_dtype).contiguous(),
        "teacher_q": torch.tensor(shard_teacher_q, dtype=torch.float32).contiguous(),
        "reward_components": torch.tensor(shard_reward_components, dtype=torch.float32).contiguous(),
        "sample_ids": torch.tensor(shard_sample_ids, dtype=torch.long).contiguous(),
        "chosen": torch.tensor(shard_chosen, dtype=torch.bool).contiguous(),
        "action_is_potion": torch.tensor(shard_action_is_potion, dtype=torch.bool).contiguous(),
        "room_type_ids": torch.tensor(shard_room_type_ids, dtype=torch.int16).contiguous(),
        "candidate_offsets": torch.tensor(shard_candidate_offsets, dtype=torch.long),
        "root_ids": shard_root_ids,
        "source_shards": shard_source_shards,
        "metadata": {
            "root_count": len(shard_root_ids),
            "candidate_count": shard_candidate_count,
            "source_shard": str(shard_path),
            "dtype": dtype,
            "sequence_length": spec.max_sequence_length,
            "scalar_dim": spec.scalar_dim,
            "entity_vocab_size": len(transformer_entity_ids()),
            "reward_component_names": list(REWARD_COMPONENT_NAMES),
            "reward_component_count": reward_component_count,
        },
    }
    if root_action_set:
        chunk_payload.update(
            {
                "before_summary": torch.tensor(shard_before_summary, dtype=float_dtype).contiguous(),
                "action_token_positions": torch.tensor(shard_action_positions, dtype=torch.int32).contiguous(),
                "after_token_positions": torch.tensor(shard_after_positions, dtype=torch.int32).contiguous(),
                "delta_token_positions": torch.tensor(shard_delta_positions, dtype=torch.int32).contiguous(),
                "legacy_token_positions": torch.tensor(shard_legacy_positions, dtype=torch.int32).contiguous(),
                "candidate_mask": torch.tensor(shard_candidate_masks, dtype=torch.bool).contiguous(),
            }
        )
        chunk_payload["metadata"]["max_actions"] = spec.max_actions
        chunk_payload["metadata"]["action_segment_width"] = spec.action_segment_width
        chunk_payload["metadata"]["uses_legacy_token"] = bool(getattr(spec, "uses_legacy_token", False))
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(chunk_payload, chunk_path)
    return {
        "shard_index": shard_index,
        "chunk": {
            "path": str(chunk_path),
            "root_count": len(shard_root_ids),
            "candidate_count": shard_candidate_count,
            "source_shard": str(shard_path),
        },
        "root_ids": shard_root_ids,
        "source_shards": shard_source_shards,
        "seed_values": seed_values,
        "root_count": len(shard_root_ids),
        "candidate_count": shard_candidate_count,
        "source_shard": str(shard_path),
        "resumed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert v3 combat teacher shards into a transformer tensor dataset.")
    parser.add_argument("--shards", nargs="*", type=Path, default=[])
    parser.add_argument("--shards-file", type=Path, default=None, help="Optional newline-delimited shard path list.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--limit-roots", type=int, default=0, help="Optional smoke-test cap; 0 means no cap.")
    parser.add_argument("--chunked", action="store_true", help="Write per-shard tensor chunks plus a small manifest.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel shard encoders for --chunked mode.")
    parser.add_argument(
        "--resume-chunks",
        action="store_true",
        help="Reuse existing chunk files whose source_shard metadata matches the current shard list.",
    )
    parser.add_argument("--root-action-set", action="store_true", help="Write one root-level main-attention sequence per combat root instead of one sequence per candidate.")
    parser.add_argument(
        "--derive-reward-components",
        action="store_true",
        help="For old labeled shards without reward_components, derive immediate reward components from env_blob during caching.",
    )
    parser.add_argument(
        "--token-schema-version",
        default=TOKEN_SCHEMA_VERSION,
        help="Candidate token schema version. Use v6 structured phase2 for the seven-way structural sweep.",
    )
    args = parser.parse_args()
    if args.root_action_set and not args.chunked:
        raise SystemExit("--root-action-set currently requires --chunked.")
    if args.root_action_set and int(args.limit_roots) > 0:
        raise SystemExit("--root-action-set does not support --limit-roots; pass a smaller shard list for smoke caching.")

    torch = require_torch()
    spec = root_token_spec() if args.root_action_set else token_spec(version=args.token_schema_version)
    float_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    token_scalar_tensors = []
    token_type_tensors = []
    entity_tensors = []
    slot_tensors = []
    mask_tensors = []
    legacy_tensors = []
    teacher_tensors = []
    reward_component_tensors = []
    sample_id_tensors = []
    chosen_tensors = []
    action_is_potion_tensors = []
    room_type_id_tensors = []
    candidate_offsets = [0]
    root_ids: list[str] = []
    source_shards: list[str] = []
    seed_values: list[int] = []
    candidate_count = 0
    root_count = 0
    limit_roots = max(0, int(args.limit_roots))
    chunk_dir = args.output.with_suffix(args.output.suffix + ".chunks")
    chunks: list[dict[str, Any]] = []

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
    if args.chunked and (int(args.workers) > 1 or args.root_action_set) and not limit_roots:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        tasks = [
            (
                shard_index,
                str(shard_path),
                str(chunk_dir / f"chunk_{shard_index:05d}.pt"),
                args.dtype,
                bool(args.root_action_set),
                str(spec.version),
                bool(args.derive_reward_components),
                bool(args.resume_chunks),
            )
            for shard_index, shard_path in enumerate(shards)
        ]
        results: list[dict[str, Any]] = []
        with mp.Pool(processes=max(1, int(args.workers))) as pool:
            for completed, result in enumerate(pool.imap_unordered(_cache_shard_chunk_worker, tasks), start=1):
                results.append(result)
                print(
                    f"cached shard {completed}/{len(shards)} "
                    f"source={result['source_shard']} roots={result['root_count']}"
                    f"{' resumed' if result.get('resumed') else ''}",
                    flush=True,
                )
        results.sort(key=lambda item: int(item["shard_index"]))
        for result in results:
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
        output = {
            "tensor_dataset_schema": CHUNKED_ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA
            if args.root_action_set
            else CHUNKED_TRANSFORMER_TENSOR_DATASET_SCHEMA,
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "feature_dims": schema().__dict__,
            "token_schema_version": ROOT_TOKEN_SCHEMA_VERSION if args.root_action_set else spec.version,
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
                "dtype": args.dtype,
                "limit_roots": None,
                "sequence_length": spec.max_sequence_length,
                "scalar_dim": spec.scalar_dim,
                "entity_vocab_size": len(transformer_entity_ids()),
                "max_actions": getattr(spec, "max_actions", None),
                "action_segment_width": getattr(spec, "action_segment_width", None),
                "uses_legacy_token": bool(getattr(spec, "uses_legacy_token", False)),
                "reward_component_names": list(REWARD_COMPONENT_NAMES),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, args.output)
        summary = dict(output["metadata"])
        summary.update(
            {
                "output": str(args.output),
                "chunk_dir": str(chunk_dir),
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "feature_dim": schema().candidate_dim,
                "token_schema": ROOT_TOKEN_SCHEMA_VERSION if args.root_action_set else spec.version,
                "root_action_set": bool(args.root_action_set),
                "token_shape": [
                    root_count if args.root_action_set else candidate_count,
                    spec.max_sequence_length,
                    spec.scalar_dim,
                ],
            }
        )
        summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    for shard_index, shard_path in enumerate(shards):
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
        shard_reward_components = []
        shard_sample_ids = []
        shard_chosen = []
        shard_action_is_potion = []
        shard_room_type_ids = []
        shard_candidate_offsets = [0]
        shard_root_ids: list[str] = []
        shard_source_shards: list[str] = []
        shard_candidate_count = 0
        shard_root_count = 0
        reward_component_count = 0
        for labeled in roots:
            if limit_roots and root_count >= limit_roots:
                break
            before_state = labeled.root.visible_before
            room_type_id = ROOM_TYPE_IDS.get(str(before_state.get("room_type") or ""), 0)
            rid = _root_id(labeled)
            root_ids.append(rid)
            source_shards.append(str(shard_path))
            shard_root_ids.append(rid)
            shard_source_shards.append(str(shard_path))
            seed = _root_seed(rid)
            if seed is not None:
                seed_values.append(seed)
            derived_reward_components = None
            if bool(args.derive_reward_components):
                derived_reward_components = _derive_root_reward_components(labeled)
            for candidate_index, candidate in enumerate(labeled.candidates):
                component_vector = _candidate_reward_components(candidate)
                if component_vector is None and derived_reward_components is not None and candidate_index < len(derived_reward_components):
                    component_vector = derived_reward_components[candidate_index]
                if component_vector is None:
                    component_vector = [0.0] * REWARD_COMPONENT_DIM
                else:
                    reward_component_count += 1
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
                shard_reward_components.append(component_vector)
                shard_sample_ids.append(shard_root_count if args.chunked else root_count)
                shard_chosen.append(bool(candidate.is_chosen))
                shard_action_is_potion.append(str(candidate.action.get("kind") or "") == "potion")
                shard_room_type_ids.append(room_type_id)
                candidate_count += 1
                shard_candidate_count += 1
            root_count += 1
            shard_root_count += 1
            candidate_offsets.append(candidate_count)
            shard_candidate_offsets.append(shard_candidate_count)
        if shard_token_scalars:
            chunk_payload = {
                "tensor_dataset_schema": TRANSFORMER_TENSOR_DATASET_SCHEMA,
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "feature_dims": schema().__dict__,
                "token_schema_version": spec.version,
                "token_schema": _token_schema_payload(spec),
                "entity_vocab": list(transformer_entity_ids()),
                "token_scalar_features": torch.tensor(shard_token_scalars, dtype=float_dtype).contiguous(),
                "token_type_ids": torch.tensor(shard_token_types, dtype=torch.uint8).contiguous(),
                "entity_ids": torch.tensor(shard_entities, dtype=torch.int32).contiguous(),
                "slot_ids": torch.tensor(shard_slots, dtype=torch.int16).contiguous(),
                "attention_mask": torch.tensor(shard_masks, dtype=torch.bool).contiguous(),
                "features": torch.tensor(shard_legacy, dtype=float_dtype).contiguous(),
                "teacher_q": torch.tensor(shard_teacher_q, dtype=torch.float32).contiguous(),
                "reward_components": torch.tensor(shard_reward_components, dtype=torch.float32).contiguous(),
                "sample_ids": torch.tensor(shard_sample_ids, dtype=torch.long).contiguous(),
                "chosen": torch.tensor(shard_chosen, dtype=torch.bool).contiguous(),
                "action_is_potion": torch.tensor(shard_action_is_potion, dtype=torch.bool).contiguous(),
                "room_type_ids": torch.tensor(shard_room_type_ids, dtype=torch.int16).contiguous(),
                "candidate_offsets": torch.tensor(shard_candidate_offsets if args.chunked else candidate_offsets[-(shard_root_count + 1) :], dtype=torch.long),
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
                    "reward_component_names": list(REWARD_COMPONENT_NAMES),
                    "reward_component_count": reward_component_count,
                },
            }
            if args.chunked:
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
            else:
                token_scalar_tensors.append(chunk_payload["token_scalar_features"])
                token_type_tensors.append(chunk_payload["token_type_ids"])
                entity_tensors.append(chunk_payload["entity_ids"])
                slot_tensors.append(chunk_payload["slot_ids"])
                mask_tensors.append(chunk_payload["attention_mask"])
                legacy_tensors.append(chunk_payload["features"])
                teacher_tensors.append(chunk_payload["teacher_q"])
                reward_component_tensors.append(chunk_payload["reward_components"])
                sample_id_tensors.append(torch.tensor([root_count - shard_root_count + int(value) for value in shard_sample_ids], dtype=torch.long))
                chosen_tensors.append(chunk_payload["chosen"])
                action_is_potion_tensors.append(chunk_payload["action_is_potion"])
                room_type_id_tensors.append(chunk_payload["room_type_ids"])
        print(f"cached shard {shard_index + 1}/{len(shards)} roots={len(roots)} total_roots={root_count}", flush=True)

    if args.chunked:
        if not chunks:
            raise SystemExit("No transformer candidates found in shards.")
        output = {
            "tensor_dataset_schema": CHUNKED_TRANSFORMER_TENSOR_DATASET_SCHEMA,
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "feature_dims": schema().__dict__,
            "token_schema_version": spec.version,
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
                "dtype": args.dtype,
                "limit_roots": limit_roots or None,
                "sequence_length": spec.max_sequence_length,
                "scalar_dim": spec.scalar_dim,
                "entity_vocab_size": len(transformer_entity_ids()),
                "reward_component_names": list(REWARD_COMPONENT_NAMES),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, args.output)
        summary = dict(output["metadata"])
        summary.update(
            {
                "output": str(args.output),
                "chunk_dir": str(chunk_dir),
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "feature_dim": schema().candidate_dim,
                "token_schema": spec.version,
                "token_shape": [candidate_count, spec.max_sequence_length, spec.scalar_dim],
            }
        )
        summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    if not token_scalar_tensors:
        raise SystemExit("No transformer candidates found in shards.")

    output = {
        "tensor_dataset_schema": TRANSFORMER_TENSOR_DATASET_SCHEMA,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_dims": schema().__dict__,
        "token_schema_version": spec.version,
        "token_schema": _token_schema_payload(spec),
        "entity_vocab": list(transformer_entity_ids()),
        "token_scalar_features": torch.cat(token_scalar_tensors, dim=0).contiguous(),
        "token_type_ids": torch.cat(token_type_tensors, dim=0).contiguous(),
        "entity_ids": torch.cat(entity_tensors, dim=0).contiguous(),
        "slot_ids": torch.cat(slot_tensors, dim=0).contiguous(),
        "attention_mask": torch.cat(mask_tensors, dim=0).contiguous(),
        "features": torch.cat(legacy_tensors, dim=0).contiguous(),
        "teacher_q": torch.cat(teacher_tensors, dim=0).contiguous(),
        "reward_components": torch.cat(reward_component_tensors, dim=0).contiguous(),
        "sample_ids": torch.cat(sample_id_tensors, dim=0).contiguous(),
        "chosen": torch.cat(chosen_tensors, dim=0).contiguous(),
        "action_is_potion": torch.cat(action_is_potion_tensors, dim=0).contiguous(),
        "room_type_ids": torch.cat(room_type_id_tensors, dim=0).contiguous(),
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
            "limit_roots": limit_roots or None,
            "sequence_length": spec.max_sequence_length,
            "scalar_dim": spec.scalar_dim,
            "entity_vocab_size": len(transformer_entity_ids()),
            "reward_component_names": list(REWARD_COMPONENT_NAMES),
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
            "token_schema": spec.version,
            "token_shape": list(output["token_scalar_features"].shape),
        }
    )
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
