#!/usr/bin/env python3
from __future__ import annotations
# Allow this CLI to run directly from its workflow subdirectory.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import gc
import json
from pathlib import Path
from typing import Any

from scripts.v3_combat.generate_v3_combat_teacher_dataset import (
    _append_jsonl,
    _git_status,
    _memory_snapshot,
    _merge_root_stats,
    label_roots,
    root_stats,
)
from spirecomm.ai.v3_combat_dataset import V3CombatLabeledRoot, load_shard, save_shard
from spirecomm.ai.v3_combat_features import FEATURE_SCHEMA_VERSION
from spirecomm.ai.v3_combat_teacher import TEACHER_VERSION, TeacherConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel existing v3 combat root shards with the current teacher.")
    parser.add_argument(
        "--source-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing old shard_*.pt files. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-roots", type=int, default=20000)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument(
        "--label-batch-shards",
        type=int,
        default=1,
        help="Label this many output shards in one worker-pool batch to reduce per-shard long-tail idle time.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--node-budget", type=int, default=768)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--memory-log", type=Path, default=None)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append new shards after existing output shards instead of overwriting or count-based source resume.",
    )
    parser.add_argument(
        "--dedupe-root-ids-from",
        action="append",
        type=Path,
        default=[],
        help="Skip source roots whose root_id already exists in these shard dirs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--only-relabel-potion",
        action="append",
        default=[],
        help=(
            "Only relabel roots containing one of these potion names/ids; other roots are reused unchanged. "
            "Names are normalized, so BlessingOfTheForge matches Blessing of the Forge."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue after existing complete output shards by skipping the same number of source roots.",
    )
    parser.add_argument(
        "--fast-resume",
        action="store_true",
        help=(
            "With --resume, skip already-covered full source shards by filename/count without loading them. "
            "Only use when source shards use the same full shard size as the output shards."
        ),
    )
    parser.add_argument(
        "--relabel-all",
        action="store_true",
        help="Relabel every root. By default, roots without potion candidates are reused unchanged.",
    )
    return parser.parse_args()


def _source_shards(source_dirs: list[Path]) -> list[Path]:
    shards: list[Path] = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            raise FileNotFoundError(f"source dir not found: {source_dir}")
        shards.extend(sorted(source_dir.glob("shard_*.pt")))
    if not shards:
        raise FileNotFoundError("no shard_*.pt files found in source dirs")
    return shards


def _shards_from_dirs(dirs: list[Path]) -> list[Path]:
    shards: list[Path] = []
    for shard_dir in dirs:
        if not shard_dir.exists():
            raise FileNotFoundError(f"dedupe dir not found: {shard_dir}")
        shards.extend(sorted(shard_dir.glob("shard_*.pt")))
    return shards


def _labeled_roots_from_payload(payload: dict[str, Any]) -> list[V3CombatLabeledRoot]:
    labeled_roots = []
    for labeled_root in payload.get("roots") or []:
        root = getattr(labeled_root, "root", None)
        if root is not None:
            labeled_roots.append(labeled_root)
    return labeled_roots


def _load_root_ids_from_shards(shards: list[Path]) -> tuple[set[str], int]:
    root_ids: set[str] = set()
    root_count = 0
    for shard_path in shards:
        payload = load_shard(shard_path)
        for labeled_root in _labeled_roots_from_payload(payload):
            root_count += 1
            root_id = _root_id(labeled_root.root)
            if root_id is not None:
                root_ids.add(root_id)
        del payload
    gc.collect()
    return root_ids, root_count


def _root_has_potion_candidate(root: Any) -> bool:
    return any(str(action.get("kind") or "") == "potion" for action in list(getattr(root, "actions", []) or []))


def _normalize_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _root_id(root: Any) -> str | None:
    value = getattr(root, "root_id", None)
    return str(value) if value is not None else None


def _action_matches_any_potion(action: dict[str, Any], potion_names: set[str]) -> bool:
    if str(action.get("kind") or "") != "potion":
        return False
    if not potion_names:
        return True
    return any(
        _normalize_key(action.get(key)) in potion_names
        for key in ("name", "potion_id", "id")
    )


def _root_has_matching_potion_candidate(root: Any, potion_names: set[str]) -> bool:
    return any(
        _action_matches_any_potion(action, potion_names)
        for action in list(getattr(root, "actions", []) or [])
    )


def _reuse_labeled_root(labeled_root: V3CombatLabeledRoot, *, reason: str) -> V3CombatLabeledRoot:
    teacher_config = dict(getattr(labeled_root, "teacher_config", {}) or {})
    source_version = str(teacher_config.get("version") or "unknown")
    teacher_config.update(
        {
            "version": TEACHER_VERSION,
            "reused_labels": True,
            "reused_from_teacher_version": source_version,
            "reuse_reason": reason,
        }
    )
    return V3CombatLabeledRoot(
        root=labeled_root.root,
        candidates=list(labeled_root.candidates),
        teacher_config=teacher_config,
    )


def _strip_unused_state_debug_fields(state: Any) -> None:
    if isinstance(state, dict):
        state.pop("rng_trace", None)
        state.pop("commands", None)


def _strip_unused_state_debug_fields_from_labeled_root(labeled_root: V3CombatLabeledRoot) -> None:
    _strip_unused_state_debug_fields(getattr(labeled_root.root, "visible_before", None))
    for candidate in list(getattr(labeled_root, "candidates", []) or []):
        _strip_unused_state_debug_fields(getattr(candidate, "visible_after", None))


def _trim_relabel_input(
    labeled_root: V3CombatLabeledRoot,
    *,
    reuse_non_potion: bool,
    relabel_potion_names: set[str],
) -> V3CombatLabeledRoot:
    _strip_unused_state_debug_fields_from_labeled_root(labeled_root)
    root = labeled_root.root
    if reuse_non_potion and not _root_has_matching_potion_candidate(root, relabel_potion_names):
        return labeled_root
    return V3CombatLabeledRoot(root=root, candidates=[], teacher_config=dict(labeled_root.teacher_config or {}))


def _label_or_reuse_roots(
    labeled_roots: list[V3CombatLabeledRoot],
    *,
    config: TeacherConfig,
    workers: int,
    reuse_non_potion: bool,
    relabel_potion_names: set[str],
    executor: ProcessPoolExecutor | None = None,
) -> tuple[list[V3CombatLabeledRoot], dict[str, int]]:
    if not reuse_non_potion:
        roots = [labeled_root.root for labeled_root in labeled_roots]
        labeled = label_roots(roots, config=config, workers=max(1, int(workers)), executor=executor)
        return labeled, {
            "roots": len(labeled_roots),
            "relabeled_roots": len(labeled),
            "reused_non_potion_roots": 0,
            "potion_candidate_roots": sum(1 for root in roots if _root_has_potion_candidate(root)),
        }

    results: list[V3CombatLabeledRoot | None] = [None] * len(labeled_roots)
    potion_indexes: list[int] = []
    potion_roots: list[Any] = []
    for index, labeled_root in enumerate(labeled_roots):
        root = labeled_root.root
        if _root_has_matching_potion_candidate(root, relabel_potion_names):
            potion_indexes.append(index)
            potion_roots.append(root)
        else:
            reason = (
                "root_has_no_matching_potion_candidate"
                if relabel_potion_names
                else "root_has_no_potion_candidate_v6_equivalent"
            )
            results[index] = _reuse_labeled_root(labeled_root, reason=reason)

    relabeled = (
        label_roots(potion_roots, config=config, workers=max(1, int(workers)), executor=executor)
        if potion_roots
        else []
    )
    if len(relabeled) != len(potion_indexes):
        raise RuntimeError(f"relabeled root count mismatch: {len(relabeled)} != {len(potion_indexes)}")
    for index, labeled_root in zip(potion_indexes, relabeled, strict=True):
        results[index] = labeled_root

    if any(root is None for root in results):
        raise RuntimeError("internal error: missing relabel/reuse result")
    return [root for root in results if root is not None], {
        "roots": len(labeled_roots),
        "relabeled_roots": len(relabeled),
        "reused_non_potion_roots": len(labeled_roots) - len(relabeled),
        "potion_candidate_roots": len(relabeled),
    }


def _relabel_stats_for_roots(
    labeled_roots: list[V3CombatLabeledRoot],
    *,
    reuse_non_potion: bool,
    relabel_potion_names: set[str],
) -> dict[str, int]:
    potion_roots = sum(
        1
        for labeled_root in labeled_roots
        if _root_has_matching_potion_candidate(labeled_root.root, relabel_potion_names)
    )
    if reuse_non_potion:
        relabeled_roots = potion_roots
        reused_non_potion_roots = len(labeled_roots) - potion_roots
    else:
        relabeled_roots = len(labeled_roots)
        reused_non_potion_roots = 0
    return {
        "roots": len(labeled_roots),
        "relabeled_roots": relabeled_roots,
        "reused_non_potion_roots": reused_non_potion_roots,
        "potion_candidate_roots": potion_roots,
    }


def _split_labeled_roots(
    labeled_roots: list[V3CombatLabeledRoot],
    lengths: list[int],
) -> list[list[V3CombatLabeledRoot]]:
    chunks: list[list[V3CombatLabeledRoot]] = []
    offset = 0
    for length in lengths:
        chunks.append(labeled_roots[offset : offset + length])
        offset += length
    if offset != len(labeled_roots):
        raise RuntimeError(f"labeled root split mismatch: {offset} != {len(labeled_roots)}")
    return chunks


def _save_relabel_shard(
    *,
    labeled_roots: list[V3CombatLabeledRoot],
    output_dir: Path,
    shard_index: int,
    config: TeacherConfig,
    workers: int,
    repo_root: Path,
    metadata: dict[str, Any],
    memory_log: Path | None,
    reuse_non_potion: bool,
    relabel_potion_names: set[str],
    executor: ProcessPoolExecutor | None = None,
) -> tuple[Path, int, dict[str, int]]:
    root_count = len(labeled_roots)
    potion_roots = sum(
        1
        for labeled_root in labeled_roots
        if _root_has_matching_potion_candidate(labeled_root.root, relabel_potion_names)
    )
    _append_jsonl(
        memory_log,
        _memory_snapshot(
            event="relabel_shard_start",
            shard_index=shard_index,
            roots=root_count,
            potion_candidate_roots=potion_roots,
            reuse_non_potion=bool(reuse_non_potion),
        ),
    )
    labeled, relabel_stats = _label_or_reuse_roots(
        labeled_roots,
        config=config,
        workers=max(1, int(workers)),
        reuse_non_potion=bool(reuse_non_potion),
        relabel_potion_names=relabel_potion_names,
        executor=executor,
    )
    shard_path = output_dir / f"shard_{shard_index:05d}.pt"
    save_shard(
        shard_path,
        labeled,
        metadata={
            **metadata,
            "root_count": len(labeled),
            "shard_index": int(shard_index),
            "relabel_strategy": "relabel_all" if not reuse_non_potion else "relabel_potion_roots_reuse_non_potion",
            "relabel_stats": relabel_stats,
            "git_status": _git_status(repo_root),
            "teacher_version": TEACHER_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "teacher_config": config.__dict__,
            "root_stats": root_stats([labeled_root.root for labeled_root in labeled_roots]),
        },
    )
    count = len(labeled)
    del labeled
    gc.collect()
    _append_jsonl(
        memory_log,
        _memory_snapshot(
            event="relabel_shard_done",
            shard_index=shard_index,
            output_roots=count,
            path=str(shard_path),
            **relabel_stats,
        ),
    )
    return shard_path, count, relabel_stats


def _save_relabel_batch(
    *,
    shard_roots: list[list[V3CombatLabeledRoot]],
    output_dir: Path,
    shard_index: int,
    config: TeacherConfig,
    workers: int,
    repo_root: Path,
    metadata: dict[str, Any],
    memory_log: Path | None,
    reuse_non_potion: bool,
    relabel_potion_names: set[str],
    executor: ProcessPoolExecutor | None = None,
) -> list[tuple[Path, int, dict[str, int]]]:
    if not shard_roots:
        return []
    flat_roots = [labeled_root for roots in shard_roots for labeled_root in roots]
    lengths = [len(roots) for roots in shard_roots]
    potion_roots = sum(
        1
        for labeled_root in flat_roots
        if _root_has_matching_potion_candidate(labeled_root.root, relabel_potion_names)
    )
    _append_jsonl(
        memory_log,
        _memory_snapshot(
            event="relabel_batch_start",
            shard_index=shard_index,
            shard_count=len(shard_roots),
            roots=len(flat_roots),
            potion_candidate_roots=potion_roots,
            reuse_non_potion=bool(reuse_non_potion),
        ),
    )
    labeled, aggregate_stats = _label_or_reuse_roots(
        flat_roots,
        config=config,
        workers=max(1, int(workers)),
        reuse_non_potion=bool(reuse_non_potion),
        relabel_potion_names=relabel_potion_names,
        executor=executor,
    )
    labeled_chunks = _split_labeled_roots(labeled, lengths)
    saved: list[tuple[Path, int, dict[str, int]]] = []
    for offset, (input_roots, labeled_chunk) in enumerate(zip(shard_roots, labeled_chunks, strict=True)):
        current_shard_index = int(shard_index) + int(offset)
        relabel_stats = _relabel_stats_for_roots(
            input_roots,
            reuse_non_potion=bool(reuse_non_potion),
            relabel_potion_names=relabel_potion_names,
        )
        shard_path = output_dir / f"shard_{current_shard_index:05d}.pt"
        save_shard(
            shard_path,
            labeled_chunk,
            metadata={
                **metadata,
                "root_count": len(labeled_chunk),
                "shard_index": current_shard_index,
                "relabel_strategy": "relabel_all" if not reuse_non_potion else "relabel_potion_roots_reuse_non_potion",
                "relabel_stats": relabel_stats,
                "git_status": _git_status(repo_root),
                "teacher_version": TEACHER_VERSION,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "teacher_config": config.__dict__,
                "root_stats": root_stats([labeled_root.root for labeled_root in input_roots]),
            },
        )
        count = len(labeled_chunk)
        saved.append((shard_path, count, relabel_stats))
        _append_jsonl(
            memory_log,
            _memory_snapshot(
                event="relabel_shard_done",
                shard_index=current_shard_index,
                output_roots=count,
                path=str(shard_path),
                **relabel_stats,
            ),
        )
    _append_jsonl(
        memory_log,
        _memory_snapshot(
            event="relabel_batch_done",
            shard_index=shard_index,
            shard_count=len(shard_roots),
            batch_roots=len(flat_roots),
            **aggregate_stats,
        ),
    )
    del labeled
    gc.collect()
    return saved


def main() -> None:
    args = parse_args()
    if args.target_roots <= 0:
        raise SystemExit("--target-roots must be positive")
    if args.shard_size <= 0:
        raise SystemExit("--shard-size must be positive")
    if args.label_batch_shards <= 0:
        raise SystemExit("--label-batch-shards must be positive")
    if args.fast_resume and not args.resume:
        raise SystemExit("--fast-resume requires --resume")
    if args.append_output and args.resume:
        raise SystemExit("--append-output and --resume are mutually exclusive")
    existing_output_shards = sorted(args.output_dir.glob("shard_*.pt")) if args.output_dir.exists() else []
    if existing_output_shards and not args.allow_existing_output and not args.resume and not args.append_output:
        raise SystemExit(f"output dir already contains shards: {args.output_dir}")

    repo_root = _REPO_ROOT
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_log = args.memory_log or (output_dir / "memory_log.jsonl")
    config = TeacherConfig(
        beam_width=int(args.beam_width),
        node_budget_per_root=int(args.node_budget),
        max_depth=int(args.max_depth),
    )

    source_shards = _source_shards(args.source_dir)
    relabel_potion_names = {_normalize_key(name) for name in args.only_relabel_potion}
    existing_output_root_ids: set[str] = set()
    existing_output_roots = 0
    if args.append_output and existing_output_shards:
        existing_output_root_ids, existing_output_roots = _load_root_ids_from_shards(existing_output_shards)
    dedupe_root_ids = set(existing_output_root_ids)
    explicit_dedupe_shards = _shards_from_dirs(list(args.dedupe_root_ids_from or []))
    if explicit_dedupe_shards:
        explicit_root_ids, _ = _load_root_ids_from_shards(explicit_dedupe_shards)
        dedupe_root_ids.update(explicit_root_ids)
    metadata = {
        "source": "relabel_existing_roots",
        "source_dirs": [str(path) for path in args.source_dir],
        "target_roots": int(args.target_roots),
        "shard_size": int(args.shard_size),
        "label_batch_shards": int(args.label_batch_shards),
        "workers": int(args.workers),
        "relabel_strategy": "relabel_all" if args.relabel_all else "relabel_potion_roots_reuse_non_potion",
        "persistent_worker_pool": bool(int(args.workers) > 1),
        "fast_resume": bool(args.fast_resume),
        "append_output": bool(args.append_output),
        "existing_output_roots": int(existing_output_roots),
        "dedupe_root_ids_from": [str(path) for path in list(args.dedupe_root_ids_from or [])],
        "dedupe_root_ids": int(len(dedupe_root_ids)),
        "only_relabel_potion": list(args.only_relabel_potion or []),
        "only_relabel_potion_normalized": sorted(relabel_potion_names),
        "source_teacher_versions": {},
    }
    aggregate_stats = root_stats([])
    aggregate_relabel_stats = {
        "roots": 0,
        "relabeled_roots": 0,
        "reused_non_potion_roots": 0,
        "potion_candidate_roots": 0,
    }
    resume_roots = len(existing_output_shards) * int(args.shard_size) if args.resume else 0
    if args.fast_resume and resume_roots % int(args.shard_size) != 0:
        raise SystemExit(
            f"--fast-resume requires resume roots to align to shard size: {resume_roots} vs {int(args.shard_size)}"
        )
    fast_resume_source_shards = int(resume_roots // int(args.shard_size)) if args.fast_resume else 0
    if resume_roots >= int(args.target_roots):
        raise SystemExit(
            f"resume output already has at least target roots: {resume_roots} >= {int(args.target_roots)}"
        )
    output_shards: list[str] = [str(path) for path in existing_output_shards] if (args.resume or args.append_output) else []
    buffer: list[V3CombatLabeledRoot] = []
    pending_shards: list[list[V3CombatLabeledRoot]] = []
    total_roots = int(existing_output_roots if args.append_output else resume_roots)
    source_roots_seen = int(resume_roots)
    source_roots_skipped = 0
    source_roots_skipped_duplicate = 0
    shard_index = len(existing_output_shards) if (args.resume or args.append_output) else 0

    def queued_roots() -> int:
        return len(buffer) + sum(len(roots) for roots in pending_shards)

    def flush_pending() -> None:
        nonlocal pending_shards, shard_index, total_roots
        if not pending_shards:
            return
        saved = _save_relabel_batch(
            shard_roots=pending_shards,
            output_dir=output_dir,
            shard_index=shard_index,
            config=config,
            workers=max(1, int(args.workers)),
            repo_root=repo_root,
            metadata=metadata,
            memory_log=memory_log,
            reuse_non_potion=not bool(args.relabel_all),
            relabel_potion_names=relabel_potion_names,
            executor=executor,
        )
        for shard_path, count, relabel_stats in saved:
            output_shards.append(str(shard_path))
            total_roots += int(count)
            for key, value in relabel_stats.items():
                aggregate_relabel_stats[key] = int(aggregate_relabel_stats.get(key, 0)) + int(value)
            shard_index += 1
        pending_shards = []
        gc.collect()

    def queue_full_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        pending_shards.append(buffer)
        buffer = []

    worker_pool_context = (
        ProcessPoolExecutor(max_workers=max(1, int(args.workers))) if int(args.workers) > 1 else nullcontext(None)
    )
    with worker_pool_context as executor:
        for source_index, shard_path in enumerate(source_shards, 1):
            if total_roots + queued_roots() >= int(args.target_roots):
                break
            if source_index <= fast_resume_source_shards:
                source_roots_skipped += int(args.shard_size)
                _append_jsonl(
                    memory_log,
                    _memory_snapshot(
                        event="source_shard_fast_skipped_for_resume",
                        index=source_index,
                        path=str(shard_path),
                        assumed_source_roots=int(args.shard_size),
                        skipped_roots=source_roots_skipped,
                        resume_roots=resume_roots,
                    ),
                )
                continue
            _append_jsonl(memory_log, _memory_snapshot(event="source_shard_start", index=source_index, path=str(shard_path)))
            payload = load_shard(shard_path)
            labeled_roots = _labeled_roots_from_payload(payload)
            roots = [labeled_root.root for labeled_root in labeled_roots]
            if args.resume and source_roots_skipped + len(labeled_roots) <= resume_roots:
                source_roots_skipped += len(labeled_roots)
                _append_jsonl(
                    memory_log,
                    _memory_snapshot(
                        event="source_shard_skipped_for_resume",
                        index=source_index,
                        path=str(shard_path),
                        source_roots=len(roots),
                        skipped_roots=source_roots_skipped,
                        resume_roots=resume_roots,
                    ),
                )
                continue
            source_roots_seen += len(roots)
            source_version = str((payload.get("metadata") or {}).get("teacher_version") or "unknown")
            metadata["source_teacher_versions"][source_version] = int(metadata["source_teacher_versions"].get(source_version, 0)) + len(roots)
            _merge_root_stats(aggregate_stats, root_stats(roots))
            for labeled_root in labeled_roots:
                if args.resume and source_roots_skipped < resume_roots:
                    source_roots_skipped += 1
                    continue
                root_id = _root_id(labeled_root.root)
                if root_id is not None and root_id in dedupe_root_ids:
                    source_roots_skipped_duplicate += 1
                    continue
                if total_roots + queued_roots() >= int(args.target_roots):
                    break
                if root_id is not None:
                    dedupe_root_ids.add(root_id)
                buffer.append(
                    _trim_relabel_input(
                        labeled_root,
                        reuse_non_potion=not bool(args.relabel_all),
                        relabel_potion_names=relabel_potion_names,
                    )
                )
                if len(buffer) >= int(args.shard_size):
                    queue_full_buffer()
            _append_jsonl(
                memory_log,
                _memory_snapshot(
                    event="source_shard_done",
                    index=source_index,
                    path=str(shard_path),
                    source_roots=len(roots),
                    output_roots=total_roots,
                    buffered_roots=len(buffer),
                    pending_shards=len(pending_shards),
                    queued_roots=queued_roots(),
                    skipped_duplicate_roots=source_roots_skipped_duplicate,
                ),
            )
            del payload, labeled_roots, roots
            if len(pending_shards) >= int(args.label_batch_shards):
                flush_pending()

        if buffer:
            queue_full_buffer()
        flush_pending()
    summary = {
        "output_dir": str(output_dir),
        "shards": output_shards,
        "roots": int(total_roots),
        "source_roots_seen": int(source_roots_seen),
        "source_dirs": [str(path) for path in args.source_dir],
        "target_roots": int(args.target_roots),
        "resume_roots": int(resume_roots),
        "append_output": bool(args.append_output),
        "existing_output_roots": int(existing_output_roots),
        "label_batch_shards": int(args.label_batch_shards),
        "teacher_version": TEACHER_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "root_stats": aggregate_stats,
        "relabel_strategy": metadata["relabel_strategy"],
        "persistent_worker_pool": metadata["persistent_worker_pool"],
        "fast_resume": metadata["fast_resume"],
        "dedupe_root_ids_from": metadata["dedupe_root_ids_from"],
        "dedupe_root_ids": int(len(dedupe_root_ids)),
        "source_roots_skipped_duplicate": int(source_roots_skipped_duplicate),
        "only_relabel_potion": list(args.only_relabel_potion or []),
        "relabel_stats": aggregate_relabel_stats,
        "source_teacher_versions": metadata["source_teacher_versions"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
