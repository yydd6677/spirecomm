#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import time
from pathlib import Path

from spirecomm.ai.torch_compat import require_torch
from spirecomm.ai.v3_combat_dataset import collate_labeled_roots, load_labeled_roots, load_shard, total_candidate_loss
from spirecomm.ai.v3_combat_features import FEATURE_SCHEMA_VERSION
from spirecomm.ai.v3_combat_model import V3CombatCandidateScorer, load_v3_combat_checkpoint, save_v3_combat_checkpoint


def _format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))


def _mem_available_kb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except Exception:
        return None
    return None


def _rss_kb() -> int | None:
    try:
        status = Path(f"/proc/{os.getpid()}/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        return None
    return None


def _memory_snapshot(torch_module=None, *, device: str = "cpu", **extra) -> dict:
    payload = {
        "time": time.time(),
        "pid": os.getpid(),
        "rss_kb": _rss_kb(),
        "mem_available_kb": _mem_available_kb(),
    }
    if torch_module is not None and str(device).startswith("cuda") and torch_module.cuda.is_available():
        try:
            index = torch_module.cuda.current_device()
            payload.update(
                {
                    "cuda_device": int(index),
                    "cuda_allocated_bytes": int(torch_module.cuda.memory_allocated(index)),
                    "cuda_reserved_bytes": int(torch_module.cuda.memory_reserved(index)),
                }
            )
        except Exception as exc:
            payload["cuda_error"] = str(exc)
    payload.update(extra)
    return payload


def _append_memory_log(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _run_epoch(
    model,
    roots,
    optimizer,
    *,
    device: str,
    batch_size: int,
    seed: int,
    loss_kwargs: dict[str, float] | None = None,
) -> dict[str, float]:
    torch = require_torch()
    rng = random.Random(seed)
    ordered = list(roots)
    if optimizer is not None:
        rng.shuffle(ordered)
        model.train()
    else:
        model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for start in range(0, len(ordered), batch_size):
        batch_roots = ordered[start : start + batch_size]
        batch = collate_labeled_roots(batch_roots, device=device)
        with torch.set_grad_enabled(optimizer is not None):
            pred_q = model(batch["features"])
            loss, metrics = total_candidate_loss(pred_q, batch, **(loss_kwargs or {}))
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def _count_roots_by_shard(shards: list[Path]) -> int:
    total = 0
    for shard in shards:
        payload = load_shard(shard)
        total += len(payload.get("roots") or [])
        del payload
    gc.collect()
    return total


def _count_roots_by_shard_once(shards: list[Path]) -> dict[Path, int]:
    counts: dict[Path, int] = {}
    for shard in shards:
        payload = load_shard(shard)
        counts[shard] = len(payload.get("roots") or [])
        del payload
    gc.collect()
    return counts


def _strip_unused_training_fields(roots) -> None:
    for labeled in roots:
        root = labeled.root
        root.env_blob = b""
        root.visible_before = {}
        root.actions = []
        root.action_keys = []
        root.chosen_action_key = None
        root.metadata = {}
        for candidate in labeled.candidates:
            candidate.action = {}
            candidate.action_key = ()
            candidate.visible_after = {}
            candidate.delta_features = []
            candidate.debug_best_line = []


def _load_labeled_roots_for_training(paths: list[Path], *, strip_unused_fields: bool) -> list:
    roots = []
    for path in paths:
        shard_roots = load_labeled_roots([path])
        if strip_unused_fields:
            _strip_unused_training_fields(shard_roots)
        roots.extend(shard_roots)
        del shard_roots
    gc.collect()
    return roots


def _run_epoch_streaming(
    model,
    shards: list[Path],
    optimizer,
    *,
    device: str,
    batch_size: int,
    seed: int,
    shard_group_size: int,
    loss_kwargs: dict[str, float] | None = None,
) -> dict[str, float]:
    rng = random.Random(seed)
    ordered_shards = list(shards)
    if optimizer is not None:
        rng.shuffle(ordered_shards)
        model.train()
    else:
        model.eval()
    totals: dict[str, float] = {}
    batches = 0
    group_size = max(1, int(shard_group_size))
    for start in range(0, len(ordered_shards), group_size):
        shard_group = ordered_shards[start : start + group_size]
        roots = _load_labeled_roots_for_training(shard_group, strip_unused_fields=True)
        if optimizer is not None:
            rng.shuffle(roots)
        metrics = _run_epoch(
            model,
            roots,
            optimizer,
            device=device,
            batch_size=batch_size,
            seed=seed,
            loss_kwargs=loss_kwargs,
        )
        shard_batches = max(1, (len(roots) + batch_size - 1) // batch_size)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * shard_batches
        batches += shard_batches
        del roots
        gc.collect()
    return {key: value / max(1, batches) for key, value in totals.items()}


def _load_tensor_dataset(path: Path) -> dict:
    torch = require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("tensor_dataset_schema") != "v3_combat_tensor_dataset_v1":
        raise ValueError(f"unsupported tensor dataset schema in {path}: {payload.get('tensor_dataset_schema')}")
    if payload.get("feature_schema") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "v3 combat tensor dataset feature schema mismatch in "
            f"{path}: {payload.get('feature_schema')} != {FEATURE_SCHEMA_VERSION}"
        )
    return payload


def _tensor_batch(payload: dict, root_indices, *, device: str) -> dict:
    torch = require_torch()
    offsets = payload["candidate_offsets"]
    root_tensor = torch.tensor(root_indices, dtype=torch.long)
    starts = offsets.index_select(0, root_tensor)
    ends = offsets.index_select(0, root_tensor + 1)
    counts = ends - starts
    if not bool((counts > 0).any().item()):
        raise ValueError("empty tensor root batch")
    root_count = int(root_tensor.numel())
    total_candidates = int(counts.sum().item())
    local_sample_ids = torch.repeat_interleave(torch.arange(root_count, dtype=torch.long), counts)
    root_starts = torch.repeat_interleave(starts, counts)
    candidate_offsets = torch.arange(total_candidates, dtype=torch.long) - torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
    candidate_indices = root_starts + candidate_offsets
    return {
        "features": payload["features"].index_select(0, candidate_indices).to(device, dtype=torch.float32, non_blocking=True),
        "teacher_q": payload["teacher_q"].index_select(0, candidate_indices).to(device, non_blocking=True),
        "sample_ids": local_sample_ids.to(device, non_blocking=True),
        "chosen": payload["chosen"].index_select(0, candidate_indices).to(device, non_blocking=True),
        "candidate_counts": counts.to(device, non_blocking=True),
        "root_count": len(root_indices),
    }


def _run_epoch_tensor(
    model,
    payload: dict,
    root_indices: list[int],
    optimizer,
    *,
    device: str,
    batch_size: int,
    seed: int,
    loss_kwargs: dict[str, float] | None = None,
) -> dict[str, float]:
    torch = require_torch()
    rng = random.Random(seed)
    ordered = list(root_indices)
    if optimizer is not None:
        rng.shuffle(ordered)
        model.train()
    else:
        model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for start in range(0, len(ordered), batch_size):
        batch_roots = ordered[start : start + batch_size]
        batch = _tensor_batch(payload, batch_roots, device=device)
        with torch.set_grad_enabled(optimizer is not None):
            pred_q = model(batch["features"])
            loss, metrics = total_candidate_loss(pred_q, batch, **(loss_kwargs or {}))
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v3 combat candidate scorer.")
    parser.add_argument("--shards", nargs="+", type=Path, default=[])
    parser.add_argument("--tensor-dataset", type=Path, default=None, help="Train from a compact tensor dataset produced by cache_v3_combat_tensor_dataset.py.")
    parser.add_argument("--output", type=Path, default=Path("models/v3_combat_scorer.pt"))
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Initialize model weights from an existing v3 combat scorer checkpoint.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--stream-shards", action="store_true", help="Load one dataset shard at a time instead of keeping all roots in RAM.")
    parser.add_argument("--stream-shard-group-size", type=int, default=1, help="When streaming, load this many shards at once to trade memory for speed.")
    parser.add_argument("--preload-train-shards", action="store_true", help="In streaming mode, keep the training split in RAM and stream only validation shards.")
    parser.add_argument("--strip-unused-fields", action="store_true", help="Drop env/debug payloads that are not used by training after loading each shard.")
    parser.add_argument("--memory-log", type=Path, default=None)
    parser.add_argument("--min-mem-available-gb", type=float, default=3.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--q-weight", type=float, default=0.5)
    parser.add_argument("--pair-weight", type=float, default=0.2)
    parser.add_argument("--bc-weight", type=float, default=0.05)
    parser.add_argument("--rank-temperature", type=float, default=1.0)
    parser.add_argument("--potion-vs-non-potion-weight", type=float, default=0.0)
    parser.add_argument("--teacher-q-clip", type=float, default=0.0)
    args = parser.parse_args()

    torch = require_torch()
    memory_log = args.memory_log or args.output.with_suffix(args.output.suffix + ".memory_log.jsonl")
    min_mem_available_kb = int(max(0.0, float(args.min_mem_available_gb)) * 1024 * 1024)
    shard_paths = list(args.shards)
    tensor_payload = None
    tensor_train_indices: list[int] = []
    tensor_validation_indices: list[int] = []
    if args.tensor_dataset is not None:
        tensor_payload = _load_tensor_dataset(args.tensor_dataset)
        root_count = int(tensor_payload["metadata"]["root_count"])
        rng = random.Random(args.seed)
        indices = list(range(root_count))
        rng.shuffle(indices)
        val_count = int(root_count * args.validation_fraction)
        tensor_validation_indices = indices[:val_count]
        tensor_train_indices = indices[val_count:] or indices
        train_count = len(tensor_train_indices)
        validation_count = len(tensor_validation_indices)
        validation_roots = []
        train_roots = None
        validation_shards = []
        train_shards = []
    elif args.stream_shards:
        if not shard_paths:
            raise SystemExit("--shards is required unless --tensor-dataset is used.")
        rng = random.Random(args.seed)
        rng.shuffle(shard_paths)
        val_shard_count = int(len(shard_paths) * args.validation_fraction)
        validation_shards = shard_paths[:val_shard_count]
        train_shards = shard_paths[val_shard_count:] or shard_paths
        train_roots = (
            _load_labeled_roots_for_training(train_shards, strip_unused_fields=True)
            if args.preload_train_shards
            else None
        )
        if train_roots is not None:
            validation_roots = (
                _load_labeled_roots_for_training(validation_shards, strip_unused_fields=True)
                if validation_shards
                else []
            )
            train_count = len(train_roots)
            validation_count = len(validation_roots)
            root_count = train_count + validation_count
        else:
            shard_counts = _count_roots_by_shard_once(shard_paths)
            root_count = sum(shard_counts.values())
            validation_count = sum(shard_counts.get(path, 0) for path in validation_shards)
            train_count = sum(shard_counts.get(path, 0) for path in train_shards)
            validation_roots = []
        if root_count <= 0:
            raise SystemExit("No v3 combat teacher roots loaded.")
    else:
        if not shard_paths:
            raise SystemExit("--shards is required unless --tensor-dataset is used.")
        roots = _load_labeled_roots_for_training(shard_paths, strip_unused_fields=args.strip_unused_fields)
        if not roots:
            raise SystemExit("No v3 combat teacher roots loaded.")
        rng = random.Random(args.seed)
        rng.shuffle(roots)
        val_count = int(len(roots) * args.validation_fraction)
        validation_roots = roots[:val_count]
        train_roots = roots[val_count:] or roots
        root_count = len(roots)
        validation_count = len(validation_roots)
        train_count = len(train_roots)
        validation_shards = []
        train_shards = []

    if root_count <= 0:
        raise SystemExit("No v3 combat teacher roots loaded.")

    init_checkpoint_metadata = None
    if args.init_checkpoint is not None:
        model, init_checkpoint = load_v3_combat_checkpoint(args.init_checkpoint, device=args.device)
        init_checkpoint_metadata = dict(init_checkpoint.get("dataset_metadata") or {})
    else:
        model = V3CombatCandidateScorer().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(args.epochs)),
            eta_min=max(0.0, float(args.min_learning_rate)),
        )
    best_validation = None
    best_state_dict = None
    best_epoch = None

    print(
        "loaded "
        f"roots={root_count} train={train_count} validation={validation_count} "
        f"stream_shards={args.stream_shards} preload_train_shards={args.preload_train_shards} "
        f"tensor_dataset={args.tensor_dataset is not None} "
        f"rank_weight={float(args.rank_weight)} q_weight={float(args.q_weight)} "
        f"pair_weight={float(args.pair_weight)} bc_weight={float(args.bc_weight)} "
        f"rank_temperature={float(args.rank_temperature)} "
        f"potion_pair_weight={float(args.potion_vs_non_potion_weight)} "
        f"teacher_q_clip={float(args.teacher_q_clip)}"
    )
    loss_kwargs = {
        "rank_weight": float(args.rank_weight),
        "q_weight": float(args.q_weight),
        "pair_weight": float(args.pair_weight),
        "bc_weight": float(args.bc_weight),
        "temperature": float(args.rank_temperature),
        "potion_pair_weight": float(args.potion_vs_non_potion_weight),
        "teacher_q_clip": float(args.teacher_q_clip),
    }
    _append_memory_log(
        memory_log,
        _memory_snapshot(
            torch,
            device=args.device,
            event="training_start",
            roots=root_count,
            train=train_count,
            validation=validation_count,
            stream_shards=args.stream_shards,
            tensor_dataset=str(args.tensor_dataset) if args.tensor_dataset is not None else None,
            init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        ),
    )
    stopped_for_low_memory = False
    for epoch in range(1, args.epochs + 1):
        _append_memory_log(memory_log, _memory_snapshot(torch, device=args.device, event="epoch_start", epoch=epoch))
        if tensor_payload is not None:
            train_metrics = _run_epoch_tensor(
                model,
                tensor_payload,
                tensor_train_indices,
                optimizer,
                device=args.device,
                batch_size=args.batch_size,
                seed=args.seed + epoch,
                loss_kwargs=loss_kwargs,
            )
        elif args.stream_shards:
            if args.preload_train_shards:
                train_metrics = _run_epoch(
                    model,
                    train_roots,
                    optimizer,
                    device=args.device,
                    batch_size=args.batch_size,
                    seed=args.seed + epoch,
                    loss_kwargs=loss_kwargs,
                )
            else:
                train_metrics = _run_epoch_streaming(
                    model,
                    train_shards,
                    optimizer,
                    device=args.device,
                    batch_size=args.batch_size,
                    seed=args.seed + epoch,
                    shard_group_size=args.stream_shard_group_size,
                    loss_kwargs=loss_kwargs,
                )
        else:
            train_metrics = _run_epoch(
                model,
                train_roots,
                optimizer,
                device=args.device,
                batch_size=args.batch_size,
                seed=args.seed + epoch,
                loss_kwargs=loss_kwargs,
            )
        print(f"epoch {epoch:03d} train {_format_metrics(train_metrics)}", flush=True)
        current_lr = float(optimizer.param_groups[0]["lr"])
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _append_memory_log(
            memory_log,
            _memory_snapshot(torch, device=args.device, event="epoch_train_done", epoch=epoch, metrics=train_metrics, learning_rate=current_lr),
        )
        validation_metrics = {}
        if validation_count:
            if tensor_payload is not None:
                validation_metrics = _run_epoch_tensor(
                    model,
                    tensor_payload,
                    tensor_validation_indices,
                    None,
                    device=args.device,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    loss_kwargs=loss_kwargs,
                )
            elif args.stream_shards:
                if args.preload_train_shards:
                    validation_metrics = _run_epoch(
                        model,
                        validation_roots,
                        None,
                        device=args.device,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        loss_kwargs=loss_kwargs,
                    )
                else:
                    validation_metrics = _run_epoch_streaming(
                        model,
                        validation_shards,
                        None,
                        device=args.device,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        shard_group_size=args.stream_shard_group_size,
                        loss_kwargs=loss_kwargs,
                    )
            else:
                validation_metrics = _run_epoch(
                    model,
                    validation_roots,
                    None,
                    device=args.device,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    loss_kwargs=loss_kwargs,
                )
            print(f"epoch {epoch:03d} valid {_format_metrics(validation_metrics)}", flush=True)
            current = validation_metrics.get("loss")
            if current is not None and (best_validation is None or current < best_validation):
                best_validation = current
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        if scheduler is not None:
            scheduler.step()
        _append_memory_log(
            memory_log,
            _memory_snapshot(
                torch,
                device=args.device,
                event="epoch_done",
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                learning_rate=current_lr,
                next_learning_rate=float(optimizer.param_groups[0]["lr"]),
            ),
        )
        available = _mem_available_kb()
        if min_mem_available_kb > 0 and available is not None and available < min_mem_available_kb:
            stopped_for_low_memory = True
            print(
                "stopping after epoch "
                f"{epoch}: MemAvailable={available / 1024 / 1024:.2f}GB "
                f"< threshold={args.min_mem_available_gb:.2f}GB",
                flush=True,
            )
            if best_state_dict is None:
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    save_v3_combat_checkpoint(
        args.output,
        model,
        training_args=vars(args),
        dataset_metadata={
            "root_count": root_count,
            "train_count": train_count,
            "validation_count": validation_count,
            "best_validation_loss": best_validation,
            "best_epoch": best_epoch,
            "stopped_for_low_memory": stopped_for_low_memory,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
            "init_checkpoint_best_validation_loss": init_checkpoint_metadata.get("best_validation_loss") if init_checkpoint_metadata else None,
        },
    )
    _append_memory_log(
        memory_log,
        _memory_snapshot(torch, device=args.device, event="training_done", best_epoch=best_epoch, best_validation_loss=best_validation, stopped_for_low_memory=stopped_for_low_memory),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "roots": root_count,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation,
                "stopped_for_low_memory": stopped_for_low_memory,
                "memory_log": str(memory_log),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
