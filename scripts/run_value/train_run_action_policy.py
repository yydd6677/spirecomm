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
import random
from pathlib import Path
from statistics import mean
from typing import Any

from spirecomm.ai.run_value import (
    ACTION_CANDIDATE_FEATURE_DIM,
    RunActionPolicyNetwork,
    encode_action_candidate,
    iter_jsonl,
    save_run_action_policy_checkpoint,
)
from spirecomm.ai.torch_compat import F, require_torch, torch


def _root_files(input_dir: Path) -> list[Path]:
    root_dir = input_dir / "roots"
    if root_dir.exists():
        paths = sorted(root_dir.glob("*.jsonl")) + sorted(root_dir.glob("*.jsonl.gz"))
    else:
        paths = sorted(input_dir.glob("*.jsonl")) + sorted(input_dir.glob("*.jsonl.gz"))
    return [path for path in paths if path.is_file()]


def _seed_is_validation(seed: int, *, val_mod: int, val_rem: int) -> bool:
    return int(val_mod) > 0 and int(seed) % int(val_mod) == int(val_rem)


def build_cache(
    *,
    input_dir: Path,
    cache_dir: Path,
    chunk_size_roots: int,
    val_mod: int,
    val_rem: int,
    max_roots: int,
    feature_variant: str,
    progress_interval: int,
) -> dict[str, Any]:
    require_torch()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stale in cache_dir.glob("chunk_*.pt"):
        stale.unlink()
    chunks = []
    root_features: list[Any] = []
    root_q: list[Any] = []
    root_counts: list[int] = []
    root_seeds: list[int] = []
    root_phases: list[str] = []
    total_roots = 0
    total_candidates = 0
    source_files = []

    def flush() -> None:
        nonlocal root_features, root_q, root_counts, root_seeds, root_phases
        if not root_counts:
            return
        flat_features = [feature for root in root_features for feature in root]
        flat_q = [score for root in root_q for score in root]
        is_validation = torch.tensor(
            [_seed_is_validation(seed, val_mod=val_mod, val_rem=val_rem) for seed in root_seeds],
            dtype=torch.bool,
        )
        payload = {
            "features": torch.tensor(flat_features, dtype=torch.float32),
            "q_env": torch.tensor(flat_q, dtype=torch.float32),
            "candidate_counts": torch.tensor(root_counts, dtype=torch.long),
            "seeds": torch.tensor(root_seeds, dtype=torch.long),
            "is_validation": is_validation,
            "phases": list(root_phases),
        }
        chunk_path = cache_dir / f"chunk_{len(chunks):05d}.pt"
        torch.save(payload, chunk_path)
        chunks.append(
            {
                "path": str(chunk_path),
                "root_count": len(root_counts),
                "candidate_count": len(flat_q),
                "validation_roots": int(is_validation.sum().item()),
            }
        )
        root_features = []
        root_q = []
        root_counts = []
        root_seeds = []
        root_phases = []

    for path in _root_files(input_dir):
        source_files.append(str(path))
        for root in iter_jsonl(path):
            before = root.get("state_before")
            candidates = list(root.get("candidates") or [])
            if not isinstance(before, dict) or len(candidates) < 2:
                continue
            features = []
            q_values = []
            for candidate in candidates:
                if "q_env" not in candidate:
                    continue
                action = candidate.get("action")
                after = candidate.get("after_state")
                if not isinstance(action, dict) or not isinstance(after, dict):
                    continue
                features.append(encode_action_candidate(before, action, after, feature_variant=str(feature_variant)))
                q_values.append(float(candidate["q_env"]))
            if len(q_values) < 2:
                continue
            root_features.append(features)
            root_q.append(q_values)
            root_counts.append(len(q_values))
            root_seeds.append(int(root.get("seed") or 0))
            root_phases.append(str(root.get("phase") or ""))
            total_roots += 1
            total_candidates += len(q_values)
            if int(progress_interval) > 0 and total_roots % int(progress_interval) == 0:
                print(
                    f"cache roots={total_roots} candidates={total_candidates} chunks={len(chunks)}",
                    flush=True,
                )
            if len(root_counts) >= int(chunk_size_roots):
                flush()
            if max_roots > 0 and total_roots >= max_roots:
                break
        if max_roots > 0 and total_roots >= max_roots:
            break
    flush()
    manifest = {
        "schema": "run_action_policy_tensor_cache_v1",
        "input_dir": str(input_dir),
        "cache_dir": str(cache_dir),
        "source_files": source_files,
        "chunks": chunks,
        "root_count": total_roots,
        "candidate_count": total_candidates,
        "candidate_feature_dim": ACTION_CANDIDATE_FEATURE_DIM,
        "feature_variant": str(feature_variant),
        "val_mod": int(val_mod),
        "val_rem": int(val_rem),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_manifest(cache_dir: Path) -> dict[str, Any]:
    return json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))


def _iter_root_batches(chunks: list[dict[str, Any]], *, batch_roots: int, device: str, train: bool, rng: random.Random):
    ordered = list(chunks)
    rng.shuffle(ordered)
    for chunk in ordered:
        payload = torch.load(Path(chunk["path"]), map_location="cpu", weights_only=False)
        counts = payload["candidate_counts"].tolist()
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + int(count))
        mask = ~payload["is_validation"] if train else payload["is_validation"]
        roots = [index for index, keep in enumerate(mask.tolist()) if keep]
        rng.shuffle(roots)
        for start in range(0, len(roots), int(batch_roots)):
            selected = roots[start : start + int(batch_roots)]
            feature_parts = []
            q_parts = []
            selected_counts = []
            for root_index in selected:
                begin, end = offsets[root_index], offsets[root_index + 1]
                feature_parts.append(payload["features"][begin:end])
                q_parts.append(payload["q_env"][begin:end])
                selected_counts.append(end - begin)
            if not feature_parts:
                continue
            yield {
                "features": torch.cat(feature_parts, dim=0).to(device),
                "q_env": torch.cat(q_parts, dim=0).to(device),
                "candidate_counts": selected_counts,
            }


def _root_loss(logits: Any, q_env: Any, counts: list[int], *, temperature: float) -> tuple[Any, dict[str, float]]:
    losses = []
    top1 = 0
    roots = 0
    cursor = 0
    regrets = []
    for count in counts:
        count = int(count)
        pred = logits[cursor : cursor + count].float()
        target = q_env[cursor : cursor + count].float()
        target_probs = F.softmax(target / max(1.0e-6, float(temperature)), dim=0)
        log_probs = F.log_softmax(pred, dim=0)
        losses.append(-(target_probs.detach() * log_probs).sum())
        pred_index = int(torch.argmax(pred).item())
        target_index = int(torch.argmax(target).item())
        top1 += 1 if pred_index == target_index else 0
        regrets.append(float((target[target_index] - target[pred_index]).detach().cpu().item()))
        roots += 1
        cursor += count
    if not losses:
        return logits.sum() * 0.0, {"loss": 0.0, "top1": 0.0, "mean_regret": 0.0}
    loss = torch.stack(losses).mean()
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "top1": float(top1) / max(1, roots),
        "mean_regret": mean(regrets) if regrets else 0.0,
    }


def _eval(model: RunActionPolicyNetwork, chunks: list[dict[str, Any]], *, batch_roots: int, device: str, temperature: float) -> dict[str, float]:
    model.eval()
    rng = random.Random(999)
    weighted = {"loss": 0.0, "top1": 0.0, "mean_regret": 0.0}
    roots = 0
    with torch.inference_mode():
        for batch in _iter_root_batches(chunks, batch_roots=batch_roots, device=device, train=False, rng=rng):
            logits = model(batch["features"])
            loss, metrics = _root_loss(logits, batch["q_env"], batch["candidate_counts"], temperature=temperature)
            n = len(batch["candidate_counts"])
            roots += n
            for key in weighted:
                weighted[key] += float(metrics[key]) * n
    return {key: value / max(1, roots) for key, value in weighted.items()} | {"roots": float(roots)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train unified run action policy from q_env roots.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--chunk-size-roots", type=int, default=1000)
    parser.add_argument("--max-roots", type=int, default=0)
    parser.add_argument("--val-mod", type=int, default=10)
    parser.add_argument("--val-rem", type=int, default=0)
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
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-roots", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--cache-progress-interval", type=int, default=1000)
    args = parser.parse_args()
    require_torch()
    if args.rebuild_cache or not (args.cache_dir / "manifest.json").exists():
        manifest = build_cache(
            input_dir=args.input_dir,
            cache_dir=args.cache_dir,
            chunk_size_roots=int(args.chunk_size_roots),
            val_mod=int(args.val_mod),
            val_rem=int(args.val_rem),
            max_roots=int(args.max_roots),
            feature_variant=str(args.feature_variant),
            progress_interval=int(args.cache_progress_interval),
        )
    else:
        manifest = load_manifest(args.cache_dir)
    chunks = list(manifest.get("chunks") or [])
    if not chunks:
        raise SystemExit("run action policy cache has no chunks")
    model = RunActionPolicyNetwork(
        input_dim=int(manifest.get("candidate_feature_dim") or ACTION_CANDIDATE_FEATURE_DIM),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)), eta_min=float(args.min_learning_rate))
    rng = random.Random(int(args.seed))
    best = None
    best_epoch = 0
    best_state = None
    history = []
    print(f"loaded qenv cache roots={manifest['root_count']} candidates={manifest['candidate_count']}", flush=True)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        weighted = {"loss": 0.0, "top1": 0.0, "mean_regret": 0.0}
        roots = 0
        batches = 0
        for batch in _iter_root_batches(chunks, batch_roots=int(args.batch_roots), device=str(args.device), train=True, rng=rng):
            logits = model(batch["features"])
            loss, metrics = _root_loss(logits, batch["q_env"], batch["candidate_counts"], temperature=float(args.target_temperature))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(batch["candidate_counts"])
            roots += n
            batches += 1
            for key in weighted:
                weighted[key] += float(metrics[key]) * n
            if int(args.progress_interval) > 0 and batches % int(args.progress_interval) == 0:
                print(f"epoch {epoch:03d} batch={batches} roots={roots} loss={metrics['loss']:.4f} top1={metrics['top1']:.3f}", flush=True)
        scheduler.step()
        train_metrics = {key: value / max(1, roots) for key, value in weighted.items()}
        valid_metrics = _eval(model, chunks, batch_roots=int(args.batch_roots), device=str(args.device), temperature=float(args.target_temperature))
        history.append({"epoch": epoch, "train": train_metrics, "validation": valid_metrics})
        current = float(valid_metrics["mean_regret"])
        if best is None or current < best:
            best = current
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch {epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"valid_top1={valid_metrics['top1']:.3f} valid_regret={valid_metrics['mean_regret']:.3f} best={best:.3f}@{best_epoch}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "manifest": manifest,
        "best_epoch": int(best_epoch),
        "best_validation_mean_regret": float(best or 0.0),
        "history": history,
    }
    save_run_action_policy_checkpoint(args.output, model, metadata=metadata)
    final_valid = _eval(model, chunks, batch_roots=int(args.batch_roots), device=str(args.device), temperature=float(args.target_temperature))
    summary = {
        "output": str(args.output),
        "best_epoch": int(best_epoch),
        "best_validation_mean_regret": best,
        "final_validation": final_valid,
        "manifest": manifest,
    }
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
