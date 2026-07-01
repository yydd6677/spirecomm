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
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

from spirecomm.ai.card_reward_model import (
    CANDIDATE_DIM,
    build_state_vector,
    candidate_vector,
    canonical_card_key,
    load_card_reward_checkpoint,
    save_card_reward_checkpoint,
)
from spirecomm.ai.torch_compat import F, torch


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_hard_records(input_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(input_root / "hard_decisions.jsonl"):
        root_id = str(row.get("root_id") or "")
        if root_id:
            records[root_id] = row
    return records


def _candidate_card(action: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    if action.get("kind") in {"skip", "proceed"} or str(action.get("name") or "").upper() == "SKIP":
        return None
    card = action.get("card")
    if isinstance(card, dict):
        return card
    return {
        key: action.get(key)
        for key in ("card_id", "id", "name", "type", "rarity", "cost", "base_cost", "upgrades", "misc")
        if key in action
    }


def _is_skip(action: dict[str, Any]) -> bool:
    if not isinstance(action, dict):
        return False
    return action.get("kind") in {"skip", "proceed"} or str(action.get("name") or "").upper() == "SKIP"


def _branch_score(branch: dict[str, Any]) -> float:
    result = branch.get("result") if isinstance(branch.get("result"), dict) else {}
    return float(result.get("branch_score") or -999999.0)


def _prior_logit(card: dict[str, Any] | None, *, summary: dict[str, Any]) -> float:
    if not card:
        return 0.0
    prior = summary.get("upgrade_rate_prior")
    if not isinstance(prior, dict) or not prior:
        return 0.0
    eps = float(summary.get("upgrade_rate_prior_eps", 1e-6) or 1e-6)
    weight = float(summary.get("upgrade_rate_prior_weight", 2.0) or 2.0)
    key = canonical_card_key(card)
    return weight * math.log(float(prior.get(key, 0.0)) + eps)


def _make_pair_examples(
    *,
    labels: list[dict[str, Any]],
    hard_records: dict[str, dict[str, Any]],
    sources: set[str],
    min_branch_gap: float,
    max_pairs_per_root: int,
    weight_scale: float,
    weight_max: float,
    base_summary: dict[str, Any],
    include_fixed_prior: bool,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for label in labels:
        source = str(label.get("source") or "")
        if source not in sources:
            continue
        record = hard_records.get(str(label.get("root_id") or ""))
        if not record:
            continue
        scenario = record.get("state") if isinstance(record.get("state"), dict) else {}
        branches = list(label.get("branches") or [])
        candidates: list[dict[str, Any]] = []
        for index, branch in enumerate(branches):
            action = branch.get("candidate") if isinstance(branch.get("candidate"), dict) else {}
            card = _candidate_card(action)
            candidates.append(
                {
                    "index": index,
                    "action": action,
                    "card": card,
                    "is_skip": _is_skip(action),
                    "branch_score": _branch_score(branch),
                    "fixed_prior": _prior_logit(card, summary=base_summary) if include_fixed_prior else 0.0,
                }
            )
        root_pairs: list[dict[str, Any]] = []
        for pos in candidates:
            for neg in candidates:
                if pos["index"] == neg["index"]:
                    continue
                gap = float(pos["branch_score"]) - float(neg["branch_score"])
                if gap < min_branch_gap:
                    continue
                weight = min(weight_max, max(1.0, math.sqrt(max(0.0, gap)) / max(1e-6, weight_scale)))
                root_pairs.append(
                    {
                        "scenario": scenario,
                        "pos_card": pos["card"],
                        "neg_card": neg["card"],
                        "pos_is_skip": bool(pos["is_skip"]),
                        "neg_is_skip": bool(neg["is_skip"]),
                        "pos_fixed_prior": float(pos["fixed_prior"]),
                        "neg_fixed_prior": float(neg["fixed_prior"]),
                        "weight": float(weight),
                        "root_id": label.get("root_id"),
                        "source": source,
                        "branch_gap": gap,
                    }
                )
        root_pairs.sort(key=lambda item: float(item["branch_gap"]), reverse=True)
        if max_pairs_per_root > 0:
            root_pairs = root_pairs[:max_pairs_per_root]
        pairs.extend(root_pairs)
    return pairs


def _split(items: list[dict[str, Any]], valid_fraction: float, seed: int):
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    valid_size = max(1, int(len(shuffled) * valid_fraction)) if len(shuffled) >= 8 else 0
    return shuffled[valid_size:], shuffled[:valid_size]


def _batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _batch_to_tensors(batch: list[dict[str, Any]], device: str):
    state_rows = []
    pos_rows = []
    neg_rows = []
    pos_skip = []
    neg_skip = []
    pos_prior = []
    neg_prior = []
    weights = []
    for item in batch:
        scenario = item["scenario"]
        state_rows.append(build_state_vector(scenario))
        pos_rows.append([0.0] * CANDIDATE_DIM if item["pos_is_skip"] or item["pos_card"] is None else candidate_vector(item["pos_card"], scenario))
        neg_rows.append([0.0] * CANDIDATE_DIM if item["neg_is_skip"] or item["neg_card"] is None else candidate_vector(item["neg_card"], scenario))
        pos_skip.append(1.0 if item["pos_is_skip"] else 0.0)
        neg_skip.append(1.0 if item["neg_is_skip"] else 0.0)
        pos_prior.append(float(item.get("pos_fixed_prior") or 0.0))
        neg_prior.append(float(item.get("neg_fixed_prior") or 0.0))
        weights.append(float(item.get("weight") or 1.0))
    return {
        "state": torch.tensor(state_rows, dtype=torch.float32, device=device),
        "pos_candidate": torch.tensor(pos_rows, dtype=torch.float32, device=device),
        "neg_candidate": torch.tensor(neg_rows, dtype=torch.float32, device=device),
        "pos_is_skip": torch.tensor(pos_skip, dtype=torch.float32, device=device),
        "neg_is_skip": torch.tensor(neg_skip, dtype=torch.float32, device=device),
        "pos_prior": torch.tensor(pos_prior, dtype=torch.float32, device=device),
        "neg_prior": torch.tensor(neg_prior, dtype=torch.float32, device=device),
        "weight": torch.tensor(weights, dtype=torch.float32, device=device),
    }


def _pair_scores(model: Any, batch: dict[str, Any]):
    hidden = model.encode_state(batch["state"])
    pos_candidate_score = model.score_candidate_with_hidden(hidden, batch["pos_candidate"])
    neg_candidate_score = model.score_candidate_with_hidden(hidden, batch["neg_candidate"])
    skip_score = model.score_skip_with_hidden(hidden)
    pos_score = torch.where(batch["pos_is_skip"] > 0.5, skip_score, pos_candidate_score) + batch["pos_prior"]
    neg_score = torch.where(batch["neg_is_skip"] > 0.5, skip_score, neg_candidate_score) + batch["neg_prior"]
    return pos_score, neg_score


def _evaluate_pairs(model: Any, pairs: list[dict[str, Any]], device: str, batch_size: int) -> dict[str, float]:
    if not pairs:
        return {"loss": 0.0, "accuracy": 0.0, "mean_margin": 0.0}
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    margins: list[float] = []
    with torch.inference_mode():
        for batch_rows in _batched(pairs, batch_size):
            batch = _batch_to_tensors(batch_rows, device)
            pos_score, neg_score = _pair_scores(model, batch)
            margin = pos_score - neg_score
            loss = (F.softplus(-margin) * batch["weight"]).mean()
            total_loss += float(loss.item()) * len(batch_rows)
            total_correct += int((margin > 0.0).sum().item())
            total += len(batch_rows)
            margins.extend(float(value) for value in margin.detach().cpu().tolist())
    return {
        "loss": total_loss / max(1, total),
        "accuracy": total_correct / max(1, total),
        "mean_margin": mean(margins) if margins else 0.0,
    }


def _distill_loss(model: Any, base_model: Any, batch: dict[str, Any]):
    pos_score, neg_score = _pair_scores(model, batch)
    with torch.no_grad():
        base_pos_score, base_neg_score = _pair_scores(base_model, batch)
    return 0.5 * (F.smooth_l1_loss(pos_score, base_pos_score) + F.smooth_l1_loss(neg_score, base_neg_score))


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune card/upgrade MLPs on non-combat branch rollout labels.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--branch-label-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--sources", default="card_reward,card_reward_skip")
    parser.add_argument("--include-fixed-prior", action="store_true")
    parser.add_argument("--min-branch-gap", type=float, default=1.0)
    parser.add_argument("--max-pairs-per-root", type=int, default=8)
    parser.add_argument("--weight-scale", type=float, default=10.0)
    parser.add_argument("--weight-max", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=0.0,
        help="Keep finetuned scores near the frozen base model on the same branch pairs.",
    )
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if torch is None:
        raise RuntimeError("torch is required")
    hard_records = _load_hard_records(args.input_root)
    labels = list(_iter_jsonl(args.branch_label_dir / "branch_labels.jsonl"))
    sources = {token.strip() for token in args.sources.split(",") if token.strip()}
    model, base_summary = load_card_reward_checkpoint(str(args.base_checkpoint), device=args.device)
    base_model, _ = load_card_reward_checkpoint(str(args.base_checkpoint), device=args.device)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    pairs = _make_pair_examples(
        labels=labels,
        hard_records=hard_records,
        sources=sources,
        min_branch_gap=float(args.min_branch_gap),
        max_pairs_per_root=int(args.max_pairs_per_root),
        weight_scale=float(args.weight_scale),
        weight_max=float(args.weight_max),
        base_summary=base_summary,
        include_fixed_prior=bool(args.include_fixed_prior),
    )
    train_pairs, valid_pairs = _split(pairs, float(args.valid_fraction), int(args.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-5)
    history = []
    best_state = None
    best_valid = None
    print(
        f"branch pairs={len(pairs)} train={len(train_pairs)} valid={len(valid_pairs)} "
        f"sources={sorted(sources)} base={args.base_checkpoint}",
        flush=True,
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        random.Random(int(args.seed) + epoch).shuffle(train_pairs)
        total_loss = 0.0
        total = 0
        for batch_rows in _batched(train_pairs, int(args.batch_size)):
            batch = _batch_to_tensors(batch_rows, args.device)
            pos_score, neg_score = _pair_scores(model, batch)
            branch_loss = (F.softplus(-(pos_score - neg_score)) * batch["weight"]).mean()
            if float(args.distill_weight) > 0.0:
                loss = branch_loss + float(args.distill_weight) * _distill_loss(model, base_model, batch)
            else:
                loss = branch_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_rows)
            total += len(batch_rows)
        train_metrics = _evaluate_pairs(model, train_pairs, args.device, int(args.batch_size))
        valid_metrics = _evaluate_pairs(model, valid_pairs, args.device, int(args.batch_size))
        entry = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total),
            "distill_weight": float(args.distill_weight),
            "train_pair_accuracy": train_metrics["accuracy"],
            "valid_pair_accuracy": valid_metrics["accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_mean_margin": valid_metrics["mean_margin"],
        }
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False, sort_keys=True), flush=True)
        current = (valid_metrics["accuracy"], -valid_metrics["loss"], valid_metrics["mean_margin"])
        if best_valid is None or current > best_valid:
            best_valid = current
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    summary = dict(base_summary or {})
    summary["branch_finetune"] = {
        "input_root": str(args.input_root),
        "branch_label_dir": str(args.branch_label_dir),
        "base_checkpoint": str(args.base_checkpoint),
        "sources": sorted(sources),
        "pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "valid_pairs": len(valid_pairs),
        "min_branch_gap": float(args.min_branch_gap),
        "max_pairs_per_root": int(args.max_pairs_per_root),
        "include_fixed_prior": bool(args.include_fixed_prior),
        "distill_weight": float(args.distill_weight),
        "history": history,
    }
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_card_reward_checkpoint(model, args.output_checkpoint, training_summary=summary)
    print(f"saved {args.output_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
