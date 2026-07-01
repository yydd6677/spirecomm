#!/usr/bin/env python3
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

import torch
import torch.nn.functional as F

from spirecomm.ai.card_reward_model import (
    CANDIDATE_DIM,
    STATE_DIM,
    CardRewardPolicyNetwork,
    build_state_vector,
    candidate_vector,
    save_card_reward_checkpoint,
)
from scripts.model_training.train_card_target_models import (
    STARTER_DECK,
    STARTER_RELIC,
    add_card_choices_at_floor,
    apply_floor_updates,
    card_balance_key,
    iter_run_events,
    parse_card_instance,
    upgrade_target_contexts,
)


CACHE_FORMAT_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train upgrade target model with copy-rate logit prior and listwise softmax."
    )
    parser.add_argument("--input-dir", default="/media/yydd/E0E24E0F6119A708/SlayTheData_ironclad_victory_a20")
    parser.add_argument("--cache-dir", default="/home/yydd/spirecomm/_cache/upgrade_target_rate_prior")
    parser.add_argument("--output", default="/home/yydd/spirecomm/models/upgrade_target_rate_prior_full.pt")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--file-progress-every", type=int, default=5000)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--batch-progress-every", type=int, default=100)
    parser.add_argument("--prior-eps", type=float, default=1e-6)
    parser.add_argument("--delta-l2", type=float, default=0.05)
    return parser.parse_args()


def _iter_reconstructed_runs(args):
    runs = 0
    for run in iter_run_events(args.input_dir, max_files=args.max_files, file_progress_every=args.file_progress_every):
        runs += 1
        if args.max_runs > 0 and runs > args.max_runs:
            break
        yield run


def collect_copy_rate_stats(args):
    selected = {}
    copy_den = {}
    contexts = 0
    runs = 0
    for run in _iter_reconstructed_runs(args):
        runs += 1
        deck = [parse_card_instance(name) for name in STARTER_DECK]
        relics = [STARTER_RELIC]
        reconstruction_issues = 0
        max_floor = int(run.get("floor_reached", 0) or 0)
        if max_floor <= 0:
            continue
        for floor in range(1, max_floor + 1):
            for candidates, target_index, _state in upgrade_target_contexts(run, deck, relics, floor, reconstruction_issues):
                contexts += 1
                target_key = card_balance_key(candidates[target_index])
                if target_key:
                    selected[target_key] = selected.get(target_key, 0) + 1
                for card in candidates:
                    key = card_balance_key(card)
                    if key:
                        copy_den[key] = copy_den.get(key, 0) + 1
            reconstruction_issues += apply_floor_updates(floor, run, deck, relics)
            add_card_choices_at_floor(run, floor, deck)
    rates = {
        key: float(selected.get(key, 0)) / float(den)
        for key, den in copy_den.items()
        if den > 0
    }
    summary = {
        "runs": runs,
        "smith_contexts": contexts,
        "class_count": len(rates),
        "top_selected": sorted(selected.items(), key=lambda item: item[1], reverse=True)[:30],
        "top_copy_rates": sorted(rates.items(), key=lambda item: item[1], reverse=True)[:30],
        "bottom_copy_rates": sorted(rates.items(), key=lambda item: item[1])[:30],
        "selected": selected,
        "copy_den": copy_den,
    }
    return rates, summary


def _example_from_context(candidates, target_index, state, rate_prior, prior_eps):
    candidate_vectors = [candidate_vector(card, state) for card in candidates]
    candidate_keys = [card_balance_key(card) for card in candidates]
    target_key = candidate_keys[target_index]
    target_mask = [1.0 if key == target_key else 0.0 for key in candidate_keys]
    prior_logits = [math.log(float(rate_prior.get(key, 0.0)) + prior_eps) for key in candidate_keys]
    return {
        "state": build_state_vector(state),
        "candidates": candidate_vectors,
        "target_mask": target_mask,
        "prior_logits": prior_logits,
        "candidate_keys": candidate_keys,
        "target_key": target_key,
    }


def _pad_examples(examples):
    max_candidates = max(len(example["candidates"]) for example in examples)
    states = []
    candidates = []
    target_masks = []
    valid_masks = []
    prior_logits = []
    target_keys = []
    for example in examples:
        count = len(example["candidates"])
        states.append(example["state"])
        candidates.append(example["candidates"] + [[0.0] * CANDIDATE_DIM for _ in range(max_candidates - count)])
        target_masks.append(example["target_mask"] + [0.0 for _ in range(max_candidates - count)])
        valid_masks.append([1.0 for _ in range(count)] + [0.0 for _ in range(max_candidates - count)])
        prior_logits.append(example["prior_logits"] + [0.0 for _ in range(max_candidates - count)])
        target_keys.append(example["target_key"])
    return {
        "state": torch.tensor(states, dtype=torch.float32),
        "candidate": torch.tensor(candidates, dtype=torch.float32),
        "target_mask": torch.tensor(target_masks, dtype=torch.float32),
        "valid_mask": torch.tensor(valid_masks, dtype=torch.float32),
        "prior_logits": torch.tensor(prior_logits, dtype=torch.float32),
        "target_keys": target_keys,
        "count": len(examples),
    }


def build_cache(args, rate_prior, rate_summary):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train_shards = []
    valid_shards = []
    train_counts = []
    valid_counts = []
    train_buffer = []
    valid_buffer = []
    train_examples = 0
    valid_examples = 0

    def flush(split, buffer, shard_names, shard_counts):
        if not buffer:
            return
        shard_name = f"{split}_listwise_{len(shard_names):05d}.pt"
        payload = _pad_examples(buffer)
        torch.save(payload, cache_dir / shard_name)
        shard_names.append(shard_name)
        shard_counts.append(int(payload["count"]))
        buffer.clear()

    print(f"Building listwise cache at {cache_dir} ...", flush=True)
    runs = 0
    for run in _iter_reconstructed_runs(args):
        runs += 1
        deck = [parse_card_instance(name) for name in STARTER_DECK]
        relics = [STARTER_RELIC]
        reconstruction_issues = 0
        max_floor = int(run.get("floor_reached", 0) or 0)
        if max_floor <= 0:
            continue
        examples = []
        for floor in range(1, max_floor + 1):
            for candidates, target_index, state in upgrade_target_contexts(run, deck, relics, floor, reconstruction_issues):
                examples.append(_example_from_context(candidates, target_index, state, rate_prior, args.prior_eps))
            reconstruction_issues += apply_floor_updates(floor, run, deck, relics)
            add_card_choices_at_floor(run, floor, deck)
        if not examples:
            continue
        if rng.random() < args.valid_fraction:
            valid_buffer.extend(examples)
            valid_examples += len(examples)
            if len(valid_buffer) >= args.shard_size:
                flush("valid", valid_buffer, valid_shards, valid_counts)
        else:
            train_buffer.extend(examples)
            train_examples += len(examples)
            if len(train_buffer) >= args.shard_size:
                flush("train", train_buffer, train_shards, train_counts)
        if args.file_progress_every > 0 and runs % args.file_progress_every == 0:
            print(
                f"cache progress runs={runs} train_examples={train_examples} valid_examples={valid_examples} "
                f"train_shards={len(train_shards)} valid_shards={len(valid_shards)}",
                flush=True,
            )

    flush("train", train_buffer, train_shards, train_counts)
    flush("valid", valid_buffer, valid_shards, valid_counts)
    if train_examples <= 0:
        raise RuntimeError("No listwise upgrade examples were built")
    meta = {
        "source": "slay_the_data_upgrade_target_listwise_copy_rate_prior",
        "cache_format_version": CACHE_FORMAT_VERSION,
        "input_dir": str(args.input_dir),
        "train_examples": train_examples,
        "valid_examples": valid_examples,
        "train_shards": train_shards,
        "valid_shards": valid_shards,
        "train_shard_counts": train_counts,
        "valid_shard_counts": valid_counts,
        "valid_fraction": float(args.valid_fraction),
        "seed": int(args.seed),
        "max_files": int(args.max_files),
        "max_runs": int(args.max_runs),
        "prior_eps": float(args.prior_eps),
        "rate_prior_kind": "copy_rate",
        "rate_summary": rate_summary,
        "rate_prior": rate_prior,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"listwise cache ready train_examples={train_examples} valid_examples={valid_examples}", flush=True)
    return meta


def cache_matches(meta, args):
    return (
        int(meta.get("cache_format_version", 0) or 0) == CACHE_FORMAT_VERSION
        and meta.get("source") == "slay_the_data_upgrade_target_listwise_copy_rate_prior"
        and float(meta.get("valid_fraction", -1.0)) == float(args.valid_fraction)
        and int(meta.get("seed", -1)) == int(args.seed)
        and int(meta.get("max_files", 0)) == int(args.max_files)
        and int(meta.get("max_runs", 0)) == int(args.max_runs)
        and float(meta.get("prior_eps", -1.0)) == float(args.prior_eps)
        and meta.get("rate_prior_kind") == "copy_rate"
    )


def ensure_cache(args):
    cache_dir = Path(args.cache_dir)
    meta_path = cache_dir / "meta.json"
    if not args.rebuild_cache and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if cache_matches(meta, args):
            train_ok = all((cache_dir / name).exists() for name in meta.get("train_shards", []))
            valid_ok = all((cache_dir / name).exists() for name in meta.get("valid_shards", []))
            if train_ok and valid_ok:
                print(f"Using listwise cache from {cache_dir}", flush=True)
                return meta
    print("Collecting copy-rate upgrade priors ...", flush=True)
    rate_prior, rate_summary = collect_copy_rate_stats(args)
    print(
        "copy-rate priors ready classes={} smith_contexts={} trip={} bash={} defend={} strike={}".format(
            len(rate_prior),
            rate_summary.get("smith_contexts"),
            "{:.6f}".format(rate_prior.get("trip", 0.0)),
            "{:.6f}".format(rate_prior.get("bash", 0.0)),
            "{:.6f}".format(rate_prior.get("defend", 0.0)),
            "{:.6f}".format(rate_prior.get("strike", 0.0)),
        ),
        flush=True,
    )
    return build_cache(args, rate_prior, rate_summary)


def shard_batch_to_device(shard, indices, device):
    return {
        "state": shard["state"][indices].to(device=device, non_blocking=True),
        "candidate": shard["candidate"][indices].to(device=device, non_blocking=True),
        "target_mask": shard["target_mask"][indices].to(device=device, non_blocking=True),
        "valid_mask": shard["valid_mask"][indices].to(device=device, non_blocking=True),
        "prior_logits": shard["prior_logits"][indices].to(device=device, non_blocking=True),
    }


def stream_batches(cache_dir, shard_names, batch_size, device, seed=0, shuffle=True):
    shard_names = list(shard_names)
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(shard_names)
    for shard_name in shard_names:
        shard = torch.load(Path(cache_dir) / shard_name, map_location="cpu")
        count = int(shard.get("count", int(shard["state"].shape[0])))
        if count <= 0:
            continue
        order = torch.randperm(count) if shuffle else torch.arange(count)
        for start in range(0, count, batch_size):
            indices = order[start:start + batch_size]
            yield shard_batch_to_device(shard, indices, device), int(indices.shape[0])


def listwise_loss_and_metrics(model, batch, delta_l2):
    state_hidden = model.encode_state(batch["state"])
    delta = model.score_candidate_with_hidden(state_hidden, batch["candidate"])
    valid = batch["valid_mask"] > 0.5
    target = batch["target_mask"] > 0.5
    logits = delta + batch["prior_logits"]
    logits = logits.masked_fill(~valid, -1.0e9)
    log_probs = F.log_softmax(logits, dim=1)
    target_log_prob = torch.logsumexp(log_probs.masked_fill(~target, -1.0e9), dim=1)
    ce = -target_log_prob.mean()
    delta_reg = ((delta ** 2) * batch["valid_mask"]).sum() / batch["valid_mask"].sum().clamp_min(1.0)
    loss = ce + float(delta_l2) * delta_reg
    predictions = torch.argmax(logits, dim=1)
    correct = target.gather(1, predictions.unsqueeze(1)).squeeze(1)
    return loss, ce.detach(), delta_reg.detach(), int(correct.sum().item()), int(correct.numel())


def evaluate(model, meta, args, device):
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_delta_reg = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch, batch_len in stream_batches(args.cache_dir, meta.get("valid_shards", []), args.batch_size, device, shuffle=False):
            loss, ce, delta_reg, correct, count = listwise_loss_and_metrics(model, batch, args.delta_l2)
            total_loss += float(loss.item()) * batch_len
            total_ce += float(ce.item()) * batch_len
            total_delta_reg += float(delta_reg.item()) * batch_len
            total_correct += correct
            total += count
    return {
        "loss": total_loss / float(total or 1),
        "ce": total_ce / float(total or 1),
        "delta_reg": total_delta_reg / float(total or 1),
        "accuracy": total_correct / float(total or 1),
    }


def train(args):
    meta = ensure_cache(args)
    device = args.device
    model = CardRewardPolicyNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output_path = Path(args.output)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_path.with_suffix("").with_name(output_path.stem + "_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_examples = int(meta.get("train_examples", 0) or 0)
    valid_examples = int(meta.get("valid_examples", 0) or 0)
    total_batches = max(1, sum(math.ceil(int(count) / float(args.batch_size)) for count in meta.get("train_shard_counts", []) if int(count) > 0))
    history = []
    best_state = None
    best_valid = None
    for epoch in range(1, args.epochs + 1):
        print(f"upgrade_rate_prior epoch {epoch}/{args.epochs} start train_examples={train_examples} valid_examples={valid_examples}", flush=True)
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_delta_reg = 0.0
        total_correct = 0
        total = 0
        for batch_index, (batch, batch_len) in enumerate(
            stream_batches(args.cache_dir, meta.get("train_shards", []), args.batch_size, device, seed=args.seed + epoch, shuffle=True),
            1,
        ):
            loss, ce, delta_reg, correct, count = listwise_loss_and_metrics(model, batch, args.delta_l2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_len
            total_ce += float(ce.item()) * batch_len
            total_delta_reg += float(delta_reg.item()) * batch_len
            total_correct += correct
            total += count
            if args.batch_progress_every > 0 and (batch_index % args.batch_progress_every == 0 or batch_index == total_batches):
                print(
                    "upgrade_rate_prior epoch {}/{} batch {}/{} loss={:.4f} ce={:.4f} acc={:.4f}".format(
                        epoch,
                        args.epochs,
                        batch_index,
                        total_batches,
                        total_loss / float(total or 1),
                        total_ce / float(total or 1),
                        total_correct / float(total or 1),
                    ),
                    flush=True,
                )
        train_metrics = {
            "loss": total_loss / float(total or 1),
            "ce": total_ce / float(total or 1),
            "delta_reg": total_delta_reg / float(total or 1),
            "accuracy": total_correct / float(total or 1),
        }
        valid_metrics = evaluate(model, meta, args, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce": train_metrics["ce"],
            "train_delta_reg": train_metrics["delta_reg"],
            "train_accuracy": train_metrics["accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_ce": valid_metrics["ce"],
            "valid_delta_reg": valid_metrics["delta_reg"],
            "valid_accuracy": valid_metrics["accuracy"],
        }
        history.append(row)
        print(
            "upgrade_rate_prior epoch {}/{} done train_loss={:.4f} train_acc={:.4f} valid_loss={:.4f} valid_acc={:.4f}".format(
                epoch,
                args.epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                valid_metrics["loss"],
                valid_metrics["accuracy"],
            ),
            flush=True,
        )
        summary = {
            "source": "slay_the_data_upgrade_target_listwise_copy_rate_prior",
            "task": "upgrade",
            "train_examples": train_examples,
            "valid_examples": valid_examples,
            "cache_dir": str(Path(args.cache_dir).resolve()),
            "history": list(history),
            "last_epoch": epoch,
            "upgrade_rate_prior": meta.get("rate_prior", {}),
            "upgrade_rate_prior_eps": float(args.prior_eps),
            "delta_l2": float(args.delta_l2),
            "rate_prior_kind": "copy_rate",
        }
        save_card_reward_checkpoint(model, str(checkpoint_dir / "latest.pt"), training_summary=summary)
        save_card_reward_checkpoint(model, str(checkpoint_dir / f"epoch_{epoch:03d}.pt"), training_summary=summary)
        current_valid = (valid_metrics["accuracy"], -valid_metrics["loss"])
        if best_valid is None or current_valid > best_valid:
            best_valid = current_valid
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            save_card_reward_checkpoint(model, str(checkpoint_dir / "best.pt"), training_summary=summary)
    if best_state is not None:
        model.load_state_dict(best_state)
    final_summary = {
        "source": "slay_the_data_upgrade_target_listwise_copy_rate_prior",
        "task": "upgrade",
        "train_examples": train_examples,
        "valid_examples": valid_examples,
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "history": history,
        "upgrade_rate_prior": meta.get("rate_prior", {}),
        "upgrade_rate_prior_eps": float(args.prior_eps),
        "delta_l2": float(args.delta_l2),
        "rate_prior_kind": "copy_rate",
        "rate_summary": meta.get("rate_summary", {}),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_card_reward_checkpoint(model, str(output_path), training_summary=final_summary)
    print(f"Saved upgrade rate-prior model to {output_path.resolve()}", flush=True)
    print(f"Epoch checkpoints saved to {checkpoint_dir.resolve()}", flush=True)
    print(f"Suggested runtime env:\n  SPIRECOMM_UPGRADE_TARGET_MODEL_PATH={output_path.resolve()}", flush=True)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
