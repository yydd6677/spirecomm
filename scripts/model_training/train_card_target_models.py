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
import os
import random
from statistics import median
from pathlib import Path

import torch
import torch.nn.functional as F

from spirecomm.ai.card_reward_model import (
    CardRewardPolicyNetwork,
    normalize_token,
    option_scores_from_pairwise_batch,
    pairwise_batch_to_tensors,
    save_card_reward_checkpoint,
)
from scripts.model_training.train_card_reward_preference_from_runs import (
    STARTER_DECK,
    STARTER_RELIC,
    apply_floor_updates,
    make_scenario,
    parse_card_instance,
    standard_reward_from_choice,
)


CACHE_FORMAT_VERSION = 4
TASKS = ("upgrade", "purge")
DEFAULT_OUTPUTS = {
    "upgrade": "upgrade_target.pt",
    "purge": "purge_target.pt",
}
UPGRADE_CLASS_BALANCE_POWER = 1.0
UPGRADE_CLASS_BALANCE_MIN_WEIGHT = 0.0
UPGRADE_CLASS_BALANCE_MAX_WEIGHT = 1000000.0
UPGRADE_CLASS_BALANCE_MIN_COUNT = 20
UPGRADE_STARTER_BASIC_MAX_WEIGHT = 0.1
UPGRADE_STARTER_BASIC_KEYS = {"bash", "strike", "defend"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train card upgrade/purge target models from SlayTheData Ironclad runs."
    )
    parser.add_argument("--task", choices=("all",) + TASKS, default="all")
    parser.add_argument("--input-dir", default="/media/yydd/E0E24E0F6119A708/SlayTheData_ironclad_victory_a20")
    parser.add_argument("--output", default="", help="Output path for a single --task run.")
    parser.add_argument("--output-dir", default="/home/yydd/spirecomm/models")
    parser.add_argument("--cache-dir", default="/home/yydd/spirecomm/_cache/card_target_from_runs")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--file-progress-every", type=int, default=1000)
    parser.add_argument("--batch-progress-every", type=int, default=200)
    parser.add_argument("--tensor-cache-pairs-per-shard", type=int, default=200000)
    parser.add_argument("--max-negatives", type=int, default=12)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--upgrade-class-balance-power", type=float, default=UPGRADE_CLASS_BALANCE_POWER)
    parser.add_argument("--upgrade-class-balance-min-weight", type=float, default=UPGRADE_CLASS_BALANCE_MIN_WEIGHT)
    parser.add_argument("--upgrade-class-balance-max-weight", type=float, default=UPGRADE_CLASS_BALANCE_MAX_WEIGHT)
    parser.add_argument("--upgrade-class-balance-min-count", type=int, default=UPGRADE_CLASS_BALANCE_MIN_COUNT)
    parser.add_argument("--upgrade-starter-basic-max-weight", type=float, default=UPGRADE_STARTER_BASIC_MAX_WEIGHT)
    return parser.parse_args()


def iter_run_events(input_dir, max_files=0, file_progress_every=1000):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError("Input dir not found: {}".format(input_path))
    files = sorted(input_path.glob("*.json"))
    if not files:
        raise FileNotFoundError("No .json files found in {}".format(input_path))
    if max_files > 0:
        files = files[:max_files]
    for file_index, path in enumerate(files, 1):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        items = data if isinstance(data, list) else [data]
        for item in items:
            event = item.get("event") if isinstance(item, dict) and "event" in item else item
            if isinstance(event, dict):
                yield event
        if file_progress_every > 0 and (file_index % file_progress_every == 0 or file_index == len(files)):
            print("Loaded files: {}/{}".format(file_index, len(files)), flush=True)


def add_card_choices_at_floor(run, floor, deck):
    for choice in run.get("card_choices") or []:
        if int(choice.get("floor", -1) or -1) != floor:
            continue
        parsed = standard_reward_from_choice(choice)
        if parsed is None or parsed.get("skip_choice"):
            continue
        chosen_card = parsed.get("chosen_card")
        if chosen_card is not None:
            deck.append(dict(chosen_card))


def card_match_keys(card_or_name):
    if isinstance(card_or_name, dict):
        values = [card_or_name.get("card_id"), card_or_name.get("name")]
    else:
        parsed = parse_card_instance(card_or_name)
        values = [card_or_name, parsed.get("card_id"), parsed.get("name")]
    keys = {normalize_token(value) for value in values if value}
    expanded = set(keys)
    for key in keys:
        if key in {"striker", "defendr"}:
            expanded.add(key[:-1])
    return {key for key in expanded if key}


def card_balance_key(card_or_name):
    keys = card_match_keys(card_or_name)
    return min(keys) if keys else normalize_token(card_or_name)


def card_type(card):
    return str(card.get("type", "")).upper()


def is_upgrade_candidate(card):
    if card_type(card) in {"CURSE", "STATUS"}:
        return False
    key = next(iter(card_match_keys(card)), "")
    if key == "searingblow":
        return True
    return int(card.get("upgrades", 0) or 0) <= 0


def find_card_index(deck, target_name, require_upgradeable=False):
    target_keys = card_match_keys(target_name)
    for index, card in enumerate(deck):
        if require_upgradeable and not is_upgrade_candidate(card):
            continue
        if card_match_keys(card) & target_keys:
            return index
    if require_upgradeable:
        for index, card in enumerate(deck):
            if card_match_keys(card) & target_keys:
                return index
    return None


def deterministic_negatives(cards, target_index, floor, max_negatives):
    candidates = [index for index in range(len(cards)) if index != target_index]
    if not candidates:
        return []
    if max_negatives <= 0 or len(candidates) <= max_negatives:
        return candidates
    offset = (floor + target_index * 7 + len(cards)) % len(candidates)
    rotated = candidates[offset:] + candidates[:offset]
    return rotated[:max_negatives]


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def upgrade_target_contexts(run, deck, relics, floor, reconstruction_issues):
    for entry in run.get("campfire_choices") or []:
        if int(entry.get("floor", -1) or -1) != floor:
            continue
        if str(entry.get("key", "")).upper() != "SMITH":
            continue
        target_name = entry.get("data")
        if not target_name:
            continue
        target_index = find_card_index(deck, target_name, require_upgradeable=True)
        if target_index is None:
            continue
        candidates = [card for card in deck if is_upgrade_candidate(card)]
        chosen_keys = card_match_keys(deck[target_index])
        candidate_target_index = None
        for index, card in enumerate(candidates):
            if card_match_keys(card) & chosen_keys:
                candidate_target_index = index
                break
        if candidate_target_index is None:
            candidates.append(deck[target_index])
            candidate_target_index = len(candidates) - 1
        if len(candidates) < 2:
            continue
        state = make_scenario(run, deck, relics, floor, [], reconstruction_issues)
        yield candidates, candidate_target_index, state


def target_weight(task, floor, pos_card=None, upgrade_class_weights=None):
    if task == "upgrade" and pos_card is not None and upgrade_class_weights:
        return float(upgrade_class_weights.get(card_balance_key(pos_card), 1.0))
    return 1.0


def upgrade_pairs(run, deck, relics, floor, reconstruction_issues, max_negatives, upgrade_class_weights=None):
    for candidates, candidate_target_index, state in upgrade_target_contexts(run, deck, relics, floor, reconstruction_issues):
        pos_card = candidates[candidate_target_index]
        weight = target_weight("upgrade", floor, pos_card=pos_card, upgrade_class_weights=upgrade_class_weights)
        for rejected_index in deterministic_negatives(candidates, candidate_target_index, floor, max_negatives):
            yield {
                "kind": "upgrade",
                "scenario": state,
                "pos_card": pos_card,
                "neg_card": candidates[rejected_index],
                "pos_is_skip": False,
                "neg_is_skip": False,
                "weight": weight,
            }


def purge_pairs(run, deck, relics, floor, reconstruction_issues, max_negatives):
    purge_floors = list(run.get("items_purged_floors") or [])
    purged_cards = list(run.get("items_purged") or [])
    for purge_floor, purged_name in zip(purge_floors, purged_cards):
        if int(purge_floor or -1) != floor:
            continue
        target_index = find_card_index(deck, purged_name, require_upgradeable=False)
        if target_index is None or len(deck) < 2:
            continue
        state = make_scenario(run, deck, relics, floor, [], reconstruction_issues)
        for rejected_index in deterministic_negatives(deck, target_index, floor, max_negatives):
            yield {
                "kind": "purge",
                "scenario": state,
                "pos_card": deck[target_index],
                "neg_card": deck[rejected_index],
                "pos_is_skip": False,
                "neg_is_skip": False,
                "weight": target_weight("purge", floor),
            }


def build_pairs_from_run(run, task, max_negatives, upgrade_class_weights=None):
    deck = [parse_card_instance(name) for name in STARTER_DECK]
    relics = [STARTER_RELIC]
    reconstruction_issues = 0
    max_floor = int(run.get("floor_reached", 0) or 0)
    if max_floor <= 0:
        return

    for floor in range(1, max_floor + 1):
        if task == "upgrade":
            yield from upgrade_pairs(run, deck, relics, floor, reconstruction_issues, max_negatives, upgrade_class_weights=upgrade_class_weights)
        elif task == "purge":
            yield from purge_pairs(run, deck, relics, floor, reconstruction_issues, max_negatives)

        reconstruction_issues += apply_floor_updates(floor, run, deck, relics)
        add_card_choices_at_floor(run, floor, deck)


def task_cache_dir(base_cache_dir, task):
    return Path(base_cache_dir) / task


def upgrade_balance_config(args):
    return {
        "power": float(args.upgrade_class_balance_power),
        "min_weight": float(args.upgrade_class_balance_min_weight),
        "max_weight": float(args.upgrade_class_balance_max_weight),
        "min_count": int(args.upgrade_class_balance_min_count),
        "starter_basic_max_weight": float(args.upgrade_starter_basic_max_weight),
    }


def cache_config(args, task):
    config = {
        "max_negatives": int(args.max_negatives),
        "seed": int(args.seed),
        "valid_fraction": float(args.valid_fraction),
        "max_files": int(args.max_files),
        "max_runs": int(args.max_runs),
    }
    if task == "upgrade":
        config["upgrade_class_balance"] = upgrade_balance_config(args)
    return config


def cache_config_matches(meta, args, task):
    def _int_meta(name, default=-1):
        value = meta.get(name, default)
        return default if value is None else int(value)

    def _float_meta(name, default=-1.0):
        value = meta.get(name, default)
        return default if value is None else float(value)

    if _int_meta("max_negatives") != int(args.max_negatives):
        return False
    if _int_meta("seed") != int(args.seed):
        return False
    if _float_meta("valid_fraction") != float(args.valid_fraction):
        return False
    if _int_meta("max_files", 0) != int(args.max_files):
        return False
    if _int_meta("max_runs", 0) != int(args.max_runs):
        return False
    if task == "upgrade":
        return meta.get("upgrade_class_balance") == upgrade_balance_config(args)
    return True


def collect_upgrade_positive_counts(args):
    counts = {}
    runs = 0
    for run in iter_run_events(args.input_dir, max_files=args.max_files, file_progress_every=args.file_progress_every):
        runs += 1
        if args.max_runs > 0 and runs > args.max_runs:
            break
        deck = [parse_card_instance(name) for name in STARTER_DECK]
        relics = [STARTER_RELIC]
        reconstruction_issues = 0
        max_floor = int(run.get("floor_reached", 0) or 0)
        if max_floor <= 0:
            continue
        for floor in range(1, max_floor + 1):
            for candidates, candidate_target_index, _state in upgrade_target_contexts(run, deck, relics, floor, reconstruction_issues):
                key = card_balance_key(candidates[candidate_target_index])
                if key:
                    counts[key] = counts.get(key, 0) + 1
            reconstruction_issues += apply_floor_updates(floor, run, deck, relics)
            add_card_choices_at_floor(run, floor, deck)
    return counts


def build_upgrade_class_weights(counts, args):
    if not counts:
        return {}, {}
    config = upgrade_balance_config(args)
    min_count = max(1, int(config["min_count"]))
    supported_counts = [int(value) for value in counts.values() if int(value) >= min_count]
    if not supported_counts:
        supported_counts = [max(min_count, int(value)) for value in counts.values()]
    reference = float(median(supported_counts))
    class_weights = {}
    for key, value in counts.items():
        effective_count = max(min_count, int(value))
        raw = (reference / float(effective_count)) ** float(config["power"])
        weight = clamp(raw, float(config["min_weight"]), float(config["max_weight"]))
        if key in UPGRADE_STARTER_BASIC_KEYS:
            weight = min(weight, float(config["starter_basic_max_weight"]))
        class_weights[key] = weight
    summary = {
        "reference_count": reference,
        "class_count": len(class_weights),
        "top_positive_counts": sorted(counts.items(), key=lambda item: item[1], reverse=True)[:25],
        "lowest_weights": sorted(class_weights.items(), key=lambda item: item[1])[:25],
        "highest_weights": sorted(class_weights.items(), key=lambda item: item[1], reverse=True)[:25],
    }
    return class_weights, summary


def ensure_cache(args, task):
    cache_dir = task_cache_dir(args.cache_dir, task)
    meta_path = cache_dir / "meta.json"
    if not args.rebuild_cache and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if (
            int(meta.get("cache_format_version", 0) or 0) == CACHE_FORMAT_VERSION
            and meta.get("task") == task
            and cache_config_matches(meta, args, task)
        ):
            train_shards = [cache_dir / name for name in meta.get("train_shards", [])]
            valid_shards = [cache_dir / name for name in meta.get("valid_shards", [])]
            if train_shards and all(path.exists() for path in train_shards) and all(path.exists() for path in valid_shards):
                print("Using {} cache from {}".format(task, cache_dir), flush=True)
                return train_shards, valid_shards, meta

    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train_shards = []
    valid_shards = []
    train_shard_counts = []
    valid_shard_counts = []
    train_buffer = []
    valid_buffer = []
    train_pairs = 0
    valid_pairs = 0
    runs = 0
    upgrade_class_weights = None
    upgrade_class_balance_summary = None
    if task == "upgrade":
        print("Collecting upgrade positive counts for class-balanced weights ...", flush=True)
        upgrade_positive_counts = collect_upgrade_positive_counts(args)
        upgrade_class_weights, upgrade_class_balance_summary = build_upgrade_class_weights(upgrade_positive_counts, args)
        bash_weight = upgrade_class_weights.get("bash")
        print(
            "Upgrade class weights ready classes={} reference_count={} bash_weight={}".format(
                len(upgrade_class_weights),
                upgrade_class_balance_summary.get("reference_count") if upgrade_class_balance_summary else None,
                "{:.4f}".format(bash_weight) if bash_weight is not None else "n/a",
            ),
            flush=True,
        )

    def flush_buffer(split_name, buffer, shard_names, shard_counts):
        if not buffer:
            return
        shard_name = "{}_pairs_{:05d}.pt".format(split_name, len(shard_names))
        shard_path = cache_dir / shard_name
        tensors = pairwise_batch_to_tensors(buffer, device="cpu")
        shard_count = int(tensors["weight"].shape[0])
        payload = {
            "state": tensors["state"].cpu(),
            "pos_candidate": tensors["pos_candidate"].cpu(),
            "neg_candidate": tensors["neg_candidate"].cpu(),
            "pos_is_skip": tensors["pos_is_skip"].cpu(),
            "neg_is_skip": tensors["neg_is_skip"].cpu(),
            "weight": tensors["weight"].cpu(),
            "count": shard_count,
        }
        torch.save(payload, shard_path)
        shard_names.append(shard_name)
        shard_counts.append(shard_count)
        buffer.clear()

    print("Building {} tensor cache at {} ...".format(task, cache_dir), flush=True)
    for run in iter_run_events(args.input_dir, max_files=args.max_files, file_progress_every=args.file_progress_every):
        runs += 1
        if args.max_runs > 0 and runs > args.max_runs:
            break
        pairs = list(build_pairs_from_run(run, task, args.max_negatives, upgrade_class_weights=upgrade_class_weights))
        if not pairs:
            continue
        if rng.random() < args.valid_fraction:
            valid_pairs += len(pairs)
            valid_buffer.extend(pairs)
            if len(valid_buffer) >= args.tensor_cache_pairs_per_shard:
                flush_buffer("valid", valid_buffer, valid_shards, valid_shard_counts)
        else:
            train_pairs += len(pairs)
            train_buffer.extend(pairs)
            if len(train_buffer) >= args.tensor_cache_pairs_per_shard:
                flush_buffer("train", train_buffer, train_shards, train_shard_counts)
        if args.file_progress_every > 0 and runs % args.file_progress_every == 0:
            print(
                "{} cache progress runs={} train_pairs={} valid_pairs={} train_shards={} valid_shards={}".format(
                    task, runs, train_pairs, valid_pairs, len(train_shards), len(valid_shards)
                ),
                flush=True,
            )

    flush_buffer("train", train_buffer, train_shards, train_shard_counts)
    flush_buffer("valid", valid_buffer, valid_shards, valid_shard_counts)
    if train_pairs <= 0:
        raise RuntimeError("No training pairs were built for task {}".format(task))

    meta = {
        "source": "slay_the_data_card_target_pairwise",
        "cache_format_version": CACHE_FORMAT_VERSION,
        "task": task,
        "input_dir": str(args.input_dir),
        "train_pairs": train_pairs,
        "valid_pairs": valid_pairs,
        "train_shards": train_shards,
        "valid_shards": valid_shards,
        "train_shard_counts": train_shard_counts,
        "valid_shard_counts": valid_shard_counts,
        **cache_config(args, task),
    }
    if task == "upgrade":
        meta["upgrade_class_balance_summary"] = upgrade_class_balance_summary
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print("{} cache ready train_pairs={} valid_pairs={}".format(task, train_pairs, valid_pairs), flush=True)
    return [cache_dir / name for name in train_shards], [cache_dir / name for name in valid_shards], meta


def shard_batch_to_device(shard, indices, device):
    return {
        "state": shard["state"][indices].to(device=device, non_blocking=True),
        "pos_candidate": shard["pos_candidate"][indices].to(device=device, non_blocking=True),
        "neg_candidate": shard["neg_candidate"][indices].to(device=device, non_blocking=True),
        "pos_is_skip": shard["pos_is_skip"][indices].to(device=device, non_blocking=True),
        "neg_is_skip": shard["neg_is_skip"][indices].to(device=device, non_blocking=True),
        "weight": shard["weight"][indices].to(device=device, non_blocking=True),
    }


def total_batches_for_shards(shard_counts, batch_size):
    return max(1, sum(max(1, math.ceil(int(count) / float(batch_size))) for count in (shard_counts or []) if int(count) > 0))


def stream_pairwise_batches(shard_paths, batch_size, device, seed=0, shuffle=True):
    shard_paths = list(shard_paths)
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(shard_paths)
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        count = int(shard.get("count", int(shard["weight"].shape[0])))
        if count <= 0:
            continue
        order = torch.randperm(count) if shuffle else torch.arange(count)
        for start in range(0, count, batch_size):
            indices = order[start:start + batch_size]
            yield shard_batch_to_device(shard, indices, device), int(indices.shape[0])


def evaluate_pairwise(model, shard_paths, device, batch_size=512):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch, batch_len in stream_pairwise_batches(shard_paths, batch_size=batch_size, device=device, shuffle=False):
            pos, neg = option_scores_from_pairwise_batch(model, batch)
            losses = F.softplus(-(pos - neg)) * batch["weight"]
            total_loss += float(losses.sum().item())
            total_correct += int((pos > neg).sum().item())
            total += batch_len
    return {
        "loss": total_loss / float(total or 1),
        "accuracy": total_correct / float(total or 1),
    }


def train_task(task, args):
    train_shards, valid_shards, meta = ensure_cache(args, task)
    train_pairs = int(meta.get("train_pairs", 0) or 0)
    valid_pairs = int(meta.get("valid_pairs", 0) or 0)
    train_shard_counts = meta.get("train_shard_counts", [])
    device = args.device
    model = CardRewardPolicyNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output_path = Path(args.output) if args.output and args.task != "all" else Path(args.output_dir) / DEFAULT_OUTPUTS[task]
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir and args.task != "all" else output_path.with_suffix("").with_name(output_path.stem + "_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_state = None
    best_valid = None

    for epoch in range(1, args.epochs + 1):
        print("{} epoch {}/{} start train_pairs={} valid_pairs={}".format(task, epoch, args.epochs, train_pairs, valid_pairs), flush=True)
        model.train()
        total_loss = 0.0
        total = 0
        total_batches = total_batches_for_shards(train_shard_counts, args.batch_size)
        for batch_index, (batch, batch_len) in enumerate(
            stream_pairwise_batches(train_shards, batch_size=args.batch_size, device=device, seed=args.seed + epoch, shuffle=True),
            1,
        ):
            pos, neg = option_scores_from_pairwise_batch(model, batch)
            loss = (F.softplus(-(pos - neg)) * batch["weight"]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_len
            total += batch_len
            if args.batch_progress_every > 0 and (batch_index % args.batch_progress_every == 0 or batch_index == total_batches):
                print(
                    "{} epoch {}/{} batch {}/{} mean_loss={:.4f}".format(
                        task, epoch, args.epochs, batch_index, total_batches, total_loss / float(total or 1)
                    ),
                    flush=True,
                )

        eval_batch_size = max(64, min(4096, args.batch_size * 8))
        train_metrics = evaluate_pairwise(model, train_shards, device, batch_size=eval_batch_size)
        valid_metrics = evaluate_pairwise(model, valid_shards, device, batch_size=eval_batch_size) if valid_pairs > 0 else {"loss": 0.0, "accuracy": 0.0}
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / float(total or 1),
                "train_pair_accuracy": train_metrics["accuracy"],
                "valid_loss": valid_metrics["loss"],
                "valid_pair_accuracy": valid_metrics["accuracy"],
            }
        )
        print(
            "{} epoch {}/{} done train_loss={:.4f} train_acc={:.4f} valid_loss={:.4f} valid_acc={:.4f}".format(
                task,
                epoch,
                args.epochs,
                total_loss / float(total or 1),
                train_metrics["accuracy"],
                valid_metrics["loss"],
                valid_metrics["accuracy"],
            ),
            flush=True,
        )
        summary = {
            "source": "slay_the_data_card_target_pairwise",
            "task": task,
            "train_pairs": train_pairs,
            "valid_pairs": valid_pairs,
            "history": list(history),
            "last_epoch": epoch,
        }
        save_card_reward_checkpoint(model, str(checkpoint_dir / "latest.pt"), training_summary=summary)
        save_card_reward_checkpoint(model, str(checkpoint_dir / "epoch_{:03d}.pt".format(epoch)), training_summary=summary)
        current_valid = (valid_metrics["accuracy"], -valid_metrics["loss"])
        if best_valid is None or current_valid > best_valid:
            best_valid = current_valid
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            save_card_reward_checkpoint(model, str(checkpoint_dir / "best.pt"), training_summary=summary)

    if best_state is not None:
        model.load_state_dict(best_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_card_reward_checkpoint(
        model,
        str(output_path),
        training_summary={
            "source": "slay_the_data_card_target_pairwise",
            "task": task,
            "train_pairs": train_pairs,
            "valid_pairs": valid_pairs,
            "cache_dir": str(task_cache_dir(args.cache_dir, task)),
            "checkpoint_dir": str(checkpoint_dir),
            "history": history,
        },
    )
    print("Saved {} target model to {}".format(task, output_path.resolve()), flush=True)
    print("Epoch checkpoints saved to {}".format(checkpoint_dir.resolve()), flush=True)
    return str(output_path)


def main():
    args = parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    outputs = {}
    for task in tasks:
        outputs[task] = train_task(task, args)
    print("Suggested runtime env:", flush=True)
    for task, output in outputs.items():
        if task == "upgrade":
            print("  SPIRECOMM_UPGRADE_TARGET_MODEL_PATH={}".format(os.path.abspath(output)))
        elif task == "purge":
            print("  SPIRECOMM_PURGE_TARGET_MODEL_PATH={}".format(os.path.abspath(output)))


if __name__ == "__main__":
    main()
