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
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from spirecomm.ai.card_reward_model import normalize_token
from spirecomm.ai.run_choice_model import (
    RunChoicePolicyNetwork,
    canonical_event_model_label,
    option_scores_from_pairwise_batch,
    option_token,
    pairwise_batch_to_tensors,
    save_run_choice_checkpoint,
)
from scripts.model_training.train_card_reward_preference_from_runs import (
    RED_CARD_RARITY,
    STARTER_DECK,
    STARTER_RELIC,
    act_for_floor,
    apply_floor_updates,
    make_scenario,
    next_boss_floor,
    parse_card_instance,
    standard_reward_from_choice,
    value_on_floor,
)


CACHE_FORMAT_VERSION = 10
TASKS = ("boss_relic", "event", "campfire", "map", "shop", "potion")
DEFAULT_OUTPUTS = {
    "boss_relic": "boss_relic.pt",
    "event": "event_choice.pt",
    "campfire": "campfire.pt",
    "map": "map_choice.pt",
    "shop": "shop_choice.pt",
    "potion": "potion_use.pt",
}
MAP_SYMBOLS = ("M", "?", "E", "$", "R", "T")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train high-level run choice models from SlayTheData run summaries."
    )
    parser.add_argument("--task", choices=("all",) + TASKS, default="all")
    parser.add_argument("--input-dir", default="/media/yydd/E0E24E0F6119A708/SlayTheData_ironclad_victory_a20")
    parser.add_argument("--output", default="", help="Output path for a single --task run.")
    parser.add_argument("--output-dir", default="/home/yydd/spirecomm/models")
    parser.add_argument("--cache-dir", default="/home/yydd/spirecomm/_cache/run_choice_from_runs")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--file-progress-every", type=int, default=1000)
    parser.add_argument("--batch-progress-every", type=int, default=200)
    parser.add_argument("--tensor-cache-pairs-per-shard", type=int, default=200000)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--event-prior-delta", action="store_true", help="Train event choice as log-prior plus neural state delta.")
    parser.add_argument("--event-prior-weight", type=float, default=1.0)
    parser.add_argument("--event-prior-eps", type=float, default=1e-6)
    parser.add_argument("--event-class-balance-alpha", type=float, default=0.5)
    parser.add_argument("--event-class-balance-min", type=float, default=0.5)
    parser.add_argument("--event-class-balance-max", type=float, default=5.0)
    parser.add_argument("--shop-prior-delta", action="store_true", help="Train shop choice as log-prior plus neural state delta.")
    parser.add_argument("--shop-prior-weight", type=float, default=1.0)
    parser.add_argument("--shop-prior-eps", type=float, default=1e-6)
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


def choice_text(value):
    return str(value or "").strip()


def event_key(entry):
    return normalize_token(entry.get("event_id") or entry.get("event_name") or entry.get("event") or "")


def event_name(entry):
    return str(entry.get("event_id") or entry.get("event_name") or entry.get("event") or "")


def event_choice_label(entry):
    for key in ("player_choice", "choice", "key", "label", "text"):
        value = entry.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _canonical_event_choice(entry):
    name = event_name(entry) or event_key(entry)
    return canonical_event_model_label(name, event_choice_label(entry))


def collect_event_option_metadata(args):
    counts_by_event = defaultdict(Counter)
    run_count = 0
    print("Collecting event option vocabulary/prior ...", flush=True)
    for run in iter_run_events(args.input_dir, max_files=args.max_files, file_progress_every=args.file_progress_every):
        run_count += 1
        if args.max_runs > 0 and run_count > args.max_runs:
            break
        for entry in run.get("event_choices") or []:
            key = event_key(entry)
            label = _canonical_event_choice(entry)
            if key and label:
                counts_by_event[key][label] += 1

    eps = max(0.0, float(args.event_prior_eps))
    alpha = max(0.0, float(args.event_class_balance_alpha))
    min_weight = float(args.event_class_balance_min)
    max_weight = float(args.event_class_balance_max)
    vocab = {key: sorted(counter) for key, counter in counts_by_event.items() if len(counter) >= 2}
    log_prior_by_token = {}
    class_weight_by_token = {}
    count_by_token = {}
    for key, counter in counts_by_event.items():
        if len(counter) < 2:
            continue
        total = float(sum(counter.values()))
        option_count = float(len(counter))
        denominator = total + eps * option_count
        for label, count in counter.items():
            token = option_token("event", {"event_id": key, "event_name": key, "label": label})
            probability = (float(count) + eps) / denominator if denominator > 0 else 0.0
            log_prior_by_token[token] = math.log(max(probability, eps if eps > 0 else 1e-12))
            raw_weight = (total / max(float(count), 1.0)) ** alpha if alpha > 0.0 else 1.0
            class_weight_by_token[token] = max(min_weight, min(max_weight, raw_weight))
            count_by_token[token] = int(count)
    serializable = {
        "vocab": vocab,
        "event_option_log_prior_by_token": log_prior_by_token,
        "event_option_class_weight_by_token": class_weight_by_token,
        "event_option_count_by_token": count_by_token,
        "event_prior_eps": eps,
        "event_class_balance_alpha": alpha,
        "event_class_balance_min": min_weight,
        "event_class_balance_max": max_weight,
    }
    print(
        "Event vocab ready  events={}  options={}".format(
            len(vocab),
            sum(len(values) for values in vocab.values()),
        ),
        flush=True,
    )
    return serializable


def collect_event_option_vocab(args):
    return collect_event_option_metadata(args)["vocab"]


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


def boss_floor_for_index(index):
    return 17 if index == 0 else 34


def make_state(run, deck, relics, floor, reconstruction_issues):
    return make_scenario(run, deck, relics, floor, [], reconstruction_issues)


def make_campfire_state(run, deck, relics, floor, reconstruction_issues):
    state = make_state(run, deck, relics, floor, reconstruction_issues)
    pre_campfire_floor = max(1, int(floor) - 1)
    state["current_hp"] = value_on_floor(run.get("current_hp_per_floor"), pre_campfire_floor, default=0)
    state["max_hp"] = value_on_floor(run.get("max_hp_per_floor"), pre_campfire_floor, default=0)
    state["gold"] = value_on_floor(run.get("gold_per_floor"), pre_campfire_floor, default=0)
    return state


def campfire_smith_rest_weight(state, picked, rejected):
    picked = _canonical_campfire_choice(picked)
    rejected = _canonical_campfire_choice(rejected)
    if {picked, rejected} != {"REST", "SMITH"}:
        return 1.0

    try:
        current_hp = float(state.get("current_hp", 0) or 0)
        max_hp = float(state.get("max_hp", 0) or 0)
    except Exception:
        return 1.0
    if max_hp <= 0:
        return 1.0
    hp_ratio = current_hp / max_hp

    if hp_ratio <= 0.20:
        rest_weight, smith_weight = 12.0, 0.20
    elif hp_ratio <= 0.35:
        rest_weight, smith_weight = 7.0, 0.50
    elif hp_ratio <= 0.50:
        rest_weight, smith_weight = 4.0, 0.75
    elif hp_ratio <= 0.65:
        rest_weight, smith_weight = 2.0, 0.90
    elif hp_ratio <= 0.80:
        rest_weight, smith_weight = 1.5, 1.00
    else:
        rest_weight, smith_weight = 1.0, 1.00
    return rest_weight if picked == "REST" else smith_weight


def boss_relic_pairs(run, deck, relics, floor, reconstruction_issues):
    boss_relics = run.get("boss_relics") or []
    expected_index = 0 if floor == 17 else 1 if floor == 34 else None
    if expected_index is None or expected_index >= len(boss_relics):
        return
    entry = boss_relics[expected_index]
    picked = choice_text(entry.get("picked"))
    not_picked = [choice_text(value) for value in (entry.get("not_picked") or []) if choice_text(value)]
    if not picked or not not_picked:
        return
    state = make_state(run, deck, relics, floor, reconstruction_issues)
    pos = {"name": picked, "relic_id": picked}
    for rejected in not_picked:
        yield {
            "kind": "boss_relic",
            "state": state,
            "pos_choice": pos,
            "neg_choice": {"name": rejected, "relic_id": rejected},
            "weight": 1.0,
        }


def event_pairs(run, deck, relics, floor, reconstruction_issues, event_metadata):
    event_vocab = (event_metadata or {}).get("vocab") or {}
    log_prior_by_token = (event_metadata or {}).get("event_option_log_prior_by_token") or {}
    class_weight_by_token = (event_metadata or {}).get("event_option_class_weight_by_token") or {}
    for entry in run.get("event_choices") or []:
        if int(entry.get("floor", -1) or -1) != floor:
            continue
        key = event_key(entry)
        picked = _canonical_event_choice(entry)
        if not key or not picked:
            continue
        options = list(event_vocab.get(key) or [])
        if picked not in options:
            options.append(picked)
        if len(options) < 2:
            continue
        name = event_name(entry) or key
        state = make_event_pre_choice_state(run, deck, relics, floor, reconstruction_issues, entry)
        pos = {"event_id": name, "event_name": name, "label": picked}
        pos_token = option_token("event", pos)
        for rejected in options:
            if rejected == picked:
                continue
            neg = {"event_id": name, "event_name": name, "label": rejected}
            neg_token = option_token("event", neg)
            yield {
                "kind": "event",
                "state": state,
                "pos_choice": pos,
                "neg_choice": neg,
                "weight": class_weight_by_token.get(pos_token, 1.0),
                "pos_prior_logit": log_prior_by_token.get(pos_token, 0.0),
                "neg_prior_logit": log_prior_by_token.get(neg_token, 0.0),
            }


def _event_numeric(entry, key, default=0.0):
    try:
        return float(entry.get(key, default) or 0.0)
    except Exception:
        return float(default)


def make_event_pre_choice_state(run, deck, relics, floor, reconstruction_issues, entry):
    """Build event state at decision time, not after the chosen option resolves.

    SlayTheData per-floor HP/gold values are end-of-floor values. For event
    decisions that immediately change HP/gold/max HP, using those values leaks
    the selected outcome into the input state. Invert the numeric effects that
    are explicitly recorded on the event choice entry.
    """
    state = make_state(run, deck, relics, floor, reconstruction_issues)

    max_hp = float(state.get("max_hp", 0) or 0)
    max_hp += _event_numeric(entry, "max_hp_loss")
    max_hp -= _event_numeric(entry, "max_hp_gain")
    max_hp = max(1.0, max_hp)

    current_hp = float(state.get("current_hp", 0) or 0)
    current_hp += _event_numeric(entry, "damage_taken")
    current_hp -= _event_numeric(entry, "damage_healed")
    current_hp = max(0.0, min(current_hp, max_hp))

    gold = float(state.get("gold", 0) or 0)
    gold -= _event_numeric(entry, "gold_gain")
    gold += _event_numeric(entry, "gold_loss")
    gold = max(0.0, gold)

    state["current_hp"] = int(round(current_hp))
    state["max_hp"] = int(round(max_hp))
    state["gold"] = int(round(gold))
    state["event_state_semantics"] = "pre_choice_numeric_reconstructed"
    return state


def _canonical_campfire_choice(choice):
    choice = choice_text(choice).upper()
    if choice == "TOKE":
        return "PURGE"
    return choice


def campfire_candidates(chosen, relics, *, can_recall=False):
    chosen = _canonical_campfire_choice(chosen)
    relic_keys = {normalize_token(value) for value in relics}
    candidates = ["REST", "SMITH"]
    if "shovel" in relic_keys:
        candidates.append("DIG")
    if "girya" in relic_keys:
        candidates.append("LIFT")
    if "peacepipe" in relic_keys:
        candidates.append("PURGE")
    if can_recall or chosen == "RECALL":
        candidates.append("RECALL")
    if chosen and chosen not in candidates:
        candidates.append(chosen)
    return candidates


def campfire_pairs(run, deck, relics, floor, reconstruction_issues, *, can_recall=False):
    for entry in run.get("campfire_choices") or []:
        if int(entry.get("floor", -1) or -1) != floor:
            continue
        picked = _canonical_campfire_choice(entry.get("key"))
        if not picked:
            continue
        options = campfire_candidates(picked, relics, can_recall=can_recall)
        if len(options) < 2:
            continue
        state = make_campfire_state(run, deck, relics, floor, reconstruction_issues)
        for rejected in options:
            if rejected == picked:
                continue
            yield {
                "kind": "campfire",
                "state": state,
                "pos_choice": picked,
                "neg_choice": rejected,
                "weight": campfire_smith_rest_weight(state, picked, rejected),
            }


def map_pairs(run, deck, relics, floor, reconstruction_issues):
    path = list(run.get("path_per_floor") or run.get("path_taken") or [])
    if floor <= 0 or floor > len(path):
        return
    picked = choice_text(path[floor - 1])
    if picked == "BOSS" or picked not in MAP_SYMBOLS:
        return
    state = make_state(run, deck, relics, floor, reconstruction_issues)
    pos = {"name": "MAP_" + picked, "symbol": picked}
    negatives = [symbol for symbol in MAP_SYMBOLS if symbol != picked]
    offset = (floor + len(deck) + len(relics)) % len(negatives)
    for step in range(min(2, len(negatives))):
        symbol = negatives[(offset + step) % len(negatives)]
        yield {
            "kind": "map",
            "state": state,
            "pos_choice": pos,
            "neg_choice": {"name": "MAP_" + symbol, "symbol": symbol},
            "weight": 1.0,
        }


def item_kind_from_name(run, item_name):
    key = normalize_token(item_name)
    if key in RED_CARD_RARITY:
        return "card"
    relic_keys = {normalize_token(name) for name in (run.get("relics") or [])}
    obtained_relic_keys = {
        normalize_token(entry.get("key"))
        for entry in (run.get("relics_obtained") or [])
        if isinstance(entry, dict)
    }
    if key in relic_keys or key in obtained_relic_keys:
        return "relic"
    return "item"


def shop_choice(action, item_name="", item_kind=""):
    if action == "leave":
        return {"name": "LEAVE", "action": "leave", "item_kind": "leave", "item_id": "leave"}
    if action == "purge":
        return {"name": "PURGE", "action": "purge", "item_kind": "purge", "item_id": "purge"}
    return {
        "name": "{}_{}".format(action.upper(), item_name),
        "action": action,
        "item_kind": item_kind or action,
        "item_id": item_name,
    }


def collect_shop_option_metadata(args):
    counts = Counter()
    run_count = 0
    shop_floor_count = 0
    print("Collecting shop option prior ...", flush=True)
    for run in iter_run_events(args.input_dir, max_files=args.max_files, file_progress_every=args.file_progress_every):
        run_count += 1
        if args.max_runs > 0 and run_count > args.max_runs:
            break
        path = list(run.get("path_per_floor") or run.get("path_taken") or [])
        shop_floors = {index + 1 for index, symbol in enumerate(path) if symbol == "$"}
        if not shop_floors:
            continue
        shop_floor_count += len(shop_floors)
        acted_floors = set()
        for purchase_floor, purchased in zip(run.get("item_purchase_floors") or [], run.get("items_purchased") or []):
            try:
                floor = int(purchase_floor or -1)
            except (TypeError, ValueError):
                continue
            if floor not in shop_floors:
                continue
            name = choice_text(purchased)
            if not name:
                continue
            acted_floors.add(floor)
            kind = item_kind_from_name(run, name)
            counts[option_token("shop", shop_choice("buy", name, kind))] += 1

        for purge_floor, _ in zip(run.get("items_purged_floors") or [], run.get("items_purged") or []):
            try:
                floor = int(purge_floor or -1)
            except (TypeError, ValueError):
                continue
            if floor not in shop_floors:
                continue
            acted_floors.add(floor)
            counts[option_token("shop", shop_choice("purge"))] += 1

        for floor in shop_floors - acted_floors:
            counts[option_token("shop", shop_choice("leave"))] += 1

    eps = max(0.0, float(args.shop_prior_eps))
    total = float(sum(counts.values()))
    option_count = float(len(counts))
    denominator = total + eps * option_count
    log_prior_by_token = {}
    for token, count in counts.items():
        probability = (float(count) + eps) / denominator if denominator > 0 else 0.0
        log_prior_by_token[token] = math.log(max(probability, eps if eps > 0 else 1e-12))
    print(
        "Shop prior ready  runs={} shop_floors={} options={} actions={}".format(
            run_count,
            shop_floor_count,
            len(counts),
            int(total),
        ),
        flush=True,
    )
    return {
        "shop_option_log_prior_by_token": log_prior_by_token,
        "shop_option_count_by_token": dict(counts),
        "shop_prior_eps": eps,
        "shop_prior_run_count": int(run_count),
        "shop_prior_floor_count": int(shop_floor_count),
    }


def potion_choice(action, potion_name=""):
    if action == "hold":
        return {"name": "HOLD", "action": "hold", "potion_id": "hold", "item_id": "hold"}
    return {
        "name": "USE_" + str(potion_name),
        "action": "use",
        "potion_id": potion_name,
        "item_id": potion_name,
    }


def potion_entries_by_floor(run):
    by_floor = defaultdict(list)
    for entry in run.get("potions_obtained") or []:
        if not isinstance(entry, dict):
            continue
        try:
            floor = int(float(entry.get("floor", -1) or -1))
        except (TypeError, ValueError):
            continue
        name = choice_text(entry.get("key") or entry.get("potion") or entry.get("name"))
        if floor > 0 and name:
            by_floor[floor].append(name)
    return by_floor


def potion_usage_count_by_floor(run):
    counts = defaultdict(int)
    for floor_value in run.get("potions_floor_usage") or []:
        try:
            floor = int(float(floor_value or -1))
        except (TypeError, ValueError):
            continue
        if floor > 0:
            counts[floor] += 1
    return counts


def potion_pairs_for_floor(run, deck, relics, floor, reconstruction_issues, potion_inventory, usage_count):
    if not potion_inventory:
        return []
    state = make_state(run, deck, relics, floor, reconstruction_issues)
    hold = potion_choice("hold")
    pairs = []
    if usage_count > 0:
        for _ in range(min(usage_count, len(potion_inventory))):
            potion_name = potion_inventory.pop(0)
            pairs.append(
                {
                    "kind": "potion",
                    "state": state,
                    "pos_choice": potion_choice("use", potion_name),
                    "neg_choice": hold,
                    "weight": 1.0,
                }
            )
        return pairs
    # Most floors do not use potions; keep these negatives light so they do not
    # drown the rarer positive "use" examples.
    for potion_name in potion_inventory[:2]:
        pairs.append(
            {
                "kind": "potion",
                "state": state,
                "pos_choice": hold,
                "neg_choice": potion_choice("use", potion_name),
                "weight": 0.15,
            }
        )
    return pairs


def _with_choice_priors(kind, pair, metadata):
    if kind == "event":
        log_prior_by_token = (metadata or {}).get("event_option_log_prior_by_token") or {}
    elif kind == "shop":
        log_prior_by_token = (metadata or {}).get("shop_option_log_prior_by_token") or {}
    else:
        log_prior_by_token = {}
    if not log_prior_by_token:
        return pair
    pair = dict(pair)
    pair["pos_prior_logit"] = log_prior_by_token.get(option_token(kind, pair["pos_choice"]), 0.0)
    pair["neg_prior_logit"] = log_prior_by_token.get(option_token(kind, pair["neg_choice"]), 0.0)
    return pair


def shop_pairs(run, deck, relics, floor, reconstruction_issues, shop_metadata=None):
    path = list(run.get("path_per_floor") or run.get("path_taken") or [])
    if floor <= 0 or floor > len(path) or path[floor - 1] != "$":
        return
    state = make_state(run, deck, relics, floor, reconstruction_issues)
    leave = shop_choice("leave")
    found_action = False

    purchase_floors = list(run.get("item_purchase_floors") or [])
    purchased_items = list(run.get("items_purchased") or [])
    for purchase_floor, purchased in zip(purchase_floors, purchased_items):
        if int(purchase_floor or -1) != floor:
            continue
        name = choice_text(purchased)
        if not name:
            continue
        found_action = True
        kind = item_kind_from_name(run, name)
        yield _with_choice_priors("shop", {
            "kind": "shop",
            "state": state,
            "pos_choice": shop_choice("buy", name, kind),
            "neg_choice": leave,
            "weight": 1.0,
        }, shop_metadata)

    purge_floors = list(run.get("items_purged_floors") or [])
    purged_cards = list(run.get("items_purged") or [])
    for purge_floor, purged_name in zip(purge_floors, purged_cards):
        if int(purge_floor or -1) != floor:
            continue
        found_action = True
        purge = shop_choice("purge")
        yield _with_choice_priors("shop", {
            "kind": "shop",
            "state": state,
            "pos_choice": purge,
            "neg_choice": leave,
            "weight": 1.0,
        }, shop_metadata)

    if not found_action:
        yield _with_choice_priors("shop", {
            "kind": "shop",
            "state": state,
            "pos_choice": leave,
            "neg_choice": shop_choice("purge"),
            "weight": 0.5,
        }, shop_metadata)


def build_pairs_from_run(run, task, event_vocab=None, event_metadata=None):
    deck = [parse_card_instance(name) for name in STARTER_DECK]
    relics = [STARTER_RELIC]
    potion_inventory = []
    potions_by_floor = potion_entries_by_floor(run) if task == "potion" else {}
    potion_usage_by_floor = potion_usage_count_by_floor(run) if task == "potion" else {}
    reconstruction_issues = 0
    max_floor = int(run.get("floor_reached", 0) or 0)
    if max_floor <= 0:
        return

    ruby_key_recalled = False
    for floor in range(1, max_floor + 1):
        if task == "boss_relic":
            yield from boss_relic_pairs(run, deck, relics, floor, reconstruction_issues)
        elif task == "event":
            yield from event_pairs(run, deck, relics, floor, reconstruction_issues, event_metadata or {"vocab": event_vocab or {}})
        elif task == "campfire":
            yield from campfire_pairs(
                run,
                deck,
                relics,
                floor,
                reconstruction_issues,
                can_recall=not ruby_key_recalled,
            )
        elif task == "map":
            yield from map_pairs(run, deck, relics, floor, reconstruction_issues)
        elif task == "shop":
            yield from shop_pairs(run, deck, relics, floor, reconstruction_issues, event_metadata)
        elif task == "potion":
            yield from potion_pairs_for_floor(
                run,
                deck,
                relics,
                floor,
                reconstruction_issues,
                potion_inventory,
                potion_usage_by_floor.get(floor, 0),
            )

        reconstruction_issues += apply_floor_updates(floor, run, deck, relics)
        add_card_choices_at_floor(run, floor, deck)
        if task == "campfire":
            for entry in run.get("campfire_choices") or []:
                if int(entry.get("floor", -1) or -1) == floor and _canonical_campfire_choice(entry.get("key")) == "RECALL":
                    ruby_key_recalled = True
                    break
        if task == "potion":
            potion_inventory.extend(potions_by_floor.get(floor, []))
            potion_inventory = potion_inventory[-3:]


def task_cache_dir(base_cache_dir, task):
    return Path(base_cache_dir) / task


def ensure_cache(args, task):
    cache_dir = task_cache_dir(args.cache_dir, task)
    meta_path = cache_dir / "meta.json"
    if not args.rebuild_cache and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        cache_matches = int(meta.get("cache_format_version", 0) or 0) == CACHE_FORMAT_VERSION and meta.get("task") == task
        if task == "event":
            cache_matches = (
                cache_matches
                and bool(meta.get("event_prior_delta")) == bool(args.event_prior_delta)
                and float(meta.get("event_prior_eps", args.event_prior_eps)) == float(args.event_prior_eps)
                and float(meta.get("event_class_balance_alpha", args.event_class_balance_alpha)) == float(args.event_class_balance_alpha)
                and float(meta.get("event_class_balance_min", args.event_class_balance_min)) == float(args.event_class_balance_min)
                and float(meta.get("event_class_balance_max", args.event_class_balance_max)) == float(args.event_class_balance_max)
            )
        if task == "shop":
            cache_matches = (
                cache_matches
                and bool(meta.get("shop_prior_delta")) == bool(args.shop_prior_delta)
                and float(meta.get("shop_prior_eps", args.shop_prior_eps)) == float(args.shop_prior_eps)
            )
        if cache_matches:
            train_shards = [cache_dir / name for name in meta.get("train_shards", [])]
            valid_shards = [cache_dir / name for name in meta.get("valid_shards", [])]
            if train_shards and all(path.exists() for path in train_shards) and all(path.exists() for path in valid_shards):
                print("Using {} cache from {}".format(task, cache_dir), flush=True)
                return train_shards, valid_shards, meta

    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    event_metadata = collect_event_option_metadata(args) if task == "event" else {}
    if task == "event" and not args.event_prior_delta:
        event_metadata["event_option_log_prior_by_token"] = {}
        event_metadata["event_option_class_weight_by_token"] = {}
    if task == "shop":
        event_metadata = collect_shop_option_metadata(args) if args.shop_prior_delta else {}
    train_shards = []
    valid_shards = []
    train_shard_counts = []
    valid_shard_counts = []
    train_buffer = []
    valid_buffer = []
    train_pairs = 0
    valid_pairs = 0
    runs = 0

    def flush_buffer(split_name, buffer, shard_names, shard_counts):
        if not buffer:
            return
        shard_name = "{}_pairs_{:05d}.pt".format(split_name, len(shard_names))
        shard_path = cache_dir / shard_name
        tensors = pairwise_batch_to_tensors(buffer, device="cpu")
        shard_count = int(tensors["weight"].shape[0])
        payload = {
            "state": tensors["state"].cpu(),
            "pos_choice": tensors["pos_choice"].cpu(),
            "neg_choice": tensors["neg_choice"].cpu(),
            "weight": tensors["weight"].cpu(),
            "count": shard_count,
        }
        if (task == "event" and args.event_prior_delta) or (task == "shop" and args.shop_prior_delta):
            payload["pos_prior_logit"] = torch.tensor(
                [float(item.get("pos_prior_logit", 0.0)) for item in buffer],
                dtype=torch.float32,
            )
            payload["neg_prior_logit"] = torch.tensor(
                [float(item.get("neg_prior_logit", 0.0)) for item in buffer],
                dtype=torch.float32,
            )
        torch.save(payload, shard_path)
        shard_names.append(shard_name)
        shard_counts.append(shard_count)
        buffer.clear()

    print("Building {} tensor cache at {} ...".format(task, cache_dir), flush=True)
    for run in iter_run_events(args.input_dir, max_files=args.max_files, file_progress_every=args.file_progress_every):
        runs += 1
        if args.max_runs > 0 and runs > args.max_runs:
            break
        pairs = list(build_pairs_from_run(run, task, event_metadata=event_metadata))
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
                "{} cache progress  runs={} train_pairs={} valid_pairs={} train_shards={} valid_shards={}".format(
                    task, runs, train_pairs, valid_pairs, len(train_shards), len(valid_shards)
                ),
                flush=True,
            )

    flush_buffer("train", train_buffer, train_shards, train_shard_counts)
    flush_buffer("valid", valid_buffer, valid_shards, valid_shard_counts)
    if train_pairs <= 0:
        raise RuntimeError("No training pairs were built for task {}".format(task))

    meta = {
        "source": "slay_the_data_run_choice_pairwise",
        "cache_format_version": CACHE_FORMAT_VERSION,
        "task": task,
        "input_dir": str(args.input_dir),
        "train_pairs": train_pairs,
        "valid_pairs": valid_pairs,
        "train_shards": train_shards,
        "valid_shards": valid_shards,
        "train_shard_counts": train_shard_counts,
        "valid_shard_counts": valid_shard_counts,
        "event_vocab": event_metadata.get("vocab", {}) if task == "event" else {},
        "seed": args.seed,
    }
    if task == "event":
        meta.update(
            {
                "event_prior_delta": bool(args.event_prior_delta),
                "event_prior_eps": float(args.event_prior_eps),
                "event_class_balance_alpha": float(args.event_class_balance_alpha),
                "event_class_balance_min": float(args.event_class_balance_min),
                "event_class_balance_max": float(args.event_class_balance_max),
                "event_option_log_prior_by_token": event_metadata.get("event_option_log_prior_by_token", {}),
                "event_option_class_weight_by_token": event_metadata.get("event_option_class_weight_by_token", {}),
                "event_option_count_by_token": event_metadata.get("event_option_count_by_token", {}),
            }
        )
    if task == "shop":
        meta.update(
            {
                "shop_prior_delta": bool(args.shop_prior_delta),
                "shop_prior_eps": float(args.shop_prior_eps),
                "shop_option_log_prior_by_token": event_metadata.get("shop_option_log_prior_by_token", {}),
                "shop_option_count_by_token": event_metadata.get("shop_option_count_by_token", {}),
                "shop_prior_run_count": int(event_metadata.get("shop_prior_run_count", 0) or 0),
                "shop_prior_floor_count": int(event_metadata.get("shop_prior_floor_count", 0) or 0),
            }
        )
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(
        "{} cache ready  train_pairs={} valid_pairs={}".format(task, train_pairs, valid_pairs),
        flush=True,
    )
    return [cache_dir / name for name in train_shards], [cache_dir / name for name in valid_shards], meta


def shard_batch_to_device(shard, indices, device):
    batch = {
        "state": shard["state"][indices].to(device=device, non_blocking=True),
        "pos_choice": shard["pos_choice"][indices].to(device=device, non_blocking=True),
        "neg_choice": shard["neg_choice"][indices].to(device=device, non_blocking=True),
        "weight": shard["weight"][indices].to(device=device, non_blocking=True),
    }
    if "pos_prior_logit" in shard and "neg_prior_logit" in shard:
        batch["pos_prior_logit"] = shard["pos_prior_logit"][indices].to(device=device, non_blocking=True)
        batch["neg_prior_logit"] = shard["neg_prior_logit"][indices].to(device=device, non_blocking=True)
    return batch


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


def scored_pair_from_batch(model, batch, prior_weight=0.0):
    pos, neg = option_scores_from_pairwise_batch(model, batch)
    if "pos_prior_logit" in batch and "neg_prior_logit" in batch:
        weight = float(prior_weight)
        pos = pos + weight * batch["pos_prior_logit"]
        neg = neg + weight * batch["neg_prior_logit"]
    return pos, neg


def evaluate_pairwise(model, shard_paths, device, batch_size=512, prior_weight=0.0):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch, batch_len in stream_pairwise_batches(shard_paths, batch_size=batch_size, device=device, shuffle=False):
            pos, neg = scored_pair_from_batch(model, batch, prior_weight=prior_weight)
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
    model = RunChoicePolicyNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output_path = Path(args.output) if args.output and args.task != "all" else Path(args.output_dir) / DEFAULT_OUTPUTS[task]
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir and args.task != "all" else output_path.with_suffix("").with_name(output_path.stem + "_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_state = None
    best_valid = None
    use_event_prior_delta = bool(task == "event" and args.event_prior_delta)
    use_shop_prior_delta = bool(task == "shop" and args.shop_prior_delta)
    prior_weight = (
        float(args.event_prior_weight)
        if use_event_prior_delta
        else float(args.shop_prior_weight)
        if use_shop_prior_delta
        else 0.0
    )

    def checkpoint_summary(epoch_history):
        summary = {
            "source": "slay_the_data_run_choice_pairwise",
            "task": task,
            "train_pairs": train_pairs,
            "valid_pairs": valid_pairs,
            "history": list(epoch_history),
            "last_epoch": epoch_history[-1]["epoch"] if epoch_history else 0,
        }
        if use_event_prior_delta:
            summary.update(
                {
                    "score_mode": "event_prior_delta",
                    "event_prior_weight": prior_weight,
                    "event_prior_eps": float(meta.get("event_prior_eps", args.event_prior_eps)),
                    "event_class_balance_alpha": float(meta.get("event_class_balance_alpha", args.event_class_balance_alpha)),
                    "event_class_balance_min": float(meta.get("event_class_balance_min", args.event_class_balance_min)),
                    "event_class_balance_max": float(meta.get("event_class_balance_max", args.event_class_balance_max)),
                    "event_option_log_prior_by_token": meta.get("event_option_log_prior_by_token", {}),
                    "event_option_count_by_token": meta.get("event_option_count_by_token", {}),
                }
            )
        if use_shop_prior_delta:
            summary.update(
                {
                    "score_mode": "shop_prior_delta",
                    "shop_prior_weight": prior_weight,
                    "shop_prior_eps": float(meta.get("shop_prior_eps", args.shop_prior_eps)),
                    "shop_option_log_prior_by_token": meta.get("shop_option_log_prior_by_token", {}),
                    "shop_option_count_by_token": meta.get("shop_option_count_by_token", {}),
                    "shop_prior_run_count": int(meta.get("shop_prior_run_count", 0) or 0),
                    "shop_prior_floor_count": int(meta.get("shop_prior_floor_count", 0) or 0),
                }
            )
        return summary

    for epoch in range(1, args.epochs + 1):
        print(
            "{} epoch {}/{} start  train_pairs={} valid_pairs={}".format(task, epoch, args.epochs, train_pairs, valid_pairs),
            flush=True,
        )
        model.train()
        total_loss = 0.0
        total = 0
        total_batches = total_batches_for_shards(train_shard_counts, args.batch_size)
        for batch_index, (batch, batch_len) in enumerate(
            stream_pairwise_batches(
                train_shards,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed + epoch,
                shuffle=True,
            ),
            1,
        ):
            pos, neg = scored_pair_from_batch(model, batch, prior_weight=prior_weight)
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
        train_metrics = evaluate_pairwise(model, train_shards, device, batch_size=eval_batch_size, prior_weight=prior_weight)
        valid_metrics = evaluate_pairwise(model, valid_shards, device, batch_size=eval_batch_size, prior_weight=prior_weight) if valid_pairs > 0 else {"loss": 0.0, "accuracy": 0.0}
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
        summary = checkpoint_summary(history)
        save_run_choice_checkpoint(model, str(checkpoint_dir / "latest.pt"), training_summary=summary)
        save_run_choice_checkpoint(model, str(checkpoint_dir / "epoch_{:03d}.pt".format(epoch)), training_summary=summary)
        current_valid = (valid_metrics["accuracy"], -valid_metrics["loss"])
        if best_valid is None or current_valid > best_valid:
            best_valid = current_valid
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            save_run_choice_checkpoint(model, str(checkpoint_dir / "best.pt"), training_summary=summary)

    if best_state is not None:
        model.load_state_dict(best_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_run_choice_checkpoint(
        model,
        str(output_path),
        training_summary=checkpoint_summary(history)
        | {
            "cache_dir": str(task_cache_dir(args.cache_dir, task)),
            "checkpoint_dir": str(checkpoint_dir),
        },
    )
    print("Saved {} model to {}".format(task, output_path.resolve()), flush=True)
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
        if task == "boss_relic":
            print("  SPIRECOMM_BOSS_RELIC_MODEL_PATH={}".format(os.path.abspath(output)))
        elif task == "event":
            print("  SPIRECOMM_EVENT_CHOICE_MODEL_PATH={}".format(os.path.abspath(output)))
        elif task == "campfire":
            print("  SPIRECOMM_CAMPFIRE_MODEL_PATH={}".format(os.path.abspath(output)))
        elif task == "map":
            print("  SPIRECOMM_MAP_CHOICE_MODEL_PATH={}".format(os.path.abspath(output)))
        elif task == "shop":
            print("  SPIRECOMM_SHOP_CHOICE_MODEL_PATH={}".format(os.path.abspath(output)))


if __name__ == "__main__":
    main()
