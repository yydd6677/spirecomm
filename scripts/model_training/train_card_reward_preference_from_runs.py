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
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

CACHE_FORMAT_VERSION = 2

from spirecomm.ai.card_reward_model import (
    CardRewardPolicyNetwork,
    normalize_token,
    option_scores_from_pairwise_batch,
    pairwise_batch_to_tensors,
    save_card_reward_checkpoint,
)


STARTER_DECK = [
    "Strike_R",
    "Strike_R",
    "Strike_R",
    "Strike_R",
    "Strike_R",
    "Defend_R",
    "Defend_R",
    "Defend_R",
    "Defend_R",
    "Bash",
]

STARTER_RELIC = "Burning Blood"

RARITY_FACTOR = {
    "COMMON": 1.00,
    "UNCOMMON": 1.10,
    "RARE": 1.25,
    "COLORLESS": 1.20,
}

RARITY_SCORE = {
    "COMMON": 0.10,
    "UNCOMMON": 0.20,
    "RARE": 0.30,
    "COLORLESS": 0.16,
}

RED_CARD_RARITY_RAW = {
    "anger": "COMMON",
    "body_slam": "COMMON",
    "clash": "COMMON",
    "cleave": "COMMON",
    "clothesline": "COMMON",
    "headbutt": "COMMON",
    "iron_wave": "COMMON",
    "perfected_strike": "COMMON",
    "pommel_strike": "COMMON",
    "sword_boomerang": "COMMON",
    "thunder_clap": "COMMON",
    "twin_strike": "COMMON",
    "wild_strike": "COMMON",
    "armaments": "COMMON",
    "battle_trance": "COMMON",
    "bloodletting": "COMMON",
    "burning_pact": "COMMON",
    "disarm": "COMMON",
    "flex": "COMMON",
    "havoc": "COMMON",
    "intimidate": "COMMON",
    "rage": "COMMON",
    "shrug_it_off": "COMMON",
    "true_grit": "COMMON",
    "warcry": "COMMON",
    "blood_for_blood": "UNCOMMON",
    "carnage": "UNCOMMON",
    "drop_kick": "UNCOMMON",
    "heavy_blade": "UNCOMMON",
    "hemokinesis": "UNCOMMON",
    "pummel": "UNCOMMON",
    "rampage": "UNCOMMON",
    "reckless_charge": "UNCOMMON",
    "searing_blow": "UNCOMMON",
    "uppercut": "UNCOMMON",
    "whirlwind": "UNCOMMON",
    "dual_wield": "UNCOMMON",
    "entrench": "UNCOMMON",
    "flame_barrier": "UNCOMMON",
    "ghostly_armor": "UNCOMMON",
    "infernal_blade": "UNCOMMON",
    "power_through": "UNCOMMON",
    "second_wind": "UNCOMMON",
    "seeing_red": "UNCOMMON",
    "sentinel": "UNCOMMON",
    "spot_weakness": "UNCOMMON",
    "brutality": "UNCOMMON",
    "combust": "UNCOMMON",
    "dark_embrace": "UNCOMMON",
    "evolve": "UNCOMMON",
    "fire_breathing": "UNCOMMON",
    "inflame": "UNCOMMON",
    "metallicize": "UNCOMMON",
    "rupture": "UNCOMMON",
    "bludgeon": "RARE",
    "feed": "RARE",
    "fiend_fire": "RARE",
    "immolate": "RARE",
    "reaper": "RARE",
    "sever_soul": "RARE",
    "double_tap": "RARE",
    "exhume": "RARE",
    "impervious": "RARE",
    "limit_break": "RARE",
    "offering": "RARE",
    "shockwave": "RARE",
    "barricade": "RARE",
    "berserk": "RARE",
    "corruption": "RARE",
    "demon_form": "RARE",
    "feel_no_pain": "RARE",
    "juggernaut": "RARE",
}
RED_CARD_RARITY = {normalize_token(key): value for key, value in RED_CARD_RARITY_RAW.items()}

ATTACK_KEYS = {
    normalize_token(name)
    for name in [
        "Strike_R",
        "Bash",
        "Anger",
        "Body Slam",
        "Clash",
        "Cleave",
        "Clothesline",
        "Headbutt",
        "Iron Wave",
        "Perfected Strike",
        "Pommel Strike",
        "Sword Boomerang",
        "Thunderclap",
        "Twin Strike",
        "Wild Strike",
        "Blood for Blood",
        "Carnage",
        "Dropkick",
        "Heavy Blade",
        "Hemokinesis",
        "Pummel",
        "Rampage",
        "Reckless Charge",
        "Searing Blow",
        "Uppercut",
        "Whirlwind",
        "Bludgeon",
        "Feed",
        "Fiend Fire",
        "Immolate",
        "Reaper",
        "Sever Soul",
    ]
}
SKILL_KEYS = {
    normalize_token(name)
    for name in [
        "Defend_R",
        "Armaments",
        "Battle Trance",
        "Bloodletting",
        "Burning Pact",
        "Disarm",
        "Flex",
        "Havoc",
        "Intimidate",
        "Rage",
        "Shrug It Off",
        "True Grit",
        "Warcry",
        "Dual Wield",
        "Entrench",
        "Flame Barrier",
        "Ghostly Armor",
        "Infernal Blade",
        "Power Through",
        "Second Wind",
        "Seeing Red",
        "Sentinel",
        "Spot Weakness",
        "Double Tap",
        "Exhume",
        "Impervious",
        "Limit Break",
        "Offering",
        "Shockwave",
    ]
}
POWER_KEYS = {
    normalize_token(name)
    for name in [
        "Brutality",
        "Combust",
        "Dark Embrace",
        "Evolve",
        "Fire Breathing",
        "Inflame",
        "Metallicize",
        "Rupture",
        "Barricade",
        "Berserk",
        "Corruption",
        "Demon Form",
        "Feel No Pain",
        "Juggernaut",
    ]
}
AOE_KEYS = {
    normalize_token(name)
    for name in ["Cleave", "Thunderclap", "Whirlwind", "Immolate", "Reaper", "Shockwave"]
}
BLOCK_KEYS = {
    normalize_token(name)
    for name in [
        "Defend_R",
        "Iron Wave",
        "Shrug It Off",
        "True Grit",
        "Entrench",
        "Flame Barrier",
        "Ghostly Armor",
        "Power Through",
        "Second Wind",
        "Sentinel",
        "Impervious",
    ]
}
DRAW_KEYS = {
    normalize_token(name)
    for name in ["Pommel Strike", "Shrug It Off", "Battle Trance", "Burning Pact", "Offering", "Warcry", "Dark Embrace"]
}
STRENGTH_KEYS = {
    normalize_token(name)
    for name in ["Inflame", "Spot Weakness", "Demon Form", "Flex", "Limit Break", "J.A.X."]
}
EXHAUST_KEYS = {
    normalize_token(name)
    for name in ["Burning Pact", "True Grit", "Second Wind", "Fiend Fire", "Offering", "Exhume", "Corruption", "Feel No Pain", "Dark Embrace"]
}
BLOCK_SCALING_KEYS = {
    normalize_token(name)
    for name in ["Barricade", "Entrench", "Body Slam", "Juggernaut", "Flame Barrier", "Impervious"]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train card reward model from SlayTheData Ironclad runs with pairwise preference.")
    parser.add_argument("--input-dir", default="/media/yydd/E0E24E0F6119A708/SlayTheData_ironclad_victory_a20")
    parser.add_argument("--output", default="/home/yydd/spirecomm/models/card_reward_from_runs.pt")
    parser.add_argument("--cache-dir", default="/home/yydd/spirecomm/_cache/card_reward_from_runs")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--file-progress-every", type=int, default=1000)
    parser.add_argument("--batch-progress-every", type=int, default=200)
    parser.add_argument("--shuffle-buffer-scenarios", type=int, default=2048)
    parser.add_argument("--tensor-cache-pairs-per-shard", type=int, default=200000)
    parser.add_argument("--checkpoint-dir", default="")
    return parser.parse_args()


def act_for_floor(floor):
    if floor <= 16:
        return 1
    if floor <= 33:
        return 2
    return 3


def next_boss_floor(floor):
    if floor <= 16:
        return 16
    if floor <= 33:
        return 33
    if floor <= 50:
        return 50
    return 57


def parse_card_instance(card_name):
    upgrades = 0
    raw = str(card_name or "")
    match = re.match(r"^(.*)\+(\d+)$", raw)
    if match:
        raw = match.group(1)
        upgrades = int(match.group(2))
    card_id = raw.strip()
    key = normalize_token(card_id)
    rarity = RED_CARD_RARITY.get(key, "COLORLESS")
    if key in ATTACK_KEYS:
        card_type = "ATTACK"
    elif key in POWER_KEYS:
        card_type = "POWER"
    elif key in SKILL_KEYS:
        card_type = "SKILL"
    elif "curse" in key or key in {"regret", "pain", "doubt", "injury", "shame", "writhe", "parasite", "normality"}:
        card_type = "CURSE"
        rarity = "CURSE"
    else:
        card_type = "SKILL"
    return {
        "card_id": card_id,
        "name": card_id,
        "type": card_type,
        "rarity": rarity,
        "upgrades": upgrades,
    }


def make_relic(relic_name):
    return {"relic_id": relic_name, "name": relic_name}


def value_on_floor(sequence, floor, default=0):
    values = list(sequence or [])
    if not values:
        return default
    index = max(0, floor - 1)
    if index < len(values):
        return values[index]
    return values[-1]


def remove_one(deck, target_name):
    target_key = normalize_token(target_name)
    for index, card in enumerate(deck):
        if normalize_token(card["card_id"]) == target_key:
            deck.pop(index)
            return True
    return False


def upgrade_one(deck, target_name):
    target_key = normalize_token(target_name)
    for card in deck:
        if normalize_token(card["card_id"]) == target_key:
            card["upgrades"] = int(card.get("upgrades", 0) or 0) + 1
            return True
    return False


def boss_proximity_weight(floor):
    distance = max(0, next_boss_floor(floor) - floor)
    if distance <= 1:
        return 1.30
    if distance <= 3:
        return 1.15
    return 1.00


def rarity_factor_for_choice(reward_cards, chosen_card=None, skip_choice=False):
    if skip_choice:
        if not reward_cards:
            return 1.0
        best = max((RARITY_FACTOR.get(card.get("rarity", "COMMON"), 1.0) for card in reward_cards), default=1.0)
        return best
    if chosen_card is None:
        return 1.0
    return RARITY_FACTOR.get(chosen_card.get("rarity", "COMMON"), 1.0)


def make_scenario(run, deck, relics, floor, reward_cards, reconstruction_issues):
    current_hp = value_on_floor(run.get("current_hp_per_floor"), floor, default=0)
    max_hp = value_on_floor(run.get("max_hp_per_floor"), floor, default=0)
    gold = value_on_floor(run.get("gold_per_floor"), floor, default=0)
    return {
        "floor": floor,
        "act": act_for_floor(floor),
        "current_hp": current_hp,
        "max_hp": max_hp,
        "gold": gold,
        "deck": [dict(card) for card in deck],
        "relics": [make_relic(name) for name in relics],
        "reward_cards": [dict(card) for card in reward_cards],
        "reconstruction_issues": reconstruction_issues,
    }


def apply_floor_updates(floor, run, deck, relics):
    issues = 0

    for entry in run.get("relics_obtained") or []:
        if int(entry.get("floor", -1) or -1) == floor:
            key = entry.get("key")
            if key and key not in relics:
                relics.append(key)

    boss_relics = run.get("boss_relics") or []
    if floor == 17 and len(boss_relics) >= 1:
        picked = boss_relics[0].get("picked")
        if picked and picked not in relics:
            relics.append(picked)
    if floor == 34 and len(boss_relics) >= 2:
        picked = boss_relics[1].get("picked")
        if picked and picked not in relics:
            relics.append(picked)

    for entry in run.get("event_choices") or []:
        if int(entry.get("floor", -1) or -1) != floor:
            continue
        for removed in entry.get("cards_removed") or []:
            if not remove_one(deck, removed):
                issues += 1
        for upgraded in entry.get("cards_upgraded") or []:
            if not upgrade_one(deck, upgraded):
                issues += 1
        for obtained in entry.get("cards_obtained") or []:
            deck.append(parse_card_instance(obtained))
        for relic_name in entry.get("relics_obtained") or []:
            if relic_name not in relics:
                relics.append(relic_name)

    for entry in run.get("campfire_choices") or []:
        if int(entry.get("floor", -1) or -1) != floor:
            continue
        if entry.get("key") == "SMITH":
            if not upgrade_one(deck, entry.get("data")):
                issues += 1

    purge_floors = list(run.get("items_purged_floors") or [])
    purged_cards = list(run.get("items_purged") or [])
    for purge_floor, purged_name in zip(purge_floors, purged_cards):
        if int(purge_floor or -1) == floor:
            if not remove_one(deck, purged_name):
                issues += 1

    purchase_floors = list(run.get("item_purchase_floors") or [])
    purchased_items = list(run.get("items_purchased") or [])
    for purchase_floor, purchased in zip(purchase_floors, purchased_items):
        if int(purchase_floor or -1) != floor:
            continue
        rarity = RED_CARD_RARITY.get(normalize_token(purchased))
        if rarity is not None:
            deck.append(parse_card_instance(purchased))
        elif purchased in run.get("relics", []):
            if purchased not in relics:
                relics.append(purchased)

    return issues


def standard_reward_from_choice(choice):
    picked = choice.get("picked")
    not_picked = list(choice.get("not_picked") or [])
    if picked in (None, ""):
        return None
    if picked == "Singing Bowl":
        picked = "SKIP"
    if picked == "SKIP":
        if len(not_picked) != 3:
            return None
        cards = [parse_card_instance(name) for name in not_picked]
        return {
            "reward_cards": cards,
            "skip_choice": True,
            "chosen_card": None,
        }
    if len(not_picked) != 2:
        return None
    cards = [parse_card_instance(name) for name in (not_picked + [picked])]
    return {
        "reward_cards": cards,
        "skip_choice": False,
        "chosen_card": parse_card_instance(picked),
    }


def build_scenarios_from_run(run):
    deck = [parse_card_instance(name) for name in STARTER_DECK]
    relics = [STARTER_RELIC]
    reconstruction_issues = 0
    scenarios = []

    max_floor = int(run.get("floor_reached", 0) or 0)
    choices_by_floor = defaultdict(list)
    for choice in run.get("card_choices") or []:
        floor = int(choice.get("floor", -1) or -1)
        if floor > 0:
            choices_by_floor[floor].append(choice)

    for floor in range(1, max_floor + 1):
        reconstruction_issues += apply_floor_updates(floor, run, deck, relics)
        for choice in choices_by_floor.get(floor, []):
            parsed = standard_reward_from_choice(choice)
            if parsed is None:
                continue
            reward_cards = parsed["reward_cards"]
            chosen_card = parsed["chosen_card"]
            skip_choice = bool(parsed["skip_choice"])
            scenario = make_scenario(run, deck, relics, floor, reward_cards, reconstruction_issues)
            scenarios.append(
                {
                    "scenario": scenario,
                    "chosen_card": chosen_card,
                    "skip_choice": skip_choice,
                }
            )
            if not skip_choice and chosen_card is not None:
                deck.append(dict(chosen_card))
    return scenarios


def load_run_scenarios(input_dir, max_files=0, max_scenarios=0, file_progress_every=1000):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError("Input dir not found: {}".format(input_path))
    files = sorted(input_path.glob("*.json"))
    if not files:
        raise FileNotFoundError("No .json files found in {}".format(input_path))
    if max_files > 0:
        files = files[:max_files]
    scenario_count = 0
    for index, path in enumerate(files, 1):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        items = data if isinstance(data, list) else [data]
        for item in items:
            event = item.get("event") if isinstance(item, dict) and "event" in item else item
            if not isinstance(event, dict):
                continue
            for scenario in build_scenarios_from_run(event):
                yield scenario
                scenario_count += 1
                if max_scenarios > 0 and scenario_count >= max_scenarios:
                    return
        if file_progress_every > 0 and (index % file_progress_every == 0 or index == len(files)):
            print(
                "Loaded files: {}/{}  scenarios={}".format(index, len(files), scenario_count),
                flush=True,
            )


def scenario_weight(scenario_record):
    scenario = scenario_record["scenario"]
    chosen_card = scenario_record.get("chosen_card")
    skip_choice = bool(scenario_record.get("skip_choice"))
    base_quality = 1.0
    boss_proximity = boss_proximity_weight(scenario["floor"])
    rarity_factor = rarity_factor_for_choice(
        scenario["reward_cards"],
        chosen_card=chosen_card,
        skip_choice=skip_choice,
    )
    weight = base_quality * boss_proximity * rarity_factor
    return max(0.50, min(4.00, weight))


def iter_pairwise_from_scenario_record(record):
    reward_cards = list(record["scenario"]["reward_cards"] or [])[:3]
    weight = float(record.get("weight", 1.0))
    if bool(record.get("skip_choice")):
        for rejected_card in reward_cards:
            yield {
                "scenario": record["scenario"],
                "pos_card": None,
                "neg_card": rejected_card,
                "pos_is_skip": True,
                "neg_is_skip": False,
                "weight": weight,
            }
        return
    chosen_card = record.get("chosen_card")
    if not chosen_card:
        return
    chosen_key = normalize_token(chosen_card.get("card_id") or chosen_card.get("name"))
    for rejected_card in reward_cards:
        rejected_key = normalize_token(rejected_card.get("card_id") or rejected_card.get("name"))
        if rejected_key == chosen_key:
            continue
        yield {
            "scenario": record["scenario"],
            "pos_card": chosen_card,
            "neg_card": rejected_card,
            "pos_is_skip": False,
            "neg_is_skip": False,
            "weight": weight,
        }
    yield {
        "scenario": record["scenario"],
        "pos_card": chosen_card,
        "neg_card": None,
        "pos_is_skip": False,
        "neg_is_skip": True,
        "weight": weight,
    }


def pair_count_for_scenario_record(record):
    return sum(1 for _ in iter_pairwise_from_scenario_record(record))


def ensure_cache(args):
    cache_dir = Path(args.cache_dir)
    meta_path = cache_dir / "meta.json"

    if not args.rebuild_cache and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if int(meta.get("cache_format_version", 0) or 0) == CACHE_FORMAT_VERSION:
            train_shards = [cache_dir / name for name in meta.get("train_shards", [])]
            valid_shards = [cache_dir / name for name in meta.get("valid_shards", [])]
            if train_shards and all(path.exists() for path in train_shards) and all(path.exists() for path in valid_shards):
                print("Using cache from {}".format(cache_dir), flush=True)
                return train_shards, valid_shards, meta
        print("Rebuilding cache at {} ...".format(cache_dir), flush=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train_count = 0
    valid_count = 0
    train_pairs = 0
    valid_pairs = 0
    processed = 0
    train_shards = []
    valid_shards = []
    train_shard_counts = []
    valid_shard_counts = []
    train_pair_buffer = []
    valid_pair_buffer = []

    print("Building scenario cache at {} ...".format(cache_dir), flush=True)
    def flush_pair_buffer(split_name, pair_buffer, shard_names, shard_counts):
        if not pair_buffer:
            return
        shard_name = "{}_pairs_{:05d}.pt".format(split_name, len(shard_names))
        shard_path = cache_dir / shard_name
        tensors = pairwise_batch_to_tensors(pair_buffer, device="cpu")
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
        pair_buffer.clear()

    for record in load_run_scenarios(
        args.input_dir,
        max_files=args.max_files,
        max_scenarios=args.max_scenarios,
        file_progress_every=args.file_progress_every,
    ):
        record = {
            "scenario": record["scenario"],
            "chosen_card": record.get("chosen_card"),
            "skip_choice": bool(record.get("skip_choice")),
            "weight": scenario_weight(record),
        }
        pairs = list(iter_pairwise_from_scenario_record(record))
        pair_count = len(pairs)
        if rng.random() < args.valid_fraction:
            valid_count += 1
            valid_pairs += pair_count
            valid_pair_buffer.extend(pairs)
            if len(valid_pair_buffer) >= args.tensor_cache_pairs_per_shard:
                flush_pair_buffer("valid", valid_pair_buffer, valid_shards, valid_shard_counts)
        else:
            train_count += 1
            train_pairs += pair_count
            train_pair_buffer.extend(pairs)
            if len(train_pair_buffer) >= args.tensor_cache_pairs_per_shard:
                flush_pair_buffer("train", train_pair_buffer, train_shards, train_shard_counts)
        processed += 1
        if processed % max(1000, args.file_progress_every) == 0:
            print(
                "Cached scenarios={} train={} valid={} train_pairs={} valid_pairs={} train_shards={} valid_shards={}".format(
                    processed,
                    train_count,
                    valid_count,
                    train_pairs,
                    valid_pairs,
                    len(train_shards),
                    len(valid_shards),
                ),
                flush=True,
            )

    flush_pair_buffer("train", train_pair_buffer, train_shards, train_shard_counts)
    flush_pair_buffer("valid", valid_pair_buffer, valid_shards, valid_shard_counts)

    if processed == 0:
        raise RuntimeError("No usable card reward scenarios were built from {}".format(args.input_dir))

    meta = {
        "source": "slay_the_data_pairwise_preference",
        "cache_format_version": CACHE_FORMAT_VERSION,
        "train_scenarios": train_count,
        "valid_scenarios": valid_count,
        "train_pairs": train_pairs,
        "valid_pairs": valid_pairs,
        "train_shards": train_shards,
        "valid_shards": valid_shards,
        "train_shard_counts": train_shard_counts,
        "valid_shard_counts": valid_shard_counts,
        "input_dir": str(args.input_dir),
        "seed": args.seed,
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(
        "Cache ready  train_scenarios={} valid_scenarios={} train_pairs={} valid_pairs={}".format(
            train_count, valid_count, train_pairs, valid_pairs
        ),
        flush=True,
    )
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


def train_model(
    train_shard_paths,
    valid_shard_paths,
    train_pair_count,
    valid_pair_count,
    train_shard_counts,
    device,
    epochs,
    batch_size,
    learning_rate,
    seed,
    output_path,
    checkpoint_dir,
    batch_progress_every=200,
):
    model = CardRewardPolicyNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    best_state = None
    best_valid = None
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(
            "Epoch {}/{} start  train_pairs={}  valid_pairs={}".format(
                epoch, epochs, train_pair_count, valid_pair_count
            ),
            flush=True,
        )
        model.train()
        total_loss = 0.0
        total = 0
        total_batches = total_batches_for_shards(train_shard_counts, batch_size)
        for batch_index, (batch, batch_len) in enumerate(
            stream_pairwise_batches(
                train_shard_paths,
                batch_size=batch_size,
                device=device,
                seed=seed + epoch,
                shuffle=True,
            ),
            1,
        ):
            pos, neg = option_scores_from_pairwise_batch(model, batch)
            loss = (F.softplus(-(pos - neg)) * batch["weight"]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_len
            total += batch_len
            if batch_progress_every > 0 and (
                batch_index % batch_progress_every == 0 or batch_index == total_batches
            ):
                print(
                    "Epoch {}/{}  batch {}/{}  mean_loss={:.4f}".format(
                        epoch,
                        epochs,
                        batch_index,
                        total_batches,
                        total_loss / float(total or 1),
                    ),
                    flush=True,
                )

        eval_batch_size = max(64, min(4096, batch_size * 8))
        train_metrics = evaluate_pairwise(model, train_shard_paths, device, batch_size=eval_batch_size)
        valid_metrics = evaluate_pairwise(model, valid_shard_paths, device, batch_size=eval_batch_size) if valid_pair_count > 0 else {"loss": 0.0, "accuracy": 0.0}
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
            "Epoch {}/{} done  train_loss={:.4f}  train_acc={:.4f}  valid_loss={:.4f}  valid_acc={:.4f}".format(
                epoch,
                epochs,
                total_loss / float(total or 1),
                train_metrics["accuracy"],
                valid_metrics["loss"],
                valid_metrics["accuracy"],
            ),
            flush=True,
        )
        epoch_summary = {
            "source": "slay_the_data_pairwise_preference",
            "train_pairs": train_pair_count,
            "valid_pairs": valid_pair_count,
            "history": list(history),
            "last_epoch": epoch,
        }
        save_card_reward_checkpoint(model, str(checkpoint_dir / "latest.pt"), training_summary=epoch_summary)
        save_card_reward_checkpoint(model, str(checkpoint_dir / "epoch_{:03d}.pt".format(epoch)), training_summary=epoch_summary)
        current_valid = (valid_metrics["accuracy"], -valid_metrics["loss"])
        if best_valid is None or current_valid > best_valid:
            best_valid = current_valid
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            save_card_reward_checkpoint(model, str(checkpoint_dir / "best.pt"), training_summary=epoch_summary)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def main():
    args = parse_args()
    train_shard_paths, valid_shard_paths, meta = ensure_cache(args)
    train_scenarios = int(meta["train_scenarios"])
    valid_scenarios = int(meta["valid_scenarios"])
    train_pairs = int(meta["train_pairs"])
    valid_pairs = int(meta["valid_pairs"])
    checkpoint_dir = args.checkpoint_dir or os.path.splitext(os.path.abspath(args.output))[0] + "_checkpoints"

    model, history = train_model(
        train_shard_paths=train_shard_paths,
        valid_shard_paths=valid_shard_paths,
        train_pair_count=train_pairs,
        valid_pair_count=valid_pairs,
        train_shard_counts=meta.get("train_shard_counts", []),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_path=args.output,
        checkpoint_dir=checkpoint_dir,
        batch_progress_every=args.batch_progress_every,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_card_reward_checkpoint(
        model,
        args.output,
        training_summary={
            "source": "slay_the_data_pairwise_preference",
            "scenario_count": train_scenarios + valid_scenarios,
            "train_scenarios": train_scenarios,
            "valid_scenarios": valid_scenarios,
            "train_pairs": train_pairs,
            "valid_pairs": valid_pairs,
            "cache_dir": str(args.cache_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "history": history,
        },
    )

    eval_batch_size = max(64, min(4096, args.batch_size * 8))
    train_metrics = evaluate_pairwise(model, train_shard_paths, args.device, batch_size=eval_batch_size)
    valid_metrics = evaluate_pairwise(model, valid_shard_paths, args.device, batch_size=eval_batch_size) if valid_pairs > 0 else {"loss": 0.0, "accuracy": 0.0}

    print("Loaded scenarios: {}".format(train_scenarios + valid_scenarios))
    print("Train scenarios: {}  Valid scenarios: {}".format(train_scenarios, valid_scenarios))
    print("Train pairs: {}  Valid pairs: {}".format(train_pairs, valid_pairs))
    print("Train pairwise loss: {:.4f}  accuracy: {:.4f}".format(train_metrics["loss"], train_metrics["accuracy"]))
    print("Valid pairwise loss: {:.4f}  accuracy: {:.4f}".format(valid_metrics["loss"], valid_metrics["accuracy"]))
    print("Saved card reward model to {}".format(os.path.abspath(args.output)))
    print("Epoch checkpoints saved to {}".format(os.path.abspath(checkpoint_dir)))
    print("Suggested runtime env:")
    print("  SPIRECOMM_CARD_REWARD_MODEL_PATH={}".format(os.path.abspath(args.output)))


if __name__ == "__main__":
    main()
