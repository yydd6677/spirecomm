import glob
import json
import math
import os
import random

from spirecomm.ai.observation import (
    EXTENDED_COMBAT_OBSERVATION_VERSION,
    LEGACY_COMBAT_OBSERVATION_VERSION,
    canonicalize_serialized_state,
    legacy_monster_power_alias,
)
from spirecomm.ai.torch_compat import F, nn, torch
from spirecomm.communication.action import EndTurnAction, PlayCardAction
from spirecomm.spire.card import CardType


MAX_HAND_SIZE = 10
MAX_MONSTERS = 7
MAX_RELICS = 16

ACTION_END_TURN = 0
ACTION_PLAY_OFFSET = 1
ACTION_DIM = ACTION_PLAY_OFFSET + MAX_HAND_SIZE
TARGET_DIM = MAX_MONSTERS

CARD_ID_BUCKETS = 2048
MONSTER_ID_BUCKETS = 512
RELIC_ID_BUCKETS = 512
ROOM_WEIGHT_STEP = 0.01
ACT_WEIGHT_STEP = 0.20
SOURCE_SAMPLE_WEIGHTS = {
    "UndoPlannerCombatPolicy": 1.0,
    "CheckpointCombatPolicy": 0.3,
    "RuleBasedCombatPolicy": 0.45,
    "fallback": 0.1,
    "fallback_state_refresh": 0.05,
}
NEOWS_LAMENT_IDS = {"NeowsBlessing"}
NEOWS_LAMENT_NAMES = {"Neow's Lament"}
INTENT_NAMES = [
    "ATTACK",
    "ATTACK_BUFF",
    "ATTACK_DEBUFF",
    "ATTACK_DEFEND",
    "BUFF",
    "DEBUFF",
    "STRONG_DEBUFF",
    "DEBUG",
    "DEFEND",
    "DEFEND_DEBUFF",
    "DEFEND_BUFF",
    "ESCAPE",
    "MAGIC",
    "NONE",
    "SLEEP",
    "STUN",
    "UNKNOWN",
]
INTENT_TO_INDEX = {name: index + 1 for index, name in enumerate(INTENT_NAMES)}

CARD_TYPE_NAMES = ["ATTACK", "SKILL", "POWER", "STATUS", "CURSE"]
CARD_RARITY_NAMES = ["BASIC", "COMMON", "UNCOMMON", "RARE", "SPECIAL", "CURSE"]
X_COST_CARD_IDS = {
    "Whirlwind",
    "Skewer",
    "Malaise",
    "Doppelganger",
    "Conjure Blade",
    "Reinforced Body",
    "Multi-Cast",
    "Tempest",
}
AOE_CARD_IDS = {
    "Whirlwind",
    "Cleave",
    "Thunderclap",
    "Immolate",
    "Reaper",
    "Dagger Spray",
    "Die Die Die",
    "Crippling Poison",
    "Corpse Explosion",
    "All Out Attack",
    "Sweeping Beam",
    "Hyperbeam",
    "Beam Cell",
    "Electrodynamics",
}
DRAW_CARD_IDS = {
    "Shrug It Off",
    "Pommel Strike",
    "Battle Trance",
    "Burning Pact",
    "Backflip",
    "Acrobatics",
    "Coolheaded",
    "Compile Driver",
    "Sweeping Beam",
    "Skim",
    "Seek",
    "Master of Strategy",
}
BLOCK_CARD_IDS = {
    "Defend_R",
    "Defend_G",
    "Defend_B",
    "Shrug It Off",
    "Flame Barrier",
    "Ghostly Armor",
    "Impervious",
    "Entrench",
    "Blur",
    "Backflip",
    "Leg Sweep",
    "Dodge and Roll",
    "Dash",
    "Glacier",
    "Leap",
    "Charge Battery",
    "Reinforced Body",
    "BootSequence",
}
STRENGTH_CARD_IDS = {
    "Inflame",
    "Spot Weakness",
    "Demon Form",
    "Limit Break",
    "Flex",
    "J.A.X.",
}
IRONCLAD_BUFF_SETUP_CARD_IDS = {
    "Rage",
    "Barricade",
    "Berserk",
    "Brutality",
    "Combust",
    "Corruption",
    "Dark Embrace",
    "Double Tap",
    "Evolve",
    "Feel No Pain",
    "Fire Breathing",
    "Flame Barrier",
    "Juggernaut",
    "Metallicize",
    "Rupture",
}
VULNERABLE_CARD_IDS = {
    "Bash",
    "Thunderclap",
    "Uppercut",
    "Shockwave",
    "Trip",
    "Crush Joints",
}
WEAK_CARD_IDS = {
    "Clothesline",
    "Intimidate",
    "Uppercut",
    "Shockwave",
    "Leg Sweep",
    "Neutralize",
    "Go for the Eyes",
    "Sucker Punch",
    "Malaise",
}
STRENGTH_DOWN_CARD_IDS = {
    "Disarm",
}
DEXTERITY_CARD_IDS = {
    "Footwork",
}
FRAIL_CARD_IDS = {
}
ENERGY_CARD_IDS = {
    "Seeing Red",
    "Offering",
    "Berserk",
    "Adrenaline",
    "Concentrate",
    "Outmaneuver",
    "Double Energy",
    "Aggregate",
    "Turbo",
    "Meteor Strike",
}
PLAYER_POWER_SPECS = [
    ("Strength", 10.0, True),
    ("Dexterity", 10.0, True),
    ("Vulnerable", 5.0, False),
    ("Weakened", 5.0, False),
    ("Frail", 5.0, False),
    ("Artifact", 5.0, False),
    ("Metallicize", 10.0, False),
    ("Rage", 10.0, False),
    ("Barricade", 1.0, False),
    ("Demon Form", 5.0, False),
    ("Berserk", 3.0, False),
    ("Flame Barrier", 16.0, False),
    ("Thorns", 10.0, False),
    ("Plated Armor", 10.0, False),
    ("No Draw", 1.0, False),
]
MONSTER_POWER_SPECS = [
    ("Strength", 12.0, True),
    ("Dexterity", 10.0, True),
    ("Vulnerable", 5.0, False),
    ("Weakened", 5.0, False),
    ("Frail", 5.0, False),
    ("Artifact", 5.0, False),
    ("Metallicize", 10.0, False),
    ("Ritual", 8.0, False),
    ("Angry", 8.0, False),
    ("Anger", 12.0, False),
    ("Sharp Hide", 10.0, False),
    ("Thorns", 10.0, False),
    ("Curl Up", 16.0, False),
    ("Mode Shift", 20.0, False),
    ("Regenerate", 10.0, False),
    ("Flight", 4.0, False),
    ("Malleable", 8.0, False),
    ("Plated Armor", 10.0, False),
]

DEFAULT_REWARD_WEIGHTS = {
    "damage_dealt": 0.03,
    "monster_kill": 0.75,
    "hp_loss": -0.25,
    "effective_block": 0.10,
    "invalid_block": -0.16,
    "low_incoming_invalid_block": -0.12,
    "incoming_damage_reduction": 0.14,
    "strength_damage_bonus": 0.14,
    "dexterity_block_bonus": 0.14,
    "enemy_vulnerable_damage_bonus": 0.14,
    "enemy_weak_damage_reduction_bonus": 0.12,
    "enemy_frail_block_reduction_bonus": 0.08,
    "enemy_strength_down_damage_reduction_bonus": 0.12,
    "enemy_dexterity_down_damage_bonus": 0.12,
    "buff_setup_bonus": 0.20,
    "energy_spent": -0.04,
    "damage_efficiency_bonus": 0.04,
    "block_efficiency_bonus": 0.06,
    "buff_efficiency_bonus": 0.08,
    "gold_gain": 0.02,
    "relic_gain": 2.5,
    "potion_gain": 0.35,
    "potion_use": -0.10,
    "combat_win": 8.0,
    "death": -12.0,
}

GLOBAL_FEATURE_DIM = 20 + len(PLAYER_POWER_SPECS)
DECK_FEATURE_DIM = 16
HAND_FEATURE_DIM = 33
MONSTER_FEATURE_DIM = 12 + len(MONSTER_POWER_SPECS)
RETURN_DISCOUNT = 0.97
PERSISTENT_RETURN_HORIZON = 12
SETUP_RETURN_DISCOUNT = 0.70
SETUP_RETURN_HORIZON = 5
SKILL_RETURN_DISCOUNT = 0.30
SKILL_RETURN_HORIZON = 2
ATTACK_RETURN_DISCOUNT = 0.30
ATTACK_RETURN_HORIZON = 2
END_RETURN_DISCOUNT = 0.40
END_RETURN_HORIZON = 1


def stable_bucket(text, num_buckets):
    if not text:
        return 0
    value = 0
    for index, char in enumerate(text):
        value = (value * 131 + (index + 17) * ord(char)) % num_buckets
    return value + 1


def card_matches(card, keys):
    if card is None:
        return False
    return card.get("card_id") in keys or card.get("name") in keys


def is_setup_card(card):
    if card is None:
        return False
    return any(
        card_matches(card, keys)
        for keys in (
            STRENGTH_CARD_IDS,
            IRONCLAD_BUFF_SETUP_CARD_IDS,
            DEXTERITY_CARD_IDS,
            VULNERABLE_CARD_IDS,
            WEAK_CARD_IDS,
            FRAIL_CARD_IDS,
            STRENGTH_DOWN_CARD_IDS,
        )
    )


def safe_ratio(numerator, denominator):
    if denominator in [0, None]:
        return 0.0
    return float(numerator) / float(denominator)


def clamp_scale(value, scale, minimum=0.0, maximum=1.0):
    if value is None:
        value = 0.0
    scaled = float(value) / float(scale)
    if scaled < minimum:
        return minimum
    if scaled > maximum:
        return maximum
    return scaled


def clamp_signed_scale(value, scale):
    if value is None:
        value = 0.0
    scaled = float(value) / float(scale)
    if scaled < -1.0:
        return -1.0
    if scaled > 1.0:
        return 1.0
    return scaled


def sum_power_amounts(powers):
    total = 0.0
    positive = 0.0
    negative = 0.0
    for power in powers or []:
        amount = power.get("amount", 0) or 0
        total += amount
        if amount >= 0:
            positive += amount
        else:
            negative += abs(amount)
    return total, positive, negative


def power_amounts_by_id(powers):
    amounts = {}
    for power in powers or []:
        power_id = power.get("power_id") or power.get("id") or power.get("name")
        if not power_id:
            continue
        power_id = legacy_monster_power_alias(str(power_id))
        amounts[power_id] = amounts.get(power_id, 0.0) + float(power.get("amount", 0) or 0.0)
    return amounts


def power_amount_for_keys(powers, keys):
    amounts = power_amounts_by_id(powers)
    return sum(float(amounts.get(key, 0.0)) for key in keys)


def encode_power_features(powers, power_specs):
    power_amounts = power_amounts_by_id(powers)
    encoded = []
    for power_id, scale, signed in power_specs:
        amount = power_amounts.get(power_id, 0.0)
        if power_id == "Anger" and amount == 0.0:
            amount = power_amounts.get("Angry", 0.0)
        if signed:
            encoded.append(clamp_signed_scale(amount, scale))
        else:
            encoded.append(clamp_scale(amount, scale))
    return encoded


def one_hot(name, vocabulary):
    return [1.0 if entry == name else 0.0 for entry in vocabulary]


def count_real_potions(potions):
    return sum(1 for potion in potions or [] if potion.get("potion_id") != "Potion Slot")


def living_monster_count(serialized_state):
    monsters = serialized_state.get("combat_state", {}).get("monsters", [])
    count = 0
    for monster in monsters:
        if monster.get("current_hp", 0) > 0 and not monster.get("half_dead") and not monster.get("is_gone"):
            count += 1
    return count


def total_incoming_damage(serialized_state):
    serialized_state = serialized_state or {}
    combat_state = serialized_state.get("combat_state") or {}
    monsters = combat_state.get("monsters", [])
    total = 0.0
    for monster in monsters:
        if monster.get("current_hp", 0) <= 0 or monster.get("half_dead") or monster.get("is_gone"):
            continue
        hits = monster.get("move_hits", 0) or 0
        damage = monster.get("move_adjusted_damage", 0) or 0
        total += hits * damage
    return total


def combat_state_before(record):
    return (record.get("state_before") or {}).get("combat_state") or {}


def combat_state_after(record):
    return (record.get("state_after") or {}).get("combat_state") or {}


def player_before(record):
    return combat_state_before(record).get("player") or {}


def player_after(record):
    return combat_state_after(record).get("player") or {}


def living_monsters_from_combat_state(combat_state):
    monsters = combat_state.get("monsters") or []
    return [
        monster
        for monster in monsters
        if monster.get("current_hp", 0) > 0 and not monster.get("half_dead") and not monster.get("is_gone")
    ]


def player_power_amount(record, power_id, after=False):
    player = player_after(record) if after else player_before(record)
    aliases = {
        "Weak": ("Weak", "Weakened"),
    }
    keys = aliases.get(power_id, (power_id,))
    return float(power_amount_for_keys(player.get("powers"), keys))


def player_power_gain(record, power_id):
    return max(0.0, player_power_amount(record, power_id, after=True) - player_power_amount(record, power_id, after=False))


def total_monster_power_amount(record, power_id, after=False):
    combat_state = combat_state_after(record) if after else combat_state_before(record)
    aliases = {
        "Weak": ("Weak", "Weakened"),
    }
    keys = aliases.get(power_id, (power_id,))
    total = 0.0
    for monster in living_monsters_from_combat_state(combat_state):
        total += float(power_amount_for_keys(monster.get("powers"), keys))
    return total


def monster_power_gain(record, power_id):
    return max(0.0, total_monster_power_amount(record, power_id, after=True) - total_monster_power_amount(record, power_id, after=False))


def monster_power_reduction(record, power_id):
    return max(0.0, total_monster_power_amount(record, power_id, after=False) - total_monster_power_amount(record, power_id, after=True))


def monster_power_coverage_ratio(record, power_id, positive=True, after=False):
    combat_state = combat_state_after(record) if after else combat_state_before(record)
    aliases = {
        "Weak": ("Weak", "Weakened"),
    }
    keys = aliases.get(power_id, (power_id,))
    monsters = living_monsters_from_combat_state(combat_state)
    if not monsters:
        return 0.0
    covered = 0.0
    for monster in monsters:
        amount = float(power_amount_for_keys(monster.get("powers"), keys))
        if positive and amount > 0.0:
            covered += 1.0
        elif not positive and amount < 0.0:
            covered += 1.0
    return covered / float(len(monsters))


def total_monster_block(combat_state):
    return sum(float(monster.get("block", 0) or 0.0) for monster in living_monsters_from_combat_state(combat_state))


def playable_cards_after(record):
    hand = combat_state_after(record).get("hand") or []
    return [card for card in hand if card and card.get("is_playable")]


def playable_attack_count_after(record):
    return sum(1 for card in playable_cards_after(record) if card.get("type") == "ATTACK")


def playable_block_count_after(record):
    return sum(1 for card in playable_cards_after(record) if is_defensive_card_payload(card))


def strength_damage_bonus(record, damage_dealt):
    if damage_dealt <= 0.0:
        return 0.0
    return min(damage_dealt, max(0.0, player_power_amount(record, "Strength", after=False)))


def dexterity_block_bonus(record, effective_block):
    if effective_block <= 0.0:
        return 0.0
    return min(effective_block, max(0.0, player_power_amount(record, "Dexterity", after=False)))


def enemy_vulnerable_damage_bonus(record, damage_dealt):
    if damage_dealt <= 0.0:
        return 0.0
    coverage = monster_power_coverage_ratio(record, "Vulnerable", positive=True, after=False)
    return damage_dealt * coverage / 3.0


def enemy_weak_damage_reduction_bonus(record, incoming_damage_reduction):
    combat_before = combat_state_before(record)
    monsters = living_monsters_from_combat_state(combat_before)
    prevented = 0.0
    for monster in monsters:
        weak_amount = power_amount_for_keys(monster.get("powers"), ("Weak", "Weakened"))
        if weak_amount <= 0.0:
            continue
        hits = float(monster.get("move_hits", 0) or 0.0)
        adjusted = float(monster.get("move_adjusted_damage", 0) or 0.0)
        if hits <= 0.0 or adjusted <= 0.0:
            continue
        prevented += (adjusted / 3.0) * hits
    return prevented


def enemy_strength_down_damage_reduction_bonus(record, incoming_damage_reduction):
    if incoming_damage_reduction <= 0.0:
        return 0.0
    coverage = monster_power_coverage_ratio(record, "Strength", positive=False, after=False)
    return incoming_damage_reduction * coverage


def enemy_dexterity_down_damage_bonus(record, damage_dealt):
    combat_before = combat_state_before(record)
    combat_after = combat_state_after(record)
    if damage_dealt <= 0.0 or not combat_before or not combat_after:
        return 0.0
    block_before = total_monster_block(combat_before)
    block_after = total_monster_block(combat_after)
    actual_block_gain = max(0.0, block_after - block_before)
    coverage = monster_power_coverage_ratio(record, "Dexterity", positive=False, after=False)
    if actual_block_gain <= 0.0 or coverage <= 0.0:
        return 0.0
    return (actual_block_gain / 3.0) * coverage


def enemy_frail_block_reduction_bonus(record):
    combat_before = combat_state_before(record)
    combat_after = combat_state_after(record)
    if not combat_before or not combat_after:
        return 0.0
    block_before = total_monster_block(combat_before)
    block_after = total_monster_block(combat_after)
    actual_block_gain = max(0.0, block_after - block_before)
    if actual_block_gain <= 0.0:
        return 0.0
    coverage = monster_power_coverage_ratio(record, "Frail", positive=True, after=False)
    if coverage <= 0.0:
        return 0.0
    return (actual_block_gain / 3.0) * coverage


def buff_setup_bonus(record):
    attack_followup = min(3.0, float(playable_attack_count_after(record)))
    block_followup = min(3.0, float(playable_block_count_after(record)))
    state_after = record.get("state_after") or {}
    if state_after is None:
        state_after = {}
    incoming_after = max(0.0, float(total_incoming_damage(state_after)))
    defensive_followup = min(3.0, 0.5 + incoming_after / 8.0)
    generic_followup = max(attack_followup, block_followup, defensive_followup)

    bonus = 0.0
    bonus += player_power_gain(record, "Strength") * attack_followup
    bonus += player_power_gain(record, "Dexterity") * block_followup
    bonus += monster_power_gain(record, "Vulnerable") * attack_followup
    bonus += monster_power_gain(record, "Weak") * defensive_followup
    bonus += monster_power_gain(record, "Frail") * attack_followup
    bonus += monster_power_reduction(record, "Strength") * defensive_followup
    bonus += monster_power_reduction(record, "Dexterity") * attack_followup
    bonus += player_power_gain(record, "Rage") * attack_followup
    bonus += player_power_gain(record, "Double Tap") * attack_followup
    bonus += player_power_gain(record, "Flame Barrier") * defensive_followup
    bonus += player_power_gain(record, "Barricade") * block_followup
    bonus += player_power_gain(record, "Metallicize") * defensive_followup
    bonus += player_power_gain(record, "Feel No Pain") * generic_followup
    bonus += player_power_gain(record, "Juggernaut") * generic_followup
    bonus += player_power_gain(record, "Berserk") * generic_followup
    bonus += player_power_gain(record, "Brutality") * generic_followup
    bonus += player_power_gain(record, "Combust") * generic_followup
    bonus += player_power_gain(record, "Corruption") * generic_followup
    bonus += player_power_gain(record, "Dark Embrace") * generic_followup
    bonus += player_power_gain(record, "Evolve") * generic_followup
    bonus += player_power_gain(record, "Fire Breathing") * generic_followup
    bonus += player_power_gain(record, "Rupture") * generic_followup
    return bonus


def effective_block_gain(record):
    state_before = record.get("state_before") or {}
    player_before = (state_before.get("combat_state") or {}).get("player") or {}
    block_before = max(0.0, float(player_before.get("block", 0) or 0.0))
    incoming_before = incoming_damage_before(record)

    delta = record.get("delta") or {}
    raw_block_gain = max(0.0, float(delta.get("player_block_delta") or 0.0))

    covered_before = min(block_before, incoming_before)
    covered_after = min(block_before + raw_block_gain, incoming_before)
    return max(0.0, covered_after - covered_before)


def is_defensive_card_payload(card):
    if not card:
        return False
    card_id = card.get("card_id")
    name = card.get("name")
    return card_id in BLOCK_CARD_IDS or name in BLOCK_CARD_IDS


def only_defensive_playables(record):
    state_before = record.get("state_before") or {}
    hand = ((state_before.get("combat_state") or {}).get("hand") or [])
    playable_cards = [card for card in hand if card and card.get("is_playable")]
    if not playable_cards:
        return False
    return all(is_defensive_card_payload(card) for card in playable_cards)


def invalid_block_amount(record):
    if only_defensive_playables(record):
        return 0.0
    delta = record.get("delta") or {}
    raw_block_gain = max(0.0, float(delta.get("player_block_delta") or 0.0))
    wasted_block = max(0.0, raw_block_gain - effective_block_gain(record))
    if wasted_block < 5.0:
        return 0.0
    return wasted_block


def incoming_damage_before(record):
    state_before = record.get("state_before") or {}
    return max(0.0, float(total_incoming_damage(state_before)))


def low_incoming_invalid_block_amount(record):
    invalid_block = invalid_block_amount(record)
    if invalid_block <= 0.0:
        return 0.0
    if incoming_damage_before(record) > 5.0:
        return 0.0
    return invalid_block


def room_progress_in_act(floor):
    floor = int(floor or 0)
    if floor <= 0:
        return 0
    return ((floor - 1) % 17) + 1


def has_neows_lament(meta_record):
    combat = (meta_record or {}).get("combat") or {}
    initial_state = combat.get("initial_state") or {}
    relics = initial_state.get("relics") or []
    relic_ids = {relic.get("relic_id") for relic in relics}
    relic_names = {relic.get("name") for relic in relics}
    return bool(relic_ids.intersection(NEOWS_LAMENT_IDS) or relic_names.intersection(NEOWS_LAMENT_NAMES))


def should_skip_combat(meta_record):
    combat = (meta_record or {}).get("combat") or {}
    combat_index = int(combat.get("combat_index") or 0)
    return has_neows_lament(meta_record) and combat_index < 3


def compute_combat_sample_weight(meta_record):
    combat = (meta_record or {}).get("combat") or {}
    initial_state = combat.get("initial_state") or {}
    act = int(combat.get("act") or initial_state.get("act") or 1)
    floor = int(combat.get("floor") or initial_state.get("floor") or 0)
    room_weight = 1.0 + ROOM_WEIGHT_STEP * max(room_progress_in_act(floor) - 1, 0)
    act_weight = 1.0 + ACT_WEIGHT_STEP * max(act - 1, 0)
    return room_weight * act_weight


def action_payload_command(action_payload):
    return (action_payload or {}).get("command")


def playable_cards_before(record):
    hand = ((combat_state_before(record)).get("hand") or [])
    return [card for card in hand if card and card.get("is_playable")]


def playable_attack_count_before(record):
    return sum(1 for card in playable_cards_before(record) if card.get("type") == "ATTACK")


def current_energy_before(record):
    player = player_before(record)
    return max(0.0, float(player.get("energy", 0) or 0.0))


def bad_end_penalty_multiplier(record):
    action_payload = record.get("action") or {}
    if action_payload_command(action_payload) != "end":
        return 1.0
    playable_cards = playable_cards_before(record)
    if not playable_cards:
        return 1.0
    multiplier = 0.4
    if current_energy_before(record) > 0.0:
        multiplier *= 0.35
    if playable_attack_count_before(record) > 0:
        multiplier *= 0.35
    return multiplier


def source_sample_weight(source):
    if not source:
        return 0.2
    return SOURCE_SAMPLE_WEIGHTS.get(source, 0.6)


def compute_transition_sample_weight(record):
    multiplier = source_sample_weight(record.get("source"))
    invalid_block = invalid_block_amount(record)
    if invalid_block > 0.0:
        multiplier *= 0.25
        if incoming_damage_before(record) <= 5.0:
            multiplier *= 0.4
    multiplier *= bad_end_penalty_multiplier(record)
    return multiplier


def preference_action_weight(state_before, action_payload, source):
    weight = source_sample_weight(source)
    command = action_payload_command(action_payload)
    if command != "end":
        return weight
    combat_state = (state_before or {}).get("combat_state") or {}
    player = combat_state.get("player") or {}
    hand = combat_state.get("hand") or []
    playable_cards = [card for card in hand if card and card.get("is_playable")]
    if not playable_cards:
        return weight
    weight *= 0.4
    if float(player.get("energy", 0) or 0.0) > 0.0:
        weight *= 0.35
    if any(card.get("type") == "ATTACK" for card in playable_cards):
        weight *= 0.35
    return weight


def weighted_mean(values, weights):
    weights = weights.float()
    denominator = weights.sum().clamp_min(1e-6)
    return (values * weights).sum() / denominator


def weighted_accuracy(predictions, targets, weights):
    correct = (predictions == targets).float()
    return weighted_mean(correct, weights)


def deck_summary_features(deck):
    type_counts = dict((name, 0.0) for name in CARD_TYPE_NAMES)
    rarity_counts = dict((name, 0.0) for name in CARD_RARITY_NAMES)
    total_cost = 0.0
    cost_cards = 0.0
    upgraded_cards = 0.0
    exhaust_cards = 0.0
    targeted_cards = 0.0

    for card in deck:
        card_type = card.get("type")
        rarity = card.get("rarity")
        if card_type in type_counts:
            type_counts[card_type] += 1.0
        if rarity in rarity_counts:
            rarity_counts[rarity] += 1.0
        cost = card.get("cost")
        if cost is not None and cost >= 0:
            total_cost += cost
            cost_cards += 1.0
        if (card.get("upgrades") or 0) > 0:
            upgraded_cards += 1.0
        if card.get("exhausts"):
            exhaust_cards += 1.0
        if card.get("has_target"):
            targeted_cards += 1.0

    deck_size = float(len(deck))
    features = [
        clamp_scale(deck_size, 40.0),
        clamp_scale(upgraded_cards, 12.0),
        clamp_scale(exhaust_cards, 12.0),
        clamp_scale(targeted_cards, 20.0),
        clamp_scale(total_cost / max(cost_cards, 1.0), 3.0),
    ]
    features.extend(clamp_scale(type_counts[name], 20.0) for name in CARD_TYPE_NAMES)
    features.extend(clamp_scale(rarity_counts[name], 12.0) for name in CARD_RARITY_NAMES)
    return features


def card_slot_features(card):
    if card is None:
        return [0.0] * HAND_FEATURE_DIM

    cost = card.get("cost", 0)
    is_x_cost = 1.0 if cost is not None and cost < 0 else 0.0
    cost_value = max(cost or 0, 0)
    is_attack = 1.0 if card.get("type") == "ATTACK" else 0.0
    is_skill = 1.0 if card.get("type") == "SKILL" else 0.0
    is_power = 1.0 if card.get("type") == "POWER" else 0.0
    is_aoe = 1.0 if card_matches(card, AOE_CARD_IDS) else 0.0
    features = [
        1.0,
        1.0 if card.get("is_playable") else 0.0,
        1.0 if card.get("has_target") else 0.0,
        1.0 if card.get("exhausts") else 0.0,
        1.0 if (card.get("upgrades") or 0) > 0 else 0.0,
        clamp_scale(cost_value, 3.0),
        clamp_scale(card.get("upgrades", 0) or 0, 3.0),
        clamp_signed_scale(card.get("misc", 0) or 0, 60.0),
        is_x_cost,
        1.0 if cost_value == 0 and not is_x_cost else 0.0,
        1.0 if cost_value >= 2 else 0.0,
        is_aoe,
        1.0 if is_aoe and is_attack else 0.0,
        1.0 if card_matches(card, DRAW_CARD_IDS) else 0.0,
        1.0 if card_matches(card, BLOCK_CARD_IDS) else 0.0,
        1.0 if card_matches(card, STRENGTH_CARD_IDS) else 0.0,
        1.0 if card_matches(card, VULNERABLE_CARD_IDS) else 0.0,
        1.0 if card_matches(card, WEAK_CARD_IDS) else 0.0,
        1.0 if card_matches(card, ENERGY_CARD_IDS) else 0.0,
        is_attack,
        is_skill,
        is_power,
    ]
    features.extend(one_hot(card.get("type"), CARD_TYPE_NAMES))
    features.extend(one_hot(card.get("rarity"), CARD_RARITY_NAMES))
    return features


def monster_slot_features(monster):
    if monster is None:
        return [0.0] * MONSTER_FEATURE_DIM

    total_power, positive_power, negative_power = sum_power_amounts(monster.get("powers"))
    features = [
        1.0,
        0.0 if monster.get("is_gone") else 1.0,
        clamp_scale(monster.get("current_hp", 0), 250.0),
        clamp_scale(safe_ratio(monster.get("current_hp", 0), monster.get("max_hp", 1)), 1.0),
        clamp_scale(monster.get("block", 0), 40.0),
        clamp_scale(monster.get("move_adjusted_damage", 0), 30.0),
        clamp_scale(monster.get("move_hits", 0), 5.0),
        1.0 if monster.get("half_dead") else 0.0,
        1.0 if monster.get("is_gone") else 0.0,
        clamp_scale(total_power, 20.0),
        clamp_scale(positive_power, 20.0),
        clamp_scale(negative_power, 20.0),
    ]
    features.extend(encode_power_features(monster.get("powers"), MONSTER_POWER_SPECS))
    return features


def relic_ids(relics):
    ids = [stable_bucket(relic.get("relic_id"), RELIC_ID_BUCKETS) for relic in relics[:MAX_RELICS]]
    while len(ids) < MAX_RELICS:
        ids.append(0)
    return ids


def _build_legacy_combat_tensors(canonical_state):
    serialized_state = canonical_state or {}
    combat_state = serialized_state.get("combat_state") or {}
    player = combat_state.get("player") or {}
    hand = combat_state.get("hand") or []
    monsters = combat_state.get("monsters") or []
    deck = serialized_state.get("deck") or []
    relics = serialized_state.get("relics") or []
    potions = serialized_state.get("potions") or []

    player_total_power, player_positive_power, player_negative_power = sum_power_amounts(player.get("powers"))
    total_monster_hp = sum(monster.get("current_hp", 0) for monster in monsters if not monster.get("is_gone"))

    global_features = [
        clamp_scale(serialized_state.get("act", 0), 4.0),
        clamp_scale(serialized_state.get("floor", 0), 60.0),
        clamp_scale(serialized_state.get("ascension_level", 0), 20.0),
        clamp_scale(serialized_state.get("gold", 0), 500.0),
        clamp_scale(serialized_state.get("current_hp", 0), 120.0),
        clamp_scale(serialized_state.get("max_hp", 0), 120.0),
        clamp_scale(safe_ratio(serialized_state.get("current_hp", 0), serialized_state.get("max_hp", 1)), 1.0),
        clamp_scale(combat_state.get("turn", 0), 20.0),
        clamp_scale(player.get("block", 0), 40.0),
        clamp_scale(player.get("energy", 0), 5.0),
        clamp_scale(combat_state.get("cards_discarded_this_turn", 0), 12.0),
        clamp_scale(len(hand), float(MAX_HAND_SIZE)),
        clamp_scale(len(combat_state.get("draw_pile") or []), 40.0),
        clamp_scale(len(combat_state.get("discard_pile") or []), 40.0),
        clamp_scale(len(combat_state.get("exhaust_pile") or []), 20.0),
        clamp_scale(living_monster_count(serialized_state), float(MAX_MONSTERS)),
        clamp_scale(total_monster_hp, 300.0),
        clamp_scale(total_incoming_damage(serialized_state), 60.0),
        clamp_scale(player_positive_power, 20.0),
        clamp_scale(player_negative_power, 20.0),
    ]
    global_features.extend(encode_power_features(player.get("powers"), PLAYER_POWER_SPECS))

    hand_card_ids = []
    hand_features = []
    for slot in range(MAX_HAND_SIZE):
        card = hand[slot] if slot < len(hand) else None
        hand_card_ids.append(stable_bucket(card.get("card_id"), CARD_ID_BUCKETS) if card is not None else 0)
        hand_features.append(card_slot_features(card))

    monster_ids = []
    monster_intents = []
    monster_features = []
    for slot in range(MAX_MONSTERS):
        monster = monsters[slot] if slot < len(monsters) else None
        monster_ids.append(stable_bucket(monster.get("monster_id"), MONSTER_ID_BUCKETS) if monster is not None else 0)
        monster_intents.append(INTENT_TO_INDEX.get(monster.get("intent"), 0) if monster is not None else 0)
        monster_features.append(monster_slot_features(monster))

    return {
        "global_features": global_features,
        "deck_features": deck_summary_features(deck),
        "hand_card_ids": hand_card_ids,
        "hand_features": hand_features,
        "monster_ids": monster_ids,
        "monster_intents": monster_intents,
        "monster_features": monster_features,
        "relic_ids": relic_ids(relics),
        "potion_count": clamp_scale(count_real_potions(potions), 3.0),
    }


def build_legacy_combat_tensors(serialized_state):
    return _build_legacy_combat_tensors(canonicalize_serialized_state(serialized_state))


def build_extended_combat_tensors(serialized_state):
    canonical_state = canonicalize_serialized_state(serialized_state)
    legacy = _build_legacy_combat_tensors(canonical_state)
    combat_state = canonical_state.get("combat_state") or {}
    hand = combat_state.get("hand") or []

    extended_hand_features = []
    for slot in range(MAX_HAND_SIZE):
        card = hand[slot] if slot < len(hand) else None
        if card is None:
            extended_hand_features.append([0.0] * 8)
            continue
        cost = card.get("cost", 0)
        base_cost = card.get("base_cost")
        cost_for_turn = card.get("cost_for_turn")
        cost_for_combat = card.get("cost_for_combat")
        delta = 0.0
        if cost is not None and base_cost is not None:
            delta = float(cost) - float(base_cost)
        extended_hand_features.append([
            1.0,
            clamp_scale(max(base_cost or 0, 0), 4.0),
            clamp_scale(max(cost or 0, 0), 4.0),
            clamp_scale(max(cost_for_turn or 0, 0), 4.0),
            clamp_scale(max(cost_for_combat or 0, 0), 4.0) if cost_for_combat is not None else 0.0,
            1.0 if cost_for_combat is not None else 0.0,
            1.0 if card.get("free_to_play_once") else 0.0,
            clamp_signed_scale(delta, 4.0),
        ])

    legacy.update(
        {
            "observation_version": EXTENDED_COMBAT_OBSERVATION_VERSION,
            "canonical_state": canonical_state,
            "extended_hand_features": extended_hand_features,
        }
    )
    return legacy


def build_state_tensors(serialized_state, observation_version=LEGACY_COMBAT_OBSERVATION_VERSION):
    if observation_version == EXTENDED_COMBAT_OBSERVATION_VERSION:
        return build_extended_combat_tensors(serialized_state)
    return build_legacy_combat_tensors(serialized_state)


def build_action_mask(serialized_state):
    serialized_state = canonicalize_serialized_state(serialized_state)
    combat_state = serialized_state.get("combat_state") or {}
    hand = combat_state.get("hand") or []
    commands = serialized_state.get("commands") or {}

    mask = [False] * ACTION_DIM
    has_playable_cards = False

    if commands.get("play"):
        for slot in range(min(len(hand), MAX_HAND_SIZE)):
            playable = bool(hand[slot].get("is_playable"))
            mask[ACTION_PLAY_OFFSET + slot] = playable
            has_playable_cards = has_playable_cards or playable

    # Hard-disable end turn while any playable card remains in hand.
    mask[ACTION_END_TURN] = bool(commands.get("end")) and not has_playable_cards

    if not any(mask):
        mask[ACTION_END_TURN] = True
    return mask


def build_target_mask(serialized_state):
    serialized_state = canonicalize_serialized_state(serialized_state)
    combat_state = serialized_state.get("combat_state") or {}
    mask = [False] * TARGET_DIM
    monsters = combat_state.get("monsters") or []
    for slot in range(min(len(monsters), MAX_MONSTERS)):
        monster = monsters[slot]
        alive = monster.get("current_hp", 0) > 0 and not monster.get("half_dead") and not monster.get("is_gone")
        mask[slot] = alive
    if not any(mask):
        mask[0] = True
    return mask


def encode_logged_action(serialized_state, action_payload):
    serialized_state = canonicalize_serialized_state(serialized_state)
    command = action_payload.get("command")
    if command == "end":
        return ACTION_END_TURN, None, False
    if command != "play":
        return None, None, False

    card_index = action_payload.get("card_index")
    hand = serialized_state.get("combat_state", {}).get("hand", [])
    if card_index is None or card_index < 0 or card_index >= len(hand) or card_index >= MAX_HAND_SIZE:
        return None, None, False

    card = hand[card_index]
    if not card.get("is_playable"):
        return None, None, False

    target_index = action_payload.get("target_index")
    if target_index is not None and (target_index < 0 or target_index >= MAX_MONSTERS):
        target_index = None

    return ACTION_PLAY_OFFSET + card_index, target_index, bool(card.get("has_target"))


def action_payload_card(state_before, action_payload):
    if (action_payload or {}).get("command") != "play":
        return None
    combat_state = (canonicalize_serialized_state(state_before) or {}).get("combat_state") or {}
    hand = combat_state.get("hand") or []
    card_index = action_payload.get("card_index")
    if card_index is None or card_index < 0 or card_index >= len(hand):
        return None
    return hand[card_index]


def action_energy_cost_from_payload(state_before, action_payload):
    card = action_payload_card(state_before, action_payload or {})
    if card is None:
        return 0.0
    player = ((state_before or {}).get("combat_state") or {}).get("player") or {}
    energy_before = float(player.get("energy", 0) or 0.0)
    cost = card.get("cost")
    if cost is None:
        return 0.0
    if float(cost) < 0.0:
        return max(0.0, energy_before)
    return max(0.0, float(cost))


def reward_efficiency_denominator(energy_spent):
    if energy_spent <= 0.0:
        return 0.2
    return energy_spent


def action_credit_assignment(state_before, action_payload):
    command = action_payload.get("command")
    if command == "end":
        return END_RETURN_DISCOUNT, END_RETURN_HORIZON

    if command != "play":
        return RETURN_DISCOUNT, PERSISTENT_RETURN_HORIZON

    combat_state = (state_before or {}).get("combat_state") or {}
    hand = combat_state.get("hand") or []
    card_index = action_payload.get("card_index")
    if card_index is None or card_index < 0 or card_index >= len(hand):
        return RETURN_DISCOUNT, PERSISTENT_RETURN_HORIZON

    card = hand[card_index]
    card_type = card.get("type")
    if card_type == "POWER":
        return RETURN_DISCOUNT, PERSISTENT_RETURN_HORIZON

    if is_setup_card(card):
        return SETUP_RETURN_DISCOUNT, SETUP_RETURN_HORIZON

    if card_type == "ATTACK":
        return ATTACK_RETURN_DISCOUNT, ATTACK_RETURN_HORIZON

    if card_type == "SKILL":
        return SKILL_RETURN_DISCOUNT, SKILL_RETURN_HORIZON

    return SKILL_RETURN_DISCOUNT, SKILL_RETURN_HORIZON


def assign_discounted_returns(episode):
    total_examples = len(episode)
    for index, example in enumerate(episode):
        horizon = max(1, int(example.get("credit_horizon", PERSISTENT_RETURN_HORIZON)))
        discount = float(example.get("credit_discount", RETURN_DISCOUNT))
        discounted_return = 0.0
        discount_power = 1.0
        end_index = min(total_examples, index + horizon)
        for future_index in range(index, end_index):
            discounted_return += discount_power * float(episode[future_index]["reward"])
            discount_power *= discount
        example["discounted_return"] = discounted_return


def compute_reward(record, terminal_summary=None, reward_weights=None):
    reward_weights = reward_weights or DEFAULT_REWARD_WEIGHTS
    delta = record.get("delta") or {}
    state_before = record.get("state_before") or {}
    action_payload = record.get("action") or {}

    damage_dealt = max(0.0, -(delta.get("monster_total_hp_delta") or 0.0))
    hp_lost = max(0.0, -(delta.get("current_hp_delta") or 0.0))
    block_gain = effective_block_gain(record)
    invalid_block = invalid_block_amount(record)
    low_incoming_invalid_block = low_incoming_invalid_block_amount(record)
    incoming_damage_reduction = max(0.0, -(delta.get("incoming_damage_delta") or 0.0))
    strength_bonus = strength_damage_bonus(record, damage_dealt)
    dexterity_bonus = dexterity_block_bonus(record, block_gain)
    vulnerable_bonus = enemy_vulnerable_damage_bonus(record, damage_dealt)
    weak_bonus = enemy_weak_damage_reduction_bonus(record, incoming_damage_reduction)
    frail_bonus = enemy_frail_block_reduction_bonus(record)
    enemy_strength_down_bonus = enemy_strength_down_damage_reduction_bonus(record, incoming_damage_reduction)
    enemy_dexterity_down_bonus = enemy_dexterity_down_damage_bonus(record, damage_dealt)
    setup_bonus = buff_setup_bonus(record)
    energy_spent = action_energy_cost_from_payload(state_before, action_payload)
    efficiency_denominator = reward_efficiency_denominator(energy_spent)
    damage_efficiency_bonus = damage_dealt / efficiency_denominator if damage_dealt > 0.0 else 0.0
    block_efficiency_bonus = block_gain / efficiency_denominator if block_gain > 0.0 else 0.0
    buff_value = (
        strength_bonus
        + dexterity_bonus
        + vulnerable_bonus
        + weak_bonus
        + frail_bonus
        + enemy_strength_down_bonus
        + enemy_dexterity_down_bonus
        + setup_bonus
    )
    buff_efficiency_bonus = buff_value / efficiency_denominator if buff_value > 0.0 else 0.0
    gold_gain = max(0.0, delta.get("gold_delta") or 0.0)
    relic_gain = float(len(delta.get("gained_relics") or []))
    potion_gain = float(len(delta.get("gained_potions") or []))
    potion_use = float(len(delta.get("lost_potions") or []))
    monster_kills = float(len(delta.get("monsters_killed") or []))
    block_weight = reward_weights.get("effective_block", reward_weights.get("block_gain", 0.0))

    reward = 0.0
    reward += reward_weights["damage_dealt"] * damage_dealt
    reward += reward_weights["monster_kill"] * monster_kills
    reward += reward_weights["hp_loss"] * hp_lost
    reward += block_weight * block_gain
    reward += reward_weights.get("invalid_block", 0.0) * invalid_block
    reward += reward_weights.get("low_incoming_invalid_block", 0.0) * low_incoming_invalid_block
    reward += reward_weights["incoming_damage_reduction"] * incoming_damage_reduction
    reward += reward_weights.get("strength_damage_bonus", 0.0) * strength_bonus
    reward += reward_weights.get("dexterity_block_bonus", 0.0) * dexterity_bonus
    reward += reward_weights.get("enemy_vulnerable_damage_bonus", 0.0) * vulnerable_bonus
    reward += reward_weights.get("enemy_weak_damage_reduction_bonus", 0.0) * weak_bonus
    reward += reward_weights.get("enemy_frail_block_reduction_bonus", 0.0) * frail_bonus
    reward += reward_weights.get("enemy_strength_down_damage_reduction_bonus", 0.0) * enemy_strength_down_bonus
    reward += reward_weights.get("enemy_dexterity_down_damage_bonus", 0.0) * enemy_dexterity_down_bonus
    reward += reward_weights.get("buff_setup_bonus", 0.0) * setup_bonus
    reward += reward_weights.get("energy_spent", 0.0) * energy_spent
    reward += reward_weights.get("damage_efficiency_bonus", 0.0) * damage_efficiency_bonus
    reward += reward_weights.get("block_efficiency_bonus", 0.0) * block_efficiency_bonus
    reward += reward_weights.get("buff_efficiency_bonus", 0.0) * buff_efficiency_bonus
    reward += reward_weights["gold_gain"] * gold_gain
    reward += reward_weights["relic_gain"] * relic_gain
    reward += reward_weights["potion_gain"] * potion_gain
    reward += reward_weights["potion_use"] * potion_use

    if delta.get("combat_finished"):
        reward += reward_weights["combat_win"]

    if terminal_summary is not None:
        summary = terminal_summary.get("summary") or {}
        if summary.get("truncated") and summary.get("state_after_combat") is None:
            reward += reward_weights["death"]

    return reward


def load_trajectory_episodes(
    trajectory_dir,
    reward_weights=None,
    source_filter=None,
    limit_files=None,
    run_id=None,
    observation_version=LEGACY_COMBAT_OBSERVATION_VERSION,
):
    reward_weights = reward_weights or DEFAULT_REWARD_WEIGHTS
    source_filter = set(source_filter or [])
    paths = sorted(glob.glob(os.path.join(trajectory_dir, "*_combat_*.jsonl")))
    if run_id is not None:
        prefix = "{}_combat_".format(run_id)
        paths = [path for path in paths if os.path.basename(path).startswith(prefix)]
    if limit_files is not None:
        paths = paths[:limit_files]

    episodes = []
    stats = {
        "files_seen": len(paths),
        "episodes_loaded": 0,
        "examples_loaded": 0,
        "skipped_actions": 0,
        "skipped_neows_lament_combats": 0,
        "skipped_neows_lament_examples": 0,
        "malformed_files": 0,
        "sources": {},
        "min_sample_weight": None,
        "max_sample_weight": None,
    }

    for path in paths:
        transitions = []
        summary_record = None
        meta_record = None
        malformed_file = False
        with open(path, "r") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_file = True
                    break
                if record.get("record_type") == "meta":
                    meta_record = record
                elif record.get("record_type") == "transition":
                    transitions.append(record)
                elif record.get("record_type") == "summary":
                    summary_record = record

        if malformed_file:
            stats["malformed_files"] += 1
            continue

        if should_skip_combat(meta_record):
            stats["skipped_neows_lament_combats"] += 1
            stats["skipped_neows_lament_examples"] += len(transitions)
            continue

        combat_sample_weight = compute_combat_sample_weight(meta_record)

        episode = []
        for index, record in enumerate(transitions):
            source = record.get("source")
            if source_filter and source not in source_filter:
                continue

            state_before = record.get("state_before")
            if state_before is None or state_before.get("combat_state") is None:
                continue

            action_index, target_index, uses_target = encode_logged_action(state_before, record.get("action") or {})
            if action_index is None:
                stats["skipped_actions"] += 1
                continue

            reward = compute_reward(
                record,
                terminal_summary=summary_record if index == len(transitions) - 1 else None,
                reward_weights=reward_weights,
            )
            credit_discount, credit_horizon = action_credit_assignment(state_before, record.get("action") or {})
            encoded_state = build_state_tensors(state_before, observation_version=observation_version)
            example = {
                "trajectory_path": path,
                "combat_id": record.get("combat_id"),
                "step_index": record.get("step_index"),
                "source": source,
                "state": encoded_state,
                "action_index": action_index,
                "target_index": target_index if target_index is not None else 0,
                "uses_target": uses_target and target_index is not None,
                "action_mask": build_action_mask(state_before),
                "target_mask": build_target_mask(state_before),
                "reward": reward,
                "sample_weight": combat_sample_weight * compute_transition_sample_weight(record),
                "terminal": bool(record.get("terminal")),
                "credit_discount": credit_discount,
                "credit_horizon": credit_horizon,
                "observation_version": observation_version,
            }
            episode.append(example)
            stats["sources"][source] = stats["sources"].get(source, 0) + 1

        if not episode:
            continue

        assign_discounted_returns(episode)

        episode_return = sum(example["reward"] for example in episode)
        for example in episode:
            example["episode_return"] = episode_return

        episodes.append(episode)
        stats["episodes_loaded"] += 1
        stats["examples_loaded"] += len(episode)
        episode_weights = [example["sample_weight"] for example in episode]
        episode_min_weight = min(episode_weights)
        episode_max_weight = max(episode_weights)
        if stats["min_sample_weight"] is None or episode_min_weight < stats["min_sample_weight"]:
            stats["min_sample_weight"] = episode_min_weight
        if stats["max_sample_weight"] is None or episode_max_weight > stats["max_sample_weight"]:
            stats["max_sample_weight"] = episode_max_weight

    return episodes, stats


def load_preference_examples(
    trajectory_dir,
    source_filter=None,
    limit_files=None,
    run_id=None,
    observation_version=LEGACY_COMBAT_OBSERVATION_VERSION,
):
    source_filter = set(source_filter or [])
    paths = sorted(glob.glob(os.path.join(trajectory_dir, "*_combat_*.jsonl")))
    if run_id is not None:
        prefix = "{}_combat_".format(run_id)
        paths = [path for path in paths if os.path.basename(path).startswith(prefix)]
    if limit_files is not None:
        paths = paths[:limit_files]

    examples = []
    stats = {
        "files_seen": len(paths),
        "episodes_loaded": 0,
        "examples_loaded": 0,
        "skipped_actions": 0,
        "skipped_neows_lament_combats": 0,
        "skipped_neows_lament_examples": 0,
        "malformed_files": 0,
        "sources": {},
    }

    for path in paths:
        meta_record = None
        malformed_file = False
        preference_records = []
        with open(path, "r") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_file = True
                    break
                if record.get("record_type") == "meta":
                    meta_record = record
                elif record.get("record_type") == "preference":
                    preference_records.append(record)

        if malformed_file:
            stats["malformed_files"] += 1
            continue

        if should_skip_combat(meta_record):
            stats["skipped_neows_lament_combats"] += 1
            stats["skipped_neows_lament_examples"] += len(preference_records)
            continue

        combat_sample_weight = compute_combat_sample_weight(meta_record)
        file_examples = 0

        for record in preference_records:
            source = record.get("source")
            if source_filter and source not in source_filter:
                continue

            state_before = record.get("state_before")
            if state_before is None or state_before.get("combat_state") is None:
                continue

            preferred_payload = record.get("chosen_action") or {}
            preferred_action_index, preferred_target_index, preferred_uses_target = encode_logged_action(
                state_before,
                preferred_payload,
            )
            if preferred_action_index is None:
                stats["skipped_actions"] += 1
                continue

            encoded_state = build_state_tensors(state_before, observation_version=observation_version)
            action_mask = build_action_mask(state_before)
            target_mask = build_target_mask(state_before)
            preferred_score = float(record.get("chosen_score", 0.0))
            seen_rejections = set()
            for candidate in record.get("candidates") or []:
                if candidate.get("preferred"):
                    continue
                rejected_payload = candidate.get("action") or {}
                rejected_action_index, rejected_target_index, rejected_uses_target = encode_logged_action(
                    state_before,
                    rejected_payload,
                )
                if rejected_action_index is None:
                    stats["skipped_actions"] += 1
                    continue

                rejection_key = (
                    rejected_action_index,
                    rejected_target_index if rejected_target_index is not None else -1,
                    bool(rejected_uses_target and rejected_target_index is not None),
                )
                if rejection_key in seen_rejections:
                    continue
                seen_rejections.add(rejection_key)

                rejected_score = float(candidate.get("score", 0.0))
                margin = max(0.25, preferred_score - rejected_score)
                preferred_weight = preference_action_weight(state_before, preferred_payload, source)
                rejected_weight = preference_action_weight(state_before, rejected_payload, source)
                pair_weight = combat_sample_weight * margin * max(preferred_weight, 0.05) * max(rejected_weight, 0.05)
                examples.append({
                    "trajectory_path": path,
                    "combat_id": record.get("combat_id"),
                    "step_index": record.get("step_index"),
                    "source": source,
                    "state": encoded_state,
                    "action_mask": action_mask,
                    "target_mask": target_mask,
                    "preferred_action_index": preferred_action_index,
                    "preferred_target_index": preferred_target_index if preferred_target_index is not None else 0,
                    "preferred_uses_target": preferred_uses_target and preferred_target_index is not None,
                    "rejected_action_index": rejected_action_index,
                    "rejected_target_index": rejected_target_index if rejected_target_index is not None else 0,
                    "rejected_uses_target": rejected_uses_target and rejected_target_index is not None,
                    "sample_weight": pair_weight,
                    "observation_version": observation_version,
                })
                file_examples += 1
                stats["sources"][source] = stats["sources"].get(source, 0) + 1

        if file_examples > 0:
            stats["episodes_loaded"] += 1
            stats["examples_loaded"] += file_examples

    return examples, stats


def flatten_episodes(episodes):
    examples = []
    for episode in episodes:
        examples.extend(episode)
    return examples


def split_episodes(episodes, validation_fraction=0.1, seed=7):
    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1 or validation_fraction <= 0:
        return shuffled, []
    validation_size = max(1, int(math.ceil(len(shuffled) * validation_fraction)))
    validation = shuffled[:validation_size]
    training = shuffled[validation_size:]
    if not training:
        training = validation
        validation = []
    return training, validation


def batch_iterator(examples, batch_size, shuffle=True, seed=None):
    indices = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [examples[index] for index in indices[start:start + batch_size]]


def examples_to_batch(examples, device):
    batch = {
        "global_features": [],
        "deck_features": [],
        "hand_card_ids": [],
        "hand_features": [],
        "monster_ids": [],
        "monster_intents": [],
        "monster_features": [],
        "relic_ids": [],
        "action_mask": [],
        "target_mask": [],
        "action_index": [],
        "target_index": [],
        "uses_target": [],
        "discounted_return": [],
        "episode_return": [],
        "sample_weight": [],
    }

    for example in examples:
        state = example["state"]
        batch["global_features"].append(state["global_features"] + [state["potion_count"]])
        batch["deck_features"].append(state["deck_features"])
        batch["hand_card_ids"].append(state["hand_card_ids"])
        batch["hand_features"].append(state["hand_features"])
        batch["monster_ids"].append(state["monster_ids"])
        batch["monster_intents"].append(state["monster_intents"])
        batch["monster_features"].append(state["monster_features"])
        batch["relic_ids"].append(state["relic_ids"])
        batch["action_mask"].append(example["action_mask"])
        batch["target_mask"].append(example["target_mask"])
        batch["action_index"].append(example["action_index"])
        batch["target_index"].append(example["target_index"])
        batch["uses_target"].append(example["uses_target"])
        batch["discounted_return"].append(example["discounted_return"])
        batch["episode_return"].append(example["episode_return"])
        batch["sample_weight"].append(example.get("sample_weight", 1.0))

    return {
        "global_features": torch.tensor(batch["global_features"], dtype=torch.float32, device=device),
        "deck_features": torch.tensor(batch["deck_features"], dtype=torch.float32, device=device),
        "hand_card_ids": torch.tensor(batch["hand_card_ids"], dtype=torch.long, device=device),
        "hand_features": torch.tensor(batch["hand_features"], dtype=torch.float32, device=device),
        "monster_ids": torch.tensor(batch["monster_ids"], dtype=torch.long, device=device),
        "monster_intents": torch.tensor(batch["monster_intents"], dtype=torch.long, device=device),
        "monster_features": torch.tensor(batch["monster_features"], dtype=torch.float32, device=device),
        "relic_ids": torch.tensor(batch["relic_ids"], dtype=torch.long, device=device),
        "action_mask": torch.tensor(batch["action_mask"], dtype=torch.bool, device=device),
        "target_mask": torch.tensor(batch["target_mask"], dtype=torch.bool, device=device),
        "action_index": torch.tensor(batch["action_index"], dtype=torch.long, device=device),
        "target_index": torch.tensor(batch["target_index"], dtype=torch.long, device=device),
        "uses_target": torch.tensor(batch["uses_target"], dtype=torch.bool, device=device),
        "discounted_return": torch.tensor(batch["discounted_return"], dtype=torch.float32, device=device),
        "episode_return": torch.tensor(batch["episode_return"], dtype=torch.float32, device=device),
        "sample_weight": torch.tensor(batch["sample_weight"], dtype=torch.float32, device=device),
    }


def preference_examples_to_batch(examples, device):
    batch = {
        "global_features": [],
        "deck_features": [],
        "hand_card_ids": [],
        "hand_features": [],
        "monster_ids": [],
        "monster_intents": [],
        "monster_features": [],
        "relic_ids": [],
        "action_mask": [],
        "target_mask": [],
        "preferred_action_index": [],
        "preferred_target_index": [],
        "preferred_uses_target": [],
        "rejected_action_index": [],
        "rejected_target_index": [],
        "rejected_uses_target": [],
        "sample_weight": [],
    }

    for example in examples:
        state = example["state"]
        batch["global_features"].append(state["global_features"] + [state["potion_count"]])
        batch["deck_features"].append(state["deck_features"])
        batch["hand_card_ids"].append(state["hand_card_ids"])
        batch["hand_features"].append(state["hand_features"])
        batch["monster_ids"].append(state["monster_ids"])
        batch["monster_intents"].append(state["monster_intents"])
        batch["monster_features"].append(state["monster_features"])
        batch["relic_ids"].append(state["relic_ids"])
        batch["action_mask"].append(example["action_mask"])
        batch["target_mask"].append(example["target_mask"])
        batch["preferred_action_index"].append(example["preferred_action_index"])
        batch["preferred_target_index"].append(example["preferred_target_index"])
        batch["preferred_uses_target"].append(example["preferred_uses_target"])
        batch["rejected_action_index"].append(example["rejected_action_index"])
        batch["rejected_target_index"].append(example["rejected_target_index"])
        batch["rejected_uses_target"].append(example["rejected_uses_target"])
        batch["sample_weight"].append(example.get("sample_weight", 1.0))

    return {
        "global_features": torch.tensor(batch["global_features"], dtype=torch.float32, device=device),
        "deck_features": torch.tensor(batch["deck_features"], dtype=torch.float32, device=device),
        "hand_card_ids": torch.tensor(batch["hand_card_ids"], dtype=torch.long, device=device),
        "hand_features": torch.tensor(batch["hand_features"], dtype=torch.float32, device=device),
        "monster_ids": torch.tensor(batch["monster_ids"], dtype=torch.long, device=device),
        "monster_intents": torch.tensor(batch["monster_intents"], dtype=torch.long, device=device),
        "monster_features": torch.tensor(batch["monster_features"], dtype=torch.float32, device=device),
        "relic_ids": torch.tensor(batch["relic_ids"], dtype=torch.long, device=device),
        "action_mask": torch.tensor(batch["action_mask"], dtype=torch.bool, device=device),
        "target_mask": torch.tensor(batch["target_mask"], dtype=torch.bool, device=device),
        "preferred_action_index": torch.tensor(batch["preferred_action_index"], dtype=torch.long, device=device),
        "preferred_target_index": torch.tensor(batch["preferred_target_index"], dtype=torch.long, device=device),
        "preferred_uses_target": torch.tensor(batch["preferred_uses_target"], dtype=torch.bool, device=device),
        "rejected_action_index": torch.tensor(batch["rejected_action_index"], dtype=torch.long, device=device),
        "rejected_target_index": torch.tensor(batch["rejected_target_index"], dtype=torch.long, device=device),
        "rejected_uses_target": torch.tensor(batch["rejected_uses_target"], dtype=torch.bool, device=device),
        "sample_weight": torch.tensor(batch["sample_weight"], dtype=torch.float32, device=device),
    }


def masked_logits(logits, mask):
    return logits.masked_fill(~mask, -1e9)


def masked_log_probs(logits, mask):
    return F.log_softmax(masked_logits(logits, mask), dim=-1)


def masked_entropy(logits, mask):
    log_probs = masked_log_probs(logits, mask)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


class CombatPolicyNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.card_embedding = nn.Embedding(CARD_ID_BUCKETS + 1, 16, padding_idx=0)
        self.monster_embedding = nn.Embedding(MONSTER_ID_BUCKETS + 1, 8, padding_idx=0)
        self.intent_embedding = nn.Embedding(len(INTENT_TO_INDEX) + 1, 4, padding_idx=0)
        self.relic_embedding = nn.Embedding(RELIC_ID_BUCKETS + 1, 8, padding_idx=0)

        input_dim = (
            GLOBAL_FEATURE_DIM + 1
            + DECK_FEATURE_DIM
            + MAX_HAND_SIZE * (HAND_FEATURE_DIM + 16)
            + MAX_MONSTERS * (MONSTER_FEATURE_DIM + 8 + 4)
            + 8
        )

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(256, ACTION_DIM)
        self.target_head = nn.Linear(256, TARGET_DIM)
        self.value_head = nn.Linear(256, 1)

    def forward(self, batch):
        hand_embeddings = self.card_embedding(batch["hand_card_ids"])
        hand_tokens = torch.cat([batch["hand_features"], hand_embeddings], dim=-1)
        hand_flat = hand_tokens.reshape(batch["hand_features"].shape[0], -1)

        monster_embeddings = self.monster_embedding(batch["monster_ids"])
        intent_embeddings = self.intent_embedding(batch["monster_intents"])
        monster_tokens = torch.cat([batch["monster_features"], monster_embeddings, intent_embeddings], dim=-1)
        monster_flat = monster_tokens.reshape(batch["monster_features"].shape[0], -1)

        relic_embeddings = self.relic_embedding(batch["relic_ids"])
        relic_mask = (batch["relic_ids"] != 0).float().unsqueeze(-1)
        relic_sum = (relic_embeddings * relic_mask).sum(dim=1)
        relic_denominator = relic_mask.sum(dim=1).clamp_min(1.0)
        relic_summary = relic_sum / relic_denominator

        features = torch.cat(
            [
                batch["global_features"],
                batch["deck_features"],
                hand_flat,
                monster_flat,
                relic_summary,
            ],
            dim=-1,
        )
        hidden = self.backbone(features)
        return {
            "action_logits": self.action_head(hidden),
            "target_logits": self.target_head(hidden),
            "value": self.value_head(hidden).squeeze(-1),
        }


def compute_behavior_cloning_loss(model_outputs, batch):
    action_log_probs = masked_log_probs(model_outputs["action_logits"], batch["action_mask"])
    weights = batch["sample_weight"]
    action_loss = weighted_mean(F.nll_loss(action_log_probs, batch["action_index"], reduction="none"), weights)
    action_predictions = masked_logits(model_outputs["action_logits"], batch["action_mask"]).argmax(dim=-1)
    action_accuracy = weighted_accuracy(action_predictions, batch["action_index"], weights)

    target_loss = torch.tensor(0.0, device=action_loss.device)
    target_accuracy = torch.tensor(1.0, device=action_loss.device)
    if batch["uses_target"].any():
        target_log_probs = masked_log_probs(
            model_outputs["target_logits"][batch["uses_target"]],
            batch["target_mask"][batch["uses_target"]],
        )
        target_weights = weights[batch["uses_target"]]
        target_loss = weighted_mean(
            F.nll_loss(target_log_probs, batch["target_index"][batch["uses_target"]], reduction="none"),
            target_weights,
        )
        target_predictions = masked_logits(
            model_outputs["target_logits"][batch["uses_target"]],
            batch["target_mask"][batch["uses_target"]],
        ).argmax(dim=-1)
        target_accuracy = weighted_accuracy(
            target_predictions,
            batch["target_index"][batch["uses_target"]],
            target_weights,
        )

    loss = action_loss + 0.35 * target_loss
    return {
        "loss": loss,
        "action_loss": action_loss.detach(),
        "target_loss": target_loss.detach(),
        "action_accuracy": action_accuracy.detach(),
        "target_accuracy": target_accuracy.detach(),
    }


def compute_policy_gradient_loss(model_outputs, batch, value_weight=0.5, entropy_weight=0.05, behavior_cloning_weight=0.05):
    del entropy_weight

    returns = batch["discounted_return"]
    values = model_outputs["value"]
    weights = batch["sample_weight"]

    action_scores = masked_logits(model_outputs["action_logits"], batch["action_mask"])
    selected_action_scores = action_scores.gather(1, batch["action_index"].unsqueeze(1)).squeeze(1)
    action_value_loss = weighted_mean((selected_action_scores - returns) ** 2, weights)

    target_value_loss = torch.tensor(0.0, device=values.device)
    if batch["uses_target"].any():
        target_scores = masked_logits(
            model_outputs["target_logits"][batch["uses_target"]],
            batch["target_mask"][batch["uses_target"]],
        )
        selected_target_scores = target_scores.gather(
            1,
            batch["target_index"][batch["uses_target"]].unsqueeze(1),
        ).squeeze(1)
        target_value_loss = weighted_mean(
            (selected_target_scores - returns[batch["uses_target"]]) ** 2,
            weights[batch["uses_target"]],
        )

    action_log_probs = masked_log_probs(model_outputs["action_logits"], batch["action_mask"])
    bc_action_loss = weighted_mean(F.nll_loss(action_log_probs, batch["action_index"], reduction="none"), weights)
    bc_target_loss = torch.tensor(0.0, device=values.device)
    if batch["uses_target"].any():
        bc_target_log_probs = masked_log_probs(
            model_outputs["target_logits"][batch["uses_target"]],
            batch["target_mask"][batch["uses_target"]],
        )
        bc_target_loss = weighted_mean(
            F.nll_loss(
                bc_target_log_probs,
                batch["target_index"][batch["uses_target"]],
                reduction="none",
            ),
            weights[batch["uses_target"]],
        )
    bc_loss = bc_action_loss + 0.35 * bc_target_loss

    value_loss = weighted_mean((values - returns) ** 2, weights)
    loss = (
        action_value_loss
        + 0.35 * target_value_loss
        + value_weight * value_loss
        + behavior_cloning_weight * bc_loss
    )
    return {
        "loss": loss,
        "action_value_loss": action_value_loss.detach(),
        "target_value_loss": target_value_loss.detach(),
        "bc_loss": bc_loss.detach(),
        "value_loss": value_loss.detach(),
        "mean_return": weighted_mean(returns, weights).detach(),
    }


def compute_preference_loss(model_outputs, batch):
    weights = batch["sample_weight"]
    action_scores = masked_logits(model_outputs["action_logits"], batch["action_mask"])
    target_scores = masked_logits(model_outputs["target_logits"], batch["target_mask"])

    preferred_scores = action_scores.gather(1, batch["preferred_action_index"].unsqueeze(1)).squeeze(1)
    rejected_scores = action_scores.gather(1, batch["rejected_action_index"].unsqueeze(1)).squeeze(1)

    preferred_target_scores = torch.zeros_like(preferred_scores)
    rejected_target_scores = torch.zeros_like(rejected_scores)
    if batch["preferred_uses_target"].any():
        preferred_target_scores[batch["preferred_uses_target"]] = target_scores[batch["preferred_uses_target"]].gather(
            1,
            batch["preferred_target_index"][batch["preferred_uses_target"]].unsqueeze(1),
        ).squeeze(1)
    if batch["rejected_uses_target"].any():
        rejected_target_scores[batch["rejected_uses_target"]] = target_scores[batch["rejected_uses_target"]].gather(
            1,
            batch["rejected_target_index"][batch["rejected_uses_target"]].unsqueeze(1),
        ).squeeze(1)

    preferred_total = preferred_scores + preferred_target_scores
    rejected_total = rejected_scores + rejected_target_scores
    margin = preferred_total - rejected_total
    loss = weighted_mean(F.softplus(-margin), weights)
    accuracy = weighted_mean((margin > 0).float(), weights)
    return {
        "loss": loss,
        "preference_loss": loss.detach(),
        "preference_accuracy": accuracy.detach(),
        "mean_margin": weighted_mean(margin, weights).detach(),
    }


def run_epoch(
    model,
    examples,
    optimizer,
    device,
    batch_size,
    mode,
    seed,
    value_weight=0.5,
    entropy_weight=0.05,
    behavior_cloning_weight=0.05,
):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    totals = {}
    steps = 0
    iterator = batch_iterator(examples, batch_size=batch_size, shuffle=training, seed=seed)

    for step_index, example_batch in enumerate(iterator):
        batch = preference_examples_to_batch(example_batch, device) if mode == "preference" else examples_to_batch(example_batch, device)
        with torch.set_grad_enabled(training):
            outputs = model(batch)
            if mode == "bc":
                metrics = compute_behavior_cloning_loss(outputs, batch)
            elif mode == "reinforce":
                metrics = compute_policy_gradient_loss(
                    outputs,
                    batch,
                    value_weight=value_weight,
                    entropy_weight=entropy_weight,
                    behavior_cloning_weight=behavior_cloning_weight,
                )
            elif mode == "preference":
                metrics = compute_preference_loss(outputs, batch)
            else:
                raise ValueError("Unsupported mode: {}".format(mode))

            if training:
                optimizer.zero_grad()
                metrics["loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach() if hasattr(value, "detach") else value)
        steps += 1

    if steps == 0:
        return {}
    return dict((key, value / steps) for key, value in totals.items())


def save_checkpoint(path, model, training_args, dataset_stats):
    training_args = dict(training_args)
    training_args.setdefault("combat_observation_version", LEGACY_COMBAT_OBSERVATION_VERSION)
    payload = {
        "model_state_dict": model.state_dict(),
        "training_args": training_args,
        "dataset_stats": dataset_stats,
        "reward_weights": DEFAULT_REWARD_WEIGHTS,
    }
    torch.save(payload, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def choose_target_index(game_state, card, predicted_target_index, fallback_agent):
    available_monsters = [
        monster for monster in game_state.monsters
        if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone
    ]
    if not available_monsters:
        return None

    if predicted_target_index is not None:
        for monster in available_monsters:
            if monster.monster_index == predicted_target_index:
                return monster

    if card.type == CardType.ATTACK:
        return fallback_agent.get_low_hp_target()
    return fallback_agent.get_high_hp_target()


def decode_action_from_prediction(game_state, fallback_agent, action_index, target_index):
    if action_index == ACTION_END_TURN:
        return EndTurnAction()

    card_slot = action_index - ACTION_PLAY_OFFSET
    if card_slot < 0 or card_slot >= len(game_state.hand):
        return fallback_agent.get_play_card_action()

    card = game_state.hand[card_slot]
    if not card.is_playable:
        return fallback_agent.get_play_card_action()

    if not card.has_target:
        return PlayCardAction(card=card)

    target = choose_target_index(game_state, card, target_index, fallback_agent)
    if target is None:
        return EndTurnAction()
    return PlayCardAction(card=card, target_monster=target)
