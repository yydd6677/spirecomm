import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

from spirecomm.ai.observation import canonicalize_card, canonicalize_serialized_state
from spirecomm.ai.torch_compat import F, nn, torch


CARD_BUCKETS = 192
RELIC_BUCKETS = 96
STATE_NUMERIC_DIM = 32
CANDIDATE_TYPE_DIM = 5
CANDIDATE_RARITY_DIM = 7
CANDIDATE_EXTRA_DIM = 6
STATE_DIM = STATE_NUMERIC_DIM + CARD_BUCKETS + RELIC_BUCKETS
CANDIDATE_DIM = CARD_BUCKETS + CANDIDATE_TYPE_DIM + CANDIDATE_RARITY_DIM + CANDIDATE_EXTRA_DIM
BOSS_NAMES = [
    "theguardian",
    "hexaghost",
    "slimeboss",
    "thechamp",
    "collector",
    "bronzeautomaton",
    "awakenedone",
    "timeeater",
    "donuanddeca",
]
BOSS_INDEX = {name: index for index, name in enumerate(BOSS_NAMES)}
RARITY_ORDER = ["BASIC", "COMMON", "UNCOMMON", "RARE", "COLORLESS", "CURSE", "SPECIAL"]
RARITY_INDEX = {name: index for index, name in enumerate(RARITY_ORDER)}
TYPE_ORDER = ["ATTACK", "SKILL", "POWER", "STATUS", "CURSE"]
TYPE_INDEX = {name: index for index, name in enumerate(TYPE_ORDER)}
IRONCLAD_STARTER_CARD_COUNTS = {
    "strike": 5,
    "defend": 4,
    "bash": 1,
}
DEFAULT_WEAK_CARD_BIAS_TIER1 = (
    "havoc",
    "perfectedstrike",
    "truegrit",
    "bodyslam",
    "disarm",
    "dropkick",
    "combust",
    "secondwind",
    "corruption",
    "finesse",
    "flashofsteel",
)
DEFAULT_WEAK_CARD_BIAS_TIER2 = (
    "warcry",
    "burningpact",
    "darkembrace",
    "dualwield",
    "entrench",
    "evolve",
    "rupture",
    "berserk",
    "searingblow",
    "sentinel",
    "barricade",
    "doubletap",
    "exhume",
    "impatience",
    "purity",
    "panache",
    "sadisticnature",
)
DEFAULT_STRONG_CARD_BIAS_BOOST = (
    "demonform",
    "metallicize",
    "feelnopain",
    "brutality",
)
ACT1_LATE_SETUP_POWER_CARDS = (
    "barricade",
    "brutality",
    "combust",
    "corruption",
    "darkembrace",
    "evolve",
    "firebreathing",
    "juggernaut",
    "rupture",
)
ARCHETYPE_STRENGTH_CARDS = (
    "flex",
    "heavyblade",
    "swordboomerang",
    "twinstrike",
    "inflame",
    "pummel",
    "spotweakness",
    "whirlwind",
    "demonform",
    "fiendfire",
    "limitbreak",
    "reaper",
)
ARCHETYPE_BLOCK_CARDS = (
    "bodyslam",
    "shrugitoff",
    "truegrit",
    "battletrance",
    "entrench",
    "flamebarrier",
    "ghostlyarmor",
    "powerthrough",
    "secondwind",
    "barricade",
    "impervious",
    "juggernaut",
)
ARCHETYPE_BLOCK_ANCHORS = ("bodyslam", "barricade")
ARCHETYPE_AOE_CARDS = ("cleave", "thunderclap", "whirlwind", "immolate")
ARCHETYPE_EXHAUST_CARDS = (
    "darkembrace",
    "feelnopain",
    "havoc",
    "truegrit",
    "burningpact",
    "secondwind",
    "seversoul",
    "corruption",
    "feed",
    "infernalblade",
    "pummel",
    "fiendfire",
    "disarm",
    "limitbreak",
    "impervious",
    "seeingred",
    "shockwave",
    "exhume",
    "intimidate",
    "offering",
    "reaper",
    "warcry",
)
ARCHETYPE_EXHAUST_ANCHORS = ("darkembrace", "feelnopain")
EARLY_CARD_REWARD_EXTRA_THRESHOLD = 5
EARLY_CARD_REWARD_SKIP_POWER = 1.5
EARLY_CARD_REWARD_SKIP_MIN_WEIGHT = 0.05
EARLY_CARD_REWARD_ATTACK_BIAS_MAX = float(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_ATTACK_BIAS_MAX", "1.0"))
EARLY_CARD_REWARD_ATTACK_BIAS_POWER = float(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_ATTACK_BIAS_POWER", "1.0"))
EARLY_FRONTLOAD_ATTACK_CARDS = {
    "anger",
    "bludgeon",
    "carnage",
    "cleave",
    "clothesline",
    "headbutt",
    "heavyblade",
    "hemokinesis",
    "immolate",
    "pommelstrike",
    "pummel",
    "rampage",
    "swordboomerang",
    "twinstrike",
    "uppercut",
    "whirlwind",
    "wildstrike",
}
EARLY_FRONTLOAD_SCALING_CARDS = {
    "demonform",
    "flex",
    "inflame",
    "limitbreak",
    "spotweakness",
}


def normalize_token(value):
    text = str(value or "").lower()
    return "".join(ch for ch in text if ch.isalnum())


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _parse_card_key_list(raw):
    return [
        normalize_token(token)
        for token in str(raw or "").replace("\n", ",").split(",")
        if normalize_token(token)
    ]


def card_score_bias_map():
    """Runtime card-score biases for positive card acquisition/target choices."""

    json_payload = os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_JSON")
    if json_payload:
        try:
            payload = json.loads(json_payload)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            return {
                normalize_token(card_key): float(value)
                for card_key, value in payload.items()
                if normalize_token(card_key)
            }

    tier1_raw = os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_TIER1_CARDS")
    tier2_raw = os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_TIER2_CARDS")
    boost_raw = os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_BOOST_CARDS")
    enabled = _env_bool("SPIRECOMM_CARD_SCORE_BIAS_ENABLED", True) or bool(tier1_raw or tier2_raw or boost_raw)
    if not enabled:
        return {}

    tier1_cards = _parse_card_key_list(tier1_raw) if tier1_raw else list(DEFAULT_WEAK_CARD_BIAS_TIER1)
    tier2_cards = _parse_card_key_list(tier2_raw) if tier2_raw else list(DEFAULT_WEAK_CARD_BIAS_TIER2)
    boost_cards = _parse_card_key_list(boost_raw) if boost_raw else list(DEFAULT_STRONG_CARD_BIAS_BOOST)
    tier1_value = float(os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_TIER1_VALUE", "-0.6"))
    tier2_value = float(os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_TIER2_VALUE", "-1.0"))
    boost_value = float(os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_BOOST_VALUE", "0.6"))
    biases = {key: tier1_value for key in tier1_cards}
    biases.update({key: tier2_value for key in tier2_cards})
    biases.update({key: boost_value for key in boost_cards})
    extra_json_payload = os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_EXTRA_JSON")
    if extra_json_payload:
        try:
            payload = json.loads(extra_json_payload)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            for card_key, value in payload.items():
                key = normalize_token(card_key)
                if key:
                    biases[key] = float(biases.get(key, 0.0)) + float(value)
    return biases


def _owned_card_count(deck_counter, card_keys):
    return sum(int(deck_counter.get(key, 0) or 0) for key in card_keys)


def _card_archetype_coefficients(state_like):
    if not _env_bool("SPIRECOMM_CARD_ARCHETYPE_BIAS_ENABLED", True):
        return {}
    deck_counter = Counter(canonical_card_key(card) for card in deck_cards_from_state(state_like))
    strength_count = _owned_card_count(deck_counter, ARCHETYPE_STRENGTH_CARDS)
    strength_half_count = int(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_STRENGTH_HALF_COUNT", "2"))
    strength_full_count = int(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_STRENGTH_FULL_COUNT", "3"))
    strength_coeff = 0.0
    if strength_count >= strength_full_count:
        strength_coeff = 1.0
    elif strength_count >= strength_half_count:
        strength_coeff = 0.5

    block_anchor_count = sum(1 for key in ARCHETYPE_BLOCK_ANCHORS if int(deck_counter.get(key, 0) or 0) > 0)
    block_coeff = min(1.0, 0.5 * float(block_anchor_count))

    aoe_count = _owned_card_count(deck_counter, ARCHETYPE_AOE_CARDS)
    aoe_max_count = int(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_AOE_MAX_COUNT", "1"))
    aoe_coeff = 1.0 if aoe_count <= aoe_max_count else 0.0

    exhaust_anchor_count = sum(1 for key in ARCHETYPE_EXHAUST_ANCHORS if int(deck_counter.get(key, 0) or 0) > 0)
    exhaust_coeff = min(1.0, 0.5 * float(exhaust_anchor_count))

    return {
        "strength": strength_coeff,
        "block": block_coeff,
        "aoe": aoe_coeff,
        "exhaust": exhaust_coeff,
    }


def _card_archetype_bias_map(state_like):
    coefficients = _card_archetype_coefficients(state_like)
    if not coefficients:
        return {}

    strength_value = float(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_STRENGTH_BIAS", "0.44"))
    block_value = float(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_BLOCK_BIAS", "0.00"))
    aoe_value = float(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_AOE_BIAS", "0.26"))
    exhaust_value = float(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_EXHAUST_BIAS", "0.91"))
    biases: Counter[str] = Counter()

    def add_group(card_keys, value, coefficient):
        delta = float(value) * float(coefficient)
        if delta == 0.0:
            return
        for key in card_keys:
            biases[key] += delta

    add_group(ARCHETYPE_STRENGTH_CARDS, strength_value, coefficients.get("strength", 0.0))
    add_group(ARCHETYPE_BLOCK_CARDS, block_value, coefficients.get("block", 0.0))
    add_group(ARCHETYPE_AOE_CARDS, aoe_value, coefficients.get("aoe", 0.0))
    add_group(ARCHETYPE_EXHAUST_CARDS, exhaust_value, coefficients.get("exhaust", 0.0))
    return dict(biases)


def _act1_late_setup_power_bias_map(state_like):
    if not _env_bool("SPIRECOMM_ACT1_LATE_SETUP_POWER_BIAS_ENABLED", False):
        return {}
    floor = floor_from_state(state_like)
    act = act_from_state(state_like)
    min_floor = int(os.environ.get("SPIRECOMM_ACT1_LATE_SETUP_POWER_MIN_FLOOR", "10"))
    max_floor = int(os.environ.get("SPIRECOMM_ACT1_LATE_SETUP_POWER_MAX_FLOOR", "15"))
    if act != 1 or floor < min_floor or floor > max_floor:
        return {}
    penalty = float(os.environ.get("SPIRECOMM_ACT1_LATE_SETUP_POWER_PENALTY", "-1.0"))
    if penalty == 0.0:
        return {}
    return {key: penalty for key in ACT1_LATE_SETUP_POWER_CARDS}


def card_score_biases_for_cards(cards, state_like=None):
    biases = card_score_bias_map()
    if state_like is not None:
        archetype_biases = _card_archetype_bias_map(state_like)
        if archetype_biases:
            merged = Counter(biases)
            merged.update(archetype_biases)
            biases = dict(merged)
        late_power_biases = _act1_late_setup_power_bias_map(state_like)
        if late_power_biases:
            merged = Counter(biases)
            merged.update(late_power_biases)
            biases = dict(merged)
    if not biases:
        return [0.0 for _ in cards]
    return [float(biases.get(canonical_card_key(card), 0.0)) for card in cards]


def stable_bucket(token, bucket_count):
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % bucket_count


def clamp_scale(value, scale):
    if scale <= 0:
        return 0.0
    return max(min(float(value) / float(scale), 1.0), -1.0)


def safe_divide(numerator, denominator):
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def canonical_card_key(card_like):
    if hasattr(card_like, "card_id"):
        key = normalize_token(getattr(card_like, "card_id"))
    else:
        key = normalize_token(card_like.get("card_id") or card_like.get("key") or card_like.get("name"))
    if key in {"striker", "defendr"}:
        return key[:-1]
    return key


def card_type_name(card_like):
    if hasattr(card_like, "type"):
        value = getattr(card_like, "type")
        return value.name if hasattr(value, "name") else str(value).upper()
    return str(card_like.get("type", "")).upper()


def card_rarity_name(card_like):
    if hasattr(card_like, "rarity"):
        value = getattr(card_like, "rarity")
        rarity = value.name if hasattr(value, "name") else str(value).upper()
    else:
        rarity = str(card_like.get("rarity", "")).upper()
    if rarity not in RARITY_INDEX:
        if str(card_like.get("pool", "")).lower() == "colorless_red":
            return "COLORLESS"
        return "SPECIAL" if rarity else "COMMON"
    return rarity


def card_upgrades(card_like):
    if hasattr(card_like, "upgrades"):
        return int(getattr(card_like, "upgrades") or 0)
    return int(card_like.get("upgrades", 0) or 0)


def relic_key(relic_like):
    if hasattr(relic_like, "relic_id"):
        return normalize_token(getattr(relic_like, "relic_id"))
    return normalize_token(relic_like.get("relic_id") or relic_like.get("key") or relic_like.get("name"))


def boss_name_from_state(state_like):
    if isinstance(state_like, dict):
        state_like = canonicalize_serialized_state(state_like)
    if hasattr(state_like, "act_boss"):
        return normalize_token(getattr(state_like, "act_boss"))
    if isinstance(state_like, dict):
        return normalize_token(state_like.get("next_boss") or state_like.get("act_boss"))
    return ""


def boss_one_hot(boss_name):
    vector = [0.0] * len(BOSS_NAMES)
    index = BOSS_INDEX.get(normalize_token(boss_name))
    if index is not None:
        vector[index] = 1.0
    return vector


def hashed_bag(items, bucket_count, key_fn):
    vector = [0.0] * bucket_count
    item_list = list(items)
    if not item_list:
        return vector
    weight = 1.0 / float(len(item_list))
    for item in item_list:
        token = key_fn(item)
        if not token:
            continue
        vector[stable_bucket(token, bucket_count)] += weight
    return vector


def deck_cards_from_state(state_like):
    if isinstance(state_like, dict):
        state_like = canonicalize_serialized_state(state_like)
    if hasattr(state_like, "deck"):
        return list(getattr(state_like, "deck") or [])
    if isinstance(state_like, dict):
        return list(state_like.get("deck") or [])
    return []


def starter_card_count_from_state(state_like):
    """Count current cards that still belong to the Ironclad starter deck."""
    counter = Counter(canonical_card_key(card) for card in deck_cards_from_state(state_like))
    return sum(
        min(int(counter.get(card_key, 0) or 0), max_count)
        for card_key, max_count in IRONCLAD_STARTER_CARD_COUNTS.items()
    )


def early_card_reward_extra_count(state_like):
    deck_size = len(deck_cards_from_state(state_like))
    return max(0, deck_size - starter_card_count_from_state(state_like))


def early_card_reward_skip_weight(state_like):
    extra = early_card_reward_extra_count(state_like)
    threshold = max(1, int(float(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_EXTRA_THRESHOLD", str(EARLY_CARD_REWARD_EXTRA_THRESHOLD)))))
    skip_power = float(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_SKIP_POWER", str(EARLY_CARD_REWARD_SKIP_POWER)))
    min_weight = float(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_SKIP_MIN_WEIGHT", str(EARLY_CARD_REWARD_SKIP_MIN_WEIGHT)))
    if extra >= threshold:
        return 1.0
    if _env_bool("SPIRECOMM_EARLY_CARD_REWARD_CONDITIONAL_FRONTLOAD", False):
        frontload_multiplier = early_card_reward_frontload_deficit_multiplier(state_like)
        if frontload_multiplier <= 0.0:
            return 1.0
    ratio = max(0.0, float(extra) / float(threshold))
    weight = max(min_weight, ratio ** skip_power)
    if _env_bool("SPIRECOMM_EARLY_CARD_REWARD_CONDITIONAL_FRONTLOAD", False):
        # Interpolate back toward neutral skip once the deck has enough Act 1 damage.
        weight = 1.0 - (1.0 - weight) * frontload_multiplier
    return weight


def early_card_reward_frontload_score(state_like):
    deck_counter = Counter(canonical_card_key(card) for card in deck_cards_from_state(state_like))
    score = 0.0
    for key, count in deck_counter.items():
        if key in EARLY_FRONTLOAD_ATTACK_CARDS:
            score += float(count)
        elif key in EARLY_FRONTLOAD_SCALING_CARDS:
            score += 0.75 * float(count)
    return score


def early_card_reward_frontload_deficit_multiplier(state_like):
    threshold = float(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_FRONTLOAD_READY_SCORE", "2.5"))
    if threshold <= 0.0:
        return 1.0
    score = early_card_reward_frontload_score(state_like)
    return max(0.0, min(1.0, (threshold - score) / threshold))


def early_card_reward_attack_bias(state_like):
    extra = early_card_reward_extra_count(state_like)
    threshold = max(1, int(float(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_EXTRA_THRESHOLD", str(EARLY_CARD_REWARD_EXTRA_THRESHOLD)))))
    attack_bias_max = float(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_ATTACK_BIAS_MAX", str(EARLY_CARD_REWARD_ATTACK_BIAS_MAX)))
    attack_bias_power = float(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_ATTACK_BIAS_POWER", str(EARLY_CARD_REWARD_ATTACK_BIAS_POWER)))
    if extra >= threshold:
        return 0.0
    remaining = max(0.0, 1.0 - float(extra) / float(threshold))
    bias = attack_bias_max * (remaining ** attack_bias_power)
    if _env_bool("SPIRECOMM_EARLY_CARD_REWARD_CONDITIONAL_FRONTLOAD", False):
        bias *= early_card_reward_frontload_deficit_multiplier(state_like)
    return bias


def relics_from_state(state_like):
    if isinstance(state_like, dict):
        state_like = canonicalize_serialized_state(state_like)
    if hasattr(state_like, "relics"):
        return list(getattr(state_like, "relics") or [])
    if isinstance(state_like, dict):
        return list(state_like.get("relics") or [])
    return []


def current_hp_from_state(state_like):
    if hasattr(state_like, "current_hp"):
        return int(getattr(state_like, "current_hp") or 0)
    if isinstance(state_like, dict):
        return int(state_like.get("current_hp", 0) or 0)
    return 0


def max_hp_from_state(state_like):
    if hasattr(state_like, "max_hp"):
        return int(getattr(state_like, "max_hp") or 0)
    if isinstance(state_like, dict):
        return int(state_like.get("max_hp", 0) or 0)
    return 0


def gold_from_state(state_like):
    if hasattr(state_like, "gold"):
        return int(getattr(state_like, "gold") or 0)
    if isinstance(state_like, dict):
        return int(state_like.get("gold", 0) or 0)
    return 0


def floor_from_state(state_like):
    if hasattr(state_like, "floor"):
        return int(getattr(state_like, "floor") or 0)
    if isinstance(state_like, dict):
        return int(state_like.get("floor", 0) or 0)
    return 0


def act_from_state(state_like):
    if hasattr(state_like, "act"):
        return int(getattr(state_like, "act") or 0)
    if isinstance(state_like, dict):
        return int(state_like.get("act", 0) or 0)
    return 0


def state_numeric_features(state_like):
    deck = deck_cards_from_state(state_like)
    relics = relics_from_state(state_like)
    deck_size = len(deck)
    deck_counter = Counter(canonical_card_key(card) for card in deck)
    type_counter = Counter(card_type_name(card) for card in deck)
    rarity_counter = Counter(card_rarity_name(card) for card in deck)
    upgraded_count = sum(1 for card in deck if card_upgrades(card) > 0)
    current_hp = current_hp_from_state(state_like)
    max_hp = max(1, max_hp_from_state(state_like))
    gold = gold_from_state(state_like)
    floor = floor_from_state(state_like)
    act = act_from_state(state_like)

    features = [
        clamp_scale(current_hp, 120.0),
        clamp_scale(max_hp, 120.0),
        safe_divide(current_hp, max_hp),
        safe_divide(max_hp - current_hp, max_hp),
        clamp_scale(gold, 500.0),
        clamp_scale(floor, 57.0),
        clamp_scale(deck_size, 45.0),
        clamp_scale(len(relics), 15.0),
        safe_divide(upgraded_count, max(1, deck_size)),
        clamp_scale(deck_counter.get("strike", 0), 8.0),
        clamp_scale(deck_counter.get("defend", 0), 8.0),
        clamp_scale(deck_counter.get("bash", 0), 3.0),
        safe_divide(type_counter.get("ATTACK", 0), max(1, deck_size)),
        safe_divide(type_counter.get("SKILL", 0), max(1, deck_size)),
        safe_divide(type_counter.get("POWER", 0), max(1, deck_size)),
        safe_divide(type_counter.get("CURSE", 0), max(1, deck_size)),
        safe_divide(rarity_counter.get("COMMON", 0), max(1, deck_size)),
        safe_divide(rarity_counter.get("UNCOMMON", 0), max(1, deck_size)),
        safe_divide(rarity_counter.get("RARE", 0), max(1, deck_size)),
        safe_divide(rarity_counter.get("COLORLESS", 0), max(1, deck_size)),
    ]
    for act_index in (1, 2, 3):
        features.append(1.0 if act == act_index else 0.0)
    features.extend(boss_one_hot(boss_name_from_state(state_like)))
    assert len(features) == STATE_NUMERIC_DIM
    return features


def build_state_vector(state_like):
    numeric = state_numeric_features(state_like)
    deck_hash = hashed_bag(deck_cards_from_state(state_like), CARD_BUCKETS, canonical_card_key)
    relic_hash = hashed_bag(relics_from_state(state_like), RELIC_BUCKETS, relic_key)
    return numeric + deck_hash + relic_hash


def candidate_vector(candidate, state_like):
    if isinstance(candidate, dict):
        candidate = canonicalize_card(candidate)
    deck = deck_cards_from_state(state_like)
    deck_counter = Counter(canonical_card_key(card) for card in deck)
    key = canonical_card_key(candidate)
    vector = [0.0] * CARD_BUCKETS
    if key:
        vector[stable_bucket(key, CARD_BUCKETS)] = 1.0
    type_name = card_type_name(candidate)
    type_features = [0.0] * CANDIDATE_TYPE_DIM
    if type_name in TYPE_INDEX:
        type_features[TYPE_INDEX[type_name]] = 1.0
    rarity_name = card_rarity_name(candidate)
    rarity_features = [0.0] * CANDIDATE_RARITY_DIM
    if rarity_name in RARITY_INDEX:
        rarity_features[RARITY_INDEX[rarity_name]] = 1.0
    existing_copies = deck_counter.get(key, 0)
    attack_ratio = safe_divide(sum(1 for card in deck if card_type_name(card) == "ATTACK"), max(1, len(deck)))
    skill_ratio = safe_divide(sum(1 for card in deck if card_type_name(card) == "SKILL"), max(1, len(deck)))
    power_ratio = safe_divide(sum(1 for card in deck if card_type_name(card) == "POWER"), max(1, len(deck)))
    extra_features = [
        clamp_scale(card_upgrades(candidate), 3.0),
        clamp_scale(existing_copies, 5.0),
        1.0 if existing_copies > 0 else 0.0,
        attack_ratio,
        skill_ratio,
        power_ratio,
    ]
    return vector + type_features + rarity_features + extra_features


class CardRewardPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.candidate_head = nn.Sequential(
            nn.Linear(128 + CANDIDATE_DIM, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.skip_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_state(self, state_tensor):
        return self.state_encoder(state_tensor)

    def score_candidate_with_hidden(self, state_hidden, candidate_tensor):
        if candidate_tensor.dim() == 2:
            candidate_input = torch.cat([state_hidden, candidate_tensor], dim=-1)
            return self.candidate_head(candidate_input).squeeze(-1)
        batch_size, candidate_count, _ = candidate_tensor.shape
        repeated_state = state_hidden.unsqueeze(1).expand(batch_size, candidate_count, state_hidden.shape[-1])
        candidate_input = torch.cat([repeated_state, candidate_tensor], dim=-1)
        return self.candidate_head(candidate_input).squeeze(-1)

    def score_candidate(self, state_tensor, candidate_tensor):
        state_hidden = self.encode_state(state_tensor)
        return self.score_candidate_with_hidden(state_hidden, candidate_tensor)

    def score_skip_with_hidden(self, state_hidden):
        return self.skip_head(state_hidden).squeeze(-1)

    def score_skip(self, state_tensor):
        state_hidden = self.encode_state(state_tensor)
        return self.score_skip_with_hidden(state_hidden)

    def forward(self, state_tensor, candidate_tensor):
        state_hidden = self.encode_state(state_tensor)
        batch_size, candidate_count, _ = candidate_tensor.shape
        candidate_scores = self.score_candidate_with_hidden(state_hidden, candidate_tensor)
        skip_score = self.score_skip_with_hidden(state_hidden).unsqueeze(1)
        return torch.cat([candidate_scores, skip_score], dim=1)


def sample_to_example(record):
    if record.get("record_type") != "expert_card_reward_choice":
        return None
    scenario = record.get("scenario") or {}
    reward_cards = list(scenario.get("reward_cards") or [])
    if len(reward_cards) != 3:
        return None
    choice = record.get("choice") or {}
    if choice.get("skip"):
        label = 3
    else:
        chosen_index = choice.get("chosen_index")
        if chosen_index not in (0, 1, 2):
            return None
        label = int(chosen_index)
    return {
        "scenario": scenario,
        "label": label,
    }


def load_expert_card_reward_examples(path):
    examples = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            example = sample_to_example(record)
            if example is not None:
                examples.append(example)
    return examples


def examples_to_tensors(examples, device):
    state_rows = []
    candidate_rows = []
    labels = []
    for example in examples:
        scenario = example["scenario"]
        state_rows.append(build_state_vector(scenario))
        candidate_rows.append([candidate_vector(card, scenario) for card in scenario.get("reward_cards", [])[:3]])
        labels.append(example["label"])
    return {
        "state": torch.tensor(state_rows, dtype=torch.float32, device=device),
        "candidate": torch.tensor(candidate_rows, dtype=torch.float32, device=device),
        "label": torch.tensor(labels, dtype=torch.long, device=device),
    }


def iter_expert_pairwise_examples(examples):
    for example in examples:
        scenario = example["scenario"]
        reward_cards = list(scenario.get("reward_cards") or [])[:3]
        label = int(example["label"])
        if len(reward_cards) != 3:
            continue
        options = [0, 1, 2, 3]
        for rejected in options:
            if rejected == label:
                continue
            yield {
                "scenario": scenario,
                "pos_index": label,
                "neg_index": rejected,
                "weight": 1.0,
            }


def pairwise_batch_to_tensors(batch, device):
    state_rows = []
    pos_rows = []
    neg_rows = []
    pos_skip = []
    neg_skip = []
    weights = []
    for item in batch:
        scenario = item["scenario"]
        reward_cards = list(scenario.get("reward_cards") or [])[:3]
        state_rows.append(build_state_vector(scenario))
        if "pos_card" in item or "neg_card" in item:
            pos_skip_flag = bool(item.get("pos_is_skip"))
            neg_skip_flag = bool(item.get("neg_is_skip"))
            pos_skip.append(1.0 if pos_skip_flag else 0.0)
            neg_skip.append(1.0 if neg_skip_flag else 0.0)
            pos_card = item.get("pos_card")
            neg_card = item.get("neg_card")
            pos_rows.append([0.0] * CANDIDATE_DIM if pos_skip_flag or pos_card is None else candidate_vector(pos_card, scenario))
            neg_rows.append([0.0] * CANDIDATE_DIM if neg_skip_flag or neg_card is None else candidate_vector(neg_card, scenario))
        else:
            if len(reward_cards) != 3:
                continue
            pos_index = int(item["pos_index"])
            neg_index = int(item["neg_index"])
            pos_skip.append(1.0 if pos_index == 3 else 0.0)
            neg_skip.append(1.0 if neg_index == 3 else 0.0)
            pos_rows.append([0.0] * CANDIDATE_DIM if pos_index == 3 else candidate_vector(reward_cards[pos_index], scenario))
            neg_rows.append([0.0] * CANDIDATE_DIM if neg_index == 3 else candidate_vector(reward_cards[neg_index], scenario))
        weights.append(float(item.get("weight", 1.0)))
    return {
        "state": torch.tensor(state_rows, dtype=torch.float32, device=device),
        "pos_candidate": torch.tensor(pos_rows, dtype=torch.float32, device=device),
        "neg_candidate": torch.tensor(neg_rows, dtype=torch.float32, device=device),
        "pos_is_skip": torch.tensor(pos_skip, dtype=torch.float32, device=device),
        "neg_is_skip": torch.tensor(neg_skip, dtype=torch.float32, device=device),
        "weight": torch.tensor(weights, dtype=torch.float32, device=device),
    }


def option_scores_from_pairwise_batch(model, batch):
    state_hidden = model.encode_state(batch["state"])
    pos_candidate_score = model.score_candidate_with_hidden(state_hidden, batch["pos_candidate"])
    neg_candidate_score = model.score_candidate_with_hidden(state_hidden, batch["neg_candidate"])
    skip_score = model.score_skip_with_hidden(state_hidden)
    pos_score = torch.where(batch["pos_is_skip"] > 0.5, skip_score, pos_candidate_score)
    neg_score = torch.where(batch["neg_is_skip"] > 0.5, skip_score, neg_candidate_score)
    return pos_score, neg_score


def split_examples(examples, valid_fraction=0.15, seed=0):
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    valid_size = max(1, int(len(shuffled) * valid_fraction)) if len(shuffled) >= 8 else max(0, len(shuffled) // 5)
    if valid_size == 0:
        return shuffled, []
    return shuffled[valid_size:], shuffled[:valid_size]


def batched(iterable, batch_size):
    for start in range(0, len(iterable), batch_size):
        yield iterable[start:start + batch_size]


def pairwise_examples_from_expert_examples(examples):
    return list(iter_expert_pairwise_examples(examples))


def evaluate_card_reward_model(model, examples, device):
    if not examples:
        return {"loss": 0.0, "accuracy": 0.0}
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for example_batch in batched(examples, 64):
            batch = examples_to_tensors(example_batch, device)
            logits = model(batch["state"], batch["candidate"])
            loss = F.cross_entropy(logits, batch["label"])
            predictions = logits.argmax(dim=1)
            total_loss += float(loss.item()) * len(example_batch)
            total_correct += int((predictions == batch["label"]).sum().item())
            total += len(example_batch)
    return {
        "loss": total_loss / float(total or 1),
        "accuracy": total_correct / float(total or 1),
    }


def train_card_reward_model(examples, device="cpu", epochs=80, batch_size=32, learning_rate=3e-4, valid_fraction=0.15, seed=0):
    train_examples, valid_examples = split_examples(examples, valid_fraction=valid_fraction, seed=seed)
    train_pairs = pairwise_examples_from_expert_examples(train_examples)
    valid_pairs = pairwise_examples_from_expert_examples(valid_examples)
    model = CardRewardPolicyNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    best_state = None
    best_valid = None
    for epoch in range(1, epochs + 1):
        model.train()
        random.Random(seed + epoch).shuffle(train_pairs)
        train_loss_sum = 0.0
        train_total = 0
        for pair_batch in batched(train_pairs, batch_size):
            batch = pairwise_batch_to_tensors(pair_batch, device)
            pos_score, neg_score = option_scores_from_pairwise_batch(model, batch)
            loss = (F.softplus(-(pos_score - neg_score)) * batch["weight"]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(pair_batch)
            train_total += len(pair_batch)
        train_metrics = {
            "loss": train_loss_sum / float(train_total or 1),
            "accuracy": evaluate_card_reward_model(model, train_examples, device)["accuracy"],
        }
        valid_metrics = evaluate_card_reward_model(model, valid_examples, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "valid_loss": valid_metrics["loss"],
                "valid_accuracy": valid_metrics["accuracy"],
            }
        )
        current_valid = (valid_metrics["accuracy"], -valid_metrics["loss"])
        if best_valid is None or current_valid > best_valid:
            best_valid = current_valid
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "train_pairs": len(train_pairs),
        "valid_pairs": len(valid_pairs),
        "history": history,
        "best_valid_accuracy": max((entry["valid_accuracy"] for entry in history), default=0.0),
    }


def save_card_reward_checkpoint(model, output_path, training_summary=None):
    payload = {
        "state_dict": model.state_dict(),
        "metadata": {
            "state_dim": STATE_DIM,
            "candidate_dim": CANDIDATE_DIM,
            "card_buckets": CARD_BUCKETS,
            "relic_buckets": RELIC_BUCKETS,
            "created_at": int(os.path.getmtime(__file__)),
        },
    }
    if training_summary is not None:
        payload["training_summary"] = training_summary
    torch.save(payload, output_path)


def load_card_reward_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model = CardRewardPolicyNetwork().to(device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint.get("training_summary", {}) if isinstance(checkpoint, dict) else {}


class CardRewardSelector:
    def __init__(self, checkpoint_path=None, device=None):
        repo_root = Path(__file__).resolve().parents[2]
        default_checkpoint = repo_root / "models" / "card_reward.pt"
        self.checkpoint_path = Path(
            checkpoint_path
            or os.environ.get("SPIRECOMM_CARD_REWARD_MODEL_PATH")
            or default_checkpoint
        )
        self.device = device or os.environ.get("SPIRECOMM_CARD_REWARD_DEVICE") or os.environ.get("SPIRECOMM_MODEL_DEVICE", "cpu")
        self.model = None
        self.branch_model = None
        self.branch_checkpoint_path = None
        self.last_branch_used = False
        if self.checkpoint_path.exists():
            self.model, _ = load_card_reward_checkpoint(str(self.checkpoint_path), device=self.device)
        branch_path = os.environ.get("SPIRECOMM_CARD_REWARD_BRANCH_MODEL_PATH")
        if branch_path:
            self.branch_checkpoint_path = Path(branch_path)
            if self.branch_checkpoint_path.exists():
                self.branch_model, _ = load_card_reward_checkpoint(str(self.branch_checkpoint_path), device=self.device)

    @property
    def available(self):
        return self.model is not None

    def choose(self, game_state, reward_cards, can_skip=True, *, return_scores: bool = True):
        self.last_branch_used = False
        if not self.available or len(reward_cards) == 0:
            return None
        state_vector = build_state_vector(game_state)
        candidate_vectors = [candidate_vector(card, game_state) for card in reward_cards]
        state_tensor = torch.tensor([state_vector], dtype=torch.float32, device=self.device)
        candidate_tensor = torch.tensor([candidate_vectors], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            state_hidden = self.model.encode_state(state_tensor)
            candidate_scores = self.model.score_candidate_with_hidden(state_hidden, candidate_tensor)[0]
            skip_score = self.model.skip_head(state_hidden).view(-1)[0]
            branch_candidate_scores = None
            branch_skip_score = None
            if self.branch_model is not None:
                branch_state_hidden = self.branch_model.encode_state(state_tensor)
                branch_candidate_scores = self.branch_model.score_candidate_with_hidden(branch_state_hidden, candidate_tensor)[0]
                branch_skip_score = self.branch_model.skip_head(branch_state_hidden).view(-1)[0]
        attack_bias = early_card_reward_attack_bias(game_state)
        if attack_bias > 0.0:
            attack_biases = [
                attack_bias if card_type_name(card) == "ATTACK" else 0.0
                for card in reward_cards
            ]
            attack_bias_tensor = torch.tensor(attack_biases, dtype=torch.float32, device=self.device)
            candidate_scores = candidate_scores + attack_bias_tensor
            if branch_candidate_scores is not None:
                branch_candidate_scores = branch_candidate_scores + attack_bias_tensor
        score_biases = card_score_biases_for_cards(reward_cards, game_state)
        if any(float(value) != 0.0 for value in score_biases):
            score_bias_tensor = torch.tensor(score_biases, dtype=torch.float32, device=self.device)
            candidate_scores = candidate_scores + score_bias_tensor
            if branch_candidate_scores is not None:
                branch_candidate_scores = branch_candidate_scores + score_bias_tensor
        if can_skip:
            skip_weight = early_card_reward_skip_weight(game_state)
            if skip_weight < 1.0:
                skip_score = skip_score + math.log(skip_weight)
                if branch_skip_score is not None:
                    branch_skip_score = branch_skip_score + math.log(skip_weight)
            logits = torch.cat([candidate_scores, skip_score.view(1)], dim=0)
            branch_logits = (
                torch.cat([branch_candidate_scores, branch_skip_score.view(1)], dim=0)
                if branch_candidate_scores is not None and branch_skip_score is not None
                else None
            )
        else:
            logits = candidate_scores
            branch_logits = branch_candidate_scores
        if branch_logits is not None and _env_bool("SPIRECOMM_CARD_REWARD_BRANCH_GATE_ENABLED", True):
            base_order = torch.argsort(logits, descending=True)
            branch_order = torch.argsort(branch_logits, descending=True)
            base_best = int(base_order[0].item())
            branch_best = int(branch_order[0].item())
            base_margin = float((logits[base_order[0]] - logits[base_order[1]]).item()) if len(logits) > 1 else float("inf")
            branch_margin = (
                float((branch_logits[branch_order[0]] - branch_logits[branch_order[1]]).item())
                if len(branch_logits) > 1
                else float("inf")
            )
            gate_margin_max = float(os.environ.get("SPIRECOMM_CARD_REWARD_BRANCH_GATE_MARGIN_MAX", "0.75"))
            branch_margin_min = float(os.environ.get("SPIRECOMM_CARD_REWARD_BRANCH_GATE_BRANCH_MARGIN_MIN", "0.10"))
            if branch_best != base_best and base_margin <= gate_margin_max and branch_margin >= branch_margin_min:
                logits = branch_logits
                self.last_branch_used = True
        choice = int(torch.argmax(logits).item())
        return {
            "choice_index": choice,
            "scores": [float(value) for value in logits.detach().cpu().tolist()] if return_scores else [],
        }


class CardTargetSelector:
    def __init__(self, checkpoint_path=None, env_var=None, default_name="card_target.pt", device=None, apply_card_score_bias=False):
        repo_root = Path(__file__).resolve().parents[2]
        default_checkpoint = repo_root / "models" / default_name
        self.checkpoint_path = Path(
            checkpoint_path
            or (os.environ.get(env_var) if env_var else None)
            or default_checkpoint
        )
        self.device = device or os.environ.get("SPIRECOMM_CARD_TARGET_DEVICE") or os.environ.get("SPIRECOMM_MODEL_DEVICE", "cpu")
        self.model = None
        self.training_summary = {}
        self.rate_prior = {}
        self.rate_prior_eps = 1e-6
        self.rate_prior_weight = 2.0
        self.apply_card_score_bias = bool(apply_card_score_bias)
        if self.checkpoint_path.exists():
            self.model, self.training_summary = load_card_reward_checkpoint(str(self.checkpoint_path), device=self.device)
            self.rate_prior = dict(self.training_summary.get("upgrade_rate_prior") or {})
            self.rate_prior_eps = float(self.training_summary.get("upgrade_rate_prior_eps", self.rate_prior_eps) or self.rate_prior_eps)
            self.rate_prior_weight = float(
                self.training_summary.get("upgrade_rate_prior_weight", self.rate_prior_weight) or self.rate_prior_weight
            )

    @property
    def available(self):
        return self.model is not None

    def choose(self, game_state, cards, *, return_scores: bool = True):
        if not self.available or len(cards) == 0:
            return None
        state_vector = build_state_vector(game_state)
        candidate_vectors = [candidate_vector(card, game_state) for card in cards]
        state_tensor = torch.tensor([state_vector], dtype=torch.float32, device=self.device)
        candidate_tensor = torch.tensor([candidate_vectors], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            state_hidden = self.model.encode_state(state_tensor)
            scores = self.model.score_candidate_with_hidden(state_hidden, candidate_tensor)[0]
            if self.rate_prior:
                prior_weight = self.rate_prior_weight
                if isinstance(self, UpgradeTargetSelector):
                    try:
                        prior_weight = float(os.environ.get("SPIRECOMM_UPGRADE_RATE_PRIOR_WEIGHT_OVERRIDE", prior_weight))
                    except (TypeError, ValueError):
                        prior_weight = self.rate_prior_weight
                prior_logits = [
                    prior_weight * math.log(float(self.rate_prior.get(canonical_card_key(card), 0.0)) + self.rate_prior_eps)
                    for card in cards
                ]
                scores = scores + torch.tensor(prior_logits, dtype=torch.float32, device=self.device)
            if self.apply_card_score_bias:
                score_biases = card_score_biases_for_cards(cards, game_state)
                if any(float(value) != 0.0 for value in score_biases):
                    if isinstance(self, UpgradeTargetSelector):
                        try:
                            bias_scale = float(os.environ.get("SPIRECOMM_UPGRADE_CARD_SCORE_BIAS_SCALE", "1.0"))
                        except (TypeError, ValueError):
                            bias_scale = 1.0
                        if bias_scale != 1.0:
                            score_biases = [float(value) * bias_scale for value in score_biases]
                    scores = scores + torch.tensor(score_biases, dtype=torch.float32, device=self.device)
        choice = int(torch.argmax(scores).item())
        return {
            "choice_index": choice,
            "scores": [float(value) for value in scores.detach().cpu().tolist()] if return_scores else [],
        }


class UpgradeTargetSelector(CardTargetSelector):
    def __init__(self, checkpoint_path=None, device=None):
        super().__init__(
            checkpoint_path=checkpoint_path,
            env_var="SPIRECOMM_UPGRADE_TARGET_MODEL_PATH",
            default_name="upgrade_target.pt",
            device=device,
            apply_card_score_bias=True,
        )


class PurgeTargetSelector(CardTargetSelector):
    def __init__(self, checkpoint_path=None, device=None):
        super().__init__(
            checkpoint_path=checkpoint_path,
            env_var="SPIRECOMM_PURGE_TARGET_MODEL_PATH",
            default_name="purge_target.pt",
            device=device,
            apply_card_score_bias=False,
        )
