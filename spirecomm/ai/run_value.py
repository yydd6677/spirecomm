from __future__ import annotations

import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

from spirecomm.ai.card_reward_model import normalize_token, stable_bucket
from spirecomm.ai.torch_compat import nn, require_torch, torch


RUN_VALUE_CHECKPOINT_VERSION = "run_value_v1"
RUN_ACTION_POLICY_CHECKPOINT_VERSION = "run_action_policy_v1"

CARD_BUCKETS = 256
RELIC_BUCKETS = 128
POTION_BUCKETS = 64
MONSTER_BUCKETS = 64
CHOICE_BUCKETS = 256
FUTURE_CARD_POOL_BUCKETS = 256
FUTURE_RELIC_POOL_BUCKETS = 128
FUTURE_BOSS_RELIC_POOL_BUCKETS = 64
FUTURE_MISC_BUCKETS = 64
MAP_BUCKETS = 128
REACHABLE_MAP_BUCKETS = 128
HAND_CARD_BUCKETS = 128
DRAW_CARD_BUCKETS = 128
DISCARD_CARD_BUCKETS = 128
EXHAUST_CARD_BUCKETS = 64
POWER_BUCKETS = 128
ACTION_BUCKETS = 128
STATE_NUMERIC_DIM = 224
ACTION_NUMERIC_DIM = 48
BASE_STATE_FEATURE_DIM = (
    STATE_NUMERIC_DIM
    + CARD_BUCKETS
    + RELIC_BUCKETS
    + POTION_BUCKETS
    + MONSTER_BUCKETS
    + CHOICE_BUCKETS
    + FUTURE_CARD_POOL_BUCKETS
    + FUTURE_RELIC_POOL_BUCKETS
    + FUTURE_BOSS_RELIC_POOL_BUCKETS
    + FUTURE_MISC_BUCKETS
    + MAP_BUCKETS
    + REACHABLE_MAP_BUCKETS
    + HAND_CARD_BUCKETS
    + DRAW_CARD_BUCKETS
    + DISCARD_CARD_BUCKETS
    + EXHAUST_CARD_BUCKETS
    + POWER_BUCKETS
)
AUG_CARD_ZONE_BUCKETS = 512
AUG_FUTURE_CARD_POOL_BUCKETS = 512
AUG_FUTURE_RELIC_POOL_BUCKETS = 256
AUG_CHOICE_BUCKETS = 256
AUG_MAP_BUCKETS = 256
AUG_POWER_BUCKETS = 128
AUG_FEATURE_DIM = (
    AUG_CARD_ZONE_BUCKETS
    + AUG_FUTURE_CARD_POOL_BUCKETS
    + AUG_FUTURE_RELIC_POOL_BUCKETS
    + AUG_CHOICE_BUCKETS
    + AUG_MAP_BUCKETS
    + AUG_POWER_BUCKETS
)
STATE_FEATURE_DIM = BASE_STATE_FEATURE_DIM + AUG_FEATURE_DIM
ACTION_FEATURE_DIM = ACTION_NUMERIC_DIM + ACTION_BUCKETS
ACTION_CANDIDATE_FEATURE_DIM = STATE_FEATURE_DIM * 3 + ACTION_FEATURE_DIM

RUN_VALUE_OUTPUTS = (
    "remaining_floor",
    "final_floor",
    "win_logit",
    "act1_clear_logit",
    "act2_clear_logit",
    "act3_clear_logit",
    "death_next_3_logit",
    "death_next_6_logit",
)

ACTION_KIND_ORDER = [
    "card",
    "potion",
    "end",
    "card_reward",
    "skip",
    "proceed",
    "reward_gold",
    "reward_relic",
    "reward_key",
    "reward_potion",
    "discard_potion",
    "map",
    "shop",
    "campfire",
    "event",
    "boss_relic",
    "card_select",
    "neow",
    "treasure",
    "confirm",
    "raw",
    "not_implemented",
]
PHASE_ORDER = [
    "NEOW",
    "MAP",
    "COMBAT",
    "CARD_REWARD",
    "CARD_SELECT",
    "EVENT",
    "SHOP",
    "CAMPFIRE",
    "TREASURE",
    "BOSS_RELIC",
    "GAME_OVER",
    "COMPLETE",
    "VICTORY",
]
ROOM_ORDER = [
    "MonsterRoom",
    "MonsterRoomElite",
    "MonsterRoomBoss",
    "EventRoom",
    "ShopRoom",
    "RestRoom",
    "TreasureRoom",
    "TreasureRoomBoss",
    "Map",
    "NeowRoom",
]
CARD_TYPES = ["ATTACK", "SKILL", "POWER", "STATUS", "CURSE"]
CARD_RARITIES = ["BASIC", "COMMON", "UNCOMMON", "RARE", "COLORLESS", "CURSE", "SPECIAL"]
EXPLICIT_RUN_FEATURE_DIM = 25


def _clip(value: float, scale: float, lo: float = -5.0, hi: float = 5.0) -> float:
    if scale <= 0.0:
        return 0.0
    return max(lo, min(hi, float(value) / float(scale)))


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * int(size)
    if 0 <= int(index) < int(size):
        values[int(index)] = 1.0
    return values


def _hash_fraction(value: Any) -> float:
    token = str(value or "")
    if not token:
        return 0.0
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") / float(2**32 - 1)


def _card_key(card: dict[str, Any]) -> str:
    return normalize_token(card.get("card_id") or card.get("id") or card.get("name") or "")


def _relic_key(relic: dict[str, Any]) -> str:
    return normalize_token(relic.get("relic_id") or relic.get("id") or relic.get("name") or "")


def _potion_key(potion: dict[str, Any]) -> str:
    return normalize_token(potion.get("potion_id") or potion.get("id") or potion.get("name") or "")


def _monster_key(monster: dict[str, Any]) -> str:
    return normalize_token(monster.get("monster_id") or monster.get("id") or monster.get("name") or "")


def _power_key(power: dict[str, Any]) -> str:
    return normalize_token(power.get("power_id") or power.get("id") or power.get("name") or "")


def _token_key(value: Any) -> str:
    if isinstance(value, dict):
        return normalize_token(
            value.get("card_id")
            or value.get("relic_id")
            or value.get("potion_id")
            or value.get("monster_id")
            or value.get("item_id")
            or value.get("event_id")
            or value.get("id")
            or value.get("name")
            or value.get("label")
            or value.get("reward_type")
            or value.get("symbol")
            or ""
        )
    return normalize_token(value)


def _safe_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, (list, tuple)) else []


def _bag(items: Iterable[Any], bucket_count: int, key_fn) -> list[float]:
    vector = [0.0] * int(bucket_count)
    total = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        key = key_fn(item)
        if not key:
            continue
        vector[stable_bucket(key, bucket_count)] += 1.0
        total += 1
    if total > 0:
        inv = 1.0 / math.sqrt(float(total))
        vector = [value * inv for value in vector]
    return vector


def _token_bag(items: Iterable[Any], bucket_count: int, *, prefix: str = "", ordered: bool = False) -> list[float]:
    vector = [0.0] * int(bucket_count)
    weight_sq = 0.0
    total = 0
    for index, item in enumerate(items):
        key = _token_key(item)
        if not key:
            continue
        token = f"{prefix}:{key}" if prefix else key
        weight = 1.0 / math.sqrt(float(index + 1)) if ordered else 1.0
        vector[stable_bucket(token, bucket_count)] += weight
        weight_sq += weight * weight
        total += 1
    if total > 0:
        inv = 1.0 / math.sqrt(max(1.0, weight_sq))
        vector = [value * inv for value in vector]
    return vector


def _add_token(vector: list[float], token: Any, *, prefix: str = "", weight: float = 1.0) -> None:
    key = _token_key(token)
    if not key:
        return
    value = f"{prefix}:{key}" if prefix else key
    vector[stable_bucket(value, len(vector))] += float(weight)


def _combat_state(state: dict[str, Any]) -> dict[str, Any]:
    combat = state.get("combat_state")
    return combat if isinstance(combat, dict) else {}


def _player_from_state(state: dict[str, Any]) -> dict[str, Any]:
    combat = _combat_state(state)
    player = combat.get("player") if combat else None
    return player if isinstance(player, dict) else {}


def _monsters_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    monsters = _combat_state(state).get("monsters") if _combat_state(state) else []
    return [monster for monster in _safe_list(monsters) if isinstance(monster, dict)]


def _cards_from_zone(state: dict[str, Any], zone: str) -> list[dict[str, Any]]:
    cards = _combat_state(state).get(zone) if _combat_state(state) else []
    return [card for card in _safe_list(cards) if isinstance(card, dict)]


def _deck_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [card for card in _safe_list(state.get("deck")) if isinstance(card, dict)]


def _relics(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [relic for relic in _safe_list(state.get("relics")) if isinstance(relic, dict)]


def _potions(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [potion for potion in _safe_list(state.get("potions")) if isinstance(potion, dict)]


def _screen_state(state: dict[str, Any]) -> dict[str, Any]:
    screen = state.get("screen_state")
    return screen if isinstance(screen, dict) else {}


def _screen_cards(screen: dict[str, Any]) -> list[dict[str, Any]]:
    cards = _safe_list(screen.get("cards"))
    return [card for card in cards if isinstance(card, dict)]


def _screen_relics(screen: dict[str, Any]) -> list[dict[str, Any]]:
    relics = _safe_list(screen.get("relics"))
    return [relic for relic in relics if isinstance(relic, dict)]


def _screen_potions(screen: dict[str, Any]) -> list[dict[str, Any]]:
    potions = _safe_list(screen.get("potions"))
    return [potion for potion in potions if isinstance(potion, dict)]


def _card_type_counts(cards: list[dict[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in CARD_TYPES}
    for card in cards:
        kind = str(card.get("type") or "").upper()
        if kind in counts:
            counts[kind] += 1
    return counts


def _card_rarity_counts(cards: list[dict[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in CARD_RARITIES}
    for card in cards:
        rarity = str(card.get("rarity") or "").upper()
        if rarity not in counts:
            rarity = "SPECIAL" if rarity else "COMMON"
        counts[rarity] = counts.get(rarity, 0) + 1
    return counts


def _deck_readiness(cards: list[dict[str, Any]]) -> list[float]:
    size = max(1, len(cards))
    attack_damage = 0.0
    block = 0.0
    magic = 0.0
    upgraded = 0
    starter = 0
    aoe = 0
    exhaust = 0
    innate = 0
    draw_like = 0
    energy_like = 0
    for card in cards:
        key = _card_key(card)
        attack_damage += float(card.get("base_damage") or 0)
        block += float(card.get("base_block") or 0)
        magic += float(card.get("base_magic") or card.get("magic_number") or 0)
        upgraded += 1 if int(card.get("upgrades") or 0) > 0 else 0
        starter += 1 if key in {"strike", "striker", "defend", "defendr", "bash"} else 0
        name = normalize_token(card.get("name") or key)
        target = str(card.get("target") or "").upper()
        aoe += 1 if "ALL" in target or name in {"cleave", "immolate", "whirlwind", "reaper"} else 0
        exhaust += 1 if bool(card.get("exhausts") or card.get("exhaust")) else 0
        innate += 1 if bool(card.get("innate") or card.get("is_innate")) else 0
        draw_like += 1 if name in {"shrugitoff", "pommelstrike", "battletrance", "offering", "darkembrace"} else 0
        energy_like += 1 if name in {"seeingred", "bloodletting", "offering", "sentinel", "corruption"} else 0
    type_counts = _card_type_counts(cards)
    rarity_counts = _card_rarity_counts(cards)
    return [
        _clip(len(cards), 60.0, 0.0, 3.0),
        _clip(attack_damage / size, 20.0, 0.0, 3.0),
        _clip(block / size, 15.0, 0.0, 3.0),
        _clip(magic / size, 10.0, 0.0, 3.0),
        _clip(upgraded / size, 1.0, 0.0, 1.0),
        _clip(starter, 10.0, 0.0, 2.0),
        _clip(aoe, 5.0, 0.0, 2.0),
        _clip(exhaust, 8.0, 0.0, 2.0),
        _clip(innate, 5.0, 0.0, 2.0),
        _clip(draw_like, 8.0, 0.0, 2.0),
        _clip(energy_like, 6.0, 0.0, 2.0),
        _clip(type_counts.get("ATTACK", 0) / size, 1.0, 0.0, 1.0),
        _clip(type_counts.get("SKILL", 0) / size, 1.0, 0.0, 1.0),
        _clip(type_counts.get("POWER", 0) / size, 1.0, 0.0, 1.0),
        _clip(rarity_counts.get("RARE", 0) / size, 1.0, 0.0, 1.0),
        _clip(rarity_counts.get("CURSE", 0), 10.0, 0.0, 2.0),
    ]


def _card_cost(card: dict[str, Any]) -> float:
    value = card.get("cost")
    if value is None:
        value = card.get("base_cost")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _count_cards(cards: list[dict[str, Any]], names: set[str]) -> int:
    count = 0
    for card in cards:
        key = _card_key(card)
        name = normalize_token(card.get("name") or key)
        if key in names or name in names:
            count += 1
    return count


def _deck_explicit_features(cards: list[dict[str, Any]]) -> list[float]:
    size = max(1, len(cards))
    attack_cards = [card for card in cards if str(card.get("type") or "").upper() == "ATTACK"]
    skill_cards = [card for card in cards if str(card.get("type") or "").upper() == "SKILL"]
    power_cards = [card for card in cards if str(card.get("type") or "").upper() == "POWER"]
    zero_one_cost_attacks = [
        card
        for card in attack_cards
        if _card_cost(card) <= 1.0 and float(card.get("base_damage") or 0.0) > 0.0
    ]
    frontload_damage = sum(float(card.get("base_damage") or 0.0) for card in attack_cards)
    cheap_frontload = sum(float(card.get("base_damage") or 0.0) for card in zero_one_cost_attacks)
    block_total = sum(float(card.get("base_block") or 0.0) for card in skill_cards)
    draw_names = {
        "battletrance",
        "darkembrace",
        "evolve",
        "offering",
        "pommelstrike",
        "shrugitoff",
        "warcry",
        "burningpact",
    }
    strength_names = {
        "flex",
        "inflame",
        "spotweakness",
        "demonform",
        "limitbreak",
        "reaper",
    }
    exhaust_payoff_names = {"feel no pain", "feelnopain", "darkembrace", "corruption"}
    exhaust_enabler_names = {
        "secondwind",
        "secondewind",
        "burningpact",
        "truegrit",
        "true grit",
        "fiendfire",
        "fiend fire",
        "exhume",
        "corruption",
    }
    aoe_names = {"cleave", "thunderclap", "whirlwind", "immolate", "reaper"}
    high_impact_upgrade_names = {
        "bash",
        "armaments",
        "battletrance",
        "demonform",
        "inflame",
        "limitbreak",
        "offering",
        "shockwave",
        "spotweakness",
        "uppercut",
        "whirlwind",
    }
    unupgraded_high_impact = 0
    for card in cards:
        key = _card_key(card)
        name = normalize_token(card.get("name") or key)
        if (key in high_impact_upgrade_names or name in high_impact_upgrade_names) and int(card.get("upgrades") or 0) <= 0:
            unupgraded_high_impact += 1
    starter_basic = _count_cards(cards, {"strike", "striker", "defend", "defendr"})
    curses = sum(1 for card in cards if str(card.get("type") or "").upper() == "CURSE")
    status_cards = sum(1 for card in cards if str(card.get("type") or "").upper() == "STATUS")
    exhaust_count = 0
    for card in cards:
        key = _card_key(card)
        name = normalize_token(card.get("name") or key)
        if bool(card.get("exhausts") or card.get("exhaust")) or key in exhaust_enabler_names or name in exhaust_enabler_names:
            exhaust_count += 1
    return [
        _clip(frontload_damage / size, 18.0, 0.0, 3.0),
        _clip(cheap_frontload / size, 10.0, 0.0, 3.0),
        _clip(block_total / size, 12.0, 0.0, 3.0),
        _clip(len(power_cards) / size, 0.4, 0.0, 3.0),
        _clip(_count_cards(cards, draw_names), 8.0, 0.0, 2.0),
        _clip(_count_cards(cards, strength_names), 6.0, 0.0, 2.0),
        _clip(exhaust_count, 10.0, 0.0, 2.0),
        _clip(_count_cards(cards, exhaust_payoff_names), 5.0, 0.0, 2.0),
        _clip(_count_cards(cards, aoe_names), 5.0, 0.0, 2.0),
        _clip(unupgraded_high_impact, 8.0, 0.0, 2.0),
        _clip(starter_basic + curses, 12.0, 0.0, 2.0),
        _clip((curses + status_cards) / size, 0.5, 0.0, 2.0),
    ]


def _path_features(state: dict[str, Any]) -> list[float]:
    actions = _safe_list(state.get("choice_list")) or _safe_list(state.get("legal_actions"))
    symbols = [str(action.get("symbol") or action.get("room_symbol") or action.get("name") or "") for action in actions if isinstance(action, dict)]
    counts = {symbol: sum(1 for value in symbols if value == symbol) for symbol in {"M", "E", "R", "$", "?", "T", "BOSS"}}
    screen = _screen_state(state)
    node = screen.get("current_node") if isinstance(screen.get("current_node"), dict) else {}
    return [
        _clip(len(actions), 8.0, 0.0, 3.0),
        _clip(counts.get("M", 0), 4.0, 0.0, 2.0),
        _clip(counts.get("E", 0), 4.0, 0.0, 2.0),
        _clip(counts.get("R", 0), 4.0, 0.0, 2.0),
        _clip(counts.get("$", 0), 4.0, 0.0, 2.0),
        _clip(counts.get("?", 0), 4.0, 0.0, 2.0),
        _bool(bool(screen.get("boss_available")) or counts.get("BOSS", 0) > 0),
        _clip(int(node.get("x") or 0), 7.0, -1.0, 1.0),
        _clip(int(node.get("y") or 0), 15.0, -1.0, 1.0),
    ]


def _choice_tokens_from_action(action: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    kind = str(action.get("kind") or "")
    item_kind = str(action.get("item_kind") or "")
    for key in (
        "kind",
        "item_kind",
        "event_id",
        "name",
        "label",
        "bonus",
        "drawback",
        "reward_type",
        "symbol",
        "mode",
        "card_id",
        "relic_id",
        "potion_id",
        "item_id",
    ):
        value = action.get(key)
        token = _token_key(value)
        if token:
            tokens.append(f"{key}:{token}")
            if key in {"card_id", "relic_id", "potion_id", "item_id", "event_id"}:
                tokens.append(f"{kind}:{token}")
            if item_kind:
                tokens.append(f"{item_kind}:{token}")
    for nested_key in ("card", "relic", "potion", "item"):
        nested = action.get(nested_key)
        if not isinstance(nested, dict):
            continue
        nested_id = _token_key(nested)
        if nested_id:
            tokens.append(f"{nested_key}:{nested_id}")
            tokens.append(f"{kind}:{nested_key}:{nested_id}")
        for key in ("type", "rarity", "tier", "color", "target"):
            token = _token_key(nested.get(key))
            if token:
                tokens.append(f"{nested_key}_{key}:{token}")
    for symbol in _safe_list(action.get("next_symbols")):
        token = _token_key(symbol)
        if token:
            tokens.append(f"next:{token}")
    return tokens


def _choice_bag(state: dict[str, Any], screen: dict[str, Any]) -> list[float]:
    actions = [action for action in (_safe_list(state.get("choice_list")) or _safe_list(state.get("legal_actions"))) if isinstance(action, dict)]
    vector = [0.0] * CHOICE_BUCKETS
    weight_sq = 0.0
    for index, action in enumerate(actions):
        slot_weight = 1.0 / math.sqrt(float(index + 1))
        for token in _choice_tokens_from_action(action):
            _add_token(vector, token, prefix="choice", weight=slot_weight)
            weight_sq += slot_weight * slot_weight
    for prefix, items in (
        ("screen_card", _screen_cards(screen)),
        ("screen_relic", _screen_relics(screen)),
        ("screen_potion", _screen_potions(screen)),
        ("screen_reward", _safe_list(screen.get("rewards"))),
        ("screen_option", _safe_list(screen.get("options"))),
    ):
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            item_weight = 1.0 / math.sqrt(float(index + 1))
            _add_token(vector, item, prefix=prefix, weight=item_weight)
            weight_sq += item_weight * item_weight
            for key in ("reward_type", "card_id", "relic_id", "potion_id", "event_id", "name", "label", "type", "rarity", "tier"):
                value = item.get(key)
                if value:
                    _add_token(vector, value, prefix=f"{prefix}_{key}", weight=item_weight)
    if weight_sq > 0.0:
        inv = 1.0 / math.sqrt(weight_sq)
        vector = [value * inv for value in vector]
    return vector


def _choice_numeric(state: dict[str, Any], screen: dict[str, Any]) -> list[float]:
    actions = [action for action in (_safe_list(state.get("choice_list")) or _safe_list(state.get("legal_actions"))) if isinstance(action, dict)]
    by_kind: dict[str, int] = {}
    by_item_kind: dict[str, int] = {}
    prices: list[float] = []
    amounts: list[float] = []
    card_damage = 0.0
    card_block = 0.0
    card_magic = 0.0
    card_cost = 0.0
    card_count = 0
    affordable = 0
    skip_like = 0
    leave_like = 0
    next_counts = {symbol: 0 for symbol in ("M", "E", "R", "$", "?", "T", "BOSS")}
    gold = int(state.get("gold") or 0)
    for action in actions:
        kind = str(action.get("kind") or "")
        item_kind = str(action.get("item_kind") or "")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        by_item_kind[item_kind] = by_item_kind.get(item_kind, 0) + 1
        if action.get("price") is not None:
            price = float(action.get("price") or 0.0)
            prices.append(price)
            if price <= gold:
                affordable += 1
        if action.get("amount") is not None:
            amounts.append(float(action.get("amount") or 0.0))
        name = normalize_token(action.get("name") or action.get("label") or "")
        skip_like += 1 if kind == "skip" or name in {"skip", "leave"} else 0
        leave_like += 1 if item_kind == "leave" or name in {"leave", "proceed"} else 0
        card = action.get("card")
        if isinstance(card, dict):
            card_count += 1
            card_damage += float(card.get("base_damage") or 0.0)
            card_block += float(card.get("base_block") or 0.0)
            card_magic += float(card.get("base_magic") or card.get("magic_number") or 0.0)
            card_cost += float(card.get("cost") if card.get("cost") is not None else card.get("base_cost") or 0.0)
        for symbol in _safe_list(action.get("next_symbols")):
            value = str(symbol or "")
            if value in next_counts:
                next_counts[value] += 1
    screen_rewards = [item for item in _safe_list(screen.get("rewards")) if isinstance(item, dict)]
    screen_options = [item for item in _safe_list(screen.get("options")) if isinstance(item, dict)]
    screen_cards = _screen_cards(screen)
    screen_relics = _screen_relics(screen)
    screen_potions = _screen_potions(screen)
    denom = max(1, len(actions))
    price_min = min(prices) if prices else 0.0
    price_max = max(prices) if prices else 0.0
    price_mean = sum(prices) / len(prices) if prices else 0.0
    return [
        _clip(len(actions), 20.0, 0.0, 3.0),
        _clip(by_kind.get("card_reward", 0), 5.0, 0.0, 2.0),
        _clip(by_kind.get("shop", 0), 20.0, 0.0, 3.0),
        _clip(by_kind.get("event", 0), 6.0, 0.0, 2.0),
        _clip(by_kind.get("boss_relic", 0), 5.0, 0.0, 2.0),
        _clip(by_kind.get("map", 0), 8.0, 0.0, 2.0),
        _clip(by_kind.get("neow", 0), 5.0, 0.0, 2.0),
        _clip(by_item_kind.get("card", 0), 10.0, 0.0, 2.0),
        _clip(by_item_kind.get("relic", 0), 5.0, 0.0, 2.0),
        _clip(by_item_kind.get("potion", 0), 5.0, 0.0, 2.0),
        _clip(affordable, 12.0, 0.0, 2.0),
        _clip(price_min, 300.0, 0.0, 3.0),
        _clip(price_max, 300.0, 0.0, 3.0),
        _clip(price_mean, 300.0, 0.0, 3.0),
        _clip(sum(amounts) / max(1, len(amounts)), 300.0, -3.0, 3.0),
        _clip(skip_like, 4.0, 0.0, 2.0),
        _clip(leave_like, 4.0, 0.0, 2.0),
        _clip(card_count, 5.0, 0.0, 2.0),
        _clip(card_damage / max(1, card_count), 30.0, 0.0, 3.0),
        _clip(card_block / max(1, card_count), 20.0, 0.0, 3.0),
        _clip(card_magic / max(1, card_count), 10.0, 0.0, 3.0),
        _clip(card_cost / max(1, card_count), 5.0, -1.0, 3.0),
        _clip(len(screen_rewards), 8.0, 0.0, 2.0),
        _clip(len(screen_options), 6.0, 0.0, 2.0),
        _clip(len(screen_cards), 8.0, 0.0, 2.0),
        _clip(len(screen_relics), 5.0, 0.0, 2.0),
        _clip(len(screen_potions), 5.0, 0.0, 2.0),
        _clip(next_counts["M"], 4.0, 0.0, 2.0),
        _clip(next_counts["E"], 4.0, 0.0, 2.0),
        _clip(next_counts["R"], 4.0, 0.0, 2.0),
        _clip(next_counts["$"], 4.0, 0.0, 2.0),
        _clip(next_counts["?"], 4.0, 0.0, 2.0),
        _clip(next_counts["T"], 4.0, 0.0, 2.0),
        _clip(next_counts["BOSS"], 2.0, 0.0, 2.0),
        _clip(sum(1 for action in actions if action.get("choice_index") is not None) / denom, 1.0, 0.0, 1.0),
    ]


def _pool_items(state: dict[str, Any], keys: Iterable[str]) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for key in keys:
        for item in _safe_list(state.get(key)):
            items.append((key, item))
    return items


def _prefixed_ordered_pool_bag(items: Iterable[tuple[str, Any]], bucket_count: int) -> list[float]:
    vector = [0.0] * int(bucket_count)
    weight_sq = 0.0
    for index, (prefix, item) in enumerate(items):
        key = _token_key(item)
        if not key:
            continue
        weight = 1.0 / math.sqrt(float(index + 1))
        _add_token(vector, key, prefix=prefix, weight=weight)
        # Coarser prefix-only mass tells the model how much of each future pool remains.
        _add_token(vector, prefix, prefix="pool_kind", weight=0.15 * weight)
        weight_sq += weight * weight + (0.15 * weight) * (0.15 * weight)
    if weight_sq > 0.0:
        inv = 1.0 / math.sqrt(weight_sq)
        vector = [value * inv for value in vector]
    return vector


def _rng_summary_features(state: dict[str, Any], *, include_hash: bool = True) -> list[float]:
    rng_trace = state.get("rng_trace")
    if not isinstance(rng_trace, dict):
        return [0.0] * 12
    streams = ("card", "misc", "monster", "relic", "treasure", "event", "neow", "shuffle")
    values: list[float] = []
    total_events = 0
    weighted_result = 0.0
    for stream in streams:
        entries = _safe_list(rng_trace.get(stream))
        total_events += len(entries)
        if entries:
            tail = entries[-4:]
            numeric_results = [float(entry.get("result")) for entry in tail if isinstance(entry, dict) and isinstance(entry.get("result"), (int, float))]
            weighted_result += sum(numeric_results[-2:]) if numeric_results else 0.0
        values.append(_clip(len(entries), 200.0, 0.0, 3.0))
    values.extend(
        [
            _clip(total_events, 800.0, 0.0, 4.0),
            _clip(weighted_result, 8.0, -4.0, 4.0),
            _hash_fraction(state.get("seed")) if include_hash else 0.0,
            _hash_fraction(state.get("dungeon_id")) if include_hash else 0.0,
        ]
    )
    return values[:12]


def _rng_state_features(state: dict[str, Any], *, include_hash: bool = True) -> list[float]:
    rng_state = state.get("rng_state")
    if not isinstance(rng_state, dict):
        return [0.0] * 36
    streams = (
        "card",
        "relic",
        "potion",
        "event",
        "monster",
        "merchant",
        "treasure",
        "map",
        "shuffle",
        "misc",
        "ai",
        "monster_hp",
    )
    values: list[float] = []
    for name in streams:
        info = rng_state.get(name)
        if not isinstance(info, dict):
            values.extend([0.0, 0.0, 0.0])
            continue
        counter = int(info.get("counter") or 0)
        call_count = int(info.get("call_count") or 0)
        seed0 = int(info.get("seed0") or 0)
        seed1 = int(info.get("seed1") or 0)
        values.extend(
            [
                _clip(counter, 500.0, 0.0, 4.0),
                _clip(call_count, 500.0, 0.0, 4.0),
                _hash_fraction(f"{name}:{seed0}:{seed1}") if include_hash else 0.0,
            ]
        )
    return values[:36]


def _map_nodes(state: dict[str, Any]) -> list[dict[str, Any]]:
    map_state = state.get("map_state")
    if not isinstance(map_state, dict):
        return []
    return [node for node in _safe_list(map_state.get("nodes")) if isinstance(node, dict)]


def _current_map_y(state: dict[str, Any]) -> int:
    map_state = state.get("map_state")
    if isinstance(map_state, dict):
        current = map_state.get("current_node")
        if isinstance(current, dict) and current.get("y") is not None:
            return int(current.get("y") or -1)
    screen_current = _screen_state(state).get("current_node")
    if isinstance(screen_current, dict) and screen_current.get("y") is not None:
        return int(screen_current.get("y") or -1)
    floor = int(state.get("floor") or 0)
    return (floor - 1) % 17 if floor > 0 else -1


def _map_state_features(state: dict[str, Any]) -> list[float]:
    nodes = _map_nodes(state)
    if not nodes:
        return [0.0] * 28
    current_y = _current_map_y(state)
    future = [node for node in nodes if int(node.get("y") or 0) > current_y]
    near = [node for node in future if int(node.get("y") or 0) <= current_y + 5]
    symbols = ("M", "E", "R", "$", "?", "T", "BOSS", "E_GREEN")

    def count(items: list[dict[str, Any]], symbol: str) -> int:
        if symbol == "E_GREEN":
            return sum(1 for node in items if str(node.get("symbol") or "") == "E" and bool(node.get("emerald")))
        return sum(1 for node in items if str(node.get("symbol") or "") == symbol)

    child_counts = [len(_safe_list(node.get("children"))) for node in future]
    values = [
        _clip(len(nodes), 120.0, 0.0, 3.0),
        _clip(len(future), 120.0, 0.0, 3.0),
        _clip(len(near), 40.0, 0.0, 3.0),
        _clip(current_y, 15.0, -1.0, 1.0),
        _clip(max(child_counts) if child_counts else 0, 6.0, 0.0, 2.0),
        _clip(sum(child_counts) / max(1, len(child_counts)), 4.0, 0.0, 2.0),
    ]
    for symbol in symbols:
        values.append(_clip(count(future, symbol), 30.0, 0.0, 3.0))
    for symbol in symbols:
        values.append(_clip(count(near, symbol), 12.0, 0.0, 3.0))
    values.extend(
        [
            _hash_fraction((state.get("map_state") or {}).get("act") if isinstance(state.get("map_state"), dict) else None),
            _bool(isinstance(state.get("map_state"), dict) and bool(state["map_state"].get("first_room_chosen"))),
        ]
    )
    return (values + [0.0] * 28)[:28]


def _map_bag(state: dict[str, Any]) -> list[float]:
    vector = [0.0] * MAP_BUCKETS
    nodes = _map_nodes(state)
    if not nodes:
        return vector
    current_y = _current_map_y(state)
    node_lookup = {
        (int(node.get("x") or 0), int(node.get("y") or 0)): str(node.get("symbol") or "")
        for node in nodes
    }
    weight_sq = 0.0
    for node in nodes:
        y = int(node.get("y") or 0)
        if y <= current_y:
            continue
        x = int(node.get("x") or 0)
        symbol = str(node.get("symbol") or "")
        distance = max(1, y - current_y)
        weight = 1.0 / math.sqrt(float(distance))
        _add_token(vector, f"row{y}:sym:{symbol}", prefix="map", weight=weight)
        _add_token(vector, f"dist{min(distance, 6)}:sym:{symbol}", prefix="map", weight=weight)
        if bool(node.get("emerald")):
            _add_token(vector, f"dist{min(distance, 6)}:green_elite", prefix="map", weight=weight)
        for child in _safe_list(node.get("children")):
            if not isinstance(child, dict):
                continue
            child_symbol = node_lookup.get((int(child.get("x") or 0), int(child.get("y") or 0)), "BOSS")
            _add_token(vector, f"{symbol}->{child_symbol}", prefix="map_edge", weight=weight)
        _add_token(vector, f"x{x}:sym:{symbol}", prefix="map", weight=0.25 * weight)
        weight_sq += weight * weight + (0.25 * weight) * (0.25 * weight)
    if weight_sq > 0.0:
        inv = 1.0 / math.sqrt(weight_sq)
        vector = [value * inv for value in vector]
    return vector


def _reachable_map_nodes(state: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = _map_nodes(state)
    if not nodes:
        return []
    node_lookup = {
        (int(node.get("x") or 0), int(node.get("y") or 0)): node
        for node in nodes
    }
    starts: list[tuple[int, int]] = []
    map_state = state.get("map_state") if isinstance(state.get("map_state"), dict) else {}
    current = map_state.get("current_node") if isinstance(map_state.get("current_node"), dict) else None
    if current is None:
        screen_current = _screen_state(state).get("current_node")
        current = screen_current if isinstance(screen_current, dict) else None
    current_xy = None
    if isinstance(current, dict) and current.get("x") is not None and current.get("y") is not None:
        current_xy = (int(current.get("x") or 0), int(current.get("y") or 0))
    if current_xy in node_lookup:
        for child in _safe_list(node_lookup[current_xy].get("children")):
            if isinstance(child, dict):
                starts.append((int(child.get("x") or 0), int(child.get("y") or 0)))
    if not starts:
        for action in _safe_list(state.get("choice_list")) or _safe_list(state.get("legal_actions")):
            if not isinstance(action, dict) or str(action.get("kind") or "") != "map":
                continue
            if action.get("x") is None:
                continue
            starts.append((int(action.get("x") or 0), int(action.get("floor") or 1) - 1))
    if not starts and current_xy is not None and current_xy[1] < 0:
        starts = [(int(node.get("x") or 0), int(node.get("y") or 0)) for node in nodes if int(node.get("y") or 0) == 0]

    seen: set[tuple[int, int]] = set()
    queue = [xy for xy in starts if xy in node_lookup]
    reachable: list[dict[str, Any]] = []
    while queue:
        xy = queue.pop(0)
        if xy in seen or xy not in node_lookup:
            continue
        seen.add(xy)
        node = node_lookup[xy]
        reachable.append(node)
        for child in _safe_list(node.get("children")):
            if isinstance(child, dict):
                child_xy = (int(child.get("x") or 0), int(child.get("y") or 0))
                if child_xy not in seen:
                    queue.append(child_xy)
    return reachable


def _reachable_map_explicit_features(state: dict[str, Any], *, hp: int, gold: int, purge_cost: int) -> list[float]:
    nodes = _reachable_map_nodes(state)
    if not nodes:
        return [0.0] * 8
    current_y = _current_map_y(state)
    future = [node for node in nodes if int(node.get("y") or 0) > current_y]
    by_symbol: dict[str, list[int]] = {}
    for node in future:
        symbol = str(node.get("symbol") or "")
        by_symbol.setdefault(symbol, []).append(max(1, int(node.get("y") or 0) - current_y))

    def min_dist(symbol: str, fallback: int = 16) -> int:
        values = by_symbol.get(symbol) or []
        return min(values) if values else fallback

    first_rest = min_dist("R")
    first_shop = min_dist("$")
    first_elite = min_dist("E")
    first_boss = min_dist("BOSS", fallback=17)
    before_rest = [node for node in future if max(1, int(node.get("y") or 0) - current_y) < first_rest]
    before_boss = [node for node in future if max(1, int(node.get("y") or 0) - current_y) < first_boss]

    def count(items: list[dict[str, Any]], symbol: str) -> int:
        return sum(1 for node in items if str(node.get("symbol") or "") == symbol)

    forced_combat_before_rest = count(before_rest, "M") + count(before_rest, "E")
    elites_before_rest = count(before_rest, "E")
    shops_before_boss = count(before_boss, "$")
    rests_before_boss = count(before_boss, "R")
    monsters_before_boss = count(before_boss, "M") + count(before_boss, "E")
    return [
        _clip(first_elite if first_elite < 16 else 0, 10.0, 0.0, 2.0),
        _clip(first_rest if first_rest < 16 else 0, 10.0, 0.0, 2.0),
        _clip(first_shop if first_shop < 16 else 0, 10.0, 0.0, 2.0),
        _clip(elites_before_rest, 4.0, 0.0, 2.0),
        _clip(forced_combat_before_rest, 8.0, 0.0, 3.0),
        _clip(shops_before_boss + rests_before_boss, 8.0, 0.0, 2.0),
        _clip(gold - purge_cost, 250.0, -2.0, 3.0),
        _clip(hp / max(1, forced_combat_before_rest), 80.0, 0.0, 3.0) if forced_combat_before_rest > 0 else _clip(hp, 80.0, 0.0, 3.0),
    ]


def _potion_tactical_features(potions: list[dict[str, Any]]) -> list[float]:
    burst_damage = {
        "firepotion",
        "explosivepotion",
        "attackpotion",
        "distilledchaos",
        "duplicationpotion",
        "liquidmemories",
        "entropicbrew",
    }
    defense = {
        "blockpotion",
        "dexteritypotion",
        "speedpotion",
        "essenceofsteel",
        "liquidbronze",
        "ancientpotion",
        "regenpotion",
    }
    scaling = {"strengthpotion", "flexpotion", "cultistpotion", "heartofiron"}
    emergency = {"fairyinabottle", "fruitjuice", "bloodpotion", "smokebomb"}
    useful = 0
    damage = 0
    block = 0
    scale = 0
    panic = 0
    for potion in potions:
        key = _potion_key(potion)
        if not key or key == "potionslot":
            continue
        useful += 1
        damage += 1 if key in burst_damage else 0
        block += 1 if key in defense else 0
        scale += 1 if key in scaling else 0
        panic += 1 if key in emergency else 0
    return [
        _clip(useful, 5.0, 0.0, 2.0),
        _clip(damage, 3.0, 0.0, 2.0),
        _clip(block, 3.0, 0.0, 2.0),
        _clip(scale + panic, 3.0, 0.0, 2.0),
    ]


def _explicit_run_features(state: dict[str, Any], *, deck: list[dict[str, Any]], potions: list[dict[str, Any]], hp: int, gold: int, purge_cost: int) -> list[float]:
    values: list[float] = []
    values.extend(_reachable_map_explicit_features(state, hp=hp, gold=gold, purge_cost=purge_cost))
    values.extend(_potion_tactical_features(potions))
    values.extend(_deck_explicit_features(deck))
    return (values + [0.0] * EXPLICIT_RUN_FEATURE_DIM)[:EXPLICIT_RUN_FEATURE_DIM]


def _reachable_map_bag(state: dict[str, Any]) -> list[float]:
    vector = [0.0] * REACHABLE_MAP_BUCKETS
    nodes = _reachable_map_nodes(state)
    if not nodes:
        return vector
    current_y = _current_map_y(state)
    weight_sq = 0.0
    for node in nodes:
        y = int(node.get("y") or 0)
        x = int(node.get("x") or 0)
        symbol = str(node.get("symbol") or "")
        distance = max(1, y - current_y)
        weight = 1.0 / math.sqrt(float(distance))
        _add_token(vector, f"dist{min(distance, 6)}:sym:{symbol}", prefix="reachable_map", weight=weight)
        _add_token(vector, f"row{y}:sym:{symbol}", prefix="reachable_map", weight=0.75 * weight)
        _add_token(vector, f"x{x}:sym:{symbol}", prefix="reachable_map", weight=0.25 * weight)
        if bool(node.get("emerald")):
            _add_token(vector, f"dist{min(distance, 6)}:green_elite", prefix="reachable_map", weight=weight)
        weight_sq += weight * weight + (0.75 * weight) * (0.75 * weight) + (0.25 * weight) * (0.25 * weight)
    if weight_sq > 0.0:
        inv = 1.0 / math.sqrt(weight_sq)
        vector = [value * inv for value in vector]
    return vector


def _power_bag(player: dict[str, Any], monsters: list[dict[str, Any]]) -> list[float]:
    vector = [0.0] * POWER_BUCKETS
    weight_sq = 0.0

    def add(power: Any, *, owner: str) -> None:
        nonlocal weight_sq
        if not isinstance(power, dict):
            return
        key = _power_key(power)
        if not key:
            return
        amount = float(power.get("amount") or 0.0)
        # Preserve sign and rough magnitude without allowing high stacks to dominate the vector.
        weight = 1.0 + min(4.0, abs(amount)) / 4.0
        if amount < 0.0:
            weight = -weight
        _add_token(vector, key, prefix=f"{owner}_power", weight=weight)
        _add_token(vector, f"{key}:amt:{int(max(-9, min(9, amount)))}", prefix=f"{owner}_power", weight=0.5)
        weight_sq += weight * weight + 0.25

    for power in _safe_list(player.get("powers")):
        add(power, owner="player")
    for monster in monsters:
        if not isinstance(monster, dict) or bool(monster.get("is_gone", False)):
            continue
        for power in _safe_list(monster.get("powers")):
            add(power, owner="monster")
    if weight_sq > 0.0:
        inv = 1.0 / math.sqrt(weight_sq)
        vector = [value * inv for value in vector]
    return vector


def _card_zone_aug_bag(
    *,
    deck: list[dict[str, Any]],
    hand: list[dict[str, Any]],
    draw_pile: list[dict[str, Any]],
    discard_pile: list[dict[str, Any]],
    exhaust_pile: list[dict[str, Any]],
) -> list[float]:
    items: list[tuple[str, Any]] = []
    for zone, cards in (
        ("deck", deck),
        ("hand", hand),
        ("draw", draw_pile),
        ("discard", discard_pile),
        ("exhaust", exhaust_pile),
    ):
        for index, card in enumerate(cards):
            key = _card_key(card)
            if not key:
                continue
            items.append((zone, key))
            if zone in {"draw", "discard", "exhaust"}:
                items.append((f"{zone}_pos{min(index, 12)}", key))
            if int(card.get("upgrades") or 0) > 0:
                items.append((f"{zone}_upgraded", key))
    return _prefixed_ordered_pool_bag(items, AUG_CARD_ZONE_BUCKETS)


def _choice_aug_bag(state: dict[str, Any], screen: dict[str, Any]) -> list[float]:
    vector = [0.0] * AUG_CHOICE_BUCKETS
    weight_sq = 0.0
    actions = [action for action in (_safe_list(state.get("choice_list")) or _safe_list(state.get("legal_actions"))) if isinstance(action, dict)]
    for index, action in enumerate(actions):
        slot_weight = 1.0 / math.sqrt(float(index + 1))
        for token in _choice_tokens_from_action(action):
            _add_token(vector, token, prefix="choice_aug", weight=slot_weight)
            _add_token(vector, f"slot{min(index, 12)}:{token}", prefix="choice_aug", weight=0.35 * slot_weight)
            weight_sq += slot_weight * slot_weight + (0.35 * slot_weight) * (0.35 * slot_weight)
    for prefix, items in (
        ("screen_card_aug", _screen_cards(screen)),
        ("screen_relic_aug", _screen_relics(screen)),
        ("screen_potion_aug", _screen_potions(screen)),
        ("screen_reward_aug", _safe_list(screen.get("rewards"))),
        ("screen_option_aug", _safe_list(screen.get("options"))),
    ):
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            item_weight = 1.0 / math.sqrt(float(index + 1))
            _add_token(vector, item, prefix=prefix, weight=item_weight)
            _add_token(vector, f"slot{min(index, 12)}:{_token_key(item)}", prefix=prefix, weight=0.35 * item_weight)
            weight_sq += item_weight * item_weight + (0.35 * item_weight) * (0.35 * item_weight)
    if weight_sq > 0.0:
        inv = 1.0 / math.sqrt(weight_sq)
        vector = [value * inv for value in vector]
    return vector


def _map_aug_bag(state: dict[str, Any]) -> list[float]:
    vector = [0.0] * AUG_MAP_BUCKETS
    nodes = _map_nodes(state)
    if not nodes:
        return vector
    current_y = _current_map_y(state)
    reachable_xy = {
        (int(node.get("x") or 0), int(node.get("y") or 0))
        for node in _reachable_map_nodes(state)
    }
    node_lookup = {
        (int(node.get("x") or 0), int(node.get("y") or 0)): str(node.get("symbol") or "")
        for node in nodes
    }
    weight_sq = 0.0
    for node in nodes:
        y = int(node.get("y") or 0)
        if y <= current_y:
            continue
        x = int(node.get("x") or 0)
        xy = (x, y)
        symbol = str(node.get("symbol") or "")
        distance = max(1, y - current_y)
        weight = 1.0 / math.sqrt(float(distance))
        if xy in reachable_xy:
            weight *= 1.35
        for token, token_weight in (
            (f"dist{min(distance, 8)}:sym:{symbol}", weight),
            (f"row{y}:sym:{symbol}", 0.75 * weight),
            (f"x{x}:sym:{symbol}", 0.35 * weight),
            (f"reachable:{xy in reachable_xy}:sym:{symbol}", 0.50 * weight),
        ):
            _add_token(vector, token, prefix="map_aug", weight=token_weight)
            weight_sq += token_weight * token_weight
        if bool(node.get("emerald")):
            _add_token(vector, f"dist{min(distance, 8)}:green_elite", prefix="map_aug", weight=weight)
            weight_sq += weight * weight
        for child in _safe_list(node.get("children")):
            if not isinstance(child, dict):
                continue
            child_symbol = node_lookup.get((int(child.get("x") or 0), int(child.get("y") or 0)), "BOSS")
            edge_weight = 0.75 * weight
            _add_token(vector, f"{symbol}->{child_symbol}", prefix="map_edge_aug", weight=edge_weight)
            weight_sq += edge_weight * edge_weight
    if weight_sq > 0.0:
        inv = 1.0 / math.sqrt(weight_sq)
        vector = [value * inv for value in vector]
    return vector


def _power_aug_bag(player: dict[str, Any], monsters: list[dict[str, Any]]) -> list[float]:
    vector = [0.0] * AUG_POWER_BUCKETS
    weight_sq = 0.0

    def add(power: Any, *, owner: str) -> None:
        nonlocal weight_sq
        if not isinstance(power, dict):
            return
        key = _power_key(power)
        if not key:
            return
        amount = float(power.get("amount") or 0.0)
        amount_bucket = int(max(-20, min(20, amount)))
        weight = 1.0 + min(5.0, abs(amount)) / 5.0
        if amount < 0.0:
            weight = -weight
        _add_token(vector, key, prefix=f"{owner}_power_aug", weight=weight)
        _add_token(vector, f"{key}:amt:{amount_bucket}", prefix=f"{owner}_power_aug", weight=0.75)
        weight_sq += weight * weight + 0.75 * 0.75

    for power in _safe_list(player.get("powers")):
        add(power, owner="player")
    for monster in monsters:
        if not isinstance(monster, dict) or bool(monster.get("is_gone", False)):
            continue
        for power in _safe_list(monster.get("powers")):
            add(power, owner="monster")
    if weight_sq > 0.0:
        inv = 1.0 / math.sqrt(weight_sq)
        vector = [value * inv for value in vector]
    return vector


def _augmented_run_features(
    state: dict[str, Any],
    *,
    deck: list[dict[str, Any]],
    relics: list[dict[str, Any]],
    potions: list[dict[str, Any]],
    monsters: list[dict[str, Any]],
    hand: list[dict[str, Any]],
    draw_pile: list[dict[str, Any]],
    discard_pile: list[dict[str, Any]],
    exhaust_pile: list[dict[str, Any]],
    screen: dict[str, Any],
    card_pool_items: list[tuple[str, Any]],
    relic_pool_items: list[tuple[str, Any]],
    boss_relic_pool_items: list[tuple[str, Any]],
    player: dict[str, Any],
) -> list[float]:
    values: list[float] = []
    values.extend(
        _card_zone_aug_bag(
            deck=deck,
            hand=hand,
            draw_pile=draw_pile,
            discard_pile=discard_pile,
            exhaust_pile=exhaust_pile,
        )
    )
    values.extend(_prefixed_ordered_pool_bag(card_pool_items, AUG_FUTURE_CARD_POOL_BUCKETS))
    values.extend(_prefixed_ordered_pool_bag(relic_pool_items + boss_relic_pool_items, AUG_FUTURE_RELIC_POOL_BUCKETS))
    values.extend(_choice_aug_bag(state, screen))
    values.extend(_map_aug_bag(state))
    values.extend(_power_aug_bag(player, monsters))
    return (values + [0.0] * AUG_FEATURE_DIM)[:AUG_FEATURE_DIM]


def encode_run_state(state: dict[str, Any], *, feature_variant: str = "current") -> list[float]:
    state = state if isinstance(state, dict) else {}
    feature_variant = str(feature_variant or "current")
    use_augmented = feature_variant.endswith("_aug")
    feature_base = feature_variant[:-4] if use_augmented else feature_variant
    use_explicit = feature_base.endswith("_explicit")
    base_feature_variant = feature_base[:-9] if use_explicit else feature_base
    use_seed_hash = base_feature_variant not in {"no_seed_structrng", "no_seed_no_rng"}
    use_rng_hash = base_feature_variant not in {"no_seed_structrng", "no_seed_no_rng"}
    use_structured_rng = base_feature_variant != "no_seed_no_rng"
    player = _player_from_state(state)
    deck = _deck_cards(state)
    relics = _relics(state)
    potions = _potions(state)
    monsters = _monsters_from_state(state)
    hand = _cards_from_zone(state, "hand")
    draw_pile = _cards_from_zone(state, "draw_pile")
    discard_pile = _cards_from_zone(state, "discard_pile")
    exhaust_pile = _cards_from_zone(state, "exhaust_pile")

    phase = str(state.get("phase") or state.get("screen") or "")
    room_type = str(state.get("room_type") or "")
    act = int(state.get("act") or 0)
    floor = int(state.get("floor") or 0)
    hp = int(state.get("current_hp") or player.get("current_hp") or 0)
    max_hp = int(state.get("max_hp") or player.get("max_hp") or 0)
    gold = int(state.get("gold") or 0)
    block = int(player.get("block") or 0)
    energy = int(player.get("energy") or 0)
    incoming = 0
    alive_monsters = 0
    monster_hp = 0
    monster_block = 0
    for monster in monsters:
        if bool(monster.get("is_gone", False)):
            continue
        alive_monsters += 1
        monster_hp += max(0, int(monster.get("current_hp") or 0))
        monster_block += max(0, int(monster.get("block") or 0))
        if "ATTACK" in str(monster.get("intent") or ""):
            incoming += max(0, int(monster.get("move_adjusted_damage") or 0)) * max(1, int(monster.get("move_hits") or 1))

    screen = _screen_state(state)
    screen_cards = _screen_cards(screen)
    screen_relics = _screen_relics(screen)
    screen_potions = _screen_potions(screen)
    shop_cards = screen_cards if phase == "SHOP" else []
    shop_relics = screen_relics if phase == "SHOP" else []
    shop_potions = screen_potions if phase == "SHOP" else []

    card_pool_keys = (
        "common_card_pool",
        "uncommon_card_pool",
        "rare_card_pool",
        "colorless_card_pool",
        "curse_card_pool",
        "src_common_card_pool",
        "src_uncommon_card_pool",
        "src_rare_card_pool",
        "src_colorless_card_pool",
        "src_curse_card_pool",
    )
    relic_pool_keys = ("common_relic_pool", "uncommon_relic_pool", "rare_relic_pool", "shop_relic_pool")
    boss_relic_pool_keys = ("boss_relic_pool",)
    card_pool_items = _pool_items(state, card_pool_keys)
    relic_pool_items = _pool_items(state, relic_pool_keys)
    boss_relic_pool_items = _pool_items(state, boss_relic_pool_keys)

    numeric = [
        _clip(act, 4.0, 0.0, 2.0),
        _clip(floor, 60.0, 0.0, 2.0),
        _clip(floor % 17, 17.0, 0.0, 1.0),
        _clip(hp, 120.0, -1.0, 2.0),
        _clip(max_hp, 150.0, 0.0, 2.0),
        _clip(hp / max(1, max_hp), 1.0, -1.0, 1.0),
        _clip(gold, 500.0, 0.0, 3.0),
        _clip(block, 120.0, 0.0, 3.0),
        _clip(energy, 10.0, 0.0, 2.0),
        _clip(incoming, 120.0, 0.0, 3.0),
        _clip(alive_monsters, 6.0, 0.0, 2.0),
        _clip(monster_hp, 500.0, 0.0, 3.0),
        _clip(monster_block, 120.0, 0.0, 3.0),
        _clip(len(deck), 60.0, 0.0, 3.0),
        _clip(len(relics), 40.0, 0.0, 3.0),
        _clip(sum(1 for potion in potions if _potion_key(potion) and _potion_key(potion) != "potionslot"), 5.0, 0.0, 2.0),
        _clip(len(hand), 12.0, 0.0, 2.0),
        _clip(len(draw_pile), 60.0, 0.0, 3.0),
        _clip(len(discard_pile), 60.0, 0.0, 3.0),
        _clip(len(exhaust_pile), 30.0, 0.0, 3.0),
        _clip(len(shop_cards), 8.0, 0.0, 2.0),
        _clip(len(shop_relics), 5.0, 0.0, 2.0),
        _clip(len(shop_potions), 5.0, 0.0, 2.0),
        _clip(int(screen.get("purge_cost") or 0), 250.0, 0.0, 2.0),
        _bool(screen.get("purge_available")),
        _bool(state.get("has_ruby_key")),
        _bool(state.get("has_emerald_key")),
        _bool(state.get("has_sapphire_key")),
        _hash_fraction(state.get("act_boss")),
        _hash_fraction(state.get("event_id")),
        _hash_fraction(state.get("seed")) if use_seed_hash else 0.0,
        _hash_fraction(state.get("dungeon_id")) if use_seed_hash else 0.0,
        _clip(len(card_pool_items), 250.0, 0.0, 3.0),
        _clip(len(relic_pool_items), 120.0, 0.0, 3.0),
        _clip(len(boss_relic_pool_items), 40.0, 0.0, 3.0),
        _clip(len(_safe_list(state.get("common_card_pool"))), 60.0, 0.0, 2.0),
        _clip(len(_safe_list(state.get("uncommon_card_pool"))), 80.0, 0.0, 2.0),
        _clip(len(_safe_list(state.get("rare_card_pool"))), 40.0, 0.0, 2.0),
        _clip(len(_safe_list(state.get("common_relic_pool"))), 60.0, 0.0, 2.0),
        _clip(len(_safe_list(state.get("rare_relic_pool"))), 60.0, 0.0, 2.0),
    ]
    numeric.extend(_one_hot(PHASE_ORDER.index(phase) if phase in PHASE_ORDER else -1, len(PHASE_ORDER)))
    numeric.extend(_one_hot(ROOM_ORDER.index(room_type) if room_type in ROOM_ORDER else -1, len(ROOM_ORDER)))
    numeric.extend(_deck_readiness(deck))
    numeric.extend(_path_features(state))
    numeric.extend(_choice_numeric(state, screen))
    if use_explicit:
        numeric.extend(
            _explicit_run_features(
                state,
                deck=deck,
                potions=potions,
                hp=hp,
                gold=gold,
                purge_cost=int(screen.get("purge_cost") or 0),
            )
        )
    if use_structured_rng:
        numeric.extend(_rng_summary_features(state, include_hash=use_rng_hash))
        numeric.extend(_rng_state_features(state, include_hash=use_rng_hash))
    else:
        numeric.extend([0.0] * 12)
        numeric.extend([0.0] * 36)
    numeric.extend(_map_state_features(state))
    numeric = (numeric + [0.0] * STATE_NUMERIC_DIM)[:STATE_NUMERIC_DIM]

    features = (
        numeric
        + _bag(deck + hand + draw_pile + discard_pile + exhaust_pile, CARD_BUCKETS, _card_key)
        + _bag(relics + [item for item in shop_relics if isinstance(item, dict)], RELIC_BUCKETS, _relic_key)
        + _bag(potions + [item for item in shop_potions if isinstance(item, dict)], POTION_BUCKETS, _potion_key)
        + _bag(monsters, MONSTER_BUCKETS, _monster_key)
        + _choice_bag(state, screen)
        + _prefixed_ordered_pool_bag(card_pool_items, FUTURE_CARD_POOL_BUCKETS)
        + _prefixed_ordered_pool_bag(relic_pool_items, FUTURE_RELIC_POOL_BUCKETS)
        + _prefixed_ordered_pool_bag(boss_relic_pool_items, FUTURE_BOSS_RELIC_POOL_BUCKETS)
        + _token_bag(
            [
                state.get("act_boss"),
                state.get("event_id"),
                state.get("dungeon_id"),
                state.get("screen"),
                state.get("room_phase"),
                state.get("implementation_status"),
            ],
            FUTURE_MISC_BUCKETS,
            prefix="misc",
        )
        + _map_bag(state)
        + _reachable_map_bag(state)
        + _bag(hand, HAND_CARD_BUCKETS, _card_key)
        + _bag(draw_pile, DRAW_CARD_BUCKETS, _card_key)
        + _bag(discard_pile, DISCARD_CARD_BUCKETS, _card_key)
        + _bag(exhaust_pile, EXHAUST_CARD_BUCKETS, _card_key)
        + _power_bag(player, monsters)
    )
    if use_augmented:
        features += _augmented_run_features(
            state,
            deck=deck,
            relics=relics,
            potions=potions,
            monsters=monsters,
            hand=hand,
            draw_pile=draw_pile,
            discard_pile=discard_pile,
            exhaust_pile=exhaust_pile,
            screen=screen,
            card_pool_items=card_pool_items,
            relic_pool_items=relic_pool_items,
            boss_relic_pool_items=boss_relic_pool_items,
            player=player,
        )
    else:
        features += [0.0] * AUG_FEATURE_DIM
    return features


def encode_action(action: dict[str, Any]) -> list[float]:
    action = action if isinstance(action, dict) else {}
    kind = str(action.get("kind") or "")
    item_kind = str(action.get("item_kind") or "")
    numeric = [
        _clip(int(action.get("choice_index") or 0), 20.0, -1.0, 3.0),
        _clip(int(action.get("target_index") or action.get("card_index") or 0), 80.0, -1.0, 3.0),
        _clip(float(action.get("price") or 0), 300.0, 0.0, 3.0),
        _clip(float(action.get("amount") or 0), 300.0, -3.0, 3.0),
        _clip(float(action.get("energy_cost") or 0), 5.0, -1.0, 3.0),
        _hash_fraction(action.get("name")),
        _hash_fraction(action.get("card_id") or action.get("relic_id") or action.get("potion_id") or action.get("item_id")),
        _bool(item_kind == "card"),
        _bool(item_kind == "relic"),
        _bool(item_kind == "potion"),
        _bool(item_kind == "purge"),
        _bool(item_kind == "leave"),
    ]
    numeric.extend(_one_hot(ACTION_KIND_ORDER.index(kind) if kind in ACTION_KIND_ORDER else -1, len(ACTION_KIND_ORDER)))
    numeric = (numeric + [0.0] * ACTION_NUMERIC_DIM)[:ACTION_NUMERIC_DIM]

    tokens = [
        kind,
        item_kind,
        action.get("name"),
        action.get("label"),
        action.get("card_id"),
        action.get("relic_id"),
        action.get("potion_id"),
        action.get("item_id"),
        action.get("mode"),
        action.get("symbol"),
    ]
    card = action.get("card")
    if isinstance(card, dict):
        tokens.extend([card.get("card_id"), card.get("name"), card.get("type"), card.get("rarity")])
    hashed = [0.0] * ACTION_BUCKETS
    for token in tokens:
        key = normalize_token(token)
        if key:
            hashed[stable_bucket(key, ACTION_BUCKETS)] += 1.0
    return numeric + hashed


def encode_action_candidate(
    before_state: dict[str, Any],
    action: dict[str, Any],
    after_state: dict[str, Any],
    *,
    feature_variant: str = "current",
) -> list[float]:
    before = encode_run_state(before_state, feature_variant=feature_variant)
    after = encode_run_state(after_state, feature_variant=feature_variant)
    delta = [a - b for a, b in zip(after, before)]
    return before + encode_action(action) + after + delta


def open_jsonl(path: str | Path, mode: str = "rt"):
    target = Path(path)
    if target.suffix == ".gz":
        return gzip.open(target, mode, encoding=None if "b" in mode else "utf-8")
    return target.open(mode, encoding=None if "b" in mode else "utf-8")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open_jsonl(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_jsonl(target, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def clone_env(env: Any) -> Any:
    return pickle.loads(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL))


def branch_after_state(env: Any, action: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        branch = clone_env(env)
        branch.step(dict(action))
        return branch.state(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


class RunValueNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = STATE_FEATURE_DIM,
        hidden_dim: int = 384,
        depth: int = 3,
        dropout: float = 0.05,
        survival_bins: int = 0,
        final_floor_bins: int = 0,
        architecture: str = "mlp",
        group_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.survival_bins = max(0, int(survival_bins))
        self.final_floor_bins = max(0, int(final_floor_bins))
        self.architecture = str(architecture or "mlp")
        self.group_dim = int(group_dim)
        self.output_dim = len(RUN_VALUE_OUTPUTS) + self.survival_bins + self.final_floor_bins
        self.group_slices: list[tuple[int, int]] = []
        if self.architecture == "res_mlp":
            self.group_encoders = nn.ModuleList()
            self.input = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(int(hidden_dim)),
                        nn.Linear(int(hidden_dim), int(hidden_dim) * 2),
                        nn.GELU(),
                        nn.Dropout(float(dropout)),
                        nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
                        nn.Dropout(float(dropout)),
                    )
                    for _ in range(max(1, int(depth)))
                ]
            )
            self.head = nn.Sequential(nn.LayerNorm(int(hidden_dim)), nn.Linear(int(hidden_dim), self.output_dim))
            self.network = nn.Identity()
        elif self.architecture == "late_fusion" and int(input_dim) in {int(BASE_STATE_FEATURE_DIM), int(STATE_FEATURE_DIM)}:
            sizes = [
                STATE_NUMERIC_DIM,
                CARD_BUCKETS,
                RELIC_BUCKETS,
                POTION_BUCKETS,
                MONSTER_BUCKETS,
                CHOICE_BUCKETS,
                FUTURE_CARD_POOL_BUCKETS,
                FUTURE_RELIC_POOL_BUCKETS,
                FUTURE_BOSS_RELIC_POOL_BUCKETS,
                FUTURE_MISC_BUCKETS,
                MAP_BUCKETS,
                REACHABLE_MAP_BUCKETS,
                HAND_CARD_BUCKETS,
                DRAW_CARD_BUCKETS,
                DISCARD_CARD_BUCKETS,
                EXHAUST_CARD_BUCKETS,
                POWER_BUCKETS,
            ]
            if int(input_dim) == int(STATE_FEATURE_DIM):
                sizes.extend(
                    [
                        AUG_CARD_ZONE_BUCKETS,
                        AUG_FUTURE_CARD_POOL_BUCKETS,
                        AUG_FUTURE_RELIC_POOL_BUCKETS,
                        AUG_CHOICE_BUCKETS,
                        AUG_MAP_BUCKETS,
                        AUG_POWER_BUCKETS,
                    ]
                )
            start = 0
            encoders: list[Any] = []
            for size in sizes:
                end = start + int(size)
                self.group_slices.append((start, end))
                encoders.append(
                    nn.Sequential(
                        nn.LayerNorm(int(size)),
                        nn.Linear(int(size), int(group_dim)),
                        nn.GELU(),
                        nn.Dropout(float(dropout)),
                    )
                )
                start = end
            self.group_encoders = nn.ModuleList(encoders)
            trunk_layers: list[Any] = [nn.LayerNorm(int(group_dim) * len(sizes))]
            current = int(group_dim) * len(sizes)
            for _ in range(max(1, int(depth))):
                trunk_layers.extend([nn.Linear(current, int(hidden_dim)), nn.GELU(), nn.Dropout(float(dropout))])
                current = int(hidden_dim)
            trunk_layers.append(nn.Linear(current, self.output_dim))
            self.network = nn.Sequential(*trunk_layers)
        else:
            self.architecture = "mlp"
            self.group_encoders = nn.ModuleList()
            layers: list[Any] = [nn.LayerNorm(input_dim)]
            current = int(input_dim)
            for _ in range(max(1, int(depth))):
                layers.extend([nn.Linear(current, int(hidden_dim)), nn.GELU(), nn.Dropout(float(dropout))])
                current = int(hidden_dim)
            layers.append(nn.Linear(current, self.output_dim))
            self.network = nn.Sequential(*layers)

    def forward(self, features: Any) -> Any:
        if self.architecture == "res_mlp":
            hidden = self.input(features)
            for block in self.blocks:
                hidden = hidden + block(hidden)
            return self.head(hidden)
        if self.architecture == "late_fusion" and self.group_slices:
            encoded = [
                encoder(features[:, start:end])
                for encoder, (start, end) in zip(self.group_encoders, self.group_slices)
            ]
            return self.network(torch.cat(encoded, dim=1))
        return self.network(features)

    def config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "survival_bins": self.survival_bins,
            "final_floor_bins": self.final_floor_bins,
            "architecture": self.architecture,
            "group_dim": self.group_dim,
        }


class RunActionPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int = ACTION_CANDIDATE_FEATURE_DIM, hidden_dim: int = 512, depth: int = 3, dropout: float = 0.05) -> None:
        super().__init__()
        layers: list[Any] = [nn.LayerNorm(input_dim)]
        current = int(input_dim)
        for _ in range(max(1, int(depth))):
            layers.extend([nn.Linear(current, int(hidden_dim)), nn.GELU(), nn.Dropout(float(dropout))])
            current = int(hidden_dim)
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)

    def forward(self, features: Any) -> Any:
        return self.network(features).squeeze(-1)

    def config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "dropout": self.dropout,
        }


def save_run_value_checkpoint(path: str | Path, model: RunValueNetwork, *, metadata: dict[str, Any] | None = None) -> None:
    require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": RUN_VALUE_CHECKPOINT_VERSION,
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "metadata": dict(metadata or {}),
            "feature_schema": {
                "state_feature_dim": STATE_FEATURE_DIM,
                "outputs": list(RUN_VALUE_OUTPUTS),
            },
        },
        target,
    )


def load_run_value_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[RunValueNetwork, dict[str, Any]]:
    require_torch()
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != RUN_VALUE_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported run value checkpoint: {checkpoint.get('checkpoint_version')}")
    config = dict(checkpoint.get("model_config") or {})
    if "survival_bins" not in config:
        final_weight = checkpoint.get("model_state_dict", {}).get("network.%d.weight" % (max(1, int(config.get("depth", 3))) * 3 + 1))
        if final_weight is not None:
            output_dim = int(final_weight.shape[0])
            config["survival_bins"] = max(0, output_dim - len(RUN_VALUE_OUTPUTS))
    config.setdefault("final_floor_bins", 0)
    model = RunValueNetwork(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metadata = checkpoint.get("metadata") if isinstance(checkpoint.get("metadata"), dict) else {}
    manifest = metadata.get("manifest") if isinstance(metadata.get("manifest"), dict) else {}
    model.feature_variant = str(manifest.get("feature_variant") or metadata.get("feature_variant") or "current")
    model.final_floor_readout = str(metadata.get("final_floor_readout") or "expected")
    model.residual_floor_baseline = metadata.get("residual_floor_baseline")
    model.value_calibration = metadata.get("value_calibration")
    model.use_survival_for_value = bool(metadata.get("use_survival_for_mae") or False)
    model.eval()
    return model, checkpoint


def save_run_action_policy_checkpoint(path: str | Path, model: RunActionPolicyNetwork, *, metadata: dict[str, Any] | None = None) -> None:
    require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": RUN_ACTION_POLICY_CHECKPOINT_VERSION,
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "metadata": dict(metadata or {}),
            "feature_schema": {
                "state_feature_dim": STATE_FEATURE_DIM,
                "action_feature_dim": ACTION_FEATURE_DIM,
                "candidate_feature_dim": ACTION_CANDIDATE_FEATURE_DIM,
            },
        },
        target,
    )


def load_run_action_policy_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[RunActionPolicyNetwork, dict[str, Any]]:
    require_torch()
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != RUN_ACTION_POLICY_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported run action policy checkpoint: {checkpoint.get('checkpoint_version')}")
    model = RunActionPolicyNetwork(**dict(checkpoint.get("model_config") or {})).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metadata = checkpoint.get("metadata") if isinstance(checkpoint.get("metadata"), dict) else {}
    manifest = metadata.get("manifest") if isinstance(metadata.get("manifest"), dict) else {}
    model.feature_variant = str(manifest.get("feature_variant") or metadata.get("feature_variant") or "current")
    model.eval()
    return model, checkpoint


def value_outputs_to_dict(outputs: Any) -> dict[str, float]:
    values = outputs.detach().float().cpu().tolist()
    return {name: float(values[index]) for index, name in enumerate(RUN_VALUE_OUTPUTS)}


def _run_value_baseline_key(
    floor: int,
    *,
    phase: str = "",
    source: str = "",
    sample_kind: str = "",
    mode: str = "floor",
) -> str:
    floor_key = str(int(floor))
    mode = str(mode or "floor")
    if mode == "floor":
        return floor_key
    if mode == "floor_phase":
        return f"{floor_key}|{phase}"
    if mode == "floor_phase_source":
        return f"{floor_key}|{phase}|{source}"
    if mode == "floor_phase_source_kind":
        return f"{floor_key}|{phase}|{source}|{sample_kind}"
    return floor_key


def _run_value_baseline_for_state(state: dict[str, Any], baseline: dict[str, Any]) -> float:
    floor = int(state.get("floor") or 0)
    key_mode = str(baseline.get("key_mode") or "floor")
    phase = str(state.get("phase") or state.get("screen") or "")
    source = str(state.get("source") or "")
    sample_kind = str(state.get("sample_kind") or "")
    group_key = _run_value_baseline_key(
        floor,
        phase=phase,
        source=source,
        sample_kind=sample_kind,
        mode=key_mode,
    )
    group_means = baseline.get("group_means") or {}
    floor_means = baseline.get("floor_means") or {}
    return float(group_means.get(group_key, floor_means.get(str(floor), baseline.get("global_mean_remaining") or 0.0)))


def _run_value_calibration_value(state: dict[str, Any], field: str) -> str:
    field = str(field)
    if field == "floor":
        return str(int(state.get("floor") or 0))
    if field == "floor_bucket":
        floor = int(state.get("floor") or 0)
        start = floor // 5 * 5
        return f"{start:02d}-{start + 4:02d}"
    if field == "phase":
        return str(state.get("phase") or state.get("screen") or "")
    if field == "source":
        return str(state.get("source") or "")
    if field == "sample_kind":
        return str(state.get("sample_kind") or "")
    return ""


def _run_value_calibration_bias_for_state(state: dict[str, Any], calibration: dict[str, Any]) -> float:
    fields = [str(field) for field in calibration.get("fields") or []]
    parent_fields = [str(field) for field in calibration.get("parent_fields") or []]
    if not fields:
        return 0.0
    group_key = "|".join(_run_value_calibration_value(state, field) for field in fields)
    parent_key = "|".join(_run_value_calibration_value(state, field) for field in parent_fields)
    group_bias = calibration.get("group_bias") or {}
    parent_bias = calibration.get("parent_bias") or {}
    return float(group_bias.get(group_key, parent_bias.get(parent_key, 0.0)))


def _fit_feature_dim(values: list[float], dim: int) -> list[float]:
    dim = int(dim)
    if dim <= 0:
        return values
    if len(values) > dim:
        return values[:dim]
    if len(values) < dim:
        return values + [0.0] * (dim - len(values))
    return values


def _clamp_remaining_floor(remaining: float, floor: int) -> float:
    return max(0.0, min(float(remaining), max(0.0, 50.0 - float(floor))))


def predict_remaining_floor(model: RunValueNetwork, state: dict[str, Any], *, device: str = "cpu") -> float:
    require_torch()
    feature_variant = str(getattr(model, "feature_variant", "current") or "current")
    raw_features = encode_run_state(state, feature_variant=feature_variant)
    raw_features = _fit_feature_dim(raw_features, int(getattr(model, "input_dim", len(raw_features)) or len(raw_features)))
    features = torch.tensor([raw_features], dtype=torch.float32, device=device)
    with torch.inference_mode():
        outputs = model(features)
    survival_bins = int(getattr(model, "survival_bins", 0) or 0)
    final_floor_bins = int(getattr(model, "final_floor_bins", 0) or 0)
    final_start = len(RUN_VALUE_OUTPUTS) + survival_bins
    readout = str(getattr(model, "final_floor_readout", "none") or "none")
    if readout != "none" and final_floor_bins > 0 and outputs.shape[1] >= final_start + final_floor_bins:
        logits = outputs[0, final_start : final_start + final_floor_bins].float()
        if readout == "mode":
            final_floor = torch.argmax(logits).float() + 1.0
        else:
            probs = torch.softmax(logits, dim=0)
            if readout == "median":
                final_floor = torch.argmax((torch.cumsum(probs, dim=0) >= 0.5).float()).float() + 1.0
            else:
                floors = torch.arange(1, final_floor_bins + 1, dtype=torch.float32, device=logits.device)
                final_floor = (probs * floors).sum()
        floor = int(state.get("floor") or 0)
        return _clamp_remaining_floor(float((final_floor - floor).detach().cpu().item()), floor)
    if survival_bins > 0 and outputs.shape[1] >= len(RUN_VALUE_OUTPUTS) + survival_bins:
        use_survival = bool(getattr(model, "use_survival_for_value", False))
        if use_survival:
            survival_logits = outputs[0, len(RUN_VALUE_OUTPUTS) : len(RUN_VALUE_OUTPUTS) + survival_bins].float()
            expected_final_floor = torch.sigmoid(survival_logits).sum()
            floor = int(state.get("floor") or 0)
            return _clamp_remaining_floor(float((expected_final_floor - floor).detach().cpu().item()), floor)
    remaining = float(outputs[0, 0].detach().cpu().item())
    residual_floor_baseline = getattr(model, "residual_floor_baseline", None)
    if isinstance(residual_floor_baseline, dict):
        remaining += _run_value_baseline_for_state(state, residual_floor_baseline)
    value_calibration = getattr(model, "value_calibration", None)
    if isinstance(value_calibration, dict):
        remaining += _run_value_calibration_bias_for_state(state, value_calibration)
    return _clamp_remaining_floor(remaining, int(state.get("floor") or 0))
