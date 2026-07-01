from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


FEATURE_SCHEMA_VERSION = "v3_combat_candidate_features_v4_potions_room"

PLAYER_POWER_IDS = [
    "Artifact",
    "Barricade",
    "Berserk",
    "Brutality",
    "Combust",
    "Confusion",
    "Constricted",
    "Corruption",
    "Dark Embrace",
    "Demon Form",
    "Dexterity",
    "Double Tap",
    "Draw Reduction",
    "Entangled",
    "Evolve",
    "Feel No Pain",
    "Fire Breathing",
    "Flame Barrier",
    "Flex",
    "Frail",
    "Hex",
    "IntangiblePlayer",
    "Juggernaut",
    "Magnetism",
    "Mayhem",
    "Metallicize",
    "No Draw",
    "NoBlockPower",
    "Panache",
    "Rage",
    "Rupture",
    "Sadistic",
    "Strength",
    "Surrounded",
    "TheBomb",
    "Vulnerable",
    "Weakened",
]

CARD_TYPES = ["ATTACK", "SKILL", "POWER", "STATUS", "CURSE"]
CARD_RARITIES = ["BASIC", "COMMON", "UNCOMMON", "RARE", "SPECIAL", "CURSE"]
INTENTS = [
    "ATTACK",
    "ATTACK_BUFF",
    "ATTACK_DEBUFF",
    "ATTACK_DEFEND",
    "BUFF",
    "DEBUFF",
    "STRONG_DEBUFF",
    "DEFEND",
    "DEFEND_BUFF",
    "DEFEND_DEBUFF",
    "ESCAPE",
    "MAGIC",
    "SLEEP",
    "STUN",
    "UNKNOWN",
]
ATTACK_INTENTS = {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}
NEGATIVE_POWER_IDS = {"Weakened", "Weak", "Vulnerable", "Frail", "No Draw", "Draw Reduction", "Entangled", "Confusion"}
ZONE_NAMES = ["hand", "draw_pile", "discard_pile", "exhaust_pile", "deck"]
COMBAT_ROOM_TYPES = ["MonsterRoom", "MonsterRoomElite", "MonsterRoomBoss"]


def _clip_norm(value: Any, scale: float, lo: float = -10.0, hi: float = 10.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if scale:
        numeric /= float(scale)
    return max(lo, min(hi, numeric))


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _card_id(card: dict[str, Any] | None) -> str:
    if not card:
        return ""
    return str(card.get("card_id") or card.get("id") or card.get("name") or "")


def _card_name(card: dict[str, Any] | None) -> str:
    if not card:
        return ""
    return str(card.get("name") or card.get("card_id") or card.get("id") or "")


@lru_cache(maxsize=1)
def card_identity_ids() -> tuple[str, ...]:
    try:
        from spirecomm.native_sim_v3.content.cards import card_catalog

        ids = sorted(
            str(card_id)
            for card_id, card in card_catalog().items()
            if card.color in {"RED", "COLORLESS", "CURSE"} or card.type == "STATUS"
        )
    except Exception:
        ids = []
    return ("__UNK__", *ids)


@lru_cache(maxsize=1)
def card_identity_index() -> dict[str, int]:
    return {card_id: index for index, card_id in enumerate(card_identity_ids())}


@lru_cache(maxsize=1)
def potion_identity_ids() -> tuple[str, ...]:
    try:
        from spirecomm.native_sim_v3.content.potions import ironclad_potion_pool

        ids = sorted(str(potion_id) for potion_id in ironclad_potion_pool())
    except Exception:
        ids = []
    return ("__UNK__", *ids)


@lru_cache(maxsize=1)
def potion_identity_index() -> dict[str, int]:
    return {potion_id: index for index, potion_id in enumerate(potion_identity_ids())}


def _card_identity_features(card: dict[str, Any] | None) -> list[float]:
    ids = card_identity_ids()
    features = [0.0] * len(ids)
    if not card:
        return features
    card_id = _card_id(card)
    index = card_identity_index().get(card_id, 0)
    features[index] = 1.0
    return features


def _potion_identity_features(potion: dict[str, Any] | None) -> list[float]:
    ids = potion_identity_ids()
    features = [0.0] * len(ids)
    if not potion:
        return features
    potion_id = str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "")
    index = potion_identity_index().get(potion_id, 0)
    features[index] = 1.0
    return features


def _card_cost(card: dict[str, Any] | None) -> int:
    if not card:
        return 0
    cost = card.get("cost_for_turn")
    if cost is None:
        cost = card.get("cost")
    try:
        return int(cost)
    except (TypeError, ValueError):
        return 0


def _powers_by_id(powers: list[dict[str, Any]] | None) -> dict[str, float]:
    totals: dict[str, float] = {}
    for power in list(powers or []):
        power_id = str(power.get("power_id") or power.get("id") or power.get("name") or "")
        if power_id == "Weak":
            power_id = "Weakened"
        if power_id == "Entangle":
            power_id = "Entangled"
        if power_id == "DemonForm":
            power_id = "Demon Form"
        if power_id == "FlameBarrier":
            power_id = "Flame Barrier"
        if power_id == "Intangible":
            power_id = "IntangiblePlayer"
        if power_id == "NoBlock":
            power_id = "NoBlockPower"
        if power_id == "SadisticNature":
            power_id = "Sadistic"
        if power_id == "The Bomb":
            power_id = "TheBomb"
        try:
            amount = float(power.get("amount") or 0)
        except (TypeError, ValueError):
            amount = 0.0
        totals[power_id] = totals.get(power_id, 0.0) + amount
    return totals


def _is_alive(monster: dict[str, Any]) -> bool:
    if bool(monster.get("is_gone", False)):
        return False
    try:
        return int(monster.get("current_hp") or 0) > 0 or bool(monster.get("half_dead", False))
    except (TypeError, ValueError):
        return False


def combat_state(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state.get("combat_state") or {})


def hand_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    return list((combat_state(state).get("hand") or []))


def monsters(state: dict[str, Any]) -> list[dict[str, Any]]:
    return list((combat_state(state).get("monsters") or []))


def incoming_damage(state: dict[str, Any]) -> int:
    return _incoming_damage_from_monsters(monsters(state))


def _incoming_damage_from_monsters(all_monsters: list[dict[str, Any]]) -> int:
    total = 0
    for monster in all_monsters:
        if not _is_alive(monster):
            continue
        if str(monster.get("intent") or "") not in ATTACK_INTENTS:
            continue
        try:
            damage = int(monster.get("move_adjusted_damage") or 0)
            hits = int(monster.get("move_hits") or 1)
        except (TypeError, ValueError):
            continue
        total += max(0, damage) * max(1, hits)
    return total


def playable_count(state: dict[str, Any], *, attack_only: bool = False) -> int:
    playable, playable_attacks = _playable_counts_from_hand(hand_cards(state))
    return playable_attacks if attack_only else playable


def _playable_counts_from_hand(hand: list[dict[str, Any]]) -> tuple[int, int]:
    count = 0
    attack_count = 0
    for card in hand:
        if not bool(card.get("is_playable", False)):
            continue
        count += 1
        if str(card.get("type") or "") == "ATTACK":
            attack_count += 1
    return count, attack_count


def _zone_cards(state: dict[str, Any], zone_name: str) -> list[dict[str, Any]]:
    if zone_name == "deck":
        return list(state.get("deck") or [])
    return list(combat_state(state).get(zone_name) or [])


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def _zone_summary(cards: list[dict[str, Any]]) -> list[float]:
    total = len(cards)
    if not cards:
        return [0.0] * 16
    attack_count = 0
    skill_count = 0
    power_count = 0
    status_count = 0
    curse_count = 0
    rare_count = 0
    clipped_cost_sum = 0
    upgraded = 0
    exhaust = 0
    playable = 0
    ethereal = 0
    retain = 0
    damage_sum = 0
    block_sum = 0
    magic_sum = 0
    for card in cards:
        card_type = str(card.get("type") or "")
        if card_type == "ATTACK":
            attack_count += 1
        elif card_type == "SKILL":
            skill_count += 1
        elif card_type == "POWER":
            power_count += 1
        elif card_type == "STATUS":
            status_count += 1
        elif card_type == "CURSE":
            curse_count += 1
        if str(card.get("rarity") or "") == "RARE":
            rare_count += 1
        cost = card.get("cost_for_turn")
        if cost is None:
            cost = card.get("cost")
        try:
            cost_int = int(cost)
        except (TypeError, ValueError):
            cost_int = 0
        clipped_cost_sum += max(0, min(4, cost_int))
        upgraded += int(int(card.get("upgrades") or 0) > 0)
        exhaust += int(bool(card.get("exhausts", False)))
        playable += int(bool(card.get("is_playable", False)))
        ethereal += int(bool(card.get("ethereal", False)))
        retain += int(bool(card.get("retain", False)) or bool(card.get("self_retain", False)))
        damage_sum += int(card.get("base_damage") or 0)
        block_sum += int(card.get("base_block") or 0)
        magic_sum += int(card.get("base_magic") or 0)
    inv_total = 1.0 / float(total)
    avg_cost = clipped_cost_sum * inv_total
    return [
        _clip_norm(total, 20.0, 0.0, 5.0),
        attack_count * inv_total,
        skill_count * inv_total,
        power_count * inv_total,
        status_count * inv_total,
        curse_count * inv_total,
        avg_cost / 2.0,
        upgraded * inv_total,
        exhaust * inv_total,
        playable * inv_total,
        ethereal * inv_total,
        retain * inv_total,
        _clip_norm(damage_sum, 93.0, 0.0, 5.0),
        _clip_norm(block_sum, 53.0, 0.0, 5.0),
        _clip_norm(magic_sum, 24.0, -5.0, 5.0),
        rare_count * inv_total,
    ]


def _monster_summary_from_monsters(all_monsters: list[dict[str, Any]], damage_total: int | None = None) -> list[float]:
    live = [monster for monster in all_monsters if _is_alive(monster)]
    if not all_monsters:
        return [0.0] * 20
    live_count = len(live)
    total_hp = sum(max(0, int(monster.get("current_hp") or 0)) for monster in live)
    max_hp_values = [max(1, int(monster.get("max_hp") or 1)) for monster in live]
    hp_ratios = [
        max(0.0, min(1.0, float(monster.get("current_hp") or 0) / float(max(1, int(monster.get("max_hp") or 1)))))
        for monster in live
    ]
    block_total = sum(max(0, int(monster.get("block") or 0)) for monster in live)
    if damage_total is None:
        damage_total = _incoming_damage_from_monsters(all_monsters)
    intent_counts = {intent: 0 for intent in INTENTS}
    positive_power_total = 0.0
    negative_power_total = 0.0
    for monster in live:
        intent = str(monster.get("intent") or "UNKNOWN")
        if intent not in intent_counts:
            intent = "UNKNOWN"
        intent_counts[intent] += 1
        for power_id, amount in _powers_by_id(list(monster.get("powers") or [])).items():
            if amount < 0 or power_id in NEGATIVE_POWER_IDS:
                negative_power_total += abs(amount)
            else:
                positive_power_total += abs(amount)
    return [
        _clip_norm(len(all_monsters), 5.0, 0.0, 2.0),
        _clip_norm(live_count, 5.0, 0.0, 2.0),
        _clip_norm(total_hp, 350.0, 0.0, 5.0),
        min(hp_ratios) if hp_ratios else 0.0,
        max(hp_ratios) if hp_ratios else 0.0,
        sum(hp_ratios) / max(1, len(hp_ratios)),
        _clip_norm(block_total, 30.0, 0.0, 5.0),
        _clip_norm(damage_total, 50.0, 0.0, 5.0),
        _ratio(sum(intent_counts[intent] for intent in ATTACK_INTENTS), max(1, live_count)),
        _ratio(intent_counts["BUFF"], max(1, live_count)),
        _ratio(intent_counts["DEBUFF"] + intent_counts["STRONG_DEBUFF"], max(1, live_count)),
        _ratio(intent_counts["DEFEND"] + intent_counts["DEFEND_BUFF"] + intent_counts["DEFEND_DEBUFF"], max(1, live_count)),
        _ratio(intent_counts["UNKNOWN"], max(1, live_count)),
        _clip_norm(sum(max_hp_values), 350.0, 0.0, 5.0),
        _clip_norm(positive_power_total, 10.0, 0.0, 5.0),
        _clip_norm(negative_power_total, 5.0, 0.0, 5.0),
        _clip_norm(max((int(monster.get("move_hits") or 1) for monster in live), default=0), 5.0, 0.0, 3.0),
        _clip_norm(max((int(monster.get("move_adjusted_damage") or 0) for monster in live), default=0), 40.0, -2.0, 3.0),
        _ratio(sum(1 for monster in all_monsters if bool(monster.get("is_gone", False))), max(1, len(all_monsters))),
        _ratio(sum(1 for monster in all_monsters if bool(monster.get("half_dead", False))), max(1, len(all_monsters))),
    ]


def _monster_summary(state: dict[str, Any]) -> list[float]:
    all_monsters = monsters(state)
    return _monster_summary_from_monsters(all_monsters, _incoming_damage_from_monsters(all_monsters))


def encode_state_summary(state: dict[str, Any]) -> list[float]:
    combat = combat_state(state)
    player = dict(combat.get("player") or {})
    powers = _powers_by_id(list(player.get("powers") or []))
    hand = list(combat.get("hand") or [])
    draw_pile = list(combat.get("draw_pile") or [])
    discard_pile = list(combat.get("discard_pile") or [])
    exhaust_pile = list(combat.get("exhaust_pile") or [])
    deck = list(state.get("deck") or [])
    all_monsters = monsters(state)
    incoming = _incoming_damage_from_monsters(all_monsters)
    playable, playable_attacks = _playable_counts_from_hand(hand)
    current_hp = int(state.get("current_hp") or player.get("current_hp") or 0)
    max_hp = max(1, int(state.get("max_hp") or player.get("max_hp") or 1))
    player_block = int(player.get("block") or 0)
    energy = int(player.get("energy") or 0)
    base = [
        _clip_norm(state.get("act", 1), 4.0, 0.0, 2.0),
        _clip_norm(state.get("floor", 0), 60.0, 0.0, 2.0),
        _clip_norm(combat.get("turn", 0), 10.0, 0.0, 3.0),
        *[_bool(str(state.get("room_type") or "") == room_type) for room_type in COMBAT_ROOM_TYPES],
        _clip_norm(current_hp, 82.0, 0.0, 2.0),
        _clip_norm(max_hp, 90.0, 0.0, 2.0),
        max(0.0, min(1.5, float(current_hp) / float(max_hp))),
        _clip_norm(player_block, 30.0, 0.0, 5.0),
        _clip_norm(energy, 6.0, 0.0, 3.0),
        _clip_norm(state.get("gold", 0), 500.0, 0.0, 5.0),
        _clip_norm(len(hand), 10.0, 0.0, 2.0),
        _clip_norm(len(draw_pile), 20.0, 0.0, 3.0),
        _clip_norm(len(discard_pile), 20.0, 0.0, 3.0),
        _clip_norm(len(exhaust_pile), 10.0, 0.0, 3.0),
        _clip_norm(len(deck), 30.0, 0.0, 3.0),
        _clip_norm(playable, 10.0, 0.0, 2.0),
        _clip_norm(playable_attacks, 8.0, 0.0, 2.0),
        _clip_norm(incoming, 50.0, 0.0, 5.0),
        _clip_norm(len(state.get("relics") or []), 15.0, 0.0, 3.0),
        _clip_norm(
            sum(
                1
                for potion in list(state.get("potions") or [])
                if str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "") != "Potion Slot"
            ),
            5.0,
            0.0,
            2.0,
        ),
    ]
    power_features = [_clip_norm(powers.get(power_id, 0.0), 7.0, -3.0, 3.0) for power_id in PLAYER_POWER_IDS]
    positive_total = sum(abs(amount) for power_id, amount in powers.items() if amount > 0 and power_id not in NEGATIVE_POWER_IDS)
    negative_total = sum(abs(amount) for power_id, amount in powers.items() if amount < 0 or power_id in NEGATIVE_POWER_IDS)
    totals = [_clip_norm(positive_total, 10.0, 0.0, 5.0), _clip_norm(negative_total, 5.0, 0.0, 5.0)]
    zone_features: list[float] = []
    for cards in (hand, draw_pile, discard_pile, exhaust_pile, deck):
        zone_features.extend(_zone_summary(cards))
    return base + power_features + totals + zone_features + _monster_summary_from_monsters(all_monsters, incoming)


def _selected_card(before_state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    action_card = action.get("card")
    if isinstance(action_card, dict):
        return action_card
    index = action.get("card_index")
    if index is None:
        index = action.get("source_index")
    try:
        card_index = int(index)
    except (TypeError, ValueError):
        return None
    hand = hand_cards(before_state)
    if 0 <= card_index < len(hand):
        return hand[card_index]
    return None


def _selected_potion(before_state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    action_potion = action.get("potion")
    if isinstance(action_potion, dict):
        return action_potion
    index = action.get("potion_index")
    try:
        potion_index = int(index)
    except (TypeError, ValueError):
        return None
    potions = list(before_state.get("potions") or [])
    if 0 <= potion_index < len(potions) and isinstance(potions[potion_index], dict):
        return potions[potion_index]
    return None


def _selected_target(before_state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    try:
        target_index = int(action.get("target_index"))
    except (TypeError, ValueError):
        return None
    mons = monsters(before_state)
    if 0 <= target_index < len(mons):
        return mons[target_index]
    return None


def encode_action_summary_base(before_state: dict[str, Any], action: dict[str, Any]) -> list[float]:
    kind = str(action.get("kind") or "")
    card = _selected_card(before_state, action)
    potion = _selected_potion(before_state, action)
    target = _selected_target(before_state, action)
    card_type = str((card or action).get("type") or "")
    rarity = str((card or action).get("rarity") or "")
    type_flags = [_bool(card_type == name) for name in CARD_TYPES]
    rarity_flags = [_bool(rarity == name) for name in CARD_RARITIES]
    target_intent = str((target or {}).get("intent") or "")
    target_max_hp = max(1, int((target or {}).get("max_hp") or 1))
    target_hp = int((target or {}).get("current_hp") or 0)
    target_block = int((target or {}).get("block") or 0)
    return [
        _bool(kind == "end"),
        _bool(kind == "card"),
        _bool(kind == "potion"),
        _bool(kind == "card_select"),
        _bool(kind == "card_reward"),
        _bool(action.get("requires_target", False)),
        _clip_norm(_card_cost(card), 4.0, -1.0, 3.0),
        _clip_norm((card or {}).get("base_damage", 0), 50.0, -1.0, 3.0),
        _clip_norm((card or {}).get("base_block", 0), 40.0, 0.0, 3.0),
        _clip_norm((card or {}).get("base_magic", 0), 6.0, -3.0, 3.0),
        _bool(int((card or {}).get("upgrades") or 0) > 0),
        _bool((card or {}).get("exhausts", False)),
        _bool((card or {}).get("ethereal", False)),
        _bool((card or {}).get("free_to_play_once", False)),
        _bool(_card_cost(card) < 0),
        _bool((card or {}).get("target") in {"ALL_ENEMY", "ALL"}),
        *_type_or_zero(type_flags, len(CARD_TYPES)),
        *_type_or_zero(rarity_flags, len(CARD_RARITIES)),
        max(0.0, min(1.5, float(target_hp) / float(target_max_hp))),
        _clip_norm(target_block, 30.0, 0.0, 5.0),
        _bool(target_intent in ATTACK_INTENTS),
        _clip_norm((target or {}).get("move_adjusted_damage", 0), 40.0, -1.0, 3.0),
        _clip_norm((target or {}).get("move_hits", 0), 5.0, 0.0, 3.0),
    ]


def encode_action_summary(before_state: dict[str, Any], action: dict[str, Any]) -> list[float]:
    card = _selected_card(before_state, action)
    potion = _selected_potion(before_state, action)
    return encode_action_summary_base(before_state, action) + _card_identity_features(card) + _potion_identity_features(potion)


def _type_or_zero(values: list[float], size: int) -> list[float]:
    if len(values) == size:
        return values
    return [0.0] * size


def _power_amount(state: dict[str, Any], power_id: str) -> float:
    player = dict(combat_state(state).get("player") or {})
    return _powers_by_id(list(player.get("powers") or [])).get(power_id, 0.0)


def _state_current_hp(state: dict[str, Any], player: dict[str, Any] | None = None) -> int:
    if state.get("current_hp") is not None:
        return int(state.get("current_hp") or 0)
    return int((player or {}).get("current_hp") or 0)


def encode_delta(before_state: dict[str, Any], action: dict[str, Any], after_state: dict[str, Any]) -> list[float]:
    before_combat = combat_state(before_state)
    after_combat = combat_state(after_state)
    before_player = dict(before_combat.get("player") or {})
    after_player = dict(after_combat.get("player") or {})
    before_mons = monsters(before_state)
    after_mons = monsters(after_state)
    try:
        target_index = int(action.get("target_index"))
    except (TypeError, ValueError):
        target_index = -1
    before_target = before_mons[target_index] if 0 <= target_index < len(before_mons) else {}
    after_target = after_mons[target_index] if 0 <= target_index < len(after_mons) else {}
    before_live = sum(1 for monster in before_mons if _is_alive(monster))
    after_live = sum(1 for monster in after_mons if _is_alive(monster))
    before_monster_hp = sum(max(0, int(monster.get("current_hp") or 0)) for monster in before_mons if _is_alive(monster))
    after_monster_hp = sum(max(0, int(monster.get("current_hp") or 0)) for monster in after_mons if _is_alive(monster))
    before_target_hp = int(before_target.get("current_hp") or 0)
    after_target_hp = int(after_target.get("current_hp") or 0)
    target_damage = max(0, before_target_hp - after_target_hp)
    total_damage = max(0, before_monster_hp - after_monster_hp)
    before_turn = int(before_combat.get("turn") or 0)
    after_turn = int(after_combat.get("turn") or before_turn)
    phase_after = str(after_state.get("phase") or "")
    return [
        _clip_norm(_state_current_hp(after_state, after_player) - _state_current_hp(before_state, before_player), 18.0, -5.0, 5.0),
        _clip_norm(int(after_player.get("block") or 0) - int(before_player.get("block") or 0), 40.0, -5.0, 5.0),
        _clip_norm(int(after_player.get("energy") or 0) - int(before_player.get("energy") or 0), 6.0, -5.0, 5.0),
        _clip_norm(len(after_combat.get("hand") or []) - len(before_combat.get("hand") or []), 10.0, -3.0, 3.0),
        _clip_norm(len(after_combat.get("draw_pile") or []) - len(before_combat.get("draw_pile") or []), 15.0, -3.0, 3.0),
        _clip_norm(len(after_combat.get("discard_pile") or []) - len(before_combat.get("discard_pile") or []), 15.0, -3.0, 3.0),
        _clip_norm(len(after_combat.get("exhaust_pile") or []) - len(before_combat.get("exhaust_pile") or []), 10.0, -3.0, 3.0),
        _clip_norm(after_live - before_live, 5.0, -2.0, 2.0),
        _clip_norm(after_monster_hp - before_monster_hp, 80.0, -5.0, 5.0),
        _clip_norm(after_target_hp - before_target_hp, 50.0, -5.0, 5.0),
        _clip_norm(int(after_target.get("block") or 0) - int(before_target.get("block") or 0), 30.0, -5.0, 5.0),
        _bool(before_target and not _is_alive(after_target)),
        _clip_norm(total_damage, 80.0, 0.0, 5.0),
        _clip_norm(target_damage, 59.0, 0.0, 5.0),
        _clip_norm(max(0, int(after_player.get("block") or 0) - int(before_player.get("block") or 0)), 40.0, 0.0, 5.0),
        _clip_norm(incoming_damage(after_state) - incoming_damage(before_state), 30.0, -5.0, 5.0),
        _clip_norm(playable_count(after_state), 10.0, 0.0, 2.0),
        _clip_norm(playable_count(after_state, attack_only=True), 10.0, 0.0, 2.0),
        _clip_norm(_power_amount(after_state, "Strength") - _power_amount(before_state, "Strength"), 5.0, -5.0, 5.0),
        _clip_norm(_power_amount(after_state, "Dexterity") - _power_amount(before_state, "Dexterity"), 5.0, -5.0, 5.0),
        _bool(phase_after in {"CARD_SELECT", "CARD_REWARD"}),
        _clip_norm(len(after_state.get("choice_list") or []), 10.0, 0.0, 2.0),
        _bool(after_turn > before_turn),
        _bool(str(after_state.get("phase") or "") in {"COMPLETE", "VICTORY"}),
        _bool(str(after_state.get("phase") or "") == "GAME_OVER"),
        _clip_norm(len(after_combat.get("exhaust_pile") or []) - len(before_combat.get("exhaust_pile") or []), 3.0, -2.0, 3.0),
        _clip_norm(len(after_combat.get("hand") or []) - len(before_combat.get("hand") or []), 5.0, -3.0, 3.0),
    ]


def action_key(action: dict[str, Any], before_state: dict[str, Any] | None = None) -> tuple[Any, ...]:
    kind = str(action.get("kind") or "")
    if kind == "end":
        return ("end",)
    if kind == "card":
        card = _selected_card(before_state or {}, action) if before_state is not None else action.get("card")
        if not isinstance(card, dict):
            card = action
        card_index = action.get("card_index", action.get("source_index"))
        target_index = action.get("target_index")
        return (
            "card",
            int(card_index) if card_index is not None else None,
            str(action.get("card_id") or _card_id(card)),
            int((card or {}).get("upgrades") or action.get("upgrades") or 0),
            int(target_index) if target_index is not None else None,
        )
    if kind == "potion":
        potion = _selected_potion(before_state or {}, action) if before_state is not None else action.get("potion")
        if not isinstance(potion, dict):
            potion = action
        target_index = action.get("target_index")
        return (
            "potion",
            int(action.get("potion_index")) if action.get("potion_index") is not None else None,
            str(action.get("potion_id") or potion.get("potion_id") or potion.get("id") or potion.get("name") or ""),
            int(target_index) if target_index is not None else None,
        )
    return (
        kind,
        int(action.get("choice_index")) if action.get("choice_index") is not None else None,
        str(action.get("card_id") or action.get("name") or ""),
        int(action.get("target_index")) if action.get("target_index") is not None else None,
    )


def root_combat_actions(env: Any, *, legal_actions: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    phase = str(getattr(env, "phase", "COMBAT"))
    if phase != "COMBAT":
        return []
    actions = legal_actions if legal_actions is not None else env.legal_actions()
    return [action for action in actions if action.get("kind") in {"card", "potion", "end"}]


def action_keys_are_unique(actions: list[dict[str, Any]], before_state: dict[str, Any]) -> bool:
    keys = [action_key(action, before_state) for action in actions]
    return len(keys) == len(set(keys))


def clone_env_blob(
    env: Any,
    *,
    strip_debug_history: bool = False,
    teacher_branch_slim: bool = False,
) -> bytes:
    flagged_objects: list[tuple[Any, bool, Any]] = []
    if teacher_branch_slim:
        candidates = [env]
        nested_combat = getattr(env, "combat", None)
        if nested_combat is not None:
            candidates.append(nested_combat)
        for candidate in candidates:
            try:
                flagged_objects.append(
                    (candidate, hasattr(candidate, "_teacher_branch_clone_slim"), getattr(candidate, "_teacher_branch_clone_slim", None))
                )
                setattr(candidate, "_teacher_branch_clone_slim", True)
            except Exception:
                pass
    try:
        if not strip_debug_history:
            return pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
        snapshots = _clear_env_debug_history_with_restore(env)
        try:
            return pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            _restore_env_debug_history(snapshots)
    finally:
        for candidate, had_flag, previous_flag in reversed(flagged_objects):
            try:
                if had_flag:
                    setattr(candidate, "_teacher_branch_clone_slim", previous_flag)
                else:
                    delattr(candidate, "_teacher_branch_clone_slim")
            except Exception:
                pass


def step_branch_from_blob(
    env_blob: bytes,
    action: dict[str, Any],
    *,
    strip_debug_history: bool = False,
    fast_combat_sync: bool = False,
) -> Any:
    branch = pickle.loads(env_blob)
    if fast_combat_sync:
        try:
            setattr(branch, "_teacher_fast_combat_sync", True)
        except Exception:
            pass
    branch.step(dict(action))
    if strip_debug_history:
        clear_env_debug_history(branch)
    return branch


def step_branch(
    env: Any,
    action: dict[str, Any],
    *,
    strip_debug_history: bool = False,
    fast_combat_sync: bool = False,
) -> Any:
    try:
        branch = pickle.loads(clone_env_blob(env, strip_debug_history=strip_debug_history))
    except Exception:
        branch = copy.deepcopy(env)
    if fast_combat_sync:
        try:
            setattr(branch, "_teacher_fast_combat_sync", True)
        except Exception:
            pass
    branch.step(dict(action))
    if strip_debug_history:
        clear_env_debug_history(branch)
    return branch


def clear_env_debug_history(env: Any) -> None:
    """Drop RNG debug-call history while preserving the RNG counters/state."""

    for calls in _env_debug_call_lists(env):
        calls.clear()


def _clear_env_debug_history_with_restore(env: Any) -> list[tuple[list[Any], list[Any]]]:
    snapshots = [(calls, list(calls)) for calls in _env_debug_call_lists(env)]
    for calls, _snapshot in snapshots:
        calls.clear()
    return snapshots


def _restore_env_debug_history(snapshots: list[tuple[list[Any], list[Any]]]) -> None:
    for calls, snapshot in snapshots:
        calls[:] = snapshot


def _env_debug_call_lists(env: Any) -> list[list[Any]]:
    random_sets = [getattr(env, "randoms", None)]
    combat_env = getattr(env, "combat", None)
    if combat_env is not None and combat_env is not env:
        random_sets.append(getattr(combat_env, "randoms", None))
    call_lists: list[list[Any]] = []
    seen: set[int] = set()
    for randoms in random_sets:
        streams = getattr(randoms, "streams", None)
        if not isinstance(streams, dict):
            continue
        for stream in streams.values():
            calls = getattr(stream, "calls", None)
            if isinstance(calls, list) and id(calls) not in seen:
                seen.add(id(calls))
                call_lists.append(calls)
    return call_lists


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    state_dim: int
    action_dim: int
    delta_dim: int

    @property
    def candidate_dim(self) -> int:
        return self.state_dim * 2 + self.action_dim + self.delta_dim


def schema() -> FeatureSchema:
    empty_state = {
        "phase": "COMBAT",
        "act": 1,
        "floor": 0,
        "current_hp": 80,
        "max_hp": 80,
        "gold": 99,
        "deck": [],
        "relics": [],
        "potions": [],
        "combat_state": {
            "turn": 1,
            "player": {"current_hp": 80, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "monsters": [],
            "hand": [],
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
        },
    }
    empty_action = {"kind": "end", "name": "END_TURN"}
    return FeatureSchema(
        version=FEATURE_SCHEMA_VERSION,
        state_dim=len(encode_state_summary(empty_state)),
        action_dim=len(encode_action_summary(empty_state, empty_action)),
        delta_dim=len(encode_delta(empty_state, empty_action, empty_state)),
    )


def encode_candidate(before_state: dict[str, Any], action: dict[str, Any], after_state: dict[str, Any]) -> list[float]:
    return (
        encode_state_summary(before_state)
        + encode_action_summary(before_state, action)
        + encode_state_summary(after_state)
        + encode_delta(before_state, action, after_state)
    )


def encode_candidate_with_before_summary(
    before_state: dict[str, Any],
    before_summary: list[float],
    action: dict[str, Any],
    after_state: dict[str, Any],
) -> list[float]:
    return (
        list(before_summary)
        + encode_action_summary(before_state, action)
        + encode_state_summary(after_state)
        + encode_delta(before_state, action, after_state)
    )


def expand_candidate(env: Any, action: dict[str, Any]) -> tuple[dict[str, Any], list[float], Any]:
    before_state = copy.deepcopy(env.state() if callable(getattr(env, "state", None)) else env.serialize())
    branch = step_branch(env, action)
    after_state = copy.deepcopy(branch.state() if callable(getattr(branch, "state", None)) else branch.serialize())
    return after_state, encode_candidate(before_state, action, after_state), branch
