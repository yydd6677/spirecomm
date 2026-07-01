from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spirecomm.native_sim_v3.content.elite_rules import emerald_elite_rules
from spirecomm.native_sim_v3.core.randoms import NativeRandomSet, java_shuffle_in_place
from spirecomm.native_sim_v3.core.state import CombatState, MonsterState, PlayerState
from spirecomm.native_sim_v3.content.cards import (
    HEALING_CARD_IDS,
    can_upgrade_card,
    card_catalog,
    initialize_source_card_pools,
    make_card,
    upgrade_card,
)
from spirecomm.native_sim_v3.content.potions import roll_random_potion


NEGATIVE_STACKABLE_POWERS = {"Strength", "Dexterity", "Focus"}
NEGATIVE_PERSISTENT_POWERS = NEGATIVE_STACKABLE_POWERS | {"Confusion", "Mode Shift", "No Draw"}
ZERO_AMOUNT_PERSISTENT_POWERS = {"Slow", "Time Warp"}
MONSTER_DEBUFF_POWERS = {
    "Weak",
    "Weakened",
    "Vulnerable",
    "Frail",
    "Entangled",
    "NoBlock",
    "Poison",
} | NEGATIVE_STACKABLE_POWERS
PLAYER_DEBUFF_POWERS = {
    "Weak",
    "Weakened",
    "Vulnerable",
    "Frail",
    "Entangled",
    "No Draw",
    "Draw Reduction",
    "NoBlock",
    "NoBlockPower",
    "Confusion",
    "Hex",
    "Flex",
    "FlexLoss",
} | NEGATIVE_STACKABLE_POWERS
ENERGY_RELIC_IDS = {
    "Busted Crown",
    "Coffee Dripper",
    "Cursed Key",
    "Ectoplasm",
    "Fusion Hammer",
    "Mark of Pain",
    "Philosopher's Stone",
    "Runic Dome",
    "Sozu",
    "Velvet Choker",
}
ATTACK_INTENTS = {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}
POWER_ID_ALIASES = {
    "Weak": "Weakened",
    "NoBlock": "NoBlockPower",
    "Entangle": "Entangled",
    "DemonForm": "Demon Form",
    "Regen": "Regeneration",
    "Regrow": "Life Link",
    "IntangiblePlayer": "Intangible",
}
POWER_PRIORITIES = {
    "Confusion": 0,
    "Pen Nib": 6,
    "Frail": 10,
    "Compulsive": 50,
    "Flight": 50,
    "Intangible": 75,
    "IntangiblePlayer": 75,
    "Weak": 99,
    "Weakened": 99,
    "Constricted": 105,
}
END_TURN_AUTOPLAY_DISCARD_CARD_IDS = {"Burn", "Decay", "Doubt", "Regret", "Shame"}
MAX_BLOCK = 999
TRANSIENT_CARD_FLAGS_FOR_PILE = {
    "_echoed",
    "_hex_already_triggered",
    "_monster_after_use_already_triggered",
    "_exclude_self_from_perfected_strike_count",
    "_strange_spoon_proc",
    "_reset_cost_for_turn_after_play",
}
TARGETED_COMBAT_POTION_IDS = {
    "Explosive Potion",
    "FearPotion",
    "Fire Potion",
    "Weak Potion",
}
COMBAT_POTION_DISCOVERY_TYPES = {
    "AttackPotion": "ATTACK",
    "SkillPotion": "SKILL",
    "PowerPotion": "POWER",
}
SUPPORTED_COMBAT_POTION_IDS = {
    "Ancient Potion",
    "BlessingOfTheForge",
    "Block Potion",
    "BloodPotion",
    "ColorlessPotion",
    "CultistPotion",
    "Dexterity Potion",
    "DistilledChaos",
    "DuplicationPotion",
    "Energy Potion",
    "ElixirPotion",
    "EntropicBrew",
    "EssenceOfSteel",
    "Explosive Potion",
    "FearPotion",
    "Fire Potion",
    "Fruit Juice",
    "GamblersBrew",
    "HeartOfIron",
    "LiquidBronze",
    "LiquidMemories",
    "PowerPotion",
    "Regen Potion",
    "SkillPotion",
    "SneckoOil",
    "SpeedPotion",
    "SteroidPotion",
    "Strength Potion",
    "Swift Potion",
    "Weak Potion",
    *COMBAT_POTION_DISCOVERY_TYPES.keys(),
}
ANY_NUMBER_CARD_SELECT_MODES = {"ELIXIR", "GAMBLING_CHIP"}

_TEACHER_SOURCE_CARD_POOL_REGISTRY: dict[int, dict[str, list[str]]] = {}
_TEACHER_SOURCE_CARD_POOL_REGISTRY_MAX = 1024


def _register_teacher_source_card_pools(source_card_pools: dict[str, list[str]]) -> int:
    if len(_TEACHER_SOURCE_CARD_POOL_REGISTRY) >= _TEACHER_SOURCE_CARD_POOL_REGISTRY_MAX:
        _TEACHER_SOURCE_CARD_POOL_REGISTRY.clear()
    source_key = id(source_card_pools)
    _TEACHER_SOURCE_CARD_POOL_REGISTRY[source_key] = source_card_pools
    return source_key


def _copy_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (list(value) if isinstance(value, list) else dict(value) if isinstance(value, dict) else value)
        for key, value in card.items()
    }


def _make_potion_slot(index: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "potion_id": "Potion Slot",
        "id": "Potion Slot",
        "name": "Potion Slot",
        "requires_target": False,
        "can_use": False,
        "can_discard": False,
    }
    if index is not None:
        payload["slot_index"] = int(index)
    return payload


def _potion_id(potion: dict[str, Any] | None) -> str:
    if not isinstance(potion, dict):
        return ""
    return str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "")


def _is_usable_potion(potion: dict[str, Any] | None) -> bool:
    return _potion_id(potion) not in {"", "Potion Slot"} and bool((potion or {}).get("can_use", True))


def _reset_card_for_pile(card: dict[str, Any]) -> None:
    card["cost_for_turn"] = card.get("cost")
    card["free_to_play_once"] = False
    for key in TRANSIENT_CARD_FLAGS_FOR_PILE:
        card.pop(key, None)


def _clear_play_once_flags_for_pile(card: dict[str, Any]) -> None:
    card["free_to_play_once"] = False
    for key in TRANSIENT_CARD_FLAGS_FOR_PILE:
        card.pop(key, None)


def _monster_draw_x(monster: MonsterState) -> float:
    try:
        return float(monster.meta.get("draw_x", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _get_power_amount(creature: PlayerState | MonsterState, power_id: str) -> int:
    powers = creature.powers
    if not powers:
        return 0
    raw_target = power_id
    canonical_target = POWER_ID_ALIASES.get(raw_target, raw_target)
    total = 0
    power_id = canonical_target
    for power in powers:
        raw_id = power.get("power_id") or power.get("id") or ""
        if raw_id == raw_target or raw_id == power_id or POWER_ID_ALIASES.get(raw_id, raw_id) == power_id:
            total += int(power.get("amount") or 0)
    return total


def _has_power(creature: PlayerState | MonsterState, power_id: str) -> bool:
    powers = creature.powers
    if not powers:
        return False
    raw_target = power_id
    canonical_target = POWER_ID_ALIASES.get(raw_target, raw_target)
    power_id = canonical_target
    for power in powers:
        raw_id = power.get("power_id") or power.get("id") or ""
        if raw_id == raw_target or raw_id == power_id or POWER_ID_ALIASES.get(raw_id, raw_id) == power_id:
            return True
    return False


def _is_weakened(creature: PlayerState | MonsterState) -> bool:
    powers = creature.powers
    if not powers:
        return False
    for power in powers:
        raw_id = power.get("power_id") or power.get("id") or ""
        if raw_id in {"Weak", "Weakened"} and int(power.get("amount") or 0) > 0:
            return True
    return False


def _has_attack_intent(monster: MonsterState) -> bool:
    if str(monster.intent) in ATTACK_INTENTS:
        return True
    if str(monster.next_move or "") in {"", "UNKNOWN"}:
        return False
    try:
        adjusted_damage = int(monster.move_adjusted_damage)
        hits = int(monster.move_hits)
    except (TypeError, ValueError):
        return False
    return adjusted_damage >= 0 and (adjusted_damage > 0 or hits > 0)


def _canonical_power_id(power_id: str) -> str:
    return POWER_ID_ALIASES.get(power_id, power_id)


def _sort_powers(creature: PlayerState | MonsterState) -> None:
    creature.powers.sort(
        key=lambda power: POWER_PRIORITIES.get(
            _canonical_power_id(str(power.get("power_id") or power.get("id") or "")),
            5,
        )
    )


def _set_power_amount(creature: PlayerState | MonsterState, power_id: str, amount: int) -> None:
    power_id = _canonical_power_id(power_id)
    matching = [
        power
        for power in creature.powers
        if _canonical_power_id(str(power.get("power_id") or power.get("id"))) == power_id
    ]
    if matching:
        keep = matching[0]
        for duplicate in matching[1:]:
            creature.powers.remove(duplicate)
        power = keep
        if amount == 0 and power_id in ZERO_AMOUNT_PERSISTENT_POWERS:
            power["power_id"] = power_id
            power["id"] = power_id
            power["name"] = power_id
            power["amount"] = 0
            power["misc"] = 0
        elif amount == 0 or (amount < 0 and power_id not in NEGATIVE_PERSISTENT_POWERS):
            creature.powers.remove(power)
        else:
            power["power_id"] = power_id
            power["id"] = power_id
            power["name"] = power_id
            power["amount"] = int(amount)
            power["misc"] = int(amount)
        return
    for power in creature.powers:
        if _canonical_power_id(str(power.get("power_id") or power.get("id"))) == power_id:
            if amount == 0 and power_id in ZERO_AMOUNT_PERSISTENT_POWERS:
                power["power_id"] = power_id
                power["id"] = power_id
                power["name"] = power_id
                power["amount"] = 0
                power["misc"] = 0
            elif amount == 0 or (amount < 0 and power_id not in NEGATIVE_PERSISTENT_POWERS):
                creature.powers.remove(power)
            else:
                power["power_id"] = power_id
                power["id"] = power_id
                power["name"] = power_id
                power["amount"] = int(amount)
                power["misc"] = int(amount)
            return
    if amount > 0 or (amount < 0 and power_id in NEGATIVE_PERSISTENT_POWERS) or (amount == 0 and power_id in ZERO_AMOUNT_PERSISTENT_POWERS):
        creature.powers.append(
            {
                "power_id": power_id,
                "id": power_id,
                "name": power_id,
                "amount": int(amount),
                "card": None,
                "damage": 0,
                "just_applied": power_id == "Draw Reduction",
                "misc": int(amount),
            }
        )
        _sort_powers(creature)


def _add_power(creature: PlayerState | MonsterState, power_id: str, amount: int) -> None:
    power_id = _canonical_power_id(power_id)
    _set_power_amount(creature, power_id, _get_power_amount(creature, power_id) + int(amount))


def _direct_add_power(creature: PlayerState | MonsterState, power_id: str, amount: int) -> None:
    power_id = _canonical_power_id(power_id)
    for power in creature.powers:
        if _canonical_power_id(str(power.get("power_id") or power.get("id"))) != power_id:
            continue
        power["power_id"] = power_id
        power["id"] = power_id
        power["name"] = power_id
        power["amount"] = int(power.get("amount") or 0) + int(amount)
        power["misc"] = int(power.get("amount") or 0)
        return
    if amount > 0 or (amount < 0 and power_id in NEGATIVE_PERSISTENT_POWERS):
        creature.powers.append(
            {
                "power_id": power_id,
                "id": power_id,
                "name": power_id,
                "amount": int(amount),
                "card": None,
                "damage": 0,
                "just_applied": power_id == "Draw Reduction",
                "misc": int(amount),
            }
        )


def _append_power(creature: PlayerState | MonsterState, power_id: str, amount: int, *, misc: int = 0) -> None:
    power_id = _canonical_power_id(power_id)
    creature.powers.append(
        {
            "power_id": power_id,
            "id": power_id,
            "name": power_id,
            "amount": int(amount),
            "card": None,
            "damage": 0,
            "just_applied": power_id == "Draw Reduction",
            "misc": int(misc),
        }
    )
    _sort_powers(creature)


def _mark_power_just_applied(creature: PlayerState | MonsterState, power_id: str) -> None:
    power_id = _canonical_power_id(power_id)
    for power in creature.powers:
        if _canonical_power_id(str(power.get("power_id") or power.get("id") or "")) == power_id:
            power["just_applied"] = True
            return


def _remove_power(creature: PlayerState | MonsterState, power_id: str) -> None:
    _set_power_amount(creature, power_id, 0)


def _remove_player_debuffs(player: PlayerState) -> None:
    retained: list[dict[str, Any]] = []
    for power in player.powers:
        power_id = str(power.get("power_id") or power.get("id") or "")
        amount = int(power.get("amount") or 0)
        if power_id in NEGATIVE_STACKABLE_POWERS and amount < 0:
            continue
        if power_id in PLAYER_DEBUFF_POWERS and power_id not in NEGATIVE_STACKABLE_POWERS:
            continue
        retained.append(power)
    player.powers = retained


def _remove_monster_debuffs(monster: MonsterState) -> None:
    retained: list[dict[str, Any]] = []
    for power in monster.powers:
        power_id = str(power.get("power_id") or power.get("id") or "")
        amount = int(power.get("amount") or 0)
        if power_id == "Shackled":
            continue
        if power_id in NEGATIVE_STACKABLE_POWERS and amount < 0:
            continue
        if power_id in MONSTER_DEBUFF_POWERS and power_id not in NEGATIVE_STACKABLE_POWERS:
            continue
        retained.append(power)
    monster.powers = retained


def _decrement_turn_powers(creature: PlayerState | MonsterState) -> None:
    if isinstance(creature, MonsterState) and bool(creature.meta.get("escaped", False)):
        return
    for power_id in ("Vulnerable", "Weakened", "Frail", "Intangible", "NoBlockPower", "Draw Reduction"):
        amount = _get_power_amount(creature, power_id)
        if amount <= 0:
            continue
        for power in creature.powers:
            if _canonical_power_id(str(power.get("power_id") or power.get("id") or "")) != power_id:
                continue
            if bool(power.get("just_applied")):
                power["just_applied"] = False
                amount = 0
            break
        if amount <= 0:
            continue
        _set_power_amount(creature, power_id, amount - 1)


def _card_cost(card: dict[str, Any]) -> int:
    cost = card.get("cost_for_turn")
    if cost is None:
        cost = card.get("cost")
    return int(cost if cost is not None else 0)


def _set_cost_for_turn_like_sts(card: dict[str, Any], amount: int) -> bool:
    if _card_cost(card) < 0:
        return False
    card["cost_for_turn"] = max(0, int(amount))
    return True


def _modify_cost_for_combat_like_sts(card: dict[str, Any], amount: int) -> bool:
    cost_for_turn = card.get("cost_for_turn")
    if cost_for_turn is None:
        cost_for_turn = card.get("cost")
    cost = card.get("cost")
    if cost_for_turn is not None and int(cost_for_turn) > 0:
        modified = max(0, int(cost_for_turn) + int(amount))
        card["cost_for_turn"] = modified
        card["cost"] = modified
        card["cost_for_combat"] = modified
        return True
    if cost is not None and int(cost) >= 0:
        modified = max(0, int(cost) + int(amount))
        card["cost"] = modified
        card["cost_for_turn"] = 0
        card["cost_for_combat"] = modified
        return True
    return False


def _apply_blood_for_blood_damage_cost_reduction(card: dict[str, Any]) -> None:
    if str(card.get("card_id") or "") != "Blood for Blood":
        return
    reduction = int(card.get("combat_cost_reduction") or 0) + 1
    _set_blood_for_blood_combat_cost_reduction(card, reduction)


def _set_blood_for_blood_combat_cost_reduction(card: dict[str, Any], reduction: int) -> None:
    if str(card.get("card_id") or "") != "Blood for Blood":
        return
    reduction = max(int(card.get("combat_cost_reduction") or 0), max(0, int(reduction)))
    base_cost = int(card.get("base_cost", card.get("cost", 0)) or 0)
    effective_cost = max(0, base_cost - reduction)
    card["combat_cost_reduction"] = reduction
    card["cost"] = effective_cost
    card["cost_for_combat"] = effective_cost
    current_turn_cost = card.get("cost_for_turn")
    if current_turn_cost is None or int(current_turn_cost) > effective_cost:
        card["cost_for_turn"] = effective_cost


def _damage_multiplier(
    amount: int,
    *,
    vulnerable: bool,
    weak: bool,
    vulnerable_multiplier: float = 1.5,
    weak_multiplier: float = 0.75,
) -> int:
    scaled = float(amount)
    if weak:
        scaled *= weak_multiplier
    if vulnerable:
        scaled *= vulnerable_multiplier
    return max(0, int(scaled))


def _round_positive_half_up(value: float) -> int:
    if value <= 0:
        return 0
    return int(value + 0.5)


def _deal_damage(target: PlayerState | MonsterState, amount: int) -> int:
    blocked = min(int(getattr(target, "block", 0) or 0), amount)
    if blocked:
        target.block -= blocked
    remaining = max(0, amount - blocked)
    if remaining > 0 and _get_power_amount(target, "Intangible") > 0:
        remaining = 1
    hp_before = int(target.current_hp)
    target.current_hp = max(0, int(target.current_hp) - remaining)
    return hp_before - int(target.current_hp)


def _refresh_card_flags(cards: list[dict[str, Any]], player: PlayerState) -> None:
    if not cards:
        return
    has_corruption = _has_power(player, "Corruption")
    entangled = _get_power_amount(player, "Entangled") > 0
    player_energy = int(player.energy)
    for card in cards:
        if card.get("live_is_playable") is not None:
            card["is_playable"] = bool(card.get("live_is_playable"))
            continue
        if str(card.get("card_id") or "") == "Blood for Blood":
            base_cost = int(card.get("base_cost", card.get("cost", 0)) or 0)
            reduction = int(card.get("combat_cost_reduction") or 0)
            effective_cost = max(0, base_cost - reduction)
            card["cost"] = effective_cost
            card["cost_for_combat"] = effective_cost
            current_turn_cost = card.get("cost_for_turn")
            if current_turn_cost is None or int(current_turn_cost) > effective_cost:
                card["cost_for_turn"] = effective_cost
        if has_corruption and str(card.get("type") or "") == "SKILL":
            _modify_cost_for_combat_like_sts(card, -9)
        card_type = str(card.get("type") or "")
        cost = _card_cost(card)
        if card_type == "CURSE" and card.get("card_id") != "Necronomicurse":
            card["is_playable"] = False
        elif entangled and card_type == "ATTACK":
            card["is_playable"] = False
        elif bool(card.get("free_to_play_once")):
            card["is_playable"] = cost >= -1
        elif cost == -1:
            card["is_playable"] = bool(card.get("free_to_play_once") or player_energy > 0)
        else:
            card["is_playable"] = bool(cost >= 0 and cost <= player_energy)


def _card_has_strike_tag(card: dict[str, Any]) -> bool:
    tags = {str(tag) for tag in list(card.get("tags") or [])}
    if "STRIKE" in tags:
        return True
    card_id = str(card.get("card_id") or "")
    return card_id in {"Strike_R", "Strike_G", "Strike_Blue", "Strike_Purple", "Twin Strike", "Wild Strike", "Pommel Strike", "Perfected Strike"}


def _count_strike_cards(state: CombatState) -> int:
    piles = (state.hand, state.draw_pile, state.discard_pile)
    return sum(1 for pile in piles for card in pile if _card_has_strike_tag(card))


def _make_shiv(uuid: str, *, accuracy_amount: int = 0, upgrades: int = 0) -> dict[str, Any]:
    upgrades = max(0, int(upgrades))
    base_damage = (6 if upgrades > 0 else 4) + max(0, int(accuracy_amount))
    return {
        "card_id": "Shiv",
        "name": "Shiv",
        "type": "ATTACK",
        "rarity": "SPECIAL",
        "color": "COLORLESS",
        "target": "ENEMY",
        "cost": 0,
        "base_cost": 0,
        "cost_for_turn": 0,
        "cost_for_combat": 0,
        "free_to_play_once": False,
        "upgrades": upgrades,
        "misc": 0,
        "exhausts": True,
        "has_target": True,
        "is_playable": True,
        "uuid": uuid,
        "base_damage": base_damage,
        "base_block": 0,
        "base_magic": 0,
        "ethereal": False,
        "retain": False,
    }


def _end_of_turn_block_power_amount(creature: PlayerState | MonsterState) -> int:
    metallicize = _get_power_amount(creature, "Metallicize")
    plated = _get_power_amount(creature, "Plated Armor")
    return max(0, metallicize) + max(0, plated)


def _apply_end_of_turn_block_powers(creature: PlayerState | MonsterState) -> None:
    _gain_block(creature, _end_of_turn_block_power_amount(creature))


def _apply_end_of_turn_negative_stat(creature: PlayerState | MonsterState, power_id: str, amount: int) -> None:
    amount = int(amount)
    if amount <= 0:
        return
    artifact = _get_power_amount(creature, "Artifact")
    if artifact > 0:
        _set_power_amount(creature, "Artifact", artifact - 1)
        return
    _add_power(creature, power_id, -amount)


def _apply_end_of_turn_temporary_powers(creature: PlayerState | MonsterState) -> None:
    generic_strength_up = _get_power_amount(creature, "Generic Strength Up Power")
    if generic_strength_up > 0:
        _add_power(creature, "Strength", generic_strength_up)
    shackled = _get_power_amount(creature, "Shackled")
    if shackled > 0:
        _add_power(creature, "Strength", shackled)
        _remove_power(creature, "Shackled")
    flex_loss = _get_power_amount(creature, "FlexLoss") + _get_power_amount(creature, "Flex")
    if flex_loss > 0:
        _apply_end_of_turn_negative_stat(creature, "Strength", flex_loss)
        _remove_power(creature, "FlexLoss")
        _remove_power(creature, "Flex")
    lose_strength = _get_power_amount(creature, "LoseStrength")
    if lose_strength > 0:
        _apply_end_of_turn_negative_stat(creature, "Strength", lose_strength)
        _remove_power(creature, "LoseStrength")
    lose_dexterity = _get_power_amount(creature, "DexLoss") + _get_power_amount(creature, "LoseDexterity")
    if lose_dexterity > 0:
        _apply_end_of_turn_negative_stat(creature, "Dexterity", lose_dexterity)
        _remove_power(creature, "DexLoss")
        _remove_power(creature, "LoseDexterity")
    malleable = _get_power_amount(creature, "Malleable")
    if malleable > 0:
        for power in creature.powers:
            if str(power.get("power_id") or power.get("id") or "") == "Malleable":
                power["amount"] = int(power.get("misc") or malleable)
                break


def _apply_monster_regenerate_power(monster: MonsterState) -> None:
    regen = _get_power_amount(monster, "Regenerate")
    if regen <= 0 or not _alive(monster):
        return
    monster.current_hp = min(int(monster.max_hp), int(monster.current_hp) + regen)


def _clear_block_if_needed(creature: PlayerState | MonsterState) -> None:
    if not _has_power(creature, "Barricade"):
        creature.block = 0


def _gain_block(creature: PlayerState | MonsterState, amount: int) -> None:
    if amount > 0:
        creature.block = min(MAX_BLOCK, int(creature.block) + int(amount))


def _alive(monster: MonsterState) -> bool:
    return int(monster.current_hp) > 0 and not bool(monster.meta.get("escaped", False))


def _counts_as_alive(monster: MonsterState) -> bool:
    return _alive(monster) or bool(monster.meta.get("half_dead", False))


def _any_monsters_alive(monsters: list[MonsterState]) -> bool:
    return any(_counts_as_alive(monster) for monster in monsters)


def _owned_relic_ids(relics: list[dict[str, Any]]) -> set[str]:
    return {str(relic.get("relic_id") or relic.get("id") or "") for relic in relics}


def _player_base_energy(
    player: PlayerState,
    relics: list[dict[str, Any]],
    relic_ids: set[str] | frozenset[str] | None = None,
) -> int:
    if relic_ids is None:
        relic_ids = _owned_relic_ids(relics)
    return int(player.base_energy) + sum(1 for relic_id in relic_ids if relic_id in ENERGY_RELIC_IDS)


def _player_bonus_energy_for_room(
    relics: list[dict[str, Any]],
    room_type: str,
    relic_ids: set[str] | frozenset[str] | None = None,
    *,
    elite_trigger: bool = False,
) -> int:
    if relic_ids is None:
        relic_ids = _owned_relic_ids(relics)
    if "SlaversCollar" in relic_ids and (room_type in {"MonsterRoomElite", "MonsterRoomBoss"} or elite_trigger):
        return 1
    return 0


def _player_turn_draw_count(player: PlayerState, relics: list[dict[str, Any]]) -> int:
    relic_ids = _owned_relic_ids(relics)
    base = 7 if "Snecko Eye" in relic_ids else int(player.draw_per_turn)
    draw_reduction = _get_power_amount(player, "Draw Reduction")
    return max(0, base - draw_reduction)


def _player_opening_draw_count(player: PlayerState, relics: list[dict[str, Any]]) -> int:
    relic_ids = _owned_relic_ids(relics)
    count = _player_turn_draw_count(player, relics)
    if "Bag of Preparation" in relic_ids:
        count += 2
    return count


def _player_opening_energy_bonus(relics: list[dict[str, Any]]) -> int:
    bonus = 0
    relic_ids = _owned_relic_ids(relics)
    if "Lantern" in relic_ids:
        bonus += 1
    for relic in relics:
        relic_id = str(relic.get("relic_id") or relic.get("id") or "")
        if relic_id == "Ancient Tea Set" and int(relic.get("counter", 0) or 0) == -2:
            bonus += 2
            relic["counter"] = -1
    return bonus


def _pop_hand_card(hand: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    if 0 <= int(index) < len(hand):
        return hand.pop(int(index))
    return None


def _spawn_split_medium(monster_id: str, randoms: NativeRandomSet, ascension_level: int, hp: int) -> MonsterState:
    if monster_id == "AcidSlime_L":
        return MonsterState(
            monster_id="AcidSlime_M",
            name="Acid Slime (M)",
            current_hp=hp,
            max_hp=hp,
            intent="ATTACK_DEBUFF",
            next_move="WOUND_TACKLE",
            meta={"wound_damage": 8 if ascension_level >= 2 else 7, "normal_damage": 12 if ascension_level >= 2 else 10, "first_move": True},
        )
    return MonsterState(
        monster_id="SpikeSlime_M",
        name="Spike Slime (M)",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK_DEBUFF",
        next_move="FLAME_TACKLE",
        meta={"tackle_damage": 10 if ascension_level >= 2 else 8, "first_move": True},
    )


def _spawn_cultist(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(50, 56) if ascension_level >= 7 else randoms.stream("monster_hp").random(48, 54))
    return MonsterState(
        monster_id="Cultist",
        name="Cultist",
        current_hp=hp,
        max_hp=hp,
        intent="BUFF",
        next_move="INCANTATION",
        meta={"opening_ai_roll": True},
    )


def _spawn_jaw_worm(randoms: NativeRandomSet, ascension_level: int, *, hard_mode: bool = False) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(42, 46) if ascension_level >= 7 else randoms.stream("monster_hp").random(40, 44))
    if ascension_level >= 17:
        meta = {"bellow_str": 5, "bellow_block": 9, "chomp_damage": 12, "thrash_damage": 7, "thrash_block": 5, "first_move": False}
    elif ascension_level >= 2:
        meta = {"bellow_str": 4, "bellow_block": 6, "chomp_damage": 12, "thrash_damage": 7, "thrash_block": 5, "first_move": False}
    else:
        meta = {"bellow_str": 3, "bellow_block": 6, "chomp_damage": 11, "thrash_damage": 7, "thrash_block": 5, "first_move": False}
    if hard_mode:
        meta["hard_mode"] = True
        meta["opening_move_source"] = "JawWorm"
    else:
        meta["opening_ai_roll"] = True
    return MonsterState(
        monster_id="JawWorm",
        name="Jaw Worm",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="CHOMP",
        meta=meta,
    )


def _spawn_acid_slime_s(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(9, 13) if ascension_level >= 7 else randoms.stream("monster_hp").random(8, 12))
    damage = 4 if ascension_level >= 2 else 3
    return MonsterState(
        monster_id="AcidSlime_S",
        name="Acid Slime (S)",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="TACKLE",
        meta={"tackle_damage": damage, "first_move": True, "opening_move_source": "AcidSlime_S"},
    )


def _spawn_spike_slime_s(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(11, 15) if ascension_level >= 7 else randoms.stream("monster_hp").random(10, 14))
    damage = 6 if ascension_level >= 2 else 5
    return MonsterState(
        monster_id="SpikeSlime_S",
        name="Spike Slime (S)",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="TACKLE",
        meta={"tackle_damage": damage, "opening_ai_roll": True},
    )


def _spawn_acid_slime_m(
    randoms: NativeRandomSet,
    ascension_level: int,
    *,
    hp_override: int | None = None,
    draw_x: float = 0.0,
) -> MonsterState:
    hp = int(
        hp_override
        if hp_override is not None
        else (randoms.stream("monster_hp").random(29, 34) if ascension_level >= 7 else randoms.stream("monster_hp").random(28, 32))
    )
    wound_damage = 8 if ascension_level >= 2 else 7
    normal_damage = 12 if ascension_level >= 2 else 10
    return MonsterState(
        monster_id="AcidSlime_M",
        name="Acid Slime (M)",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK_DEBUFF",
        next_move="WOUND_TACKLE",
        meta={"wound_damage": wound_damage, "normal_damage": normal_damage, "first_move": True, "opening_move_source": "AcidSlime_M", "draw_x": draw_x},
    )


def _spawn_spike_slime_m(
    randoms: NativeRandomSet,
    ascension_level: int,
    *,
    hp_override: int | None = None,
    draw_x: float = 0.0,
) -> MonsterState:
    hp = int(
        hp_override
        if hp_override is not None
        else (randoms.stream("monster_hp").random(29, 34) if ascension_level >= 7 else randoms.stream("monster_hp").random(28, 32))
    )
    damage = 10 if ascension_level >= 2 else 8
    return MonsterState(
        monster_id="SpikeSlime_M",
        name="Spike Slime (M)",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK_DEBUFF",
        next_move="FLAME_TACKLE",
        meta={"tackle_damage": damage, "first_move": True, "opening_move_source": "SpikeSlime_M", "draw_x": draw_x},
    )


def _spawn_louse_normal(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(11, 16) if ascension_level >= 7 else randoms.stream("monster_hp").random(10, 15))
    bite = int(randoms.stream("monster_hp").random(6, 8) if ascension_level >= 2 else randoms.stream("monster_hp").random(5, 7))
    strength_gain = 4 if ascension_level >= 17 else 3
    monster = MonsterState(
        monster_id="FuzzyLouseNormal",
        name="Louse",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="BITE",
        meta={"bite_damage": bite, "strength_gain": strength_gain, "pending_curl_up": True, "opening_move_source": "FuzzyLouseNormal"},
    )
    return monster


def _spawn_louse_defensive(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(12, 18) if ascension_level >= 7 else randoms.stream("monster_hp").random(11, 17))
    bite = int(randoms.stream("monster_hp").random(6, 8) if ascension_level >= 2 else randoms.stream("monster_hp").random(5, 7))
    monster = MonsterState(
        monster_id="FuzzyLouseDefensive",
        name="Louse",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="BITE",
        meta={"bite_damage": bite, "pending_curl_up": True, "opening_move_source": "FuzzyLouseDefensive"},
    )
    return monster


def _spawn_blue_slaver(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(48, 52) if ascension_level >= 7 else randoms.stream("monster_hp").random(46, 50))
    return MonsterState(
        monster_id="SlaverBlue",
        name="Slaver",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="STAB",
        meta={"stab_damage": 13 if ascension_level >= 2 else 12, "rake_damage": 8 if ascension_level >= 2 else 7, "weak_amount": 1, "opening_move_source": "SlaverBlue"},
    )


def _spawn_red_slaver(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(48, 52) if ascension_level >= 7 else randoms.stream("monster_hp").random(46, 50))
    return MonsterState(
        monster_id="SlaverRed",
        name="Slaver",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="STAB",
        meta={
            "stab_damage": 14 if ascension_level >= 2 else 13,
            "scrape_damage": 9 if ascension_level >= 2 else 8,
            "vuln_amount": 1,
            "used_entangle": False,
            "first_turn": True,
            # rollMove() still consumes aiRng.random(99) on turn 1 even though getMove ignores num and fixes STAB.
            "opening_ai_roll": True,
        },
    )


def _spawn_looter(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(46, 50) if ascension_level >= 7 else randoms.stream("monster_hp").random(44, 48))
    gold_amt = 20 if ascension_level >= 17 else 15
    monster = MonsterState(
        monster_id="Looter",
        name="Looter",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="MUG",
        meta={"swipe_damage": 11 if ascension_level >= 2 else 10, "lunge_damage": 14 if ascension_level >= 2 else 12, "escape_block": 6, "gold_amt": gold_amt, "slash_count": 0, "stolen_gold": 0, "opening_ai_roll": True},
    )
    _append_power(monster, "Thievery", gold_amt, misc=gold_amt)
    return monster


def _spawn_mugger(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(50, 54) if ascension_level >= 7 else randoms.stream("monster_hp").random(48, 52))
    gold_amt = 20 if ascension_level >= 17 else 15
    monster = MonsterState(
        monster_id="Mugger",
        name="Mugger",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="MUG",
        meta={
            "swipe_damage": 11 if ascension_level >= 2 else 10,
            "big_swipe_damage": 18 if ascension_level >= 2 else 16,
            "escape_block": 17 if ascension_level >= 17 else 11,
            "gold_amt": gold_amt,
            "slash_count": 0,
            "stolen_gold": 0,
            "opening_ai_roll": True,
        },
    )
    _append_power(monster, "Thievery", gold_amt, misc=gold_amt)
    return monster


def _spawn_fungi_beast(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(24, 28) if ascension_level >= 7 else randoms.stream("monster_hp").random(22, 28))
    monster = MonsterState(
        monster_id="FungiBeast",
        name="Fungi Beast",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="BITE",
        meta={"bite_damage": 6, "grow_strength": 4 if ascension_level >= 2 else 3, "opening_move_source": "FungiBeast"},
    )
    _append_power(monster, "Spore Cloud", 2, misc=2)
    return monster


def _spawn_acid_slime_l(randoms: NativeRandomSet, ascension_level: int, *, hp_override: int | None = None, draw_x: float = 0.0) -> MonsterState:
    hp = int(hp_override if hp_override is not None else (randoms.stream("monster_hp").random(68, 72) if ascension_level >= 7 else randoms.stream("monster_hp").random(65, 69)))
    monster = MonsterState(
        monster_id="AcidSlime_L",
        name="Acid Slime (L)",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK_DEBUFF",
        next_move="WOUND_TACKLE",
        meta={
            "wound_damage": 12 if ascension_level >= 2 else 11,
            "normal_damage": 18 if ascension_level >= 2 else 16,
            "split_triggered": False,
            "opening_move_source": "AcidSlime_L",
            "draw_x": draw_x,
        },
    )
    _append_power(monster, "Split", -1)
    return monster


def _spawn_spike_slime_l(randoms: NativeRandomSet, ascension_level: int, *, hp_override: int | None = None, draw_x: float = 0.0) -> MonsterState:
    hp = int(hp_override if hp_override is not None else (randoms.stream("monster_hp").random(67, 73) if ascension_level >= 7 else randoms.stream("monster_hp").random(64, 70)))
    monster = MonsterState(
        monster_id="SpikeSlime_L",
        name="Spike Slime (L)",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK_DEBUFF",
        next_move="FLAME_TACKLE",
        meta={"tackle_damage": 18 if ascension_level >= 2 else 16, "split_triggered": False, "opening_move_source": "SpikeSlime_L", "draw_x": draw_x},
    )
    _append_power(monster, "Split", -1)
    return monster


def _get_louse(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    return _spawn_louse_normal(randoms, ascension_level) if randoms.stream("misc").random_boolean() else _spawn_louse_defensive(randoms, ascension_level)


def _roll_louse_normal_next_move(
    randoms: NativeRandomSet,
    ascension_level: int,
    move_history: list[str],
) -> str:
    num = int(randoms.stream("ai").random(99))
    last_move = move_history[-1] if move_history else None
    last_two = move_history[-2:]
    if ascension_level >= 17:
        if num < 25:
            return "BITE" if last_move == "STRENGTHEN" else "STRENGTHEN"
        if len(last_two) == 2 and last_two[0] == "BITE" and last_two[1] == "BITE":
            return "STRENGTHEN"
        return "BITE"
    if num < 25:
        return "BITE" if len(last_two) == 2 and last_two[0] == "STRENGTHEN" and last_two[1] == "STRENGTHEN" else "STRENGTHEN"
    if len(last_two) == 2 and last_two[0] == "BITE" and last_two[1] == "BITE":
        return "STRENGTHEN"
    return "BITE"


def _roll_louse_defensive_next_move(
    randoms: NativeRandomSet,
    ascension_level: int,
    move_history: list[str],
) -> str:
    num = int(randoms.stream("ai").random(99))
    last_move = move_history[-1] if move_history else None
    last_two = move_history[-2:]
    if ascension_level >= 17:
        if num < 25:
            return "BITE" if last_move == "WEAKEN" else "WEAKEN"
        if len(last_two) == 2 and last_two[0] == "BITE" and last_two[1] == "BITE":
            return "WEAKEN"
        return "BITE"
    if num < 25:
        return "BITE" if len(last_two) == 2 and last_two[0] == "WEAKEN" and last_two[1] == "WEAKEN" else "WEAKEN"
    if len(last_two) == 2 and last_two[0] == "BITE" and last_two[1] == "BITE":
        return "WEAKEN"
    return "BITE"


def _get_slaver(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    return _spawn_red_slaver(randoms, ascension_level) if randoms.stream("misc").random_boolean() else _spawn_blue_slaver(randoms, ascension_level)


def _bottom_get_strong_humanoid(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    choices = [
        _spawn_cultist(randoms, ascension_level),
        _get_slaver(randoms, ascension_level),
        _spawn_looter(randoms, ascension_level),
    ]
    return choices[int(randoms.stream("misc").random(0, len(choices) - 1))]


def _bottom_get_strong_wildlife(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    choices = [
        _spawn_fungi_beast(randoms, ascension_level),
        _spawn_jaw_worm(randoms, ascension_level),
    ]
    return choices[int(randoms.stream("misc").random(0, len(choices) - 1))]


def _bottom_get_weak_wildlife(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    choices = [
        _get_louse(randoms, ascension_level),
        _spawn_spike_slime_m(randoms, ascension_level),
        _spawn_acid_slime_m(randoms, ascension_level),
    ]
    return choices[int(randoms.stream("misc").random(0, len(choices) - 1))]


def _spawn_many_small_slimes(randoms: NativeRandomSet, ascension_level: int) -> list[MonsterState]:
    pool = ["SpikeSlime_S", "SpikeSlime_S", "SpikeSlime_S", "AcidSlime_S", "AcidSlime_S"]
    monsters: list[MonsterState] = []
    for _ in range(5):
        index = int(randoms.stream("misc").random(0, len(pool) - 1))
        key = pool.pop(index)
        monsters.append(_spawn_spike_slime_s(randoms, ascension_level) if key == "SpikeSlime_S" else _spawn_acid_slime_s(randoms, ascension_level))
    return monsters


def _spawn_gremlin_nob(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(85, 90) if ascension_level >= 8 else randoms.stream("monster_hp").random(82, 86))
    return MonsterState(
        monster_id="GremlinNob",
        name="Gremlin Nob",
        current_hp=hp,
        max_hp=hp,
        intent="BUFF",
        next_move="BELLOW",
        meta={
            "bash_damage": 8 if ascension_level >= 3 else 6,
            "rush_damage": 16 if ascension_level >= 3 else 14,
            "used_bellow": False,
            "anger_amount": 3 if ascension_level >= 18 else 2,
            "opening_ai_roll": True,
        },
    )


def _spawn_lagavulin(randoms: NativeRandomSet, ascension_level: int, *, asleep: bool = True) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(112, 115) if ascension_level >= 8 else randoms.stream("monster_hp").random(109, 111))
    monster = MonsterState(
        monster_id="Lagavulin",
        name="Lagavulin",
        current_hp=hp,
        max_hp=hp,
        intent="SLEEP" if asleep else "STRONG_DEBUFF",
        next_move="SLEEP" if asleep else "DEBUFF",
        meta={"attack_damage": 20 if ascension_level >= 3 else 18, "debuff_amount": -2 if ascension_level >= 18 else -1, "asleep": asleep, "opened": not asleep, "idle_count": 0, "debuff_turn_count": 0},
    )
    if asleep:
        _gain_block(monster, 8)
        _append_power(monster, "Metallicize", 8, misc=8)
    return monster


def _spawn_sentry(randoms: NativeRandomSet, ascension_level: int, *, first_move: str) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(39, 45) if ascension_level >= 8 else randoms.stream("monster_hp").random(38, 42))
    monster = MonsterState(
        monster_id="Sentry",
        name="Sentry",
        current_hp=hp,
        max_hp=hp,
        intent="DEBUFF" if first_move == "BOLT" else "ATTACK",
        next_move=first_move,
        meta={"beam_damage": 10 if ascension_level >= 3 else 9, "dazed_amount": 3 if ascension_level >= 18 else 2, "first_move": False},
    )
    _append_power(monster, "Artifact", 1, misc=1)
    return monster


def _spawn_slime_boss(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = 150 if ascension_level >= 9 else 140
    monster = MonsterState(
        monster_id="SlimeBoss",
        name="Slime Boss",
        current_hp=hp,
        max_hp=hp,
        intent="STRONG_DEBUFF",
        next_move="GOOP_SPRAY",
        meta={
            "sticky_slimed": 5 if ascension_level >= 19 else 3,
            "tackle_damage": 10 if ascension_level >= 4 else 9,
            "slam_damage": 38 if ascension_level >= 4 else 35,
            "split_triggered": False,
            "is_boss": True,
            "draw_x": 0.0,
            # SlimeBoss.getMove ignores its first roll but AbstractMonster.init
            # still consumes aiRng.random(99), which affects split child moves.
            "opening_ai_roll": True,
        },
    )
    _append_power(monster, "Split", -1)
    return monster


def _spawn_the_guardian(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = 250 if ascension_level >= 9 else 240
    threshold = 40 if ascension_level >= 19 else 35 if ascension_level >= 9 else 30
    monster = MonsterState(
        monster_id="TheGuardian",
        name="The Guardian",
        current_hp=hp,
        max_hp=hp,
        intent="DEFEND",
        next_move="CHARGE_UP",
        meta={
            "is_open": True,
            "close_up_triggered": False,
            "mode_shift_threshold": threshold,
            "mode_shift_increase": 10,
            "dmg_taken": 0,
            "fierce_bash_damage": 36 if ascension_level >= 4 else 32,
            "roll_damage": 10 if ascension_level >= 4 else 9,
            "whirlwind_damage": 5,
            "whirlwind_hits": 4,
            "twin_slam_damage": 8,
            "charge_block": 9,
            "defensive_block": 20,
            "thorns_damage": 4 if ascension_level >= 19 else 3,
            "vent_debuff": 2,
            "is_boss": True,
        },
    )
    _append_power(monster, "Mode Shift", threshold, misc=threshold)
    return monster


def _spawn_hexaghost(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = 264 if ascension_level >= 9 else 250
    inferno_damage = 3 if ascension_level >= 4 else 2
    tackle_damage = 6 if ascension_level >= 4 else 5
    sear_burns = 2 if ascension_level >= 19 else 1
    strength_gain = 3 if ascension_level >= 19 else 2
    return MonsterState(
        monster_id="Hexaghost",
        name="Hexaghost",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="ACTIVATE",
        meta={
            "activated": False,
            "orb_active_count": 0,
            "burn_upgraded": False,
            "divider_hits": 6,
            "tackle_damage": tackle_damage,
            "tackle_hits": 2,
            "sear_damage": 6,
            "sear_burns": sear_burns,
            "strength_gain": strength_gain,
            "strengthen_block": 12,
            "inferno_damage": inferno_damage,
            "inferno_hits": 6,
            "divider_damage": 0,
            "is_boss": True,
        },
    )


def _spawn_champ(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = 440 if ascension_level >= 9 else 420
    if ascension_level >= 19:
        slash_dmg = 18
        slap_dmg = 14
        str_amt = 4
        forge_amt = 7
        block_amt = 20
    elif ascension_level >= 9:
        slash_dmg = 18
        slap_dmg = 14
        str_amt = 3
        forge_amt = 6
        block_amt = 18
    elif ascension_level >= 4:
        slash_dmg = 18
        slap_dmg = 14
        str_amt = 3
        forge_amt = 5
        block_amt = 15
    else:
        slash_dmg = 16
        slap_dmg = 12
        str_amt = 2
        forge_amt = 5
        block_amt = 15
    initial_roll = int(randoms.stream("ai").random(99))
    forge_times = 0
    if initial_roll <= (30 if ascension_level >= 19 else 15):
        initial_move = "DEFENSIVE_STANCE"
        forge_times = 1
    elif initial_roll <= 30:
        initial_move = "GLOAT"
    elif initial_roll <= 55:
        initial_move = "FACE_SLAP"
    else:
        initial_move = "HEAVY_SLASH"
    return MonsterState(
        monster_id="Champ",
        name="The Champ",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move=initial_move,
        meta={
            "slash_damage": slash_dmg,
            "execute_damage": 10,
            "slap_damage": slap_dmg,
            "str_amt": str_amt,
            "forge_amt": forge_amt,
            "block_amt": block_amt,
            "num_turns": 1,
            "forge_times": forge_times,
            "forge_threshold": 2,
            "threshold_reached": False,
            "first_turn": True,
            "is_boss": True,
        },
    )


def _spawn_torch_head(randoms: NativeRandomSet, ascension_level: int, *, draw_x: float = 0.0) -> MonsterState:
    # TorchHead's constructor rolls HP for super(...), then setHp(...) rolls
    # again and overwrites it. The first roll only advances monsterHpRng.
    randoms.stream("monster_hp").random(40, 45) if ascension_level >= 9 else randoms.stream("monster_hp").random(38, 40)
    hp = int(randoms.stream("monster_hp").random(40, 45) if ascension_level >= 9 else randoms.stream("monster_hp").random(38, 40))
    # SpawnMonsterAction.init() calls rollMove(); TorchHead.getMove ignores the
    # value but still advances aiRng.
    randoms.stream("ai").random(99)
    monster = MonsterState(
        monster_id="TorchHead",
        name="Torch Head",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="TACKLE",
        meta={"tackle_damage": 7, "is_minion": True, "draw_x": float(draw_x)},
    )
    _append_power(monster, "Minion", -1)
    return monster


def _spawn_collector(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = 300 if ascension_level >= 9 else 282
    # TheCollector constructor calls setHp(hp), which consumes monsterHpRng even
    # though min == max. Its init roll also consumes aiRng before fixing SPAWN.
    randoms.stream("monster_hp").random(hp, hp)
    randoms.stream("ai").random(99)
    if ascension_level >= 19:
        fireball = 21
        str_amt = 5
        block_amt = 18
        mega = 5
    elif ascension_level >= 4:
        fireball = 21
        str_amt = 4
        block_amt = 18 if ascension_level >= 9 else 15
        mega = 3
    else:
        fireball = 18
        str_amt = 3
        block_amt = 18 if ascension_level >= 9 else 15
        mega = 3
    return MonsterState(
        monster_id="TheCollector",
        name="The Collector",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="SPAWN",
        meta={
            "fireball_damage": fireball,
            "str_amt": str_amt,
            "block_amt": block_amt,
            "mega_debuff_amt": mega,
            "turns_taken": 0,
            "ult_used": False,
            "initial_spawn": True,
            "is_boss": True,
        },
    )


def _spawn_bronze_orb(randoms: NativeRandomSet, ascension_level: int, count: int) -> MonsterState:
    # Java's BronzeOrb constructor rolls HP once in super(...), then setHp(...)
    # rolls again and overwrites it. The first roll only advances monsterHpRng.
    randoms.stream("monster_hp").random(52, 58)
    hp = int(randoms.stream("monster_hp").random(54, 60) if ascension_level >= 9 else randoms.stream("monster_hp").random(52, 58))
    monster = MonsterState(
        monster_id="BronzeOrb",
        name="Orb",
        current_hp=hp,
        max_hp=hp,
        intent="STRONG_DEBUFF",
        next_move="STASIS",
        meta={
            "beam_damage": 8,
            "block_amt": 12,
            "used_stasis": False,
            "count": count,
            "draw_x": -300.0 if count % 2 == 0 else 200.0,
            "is_minion": True,
        },
    )
    _append_power(monster, "Minion", -1)
    return monster


def _spawn_automaton(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = 320 if ascension_level >= 9 else 300
    # BronzeAutomaton calls setHp(fixed_hp) after super(...), which still
    # advances monsterHpRng even though min == max.
    randoms.stream("monster_hp").random(hp, hp)
    # init() calls rollMove(), consuming aiRng even though firstTurn forces
    # SPAWN_ORBS regardless of the rolled value; getMove also clears firstTurn.
    randoms.stream("ai").random(99)
    return MonsterState(
        monster_id="BronzeAutomaton",
        name="Bronze Automaton",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="SPAWN_ORBS",
        meta={
            "flail_damage": 8 if ascension_level >= 4 else 7,
            "beam_damage": 50 if ascension_level >= 4 else 45,
            "str_amt": 4 if ascension_level >= 4 else 3,
            "block_amt": 12 if ascension_level >= 9 else 9,
            "num_turns": 0,
            "first_turn": False,
            "is_boss": True,
        },
    )


def _spawn_chosen(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(98, 103) if ascension_level >= 7 else randoms.stream("monster_hp").random(95, 99))
    randoms.stream("ai").random(99)
    return MonsterState(
        monster_id="Chosen",
        name="Chosen",
        current_hp=hp,
        max_hp=hp,
        intent="STRONG_DEBUFF" if ascension_level >= 17 else "ATTACK",
        next_move="HEX" if ascension_level >= 17 else "POKE",
        meta={
            "first_turn": ascension_level < 17,
            "used_hex": False,
            "zap_damage": 21 if ascension_level >= 2 else 18,
            "debilitate_damage": 12 if ascension_level >= 2 else 10,
            "poke_damage": 6 if ascension_level >= 2 else 5,
            "drain_weak": 3,
            "drain_strength": 3,
            "debilitate_vuln": 2,
        },
    )


def _spawn_spheric_guardian(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    damage = 11 if ascension_level >= 2 else 10
    monster = MonsterState(
        monster_id="SphericGuardian",
        name="Spheric Guardian",
        current_hp=20,
        max_hp=20,
        intent="DEFEND",
        next_move="INITIAL_BLOCK_GAIN",
        meta={
            "first_move": True,
            "second_move": True,
            "attack_damage": damage,
            "block_attack_block": 15,
            "activate_block": 35 if ascension_level >= 17 else 25,
            "frail_amount": 5,
            "artifact_amount": 3,
        },
    )
    _append_power(monster, "Barricade", -1, misc=-1)
    _append_power(monster, "Artifact", 3, misc=3)
    monster.block = 40
    return monster


def _spawn_book_of_stabbing(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(168, 172) if ascension_level >= 8 else randoms.stream("monster_hp").random(160, 164))
    stab_damage = 7 if ascension_level >= 3 else 6
    big_stab_damage = 24 if ascension_level >= 3 else 21
    monster = MonsterState(
        monster_id="BookOfStabbing",
        name="Book of Stabbing",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="BIG_STAB",
        meta={
            "stab_damage": stab_damage,
            "big_stab_damage": big_stab_damage,
            "stab_count": 1,
            "opening_move_source": "BookOfStabbing",
        },
    )
    _append_power(monster, "Painful Stabs", -1, misc=-1)
    return monster


def _spawn_snake_plant(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(78, 82) if ascension_level >= 7 else randoms.stream("monster_hp").random(75, 79))
    damage = 8 if ascension_level >= 2 else 7
    monster = MonsterState(
        monster_id="SnakePlant",
        name="Snake Plant",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="CHOMPY_CHOMPS",
        meta={"chomp_damage": damage, "frail_amount": 2, "weak_amount": 2, "opening_move_source": "SnakePlant"},
    )
    _append_power(monster, "Malleable", 3, misc=3)
    return monster


def _spawn_shelled_parasite(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(70, 75) if ascension_level >= 7 else randoms.stream("monster_hp").random(68, 72))
    monster = MonsterState(
        monster_id="ShelledParasite",
        name="Shelled Parasite",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="DOUBLE_STRIKE",
        meta={
            "first_move": True,
            "fell_damage": 21 if ascension_level >= 2 else 18,
            "double_damage": 7 if ascension_level >= 2 else 6,
            "suck_damage": 12 if ascension_level >= 2 else 10,
            "frail_amount": 2,
            "opening_move_source": "ShelledParasite",
        },
    )
    _append_power(monster, "Plated Armor", 14, misc=14)
    monster.block = 14
    return monster


def _spawn_centurion(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(78, 83) if ascension_level >= 7 else randoms.stream("monster_hp").random(76, 80))
    return MonsterState(
        monster_id="Centurion",
        name="Centurion",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="SLASH",
        meta={
            "slash_damage": 14 if ascension_level >= 2 else 12,
            "fury_damage": 7 if ascension_level >= 2 else 6,
            "fury_hits": 3,
            "block_amount": 20 if ascension_level >= 17 else 15,
            "opening_move_source": "Centurion",
        },
    )


def _spawn_healer(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(50, 58) if ascension_level >= 7 else randoms.stream("monster_hp").random(48, 56))
    return MonsterState(
        monster_id="Healer",
        name="Mystic",
        current_hp=hp,
        max_hp=hp,
        intent="BUFF",
        next_move="BUFF",
        meta={
            "magic_damage": 9 if ascension_level >= 2 else 8,
            "frail_amount": 2,
            "heal_amount": 20 if ascension_level >= 17 else 16,
            "strength_amount": 4 if ascension_level >= 17 else 3 if ascension_level >= 2 else 2,
            "opening_move_source": "Healer",
        },
    )


def _spawn_gremlin_leader(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(145, 155) if ascension_level >= 8 else randoms.stream("monster_hp").random(140, 148))
    return MonsterState(
        monster_id="GremlinLeader",
        name="Gremlin Leader",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="ENCOURAGE",
        meta={
            "strength_amount": 5 if ascension_level >= 18 else 4 if ascension_level >= 3 else 3,
            "block_amount": 10 if ascension_level >= 18 else 6,
            "stab_damage": 6,
            "stab_hits": 3,
            "opening_move_source": "GremlinLeader",
        },
    )


def _spawn_taskmaster(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    randoms.stream("monster_hp").random(54, 60)
    hp = int(randoms.stream("monster_hp").random(57, 64) if ascension_level >= 8 else randoms.stream("monster_hp").random(54, 60))
    return MonsterState(
        monster_id="SlaverBoss",
        name="Taskmaster",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK_DEBUFF",
        next_move="SCOURING_WHIP",
        meta={"whip_damage": 7, "wound_count": 3 if ascension_level >= 18 else 2 if ascension_level >= 3 else 1, "opening_ai_roll": True},
    )


def _spawn_orb_walker(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    randoms.stream("monster_hp").random(90, 96)
    hp = int(randoms.stream("monster_hp").random(92, 102) if ascension_level >= 7 else randoms.stream("monster_hp").random(90, 96))
    laser_damage = 11 if ascension_level >= 2 else 10
    claw_damage = 16 if ascension_level >= 2 else 15
    next_move = "CLAW" if int(randoms.stream("ai").random(99)) < 40 else "LASER"
    monster = MonsterState(
        monster_id="OrbWalker",
        name="Orb Walker",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move=next_move,
        meta={
            "laser_damage": laser_damage,
            "claw_damage": claw_damage,
            "strength_up": 5 if ascension_level >= 17 else 3,
        },
    )
    _append_power(monster, "Generic Strength Up Power", int(monster.meta["strength_up"]))
    return monster


def _giant_head_roll_next_move(monster: MonsterState, randoms: NativeRandomSet) -> None:
    count = int(monster.meta.get("count", 5))
    if count <= 1:
        if count > -6:
            count -= 1
        monster.meta["count"] = count
        monster.next_move = "IT_IS_TIME"
        return
    count -= 1
    monster.meta["count"] = count
    roll = int(randoms.stream("ai").random(99))
    last_two = monster.move_history[-2:]
    if roll < 50:
        monster.next_move = "COUNT" if last_two == ["GLARE", "GLARE"] else "GLARE"
    else:
        monster.next_move = "GLARE" if last_two == ["COUNT", "COUNT"] else "COUNT"


def _spawn_giant_head(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = 520 if ascension_level >= 8 else 500
    monster = MonsterState(
        monster_id="GiantHead",
        name="Giant Head",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="UNKNOWN",
        meta={
            "count_damage": 13,
            "starting_death_damage": 40 if ascension_level >= 3 else 30,
            "count": 4 if ascension_level >= 18 else 5,
        },
    )
    _giant_head_roll_next_move(monster, randoms)
    return monster


def _nemesis_roll_next_move(monster: MonsterState, randoms: NativeRandomSet) -> None:
    scythe_cooldown = int(monster.meta.get("scythe_cooldown", 0)) - 1
    monster.meta["scythe_cooldown"] = scythe_cooldown
    if bool(monster.meta.get("first_move", True)):
        monster.meta["first_move"] = False
        monster.next_move = "TRI_ATTACK" if int(randoms.stream("ai").random(99)) < 50 else "TRI_BURN"
        return
    num = int(randoms.stream("ai").random(99))
    last_move = monster.move_history[-1] if monster.move_history else None
    last_two = monster.move_history[-2:]
    if num < 30:
        if last_move != "SCYTHE" and scythe_cooldown <= 0:
            monster.next_move = "SCYTHE"
            monster.meta["scythe_cooldown"] = 2
        elif randoms.stream("ai").random_boolean():
            monster.next_move = "TRI_BURN" if last_two == ["TRI_ATTACK", "TRI_ATTACK"] else "TRI_ATTACK"
        elif last_move != "TRI_BURN":
            monster.next_move = "TRI_BURN"
        else:
            monster.next_move = "TRI_ATTACK"
    elif num < 65:
        if last_two != ["TRI_ATTACK", "TRI_ATTACK"]:
            monster.next_move = "TRI_ATTACK"
        elif randoms.stream("ai").random_boolean():
            if int(monster.meta.get("scythe_cooldown", 0)) > 0:
                monster.next_move = "TRI_BURN"
            else:
                monster.next_move = "SCYTHE"
                monster.meta["scythe_cooldown"] = 2
        else:
            monster.next_move = "TRI_BURN"
    elif last_move != "TRI_BURN":
        monster.next_move = "TRI_BURN"
    elif randoms.stream("ai").random_boolean() and int(monster.meta.get("scythe_cooldown", 0)) <= 0:
        monster.next_move = "SCYTHE"
        monster.meta["scythe_cooldown"] = 2
    else:
        monster.next_move = "TRI_ATTACK"


def _spawn_nemesis(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = 200 if ascension_level >= 8 else 185
    monster = MonsterState(
        monster_id="Nemesis",
        name="Nemesis",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="UNKNOWN",
        meta={
            "fire_damage": 7 if ascension_level >= 3 else 6,
            "scythe_damage": 45,
            "burn_amount": 5 if ascension_level >= 18 else 3,
            "scythe_cooldown": 0,
            "first_move": True,
        },
    )
    _nemesis_roll_next_move(monster, randoms)
    return monster


def _awakened_one_roll_next_move(monster: MonsterState, randoms: NativeRandomSet) -> None:
    num = int(randoms.stream("ai").random(99))
    form = int(monster.meta.get("form", 1) or 1)
    first_turn = bool(monster.meta.get("first_turn", False))
    last_move = monster.move_history[-1:] if monster.move_history else []
    last_two = monster.move_history[-2:] if len(monster.move_history) >= 2 else monster.move_history[-1:]
    if form == 1:
        if first_turn:
            monster.next_move = "SLASH"
            return
        if num < 25:
            monster.next_move = "SOUL_STRIKE" if last_move != ["SOUL_STRIKE"] else "SLASH"
            return
        monster.next_move = "SLASH" if last_two != ["SLASH", "SLASH"] else "SOUL_STRIKE"
        return
    if first_turn:
        monster.next_move = "DARK_ECHO"
        return
    if num < 50:
        monster.next_move = "SLUDGE" if last_two != ["SLUDGE", "SLUDGE"] else "TACKLE"
        return
    monster.next_move = "TACKLE" if last_two != ["TACKLE", "TACKLE"] else "SLUDGE"


def _spawn_awakened_one(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = 320 if ascension_level >= 9 else 300
    monster = MonsterState(
        monster_id="AwakenedOne",
        name="Awakened One",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="UNKNOWN",
        meta={
            "form": 1,
            "first_turn": True,
            "half_dead": False,
            "slash_damage": 20,
            "soul_strike_damage": 6,
            "dark_echo_damage": 40,
            "sludge_damage": 18,
            "tackle_damage": 10,
            "curiosity": 2 if ascension_level >= 19 else 1,
            "regenerate": 15 if ascension_level >= 19 else 10,
        },
    )
    _awakened_one_roll_next_move(monster, randoms)
    return monster


def _darkling_roll_next_move(monster: MonsterState, randoms: NativeRandomSet, monsters: list[MonsterState], ascension_level: int) -> None:
    def _reroll(low: int, high: int) -> None:
        _darkling_roll_next_move_with_num(monster, int(randoms.stream("ai").random(low, high)), randoms, monsters, ascension_level)

    def _darkling_roll_next_move_with_num(
        current: MonsterState,
        num: int,
        randoms: NativeRandomSet,
        monsters: list[MonsterState],
        ascension_level: int,
    ) -> None:
        if bool(current.meta.get("half_dead", False)):
            current.next_move = "REINCARNATE"
            return
        last_move = current.move_history[-1:] if current.move_history else []
        last_two = current.move_history[-2:] if len(current.move_history) >= 2 else current.move_history[-1:]
        if bool(current.meta.get("first_move", False)):
            current.meta["first_move"] = False
            current.next_move = "HARDEN" if num < 50 else "NIP"
            return
        monster_index = monsters.index(current) if current in monsters else 0
        if num < 40:
            if last_move != ["CHOMP"] and monster_index % 2 == 0:
                current.next_move = "CHOMP"
            else:
                _reroll(40, 99)
            return
        if num < 70:
            current.next_move = "HARDEN" if last_move != ["HARDEN"] else "NIP"
            return
        if last_two != ["NIP", "NIP"]:
            current.next_move = "NIP"
            return
        _reroll(0, 99)

    _darkling_roll_next_move_with_num(monster, int(randoms.stream("ai").random(99)), randoms, monsters, ascension_level)


def _spawn_darkling(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(50, 59) if ascension_level >= 7 else randoms.stream("monster_hp").random(48, 56))
    nip_damage = int(randoms.stream("monster_hp").random(9, 13) if ascension_level >= 2 else randoms.stream("monster_hp").random(7, 11))
    return MonsterState(
        monster_id="Darkling",
        name="Darkling",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="UNKNOWN",
        meta={
            "first_move": True,
            "half_dead": False,
            "chomp_damage": 9 if ascension_level >= 2 else 8,
            "nip_damage": nip_damage,
        },
    )


def _spawn_snake_dagger(randoms: NativeRandomSet, ascension_level: int, *, reptomancer_slot: int | None = None) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(20, 25))
    meta: dict[str, Any] = {
        "first_move": True,
        "wound_damage": 9,
        "explode_damage": 25,
    }
    if reptomancer_slot is not None:
        slot = int(reptomancer_slot)
        meta["reptomancer_slot"] = slot
        # Reptomancer.java POSX/POSY slots: 0/2 to the right, 1/3 to the left.
        meta["draw_x"] = (210.0, -220.0, 180.0, -250.0)[slot]
    return MonsterState(
        monster_id="Dagger",
        name="Snake Dagger",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="WOUND",
        meta=meta,
    )


def _reptomancer_can_spawn(monsters: list[MonsterState], reptomancer: MonsterState) -> bool:
    alive_count = 0
    for monster in monsters:
        if monster is reptomancer or not _alive(monster):
            continue
        alive_count += 1
    return alive_count <= 3


def _reptomancer_roll_next_move(monster: MonsterState, randoms: NativeRandomSet, monsters: list[MonsterState]) -> None:
    def _roll(num: int) -> str:
        last_move = monster.move_history[-1:] if monster.move_history else []
        last_two = monster.move_history[-2:] if len(monster.move_history) >= 2 else monster.move_history[-1:]
        if bool(monster.meta.get("first_move", False)):
            monster.meta["first_move"] = False
            return "SPAWN_DAGGER"
        if num < 33:
            if last_move != ["SNAKE_STRIKE"]:
                return "SNAKE_STRIKE"
            return _roll(int(randoms.stream("ai").random(33, 99)))
        if num < 66:
            if last_two != ["SPAWN_DAGGER", "SPAWN_DAGGER"]:
                return "SPAWN_DAGGER" if _reptomancer_can_spawn(monsters, monster) else "SNAKE_STRIKE"
            return "SNAKE_STRIKE"
        if last_move != ["BIG_BITE"]:
            return "BIG_BITE"
        return _roll(int(randoms.stream("ai").random(0, 65)))

    monster.next_move = _roll(int(randoms.stream("ai").random(99)))


def _spawn_reptomancer(randoms: NativeRandomSet, ascension_level: int) -> list[MonsterState]:
    hp = int(randoms.stream("monster_hp").random(190, 200) if ascension_level >= 8 else randoms.stream("monster_hp").random(180, 190))
    reptomancer = MonsterState(
        monster_id="Reptomancer",
        name="Reptomancer",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="UNKNOWN",
        meta={
            "first_move": True,
            "daggers_per_spawn": 2 if ascension_level >= 18 else 1,
            "snake_strike_damage": 16 if ascension_level >= 3 else 13,
            "big_bite_damage": 34 if ascension_level >= 3 else 30,
        },
    )
    left = _spawn_snake_dagger(randoms, ascension_level, reptomancer_slot=1)
    right = _spawn_snake_dagger(randoms, ascension_level, reptomancer_slot=0)
    monsters = [left, reptomancer, right]
    _reptomancer_roll_next_move(reptomancer, randoms, monsters)
    return monsters


def _time_eater_roll_next_move(monster: MonsterState, randoms: NativeRandomSet) -> None:
    def _roll(num: int) -> str:
        if int(monster.current_hp) < int(monster.max_hp) // 2 and not bool(monster.meta.get("used_haste", False)):
            monster.meta["used_haste"] = True
            return "HASTE"
        last_move = monster.move_history[-1:] if monster.move_history else []
        last_two = monster.move_history[-2:] if len(monster.move_history) >= 2 else monster.move_history[-1:]
        if num < 45:
            if last_two != ["REVERBERATE", "REVERBERATE"]:
                return "REVERBERATE"
            return _roll(int(randoms.stream("ai").random(50, 99)))
        if num < 80:
            if last_move != ["HEAD_SLAM"]:
                return "HEAD_SLAM"
            return "REVERBERATE" if randoms.stream("ai").random_boolean(0.66) else "RIPPLE"
        if last_move != ["RIPPLE"]:
            return "RIPPLE"
        return _roll(int(randoms.stream("ai").random(0, 74)))

    monster.next_move = _roll(int(randoms.stream("ai").random(99)))


def _spawn_time_eater(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = 480 if ascension_level >= 9 else 456
    monster = MonsterState(
        monster_id="TimeEater",
        name="Time Eater",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="UNKNOWN",
        meta={
            "first_turn": True,
            "used_haste": False,
            "reverberate_damage": 8 if ascension_level >= 4 else 7,
            "head_slam_damage": 32 if ascension_level >= 4 else 26,
            "haste_block": 32 if ascension_level >= 19 else 0,
            "asc19": ascension_level >= 19,
        },
    )
    _time_eater_roll_next_move(monster, randoms)
    return monster


def _spawn_donu(ascension_level: int) -> MonsterState:
    return MonsterState(
        monster_id="Donu",
        name="Donu",
        current_hp=265 if ascension_level >= 9 else 250,
        max_hp=265 if ascension_level >= 9 else 250,
        intent="UNKNOWN",
        next_move="CIRCLE",
        meta={
            "is_attacking": False,
            "beam_damage": 12 if ascension_level >= 4 else 10,
            "artifact": 3 if ascension_level >= 19 else 2,
        },
    )


def _spawn_deca(ascension_level: int) -> MonsterState:
    return MonsterState(
        monster_id="Deca",
        name="Deca",
        current_hp=265 if ascension_level >= 9 else 250,
        max_hp=265 if ascension_level >= 9 else 250,
        intent="UNKNOWN",
        next_move="BEAM",
        meta={
            "is_attacking": True,
            "beam_damage": 12 if ascension_level >= 4 else 10,
            "artifact": 3 if ascension_level >= 19 else 2,
            "asc19": ascension_level >= 19,
        },
    )


def _spawn_transient(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    starting = 40 if ascension_level >= 2 else 30
    return MonsterState(
        monster_id="Transient",
        name="Transient",
        current_hp=999,
        max_hp=999,
        intent="ATTACK",
        next_move="ATTACK",
        meta={"attack_index": 0, "starting_damage": starting},
    )


def _maw_roll_next_move(monster: MonsterState, randoms: NativeRandomSet) -> None:
    monster.meta["turn_count"] = int(monster.meta.get("turn_count", 1)) + 1
    if not bool(monster.meta.get("roared", False)):
        monster.next_move = "ROAR"
        return
    num = int(randoms.stream("ai").random(99))
    if num < 50 and monster.move_history[-1:] != ["NOM"]:
        monster.next_move = "NOM"
        return
    if monster.move_history[-1:] in (["SLAM"], ["NOM"]):
        monster.next_move = "DROOL"
        return
    monster.next_move = "SLAM"


def _spawn_maw(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    return MonsterState(
        monster_id="Maw",
        name="The Maw",
        current_hp=300,
        max_hp=300,
        intent="UNKNOWN",
        next_move="ROAR",
        meta={
            "opening_ai_roll": True,
            "roared": False,
            "turn_count": 1,
            "str_up": 5 if ascension_level >= 17 else 3,
            "terrify_duration": 5 if ascension_level >= 17 else 3,
            "slam_damage": 30 if ascension_level >= 2 else 25,
            "nom_damage": 5,
        },
    )


def _spire_growth_roll_next_move(monster: MonsterState, randoms: NativeRandomSet, player: PlayerState, ascension_level: int) -> None:
    num = int(randoms.stream("ai").random(99))
    player_constricted = _get_power_amount(player, "Constricted") > 0
    if ascension_level >= 17 and not player_constricted and monster.move_history[-1:] != ["CONSTRICT"]:
        monster.next_move = "CONSTRICT"
        return
    if num < 50 and monster.move_history[-2:] != ["QUICK_TACKLE", "QUICK_TACKLE"]:
        monster.next_move = "QUICK_TACKLE"
        return
    if not player_constricted and monster.move_history[-1:] != ["CONSTRICT"]:
        monster.next_move = "CONSTRICT"
        return
    if monster.move_history[-2:] != ["SMASH", "SMASH"]:
        monster.next_move = "SMASH"
        return
    monster.next_move = "QUICK_TACKLE"


def _spawn_spire_growth(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    return MonsterState(
        monster_id="Serpent",
        name="Spire Growth",
        current_hp=190 if ascension_level >= 7 else 170,
        max_hp=190 if ascension_level >= 7 else 170,
        intent="UNKNOWN",
        next_move="UNKNOWN",
        meta={
            "tackle_damage": 18 if ascension_level >= 2 else 16,
            "smash_damage": 25 if ascension_level >= 2 else 22,
            "constrict_damage": 12 if ascension_level >= 17 else 10,
            "opening_move_source": "SpireGrowth",
        },
    )


def _writhing_mass_roll_next_move(
    monster: MonsterState,
    randoms: NativeRandomSet,
    *,
    include_current_move: bool = False,
) -> None:
    history = list(monster.move_history)
    if include_current_move and monster.next_move in {"BIG_HIT", "MULTI_HIT", "ATTACK_BLOCK", "ATTACK_DEBUFF", "MEGA_DEBUFF"}:
        history.append(monster.next_move)

    def _roll(num: int) -> str:
        last = history[-1:]
        if bool(monster.meta.get("first_move", False)):
            monster.meta["first_move"] = False
            if num < 33:
                return "MULTI_HIT"
            if num < 66:
                return "ATTACK_BLOCK"
            return "ATTACK_DEBUFF"
        if num < 10:
            if last != ["BIG_HIT"]:
                return "BIG_HIT"
            return _roll(int(randoms.stream("ai").random(10, 99)))
        if num < 20:
            if not bool(monster.meta.get("used_mega_debuff", False)) and last != ["MEGA_DEBUFF"]:
                return "MEGA_DEBUFF"
            if randoms.stream("ai").random_boolean(0.1):
                return "BIG_HIT"
            return _roll(int(randoms.stream("ai").random(20, 99)))
        if num < 40:
            if last != ["ATTACK_DEBUFF"]:
                return "ATTACK_DEBUFF"
            if randoms.stream("ai").random_boolean(0.4):
                return _roll(int(randoms.stream("ai").random(0, 19)))
            return _roll(int(randoms.stream("ai").random(40, 99)))
        if num < 70:
            if last != ["MULTI_HIT"]:
                return "MULTI_HIT"
            if randoms.stream("ai").random_boolean(0.3):
                return "ATTACK_BLOCK"
            return _roll(int(randoms.stream("ai").random(0, 39)))
        if last != ["ATTACK_BLOCK"]:
            return "ATTACK_BLOCK"
        return _roll(int(randoms.stream("ai").random(0, 69)))

    monster.next_move = _roll(int(randoms.stream("ai").random(99)))


def _spawn_writhing_mass(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    monster = MonsterState(
        monster_id="WrithingMass",
        name="Writhing Mass",
        current_hp=175 if ascension_level >= 7 else 160,
        max_hp=175 if ascension_level >= 7 else 160,
        intent="UNKNOWN",
        next_move="UNKNOWN",
        meta={
            "first_move": True,
            "used_mega_debuff": False,
            "big_hit_damage": 38 if ascension_level >= 2 else 32,
            "multi_hit_damage": 9 if ascension_level >= 2 else 7,
            "attack_block_damage": 16 if ascension_level >= 2 else 15,
            "attack_debuff_damage": 12 if ascension_level >= 2 else 10,
            "normal_debuff_amount": 2,
        },
    )
    _writhing_mass_roll_next_move(monster, randoms)
    return monster


def _spawn_repulsor(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(31, 38) if ascension_level >= 7 else randoms.stream("monster_hp").random(29, 35))
    next_move = "ATTACK" if int(randoms.stream("ai").random(99)) < 20 else "DAZE"
    return MonsterState(
        monster_id="Repulsor",
        name="Repulsor",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move=next_move,
        meta={
            "attack_damage": 13 if ascension_level >= 2 else 11,
            "dazed_amount": 2,
        },
    )


def _spawn_spiker(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(44, 60) if ascension_level >= 7 else randoms.stream("monster_hp").random(42, 56))
    next_move = "ATTACK" if int(randoms.stream("ai").random(99)) < 50 else "BUFF_THORNS"
    return MonsterState(
        monster_id="Spiker",
        name="Spiker",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move=next_move,
        meta={
            "attack_damage": 9 if ascension_level >= 2 else 7,
            "starting_thorns": 7 if ascension_level >= 17 else 4 if ascension_level >= 2 else 3,
            "buff_thorns": 2,
            "thorns_count": 0,
        },
    )


def _spawn_exploder(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(30, 35) if ascension_level >= 7 else randoms.stream("monster_hp").random(30, 30))
    randoms.stream("ai").random(99)
    return MonsterState(
        monster_id="Exploder",
        name="Exploder",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="ATTACK",
        meta={
            "attack_damage": 11 if ascension_level >= 2 else 9,
            "turn_count": 0,
            "explosive_damage": 30,
        },
    )


def _spawn_shapes(randoms: NativeRandomSet, ascension_level: int, *, weak: bool) -> list[MonsterState]:
    pool = ["Repulsor", "Repulsor", "Exploder", "Exploder", "Spiker", "Spiker"]
    monsters: list[MonsterState] = []
    spawners = {
        "Repulsor": lambda: _spawn_repulsor(randoms, ascension_level),
        "Exploder": lambda: _spawn_exploder(randoms, ascension_level),
        "Spiker": lambda: _spawn_spiker(randoms, ascension_level),
    }
    count = 3 if weak else 4
    for _ in range(count):
        index = int(randoms.stream("misc").random(0, len(pool) - 1))
        monster_id = pool.pop(index)
        monsters.append(spawners[monster_id]())
    return monsters


def _spawn_random_ancient_shape(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    index = int(randoms.stream("misc").random(0, 2))
    if index == 0:
        return _spawn_spiker(randoms, ascension_level)
    if index == 1:
        return _spawn_repulsor(randoms, ascension_level)
    return _spawn_exploder(randoms, ascension_level)


def _spawn_bandit_pointy(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(34, 34) if ascension_level >= 7 else randoms.stream("monster_hp").random(30, 30))
    damage = 6 if ascension_level >= 2 else 5
    return MonsterState(
        monster_id="BanditChild",
        name="Pointy",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="POINTY_SPECIAL",
        meta={"attack_damage": damage, "attack_hits": 2},
    )


def _spawn_bandit_bear(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(40, 44) if ascension_level >= 7 else randoms.stream("monster_hp").random(38, 42))
    return MonsterState(
        monster_id="BanditBear",
        name="Bear",
        current_hp=hp,
        max_hp=hp,
        intent="STRONG_DEBUFF",
        next_move="BEAR_HUG",
        meta={
            "maul_damage": 20 if ascension_level >= 2 else 18,
            "lunge_damage": 10 if ascension_level >= 2 else 9,
            "lunge_block": 9,
            "dexterity_loss": 4 if ascension_level >= 17 else 2,
        },
    )


def _spawn_bandit_leader(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(37, 41) if ascension_level >= 7 else randoms.stream("monster_hp").random(35, 39))
    return MonsterState(
        monster_id="BanditLeader",
        name="Romeo",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="MOCK",
        meta={
            "slash_damage": 17 if ascension_level >= 2 else 15,
            "agonize_damage": 12 if ascension_level >= 2 else 10,
            "weak_amount": 3 if ascension_level >= 17 else 2,
        },
    )


def _spawn_snecko(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(120, 125) if ascension_level >= 7 else randoms.stream("monster_hp").random(114, 120))
    return MonsterState(
        monster_id="Snecko",
        name="Snecko",
        current_hp=hp,
        max_hp=hp,
        intent="STRONG_DEBUFF",
        next_move="GLARE",
        meta={
            "opening_move_source": "Snecko",
            "bite_damage": 18 if ascension_level >= 2 else 15,
            "tail_damage": 10 if ascension_level >= 2 else 8,
        },
    )


def _spawn_byrd(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(26, 33) if ascension_level >= 7 else randoms.stream("monster_hp").random(25, 31))
    flight = 4 if ascension_level >= 17 else 3
    randoms.stream("ai").random(99)
    monster = MonsterState(
        monster_id="Byrd",
        name="Byrd",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="CAW" if randoms.stream("ai").random_boolean(0.375) else "PECK",
        meta={
            "is_flying": True,
            "peck_damage": 1,
            "peck_count": 6 if ascension_level >= 2 else 5,
            "swoop_damage": 14 if ascension_level >= 2 else 12,
            "headbutt_damage": 3,
            "caw_strength": 1,
            "flight_amount": flight,
        },
    )
    _append_power(monster, "Flight", flight, misc=flight)
    return monster


def _spawn_gremlin_warrior(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(21, 25) if ascension_level >= 7 else randoms.stream("monster_hp").random(20, 24))
    monster = MonsterState(
        monster_id="GremlinWarrior",
        name="Mad Gremlin",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="SCRATCH",
        meta={"scratch_damage": 5 if ascension_level >= 2 else 4, "opening_ai_roll": True},
    )
    _append_power(monster, "Angry", 2 if ascension_level >= 17 else 1, misc=2 if ascension_level >= 17 else 1)
    return monster


def _spawn_gremlin_fat(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(14, 18) if ascension_level >= 7 else randoms.stream("monster_hp").random(13, 17))
    return MonsterState(
        monster_id="GremlinFat",
        name="Fat Gremlin",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK_DEBUFF",
        next_move="BLUNT",
        meta={
            "blunt_damage": 5 if ascension_level >= 2 else 4,
            "weak_amount": 1,
            "frail_amount": 1 if ascension_level >= 17 else 0,
            "opening_ai_roll": True,
        },
    )


def _spawn_gremlin_thief(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(11, 15) if ascension_level >= 7 else randoms.stream("monster_hp").random(10, 14))
    return MonsterState(
        monster_id="GremlinThief",
        name="Sneaky Gremlin",
        current_hp=hp,
        max_hp=hp,
        intent="ATTACK",
        next_move="PUNCTURE",
        meta={"puncture_damage": 10 if ascension_level >= 2 else 9, "opening_ai_roll": True},
    )


def _spawn_gremlin_tsundere(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(13, 17) if ascension_level >= 7 else randoms.stream("monster_hp").random(12, 15))
    if ascension_level >= 17:
        block_amount = 11
    elif ascension_level >= 7:
        block_amount = 8
    else:
        block_amount = 7
    return MonsterState(
        monster_id="GremlinTsundere",
        name="Shield Gremlin",
        current_hp=hp,
        max_hp=hp,
        intent="DEFEND",
        next_move="PROTECT",
        meta={"block_amount": block_amount, "bash_damage": 8 if ascension_level >= 2 else 6, "opening_ai_roll": True},
    )


def _spawn_gremlin_wizard(randoms: NativeRandomSet, ascension_level: int) -> MonsterState:
    hp = int(randoms.stream("monster_hp").random(22, 26) if ascension_level >= 7 else randoms.stream("monster_hp").random(21, 25))
    return MonsterState(
        monster_id="GremlinWizard",
        name="Gremlin Wizard",
        current_hp=hp,
        max_hp=hp,
        intent="UNKNOWN",
        next_move="CHARGE",
        meta={"magic_damage": 30 if ascension_level >= 2 else 25, "current_charge": 1, "opening_ai_roll": True},
    )


def _spawn_gremlin_gang(randoms: NativeRandomSet, ascension_level: int) -> list[MonsterState]:
    pool = [
        "GremlinWarrior",
        "GremlinWarrior",
        "GremlinThief",
        "GremlinThief",
        "GremlinFat",
        "GremlinFat",
        "GremlinTsundere",
        "GremlinWizard",
    ]
    monsters: list[MonsterState] = []
    spawners = {
        "GremlinWarrior": lambda: _spawn_gremlin_warrior(randoms, ascension_level),
        "GremlinThief": lambda: _spawn_gremlin_thief(randoms, ascension_level),
        "GremlinFat": lambda: _spawn_gremlin_fat(randoms, ascension_level),
        "GremlinTsundere": lambda: _spawn_gremlin_tsundere(randoms, ascension_level),
        "GremlinWizard": lambda: _spawn_gremlin_wizard(randoms, ascension_level),
    }
    for _ in range(4):
        index = int(randoms.stream("misc").random(0, len(pool) - 1))
        monster_id = pool.pop(index)
        monsters.append(spawners[monster_id]())
    return monsters


def _minion_power_payload() -> dict[str, Any]:
    return {
        "power_id": "Minion",
        "id": "Minion",
        "name": "Minion",
        "amount": -1,
        "card": None,
        "damage": 0,
        "just_applied": False,
        "misc": 0,
    }


def _spawn_random_gremlin_minion(
    randoms: NativeRandomSet,
    ascension_level: int,
    *,
    rng_stream: str = "misc",
    draw_x: float | None = None,
) -> MonsterState:
    pool = [
        "GremlinWarrior",
        "GremlinWarrior",
        "GremlinThief",
        "GremlinThief",
        "GremlinFat",
        "GremlinFat",
        "GremlinTsundere",
        "GremlinWizard",
    ]
    spawners = {
        "GremlinWarrior": lambda: _spawn_gremlin_warrior(randoms, ascension_level),
        "GremlinThief": lambda: _spawn_gremlin_thief(randoms, ascension_level),
        "GremlinFat": lambda: _spawn_gremlin_fat(randoms, ascension_level),
        "GremlinTsundere": lambda: _spawn_gremlin_tsundere(randoms, ascension_level),
        "GremlinWizard": lambda: _spawn_gremlin_wizard(randoms, ascension_level),
    }
    monster_id = pool[int(randoms.stream(rng_stream).random(0, len(pool) - 1))]
    monster = spawners[monster_id]()
    if draw_x is not None:
        monster.meta["draw_x"] = float(draw_x)
    return monster


def build_encounter(encounter_name: str, randoms: NativeRandomSet, ascension_level: int) -> list[MonsterState]:
    if encounter_name == "Cultist":
        return [_spawn_cultist(randoms, ascension_level)]
    if encounter_name == "3 Cultists":
        return [_spawn_cultist(randoms, ascension_level), _spawn_cultist(randoms, ascension_level), _spawn_cultist(randoms, ascension_level)]
    if encounter_name == "Jaw Worm":
        return [_spawn_jaw_worm(randoms, ascension_level)]
    if encounter_name == "Blue Slaver":
        return [_spawn_blue_slaver(randoms, ascension_level)]
    if encounter_name == "Red Slaver":
        return [_spawn_red_slaver(randoms, ascension_level)]
    if encounter_name == "Looter":
        return [_spawn_looter(randoms, ascension_level)]
    if encounter_name == "Gremlin Gang":
        return _spawn_gremlin_gang(randoms, ascension_level)
    if encounter_name == "Small Slimes":
        if randoms.stream("misc").random_boolean():
            return [_spawn_spike_slime_s(randoms, ascension_level), _spawn_acid_slime_m(randoms, ascension_level)]
        return [_spawn_acid_slime_s(randoms, ascension_level), _spawn_spike_slime_m(randoms, ascension_level)]
    if encounter_name == "Large Slime":
        return [_spawn_acid_slime_l(randoms, ascension_level)] if randoms.stream("misc").random_boolean() else [_spawn_spike_slime_l(randoms, ascension_level)]
    if encounter_name == "Lots of Slimes":
        return _spawn_many_small_slimes(randoms, ascension_level)
    if encounter_name == "2 Louse":
        return [_get_louse(randoms, ascension_level), _get_louse(randoms, ascension_level)]
    if encounter_name == "3 Louse":
        return [_get_louse(randoms, ascension_level), _get_louse(randoms, ascension_level), _get_louse(randoms, ascension_level)]
    if encounter_name == "2 Fungi Beasts":
        return [_spawn_fungi_beast(randoms, ascension_level), _spawn_fungi_beast(randoms, ascension_level)]
    if encounter_name == "The Mushroom Lair":
        return [
            _spawn_fungi_beast(randoms, ascension_level),
            _spawn_fungi_beast(randoms, ascension_level),
            _spawn_fungi_beast(randoms, ascension_level),
        ]
    if encounter_name == "Exordium Thugs":
        return [_bottom_get_weak_wildlife(randoms, ascension_level), _bottom_get_strong_humanoid(randoms, ascension_level)]
    if encounter_name == "Exordium Wildlife":
        return [_bottom_get_strong_wildlife(randoms, ascension_level), _bottom_get_weak_wildlife(randoms, ascension_level)]
    if encounter_name == "Gremlin Nob":
        return [_spawn_gremlin_nob(randoms, ascension_level)]
    if encounter_name == "Gremlin Leader":
        minions = [
            _spawn_random_gremlin_minion(randoms, ascension_level, draw_x=-366.0),
            _spawn_random_gremlin_minion(randoms, ascension_level, draw_x=-170.0),
        ]
        for slot, minion in enumerate(minions):
            _append_power(minion, "Minion", -1)
            minion.meta["is_minion"] = True
            minion.meta["gremlin_leader_slot"] = slot
        return [*minions, _spawn_gremlin_leader(randoms, ascension_level)]
    if encounter_name == "Lagavulin":
        return [_spawn_lagavulin(randoms, ascension_level, asleep=True)]
    if encounter_name == "Lagavulin Event":
        return [_spawn_lagavulin(randoms, ascension_level, asleep=False)]
    if encounter_name == "3 Sentries":
        return [
            _spawn_sentry(randoms, ascension_level, first_move="BOLT"),
            _spawn_sentry(randoms, ascension_level, first_move="BEAM"),
            _spawn_sentry(randoms, ascension_level, first_move="BOLT"),
        ]
    if encounter_name == "Slime Boss":
        return [_spawn_slime_boss(randoms, ascension_level)]
    if encounter_name == "The Guardian":
        return [_spawn_the_guardian(randoms, ascension_level)]
    if encounter_name == "Hexaghost":
        return [_spawn_hexaghost(randoms, ascension_level)]
    if encounter_name == "Champ":
        return [_spawn_champ(randoms, ascension_level)]
    if encounter_name == "Collector":
        return [_spawn_collector(randoms, ascension_level)]
    if encounter_name == "Automaton":
        return [_spawn_automaton(randoms, ascension_level)]
    if encounter_name == "3 Byrds":
        return [_spawn_byrd(randoms, ascension_level), _spawn_byrd(randoms, ascension_level), _spawn_byrd(randoms, ascension_level)]
    if encounter_name == "4 Byrds":
        return [_spawn_byrd(randoms, ascension_level), _spawn_byrd(randoms, ascension_level), _spawn_byrd(randoms, ascension_level), _spawn_byrd(randoms, ascension_level)]
    if encounter_name == "Chosen":
        return [_spawn_chosen(randoms, ascension_level)]
    if encounter_name == "Chosen and Byrds":
        return [_spawn_byrd(randoms, ascension_level), _spawn_chosen(randoms, ascension_level)]
    if encounter_name == "Spheric Guardian":
        return [_spawn_spheric_guardian(randoms, ascension_level)]
    if encounter_name == "Book of Stabbing":
        return [_spawn_book_of_stabbing(randoms, ascension_level)]
    if encounter_name == "Slavers":
        return [_spawn_blue_slaver(randoms, ascension_level), _spawn_taskmaster(randoms, ascension_level), _spawn_red_slaver(randoms, ascension_level)]
    if encounter_name == "Colosseum Slavers":
        return [_spawn_blue_slaver(randoms, ascension_level), _spawn_red_slaver(randoms, ascension_level)]
    if encounter_name == "Colosseum Nobs":
        return [_spawn_taskmaster(randoms, ascension_level), _spawn_gremlin_nob(randoms, ascension_level)]
    if encounter_name == "Masked Bandits":
        return [_spawn_bandit_pointy(randoms, ascension_level), _spawn_bandit_leader(randoms, ascension_level), _spawn_bandit_bear(randoms, ascension_level)]
    if encounter_name == "Snecko":
        return [_spawn_snecko(randoms, ascension_level)]
    if encounter_name == "Orb Walker":
        return [_spawn_orb_walker(randoms, ascension_level)]
    if encounter_name == "2 Orb Walkers":
        return [_spawn_orb_walker(randoms, ascension_level), _spawn_orb_walker(randoms, ascension_level)]
    if encounter_name == "Giant Head":
        return [_spawn_giant_head(randoms, ascension_level)]
    if encounter_name == "Nemesis":
        return [_spawn_nemesis(randoms, ascension_level)]
    if encounter_name == "Awakened One":
        return [
            _spawn_cultist(randoms, ascension_level),
            _spawn_cultist(randoms, ascension_level),
            _spawn_awakened_one(randoms, ascension_level),
        ]
    if encounter_name == "3 Darklings":
        monsters = [_spawn_darkling(randoms, ascension_level) for _ in range(3)]
        for monster in monsters:
            _darkling_roll_next_move(monster, randoms, monsters, ascension_level)
        return monsters
    if encounter_name == "Reptomancer":
        return _spawn_reptomancer(randoms, ascension_level)
    if encounter_name == "Time Eater":
        return [_spawn_time_eater(randoms, ascension_level)]
    if encounter_name == "Donu and Deca":
        return [_spawn_deca(ascension_level), _spawn_donu(ascension_level)]
    if encounter_name == "Transient":
        return [_spawn_transient(randoms, ascension_level)]
    if encounter_name == "Maw":
        return [_spawn_maw(randoms, ascension_level)]
    if encounter_name == "Spire Growth":
        return [_spawn_spire_growth(randoms, ascension_level)]
    if encounter_name == "Writhing Mass":
        return [_spawn_writhing_mass(randoms, ascension_level)]
    if encounter_name == "Jaw Worm Horde":
        return [
            _spawn_jaw_worm(randoms, ascension_level, hard_mode=True),
            _spawn_jaw_worm(randoms, ascension_level, hard_mode=True),
            _spawn_jaw_worm(randoms, ascension_level, hard_mode=True),
        ]
    if encounter_name == "3 Shapes":
        return _spawn_shapes(randoms, ascension_level, weak=True)
    if encounter_name == "4 Shapes":
        return _spawn_shapes(randoms, ascension_level, weak=False)
    if encounter_name == "Sphere and 2 Shapes":
        return [
            _spawn_random_ancient_shape(randoms, ascension_level),
            _spawn_random_ancient_shape(randoms, ascension_level),
            _spawn_spheric_guardian(randoms, ascension_level),
        ]
    if encounter_name == "Snake Plant":
        return [_spawn_snake_plant(randoms, ascension_level)]
    if encounter_name == "Shelled Parasite and Fungi":
        return [_spawn_shelled_parasite(randoms, ascension_level), _spawn_fungi_beast(randoms, ascension_level)]
    if encounter_name == "Shell Parasite":
        return [_spawn_shelled_parasite(randoms, ascension_level)]
    if encounter_name == "Centurion and Healer":
        return [_spawn_centurion(randoms, ascension_level), _spawn_healer(randoms, ascension_level)]
    if encounter_name == "2 Thieves":
        return [_spawn_looter(randoms, ascension_level), _spawn_mugger(randoms, ascension_level)]
    if encounter_name == "Cultist and Chosen":
        return [_spawn_cultist(randoms, ascension_level), _spawn_chosen(randoms, ascension_level)]
    if encounter_name == "Sentry and Sphere":
        return [_spawn_sentry(randoms, ascension_level, first_move="BOLT"), _spawn_spheric_guardian(randoms, ascension_level)]
    raise NotImplementedError(f"native_sim_v3 encounter {encounter_name!r} is not implemented yet.")


@dataclass(slots=True)
class CombatStepResult:
    outcome: str = "UNDECIDED"


class CombatEngine:
    def __init__(
        self,
        *,
        encounter_name: str,
        room_type: str = "MonsterRoom",
        randoms: NativeRandomSet,
        ascension_level: int,
        act: int = 1,
        character: str = "IRONCLAD",
        player: PlayerState,
        master_deck: list[dict[str, Any]],
        relics: list[dict[str, Any]] | None = None,
        potions: list[dict[str, Any]] | None = None,
        gold: int = 99,
        source_card_pools: dict[str, list[str]] | None = None,
        has_emerald_key: bool = False,
        prebuilt_monsters: list[MonsterState] | None = None,
        elite_trigger: bool = False,
    ) -> None:
        self.encounter_name = encounter_name
        self.randoms = randoms
        self.ascension_level = int(ascension_level)
        self.act = int(act)
        self.player_class = str(character)
        self.player = player
        self.relics = list(relics or [])
        for relic in self.relics:
            if str(relic.get("relic_id") or relic.get("id") or "") == "Necronomicon":
                relic["counter"] = -1
        self._relic_ids = _owned_relic_ids(self.relics)
        self.potions = list(potions or [])
        self.gold = int(gold)
        self.has_emerald_key = bool(has_emerald_key)
        self.elite_trigger = bool(elite_trigger)
        self.bonus_reward_gold = 0
        self.master_deck = [_copy_card(card) for card in master_deck]
        self.source_card_pools = {
            key: list(values)
            for key, values in (
                source_card_pools or initialize_source_card_pools(character=self.player_class)
            ).items()
        }
        self.state = CombatState(
            player=self.player,
            monsters=list(prebuilt_monsters) if prebuilt_monsters is not None else build_encounter(encounter_name, randoms, self.ascension_level),
            hand=[],
            draw_pile=[],
            discard_pile=[],
            exhaust_pile=[],
            turn=0,
            encounter_name=encounter_name,
            room_type=room_type,
        )
        self.outcome = "UNDECIDED"
        self.player_damage_taken_this_combat = 0
        self.cards_played_this_turn = 0
        self._double_attack_damage = False
        self._necronomicon_activated_turn: int | None = None
        self._in_monster_turn = False
        self._defer_curl_up_depth = 0
        self._pending_curl_up_monsters: list[MonsterState] = []
        self._defer_plated_armor_reduce_depth = 0
        self._pending_plated_armor_reductions: list[MonsterState] = []
        self._defer_malleable_block_depth = 0
        self._pending_malleable_blocks: list[tuple[MonsterState, int]] = []
        self._defer_guardian_mode_shift_depth = 0
        self._pending_guardian_mode_shift_monsters: list[MonsterState] = []
        self._preserve_next_guardian_mode_shift_block_clear = False
        self._defer_gremlin_horn_depth = 0
        self._pending_gremlin_horn_rewards = 0
        self._defer_flight_reduction_depth = 0
        self._defer_dark_embrace_draw_depth = 0
        self._pending_dark_embrace_draws = 0
        self._pending_dark_embrace_draw_batches: list[int] = []
        self._defer_fire_breathing_damage_depth = 0
        self._pending_fire_breathing_damage: list[int] = []
        self._defer_juggernaut_damage_depth = 0
        self._pending_juggernaut_damage: list[int] = []
        self._defer_feel_no_pain_block_depth = 0
        self._pending_feel_no_pain_blocks: list[int] = []
        self._defer_stasis_release_until_after_end_turn_discard = False
        self._pending_stasis_release_cards: list[dict[str, Any]] = []
        self._pending_bronze_roll_events: list[tuple[str, MonsterState | None]] = []
        self._pending_bronze_automaton_ai_rolls = 0
        self._early_moved_used_card_ids: set[int] = set()
        self._monster_after_use_triggered_card_ids: set[int] = set()
        self.pending_card_select: dict[str, Any] | None = None
        self._pending_end_turn_resume = False
        self._victory_relics_handled = False
        self._apply_pre_battle_monster_effects()
        self._apply_pre_battle_relic_effects()
        self._consume_opening_move_rolls()
        self._setup_opening_draw()

    def _relic(self, relic_id: str) -> dict[str, Any] | None:
        for relic in self.relics:
            if str(relic.get("relic_id") or relic.get("id") or "") == relic_id:
                return relic
        return None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        if bool(state.pop("_teacher_branch_clone_slim", False)):
            source_card_pools = state.get("source_card_pools")
            if isinstance(source_card_pools, dict):
                source_key = _register_teacher_source_card_pools(source_card_pools)
                state["source_card_pools"] = None
                state["_teacher_source_card_pools_key"] = source_key
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        source_key = self.__dict__.pop("_teacher_source_card_pools_key", None)
        if source_key is not None and self.__dict__.get("source_card_pools") is None:
            source_card_pools = _TEACHER_SOURCE_CARD_POOL_REGISTRY.get(int(source_key))
            if source_card_pools is None:
                source_card_pools = initialize_source_card_pools(character=str(getattr(self, "player_class", "IRONCLAD") or "IRONCLAD"))
            self.source_card_pools = source_card_pools
        if "_relic_ids" not in self.__dict__:
            self._relic_ids = _owned_relic_ids(getattr(self, "relics", []) or [])
        if "_necronomicon_activated_turn" not in self.__dict__:
            self._necronomicon_activated_turn = None

    def _apply_pre_battle_relic_effects(self) -> None:
        top_power_effects_by_relic: list[tuple[int, list[tuple[str, str, int, int]]]] = []
        for index, relic in enumerate(self.relics):
            effects = self._top_start_power_effects_for_relic(relic)
            if effects:
                top_power_effects_by_relic.append((index, effects))

        top_power_relic_indexes = {index for index, _ in top_power_effects_by_relic}
        # Base-game relics enqueue these with addToTop during atBattleStart; later
        # relics therefore resolve before earlier relics.
        for _, effects in reversed(top_power_effects_by_relic):
            for mode, power_id, amount, misc in effects:
                if mode == "append":
                    _append_power(self.player, power_id, amount, misc=misc)
                else:
                    _add_power(self.player, power_id, amount)

        for index, relic in enumerate(self.relics):
            if index in top_power_relic_indexes:
                continue
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id == "NeowsBlessing":
                counter = int(relic.get("counter", 0) or 0)
                if counter <= 0:
                    continue
                counter -= 1
                if counter == 0:
                    relic["counter"] = -2
                    relic["used_up"] = True
                else:
                    relic["counter"] = counter
                for monster in self.state.monsters:
                    monster.current_hp = min(int(monster.current_hp), 1)
                continue
            if relic_id == "Girya":
                counter = int(relic.get("counter", 0) or 0)
                if counter > 0:
                    _add_power(self.player, "Strength", counter)
                continue
            if relic_id == "Du-Vu Doll":
                counter = int(relic.get("counter", 0) or 0)
                if counter > 0:
                    _add_power(self.player, "Strength", counter)
                continue
            if relic_id == "FossilizedHelix":
                _append_power(self.player, "Buffer", 1, misc=1)
                continue
            if relic_id == "Bronze Scales":
                _append_power(self.player, "Thorns", 3, misc=3)
                continue
            if relic_id == "Thread and Needle":
                _append_power(self.player, "Plated Armor", 4, misc=4)
                continue
            if relic_id == "Blood Vial":
                self._heal_player(2)
                continue
            if relic_id == "Akabeko":
                _append_power(self.player, "Vigor", 8, misc=8)
                continue
            if relic_id == "Vajra":
                _add_power(self.player, "Strength", 1)
                continue
            if relic_id == "Oddly Smooth Stone":
                _add_power(self.player, "Dexterity", 1)
                continue
            if relic_id == "ClockworkSouvenir":
                _add_power(self.player, "Artifact", 1)
                continue
            if relic_id == "MutagenicStrength":
                _add_power(self.player, "Strength", 3)
                self._apply_player_temporary_strength_loss(3)
                continue
            if relic_id == "TwistedFunnel":
                for monster in self.state.monsters:
                    if _alive(monster):
                        self._apply_monster_debuff(monster, "Poison", 4)
                continue
            if relic_id == "Pantograph" and self.state.room_type == "MonsterRoomBoss":
                self._heal_player(25)
                continue
            if relic_id == "Sling" and (self.state.room_type == "MonsterRoomElite" or self.elite_trigger):
                _add_power(self.player, "Strength", 2)
                continue
            if relic_id in {"Happy Flower", "Incense Burner", "Nunchaku", "InkBottle", "Sundial"}:
                relic.setdefault("counter", 0)
                continue
            if relic_id == "Velvet Choker":
                relic["counter"] = 0
                continue
            if relic_id == "Pen Nib":
                relic.setdefault("counter", 0)
                if int(relic.get("counter", 0) or 0) == 9:
                    _append_power(self.player, "Pen Nib", 1, misc=1)
                continue
            if relic_id in {"Kunai", "Shuriken", "Ornamental Fan", "Letter Opener", "StoneCalendar"}:
                relic["counter"] = 0
                continue
            if relic_id in {"HornCleat", "CaptainsWheel"}:
                relic["counter"] = 0
                relic["grayscale"] = False
                continue
            if relic_id == "Pocketwatch":
                relic["counter"] = 0
                relic["first_turn"] = True
                continue
            if relic_id == "Centennial Puzzle":
                relic["used_this_combat"] = False
                continue
            if relic_id == "Red Skull":
                relic["is_active"] = False
                if self._player_is_bloodied():
                    _direct_add_power(self.player, "Strength", 3)
                    relic["is_active"] = True
                continue
            if relic_id == "Art of War":
                relic["first_turn"] = True
                relic["gain_energy_next"] = True
                continue
            if relic_id == "OrangePellets":
                relic["attack_played"] = False
                relic["skill_played"] = False
                relic["power_played"] = False
                continue
            if relic_id == "Snecko Eye":
                self._apply_player_debuff("Confusion", -1)
                continue
            if relic_id == "Bag of Marbles":
                for monster in self.state.monsters:
                    if _alive(monster):
                        self._apply_monster_debuff(monster, "Vulnerable", 1)
                continue
            if relic_id == "Red Mask":
                for monster in self.state.monsters:
                    if _alive(monster):
                        self._apply_monster_debuff(monster, "Weak", 1)
                continue
            if relic_id == "GremlinMask":
                self._apply_player_debuff("Weak", 1)
                continue
            if relic_id == "Philosopher's Stone":
                for monster in self.state.monsters:
                    self._apply_spawn_relic_effects(monster)
                continue
            if relic_id == "Enchiridion":
                generated = self._make_random_card(
                    predicate=lambda _cid, card_def: card_def.type == "POWER",
                    uuid_prefix="enchiridion",
                    free_this_turn=True,
                    source_scope="COLORED",
                )
                if generated is not None:
                    self._add_temp_card_to_hand_or_discard(generated, reset_for_discard=False)
                continue

    def _top_start_power_effects_for_relic(self, relic: dict[str, Any]) -> list[tuple[str, str, int, int]]:
        relic_id = str(relic.get("relic_id") or relic.get("id") or "")
        if relic_id == "Girya":
            counter = int(relic.get("counter", 0) or 0)
            return [("add", "Strength", counter, counter)] if counter > 0 else []
        if relic_id == "Du-Vu Doll":
            counter = int(relic.get("counter", 0) or 0)
            return [("add", "Strength", counter, counter)] if counter > 0 else []
        if relic_id == "Bronze Scales":
            return [("append", "Thorns", 3, 3)]
        if relic_id == "Thread and Needle":
            return [("append", "Plated Armor", 4, 4)]
        if relic_id == "Akabeko":
            return [("append", "Vigor", 8, 8)]
        if relic_id == "Vajra":
            return [("add", "Strength", 1, 1)]
        if relic_id == "Oddly Smooth Stone":
            return [("add", "Dexterity", 1, 1)]
        if relic_id == "ClockworkSouvenir":
            return [("add", "Artifact", 1, 1)]
        if relic_id == "MutagenicStrength":
            return [("add", "Flex", 3, 3), ("add", "Strength", 3, 3)]
        if relic_id == "Sling" and (self.state.room_type == "MonsterRoomElite" or self.elite_trigger):
            return [("add", "Strength", 2, 2)]
        return []

    def _apply_spawn_relic_effects(self, monster: MonsterState) -> None:
        relic_ids = self._relic_ids
        if "Philosopher's Stone" in relic_ids and _alive(monster):
            _direct_add_power(monster, "Strength", 1)

    def _append_spawned_monster(self, monster: MonsterState) -> None:
        self._apply_spawn_relic_effects(monster)
        self._consume_monster_opening_ai_roll(monster)
        self.state.monsters.append(monster)

    def _insert_spawned_monster_by_draw_x(self, monster: MonsterState, *, consume_opening_ai: bool = True) -> None:
        self._apply_spawn_relic_effects(monster)
        if consume_opening_ai:
            self._consume_monster_opening_ai_roll(monster)
        position = 0
        monster_x = _monster_draw_x(monster)
        for existing in self.state.monsters:
            if monster_x > _monster_draw_x(existing):
                position += 1
                continue
            break
        self.state.monsters.insert(position, monster)

    def _insert_reptomancer_dagger(self, reptomancer: MonsterState, dagger: MonsterState, slot: int) -> None:
        self._apply_spawn_relic_effects(dagger)
        self._consume_monster_opening_ai_roll(dagger)
        dagger.meta["reptomancer_slot"] = int(slot)
        for index, existing in enumerate(self.state.monsters):
            if existing.monster_id == "Dagger" and int(existing.meta.get("reptomancer_slot", -1)) == int(slot):
                self.state.monsters.insert(index, dagger)
                return
        try:
            reptomancer_index = self.state.monsters.index(reptomancer)
        except ValueError:
            reptomancer_index = len(self.state.monsters)
        insert_index = reptomancer_index + 1 if int(slot) in {0, 2} else reptomancer_index
        self.state.monsters.insert(max(0, min(insert_index, len(self.state.monsters))), dagger)

    def _extend_spawned_monsters(self, monsters: list[MonsterState]) -> None:
        for monster in monsters:
            self._apply_spawn_relic_effects(monster)
        self.state.monsters.extend(monsters)

    def _insert_split_children_around_parent(self, parent: MonsterState, children: list[MonsterState]) -> None:
        for child in children:
            self._apply_spawn_relic_effects(child)
            position = 0
            child_x = _monster_draw_x(child)
            for existing in self.state.monsters:
                if child_x > _monster_draw_x(existing):
                    position += 1
            self.state.monsters.insert(position, child)

    def _roll_spawned_slime_initial_move(self, monster: MonsterState) -> None:
        monster.meta.pop("first_move", None)
        monster.meta.pop("opening_move_source", None)
        if monster.monster_id == "AcidSlime_M":
            self._acid_slime_m_roll_next_move(monster)
        elif monster.monster_id == "SpikeSlime_M":
            self._spike_slime_m_roll_next_move(monster)
        elif monster.monster_id == "AcidSlime_L":
            self._acid_slime_l_roll_next_move(monster)
        elif monster.monster_id == "SpikeSlime_L":
            self._spike_slime_l_roll_next_move(monster)

    def _apply_pre_battle_monster_effects(self) -> None:
        for monster in self.state.monsters:
            if monster.monster_id in {"FuzzyLouseNormal", "FuzzyLouseDefensive"}:
                if self.ascension_level >= 17:
                    curl_up = int(self.randoms.stream("monster_hp").random(9, 12))
                elif self.ascension_level >= 7:
                    curl_up = int(self.randoms.stream("monster_hp").random(4, 8))
                else:
                    curl_up = int(self.randoms.stream("monster_hp").random(3, 7))
                _append_power(monster, "Curl Up", curl_up, misc=curl_up)
                monster.meta["pending_curl_up"] = False
            elif monster.monster_id == "Spiker":
                _add_power(monster, "Thorns", int(monster.meta.get("starting_thorns", 3)))
            elif monster.monster_id == "Exploder":
                _add_power(monster, "Explosive", 3)
            elif monster.monster_id == "AwakenedOne":
                _append_power(monster, "Regenerate", int(monster.meta.get("regenerate", 10)))
                _append_power(monster, "Curiosity", int(monster.meta.get("curiosity", 1)))
                _append_power(monster, "Unawakened", -1)
                if self.ascension_level >= 4:
                    _add_power(monster, "Strength", 2)
            elif monster.monster_id == "Darkling":
                _append_power(monster, "Life Link", -1)
            elif monster.monster_id == "JawWorm" and bool(monster.meta.get("hard_mode", False)):
                _add_power(monster, "Strength", int(monster.meta.get("bellow_str", 3)))
                _gain_block(monster, int(monster.meta.get("bellow_block", 6)))
            elif monster.monster_id == "TimeEater":
                _append_power(monster, "Time Warp", 0)
            elif monster.monster_id in {"Donu", "Deca"}:
                _append_power(monster, "Artifact", int(monster.meta.get("artifact", 2)), misc=int(monster.meta.get("artifact", 2)))
            elif monster.monster_id == "BronzeAutomaton":
                _append_power(monster, "Artifact", 3, misc=3)
            elif monster.monster_id == "GiantHead":
                _append_power(monster, "Slow", 0)
            elif monster.monster_id == "Transient":
                _append_power(monster, "Fading", 6 if self.ascension_level >= 17 else 5)
                _append_power(monster, "Shifting", 1)
            elif monster.monster_id == "WrithingMass":
                _append_power(monster, "Malleable", 3, misc=3)
                _append_power(monster, "Compulsive", -1)

    def _consume_monster_opening_ai_roll(self, monster: MonsterState) -> None:
        if not bool(monster.meta.get("opening_ai_roll", False)):
            return
        self.randoms.stream("ai").random(99)
        monster.meta["opening_ai_roll"] = False

    def _consume_opening_move_rolls(self) -> None:
        for monster in self.state.monsters:
            opening_move_source = str(monster.meta.get("opening_move_source") or "")
            if opening_move_source == "AcidSlime_M":
                self._acid_slime_m_roll_next_move(monster)
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "SpikeSlime_M":
                self._spike_slime_m_roll_next_move(monster)
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "AcidSlime_L":
                self._acid_slime_l_roll_next_move(monster)
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "SpikeSlime_L":
                self._spike_slime_l_roll_next_move(monster)
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "AcidSlime_S":
                self._acid_slime_s_roll_opening_move(monster)
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "JawWorm":
                self._jaw_worm_roll_next_move(monster)
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "FuzzyLouseNormal":
                monster.next_move = _roll_louse_normal_next_move(self.randoms, self.ascension_level, [])
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "FuzzyLouseDefensive":
                monster.next_move = _roll_louse_defensive_next_move(self.randoms, self.ascension_level, [])
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "SlaverBlue":
                num = int(self.randoms.stream("ai").random(99))
                monster.next_move = "STAB" if num >= 40 else "RAKE"
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "FungiBeast":
                self._fungi_beast_roll_next_move(monster)
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "ShelledParasite":
                self._shelled_parasite_roll_next_move(monster)
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "SnakePlant":
                self._snake_plant_roll_next_move(monster)
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "Centurion":
                num = int(self.randoms.stream("ai").random(99))
                allies_alive = sum(1 for current in self.state.monsters if _alive(current))
                monster.next_move = "PROTECT" if num >= 65 and allies_alive > 1 else "SLASH"
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "Healer":
                num = int(self.randoms.stream("ai").random(99))
                need_to_heal = sum(max(0, int(current.max_hp) - int(current.current_hp)) for current in self.state.monsters if _alive(current))
                heal_threshold = 20 if self.ascension_level >= 17 else 15
                if need_to_heal > heal_threshold:
                    monster.next_move = "HEAL"
                elif num >= 40:
                    monster.next_move = "ATTACK"
                else:
                    monster.next_move = "BUFF"
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "Snecko":
                # rollMove() consumes aiRng.random(99); Snecko.getMove ignores it on turn 1 and fixes GLARE.
                self.randoms.stream("ai").random(99)
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "BookOfStabbing":
                self._book_of_stabbing_roll_next_move(monster, int(self.randoms.stream("ai").random(99)))
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "GremlinLeader":
                self._gremlin_leader_roll_next_move(monster)
                monster.meta.pop("opening_move_source", None)
                continue
            if opening_move_source == "SpireGrowth":
                _spire_growth_roll_next_move(monster, self.randoms, self.player, self.ascension_level)
                monster.meta.pop("opening_move_source", None)
                continue
            self._consume_monster_opening_ai_roll(monster)

    def _roll_emerald_elite_buff(self) -> int:
        return int(self.randoms.stream("map").random(0, 3))

    def _apply_emerald_elite_buff(self) -> None:
        if self.state.room_type != "MonsterRoomElite" or not self.has_emerald_key:
            return
        buff_roll = self._roll_emerald_elite_buff()
        rules = emerald_elite_rules()
        for monster in self.state.monsters:
            if buff_roll == 0:
                _add_power(monster, "Strength", rules.strength_amount(self.act))
            elif buff_roll == 1:
                increase = _round_positive_half_up(float(monster.max_hp) * rules.max_hp_bonus_ratio)
                monster.max_hp += increase
                monster.current_hp += increase
            elif buff_roll == 2:
                _add_power(monster, "Metallicize", rules.metallicize_amount(self.act))
            else:
                _add_power(monster, "Regenerate", rules.regenerate_amount(self.act))

    def _shuffle_cards(self, cards: list[dict[str, Any]], *, count_as_shuffle: bool = True) -> None:
        seed = int(self.randoms.stream("shuffle").random_long())
        java_shuffle_in_place(cards, seed)
        if count_as_shuffle:
            for relic in self.relics:
                relic_id = str(relic.get("relic_id") or relic.get("id") or "")
                if relic_id == "Sundial":
                    counter = int(relic.get("counter", 0) or 0) + 1
                    if counter == 3:
                        relic["counter"] = 0
                        self.player.energy += 2
                    else:
                        relic["counter"] = counter
                elif relic_id == "TheAbacus":
                    self._gain_player_block(6)

    def _add_card_to_draw_pile_random_spot(self, card: dict[str, Any]) -> None:
        if not self.state.draw_pile:
            self.state.draw_pile.append(card)
            return
        index = int(self.randoms.stream("card_random").random(len(self.state.draw_pile) - 1))
        self.state.draw_pile.insert(index, card)

    def _make_temp_card_in_hand(self, card_id: str, *, uuid: str) -> None:
        card = make_card(card_id, uuid=uuid)
        if len(self.state.hand) < 10:
            self.state.hand.append(card)
        else:
            self.state.discard_pile.append(card)

    def _grant_stolen_gold_on_death(self, monster: MonsterState) -> None:
        if monster.monster_id in {"Looter", "Mugger"}:
            stolen_gold = int(monster.meta.get("stolen_gold", 0) or 0)
            if stolen_gold > 0:
                self.bonus_reward_gold += stolen_gold
                monster.meta["stolen_gold"] = 0

    def _release_stasis_cards_on_death(self, monster: MonsterState) -> None:
        for power in list(monster.powers):
            if _canonical_power_id(str(power.get("power_id") or power.get("id") or "")) != "Stasis":
                continue
            card = power.get("card")
            if not isinstance(card, dict):
                continue
            if self._defer_stasis_release_until_after_end_turn_discard:
                self._pending_stasis_release_cards.append(card)
                continue
            if len(self.state.hand) < 10:
                self.state.hand.append(card)
            else:
                self.state.discard_pile.append(card)

    def _flush_pending_stasis_release_cards(self) -> None:
        pending = list(self._pending_stasis_release_cards)
        self._pending_stasis_release_cards = []
        for card in pending:
            if len(self.state.hand) < 10:
                self.state.hand.append(card)
            else:
                self.state.discard_pile.append(card)

    def _kill_monster(self, monster: MonsterState, *, escaped: bool = False, suppress_victory: bool = False) -> None:
        specimen_poison = _get_power_amount(monster, "Poison")
        spore_cloud = _get_power_amount(monster, "Spore Cloud")
        monster.meta["escaped"] = bool(escaped)
        if escaped:
            monster.block = 0
            monster.intent = "ESCAPE"
            monster.move_adjusted_damage = 0
            monster.move_hits = 0
        else:
            monster.current_hp = 0
            monster.block = 0
            self._release_stasis_cards_on_death(monster)
            monster.powers = []
            if monster.monster_id == "Mugger":
                self.randoms.stream("ai").random(2)
            if spore_cloud > 0 and any(current is not monster and _alive(current) for current in self.state.monsters):
                self._apply_player_debuff("Vulnerable", spore_cloud)
            if monster.monster_id == "AwakenedOne":
                for current in self.state.monsters:
                    if current is monster or not _alive(current):
                        continue
                    if current.monster_id == "Cultist":
                        self._kill_monster(current, escaped=True)
            if monster.monster_id == "Reptomancer":
                for current in self.state.monsters:
                    if current is monster or not _alive(current):
                        continue
                    self._kill_monster(current)
            if specimen_poison > 0 and "The Specimen" in self._relic_ids:
                remaining = [current for current in self.state.monsters if current is not monster and _alive(current)]
                if remaining:
                    picked = remaining[int(self.randoms.stream("misc").random(0, len(remaining) - 1))]
                    self._apply_monster_debuff(picked, "Poison", specimen_poison)
            self._grant_stolen_gold_on_death(monster)
            if "Gremlin Horn" in self._relic_ids:
                remaining = [current for current in self.state.monsters if current is not monster and _alive(current)]
                if remaining:
                    self._trigger_gremlin_horn_reward()
            if monster.monster_id in {"TheCollector", "BronzeAutomaton"}:
                # These bosses enqueue SuicideAction for every remaining
                # monster after super.die(); model it immediately so victory is
                # decided against the same monster set as STS.
                for current in self.state.monsters:
                    if current is monster or not _alive(current):
                        continue
                    self._kill_monster(current, suppress_victory=True)
            if monster.monster_id == "GremlinLeader":
                for current in self.state.monsters:
                    if current is monster or not _alive(current):
                        continue
                    self._kill_monster(current, escaped=True)
        if not suppress_victory and int(self.player.current_hp) > 0 and not _any_monsters_alive(self.state.monsters):
            self.outcome = "VICTORY"

    def _trigger_gremlin_horn_reward(self) -> None:
        if self._defer_gremlin_horn_depth > 0:
            self._pending_gremlin_horn_rewards += 1
            return
        self.player.energy += 1
        self.draw_cards(1)

    def _flush_deferred_gremlin_horn_rewards(self) -> None:
        pending = int(self._pending_gremlin_horn_rewards)
        self._pending_gremlin_horn_rewards = 0
        for _ in range(pending):
            self.player.energy += 1
            self.draw_cards(1)

    def _handle_special_monster_zero_hp(self, monster: MonsterState) -> bool:
        if monster.monster_id == "AwakenedOne" and int(monster.meta.get("form", 1) or 1) == 1 and not bool(monster.meta.get("half_dead", False)):
            monster.meta["half_dead"] = True
            monster.meta["form"] = 2
            monster.meta["first_turn"] = True
            monster.next_move = "REBIRTH"
            monster.intent = "UNKNOWN"
            retained: list[dict[str, Any]] = []
            for power in monster.powers:
                power_id = str(power.get("power_id") or power.get("id") or "")
                amount_value = int(power.get("amount") or 0)
                if power_id in {"Curiosity", "Unawakened", "Shackled"}:
                    continue
                if power_id in MONSTER_DEBUFF_POWERS and (power_id not in NEGATIVE_STACKABLE_POWERS or amount_value < 0):
                    continue
                retained.append(power)
            monster.powers = retained
            monster.block = 0
            return True
        if monster.monster_id == "Darkling" and not bool(monster.meta.get("half_dead", False)):
            monster.meta["half_dead"] = True
            monster.next_move = "COUNT"
            monster.intent = "UNKNOWN"
            monster.block = 0
            monster.powers = []
            if "Gremlin Horn" in self._relic_ids and any(current is not monster and _alive(current) for current in self.state.monsters):
                self._trigger_gremlin_horn_reward()
            darklings = [current for current in self.state.monsters if current.monster_id == "Darkling"]
            all_half_dead = all(bool(current.meta.get("half_dead", False)) for current in darklings)
            if all_half_dead:
                for current in darklings:
                    current.meta["half_dead"] = False
                    current.current_hp = 0
                if not any(_alive(current) for current in self.state.monsters):
                    self.outcome = "VICTORY"
            return True
        return False

    def _wake_lagavulin_after_hp_loss(self, monster: MonsterState) -> None:
        if monster.monster_id != "Lagavulin" or bool(monster.meta.get("opened", False)) or not _alive(monster):
            return
        monster.meta["opened"] = True
        monster.meta["asleep"] = False
        monster.next_move = "STUN"
        monster.intent = "STUN"
        monster.move_adjusted_damage = -1
        monster.move_hits = 1
        metallicize = _get_power_amount(monster, "Metallicize")
        if metallicize > 8:
            _set_power_amount(monster, "Metallicize", metallicize - 8)
        elif metallicize > 0:
            _remove_power(monster, "Metallicize")

    def _deal_non_attack_damage_to_monster(self, monster: MonsterState, amount: int) -> int:
        dealt = _deal_damage(monster, amount)
        if dealt > 0 and int(monster.current_hp) <= 0:
            if not self._handle_special_monster_zero_hp(monster):
                self._kill_monster(monster)
        if dealt > 0:
            self._wake_lagavulin_after_hp_loss(monster)
            self._handle_guardian_hp_loss_reactions(monster, dealt)
            self._trigger_slime_split_after_hp_loss(monster)
        if int(self.player.current_hp) > 0 and not _any_monsters_alive(self.state.monsters):
            self.outcome = "VICTORY"
        return dealt

    def _setup_opening_draw(self) -> None:
        self.state.draw_pile = [_copy_card(card) for card in self.master_deck]
        relic_ids = self._relic_ids
        self._apply_emerald_elite_buff()
        if "PreservedInsect" in relic_ids and (self.state.room_type == "MonsterRoomElite" or self.elite_trigger):
            for monster in self.state.monsters:
                reduced_current = max(1, int(int(monster.max_hp) * 0.75))
                monster.current_hp = min(int(monster.current_hp), reduced_current)
        self._shuffle_cards(self.state.draw_pile, count_as_shuffle=False)
        opening_top_cards = [
            card for card in self.state.draw_pile
            if bool(
                card.get("innate")
                or card.get("bottled")
                or card.get("in_bottle_flame")
                or card.get("in_bottle_lightning")
                or card.get("in_bottle_tornado")
            )
        ]
        if opening_top_cards:
            self.state.draw_pile = [
                card for card in self.state.draw_pile
                if not bool(
                    card.get("innate")
                    or card.get("bottled")
                    or card.get("in_bottle_flame")
                    or card.get("in_bottle_lightning")
                    or card.get("in_bottle_tornado")
                )
            ] + opening_top_cards
        if "Ninja Scroll" in relic_ids:
            for shiv_index in range(3):
                self.state.hand.append(
                    _make_shiv(
                        f"ninja-scroll-shiv-{shiv_index}",
                        accuracy_amount=_get_power_amount(self.player, "Accuracy"),
                    )
                )
        self.player.block = 10 if "Anchor" in relic_ids else 0
        self.player.energy = (
            _player_base_energy(self.player, self.relics)
            + _player_bonus_energy_for_room(self.relics, self.state.room_type, elite_trigger=self.elite_trigger)
            + _player_opening_energy_bonus(self.relics)
        )
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id == "Unceasing Top":
                relic["can_draw"] = True
                relic["disabled_until_end_of_turn"] = False
        self.state.turn = 0
        self._apply_turn_start_relic_counters()
        self._apply_brimstone_turn_start()
        self.draw_cards(_player_opening_draw_count(self.player, self.relics))
        # The real first turn still fires atTurnStartPostDraw relic hooks after the opening draw.
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id == "Pocketwatch":
                relic["first_turn"] = False
                relic["counter"] = 0
            elif relic_id == "Art of War":
                relic["first_turn"] = False
        if "Mark of Pain" in relic_ids:
            for idx in range(2):
                self._add_card_to_draw_pile_random_spot(make_card("Wound", uuid=f"mark-of-pain-wound-{idx}"))
        self._trigger_warped_tongs_post_draw()
        if "Gambling Chip" in relic_ids and self.state.hand:
            self.pending_card_select = {
                "mode": "GAMBLING_CHIP",
                "cards": [dict(card) for card in self.state.hand],
                "source_indexes": list(range(len(self.state.hand))),
                "num_cards": 99,
                "any_number": False,
                "can_pick_zero": True,
                "selected_source_indexes": [],
                "selected_cards": [],
                "confirm_up": False,
            }
        self._apply_turn_start_damage_relics()
        self._update_bloodied_relics(remove_when_not_bloodied=False)
        self._update_monster_intents()
        _refresh_card_flags(self.state.hand, self.player)

    def draw_cards(self, count: int) -> None:
        if _get_power_amount(self.player, "No Draw") != 0:
            return
        drew_any = False
        for _ in range(int(count)):
            if len(self.state.hand) >= 10:
                break
            if not self.state.draw_pile and self.state.discard_pile:
                self.state.draw_pile = list(self.state.discard_pile)
                self.state.discard_pile = []
                self._shuffle_cards(self.state.draw_pile)
            if not self.state.draw_pile:
                if drew_any:
                    # StS queues an EmptyDeckShuffleAction when a DrawCardAction
                    # asks for more cards than remain. That action consumes
                    # shuffle RNG even if the discard pile is empty.
                    self.randoms.stream("shuffle").random_long()
                break
            drawn = self.state.draw_pile.pop()
            drew_any = True
            if _get_power_amount(self.player, "Confusion") != 0 and int(drawn.get("cost", -99)) >= 0:
                new_cost = int(self.randoms.stream("card_random").random(3))
                drawn["cost_for_turn"] = new_cost
                drawn["free_to_play_once"] = False
            self.state.hand.append(drawn)
            if str(drawn.get("card_id") or "") == "Void":
                self.player.energy = max(0, int(self.player.energy) - 1)
            if str(drawn.get("type") or "") in {"STATUS", "CURSE"}:
                fire_breathing = _get_power_amount(self.player, "Fire Breathing")
                if fire_breathing > 0:
                    if self._defer_fire_breathing_damage_depth > 0:
                        self._pending_fire_breathing_damage.append(int(fire_breathing))
                    else:
                        self._deal_fire_breathing_damage(fire_breathing)
                    if self.outcome == "VICTORY":
                        break
            if str(drawn.get("type") or "") == "STATUS":
                evolve = _get_power_amount(self.player, "Evolve")
                if evolve > 0:
                    self.draw_cards(evolve)

    def legal_actions(self) -> list[dict[str, Any]]:
        if self.outcome != "UNDECIDED":
            return []
        if self.pending_card_select is not None:
            pending_mode = str(self.pending_card_select.get("mode") or "").upper()
            if bool(self.pending_card_select.get("confirm_up", False)) and pending_mode not in ANY_NUMBER_CARD_SELECT_MODES:
                return [
                    {
                        "kind": "confirm",
                        "name": "CONFIRM",
                        "mode": self.pending_card_select.get("mode"),
                        "choice_index": 0,
                    }
                ]
            actions: list[dict[str, Any]] = []
            selected_source_indexes = {
                int(index) for index in list(self.pending_card_select.get("selected_source_indexes") or [])
            }
            for choice_index, card in enumerate(list(self.pending_card_select.get("cards") or [])):
                if pending_mode in {"DISCOVERY", "NILRYS_CODEX"}:
                    actions.append(
                        {
                            "kind": "card_reward",
                            "mode": pending_mode,
                            "name": card.get("name"),
                            "card_id": card.get("card_id"),
                            "type": card.get("type"),
                            "rarity": card.get("rarity"),
                            "upgrades": int(card.get("upgrades") or 0),
                            "cost": card.get("cost"),
                            "base_cost": card.get("base_cost"),
                            "cost_for_turn": card.get("cost_for_turn"),
                            "choice_index": choice_index,
                            "card_index": choice_index,
                            "reward_index": 0,
                            "card": dict(card),
                        }
                    )
                    continue
                source_indexes = list(self.pending_card_select.get("source_indexes") or [])
                target_index = source_indexes[choice_index] if choice_index < len(source_indexes) else choice_index
                if pending_mode in ANY_NUMBER_CARD_SELECT_MODES and int(target_index) in selected_source_indexes:
                    continue
                action_choice_index = len(actions) if pending_mode in ANY_NUMBER_CARD_SELECT_MODES else choice_index
                actions.append(
                    {
                        "kind": "card_select",
                        "mode": self.pending_card_select.get("mode"),
                        "name": card.get("name"),
                        "card_id": card.get("card_id"),
                        "type": card.get("type"),
                        "rarity": card.get("rarity"),
                        "upgrades": int(card.get("upgrades") or 0),
                        "cost": card.get("cost"),
                        "base_cost": card.get("base_cost"),
                        "cost_for_turn": card.get("cost_for_turn"),
                        "choice_index": action_choice_index,
                        "target_index": int(target_index),
                        "card": dict(card),
                    }
                )
            if pending_mode in ANY_NUMBER_CARD_SELECT_MODES:
                actions.append(
                    {
                        "kind": "confirm",
                        "name": "CONFIRM",
                        "mode": self.pending_card_select.get("mode"),
                        "choice_index": len(actions),
                    }
                )
            if pending_mode in {"DISCOVERY", "NILRYS_CODEX"} and bool(self.pending_card_select.get("can_skip", False)):
                actions.append(
                    {
                        "kind": "skip",
                        "name": "SKIP",
                        "mode": pending_mode,
                        "choice_index": len(actions),
                    }
                )
            return actions
        _refresh_card_flags(self.state.hand, self.player)
        actions: list[dict[str, Any]] = []
        actions.extend(self._legal_potion_actions())
        relic_ids = self._relic_ids
        normality_locked = self.cards_played_this_turn >= 3 and any(card.get("card_id") == "Normality" for card in self.state.hand)
        velvet_counter = max(
            (
                int(relic.get("counter", 0) or 0)
                for relic in self.relics
                if str(relic.get("relic_id") or relic.get("id") or "") == "Velvet Choker"
            ),
            default=0,
        )
        velvet_locked = "Velvet Choker" in relic_ids and (velvet_counter >= 6 or self.cards_played_this_turn >= 6)
        for index, card in enumerate(self.state.hand):
            if normality_locked or velvet_locked:
                continue
            if card.get("type") == "CURSE" and "Blue Candle" in relic_ids and card.get("card_id") != "Necronomicurse":
                actions.append(
                    {
                        "kind": "card",
                        "name": card["name"],
                        "card_id": card["card_id"],
                        "card_index": index,
                        "source_index": index,
                        "requires_target": False,
                    }
                )
                continue
            if card.get("type") == "STATUS" and "Medical Kit" in relic_ids:
                actions.append(
                    {
                        "kind": "card",
                        "name": card["name"],
                        "card_id": card["card_id"],
                        "card_index": index,
                        "source_index": index,
                        "requires_target": False,
                    }
                )
                continue
            if not card.get("is_playable"):
                continue
            if card.get("card_id") == "Clash" and any(hand_card.get("type") != "ATTACK" for hand_card in self.state.hand if hand_card is not card):
                continue
            if card.get("card_id") == "Secret Technique" and not any(draw_card.get("type") == "SKILL" for draw_card in self.state.draw_pile):
                continue
            if card.get("card_id") == "Secret Weapon" and not any(draw_card.get("type") == "ATTACK" for draw_card in self.state.draw_pile):
                continue
            if card.get("has_target"):
                for target_index, monster in enumerate(self.state.monsters):
                    if not _alive(monster):
                        continue
                    actions.append(
                        {
                            "kind": "card",
                            "name": card["name"],
                            "card_id": card["card_id"],
                            "card_index": index,
                            "source_index": index,
                            "target_index": target_index,
                            "model_target_index": target_index,
                            "requires_target": True,
                        }
                    )
            else:
                actions.append(
                    {
                        "kind": "card",
                        "name": card["name"],
                        "card_id": card["card_id"],
                        "card_index": index,
                        "source_index": index,
                        "requires_target": False,
                    }
                )
        actions.append({"kind": "end", "name": "END_TURN", "action_index": 0})
        return actions

    def step(self, action: dict[str, Any]) -> CombatStepResult:
        if self.outcome != "UNDECIDED":
            if self.outcome == "VICTORY":
                self._handle_victory_relics_once()
            return CombatStepResult(self.outcome)
        if self.pending_card_select is not None:
            pending_mode = str(self.pending_card_select.get("mode") or "").upper()
            allowed_kinds = {"card_reward", "card_select", "skip"} if pending_mode == "DISCOVERY" else {"card_select"}
            if pending_mode == "NILRYS_CODEX":
                allowed_kinds = {"card_reward", "skip"}
            elif pending_mode in ANY_NUMBER_CARD_SELECT_MODES or bool(self.pending_card_select.get("confirm_up", False)):
                allowed_kinds = {"card_select", "confirm"}
            if action.get("kind") not in allowed_kinds:
                raise NotImplementedError(f"Unsupported combat card select action: {action}")
            self._resolve_pending_card_select(action)
        elif action.get("kind") == "end":
            self._end_turn()
        elif action.get("kind") == "card":
            self._play_card(action)
        elif action.get("kind") == "potion":
            self._use_potion(action)
        else:
            raise NotImplementedError(f"Unsupported combat action: {action}")
        self._update_bloodied_relics(remove_when_not_bloodied=False)
        if self.outcome == "VICTORY":
            self._handle_victory_relics_once()
        if self.outcome == "UNDECIDED":
            self._update_monster_intents()
        if bool(getattr(self, "_teacher_fast_step_refresh", False)):
            _refresh_card_flags(self.state.hand, self.player)
        else:
            self._refresh_all_card_flags()
        return CombatStepResult(self.outcome)

    def _handle_victory_relics_once(self) -> None:
        if self._victory_relics_handled:
            return
        self._victory_relics_handled = True
        self._handle_victory_relics()

    def _legal_potion_actions(self) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for potion_index, potion in enumerate(list(self.potions or [])):
            potion_id = _potion_id(potion)
            if not _is_usable_potion(potion) or potion_id not in SUPPORTED_COMBAT_POTION_IDS:
                continue
            base_action = {
                "kind": "potion",
                "name": potion.get("name") or potion_id,
                "potion_id": potion_id,
                "potion_index": potion_index,
                "requires_target": potion_id in TARGETED_COMBAT_POTION_IDS,
            }
            if potion_id in TARGETED_COMBAT_POTION_IDS:
                for target_index, monster in enumerate(self.state.monsters):
                    if _alive(monster):
                        actions.append({**base_action, "target_index": target_index, "model_target_index": target_index})
            else:
                actions.append(base_action)
        return actions

    def _potion_multiplier(self) -> int:
        return 2 if "Sacred Bark" in self._relic_ids else 1

    def _consume_potion_slot(self, potion_index: int) -> None:
        if not (0 <= potion_index < len(self.potions)):
            raise IndexError(potion_index)
        self.potions[potion_index] = _make_potion_slot(potion_index)

    def _heal_from_toy_ornithopter(self) -> None:
        if "Toy Ornithopter" in self._relic_ids:
            self._heal_player(5)

    def _defer_or_apply_toy_ornithopter_heal_after_potion(self, potion_id: str) -> None:
        if "Toy Ornithopter" not in self._relic_ids:
            return
        if self.pending_card_select is not None and str(self.pending_card_select.get("potion_id") or "") == str(potion_id):
            self.pending_card_select["deferred_toy_ornithopter_heal"] = (
                int(self.pending_card_select.get("deferred_toy_ornithopter_heal", 0) or 0) + 5
            )
            return
        self._heal_player(5)

    def _flush_deferred_toy_ornithopter_heal(self, pending: dict[str, Any]) -> None:
        heal_amount = int(pending.pop("deferred_toy_ornithopter_heal", 0) or 0)
        if heal_amount > 0:
            self._heal_player(heal_amount)

    def _use_potion(self, action: dict[str, Any]) -> None:
        potion_index = int(action.get("potion_index", -1))
        if not (0 <= potion_index < len(self.potions)):
            raise IndexError(potion_index)
        potion = self.potions[potion_index]
        potion_id = _potion_id(potion)
        if not _is_usable_potion(potion) or potion_id not in SUPPORTED_COMBAT_POTION_IDS:
            raise NotImplementedError(f"Unsupported combat potion action: {action}")
        target = None
        target_index = action.get("target_index")
        if target_index is not None and 0 <= int(target_index) < len(self.state.monsters):
            target = self.state.monsters[int(target_index)]
        multiplier = self._potion_multiplier()
        self._consume_potion_slot(potion_index)

        if potion_id == "Fire Potion":
            if target is not None:
                self._deal_non_attack_damage_to_monster(target, 20 * multiplier)
        elif potion_id == "Explosive Potion":
            for monster in list(self.state.monsters):
                if _alive(monster):
                    self._deal_non_attack_damage_to_monster(monster, 10 * multiplier)
                    if self.outcome == "VICTORY":
                        break
        elif potion_id == "Weak Potion":
            if target is not None:
                self._apply_monster_debuff(target, "Weak", 3 * multiplier)
        elif potion_id == "FearPotion":
            if target is not None:
                self._apply_monster_debuff(target, "Vulnerable", 3 * multiplier)
        elif potion_id == "Strength Potion":
            _add_power(self.player, "Strength", 2 * multiplier)
        elif potion_id == "Dexterity Potion":
            _add_power(self.player, "Dexterity", 2 * multiplier)
        elif potion_id == "Block Potion":
            self._gain_player_block(12 * multiplier)
        elif potion_id == "Energy Potion":
            self.player.energy += 2 * multiplier
        elif potion_id == "ElixirPotion":
            if self.state.hand:
                self.pending_card_select = {
                    "mode": "ELIXIR",
                    "potion_id": potion_id,
                    "cards": [dict(card) for card in self.state.hand],
                    "source_indexes": list(range(len(self.state.hand))),
                    "num_cards": 99,
                    "any_number": True,
                    "can_pick_zero": True,
                    "selected_source_indexes": [],
                    "selected_cards": [],
                    "confirm_up": False,
                }
        elif potion_id == "Swift Potion":
            self.draw_cards(3 * multiplier)
        elif potion_id == "BloodPotion":
            self._heal_player(max(1, int(int(self.player.max_hp) * 0.2 * multiplier)))
        elif potion_id == "SteroidPotion":
            _add_power(self.player, "Strength", 5 * multiplier)
            self._apply_player_temporary_strength_loss(5 * multiplier)
        elif potion_id == "SpeedPotion":
            _add_power(self.player, "Dexterity", 5 * multiplier)
            artifact = _get_power_amount(self.player, "Artifact")
            if artifact > 0:
                _set_power_amount(self.player, "Artifact", artifact - 1)
            else:
                _add_power(self.player, "DexLoss", 5 * multiplier)
        elif potion_id == "Ancient Potion":
            _add_power(self.player, "Artifact", 1 * multiplier)
        elif potion_id == "BlessingOfTheForge":
            self.state.hand = [upgrade_card(card) if can_upgrade_card(card) else card for card in self.state.hand]
        elif potion_id in COMBAT_POTION_DISCOVERY_TYPES:
            wanted_type = COMBAT_POTION_DISCOVERY_TYPES[potion_id]
            generated_cards = self._make_random_card_choices(
                predicate=lambda _cid, card_def, wanted_type=wanted_type: card_def.type == wanted_type,
                uuid_prefix=f"potion-{potion_id}",
                count=3,
                source_scope="COLORED",
            )
            if generated_cards:
                self.pending_card_select = {
                    "mode": "DISCOVERY",
                    "potion_id": potion_id,
                    "cards": generated_cards,
                    "num_cards": 1,
                    "copies": 1,
                    "can_skip": True,
                }
        elif potion_id == "ColorlessPotion":
            generated_cards = self._make_random_card_choices(
                predicate=lambda _cid, card_def: card_def.type not in {"CURSE", "STATUS"},
                uuid_prefix="potion-colorless",
                count=3,
                source_scope="COLORLESS",
            )
            if generated_cards:
                self.pending_card_select = {
                    "mode": "DISCOVERY",
                    "potion_id": potion_id,
                    "cards": generated_cards,
                    "num_cards": 1,
                    "copies": 1,
                    "can_skip": False,
                }
        elif potion_id == "DuplicationPotion":
            _add_power(self.player, "DuplicationPower", 1 * multiplier)
        elif potion_id == "EssenceOfSteel":
            _add_power(self.player, "Plated Armor", 4 * multiplier)
        elif potion_id == "Regen Potion":
            _add_power(self.player, "Regeneration", 5 * multiplier)
        elif potion_id == "LiquidBronze":
            _add_power(self.player, "Thorns", 3 * multiplier)
        elif potion_id == "EntropicBrew":
            generated_potions = [
                roll_random_potion(self.randoms, player_class=self.player_class, limited=True)
                for _ in range(len(self.potions or []))
            ]
            generated_index = 0
            for index, current in enumerate(list(self.potions)):
                if not _is_usable_potion(current) and generated_index < len(generated_potions):
                    self.potions[index] = generated_potions[generated_index]
                    generated_index += 1
        elif potion_id == "GamblersBrew":
            if self.state.hand:
                self.pending_card_select = {
                    "mode": "GAMBLING_CHIP",
                    "potion_id": potion_id,
                    "cards": [dict(card) for card in self.state.hand],
                    "source_indexes": list(range(len(self.state.hand))),
                    "num_cards": 99,
                    # CommMod cannot surface HandCardSelectScreen's private anyNumber flag
                    # for Gambler's Brew; keep behavior keyed by mode, but serialize false.
                    "any_number": False,
                    "can_pick_zero": True,
                    "selected_source_indexes": [],
                    "selected_cards": [],
                    "confirm_up": False,
                }
        elif potion_id == "Fruit Juice":
            self.player.max_hp += 5 * multiplier
            self.player.current_hp += 5 * multiplier
        elif potion_id == "HeartOfIron":
            _add_power(self.player, "Metallicize", 6 * multiplier)
        elif potion_id == "CultistPotion":
            _add_power(self.player, "Ritual", 1 * multiplier)
        elif potion_id == "SneckoOil":
            self.draw_cards(5 * multiplier)
            for card in self.state.hand:
                try:
                    base_cost = int(card.get("cost"))
                except (TypeError, ValueError):
                    base_cost = -99
                if base_cost >= 0:
                    new_cost = int(self.randoms.stream("card_random").random(3))
                    if base_cost != new_cost:
                        # STS RandomizeHandCostAction writes both cost and
                        # costForTurn, not just the temporary visible cost.
                        card["cost"] = new_cost
                        card["cost_for_turn"] = new_cost
                        card["free_to_play_once"] = False
                        card["is_cost_modified"] = True
        elif potion_id == "LiquidMemories":
            if self.state.discard_pile:
                if len(self.state.discard_pile) <= 1:
                    recalled = self.state.discard_pile.pop(0)
                    _set_cost_for_turn_like_sts(recalled, 0)
                    recalled["_reset_cost_for_turn_after_play"] = True
                    if len(self.state.hand) < 10:
                        self.state.hand.append(recalled)
                    else:
                        _reset_card_for_pile(recalled)
                        self.state.discard_pile.append(recalled)
                else:
                    self.pending_card_select = {
                        "mode": "LIQUID_MEMORIES",
                        "potion_id": potion_id,
                        "cards": [dict(card) for card in self.state.discard_pile],
                        "source_indexes": list(range(len(self.state.discard_pile))),
                        "num_cards": 1,
                    }
        elif potion_id == "DistilledChaos":
            targets: list[MonsterState | None] = []
            for _ in range(3 * multiplier):
                alive = [monster for monster in self.state.monsters if _alive(monster)]
                if alive:
                    targets.append(alive[int(self.randoms.stream("card_random").random(0, len(alive) - 1))])
                else:
                    targets.append(None)
            queued_cards: list[tuple[dict[str, Any], MonsterState | None]] = []
            for forced_target in targets:
                if not self.state.draw_pile and self.state.discard_pile:
                    self.state.draw_pile = list(self.state.discard_pile)
                    self.state.discard_pile = []
                    self._shuffle_cards(self.state.draw_pile)
                if not self.state.draw_pile:
                    break
                top_card = self.state.draw_pile.pop()
                queued_cards.append((top_card, forced_target))
            for top_card, forced_target in queued_cards:
                if self.outcome != "UNDECIDED" or self.pending_card_select is not None:
                    break
                top_card["free_to_play_once"] = True
                top_card["_exclude_self_from_perfected_strike_count"] = True
                self._resolve_generated_card_play(top_card, forced_target=forced_target, target_was_preselected=True)
        else:
            raise NotImplementedError(f"Unsupported combat potion effect: {potion_id}")

        self._defer_or_apply_toy_ornithopter_heal_after_potion(potion_id)
        if int(self.player.current_hp) > 0 and not _any_monsters_alive(self.state.monsters):
            self.outcome = "VICTORY"

    def _refresh_all_card_flags(self) -> None:
        for pile in (self.state.hand, self.state.draw_pile, self.state.discard_pile, self.state.exhaust_pile):
            _refresh_card_flags(pile, self.player)

    def _resolve_pending_card_select(self, action: dict[str, Any]) -> None:
        pending = self.pending_card_select or {}
        source_indexes = list(pending.get("source_indexes") or [])
        choice_index = action.get("choice_index")
        source_index = action.get("target_index")
        if source_index is None and choice_index is not None and 0 <= int(choice_index) < len(source_indexes):
            source_index = source_indexes[int(choice_index)]
        if source_index is None:
            source_index = 0
        source_index = int(source_index)
        pending_mode = str(pending.get("mode") or "").upper()
        if pending_mode == "DISCOVERY":
            if str(action.get("kind") or "").lower() == "skip":
                self._burn_discovery_choice_regeneration(pending)
                self._move_pending_played_cards(pending)
                self.pending_card_select = None
                self._handle_pending_post_card_select_effects(pending)
                self._maybe_trigger_unceasing_top()
                return
            cards = list(pending.get("cards") or [])
            selected_index = action.get("card_index")
            if selected_index is None:
                selected_index = action.get("choice_index")
            if selected_index is None:
                selected_index = source_index
            selected_card: dict[str, Any] | None = None
            if selected_index is not None and 0 <= int(selected_index) < len(cards):
                selected_card = cards[int(selected_index)]
            if selected_card is None:
                wanted_card = action.get("card") if isinstance(action.get("card"), dict) else {}
                wanted_id = wanted_card.get("card_id") or action.get("card_id")
                for card in cards:
                    if wanted_id is not None and str(card.get("card_id") or "") == str(wanted_id):
                        selected_card = card
                        break
            self._burn_discovery_choice_regeneration(pending)
            if selected_card is not None:
                copies = max(1, int(pending.get("copies") or 1))
                for copy_index in range(copies):
                    generated = _copy_card(selected_card)
                    generated["uuid"] = f"discovery-choice-{copy_index}-{generated.get('card_id')}"
                    _set_cost_for_turn_like_sts(generated, 0)
                    generated["_reset_cost_for_turn_after_play"] = True
                    if _has_power(self.player, "Corruption") and str(generated.get("type") or "") == "SKILL":
                        generated["cost_for_turn"] = -9
                    if len(self.state.hand) + 1 <= 10:
                        self.state.hand.append(generated)
                    else:
                        _reset_card_for_pile(generated)
                        self.state.discard_pile.append(generated)
            self._move_pending_played_cards(pending)
            self.pending_card_select = None
            self._handle_pending_post_card_select_effects(pending)
            self._maybe_trigger_unceasing_top()
            return
        if pending_mode == "NILRYS_CODEX":
            cards = list(pending.get("cards") or [])
            selected_index = action.get("card_index")
            if selected_index is None:
                selected_index = action.get("choice_index")
            selected_card: dict[str, Any] | None = None
            if action.get("kind") != "skip" and selected_index is not None and 0 <= int(selected_index) < len(cards):
                selected_card = cards[int(selected_index)]
            if selected_card is None and action.get("kind") != "skip":
                wanted_card = action.get("card") if isinstance(action.get("card"), dict) else {}
                wanted_id = wanted_card.get("card_id") or action.get("card_id")
                for card in cards:
                    if wanted_id is not None and str(card.get("card_id") or "") == str(wanted_id):
                        selected_card = card
                        break
            if selected_card is not None:
                generated = _copy_card(selected_card)
                generated["uuid"] = f"nilrys-codex-choice-{generated.get('card_id')}"
                self._add_card_to_draw_pile_random_spot(generated)
            resume_end_turn = bool(pending.get("resume_end_turn", False))
            self.pending_card_select = None
            if resume_end_turn:
                self._end_turn()
            return
        if pending_mode in {"SECRET_TECHNIQUE", "SECRET_WEAPON"}:
            selected_index = source_index if 0 <= source_index < len(self.state.draw_pile) else None
            if selected_index is None:
                wanted_card = action.get("card") if isinstance(action.get("card"), dict) else {}
                wanted_id = wanted_card.get("card_id") or action.get("card_id")
                for index, draw_card in enumerate(self.state.draw_pile):
                    if wanted_id is not None and str(draw_card.get("card_id") or "") == str(wanted_id):
                        selected_index = index
                        break
            if selected_index is not None and 0 <= int(selected_index) < len(self.state.draw_pile):
                selected_card = self.state.draw_pile.pop(int(selected_index))
                if len(self.state.hand) < 10:
                    self.state.hand.append(selected_card)
                else:
                    _clear_play_once_flags_for_pile(selected_card)
                    self.state.discard_pile.append(selected_card)
            self._move_pending_played_cards(pending)
            self.pending_card_select = None
            self._handle_pending_post_card_select_effects(pending)
            self._maybe_trigger_unceasing_top()
            return
        if pending_mode == "GAMBLING_CHIP":
            if str(action.get("kind") or "") == "confirm":
                selected_indexes = [int(index) for index in list(pending.get("selected_source_indexes") or [])]
                discarded_cards = [
                    self.state.hand[index]
                    for index in selected_indexes
                    if 0 <= index < len(self.state.hand)
                ]
                discard_count = 0
                for selected_index in sorted(set(selected_indexes), reverse=True):
                    if 0 <= selected_index < len(self.state.hand):
                        self.state.hand.pop(selected_index)
                for discarded in discarded_cards:
                    if discarded is not None:
                        _reset_card_for_pile(discarded)
                        self.state.discard_pile.append(discarded)
                        discard_count += 1
                self.pending_card_select = None
                self.state.cards_discarded_this_turn += discard_count
                if discard_count > 0:
                    self.draw_cards(discard_count)
                self._flush_deferred_toy_ornithopter_heal(pending)
                self._maybe_trigger_unceasing_top()
                return
            selected_indexes = [int(index) for index in list(pending.get("selected_source_indexes") or [])]
            if 0 <= source_index < len(self.state.hand) and source_index not in selected_indexes:
                selected_indexes.append(source_index)
            pending["selected_source_indexes"] = selected_indexes
            pending["selected_cards"] = [
                dict(self.state.hand[index])
                for index in selected_indexes
                if 0 <= index < len(self.state.hand)
            ]
            # CommMod reports Gambler's Brew's hand-select confirm flag as down
            # while selected cards are staged off-hand; keep confirm legal but
            # mirror the visible flag.
            pending["confirm_up"] = False
            self.pending_card_select = pending
            return
        if pending_mode == "ELIXIR":
            if str(action.get("kind") or "") == "confirm":
                selected_indexes = sorted(
                    {int(index) for index in list(pending.get("selected_source_indexes") or [])},
                    reverse=True,
                )
                for selected_index in selected_indexes:
                    if 0 <= selected_index < len(self.state.hand):
                        exhausted = self.state.hand.pop(selected_index)
                        self._exhaust_card(exhausted)
                self.pending_card_select = None
                self._flush_deferred_toy_ornithopter_heal(pending)
                return
            selected_indexes = {
                int(index) for index in list(pending.get("selected_source_indexes") or [])
            }
            if 0 <= source_index < len(self.state.hand):
                selected_indexes.add(source_index)
            pending["selected_source_indexes"] = sorted(selected_indexes)
            pending["selected_cards"] = [
                dict(self.state.hand[index])
                for index in sorted(selected_indexes)
                if 0 <= index < len(self.state.hand)
            ]
            pending["confirm_up"] = True
            self.pending_card_select = pending
            return
        if pending_mode == "PREPARED":
            selected_index = source_index if 0 <= source_index < len(self.state.hand) else None
            if selected_index is None:
                wanted_card = action.get("card") if isinstance(action.get("card"), dict) else {}
                wanted_id = wanted_card.get("card_id") or action.get("card_id")
                for index, hand_card in enumerate(self.state.hand):
                    if wanted_id is not None and str(hand_card.get("card_id") or "") == str(wanted_id):
                        selected_index = index
                        break
            remaining = max(0, int(pending.get("remaining") or pending.get("num_cards") or 1))
            if selected_index is not None and 0 <= selected_index < len(self.state.hand):
                discarded = self.state.hand.pop(selected_index)
                _clear_play_once_flags_for_pile(discarded)
                self.state.discard_pile.append(discarded)
                remaining -= 1
            if remaining <= 0 or not self.state.hand:
                self._move_pending_played_cards(pending)
                self.pending_card_select = None
                self._handle_pending_post_card_select_effects(pending)
                self._maybe_trigger_unceasing_top()
                return
            if len(self.state.hand) <= remaining:
                while self.state.hand:
                    discarded = self.state.hand.pop(0)
                    _clear_play_once_flags_for_pile(discarded)
                    self.state.discard_pile.append(discarded)
                self._move_pending_played_cards(pending)
                self.pending_card_select = None
                self._handle_pending_post_card_select_effects(pending)
                self._maybe_trigger_unceasing_top()
                return
            pending["remaining"] = remaining
            pending["num_cards"] = remaining
            pending["cards"] = [dict(hand_card) for hand_card in self.state.hand]
            pending["source_indexes"] = list(range(len(self.state.hand)))
            self.pending_card_select = pending
            return
        if pending.get("mode") == "DUAL_WIELD":
            selected_card: dict[str, Any] | None = None
            selected_index: int | None = None
            if 0 <= source_index < len(self.state.hand):
                selected_card = self.state.hand[source_index]
                selected_index = source_index
            if selected_card is None:
                wanted_card = action.get("card") if isinstance(action.get("card"), dict) else {}
                wanted_id = wanted_card.get("card_id") or action.get("card_id")
                for index, hand_card in enumerate(self.state.hand):
                    if wanted_id is not None and str(hand_card.get("card_id") or "") == str(wanted_id):
                        selected_card = hand_card
                        selected_index = index
                        break
            if selected_card is not None and selected_index is not None:
                copies = int(pending.get("copies") or 1)
                eligible_indexes = {int(index) for index in source_indexes}
                remaining_eligible = [
                    hand_card
                    for index, hand_card in enumerate(self.state.hand)
                    if index in eligible_indexes and index != selected_index
                ]
                cannot_duplicate = [
                    hand_card
                    for index, hand_card in enumerate(self.state.hand)
                    if index not in eligible_indexes
                ]
                self.state.hand = remaining_eligible + cannot_duplicate
                for copy_index in range(copies + 1):
                    self._add_temp_card_to_hand_or_discard(
                        _copy_card({**selected_card, "uuid": f"dual-wield-copy-{copy_index}-{selected_card.get('card_id')}"}),
                        reset_for_discard=False,
                    )
            self._move_pending_played_cards(pending)
            self.pending_card_select = None
            self._handle_pending_post_card_select_effects(pending)
            return
        if pending.get("mode") == "ARMAMENTS":
            selectable_count = int(pending.get("selectable_count") or len(self.state.hand))
            selectable_hand = list(self.state.hand[:selectable_count])
            appended_during_select = list(self.state.hand[selectable_count:])
            selected_index: int | None = source_index if 0 <= source_index < len(selectable_hand) else None
            if selected_index is None:
                wanted_card = action.get("card") if isinstance(action.get("card"), dict) else {}
                wanted_id = wanted_card.get("card_id") or action.get("card_id")
                for index, hand_card in enumerate(selectable_hand):
                    if wanted_id is not None and str(hand_card.get("card_id") or "") == str(wanted_id):
                        selected_index = index
                        break
            if selected_index is not None:
                selected_card = selectable_hand.pop(selected_index)
                selectable_hand.append(upgrade_card(selected_card))
            self.state.hand = (
                selectable_hand
                + list(pending.get("cannot_upgrade_cards") or [])
                + appended_during_select
            )
            self._move_pending_played_cards(pending)
            self.pending_card_select = None
            self._handle_pending_post_card_select_effects(pending)
            return
        if pending.get("mode") in {"BURNING_PACT", "TRUE_GRIT"}:
            if pending.get("mode") == "TRUE_GRIT" and str(action.get("kind") or "") != "confirm" and not bool(pending.get("confirm_up", False)):
                selected_index = source_index if 0 <= source_index < len(self.state.hand) else None
                if selected_index is None:
                    wanted_card = action.get("card") if isinstance(action.get("card"), dict) else {}
                    wanted_id = wanted_card.get("card_id") or action.get("card_id")
                    for index, hand_card in enumerate(self.state.hand):
                        if wanted_id is not None and str(hand_card.get("card_id") or "") == str(wanted_id):
                            selected_index = index
                            break
                if selected_index is not None:
                    pending["selected_source_indexes"] = [int(selected_index)]
                    pending["selected_cards"] = [dict(self.state.hand[int(selected_index)])]
                pending["confirm_up"] = True
                self.pending_card_select = pending
                return
            selected_indexes = list(pending.get("selected_source_indexes") or [])
            if selected_indexes:
                source_index = int(selected_indexes[0])
            selected_index = source_index if 0 <= source_index < len(self.state.hand) else None
            deferred_dead_branch_cards: list[dict[str, Any]] | None = None
            defer_feel_no_pain = bool(pending.get("deferred_juggernaut_damage"))
            if selected_index is None:
                wanted_card = action.get("card") if isinstance(action.get("card"), dict) else {}
                wanted_id = wanted_card.get("card_id") or action.get("card_id")
                for index, hand_card in enumerate(self.state.hand):
                    if wanted_id is not None and str(hand_card.get("card_id") or "") == str(wanted_id):
                        selected_index = index
                        break
            if selected_index is not None:
                exhausted = self.state.hand.pop(selected_index)
                if pending.get("mode") == "BURNING_PACT":
                    deferred_dead_branch_cards = []
                    self._defer_dark_embrace_draw_depth += 1
                    if defer_feel_no_pain:
                        self._defer_feel_no_pain_block_depth += 1
                    try:
                        self._exhaust_card(exhausted, defer_dead_branch_to=deferred_dead_branch_cards)
                    finally:
                        if defer_feel_no_pain:
                            self._defer_feel_no_pain_block_depth -= 1
                        self._defer_dark_embrace_draw_depth -= 1
                else:
                    self._exhaust_card(exhausted)
            self.draw_cards(int(pending.get("draw_count") or 0))
            if deferred_dead_branch_cards:
                self._add_temp_cards_to_hand_or_discard(deferred_dead_branch_cards)
            self._move_pending_played_cards(pending)
            self._flush_pending_dark_embrace_draws()
            self._flush_deferred_juggernaut_damage(pending)
            self._flush_pending_feel_no_pain_blocks()
            self.pending_card_select = None
            self._handle_pending_post_card_select_effects(pending)
            self._maybe_trigger_unceasing_top()
            return
        if pending.get("mode") in {"WARCRY", "PUT_ON_DECK"}:
            selected_index = source_index if 0 <= source_index < len(self.state.hand) else None
            if selected_index is None:
                wanted_card = action.get("card") if isinstance(action.get("card"), dict) else {}
                wanted_id = wanted_card.get("card_id") or action.get("card_id")
                for index, hand_card in enumerate(self.state.hand):
                    if wanted_id is not None and str(hand_card.get("card_id") or "") == str(wanted_id):
                        selected_index = index
                        break
            if selected_index is not None:
                selected = self.state.hand.pop(selected_index)
                self.state.draw_pile.append(selected)
            self._move_pending_played_cards(pending)
            self.pending_card_select = None
            self._handle_pending_post_card_select_effects(pending)
            self._maybe_trigger_unceasing_top()
            return
        if pending_mode == "LIQUID_MEMORIES":
            selected_index = source_index if 0 <= source_index < len(self.state.discard_pile) else None
            if selected_index is None:
                wanted_card = action.get("card") if isinstance(action.get("card"), dict) else {}
                wanted_id = wanted_card.get("card_id") or action.get("card_id")
                for index, discard_card in enumerate(self.state.discard_pile):
                    if wanted_id is not None and str(discard_card.get("card_id") or "") == str(wanted_id):
                        selected_index = index
                        break
            if selected_index is not None:
                recalled = self.state.discard_pile.pop(selected_index)
                _set_cost_for_turn_like_sts(recalled, 0)
                recalled["_reset_cost_for_turn_after_play"] = True
                if len(self.state.hand) < 10:
                    self.state.hand.append(recalled)
                else:
                    _reset_card_for_pile(recalled)
                    self.state.discard_pile.append(recalled)
            self.pending_card_select = None
            self._flush_deferred_toy_ornithopter_heal(pending)
            self._maybe_trigger_unceasing_top()
            return
        if pending_mode == "EXHUME":
            selected_index = source_index if 0 <= source_index < len(self.state.exhaust_pile) else None
            if selected_index is None:
                wanted_card = action.get("card") if isinstance(action.get("card"), dict) else {}
                wanted_id = wanted_card.get("card_id") or action.get("card_id")
                for index, exhausted_card in enumerate(self.state.exhaust_pile):
                    if wanted_id is not None and str(exhausted_card.get("card_id") or "") == str(wanted_id):
                        selected_index = index
                        break
            if selected_index is not None:
                self.state.hand.append(self.state.exhaust_pile.pop(selected_index))
            self._move_pending_played_cards(pending)
            self.pending_card_select = None
            self._handle_pending_post_card_select_effects(pending)
            self._maybe_trigger_unceasing_top()
            return
        if not (0 <= source_index < len(self.state.discard_pile)):
            wanted_card = action.get("card") if isinstance(action.get("card"), dict) else {}
            wanted_id = wanted_card.get("card_id") or action.get("card_id")
            for index, card in enumerate(self.state.discard_pile):
                if wanted_id is not None and str(card.get("card_id") or "") == str(wanted_id):
                    source_index = index
                    break
        if 0 <= source_index < len(self.state.discard_pile):
            self.state.draw_pile.append(self.state.discard_pile.pop(source_index))
        if pending_mode == "HEADBUTT":
            self._flush_deferred_gremlin_horn_rewards()
        self._move_pending_played_cards(pending)
        self.pending_card_select = None
        self._handle_pending_post_card_select_effects(pending)

    def _move_used_card(self, card: dict[str, Any]) -> None:
        card_object_id = id(card)
        if card_object_id in self._early_moved_used_card_ids:
            self._early_moved_used_card_ids.discard(card_object_id)
            self._monster_after_use_triggered_card_ids.discard(card_object_id)
            return
        if card["type"] == "POWER":
            card.pop("_force_exhaust_after_play", None)
            card.pop("_reset_cost_for_turn_after_play", None)
            return
        if bool(card.pop("_reset_cost_for_turn_after_play", False)):
            _reset_card_for_pile(card)
        if bool(card.pop("_force_exhaust_after_play", False)):
            self._exhaust_card(card)
            return
        card_id = str(card.get("card_id") or "")
        relic_ids = self._relic_ids
        if card.get("type") == "CURSE" and "Blue Candle" in relic_ids and card_id != "Necronomicurse":
            self._lose_player_hp(1)
        would_exhaust = self._used_card_would_exhaust(card)
        if self._resolve_strange_spoon_proc(card, would_exhaust=would_exhaust):
            _clear_play_once_flags_for_pile(card)
            self.state.discard_pile.append(card)
        elif would_exhaust:
            self._exhaust_card(card)
        else:
            _clear_play_once_flags_for_pile(card)
            self.state.discard_pile.append(card)

    def _move_used_card_and_flush_deferred_discards(self, card: dict[str, Any]) -> None:
        deferred_discards = list(card.pop("_deferred_post_move_discards", []) or [])
        deferred_exhausts = list(card.pop("_deferred_post_move_exhausts", []) or [])
        flush_after_move = str(card.get("card_id") or "") == "Second Wind"
        would_exhaust = self._used_card_would_exhaust(card)
        self._resolve_strange_spoon_proc(card, would_exhaust=would_exhaust)
        if (
            self._pending_dark_embrace_draws > 0
            and not flush_after_move
            and self._used_card_will_discard(card)
        ):
            self._flush_pending_dark_embrace_draws()
        self._move_used_card(card)
        if flush_after_move:
            self._flush_pending_dark_embrace_draws()
        if deferred_exhausts:
            self.state.exhaust_pile.extend(deferred_exhausts)
        if deferred_discards:
            self.state.discard_pile.extend(deferred_discards)

    def _used_card_will_discard(self, card: dict[str, Any]) -> bool:
        if bool(card.get("_force_exhaust_after_play", False)):
            return False
        if bool(card.get("_strange_spoon_proc", False)):
            return True
        if self._used_card_would_exhaust(card):
            return False
        return True

    def _used_card_would_exhaust(self, card: dict[str, Any]) -> bool:
        card_id = str(card.get("card_id") or "")
        relic_ids = self._relic_ids
        if card.get("type") == "CURSE" and "Blue Candle" in relic_ids and card_id != "Necronomicurse":
            return True
        if card.get("type") == "STATUS" and "Medical Kit" in relic_ids:
            return True
        if card.get("exhausts") or (card.get("type") == "SKILL" and _has_power(self.player, "Corruption")):
            return True
        return False

    def _resolve_strange_spoon_proc(self, card: dict[str, Any], *, would_exhaust: bool) -> bool:
        if "_strange_spoon_proc" in card:
            return bool(card.get("_strange_spoon_proc"))
        proc = False
        if would_exhaust and "Strange Spoon" in self._relic_ids and card.get("type") != "POWER":
            proc = bool(self.randoms.stream("card_random").random_boolean())
        card["_strange_spoon_proc"] = proc
        return proc

    def _move_pending_played_cards(self, pending: dict[str, Any]) -> None:
        self._trigger_pending_deferred_hex(pending)
        played_cards: list[dict[str, Any]] = []
        existing_cards = pending.get("played_cards")
        if isinstance(existing_cards, list):
            played_cards.extend(card for card in existing_cards if isinstance(card, dict))
        elif isinstance(pending.get("played_card"), dict):
            played_cards.append(pending["played_card"])
        for played_card in played_cards:
            self._move_used_card(played_card)

    def _trigger_pending_deferred_hex(self, pending: dict[str, Any]) -> None:
        card = pending.pop("deferred_hex_card", None)
        if isinstance(card, dict):
            self._trigger_hex_after_card_effect(card)

    def _defer_used_card_if_pending(self, card: dict[str, Any]) -> bool:
        if id(card) in self._early_moved_used_card_ids:
            return False
        if self.pending_card_select is None:
            return False
        if not bool(self.pending_card_select.get("defer_move_used_card", False)):
            return False
        played_cards = list(self.pending_card_select.get("played_cards") or [])
        existing = self.pending_card_select.get("played_card")
        if isinstance(existing, dict) and not played_cards:
            played_cards.append(existing)
        played_cards.append(card)
        self.pending_card_select["played_cards"] = played_cards
        self.pending_card_select["played_card"] = card
        return True

    def _handle_or_defer_any_card_relics(self) -> None:
        self._handle_any_card_relics(defer_ink_bottle_draw=self.pending_card_select is not None)

    def _handle_deferred_any_card_relics(self, pending: dict[str, Any]) -> None:
        draw_count = int(pending.pop("deferred_ink_bottle_draws", 0) or 0)
        if draw_count > 0:
            self.draw_cards(draw_count)

    def _handle_pending_post_card_select_effects(self, pending: dict[str, Any]) -> None:
        self._flush_deferred_toy_ornithopter_heal(pending)
        if bool(pending.pop("deferred_guardian_mode_shift", False)):
            self._flush_pending_guardian_mode_shift()
        if bool(pending.pop("deferred_flight_reduction", False)):
            self._flush_deferred_flight_reductions()
        if bool(pending.pop("deferred_plated_armor_reduction", False)):
            self._flush_pending_plated_armor_reductions()
        rage_block = int(pending.pop("deferred_rage_block", 0) or 0)
        if rage_block > 0:
            self._gain_player_block(rage_block)
        snapshot = list(pending.pop("deferred_monster_on_use_snapshot", []) or [])
        if snapshot:
            self._resolve_monster_on_use_card_power_snapshot(snapshot)
            if self.outcome == "DEFEAT":
                return
        self._flush_deferred_juggernaut_damage(pending)
        if self.outcome != "UNDECIDED":
            return
        self._flush_deferred_fire_breathing_damage(pending)
        if self.outcome != "UNDECIDED":
            return
        self._handle_deferred_any_card_relics(pending)
        self._resolve_pending_deferred_double_tap_replay(pending)

    def _deal_fire_breathing_damage(self, amount: int) -> None:
        for monster in self.state.monsters:
            if _alive(monster):
                self._deal_non_attack_damage_to_monster(monster, int(amount))

    def _flush_deferred_fire_breathing_damage(self, pending: dict[str, Any] | None = None) -> None:
        if pending is None:
            amounts = list(self._pending_fire_breathing_damage)
            self._pending_fire_breathing_damage = []
        else:
            amounts = [int(amount) for amount in list(pending.pop("deferred_fire_breathing_damage", []) or [])]
        for amount in amounts:
            self._deal_fire_breathing_damage(amount)
            if self.outcome != "UNDECIDED":
                return

    def _flush_pending_dark_embrace_draws(self) -> None:
        batches = [int(amount) for amount in list(self._pending_dark_embrace_draw_batches) if int(amount) > 0]
        draw_count = int(self._pending_dark_embrace_draws)
        self._pending_dark_embrace_draws = 0
        self._pending_dark_embrace_draw_batches = []
        if not batches and draw_count > 0:
            batches = [draw_count]
        for amount in batches:
            self.draw_cards(amount)

    def _defer_or_gain_rage_block(self, rage: int) -> None:
        if rage <= 0:
            return
        if (
            self.pending_card_select is not None
            and str(self.pending_card_select.get("mode") or "").upper() == "HEADBUTT"
            and self.outcome == "UNDECIDED"
        ):
            self.pending_card_select["deferred_rage_block"] = (
                int(self.pending_card_select.get("deferred_rage_block", 0) or 0) + int(rage)
            )
            return
        self._gain_player_block(rage)

    def _add_temp_card_to_hand_or_discard(self, card: dict[str, Any], *, reset_for_discard: bool = True) -> None:
        if len(self.state.hand) < 10:
            self.state.hand.append(card)
            return
        if reset_for_discard:
            _reset_card_for_pile(card)
        self.state.discard_pile.append(card)

    def _draw_matching_cards_from_draw_pile_to_hand(self, amount: int, card_type: str) -> None:
        tmp: list[dict[str, Any]] = []
        for draw_card in self.state.draw_pile:
            if draw_card.get("type") != card_type:
                continue
            if not tmp:
                tmp.append(draw_card)
            else:
                insert_index = int(self.randoms.stream("card_random").random(len(tmp) - 1))
                tmp.insert(insert_index, draw_card)
        for _ in range(max(0, int(amount))):
            if not tmp:
                continue
            java_shuffle_in_place(tmp, int(self.randoms.stream("shuffle").random_long()))
            picked = tmp.pop(0)
            if picked not in self.state.draw_pile:
                continue
            self.state.draw_pile.remove(picked)
            self._add_temp_card_to_hand_or_discard(picked)

    def _add_temp_cards_to_hand_or_discard(self, cards: list[dict[str, Any]], *, reset_for_discard: bool = True) -> None:
        for card in cards:
            self._add_temp_card_to_hand_or_discard(card, reset_for_discard=reset_for_discard)

    def _exhaust_card(self, card: dict[str, Any], *, defer_dead_branch_to: list[dict[str, Any]] | None = None) -> None:
        _clear_play_once_flags_for_pile(card)
        self.state.exhaust_pile.append(card)
        if str(card.get("card_id") or "") == "Necronomicurse" and any(
            str(relic.get("relic_id") or relic.get("id") or "") == "Necronomicon" for relic in self.relics
        ):
            self.state.hand.append(make_card("Necronomicurse", uuid=f"necronomicurse-copy-{len(self.state.hand)}"))
        if "Dead Branch" in self._relic_ids and _any_monsters_alive(self.state.monsters):
            generated = self._make_random_card(
                predicate=lambda _cid, card_def: card_def.type not in {"CURSE", "STATUS"},
                uuid_prefix="dead-branch",
                free_this_turn=False,
                source_scope="COLORED",
            )
            if generated is not None:
                if defer_dead_branch_to is not None:
                    defer_dead_branch_to.append(generated)
                else:
                    self._add_temp_card_to_hand_or_discard(generated)
        if "Charon's Ashes" in self._relic_ids:
            for monster in self.state.monsters:
                if _alive(monster):
                    self._deal_non_attack_damage_to_monster(monster, 3)
        dark_embrace = _get_power_amount(self.player, "Dark Embrace")
        if dark_embrace > 0:
            if self._defer_dark_embrace_draw_depth > 0:
                self._pending_dark_embrace_draws += dark_embrace
                self._pending_dark_embrace_draw_batches.append(dark_embrace)
            else:
                self.draw_cards(dark_embrace)
        feel_no_pain = _get_power_amount(self.player, "Feel No Pain")
        if feel_no_pain > 0:
            if self._defer_feel_no_pain_block_depth > 0:
                self._pending_feel_no_pain_blocks.append(int(feel_no_pain))
            else:
                self._gain_player_block(feel_no_pain)
        if str(card.get("card_id") or "") == "Sentinel":
            self.player.energy += 3 if int(card.get("upgrades") or 0) > 0 else 2
        _refresh_card_flags(self.state.hand, self.player)

    def _record_player_hp_loss(self, amount: int, *, trigger_rupture: bool = False) -> None:
        loss = max(0, int(amount))
        if loss <= 0:
            return
        # StS AbstractPlayer.damagedThisCombat counts damage events, not HP amount.
        self.player_damage_taken_this_combat += 1
        rupture = _get_power_amount(self.player, "Rupture")
        if trigger_rupture and rupture > 0:
            _add_power(self.player, "Strength", rupture)
        for pile in (self.state.hand, self.state.discard_pile, self.state.draw_pile):
            for current in pile:
                _apply_blood_for_blood_damage_cost_reduction(current)
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id == "Centennial Puzzle" and not bool(relic.get("used_this_combat")):
                relic["used_this_combat"] = True
                self.draw_cards(3)
            elif relic_id == "Self Forming Clay":
                _add_power(self.player, "NextTurnBlock", 3)
        _refresh_card_flags(self.state.hand, self.player)

    def _lose_player_hp(self, amount: int) -> None:
        self._damage_player(int(amount), damage_type="HP_LOSS", trigger_rupture=True)

    def _maybe_revive_with_lizard_tail(self) -> bool:
        if int(self.player.current_hp) > 0:
            return False
        relic_ids = self._relic_ids
        if "Mark of the Bloom" in relic_ids:
            self.outcome = "DEFEAT"
            return False
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id != "Lizard Tail":
                continue
            if bool(relic.get("used_up")) or int(relic.get("counter", -1) or -1) == -2:
                continue
            relic["counter"] = -2
            relic["used_up"] = True
            self.player.current_hp = max(1, int(self.player.max_hp) // 2)
            return True
        self.outcome = "DEFEAT"
        return False

    def _maybe_revive_with_fairy_potion(self) -> bool:
        if int(self.player.current_hp) > 0:
            return False
        if "Mark of the Bloom" in self._relic_ids:
            return False
        for index, potion in enumerate(self.potions):
            potion_id = str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "")
            if potion_id != "FairyPotion":
                continue
            self.potions[index] = _make_potion_slot(index)
            self.player.current_hp = 0
            self._heal_player(max(1, int(int(self.player.max_hp) * 0.3)))
            self._update_bloodied_relics()
            return int(self.player.current_hp) > 0
        return False

    def _apply_final_player_hp_loss(self, amount: int, *, trigger_rupture: bool = False) -> int:
        loss = max(0, int(amount))
        if loss <= 0:
            return 0
        if loss > 1 and _get_power_amount(self.player, "Intangible") > 0:
            loss = 1
        if "TungstenRod" in self._relic_ids:
            loss = max(0, loss - 1)
        if loss <= 0:
            return 0
        self.player.current_hp = max(0, int(self.player.current_hp) - loss)
        self._record_player_hp_loss(loss, trigger_rupture=trigger_rupture)
        if int(self.player.current_hp) > 0:
            self._update_bloodied_relics(remove_when_not_bloodied=False)
        if self.player.current_hp <= 0:
            revived = self._maybe_revive_with_fairy_potion()
            if not revived:
                revived = self._maybe_revive_with_lizard_tail()
            if not revived:
                self.outcome = "DEFEAT"
        if int(self.player.current_hp) > 0 and not _any_monsters_alive(self.state.monsters):
            self.outcome = "VICTORY"
        return loss

    def _heal_player(self, amount: int) -> int:
        heal = max(0, int(amount))
        if heal <= 0:
            return 0
        relic_ids = self._relic_ids
        if "Mark of the Bloom" in relic_ids:
            return 0
        if "Magic Flower" in relic_ids:
            heal = _round_positive_half_up(float(heal) * 1.5)
        before_hp = int(self.player.current_hp)
        self.player.current_hp = min(int(self.player.max_hp), before_hp + heal)
        healed = int(self.player.current_hp) - before_hp
        if healed > 0:
            self._update_bloodied_relics()
        return healed

    def _retaliate_against_attacker(self, attacker: MonsterState | None, amount: int) -> int:
        if attacker is None or not _alive(attacker):
            return 0
        dealt = _deal_damage(attacker, max(0, int(amount)))
        if dealt > 0 and _alive(attacker):
            self._wake_lagavulin_after_hp_loss(attacker)
            self._handle_guardian_hp_loss_reactions(attacker, dealt)
            self._trigger_slime_split_after_hp_loss(attacker)
        if int(attacker.current_hp) <= 0:
            if not self._handle_special_monster_zero_hp(attacker):
                self._kill_monster(attacker)
            if int(self.player.current_hp) > 0 and not _any_monsters_alive(self.state.monsters):
                self.outcome = "VICTORY"
        return dealt

    def _damage_player(
        self,
        amount: int,
        *,
        attacker: MonsterState | None = None,
        damage_type: str = "NORMAL",
        apply_player_vulnerable: bool = True,
        trigger_rupture: bool = False,
        defer_player_retaliation: bool = False,
    ) -> int:
        damage = max(0, int(amount))
        if damage <= 0:
            if attacker is not None and damage_type == "NORMAL" and not defer_player_retaliation:
                self._retaliate_player_powers(attacker)
            return 0
        if damage_type == "NORMAL" and apply_player_vulnerable:
            vulnerable = _get_power_amount(self.player, "Vulnerable") > 0
            vulnerable_multiplier = 1.25 if vulnerable and "Odd Mushroom" in self._relic_ids else 1.5
            damage = _damage_multiplier(
                damage,
                vulnerable=vulnerable,
                weak=False,
                vulnerable_multiplier=vulnerable_multiplier,
            )
        if damage > 1 and _get_power_amount(self.player, "Intangible") > 0:
            damage = 1
        if damage_type != "HP_LOSS":
            blocked = min(int(self.player.block), damage)
            if blocked:
                self.player.block -= blocked
            damage = max(0, damage - blocked)
        if damage_type == "NORMAL" and attacker is not None and 1 < damage <= 5 and "Torii" in self._relic_ids:
            damage = 1
        buffer = _get_power_amount(self.player, "Buffer")
        if damage > 0 and buffer > 0:
            _set_power_amount(self.player, "Buffer", buffer - 1)
            damage = 0
        if attacker is not None and damage_type == "NORMAL" and not defer_player_retaliation:
            self._retaliate_player_powers(attacker)
        if damage <= 0:
            if int(self.player.current_hp) > 0 and not _any_monsters_alive(self.state.monsters):
                self.outcome = "VICTORY"
            return 0
        dealt = self._apply_final_player_hp_loss(damage, trigger_rupture=trigger_rupture)
        plated = _get_power_amount(self.player, "Plated Armor")
        if dealt > 0 and damage_type not in {"HP_LOSS", "THORNS"} and attacker is not None and plated > 0:
            _set_power_amount(self.player, "Plated Armor", plated - 1)
        if (
            dealt > 0
            and damage_type == "NORMAL"
            and attacker is not None
            and _get_power_amount(attacker, "Painful Stabs") != 0
        ):
            self.state.discard_pile.append(make_card("Wound", uuid=f"painful-stabs-wound-{len(self.state.discard_pile)}"))
        return dealt

    def _retaliate_player_powers(self, attacker: MonsterState | None) -> None:
        if attacker is None or not _alive(attacker):
            return
        flame_barrier = _get_power_amount(self.player, "FlameBarrier")
        if flame_barrier > 0:
            self._retaliate_against_attacker(attacker, flame_barrier)
        thorns = _get_power_amount(self.player, "Thorns")
        if thorns > 0 and _alive(attacker):
            self._retaliate_against_attacker(attacker, thorns)

    def _monster_attack_player(
        self,
        monster: MonsterState,
        base_damage: int,
        *,
        hits: int = 1,
        defer_player_retaliation: bool = False,
    ) -> int:
        if not _alive(monster):
            return 0
        strength = _get_power_amount(monster, "Strength")
        weak = _is_weakened(monster)
        damage = self._scale_monster_attack_damage(int(base_damage) + strength, weak)
        total = 0
        self._defer_guardian_mode_shift_depth += 1
        try:
            for _ in range(max(0, int(hits))):
                if not _alive(monster):
                    break
                total += self._damage_player(
                    damage,
                    attacker=monster,
                    apply_player_vulnerable=False,
                    defer_player_retaliation=defer_player_retaliation,
                )
                if int(self.player.current_hp) <= 0:
                    break
        finally:
            self._defer_guardian_mode_shift_depth -= 1
            if self._defer_guardian_mode_shift_depth == 0:
                self._flush_pending_guardian_mode_shift()
        return total

    def _player_is_bloodied(self) -> bool:
        return int(self.player.current_hp) > 0 and int(self.player.current_hp) * 2 <= int(self.player.max_hp)

    def _update_bloodied_relics(self, *, remove_when_not_bloodied: bool = True) -> None:
        bloodied = self._player_is_bloodied()
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id != "Red Skull":
                continue
            active = bool(relic.get("is_active"))
            if bloodied and not active:
                _add_power(self.player, "Strength", 3)
                relic["is_active"] = True
            elif not bloodied and active and remove_when_not_bloodied:
                _add_power(self.player, "Strength", -3)
                relic["is_active"] = False

    def _handle_victory_relics(self) -> None:
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id == "Meat on the Bone" and self._player_is_bloodied():
                self._heal_player(12)
            elif relic_id == "Red Skull":
                relic["is_active"] = False
            elif relic_id in {"HornCleat", "CaptainsWheel"}:
                relic["counter"] = -1
                relic["grayscale"] = False
            elif relic_id in {"Kunai", "Shuriken", "Ornamental Fan", "Letter Opener"}:
                relic["counter"] = -1
            elif relic_id == "StoneCalendar":
                relic["counter"] = -1
            elif relic_id == "Velvet Choker":
                relic["counter"] = -1
            elif relic_id == "Pocketwatch":
                relic["counter"] = -1
                relic["first_turn"] = False
        self._update_bloodied_relics()

    def _advance_turn_counter_block_relics(self) -> None:
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id == "HornCleat":
                if not bool(relic.get("grayscale", False)):
                    relic["counter"] = int(relic.get("counter", 0) or 0) + 1
                if int(relic.get("counter", 0) or 0) == 2:
                    self._gain_player_block(14)
                    relic["counter"] = -1
                    relic["grayscale"] = True
            elif relic_id == "CaptainsWheel":
                if not bool(relic.get("grayscale", False)):
                    relic["counter"] = int(relic.get("counter", 0) or 0) + 1
                if int(relic.get("counter", 0) or 0) == 3:
                    self._gain_player_block(18)
                    relic["counter"] = -1
                    relic["grayscale"] = True

    def _apply_turn_start_relic_counters(self) -> None:
        self._advance_turn_counter_block_relics()
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id == "Happy Flower":
                counter = int(relic.get("counter", 0) or 0)
                counter = counter + 2 if counter == -1 else counter + 1
                if counter == 3:
                    self.player.energy += 1
                    counter = 0
                relic["counter"] = counter
            elif relic_id == "Incense Burner":
                counter = int(relic.get("counter", 0) or 0)
                counter = counter + 2 if counter == -1 else counter + 1
                if counter == 6:
                    _add_power(self.player, "Intangible", 1)
                    counter = 0
                relic["counter"] = counter
            elif relic_id == "StoneCalendar":
                relic["counter"] = int(relic.get("counter", 0) or 0) + 1

    def _apply_brimstone_turn_start(self) -> None:
        if "Brimstone" not in self._relic_ids:
            return
        _add_power(self.player, "Strength", 2)
        for monster in self.state.monsters:
            if _alive(monster):
                _add_power(monster, "Strength", 1)

    def _scale_player_attack_damage(self, amount: int, target: MonsterState | None, attacker_weak: bool) -> int:
        vulnerable = target is not None and _get_power_amount(target, "Vulnerable") > 0
        vulnerable_multiplier = (
            1.75 if vulnerable and "Paper Frog" in self._relic_ids else 1.5
        )
        scaled = float(amount)
        if self._double_attack_damage:
            scaled *= 2.0
        if attacker_weak:
            scaled *= 0.75
        if vulnerable:
            scaled *= vulnerable_multiplier
        slow = _get_power_amount(target, "Slow") if target is not None else 0
        if slow > 0:
            scaled *= 1.0 + 0.1 * slow
        return max(0, int(scaled))

    def _scale_monster_attack_damage(self, amount: int, weak: bool) -> int:
        vulnerable = _get_power_amount(self.player, "Vulnerable") > 0
        relic_ids: set[str] | frozenset[str] | None = None
        if weak or vulnerable:
            relic_ids = self._relic_ids
        weak_multiplier = 0.6 if weak and relic_ids is not None and "Paper Crane" in relic_ids else 0.75
        vulnerable_multiplier = (
            1.25 if vulnerable and relic_ids is not None and "Odd Mushroom" in relic_ids else 1.5
        )
        damage = _damage_multiplier(
            int(amount),
            vulnerable=vulnerable,
            weak=weak,
            vulnerable_multiplier=vulnerable_multiplier,
            weak_multiplier=weak_multiplier,
        )
        if damage > 1 and _get_power_amount(self.player, "Intangible") > 0:
            damage = 1
        return damage

    def _reduce_monster_plated_armor_after_hp_loss(self, monster: MonsterState) -> None:
        plated = _get_power_amount(monster, "Plated Armor")
        if plated <= 0:
            return
        next_amount = plated - 1
        _set_power_amount(monster, "Plated Armor", next_amount)
        if next_amount <= 0 and _alive(monster):
            monster.next_move = "STUNNED"
            monster.intent = "STUN"
            monster.move_adjusted_damage = 0
            monster.move_hits = 0

    def _reduce_monster_plated_armor_or_defer(self, monster: MonsterState) -> None:
        if self._defer_plated_armor_reduce_depth > 0:
            self._pending_plated_armor_reductions.append(monster)
            return
        self._reduce_monster_plated_armor_after_hp_loss(monster)

    def _flush_pending_plated_armor_reductions(self) -> None:
        pending = list(self._pending_plated_armor_reductions)
        self._pending_plated_armor_reductions.clear()
        for monster in pending:
            if _alive(monster):
                self._reduce_monster_plated_armor_after_hp_loss(monster)

    def _trigger_slime_split_after_hp_loss(self, monster: MonsterState) -> None:
        if monster.monster_id not in {"SlimeBoss", "AcidSlime_L", "SpikeSlime_L"}:
            return
        if (
            int(monster.current_hp) > 0
            and int(monster.current_hp) <= int(monster.max_hp) // 2
            and monster.next_move != "SPLIT"
            and not bool(monster.meta.get("split_triggered", False))
        ):
            monster.meta["split_triggered"] = True
            monster.next_move = "SPLIT"
            monster.intent = "UNKNOWN"
            monster.move_adjusted_damage = 0
            monster.move_hits = 0

    def _force_pending_slime_split_move(self, monster: MonsterState) -> bool:
        if monster.monster_id not in {"SlimeBoss", "AcidSlime_L", "SpikeSlime_L"}:
            return False
        if (
            int(monster.current_hp) > 0
            and int(monster.current_hp) <= int(monster.max_hp) // 2
            and bool(monster.meta.get("split_triggered", False))
        ):
            monster.next_move = "SPLIT"
            monster.intent = "UNKNOWN"
            monster.move_adjusted_damage = 0
            monster.move_hits = 0
            return True
        return False

    def _roll_then_force_pending_large_slime_split_move(self, monster: MonsterState) -> bool:
        if monster.monster_id not in {"AcidSlime_L", "SpikeSlime_L"}:
            return False
        if not self._force_pending_slime_split_move(monster):
            return False
        # In the real action queue, the attack's RollMoveAction still consumes
        # aiRng before the later SetMoveAction restores SPLIT after thorns-like
        # retaliation triggered the split.
        if monster.monster_id == "AcidSlime_L":
            self._acid_slime_l_roll_next_move(monster)
        else:
            self._spike_slime_l_roll_next_move(monster)
        self._force_pending_slime_split_move(monster)
        return True

    def _exhaust_cards(self, cards: list[dict[str, Any]]) -> None:
        for card in cards:
            self._exhaust_card(card)

    def _player_deal_damage(self, monster: MonsterState, amount: int, *, attack_card: bool = True) -> int:
        relic_ids = self._relic_ids if attack_card else None
        monster_thorns = _get_power_amount(monster, "Thorns") if attack_card else 0
        slow = _get_power_amount(monster, "Slow")
        if not attack_card and slow > 0:
            amount = int(int(amount) * (1.0 + 0.1 * slow))
        block_before = int(monster.block)
        hp_before = int(monster.current_hp)
        if attack_card:
            flight = _get_power_amount(monster, "Flight")
            if flight > 0:
                amount = max(0, int(amount) // 2)
        if attack_card and "Boot" in relic_ids:
            dealt = self._deal_player_attack_damage_with_boot(monster, amount)
        else:
            dealt = _deal_damage(monster, amount)
        if dealt > 0 and int(monster.current_hp) <= 0:
            if not self._handle_special_monster_zero_hp(monster):
                self._kill_monster(monster)
        if dealt > 0:
            if attack_card and _alive(monster):
                self._reduce_monster_plated_armor_or_defer(monster)
            self._wake_lagavulin_after_hp_loss(monster)
        if attack_card and block_before > 0 and int(monster.block) == 0 and "HandDrill" in relic_ids:
            self._apply_monster_debuff(monster, "Vulnerable", 2)
        if dealt > 0 and attack_card:
            curl_up = _get_power_amount(monster, "Curl Up")
            if curl_up > 0 and dealt < hp_before:
                self._trigger_or_defer_curl_up(monster, curl_up)
            angry = _get_power_amount(monster, "Angry")
            if angry > 0:
                _add_power(monster, "Strength", angry)
            malleable = _get_power_amount(monster, "Malleable")
            if malleable > 0 and dealt < hp_before:
                self._trigger_or_defer_malleable_block(monster, malleable)
                for power in monster.powers:
                    if _canonical_power_id(str(power.get("power_id") or power.get("id") or "")) == "Malleable":
                        power["amount"] = malleable + 1
                        break
            if _get_power_amount(monster, "Shifting") > 0 and dealt < hp_before:
                _add_power(monster, "Strength", -dealt)
                if _get_power_amount(monster, "Artifact") <= 0:
                    _add_power(monster, "LoseStrength", dealt)
            if _has_power(monster, "Compulsive") and dealt < hp_before and _alive(monster):
                _writhing_mass_roll_next_move(monster, self.randoms, include_current_move=True)
            flight = _get_power_amount(monster, "Flight")
            if flight > 0 and int(monster.current_hp) > 0:
                self._reduce_flight_or_defer(monster, 1)
        if monster_thorns > 0:
            self._damage_player(
                monster_thorns,
                attacker=monster,
                damage_type="THORNS",
                apply_player_vulnerable=False,
            )
        if dealt > 0 and _alive(monster):
            self._handle_guardian_hp_loss_reactions(monster, dealt)
            self._trigger_slime_split_after_hp_loss(monster)
        return dealt

    def _handle_guardian_hp_loss_reactions(self, monster: MonsterState, dealt: int) -> None:
        if dealt <= 0 or not _alive(monster):
            return
        if monster.monster_id != "TheGuardian" or not bool(monster.meta.get("is_open", True)):
            return
        monster.meta["dmg_taken"] = int(monster.meta.get("dmg_taken", 0)) + int(dealt)
        mode_shift_amount = _get_power_amount(monster, "Mode Shift")
        if mode_shift_amount > 0:
            _set_power_amount(monster, "Mode Shift", mode_shift_amount - int(dealt))
        threshold = int(monster.meta.get("mode_shift_threshold", 30))
        if int(monster.meta.get("dmg_taken", 0)) >= threshold and not bool(monster.meta.get("close_up_triggered", False)):
            monster.meta["close_up_triggered"] = True
            monster.meta["dmg_taken"] = 0
            self._trigger_or_defer_guardian_mode_shift(monster)

    def _deal_player_attack_damage_with_boot(self, monster: MonsterState, amount: int) -> int:
        blocked = min(int(monster.block or 0), max(0, int(amount)))
        if blocked:
            monster.block -= blocked
        remaining = max(0, int(amount) - blocked)
        if 0 < remaining < 5:
            remaining = 5
        if remaining > 0 and _get_power_amount(monster, "Intangible") > 0:
            remaining = 1
        hp_before = int(monster.current_hp)
        monster.current_hp = max(0, int(monster.current_hp) - remaining)
        return hp_before - int(monster.current_hp)

    def _trigger_or_defer_curl_up(self, monster: MonsterState, amount: int) -> None:
        if self._defer_curl_up_depth > 0:
            if not any(current is monster for current in self._pending_curl_up_monsters):
                self._pending_curl_up_monsters.append(monster)
            return
        self._apply_curl_up(monster, amount)

    def _apply_curl_up(self, monster: MonsterState, amount: int) -> None:
        if amount <= 0 or not _alive(monster):
            return
        _gain_block(monster, amount)
        _remove_power(monster, "Curl Up")
        monster.meta["state"] = "CLOSED"

    def _flush_pending_curl_up(self) -> None:
        pending = list(self._pending_curl_up_monsters)
        self._pending_curl_up_monsters.clear()
        for monster in pending:
            curl_up = _get_power_amount(monster, "Curl Up")
            if curl_up > 0:
                self._apply_curl_up(monster, curl_up)

    def _trigger_or_defer_malleable_block(self, monster: MonsterState, amount: int) -> None:
        if self._defer_malleable_block_depth > 0:
            self._pending_malleable_blocks.append((monster, int(amount)))
            return
        _gain_block(monster, int(amount))

    def _flush_pending_malleable_blocks(self) -> None:
        pending = list(self._pending_malleable_blocks)
        self._pending_malleable_blocks.clear()
        for monster, amount in pending:
            if _alive(monster):
                _gain_block(monster, int(amount))

    def _reduce_flight_or_defer(self, monster: MonsterState, amount: int = 1) -> None:
        if self._defer_flight_reduction_depth > 0:
            monster.meta["_pending_flight_reduction"] = int(monster.meta.get("_pending_flight_reduction", 0) or 0) + int(amount)
            return
        self._apply_flight_reduction(monster, amount)

    def _apply_flight_reduction(self, monster: MonsterState, amount: int) -> None:
        flight = _get_power_amount(monster, "Flight")
        if flight <= 0:
            return
        next_amount = flight - int(amount)
        _set_power_amount(monster, "Flight", next_amount)
        if next_amount <= 0 and monster.monster_id == "Byrd" and _alive(monster):
            monster.meta["is_flying"] = False
            monster.next_move = "STUNNED"
            monster.intent = "STUN"

    def _flush_deferred_flight_reductions(self) -> None:
        for monster in self.state.monsters:
            pending = int(monster.meta.pop("_pending_flight_reduction", 0) or 0)
            if pending > 0:
                self._apply_flight_reduction(monster, pending)

    def _trigger_or_defer_guardian_mode_shift(self, monster: MonsterState) -> None:
        if self._defer_guardian_mode_shift_depth > 0:
            if not any(current is monster for current in self._pending_guardian_mode_shift_monsters):
                self._pending_guardian_mode_shift_monsters.append(monster)
            return
        self._apply_guardian_mode_shift(monster)

    def _apply_guardian_mode_shift(self, monster: MonsterState) -> None:
        if monster.monster_id != "TheGuardian" or not _alive(monster):
            return
        if not bool(monster.meta.get("is_open", True)):
            return
        threshold = int(monster.meta.get("mode_shift_threshold", 30))
        monster.meta["is_open"] = False
        _gain_block(monster, int(monster.meta.get("defensive_block", 20)))
        if self._preserve_next_guardian_mode_shift_block_clear:
            monster.meta["_skip_next_block_clear"] = True
        monster.meta["mode_shift_threshold"] = threshold + int(monster.meta.get("mode_shift_increase", 10))
        _remove_power(monster, "Mode Shift")
        monster.next_move = "CLOSE_UP"
        monster.intent = "BUFF"

    def _flush_pending_guardian_mode_shift(self) -> None:
        pending = list(self._pending_guardian_mode_shift_monsters)
        self._pending_guardian_mode_shift_monsters.clear()
        for monster in pending:
            self._apply_guardian_mode_shift(monster)

    def _apply_card_effect_with_damage_queue(
        self,
        card: dict[str, Any],
        target: MonsterState | None,
        *,
        defer_juggernaut_flush: bool = False,
    ) -> list[int]:
        juggernaut_start = len(self._pending_juggernaut_damage)
        self._defer_juggernaut_damage_depth += 1
        queued_juggernaut_damage: list[int] = []
        try:
            strike_dummy_bonus = 3 if card.get("type") == "ATTACK" and "StrikeDummy" in self._relic_ids and _card_has_strike_tag(card) else 0
            original_base_damage = card.get("base_damage")
            if strike_dummy_bonus:
                card["base_damage"] = int(card.get("base_damage") or 0) + strike_dummy_bonus
            if card.get("type") != "ATTACK":
                try:
                    self._apply_card_effect(card, target)
                finally:
                    if strike_dummy_bonus:
                        card["base_damage"] = original_base_damage
            else:
                self._defer_curl_up_depth += 1
                self._defer_plated_armor_reduce_depth += 1
                self._defer_malleable_block_depth += 1
                self._defer_guardian_mode_shift_depth += 1
                try:
                    self._apply_card_effect(card, target)
                finally:
                    if strike_dummy_bonus:
                        card["base_damage"] = original_base_damage
                    self._defer_guardian_mode_shift_depth -= 1
                    if self._defer_guardian_mode_shift_depth == 0:
                        if (
                            self.pending_card_select is not None
                            and str(self.pending_card_select.get("mode") or "").upper() == "HEADBUTT"
                            and self._pending_guardian_mode_shift_monsters
                        ):
                            self.pending_card_select["deferred_guardian_mode_shift"] = True
                        else:
                            self._flush_pending_guardian_mode_shift()
                    self._defer_malleable_block_depth -= 1
                    if self._defer_malleable_block_depth == 0:
                        if (
                            self._pending_malleable_blocks
                            and self._defer_juggernaut_damage_depth == 1
                            and len(self._pending_juggernaut_damage) > juggernaut_start
                        ):
                            queued_juggernaut_damage = [
                                int(amount) for amount in self._pending_juggernaut_damage[juggernaut_start:]
                            ]
                            del self._pending_juggernaut_damage[juggernaut_start:]
                            self._queue_or_resolve_deferred_juggernaut_damage(queued_juggernaut_damage)
                        self._flush_pending_malleable_blocks()
                    self._defer_curl_up_depth -= 1
                    if self._defer_curl_up_depth == 0:
                        self._flush_pending_curl_up()
                    self._defer_plated_armor_reduce_depth -= 1
                    if self._defer_plated_armor_reduce_depth == 0:
                        if self.pending_card_select is not None and self._pending_plated_armor_reductions:
                            self.pending_card_select["deferred_plated_armor_reduction"] = True
                        else:
                            self._flush_pending_plated_armor_reductions()
        finally:
            self._defer_juggernaut_damage_depth -= 1
            if self._defer_juggernaut_damage_depth == 0:
                queued_juggernaut_damage = [int(amount) for amount in self._pending_juggernaut_damage[juggernaut_start:]]
                del self._pending_juggernaut_damage[juggernaut_start:]
                if not defer_juggernaut_flush:
                    self._queue_or_resolve_deferred_juggernaut_damage(queued_juggernaut_damage)
                    queued_juggernaut_damage = []
        return queued_juggernaut_damage

    def _gain_player_block(self, amount: int) -> None:
        if amount <= 0:
            return
        _gain_block(self.player, amount)
        juggernaut = _get_power_amount(self.player, "Juggernaut")
        if juggernaut > 0:
            if self._defer_juggernaut_damage_depth > 0:
                self._pending_juggernaut_damage.append(int(juggernaut))
            else:
                self._resolve_juggernaut_damage(int(juggernaut))

    def _resolve_juggernaut_damage(self, amount: int) -> None:
        if amount <= 0 or self.outcome != "UNDECIDED":
            return
        living = [monster for monster in self.state.monsters if _alive(monster)]
        if living:
            target_index = int(self.randoms.stream("card_random").random(0, len(living) - 1))
            self._player_deal_damage(living[target_index], int(amount), attack_card=False)
            if not _any_monsters_alive(self.state.monsters):
                self.outcome = "VICTORY"

    def _queue_or_resolve_deferred_juggernaut_damage(self, amounts: list[int]) -> None:
        pending_amounts = [int(amount) for amount in amounts if int(amount) > 0]
        if not pending_amounts:
            return
        if self.pending_card_select is not None:
            existing = [int(amount) for amount in list(self.pending_card_select.get("deferred_juggernaut_damage") or [])]
            existing.extend(pending_amounts)
            self.pending_card_select["deferred_juggernaut_damage"] = existing
            return
        for amount in pending_amounts:
            self._resolve_juggernaut_damage(amount)
            if self.outcome != "UNDECIDED":
                break

    def _flush_deferred_juggernaut_damage(self, pending: dict[str, Any]) -> None:
        amounts = [int(amount) for amount in list(pending.pop("deferred_juggernaut_damage", []) or []) if int(amount) > 0]
        for amount in amounts:
            self._resolve_juggernaut_damage(amount)
            if self.outcome != "UNDECIDED":
                break

    def _flush_pending_feel_no_pain_blocks(self) -> None:
        blocks = [int(amount) for amount in list(self._pending_feel_no_pain_blocks) if int(amount) > 0]
        self._pending_feel_no_pain_blocks = []
        for amount in blocks:
            self._gain_player_block(amount)
            if self.outcome != "UNDECIDED":
                break

    def _trigger_player_hand_on_other_card_played(self, played_card: dict[str, Any]) -> None:
        if str(played_card.get("card_id") or "") == "Pain":
            return
        pain_count = sum(1 for hand_card in self.state.hand if hand_card.get("card_id") == "Pain")
        if pain_count > 0:
            self._lose_player_hp(pain_count)

    def _apply_monster_debuff(self, monster: MonsterState | None, power_id: str, amount: int) -> None:
        if monster is None or amount == 0 or not _alive(monster):
            return
        power_id = _canonical_power_id(power_id)
        if power_id not in MONSTER_DEBUFF_POWERS:
            _add_power(monster, power_id, amount)
            return
        artifact = _get_power_amount(monster, "Artifact")
        if artifact > 0:
            _set_power_amount(monster, "Artifact", artifact - 1)
            return
        before_amount = _get_power_amount(monster, power_id)
        _add_power(monster, power_id, amount)
        after_amount = _get_power_amount(monster, power_id)
        if after_amount != before_amount:
            if power_id == "Vulnerable" and "Champion Belt" in self._relic_ids:
                self._apply_monster_debuff(monster, "Weak", 1)
                if self.outcome != "UNDECIDED":
                    return
            sadistic = _get_power_amount(self.player, "SadisticNature")
            if sadistic > 0:
                self._player_deal_damage(monster, sadistic, attack_card=False)
                if not _any_monsters_alive(self.state.monsters):
                    self.outcome = "VICTORY"

    def _card_block_amount(self, base_amount: int) -> int:
        if _get_power_amount(self.player, "NoBlock") > 0:
            return 0
        block = int(base_amount) + _get_power_amount(self.player, "Dexterity")
        block = max(0, block)
        if _get_power_amount(self.player, "Frail") > 0:
            block = int(float(block) * 0.75)
        return max(0, block)

    def _gain_player_card_block(self, base_amount: int) -> None:
        self._gain_player_block(self._card_block_amount(base_amount))

    def _apply_player_debuff(self, power_id: str, amount: int, *, source_monster: bool = False) -> None:
        if amount == 0:
            return
        if power_id not in PLAYER_DEBUFF_POWERS and not (
            power_id in {"Hex"} or (power_id in NEGATIVE_STACKABLE_POWERS and amount < 0)
        ):
            _add_power(self.player, power_id, amount)
            return
        relic_ids = self._relic_ids
        normalized = _canonical_power_id(power_id)
        if normalized == "Weakened" and "Ginger" in relic_ids:
            return
        if normalized == "Frail" and "Turnip" in relic_ids:
            return
        artifact = _get_power_amount(self.player, "Artifact")
        if artifact > 0:
            _set_power_amount(self.player, "Artifact", artifact - 1)
            return
        before_amount = _get_power_amount(self.player, normalized)
        if normalized == "No Draw":
            _set_power_amount(self.player, normalized, -1)
            return
        if normalized == "Confusion":
            _set_power_amount(self.player, normalized, -1)
            return
        _add_power(self.player, normalized, amount)
        monster_turn_source = getattr(self, "_in_monster_turn", False) and normalized in {
            "Weakened",
            "Frail",
            "Vulnerable",
            "NoBlockPower",
        }
        end_turn_curse_source = bool(source_monster) and normalized in {"Weakened", "Frail"}
        if before_amount <= 0 and (monster_turn_source or end_turn_curse_source):
            _mark_power_just_applied(self.player, normalized)

    def _apply_player_temporary_strength_loss(self, amount: int) -> None:
        amount = int(amount)
        if amount <= 0:
            return
        artifact = _get_power_amount(self.player, "Artifact")
        if artifact > 0:
            _set_power_amount(self.player, "Artifact", artifact - 1)
            return
        _add_power(self.player, "Flex", amount)

    def _apply_turn_start_damage_relics(self) -> None:
        relic_ids = self._relic_ids
        if "Mercury Hourglass" in relic_ids:
            for monster in self.state.monsters:
                if _alive(monster):
                    self._deal_non_attack_damage_to_monster(monster, 3)

    def _handle_attack_card_relics(self) -> None:
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id == "Nunchaku":
                counter = int(relic.get("counter", 0) or 0) + 1
                if counter % 10 == 0:
                    relic["counter"] = 0
                    self.player.energy += 1
                else:
                    relic["counter"] = counter
            elif relic_id == "Kunai":
                counter = int(relic.get("counter", 0) or 0) + 1
                if counter % 3 == 0:
                    relic["counter"] = 0
                    _add_power(self.player, "Dexterity", 1)
                else:
                    relic["counter"] = counter
            elif relic_id == "Shuriken":
                counter = int(relic.get("counter", 0) or 0) + 1
                if counter % 3 == 0:
                    relic["counter"] = 0
                    _add_power(self.player, "Strength", 1)
                else:
                    relic["counter"] = counter
            elif relic_id == "Ornamental Fan":
                counter = int(relic.get("counter", 0) or 0) + 1
                if counter % 3 == 0:
                    relic["counter"] = 0
                    self._gain_player_block(4)
                else:
                    relic["counter"] = counter
            elif relic_id == "Art of War":
                relic["gain_energy_next"] = False
            elif relic_id == "Pen Nib":
                counter = int(relic.get("counter", 0) or 0) + 1
                if counter == 10:
                    relic["counter"] = 0
                else:
                    relic["counter"] = counter
                    if counter == 9:
                        _append_power(self.player, "Pen Nib", 1, misc=1)
            elif relic_id == "OrangePellets":
                relic["attack_played"] = True
                self._trigger_orange_pellets_if_ready(relic)

    def _handle_skill_card_relics(self) -> None:
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id == "Letter Opener":
                counter = int(relic.get("counter", 0) or 0) + 1
                if counter % 3 == 0:
                    relic["counter"] = 0
                    for monster in self.state.monsters:
                        if _alive(monster):
                            self._deal_non_attack_damage_to_monster(monster, 5)
                    if self.outcome == "VICTORY":
                        return
                else:
                    relic["counter"] = counter
            elif relic_id == "OrangePellets":
                relic["skill_played"] = True
                self._trigger_orange_pellets_if_ready(relic)

    def _trigger_orange_pellets_if_ready(self, relic: dict[str, Any]) -> None:
        if not (
            bool(relic.get("attack_played"))
            and bool(relic.get("skill_played"))
            and bool(relic.get("power_played"))
        ):
            return
        _remove_player_debuffs(self.player)
        relic["attack_played"] = False
        relic["skill_played"] = False
        relic["power_played"] = False

    def _handle_any_card_relics(self, *, defer_ink_bottle_draw: bool = False) -> None:
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id == "InkBottle":
                counter = int(relic.get("counter", 0) or 0) + 1
                if counter == 10:
                    relic["counter"] = 0
                    if defer_ink_bottle_draw and self.pending_card_select is not None:
                        self.pending_card_select["deferred_ink_bottle_draws"] = (
                            int(self.pending_card_select.get("deferred_ink_bottle_draws", 0) or 0) + 1
                        )
                    else:
                        self.draw_cards(1)
                else:
                    relic["counter"] = counter
            elif relic_id == "Pocketwatch":
                relic["counter"] = int(relic.get("counter", 0) or 0) + 1
            elif relic_id == "Velvet Choker":
                counter = int(relic.get("counter", 0) or 0)
                if counter < 6:
                    relic["counter"] = counter + 1

    def _trigger_time_warp_after_card(self) -> bool:
        triggered = False
        for monster in self.state.monsters:
            if not _alive(monster):
                continue
            if _get_power_amount(monster, "Time Warp") <= 0 and not any(
                str(power.get("power_id") or power.get("id") or "") == "Time Warp"
                for power in monster.powers
            ):
                continue
            for power in monster.powers:
                if str(power.get("power_id") or power.get("id") or "") != "Time Warp":
                    continue
                amount = int(power.get("amount") or 0) + 1
                if amount >= 12:
                    power["amount"] = 0
                    triggered = True
                else:
                    power["amount"] = amount
                break
        if triggered:
            for monster in self.state.monsters:
                if _alive(monster):
                    _add_power(monster, "Strength", 2)
        return triggered

    def _handle_power_card_relics(self) -> None:
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id == "Bird Faced Urn":
                self._heal_player(2)
            elif relic_id == "Mummified Hand":
                eligible = [
                    hand_card
                    for hand_card in self.state.hand
                    if int(hand_card.get("cost", 0) or 0) > 0
                    and _card_cost(hand_card) > 0
                    and not bool(hand_card.get("free_to_play_once"))
                ]
                if eligible:
                    picked = eligible[int(self.randoms.stream("card_random").random(0, len(eligible) - 1))]
                    picked["cost_for_turn"] = 0
                    picked["_reset_cost_for_turn_after_play"] = True
            elif relic_id == "OrangePellets":
                relic["power_played"] = True
                self._trigger_orange_pellets_if_ready(relic)
        for monster in self.state.monsters:
            if _alive(monster):
                curiosity = _get_power_amount(monster, "Curiosity")
                if curiosity > 0:
                    _add_power(monster, "Strength", curiosity)

    def _random_card_candidates(
        self,
        *,
        predicate: Any,
        source_scope: str = "COLORED",
        exclude_healing: bool = True,
    ) -> list[str]:
        catalog = card_catalog()
        source_keys = {
            "COLORED": ("SRC_COMMON", "SRC_UNCOMMON", "SRC_RARE"),
            "COLORLESS": ("SRC_COLORLESS",),
            "ALL": ("SRC_COMMON", "SRC_UNCOMMON", "SRC_RARE", "SRC_COLORLESS"),
            "CURSE": ("SRC_CURSE",),
        }.get(str(source_scope), ("SRC_COMMON", "SRC_UNCOMMON", "SRC_RARE"))
        candidates: list[str] = []
        for source_key in source_keys:
            for card_id in self.source_card_pools.get(source_key, []):
                card_def = catalog.get(card_id)
                if card_def is None:
                    continue
                if exclude_healing and card_id in HEALING_CARD_IDS:
                    continue
                if predicate(card_id, card_def):
                    candidates.append(card_id)
        return candidates

    def _make_random_card(
        self,
        *,
        predicate: Any,
        uuid_prefix: str,
        free_this_turn: bool = False,
        source_scope: str = "COLORED",
        exclude_healing: bool = True,
    ) -> dict[str, Any] | None:
        candidates = self._random_card_candidates(
            predicate=predicate,
            source_scope=source_scope,
            exclude_healing=exclude_healing,
        )
        if not candidates:
            return None
        pick_index = int(self.randoms.stream("card_random").random(0, len(candidates) - 1))
        generated = make_card(candidates[pick_index], uuid=f"{uuid_prefix}-{len(self.state.hand)}-{pick_index}")
        _set_blood_for_blood_combat_cost_reduction(generated, self.player_damage_taken_this_combat)
        if free_this_turn:
            _set_cost_for_turn_like_sts(generated, 0)
            generated["_reset_cost_for_turn_after_play"] = True
        return generated

    def _make_random_card_choices(
        self,
        *,
        predicate: Any,
        uuid_prefix: str,
        count: int,
        source_scope: str = "COLORED",
        exclude_healing: bool = True,
    ) -> list[dict[str, Any]]:
        candidates = self._random_card_candidates(
            predicate=predicate,
            source_scope=source_scope,
            exclude_healing=exclude_healing,
        )
        if not candidates:
            return []
        cards: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        unique_candidate_count = len(set(candidates))
        while len(cards) < int(count) and len(seen_ids) < unique_candidate_count:
            pick_index = int(self.randoms.stream("card_random").random(0, len(candidates) - 1))
            card_id = candidates[pick_index]
            if card_id in seen_ids:
                continue
            seen_ids.add(card_id)
            generated = make_card(card_id, uuid=f"{uuid_prefix}-{len(cards)}-{pick_index}")
            _set_blood_for_blood_combat_cost_reduction(generated, self.player_damage_taken_this_combat)
            cards.append(generated)
        return cards

    def _burn_discovery_choice_regeneration(self, pending: dict[str, Any]) -> None:
        burn = pending.get("burn_after_choice")
        if not isinstance(burn, dict):
            return
        source_scope = str(burn.get("source_scope") or "COLORED")
        wanted_type = burn.get("card_type")
        if wanted_type is None:
            predicate = lambda _cid, card_def: card_def.type not in {"CURSE", "STATUS"}
        else:
            wanted = str(wanted_type)
            predicate = lambda _cid, card_def, wanted=wanted: card_def.type == wanted
        self._make_random_card_choices(
            predicate=predicate,
            uuid_prefix="discovery-burn",
            count=int(burn.get("count") or 3),
            source_scope=source_scope,
        )

    def _start_prepared_discard(self, discard_count: int) -> None:
        remaining = min(max(0, int(discard_count)), len(self.state.hand))
        if remaining <= 0:
            return
        if len(self.state.hand) <= remaining:
            while self.state.hand:
                discarded = self.state.hand.pop(0)
                _clear_play_once_flags_for_pile(discarded)
                self.state.discard_pile.append(discarded)
            return
        self.pending_card_select = {
            "mode": "PREPARED",
            "cards": [dict(hand_card) for hand_card in self.state.hand],
            "source_indexes": list(range(len(self.state.hand))),
            "num_cards": remaining,
            "remaining": remaining,
            "defer_move_used_card": True,
        }

    def _sync_accuracy_shivs(self) -> None:
        accuracy_amount = max(0, _get_power_amount(self.player, "Accuracy"))
        for pile in (self.state.hand, self.state.draw_pile, self.state.discard_pile, self.state.exhaust_pile):
            for pile_card in pile:
                if str(pile_card.get("card_id") or "") != "Shiv":
                    continue
                base = 6 if int(pile_card.get("upgrades") or 0) > 0 else 4
                pile_card["base_damage"] = base + accuracy_amount

    def _apply_card_effect(self, card: dict[str, Any], target: MonsterState | None) -> None:
        attacker_strength = _get_power_amount(self.player, "Strength")
        if str(card.get("type") or "") == "ATTACK":
            attacker_strength += _get_power_amount(self.player, "Vigor")
        attacker_weak = _is_weakened(self.player)
        if card["card_id"] == "Strike_R":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Shiv":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Cloak And Dagger":
            self._gain_player_card_block(int(card.get("base_block") or 0))
            for shiv_index in range(max(0, int(card.get("base_magic") or 1))):
                shiv = _make_shiv(
                    f"cloak-and-dagger-shiv-{self.state.turn}-{self.cards_played_this_turn}-{shiv_index}",
                    accuracy_amount=_get_power_amount(self.player, "Accuracy"),
                )
                if len(self.state.hand) < 10:
                    self.state.hand.append(shiv)
                else:
                    self.state.discard_pile.append(shiv)
        elif card["card_id"] == "Accuracy":
            _add_power(self.player, "Accuracy", int(card.get("base_magic") or 4))
            self._sync_accuracy_shivs()
        elif card["card_id"] == "Defend_R":
            self._gain_player_card_block(int(card.get("base_block") or 0))
        elif card["card_id"] == "Battle Trance":
            self.draw_cards(int(card.get("base_magic") or 3))
            self._apply_player_debuff("No Draw", -1)
        elif card["card_id"] == "Good Instincts":
            self._gain_player_card_block(int(card.get("base_block") or 0))
        elif card["card_id"] == "Dodge and Roll":
            block_amount = self._card_block_amount(int(card.get("base_block") or 0))
            self._gain_player_block(block_amount)
            _add_power(self.player, "NextTurnBlock", block_amount)
        elif card["card_id"] == "Leap":
            self._gain_player_card_block(int(card.get("base_block") or 0))
        elif card["card_id"] == "Bandage Up":
            self._heal_player(int(card.get("base_magic") or 4))
        elif card["card_id"] == "Deep Breath":
            if self.state.discard_pile:
                self.state.draw_pile.extend(self.state.discard_pile)
                self.state.discard_pile = []
                self._shuffle_cards(self.state.draw_pile)
            self.draw_cards(int(card.get("base_magic") or 1))
        elif card["card_id"] == "Prepared":
            draw_count = int(card.get("base_magic") or 1)
            self.draw_cards(draw_count)
            self._start_prepared_discard(draw_count)
        elif card["card_id"] == "Panacea":
            _add_power(self.player, "Artifact", int(card.get("base_magic") or 1))
        elif card["card_id"] == "Discovery":
            generated_cards = self._make_random_card_choices(
                predicate=lambda _cid, card_def: card_def.type not in {"CURSE", "STATUS"},
                uuid_prefix="discovery",
                count=3,
                source_scope="COLORED",
            )
            if generated_cards:
                self.pending_card_select = {
                    "mode": "DISCOVERY",
                    "cards": generated_cards,
                    "num_cards": 1,
                    "copies": 1,
                    "defer_move_used_card": True,
                }
        elif card["card_id"] == "Finesse":
            self._gain_player_card_block(int(card.get("base_block") or 0))
            self.draw_cards(1)
        elif card["card_id"] == "Forethought":
            if int(card.get("upgrades") or 0) > 0:
                to_move = list(self.state.hand)
                self.state.hand = []
                for chosen in to_move:
                    chosen["cost_for_combat"] = 0
                    chosen["cost_for_turn"] = 0
                    self.state.draw_pile.append(chosen)
            elif self.state.hand:
                chosen = _pop_hand_card(self.state.hand, 0)
                if chosen is not None:
                    chosen["cost_for_combat"] = 0
                    chosen["cost_for_turn"] = 0
                    self.state.draw_pile.append(chosen)
        elif card["card_id"] == "Enlightenment":
            for current in self.state.hand:
                if int(_card_cost(current)) > 1:
                    current["cost_for_turn"] = 1
                if int(card.get("upgrades") or 0) > 0 and int(current.get("cost", current.get("base_cost", 0)) or 0) > 1:
                    current["cost"] = 1
                    current["base_cost"] = 1
                    current["cost_for_combat"] = 1
        elif card["card_id"] == "Ghostly":
            _add_power(self.player, "Intangible", 1)
        elif card["card_id"] == "Ghostly Armor":
            self._gain_player_card_block(int(card.get("base_block") or 0))
        elif card["card_id"] == "Power Through":
            for index in range(2):
                self._make_temp_card_in_hand("Wound", uuid=f"power-through-wound-{self.state.turn}-{index}")
            self._gain_player_card_block(int(card.get("base_block") or 0))
        elif card["card_id"] == "Bloodletting":
            self._lose_player_hp(3)
            self.player.energy += int(card.get("base_magic") or 2)
        elif card["card_id"] == "Offering":
            self._lose_player_hp(6)
            self.player.energy += 2
            self.draw_cards(int(card.get("base_magic") or 3))
        elif card["card_id"] == "Berserk":
            self._apply_player_debuff("Vulnerable", int(card.get("base_magic") or 2))
            _add_power(self.player, "Berserk", 1)
        elif card["card_id"] == "Burning Pact":
            draw_count = int(card.get("base_magic") or 2)
            if self.state.hand:
                deferred_dead_branch_cards: list[dict[str, Any]] = []
                if len(self.state.hand) == 1:
                    exhausted = _pop_hand_card(self.state.hand, 0)
                    if exhausted is not None:
                        self._defer_dark_embrace_draw_depth += 1
                        try:
                            self._exhaust_card(exhausted, defer_dead_branch_to=deferred_dead_branch_cards)
                        finally:
                            self._defer_dark_embrace_draw_depth -= 1
                    self.draw_cards(draw_count)
                    self._add_temp_cards_to_hand_or_discard(deferred_dead_branch_cards)
                    self._flush_pending_dark_embrace_draws()
                else:
                    self.pending_card_select = {
                        "mode": "BURNING_PACT",
                        "cards": [dict(hand_card) for hand_card in self.state.hand],
                        "source_indexes": list(range(len(self.state.hand))),
                        "num_cards": 1,
                        "draw_count": draw_count,
                        "defer_move_used_card": True,
                    }
            else:
                self.draw_cards(draw_count)
        elif card["card_id"] == "Havoc":
            if not self.state.draw_pile and self.state.discard_pile:
                self.state.draw_pile = list(self.state.discard_pile)
                self.state.discard_pile = []
                self._shuffle_cards(self.state.draw_pile)
            if self.state.draw_pile:
                if not self._has_monster_after_use_already_triggered(card):
                    self._trigger_monster_on_after_use_card_powers()
                    self._mark_monster_after_use_triggered(card)
                havoc_target = None
                alive = [monster for monster in self.state.monsters if _alive(monster)]
                if alive:
                    havoc_target = alive[int(self.randoms.stream("card_random").random(0, len(alive) - 1))]
                top_card = self.state.draw_pile.pop()
                self._apply_hex_for_card(card)
                card["_hex_already_triggered"] = True
                top_card["free_to_play_once"] = True
                top_card["_force_exhaust_after_play"] = True
                top_card["_exclude_self_from_perfected_strike_count"] = True
                _refresh_card_flags([top_card], self.player)
                # PlayTopCardAction is queued after Havoc's use finishes, so the
                # Havoc card itself is already in the discard pile before a
                # generated draw/shuffle effect such as Warcry resolves.
                self._move_used_card(card)
                self._early_moved_used_card_ids.add(id(card))
                if bool(top_card.get("has_target")) and havoc_target is None:
                    return
                if not bool(top_card.get("is_playable", False)):
                    self._exhaust_card(top_card)
                    return
                deferred_discard_start = len(self.state.discard_pile)
                self._resolve_generated_card_play(top_card, forced_target=havoc_target, target_was_preselected=True)
                for exhausted_index, exhausted_card in enumerate(self.state.exhaust_pile):
                    if exhausted_card is top_card:
                        card["_deferred_post_move_exhausts"] = [self.state.exhaust_pile.pop(exhausted_index)]
                        break
                if len(self.state.discard_pile) > deferred_discard_start:
                    card["_deferred_post_move_discards"] = self.state.discard_pile[deferred_discard_start:]
                    del self.state.discard_pile[deferred_discard_start:]
        elif card["card_id"] == "Infernal Blade":
            generated = self._make_random_card(
                predicate=lambda _cid, card_def: card_def.type == "ATTACK" and card_def.color not in {"CURSE", "STATUS"},
                uuid_prefix="infernal-blade",
                free_this_turn=True,
                source_scope="COLORED",
            )
            if generated is not None:
                self._add_temp_card_to_hand_or_discard(generated, reset_for_discard=False)
        elif card["card_id"] == "Dark Embrace":
            _add_power(self.player, "Dark Embrace", int(card.get("base_magic") or 1))
        elif card["card_id"] == "Feel No Pain":
            _add_power(self.player, "Feel No Pain", int(card.get("base_magic") or 3))
        elif card["card_id"] == "Evolve":
            _add_power(self.player, "Evolve", int(card.get("base_magic") or 1))
        elif card["card_id"] == "Panache":
            _append_power(self.player, "Panache", 5, misc=int(card.get("base_magic") or 10))
        elif card["card_id"] == "Sadistic Nature":
            _add_power(self.player, "SadisticNature", int(card.get("base_magic") or 5))
        elif card["card_id"] == "Impatience":
            if not any(hand_card.get("type") == "ATTACK" for hand_card in self.state.hand):
                self.draw_cards(int(card.get("base_magic") or 2))
        elif card["card_id"] == "Warcry":
            before_deferred_fire_breathing = len(self._pending_fire_breathing_damage)
            self._defer_fire_breathing_damage_depth += 1
            try:
                self.draw_cards(int(card.get("base_magic") or 1))
            finally:
                self._defer_fire_breathing_damage_depth -= 1
            deferred_fire_breathing = self._pending_fire_breathing_damage[before_deferred_fire_breathing:]
            del self._pending_fire_breathing_damage[before_deferred_fire_breathing:]
            if self.state.hand:
                if len(self.state.hand) == 1:
                    chosen = _pop_hand_card(self.state.hand, 0)
                    if chosen is not None:
                        self.state.draw_pile.append(chosen)
                    for amount in deferred_fire_breathing:
                        self._deal_fire_breathing_damage(amount)
                        if self.outcome != "UNDECIDED":
                            break
                else:
                    self.pending_card_select = {
                        "mode": "WARCRY",
                        "cards": [dict(hand_card) for hand_card in self.state.hand],
                        "source_indexes": list(range(len(self.state.hand))),
                        "num_cards": 1,
                        "defer_move_used_card": True,
                        "deferred_fire_breathing_damage": list(deferred_fire_breathing),
                    }
            else:
                for amount in deferred_fire_breathing:
                    self._deal_fire_breathing_damage(amount)
                    if self.outcome != "UNDECIDED":
                        break
        elif card["card_id"] == "Flex":
            amount = int(card.get("base_magic") or 2)
            _add_power(self.player, "Strength", amount)
            self._apply_player_temporary_strength_loss(amount)
        elif card["card_id"] == "Seeing Red":
            self.player.energy += int(card.get("base_magic") or 2)
        elif card["card_id"] == "Armaments":
            self._gain_player_card_block(int(card.get("base_block") or 0))
            if int(card.get("upgrades") or 0) > 0:
                self.state.hand = [upgrade_card(card_in_hand) for card_in_hand in self.state.hand]
            else:
                upgradable_with_indexes = [
                    (index, hand_card)
                    for index, hand_card in enumerate(self.state.hand)
                    if can_upgrade_card(hand_card)
                ]
                upgradable_cards = [hand_card for _, hand_card in upgradable_with_indexes]
                if len(upgradable_cards) == 1:
                    for index, hand_card in enumerate(self.state.hand):
                        if hand_card is upgradable_cards[0]:
                            self.state.hand[index] = upgrade_card(hand_card)
                            break
                elif len(upgradable_cards) > 1:
                    cannot_upgrade_cards = [hand_card for hand_card in self.state.hand if not can_upgrade_card(hand_card)]
                    self.state.hand = upgradable_cards
                    self.pending_card_select = {
                        "mode": "ARMAMENTS",
                        "cards": [dict(hand_card) for hand_card in upgradable_cards],
                        "source_indexes": list(range(len(upgradable_cards))),
                        "cannot_upgrade_cards": cannot_upgrade_cards,
                        "selectable_count": len(upgradable_cards),
                        "num_cards": 1,
                        "defer_move_used_card": True,
                    }
        elif card["card_id"] == "Entrench":
            self._gain_player_block(int(self.player.block))
        elif card["card_id"] == "Shrug It Off":
            self._gain_player_card_block(int(card.get("base_block") or 0))
            self.draw_cards(int(card.get("base_magic") or 1))
        elif card["card_id"] == "Sentinel":
            self._gain_player_card_block(int(card.get("base_block") or 0))
        elif card["card_id"] == "True Grit":
            self._gain_player_card_block(int(card.get("base_block") or 0))
            if self.state.hand:
                if len(self.state.hand) <= 1:
                    exhaust_index = len(self.state.hand) - 1
                elif int(card.get("upgrades") or 0) > 0:
                    self.pending_card_select = {
                        "mode": "TRUE_GRIT",
                        "cards": [dict(hand_card) for hand_card in self.state.hand],
                        "source_indexes": list(range(len(self.state.hand))),
                        "num_cards": 1,
                        "defer_move_used_card": True,
                    }
                    return
                else:
                    exhaust_index = int(self.randoms.stream("card_random").random(0, len(self.state.hand) - 1))
                exhausted = _pop_hand_card(self.state.hand, exhaust_index)
                if exhausted is not None:
                    self._exhaust_card(exhausted)
        elif card["card_id"] == "Second Wind":
            non_attack_cards = [hand_card for hand_card in self.state.hand if hand_card.get("type") != "ATTACK"]
            self.state.hand = [hand_card for hand_card in self.state.hand if hand_card.get("type") == "ATTACK"]
            self._exhaust_cards(list(reversed(non_attack_cards)))
            block_per_card = self._card_block_amount(int(card.get("base_block") or 0))
            for _ in non_attack_cards:
                self._gain_player_block(block_per_card)
        elif card["card_id"] == "Dual Wield":
            targets = [
                (index, hand_card)
                for index, hand_card in enumerate(self.state.hand)
                if hand_card.get("type") in {"ATTACK", "POWER"}
            ]
            if targets:
                copies = 2 if int(card.get("upgrades") or 0) > 0 else max(1, int(card.get("base_magic") or 1))
                if len(targets) > 1:
                    self.pending_card_select = {
                        "mode": "DUAL_WIELD",
                        "cards": [dict(hand_card) for _, hand_card in targets],
                        "source_indexes": [index for index, _ in targets],
                        "num_cards": 1,
                        "copies": copies,
                    }
                    return
                target_card = targets[0][1]
                for copy_index in range(copies):
                    self._add_temp_card_to_hand_or_discard(
                        _copy_card({**target_card, "uuid": f"dual-wield-copy-{copy_index}-{target_card.get('card_id')}"}),
                        reset_for_discard=False,
                    )
        elif card["card_id"] == "Purity":
            exhaust_count = min(int(card.get("base_magic") or 3), len(self.state.hand))
            to_exhaust = [self.state.hand.pop(0) for _ in range(exhaust_count)]
            self._exhaust_cards(to_exhaust)
        elif card["card_id"] == "Double Tap":
            _add_power(self.player, "Double Tap", int(card.get("base_magic") or 0))
        elif card["card_id"] == "Rage":
            _add_power(self.player, "Rage", int(card.get("base_magic") or 0))
        elif card["card_id"] == "Bash":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
                self._apply_monster_debuff(target, "Vulnerable", int(card.get("base_magic") or 0))
        elif card["card_id"] == "Flash of Steel":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
            self.draw_cards(1)
        elif card["card_id"] == "HandOfGreed":
            if target is not None:
                target_before = int(target.current_hp)
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
                if target_before > 0 and not _alive(target):
                    self.bonus_reward_gold += int(card.get("base_magic") or 20)
        elif card["card_id"] == "Mind Blast":
            if target is not None:
                damage = self._scale_player_attack_damage(len(self.state.draw_pile) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Clothesline":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
                self._apply_monster_debuff(target, "Weakened", int(card.get("base_magic") or 0))
        elif card["card_id"] == "Intimidate":
            for monster in self.state.monsters:
                if _alive(monster):
                    self._apply_monster_debuff(monster, "Weak", int(card.get("base_magic") or 1))
        elif card["card_id"] == "Disarm":
            if target is not None:
                self._apply_monster_debuff(target, "Strength", -int(card.get("base_magic") or 0))
        elif card["card_id"] == "Dark Shackles":
            if target is not None:
                amount = int(card.get("base_magic") or 9)
                artifact_before = _get_power_amount(target, "Artifact")
                self._apply_monster_debuff(target, "Strength", -amount)
                if artifact_before <= 0 and _get_power_amount(target, "Artifact") <= 0:
                    _append_power(target, "Shackled", amount, misc=amount)
        elif card["card_id"] == "J.A.X.":
            self._lose_player_hp(3)
            _add_power(self.player, "Strength", int(card.get("base_magic") or 2))
        elif card["card_id"] == "Hemokinesis":
            self._lose_player_hp(int(card.get("base_magic") or 2))
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Headbutt":
            defer_headbutt_flight_reduction = False
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._defer_gremlin_horn_depth += 1
                self._defer_flight_reduction_depth += 1
                try:
                    self._player_deal_damage(target, damage)
                finally:
                    self._defer_flight_reduction_depth -= 1
                    defer_headbutt_flight_reduction = self._defer_flight_reduction_depth <= 0
                    self._defer_gremlin_horn_depth -= 1
            if self.outcome != "UNDECIDED":
                self._flush_deferred_gremlin_horn_rewards()
                if defer_headbutt_flight_reduction:
                    self._flush_deferred_flight_reductions()
                return
            if len(self.state.discard_pile) == 1:
                self.state.draw_pile.append(self.state.discard_pile.pop(0))
                self._flush_deferred_gremlin_horn_rewards()
                if defer_headbutt_flight_reduction:
                    self._flush_deferred_flight_reductions()
            elif self.state.discard_pile:
                self.pending_card_select = {
                    "mode": "HEADBUTT",
                    "cards": [dict(discard_card) for discard_card in self.state.discard_pile],
                    "source_indexes": list(range(len(self.state.discard_pile))),
                    "num_cards": 1,
                    "defer_move_used_card": True,
                    "deferred_flight_reduction": defer_headbutt_flight_reduction,
                }
            else:
                self._flush_deferred_gremlin_horn_rewards()
                if defer_headbutt_flight_reduction:
                    self._flush_deferred_flight_reductions()
        elif card["card_id"] == "Dropkick":
            if target is not None:
                target_was_vulnerable = _get_power_amount(target, "Vulnerable") > 0
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
                if target_was_vulnerable:
                    self.draw_cards(1)
                    self.player.energy += 1
        elif card["card_id"] == "Uppercut":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
                amount = int(card.get("base_magic") or 1)
                self._apply_monster_debuff(target, "Weak", amount)
                self._apply_monster_debuff(target, "Vulnerable", amount)
        elif card["card_id"] == "Pommel Strike":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
            self.draw_cards(int(card.get("base_magic") or 0))
        elif card["card_id"] == "Cleave":
            for monster in self.state.monsters:
                if _alive(monster):
                    damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, monster, attacker_weak)
                    self._player_deal_damage(monster, damage)
        elif card["card_id"] == "Immolate":
            for monster in self.state.monsters:
                if _alive(monster):
                    damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, monster, attacker_weak)
                    self._player_deal_damage(monster, damage)
            self.state.discard_pile.append(make_card("Burn", uuid=f"immolate-burn-{len(self.state.discard_pile)}"))
        elif card["card_id"] == "Sword Boomerang":
            hits = int(card.get("base_magic") or 0)
            self._defer_flight_reduction_depth += 1
            try:
                for _ in range(hits):
                    alive = [monster for monster in self.state.monsters if _alive(monster)]
                    if not alive:
                        break
                    picked = alive[int(self.randoms.stream("card_random").random(0, len(alive) - 1))]
                    damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, picked, attacker_weak)
                    self._player_deal_damage(picked, damage)
            finally:
                self._defer_flight_reduction_depth -= 1
                if self._defer_flight_reduction_depth <= 0:
                    self._flush_deferred_flight_reductions()
        elif card["card_id"] == "Twin Strike":
            if target is not None:
                self._defer_flight_reduction_depth += 1
                try:
                    for _ in range(int(card.get("base_magic") or 2)):
                        damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                        self._player_deal_damage(target, damage)
                        if not _alive(target):
                            break
                finally:
                    self._defer_flight_reduction_depth -= 1
                    if self._defer_flight_reduction_depth <= 0:
                        self._flush_deferred_flight_reductions()
        elif card["card_id"] == "Anger":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
            anger_copy = _copy_card(card)
            anger_copy["uuid"] = f"anger-copy-{len(self.state.discard_pile)}"
            anger_copy.pop("energy_on_use", None)
            anger_copy.pop("_echoed", None)
            anger_copy.pop("_force_exhaust_after_play", None)
            anger_copy.pop("_deferred_post_move_discards", None)
            self.state.discard_pile.append(anger_copy)
        elif card["card_id"] == "Pummel":
            if target is not None:
                self._defer_flight_reduction_depth += 1
                try:
                    for _ in range(int(card.get("base_magic") or 4)):
                        damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                        self._player_deal_damage(target, damage)
                        if not _alive(target):
                            break
                finally:
                    self._defer_flight_reduction_depth -= 1
                    if self._defer_flight_reduction_depth <= 0:
                        self._flush_deferred_flight_reductions()
        elif card["card_id"] == "Carnage":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Swift Strike":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Iron Wave":
            self._gain_player_card_block(int(card.get("base_block") or 0))
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Perfected Strike":
            if target is not None:
                strike_count = _count_strike_cards(self.state)
                if not bool(card.get("_exclude_self_from_perfected_strike_count", False)) and _card_has_strike_tag(card):
                    strike_count += 1
                strike_bonus = strike_count * int(card.get("base_magic") or 2)
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + strike_bonus + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Feed":
            if target is not None:
                target_before = int(target.current_hp)
                target_was_minion = _has_power(target, "Minion")
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
                if target_before > 0 and not _alive(target) and not target_was_minion:
                    gain = int(card.get("base_magic") or 3)
                    self.player.max_hp += gain
                    self.player.current_hp += gain
        elif card["card_id"] == "RitualDagger":
            if target is not None:
                target_before = int(target.current_hp)
                target_was_minion = _has_power(target, "Minion")
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or card.get("misc") or 15) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
                if target_before > 0 and not _alive(target) and not target_was_minion:
                    gain = int(card.get("base_magic") or 3)
                    card["misc"] = int(card.get("misc") or 15) + gain
                    card["base_damage"] = int(card.get("misc") or 15)
                    for deck_card in self.master_deck:
                        if str(deck_card.get("uuid") or "") == str(card.get("uuid") or ""):
                            deck_card["misc"] = int(card["misc"])
                            deck_card["base_damage"] = int(card["base_damage"])
        elif card["card_id"] == "Reaper":
            heal_amount = 0
            for monster in self.state.monsters:
                if _alive(monster):
                    damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, monster, attacker_weak)
                    heal_amount += self._player_deal_damage(monster, damage)
            if heal_amount > 0:
                self._heal_player(heal_amount)
        elif card["card_id"] == "Master of Strategy":
            self.draw_cards(int(card.get("base_magic") or 3))
        elif card["card_id"] == "Thunderclap":
            for monster in self.state.monsters:
                if _alive(monster):
                    damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, monster, attacker_weak)
                    self._player_deal_damage(monster, damage)
                    self._apply_monster_debuff(monster, "Vulnerable", int(card.get("base_magic") or 1))
        elif card["card_id"] == "Blind":
            if int(card.get("upgrades") or 0) > 0:
                for monster in self.state.monsters:
                    if _alive(monster):
                        self._apply_monster_debuff(monster, "Weak", int(card.get("base_magic") or 2))
            elif target is not None:
                self._apply_monster_debuff(target, "Weak", int(card.get("base_magic") or 2))
        elif card["card_id"] == "Bite":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                dealt = self._player_deal_damage(target, damage)
                if dealt > 0:
                    self._heal_player(int(card.get("base_magic") or 2))
        elif card["card_id"] == "Trip":
            if int(card.get("upgrades") or 0) > 0:
                for monster in self.state.monsters:
                    if _alive(monster):
                        self._apply_monster_debuff(monster, "Vulnerable", int(card.get("base_magic") or 2))
            elif target is not None:
                self._apply_monster_debuff(target, "Vulnerable", int(card.get("base_magic") or 2))
        elif card["card_id"] == "Shockwave":
            amount = int(card.get("base_magic") or 3)
            for monster in self.state.monsters:
                if _alive(monster):
                    self._apply_monster_debuff(monster, "Weak", amount)
                    self._apply_monster_debuff(monster, "Vulnerable", amount)
        elif card["card_id"] == "Whirlwind":
            hits = int(card.get("energy_on_use") or 0)
            if "Chemical X" in self._relic_ids:
                hits += 2
            base_damage = int(card.get("base_damage") or 0)
            self._defer_flight_reduction_depth += 1
            try:
                for _ in range(hits):
                    for monster in self.state.monsters:
                        if _alive(monster):
                            damage = self._scale_player_attack_damage(base_damage + attacker_strength, monster, attacker_weak)
                            self._player_deal_damage(monster, damage)
            finally:
                self._defer_flight_reduction_depth -= 1
                if self._defer_flight_reduction_depth <= 0:
                    self._flush_deferred_flight_reductions()
        elif card["card_id"] == "Body Slam":
            if target is not None:
                damage = self._scale_player_attack_damage(int(self.player.block) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Bludgeon":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Clash":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Blood for Blood":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Searing Blow":
            if target is not None:
                bonus = int(card.get("upgrades") or 0) * (int(card.get("upgrades") or 0) + 3)
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + bonus + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Reckless Charge":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
            self._add_card_to_draw_pile_random_spot(make_card("Dazed", uuid=f"reckless-charge-dazed-{len(self.state.draw_pile)}"))
        elif card["card_id"] == "Dramatic Entrance":
            for monster in self.state.monsters:
                if _alive(monster):
                    damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, monster, attacker_weak)
                    self._player_deal_damage(monster, damage)
        elif card["card_id"] == "Rampage":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
            card["base_damage"] = int(card.get("base_damage") or 0) + int(card.get("base_magic") or 0)
        elif card["card_id"] == "Heavy Blade":
            if target is not None:
                heavy_strength = attacker_strength * int(card.get("base_magic") or 0)
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + heavy_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Go for the Eyes":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
                if _has_attack_intent(target):
                    self._apply_monster_debuff(target, "Weak", int(card.get("base_magic") or 1))
        elif card["card_id"] == "Bane":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
                if _alive(target) and _get_power_amount(target, "Poison") > 0:
                    self._player_deal_damage(target, damage)
        elif card["card_id"] == "Spot Weakness":
            if target is not None and _has_attack_intent(target):
                _add_power(self.player, "Strength", int(card.get("base_magic") or 0))
        elif card["card_id"] == "Wild Strike":
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
            self._add_card_to_draw_pile_random_spot(make_card("Wound", uuid=f"wound-{len(self.state.draw_pile)}"))
        elif card["card_id"] == "Fiend Fire":
            exhaust_count = len(self.state.hand)
            deferred_dead_branch_cards: list[dict[str, Any]] = []
            for _ in range(exhaust_count):
                if not self.state.hand:
                    break
                exhaust_index = int(self.randoms.stream("card_random").random(len(self.state.hand) - 1))
                self._exhaust_card(self.state.hand.pop(exhaust_index), defer_dead_branch_to=deferred_dead_branch_cards)
            if target is not None and _alive(target):
                for _ in range(exhaust_count):
                    damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                    self._player_deal_damage(target, damage)
                    if not _alive(target):
                        break
            self._add_temp_cards_to_hand_or_discard(deferred_dead_branch_cards)
        elif card["card_id"] == "Sever Soul":
            non_attack_cards = [hand_card for hand_card in self.state.hand if hand_card.get("type") != "ATTACK"]
            self.state.hand = [hand_card for hand_card in self.state.hand if hand_card.get("type") == "ATTACK"]
            self._exhaust_cards(list(reversed(non_attack_cards)))
            if target is not None:
                damage = self._scale_player_attack_damage(int(card.get("base_damage") or 0) + attacker_strength, target, attacker_weak)
                self._player_deal_damage(target, damage)
        elif card["card_id"] == "Combust":
            _append_power(self.player, "Combust", int(card.get("base_magic") or 0), misc=1)
        elif card["card_id"] == "Inflame":
            _add_power(self.player, "Strength", int(card.get("base_magic") or 0))
        elif card["card_id"] == "Metallicize":
            _add_power(self.player, "Metallicize", int(card.get("base_magic") or 0))
        elif card["card_id"] == "Rupture":
            _add_power(self.player, "Rupture", int(card.get("base_magic") or 1))
        elif card["card_id"] == "Fire Breathing":
            _add_power(self.player, "Fire Breathing", int(card.get("base_magic") or 6))
        elif card["card_id"] == "The Bomb":
            _append_power(self.player, "TheBomb0", 3, misc=int(card.get("base_magic") or 40))
        elif card["card_id"] == "Flame Barrier":
            self._gain_player_card_block(int(card.get("base_block") or 0))
            _add_power(self.player, "FlameBarrier", int(card.get("base_magic") or 4))
        elif card["card_id"] == "Brutality":
            _add_power(self.player, "Brutality", 1)
        elif card["card_id"] == "Juggernaut":
            _add_power(self.player, "Juggernaut", int(card.get("base_magic") or 0))
        elif card["card_id"] == "Impervious":
            self._gain_player_card_block(int(card.get("base_block") or 0))
        elif card["card_id"] == "PanicButton":
            self._gain_player_card_block(int(card.get("base_block") or 0))
            self._apply_player_debuff("NoBlock", int(card.get("base_magic") or 2))
        elif card["card_id"] == "Magnetism":
            _append_power(self.player, "Magnetism", int(card.get("base_magic") or 1), misc=int(card.get("base_magic") or 1))
        elif card["card_id"] == "Barricade":
            _append_power(self.player, "Barricade", -1, misc=-1)
        elif card["card_id"] == "Corruption":
            if not _has_power(self.player, "Corruption"):
                _append_power(self.player, "Corruption", -1, misc=-1)
        elif card["card_id"] == "Demon Form":
            _add_power(self.player, "DemonForm", int(card.get("base_magic") or 0))
        elif card["card_id"] == "Exhume":
            source_indexes = [
                index
                for index, exhausted in enumerate(self.state.exhaust_pile)
                if exhausted.get("card_id") != "Exhume"
            ]
            if len(source_indexes) == 1:
                exhumed = self.state.exhaust_pile.pop(source_indexes[0])
                if _has_power(self.player, "Corruption") and exhumed.get("type") == "SKILL":
                    _set_cost_for_turn_like_sts(exhumed, -9)
                self.state.hand.append(exhumed)
            elif source_indexes:
                self.pending_card_select = {
                    "mode": "EXHUME",
                    "cards": [_copy_card(self.state.exhaust_pile[index]) for index in source_indexes],
                    "source_indexes": source_indexes,
                    "num_cards": 1,
                    "defer_move_used_card": True,
                }
        elif card["card_id"] == "Jack Of All Trades":
            copies = 2 if int(card.get("upgrades") or 0) > 0 else 1
            for _ in range(copies):
                generated = self._make_random_card(
                    predicate=lambda _cid, card_def: card_def.color == "COLORLESS" and card_def.type not in {"CURSE", "STATUS"},
                    uuid_prefix="jack-of-all-trades",
                    free_this_turn=False,
                    source_scope="COLORLESS",
                )
                if generated is not None:
                    self._add_temp_card_to_hand_or_discard(generated, reset_for_discard=False)
        elif card["card_id"] == "Violence":
            self._draw_matching_cards_from_draw_pile_to_hand(int(card.get("base_magic") or 3), "ATTACK")
        elif card["card_id"] == "Secret Technique":
            ordered: list[int] = []
            for index, draw_card in enumerate(self.state.draw_pile):
                if draw_card.get("type") != "SKILL":
                    continue
                if not ordered:
                    ordered.append(index)
                else:
                    ordered.insert(int(self.randoms.stream("card_random").random(len(ordered) - 1)), index)
            source_indexes = ordered
            if len(source_indexes) == 1:
                selected_card = self.state.draw_pile.pop(source_indexes[0])
                if len(self.state.hand) < 10:
                    self.state.hand.append(selected_card)
                else:
                    _clear_play_once_flags_for_pile(selected_card)
                    self.state.discard_pile.append(selected_card)
            elif source_indexes:
                self.pending_card_select = {
                    "mode": "SECRET_TECHNIQUE",
                    "cards": [_copy_card(self.state.draw_pile[index]) for index in source_indexes],
                    "source_indexes": source_indexes,
                    "num_cards": 1,
                    "defer_move_used_card": True,
                }
        elif card["card_id"] == "Secret Weapon":
            ordered: list[int] = []
            for index, draw_card in enumerate(self.state.draw_pile):
                if draw_card.get("type") != "ATTACK":
                    continue
                if not ordered:
                    ordered.append(index)
                else:
                    ordered.insert(int(self.randoms.stream("card_random").random(len(ordered) - 1)), index)
            source_indexes = ordered
            if len(source_indexes) == 1:
                selected_card = self.state.draw_pile.pop(source_indexes[0])
                if len(self.state.hand) < 10:
                    self.state.hand.append(selected_card)
                else:
                    _clear_play_once_flags_for_pile(selected_card)
                    self.state.discard_pile.append(selected_card)
            elif source_indexes:
                self.pending_card_select = {
                    "mode": "SECRET_WEAPON",
                    "cards": [_copy_card(self.state.draw_pile[index]) for index in source_indexes],
                    "source_indexes": source_indexes,
                    "num_cards": 1,
                    "defer_move_used_card": True,
                }
        elif card["card_id"] == "Transmutation":
            effect = int(card.get("energy_on_use") or 0)
            if any((relic.get("relic_id") or relic.get("id")) == "Chemical X" for relic in getattr(self.player, "relics", [])):
                effect += 2
            for _ in range(effect):
                generated = self._make_random_card(
                    predicate=lambda _cid, card_def: card_def.color == "COLORLESS" and card_def.type not in {"CURSE", "STATUS"},
                    uuid_prefix="transmutation",
                    free_this_turn=True,
                    source_scope="COLORLESS",
                )
                if generated is not None:
                    if int(card.get("upgrades") or 0) > 0:
                        generated = upgrade_card(generated)
                    self._add_temp_card_to_hand_or_discard(generated, reset_for_discard=False)
        elif card["card_id"] == "Apotheosis":
            for pile in (self.state.hand, self.state.draw_pile, self.state.discard_pile, self.state.exhaust_pile):
                for index, pile_card in enumerate(list(pile)):
                    pile[index] = upgrade_card(pile_card)
        elif card["card_id"] == "Limit Break":
            strength = _get_power_amount(self.player, "Strength")
            if strength > 0:
                _add_power(self.player, "Strength", strength)
        elif card["card_id"] == "Thinking Ahead":
            self.draw_cards(2)
            if self.state.hand:
                if len(self.state.hand) == 1:
                    self.randoms.stream("card_random").random(0)
                    chosen = _pop_hand_card(self.state.hand, 0)
                    if chosen is not None:
                        self.state.draw_pile.append(chosen)
                else:
                    self.pending_card_select = {
                        "mode": "PUT_ON_DECK",
                        "cards": [dict(hand_card) for hand_card in self.state.hand],
                        "source_indexes": list(range(len(self.state.hand))),
                        "num_cards": 1,
                        "defer_move_used_card": True,
                    }
        elif card["card_id"] == "Madness":
            if self.state.hand:
                better_possible = any(_card_cost(hand_card) > 0 for hand_card in self.state.hand)
                possible = any(int(hand_card.get("cost", 0) or 0) > 0 for hand_card in self.state.hand)
                if better_possible or possible:
                    while True:
                        picked_index = int(self.randoms.stream("card_random").random(len(self.state.hand) - 1))
                        candidate = self.state.hand[picked_index]
                        if better_possible and _card_cost(candidate) <= 0:
                            continue
                        if not better_possible and int(candidate.get("cost", 0) or 0) <= 0:
                            continue
                        break
                    picked = self.state.hand[picked_index]
                    picked["cost"] = 0
                    picked["cost_for_turn"] = 0
                    picked["cost_for_combat"] = 0
        elif card["card_id"] == "Chrysalis":
            for _ in range(int(card.get("base_magic") or 3)):
                generated = self._make_random_card(
                    predicate=lambda _cid, card_def: card_def.type == "SKILL" and card_def.color not in {"CURSE", "STATUS"},
                    uuid_prefix="chrysalis",
                    free_this_turn=False,
                    source_scope="COLORED",
                )
                if generated is not None:
                    if int(generated.get("cost") or 0) > 0:
                        generated["cost"] = 0
                        generated["cost_for_turn"] = 0
                    self._add_card_to_draw_pile_random_spot(generated)
        elif card["card_id"] == "Metamorphosis":
            for _ in range(int(card.get("base_magic") or 3)):
                generated = self._make_random_card(
                    predicate=lambda _cid, card_def: card_def.type == "ATTACK" and card_def.color not in {"CURSE", "STATUS"},
                    uuid_prefix="metamorphosis",
                    free_this_turn=False,
                    source_scope="COLORED",
                )
                if generated is not None:
                    if int(generated.get("cost") or 0) > 0:
                        generated["cost"] = 0
                        generated["cost_for_turn"] = 0
                    self._add_card_to_draw_pile_random_spot(generated)
        elif card["card_id"] == "Mayhem":
            _append_power(self.player, "Mayhem", int(card.get("base_magic") or 1), misc=int(card.get("base_magic") or 1))
        elif card["card_id"] in {"Pride", "Writhe"}:
            pass
        elif card["card_id"] in {
            "AscendersBane",
            "Burn",
            "Clumsy",
            "CurseOfTheBell",
            "Dazed",
            "Doubt",
            "Injury",
            "Necronomicurse",
            "Normality",
            "Pain",
            "Parasite",
            "Regret",
            "Shame",
            "Slimed",
            "Void",
            "Wound",
            "Decay",
        }:
            pass
        else:
            raise NotImplementedError(f"native_sim_v3 card {card['card_id']!r} is not implemented yet.")
        if target is not None and not _alive(target):
            self._grant_stolen_gold_on_death(target)

    def _consume_card_replay_source(self, card: dict[str, Any]) -> str | None:
        duplication_amount = _get_power_amount(self.player, "DuplicationPower")
        double_tap_amount = _get_power_amount(self.player, "Double Tap")
        necronomicon_relic = self._relic("Necronomicon")
        necronomicon_replays_attack = (
            necronomicon_relic is not None
            and card.get("type") == "ATTACK"
            and (
                (_card_cost(card) >= 2 and not bool(card.get("free_to_play_once")))
                or (int(card.get("cost") or 0) == -1 and int(card.get("energy_on_use") or 0) >= 2)
            )
            and self._necronomicon_activated_turn != int(self.state.turn)
        )
        if duplication_amount <= 0 and double_tap_amount <= 0 and not necronomicon_replays_attack:
            return None
        if duplication_amount <= 0 and card.get("type") != "ATTACK":
            return None
        if duplication_amount > 0:
            _set_power_amount(self.player, "DuplicationPower", duplication_amount - 1)
            return "DuplicationPower"
        elif double_tap_amount > 0:
            _set_power_amount(self.player, "Double Tap", double_tap_amount - 1)
            return "Double Tap"
        elif necronomicon_relic is not None:
            self._necronomicon_activated_turn = int(self.state.turn)
            return "Necronomicon"
        return None

    def _resolve_double_tap_replay(
        self,
        card: dict[str, Any],
        target: MonsterState | None,
        *,
        consume_source: bool = True,
    ) -> None:
        if consume_source and self._consume_card_replay_source(card) is None:
            return
        if not _any_monsters_alive(self.state.monsters):
            return
        if bool(card.get("has_target")) and (target is None or not _alive(target)):
            return
        replay_card = _copy_card(card)
        replay_card["_echoed"] = True
        self.cards_played_this_turn += 1
        self._trigger_player_hand_on_other_card_played(replay_card)
        if self.outcome == "DEFEAT":
            return
        self._trigger_gremlin_nob_anger(replay_card)
        self._trigger_on_use_card_powers(replay_card)
        pen_nib_active = bool(replay_card.get("type") == "ATTACK" and _get_power_amount(self.player, "Pen Nib") > 0)
        self._double_attack_damage = pen_nib_active
        monster_on_use_snapshot = self._monster_on_use_card_power_snapshot(replay_card)
        self._defer_dark_embrace_draw_depth += 1
        try:
            self._apply_card_effect_with_damage_queue(replay_card, target)
        finally:
            self._defer_dark_embrace_draw_depth -= 1
        self._trigger_hex_after_card_effect(replay_card)
        if replay_card.get("type") == "ATTACK":
            self._handle_attack_card_relics()
            if pen_nib_active:
                _remove_power(self.player, "Pen Nib")
                self._double_attack_damage = False
            _remove_power(self.player, "Vigor")
            rage = _get_power_amount(self.player, "Rage")
            self._defer_or_gain_rage_block(rage)
        elif replay_card.get("type") == "SKILL":
            self._handle_skill_card_relics()
        elif replay_card.get("type") == "POWER":
            self._handle_power_card_relics()
        self._handle_or_defer_any_card_relics()
        if self.pending_card_select is not None and self.outcome == "UNDECIDED":
            self.pending_card_select["deferred_monster_on_use_snapshot"] = monster_on_use_snapshot
        else:
            self._resolve_monster_on_use_card_power_snapshot(monster_on_use_snapshot)
        if self.outcome == "DEFEAT":
            if pen_nib_active:
                _remove_power(self.player, "Pen Nib")
            self._double_attack_damage = False
            return
        if not self._has_monster_after_use_already_triggered(replay_card):
            self._trigger_monster_on_after_use_card_powers()
        if self._trigger_time_warp_after_card() and self.outcome == "UNDECIDED":
            if pen_nib_active:
                _remove_power(self.player, "Pen Nib")
            self._double_attack_damage = False
            self._maybe_trigger_unceasing_top()
            if not _any_monsters_alive(self.state.monsters):
                self.outcome = "VICTORY"
            elif self.player.current_hp <= 0:
                self.outcome = "DEFEAT"
            if self.outcome == "UNDECIDED":
                self._update_monster_intents()
                self._end_turn()
            return
        self._double_attack_damage = False
        self._maybe_trigger_unceasing_top()
        if target is not None and not _alive(target):
            self._grant_stolen_gold_on_death(target)

    def _defer_or_resolve_double_tap_replay(self, card: dict[str, Any], target: MonsterState | None) -> None:
        if self.pending_card_select is None or self.outcome != "UNDECIDED":
            self._resolve_double_tap_replay(card, target)
            return
        replay_source = self._consume_card_replay_source(card)
        if replay_source is None:
            return
        target_index = None
        if target is not None:
            for index, monster in enumerate(self.state.monsters):
                if monster is target:
                    target_index = index
                    break
        self.pending_card_select["deferred_double_tap_replay"] = {
            "card": card,
            "target_index": target_index,
            "replay_source": replay_source,
        }

    def _resolve_pending_deferred_double_tap_replay(self, pending: dict[str, Any]) -> None:
        deferred = pending.pop("deferred_double_tap_replay", None)
        if not isinstance(deferred, dict):
            return
        card = deferred.get("card")
        if not isinstance(card, dict):
            return
        target = None
        target_index = deferred.get("target_index")
        if target_index is not None and 0 <= int(target_index) < len(self.state.monsters):
            target = self.state.monsters[int(target_index)]
        self._resolve_double_tap_replay(card, target, consume_source=False)

    def _trigger_on_use_card_powers(self, card: dict[str, Any]) -> None:
        for power in list(self.player.powers):
            power_id = str(power.get("power_id") or power.get("id") or "")
            if power_id == "Panache":
                remaining = int(power.get("amount") or 0) - 1
                if remaining <= 0:
                    damage = int(power.get("misc") or 0)
                    for monster in self.state.monsters:
                        if _alive(monster):
                            self._deal_non_attack_damage_to_monster(monster, damage)
                    power["amount"] = 5
                else:
                    power["amount"] = remaining

    def _apply_hex_for_card(self, card: dict[str, Any]) -> None:
        if card.get("type") == "ATTACK":
            return
        amount = _get_power_amount(self.player, "Hex")
        for index in range(max(0, amount)):
            self._add_card_to_draw_pile_random_spot(make_card("Dazed", uuid=f"hex-dazed-{self.state.turn}-{index}-{len(self.state.draw_pile)}"))

    def _trigger_hex_after_card_effect(self, card: dict[str, Any]) -> None:
        if bool(card.pop("_hex_already_triggered", False)):
            return
        self._apply_hex_for_card(card)

    def _monster_on_use_card_power_snapshot(self, card: dict[str, Any]) -> list[tuple[MonsterState, int]]:
        if card.get("type") != "ATTACK":
            return []
        snapshot: list[tuple[MonsterState, int]] = []
        for monster in self.state.monsters:
            if not _alive(monster):
                continue
            sharp_hide = _get_power_amount(monster, "Sharp Hide")
            if sharp_hide <= 0:
                continue
            snapshot.append((monster, sharp_hide))
        return snapshot

    def _resolve_monster_on_use_card_power_snapshot(self, snapshot: list[tuple[MonsterState, int]]) -> None:
        for monster, sharp_hide in snapshot:
            self._damage_player(
                sharp_hide,
                attacker=monster,
                damage_type="THORNS",
                apply_player_vulnerable=False,
            )
            if self.player.current_hp <= 0:
                self.outcome = "DEFEAT"
                return

    def _trigger_monster_on_use_card_powers(self, card: dict[str, Any]) -> None:
        snapshot = self._monster_on_use_card_power_snapshot(card)
        self._resolve_monster_on_use_card_power_snapshot(snapshot)

    def _trigger_monster_on_after_use_card_powers(self) -> None:
        for monster in self.state.monsters:
            if not _alive(monster):
                continue
            slow = _get_power_amount(monster, "Slow")
            if slow >= 0 and any(
                str(power.get("power_id") or power.get("id") or "") == "Slow"
                for power in monster.powers
            ):
                _set_power_amount(monster, "Slow", slow + 1)

    def _has_monster_after_use_already_triggered(self, card: dict[str, Any]) -> bool:
        return bool(card.get("_monster_after_use_already_triggered", False)) or id(card) in self._monster_after_use_triggered_card_ids

    def _mark_monster_after_use_triggered(self, card: dict[str, Any]) -> None:
        card["_monster_after_use_already_triggered"] = True
        self._monster_after_use_triggered_card_ids.add(id(card))

    def _trigger_gremlin_nob_anger(self, card: dict[str, Any]) -> None:
        if card.get("type") != "SKILL":
            return
        for monster in self.state.monsters:
            if monster.monster_id != "GremlinNob" or not _alive(monster):
                continue
            anger_amount = _get_power_amount(monster, "Anger")
            if anger_amount > 0:
                _add_power(monster, "Strength", anger_amount)

    def _resolve_generated_card_play(
        self,
        card: dict[str, Any],
        *,
        forced_target: MonsterState | None = None,
        target_was_preselected: bool = False,
    ) -> None:
        if _card_cost(card) < 0 and not bool(card.get("free_to_play_once")):
            self._move_used_card_and_flush_deferred_discards(card)
            return
        if _card_cost(card) < 0 and bool(card.get("free_to_play_once")):
            card["energy_on_use"] = int(self.player.energy)
        self.cards_played_this_turn += 1
        self._trigger_player_hand_on_other_card_played(card)
        if self.outcome == "DEFEAT":
            return
        self._trigger_gremlin_nob_anger(card)
        self._trigger_on_use_card_powers(card)
        pen_nib_active = bool(card.get("type") == "ATTACK" and _get_power_amount(self.player, "Pen Nib") > 0)
        self._double_attack_damage = pen_nib_active
        target = None
        if card.get("has_target"):
            if target_was_preselected:
                target = forced_target
            else:
                alive = [monster for monster in self.state.monsters if _alive(monster)]
                if alive:
                    target = alive[int(self.randoms.stream("card_random").random(0, len(alive) - 1))]
        monster_on_use_snapshot = self._monster_on_use_card_power_snapshot(card)
        previous_stasis_defer = self._defer_stasis_release_until_after_end_turn_discard
        self._defer_stasis_release_until_after_end_turn_discard = True
        try:
            deferred_juggernaut_damage = self._apply_card_effect_with_damage_queue(
                card,
                target,
                defer_juggernaut_flush=True,
            )
        finally:
            self._defer_stasis_release_until_after_end_turn_discard = previous_stasis_defer
        if not previous_stasis_defer:
            self._flush_pending_stasis_release_cards()
        if self.pending_card_select is not None and bool(self.pending_card_select.get("defer_move_used_card", False)):
            self.pending_card_select["deferred_hex_card"] = card
        else:
            self._trigger_hex_after_card_effect(card)
        if card.get("type") == "ATTACK":
            self._handle_attack_card_relics()
            if pen_nib_active:
                _remove_power(self.player, "Pen Nib")
                self._double_attack_damage = False
            _remove_power(self.player, "Vigor")
            rage = _get_power_amount(self.player, "Rage")
            self._defer_or_gain_rage_block(rage)
        elif card.get("type") == "SKILL":
            self._handle_skill_card_relics()
        elif card.get("type") == "POWER":
            self._handle_power_card_relics()
        self._handle_or_defer_any_card_relics()
        if self.pending_card_select is not None and self.outcome == "UNDECIDED":
            self.pending_card_select["deferred_monster_on_use_snapshot"] = monster_on_use_snapshot
        else:
            self._resolve_monster_on_use_card_power_snapshot(monster_on_use_snapshot)
        if self.outcome == "DEFEAT":
            return
        if not self._has_monster_after_use_already_triggered(card):
            self._trigger_monster_on_after_use_card_powers()
        if self._trigger_time_warp_after_card() and self.outcome == "UNDECIDED":
            if not self._defer_used_card_if_pending(card):
                self._move_used_card_and_flush_deferred_discards(card)
                self._queue_or_resolve_deferred_juggernaut_damage(deferred_juggernaut_damage)
            else:
                self._queue_or_resolve_deferred_juggernaut_damage(deferred_juggernaut_damage)
            self._maybe_trigger_unceasing_top()
            if not _any_monsters_alive(self.state.monsters):
                self.outcome = "VICTORY"
            elif self.player.current_hp <= 0:
                self.outcome = "DEFEAT"
            if self.outcome == "UNDECIDED":
                self._update_monster_intents()
                self._end_turn()
            return
        self._double_attack_damage = False
        if not self._defer_used_card_if_pending(card):
            self._move_used_card_and_flush_deferred_discards(card)
            self._queue_or_resolve_deferred_juggernaut_damage(deferred_juggernaut_damage)
        else:
            self._queue_or_resolve_deferred_juggernaut_damage(deferred_juggernaut_damage)
        if (
            (card.get("type") == "ATTACK" or _get_power_amount(self.player, "DuplicationPower") > 0)
            and not bool(card.get("_echoed", False))
        ):
            self._defer_or_resolve_double_tap_replay(card, target)
        self._maybe_trigger_unceasing_top()

    def _play_card(self, action: dict[str, Any]) -> None:
        card_index = int(action["card_index"])
        if not (0 <= card_index < len(self.state.hand)):
            return
        card = self.state.hand[card_index]
        if card.get("card_id") == "Clash" and any(hand_card.get("type") != "ATTACK" for hand_card in self.state.hand):
            return
        card = self.state.hand.pop(card_index)
        self.cards_played_this_turn += 1
        self._trigger_player_hand_on_other_card_played(card)
        if self.outcome == "DEFEAT":
            return
        cost = _card_cost(card)
        if cost == -1:
            energy_on_use = int(self.player.energy)
            card["energy_on_use"] = energy_on_use
            if not bool(card.get("free_to_play_once")):
                self.player.energy = 0
        else:
            self.player.energy = max(0, int(self.player.energy) - max(0, cost))
        self._trigger_gremlin_nob_anger(card)
        self._trigger_on_use_card_powers(card)
        pen_nib_active = bool(card.get("type") == "ATTACK" and _get_power_amount(self.player, "Pen Nib") > 0)
        self._double_attack_damage = pen_nib_active
        target = None
        target_index = action.get("target_index")
        if target_index is not None and 0 <= int(target_index) < len(self.state.monsters):
            target = self.state.monsters[int(target_index)]
        monster_on_use_snapshot = self._monster_on_use_card_power_snapshot(card)
        previous_stasis_defer = self._defer_stasis_release_until_after_end_turn_discard
        self._defer_stasis_release_until_after_end_turn_discard = True
        self._defer_dark_embrace_draw_depth += 1
        try:
            deferred_juggernaut_damage = self._apply_card_effect_with_damage_queue(
                card,
                target,
                defer_juggernaut_flush=True,
            )
        finally:
            self._defer_dark_embrace_draw_depth -= 1
            self._defer_stasis_release_until_after_end_turn_discard = previous_stasis_defer
        if not previous_stasis_defer:
            self._flush_pending_stasis_release_cards()
        if self.pending_card_select is not None and bool(self.pending_card_select.get("defer_move_used_card", False)):
            self.pending_card_select["deferred_hex_card"] = card
        else:
            self._trigger_hex_after_card_effect(card)
        if card.get("type") == "ATTACK":
            self._handle_attack_card_relics()
            if pen_nib_active:
                _remove_power(self.player, "Pen Nib")
                self._double_attack_damage = False
            _remove_power(self.player, "Vigor")
            rage = _get_power_amount(self.player, "Rage")
            self._defer_or_gain_rage_block(rage)
        elif card.get("type") == "SKILL":
            self._handle_skill_card_relics()
        elif card.get("type") == "POWER":
            self._handle_power_card_relics()
        self._handle_or_defer_any_card_relics()
        if self.pending_card_select is not None and self.outcome == "UNDECIDED":
            self.pending_card_select["deferred_monster_on_use_snapshot"] = monster_on_use_snapshot
        else:
            self._resolve_monster_on_use_card_power_snapshot(monster_on_use_snapshot)
        if self.outcome == "DEFEAT":
            return
        if not self._has_monster_after_use_already_triggered(card):
            self._trigger_monster_on_after_use_card_powers()
        if self._trigger_time_warp_after_card() and self.outcome == "UNDECIDED":
            self._double_attack_damage = False
            if not self._defer_used_card_if_pending(card):
                self._move_used_card_and_flush_deferred_discards(card)
                self._flush_pending_dark_embrace_draws()
                self._queue_or_resolve_deferred_juggernaut_damage(deferred_juggernaut_damage)
            else:
                self._queue_or_resolve_deferred_juggernaut_damage(deferred_juggernaut_damage)
            self._maybe_trigger_unceasing_top()
            if not _any_monsters_alive(self.state.monsters):
                self.outcome = "VICTORY"
            elif self.player.current_hp <= 0:
                self.outcome = "DEFEAT"
            if self.outcome == "UNDECIDED":
                self._update_monster_intents()
                self._end_turn()
            return
        self._double_attack_damage = False
        if not self._defer_used_card_if_pending(card):
            self._move_used_card_and_flush_deferred_discards(card)
            self._flush_pending_dark_embrace_draws()
            self._queue_or_resolve_deferred_juggernaut_damage(deferred_juggernaut_damage)
        else:
            self._queue_or_resolve_deferred_juggernaut_damage(deferred_juggernaut_damage)
        if (
            (card.get("type") == "ATTACK" or _get_power_amount(self.player, "DuplicationPower") > 0)
            and not bool(card.get("_echoed", False))
        ):
            self._defer_or_resolve_double_tap_replay(card, target)
        self._maybe_trigger_unceasing_top()
        if not _any_monsters_alive(self.state.monsters):
            self.outcome = "VICTORY"
        elif self.player.current_hp <= 0:
            self.outcome = "DEFEAT"
        self._update_monster_intents()

    def _cultist_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "INCANTATION":
            ritual_amount = 4 if self.ascension_level >= 2 else 3
            _add_power(monster, "Ritual", ritual_amount)
            monster.move_history.append("INCANTATION")
            monster.next_move = "DARK_STRIKE"
        else:
            self._damage_player(
                int(monster.move_adjusted_damage or 0),
                attacker=monster,
                apply_player_vulnerable=False,
            )
            ritual = _get_power_amount(monster, "Ritual")
            if ritual > 0:
                _add_power(monster, "Strength", ritual)
            monster.move_history.append("DARK_STRIKE")
            monster.next_move = "DARK_STRIKE"
        self.randoms.stream("ai").random(99)
        _decrement_turn_powers(monster)

    def _jaw_worm_roll_next_move(self, monster: MonsterState) -> None:
        if bool(monster.meta.get("first_move", False)):
            monster.meta["first_move"] = False
            monster.next_move = "CHOMP"
            return
        num = int(self.randoms.stream("ai").random(99))
        last_move = monster.move_history[-1] if monster.move_history else None
        last_two = monster.move_history[-2:]
        if num < 25:
            if last_move == "CHOMP":
                monster.next_move = "BELLOW" if self.randoms.stream("ai").random_boolean(0.5625) else "THRASH"
            else:
                monster.next_move = "CHOMP"
            return
        if num < 55:
            if len(last_two) == 2 and last_two[0] == "THRASH" and last_two[1] == "THRASH":
                monster.next_move = "CHOMP" if self.randoms.stream("ai").random_boolean(0.357) else "BELLOW"
            else:
                monster.next_move = "THRASH"
            return
        if last_move == "BELLOW":
            monster.next_move = "CHOMP" if self.randoms.stream("ai").random_boolean(0.416) else "THRASH"
        else:
            monster.next_move = "BELLOW"

    def _jaw_worm_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "CHOMP":
            self._monster_attack_player(monster, int(monster.meta.get("chomp_damage", 11)))
            monster.move_history.append("CHOMP")
        elif monster.next_move == "BELLOW":
            _add_power(monster, "Strength", int(monster.meta.get("bellow_str", 3)))
            _gain_block(monster, int(monster.meta.get("bellow_block", 6)))
            monster.move_history.append("BELLOW")
        elif monster.next_move == "THRASH":
            self._monster_attack_player(monster, int(monster.meta.get("thrash_damage", 7)))
            if _alive(monster):
                _gain_block(monster, int(monster.meta.get("thrash_block", 5)))
            monster.move_history.append("THRASH")
        else:
            raise NotImplementedError(f"native_sim_v3 Jaw Worm move {monster.next_move!r} is not implemented yet.")
        self._jaw_worm_roll_next_move(monster)
        _decrement_turn_powers(monster)

    def _blue_slaver_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "STAB":
            self._monster_attack_player(monster, int(monster.meta.get("stab_damage", 12)))
            monster.move_history.append("STAB")
        elif monster.next_move == "RAKE":
            self._monster_attack_player(monster, int(monster.meta.get("rake_damage", 7)))
            weak_amount = int(monster.meta.get("weak_amount", 1))
            if self.ascension_level >= 17:
                weak_amount += 1
            self._apply_player_debuff("Weak", weak_amount)
            monster.move_history.append("RAKE")
        else:
            raise NotImplementedError(f"native_sim_v3 Blue Slaver move {monster.next_move!r} is not implemented yet.")
        num = int(self.randoms.stream("ai").random(99))
        if num >= 40 and not (len(monster.move_history) >= 2 and monster.move_history[-1] == "STAB" and monster.move_history[-2] == "STAB"):
            monster.next_move = "STAB"
        elif self.ascension_level >= 17:
            monster.next_move = "RAKE" if monster.move_history[-1:] != ["RAKE"] else "STAB"
        else:
            monster.next_move = "RAKE" if not (len(monster.move_history) >= 2 and monster.move_history[-1] == "RAKE" and monster.move_history[-2] == "RAKE") else "STAB"
        _decrement_turn_powers(monster)

    def _red_slaver_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "ENTANGLE":
            self._apply_player_debuff("Entangled", 1)
            monster.meta["used_entangle"] = True
            monster.move_history.append("ENTANGLE")
        elif monster.next_move == "STAB":
            self._monster_attack_player(monster, int(monster.meta.get("stab_damage", 13)))
            monster.move_history.append("STAB")
        elif monster.next_move == "SCRAPE":
            self._monster_attack_player(monster, int(monster.meta.get("scrape_damage", 8)))
            vuln_amount = int(monster.meta.get("vuln_amount", 1))
            if self.ascension_level >= 17:
                vuln_amount += 1
            self._apply_player_debuff("Vulnerable", vuln_amount)
            monster.move_history.append("SCRAPE")
        else:
            raise NotImplementedError(f"native_sim_v3 Red Slaver move {monster.next_move!r} is not implemented yet.")
        num = int(self.randoms.stream("ai").random(99))
        used_entangle = bool(monster.meta.get("used_entangle", False))
        if num >= 75 and not used_entangle:
            monster.next_move = "ENTANGLE"
        elif num >= 55 and used_entangle and not (len(monster.move_history) >= 2 and monster.move_history[-1] == "STAB" and monster.move_history[-2] == "STAB"):
            monster.next_move = "STAB"
        elif self.ascension_level >= 17:
            monster.next_move = "SCRAPE" if monster.move_history[-1:] != ["SCRAPE"] else "STAB"
        else:
            monster.next_move = "SCRAPE" if not (len(monster.move_history) >= 2 and monster.move_history[-1] == "SCRAPE" and monster.move_history[-2] == "SCRAPE") else "STAB"
        _decrement_turn_powers(monster)

    def _fungi_beast_roll_next_move(self, monster: MonsterState) -> None:
        num = int(self.randoms.stream("ai").random(99))
        if num < 60:
            monster.next_move = "GROW" if len(monster.move_history) >= 2 and monster.move_history[-1] == "BITE" and monster.move_history[-2] == "BITE" else "BITE"
        else:
            monster.next_move = "BITE" if monster.move_history[-1:] == ["GROW"] else "GROW"

    def _fungi_beast_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "BITE":
            self._monster_attack_player(monster, int(monster.meta.get("bite_damage", 6)))
            monster.move_history.append("BITE")
        elif monster.next_move == "GROW":
            grow_amount = int(monster.meta.get("grow_strength", 3))
            if self.ascension_level >= 17:
                grow_amount += 1
            _add_power(monster, "Strength", grow_amount)
            monster.move_history.append("GROW")
        else:
            raise NotImplementedError(f"native_sim_v3 Fungi Beast move {monster.next_move!r} is not implemented yet.")
        self._fungi_beast_roll_next_move(monster)
        _decrement_turn_powers(monster)

    def _looter_take_turn(self, monster: MonsterState) -> None:
        gold_amt = int(monster.meta.get("gold_amt", 15))
        if monster.next_move == "MUG":
            slash_count = int(monster.meta.get("slash_count", 0))
            if monster.monster_id == "Mugger":
                self.randoms.stream("ai").random(2)
            if monster.monster_id == "Looter" and slash_count == 0:
                # The real game rolls aiRng.randomBoolean(0.6f) here to decide
                # whether to play the opening slash bark. The bark does not
                # affect simulator state, but the RNG consumption shifts the
                # later MUG -> SMOKE_BOMB/LUNGE branch.
                self.randoms.stream("ai").random_boolean(0.6)
            elif monster.monster_id == "Mugger" and slash_count == 1:
                self.randoms.stream("ai").random_boolean(0.6)
            stolen = min(gold_amt, self.gold)
            self.gold -= stolen
            monster.meta["stolen_gold"] = int(monster.meta.get("stolen_gold", 0)) + stolen
            self._monster_attack_player(monster, int(monster.meta.get("swipe_damage", 10)))
            monster.meta["slash_count"] = int(monster.meta.get("slash_count", 0)) + 1
            monster.move_history.append("MUG")
            if int(monster.meta.get("slash_count", 0)) == 2:
                next_attack = "BIGSWIPE" if monster.monster_id == "Mugger" else "LUNGE"
                monster.next_move = "SMOKE_BOMB" if self.randoms.stream("ai").random_boolean(0.5) else next_attack
            else:
                monster.next_move = "MUG"
        elif monster.next_move in {"LUNGE", "BIGSWIPE"}:
            if monster.monster_id == "Mugger":
                self.randoms.stream("ai").random(2)
            stolen = min(gold_amt, self.gold)
            self.gold -= stolen
            monster.meta["stolen_gold"] = int(monster.meta.get("stolen_gold", 0)) + stolen
            damage_key = "big_swipe_damage" if monster.next_move == "BIGSWIPE" else "lunge_damage"
            default_damage = 16 if monster.next_move == "BIGSWIPE" else 12
            self._monster_attack_player(monster, int(monster.meta.get(damage_key, default_damage)))
            monster.meta["slash_count"] = int(monster.meta.get("slash_count", 0)) + 1
            monster.move_history.append(monster.next_move)
            monster.next_move = "SMOKE_BOMB"
        elif monster.next_move == "SMOKE_BOMB":
            _gain_block(monster, int(monster.meta.get("escape_block", 6)))
            monster.move_history.append("SMOKE_BOMB")
            monster.next_move = "ESCAPE"
        elif monster.next_move == "ESCAPE":
            monster.move_history.append("ESCAPE")
            self._kill_monster(monster, escaped=True)
        else:
            raise NotImplementedError(f"native_sim_v3 {monster.monster_id} move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _gremlin_nob_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "BELLOW":
            _append_power(monster, "Anger", int(monster.meta.get("anger_amount", 2)), misc=int(monster.meta.get("anger_amount", 2)))
            monster.meta["used_bellow"] = True
            monster.move_history.append("BELLOW")
        elif monster.next_move == "SKULL_BASH":
            self._monster_attack_player(monster, int(monster.meta.get("bash_damage", 6)))
            self._apply_player_debuff("Vulnerable", 2)
            monster.move_history.append("SKULL_BASH")
        elif monster.next_move == "BULL_RUSH":
            self._monster_attack_player(monster, int(monster.meta.get("rush_damage", 14)))
            monster.move_history.append("BULL_RUSH")
        else:
            raise NotImplementedError(f"native_sim_v3 Gremlin Nob move {monster.next_move!r} is not implemented yet.")
        if not bool(monster.meta.get("used_bellow", False)):
            monster.next_move = "BELLOW"
        else:
            num = int(self.randoms.stream("ai").random(99))
            if self.ascension_level >= 18:
                if not (monster.move_history[-1:] == ["SKULL_BASH"] or monster.move_history[-2:-1] == ["SKULL_BASH"]):
                    monster.next_move = "SKULL_BASH"
                elif len(monster.move_history) >= 2 and monster.move_history[-1] == "BULL_RUSH" and monster.move_history[-2] == "BULL_RUSH":
                    monster.next_move = "SKULL_BASH"
                else:
                    monster.next_move = "SKULL_BASH" if num < 33 else "BULL_RUSH"
            else:
                if len(monster.move_history) >= 2 and monster.move_history[-1] == "BULL_RUSH" and monster.move_history[-2] == "BULL_RUSH":
                    monster.next_move = "SKULL_BASH"
                else:
                    monster.next_move = "SKULL_BASH" if num < 33 else "BULL_RUSH"
        _decrement_turn_powers(monster)

    def _lagavulin_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "SLEEP":
            monster.meta["idle_count"] = int(monster.meta.get("idle_count", 0)) + 1
            if int(monster.meta["idle_count"]) >= 3:
                monster.meta["opened"] = True
                monster.meta["asleep"] = False
                metallicize = _get_power_amount(monster, "Metallicize")
                if metallicize > 8:
                    _set_power_amount(monster, "Metallicize", metallicize - 8)
                elif metallicize > 0:
                    _remove_power(monster, "Metallicize")
                monster.next_move = "STRONG_ATTACK"
            else:
                monster.next_move = "SLEEP"
        elif monster.next_move == "STUN":
            monster.move_history.append("STUN")
            monster.next_move = "STRONG_ATTACK"
        elif monster.next_move == "DEBUFF":
            amount = int(monster.meta.get("debuff_amount", -1))
            self._apply_player_debuff("Dexterity", amount)
            self._apply_player_debuff("Strength", amount)
            monster.meta["debuff_turn_count"] = 0
            monster.move_history.append("DEBUFF")
            monster.next_move = "STRONG_ATTACK"
        elif monster.next_move == "STRONG_ATTACK":
            self._monster_attack_player(monster, int(monster.meta.get("attack_damage", 18)))
            monster.meta["debuff_turn_count"] = int(monster.meta.get("debuff_turn_count", 0)) + 1
            monster.move_history.append("STRONG_ATTACK")
            if int(monster.meta.get("debuff_turn_count", 0)) >= 2 or (len(monster.move_history) >= 2 and monster.move_history[-1] == "STRONG_ATTACK" and monster.move_history[-2] == "STRONG_ATTACK"):
                monster.next_move = "DEBUFF"
            else:
                monster.next_move = "STRONG_ATTACK"
        else:
            raise NotImplementedError(f"native_sim_v3 Lagavulin move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _sentry_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "BOLT":
            self.state.discard_pile.extend([make_card("Dazed", uuid=f"temp-dazed-{len(self.state.discard_pile)+i}") for i in range(int(monster.meta.get("dazed_amount", 2)))])
            monster.move_history.append("BOLT")
            monster.next_move = "BEAM"
        elif monster.next_move == "BEAM":
            self._monster_attack_player(monster, int(monster.meta.get("beam_damage", 9)))
            monster.move_history.append("BEAM")
            monster.next_move = "BOLT"
        else:
            raise NotImplementedError(f"native_sim_v3 Sentry move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _repulsor_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "ATTACK":
            self._monster_attack_player(monster, int(monster.meta.get("attack_damage", 11)))
            monster.move_history.append("ATTACK")
        elif monster.next_move == "DAZE":
            for index in range(int(monster.meta.get("dazed_amount", 2))):
                self._add_card_to_draw_pile_random_spot(
                    make_card("Dazed", uuid=f"repulsor-dazed-{self.state.turn}-{index}-{len(self.state.draw_pile)}")
                )
            monster.move_history.append("DAZE")
        else:
            raise NotImplementedError(f"native_sim_v3 Repulsor move {monster.next_move!r} is not implemented yet.")
        next_roll = int(self.randoms.stream("ai").random(99))
        if next_roll < 20 and (not monster.move_history or monster.move_history[-1] != "ATTACK"):
            monster.next_move = "ATTACK"
        else:
            monster.next_move = "DAZE"
        _decrement_turn_powers(monster)

    def _spiker_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "ATTACK":
            self._monster_attack_player(monster, int(monster.meta.get("attack_damage", 7)))
            monster.move_history.append("ATTACK")
        elif monster.next_move == "BUFF_THORNS":
            monster.meta["thorns_count"] = int(monster.meta.get("thorns_count", 0)) + 1
            _add_power(monster, "Thorns", int(monster.meta.get("buff_thorns", 2)))
            monster.move_history.append("BUFF_THORNS")
        else:
            raise NotImplementedError(f"native_sim_v3 Spiker move {monster.next_move!r} is not implemented yet.")
        if int(monster.meta.get("thorns_count", 0)) > 5:
            monster.next_move = "ATTACK"
        else:
            next_roll = int(self.randoms.stream("ai").random(99))
            if next_roll < 50 and (not monster.move_history or monster.move_history[-1] != "ATTACK"):
                monster.next_move = "ATTACK"
            else:
                monster.next_move = "BUFF_THORNS"
        _decrement_turn_powers(monster)

    def _exploder_take_turn(self, monster: MonsterState) -> None:
        monster.meta["turn_count"] = int(monster.meta.get("turn_count", 0)) + 1
        if monster.next_move == "ATTACK":
            self._monster_attack_player(monster, int(monster.meta.get("attack_damage", 9)))
        elif monster.next_move != "BLOCK":
            raise NotImplementedError(f"native_sim_v3 Exploder move {monster.next_move!r} is not implemented yet.")
        explosive = _get_power_amount(monster, "Explosive")
        if explosive <= 1 and _alive(monster):
            self._kill_monster(monster)
            self._damage_player(int(monster.meta.get("explosive_damage", 30)), attacker=monster, damage_type="THORNS")
            _decrement_turn_powers(monster)
            return
        if explosive > 0:
            _set_power_amount(monster, "Explosive", explosive - 1)
        self.randoms.stream("ai").random(99)
        monster.next_move = "ATTACK" if int(monster.meta.get("turn_count", 0)) < 2 else "BLOCK"
        _decrement_turn_powers(monster)

    def _giant_head_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "GLARE":
            self._apply_player_debuff("Weak", 1)
            monster.move_history.append("GLARE")
        elif monster.next_move == "COUNT":
            self._monster_attack_player(monster, int(monster.meta.get("count_damage", 13)))
            monster.move_history.append("COUNT")
        elif monster.next_move == "IT_IS_TIME":
            count = int(monster.meta.get("count", 0))
            index = 1 - count
            if index > 7:
                index = 7
            damage = int(monster.meta.get("starting_death_damage", 30)) + max(0, index - 1) * 5
            self._monster_attack_player(monster, damage)
            monster.move_history.append("IT_IS_TIME")
        else:
            raise NotImplementedError(f"native_sim_v3 Giant Head move {monster.next_move!r} is not implemented yet.")
        _giant_head_roll_next_move(monster, self.randoms)
        _decrement_turn_powers(monster)

    def _nemesis_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "SCYTHE":
            self._damage_player(int(monster.meta.get("scythe_damage", 45)), attacker=monster)
            monster.move_history.append("SCYTHE")
        elif monster.next_move == "TRI_ATTACK":
            damage = int(monster.meta.get("fire_damage", 6))
            for _ in range(3):
                self._damage_player(damage, attacker=monster)
            monster.move_history.append("TRI_ATTACK")
        elif monster.next_move == "TRI_BURN":
            for burn_index in range(int(monster.meta.get("burn_amount", 3))):
                self.state.discard_pile.append(make_card("Burn", uuid=f"nemesis-burn-{self.state.turn}-{burn_index}-{len(self.state.discard_pile)}"))
            monster.move_history.append("TRI_BURN")
        else:
            raise NotImplementedError(f"native_sim_v3 Nemesis move {monster.next_move!r} is not implemented yet.")
        if _get_power_amount(monster, "Intangible") <= 0:
            _add_power(monster, "Intangible", 1)
            _mark_power_just_applied(monster, "Intangible")
        _nemesis_roll_next_move(monster, self.randoms)
        _decrement_turn_powers(monster)

    def _awakened_one_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "SLASH":
            monster.meta["first_turn"] = False
            self._monster_attack_player(monster, int(monster.meta.get("slash_damage", 20)))
            monster.move_history.append("SLASH")
        elif monster.next_move == "SOUL_STRIKE":
            self._monster_attack_player(monster, int(monster.meta.get("soul_strike_damage", 6)), hits=4)
            monster.move_history.append("SOUL_STRIKE")
        elif monster.next_move == "REBIRTH":
            monster.current_hp = int(monster.max_hp)
            monster.meta["half_dead"] = False
            monster.move_history.append("REBIRTH")
        elif monster.next_move == "DARK_ECHO":
            monster.meta["first_turn"] = False
            self._monster_attack_player(monster, int(monster.meta.get("dark_echo_damage", 40)))
            monster.move_history.append("DARK_ECHO")
        elif monster.next_move == "SLUDGE":
            self._monster_attack_player(monster, int(monster.meta.get("sludge_damage", 18)))
            self.state.draw_pile.append(make_card("Void", uuid=f"awakened-void-{self.state.turn}-{len(self.state.draw_pile)}"))
            monster.move_history.append("SLUDGE")
        elif monster.next_move == "TACKLE":
            self._monster_attack_player(monster, int(monster.meta.get("tackle_damage", 10)), hits=3)
            monster.move_history.append("TACKLE")
        else:
            raise NotImplementedError(f"native_sim_v3 Awakened One move {monster.next_move!r} is not implemented yet.")
        if bool(monster.meta.get("half_dead", False)) and int(monster.current_hp) <= 0:
            # Awakened One can hit zero from thorns during its own attack.
            # The source damage() path queues SetMove(3) for REBIRTH; do not
            # overwrite it with the normal RollMoveAction at the end of takeTurn.
            monster.next_move = "REBIRTH"
            monster.intent = "UNKNOWN"
            monster.move_adjusted_damage = 0
            monster.move_hits = 0
            return
        _awakened_one_roll_next_move(monster, self.randoms)
        _decrement_turn_powers(monster)

    def _darkling_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "CHOMP":
            damage = int(monster.meta.get("chomp_damage", 8))
            self._monster_attack_player(monster, damage, hits=2)
            monster.move_history.append("CHOMP")
        elif monster.next_move == "HARDEN":
            _gain_block(monster, 12)
            if self.ascension_level >= 17:
                _add_power(monster, "Strength", 2)
            monster.move_history.append("HARDEN")
        elif monster.next_move == "NIP":
            self._monster_attack_player(monster, int(monster.meta.get("nip_damage", 7)))
            monster.move_history.append("NIP")
        elif monster.next_move == "COUNT":
            monster.move_history.append("COUNT")
        elif monster.next_move == "REINCARNATE":
            monster.current_hp = max(1, int(monster.max_hp) // 2)
            monster.meta["half_dead"] = False
            _append_power(monster, "Life Link", -1)
            self._apply_spawn_relic_effects(monster)
            monster.move_history.append("REINCARNATE")
        else:
            raise NotImplementedError(f"native_sim_v3 Darkling move {monster.next_move!r} is not implemented yet.")
        _darkling_roll_next_move(monster, self.randoms, self.state.monsters, self.ascension_level)
        _decrement_turn_powers(monster)

    def _snake_dagger_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "WOUND":
            self._damage_player(int(monster.meta.get("wound_damage", 9)), attacker=monster)
            self.state.discard_pile.append(make_card("Wound", uuid=f"snake-dagger-wound-{self.state.turn}-{len(self.state.discard_pile)}"))
            monster.move_history.append("WOUND")
        elif monster.next_move == "EXPLODE":
            self._damage_player(int(monster.meta.get("explode_damage", 25)), attacker=monster)
            self._kill_monster(monster)
            monster.move_history.append("EXPLODE")
        else:
            raise NotImplementedError(f"native_sim_v3 Snake Dagger move {monster.next_move!r} is not implemented yet.")
        monster.next_move = "EXPLODE"
        _decrement_turn_powers(monster)

    def _reptomancer_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "SNAKE_STRIKE":
            damage = int(monster.meta.get("snake_strike_damage", 13))
            self._damage_player(damage, attacker=monster)
            self._damage_player(damage, attacker=monster)
            self._apply_player_debuff("Weak", 1)
            monster.move_history.append("SNAKE_STRIKE")
        elif monster.next_move == "SPAWN_DAGGER":
            daggers_spawned = 0
            daggers_per_spawn = int(monster.meta.get("daggers_per_spawn", 1))
            for slot in range(4):
                if daggers_spawned >= daggers_per_spawn or not _reptomancer_can_spawn(self.state.monsters, monster):
                    break
                slot_alive = any(
                    current.monster_id == "Dagger"
                    and int(current.meta.get("reptomancer_slot", -1)) == slot
                    and _alive(current)
                    for current in self.state.monsters
                )
                if slot_alive:
                    continue
                dagger = _spawn_snake_dagger(self.randoms, self.ascension_level, reptomancer_slot=slot)
                self._insert_reptomancer_dagger(monster, dagger, slot)
                daggers_spawned += 1
            monster.move_history.append("SPAWN_DAGGER")
        elif monster.next_move == "BIG_BITE":
            self._damage_player(int(monster.meta.get("big_bite_damage", 30)), attacker=monster)
            monster.move_history.append("BIG_BITE")
        else:
            raise NotImplementedError(f"native_sim_v3 Reptomancer move {monster.next_move!r} is not implemented yet.")
        _reptomancer_roll_next_move(monster, self.randoms, self.state.monsters)
        _decrement_turn_powers(monster)

    def _time_eater_take_turn(self, monster: MonsterState) -> None:
        monster.meta["first_turn"] = False
        if monster.next_move == "REVERBERATE":
            self._monster_attack_player(monster, int(monster.meta.get("reverberate_damage", 7)), hits=3)
            monster.move_history.append("REVERBERATE")
        elif monster.next_move == "RIPPLE":
            _gain_block(monster, 20)
            self._apply_player_debuff("Vulnerable", 1)
            self._apply_player_debuff("Weak", 1)
            if bool(monster.meta.get("asc19", False)):
                self._apply_player_debuff("Frail", 1)
            monster.move_history.append("RIPPLE")
        elif monster.next_move == "HEAD_SLAM":
            self._monster_attack_player(monster, int(monster.meta.get("head_slam_damage", 26)))
            self._apply_player_debuff("Draw Reduction", 1)
            if bool(monster.meta.get("asc19", False)):
                self.state.discard_pile.extend(
                    [make_card("Slimed", uuid=f"time-eater-slimed-{len(self.state.discard_pile)+i}") for i in range(2)]
                )
            monster.move_history.append("HEAD_SLAM")
        elif monster.next_move == "HASTE":
            _remove_monster_debuffs(monster)
            heal_target = max(int(monster.current_hp), int(monster.max_hp) // 2)
            monster.current_hp = heal_target
            if int(monster.meta.get("haste_block", 0)) > 0:
                _gain_block(monster, int(monster.meta.get("haste_block", 0)))
            monster.move_history.append("HASTE")
        else:
            raise NotImplementedError(f"native_sim_v3 Time Eater move {monster.next_move!r} is not implemented yet.")
        _time_eater_roll_next_move(monster, self.randoms)
        _decrement_turn_powers(monster)

    def _donu_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "BEAM":
            damage = int(monster.meta.get("beam_damage", 10))
            self._damage_player(damage, attacker=monster)
            self._damage_player(damage, attacker=monster)
            monster.meta["is_attacking"] = False
            monster.move_history.append("BEAM")
        elif monster.next_move in {"CIRCLE", "CIRCLE_OF_PROTECTION"}:
            for current in self.state.monsters:
                if _alive(current):
                    _add_power(current, "Strength", 3)
            monster.meta["is_attacking"] = True
            monster.move_history.append("CIRCLE")
        else:
            raise NotImplementedError(f"native_sim_v3 Donu move {monster.next_move!r} is not implemented yet.")
        monster.next_move = "BEAM" if bool(monster.meta.get("is_attacking", False)) else "CIRCLE"
        _decrement_turn_powers(monster)

    def _deca_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "BEAM":
            damage = int(monster.meta.get("beam_damage", 10))
            self._damage_player(damage, attacker=monster)
            self._damage_player(damage, attacker=monster)
            self.state.discard_pile.extend([make_card("Dazed", uuid=f"deca-dazed-{len(self.state.discard_pile)+i}") for i in range(2)])
            monster.meta["is_attacking"] = False
            monster.move_history.append("BEAM")
        elif monster.next_move in {"SQUARE", "SQUARE_OF_PROTECTION"}:
            for current in self.state.monsters:
                if not _alive(current):
                    continue
                _gain_block(current, 16)
                if bool(monster.meta.get("asc19", False)):
                    _add_power(current, "Plated Armor", 3)
            monster.meta["is_attacking"] = True
            monster.move_history.append("SQUARE")
        else:
            raise NotImplementedError(f"native_sim_v3 Deca move {monster.next_move!r} is not implemented yet.")
        monster.next_move = "BEAM" if bool(monster.meta.get("is_attacking", False)) else "SQUARE"
        _decrement_turn_powers(monster)

    def _transient_take_turn(self, monster: MonsterState) -> None:
        fading = _get_power_amount(monster, "Fading")
        if fading <= 1:
            self._kill_monster(monster)
            _remove_power(monster, "Fading")
            return
        damage = int(monster.move_adjusted_damage or 0)
        if damage <= 0:
            damage = int(monster.meta.get("starting_damage", 30)) + 10 * int(monster.meta.get("attack_index", 0))
        self._damage_player(damage, attacker=monster)
        monster.move_history.append("ATTACK")
        monster.meta["attack_index"] = min(6, int(monster.meta.get("attack_index", 0)) + 1)
        _set_power_amount(monster, "Fading", fading - 1)
        _decrement_turn_powers(monster)

    def _maw_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "ROAR":
            self._apply_player_debuff("Weak", int(monster.meta.get("terrify_duration", 3)))
            self._apply_player_debuff("Frail", int(monster.meta.get("terrify_duration", 3)))
            monster.meta["roared"] = True
            monster.move_history.append("ROAR")
        elif monster.next_move == "SLAM":
            self._monster_attack_player(monster, int(monster.meta.get("slam_damage", 25)))
            monster.move_history.append("SLAM")
        elif monster.next_move == "DROOL":
            _add_power(monster, "Strength", int(monster.meta.get("str_up", 3)))
            monster.move_history.append("DROOL")
        elif monster.next_move == "NOM":
            hits = max(1, int(monster.meta.get("turn_count", 1)) // 2)
            self._monster_attack_player(monster, int(monster.meta.get("nom_damage", 5)), hits=hits)
            monster.move_history.append("NOM")
        else:
            raise NotImplementedError(f"native_sim_v3 Maw move {monster.next_move!r} is not implemented yet.")
        _maw_roll_next_move(monster, self.randoms)
        _decrement_turn_powers(monster)

    def _spire_growth_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "QUICK_TACKLE":
            self._damage_player(int(monster.meta.get("tackle_damage", 16)), attacker=monster)
            monster.move_history.append("QUICK_TACKLE")
        elif monster.next_move == "CONSTRICT":
            self._apply_player_debuff("Constricted", int(monster.meta.get("constrict_damage", 10)))
            monster.move_history.append("CONSTRICT")
        elif monster.next_move == "SMASH":
            self._damage_player(int(monster.meta.get("smash_damage", 22)), attacker=monster)
            monster.move_history.append("SMASH")
        else:
            raise NotImplementedError(f"native_sim_v3 Spire Growth move {monster.next_move!r} is not implemented yet.")
        _spire_growth_roll_next_move(monster, self.randoms, self.player, self.ascension_level)
        _decrement_turn_powers(monster)

    def _writhing_mass_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "BIG_HIT":
            self._monster_attack_player(monster, int(monster.meta.get("big_hit_damage", 32)))
            monster.move_history.append("BIG_HIT")
        elif monster.next_move == "MULTI_HIT":
            damage = int(monster.meta.get("multi_hit_damage", 7))
            self._monster_attack_player(monster, damage, hits=3)
            monster.move_history.append("MULTI_HIT")
        elif monster.next_move == "ATTACK_BLOCK":
            damage = int(monster.meta.get("attack_block_damage", 15))
            self._monster_attack_player(monster, damage)
            _gain_block(monster, damage)
            monster.move_history.append("ATTACK_BLOCK")
        elif monster.next_move == "ATTACK_DEBUFF":
            damage = int(monster.meta.get("attack_debuff_damage", 10))
            self._monster_attack_player(monster, damage)
            debuff = int(monster.meta.get("normal_debuff_amount", 2))
            self._apply_player_debuff("Weak", debuff)
            self._apply_player_debuff("Vulnerable", debuff)
            monster.move_history.append("ATTACK_DEBUFF")
        elif monster.next_move == "MEGA_DEBUFF":
            monster.meta["used_mega_debuff"] = True
            parasite = make_card("Parasite", uuid=f"writhing-mass-parasite-{len(self.master_deck)}")
            self.master_deck.append(_copy_card(parasite))
            monster.move_history.append("MEGA_DEBUFF")
        else:
            raise NotImplementedError(f"native_sim_v3 Writhing Mass move {monster.next_move!r} is not implemented yet.")
        _writhing_mass_roll_next_move(monster, self.randoms)
        _decrement_turn_powers(monster)

    def _slime_boss_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "GOOP_SPRAY":
            self.state.discard_pile.extend(
                [make_card("Slimed", uuid=f"boss-slimed-{len(self.state.discard_pile)+i}") for i in range(int(monster.meta.get("sticky_slimed", 3)))]
            )
            monster.move_history.append("GOOP_SPRAY")
            monster.next_move = "PREP_SLAM"
        elif monster.next_move == "PREP_SLAM":
            monster.move_history.append("PREP_SLAM")
            monster.next_move = "SLAM"
        elif monster.next_move == "SLAM":
            self._monster_attack_player(monster, int(monster.meta.get("slam_damage", 35)))
            monster.move_history.append("SLAM")
            if not self._force_pending_slime_split_move(monster):
                monster.next_move = "GOOP_SPRAY"
        elif monster.next_move == "SPLIT":
            hp = max(1, int(monster.current_hp))
            self._kill_monster(monster, suppress_victory=True)
            children = [
                _spawn_spike_slime_l(self.randoms, self.ascension_level, hp_override=hp, draw_x=-385.0),
                _spawn_acid_slime_l(self.randoms, self.ascension_level, hp_override=hp, draw_x=120.0),
            ]
            for child in children:
                self._roll_spawned_slime_initial_move(child)
            self._insert_split_children_around_parent(
                monster,
                children,
            )
            self._update_monster_intents()
            return
        else:
            raise NotImplementedError(f"native_sim_v3 Slime Boss move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _guardian_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "CHARGE_UP":
            _gain_block(monster, int(monster.meta.get("charge_block", 9)))
            monster.move_history.append("CHARGE_UP")
            monster.next_move = "FIERCE_BASH"
        elif monster.next_move == "FIERCE_BASH":
            monster.move_history.append("FIERCE_BASH")
            monster.next_move = "VENT_STEAM"
            self._monster_attack_player(monster, int(monster.meta.get("fierce_bash_damage", 32)))
        elif monster.next_move == "VENT_STEAM":
            debuff = int(monster.meta.get("vent_debuff", 2))
            self._apply_player_debuff("Weak", debuff)
            self._apply_player_debuff("Vulnerable", debuff)
            monster.move_history.append("VENT_STEAM")
            monster.next_move = "WHIRLWIND"
        elif monster.next_move == "WHIRLWIND":
            monster.move_history.append("WHIRLWIND")
            monster.next_move = "CHARGE_UP"
            self._monster_attack_player(
                monster,
                int(monster.meta.get("whirlwind_damage", 5)),
                hits=int(monster.meta.get("whirlwind_hits", 4)),
            )
        elif monster.next_move == "CLOSE_UP":
            _add_power(monster, "Sharp Hide", int(monster.meta.get("thorns_damage", 3)))
            monster.move_history.append("CLOSE_UP")
            monster.next_move = "ROLL_ATTACK"
        elif monster.next_move == "ROLL_ATTACK":
            monster.move_history.append("ROLL_ATTACK")
            monster.next_move = "TWIN_SLAM"
            self._monster_attack_player(monster, int(monster.meta.get("roll_damage", 9)))
        elif monster.next_move == "TWIN_SLAM":
            monster.move_history.append("TWIN_SLAM")
            monster.next_move = "WHIRLWIND"
            monster.meta["is_open"] = True
            monster.meta["close_up_triggered"] = False
            self._monster_attack_player(
                monster,
                int(monster.meta.get("twin_slam_damage", 8)),
                hits=2,
            )
            _remove_power(monster, "Sharp Hide")
            monster.meta["dmg_taken"] = 0
            _append_power(monster, "Mode Shift", int(monster.meta.get("mode_shift_threshold", 30)), misc=int(monster.meta.get("mode_shift_threshold", 30)))
            if int(monster.block) > 0:
                monster.block = 0
        else:
            raise NotImplementedError(f"native_sim_v3 The Guardian move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _hexaghost_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "ACTIVATE":
            monster.meta["activated"] = True
            monster.meta["divider_damage"] = int(self.player.current_hp) // 12 + 1
            monster.move_history.append("ACTIVATE")
            monster.next_move = "DIVIDER"
        elif monster.next_move == "DIVIDER":
            damage = int(monster.meta.get("divider_damage", 0))
            self._monster_attack_player(monster, damage, hits=int(monster.meta.get("divider_hits", 6)))
            monster.meta["orb_active_count"] = 0
            monster.move_history.append("DIVIDER")
            monster.next_move = "SEAR"
        elif monster.next_move == "TACKLE":
            damage = int(monster.meta.get("tackle_damage", 5))
            self._monster_attack_player(monster, damage, hits=int(monster.meta.get("tackle_hits", 2)))
            monster.meta["orb_active_count"] = int(monster.meta.get("orb_active_count", 0)) + 1
            monster.move_history.append("TACKLE")
            if int(monster.meta.get("orb_active_count", 0)) in {2, 5}:
                monster.next_move = "SEAR"
        elif monster.next_move == "SEAR":
            self._monster_attack_player(monster, int(monster.meta.get("sear_damage", 6)))
            burn_upgrades = 1 if bool(monster.meta.get("burn_upgraded", False)) else 0
            for _ in range(int(monster.meta.get("sear_burns", 1))):
                self.state.discard_pile.append(
                    make_card("Burn", upgrades=burn_upgrades, uuid=f"hexaghost-burn-{len(self.state.discard_pile)}")
                )
            monster.meta["orb_active_count"] = int(monster.meta.get("orb_active_count", 0)) + 1
            monster.move_history.append("SEAR")
            count = int(monster.meta.get("orb_active_count", 0))
            if count == 1:
                monster.next_move = "TACKLE"
            elif count == 3:
                monster.next_move = "INFLAME"
            elif count == 6:
                monster.next_move = "INFERNO"
        elif monster.next_move == "INFLAME":
            _gain_block(monster, int(monster.meta.get("strengthen_block", 12)))
            _add_power(monster, "Strength", int(monster.meta.get("strength_gain", 2)))
            monster.meta["orb_active_count"] = int(monster.meta.get("orb_active_count", 0)) + 1
            monster.move_history.append("INFLAME")
            monster.next_move = "TACKLE"
        elif monster.next_move == "INFERNO":
            damage = int(monster.meta.get("inferno_damage", 2))
            self._monster_attack_player(monster, damage, hits=int(monster.meta.get("inferno_hits", 6)))
            for pile_name in ("draw_pile", "discard_pile"):
                pile = getattr(self.state, pile_name)
                for burn in pile:
                    if burn.get("card_id") == "Burn":
                        burn["upgrades"] = max(1, int(burn.get("upgrades") or 0))
                        burn["base_magic"] = max(int(burn.get("base_magic") or 0), 4)
            burn_base_index = len(self.state.discard_pile)
            for burn_index in range(3):
                self.state.discard_pile.append(make_card("Burn", upgrades=1, uuid=f"hexaghost-inferno-burn-{burn_base_index + burn_index}"))
            monster.meta["burn_upgraded"] = True
            monster.meta["orb_active_count"] = 0
            monster.move_history.append("INFERNO")
            monster.next_move = "SEAR"
        else:
            raise NotImplementedError(f"native_sim_v3 Hexaghost move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _byrd_roll_next_move(self, monster: MonsterState) -> None:
        if not bool(monster.meta.get("is_flying", True)):
            monster.next_move = "HEADBUTT"
            return
        num = int(self.randoms.stream("ai").random(99))
        last_move = monster.move_history[-1] if monster.move_history else None
        last_two = monster.move_history[-2:]
        if num < 50:
            if len(last_two) == 2 and last_two[0] == "PECK" and last_two[1] == "PECK":
                monster.next_move = "SWOOP" if self.randoms.stream("ai").random_boolean(0.4) else "CAW"
            else:
                monster.next_move = "PECK"
        elif num < 70:
            if last_move == "SWOOP":
                monster.next_move = "CAW" if self.randoms.stream("ai").random_boolean(0.375) else "PECK"
            else:
                monster.next_move = "SWOOP"
        elif last_move == "CAW":
            monster.next_move = "SWOOP" if self.randoms.stream("ai").random_boolean(0.2857) else "PECK"
        else:
            monster.next_move = "CAW"

    def _byrd_take_turn(self, monster: MonsterState) -> None:
        if bool(monster.meta.get("is_flying", True)):
            flight = _get_power_amount(monster, "Flight")
            stored = int(monster.meta.get("flight_amount", flight))
            if flight != stored:
                _set_power_amount(monster, "Flight", stored)
        if monster.next_move == "PECK":
            damage = int(monster.meta.get("peck_damage", 1))
            for _ in range(int(monster.meta.get("peck_count", 5))):
                self._monster_attack_player(monster, damage)
            monster.move_history.append("PECK")
            self._byrd_roll_next_move(monster)
        elif monster.next_move == "CAW":
            _add_power(monster, "Strength", int(monster.meta.get("caw_strength", 1)))
            monster.move_history.append("CAW")
            self._byrd_roll_next_move(monster)
        elif monster.next_move == "SWOOP":
            self._monster_attack_player(monster, int(monster.meta.get("swoop_damage", 12)))
            monster.move_history.append("SWOOP")
            self._byrd_roll_next_move(monster)
        elif monster.next_move == "STUNNED":
            monster.move_history.append("STUNNED")
            # The real RollMoveAction still consumes aiRng.random(99) here even
            # though grounded Byrd deterministically chooses Headbutt.
            self.randoms.stream("ai").random(99)
            monster.next_move = "HEADBUTT"
        elif monster.next_move == "HEADBUTT":
            self._monster_attack_player(monster, int(monster.meta.get("headbutt_damage", 3)))
            monster.move_history.append("HEADBUTT")
            monster.next_move = "GO_AIRBORNE"
        elif monster.next_move == "GO_AIRBORNE":
            monster.meta["is_flying"] = True
            _set_power_amount(monster, "Flight", int(monster.meta.get("flight_amount", 3)))
            monster.move_history.append("GO_AIRBORNE")
            self._byrd_roll_next_move(monster)
        else:
            raise NotImplementedError(f"native_sim_v3 Byrd move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _snecko_roll_next_move(self, monster: MonsterState) -> None:
        num = int(self.randoms.stream("ai").random(99))
        if num < 40:
            monster.next_move = "TAIL"
        elif len(monster.move_history) >= 2 and monster.move_history[-1] == "BITE" and monster.move_history[-2] == "BITE":
            monster.next_move = "TAIL"
        else:
            monster.next_move = "BITE"

    def _snecko_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "GLARE":
            self._apply_player_debuff("Confusion", -1)
            monster.move_history.append("GLARE")
            self._snecko_roll_next_move(monster)
        elif monster.next_move == "BITE":
            self._monster_attack_player(monster, int(monster.meta.get("bite_damage", 15)))
            monster.move_history.append("BITE")
            self._snecko_roll_next_move(monster)
        elif monster.next_move == "TAIL":
            self._monster_attack_player(monster, int(monster.meta.get("tail_damage", 8)))
            if self.ascension_level >= 17:
                self._apply_player_debuff("Weak", 2)
            self._apply_player_debuff("Vulnerable", 2)
            monster.move_history.append("TAIL")
            self._snecko_roll_next_move(monster)
        else:
            raise NotImplementedError(f"native_sim_v3 Snecko move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _snake_plant_roll_next_move(self, monster: MonsterState) -> None:
        num = int(self.randoms.stream("ai").random(99))
        last_move = monster.move_history[-1] if monster.move_history else None
        last_two = monster.move_history[-2:]
        if num < 65:
            if len(last_two) == 2 and last_two[0] == "CHOMPY_CHOMPS" and last_two[1] == "CHOMPY_CHOMPS":
                monster.next_move = "SPORES"
            else:
                monster.next_move = "CHOMPY_CHOMPS"
        elif self.ascension_level >= 17 and (last_move == "SPORES" or (len(monster.move_history) >= 2 and monster.move_history[-2] == "SPORES")):
            monster.next_move = "CHOMPY_CHOMPS"
        elif last_move == "SPORES":
            monster.next_move = "CHOMPY_CHOMPS"
        else:
            monster.next_move = "SPORES"

    def _snake_plant_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "CHOMPY_CHOMPS":
            self._monster_attack_player(monster, int(monster.meta.get("chomp_damage", 7)), hits=3)
            monster.move_history.append("CHOMPY_CHOMPS")
        elif monster.next_move == "SPORES":
            self._apply_player_debuff("Frail", int(monster.meta.get("frail_amount", 2)))
            self._apply_player_debuff("Weak", int(monster.meta.get("weak_amount", 2)))
            monster.move_history.append("SPORES")
        else:
            raise NotImplementedError(f"native_sim_v3 SnakePlant move {monster.next_move!r} is not implemented yet.")
        self._snake_plant_roll_next_move(monster)
        _decrement_turn_powers(monster)

    def _shelled_parasite_roll_next_move(self, monster: MonsterState) -> None:
        num = int(self.randoms.stream("ai").random(99))
        if bool(monster.meta.get("first_move", False)):
            monster.meta["first_move"] = False
            if self.ascension_level >= 17:
                monster.next_move = "FELL"
            elif self.randoms.stream("ai").random_boolean():
                monster.next_move = "DOUBLE_STRIKE"
            else:
                monster.next_move = "LIFE_SUCK"
            return
        reroll = 100
        if num < 20:
            if monster.move_history[-1:] != ["FELL"]:
                monster.next_move = "FELL"
                return
            else:
                # STS recurses with getMove(aiRng.random(20, 99)); the original
                # low roll must not keep forcing DOUBLE_STRIKE after reroll.
                num = int(self.randoms.stream("ai").random(20, 99))
        if num < 60:
            if not (len(monster.move_history) >= 2 and monster.move_history[-1] == "DOUBLE_STRIKE" and monster.move_history[-2] == "DOUBLE_STRIKE"):
                monster.next_move = "DOUBLE_STRIKE"
            else:
                monster.next_move = "LIFE_SUCK"
        elif not (len(monster.move_history) >= 2 and monster.move_history[-1] == "LIFE_SUCK" and monster.move_history[-2] == "LIFE_SUCK"):
            monster.next_move = "LIFE_SUCK"
        else:
            monster.next_move = "DOUBLE_STRIKE"

    def _shelled_parasite_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "FELL":
            self._monster_attack_player(monster, int(monster.meta.get("fell_damage", 18)))
            self._apply_player_debuff("Frail", int(monster.meta.get("frail_amount", 2)))
            monster.move_history.append("FELL")
        elif monster.next_move == "DOUBLE_STRIKE":
            damage = int(monster.meta.get("double_damage", 6))
            self._monster_attack_player(monster, damage, hits=2)
            monster.move_history.append("DOUBLE_STRIKE")
        elif monster.next_move == "LIFE_SUCK":
            damage = int(monster.meta.get("suck_damage", 10))
            dealt = self._monster_attack_player(monster, damage, defer_player_retaliation=True)
            monster.current_hp = min(int(monster.max_hp), int(monster.current_hp) + dealt)
            self._retaliate_player_powers(monster)
            monster.move_history.append("LIFE_SUCK")
        elif monster.next_move == "STUNNED":
            monster.move_history.append("STUNNED")
            monster.next_move = "FELL"
        else:
            raise NotImplementedError(f"native_sim_v3 ShelledParasite move {monster.next_move!r} is not implemented yet.")
        self._shelled_parasite_roll_next_move(monster)
        _decrement_turn_powers(monster)

    def _centurion_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "SLASH":
            self._monster_attack_player(monster, int(monster.meta.get("slash_damage", 12)))
            monster.move_history.append("SLASH")
        elif monster.next_move == "PROTECT":
            valid_targets = [current for current in self.state.monsters if current is not monster and _alive(current)]
            if valid_targets:
                target = valid_targets[int(self.randoms.stream("ai").random(0, len(valid_targets) - 1))]
            else:
                target = monster
            _gain_block(target, int(monster.meta.get("block_amount", 15)))
            monster.move_history.append("PROTECT")
        elif monster.next_move == "FURY":
            damage = int(monster.meta.get("fury_damage", 6))
            self._monster_attack_player(monster, damage, hits=int(monster.meta.get("fury_hits", 3)))
            monster.move_history.append("FURY")
        else:
            raise NotImplementedError(f"native_sim_v3 Centurion move {monster.next_move!r} is not implemented yet.")
        num = int(self.randoms.stream("ai").random(99))
        allies_alive = sum(1 for current in self.state.monsters if _alive(current))
        if num >= 65 and monster.move_history[-2:] != ["PROTECT", "PROTECT"] and monster.move_history[-2:] != ["FURY", "FURY"]:
            monster.next_move = "PROTECT" if allies_alive > 1 else "FURY"
        elif monster.move_history[-2:] != ["SLASH", "SLASH"]:
            monster.next_move = "SLASH"
        else:
            monster.next_move = "PROTECT" if allies_alive > 1 else "FURY"
        _decrement_turn_powers(monster)

    def _healer_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "ATTACK":
            self._monster_attack_player(monster, int(monster.meta.get("magic_damage", 8)))
            self._apply_player_debuff("Frail", int(monster.meta.get("frail_amount", 2)))
            monster.move_history.append("ATTACK")
        elif monster.next_move == "HEAL":
            for current in self.state.monsters:
                if _alive(current):
                    current.current_hp = min(int(current.max_hp), int(current.current_hp) + int(monster.meta.get("heal_amount", 16)))
            monster.move_history.append("HEAL")
        elif monster.next_move == "BUFF":
            for current in self.state.monsters:
                if _alive(current):
                    _add_power(current, "Strength", int(monster.meta.get("strength_amount", 2)))
            monster.move_history.append("BUFF")
        else:
            raise NotImplementedError(f"native_sim_v3 Healer move {monster.next_move!r} is not implemented yet.")
        need_to_heal = sum(max(0, int(current.max_hp) - int(current.current_hp)) for current in self.state.monsters if _alive(current))
        num = int(self.randoms.stream("ai").random(99))
        heal_threshold = 20 if self.ascension_level >= 17 else 15
        if need_to_heal > heal_threshold and not (len(monster.move_history) >= 2 and monster.move_history[-1] == "HEAL" and monster.move_history[-2] == "HEAL"):
            monster.next_move = "HEAL"
        elif ((self.ascension_level >= 17 and num >= 40 and monster.move_history[-1:] != ["ATTACK"]) or (self.ascension_level < 17 and num >= 40 and not (len(monster.move_history) >= 2 and monster.move_history[-1] == "ATTACK" and monster.move_history[-2] == "ATTACK"))):
            monster.next_move = "ATTACK"
        elif not (len(monster.move_history) >= 2 and monster.move_history[-1] == "BUFF" and monster.move_history[-2] == "BUFF"):
            monster.next_move = "BUFF"
        else:
            monster.next_move = "ATTACK"
        _decrement_turn_powers(monster)

    def _champ_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "ANGER":
            _remove_monster_debuffs(monster)
            _add_power(monster, "Strength", int(monster.meta.get("str_amt", 2)) * 3)
            monster.meta["threshold_reached"] = True
            monster.move_history.append("ANGER")
        elif monster.next_move == "HEAVY_SLASH":
            self._monster_attack_player(monster, int(monster.meta.get("slash_damage", 16)))
            monster.move_history.append("HEAVY_SLASH")
        elif monster.next_move == "DEFENSIVE_STANCE":
            _gain_block(monster, int(monster.meta.get("block_amt", 15)))
            _add_power(monster, "Metallicize", int(monster.meta.get("forge_amt", 5)))
            monster.move_history.append("DEFENSIVE_STANCE")
        elif monster.next_move == "EXECUTE":
            damage = int(monster.meta.get("execute_damage", 10))
            self._monster_attack_player(monster, damage, hits=2)
            monster.move_history.append("EXECUTE")
        elif monster.next_move == "FACE_SLAP":
            self._monster_attack_player(monster, int(monster.meta.get("slap_damage", 12)))
            self._apply_player_debuff("Frail", 2)
            self._apply_player_debuff("Vulnerable", 2)
            monster.move_history.append("FACE_SLAP")
        elif monster.next_move == "GLOAT":
            _add_power(monster, "Strength", int(monster.meta.get("str_amt", 2)))
            monster.move_history.append("GLOAT")
        elif monster.next_move == "TAUNT":
            self._apply_player_debuff("Weak", 2)
            self._apply_player_debuff("Vulnerable", 2)
            monster.meta["num_turns"] = 0
            monster.move_history.append("TAUNT")
        else:
            raise NotImplementedError(f"native_sim_v3 Champ move {monster.next_move!r} is not implemented yet.")
        # RollMoveAction passes AbstractDungeon.aiRng.random(99) into getMove()
        # even when Champ's forced branches ignore the roll.
        num = int(self.randoms.stream("ai").random(99))
        num_turns = int(monster.meta.get("num_turns", 0)) + 1
        monster.meta["num_turns"] = num_turns
        last = monster.move_history[-1] if monster.move_history else None
        if int(monster.current_hp) < int(monster.max_hp) // 2 and not bool(monster.meta.get("threshold_reached", False)):
            monster.next_move = "ANGER"
            monster.intent = "BUFF"
            _decrement_turn_powers(monster)
            return
        if bool(monster.meta.get("threshold_reached", False)) and last != "EXECUTE" and (len(monster.move_history) < 2 or monster.move_history[-2] != "EXECUTE"):
            monster.next_move = "EXECUTE"
        elif num_turns == 4 and not bool(monster.meta.get("threshold_reached", False)):
            monster.next_move = "TAUNT"
            monster.meta["num_turns"] = 0
        else:
            if self.ascension_level >= 19:
                if last != "DEFENSIVE_STANCE" and int(monster.meta.get("forge_times", 0)) < int(monster.meta.get("forge_threshold", 2)) and num <= 30:
                    monster.meta["forge_times"] = int(monster.meta.get("forge_times", 0)) + 1
                    monster.next_move = "DEFENSIVE_STANCE"
                elif last not in {"GLOAT", "DEFENSIVE_STANCE"} and num <= 30:
                    monster.next_move = "GLOAT"
                elif last != "FACE_SLAP" and num <= 55:
                    monster.next_move = "FACE_SLAP"
                elif last != "HEAVY_SLASH":
                    monster.next_move = "HEAVY_SLASH"
                else:
                    monster.next_move = "FACE_SLAP"
            else:
                if last != "DEFENSIVE_STANCE" and int(monster.meta.get("forge_times", 0)) < int(monster.meta.get("forge_threshold", 2)) and num <= 15:
                    monster.meta["forge_times"] = int(monster.meta.get("forge_times", 0)) + 1
                    monster.next_move = "DEFENSIVE_STANCE"
                elif last not in {"GLOAT", "DEFENSIVE_STANCE"} and num <= 30:
                    monster.next_move = "GLOAT"
                elif last != "FACE_SLAP" and num <= 55:
                    monster.next_move = "FACE_SLAP"
                elif last != "HEAVY_SLASH":
                    monster.next_move = "HEAVY_SLASH"
                else:
                    monster.next_move = "FACE_SLAP"
        _decrement_turn_powers(monster)

    def _collector_minions_alive(self) -> int:
        return sum(1 for current in self.state.monsters if _alive(current) and current.monster_id == "TorchHead")

    def _collector_spawn_torch_heads(self) -> None:
        needed = max(0, 2 - self._collector_minions_alive())
        for offset in range(needed):
            # Collector's SpawnMonsterAction places Torch Heads to the left of
            # Collector; Communication Mod exposes monsters in draw-x order.
            draw_x = -185.0 * (offset + 1)
            self._insert_spawned_monster_by_draw_x(
                _spawn_torch_head(self.randoms, self.ascension_level, draw_x=draw_x)
            )

    def _collector_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "SPAWN":
            self._collector_spawn_torch_heads()
            monster.meta["initial_spawn"] = False
            monster.move_history.append("SPAWN")
        elif monster.next_move == "FIREBALL":
            self._monster_attack_player(monster, int(monster.meta.get("fireball_damage", 18)))
            monster.move_history.append("FIREBALL")
        elif monster.next_move == "BUFF":
            _gain_block(monster, int(monster.meta.get("block_amt", 15)) + (5 if self.ascension_level >= 19 else 0))
            for current in self.state.monsters:
                if _alive(current):
                    _add_power(current, "Strength", int(monster.meta.get("str_amt", 3)))
            monster.move_history.append("BUFF")
        elif monster.next_move == "MEGA_DEBUFF":
            amount = int(monster.meta.get("mega_debuff_amt", 3))
            self._apply_player_debuff("Weak", amount)
            self._apply_player_debuff("Vulnerable", amount)
            self._apply_player_debuff("Frail", amount)
            monster.meta["ult_used"] = True
            monster.move_history.append("MEGA_DEBUFF")
        elif monster.next_move == "REVIVE":
            self._collector_spawn_torch_heads()
            monster.move_history.append("REVIVE")
        else:
            raise NotImplementedError(f"native_sim_v3 Collector move {monster.next_move!r} is not implemented yet.")
        monster.meta["turns_taken"] = int(monster.meta.get("turns_taken", 0)) + 1
        num = int(self.randoms.stream("ai").random(99))
        if bool(monster.meta.get("initial_spawn", False)):
            monster.next_move = "SPAWN"
        elif int(monster.meta.get("turns_taken", 0)) >= 3 and not bool(monster.meta.get("ult_used", False)):
            monster.next_move = "MEGA_DEBUFF"
        elif num <= 25 and self._collector_minions_alive() < 2 and (not monster.move_history or monster.move_history[-1] != "REVIVE"):
            monster.next_move = "REVIVE"
        elif num <= 70 and not (len(monster.move_history) >= 2 and monster.move_history[-1] == "FIREBALL" and monster.move_history[-2] == "FIREBALL"):
            monster.next_move = "FIREBALL"
        elif not monster.move_history or monster.move_history[-1] != "BUFF":
            monster.next_move = "BUFF"
        else:
            monster.next_move = "FIREBALL"
        _decrement_turn_powers(monster)

    def _torch_head_take_turn(self, monster: MonsterState) -> None:
        self._monster_attack_player(monster, int(monster.meta.get("tackle_damage", 7)))
        monster.move_history.append("TACKLE")
        monster.next_move = "TACKLE"
        _decrement_turn_powers(monster)

    def _choose_bronze_orb_next_move(self, monster: MonsterState) -> None:
        num = int(self.randoms.stream("ai").random(99))
        if not bool(monster.meta.get("used_stasis", False)) and num >= 25:
            monster.next_move = "STASIS"
            monster.meta["used_stasis"] = True
        elif num >= 70 and not (len(monster.move_history) >= 2 and monster.move_history[-1] == "SUPPORT" and monster.move_history[-2] == "SUPPORT"):
            monster.next_move = "SUPPORT"
        elif not (len(monster.move_history) >= 2 and monster.move_history[-1] == "BEAM" and monster.move_history[-2] == "BEAM"):
            monster.next_move = "BEAM"
        else:
            monster.next_move = "SUPPORT"

    def _pop_stasis_card_from_pile(self, pile: list[dict[str, Any]]) -> dict[str, Any] | None:
        for rarity in ("RARE", "UNCOMMON", "COMMON"):
            candidates = sorted(
                (card for card in pile if str(card.get("rarity") or "") == rarity),
                key=lambda card: str(card.get("card_id") or card.get("id") or ""),
            )
            if not candidates:
                continue
            picked = candidates[int(self.randoms.stream("card_random").random(len(candidates) - 1))]
            pile.remove(picked)
            return picked
        if not pile:
            return None
        return pile.pop(int(self.randoms.stream("card_random").random(len(pile) - 1)))

    def _apply_bronze_orb_stasis(self, monster: MonsterState) -> None:
        if not self.state.draw_pile and not self.state.discard_pile:
            return
        source_pile = self.state.draw_pile if self.state.draw_pile else self.state.discard_pile
        card = self._pop_stasis_card_from_pile(source_pile)
        if card is None:
            return
        monster.powers.append(
            {
                "power_id": "Stasis",
                "id": "Stasis",
                "name": "Stasis",
                "amount": -1,
                "card": card,
                "damage": 0,
                "just_applied": False,
                "misc": 0,
            }
        )
        _sort_powers(monster)

    def _bronze_orb_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "BEAM":
            self._monster_attack_player(monster, int(monster.meta.get("beam_damage", 8)))
            monster.move_history.append("BEAM")
        elif monster.next_move == "SUPPORT":
            target = next((m for m in self.state.monsters if _alive(m) and m.monster_id == "BronzeAutomaton"), None)
            if target is not None:
                _gain_block(target, int(monster.meta.get("block_amt", 12)))
            monster.move_history.append("SUPPORT")
        elif monster.next_move == "STASIS":
            monster.meta["used_stasis"] = True
            self._apply_bronze_orb_stasis(monster)
            monster.move_history.append("STASIS")
        else:
            raise NotImplementedError(f"native_sim_v3 BronzeOrb move {monster.next_move!r} is not implemented yet.")
        self._pending_bronze_roll_events.append(("orb", monster))
        _decrement_turn_powers(monster)

    def _automaton_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "SPAWN_ORBS":
            alive_orbs = sum(1 for current in self.state.monsters if _alive(current) and current.monster_id == "BronzeOrb")
            for count in range(alive_orbs, 2):
                orb = _spawn_bronze_orb(self.randoms, self.ascension_level, count)
                self._choose_bronze_orb_next_move(orb)
                self._insert_spawned_monster_by_draw_x(orb)
            monster.meta["first_turn"] = False
            monster.move_history.append("SPAWN_ORBS")
        elif monster.next_move == "FLAIL":
            damage = int(monster.meta.get("flail_damage", 7))
            self._monster_attack_player(monster, damage, hits=2)
            monster.move_history.append("FLAIL")
        elif monster.next_move == "BOOST":
            _gain_block(monster, int(monster.meta.get("block_amt", 9)))
            _add_power(monster, "Strength", int(monster.meta.get("str_amt", 3)))
            monster.move_history.append("BOOST")
        elif monster.next_move == "HYPER_BEAM":
            self._monster_attack_player(monster, int(monster.meta.get("beam_damage", 45)))
            monster.move_history.append("HYPER_BEAM")
        elif monster.next_move == "STUNNED":
            monster.move_history.append("STUNNED")
        else:
            raise NotImplementedError(f"native_sim_v3 BronzeAutomaton move {monster.next_move!r} is not implemented yet.")
        completed_move = monster.move_history[-1] if monster.move_history else ""
        num_turns = int(monster.meta.get("num_turns", 0))
        if bool(monster.meta.get("first_turn", False)):
            monster.next_move = "SPAWN_ORBS"
        elif num_turns == 4:
            monster.next_move = "HYPER_BEAM"
            monster.meta["num_turns"] = 0
            self._pending_bronze_roll_events.append(("automaton_ai", None))
            _decrement_turn_powers(monster)
            return
        elif monster.move_history[-1:] == ["HYPER_BEAM"]:
            monster.next_move = "BOOST" if self.ascension_level >= 19 else "STUNNED"
            monster.meta["num_turns"] = num_turns
            self._pending_bronze_roll_events.append(("automaton_ai", None))
            _decrement_turn_powers(monster)
            return
        elif monster.move_history[-1:] == ["STUNNED"] or monster.move_history[-1:] == ["BOOST"] or monster.move_history[-1:] == ["SPAWN_ORBS"]:
            monster.next_move = "FLAIL"
        else:
            monster.next_move = "BOOST"
        monster.meta["num_turns"] = num_turns + 1
        self._pending_bronze_roll_events.append(("automaton_ai", None))
        _decrement_turn_powers(monster)

    def _gremlin_leader_num_alive_minions(self) -> int:
        count = 0
        for monster in self.state.monsters:
            if not _alive(monster) or monster.monster_id == "GremlinLeader":
                continue
            if monster.monster_id in {"GremlinWarrior", "GremlinThief", "GremlinFat", "GremlinTsundere", "GremlinWizard"}:
                count += 1
        return count

    def _gremlin_leader_roll_next_move(self, monster: MonsterState, num: int | None = None) -> None:
        if num is None:
            num = int(self.randoms.stream("ai").random(99))
        alive_minions = self._gremlin_leader_num_alive_minions()
        last_move = monster.move_history[-1] if monster.move_history else None
        if alive_minions == 0:
            if num < 75:
                monster.next_move = "RALLY" if last_move != "RALLY" else "STAB"
            else:
                monster.next_move = "STAB" if last_move != "STAB" else "RALLY"
        elif alive_minions < 2:
            if num < 50:
                if last_move != "RALLY":
                    monster.next_move = "RALLY"
                else:
                    self._gremlin_leader_roll_next_move(monster, int(self.randoms.stream("ai").random(50, 99)))
            elif num < 80:
                monster.next_move = "ENCOURAGE" if last_move != "ENCOURAGE" else "STAB"
            elif last_move != "STAB":
                monster.next_move = "STAB"
            else:
                self._gremlin_leader_roll_next_move(monster, int(self.randoms.stream("ai").random(0, 80)))
        else:
            if num < 66:
                monster.next_move = "ENCOURAGE" if last_move != "ENCOURAGE" else "STAB"
            else:
                monster.next_move = "STAB" if last_move != "STAB" else "ENCOURAGE"

    def _gremlin_leader_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "RALLY":
            slots = [(-366.0, 0), (-170.0, 1), (-532.0, 2)]
            living_minions_by_slot = {
                int(current.meta.get("gremlin_leader_slot"))
                for current in self.state.monsters
                if _alive(current)
                and current.monster_id in {"GremlinWarrior", "GremlinThief", "GremlinFat", "GremlinTsundere", "GremlinWizard"}
                and current.meta.get("gremlin_leader_slot") is not None
            }
            summoned = 0
            minions: list[MonsterState] = []
            for draw_x, slot in slots:
                if summoned >= 2:
                    break
                if slot in living_minions_by_slot:
                    continue
                summoned += 1
                minion = _spawn_random_gremlin_minion(
                    self.randoms,
                    self.ascension_level,
                    rng_stream="ai",
                    draw_x=draw_x,
                )
                minion.meta["is_minion"] = True
                minion.meta["gremlin_leader_slot"] = slot
                minion.powers.insert(0, _minion_power_payload())
                _sort_powers(minion)
                minions.append(minion)
            for minion in minions:
                # SummonGremlinAction chooses both gremlin types when the leader
                # queues actions, then each action later initializes its monster.
                self._insert_spawned_monster_by_draw_x(minion, consume_opening_ai=False)
                self._consume_monster_opening_ai_roll(minion)
            monster.move_history.append("RALLY")
        elif monster.next_move == "ENCOURAGE":
            self.randoms.stream("ai").random(0, 2)
            for current in self.state.monsters:
                if not _alive(current):
                    continue
                _add_power(current, "Strength", int(monster.meta.get("strength_amount", 3)))
                if current is not monster:
                    _gain_block(current, int(monster.meta.get("block_amount", 6)))
            monster.move_history.append("ENCOURAGE")
        elif monster.next_move == "STAB":
            damage = int(monster.meta.get("stab_damage", 6))
            self._monster_attack_player(monster, damage, hits=int(monster.meta.get("stab_hits", 3)))
            monster.move_history.append("STAB")
        else:
            raise NotImplementedError(f"native_sim_v3 Gremlin Leader move {monster.next_move!r} is not implemented yet.")
        self._gremlin_leader_roll_next_move(monster)
        _decrement_turn_powers(monster)

    def _chosen_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "POKE":
            damage = int(monster.meta.get("poke_damage", 5))
            self._monster_attack_player(monster, damage, hits=2)
            monster.move_history.append("POKE")
        elif monster.next_move == "HEX":
            self._apply_player_debuff("Hex", 1)
            monster.meta["used_hex"] = True
            monster.move_history.append("HEX")
        elif monster.next_move == "DEBILITATE":
            self._monster_attack_player(monster, int(monster.meta.get("debilitate_damage", 10)))
            self._apply_player_debuff("Vulnerable", int(monster.meta.get("debilitate_vuln", 2)))
            monster.move_history.append("DEBILITATE")
        elif monster.next_move == "DRAIN":
            self._apply_player_debuff("Weak", int(monster.meta.get("drain_weak", 3)))
            _add_power(monster, "Strength", int(monster.meta.get("drain_strength", 3)))
            monster.move_history.append("DRAIN")
        elif monster.next_move == "ZAP":
            self._monster_attack_player(monster, int(monster.meta.get("zap_damage", 18)))
            monster.move_history.append("ZAP")
        else:
            raise NotImplementedError(f"native_sim_v3 Chosen move {monster.next_move!r} is not implemented yet.")
        num = int(self.randoms.stream("ai").random(99))
        if self.ascension_level >= 17:
            if not bool(monster.meta.get("used_hex", False)):
                monster.next_move = "HEX"
            elif monster.next_move not in {"DEBILITATE", "DRAIN"}:
                monster.next_move = "DEBILITATE" if num < 50 else "DRAIN"
            elif num < 40:
                monster.next_move = "ZAP"
            else:
                monster.next_move = "POKE"
        else:
            if bool(monster.meta.get("first_turn", False)):
                monster.meta["first_turn"] = False
                monster.next_move = "HEX" if not bool(monster.meta.get("used_hex", False)) else ("DEBILITATE" if num < 50 else "DRAIN")
            elif not bool(monster.meta.get("used_hex", False)):
                monster.next_move = "HEX"
            elif monster.next_move not in {"DEBILITATE", "DRAIN"}:
                monster.next_move = "DEBILITATE" if num < 50 else "DRAIN"
            elif num < 40:
                monster.next_move = "ZAP"
            else:
                monster.next_move = "POKE"
        _decrement_turn_powers(monster)

    def _spheric_guardian_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "INITIAL_BLOCK_GAIN":
            _gain_block(monster, int(monster.meta.get("activate_block", 25)))
            monster.move_history.append("INITIAL_BLOCK_GAIN")
        elif monster.next_move == "FRAIL_ATTACK":
            self._monster_attack_player(monster, int(monster.meta.get("attack_damage", 10)))
            self._apply_player_debuff("Frail", int(monster.meta.get("frail_amount", 5)))
            monster.move_history.append("FRAIL_ATTACK")
        elif monster.next_move == "BIG_ATTACK":
            self._monster_attack_player(monster, int(monster.meta.get("attack_damage", 10)), hits=2)
            monster.move_history.append("BIG_ATTACK")
        elif monster.next_move == "BLOCK_ATTACK":
            _gain_block(monster, int(monster.meta.get("block_attack_block", 15)))
            self._monster_attack_player(monster, int(monster.meta.get("attack_damage", 10)))
            monster.move_history.append("BLOCK_ATTACK")
        else:
            raise NotImplementedError(f"native_sim_v3 Spheric Guardian move {monster.next_move!r} is not implemented yet.")
        if bool(monster.meta.get("first_move", False)):
            monster.meta["first_move"] = False
            monster.next_move = "FRAIL_ATTACK"
        elif bool(monster.meta.get("second_move", False)):
            monster.meta["second_move"] = False
            monster.next_move = "BIG_ATTACK"
        elif monster.next_move == "BIG_ATTACK":
            monster.next_move = "BLOCK_ATTACK"
        else:
            monster.next_move = "BIG_ATTACK"
        _decrement_turn_powers(monster)

    def _book_of_stabbing_roll_next_move(self, monster: MonsterState, num: int) -> None:
        if int(num) < 15:
            if monster.move_history and monster.move_history[-1] == "BIG_STAB":
                monster.meta["stab_count"] = int(monster.meta.get("stab_count", 1)) + 1
                monster.next_move = "STAB"
            else:
                monster.next_move = "BIG_STAB"
                if self.ascension_level >= 18:
                    monster.meta["stab_count"] = int(monster.meta.get("stab_count", 1)) + 1
        elif len(monster.move_history) >= 2 and monster.move_history[-1] == "STAB" and monster.move_history[-2] == "STAB":
            monster.next_move = "BIG_STAB"
            if self.ascension_level >= 18:
                monster.meta["stab_count"] = int(monster.meta.get("stab_count", 1)) + 1
        else:
            monster.meta["stab_count"] = int(monster.meta.get("stab_count", 1)) + 1
            monster.next_move = "STAB"

    def _book_of_stabbing_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "STAB":
            hits = int(monster.meta.get("stab_count", 1))
            self._monster_attack_player(monster, int(monster.meta.get("stab_damage", 6)), hits=hits)
            monster.move_history.append("STAB")
        elif monster.next_move == "BIG_STAB":
            self._monster_attack_player(monster, int(monster.meta.get("big_stab_damage", 21)))
            monster.move_history.append("BIG_STAB")
        else:
            raise NotImplementedError(f"native_sim_v3 Book of Stabbing move {monster.next_move!r} is not implemented yet.")
        self._book_of_stabbing_roll_next_move(monster, int(self.randoms.stream("ai").random(99)))
        _decrement_turn_powers(monster)

    def _taskmaster_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "SCOURING_WHIP":
            self._monster_attack_player(monster, int(monster.meta.get("whip_damage", 7)))
            for i in range(int(monster.meta.get("wound_count", 1))):
                self.state.discard_pile.append(make_card("Wound", uuid=f"taskmaster-wound-{len(self.state.discard_pile)+i}"))
            if self.ascension_level >= 18:
                _add_power(monster, "Strength", 1)
            monster.move_history.append("SCOURING_WHIP")
        else:
            raise NotImplementedError(f"native_sim_v3 Taskmaster move {monster.next_move!r} is not implemented yet.")
        self.randoms.stream("ai").random(99)
        _decrement_turn_powers(monster)

    def _orb_walker_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "LASER":
            self._monster_attack_player(monster, int(monster.meta.get("laser_damage", 10)))
            self.state.discard_pile.append(make_card("Burn", uuid=f"orbwalker-burn-discard-{len(self.state.discard_pile)}"))
            self._add_card_to_draw_pile_random_spot(make_card("Burn", uuid=f"orbwalker-burn-draw-{len(self.state.draw_pile)}"))
            monster.move_history.append("LASER")
        elif monster.next_move == "CLAW":
            self._monster_attack_player(monster, int(monster.meta.get("claw_damage", 15)))
            monster.move_history.append("CLAW")
        else:
            raise NotImplementedError(f"native_sim_v3 OrbWalker move {monster.next_move!r} is not implemented yet.")
        num = int(self.randoms.stream("ai").random(99))
        last_two = monster.move_history[-2:]
        if num < 40:
            monster.next_move = "CLAW" if len(last_two) < 2 or last_two != ["CLAW", "CLAW"] else "LASER"
        else:
            monster.next_move = "LASER" if len(last_two) < 2 or last_two != ["LASER", "LASER"] else "CLAW"
        _decrement_turn_powers(monster)

    def _bandit_pointy_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "POINTY_SPECIAL":
            self._monster_attack_player(
                monster,
                int(monster.meta.get("attack_damage", 5)),
                hits=int(monster.meta.get("attack_hits", 2)),
            )
            monster.move_history.append("POINTY_SPECIAL")
            monster.next_move = "POINTY_SPECIAL"
        else:
            raise NotImplementedError(f"native_sim_v3 Bandit Pointy move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _bandit_bear_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "BEAR_HUG":
            self._apply_player_debuff("Dexterity", -int(monster.meta.get("dexterity_loss", 2)))
            monster.move_history.append("BEAR_HUG")
            monster.next_move = "LUNGE"
        elif monster.next_move == "LUNGE":
            self._monster_attack_player(monster, int(monster.meta.get("lunge_damage", 9)))
            _gain_block(monster, int(monster.meta.get("lunge_block", 9)))
            monster.move_history.append("LUNGE")
            monster.next_move = "MAUL"
        elif monster.next_move == "MAUL":
            self._monster_attack_player(monster, int(monster.meta.get("maul_damage", 18)))
            monster.move_history.append("MAUL")
            monster.next_move = "LUNGE"
        else:
            raise NotImplementedError(f"native_sim_v3 Bandit Bear move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _bandit_leader_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "MOCK":
            monster.move_history.append("MOCK")
            monster.next_move = "AGONIZING_SLASH"
        elif monster.next_move == "AGONIZING_SLASH":
            self._monster_attack_player(monster, int(monster.meta.get("agonize_damage", 10)))
            self._apply_player_debuff("Weak", int(monster.meta.get("weak_amount", 2)))
            monster.move_history.append("AGONIZING_SLASH")
            monster.next_move = "CROSS_SLASH"
        elif monster.next_move == "CROSS_SLASH":
            self._monster_attack_player(monster, int(monster.meta.get("slash_damage", 15)))
            monster.move_history.append("CROSS_SLASH")
            if self.ascension_level >= 17 and monster.move_history[-2:] != ["CROSS_SLASH", "CROSS_SLASH"]:
                monster.next_move = "CROSS_SLASH"
            else:
                monster.next_move = "AGONIZING_SLASH"
        else:
            raise NotImplementedError(f"native_sim_v3 Bandit Leader move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _gremlin_warrior_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "SCRATCH":
            self._monster_attack_player(monster, int(monster.meta.get("scratch_damage", 4)))
            monster.move_history.append("SCRATCH")
            if bool(monster.meta.get("escape_next", False)):
                monster.next_move = "ESCAPE"
            else:
                monster.next_move = "SCRATCH"
        elif monster.next_move == "ESCAPE":
            monster.move_history.append("ESCAPE")
            self._kill_monster(monster, escaped=True)
        else:
            raise NotImplementedError(f"native_sim_v3 Gremlin Warrior move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _gremlin_fat_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "BLUNT":
            self._monster_attack_player(monster, int(monster.meta.get("blunt_damage", 4)))
            self._apply_player_debuff("Weak", int(monster.meta.get("weak_amount", 1)))
            frail_amount = int(monster.meta.get("frail_amount", 0))
            if frail_amount > 0:
                self._apply_player_debuff("Frail", frail_amount)
            monster.move_history.append("BLUNT")
            if bool(monster.meta.get("escape_next", False)):
                monster.next_move = "ESCAPE"
            else:
                # The source uses RollMoveAction even though Fat Gremlin has a
                # deterministic next move, so aiRng still advances here.
                self.randoms.stream("ai").random(99)
                monster.next_move = "BLUNT"
        elif monster.next_move == "ESCAPE":
            monster.move_history.append("ESCAPE")
            self._kill_monster(monster, escaped=True)
        else:
            raise NotImplementedError(f"native_sim_v3 Gremlin Fat move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _gremlin_thief_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "PUNCTURE":
            self._monster_attack_player(monster, int(monster.meta.get("puncture_damage", 9)))
            monster.move_history.append("PUNCTURE")
            if bool(monster.meta.get("escape_next", False)):
                monster.next_move = "ESCAPE"
            else:
                monster.next_move = "PUNCTURE"
        elif monster.next_move == "ESCAPE":
            monster.move_history.append("ESCAPE")
            self._kill_monster(monster, escaped=True)
        else:
            raise NotImplementedError(f"native_sim_v3 Gremlin Thief move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _gremlin_tsundere_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "PROTECT":
            allies = [
                current
                for current in self.state.monsters
                if current is not monster and _alive(current) and current.intent != "ESCAPE"
            ]
            picked = allies[int(self.randoms.stream("ai").random(0, len(allies) - 1))] if allies else monster
            _gain_block(picked, int(monster.meta.get("block_amount", 7)))
            monster.move_history.append("PROTECT")
            if bool(monster.meta.get("escape_next", False)):
                monster.next_move = "ESCAPE"
            else:
                alive_count = sum(1 for current in self.state.monsters if _alive(current))
                monster.next_move = "PROTECT" if alive_count > 1 else "BASH"
        elif monster.next_move == "BASH":
            self._monster_attack_player(monster, int(monster.meta.get("bash_damage", 6)))
            monster.move_history.append("BASH")
            if bool(monster.meta.get("escape_next", False)):
                monster.next_move = "ESCAPE"
            else:
                monster.next_move = "BASH"
        elif monster.next_move == "ESCAPE":
            monster.move_history.append("ESCAPE")
            self._kill_monster(monster, escaped=True)
        else:
            raise NotImplementedError(f"native_sim_v3 Gremlin Tsundere move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _gremlin_wizard_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "CHARGE":
            monster.meta["current_charge"] = int(monster.meta.get("current_charge", 1)) + 1
            monster.move_history.append("CHARGE")
            if bool(monster.meta.get("escape_next", False)):
                monster.next_move = "ESCAPE"
            elif int(monster.meta.get("current_charge", 0)) == 3:
                monster.next_move = "DOPE_MAGIC"
            else:
                monster.next_move = "CHARGE"
        elif monster.next_move == "DOPE_MAGIC":
            monster.meta["current_charge"] = 0
            self._monster_attack_player(monster, int(monster.meta.get("magic_damage", 25)))
            monster.move_history.append("DOPE_MAGIC")
            if bool(monster.meta.get("escape_next", False)):
                monster.next_move = "ESCAPE"
            elif self.ascension_level >= 17:
                monster.next_move = "DOPE_MAGIC"
            else:
                monster.next_move = "CHARGE"
        elif monster.next_move == "ESCAPE":
            monster.move_history.append("ESCAPE")
            self._kill_monster(monster, escaped=True)
        else:
            raise NotImplementedError(f"native_sim_v3 Gremlin Wizard move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _acid_slime_s_roll_opening_move(self, monster: MonsterState) -> None:
        # AbstractMonster.rollMove consumes a random(99) before AcidSlime_S
        # optionally consumes randomBoolean inside getMove().
        self.randoms.stream("ai").random(99)
        if self.ascension_level >= 17:
            last_two = monster.move_history[-2:]
            if len(last_two) == 2 and last_two[0] == "TACKLE" and last_two[1] == "TACKLE":
                monster.next_move = "TACKLE"
            else:
                monster.next_move = "LICK"
            return
        monster.next_move = "TACKLE" if self.randoms.stream("ai").random_boolean() else "LICK"

    def _acid_slime_s_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "TACKLE":
            self._monster_attack_player(monster, int(monster.meta.get("tackle_damage", 3)))
            monster.move_history.append("TACKLE")
            monster.next_move = "LICK"
        elif monster.next_move == "LICK":
            self._apply_player_debuff("Weak", 1)
            monster.move_history.append("LICK")
            monster.next_move = "TACKLE"
        else:
            raise NotImplementedError(f"native_sim_v3 Acid Slime S move {monster.next_move!r} is not implemented yet.")
        _decrement_turn_powers(monster)

    def _spike_slime_s_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move != "TACKLE":
            raise NotImplementedError(f"native_sim_v3 Spike Slime S move {monster.next_move!r} is not implemented yet.")
        self._monster_attack_player(monster, int(monster.meta.get("tackle_damage", 5)))
        monster.move_history.append("TACKLE")
        self.randoms.stream("ai").random(99)
        monster.next_move = "TACKLE"
        _decrement_turn_powers(monster)

    def _acid_slime_m_roll_next_move(self, monster: MonsterState) -> None:
        num = int(self.randoms.stream("ai").random(99))
        last_move = monster.move_history[-1] if monster.move_history else None
        last_two = monster.move_history[-2:]
        if self.ascension_level >= 17:
            if num < 40:
                if len(last_two) == 2 and last_two[0] == "WOUND_TACKLE" and last_two[1] == "WOUND_TACKLE":
                    monster.next_move = "NORMAL_TACKLE" if self.randoms.stream("ai").random_boolean() else "LICK"
                else:
                    monster.next_move = "WOUND_TACKLE"
                return
            if num < 80:
                if len(last_two) == 2 and last_two[0] == "NORMAL_TACKLE" and last_two[1] == "NORMAL_TACKLE":
                    monster.next_move = "WOUND_TACKLE" if self.randoms.stream("ai").random_boolean(0.5) else "LICK"
                else:
                    monster.next_move = "NORMAL_TACKLE"
                return
            if last_move == "LICK":
                monster.next_move = "WOUND_TACKLE" if self.randoms.stream("ai").random_boolean(0.4) else "NORMAL_TACKLE"
            else:
                monster.next_move = "LICK"
            return
        if num < 30:
            if len(last_two) == 2 and last_two[0] == "WOUND_TACKLE" and last_two[1] == "WOUND_TACKLE":
                monster.next_move = "NORMAL_TACKLE" if self.randoms.stream("ai").random_boolean() else "LICK"
            else:
                monster.next_move = "WOUND_TACKLE"
            return
        if num < 70:
            if last_move == "NORMAL_TACKLE":
                monster.next_move = "WOUND_TACKLE" if self.randoms.stream("ai").random_boolean(0.4) else "LICK"
            else:
                monster.next_move = "NORMAL_TACKLE"
            return
        if len(last_two) == 2 and last_two[0] == "LICK" and last_two[1] == "LICK":
            monster.next_move = "WOUND_TACKLE" if self.randoms.stream("ai").random_boolean(0.4) else "NORMAL_TACKLE"
        else:
            monster.next_move = "LICK"

    def _acid_slime_m_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "WOUND_TACKLE":
            self._monster_attack_player(monster, int(monster.meta.get("wound_damage", 7)))
            self.state.discard_pile.append(make_card("Slimed", uuid="temp-slimed"))
            monster.move_history.append("WOUND_TACKLE")
        elif monster.next_move == "NORMAL_TACKLE":
            self._monster_attack_player(monster, int(monster.meta.get("normal_damage", 10)))
            monster.move_history.append("NORMAL_TACKLE")
        elif monster.next_move == "LICK":
            self._apply_player_debuff("Weak", 1)
            monster.move_history.append("LICK")
        else:
            raise NotImplementedError(f"native_sim_v3 Acid Slime M move {monster.next_move!r} is not implemented yet.")
        self._acid_slime_m_roll_next_move(monster)
        _decrement_turn_powers(monster)

    def _spike_slime_m_roll_next_move(self, monster: MonsterState) -> None:
        num = int(self.randoms.stream("ai").random(99))
        last_move = monster.move_history[-1] if monster.move_history else None
        last_two = monster.move_history[-2:]
        if num < 30:
            if len(last_two) == 2 and last_two[0] == "FLAME_TACKLE" and last_two[1] == "FLAME_TACKLE":
                monster.next_move = "LICK"
            else:
                monster.next_move = "FLAME_TACKLE"
            return
        if len(last_two) == 2 and last_two[0] == "LICK" and last_two[1] == "LICK":
            monster.next_move = "FLAME_TACKLE"
        elif self.ascension_level >= 17 and last_move == "LICK":
            monster.next_move = "FLAME_TACKLE"
        else:
            monster.next_move = "LICK"

    def _spike_slime_m_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "FLAME_TACKLE":
            self._monster_attack_player(monster, int(monster.meta.get("tackle_damage", 8)))
            self.state.discard_pile.append(make_card("Slimed", uuid="temp-slimed"))
            monster.move_history.append("FLAME_TACKLE")
        elif monster.next_move == "LICK":
            self._apply_player_debuff("Frail", 1)
            monster.move_history.append("LICK")
        else:
            raise NotImplementedError(f"native_sim_v3 Spike Slime M move {monster.next_move!r} is not implemented yet.")
        self._spike_slime_m_roll_next_move(monster)
        _decrement_turn_powers(monster)

    def _acid_slime_l_roll_next_move(self, monster: MonsterState) -> None:
        num = int(self.randoms.stream("ai").random(99))
        last_move = monster.move_history[-1] if monster.move_history else None
        last_two = monster.move_history[-2:]
        if self.ascension_level >= 17:
            if num < 40:
                if len(last_two) == 2 and last_two[0] == "WOUND_TACKLE" and last_two[1] == "WOUND_TACKLE":
                    monster.next_move = "NORMAL_TACKLE" if self.randoms.stream("ai").random_boolean(0.6) else "WEAK_LICK"
                else:
                    monster.next_move = "WOUND_TACKLE"
            elif num < 70:
                if len(last_two) == 2 and last_two[0] == "NORMAL_TACKLE" and last_two[1] == "NORMAL_TACKLE":
                    monster.next_move = "WOUND_TACKLE" if self.randoms.stream("ai").random_boolean(0.6) else "WEAK_LICK"
                else:
                    monster.next_move = "NORMAL_TACKLE"
            elif last_move == "WEAK_LICK":
                monster.next_move = "WOUND_TACKLE" if self.randoms.stream("ai").random_boolean(0.4) else "NORMAL_TACKLE"
            else:
                monster.next_move = "WEAK_LICK"
            return
        if num < 30:
            if len(last_two) == 2 and last_two[0] == "WOUND_TACKLE" and last_two[1] == "WOUND_TACKLE":
                monster.next_move = "NORMAL_TACKLE" if self.randoms.stream("ai").random_boolean() else "WEAK_LICK"
            else:
                monster.next_move = "WOUND_TACKLE"
        elif num < 70:
            if last_move == "NORMAL_TACKLE":
                monster.next_move = "WOUND_TACKLE" if self.randoms.stream("ai").random_boolean(0.4) else "WEAK_LICK"
            else:
                monster.next_move = "NORMAL_TACKLE"
        elif len(last_two) == 2 and last_two[0] == "WEAK_LICK" and last_two[1] == "WEAK_LICK":
            monster.next_move = "WOUND_TACKLE" if self.randoms.stream("ai").random_boolean(0.4) else "NORMAL_TACKLE"
        else:
            monster.next_move = "WEAK_LICK"

    def _acid_slime_l_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "WOUND_TACKLE":
            self._monster_attack_player(monster, int(monster.meta.get("wound_damage", 11)))
            self.state.discard_pile.extend([make_card("Slimed", uuid=f"temp-slimed-{len(self.state.discard_pile)+i}") for i in range(2)])
            monster.move_history.append("WOUND_TACKLE")
        elif monster.next_move == "NORMAL_TACKLE":
            self._monster_attack_player(monster, int(monster.meta.get("normal_damage", 16)))
            monster.move_history.append("NORMAL_TACKLE")
        elif monster.next_move == "WEAK_LICK":
            self._apply_player_debuff("Weak", 2)
            monster.move_history.append("WEAK_LICK")
        elif monster.next_move == "SPLIT":
            child_hp = max(1, int(monster.current_hp))
            parent_x = _monster_draw_x(monster)
            self._kill_monster(monster, suppress_victory=True)
            children = [
                _spawn_acid_slime_m(self.randoms, self.ascension_level, hp_override=child_hp, draw_x=parent_x - 134.0),
                _spawn_acid_slime_m(self.randoms, self.ascension_level, hp_override=child_hp, draw_x=parent_x + 134.0),
            ]
            for child in children:
                self._roll_spawned_slime_initial_move(child)
            self._insert_split_children_around_parent(
                monster,
                children,
            )
            return
        else:
            raise NotImplementedError(f"native_sim_v3 Acid Slime L move {monster.next_move!r} is not implemented yet.")
        if not self._roll_then_force_pending_large_slime_split_move(monster):
            self._acid_slime_l_roll_next_move(monster)
        _decrement_turn_powers(monster)

    def _spike_slime_l_roll_next_move(self, monster: MonsterState) -> None:
        num = int(self.randoms.stream("ai").random(99))
        last_two = monster.move_history[-2:]
        if num < 30:
            monster.next_move = "FRAIL_LICK" if len(last_two) == 2 and last_two[0] == "FLAME_TACKLE" and last_two[1] == "FLAME_TACKLE" else "FLAME_TACKLE"
        elif self.ascension_level >= 17:
            monster.next_move = "FLAME_TACKLE" if monster.move_history[-1:] == ["FRAIL_LICK"] else "FRAIL_LICK"
        elif len(last_two) == 2 and last_two[0] == "FRAIL_LICK" and last_two[1] == "FRAIL_LICK":
            monster.next_move = "FLAME_TACKLE"
        else:
            monster.next_move = "FRAIL_LICK"

    def _set_spike_slime_l_intent_from_move(self, monster: MonsterState) -> None:
        if monster.next_move == "FLAME_TACKLE":
            monster.intent = "ATTACK_DEBUFF"
            monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("tackle_damage", 16)), False)
            monster.move_hits = 1
        elif monster.next_move == "FRAIL_LICK":
            monster.intent = "DEBUFF"
            monster.move_adjusted_damage = 0
            monster.move_hits = 0
        else:
            monster.intent = "UNKNOWN"
            monster.move_adjusted_damage = 0
            monster.move_hits = 0

    def _spike_slime_l_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "FLAME_TACKLE":
            self._monster_attack_player(monster, int(monster.meta.get("tackle_damage", 16)))
            self.state.discard_pile.extend([make_card("Slimed", uuid=f"temp-slimed-{len(self.state.discard_pile)+i}") for i in range(2)])
            monster.move_history.append("FLAME_TACKLE")
        elif monster.next_move == "FRAIL_LICK":
            self._apply_player_debuff("Frail", 3 if self.ascension_level >= 17 else 2)
            monster.move_history.append("FRAIL_LICK")
        elif monster.next_move == "SPLIT":
            child_hp = max(1, int(monster.current_hp))
            parent_x = _monster_draw_x(monster)
            self._kill_monster(monster, suppress_victory=True)
            children = [
                _spawn_spike_slime_m(self.randoms, self.ascension_level, hp_override=child_hp, draw_x=parent_x - 134.0),
                _spawn_spike_slime_m(self.randoms, self.ascension_level, hp_override=child_hp, draw_x=parent_x + 134.0),
            ]
            for child in children:
                self._roll_spawned_slime_initial_move(child)
            self._insert_split_children_around_parent(
                monster,
                children,
            )
            self._spike_slime_l_roll_next_move(monster)
            self._set_spike_slime_l_intent_from_move(monster)
            return
        else:
            raise NotImplementedError(f"native_sim_v3 Spike Slime L move {monster.next_move!r} is not implemented yet.")
        if not self._roll_then_force_pending_large_slime_split_move(monster):
            self._spike_slime_l_roll_next_move(monster)
        _decrement_turn_powers(monster)

    def _louse_normal_roll_next_move(self, monster: MonsterState) -> None:
        monster.next_move = _roll_louse_normal_next_move(self.randoms, self.ascension_level, list(monster.move_history))

    def _louse_normal_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "BITE":
            monster.meta["state"] = "OPEN"
            damage = int(monster.move_adjusted_damage or 0)
            if damage <= 0:
                self._monster_attack_player(monster, int(monster.meta.get("bite_damage", 5)))
            else:
                self._damage_player(damage, attacker=monster, apply_player_vulnerable=False)
            monster.move_history.append("BITE")
        elif monster.next_move == "STRENGTHEN":
            monster.meta["state"] = "OPEN"
            _add_power(monster, "Strength", int(monster.meta.get("strength_gain", 3)))
            monster.move_history.append("STRENGTHEN")
        else:
            raise NotImplementedError(f"native_sim_v3 LouseNormal move {monster.next_move!r} is not implemented yet.")
        self._louse_normal_roll_next_move(monster)
        _decrement_turn_powers(monster)

    def _louse_defensive_roll_next_move(self, monster: MonsterState) -> None:
        monster.next_move = _roll_louse_defensive_next_move(self.randoms, self.ascension_level, list(monster.move_history))

    def _louse_defensive_take_turn(self, monster: MonsterState) -> None:
        if monster.next_move == "BITE":
            monster.meta["state"] = "OPEN"
            damage = int(monster.move_adjusted_damage or 0)
            if damage <= 0:
                self._monster_attack_player(monster, int(monster.meta.get("bite_damage", 5)))
            else:
                self._damage_player(damage, attacker=monster, apply_player_vulnerable=False)
            monster.move_history.append("BITE")
        elif monster.next_move == "WEAKEN":
            monster.meta["state"] = "OPEN"
            self._apply_player_debuff("Weak", 2)
            monster.move_history.append("WEAKEN")
        else:
            raise NotImplementedError(f"native_sim_v3 LouseDefensive move {monster.next_move!r} is not implemented yet.")
        self._louse_defensive_roll_next_move(monster)
        _decrement_turn_powers(monster)

    def _take_monster_turns(self) -> None:
        self._in_monster_turn = True
        try:
            for monster in list(self.state.monsters):
                if _alive(monster):
                    if bool(monster.meta.pop("_skip_next_block_clear", False)):
                        pass
                    else:
                        _clear_block_if_needed(monster)
            # MonsterGroup queues actions for monsters present at turn start.
            # Monsters spawned mid-turn (for example slime split children) do not
            # immediately receive their own turn in the same monster phase.
            for monster in list(self.state.monsters):
                if monster.monster_id == "AwakenedOne" and bool(monster.meta.get("half_dead", False)):
                    self._awakened_one_take_turn(monster)
                    continue
                if monster.monster_id == "Darkling" and bool(monster.meta.get("half_dead", False)):
                    self._darkling_take_turn(monster)
                    continue
                if not _alive(monster):
                    continue
                if monster.monster_id == "Cultist":
                    self._cultist_take_turn(monster)
                elif monster.monster_id == "JawWorm":
                    self._jaw_worm_take_turn(monster)
                elif monster.monster_id == "SlaverBlue":
                    self._blue_slaver_take_turn(monster)
                elif monster.monster_id == "SlaverRed":
                    self._red_slaver_take_turn(monster)
                elif monster.monster_id in {"Looter", "Mugger"}:
                    self._looter_take_turn(monster)
                elif monster.monster_id == "GremlinNob":
                    self._gremlin_nob_take_turn(monster)
                elif monster.monster_id == "Lagavulin":
                    self._lagavulin_take_turn(monster)
                elif monster.monster_id == "Sentry":
                    self._sentry_take_turn(monster)
                elif monster.monster_id == "Repulsor":
                    self._repulsor_take_turn(monster)
                elif monster.monster_id == "Spiker":
                    self._spiker_take_turn(monster)
                elif monster.monster_id == "Exploder":
                    self._exploder_take_turn(monster)
                elif monster.monster_id == "GiantHead":
                    self._giant_head_take_turn(monster)
                elif monster.monster_id == "Nemesis":
                    self._nemesis_take_turn(monster)
                elif monster.monster_id == "AwakenedOne":
                    self._awakened_one_take_turn(monster)
                elif monster.monster_id == "Darkling":
                    self._darkling_take_turn(monster)
                elif monster.monster_id == "Dagger":
                    self._snake_dagger_take_turn(monster)
                elif monster.monster_id == "Reptomancer":
                    self._reptomancer_take_turn(monster)
                elif monster.monster_id == "TimeEater":
                    self._time_eater_take_turn(monster)
                elif monster.monster_id == "Donu":
                    self._donu_take_turn(monster)
                elif monster.monster_id == "Deca":
                    self._deca_take_turn(monster)
                elif monster.monster_id == "Transient":
                    self._transient_take_turn(monster)
                elif monster.monster_id == "Maw":
                    self._maw_take_turn(monster)
                elif monster.monster_id in {"SpireGrowth", "Serpent"}:
                    self._spire_growth_take_turn(monster)
                elif monster.monster_id == "WrithingMass":
                    self._writhing_mass_take_turn(monster)
                elif monster.monster_id == "SlimeBoss":
                    self._slime_boss_take_turn(monster)
                elif monster.monster_id == "TheGuardian":
                    self._guardian_take_turn(monster)
                elif monster.monster_id == "Hexaghost":
                    self._hexaghost_take_turn(monster)
                elif monster.monster_id == "Byrd":
                    self._byrd_take_turn(monster)
                elif monster.monster_id == "Snecko":
                    self._snecko_take_turn(monster)
                elif monster.monster_id == "Chosen":
                    self._chosen_take_turn(monster)
                elif monster.monster_id == "SphericGuardian":
                    self._spheric_guardian_take_turn(monster)
                elif monster.monster_id == "BookOfStabbing":
                    self._book_of_stabbing_take_turn(monster)
                elif monster.monster_id == "SlaverBoss":
                    self._taskmaster_take_turn(monster)
                elif monster.monster_id == "OrbWalker":
                    self._orb_walker_take_turn(monster)
                elif monster.monster_id == "BanditChild":
                    self._bandit_pointy_take_turn(monster)
                elif monster.monster_id == "BanditBear":
                    self._bandit_bear_take_turn(monster)
                elif monster.monster_id == "BanditLeader":
                    self._bandit_leader_take_turn(monster)
                elif monster.monster_id == "SnakePlant":
                    self._snake_plant_take_turn(monster)
                elif monster.monster_id == "ShelledParasite":
                    self._shelled_parasite_take_turn(monster)
                elif monster.monster_id == "Centurion":
                    self._centurion_take_turn(monster)
                elif monster.monster_id == "Healer":
                    self._healer_take_turn(monster)
                elif monster.monster_id == "Champ":
                    self._champ_take_turn(monster)
                elif monster.monster_id == "TheCollector":
                    self._collector_take_turn(monster)
                elif monster.monster_id == "TorchHead":
                    self._torch_head_take_turn(monster)
                elif monster.monster_id == "BronzeOrb":
                    self._bronze_orb_take_turn(monster)
                elif monster.monster_id == "BronzeAutomaton":
                    self._automaton_take_turn(monster)
                elif monster.monster_id == "GremlinLeader":
                    self._gremlin_leader_take_turn(monster)
                elif monster.monster_id == "GremlinWarrior":
                    self._gremlin_warrior_take_turn(monster)
                elif monster.monster_id == "GremlinFat":
                    self._gremlin_fat_take_turn(monster)
                elif monster.monster_id == "GremlinThief":
                    self._gremlin_thief_take_turn(monster)
                elif monster.monster_id == "GremlinTsundere":
                    self._gremlin_tsundere_take_turn(monster)
                elif monster.monster_id == "GremlinWizard":
                    self._gremlin_wizard_take_turn(monster)
                elif monster.monster_id == "FungiBeast":
                    self._fungi_beast_take_turn(monster)
                elif monster.monster_id == "AcidSlime_S":
                    self._acid_slime_s_take_turn(monster)
                elif monster.monster_id == "SpikeSlime_S":
                    self._spike_slime_s_take_turn(monster)
                elif monster.monster_id == "AcidSlime_M":
                    self._acid_slime_m_take_turn(monster)
                elif monster.monster_id == "SpikeSlime_M":
                    self._spike_slime_m_take_turn(monster)
                elif monster.monster_id == "AcidSlime_L":
                    self._acid_slime_l_take_turn(monster)
                elif monster.monster_id == "SpikeSlime_L":
                    self._spike_slime_l_take_turn(monster)
                elif monster.monster_id == "FuzzyLouseNormal":
                    self._louse_normal_take_turn(monster)
                elif monster.monster_id == "FuzzyLouseDefensive":
                    self._louse_defensive_take_turn(monster)
                else:
                    raise NotImplementedError(f"native_sim_v3 monster {monster.monster_id!r} is not implemented yet.")
                _apply_end_of_turn_block_powers(monster)
                _apply_end_of_turn_temporary_powers(monster)
                if self.player.current_hp <= 0:
                    self.outcome = "DEFEAT"
                    return
        finally:
            self._in_monster_turn = False
        pending_bronze_roll_events = list(self._pending_bronze_roll_events)
        self._pending_bronze_roll_events = []
        self._pending_bronze_automaton_ai_rolls = 0
        for event_kind, monster in pending_bronze_roll_events:
            if event_kind == "orb":
                if monster is not None and _alive(monster):
                    self._choose_bronze_orb_next_move(monster)
            elif event_kind == "automaton_ai":
                self.randoms.stream("ai").random(99)
        for monster in self.state.monsters:
            _apply_monster_regenerate_power(monster)
        # Player turn-based debuffs expire at end of round after monsters act,
        # but before the next turn's intents are recalculated.
        _decrement_turn_powers(self.player)
        self._update_monster_intents()

    def _start_player_turn(self) -> None:
        self.state.turn += 1
        self.cards_played_this_turn = 0
        self.state.cards_discarded_this_turn = 0
        _remove_power(self.player, "FlameBarrier")
        for monster in self.state.monsters:
            if _alive(monster) and any(str(power.get("power_id") or power.get("id") or "") == "Slow" for power in monster.powers):
                _set_power_amount(monster, "Slow", 0)
        relic_ids = self._relic_ids
        if not _has_power(self.player, "Barricade"):
            if "Calipers" in relic_ids and int(self.player.block) > 15:
                self.player.block = 15
            else:
                self.player.block = 0
        next_turn_block = _get_power_amount(self.player, "NextTurnBlock")
        if next_turn_block > 0:
            _remove_power(self.player, "NextTurnBlock")
            self._gain_player_block(next_turn_block)
        retained_energy = int(self.player.energy) if "Ice Cream" in relic_ids else 0
        self.player.energy = (
            retained_energy
            + _player_base_energy(self.player, self.relics, relic_ids)
            + _player_bonus_energy_for_room(self.relics, self.state.room_type, relic_ids, elite_trigger=self.elite_trigger)
        )
        self._apply_turn_start_relic_counters()
        self._apply_brimstone_turn_start()
        self.player.energy += _get_power_amount(self.player, "Berserk")
        pocketwatch_draw = 0
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id in {"Kunai", "Shuriken", "Ornamental Fan", "Letter Opener"}:
                relic["counter"] = 0
            elif relic_id == "Velvet Choker":
                relic["counter"] = 0
            elif relic_id == "Pocketwatch":
                if int(relic.get("counter", 0) or 0) <= 3 and not bool(relic.get("first_turn", False)):
                    pocketwatch_draw = 3
                else:
                    relic["first_turn"] = False
                relic["counter"] = 0
            elif relic_id == "Art of War":
                if bool(relic.get("gain_energy_next", False)) and not bool(relic.get("first_turn", False)):
                    self.player.energy += 1
                relic["first_turn"] = False
                relic["gain_energy_next"] = True
            elif relic_id == "OrangePellets":
                relic["attack_played"] = False
                relic["skill_played"] = False
                relic["power_played"] = False
            elif relic_id == "Unceasing Top":
                relic["can_draw"] = True
                relic["disabled_until_end_of_turn"] = False
        demon_form = _get_power_amount(self.player, "DemonForm")
        if demon_form > 0:
            _add_power(self.player, "Strength", demon_form)
        brutality = _get_power_amount(self.player, "Brutality")
        if brutality > 0:
            self._lose_player_hp(brutality)
            if self.outcome == "DEFEAT":
                self.outcome = "DEFEAT"
                return
            self.draw_cards(brutality)
        magnetism = _get_power_amount(self.player, "Magnetism")
        if magnetism > 0:
            for copy_index in range(magnetism):
                generated = self._make_random_card(
                    predicate=lambda _cid, card_def: card_def.color == "COLORLESS" and card_def.rarity in {"UNCOMMON", "RARE"},
                    uuid_prefix="magnetism",
                    free_this_turn=False,
                    source_scope="COLORLESS",
                )
                if generated is not None:
                    self._add_temp_card_to_hand_or_discard(generated, reset_for_discard=False)
        panache = next((power for power in self.player.powers if str(power.get("power_id") or power.get("id")) == "Panache"), None)
        if panache is not None:
            panache["amount"] = 5
        self._apply_turn_start_damage_relics()
        if self.outcome != "UNDECIDED":
            return
        mayhem_cards: list[dict[str, Any]] = []
        mayhem = _get_power_amount(self.player, "Mayhem")
        for _ in range(max(0, int(mayhem))):
            if not self.state.draw_pile or self.outcome != "UNDECIDED":
                break
            top_card = self.state.draw_pile.pop()
            top_card["free_to_play_once"] = True
            mayhem_cards.append(top_card)
        self.draw_cards(_player_turn_draw_count(self.player, self.relics))
        if pocketwatch_draw > 0:
            self.draw_cards(pocketwatch_draw)
        for top_card in mayhem_cards:
            if self.outcome != "UNDECIDED":
                break
            self._resolve_generated_card_play(top_card)
        self._trigger_warped_tongs_post_draw()
        _refresh_card_flags(self.state.hand, self.player)

    def _trigger_warped_tongs_post_draw(self) -> None:
        if "WarpedTongs" not in self._relic_ids:
            return
        upgradable_indexes = [idx for idx, card in enumerate(self.state.hand) if can_upgrade_card(card)]
        if not upgradable_indexes:
            return
        java_shuffle_in_place(upgradable_indexes, int(self.randoms.stream("shuffle").random_long()))
        picked_index = upgradable_indexes[0]
        self.state.hand[picked_index] = upgrade_card(self.state.hand[picked_index])

    def _maybe_trigger_unceasing_top(self) -> None:
        if self.outcome != "UNDECIDED" or self.state.hand or _get_power_amount(self.player, "No Draw") != 0:
            return
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id != "Unceasing Top":
                continue
            if not bool(relic.get("can_draw", False)) or bool(relic.get("disabled_until_end_of_turn", False)):
                return
            if not self.state.draw_pile and not self.state.discard_pile:
                return
            self.draw_cards(1)
            return

    def _apply_player_end_of_turn_powers(self) -> None:
        if not _any_monsters_alive(self.state.monsters):
            self.outcome = "VICTORY"
            return
        if int(self.player.block) == 0 and "Orichalcum" in self._relic_ids:
            self._gain_player_block(6)
        self._gain_player_block(_end_of_turn_block_power_amount(self.player))
        _remove_power(self.player, "Rage")
        if self.state.hand and "CloakClasp" in self._relic_ids:
            self._gain_player_block(len(self.state.hand))
        ritual = _get_power_amount(self.player, "Ritual")
        if ritual > 0:
            _add_power(self.player, "Strength", ritual)
        regen = _get_power_amount(self.player, "Regeneration")
        if regen > 0:
            self._heal_player(regen)
            _set_power_amount(self.player, "Regeneration", regen - 1)
        _apply_end_of_turn_temporary_powers(self.player)
        for card in self.state.hand:
            card_id = str(card.get("card_id") or "")
            if card_id == "Decay":
                self._damage_player(2, damage_type="THORNS", apply_player_vulnerable=False, trigger_rupture=True)
            elif card_id == "Burn":
                self._damage_player(
                    int(card.get("base_magic") or 2),
                    damage_type="THORNS",
                    apply_player_vulnerable=False,
                    trigger_rupture=True,
                )
            if self.outcome == "DEFEAT":
                return
        doubt_count = sum(1 for card in self.state.hand if card.get("card_id") == "Doubt")
        if doubt_count:
            self._apply_player_debuff("Weak", doubt_count, source_monster=True)
        shame_count = sum(1 for card in self.state.hand if card.get("card_id") == "Shame")
        if shame_count:
            self._apply_player_debuff("Frail", shame_count, source_monster=True)
        constricted = _get_power_amount(self.player, "Constricted")
        if constricted:
            self._lose_player_hp(constricted)
            if self.outcome == "DEFEAT":
                self.outcome = "DEFEAT"
                return
        regret_count = sum(1 for card in self.state.hand if card.get("card_id") == "Regret")
        if regret_count:
            self._lose_player_hp(len(self.state.hand) * regret_count)
            if self.outcome == "DEFEAT":
                self.outcome = "DEFEAT"
                return
        pride_count = sum(1 for card in self.state.hand if card.get("card_id") == "Pride")
        for pride_index in range(pride_count):
            self.state.draw_pile.append(make_card("Pride", uuid=f"pride-copy-{self.state.turn}-{pride_index}-{len(self.state.draw_pile)}"))
        combust_powers = [
            power for power in self.player.powers if str(power.get("power_id") or power.get("id")) == "Combust"
        ]
        for power in combust_powers:
            hp_loss = int(power.get("misc") or 0)
            damage_amount = int(power.get("amount") or 0)
            if hp_loss > 0:
                self._lose_player_hp(hp_loss)
            for monster in self.state.monsters:
                if _alive(monster):
                    self._deal_non_attack_damage_to_monster(monster, damage_amount)
            if self.outcome == "DEFEAT":
                self.outcome = "DEFEAT"
                return
            if not _any_monsters_alive(self.state.monsters):
                self.outcome = "VICTORY"
                return
        bomb_powers = [
            power for power in self.player.powers if str(power.get("power_id") or power.get("id")).startswith("TheBomb")
        ]
        for power in list(bomb_powers):
            countdown = int(power.get("amount") or 0)
            damage_amount = int(power.get("misc") or 0)
            if countdown <= 1:
                self.player.powers.remove(power)
                for monster in self.state.monsters:
                    if _alive(monster):
                        self._deal_non_attack_damage_to_monster(monster, damage_amount)
                if self.outcome == "VICTORY":
                    return
            else:
                power["amount"] = countdown - 1
        stone_calendar = next(
            (relic for relic in self.relics if str(relic.get("relic_id") or relic.get("id") or "") == "StoneCalendar"),
            None,
        )
        if stone_calendar is not None and int(stone_calendar.get("counter", 0) or 0) == 7:
            for monster in self.state.monsters:
                if _alive(monster):
                    self._deal_non_attack_damage_to_monster(monster, 52)
            stone_calendar["counter"] = -1
            if self.outcome == "VICTORY":
                return

    def _end_turn(self) -> None:
        if not self._pending_end_turn_resume:
            self._preserve_next_guardian_mode_shift_block_clear = True
            self._defer_gremlin_horn_depth += 1
            self._defer_stasis_release_until_after_end_turn_discard = True
            try:
                self._apply_player_end_of_turn_powers()
            finally:
                self._defer_gremlin_horn_depth -= 1
                self._defer_stasis_release_until_after_end_turn_discard = False
                self._preserve_next_guardian_mode_shift_block_clear = False
            if self.outcome != "UNDECIDED":
                return
            if "Nilry's Codex" in self._relic_ids:
                codex_options = self._make_random_card_choices(
                    predicate=lambda _cid, card_def: card_def.type not in {"CURSE", "STATUS"},
                    uuid_prefix="nilrys-codex",
                    count=3,
                    source_scope="COLORED",
                )
                if codex_options:
                    self.pending_card_select = {
                        "mode": "NILRYS_CODEX",
                        "cards": codex_options,
                        "num_cards": 1,
                        "can_skip": True,
                        "resume_end_turn": True,
                    }
                    self._pending_end_turn_resume = True
                    return
        self._pending_end_turn_resume = False
        autoplay_cards: list[dict[str, Any]] = []
        retained_for_end_turn_discard: list[dict[str, Any]] = []
        for card in self.state.hand:
            if not bool(card.get("ethereal")) and str(card.get("card_id") or "") in END_TURN_AUTOPLAY_DISCARD_CARD_IDS:
                autoplay_cards.append(card)
            else:
                retained_for_end_turn_discard.append(card)
        # End-turn hand processing repeatedly removes the top card, which is
        # the reverse of Comm's visible hand ordering.
        ethereal_cards = list(reversed([card for card in retained_for_end_turn_discard if bool(card.get("ethereal"))]))
        non_ethereal_cards = [card for card in retained_for_end_turn_discard if not bool(card.get("ethereal"))]
        runic_pyramid = "Runic Pyramid" in self._relic_ids
        if runic_pyramid:
            self.state.hand = list(non_ethereal_cards)
            for card in self.state.hand:
                if bool(card.pop("_reset_cost_for_turn_after_play", False)):
                    _reset_card_for_pile(card)
                else:
                    _clear_play_once_flags_for_pile(card)
        else:
            self.state.hand = []
        self._defer_dark_embrace_draw_depth += 1
        try:
            self._exhaust_cards(ethereal_cards)
        finally:
            self._defer_dark_embrace_draw_depth -= 1
        for card in autoplay_cards:
            _reset_card_for_pile(card)
        self.state.discard_pile.extend(autoplay_cards)
        if not runic_pyramid:
            # Real end-turn discard repeatedly removes getTopCard(), which is
            # the reverse of Comm's visible hand ordering for ordinary cards.
            end_turn_discards = list(reversed(non_ethereal_cards))
            for card in end_turn_discards:
                if bool(card.pop("_reset_cost_for_turn_after_play", False)):
                    _reset_card_for_pile(card)
                else:
                    _clear_play_once_flags_for_pile(card)
            self.state.discard_pile.extend(end_turn_discards)
        self._flush_pending_stasis_release_cards()
        self._flush_deferred_gremlin_horn_rewards()
        self._flush_pending_dark_embrace_draws()
        _remove_power(self.player, "Double Tap")
        _remove_power(self.player, "DuplicationPower")
        _remove_power(self.player, "No Draw")
        _remove_power(self.player, "Entangled")
        self._take_monster_turns()
        if self.outcome == "UNDECIDED":
            self._start_player_turn()

    def _update_monster_intents(self) -> None:
        for monster in self.state.monsters:
            if monster.monster_id == "AwakenedOne" and bool(monster.meta.get("half_dead", False)):
                monster.intent = "UNKNOWN"
                monster.move_adjusted_damage = 0
                monster.move_hits = 0
                continue
            if monster.monster_id == "Darkling" and bool(monster.meta.get("half_dead", False)):
                if monster.next_move == "COUNT":
                    monster.intent = "UNKNOWN"
                else:
                    monster.intent = "BUFF"
                monster.move_adjusted_damage = 0
                monster.move_hits = 0
                continue
            if not _alive(monster):
                continue
            strength = _get_power_amount(monster, "Strength")
            weak = _is_weakened(monster)
            if monster.monster_id == "Cultist":
                if monster.next_move == "INCANTATION":
                    monster.intent = "BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(6 + strength, weak)
                    monster.move_hits = 1
            elif monster.monster_id == "JawWorm":
                if monster.next_move == "CHOMP":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("chomp_damage", 11)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "BELLOW":
                    monster.intent = "DEFEND_BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "THRASH":
                    monster.intent = "ATTACK_DEFEND"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("thrash_damage", 7)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "SlaverBlue":
                if monster.next_move == "STAB":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("stab_damage", 12)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "RAKE":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("rake_damage", 7)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Champ":
                if monster.next_move == "HEAVY_SLASH":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("slash_damage", 16)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "DEFENSIVE_STANCE":
                    monster.intent = "DEFEND_BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "EXECUTE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("execute_damage", 10)) + strength, weak)
                    monster.move_hits = 2
                elif monster.next_move == "FACE_SLAP":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("slap_damage", 12)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move in {"GLOAT", "ANGER"}:
                    monster.intent = "BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "TAUNT":
                    monster.intent = "DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "TheCollector":
                if monster.next_move in {"SPAWN", "REVIVE"}:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "FIREBALL":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("fireball_damage", 18)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "BUFF":
                    monster.intent = "DEFEND_BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "MEGA_DEBUFF":
                    monster.intent = "STRONG_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "TorchHead":
                monster.intent = "ATTACK"
                monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("tackle_damage", 7)) + strength, weak)
                monster.move_hits = 1
            elif monster.monster_id == "BronzeOrb":
                if monster.next_move == "BEAM":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("beam_damage", 8)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "SUPPORT":
                    monster.intent = "DEFEND"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "STASIS":
                    monster.intent = "STRONG_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "BronzeAutomaton":
                if monster.next_move == "SPAWN_ORBS":
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "FLAIL":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("flail_damage", 7)) + strength, weak)
                    monster.move_hits = 2
                elif monster.next_move == "BOOST":
                    monster.intent = "DEFEND_BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "HYPER_BEAM":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("beam_damage", 45)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "STUNNED":
                    monster.intent = "STUN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "GremlinLeader":
                if monster.next_move == "RALLY":
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "ENCOURAGE":
                    monster.intent = "DEFEND_BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "STAB":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("stab_damage", 6)) + strength, weak)
                    monster.move_hits = int(monster.meta.get("stab_hits", 3))
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "SlaverRed":
                if monster.next_move == "STAB":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("stab_damage", 13)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "SCRAPE":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("scrape_damage", 8)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "ENTANGLE":
                    monster.intent = "STRONG_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id in {"Looter", "Mugger"}:
                if monster.next_move == "MUG":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("swipe_damage", 10)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move in {"LUNGE", "BIGSWIPE"}:
                    monster.intent = "ATTACK"
                    damage_key = "big_swipe_damage" if monster.next_move == "BIGSWIPE" else "lunge_damage"
                    default_damage = 16 if monster.next_move == "BIGSWIPE" else 12
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get(damage_key, default_damage)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "SMOKE_BOMB":
                    monster.intent = "DEFEND"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "ESCAPE":
                    monster.intent = "ESCAPE"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "GremlinNob":
                if monster.next_move == "BELLOW":
                    monster.intent = "BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "SKULL_BASH":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("bash_damage", 6)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "BULL_RUSH":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("rush_damage", 14)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Lagavulin":
                if monster.next_move == "SLEEP":
                    monster.intent = "SLEEP"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "STUN":
                    monster.intent = "STUN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "DEBUFF":
                    monster.intent = "STRONG_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "STRONG_ATTACK":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("attack_damage", 18)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Sentry":
                if monster.next_move == "BOLT":
                    monster.intent = "DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "BEAM":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("beam_damage", 9)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Repulsor":
                if monster.next_move == "ATTACK":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("attack_damage", 11)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "DAZE":
                    monster.intent = "DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Spiker":
                if monster.next_move == "ATTACK":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("attack_damage", 7)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "BUFF_THORNS":
                    monster.intent = "BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Exploder":
                if monster.next_move == "ATTACK":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("attack_damage", 9)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "BLOCK":
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "GiantHead":
                if monster.next_move == "GLARE":
                    monster.intent = "DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "COUNT":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("count_damage", 13)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "IT_IS_TIME":
                    count = int(monster.meta.get("count", 0))
                    index = 1 - count
                    if index > 7:
                        index = 7
                    damage = int(monster.meta.get("starting_death_damage", 30)) + max(0, index - 1) * 5
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(damage + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Nemesis":
                if monster.next_move == "SCYTHE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("scythe_damage", 45)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "TRI_ATTACK":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("fire_damage", 6)) + strength, weak)
                    monster.move_hits = 3
                elif monster.next_move == "TRI_BURN":
                    monster.intent = "DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "AwakenedOne":
                if monster.next_move == "SLASH":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("slash_damage", 20)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "SOUL_STRIKE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("soul_strike_damage", 6)) + strength, weak)
                    monster.move_hits = 4
                elif monster.next_move == "REBIRTH":
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "DARK_ECHO":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("dark_echo_damage", 40)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "SLUDGE":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("sludge_damage", 18)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "TACKLE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("tackle_damage", 10)) + strength, weak)
                    monster.move_hits = 3
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Darkling":
                if monster.next_move == "CHOMP":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("chomp_damage", 8)) + strength, weak)
                    monster.move_hits = 2
                elif monster.next_move == "HARDEN":
                    monster.intent = "DEFEND_BUFF" if self.ascension_level >= 17 else "DEFEND"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "NIP":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("nip_damage", 7)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "COUNT":
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "REINCARNATE":
                    monster.intent = "BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Dagger":
                if monster.next_move == "WOUND":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("wound_damage", 9)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "EXPLODE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("explode_damage", 25)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Reptomancer":
                if monster.next_move == "SNAKE_STRIKE":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("snake_strike_damage", 13)) + strength, weak)
                    monster.move_hits = 2
                elif monster.next_move == "SPAWN_DAGGER":
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "BIG_BITE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("big_bite_damage", 30)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "TimeEater":
                if monster.next_move == "REVERBERATE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("reverberate_damage", 7)) + strength, weak)
                    monster.move_hits = 3
                elif monster.next_move == "RIPPLE":
                    monster.intent = "DEFEND_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "HEAD_SLAM":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("head_slam_damage", 26)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "HASTE":
                    monster.intent = "BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Donu":
                if monster.next_move == "BEAM":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("beam_damage", 10)) + strength, weak)
                    monster.move_hits = 2
                elif monster.next_move in {"CIRCLE", "CIRCLE_OF_PROTECTION"}:
                    monster.intent = "BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Deca":
                if monster.next_move == "BEAM":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("beam_damage", 10)) + strength, weak)
                    monster.move_hits = 2
                elif monster.next_move in {"SQUARE", "SQUARE_OF_PROTECTION"}:
                    monster.intent = "DEFEND_BUFF" if bool(monster.meta.get("asc19", False)) else "DEFEND"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Transient":
                monster.intent = "ATTACK"
                damage = int(monster.meta.get("starting_damage", 30)) + 10 * int(monster.meta.get("attack_index", 0))
                monster.move_adjusted_damage = self._scale_monster_attack_damage(damage + strength, weak)
                monster.move_hits = 1
            elif monster.monster_id == "Maw":
                if monster.next_move == "ROAR":
                    monster.intent = "STRONG_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "SLAM":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("slam_damage", 25)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "DROOL":
                    monster.intent = "BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "NOM":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("nom_damage", 5)) + strength, weak)
                    monster.move_hits = max(1, int(monster.meta.get("turn_count", 1)) // 2)
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id in {"SpireGrowth", "Serpent"}:
                if monster.next_move == "QUICK_TACKLE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("tackle_damage", 16)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "CONSTRICT":
                    monster.intent = "STRONG_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "SMASH":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("smash_damage", 22)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "WrithingMass":
                if monster.next_move == "BIG_HIT":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("big_hit_damage", 32)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "MULTI_HIT":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("multi_hit_damage", 7)) + strength, weak)
                    monster.move_hits = 3
                elif monster.next_move == "ATTACK_BLOCK":
                    monster.intent = "ATTACK_DEFEND"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("attack_block_damage", 15)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "ATTACK_DEBUFF":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("attack_debuff_damage", 10)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "MEGA_DEBUFF":
                    monster.intent = "STRONG_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "SlimeBoss":
                if monster.next_move == "GOOP_SPRAY":
                    monster.intent = "STRONG_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "PREP_SLAM":
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "SLAM":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("slam_damage", 35)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "TheGuardian":
                if monster.next_move == "CHARGE_UP":
                    monster.intent = "DEFEND"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "FIERCE_BASH":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("fierce_bash_damage", 32)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "VENT_STEAM":
                    monster.intent = "STRONG_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "WHIRLWIND":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("whirlwind_damage", 5)) + strength, weak)
                    monster.move_hits = int(monster.meta.get("whirlwind_hits", 4))
                elif monster.next_move == "CLOSE_UP":
                    monster.intent = "BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "ROLL_ATTACK":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("roll_damage", 9)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "TWIN_SLAM":
                    monster.intent = "ATTACK_BUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("twin_slam_damage", 8)) + strength, weak)
                    monster.move_hits = 2
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Hexaghost":
                if monster.next_move == "ACTIVATE":
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "DIVIDER":
                    monster.intent = "ATTACK"
                    divider_damage = int(monster.meta.get("divider_damage", max(1, int(self.player.current_hp) // 12 + 1)))
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(divider_damage + strength, weak)
                    monster.move_hits = int(monster.meta.get("divider_hits", 6))
                elif monster.next_move == "TACKLE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("tackle_damage", 5)) + strength, weak)
                    monster.move_hits = int(monster.meta.get("tackle_hits", 2))
                elif monster.next_move == "SEAR":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("sear_damage", 6)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "INFLAME":
                    monster.intent = "DEFEND_BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "INFERNO":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("inferno_damage", 2)) + strength, weak)
                    monster.move_hits = int(monster.meta.get("inferno_hits", 6))
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Byrd":
                if monster.next_move == "PECK":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("peck_damage", 1)) + strength, weak)
                    monster.move_hits = int(monster.meta.get("peck_count", 5))
                elif monster.next_move == "CAW":
                    monster.intent = "BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "SWOOP":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("swoop_damage", 12)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "STUNNED":
                    monster.intent = "STUN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "HEADBUTT":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("headbutt_damage", 3)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "GO_AIRBORNE":
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Snecko":
                if monster.next_move == "GLARE":
                    monster.intent = "STRONG_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "BITE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("bite_damage", 15)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "TAIL":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("tail_damage", 8)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Chosen":
                if monster.next_move == "POKE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("poke_damage", 5)) + strength, weak)
                    monster.move_hits = 2
                elif monster.next_move == "ZAP":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("zap_damage", 18)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "DEBILITATE":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("debilitate_damage", 10)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "DRAIN":
                    monster.intent = "DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "HEX":
                    monster.intent = "STRONG_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "SphericGuardian":
                if monster.next_move == "INITIAL_BLOCK_GAIN":
                    monster.intent = "DEFEND"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "FRAIL_ATTACK":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("attack_damage", 10)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "BIG_ATTACK":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("attack_damage", 10)) + strength, weak)
                    monster.move_hits = 2
                elif monster.next_move == "BLOCK_ATTACK":
                    monster.intent = "ATTACK_DEFEND"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("attack_damage", 10)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "BookOfStabbing":
                if monster.next_move == "STAB":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("stab_damage", 6)) + strength, weak)
                    monster.move_hits = int(monster.meta.get("stab_count", 1))
                elif monster.next_move == "BIG_STAB":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("big_stab_damage", 21)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "SnakePlant":
                if monster.next_move == "CHOMPY_CHOMPS":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("chomp_damage", 7)) + strength, weak)
                    monster.move_hits = 3
                elif monster.next_move == "SPORES":
                    monster.intent = "STRONG_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "ShelledParasite":
                if monster.next_move == "FELL":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("fell_damage", 18)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "DOUBLE_STRIKE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("double_damage", 6)) + strength, weak)
                    monster.move_hits = 2
                elif monster.next_move == "LIFE_SUCK":
                    monster.intent = "ATTACK_BUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("suck_damage", 10)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "STUNNED":
                    monster.intent = "STUN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Centurion":
                if monster.next_move == "SLASH":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("slash_damage", 12)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "PROTECT":
                    monster.intent = "DEFEND"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "FURY":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("fury_damage", 6)) + strength, weak)
                    monster.move_hits = int(monster.meta.get("fury_hits", 3))
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "Healer":
                if monster.next_move == "ATTACK":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("magic_damage", 8)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move in {"HEAL", "BUFF"}:
                    monster.intent = "BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "OrbWalker":
                if monster.next_move == "LASER":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("laser_damage", 10)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "CLAW":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("claw_damage", 15)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "SlaverBoss":
                if monster.next_move == "SCOURING_WHIP":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("whip_damage", 7)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "BanditChild":
                monster.intent = "ATTACK"
                monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("attack_damage", 5)) + strength, weak)
                monster.move_hits = int(monster.meta.get("attack_hits", 2))
            elif monster.monster_id == "BanditBear":
                if monster.next_move == "BEAR_HUG":
                    monster.intent = "STRONG_DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "LUNGE":
                    monster.intent = "ATTACK_DEFEND"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("lunge_damage", 9)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "MAUL":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("maul_damage", 18)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "BanditLeader":
                if monster.next_move == "MOCK":
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "AGONIZING_SLASH":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("agonize_damage", 10)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "CROSS_SLASH":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("slash_damage", 15)) + strength, weak)
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "GremlinWarrior":
                if monster.next_move == "SCRATCH":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("scratch_damage", 4)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "ESCAPE":
                    monster.intent = "ESCAPE"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "GremlinFat":
                if monster.next_move == "BLUNT":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("blunt_damage", 4)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "ESCAPE":
                    monster.intent = "ESCAPE"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "GremlinThief":
                if monster.next_move == "PUNCTURE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("puncture_damage", 9)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "ESCAPE":
                    monster.intent = "ESCAPE"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "GremlinTsundere":
                if monster.next_move == "PROTECT":
                    monster.intent = "DEFEND"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "BASH":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("bash_damage", 6)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "ESCAPE":
                    monster.intent = "ESCAPE"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "GremlinWizard":
                if monster.next_move == "CHARGE":
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "DOPE_MAGIC":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("magic_damage", 25)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "ESCAPE":
                    monster.intent = "ESCAPE"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "FungiBeast":
                if monster.next_move == "BITE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("bite_damage", 6)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "GROW":
                    monster.intent = "BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "AcidSlime_S":
                if monster.next_move == "TACKLE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("tackle_damage", 3)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "LICK":
                    monster.intent = "DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "SpikeSlime_S":
                monster.intent = "ATTACK"
                monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("tackle_damage", 5)) + strength, weak)
                monster.move_hits = 1
            elif monster.monster_id == "AcidSlime_M":
                if monster.next_move == "WOUND_TACKLE":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("wound_damage", 7)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "NORMAL_TACKLE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("normal_damage", 10)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "LICK":
                    monster.intent = "DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "SpikeSlime_M":
                if monster.next_move == "FLAME_TACKLE":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("tackle_damage", 8)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "LICK":
                    monster.intent = "DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "AcidSlime_L":
                if monster.next_move == "WOUND_TACKLE":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("wound_damage", 11)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "NORMAL_TACKLE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("normal_damage", 16)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "WEAK_LICK":
                    monster.intent = "DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "SPLIT":
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "SpikeSlime_L":
                if monster.next_move == "FLAME_TACKLE":
                    monster.intent = "ATTACK_DEBUFF"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("tackle_damage", 16)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "FRAIL_LICK":
                    monster.intent = "DEBUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                elif monster.next_move == "SPLIT":
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "FuzzyLouseNormal":
                if monster.next_move == "BITE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("bite_damage", 5)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "STRENGTHEN":
                    monster.intent = "BUFF"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            elif monster.monster_id == "FuzzyLouseDefensive":
                if monster.next_move == "BITE":
                    monster.intent = "ATTACK"
                    monster.move_adjusted_damage = self._scale_monster_attack_damage(int(monster.meta.get("bite_damage", 5)) + strength, weak)
                    monster.move_hits = 1
                elif monster.next_move == "WEAKEN":
                    monster.intent = "DEBUFF"
                    monster.move_adjusted_damage = -1
                    monster.move_hits = 1
                else:
                    monster.intent = "UNKNOWN"
                    monster.move_adjusted_damage = 0
                    monster.move_hits = 0
            else:
                monster.intent = "UNKNOWN"
                monster.move_adjusted_damage = 0
                monster.move_hits = 0
