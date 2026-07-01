from __future__ import annotations

import math
import struct
from typing import Any

from spirecomm.native_sim_v2.helpers_cards import CARD_LIBRARY, COLORLESS_CARD_ID_ORDER, COLORLESS_CARD_IDS, card_to_spirecomm, clone_card, ironclad_card_pool, ironclad_locked_card_ids, ironclad_type_rarity_card_pool, make_card, roll_colorless_card, starter_deck
from spirecomm.native_sim.mapgen import generate_act_map
from spirecomm.native_sim.potions import empty_potion_slots, get_random_potion, make_potion, potions_to_spirecomm, roll_potion
from spirecomm.native_sim.randoms import NativeRandomStreams, StsRandom, java_collections_shuffle
from spirecomm.native_sim_v2.helpers_relics import draw_relic_from_pool, init_ironclad_relic_pools, ironclad_locked_relic_ids, make_relic
from spirecomm.native_sim.schema import CardInstance, MonsterState, PlayerState, PotionInstance

def _sts_round(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0 else int(math.ceil(value - 0.5))

def _f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]

def _card_can_upgrade(card: CardInstance) -> bool:
    if card.card_id == "Searing Blow":
        return True
    if card.card_def.card_type not in {"ATTACK", "SKILL", "POWER"}:
        return False
    return card.upgrades <= 0

def _card_can_armaments_plus_upgrade(card: CardInstance) -> bool:
    if card.card_id == "Burn":
        return card.upgrades <= 0
    return _card_can_upgrade(card)

def _card_upgrade_count(card: CardInstance) -> int:
    if card.card_id == "Searing Blow":
        return max(int(card.upgrades), int(card.misc))
    return int(card.upgrades)

def _ensure_card_upgraded(card: CardInstance) -> None:
    if card.card_id == "Searing Blow":
        if _card_upgrade_count(card) <= 0:
            card.upgrades = 1
            card.misc = 1
        return
    card.upgrades = max(card.upgrades, 1)

def _increment_card_upgrade(card: CardInstance) -> None:
    if card.card_id == "Searing Blow":
        next_count = _card_upgrade_count(card) + 1
        card.upgrades = next_count
        card.misc = next_count
        return
    card.upgrades += 1

def _card_can_transform(card: CardInstance) -> bool:
    return card.card_id not in {"AscendersBane", "Necronomicurse", "CurseOfTheBell"}

STRIKE_CARD_IDS = {
    "Meteor Strike",
    "Perfected Strike",
    "Pommel Strike",
    "Sneaky Strike",
    "Strike_B",
    "Strike_G",
    "Strike_P",
    "Strike_R",
    "Swift Strike",
    "Thunder Strike",
    "Twin Strike",
    "Wild Strike",
    "Windmill Strike",
}

NEOW_BONUS_LABELS = {
    "THREE_CARDS": "Choose a card to obtain.",
    "ONE_RANDOM_RARE_CARD": "Obtain a random rare card.",
    "REMOVE_CARD": "Remove a card.",
    "UPGRADE_CARD": "Upgrade a card.",
    "TRANSFORM_CARD": "Transform a card.",
    "RANDOM_COLORLESS": "Choose a colorless card to obtain.",
    "THREE_SMALL_POTIONS": "Obtain three potions.",
    "RANDOM_COMMON_RELIC": "Obtain a random common relic.",
    "TEN_PERCENT_HP_BONUS": "Max Hp +10%.",
    "THREE_ENEMY_KILL": "Obtain Neow's Lament.",
    "HUNDRED_GOLD": "Obtain 100 gold.",
    "RANDOM_COLORLESS_2": "Choose a rare colorless card to obtain.",
    "REMOVE_TWO": "Remove two cards.",
    "ONE_RARE_RELIC": "Obtain a random rare relic.",
    "THREE_RARE_CARDS": "Choose a rare card to obtain.",
    "TWO_FIFTY_GOLD": "Obtain 250 gold.",
    "TRANSFORM_TWO_CARDS": "Transform two cards in your deck.",
    "TWENTY_PERCENT_HP_BONUS": "Max Hp +20%.",
    "BOSS_RELIC": "Obtain a random boss relic.",
}

NEOW_DRAWBACK_LABELS = {
    "INVALID": "INVALID",
    "NONE": "",
    "TEN_PERCENT_HP_LOSS": "Max Hp -10%.",
    "NO_GOLD": "Lose all gold.",
    "CURSE": "Obtain a curse.",
    "PERCENT_DAMAGE": "Take 30% Hp damage.",
    "LOSE_STARTER_RELIC": "Lose your starter relic.",
}

TRANSFORM_CARD_POOL_IRONCLAD: tuple[str, ...] = (
    "Anger", "Cleave", "Warcry", "Flex", "Iron Wave", "Body Slam", "True Grit", "Shrug It Off", "Clash", "Thunderclap",
    "Pommel Strike", "Twin Strike", "Clothesline", "Armaments", "Havoc", "Headbutt", "Wild Strike", "Heavy Blade",
    "Perfected Strike", "Sword Boomerang", "Evolve", "Uppercut", "Ghostly Armor", "Fire Breathing", "Dropkick",
    "Carnage", "Bloodletting", "Rupture", "Second Wind", "Searing Blow", "Battle Trance", "Sentinel", "Entrench",
    "Rage", "Feel No Pain", "Disarm", "Seeing Red", "Dark Embrace", "Combust", "Whirlwind", "Sever Soul", "Rampage",
    "Shockwave", "Metallicize", "Burning Pact", "Pummel", "Flame Barrier", "Blood for Blood", "Intimidate",
    "Hemokinesis", "Reckless Charge", "Infernal Blade", "Dual Wield", "Power Through", "Inflame", "Spot Weakness",
    "Double Tap", "Demon Form", "Bludgeon", "Feed", "Limit Break", "Corruption", "Barricade", "Fiend Fire", "Berserk",
    "Impervious", "Juggernaut", "Brutality", "Reaper", "Exhume", "Offering", "Immolate",
)

COMBAT_CARD_POOL_IRONCLAD: tuple[str, ...] = (
    "Sword Boomerang", "Perfected Strike", "Heavy Blade", "Wild Strike", "Headbutt", "Havoc", "Armaments",
    "Clothesline", "Twin Strike", "Pommel Strike", "Thunderclap", "Clash", "Shrug It Off", "True Grit", "Body Slam",
    "Iron Wave", "Flex", "Warcry", "Cleave", "Anger", "Evolve", "Uppercut", "Ghostly Armor", "Fire Breathing",
    "Dropkick", "Carnage", "Bloodletting", "Rupture", "Second Wind", "Searing Blow", "Battle Trance", "Sentinel",
    "Entrench", "Rage", "Feel No Pain", "Disarm", "Seeing Red", "Dark Embrace", "Combust", "Whirlwind",
    "Sever Soul", "Rampage", "Shockwave", "Metallicize", "Burning Pact", "Pummel", "Flame Barrier",
    "Blood for Blood", "Intimidate", "Hemokinesis", "Reckless Charge", "Infernal Blade", "Dual Wield",
    "Power Through", "Inflame", "Spot Weakness", "Double Tap", "Demon Form", "Bludgeon", "Limit Break",
    "Corruption", "Barricade", "Fiend Fire", "Berserk", "Impervious", "Juggernaut", "Brutality", "Exhume",
    "Offering", "Immolate",
)

COMBAT_ATTACK_CARD_POOL_IRONCLAD: tuple[str, ...] = (
    "Sword Boomerang", "Perfected Strike", "Heavy Blade", "Wild Strike", "Headbutt", "Clothesline", "Twin Strike",
    "Pommel Strike", "Thunderclap", "Clash", "Body Slam", "Iron Wave", "Cleave", "Anger", "Uppercut", "Dropkick",
    "Carnage", "Searing Blow", "Whirlwind", "Sever Soul", "Rampage", "Pummel", "Blood for Blood", "Hemokinesis",
    "Reckless Charge", "Bludgeon", "Fiend Fire", "Immolate",
)

COMBAT_SKILL_CARD_POOL_IRONCLAD: tuple[str, ...] = (
    "Havoc", "Armaments", "Shrug It Off", "True Grit", "Flex", "Warcry", "Ghostly Armor", "Bloodletting",
    "Second Wind", "Battle Trance", "Sentinel", "Entrench", "Rage", "Disarm", "Seeing Red", "Shockwave",
    "Burning Pact", "Flame Barrier", "Intimidate", "Infernal Blade", "Dual Wield", "Power Through",
    "Spot Weakness", "Double Tap", "Limit Break", "Impervious", "Exhume", "Offering",
)

COMBAT_POWER_CARD_POOL_IRONCLAD: tuple[str, ...] = (
    "Evolve", "Fire Breathing", "Rupture", "Feel No Pain", "Dark Embrace", "Combust", "Metallicize", "Inflame",
    "Demon Form", "Corruption", "Barricade", "Berserk", "Juggernaut", "Brutality",
)

COMBAT_COLORLESS_CARD_POOL: tuple[str, ...] = (
    "Madness", "Thinking Ahead", "Mind Blast", "Metamorphosis", "Jack Of All Trades", "Swift Strike",
    "Good Instincts", "Master of Strategy", "Magnetism", "Finesse", "Discovery", "Chrysalis", "Transmutation",
    "Panacea", "Purity", "Enlightenment", "Forethought", "Flash of Steel", "Hand of Greed", "Mayhem",
    "Apotheosis", "Secret Weapon", "Panache", "Violence", "Deep Breath", "Secret Technique", "Blind", "The Bomb",
    "Impatience", "Dramatic Entrance", "Trip", "Panic Button", "Sadistic Nature", "Dark Shackles",
)

COMBAT_CARD_SORT_ORDER: tuple[str, ...] = COMBAT_CARD_POOL_IRONCLAD + COMBAT_COLORLESS_CARD_POOL
COMBAT_CARD_SORT_INDEX: dict[str, int] = {
    card_id: index for index, card_id in enumerate(COMBAT_CARD_SORT_ORDER)
}
BRONZE_ORB_STASIS_SORT_INDEX_OVERRIDES: dict[str, int] = {
    "Barricade": 22,
    "Burning Pact": 49,
    "Carnage": 54,
    "Dark Embrace": 89,
    "Feel No Pain": 145,
    "Intimidate": 195,
}

NEOW_MID_TIER_BY_DRAWBACK = {
    "TEN_PERCENT_HP_LOSS": [
        "RANDOM_COLORLESS_2",
        "REMOVE_TWO",
        "ONE_RARE_RELIC",
        "THREE_RARE_CARDS",
        "TWO_FIFTY_GOLD",
        "TRANSFORM_TWO_CARDS",
    ],
    "NO_GOLD": [
        "RANDOM_COLORLESS_2",
        "REMOVE_TWO",
        "ONE_RARE_RELIC",
        "THREE_RARE_CARDS",
        "TRANSFORM_TWO_CARDS",
        "TWENTY_PERCENT_HP_BONUS",
    ],
    "CURSE": [
        "RANDOM_COLORLESS_2",
        "ONE_RARE_RELIC",
        "THREE_RARE_CARDS",
        "TWO_FIFTY_GOLD",
        "TRANSFORM_TWO_CARDS",
        "TWENTY_PERCENT_HP_BONUS",
    ],
}

def _apply_damage(amount: int, monster: MonsterState) -> int:
    if amount <= 0 or not monster.alive:
        return 0
    blocked = min(monster.block, amount)
    monster.block -= blocked
    hp_damage = amount - blocked
    hp_lost = min(monster.current_hp, hp_damage)
    monster.current_hp = max(0, monster.current_hp - hp_damage)
    if monster.current_hp <= 0:
        monster.is_gone = True
    return hp_lost

def _player_attack_damage(base: int, player: PlayerState, monster: MonsterState) -> int:
    damage = base + player.power("Strength")
    if player.power("Weakened") > 0:
        damage = int(damage * 0.75)
    if monster.power("Vulnerable") > 0:
        damage = int(damage * 1.5)
    return max(0, damage)

def _player_block_amount(base: int, player: PlayerState) -> int:
    if player.power("No Block") > 0:
        return 0
    block = base + player.power("Dexterity")
    if player.power("Frail") > 0:
        block = int(block * 0.75)
    return max(0, block)

def _base_damage_for_card(card: CardInstance) -> int:
    upgraded = card.upgrades > 0
    if card.card_id == "Strike_R":
        return 9 if upgraded else 6
    if card.card_id == "Bash":
        return 10 if upgraded else 8
    if card.card_id == "Anger":
        return 8 if upgraded else 6
    if card.card_id == "Clash":
        return 18 if upgraded else 14
    if card.card_id == "Cleave":
        return 11 if upgraded else 8
    if card.card_id == "Clothesline":
        return 14 if upgraded else 12
    if card.card_id == "Headbutt":
        return 12 if upgraded else 9
    if card.card_id == "Heavy Blade":
        return 14
    if card.card_id == "Iron Wave":
        return 7 if upgraded else 5
    if card.card_id == "Pommel Strike":
        return 10 if upgraded else 9
    if card.card_id == "Sword Boomerang":
        return 3
    if card.card_id == "Thunderclap":
        return 7 if upgraded else 4
    if card.card_id == "Twin Strike":
        return 7 if upgraded else 5
    if card.card_id == "Wild Strike":
        return 17 if upgraded else 12
    if card.card_id == "Blood for Blood":
        return 22 if upgraded else 18
    if card.card_id == "Carnage":
        return 28 if upgraded else 20
    if card.card_id == "Dropkick":
        return 8 if upgraded else 5
    if card.card_id == "Dramatic Entrance":
        return 12 if upgraded else 8
    if card.card_id == "Hemokinesis":
        return 20 if upgraded else 15
    if card.card_id == "Pummel":
        return 2
    if card.card_id == "Rampage":
        return 8
    if card.card_id == "Reckless Charge":
        return 10 if upgraded else 7
    if card.card_id == "Searing Blow":
        n = _card_upgrade_count(card)
        return n * (n + 7) // 2 + 12
    if card.card_id == "Sever Soul":
        return 22 if upgraded else 16
    if card.card_id == "Uppercut":
        return 13
    if card.card_id == "Whirlwind":
        return 8 if upgraded else 5
    if card.card_id == "Bludgeon":
        return 42 if upgraded else 32
    if card.card_id == "Feed":
        return 12 if upgraded else 10
    if card.card_id == "Fiend Fire":
        return 10 if upgraded else 7
    if card.card_id == "Immolate":
        return 28 if upgraded else 21
    if card.card_id == "Reaper":
        return 5 if upgraded else 4
    if card.card_id == "Flash of Steel":
        return 6 if upgraded else 3
    if card.card_id == "Swift Strike":
        return 10 if upgraded else 7
    if card.card_id in {"Hand of Greed", "HandOfGreed"}:
        return 25 if upgraded else 20
    if card.card_id == "Bite":
        return 8 if upgraded else 7
    if card.card_id == "Ritual Dagger":
        return 20 if upgraded else 15
    return 0

def _spirecomm_monster_id(monster_id: str) -> str:
    return {
        "RedLouse": "FuzzyLouseNormal",
        "GreenLouse": "FuzzyLouseDefensive",
        "TheChamp": "Champ",
        "TheCollector": "Collector",
        "ShelledParasite": "Shelled Parasite",
        "GremlinWarrior": "GremlinWarrior",
        "GremlinThief": "GremlinThief",
        "GremlinFat": "GremlinFat",
        "GremlinTsundere": "GremlinTsundere",
        "GremlinWizard": "GremlinWizard",
        "Mystic": "Healer",
        "Romeo": "SlaverBoss",
    }.get(monster_id, monster_id)

def _serialize_move_name(move: str | None) -> str | None:
    if move is None:
        return None
    return {
        "CHAMP_HEAVY_SLASH": "THE_CHAMP_HEAVY_SLASH",
        "CHAMP_DEFENSIVE_STANCE": "THE_CHAMP_DEFENSIVE_STANCE",
        "CHAMP_EXECUTE": "THE_CHAMP_EXECUTE",
        "CHAMP_FACE_SLAP": "THE_CHAMP_FACE_SLAP",
        "CHAMP_GLOAT": "THE_CHAMP_GLOAT",
        "CHAMP_TAUNT": "THE_CHAMP_TAUNT",
        "CHAMP_ANGER": "THE_CHAMP_ANGER",
        "BOOK_MULTI_STAB": "BOOK_OF_STABBING_MULTI_STAB",
        "MYSTIC_ATTACK": "MYSTIC_ATTACK_DEBUFF",
        "THE_COLLECTOR_SPAWN": "THE_COLLECTOR_SPAWN",
        "COLLECTOR_BUFF": "THE_COLLECTOR_BUFF",
        "COLLECTOR_FIREBALL": "THE_COLLECTOR_FIREBALL",
        "COLLECTOR_MEGA_DEBUFF": "THE_COLLECTOR_MEGA_DEBUFF",
        "SHELLED_DOUBLE_STRIKE": "SHELLED_PARASITE_DOUBLE_STRIKE",
        "SHELLED_FELL": "SHELLED_PARASITE_FELL",
        "SHELLED_STUNNED": "SHELLED_PARASITE_STUNNED",
        "SHELLED_SUCK": "SHELLED_PARASITE_SUCK",
    }.get(move, move)

def _serialize_named_power(power_id: str, amount: int) -> dict[str, Any]:
    return {
        "power_id": power_id,
        "id": power_id,
        "name": power_id,
        "amount": amount,
        "card": None,
        "damage": 0,
        "just_applied": False,
        "misc": amount,
    }

def _combat_strike_count(env: Any, current_card: CardInstance | None = None) -> int:
    cards: list[CardInstance] = []
    if current_card is not None:
        cards.append(current_card)
    cards.extend(card for card in env.hand if card is not current_card)
    cards.extend(card for card in env.draw_pile if card is not current_card)
    cards.extend(card for card in env.discard_pile if card is not current_card)
    return sum(1 for card in cards if card.card_id in STRIKE_CARD_IDS)


__all__ = [
    name
    for name in globals()
    if name not in {"math", "struct"} and not name.startswith("__")
]
