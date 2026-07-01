from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import struct
from typing import Any

from spirecomm.native_sim.cards import COLORLESS_CARD_ID_ORDER, COLORLESS_CARD_IDS, CARD_LIBRARY, card_to_spirecomm, clone_card, ironclad_card_pool, ironclad_locked_card_ids, ironclad_type_rarity_card_pool, make_card, roll_colorless_card, starter_deck
from spirecomm.native_sim.mapgen import generate_act_map
from spirecomm.native_sim.monsters import (
    _set_move,
    choose_next_move,
    encounter_to_monster_ids,
    encounter_to_monsters,
    generate_elite_schedule,
    generate_monster_schedules,
    generate_strong_monster_schedule,
    make_monster,
    monster_adjusted_damage,
    roll_act1_encounter,
)
from spirecomm.native_sim.potions import empty_potion_slots, get_random_potion, make_potion, potions_to_spirecomm, roll_potion
from spirecomm.native_sim.randoms import NativeRandomStreams, StsRandom, java_collections_shuffle
from spirecomm.native_sim.relics import (
    draw_relic_from_pool,
    init_ironclad_relic_pools,
    ironclad_locked_relic_ids,
    make_relic,
)
from spirecomm.native_sim.schema import CardInstance, MonsterState, PlayerState, PotionInstance


def _sts_round(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0 else int(math.ceil(value - 0.5))


def _f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _card_can_upgrade(card: CardInstance) -> bool:
    if card.card_def.card_type not in {"ATTACK", "SKILL", "POWER"}:
        return False
    if card.card_id == "Searing Blow":
        return True
    return card.upgrades <= 0


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
    if card.card_id == "Hemokinesis":
        return 20 if upgraded else 15
    if card.card_id == "Pummel":
        return 2
    if card.card_id == "Rampage":
        return 8
    if card.card_id == "Reckless Charge":
        return 10 if upgraded else 7
    if card.card_id == "Searing Blow":
        return 12 + card.upgrades * 4
    if card.card_id == "Sever Soul":
        return 22 if upgraded else 16
    if card.card_id == "Uppercut":
        return 13 if upgraded else 10
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
        "MYSTIC_ATTACK": "MYSTIC_ATTACK_DEBUFF",
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
    cards.extend(env.hand)
    cards.extend(env.draw_pile)
    cards.extend(env.discard_pile)
    cards.extend(env.exhaust_pile)
    return sum(1 for card in cards if card.card_id in STRIKE_CARD_IDS)


@dataclass
class NativeCombatEnv:
    seed: int
    ascension_level: int = 0
    floor: int = 1
    act: int = 1
    act_boss: str = "Hexaghost"
    elite: bool = False
    external_misc_rng: StsRandom | None = None
    rng: StsRandom = field(init=False)
    ai_rng: StsRandom = field(init=False)
    monster_hp_rng: StsRandom = field(init=False)
    shuffle_rng: StsRandom = field(init=False)
    card_random_rng: StsRandom = field(init=False)
    misc_rng: StsRandom = field(init=False)
    player: PlayerState = field(default_factory=PlayerState)
    deck: list[CardInstance] = field(default_factory=starter_deck)
    draw_pile: list[CardInstance] = field(default_factory=list)
    discard_pile: list[CardInstance] = field(default_factory=list)
    exhaust_pile: list[CardInstance] = field(default_factory=list)
    hand: list[CardInstance] = field(default_factory=list)
    monsters: list[MonsterState] = field(default_factory=list)
    relics: list[dict[str, Any]] = field(default_factory=lambda: [make_relic("Burning Blood")])
    potions: list[PotionInstance] = field(default_factory=empty_potion_slots)
    gold: int = 99
    turn: int = 0
    cards_discarded_this_turn: int = 0
    hp_lost_this_combat: int = 0
    hp_loss_events_this_combat: int = 0
    attack_played_this_turn: int = 0
    skills_played_this_turn: int = 0
    cards_played_this_turn: int = 0
    card_types_played_this_turn: set[str] = field(default_factory=set)
    locked_card_ids: set[str] = field(default_factory=set)
    outcome: str = "UNDECIDED"
    gold_gain: int = 0
    reward_gold_bonus: int = 0
    monster_damage_this_turn: dict[int, int] = field(default_factory=dict)
    player_powers_just_applied: set[str] = field(default_factory=set)
    card_select_context: str | None = None
    card_select_options: list[dict[str, Any]] = field(default_factory=list)
    card_select_generated_cards: list[CardInstance] = field(default_factory=list)
    pending_resolve_card: CardInstance | None = None
    pending_resolve_force_exhaust: bool = False
    pending_resolve_hex_dazed: int = 0
    pending_counter_damage: int = 0
    pending_juggernaut_damage: int = 0
    pending_attack_replays: int = 0
    pending_attack_target_index: int | None = None
    pending_bottle_relic_id: str | None = None
    opening_innate_count: int = 0
    combust_hp_loss: int = 0
    pending_monster_block_gains: list[tuple[MonsterState, int]] = field(default_factory=list)
    pending_autoplay_cards: list[tuple[CardInstance, int, bool, int | None]] = field(default_factory=list)
    _next_uuid: int = 0
    _double_tap_active: bool = False
    _draw_triggered_shuffle: bool = False
    _processing_autoplay_cards: bool = False
    scheduled_encounter: list[str] | str | None = None

    def __post_init__(self) -> None:
        start_random = StsRandom(self.seed + self.floor)
        self.rng = start_random.copy()
        self.ai_rng = start_random.copy()
        self.monster_hp_rng = start_random.copy()
        self.monster_hp_rng.ascension_level = self.ascension_level
        self.shuffle_rng = start_random.copy()
        self.card_random_rng = start_random.copy()
        self.misc_rng = self.external_misc_rng.copy() if self.external_misc_rng is not None else start_random.copy()
        self.start_combat(elite=self.elite)

    @property
    def in_combat(self) -> bool:
        return self.outcome == "UNDECIDED" and any(monster.alive for monster in self.monsters) and self.player.current_hp > 0

    def start_combat(self, *, elite: bool = False) -> None:
        if self.scheduled_encounter:
            if isinstance(self.scheduled_encounter, str):
                self.monsters = encounter_to_monsters(
                    self.scheduled_encounter,
                    self.monster_hp_rng,
                    self.misc_rng,
                    self.ascension_level,
                )
            else:
                self.monsters = [make_monster(monster_id, self.monster_hp_rng) for monster_id in self.scheduled_encounter]
        else:
            self.monsters = roll_act1_encounter(self.monster_hp_rng, floor=self.floor, act=self.act, act_boss=self.act_boss, elite=elite)
        if elite and self._has_relic("Preserved Insect"):
            for monster in self.monsters:
                monster.max_hp = max(1, int(monster.max_hp * 0.75))
                monster.current_hp = min(monster.current_hp, monster.max_hp)
        for index, monster in enumerate(self.monsters):
            monster.ai_state["ascension_level"] = self.ascension_level
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", self.monsters)
        self._apply_ascension_monster_scaling(elite=elite)
        for monster in self.monsters:
            if monster.alive:
                choose_next_move(monster, self.ai_rng)
        self._apply_monster_prebattle_actions()
        for card in self.deck:
            card.cost_for_turn = None
            if card.card_id == "Blood for Blood":
                card.misc = 0
        self._init_opening_draw_pile()
        self.discard_pile = []
        self.exhaust_pile = []
        self.hand = []
        self.turn = 0
        self.hp_lost_this_combat = 0
        self.combust_hp_loss = 0
        self.pending_monster_block_gains = []
        self.gold_gain = 0
        self.reward_gold_bonus = 0
        self.monster_damage_this_turn = {}
        self.player_powers_just_applied = set()
        self.outcome = "UNDECIDED"
        self.player.energy = 0
        relic = self._relic("Centennial Puzzle")
        if relic is not None:
            # Centennial Puzzle refreshes each combat; the 0/1 counter here is
            # only a per-combat used flag, not a run-long persistent state.
            relic["counter"] = 0
        for relic_id in ("Incense Burner", "Necronomicon"):
            relic = self._relic(relic_id)
            if relic is not None and int(relic.get("counter", -1)) < 0:
                relic["counter"] = int(relic.get("counter", -1))
        self._apply_start_combat_relics()
        self._start_opening_turn()

    def _apply_ascension_monster_scaling(self, *, elite: bool) -> None:
        is_boss = self.floor in {16, 33, 50, 54}
        is_act4_elite = self.floor == 53
        hp_scale = 1.0
        damage_bonus = 0
        strength_bonus = 0
        if self.ascension_level >= 2 and not elite and not is_boss and not is_act4_elite:
            hp_scale += 0.05
            damage_bonus += 1
        if self.ascension_level >= 3 and (elite or is_act4_elite):
            hp_scale += 0.08
            damage_bonus += 1
        if self.ascension_level >= 4 and is_boss:
            hp_scale += 0.08
            damage_bonus += 1
        if self.ascension_level >= 17 and not elite and not is_boss and not is_act4_elite:
            strength_bonus += 1
        if self.ascension_level >= 18 and (elite or is_act4_elite):
            strength_bonus += 1
        if self.ascension_level >= 19 and is_boss:
            strength_bonus += 1
        for monster in self.monsters:
            if hp_scale != 1.0:
                old_max = monster.max_hp
                monster.max_hp = max(1, int(monster.max_hp * hp_scale))
                monster.current_hp = max(1, int(monster.current_hp * monster.max_hp / max(1, old_max)))
            if damage_bonus and monster.move_base_damage > 0:
                monster.move_base_damage += damage_bonus
            if strength_bonus:
                monster.add_power("Strength", strength_bonus)

    def _has_relic(self, relic_id: str) -> bool:
        return any(relic.get("relic_id") == relic_id for relic in self.relics)

    def _relic(self, relic_id: str) -> dict[str, Any] | None:
        return next((relic for relic in self.relics if relic.get("relic_id") == relic_id), None)

    def _ironclad_card_pool(self, *, card_type: str | None = None, rarity: str | None = None):
        return ironclad_card_pool(card_type=card_type, rarity=rarity, exclude_ids=self.locked_card_ids)

    def _random_class_card_of_rarity(self, rarity: str) -> CardInstance:
        pool = self._ironclad_card_pool(rarity=rarity)
        if not pool:
            pool = self._ironclad_card_pool()
        chosen = pool[int(self.randoms.card.random(len(pool) - 1))]
        return make_card(chosen.card_id, uuid=f"rarity-{self.floor}-{chosen.card_id}")

    def _random_class_card_of_rarity_with_rng(self, rarity: str, rng: StsRandom) -> CardInstance:
        pool = self._ironclad_card_pool(rarity=rarity)
        if not pool:
            pool = self._ironclad_card_pool()
        chosen = pool[int(rng.random(len(pool) - 1))]
        return self._make_deck_card(chosen.card_id, uuid=f"rarity-{self.floor}-{chosen.card_id}")

    def _consume_match_and_keep_rng(self) -> list[CardInstance]:
        cards = [
            self._random_class_card_of_rarity("RARE"),
            self._random_class_card_of_rarity("UNCOMMON"),
            self._random_class_card_of_rarity("COMMON"),
        ]
        # returnColorlessCard(UNCOMMON) shuffles the colorless pool using the
        # shuffle RNG, not the card RNG.
        self.randoms.shuffle.random_long()
        colorless_pool = [CARD_LIBRARY[card_id] for card_id in COLORLESS_CARD_ID_ORDER if CARD_LIBRARY[card_id].rarity == "UNCOMMON"]
        if colorless_pool:
            cards.append(make_card(colorless_pool[0].card_id, uuid=f"match-colorless-{self.floor}"))
        curse_pool = [
            card_id for card_id, card_def in CARD_LIBRARY.items()
            if card_def.card_type == "CURSE" and card_id not in {"AscendersBane", "CurseOfTheBell"}
        ]
        if curse_pool:
            self.randoms.card.random(len(curse_pool) - 1)
        cards.append(make_card("Strike_R", uuid=f"match-starter-{self.floor}"))
        self.randoms.misc.random_long()
        return cards

    def _advance_relic_counter(self, relic_id: str, threshold: int) -> bool:
        relic = self._relic(relic_id)
        if relic is None:
            return False
        counter = int(relic.get("counter", -1))
        if counter < 0:
            counter = 0
        counter += 1
        if counter >= threshold:
            relic["counter"] = 0
            return True
        relic["counter"] = counter
        return False

    def _apply_start_combat_relics(self) -> None:
        if self._has_relic("Vajra"):
            self.player.add_power("Strength", 1)
        if self._has_relic("Oddly Smooth Stone"):
            self.player.add_power("Dexterity", 1)
        if self._has_relic("Akabeko"):
            self.player.add_power("Akabeko", 8)
        if self._has_relic("Mutagenic Strength"):
            self.player.add_power("Strength", 3)
            self._apply_player_power("Flex Strength Down", 3)
        if self._has_relic("Bronze Scales"):
            self.player.add_power("Thorns", 3)
        if self._has_relic("Thread and Needle"):
            self.player.add_power("Plated Armor", 4)
        if self._has_relic("Fossilized Helix"):
            self.player.add_power("Buffer", 1)
        if self._has_relic("Pantograph") and self.floor in {16, 33, 50, 54}:
            self._heal(25)
        if self._has_relic("Philosopher's Stone"):
            for monster in self.monsters:
                monster.add_power("Strength", 1)
        if self._has_relic("Sling of Courage") and self.elite:
            self.player.add_power("Strength", 2)
        if (girya := self._relic("Girya")) is not None and int(girya.get("counter", 0)) > 0:
            self.player.add_power("Strength", int(girya.get("counter", 0)))
        if self._has_relic("Snecko Eye"):
            self._apply_player_power("Confusion", 1)
        if self._has_relic("Brimstone"):
            self.player.add_power("Strength", 2)
            for monster in self.monsters:
                if monster.alive:
                    monster.add_power("Strength", 1)
        if self._has_relic("Incense Burner") and self._advance_relic_counter("Incense Burner", 6):
            self.player.add_power("Intangible", 1)
        if self._has_relic("Mercury Hourglass"):
            self._deal_direct_damage_all(3)

    def _card_draw_per_turn(self) -> int:
        count = 5
        if self._has_relic("Snecko Eye"):
            count += 2
        if self._has_relic("Ring of the Serpent"):
            count += 1
        return count

    def _monster_vulnerable_multiplier(self) -> float:
        return 1.25 if self._has_relic("Odd Mushroom") else 1.5

    def _apply_opening_pre_draw_relics(self) -> None:
        miracle_count = 0
        if self._has_relic("Holy Water"):
            miracle_count += 3
        if self._has_relic("Pure Water"):
            miracle_count += 1
        for _ in range(miracle_count):
            if len(self.hand) < 10:
                self.hand.append(make_card("Miracle", uuid=self._new_uuid("Miracle")))
        if self._has_relic("Ninja Scroll"):
            for _ in range(3):
                if len(self.hand) < 10:
                    self.hand.append(make_card("Shiv", uuid=self._new_uuid("Shiv")))
        if self._has_relic("Toolbox"):
            self._add_random_card_to_hand(colorless=True, cost_for_turn=0)

    def _apply_opening_post_draw_relics(self) -> None:
        if self._has_relic("Bag of Marbles"):
            for monster in self.monsters:
                self._apply_monster_power(monster, "Vulnerable", 1)
        if self._has_relic("Bag of Preparation"):
            self.draw_cards(2)
        if self._has_relic("Clockwork Souvenir"):
            self.player.add_power("Artifact", 1)
        if self._has_relic("Red Mask"):
            for monster in self.monsters:
                self._apply_monster_power(monster, "Weakened", 1)
        if self._has_relic("Mark of Pain"):
            for _ in range(2):
                self.draw_pile.append(make_card("Wound", uuid=self._new_uuid("Wound")))
            self._shuffle_cards(self.draw_pile)
        if self._has_relic("Ring of the Snake"):
            self.draw_cards(2)
        if self._has_relic("Twisted Funnel"):
            for monster in self.monsters:
                self._apply_monster_power(monster, "Poison", 4)

    def _apply_happy_flower_start_of_turn(self) -> None:
        relic = self._relic("Happy Flower")
        if relic is None:
            return
        counter = int(relic.get("counter", 0))
        if counter < 0:
            counter = 0
        counter += 1
        if counter >= 3:
            self.player.energy += 1
            counter = 0
        relic["counter"] = counter

    def _start_opening_turn(self) -> None:
        self.turn = 1
        self.monster_damage_this_turn = {}
        self.cards_discarded_this_turn = 0
        self.attack_played_this_turn = 0
        self.skills_played_this_turn = 0
        self.cards_played_this_turn = 0
        self.card_types_played_this_turn = set()
        self.player.energy = self._base_energy()
        self._apply_happy_flower_start_of_turn()
        self._apply_opening_pre_draw_relics()
        self.draw_cards(self._card_draw_per_turn())
        if self.opening_innate_count > self._card_draw_per_turn():
            self.draw_cards(self.opening_innate_count - self._card_draw_per_turn())
        self._apply_opening_post_draw_relics()
        if self._has_relic("Anchor"):
            self.player.block += 10
        if self._has_relic("Lantern"):
            self.player.energy += 1
        if self._has_relic("Blood Vial"):
            self._heal(2)
        if self._has_relic("Ancient Tea Set"):
            relic = self._relic("Ancient Tea Set")
            if relic is not None and int(relic.get("counter", 0)) > 0:
                self.player.energy += 2
                relic["counter"] = 0
        if self._has_relic("Warped Tongs"):
            candidates = [card for card in self.hand if card.card_def.card_type not in {"STATUS", "CURSE"} and card.upgrades <= 0]
            if candidates:
                self._upgrade_combat_card(self.card_random_rng.choice(candidates))
        if self._has_relic("Gambling Chip") and self.hand:
            self._open_combat_card_select("GAMBLE", [0])

    def _apply_monster_end_of_turn_triggers(self, monster: MonsterState) -> None:
        if not monster.alive:
            return
        if monster.power("Metallicize") > 0:
            monster.block += monster.power("Metallicize")
        if monster.power("Malleable") > 0:
            monster.powers["Malleable"] = 3
        if monster.power("Plated Armor") > 0:
            monster.block += monster.power("Plated Armor")
        if monster.power("Intangible") > 0:
            monster.add_power("Intangible", -1)
            if monster.power("Intangible") <= 0:
                monster.powers.pop("Intangible", None)
        if monster.power("Regenerate") > 0:
            monster.current_hp = min(monster.max_hp, monster.current_hp + monster.power("Regenerate"))

    def _apply_monster_prebattle_actions(self) -> None:
        for monster in self.monsters:
            pending = monster.ai_state.pop("pending_prebattle", None)
            if pending == "LOUSE":
                if self.ascension_level >= 17:
                    monster.add_power("Curl Up", self.monster_hp_rng.randint(9, 12))
                elif self.ascension_level >= 7:
                    monster.add_power("Curl Up", self.monster_hp_rng.randint(4, 8))
                else:
                    monster.add_power("Curl Up", self.monster_hp_rng.randint(3, 7))

    def start_player_turn(self) -> None:
        self.turn += 1
        self.monster_damage_this_turn = {}
        for pile in (self.hand, self.draw_pile, self.discard_pile):
            for card in pile:
                if card.uuid.startswith("Discovery-"):
                    card.cost_for_turn = None
        base_energy = self._base_energy()
        self.player.energy = self.player.energy + base_energy if self._has_relic("Ice Cream") else base_energy
        if self.player.powers.pop("Art of War Energy", 0):
            self.player.energy += 1
        if self.player.powers.pop("Pocketwatch Draw", 0):
            self.draw_cards(3)
        if self._has_relic("Ancient Tea Set") and self.turn == 1:
            relic = self._relic("Ancient Tea Set")
            if relic is not None and int(relic.get("counter", 0)) > 0:
                self.player.energy += 2
                relic["counter"] = 0
        if self._has_relic("Incense Burner") and self._advance_relic_counter("Incense Burner", 6):
            self.player.add_power("Intangible", 1)
        if self.player.power("Barricade") <= 0 and self._has_relic("Calipers"):
            self.player.block = max(0, self.player.block - 15)
        elif self.player.power("Barricade") <= 0:
            self.player.block = 0
        if self._has_relic("Red Skull") and self.player.current_hp <= self.player.max_hp // 2:
            self.player.add_power("Strength", 3)
        if self.turn == 2 and self._has_relic("Horn Cleat"):
            self.player.block += 14
        if self.turn == 3 and self._has_relic("Captain's Wheel"):
            self.player.block += 18
        self._apply_happy_flower_start_of_turn()
        self.cards_discarded_this_turn = 0
        self.attack_played_this_turn = 0
        self.skills_played_this_turn = 0
        self.cards_played_this_turn = 0
        self.card_types_played_this_turn = set()
        if self.player.power("Self-Forming Clay Block") > 0:
            self.player.block += self.player.powers.pop("Self-Forming Clay Block")
        for monster in self.monsters:
            if monster.monster_id == "Nemesis" and monster.ai_state.pop("intangible_next", 0):
                monster.add_power("Intangible", 1)
        if self.player.power("Demon Form") > 0:
            self.player.add_power("Strength", self.player.power("Demon Form"))
        if self.player.power("Ritual") > 0:
            self.player.add_power("Strength", self.player.power("Ritual"))
        if self.player.power("Magnetism") > 0:
            for _ in range(self.player.power("Magnetism")):
                self._add_random_card_to_hand(colorless=True)
        if self.player.power("Mayhem") > 0:
            for _ in range(self.player.power("Mayhem")):
                self._play_random_top_card()
        if self.player.power("The Bomb") > 0:
            self.player.add_power("The Bomb", -1)
            if self.player.power("The Bomb") <= 0:
                damage = self.player.powers.pop("The Bomb Damage", 40)
                self._deal_direct_damage_all(damage)
        if self.player.power("Regen") > 0:
            self._heal(self.player.power("Regen"))
            self.player.add_power("Regen", -1)
        if self.player.power("Brutality") > 0:
            self._lose_hp(self.player.power("Brutality"))
            self.draw_cards(self.player.power("Brutality"))
        if self._has_relic("Brimstone"):
            self.player.add_power("Strength", 2)
            for monster in self.monsters:
                if monster.alive:
                    monster.add_power("Strength", 1)
        if self._has_relic("Mercury Hourglass"):
            self._deal_direct_damage_all(3)
        if self.turn == 7 and self._has_relic("Stone Calendar"):
            self._deal_direct_damage_all(52)
        self._check_outcome()
        if self.outcome != "UNDECIDED":
            return
        self.draw_cards(self._card_draw_per_turn())
        if self._has_relic("Warped Tongs"):
            candidates = [card for card in self.hand if card.card_def.card_type not in {"STATUS", "CURSE"} and card.upgrades <= 0]
            if candidates:
                self._upgrade_combat_card(self.card_random_rng.choice(candidates))

    def _move_innate_cards_to_top(self) -> None:
        innate_ids = {"AscendersBane", "Writhe", "Dramatic Entrance", "Mind Blast"}
        innate_cards: list[CardInstance] = []
        for pile in (self.draw_pile, self.discard_pile):
            for card in list(pile):
                if card.card_id in innate_ids:
                    pile.remove(card)
                    innate_cards.append(card)
        self.draw_pile.extend(innate_cards)

    def _move_bottled_cards_to_hand(self) -> None:
        for bottle in self._bottled_card_refs():
            for hand_index, card in enumerate(list(self.hand)):
                if (
                    bottle.get("card_uuid") and card.uuid == bottle.get("card_uuid")
                ) or card.card_id == bottle.get("card_id"):
                    self.hand.pop(hand_index)
                    self.hand.insert(0, card)
                    break

    def _init_opening_draw_pile(self) -> None:
        idxs = list(range(len(self.deck)))
        java_collections_shuffle(idxs, self.shuffle_rng.random_long())
        innate_ids = {"AscendersBane", "Writhe", "Dramatic Entrance", "Mind Blast"}
        bottled_indexes = self._bottled_deck_indexes()
        normal_cards: list[CardInstance] = []
        innate_cards: list[CardInstance] = []
        for deck_idx in idxs:
            card = self.deck[deck_idx]
            if deck_idx in bottled_indexes or card.card_id in innate_ids:
                innate_cards.append(card)
            else:
                normal_cards.append(card)
        self.opening_innate_count = len(innate_cards)
        self.draw_pile = normal_cards + innate_cards

    def _bottled_deck_indexes(self) -> set[int]:
        bottled_uuids = {
            str(relic.get("card_uuid"))
            for relic in self.relics
            if relic.get("relic_id") in {"Bottled Flame", "Bottled Lightning", "Bottled Tornado"} and relic.get("card_uuid")
        }
        if bottled_uuids:
            return {index for index, card in enumerate(self.deck) if card.uuid in bottled_uuids}
        bottled_ids = {
            str(relic.get("card_id"))
            for relic in self.relics
            if relic.get("relic_id") in {"Bottled Flame", "Bottled Lightning", "Bottled Tornado"} and relic.get("card_id")
        }
        if not bottled_ids:
            return set()
        return {index for index, card in enumerate(self.deck) if card.card_id in bottled_ids}

    def _move_bottled_cards_to_top(self) -> None:
        for bottle in self._bottled_card_refs():
            for pile in (self.draw_pile, self.discard_pile):
                for card in list(pile):
                    if (
                        bottle.get("card_uuid") and card.uuid == bottle.get("card_uuid")
                    ) or card.card_id == bottle.get("card_id"):
                        pile.remove(card)
                        self.draw_pile.append(card)
                        break
                else:
                    continue
                break

    def _bottled_card_refs(self) -> list[dict[str, str]]:
        bottled_types = {
            "Bottled Flame": "ATTACK",
            "Bottled Lightning": "SKILL",
            "Bottled Tornado": "POWER",
        }
        card_refs: list[dict[str, str]] = []
        for relic_id, card_type in bottled_types.items():
            relic = self._relic(relic_id)
            if relic is None:
                continue
            card_id = relic.get("card_id")
            card_uuid = relic.get("card_uuid")
            if not card_id:
                for card in self.deck:
                    if card.card_def.card_type == card_type:
                        card_id = card.card_id
                        card_uuid = card.uuid
                        relic["card_id"] = card_id
                        relic["card_uuid"] = card_uuid
                        break
            if not card_id:
                continue
            card_refs.append({
                "card_id": str(card_id),
                "card_uuid": str(card_uuid) if card_uuid else "",
            })
        return card_refs

    def _base_energy(self) -> int:
        extra_energy_relics = {
            "Busted Crown", "Coffee Dripper", "Cursed Key", "Ectoplasm", "Fusion Hammer",
            "Mark of Pain", "Philosopher's Stone", "Runic Dome", "Sozu", "Velvet Choker",
        }
        energy = 3 + sum(1 for relic_id in extra_energy_relics if self._has_relic(relic_id))
        if self._has_relic("Slaver's Collar") and (self.elite or self.floor in {16, 33, 50, 53, 54}):
            energy += 1
        energy += self.player.power("Berserk Energy")
        return energy

    def draw_cards(self, count: int) -> None:
        draw_queue = [count]
        queue_index = 0
        while queue_index < len(draw_queue):
            amount = draw_queue[queue_index]
            queue_index += 1
            if amount <= 0 or self.player.power("No Draw") > 0:
                continue
            drew_any = False
            for _ in range(amount):
                if len(self.hand) >= 10:
                    break
                if not self.draw_pile:
                    if not self.discard_pile:
                        if drew_any:
                            self.shuffle_rng.random_long()
                        break
                    self._draw_triggered_shuffle = True
                    java_collections_shuffle(self.discard_pile, self.shuffle_rng.random_long())
                    self.draw_pile = list(self.discard_pile)
                    self.discard_pile = []
                    if self._advance_relic_counter("Sundial", 3):
                        self.player.energy += 2
                card = self.draw_pile.pop()
                drew_any = True
                if self.player.power("Confusion") > 0 and card.card_def.card_type in {"ATTACK", "SKILL", "POWER"} and not card.card_def.x_cost:
                    card.cost_for_turn = self.card_random_rng.randint(0, 3)
                self.hand.append(card)
                if card.card_id == "Void":
                    self.player.energy = max(0, self.player.energy - 1)
                if card.card_def.card_type == "STATUS":
                    if self.player.power("Evolve") > 0:
                        draw_queue.append(self.player.power("Evolve"))
                    if self.player.power("Fire Breathing") > 0:
                        self._deal_direct_damage_all(self.player.power("Fire Breathing"))
                        self._check_outcome()
                        if self.outcome != "UNDECIDED":
                            return
                elif card.card_def.card_type == "CURSE":
                    if self.player.power("Fire Breathing") > 0:
                        self._deal_direct_damage_all(self.player.power("Fire Breathing"))
                        self._check_outcome()
                        if self.outcome != "UNDECIDED":
                            return

    def playable(self, card: CardInstance, *, free_to_play: bool = False) -> bool:
        if card.card_def.card_type == "STATUS" and card.card_id != "Slimed" and not self._has_relic("Medical Kit"):
            return False
        if card.card_def.card_type == "CURSE" and not self._has_relic("Blue Candle"):
            return False
        if any(other.card_id == "Normality" for other in self.hand) and self.cards_played_this_turn >= 3:
            return False
        if self.player.power("No Attack") > 0 and card.card_def.card_type == "ATTACK":
            return False
        if card.card_id == "Clash" and any(other is not card and other.card_def.card_type != "ATTACK" for other in self.hand):
            return False
        if self._has_relic("Velvet Choker") and self.cards_played_this_turn >= 6:
            return False
        if free_to_play:
            return True
        if card.card_def.x_cost:
            return True
        return self._card_energy_cost(card) <= self.player.energy

    def _card_energy_cost(self, card: CardInstance) -> int:
        if self.player.power("Corruption") > 0 and card.card_def.card_type == "SKILL":
            return 0
        if card.card_id == "Blood for Blood":
            base_cost = 3 if card.upgrades > 0 else 4
            cost = card.cost_for_turn if card.cost_for_turn is not None else base_cost - card.misc
            return max(0, cost)
        cost = card.cost if card.cost_for_turn is None else card.cost_for_turn
        return max(0, cost)

    def _upgrade_combat_card(self, card: CardInstance) -> None:
        if card.upgrades > 0:
            return
        card.upgrades += 1
        if card.card_id == "Blood for Blood":
            card.misc = 0
            card.cost_for_turn = 3

    def _shuffle_cards(self, cards: list[CardInstance]) -> None:
        java_collections_shuffle(cards, self.shuffle_rng.random_long())

    def legal_actions(self) -> list[dict[str, Any]]:
        if self.card_select_context is not None:
            return list(self.card_select_options)
        actions: list[dict[str, Any]] = []
        alive_slots = {id(monster): slot for slot, monster in enumerate(self._alive_monsters())}
        for index, card in enumerate(self.hand):
            if index >= 10:
                continue
            if not self.playable(card):
                continue
            requires_target = card.card_def.has_target and not (card.card_id == "Blind" and card.upgrades > 0)
            if requires_target:
                appended = False
                for target_index, monster in enumerate(self.monsters):
                    if monster.alive:
                        actions.append({
                            "kind": "card",
                            "name": card.name,
                            "card_id": card.card_id,
                            "card_index": index,
                            "source_index": index,
                            "target_index": target_index,
                            "model_target_index": alive_slots.get(id(monster), target_index),
                            "requires_target": True,
                            "bits": index + 1,
                        })
            else:
                actions.append({
                    "kind": "card",
                    "name": card.name,
                    "card_id": card.card_id,
                    "card_index": index,
                    "source_index": index,
                    "target_index": 0,
                    "requires_target": False,
                    "bits": index + 1,
                })
        for index, potion in enumerate(self.potions):
            if not potion.can_use:
                continue
            can_manually_use = potion.potion_id != "FairyPotion"
            if can_manually_use and potion.requires_target:
                for target_index, monster in enumerate(self.monsters):
                    if monster.alive:
                        actions.append({
                            "kind": "potion",
                            "name": potion.name,
                            "potion_id": potion.potion_id,
                            "potion_index": index,
                            "target_index": target_index,
                            "model_target_index": alive_slots.get(id(monster), target_index),
                            "action": "use",
                            "requires_target": True,
                            "bits": 100 + index,
                        })
            elif can_manually_use:
                actions.append({
                    "kind": "potion",
                    "name": potion.name,
                    "potion_id": potion.potion_id,
                    "potion_index": index,
                    "target_index": 0,
                    "action": "use",
                    "requires_target": False,
                    "bits": 100 + index,
                })
            if potion.potion_id == "FairyPotion":
                actions.append({
                    "kind": "potion",
                    "name": potion.name,
                    "potion_id": potion.potion_id,
                    "potion_index": index,
                    "target_index": 8191,
                    "action": "discard",
                    "requires_target": False,
                    "bits": 200 + index,
                })
        actions.append({"kind": "end", "name": "END_TURN", "action_index": 0, "bits": 0})
        return actions

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if action.get("kind") == "card_select":
            self._resolve_card_select(action)
            if self.card_select_context is None:
                self._resolve_pending_monster_block_gains()
            return self.to_spirecomm_state()
        if action.get("kind") == "end":
            self.end_turn()
        elif action.get("kind") == "card":
            self.play_card(int(action.get("card_index", 0)), int(action.get("target_index", 0) or 0))
        elif action.get("kind") == "potion":
            potion_index = int(action.get("potion_index", 0))
            if action.get("action") == "discard":
                self.potions[potion_index] = PotionInstance()
            else:
                self.use_potion(potion_index, int(action.get("target_index", 0) or 0))
        else:
            raise ValueError(f"unsupported native combat action: {action}")
        if self.card_select_context is None:
            self._resolve_pending_monster_block_gains()
        return self.to_spirecomm_state()

    def _open_combat_card_select(
        self,
        context: str,
        selectable_indexes: list[int],
        *,
        pending_card: CardInstance | None = None,
        pending_force_exhaust: bool = False,
        pending_hex_dazed: int = 0,
    ) -> None:
        self.card_select_context = context
        self.pending_resolve_card = pending_card
        self.pending_resolve_force_exhaust = pending_force_exhaust
        self.pending_resolve_hex_dazed = max(0, int(pending_hex_dazed))
        self.card_select_generated_cards = []
        self.card_select_options = []
        for hand_index in selectable_indexes:
            if 0 <= hand_index < len(self.hand):
                card = self.hand[hand_index]
                self.card_select_options.append({
                    "kind": "card_select",
                    "name": context,
                    "select_type": context,
                    "choice_index": hand_index,
                    "select_index": hand_index,
                    "deck_index": hand_index,
                    "card_id": card.card_id,
                    "bits": 2000 + hand_index,
                })

    def _open_generated_combat_card_select(
        self,
        context: str,
        cards: list[CardInstance],
        *,
        pending_card: CardInstance | None = None,
        pending_force_exhaust: bool = False,
        pending_hex_dazed: int = 0,
    ) -> None:
        self.card_select_context = context
        self.pending_resolve_card = pending_card
        self.pending_resolve_force_exhaust = pending_force_exhaust
        self.pending_resolve_hex_dazed = max(0, int(pending_hex_dazed))
        self.card_select_generated_cards = cards
        self.card_select_options = [
            {
                "kind": "card_select",
                "name": context,
                "select_type": context,
                "choice_index": index,
                "select_index": index,
                "deck_index": index,
                "card_id": card.card_id,
                "card": card_to_spirecomm(card),
                "bits": 3000 + index,
            }
            for index, card in enumerate(cards)
        ]

    def _open_confirm_combat_card_select(
        self,
        context: str,
        *,
        pending_card: CardInstance | None = None,
        pending_force_exhaust: bool = False,
        pending_hex_dazed: int = 0,
    ) -> None:
        self.card_select_context = context
        self.pending_resolve_card = pending_card
        self.pending_resolve_force_exhaust = pending_force_exhaust
        self.pending_resolve_hex_dazed = max(0, int(pending_hex_dazed))
        self.card_select_generated_cards = []
        self.card_select_options = [
            {
                "kind": "multi_card_select",
                "name": context,
                "select_type": context,
                "choice_index": 0,
                "select_index": 0,
                "deck_index": 0,
                "bits": 3500,
            }
        ]

    def _open_discard_card_select(
        self,
        context: str,
        selectable_indexes: list[int],
        *,
        pending_card: CardInstance | None = None,
        pending_force_exhaust: bool = False,
        pending_hex_dazed: int = 0,
    ) -> None:
        self.card_select_context = context
        self.pending_resolve_card = pending_card
        self.pending_resolve_force_exhaust = pending_force_exhaust
        self.pending_resolve_hex_dazed = max(0, int(pending_hex_dazed))
        self.card_select_generated_cards = []
        self.card_select_options = []
        for discard_index in selectable_indexes:
            if 0 <= discard_index < len(self.discard_pile):
                card = self.discard_pile[discard_index]
                self.card_select_options.append({
                    "kind": "card_select",
                    "name": context,
                    "select_type": context,
                    "choice_index": discard_index,
                    "select_index": discard_index,
                    "deck_index": discard_index,
                    "card_id": card.card_id,
                    "bits": 4000 + discard_index,
                })

    def _open_exhaust_card_select(
        self,
        context: str,
        selectable_indexes: list[int],
        *,
        pending_card: CardInstance | None = None,
        pending_force_exhaust: bool = False,
        pending_hex_dazed: int = 0,
    ) -> None:
        self.card_select_context = context
        self.pending_resolve_card = pending_card
        self.pending_resolve_force_exhaust = pending_force_exhaust
        self.pending_resolve_hex_dazed = max(0, int(pending_hex_dazed))
        self.card_select_generated_cards = []
        self.card_select_options = []
        for exhaust_index in selectable_indexes:
            if 0 <= exhaust_index < len(self.exhaust_pile):
                card = self.exhaust_pile[exhaust_index]
                self.card_select_options.append({
                    "kind": "card_select",
                    "name": context,
                    "select_type": context,
                    "choice_index": exhaust_index,
                    "select_index": exhaust_index,
                    "deck_index": exhaust_index,
                    "card_id": card.card_id,
                    "bits": 5000 + exhaust_index,
                })

    def _resolve_card_select(self, action: dict[str, Any]) -> None:
        context = self.card_select_context
        hand_index = action.get("select_index")
        if hand_index is None:
            hand_index = action.get("deck_index")
        if hand_index is None:
            hand_index = action.get("choice_index")
        if hand_index is None:
            raise ValueError("card_select action missing index")
        hand_index = int(hand_index)
        if context == "ARMAMENTS":
            if 0 <= hand_index < len(self.hand):
                upgraded_card = self.hand[hand_index]
                self._upgrade_combat_card(upgraded_card)
                valid_cards: list[CardInstance] = []
                invalid_cards: list[CardInstance] = []
                for index, card in enumerate(self.hand):
                    if index == hand_index:
                        continue
                    if _card_can_upgrade(card):
                        valid_cards.append(card)
                    else:
                        invalid_cards.append(card)
                self.hand = valid_cards + [upgraded_card] + invalid_cards
            if self.pending_resolve_card is not None:
                self._resolve_after_use_card_move(
                    self.pending_resolve_card,
                    force_exhaust=self.pending_resolve_force_exhaust,
                    add_hex_dazed=self.pending_resolve_hex_dazed,
                )
        elif context in {"BURNING_PACT", "EXHAUST_ONE"}:
            deferred_dark_embrace_draws: list[int] = []
            if 0 <= hand_index < len(self.hand):
                exhausted_card = self.hand.pop(hand_index)
                self._exhaust_card(exhausted_card, defer_dark_embrace_draws=deferred_dark_embrace_draws)
            draw_count = 2
            if self.pending_resolve_card is not None and self.pending_resolve_card.upgrades > 0:
                draw_count = 3
            self.draw_cards(draw_count)
            if self.pending_resolve_card is not None:
                self._resolve_after_use_card_move(
                    self.pending_resolve_card,
                    force_exhaust=self.pending_resolve_force_exhaust,
                    defer_dark_embrace_draws=deferred_dark_embrace_draws,
                    add_hex_dazed=self.pending_resolve_hex_dazed,
                )
            for deferred_draw_count in deferred_dark_embrace_draws:
                self.draw_cards(deferred_draw_count)
        elif context == "DUAL_WIELD":
            if 0 <= hand_index < len(self.hand):
                selected_card = clone_card(self.hand[hand_index])
                valid_cards: list[CardInstance] = []
                invalid_cards: list[CardInstance] = []
                for index, card in enumerate(self.hand):
                    if index == hand_index:
                        continue
                    if card.card_def.card_type in {"ATTACK", "POWER"}:
                        valid_cards.append(card)
                    else:
                        invalid_cards.append(card)
                self.hand = valid_cards + invalid_cards + [selected_card]
                copy_count = 1
                if self.pending_resolve_card is not None and self.pending_resolve_card.upgrades > 0:
                    copy_count = 2
                for _ in range(copy_count):
                    if len(self.hand) < 10:
                        self.hand.append(clone_card(selected_card))
                    else:
                        self.discard_pile.append(clone_card(selected_card))
            if self.pending_resolve_card is not None:
                self._resolve_after_use_card_move(
                    self.pending_resolve_card,
                    force_exhaust=self.pending_resolve_force_exhaust,
                    add_hex_dazed=self.pending_resolve_hex_dazed,
                )
        elif context == "HEADBUTT":
            if 0 <= hand_index < len(self.discard_pile):
                selected_card = self.discard_pile.pop(hand_index)
                self.draw_pile.append(selected_card)
            pending_card = self.pending_resolve_card
            pending_replays = self.pending_attack_replays
            pending_target_index = self.pending_attack_target_index
            if pending_card is not None:
                self._resolve_after_use_card_move(
                    pending_card,
                    force_exhaust=self.pending_resolve_force_exhaust,
                    add_hex_dazed=self.pending_resolve_hex_dazed,
                )
            if pending_replays > 0 and pending_card is not None:
                target: MonsterState | None = None
                if pending_target_index is not None and 0 <= pending_target_index < len(self.monsters):
                    target = self.monsters[pending_target_index]
                if self._resolve_headbutt_replay(pending_card, target, pending_replays - 1):
                    self.pending_resolve_card = None
                    return
            if self.pending_counter_damage > 0:
                counter_damage = self.pending_counter_damage
                self.pending_counter_damage = 0
                self._take_counter_damage(counter_damage)
        elif context == "WARCRY":
            if 0 <= hand_index < len(self.hand):
                selected_card = self.hand.pop(hand_index)
                self.draw_pile.append(selected_card)
            if self.pending_resolve_card is not None:
                pending = self.pending_resolve_card
                self._resolve_after_use_card_move(
                    pending,
                    force_exhaust=self.pending_resolve_force_exhaust,
                    add_hex_dazed=self.pending_resolve_hex_dazed,
                )
        elif context == "FORETHOUGHT":
            if 0 <= hand_index < len(self.hand):
                selected_card = self.hand.pop(hand_index)
                if selected_card.cost > 0:
                    selected_card.cost_for_turn = 0
                self.draw_pile.insert(0, selected_card)
            if self.pending_resolve_card is not None:
                self._resolve_after_use_card_move(
                    self.pending_resolve_card,
                    force_exhaust=self.pending_resolve_force_exhaust,
                    add_hex_dazed=self.pending_resolve_hex_dazed,
                )
        elif context == "EXHUME":
            if 0 <= hand_index < len(self.exhaust_pile):
                selected_card = self.exhaust_pile.pop(hand_index)
                if len(self.hand) < 10:
                    self.hand.append(selected_card)
                else:
                    self.discard_pile.append(selected_card)
            if self.pending_resolve_card is not None:
                self._resolve_after_use_card_move(
                    self.pending_resolve_card,
                    force_exhaust=True,
                    add_hex_dazed=self.pending_resolve_hex_dazed,
                )
        elif context == "GAMBLE":
            pass
        elif context == "EXHAUST_MANY":
            if self.pending_resolve_card is not None:
                self._resolve_after_use_card_move(
                    self.pending_resolve_card,
                    force_exhaust=self.pending_resolve_force_exhaust,
                    add_hex_dazed=self.pending_resolve_hex_dazed,
                )
        elif context == "DISCOVERY":
            if 0 <= hand_index < len(self.card_select_generated_cards):
                selected_card = clone_card(self.card_select_generated_cards[hand_index])
                selected_card.cost_for_turn = 0
                if len(self.hand) >= 10:
                    self.discard_pile.append(selected_card)
                else:
                    self.hand.append(selected_card)
            if self.pending_resolve_card is not None:
                if self.pending_resolve_card.upgrades > 0:
                    self._resolve_after_use_card_move(
                        self.pending_resolve_card,
                        force_exhaust=self.pending_resolve_force_exhaust,
                        add_hex_dazed=self.pending_resolve_hex_dazed,
                    )
                else:
                    self._resolve_after_use_card_move(
                        self.pending_resolve_card,
                        force_exhaust=True,
                        add_hex_dazed=self.pending_resolve_hex_dazed,
                    )
        else:
            raise ValueError(f"unsupported combat card select context: {context}")
        if self.pending_juggernaut_damage > 0:
            juggernaut_damage = self.pending_juggernaut_damage
            self.pending_juggernaut_damage = 0
            self._trigger_juggernaut(juggernaut_damage)
        self.pending_resolve_card = None
        self.pending_resolve_force_exhaust = False
        self.pending_resolve_hex_dazed = 0
        self.pending_counter_damage = 0
        self.pending_attack_replays = 0
        self.pending_attack_target_index = None
        self.card_select_context = None
        self.card_select_generated_cards = []
        self.card_select_options = []
        self._drain_pending_autoplay_cards()
        self._check_outcome()

    def _new_uuid(self, prefix: str) -> str:
        self._next_uuid += 1
        return f"{prefix}-{self.seed}-{self.turn}-{self._next_uuid}"

    def _add_to_discard(self, card_id: str, upgrades: int = 0) -> None:
        self.discard_pile.append(make_card(card_id, upgrades=upgrades, uuid=self._new_uuid(card_id)))

    def _add_to_hand(
        self,
        card_id: str,
        upgrades: int = 0,
        cost_for_turn: int | None = None,
        cost_for_combat: int | None = None,
    ) -> None:
        card = make_card(card_id, upgrades=upgrades, uuid=self._new_uuid(card_id))
        card.cost_for_combat = cost_for_combat
        card.cost_for_turn = cost_for_turn
        if len(self.hand) >= 10:
            self.discard_pile.append(card)
        else:
            self.hand.append(card)

    def _move_card_to_discard(self, card: CardInstance) -> None:
        card.cost_for_turn = None
        self.discard_pile.append(card)

    def _card_exhausts_on_use(self, card: CardInstance) -> bool:
        return bool(card.card_def.exhausts)

    def _resolve_after_use_card_move(
        self,
        card: CardInstance,
        *,
        force_exhaust: bool = False,
        defer_dark_embrace_draws: list[int] | None = None,
        add_hex_dazed: int = 0,
    ) -> None:
        for _ in range(max(0, int(add_hex_dazed))):
            self._insert_temp_card_into_draw_pile("Dazed")
        if force_exhaust or self._card_exhausts_on_use(card) or (self.player.power("Corruption") > 0 and card.card_def.card_type == "SKILL"):
            self._exhaust_card(card, defer_dark_embrace_draws=defer_dark_embrace_draws)
        elif card.card_def.card_type != "POWER":
            self._move_card_to_discard(card)

    def _resolve_pending_monster_block_gains(self) -> None:
        if not self.pending_monster_block_gains:
            return
        pending = self.pending_monster_block_gains
        self.pending_monster_block_gains = []
        for monster, amount in pending:
            if amount > 0:
                monster.block += amount

    def _alive_monsters(self) -> list[MonsterState]:
        return [monster for monster in self.monsters if monster.alive]

    def _random_alive_monster(self) -> MonsterState | None:
        alive = self._alive_monsters()
        if not alive:
            return None
        return alive[int(self.card_random_rng.random(len(alive) - 1))]

    def _deal_attack_damage(
        self,
        base: int,
        monster: MonsterState | None,
        hits: int = 1,
        *,
        strength_multiplier: int = 1,
        defer_counter_damage: bool = False,
    ) -> int:
        if monster is None:
            return 0
        total = 0
        curl_up_amount = monster.power("Curl Up")
        for _ in range(max(1, hits)):
            damage = float(base + self.player.power("Strength") * max(1, strength_multiplier))
            if self.player.power("Akabeko") > 0:
                damage += self.player.power("Akabeko")
                self.player.powers.pop("Akabeko", None)
            if self.player.power("Weakened") > 0:
                damage *= 0.75
            if monster.power("Vulnerable") > 0:
                damage *= 1.75 if self._has_relic("Paper Frog") else 1.5
            if monster.power("Slow") > 0:
                damage *= 1.0 + 0.1 * monster.power("Slow")
            if monster.power("Flight") > 0:
                damage *= 0.5
                monster.add_power("Flight", -1)
                if monster.monster_id == "Byrd" and monster.power("Flight") <= 0:
                    monster.powers["Flight"] = 0
                    monster.move = "BYRD_STUNNED"
                    monster.intent = "STUN"
                    monster.move_base_damage = 0
                    monster.move_hits = 0
            if self.player.power("Pen Nib") > 0:
                damage *= 2
                self.player.add_power("Pen Nib", -1)
            if self._has_relic("The Boot") and 0 < damage < 5:
                damage = 5
            was_alive = monster.alive
            if was_alive and monster.power("Angry") > 0:
                monster.add_power("Strength", monster.power("Angry"))
            block_before = monster.block
            dealt = self._apply_damage_to_monster(max(0, math.floor(damage)), monster)
            total += dealt
            if self._has_relic("Hand Drill") and block_before > 0 and monster.block <= 0 and monster.alive:
                self._apply_monster_power(monster, "Vulnerable", 2)
            self._apply_guardian_mode_shift(monster, dealt)
            if monster.monster_id == "Lagavulin" and monster.ai_state.get("asleep", 0) and dealt > 0:
                monster.ai_state["asleep"] = 0
                monster.ai_state["latent_awake"] = 0
                monster.ai_state["awoken"] = 1
                if monster.power("Metallicize") > 0:
                    monster.add_power("Metallicize", -8)
            if dealt > 0 and monster.alive and monster.power("Malleable") > 0:
                malleable_block = monster.power("Malleable")
                monster.block += malleable_block
                monster.add_power("Malleable", 1)
            counter_damage = monster.power("Thorns") + monster.power("Sharp Hide")
            if counter_damage > 0:
                if defer_counter_damage:
                    self.pending_counter_damage += counter_damage
                else:
                    self._take_counter_damage(counter_damage)
            if was_alive and not monster.alive:
                self._on_monster_defeated(monster)
            elif was_alive:
                self._maybe_split_slime_boss(monster)
            if not monster.alive:
                break
        if total > 0 and curl_up_amount > 0:
            self.pending_monster_block_gains.append((monster, curl_up_amount))
            monster.powers.pop("Curl Up", None)
        return total

    def _take_counter_damage(self, amount: int) -> None:
        amount = max(0, int(amount))
        if amount <= 0:
            return
        blocked = min(self.player.block, amount)
        self.player.block -= blocked
        amount -= blocked
        if amount <= 0:
            return
        if self.player.power("Intangible") > 0:
            amount = min(amount, 1)
        if self.player.power("Buffer") > 0:
            self.player.add_power("Buffer", -1)
            return
        if self._has_relic("Tungsten Rod"):
            amount = max(0, amount - 1)
            if amount <= 0:
                return
        self.player.current_hp = max(0, self.player.current_hp - amount)
        if amount > 0 and (relic := self._relic("Centennial Puzzle")) is not None and int(relic.get("counter", 0)) == 0:
            relic["counter"] = 1
            self.draw_cards(3)
        if amount > 0 and self._has_relic("Runic Cube"):
            self.draw_cards(1)
        if self.player.current_hp <= 0 and (relic := self._relic("Lizard Tail")) is not None and int(relic.get("counter", -1)) != 0:
            relic["counter"] = 0
            self.player.current_hp = max(1, self.player.max_hp // 2)
        if self.player.current_hp <= 0:
            for index, potion in enumerate(self.potions):
                if potion.potion_id == "FairyPotion":
                    self.potions[index] = PotionInstance()
                    self.player.current_hp = max(1, int(self.player.max_hp * 0.3))
                    break
        self.hp_lost_this_combat += amount
        self.hp_loss_events_this_combat += 1
        self._on_player_took_damage_cards()
        if self._has_relic("Self-Forming Clay"):
            self.player.add_power("Self-Forming Clay Block", 3)

    def _deal_damage_all(self, base: int, hits: int = 1) -> int:
        return sum(self._deal_attack_damage(base, monster, hits=hits) for monster in list(self._alive_monsters()))

    def _deal_direct_damage_to_monster(self, amount: int, monster: MonsterState | None) -> int:
        if monster is None:
            return 0
        was_alive = monster.alive
        amount = max(0, int(amount))
        if amount <= 0:
            return 0
        if monster.monster_id == "CorruptHeart":
            cap = 300
            already = self.monster_damage_this_turn.get(id(monster), 0)
            remaining = max(0, cap - already)
            blocked = min(monster.block, amount)
            hp_damage = max(0, amount - blocked)
            amount = blocked + min(hp_damage, remaining)
        if monster.power("Intangible") > 0 and amount > 0:
            blocked = min(monster.block, amount)
            hp_damage = max(0, amount - blocked)
            amount = blocked + (1 if hp_damage > 0 else 0)
        dealt = _apply_damage(amount, monster)
        self._apply_guardian_mode_shift(monster, dealt)
        if amount > 0 and dealt > 0 and monster.monster_id == "Lagavulin" and monster.ai_state.get("asleep", 0):
            monster.ai_state["asleep"] = 0
            monster.ai_state["latent_awake"] = 0
            if monster.power("Metallicize") > 0:
                monster.add_power("Metallicize", -8)
                if monster.power("Metallicize") <= 0:
                    monster.powers.pop("Metallicize", None)
            monster.block = 0
        if was_alive and not monster.alive:
            self._on_monster_defeated(monster)
        elif was_alive:
            self._maybe_split_slime_boss(monster)
        return dealt

    def _deal_direct_damage_all(self, amount: int) -> int:
        return sum(self._deal_direct_damage_to_monster(amount, monster) for monster in list(self._alive_monsters()))

    def _deal_combust_damage_all(self, amount: int) -> int:
        total = 0
        for monster in list(self._alive_monsters()):
            if monster.monster_id == "Lagavulin" and monster.ai_state.get("asleep", 0):
                continue
            total += self._deal_direct_damage_to_monster(amount, monster)
        return total

    def _apply_damage_to_monster(self, amount: int, monster: MonsterState) -> int:
        amount = max(0, int(amount))
        if amount <= 0:
            return 0
        if monster.monster_id == "CorruptHeart":
            cap = 300
            already = self.monster_damage_this_turn.get(id(monster), 0)
            remaining = max(0, cap - already)
            blocked = min(monster.block, amount)
            hp_damage = max(0, amount - blocked)
            amount = blocked + min(hp_damage, remaining)
        if monster.power("Intangible") > 0:
            blocked = min(monster.block, amount)
            hp_damage = max(0, amount - blocked)
            amount = blocked + (1 if hp_damage > 0 else 0)
        dealt = _apply_damage(amount, monster)
        if dealt > 0 and monster.power("Plated Armor") > 0:
            monster.add_power("Plated Armor", -1)
            if monster.power("Plated Armor") <= 0:
                monster.powers.pop("Plated Armor", None)
                if monster.monster_id == "ShelledParasite" and monster.alive:
                    monster.move = "SHELLED_STUNNED"
                    monster.intent = "STUN"
                    monster.move_base_damage = 0
                    monster.move_hits = 0
        if monster.monster_id == "CorruptHeart" and dealt > 0:
            self.monster_damage_this_turn[id(monster)] = self.monster_damage_this_turn.get(id(monster), 0) + dealt
        return dealt

    def _apply_guardian_mode_shift(self, monster: MonsterState, dealt: int) -> None:
        if monster.monster_id != "TheGuardian" or dealt <= 0 or monster.power("Mode Shift") <= 0:
            return
        monster.add_power("Mode Shift", -dealt)
        if monster.power("Mode Shift") > 0:
            return
        monster.powers.pop("Mode Shift", None)
        monster.move = "THE_GUARDIAN_DEFENSIVE_MODE"
        monster.intent = "BUFF"
        monster.move_base_damage = 0
        monster.move_hits = 0
        monster.block += 20

    def _trigger_juggernaut(self, amount: int) -> None:
        if amount <= 0:
            return
        target = self._random_alive_monster()
        if target is None:
            return
        was_alive = target.alive
        self._apply_damage_to_monster(amount, target)
        if was_alive and not target.alive:
            self._on_monster_defeated(target)
        elif was_alive:
            self._maybe_split_slime_boss(target)

    def _gain_block(self, amount: int, *, defer_juggernaut: bool = False, apply_block_modifiers: bool = True) -> None:
        gained = _player_block_amount(amount, self.player) if apply_block_modifiers else max(0, int(amount))
        if gained <= 0:
            return
        self.player.block += gained
        if self.player.power("Juggernaut") > 0:
            if defer_juggernaut:
                self.pending_juggernaut_damage += self.player.power("Juggernaut")
            else:
                self._trigger_juggernaut(self.player.power("Juggernaut"))

    def _lose_hp(self, amount: int, *, from_attack: bool = False) -> None:
        amount = max(0, int(amount))
        if amount <= 0:
            return
        if self.player.power("Intangible") > 0:
            amount = min(amount, 1)
        if from_attack and self.player.power("Buffer") > 0:
            self.player.add_power("Buffer", -1)
            return
        if self._has_relic("Torii") and 1 < amount <= 5:
            amount = 1
        if self._has_relic("Tungsten Rod"):
            amount = max(0, amount - 1)
            if amount <= 0:
                return
        self.player.current_hp = max(0, self.player.current_hp - amount)
        if amount > 0 and (relic := self._relic("Centennial Puzzle")) is not None and int(relic.get("counter", 0)) == 0:
            relic["counter"] = 1
            self.draw_cards(3)
        if amount > 0 and self._has_relic("Runic Cube"):
            self.draw_cards(1)
        if self.player.current_hp <= 0 and (relic := self._relic("Lizard Tail")) is not None and int(relic.get("counter", -1)) != 0:
            relic["counter"] = 0
            self.player.current_hp = max(1, self.player.max_hp // 2)
        if self.player.current_hp <= 0:
            for index, potion in enumerate(self.potions):
                if potion.potion_id == "FairyPotion":
                    self.potions[index] = PotionInstance()
                    self.player.current_hp = max(1, int(self.player.max_hp * 0.3))
                    break
        self.hp_lost_this_combat += amount
        self.hp_loss_events_this_combat += 1
        self._on_player_took_damage_cards()
        if self.player.power("Rupture") > 0 and not from_attack:
            self.player.add_power("Strength", self.player.power("Rupture"))
        if self._has_relic("Self-Forming Clay"):
            self.player.add_power("Self-Forming Clay Block", 3)

    def _on_player_took_damage_cards(self) -> None:
        for pile in (self.hand, self.draw_pile, self.discard_pile):
            for card in pile:
                if card.card_id != "Blood for Blood":
                    continue
                card.misc += 1
                if card.cost_for_turn is not None:
                    card.cost_for_turn = max(0, card.cost_for_turn - 1)

    def _heal(self, amount: int) -> None:
        amount = max(0, int(amount))
        if amount > 0 and self._has_relic("Magic Flower"):
            amount = int(amount * 1.5)
        self.player.current_hp = min(self.player.max_hp, self.player.current_hp + amount)

    def _on_monster_defeated(self, monster: MonsterState) -> None:
        if monster.monster_id == "Darkling":
            monster.current_hp = 0
            monster.is_gone = False
            monster.half_dead = True
            return
        if monster.power("Spore Cloud") > 0:
            self._apply_player_power("Vulnerable", 2, just_applied=False)
        if monster.monster_id == "BronzeOrb":
            stasis_card = monster.ai_state.pop("stasis_card", None)
            if isinstance(stasis_card, CardInstance):
                if len(self.hand) < 10:
                    self.hand.append(stasis_card)
                else:
                    self.discard_pile.append(stasis_card)
        if monster.monster_id in {"Looter", "Mugger"}:
            stolen_gold = int(monster.ai_state.pop("stolen_gold", 0) or 0)
            if stolen_gold > 0:
                self.reward_gold_bonus += stolen_gold
        if monster.monster_id == "AwakenedOne" and monster.power("Awakened Reborn") <= 0:
            monster.current_hp = 300
            monster.max_hp = max(monster.max_hp, 300)
            monster.is_gone = False
            monster.half_dead = False
            monster.powers = {"Awakened Reborn": 1}
            choose_next_move(monster, self.ai_rng)
            return
        self._on_monster_killed()

    def _on_monster_killed(self) -> None:
        if self._has_relic("Gremlin Horn"):
            self.player.energy += 1
            self.draw_cards(1)

    def _maybe_split_slime_boss(self, monster: MonsterState) -> None:
        if monster.current_hp <= 0 or monster.current_hp > monster.max_hp // 2:
            return
        if monster.monster_id == "AcidSlime_L":
            monster.move = "ACID_SLIME_L_SPLIT"
            monster.intent = "UNKNOWN"
            monster.move_base_damage = 0
            monster.move_hits = 0
            return
        if monster.monster_id == "SpikeSlime_L":
            monster.move = "SPIKE_SLIME_L_SPLIT"
            monster.intent = "UNKNOWN"
            monster.move_base_damage = 0
            monster.move_hits = 0
            return
        if monster.monster_id != "SlimeBoss":
            return
        monster.move = "SLIME_BOSS_SPLIT"
        monster.intent = "MAGIC"
        monster.move_base_damage = 0
        monster.move_hits = 0

    def _spawn_split_child(self, monster_id: str, hp: int) -> MonsterState:
        if monster_id == "AcidSlime_M":
            slime = MonsterState(
                "AcidSlime_M",
                "Acid Slime (M)",
                hp,
                hp,
                "ACID_SLIME_M_TACKLE",
                "ATTACK",
                move_base_damage=12 if self.ascension_level >= 2 else 10,
                move_hits=1,
            )
        elif monster_id == "SpikeSlime_M":
            slime = MonsterState(
                "SpikeSlime_M",
                "Spike Slime (M)",
                hp,
                hp,
                "SPIKE_SLIME_M_FLAME_TACKLE",
                "ATTACK_DEBUFF",
                move_base_damage=10 if self.ascension_level >= 2 else 8,
                move_hits=1,
            )
        elif monster_id == "AcidSlime_L":
            slime = MonsterState(
                "AcidSlime_L",
                "Acid Slime (L)",
                hp,
                hp,
                "ACID_SLIME_L_CORROSIVE_SPIT",
                "ATTACK_DEBUFF",
                move_base_damage=12 if self.ascension_level >= 2 else 11,
                move_hits=1,
            )
        elif monster_id == "SpikeSlime_L":
            slime = MonsterState(
                "SpikeSlime_L",
                "Spike Slime (L)",
                hp,
                hp,
                "SPIKE_SLIME_L_FLAME_TACKLE",
                "ATTACK_DEBUFF",
                move_base_damage=12 if self.ascension_level >= 2 else 10,
                move_hits=1,
            )
        else:
            slime = make_monster(monster_id, self.monster_hp_rng, self.ascension_level)
            slime.max_hp = hp
            slime.current_hp = hp
        slime.ai_state["ascension_level"] = self.ascension_level
        choose_next_move(slime, self.ai_rng)
        return slime

    def _summon_monster(self, monster_id: str, *, max_alive: int = 5) -> bool:
        if len(self._alive_monsters()) >= max_alive:
            return False
        summoned = make_monster(monster_id, self.monster_hp_rng, self.ascension_level)
        choose_next_move(summoned, self.rng)
        self.monsters.append(summoned)
        return True

    def _is_debuff_power(self, power_id: str, amount: int) -> bool:
        return power_id in {
            "Vulnerable",
            "Weakened",
            "Frail",
            "Shackled",
            "Dexterity Down",
            "Flex Strength Down",
            "Confusion",
            "Hex",
            "No Draw",
            "No Block",
            "No Attack",
        } or (power_id in {"Strength", "Dexterity"} and amount < 0)

    def _apply_monster_power(self, monster: MonsterState | None, power_id: str, amount: int) -> bool:
        if monster is None or amount == 0:
            return False
        is_debuff = self._is_debuff_power(power_id, amount)
        if is_debuff and monster.power("Artifact") > 0:
            monster.add_power("Artifact", -1)
            return False
        monster.add_power(power_id, amount)
        if power_id == "Vulnerable" and amount > 0 and self._has_relic("Champion Belt"):
            self._apply_monster_power(monster, "Weakened", 1)
        if is_debuff and self.player.power("Sadistic Nature") > 0 and monster.alive:
            was_alive = monster.alive
            self._apply_damage_to_monster(self.player.power("Sadistic Nature"), monster)
            if was_alive and not monster.alive:
                self._on_monster_defeated(monster)
            elif was_alive:
                self._maybe_split_slime_boss(monster)
        return True

    def _apply_player_power(self, power_id: str, amount: int, *, just_applied: bool | None = None) -> None:
        if amount == 0:
            return
        if power_id == "Weakened" and self._has_relic("Ginger"):
            return
        if power_id == "Frail" and self._has_relic("Turnip"):
            return
        is_debuff = self._is_debuff_power(power_id, amount)
        existing_amount = self.player.power(power_id)
        if is_debuff and self.player.power("Artifact") > 0:
            self.player.add_power("Artifact", -1)
            return
        self.player.add_power(power_id, amount)
        if is_debuff and amount > 0:
            if just_applied is None:
                if power_id in {"Weakened", "Frail", "Vulnerable", "No Draw"}:
                    just_applied = existing_amount <= 0
                else:
                    just_applied = True
            if just_applied:
                self.player_powers_just_applied.add(power_id)

    def _random_card_id(self, *, card_type: str | None = None, colorless: bool = False) -> str:
        if colorless:
            candidates = [
                card_id for card_id in COMBAT_COLORLESS_CARD_POOL
                if card_type is None or CARD_LIBRARY[card_id].card_type == card_type
            ]
        elif card_type == "ATTACK":
            candidates = list(COMBAT_ATTACK_CARD_POOL_IRONCLAD)
        elif card_type == "SKILL":
            candidates = list(COMBAT_SKILL_CARD_POOL_IRONCLAD)
        elif card_type == "POWER":
            candidates = list(COMBAT_POWER_CARD_POOL_IRONCLAD)
        else:
            candidates = list(COMBAT_CARD_POOL_IRONCLAD)
        candidates = [card_id for card_id in candidates if card_id not in self.locked_card_ids]
        return candidates[int(self.card_random_rng.random(len(candidates) - 1))]

    def _add_random_card_to_hand(self, *, card_type: str | None = None, colorless: bool = False, cost_for_turn: int | None = None) -> None:
        self._add_to_hand(self._random_card_id(card_type=card_type, colorless=colorless), cost_for_turn=cost_for_turn)

    def _random_combat_card_id(self, *, card_type: str | None = None) -> str:
        if card_type == "ATTACK":
            candidates = COMBAT_ATTACK_CARD_POOL_IRONCLAD
        elif card_type == "SKILL":
            candidates = COMBAT_SKILL_CARD_POOL_IRONCLAD
        elif card_type == "POWER":
            candidates = COMBAT_POWER_CARD_POOL_IRONCLAD
        else:
            candidates = COMBAT_CARD_POOL_IRONCLAD
        return candidates[int(self.card_random_rng.random(len(candidates) - 1))]

    def _put_random_cards_in_draw_pile(self, *, card_type: str, count: int, cost_for_turn: int = 0) -> None:
        card_ids = [self._random_combat_card_id(card_type=card_type) for _ in range(count)]
        for card_id in card_ids:
            card = make_card(card_id, uuid=self._new_uuid(card_id))
            card.cost_for_combat = cost_for_turn
            card.cost_for_turn = cost_for_turn
            insert_index = 0 if not self.draw_pile else int(self.card_random_rng.random(len(self.draw_pile) - 1))
            self.draw_pile.insert(insert_index, card)

    def _insert_temp_card_into_draw_pile(self, card_id: str, *, upgrades: int = 0) -> None:
        card = make_card(card_id, upgrades=upgrades, uuid=self._new_uuid(card_id))
        insert_index = 0 if not self.draw_pile else int(self.card_random_rng.random(len(self.draw_pile) - 1))
        self.draw_pile.insert(insert_index, card)

    def _discovery_card_options(self, *, card_type: str | None = None, colorless: bool = False) -> list[CardInstance]:
        options: list[CardInstance] = []
        seen: set[str] = set()
        if colorless:
            candidates = [
                card_id for card_id in COMBAT_COLORLESS_CARD_POOL
                if card_type is None or CARD_LIBRARY[card_id].card_type == card_type
            ]
        elif card_type == "ATTACK":
            candidates = list(COMBAT_ATTACK_CARD_POOL_IRONCLAD)
        elif card_type == "SKILL":
            candidates = list(COMBAT_SKILL_CARD_POOL_IRONCLAD)
        elif card_type == "POWER":
            candidates = list(COMBAT_POWER_CARD_POOL_IRONCLAD)
        else:
            candidates = list(COMBAT_CARD_POOL_IRONCLAD)
        candidates = [card_id for card_id in candidates if card_id not in self.locked_card_ids]
        while len(options) < 3 and candidates:
            card_id = candidates[int(self.card_random_rng.random(len(candidates) - 1))]
            if card_id in seen:
                continue
            seen.add(card_id)
            options.append(make_card(card_id, uuid=self._new_uuid(f"Discovery-{card_id}")))
        return options

    def _fetch_from_draw_to_hand(self, card_type: str) -> None:
        for card in list(self.draw_pile):
            if card.card_def.card_type == card_type:
                self.draw_pile.remove(card)
                if len(self.hand) >= 10:
                    self.discard_pile.append(card)
                else:
                    self.hand.append(card)
                return

    def _exhaust_card(self, card: CardInstance, *, defer_dark_embrace_draws: list[int] | None = None) -> None:
        if self._has_relic("Strange Spoon") and card.card_def.card_type in {"ATTACK", "SKILL", "POWER"} and self.rng.random() < 0.5:
            self.discard_pile.append(card)
            return
        self.exhaust_pile.append(card)
        if card.card_id == "Sentinel":
            self.player.energy += 3 if card.upgrades else 2
        if self._has_relic("Charon's Ashes"):
            self._deal_direct_damage_all(3)
        if self.player.power("Feel No Pain") > 0:
            self._gain_block(self.player.power("Feel No Pain"), apply_block_modifiers=False)
        if self.player.power("Dark Embrace") > 0:
            if defer_dark_embrace_draws is not None:
                defer_dark_embrace_draws.append(self.player.power("Dark Embrace"))
            else:
                self.draw_cards(self.player.power("Dark Embrace"))
        if self._has_relic("Dead Branch"):
            candidates = [card_def.card_id for card_def in self._ironclad_card_pool()]
            self._add_to_hand(self.rng.choice(candidates))
        if card.card_id == "Necronomicurse":
            if card in self.exhaust_pile:
                self.exhaust_pile.remove(card)
            if len(self.hand) < 10:
                self.hand.append(card)
            else:
                self.discard_pile.append(card)

    def _exhaust_non_attacks_from_hand(self) -> int:
        exhausted = 0
        for hand_index in range(len(self.hand) - 1, -1, -1):
            other = self.hand[hand_index]
            if other.card_def.card_type != "ATTACK":
                self.hand.pop(hand_index)
                self._exhaust_card(other)
                exhausted += 1
        return exhausted

    def _top_discard_card(self) -> CardInstance | None:
        return self.discard_pile[-1] if self.discard_pile else None

    def _top_card_from_draw_pile(self) -> CardInstance | None:
        if not self.draw_pile:
            if not self.discard_pile:
                return None
            self.draw_pile = list(self.discard_pile)
            self.discard_pile = []
            self._shuffle_cards(self.draw_pile)
        return self.draw_pile.pop()

    def _drain_pending_autoplay_cards(self) -> None:
        if self._processing_autoplay_cards:
            return
        self._processing_autoplay_cards = True
        try:
            while self.pending_autoplay_cards and self.card_select_context is None and self.outcome == "UNDECIDED":
                top, target_index, force_exhaust, energy_on_use = self.pending_autoplay_cards.pop(0)
                self.hand.append(top)
                index = len(self.hand) - 1
                if not self.playable(top, free_to_play=True):
                    self.hand.pop(index)
                    continue
                self.play_card(
                    index,
                    target_index,
                    force_exhaust=force_exhaust,
                    free_to_play=True,
                    energy_on_use=energy_on_use,
                )
        finally:
            self._processing_autoplay_cards = False

    def _play_random_top_card(self) -> None:
        top = self._top_card_from_draw_pile()
        if top is None:
            return
        target = self._random_alive_monster()
        energy_on_use = self.player.energy if top.card_def.x_cost else None
        self.pending_autoplay_cards.append(
            (
                top,
                self.monsters.index(target) if target in self.monsters else 0,
                top.card_def.card_type != "POWER",
                energy_on_use,
            )
        )

    def _replay_attack_card_effect(self, card: CardInstance, target: MonsterState | None, energy_spent: int) -> None:
        card_id = card.card_id
        if card_id == "Strike_R":
            base = 9 if card.upgrades else 6
            if self._has_relic("Strike Dummy"):
                base += 3
            self._deal_attack_damage(base, target)
        elif card_id == "Bash":
            self._deal_attack_damage(10 if card.upgrades else 8, target)
            self._apply_monster_power(target, "Vulnerable", 3 if card.upgrades else 2)
        elif card_id == "Anger":
            self._deal_attack_damage(8 if card.upgrades else 6, target)
            self.discard_pile.append(make_card("Anger", upgrades=card.upgrades, uuid=self._new_uuid("Anger")))
        elif card_id == "Body Slam":
            self._deal_attack_damage(self.player.block, target)
        elif card_id == "Clash":
            self._deal_attack_damage(18 if card.upgrades else 14, target)
        elif card_id == "Cleave":
            self._deal_damage_all(11 if card.upgrades else 8)
        elif card_id == "Clothesline":
            self._deal_attack_damage(14 if card.upgrades else 12, target)
            self._apply_monster_power(target, "Weakened", 3 if card.upgrades else 2)
        elif card_id == "Headbutt":
            self._deal_attack_damage(12 if card.upgrades else 9, target)
            if self.discard_pile:
                self.draw_pile.append(self.discard_pile.pop())
        elif card_id == "Heavy Blade":
            self._deal_attack_damage(14, target, strength_multiplier=5 if card.upgrades else 3)
        elif card_id == "Iron Wave":
            self.pending_counter_damage = 0
            self._deal_attack_damage(7 if card.upgrades else 5, target, defer_counter_damage=True)
            counter_damage = self.pending_counter_damage
            self.pending_counter_damage = 0
            self._gain_block(_player_block_amount(7 if card.upgrades else 5, self.player))
            if counter_damage > 0:
                self._take_counter_damage(counter_damage)
        elif card_id == "Perfected Strike":
            strike_count = _combat_strike_count(self, card)
            self._deal_attack_damage((6 if card.upgrades else 6) + strike_count * (3 if card.upgrades else 2), target)
        elif card_id == "Pommel Strike":
            self._deal_attack_damage(10 if card.upgrades else 9, target)
            self.draw_cards(2 if card.upgrades else 1)
        elif card_id == "Sword Boomerang":
            for _ in range(4 if card.upgrades else 3):
                self._deal_attack_damage(3, self._random_alive_monster())
        elif card_id == "Thunderclap":
            for monster in self.monsters:
                if monster.alive:
                    self._deal_attack_damage(7 if card.upgrades else 4, monster)
                    self._apply_monster_power(monster, "Vulnerable", 1)
        elif card_id == "Twin Strike":
            base = 7 if card.upgrades else 5
            if self._has_relic("Strike Dummy"):
                base += 3
            self._deal_attack_damage(base, target, hits=2)
        elif card_id == "Wild Strike":
            self._deal_attack_damage(17 if card.upgrades else 12, target)
            self._insert_temp_card_into_draw_pile("Wound")
        elif card_id == "Blood for Blood":
            self._deal_attack_damage(22 if card.upgrades else 18, target)
        elif card_id == "Carnage":
            self._deal_attack_damage(28 if card.upgrades else 20, target)
        elif card_id == "Dropkick":
            self._deal_attack_damage(8 if card.upgrades else 5, target)
            if target and target.power("Vulnerable") > 0:
                self.player.energy += 1
                self.draw_cards(1)
        elif card_id == "Hemokinesis":
            self._lose_hp(2)
            self._deal_attack_damage(20 if card.upgrades else 15, target)
        elif card_id == "Pummel":
            self._deal_attack_damage(2, target, hits=5 if card.upgrades else 4)
        elif card_id == "Rampage":
            self._deal_attack_damage(8 + card.misc, target)
            card.misc += 8 if card.upgrades else 5
        elif card_id == "Reckless Charge":
            self._deal_attack_damage(10 if card.upgrades else 7, target)
            self._insert_temp_card_into_draw_pile("Dazed")
        elif card_id == "Searing Blow":
            self._deal_attack_damage(12 + card.upgrades * 4, target)
        elif card_id == "Sever Soul":
            self._exhaust_non_attacks_from_hand()
            self._deal_attack_damage(22 if card.upgrades else 16, target)
        elif card_id == "Uppercut":
            self._deal_attack_damage(13 if card.upgrades else 10, target)
            self._apply_monster_power(target, "Vulnerable", 2 if card.upgrades else 1)
            self._apply_monster_power(target, "Weakened", 2 if card.upgrades else 1)
        elif card_id == "Whirlwind":
            x_amount = energy_spent + (2 if self._has_relic("Chemical X") else 0)
            for _ in range(max(0, x_amount)):
                self._deal_damage_all(8 if card.upgrades else 5)
        elif card_id == "Bludgeon":
            self._deal_attack_damage(42 if card.upgrades else 32, target)
        elif card_id == "Feed":
            before_alive = target.alive if target else False
            self._deal_attack_damage(12 if card.upgrades else 10, target)
            if before_alive and target and not target.alive:
                self.player.max_hp += 4 if card.upgrades else 3
                self.player.current_hp += 4 if card.upgrades else 3
        elif card_id == "Fiend Fire":
            count = len(self.hand)
            while self.hand:
                exhausted = self.hand[self.card_random_rng.random(len(self.hand) - 1)]
                self.hand.remove(exhausted)
                self._exhaust_card(exhausted)
            self._deal_attack_damage(10 if card.upgrades else 7, target, hits=count)
        elif card_id == "Immolate":
            self._deal_damage_all(28 if card.upgrades else 21)
            self.discard_pile.append(make_card("Burn", uuid=self._new_uuid("Burn")))
        elif card_id == "Reaper":
            healed = self._deal_damage_all(5 if card.upgrades else 4)
            self.player.current_hp = min(self.player.max_hp, self.player.current_hp + healed)
        elif card_id == "Dramatic Entrance":
            self._deal_damage_all(12 if card.upgrades else 8)
        elif card_id == "Flash of Steel":
            self._deal_attack_damage(6 if card.upgrades else 3, target)
            self.draw_cards(1)
        elif card_id == "Mind Blast":
            self._deal_attack_damage(len(self.deck), target)
        elif card_id == "Swift Strike":
            self._deal_attack_damage(10 if card.upgrades else 7, target)
        elif card_id in {"Hand of Greed", "HandOfGreed"}:
            before_alive = target.alive if target else False
            self._deal_attack_damage(25 if card.upgrades else 20, target)
            if before_alive and target and not target.alive:
                self.gold += 25 if card.upgrades else 20
        elif card_id == "Bite":
            self._deal_attack_damage(8 if card.upgrades else 7, target)
            self._heal(2)
        elif card_id == "Ritual Dagger":
            before_alive = target.alive if target else False
            self._deal_attack_damage(15 if card.upgrades else 15, target)
            if before_alive and target and not target.alive:
                card.misc += 5 if card.upgrades else 3

    def _resolve_headbutt_replay(self, card: CardInstance, target: MonsterState | None, remaining_replays: int) -> bool:
        self.pending_counter_damage = 0
        self._deal_attack_damage(12 if card.upgrades else 9, target, defer_counter_damage=True)
        counter_damage = self.pending_counter_damage
        self.pending_counter_damage = 0
        self._check_outcome()
        if self.outcome != "UNDECIDED":
            if counter_damage > 0:
                self._take_counter_damage(counter_damage)
            return False
        if len(self.discard_pile) == 1:
            self.draw_pile.append(self.discard_pile.pop(0))
            if counter_damage > 0:
                self._take_counter_damage(counter_damage)
            return False
        if len(self.discard_pile) > 1:
            self.pending_counter_damage = counter_damage
            self.pending_attack_replays = remaining_replays
            self.pending_attack_target_index = self.monsters.index(target) if target in self.monsters else None
            self._open_discard_card_select("HEADBUTT", list(range(len(self.discard_pile))), pending_card=None)
            return True
        if counter_damage > 0:
            self._take_counter_damage(counter_damage)
        return False

    def play_card(
        self,
        hand_index: int,
        target_index: int = 0,
        *,
        force_exhaust: bool = False,
        free_to_play: bool = False,
        energy_on_use: int | None = None,
    ) -> None:
        if hand_index < 0 or hand_index >= len(self.hand):
            raise IndexError(hand_index)
        card = self.hand.pop(hand_index)
        if not self.playable(card, free_to_play=free_to_play):
            raise ValueError(f"card is not playable: {card.name}")
        target = self.monsters[target_index] if self.monsters else None
        if card.card_def.x_cost:
            energy_spent = max(0, int(energy_on_use if energy_on_use is not None else (0 if free_to_play else self.player.energy)))
        else:
            energy_spent = 0 if free_to_play else self._card_energy_cost(card)
        if not free_to_play:
            self.player.energy = max(0, self.player.energy - energy_spent)
        post_play_top_card = False
        if any(other.card_id == "Pain" for other in self.hand):
            self._lose_hp(1)
        for monster in self._alive_monsters():
            if monster.power("Beat of Death") > 0:
                self._lose_hp(monster.power("Beat of Death"))

        card_id = card.card_id
        played_attack = card.card_def.card_type == "ATTACK"
        pending_hex_dazed = 1 if self.player.power("Hex") > 0 and card.card_def.card_type in {"SKILL", "POWER"} else 0

        def _insert_pending_hex_dazed_now() -> None:
            nonlocal pending_hex_dazed
            if pending_hex_dazed <= 0:
                return
            for _ in range(pending_hex_dazed):
                self._insert_temp_card_into_draw_pile("Dazed")
            pending_hex_dazed = 0

        force_end_turn_after_card = False
        self.cards_played_this_turn += 1
        if card.card_def.card_type in {"ATTACK", "SKILL", "POWER"}:
            self.card_types_played_this_turn.add(card.card_def.card_type)
            if self._has_relic("Orange Pellets") and {"ATTACK", "SKILL", "POWER"}.issubset(self.card_types_played_this_turn):
                for debuff in ("Vulnerable", "Weakened", "Frail", "Confusion", "No Draw", "No Block", "No Attack", "Dexterity Down", "Flex Strength Down"):
                    self.player.powers.pop(debuff, None)
        for monster in self._alive_monsters():
            if monster.monster_id == "TimeEater":
                monster.add_power("Time Warp", 1)
                if monster.power("Time Warp") >= 12:
                    monster.powers["Time Warp"] = 0
                    monster.add_power("Strength", 2)
                    self.player.energy = 0
                    force_end_turn_after_card = True
            if monster.monster_id == "GiantHead":
                monster.add_power("Slow", 1)
        if self.player.power("Panache") > 0 and self.cards_played_this_turn % 5 == 0:
            self._deal_direct_damage_all(self.player.power("Panache"))
        exhaust_override: bool | None = None
        attack_replays = 0
        if card.card_def.card_type == "SKILL":
            for monster in self._alive_monsters():
                if monster.power("Enrage") > 0:
                    monster.add_power("Strength", monster.power("Enrage"))
            self.skills_played_this_turn += 1
            if self._has_relic("Letter Opener") and self.skills_played_this_turn >= 3 and self.skills_played_this_turn % 3 == 0:
                self._deal_direct_damage_all(5)
        if card.card_def.card_type == "POWER":
            for monster in self._alive_monsters():
                if monster.power("Curiosity") > 0:
                    monster.add_power("Strength", 1)
            if self._has_relic("Bird Faced Urn"):
                self._heal(2)
            if self._has_relic("Mummified Hand"):
                candidates = [other for other in self.hand if other.cost > 0 and self._card_energy_cost(other) > 0]
                if candidates:
                    self.card_random_rng.random(len(candidates) - 1)
                    candidates[0].cost_for_turn = 0
        if played_attack:
            self.attack_played_this_turn += 1
            if self._has_relic("Necronomicon") and energy_spent >= 2:
                relic = self._relic("Necronomicon")
                if relic is not None and int(relic.get("counter", 0)) != self.turn:
                    relic["counter"] = self.turn
                    attack_replays += 1
            if self._advance_relic_counter("Nunchaku", 10):
                self.player.energy += 1
            if self._advance_relic_counter("Pen Nib", 10):
                self.player.add_power("Pen Nib", 1)
            if self.player.power("Rage") > 0:
                self._gain_block(self.player.power("Rage"), apply_block_modifiers=False)
            if self.player.power("Double Tap") > 0:
                self.player.add_power("Double Tap", -1)
                attack_replays += 1
        if self._advance_relic_counter("Ink Bottle", 10):
            _insert_pending_hex_dazed_now()
            self.draw_cards(1)

        if card.card_def.card_type == "CURSE":
            self._lose_hp(1)
            exhaust_override = True
        elif card.card_def.card_type == "STATUS":
            exhaust_override = True
        elif card_id == "Strike_R":
            base = 9 if card.upgrades else 6
            if self._has_relic("Strike Dummy"):
                base += 3
            self._deal_attack_damage(base, target)
        elif card_id == "Defend_R":
            self._gain_block(8 if card.upgrades else 5)
        elif card_id == "Bash":
            self._deal_attack_damage(10 if card.upgrades else 8, target)
            self._apply_monster_power(target, "Vulnerable", 3 if card.upgrades else 2)
        elif card_id == "Anger":
            self._deal_attack_damage(8 if card.upgrades else 6, target)
            self.discard_pile.append(make_card("Anger", upgrades=card.upgrades, uuid=self._new_uuid("Anger")))
        elif card_id == "Body Slam":
            self._deal_attack_damage(self.player.block, target)
        elif card_id == "Clash":
            self._deal_attack_damage(18 if card.upgrades else 14, target)
        elif card_id == "Cleave":
            self._deal_damage_all(11 if card.upgrades else 8)
        elif card_id == "Clothesline":
            self._deal_attack_damage(14 if card.upgrades else 12, target)
            self._apply_monster_power(target, "Weakened", 3 if card.upgrades else 2)
        elif card_id == "Headbutt":
            self.pending_counter_damage = 0
            self._deal_attack_damage(12 if card.upgrades else 9, target, defer_counter_damage=True)
            counter_damage = self.pending_counter_damage
            self.pending_counter_damage = 0
            self._check_outcome()
            if self.outcome != "UNDECIDED":
                pass
            elif len(self.discard_pile) == 1:
                self.draw_pile.append(self.discard_pile.pop(0))
                if counter_damage > 0:
                    self._take_counter_damage(counter_damage)
            elif len(self.discard_pile) > 1:
                self.pending_counter_damage = counter_damage
                self.pending_attack_replays = attack_replays
                self.pending_attack_target_index = self.monsters.index(target) if target in self.monsters else None
                self._open_discard_card_select(
                    "HEADBUTT",
                    list(range(len(self.discard_pile))),
                    pending_card=card,
                    pending_force_exhaust=force_exhaust,
                    pending_hex_dazed=pending_hex_dazed,
                )
                return
            elif counter_damage > 0:
                self._take_counter_damage(counter_damage)
        elif card_id == "Heavy Blade":
            self._deal_attack_damage(14, target, strength_multiplier=5 if card.upgrades else 3)
        elif card_id == "Iron Wave":
            self.pending_counter_damage = 0
            self._deal_attack_damage(7 if card.upgrades else 5, target, defer_counter_damage=True)
            counter_damage = self.pending_counter_damage
            self.pending_counter_damage = 0
            self._gain_block(_player_block_amount(7 if card.upgrades else 5, self.player))
            if counter_damage > 0:
                self._take_counter_damage(counter_damage)
        elif card_id == "Perfected Strike":
            strike_count = _combat_strike_count(self, card)
            self._deal_attack_damage((6 if card.upgrades else 6) + strike_count * (3 if card.upgrades else 2), target)
        elif card_id == "Flex":
            amount = 4 if card.upgrades else 2
            self.player.add_power("Strength", amount)
            self._apply_player_power("Flex Strength Down", amount)
        elif card_id == "Inflame":
            self.player.add_power("Strength", 3 if card.upgrades else 2)
        elif card_id == "Pommel Strike":
            self._deal_attack_damage(10 if card.upgrades else 9, target)
            _insert_pending_hex_dazed_now()
            self.draw_cards(2 if card.upgrades else 1)
        elif card_id == "Shrug It Off":
            self._gain_block(11 if card.upgrades else 8)
            _insert_pending_hex_dazed_now()
            self.draw_cards(1)
        elif card_id == "Sword Boomerang":
            for _ in range(4 if card.upgrades else 3):
                self._deal_attack_damage(3, self._random_alive_monster())
        elif card_id == "Thunderclap":
            for monster in self.monsters:
                if monster.alive:
                    self._deal_attack_damage(7 if card.upgrades else 4, monster)
                    self._apply_monster_power(monster, "Vulnerable", 1)
        elif card_id == "Twin Strike":
            base = 7 if card.upgrades else 5
            if self._has_relic("Strike Dummy"):
                base += 3
            self._deal_attack_damage(base, target, hits=2)
        elif card_id == "Wild Strike":
            self._deal_attack_damage(17 if card.upgrades else 12, target)
            self._insert_temp_card_into_draw_pile("Wound")
        elif card_id == "Blood for Blood":
            self._deal_attack_damage(22 if card.upgrades else 18, target)
        elif card_id == "Carnage":
            self._deal_attack_damage(28 if card.upgrades else 20, target)
        elif card_id == "Dropkick":
            self._deal_attack_damage(8 if card.upgrades else 5, target)
            if target and target.power("Vulnerable") > 0:
                self.player.energy += 1
                _insert_pending_hex_dazed_now()
                self.draw_cards(1)
        elif card_id == "Hemokinesis":
            self._lose_hp(2)
            self._deal_attack_damage(20 if card.upgrades else 15, target)
        elif card_id == "Pummel":
            self._deal_attack_damage(2, target, hits=5 if card.upgrades else 4)
        elif card_id == "Rampage":
            self._deal_attack_damage(8 + card.misc, target)
            card.misc += 8 if card.upgrades else 5
        elif card_id == "Reckless Charge":
            self._deal_attack_damage(10 if card.upgrades else 7, target)
            self._insert_temp_card_into_draw_pile("Dazed")
        elif card_id == "Searing Blow":
            self._deal_attack_damage(12 + card.upgrades * 4, target)
        elif card_id == "Sever Soul":
            self._exhaust_non_attacks_from_hand()
            self._deal_attack_damage(22 if card.upgrades else 16, target)
        elif card_id == "Uppercut":
            self._deal_attack_damage(13 if card.upgrades else 10, target)
            self._apply_monster_power(target, "Vulnerable", 2 if card.upgrades else 1)
            self._apply_monster_power(target, "Weakened", 2 if card.upgrades else 1)
        elif card_id == "Whirlwind":
            x_amount = energy_spent + (2 if self._has_relic("Chemical X") else 0)
            for _ in range(max(0, x_amount)):
                self._deal_damage_all(8 if card.upgrades else 5)
        elif card_id == "Bludgeon":
            self._deal_attack_damage(42 if card.upgrades else 32, target)
        elif card_id == "Feed":
            before_alive = target.alive if target else False
            self._deal_attack_damage(12 if card.upgrades else 10, target)
            if before_alive and target and not target.alive:
                self.player.max_hp += 4 if card.upgrades else 3
                self.player.current_hp += 4 if card.upgrades else 3
        elif card_id == "Fiend Fire":
            count = len(self.hand)
            while self.hand:
                exhausted = self.hand[self.card_random_rng.random(len(self.hand) - 1)]
                self.hand.remove(exhausted)
                self._exhaust_card(exhausted)
            self._deal_attack_damage(10 if card.upgrades else 7, target, hits=count)
        elif card_id == "Immolate":
            self._deal_damage_all(28 if card.upgrades else 21)
            self.discard_pile.append(make_card("Burn", uuid=self._new_uuid("Burn")))
        elif card_id == "Reaper":
            healed = self._deal_damage_all(5 if card.upgrades else 4)
            self.player.current_hp = min(self.player.max_hp, self.player.current_hp + healed)
        elif card_id == "Armaments":
            upgradeable = [] if card.upgrades else [index for index, hand_card in enumerate(self.hand) if _card_can_upgrade(hand_card)]
            self._gain_block(5 if card.upgrades else 5, defer_juggernaut=len(upgradeable) > 1)
            if card.upgrades:
                for hand_card in self.hand:
                    if _card_can_upgrade(hand_card):
                        self._upgrade_combat_card(hand_card)
            else:
                if len(upgradeable) == 1:
                    self._upgrade_combat_card(self.hand[upgradeable[0]])
                elif len(upgradeable) > 1:
                    self._open_combat_card_select(
                        "ARMAMENTS",
                        upgradeable,
                        pending_card=card,
                        pending_force_exhaust=force_exhaust,
                        pending_hex_dazed=pending_hex_dazed,
                    )
                    return
        elif card_id == "Havoc":
            post_play_top_card = True
        elif card_id == "True Grit":
            self._gain_block(9 if card.upgrades else 7)
            candidates = self.hand if card.upgrades else list(self.hand)
            if candidates:
                chosen = candidates[0] if card.upgrades else self.card_random_rng.choice(candidates)
                self.hand.remove(chosen)
                self._exhaust_card(chosen)
        elif card_id == "Warcry":
            _insert_pending_hex_dazed_now()
            self.draw_cards(2 if card.upgrades else 1)
            if len(self.hand) == 1:
                self.card_random_rng.random(1)
                self.draw_pile.append(self.hand.pop(0))
            elif len(self.hand) > 1:
                self._open_combat_card_select(
                    "WARCRY",
                    list(range(len(self.hand))),
                    pending_card=card,
                    pending_force_exhaust=force_exhaust,
                    pending_hex_dazed=pending_hex_dazed,
                )
                return
        elif card_id == "Battle Trance":
            _insert_pending_hex_dazed_now()
            self.draw_cards(4 if card.upgrades else 3)
            self._apply_player_power("No Draw", 1)
        elif card_id == "Bloodletting":
            self._lose_hp(3)
            self.player.energy += 3 if card.upgrades else 2
        elif card_id == "Burning Pact":
            if len(self.hand) == 1:
                deferred_dark_embrace_draws: list[int] = []
                exhausted_card = self.hand.pop(0)
                self._exhaust_card(exhausted_card, defer_dark_embrace_draws=deferred_dark_embrace_draws)
                draw_count = 3 if card.upgrades else 2
                _insert_pending_hex_dazed_now()
                self.draw_cards(draw_count)
                self._resolve_after_use_card_move(
                    card,
                    force_exhaust=force_exhaust,
                    defer_dark_embrace_draws=deferred_dark_embrace_draws,
                    add_hex_dazed=pending_hex_dazed,
                )
                for deferred_draw_count in deferred_dark_embrace_draws:
                    self.draw_cards(deferred_draw_count)
                self._check_outcome()
                return
            elif self.hand:
                self._open_combat_card_select(
                    "EXHAUST_ONE",
                    list(range(len(self.hand))),
                    pending_card=card,
                    pending_force_exhaust=force_exhaust,
                    pending_hex_dazed=pending_hex_dazed,
                )
                return
            else:
                _insert_pending_hex_dazed_now()
                self.draw_cards(3 if card.upgrades else 2)
        elif card_id == "Disarm":
            self._apply_monster_power(target, "Strength", -2)
        elif card_id == "Dual Wield":
            selectable = [index for index, other in enumerate(self.hand) if other.card_def.card_type in {"ATTACK", "POWER"}]
            if len(selectable) == 1:
                copy_count = 2 if card.upgrades > 0 else 1
                selected_card = clone_card(self.hand[selectable[0]])
                for _ in range(copy_count):
                    if len(self.hand) < 10:
                        self.hand.append(clone_card(selected_card))
                    else:
                        self.discard_pile.append(clone_card(selected_card))
            elif selectable:
                self._open_combat_card_select(
                    "DUAL_WIELD",
                    selectable,
                    pending_card=card,
                    pending_force_exhaust=force_exhaust,
                    pending_hex_dazed=pending_hex_dazed,
                )
                return
        elif card_id == "Entrench":
            self.player.block *= 2
        elif card_id == "Flame Barrier":
            self._gain_block(16 if card.upgrades else 12)
            self.player.add_power("Flame Barrier", 6 if card.upgrades else 4)
        elif card_id == "Ghostly Armor":
            self._gain_block(13 if card.upgrades else 10)
        elif card_id == "Infernal Blade":
            attack_ids = [card_id for card_id in COMBAT_ATTACK_CARD_POOL_IRONCLAD if card_id not in self.locked_card_ids]
            if attack_ids:
                self._add_to_hand(attack_ids[int(self.card_random_rng.random(len(attack_ids) - 1))], cost_for_turn=0)
        elif card_id == "Intimidate":
            for monster in self._alive_monsters():
                self._apply_monster_power(monster, "Weakened", 2 if card.upgrades else 1)
        elif card_id == "Power Through":
            self._gain_block(20 if card.upgrades else 15)
            self._add_to_hand("Wound")
            self._add_to_hand("Wound")
        elif card_id == "Rage":
            self.player.add_power("Rage", 5 if card.upgrades else 3)
        elif card_id == "Second Wind":
            exhausted = 0
            for other in list(self.hand):
                if other.card_def.card_type != "ATTACK":
                    self.hand.remove(other)
                    self._exhaust_card(other)
                    exhausted += 1
            self._gain_block((7 if card.upgrades else 5) * exhausted)
        elif card_id == "Seeing Red":
            self.player.energy += 2
        elif card_id == "Sentinel":
            self._gain_block(8 if card.upgrades else 5)
        elif card_id == "Spot Weakness":
            if target and target.intent.startswith("ATTACK"):
                self.player.add_power("Strength", 4 if card.upgrades else 3)
        elif card_id == "Shockwave":
            for monster in self._alive_monsters():
                self._apply_monster_power(monster, "Weakened", 5 if card.upgrades else 3)
                self._apply_monster_power(monster, "Vulnerable", 5 if card.upgrades else 3)
        elif card_id == "Double Tap":
            self.player.add_power("Double Tap", 2 if card.upgrades else 1)
        elif card_id == "Exhume":
            selectable = [index for index, other in enumerate(self.exhaust_pile) if other.card_id != "Exhume"]
            if len(self.hand) < 10 and len(selectable) == 1:
                exhumed = self.exhaust_pile.pop(selectable[0])
                self.hand.append(exhumed)
            elif len(self.hand) < 10 and len(selectable) > 1:
                self._open_exhaust_card_select(
                    "EXHUME",
                    selectable,
                    pending_card=card,
                    pending_force_exhaust=force_exhaust,
                    pending_hex_dazed=pending_hex_dazed,
                )
                return
        elif card_id == "Impervious":
            self._gain_block(40 if card.upgrades else 30)
        elif card_id == "Limit Break":
            strength = self.player.power("Strength")
            if strength != 0:
                self.player.add_power("Strength", strength)
            if card.upgrades:
                exhaust_override = False
        elif card_id == "Offering":
            self._lose_hp(6)
            self.player.energy += 2
            _insert_pending_hex_dazed_now()
            self.draw_cards(5 if card.upgrades else 3)
        elif card_id == "J.A.X.":
            self._lose_hp(3)
            self.player.add_power("Strength", 2)
        elif card_id == "Bandage Up":
            self._heal(6 if card.upgrades else 4)
        elif card_id == "Blind":
            if card.upgrades:
                for monster in self._alive_monsters():
                    self._apply_monster_power(monster, "Weakened", 2)
            elif target:
                self._apply_monster_power(target, "Weakened", 2)
        elif card_id == "Dark Shackles":
            if target:
                amount = 15 if card.upgrades else 9
                had_artifact = target.power("Artifact") > 0
                if had_artifact:
                    self._apply_monster_power(target, "Strength", -amount)
                    target.add_power("Shackled", amount)
                else:
                    # Mirror lightspeed's current DARK_SHACKLES implementation:
                    # it debuffs MS::STRENGTH with a positive amount.
                    self._apply_monster_power(target, "Strength", amount)
        elif card_id == "Deep Breath":
            if self.discard_pile:
                java_collections_shuffle(self.discard_pile, self.shuffle_rng.random_long())
                if not self.draw_pile:
                    self.draw_pile = list(self.discard_pile)
                else:
                    self.draw_pile.extend(self.discard_pile)
                self.discard_pile = []
                self._shuffle_cards(self.draw_pile)
            _insert_pending_hex_dazed_now()
            self.draw_cards(2 if card.upgrades else 1)
        elif card_id == "Discovery":
            options = self._discovery_card_options()
            if options:
                self._open_generated_combat_card_select(
                    "DISCOVERY",
                    options,
                    pending_card=card,
                    pending_force_exhaust=force_exhaust,
                    pending_hex_dazed=pending_hex_dazed,
                )
                return
        elif card_id == "Dramatic Entrance":
            self._deal_damage_all(12 if card.upgrades else 8)
        elif card_id == "Enlightenment":
            for other in self.hand:
                if other.card_def.card_type in {"ATTACK", "SKILL", "POWER"} and other.cost > 1:
                    other.cost_for_turn = 1
        elif card_id == "Finesse":
            self._gain_block(4 if card.upgrades else 2)
            _insert_pending_hex_dazed_now()
            self.draw_cards(1)
        elif card_id == "Flash of Steel":
            self._deal_attack_damage(6 if card.upgrades else 3, target)
            _insert_pending_hex_dazed_now()
            self.draw_cards(1)
        elif card_id == "Forethought":
            if len(self.hand) == 1:
                chosen = self.hand.pop(0)
                if chosen.cost > 0:
                    chosen.cost_for_turn = 0
                self.draw_pile.insert(0, chosen)
            elif len(self.hand) > 1:
                self._open_combat_card_select(
                    "FORETHOUGHT",
                    list(range(len(self.hand))),
                    pending_card=card,
                    pending_force_exhaust=force_exhaust,
                    pending_hex_dazed=pending_hex_dazed,
                )
                return
        elif card_id == "Good Instincts":
            self._gain_block(9 if card.upgrades else 6)
        elif card_id == "Impatience":
            # Mirror lightspeed's current Impatience implementation, which
            # always draws because the hasAttack sentinel never flips true.
            _insert_pending_hex_dazed_now()
            self.draw_cards(3 if card.upgrades else 2)
        elif card_id == "Jack Of All Trades":
            for _ in range(2 if card.upgrades else 1):
                self._add_random_card_to_hand(colorless=True)
        elif card_id == "Madness":
            have_nonzero_cost = False
            have_nonzero_turn_cost = False
            for other in self.hand:
                if other.cost_for_turn is not None and other.cost_for_turn > 0:
                    have_nonzero_turn_cost = True
                    break
                if other.cost > 0:
                    have_nonzero_cost = True
            if have_nonzero_turn_cost or have_nonzero_cost:
                while True:
                    chosen = self.hand[self.card_random_rng.random(len(self.hand) - 1)]
                    if have_nonzero_turn_cost:
                        if chosen.cost_for_turn is not None and chosen.cost_for_turn > 0:
                            chosen.card_def = replace(chosen.card_def, cost=0, upgraded_cost=0)
                            chosen.cost_for_turn = 0
                            break
                    elif chosen.cost > 0:
                        chosen.card_def = replace(chosen.card_def, cost=0, upgraded_cost=0)
                        chosen.cost_for_turn = 0
                        break
        elif card_id == "Mind Blast":
            self._deal_attack_damage(len(self.deck), target)
        elif card_id == "Panacea":
            self.player.add_power("Artifact", 2 if card.upgrades else 1)
        elif card_id == "Panic Button":
            self._gain_block(40 if card.upgrades else 30)
            # Lightspeed applies No Block with a two-turn duration that ticks
            # down at the end of the current turn, leaving only the next player
            # turn blocked.
            self._apply_player_power("No Block", 2, just_applied=False)
        elif card_id == "Purity":
            self._open_confirm_combat_card_select(
                "EXHAUST_MANY",
                pending_card=card,
                pending_force_exhaust=force_exhaust,
                pending_hex_dazed=pending_hex_dazed,
            )
            return
        elif card_id == "Swift Strike":
            self._deal_attack_damage(10 if card.upgrades else 7, target)
        elif card_id == "Trip":
            for monster in self._alive_monsters():
                self._apply_monster_power(monster, "Vulnerable", 3 if card.upgrades else 2)
        elif card_id == "Apotheosis":
            for pile in (self.hand, self.draw_pile, self.discard_pile, self.deck):
                for other in pile:
                    if other.card_def.card_type not in {"STATUS", "CURSE"}:
                        other.upgrades = max(other.upgrades, 1)
        elif card_id == "Chrysalis":
            self._put_random_cards_in_draw_pile(card_type="SKILL", count=5 if card.upgrades else 3, cost_for_turn=0)
        elif card_id in {"Hand of Greed", "HandOfGreed"}:
            before_alive = target.alive if target else False
            self._deal_attack_damage(25 if card.upgrades else 20, target)
            if before_alive and target and not target.alive:
                self.gold_gain += 25 if card.upgrades else 20
        elif card_id == "Magnetism":
            self.player.add_power("Magnetism", 1)
        elif card_id == "Master of Strategy":
            _insert_pending_hex_dazed_now()
            self.draw_cards(4 if card.upgrades else 3)
        elif card_id == "Mayhem":
            self.player.add_power("Mayhem", 1)
        elif card_id == "Metamorphosis":
            self._put_random_cards_in_draw_pile(card_type="ATTACK", count=5 if card.upgrades else 3, cost_for_turn=0)
        elif card_id == "Panache":
            self.player.add_power("Panache", 14 if card.upgrades else 10)
        elif card_id == "Sadistic Nature":
            self.player.add_power("Sadistic Nature", 7 if card.upgrades else 5)
        elif card_id == "Secret Technique":
            self._fetch_from_draw_to_hand("SKILL")
        elif card_id == "Secret Weapon":
            self._fetch_from_draw_to_hand("ATTACK")
        elif card_id == "The Bomb":
            self.player.add_power("The Bomb", 3)
            self.player.powers["The Bomb Damage"] = 50 if card.upgrades else 40
        elif card_id == "Thinking Ahead":
            _insert_pending_hex_dazed_now()
            self.draw_cards(2)
            if len(self.hand) == 1:
                self.card_random_rng.random(1)
                self.draw_pile.append(self.hand.pop(0))
            elif len(self.hand) > 1:
                self._open_combat_card_select(
                    "WARCRY",
                    list(range(len(self.hand))),
                    pending_card=card,
                    pending_force_exhaust=force_exhaust,
                    pending_hex_dazed=pending_hex_dazed,
                )
                return
        elif card_id == "Transmutation":
            x_amount = energy_spent + (2 if self._has_relic("Chemical X") else 0)
            for _ in range(max(0, x_amount)):
                self._add_random_card_to_hand(colorless=True, cost_for_turn=0)
        elif card_id == "Violence":
            fetched = 0
            limit = 4 if card.upgrades else 3
            for other in list(self.draw_pile):
                if fetched >= limit:
                    break
                if other.card_def.card_type == "ATTACK":
                    self.draw_pile.remove(other)
                    if len(self.hand) >= 10:
                        self.discard_pile.append(other)
                    else:
                        self.hand.append(other)
                    fetched += 1
        elif card_id == "Apparition":
            self.player.add_power("Intangible", 1)
        elif card_id == "Bite":
            before_hp = self.player.current_hp
            self._deal_attack_damage(8 if card.upgrades else 7, target)
            if self.player.current_hp >= before_hp:
                self._heal(3 if card.upgrades else 2)
        elif card_id == "Ritual Dagger":
            before_alive = target.alive if target else False
            self._deal_attack_damage((20 if card.upgrades else 15) + card.misc, target)
            if before_alive and target and not target.alive:
                card.misc += 5 if card.upgrades else 3
        elif card_id == "Barricade":
            self.player.add_power("Barricade", 1)
        elif card_id == "Berserk":
            self._apply_player_power("Vulnerable", 1 if card.upgrades else 2, just_applied=False)
            self.player.add_power("Berserk Energy", 1)
        elif card_id == "Brutality":
            self.player.add_power("Brutality", 1)
        elif card_id == "Combust":
            self.player.add_power("Combust", 7 if card.upgrades else 5)
            self.combust_hp_loss += 1
        elif card_id == "Corruption":
            self.player.add_power("Corruption", 1)
        elif card_id == "Dark Embrace":
            self.player.add_power("Dark Embrace", 1)
        elif card_id == "Demon Form":
            self.player.add_power("Demon Form", 3 if card.upgrades else 2)
        elif card_id == "Evolve":
            self.player.add_power("Evolve", 2 if card.upgrades else 1)
        elif card_id == "Feel No Pain":
            self.player.add_power("Feel No Pain", 4 if card.upgrades else 3)
        elif card_id == "Fire Breathing":
            self.player.add_power("Fire Breathing", 10 if card.upgrades else 6)
        elif card_id == "Juggernaut":
            self.player.add_power("Juggernaut", 7 if card.upgrades else 5)
        elif card_id == "Metallicize":
            self.player.add_power("Metallicize", 4 if card.upgrades else 3)
        elif card_id == "Rupture":
            self.player.add_power("Rupture", 2 if card.upgrades else 1)
        else:
            raise ValueError(f"unsupported native simulator card effect: {card_id}")
        for _ in range(attack_replays):
            if self.outcome != "UNDECIDED":
                break
            self._replay_attack_card_effect(card, target, energy_spent)

        exhaust_played = force_exhaust or self._card_exhausts_on_use(card) or (self.player.power("Corruption") > 0 and card.card_def.card_type == "SKILL")
        if exhaust_override is not None:
            exhaust_played = exhaust_override
        if post_play_top_card:
            self._play_random_top_card()
        if pending_hex_dazed > 0:
            for _ in range(pending_hex_dazed):
                self._insert_temp_card_into_draw_pile("Dazed")
        if exhaust_played:
            self._exhaust_card(card)
        elif card.card_def.card_type == "POWER":
            pass
        else:
            self._move_card_to_discard(card)
        self._drain_pending_autoplay_cards()
        if self._has_relic("Unceasing Top") and not self.hand and self.player.energy > 0:
            self.draw_cards(1)
        if played_attack and self.attack_played_this_turn % 3 == 0:
            if self._has_relic("Shuriken"):
                self.player.add_power("Strength", 1)
            if self._has_relic("Kunai"):
                self.player.add_power("Dexterity", 1)
            if self._has_relic("Ornamental Fan"):
                self.player.block += 4
        self._check_outcome()
        if force_end_turn_after_card and self.outcome == "UNDECIDED":
            self.end_turn()

    def use_potion(self, potion_index: int, target_index: int = 0) -> None:
        if potion_index < 0 or potion_index >= len(self.potions):
            raise IndexError(potion_index)
        potion = self.potions[potion_index]
        if not potion.can_use:
            raise ValueError(f"potion slot is empty: {potion_index}")
        target = self.monsters[target_index] if self.monsters and target_index < len(self.monsters) else None
        potion_id = potion.potion_id
        potion_multiplier = 2 if self._has_relic("Sacred Bark") else 1
        if potion_id == "Fire Potion":
            if target:
                was_alive = target.alive
                self._apply_damage_to_monster(20 * potion_multiplier, target)
                if was_alive and not target.alive:
                    self._on_monster_defeated(target)
                elif was_alive:
                    self._maybe_split_slime_boss(target)
        elif potion_id == "Explosive Potion":
            for monster in list(self._alive_monsters()):
                was_alive = monster.alive
                self._apply_damage_to_monster(10 * potion_multiplier, monster)
                if was_alive and not monster.alive:
                    self._on_monster_defeated(monster)
                elif was_alive:
                    self._maybe_split_slime_boss(monster)
        elif potion_id == "Weak Potion":
            if target:
                self._apply_monster_power(target, "Weakened", 3 * potion_multiplier)
        elif potion_id == "Fear Potion":
            if target:
                self._apply_monster_power(target, "Vulnerable", 3 * potion_multiplier)
        elif potion_id == "Strength Potion":
            self.player.add_power("Strength", 2 * potion_multiplier)
        elif potion_id == "Dexterity Potion":
            self.player.add_power("Dexterity", 2 * potion_multiplier)
        elif potion_id == "Block Potion":
            self._gain_block(12 * potion_multiplier)
        elif potion_id == "Energy Potion":
            self.player.energy += 2 * potion_multiplier
        elif potion_id == "Swift Potion":
            self.draw_cards(3 * potion_multiplier)
        elif potion_id == "Blood Potion":
            self._heal(max(1, int(self.player.max_hp * 0.2 * potion_multiplier)))
        elif potion_id == "Flex Potion":
            self.player.add_power("Strength", 5 * potion_multiplier)
            self.player.add_power("Flex Strength Down", 5 * potion_multiplier)
        elif potion_id == "Attack Potion":
            attack_ids = [card.card_id for card in self._ironclad_card_pool(card_type="ATTACK")]
            self._add_to_hand(self.rng.choice(attack_ids), cost_for_turn=0)
        elif potion_id == "Skill Potion":
            skill_ids = [card.card_id for card in self._ironclad_card_pool(card_type="SKILL")]
            self._add_to_hand(self.rng.choice(skill_ids), cost_for_turn=0)
        elif potion_id == "Power Potion":
            power_ids = [card.card_id for card in self._ironclad_card_pool(card_type="POWER")]
            self._add_to_hand(self.rng.choice(power_ids), cost_for_turn=0)
        elif potion_id == "Ancient Potion":
            self.player.add_power("Artifact", 1)
        elif potion_id == "Blessing of the Forge":
            for card in self.hand:
                if card.card_def.card_type not in {"STATUS", "CURSE"}:
                    card.upgrades = max(card.upgrades, 1)
        elif potion_id == "Colorless Potion":
            for _ in range(potion_multiplier):
                self._add_random_card_to_hand(colorless=True, cost_for_turn=0)
        elif potion_id == "Duplication Potion":
            self.player.add_power("Double Tap", potion_multiplier)
        elif potion_id == "ElixirPotion":
            self.exhaust_pile.extend(card for card in self.hand if card.card_def.card_type == "STATUS")
            self.hand = [card for card in self.hand if card.card_def.card_type != "STATUS"]
        elif potion_id == "Essence of Steel":
            self.player.add_power("Plated Armor", 4 * potion_multiplier)
        elif potion_id == "Speed Potion":
            self.player.add_power("Dexterity", 5 * potion_multiplier)
            self._apply_player_power("Dexterity Down", 5 * potion_multiplier)
        elif potion_id == "Steroid Potion":
            self.player.add_power("Strength", 5 * potion_multiplier)
            self._apply_player_power("Flex Strength Down", 5 * potion_multiplier)
        elif potion_id == "Regen Potion":
            self.player.add_power("Regen", 5 * potion_multiplier)
        elif potion_id == "Liquid Bronze":
            self.player.add_power("Thorns", 3 * potion_multiplier)
        elif potion_id == "Liquid Memories":
            if self.discard_pile:
                card = self.discard_pile.pop()
                card.cost_for_turn = 0
                self.hand.append(card) if len(self.hand) < 10 else self.discard_pile.append(card)
        elif potion_id == "Gambler's Brew":
            count = len(self.hand)
            self.discard_pile.extend(self.hand)
            self.hand = []
            self.draw_cards(count)
        elif potion_id == "Entropic Brew":
            self.potions[potion_index] = PotionInstance()
            for index, current in enumerate(self.potions):
                if not current.can_use:
                    self.potions[index] = roll_potion(self.randoms.potion, limited=True)
            if self._has_relic("Toy Ornithopter"):
                self._heal(5)
            self._check_outcome()
            return
        elif potion_id == "Fruit Juice":
            self.player.max_hp += 5 * potion_multiplier
            self.player.current_hp += 5 * potion_multiplier
        elif potion_id == "Heart of Iron":
            self.player.add_power("Metallicize", 6 * potion_multiplier)
        elif potion_id == "Distilled Chaos":
            for _ in range(3 * potion_multiplier):
                self._play_random_top_card()
        elif potion_id == "Cultist Potion":
            self.player.add_power("Ritual", 1 * potion_multiplier)
        elif potion_id == "Snecko Oil":
            self.draw_cards(5 * potion_multiplier)
            for card in self.hand:
                card.cost_for_turn = self.card_random_rng.randint(0, 3)
        elif potion_id == "Smoke Bomb":
            if self.floor not in {16, 33, 50, 53, 54}:
                for monster in self.monsters:
                    monster.current_hp = 0
                    monster.is_gone = True
                self.outcome = "PLAYER_VICTORY"
        elif potion_id == "FairyPotion":
            self._heal(max(1, int(self.player.max_hp * 0.3 * potion_multiplier)))
        else:
            raise ValueError(f"unsupported potion effect: {potion_id}")
        if self._has_relic("Toy Ornithopter"):
            self._heal(5)
        self.potions[potion_index] = PotionInstance()
        self._check_outcome()

    def end_turn(self) -> None:
        discarded = 0
        deferred_dark_embrace_draws: list[int] = []
        original_hand_count = len(self.hand)
        for card in list(self.hand):
            if card.card_id not in {"Burn", "Decay", "Regret", "Doubt", "Shame"}:
                continue
            self.hand.remove(card)
            if card.card_id == "Burn":
                self._take_counter_damage(4 if card.upgrades else 2)
            elif card.card_id == "Decay":
                self._take_counter_damage(2)
            elif card.card_id == "Regret":
                self._lose_hp(original_hand_count)
            elif card.card_id == "Doubt":
                self._apply_player_power("Weakened", 1)
            elif card.card_id == "Shame":
                self._apply_player_power("Frail", 1)
            self._move_card_to_discard(card)
            discarded += 1
        for hand_index in range(len(self.hand) - 1, -1, -1):
            card = self.hand[hand_index]
            if self._has_relic("Runic Pyramid") and not card.card_def.ethereal and card.card_id != "Burn":
                continue
            self.hand.pop(hand_index)
            if card.card_def.ethereal:
                self._exhaust_card(card, defer_dark_embrace_draws=deferred_dark_embrace_draws)
            else:
                self._move_card_to_discard(card)
                discarded += 1
        self.cards_discarded_this_turn += discarded
        for draw_count in deferred_dark_embrace_draws:
            self.draw_cards(draw_count)
        # Entangled/No Attack is cleared at the end of the affected player turn,
        # not by the normal debuff duration tick.
        self.player.powers.pop("No Attack", None)
        if self._has_relic("Art of War") and self.attack_played_this_turn == 0:
            self.player.powers["Art of War Energy"] = 1
        if self._has_relic("Pocketwatch") and self.cards_played_this_turn <= 3:
            self.player.powers["Pocketwatch Draw"] = 1
        if self.player.power("Flex Strength Down") > 0:
            amount = self.player.power("Flex Strength Down")
            self.player.add_power("Strength", -amount)
            self.player.powers.pop("Flex Strength Down", None)
        self.player.powers.pop("No Draw", None)
        if self._has_relic("Orichalcum") and self.player.block == 0:
            self.player.block = 6
        if self.player.power("Metallicize") > 0:
            self._gain_block(self.player.power("Metallicize"), apply_block_modifiers=False)
        if self.player.power("Plated Armor") > 0:
            self._gain_block(self.player.power("Plated Armor"), apply_block_modifiers=False)
        if self.player.power("Dexterity Down") > 0:
            amount = self.player.power("Dexterity Down")
            self.player.add_power("Dexterity", -amount)
            self.player.powers.pop("Dexterity Down", None)
        if self.player.power("Combust") > 0 and any(monster.alive for monster in self.monsters):
            self._lose_hp(self.combust_hp_loss)
            self._deal_direct_damage_all(self.player.power("Combust"))
        if self._has_relic("Nilry's Codex"):
            self.draw_pile.append(make_card(self._random_card_id(), uuid=self._new_uuid("NilrysCodex")))
        for monster in self.monsters:
            if monster.alive and monster.power("Barricade") <= 0:
                monster.block = 0
        monster_turn_index = 0
        pending_end_of_round_ai_noops = 0
        while monster_turn_index < len(self.monsters):
            monster = self.monsters[monster_turn_index]
            extra_roll_index: int | None = None
            if monster.alive:
                extra_roll_index = self._monster_take_turn(monster, monster_turn_index)
                if self.outcome != "UNDECIDED":
                    break
                if monster.alive:
                    if not monster.ai_state.pop("skip_end_round_roll", False):
                        choose_next_move(monster, self.ai_rng)
            if extra_roll_index is None:
                if monster.ai_state.pop("extra_roll_move_on_turn", False):
                    self.ai_rng.random(99)
                monster_turn_index += 1
            else:
                if 0 <= extra_roll_index < len(self.monsters):
                    extra_roll_monster = self.monsters[extra_roll_index]
                    if extra_roll_monster.ai_state.pop("extra_roll_move_on_turn", False):
                        self.ai_rng.random(99)
                monster_turn_index = extra_roll_index + 1
        for _ in range(pending_end_of_round_ai_noops):
            self.ai_rng.random(99)
        for monster in self.monsters:
            self._apply_monster_end_of_turn_triggers(monster)
        if any(monster.monster_id == "Darkling" and monster.alive for monster in self.monsters):
            for monster in self.monsters:
                if monster.monster_id == "Darkling" and monster.half_dead:
                    monster.half_dead = False
                    monster.current_hp = max(1, monster.max_hp // 2)
        for monster in self.monsters:
            if not monster.alive:
                continue
            ritual = monster.power("Ritual")
            if ritual > 0:
                if monster.ai_state.pop("ritual_just_applied", False):
                    pass
                else:
                    monster.add_power("Strength", ritual)
            if monster.monster_id == "GiantHead":
                monster.powers["Slow"] = 0
        for monster in self.monsters:
            shackled = monster.power("Shackled")
            if shackled > 0:
                monster.add_power("Strength", shackled)
                monster.powers.pop("Shackled", None)
        self._tick_powers_at_end_of_round(self.player.powers, self.player_powers_just_applied)
        for monster in self.monsters:
            if monster.current_hp <= 0 or monster.ai_state.get("escaping"):
                continue
            self._tick_powers_at_end_of_round(monster.powers)
        self.player.powers.pop("Flame Barrier", None)
        self.player.powers.pop("Rage", None)
        self.player.powers.pop("Double Tap", None)
        self._check_outcome()
        if self.outcome == "UNDECIDED":
            self.start_player_turn()

    def _monster_take_turn(self, monster: MonsterState, monster_turn_index: int | None = None) -> int | None:
        starting_move = monster.move
        if "Flight" in monster.powers:
            monster.powers["Flight"] = 4 if self.ascension_level >= 17 else 3
        if monster.move in {"ACID_SLIME_L_SPLIT", "SPIKE_SLIME_L_SPLIT", "SLIME_BOSS_SPLIT"}:
            split_hp = max(1, int(monster.current_hp))
            if monster.move == "ACID_SLIME_L_SPLIT":
                child_ids = ("AcidSlime_M", "AcidSlime_M")
            elif monster.move == "SPIKE_SLIME_L_SPLIT":
                child_ids = ("SpikeSlime_M", "SpikeSlime_M")
            else:
                child_ids = ("SpikeSlime_L", "AcidSlime_L")
            if monster.move == "SLIME_BOSS_SPLIT":
                child_monsters = [self._spawn_split_child(child_ids[0], split_hp)]
                child_monsters.append(self._spawn_split_child(child_ids[1], split_hp))
            else:
                child_monsters = [self._spawn_split_child(child_id, split_hp) for child_id in child_ids]
            if monster_turn_index is not None and 0 <= monster_turn_index < len(self.monsters) and self.monsters[monster_turn_index] is monster:
                monster_index = monster_turn_index
            else:
                try:
                    monster_index = self.monsters.index(monster)
                except ValueError:
                    monster_index = -1
            monster.current_hp = 0
            monster.is_gone = True
            if monster.move == "SLIME_BOSS_SPLIT":
                placeholder = MonsterState(
                    "INVALID = 0",
                    "INVALID = 0",
                    0,
                    0,
                    "INVALID",
                    "UNKNOWN",
                )
                self.monsters = [child_monsters[0], placeholder, child_monsters[1]]
                return len(self.monsters)
            elif monster_index >= 0:
                self.monsters[monster_index] = child_monsters[0]
                if len(child_monsters) > 1:
                    secondary_child = child_monsters[1]
                    secondary_child.ai_state["extra_roll_move_on_turn"] = True
                    second_index = monster_index + 1
                    if second_index < len(self.monsters):
                        self.monsters[second_index] = secondary_child
                        if len(self.monsters) < 4:
                            self.monsters.append(
                                MonsterState(
                                    "INVALID = 0",
                                    "INVALID = 0",
                                    0,
                                    0,
                                    "INVALID",
                                    "UNKNOWN",
                                )
                            )
                    else:
                        self.monsters.append(secondary_child)
                self.ai_rng.random(99)
                return monster_index + 1
            else:
                self.monsters.extend(child_monsters)
            return
        if monster.move == "HEXAGHOST_ACTIVATE":
            _set_move(monster, "HEXAGHOST_DIVIDER")
            monster.move_base_damage = self.player.current_hp // 12 + 1
            monster.move_hits = 6
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
            return
        if monster.move == "CULTIST_INCANTATION":
            monster.add_power("Ritual", 3)
            monster.ai_state["ritual_just_applied"] = True
            _set_move(monster, "CULTIST_DARK_STRIKE")
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
            return
        if monster.move == "THE_GUARDIAN_CHARGING_UP":
            monster.block += 9
            _set_move(monster, "THE_GUARDIAN_FIERCE_BASH")
            monster.ai_state["cycle_index"] = int(monster.ai_state.get("cycle_index", 0)) + 1
            monster.ai_state["skip_end_round_roll"] = True
            return
        if monster.move == "THE_GUARDIAN_DEFENSIVE_MODE":
            monster.add_power("Sharp Hide", 4 if self.ascension_level >= 19 else 3)
            _set_move(monster, "THE_GUARDIAN_ROLL_ATTACK")
            monster.ai_state["skip_end_round_roll"] = True
            return
        if monster.move in {"LOUSE_GROW", "RED_LOUSE_GROW", "FUNGI_BEAST_GROW"}:
            monster.add_power("Strength", 3)
            return
        if monster.move in {
            "ACID_SLIME_S_LICK",
            "ACID_SLIME_M_LICK",
            "ACID_SLIME_L_LICK",
            "LOUSE_SPIT_WEB",
            "FAT_GREMLIN_WEAKEN",
        }:
            self._apply_player_power("Weakened", 2 if monster.move == "ACID_SLIME_L_LICK" else 1)
            if monster.move == "ACID_SLIME_S_LICK":
                monster.move = "ACID_SLIME_S_TACKLE"
                monster.intent = "ATTACK"
                monster.move_base_damage = 4 if self.ascension_level >= 2 else 3
                monster.move_hits = 1
                monster.ai_state["skip_end_round_roll"] = True
            return
        if monster.move == "THE_GUARDIAN_VENT_STEAM":
            self._apply_player_power("Vulnerable", 2)
            self._apply_player_power("Weakened", 2)
            _set_move(monster, "THE_GUARDIAN_WHIRLWIND")
            monster.ai_state["skip_end_round_roll"] = True
            return
        if monster.move == "GREEN_LOUSE_SPIT_WEB":
            self._apply_player_power("Weakened", 2)
            return
        if monster.move == "SPIKE_SLIME_M_LICK":
            self._apply_player_power("Frail", 1)
            return
        if monster.move == "SPIKE_SLIME_L_LICK":
            self._apply_player_power("Frail", 3 if self.ascension_level >= 17 else 2)
            return
        if monster.move == "SHELLED_STUNNED":
            _set_move(monster, "SHELLED_FELL")
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
            return
        if monster.move in {"ACID_SLIME_M_CORROSIVE_SPIT", "SPIKE_SLIME_M_FLAME_TACKLE"}:
            self.discard_pile.append(make_card("Slimed", uuid=self._new_uuid("Slimed")))
        if monster.move in {"ACID_SLIME_L_CORROSIVE_SPIT", "SPIKE_SLIME_L_FLAME_TACKLE"}:
            self.discard_pile.append(make_card("Slimed", uuid=self._new_uuid("Slimed")))
            self.discard_pile.append(make_card("Slimed", uuid=self._new_uuid("Slimed")))
        if monster.move == "HEXAGHOST_SEAR":
            self.discard_pile.append(make_card("Burn", uuid=self._new_uuid("Burn")))
        if monster.move == "SENTRY_BOLT":
            self.discard_pile.append(make_card("Dazed", uuid=self._new_uuid("Dazed")))
            self.discard_pile.append(make_card("Dazed", uuid=self._new_uuid("Dazed")))
            _set_move(monster, "SENTRY_BEAM")
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
            return
        if monster.move == "SLIME_BOSS_PREPARING":
            _set_move(monster, "SLIME_BOSS_SLAM")
            monster.ai_state["skip_end_round_roll"] = True
            return
        if monster.move == "SLIME_BOSS_GOOP_SPRAY":
            for _ in range(3):
                self.discard_pile.append(make_card("Slimed", uuid=self._new_uuid("Slimed")))
            _set_move(monster, "SLIME_BOSS_PREPARING")
            monster.ai_state["skip_end_round_roll"] = True
            return
        if monster.move == "RED_SLAVER_ENTANGLE":
            self._apply_player_power("No Attack", 1)
            return
        if monster.move == "REPULSOR_REPULSE":
            self.discard_pile.append(make_card("Dazed", uuid=self._new_uuid("Dazed")))
            self.discard_pile.append(make_card("Dazed", uuid=self._new_uuid("Dazed")))
        if monster.move == "NEMESIS_DEBUFF":
            for _ in range(3):
                self.discard_pile.append(make_card("Burn", uuid=self._new_uuid("Burn")))
        if monster.move == "BRONZE_ORB_STASIS":
            if "stasis_card" not in monster.ai_state:
                stolen: CardInstance | None = None
                for pile in (self.draw_pile, self.discard_pile, self.hand):
                    candidates = [card for card in pile if card.card_def.card_type not in {"STATUS", "CURSE"}]
                    if candidates:
                        stolen = candidates[0]
                        pile.remove(stolen)
                        break
                if stolen is not None:
                    monster.ai_state["stasis_card"] = stolen
            return
        if monster.move == "LOOTER_ESCAPE" or monster.move == "MUGGER_ESCAPE":
            monster.is_gone = True
            self._check_outcome()
            return
        looter_turn_number = len(monster.move_history)
        if monster.move == "LOOTER_MUG":
            if looter_turn_number == 1:
                self.ai_rng.random_boolean(0.6)
            stolen = min(self.gold, 15)
            self.gold -= stolen
            monster.ai_state["stolen_gold"] = int(monster.ai_state.get("stolen_gold", 0) or 0) + stolen
        if monster.move == "LOOTER_LUNGE":
            stolen = min(self.gold, 15)
            self.gold -= stolen
            monster.ai_state["stolen_gold"] = int(monster.ai_state.get("stolen_gold", 0) or 0) + stolen
        if monster.move == "MUGGER_MUG":
            self.ai_rng.random(2)
            if looter_turn_number == 2:
                self.ai_rng.random_boolean(0.6)
            stolen = min(self.gold, 15)
            self.gold -= stolen
            monster.ai_state["stolen_gold"] = int(monster.ai_state.get("stolen_gold", 0) or 0) + stolen
        if monster.move == "MUGGER_LUNGE":
            self.ai_rng.random(2)
            stolen = min(self.gold, 15)
            self.gold -= stolen
            monster.ai_state["stolen_gold"] = int(monster.ai_state.get("stolen_gold", 0) or 0) + stolen
        if monster.move == "LOOTER_SMOKE_BOMB":
            monster.block += 6
            _set_move(monster, "LOOTER_ESCAPE")
            monster.ai_state["skip_end_round_roll"] = True
            return
        if monster.move == "MUGGER_SMOKE_BOMB":
            monster.block += 17 if self.ascension_level >= 17 else 11
            _set_move(monster, "MUGGER_ESCAPE")
            monster.ai_state["skip_end_round_roll"] = True
            return
        if monster.move == "BEAR_BEAR_HUG":
            self._apply_player_power("Dexterity", -2)
            return
        if monster.move == "ROMEO_AGONIZING_SLASH":
            self._apply_player_power("Weakened", 2)
        if monster.move == "ROMEO_MOCK":
            self._apply_player_power("Weakened", 2)
            self._apply_player_power("Frail", 2)
        if monster.move == "FAT_GREMLIN_SMASH":
            self._apply_player_power("Weakened", 1)
        if monster.move == "GREMLIN_NOB_BELLOW":
            monster.add_power("Enrage", 2 if self.ascension_level < 18 else 3)
            return
        if monster.move == "LAGAVULIN_SLEEP":
            if not monster.ai_state.get("asleep", 0):
                return
            monster.ai_state["sleep_turns"] = int(monster.ai_state.get("sleep_turns", 0)) + 1
            if monster.ai_state["sleep_turns"] >= 3:
                monster.ai_state["latent_awake"] = 1
            return
        if monster.move == "LAGAVULIN_SIPHON_SOUL":
            self._apply_player_power("Strength", -1)
            self._apply_player_power("Dexterity", -1)
            return
        if monster.move == "JAW_WORM_BELLOW":
            monster.block += 6
            monster.add_power("Strength", 3)
        elif monster.move == "BYRD_STUNNED":
            return
        elif monster.move == "BYRD_FLY":
            monster.add_power("Flight", 3 if self.ascension_level < 17 else 4)
            return
        elif monster.move == "BYRD_CAW":
            monster.add_power("Strength", 1)
            return
        elif monster.move == "SPIKER_GROW":
            monster.add_power("Thorns", 2)
            return
        elif monster.move in {"MYSTIC_BUFF", "GREMLIN_LEADER_ENCOURAGE", "COLLECTOR_BUFF", "AUTOMATON_BOOST", "ORB_WALKER_CHARGE", "MAW_ROAR", "GIANT_HEAD_COUNT", "DONU_CIRCLE_OF_POWER"}:
            if monster.move == "MYSTIC_BUFF":
                strength_amount = 4 if self.ascension_level >= 17 else 3 if self.ascension_level >= 2 else 2
                centurion = next((ally for ally in self.monsters if ally.alive and ally.monster_id == "Centurion"), None)
                if centurion is not None and centurion is not monster:
                    centurion.add_power("Strength", strength_amount)
                monster.add_power("Strength", strength_amount)
            else:
                monster.add_power("Strength", 2)
            if monster.move == "COLLECTOR_BUFF":
                for _ in range(2):
                    if sum(1 for ally in self._alive_monsters() if ally.monster_id == "TorchHead") >= 2:
                        break
                    self._summon_monster("TorchHead", max_alive=4)
            return
        elif monster.move == "GREMLIN_LEADER_RALLY":
            for _ in range(2):
                self._summon_monster(self.misc_rng.choice(["GremlinFat", "GremlinWizard", "GremlinThief", "GremlinTsundere", "GremlinWarrior"]), max_alive=4)
            return
        elif monster.move == "SHIELD_GREMLIN_PROTECT":
            source_index = self.monsters.index(monster)
            targets = [
                ally for index, ally in enumerate(self.monsters)
                if index != source_index and ally.alive
            ]
            target = targets[self.ai_rng.random(len(targets) - 1)] if targets else monster
            if self.ascension_level >= 17:
                block_amount = 11
            elif self.ascension_level >= 7:
                block_amount = 8
            else:
                block_amount = 7
            target.block += block_amount
            monster.ai_state["skip_end_round_roll"] = True
            return
        elif monster.move == "SHIELD_GREMLIN_SHIELD_BASH":
            monster.ai_state["skip_end_round_roll"] = True
            return
        elif monster.move == "MYSTIC_HEAL":
            heal_amount = 20 if self.ascension_level >= 17 else 16
            centurion = next((ally for ally in self.monsters if ally.alive and ally.monster_id == "Centurion"), None)
            if centurion is not None and centurion is not monster:
                centurion.current_hp = min(centurion.max_hp, centurion.current_hp + heal_amount)
            monster.current_hp = min(monster.max_hp, monster.current_hp + heal_amount)
            return
        elif monster.move == "SPHERIC_GUARDIAN_ACTIVATE":
            monster.block += 35 if self.ascension_level >= 17 else 25
            _set_move(monster, "SPHERIC_GUARDIAN_ATTACK_DEBUFF")
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
            return
        elif monster.move in {"CENTURION_DEFEND", "DECA_SQUARE_OF_PROTECTION", "SHIELD_FORTIFY"}:
            if monster.move == "CENTURION_DEFEND":
                mystic = next((ally for ally in self.monsters if ally.alive and ally.monster_id == "Mystic"), None)
                if mystic is not None and mystic is not monster:
                    mystic.block += 20 if self.ascension_level >= 17 else 15
            else:
                monster.block += 12
            if monster.move == "SHIELD_FORTIFY":
                monster.block += 18
            return
        elif monster.move == "HEXAGHOST_INFLAME":
            monster.block += 12
            monster.add_power("Strength", 3 if self.ascension_level >= 19 else 2)
            counter = int(monster.ai_state.get("hex_counter", 0))
            monster.ai_state["hex_counter"] = counter + 1
            _set_move(monster, "HEXAGHOST_TACKLE")
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
            return
        elif monster.move == "SNECKO_PERPLEXING_GLARE":
            self._apply_player_power("Confusion", 1)
            return
        elif monster.move == "CHOSEN_HEX":
            self._apply_player_power("Hex", 1)
            return
        elif monster.move in {"SNAKE_PLANT_ENFEEBLING_SPORES", "COLLECTOR_MEGA_DEBUFF", "MAW_DROOL", "WRITHING_WITHER", "REPULSOR_REPULSE", "NEMESIS_DEBUFF", "TIME_EATER_RIPPLE"}:
            self._apply_player_power("Weakened", 2)
            if monster.move == "SNAKE_PLANT_ENFEEBLING_SPORES":
                self._apply_player_power("Frail", 2)
            if monster.move == "TIME_EATER_RIPPLE":
                monster.current_hp = max(monster.current_hp, monster.max_hp // 2)
                monster.add_power("Strength", 2)
            return
        elif monster.move == "HEART_DEBILITATE":
            self._apply_player_power("Weakened", 2)
            self._apply_player_power("Vulnerable", 2)
            self._apply_player_power("Frail", 2)
            self.discard_pile.append(make_card("Dazed", uuid=self._new_uuid("Dazed")))
            return
        elif monster.move == "CHOSEN_DRAIN":
            self._apply_player_power("Weakened", 3)
            monster.add_power("Strength", 3)
            return
        elif monster.move == "CHOSEN_DEBILITATE":
            pass
        elif monster.move == "WRITHING_IMPLANT":
            self._apply_player_power("Vulnerable", 2)
            return
        elif monster.move == "SPHERIC_GUARDIAN_ATTACK_DEBUFF":
            pass
        elif monster.move in {"CHAMP_DEFENSIVE_STANCE"}:
            monster.block += 15
            monster.add_power("Strength", 2)
            return
        elif monster.move == "REPTOMANCER_SUMMON":
            self._summon_monster("SnakeDagger", max_alive=5)
            self._summon_monster("SnakeDagger", max_alive=5)
            return
        elif monster.move == "AWAKENED_REBIRTH":
            monster.current_hp = max(monster.current_hp, 300)
            monster.is_gone = False
            monster.add_power("Awakened Reborn", 1)
            return
        elif monster.move == "GREMLIN_WIZARD_CHARGING":
            charge = int(monster.ai_state.get("charge", 0)) + 1
            monster.ai_state["charge"] = charge
            if charge >= 3:
                _set_move(monster, "GREMLIN_WIZARD_ULTIMATE_BLAST")
            monster.ai_state["skip_end_round_roll"] = True
            return
        elif monster.move == "DARKLING_REINCARNATE":
            return
        damage = monster_adjusted_damage(monster, self.player, vulnerable_multiplier=self._monster_vulnerable_multiplier())
        total_hp_damage = 0
        for _ in range(max(1, monster.move_hits) if damage else 0):
            blocked = min(self.player.block, damage)
            self.player.block -= blocked
            hp_damage = damage - blocked
            total_hp_damage += hp_damage
            hp_lost_before = self.hp_lost_this_combat
            self._lose_hp(hp_damage, from_attack=True)
            if self.hp_lost_this_combat > hp_lost_before and self.player.power("Plated Armor") > 0:
                self.player.add_power("Plated Armor", -1)
            if self.player.power("Thorns") > 0:
                was_alive = monster.alive
                self._deal_direct_damage_to_monster(self.player.power("Thorns"), monster)
                if was_alive and not monster.alive:
                    self._on_monster_defeated(monster)
                elif was_alive:
                    self._maybe_split_slime_boss(monster)
            if self.player.power("Flame Barrier") > 0:
                was_alive = monster.alive
                self._deal_direct_damage_to_monster(self.player.power("Flame Barrier"), monster)
                if was_alive and not monster.alive:
                    self._on_monster_defeated(monster)
                elif was_alive:
                    self._maybe_split_slime_boss(monster)
        if starting_move == "JAW_WORM_THRASH" and monster.alive:
            monster.block += 5
        if monster.move == "GREMLIN_NOB_SKULL_BASH":
            self._apply_player_power("Vulnerable", 2)
        if starting_move in {"CULTIST_DARK_STRIKE", "FAT_GREMLIN_SMASH", "MAD_GREMLIN_SCRATCH"}:
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
        if starting_move == "SNEAKY_GREMLIN_PUNCTURE":
            monster.ai_state["skip_end_round_roll"] = True
        if starting_move == "BYRD_HEADBUTT":
            _set_move(monster, "BYRD_FLY")
            monster.ai_state["skip_end_round_roll"] = True
        if starting_move == "SLIME_BOSS_SLAM":
            _set_move(monster, "SLIME_BOSS_GOOP_SPRAY")
            monster.ai_state["skip_end_round_roll"] = True
        if starting_move == "SENTRY_BEAM":
            _set_move(monster, "SENTRY_BOLT")
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
        if starting_move == "GREMLIN_WIZARD_ULTIMATE_BLAST" and self.ascension_level < 17:
            monster.ai_state["charge"] = 0
            _set_move(monster, "GREMLIN_WIZARD_CHARGING")
            monster.ai_state["skip_end_round_roll"] = True
        if monster.move == "ACID_SLIME_S_TACKLE":
            monster.move = "ACID_SLIME_S_LICK"
            monster.intent = "DEBUFF"
            monster.move_base_damage = 0
            monster.move_hits = 0
            monster.ai_state["skip_end_round_roll"] = True
        if monster.move == "SPHERIC_GUARDIAN_ATTACK_DEBUFF":
            _set_move(monster, "SPHERIC_GUARDIAN_SLAM")
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
        elif monster.move == "SPHERIC_GUARDIAN_HARDEN":
            monster.block += 15
            _set_move(monster, "SPHERIC_GUARDIAN_SLAM")
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
        elif monster.move == "SPHERIC_GUARDIAN_SLAM":
            _set_move(monster, "SPHERIC_GUARDIAN_HARDEN")
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
        if monster.move in {"EXPLODER_EXPLODE", "DAGGER_EXPLODE"}:
            monster.current_hp = 0
            monster.is_gone = True
            self._on_monster_killed()
        if monster.monster_id == "Transient" and self.turn >= 5:
            monster.current_hp = 0
            monster.is_gone = True
        if monster.monster_id == "Darkling" and monster.half_dead and any(ally.monster_id == "Darkling" and ally.alive for ally in self.monsters):
            monster.half_dead = False
            monster.current_hp = max(1, monster.max_hp // 2)
        if monster.move == "SHIELD_BASH":
            monster.block += 30
        if monster.move == "SPEAR_BURN_STRIKE":
            self.discard_pile.append(make_card("Burn", uuid=self._new_uuid("Burn")))
            self.discard_pile.append(make_card("Burn", uuid=self._new_uuid("Burn")))
        if monster.move == "SPEAR_PIERCER":
            self._apply_player_power("Vulnerable", 2)
        if monster.move in {"SLAVER_RAKE", "BLUE_SLAVER_RAKE"}:
            self._apply_player_power("Weakened", 2 if self.ascension_level >= 17 else 1)
        if monster.move == "RED_SLAVER_SCRAPE":
            self._apply_player_power("Vulnerable", 2 if self.ascension_level >= 17 else 1)
        if starting_move == "TASKMASTER_SCOURING_WHIP":
            if self.ascension_level >= 18:
                monster.add_power("Strength", 1)
                wound_count = 3
            elif self.ascension_level >= 3:
                wound_count = 2
            else:
                wound_count = 1
            for _ in range(wound_count):
                self.discard_pile.append(make_card("Wound", uuid=self._new_uuid("Wound")))
        if starting_move == "CHOSEN_DEBILITATE":
            self._apply_player_power("Vulnerable", 2)
        if starting_move == "SPHERIC_GUARDIAN_ATTACK_DEBUFF":
            self._apply_player_power("Frail", 5)
        if starting_move == "MYSTIC_ATTACK":
            self._apply_player_power("Frail", 2)
        if starting_move == "SHELLED_FELL":
            self._apply_player_power("Frail", 2)
        if starting_move == "SNECKO_TAIL_WHIP":
            self._apply_player_power("Vulnerable", 2)
        if starting_move == "SHELLED_SUCK" and total_hp_damage > 0:
            monster.current_hp = min(monster.max_hp, monster.current_hp + total_hp_damage)
        if starting_move == "THE_GUARDIAN_FIERCE_BASH":
            _set_move(monster, "THE_GUARDIAN_VENT_STEAM")
            monster.ai_state["skip_end_round_roll"] = True
        elif starting_move == "THE_GUARDIAN_ROLL_ATTACK":
            _set_move(monster, "THE_GUARDIAN_TWIN_SLAM")
            monster.ai_state["skip_end_round_roll"] = True
        elif starting_move == "THE_GUARDIAN_TWIN_SLAM":
            monster.powers.pop("Sharp Hide", None)
            next_amount = int(monster.ai_state.get("mode_shift_amount", 30)) + 10
            monster.ai_state["mode_shift_amount"] = next_amount
            monster.add_power("Mode Shift", next_amount)
            _set_move(monster, "THE_GUARDIAN_WHIRLWIND")
            monster.ai_state["skip_end_round_roll"] = True
        elif starting_move == "THE_GUARDIAN_WHIRLWIND":
            _set_move(monster, "THE_GUARDIAN_CHARGING_UP")
            monster.ai_state["skip_end_round_roll"] = True
        if monster.move == "HEXAGHOST_DIVIDER":
            monster.ai_state["hex_counter"] = 0
            _set_move(monster, "HEXAGHOST_SEAR")
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
        elif monster.move == "HEXAGHOST_INFERNO":
            monster.ai_state["hex_counter"] = 0
            _set_move(monster, "HEXAGHOST_SEAR")
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
        elif monster.move == "HEXAGHOST_SEAR":
            counter = int(monster.ai_state.get("hex_counter", 0))
            if counter == 0:
                _set_move(monster, "HEXAGHOST_TACKLE")
            elif counter == 2:
                _set_move(monster, "HEXAGHOST_INFLAME")
            else:
                _set_move(monster, "HEXAGHOST_INFERNO")
            monster.ai_state["hex_counter"] = counter + 1
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
        elif monster.move == "HEXAGHOST_TACKLE":
            counter = int(monster.ai_state.get("hex_counter", 0))
            monster.ai_state["hex_counter"] = counter + 1
            _set_move(monster, "HEXAGHOST_SEAR")
            self.ai_rng.random(99)
            monster.ai_state["skip_end_round_roll"] = True
        if monster.move == "LOOTER_LUNGE":
            _set_move(monster, "LOOTER_SMOKE_BOMB")
            monster.ai_state["skip_end_round_roll"] = True
        elif monster.move == "LOOTER_MUG":
            if looter_turn_number == 1:
                _set_move(monster, "LOOTER_MUG")
            else:
                next_move = "LOOTER_SMOKE_BOMB" if self.ai_rng.random_boolean(0.5) else "LOOTER_LUNGE"
                _set_move(monster, next_move)
            monster.ai_state["skip_end_round_roll"] = True
        elif monster.move == "MUGGER_LUNGE":
            _set_move(monster, "MUGGER_SMOKE_BOMB")
            monster.ai_state["skip_end_round_roll"] = True
        elif monster.move == "MUGGER_MUG":
            if looter_turn_number == 2:
                next_move = "MUGGER_SMOKE_BOMB" if self.ai_rng.random_boolean(0.5) else "MUGGER_LUNGE"
                _set_move(monster, next_move)
            else:
                _set_move(monster, "MUGGER_MUG")
            monster.ai_state["skip_end_round_roll"] = True

    @staticmethod
    def _tick_powers_at_end_of_round(powers: dict[str, int], just_applied: set[str] | None = None) -> None:
        for key in ("Vulnerable", "Weakened", "Frail", "Intangible", "No Block"):
            if powers.get(key, 0) > 0:
                if just_applied is not None and key in just_applied:
                    continue
                powers[key] -= 1
                if powers[key] <= 0:
                    powers.pop(key, None)
        if just_applied is not None:
            just_applied.clear()

    def _check_outcome(self) -> None:
        if self.player.current_hp <= 0:
            self.outcome = "PLAYER_LOSS"
        elif not any(monster.alive for monster in self.monsters):
            self.outcome = "PLAYER_VICTORY"

    def to_spirecomm_state(self) -> dict[str, Any]:
        visible_monsters = list(self.monsters)
        combat_state = {
            "card_in_play": None,
            "cards_discarded_this_turn": self.cards_discarded_this_turn,
            "discard_pile": [card_to_spirecomm(card) for card in self.discard_pile],
            "draw_pile": [card_to_spirecomm(card) for card in self.draw_pile],
            "exhaust_pile": [card_to_spirecomm(card) for card in self.exhaust_pile],
            "hand": [self._card_to_spirecomm_in_combat(card) for card in self.hand],
            "limbo": [],
            "monsters": [self._monster_to_spirecomm(index, monster) for index, monster in enumerate(visible_monsters)],
            "player": {
                "block": self.player.block,
                "current_hp": self.player.current_hp,
                "energy": self.player.energy,
                "max_hp": self.player.max_hp,
                "orbs": [],
                "powers": self._player_powers_to_spirecomm(),
            },
            "turn": self.turn,
        }
        return {
            "act": self.act,
            "act_boss": self.act_boss,
            "ascension_level": self.ascension_level,
            "character": "IRONCLAD",
            "choice_available": False,
            "choice_list": [],
            "combat_state": combat_state,
            "commands": {
                "cancel": False,
                "end": self.outcome == "UNDECIDED",
                "play": self.outcome == "UNDECIDED",
                "potion": self.outcome == "UNDECIDED" and any(potion.can_use for potion in self.potions),
                "proceed": False,
            },
            "current_hp": self.player.current_hp,
            "deck": [card_to_spirecomm(card) for card in self.deck],
            "floor": self.floor,
            "gold": self.gold,
            "in_combat": self.outcome == "UNDECIDED",
            "max_hp": self.player.max_hp,
            "potions": potions_to_spirecomm(self.potions),
            "relics": self.relics,
            "room_phase": "COMBAT",
            "room_type": "MonsterRoom",
            "screen": "COMBAT",
            "screen_up": False,
            "seed": self.seed,
        }

    def _card_to_spirecomm_in_combat(self, card: CardInstance) -> dict[str, Any]:
        serialized = card_to_spirecomm(card, is_playable=self.playable(card))
        if not card.card_def.x_cost and card.card_def.card_type not in {"STATUS", "CURSE"}:
            serialized["cost_for_turn"] = self._card_energy_cost(card)
        return serialized

    def _player_powers_to_spirecomm(self) -> list[dict[str, Any]]:
        player = self.player
        powers: list[dict[str, Any]] = []
        for power_id, amount in (
            ("Strength", player.power("Strength")),
            ("Dexterity", player.power("Dexterity")),
            ("Vulnerable", player.power("Vulnerable")),
            ("Weakened", player.power("Weakened")),
            ("Frail", player.power("Frail")),
            ("Artifact", player.power("Artifact")),
            ("Metallicize", player.power("Metallicize")),
            ("Rage", player.power("Rage")),
            ("Barricade", 1 if player.power("Barricade") > 0 else 0),
            ("Demon Form", player.power("Demon Form")),
            ("Berserk", player.power("Berserk")),
            ("Flame Barrier", player.power("Flame Barrier")),
            ("Thorns", player.power("Thorns")),
            ("Plated Armor", player.power("Plated Armor")),
            ("No Draw", 1 if player.power("No Draw") > 0 else 0),
        ):
            if amount:
                powers.append(_serialize_named_power(power_id, amount))
        return powers

    def _monster_powers_to_spirecomm(self, monster: MonsterState) -> list[dict[str, Any]]:
        powers: list[dict[str, Any]] = []
        for power_id, amount in (
            ("Strength", monster.power("Strength")),
            ("Vulnerable", monster.power("Vulnerable")),
            ("Weakened", monster.power("Weakened")),
            ("Artifact", monster.power("Artifact")),
            ("Metallicize", monster.power("Metallicize")),
            ("Ritual", monster.power("Ritual")),
            ("Angry", monster.power("Angry")),
            ("Sharp Hide", monster.power("Sharp Hide")),
            ("Thorns", monster.power("Thorns")),
            ("Curl Up", monster.power("Curl Up")),
            ("Mode Shift", monster.power("Mode Shift")),
            ("Regenerate", monster.power("Regenerate")),
            ("Flight", monster.power("Flight")),
            ("Malleable", monster.power("Malleable")),
            ("Plated Armor", monster.power("Plated Armor")),
        ):
            if amount:
                powers.append(_serialize_named_power(power_id, amount))
        return powers

    def _monster_to_spirecomm(self, index: int, monster: MonsterState) -> dict[str, Any]:
        adjusted_damage = monster_adjusted_damage(monster, self.player, vulnerable_multiplier=self._monster_vulnerable_multiplier())
        runic_dome = self._has_relic("Runic Dome")
        if runic_dome:
            adjusted_damage = 0
        return {
            "block": monster.block,
            "current_hp": monster.current_hp,
            "half_dead": monster.half_dead,
            "intent": "UNKNOWN" if runic_dome else monster.intent,
            "is_gone": monster.is_gone,
            "last_move_id": monster.move_history[0] if monster.move_history else None,
            "max_hp": monster.max_hp,
            "monster_id": _spirecomm_monster_id(monster.monster_id),
            "monster_index": index,
            "move_adjusted_damage": adjusted_damage,
            "move_base_damage": 0 if runic_dome else monster.move_base_damage,
            "move_hits": 0 if runic_dome else monster.move_hits,
            "move_id": None if runic_dome else _serialize_move_name(monster.move),
            "move_name": None if runic_dome else _serialize_move_name(monster.move),
            "name": monster.name,
            "powers": self._monster_powers_to_spirecomm(monster),
            "second_last_move_id": monster.move_history[1] if len(monster.move_history) > 1 else None,
        }


@dataclass
class NativeRunEnv:
    seed: int
    ascension_level: int = 0
    ironclad_unlock_level: int = 5
    ironclad_relic_unlock_level: int = 5
    enable_neow: bool = False
    enable_act4_keys: bool = True
    start_on_map: bool = False
    floor: int = 1
    act: int = 1
    act_boss: str = "Hexaghost"
    rng: StsRandom = field(init=False)
    randoms: NativeRandomStreams = field(init=False)
    deck: list[CardInstance] = field(default_factory=starter_deck)
    player: PlayerState = field(default_factory=PlayerState)
    relics: list[dict[str, Any]] = field(default_factory=lambda: [make_relic("Burning Blood")])
    potions: list[PotionInstance] = field(default_factory=empty_potion_slots)
    gold: int = 99
    phase: str = "COMBAT"
    reward_cards: list[CardInstance] = field(default_factory=list)
    reward_card_bundles: list[list[CardInstance]] = field(default_factory=list)
    reward_close_required: bool = False
    reward_gold: int = 0
    reward_gold_piles: list[int] = field(default_factory=list)
    reward_emerald_key: bool = False
    potion_chance_counter: int = 0
    reward_relics: list[dict[str, Any]] = field(default_factory=list)
    reward_potions: list[PotionInstance] = field(default_factory=list)
    boss_relic_options: list[dict[str, Any]] = field(default_factory=list)
    map_options: list[dict[str, Any]] = field(default_factory=list)
    treasure_options: list[dict[str, Any]] = field(default_factory=list)
    chest_options: list[dict[str, Any]] = field(default_factory=list)
    chest_gold_amount: int = 0
    chest_have_gold: bool = False
    chest_relic_tier: str = "COMMON"
    chest_size: str = "SMALL"
    shop_items: list[dict[str, Any]] = field(default_factory=list)
    campfire_options: list[dict[str, Any]] = field(default_factory=list)
    event_options: list[dict[str, Any]] = field(default_factory=list)
    neow_options: list[dict[str, Any]] = field(default_factory=list)
    card_select_options: list[dict[str, Any]] = field(default_factory=list)
    map_graph: dict[str, dict[str, Any]] = field(default_factory=dict)
    map_layers: dict[int, list[str]] = field(default_factory=dict)
    current_map_node_id: str | None = None
    current_node_symbol: str = "M"
    keys: set[str] = field(default_factory=set)
    a20_second_boss_done: bool = False
    pending_event_relic_id: str | None = None
    pending_event_gold: int = 0
    card_rarity_factor: int = 5
    locked_card_ids: set[str] = field(init=False)
    monster_chance: float = _f32(0.10)
    shop_chance: float = _f32(0.03)
    treasure_chance: float = _f32(0.02)
    shop_remove_count: int = 0
    event_list: list[str] = field(default_factory=list)
    shrine_list: list[str] = field(default_factory=list)
    special_one_time_event_list: list[str] = field(default_factory=list)
    current_event_id: str | None = None
    event_state: dict[str, Any] = field(default_factory=dict)
    note_for_yourself_card_id: str = "Iron Wave"
    reward_context: str | None = None
    card_select_context: str | None = None
    card_select_count: int = 0
    card_select_available_indexes: list[int] = field(default_factory=list)
    card_select_selected_indexes: list[int] = field(default_factory=list)
    relic_pools: dict[str, list[str]] = field(default_factory=dict)
    monster_list: list[str] = field(default_factory=list)
    monster_list_offset: int = 0
    elite_monster_list: list[str] = field(default_factory=list)
    elite_monster_list_offset: int = 0
    second_act_boss: str | None = None
    combat: NativeCombatEnv = field(init=False)

    def __post_init__(self) -> None:
        self.randoms = NativeRandomStreams(self.seed)
        self.rng = self.randoms.misc
        self.locked_card_ids = ironclad_locked_card_ids(self.ironclad_unlock_level)
        self._act_bosses = {4: "Corrupt Heart"}
        self._generate_monster_schedules_for_act(1)
        self.relic_pools = init_ironclad_relic_pools(
            self.randoms.relic,
            locked_relic_ids=ironclad_locked_relic_ids(self.ironclad_relic_unlock_level),
        )
        self._generate_neow_options()
        self.act_boss = self._act_bosses.get(self.act, self.act_boss)
        if self.ascension_level >= 10 and not any(card.card_id == "AscendersBane" for card in self.deck):
            self.deck.append(make_card("AscendersBane", uuid="starter-ascenders-bane"))
        if self.ascension_level >= 6:
            self.player.current_hp = max(1, self.player.current_hp - max(1, self.player.max_hp // 10))
        self.special_one_time_event_list = self._one_time_event_pool()
        self._reset_act_random_room_state()
        self._generate_act_map(1)
        if self.enable_neow:
            self._enter_neow()
        elif self.start_on_map:
            self.floor = 0
            self.phase = "MAP"
            self._enter_map()
        else:
            self._start_combat()

    def _enter_neow(self) -> None:
        self.phase = "NEOW"
        self.neow_options = list(self.neow_options)

    def _make_neow_option(self, index: int, bonus: str, drawback: str) -> dict[str, Any]:
        return {
            "kind": "neow",
            "name": f"OPTION_{index}",
            "label": f"OPTION_{index}",
            "choice_index": index,
            "bonus": bonus,
            "drawback": drawback,
            "bonus_text": NEOW_BONUS_LABELS[bonus],
            "drawback_text": NEOW_DRAWBACK_LABELS[drawback],
        }

    def _generate_neow_options(self) -> None:
        rng = self.randoms.neow
        options: list[dict[str, Any]] = []

        first_bonus = [
            "THREE_CARDS",
            "ONE_RANDOM_RARE_CARD",
            "REMOVE_CARD",
            "UPGRADE_CARD",
            "TRANSFORM_CARD",
            "RANDOM_COLORLESS",
        ][int(rng.random(0, 5))]
        options.append(self._make_neow_option(0, first_bonus, "NONE"))

        second_bonus = [
            "THREE_SMALL_POTIONS",
            "RANDOM_COMMON_RELIC",
            "TEN_PERCENT_HP_BONUS",
            "THREE_ENEMY_KILL",
            "HUNDRED_GOLD",
        ][int(rng.random(0, 4))]
        options.append(self._make_neow_option(1, second_bonus, "NONE"))

        drawback = [
            "TEN_PERCENT_HP_LOSS",
            "NO_GOLD",
            "CURSE",
            "PERCENT_DAMAGE",
        ][int(rng.random(0, 3))]
        if drawback in NEOW_MID_TIER_BY_DRAWBACK:
            third_bonus = NEOW_MID_TIER_BY_DRAWBACK[drawback][int(rng.random(0, 5))]
        else:
            third_bonus = [
                "RANDOM_COLORLESS_2",
                "REMOVE_TWO",
                "ONE_RARE_RELIC",
                "THREE_RARE_CARDS",
                "TWO_FIFTY_GOLD",
                "TRANSFORM_TWO_CARDS",
                "TWENTY_PERCENT_HP_BONUS",
            ][int(rng.random(0, 6))]
        options.append(self._make_neow_option(2, third_bonus, drawback))
        options.append(self._make_neow_option(3, "BOSS_RELIC", "LOSE_STARTER_RELIC"))
        rng.random(0, 0)
        self.neow_options = options

    def _generate_monster_schedules_for_act(self, act: int) -> None:
        if act == 4:
            self.monster_list = []
            self.monster_list_offset = 0
            self.elite_monster_list = ["ShieldAndSpear"]
            self.elite_monster_list_offset = 0
            self._act_bosses[4] = "CorruptHeart"
            return
        monsters, elites, boss, second_boss = generate_monster_schedules(self.randoms.monster, act, self.ascension_level)
        self.monster_list = monsters
        self.monster_list_offset = 0
        self.elite_monster_list = elites
        self.elite_monster_list_offset = 0
        boss_name = {
            "TheGuardian": "The Guardian",
            "Hexaghost": "Hexaghost",
            "SlimeBoss": "Slime Boss",
            "BronzeAutomaton": "Bronze Automaton",
            "TheCollector": "The Collector",
            "TheChamp": "The Champ",
            "AwakenedOne": "Awakened One",
            "TimeEater": "Time Eater",
            "DonuDeca": "Donu and Deca",
        }.get(boss, boss)
        self._act_bosses[act] = boss_name
        if second_boss:
            self.second_act_boss = {
                "AwakenedOne": "Awakened One",
                "TimeEater": "Time Eater",
                "DonuDeca": "Donu and Deca",
            }.get(second_boss, second_boss)

    def _next_monster_encounter(self) -> str:
        if self.monster_list_offset >= len(self.monster_list):
            self.monster_list = generate_strong_monster_schedule(self.randoms.monster, self.act)
            self.monster_list_offset = 0
        encounter = self.monster_list[self.monster_list_offset]
        self.monster_list_offset += 1
        return encounter

    def _next_elite_encounter(self) -> str:
        if self.elite_monster_list_offset >= len(self.elite_monster_list):
            self.elite_monster_list = generate_elite_schedule(self.randoms.monster, self.act)
            self.elite_monster_list_offset = 0
        encounter = self.elite_monster_list[self.elite_monster_list_offset]
        self.elite_monster_list_offset += 1
        return encounter

    def _start_combat(self, *, elite: bool = False) -> None:
        self.phase = "COMBAT"
        if elite and self.current_node_symbol not in {"E_GREEN", "ACT4_ELITE"}:
            self.current_node_symbol = "E"
        scheduled_encounter: list[str] | str | None = None
        if self.floor == 53:
            scheduled_encounter = ["SpireShield", "SpireSpear"]
        elif self.floor == 54:
            scheduled_encounter = ["CorruptHeart"]
        elif self.floor in {16, 33, 50}:
            scheduled_encounter = {
                "Hexaghost": "Hexaghost",
                "Slime Boss": "SlimeBoss",
                "The Guardian": "TheGuardian",
                "The Champ": "TheChamp",
                "The Collector": "TheCollector",
                "Bronze Automaton": "BronzeAutomaton",
                "Awakened One": "AwakenedOne",
                "Time Eater": "TimeEater",
                "Donu and Deca": "DonuDeca",
            }.get(self.act_boss)
        elif elite:
            scheduled_encounter = self._next_elite_encounter()
        else:
            scheduled_encounter = self._next_monster_encounter()
        self.player.powers = {}
        self.player.block = 0
        self.combat = NativeCombatEnv(
            seed=self.seed,
            ascension_level=self.ascension_level,
            floor=self.floor,
            act=self.act,
            act_boss=self.act_boss,
            elite=elite,
            external_misc_rng=self.randoms.misc,
            scheduled_encounter=scheduled_encounter,
            player=self.player,
            deck=[clone_card(card) for card in self.deck],
            relics=list(self.relics),
            potions=list(self.potions),
            gold=self.gold,
            locked_card_ids=set(self.locked_card_ids),
        )
        self._apply_burning_elite_buff()

    def _apply_burning_elite_buff(self) -> None:
        if self.current_node_symbol != "E_GREEN" or not self.current_map_node_id or self.combat is None:
            return
        node = self.map_graph.get(self.current_map_node_id, {})
        buff_type = node.get("burning_elite_buff")
        if buff_type is None:
            return
        for monster in self.combat.monsters:
            if buff_type == 0:
                monster.add_power("Strength", self.act)
            elif buff_type == 1:
                increase = _sts_round(float(monster.max_hp) * 0.25)
                monster.max_hp += increase
                monster.current_hp += increase
            elif buff_type == 2:
                monster.add_power("Metallicize", self.act * 2 + 2)
            elif buff_type == 3:
                monster.add_power("Regenerate", self.act * 2 + 1)

    def _start_event_boss_combat(self, *, act_boss: str | None = None) -> None:
        self.phase = "COMBAT"
        self.current_node_symbol = "E"
        boss = act_boss or self.randoms.monster.choice(["Hexaghost", "Slime Boss", "The Guardian"])
        self.combat = NativeCombatEnv(
            seed=self.seed,
            ascension_level=self.ascension_level,
            floor=16,
            act=1,
            act_boss=boss,
            elite=False,
            external_misc_rng=self.randoms.misc,
            player=self.player,
            deck=[clone_card(card) for card in self.deck],
            relics=list(self.relics),
            potions=list(self.potions),
            gold=self.gold,
            locked_card_ids=set(self.locked_card_ids),
        )

    def _start_event_combat(
        self,
        monster_ids: list[str],
        *,
        relic_id: str | None = None,
        gold_gain: int = 0,
        elite: bool = False,
    ) -> None:
        self.phase = "COMBAT"
        self.current_node_symbol = "EVENT_COMBAT"
        self.pending_event_relic_id = relic_id
        self.pending_event_gold = max(0, int(gold_gain))
        self.combat = NativeCombatEnv(
            seed=self.seed,
            ascension_level=self.ascension_level,
            floor=self.floor,
            act=self.act,
            act_boss=self.act_boss,
            elite=elite,
            external_misc_rng=self.randoms.misc,
            scheduled_encounter=list(monster_ids),
            player=self.player,
            deck=[clone_card(card) for card in self.deck],
            relics=list(self.relics),
            potions=list(self.potions),
            gold=self.gold,
            locked_card_ids=set(self.locked_card_ids),
        )

    def _has_relic(self, relic_id: str) -> bool:
        return any(relic.get("relic_id") == relic_id for relic in self.relics)

    def _relic(self, relic_id: str) -> dict[str, Any] | None:
        return next((relic for relic in self.relics if relic.get("relic_id") == relic_id), None)

    def _owned_relic_ids(self) -> set[str]:
        return {str(item.get("relic_id")) for item in self.relics}

    def _roll_relic_of_tier(self, tier: str, *, shop_room: bool = False, from_front: bool = True) -> dict[str, Any]:
        return draw_relic_from_pool(
            self.relic_pools,
            tier,
            owned=self._owned_relic_ids(),
            floor=self.floor,
            shop_room=shop_room,
            from_front=from_front,
            deck=self.deck,
        )

    def _roll_screenless_relic_of_tier(self, tier: str, *, shop_room: bool = False, from_front: bool = True) -> dict[str, Any]:
        blocked = {"Bottled Flame", "Bottled Lightning", "Bottled Tornado", "Whetstone"}
        while True:
            relic = self._roll_relic_of_tier(tier, shop_room=shop_room, from_front=from_front)
            if str(relic.get("relic_id") or "") not in blocked:
                return relic

    def _roll_relic_tier_for_act(self, act: int) -> str:
        common_chance = 0 if act == 4 else 50
        uncommon_chance = 100 if act == 4 else 33
        roll = int(self.randoms.relic.random(99))
        if roll < common_chance:
            return "COMMON"
        if roll < common_chance + uncommon_chance:
            return "UNCOMMON"
        return "RARE"

    def _roll_relic(self, *, shop_room: bool = False, from_front: bool = True, elite: bool = False) -> dict[str, Any]:
        if elite:
            roll = int(self.randoms.relic.random(99))
            tier = "COMMON" if roll < 50 else "RARE" if roll > 82 else "UNCOMMON"
        else:
            tier = self._roll_relic_tier_for_act(self.act)
        return self._roll_relic_of_tier(tier, shop_room=shop_room, from_front=from_front)

    def _roll_boss_relics(self, count: int = 3) -> list[dict[str, Any]]:
        return [self._roll_relic_of_tier("BOSS") for _ in range(count)]

    def _ironclad_card_pool(self, *, card_type: str | None = None, rarity: str | None = None):
        return ironclad_card_pool(card_type=card_type, rarity=rarity, exclude_ids=self.locked_card_ids)

    def _random_class_card_of_rarity(self, rarity: str) -> CardInstance:
        pool = self._ironclad_card_pool(rarity=rarity)
        if not pool:
            pool = self._ironclad_card_pool()
        chosen = pool[int(self.randoms.card.random(len(pool) - 1))]
        return make_card(chosen.card_id, uuid=f"rarity-{self.floor}-{chosen.card_id}")

    def _consume_match_and_keep_rng(self) -> list[CardInstance]:
        cards = [
            self._random_class_card_of_rarity("RARE"),
            self._random_class_card_of_rarity("UNCOMMON"),
            self._random_class_card_of_rarity("COMMON"),
        ]
        # returnColorlessCard(UNCOMMON) shuffles the colorless pool using the
        # shuffle RNG, not cardRng.
        self.randoms.shuffle.random_long()
        colorless_pool = [
            CARD_LIBRARY[card_id] for card_id in COLORLESS_CARD_ID_ORDER
            if CARD_LIBRARY[card_id].rarity == "UNCOMMON"
        ]
        if colorless_pool:
            cards.append(make_card(colorless_pool[0].card_id, uuid=f"match-colorless-{self.floor}"))
        curse_pool = [
            card_id for card_id, card_def in CARD_LIBRARY.items()
            if card_def.card_type == "CURSE" and card_id not in {"AscendersBane", "CurseOfTheBell"}
        ]
        if curse_pool:
            self.randoms.card.random(len(curse_pool) - 1)
        cards.append(make_card("Strike_R", uuid=f"match-starter-{self.floor}"))
        self.randoms.misc.random_long()
        return cards

    def _random_curse_id(self) -> str:
        return self.randoms.card.choice([
            "Regret",
            "Injury",
            "Shame",
            "Parasite",
            "Normality",
            "Doubt",
            "Writhe",
            "Pain",
            "Decay",
            "Clumsy",
        ])

    def _random_class_card_of_rarity_from_rng(self, rng: StsRandom, rarity: str) -> CardInstance:
        pool = self._ironclad_card_pool(rarity=rarity) or self._ironclad_card_pool()
        chosen = pool[int(rng.random(len(pool) - 1))]
        return CardInstance(chosen, uuid=f"random-{rarity.lower()}-{chosen.card_id}-{self.floor}")

    def _transformed_card_from_rng(self, rng: StsRandom, exclude_card_id: str) -> CardInstance:
        exclude_in_pool = (
            exclude_card_id in TRANSFORM_CARD_POOL_IRONCLAD
            and CARD_LIBRARY[exclude_card_id].rarity != "BASIC"
        )
        if exclude_in_pool:
            index = int(rng.random(len(TRANSFORM_CARD_POOL_IRONCLAD) - 2))
            chosen = TRANSFORM_CARD_POOL_IRONCLAD[index + 1] if TRANSFORM_CARD_POOL_IRONCLAD[index] == exclude_card_id else TRANSFORM_CARD_POOL_IRONCLAD[index]
        else:
            chosen = TRANSFORM_CARD_POOL_IRONCLAD[int(rng.random(len(TRANSFORM_CARD_POOL_IRONCLAD) - 1))]
        return make_card(chosen, uuid=f"transform-{chosen}-{self.floor}")

    def _neow_card_reward(self, *, rare_only: bool = False) -> list[CardInstance]:
        reward: list[CardInstance] = []
        seen: set[str] = set()
        for _ in range(3):
            rarity = "UNCOMMON" if self.randoms.neow.random_boolean(0.33) else "COMMON"
            if rare_only:
                rarity = "RARE"
            card = self._random_class_card_of_rarity_from_rng(self.randoms.neow, rarity)
            while card.card_id in seen:
                card = self._random_class_card_of_rarity_from_rng(self.randoms.neow, rarity)
            seen.add(card.card_id)
            card.uuid = f"neow-reward-{card.card_id}-{len(reward)}"
            reward.append(card)
        return reward

    def _neow_colorless_card_reward(self, *, rare_only: bool = False) -> list[CardInstance]:
        reward: list[CardInstance] = []
        seen: set[str] = set()
        for _ in range(3):
            rarity = "UNCOMMON" if self.randoms.neow.random_boolean(0.33) else "COMMON"
            if rare_only:
                rarity = "RARE"
            elif rarity == "COMMON":
                rarity = "UNCOMMON"
            pool = [
                CARD_LIBRARY[card_id] for card_id in COLORLESS_CARD_ID_ORDER
                if CARD_LIBRARY[card_id].rarity == rarity
            ]
            chosen = pool[int(self.randoms.card.random(len(pool) - 1))]
            while chosen.card_id in seen:
                chosen = pool[int(self.randoms.card.random(len(pool) - 1))]
            seen.add(chosen.card_id)
            reward.append(CardInstance(chosen, uuid=f"neow-colorless-{chosen.card_id}-{len(reward)}"))
        return reward

    def _open_neow_card_reward(self, *, rare_only: bool = False, colorless: bool = False) -> None:
        self.phase = "CARD_REWARD"
        self.reward_context = "NEOW"
        self.reward_close_required = False
        self.reward_card_bundles = [self._neow_colorless_card_reward(rare_only=rare_only) if colorless else self._neow_card_reward(rare_only=rare_only)]
        self._refresh_reward_cards()

    def _open_card_select(self, context: str, count: int) -> None:
        self.phase = "CARD_SELECT"
        self.card_select_context = context
        self.card_select_count = count
        bottled_blocked_contexts = {
            "NEOW_REMOVE",
            "NEOW_TRANSFORM",
            "EVENT_REMOVE",
            "EVENT_TRANSFORM",
            "BONFIRE_SPIRITS",
        }
        bottled_indexes = self._bottled_deck_indexes() if context in bottled_blocked_contexts else set()
        upgrade_contexts = {"NEOW_UPGRADE", "EVENT_UPGRADE"}
        if context == "TRANSFORM_UPGRADE" and self.card_select_available_indexes:
            selectable_indexes = [
                index
                for index in self.card_select_available_indexes
                if 0 <= index < len(self.deck)
            ]
        else:
            selectable_indexes = [
                index
                for index, card in enumerate(self.deck)
                if index not in bottled_indexes and (context not in upgrade_contexts or _card_can_upgrade(card))
            ]
            self.card_select_available_indexes = list(selectable_indexes)
            self.card_select_selected_indexes = []
        self.card_select_options = [
            {
                "kind": "card_select",
                "name": self.deck[index].name,
                "select_type": context,
                "label": self.deck[index].name,
                "choice_index": index,
                "target_index": index,
                "card": card_to_spirecomm(self.deck[index]),
            }
            for index in selectable_indexes
        ]

    def _open_library_card_select(self) -> None:
        self.phase = "CARD_SELECT"
        self.card_select_context = "LIBRARY_OBTAIN"
        self.card_select_count = 1
        seen: set[str] = set()
        cards: list[CardInstance | None] = [None] * 20
        for index in range(19, -1, -1):
            rarity = self._roll_card_rarity(room="?")
            pool = self._ironclad_card_pool(rarity=rarity) or self._ironclad_card_pool()
            chosen = self.randoms.card.choice(pool)
            while chosen.card_id in seen:
                chosen = self.randoms.card.choice(pool)
            seen.add(chosen.card_id)
            cards[index] = CardInstance(chosen, uuid=f"library-{self.floor}-{index}-{chosen.card_id}")
        self.card_select_generated_cards = [card for card in cards if card is not None]
        self.card_select_options = [
            {
                "kind": "card_select",
                "name": card.name,
                "select_type": "OBTAIN",
                "label": card.name,
                "choice_index": index,
                "target_index": -1,
                "deck_index": -1,
                "card": card_to_spirecomm(card),
            }
            for index, card in enumerate(self.card_select_generated_cards)
        ]

    def _bottled_deck_indexes(self) -> set[int]:
        bottled_uuids = {
            str(relic.get("card_uuid"))
            for relic in self.relics
            if relic.get("relic_id") in {"Bottled Flame", "Bottled Lightning", "Bottled Tornado"} and relic.get("card_uuid")
        }
        if bottled_uuids:
            return {index for index, card in enumerate(self.deck) if card.uuid in bottled_uuids}
        bottled_ids = {
            str(relic.get("card_id"))
            for relic in self.relics
            if relic.get("relic_id") in {"Bottled Flame", "Bottled Lightning", "Bottled Tornado"} and relic.get("card_id")
        }
        if not bottled_ids:
            return set()
        return {index for index, card in enumerate(self.deck) if card.card_id in bottled_ids}

    def _open_bottle_card_select(self, relic_id: str) -> bool:
        type_by_relic = {
            "Bottled Flame": "ATTACK",
            "Bottled Lightning": "SKILL",
            "Bottled Tornado": "POWER",
        }
        card_type = type_by_relic.get(relic_id)
        if card_type is None:
            return False
        self.phase = "CARD_SELECT"
        self.card_select_context = "BOTTLE_REWARD"
        self.pending_bottle_relic_id = relic_id
        self.card_select_count = 1
        self.card_select_options = []
        for index, card in enumerate(self.deck):
            if card.card_def.card_type != card_type:
                continue
            self.card_select_options.append({
                "kind": "card_select",
                "name": "BOTTLE",
                "select_type": "BOTTLE",
                "label": card.name,
                "choice_index": index,
                "target_index": index,
                "deck_index": index,
                "card": card_to_spirecomm(card),
            })
        return bool(self.card_select_options)

    def _open_neow_card_select(self, context: str, count: int) -> None:
        self._open_card_select(context, count)

    def _apply_neow_drawback(self, drawback: str) -> None:
        if drawback == "TEN_PERCENT_HP_LOSS":
            self.player.max_hp = max(1, int(self.player.max_hp * 0.9))
            self.player.current_hp = min(self.player.current_hp, self.player.max_hp)
        elif drawback == "NO_GOLD":
            self.gold = 0
        elif drawback == "PERCENT_DAMAGE":
            self.player.current_hp = max(1, (self.player.current_hp // 10) * 7)
        elif drawback == "LOSE_STARTER_RELIC":
            self.relics = [relic for relic in self.relics if relic.get("relic_id") != "Burning Blood"]

    def _complete_neow(self) -> dict[str, Any]:
        self.neow_options = []
        self.floor = 0
        self.phase = "MAP"
        self._enter_map()
        return self.state()

    def _apply_neow_bonus(self, bonus: str) -> dict[str, Any]:
        if bonus == "THREE_CARDS":
            self._open_neow_card_reward(rare_only=False, colorless=False)
            return self.state()
        if bonus == "ONE_RANDOM_RARE_CARD":
            chosen = self._random_class_card_of_rarity_from_rng(self.randoms.neow, "RARE")
            self._add_card_to_deck(chosen.card_id, uuid=f"neow-rare-{chosen.card_id}")
            return self._complete_neow()
        if bonus == "REMOVE_CARD":
            self._open_neow_card_select("NEOW_REMOVE", 1)
            return self.state()
        if bonus == "UPGRADE_CARD":
            self._open_neow_card_select("NEOW_UPGRADE", 1)
            return self.state()
        if bonus == "TRANSFORM_CARD":
            self._open_neow_card_select("NEOW_TRANSFORM", 1)
            return self.state()
        if bonus == "RANDOM_COLORLESS":
            self._open_neow_card_reward(rare_only=False, colorless=True)
            return self.state()
        if bonus == "THREE_SMALL_POTIONS":
            self._add_random_potion_reward(count=3)
            return self._complete_neow()
        if bonus == "RANDOM_COMMON_RELIC":
            self._obtain_relic(self._roll_relic_of_tier("COMMON"))
            return self._complete_neow()
        if bonus == "TEN_PERCENT_HP_BONUS":
            hp_gain = int(self.player.max_hp * 0.1)
            self.player.max_hp += hp_gain
            self.player.current_hp += hp_gain
            return self._complete_neow()
        if bonus == "THREE_ENEMY_KILL":
            self._obtain_relic(make_relic("Neow's Lament"))
            return self._complete_neow()
        if bonus == "HUNDRED_GOLD":
            self._gain_gold(100)
            return self._complete_neow()
        if bonus == "RANDOM_COLORLESS_2":
            self._open_neow_card_reward(rare_only=True, colorless=True)
            return self.state()
        if bonus == "REMOVE_TWO":
            self._open_neow_card_select("NEOW_REMOVE", 2)
            return self.state()
        if bonus == "ONE_RARE_RELIC":
            self._obtain_relic(self._roll_relic_of_tier("RARE"))
            return self._complete_neow()
        if bonus == "THREE_RARE_CARDS":
            self._open_neow_card_reward(rare_only=True, colorless=False)
            return self.state()
        if bonus == "TWO_FIFTY_GOLD":
            self._gain_gold(250)
            return self._complete_neow()
        if bonus == "TRANSFORM_TWO_CARDS":
            self._open_neow_card_select("NEOW_TRANSFORM", 2)
            return self.state()
        if bonus == "TWENTY_PERCENT_HP_BONUS":
            hp_gain = int(self.player.max_hp * 0.2)
            self.player.max_hp += hp_gain
            self.player.current_hp += hp_gain
            return self._complete_neow()
        if bonus == "BOSS_RELIC":
            self.relics = [relic for relic in self.relics if relic.get("relic_id") != "Burning Blood"]
            self._obtain_relic(self._roll_boss_relics(count=1)[0])
            return self._complete_neow()
        return self._complete_neow()

    def _roll_card_rarity(self, room: str | None = None) -> str:
        room = room or self.current_node_symbol
        # The real game still consumes the card RNG rarity roll for boss
        # rewards before forcing the outcome to rare. Keeping that counter
        # aligned matters for every later card reward in the run.
        roll = int(self.randoms.card.random(99)) + self.card_rarity_factor
        if room == "BOSS":
            return "RARE"
        rare_chance = 10 if room in {"E", "E_GREEN", "ACT4_ELITE"} else 3
        uncommon_chance = 40 if room in {"E", "E_GREEN", "ACT4_ELITE"} else 37
        if room != "R" and self._has_relic("N'loth's Gift"):
            rare_chance *= 3
        if roll < rare_chance:
            return "RARE"
        if roll < rare_chance + uncommon_chance:
            return "UNCOMMON"
        return "COMMON"

    def _roll_shop_card_rarity(self) -> str:
        roll = int(self.randoms.card.random(99)) + self.card_rarity_factor
        if roll < 9:
            return "RARE"
        if roll >= 46:
            return "COMMON"
        return "UNCOMMON"

    def _upgraded_card_chance(self) -> float:
        if self.act < 2:
            return 0.0
        if self.act == 2:
            return 0.25 if self.ascension_level < 12 else 0.125
        return 0.50 if self.ascension_level < 12 else 0.25

    def _roll_card_reward(self, count: int = 3, *, room: str | None = None) -> list[CardInstance]:
        options: list[CardInstance] = []
        seen: set[str] = set()
        rarities: list[str] = []
        for _ in range(max(0, int(count))):
            rarity = self._roll_card_rarity(room)
            rarities.append(rarity)
            if rarity == "COMMON":
                self.card_rarity_factor = max(self.card_rarity_factor - 1, -40)
            elif rarity == "RARE":
                self.card_rarity_factor = 5
            rarity_pool = self._ironclad_card_pool(rarity=rarity) or self._ironclad_card_pool()
            chosen = self.randoms.card.choice(rarity_pool)
            while chosen.card_id in seen:
                chosen = self.randoms.card.choice(rarity_pool)
            seen.add(chosen.card_id)
            options.append(CardInstance(chosen, uuid=f"reward-{chosen.card_id}-{len(options)}"))
        upgraded_chance = self._upgraded_card_chance()
        for card, rarity in zip(options, rarities):
            if rarity != "RARE" and self.randoms.card.random_boolean(upgraded_chance):
                card.upgrades = max(card.upgrades, 1)
        return options

    def _refresh_reward_cards(self) -> None:
        self.reward_cards = [card for bundle in self.reward_card_bundles for card in bundle]

    def _roll_shop_cards(self) -> tuple[list[tuple[CardInstance, str]], list[tuple[CardInstance, str]]]:
        def get_card_from_pool(card_type: str, rarity: str) -> CardInstance:
            pool = ironclad_type_rarity_card_pool(card_type, rarity, exclude_ids=self.locked_card_ids)
            if not pool:
                pool = self._ironclad_card_pool(card_type=card_type, rarity=rarity)
            chosen = pool[int(self.randoms.card.random(len(pool) - 1))]
            return CardInstance(chosen, uuid=f"shop-{chosen.card_id}-{self.floor}")

        def assign_random_card_excluding(card_type: str, exclude_id: str) -> tuple[CardInstance, str]:
            while True:
                rarity = self._roll_shop_card_rarity()
                card = get_card_from_pool(card_type, rarity)
                if card.card_id != exclude_id:
                    return card, rarity

        attack_a_rarity = self._roll_shop_card_rarity()
        attack_a = get_card_from_pool("ATTACK", attack_a_rarity)
        attack_b, attack_b_rarity = assign_random_card_excluding("ATTACK", attack_a.card_id)
        skill_a_rarity = self._roll_shop_card_rarity()
        skill_a = get_card_from_pool("SKILL", skill_a_rarity)
        skill_b, skill_b_rarity = assign_random_card_excluding("SKILL", skill_a.card_id)
        power_rarity = self._roll_shop_card_rarity()
        if power_rarity == "COMMON":
            power_rarity = "UNCOMMON"
        power_card = get_card_from_pool("POWER", power_rarity)

        colorless_cards: list[tuple[CardInstance, str]] = []
        for rarity in ("UNCOMMON", "RARE"):
            colorless_pool = [
                CARD_LIBRARY[card_id]
                for card_id in COLORLESS_CARD_ID_ORDER
                if CARD_LIBRARY[card_id].rarity == rarity
            ]
            if not colorless_pool and rarity == "UNCOMMON":
                colorless_pool = [
                    CARD_LIBRARY[card_id]
                    for card_id in COLORLESS_CARD_ID_ORDER
                    if CARD_LIBRARY[card_id].rarity == "COMMON"
                ]
            chosen = self.randoms.card.choice(colorless_pool)
            colorless_cards.append((CardInstance(chosen, uuid=f"shop-colorless-{chosen.card_id}"), rarity))
        return [
            (attack_a, attack_a_rarity),
            (attack_b, attack_b_rarity),
            (skill_a, skill_a_rarity),
            (skill_b, skill_b_rarity),
            (power_card, power_rarity),
        ], colorless_cards

    def _add_curse_to_deck(self, card_id: str | None = None, *, uuid: str | None = None) -> bool:
        card_id = card_id or self._random_curse_id()
        omamori = self._relic("Omamori")
        if omamori is not None:
            counter = int(omamori.get("counter", -1))
            counter = 2 if counter < 0 else counter
            if counter > 0:
                omamori["counter"] = counter - 1
                return False
        self.deck.append(make_card(card_id, uuid=uuid or f"curse-{self.floor}-{card_id}"))
        if self._has_relic("Darkstone Periapt"):
            self.player.max_hp += 6
            self.player.current_hp += 6
        return True

    def _apply_shop_discount(self, price: int, *, include_ascension: bool = True) -> int:
        discounted = int(price)
        if include_ascension and self.ascension_level >= 16:
            discounted = max(0, round(discounted * 0.8))
        if self._has_relic("The Courier"):
            discounted = max(0, round(discounted * 0.8))
        if self._has_relic("Membership Card"):
            discounted = max(0, round(discounted * 0.5))
        return discounted

    def _enter_card_reward(self, *, extra_gold_rewards: list[int] | None = None, include_base_gold: bool = True) -> None:
        self.phase = "CARD_REWARD"
        self.reward_close_required = False
        self.reward_card_bundles = []
        self.reward_gold = 0
        self.reward_gold_piles = []
        self.reward_emerald_key = False
        self.reward_relics = []
        self.reward_potions = []
        reward_count = 3 + (1 if any(relic.get("relic_id") == "Question Card" for relic in self.relics) else 0)
        reward_bundle_count = 2 if self.current_node_symbol == "M" and any(relic.get("relic_id") == "Prayer Wheel" for relic in self.relics) else 1
        if any(relic.get("relic_id") == "Busted Crown" for relic in self.relics):
            reward_count = max(1, reward_count - 2)
        for _ in range(reward_bundle_count):
            self.reward_card_bundles.append(self._roll_card_reward(count=reward_count))
        self._refresh_reward_cards()
        if include_base_gold and not any(relic.get("relic_id") == "Ectoplasm" for relic in self.relics):
            if self.current_node_symbol in {"E", "E_GREEN", "ACT4_ELITE"}:
                gold_gain = self.randoms.treasure.randint(25, 35)
            elif self.current_node_symbol == "BOSS":
                gold_gain = 100 + self.randoms.misc.randint(-5, 5)
                if self.ascension_level >= 13:
                    gold_gain = int(round(gold_gain * 0.75))
            else:
                gold_gain = self.randoms.treasure.randint(10, 20)
            if any(relic.get("relic_id") == "Golden Idol" for relic in self.relics):
                gold_gain += _sts_round(float(gold_gain) * 0.25)
            self.reward_gold_piles.append(gold_gain)
            self.reward_gold += gold_gain
        if extra_gold_rewards:
            for amount in extra_gold_rewards:
                amount = int(amount)
                if amount > 0:
                    self.reward_gold_piles.append(amount)
                    self.reward_gold += amount
        if self.current_node_symbol in {"E", "E_GREEN", "ACT4_ELITE"}:
            self.reward_relics.append(self._roll_relic(elite=True))
            if any(relic.get("relic_id") == "Black Star" for relic in self.relics):
                self.reward_relics.append(self._roll_relic(elite=True))
            if self.enable_act4_keys and self.current_node_symbol == "E_GREEN" and "emerald" not in self.keys:
                self.reward_emerald_key = True
        self.player.block = 0
        self.player.energy = 0
        self.player.powers = {}
        if self._has_relic("Black Blood"):
            self._heal_run(12)
        elif self._has_relic("Burning Blood"):
            self._heal_run(6)
        if any(relic.get("relic_id") == "Meat on the Bone" for relic in self.relics) and self.player.current_hp <= self.player.max_hp // 2:
            self._heal_run(12)
        if any(relic.get("relic_id") == "Face Of Cleric" for relic in self.relics):
            self.player.max_hp += 1
            self.player.current_hp += 1
        chance = 100 if self._has_relic("White Beast Statue") else 40 + self.potion_chance_counter
        rewards_size = len(self.reward_card_bundles) + len(self.reward_relics) + len(self.reward_potions) + len(self.reward_gold_piles)
        if rewards_size >= 4:
            chance = 0
        if not any(relic.get("relic_id") == "Sozu" for relic in self.relics):
            if int(self.randoms.potion.random(99)) >= chance:
                self.potion_chance_counter += 10
            else:
                self.reward_potions.append(roll_potion(self.randoms.potion))
                self.potion_chance_counter -= 10

    def _add_potion_if_space(self, potion: PotionInstance) -> bool:
        if any(relic.get("relic_id") == "Sozu" for relic in self.relics):
            return False
        for index, current in enumerate(self.potions):
            if not current.can_use:
                self.potions[index] = potion
                return True
        return False

    def _add_random_potion_reward(self, *, count: int = 1) -> int:
        added = 0
        for _ in range(max(0, int(count))):
            if self._add_potion_if_space(roll_potion(self.randoms.potion)):
                added += 1
        return added

    def _open_potion_reward_screen(self, *, count: int = 1, context: str = "EVENT") -> None:
        self.phase = "CARD_REWARD"
        self.reward_context = context
        self.reward_close_required = False
        self.reward_card_bundles = []
        self.reward_cards = []
        self.reward_gold = 0
        self.reward_gold_piles = []
        self.reward_emerald_key = False
        self.reward_relics = []
        self.reward_potions = []
        for _ in range(max(0, int(count))):
            self.reward_potions.append(get_random_potion(self.randoms.potion, "IRONCLAD"))

    def _open_relic_reward_screen(self, relic: dict[str, Any], *, context: str = "EVENT") -> None:
        self.phase = "CARD_REWARD"
        self.reward_context = context
        self.reward_close_required = False
        self.reward_card_bundles = []
        self.reward_cards = []
        self.reward_gold = 0
        self.reward_gold_piles = []
        self.reward_emerald_key = False
        self.reward_relics = [relic]
        self.reward_potions = []

    def _add_colorless_cards_to_deck(self, count: int, *, prefix: str) -> None:
        for index in range(max(0, int(count))):
            reward = roll_colorless_card(self.randoms.card)
            self._add_card_to_deck(reward.card_id, upgrades=reward.upgrades, uuid=f"{prefix}-{self.floor}-{index}")

    def _gain_gold(self, amount: int) -> None:
        amount = max(0, int(amount))
        if amount <= 0 or any(relic.get("relic_id") == "Ectoplasm" for relic in self.relics):
            return
        self.gold += amount
        if any(relic.get("relic_id") == "Bloody Idol" for relic in self.relics):
            self._heal_run(5)

    def _heal_run(self, amount: int) -> None:
        amount = max(0, int(amount))
        if self._has_relic("Mark of the Bloom"):
            return
        self.player.current_hp = min(self.player.max_hp, self.player.current_hp + amount)

    def _restore_from_run_death(self) -> bool:
        if self.player.current_hp > 0:
            return False
        if self._has_relic("Mark of the Bloom"):
            self.phase = "GAME_OVER"
            return True
        bark_multiplier = 2 if self._has_relic("Sacred Bark") else 1
        for index, potion in enumerate(self.potions):
            if potion.potion_id != "FairyPotion":
                continue
            self.potions[index] = PotionInstance()
            self.player.current_hp = max(1, int(self.player.max_hp * (0.3 * bark_multiplier)))
            return True
        if (relic := self._relic("Lizard Tail")) is not None and int(relic.get("counter", -1)) != 0:
            relic["counter"] = 0
            self.player.current_hp = max(1, self.player.max_hp // 2)
            return True
        self.phase = "GAME_OVER"
        return True

    def _enter_map(self) -> None:
        self.phase = "MAP"
        next_floor = self.floor + 1
        if next_floor > 54:
            self.phase = "COMPLETE"
            return
        if next_floor in {51, 52, 53, 54}:
            symbol = {51: "R", 52: "$", 53: "ACT4_ELITE", 54: "HEART"}[next_floor]
            self.map_options = [{
                "kind": "map",
                "name": symbol,
                "symbol": symbol,
                "floor": next_floor,
                "choice_index": 0,
                "node_id": f"act4-{next_floor}-{symbol}",
            }]
            return

        next_act = self._act_for_floor(next_floor)
        if not self.map_layers or not self.map_layers.get(next_floor):
            self._generate_act_map(next_act)

        if self.current_map_node_id:
            current_node = self.map_graph.get(self.current_map_node_id, {})
            node_ids = [
                node_id for node_id in current_node.get("children", [])
                if self.map_graph.get(node_id, {}).get("floor") == next_floor
            ]
        else:
            node_ids = [
                node_id for node_id in self.map_layers.get(next_floor, [])
                if self.map_graph.get(node_id, {}).get("symbol")
            ]
        if not node_ids:
            self._generate_act_map(next_act)
            node_ids = [
                node_id for node_id in self.map_layers.get(next_floor, [])
                if self.map_graph.get(node_id, {}).get("symbol")
            ]
        self.map_options = [
            {
                "kind": "map",
                "name": self.map_graph[node_id]["symbol"],
                "symbol": self.map_graph[node_id]["symbol"],
                "floor": next_floor,
                "choice_index": index,
                "node_id": node_id,
                "x": 0 if self.map_graph[node_id]["symbol"] == "BOSS" else self.map_graph[node_id]["x"],
            }
            for index, node_id in enumerate(node_ids)
        ]

    @staticmethod
    def _act_for_floor(floor: int) -> int:
        if floor <= 16:
            return 1
        if floor <= 33:
            return 2
        if floor <= 50:
            return 3
        return 4

    @staticmethod
    def _act_floor_range(act: int) -> tuple[int, int]:
        return {
            1: (1, 16),
            2: (18, 33),
            3: (35, 50),
        }.get(act, (51, 54))

    def _generate_act_map(self, act: int) -> None:
        if act >= 4:
            return
        start_floor, boss_floor = self._act_floor_range(act)
        graph, layers = generate_act_map(
            seed=self.seed,
            ascension_level=self.ascension_level,
            act=act,
            start_floor=start_floor,
            set_burning=self.enable_act4_keys and "emerald" not in self.keys,
        )
        # The native simulator marks the burning elite as E_GREEN for model
        # visibility. Real run-history path symbols still record it as E.
        self.map_graph = graph
        self.map_layers = layers
        self.current_map_node_id = None

    def _roll_map_symbol(self, act: int, floor: int, start_floor: int, boss_floor: int, rng: StsRandom) -> str:
        if floor == boss_floor:
            return "BOSS"
        if floor == boss_floor - 1:
            return "R"
        relative = floor - start_floor
        if relative <= 1:
            return rng.choices(["M", "?", "$"], weights=[0.72, 0.22, 0.06], k=1)[0]
        if relative <= 3:
            return rng.choices(["M", "?", "$", "T"], weights=[0.54, 0.24, 0.12, 0.10], k=1)[0]
        return rng.choices(["M", "?", "$", "R", "T", "E"], weights=[0.42, 0.22, 0.11, 0.12, 0.04, 0.09], k=1)[0]

    def _enter_boss_relic(self) -> None:
        self.phase = "BOSS_RELIC"
        self.boss_relic_options = self._roll_boss_relics(count=3)
        for index, relic in enumerate(self.boss_relic_options):
            relic["kind"] = "boss_relic"
            relic["choice_index"] = index

    def _enter_boss_treasure_room(self) -> None:
        self.floor += 1
        floor_rng = StsRandom(self.seed + self.floor)
        self.randoms.misc = floor_rng.copy()
        self.randoms.shuffle = floor_rng.copy()
        self.randoms.card_random = floor_rng.copy()
        self.rng = self.randoms.misc
        self._enter_boss_relic()

    def _transition_to_next_act(self) -> None:
        next_act = min(self.act + 1, 4)
        if next_act != self.act:
            self.act = next_act
            self._sync_card_rng_for_act_transition()
            self._reset_act_random_room_state()
            self._generate_monster_schedules_for_act(self.act)
            if hasattr(self, "_act_bosses"):
                self.act_boss = self._act_bosses.get(self.act, self.act_boss)
            self._generate_act_map(self.act)
            self.current_map_node_id = None
            missing_hp = max(0, self.player.max_hp - self.player.current_hp)
            if self.ascension_level >= 5:
                self._heal_run(_sts_round(missing_hp * 0.75))
            else:
                self._heal_run(self.player.max_hp)
        self._enter_map()

    def _sync_card_rng_for_act_transition(self) -> None:
        target_counter = {2: 250, 3: 500, 4: 750}.get(self.act)
        if target_counter is not None and self.randoms.card.counter < target_counter:
            self.randoms.card.set_counter(target_counter)

    def _reset_act_random_room_state(self) -> None:
        self.monster_chance = _f32(0.10)
        self.shop_chance = _f32(0.03)
        self.treasure_chance = _f32(0.02)
        self.potion_chance_counter = 0
        self.event_list, self.shrine_list = self._event_pools_for_act()

    def _current_map_row(self) -> int:
        if not self.current_map_node_id:
            return 0
        return int(self.map_graph.get(self.current_map_node_id, {}).get("row", 0))

    def _can_add_one_time_event(self, event_id: str) -> bool:
        if event_id == "The Divine Fountain":
            return any(card.card_def.card_type == "CURSE" for card in self.deck)
        if event_id == "Designer In-Spire":
            return self.act in {2, 3} and self.gold >= 75
        if event_id == "Duplicator":
            return self.act in {2, 3}
        if event_id == "Face Trader":
            return self.act in {1, 2}
        if event_id == "Knowing Skull":
            return self.act == 2 and self.player.current_hp > 12
        if event_id == "N'loth":
            return self.act == 2 and len(self.relics) >= 2
        if event_id == "The Joust":
            return self.act == 2 and self.gold >= 50
        if event_id == "The Woman in Blue":
            return self.gold >= 50
        if event_id == "Secret Portal":
            return self.act == 3
        return True

    def _can_add_event(self, event_id: str) -> bool:
        if event_id in {"Dead Adventurer", "Hypnotizing Colored Mushrooms"}:
            return self.floor > 6
        if event_id == "The Moai Head":
            return self.player.current_hp <= self.player.max_hp // 2 or self._has_relic("Golden Idol")
        if event_id == "The Cleric":
            return self.gold >= 35
        if event_id == "Old Beggar":
            return self.gold >= 75
        if event_id == "Colosseum":
            return self._current_map_row() > 7
        return True

    def _draw_event_id(self) -> str:
        if not (self.event_list or self.shrine_list or self.special_one_time_event_list):
            self._reset_act_random_room_state()
        # StS/lightspeed select the concrete event from a copy of eventRng:
        # question-room outcome consumes eventRng, but event selection itself
        # only mutates event pools.
        event_rng = self.randoms.event.copy()
        use_shrine = event_rng.random(1.0) < 0.25
        available_shrines = [
            event_id for event_id in self.shrine_list
            if event_id != "Match and Keep"
        ] + [
            event_id for event_id in self.special_one_time_event_list
            if self._can_add_one_time_event(event_id)
        ]
        available_events = [event_id for event_id in self.event_list if self._can_add_event(event_id)]
        if use_shrine:
            if not available_shrines and available_events:
                event_id = event_rng.choice(available_events)
                self.event_list.remove(event_id)
                return event_id
            combined = available_shrines or available_events
            event_id = event_rng.choice(combined)
            if event_id in self.shrine_list:
                self.shrine_list.remove(event_id)
            elif event_id in self.special_one_time_event_list:
                self.special_one_time_event_list.remove(event_id)
            return event_id
        pool = available_events or available_shrines
        event_id = event_rng.choice(pool)
        if event_id in self.event_list:
            self.event_list.remove(event_id)
        elif event_id in self.shrine_list:
            self.shrine_list.remove(event_id)
        elif event_id in self.special_one_time_event_list:
            self.special_one_time_event_list.remove(event_id)
        return event_id

    def _question_room_outcome(self, *, last_room_was_shop: bool = False) -> str:
        tiny_chest = self._relic("Tiny Chest")
        if tiny_chest is not None:
            counter = int(tiny_chest.get("counter", 0))
            if counter >= 3:
                tiny_chest["counter"] = 0
                return "T"
            tiny_chest["counter"] = counter + 1
        roll = self.randoms.event.random()
        monster_size = int(_f32(self.monster_chance * _f32(100.0)))
        shop_size = monster_size + (0 if last_room_was_shop else int(_f32(self.shop_chance * _f32(100.0))))
        treasure_size = shop_size + int(_f32(self.treasure_chance * _f32(100.0)))
        index = int(roll * 100.0)
        if index < monster_size:
            outcome = "M"
        elif index < shop_size:
            outcome = "$"
        elif index < treasure_size:
            outcome = "T"
        else:
            outcome = "?"
        if outcome == "M" and self._has_relic("Juzu Bracelet"):
            outcome = "?"
        self.monster_chance = _f32(0.10) if outcome == "M" else _f32(self.monster_chance + _f32(0.10))
        self.shop_chance = _f32(0.03) if outcome == "$" else _f32(self.shop_chance + _f32(0.03))
        self.treasure_chance = _f32(0.02) if outcome == "T" else _f32(self.treasure_chance + _f32(0.02))
        return outcome

    def _advance_to_node(self, symbol: str) -> None:
        node_id = None
        if symbol in self.map_graph:
            node_id = symbol
            if self.current_map_node_id and node_id not in set(self.map_graph.get(self.current_map_node_id, {}).get("children", [])):
                wing_boots = self._relic("Wing Boots")
                if wing_boots is not None and int(wing_boots.get("counter", -1)) != 0:
                    counter = int(wing_boots.get("counter", -1))
                    wing_boots["counter"] = 2 if counter < 0 else max(0, counter - 1)
            symbol = str(self.map_graph[node_id]["symbol"])
        previous_symbol = self.current_node_symbol
        if symbol == "?":
            symbol = self._question_room_outcome(last_room_was_shop=previous_symbol == "$")
        battle_symbol = symbol
        self.current_node_symbol = symbol
        self.floor += 1
        maw_bank = self._relic("Maw Bank")
        if maw_bank is not None and int(maw_bank.get("counter", -1)) != 0:
            self._gain_gold(12)
        self.current_map_node_id = node_id
        old_act = self.act
        self.act = self._act_for_floor(self.floor)
        floor_rng = StsRandom(self.seed + self.floor)
        self.randoms.misc = floor_rng.copy()
        self.randoms.shuffle = floor_rng.copy()
        self.randoms.card_random = floor_rng.copy()
        self.rng = self.randoms.misc
        if self.act != old_act:
            self.player.current_hp = self.player.max_hp
            self._sync_card_rng_for_act_transition()
            self._reset_act_random_room_state()
            self._generate_monster_schedules_for_act(self.act)
        if hasattr(self, "_act_bosses"):
            self.act_boss = self._act_bosses.get(self.act, self.act_boss)
        if battle_symbol not in {"M", "BOSS", "E", "E_GREEN", "ACT4_ELITE", "HEART"}:
            tea_set = self._relic("Ancient Tea Set")
            if tea_set is not None:
                tea_set["counter"] = 0
        if battle_symbol in {"M", "BOSS", "E", "E_GREEN", "ACT4_ELITE", "HEART"}:
            self._start_combat(elite=battle_symbol in {"E", "E_GREEN", "ACT4_ELITE"})
        elif symbol == "R":
            self._enter_campfire()
        elif symbol == "$":
            self._enter_shop()
        elif symbol == "?":
            self._enter_event()
        elif symbol == "T":
            self._enter_treasure_room()
        else:
            self._start_combat()

    def _enter_campfire(self) -> None:
        self.phase = "CAMPFIRE"
        self.campfire_options = []
        if not any(relic.get("relic_id") == "Coffee Dripper" for relic in self.relics):
            self.campfire_options.append({"kind": "campfire", "name": "REST", "choice_index": len(self.campfire_options)})
        if not any(relic.get("relic_id") == "Fusion Hammer" for relic in self.relics):
            self.campfire_options.append({"kind": "campfire", "name": "SMITH", "choice_index": len(self.campfire_options)})
        if self._has_relic("Peace Pipe"):
            self.campfire_options.append({"kind": "campfire", "name": "TOKE", "choice_index": len(self.campfire_options)})
        if self._has_relic("Shovel"):
            self.campfire_options.append({"kind": "campfire", "name": "DIG", "choice_index": len(self.campfire_options)})
        girya = self._relic("Girya")
        if girya is not None and int(girya.get("counter", 0)) < 3:
            self.campfire_options.append({"kind": "campfire", "name": "LIFT", "choice_index": len(self.campfire_options)})
        if self.enable_act4_keys and "ruby" not in self.keys:
            self.campfire_options.append({"kind": "campfire", "name": "RECALL", "choice_index": len(self.campfire_options)})
        if not self.campfire_options:
            self.campfire_options.append({"kind": "campfire", "name": "LEAVE", "choice_index": 0})

    def _enter_treasure_room(self) -> None:
        self.phase = "TREASURE"
        size_roll = int(self.randoms.treasure.random(99))
        if size_roll < 50:
            chest_index = 0
            self.chest_size = "SMALL"
        elif size_roll < 83:
            chest_index = 1
            self.chest_size = "MEDIUM"
        else:
            chest_index = 2
            self.chest_size = "LARGE"

        tier_gold_roll = int(self.randoms.treasure.random(99))
        gold_chances = [50, 35, 50]
        self.chest_have_gold = tier_gold_roll < gold_chances[chest_index]
        self.chest_gold_amount = 0

        tier_chances = [(75, 25), (35, 50), (0, 75)]
        common_chance, uncommon_chance = tier_chances[chest_index]
        if tier_gold_roll < common_chance:
            self.chest_relic_tier = "COMMON"
        elif tier_gold_roll < common_chance + uncommon_chance:
            self.chest_relic_tier = "UNCOMMON"
        else:
            self.chest_relic_tier = "RARE"

        self.treasure_options = [
            {"kind": "treasure", "name": "LEAVE", "choice_index": 0},
            {"kind": "treasure", "name": "OPEN_CHEST", "choice_index": 1},
        ]

    def _enter_shop(self) -> None:
        self.phase = "SHOP"
        if self._has_relic("Meal Ticket"):
            self._heal_run(15)
        cards, colorless_cards = self._roll_shop_cards()
        card_items: list[dict[str, Any]] = []
        for index, (card, rarity) in enumerate(cards):
            card_items.append({
                "kind": "shop",
                "name": card.name,
                "item_kind": "card",
                "item_id": card.card_id,
                "price": self._shop_card_price(card, rarity_override=rarity, sale=False),
                "card": card_to_spirecomm(card),
                "choice_index": index,
            })
        for colorless, rarity in colorless_cards:
            card_items.append({
                "kind": "shop",
                "name": colorless.name,
                "item_kind": "card",
                "item_id": colorless.card_id,
                "price": self._shop_card_price(colorless, colorless=True, rarity_override=rarity, sale=False),
                "card": card_to_spirecomm(colorless),
                "choice_index": len(card_items),
            })
        sale_idx = int(self.randoms.merchant.random(4))
        if 0 <= sale_idx < len(card_items):
            card_items[sale_idx]["price"] //= 2
        self.shop_items = list(card_items)
        shop_relics: list[dict[str, Any]] = []
        for _ in range(2):
            tier_roll = int(self.randoms.merchant.random(99))
            tier = "COMMON" if tier_roll < 48 else "UNCOMMON" if tier_roll < 82 else "RARE"
            relic = self._roll_relic_of_tier(tier, shop_room=True, from_front=False)
            relic["price"] = self._shop_relic_price(relic)
            shop_relics.append(relic)
        shop_relic = self._roll_relic_of_tier("SHOP", shop_room=True, from_front=False)
        shop_relic["price"] = self._shop_relic_price(shop_relic, rarity_override="SHOP")
        shop_relics.append(shop_relic)
        for relic in shop_relics:
            self.shop_items.append({
                "kind": "shop",
                "name": relic["name"],
                "item_kind": "relic",
                "item_id": relic["relic_id"],
                "price": relic["price"],
                "relic": relic,
                "choice_index": len(self.shop_items),
            })
        if not any(item.get("relic_id") == "Sozu" for item in self.relics):
            for _ in range(3):
                potion = roll_potion(self.randoms.potion)
                potion.price = self._shop_potion_price(potion)
                self.shop_items.append({
                    "kind": "shop",
                    "name": potion.name,
                    "item_kind": "potion",
                    "item_id": potion.potion_id,
                    "price": potion.price,
                    "potion_id": potion.potion_id,
                    "choice_index": len(self.shop_items),
                })
        purge_base = 50 if self._has_relic("Smiling Mask") else 75 + 25 * self.shop_remove_count
        purge_price = self._apply_shop_discount(purge_base, include_ascension=False)
        self.shop_items.append({"kind": "shop", "name": "PURGE", "item_kind": "purge", "price": purge_price, "choice_index": len(self.shop_items)})
        self.shop_items.append({"kind": "shop", "name": "LEAVE", "item_kind": "leave", "price": 0, "choice_index": len(self.shop_items)})

    def _shop_card_price(self, card: CardInstance, *, colorless: bool = False, rarity_override: str | None = None, sale: bool = False) -> int:
        rarity = rarity_override or card.card_def.rarity
        base = {
            "COMMON": 50,
            "UNCOMMON": 75,
            "RARE": 150,
        }.get(rarity, 75)
        if colorless:
            base = float(base) * 1.2
        price = max(0, int(float(base) * self.randoms.merchant.random(0.9, 1.1)))
        if sale:
            price //= 2
        return self._apply_shop_discount(price)

    def _shop_relic_price(self, relic: dict[str, Any], *, rarity_override: str | None = None) -> int:
        rarity = rarity_override or str(relic.get("tier") or relic.get("rarity") or "COMMON")
        base = {
            "COMMON": 150,
            "UNCOMMON": 250,
            "IRONCLAD": 250,
            "RARE": 300,
            "SHOP": 150,
            "EVENT": 250,
        }.get(rarity, 200)
        return self._apply_shop_discount(max(0, round(base * self.randoms.merchant.random(0.95, 1.05))))

    def _shop_potion_price(self, potion: PotionInstance) -> int:
        base = {
            "COMMON": 50,
            "UNCOMMON": 75,
            "RARE": 100,
        }.get(potion.potion_def.rarity if potion.potion_def else "COMMON", 50)
        return self._apply_shop_discount(max(0, round(base * self.randoms.merchant.random(0.95, 1.05))))

    def _shop_courier_restock_card_price(self, rarity: str, *, colorless: bool = False) -> int:
        base = {
            "COMMON": 50,
            "UNCOMMON": 75,
            "RARE": 150,
        }.get(rarity, 75)
        price = float(base) * float(self.randoms.merchant.random(0.9, 1.1))
        if colorless:
            price *= 1.2
        # Mirror lightspeed's current getNewCardPrice implementation, which
        # mistakenly checks The Courier twice instead of Membership Card.
        if self._has_relic("The Courier"):
            price *= 0.8
        if self._has_relic("The Courier"):
            price *= 0.5
        return int(price)

    def _shop_courier_restock_misc_price(self) -> int:
        # Mirror lightspeed's current getNewPrice implementation, which drops
        # the provided base price and discounts entirely.
        return int(round(self.randoms.merchant.random(0.95, 1.05)))

    def _restock_shop_item(self, item_kind: str, removed_item: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self._has_relic("The Courier") or item_kind not in {"card", "relic", "potion"}:
            return None
        if item_kind == "card":
            removed_card_id = str((removed_item or {}).get("item_id") or "")
            if removed_card_id in COLORLESS_CARD_IDS:
                rarity = "RARE" if self.randoms.merchant.random() < 0.30 else "UNCOMMON"
                colorless_pool = [CARD_LIBRARY[card_id] for card_id in COLORLESS_CARD_ID_ORDER if CARD_LIBRARY[card_id].rarity == rarity]
                chosen = colorless_pool[int(self.randoms.card.random(len(colorless_pool) - 1))]
                card = self._make_deck_card(chosen.card_id, uuid=f"shop-colorless-{chosen.card_id}-{self.floor}")
                price = self._shop_courier_restock_card_price(rarity, colorless=True)
            else:
                rarity = self._roll_card_rarity(room="$")
                card = self._random_class_card_of_rarity_from_rng(self.randoms.math_util, rarity)
                price = self._shop_courier_restock_card_price(rarity, colorless=False)
            return {
                "kind": "shop",
                "name": card.name,
                "item_kind": "card",
                "item_id": card.card_id,
                "price": price,
                "card": card_to_spirecomm(card),
            }
        if item_kind == "relic":
            tier_roll = int(self.randoms.merchant.random(99))
            tier = "COMMON" if tier_roll < 48 else "UNCOMMON" if tier_roll < 82 else "RARE"
            relic = self._roll_relic_of_tier(tier, shop_room=True, from_front=False)
            relic["price"] = self._shop_courier_restock_misc_price()
            return {
                "kind": "shop",
                "name": relic["name"],
                "item_kind": "relic",
                "item_id": relic["relic_id"],
                "price": relic["price"],
                "relic": relic,
            }
        potion = roll_potion(self.randoms.potion)
        potion.price = self._shop_courier_restock_misc_price()
        return {
            "kind": "shop",
            "name": potion.name,
            "item_kind": "potion",
            "item_id": potion.potion_id,
            "price": potion.price,
            "potion_id": potion.potion_id,
        }

    def _remove_shop_item(self, action: dict[str, Any]) -> None:
        item_kind = str(action.get("item_kind") or "")
        action_id = action.get("item_id")
        action_name = action.get("name")
        action_price = int(action.get("price", 0) or 0)
        kept: list[dict[str, Any]] = []
        removed = False
        for item in self.shop_items:
            same_item = (
                item.get("item_kind") == item_kind
                and item.get("item_id") == action_id
                and str(item.get("name") or "") == str(action_name or "")
                and int(item.get("price", 0) or 0) == action_price
            )
            if not removed and same_item:
                removed = True
                continue
            kept.append(item)
        self.shop_items = kept
        replacement = self._restock_shop_item(item_kind, action)
        if replacement is not None:
            replacement["choice_index"] = len(self.shop_items)
            self.shop_items.insert(max(0, len(self.shop_items) - 1), replacement)

    def _enter_event(self) -> None:
        self.phase = "EVENT"
        event_id = self._draw_event_id()
        self.current_event_id = event_id
        self.event_state = {}
        if event_id == "Bonfire Spirits":
            self._open_card_select("BONFIRE_SPIRITS", 1)
            return
        if event_id == "Lab":
            self._open_potion_reward_screen(count=2 if self.ascension_level >= 15 else 3, context="EVENT")
            return
        if event_id == "Scrap Ooze":
            self.event_state["counter"] = 0
        elif event_id == "Dead Adventurer":
            rewards = [0, 1, 2]
            java_collections_shuffle(rewards, self.randoms.misc.random_long())
            self.event_state["phase"] = 0
            self.event_state["rewards"] = rewards
            self.event_state["encounter"] = self.randoms.misc.choice(["Three Sentries", "Gremlin Nob", "Lagavulin"])
        elif event_id == "Cursed Tome":
            self.event_state["phase"] = 0
        elif event_id in {"We Meet Again!", "WeMeetAgain"}:
            potion_indexes = [idx for idx, potion in enumerate(self.potions) if potion.can_use]
            if potion_indexes:
                java_collections_shuffle(potion_indexes, self.randoms.misc.next_long())
                self.event_state["potion_idx"] = potion_indexes[0]
            else:
                self.event_state["potion_idx"] = None
            self.event_state["gold"] = -1 if self.gold < 50 else int(self.randoms.misc.random(50, min(150, self.gold)))
            card_indexes = [
                idx
                for idx, card in enumerate(self.deck)
                if card.card_def.rarity != "BASIC" and card.card_def.card_type != "CURSE"
            ]
            if card_indexes:
                java_collections_shuffle(card_indexes, self.randoms.misc.random_long())
                selected_idx = card_indexes[0]
                for idx, card in enumerate(self.deck[:selected_idx]):
                    if card.card_id == self.deck[selected_idx].card_id:
                        selected_idx = idx
                        break
                self.event_state["card_idx"] = selected_idx
            else:
                self.event_state["card_idx"] = None
        elif event_id == "World of Goop":
            low, high = (35, 75) if self.ascension_level >= 15 else (20, 50)
            self.event_state["gold_loss"] = min(self.gold, int(self.randoms.misc.random(low, high)))
        option_labels = {
            "Big Fish": ["Banana", "Donut", "Box"],
            "Golden Idol": ["Take", "Ignored"],
            "Golden Shrine": ["Prayed", "Desecrated", "Ignored"],
            "Shining Light": ["Entered Light", "Ignored"],
            "The Cleric": ["Healed", "Card Removal", "Leave"],
            "Living Wall": ["Forget", "Change", "Grow"],
            "World of Goop": ["Gather Gold", "Left Gold"],
            "Scrap Ooze": ["Success", "Fled"],
            "The Ssssserpent": ["Agreed", "Ignored"],
            "Hypnotizing Colored Mushrooms": ["Fought Mushrooms", "Ignored"],
            "Mindbloom": ["Fight", "Gold", "Heal", "Upgrade"],
            "Falling": ["Removed Attack", "Removed Skill", "Removed Power"],
            "Winding Halls": ["Embrace Madness", "Max HP", "Writhe"],
            "Wing Statue": ["Card Removal", "Gained Gold", "Ignored"],
            "Drug Dealer": ["Got JAX", "Inject Mutagens", "Ignored"],
            "Ancient Writing": ["Elegance", "Simplicity"],
            "Old Beggar": ["Gave Gold", "Ignored"],
            "Cursed Tome": ["Read", "Ignored"],
            "Augmenter": ["JAX", "Mutagenic Strength", "Transform"],
            "The Nest": ["Stole From Cult", "Ignored"],
            "The Library": ["Read", "Heal"],
            "Accursed Blacksmith": ["Forge", "Rummage", "Ignored"],
            "The Mausoleum": ["Opened", "Ignored"],
            "Tomb of Lord Red Mask": ["Got Gold", "Paid", "Ignored"],
            "Masked Bandits": ["Paid Fearfully", "Fought Bandits"],
            "Vampires": ["Accepted", "Refused"],
            "Ghosts": ["Accepted", "Refused"],
            "Duplicator": ["Duplicated", "Ignored"],
            "N'loth": ["Gave Relic", "Ignored"],
            "The Joust": ["Murderer", "Owner", "Ignored"],
            "The Divine Fountain": ["Drank", "Ignored"],
            "Knowing Skull": ["Asked", "Ignored"],
            "Note For Yourself": ["Took Card", "Ignored"],
            "Secret Portal": ["Entered Portal", "Ignored"],
            "We Meet Again!": ["Gave Potion", "Gave Gold", "Gave Card", "Ignored"],
            "Bonfire Spirits": ["Card Offered", "Ignored"],
            "Designer In-Spire": ["Adjusted", "Cleaned Up", "Full Service", "Ignored"],
            "Face Trader": ["Touched", "Traded", "Ignored"],
            "Forgotten Altar": ["Shed Blood", "Smashed Altar", "Ignored"],
            "Lab": ["Obtained Potions"],
            "Match and Keep": ["Played"],
            "The Moai Head": ["Jumped", "Offered Golden Idol", "Ignored"],
            "Purifier": ["Purged", "Ignored"],
            "Sensory Stone": ["Recall", "Remember", "Live Forever", "Ignored"],
            "The Woman in Blue": ["Bought 1 Potion", "Bought 2 Potions", "Bought 3 Potions", "Ignored"],
            "Transmogrifier": ["Transformed", "Ignored"],
            "Upgrade Shrine": ["Upgraded", "Ignored"],
            "Wheel of Change": ["Spun"],
            "Pleading Vagrant": ["Gave Gold", "Robbed", "Ignored"],
            "Dead Adventurer": ["Searched", "Escaped"],
            "Ominous Forge": ["Forge", "Rummage", "Ignored"],
            "Mysterious Sphere": ["Fought Orb Walkers", "Ignored"],
            "Colosseum": ["Fought", "Cowardice"],
        }.get(event_id, ["Ignored"])
        if event_id == "Golden Idol":
            if any(relic.get("relic_id") == "Golden Idol" for relic in self.relics):
                self.event_options = [
                    {"kind": "event", "event_id": event_id, "name": "Outrun", "label": "Outrun", "choice_index": 2},
                    {"kind": "event", "event_id": event_id, "name": "Smash", "label": "Smash", "choice_index": 3},
                    {"kind": "event", "event_id": event_id, "name": "Hide", "label": "Hide", "choice_index": 4},
                ]
            else:
                self.event_options = [
                    {"kind": "event", "event_id": event_id, "name": "Take", "label": "Take", "choice_index": 0},
                    {"kind": "event", "event_id": event_id, "name": "Ignored", "label": "Ignored", "choice_index": 1},
                ]
        elif event_id == "The Cleric":
            self.event_options = [
                {"kind": "event", "event_id": event_id, "name": "Healed", "label": "Healed", "choice_index": 0},
            ]
            purge_cost = 75 if self.ascension_level >= 15 else 50
            if self.gold >= purge_cost:
                self.event_options.append(
                    {"kind": "event", "event_id": event_id, "name": "Card Removal", "label": "Card Removal", "choice_index": 1}
                )
            self.event_options.append(
                {"kind": "event", "event_id": event_id, "name": "Leave", "label": "Leave", "choice_index": 2}
            )
        elif event_id == "Wing Statue":
            self.event_options = [
                {"kind": "event", "event_id": event_id, "name": "Card Removal", "label": "Card Removal", "choice_index": 0},
            ]
            if any(_base_damage_for_card(card) >= 10 for card in self.deck):
                self.event_options.append(
                    {"kind": "event", "event_id": event_id, "name": "Gained Gold", "label": "Gained Gold", "choice_index": 1}
                )
            self.event_options.append(
                {"kind": "event", "event_id": event_id, "name": "Ignored", "label": "Ignored", "choice_index": 2}
            )
        elif event_id in {"We Meet Again!", "WeMeetAgain"}:
            self.event_options = []
            if self.event_state.get("potion_idx") is not None:
                self.event_options.append({"kind": "event", "event_id": event_id, "name": "Gave Potion", "label": "Gave Potion", "choice_index": 0})
            if int(self.event_state.get("gold", -1)) >= 0:
                self.event_options.append({"kind": "event", "event_id": event_id, "name": "Gave Gold", "label": "Gave Gold", "choice_index": 1})
            if self.event_state.get("card_idx") is not None:
                self.event_options.append({"kind": "event", "event_id": event_id, "name": "Gave Card", "label": "Gave Card", "choice_index": 2})
            self.event_options.append({"kind": "event", "event_id": event_id, "name": "Ignored", "label": "Ignored", "choice_index": 3})
        elif event_id == "Cursed Tome":
            self._set_cursed_tome_options()
        else:
            self.event_options = [
                {"kind": "event", "event_id": event_id, "name": label, "label": label, "choice_index": index}
                for index, label in enumerate(option_labels)
            ]

    def _set_cursed_tome_options(self) -> None:
        phase = int(self.event_state.get("phase", 0))
        if phase == 0:
            options = [(0, "Read"), (1, "Ignored")]
        elif phase in {1, 2, 3}:
            options = [(phase + 1, "Continue")]
        elif phase == 4:
            options = [(5, "Take Book"), (6, "Leave")]
        else:
            options = []
        self.event_options = [
            {"kind": "event", "event_id": "Cursed Tome", "name": label, "label": label, "choice_index": idx}
            for idx, label in options
        ]

    def _one_time_event_pool(self) -> list[str]:
        pool = [
            "Ominous Forge", "Bonfire Spirits", "Designer In-Spire", "Duplicator",
            "Face Trader", "The Divine Fountain", "Knowing Skull", "Lab", "N'loth",
            "Secret Portal", "The Joust", "We Meet Again!", "The Woman in Blue",
        ]
        if self.ascension_level < 15:
            pool.insert(9, "Note For Yourself")
        return pool

    def _event_pools_for_act(self) -> tuple[list[str], list[str]]:
        shrines_common = ["Match and Keep", "Wheel of Change", "Golden Shrine", "Transmogrifier", "Purifier", "Upgrade Shrine"]
        if self.act <= 1:
            return (
                [
                    "Big Fish", "The Cleric", "Dead Adventurer", "Golden Idol",
                    "Wing Statue", "World of Goop", "The Ssssserpent", "Living Wall",
                    "Hypnotizing Colored Mushrooms", "Scrap Ooze", "Shining Light",
                ],
                ["Match and Keep", "Golden Shrine", "Transmogrifier", "Purifier", "Upgrade Shrine", "Wheel of Change"],
            )
        if self.act == 2:
            return (
                [
                    "Pleading Vagrant", "Ancient Writing", "Old Beggar", "Colosseum",
                    "Cursed Tome", "Augmenter", "Forgotten Altar", "Ghosts",
                    "Masked Bandits", "The Nest", "The Library", "The Mausoleum", "Vampires",
                ],
                list(shrines_common),
            )
        return (
            [
                "Falling", "Mindbloom", "The Moai Head", "Mysterious Sphere",
                "Sensory Stone", "Tomb of Lord Red Mask", "Winding Halls",
            ],
            list(shrines_common),
        )

    def _event_pool_for_act(self) -> list[str]:
        events, shrines = self._event_pools_for_act()
        return events + shrines + list(self.special_one_time_event_list or self._one_time_event_pool())

    def _enter_chest(self) -> None:
        self.phase = "CHEST"
        gold_amounts = {"SMALL": 25, "MEDIUM": 50, "LARGE": 75}
        if self.chest_have_gold:
            base_gold = gold_amounts.get(self.chest_size, 25)
            self.chest_gold_amount = int(round(self.randoms.treasure.random(base_gold * 0.9, base_gold * 1.1)))
        else:
            self.chest_gold_amount = 0

        relic = self._roll_relic_of_tier(self.chest_relic_tier)
        relic["kind"] = "chest"
        relic["name"] = relic.get("name", "Relic")
        relic["item_kind"] = "relic"
        relic["choice_index"] = 0
        relic["gold_amount"] = self.chest_gold_amount
        relic["chest_size"] = self.chest_size
        self.chest_options = [relic]
        if self.enable_act4_keys and "sapphire" not in self.keys:
            self.chest_options.append({
                "kind": "chest",
                "name": "SAPPHIRE_KEY",
                "item_kind": "sapphire_key",
                "choice_index": 1,
                "gold_amount": self.chest_gold_amount,
                "chest_size": self.chest_size,
            })

    def _obtain_relic(self, relic: dict[str, Any]) -> None:
        self.relics.append(relic)
        relic_id = relic.get("relic_id")
        if relic_id == "Strawberry":
            self.player.max_hp += 7
            self.player.current_hp += 7
        elif relic_id == "Pear":
            self.player.max_hp += 10
            self.player.current_hp += 10
        elif relic_id == "Mango":
            self.player.max_hp += 14
            self.player.current_hp += 14
        elif relic_id == "Black Blood":
            self.relics = [item for item in self.relics if item.get("relic_id") != "Burning Blood"]
        elif relic_id == "Mark of Pain":
            self.deck.append(make_card("Wound", uuid=f"relic-{self.floor}-wound-0"))
            self.deck.append(make_card("Wound", uuid=f"relic-{self.floor}-wound-1"))
        elif relic_id == "Potion Belt":
            self.potions.extend(empty_potion_slots(2))
        elif relic_id == "Cauldron":
            for _ in range(len(self.potions)):
                self._add_potion_if_space(roll_potion(self.randoms.potion))
        elif relic_id == "Lee's Waffle":
            self.player.max_hp += 7
            self.player.current_hp = self.player.max_hp
        elif relic_id == "Dolly's Mirror":
            if self.deck:
                source = self.deck[0]
                self._add_card_to_deck(source.card_id, upgrades=source.upgrades, uuid=f"dolly-{self.floor}-{source.card_id}")
        elif relic_id == "Orrery":
            for index, card in enumerate(self._roll_card_reward(count=5)):
                self._add_card_to_deck(card.card_id, upgrades=card.upgrades, uuid=f"orrery-{self.floor}-{index}-{card.card_id}")
        elif relic_id == "Old Coin":
            self._gain_gold(300)
        elif relic_id == "Matryoshka":
            relic["counter"] = 2
        elif relic_id == "Omamori":
            relic["counter"] = 2
        elif relic_id == "Maw Bank":
            relic["counter"] = 1
        elif relic_id == "Tiny Chest":
            relic["counter"] = max(0, int(relic.get("counter", 0)))
        elif relic_id == "Wing Boots":
            relic["counter"] = 3
        elif relic_id == "Girya":
            relic["counter"] = max(0, int(relic.get("counter", 0)))
        elif relic_id in {"Bottled Flame", "Bottled Lightning", "Bottled Tornado"}:
            wanted_type = {
                "Bottled Flame": "ATTACK",
                "Bottled Lightning": "SKILL",
                "Bottled Tornado": "POWER",
            }[str(relic_id)]
            bottled = next((card for card in self.deck if card.card_def.card_type == wanted_type), None)
            if bottled is not None:
                relic["card_id"] = bottled.card_id
        elif relic_id == "Face Of Cleric":
            self.player.max_hp += 1
            self.player.current_hp += 1
        elif relic_id == "Necronomicon":
            if not any(card.card_id == "Necronomicurse" for card in self.deck):
                self._add_curse_to_deck("Necronomicurse", uuid=f"necronomicon-{self.floor}")
        elif relic_id == "Tiny House":
            self.player.max_hp += 5
            self.player.current_hp = min(self.player.max_hp, self.player.current_hp + 5)
            self._gain_gold(50)
            card = self._roll_card_reward(count=1)[0]
            self._add_card_to_deck(card.card_id, upgrades=card.upgrades, uuid=f"tiny-house-{self.floor}-{card.card_id}")
            self._add_potion_if_space(roll_potion(self.randoms.potion))
            index = self._first_upgradable_index()
            if index is not None:
                self.deck[index].upgrades += 1
        elif relic_id == "Empty Cage":
            for _ in range(2):
                index = self._first_purge_index()
                if index is not None:
                    self.deck.pop(index)
        elif relic_id == "Calling Bell":
            self._add_curse_to_deck("CurseOfTheBell", uuid=f"calling-bell-{self.floor}")
            owned = {item.get("relic_id") for item in self.relics}
            for _ in range(3):
                new_relic = self._roll_relic()
                owned.add(new_relic.get("relic_id"))
                self._obtain_relic(new_relic)
        elif relic_id == "Pandora's Box":
            reward_ids = [card.card_id for card in self._roll_card_reward(count=20)]
            replacement = iter(reward_ids)
            for index, card in enumerate(list(self.deck)):
                if card.card_id in {"Strike_R", "Defend_R"}:
                    self.deck[index] = self._make_deck_card(next(replacement, "Pommel Strike"), uuid=f"pandora-{self.floor}-{index}")
        elif relic_id == "Astrolabe":
            self.card_select_available_indexes = [
                index
                for index, card in enumerate(self.deck)
                if card.card_def.card_type not in {"STATUS", "CURSE"}
            ]
            self.card_select_selected_indexes = []
            if self.card_select_available_indexes:
                self._open_card_select("TRANSFORM_UPGRADE", min(3, len(self.card_select_available_indexes)))
        elif relic_id == "War Paint":
            upgradeable = [
                index
                for index, card in enumerate(self.deck)
                if card.card_def.card_type == "SKILL" and _card_can_upgrade(card)
            ]
            java_collections_shuffle(upgradeable, self.randoms.misc.random_long())
            for index in upgradeable[:2]:
                self.deck[index].upgrades += 1
        elif relic_id == "Whetstone":
            upgradeable = [
                index
                for index, card in enumerate(self.deck)
                if card.card_def.card_type == "ATTACK" and _card_can_upgrade(card)
            ]
            java_collections_shuffle(upgradeable, self.randoms.misc.random_long())
            for index in upgradeable[:2]:
                self.deck[index].upgrades += 1

    def _make_deck_card(self, card_id: str, *, upgrades: int = 0, uuid: str = "") -> CardInstance:
        card = make_card(card_id, upgrades=upgrades, uuid=uuid)
        if card.card_def.card_type == "ATTACK" and any(relic.get("relic_id") == "Molten Egg" for relic in self.relics):
            card.upgrades = max(card.upgrades, 1)
        if card.card_def.card_type == "SKILL" and any(relic.get("relic_id") == "Toxic Egg" for relic in self.relics):
            card.upgrades = max(card.upgrades, 1)
        if card.card_def.card_type == "POWER" and any(relic.get("relic_id") == "Frozen Egg" for relic in self.relics):
            card.upgrades = max(card.upgrades, 1)
        return card

    def _add_card_to_deck(self, card_id: str, *, upgrades: int = 0, uuid: str = "") -> CardInstance:
        card = self._make_deck_card(card_id, upgrades=upgrades, uuid=uuid)
        self.deck.append(card)
        if self._has_relic("Ceramic Fish"):
            self._gain_gold(9)
        return card

    def _lose_run_hp(self, amount: int) -> None:
        amount = max(0, int(amount))
        if amount <= 0:
            return
        if self._has_relic("Tungsten Rod"):
            amount = max(0, amount - 1)
            if amount <= 0:
                return
        self.player.current_hp = max(0, self.player.current_hp - amount)
        if self.player.current_hp <= 0:
            self._restore_from_run_death()

    def _first_upgradable_index(self) -> int | None:
        for index, card in enumerate(self.deck):
            if card.upgrades <= 0 and card.card_def.card_type not in {"STATUS", "CURSE"}:
                return index
        return None

    def _first_purge_index(self) -> int | None:
        priority = ["AscendersBane", "Wound", "Burn", "Dazed", "Strike_R", "Defend_R"]
        for card_id in priority:
            for index, card in enumerate(self.deck):
                if card.card_id == card_id:
                    return index
        return 0 if self.deck else None

    def _advance_floor(self) -> None:
        self._enter_map()

    def legal_actions(self) -> list[dict[str, Any]]:
        if self.phase == "COMBAT":
            if self.combat.outcome != "UNDECIDED":
                return [{"kind": "end", "name": "RESOLVE_COMBAT", "action_index": 0, "bits": 0}]
            return self.combat.legal_actions()
        if self.phase == "NEOW":
            return list(self.neow_options)
        if self.phase == "CARD_REWARD":
            actions: list[dict[str, Any]] = []
            flat_index = 0
            for reward_index, bundle in enumerate(self.reward_card_bundles):
                for card_index, card in enumerate(bundle):
                    actions.append({
                        "kind": "card_reward",
                        "name": card.name,
                        "card_id": card.card_id,
                        "choice_index": flat_index,
                        "reward_index": reward_index,
                        "card_index": card_index,
                        "card": card_to_spirecomm(card),
                    })
                    flat_index += 1
            for index, relic in enumerate(self.reward_relics):
                relic_id = relic.get("relic_id")
                action_relic_id = "Paper Phrog" if relic_id == "Paper Frog" else relic_id
                actions.append({
                    "kind": "reward_relic",
                    "name": str(relic.get("name") or relic.get("relic_id") or "RELIC"),
                    "relic_id": action_relic_id,
                    "choice_index": len(actions),
                    "reward_index": index,
                })
            for index, potion in enumerate(self.reward_potions):
                if potion.can_use:
                    actions.append({
                        "kind": "reward_potion",
                        "name": potion.name,
                        "potion_id": potion.potion_id,
                        "choice_index": len(actions),
                        "reward_index": index,
                    })
            for index, amount in enumerate(self.reward_gold_piles):
                actions.append({
                    "kind": "reward_gold",
                    "name": "GOLD",
                    "choice_index": len(actions),
                    "amount": amount,
                    "reward_index": index,
                })
            if self.reward_emerald_key:
                actions.append({
                    "kind": "reward_key",
                    "name": "KEY",
                    "choice_index": len(actions),
                    "key": "emerald",
                })
            actions.append({"kind": "skip", "name": "SKIP", "choice_index": len(actions)})
            return actions
        if self.phase == "CARD_SELECT":
            return list(self.card_select_options)
        if self.phase == "BOSS_RELIC":
            actions = list(self.boss_relic_options)
            actions.append({
                "kind": "boss_relic",
                "name": "SKIP",
                "relic_id": "SKIP",
                "choice_index": len(actions),
            })
            return actions
        if self.phase == "MAP":
            return list(self.map_options)
        if self.phase == "CAMPFIRE":
            return list(self.campfire_options)
        if self.phase == "SHOP":
            return [item for item in self.shop_items if item.get("item_kind") == "leave" or self.gold >= int(item.get("price", 0))]
        if self.phase == "EVENT":
            return list(self.event_options)
        if self.phase == "TREASURE":
            return list(self.treasure_options)
        if self.phase == "CHEST":
            return list(self.chest_options)
        return []

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.phase == "COMBAT":
            state = self.combat.to_spirecomm_state() if self.combat.outcome != "UNDECIDED" else self.combat.step(action)
            self.player = self.combat.player
            if self.combat.outcome == "PLAYER_VICTORY":
                self.player.powers = {}
                self.player.block = 0
                self.potions = self.combat.potions
                self.gold = self.combat.gold
                self._gain_gold(self.combat.gold_gain)
                if self.floor == 50 and self.ascension_level >= 20 and not self.a20_second_boss_done:
                    self.a20_second_boss_done = True
                    act3_bosses = ["Awakened One", "Time Eater", "Donu and Deca"]
                    candidates = [boss for boss in act3_bosses if boss != self.act_boss] or act3_bosses
                    self.act_boss = self.randoms.monster.choice(candidates)
                    self.current_node_symbol = "BOSS"
                    self._start_combat(elite=False)
                    return self.state()
                if self.floor == 53:
                    self._enter_map()
                    return self.state()
                if self.floor >= 54:
                    self.phase = "COMPLETE"
                    return self.state()
                extra_gold_rewards: list[int] = []
                if self.combat.reward_gold_bonus > 0:
                    extra_gold_rewards.append(self.combat.reward_gold_bonus)
                if self.pending_event_gold > 0:
                    extra_gold_rewards.append(self.pending_event_gold)
                    self.pending_event_gold = 0
                self._enter_card_reward(
                    extra_gold_rewards=extra_gold_rewards,
                    include_base_gold=self.current_node_symbol != "EVENT_COMBAT",
                )
                if self.pending_event_relic_id:
                    if not self._has_relic(self.pending_event_relic_id):
                        self.reward_relics.append(make_relic(self.pending_event_relic_id))
                    self.pending_event_relic_id = None
                return self.state()
            if self.combat.outcome == "PLAYER_LOSS":
                self.phase = "GAME_OVER"
            return state
        if self.phase == "NEOW":
            option_index = int(action.get("choice_index", 0))
            if 0 <= option_index < len(self.neow_options):
                option = self.neow_options[option_index]
                drawback = str(option.get("drawback") or "NONE")
                self._apply_neow_drawback(drawback)
                if drawback == "CURSE":
                    self._add_curse_to_deck(uuid=f"neow-curse-{self.floor}")
                return self._apply_neow_bonus(str(option.get("bonus") or "THREE_CARDS"))
            return self.state()
        if self.phase == "CARD_REWARD":
            if action.get("kind") == "card_reward":
                reward_index = action.get("reward_index")
                card_index = action.get("card_index")
                chosen: CardInstance | None = None
                if reward_index is not None and card_index is not None:
                    reward_index = int(reward_index)
                    card_index = int(card_index)
                    if 0 <= reward_index < len(self.reward_card_bundles) and 0 <= card_index < len(self.reward_card_bundles[reward_index]):
                        chosen = self.reward_card_bundles[reward_index][card_index]
                        self.reward_card_bundles.pop(reward_index)
                else:
                    index = int(action.get("choice_index", 0))
                    flat_cards = [(bundle_index, bundle_card_index, card) for bundle_index, bundle in enumerate(self.reward_card_bundles) for bundle_card_index, card in enumerate(bundle)]
                    if 0 <= index < len(flat_cards):
                        reward_index, _, chosen = flat_cards[index]
                        self.reward_card_bundles.pop(reward_index)
                if chosen is not None:
                    self._add_card_to_deck(chosen.card_id, upgrades=chosen.upgrades, uuid=f"deck-{self.floor}-{chosen.card_id}")
                    self._refresh_reward_cards()
                return self.state()
            if action.get("kind") == "reward_gold" and self.reward_gold_piles:
                reward_index = int(action.get("reward_index", 0))
                if 0 <= reward_index < len(self.reward_gold_piles):
                    amount = self.reward_gold_piles.pop(reward_index)
                    self._gain_gold(amount)
                    self.reward_gold = sum(self.reward_gold_piles)
                return self.state()
            if action.get("kind") == "reward_relic":
                reward_index = int(action.get("reward_index", 0))
                if 0 <= reward_index < len(self.reward_relics):
                    relic = self.reward_relics.pop(reward_index)
                    self._obtain_relic(relic)
                    if self._open_bottle_card_select(str(relic.get("relic_id") or "")):
                        return self.state()
                return self.state()
            if action.get("kind") == "reward_potion":
                reward_index = int(action.get("reward_index", 0))
                if 0 <= reward_index < len(self.reward_potions):
                    potion = self.reward_potions.pop(reward_index)
                    self._add_potion_if_space(potion)
                return self.state()
            if action.get("kind") == "reward_key" and self.reward_emerald_key:
                self.keys.add("emerald")
                self.reward_emerald_key = False
                return self.state()
            if action.get("kind") == "skip" and any(relic.get("relic_id") == "Singing Bowl" for relic in self.relics) and self.reward_card_bundles:
                self.player.max_hp += 2
                self.player.current_hp += 2
            self.reward_card_bundles = []
            self.reward_cards = []
            self.reward_gold = 0
            self.reward_gold_piles = []
            self.reward_emerald_key = False
            self.reward_relics = []
            self.reward_potions = []
            self.reward_close_required = False
            if self.reward_context == "NEOW":
                self.reward_context = None
                return self._complete_neow()
            if self.reward_context == "REST":
                self.reward_context = None
                self._enter_map()
                return self.state()
            if self.floor in {16, 33}:
                self._enter_boss_treasure_room()
                return self.state()
            if self.floor == 50:
                if self.enable_act4_keys and {"ruby", "sapphire", "emerald"}.issubset(self.keys):
                    self._advance_floor()
                else:
                    self.phase = "COMPLETE"
                return self.state()
            self._advance_floor()
            return self.state()
        if self.phase == "CARD_SELECT":
            index = int(action.get("target_index", action.get("choice_index", 0)) or 0)
            if self.card_select_context == "LIBRARY_OBTAIN":
                choice_index = int(action.get("choice_index", 0) or 0)
                if 0 <= choice_index < len(self.card_select_generated_cards):
                    selected = self.card_select_generated_cards[choice_index]
                    self._add_card_to_deck(selected.card_id, upgrades=selected.upgrades, uuid=f"library-{self.floor}-{selected.card_id}")
                self.card_select_options = []
                self.card_select_generated_cards = []
                self.card_select_context = None
                self.card_select_count = 0
                self.card_select_available_indexes = []
                self.card_select_selected_indexes = []
                self._advance_floor()
                return self.state()
            if self.card_select_context == "TRANSFORM_UPGRADE":
                if index in self.card_select_available_indexes and 0 <= index < len(self.deck):
                    self.card_select_selected_indexes.append(index)
                    self.card_select_available_indexes = [
                        candidate
                        for candidate in self.card_select_available_indexes
                        if candidate != index
                    ]
                    self.card_select_count -= 1
                if self.card_select_count > 0:
                    self._open_card_select("TRANSFORM_UPGRADE", self.card_select_count)
                    return self.state()
                selected_entries = [
                    (deck_index, self.deck[deck_index].card_id)
                    for deck_index in self.card_select_selected_indexes
                    if 0 <= deck_index < len(self.deck)
                ]
                for deck_index, _ in sorted(selected_entries, key=lambda item: item[0], reverse=True):
                    self.deck.pop(deck_index)
                for selection_order, (_, removed_card_id) in enumerate(selected_entries):
                    transformed = self._transformed_card_from_rng(self.randoms.misc, removed_card_id)
                    self._add_card_to_deck(
                        transformed.card_id,
                        upgrades=max(1, transformed.upgrades),
                        uuid=f"astrolabe-{self.floor}-{selection_order}-{removed_card_id}",
                    )
                self.card_select_options = []
                self.card_select_context = None
                self.card_select_count = 0
                self.card_select_available_indexes = []
                self.card_select_selected_indexes = []
                self._transition_to_next_act()
                return self.state()
            if 0 <= index < len(self.deck):
                if self.card_select_context == "NEOW_REMOVE":
                    self.deck.pop(index)
                elif self.card_select_context == "NEOW_UPGRADE":
                    self.deck[index].upgrades += 1
                elif self.card_select_context == "NEOW_TRANSFORM":
                    removed = self.deck.pop(index)
                    card = self._transformed_card_from_rng(self.randoms.neow, removed.card_id)
                    self._add_card_to_deck(card.card_id, upgrades=card.upgrades, uuid=f"neow-transform-{self.floor}-{index}")
                elif self.card_select_context == "EVENT_REMOVE":
                    self.deck.pop(index)
                elif self.card_select_context == "EVENT_UPGRADE":
                    self.deck[index].upgrades += 1
                elif self.card_select_context == "EVENT_TRANSFORM":
                    removed = self.deck.pop(index)
                    card = self._transformed_card_from_rng(self.randoms.misc, removed.card_id)
                    self._add_card_to_deck(card.card_id, upgrades=card.upgrades, uuid=f"event-transform-{self.floor}-{index}")
                elif self.card_select_context == "BONFIRE_SPIRITS":
                    offered = self.deck.pop(index)
                    rarity = offered.card_def.rarity
                    if rarity == "CURSE":
                        if not self._has_relic("Spirit Poop"):
                            self._obtain_relic(make_relic("Spirit Poop"))
                    elif rarity == "BASIC":
                        pass
                    elif rarity in {"COMMON", "SPECIAL"}:
                        self._heal_run(5)
                    elif rarity == "UNCOMMON":
                        self._heal_run(10)
                    elif rarity == "RARE":
                        self.player.max_hp += 10
                        self._heal_run(self.player.max_hp)
                elif self.card_select_context == "BOTTLE_REWARD":
                    relic = self._relic(str(self.pending_bottle_relic_id or ""))
                    if relic is not None:
                        relic["card_id"] = self.deck[index].card_id
                        relic["card_uuid"] = self.deck[index].uuid
                self.card_select_count -= 1
            if self.card_select_count > 0:
                self._open_card_select(str(self.card_select_context or ""), self.card_select_count)
                return self.state()
            self.card_select_options = []
            context = self.card_select_context
            self.card_select_context = None
            self.pending_bottle_relic_id = None
            self.card_select_count = 0
            self.card_select_available_indexes = []
            self.card_select_selected_indexes = []
            if context and str(context).startswith("NEOW_"):
                return self._complete_neow()
            if context == "BOTTLE_REWARD":
                self._advance_floor()
                return self.state()
            self._advance_floor()
            return self.state()
        if self.phase == "BOSS_RELIC":
            index = int(action.get("choice_index", 0))
            if 0 <= index < len(self.boss_relic_options):
                self._obtain_relic(dict(self.boss_relic_options[index]))
            self.boss_relic_options = []
            if self.phase != "BOSS_RELIC":
                return self.state()
            self._transition_to_next_act()
            return self.state()
        if self.phase == "MAP":
            self._advance_to_node(str(action.get("node_id") or action.get("symbol") or action.get("name") or "M"))
            return self.state()
        if self.phase == "CAMPFIRE":
            if action.get("name") == "REST":
                self._heal_run(max(1, int(self.player.max_hp * 0.3)))
                if any(relic.get("relic_id") == "Eternal Feather" for relic in self.relics):
                    self._heal_run(3 * (len(self.deck) // 5))
                if self._has_relic("Regal Pillow"):
                    self._heal_run(15)
                if self._has_relic("Dream Catcher"):
                    reward_count = 3 + (1 if self._has_relic("Question Card") else 0)
                    if self._has_relic("Busted Crown"):
                        reward_count = max(1, reward_count - 2)
                    self.phase = "CARD_REWARD"
                    self.reward_context = "REST"
                    self.reward_close_required = False
                    self.reward_card_bundles = [self._roll_card_reward(count=reward_count, room="R")]
                    self._refresh_reward_cards()
                    self.reward_gold = 0
                    self.reward_gold_piles = []
                    self.reward_emerald_key = False
                    self.reward_relics = []
                    self.reward_potions = []
                    return self.state()
            elif action.get("name") == "SMITH":
                index = int(action["target_index"]) if action.get("target_index") is not None else self._first_upgradable_index()
                if index is not None:
                    self.deck[index].upgrades += 1
            elif action.get("name") in {"PURGE", "TOKE"}:
                index = int(action["target_index"]) if action.get("target_index") is not None else self._first_purge_index()
                if index is not None:
                    self.deck.pop(index)
            elif action.get("name") == "DIG":
                tea_set = next((relic for relic in self.relics if relic.get("relic_id") == "Ancient Tea Set"), None)
                if tea_set is not None:
                    tea_set["counter"] = 1
                self.phase = "CARD_REWARD"
                self.reward_context = "REST"
                self.reward_close_required = False
                self.reward_card_bundles = []
                self.reward_cards = []
                self.reward_gold = 0
                self.reward_gold_piles = []
                self.reward_emerald_key = False
                self.reward_relics = [self._roll_relic()]
                self.reward_potions = []
                return self.state()
            elif action.get("name") == "LIFT":
                girya = self._relic("Girya")
                if girya is not None:
                    girya["counter"] = min(3, max(0, int(girya.get("counter", 0))) + 1)
            elif action.get("name") == "RECALL":
                self.keys.add("ruby")
            tea_set = next((relic for relic in self.relics if relic.get("relic_id") == "Ancient Tea Set"), None)
            if tea_set is not None:
                tea_set["counter"] = 1
            self._enter_map()
            return self.state()
        if self.phase == "SHOP":
            item_kind = action.get("item_kind")
            price = int(action.get("price", 0) or 0)
            if item_kind == "leave":
                self._enter_map()
                return self.state()
            if self.gold >= price:
                self.gold -= price
                if price > 0 and (maw_bank := self._relic("Maw Bank")) is not None:
                    maw_bank["counter"] = 0
                if item_kind == "card" and action.get("card"):
                    self._add_card_to_deck(action["card"]["card_id"], upgrades=action["card"].get("upgrades", 0), uuid=f"shop-{self.floor}-{action['card']['card_id']}")
                    self._remove_shop_item(action)
                elif item_kind == "relic" and action.get("relic"):
                    self._obtain_relic(dict(action["relic"]))
                    self._remove_shop_item(action)
                elif item_kind == "potion" and action.get("potion_id"):
                    self._add_potion_if_space(make_potion(str(action["potion_id"]), price=0))
                    self._remove_shop_item(action)
                elif item_kind == "purge":
                    index = int(action["target_index"]) if action.get("target_index") is not None else self._first_purge_index()
                    if index is not None:
                        self.deck.pop(index)
                    self.shop_remove_count += 1
                    self.shop_items = [item for item in self.shop_items if item.get("item_kind") != "purge"]
            return self.state()
        if self.phase == "TREASURE":
            if action.get("name") == "OPEN_CHEST":
                self.treasure_options = []
                self._enter_chest()
                return self.state()
            self.treasure_options = []
            self._enter_map()
            return self.state()
        if self.phase == "EVENT":
            event_id = str(action.get("event_id") or "")
            name = str(action.get("name") or action.get("label") or "")
            if event_id == "Big Fish" and name in {"Banana", "BANANA"}:
                self._heal_run(max(1, self.player.max_hp // 3))
            elif event_id == "Big Fish" and name in {"Donut", "DONUT"}:
                self.player.max_hp += 5
                self.player.current_hp += 5
            elif event_id == "Big Fish" and name in {"Box", "BOX"}:
                self._obtain_relic(self._roll_screenless_relic_of_tier(self._roll_relic_tier_for_act(self.act)))
                self._add_curse_to_deck("Regret", uuid=f"big-fish-box-{self.floor}")
            elif event_id == "Golden Idol" and name in {"Take", "Take Damage", "TAKE_GOLDEN_IDOL"}:
                if not any(relic.get("relic_id") == "Golden Idol" for relic in self.relics):
                    self._obtain_relic(make_relic("Golden Idol"))
                self.event_options = [
                    {"kind": "event", "event_id": event_id, "name": "Outrun", "label": "Outrun", "choice_index": 2},
                    {"kind": "event", "event_id": event_id, "name": "Smash", "label": "Smash", "choice_index": 3},
                    {"kind": "event", "event_id": event_id, "name": "Hide", "label": "Hide", "choice_index": 4},
                ]
                return self.state()
            elif event_id == "Golden Idol" and name in {"Outrun", "Take Wound"}:
                self.deck.append(make_card("Injury", uuid=f"golden-idol-injury-{self.floor}"))
                self._advance_floor()
                return self.state()
            elif event_id == "Golden Idol" and name in {"Smash", "Lose Max HP"}:
                self._lose_run_hp(max(1, int(self.player.max_hp * 0.25)))
                self._advance_floor()
                return self.state()
            elif event_id == "Golden Idol" and name in {"Hide"}:
                self.player.max_hp = max(1, self.player.max_hp - max(1, int(self.player.max_hp * 0.1)))
                self.player.current_hp = min(self.player.current_hp, self.player.max_hp)
                self._advance_floor()
                return self.state()
            elif event_id == "Shining Light" and name in {"Entered Light", "UPGRADE_TWO"}:
                self._lose_run_hp(max(1, int(self.player.max_hp * 0.2)))
                if self.phase == "GAME_OVER":
                    return self.state()
                upgradeable = [
                    idx for idx, card in enumerate(self.deck)
                    if _card_can_upgrade(card)
                ]
                java_collections_shuffle(upgradeable, self.randoms.misc.random_long())
                for index in upgradeable[:2]:
                    self.deck[index].upgrades += 1
            elif event_id == "Golden Shrine" and name in {"Prayed", "GAIN_GOLD"}:
                self._gain_gold(50 if self.ascension_level >= 15 else 100)
            elif event_id == "Golden Shrine" and name in {"Desecrated", "GAIN_GOLD_CURSE"}:
                self._gain_gold(275)
                self._add_curse_to_deck("Regret", uuid=f"golden-shrine-regret-{self.floor}")
            elif event_id == "The Divine Fountain" and name in {"Drank", "DRANK"}:
                self.deck = [card for card in self.deck if card.card_def.card_type != "CURSE"]
            elif event_id == "The Cleric" and name in {"Healed", "HEAL"}:
                self.gold = max(0, self.gold - 35)
                self._heal_run(int(self.player.max_hp * 0.25))
            elif event_id == "The Cleric" and name in {"Card Removal", "PURGE"}:
                self.gold = max(0, self.gold - (75 if self.ascension_level >= 15 else 50))
                self._open_card_select("EVENT_REMOVE", 1)
                return self.state()
            elif event_id == "Living Wall" and name in {"Forget", "REMOVE"}:
                self._open_card_select("EVENT_REMOVE", 1)
                return self.state()
            elif event_id == "Living Wall" and name in {"Change", "TRANSFORM"}:
                self._open_card_select("EVENT_TRANSFORM", 1)
                return self.state()
            elif event_id == "Living Wall" and name in {"Grow", "UPGRADE"}:
                self._open_card_select("EVENT_UPGRADE", 1)
                return self.state()
            elif event_id == "Ancient Writing" and name == "Elegance":
                self._open_card_select("EVENT_REMOVE", 1)
                return self.state()
            elif event_id == "Ancient Writing" and name == "Simplicity":
                for card in self.deck:
                    if card.card_id in {"Strike_R", "Defend_R"}:
                        card.upgrades = max(card.upgrades, 1)
            elif event_id == "Cursed Tome":
                choice_index = int(action.get("choice_index", 0))
                phase = int(self.event_state.get("phase", 0))
                if choice_index == 0:
                    self.event_state["phase"] = phase + 1
                    self._set_cursed_tome_options()
                    return self.state()
                if choice_index == 1:
                    self._advance_floor()
                    return self.state()
                if choice_index in {2, 3, 4}:
                    self._lose_run_hp(max(0, phase))
                    if self.phase == "GAME_OVER":
                        return self.state()
                    self.event_state["phase"] = phase + 1
                    self._set_cursed_tome_options()
                    return self.state()
                if choice_index == 5:
                    self._lose_run_hp(15 if self.ascension_level >= 15 else 10)
                    if self.phase == "GAME_OVER":
                        return self.state()
                    roll = int(self.randoms.misc.random(2))
                    relic_id = ["Necronomicon", "Enchiridion", "Nilry's Codex"][roll]
                    self.phase = "CARD_REWARD"
                    self.reward_card_bundles = []
                    self.reward_cards = []
                    self.reward_gold = 0
                    self.reward_gold_piles = []
                    self.reward_potions = []
                    self.reward_relics = [make_relic(relic_id)]
                    self.reward_emerald_key = False
                    return self.state()
                if choice_index == 6:
                    self._lose_run_hp(3)
                    if self.phase == "GAME_OVER":
                        return self.state()
                    self._advance_floor()
                    return self.state()
            elif event_id == "World of Goop" and name in {"Left Gold", "LOSE_GOLD"}:
                gold_loss = int(self.event_state.get("gold_loss", 0))
                self.gold = max(0, self.gold - max(0, gold_loss))
            elif event_id == "World of Goop" and name in {"Gather Gold", "LOSE_HP_GAIN_GOLD"}:
                self._lose_run_hp(11)
                self._gain_gold(75)
                if self.phase == "GAME_OVER":
                    return self.state()
            elif event_id == "Hypnotizing Colored Mushrooms" and name == "Fought Mushrooms":
                gold_amt = int(self.randoms.misc.random(20, 30))
                self._start_event_combat(["FungiBeast", "FungiBeast", "FungiBeast"], relic_id="Odd Mushroom", gold_gain=gold_amt)
                return self.state()
            elif event_id == "Hypnotizing Colored Mushrooms" and name == "Ignored":
                self._heal_run(max(1, int(self.player.max_hp * 0.25)))
                self._add_curse_to_deck("Parasite", uuid=f"mushrooms-{self.floor}")
            elif event_id == "Scrap Ooze" and name in {"Success", "LOSE_HP_GAIN_RELIC"}:
                self._lose_run_hp(5 if self.ascension_level >= 15 else 3)
                if self.phase == "GAME_OVER":
                    return self.state()
                attempts = int(self.event_state.get("counter", 0))
                roll = int(self.randoms.misc.random(99))
                relic_chance = attempts * 10 + 25
                if roll >= 99 - relic_chance:
                    self._obtain_relic(self._roll_screenless_relic_of_tier(self._roll_relic_tier_for_act(1)))
                else:
                    self.event_state["counter"] = attempts + 1
                    return self.state()
            elif event_id == "Mindbloom" and name == "Fight":
                self._start_event_boss_combat()
                return self.state()
            elif event_id == "Mindbloom" and name == "Gold":
                self._gain_gold(999)
            elif event_id == "Mindbloom" and name == "Heal":
                self.player.current_hp = self.player.max_hp
            elif event_id == "Mindbloom" and name == "Upgrade":
                for card in self.deck:
                    if card.card_def.card_type not in {"STATUS", "CURSE"}:
                        card.upgrades = max(card.upgrades, 1)
            elif event_id == "Falling":
                target_type = "ATTACK" if name == "Removed Attack" else "SKILL" if name == "Removed Skill" else "POWER"
                for index, card in enumerate(self.deck):
                    if card.card_def.card_type == target_type:
                        self.deck.pop(index)
                        break
            elif event_id == "Winding Halls" and name == "Max HP":
                self.player.max_hp = max(1, self.player.max_hp - 5)
                self.player.current_hp = min(self.player.current_hp, self.player.max_hp)
            elif event_id == "Winding Halls" and name == "Writhe":
                self._add_curse_to_deck("Writhe", uuid=f"winding-halls-{self.floor}")
            elif event_id == "Winding Halls" and name == "Embrace Madness":
                self._lose_run_hp(12)
                self._add_card_to_deck("Madness", uuid=f"winding-halls-madness-{self.floor}-0")
                self._add_card_to_deck("Madness", uuid=f"winding-halls-madness-{self.floor}-1")
            elif event_id == "Wing Statue" and name == "Card Removal":
                self._lose_run_hp(7)
                self._open_card_select("EVENT_REMOVE", 1)
                return self.state()
            elif event_id == "Wing Statue" and name == "Gained Gold":
                self._gain_gold(int(self.randoms.misc.random(50, 80)))
            elif event_id == "Drug Dealer" and name in {"Got JAX", "Obtain J.A.X."}:
                self._add_card_to_deck("J.A.X.", uuid=f"jax-{self.floor}")
            elif event_id == "Drug Dealer" and name in {"Inject Mutagens", "Became Test Subject"}:
                self._obtain_relic(make_relic("Mutagenic Strength"))
            elif event_id == "The Library" and name == "Heal":
                self._heal_run(max(1, self.player.max_hp // 3))
            elif event_id == "The Library" and name == "Read":
                self._open_library_card_select()
                return self.state()
            elif event_id == "Accursed Blacksmith" and name == "Forge":
                for card in self.deck:
                    if card.card_def.card_type not in {"STATUS", "CURSE"}:
                        card.upgrades = max(card.upgrades, 1)
            elif event_id == "Accursed Blacksmith" and name == "Rummage":
                if not self._has_relic("Warped Tongs"):
                    self._obtain_relic(make_relic("Warped Tongs"))
                self._add_curse_to_deck("Pain", uuid=f"blacksmith-{self.floor}")
            elif event_id == "The Mausoleum" and name == "Opened":
                self._obtain_relic(self._roll_relic())
                if self.ascension_level >= 15 or self.randoms.misc.random_boolean():
                    self._add_curse_to_deck("Writhe", uuid=f"mausoleum-{self.floor}")
            elif event_id == "Masked Bandits" and name == "Paid Fearfully":
                self.gold = 0
            elif event_id == "Masked Bandits" and name == "Fought Bandits":
                self._start_event_combat(["Bear", "Pointy", "Romeo"], relic_id="Red Mask", gold_gain=222)
                return self.state()
            elif event_id == "Vampires" and name == "Accepted":
                self.player.max_hp = max(1, int(self.player.max_hp * 0.7))
                self.player.current_hp = min(self.player.current_hp, self.player.max_hp)
                self.deck = [card for card in self.deck if card.card_id != "Strike_R"]
                for index in range(5):
                    self._add_card_to_deck("Bite", uuid=f"vampires-{self.floor}-{index}")
            elif event_id == "Ghosts" and name == "Accepted":
                self.player.max_hp = max(1, self.player.max_hp // 2)
                self.player.current_hp = min(self.player.current_hp, self.player.max_hp)
                for index in range(5):
                    self._add_card_to_deck("Apparition", uuid=f"apparition-{self.floor}-{index}")
            elif event_id == "Duplicator" and name == "Duplicated" and self.deck:
                source = self.deck[0]
                self._add_card_to_deck(source.card_id, upgrades=source.upgrades, uuid=f"duplicator-{self.floor}-{source.card_id}")
            elif event_id == "N'loth" and name == "Gave Relic" and len(self.relics) > 1:
                self.relics.pop(1)
                self._obtain_relic(self._roll_relic())
            elif event_id == "Note For Yourself" and name == "Took Card":
                self._add_card_to_deck(self.note_for_yourself_card_id, uuid=f"note-for-yourself-{self.floor}")
                self._open_card_select("EVENT_REMOVE", 1)
                return self.state()
            elif event_id == "The Joust" and name == "Murderer":
                self.gold = max(0, self.gold - 50)
                if self.randoms.event.random() < 0.7:
                    self._gain_gold(250)
            elif event_id == "The Joust" and name == "Owner":
                self.gold = max(0, self.gold - 50)
                if self.randoms.event.random() < 0.3:
                    self._gain_gold(250)
            elif event_id == "Designer In-Spire" and name == "Adjusted":
                self.gold = max(0, self.gold - 40)
                index = self._first_upgradable_index()
                if index is not None:
                    self.deck[index].upgrades += 1
            elif event_id == "Designer In-Spire" and name == "Cleaned Up":
                self.gold = max(0, self.gold - 50)
                index = self._first_purge_index()
                if index is not None:
                    self.deck.pop(index)
            elif event_id == "Designer In-Spire" and name == "Full Service":
                self.gold = max(0, self.gold - 75)
                index = self._first_upgradable_index()
                if index is not None:
                    self.deck[index].upgrades += 1
                index = self._first_purge_index()
                if index is not None:
                    self.deck.pop(index)
            elif event_id == "Face Trader" and name == "Touched":
                self._gain_gold(75)
                self._lose_run_hp(max(1, int(self.player.max_hp * 0.10)))
            elif event_id == "Face Trader" and name == "Traded":
                self._obtain_relic(self._roll_relic())
            elif event_id == "Forgotten Altar" and name == "Shed Blood":
                self._lose_run_hp(max(1, int(self.player.max_hp * 0.25)))
                self.player.max_hp += 5
                self.player.current_hp += 5
            elif event_id == "Forgotten Altar" and name == "Smashed Altar":
                if any(item.get("relic_id") == "Golden Idol" for item in self.relics):
                    self.relics = [item for item in self.relics if item.get("relic_id") != "Golden Idol"]
                    self._obtain_relic(self._roll_relic())
                else:
                    self._add_curse_to_deck("Decay", uuid=f"forgotten-altar-{self.floor}")
            elif event_id == "Lab" and name == "Obtained Potions":
                self._add_random_potion_reward(count=3)
            elif event_id == "Match and Keep" and name == "Played":
                cards = self._consume_match_and_keep_rng()
                for index, reward in enumerate(cards[:2]):
                    self._add_card_to_deck(reward.card_id, upgrades=reward.upgrades, uuid=f"match-and-keep-{self.floor}-{index}")
            elif event_id == "Purifier" and name in {"Purged", "REMOVE"}:
                self._open_card_select("EVENT_REMOVE", 1)
                return self.state()
            elif event_id == "The Moai Head" and name == "Jumped":
                self._heal_run(self.player.max_hp)
                self.player.max_hp = max(1, self.player.max_hp - 12)
                self.player.current_hp = min(self.player.current_hp, self.player.max_hp)
            elif event_id == "The Moai Head" and name == "Offered Golden Idol":
                if any(item.get("relic_id") == "Golden Idol" for item in self.relics):
                    self.relics = [item for item in self.relics if item.get("relic_id") != "Golden Idol"]
                    self.player.max_hp += 10
                    self.player.current_hp += 10
                else:
                    self._heal_run(20)
            elif event_id == "Sensory Stone" and name == "Recall":
                self._add_colorless_cards_to_deck(1, prefix="sensory-stone-recall")
            elif event_id == "Sensory Stone" and name == "Remember":
                self._add_colorless_cards_to_deck(2, prefix="sensory-stone-remember")
                self._lose_run_hp(5)
            elif event_id == "Sensory Stone" and name == "Live Forever":
                self._add_colorless_cards_to_deck(3, prefix="sensory-stone-live-forever")
                self._lose_run_hp(10)
            elif event_id == "The Woman in Blue" and name.startswith("Bought"):
                amount = 1 if "1" in name else 2 if "2" in name else 3
                self._open_potion_reward_screen(count=amount, context="EVENT")
                return self.state()
            elif event_id == "Transmogrifier" and name == "Transformed":
                self._open_card_select("EVENT_TRANSFORM", 1)
                return self.state()
            elif event_id == "Upgrade Shrine" and name == "Upgraded":
                self._open_card_select("EVENT_UPGRADE", 1)
                return self.state()
            elif event_id == "Wheel of Change" and name == "Spun":
                result = int(self.randoms.misc.random(5))
                if result == 0:
                    self._gain_gold(self.act * 100)
                elif result == 1:
                    relic = self._roll_screenless_relic_of_tier(self._roll_relic_tier_for_act(self.act))
                    self._open_relic_reward_screen(relic, context="EVENT")
                    return self.state()
                elif result == 2:
                    self._heal_run(self.player.max_hp)
                elif result == 3:
                    self._add_curse_to_deck("Decay", uuid=f"wheel-{self.floor}")
                elif result == 4:
                    self._open_card_select("EVENT_REMOVE", 1)
                    return self.state()
                else:
                    self._lose_run_hp(max(1, int(self.player.max_hp * (0.15 if self.ascension_level >= 15 else 0.10))))
                    if self.phase == "GAME_OVER":
                        return self.state()
            elif event_id == "Pleading Vagrant" and name == "Gave Gold":
                self.gold = max(0, self.gold - 85)
                self._obtain_relic(self._roll_relic())
            elif event_id == "Pleading Vagrant" and name == "Robbed":
                self._add_curse_to_deck(uuid=f"pleading-vagrant-{self.floor}")
                self._obtain_relic(self._roll_relic())
            elif event_id == "Dead Adventurer" and name == "Searched":
                phase = int(self.event_state.get("phase", 0))
                rewards = list(self.event_state.get("rewards", [0, 1, 2]))
                encounter_chance = phase * 25 + (35 if self.ascension_level >= 15 else 25)
                did_encounter = int(self.randoms.misc.random(99)) < encounter_chance
                if did_encounter:
                    gold_gain = int(self.randoms.misc.random(25, 35))
                    relic_id = None
                    for reward in rewards[phase:]:
                        if reward == 0:
                            gold_gain += 30
                        elif reward == 2 and relic_id is None:
                            relic_id = self._roll_screenless_relic_of_tier(self._roll_relic_tier_for_act(1)).get("relic_id")
                    encounter = str(self.event_state.get("encounter") or "Gremlin Nob")
                    if encounter == "Three Sentries":
                        monster_ids = ["Sentry", "Sentry", "Sentry"]
                    elif encounter == "Lagavulin":
                        monster_ids = ["LagavulinEvent"]
                    else:
                        monster_ids = ["GremlinNob"]
                    self._start_event_combat(monster_ids, relic_id=relic_id, gold_gain=gold_gain, elite=True)
                    return self.state()
                reward = rewards[phase] if phase < len(rewards) else 1
                if reward == 0:
                    self._gain_gold(30)
                elif reward == 2:
                    self._obtain_relic(self._roll_screenless_relic_of_tier(self._roll_relic_tier_for_act(1)))
                self.event_state["phase"] = phase + 1
                return self.state()
            elif event_id == "Ominous Forge" and name == "Forge":
                self._open_card_select("EVENT_UPGRADE", 1)
                return self.state()
            elif event_id == "Ominous Forge" and name == "Rummage":
                self._add_curse_to_deck("Pain", uuid=f"ominous-forge-{self.floor}")
                if not self._has_relic("Warped Tongs"):
                    self._obtain_relic(make_relic("Warped Tongs"))
            elif event_id == "The Ssssserpent" and name == "Agreed":
                self._add_curse_to_deck("Doubt", uuid=f"liars-game-{self.floor}")
                self._gain_gold(150 if self.ascension_level >= 15 else 175)
            elif event_id == "Mysterious Sphere" and name == "Fought Orb Walkers":
                relic_id = self._roll_relic().get("relic_id")
                self._start_event_combat(["OrbWalker", "OrbWalker"], relic_id=relic_id)
                return self.state()
            elif event_id == "Colosseum" and name == "Fought":
                relic_id = self._roll_relic().get("relic_id")
                self._start_event_combat(["SlaverBlue", "SlaverRed", "Taskmaster"], relic_id=relic_id, gold_gain=100)
                return self.state()
            elif event_id in {"We Meet Again!", "WeMeetAgain"} and name in {"Gave Potion", "Gave Gold", "Gave Card"}:
                if name == "Gave Potion":
                    potion_idx = self.event_state.get("potion_idx")
                    if potion_idx is not None and 0 <= int(potion_idx) < len(self.potions):
                        self.potions[int(potion_idx)] = PotionInstance()
                elif name == "Gave Gold":
                    gold_loss = int(self.event_state.get("gold", -1))
                    if gold_loss > 0:
                        self.gold = max(0, self.gold - gold_loss)
                elif name == "Gave Card":
                    card_idx = self.event_state.get("card_idx")
                    if card_idx is not None and 0 <= int(card_idx) < len(self.deck):
                        self.deck.pop(int(card_idx))
                self._obtain_relic(self._roll_screenless_relic_of_tier(self._roll_relic_tier_for_act(self.act)))
            elif event_id == "Tomb of Lord Red Mask" and name == "Got Gold":
                self._gain_gold(222)
            elif event_id == "Tomb of Lord Red Mask" and name == "Paid":
                self.gold = 0
                self._obtain_relic(make_relic("Red Mask"))
            self.current_event_id = None
            self.event_state = {}
            self._enter_map()
            return self.state()
        if self.phase == "CHEST":
            gold_amount = int(action.get("gold_amount", self.chest_gold_amount) or 0)
            if gold_amount > 0:
                self._gain_gold(gold_amount)
            if action.get("item_kind") == "sapphire_key":
                self.keys.add("sapphire")
            elif action.get("item_kind") == "relic":
                if any(relic.get("relic_id") == "Cursed Key" for relic in self.relics):
                    self._add_curse_to_deck(uuid=f"cursed-key-{self.floor}")
                self._obtain_relic(dict(action))
                matryoshka = next((relic for relic in self.relics if relic.get("relic_id") == "Matryoshka" and int(relic.get("counter", 0)) > 0), None)
                if matryoshka is not None:
                    matryoshka["counter"] = int(matryoshka.get("counter", 0)) - 1
                    self._obtain_relic(self._roll_relic())
            self.chest_options = []
            self._enter_map()
            return self.state()
        return self.state()

    def state(self) -> dict[str, Any]:
        if self.phase == "COMBAT":
            return self.combat.to_spirecomm_state()
        return {
            "act": self.act,
            "act_boss": self.act_boss,
            "ascension_level": self.ascension_level,
            "character": "IRONCLAD",
            "choice_available": self.phase in {"NEOW", "CARD_REWARD", "CARD_SELECT", "BOSS_RELIC", "MAP", "CAMPFIRE", "SHOP", "EVENT", "TREASURE", "CHEST"},
            "choice_list": self._choice_list(),
            "combat_state": None,
            "commands": {
                "cancel": False,
                "end": False,
                "play": False,
                "potion": False,
                "proceed": self.phase == "CARD_REWARD",
            },
            "current_hp": self.player.current_hp,
            "deck": [card_to_spirecomm(card) for card in self.deck],
            "floor": self.floor,
            "gold": self.gold,
            "in_combat": False,
            "keys": sorted(self.keys),
            "max_hp": self.player.max_hp,
            "potions": potions_to_spirecomm(self.potions),
            "relics": self.relics,
            "room_phase": "COMPLETE" if self.phase in {"NEOW", "CARD_REWARD", "CARD_SELECT", "BOSS_RELIC", "MAP", "CAMPFIRE", "SHOP", "EVENT", "TREASURE", "CHEST"} else self.phase,
            "room_type": self._room_type(),
            "screen": self.phase,
            "screen_up": self.phase in {"NEOW", "CARD_REWARD", "CARD_SELECT", "BOSS_RELIC", "MAP", "CAMPFIRE", "SHOP", "EVENT", "TREASURE", "CHEST"},
            "seed": self.seed,
        }

    def _choice_list(self) -> list[dict[str, Any]]:
        if self.phase == "CARD_REWARD":
            items: list[dict[str, Any]] = [card_to_spirecomm(card) for card in self.reward_cards]
            items.extend(dict(relic) for relic in self.reward_relics)
            items.extend(potion.to_spirecomm() for potion in self.reward_potions if potion.can_use)
            for amount in self.reward_gold_piles:
                items.append({"kind": "reward_gold", "name": "GOLD", "amount": amount})
            if self.reward_emerald_key:
                items.append({"kind": "reward_key", "name": "KEY", "key": "emerald"})
            return items
        if self.phase == "NEOW":
            return list(self.neow_options)
        if self.phase == "BOSS_RELIC":
            items = list(self.boss_relic_options)
            items.append({
                "kind": "boss_relic",
                "name": "SKIP",
                "relic_id": "SKIP",
                "choice_index": len(items),
            })
            return items
        if self.phase == "MAP":
            return list(self.map_options)
        if self.phase == "CAMPFIRE":
            return list(self.campfire_options)
        if self.phase == "SHOP":
            return list(self.shop_items)
        if self.phase == "EVENT":
            return list(self.event_options)
        if self.phase == "TREASURE":
            return list(self.treasure_options)
        if self.phase == "CARD_SELECT":
            return list(self.card_select_options)
        if self.phase == "CHEST":
            return list(self.chest_options)
        return []

    def _room_type(self) -> str:
        return {
            "MAP": "MapRoom",
            "NEOW": "EventRoom",
            "CARD_SELECT": "CardSelectRoom",
            "BOSS_RELIC": "TreasureRoomBoss",
            "CAMPFIRE": "RestRoom",
            "SHOP": "ShopRoom",
            "TREASURE": "TreasureRoom",
            "EVENT": "EventRoom",
            "CHEST": "TreasureRoom",
            "CARD_REWARD": "MonsterRoom",
            "COMPLETE": "VictoryRoom",
            "GAME_OVER": "GameOverRoom",
        }.get(self.phase, "MonsterRoom")
