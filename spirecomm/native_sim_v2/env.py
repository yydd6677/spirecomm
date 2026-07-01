from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any

from spirecomm.native_sim.schema import CardInstance, MonsterState, PlayerState, PotionInstance
from spirecomm.native_sim_v2.combat_core import (
    card_exhausts_on_use,
    end_turn as v2_end_turn,
    play_card as v2_play_card,
    resolve_pending_delayed_reactions,
    resolve_after_use_card_move,
    resolve_card_select,
    step as v2_step,
)
from spirecomm.native_sim_v2.helpers_cards import CARD_LIBRARY, card_to_spirecomm, clone_card
from spirecomm.native_sim_v2.helpers_common import *
from spirecomm.native_sim_v2.helpers_combat import CombatHelpersMixin
from spirecomm.native_sim_v2.helpers_run import RunHelpersMixin
from spirecomm.native_sim_v2.monster_support import _set_move, monster_adjusted_damage
from spirecomm.native_sim_v2.monsters import choose_next_move, encounter_to_monsters, make_monster, roll_act1_encounter
from spirecomm.native_sim_v2.randoms import NativeRandomStreams, StsRandom
from spirecomm.native_sim_v2.run_core import (
    legal_actions as v2_run_legal_actions,
    start_combat as v2_run_start_combat,
    start_event_boss_combat as v2_run_start_event_boss_combat,
    state as v2_run_state,
    step as v2_run_step,
)
from spirecomm.native_sim_v2.serialize import combat_state as serialize_combat_state


@dataclass
class NativeCombatEnv(CombatHelpersMixin):
    seed: int
    ascension_level: int = 0
    floor: int = 1
    act: int = 1
    act_boss: str = "Hexaghost"
    elite: bool = False
    external_misc_rng: StsRandom | None = None
    external_potion_rng: StsRandom | None = None
    rng: StsRandom = field(init=False)
    ai_rng: StsRandom = field(init=False)
    monster_hp_rng: StsRandom = field(init=False)
    shuffle_rng: StsRandom = field(init=False)
    card_random_rng: StsRandom = field(init=False)
    misc_rng: StsRandom = field(init=False)
    potion_rng: StsRandom = field(init=False)
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
    panache_counter: int = 0
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
    card_select_source_indexes: list[int] = field(default_factory=list)
    pending_resolve_card: CardInstance | None = None
    pending_resolve_force_exhaust: bool = False
    pending_resolve_hex_dazed: int = 0
    pending_resolve_used_free_to_play_once: bool = False
    pending_resolve_status_effects: list[tuple[str, int]] = field(default_factory=list)
    pending_counter_damage: int = 0
    pending_juggernaut_damage: int = 0
    pending_after_use_direct_damage_all: int = 0
    pending_after_use_energy_gain: int = 0
    pending_monster_direct_damage: list[tuple[MonsterState, int]] = field(default_factory=list)
    pending_attack_replays: int = 0
    pending_attack_target_index: int | None = None
    pending_attack_relic_proc: bool = False
    pending_spore_cloud_player_turn_triggers: int = 0
    pending_spore_cloud_monster_turn_triggers: int = 0
    pending_monster_kill_triggers: int = 0
    pending_bottle_relic_id: str | None = None
    pending_unceasing_top_draw: bool = False
    pending_end_turn_resume: bool = False
    pending_start_turn_resume: bool = False
    pending_start_turn_autoplay: bool = False
    pending_start_turn_post_draw_resume: bool = False
    pending_start_turn_fire_breathing_damage: list[int] = field(default_factory=list)
    pending_start_turn_evolve_draws: int = 0
    processing_start_turn: bool = False
    processing_player_action: bool = False
    processing_end_turn_cleanup: bool = False
    processing_monster_turns: bool = False
    monster_turn_damage_draws_are_end_turn: bool = False
    end_turn_regular_discard_count: int = 0
    opening_innate_count: int = 0
    combust_hp_loss: int = 0
    pending_monster_block_gains: list[tuple[MonsterState, int]] = field(default_factory=list)
    pending_autoplay_cards: list[tuple[CardInstance, int, bool, int | None]] = field(default_factory=list)
    _next_uuid: int = 0
    _double_tap_active: bool = False
    _attack_pen_nib_active: bool = False
    _draw_triggered_shuffle: bool = False
    _processing_autoplay_cards: bool = False
    _red_skull_active: bool = False
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
            self.potion_rng = self.external_potion_rng if self.external_potion_rng is not None else start_random.copy()
            self.start_combat(elite=self.elite)

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
                    if monster.monster_id == "INVALID = 0":
                        continue
                    # Preserved Insect only trims the elite's opening HP; the
                    # underlying max HP stays intact so sleepers like Lagavulin
                    # can still heal back up with Regenerate.
                    monster.current_hp = max(1, int(monster.current_hp * 0.75))
            for index, monster in enumerate(self.monsters):
                monster.ai_state["ascension_level"] = self.ascension_level
                monster.ai_state["spawn_index"] = index
                setattr(monster, "_group_ref", self.monsters)
            self._apply_ascension_monster_scaling(elite=elite)
            for monster in self.monsters:
                if monster.alive:
                    if monster.ai_state.get("fixed_opening_move", 0) and not monster.move_history:
                        if monster.monster_id == "TheCollector":
                            self.ai_rng.random(99)
                        continue
                    choose_next_move(monster, self.ai_rng)
            self._apply_monster_prebattle_actions()
            for card in self.deck:
                card.cost_for_combat = None
                card.cost_for_turn = None
                card.free_to_play_once = False
                if card.card_id == "Blood for Blood":
                    card.misc = 0
            self._init_opening_draw_pile()
            self.discard_pile = []
            self.exhaust_pile = []
            self.hand = []
            self.turn = 0
            self.pending_after_use_direct_damage_all = 0
            self.hp_lost_this_combat = 0
            self.combust_hp_loss = 0
            self.pending_monster_block_gains = []
            self.pending_monster_direct_damage = []
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
            relic = self._relic("Incense Burner")
            if relic is not None and int(relic.get("counter", -1)) < 0:
                relic["counter"] = int(relic.get("counter", -1))
            relic = self._relic("Necronomicon")
            if relic is not None:
                # Lightspeed tracks Necronomicon with a battle-local boolean, so
                # carrying over the last combat's turn marker can incorrectly
                # suppress the first replay of a new fight.
                relic["counter"] = -1
            self._red_skull_active = False
            self._sync_red_skull_strength()
            self._apply_start_combat_relics()
            self._sync_red_skull_strength()
            self._start_opening_turn()

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
                    self._insert_temp_card_into_draw_pile("Wound")
            if self._has_relic("Ring of the Snake"):
                self.draw_cards(2)
            if self._has_relic("Twisted Funnel"):
                for monster in self.monsters:
                    self._apply_monster_power(monster, "Poison", 4)

    def start_player_turn(self) -> None:
            previous_processing_start_turn = self.processing_start_turn
            self.processing_start_turn = True
            try:
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
                if self._has_relic("Incense Burner") and self._advance_relic_counter("Incense Burner", 6):
                    self.player.add_power("Intangible", 1)
                if self.player.power("Barricade") <= 0 and self._has_relic("Calipers"):
                    self.player.block = max(0, self.player.block - 15)
                elif self.player.power("Barricade") <= 0:
                    self.player.block = 0
                if self.turn == 1 and self._has_relic("Horn Cleat"):
                    self.player.block += 14
                    if self.player.power("Juggernaut") > 0:
                        self._trigger_juggernaut(self.player.power("Juggernaut"))
                if self.turn == 2 and self._has_relic("Captain's Wheel"):
                    self.player.block += 18
                    if self.player.power("Juggernaut") > 0:
                        self._trigger_juggernaut(self.player.power("Juggernaut"))
                self._apply_happy_flower_start_of_turn()
                self.cards_discarded_this_turn = 0
                self.attack_played_this_turn = 0
                self.skills_played_this_turn = 0
                self.cards_played_this_turn = 0
                self.card_types_played_this_turn = set()
                if self.player.power("Self-Forming Clay Block") > 0:
                    gained = self.player.powers.pop("Self-Forming Clay Block")
                    self._gain_block(gained, defer_juggernaut=True, apply_block_modifiers=False)
                for monster in self.monsters:
                    if monster.monster_id == "Nemesis" and monster.ai_state.pop("intangible_next", 0):
                        monster.add_power("Intangible", 1)
                if self.player.power("Demon Form") > 0:
                    self.player.add_power("Strength", self.player.power("Demon Form"))
                if self.player.power("Ritual") > 0:
                    self.player.add_power("Strength", self.player.power("Ritual"))
                if self.player.power("Panache") > 0:
                    self.panache_counter = 5
                if self.player.power("Magnetism") > 0:
                    # Mirror the current lightspeed implementation: Magnetism is
                    # applied as a player power, but its start-of-turn card creation
                    # is still a no-op there.
                    pass
                self.pending_start_turn_autoplay = False
                self.pending_start_turn_post_draw_resume = False
                if self.player.power("Mayhem") > 0:
                    for _ in range(self.player.power("Mayhem")):
                        self._play_random_top_card(force_exhaust=False)
                    self.pending_start_turn_autoplay = bool(self.pending_autoplay_cards)
                self.pending_start_turn_resume = False
                self._resume_start_player_turn_after_autoplay()
            finally:
                self.processing_start_turn = previous_processing_start_turn

    def _resume_start_player_turn_after_autoplay(self) -> None:
            previous_processing_start_turn = self.processing_start_turn
            self.processing_start_turn = True
            try:
                if self.pending_start_turn_post_draw_resume:
                    self.pending_start_turn_post_draw_resume = False
                    self.pending_start_turn_resume = False
                    if self.player.power("Brutality") > 0:
                        brutality_amount = self.player.power("Brutality")
                        self._lose_hp(brutality_amount, self_damage=False)
                        if self.outcome != "UNDECIDED":
                            return
                        self.draw_cards(brutality_amount)
                    if self._has_relic("Warped Tongs"):
                        candidates = [card for card in self.hand if card.card_def.card_type not in {"STATUS", "CURSE"} and card.upgrades <= 0]
                        if candidates:
                            self._upgrade_combat_card(self.card_random_rng.choice(candidates))
                    return
                if self.pending_autoplay_cards and not self.pending_start_turn_autoplay:
                    self._drain_pending_autoplay_cards()
                    if self.card_select_context is not None or self.outcome != "UNDECIDED":
                        self.pending_start_turn_resume = True
                        return
                self.pending_start_turn_resume = False
                if self.player.power("Regen") > 0:
                    self._heal(self.player.power("Regen"))
                    self.player.add_power("Regen", -1)
                if self._has_relic("Brimstone"):
                    self.player.add_power("Strength", 2)
                    for monster in self.monsters:
                        if monster.alive:
                            monster.add_power("Strength", 1)
                if self._has_relic("Mercury Hourglass"):
                    self._deal_direct_damage_all(3)
                    self._resolve_pending_monster_kill_triggers()
                if not any(monster.alive for monster in self.monsters) and self.player.power("Brutality") > 0:
                    brutality_amount = self.player.power("Brutality")
                    self._lose_hp(brutality_amount, self_damage=False)
                    if self.outcome != "UNDECIDED":
                        return
                self._check_outcome()
                if self.outcome != "UNDECIDED":
                    return
                if self._resolve_pending_start_turn_fire_breathing_damage():
                    return
                if self.pending_start_turn_evolve_draws > 0:
                    deferred_draws = self.pending_start_turn_evolve_draws
                    self.pending_start_turn_evolve_draws = 0
                    self.draw_cards(
                        deferred_draws,
                        deferred_status_damage_all=self.pending_start_turn_fire_breathing_damage,
                    )
                # Monster-turn and end-turn kills (for example via thorns, Flame
                # Barrier, Combust, Mercury Hourglass, or The Bomb) should grant
                # Gremlin Horn / Spore Cloud before the new turn's normal draw.
                self._resolve_pending_monster_kill_triggers()
                self.draw_cards(
                    self._card_draw_per_turn(),
                    deferred_status_damage_all=self.pending_start_turn_fire_breathing_damage,
                )
                self._resolve_pending_start_turn_fire_breathing_damage()
                self._resolve_pending_juggernaut_damage()
                self._resolve_pending_monster_kill_triggers()
                if not any(monster.alive for monster in self.monsters) and self.player.power("Brutality") > 0:
                    brutality_amount = self.player.power("Brutality")
                    self._lose_hp(brutality_amount, self_damage=False)
                    if self.outcome != "UNDECIDED":
                        return
                self._check_outcome()
                if self.outcome != "UNDECIDED":
                    return
                if self.pending_start_turn_autoplay:
                    self.pending_start_turn_autoplay = False
                    if self.pending_autoplay_cards:
                        self._drain_pending_autoplay_cards()
                        if self.card_select_context is not None or self.outcome != "UNDECIDED":
                            self.pending_start_turn_post_draw_resume = True
                            self.pending_start_turn_resume = True
                            return
                if self.player.power("Brutality") > 0:
                    brutality_amount = self.player.power("Brutality")
                    self._lose_hp(brutality_amount, self_damage=False)
                    if self.outcome != "UNDECIDED":
                        return
                    self.draw_cards(brutality_amount)
                if self._has_relic("Warped Tongs"):
                    candidates = [card for card in self.hand if card.card_def.card_type not in {"STATUS", "CURSE"} and card.upgrades <= 0]
                    if candidates:
                        self._upgrade_combat_card(self.card_random_rng.choice(candidates))
            finally:
                self.processing_start_turn = previous_processing_start_turn

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
                requires_target = card.card_def.has_target and not (
                    (card.card_id == "Blind" and card.upgrades > 0)
                    or (card.card_id == "Trip" and card.upgrades > 0)
                )
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


    def _monster_take_turn(self, monster: MonsterState, monster_turn_index: int | None = None) -> int | None:
            starting_move = monster.move
            deferred_post_attack_discard_cards: list[CardInstance] = []
            monster.ai_state["roll_move_if_gone"] = starting_move in {
                "ACID_SLIME_M_CORROSIVE_SPIT",
                "ACID_SLIME_M_LICK",
                "ACID_SLIME_M_TACKLE",
                "ACID_SLIME_L_CORROSIVE_SPIT",
                "ACID_SLIME_L_LICK",
                "ACID_SLIME_L_TACKLE",
                "SPIKE_SLIME_M_FLAME_TACKLE",
                "SPIKE_SLIME_M_LICK",
                "SPIKE_SLIME_L_FLAME_TACKLE",
                "SPIKE_SLIME_L_LICK",
                "RED_LOUSE_BITE",
                "RED_LOUSE_GROW",
                "GREEN_LOUSE_BITE",
                "GREEN_LOUSE_SPIT_WEB",
                "FUNGI_BEAST_BITE",
                "FUNGI_BEAST_GROW",
                "JAW_WORM_CHOMP",
                "JAW_WORM_THRASH",
                "BLUE_SLAVER_RAKE",
                "BLUE_SLAVER_STAB",
                "RED_SLAVER_ENTANGLE",
                "RED_SLAVER_SCRAPE",
                "RED_SLAVER_STAB",
                "TASKMASTER_SCOURING_WHIP",
                "BYRD_PECK",
                "BYRD_SWOOP",
            }
            if monster.monster_id == "Byrd" and "Flight" in monster.powers:
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
                    prior_len = len(self.monsters)
                    self.monsters[monster_index] = child_monsters[0]
                    if len(child_monsters) > 1:
                        secondary_child = child_monsters[1]
                        trailing_large_acid = (
                            monster.move == "SPIKE_SLIME_L_SPLIT"
                            and any(
                                later is not monster
                                and later.monster_id == "AcidSlime_L"
                                and later.alive
                                for later in self.monsters[monster_index + 1 :]
                            )
                        )
                        if not trailing_large_acid:
                            secondary_child.ai_state["extra_roll_move_on_turn"] = True
                        if (
                            monster_index + 1 < len(self.monsters)
                            and (
                                self.monsters[monster_index + 1].monster_id == "INVALID = 0"
                                or self.monsters[monster_index + 1].is_gone
                                or not self.monsters[monster_index + 1].alive
                            )
                        ):
                            self.monsters[monster_index + 1] = secondary_child
                        else:
                            self.monsters.insert(monster_index + 1, secondary_child)
                        if len(self.monsters) > 4:
                            self.monsters = self.monsters[:4]
                        target_len = min(prior_len + 1, 4)
                        while len(self.monsters) < target_len:
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
                        for slot_index, current in enumerate(self.monsters):
                            current.ai_state["spawn_index"] = slot_index
                            setattr(current, "_group_ref", self.monsters)
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
                choose_next_move(monster, self.ai_rng)
                monster.ai_state["skip_end_round_roll"] = True
                return
            if monster.move in {"ACID_SLIME_M_CORROSIVE_SPIT", "SPIKE_SLIME_M_FLAME_TACKLE"}:
                deferred_post_attack_discard_cards.append(make_card("Slimed", uuid=self._new_uuid("Slimed")))
            if monster.move in {"ACID_SLIME_L_CORROSIVE_SPIT", "SPIKE_SLIME_L_FLAME_TACKLE"}:
                deferred_post_attack_discard_cards.append(make_card("Slimed", uuid=self._new_uuid("Slimed")))
                deferred_post_attack_discard_cards.append(make_card("Slimed", uuid=self._new_uuid("Slimed")))
            if monster.move == "HEXAGHOST_SEAR":
                deferred_post_attack_discard_cards.append(
                    make_card("Burn", upgrades=1 if self.turn > 8 else 0, uuid=self._new_uuid("Burn"))
                )
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
            if monster.move == "BRONZE_AUTOMATON_SPAWN_ORBS":
                for slot in (0, 2):
                    if slot >= len(self.monsters):
                        while len(self.monsters) <= slot:
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
                    existing = self.monsters[slot]
                    orb = make_monster("BronzeOrb", self.monster_hp_rng, self.ascension_level)
                    for power_id, amount in existing.powers.items():
                        orb.add_power(power_id, amount)
                    orb.block = existing.block
                    orb.add_power("Minion", 1)
                    if self._has_relic("Philosopher's Stone"):
                        orb.add_power("Strength", 1)
                    orb.ai_state["ascension_level"] = self.ascension_level
                    orb.ai_state["spawn_index"] = slot
                    self.monsters[slot] = orb
                for index, ally in enumerate(self.monsters):
                    ally.ai_state["spawn_index"] = index
                    setattr(ally, "_group_ref", self.monsters)
                choose_next_move(self.monsters[0], self.ai_rng)
                choose_next_move(self.monsters[2], self.ai_rng)
                _set_move(monster, "BRONZE_AUTOMATON_FLAIL")
                self.ai_rng.random(99)
                monster.ai_state["skip_end_round_roll"] = True
                return 2
            if monster.move == "BRONZE_AUTOMATON_BOOST":
                monster.add_power("Strength", 4 if self.ascension_level >= 4 else 3)
                monster.block += 12 if self.ascension_level >= 9 else 9
                last_boost_was_flail = bool(monster.ai_state.get("last_boost_was_flail", False))
                if last_boost_was_flail:
                    _set_move(monster, "BRONZE_AUTOMATON_HYPER_BEAM")
                    monster.ai_state["last_boost_was_flail"] = False
                else:
                    _set_move(monster, "BRONZE_AUTOMATON_FLAIL")
                    monster.ai_state["last_boost_was_flail"] = True
                self.ai_rng.random(99)
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
                    source_pile: list[CardInstance] | None = None
                    if self.draw_pile:
                        source_pile = self.draw_pile
                    elif self.discard_pile:
                        source_pile = self.discard_pile
                    stolen: CardInstance | None = None
                    if source_pile:
                        rarity_order = {"RARE": 3, "UNCOMMON": 2, "COMMON": 1}
                        target_priority = max((rarity_order.get(card.card_def.rarity, 0) for card in source_pile), default=0)
                        if target_priority > 0:
                            matching_indexes = [
                                index
                                for index, card in enumerate(source_pile)
                                if rarity_order.get(card.card_def.rarity, 0) == target_priority
                            ]
                            matching_indexes.sort(
                                key=lambda index: BRONZE_ORB_STASIS_SORT_INDEX_OVERRIDES.get(
                                    source_pile[index].card_id,
                                    COMBAT_CARD_SORT_INDEX.get(
                                        source_pile[index].card_id,
                                        len(COMBAT_CARD_SORT_INDEX),
                                    ),
                                )
                            )
                            chosen_list_index = int(self.card_random_rng.random(len(matching_indexes) - 1))
                            remove_index = matching_indexes[chosen_list_index]
                        else:
                            remove_index = int(self.card_random_rng.random(len(source_pile) - 1))
                        stolen = source_pile.pop(remove_index)
                    if stolen is not None:
                        monster.ai_state["stasis_card"] = stolen
                monster.ai_state["bronze_orb_used_stasis"] = 1
                choose_next_move(monster, self.ai_rng)
                monster.ai_state["skip_end_round_roll"] = True
                return
            if monster.move == "BRONZE_ORB_SUPPORT_BEAM":
                if len(self.monsters) > 1 and self.monsters[1].alive:
                    self.monsters[1].block += 12
                choose_next_move(monster, self.ai_rng)
                monster.ai_state["skip_end_round_roll"] = True
                return
            if monster.move == "LOOTER_ESCAPE" or monster.move == "MUGGER_ESCAPE":
                monster.is_gone = True
                monster.ai_state["escaping"] = True
                monster.ai_state["skip_end_round_roll"] = True
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
            if monster.move == "LAGAVULIN_STUN":
                return
            if monster.move == "LAGAVULIN_SIPHON_SOUL":
                amount = -2 if self.ascension_level >= 18 else -1
                self._apply_player_power("Dexterity", amount)
                self._apply_player_power("Strength", amount)
                return
            if monster.move == "JAW_WORM_BELLOW":
                monster.block += 6
                monster.add_power("Strength", 3)
            elif monster.move == "BYRD_STUNNED":
                _set_move(monster, "BYRD_HEADBUTT")
                self.ai_rng.random(99)
                monster.ai_state["skip_end_round_roll"] = True
                return
            elif monster.move == "BYRD_FLY":
                monster.add_power("Flight", 4 if self.ascension_level >= 17 else 3)
                return
            elif monster.move == "BYRD_CAW":
                monster.add_power("Strength", 1)
                return
            elif monster.move == "GREMLIN_LEADER_ENCOURAGE":
                strength_amount = 5 if self.ascension_level >= 18 else 4 if self.ascension_level >= 3 else 3
                block_amount = 10 if self.ascension_level >= 3 else 6
                self.ai_rng.random(0, 2)
                for ally in self.monsters[:3]:
                    if ally.alive and ally.monster_id != "INVALID = 0":
                        ally.add_power("Strength", strength_amount)
                        ally.block += block_amount
                monster.add_power("Strength", strength_amount)
                return
            elif monster.move == "SPIKER_GROW":
                monster.add_power("Thorns", 2)
                return
            elif monster.move == "THE_COLLECTOR_SPAWN":
                open_slots = []
                if len(self.monsters) > 1 and (self.monsters[1].monster_id == "INVALID = 0" or self.monsters[1].is_gone or not self.monsters[1].alive):
                    open_slots.append(1)
                if len(self.monsters) > 0 and (self.monsters[0].monster_id == "INVALID = 0" or self.monsters[0].is_gone or not self.monsters[0].alive):
                    open_slots.append(0)
                for slot in open_slots[:2]:
                    if sum(1 for ally in self._alive_monsters() if ally.monster_id == "TorchHead") >= 2:
                        break
                    summoned = make_monster("TorchHead", self.monster_hp_rng, self.ascension_level)
                    # Torch Heads are a game bug: construct() rolls HP once,
                    # then the spawn action rolls HP a second time and keeps
                    # that second value.
                    summoned = make_monster("TorchHead", self.monster_hp_rng, self.ascension_level)
                    if self._has_relic("Philosopher's Stone"):
                        summoned.add_power("Strength", 1)
                    choose_next_move(summoned, self.ai_rng)
                    summoned.ai_state["spawn_index"] = slot
                    self.monsters[slot] = summoned
                for index, ally in enumerate(self.monsters):
                    ally.ai_state["spawn_index"] = index
                    setattr(ally, "_group_ref", self.monsters)
                # The Collector's opening spawn is a fixed move that is not
                # pre-recorded in move_history. Once it resolves, lightspeed
                # immediately rolls the next move, so consume the opening move
                # here and skip the generic end-of-round roll.
                monster.ai_state.pop("fixed_opening_move", None)
                if not monster.move_history:
                    monster.move_history = ["THE_COLLECTOR_SPAWN"]
                choose_next_move(monster, self.ai_rng)
                monster.ai_state["skip_end_round_roll"] = True
                return
            elif monster.move in {"MYSTIC_BUFF", "GREMLIN_LEADER_ENCOURAGE", "COLLECTOR_BUFF", "BRONZE_AUTOMATON_BOOST", "ORB_WALKER_CHARGE", "MAW_ROAR", "GIANT_HEAD_COUNT", "DONU_CIRCLE_OF_POWER"}:
                if monster.move == "MYSTIC_BUFF":
                    strength_amount = 4 if self.ascension_level >= 17 else 3 if self.ascension_level >= 2 else 2
                    centurion = next((ally for ally in self.monsters if ally.alive and ally.monster_id == "Centurion"), None)
                    if centurion is not None and centurion is not monster:
                        centurion.add_power("Strength", strength_amount)
                    monster.add_power("Strength", strength_amount)
                else:
                    if monster.move == "COLLECTOR_BUFF":
                        strength_amount = 5 if self.ascension_level >= 19 else 4 if self.ascension_level >= 4 else 3
                        block_amount = 23 if self.ascension_level >= 19 else 18 if self.ascension_level >= 4 else 15
                        for ally in self.monsters[:2]:
                            if ally.alive and ally.monster_id == "TorchHead":
                                ally.add_power("Strength", strength_amount)
                        monster.add_power("Strength", strength_amount)
                        monster.block += block_amount
                    else:
                        monster.add_power("Strength", 2)
                return
            elif monster.move == "GREMLIN_LEADER_RALLY":
                gremlin_pool = [
                    "GremlinWarrior",
                    "GremlinWarrior",
                    "GremlinThief",
                    "GremlinThief",
                    "GremlinFat",
                    "GremlinFat",
                    "GremlinTsundere",
                    "GremlinWizard",
                ]
                open_slots = []
                if len(self.monsters) > 1 and (self.monsters[1].monster_id == "INVALID = 0" or self.monsters[1].is_gone or not self.monsters[1].alive):
                    open_slots.append(1)
                if len(self.monsters) > 2 and (self.monsters[2].monster_id == "INVALID = 0" or self.monsters[2].is_gone or not self.monsters[2].alive):
                    open_slots.append(2)
                if len(open_slots) < 2 and len(self.monsters) > 0 and (self.monsters[0].monster_id == "INVALID = 0" or self.monsters[0].is_gone or not self.monsters[0].alive):
                    open_slots.append(0)
                summoned_entries = []
                for slot in open_slots[:2]:
                    gremlin_id = gremlin_pool[int(self.ai_rng.random(7))]
                    summoned = make_monster(gremlin_id, self.monster_hp_rng, self.ascension_level)
                    summoned.ai_state["leader_minion"] = 1
                    summoned.ai_state["leader_summoned"] = 1
                    if gremlin_id == "GremlinWarrior":
                        summoned.powers.pop("Angry", None)
                    summoned.ai_state["spawn_index"] = slot
                    self.monsters[slot] = summoned
                    if self._has_relic("Philosopher's Stone"):
                        summoned.add_power("Strength", 1)
                    summoned_entries.append(summoned)
                for summoned in summoned_entries:
                    choose_next_move(summoned, self.ai_rng)
                for index, ally in enumerate(self.monsters):
                    ally.ai_state["spawn_index"] = index
                    setattr(ally, "_group_ref", self.monsters)
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
                if sum(1 for ally in self.monsters if ally.alive) <= 1:
                    _set_move(monster, "SHIELD_GREMLIN_SHIELD_BASH")
                monster.ai_state["skip_end_round_roll"] = True
                return
            elif monster.move == "SHIELD_GREMLIN_SHIELD_BASH":
                pass
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
                # Lightspeed applies the debuff here, but does not reroll the
                # retained hand immediately. Only cards drawn afterwards pick
                # up their randomized cost.
                self._apply_player_power("Confusion", 1, reroll_confusion_current_hand=False)
                return
            elif monster.move == "CHOSEN_HEX":
                self._apply_player_power("Hex", 1)
                return
            elif monster.move in {"SNAKE_PLANT_ENFEEBLING_SPORES", "COLLECTOR_MEGA_DEBUFF", "MAW_DROOL", "WRITHING_WITHER", "REPULSOR_REPULSE", "NEMESIS_DEBUFF", "TIME_EATER_RIPPLE"}:
                if monster.move == "COLLECTOR_MEGA_DEBUFF":
                    self._apply_player_power("Weakened", 3)
                    self._apply_player_power("Vulnerable", 3)
                    self._apply_player_power("Frail", 3)
                    return
                if monster.move == "SNAKE_PLANT_ENFEEBLING_SPORES":
                    self._apply_player_power("Frail", 2)
                self._apply_player_power("Weakened", 2)
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
                forge_amount = 7 if self.ascension_level >= 19 else 6 if self.ascension_level >= 9 else 5
                block_amount = 20 if self.ascension_level >= 19 else 18 if self.ascension_level >= 9 else 15
                monster.block += block_amount
                monster.add_power("Metallicize", forge_amount)
                return
            elif monster.move == "CHAMP_GLOAT":
                strength_amount = 5 if self.ascension_level >= 19 else 4 if self.ascension_level >= 9 else 3
                monster.add_power("Strength", strength_amount)
                return
            elif monster.move == "CHAMP_TAUNT":
                self._apply_player_power("Weakened", 2)
                self._apply_player_power("Vulnerable", 2)
                return
            elif monster.move == "CHAMP_ANGER":
                for power_id in ("Vulnerable", "Weakened", "Frail", "Poison", "Lock-On", "Slow", "Shackled"):
                    monster.powers.pop(power_id, None)
                if monster.power("Strength") < 0:
                    monster.powers.pop("Strength", None)
                strength_amount = 12 if self.ascension_level >= 19 else 9 if self.ascension_level >= 4 else 6
                monster.add_power("Strength", strength_amount)
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
            if starting_move == "SPHERIC_GUARDIAN_HARDEN":
                monster.block += 15
            if starting_move == "SLIME_BOSS_SLAM":
                _set_move(monster, "SLIME_BOSS_GOOP_SPRAY")
                monster.ai_state["skip_end_round_roll"] = True
            total_hp_damage = 0
            delayed_retaliatory_damage: list[tuple[str, int]] = []
            hit_count = max(1, monster.move_hits) if damage else 0
            if (
                hit_count <= 0
                and monster.move_base_damage > 0
                and monster.move_hits > 0
                and (self.player.power("Thorns") > 0 or self.player.power("Flame Barrier") > 0)
            ):
                hit_count = max(1, monster.move_hits)
            for _ in range(hit_count):
                hit_damage = damage
                if self.player.power("Intangible") > 0:
                    hit_damage = min(hit_damage, 1)
                blocked = min(self.player.block, hit_damage)
                self.player.block -= blocked
                hp_damage = hit_damage - blocked
                hp_lost_before = self.hp_lost_this_combat
                self._lose_hp(hp_damage, from_attack=True)
                actual_hp_damage = self.hp_lost_this_combat - hp_lost_before
                total_hp_damage += actual_hp_damage
                if actual_hp_damage > 0 and self.player.power("Plated Armor") > 0:
                    self.player.add_power("Plated Armor", -1)
                if actual_hp_damage > 0 and int(monster.ai_state.get("painful_stabs", 0)) > 0:
                    self.discard_pile.append(make_card("Wound", uuid=self._new_uuid("Wound")))
                if self.player.power("Thorns") > 0:
                    if starting_move == "SHELLED_SUCK":
                        delayed_retaliatory_damage.append(("Thorns", self.player.power("Thorns")))
                    else:
                        self._deal_retaliatory_damage_to_monster(self.player.power("Thorns"), monster)
                if self.player.power("Flame Barrier") > 0:
                    if starting_move == "SHELLED_SUCK":
                        delayed_retaliatory_damage.append(("Flame Barrier", self.player.power("Flame Barrier")))
                    else:
                        self._deal_flame_barrier_damage_to_monster(self.player.power("Flame Barrier"), monster)
            if deferred_post_attack_discard_cards:
                self.discard_pile.extend(deferred_post_attack_discard_cards)
            if starting_move == "JAW_WORM_THRASH":
                monster.block += 5
            if monster.move == "GREMLIN_NOB_SKULL_BASH":
                self._apply_player_power("Vulnerable", 2)
            if monster.move == "CHAMP_FACE_SLAP":
                self._apply_player_power("Frail", 2)
                self._apply_player_power("Vulnerable", 2)
            if starting_move in {"CULTIST_DARK_STRIKE", "FAT_GREMLIN_SMASH", "MAD_GREMLIN_SCRATCH"}:
                self.ai_rng.random(99)
                monster.ai_state["skip_end_round_roll"] = True
            if starting_move == "SNEAKY_GREMLIN_PUNCTURE":
                monster.ai_state["skip_end_round_roll"] = True
            if starting_move == "SHIELD_GREMLIN_SHIELD_BASH":
                monster.ai_state["skip_end_round_roll"] = True
            if starting_move == "BYRD_HEADBUTT":
                _set_move(monster, "BYRD_FLY")
                monster.ai_state["skip_end_round_roll"] = True
            if starting_move == "BRONZE_AUTOMATON_FLAIL":
                _set_move(monster, "BRONZE_AUTOMATON_BOOST")
                self.ai_rng.random(99)
                monster.ai_state["skip_end_round_roll"] = True
            if starting_move == "BRONZE_AUTOMATON_HYPER_BEAM":
                _set_move(monster, "BRONZE_AUTOMATON_BOOST" if self.ascension_level >= 19 else "BRONZE_AUTOMATON_STUNNED")
                self.ai_rng.random(99)
                monster.ai_state["skip_end_round_roll"] = True
            if starting_move == "BRONZE_AUTOMATON_STUNNED":
                _set_move(monster, "BRONZE_AUTOMATON_FLAIL")
                self.ai_rng.random(99)
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
            if starting_move == "SPIKE_SLIME_S_TACKLE":
                self.ai_rng.random(99)
                monster.ai_state["skip_end_round_roll"] = True
            if monster.move == "SPHERIC_GUARDIAN_ATTACK_DEBUFF":
                _set_move(monster, "SPHERIC_GUARDIAN_SLAM")
                self.ai_rng.random(99)
                monster.ai_state["skip_end_round_roll"] = True
            elif monster.move == "SPHERIC_GUARDIAN_HARDEN":
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
            if starting_move == "SPHERIC_GUARDIAN_ATTACK_DEBUFF":
                self._apply_player_power("Frail", 5)
            if starting_move == "MYSTIC_ATTACK":
                self._apply_player_power("Frail", 2)
            if starting_move == "SHELLED_FELL":
                self._apply_player_power("Frail", 2)
            if starting_move == "CHOSEN_DEBILITATE":
                self._apply_player_power("Vulnerable", 2)
            if starting_move == "SNECKO_TAIL_WHIP":
                self._apply_player_power("Vulnerable", 2)
            if starting_move == "SHELLED_SUCK" and total_hp_damage > 0:
                monster.current_hp = min(monster.max_hp, monster.current_hp + min(damage, total_hp_damage))
            if starting_move == "SHELLED_SUCK":
                for retaliatory_source, retaliatory_damage in delayed_retaliatory_damage:
                    if not monster.alive:
                        break
                    if retaliatory_source == "Thorns":
                        self._deal_retaliatory_damage_to_monster(retaliatory_damage, monster)
                    else:
                        self._deal_flame_barrier_damage_to_monster(retaliatory_damage, monster)
            guardian_shift_flag = bool(monster.ai_state.pop("guardian_shifted_this_round", 0))
            guardian_shifted_this_round = (
                monster.monster_id == "TheGuardian"
                and starting_move != "THE_GUARDIAN_DEFENSIVE_MODE"
                and (guardian_shift_flag or monster.move == "THE_GUARDIAN_DEFENSIVE_MODE")
            )
            if guardian_shifted_this_round:
                monster.ai_state["skip_end_round_roll"] = True
            if starting_move == "THE_GUARDIAN_FIERCE_BASH" and not guardian_shifted_this_round:
                _set_move(monster, "THE_GUARDIAN_VENT_STEAM")
                monster.ai_state["skip_end_round_roll"] = True
            elif starting_move == "THE_GUARDIAN_ROLL_ATTACK" and not guardian_shifted_this_round:
                _set_move(monster, "THE_GUARDIAN_TWIN_SLAM")
                monster.ai_state["skip_end_round_roll"] = True
            elif starting_move == "THE_GUARDIAN_TWIN_SLAM" and not guardian_shifted_this_round:
                monster.powers.pop("Sharp Hide", None)
                next_amount = int(monster.ai_state.get("mode_shift_amount", 30)) + 10
                monster.ai_state["mode_shift_amount"] = next_amount
                monster.add_power("Mode Shift", next_amount)
                _set_move(monster, "THE_GUARDIAN_WHIRLWIND")
                monster.ai_state["skip_end_round_roll"] = True
            elif starting_move == "THE_GUARDIAN_WHIRLWIND" and not guardian_shifted_this_round:
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

    def _card_exhausts_on_use(self, card: CardInstance) -> bool:
        return card_exhausts_on_use(card)

    def _resolve_after_use_card_move(
        self,
        card: CardInstance,
        *,
        force_exhaust: bool = False,
        defer_dark_embrace_draws: list[int] | None = None,
        add_hex_dazed: int = 0,
    ) -> None:
        resolve_after_use_card_move(
            self,
            card,
            force_exhaust=force_exhaust,
            defer_dark_embrace_draws=defer_dark_embrace_draws,
            add_hex_dazed=add_hex_dazed,
        )

    def _resolve_card_select(self, action: dict[str, object]) -> bool:
        return resolve_card_select(self, action)

    def replay_attack_card_effect_impl(
            self,
            card: CardInstance,
            target,
            energy_spent: int,
            *,
            body_slam_block_snapshot: int | None = None,
        ) -> None:
            card_id = card.card_id
            if card_id == "Strike_R":
                base = 9 if card.upgrades else 6
                base += self._strike_dummy_bonus(card)
                self._deal_attack_damage(base, target)
            elif card_id == "Bash":
                self._deal_attack_damage(10 if card.upgrades else 8, target)
                self._apply_monster_power(target, "Vulnerable", 3 if card.upgrades else 2)
            elif card_id == "Anger":
                self._deal_attack_damage(8 if card.upgrades else 6, target)
                self.discard_pile.append(make_card("Anger", upgrades=card.upgrades, uuid=self._new_uuid("Anger")))
            elif card_id == "Body Slam":
                self._deal_attack_damage(
                    self.player.block if body_slam_block_snapshot is None else body_slam_block_snapshot,
                    target,
                )
            elif card_id == "Clash":
                self._deal_attack_damage(18 if card.upgrades else 14, target)
            elif card_id == "Cleave":
                vigor_bonus = self._consume_attack_vigor_bonus()
                self._deal_damage_all(
                    11 if card.upgrades else 8,
                    vigor_bonus=vigor_bonus,
                    defer_attack_relic_proc=True,
                )
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
                # Lightspeed counts the in-flight Perfected Strike on the
                # original play while it is still the current hand card, but
                # replay copies from Double Tap/Necronomicon no longer count
                # that same card again after it has already left hand.
                strike_count = _combat_strike_count(self)
                self._deal_attack_damage((6 if card.upgrades else 6) + strike_count * (3 if card.upgrades else 2) + self._strike_dummy_bonus(card), target)
            elif card_id == "Pommel Strike":
                self._deal_attack_damage((10 if card.upgrades else 9) + self._strike_dummy_bonus(card), target)
                self.draw_cards(2 if card.upgrades else 1)
            elif card_id == "Sword Boomerang":
                vigor_bonus = self._consume_attack_vigor_bonus()
                hit_count = 4 if card.upgrades else 3
                alive_targets = self._alive_monsters()
                if len(alive_targets) == 1:
                    for _ in range(hit_count):
                        self._random_alive_monster(burn_if_single=True)
                    self._deal_attack_damage(
                        3,
                        alive_targets[0],
                        hits=hit_count,
                        vigor_bonus=vigor_bonus,
                        defer_attack_relic_proc=True,
                    )
                else:
                    for _ in range(hit_count):
                        self._deal_attack_damage(
                            3,
                            self._random_alive_monster(),
                            vigor_bonus=vigor_bonus,
                            defer_attack_relic_proc=True,
                        )
                    if self.pending_attack_relic_proc:
                        self._resolve_pending_attack_relic_proc()
            elif card_id == "Thunderclap":
                vigor_bonus = self._consume_attack_vigor_bonus()
                self.pending_counter_damage = 0
                for monster in self.monsters:
                    if monster.alive:
                        self._deal_attack_damage(
                            7 if card.upgrades else 4,
                            monster,
                            vigor_bonus=vigor_bonus,
                            defer_counter_damage=True,
                        )
                        if monster.alive:
                            self._apply_monster_power(monster, "Vulnerable", 1)
                counter_damage = self.pending_counter_damage
                self.pending_counter_damage = 0
                if self.pending_attack_relic_proc:
                    self._resolve_pending_attack_relic_proc()
                if counter_damage > 0:
                    self._take_counter_damage(counter_damage)
            elif card_id == "Twin Strike":
                base = 7 if card.upgrades else 5
                base += self._strike_dummy_bonus(card)
                self._deal_attack_damage(base, target, hits=2, defer_attack_relic_proc=True)
            elif card_id == "Wild Strike":
                self._deal_attack_damage((17 if card.upgrades else 12) + self._strike_dummy_bonus(card), target)
                self._insert_temp_card_into_draw_pile("Wound")
            elif card_id == "Blood for Blood":
                self._deal_attack_damage(22 if card.upgrades else 18, target)
            elif card_id == "Carnage":
                self._deal_attack_damage(28 if card.upgrades else 20, target)
            elif card_id == "Dropkick":
                self._consume_attack_vigor_bonus()
                self._deal_attack_damage(8 if card.upgrades else 5, target, vigor_bonus=0)
                if target and target.power("Vulnerable") > 0:
                    self.player.energy += 1
                    self.draw_cards(1)
            elif card_id == "Hemokinesis":
                strength_before_self_damage = self.player.power("Strength")
                self._lose_hp(2, self_damage=True)
                self._deal_attack_damage(
                    20 if card.upgrades else 15,
                    target,
                    strength_override=strength_before_self_damage,
                )
            elif card_id == "Pummel":
                self._deal_attack_damage(2, target, hits=5 if card.upgrades else 4, defer_attack_relic_proc=True)
            elif card_id == "Rampage":
                self._deal_attack_damage(8 + card.misc, target)
                card.misc += 8 if card.upgrades else 5
            elif card_id == "Reckless Charge":
                self._deal_attack_damage(10 if card.upgrades else 7, target)
                self._insert_temp_card_into_draw_pile("Dazed")
            elif card_id == "Searing Blow":
                self._deal_attack_damage(_base_damage_for_card(card), target)
            elif card_id == "Sever Soul":
                deferred_after_exhaust_actions: list[tuple[str, int | str]] = []
                exhausted_cards = self._pop_non_attacks_from_hand()
                self._deal_attack_damage(22 if card.upgrades else 16, target)
                for exhausted_card in exhausted_cards:
                    self._exhaust_card(
                        exhausted_card,
                        deferred_after_exhaust_actions=deferred_after_exhaust_actions,
                    )
                self._resolve_deferred_after_exhaust_actions(deferred_after_exhaust_actions)
            elif card_id == "Uppercut":
                self._deal_attack_damage(13, target)
                self._apply_monster_power(target, "Vulnerable", 2 if card.upgrades else 1)
                self._apply_monster_power(target, "Weakened", 2 if card.upgrades else 1)
            elif card_id == "Whirlwind":
                x_amount = energy_spent + (2 if self._has_relic("Chemical X") else 0)
                vigor_bonus = self._consume_attack_vigor_bonus()
                if x_amount <= 0:
                    if self.monsters:
                        sharp_hide_damage = self.monsters[0].power("Sharp Hide")
                        if sharp_hide_damage > 0:
                            self._take_counter_damage(sharp_hide_damage)
                else:
                    whirlwind_sharp_hide = [
                        (monster, monster.power("Sharp Hide"))
                        for monster in self._alive_monsters()
                        if monster.power("Sharp Hide") > 0
                    ]
                    for monster, _ in whirlwind_sharp_hide:
                        monster.powers["Sharp Hide"] = 0
                    try:
                        for _ in range(max(0, x_amount)):
                            self._deal_damage_all(8 if card.upgrades else 5, vigor_bonus=vigor_bonus)
                    finally:
                        for monster, amount in whirlwind_sharp_hide:
                            if monster.alive:
                                monster.powers["Sharp Hide"] = amount
                    sharp_hide_total = sum(amount for monster, amount in whirlwind_sharp_hide if monster.alive)
                    if sharp_hide_total > 0:
                        self._take_counter_damage(sharp_hide_total)
            elif card_id == "Bludgeon":
                self._deal_attack_damage(42 if card.upgrades else 32, target)
            elif card_id == "Feed":
                before_alive = target.alive if target else False
                self._deal_attack_damage(12 if card.upgrades else 10, target)
                if before_alive and self._defeat_reward_triggers(target):
                    self.player.max_hp += 4 if card.upgrades else 3
                    self._heal(4 if card.upgrades else 3)
            elif card_id == "Fiend Fire":
                count = len(self.hand)
                exhaust_count = len(self.hand)
                deferred_after_exhaust_actions: list[tuple[str, int | str]] = []
                for _ in range(exhaust_count):
                    if not self.hand:
                        break
                    exhausted = self.hand[self.card_random_rng.random(len(self.hand) - 1)]
                    self.hand.remove(exhausted)
                    self._exhaust_card(
                        exhausted,
                        deferred_after_exhaust_actions=deferred_after_exhaust_actions,
                    )
                self._deal_attack_damage(
                    10 if card.upgrades else 7,
                    target,
                    hits=count,
                    defer_attack_relic_proc=True,
                )
                self._resolve_deferred_after_exhaust_actions(deferred_after_exhaust_actions)
            elif card_id == "Immolate":
                vigor_bonus = self._consume_attack_vigor_bonus()
                self._deal_damage_all(28 if card.upgrades else 21, vigor_bonus=vigor_bonus)
                self.discard_pile.append(make_card("Burn", uuid=self._new_uuid("Burn")))
            elif card_id == "Reaper":
                vigor_bonus = self._consume_attack_vigor_bonus()
                healed = self._deal_damage_all(5 if card.upgrades else 4, vigor_bonus=vigor_bonus)
                self._heal(healed)
            elif card_id == "Dramatic Entrance":
                vigor_bonus = self._consume_attack_vigor_bonus()
                self._deal_damage_all(
                    12 if card.upgrades else 8,
                    vigor_bonus=vigor_bonus,
                    defer_attack_relic_proc=True,
                )
            elif card_id == "Flash of Steel":
                self._deal_attack_damage(6 if card.upgrades else 3, target)
                self.draw_cards(1)
            elif card_id == "Mind Blast":
                self._deal_attack_damage(len(self.draw_pile), target)
            elif card_id == "Swift Strike":
                self._deal_attack_damage((10 if card.upgrades else 7) + self._strike_dummy_bonus(card), target)
            elif card_id in {"Hand of Greed", "HandOfGreed"}:
                before_alive = target.alive if target else False
                pre_attack_curl_up = target.power("Curl Up") if target else 0
                self._deal_attack_damage(
                    25 if card.upgrades else 20,
                    target,
                    defer_monster_block_gains=True,
                    deferred_monster_block_requires_alive=True,
                )
                self._resolve_pending_monster_block_gains()
                if target is not None and not target.alive and pre_attack_curl_up > 0:
                    target.block = 0
                    target.powers["Curl Up"] = pre_attack_curl_up
                if before_alive and self._defeat_reward_triggers(target):
                    self.gold += 25 if card.upgrades else 20
            elif card_id == "Bite":
                self._deal_attack_damage(8 if card.upgrades else 7, target)
                self._heal(2)
            elif card_id == "Ritual Dagger":
                before_alive = target.alive if target else False
                self._deal_attack_damage(15 if card.upgrades else 15, target)
                if before_alive and self._defeat_reward_triggers(target):
                    card.misc += 5 if card.upgrades else 3

    def _replay_attack_card_effect(self, card: CardInstance, target, energy_spent: int) -> None:
        previous_processing_player_action = self.processing_player_action
        self.processing_player_action = True
        try:
            if card.card_def.has_target and target is not None and not target.alive:
                return
            if card.card_def.x_cost and self._processing_autoplay_cards and energy_spent > 0:
                # Havoc/Mayhem autoplayed X-cost attacks keep current energy for
                # their first free play, but lightspeed spends that stored X when
                # Double Tap/Necronomicon replays the attack.
                self.player.energy = max(0, self.player.energy - energy_spent)
            resolve_pending_delayed_reactions(self)
            pending_after_use_status_effects: list[tuple[str, int]] = []
            pending_after_use_direct_damage_all = 0
            pending_after_use_energy_gain = 0
            body_slam_block_snapshot = self.player.block if card.card_id == "Body Slam" else None
            self.cards_played_this_turn += 1
            self.card_types_played_this_turn.add("ATTACK")
            self.attack_played_this_turn += 1
            if self.attack_played_this_turn % 3 == 0:
                self.pending_attack_relic_proc = True
            if self._advance_relic_counter("Nunchaku", 10):
                self.player.energy += 1
                pending_after_use_energy_gain += 1
            if self._advance_relic_counter("Pen Nib", 10):
                self.player.add_power("Pen Nib", 1)
            self._attack_pen_nib_active = self.player.power("Pen Nib") > 0
            if self._attack_pen_nib_active:
                self.player.add_power("Pen Nib", -1)
            if self.player.power("Panache") > 0:
                self.panache_counter -= 1
                if self.panache_counter <= 0:
                    pending_after_use_direct_damage_all += self.player.power("Panache")
            if self.player.power("Rage") > 0:
                self._gain_block(self.player.power("Rage"), apply_block_modifiers=False)
            if self._advance_relic_counter("Ink Bottle", 10):
                pending_after_use_status_effects.append(("draw", 1))
            self.replay_attack_card_effect_impl(
                card,
                target,
                energy_spent,
                body_slam_block_snapshot=body_slam_block_snapshot,
            )
            if self.card_select_context == "HEADBUTT" and pending_after_use_energy_gain > 0:
                self.player.energy = max(0, self.player.energy - pending_after_use_energy_gain)
                self.pending_after_use_energy_gain += pending_after_use_energy_gain
            resolve_pending_delayed_reactions(self)
            if pending_after_use_direct_damage_all > 0:
                self._deal_direct_damage_all(pending_after_use_direct_damage_all)
            if pending_after_use_status_effects:
                if self.card_select_context is None:
                    self._resolve_deferred_status_draw_effects(pending_after_use_status_effects)
                else:
                    self.pending_resolve_status_effects.extend(pending_after_use_status_effects)
            if self.pending_attack_relic_proc and self.card_select_context is None:
                self._resolve_pending_attack_relic_proc()
            self._attack_pen_nib_active = False
        finally:
            self.processing_player_action = previous_processing_player_action

    def play_card_impl(
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
            pre_play_hand = list(self.hand)
            card = self.hand[hand_index]
            duplicate_index = 0
            while duplicate_index < len(self.hand):
                if self.hand[duplicate_index].uuid == card.uuid:
                    # Mirror lightspeed's current removeFromHandById quirk on
                    # the original hand array. Because it erases while walking
                    # forward, adjacent duplicate uniqueIds survive while later
                    # matching entries can still be removed.
                    self.hand.pop(duplicate_index)
                duplicate_index += 1
            if not self.playable(card, free_to_play=free_to_play):
                raise ValueError(f"card is not playable: {card.name}")
            target = self.monsters[target_index] if self.monsters else None
            if card.card_def.x_cost:
                energy_spent = max(0, int(energy_on_use if energy_on_use is not None else (0 if free_to_play else self.player.energy)))
            else:
                energy_spent = 0 if (free_to_play or card.free_to_play_once) else self._card_energy_cost(card)
            used_free_to_play_once = bool(card.free_to_play_once)
            if not free_to_play and not card.free_to_play_once:
                self.player.energy = max(0, self.player.energy - energy_spent)
            post_play_top_card = False
            if any(other.card_id == "Pain" for other in self.hand):
                self._lose_hp(1, self_damage=True)
            for monster in self._alive_monsters():
                if monster.power("Beat of Death") > 0:
                    self._lose_hp(monster.power("Beat of Death"))

            card_id = card.card_id
            played_attack = card.card_def.card_type == "ATTACK"
            pending_hex_dazed = 1 if self.player.power("Hex") > 0 and card.card_def.card_type != "ATTACK" else 0

            def _insert_pending_hex_dazed_now() -> None:
                nonlocal pending_hex_dazed
                if pending_hex_dazed <= 0:
                    return
                for _ in range(pending_hex_dazed):
                    self._insert_temp_card_into_draw_pile("Dazed")
                pending_hex_dazed = 0

            pending_after_use_direct_damage_all = 0
            pending_after_use_status_effects: list[tuple[str, int]] = []
            deferred_after_use_exhaust_actions: list[tuple[str, int | str]] = []
            attack_relic_proc_due = False
            pending_after_use_energy_gain = 0
            self.defer_juggernaut_for_current_card = bool(card.card_def.card_type == "SKILL")

            def _stash_pending_after_use_effects() -> None:
                nonlocal pending_after_use_direct_damage_all
                if pending_after_use_direct_damage_all > 0:
                    self.pending_after_use_direct_damage_all += pending_after_use_direct_damage_all
                    pending_after_use_direct_damage_all = 0
                if pending_after_use_energy_gain > 0:
                    self.pending_after_use_energy_gain += pending_after_use_energy_gain
                if self.pending_resolve_card is card:
                    self.pending_resolve_used_free_to_play_once = used_free_to_play_once
                if pending_after_use_status_effects:
                    self.pending_resolve_status_effects.extend(pending_after_use_status_effects)
                    pending_after_use_status_effects.clear()
                if attack_relic_proc_due:
                    self.pending_attack_relic_proc = True
                self.defer_juggernaut_for_current_card = False
                self._attack_pen_nib_active = False

            force_end_turn_after_card = False
            body_slam_block_snapshot = self.player.block if card.card_id == "Body Slam" else None
            attack_relic_proc_due_this_card = bool(
                played_attack and (self.attack_played_this_turn + 1) % 3 == 0
            )
            if attack_relic_proc_due_this_card:
                self.pending_attack_relic_proc = True
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
            if self.player.power("Panache") > 0:
                self.panache_counter -= 1
                if self.panache_counter <= 0:
                    pending_after_use_direct_damage_all += self.player.power("Panache")
            exhaust_override: bool | None = None
            attack_replays = 0
            after_use_card_moved = False
            if card.card_def.card_type == "SKILL":
                for monster in self._alive_monsters():
                    if monster.power("Enrage") > 0:
                        monster.add_power("Strength", monster.power("Enrage"))
                self.skills_played_this_turn += 1
                if self._has_relic("Letter Opener") and self.skills_played_this_turn >= 3 and self.skills_played_this_turn % 3 == 0:
                    pending_after_use_direct_damage_all += 5
                    self.defer_juggernaut_for_current_card = True
            if card.card_def.card_type == "POWER":
                for monster in self._alive_monsters():
                    if monster.power("Curiosity") > 0:
                        monster.add_power("Strength", 1)
                if self._has_relic("Bird Faced Urn"):
                    self._heal(2)
                if self._has_relic("Mummified Hand"):
                    mummified_hand_source = pre_play_hand
                    if self._processing_autoplay_cards and free_to_play:
                        mummified_hand_source = [other for other in pre_play_hand if other is not card]
                    candidates = [
                        other
                        for other in mummified_hand_source
                        if other.cost > 0
                        and self._card_energy_cost(other) > 0
                        and not other.free_to_play_once
                    ]
                    if candidates:
                        chosen_index = int(self.card_random_rng.random(len(candidates) - 1))
                        candidates[chosen_index].cost_for_turn = 0
            if played_attack:
                self.attack_played_this_turn += 1
                necronomicon_replays_attack = (
                    not free_to_play
                    and (
                        (card.card_def.x_cost and energy_spent >= 2)
                        or (not card.card_def.x_cost and self._card_energy_cost(card) >= 2)
                    )
                )
                if self._has_relic("Necronomicon") and necronomicon_replays_attack:
                    relic = self._relic("Necronomicon")
                    if relic is not None and int(relic.get("counter", 0)) != self.turn:
                        relic["counter"] = self.turn
                        attack_replays += 1
                if self._advance_relic_counter("Nunchaku", 10):
                    self.player.energy += 1
                    pending_after_use_energy_gain += 1
                if self._advance_relic_counter("Pen Nib", 10):
                    self.player.add_power("Pen Nib", 1)
                self._attack_pen_nib_active = self.player.power("Pen Nib") > 0
                if self._attack_pen_nib_active:
                    self.player.add_power("Pen Nib", -1)
                if self.player.power("Rage") > 0:
                    if card_id == "Headbutt":
                        pending_after_use_status_effects.append(("gain_block_no_mod", self.player.power("Rage")))
                    else:
                        self._gain_block(self.player.power("Rage"), apply_block_modifiers=False)
                if self.player.power("Double Tap") > 0:
                    self.player.add_power("Double Tap", -1)
                    attack_replays += 1
            if self._advance_relic_counter("Ink Bottle", 10):
                pending_after_use_status_effects.append(("draw", 1))

            if card.card_def.card_type == "CURSE":
                self._lose_hp(1, self_damage=True)
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
                self._deal_attack_damage(
                    self.player.block if body_slam_block_snapshot is None else body_slam_block_snapshot,
                    target,
                )
            elif card_id == "Clash":
                self._deal_attack_damage(18 if card.upgrades else 14, target)
            elif card_id == "Cleave":
                vigor_bonus = self._consume_attack_vigor_bonus()
                self._deal_damage_all(
                    11 if card.upgrades else 8,
                    vigor_bonus=vigor_bonus,
                    defer_attack_relic_proc=True,
                )
            elif card_id == "Clothesline":
                self._deal_attack_damage(14 if card.upgrades else 12, target)
                self._apply_monster_power(target, "Weakened", 3 if card.upgrades else 2)
            elif card_id == "Headbutt":
                self.pending_counter_damage = 0
                self._deal_attack_damage(
                    12 if card.upgrades else 9,
                    target,
                    defer_counter_damage=True,
                    defer_monster_block_gains=True,
                    deferred_monster_block_requires_alive=attack_replays > 0,
                )
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
                    if pending_after_use_energy_gain > 0:
                        self.player.energy = max(0, self.player.energy - pending_after_use_energy_gain)
                    self.pending_counter_damage = counter_damage
                    self.pending_attack_replays = attack_replays
                    self.pending_attack_target_index = self.monsters.index(target) if target in self.monsters else None
                    self._resolve_pending_spore_cloud_triggers()
                    self._open_discard_card_select(
                        "HEADBUTT",
                        list(range(len(self.discard_pile))),
                        pending_card=card,
                        pending_force_exhaust=force_exhaust,
                        pending_hex_dazed=pending_hex_dazed,
                    )
                    _stash_pending_after_use_effects()
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
                self._deal_attack_damage((6 if card.upgrades else 6) + strike_count * (3 if card.upgrades else 2) + self._strike_dummy_bonus(card), target)
            elif card_id == "Flex":
                amount = 4 if card.upgrades else 2
                self.player.add_power("Strength", amount)
                self._apply_player_power("Flex Strength Down", amount)
            elif card_id == "Inflame":
                self.player.add_power("Strength", 3 if card.upgrades else 2)
            elif card_id == "Pommel Strike":
                pommel_draw_count = 2 if card.upgrades else 1
                self._deal_attack_damage((10 if card.upgrades else 9) + self._strike_dummy_bonus(card), target)
                _insert_pending_hex_dazed_now()
                if (
                    target is not None
                    and target.monster_id == "TheGuardian"
                    and target.move == "THE_GUARDIAN_DEFENSIVE_MODE"
                    and self.pending_monster_block_gains
                    and self.player.power("Fire Breathing") > 0
                ):
                    # Lightspeed applies The Guardian's Mode Shift block
                    # before Pommel Strike's status-draw Fire Breathing damage.
                    self._resolve_pending_monster_block_gains()
                if (
                    self.player.power("Evolve") > 0
                    and len(self.draw_pile) >= pommel_draw_count
                    and self.pending_monster_kill_triggers <= 0
                    and self.pending_spore_cloud_player_turn_triggers <= 0
                    and self.pending_spore_cloud_monster_turn_triggers <= 0
                ):
                    self._resolve_after_use_card_move(
                        card,
                        force_exhaust=force_exhaust,
                        add_hex_dazed=0,
                    )
                    after_use_card_moved = True
                self.draw_cards(pommel_draw_count)
            elif card_id == "Shrug It Off":
                self._gain_block(11 if card.upgrades else 8)
                shrug_first_draw_triggers_status_chain = (
                    bool(self.draw_pile)
                    and (
                        (
                            self.player.power("Evolve") > 0
                            and self.draw_pile[-1].card_def.card_type == "STATUS"
                        )
                        or (
                            self.player.power("Fire Breathing") > 0
                            and self.draw_pile[-1].card_def.card_type in {"STATUS", "CURSE"}
                        )
                    )
                )
                if (
                    shrug_first_draw_triggers_status_chain
                    and self.pending_monster_kill_triggers <= 0
                    and self.pending_spore_cloud_player_turn_triggers <= 0
                    and self.pending_spore_cloud_monster_turn_triggers <= 0
                ):
                    self._resolve_after_use_card_move(
                        card,
                        force_exhaust=force_exhaust,
                        add_hex_dazed=pending_hex_dazed,
                    )
                    pending_hex_dazed = 0
                    after_use_card_moved = True
                self.draw_cards(1, deferred_status_effects=pending_after_use_status_effects)
            elif card_id == "Sword Boomerang":
                vigor_bonus = self._consume_attack_vigor_bonus()
                hit_count = 4 if card.upgrades else 3
                alive_targets = self._alive_monsters()
                if len(alive_targets) == 1:
                    for _ in range(hit_count):
                        self._random_alive_monster(burn_if_single=True)
                    self._deal_attack_damage(
                        3,
                        alive_targets[0],
                        hits=hit_count,
                        vigor_bonus=vigor_bonus,
                        defer_attack_relic_proc=True,
                    )
                else:
                    for _ in range(hit_count):
                        self._deal_attack_damage(
                            3,
                            self._random_alive_monster(),
                            vigor_bonus=vigor_bonus,
                            defer_attack_relic_proc=True,
                        )
                    if self.pending_attack_relic_proc:
                        self._resolve_pending_attack_relic_proc()
            elif card_id == "Thunderclap":
                vigor_bonus = self._consume_attack_vigor_bonus()
                self.pending_counter_damage = 0
                for monster in self.monsters:
                    if monster.alive:
                        self._deal_attack_damage(
                            7 if card.upgrades else 4,
                            monster,
                            vigor_bonus=vigor_bonus,
                            defer_counter_damage=True,
                        )
                        if monster.alive:
                            self._apply_monster_power(monster, "Vulnerable", 1)
                counter_damage = self.pending_counter_damage
                self.pending_counter_damage = 0
                if self.pending_attack_relic_proc:
                    self._resolve_pending_attack_relic_proc()
                if counter_damage > 0:
                    self._take_counter_damage(counter_damage)
            elif card_id == "Twin Strike":
                base = 7 if card.upgrades else 5
                base += self._strike_dummy_bonus(card)
                self._deal_attack_damage(base, target, hits=2, defer_attack_relic_proc=True)
            elif card_id == "Wild Strike":
                self._deal_attack_damage((17 if card.upgrades else 12) + self._strike_dummy_bonus(card), target)
                self._insert_temp_card_into_draw_pile("Wound")
            elif card_id == "Blood for Blood":
                self._deal_attack_damage(22 if card.upgrades else 18, target)
            elif card_id == "Carnage":
                self._deal_attack_damage(28 if card.upgrades else 20, target)
            elif card_id == "Dropkick":
                self._consume_attack_vigor_bonus()
                self._deal_attack_damage(8 if card.upgrades else 5, target, vigor_bonus=0)
                if target and target.power("Vulnerable") > 0:
                    self.player.energy += 1
                    if (
                        self.player.power("Evolve") > 0
                        and len(self.draw_pile) >= 1
                        and self.pending_monster_kill_triggers <= 0
                        and self.pending_spore_cloud_player_turn_triggers <= 0
                        and self.pending_spore_cloud_monster_turn_triggers <= 0
                    ):
                        self._resolve_after_use_card_move(
                            card,
                            force_exhaust=force_exhaust,
                            add_hex_dazed=0,
                        )
                        after_use_card_moved = True
                    _insert_pending_hex_dazed_now()
                    self.draw_cards(1)
            elif card_id == "Hemokinesis":
                strength_before_self_damage = self.player.power("Strength")
                self._lose_hp(2, self_damage=True)
                self._deal_attack_damage(
                    20 if card.upgrades else 15,
                    target,
                    strength_override=strength_before_self_damage,
                )
            elif card_id == "Pummel":
                self._deal_attack_damage(2, target, hits=5 if card.upgrades else 4, defer_attack_relic_proc=True)
            elif card_id == "Rampage":
                self._deal_attack_damage(8 + card.misc, target)
                card.misc += 8 if card.upgrades else 5
            elif card_id == "Reckless Charge":
                self._deal_attack_damage(10 if card.upgrades else 7, target)
                self._insert_temp_card_into_draw_pile("Dazed")
            elif card_id == "Searing Blow":
                self._deal_attack_damage(_base_damage_for_card(card), target)
            elif card_id == "Sever Soul":
                deferred_after_exhaust_actions: list[tuple[str, int | str]] = []
                exhausted_cards = self._pop_non_attacks_from_hand()
                self._deal_attack_damage(22 if card.upgrades else 16, target)
                self._resolve_after_use_card_move(
                    card,
                    force_exhaust=force_exhaust,
                    add_hex_dazed=pending_hex_dazed,
                )
                after_use_card_moved = True
                pending_hex_dazed = 0
                for exhausted_card in exhausted_cards:
                    self._exhaust_card(
                        exhausted_card,
                        deferred_after_exhaust_actions=deferred_after_exhaust_actions,
                    )
                self._resolve_deferred_after_exhaust_actions(deferred_after_exhaust_actions)
            elif card_id == "Uppercut":
                self._deal_attack_damage(13, target)
                self._apply_monster_power(target, "Weakened", 2 if card.upgrades else 1)
                self._apply_monster_power(target, "Vulnerable", 2 if card.upgrades else 1)
            elif card_id == "Whirlwind":
                x_amount = energy_spent + (2 if self._has_relic("Chemical X") else 0)
                vigor_bonus = self._consume_attack_vigor_bonus()
                if x_amount <= 0:
                    if self.monsters:
                        sharp_hide_damage = self.monsters[0].power("Sharp Hide")
                        if sharp_hide_damage > 0:
                            self._take_counter_damage(sharp_hide_damage)
                else:
                    whirlwind_sharp_hide = [
                        (monster, monster.power("Sharp Hide"))
                        for monster in self._alive_monsters()
                        if monster.power("Sharp Hide") > 0
                    ]
                    for monster, _ in whirlwind_sharp_hide:
                        monster.powers["Sharp Hide"] = 0
                    try:
                        for _ in range(max(0, x_amount)):
                            self._deal_damage_all(8 if card.upgrades else 5, vigor_bonus=vigor_bonus)
                    finally:
                        for monster, amount in whirlwind_sharp_hide:
                            if monster.alive:
                                monster.powers["Sharp Hide"] = amount
                    sharp_hide_total = sum(amount for monster, amount in whirlwind_sharp_hide if monster.alive)
                    if sharp_hide_total > 0:
                        self._take_counter_damage(sharp_hide_total)
            elif card_id == "Bludgeon":
                self._deal_attack_damage(42 if card.upgrades else 32, target)
            elif card_id == "Feed":
                before_alive = target.alive if target else False
                self._deal_attack_damage(12 if card.upgrades else 10, target)
                if before_alive and self._defeat_reward_triggers(target):
                    self.player.max_hp += 4 if card.upgrades else 3
                    self._heal(4 if card.upgrades else 3)
            elif card_id in {"Hand of Greed", "HandOfGreed"}:
                before_alive = target.alive if target else False
                pre_attack_curl_up = target.power("Curl Up") if target else 0
                self._deal_attack_damage(
                    25 if card.upgrades else 20,
                    target,
                    defer_monster_block_gains=True,
                    deferred_monster_block_requires_alive=True,
                )
                self._resolve_pending_monster_block_gains()
                if target is not None and target.current_hp <= 0 and pre_attack_curl_up > 0:
                    target.block = 0
                    target.powers["Curl Up"] = pre_attack_curl_up
                if before_alive and self._defeat_reward_triggers(target):
                    self.gold += 25 if card.upgrades else 20
            elif card_id == "Fiend Fire":
                count = len(self.hand)
                exhaust_count = len(self.hand)
                deferred_after_exhaust_actions: list[tuple[str, int | str]] = []
                for _ in range(exhaust_count):
                    if not self.hand:
                        break
                    exhausted = self.hand[self.card_random_rng.random(len(self.hand) - 1)]
                    self.hand.remove(exhausted)
                    self._exhaust_card(
                        exhausted,
                        deferred_after_exhaust_actions=deferred_after_exhaust_actions,
                    )
                self._deal_attack_damage(
                    10 if card.upgrades else 7,
                    target,
                    hits=count,
                    defer_attack_relic_proc=True,
                )
                if count == 0 and target is not None and target.alive:
                    sharp_hide_damage = target.power("Sharp Hide")
                    if sharp_hide_damage > 0:
                        self._take_counter_damage(sharp_hide_damage)
                    thorn_damage = target.power("Thorns")
                    if thorn_damage > 0:
                        self._take_counter_damage(thorn_damage)
                self._resolve_deferred_after_exhaust_actions(deferred_after_exhaust_actions)
                self._resolve_after_use_card_move(
                    card,
                    force_exhaust=force_exhaust,
                    add_hex_dazed=pending_hex_dazed,
                )
                after_use_card_moved = True
                pending_hex_dazed = 0
            elif card_id == "Immolate":
                vigor_bonus = self._consume_attack_vigor_bonus()
                self._deal_damage_all(28 if card.upgrades else 21, vigor_bonus=vigor_bonus)
                self.discard_pile.append(make_card("Burn", uuid=self._new_uuid("Burn")))
            elif card_id == "Reaper":
                vigor_bonus = self._consume_attack_vigor_bonus()
                healed = self._deal_damage_all(5 if card.upgrades else 4, vigor_bonus=vigor_bonus)
                self._heal(healed)
            elif card_id == "Armaments":
                upgradeable = [] if card.upgrades else [index for index, hand_card in enumerate(self.hand) if _card_can_upgrade(hand_card)]
                self._gain_block(5 if card.upgrades else 5, defer_juggernaut=len(upgradeable) > 1)
                if card.upgrades:
                    for hand_card in self.hand:
                        if _card_can_armaments_plus_upgrade(hand_card):
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
                        _stash_pending_after_use_effects()
                        return
            elif card_id == "Havoc":
                # Lightspeed fixes Havoc's autoplay target before later
                # after-use skill triggers such as Letter Opener resolve.
                self._play_random_top_card(force_exhaust=True)
            elif card_id == "True Grit":
                defer_true_grit_juggernaut = bool(self.hand)
                self._gain_block(9 if card.upgrades else 7, defer_juggernaut=defer_true_grit_juggernaut)
                if card.upgrades:
                    if len(self.hand) == 1:
                        exhausted_card = self.hand.pop(0)
                        self._exhaust_card(exhausted_card)
                        self._resolve_pending_juggernaut_damage()
                    elif self.hand:
                        self._open_combat_card_select(
                            "EXHAUST_ONE",
                            list(range(len(self.hand))),
                            pending_card=card,
                            pending_force_exhaust=force_exhaust,
                            pending_hex_dazed=pending_hex_dazed,
                        )
                        _stash_pending_after_use_effects()
                        return
                elif self.hand:
                    chosen = self.card_random_rng.choice(list(self.hand))
                    self.hand.remove(chosen)
                    if (
                        self.player.power("Dark Embrace") > 0
                        and not after_use_card_moved
                        and not force_exhaust
                        and self.player.power("Corruption") <= 0
                    ):
                        self._move_card_to_discard(card)
                        after_use_card_moved = True
                    self._exhaust_card(chosen)
                    self._resolve_pending_juggernaut_damage()
            elif card_id == "Warcry":
                self.draw_cards(2 if card.upgrades else 1, deferred_status_effects=pending_after_use_status_effects)
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
                    _stash_pending_after_use_effects()
                    return
            elif card_id == "Battle Trance":
                self.draw_cards(4 if card.upgrades else 3, deferred_status_effects=pending_after_use_status_effects)
                self._apply_player_power("No Draw", 1)
            elif card_id == "Bloodletting":
                self._lose_hp(3, self_damage=True)
                self.player.energy += 3 if card.upgrades else 2
            elif card_id == "Burning Pact":
                if len(self.hand) == 1:
                    deferred_dark_embrace_draws: list[int] = []
                    deferred_after_exhaust_actions: list[tuple[str, int | str]] = []
                    exhausted_card = self.hand.pop(0)
                    self._exhaust_card(
                        exhausted_card,
                        defer_dark_embrace_draws=deferred_dark_embrace_draws,
                        deferred_after_exhaust_actions=deferred_after_exhaust_actions,
                    )
                    draw_count = 3 if card.upgrades else 2
                    self.draw_cards(draw_count, deferred_status_effects=pending_after_use_status_effects)
                    self._resolve_after_use_card_move(
                        card,
                        force_exhaust=force_exhaust,
                        defer_dark_embrace_draws=deferred_dark_embrace_draws,
                        add_hex_dazed=pending_hex_dazed,
                    )
                    if pending_after_use_direct_damage_all > 0:
                        self._deal_direct_damage_all(pending_after_use_direct_damage_all)
                    self._resolve_deferred_status_draw_effects(pending_after_use_status_effects)
                    self._resolve_deferred_after_exhaust_actions(deferred_after_exhaust_actions)
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
                    _stash_pending_after_use_effects()
                    return
                else:
                    self.draw_cards(3 if card.upgrades else 2, deferred_status_effects=pending_after_use_status_effects)
            elif card_id == "Disarm":
                self._apply_monster_power(target, "Strength", -2)
            elif card_id == "Dual Wield":
                selectable = [index for index, other in enumerate(self.hand) if other.card_def.card_type in {"ATTACK", "POWER"}]
                if len(selectable) == 1:
                    copy_count = 2 if card.upgrades > 0 else 1
                    selected_card = clone_card(self.hand[selectable[0]], reset_cost_for_turn=False)
                    selected_card.uuid = self._new_uuid(f"dual-wield-{selected_card.card_id}")
                    for _ in range(copy_count):
                        copied_card = clone_card(selected_card, reset_cost_for_turn=False)
                        copied_card.uuid = self._new_uuid(f"dual-wield-{selected_card.card_id}")
                        if len(self.hand) < 10:
                            self.hand.append(copied_card)
                        else:
                            self.discard_pile.append(copied_card)
                elif selectable:
                    self._open_combat_card_select(
                        "DUAL_WIELD",
                        selectable,
                        pending_card=card,
                        pending_force_exhaust=force_exhaust,
                        pending_hex_dazed=pending_hex_dazed,
                    )
                    _stash_pending_after_use_effects()
                    return
            elif card_id == "Entrench":
                before_block = self.player.block
                self.player.block *= 2
                if self.player.block > before_block and self.player.power("Juggernaut") > 0:
                    self._trigger_juggernaut(self.player.power("Juggernaut"))
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
                deferred_after_exhaust_actions: list[tuple[str, int | str]] = []
                exhausted = self._exhaust_non_attacks_from_hand(
                    deferred_after_exhaust_actions=deferred_after_exhaust_actions,
                )
                block_per_card = 7 if card.upgrades else 5
                for _ in range(exhausted):
                    self._gain_block(block_per_card)
                move_current_second_wind_last = (
                    pending_hex_dazed <= 0
                    and (
                        force_exhaust
                        or self._card_exhausts_on_use(card)
                        or (self.player.power("Corruption") > 0 and card.card_def.card_type == "SKILL")
                    )
                    and any(action_kind == "add_to_hand" for action_kind, _ in deferred_after_exhaust_actions)
                )
                if move_current_second_wind_last:
                    self._resolve_deferred_after_exhaust_actions(deferred_after_exhaust_actions)
                    deferred_after_exhaust_actions = []
                self._resolve_after_use_card_move(
                    card,
                    force_exhaust=force_exhaust,
                    add_hex_dazed=pending_hex_dazed,
                )
                after_use_card_moved = True
                pending_hex_dazed = 0
                if deferred_after_exhaust_actions:
                    self._resolve_deferred_after_exhaust_actions(deferred_after_exhaust_actions)
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
                    _stash_pending_after_use_effects()
                    return
            elif card_id == "Impervious":
                self._gain_block(40 if card.upgrades else 30)
            elif card_id == "Limit Break":
                strength = self.player.power("Strength")
                if strength != 0:
                    self.player.add_power("Strength", strength)
                if (
                    card.upgrades
                    and not getattr(card, "_temporary_upgrade", False)
                    and self.player.power("Corruption") <= 0
                ):
                    exhaust_override = False
            elif card_id == "Offering":
                self._lose_hp(6, self_damage=True)
                self.player.energy += 2
                self.draw_cards(5 if card.upgrades else 3, deferred_status_effects=pending_after_use_status_effects)
                if (
                    pending_after_use_status_effects
                    and not after_use_card_moved
                    and (self._has_relic("Dead Branch") or self.player.power("Dark Embrace") > 0)
                ):
                    self._exhaust_card(
                        card,
                        deferred_after_exhaust_actions=deferred_after_use_exhaust_actions,
                    )
                    after_use_card_moved = True
            elif card_id == "J.A.X.":
                self._lose_hp(3, self_damage=True)
                self.player.add_power("Strength", 3 if card.upgrades else 2)
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
                    if self._advance_relic_counter("Sundial", 3):
                        self.player.energy += 2
                self.draw_cards(2 if card.upgrades else 1, deferred_status_effects=pending_after_use_status_effects)
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
                    _stash_pending_after_use_effects()
                    return
            elif card_id == "Dramatic Entrance":
                vigor_bonus = self._consume_attack_vigor_bonus()
                self._deal_damage_all(
                    12 if card.upgrades else 8,
                    vigor_bonus=vigor_bonus,
                    defer_attack_relic_proc=True,
                )
            elif card_id == "Enlightenment":
                for other in self.hand:
                    if other.card_def.card_type in {"ATTACK", "SKILL", "POWER"} and other.cost > 1:
                        other.cost_for_turn = 1
                        if card.upgrades:
                            other.cost_for_combat = 1
            elif card_id == "Finesse":
                self._gain_block(4 if card.upgrades else 2)
                finesse_post_draw_hex_to_hand = 0
                finesse_after_current_draw_actions: list[tuple[str, int | str]] | None = None
                if (
                    self.player.power("Evolve") > 0
                    and len(self.draw_pile) >= 1
                    and self.pending_monster_kill_triggers <= 0
                    and self.pending_spore_cloud_player_turn_triggers <= 0
                    and self.pending_spore_cloud_monster_turn_triggers <= 0
                ):
                    self._resolve_after_use_card_move(
                        card,
                        force_exhaust=force_exhaust,
                        add_hex_dazed=0,
                    )
                    after_use_card_moved = True
                    if pending_hex_dazed > 0:
                        if self.player.power("Evolve") == 1:
                            finesse_after_current_draw_actions = []
                            for _ in range(pending_hex_dazed):
                                finesse_after_current_draw_actions.append(("add_to_draw_top", "Dazed"))
                        else:
                            finesse_post_draw_hex_to_hand = pending_hex_dazed
                            self.pending_finesse_hex_generated_dazed = pending_hex_dazed
                        pending_hex_dazed = 0
                self.draw_cards(
                    1,
                    deferred_status_effects=pending_after_use_status_effects,
                    after_current_batch_actions=finesse_after_current_draw_actions,
                )
                for _ in range(finesse_post_draw_hex_to_hand):
                    pending_after_use_status_effects.append(("add_to_hand", "Dazed"))
                self.pending_finesse_hex_generated_dazed = 0
            elif card_id == "Flash of Steel":
                self._deal_attack_damage(6 if card.upgrades else 3, target)
                _insert_pending_hex_dazed_now()
                if (
                    self.player.power("Evolve") > 0
                    and len(self.draw_pile) >= 1
                    and self.pending_monster_kill_triggers <= 0
                    and self.pending_spore_cloud_player_turn_triggers <= 0
                    and self.pending_spore_cloud_monster_turn_triggers <= 0
                ):
                    self._resolve_after_use_card_move(
                        card,
                        force_exhaust=force_exhaust,
                        add_hex_dazed=0,
                    )
                    after_use_card_moved = True
                self.draw_cards(1)
            elif card_id == "Forethought":
                if len(self.hand) == 1:
                    chosen = self.hand.pop(0)
                    if chosen.cost > 0:
                        chosen.free_to_play_once = True
                    self.draw_pile.insert(0, chosen)
                elif len(self.hand) > 1:
                    self._open_combat_card_select(
                        "FORETHOUGHT",
                        list(range(len(self.hand))),
                        pending_card=card,
                        pending_force_exhaust=force_exhaust,
                        pending_hex_dazed=pending_hex_dazed,
                    )
                    _stash_pending_after_use_effects()
                    return
            elif card_id == "Good Instincts":
                self._gain_block(9 if card.upgrades else 6)
            elif card_id == "Impatience":
                # Mirror lightspeed's current Impatience implementation, which
                # always draws because the hasAttack sentinel never flips true.
                impatience_draw_count = 3 if card.upgrades else 2
                if (
                    self.player.power("Evolve") > 0
                    and len(self.draw_pile) >= impatience_draw_count
                    and self.pending_monster_kill_triggers <= 0
                    and self.pending_spore_cloud_player_turn_triggers <= 0
                    and self.pending_spore_cloud_monster_turn_triggers <= 0
                ):
                    self._resolve_after_use_card_move(
                        card,
                        force_exhaust=force_exhaust,
                        add_hex_dazed=pending_hex_dazed,
                    )
                    pending_hex_dazed = 0
                    after_use_card_moved = True
                self.draw_cards(impatience_draw_count, deferred_status_effects=pending_after_use_status_effects)
            elif card_id == "Jack Of All Trades":
                generated_count = 2 if card.upgrades else 1
                generated_ids = [self._random_card_id(colorless=True) for _ in range(generated_count)]
                for generated_id in reversed(generated_ids):
                    self._add_to_hand(generated_id)
            elif card_id == "Madness":
                def _madness_base_cost(candidate: CardInstance) -> int:
                    if candidate.cost_for_combat is not None:
                        return candidate.cost_for_combat
                    if candidate.upgrades > 0 and candidate.card_def.upgraded_cost is not None:
                        return candidate.card_def.upgraded_cost
                    return candidate.card_def.cost

                have_non_zero_cost = False
                have_non_zero_turn_cost = False
                for other in self.hand:
                    effective_turn_cost = other.cost if other.cost_for_turn is None else other.cost_for_turn
                    if effective_turn_cost > 0:
                        have_non_zero_turn_cost = True
                        break
                    if _madness_base_cost(other) > 0:
                        have_non_zero_cost = True
                if have_non_zero_turn_cost or have_non_zero_cost:
                    while True:
                        random_idx = self.card_random_rng.random(len(self.hand) - 1)
                        chosen = self.hand[random_idx]
                        effective_turn_cost = chosen.cost if chosen.cost_for_turn is None else chosen.cost_for_turn
                        if have_non_zero_turn_cost:
                            if effective_turn_cost <= 0:
                                continue
                        else:
                            if _madness_base_cost(chosen) <= 0:
                                continue
                        chosen.card_def = replace(chosen.card_def, cost=0, upgraded_cost=0)
                        chosen.cost_for_turn = 0
                        break
            elif card_id == "Mind Blast":
                self._deal_attack_damage(len(self.draw_pile), target)
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
                _stash_pending_after_use_effects()
                return
            elif card_id == "Swift Strike":
                self._deal_attack_damage((10 if card.upgrades else 7) + self._strike_dummy_bonus(card), target)
            elif card_id == "Trip":
                if card.upgrades:
                    for monster in self._alive_monsters():
                        self._apply_monster_power(monster, "Vulnerable", 2)
                else:
                    self._apply_monster_power(target, "Vulnerable", 2)
            elif card_id == "Apotheosis":
                seen_cards: set[int] = set()
                for pile in (self.hand, self.draw_pile, self.discard_pile, self.exhaust_pile, self.deck):
                    for other in pile:
                        if other is card:
                            continue
                        if other.card_def.card_type in {"STATUS", "CURSE"}:
                            continue
                        marker = id(other)
                        if marker in seen_cards:
                            continue
                        seen_cards.add(marker)
                        self._upgrade_combat_card(other)
            elif card_id == "Chrysalis":
                self._put_random_cards_in_draw_pile(card_type="SKILL", count=5 if card.upgrades else 3, cost_for_turn=0)
            elif card_id in {"Hand of Greed", "HandOfGreed"}:
                before_alive = target.alive if target else False
                self._deal_attack_damage(25 if card.upgrades else 20, target)
                if before_alive and self._defeat_reward_triggers(target):
                    self.gold_gain += 25 if card.upgrades else 20
            elif card_id == "Magnetism":
                self.player.add_power("Magnetism", 1)
            elif card_id == "Master of Strategy":
                self.draw_cards(4 if card.upgrades else 3, deferred_status_effects=pending_after_use_status_effects)
            elif card_id == "Mayhem":
                self.player.add_power("Mayhem", 1)
            elif card_id == "Metamorphosis":
                self._put_random_cards_in_draw_pile(card_type="ATTACK", count=5 if card.upgrades else 3, cost_for_turn=0)
            elif card_id == "Panache":
                self.player.add_power("Panache", 14 if card.upgrades else 10)
            elif card_id == "Sadistic Nature":
                self.player.add_power("Sadistic Nature", 7 if card.upgrades else 5)
            elif card_id == "Secret Technique":
                selectable = self._draw_to_hand_candidate_indexes("SKILL")
                if len(selectable) == 1:
                    self._choose_draw_pile_card_to_hand(selectable[0])
                elif len(selectable) > 1:
                    self._open_draw_pile_card_select(
                        "SECRET_TECHNIQUE",
                        selectable,
                        pending_card=card,
                        pending_force_exhaust=force_exhaust,
                        pending_hex_dazed=pending_hex_dazed,
                    )
                    _stash_pending_after_use_effects()
                    return
            elif card_id == "Secret Weapon":
                selectable = self._draw_to_hand_candidate_indexes("ATTACK")
                if len(selectable) == 1:
                    self._choose_draw_pile_card_to_hand(selectable[0])
                elif len(selectable) > 1:
                    self._open_draw_pile_card_select(
                        "SECRET_WEAPON",
                        selectable,
                        pending_card=card,
                        pending_force_exhaust=force_exhaust,
                        pending_hex_dazed=pending_hex_dazed,
                    )
                    _stash_pending_after_use_effects()
                    return
            elif card_id == "The Bomb":
                self.player.add_power("The Bomb", 3)
                self.player.powers["The Bomb Damage"] = 50 if card.upgrades else 40
            elif card_id == "Thinking Ahead":
                self.draw_cards(2, deferred_status_effects=pending_after_use_status_effects)
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
                    _stash_pending_after_use_effects()
                    return
            elif card_id == "Transmutation":
                x_amount = energy_spent + (2 if self._has_relic("Chemical X") else 0)
                for _ in range(max(0, x_amount)):
                    self._add_random_card_to_hand(colorless=True, cost_for_turn=0)
            elif card_id == "Violence":
                limit = 4 if card.upgrades else 3
                attack_indexes: list[int] = []
                for idx, other in enumerate(self.draw_pile):
                    if other.card_def.card_type == "ATTACK":
                        if not attack_indexes:
                            attack_indexes.append(idx)
                        else:
                            random_idx = int(self.card_random_rng.random(len(attack_indexes) - 1))
                            attack_indexes.insert(random_idx, idx)
                if attack_indexes:
                    remove_indexes: list[int] = []
                    moved_count = 0
                    while moved_count < limit:
                        if len(attack_indexes) - moved_count <= 0:
                            break
                        tail = attack_indexes[moved_count:]
                        if tail:
                            shuffled_tail = list(tail)
                            self._shuffle_cards(shuffled_tail)
                            attack_indexes[moved_count:] = shuffled_tail
                        remove_idx = attack_indexes[moved_count]
                        remove_indexes.append(remove_idx)
                        other = self.draw_pile[remove_idx]
                        moved_card = clone_card(other, reset_cost_for_turn=False)
                        if len(self.hand) >= 10:
                            self.discard_pile.append(moved_card)
                        else:
                            self.hand.append(moved_card)
                        moved_count += 1
                        if len(attack_indexes) - moved_count <= 0 and moved_count < limit:
                            # Mirror lightspeed's current ViolenceAction bug:
                            # when fewer attacks exist than requested, it returns
                            # before removing the moved cards from draw_pile.
                            remove_indexes = []
                            break
                    for remove_idx in sorted(remove_indexes, reverse=True):
                        self.draw_pile.pop(remove_idx)
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
                if before_alive and self._defeat_reward_triggers(target):
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
                self._apply_corruption_to_existing_cards()
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
            if pending_after_use_direct_damage_all > 0:
                self._deal_direct_damage_all(pending_after_use_direct_damage_all)
            exhaust_played = force_exhaust or self._card_exhausts_on_use(card) or (self.player.power("Corruption") > 0 and card.card_def.card_type == "SKILL")
            if exhaust_override is not None and not force_exhaust:
                exhaust_played = exhaust_override
            if pending_hex_dazed > 0:
                for _ in range(pending_hex_dazed):
                    self._insert_temp_card_into_draw_pile("Dazed")
            if (
                self.pending_monster_kill_triggers > 0
                and not after_use_card_moved
                and not exhaust_played
                and card.card_def.card_type != "POWER"
            ):
                # Lightspeed resolves Gremlin Horn-style kill triggers only
                # after the played card has already fallen into discard, so an
                # empty-draw shuffle includes the just-played card.
                self._move_card_to_discard(card)
                after_use_card_moved = True
            self._resolve_pending_monster_kill_triggers()
            if (
                pending_after_use_status_effects
                and card_id == "True Grit"
                and card.upgrades <= 0
                and not after_use_card_moved
                and not exhaust_played
            ):
                # Lightspeed resolves Ink Bottle-style after-use draws after the
                # played True Grit has already fallen into discard, so an
                # empty-draw shuffle includes the current card.
                self._move_card_to_discard(card)
                after_use_card_moved = True
            if pending_after_use_status_effects:
                if deferred_after_use_exhaust_actions:
                    flushed_deferred_after_exhaust_actions = False
                    for effect_kind, amount in pending_after_use_status_effects:
                        if amount <= 0:
                            continue
                        if effect_kind == "draw":
                            self.draw_cards(
                                amount,
                                after_current_batch_actions=(
                                    deferred_after_use_exhaust_actions
                                    if not flushed_deferred_after_exhaust_actions
                                    else None
                                ),
                            )
                            if not flushed_deferred_after_exhaust_actions:
                                deferred_after_use_exhaust_actions = []
                                flushed_deferred_after_exhaust_actions = True
                        elif effect_kind == "damage_all":
                            self._deal_direct_damage_all(amount)
                            self._resolve_pending_monster_kill_triggers()
                            self._check_outcome()
                            if self.outcome != "UNDECIDED":
                                return
                        elif effect_kind == "gain_block_no_mod":
                            self._gain_block(amount, apply_block_modifiers=False)
                else:
                    self._resolve_deferred_status_draw_effects(pending_after_use_status_effects)
            if deferred_after_use_exhaust_actions:
                self._resolve_deferred_after_exhaust_actions(deferred_after_use_exhaust_actions)
            self.defer_juggernaut_for_current_card = False
            if used_free_to_play_once:
                # Consume Forethought-style temporary free-play only after all
                # on-use reactions have inspected the current card. Lightspeed's
                # Mummified Hand target scan still sees the in-flight power as
                # free-to-play once.
                card.free_to_play_once = False
            if not after_use_card_moved:
                if exhaust_played:
                    self._exhaust_card(card)
                elif card.card_def.card_type == "POWER":
                    pass
                else:
                    self._move_card_to_discard(card)
            if self.pending_juggernaut_damage > 0:
                self._resolve_pending_juggernaut_damage()
            self._attack_pen_nib_active = False
            for _ in range(attack_replays):
                if self.outcome != "UNDECIDED":
                    break
                self._replay_attack_card_effect(card, target, energy_spent)
            self._resolve_pending_monster_kill_triggers()
            self._drain_pending_autoplay_cards()
            if self._has_relic("Unceasing Top") and not self.hand:
                if self.draw_pile:
                    self.draw_cards(
                        1,
                        suppress_status_draw_triggers=True,
                        allow_fire_breathing_when_suppressed=True,
                        deferred_status_damage_all=self.pending_start_turn_fire_breathing_damage,
                    )
                elif self.discard_pile:
                    self.pending_unceasing_top_draw = True
            if attack_relic_proc_due:
                self.pending_attack_relic_proc = True
            if self.pending_attack_relic_proc and self.card_select_context is None:
                self._resolve_pending_attack_relic_proc()
            self._check_outcome()
            if force_end_turn_after_card and self.outcome == "UNDECIDED":
                self.end_turn()
            self._attack_pen_nib_active = False

    def _resolve_pending_attack_relic_proc(self) -> None:
            if not self.pending_attack_relic_proc:
                return
            self.pending_attack_relic_proc = False
            if self._has_relic("Shuriken"):
                self.player.add_power("Strength", 1)
            if self._has_relic("Kunai"):
                self.player.add_power("Dexterity", 1)
            if self._has_relic("Ornamental Fan"):
                defer_fan_juggernaut = bool(self.pending_monster_block_gains)
                self._gain_block(
                    4,
                    defer_juggernaut=defer_fan_juggernaut,
                    apply_block_modifiers=False,
                )
                if defer_fan_juggernaut:
                    self._resolve_pending_monster_block_gains()
                    if self.pending_juggernaut_damage > 0:
                        self._resolve_pending_juggernaut_damage()

    def end_turn_impl(self) -> None:
            if not self.pending_end_turn_resume and self._has_relic("Nilry's Codex"):
                codex_options = self._discovery_card_options()
                if codex_options:
                    self.card_select_context = "CODEX"
                    self.card_select_generated_cards = codex_options
                    self.card_select_source_indexes = []
                    self.card_select_options = [
                        {
                            "kind": "single_card_select",
                            "name": "CODEX",
                            "select_type": "CODEX",
                            "choice_index": index,
                            "select_index": index,
                            "deck_index": index,
                            "card_id": card.card_id,
                            "card": card_to_spirecomm(card),
                            "bits": 3000 + index,
                        }
                        for index, card in enumerate(codex_options)
                    ]
                    self.card_select_options.append(
                        {
                            "kind": "single_card_select",
                            "name": "CODEX",
                            "select_type": "CODEX",
                            "choice_index": len(codex_options),
                            "select_index": len(codex_options),
                            "deck_index": len(codex_options),
                            "bits": 3000 + len(codex_options),
                        }
                    )
                    self.pending_end_turn_resume = True
                    return
            self.pending_end_turn_resume = False
            self.processing_end_turn_cleanup = True
            try:
                self.monster_turn_damage_draws_are_end_turn = False
                had_alive_monsters_at_end_turn_start = any(monster.alive for monster in self.monsters)
                alive_after_hand_cleanup = had_alive_monsters_at_end_turn_start
                if self.pending_unceasing_top_draw:
                    self.pending_unceasing_top_draw = False
                    self.draw_cards(
                        1,
                        suppress_status_draw_triggers=True,
                        allow_fire_breathing_when_suppressed=True,
                        deferred_status_damage_all=self.pending_start_turn_fire_breathing_damage,
                    )
                # Lightspeed queues Orichalcum's end-turn block before the hand
                # cleanup that exhausts ethereal cards, so Feel No Pain exhaust
                # triggers can stack on top of that block instead of suppressing it.
                if self._has_relic("Orichalcum") and self.player.block == 0:
                    self._gain_block(6, defer_juggernaut=True, apply_block_modifiers=False)
                discarded = 0
                regular_discarded = 0
                deferred_end_turn_counter_damage: list[int] = []
                deferred_end_turn_ethereal_exhausts: list[CardInstance] = []
                deferred_end_turn_status_ethereal_exhausts_after_no_draw: list[CardInstance] = []
                deferred_end_turn_dark_embrace_draws: list[int] = []
                deferred_end_turn_dark_embrace_fire_breathing_damage: list[int] = []
                deferred_end_turn_feel_no_pain_blocks: list[int] = []
                original_hand_count = len(self.hand)
                def _exhaust_end_turn_ethereal(card: CardInstance) -> None:
                    defer_dark_embrace_draws = None
                    if self.player.power("Dark Embrace") > 0:
                        if self.player.power("No Draw") > 0:
                            defer_dark_embrace_draws = deferred_end_turn_dark_embrace_draws
                    self._exhaust_card(
                        card,
                        defer_dark_embrace_draws=defer_dark_embrace_draws,
                        deferred_dark_embrace_status_damage_all=deferred_end_turn_dark_embrace_fire_breathing_damage,
                        defer_feel_no_pain_blocks=(
                            deferred_end_turn_feel_no_pain_blocks
                            if deferred_end_turn_counter_damage
                            else None
                        ),
                    )

                def _flush_deferred_end_turn_ethereal_exhausts() -> None:
                    while deferred_end_turn_ethereal_exhausts:
                        deferred_card = deferred_end_turn_ethereal_exhausts.pop(0)
                        _exhaust_end_turn_ethereal(deferred_card)

                # Turn-only cost state is cleared at the turn boundary before
                # any end-turn Dark Embrace draws. Those draws belong to the
                # next turn's hand, so effects like Confusion must keep their
                # freshly rolled cost even if hand cleanup is still in flight.
                self._clear_temporary_cost_state()
                for card in list(self.hand):
                    if card.card_id not in {"Burn", "Decay", "Regret", "Doubt", "Shame"}:
                        continue
                    self.hand.remove(card)
                    if card.card_id == "Burn":
                        deferred_end_turn_counter_damage.append(4 if card.upgrades else 2)
                    elif card.card_id == "Decay":
                        self.monster_turn_damage_draws_are_end_turn = True
                        self._take_counter_damage(2, self_damage=True)
                        if self.outcome != "UNDECIDED":
                            return
                    elif card.card_id == "Regret":
                        self._lose_hp(original_hand_count, self_damage=True)
                    elif card.card_id == "Doubt":
                        self._apply_player_power("Weakened", 1)
                    elif card.card_id == "Shame":
                        self._apply_player_power("Frail", 1)
                    self._move_card_to_discard(card)
                    discarded += 1
                for hand_index in range(len(self.hand) - 1, -1, -1):
                    card = self.hand[hand_index]
                    if self._has_relic("Runic Pyramid") and not self._card_is_ethereal(card) and card.card_id != "Burn":
                        continue
                    self.hand.pop(hand_index)
                    if self._card_is_ethereal(card):
                        if (
                            self.player.power("Dark Embrace") > 0
                            and self.player.power("No Draw") > 0
                            and self._has_relic("Dead Branch")
                            and card.card_def.card_type == "STATUS"
                        ):
                            deferred_end_turn_status_ethereal_exhausts_after_no_draw.append(card)
                        elif self.player.power("Dark Embrace") > 0 and self.hand:
                            deferred_end_turn_ethereal_exhausts.append(card)
                        else:
                            _flush_deferred_end_turn_ethereal_exhausts()
                            _exhaust_end_turn_ethereal(card)
                    else:
                        self._move_card_to_discard(card)
                        discarded += 1
                        regular_discarded += 1
                _flush_deferred_end_turn_ethereal_exhausts()
                alive_after_hand_cleanup = any(monster.alive for monster in self.monsters)
                self.cards_discarded_this_turn += discarded
                self.end_turn_regular_discard_count = regular_discarded
                # Entangled/No Attack is cleared at the end of the affected player turn,
                # not by the normal debuff duration tick.
                self.player.powers.pop("No Attack", None)
                if self._has_relic("Art of War") and self.attack_played_this_turn == 0:
                    self.player.powers["Art of War Energy"] = 1
                if self._has_relic("Pocketwatch") and self.cards_played_this_turn <= 3:
                    self.player.powers["Pocketwatch Draw"] = 1
                if self.player.power("Flex Strength Down") > 0:
                    amount = self.player.power("Flex Strength Down")
                    self._apply_player_power("Strength", -amount, just_applied=False)
                    self.player.powers.pop("Flex Strength Down", None)
                self.player.powers.pop("No Draw", None)
                for card in deferred_end_turn_status_ethereal_exhausts_after_no_draw:
                    self._exhaust_card(
                        card,
                        deferred_dark_embrace_status_damage_all=deferred_end_turn_dark_embrace_fire_breathing_damage,
                        defer_feel_no_pain_blocks=(
                            deferred_end_turn_feel_no_pain_blocks
                            if deferred_end_turn_counter_damage
                            else None
                        ),
                    )
                for amount in deferred_end_turn_dark_embrace_draws:
                    self.draw_cards(
                        amount,
                        # Lightspeed's current post-No Draw Dark Embrace
                        # draws can reshuffle the same turn's regular
                        # discards, even when the ethereal source is a
                        # non-status card like Carnage.
                        exclude_end_turn_regular_discards=False,
                    )
                stone_calendar_shifted_guardian = False
                if self.turn == 6 and self._has_relic("Stone Calendar"):
                    pre_stone_calendar_guardian_mode_shift = {
                        id(monster): monster.power("Mode Shift")
                        for monster in self.monsters
                        if monster.monster_id == "TheGuardian"
                    }
                    self._deal_direct_damage_all(52)
                    stone_calendar_shifted_guardian = any(
                        monster.monster_id == "TheGuardian"
                        and pre_stone_calendar_guardian_mode_shift.get(id(monster), 0) > 0
                        and monster.power("Mode Shift") <= 0
                        for monster in self.monsters
                    )
                if self.player.power("Metallicize") > 0:
                    self._gain_block(self.player.power("Metallicize"), apply_block_modifiers=False)
                if self.player.power("Plated Armor") > 0:
                    self._gain_block(self.player.power("Plated Armor"), apply_block_modifiers=False)
                self.pending_start_turn_fire_breathing_damage.extend(
                    deferred_end_turn_dark_embrace_fire_breathing_damage
                )
                if deferred_end_turn_counter_damage:
                    self.monster_turn_damage_draws_are_end_turn = True
                for amount in deferred_end_turn_counter_damage:
                    self._take_counter_damage(amount, self_damage=True)
                    if self.outcome != "UNDECIDED":
                        return
                for amount in deferred_end_turn_feel_no_pain_blocks:
                    self._gain_block(amount, apply_block_modifiers=False)
                if self.player.power("Dexterity Down") > 0:
                    amount = self.player.power("Dexterity Down")
                    self._apply_player_power("Dexterity", -amount, just_applied=False)
                    self.player.powers.pop("Dexterity Down", None)
                combust_should_resolve = any(monster.alive for monster in self.monsters)
                if (
                    not combust_should_resolve
                    and had_alive_monsters_at_end_turn_start
                    and not alive_after_hand_cleanup
                ):
                    # If end-turn hand cleanup itself killed the last enemies
                    # (for example via Charon's Ashes), lightspeed still lets
                    # Combust finish resolving. Later end-turn effects like
                    # Stone Calendar should not keep Combust alive.
                    combust_should_resolve = True
                if self.player.power("Combust") > 0 and combust_should_resolve:
                    self.monster_turn_damage_draws_are_end_turn = True
                    self._lose_hp(self.combust_hp_loss, self_damage=True)
                    if not stone_calendar_shifted_guardian:
                        self._deal_direct_damage_all(self.player.power("Combust"))
                        self._resolve_pending_monster_kill_triggers()
                if self.player.power("The Bomb") > 0:
                    self.player.add_power("The Bomb", -1)
                    if self.player.power("The Bomb") <= 0:
                        damage = self.player.powers.pop("The Bomb Damage", 40)
                        self._deal_direct_damage_all(damage)
                        self._resolve_pending_monster_kill_triggers()
                        self._check_outcome()
                        if self.outcome != "UNDECIDED":
                            return
                # Kills during end-turn hand cleanup (for example Charon's
                # Ashes from ethereal exhaust) should apply Spore Cloud before
                # the surviving monsters attack, but Gremlin Horn-style kill
                # rewards still wait until the next player turn.
                self._resolve_pending_spore_cloud_triggers()
                self._resolve_pending_monster_block_gains()
                for monster in self.monsters:
                    if monster.alive and monster.power("Barricade") <= 0:
                        monster.block = 0
                self._resolve_pending_juggernaut_damage()
                if self._resolve_pending_start_turn_fire_breathing_damage():
                    return
                monster_turn_index = 0
                pending_end_of_round_ai_noops = 0
                self.processing_monster_turns = True
                try:
                    while monster_turn_index < len(self.monsters):
                        monster = self.monsters[monster_turn_index]
                        started_turn_alive = monster.alive
                        extra_roll_index: int | None = None
                        if started_turn_alive:
                            extra_roll_index = self._monster_take_turn(monster, monster_turn_index)
                            self._resolve_pending_monster_kill_triggers()
                            if self.outcome != "UNDECIDED":
                                break
                            roll_move_if_gone = bool(monster.ai_state.pop("roll_move_if_gone", False))
                            should_roll = (monster.alive or roll_move_if_gone) and monster.monster_id != "TorchHead"
                            if should_roll and not monster.ai_state.pop("skip_end_round_roll", False):
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
                finally:
                    self.processing_monster_turns = False
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
                    if not monster.alive:
                        continue
                    shackled = monster.power("Shackled")
                    if shackled > 0:
                        monster.add_power("Strength", shackled)
                        monster.powers.pop("Shackled", None)
                self._tick_powers_at_end_of_round(self.player.powers, self.player_powers_just_applied)
                for monster in self.monsters:
                    if monster.current_hp <= 0 or monster.is_gone or monster.ai_state.get("escaping"):
                        continue
                    self._tick_powers_at_end_of_round(monster.powers)
                self.player.powers.pop("Flame Barrier", None)
                self.player.powers.pop("Rage", None)
                self.player.powers.pop("Double Tap", None)
                self._check_outcome()
                if self.outcome == "UNDECIDED":
                    self.start_player_turn()
            finally:
                self.monster_turn_damage_draws_are_end_turn = False
                self.end_turn_regular_discard_count = 0
                self.processing_end_turn_cleanup = False

    def _resolve_pending_start_turn_fire_breathing_damage(self) -> bool:
            if not self.pending_start_turn_fire_breathing_damage:
                return False
            deferred_damage = list(self.pending_start_turn_fire_breathing_damage)
            self.pending_start_turn_fire_breathing_damage = []
            for amount in deferred_damage:
                self._deal_direct_damage_all(amount)
                self._resolve_pending_monster_kill_triggers()
                self._check_outcome()
                if self.outcome != "UNDECIDED":
                    return True
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
        v2_play_card(
            self,
            hand_index,
            target_index,
            force_exhaust=force_exhaust,
            free_to_play=free_to_play,
            energy_on_use=energy_on_use,
        )

    def end_turn(self) -> None:
        v2_end_turn(self)

    def step(self, action: dict[str, object]):
        return v2_step(self, action)

    def to_spirecomm_state(self) -> dict[str, object]:
        return serialize_combat_state(self)


@dataclass
class NativeRunEnv(RunHelpersMixin):
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
    card_select_completion: str | None = None
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
                # In the aligned simulator, Mark of Pain only injects Wounds at
                # combat start, not permanently into the run deck.
                pass
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
                # Align with lightspeed's current obtainRelic implementation,
                # which does not apply Old Coin's immediate +300 gold effect.
                pass
            elif relic_id == "Matryoshka":
                relic["counter"] = 2
            elif relic_id == "Omamori":
                relic["counter"] = 2
            elif relic_id == "Maw Bank":
                relic["counter"] = 1
            elif relic_id == "Ink Bottle":
                relic["counter"] = 0
            elif relic_id == "Sundial":
                relic["counter"] = 0
            elif relic_id == "Lizard Tail":
                relic["counter"] = 0
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
                # Align with lightspeed's current Cursed Tome / obtainRelic flow,
                # which grants Necronomicon without adding Necronomicurse to the run deck.
                pass
            elif relic_id == "Tiny House":
                self.player.max_hp += 5
                self.player.current_hp = min(self.player.max_hp, self.player.current_hp + 5)
                self._gain_gold(50)
                card = self._roll_card_reward(count=1)[0]
                self._add_card_to_deck(card.card_id, upgrades=card.upgrades, uuid=f"tiny-house-{self.floor}-{card.card_id}")
                self._add_potion_if_space(roll_potion(self.randoms.potion))
                index = self._first_upgradable_index()
                if index is not None:
                    _increment_card_upgrade(self.deck[index])
            elif relic_id == "Empty Cage":
                removable = [index for index, card in enumerate(self.deck) if _card_can_transform(card)]
                if removable:
                    self._open_card_select("EVENT_REMOVE", min(2, len(removable)))
                    self.card_select_completion = "TRANSITION_NEXT_ACT"
            elif relic_id == "Calling Bell":
                self._add_curse_to_deck("CurseOfTheBell", uuid=f"calling-bell-{self.floor}")
                self.phase = "CARD_REWARD"
                self.reward_context = "BOSS_RELIC"
                self.reward_close_required = False
                self.reward_card_bundles = []
                self.reward_cards = []
                self.reward_gold = 0
                self.reward_gold_piles = []
                self.reward_emerald_key = False
                self.reward_potions = []
                self.reward_relics = [
                    self._roll_screenless_relic_of_tier("COMMON"),
                    self._roll_screenless_relic_of_tier("UNCOMMON"),
                    self._roll_screenless_relic_of_tier("RARE"),
                ]
            elif relic_id == "Pandora's Box":
                transformed_cards: list[CardInstance] = []
                retained_deck: list[CardInstance] = []
                for index, card in enumerate(list(self.deck)):
                    if card.card_id in {"Strike_R", "Defend_R"}:
                        transformed = self._pandora_transformed_card_from_rng(self.randoms.misc, card.card_id)
                        transformed_cards.append(
                            self._make_deck_card(
                                transformed.card_id,
                                upgrades=transformed.upgrades,
                                uuid=f"pandora-{self.floor}-{index}",
                            )
                        )
                        if self._has_relic("Ceramic Fish"):
                            self._gain_gold(9)
                    else:
                        retained_deck.append(card)
                self.deck = retained_deck + list(reversed(transformed_cards))
            elif relic_id == "Astrolabe":
                self.card_select_available_indexes = [
                    index
                    for index, card in enumerate(self.deck)
                    if _card_can_transform(card)
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
                    _increment_card_upgrade(self.deck[index])
            elif relic_id == "Whetstone":
                upgradeable = [
                    index
                    for index, card in enumerate(self.deck)
                    if card.card_def.card_type == "ATTACK" and _card_can_upgrade(card)
                ]
                java_collections_shuffle(upgradeable, self.randoms.misc.random_long())
                for index in upgradeable[:2]:
                    _increment_card_upgrade(self.deck[index])

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
                "x": 0,
                "child_count": 0,
                "next_symbols": [],
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
                "child_count": len(self.map_graph.get(node_id, {}).get("children", [])),
                "next_symbols": [
                    self.map_graph[child_id]["symbol"]
                    for child_id in self.map_graph.get(node_id, {}).get("children", [])
                    if child_id in self.map_graph and self.map_graph[child_id].get("symbol")
                ],
            }
            for index, node_id in enumerate(node_ids)
        ]

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

    def _enter_treasure_room(self) -> None:
            self.phase = "TREASURE"
            tea_set = self._relic("Ancient Tea Set")
            if tea_set is not None and int(tea_set.get("counter", 0)) > 0:
                tea_set["counter"] = 0
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

    def _advance_floor(self) -> None:
            if self.phase == "EVENT" and self.current_node_symbol == "?":
                tea_set = self._relic("Ancient Tea Set")
                if tea_set is not None and int(tea_set.get("counter", 0)) > 0:
                    tea_set["counter"] = 0
            self._enter_map()

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

    def _start_combat(self, *, elite: bool = False) -> None:
        v2_run_start_combat(self, elite=elite)

    def _start_event_boss_combat(self, *, act_boss: str | None = None) -> None:
        v2_run_start_event_boss_combat(self, act_boss=act_boss)

    def legal_actions(self) -> list[dict[str, object]]:
        return v2_run_legal_actions(self)

    def step(self, action: dict[str, object]) -> dict[str, object]:
        return v2_run_step(self, action)

    def state(self) -> dict[str, object]:
        return v2_run_state(self)
