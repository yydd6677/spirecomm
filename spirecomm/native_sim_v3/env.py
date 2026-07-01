from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spirecomm.native_sim_v3.content import (
    V3_IMPLEMENTATION_STATUS,
    act_for_dungeon_id,
    can_upgrade_card,
    dungeon_id_for_act,
    draw_neow_rare_card,
    generate_blessing_options,
    initialize_runtime_card_pools,
    initialize_source_card_pools,
    initialize_relic_pools,
    is_banned_relic_id,
    next_dungeon_id,
    make_card,
    make_relic,
    pop_random_non_campfire_relic_from_pools,
    pop_random_relic_from_pools,
    pop_random_screenless_relic_from_pools,
    roll_random_relic_tier,
    starting_profile,
    starter_deck,
    upgrade_card,
    starter_relics,
    truly_random_card_from_source_pools,
)
from spirecomm.native_sim_v3.content.act_chances import act_chances
from spirecomm.native_sim_v3.combat.engine import CombatEngine
from spirecomm.native_sim_v3.core.randoms import NativeRandomSet
from spirecomm.native_sim_v3.core.state import PlayerState, RunState
from spirecomm.native_sim_v3.reference.source_map import CORE_SOURCE_MAP
from spirecomm.native_sim_v3.run.boss import (
    apply_boss_relic_choice,
    draw_boss_relic_choices,
)
from spirecomm.native_sim_v3.run.encounters import generate_monster_lists_for_dungeon
from spirecomm.native_sim_v3.run.events import (
    BASE_ELITE_CHANCE,
    BASE_MONSTER_CHANCE,
    BASE_SHOP_CHANCE,
    BASE_TREASURE_CHANCE,
    EventState,
    generate_event_for_act,
    initialize_event_pools_for_dungeon,
    initialize_special_one_time_event_list,
    RESET_ELITE_CHANCE,
    RESET_MONSTER_CHANCE,
    RESET_SHOP_CHANCE,
    RESET_TREASURE_CHANCE,
    resolve_event_choice,
    roll_question_room_result,
)
from spirecomm.native_sim_v3.run.map import available_map_actions, generate_act_map
from spirecomm.native_sim_v3.run.neow import NeowRewardType, draw_neow_cards
from spirecomm.native_sim_v3.run.neow import draw_neow_curse
from spirecomm.native_sim_v3.run.neow import transform_card as neow_transform_card
from spirecomm.native_sim_v3.content.events import note_for_yourself_preference
from spirecomm.native_sim_v3.content.potions import draw_random_potion, potion_priority_value, roll_random_potion
from spirecomm.native_sim_v3.content.shop import shop_rules
from spirecomm.native_sim_v3.run.rewards import (
    apply_reward_preview_relics,
    generate_card_reward_groups_with_state,
    generate_card_reward_with_state,
    generate_elite_relic_rewards,
    generate_monster_room_rewards,
    roll_post_combat_potion,
)
from spirecomm.native_sim_v3.content.reward_rules import card_blizz_rules
from spirecomm.native_sim_v3.content.campfire_rules import regal_pillow_bonus
from spirecomm.native_sim_v3.run.campfire import CampfireState, rest_amount
from spirecomm.native_sim_v3.run.shop import ShopState, _java_round_positive, _visible_shop_card_name, generate_shop
from spirecomm.native_sim_v3.run.treasure import TreasureState, generate_treasure, open_treasure
from spirecomm.native_sim_v3.serialize import combat_state as serialize_combat_state
from spirecomm.native_sim_v3.serialize import run_state as serialize_run_state
from spirecomm.native_sim_v3.core.randoms import java_shuffle_in_place


@dataclass(slots=True)
class _NotImplementedAction:
    kind: str = "not_implemented"
    name: str = "NOT_IMPLEMENTED"


def _potion_id(potion: dict[str, Any]) -> str:
    return str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "")


def _is_potion_slot(potion: dict[str, Any]) -> bool:
    return _potion_id(potion) == "Potion Slot"


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


def _normalize_potion_slots(potions: list[dict[str, Any]], max_slots: int) -> list[dict[str, Any]]:
    normalized = [dict(potion) for potion in list(potions or [])]
    while len(normalized) < int(max_slots):
        normalized.append(_make_potion_slot(len(normalized)))
    return normalized[: int(max_slots)]


def _has_open_potion_slot(potions: list[dict[str, Any]], max_slots: int) -> bool:
    normalized = _normalize_potion_slots(potions, max_slots)
    return any(_is_potion_slot(potion) for potion in normalized)


def _add_potion_to_first_slot(potions: list[dict[str, Any]], potion: dict[str, Any], max_slots: int) -> bool:
    normalized = _normalize_potion_slots(potions, max_slots)
    for index, current in enumerate(normalized):
        if _is_potion_slot(current):
            normalized[index] = dict(potion)
            potions[:] = normalized
            return True
    potions[:] = normalized
    return False


def _add_or_replace_lowest_priority_potion(potions: list[dict[str, Any]], potion: dict[str, Any], max_slots: int) -> bool:
    normalized = _normalize_potion_slots(potions, max_slots)
    for index, current in enumerate(normalized):
        if _is_potion_slot(current):
            normalized[index] = dict(potion)
            potions[:] = normalized
            return True

    if not normalized:
        potions[:] = normalized
        return False

    lowest_index, lowest_potion = min(
        enumerate(normalized),
        key=lambda item: (potion_priority_value(item[1]), item[0]),
    )
    if potion_priority_value(potion) > potion_priority_value(lowest_potion):
        normalized[lowest_index] = dict(potion)
        potions[:] = normalized
        return True
    potions[:] = normalized
    return False


def _remove_potion_at_slot(potions: list[dict[str, Any]], index: int, max_slots: int) -> bool:
    normalized = _normalize_potion_slots(potions, max_slots)
    if 0 <= int(index) < len(normalized) and not _is_potion_slot(normalized[int(index)]):
        normalized[int(index)] = _make_potion_slot(int(index))
        potions[:] = normalized
        return True
    potions[:] = normalized
    return False


def _lowest_priority_potion_slot(potions: list[dict[str, Any]], max_slots: int) -> int | None:
    normalized = _normalize_potion_slots(potions, max_slots)
    candidates = [
        (index, potion)
        for index, potion in enumerate(normalized)
        if not _is_potion_slot(potion) and bool(potion.get("can_discard", True))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (potion_priority_value(item[1]), item[0]))[0]


class NativeCombatEnv:
    """Independent v3 combat adapter."""

    sim_backend = "v3"
    _TEACHER_BRANCH_DROP_FIELDS = frozenset(
        {
            "state",
            "source_card_pools",
            "reference_sources",
            "master_deck",
            "relics",
            "potions",
            "player",
        }
    )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        if bool(state.pop("_teacher_branch_clone_slim", False)):
            for key in self._TEACHER_BRANCH_DROP_FIELDS:
                if key not in state:
                    continue
                value = state[key]
                if isinstance(value, list):
                    state[key] = []
                elif isinstance(value, dict):
                    state[key] = {}
                else:
                    state[key] = None
        return state

    def __init__(
        self,
        seed: int,
        ascension_level: int = 0,
        *,
        character: str = "IRONCLAD",
        act: int = 1,
        floor: int = 0,
        player: PlayerState | None = None,
        master_deck: list[dict[str, Any]] | None = None,
        relics: list[dict[str, Any]] | None = None,
        potions: list[dict[str, Any]] | None = None,
        gold: int = 99,
        act_boss: str | None = None,
        encounter_name: str = "Cultist",
        room_type: str = "MonsterRoom",
        source_card_pools: dict[str, list[str]] | None = None,
        randoms: NativeRandomSet | None = None,
        has_emerald_key: bool = False,
        prebuilt_monsters: list[Any] | None = None,
        elite_trigger: bool = False,
        **_: Any,
    ) -> None:
        self.seed = int(seed)
        self.ascension_level = int(ascension_level)
        self.player_class = str(character)
        self.act = int(act)
        self.dungeon_id = dungeon_id_for_act(self.act)
        self.floor = int(floor)
        self.randoms = randoms or NativeRandomSet(seed=self.seed)
        self.outcome = "UNDECIDED"
        base_player = player or PlayerState(current_hp=80, max_hp=80)
        self.player = PlayerState(
            current_hp=int(base_player.current_hp),
            max_hp=int(base_player.max_hp),
            base_energy=int(base_player.base_energy),
            draw_per_turn=int(base_player.draw_per_turn),
        )
        self.master_deck = list(master_deck or [])
        self.relics = list(relics or [])
        self.potions = list(potions or [])
        self.gold = int(gold)
        self.act_boss = act_boss
        self.encounter_name = encounter_name
        self.room_type = str(room_type)
        self.has_emerald_key = bool(has_emerald_key)
        self.elite_trigger = bool(elite_trigger)
        self.source_card_pools = {
            key: list(values)
            for key, values in (
                source_card_pools or initialize_source_card_pools(character=self.player_class)
            ).items()
        }
        self.reference_sources = dict(CORE_SOURCE_MAP["combat"])
        self.engine = CombatEngine(
            encounter_name=encounter_name,
            room_type=self.room_type,
            randoms=self.randoms,
            ascension_level=self.ascension_level,
            act=self.act,
            character=self.player_class,
            player=self.player,
            master_deck=self.master_deck,
            relics=self.relics,
            potions=self.potions,
            gold=self.gold,
            source_card_pools=self.source_card_pools,
            has_emerald_key=self.has_emerald_key,
            prebuilt_monsters=prebuilt_monsters,
            elite_trigger=self.elite_trigger,
        )
        self.state = self.engine.state
        self.outcome = str(getattr(self.engine, "outcome", "UNDECIDED") or "UNDECIDED")

    def serialize(self) -> dict[str, Any]:
        return serialize_combat_state(self)

    def legal_actions(self) -> list[dict[str, Any]]:
        return self.engine.legal_actions()

    def step(self, action: dict[str, Any]) -> str:
        if bool(getattr(self, "_teacher_fast_combat_sync", False)):
            try:
                setattr(self.engine, "_teacher_fast_step_refresh", True)
            except Exception:
                pass
            result = self.engine.step(action)
            self.state = self.engine.state
            self.outcome = result.outcome
            return self.outcome
        result = self.engine.step(action)
        self.master_deck = list(self.engine.master_deck)
        self.player = self.engine.player
        self.potions = list(self.engine.potions)
        self.gold = int(self.engine.gold)
        self.state = self.engine.state
        self.outcome = result.outcome
        return self.outcome

    def to_spirecomm_state(self) -> dict[str, Any]:
        return self.serialize()


class NativeRunEnv:
    """Independent v3 run adapter with the stable public env shape."""

    sim_backend = "v3"
    _TEACHER_BRANCH_DROP_FIELDS = frozenset(
        {
            "_core_state",
            "_map_dp_state_by_node",
            "map",
            "card_pools",
            "source_card_pools",
            "relic_pools",
            "boss_relic_pool",
            "monster_list",
            "elite_monster_list",
            "boss_list",
            "special_one_time_event_list",
            "event_list",
            "shrine_list",
            "neow_options",
            "current_event",
            "current_campfire",
            "current_shop",
            "current_treasure",
            "current_card_select",
            "pending_event_rewards",
            "boss_relic_options",
            "reward_cards",
            "reward_card_groups",
            "reward_gold",
            "reward_stolen_gold",
            "reward_potions",
            "reward_relics",
            "reward_order",
            "pending_cursed_key_chest_curse",
        }
    )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        if bool(state.pop("_teacher_branch_clone_slim", False)):
            for key in self._TEACHER_BRANCH_DROP_FIELDS:
                if key not in state:
                    continue
                value = state[key]
                if isinstance(value, list):
                    state[key] = []
                elif isinstance(value, dict):
                    state[key] = {}
                elif isinstance(value, bool):
                    state[key] = False
                elif isinstance(value, int):
                    state[key] = 0
                else:
                    state[key] = None
            state["_teacher_fast_terminal_state"] = None
        return state

    def __init__(
        self,
        seed: int,
        ascension_level: int = 0,
        *,
        character: str = "IRONCLAD",
        final_act_available: bool = True,
        has_ruby_key: bool = False,
        has_emerald_key: bool = False,
        has_sapphire_key: bool = False,
        highest_unlocked_ascension: int | None = None,
        note_for_yourself_card_id: str | None = None,
        note_for_yourself_upgrades: int | None = None,
        note_for_yourself_preferences_dir: str | None = None,
        neow_mini_blessing: bool = False,
        enable_neow: bool = True,
        start_on_map: bool = False,
        **_: Any,
    ) -> None:
        self.seed = int(seed)
        self.ascension_level = int(ascension_level)
        self.enable_neow = bool(enable_neow)
        self.neow_mini_blessing = bool(neow_mini_blessing)
        self.start_on_map = bool(start_on_map)
        self.player_class = str(character)
        self.final_act_available = bool(final_act_available)
        self.has_ruby_key = bool(has_ruby_key)
        self.has_emerald_key = bool(has_emerald_key)
        self.has_sapphire_key = bool(has_sapphire_key)
        self.highest_unlocked_ascension = None if highest_unlocked_ascension is None else int(highest_unlocked_ascension)
        note_preference = note_for_yourself_preference(note_for_yourself_preferences_dir)
        self.note_for_yourself_card_id = str(
            note_preference.card_id if note_for_yourself_card_id is None else note_for_yourself_card_id
        )
        self.note_for_yourself_upgrades = int(
            note_preference.upgrades if note_for_yourself_upgrades is None else note_for_yourself_upgrades
        )
        self.note_for_yourself_preference_source = note_preference.source_path
        self.starting_profile = starting_profile(self.player_class)
        self.randoms = NativeRandomSet(seed=self.seed)
        self.player = PlayerState(
            current_hp=int(self.starting_profile.current_hp),
            max_hp=int(self.starting_profile.max_hp),
            base_energy=int(self.starting_profile.base_energy),
            draw_per_turn=int(self.starting_profile.card_draw),
        )
        self.deck = list(starter_deck(self.player_class))
        self.relics = list(starter_relics(self.player_class))
        self.max_potion_slots = 3
        self.potions: list[dict[str, Any]] = _normalize_potion_slots([], self.max_potion_slots)
        self.gold = int(self.starting_profile.gold)
        self.shop_purge_base_cost = int(shop_rules().purge_cost)
        self.floor = 0
        self.act = 1
        self.dungeon_id = dungeon_id_for_act(self.act)
        self.phase = "MAP" if self.start_on_map else ("NEOW" if self.enable_neow else "MAP")
        self.current_room_type = "NeowRoom" if self.phase == "NEOW" else "None"
        self.current_map_node: tuple[int, int] | None = None
        self.first_room_chosen = False
        self.map = generate_act_map(
            self.randoms,
            act=self.act,
            ascension_level=self.ascension_level,
            final_act_available=self.final_act_available,
            has_emerald_key=self.has_emerald_key,
        )
        self.current_map_node = None
        self.monster_list, self.elite_monster_list, self.boss_list = generate_monster_lists_for_dungeon(
            self.randoms,
            self.dungeon_id,
        )
        self.act_boss = self.boss_list[0] if self.boss_list else None
        self.card_pools = initialize_runtime_card_pools(self.player_class)
        self.relic_pools = initialize_relic_pools(
            self.randoms,
            owned_relic_ids={str(relic.get("relic_id") or relic.get("id")) for relic in self.relics},
            character=self.player_class,
        )
        self.source_card_pools = initialize_source_card_pools(self.card_pools, character=self.player_class)
        self.boss_relic_pool = self.relic_pools.setdefault("BOSS", [])
        self.card_blizz_randomizer = card_blizz_rules().start_offset
        self.blizzard_potion_mod = 0
        self.question_elite_chance = BASE_ELITE_CHANCE
        self.question_monster_chance = BASE_MONSTER_CHANCE
        self.question_shop_chance = BASE_SHOP_CHANCE
        self.question_treasure_chance = BASE_TREASURE_CHANCE
        self.special_one_time_event_list = initialize_special_one_time_event_list(
            ascension_level=self.ascension_level,
            highest_unlocked_ascension=self.highest_unlocked_ascension,
        )
        self.event_list, self.shrine_list = initialize_event_pools_for_dungeon(self.dungeon_id)
        self.neow_options = generate_blessing_options(
            self.randoms,
            current_hp=self.player.current_hp,
            max_hp=self.player.max_hp,
            mini_blessing=self.neow_mini_blessing,
        ) if self.phase == "NEOW" else []
        self.neow_pending_continue = False
        self.neow_reward_context = False
        self.event_reward_context = False
        self.current_event: EventState | None = None
        self.current_campfire: CampfireState | None = None
        self.current_shop: ShopState | None = None
        self.current_treasure: TreasureState | None = None
        self.current_card_select: dict[str, Any] | None = None
        self.pending_show_card_obtain_effects: list[dict[str, Any]] = []
        self.pending_event_rewards: dict[str, Any] | None = None
        self.boss_relic_options: list[dict[str, Any]] = []
        self.reward_cards: list[dict[str, Any]] = []
        self.reward_card_groups: list[list[dict[str, Any]]] = []
        self.reward_gold: int | None = None
        self.reward_stolen_gold: int | None = None
        self.reward_potions: list[dict[str, Any]] = []
        self.reward_potion_replacement_target_id: str | None = None
        self.reward_relics: list[dict[str, Any]] = []
        self.reward_emerald_key = False
        self.reward_sapphire_key_relic_id: str | None = None
        self.reward_order: list[dict[str, Any]] = []
        self.pending_cursed_key_chest_curse: dict[str, Any] | None = None
        self.reward_card_screen_open = False
        self.reward_card_reward_declined = False
        self.boss_reward_pending_boss_relic = False
        self.boss_relic_pending_act_advance = False
        self.reward_return_phase: str | None = None
        self.reward_return_room_type: str | None = None
        self.reward_return_neow_continue = False
        self.combat: NativeCombatEnv | None = None
        self._core_state = RunState(
            seed=self.seed,
            ascension_level=self.ascension_level,
            act=self.act,
            dungeon_id=self.dungeon_id,
            floor=self.floor,
            phase=self.phase,
            player=self.player,
            deck=list(self.deck),
            relics=list(self.relics),
            potions=list(self.potions),
            gold=self.gold,
            has_ruby_key=self.has_ruby_key,
            has_emerald_key=self.has_emerald_key,
            has_sapphire_key=self.has_sapphire_key,
            common_card_pool=list(self.card_pools.get("CLASS_COMMON", [])),
            uncommon_card_pool=list(self.card_pools.get("CLASS_UNCOMMON", [])),
            rare_card_pool=list(self.card_pools.get("CLASS_RARE", [])),
            colorless_card_pool=[*list(self.card_pools.get("COLORLESS_UNCOMMON", [])), *list(self.card_pools.get("COLORLESS_RARE", []))],
            curse_card_pool=list(self.card_pools.get("CURSE", [])),
            common_relic_pool=list(self.relic_pools.get("COMMON", [])),
            uncommon_relic_pool=list(self.relic_pools.get("UNCOMMON", [])),
            rare_relic_pool=list(self.relic_pools.get("RARE", [])),
            shop_relic_pool=list(self.relic_pools.get("SHOP", [])),
            src_common_card_pool=list(self.source_card_pools.get("SRC_COMMON", [])),
            src_uncommon_card_pool=list(self.source_card_pools.get("SRC_UNCOMMON", [])),
            src_rare_card_pool=list(self.source_card_pools.get("SRC_RARE", [])),
            src_colorless_card_pool=list(self.source_card_pools.get("SRC_COLORLESS", [])),
            src_curse_card_pool=list(self.source_card_pools.get("SRC_CURSE", [])),
            act_boss=self.act_boss,
            boss_relic_pool=list(self.boss_relic_pool),
            implementation_status=V3_IMPLEMENTATION_STATUS,
        )
        self.reference_sources = dict(CORE_SOURCE_MAP["run"])

    def _reset_player_post_combat_state(self) -> None:
        self.player.block = 0
        self.player.energy = int(self.player.base_energy)
        self.player.powers = []

    def _combat_pending_select(self) -> dict[str, Any] | None:
        if self.combat is None:
            return None
        pending = getattr(getattr(self.combat, "engine", None), "pending_card_select", None)
        return pending if isinstance(pending, dict) else None

    def _combat_pending_discovery_reward(self) -> bool:
        pending = self._combat_pending_select()
        return str((pending or {}).get("mode") or "").upper() in {"DISCOVERY", "NILRYS_CODEX"}

    def state(self) -> dict[str, Any]:
        fast_terminal_state = getattr(self, "_teacher_fast_terminal_state", None)
        if isinstance(fast_terminal_state, dict):
            return dict(fast_terminal_state)
        self._sync_state()
        if self.phase in {"COMBAT", "CARD_SELECT", "CARD_REWARD"} and self._combat_pending_select() is not None:
            return self._combat_state_with_run_context()
        if self.phase == "COMBAT" and self.combat is not None:
            return self._combat_state_with_run_context()
        payload = serialize_run_state(self._core_state)
        actions = self.legal_actions()
        if self.phase == "CARD_REWARD":
            payload["screen_state"] = self._reward_card_screen_state()
        elif self.phase == "CARD_SELECT":
            payload["screen_state"] = self._card_select_screen_state()
        elif self.phase == "MAP":
            payload["screen_state"] = self._map_screen_state(actions)
        elif self.phase == "EVENT":
            payload["screen_state"] = self._event_screen_state()
        elif self.phase == "SHOP":
            payload["screen_state"] = self._shop_screen_state()
        elif self.phase == "TREASURE":
            payload["screen_state"] = self._treasure_screen_state()
        if self.phase in {"COMPLETE", "GAME_OVER"}:
            payload["room_type"] = self.current_room_type
        payload["map_state"] = self._map_state()
        payload["rng_state"] = self.randoms.debug_state()
        payload["choice_available"] = bool(actions)
        payload["choice_list"] = [
            action
            for action in actions
            if not (self.phase == "CARD_REWARD" and str(action.get("kind") or "").lower() in {"skip", "proceed"})
        ]
        payload["commands"] = {
            "cancel": False,
            "end": False,
            "play": False,
            "potion": False,
            "proceed": self.phase == "CARD_REWARD",
        }
        return payload

    def _combat_state_with_run_context(self) -> dict[str, Any]:
        payload = self.combat.serialize() if self.combat is not None else {}
        run_payload = serialize_run_state(self._core_state)
        for key in (
            "seed",
            "ascension_level",
            "act",
            "act_boss",
            "dungeon_id",
            "floor",
            "has_ruby_key",
            "has_emerald_key",
            "has_sapphire_key",
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
            "common_relic_pool",
            "uncommon_relic_pool",
            "rare_relic_pool",
            "shop_relic_pool",
            "boss_relic_pool",
            "implementation_status",
        ):
            if key not in payload or payload.get(key) in (None, [], ""):
                payload[key] = run_payload.get(key)
        payload["map_state"] = self._map_state()
        payload["rng_state"] = self.randoms.debug_state()
        return payload

    def _map_state(self) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        for row in self.map:
            for node in row:
                if node.room_symbol is None and not node.has_edges():
                    continue
                nodes.append(
                    {
                        "x": int(node.x),
                        "y": int(node.y),
                        "symbol": node.room_symbol,
                        "emerald": bool(node.has_emerald_key),
                        "children": [
                            {"x": int(edge.dst_x), "y": int(edge.dst_y)}
                            for edge in node.edges
                        ],
                    }
                )
        return {
            "act": int(self.act),
            "current_node": (
                {"x": int(self.current_map_node[0]), "y": int(self.current_map_node[1])}
                if self.current_map_node is not None
                else None
            ),
            "first_room_chosen": bool(self.first_room_chosen),
            "nodes": nodes,
        }

    def _map_screen_state(self, actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if actions is None:
            actions = self.legal_actions()
        boss_available = any(str(action.get("symbol") or "").upper() == "BOSS" for action in actions)
        screen_state: dict[str, Any] = {"boss_available": boss_available}
        if self.current_map_node is not None:
            x, y = self.current_map_node
            screen_state["current_node"] = {"x": int(x), "y": int(y)}
        elif not self.first_room_chosen:
            # The real map screen exposes a synthetic anchor before the first
            # room choice. It is visible-only; legal first-row choices do not
            # use it for path validation.
            screen_state["current_node"] = (
                {"x": -1, "y": 15}
                if int(self.act) >= 3 and int(self.floor) == self._act_floor_offset()
                else {"x": 0, "y": -1}
            )
        return screen_state

    def _event_screen_state(self) -> dict[str, Any]:
        if self.current_event is None:
            return {"options": []}
        if self.current_event.event_id == "Addict":
            if self.current_event.screen == "INTRO":
                options: list[dict[str, Any]] = []
                enabled_index = 0
                if int(self.gold) >= 85:
                    options.append(
                        {
                            "label": "Pay 85 Gold",
                            "text": "[Pay] 85 Gold. Obtain a Relic.",
                            "disabled": False,
                            "choice_index": enabled_index,
                        }
                    )
                    enabled_index += 1
                else:
                    options.append(
                        {
                            "label": "Locked",
                            "text": "[Locked] Requires: 85 Gold.",
                            "disabled": True,
                        }
                    )
                options.append(
                    {
                        "label": "Rob",
                        "text": "[Rob] Obtain a Relic. Become Cursed - Shame.",
                        "disabled": False,
                        "choice_index": enabled_index,
                    }
                )
                enabled_index += 1
                options.append({"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": enabled_index})
                return {"event_id": "Addict", "event_name": None, "options": options}
            return {
                "event_id": "Addict",
                "event_name": None,
                "options": [{"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": 0}],
            }
        if self.current_event.event_id == "WeMeetAgain":
            if self.current_event.screen == "INTRO":
                options: list[dict[str, Any]] = []
                enabled_index = 0
                if self.current_event.data.get("potion_index") is not None:
                    options.append(
                        {
                            "label": "Give Potion",
                            "text": "[Give Potion] Lose a Potion. Obtain a Relic.",
                            "disabled": False,
                            "choice_index": enabled_index,
                        }
                    )
                    enabled_index += 1
                else:
                    options.append(
                        {
                            "label": "Locked",
                            "text": "[Locked] Requires: Potion.",
                            "disabled": True,
                        }
                    )
                gold_amount = int(self.current_event.data.get("gold_amount") or 0)
                if gold_amount > 0:
                    options.append(
                        {
                            "label": "Give Gold",
                            "text": f"[Give Gold] Lose {gold_amount} Gold. Obtain a Relic.",
                            "disabled": False,
                            "choice_index": enabled_index,
                        }
                    )
                    enabled_index += 1
                else:
                    options.append(
                        {
                            "label": "Locked",
                            "text": "[Locked] Requires: Gold.",
                            "disabled": True,
                        }
                    )
                if self.current_event.data.get("card_uuid") is not None:
                    options.append(
                        {
                            "label": "Give Card",
                            "text": "[Give Card] Lose a Card. Obtain a Relic.",
                            "disabled": False,
                            "choice_index": enabled_index,
                        }
                    )
                    enabled_index += 1
                else:
                    options.append(
                        {
                            "label": "Locked",
                            "text": "[Locked] Requires: Non-Basic Card.",
                            "disabled": True,
                        }
                    )
                options.append(
                    {
                        "label": "Attack",
                        "text": "[Attack]",
                        "disabled": False,
                        "choice_index": enabled_index,
                    }
                )
                return {"event_id": "WeMeetAgain", "event_name": None, "options": options}
            return {
                "event_id": "WeMeetAgain",
                "event_name": None,
                "options": [{"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": 0}],
            }
        if self.current_event.event_id == "The Cleric":
            if self.current_event.screen == "INTRO":
                options: list[dict[str, Any]] = []
                enabled_index = 0
                gold = int(self.gold)
                heal_amt = int(self.player.max_hp * 0.25)
                purify_cost = 75 if int(self.ascension_level) >= 15 else 50
                if gold >= 35:
                    options.append(
                        {
                            "label": "Heal",
                            "text": f"[Heal] 35 Gold: Heal {heal_amt} HP.",
                            "disabled": False,
                            "choice_index": enabled_index,
                        }
                    )
                    enabled_index += 1
                else:
                    options.append(
                        {
                            "label": "Locked",
                            "text": "[Locked] Requires: 35 Gold.",
                            "disabled": True,
                        }
                    )
                if gold >= 50:
                    option = {
                        "label": "Purify",
                        "text": f"[Purify] {purify_cost} Gold: Remove a card from your deck.",
                        "disabled": gold < purify_cost,
                    }
                    if not option["disabled"]:
                        option["choice_index"] = enabled_index
                        enabled_index += 1
                    options.append(option)
                else:
                    options.append(
                        {
                            "label": "Locked",
                            "text": "[Locked] Requires: 50 Gold.",
                            "disabled": True,
                        }
                    )
                options.append({"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": enabled_index})
                return {"event_id": "The Cleric", "event_name": "The Cleric", "options": options}
            return {
                "event_id": "The Cleric",
                "event_name": "The Cleric",
                "options": [{"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": 0}],
            }
        if self.current_event.event_id == "Golden Wing" and self.current_event.screen == "INTRO":
            can_attack = any(str(card.get("type") or "") == "ATTACK" and int(card.get("base_damage") or 0) >= 10 for card in self.deck)
            options: list[dict[str, Any]] = [
                {
                    "label": "Pray",
                    "text": "[Pray] Remove a card from your deck. Lose 7 HP.",
                    "disabled": False,
                    "choice_index": 0,
                }
            ]
            if can_attack:
                options.append(
                    {
                        "label": "Smash",
                        "text": "[Smash] Gain 50 - 80 Gold.",
                        "disabled": False,
                        "choice_index": 1,
                    }
                )
                leave_choice_index = 2
            else:
                options.append(
                    {
                        "label": "Locked",
                        "text": "[Locked] Requires: Card with 10 or more damage.",
                        "disabled": True,
                    }
                )
                leave_choice_index = 1
            options.append({"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": leave_choice_index})
            return {"event_id": "Golden Wing", "event_name": "Wing Statue", "options": options}
        if self.current_event.event_id == "Golden Wing" and self.current_event.screen == "PURGE":
            return {
                "event_id": "Golden Wing",
                "event_name": "Wing Statue",
                "options": [{"label": "Continue", "text": "[Continue]", "disabled": False, "choice_index": 0}],
            }
        if self.current_event.event_id == "Golden Wing":
            return {
                "event_id": "Golden Wing",
                "event_name": "Wing Statue",
                "options": [{"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": 0}],
            }
        if self.current_event.event_id == "World of Goop":
            if self.current_event.screen == "INTRO":
                gold_loss = int(self.current_event.data.get("gold_loss") or 0)
                return {
                    "event_id": "World of Goop",
                    "event_name": "World of Goop",
                    "options": [
                        {
                            "label": "Gather Gold",
                            "text": "[Gather Gold] Gain 75 Gold. Lose 11 HP.",
                            "disabled": False,
                            "choice_index": 0,
                        },
                        {
                            "label": "Leave It",
                            "text": f"[Leave It] Lose {gold_loss} Gold.",
                            "disabled": False,
                            "choice_index": 1,
                        },
                    ],
                }
            return {
                "event_id": "World of Goop",
                "event_name": "World of Goop",
                "options": [{"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": 0}],
            }
        if self.current_event.event_id == "Big Fish":
            if self.current_event.screen == "INTRO":
                heal_amt = self.player.max_hp // 3
                return {
                    "event_id": "Big Fish",
                    "event_name": "Big Fish",
                    "options": [
                        {"label": "Banana", "text": f"[Banana] Heal {heal_amt} HP.", "disabled": False, "choice_index": 0},
                        {"label": "Donut", "text": "[Donut] Max HP +5.", "disabled": False, "choice_index": 1},
                        {"label": "Box", "text": "[Box] Obtain a Relic. Become Cursed - Regret.", "disabled": False, "choice_index": 2},
                    ],
                }
            return {
                "event_id": "Big Fish",
                "event_name": "Big Fish",
                "options": [{"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": 0}],
            }
        if self.current_event.event_id == "Purifier":
            if self.current_event.screen == "INTRO":
                return {
                    "event_id": "Purifier",
                    "event_name": "Purifier",
                    "options": [
                        {"label": "Pray", "text": "[Pray] Remove a card from your deck.", "disabled": False, "choice_index": 0},
                        {"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": 1},
                    ],
                }
            return {
                "event_id": "Purifier",
                "event_name": "Purifier",
                "options": [{"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": 0}],
            }
        if self.current_event.event_id == "Tomb of Lord Red Mask":
            if self.current_event.screen == "INTRO":
                if any(str(relic.get("relic_id") or relic.get("id")) == "Red Mask" for relic in self.relics):
                    options = [
                        {
                            "label": "Wear Mask",
                            "text": "[Don the Red Mask] Gain 222 Gold.",
                            "disabled": False,
                            "choice_index": 0,
                        },
                        {"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": 1},
                    ]
                else:
                    options = [
                        {
                            "label": "Locked",
                            "text": "[Locked] Requires: Red Mask.",
                            "disabled": True,
                        },
                        {
                            "label": f"Offer: {self.gold} Gold",
                            "text": f"[Offer: {self.gold} Gold] Lose all Gold. Obtain a Relic.",
                            "disabled": False,
                            "choice_index": 0,
                        },
                        {"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": 1},
                    ]
                return {"event_id": "Tomb of Lord Red Mask", "event_name": "Tomb of Lord Red Mask", "options": options}
            return {
                "event_id": "Tomb of Lord Red Mask",
                "event_name": "Tomb of Lord Red Mask",
                "options": [{"label": "Leave", "text": "[Leave]", "disabled": False, "choice_index": 0}],
            }
        if self.current_event.event_id == "Forgotten Altar" and self.current_event.screen == "INTRO":
            hp_loss = int(self.current_event.data.get("hp_loss") or 0)
            if any(str(relic.get("relic_id") or relic.get("id")) == "Golden Idol" for relic in self.relics):
                options = [
                    {"label": "Offer Idol", "text": "[Offer] Lose Golden Idol. Obtain a Relic.", "disabled": False, "choice_index": 0},
                    {"label": "Shed Blood", "text": f"[Shed Blood] Max HP +5. Lose {hp_loss} HP.", "disabled": False, "choice_index": 1},
                    {"label": "Smash", "text": "[Smash] Become Cursed - Decay.", "disabled": False, "choice_index": 2},
                ]
            else:
                options = [
                    {"label": "Locked", "text": "[Locked] Requires: Golden Idol.", "disabled": True},
                    {"label": "Shed Blood", "text": f"[Shed Blood] Max HP +5. Lose {hp_loss} HP.", "disabled": False, "choice_index": 0},
                    {"label": "Smash", "text": "[Smash] Become Cursed - Decay.", "disabled": False, "choice_index": 1},
                ]
            return {"event_id": "Forgotten Altar", "event_name": None, "options": options}
        return {
            "event_id": self.current_event.event_id,
            "event_name": None,
            "options": [dict(action) for action in self.legal_actions()],
        }

    def _card_select_screen_state(self) -> dict[str, Any]:
        if self.current_card_select is None:
            return {"cards": []}
        mode = str(self.current_card_select.get("mode") or "")
        cards: list[dict[str, Any]]
        if self.current_card_select.get("cards") is not None:
            cards = [dict(card) for card in list(self.current_card_select.get("cards") or [])]
        else:
            cards = []
            for target_index in list(self.current_card_select.get("candidate_indexes") or []):
                index = int(target_index)
                if 0 <= index < len(self.deck):
                    cards.append(dict(self.deck[index]))
        required_picks = int(
            self.current_card_select.get("total_picks")
            or self.current_card_select.get("remaining_picks")
            or 1
        )
        return {
            "cards": cards,
            "selected_cards": [dict(card) for card in list(self.current_card_select.get("selected_cards") or [])],
            "num_cards": required_picks,
            "max_cards": required_picks,
            "any_number": False,
            "can_pick_zero": False,
            "for_upgrade": mode == "upgrade",
            "for_transform": bool(self.current_card_select.get("visible_for_transform", mode == "transform")),
            "for_purge": bool(self.current_card_select.get("visible_for_purge", mode in {"purge", "remove"})),
            "confirm_up": bool(self.current_card_select.get("confirm_up", False)),
        }

    def _shop_potion_screen_payload(self, potion: dict[str, Any]) -> dict[str, Any]:
        payload = self._reward_potion_screen_payload(potion)
        payload["price"] = int(potion.get("price") or 0)
        payload["can_use"] = bool(potion.get("can_use_out_of_combat", False))
        return payload

    def _shop_screen_state(self) -> dict[str, Any]:
        if self.current_shop is None:
            return {"cards": [], "relics": [], "potions": [], "purge_available": False, "purge_cost": None}
        cards = []
        for card in self.current_shop.cards:
            payload = dict(card)
            payload["name"] = _visible_shop_card_name(payload)
            cards.append(payload)
        return {
            "cards": cards,
            "relics": [dict(relic) for relic in self.current_shop.relics],
            "potions": [self._shop_potion_screen_payload(dict(potion)) for potion in self.current_shop.potions],
            "purge_available": bool(self.deck and self.current_shop.purge_available),
            "purge_cost": int(self.current_shop.purge_cost),
        }

    def _treasure_screen_state(self) -> dict[str, Any]:
        if self.current_treasure is None:
            return {}
        return {
            "chest_type": self.current_treasure.chest_type,
            "chest_open": bool(self.current_treasure.opened),
        }

    def _reward_potion_screen_payload(self, potion: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": potion.get("id") or potion.get("potion_id") or potion.get("name"),
            "potion_id": potion.get("potion_id") or potion.get("id") or potion.get("name"),
            "name": potion.get("name") or potion.get("potion_id") or potion.get("id"),
            "requires_target": bool(potion.get("requires_target", False)),
            "can_use": False,
            "can_discard": bool(potion.get("can_discard", True)),
        }

    def _pending_reward_card_group_count(self) -> int:
        if not self.reward_cards:
            return 0
        return 1 + sum(1 for group in self.reward_card_groups if group)

    def _open_reward_card_group(self, reward_index: int) -> bool:
        if not self.reward_cards:
            return False
        groups = [[dict(card) for card in self.reward_cards]]
        groups.extend([[dict(card) for card in group] for group in self.reward_card_groups if group])
        if reward_index < 0 or reward_index >= len(groups):
            return False
        selected = groups.pop(reward_index)
        self.reward_cards = apply_reward_preview_relics(selected, owned_relic_ids=self._current_relic_ids())
        self.reward_card_groups = [[dict(card) for card in group] for group in groups]
        self.reward_card_screen_open = True
        self.reward_card_reward_declined = False
        return True

    def _reward_card_screen_state(self) -> dict[str, Any]:
        relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
        screen_state: dict[str, Any] = {
            "skip_available": True,
            "bowl_available": bool(self.reward_card_screen_open and self.reward_cards and not self.neow_reward_context and "Singing Bowl" in relic_ids),
        }
        if self.reward_card_screen_open and self.reward_cards:
            screen_state["cards"] = [dict(card) for card in self.reward_cards]
            return screen_state
        if self.reward_order:
            screen_state["rewards"] = self._ordered_reward_screen_items()
            return screen_state
        rewards: list[dict[str, Any]] = []
        if self.reward_stolen_gold is not None:
            rewards.append({"reward_type": "STOLEN_GOLD", "gold": int(self.reward_stolen_gold)})
        if self.reward_gold is not None:
            rewards.append({"reward_type": "GOLD", "gold": int(self.reward_gold)})
        for relic in self.reward_relics:
            rewards.append({"reward_type": "RELIC", "relic": dict(relic)})
        if self.reward_sapphire_key_relic_id is not None:
            linked_relic = next(
                (relic for relic in self.reward_relics if str(relic.get("relic_id")) == self.reward_sapphire_key_relic_id),
                None,
            )
            if linked_relic is not None:
                rewards.append({"reward_type": "SAPPHIRE_KEY", "link": dict(linked_relic)})
        if self.reward_emerald_key:
            rewards.append({"reward_type": "EMERALD_KEY"})
        for potion in self.reward_potions:
            rewards.append({"reward_type": "POTION", "potion": self._reward_potion_screen_payload(potion)})
        for _ in range(self._pending_reward_card_group_count()):
            rewards.append({"reward_type": "CARD"})
        screen_state["rewards"] = rewards
        return screen_state

    def _reward_relic_by_id(self, relic_id: str | None) -> dict[str, Any] | None:
        if relic_id is None:
            return None
        return next((relic for relic in self.reward_relics if str(relic.get("relic_id")) == str(relic_id)), None)

    def _ordered_reward_action(self, item: dict[str, Any], choice_index: int) -> dict[str, Any] | None:
        kind = str(item.get("kind") or "")
        if kind == "reward_gold" and self.reward_gold is not None:
            return {
                "kind": "reward_gold",
                "name": "GOLD",
                "reward_type": "GOLD",
                "choice_index": choice_index,
                "amount": int(self.reward_gold),
                "reward_index": 0,
            }
        if kind == "reward_relic":
            relic = self._reward_relic_by_id(str(item.get("relic_id") or ""))
            if relic is None:
                return None
            return {
                "kind": "reward_relic",
                "name": relic["name"],
                "relic_id": relic["relic_id"],
                "choice_index": choice_index,
                "reward_index": 0,
            }
        if kind == "reward_key" and item.get("key") == "sapphire" and self.reward_sapphire_key_relic_id is not None:
            linked_relic = self._reward_relic_by_id(self.reward_sapphire_key_relic_id)
            if linked_relic is None:
                return None
            return {
                "kind": "reward_key",
                "name": "SAPPHIRE_KEY",
                "key": "sapphire",
                "choice_index": choice_index,
                "reward_index": 0,
                "relic_id": self.reward_sapphire_key_relic_id,
                "linked_relic_id": self.reward_sapphire_key_relic_id,
                "linked_relic_name": linked_relic["name"],
            }
        return None

    def _ordered_reward_actions(self) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for item in self.reward_order:
            action = self._ordered_reward_action(item, len(actions))
            if action is not None:
                actions.append(action)
        return actions

    def _ordered_reward_screen_items(self) -> list[dict[str, Any]]:
        rewards: list[dict[str, Any]] = []
        for item in self.reward_order:
            kind = str(item.get("kind") or "")
            if kind == "reward_gold" and self.reward_gold is not None:
                rewards.append({"reward_type": "GOLD", "gold": int(self.reward_gold)})
                continue
            if kind == "reward_relic":
                relic = self._reward_relic_by_id(str(item.get("relic_id") or ""))
                if relic is not None:
                    rewards.append({"reward_type": "RELIC", "relic": dict(relic)})
                continue
            if kind == "reward_key" and item.get("key") == "sapphire" and self.reward_sapphire_key_relic_id is not None:
                linked_relic = self._reward_relic_by_id(self.reward_sapphire_key_relic_id)
                if linked_relic is not None:
                    rewards.append({"reward_type": "SAPPHIRE_KEY", "link": dict(linked_relic)})
        return rewards

    def _reward_cards_should_open_directly(self) -> bool:
        return bool(
            self.neow_reward_context
            or self.event_reward_context
            or self.reward_return_phase is not None
            or self.current_room_type in {"RestRoom"}
        )

    def _offer_sapphire_key_for_last_reward_relic(self, *, append_to_order: bool = False) -> None:
        self.reward_sapphire_key_relic_id = None
        if not self.final_act_available or self.has_sapphire_key or not self.reward_relics:
            return
        self.reward_sapphire_key_relic_id = str(self.reward_relics[-1].get("relic_id"))
        if append_to_order:
            self.reward_order.append(
                {
                    "kind": "reward_key",
                    "key": "sapphire",
                    "relic_id": self.reward_sapphire_key_relic_id,
                }
            )

    def _has_pending_reward_items(self) -> bool:
        return bool(
            self.reward_cards
            or self.reward_card_groups
            or self.reward_gold is not None
            or self.reward_stolen_gold is not None
            or self.reward_potions
            or self.reward_relics
            or self.reward_emerald_key
            or self.reward_sapphire_key_relic_id is not None
        )

    def _open_boss_relic_reward(self) -> None:
        self.boss_reward_pending_boss_relic = False
        self.boss_relic_options = draw_boss_relic_choices(
            self.boss_relic_pool,
            act_num=int(self.act),
            owned_relic_ids={str(relic.get("relic_id") or relic.get("id") or "") for relic in self.relics},
        )
        self.phase = "BOSS_RELIC"
        self.current_room_type = "TreasureRoomBoss"

    def _open_boss_chest_room(self) -> None:
        self.floor = self._act_floor_offset() + 17
        self.randoms.reset_floor_streams(self.floor)
        self.current_treasure = TreasureState(chest_type="BossChest", gold_amount=0, relic_tier="BOSS")
        self.phase = "TREASURE"
        self.current_room_type = "TreasureRoomBoss"
        self._handle_room_entry_relics(self.current_room_type)

    def _wait_on_boss_chest_after_relic_choice(self) -> None:
        self.boss_relic_pending_act_advance = True
        self.current_treasure = TreasureState(chest_type="BossChest", gold_amount=0, relic_tier="BOSS", opened=True)
        self.phase = "TREASURE"
        self.current_room_type = "TreasureRoomBoss"

    def _finish_empty_boss_combat_reward(self) -> bool:
        if not self.boss_reward_pending_boss_relic:
            return False
        self.phase = "CARD_REWARD"
        self.current_room_type = "CARD_REWARD"
        self.reward_card_screen_open = False
        return True

    def _setup_boss_combat_reward(self, *, bonus_reward_gold: int = 0) -> None:
        owned_relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
        tmp_gold = 100 + int(self.randoms.stream("misc").random(-5, 5))
        if self.ascension_level >= 13:
            tmp_gold = int(float(tmp_gold) * 0.75 + 0.5)
        if "Golden Idol" in owned_relic_ids:
            tmp_gold += int(float(tmp_gold) * 0.25 + 0.5)
        self.reward_gold = int(tmp_gold) or None
        self.reward_stolen_gold = int(bonus_reward_gold) or None
        self.reward_relics = []
        self.reward_emerald_key = False
        self.reward_sapphire_key_relic_id = None
        potion, next_blizzard_potion_mod = roll_post_combat_potion(
            self.randoms,
            reward_count=1,
            blizzard_potion_mod=self.blizzard_potion_mod,
            owned_relic_ids=owned_relic_ids,
            player_class=self.player_class,
        )
        self.blizzard_potion_mod = int(next_blizzard_potion_mod)
        self.reward_potions = [dict(potion)] if potion is not None else []
        card_upgraded_chance = act_chances(str(self.dungeon_id)).card_upgraded_chance(self.ascension_level)
        card_groups, self.card_blizz_randomizer = generate_card_reward_groups_with_state(
            self.randoms,
            group_count=1,
            card_blizz_randomizer=self.card_blizz_randomizer,
            card_upgraded_chance=card_upgraded_chance,
            rare_chance=100,
            uncommon_chance=0,
            owned_relic_ids=owned_relic_ids,
            runtime_card_pools=self.card_pools,
        )
        self.reward_cards = [dict(card) for card in card_groups[0]] if card_groups else []
        self.reward_card_groups = []
        self.reward_card_screen_open = False
        self.reward_card_reward_declined = False
        self.boss_reward_pending_boss_relic = True
        self.phase = "CARD_REWARD"
        self.current_room_type = "MonsterRoomBoss"

    def _setup_event_combat_reward(self, payload: dict[str, Any], *, bonus_reward_gold: int = 0) -> None:
        owned_relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
        event_gold = int(payload.get("gold") or 0)
        if event_gold > 0 and "Golden Idol" in owned_relic_ids:
            event_gold += int(float(event_gold) * 0.25 + 0.5)
        self.reward_gold = event_gold or None
        self.reward_stolen_gold = int(bonus_reward_gold) or None
        self.reward_relics = [dict(relic) for relic in list(payload.get("relics") or [])]
        self.reward_emerald_key = False
        self.reward_sapphire_key_relic_id = None

        reward_count = int(self.reward_gold is not None) + int(self.reward_stolen_gold is not None)
        reward_count += len(self.reward_relics)
        reward_potions = [dict(potion) for potion in list(payload.get("potions") or [])]
        reward_count += len(reward_potions)
        potion, next_blizzard_potion_mod = roll_post_combat_potion(
            self.randoms,
            reward_count=reward_count,
            blizzard_potion_mod=self.blizzard_potion_mod,
            owned_relic_ids=owned_relic_ids,
            player_class=self.player_class,
        )
        self.blizzard_potion_mod = int(next_blizzard_potion_mod)
        if potion is not None:
            reward_potions.append(dict(potion))
        self.reward_potions = reward_potions

        card_groups = [list(group) for group in list(payload.get("card_groups") or [])]
        cards = list(card_groups[0]) if card_groups else list(payload.get("cards") or [])
        if not cards and not card_groups and bool(payload.get("auto_card_reward", True)):
            card_upgraded_chance = act_chances(str(self.dungeon_id)).card_upgraded_chance(self.ascension_level)
            card_groups, self.card_blizz_randomizer = generate_card_reward_groups_with_state(
                self.randoms,
                group_count=1,
                card_blizz_randomizer=self.card_blizz_randomizer,
                card_upgraded_chance=card_upgraded_chance,
                owned_relic_ids=owned_relic_ids,
                runtime_card_pools=self.card_pools,
            )
            cards = list(card_groups[0]) if card_groups else []
        self.reward_cards = [dict(card) for card in cards]
        self.reward_card_groups = [
            [dict(card) for card in group]
            for group in card_groups[1:]
        ]
        self.reward_card_screen_open = False
        self.reward_card_reward_declined = False
        self.phase = "CARD_REWARD"

    def legal_actions(self) -> list[dict[str, Any]]:
        if self._combat_pending_discovery_reward():
            return self.combat.legal_actions()
        if self.phase == "COMBAT" and self.combat is not None:
            return self.combat.legal_actions()
        if self.phase == "EVENT" and self.current_event is not None:
            return self.current_event.actions(
                ascension_level=self.ascension_level,
                max_hp=self.player.max_hp,
                gold=self.gold,
                deck=self.deck,
                relics=self.relics,
                potions=self.potions,
            )
        if self.phase == "CAMPFIRE" and self.current_campfire is not None:
            actions = self.current_campfire.actions(deck=self.deck, relics=self.relics)
            if actions:
                return actions
            return [{"kind": "campfire", "name": "proceed", "label": "proceed", "choice_index": 0}]
        if self.phase == "NEOW":
            if self.neow_pending_continue:
                return [{"kind": "neow", "name": "OPTION_0", "label": "OPTION_0", "choice_index": 0, "bonus": "CONTINUE", "drawback": "NONE", "bonus_text": "Continue."}]
            return [option.to_action() for option in self.neow_options]
        if self.phase == "CARD_REWARD":
            actions: list[dict[str, Any]] = []
            if self.reward_order:
                actions.extend(self._ordered_reward_actions())
            elif self.reward_stolen_gold is not None:
                actions.append(
                    {
                        "kind": "reward_gold",
                        "name": "STOLEN_GOLD",
                        "reward_type": "STOLEN_GOLD",
                        "choice_index": len(actions),
                        "amount": int(self.reward_stolen_gold),
                        "reward_index": 0,
                    }
                )
            if not self.reward_order and self.reward_gold is not None:
                actions.append(
                    {
                        "kind": "reward_gold",
                        "name": "GOLD",
                        "reward_type": "GOLD",
                        "choice_index": len(actions),
                        "amount": int(self.reward_gold),
                        "reward_index": 0,
                    }
                )
            if not self.reward_order:
                for relic in self.reward_relics:
                    actions.append(
                        {
                            "kind": "reward_relic",
                            "name": relic["name"],
                            "relic_id": relic["relic_id"],
                            "choice_index": len(actions),
                            "reward_index": 0,
                        }
                    )
                if self.reward_sapphire_key_relic_id is not None:
                    linked_relic = next(
                        (relic for relic in self.reward_relics if str(relic.get("relic_id")) == self.reward_sapphire_key_relic_id),
                        None,
                    )
                    if linked_relic is not None:
                        actions.append(
                            {
                                "kind": "reward_key",
                                "name": "SAPPHIRE_KEY",
                                "key": "sapphire",
                                "choice_index": len(actions),
                                "reward_index": 0,
                                "relic_id": self.reward_sapphire_key_relic_id,
                                "linked_relic_id": self.reward_sapphire_key_relic_id,
                                "linked_relic_name": linked_relic["name"],
                            }
                        )
            if self.reward_emerald_key:
                actions.append(
                    {
                        "kind": "reward_key",
                        "name": "EMERALD_KEY",
                        "key": "emerald",
                        "choice_index": len(actions),
                        "reward_index": 0,
                    }
                )
            for potion in self.reward_potions:
                actions.append(
                    {
                        "kind": "reward_potion",
                        "name": potion["name"],
                        "potion_id": potion["potion_id"],
                        "potion": self._reward_potion_screen_payload(potion),
                        "choice_index": len(actions),
                        "reward_index": 0,
                    }
                )
            if self.reward_potions and not _has_open_potion_slot(self.potions, self.max_potion_slots):
                for potion_index, potion in enumerate(_normalize_potion_slots(self.potions, self.max_potion_slots)):
                    if _is_potion_slot(potion) or not bool(potion.get("can_discard", True)):
                        continue
                    actions.append(
                        {
                            "kind": "discard_potion",
                            "name": f"Discard {potion.get('name') or potion.get('potion_id') or potion_index}",
                            "potion_index": potion_index,
                            "potion": self._reward_potion_screen_payload(potion),
                        }
                    )
            if self.reward_cards and not self.reward_card_screen_open:
                for reward_index in range(self._pending_reward_card_group_count()):
                    actions.append(
                        {
                            "kind": "raw",
                            "name": "CARD",
                            "label": "CARD",
                            "choice_index": len(actions),
                            "reward_index": reward_index,
                        }
                    )
            else:
                actions.extend([
                    {
                        "kind": "card_reward",
                        "name": card["name"],
                        "card_id": card["card_id"],
                        "choice_index": len(actions) + index,
                        "reward_index": 0,
                        "card_index": index,
                        "card": dict(card),
                    }
                    for index, card in enumerate(self.reward_cards)
                ])
            relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
            if self.reward_card_screen_open and self.reward_cards and not self.neow_reward_context and "Singing Bowl" in relic_ids:
                actions.append({"kind": "singing_bowl", "name": "SINGING_BOWL", "choice_index": len(actions)})
            if self.reward_card_screen_open and self.reward_cards:
                actions.append({"kind": "skip", "name": "SKIP", "choice_index": len(actions)})
            if not self.neow_reward_context:
                actions.append({"kind": "proceed", "name": "PROCEED"})
            if self.neow_reward_context and not actions:
                actions.append({"kind": "proceed", "name": "PROCEED"})
            return actions
        if self.phase == "MAP":
            return available_map_actions(
                self.map,
                current_node=self.current_map_node,
                first_room_chosen=self.first_room_chosen,
                floor_offset=self._act_floor_offset(),
                winged_charges=int((self._winged_greaves() or {}).get("counter") or 0),
            )
        if self.phase == "SHOP" and self.current_shop is not None:
            actions = []
            relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
            for action in self.current_shop.actions():
                item_kind = action.get("item_kind")
                if item_kind == "relic" and is_banned_relic_id(action.get("item_id")):
                    continue
                price = int(action.get("price") or 0)
                if item_kind == "leave":
                    actions.append(action)
                    continue
                if item_kind == "potion" and not _has_open_potion_slot(self.potions, self.max_potion_slots):
                    continue
                if item_kind == "potion" and "Sozu" in relic_ids:
                    continue
                if item_kind == "purge":
                    if self.deck and self.gold >= price:
                        actions.append(action)
                    continue
                if self.gold >= price:
                    actions.append(action)
            return actions or [{"kind": "shop", "item_kind": "leave", "item_id": "leave", "name": "Leave", "price": 0, "label": "Leave"}]
        if self.phase == "TREASURE" and self.current_treasure is not None:
            return self.current_treasure.actions()
        if self.phase == "CARD_SELECT" and self.current_card_select is not None:
            if bool(self.current_card_select.get("confirm_up", False)):
                return [{"kind": "confirm", "name": "CONFIRM", "choice_index": 0}]
            actions: list[dict[str, Any]] = []
            select_cards = self.current_card_select.get("cards")
            if select_cards is not None:
                for choice_index, card in enumerate(list(select_cards or [])):
                    actions.append(
                        {
                            "kind": "card_select",
                            "mode": self.current_card_select.get("mode"),
                            "name": card.get("name"),
                            "card_id": card.get("card_id"),
                            "type": card.get("type"),
                            "rarity": card.get("rarity"),
                            "upgrades": int(card.get("upgrades") or 0),
                            "choice_index": choice_index,
                            "card_index": choice_index,
                        }
                    )
                return actions
            selected_target_indexes = {
                int(index) for index in list(self.current_card_select.get("selected_target_indexes") or [])
            }
            for choice_index, target_index in enumerate(self.current_card_select.get("candidate_indexes") or []):
                if int(target_index) in selected_target_indexes:
                    continue
                card = self.deck[int(target_index)]
                actions.append(
                    {
                        "kind": "card_select",
                        "mode": self.current_card_select.get("mode"),
                        "name": card.get("name"),
                        "card_id": card.get("card_id"),
                        "type": card.get("type"),
                        "rarity": card.get("rarity"),
                        "upgrades": int(card.get("upgrades") or 0),
                        "choice_index": choice_index,
                        "target_index": int(target_index),
                    }
                )
            return actions
        if self.phase == "CARD_SELECT" and self._combat_pending_select() is not None:
            return self.combat.legal_actions()
        if self.phase == "BOSS_RELIC":
            return [
                {
                    "kind": "boss_relic",
                    "name": relic["name"],
                    "relic_id": relic["relic_id"],
                    "choice_index": index,
                }
                for index, relic in enumerate(self.boss_relic_options)
            ] + [{"kind": "skip", "name": "SKIP", "choice_index": len(self.boss_relic_options)}]
        if self.phase in {"GAME_OVER", "COMPLETE", "VICTORY"}:
            return []
        return [{"kind": "not_implemented", "name": "NOT_IMPLEMENTED"}]

    def step(self, _action: dict[str, Any]) -> None:
        action = dict(_action)
        if self.phase == "COMBAT":
            self._step_combat(action)
            self._sync_state()
            return
        if self.phase == "NEOW":
            self._step_neow(action)
            self._sync_state()
            return
        if self.phase == "CARD_REWARD":
            if self._combat_pending_discovery_reward():
                self._step_combat(action)
                self._sync_state()
                return
            self._step_card_reward(action)
            self._sync_state()
            return
        if self.phase == "EVENT":
            self._step_event(action)
            self._sync_state()
            return
        if self.phase == "MAP":
            self._step_map(action)
            self._sync_state()
            return
        if self.phase == "SHOP":
            self._step_shop(action)
            self._sync_state()
            return
        if self.phase == "CAMPFIRE":
            self._step_campfire(action)
            self._sync_state()
            return
        if self.phase == "TREASURE":
            self._step_treasure(action)
            self._sync_state()
            return
        if self.phase == "CARD_SELECT":
            if self._combat_pending_select() is not None:
                self._step_combat(action)
                self._sync_state()
                return
            self._step_card_select(action)
            self._sync_state()
            return
        if self.phase == "BOSS_RELIC":
            self._step_boss_relic(action)
            self._sync_state()
            return
        raise NotImplementedError(
            f"native_sim_v3 phase {self.phase!r} is not implemented yet. "
            "This is now a residual v3 coverage gap rather than an expected default path."
        )

    def _sync_state(self) -> None:
        if isinstance(getattr(self, "_teacher_fast_terminal_state", None), dict):
            return
        if self.combat is not None:
            combat_engine = getattr(self.combat, "engine", None)
            combat_outcome = str(getattr(combat_engine, "outcome", "") or "")
            pending_select = self._combat_pending_select()
            if combat_outcome in {"VICTORY", "DEFEAT"} and self.phase == "COMBAT":
                self._step_combat({"kind": "end", "name": "END_TURN", "action_index": 0})
            if combat_outcome == "UNDECIDED" and pending_select is not None:
                if str((pending_select or {}).get("mode") or "").upper() in {"DISCOVERY", "NILRYS_CODEX"}:
                    self.phase = "CARD_REWARD"
                    self.current_room_type = "CARD_REWARD"
                else:
                    self.phase = "CARD_SELECT"
                    self.current_room_type = "CARD_SELECT"
            elif combat_outcome == "UNDECIDED" and self.phase in {"CARD_SELECT", "CARD_REWARD"}:
                self.phase = "COMBAT"
                self.current_room_type = getattr(self.combat, "room_type", "MonsterRoom")
            if (
                bool(getattr(self, "_teacher_fast_combat_sync", False))
                and combat_outcome == "UNDECIDED"
                and self.phase in {"COMBAT", "CARD_SELECT", "CARD_REWARD"}
            ):
                return
        self._refresh_deck_scaled_relics()
        self._core_state.act = self.act
        self._core_state.dungeon_id = self.dungeon_id
        self._core_state.floor = self.floor
        self._core_state.phase = self.phase
        self._core_state.player = self.player
        self._core_state.deck = list(self.deck)
        self._core_state.relics = list(self.relics)
        self._core_state.potions = list(self.potions)
        self._core_state.gold = self.gold
        self._core_state.has_ruby_key = self.has_ruby_key
        self._core_state.has_emerald_key = self.has_emerald_key
        self._core_state.has_sapphire_key = self.has_sapphire_key
        self._core_state.common_card_pool = list(self.card_pools.get("CLASS_COMMON", []))
        self._core_state.uncommon_card_pool = list(self.card_pools.get("CLASS_UNCOMMON", []))
        self._core_state.rare_card_pool = list(self.card_pools.get("CLASS_RARE", []))
        self._core_state.colorless_card_pool = [
            *list(self.card_pools.get("COLORLESS_UNCOMMON", [])),
            *list(self.card_pools.get("COLORLESS_RARE", [])),
        ]
        self._core_state.curse_card_pool = list(self.card_pools.get("CURSE", []))
        self._core_state.common_relic_pool = list(self.relic_pools.get("COMMON", []))
        self._core_state.uncommon_relic_pool = list(self.relic_pools.get("UNCOMMON", []))
        self._core_state.rare_relic_pool = list(self.relic_pools.get("RARE", []))
        self._core_state.shop_relic_pool = list(self.relic_pools.get("SHOP", []))
        self._core_state.src_common_card_pool = list(self.source_card_pools.get("SRC_COMMON", []))
        self._core_state.src_uncommon_card_pool = list(self.source_card_pools.get("SRC_UNCOMMON", []))
        self._core_state.src_rare_card_pool = list(self.source_card_pools.get("SRC_RARE", []))
        self._core_state.src_colorless_card_pool = list(self.source_card_pools.get("SRC_COLORLESS", []))
        self._core_state.src_curse_card_pool = list(self.source_card_pools.get("SRC_CURSE", []))
        self._core_state.act_boss = self.act_boss
        self._core_state.boss_relic_pool = list(self.boss_relic_pool)
        self._core_state.event_id = self.current_event.event_id if self.current_event is not None and self.phase == "EVENT" else None
        self._core_state.combat = self.combat.state if self.combat is not None and self.phase == "COMBAT" else None

    def _has_relic(self, relic_id: str) -> bool:
        return any(str(relic.get("relic_id") or relic.get("id")) == relic_id for relic in self.relics)

    def _winged_greaves(self) -> dict[str, Any] | None:
        return next((relic for relic in self.relics if str(relic.get("relic_id") or relic.get("id")) == "WingedGreaves"), None)

    def _gain_gold(self, amount: int) -> int:
        delta = int(amount)
        if delta <= 0:
            if delta < 0:
                self.gold = max(0, self.gold + delta)
            return 0
        if self._has_relic("Ectoplasm"):
            return 0
        self.gold += delta
        if self._has_relic("Bloody Idol"):
            self._heal_player(5)
        return delta

    def _spend_gold(self, amount: int) -> int:
        spend = max(0, int(amount))
        if spend <= 0:
            return 0
        actual = min(int(self.gold), spend)
        self.gold = max(0, int(self.gold) - actual)
        if actual > 0 and self._has_relic("MawBank"):
            for relic in self.relics:
                relic_id = str(relic.get("relic_id") or relic.get("id") or "")
                if relic_id == "MawBank" and not bool(relic.get("used_up")):
                    relic["counter"] = -2
                    relic["used_up"] = True
                    break
        return actual

    def _apply_gold_result(self, result_gold: int) -> None:
        target = int(result_gold)
        current = int(self.gold)
        if target <= current:
            self._spend_gold(current - target)
            return
        self._gain_gold(target - current)

    def _handle_room_entry_relics(self, room_type: str) -> None:
        room_name = str(room_type or "")
        if room_name in {"", "None", "Map", "NeowRoom"}:
            return
        if self._has_relic("MawBank"):
            for relic in self.relics:
                relic_id = str(relic.get("relic_id") or relic.get("id") or "")
                if relic_id == "MawBank" and not bool(relic.get("used_up")):
                    self._gain_gold(12)
                    break
        if room_name == "EventRoom" and self._has_relic("SsserpentHead"):
            self._gain_gold(50)
        if room_name == "ShopRoom" and self._has_relic("MealTicket"):
            self._heal_player(15)
        if room_name == "RestRoom":
            if self._has_relic("Eternal Feather"):
                self._heal_player((len(self.deck) // 5) * 3)
            for relic in self.relics:
                relic_id = str(relic.get("relic_id") or relic.get("id") or "")
                if relic_id == "Ancient Tea Set":
                    relic["counter"] = -2
                    break

    def _current_node_has_emerald_key(self) -> bool:
        if self.current_map_node is None:
            return False
        x, y = self.current_map_node
        if y < 0 or y >= len(self.map):
            return False
        return any(node.x == x and bool(node.has_emerald_key) for node in self.map[y])

    def _refresh_deck_scaled_relics(self) -> None:
        curse_count = sum(1 for card in self.deck if str(card.get("type") or "") == "CURSE")
        for relic in self.relics:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id == "Du-Vu Doll":
                relic["counter"] = curse_count

    def _upgrade_random_deck_cards_by_type(self, card_type: str, count: int) -> list[int]:
        upgradable_indexes = [
            index
            for index, card in enumerate(self.deck)
            if str(card.get("type") or "") == card_type and can_upgrade_card(card)
        ]
        if not upgradable_indexes:
            return []
        java_shuffle_in_place(upgradable_indexes, int(self.randoms.stream("misc").random_long()))
        chosen = upgradable_indexes[: max(0, int(count))]
        for target_index in chosen:
            old = self.deck[target_index]
            self.deck[target_index] = make_card(
                str(old["card_id"]),
                upgrades=int(old.get("upgrades") or 0) + 1,
                uuid=str(old.get("uuid") or f"relic-upgrade-{target_index}"),
            )
        return chosen

    def _step_neow(self, action: dict[str, Any]) -> None:
        if self.neow_pending_continue:
            self.neow_pending_continue = False
            self.phase = "MAP"
            self.current_room_type = "None"
            return
        bonus = NeowRewardType(str(action.get("bonus")))
        drawback = str(action.get("drawback") or "NONE")
        hp_bonus = int(self.player.max_hp * 0.1)
        if drawback == "NO_GOLD":
            self.gold = 0
        elif drawback == "TEN_PERCENT_HP_LOSS":
            self.player.max_hp = max(1, self.player.max_hp - hp_bonus)
            self.player.current_hp = min(self.player.current_hp, self.player.max_hp)
        elif drawback == "PERCENT_DAMAGE":
            hp_loss = max(0, (self.player.current_hp // 10) * 3)
            self.player.current_hp = max(1, self.player.current_hp - hp_loss)
        elif drawback == "CURSE":
            self._obtain_card(draw_neow_curse(self.randoms, runtime_pools=self.card_pools))
        if bonus == NeowRewardType.ONE_RANDOM_RARE_CARD:
            self._obtain_card(draw_neow_rare_card(self.randoms, runtime_pools=self.card_pools))
            self.neow_pending_continue = True
        elif bonus == NeowRewardType.THREE_CARDS:
            self.reward_cards = draw_neow_cards(self.randoms, rare_only=False, colorless=False, runtime_pools=self.card_pools)
            self.reward_card_groups = []
            self.reward_card_screen_open = True
            self.phase = "CARD_REWARD"
            self.neow_reward_context = True
        elif bonus == NeowRewardType.RANDOM_COLORLESS:
            self.reward_cards = draw_neow_cards(self.randoms, rare_only=False, colorless=True, runtime_pools=self.card_pools)
            self.reward_card_groups = []
            self.reward_card_screen_open = True
            self.phase = "CARD_REWARD"
            self.neow_reward_context = True
        elif bonus == NeowRewardType.RANDOM_COLORLESS_2:
            self.reward_cards = draw_neow_cards(self.randoms, rare_only=True, colorless=True, runtime_pools=self.card_pools)
            self.reward_card_groups = []
            self.reward_card_screen_open = True
            self.phase = "CARD_REWARD"
            self.neow_reward_context = True
        elif bonus == NeowRewardType.THREE_RARE_CARDS:
            self.reward_cards = draw_neow_cards(self.randoms, rare_only=True, colorless=False, runtime_pools=self.card_pools)
            self.reward_card_groups = []
            self.reward_card_screen_open = True
            self.phase = "CARD_REWARD"
            self.neow_reward_context = True
        elif bonus == NeowRewardType.REMOVE_CARD:
            self._open_card_select(mode="remove", return_phase="NEOW", source="neow", remaining_picks=1)
        elif bonus == NeowRewardType.UPGRADE_CARD:
            self._open_card_select(mode="upgrade", return_phase="NEOW", source="neow", remaining_picks=1)
        elif bonus == NeowRewardType.TRANSFORM_CARD:
            self._open_card_select(mode="transform", return_phase="NEOW", source="neow", remaining_picks=1)
        elif bonus == NeowRewardType.REMOVE_TWO:
            self._open_card_select(mode="remove", return_phase="NEOW", source="neow", remaining_picks=2)
        elif bonus == NeowRewardType.TRANSFORM_TWO_CARDS:
            self._open_card_select(mode="transform", return_phase="NEOW", source="neow", remaining_picks=2)
        elif bonus == NeowRewardType.THREE_SMALL_POTIONS:
            self.reward_gold = None
            self.reward_relics = []
            self.reward_cards = []
            self.reward_card_groups = []
            self.reward_emerald_key = False
            self.reward_card_screen_open = False
            self.reward_potions = [
                draw_random_potion(self.randoms, player_class=self.player_class)
                for _ in range(3)
            ]
            # The real Neow potion reward path still constructs a hidden card
            # RewardItem when opening the reward screen. It is not offered to
            # the player, but it consumes cardRng and advances cardBlizz.
            _, self.card_blizz_randomizer = generate_card_reward_with_state(
                self.randoms,
                card_blizz_randomizer=self.card_blizz_randomizer,
                owned_relic_ids=self._current_relic_ids(),
                runtime_card_pools=self.card_pools,
            )
            self.phase = "CARD_REWARD"
            self.neow_reward_context = True
        elif bonus == NeowRewardType.RANDOM_COMMON_RELIC:
            self._obtain_relic(self._pop_relic_from_pool("COMMON"), source="neow")
            if self.phase == "NEOW":
                self.neow_pending_continue = True
        elif bonus == NeowRewardType.ONE_RARE_RELIC:
            self._obtain_relic(self._pop_relic_from_pool("RARE"), source="neow")
            if self.phase == "NEOW":
                self.neow_pending_continue = True
        elif bonus == NeowRewardType.BOSS_RELIC:
            chosen_relic = self._pop_relic_from_pool("BOSS")
            if self.relics:
                self.relics.pop(0)
            self.relics = apply_boss_relic_choice(
                self.relics,
                chosen_relic,
            )
            self._apply_obtained_relic_side_effects(chosen_relic, source="neow_boss_relic")
            if self.phase == "NEOW":
                self.neow_pending_continue = True
        elif bonus == NeowRewardType.THREE_ENEMY_KILL:
            self._obtain_relic(make_relic("NeowsBlessing"), source="neow")
            if self.phase == "NEOW":
                self.neow_pending_continue = True
        elif bonus == NeowRewardType.TEN_PERCENT_HP_BONUS:
            self.player.max_hp += hp_bonus
            self.player.current_hp += hp_bonus
            self.neow_pending_continue = True
        elif bonus == NeowRewardType.TWENTY_PERCENT_HP_BONUS:
            self.player.max_hp += hp_bonus * 2
            self.player.current_hp += hp_bonus * 2
            self.neow_pending_continue = True
        elif bonus == NeowRewardType.HUNDRED_GOLD:
            self._gain_gold(100)
            self.neow_pending_continue = True
        elif bonus == NeowRewardType.TWO_FIFTY_GOLD:
            self._gain_gold(250)
            self.neow_pending_continue = True
        else:
            raise NotImplementedError(
                f"native_sim_v3 Neow reward {bonus.value!r} is not implemented yet."
            )

    def _step_card_reward(self, action: dict[str, Any]) -> None:
        if action.get("kind") == "raw" and str(action.get("label") or action.get("name") or "").upper() == "CARD" and self.reward_cards:
            self._open_reward_card_group(int(action.get("reward_index") or 0))
            return
        if self.pending_cursed_key_chest_curse is not None and action.get("kind") != "raw":
            self._apply_pending_cursed_key_chest_curse()
        if action.get("kind") == "proceed":
            self.reward_cards = []
            self.reward_card_groups = []
            self.reward_card_screen_open = False
            self.reward_card_reward_declined = False
            self.reward_gold = None
            self.reward_stolen_gold = None
            self.reward_potions = []
            self.reward_potion_replacement_target_id = None
            self.reward_relics = []
            self.reward_emerald_key = False
            self.reward_sapphire_key_relic_id = None
            self.reward_order = []
            self.neow_reward_context = False
            self.neow_pending_continue = False
            if self.current_treasure is not None and self.current_treasure.opened:
                self.current_treasure = None
            if self.boss_reward_pending_boss_relic:
                self._open_boss_chest_room()
                return
            if self.reward_return_phase is not None:
                self._finish_reward_return()
                return
            self.phase = "MAP"
            self.current_room_type = "Map"
            return
        if action.get("kind") == "skip":
            if self.reward_cards:
                if self.reward_card_screen_open and not self._reward_cards_should_open_directly():
                    self.reward_card_screen_open = False
                    self.reward_card_reward_declined = True
                    self.phase = "CARD_REWARD"
                    return
                self.reward_cards = []
                self._promote_next_reward_card_group()
                self.reward_card_screen_open = self._reward_cards_should_open_directly()
                self.reward_card_reward_declined = False
                if self.reward_cards or self.reward_gold is not None or self.reward_stolen_gold is not None or self.reward_potions or self.reward_relics or self.reward_emerald_key or self.reward_sapphire_key_relic_id is not None:
                    self.phase = "CARD_REWARD"
                    return
            self.reward_cards = []
            self._apply_pending_cursed_key_chest_curse()
            self.reward_card_groups = []
            self.reward_card_screen_open = False
            self.reward_card_reward_declined = False
            self.reward_gold = None
            self.reward_stolen_gold = None
            self.reward_potions = []
            self.reward_potion_replacement_target_id = None
            self.reward_relics = []
            self.reward_emerald_key = False
            self.reward_sapphire_key_relic_id = None
            if self.current_treasure is not None and self.current_treasure.opened:
                self.current_treasure = None
            if self.reward_return_phase is not None:
                self._finish_reward_return()
                return
            if self._finish_empty_boss_combat_reward():
                return
            if self.neow_reward_context:
                self.phase = "NEOW"
                self.neow_reward_context = False
                self.neow_pending_continue = True
                self.current_room_type = "NeowRoom"
                self.reward_card_screen_open = False
            else:
                self.phase = "MAP"
                self.current_room_type = "Map"
                self.reward_card_screen_open = False
            return
        if action.get("kind") == "reward_gold":
            reward_type = str(action.get("reward_type") or action.get("name") or "GOLD")
            if reward_type == "STOLEN_GOLD" and self.reward_stolen_gold is not None:
                self._gain_gold(int(self.reward_stolen_gold))
                self.reward_stolen_gold = None
                return
            if reward_type != "STOLEN_GOLD" and self.reward_gold is not None:
                self._gain_gold(int(self.reward_gold))
                self.reward_gold = None
                return
        if action.get("kind") == "reward_potion":
            potion_id = action.get("potion_id")
            for index, potion in enumerate(self.reward_potions):
                if potion["potion_id"] == potion_id:
                    relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
                    if "Sozu" in relic_ids:
                        del self.reward_potions[index]
                        return
                    if not _add_potion_to_first_slot(self.potions, dict(potion), self.max_potion_slots):
                        return
                    del self.reward_potions[index]
                    if self.reward_potion_replacement_target_id == potion_id:
                        self.reward_potion_replacement_target_id = None
                    return
        if action.get("kind") == "discard_potion":
            potion_index = action.get("potion_index")
            if potion_index is not None:
                _remove_potion_at_slot(self.potions, int(potion_index), self.max_potion_slots)
                target_id = action.get("replace_target_potion_id")
                if target_id is not None:
                    self.reward_potion_replacement_target_id = str(target_id)
            return
        if action.get("kind") == "reward_relic":
            relic_id = action.get("relic_id")
            for index, relic in enumerate(self.reward_relics):
                if relic["relic_id"] == relic_id:
                    self._obtain_relic(dict(relic), source="reward_relic")
                    del self.reward_relics[index]
                    if self.reward_sapphire_key_relic_id == relic_id:
                        self.reward_sapphire_key_relic_id = None
                    return
        if action.get("kind") == "reward_key" and action.get("key") == "sapphire" and self.reward_sapphire_key_relic_id is not None:
            linked_relic_id = self.reward_sapphire_key_relic_id
            self.reward_relics = [
                dict(relic)
                for relic in self.reward_relics
                if str(relic.get("relic_id")) != linked_relic_id
            ]
            self.has_sapphire_key = True
            self.reward_sapphire_key_relic_id = None
            return
        if action.get("kind") == "reward_key" and self.reward_emerald_key:
            self.has_emerald_key = True
            self.reward_emerald_key = False
            return
        if action.get("kind") == "singing_bowl" and self.reward_cards:
            self._increase_max_hp(2)
            self.reward_cards = []
            self._promote_next_reward_card_group()
            self.reward_card_screen_open = self._reward_cards_should_open_directly()
            self.reward_card_reward_declined = False
            if self.reward_cards or self.reward_gold is not None or self.reward_stolen_gold is not None or self.reward_potions or self.reward_relics or self.reward_emerald_key or self.reward_sapphire_key_relic_id is not None:
                self.phase = "CARD_REWARD"
                return
            if self.neow_reward_context:
                self.phase = "NEOW"
                self.neow_reward_context = False
                self.neow_pending_continue = True
                self.current_room_type = "NeowRoom"
                self.reward_card_screen_open = False
                return
            if self.event_reward_context:
                self.event_reward_context = False
                if self.current_event is not None:
                    self.phase = "EVENT"
                    self.current_room_type = "EventRoom"
                elif self.reward_return_phase is not None:
                    self._finish_reward_return()
                else:
                    self.phase = "MAP"
                    self.current_room_type = "Map"
                self.reward_card_screen_open = False
                return
            if self._finish_empty_boss_combat_reward():
                return
            if self.reward_return_phase is not None:
                self._finish_reward_return()
                return
            self.phase = "MAP"
            self.current_room_type = "Map"
            return
        if action.get("kind") == "card_reward":
            self._obtain_card(dict(action["card"]))
        self.reward_cards = []
        self._promote_next_reward_card_group()
        self.reward_card_screen_open = self._reward_cards_should_open_directly()
        self.reward_card_reward_declined = False
        if self.reward_cards or self.reward_gold is not None or self.reward_stolen_gold is not None or self.reward_potions or self.reward_relics or self.reward_emerald_key or self.reward_sapphire_key_relic_id is not None:
            return
        if self.neow_reward_context:
            if self.current_treasure is not None and self.current_treasure.opened:
                self.current_treasure = None
            self.phase = "NEOW"
            self.neow_reward_context = False
            self.neow_pending_continue = True
            self.current_room_type = "NeowRoom"
            self.reward_card_screen_open = False
            return
        if self.event_reward_context:
            self.event_reward_context = False
            if self.current_event is not None:
                self.phase = "EVENT"
                self.current_room_type = "EventRoom"
            elif self.reward_return_phase is not None:
                self._finish_reward_return()
            else:
                self.phase = "MAP"
                self.current_room_type = "Map"
            self.reward_card_screen_open = False
            return
        if self.reward_return_phase is not None:
            self.phase = "CARD_REWARD"
            self.reward_card_screen_open = False
            return
        # Combat reward screens remain open after taking the last reward until
        # the player explicitly proceeds.
        self.phase = "CARD_REWARD"

    def _apply_pending_cursed_key_chest_curse(self) -> None:
        if self.pending_cursed_key_chest_curse is None:
            return
        curse = dict(self.pending_cursed_key_chest_curse)
        self.pending_cursed_key_chest_curse = None
        self._obtain_card(curse)

    def _step_map(self, action: dict[str, Any]) -> None:
        node_id = str(action["node_id"])
        symbol = str(action.get("symbol") or "")
        if symbol != "BOSS":
            _, row, x = node_id.split("-")
            y = int(row.removeprefix("r"))
            x_value = int(x.removeprefix("x"))
            if self.current_map_node is not None:
                current_x, current_y = self.current_map_node
                if 0 <= current_y < len(self.map) and 0 <= current_x < len(self.map[current_y]):
                    current_edges = self.map[current_y][current_x].edges
                    normal_connection = any(
                        edge.dst_x == x_value and edge.dst_y == y
                        for edge in current_edges
                    )
                    winged_connection = any(edge.dst_y == y for edge in current_edges)
                    winged_relic = self._winged_greaves()
                    if (
                        not normal_connection
                        and winged_connection
                        and winged_relic is not None
                        and int(winged_relic.get("counter") or 0) > 0
                    ):
                        winged_relic["counter"] = int(winged_relic.get("counter") or 0) - 1
                        if int(winged_relic.get("counter") or 0) <= 0:
                            winged_relic["counter"] = -2
            self.current_map_node = (x_value, y)
            self.first_room_chosen = True
            action_floor = int(action["floor"])
            self.floor = self._act_floor_offset() + action_floor if action_floor <= 16 else action_floor
            self.randoms.reset_floor_streams(self.floor)
        else:
            self.first_room_chosen = True
            action_floor = int(action["floor"])
            self.floor = self._act_floor_offset() + action_floor if action_floor <= 16 else action_floor
            self.randoms.reset_floor_streams(self.floor)
        self._apply_pending_show_card_obtain_effects()
        if symbol in {"M", "E", "E_GREEN"}:
            if symbol in {"E", "E_GREEN"}:
                if not self.elite_monster_list:
                    raise NotImplementedError("native_sim_v3 has no more generated elite encounters for this act yet.")
                encounter_name = self.elite_monster_list.pop(0)
            else:
                if not self.monster_list:
                    raise NotImplementedError("native_sim_v3 has no more generated monster encounters for this act yet.")
                encounter_name = self.monster_list.pop(0)
            room_type = "MonsterRoomElite" if symbol in {"E", "E_GREEN"} else "MonsterRoom"
            self.current_room_type = room_type
            self._handle_room_entry_relics(room_type)
            self.combat = NativeCombatEnv(
                seed=self.seed,
                ascension_level=self.ascension_level,
                character=self.player_class,
                act=self.act,
                floor=self.floor,
                room_type=room_type,
                player=self.player,
                master_deck=self.deck,
                relics=self.relics,
                potions=self.potions,
                gold=self.gold,
                act_boss=self.act_boss,
                encounter_name=encounter_name,
                source_card_pools=self.source_card_pools,
                randoms=self.randoms,
                has_emerald_key=symbol == "E_GREEN",
            )
            self.phase = "COMBAT"
        elif symbol == "?":
            room_result, next_chances = roll_question_room_result(
                self.randoms,
                floor=self.floor,
                current_room_type=self.current_room_type,
                relics=self.relics,
                elite_chance=self.question_elite_chance,
                monster_chance=self.question_monster_chance,
                shop_chance=self.question_shop_chance,
                treasure_chance=self.question_treasure_chance,
            )
            self.question_elite_chance = float(next_chances["elite"])
            self.question_monster_chance = float(next_chances["monster"])
            self.question_shop_chance = float(next_chances["shop"])
            self.question_treasure_chance = float(next_chances["treasure"])
            if room_result == "EVENT":
                self.current_room_type = "EventRoom"
                self._handle_room_entry_relics(self.current_room_type)
                self.current_event = generate_event_for_act(
                    self.randoms,
                    ascension_level=self.ascension_level,
                    act=self.act,
                    floor=self.floor,
                    gold=self.gold,
                    relics=self.relics,
                    deck=self.deck,
                    potions=self.potions,
                    current_hp=self.player.current_hp,
                    max_hp=self.player.max_hp,
                    current_node_y=self.current_map_node[1] if self.current_map_node is not None else None,
                    map_height=len(self.map),
                    event_list=self.event_list,
                    shrine_list=self.shrine_list,
                    special_one_time_event_list=self.special_one_time_event_list,
                    # Secret Portal is gated on CardCrawlGame.playtime >= 800s.
                    # Native runs are deterministic and do not advance a wall clock,
                    # so approximate real bridge playtime by progressed floor.
                    playtime_seconds=float(self.floor) * 20.0,
                    relic_drawer=self._pop_relic_from_pool,
                    screenless_relic_drawer=self._pop_screenless_relic_from_pool,
                    runtime_card_pools=self.card_pools,
                    player_class=self.player_class,
                    note_for_yourself_card_id=self.note_for_yourself_card_id,
                    note_for_yourself_upgrades=self.note_for_yourself_upgrades,
                )
                self.phase = "EVENT"
            elif room_result == "MONSTER":
                if not self.monster_list:
                    raise NotImplementedError("native_sim_v3 has no more generated monster encounters for this act yet.")
                encounter_name = self.monster_list.pop(0)
                self.current_room_type = "MonsterRoom"
                self._handle_room_entry_relics(self.current_room_type)
                self.combat = NativeCombatEnv(
                    seed=self.seed,
                    ascension_level=self.ascension_level,
                    character=self.player_class,
                    act=self.act,
                    floor=self.floor,
                    room_type="MonsterRoom",
                    player=self.player,
                    master_deck=self.deck,
                    relics=self.relics,
                    potions=self.potions,
                    gold=self.gold,
                    act_boss=self.act_boss,
                    encounter_name=encounter_name,
                    source_card_pools=self.source_card_pools,
                    randoms=self.randoms,
                )
                self.phase = "COMBAT"
            elif room_result == "ELITE":
                if not self.elite_monster_list:
                    raise NotImplementedError("native_sim_v3 has no more generated elite encounters for this act yet.")
                encounter_name = self.elite_monster_list.pop(0)
                self.current_room_type = "MonsterRoomElite"
                self._handle_room_entry_relics(self.current_room_type)
                self.combat = NativeCombatEnv(
                    seed=self.seed,
                    ascension_level=self.ascension_level,
                    character=self.player_class,
                    act=self.act,
                    floor=self.floor,
                    room_type="MonsterRoomElite",
                    player=self.player,
                    master_deck=self.deck,
                    relics=self.relics,
                    potions=self.potions,
                    gold=self.gold,
                    act_boss=self.act_boss,
                    encounter_name=encounter_name,
                    source_card_pools=self.source_card_pools,
                    randoms=self.randoms,
                )
                self.phase = "COMBAT"
            elif room_result == "SHOP":
                self.current_room_type = "ShopRoom"
                self._handle_room_entry_relics(self.current_room_type)
                self.current_shop = generate_shop(
                    self.randoms,
                    act=self.act,
                    dungeon_id=self.dungeon_id,
                    card_blizz_randomizer=self.card_blizz_randomizer,
                    floor_num=self.floor,
                    ascension_level=self.ascension_level,
                    owned_relic_ids={str(relic.get("relic_id") or relic.get("id")) for relic in self.relics},
                    relic_drawer=self._pop_relic_end_from_pool,
                    runtime_card_pools=self.card_pools,
                    player_class=self.player_class,
                    purge_base_cost=self.shop_purge_base_cost,
                )
                self.phase = "SHOP"
            elif room_result == "TREASURE":
                self.current_room_type = "TreasureRoom"
                self._handle_room_entry_relics(self.current_room_type)
                self.current_treasure = generate_treasure(self.randoms, act=self.act, dungeon_id=self.dungeon_id)
                self.phase = "TREASURE"
            else:
                raise NotImplementedError(f"native_sim_v3 question room result {room_result!r} is not implemented.")
        elif symbol == "R":
            self.current_campfire = CampfireState(can_recall=self.final_act_available and not self.has_ruby_key)
            self.phase = "CAMPFIRE"
            self.current_room_type = "RestRoom"
            self._handle_room_entry_relics(self.current_room_type)
        elif symbol == "$":
            self.current_room_type = "ShopRoom"
            self._handle_room_entry_relics(self.current_room_type)
            self.current_shop = generate_shop(
                self.randoms,
                act=self.act,
                dungeon_id=self.dungeon_id,
                card_blizz_randomizer=self.card_blizz_randomizer,
                floor_num=self.floor,
                ascension_level=self.ascension_level,
                owned_relic_ids={str(relic.get("relic_id") or relic.get("id")) for relic in self.relics},
                relic_drawer=self._pop_relic_end_from_pool,
                runtime_card_pools=self.card_pools,
                player_class=self.player_class,
                purge_base_cost=self.shop_purge_base_cost,
            )
            self.phase = "SHOP"
        elif symbol == "T":
            self.current_room_type = "TreasureRoom"
            self._handle_room_entry_relics(self.current_room_type)
            self.current_treasure = generate_treasure(self.randoms, act=self.act, dungeon_id=self.dungeon_id)
            self.phase = "TREASURE"
        elif symbol == "BOSS":
            if self.boss_list:
                encounter_name = self.boss_list.pop(0)
                self.act_boss = self.boss_list[0] if self.boss_list else encounter_name
            elif self.act_boss is not None:
                encounter_name = str(self.act_boss)
            else:
                raise NotImplementedError("native_sim_v3 has no generated Act boss for this run yet.")
            self.current_room_type = "MonsterRoomBoss"
            self._handle_room_entry_relics(self.current_room_type)
            self.combat = NativeCombatEnv(
                seed=self.seed,
                ascension_level=self.ascension_level,
                character=self.player_class,
                act=self.act,
                floor=self.floor,
                room_type="MonsterRoomBoss",
                player=self.player,
                master_deck=self.deck,
                relics=self.relics,
                potions=self.potions,
                gold=self.gold,
                act_boss=encounter_name,
                encounter_name=encounter_name,
                source_card_pools=self.source_card_pools,
                randoms=self.randoms,
            )
            self.phase = "COMBAT"
        elif symbol == "VICTORY":
            self.phase = "VICTORY"
            self.current_room_type = "TrueVictoryRoom"
        else:
            self.phase = "MAP"

    def _step_combat(self, action: dict[str, Any]) -> None:
        if self.combat is None:
            raise RuntimeError("native_sim_v3 combat step requested without an active combat env")
        teacher_fast_sync = bool(getattr(self, "_teacher_fast_combat_sync", False))
        if teacher_fast_sync:
            combat_engine = getattr(self.combat, "engine", None)
            if combat_engine is not None:
                try:
                    setattr(combat_engine, "_teacher_fast_step_refresh", True)
                except Exception:
                    pass
        outcome = self.combat.step(action)
        combat_engine = getattr(self.combat, "engine", None)
        if teacher_fast_sync and combat_engine is not None and outcome == "UNDECIDED":
            self.gold = int(getattr(combat_engine, "gold", self.gold))
            bonus_reward_gold = 0
            all_monsters_escaped = False
        else:
            combat_master_deck = (
                getattr(combat_engine, "master_deck", None)
                if teacher_fast_sync and combat_engine is not None
                else getattr(self.combat, "master_deck", None)
            )
            if combat_master_deck is not None:
                self.deck = list(combat_master_deck)
            if combat_engine is not None:
                self.gold = int(getattr(combat_engine, "gold", self.gold))
                self.potions = list(getattr(combat_engine, "potions", self.potions))
                if teacher_fast_sync:
                    bonus_reward_gold = 0
                    all_monsters_escaped = False
                else:
                    bonus_reward_gold = int(getattr(combat_engine, "bonus_reward_gold", 0) or 0)
                    combat_monsters = list(getattr(getattr(combat_engine, "state", None), "monsters", []) or [])
                    all_monsters_escaped = bool(combat_monsters) and all(
                        bool(getattr(monster, "meta", {}).get("escaped", False))
                        for monster in combat_monsters
                    )
            else:
                self.gold = int(getattr(self.combat, "gold", self.gold))
                self.potions = list(getattr(self.combat, "potions", self.potions))
                bonus_reward_gold = 0
                all_monsters_escaped = False
        if teacher_fast_sync and outcome in {"VICTORY", "DEFEAT"}:
            self.player = getattr(combat_engine, "player", self.combat.player) if combat_engine is not None else self.combat.player
            self.current_room_type = getattr(self.combat, "room_type", self.current_room_type)
            if outcome == "VICTORY":
                self._reset_player_post_combat_state()
                self._apply_post_combat_relic_effects()
                if self.combat.encounter_name == "The Heart":
                    self.phase = "VICTORY"
                    self.current_room_type = "TrueVictoryRoom"
                elif self.current_room_type == "MonsterRoomBoss" and self.act >= 3:
                    self.phase = "COMPLETE"
                else:
                    self.phase = "CARD_REWARD"
            else:
                self.phase = "GAME_OVER"
            self.combat = None
            self._teacher_fast_terminal_state = {
                "backend": "v3",
                "implementation_status": "combat_vertical_slice",
                "phase": self.phase,
                "screen": self.phase,
                "screen_type": self.phase,
                "screen_up": False,
                "ascension_level": int(self.ascension_level),
                "act": int(self.act),
                "dungeon_id": self.dungeon_id,
                "floor": int(self.floor),
                "room_type": self.current_room_type,
                "current_hp": int(self.player.current_hp),
                "max_hp": int(self.player.max_hp),
                "gold": int(self.gold),
                "act_boss": self.act_boss,
                "deck": list(self.deck),
                "relics": list(self.relics),
                "potions": list(self.potions),
                "combat_state": {},
                "screen_state": {},
                "choice_available": False,
                "choice_list": [],
                "commands": {},
                "reference_sources": dict(self.reference_sources),
            }
            return
        pending_select = getattr(combat_engine, "pending_card_select", None)
        if outcome == "UNDECIDED" and pending_select is not None:
            if str((pending_select or {}).get("mode") or "").upper() in {"DISCOVERY", "NILRYS_CODEX"}:
                self.phase = "CARD_REWARD"
                self.current_room_type = "CARD_REWARD"
            else:
                self.phase = "CARD_SELECT"
                self.current_room_type = "CARD_SELECT"
            return
        if outcome == "UNDECIDED" and self.phase in {"CARD_SELECT", "CARD_REWARD"}:
            self.phase = "COMBAT"
            self.current_room_type = getattr(self.combat, "room_type", "MonsterRoom")
            return
        if outcome == "DEFEAT":
            self.player = self.combat.player
            self.phase = "GAME_OVER"
            self.current_room_type = getattr(self.combat, "room_type", self.current_room_type)
            return
        if outcome == "VICTORY":
            self.player = self.combat.player
            self._reset_player_post_combat_state()
            self._apply_post_combat_relic_effects()
            if self.combat.encounter_name == "The Heart":
                self.reward_gold = None
                self.reward_stolen_gold = None
                self.reward_potions = []
                self.reward_relics = []
                self.reward_emerald_key = False
                self.reward_sapphire_key_relic_id = None
                self.reward_cards = []
                self.reward_card_groups = []
                self.reward_card_screen_open = False
                self.phase = "VICTORY"
                self.current_room_type = "TrueVictoryRoom"
            elif self.current_room_type == "MonsterRoomBoss" and self.act >= 3:
                self.reward_gold = None
                self.reward_stolen_gold = None
                self.reward_potions = []
                self.reward_relics = []
                self.reward_emerald_key = False
                self.reward_sapphire_key_relic_id = None
                self.reward_cards = []
                self.reward_card_groups = []
                self.reward_card_screen_open = False
                self.boss_reward_pending_boss_relic = False
                self.phase = "COMPLETE"
                self.current_room_type = "MonsterRoomBoss"
            elif self.current_room_type == "MonsterRoomBoss":
                self._setup_boss_combat_reward(bonus_reward_gold=bonus_reward_gold)
            else:
                if self.current_room_type == "EventRoom" and self.pending_event_rewards is not None:
                    payload = dict(self.pending_event_rewards)
                    self.pending_event_rewards = None
                    reopened_event = payload.get("reopen_event")
                    if isinstance(reopened_event, EventState):
                        self.current_event = reopened_event
                        self.phase = "EVENT"
                        self.current_room_type = "EventRoom"
                        self.combat = None
                        return
                    self._setup_event_combat_reward(payload, bonus_reward_gold=bonus_reward_gold)
                    if self.reward_gold is not None or self.reward_stolen_gold is not None or self.reward_potions or self.reward_relics or self.reward_cards or self.reward_card_groups or self.reward_emerald_key:
                        self.phase = "CARD_REWARD"
                    else:
                        self.phase = "MAP"
                        self.current_room_type = "Map"
                else:
                    owned_relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
                    reward_relics: list[dict[str, Any]] = []
                    if self.current_room_type == "MonsterRoomElite":
                        reward_relics = generate_elite_relic_rewards(
                            self.randoms,
                            act=self.act,
                            owned_relic_ids=owned_relic_ids,
                            black_star="Black Star" in owned_relic_ids,
                            relic_drawer=self._pop_relic_from_pool,
                        )
                    emerald_key_reward = bool(
                        self.current_room_type == "MonsterRoomElite"
                        and self.final_act_available
                        and not self.has_emerald_key
                        and self._current_node_has_emerald_key()
                    )
                    suppress_normal_rewards = bool(
                        self.current_room_type == "MonsterRoom"
                        and all_monsters_escaped
                    )
                    rewards = generate_monster_room_rewards(
                        self.randoms,
                        act=self.act,
                        dungeon_id=self.dungeon_id,
                        room_type=self.current_room_type,
                        ascension_level=self.ascension_level,
                        card_blizz_randomizer=self.card_blizz_randomizer,
                        blizzard_potion_mod=self.blizzard_potion_mod,
                        owned_relic_ids=owned_relic_ids,
                        reward_count=int(not suppress_normal_rewards) + len(reward_relics) + int(emerald_key_reward),
                        player_class=self.player_class,
                        prayer_wheel=self.current_room_type == "MonsterRoom" and "Prayer Wheel" in owned_relic_ids,
                        include_gold=not suppress_normal_rewards,
                        potion_chance_override=0 if suppress_normal_rewards else None,
                        runtime_card_pools=self.card_pools,
                    )
                    self.reward_gold = None if suppress_normal_rewards else int(rewards["gold"] or 0) or None
                    self.reward_stolen_gold = bonus_reward_gold or None
                    potion = rewards["potion"]
                    self.reward_potions = [dict(potion)] if potion is not None else []
                    self.reward_relics = [dict(relic) for relic in reward_relics]
                    self.reward_emerald_key = emerald_key_reward
                    self._offer_sapphire_key_for_last_reward_relic()
                    self.blizzard_potion_mod = int(rewards["blizzard_potion_mod"])
                    card_groups = [list(group) for group in list(rewards.get("card_groups") or [])]
                    cards = list(card_groups[0]) if card_groups else list(rewards["cards"])
                    self.card_blizz_randomizer = int(rewards["card_blizz_randomizer"])
                    self.reward_cards = [dict(card) for card in cards]
                    self.reward_card_groups = [
                        [dict(card) for card in group]
                        for group in card_groups[1:]
                    ]
                    self.reward_card_screen_open = False
                    self.phase = "CARD_REWARD"
                    if self.current_room_type not in {"MonsterRoomBoss", "TreasureRoom", "EventRoom"}:
                        self.current_room_type = "MonsterRoom"
            self.combat = None

    def _apply_post_combat_relic_effects(self) -> None:
        relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
        heal_amount = 0
        if "Black Blood" in relic_ids:
            heal_amount = 12
        elif "Burning Blood" in relic_ids:
            heal_amount = 6
        if heal_amount > 0:
            self._heal_player(heal_amount, apply_magic_flower=True)
        if "FaceOfCleric" in relic_ids:
            self._increase_max_hp(1)

    def _step_event(self, action: dict[str, Any]) -> None:
        if self.current_event is None:
            raise RuntimeError("native_sim_v3 EVENT step requested without an active event")
        if self.current_event.event_id == "The Library":
            self.current_event.data["card_blizz_randomizer"] = self.card_blizz_randomizer
        result = resolve_event_choice(
            self.current_event,
            action_index=int(action.get("button_index", action.get("choice_index") or 0) or 0),
            randoms=self.randoms,
            ascension_level=self.ascension_level,
            dungeon_id=self.dungeon_id,
            current_hp=self.player.current_hp,
            max_hp=self.player.max_hp,
            gold=self.gold,
            deck=self.deck,
            relics=self.relics,
            potions=self.potions,
            relic_drawer=self._pop_relic_from_pool,
            screenless_relic_drawer=self._pop_screenless_relic_from_pool,
            runtime_card_pools=self.card_pools,
            player_class=self.player_class,
            final_act_available=self.final_act_available,
            has_ruby_key=self.has_ruby_key,
            has_emerald_key=self.has_emerald_key,
            has_sapphire_key=self.has_sapphire_key,
        )
        self._apply_gold_result(int(result.get("gold", self.gold)))
        self.player.current_hp = int(result.get("hp", self.player.current_hp))
        self.player.max_hp = int(result.get("max_hp", self.player.max_hp))
        if "heal_player" in result:
            self._heal_player(int(result.get("heal_player") or 0))
        if "card_blizz_randomizer" in result:
            self.card_blizz_randomizer = int(result["card_blizz_randomizer"])
        if result.get("upgrade_indexes"):
            for index in list(result.get("upgrade_indexes") or []):
                if 0 <= int(index) < len(self.deck):
                    card = self.deck[int(index)]
                    new_upgrades = int(card.get("upgrades") or 0) + 1
                    self.deck[int(index)] = make_card(str(card["card_id"]), upgrades=new_upgrades, uuid=str(card.get("uuid") or f"upgrade-{index}"))
        potions = list(result.get("potions") or [])
        remove_potion_index = result.get("remove_potion_index")
        if remove_potion_index is not None:
            _remove_potion_at_slot(self.potions, int(remove_potion_index), self.max_potion_slots)
        remove_card_uuid = result.get("remove_card_uuid")
        if remove_card_uuid is not None:
            self._remove_deck_card_by_uuid(str(remove_card_uuid))
        remove_card_uuids = {str(card_uuid) for card_uuid in list(result.get("remove_card_uuids") or [])}
        if remove_card_uuids:
            self._remove_deck_cards_by_uuid(remove_card_uuids)
        if result.get("remove_starter_strikes"):
            self.deck = [
                card for card in self.deck
                if "STARTER_STRIKE" not in set(card.get("tags") or [])
                and str(card.get("card_id") or "") != "Strike_R"
            ]
        remove_relic_id = result.get("remove_relic_id")
        if remove_relic_id is not None:
            self._remove_relic(str(remove_relic_id))
        for card in list(result.get("add_cards") or []):
            self._obtain_card(dict(card))
        for potion in list(result.get("add_potions") or []):
            _add_or_replace_lowest_priority_potion(self.potions, dict(potion), self.max_potion_slots)
        for relic in list(result.get("add_relics") or []):
            self._obtain_relic(dict(relic), source="event")
        if result.get("open_rewards"):
            leave_event_after_rewards = bool(result.get("leave_event_after_rewards"))
            self.reward_gold = None
            self.reward_stolen_gold = None
            self.reward_potions = [dict(potion) for potion in potions]
            self.reward_relics = [dict(relic) for relic in list(result.get("relic_rewards") or [])]
            self.reward_emerald_key = False
            self.reward_cards = []
            self.reward_card_groups = []
            self.reward_card_screen_open = False
            self.reward_return_phase = None if leave_event_after_rewards else "EVENT"
            self.reward_return_room_type = None if leave_event_after_rewards else "EventRoom"
            self.reward_return_neow_continue = False
            if leave_event_after_rewards:
                self.current_event = None
            self.phase = "CARD_REWARD"
            self.current_room_type = "EventRoom"
            return
        if result.get("open_event_card_reward"):
            self.reward_gold = None
            self.reward_stolen_gold = None
            self.reward_potions = []
            self.reward_relics = []
            self.reward_emerald_key = False
            self.reward_cards = [dict(card) for card in list(result.get("reward_cards") or [])]
            self.reward_card_groups = [
                [dict(card) for card in group]
                for group in list(result.get("reward_card_groups") or [])
            ]
            self.reward_card_screen_open = True
            self.event_reward_context = True
            self.phase = "CARD_REWARD"
            self.current_room_type = "EventRoom"
            return
        if result.get("open_card_select"):
            mode = str(result.get("card_select_mode") or "purge")
            candidate_indexes = result.get("candidate_indexes")
            if candidate_indexes is None:
                candidate_indexes = self._card_select_candidates(mode)
            select_cards = result.get("card_select_cards")
            self.current_card_select = {
                "mode": mode,
                "candidate_indexes": list(candidate_indexes or []),
                "cards": [dict(card) for card in list(select_cards or [])] if select_cards is not None else None,
                "return_phase": str(result.get("return_phase") or "MAP"),
                "source": "event",
                "clear_event_on_finish": bool(result.get("clear_event_on_finish", True)),
                "selection_effect": result.get("card_select_effect"),
                "remaining_picks": int(result.get("remaining_picks") or 1),
                "total_picks": int(result.get("remaining_picks") or 1),
                "selected_target_indexes": [],
                "selected_cards": [],
                "visible_for_purge": bool(result.get("visible_for_purge", mode in {"purge", "remove"})),
                "requires_confirm": bool(result.get("requires_confirm", True)),
            }
            self.phase = "CARD_SELECT"
            return
        if result.get("open_combat"):
            encounter_name = str(result.get("encounter_name") or "")
            if not encounter_name:
                raise RuntimeError("native_sim_v3 event combat requested without an encounter name")
            self.pending_event_rewards = dict(result.get("event_rewards") or {})
            self.current_event = None
            self.combat = NativeCombatEnv(
                seed=self.seed,
                ascension_level=self.ascension_level,
                character=self.player_class,
                act=self.act,
                floor=self.floor,
                room_type="EventRoom",
                player=self.player,
                master_deck=self.deck,
                relics=self.relics,
                potions=self.potions,
                gold=self.gold,
                act_boss=self.act_boss,
                encounter_name=encounter_name,
                source_card_pools=self.source_card_pools,
                randoms=self.randoms,
                prebuilt_monsters=result.get("prebuilt_monsters"),
                elite_trigger=bool(result.get("elite_trigger")),
            )
            self.phase = "COMBAT"
            self.current_room_type = "EventRoom"
            return
        if result.get("warp_to_boss"):
            if self.boss_list:
                encounter_name = self.boss_list.pop(0)
                self.act_boss = self.boss_list[0] if self.boss_list else encounter_name
            elif self.act_boss is not None:
                encounter_name = str(self.act_boss)
            else:
                raise RuntimeError("native_sim_v3 event requested boss warp without an available Act boss")
            self.floor = self._act_floor_offset() + 16
            self.current_event = None
            self.combat = NativeCombatEnv(
                seed=self.seed,
                ascension_level=self.ascension_level,
                character=self.player_class,
                act=self.act,
                floor=self.floor,
                room_type="MonsterRoomBoss",
                player=self.player,
                master_deck=self.deck,
                relics=self.relics,
                potions=self.potions,
                gold=self.gold,
                act_boss=encounter_name,
                encounter_name=encounter_name,
                source_card_pools=self.source_card_pools,
                randoms=self.randoms,
            )
            self.phase = "COMBAT"
            self.current_room_type = "MonsterRoomBoss"
            return
        if result.get("advance_to_final_act"):
            self.current_event = None
            self._advance_to_next_act()
            return
        if result.get("victory"):
            self.current_event = None
            self.phase = "VICTORY"
            self.current_room_type = "VictoryRoom"
            return
        if result.get("game_over"):
            self.current_event = None
            self.phase = "GAME_OVER"
            self.current_room_type = "EventRoom"
            return
        if result.get("stay_event"):
            self.phase = "EVENT"
            self.current_room_type = "EventRoom"
            return
        if result.get("leave_event"):
            self.current_event = None
            self.phase = "MAP"
            self.current_room_type = "Map"

    def _step_shop(self, action: dict[str, Any]) -> None:
        if self.current_shop is None:
            raise RuntimeError("native_sim_v3 SHOP step requested without generated shop state")
        relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
        item_kind = str(action.get("item_kind") or "")
        if item_kind == "leave":
            self.current_shop = None
            self.phase = "MAP"
            # In the real game, returning to the map leaves currRoom as ShopRoom.
            # EventHelper uses that hidden room to suppress shop rolls on the next ?.
            self.current_room_type = "ShopRoom"
            return
        if item_kind == "card":
            index = int(action.get("shop_index") or 0)
            card = self.current_shop.cards[index]
            if self.gold >= int(card["price"]):
                self._spend_gold(int(card["price"]))
                self._obtain_card(dict(card))
                if "The Courier" in relic_ids:
                    self.current_shop.cards[index] = self._generate_shop_replacement_card(card)
                else:
                    del self.current_shop.cards[index]
            return
        if item_kind == "relic":
            index = int(action.get("shop_index") or 0)
            relic = self.current_shop.relics[index]
            if is_banned_relic_id(relic.get("relic_id") or relic.get("id")):
                return
            if self.gold >= int(relic["price"]):
                self._spend_gold(int(relic["price"]))
                obtained_relic = {k: v for k, v in relic.items() if k not in {"price", "base_price"}}
                self._obtain_relic(obtained_relic, source="shop")
                purchased_relic_id = str(relic.get("relic_id") or "")
                if purchased_relic_id == "Membership Card":
                    self._apply_shop_discount(0.5, affect_purge=True)
                if purchased_relic_id == "Smiling Mask":
                    self.current_shop.purge_cost = 50
                if "The Courier" in relic_ids or purchased_relic_id == "The Courier":
                    self.current_shop.relics[index] = self._generate_shop_replacement_relic(
                        exclude={
                            "Old Coin",
                            "Smiling Mask",
                            "Maw Bank",
                            "The Courier",
                            *{str(current.get("relic_id") or current.get("id")) for current in self.relics},
                            *{
                                str(current.get("relic_id") or current.get("id"))
                                for current in self.current_shop.relics
                                if current is not relic
                            },
                        }
                    )
                else:
                    del self.current_shop.relics[index]
            return
        if item_kind == "potion":
            index = int(action.get("shop_index") or 0)
            potion = self.current_shop.potions[index]
            if "Sozu" in relic_ids:
                return
            if self.gold >= int(potion["price"]) and _has_open_potion_slot(self.potions, self.max_potion_slots):
                self._spend_gold(int(potion["price"]))
                _add_potion_to_first_slot(
                    self.potions,
                    {k: v for k, v in potion.items() if k not in {"price", "shop_index"}},
                    self.max_potion_slots,
                )
                if "The Courier" in relic_ids:
                    self.current_shop.potions[index] = self._generate_shop_replacement_potion(index)
                else:
                    del self.current_shop.potions[index]
            return
        if item_kind == "purge":
            target_index = action.get("target_index")
            self._last_shop_purge_removed = False
            self._last_shop_purge_removed_card = None
            self._last_shop_purge_target_index = int(target_index) if target_index is not None else None
            if target_index is not None and self.gold >= int(self.current_shop.purge_cost) and 0 <= int(target_index) < len(self.deck):
                self._spend_gold(int(self.current_shop.purge_cost))
                self._last_shop_purge_removed_card = self._remove_deck_card_at(int(target_index))
                self._last_shop_purge_removed = True
                self.current_shop.purge_available = False
                next_cost = int(getattr(self.current_shop, "purge_base_cost", 75)) + 25
                self.shop_purge_base_cost = next_cost
                self.current_shop.purge_base_cost = next_cost
                if "Smiling Mask" in relic_ids:
                    self.current_shop.purge_cost = 50
                elif "The Courier" in relic_ids and "Membership Card" in relic_ids:
                    self.current_shop.purge_cost = _java_round_positive(next_cost * 0.8 * 0.5)
                elif "The Courier" in relic_ids:
                    self.current_shop.purge_cost = _java_round_positive(next_cost * 0.8)
                elif "Membership Card" in relic_ids:
                    self.current_shop.purge_cost = _java_round_positive(next_cost * 0.5)
                else:
                    self.current_shop.purge_cost = next_cost
            return
        raise NotImplementedError(f"native_sim_v3 SHOP action kind {item_kind!r} is not implemented yet.")

    def _step_campfire(self, action: dict[str, Any]) -> None:
        if self.current_campfire is None:
            raise RuntimeError("native_sim_v3 CAMPFIRE step requested without campfire state")
        relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
        name = str(action.get("name") or "").upper()
        if name == "REST":
            heal_amount = rest_amount(self.player.max_hp)
            if "Regal Pillow" in relic_ids:
                heal_amount += regal_pillow_bonus()
            self._heal_player(heal_amount)
            self.current_campfire = None
            if "Dream Catcher" in relic_ids:
                self.reward_cards, self.card_blizz_randomizer = generate_card_reward_with_state(
                    self.randoms,
                    count=3,
                    card_blizz_randomizer=self.card_blizz_randomizer,
                    owned_relic_ids=relic_ids,
                    runtime_card_pools=self.card_pools,
                )
                self.reward_card_groups = []
                self.reward_gold = None
                self.reward_stolen_gold = None
                self.reward_potions = []
                self.reward_relics = []
                self.reward_emerald_key = False
                self.reward_card_screen_open = True
                self.phase = "CARD_REWARD"
                self.current_room_type = "RestRoom"
            else:
                self.phase = "MAP"
                self.current_room_type = "Map"
            return
        if name == "SMITH":
            self.current_campfire = None
            self._open_card_select(mode="upgrade", return_phase="MAP", source="campfire")
            return
        if name == "TOKE":
            self.current_campfire = None
            self._open_card_select(mode="purge", return_phase="MAP", source="campfire")
            return
        if name == "DIG":
            self.current_campfire = None
            self.reward_gold = None
            self.reward_stolen_gold = None
            self.reward_cards = []
            self.reward_card_groups = []
            self.reward_card_screen_open = False
            self.reward_potions = []
            self.reward_relics = [self._generate_campfire_dig_relic()]
            self.reward_emerald_key = False
            self.phase = "CARD_REWARD"
            self.current_room_type = "RestRoom"
            return
        if name == "LIFT":
            girya = next((relic for relic in self.relics if str(relic.get("relic_id") or relic.get("id")) == "Girya"), None)
            if girya is not None:
                girya["counter"] = min(3, int(girya.get("counter") or 0) + 1)
            self.current_campfire = None
            self.phase = "MAP"
            self.current_room_type = "Map"
            return
        if name == "RECALL":
            self.current_campfire = None
            self.has_ruby_key = True
            self.phase = "MAP"
            self.current_room_type = "Map"
            return
        if name == "PROCEED":
            self.current_campfire = None
            self.phase = "MAP"
            self.current_room_type = "Map"
            return
        raise NotImplementedError(f"native_sim_v3 campfire action {name!r} is not implemented yet.")

    def _step_treasure(self, action: dict[str, Any]) -> None:
        if self.current_treasure is None:
            raise RuntimeError("native_sim_v3 TREASURE step requested without generated treasure state")
        if self.boss_relic_pending_act_advance:
            self.boss_relic_pending_act_advance = False
            self.current_treasure = None
            if self.act == 3:
                self._open_spire_heart_victory_room()
            else:
                self._advance_to_next_act()
            return
        if not self.current_treasure.opened:
            if self.current_room_type == "TreasureRoomBoss" and self.boss_reward_pending_boss_relic:
                self.current_treasure.opened = True
                self._open_boss_relic_reward()
                return
            relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
            if "Cursed Key" in relic_ids and self.current_room_type == "TreasureRoom":
                curse = draw_neow_curse(self.randoms, runtime_pools=self.card_pools)
                omamori = next(
                    (
                        relic
                        for relic in self.relics
                        if str(relic.get("relic_id") or relic.get("id")) == "Omamori"
                    ),
                    None,
                )
                if omamori is not None and int(omamori.get("counter") or 0) > 0 and not bool(omamori.get("used_up")):
                    self._obtain_card(curse)
                else:
                    self.pending_cursed_key_chest_curse = curse
            rewards = open_treasure(
                self.current_treasure,
                self.randoms,
                relic_drawer=self._pop_relic_from_pool,
                player_relics=self.relics,
            )
            self.reward_gold = int(rewards.get("gold") or 0) or None
            self.reward_stolen_gold = None
            self.reward_relics = [dict(relic) for relic in list(rewards.get("relics") or [])]
            self.reward_potions = []
            self.reward_emerald_key = False
            self.reward_sapphire_key_relic_id = None
            self.reward_order = [dict(item) for item in list(rewards.get("order") or [])]
            self._offer_sapphire_key_for_last_reward_relic(append_to_order=True)
            self.reward_cards = []
            self.reward_card_groups = []
            self.reward_card_screen_open = False
            self.phase = "CARD_REWARD"
            self.current_room_type = "TreasureRoom"
            return
        self.current_treasure = None
        self.phase = "MAP"
        self.current_room_type = "Map"

    def _step_card_select(self, action: dict[str, Any]) -> None:
        if self.current_card_select is None:
            raise RuntimeError("native_sim_v3 CARD_SELECT step requested without generated card-select state")
        mode = str(self.current_card_select.get("mode") or "")
        confirming_selection = str(action.get("kind") or "").lower() == "confirm"
        select_cards = self.current_card_select.get("cards")
        selected_cards_to_apply: list[dict[str, Any]] = []
        target_index = action.get("target_index")
        if confirming_selection:
            selected_cards_to_apply = [dict(card) for card in list(self.current_card_select.get("selected_cards") or [])]
            selected_target_indexes = list(self.current_card_select.get("selected_target_indexes") or [])
            if selected_target_indexes:
                target_index = int(selected_target_indexes[0])
            if mode == "pandora_confirm":
                pending_deck = self.current_card_select.get("pending_deck")
                if pending_deck is not None:
                    self.deck = [dict(card) for card in list(pending_deck or [])]
        elif select_cards is not None:
            choice_index = int(action.get("choice_index") or action.get("card_index") or 0)
            if 0 <= choice_index < len(select_cards):
                selected_card = dict(select_cards[choice_index])
                if mode == "library":
                    self._obtain_card(selected_card)
                elif bool(self.current_card_select.get("requires_confirm", False)):
                    self.current_card_select["selected_cards"] = [selected_card]
                    self.current_card_select["selected_target_indexes"] = [choice_index]
                    self.current_card_select["confirm_up"] = True
                    self.phase = "CARD_SELECT"
                    return
            if mode == "pandora_confirm":
                pending_deck = self.current_card_select.get("pending_deck")
                if pending_deck is not None:
                    self.deck = [dict(card) for card in list(pending_deck or [])]
        if not confirming_selection and target_index is not None and 0 <= int(target_index) < len(self.deck):
            target_index = int(target_index)
            selected_card = dict(self.deck[target_index])
            total_picks = int(
                self.current_card_select.get("total_picks")
                or self.current_card_select.get("remaining_picks")
                or 1
            )
            if total_picks > 1:
                selected_target_indexes = list(self.current_card_select.get("selected_target_indexes") or [])
                if target_index not in {int(index) for index in selected_target_indexes}:
                    selected_target_indexes.append(target_index)
                    selected_cards = list(self.current_card_select.get("selected_cards") or [])
                    selected_cards.append(dict(selected_card))
                    self.current_card_select["selected_target_indexes"] = selected_target_indexes
                    self.current_card_select["selected_cards"] = selected_cards
                if len(selected_target_indexes) < total_picks:
                    self.phase = "CARD_SELECT"
                    return
                selected_cards_to_apply = [dict(card) for card in list(self.current_card_select.get("selected_cards") or [])]
            else:
                selected_cards_to_apply = [selected_card]
                if bool(self.current_card_select.get("requires_confirm", False)):
                    self.current_card_select["selected_target_indexes"] = [target_index]
                    self.current_card_select["selected_cards"] = [dict(selected_card)]
                    self.current_card_select["confirm_up"] = True
                    self.phase = "CARD_SELECT"
                    return

        if (
            not confirming_selection
            and bool(self.current_card_select.get("requires_confirm", False))
            and selected_cards_to_apply
        ):
            self.current_card_select["selected_cards"] = [dict(card) for card in selected_cards_to_apply]
            self.current_card_select["confirm_up"] = True
            self.phase = "CARD_SELECT"
            return

        for selected_card in selected_cards_to_apply:
            if (
                mode in {"purge", "remove"}
                and self.current_card_select.get("source") == "event"
                and self.current_event is not None
                and self.current_event.event_id == "NoteForYourself"
            ):
                self.note_for_yourself_card_id = str(selected_card.get("card_id") or self.note_for_yourself_card_id)
                self.note_for_yourself_upgrades = int(selected_card.get("upgrades") or 0)
            if mode in {"purge", "remove"}:
                selected_uuid = str(selected_card.get("uuid") or "")
                removed = False
                if selected_uuid:
                    removed = self._remove_deck_card_by_uuid(selected_uuid) is not None
                elif 0 <= int(target_index) < len(self.deck):
                    self._remove_deck_card_at(int(target_index))
            elif mode == "duplicate":
                duplicate = dict(selected_card)
                original_uuid = str(selected_card.get("uuid") or selected_card.get("card_id") or "card")
                duplicate["uuid"] = f"duplicate-{len(self.deck)}-{original_uuid}"
                for bottle_key in ("bottled", "in_bottle_flame", "in_bottle_lightning", "in_bottle_tornado"):
                    duplicate[bottle_key] = False
                self._obtain_card(duplicate)
            elif mode == "transform":
                selected_uuid = str(selected_card.get("uuid") or "")
                if selected_uuid:
                    self._remove_deck_card_by_uuid(selected_uuid)
                elif 0 <= int(target_index) < len(self.deck):
                    self._remove_deck_card_at(int(target_index))
                transform_rng_stream = str(self.current_card_select.get("transform_rng_stream") or "")
                if not transform_rng_stream:
                    transform_rng_stream = "neow" if self.current_card_select.get("source") == "neow" else "misc"
                transformed = neow_transform_card(
                    self.randoms,
                    selected_card,
                    runtime_pools=self.card_pools,
                    source_pools=self.source_card_pools,
                    rng_stream=transform_rng_stream,
                )
                if bool(self.current_card_select.get("auto_upgrade_transformed", False)) and can_upgrade_card(transformed):
                    transformed = upgrade_card(transformed)
                self._obtain_card(transformed)
            elif mode == "upgrade":
                selected_uuid = str(selected_card.get("uuid") or "")
                upgrade_index = int(target_index) if 0 <= int(target_index) < len(self.deck) else None
                if upgrade_index is None:
                    upgrade_index = next(
                        (
                            index for index, card in enumerate(self.deck)
                            if selected_uuid and str(card.get("uuid") or "") == selected_uuid
                        ),
                        None,
                    )
                if upgrade_index is not None:
                    old = self.deck[int(upgrade_index)]
                    new_upgrades = int(old.get("upgrades") or 0) + 1
                    upgraded = make_card(str(old["card_id"]), upgrades=new_upgrades, uuid=str(old.get("uuid") or f"upgrade-{upgrade_index}"))
                    self.deck[int(upgrade_index)] = upgraded
            elif mode == "bottle_flame":
                self.deck[int(target_index)]["bottled"] = True
                self.deck[int(target_index)]["in_bottle_flame"] = True
                relic = next((item for item in self.relics if str(item.get("relic_id") or item.get("id")) == "Bottled Flame"), None)
                if relic is not None:
                    relic["card_id"] = str(selected_card.get("card_id") or "")
                    relic["card_uuid"] = str(selected_card.get("uuid") or "")
            elif mode == "bottle_lightning":
                self.deck[int(target_index)]["bottled"] = True
                self.deck[int(target_index)]["in_bottle_lightning"] = True
                relic = next((item for item in self.relics if str(item.get("relic_id") or item.get("id")) == "Bottled Lightning"), None)
                if relic is not None:
                    relic["card_id"] = str(selected_card.get("card_id") or "")
                    relic["card_uuid"] = str(selected_card.get("uuid") or "")
            elif mode == "bottle_tornado":
                self.deck[int(target_index)]["bottled"] = True
                self.deck[int(target_index)]["in_bottle_tornado"] = True
                relic = next((item for item in self.relics if str(item.get("relic_id") or item.get("id")) == "Bottled Tornado"), None)
                if relic is not None:
                    relic["card_id"] = str(selected_card.get("card_id") or "")
                    relic["card_uuid"] = str(selected_card.get("uuid") or "")
            if self.current_card_select.get("selection_effect") == "bonfire":
                self._apply_bonfire_reward(selected_card)
            elif self.current_card_select.get("selection_effect") == "designer_remove_and_upgrade":
                candidate_indexes = [
                    index for index, card in enumerate(self.deck)
                    if int(card.get("upgrades") or 0) <= 0 and str(card.get("type") or "") not in {"STATUS", "CURSE"}
                ]
                if candidate_indexes:
                    ordered = list(candidate_indexes)
                    java_shuffle_in_place(ordered, int(self.randoms.stream("misc").random_long()))
                    upgrade_index = ordered[0]
                    old = self.deck[upgrade_index]
                    self.deck[upgrade_index] = make_card(
                        str(old["card_id"]),
                        upgrades=int(old.get("upgrades") or 0) + 1,
                        uuid=str(old.get("uuid") or f"designer-upgrade-{upgrade_index}"),
                    )
        completed_multi_select = (
            bool(selected_cards_to_apply)
            and int(self.current_card_select.get("total_picks") or 1) > 1
        )
        remaining_picks = 0 if completed_multi_select else int(self.current_card_select.get("remaining_picks") or 1) - 1
        if remaining_picks > 0:
            self.current_card_select["remaining_picks"] = remaining_picks
            self.current_card_select["candidate_indexes"] = self._card_select_candidates(mode)
            if self.current_card_select["candidate_indexes"]:
                self.phase = "CARD_SELECT"
                return
        return_phase = str(self.current_card_select.get("return_phase") or "MAP")
        source = str(self.current_card_select.get("source") or "")
        clear_event_on_finish = bool(self.current_card_select.get("clear_event_on_finish", source == "event"))
        self.current_card_select = None
        if source == "event" and clear_event_on_finish:
            self.current_event = None
        if source == "neow":
            self.neow_pending_continue = True
        if return_phase == "SPIRE_HEART":
            self._open_spire_heart_victory_room()
            return
        if return_phase == "ADVANCE_ACT":
            if source == "boss_relic":
                self._wait_on_boss_chest_after_relic_choice()
                return
            self._advance_to_next_act()
            return
        self.phase = return_phase
        if return_phase == "MAP":
            self.current_room_type = "Map"
        elif return_phase == "EVENT":
            self.current_room_type = "EventRoom"

    def _card_select_candidates(self, mode: str) -> list[int]:
        if mode == "upgrade":
            return [
                index for index, card in enumerate(self.deck)
                if can_upgrade_card(card)
            ]
        if mode == "bottle_flame":
            return [
                index for index in range(len(self.deck) - 1, -1, -1)
                if str(self.deck[index].get("type") or "") == "ATTACK"
            ]
        if mode == "bottle_lightning":
            return [
                index for index in range(len(self.deck) - 1, -1, -1)
                if str(self.deck[index].get("type") or "") == "SKILL"
            ]
        if mode == "bottle_tornado":
            return [
                index for index in range(len(self.deck) - 1, -1, -1)
                if str(self.deck[index].get("type") or "") == "POWER"
            ]
        if mode == "purge":
            return [
                index for index, card in enumerate(self.deck)
                if str(card.get("type") or "") != "STATUS"
                and not bool(card.get("bottled") or card.get("in_bottle_flame") or card.get("in_bottle_lightning") or card.get("in_bottle_tornado"))
            ]
        return [
            index for index, card in enumerate(self.deck)
            if str(card.get("type") or "") != "STATUS"
        ]

    def _open_card_select(
        self,
        *,
        mode: str,
        return_phase: str,
        source: str,
        remaining_picks: int = 1,
        requires_confirm: bool = True,
    ) -> None:
        self.current_card_select = {
            "mode": mode,
            "candidate_indexes": self._card_select_candidates(mode),
            "return_phase": return_phase,
            "source": source,
            "remaining_picks": int(remaining_picks),
            "total_picks": int(remaining_picks),
            "selected_target_indexes": [],
            "selected_cards": [],
            "requires_confirm": bool(requires_confirm),
        }
        self.phase = "CARD_SELECT"

    def _step_boss_relic(self, action: dict[str, Any]) -> None:
        index = int(action.get("choice_index") or 0)
        if action.get("kind") == "boss_relic" and 0 <= index < len(self.boss_relic_options):
            chosen_relic = dict(self.boss_relic_options[index])
            self.relics = apply_boss_relic_choice(self.relics, chosen_relic)
            self._apply_obtained_relic_side_effects(chosen_relic, source="boss_relic")
        self.boss_relic_options = []
        self.boss_reward_pending_boss_relic = False
        if self.phase in {"CARD_REWARD", "CARD_SELECT"}:
            return
        self._wait_on_boss_chest_after_relic_choice()

    def _act_floor_offset(self) -> int:
        return {1: 0, 2: 17, 3: 34, 4: 51}.get(self.act, 0)

    def _apply_pending_show_card_obtain_effects(self) -> None:
        if not self.pending_show_card_obtain_effects:
            return
        pending = [dict(card) for card in self.pending_show_card_obtain_effects]
        self.pending_show_card_obtain_effects = []
        for card in pending:
            self._obtain_card(card)

    def _apply_act_transition_card_rng_counter(self) -> None:
        card_rng = self.randoms.stream("card")
        if 0 < card_rng.counter < 250:
            card_rng.set_counter(250)
        elif 250 < card_rng.counter < 500:
            card_rng.set_counter(500)
        elif 500 < card_rng.counter < 750:
            card_rng.set_counter(750)

    def _advance_to_next_act(self) -> None:
        if self.act >= 4:
            self.phase = "VICTORY"
            return
        if self.act == 3 and not (self.final_act_available and self.has_ruby_key and self.has_emerald_key and self.has_sapphire_key):
            self.phase = "VICTORY"
            return
        if self.act == 3:
            self.dungeon_id = dungeon_id_for_act(4)
            self.act = 4
        else:
            next_dungeon = next_dungeon_id(self.dungeon_id, endless=False)
            if next_dungeon is None:
                self.phase = "VICTORY"
                return
            self.dungeon_id = str(next_dungeon)
            self.act = act_for_dungeon_id(self.dungeon_id)
        self._apply_act_transition_card_rng_counter()
        self.floor = self._act_floor_offset()
        self.player.current_hp = int(self.player.max_hp)
        self.randoms.reset_act_stream(self.act)
        self.blizzard_potion_mod = 0
        self.current_map_node = None
        self.first_room_chosen = False
        self.current_event = None
        self.current_campfire = None
        self.current_shop = None
        self.current_treasure = None
        self.boss_relic_options = []
        self.reward_cards = []
        self.reward_gold = None
        self.reward_stolen_gold = None
        self.reward_potions = []
        self.reward_relics = []
        self.reward_emerald_key = False
        self.reward_sapphire_key_relic_id = None
        self.reward_card_screen_open = False
        self.boss_reward_pending_boss_relic = False
        self.boss_relic_pending_act_advance = False
        self.map = generate_act_map(
            self.randoms,
            act=self.act,
            ascension_level=self.ascension_level,
            final_act_available=self.final_act_available,
            has_emerald_key=self.has_emerald_key,
        )
        self.monster_list, self.elite_monster_list, self.boss_list = generate_monster_lists_for_dungeon(
            self.randoms,
            self.dungeon_id,
        )
        self.act_boss = self.boss_list[0] if self.boss_list else None
        if self.act == 4:
            self.boss_relic_pool = []
            self.event_list, self.shrine_list = [], []
        else:
            self.event_list, self.shrine_list = initialize_event_pools_for_dungeon(self.dungeon_id)
        self.question_elite_chance = RESET_ELITE_CHANCE
        self.question_monster_chance = RESET_MONSTER_CHANCE
        self.question_shop_chance = RESET_SHOP_CHANCE
        self.question_treasure_chance = RESET_TREASURE_CHANCE
        self.phase = "MAP"
        self.current_room_type = "Map"

    def _finish_reward_return(self) -> None:
        phase = str(self.reward_return_phase or "")
        room_type = self.reward_return_room_type
        neow_continue = bool(self.reward_return_neow_continue)
        self.reward_return_phase = None
        self.reward_return_room_type = None
        self.reward_return_neow_continue = False
        self.reward_emerald_key = False
        self.reward_sapphire_key_relic_id = None
        self.reward_card_screen_open = False
        if phase == "SPIRE_HEART":
            self._open_spire_heart_victory_room()
            return
        if phase == "ADVANCE_ACT":
            self._advance_to_next_act()
            return
        self.phase = phase or "MAP"
        self.current_room_type = room_type or ("NeowRoom" if phase == "NEOW" else "Map")
        if phase == "NEOW" and neow_continue:
            self.neow_pending_continue = True

    def _open_spire_heart_victory_room(self) -> None:
        self.current_event = EventState("Spire Heart")
        self.current_campfire = None
        self.current_shop = None
        self.current_treasure = None
        self.phase = "EVENT"
        self.current_room_type = "VictoryRoom"

    def _boss_relic_return_phase(self) -> str:
        return "SPIRE_HEART" if self.act == 3 else "ADVANCE_ACT"

    def _obtain_relic(self, relic: dict[str, Any], *, source: str) -> None:
        relic = dict(relic)
        relic_id = str(relic.get("relic_id") or relic.get("id") or "")
        if relic_id and relic.get("counter") is None:
            try:
                default_counter = make_relic(relic_id).get("counter")
            except KeyError:
                default_counter = None
            if default_counter is not None:
                relic["counter"] = default_counter
        if relic_id == "Circlet":
            existing = next(
                (owned for owned in self.relics if str(owned.get("relic_id") or owned.get("id") or "") == "Circlet"),
                None,
            )
            if existing is not None:
                existing["counter"] = int(existing.get("counter") or 0) + 1
                return
            relic.setdefault("counter", 1)
        if relic_id == "Girya":
            relic.setdefault("counter", 0)
        if relic_id in {"Happy Flower", "Incense Burner", "Nunchaku", "InkBottle", "Sundial", "Pen Nib"}:
            relic.setdefault("counter", 0)
        self.relics.append(relic)
        self._apply_obtained_relic_side_effects(self.relics[-1], source=source)

    def _remove_relic(self, remove_relic_id: str) -> dict[str, Any] | None:
        for index, relic in enumerate(self.relics):
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id != str(remove_relic_id):
                continue
            self._apply_removed_relic_side_effects(relic)
            return self.relics.pop(index)
        return None

    def _current_relic_ids(self) -> set[str]:
        return {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}

    def _refresh_pending_reward_card_preview_relics(self) -> None:
        relic_ids = self._current_relic_ids()
        self.reward_cards = apply_reward_preview_relics(self.reward_cards, owned_relic_ids=relic_ids)
        self.reward_card_groups = [
            apply_reward_preview_relics([dict(card) for card in group], owned_relic_ids=relic_ids)
            for group in self.reward_card_groups
        ]

    def _apply_obtained_relic_side_effects(self, relic: dict[str, Any], *, source: str) -> None:
        relic_id = str(relic.get("relic_id") or relic.get("id") or "")
        if relic_id == "Necronomicon":
            self._obtain_card(make_card("Necronomicurse"))
            return
        if relic_id in {"Molten Egg 2", "Toxic Egg 2", "Frozen Egg 2"}:
            self._refresh_pending_reward_card_preview_relics()
            return
        if relic_id == "Strawberry":
            self._increase_max_hp(7)
            return
        if relic_id == "Pear":
            self._increase_max_hp(10)
            return
        if relic_id == "Mango":
            self._increase_max_hp(14)
            return
        if relic_id == "Lee's Waffle":
            self.player.max_hp += 7
            self._heal_player(self.player.max_hp)
            return

        if relic_id == "Old Coin":
            self._gain_gold(300)
            return
        if relic_id == "Potion Belt":
            self.max_potion_slots += 2
            self.potions = _normalize_potion_slots(self.potions, self.max_potion_slots)
            return
        if relic_id == "Whetstone":
            if source == "neow":
                self.randoms.stream("misc").random_boolean()
            self._upgrade_random_deck_cards_by_type("ATTACK", 2)
            return
        if relic_id == "War Paint":
            if source == "neow":
                self.randoms.stream("misc").random_boolean()
            self._upgrade_random_deck_cards_by_type("SKILL", 2)
            return
        if relic_id == "DollysMirror":
            if self._card_select_candidates("duplicate"):
                self._open_card_select(
                    mode="duplicate",
                    return_phase="SHOP" if source == "shop" else self.phase,
                    source=source,
                    remaining_picks=1,
                    requires_confirm=False,
                )
            return
        if relic_id == "Du-Vu Doll":
            self._refresh_deck_scaled_relics()
            return
        if relic_id in {"Happy Flower", "Incense Burner"}:
            relic["counter"] = 0
            return
        if relic_id == "Bottled Flame":
            if self._card_select_candidates("bottle_flame"):
                self._open_card_select(
                    mode="bottle_flame",
                    return_phase="SHOP" if source == "shop" else self.phase,
                    source=source,
                    remaining_picks=1,
                    requires_confirm=False,
                )
                if source == "event" and self.current_card_select is not None:
                    self.current_card_select["clear_event_on_finish"] = False
            return
        if relic_id == "Bottled Lightning":
            if self._card_select_candidates("bottle_lightning"):
                self._open_card_select(
                    mode="bottle_lightning",
                    return_phase="SHOP" if source == "shop" else self.phase,
                    source=source,
                    remaining_picks=1,
                    requires_confirm=False,
                )
                if source == "event" and self.current_card_select is not None:
                    self.current_card_select["clear_event_on_finish"] = False
            return
        if relic_id == "Bottled Tornado":
            if self._card_select_candidates("bottle_tornado"):
                self._open_card_select(
                    mode="bottle_tornado",
                    return_phase="SHOP" if source == "shop" else self.phase,
                    source=source,
                    remaining_picks=1,
                    requires_confirm=False,
                )
                if source == "event" and self.current_card_select is not None:
                    self.current_card_select["clear_event_on_finish"] = False
            return
        if relic_id == "Astrolabe":
            purgeable_indexes = self._card_select_candidates("transform")
            return_phase = self._boss_relic_return_phase() if source == "boss_relic" else self.phase
            select_source = "neow" if source == "neow_boss_relic" else source
            if len(purgeable_indexes) > 3:
                self._open_card_select(
                    mode="transform",
                    return_phase=return_phase,
                    source=select_source,
                    remaining_picks=3,
                    requires_confirm=False,
                )
                if self.current_card_select is not None:
                    self.current_card_select["transform_rng_stream"] = "misc"
                    self.current_card_select["visible_for_transform"] = False
                    self.current_card_select["auto_upgrade_transformed"] = True
                return
            for target_index in purgeable_indexes:
                self._apply_removed_deck_card_side_effects(self.deck[target_index])
                transformed = neow_transform_card(
                    self.randoms,
                    self.deck[target_index],
                    runtime_pools=self.card_pools,
                    source_pools=self.source_card_pools,
                    rng_stream="misc",
                )
                if can_upgrade_card(transformed):
                    transformed = upgrade_card(transformed)
                self.deck[target_index] = transformed
            self._refresh_deck_scaled_relics()
            return
        if relic_id == "Empty Cage":
            purgeable_indexes = self._card_select_candidates("remove")
            return_phase = self._boss_relic_return_phase() if source == "boss_relic" else self.phase
            select_source = "neow" if source == "neow_boss_relic" else source
            if len(purgeable_indexes) > 2:
                self._open_card_select(
                    mode="remove",
                    return_phase=return_phase,
                    source=select_source,
                    remaining_picks=2,
                    requires_confirm=False,
                )
                return
            for target_index in sorted(purgeable_indexes[:2], reverse=True):
                self._remove_deck_card_at(int(target_index))
            return
        if relic_id == "Pandora's Box":
            transformed_count = 0
            kept_deck: list[dict[str, Any]] = []
            for card in self.deck:
                card_id = str(card.get("card_id") or "")
                if card_id in {"Strike_R", "Defend_R"}:
                    transformed_count += 1
                    continue
                kept_deck.append(card)
            generated_cards: list[dict[str, Any]] = []
            for _ in range(transformed_count):
                generated = truly_random_card_from_source_pools(
                    self.randoms,
                    source_pools=self.source_card_pools,
                    include_colorless=False,
                    in_combat=False,
                    rng_stream="card_random",
                )
                if generated is not None:
                    generated_cards.extend(
                        apply_reward_preview_relics([dict(generated)], owned_relic_ids=self._current_relic_ids())
                    )
            if generated_cards:
                return_phase = self._boss_relic_return_phase() if source == "boss_relic" else self.phase
                select_source = "neow" if source == "neow_boss_relic" else source
                self.deck = [dict(card) for card in kept_deck]
                display_cards = [dict(card) for card in reversed(generated_cards)]
                self.current_card_select = {
                    "mode": "pandora_confirm",
                    "cards": display_cards,
                    "pending_deck": [*kept_deck, *display_cards],
                    "return_phase": return_phase,
                    "source": select_source,
                    "remaining_picks": 1,
                    "total_picks": 1,
                    "confirm_up": True,
                }
                self.phase = "CARD_SELECT"
                self.current_room_type = "CARD_SELECT"
                return
            self.deck = kept_deck
            return
        if relic_id == "Calling Bell":
            self._obtain_card(make_card("CurseOfTheBell"))
            self.reward_gold = None
            self.reward_stolen_gold = None
            self.reward_potions = []
            self.reward_relics = [
                self._pop_screenless_relic_from_pool("COMMON"),
                self._pop_screenless_relic_from_pool("UNCOMMON"),
                self._pop_screenless_relic_from_pool("RARE"),
            ]
            self.reward_emerald_key = False
            self.reward_cards = []
            self.reward_card_groups = []
            self.reward_card_screen_open = False
            self.reward_return_phase = self._boss_relic_return_phase() if source == "boss_relic" else self.phase
            self.reward_return_room_type = "TreasureRoomBoss" if source == "boss_relic" else self.current_room_type
            self.reward_return_neow_continue = source == "neow_boss_relic"
            self.phase = "CARD_REWARD"
            return
        if relic_id == "Orrery":
            reward_groups, self.card_blizz_randomizer = generate_card_reward_groups_with_state(
                self.randoms,
                group_count=4,
                card_blizz_randomizer=self.card_blizz_randomizer,
                owned_relic_ids={str(current.get("relic_id") or current.get("id")) for current in self.relics},
                runtime_card_pools=self.card_pools,
            )
            self.reward_gold = None
            self.reward_stolen_gold = None
            self.reward_potions = []
            self.reward_relics = []
            self.reward_emerald_key = False
            self.reward_cards = [dict(card) for card in reward_groups[0]] if reward_groups else []
            self.reward_card_groups = [[dict(card) for card in group] for group in reward_groups[1:]]
            self.reward_card_screen_open = True
            self.reward_return_phase = "SHOP" if source == "shop" else self.phase
            self.reward_return_room_type = "ShopRoom" if source == "shop" else self.current_room_type
            self.reward_return_neow_continue = source == "neow_boss_relic"
            self.phase = "CARD_REWARD"
            return
        if relic_id == "Tiny House":
            upgradable_indexes = [
                index
                for index, card in enumerate(self.deck)
                if can_upgrade_card(card)
            ]
            if upgradable_indexes:
                java_shuffle_in_place(upgradable_indexes, int(self.randoms.stream("misc").random_long()))
                target_index = int(upgradable_indexes[0])
                target_card = dict(self.deck[target_index])
                self.deck[target_index] = make_card(
                    str(target_card["card_id"]),
                    upgrades=int(target_card.get("upgrades") or 0) + 1,
                    uuid=str(target_card.get("uuid") or f"tiny-house-{target_card['card_id']}"),
                )
            self._increase_max_hp(5)
            self.reward_gold = 50
            self.reward_stolen_gold = None
            self.reward_potions = [
                draw_random_potion(self.randoms, player_class=self.player_class, stream_name="misc")
            ]
            self.reward_relics = []
            self.reward_emerald_key = False
            card_upgraded_chance = act_chances(str(self.dungeon_id)).card_upgraded_chance(self.ascension_level)
            self.reward_cards, self.card_blizz_randomizer = generate_card_reward_with_state(
                self.randoms,
                count=3,
                card_blizz_randomizer=self.card_blizz_randomizer,
                card_upgraded_chance=card_upgraded_chance,
                owned_relic_ids=self._current_relic_ids(),
                runtime_card_pools=self.card_pools,
            )
            self.reward_card_groups = []
            self.reward_card_screen_open = False
            self.reward_return_phase = self._boss_relic_return_phase() if source == "boss_relic" else self.phase
            self.reward_return_room_type = "TreasureRoomBoss" if source == "boss_relic" else self.current_room_type
            self.reward_return_neow_continue = source == "neow_boss_relic"
            self.phase = "CARD_REWARD"
            return
        if relic_id == "Cauldron":
            self.reward_gold = None
            self.reward_stolen_gold = None
            self.reward_potions = [draw_random_potion(self.randoms, player_class=self.player_class) for _ in range(5)]
            self.reward_relics = []
            self.reward_emerald_key = False
            self.reward_cards = []
            self.reward_card_groups = []
            self.reward_card_screen_open = False
            self.reward_return_phase = "MAP" if source == "shop" else self.phase
            self.reward_return_room_type = "ShopRoom" if source == "shop" else self.current_room_type
            self.reward_return_neow_continue = False
            self.phase = "CARD_REWARD"
            return

    def _apply_removed_relic_side_effects(self, relic: dict[str, Any]) -> None:
        relic_id = str(relic.get("relic_id") or relic.get("id") or "")
        # Native AbstractRelic.onUnequip hooks with persistent run-state effects belong here.
        if relic_id == "Necronomicon":
            for index, card in enumerate(self.deck):
                if str(card.get("card_id") or card.get("id") or "") == "Necronomicurse":
                    self._remove_deck_card_at(index)
                    break

    def _promote_next_reward_card_group(self) -> None:
        if self.reward_cards or not self.reward_card_groups:
            return
        self.reward_cards = apply_reward_preview_relics(
            [dict(card) for card in self.reward_card_groups.pop(0)],
            owned_relic_ids=self._current_relic_ids(),
        )

    def _apply_bonfire_reward(self, card: dict[str, Any]) -> None:
        rarity = str(card.get("rarity") or "")
        relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
        if rarity == "CURSE":
            self._obtain_relic(make_relic("Circlet" if "Spirit Poop" in relic_ids else "Spirit Poop"), source="event")
            return
        if rarity in {"COMMON", "SPECIAL"}:
            self._heal_player(5)
            return
        if rarity == "UNCOMMON":
            self._heal_player(int(self.player.max_hp) - int(self.player.current_hp))
            return
        if rarity == "RARE":
            self._increase_max_hp(10, heal=False)
            self._heal_player(int(self.player.max_hp) - int(self.player.current_hp))

    def _increase_max_hp(self, amount: int, *, heal: bool = True) -> None:
        delta = max(0, int(amount))
        if delta <= 0:
            return
        self.player.max_hp = int(self.player.max_hp) + delta
        if heal:
            self._heal_player(delta)

    def _decrease_max_hp(self, amount: int) -> None:
        delta = max(0, int(amount))
        if delta <= 0:
            return
        self.player.max_hp = max(1, int(self.player.max_hp) - delta)
        self.player.current_hp = min(int(self.player.current_hp), int(self.player.max_hp))

    def _apply_removed_deck_card_side_effects(self, card: dict[str, Any]) -> None:
        card_id = str(card.get("card_id") or card.get("id") or "")
        if card_id == "Parasite":
            self._decrease_max_hp(3)

    def _remove_deck_card_at(self, index: int) -> dict[str, Any] | None:
        if not 0 <= int(index) < len(self.deck):
            return None
        removed = self.deck.pop(int(index))
        self._apply_removed_deck_card_side_effects(removed)
        self._refresh_deck_scaled_relics()
        return removed

    def _remove_deck_card_by_uuid(self, card_uuid: str) -> dict[str, Any] | None:
        target_uuid = str(card_uuid)
        if not target_uuid:
            return None
        for index, card in enumerate(self.deck):
            if str(card.get("uuid") or "") == target_uuid:
                return self._remove_deck_card_at(index)
        return None

    def _remove_deck_cards_by_uuid(self, card_uuids: set[str]) -> list[dict[str, Any]]:
        targets = {str(card_uuid) for card_uuid in card_uuids if str(card_uuid)}
        removed_cards: list[dict[str, Any]] = []
        if not targets:
            return removed_cards
        index = 0
        while index < len(self.deck):
            if str(self.deck[index].get("uuid") or "") in targets:
                removed = self._remove_deck_card_at(index)
                if removed is not None:
                    removed_cards.append(removed)
                continue
            index += 1
        return removed_cards

    def _heal_player(self, amount: int, *, apply_magic_flower: bool = False) -> None:
        if amount <= 0:
            return
        relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
        if "Mark of the Bloom" in relic_ids:
            return
        heal = int(amount)
        if apply_magic_flower and "Magic Flower" in relic_ids:
            heal = _java_round_positive(float(heal) * 1.5)
        self.player.current_hp = min(self.player.max_hp, int(self.player.current_hp) + heal)

    def _obtained_card_payload(self, card: dict[str, Any]) -> dict[str, Any]:
        payload = dict(card)
        card_id = str(payload.get("card_id") or payload.get("id") or "")
        if not card_id:
            return make_card(str(card["card_id"])) if "cost" not in card else payload
        try:
            upgrades = int(payload.get("upgrades") or 0)
        except (TypeError, ValueError):
            upgrades = 0
        uuid = payload.get("uuid")
        try:
            canonical = make_card(card_id, upgrades=upgrades, uuid=str(uuid) if uuid not in {None, ""} else None)
        except KeyError:
            return payload
        canonical.update(payload)
        canonical["card_id"] = card_id
        return canonical

    def _obtain_card(self, card: dict[str, Any]) -> dict[str, Any]:
        payload = self._obtained_card_payload(card)
        card_type = str(payload.get("type") or "")
        card_color = str(payload.get("color") or "")
        card_rarity = str(payload.get("rarity") or "")
        if card_type == "CURSE" or card_color == "CURSE" or card_rarity == "CURSE":
            for relic in self.relics:
                relic_id = str(relic.get("relic_id") or relic.get("id") or "")
                if relic_id != "Omamori":
                    continue
                counter = int(relic.get("counter", 0) or 0)
                if counter <= 0 or bool(relic.get("used_up")):
                    continue
                counter -= 1
                relic["counter"] = counter
                if counter <= 0:
                    relic["counter"] = 0
                    relic["used_up"] = True
                return payload
        if int(payload.get("upgrades") or 0) <= 0 and str(payload.get("can_upgrade") or "").lower() != "false":
            if self._has_relic("Molten Egg 2") and card_type == "ATTACK":
                payload = upgrade_card(payload)
            elif self._has_relic("Toxic Egg 2") and card_type == "SKILL":
                payload = upgrade_card(payload)
            elif self._has_relic("Frozen Egg 2") and card_type == "POWER":
                payload = upgrade_card(payload)
        self.deck.append(payload)
        if self._has_relic("CeramicFish"):
            self._gain_gold(9)
        if self._has_relic("Darkstone Periapt") and (
            card_type == "CURSE" or card_color == "CURSE" or card_rarity == "CURSE"
        ):
            self._increase_max_hp(6)
        self._refresh_deck_scaled_relics()
        return payload

    def _apply_shop_discount(self, multiplier: float, *, affect_purge: bool) -> None:
        if self.current_shop is None:
            return
        for payload in [*self.current_shop.cards, *self.current_shop.relics, *self.current_shop.potions]:
            payload["price"] = _java_round_positive(float(payload["price"]) * float(multiplier))
        relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in self.relics}
        if "Smiling Mask" in relic_ids:
            self.current_shop.purge_cost = 50
        elif affect_purge:
            self.current_shop.purge_cost = _java_round_positive(float(self.current_shop.purge_cost) * float(multiplier))

    def _generate_shop_replacement_card(self, purchased_card: dict[str, Any]) -> dict[str, Any]:
        from spirecomm.native_sim_v3.run.shop import generate_shop_replacement_card

        return generate_shop_replacement_card(
            self.randoms,
            purchased_card=purchased_card,
            existing_cards=self.current_shop.cards,
            act=self.act,
            dungeon_id=self.dungeon_id,
            card_blizz_randomizer=self.card_blizz_randomizer,
            ascension_level=self.ascension_level,
            owned_relic_ids={str(relic.get("relic_id") or relic.get("id")) for relic in self.relics},
            runtime_card_pools=self.card_pools,
        )

    def _generate_shop_replacement_relic(self, *, exclude: set[str]) -> dict[str, Any]:
        del exclude
        from spirecomm.native_sim_v3.run.shop import generate_shop_replacement_relic

        return generate_shop_replacement_relic(
            self.randoms,
            player_class=self.player_class,
            floor_num=self.floor,
            act=self.act,
            dungeon_id=self.dungeon_id,
            ascension_level=self.ascension_level,
            owned_relic_ids={str(relic.get("relic_id") or relic.get("id")) for relic in self.relics},
            relic_drawer=self._pop_relic_end_from_pool,
        )

    def _generate_shop_replacement_potion(self, index: int) -> dict[str, Any]:
        from spirecomm.native_sim_v3.run.shop import generate_shop_replacement_potion

        return generate_shop_replacement_potion(
            self.randoms,
            shop_index=index,
            dungeon_id=self.dungeon_id,
            ascension_level=self.ascension_level,
            owned_relic_ids={str(relic.get("relic_id") or relic.get("id")) for relic in self.relics},
        )

    def _generate_campfire_dig_relic(self) -> dict[str, Any]:
        return self._pop_non_campfire_relic_from_pool(roll_random_relic_tier(self.randoms, act=self.act))

    def _pop_relic_from_pool(self, tier: str) -> dict[str, Any]:
        return pop_random_relic_from_pools(
            self.relic_pools,
            tier,
            floor_num=self.floor,
            current_room_type=self.current_room_type,
            owned_relic_ids={str(relic.get("relic_id") or relic.get("id")) for relic in self.relics},
            deck=self.deck,
            act=self.act,
        )

    def _pop_relic_end_from_pool(self, tier: str) -> dict[str, Any]:
        from spirecomm.native_sim_v3.content.relics import pop_random_relic_end_from_pools

        return pop_random_relic_end_from_pools(
            self.relic_pools,
            tier,
            floor_num=self.floor,
            current_room_type=self.current_room_type,
            owned_relic_ids={str(relic.get("relic_id") or relic.get("id")) for relic in self.relics},
            deck=self.deck,
            act=self.act,
        )

    def _pop_screenless_relic_from_pool(
        self,
        tier: str | None = None,
        exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        del exclude
        chosen_tier = str(tier) if tier is not None else roll_random_relic_tier(self.randoms, act=self.act)
        return pop_random_screenless_relic_from_pools(
            self.relic_pools,
            chosen_tier,
            floor_num=self.floor,
            current_room_type=self.current_room_type,
            owned_relic_ids={str(relic.get("relic_id") or relic.get("id")) for relic in self.relics},
            deck=self.deck,
            act=self.act,
        )

    def _pop_non_campfire_relic_from_pool(
        self,
        tier: str | None = None,
    ) -> dict[str, Any]:
        chosen_tier = str(tier) if tier is not None else roll_random_relic_tier(self.randoms, act=self.act)
        return pop_random_non_campfire_relic_from_pools(
            self.relic_pools,
            chosen_tier,
            floor_num=self.floor,
            current_room_type=self.current_room_type,
            owned_relic_ids={str(relic.get("relic_id") or relic.get("id")) for relic in self.relics},
            deck=self.deck,
            act=self.act,
        )
