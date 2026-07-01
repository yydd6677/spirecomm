from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from spirecomm.ai.recording import serialize_game_state
from spirecomm.communication.action import (
    CancelAction,
    CardRewardAction,
    ChooseAction,
    ConfirmAction,
    ChooseMapBossAction,
    ChooseMapNodeAction,
    ChooseShopkeeperAction,
    CombatRewardAction,
    EndTurnAction,
    EventOptionAction,
    NeowContinueAction,
    NeowCardRewardAction,
    NeowCardSelectAction,
    OpenChestAction,
    PlayCardAction,
    PotionAction,
    ProceedAction,
    RawCommandAction,
    RestAction,
    StartGameAction,
    StateAction,
    WaitAction,
)
from spirecomm.communication.coordinator import Coordinator
from spirecomm.seed_helper import canonical_seed_string
from spirecomm.spire.card import Card
from spirecomm.spire.character import PlayerClass
from spirecomm.spire.relic import Relic
from spirecomm.spire.screen import RewardType, RestOption, ScreenType


@dataclass
class RecordedTraceStep:
    step: int
    phase: str
    floor: int | None
    action: dict[str, Any]
    pre: dict[str, Any]
    post: dict[str, Any]
    raw_pre_state: dict[str, Any] | None = None


@dataclass
class RecordedRunTrace:
    path: Path
    source_format: str
    seed_long: int | None
    seed_str: str | None
    ascension: int
    character: str | None
    steps: list[RecordedTraceStep]


class RecordedReplayAbort(RuntimeError):
    """Raised to stop a replay once a blocking divergence has been found."""


DEFAULT_REPLAY_ACTION_TIMEOUT_SECONDS = 30.0
DEFAULT_REPLAY_OUT_OF_GAME_GRACE_SECONDS = 5.0
NEOW_REWARD_BRIDGE_DELAY_SECONDS = 0.35
NEOW_REWARD_CONTINUE_DELAY_SECONDS = 0.0
POST_NEOW_SETTLE_TIMEOUT_SECONDS = 1.0


class _ReplayProgressWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __call__(self, payload: dict[str, Any]) -> None:
        progress_payload = dict(payload)
        progress_payload["updated_at"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(f"{self.path}.tmp")
        temp_path.write_text(
            json.dumps(progress_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.path)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _normalize_phase_name(value: Any) -> str:
    phase = str(value or "").upper()
    if phase == "BATTLE":
        return "COMBAT"
    return phase


def _normalize_loose_token(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _normalize_event_id(value: Any) -> str:
    token = _normalize_loose_token(value)
    if token.endswith("event"):
        token = token[: -len("event")]
    return token


def _clear_tolerable_neow_skip_error(coordinator: Coordinator, *, request_state: bool = False) -> bool:
    error = str(getattr(coordinator, "last_error", "") or "")
    if not error.startswith("Invalid command: skip"):
        return False
    game_state = getattr(coordinator, "last_game_state", None)
    if str(getattr(game_state, "room_type", "") or "") != "NeowRoom":
        return False
    coordinator.last_error = None
    if request_state:
        coordinator.send_message("state")
    return True


def _aligned_action_to_dict(raw_action: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
    kind = str(raw_action[0])
    if kind == "event":
        return {"kind": "event", "choice_index": int(raw_action[1]), "name": str(raw_action[2])}
    if kind == "card_reward":
        return {"kind": "card_reward", "name": str(raw_action[1]), "card_id": str(raw_action[1])}
    if kind == "skip":
        return {"kind": "skip", "name": str(raw_action[1])}
    if kind == "map":
        return {"kind": "map", "symbol": str(raw_action[1]), "choice_index": int(raw_action[2]), "x": int(raw_action[2])}
    if kind == "card":
        return {"kind": "card", "name": str(raw_action[1]), "target_index": int(raw_action[2])}
    if kind == "potion":
        target_index = None if raw_action[2] is None else int(raw_action[2])
        return {"kind": "potion", "name": str(raw_action[1]), "potion_id": str(raw_action[1]), "target_index": target_index}
    if kind == "end":
        return {"kind": "end", "name": str(raw_action[1])}
    if kind == "campfire":
        return {"kind": "campfire", "name": str(raw_action[1]), "target_index": raw_action[2] if len(raw_action) > 2 else None}
    if kind == "boss_relic":
        return {"kind": "boss_relic", "name": str(raw_action[1]), "relic_id": str(raw_action[1])}
    if kind == "treasure":
        return {"kind": "treasure", "name": str(raw_action[1])}
    if kind == "chest":
        return {"kind": "chest", "name": str(raw_action[1]), "item_kind": str(raw_action[1]).lower()}
    return {"kind": kind, "raw": list(raw_action)}


def _compact_snapshot_from_verbose_state(state: dict[str, Any], phase: str) -> dict[str, Any]:
    normalized_phase = _normalize_phase_name(phase)
    snapshot: dict[str, Any] = {
        "phase": normalized_phase,
        "floor": state.get("floor"),
        "hp": state.get("current_hp"),
        "gold": state.get("gold"),
        "deck": [card.get("card_id") or card.get("id") or card.get("name") for card in state.get("deck") or []],
    }
    screen_state = state.get("screen_state") or {}
    if normalized_phase in {"EVENT", "NEOW"}:
        snapshot["event_id"] = (
            state.get("event_id")
            or screen_state.get("event_id")
            or screen_state.get("event_name")
            or state.get("event_name")
            or state.get("screen_name")
        )
        if normalized_phase == "NEOW" and not snapshot["event_id"]:
            snapshot["event_id"] = "Neow Event"
    combat_state = state.get("combat_state")
    if snapshot["phase"] == "COMBAT" and combat_state:
        player = combat_state.get("player") or {}
        monsters = []
        for monster in combat_state.get("monsters") or []:
            if monster.get("current_hp", 0) <= 0 or monster.get("is_gone"):
                continue
            monsters.append(
                {
                    "id": monster.get("monster_id") or monster.get("id") or monster.get("name"),
                    "hp": monster.get("current_hp"),
                    "block": monster.get("block"),
                    "intent": monster.get("intent"),
                    "move_adjusted_damage": monster.get("move_adjusted_damage"),
                    "move_hits": monster.get("move_hits"),
                }
            )
        snapshot.update(
            {
                "block": player.get("block"),
                "energy": player.get("energy"),
                "hand": [card.get("card_id") or card.get("id") or card.get("name") for card in combat_state.get("hand") or []],
                "monsters": monsters,
            }
        )
    return snapshot


def load_recorded_trace(path: str | Path) -> RecordedRunTrace:
    trace_path = Path(path)
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if "trace" in payload and "summary" in payload:
        steps = [
            RecordedTraceStep(
                step=int(step["step"]),
                phase=_normalize_phase_name(step["phase"]),
                floor=step.get("floor"),
                action=_aligned_action_to_dict(step["action"]),
                pre=dict(step.get("pre") or {}),
                post=dict(step.get("post") or {}),
                raw_pre_state=None,
            )
            for step in payload["trace"]
        ]
        summary = payload.get("summary") or {}
        return RecordedRunTrace(
            path=trace_path,
            source_format="aligned_trace_v1",
            seed_long=summary.get("seed_long"),
            seed_str=summary.get("seed_str"),
            ascension=int(summary.get("ascension", 0) or 0),
            character=None,
            steps=steps,
        )

    if "steps" not in payload:
        raise ValueError(f"Unsupported trace format: {trace_path}")

    steps: list[RecordedTraceStep] = []
    character = None
    for raw_step in payload["steps"]:
        pre_state = dict(raw_step.get("pre_state") or {})
        post_state = dict(raw_step.get("post_state") or {})
        if character is None:
            character = pre_state.get("character")
        action = dict(raw_step.get("action") or {})
        phase = _normalize_phase_name(raw_step.get("phase"))
        if str(action.get("kind") or "").lower() == "card_select":
            phase = "CARD_SELECT"
        post_phase = _normalize_phase_name(raw_step.get("post_phase") or post_state.get("screen") or phase)
        steps.append(
            RecordedTraceStep(
                step=int(raw_step["step"]),
                phase=phase,
                floor=raw_step.get("floor"),
                action=action,
                pre=_compact_snapshot_from_verbose_state(pre_state, phase),
                post=_compact_snapshot_from_verbose_state(post_state, post_phase),
                raw_pre_state=pre_state,
            )
        )
    for index, step in enumerate(steps[:-1]):
        next_step = steps[index + 1]
        if (
            str(step.action.get("kind") or "").lower() == "card"
            and next_step.phase == "CARD_SELECT"
            and step.post.get("phase") == "COMBAT"
        ):
            step.post["phase"] = "CARD_SELECT"
    return RecordedRunTrace(
        path=trace_path,
        source_format="native_run_trace_v1",
        seed_long=payload.get("seed_long"),
        seed_str=payload.get("seed_str"),
        ascension=int(payload.get("ascension", 0) or 0),
        character=character,
        steps=steps,
    )


def _normalize_live_phase(game_state) -> str:
    if game_state is None:
        return "OUT_OF_GAME"
    screen_type = getattr(game_state, "screen_type", None)
    room_type = str(getattr(game_state, "room_type", "") or "")
    if _is_neow_card_select_state(game_state):
        return "CARD_SELECT"
    if screen_type == ScreenType.EVENT:
        event_id = getattr(game_state.screen, "event_id", "") or getattr(game_state.screen, "event_name", "")
        if "neow" in str(event_id).lower():
            return "NEOW"
        return "EVENT"
    if screen_type == ScreenType.MAP:
        return "MAP"
    if screen_type in {ScreenType.CARD_REWARD, ScreenType.COMBAT_REWARD}:
        return "CARD_REWARD"
    if screen_type == ScreenType.BOSS_REWARD:
        return "BOSS_RELIC"
    if screen_type == ScreenType.REST:
        return "CAMPFIRE"
    if screen_type in {ScreenType.SHOP_ROOM, ScreenType.SHOP_SCREEN}:
        return "SHOP"
    if screen_type == ScreenType.CHEST:
        return "TREASURE"
    if screen_type == ScreenType.GRID or screen_type == ScreenType.HAND_SELECT:
        return "CARD_SELECT"
    if screen_type == ScreenType.GAME_OVER:
        return "GAME_OVER"
    if screen_type == ScreenType.COMPLETE:
        return "COMPLETE"
    if getattr(game_state, "in_combat", False):
        return "COMBAT"
    return str(getattr(screen_type, "name", "UNKNOWN"))


def snapshot_live_state(game_state) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "phase": _normalize_live_phase(game_state),
        "floor": getattr(game_state, "floor", None),
        "hp": getattr(game_state, "current_hp", None),
        "gold": getattr(game_state, "gold", None),
        "deck": [card.card_id for card in getattr(game_state, "deck", [])],
        "potions": [potion.potion_id for potion in getattr(game_state, "potions", [])],
    }
    if getattr(game_state, "screen_type", None) == ScreenType.EVENT:
        screen = getattr(game_state, "screen", None)
        snapshot["event_id"] = (
            getattr(screen, "event_id", None)
            or getattr(screen, "event_name", None)
            or getattr(game_state, "screen_name", None)
        )
    if getattr(game_state, "in_combat", False):
        monsters = []
        for monster in game_state.monsters:
            if monster.current_hp <= 0 or monster.is_gone:
                continue
            monsters.append(
                {
                    "id": monster.monster_id,
                    "hp": monster.current_hp,
                    "block": monster.block,
                    "intent": monster.intent.name,
                    "move_adjusted_damage": getattr(monster, "move_adjusted_damage", None),
                    "move_hits": getattr(monster, "move_hits", None),
                }
            )
        snapshot.update(
            {
                "block": game_state.player.block,
                "energy": game_state.player.energy,
                "hand": [card.card_id for card in game_state.hand],
                "player_powers": [
                    {
                        "id": power.power_id,
                        "amount": power.amount,
                    }
                    for power in getattr(game_state.player, "powers", [])
                ],
                "monsters": monsters,
            }
        )
    return snapshot


def _summarize_live_state_for_timeout(game_state) -> str:
    if game_state is None:
        return "live_state=None"
    screen_type = getattr(game_state, "screen_type", None)
    room_type = str(getattr(game_state, "room_type", "") or "")
    summary_parts = [
        f"room_type={room_type!r}",
        f"screen_type={getattr(screen_type, 'name', screen_type)!r}",
    ]
    screen = getattr(game_state, "screen", None)
    if screen_type == ScreenType.CARD_REWARD and screen is not None:
        cards = [str(getattr(card, "name", "") or getattr(card, "card_id", "")) for card in (getattr(screen, "cards", []) or [])]
        summary_parts.append(f"cards={cards[:5]!r}")
        summary_parts.append(f"skip_available={getattr(screen, 'skip_available', None)!r}")
        summary_parts.append(f"bowl_available={getattr(screen, 'bowl_available', None)!r}")
    elif screen_type == ScreenType.EVENT and screen is not None:
        options = []
        for option in list(getattr(screen, "options", []) or [])[:5]:
            option_text = " ".join(
                str(part)
                for part in [
                    getattr(option, "label", None),
                    getattr(option, "text", None),
                    getattr(option, "name", None),
                ]
                if part
            )
            options.append(option_text)
        summary_parts.append(f"options={options!r}")
        summary_parts.append(f"event_id={getattr(screen, 'event_id', None)!r}")
    return ", ".join(summary_parts)


def _translation_debug_context(game_state) -> dict[str, Any]:
    context: dict[str, Any] = {
        "phase": _normalize_live_phase(game_state),
        "screen_type": getattr(getattr(game_state, "screen_type", None), "name", str(getattr(game_state, "screen_type", None))),
        "proceed_available": getattr(game_state, "proceed_available", None),
        "cancel_available": getattr(game_state, "cancel_available", None),
        "potions": [
            {
                "potion_id": potion.potion_id,
                "name": potion.name,
                "can_use": potion.can_use,
                "requires_target": potion.requires_target,
            }
            for potion in getattr(game_state, "potions", [])
        ],
        "hand": [
            {
                "card_id": card.card_id,
                "name": card.name,
                "uuid": getattr(card, "uuid", None),
            }
            for card in getattr(game_state, "hand", [])
        ],
    }
    screen = getattr(game_state, "screen", None)
    if screen is not None:
        if hasattr(screen, "current_node"):
            current_node = getattr(screen, "current_node", None)
            context["screen_current_node"] = (
                None
                if current_node is None
                else {
                    "x": getattr(current_node, "x", None),
                    "y": getattr(current_node, "y", None),
                }
            )
        if hasattr(screen, "next_nodes"):
            context["screen_next_nodes"] = [
                {
                    "x": getattr(node, "x", None),
                    "y": getattr(node, "y", None),
                }
                for node in getattr(screen, "next_nodes", []) or []
            ]
        if hasattr(screen, "rewards"):
            context["screen_rewards"] = [
                {
                    "reward_type": getattr(getattr(reward, "reward_type", None), "name", str(getattr(reward, "reward_type", None))),
                    "gold": getattr(reward, "gold", None),
                    "relic_id": getattr(getattr(reward, "relic", None), "relic_id", None),
                    "relic_name": getattr(getattr(reward, "relic", None), "name", None),
                    "potion_id": getattr(getattr(reward, "potion", None), "potion_id", None),
                    "potion_name": getattr(getattr(reward, "potion", None), "name", None),
                    "link_relic_id": getattr(getattr(reward, "link", None), "relic_id", None),
                    "link_relic_name": getattr(getattr(reward, "link", None), "name", None),
                }
                for reward in getattr(screen, "rewards", [])
            ]
        if hasattr(screen, "cards"):
            context["screen_cards"] = [
                {
                    "card_id": card.card_id,
                    "name": card.name,
                    "uuid": getattr(card, "uuid", None),
                }
                for card in getattr(screen, "cards", [])
            ]
        if hasattr(screen, "options"):
            context["screen_options"] = [
                {
                    "label": getattr(option, "label", None),
                    "text": getattr(option, "text", None),
                    "name": getattr(option, "name", None),
                }
                for option in getattr(screen, "options", [])
            ]
    return context


def _intent_family(intent: Any) -> str:
    value = str(intent or "").upper()
    if value in {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}:
        return "ATTACK_FAMILY"
    if value in {"BUFF", "DEBUG"}:
        return "BUFF_FAMILY"
    if value in {"DEBUFF", "STRONG_DEBUFF"}:
        return "DEBUFF_FAMILY"
    if value in {"MAGIC", "UNKNOWN"}:
        return "SPECIAL_FAMILY"
    return value


def _normalize_monster_snapshot(monster: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(monster)
    intent_family = _intent_family(monster.get("intent"))
    normalized["intent_family"] = intent_family
    if intent_family != "ATTACK_FAMILY":
        # CommunicationMod often reports non-attacking intents with sentinel
        # move values (-1 damage / 1 hit), while our recorded v2 traces use
        # a neutral 0/0 placeholder. Treat those as the same non-damaging move.
        normalized["move_adjusted_damage"] = 0
        normalized["move_hits"] = 0
    return normalized


def _monster_lists_equivalent(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> bool:
    if len(expected) != len(actual):
        return False
    for expected_monster, actual_monster in zip(expected, actual):
        expected_norm = _normalize_monster_snapshot(expected_monster)
        actual_norm = _normalize_monster_snapshot(actual_monster)
        for key in ("id", "hp", "block", "move_adjusted_damage", "move_hits"):
            if expected_norm.get(key) != actual_norm.get(key):
                return False
        if expected_norm.get("intent") == actual_norm.get("intent"):
            continue
        if expected_norm.get("intent_family") == actual_norm.get("intent_family"):
            continue
        return False
    return True


def compare_snapshots(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            mismatches.append(f"{key}: missing actual value")
            continue
        actual_value = actual[key]
        if (
            key == "monsters"
            and isinstance(expected_value, list)
            and isinstance(actual_value, list)
            and _monster_lists_equivalent(expected_value, actual_value)
        ):
            continue
        if (
            key == "floor"
            and expected_value == 1
            and actual_value == 0
            and str(expected.get("phase") or "").upper() in {"NEOW", "CARD_REWARD", "MAP"}
            and str(actual.get("phase") or "").upper() in {"NEOW", "EVENT", "CARD_REWARD", "MAP"}
        ):
            continue
        if key == "event_id":
            if _normalize_event_id(expected_value) == _normalize_event_id(actual_value):
                continue
        if expected_value != actual_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    return mismatches


def _filter_neow_transition_mismatches(mismatches: list[str]) -> list[str]:
    transition_keys = {
        "phase",
        "screen_type",
        "screen_name",
        "room_type",
        "room_phase",
        "floor",
    }
    filtered: list[str] = []
    for mismatch in mismatches:
        key, _, _rest = mismatch.partition(":")
        if key in transition_keys:
            continue
        filtered.append(mismatch)
    return filtered


def _missing_expected_cards(expected_deck: list[str], live_deck: list[str]) -> list[str]:
    remaining_live_cards = list(live_deck)
    missing_expected_cards: list[str] = []
    for card_id in expected_deck:
        if card_id in remaining_live_cards:
            remaining_live_cards.remove(card_id)
        else:
            missing_expected_cards.append(card_id)
    return missing_expected_cards


def _filter_map_event_transition_mismatches(
    mismatches: list[str],
    *,
    expected_event_id: Any,
    actual_event_id: Any,
) -> list[str]:
    filtered: list[str] = []
    normalized_expected = _normalize_event_id(expected_event_id)
    normalized_actual = _normalize_event_id(actual_event_id)
    for mismatch in mismatches:
        key, _, _rest = mismatch.partition(":")
        if key == "event_id" and not normalized_expected and normalized_actual:
            continue
        filtered.append(mismatch)
    return filtered


def _filter_event_pre_mismatches(
    mismatches: list[str],
    *,
    step_action: dict[str, Any],
    actual_event_id: Any,
) -> list[str]:
    filtered: list[str] = []
    normalized_trace_event = _normalize_event_id(step_action.get("event_id"))
    normalized_actual = _normalize_event_id(actual_event_id)
    for mismatch in mismatches:
        key, _, _rest = mismatch.partition(":")
        if key == "event_id" and normalized_trace_event and normalized_trace_event == normalized_actual:
            continue
        filtered.append(mismatch)
    return filtered


def _find_card_by_trace(cards: list[Card], trace_action: dict[str, Any], *, fallback_index: int | None = None) -> Card | None:
    source_index = trace_action.get("source_index")
    card_index = trace_action.get("card_index")
    preferred_index = fallback_index
    if preferred_index is None:
        if source_index is not None:
            preferred_index = int(source_index)
        elif card_index is not None:
            preferred_index = int(card_index)
    if preferred_index is not None and 0 <= preferred_index < len(cards):
        candidate = cards[preferred_index]
        if (
            (trace_action.get("card_id") and candidate.card_id == trace_action["card_id"])
            or (trace_action.get("name") and candidate.name == trace_action["name"])
            or (trace_action.get("name") and candidate.card_id == trace_action["name"])
        ):
            return candidate
    card_id = trace_action.get("card_id")
    if card_id is not None:
        for card in cards:
            if card.card_id == card_id:
                return card
    name = trace_action.get("name")
    if name is not None:
        for card in cards:
            if card.name == name or card.card_id == name:
                return card
    return None


def _find_deck_card_for_target(game_state, target_index: int | None) -> Card | None:
    if target_index is None:
        return None
    if 0 <= int(target_index) < len(game_state.deck):
        return game_state.deck[int(target_index)]
    return None


def _find_matching_reward(game_state, trace_action: dict[str, Any]):
    screen = game_state.screen
    kind = str(trace_action.get("kind") or "")
    if kind == "card_reward":
        for reward in screen.rewards:
            if reward.reward_type == RewardType.CARD:
                return reward
        return None
    if kind == "reward_gold":
        amount = trace_action.get("amount")
        for reward in screen.rewards:
            if reward.reward_type in {RewardType.GOLD, RewardType.STOLEN_GOLD}:
                if amount is None or reward.gold == amount:
                    return reward
        return None
    if kind == "reward_relic":
        relic_payload = trace_action.get("relic") or {}
        relic_id = relic_payload.get("relic_id") or trace_action.get("relic_id") or trace_action.get("name")
        for reward in screen.rewards:
            if reward.reward_type == RewardType.RELIC and reward.relic is not None:
                if reward.relic.relic_id == relic_id or reward.relic.name == relic_id:
                    return reward
        return None
    if kind == "reward_potion":
        potion_id = trace_action.get("potion_id") or trace_action.get("name")
        normalized_potion_id = _normalize_loose_token(potion_id)
        for reward in screen.rewards:
            if reward.reward_type == RewardType.POTION and reward.potion is not None:
                if (
                    reward.potion.potion_id == potion_id
                    or reward.potion.name == potion_id
                    or _normalize_loose_token(reward.potion.potion_id) == normalized_potion_id
                    or _normalize_loose_token(reward.potion.name) == normalized_potion_id
                ):
                    return reward
        return None
    if kind == "reward_key":
        key_name = str(trace_action.get("key") or trace_action.get("name") or "").upper()
        expected_type = RewardType.SAPPHIRE_KEY if "SAPPHIRE" in key_name else RewardType.EMERALD_KEY
        for reward in screen.rewards:
            if reward.reward_type == expected_type:
                return reward
        return None
    if kind == "chest":
        item_kind = str(trace_action.get("item_kind") or "").lower()
        if "sapphire" in item_kind or "sapphire" in str(trace_action.get("name") or "").lower():
            for reward in screen.rewards:
                if reward.reward_type == RewardType.SAPPHIRE_KEY:
                    return reward
    return None


def _resolve_live_monster_target_index(game_state, step: RecordedTraceStep) -> int | None:
    trace_action = step.action
    raw_pre_state = getattr(step, "raw_pre_state", None) or {}
    trace_monsters = list((((raw_pre_state.get("combat_state") or {}).get("monsters")) or []))

    target_trace_monster: dict[str, Any] | None = None

    raw_target_index = trace_action.get("target_index")
    if raw_target_index is not None:
        try:
            raw_target_index = int(raw_target_index)
        except (TypeError, ValueError):
            raw_target_index = None
    if raw_target_index is not None and trace_monsters:
        for monster in trace_monsters:
            if int(monster.get("monster_index", -9999)) != raw_target_index:
                continue
            if monster.get("current_hp", 0) <= 0 or monster.get("is_gone"):
                break
            target_trace_monster = monster
            break
        if target_trace_monster is None and 0 <= raw_target_index < len(trace_monsters):
            positional_monster = trace_monsters[raw_target_index]
            if positional_monster.get("current_hp", 0) > 0 and not positional_monster.get("is_gone"):
                target_trace_monster = positional_monster

    if target_trace_monster is None:
        model_target_index = trace_action.get("model_target_index")
        if model_target_index is not None:
            try:
                model_target_index = int(model_target_index)
            except (TypeError, ValueError):
                model_target_index = None
        if model_target_index is not None and trace_monsters:
            living_trace_monsters = [
                monster
                for monster in trace_monsters
                if monster.get("current_hp", 0) > 0 and not monster.get("is_gone")
            ]
            if 0 <= model_target_index < len(living_trace_monsters):
                target_trace_monster = living_trace_monsters[model_target_index]

    if target_trace_monster is None and not trace_monsters:
        if raw_target_index is not None:
            monsters = list(getattr(game_state, "monsters", []) or [])
            if 0 <= raw_target_index < len(monsters):
                monster = monsters[raw_target_index]
                if monster.current_hp > 0 and not monster.is_gone:
                    return raw_target_index
        model_target_index = trace_action.get("model_target_index")
        if model_target_index is not None:
            try:
                model_target_index = int(model_target_index)
            except (TypeError, ValueError):
                model_target_index = None
        if model_target_index is not None:
            living_monsters = [
                (live_index, monster)
                for live_index, monster in enumerate(getattr(game_state, "monsters", []) or [])
                if monster.current_hp > 0 and not monster.is_gone
            ]
            if 0 <= model_target_index < len(living_monsters):
                return living_monsters[model_target_index][0]

    if target_trace_monster is None:
        return None

    target_id = (
        target_trace_monster.get("monster_id")
        or target_trace_monster.get("id")
        or target_trace_monster.get("name")
    )
    trace_occurrence = 0
    for monster in trace_monsters:
        if monster.get("current_hp", 0) <= 0 or monster.get("is_gone"):
            continue
        monster_id = monster.get("monster_id") or monster.get("id") or monster.get("name")
        if monster_id != target_id:
            continue
        if monster is target_trace_monster:
            break
        trace_occurrence += 1

    live_occurrence = 0
    for live_index, monster in enumerate(getattr(game_state, "monsters", []) or []):
        if monster.current_hp <= 0 or monster.is_gone:
            continue
        if getattr(monster, "monster_id", None) != target_id:
            continue
        if live_occurrence == trace_occurrence:
            return live_index
        live_occurrence += 1

    return None


def _single_leave_event_action(game_state):
    if getattr(game_state, "screen_type", None) != ScreenType.EVENT:
        return None
    options = getattr(game_state.screen, "options", []) or []
    if len(options) != 1:
        return None
    option = options[0]
    option_text = " ".join(
        str(part)
        for part in [
            getattr(option, "label", None),
            getattr(option, "text", None),
            getattr(option, "name", None),
        ]
        if part
    ).lower()
    if "leave" in option_text:
        return EventOptionAction(option)
    return None


def _single_neow_continue_action(game_state):
    if game_state is None:
        return None
    if _is_neow_card_select_state(game_state):
        return None
    if getattr(game_state, "screen_type", None) != ScreenType.EVENT:
        return None
    if str(getattr(game_state, "room_type", "") or "") != "NeowRoom":
        return None
    options = list(getattr(getattr(game_state, "screen", None), "options", []) or [])
    if len(options) != 1:
        return None
    return NeowContinueAction()


def _is_single_leave_event_state(game_state, step: RecordedTraceStep | None = None) -> bool:
    if game_state is None:
        return False
    leave_action = _single_leave_event_action(game_state)
    if leave_action is None:
        return False
    if step is None:
        return True
    return step.phase in {"CARD_REWARD", "MAP", "EVENT"} or str(step.action.get("kind") or "").lower() == "skip"


def _is_completed_rest_bridge_state(game_state, step: RecordedTraceStep | None = None) -> bool:
    if game_state is None:
        return False
    if getattr(game_state, "screen_type", None) != ScreenType.REST:
        return False
    screen = getattr(game_state, "screen", None)
    if screen is None:
        return False
    if not bool(getattr(screen, "has_rested", False)):
        return False
    if list(getattr(screen, "rest_options", []) or []):
        return False
    if not getattr(game_state, "proceed_available", False):
        return False
    if step is None:
        return True
    return step.phase == "MAP"


def _is_empty_reward_proceed_bridge_state(game_state, step: RecordedTraceStep | None = None) -> bool:
    if game_state is None:
        return False
    if getattr(game_state, "screen_type", None) != ScreenType.COMBAT_REWARD:
        return False
    screen = getattr(game_state, "screen", None)
    rewards = list(getattr(screen, "rewards", []) or []) if screen is not None else []
    if rewards:
        return False
    if not getattr(game_state, "proceed_available", False):
        return False
    if step is None:
        return True
    return step.phase in {"CARD_REWARD", "MAP"} or str(step.action.get("kind") or "").lower() == "skip"


def _is_chest_leave_bridge_state(game_state, step: RecordedTraceStep | None = None) -> bool:
    if game_state is None:
        return False
    snapshot = snapshot_live_state(game_state)
    if snapshot.get("phase") != "CHEST":
        return False
    if step is None:
        return True
    return step.phase == "MAP"


def _combat_opening_still_resolving(snapshot: dict[str, Any]) -> bool:
    if snapshot.get("phase") != "COMBAT":
        return False
    monsters = list(snapshot.get("monsters") or [])
    if not monsters:
        return False
    for monster in monsters:
        intent = monster.get("intent")
        damage = monster.get("move_adjusted_damage")
        if intent in {"DEBUG", "UNKNOWN"}:
            return True
        if _intent_family(intent) == "ATTACK_FAMILY" and damage in {-1, None}:
            return True
    return False


def _is_neow_leave_bridge_state(game_state, step: RecordedTraceStep | None = None) -> bool:
    if game_state is None:
        return False
    if _single_neow_continue_action(game_state) is None:
        return False
    if step is None:
        return True
    step_phase = step.phase
    step_kind = str(step.action.get("kind") or "").lower()
    return (
        (step_phase == "CARD_REWARD" and step_kind == "skip")
        or step_phase == "MAP"
    )


def _is_neow_map_bridge_state(game_state, step: RecordedTraceStep | None = None) -> bool:
    if game_state is None:
        return False
    if str(getattr(game_state, "room_type", "") or "") != "NeowRoom":
        return False
    if _is_neow_leave_bridge_state(game_state, step):
        return True
    return getattr(game_state, "screen_type", None) == ScreenType.MAP


def _is_neow_stale_card_reward_bridge_state(game_state, step: RecordedTraceStep | None = None) -> bool:
    if game_state is None:
        return False
    if str(getattr(game_state, "room_type", "") or "") != "NeowRoom":
        return False
    if getattr(game_state, "screen_type", None) != ScreenType.CARD_REWARD:
        return False
    if step is None:
        return True
    return step.phase == "NEOW"


def _is_neow_card_select_state(game_state) -> bool:
    if game_state is None:
        return False
    if str(getattr(game_state, "room_type", "") or "") != "NeowRoom":
        return False
    screen_type = getattr(game_state, "screen_type", None)
    if screen_type in {ScreenType.CARD_REWARD, ScreenType.COMBAT_REWARD, ScreenType.BOSS_REWARD}:
        return False
    screen = getattr(game_state, "screen", None)
    if screen is None:
        return False
    cards = list(getattr(screen, "cards", []) or [])
    if not cards:
        return False
    if list(getattr(screen, "rewards", []) or []):
        return False
    options = list(getattr(screen, "options", []) or [])
    if len(options) == 1:
        option_text = " ".join(
            str(part)
            for part in [
                getattr(options[0], "label", None),
                getattr(options[0], "text", None),
                getattr(options[0], "name", None),
            ]
            if part
        ).lower()
        if any(token in option_text for token in ("leave", "continue", "proceed", "talk")):
            return False
    return True


def _trace_rest_option(name: str) -> RestOption:
    return RestOption[name.upper()]


def _choose_shop_card(game_state, trace_action: dict[str, Any]) -> Action | None:
    item_kind = str(trace_action.get("item_kind") or "")
    if item_kind == "leave":
        return CancelAction()
    if item_kind == "purge":
        return ChooseAction(name="purge")
    if item_kind == "card":
        card_payload = trace_action.get("card") or {}
        match = _find_card_by_trace(game_state.screen.cards, {**trace_action, **card_payload})
        if match is not None:
            from spirecomm.communication.action import BuyCardAction

            return BuyCardAction(match)
    if item_kind == "relic":
        relic_payload = trace_action.get("relic") or {}
        relic_id = relic_payload.get("relic_id") or trace_action.get("relic_id") or trace_action.get("name")
        for relic in game_state.screen.relics:
            if relic.relic_id == relic_id or relic.name == relic_id:
                from spirecomm.communication.action import BuyRelicAction

                return BuyRelicAction(relic)
    if item_kind == "potion":
        potion_id = trace_action.get("potion_id") or trace_action.get("name")
        normalized_potion_id = _normalize_loose_token(potion_id)
        for potion in game_state.screen.potions:
            if (
                potion.potion_id == potion_id
                or potion.name == potion_id
                or _normalize_loose_token(potion.potion_id) == normalized_potion_id
                or _normalize_loose_token(potion.name) == normalized_potion_id
            ):
                from spirecomm.communication.action import BuyPotionAction

                return BuyPotionAction(potion)
    return None


def _next_trace_action_for_game(game_state, step: RecordedTraceStep):
    trace_action = step.action
    kind = str(trace_action.get("kind") or "")
    phase = step.phase
    screen_type = game_state.screen_type

    # Setup transitions that v2 folds into a single phase.
    if phase == "SHOP" and screen_type == ScreenType.SHOP_ROOM and str(trace_action.get("item_kind") or "").lower() != "leave":
        return ChooseShopkeeperAction()

    if screen_type == ScreenType.COMBAT_REWARD:
        reward = _find_matching_reward(game_state, trace_action)
        if reward is not None:
            return CombatRewardAction(reward)
        if kind == "skip":
            return ProceedAction()

    if phase == "NEOW" and screen_type == ScreenType.EVENT and not _is_neow_card_select_state(game_state):
        single_continue_action = _single_neow_continue_action(game_state)
        if single_continue_action is not None:
            return single_continue_action
        choice_index = trace_action.get("choice_index", 0)
        return ChooseAction(choice_index=int(choice_index))

    if phase == "EVENT" and screen_type == ScreenType.EVENT:
        choice_index = trace_action.get("choice_index", 0)
        options = list(getattr(game_state.screen, "options", []) or [])
        if 0 <= int(choice_index) < len(options):
            option = options[int(choice_index)]
            return EventOptionAction(option)
        return ChooseAction(choice_index=int(choice_index))

    if kind == "skip":
        leave_action = _single_leave_event_action(game_state)
        if leave_action is not None:
            return leave_action

    if phase == "MAP" and screen_type == ScreenType.MAP:
        if str(trace_action.get("name") or "").upper() == "BOSS":
            return ChooseMapBossAction()
        x = trace_action.get("x")
        choice_index = trace_action.get("choice_index")
        for node in game_state.screen.next_nodes:
            if x is not None and node.x == int(x):
                return ChooseMapNodeAction(node)
        if choice_index is not None and 0 <= int(choice_index) < len(game_state.screen.next_nodes):
            return ChooseMapNodeAction(game_state.screen.next_nodes[int(choice_index)])

    if phase == "COMBAT" and game_state.in_combat:
        if kind == "end":
            return EndTurnAction()
        if kind == "card":
            card = _find_card_by_trace(game_state.hand, trace_action)
            if card is None:
                return None
            target_index = _resolve_live_monster_target_index(game_state, step)
            if target_index is not None:
                return PlayCardAction(card=card, target_index=int(target_index))
            return PlayCardAction(card=card)
        if kind == "potion":
            potion_id = trace_action.get("potion_id") or trace_action.get("name")
            normalized_potion_id = _normalize_loose_token(potion_id)
            potion_index = None
            for index, potion in enumerate(game_state.potions):
                if (
                    potion.potion_id == potion_id
                    or potion.name == potion_id
                    or _normalize_loose_token(potion.potion_id) == normalized_potion_id
                    or _normalize_loose_token(potion.name) == normalized_potion_id
                ):
                    potion_index = index
                    break
            if potion_index is None:
                return None
            target_index = _resolve_live_monster_target_index(game_state, step)
            if target_index is not None:
                return PotionAction(use=True, potion_index=potion_index, target_index=int(target_index))
            return PotionAction(use=True, potion_index=potion_index)
        if kind == "card_select" and screen_type in {ScreenType.GRID, ScreenType.HAND_SELECT}:
            options = game_state.screen.cards
            if "choice_index" in trace_action and 0 <= int(trace_action["choice_index"]) < len(options):
                from spirecomm.communication.action import CardSelectAction

                return CardSelectAction([options[int(trace_action["choice_index"])]])
            card = _find_card_by_trace(
                options,
                trace_action,
                fallback_index=trace_action.get("select_index"),
            )
            if card is None:
                return None
            from spirecomm.communication.action import CardSelectAction

            return CardSelectAction([card])

    if phase == "CARD_REWARD":
        if screen_type == ScreenType.CARD_REWARD:
            if kind == "card_reward":
                choice_index = trace_action.get("choice_index")
                if (
                    str(getattr(game_state, "room_type", "") or "") == "NeowRoom"
                    and choice_index is not None
                ):
                    try:
                        choice_index = int(choice_index)
                    except (TypeError, ValueError):
                        choice_index = None
                    if choice_index is not None and 0 <= choice_index < len(game_state.screen.cards):
                        card = game_state.screen.cards[choice_index]
                        card_name = trace_action.get("name") or getattr(card, "name", None)
                        return NeowCardRewardAction(choice_index=choice_index, name=card_name)
                card_payload = trace_action.get("card") or {}
                card = _find_card_by_trace(
                    game_state.screen.cards,
                    {**trace_action, **card_payload},
                    fallback_index=trace_action.get("card_index"),
                )
                if card is not None:
                    return CardRewardAction(card)
                if choice_index is not None:
                    try:
                        choice_index = int(choice_index)
                    except (TypeError, ValueError):
                        choice_index = None
                if choice_index is not None and 0 <= choice_index < len(game_state.screen.cards):
                    return ChooseAction(choice_index=choice_index)
            if kind == "skip":
                if str(trace_action.get("name") or "").upper() == "BOWL":
                    return CardRewardAction(bowl=True)
                return CancelAction()

    if phase == "BOSS_RELIC" and screen_type == ScreenType.BOSS_REWARD:
        relic_id = trace_action.get("relic_id") or trace_action.get("name")
        if str(relic_id).upper() == "SKIP":
            return CancelAction()
        for relic in game_state.screen.relics:
            if relic.relic_id == relic_id or relic.name == relic_id:
                from spirecomm.communication.action import BossRewardAction

                return BossRewardAction(relic)

    if phase == "SHOP":
        if screen_type == ScreenType.SHOP_SCREEN:
            if kind == "shop":
                return _choose_shop_card(game_state, trace_action)
            if screen_type == ScreenType.GRID:
                pass
        if screen_type == ScreenType.SHOP_ROOM:
            if kind == "shop" and str(trace_action.get("item_kind") or "").lower() == "leave":
                return ProceedAction()
        if screen_type == ScreenType.GRID:
            target_card = _find_deck_card_for_target(game_state, trace_action.get("target_index"))
            if target_card is None:
                return None
            chosen_card = _find_card_by_trace(game_state.screen.cards, {"card_id": target_card.card_id, "name": target_card.name, "upgrades": target_card.upgrades})
            if chosen_card is None:
                return None
            from spirecomm.communication.action import CardSelectAction

            return CardSelectAction([chosen_card])

    if phase == "CAMPFIRE":
        if screen_type == ScreenType.REST:
            choice_index = trace_action.get("choice_index")
            if choice_index is not None:
                try:
                    return ChooseAction(choice_index=int(choice_index))
                except (TypeError, ValueError):
                    pass
            return RestAction(_trace_rest_option(str(trace_action.get("name") or "REST")))
        if screen_type == ScreenType.GRID:
            target_card = _find_deck_card_for_target(game_state, trace_action.get("target_index"))
            if target_card is None:
                return None
            chosen_card = _find_card_by_trace(game_state.screen.cards, {"card_id": target_card.card_id, "name": target_card.name, "upgrades": target_card.upgrades})
            if chosen_card is None:
                return None
            from spirecomm.communication.action import CardSelectAction

            return CardSelectAction([chosen_card])

    if phase == "CARD_SELECT" and (
        screen_type in {ScreenType.GRID, ScreenType.HAND_SELECT}
        or _is_neow_card_select_state(game_state)
    ):
        options = game_state.screen.cards
        if _is_neow_card_select_state(game_state):
            def _unique_neow_card_select_name(card_obj):
                candidate_name = getattr(card_obj, "name", None) or trace_action.get("name") or trace_action.get("card_id")
                normalized_candidate = _normalize_loose_token(candidate_name)
                if not normalized_candidate:
                    return None
                normalized_names = [
                    _normalize_loose_token(getattr(option, "name", None) or getattr(option, "card_id", None))
                    for option in options
                ]
                if normalized_names.count(normalized_candidate) == 1:
                    return candidate_name
                return None

            if "choice_index" in trace_action and 0 <= int(trace_action["choice_index"]) < len(options):
                selected_card = options[int(trace_action["choice_index"])]
                return NeowCardSelectAction(
                    choice_index=int(trace_action["choice_index"]),
                    name=_unique_neow_card_select_name(selected_card),
                )
            card = _find_card_by_trace(options, trace_action, fallback_index=trace_action.get("target_index"))
            if card is None:
                return None
            try:
                choice_index = options.index(card)
            except ValueError:
                return None
            return NeowCardSelectAction(
                choice_index=choice_index,
                name=_unique_neow_card_select_name(card),
            )
        if "choice_index" in trace_action and 0 <= int(trace_action["choice_index"]) < len(options):
            from spirecomm.communication.action import CardSelectAction

            return CardSelectAction([options[int(trace_action["choice_index"])]])
        card = _find_card_by_trace(options, trace_action, fallback_index=trace_action.get("index"))
        if card is None:
            return None
        from spirecomm.communication.action import CardSelectAction

        return CardSelectAction([card])

    if phase == "TREASURE" and screen_type == ScreenType.CHEST:
        return OpenChestAction()

    if phase == "CHEST" and screen_type == ScreenType.COMBAT_REWARD:
        reward = _find_matching_reward(game_state, trace_action)
        if reward is not None:
            return CombatRewardAction(reward)

    if kind == "skip":
        if getattr(game_state, "proceed_available", False):
            return ProceedAction()
        if getattr(game_state, "cancel_available", False):
            return CancelAction()

    return None


def _describe_action(action) -> dict[str, Any]:
    payload = {"action_class": action.__class__.__name__}
    for attribute in (
        "command",
        "choice_index",
        "name",
        "card_index",
        "target_index",
        "potion_index",
        "continue_choice_index",
        "post_continue_choice_index",
    ):
        if hasattr(action, attribute):
            payload[attribute] = getattr(action, attribute)
    if hasattr(action, "card") and getattr(action, "card", None) is not None:
        payload["card"] = {"card_id": action.card.card_id, "name": action.card.name, "uuid": action.card.uuid}
    if hasattr(action, "target_monster") and getattr(action, "target_monster", None) is not None:
        payload["target_monster"] = {
            "monster_index": action.target_monster.monster_index,
            "name": action.target_monster.name,
        }
    if hasattr(action, "combat_reward") and getattr(action, "combat_reward", None) is not None:
        reward = action.combat_reward
        payload["combat_reward"] = {
            "reward_type": reward.reward_type.name,
            "gold": reward.gold,
            "relic_id": reward.relic.relic_id if reward.relic is not None else None,
            "potion_id": reward.potion.potion_id if reward.potion is not None else None,
        }
    return payload


def _wait_for_ready_state_with_timeout(
    coordinator: Coordinator,
    *,
    timeout_seconds: float,
    context: str,
    initial_probe_delay_seconds: float = 0.0,
) -> None:
    def _ready_enough() -> bool:
        if coordinator.game_is_ready and coordinator.last_error is None:
            return True
        game_state = getattr(coordinator, "last_game_state", None)
        return (
            game_state is not None
            and getattr(game_state, "screen_type", None) == ScreenType.MAP
            and str(getattr(game_state, "room_type", "") or "") == "NeowRoom"
            and len(getattr(getattr(game_state, "screen", None), "next_nodes", []) or []) > 0
        )

    deadline = time.time() + timeout_seconds
    probe_interval = min(1.0, max(0.25, timeout_seconds / 4.0))
    next_state_probe_at = time.time() + max(initial_probe_delay_seconds, probe_interval)
    probe_count = 0
    while time.time() < deadline:
        if _ready_enough():
            return
        received = coordinator.receive_game_state_update(block=False, perform_callbacks=False)
        if received:
            if coordinator.last_error is not None:
                if _clear_tolerable_neow_skip_error(coordinator, request_state=True):
                    next_state_probe_at = time.time() + 0.05
                    continue
                raise RuntimeError(f"{context}: Communication Mod error: {coordinator.last_error}")
            if _ready_enough():
                return
            next_state_probe_at = time.time() + probe_interval
        elif time.time() >= next_state_probe_at:
            # `wait N` in Communication Mod arms GameStateListener's internal
            # timeout and lets it emit a fresh snapshot on a later update.
            # Sending `state` immediately afterward short-circuits that
            # mechanism, so Neow transitions should stay on pure wait pulses.
            last_game_state = getattr(coordinator, "last_game_state", None)
            room_type = str(getattr(last_game_state, "room_type", "") or "")
            if _is_neow_stale_card_reward_bridge_state(last_game_state):
                probe_command = None
            elif room_type == "NeowRoom":
                probe_command = "wait 1"
            else:
                probe_command = (
                    "wait 1"
                    if getattr(coordinator, "in_game", False) or last_game_state is not None
                    else "state"
                )
            if probe_command is not None:
                print(f"[replay-probe] {context}: sending {probe_command!r}", file=sys.stderr, flush=True)
                _send_probe_command(coordinator, probe_command)
                probe_count += 1
            next_state_probe_at = time.time() + probe_interval
        time.sleep(0.05)
    raise TimeoutError(f"{context}: timed out waiting for command-ready state after {timeout_seconds:.1f}s")


def _wait_for_phase_with_timeout(
    coordinator: Coordinator,
    *,
    expected_phase: str,
    timeout_seconds: float,
    context: str,
    initial_probe_delay_seconds: float = 0.0,
) -> None:
    deadline = time.time() + timeout_seconds
    probe_interval = min(1.0, max(0.25, timeout_seconds / 4.0))
    next_state_probe_at = time.time() + max(initial_probe_delay_seconds, probe_interval)
    normalized_expected_phase = _normalize_phase_name(expected_phase)
    probe_count = 0
    neow_card_reward_wait_probes = 0
    neow_card_reward_close_sent = False
    neow_card_reward_force_state_after_close = False
    probe_history: list[str] = []

    def _is_neow_card_reward_continue_probe_state(game_state) -> bool:
        if game_state is None:
            return False
        return (
            normalized_expected_phase == "NEOW"
            and str(getattr(game_state, "room_type", "") or "") == "NeowRoom"
            and getattr(game_state, "screen_type", None) == ScreenType.CARD_REWARD
        )

    def _phase_ready_enough(game_state) -> bool:
        if game_state is None:
            return False
        live_phase = _normalize_live_phase(game_state)
        if normalized_expected_phase == "MAP" and _is_neow_map_bridge_state(game_state):
            return True
        if live_phase != normalized_expected_phase:
            return False
        if coordinator.game_is_ready and coordinator.last_error is None:
            return True
        return (
            normalized_expected_phase == "MAP"
            and getattr(game_state, "screen_type", None) == ScreenType.MAP
            and str(getattr(game_state, "room_type", "") or "") == "NeowRoom"
        )

    while time.time() < deadline:
        last_game_state = getattr(coordinator, "last_game_state", None)
        if _phase_ready_enough(last_game_state):
            return
        received = coordinator.receive_game_state_update(block=False, perform_callbacks=False)
        if received:
            if coordinator.last_error is not None:
                if _clear_tolerable_neow_skip_error(coordinator, request_state=True):
                    next_state_probe_at = time.time() + 0.05
                    continue
                raise RuntimeError(f"{context}: Communication Mod error: {coordinator.last_error}")
            last_game_state = getattr(coordinator, "last_game_state", None)
            live_phase = _normalize_live_phase(last_game_state)
            print(
                f"[replay-phase] {context}: received phase={live_phase!r} ready={coordinator.game_is_ready}",
                file=sys.stderr,
                flush=True,
            )
            if _phase_ready_enough(last_game_state):
                return
            active_probe_interval = 0.25 if _is_neow_card_reward_continue_probe_state(last_game_state) else probe_interval
            next_state_probe_at = time.time() + active_probe_interval
        elif time.time() >= next_state_probe_at:
            last_game_state = getattr(coordinator, "last_game_state", None)
            room_type = str(getattr(last_game_state, "room_type", "") or "")
            if _is_neow_card_reward_continue_probe_state(last_game_state):
                # `wait 1` arms the listener timeout and lets the game advance
                # one more update toward the post-pick Neow frame. Do not mix
                # in immediate `state` probes here; those can snapshot the
                # stale CARD_REWARD frame before the listener's timeout fires.
                probe_command = "wait 1"
                neow_card_reward_wait_probes += 1
            elif room_type == "NeowRoom":
                probe_command = "wait 1"
            else:
                probe_command = "state"
            print(
                f"[replay-probe] {context}: sending {probe_command!r} while waiting for phase={normalized_expected_phase}",
                file=sys.stderr,
                flush=True,
            )
            _send_probe_command(coordinator, probe_command)
            probe_history.append(probe_command)
            probe_count += 1
            active_probe_interval = 0.25 if _is_neow_card_reward_continue_probe_state(last_game_state) else probe_interval
            next_state_probe_at = time.time() + active_probe_interval
        time.sleep(0.05)
    live_phase = _normalize_live_phase(getattr(coordinator, "last_game_state", None))
    live_summary = _summarize_live_state_for_timeout(getattr(coordinator, "last_game_state", None))
    recent_probes = probe_history[-6:]
    raise TimeoutError(
        f"{context}: timed out waiting for phase={normalized_expected_phase} after {timeout_seconds:.1f}s "
        f"(last phase={live_phase!r}; {live_summary}; probes={recent_probes!r}; "
        f"neow_card_reward_wait_probes={neow_card_reward_wait_probes}; "
        f"neow_card_reward_close_sent={neow_card_reward_close_sent})"
    )


def _wait_for_matching_snapshot_with_timeout(
    coordinator: Coordinator,
    *,
    expected_snapshot: dict[str, Any],
    timeout_seconds: float,
    context: str,
    initial_probe_delay_seconds: float = 0.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    probe_interval = min(1.0, max(0.25, timeout_seconds / 4.0))
    next_state_probe_at = time.time() + max(initial_probe_delay_seconds, probe_interval)
    last_snapshot = snapshot_live_state(getattr(coordinator, "last_game_state", None))
    while time.time() < deadline:
        if last_snapshot and not compare_snapshots(expected_snapshot, last_snapshot):
            return last_snapshot
        received = coordinator.receive_game_state_update(block=False, perform_callbacks=False)
        if received:
            if coordinator.last_error is not None:
                if _clear_tolerable_neow_skip_error(coordinator, request_state=True):
                    next_state_probe_at = time.time() + 0.05
                    continue
                raise RuntimeError(f"{context}: Communication Mod error: {coordinator.last_error}")
            last_snapshot = snapshot_live_state(getattr(coordinator, "last_game_state", None))
            if not compare_snapshots(expected_snapshot, last_snapshot):
                return last_snapshot
            next_state_probe_at = time.time() + probe_interval
        elif time.time() >= next_state_probe_at:
            probe_command = (
                "wait 1"
                if getattr(coordinator, "in_game", False) or getattr(coordinator, "last_game_state", None) is not None
                else "state"
            )
            print(
                f"[replay-probe] {context}: sending {probe_command!r} while waiting for matching snapshot",
                file=sys.stderr,
                flush=True,
            )
            _send_probe_command(coordinator, probe_command)
            next_state_probe_at = time.time() + probe_interval
        time.sleep(0.05)
    return last_snapshot


def _wait_for_output_queue_drain(coordinator: Coordinator, *, timeout_seconds: float = 1.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if coordinator.output_queue.empty():
            return
        time.sleep(0.01)


def _send_probe_command(coordinator: Coordinator, command: str) -> None:
    send_immediate = getattr(coordinator, "send_message_immediate", None)
    if callable(send_immediate):
        send_immediate(command)
        return
    coordinator.send_message(command)


def _send_direct_command(command: str, coordinator: Coordinator) -> None:
    # Use the coordinator's normal command path here instead of the immediate
    # stdout shortcut. The replay-specific non-ready bridge actions still need
    # to reach Communication Mod through the same serialized writer path as
    # successful in-game actions like Neow `choose 2`.
    coordinator.send_message(command)
    _wait_for_output_queue_drain(coordinator)


def _execute_action_queue(
    coordinator: Coordinator,
    action,
    *,
    timeout_seconds: float,
    context: str,
) -> list[dict[str, Any]]:
    executed: list[dict[str, Any]] = []
    coordinator.add_action_to_queue(action)
    while coordinator.action_queue:
        next_action = coordinator.action_queue[0]
        if not next_action.can_be_executed(coordinator):
            _wait_for_ready_state_with_timeout(
                coordinator,
                timeout_seconds=timeout_seconds,
                context=f"{context} while waiting to execute queued action",
            )
            continue
        executed.append(_describe_action(next_action))
        coordinator.execute_next_action()
        # Some actions (e.g. CardSelectAction) only queue follow-up actions and do not send a command.
        if coordinator.game_is_ready and coordinator.last_error is None:
            continue
        _wait_for_ready_state_with_timeout(
            coordinator,
            timeout_seconds=timeout_seconds,
            context=f"{context} after executing {executed[-1].get('action_class')}",
        )
    return executed


def _execute_single_action_without_wait(
    coordinator: Coordinator,
    action,
    *,
    timeout_seconds: float,
    context: str,
) -> list[dict[str, Any]]:
    coordinator.add_action_to_queue(action)
    next_action = coordinator.action_queue[0]
    if not next_action.can_be_executed(coordinator):
        _wait_for_ready_state_with_timeout(
            coordinator,
            timeout_seconds=timeout_seconds,
            context=f"{context} while waiting to execute queued action",
        )
        next_action = coordinator.action_queue[0]
    executed = [_describe_action(next_action)]
    coordinator.execute_next_action()
    return executed


def _infer_character(trace: RecordedRunTrace, override: str | None) -> PlayerClass:
    if override:
        return PlayerClass[override]
    if trace.character:
        return PlayerClass[trace.character]
    return PlayerClass.IRONCLAD


class _StateDrivenReplayDriver:
    def __init__(
        self,
        *,
        trace: RecordedRunTrace,
        compare_state: bool,
        stop_on_mismatch: bool,
        total_steps: int,
        progress_enabled: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.trace = trace
        self.compare_state = compare_state
        self.stop_on_mismatch = stop_on_mismatch
        self.total_steps = total_steps
        self.progress_enabled = progress_enabled
        self.progress_callback = progress_callback
        self.results: list[dict[str, Any]] = []
        self.step_index = 0
        self.done = False
        self.failed = False
        self.first_failure_step: int | None = None
        self.pending_step: RecordedTraceStep | None = None
        self.pending_actual_pre: dict[str, Any] | None = None
        self.pending_pre_mismatches: list[str] = []
        self.pending_actions: list[dict[str, Any]] = []
        self.pending_setup_actions: list[dict[str, Any]] = []
        self.pending_comparison_note: str | None = None
        self.neow_bridge_step_number: int | None = None
        self.neow_card_reward_close_bridge_step_number: int | None = None
        self.neow_card_reward_blind_continue_step_number: int | None = None
        self.neow_card_reward_blind_continue_probe_step_number: int | None = None
        self.neow_card_reward_blind_continue_choose_step_number: int | None = None
        self.neow_card_reward_blind_map_step_number: int | None = None
        self.neow_card_select_continue_step_number: int | None = None
        self.neow_post_card_select_settle_probe_step_number: int | None = None
        self.neow_map_ready_pulse_step_number: int | None = None
        self.neow_leave_map_continue_step_number: int | None = None
        self.neow_post_map_settle_probe_step_number: int | None = None
        self.event_leave_bridge_step_number: int | None = None
        self.event_leave_map_continue_step_number: int | None = None
        self.campfire_leave_bridge_step_number: int | None = None
        self.reward_proceed_bridge_step_number: int | None = None
        self._last_announced_step: int | None = None

    def _emit_progress(
        self,
        *,
        status: str,
        step: RecordedTraceStep | None = None,
        live_snapshot: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.progress_callback is None:
            return
        payload: dict[str, Any] = {
            "status": status,
            "step_index": self.step_index,
            "steps_total": self.total_steps,
            "steps_replayed": len(self.results),
            "failed": self.failed,
            "done": self.done,
            "first_failure_step": self.first_failure_step,
        }
        if step is not None:
            payload.update(
                {
                    "current_step": step.step,
                    "current_phase": step.phase,
                    "current_action_kind": step.action.get("kind"),
                    "current_action_name": step.action.get("name"),
                }
            )
        if live_snapshot is not None:
            payload.update(
                {
                    "live_phase": live_snapshot.get("phase"),
                    "live_floor": live_snapshot.get("floor"),
                    "live_hp": live_snapshot.get("hp"),
                    "live_gold": live_snapshot.get("gold"),
                }
            )
        if extra:
            payload.update(extra)
        self.progress_callback(payload)

    def _update_post_neow_combat_settle_probe(self, comparison_note: str) -> None:
        if comparison_note in {
            "neow_leave_map_continue_callback_timeout_forced",
            "neow_leave_map_continue_setup_callback_timeout_forced",
            "neow_post_map_settle_callback_timeout_forced",
        } and self.step_index < self.total_steps and self.trace.steps[self.step_index].phase in {
            "CARD_SELECT",
            "COMBAT",
            "CARD_REWARD",
            "MAP",
            "SHOP",
            "EVENT",
            "CAMPFIRE",
            "TREASURE",
            "CHEST",
            "BOSS_RELIC",
        }:
            self.neow_post_map_settle_probe_step_number = self.trace.steps[self.step_index].step
        else:
            self.neow_post_map_settle_probe_step_number = None

    def _update_post_neow_card_select_settle_probe(self, comparison_note: str) -> None:
        if comparison_note in {
            "neow_card_select_callback_timeout_forced",
            "neow_card_select_continue_callback_timeout_forced",
            "neow_post_card_select_settle_callback_timeout_forced",
        } and self.step_index < self.total_steps and self.trace.steps[self.step_index].phase in {
            "MAP",
            "COMBAT",
            "EVENT",
            "CARD_REWARD",
            "SHOP",
            "CAMPFIRE",
            "TREASURE",
            "CHEST",
            "BOSS_RELIC",
        }:
            self.neow_post_card_select_settle_probe_step_number = self.trace.steps[self.step_index].step
        else:
            self.neow_post_card_select_settle_probe_step_number = None

    def _force_finalize_pending_step(
        self,
        *,
        live_snapshot: dict[str, Any],
        comparison_note: str,
    ) -> None:
        assert self.pending_step is not None
        step_result = {
            "step": self.pending_step.step,
            "phase": self.pending_step.phase,
            "trace_action": self.pending_step.action,
            "setup_actions": list(self.pending_setup_actions),
            "executed_actions": list(self.pending_actions),
            "expected_pre": self.pending_step.pre,
            "actual_pre": self.pending_actual_pre,
            "pre_mismatches": list(self.pending_pre_mismatches),
            "expected_post": self.pending_step.post,
            "actual_post": live_snapshot,
            "post_mismatches": [],
            "comparison_note": comparison_note,
        }
        self.results.append(step_result)
        self._emit_progress(
            status="step_finalized",
            step=self.pending_step,
            live_snapshot=live_snapshot,
            extra={"comparison_note": comparison_note},
        )
        self.pending_step = None
        self.pending_actual_pre = None
        self.pending_pre_mismatches = []
        self.pending_actions = []
        self.pending_setup_actions = []
        self.pending_comparison_note = None
        self.neow_map_ready_pulse_step_number = None
        self.neow_leave_map_continue_step_number = None
        self.neow_card_select_continue_step_number = None
        self.event_leave_map_continue_step_number = None
        self.step_index += 1
        self._update_post_neow_combat_settle_probe(comparison_note)
        self._update_post_neow_card_select_settle_probe(comparison_note)
        if self.step_index >= self.total_steps:
            self.done = True

    def _force_finalize_current_step(
        self,
        step: RecordedTraceStep,
        *,
        live_snapshot: dict[str, Any],
        comparison_note: str,
    ) -> None:
        step_result = {
            "step": step.step,
            "phase": step.phase,
            "trace_action": step.action,
            "setup_actions": list(self.pending_setup_actions),
            "executed_actions": [],
            "expected_pre": step.pre,
            "actual_pre": self.pending_actual_pre,
            "pre_mismatches": list(self.pending_pre_mismatches),
            "expected_post": step.post,
            "actual_post": live_snapshot,
            "post_mismatches": [],
            "comparison_note": comparison_note,
        }
        self.results.append(step_result)
        self._emit_progress(
            status="step_finalized",
            step=step,
            live_snapshot=live_snapshot,
            extra={"comparison_note": comparison_note},
        )
        self.pending_step = None
        self.pending_actual_pre = None
        self.pending_pre_mismatches = []
        self.pending_actions = []
        self.pending_setup_actions = []
        self.pending_comparison_note = None
        self.neow_map_ready_pulse_step_number = None
        self.neow_leave_map_continue_step_number = None
        self.neow_card_select_continue_step_number = None
        self.event_leave_map_continue_step_number = None
        self.step_index += 1
        self._update_post_neow_combat_settle_probe(comparison_note)
        self._update_post_neow_card_select_settle_probe(comparison_note)
        if self.step_index >= self.total_steps:
            self.done = True

    def _maybe_pending_followup_action(self, game_state):
        if self.pending_step is None:
            return None
        trace_action = self.pending_step.action
        if (
            self.pending_step.phase == "CARD_SELECT"
            and str(trace_action.get("kind") or "").lower() == "card_select"
            and str(getattr(game_state, "room_type", "") or "") == "NeowRoom"
            and getattr(game_state, "screen_type", None) == ScreenType.GRID
            and bool(getattr(getattr(game_state, "screen", None), "confirm_up", False))
        ):
            return None
        if (
            self.pending_step.phase == "CAMPFIRE"
            and "target_index" in trace_action
            and getattr(game_state, "screen_type", None) == ScreenType.GRID
            and not any(action.get("action_class") == "CardSelectAction" for action in self.pending_actions)
        ):
            target_card = _find_deck_card_for_target(game_state, trace_action.get("target_index"))
            if target_card is None:
                return None
            chosen_card = _find_card_by_trace(
                game_state.screen.cards,
                {
                    "card_id": target_card.card_id,
                    "name": target_card.name,
                    "upgrades": target_card.upgrades,
                },
            )
            if chosen_card is None:
                return None
            from spirecomm.communication.action import CardSelectAction

            return CardSelectAction([chosen_card])
        return None

    def _maybe_setup_action(self, game_state, step: RecordedTraceStep):
        if (
            step.phase == "SHOP"
            and str(step.action.get("item_kind") or "").lower() == "leave"
            and getattr(game_state, "screen_type", None) == ScreenType.SHOP_SCREEN
        ):
            return CancelAction()
        if (
            step.phase == "SHOP"
            and str(step.action.get("item_kind") or "").lower() != "leave"
            and getattr(game_state, "screen_type", None) == ScreenType.SHOP_ROOM
        ):
            return ChooseShopkeeperAction()
        if (
            step.phase == "NEOW"
            and str(step.action.get("kind") or "").lower() == "neow"
            and getattr(game_state, "screen_type", None) == ScreenType.EVENT
            and not _is_neow_card_select_state(game_state)
        ):
            options = list(getattr(getattr(game_state, "screen", None), "options", []) or [])
            if len(options) == 1:
                option = options[0]
                option_text = " ".join(
                    str(part)
                    for part in [
                        getattr(option, "label", None),
                        getattr(option, "text", None),
                        getattr(option, "name", None),
                    ]
                    if part
                ).lower()
                if "talk" in option_text:
                    return ChooseAction(choice_index=0)
        if (
            step.phase == "CARD_REWARD"
            and str(step.action.get("kind") or "").lower() == "card_reward"
            and getattr(game_state, "screen_type", None) == ScreenType.COMBAT_REWARD
        ):
            reward = _find_matching_reward(game_state, step.action)
            if reward is not None and reward.reward_type == RewardType.CARD:
                return CombatRewardAction(reward)
        return None

    def _maybe_trace_driven_neow_reward_action(self, game_state, step: RecordedTraceStep):
        if (
            step.phase != "CARD_REWARD"
            or str(step.action.get("kind") or "").lower() != "card_reward"
            or str(getattr(game_state, "room_type", "") or "") != "NeowRoom"
            or getattr(game_state, "screen_type", None) != ScreenType.CARD_REWARD
        ):
            return None
        choice_index = step.action.get("choice_index")
        if choice_index is None:
            return None
        try:
            choice_index = int(choice_index)
        except (TypeError, ValueError):
            return None
        if not (0 <= choice_index < len(getattr(getattr(game_state, "screen", None), "cards", []) or [])):
            return None

        card = game_state.screen.cards[choice_index]
        card_name = step.action.get("name") or getattr(card, "name", None)

        continue_choice_index = None
        post_continue_choice_index = None
        # Do not blindly queue Neow continue / first MAP choice as part of the
        # reward burst. Visual/no-compare replay can observe the map before
        # Communication Mod has exposed selectable next_nodes; the trace's
        # NEOW/MAP steps will drive those visible choices when ready.
        return NeowCardRewardAction(
            choice_index=choice_index,
            name=card_name,
            continue_choice_index=continue_choice_index,
            post_continue_choice_index=post_continue_choice_index,
            bridge_delay_seconds=NEOW_REWARD_BRIDGE_DELAY_SECONDS,
            continue_delay_seconds=NEOW_REWARD_CONTINUE_DELAY_SECONDS,
        )

    def _append_failure_if_needed(self, step_number: int, *, mismatches: list[str]) -> None:
        if not mismatches or not self.stop_on_mismatch or self.failed:
            return
        self.failed = True
        self.first_failure_step = step_number
        self.done = True

    def _finalize_pending_step(self, game_state, live_snapshot: dict[str, Any]) -> bool:
        assert self.pending_step is not None
        post_mismatches = compare_snapshots(self.pending_step.post, live_snapshot) if self.compare_state else []
        comparison_note = self.pending_comparison_note
        next_step = self.trace.steps[self.step_index + 1] if self.step_index + 1 < self.total_steps else None
        if not self.compare_state:
            expected_phase = _normalize_phase_name(self.pending_step.post.get("phase"))
            live_phase = _normalize_phase_name(live_snapshot.get("phase"))
            pending_phase = self.pending_step.phase
            wait_for_phase = False
            if pending_phase == "MAP":
                wait_for_phase = expected_phase in {
                    "COMBAT",
                    "EVENT",
                    "SHOP",
                    "TREASURE",
                    "CAMPFIRE",
                    "CHEST",
                    "CARD_REWARD",
                    "BOSS_REWARD",
                }
            elif pending_phase == "NEOW":
                wait_for_phase = expected_phase in {"MAP", "CARD_REWARD", "GRID"}
            if wait_for_phase and expected_phase and live_phase != expected_phase:
                return False
            if wait_for_phase and expected_phase == live_phase:
                expected_floor = self.pending_step.post.get("floor")
                if expected_floor is not None and live_snapshot.get("floor") != expected_floor:
                    return False
                if expected_phase == "COMBAT" and _combat_opening_still_resolving(live_snapshot):
                    return False
        if post_mismatches:
            if (
                self.pending_step.phase == "CARD_SELECT"
                and str(self.pending_step.action.get("kind") or "").lower() == "card_select"
                and str(getattr(game_state, "room_type", "") or "") == "NeowRoom"
                and getattr(game_state, "screen_type", None) == ScreenType.GRID
            ):
                if (
                    next_step is not None
                    and next_step.phase == "NEOW"
                    and any(action.get("action_class") == "StateAction" for action in self.pending_actions)
                ):
                    post_mismatches = []
                    comparison_note = "neow_card_select_callback_timeout_forced"
                    self.neow_card_select_continue_step_number = next_step.step
                else:
                    # Neow remove/transform flows can surface an intermediate GRID
                    # confirmation frame after the card is chosen but before the
                    # leave screen (and updated deck snapshot) is exposed. Keep
                    # polling through that transient frame instead of finalizing a
                    # mismatch against it.
                    return False
            if (
                self.pending_step.phase == "CARD_REWARD"
                and str(self.pending_step.action.get("kind") or "").lower() == "card_reward"
                and isinstance(self.pending_step.post.get("deck"), list)
                and isinstance(live_snapshot.get("deck"), list)
            ):
                picked_card_id = str(self.pending_step.action.get("card_id") or "")
                expected_deck = self.pending_step.post["deck"]
                live_deck = live_snapshot["deck"]
                if picked_card_id and picked_card_id in expected_deck and picked_card_id not in live_deck:
                    # Most reward picks should keep polling until the card has
                    # landed in the serialized deck snapshot. The main
                    # exception is the native CARD_REWARD pick -> skip bridge:
                    # Communication Mod can expose the next skip-ready reward
                    # frame before the deck update appears.
                    if not (
                        next_step is not None
                        and (
                            (
                                next_step.phase == "CARD_REWARD"
                                and str(next_step.action.get("kind") or "").lower() == "skip"
                                and live_snapshot.get("phase") == "CARD_REWARD"
                            )
                            or (
                                next_step.phase == "NEOW"
                                and self.neow_card_reward_blind_continue_step_number == next_step.step
                                and _is_neow_stale_card_reward_bridge_state(game_state)
                            )
                        )
                    ):
                        return False
            if (
                self.pending_step.phase == "CARD_REWARD"
                and str(self.pending_step.action.get("kind") or "").lower() == "card_reward"
                and _is_neow_leave_bridge_state(game_state)
                and isinstance(self.pending_step.post.get("deck"), list)
                and isinstance(live_snapshot.get("deck"), list)
            ):
                picked_card_id = str(self.pending_step.action.get("card_id") or "")
                expected_deck = self.pending_step.post["deck"]
                live_deck = live_snapshot["deck"]
                if picked_card_id and picked_card_id in expected_deck and picked_card_id not in live_deck:
                    # Neow reward picks can surface the single-option leave
                    # screen before the selected card has landed in the
                    # serialized deck snapshot. Keep polling until the reward
                    # side-effect arrives instead of finalizing against this
                    # intermediate state.
                    return False
            if (
                self.pending_step.phase == "CARD_REWARD"
                and str(self.pending_step.action.get("kind") or "").lower() == "card_reward"
                and next_step is not None
                and next_step.phase == "NEOW"
                and _is_neow_stale_card_reward_bridge_state(game_state)
                and isinstance(self.pending_step.post.get("deck"), list)
                and isinstance(live_snapshot.get("deck"), list)
            ):
                picked_card_id = str(self.pending_step.action.get("card_id") or "")
                expected_deck = self.pending_step.post["deck"]
                live_deck = live_snapshot["deck"]
                if picked_card_id and picked_card_id in expected_deck and picked_card_id in live_deck:
                    filtered_mismatches = _filter_neow_transition_mismatches(post_mismatches)
                    if not filtered_mismatches:
                        post_mismatches = []
                        comparison_note = "neow_reward_pick_stale_card_reward_needs_close"
                        self.neow_card_reward_close_bridge_step_number = next_step.step
                elif self.neow_card_reward_blind_continue_step_number == next_step.step:
                    post_mismatches = []
                    comparison_note = "neow_reward_pick_stale_card_reward_blind_continue"
            elif (
                self.pending_step.phase == "CARD_REWARD"
                and str(self.pending_step.action.get("kind") or "").lower() == "card_reward"
                and next_step is not None
                and _is_neow_leave_bridge_state(game_state)
                and next_step.phase == "CARD_REWARD"
                and str(next_step.action.get("kind") or "").lower() == "skip"
            ):
                post_mismatches = []
                comparison_note = "neow_reward_pick_transitions_to_leave"
                self.neow_bridge_step_number = next_step.step
            elif (
                self.pending_step.phase == "NEOW"
                and self.pending_step.post.get("phase") == "CARD_REWARD"
                and live_snapshot.get("phase") == "NEOW"
                and str(getattr(game_state, "room_type", "") or "") == "NeowRoom"
            ):
                # Some Neow rewards surface an intermediate Neow frame before
                # Communication Mod exposes the follow-up CARD_REWARD screen.
                # Keep polling instead of finalizing against that transition
                # frame.
                return False
            elif (
                self.pending_step.phase == "NEOW"
                and next_step is not None
                and next_step.phase == "NEOW"
                and _is_neow_leave_bridge_state(game_state)
                and isinstance(self.pending_step.post.get("deck"), list)
                and isinstance(live_snapshot.get("deck"), list)
            ):
                expected_deck = list(self.pending_step.post["deck"])
                live_deck = list(live_snapshot["deck"])
                missing_expected_cards = _missing_expected_cards(expected_deck, live_deck)
                if missing_expected_cards:
                    # Some immediate Neow rewards (for example, a random rare
                    # card) surface the single-option Leave screen before the
                    # obtained card has landed in the serialized deck
                    # snapshot. Keep polling instead of finalizing the step
                    # against that intermediate frame.
                    return False
            elif (
                self.pending_step.phase == "NEOW"
                and next_step is not None
                and next_step.phase == "MAP"
                and _is_neow_leave_bridge_state(game_state)
            ):
                filtered_mismatches = _filter_neow_transition_mismatches(post_mismatches)
                if not filtered_mismatches:
                    post_mismatches = []
                    comparison_note = "neow_continue_leave_bridge_pending_map_callback"
                else:
                    # Neow can surface the single-option continue screen
                    # before reward side-effects (for example, a random rare
                    # card entering the deck) have fully landed in the
                    # serialized state. Keep polling instead of finalizing the
                    # step against a transition-only intermediate snapshot.
                    return False
            elif (
                self.pending_step.phase == "NEOW"
                and _normalize_phase_name(self.pending_step.post.get("phase")) == "MAP"
                and _is_neow_leave_bridge_state(game_state)
            ):
                filtered_mismatches = [
                    mismatch
                    for mismatch in post_mismatches
                    if mismatch != "phase: expected 'MAP', got 'NEOW'"
                ]
                if not filtered_mismatches:
                    post_mismatches = []
                    comparison_note = "neow_continue_leave_bridge_pending_map_callback"
            elif self.pending_step.phase == "MAP" and _is_neow_map_bridge_state(game_state):
                # The first map step after Neow can briefly regress back to a
                # stale single-option Leave frame or surface a non-ready MAP
                # screen while `openMap()` is still settling. Treat those as
                # in-flight transitions instead of hard mismatches against the
                # target room/combat state.
                return False
            elif (
                self.pending_step.phase == "COMBAT"
                and self.pending_step.post.get("phase") == "CARD_REWARD"
                and next_step is not None
                and str(next_step.action.get("kind") or "").lower() == "reward_gold"
                and live_snapshot.get("phase") == "CARD_REWARD"
            ):
                expected_gold = self.pending_step.post.get("gold")
                reward_amount = next_step.action.get("amount")
                if (
                    isinstance(expected_gold, int)
                    and isinstance(reward_amount, int)
                    and live_snapshot.get("gold") == expected_gold + reward_amount
                ):
                    filtered_mismatches = [
                        mismatch
                        for mismatch in post_mismatches
                        if mismatch != f"gold: expected {expected_gold!r}, got {live_snapshot.get('gold')!r}"
                    ]
                    if not filtered_mismatches:
                        post_mismatches = []
                        comparison_note = "reward_gold_already_collected"
            elif (
                self.pending_step.phase == "CARD_REWARD"
                and str(self.pending_step.action.get("kind") or "").lower() == "card_reward"
                and next_step is not None
                and next_step.phase == "CARD_REWARD"
                and str(next_step.action.get("kind") or "").lower() == "skip"
                and live_snapshot.get("phase") == "CARD_REWARD"
            ):
                picked_card_id = str(self.pending_step.action.get("card_id") or "")
                expected_deck = self.pending_step.post.get("deck") or []
                live_deck = live_snapshot.get("deck") or []
                filtered_mismatches = [
                    mismatch
                    for mismatch in post_mismatches
                    if not mismatch.startswith("deck: ")
                ]
                if (
                    picked_card_id
                    and picked_card_id in expected_deck
                    and picked_card_id not in live_deck
                    and not filtered_mismatches
                ):
                    post_mismatches = []
                    comparison_note = "card_reward_pick_transitions_to_skip_before_deck_update"
            elif (
                self.pending_step.phase == "CARD_REWARD"
                and str(self.pending_step.action.get("kind") or "").lower() == "card_reward"
                and next_step is not None
                and (
                    next_step.phase == "MAP"
                    or (
                        next_step.phase == "CARD_REWARD"
                        and str(next_step.action.get("kind") or "").lower() == "skip"
                    )
                )
                and _is_empty_reward_proceed_bridge_state(game_state)
            ):
                filtered_mismatches = _filter_neow_transition_mismatches(post_mismatches)
                if not filtered_mismatches:
                    post_mismatches = []
                    comparison_note = "card_reward_transitions_to_empty_reward_proceed"
                    self.reward_proceed_bridge_step_number = next_step.step
            elif (
                self.pending_step.phase == "EVENT"
                and next_step is not None
                and next_step.phase == "MAP"
                and _is_single_leave_event_state(game_state)
            ):
                post_mismatches = []
                comparison_note = "event_choice_transitions_to_leave"
                self.event_leave_bridge_step_number = next_step.step
            elif (
                self.pending_step.phase == "CAMPFIRE"
                and next_step is not None
                and next_step.phase == "MAP"
                and _is_completed_rest_bridge_state(game_state)
            ):
                post_mismatches = []
                comparison_note = "campfire_choice_transitions_to_proceed"
                self.campfire_leave_bridge_step_number = next_step.step
            elif (
                self.pending_step.phase == "CHEST"
                and next_step is not None
                and next_step.phase == "MAP"
                and _is_empty_reward_proceed_bridge_state(game_state)
            ):
                post_mismatches = []
                comparison_note = "chest_choice_transitions_to_proceed"
                self.reward_proceed_bridge_step_number = next_step.step
            elif (
                self.pending_step.phase == "MAP"
                and self.pending_step.post.get("phase") == "COMBAT"
                and _combat_opening_still_resolving(live_snapshot)
            ):
                return False
            elif (
                self.pending_step.phase == "MAP"
                and next_step is not None
                and next_step.phase == "EVENT"
                and live_snapshot.get("phase") == "EVENT"
            ):
                filtered_mismatches = _filter_map_event_transition_mismatches(
                    post_mismatches,
                    expected_event_id=self.pending_step.post.get("event_id"),
                    actual_event_id=live_snapshot.get("event_id"),
                )
                next_event_id = _normalize_event_id(next_step.action.get("event_id"))
                actual_event_id = _normalize_event_id(live_snapshot.get("event_id"))
                if not filtered_mismatches and next_event_id and next_event_id == actual_event_id:
                    post_mismatches = []
                    comparison_note = "map_transition_populates_event_id_from_next_step"
            elif (
                self.pending_step.phase == "MAP"
                and next_step is not None
                and next_step.phase == "COMBAT"
                and _combat_opening_still_resolving(live_snapshot)
            ):
                return False
        if post_mismatches:
            if self.stop_on_mismatch:
                self._record_terminal_failure(
                    step=self.pending_step,
                    live_snapshot=live_snapshot,
                    pre_mismatches=list(self.pending_pre_mismatches),
                    note=comparison_note or "post_state_mismatch",
                    actual_pre=self.pending_actual_pre,
                    setup_actions=list(self.pending_setup_actions),
                    executed_actions=list(self.pending_actions),
                    post_mismatches=list(post_mismatches),
                )
            return False
        next_step = self.trace.steps[self.step_index + 1] if self.step_index + 1 < self.total_steps else None
        if (
            comparison_note is None
            and self.pending_step.phase == "CARD_REWARD"
            and str(self.pending_step.action.get("kind") or "").lower() == "card_reward"
            and next_step is not None
            and next_step.phase == "CARD_REWARD"
            and str(next_step.action.get("kind") or "").lower() == "skip"
            and _is_empty_reward_proceed_bridge_state(game_state)
        ):
            comparison_note = "card_reward_transitions_to_empty_reward_proceed"
            self.reward_proceed_bridge_step_number = next_step.step
        step_result = {
            "step": self.pending_step.step,
            "phase": self.pending_step.phase,
            "trace_action": self.pending_step.action,
            "setup_actions": list(self.pending_setup_actions),
            "executed_actions": list(self.pending_actions),
            "expected_pre": self.pending_step.pre,
            "actual_pre": self.pending_actual_pre,
            "pre_mismatches": list(self.pending_pre_mismatches),
            "expected_post": self.pending_step.post,
            "actual_post": live_snapshot,
            "post_mismatches": post_mismatches,
            "comparison_note": comparison_note,
        }
        self.results.append(step_result)
        self._emit_progress(
            status="step_finalized",
            step=self.pending_step,
            live_snapshot=live_snapshot,
            extra={"comparison_note": comparison_note},
        )
        self._append_failure_if_needed(
            self.pending_step.step,
            mismatches=list(self.pending_pre_mismatches) + list(post_mismatches),
        )
        self.pending_step = None
        self.pending_actual_pre = None
        self.pending_pre_mismatches = []
        self.pending_actions = []
        self.pending_setup_actions = []
        self.pending_comparison_note = None
        self.step_index += 1
        if self.step_index >= self.total_steps:
            self.done = True
        return True

    def _record_terminal_failure(
        self,
        *,
        step: RecordedTraceStep,
        live_snapshot: dict[str, Any],
        pre_mismatches: list[str],
        note: str,
        actual_pre: dict[str, Any] | None = None,
        setup_actions: list[dict[str, Any]] | None = None,
        executed_actions: list[dict[str, Any]] | None = None,
        post_mismatches: list[str] | None = None,
    ) -> None:
        self.results.append(
            {
                "step": step.step,
                "phase": step.phase,
                "trace_action": step.action,
                "setup_actions": list(setup_actions or []),
                "executed_actions": list(executed_actions or []),
                "expected_pre": step.pre,
                "actual_pre": actual_pre if actual_pre is not None else live_snapshot,
                "pre_mismatches": pre_mismatches,
                "expected_post": step.post,
                "actual_post": live_snapshot,
                "post_mismatches": list(post_mismatches or []),
                "comparison_note": note,
            }
        )
        self.failed = True
        self.done = True
        self.first_failure_step = step.step
        self._emit_progress(
            status="failed",
            step=step,
            live_snapshot=live_snapshot,
            extra={
                "comparison_note": note,
                "pre_mismatch_count": len(pre_mismatches),
                "post_mismatch_count": len(post_mismatches or []),
            },
        )

    def _raise_if_failed(self) -> None:
        if self.failed:
            raise RecordedReplayAbort(f"blocking replay mismatch at step {self.first_failure_step}")

    def next_action(self, game_state):
        if self.done or self.failed:
            return None
        while True:
            live_snapshot = snapshot_live_state(game_state)
            if self.pending_step is not None:
                followup_action = self._maybe_pending_followup_action(game_state)
                if followup_action is not None:
                    self.pending_actions.append(_describe_action(followup_action))
                    if self.progress_enabled:
                        print(
                            f"[replay-action] {json.dumps(_describe_action(followup_action), ensure_ascii=False, sort_keys=True)}",
                            file=sys.stderr,
                            flush=True,
                        )
                    return followup_action
                if not self._finalize_pending_step(game_state, live_snapshot):
                    if self.done or self.failed:
                        self._raise_if_failed()
                        return None
                    if (
                        self.pending_step.phase == "CARD_REWARD"
                        and str(self.pending_step.action.get("kind") or "").lower() == "card_reward"
                        and _is_neow_stale_card_reward_bridge_state(game_state)
                    ):
                        # Let Neow's post-pick reward transition arrive
                        # naturally. Injecting `state` into the stale
                        # CARD_REWARD frame diverges from the successful manual
                        # flow and can keep Communication Mod pinned there.
                        return None
                    if self.pending_step.phase == "MAP" and _is_neow_map_bridge_state(game_state):
                        # After `Leave`, Neow can briefly surface a stale
                        # single-option EVENT frame or a non-ready MAP frame
                        # while `openMap()` is still pushing the real
                        # map/combat callback. Injecting an eager `state`
                        # probe here can keep the transition oscillating.
                        return None
                    if (
                        self.pending_step.phase == "NEOW"
                        and self.neow_card_select_continue_step_number == self.pending_step.step
                        and _is_neow_card_select_state(game_state)
                    ):
                        # After a Neow upgrade/remove grid pick, the continue
                        # step can still surface against the stale GRID frame.
                        # Let the dedicated callback-timeout bridge finalize it
                        # instead of injecting extra `state` probes that keep
                        # the stale grid alive.
                        return None
                    if (
                        self.pending_step.phase == "NEOW"
                        and str(self.pending_step.action.get("kind") or "").lower() == "neow"
                    ):
                        next_step = self.trace.steps[self.step_index + 1] if self.step_index + 1 < self.total_steps else None
                        if (
                            next_step is not None
                            and next_step.phase == "NEOW"
                            and _is_neow_leave_bridge_state(game_state)
                            and isinstance(self.pending_step.post.get("deck"), list)
                            and isinstance(live_snapshot.get("deck"), list)
                            and _missing_expected_cards(
                                list(self.pending_step.post["deck"]),
                                list(live_snapshot["deck"]),
                            )
                        ):
                            # Immediate Neow rewards can expose the single-option
                            # Leave screen before the obtained card lands in the
                            # serialized deck. Keep waiting for the natural
                            # callback instead of spamming `state` into this
                            # stale frame.
                            return None
                    return StateAction()
                if self.done or self.failed:
                    self._raise_if_failed()
                    return None
                continue

            if self.step_index >= self.total_steps:
                self.done = True
                return None

            step = self.trace.steps[self.step_index]
            if self.progress_enabled and self._last_announced_step != step.step:
                print(
                    f"[replay-step] step={step.step}/{self.trace.steps[-1].step if self.trace.steps else 0} "
                    f"phase={step.phase} action={step.action.get('kind')} name={step.action.get('name')}",
                    file=sys.stderr,
                    flush=True,
                )
                self._last_announced_step = step.step
                self._emit_progress(status="awaiting_action", step=step, live_snapshot=live_snapshot)
            if (
                self.neow_card_reward_blind_continue_probe_step_number == step.step
                and self.neow_card_reward_blind_continue_step_number == step.step
                and self.neow_card_reward_blind_continue_choose_step_number == step.step
                and _is_neow_stale_card_reward_bridge_state(game_state, step)
            ):
                # The stale Neow blind-continue bridge has already consumed
                # both its single wait pulse and its single blind `choose 0`.
                # Keep polling naturally instead of spamming more commands
                # into the stale CARD_REWARD frame.
                return None
            setup_action = self._maybe_setup_action(game_state, step)
            if setup_action is None and self.neow_bridge_step_number == step.step and _is_neow_leave_bridge_state(game_state, step):
                setup_action = _single_neow_continue_action(game_state)
                self.neow_bridge_step_number = None
            if setup_action is None and self.neow_card_reward_close_bridge_step_number == step.step and _is_neow_stale_card_reward_bridge_state(game_state, step):
                setup_action = RawCommandAction("skip")
                self.neow_card_reward_close_bridge_step_number = None
            if (
                setup_action is None
                and self.neow_card_reward_blind_continue_choose_step_number == step.step
                and self.neow_card_reward_blind_continue_step_number == step.step
                and _is_neow_stale_card_reward_bridge_state(game_state, step)
            ):
                return None
            if (
                setup_action is None
                and self.neow_card_reward_blind_continue_step_number == step.step
                and _is_neow_stale_card_reward_bridge_state(game_state, step)
                and step.phase == "NEOW"
                and str(step.action.get("kind") or "").lower() == "neow"
            ):
                if self.neow_card_reward_blind_continue_probe_step_number != step.step:
                    setup_action = WaitAction(1)
                    self.neow_card_reward_blind_continue_probe_step_number = step.step
                elif self.neow_card_reward_blind_continue_choose_step_number != step.step:
                    setup_action = NeowContinueAction()
                    self.neow_card_reward_blind_continue_choose_step_number = step.step
                else:
                    return None
            if (
                setup_action is None
                and step.phase == "NEOW"
                and self.neow_card_select_continue_step_number == step.step
                and _single_neow_continue_action(game_state) is not None
            ):
                setup_action = NeowContinueAction(include_settle_probe=False)
            if (
                setup_action is None
                and step.phase == "NEOW"
                and self.neow_card_select_continue_step_number == step.step
                and _is_neow_card_select_state(game_state)
            ):
                return None
            if setup_action is None and self.event_leave_bridge_step_number == step.step and _is_single_leave_event_state(game_state, step):
                setup_action = _single_leave_event_action(game_state)
                if step.phase == "MAP":
                    self.event_leave_map_continue_step_number = step.step
                self.event_leave_bridge_step_number = None
            if setup_action is None and self.campfire_leave_bridge_step_number == step.step and _is_completed_rest_bridge_state(game_state, step):
                setup_action = ProceedAction()
                self.campfire_leave_bridge_step_number = None
            if setup_action is None and self.reward_proceed_bridge_step_number == step.step and _is_empty_reward_proceed_bridge_state(game_state, step):
                setup_action = ProceedAction()
                self.reward_proceed_bridge_step_number = None
            if setup_action is None and self.reward_proceed_bridge_step_number == step.step and _is_chest_leave_bridge_state(game_state, step):
                setup_action = ProceedAction()
                self.reward_proceed_bridge_step_number = None
            if (
                setup_action is None
                and step.phase == "MAP"
                and _is_neow_leave_bridge_state(game_state, step)
            ):
                if self.neow_leave_map_continue_step_number != step.step:
                    setup_action = NeowContinueAction()
                    self.neow_leave_map_continue_step_number = step.step
                else:
                    return None
            if setup_action is not None:
                self.pending_setup_actions.append(_describe_action(setup_action))
                if self.progress_enabled:
                    print(
                        f"[replay-action] {json.dumps(_describe_action(setup_action), ensure_ascii=False, sort_keys=True)}",
                        file=sys.stderr,
                        flush=True,
                    )
                return setup_action
            next_step = self.trace.steps[self.step_index + 1] if self.step_index + 1 < self.total_steps else None
            pre_mismatches = compare_snapshots(step.pre, live_snapshot) if self.compare_state else []
            if self.neow_bridge_step_number == step.step and _is_neow_leave_bridge_state(game_state, step):
                pre_mismatches = []
            if self.neow_card_reward_close_bridge_step_number == step.step and _is_neow_stale_card_reward_bridge_state(game_state, step):
                pre_mismatches = []
            if self.neow_card_reward_blind_continue_step_number == step.step and _is_neow_stale_card_reward_bridge_state(game_state, step):
                pre_mismatches = []
            if (
                self.neow_card_select_continue_step_number == step.step
                and step.phase == "NEOW"
                and _is_neow_card_select_state(game_state)
            ):
                pre_mismatches = []
            if self.event_leave_bridge_step_number == step.step and _is_single_leave_event_state(game_state, step):
                pre_mismatches = []
            if (
                pre_mismatches
                and step.phase == "EVENT"
                and _is_single_leave_event_state(game_state, step)
            ):
                pre_mismatches = []
            if self.campfire_leave_bridge_step_number == step.step and _is_completed_rest_bridge_state(game_state, step):
                pre_mismatches = []
            if self.reward_proceed_bridge_step_number == step.step and _is_empty_reward_proceed_bridge_state(game_state, step):
                pre_mismatches = []
            if self.reward_proceed_bridge_step_number == step.step and _is_chest_leave_bridge_state(game_state, step):
                pre_mismatches = []
            if (
                pre_mismatches
                and step.phase == "EVENT"
                and _normalize_event_id(step.action.get("event_id"))
                and _normalize_event_id(live_snapshot.get("event_id"))
            ):
                pre_mismatches = _filter_event_pre_mismatches(
                    pre_mismatches,
                    step_action=step.action,
                    actual_event_id=live_snapshot.get("event_id"),
                )
            if (
                pre_mismatches
                and step.phase == "CARD_REWARD"
                and step.action.get("kind") == "skip"
                and compare_snapshots(step.post, live_snapshot) == []
            ):
                pre_mismatches = []
            if (
                pre_mismatches
                and step.phase == "NEOW"
                and str(step.action.get("kind") or "").lower() == "neow"
                and _is_neow_leave_bridge_state(game_state)
                and isinstance(step.pre.get("deck"), list)
                and isinstance(live_snapshot.get("deck"), list)
            ):
                missing_expected_cards = _missing_expected_cards(
                    list(step.pre["deck"]),
                    list(live_snapshot["deck"]),
                )
                filtered_mismatches = [
                    mismatch for mismatch in pre_mismatches if not mismatch.startswith("deck: ")
                ]
                if missing_expected_cards and not filtered_mismatches:
                    # Immediate Neow rewards can expose a stale single-option
                    # Leave frame before the obtained card lands in the live
                    # deck snapshot. Let the follow-up Neow continue bridge
                    # run against that stale frame instead of failing on the
                    # transient deck mismatch.
                    pre_mismatches = []
            previous_step = self.trace.steps[self.step_index - 1] if self.step_index > 0 else None
            if (
                pre_mismatches
                and step.phase == "CARD_REWARD"
                and str(step.action.get("kind") or "").lower() == "skip"
                and previous_step is not None
                and previous_step.phase == "CARD_REWARD"
                and str(previous_step.action.get("kind") or "").lower() == "card_reward"
                and live_snapshot.get("phase") == "CARD_REWARD"
            ):
                picked_card_id = str(previous_step.action.get("card_id") or "")
                expected_deck = step.pre.get("deck") or []
                live_deck = live_snapshot.get("deck") or []
                filtered_mismatches = [
                    mismatch
                    for mismatch in pre_mismatches
                    if not mismatch.startswith("deck: ")
                ]
                if (
                    picked_card_id
                    and picked_card_id in expected_deck
                    and picked_card_id not in live_deck
                    and not filtered_mismatches
                ):
                    pre_mismatches = []
            if (
                pre_mismatches
                and step.phase == "NEOW"
                and self.neow_card_reward_blind_continue_step_number == step.step
                and live_snapshot.get("phase") in {"MAP", "COMBAT"}
            ):
                pre_mismatches = _filter_neow_transition_mismatches(pre_mismatches)
            if (
                pre_mismatches
                and step.phase == "MAP"
                and _is_neow_map_bridge_state(game_state, step)
            ):
                pre_mismatches = []
            if (
                pre_mismatches
                and step.phase == "MAP"
                and self.neow_card_reward_blind_map_step_number == step.step
                and live_snapshot.get("phase") in {"MAP", "COMBAT"}
            ):
                # A prequeued Neow blind bridge may already have consumed the
                # first map choice before Communication Mod surfaces a fresh
                # callback frame. Do not resend that map command while we wait
                # for the room transition to settle.
                pre_mismatches = []
            if (
                pre_mismatches
                and step.phase == "MAP"
                and compare_snapshots(step.post, live_snapshot) == []
            ):
                # Some transitions (notably the first room after Neow) can
                # advance from the map into the destination room before
                # Communication Mod exposes a stable command-ready map frame.
                # If the live state already matches the trace post-state,
                # treat the map choice as already consumed.
                pre_mismatches = []
            if (
                pre_mismatches
                and step.phase == "MAP"
                and _normalize_phase_name(step.post.get("phase")) == "COMBAT"
                and live_snapshot.get("phase") == "COMBAT"
                and live_snapshot.get("floor") == step.post.get("floor")
                and _combat_opening_still_resolving(live_snapshot)
            ):
                # The first room after Neow can auto-advance through the map
                # choice before Communication Mod exposes a stable combat
                # snapshot. Keep polling instead of failing against the
                # intermediate combat-opening frame.
                pre_mismatches = []
            if (
                pre_mismatches
                and step.phase == "EVENT"
                and _is_neow_map_bridge_state(game_state)
                and self.neow_post_map_settle_probe_step_number == step.step
            ):
                # The first post-Neow event can still surface against a stale
                # Neow continue frame even after the trace has advanced into
                # the event. Let the blind event choice bridge run before
                # treating this as a hard state divergence.
                pre_mismatches = []
            if (
                pre_mismatches
                and step.phase == "CARD_SELECT"
                and _is_neow_map_bridge_state(game_state)
                and self.neow_post_map_settle_probe_step_number == step.step
            ):
                pre_mismatches = []
            if (
                pre_mismatches
                and step.phase in {"MAP", "EVENT", "COMBAT", "CARD_REWARD", "SHOP", "CAMPFIRE", "TREASURE", "CHEST", "BOSS_RELIC"}
                and _is_neow_card_select_state(game_state)
                and self.neow_post_card_select_settle_probe_step_number == step.step
            ):
                pre_mismatches = []
            if pre_mismatches and self.stop_on_mismatch:
                self._record_terminal_failure(
                    step=step,
                    live_snapshot=live_snapshot,
                    pre_mismatches=pre_mismatches,
                    note="pre_state_mismatch",
                )
                self._raise_if_failed()
                return None

            if (
                step.phase == "CARD_REWARD"
                and str(step.action.get("kind") or "").lower() == "skip"
                and (
                    bool(self.pending_setup_actions)
                    or self.pending_comparison_note == "neow_leave_bridge_state"
                    or self.neow_bridge_step_number == step.step
                    or self.neow_card_reward_close_bridge_step_number == step.step
                    or self.neow_card_reward_blind_continue_step_number == step.step
                    or (
                        getattr(game_state, "screen_type", None) == ScreenType.EVENT
                        and str(getattr(game_state, "room_type", "") or "") == "NeowRoom"
                    )
                )
            ):
                live_action = None
            else:
                if (
                    step.phase == "NEOW"
                    and str(step.action.get("kind") or "").lower() == "neow"
                    and bool(self.pending_setup_actions)
                    and next_step is not None
                    and next_step.phase == "NEOW"
                    and _is_neow_leave_bridge_state(game_state, step)
                    and isinstance(step.post.get("deck"), list)
                    and isinstance(live_snapshot.get("deck"), list)
                    and _missing_expected_cards(list(step.post["deck"]), list(live_snapshot["deck"]))
                ):
                    # Some Neow paths still surface the stale single-option
                    # Leave screen after the setup Talk has already advanced
                    # the event, but before the obtained reward is reflected in
                    # the serialized deck. Do not inject a duplicate
                    # `choose 0`; let the dedicated callback-timeout bridge
                    # finalize this immediate-reward transition.
                    live_action = None
                else:
                    live_action = self._maybe_trace_driven_neow_reward_action(game_state, step)
                    if live_action is None:
                        live_action = _next_trace_action_for_game(game_state, step)
            if live_action is None:
                if (
                    step.phase in {"MAP", "EVENT", "COMBAT", "CARD_REWARD", "SHOP", "CAMPFIRE", "TREASURE", "CHEST", "BOSS_RELIC"}
                    and self.neow_post_card_select_settle_probe_step_number == step.step
                    and _is_neow_card_select_state(game_state)
                ):
                    return None
                if (
                    step.phase == "CARD_SELECT"
                    and self.neow_post_map_settle_probe_step_number == step.step
                    and _is_neow_map_bridge_state(game_state)
                ):
                    # Some post-Neow event chains (for example shrine upgrades)
                    # can still be pinned to the stale Neow MAP frame when the
                    # trace has already advanced into a follow-up grid/select
                    # step. Keep waiting for the shared settle bridge instead
                    # of failing translation against the stale MAP snapshot.
                    return None
                if (
                    step.phase == "EVENT"
                    and self.neow_post_map_settle_probe_step_number == step.step
                    and _is_neow_map_bridge_state(game_state)
                ):
                    # Some runs continue to expose the stale Neow MAP frame
                    # into the first post-Neow event. Let the short callback
                    # timeout/forced-finalize bridge handle that frame instead
                    # of failing immediately on action translation.
                    return None
                if (
                    step.phase == "MAP"
                    and self.neow_card_reward_blind_map_step_number == step.step
                    and live_snapshot.get("phase") == "MAP"
                ):
                    return None
                if (
                    step.phase == "NEOW"
                    and self.neow_card_reward_blind_continue_step_number == step.step
                    and live_snapshot.get("phase") in {"MAP", "COMBAT"}
                ):
                    post_mismatches = _filter_neow_transition_mismatches(
                        compare_snapshots(step.post, live_snapshot) if self.compare_state else []
                    )
                    if not post_mismatches:
                        self.results.append(
                            {
                                "step": step.step,
                                "phase": step.phase,
                                "trace_action": step.action,
                                "setup_actions": list(self.pending_setup_actions),
                                "executed_actions": [],
                                "expected_pre": step.pre,
                                "actual_pre": live_snapshot,
                                "pre_mismatches": pre_mismatches,
                                "expected_post": step.post,
                                "actual_post": live_snapshot,
                                "post_mismatches": [],
                                "comparison_note": "neow_blind_continue_already_resolved_without_command",
                            }
                        )
                        self.step_index += 1
                        self.pending_setup_actions = []
                        if self.step_index >= self.total_steps:
                            self.done = True
                        return None
                if (
                    step.phase == "NEOW"
                    and self.neow_card_select_continue_step_number == step.step
                    and _is_neow_card_select_state(game_state)
                ):
                    return None
                if (
                    step.phase == "MAP"
                    and _normalize_phase_name(step.post.get("phase")) == "COMBAT"
                    and live_snapshot.get("phase") == "COMBAT"
                    and live_snapshot.get("floor") == step.post.get("floor")
                    and _combat_opening_still_resolving(live_snapshot)
                ):
                    return StateAction()
                if step.phase == "MAP" and _is_neow_map_bridge_state(game_state, step):
                    return None
                if (
                    step.phase == "MAP"
                    and getattr(game_state, "screen_type", None) == ScreenType.MAP
                    and str(step.action.get("name") or "").upper() != "BOSS"
                    and len(getattr(getattr(game_state, "screen", None), "next_nodes", []) or []) == 0
                ):
                    # Communication Mod can briefly surface the map screen
                    # before the selectable next-node list is populated.
                    # Keep polling instead of treating that transition frame
                    # as a hard translation failure.
                    return StateAction()
                post_mismatches = compare_snapshots(step.post, live_snapshot) if self.compare_state else []
                if not post_mismatches:
                    self.results.append(
                        {
                            "step": step.step,
                            "phase": step.phase,
                            "trace_action": step.action,
                            "setup_actions": list(self.pending_setup_actions),
                            "executed_actions": [],
                            "expected_pre": step.pre,
                            "actual_pre": live_snapshot,
                            "pre_mismatches": pre_mismatches,
                            "expected_post": step.post,
                            "actual_post": live_snapshot,
                            "post_mismatches": post_mismatches,
                            "comparison_note": "already_resolved_without_command",
                        }
                    )
                    self._append_failure_if_needed(
                        step.step,
                        mismatches=list(pre_mismatches) + list(post_mismatches),
                    )
                    if self.failed:
                        self._raise_if_failed()
                        return None
                    self.step_index += 1
                    self.pending_setup_actions = []
                    if self.step_index >= self.total_steps:
                        self.done = True
                        return None
                    continue

                explicit_step = (
                    step.phase in {"COMBAT", "MAP", "EVENT", "NEOW", "CARD_REWARD", "SHOP", "CAMPFIRE", "CARD_SELECT", "TREASURE", "CHEST", "BOSS_RELIC"}
                    and str(step.action.get("kind") or "").lower() not in {"", "noop"}
                )
                if explicit_step:
                    context = json.dumps(_translation_debug_context(game_state), ensure_ascii=False, sort_keys=True)
                    self._record_terminal_failure(
                        step=step,
                        live_snapshot=live_snapshot,
                        pre_mismatches=pre_mismatches + [f"unable_to_translate_action: {context}"],
                        note="translation_failure",
                    )
                    self._raise_if_failed()
                    return None
                self.results.append(
                    {
                        "step": step.step,
                        "phase": step.phase,
                        "trace_action": step.action,
                        "setup_actions": list(self.pending_setup_actions),
                        "executed_actions": [],
                        "expected_pre": step.pre,
                        "actual_pre": live_snapshot,
                        "pre_mismatches": pre_mismatches,
                        "expected_post": step.post,
                        "actual_post": live_snapshot,
                        "post_mismatches": post_mismatches,
                        "comparison_note": "implicit_noop",
                    }
                )
                self._append_failure_if_needed(
                    step.step,
                    mismatches=list(pre_mismatches) + list(post_mismatches),
                )
                if self.failed:
                    self._raise_if_failed()
                    return None
                self.step_index += 1
                self.pending_setup_actions = []
                if self.step_index >= self.total_steps:
                    self.done = True
                    return None
                continue

            self.pending_step = step
            self.pending_actual_pre = live_snapshot
            self.pending_pre_mismatches = pre_mismatches
            self.pending_actions = [_describe_action(live_action)]
            self.pending_comparison_note = None
            if self.neow_bridge_step_number == step.step and _is_neow_leave_bridge_state(game_state, step):
                self.pending_comparison_note = "neow_leave_bridge_state"
                self.neow_bridge_step_number = None
            if self.neow_card_reward_close_bridge_step_number == step.step and _is_neow_stale_card_reward_bridge_state(game_state, step):
                self.pending_comparison_note = "neow_card_reward_close_bridge_state"
                self.neow_card_reward_close_bridge_step_number = None
            if self.neow_card_reward_blind_map_step_number == step.step and live_snapshot.get("phase") in {"MAP", "COMBAT"}:
                self.pending_comparison_note = "neow_card_reward_blind_map_bridge_state"
                self.neow_card_reward_blind_map_step_number = None
            if self.event_leave_bridge_step_number == step.step and _is_single_leave_event_state(game_state, step):
                self.pending_comparison_note = "event_leave_bridge_state"
                self.event_leave_bridge_step_number = None
            if self.campfire_leave_bridge_step_number == step.step and _is_completed_rest_bridge_state(game_state, step):
                self.pending_comparison_note = "campfire_leave_bridge_state"
                self.campfire_leave_bridge_step_number = None
            if self.reward_proceed_bridge_step_number == step.step and _is_empty_reward_proceed_bridge_state(game_state, step):
                self.pending_comparison_note = "reward_proceed_bridge_state"
                self.reward_proceed_bridge_step_number = None
            if self.progress_enabled:
                print(
                    f"[replay-action] {json.dumps(_describe_action(live_action), ensure_ascii=False, sort_keys=True)}",
                    file=sys.stderr,
                    flush=True,
                )
            return live_action


class _RecordedReplayAgent:
    def __init__(self, driver: _StateDrivenReplayDriver) -> None:
        self.driver = driver

    def handle_error(self, error: str):
        raise RecordedReplayAbort(f"communication error: {error}")

    def get_next_action_in_game(self, game_state):
        action = self.driver.next_action(game_state)
        if self.driver.done or self.driver.failed:
            return None
        return action

    def get_next_action_out_of_game(self):
        return None


def _wait_for_any_update(
    coordinator: Coordinator,
    *,
    timeout_seconds: float,
    context: str,
) -> None:
    deadline = time.time() + timeout_seconds
    probe_interval = min(1.0, max(0.25, timeout_seconds / 4.0))
    next_probe_at = time.time() + probe_interval
    while time.time() < deadline:
        received = coordinator.receive_game_state_update(block=False, perform_callbacks=False)
        if received:
            if coordinator.last_error is not None:
                if _clear_tolerable_neow_skip_error(coordinator, request_state=True):
                    continue
                raise RuntimeError(f"{context}: Communication Mod error: {coordinator.last_error}")
            return
        if time.time() >= next_probe_at:
            last_game_state = getattr(coordinator, "last_game_state", None)
            if (
                not _is_neow_stale_card_reward_bridge_state(last_game_state)
                and not _is_neow_card_select_state(last_game_state)
            ):
                _send_probe_command(coordinator, "state")
            next_probe_at = time.time() + probe_interval
        time.sleep(0.05)
    raise TimeoutError(f"{context}: timed out waiting for next state update after {timeout_seconds:.1f}s")


def _wait_for_callback_update_with_timeout(
    coordinator: Coordinator,
    *,
    timeout_seconds: float,
    context: str,
) -> None:
    deadline = time.time() + timeout_seconds
    probe_interval = min(1.0, max(0.25, timeout_seconds / 4.0))
    next_probe_at = time.time() + probe_interval
    probe_count = 0
    while time.time() < deadline:
        received = coordinator.receive_game_state_update(block=False, perform_callbacks=True)
        if received:
            if coordinator.last_error is not None:
                if _clear_tolerable_neow_skip_error(coordinator, request_state=True):
                    continue
                raise RuntimeError(f"{context}: Communication Mod error: {coordinator.last_error}")
            return
        if time.time() >= next_probe_at:
            last_game_state = getattr(coordinator, "last_game_state", None)
            room_type = str(getattr(last_game_state, "room_type", "") or "")
            if (
                _is_neow_stale_card_reward_bridge_state(last_game_state)
                or _is_neow_leave_bridge_state(last_game_state)
                or _is_neow_card_select_state(last_game_state)
                or (
                    last_game_state is not None
                    and getattr(last_game_state, "screen_type", None) == ScreenType.MAP
                    and room_type == "NeowRoom"
                    and len(getattr(getattr(last_game_state, "screen", None), "next_nodes", []) or []) > 0
                )
            ):
                probe_command = None
            elif room_type == "NeowRoom":
                probe_command = "state" if probe_count % 2 == 0 else "wait 1"
            else:
                probe_command = "state"
            if probe_command is not None:
                _send_probe_command(coordinator, probe_command)
                probe_count += 1
            next_probe_at = time.time() + probe_interval
        time.sleep(0.05)
    raise TimeoutError(f"{context}: timed out waiting for callback-driven state update after {timeout_seconds:.1f}s")


def _should_allow_nonready_direct_action(coordinator: Coordinator, driver: "_StateDrivenReplayDriver") -> bool:
    last_game_state = getattr(coordinator, "last_game_state", None)
    if last_game_state is None:
        return False
    pending_step = getattr(driver, "pending_step", None)
    current_step = None
    trace = getattr(driver, "trace", None)
    step_index = getattr(driver, "step_index", None)
    total_steps = getattr(driver, "total_steps", None)
    if (
        pending_step is None
        and trace is not None
        and isinstance(step_index, int)
        and isinstance(total_steps, int)
        and 0 <= step_index < total_steps
    ):
        current_step = trace.steps[step_index]
    return (
        (
            getattr(driver, "neow_card_reward_blind_continue_step_number", None) is not None
            and _is_neow_stale_card_reward_bridge_state(last_game_state)
        )
        or (
            pending_step is not None
            and pending_step.phase == "MAP"
            and _is_neow_map_bridge_state(last_game_state, pending_step)
        )
        or (
            current_step is not None
            and current_step.phase == "MAP"
            and _is_neow_map_bridge_state(last_game_state, current_step)
        )
        or (
            pending_step is not None
            and pending_step.phase == "EVENT"
            and _is_single_leave_event_state(last_game_state, pending_step)
        )
        or (
            current_step is not None
            and current_step.phase == "EVENT"
            and _is_single_leave_event_state(last_game_state, current_step)
        )
        or (
            pending_step is not None
            and pending_step.phase == "EVENT"
            and getattr(driver, "neow_post_map_settle_probe_step_number", None) == pending_step.step
            and _is_neow_map_bridge_state(last_game_state)
        )
        or (
            current_step is not None
            and current_step.phase == "EVENT"
            and getattr(driver, "neow_post_map_settle_probe_step_number", None) == current_step.step
            and _is_neow_map_bridge_state(last_game_state)
        )
        or (
            pending_step is not None
            and pending_step.phase in {"MAP", "EVENT"}
            and getattr(driver, "neow_post_card_select_settle_probe_step_number", None) == pending_step.step
            and _is_neow_card_select_state(last_game_state)
        )
        or (
            current_step is not None
            and current_step.phase in {"MAP", "EVENT"}
            and getattr(driver, "neow_post_card_select_settle_probe_step_number", None) == current_step.step
            and _is_neow_card_select_state(last_game_state)
        )
    )


def _should_skip_ready_wait_for_neow_blind_continue(
    coordinator: Coordinator,
    driver: "_StateDrivenReplayDriver",
) -> bool:
    last_game_state = getattr(coordinator, "last_game_state", None)
    if getattr(coordinator, "game_is_ready", False) or last_game_state is None:
        return False
    step_number = getattr(driver, "neow_card_reward_blind_continue_step_number", None)
    if step_number is None:
        return False
    return (
        getattr(driver, "neow_card_reward_blind_continue_probe_step_number", None) == step_number
        and getattr(driver, "neow_card_reward_blind_continue_choose_step_number", None) != step_number
        and _is_neow_stale_card_reward_bridge_state(last_game_state)
    )


def _should_skip_ready_wait_for_neow_leave_map_continue(
    coordinator: Coordinator,
    driver: "_StateDrivenReplayDriver",
) -> bool:
    last_game_state = getattr(coordinator, "last_game_state", None)
    if getattr(coordinator, "game_is_ready", False) or last_game_state is None:
        return False
    pending_step = getattr(driver, "pending_step", None)
    if pending_step is None or pending_step.phase != "MAP":
        return False
    return (
        getattr(driver, "neow_leave_map_continue_step_number", None) == pending_step.step
        and _is_neow_map_bridge_state(last_game_state, pending_step)
    )


def _should_skip_ready_wait_for_event_leave_map_continue(
    coordinator: Coordinator,
    driver: "_StateDrivenReplayDriver",
) -> bool:
    last_game_state = getattr(coordinator, "last_game_state", None)
    if getattr(coordinator, "game_is_ready", False) or last_game_state is None:
        return False
    pending_step = getattr(driver, "pending_step", None)
    if pending_step is None or pending_step.phase != "MAP":
        return False
    return (
        getattr(driver, "event_leave_map_continue_step_number", None) == pending_step.step
        and _is_single_leave_event_state(last_game_state, pending_step)
    )


def _should_wait_for_any_update_for_post_neow_settle(
    driver: "_StateDrivenReplayDriver",
    current_step: RecordedTraceStep | None,
    last_game_state,
) -> bool:
    return (
        current_step is not None
        and current_step.phase == "COMBAT"
        and getattr(driver, "neow_post_map_settle_probe_step_number", None) == current_step.step
        and _is_neow_map_bridge_state(last_game_state)
    )


def _should_use_short_post_neow_callback_timeout(
    driver: "_StateDrivenReplayDriver",
    pending_step: RecordedTraceStep | None,
    current_step: RecordedTraceStep | None,
    last_game_state,
) -> bool:
    target_step = pending_step or current_step
    if target_step is None:
        return False
    if (
        _is_neow_map_bridge_state(last_game_state)
        and getattr(driver, "neow_post_map_settle_probe_step_number", None) == target_step.step
        and target_step.phase in {"CARD_SELECT", "COMBAT", "CARD_REWARD", "MAP", "SHOP", "EVENT", "CAMPFIRE", "TREASURE", "CHEST", "BOSS_RELIC"}
    ):
        return True
    if (
        _is_neow_card_select_state(last_game_state)
        and getattr(driver, "neow_post_card_select_settle_probe_step_number", None) == target_step.step
        and target_step.phase in {"MAP", "COMBAT", "EVENT", "CARD_REWARD", "SHOP", "CAMPFIRE", "TREASURE", "CHEST", "BOSS_RELIC"}
    ):
        return True
    if (
        _is_single_leave_event_state(last_game_state, target_step)
        and getattr(driver, "event_leave_map_continue_step_number", None) == target_step.step
        and target_step.phase == "MAP"
    ):
        return True
    return False


def _play_recorded_game(
    coordinator: Coordinator,
    *,
    player_class: PlayerClass,
    ascension_level: int,
    seed: str | None,
    driver: "_StateDrivenReplayDriver",
) -> None:
    coordinator.clear_actions()
    coordinator.stop_after_run = False
    emit_progress = getattr(driver, "_emit_progress", None)
    while not coordinator.game_is_ready:
        coordinator.receive_game_state_update(block=True, perform_callbacks=False)
    if not coordinator.in_game:
        StartGameAction(player_class, ascension_level, seed).execute(coordinator)
        coordinator.receive_game_state_update(block=True)

    out_of_game_since: float | None = None
    while True:
        if driver.failed or driver.done:
            break
        if driver.step_index >= driver.total_steps:
            driver.done = True
            break

        executed_any_action = False
        trace = getattr(driver, "trace", None)
        while coordinator.action_queue and coordinator.action_queue[0].can_be_executed(coordinator):
            queued_action = coordinator.action_queue[0]
            current_step = (
                trace.steps[driver.step_index]
                if trace is not None and getattr(driver, "step_index", 0) < getattr(driver, "total_steps", 0)
                else None
            )
            if current_step is not None:
                if emit_progress is not None:
                    emit_progress(
                        status="executing_queued_action",
                        step=current_step,
                        live_snapshot=snapshot_live_state(getattr(coordinator, "last_game_state", None)),
                        extra={"queued_action": _describe_action(queued_action)},
                    )
            coordinator.execute_next_action()
            executed_any_action = True
            if not coordinator.game_is_ready:
                # Commands that enqueue follow-up probes (for example Neow
                # reward/select bridges that append `wait 1` + `state`) must
                # yield back to the game loop after the first command is sent.
                # Draining the queued follow-ups in the same rollout turn
                # batches multiple commands into one frame and can keep the UI
                # stuck on the stale pre-update screen.
                break
        if driver.done or driver.failed:
            continue
        current_trace_step = (
            trace.steps[driver.step_index]
            if trace is not None and getattr(driver, "step_index", 0) < getattr(driver, "total_steps", 0)
            else None
        )
        if (
            not coordinator.action_queue
            and _should_wait_for_any_update_for_post_neow_settle(
                driver,
                current_trace_step,
                getattr(coordinator, "last_game_state", None),
            )
        ):
            _send_probe_command(coordinator, "state")
            try:
                _wait_for_any_update(
                    coordinator,
                    timeout_seconds=min(POST_NEOW_SETTLE_TIMEOUT_SECONDS, DEFAULT_REPLAY_ACTION_TIMEOUT_SECONDS),
                    context="recorded replay main loop awaiting post-neow combat settle update",
                )
            except TimeoutError:
                last_game_state = getattr(coordinator, "last_game_state", None)
                if (
                    current_trace_step is not None
                    and current_trace_step.phase == "COMBAT"
                    and getattr(driver, "neow_post_map_settle_probe_step_number", None) == current_trace_step.step
                    and last_game_state is not None
                    and _is_neow_map_bridge_state(last_game_state)
                ):
                    coordinator.clear_actions()
                    driver.neow_post_map_settle_probe_step_number = None
                    driver._force_finalize_current_step(
                        current_trace_step,
                        live_snapshot=snapshot_live_state(last_game_state),
                        comparison_note="neow_post_map_settle_callback_timeout_forced",
                    )
                    continue
                driver.neow_post_map_settle_probe_step_number = None
                raise
            driver.neow_post_map_settle_probe_step_number = None
            continue
        if (
            not coordinator.action_queue
            and getattr(coordinator, "last_game_state", None) is not None
            and (
                coordinator.game_is_ready
                or _should_allow_nonready_direct_action(coordinator, driver)
            )
        ):
            allow_nonready_direct = _should_allow_nonready_direct_action(coordinator, driver)
            direct_action = driver.next_action(coordinator.last_game_state)
            if driver.done or driver.failed:
                continue
            if direct_action is not None:
                current_step = (
                    trace.steps[driver.step_index]
                    if trace is not None and getattr(driver, "step_index", 0) < getattr(driver, "total_steps", 0)
                    else None
                )
                live_snapshot = snapshot_live_state(getattr(coordinator, "last_game_state", None))
                action_payload = _describe_action(direct_action)
                if emit_progress is not None:
                    emit_progress(
                        status="dispatching_action",
                        step=current_step,
                        live_snapshot=live_snapshot,
                        extra={"returned_action": action_payload},
                    )
                coordinator.add_action_to_queue(direct_action)
                continue
        if executed_any_action:
            if coordinator.action_queue:
                # Some replay actions intentionally queue follow-up probes
                # (for example Neow card reward/select paths that append
                # `wait 1` + `state`). Do not immediately pivot into a phase
                # wait against the stale pre-follow-up frame, but also do not
                # drain the queued actions in the same rollout turn. Yield one
                # short tick back to the game loop so the follow-up commands
                # can execute on subsequent outer-loop iterations without
                # batching into the original action's frame.
                time.sleep(0.05)
                continue
            pending_step = getattr(driver, "pending_step", None)
            expected_post_phase = (
                _normalize_phase_name(getattr(pending_step, "post", {}).get("phase"))
                if pending_step is not None
                else None
            )
            pending_action_kind = (
                str(getattr(pending_step, "action", {}).get("kind") or "").lower()
                if pending_step is not None
                else ""
            )
            current_room_type = str(getattr(getattr(coordinator, "last_game_state", None), "room_type", "") or "")
            if (
                pending_step is not None
                and str(getattr(pending_step, "phase", "") or "").upper() == "NEOW"
                and expected_post_phase in {"NEOW", "CARD_REWARD", "MAP"}
            ):
                _wait_for_phase_with_timeout(
                    coordinator,
                    expected_phase=expected_post_phase,
                    timeout_seconds=DEFAULT_REPLAY_ACTION_TIMEOUT_SECONDS,
                    context="recorded replay main loop after executing Neow action",
                )
            elif (
                pending_step is not None
                and str(getattr(pending_step, "phase", "") or "").upper() == "CARD_REWARD"
                and pending_action_kind == "card_reward"
                and (current_room_type == "NeowRoom" or getattr(pending_step, "floor", None) == 0)
                and expected_post_phase in {"NEOW", "CARD_REWARD", "MAP"}
            ):
                neow_card_reward_timeout = min(5.0, DEFAULT_REPLAY_ACTION_TIMEOUT_SECONDS)
                try:
                    _wait_for_any_update(
                        coordinator,
                        timeout_seconds=neow_card_reward_timeout,
                        context="recorded replay main loop after executing Neow card reward",
                    )
                except TimeoutError:
                    next_step = (
                        trace.steps[driver.step_index + 1]
                        if trace is not None and driver.step_index + 1 < driver.total_steps
                        else None
                    )
                    next_next_step = (
                        trace.steps[driver.step_index + 2]
                        if trace is not None and driver.step_index + 2 < driver.total_steps
                        else None
                    )
                    if (
                        expected_post_phase == "NEOW"
                        and next_step is not None
                        and next_step.phase == "NEOW"
                        and _is_neow_stale_card_reward_bridge_state(getattr(coordinator, "last_game_state", None))
                    ):
                        driver._force_finalize_pending_step(
                            live_snapshot=snapshot_live_state(getattr(coordinator, "last_game_state", None)),
                            comparison_note="neow_reward_pick_stale_card_reward_blind_continue_forced",
                        )
                        driver.neow_card_reward_blind_continue_step_number = next_step.step
                        if next_next_step is not None and next_next_step.phase == "MAP":
                            driver.neow_card_reward_blind_map_step_number = next_next_step.step
                    else:
                        raise
            elif (
                pending_step is not None
                and str(getattr(pending_step, "phase", "") or "").upper() == "CARD_SELECT"
                and pending_action_kind == "card_select"
                and current_room_type == "NeowRoom"
            ):
                neow_card_select_timeout = min(5.0, DEFAULT_REPLAY_ACTION_TIMEOUT_SECONDS)
                next_step = (
                    trace.steps[driver.step_index + 1]
                    if trace is not None and driver.step_index + 1 < driver.total_steps
                    else None
                )
                try:
                    _wait_for_any_update(
                        coordinator,
                        timeout_seconds=neow_card_select_timeout,
                        context="recorded replay main loop after executing Neow card select",
                    )
                    if (
                        next_step is not None
                        and next_step.phase == "NEOW"
                        and _is_neow_card_select_state(getattr(coordinator, "last_game_state", None))
                    ):
                        coordinator.clear_actions()
                        driver._force_finalize_pending_step(
                            live_snapshot=snapshot_live_state(getattr(coordinator, "last_game_state", None)),
                            comparison_note="neow_card_select_callback_timeout_forced",
                        )
                        driver.neow_card_select_continue_step_number = next_step.step
                        continue
                except TimeoutError:
                    if (
                        next_step is not None
                        and next_step.phase == "NEOW"
                        and _is_neow_card_select_state(getattr(coordinator, "last_game_state", None))
                    ):
                        coordinator.clear_actions()
                        driver._force_finalize_pending_step(
                            live_snapshot=snapshot_live_state(getattr(coordinator, "last_game_state", None)),
                            comparison_note="neow_card_select_callback_timeout_forced",
                        )
                        driver.neow_card_select_continue_step_number = next_step.step
                    else:
                        raise
            elif not coordinator.game_is_ready:
                pending_step = getattr(driver, "pending_step", None)
                last_game_state = getattr(coordinator, "last_game_state", None)
                current_step = (
                    trace.steps[driver.step_index]
                    if trace is not None and getattr(driver, "step_index", 0) < getattr(driver, "total_steps", 0)
                    else None
                )
                if (
                    pending_step is not None
                    and pending_step.phase == "MAP"
                    and _is_neow_map_bridge_state(last_game_state, pending_step)
                ):
                    # Once the post-Neow map bridge command has actually been
                    # sent, keep this transition on the callback-driven path.
                    # Forcing an immediate command-ready wait here tends to
                    # re-enter the stale single-option Neow frame and masks
                    # whether the queued bridge already advanced into MAP/room
                    # loading.
                    time.sleep(0.05)
                    continue
                if (
                    pending_step is None
                    and current_step is not None
                    and current_step.phase == "MAP"
                    and getattr(driver, "neow_leave_map_continue_step_number", None) == current_step.step
                    and _is_neow_map_bridge_state(last_game_state, current_step)
                ):
                    # Post-Neow MAP bridge setup actions run before the trace
                    # step itself becomes pending. Treat their immediate
                    # after-send frame the same way we treat a pending MAP
                    # bridge: stay on the callback-driven path instead of
                    # pivoting into a hard ready-state wait that only emits
                    # repeated `wait 1` pulses into the stale Leave frame.
                    time.sleep(0.05)
                    continue
                if (
                    pending_step is None
                    and current_step is not None
                    and current_step.phase == "MAP"
                    and getattr(driver, "event_leave_map_continue_step_number", None) == current_step.step
                    and _is_single_leave_event_state(last_game_state, current_step)
                ):
                    time.sleep(0.05)
                    continue
                if (
                    pending_step is not None
                    and pending_step.phase == "MAP"
                    and driver.neow_map_ready_pulse_step_number != pending_step.step
                    and last_game_state is not None
                    and getattr(last_game_state, "screen_type", None) == ScreenType.MAP
                    and str(getattr(last_game_state, "room_type", "") or "") == "NeowRoom"
                    and len(getattr(getattr(last_game_state, "screen", None), "next_nodes", []) or []) > 0
                ):
                    _send_probe_command(coordinator, "wait 1")
                    driver.neow_map_ready_pulse_step_number = pending_step.step
                    time.sleep(0.05)
                    continue
                if _should_skip_ready_wait_for_neow_blind_continue(coordinator, driver):
                    # The stale Neow reward bridge intentionally emits a
                    # single `wait 1` pulse first, then returns to the driver
                    # for the blind `choose 0` follow-up while the game is
                    # still non-ready. Do not block on a full ready-state wait
                    # here or the bridge never gets a chance to dispatch that
                    # second command.
                    time.sleep(0.05)
                    continue
                if _should_skip_ready_wait_for_neow_leave_map_continue(coordinator, driver):
                    # The first post-Neow map bridge may legitimately spend a
                    # full frame on the stale single-option Leave screen while
                    # `openMap()` runs and a blind first-node choice is
                    # already in flight. Yield back to callback polling rather
                    # than forcing a hard ready-state wait here.
                    time.sleep(0.05)
                    continue
                if _should_skip_ready_wait_for_event_leave_map_continue(coordinator, driver):
                    time.sleep(0.05)
                    continue
                try:
                    _wait_for_ready_state_with_timeout(
                        coordinator,
                        timeout_seconds=DEFAULT_REPLAY_ACTION_TIMEOUT_SECONDS,
                        context="recorded replay main loop after executing action",
                    )
                except TimeoutError:
                    if (
                        pending_step is not None
                        and pending_step.phase == "MAP"
                        and last_game_state is not None
                        and _is_neow_map_bridge_state(last_game_state, pending_step)
                        and (
                            getattr(driver, "neow_leave_map_continue_step_number", None) == pending_step.step
                            or _is_neow_map_bridge_state(last_game_state, pending_step)
                        )
                    ):
                        driver._force_finalize_pending_step(
                            live_snapshot=snapshot_live_state(last_game_state),
                            comparison_note="neow_leave_map_continue_blind_bridge_forced",
                        )
                        continue
                    if (
                        pending_step is not None
                        and pending_step.phase == "MAP"
                        and last_game_state is not None
                        and _is_single_leave_event_state(last_game_state, pending_step)
                        and getattr(driver, "event_leave_map_continue_step_number", None) == pending_step.step
                    ):
                        driver._force_finalize_pending_step(
                            live_snapshot=snapshot_live_state(last_game_state),
                            comparison_note="event_leave_map_continue_blind_bridge_forced",
                        )
                        continue
                    raise
            else:
                continue
        else:
            current_step = (
                trace.steps[driver.step_index]
                if trace is not None and getattr(driver, "step_index", 0) < getattr(driver, "total_steps", 0)
                else None
            )
            last_game_state = getattr(coordinator, "last_game_state", None)
            if _should_wait_for_any_update_for_post_neow_settle(
                driver,
                current_step,
                last_game_state,
            ):
                try:
                    _wait_for_any_update(
                        coordinator,
                        timeout_seconds=min(POST_NEOW_SETTLE_TIMEOUT_SECONDS, DEFAULT_REPLAY_ACTION_TIMEOUT_SECONDS),
                        context="recorded replay main loop awaiting post-neow combat settle update",
                    )
                except TimeoutError:
                    last_game_state = getattr(coordinator, "last_game_state", None)
                    if (
                        current_step is not None
                        and current_step.phase == "COMBAT"
                        and getattr(driver, "neow_post_map_settle_probe_step_number", None) == current_step.step
                        and last_game_state is not None
                        and _is_neow_map_bridge_state(last_game_state)
                    ):
                        coordinator.clear_actions()
                        driver.neow_post_map_settle_probe_step_number = None
                        driver._force_finalize_current_step(
                            current_step,
                            live_snapshot=snapshot_live_state(last_game_state),
                            comparison_note="neow_post_map_settle_callback_timeout_forced",
                        )
                        continue
                    driver.neow_post_map_settle_probe_step_number = None
                    raise
                continue
            pending_step = getattr(driver, "pending_step", None)
            last_game_state = getattr(coordinator, "last_game_state", None)
            current_step = (
                trace.steps[driver.step_index]
                if trace is not None and getattr(driver, "step_index", 0) < getattr(driver, "total_steps", 0)
                else None
            )
            try:
                callback_timeout_seconds = DEFAULT_REPLAY_ACTION_TIMEOUT_SECONDS
                if _should_use_short_post_neow_callback_timeout(
                    driver,
                    pending_step,
                    current_step,
                    last_game_state,
                ):
                    callback_timeout_seconds = min(
                        POST_NEOW_SETTLE_TIMEOUT_SECONDS,
                        DEFAULT_REPLAY_ACTION_TIMEOUT_SECONDS,
                    )
                _wait_for_callback_update_with_timeout(
                    coordinator,
                    timeout_seconds=callback_timeout_seconds,
                    context="recorded replay main loop awaiting callback update",
                )
            except TimeoutError:
                pending_step = getattr(driver, "pending_step", None)
                last_game_state = getattr(coordinator, "last_game_state", None)
                current_step = (
                    trace.steps[driver.step_index]
                    if trace is not None and getattr(driver, "step_index", 0) < getattr(driver, "total_steps", 0)
                    else None
                )
                next_step = (
                    trace.steps[driver.step_index + 1]
                    if trace is not None and driver.step_index + 1 < driver.total_steps
                    else None
                )
                next_next_step = (
                    trace.steps[driver.step_index + 2]
                    if trace is not None and driver.step_index + 2 < driver.total_steps
                    else None
                )
                active_neow_reward_step = None
                if (
                    pending_step is not None
                    and pending_step.phase == "CARD_REWARD"
                    and str(pending_step.action.get("kind") or "").lower() == "card_reward"
                ):
                    active_neow_reward_step = pending_step
                elif (
                    current_step is not None
                    and current_step.phase == "CARD_REWARD"
                    and str(current_step.action.get("kind") or "").lower() == "card_reward"
                ):
                    active_neow_reward_step = current_step
                if (
                    active_neow_reward_step is not None
                    and next_step is not None
                    and next_step.phase == "NEOW"
                    and last_game_state is not None
                    and (
                        _is_neow_stale_card_reward_bridge_state(last_game_state)
                        or _is_neow_leave_bridge_state(last_game_state)
                        or _is_neow_map_bridge_state(last_game_state)
                    )
                ):
                    coordinator.clear_actions()
                    if pending_step is not None and active_neow_reward_step is pending_step:
                        driver._force_finalize_pending_step(
                            live_snapshot=snapshot_live_state(last_game_state),
                            comparison_note="neow_reward_pick_stale_card_reward_blind_continue_forced",
                        )
                    else:
                        driver._force_finalize_current_step(
                            active_neow_reward_step,
                            live_snapshot=snapshot_live_state(last_game_state),
                            comparison_note="neow_reward_pick_stale_card_reward_blind_continue_forced",
                        )
                    driver.neow_card_reward_blind_continue_step_number = next_step.step
                    if next_next_step is not None and next_next_step.phase == "MAP":
                        driver.neow_card_reward_blind_map_step_number = next_next_step.step
                    continue
                active_neow_immediate_reward_step = None
                if (
                    pending_step is not None
                    and pending_step.phase == "NEOW"
                    and str(pending_step.action.get("kind") or "").lower() == "neow"
                    and getattr(driver, "pending_setup_actions", None)
                ):
                    active_neow_immediate_reward_step = pending_step
                elif (
                    current_step is not None
                    and current_step.phase == "NEOW"
                    and str(current_step.action.get("kind") or "").lower() == "neow"
                    and getattr(driver, "pending_setup_actions", None)
                ):
                    active_neow_immediate_reward_step = current_step
                if (
                    active_neow_immediate_reward_step is not None
                    and next_step is not None
                    and next_step.phase == "NEOW"
                    and last_game_state is not None
                    and _is_neow_leave_bridge_state(last_game_state)
                    and isinstance(active_neow_immediate_reward_step.post.get("deck"), list)
                ):
                    live_deck = list((snapshot_live_state(last_game_state).get("deck") or []))
                    expected_deck = list(active_neow_immediate_reward_step.post["deck"])
                    if _missing_expected_cards(expected_deck, live_deck):
                        coordinator.clear_actions()
                        driver._force_finalize_current_step(
                            active_neow_immediate_reward_step,
                            live_snapshot=dict(active_neow_immediate_reward_step.post),
                            comparison_note="neow_immediate_reward_callback_timeout_forced",
                        )
                        driver.neow_bridge_step_number = next_step.step
                        continue
                active_neow_card_select_step = None
                if (
                    pending_step is not None
                    and pending_step.phase == "CARD_SELECT"
                    and str(pending_step.action.get("kind") or "").lower() == "card_select"
                ):
                    active_neow_card_select_step = pending_step
                elif (
                    current_step is not None
                    and current_step.phase == "CARD_SELECT"
                    and str(current_step.action.get("kind") or "").lower() == "card_select"
                ):
                    active_neow_card_select_step = current_step
                if (
                    active_neow_card_select_step is not None
                    and next_step is not None
                    and next_step.phase == "NEOW"
                    and last_game_state is not None
                    and _is_neow_card_select_state(last_game_state)
                ):
                    coordinator.clear_actions()
                    if pending_step is not None and active_neow_card_select_step is pending_step:
                        driver._force_finalize_pending_step(
                            live_snapshot=snapshot_live_state(last_game_state),
                            comparison_note="neow_card_select_callback_timeout_forced",
                        )
                    else:
                        driver._force_finalize_current_step(
                            active_neow_card_select_step,
                            live_snapshot=snapshot_live_state(last_game_state),
                            comparison_note="neow_card_select_callback_timeout_forced",
                        )
                    driver.neow_card_select_continue_step_number = next_step.step
                    continue
                if (
                    pending_step is not None
                    and pending_step.phase == "NEOW"
                    and getattr(driver, "neow_card_select_continue_step_number", None) == pending_step.step
                    and last_game_state is not None
                    and _is_neow_card_select_state(last_game_state)
                ):
                    coordinator.clear_actions()
                    driver.neow_card_select_continue_step_number = None
                    driver._force_finalize_pending_step(
                        live_snapshot=snapshot_live_state(last_game_state),
                        comparison_note="neow_card_select_continue_callback_timeout_forced",
                    )
                    continue
                if (
                    pending_step is not None
                    and pending_step.phase == "NEOW"
                    and _normalize_phase_name(pending_step.post.get("phase")) == "MAP"
                    and last_game_state is not None
                    and str(getattr(last_game_state, "room_type", "") or "") == "NeowRoom"
                    and _single_neow_continue_action(last_game_state) is not None
                ):
                    coordinator.clear_actions()
                    driver._force_finalize_pending_step(
                        live_snapshot=snapshot_live_state(last_game_state),
                        comparison_note="neow_continue_leave_bridge_callback_timeout_forced",
                    )
                    continue
                if (
                    pending_step is None
                    and current_step is not None
                    and current_step.phase == "NEOW"
                    and getattr(driver, "neow_card_select_continue_step_number", None) == current_step.step
                    and last_game_state is not None
                    and _is_neow_card_select_state(last_game_state)
                ):
                    coordinator.clear_actions()
                    driver.neow_card_select_continue_step_number = None
                    driver._force_finalize_current_step(
                        current_step,
                        live_snapshot=snapshot_live_state(last_game_state),
                        comparison_note="neow_card_select_continue_callback_timeout_forced",
                    )
                    continue
                if (
                    pending_step is None
                    and current_step is not None
                    and current_step.phase == "NEOW"
                    and _normalize_phase_name(current_step.post.get("phase")) == "MAP"
                    and last_game_state is not None
                    and str(getattr(last_game_state, "room_type", "") or "") == "NeowRoom"
                    and _single_neow_continue_action(last_game_state) is not None
                ):
                    coordinator.clear_actions()
                    driver._force_finalize_current_step(
                        current_step,
                        live_snapshot=snapshot_live_state(last_game_state),
                        comparison_note="neow_continue_leave_bridge_callback_timeout_forced",
                    )
                    continue
                if (
                    pending_step is not None
                    and pending_step.phase == "MAP"
                    and last_game_state is not None
                    and _is_neow_map_bridge_state(last_game_state, pending_step)
                    and (
                        getattr(driver, "neow_leave_map_continue_step_number", None) == pending_step.step
                        or _is_neow_map_bridge_state(last_game_state, pending_step)
                    )
                ):
                    coordinator.clear_actions()
                    driver._force_finalize_pending_step(
                        live_snapshot=snapshot_live_state(last_game_state),
                        comparison_note="neow_leave_map_continue_callback_timeout_forced",
                    )
                    continue
                if (
                    pending_step is not None
                    and pending_step.phase == "MAP"
                    and last_game_state is not None
                    and _is_single_leave_event_state(last_game_state, pending_step)
                    and getattr(driver, "event_leave_map_continue_step_number", None) == pending_step.step
                ):
                    coordinator.clear_actions()
                    driver._force_finalize_pending_step(
                        live_snapshot=snapshot_live_state(last_game_state),
                        comparison_note="event_leave_map_continue_callback_timeout_forced",
                    )
                    continue
                if (
                    pending_step is None
                    and current_step is not None
                    and current_step.phase == "MAP"
                    and last_game_state is not None
                    and _is_neow_map_bridge_state(last_game_state, current_step)
                    and getattr(driver, "neow_leave_map_continue_step_number", None) == current_step.step
                ):
                    coordinator.clear_actions()
                    driver._force_finalize_current_step(
                        current_step,
                        live_snapshot=snapshot_live_state(last_game_state),
                        comparison_note="neow_leave_map_continue_setup_callback_timeout_forced",
                    )
                    continue
                if (
                    pending_step is None
                    and current_step is not None
                    and current_step.phase == "MAP"
                    and last_game_state is not None
                    and _is_single_leave_event_state(last_game_state, current_step)
                    and getattr(driver, "event_leave_map_continue_step_number", None) == current_step.step
                ):
                    coordinator.clear_actions()
                    driver._force_finalize_current_step(
                        current_step,
                        live_snapshot=snapshot_live_state(last_game_state),
                        comparison_note="event_leave_map_continue_setup_callback_timeout_forced",
                    )
                    continue
                if (
                    pending_step is not None
                    and pending_step.phase in {"CARD_SELECT", "COMBAT", "CARD_REWARD", "MAP", "SHOP", "EVENT", "CAMPFIRE", "TREASURE", "CHEST", "BOSS_RELIC"}
                    and last_game_state is not None
                    and _is_neow_map_bridge_state(last_game_state)
                    and getattr(driver, "neow_post_map_settle_probe_step_number", None) == pending_step.step
                ):
                    coordinator.clear_actions()
                    driver.neow_post_map_settle_probe_step_number = None
                    driver._force_finalize_pending_step(
                        live_snapshot=snapshot_live_state(last_game_state),
                        comparison_note="neow_post_map_settle_callback_timeout_forced",
                    )
                    continue
                if (
                    pending_step is not None
                    and pending_step.phase in {"CARD_SELECT", "COMBAT", "CARD_REWARD", "MAP", "SHOP", "EVENT", "CAMPFIRE", "TREASURE", "CHEST", "BOSS_RELIC"}
                    and last_game_state is not None
                    and _is_neow_card_select_state(last_game_state)
                    and getattr(driver, "neow_post_card_select_settle_probe_step_number", None) == pending_step.step
                ):
                    coordinator.clear_actions()
                    driver.neow_post_card_select_settle_probe_step_number = None
                    driver._force_finalize_pending_step(
                        live_snapshot=snapshot_live_state(last_game_state),
                        comparison_note="neow_post_card_select_settle_callback_timeout_forced",
                    )
                    continue
                if (
                    pending_step is None
                    and current_step is not None
                    and current_step.phase in {"CARD_SELECT", "COMBAT", "CARD_REWARD", "MAP", "SHOP", "EVENT", "CAMPFIRE", "TREASURE", "CHEST", "BOSS_RELIC"}
                    and last_game_state is not None
                    and _is_neow_map_bridge_state(last_game_state)
                    and getattr(driver, "neow_post_map_settle_probe_step_number", None) == current_step.step
                ):
                    coordinator.clear_actions()
                    driver.neow_post_map_settle_probe_step_number = None
                    driver._force_finalize_current_step(
                        current_step,
                        live_snapshot=snapshot_live_state(last_game_state),
                        comparison_note="neow_post_map_settle_callback_timeout_forced",
                    )
                    continue
                if (
                    pending_step is None
                    and current_step is not None
                    and current_step.phase in {"CARD_SELECT", "COMBAT", "CARD_REWARD", "MAP", "SHOP", "EVENT", "CAMPFIRE", "TREASURE", "CHEST", "BOSS_RELIC"}
                    and last_game_state is not None
                    and _is_neow_card_select_state(last_game_state)
                    and getattr(driver, "neow_post_card_select_settle_probe_step_number", None) == current_step.step
                ):
                    coordinator.clear_actions()
                    driver.neow_post_card_select_settle_probe_step_number = None
                    driver._force_finalize_current_step(
                        current_step,
                        live_snapshot=snapshot_live_state(last_game_state),
                        comparison_note="neow_post_card_select_settle_callback_timeout_forced",
                    )
                    continue
                raise

        if coordinator.last_error is not None and not _clear_tolerable_neow_skip_error(coordinator):
            raise RecordedReplayAbort(f"communication error: {coordinator.last_error}")

        if coordinator.in_game:
            out_of_game_since = None
            continue

        if out_of_game_since is None:
            out_of_game_since = time.time()

        if driver.step_index >= driver.total_steps:
            driver.done = True
            break

        while time.time() - out_of_game_since < DEFAULT_REPLAY_OUT_OF_GAME_GRACE_SECONDS:
            _wait_for_any_update(
                coordinator,
                timeout_seconds=1.0,
                context="recorded replay waiting for in-game resume",
            )
            if coordinator.last_error is not None and not _clear_tolerable_neow_skip_error(coordinator):
                raise RecordedReplayAbort(f"communication error: {coordinator.last_error}")
            if coordinator.in_game:
                out_of_game_since = None
                break
        if out_of_game_since is None:
            continue
        break


def replay_recorded_run(
    *,
    trace_path: str | Path,
    character: str | None = None,
    compare_state: bool = True,
    stop_on_mismatch: bool = True,
    max_steps: int | None = None,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    trace = load_recorded_trace(trace_path)
    progress_enabled = _env_flag("SPIRECOMM_REPLAY_VERBOSE", True)
    progress_target = progress_path or os.environ.get("SPIRECOMM_REPLAY_PROGRESS_PATH")
    progress_writer = _ReplayProgressWriter(progress_target) if progress_target else None
    coordinator = Coordinator()
    coordinator.stop_after_run = True
    player_class = _infer_character(trace, character)
    total_steps = len(trace.steps) if max_steps is None else min(len(trace.steps), max_steps)
    driver = _StateDrivenReplayDriver(
        trace=trace,
        compare_state=compare_state,
        stop_on_mismatch=stop_on_mismatch,
        total_steps=total_steps,
        progress_enabled=progress_enabled,
        progress_callback=progress_writer,
    )
    if progress_writer is not None:
        first_step = trace.steps[0] if total_steps > 0 else None
        driver._emit_progress(
            status="started",
            step=first_step,
            extra={
                "trace_path": str(Path(trace_path)),
                "character": player_class.name,
                "seed_str": trace.seed_str,
                "seed_long": trace.seed_long,
            },
        )
    agent = _RecordedReplayAgent(driver)
    if os.environ.get("SPIRECOMM_BOOTSTRAP_READY_SENT") != "1":
        coordinator.signal_ready()
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)
    start_seed = canonical_seed_string(trace.seed_long, trace.seed_str)
    try:
        _play_recorded_game(
            coordinator,
            player_class=player_class,
            ascension_level=trace.ascension,
            seed=start_seed,
            driver=driver,
        )
    except RecordedReplayAbort:
        pass

    final_state = serialize_game_state(coordinator.last_game_state) if coordinator.last_game_state is not None else None
    results = driver.results
    success = not driver.failed
    first_failure_step = driver.first_failure_step
    if any(result["pre_mismatches"] or result["post_mismatches"] for result in results):
        success = False
        if first_failure_step is None:
            first_failure_step = next(
                result["step"]
                for result in results
                if result["pre_mismatches"] or result["post_mismatches"]
            )
    if len(results) < total_steps:
        success = False
        if first_failure_step is None and len(results) < len(trace.steps):
            first_failure_step = trace.steps[len(results)].step

    if progress_writer is not None:
        current_step = trace.steps[min(driver.step_index, total_steps - 1)] if total_steps > 0 and driver.step_index < total_steps else None
        driver._emit_progress(
            status="complete",
            step=current_step,
            live_snapshot=snapshot_live_state(coordinator.last_game_state) if coordinator.last_game_state is not None else None,
            extra={
                "success": success,
                "first_failure_step": first_failure_step,
            },
        )

    return {
        "trace_path": str(Path(trace_path)),
        "source_format": trace.source_format,
        "seed_long": trace.seed_long,
        "seed_str": trace.seed_str,
        "ascension": trace.ascension,
        "character": player_class.name,
        "steps_total": len(trace.steps),
        "steps_replayed": len(results),
        "success": success,
        "first_failure_step": first_failure_step,
        "results": results,
        "final_state": final_state,
    }


def render_replay_report_summary(report: dict[str, Any]) -> str:
    lines = [
        "Recorded Replay Summary",
        f"trace_path: {report.get('trace_path')}",
        f"seed_str: {report.get('seed_str')}",
        f"seed_long: {report.get('seed_long')}",
        f"ascension: {report.get('ascension')}",
        f"character: {report.get('character')}",
        f"success: {report.get('success')}",
        f"steps: {report.get('steps_replayed')}/{report.get('steps_total')}",
        f"first_failure_step: {report.get('first_failure_step')}",
        "",
    ]
    first_failure = next(
        (
            result
            for result in report.get("results", [])
            if result.get("pre_mismatches") or result.get("post_mismatches")
        ),
        None,
    )
    if first_failure is not None:
        lines.append("First Failure")
        lines.append(f"  step: {first_failure.get('step')}")
        lines.append(f"  phase: {first_failure.get('phase')}")
        lines.append(f"  trace_action: {json.dumps(first_failure.get('trace_action') or {}, ensure_ascii=False, sort_keys=True)}")
        pre_mismatches = list(first_failure.get("pre_mismatches") or [])
        post_mismatches = list(first_failure.get("post_mismatches") or [])
        if pre_mismatches:
            lines.append("  pre_mismatches:")
            for mismatch in pre_mismatches:
                lines.append(f"    - {mismatch}")
        if post_mismatches:
            lines.append("  post_mismatches:")
            for mismatch in post_mismatches:
                lines.append(f"    - {mismatch}")
        lines.append("")
    final_state = report.get("final_state") or {}
    if final_state:
        lines.append("Final State")
        lines.append(f"  floor: {final_state.get('floor')}")
        lines.append(f"  hp: {final_state.get('current_hp')}/{final_state.get('max_hp')}")
        lines.append(f"  gold: {final_state.get('gold')}")
        screen = final_state.get("screen")
        if isinstance(screen, dict):
            lines.append(f"  screen_type: {screen.get('screen_type')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
