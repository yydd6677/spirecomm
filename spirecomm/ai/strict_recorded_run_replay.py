from __future__ import annotations

import copy
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spirecomm.ai.recording import serialize_game_state
from spirecomm.ai.strict_trace import STRICT_TRACE_SCHEMA, normalize_verbose_state_for_strict
from spirecomm.communication.coordinator import Coordinator
from spirecomm.seed_helper import canonical_seed_string
from spirecomm.spire.character import PlayerClass


DEFAULT_STRICT_REPLAY_ACTION_TIMEOUT_SECONDS = 15.0
DEFAULT_STRICT_REPLAY_READY_TIMEOUT_SECONDS = 10.0
STRICT_REPLAY_OBSERVE_WAIT_INTERVAL_SECONDS = 0.25
STRICT_REPLAY_POST_MATCH_SETTLE_SECONDS = 0.35
STRICT_REPLAY_GAMEPLAY_COMMAND_SETTLE_SECONDS = 0.5
STRICT_REPLAY_VISIBLE_SCREEN_ACTION_SETTLE_SECONDS = 0.25
STRICT_REPLAY_NEOW_TRANSITION_SETTLE_SECONDS = 1.0
STRICT_REPLAY_NEOW_TRANSITION_MATCH_TIMEOUT_SECONDS = 1.5
STRICT_REPLAY_COMMAND_JOURNAL_ENV = "SPIRECOMM_STRICT_COMMAND_JOURNAL_PATH"
STRICT_REPLAY_COMMAND_QUEUE_DIR_ENV = "SPIRECOMM_STRICT_COMMAND_QUEUE_DIR"
STRICT_REPLAY_COMMAND_TRANSPORT_ENV = "SPIRECOMM_STRICT_COMMAND_TRANSPORT"
STRICT_REPLAY_PAUSE_ON_DIVERGENCE_ENV = "SPIRECOMM_STRICT_PAUSE_ON_DIVERGENCE"
STRICT_REPLAY_PAUSE_MANIFEST_ENV = "SPIRECOMM_STRICT_PAUSE_MANIFEST_PATH"
STRICT_REPLAY_RESUME_REQUEST_ENV = "SPIRECOMM_STRICT_RESUME_REQUEST_PATH"
STRICT_REPLAY_RESUME_RESULT_ENV = "SPIRECOMM_STRICT_RESUME_RESULT_PATH"
STRICT_REPLAY_PAUSE_POLL_INTERVAL_SECONDS = 1.0
_STRICT_COMMAND_SEQUENCE = 0


@dataclass
class StrictRecordedTraceStep:
    step: int
    phase: str
    floor: int | None
    action: dict[str, Any]
    strict_action: dict[str, Any]
    strict_pre_state: dict[str, Any]
    strict_post_state: dict[str, Any]
    post_phase: str | None = None


@dataclass
class StrictRecordedRunTrace:
    path: Path
    trace_schema: str
    seed_long: int | None
    seed_str: str | None
    ascension: int
    character: str | None
    steps: list[StrictRecordedTraceStep]
    trace_policy: str | None = None
    model_required: bool | None = None


class UnsupportedStrictTrace(RuntimeError):
    pass


@dataclass
class _PauseResumeOutcome:
    trace: StrictRecordedRunTrace | None = None
    next_step_index: int | None = None
    report: dict[str, Any] | None = None


_STRICT_VISIBLE_SCREEN_PHASES = {
    "CARD_REWARD",
    "CARD_SELECT",
    "EVENT",
    "MAP",
    "SHOP",
    "CAMPFIRE",
    "TREASURE",
    "CHEST",
    "BOSS_RELIC",
}


def _should_delay_before_strict_action(step: StrictRecordedTraceStep) -> bool:
    return step.phase in (_STRICT_VISIBLE_SCREEN_PHASES - {"CARD_REWARD"})


def _is_visible_screen_phase(phase: str | None) -> bool:
    return str(phase or "") in _STRICT_VISIBLE_SCREEN_PHASES


def _post_match_settle_seconds_for_phase(phase: str | None) -> float:
    if _is_visible_screen_phase(phase):
        return 0.05
    return STRICT_REPLAY_POST_MATCH_SETTLE_SECONDS


def _gameplay_command_settle_seconds_for_step(step: StrictRecordedTraceStep) -> float:
    if _is_visible_screen_phase(step.phase) or _is_visible_screen_phase(
        step.post_phase or step.strict_post_state.get("phase")
    ):
        return max(
            STRICT_REPLAY_GAMEPLAY_COMMAND_SETTLE_SECONDS,
            STRICT_REPLAY_VISIBLE_SCREEN_ACTION_SETTLE_SECONDS,
        )
    return STRICT_REPLAY_GAMEPLAY_COMMAND_SETTLE_SECONDS


@dataclass
class _StateTapFollower:
    path: Path
    offset: int = 0
    pending_fragment: str = ""

    def ingest_updates(self, coordinator: Coordinator) -> list[dict[str, Any]]:
        ingested: list[dict[str, Any]] = []
        if not self.path.exists():
            return ingested
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        if not chunk:
            return ingested
        data = self.pending_fragment + chunk
        self.pending_fragment = ""
        for raw_line in data.splitlines(keepends=True):
            if not raw_line.endswith("\n") and not raw_line.endswith("\r"):
                self.pending_fragment = raw_line
                continue
            line = raw_line.rstrip("\r\n")
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            communication_state = payload.get("communication_state")
            source = payload.get("source", "state_tap")
            if isinstance(communication_state, dict):
                payload = communication_state
            coordinator.ingest_communication_state(
                payload,
                raw_message=stripped,
                source=str(source),
            )
            ingested.append(
                {
                    "sequence": coordinator.raw_message_sequence,
                    "raw_message": stripped,
                    "communication_state": payload,
                    "source": str(source),
                }
            )
        return ingested


def _iter_state_tap_updates(
    coordinator: Coordinator,
    state_tap: _StateTapFollower | None,
) -> list[dict[str, Any]]:
    if state_tap is None:
        return []
    return state_tap.ingest_updates(coordinator)


def _restore_matched_state_tap_frame(coordinator: Coordinator, tap_event: dict[str, Any]) -> None:
    communication_state = dict(tap_event.get("communication_state") or {})
    if not communication_state:
        return
    source = str(tap_event.get("source") or "state_tap")
    coordinator.ingest_communication_state(
        communication_state,
        raw_message=tap_event.get("raw_message"),
        source=f"{source}_strict_match",
    )


def _drain_pending_updates(
    coordinator: Coordinator,
    state_tap: _StateTapFollower | None,
    *,
    settle_seconds: float,
) -> None:
    deadline = time.time() + max(0.0, settle_seconds)
    while time.time() < deadline:
        observed = False
        if _iter_state_tap_updates(coordinator, state_tap):
            observed = True
        if coordinator.receive_game_state_update(block=False, perform_callbacks=False):
            observed = True
        if not observed:
            time.sleep(0.05)


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


def _env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _strict_command_journal_path() -> Path | None:
    raw = os.environ.get(STRICT_REPLAY_COMMAND_JOURNAL_ENV)
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return Path(raw)


def _strict_command_transport() -> str:
    raw = os.environ.get(STRICT_REPLAY_COMMAND_TRANSPORT_ENV, "stdout")
    transport = raw.strip().lower()
    if not transport:
        return "stdout"
    return transport


def _is_strict_observation_command(command: str) -> bool:
    normalized = command.strip().lower()
    return normalized == "state" or normalized == "wait 1"


def _append_strict_command_journal(command: str) -> None:
    journal_path = _strict_command_journal_path()
    if journal_path is None:
        return
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(command)
        handle.write("\n")


def _strict_command_queue_dir() -> Path | None:
    raw = os.environ.get(STRICT_REPLAY_COMMAND_QUEUE_DIR_ENV)
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return Path(raw)


def _enqueue_strict_command_file(command: str) -> None:
    global _STRICT_COMMAND_SEQUENCE
    queue_dir = _strict_command_queue_dir()
    if queue_dir is None:
        return
    queue_dir.mkdir(parents=True, exist_ok=True)
    _STRICT_COMMAND_SEQUENCE += 1
    final_path = queue_dir / f"{_STRICT_COMMAND_SEQUENCE:08d}.cmd"
    temp_path = queue_dir / f".{_STRICT_COMMAND_SEQUENCE:08d}.tmp"
    temp_path.write_text(command + "\n", encoding="utf-8")
    os.replace(temp_path, final_path)


def _strict_debug_log_path(raw_state_log_path: Path) -> Path:
    return raw_state_log_path.with_name(raw_state_log_path.stem + ".debug.jsonl")


def _append_strict_debug(raw_state_log_path: Path, payload: dict[str, Any]) -> None:
    _append_jsonl(
        _strict_debug_log_path(raw_state_log_path),
        {
            "timestamp": time.time(),
            **payload,
        },
    )


def _send_strict_command(coordinator: Coordinator, command: str) -> None:
    """Synchronously send a strict replay command.

    Strict replay treats command dispatch as part of the audited surface.
    Journal/file-queue transports are audit and alternate-delivery helpers,
    not replacements for the coordinator's primary writer lane unless
    explicitly requested. The default strict path therefore logs commands to
    the audit journal and still sends them through the coordinator.
    """

    normalized = command.strip().lower()
    if normalized == "ready":
        coordinator.send_message(command)
        return
    transport = _strict_command_transport()
    if normalized.startswith("start "):
        send_immediate = getattr(coordinator, "send_message_immediate", None)
        if callable(send_immediate):
            send_immediate(command)
        else:
            coordinator.send_message(command)
        coordinator.game_is_ready = False
        return
    if _is_strict_observation_command(command):
        coordinator.send_message(command)
        coordinator.game_is_ready = False
        return
    if transport == "journal":
        journal_path = _strict_command_journal_path()
        if journal_path is not None:
            _append_strict_command_journal(command)
        coordinator.game_is_ready = False
        return
    if transport == "file_queue":
        queue_dir = _strict_command_queue_dir()
        if queue_dir is not None:
            _enqueue_strict_command_file(command)
            coordinator.game_is_ready = False
            return
    if transport in {"stdout_immediate", "immediate", "stdout"}:
        send_immediate = getattr(coordinator, "send_message_immediate", None)
        if callable(send_immediate):
            send_immediate(command)
        else:
            coordinator.send_message(command)
        coordinator.game_is_ready = False
        return
    coordinator.send_message(command)
    coordinator.game_is_ready = False


def _queue_strict_gameplay_command(
    coordinator: Coordinator,
    command: str,
    *,
    live_pre_state: dict[str, Any] | None = None,
    prefer_immediate: bool = False,
) -> None:
    """Synchronously send a strict replay gameplay command.

    Audited gameplay actions should use the same plain coordinator command
    path a normal replay session would use. Strict replay keeps alternate
    transport plumbing for observation commands, but gameplay choices
    themselves stay on the ordinary `choose` lane.
    """
    transport = _strict_command_transport()
    if transport == "journal":
        journal_path = _strict_command_journal_path()
        if journal_path is not None:
            _append_strict_command_journal(command)
            coordinator.game_is_ready = False
            return
    if transport == "file_queue":
        queue_dir = _strict_command_queue_dir()
        if queue_dir is not None:
            _enqueue_strict_command_file(command)
            coordinator.game_is_ready = False
            return
    if prefer_immediate or transport in {"stdout_immediate", "immediate"}:
        send_immediate = getattr(coordinator, "send_message_immediate", None)
        if callable(send_immediate):
            send_immediate(command)
        else:
            coordinator.send_message(command)
        coordinator.game_is_ready = False
        return
    coordinator.send_message(command)
    coordinator.game_is_ready = False


def _state_tap_follower_from_env() -> _StateTapFollower | None:
    raw = os.environ.get("SPIRECOMM_STRICT_STATE_TAP_PATH")
    if not raw:
        return None
    return _StateTapFollower(path=Path(raw))


def _card_metadata_key(card: dict[str, Any]) -> tuple[Any, ...]:
    return (
        card.get("card_id") or card.get("id") or card.get("name"),
        int(card.get("upgrades", 0) or 0),
        card.get("type"),
        card.get("rarity"),
        card.get("cost"),
        bool(card.get("has_target")) if card.get("has_target") is not None else None,
    )


def _canonicalize_loaded_strict_state_metadata(
    strict_state: dict[str, Any],
    raw_state: Any,
) -> dict[str, Any]:
    if not isinstance(raw_state, dict):
        return dict(strict_state or {})
    if str((strict_state or {}).get("phase") or raw_state.get("phase") or "").upper() != "COMBAT":
        return dict(strict_state or {})
    raw_combat = raw_state.get("combat_state")
    strict_combat = (strict_state or {}).get("combat_state")
    if not isinstance(raw_combat, dict) or not isinstance(strict_combat, dict):
        return dict(strict_state or {})
    raw_ethereal_exhaust_keys = {
        _card_metadata_key(card)
        for card in list(raw_combat.get("exhaust_pile") or [])
        if isinstance(card, dict) and bool(card.get("ethereal"))
    }
    if not raw_ethereal_exhaust_keys:
        return dict(strict_state or {})
    result = copy.deepcopy(strict_state or {})
    result_combat = result.get("combat_state")
    if not isinstance(result_combat, dict):
        return result
    for card in list(result_combat.get("exhaust_pile") or []):
        if isinstance(card, dict) and _card_metadata_key(card) in raw_ethereal_exhaust_keys:
            card["exhausts"] = True
    return result


def load_strict_recorded_trace(path: str | Path) -> StrictRecordedRunTrace:
    trace_path = Path(path)
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_schema = payload.get("trace_schema")
    if trace_schema != STRICT_TRACE_SCHEMA:
        raise UnsupportedStrictTrace(
            f"strict replay only accepts {STRICT_TRACE_SCHEMA}; got {trace_schema!r}"
        )
    steps = []
    for raw_step in payload.get("steps", []):
        phase = str(raw_step["phase"])
        post_phase = raw_step.get("post_phase")
        raw_pre_state = raw_step.get("pre_state")
        raw_post_state = raw_step.get("post_state")
        strict_pre_state = _canonicalize_loaded_strict_state_metadata(
            dict(raw_step.get("strict_pre_state") or {}),
            raw_pre_state,
        )
        strict_post_state = _canonicalize_loaded_strict_state_metadata(
            dict(raw_step.get("strict_post_state") or {}),
            raw_post_state,
        )
        steps.append(
            StrictRecordedTraceStep(
                step=int(raw_step["step"]),
                phase=phase,
                floor=raw_step.get("floor"),
                action=dict(raw_step.get("action") or {}),
                strict_action=dict(raw_step.get("strict_action") or {}),
                strict_pre_state=strict_pre_state,
                strict_post_state=strict_post_state,
                post_phase=post_phase,
            )
        )
    return StrictRecordedRunTrace(
        path=trace_path,
        trace_schema=str(trace_schema),
        seed_long=payload.get("seed_long"),
        seed_str=payload.get("seed_str"),
        ascension=int(payload.get("ascension", 0) or 0),
        character=payload.get("character"),
        steps=steps,
        trace_policy=payload.get("trace_policy"),
        model_required=payload.get("model_required"),
    )


def _strict_action_prefix_entry(
    step: StrictRecordedTraceStep,
    *,
    action_kind: str | None = None,
    commands: list[str] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "step": int(step.step),
        "phase": str(step.phase),
        "strict_action": copy.deepcopy(step.strict_action),
    }
    if action_kind is not None:
        entry["action_kind"] = action_kind
    if commands is not None:
        entry["commands"] = list(commands)
    return entry


def _strict_action_prefix_core(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": int(entry.get("step", -1)),
        "phase": str(entry.get("phase") or ""),
        "strict_action": copy.deepcopy(entry.get("strict_action") or {}),
    }


def _trace_action_prefix_core(
    trace: StrictRecordedRunTrace,
    *,
    next_step_index: int,
) -> list[dict[str, Any]]:
    return [
        _strict_action_prefix_core(_strict_action_prefix_entry(step))
        for step in trace.steps[:next_step_index]
    ]


def _manifest_action_prefix_core(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _strict_action_prefix_core(dict(entry or {}))
        for entry in list(manifest.get("action_prefix") or [])
    ]


def validate_strict_pause_resume_trace(
    manifest: dict[str, Any],
    trace: StrictRecordedRunTrace,
) -> dict[str, Any]:
    """Validate that a regenerated strict trace can resume a paused live run."""

    try:
        next_step_index = int(manifest.get("next_step_to_send"))
    except (TypeError, ValueError):
        return {"ok": False, "reason": "invalid_next_step_to_send"}

    if trace.seed_long != manifest.get("seed_long"):
        return {
            "ok": False,
            "reason": "seed_mismatch",
            "manifest_seed_long": manifest.get("seed_long"),
            "trace_seed_long": trace.seed_long,
        }
    if int(trace.ascension) != int(manifest.get("ascension", 0) or 0):
        return {
            "ok": False,
            "reason": "ascension_mismatch",
            "manifest_ascension": manifest.get("ascension"),
            "trace_ascension": trace.ascension,
        }
    manifest_trace_policy = manifest.get("trace_policy")
    if manifest_trace_policy is not None and trace.trace_policy is not None and trace.trace_policy != manifest_trace_policy:
        return {
            "ok": False,
            "reason": "trace_policy_mismatch",
            "manifest_trace_policy": manifest_trace_policy,
            "trace_policy": trace.trace_policy,
        }
    manifest_character = str(manifest.get("character") or "").upper()
    if trace.character is not None and str(trace.character).upper() != manifest_character:
        return {
            "ok": False,
            "reason": "character_mismatch",
            "manifest_character": manifest.get("character"),
            "trace_character": trace.character,
        }
    if next_step_index < 0 or next_step_index > len(trace.steps):
        return {
            "ok": False,
            "reason": "resume_step_out_of_range",
            "next_step_to_send": next_step_index,
            "steps_total": len(trace.steps),
        }

    manifest_prefix = _manifest_action_prefix_core(manifest)
    trace_prefix = _trace_action_prefix_core(trace, next_step_index=next_step_index)
    if trace_prefix != manifest_prefix:
        return {
            "ok": False,
            "reason": "action_prefix_mismatch",
            "next_step_to_send": next_step_index,
            "manifest_prefix_length": len(manifest_prefix),
            "trace_prefix_length": len(trace_prefix),
        }

    expected_live_state = copy.deepcopy(manifest.get("actual_visible"))
    if next_step_index == len(trace.steps):
        trace_post_state = copy.deepcopy(trace.steps[-1].strict_post_state) if trace.steps else None
        if trace_post_state != expected_live_state:
            return {
                "ok": False,
                "reason": "terminal_live_state_mismatch",
                "next_step_to_send": next_step_index,
                "trace_post_state": trace_post_state,
                "actual_visible": expected_live_state,
            }
        return {
            "ok": True,
            "reason": None,
            "next_step_to_send": next_step_index,
            "terminal_resume": True,
        }

    trace_pre_state = copy.deepcopy(trace.steps[next_step_index].strict_pre_state)
    if trace_pre_state != expected_live_state:
        if _is_shop_purge_card_select_resume_boundary(manifest, expected_live_state, trace_pre_state):
            return {
                "ok": True,
                "reason": "shop_purge_card_select_resume_boundary",
                "next_step_to_send": next_step_index,
                "next_trace_step": trace.steps[next_step_index].step,
            }
        return {
            "ok": False,
            "reason": "live_state_mismatch",
            "next_step_to_send": next_step_index,
            "trace_pre_state": trace_pre_state,
            "actual_visible": expected_live_state,
        }

    return {
        "ok": True,
        "reason": None,
        "next_step_to_send": next_step_index,
        "next_trace_step": trace.steps[next_step_index].step,
    }


def _infer_character(trace: StrictRecordedRunTrace, override: str | None) -> PlayerClass:
    if override:
        return PlayerClass[override.upper()]
    if trace.character:
        return PlayerClass[str(trace.character).upper()]
    return PlayerClass.IRONCLAD


def _raw_message_logger(raw_state_log_path: Path):
    def _callback(payload: dict[str, Any]) -> None:
        state = dict(payload.get("communication_state") or {})
        _append_jsonl(
            raw_state_log_path,
            {
                "sequence": payload.get("sequence"),
                "source": payload.get("source"),
                "timestamp": time.time(),
                "ready_for_command": state.get("ready_for_command"),
                "in_game": state.get("in_game"),
                "error": state.get("error"),
                "available_commands": state.get("available_commands"),
                "game_state": state.get("game_state"),
            },
        )

    return _callback


def _live_strict_state(coordinator: Coordinator, *, phase_hint: str | None = None) -> dict[str, Any]:
    return normalize_verbose_state_for_strict(
        getattr(coordinator, "last_raw_game_state", None),
        phase_hint=phase_hint,
    )


def _observed_strict_state(coordinator: Coordinator) -> dict[str, Any]:
    return normalize_verbose_state_for_strict(getattr(coordinator, "last_raw_game_state", None))


def _choice_items(state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(state, dict):
        return []
    return list((state.get("screen_state") or {}).get("choices") or [])


def _is_shop_screen_state(state: dict[str, Any] | None) -> bool:
    if not isinstance(state, dict):
        return False
    return str(state.get("phase") or "") == "SHOP" and str(state.get("screen_type") or "") == "SHOP"


def _card_identity_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_id = str(left.get("card_id") or left.get("id") or "")
    right_id = str(right.get("card_id") or right.get("id") or "")
    if left_id and right_id and left_id != right_id:
        return False
    if int(left.get("upgrades") or 0) != int(right.get("upgrades") or 0):
        return False
    left_name = str(left.get("name") or "")
    right_name = str(right.get("name") or "")
    return not left_name or not right_name or left_name == right_name


def _card_select_choice_index_for_deck_target(live_state: dict[str, Any], target_index: int) -> int | None:
    """Map a trace deck index to the visible grid choice index.

    Shop purge traces store the target as a master-deck index. STS opens a grid
    that omits unremovable cards such as bottled cards, so the grid choice index
    can be lower than the deck index.
    """
    deck = list(live_state.get("deck") or [])
    if target_index < 0 or target_index >= len(deck):
        return None
    choices = list((live_state.get("screen_state") or {}).get("choices") or [])
    choice_cursor = 0
    for deck_index, deck_card_raw in enumerate(deck):
        if choice_cursor >= len(choices):
            return None
        choice = choices[choice_cursor]
        if not isinstance(choice, dict):
            return None
        choice_card = choice.get("card")
        if not isinstance(choice_card, dict):
            return None
        deck_card = dict(deck_card_raw or {})
        if not _card_identity_matches(deck_card, choice_card):
            if deck_index == target_index:
                return None
            continue
        if deck_index == target_index:
            return int(choice.get("choice_index", choice_cursor))
        choice_cursor += 1
    return None


def _manifest_last_shop_purge_target_index(manifest: dict[str, Any]) -> int | None:
    for entry in reversed(list(manifest.get("action_prefix") or [])):
        if not isinstance(entry, dict):
            continue
        action = dict(entry.get("strict_action") or {})
        if (
            str(entry.get("phase") or "") == "SHOP"
            and str(action.get("kind") or "") == "choose_by_name"
            and str(action.get("item_kind") or "").lower() == "purge"
            and action.get("target_index") is not None
        ):
            return int(action["target_index"])
    return None


def _is_shop_purge_card_select_resume_boundary(
    manifest: dict[str, Any],
    actual_visible: dict[str, Any] | None,
    trace_pre_state: dict[str, Any] | None,
) -> bool:
    if not isinstance(actual_visible, dict) or not isinstance(trace_pre_state, dict):
        return False
    if str(actual_visible.get("phase") or "") != "CARD_SELECT":
        return False
    if not bool((actual_visible.get("screen_state") or {}).get("for_purge")):
        return False
    if str(trace_pre_state.get("phase") or "") != "SHOP":
        return False
    target_index = _manifest_last_shop_purge_target_index(manifest)
    if target_index is None:
        return False
    return _card_select_choice_index_for_deck_target(actual_visible, target_index) is not None


def _shop_state_without_inventory_choices(state: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(state)
    screen_state = normalized.get("screen_state")
    if isinstance(screen_state, dict):
        # The Courier restocks purchased shop slots immediately. Native trace generation
        # can disagree with live STS on that replacement while the actual purchase result
        # (gold/deck/relics/potions) is already aligned. Keep the rest of shop metadata.
        screen_state["choices"] = []
    return normalized


def _card_reward_state_without_sapphire_choice(state: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(state)
    screen_state = normalized.get("screen_state")
    if not isinstance(screen_state, dict):
        return normalized
    choices = list(screen_state.get("choices") or [])
    filtered: list[dict[str, Any]] = []
    removed = False
    for choice in choices:
        if isinstance(choice, dict) and str(choice.get("kind") or "") == "reward_key" and str(choice.get("key") or "") == "sapphire":
            removed = True
            continue
        filtered.append(dict(choice) if isinstance(choice, dict) else choice)
    if not removed:
        return normalized
    for index, choice in enumerate(filtered):
        if isinstance(choice, dict):
            choice["choice_index"] = index
    screen_state["choices"] = filtered
    return normalized


def _card_reward_sapphire_choice_surface_matches(expected_state: dict[str, Any], live_state: dict[str, Any]) -> bool:
    if str(expected_state.get("phase") or "") != "CARD_REWARD" or str(live_state.get("phase") or "") != "CARD_REWARD":
        return False
    expected_choices = _choice_items(expected_state)
    live_choices = _choice_items(live_state)
    if not any(
        str(choice.get("kind") or "") == "reward_key" and str(choice.get("key") or "") == "sapphire"
        for choice in expected_choices
        if isinstance(choice, dict)
    ):
        return False
    if any(
        str(choice.get("kind") or "") == "reward_key" and str(choice.get("key") or "") == "sapphire"
        for choice in live_choices
        if isinstance(choice, dict)
    ):
        return False
    return _card_reward_state_without_sapphire_choice(expected_state) == live_state


def _card_select_confirm_state_without_choices(state: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(state)
    screen_state = normalized.get("screen_state")
    if isinstance(screen_state, dict):
        screen_state["choices"] = []
    return normalized


def _card_select_confirm_surface_matches(expected_state: dict[str, Any], live_state: dict[str, Any]) -> bool:
    if str(expected_state.get("phase") or "") != "CARD_SELECT" or str(live_state.get("phase") or "") != "CARD_SELECT":
        return False
    expected_screen = expected_state.get("screen_state")
    live_screen = live_state.get("screen_state")
    if not isinstance(expected_screen, dict) or not isinstance(live_screen, dict):
        return False
    if not bool(expected_screen.get("confirm_up")) or not bool(live_screen.get("confirm_up")):
        return False
    # STS hand/grid confirm screens may remove the selected card from the visible
    # choice list while v3 keeps the full source list. The following confirm step
    # validates the selected-card effect, so choices are only presentation here.
    return _card_select_confirm_state_without_choices(expected_state) == _card_select_confirm_state_without_choices(live_state)


def _strict_state_matches(
    expected_state: dict[str, Any],
    live_state: dict[str, Any],
    *,
    allow_shop_inventory_mismatch: bool = False,
) -> bool:
    if live_state == expected_state:
        return True
    if _card_reward_sapphire_choice_surface_matches(expected_state, live_state):
        return True
    if _card_select_confirm_surface_matches(expected_state, live_state):
        return True
    if allow_shop_inventory_mismatch and _is_shop_screen_state(expected_state) and _is_shop_screen_state(live_state):
        return _shop_state_without_inventory_choices(live_state) == _shop_state_without_inventory_choices(expected_state)
    return False


def _visible_neow_choice_ui_family(state: dict[str, Any] | None) -> str | None:
    if not isinstance(state, dict):
        return None
    if str(state.get("room_type") or "") != "NeowRoom":
        return None
    phase = str(state.get("phase") or "").upper()
    screen_type = str(state.get("screen_type") or "").upper()
    choices = _choice_items(state)
    if not choices:
        return None
    if phase == "NEOW" and screen_type == "EVENT":
        if len(choices) <= 1:
            return None
        if all(str((choice or {}).get("kind") or "").lower() == "neow" for choice in choices):
            return "neow_event"
        return None
    if phase == "CARD_REWARD" and screen_type == "CARD_REWARD":
        return "neow_card_reward"
    if phase == "CARD_SELECT" and screen_type in {"GRID", "CARD_SELECT"}:
        return "neow_card_select"
    return None


def _is_visible_single_neow_transition_choice(state: dict[str, Any] | None) -> bool:
    if not isinstance(state, dict):
        return False
    if str(state.get("room_type") or "") != "NeowRoom":
        return False
    if str(state.get("phase") or "").upper() != "NEOW":
        return False
    if str(state.get("screen_type") or "").upper() != "EVENT":
        return False
    choices = _choice_items(state)
    if len(choices) != 1:
        return False
    choice = dict(choices[0] or {})
    if str(choice.get("kind") or "").lower() != "neow":
        return False
    bonus = choice.get("bonus")
    return bonus is None or str(bonus).upper() == "CONTINUE"


def _advance_visible_neow_transition(
    coordinator: Coordinator,
    state_tap: _StateTapFollower | None,
    *,
    expected_state: dict[str, Any],
    timeout_seconds: float,
    debug_log_path: Path,
    step: int,
) -> tuple[list[str], dict[str, Any] | None]:
    if _visible_neow_choice_ui_family(expected_state) != "neow_event":
        return [], None

    commands: list[str] = []
    deadline = time.time() + max(0.0, timeout_seconds)
    max_transition_commands = 4
    settle_seconds = max(
        STRICT_REPLAY_GAMEPLAY_COMMAND_SETTLE_SECONDS,
        STRICT_REPLAY_VISIBLE_SCREEN_ACTION_SETTLE_SECONDS,
        STRICT_REPLAY_NEOW_TRANSITION_SETTLE_SECONDS,
    )

    while len(commands) < max_transition_commands and time.time() < deadline:
        live_state = _observed_strict_state(coordinator)
        if live_state == expected_state and coordinator.game_is_ready:
            return commands, live_state
        if not coordinator.game_is_ready:
            if not _wait_for_ready_state(
                coordinator,
                state_tap,
                timeout_seconds=min(1.0, max(0.0, deadline - time.time())),
                context=f"strict replay neow transition step {step}",
            ):
                return commands, live_state
            live_state = _observed_strict_state(coordinator)
            if live_state == expected_state and coordinator.game_is_ready:
                return commands, live_state
        if not _is_visible_single_neow_transition_choice(live_state):
            return commands, live_state

        before_sequence = coordinator.raw_message_sequence
        _append_strict_debug(
            debug_log_path,
            {
                "event": "neow_transition_command",
                "step": step,
                "raw_message_sequence_before_action": before_sequence,
                "command": "choose 0",
                "live_state": live_state,
            },
        )
        _queue_strict_gameplay_command(
            coordinator,
            "choose 0",
            live_pre_state=live_state,
        )
        commands.append("choose 0")
        time.sleep(settle_seconds)
        matched_expected, saw_fresh_expected, live_after = _wait_for_expected_post_state(
            coordinator,
            state_tap,
            after_sequence=before_sequence,
            expected_state=expected_state,
            phase_hint=str(expected_state.get("phase") or "NEOW"),
            timeout_seconds=min(
                STRICT_REPLAY_NEOW_TRANSITION_MATCH_TIMEOUT_SECONDS,
                max(0.5, deadline - time.time()),
            ),
            require_ready=True,
            context=f"strict replay neow transition step {step}",
            initial_state_probe_delay=0.0,
            initial_wait_probe_delay=settle_seconds * 2.0,
        )
        if matched_expected:
            _append_strict_debug(
                debug_log_path,
                {
                    "event": "neow_transition_expected_state_reached",
                    "step": step,
                    "commands": commands,
                    "saw_fresh_state": saw_fresh_expected,
                    "live_state": live_after,
                },
            )
            return commands, live_after
        if live_after is not None:
            _append_strict_debug(
                debug_log_path,
                {
                    "event": "neow_transition_expected_state_not_reached",
                    "step": step,
                    "commands": commands,
                    "saw_fresh_state": saw_fresh_expected,
                    "live_state": live_after,
                },
            )

    return commands, _observed_strict_state(coordinator)


def _wait_for_fresh_matching_state(
    coordinator: Coordinator,
    state_tap: _StateTapFollower | None,
    *,
    expected_state: dict[str, Any],
    phase_hint: str,
    timeout_seconds: float,
    allow_state_probe: bool = True,
    allow_wait_probe: bool = True,
    require_ready: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    matched, _saw_fresh, actual = _wait_for_expected_post_state(
        coordinator,
        state_tap,
        after_sequence=coordinator.raw_message_sequence,
        expected_state=expected_state,
        phase_hint=phase_hint,
        timeout_seconds=timeout_seconds,
        context="strict replay waiting for fresh matching state",
        require_ready=require_ready,
        allow_state_probe=allow_state_probe,
        allow_wait_probe=allow_wait_probe,
    )
    return (actual if matched else None), actual


def _sync_known_pre_state(
    coordinator: Coordinator,
    state_tap: _StateTapFollower | None,
    *,
    step: StrictRecordedTraceStep,
    live_pre: dict[str, Any],
    timeout_seconds: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    if (
        str(step.phase or "").upper() == "NEOW"
        and _is_visible_single_neow_transition_choice(live_pre)
        and _visible_neow_choice_ui_family(step.strict_pre_state) == "neow_event"
    ):
        _queue_strict_gameplay_command(coordinator, "choose 0", live_pre_state=live_pre)
        _wait_for_expected_post_state(
            coordinator,
            state_tap,
            after_sequence=coordinator.raw_message_sequence,
            expected_state=live_pre,
            phase_hint=step.phase,
            timeout_seconds=timeout_seconds,
            context=f"strict replay syncing hidden Neow transition before step {step.step}",
            require_ready=False,
            allow_state_probe=False,
            allow_wait_probe=False,
        )
        matched_state, actual_state = _wait_for_fresh_matching_state(
            coordinator,
            state_tap,
            expected_state=step.strict_pre_state,
            phase_hint=step.phase,
            timeout_seconds=timeout_seconds,
            allow_state_probe=False,
            allow_wait_probe=False,
            require_ready=True,
        )
        return matched_state or actual_state, ["choose 0"]
    return live_pre, []


def _sync_paused_shop_purge_card_select(
    coordinator: Coordinator,
    state_tap: _StateTapFollower | None,
    *,
    manifest: dict[str, Any],
    current_visible: dict[str, Any],
    trace_pre: dict[str, Any],
    raw_state_log_path: Path,
    pause_id: str,
) -> tuple[bool, dict[str, Any] | None, list[str]]:
    if str(current_visible.get("phase") or "") != "CARD_SELECT":
        return False, current_visible, []
    if not bool((current_visible.get("screen_state") or {}).get("for_purge")):
        return False, current_visible, []
    if str(trace_pre.get("phase") or "") != "SHOP":
        return False, current_visible, []
    target_index = _manifest_last_shop_purge_target_index(manifest)
    if target_index is None:
        return False, current_visible, []
    choice_index = _card_select_choice_index_for_deck_target(current_visible, target_index)
    if choice_index is None:
        return False, current_visible, []
    command = f"choose {choice_index}"
    before_sequence = coordinator.raw_message_sequence
    _queue_strict_gameplay_command(coordinator, command, live_pre_state=current_visible)
    _append_strict_debug(
        raw_state_log_path,
        {
            "event": "paused_shop_purge_card_select_sync_dispatched",
            "pause_id": pause_id,
            "target_index": target_index,
            "choice_index": choice_index,
            "command": command,
            "live_state": current_visible,
        },
    )
    matched, _saw_fresh, actual = _wait_for_expected_post_state(
        coordinator,
        state_tap,
        after_sequence=before_sequence,
        expected_state=trace_pre,
        phase_hint=str(trace_pre.get("phase") or "SHOP"),
        timeout_seconds=DEFAULT_STRICT_REPLAY_ACTION_TIMEOUT_SECONDS,
        context="strict replay syncing paused shop purge card-select transition",
        require_ready=_expected_state_requires_ready(trace_pre),
        initial_state_probe_delay=STRICT_REPLAY_OBSERVE_WAIT_INTERVAL_SECONDS,
        initial_wait_probe_delay=STRICT_REPLAY_OBSERVE_WAIT_INTERVAL_SECONDS,
    )
    return bool(matched), actual, [command]


def _expected_shop_screen_requires_open(expected_state: dict[str, Any] | None) -> bool:
    if not isinstance(expected_state, dict) or str(expected_state.get("phase") or "") != "SHOP":
        return False
    for choice in _choice_items(expected_state):
        if str(choice.get("kind") or "") == "shop" and str(choice.get("item_kind") or "") in {
            "card",
            "relic",
            "potion",
            "purge",
        }:
            return True
    return False


def _raw_state_is_shop_room(raw_state: Any) -> bool:
    if not isinstance(raw_state, dict):
        return False
    screen_type = str(raw_state.get("screen_type") or raw_state.get("screen") or "").upper()
    if screen_type == "SHOP_ROOM":
        return True
    choice_list = [str(choice).lower() for choice in list(raw_state.get("choice_list") or [])]
    return str(raw_state.get("room_type") or "") == "ShopRoom" and "shop" in choice_list


def _expected_map_state(expected_state: dict[str, Any] | None) -> bool:
    if not isinstance(expected_state, dict):
        return False
    return str(expected_state.get("phase") or "") == "MAP" and str(expected_state.get("screen_type") or "") == "MAP"


def _raw_state_is_rest_room(raw_state: Any) -> bool:
    if not isinstance(raw_state, dict):
        return False
    return (
        str(raw_state.get("room_type") or "") == "RestRoom"
        and str(raw_state.get("screen_type") or raw_state.get("screen") or "").upper() == "REST"
    )


def _advance_visible_shop_room_transition(
    coordinator: Coordinator,
    state_tap: _StateTapFollower | None,
    *,
    expected_state: dict[str, Any],
    timeout_seconds: float,
    debug_log_path: Path,
    step: int,
) -> tuple[list[str], dict[str, Any] | None]:
    if not _expected_shop_screen_requires_open(expected_state):
        return [], None
    raw_state = getattr(coordinator, "last_raw_game_state", None)
    if not _raw_state_is_shop_room(raw_state):
        return [], None

    live_state = _observed_strict_state(coordinator)
    before_sequence = coordinator.raw_message_sequence
    _append_strict_debug(
        debug_log_path,
        {
            "event": "shop_room_transition_command",
            "step": step,
            "raw_message_sequence_before_action": before_sequence,
            "command": "choose shop",
            "live_state": live_state,
        },
    )
    _queue_strict_gameplay_command(
        coordinator,
        "choose shop",
        live_pre_state=live_state,
    )
    settle_seconds = max(
        STRICT_REPLAY_GAMEPLAY_COMMAND_SETTLE_SECONDS,
        STRICT_REPLAY_VISIBLE_SCREEN_ACTION_SETTLE_SECONDS,
    )
    time.sleep(settle_seconds)
    matched_state, _, live_post = _wait_for_expected_post_state(
        coordinator,
        state_tap,
        after_sequence=before_sequence,
        expected_state=expected_state,
        phase_hint="SHOP",
        timeout_seconds=timeout_seconds,
        context=f"strict replay opening shop room before step {step}",
        require_ready=True,
        initial_state_probe_delay=settle_seconds,
        initial_wait_probe_delay=settle_seconds * 2.0,
    )
    return ["choose shop"], live_post if matched_state else _observed_strict_state(coordinator)


def _classify_neow_choice_failure_kind(
    *,
    expected_state: dict[str, Any],
    actual_state: dict[str, Any] | None,
    saw_fresh_state: bool,
    mismatch_kind: str,
    missing_kind: str,
) -> str:
    expected_family = _visible_neow_choice_ui_family(expected_state)
    if expected_family is None:
        return mismatch_kind if saw_fresh_state else missing_kind
    actual_family = _visible_neow_choice_ui_family(actual_state)
    if actual_family != expected_family:
        return "visible_choice_ui_not_observed"
    return mismatch_kind


def _expected_state_requires_ready(expected_state: dict[str, Any] | None) -> bool:
    if not isinstance(expected_state, dict):
        return False
    if _is_visible_screen_phase(expected_state.get("phase")):
        return True
    if _visible_neow_choice_ui_family(expected_state) is not None:
        return True
    return _is_visible_single_neow_transition_choice(expected_state)


def _choice_pre_state_can_use_unready_card_reward(
    coordinator: Coordinator,
    expected_state: dict[str, Any] | None,
    strict_action: dict[str, Any] | None,
) -> bool:
    if not isinstance(expected_state, dict) or not isinstance(strict_action, dict):
        return False
    if str(expected_state.get("phase") or "") != "CARD_REWARD":
        return False
    if str(strict_action.get("kind") or "") != "choose_by_index":
        return False
    available_commands = set((coordinator.last_communication_state or {}).get("available_commands") or [])
    return "choose" in available_commands


def _decision_pre_state_requires_pipe_sync(state: dict[str, Any] | None) -> bool:
    if not isinstance(state, dict):
        return False
    if _visible_neow_choice_ui_family(state) is not None:
        return True
    return _is_visible_screen_phase(state.get("phase"))


def _current_pipe_ready_matches(
    coordinator: Coordinator,
    expected_state: dict[str, Any],
    *,
    phase_hint: str,
    require_ready: bool = True,
    allow_shop_inventory_mismatch: bool = False,
) -> bool:
    if getattr(coordinator, "last_communication_source", None) != "pipe":
        return False
    if require_ready and not coordinator.game_is_ready:
        return False
    return _strict_state_matches(
        expected_state,
        _live_strict_state(coordinator, phase_hint=phase_hint),
        allow_shop_inventory_mismatch=allow_shop_inventory_mismatch,
    )


def _is_neow_continue_trace_step(step: StrictRecordedTraceStep | None) -> bool:
    if step is None:
        return False
    action = dict(step.strict_action or {})
    if str(action.get("kind") or "") != "choose_by_index":
        return False
    try:
        if int(action.get("choice_index")) != 0:
            return False
    except (TypeError, ValueError):
        return False
    return _is_visible_single_neow_transition_choice(step.strict_pre_state)


def _can_collapse_card_reward_neow_continue(
    step: StrictRecordedTraceStep,
    next_step: StrictRecordedTraceStep | None,
) -> bool:
    if str(step.phase or "").upper() != "CARD_REWARD":
        return False
    if not _is_neow_continue_trace_step(next_step):
        return False
    if step.strict_post_state != next_step.strict_pre_state:
        return False
    next_post_phase = str((next_step.strict_post_state or {}).get("phase") or "").upper()
    next_post_screen = str((next_step.strict_post_state or {}).get("screen_type") or "").upper()
    return next_post_phase == "MAP" or next_post_screen == "MAP"


def _allows_shop_inventory_mismatch_after_action(step: StrictRecordedTraceStep | None) -> bool:
    if step is None or str(step.phase) != "SHOP":
        return False
    action = dict(step.strict_action or {})
    if str(action.get("kind") or "") not in {"choose_by_name", "choose_by_index"}:
        return False
    return str(action.get("item_kind") or "").lower() in {"card", "relic", "potion"}


def _allows_shop_inventory_mismatch_before_action(step: StrictRecordedTraceStep | None) -> bool:
    if step is None or str(step.phase) != "SHOP":
        return False
    action = dict(step.strict_action or {})
    return str(action.get("kind") or "") == "leave" or str(action.get("item_kind") or "").lower() == "leave"


def _stabilize_pre_state_after_bootstrap(
    coordinator: Coordinator,
    state_tap: _StateTapFollower | None,
    *,
    step: StrictRecordedTraceStep,
    timeout_seconds: float,
) -> tuple[bool, bool, dict[str, Any] | None]:
    _drain_pending_updates(
        coordinator,
        state_tap,
        settle_seconds=STRICT_REPLAY_POST_MATCH_SETTLE_SECONDS,
    )
    return _wait_for_expected_post_state(
        coordinator,
        state_tap,
        after_sequence=coordinator.raw_message_sequence,
        expected_state=step.strict_pre_state,
        phase_hint=step.phase,
        timeout_seconds=timeout_seconds,
        context=f"strict replay stabilizing pre-state after bootstrap before step {step.step}",
        require_ready=True,
        allow_state_probe=True,
        allow_wait_probe=False,
    )


def _wait_for_ready_state(
    coordinator: Coordinator,
    state_tap: _StateTapFollower | None,
    *,
    timeout_seconds: float,
    context: str,
) -> bool:
    deadline = time.time() + timeout_seconds
    last_observed_sequence = coordinator.raw_message_sequence
    while time.time() < deadline:
        before_sequence = coordinator.raw_message_sequence
        for tap_event in _iter_state_tap_updates(coordinator, state_tap):
            last_observed_sequence = tap_event["sequence"]
            communication_state = dict(tap_event.get("communication_state") or {})
            if communication_state.get("error") is not None:
                return False
            if communication_state.get("ready_for_command"):
                return True
        if coordinator.raw_message_sequence > before_sequence and coordinator.last_error is not None:
            return False
        if coordinator.receive_game_state_update(block=False, perform_callbacks=False):
            if coordinator.last_error is not None:
                return False
            if coordinator.raw_message_sequence > last_observed_sequence:
                last_observed_sequence = coordinator.raw_message_sequence
            if coordinator.game_is_ready:
                return True
        time.sleep(0.05)
    return False


def _wait_for_fresh_state(
    coordinator: Coordinator,
    state_tap: _StateTapFollower | None,
    *,
    after_sequence: int,
    timeout_seconds: float,
    context: str,
    allow_wait_probe: bool = True,
    allow_state_probe: bool = True,
    require_ready: bool = True,
    initial_state_probe_delay: float = 0.0,
    initial_wait_probe_delay: float | None = None,
) -> bool:
    deadline = time.time() + timeout_seconds
    next_state_probe = time.time() + max(0.0, initial_state_probe_delay)
    wait_probe_delay = (
        STRICT_REPLAY_OBSERVE_WAIT_INTERVAL_SECONDS * 2.0
        if initial_wait_probe_delay is None
        else max(0.0, initial_wait_probe_delay)
    )
    next_wait_probe = time.time() + wait_probe_delay
    last_observed_sequence = after_sequence
    while time.time() < deadline:
        before_sequence = coordinator.raw_message_sequence
        for tap_event in _iter_state_tap_updates(coordinator, state_tap):
            last_observed_sequence = tap_event["sequence"]
            communication_state = dict(tap_event.get("communication_state") or {})
            if communication_state.get("error") is not None:
                return False
            if (
                tap_event["sequence"] > after_sequence
                and communication_state.get("game_state") is not None
                and (communication_state.get("ready_for_command") or not require_ready)
            ):
                return True
        if coordinator.raw_message_sequence > before_sequence and coordinator.last_error is not None:
            return False
        if coordinator.receive_game_state_update(block=False, perform_callbacks=False):
            if coordinator.last_error is not None:
                return False
            if (
                coordinator.raw_message_sequence > after_sequence
                and coordinator.last_raw_game_state is not None
                and (coordinator.game_is_ready or not require_ready)
            ):
                return True
            if coordinator.raw_message_sequence > last_observed_sequence:
                last_observed_sequence = coordinator.raw_message_sequence
        now = time.time()
        available_commands = set((coordinator.last_communication_state or {}).get("available_commands") or [])
        if allow_state_probe and "state" in available_commands and now >= next_state_probe:
            _send_strict_command(coordinator, "state")
            next_state_probe = now + STRICT_REPLAY_OBSERVE_WAIT_INTERVAL_SECONDS
        if allow_wait_probe and coordinator.in_game and "wait" in available_commands and now >= next_wait_probe:
            _send_strict_command(coordinator, "wait 1")
            next_wait_probe = now + STRICT_REPLAY_OBSERVE_WAIT_INTERVAL_SECONDS
        time.sleep(0.05)
    return False


def _wait_for_expected_post_state(
    coordinator: Coordinator,
    state_tap: _StateTapFollower | None,
    *,
    after_sequence: int,
    expected_state: dict[str, Any],
    phase_hint: str,
    timeout_seconds: float,
    context: str,
    require_ready: bool = False,
    initial_state_probe_delay: float = 0.0,
    initial_wait_probe_delay: float | None = None,
    allow_state_probe: bool = True,
    allow_wait_probe: bool = True,
    confirm_card_select_transition: bool = False,
    card_select_transition_choice_index: int | None = None,
    transition_commands: list[str] | None = None,
    debug_log_path: Path | None = None,
    step: int | None = None,
    allow_shop_inventory_mismatch: bool = False,
) -> tuple[bool, bool, dict[str, Any] | None]:
    deadline = time.time() + timeout_seconds
    next_state_probe = time.time() + max(0.0, initial_state_probe_delay)
    wait_probe_delay = (
        STRICT_REPLAY_OBSERVE_WAIT_INTERVAL_SECONDS * 2.0
        if initial_wait_probe_delay is None
        else max(0.0, initial_wait_probe_delay)
    )
    next_wait_probe = time.time() + wait_probe_delay
    saw_fresh_state = False
    last_live_state: dict[str, Any] | None = None
    last_observed_sequence = after_sequence
    card_select_confirm_sent = False
    card_select_choice_sent = False
    shop_room_open_sent = False
    shop_room_proceed_sent = False
    rest_room_proceed_sent = False

    def maybe_send_card_select_choice(
        live_state: dict[str, Any],
        *,
        ready_for_command: bool,
        sequence: int,
    ) -> bool:
        nonlocal after_sequence, last_observed_sequence, card_select_choice_sent
        if card_select_choice_sent or card_select_transition_choice_index is None or not ready_for_command:
            return False
        if str(live_state.get("phase") or "") != "CARD_SELECT":
            return False
        screen_state = dict(live_state.get("screen_state") or {})
        if not bool(screen_state.get("for_purge")):
            return False
        choice_index = int(card_select_transition_choice_index)
        choices = list(screen_state.get("choices") or [])
        if choices and not any(int(choice.get("choice_index", -1)) == choice_index for choice in choices):
            mapped_choice_index = _card_select_choice_index_for_deck_target(live_state, choice_index)
            if mapped_choice_index is None:
                return False
            choice_index = mapped_choice_index
        if choices and not any(int(choice.get("choice_index", -1)) == choice_index for choice in choices):
            return False
        command = f"choose {choice_index}"
        _queue_strict_gameplay_command(
            coordinator,
            command,
            live_pre_state=live_state,
        )
        card_select_choice_sent = True
        if transition_commands is not None:
            transition_commands.append(command)
        if debug_log_path is not None:
            _append_strict_debug(
                debug_log_path,
                {
                    "event": "card_select_choice_dispatched",
                    "step": step,
                    "raw_message_sequence_before_action": sequence,
                    "command": command,
                    "live_state": live_state,
                },
            )
        after_sequence = max(after_sequence, int(sequence))
        last_observed_sequence = max(last_observed_sequence, after_sequence)
        return True

    def maybe_send_card_select_confirm(
        live_state: dict[str, Any],
        *,
        ready_for_command: bool,
        sequence: int,
    ) -> bool:
        nonlocal after_sequence, last_observed_sequence, card_select_confirm_sent
        if card_select_confirm_sent or not confirm_card_select_transition or not ready_for_command:
            return False
        if str(live_state.get("phase") or "") != "CARD_SELECT":
            return False
        if not bool((live_state.get("screen_state") or {}).get("confirm_up")):
            return False
        _queue_strict_gameplay_command(
            coordinator,
            "confirm",
            live_pre_state=live_state,
        )
        card_select_confirm_sent = True
        if transition_commands is not None:
            transition_commands.append("confirm")
        if debug_log_path is not None:
            _append_strict_debug(
                debug_log_path,
                {
                    "event": "card_select_confirm_dispatched",
                    "step": step,
                    "raw_message_sequence_before_action": sequence,
                    "command": "confirm",
                    "live_state": live_state,
                },
            )
        after_sequence = max(after_sequence, int(sequence))
        last_observed_sequence = max(last_observed_sequence, after_sequence)
        return True

    def maybe_open_shop_room(
        raw_state: Any,
        live_state: dict[str, Any],
        *,
        ready_for_command: bool,
        sequence: int,
    ) -> bool:
        nonlocal after_sequence, last_observed_sequence, shop_room_open_sent
        if shop_room_open_sent or not ready_for_command:
            return False
        if not _expected_shop_screen_requires_open(expected_state):
            return False
        if not _raw_state_is_shop_room(raw_state):
            return False
        _queue_strict_gameplay_command(
            coordinator,
            "choose shop",
            live_pre_state=live_state,
        )
        shop_room_open_sent = True
        if transition_commands is not None:
            transition_commands.append("choose shop")
        if debug_log_path is not None:
            _append_strict_debug(
                debug_log_path,
                {
                    "event": "shop_room_transition_dispatched",
                    "step": step,
                    "raw_message_sequence_before_action": sequence,
                    "command": "choose shop",
                    "live_state": live_state,
                },
            )
        after_sequence = max(after_sequence, int(sequence))
        last_observed_sequence = max(last_observed_sequence, after_sequence)
        return True

    def maybe_proceed_from_shop_room(
        raw_state: Any,
        live_state: dict[str, Any],
        *,
        ready_for_command: bool,
        sequence: int,
        available_commands: list[str] | tuple[str, ...] | set[str] | None,
    ) -> bool:
        nonlocal after_sequence, last_observed_sequence, shop_room_proceed_sent
        if shop_room_proceed_sent or not ready_for_command:
            return False
        if not _expected_map_state(expected_state):
            return False
        if not _raw_state_is_shop_room(raw_state):
            return False
        if "proceed" not in set(available_commands or []):
            return False
        _queue_strict_gameplay_command(
            coordinator,
            "proceed",
            live_pre_state=live_state,
        )
        shop_room_proceed_sent = True
        if transition_commands is not None:
            transition_commands.append("proceed")
        if debug_log_path is not None:
            _append_strict_debug(
                debug_log_path,
                {
                    "event": "shop_room_proceed_dispatched",
                    "step": step,
                    "raw_message_sequence_before_action": sequence,
                    "command": "proceed",
                    "live_state": live_state,
                },
            )
        after_sequence = max(after_sequence, int(sequence))
        last_observed_sequence = max(last_observed_sequence, after_sequence)
        return True

    def maybe_proceed_from_rest_room(
        raw_state: Any,
        live_state: dict[str, Any],
        *,
        ready_for_command: bool,
        sequence: int,
        available_commands: list[str] | tuple[str, ...] | set[str] | None,
    ) -> bool:
        nonlocal after_sequence, last_observed_sequence, rest_room_proceed_sent
        if rest_room_proceed_sent or not ready_for_command:
            return False
        if not _expected_map_state(expected_state):
            return False
        if not _raw_state_is_rest_room(raw_state):
            return False
        if "proceed" not in set(available_commands or []):
            return False
        _queue_strict_gameplay_command(
            coordinator,
            "proceed",
            live_pre_state=live_state,
        )
        rest_room_proceed_sent = True
        if transition_commands is not None:
            transition_commands.append("proceed")
        if debug_log_path is not None:
            _append_strict_debug(
                debug_log_path,
                {
                    "event": "rest_room_proceed_dispatched",
                    "step": step,
                    "raw_message_sequence_before_action": sequence,
                    "command": "proceed",
                    "live_state": live_state,
                },
            )
        after_sequence = max(after_sequence, int(sequence))
        last_observed_sequence = max(last_observed_sequence, after_sequence)
        return True

    while time.time() < deadline:
        before_sequence = coordinator.raw_message_sequence
        for tap_event in _iter_state_tap_updates(coordinator, state_tap):
            communication_state = dict(tap_event.get("communication_state") or {})
            raw_game_state = communication_state.get("game_state")
            if raw_game_state is None:
                continue
            saw_fresh_state = True
            last_observed_sequence = tap_event["sequence"]
            last_live_state = normalize_verbose_state_for_strict(raw_game_state)
            live_compare_state = normalize_verbose_state_for_strict(
                raw_game_state,
                phase_hint=phase_hint,
            )
            if _strict_state_matches(
                expected_state,
                live_compare_state,
                allow_shop_inventory_mismatch=allow_shop_inventory_mismatch,
            ) and (
                communication_state.get("ready_for_command") or not require_ready
            ):
                _drain_pending_updates(
                    coordinator,
                    state_tap,
                    settle_seconds=_post_match_settle_seconds_for_phase(phase_hint),
                )
                if (
                    getattr(coordinator, "last_communication_source", None) == "pipe"
                    and _strict_state_matches(
                        expected_state,
                        _live_strict_state(coordinator, phase_hint=phase_hint),
                        allow_shop_inventory_mismatch=allow_shop_inventory_mismatch,
                    )
                    and (coordinator.game_is_ready or not require_ready)
                ):
                    return True, True, _live_strict_state(coordinator)
                _restore_matched_state_tap_frame(coordinator, tap_event)
                return True, True, last_live_state
            if maybe_send_card_select_choice(
                last_live_state,
                ready_for_command=bool(communication_state.get("ready_for_command")),
                sequence=int(tap_event["sequence"]),
            ):
                continue
            if maybe_send_card_select_confirm(
                last_live_state,
                ready_for_command=bool(communication_state.get("ready_for_command")),
                sequence=int(tap_event["sequence"]),
            ):
                continue
            if maybe_open_shop_room(
                raw_game_state,
                live_compare_state,
                ready_for_command=bool(communication_state.get("ready_for_command")),
                sequence=int(tap_event["sequence"]),
            ):
                continue
            if maybe_proceed_from_shop_room(
                raw_game_state,
                live_compare_state,
                ready_for_command=bool(communication_state.get("ready_for_command")),
                sequence=int(tap_event["sequence"]),
                available_commands=communication_state.get("available_commands"),
            ):
                continue
            if maybe_proceed_from_rest_room(
                raw_game_state,
                live_compare_state,
                ready_for_command=bool(communication_state.get("ready_for_command")),
                sequence=int(tap_event["sequence"]),
                available_commands=communication_state.get("available_commands"),
            ):
                continue
        if coordinator.receive_game_state_update(block=False, perform_callbacks=False):
            if coordinator.last_error is not None:
                return False, saw_fresh_state, last_live_state
            if (
                coordinator.raw_message_sequence > after_sequence
                and coordinator.last_raw_game_state is not None
            ):
                saw_fresh_state = True
                if coordinator.raw_message_sequence > last_observed_sequence:
                    last_observed_sequence = coordinator.raw_message_sequence
                last_live_state = _live_strict_state(coordinator)
                live_compare_state = _live_strict_state(coordinator, phase_hint=phase_hint)
                if _strict_state_matches(
                    expected_state,
                    live_compare_state,
                    allow_shop_inventory_mismatch=allow_shop_inventory_mismatch,
                ) and (coordinator.game_is_ready or not require_ready):
                    return True, True, last_live_state
                if maybe_send_card_select_choice(
                    last_live_state,
                    ready_for_command=bool(coordinator.game_is_ready),
                    sequence=int(coordinator.raw_message_sequence),
                ):
                    continue
                if maybe_send_card_select_confirm(
                    last_live_state,
                    ready_for_command=bool(coordinator.game_is_ready),
                    sequence=int(coordinator.raw_message_sequence),
                ):
                    continue
                if maybe_proceed_from_shop_room(
                    getattr(coordinator, "last_raw_game_state", None),
                    live_compare_state,
                    ready_for_command=bool(coordinator.game_is_ready),
                    sequence=int(coordinator.raw_message_sequence),
                    available_commands=(coordinator.last_communication_state or {}).get("available_commands"),
                ):
                    continue
                if maybe_proceed_from_rest_room(
                    getattr(coordinator, "last_raw_game_state", None),
                    live_compare_state,
                    ready_for_command=bool(coordinator.game_is_ready),
                    sequence=int(coordinator.raw_message_sequence),
                    available_commands=(coordinator.last_communication_state or {}).get("available_commands"),
                ):
                    continue
                if maybe_open_shop_room(
                    getattr(coordinator, "last_raw_game_state", None),
                    live_compare_state,
                    ready_for_command=bool(coordinator.game_is_ready),
                    sequence=int(coordinator.raw_message_sequence),
                ):
                    continue
        now = time.time()
        available_commands = set((coordinator.last_communication_state or {}).get("available_commands") or [])
        if allow_state_probe and "state" in available_commands and now >= next_state_probe:
            _send_strict_command(coordinator, "state")
            next_state_probe = now + STRICT_REPLAY_OBSERVE_WAIT_INTERVAL_SECONDS
        if allow_wait_probe and coordinator.in_game and "wait" in available_commands and now >= next_wait_probe:
            _send_strict_command(coordinator, "wait 1")
            next_wait_probe = now + STRICT_REPLAY_OBSERVE_WAIT_INTERVAL_SECONDS
        time.sleep(0.05)
    return False, saw_fresh_state, last_live_state


def _wait_for_expected_pipe_state(
    coordinator: Coordinator,
    *,
    expected_state: dict[str, Any],
    phase_hint: str,
    timeout_seconds: float,
    require_ready: bool = True,
    allow_shop_inventory_mismatch: bool = False,
) -> tuple[bool, bool, dict[str, Any] | None]:
    deadline = time.time() + timeout_seconds
    saw_fresh_state = False
    last_live_state: dict[str, Any] | None = None
    while time.time() < deadline:
        if coordinator.receive_game_state_update(block=False, perform_callbacks=False):
            if coordinator.last_error is not None:
                return False, saw_fresh_state, last_live_state
            if coordinator.last_raw_game_state is not None:
                saw_fresh_state = True
                last_live_state = _live_strict_state(coordinator)
                live_compare_state = _live_strict_state(coordinator, phase_hint=phase_hint)
                if _strict_state_matches(
                    expected_state,
                    live_compare_state,
                    allow_shop_inventory_mismatch=allow_shop_inventory_mismatch,
                ) and (
                    coordinator.game_is_ready or not require_ready
                ):
                    return True, True, last_live_state
        time.sleep(0.05)
    return False, saw_fresh_state, last_live_state


def _start_recorded_game(
    coordinator: Coordinator,
    state_tap: _StateTapFollower | None,
    *,
    player_class: PlayerClass,
    ascension_level: int,
    seed: str | None,
) -> None:
    ready_timeout = _env_float(
        "SPIRECOMM_STRICT_REPLAY_READY_TIMEOUT_SECONDS",
        DEFAULT_STRICT_REPLAY_READY_TIMEOUT_SECONDS,
    )
    if os.environ.get("SPIRECOMM_BOOTSTRAP_READY_SENT") != "1":
        coordinator.signal_ready()
    if not _wait_for_ready_state(
        coordinator,
        state_tap,
        timeout_seconds=ready_timeout,
        context="strict replay initial ready",
    ):
        raise RuntimeError("strict replay did not receive an initial command-ready state")
    if coordinator.in_game:
        return
    start_command = f"start {player_class.name} {ascension_level}"
    if seed is not None:
        start_command += f" {seed}"
    _send_strict_command(coordinator, start_command)
    if not _wait_for_fresh_state(
        coordinator,
        state_tap,
        after_sequence=0,
        timeout_seconds=ready_timeout,
        context="strict replay start game",
        allow_wait_probe=False,
        allow_state_probe=state_tap is None,
        require_ready=False,
    ):
        raise RuntimeError("strict replay did not receive a fresh in-game state after start")


def _execute_strict_action(
    coordinator: Coordinator,
    *,
    live_pre_state: dict[str, Any],
    strict_action: dict[str, Any],
) -> tuple[str, list[str]]:
    kind = str(strict_action.get("kind") or "")
    commands: list[str] = []
    if kind == "play_card":
        hand_index = int(strict_action.get("hand_index", -1))
        target_index = strict_action.get("target_index")
        if hand_index < 0:
            raise ValueError("strict trace is missing a valid hand_index for play_card")
        command = f"play {hand_index + 1}"
        if target_index is not None:
            command += f" {int(target_index)}"
        commands.append(command)
    elif kind == "use_potion":
        potion_index = strict_action.get("potion_index")
        if potion_index is None:
            raise ValueError("strict trace is missing potion_index for use_potion")
        command = f"potion use {int(potion_index)}"
        target_index = strict_action.get("target_index")
        if target_index is not None:
            command += f" {int(target_index)}"
        commands.append(command)
    elif kind == "discard_potion":
        potion_index = strict_action.get("potion_index")
        if potion_index is None:
            raise ValueError("strict trace is missing potion_index for discard_potion")
        commands.append(f"potion discard {int(potion_index)}")
    elif kind == "end_turn":
        commands.append("end")
    elif kind == "choose_by_index":
        choice_index = strict_action.get("choice_index")
        if choice_index is None:
            raise ValueError("strict trace is missing choice_index for choose_by_index")
        phase = str(live_pre_state.get("phase") or "")
        screen_state = dict(live_pre_state.get("screen_state") or {})
        if (
            phase == "TREASURE"
            and str(screen_state.get("chest_type") or "") == "BossChest"
            and bool(screen_state.get("chest_open"))
        ):
            commands.append("proceed")
        else:
            commands.append(f"choose {int(choice_index)}")
        if phase == "CARD_SELECT" and screen_state.get("confirm_up"):
            commands.append("confirm")
    elif kind == "choose_by_name":
        name = strict_action.get("name")
        if not name:
            raise ValueError("strict trace is missing a name for choose_by_name")
        if (
            str(live_pre_state.get("phase") or "") == "SHOP"
            and str(strict_action.get("item_kind") or "").lower() == "leave"
        ):
            commands.append("leave")
        else:
            commands.append(f"choose {name}")
    elif kind == "skip":
        commands.append("skip")
    elif kind == "proceed":
        commands.append("proceed")
    elif kind == "confirm":
        commands.append("confirm")
    elif kind == "leave":
        commands.append("leave")
    else:
        raise ValueError(f"unsupported strict action kind: {kind!r}")

    for command in commands:
        _queue_strict_gameplay_command(
            coordinator,
            command,
            live_pre_state=live_pre_state,
        )
    return kind, commands


def _strict_failure_report(
    *,
    trace: StrictRecordedRunTrace | None,
    trace_path: Path,
    player_class: PlayerClass | None,
    total_steps: int,
    steps_replayed: int,
    first_failure_step: int | None,
    first_failure_phase: str | None,
    failure_kind: str,
    trace_action: dict[str, Any] | None,
    expected_visible: dict[str, Any] | None,
    actual_visible: dict[str, Any] | None,
    raw_state_log_path: Path,
    runner_error: str | None = None,
) -> dict[str, Any]:
    command_journal_path = _strict_command_journal_path()
    command_queue_dir = _strict_command_queue_dir()
    return {
        "trace_path": str(trace_path),
        "trace_schema": trace.trace_schema if trace is not None else None,
        "replay_mode": "strict",
        "seed_long": trace.seed_long if trace is not None else None,
        "seed_str": trace.seed_str if trace is not None else None,
        "ascension": trace.ascension if trace is not None else None,
        "character": player_class.name if player_class is not None else None,
        "steps_total": total_steps,
        "steps_replayed": steps_replayed,
        "success": False,
        "first_failure_step": first_failure_step,
        "first_failure_phase": first_failure_phase,
        "failure_kind": failure_kind,
        "trace_action": trace_action,
        "expected_visible": expected_visible,
        "actual_visible": actual_visible,
        "raw_state_log_path": str(raw_state_log_path),
        "command_journal_path": str(command_journal_path) if command_journal_path is not None else None,
        "command_queue_dir": str(command_queue_dir) if command_queue_dir is not None else None,
        "runner_error": runner_error,
        "results": [],
        "final_state": None,
    }


_PAUSABLE_STRICT_FAILURE_KINDS = {
    "pre_state_mismatch",
    "post_state_mismatch",
    "missing_fresh_live_state",
    "command_channel_state_not_observed",
    "visible_choice_ui_not_observed",
}


def _is_pausable_strict_failure(failure_report: dict[str, Any]) -> bool:
    return str(failure_report.get("failure_kind") or "") in _PAUSABLE_STRICT_FAILURE_KINDS


def _pause_control_paths(
    *,
    raw_log_target: Path,
    pause_manifest_path: str | Path | None,
    resume_request_path: str | Path | None,
    resume_result_path: str | Path | None,
    pause_report_path: str | Path | None,
) -> tuple[Path, Path, Path, Path]:
    report_path = Path(
        pause_report_path
        or os.environ.get("SPIRECOMM_REPLAY_REPORT")
        or raw_log_target.with_suffix(".paused_report.json")
    )
    manifest_path = Path(
        pause_manifest_path
        or os.environ.get(STRICT_REPLAY_PAUSE_MANIFEST_ENV)
        or report_path.with_suffix(".pause.json")
    )
    request_path = Path(
        resume_request_path
        or os.environ.get(STRICT_REPLAY_RESUME_REQUEST_ENV)
        or report_path.with_suffix(".resume.json")
    )
    result_path = Path(
        resume_result_path
        or os.environ.get(STRICT_REPLAY_RESUME_RESULT_ENV)
        or report_path.with_suffix(".resume_result.json")
    )
    return manifest_path, request_path, result_path, report_path


def _build_pause_manifest(
    *,
    pause_id: str,
    trace: StrictRecordedRunTrace,
    trace_path: Path,
    player_class: PlayerClass,
    failure_report: dict[str, Any],
    next_step_index: int,
    action_prefix: list[dict[str, Any]],
    raw_state_log_path: Path,
    pause_manifest_path: Path,
    resume_request_path: Path,
    resume_result_path: Path,
) -> dict[str, Any]:
    next_trace_step = None
    if 0 <= next_step_index < len(trace.steps):
        next_trace_step = trace.steps[next_step_index].step
    return {
        "pause_id": pause_id,
        "paused_at": time.time(),
        "pid": os.getpid(),
        "seed_long": trace.seed_long,
        "seed_str": trace.seed_str,
        "ascension": trace.ascension,
        "character": player_class.name,
        "trace_path": str(trace_path),
        "trace_schema": trace.trace_schema,
        "trace_policy": trace.trace_policy,
        "model_required": trace.model_required,
        "failure_kind": failure_report.get("failure_kind"),
        "failure_step": failure_report.get("first_failure_step"),
        "failure_phase": failure_report.get("first_failure_phase"),
        "expected_visible": copy.deepcopy(failure_report.get("expected_visible")),
        "actual_visible": copy.deepcopy(failure_report.get("actual_visible")),
        "next_step_to_send": int(next_step_index),
        "next_step_index": int(next_step_index),
        "next_trace_step": next_trace_step,
        "action_prefix": copy.deepcopy(action_prefix),
        "raw_state_log_path": str(raw_state_log_path),
        "command_journal_path": failure_report.get("command_journal_path"),
        "command_queue_dir": failure_report.get("command_queue_dir"),
        "pause_manifest_path": str(pause_manifest_path),
        "resume_request_path": str(resume_request_path),
        "resume_result_path": str(resume_result_path),
    }


def _paused_failure_report(
    failure_report: dict[str, Any],
    *,
    pause_id: str,
    pause_manifest_path: Path,
    resume_request_path: Path,
    resume_result_path: Path,
    next_step_index: int,
    action_prefix: list[dict[str, Any]],
) -> dict[str, Any]:
    paused_report = dict(failure_report)
    paused_report.update(
        {
            "paused": True,
            "pause_id": pause_id,
            "pause_manifest_path": str(pause_manifest_path),
            "resume_request_path": str(resume_request_path),
            "resume_result_path": str(resume_result_path),
            "next_step_to_send": int(next_step_index),
            "action_prefix_length": len(action_prefix),
        }
    )
    return paused_report


def _pause_until_resume_or_abort(
    *,
    coordinator: Coordinator,
    state_tap: _StateTapFollower | None,
    trace: StrictRecordedRunTrace,
    trace_path: Path,
    player_class: PlayerClass,
    total_steps: int,
    failure_report: dict[str, Any],
    next_step_index: int,
    action_prefix: list[dict[str, Any]],
    raw_state_log_path: Path,
    pause_manifest_path: Path,
    resume_request_path: Path,
    resume_result_path: Path,
    pause_report_path: Path,
    pause_wait_timeout_seconds: float | None,
) -> _PauseResumeOutcome:
    del total_steps
    pause_id = f"{int(time.time() * 1000)}-{os.getpid()}-{failure_report.get('first_failure_step')}"
    manifest = _build_pause_manifest(
        pause_id=pause_id,
        trace=trace,
        trace_path=trace_path,
        player_class=player_class,
        failure_report=failure_report,
        next_step_index=next_step_index,
        action_prefix=action_prefix,
        raw_state_log_path=raw_state_log_path,
        pause_manifest_path=pause_manifest_path,
        resume_request_path=resume_request_path,
        resume_result_path=resume_result_path,
    )
    paused_report = _paused_failure_report(
        failure_report,
        pause_id=pause_id,
        pause_manifest_path=pause_manifest_path,
        resume_request_path=resume_request_path,
        resume_result_path=resume_result_path,
        next_step_index=next_step_index,
        action_prefix=action_prefix,
    )
    if resume_result_path.exists():
        resume_result_path.unlink()
    if resume_request_path.exists():
        resume_request_path.unlink()
    _write_json_atomic(pause_manifest_path, manifest)
    _write_json_atomic(pause_report_path, paused_report)
    _append_strict_debug(
        raw_state_log_path,
        {
            "event": "strict_replay_paused",
            "pause_id": pause_id,
            "failure_step": failure_report.get("first_failure_step"),
            "failure_kind": failure_report.get("failure_kind"),
            "next_step_to_send": next_step_index,
            "pause_manifest_path": str(pause_manifest_path),
            "resume_request_path": str(resume_request_path),
        },
    )

    deadline = None if pause_wait_timeout_seconds is None else time.time() + pause_wait_timeout_seconds
    while True:
        _iter_state_tap_updates(coordinator, state_tap)
        coordinator.receive_game_state_update(block=False, perform_callbacks=False)

        if resume_request_path.exists():
            try:
                request = json.loads(resume_request_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                try:
                    resume_request_path.unlink()
                except FileNotFoundError:
                    pass
                _write_json_atomic(
                    resume_result_path,
                    {
                        "accepted": False,
                        "pause_id": pause_id,
                        "reason": "invalid_resume_request_json",
                        "timestamp": time.time(),
                    },
                )
                time.sleep(STRICT_REPLAY_PAUSE_POLL_INTERVAL_SECONDS)
                continue
            try:
                resume_request_path.unlink()
            except FileNotFoundError:
                pass

            if str(request.get("pause_id") or "") != pause_id:
                _write_json_atomic(
                    resume_result_path,
                    {
                        "accepted": False,
                        "pause_id": pause_id,
                        "request_pause_id": request.get("pause_id"),
                        "reason": "pause_id_mismatch",
                        "timestamp": time.time(),
                    },
                )
                continue

            command = str(request.get("command") or "").lower()
            if command == "abort":
                aborted = dict(failure_report)
                aborted.update(
                    {
                        "paused": False,
                        "pause_id": pause_id,
                        "failure_kind": "replay_pause_aborted",
                        "runner_error": request.get("reason") or "paused strict replay aborted",
                    }
                )
                _write_json_atomic(
                    resume_result_path,
                    {
                        "accepted": True,
                        "pause_id": pause_id,
                        "command": "abort",
                        "timestamp": time.time(),
                    },
                )
                return _PauseResumeOutcome(report=aborted)

            if command != "resume":
                _write_json_atomic(
                    resume_result_path,
                    {
                        "accepted": False,
                        "pause_id": pause_id,
                        "reason": "unsupported_resume_command",
                        "command": request.get("command"),
                        "timestamp": time.time(),
                    },
                )
                continue

            try:
                requested_next_index = int(request.get("next_step_to_send", next_step_index))
            except (TypeError, ValueError):
                _write_json_atomic(
                    resume_result_path,
                    {
                        "accepted": False,
                        "pause_id": pause_id,
                        "reason": "invalid_request_next_step_to_send",
                        "request_next_step_to_send": request.get("next_step_to_send"),
                        "timestamp": time.time(),
                    },
                )
                continue
            if requested_next_index != next_step_index:
                _write_json_atomic(
                    resume_result_path,
                    {
                        "accepted": False,
                        "pause_id": pause_id,
                        "reason": "next_step_to_send_mismatch",
                        "expected_next_step_to_send": next_step_index,
                        "request_next_step_to_send": requested_next_index,
                        "timestamp": time.time(),
                    },
                )
                continue

            requested_trace_path = Path(str(request.get("trace_path") or ""))
            try:
                resume_trace = load_strict_recorded_trace(requested_trace_path)
            except Exception as exc:
                _write_json_atomic(
                    resume_result_path,
                    {
                        "accepted": False,
                        "pause_id": pause_id,
                        "reason": "resume_trace_load_failed",
                        "error": repr(exc),
                        "trace_path": str(requested_trace_path),
                        "timestamp": time.time(),
                    },
                )
                continue

            validation = validate_strict_pause_resume_trace(manifest, resume_trace)
            if validation.get("ok"):
                if next_step_index == len(resume_trace.steps):
                    trace_pre = resume_trace.steps[-1].strict_post_state if resume_trace.steps else {}
                else:
                    trace_pre = resume_trace.steps[next_step_index].strict_pre_state
                current_visible = _live_strict_state(
                    coordinator,
                    phase_hint=str(trace_pre.get("phase") or ""),
                )
                if current_visible != trace_pre:
                    synced, synced_visible, sync_commands = _sync_paused_shop_purge_card_select(
                        coordinator,
                        state_tap,
                        manifest=manifest,
                        current_visible=current_visible,
                        trace_pre=trace_pre,
                        raw_state_log_path=raw_state_log_path,
                        pause_id=pause_id,
                    )
                    if synced:
                        validation = dict(validation)
                        validation["paused_transition_sync_commands"] = sync_commands
                    else:
                        validation = {
                            "ok": False,
                            "reason": "current_live_state_mismatch",
                            "current_visible": synced_visible or current_visible,
                            "trace_pre_state": trace_pre,
                        }

            if not validation.get("ok"):
                _write_json_atomic(
                    resume_result_path,
                    {
                        "accepted": False,
                        "pause_id": pause_id,
                        "trace_path": str(requested_trace_path),
                        "validation": validation,
                        "timestamp": time.time(),
                    },
                )
                continue

            _write_json_atomic(
                resume_result_path,
                {
                    "accepted": True,
                    "pause_id": pause_id,
                    "command": "resume",
                    "trace_path": str(requested_trace_path),
                    "next_step_to_send": next_step_index,
                    "validation": validation,
                    "timestamp": time.time(),
                },
            )
            _append_strict_debug(
                raw_state_log_path,
                {
                    "event": "strict_replay_resumed",
                    "pause_id": pause_id,
                    "trace_path": str(requested_trace_path),
                    "next_step_to_send": next_step_index,
                },
            )
            return _PauseResumeOutcome(trace=resume_trace, next_step_index=next_step_index)

        if deadline is not None and time.time() >= deadline:
            timed_out = dict(failure_report)
            timed_out.update(
                {
                    "paused": False,
                    "pause_id": pause_id,
                    "failure_kind": "replay_pause_timeout",
                    "runner_error": f"paused strict replay exceeded {pause_wait_timeout_seconds:.3f}s without resume",
                }
            )
            _write_json_atomic(
                resume_result_path,
                {
                    "accepted": False,
                    "pause_id": pause_id,
                    "reason": "pause_wait_timeout",
                    "timestamp": time.time(),
                },
            )
            return _PauseResumeOutcome(report=timed_out)

        time.sleep(STRICT_REPLAY_PAUSE_POLL_INTERVAL_SECONDS)


def replay_recorded_run_strict(
    *,
    trace_path: str | Path,
    character: str | None = None,
    max_steps: int | None = None,
    raw_state_log_path: str | Path | None = None,
    pause_on_divergence: bool | None = None,
    pause_manifest_path: str | Path | None = None,
    resume_request_path: str | Path | None = None,
    resume_result_path: str | Path | None = None,
    pause_report_path: str | Path | None = None,
    pause_wait_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    trace_path = Path(trace_path)
    raw_log_target = Path(raw_state_log_path or os.environ.get("SPIRECOMM_RAW_STATE_LOG") or trace_path.with_suffix(".strict_raw_state.jsonl"))
    if raw_log_target.exists():
        raw_log_target.unlink()
    debug_log_target = _strict_debug_log_path(raw_log_target)
    if debug_log_target.exists():
        debug_log_target.unlink()
    command_journal_path = _strict_command_journal_path()
    if command_journal_path is not None and command_journal_path.exists():
        command_journal_path.unlink()
    command_queue_dir = _strict_command_queue_dir()
    if command_queue_dir is not None and command_queue_dir.exists():
        shutil.rmtree(command_queue_dir)
    if command_queue_dir is not None:
        command_queue_dir.mkdir(parents=True, exist_ok=True)
    global _STRICT_COMMAND_SEQUENCE
    _STRICT_COMMAND_SEQUENCE = 0
    state_tap = _state_tap_follower_from_env()
    pause_enabled = (
        _env_flag(STRICT_REPLAY_PAUSE_ON_DIVERGENCE_ENV, False)
        if pause_on_divergence is None
        else bool(pause_on_divergence)
    )
    pause_wait_timeout = (
        _env_optional_float("SPIRECOMM_STRICT_PAUSE_TIMEOUT_SECONDS")
        if pause_wait_timeout_seconds is None
        else pause_wait_timeout_seconds
    )
    (
        resolved_pause_manifest_path,
        resolved_resume_request_path,
        resolved_resume_result_path,
        resolved_pause_report_path,
    ) = _pause_control_paths(
        raw_log_target=raw_log_target,
        pause_manifest_path=pause_manifest_path,
        resume_request_path=resume_request_path,
        resume_result_path=resume_result_path,
        pause_report_path=pause_report_path,
    )

    try:
        trace = load_strict_recorded_trace(trace_path)
    except UnsupportedStrictTrace as exc:
        return _strict_failure_report(
            trace=None,
            trace_path=trace_path,
            player_class=None,
            total_steps=0,
            steps_replayed=0,
            first_failure_step=None,
            first_failure_phase=None,
            failure_kind="unsupported_legacy_trace",
            trace_action=None,
            expected_visible=None,
            actual_visible=None,
            raw_state_log_path=raw_log_target,
            runner_error=str(exc),
        )

    coordinator = Coordinator()
    coordinator.stop_after_run = True
    coordinator.register_raw_message_callback(_raw_message_logger(raw_log_target))
    player_class = _infer_character(trace, character)
    total_steps = len(trace.steps) if max_steps is None else min(len(trace.steps), max_steps)
    action_timeout = _env_float(
        "SPIRECOMM_STRICT_REPLAY_ACTION_TIMEOUT_SECONDS",
        DEFAULT_STRICT_REPLAY_ACTION_TIMEOUT_SECONDS,
    )

    try:
        _start_recorded_game(
            coordinator,
            state_tap,
            player_class=player_class,
            ascension_level=trace.ascension,
            seed=canonical_seed_string(trace.seed_long, trace.seed_str),
        )
    except Exception as exc:
        return _strict_failure_report(
            trace=trace,
            trace_path=trace_path,
            player_class=player_class,
            total_steps=total_steps,
            steps_replayed=0,
            first_failure_step=None,
            first_failure_phase=None,
            failure_kind="missing_fresh_live_state",
            trace_action=None,
            expected_visible=None,
            actual_visible=_observed_strict_state(coordinator),
            raw_state_log_path=raw_log_target,
            runner_error=repr(exc),
        )

    results: list[dict[str, Any]] = []
    sent_action_prefix: list[dict[str, Any]] = []

    def maybe_pause_or_return_failure(
        failure_report: dict[str, Any],
        *,
        next_step_index: int,
    ) -> _PauseResumeOutcome:
        if not pause_enabled or not _is_pausable_strict_failure(failure_report):
            return _PauseResumeOutcome(report=failure_report)
        return _pause_until_resume_or_abort(
            coordinator=coordinator,
            state_tap=state_tap,
            trace=trace,
            trace_path=trace_path,
            player_class=player_class,
            total_steps=total_steps,
            failure_report=failure_report,
            next_step_index=next_step_index,
            action_prefix=sent_action_prefix,
            raw_state_log_path=raw_log_target,
            pause_manifest_path=resolved_pause_manifest_path,
            resume_request_path=resolved_resume_request_path,
            resume_result_path=resolved_resume_result_path,
            pause_report_path=resolved_pause_report_path,
            pause_wait_timeout_seconds=pause_wait_timeout,
        )

    def install_resume_or_report(outcome: _PauseResumeOutcome) -> dict[str, Any] | None:
        nonlocal trace, trace_path, total_steps, index
        if outcome.report is not None:
            return outcome.report
        if outcome.trace is None or outcome.next_step_index is None:
            return {
                "trace_path": str(trace_path),
                "trace_schema": trace.trace_schema,
                "replay_mode": "strict",
                "seed_long": trace.seed_long,
                "seed_str": trace.seed_str,
                "ascension": trace.ascension,
                "character": player_class.name,
                "steps_total": total_steps,
                "steps_replayed": len(results),
                "success": False,
                "first_failure_step": None,
                "first_failure_phase": None,
                "failure_kind": "invalid_pause_resume_outcome",
                "trace_action": None,
                "expected_visible": None,
                "actual_visible": _observed_strict_state(coordinator),
                "raw_state_log_path": str(raw_log_target),
                "results": results,
                "final_state": None,
            }
        trace = outcome.trace
        trace_path = trace.path
        total_steps = len(trace.steps) if max_steps is None else min(len(trace.steps), max_steps)
        index = int(outcome.next_step_index)
        return None

    index = 0
    while index < total_steps:
        step = trace.steps[index]
        _append_strict_debug(
            raw_log_target,
            {
                "event": "step_begin",
                "step": step.step,
                "phase": step.phase,
                "raw_message_sequence": coordinator.raw_message_sequence,
                "game_is_ready": coordinator.game_is_ready,
                "strict_action": step.strict_action,
            },
        )
        live_pre = _live_strict_state(coordinator, phase_hint=step.phase)
        allow_pre_shop_inventory_mismatch = _allows_shop_inventory_mismatch_before_action(step)
        pre_sync_commands: list[str] = []
        if not _strict_state_matches(
            step.strict_pre_state,
            live_pre,
            allow_shop_inventory_mismatch=allow_pre_shop_inventory_mismatch,
        ):
            transition_commands, transitioned_live_pre = _advance_visible_neow_transition(
                coordinator,
                state_tap,
                expected_state=step.strict_pre_state,
                timeout_seconds=action_timeout,
                debug_log_path=raw_log_target,
                step=step.step,
            )
            if transition_commands:
                pre_sync_commands.extend(transition_commands)
                live_pre = transitioned_live_pre or _live_strict_state(coordinator, phase_hint=step.phase)
        if not _strict_state_matches(
            step.strict_pre_state,
            live_pre,
            allow_shop_inventory_mismatch=allow_pre_shop_inventory_mismatch,
        ):
            transition_commands, transitioned_live_pre = _advance_visible_shop_room_transition(
                coordinator,
                state_tap,
                expected_state=step.strict_pre_state,
                timeout_seconds=action_timeout,
                debug_log_path=raw_log_target,
                step=step.step,
            )
            if transition_commands:
                pre_sync_commands.extend(transition_commands)
                live_pre = transitioned_live_pre or _live_strict_state(coordinator, phase_hint=step.phase)
        pre_state_requires_ready = not _choice_pre_state_can_use_unready_card_reward(
            coordinator,
            step.strict_pre_state,
            step.strict_action,
        )
        if (
            not _strict_state_matches(
                step.strict_pre_state,
                live_pre,
                allow_shop_inventory_mismatch=allow_pre_shop_inventory_mismatch,
            )
            or (pre_state_requires_ready and not coordinator.game_is_ready)
        ):
            matched_ready_pre_state, saw_fresh_ready_pre_state, ready_live_pre = _wait_for_expected_post_state(
                coordinator,
                state_tap,
                after_sequence=coordinator.raw_message_sequence,
                expected_state=step.strict_pre_state,
                phase_hint=step.phase,
                timeout_seconds=action_timeout,
                context=f"strict replay waiting for ready pre-state before step {step.step}",
                require_ready=pre_state_requires_ready,
                allow_shop_inventory_mismatch=allow_pre_shop_inventory_mismatch,
            )
            actual_pre = ready_live_pre or _observed_strict_state(coordinator)
            if not matched_ready_pre_state:
                failure_report = _strict_failure_report(
                    trace=trace,
                    trace_path=trace_path,
                    player_class=player_class,
                    total_steps=total_steps,
                    steps_replayed=len(results),
                    first_failure_step=step.step,
                    first_failure_phase=step.phase,
                    failure_kind=_classify_neow_choice_failure_kind(
                        expected_state=step.strict_pre_state,
                        actual_state=actual_pre,
                        saw_fresh_state=saw_fresh_ready_pre_state,
                        mismatch_kind="pre_state_mismatch",
                        missing_kind="missing_fresh_live_state",
                    ),
                    trace_action=step.strict_action,
                    expected_visible=step.strict_pre_state,
                    actual_visible=actual_pre,
                    raw_state_log_path=raw_log_target,
                    runner_error=(None if saw_fresh_ready_pre_state else coordinator.last_error),
                )
                pause_report = install_resume_or_report(
                    maybe_pause_or_return_failure(
                        failure_report,
                        next_step_index=index,
                    )
                )
                if pause_report is not None:
                    return pause_report
                continue
            live_pre = ready_live_pre or _observed_strict_state(coordinator)
        if (
            _decision_pre_state_requires_pipe_sync(step.strict_pre_state)
            and not _current_pipe_ready_matches(
                coordinator,
                step.strict_pre_state,
                phase_hint=step.phase,
                require_ready=pre_state_requires_ready,
                allow_shop_inventory_mismatch=allow_pre_shop_inventory_mismatch,
            )
        ):
            matched_pipe_pre_state, saw_fresh_pipe_pre_state, pipe_live_pre = _wait_for_expected_pipe_state(
                coordinator,
                expected_state=step.strict_pre_state,
                phase_hint=step.phase,
                timeout_seconds=action_timeout,
                require_ready=pre_state_requires_ready,
                allow_shop_inventory_mismatch=allow_pre_shop_inventory_mismatch,
            )
            actual_pipe_pre = pipe_live_pre or _observed_strict_state(coordinator)
            if not matched_pipe_pre_state:
                failure_report = _strict_failure_report(
                    trace=trace,
                    trace_path=trace_path,
                    player_class=player_class,
                    total_steps=total_steps,
                    steps_replayed=len(results),
                    first_failure_step=step.step,
                    first_failure_phase=step.phase,
                    failure_kind=_classify_neow_choice_failure_kind(
                        expected_state=step.strict_pre_state,
                        actual_state=actual_pipe_pre,
                        saw_fresh_state=saw_fresh_pipe_pre_state,
                        mismatch_kind="pre_state_mismatch",
                        missing_kind="command_channel_state_not_observed",
                    ),
                    trace_action=step.strict_action,
                    expected_visible=step.strict_pre_state,
                    actual_visible=actual_pipe_pre,
                    raw_state_log_path=raw_log_target,
                    runner_error=(None if saw_fresh_pipe_pre_state else coordinator.last_error),
                )
                pause_report = install_resume_or_report(
                    maybe_pause_or_return_failure(
                        failure_report,
                        next_step_index=index,
                    )
                )
                if pause_report is not None:
                    return pause_report
                continue
            live_pre = pipe_live_pre or _observed_strict_state(coordinator)
        _append_strict_debug(
            raw_log_target,
            {
                "event": "pre_state_ready",
                "step": step.step,
                "phase": step.phase,
                "raw_message_sequence": coordinator.raw_message_sequence,
                "game_is_ready": coordinator.game_is_ready,
                "communication_source": getattr(coordinator, "last_communication_source", None),
                "pre_sync_commands": pre_sync_commands,
                "live_pre": live_pre,
            },
        )
        if _should_delay_before_strict_action(step):
            _append_strict_debug(
                raw_log_target,
                {
                    "event": "visible_screen_action_settle_sleep",
                    "step": step.step,
                    "phase": step.phase,
                    "sleep_seconds": STRICT_REPLAY_VISIBLE_SCREEN_ACTION_SETTLE_SECONDS,
                },
            )
            time.sleep(STRICT_REPLAY_VISIBLE_SCREEN_ACTION_SETTLE_SECONDS)
        try:
            before_sequence = coordinator.raw_message_sequence
            action_kind, commands = _execute_strict_action(
                coordinator,
                live_pre_state=live_pre,
                strict_action=step.strict_action,
            )
            sent_action_prefix.append(
                _strict_action_prefix_entry(
                    step,
                    action_kind=action_kind,
                    commands=commands,
                )
            )
            setattr(coordinator, "strict_last_action_sequence", before_sequence)
            _append_strict_debug(
                raw_log_target,
                {
                    "event": "action_dispatched",
                    "step": step.step,
                    "phase": step.phase,
                    "raw_message_sequence_before_action": before_sequence,
                    "raw_message_sequence_after_action": coordinator.raw_message_sequence,
                    "game_is_ready_after_action": coordinator.game_is_ready,
                    "action_kind": action_kind,
                    "commands": commands,
                },
            )
        except Exception as exc:
            _append_strict_debug(
                raw_log_target,
                {
                    "event": "action_dispatch_error",
                    "step": step.step,
                    "phase": step.phase,
                    "error": repr(exc),
                },
            )
            return _strict_failure_report(
                trace=trace,
                trace_path=trace_path,
                player_class=player_class,
                total_steps=total_steps,
                steps_replayed=len(results),
                first_failure_step=step.step,
                first_failure_phase=step.phase,
                failure_kind="action_translation_failure",
                trace_action=step.strict_action,
                expected_visible=step.strict_pre_state,
                actual_visible=live_pre,
                raw_state_log_path=raw_log_target,
                runner_error=repr(exc),
            )

        gameplay_settle_seconds = _gameplay_command_settle_seconds_for_step(step)
        if gameplay_settle_seconds > 0:
            time.sleep(gameplay_settle_seconds)
        next_step = trace.steps[index + 1] if index + 1 < total_steps else None
        card_select_transition_commands: list[str] = []
        if _can_collapse_card_reward_neow_continue(step, next_step):
            try:
                transition_before_sequence = coordinator.raw_message_sequence
                transition_action_kind, transition_commands = _execute_strict_action(
                    coordinator,
                    live_pre_state=next_step.strict_pre_state,
                    strict_action=next_step.strict_action,
                )
                sent_action_prefix.append(
                    _strict_action_prefix_entry(
                        next_step,
                        action_kind=transition_action_kind,
                        commands=transition_commands,
                    )
                )
                _append_strict_debug(
                    raw_log_target,
                    {
                        "event": "collapsed_neow_continue_dispatched",
                        "step": step.step,
                        "collapsed_step": next_step.step,
                        "raw_message_sequence_before_action": transition_before_sequence,
                        "raw_message_sequence_after_action": coordinator.raw_message_sequence,
                        "action_kind": transition_action_kind,
                        "commands": transition_commands,
                    },
                )
            except Exception as exc:
                _append_strict_debug(
                    raw_log_target,
                    {
                        "event": "collapsed_neow_continue_dispatch_error",
                        "step": step.step,
                        "collapsed_step": next_step.step,
                        "error": repr(exc),
                    },
                )
                return _strict_failure_report(
                    trace=trace,
                    trace_path=trace_path,
                    player_class=player_class,
                    total_steps=total_steps,
                    steps_replayed=len(results),
                    first_failure_step=next_step.step,
                    first_failure_phase=next_step.phase,
                    failure_kind="action_translation_failure",
                    trace_action=next_step.strict_action,
                    expected_visible=next_step.strict_pre_state,
                    actual_visible=_observed_strict_state(coordinator),
                    raw_state_log_path=raw_log_target,
                    runner_error=repr(exc),
                )

            transition_settle_seconds = _gameplay_command_settle_seconds_for_step(next_step)
            if transition_settle_seconds > 0:
                time.sleep(transition_settle_seconds)
            matched_collapsed_post_state, saw_fresh_collapsed_post_state, live_collapsed_post = _wait_for_expected_post_state(
                coordinator,
                state_tap,
                after_sequence=transition_before_sequence,
                expected_state=next_step.strict_post_state,
                phase_hint=next_step.post_phase or next_step.phase,
                timeout_seconds=action_timeout,
                context=f"strict replay waiting for collapsed post-state after step {next_step.step}",
                require_ready=_expected_state_requires_ready(next_step.strict_post_state),
                initial_state_probe_delay=transition_settle_seconds,
                initial_wait_probe_delay=transition_settle_seconds * 2.0,
                allow_shop_inventory_mismatch=_allows_shop_inventory_mismatch_after_action(next_step),
            )
            actual_collapsed_post = live_collapsed_post or _observed_strict_state(coordinator)
            if not matched_collapsed_post_state:
                failure_report = _strict_failure_report(
                    trace=trace,
                    trace_path=trace_path,
                    player_class=player_class,
                    total_steps=total_steps,
                    steps_replayed=len(results),
                    first_failure_step=next_step.step,
                    first_failure_phase=next_step.phase,
                    failure_kind=_classify_neow_choice_failure_kind(
                        expected_state=next_step.strict_post_state,
                        actual_state=actual_collapsed_post,
                        saw_fresh_state=saw_fresh_collapsed_post_state,
                        mismatch_kind="post_state_mismatch",
                        missing_kind="missing_fresh_live_state",
                    ),
                    trace_action=next_step.strict_action,
                    expected_visible=next_step.strict_post_state,
                    actual_visible=actual_collapsed_post,
                    raw_state_log_path=raw_log_target,
                    runner_error=(None if saw_fresh_collapsed_post_state else coordinator.last_error),
                )
                pause_report = install_resume_or_report(
                    maybe_pause_or_return_failure(
                        failure_report,
                        next_step_index=index + 2,
                    )
                )
                if pause_report is not None:
                    return pause_report
                continue

            results.append(
                {
                    "step": step.step,
                    "phase": step.phase,
                    "pre_sync_commands": pre_sync_commands,
                    "action_kind": action_kind,
                    "commands": commands,
                    "collapsed_next_step": next_step.step,
                    "success": True,
                }
            )
            results.append(
                {
                    "step": next_step.step,
                    "phase": next_step.phase,
                    "pre_sync_commands": ["collapsed_from_previous_step"],
                    "action_kind": transition_action_kind,
                    "commands": transition_commands,
                    "success": True,
                }
            )
            index += 2
            continue
        matched_post_state, saw_fresh_post_state, live_post = _wait_for_expected_post_state(
            coordinator,
            state_tap,
            after_sequence=before_sequence,
            expected_state=step.strict_post_state,
            phase_hint=step.post_phase or step.phase,
            timeout_seconds=action_timeout,
            context=f"strict replay waiting for post-state after step {step.step}",
            require_ready=_expected_state_requires_ready(step.strict_post_state),
            initial_state_probe_delay=gameplay_settle_seconds,
            initial_wait_probe_delay=gameplay_settle_seconds * 2.0,
            confirm_card_select_transition=(
                (
                    step.phase == "CARD_SELECT"
                    and str(step.strict_action.get("kind") or "") == "choose_by_index"
                    and str(step.strict_post_state.get("phase") or "") != "CARD_SELECT"
                )
                or (
                    step.phase == "SHOP"
                    and str(step.strict_action.get("kind") or "") == "choose_by_name"
                    and str(step.strict_action.get("item_kind") or "").lower() == "purge"
                    and step.strict_action.get("target_index") is not None
                    and str(step.strict_post_state.get("phase") or "") != "CARD_SELECT"
                )
            ),
            card_select_transition_choice_index=(
                int(step.strict_action["target_index"])
                if (
                    step.phase == "SHOP"
                    and str(step.strict_action.get("kind") or "") == "choose_by_name"
                    and str(step.strict_action.get("item_kind") or "").lower() == "purge"
                    and step.strict_action.get("target_index") is not None
                    and str(step.strict_post_state.get("phase") or "") != "CARD_SELECT"
                )
                else None
            ),
            transition_commands=card_select_transition_commands,
            debug_log_path=raw_log_target,
            step=step.step,
            allow_shop_inventory_mismatch=_allows_shop_inventory_mismatch_after_action(step),
        )
        if card_select_transition_commands:
            commands.extend(card_select_transition_commands)
        actual_post = live_post or _observed_strict_state(coordinator)
        if not matched_post_state:
            failure_report = _strict_failure_report(
                trace=trace,
                trace_path=trace_path,
                player_class=player_class,
                total_steps=total_steps,
                steps_replayed=len(results),
                first_failure_step=step.step,
                first_failure_phase=step.phase,
                failure_kind=_classify_neow_choice_failure_kind(
                    expected_state=step.strict_post_state,
                    actual_state=actual_post,
                    saw_fresh_state=saw_fresh_post_state,
                    mismatch_kind="post_state_mismatch",
                    missing_kind="missing_fresh_live_state",
                ),
                trace_action=step.strict_action,
                expected_visible=step.strict_post_state,
                actual_visible=actual_post,
                raw_state_log_path=raw_log_target,
                runner_error=(None if saw_fresh_post_state else coordinator.last_error),
            )
            pause_report = install_resume_or_report(
                maybe_pause_or_return_failure(
                    failure_report,
                    next_step_index=index + 1,
                )
            )
            if pause_report is not None:
                return pause_report
            continue

        results.append(
            {
                "step": step.step,
                "phase": step.phase,
                "pre_sync_commands": pre_sync_commands,
                "action_kind": action_kind,
                "commands": commands,
                "success": True,
            }
        )
        index += 1

    final_state = serialize_game_state(coordinator.last_game_state) if coordinator.last_game_state is not None else None
    return {
        "trace_path": str(trace_path),
        "trace_schema": trace.trace_schema,
        "replay_mode": "strict",
        "seed_long": trace.seed_long,
        "seed_str": trace.seed_str,
        "ascension": trace.ascension,
        "character": player_class.name,
        "steps_total": len(trace.steps),
        "steps_replayed": max(len(results), index),
        "success": True,
        "first_failure_step": None,
        "first_failure_phase": None,
        "failure_kind": None,
        "trace_action": None,
        "expected_visible": None,
        "actual_visible": None,
        "raw_state_log_path": str(raw_log_target),
        "command_journal_path": str(command_journal_path) if command_journal_path is not None else None,
        "command_queue_dir": str(command_queue_dir) if command_queue_dir is not None else None,
        "results": results,
        "final_state": final_state,
    }


def render_strict_replay_report_summary(report: dict[str, Any]) -> str:
    lines = [
        "Strict Recorded Replay Summary",
        f"trace_path: {report.get('trace_path')}",
        f"trace_schema: {report.get('trace_schema')}",
        f"replay_mode: {report.get('replay_mode')}",
        f"success: {report.get('success')}",
        f"paused: {report.get('paused')}",
        f"steps_replayed: {report.get('steps_replayed')}/{report.get('steps_total')}",
        f"first_failure_step: {report.get('first_failure_step')}",
        f"first_failure_phase: {report.get('first_failure_phase')}",
        f"failure_kind: {report.get('failure_kind')}",
        f"raw_state_log_path: {report.get('raw_state_log_path')}",
    ]
    if report.get("runner_error"):
        lines.append(f"runner_error: {report['runner_error']}")
    if report.get("pause_manifest_path"):
        lines.append(f"pause_manifest_path: {report['pause_manifest_path']}")
    if report.get("resume_request_path"):
        lines.append(f"resume_request_path: {report['resume_request_path']}")
    if report.get("trace_action") is not None:
        lines.append("trace_action:")
        lines.append(json.dumps(report["trace_action"], ensure_ascii=False, indent=2))
    if report.get("expected_visible") is not None:
        lines.append("expected_visible:")
        lines.append(json.dumps(report["expected_visible"], ensure_ascii=False, indent=2))
    if report.get("actual_visible") is not None:
        lines.append("actual_visible:")
        lines.append(json.dumps(report["actual_visible"], ensure_ascii=False, indent=2))
    return "\n".join(lines) + "\n"
