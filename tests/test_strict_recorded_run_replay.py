from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spirecomm.ai.strict_recorded_run_replay import StrictRecordedTraceStep
from spirecomm.ai.strict_recorded_run_replay import _card_select_choice_index_for_deck_target
from spirecomm.ai.strict_recorded_run_replay import _execute_strict_action
from spirecomm.ai.strict_recorded_run_replay import _gameplay_command_settle_seconds_for_step
from spirecomm.ai.strict_recorded_run_replay import _queue_strict_gameplay_command
from spirecomm.ai.strict_recorded_run_replay import _start_recorded_game
from spirecomm.ai.strict_recorded_run_replay import _send_strict_command
from spirecomm.ai.strict_recorded_run_replay import _stabilize_pre_state_after_bootstrap
from spirecomm.ai.strict_recorded_run_replay import replay_recorded_run_strict
from spirecomm.ai.strict_recorded_run_replay import _StateTapFollower
from spirecomm.ai.strict_recorded_run_replay import _sync_known_pre_state
from spirecomm.ai.strict_recorded_run_replay import _wait_for_expected_post_state
from spirecomm.ai.strict_recorded_run_replay import load_strict_recorded_trace
from spirecomm.ai.strict_recorded_run_replay import validate_strict_pause_resume_trace
from spirecomm.ai.strict_trace import (
    STRICT_TRACE_SCHEMA,
    build_strict_step_payload,
    normalize_verbose_state_for_strict,
)
from spirecomm.spire.character import PlayerClass


class _FakeCoordinator:
    raw_states: list[dict] = []

    @staticmethod
    def _coerce_entry(entry):
        if isinstance(entry, tuple):
            state, ready = entry
            return dict(state), bool(ready)
        return dict(entry), True

    def __init__(self):
        if not type(self).raw_states:
            raise RuntimeError("raw_states must be populated before constructing _FakeCoordinator")
        self._states = [self._coerce_entry(state) for state in type(self).raw_states]
        self.last_raw_game_state, initial_ready = self._states[0]
        self.last_game_state = None
        self.last_communication_state = {
            "ready_for_command": initial_ready,
            "in_game": True,
            "error": None,
            "available_commands": ["choose", "state", "wait"],
            "game_state": self.last_raw_game_state,
        }
        self.last_error = None
        self.raw_message_sequence = 1
        self.game_is_ready = initial_ready
        self.in_game = True
        self.stop_after_run = True
        self._raw_callback = None
        self.commands: list[str] = []

    def register_raw_message_callback(self, callback):
        self._raw_callback = callback

    def ingest_communication_state(self, communication_state, *, raw_message=None, source="tap"):
        self.last_raw_game_state = communication_state.get("game_state")
        self.last_communication_state = dict(communication_state)
        self.raw_message_sequence += 1
        self.last_error = communication_state.get("error")
        self.game_is_ready = bool(communication_state.get("ready_for_command"))
        self.in_game = bool(communication_state.get("in_game", self.in_game))
        if self._raw_callback is not None:
            self._raw_callback(
                {
                    "sequence": self.raw_message_sequence,
                    "raw_message": raw_message,
                    "communication_state": communication_state,
                    "source": source,
                }
            )

    def signal_ready(self):
        return None

    def send_message(self, message):
        self.commands.append(message)
        self.game_is_ready = False

    def send_message_immediate(self, message):
        self.send_message(message)

    def receive_game_state_update(self, block=False, perform_callbacks=False, timeout=None):
        del block, perform_callbacks, timeout
        if not self._states:
            return False
        if len(self._states) == 1:
            return False
        self._states.pop(0)
        self.last_raw_game_state, next_ready = self._states[0]
        self.raw_message_sequence += 1
        self.game_is_ready = next_ready
        self.last_communication_state = {
            "ready_for_command": next_ready,
            "in_game": True,
            "error": None,
            "available_commands": ["choose", "state", "wait"],
            "game_state": self.last_raw_game_state,
        }
        if self._raw_callback is not None:
            self._raw_callback(
                {
                    "sequence": self.raw_message_sequence,
                    "raw_message": "{}",
                    "communication_state": {
                        "ready_for_command": next_ready,
                        "in_game": True,
                        "error": None,
                        "available_commands": ["choose", "state", "wait"],
                        "game_state": self.last_raw_game_state,
                    },
                }
            )
        return True


class StrictRecordedRunReplayTest(unittest.TestCase):
    def test_shop_purge_card_select_maps_deck_index_to_visible_grid_choice(self):
        deck = [
            {"card_id": "Strike_R", "name": "Strike_R", "upgrades": 1},
            {"card_id": "True Grit", "name": "True Grit", "upgrades": 1},
            {"card_id": "Parasite", "name": "Parasite", "upgrades": 0},
        ]
        live_state = {
            "deck": deck,
            "screen_state": {
                "for_purge": True,
                "choices": [
                    {"choice_index": 0, "card": dict(deck[0])},
                    # Bottled True Grit is absent from STS's purge grid.
                    {"choice_index": 1, "card": dict(deck[2])},
                ],
            },
        }

        self.assertEqual(_card_select_choice_index_for_deck_target(live_state, 2), 1)

    def _minimal_combat_raw_state(self, *, hp=80, turn=0):
        return {
            "phase": "COMBAT",
            "floor": 1,
            "act": 1,
            "current_hp": hp,
            "max_hp": 80,
            "gold": 0,
            "deck": [],
            "relics": [],
            "potions": [],
            "combat_state": {
                "turn": turn,
                "player": {"block": 0, "powers": []},
                "monsters": [],
                "hand": [],
                "draw_pile": [],
                "discard_pile": [],
                "exhaust_pile": [],
            },
        }

    def _write_minimal_strict_trace(self, path: Path, *, pre_state: dict, post_state: dict, seed=1):
        payload = {
            "trace_schema": STRICT_TRACE_SCHEMA,
            "seed_long": seed,
            "seed_str": str(seed),
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "COMBAT",
                    "floor": 1,
                    "action": {"kind": "end"},
                    "strict_action": {"kind": "end_turn", "phase": "COMBAT", "raw_kind": "end"},
                    "strict_pre_state": pre_state,
                    "strict_post_state": post_state,
                    "post_phase": "COMBAT",
                }
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_visible_screen_steps_use_full_gameplay_settle_window(self):
        step = StrictRecordedTraceStep(
            step=1,
            phase="CARD_REWARD",
            floor=0,
            action={},
            strict_action={},
            strict_pre_state={},
            strict_post_state={"phase": "NEOW"},
            post_phase="NEOW",
        )

        self.assertGreaterEqual(
            _gameplay_command_settle_seconds_for_step(step),
            0.5,
        )

    def test_validate_strict_pause_resume_trace_accepts_matching_prefix_and_live_state(self):
        pre = normalize_verbose_state_for_strict(self._minimal_combat_raw_state(), phase_hint="COMBAT")
        post = normalize_verbose_state_for_strict(self._minimal_combat_raw_state(turn=1), phase_hint="COMBAT")
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.json"
            self._write_minimal_strict_trace(trace_path, pre_state=pre, post_state=post)
            trace = load_strict_recorded_trace(trace_path)
            manifest = {
                "seed_long": 1,
                "ascension": 0,
                "character": "IRONCLAD",
                "next_step_to_send": 0,
                "actual_visible": pre,
                "action_prefix": [],
            }

            result = validate_strict_pause_resume_trace(manifest, trace)

        self.assertTrue(result["ok"], msg=str(result))

    def test_validate_strict_pause_resume_trace_rejects_action_prefix_mismatch(self):
        pre = normalize_verbose_state_for_strict(self._minimal_combat_raw_state(), phase_hint="COMBAT")
        post = normalize_verbose_state_for_strict(self._minimal_combat_raw_state(turn=1), phase_hint="COMBAT")
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.json"
            self._write_minimal_strict_trace(trace_path, pre_state=pre, post_state=post)
            trace = load_strict_recorded_trace(trace_path)
            manifest = {
                "seed_long": 1,
                "ascension": 0,
                "character": "IRONCLAD",
                "next_step_to_send": 0,
                "actual_visible": pre,
                "action_prefix": [
                    {
                        "step": 0,
                        "phase": "COMBAT",
                        "strict_action": {"kind": "choose_by_index", "choice_index": 0},
                    }
                ],
            }

            result = validate_strict_pause_resume_trace(manifest, trace)

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "action_prefix_mismatch")

    def test_validate_strict_pause_resume_trace_rejects_live_state_mismatch(self):
        pre = normalize_verbose_state_for_strict(self._minimal_combat_raw_state(), phase_hint="COMBAT")
        post = normalize_verbose_state_for_strict(self._minimal_combat_raw_state(turn=1), phase_hint="COMBAT")
        wrong_live = normalize_verbose_state_for_strict(self._minimal_combat_raw_state(hp=79), phase_hint="COMBAT")
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.json"
            self._write_minimal_strict_trace(trace_path, pre_state=pre, post_state=post)
            trace = load_strict_recorded_trace(trace_path)
            manifest = {
                "seed_long": 1,
                "ascension": 0,
                "character": "IRONCLAD",
                "next_step_to_send": 0,
                "actual_visible": wrong_live,
                "action_prefix": [],
            }

            result = validate_strict_pause_resume_trace(manifest, trace)

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "live_state_mismatch")

    def test_pause_manifest_for_post_state_mismatch_resumes_after_sent_action(self):
        raw_pre = self._minimal_combat_raw_state()
        pre = normalize_verbose_state_for_strict(raw_pre, phase_hint="COMBAT")
        post = normalize_verbose_state_for_strict(self._minimal_combat_raw_state(turn=1), phase_hint="COMBAT")
        raw_wrong_post = self._minimal_combat_raw_state(hp=79, turn=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            trace_path = tmp / "strict_trace.json"
            manifest_path = tmp / "pause.json"
            report_path = tmp / "report.json"
            self._write_minimal_strict_trace(trace_path, pre_state=pre, post_state=post)
            _FakeCoordinator.raw_states = [raw_pre, raw_wrong_post]
            with patch("spirecomm.ai.strict_recorded_run_replay.Coordinator", _FakeCoordinator):
                with patch("spirecomm.ai.strict_recorded_run_replay._start_recorded_game", return_value=None):
                    with patch("spirecomm.ai.strict_recorded_run_replay.STRICT_REPLAY_PAUSE_POLL_INTERVAL_SECONDS", 0.01):
                        with patch.dict("os.environ", {"SPIRECOMM_STRICT_REPLAY_ACTION_TIMEOUT_SECONDS": "0.01"}, clear=False):
                            report = replay_recorded_run_strict(
                                trace_path=trace_path,
                                pause_on_divergence=True,
                                pause_manifest_path=manifest_path,
                                resume_request_path=tmp / "resume.json",
                                resume_result_path=tmp / "resume_result.json",
                                pause_report_path=report_path,
                                pause_wait_timeout_seconds=0.01,
                            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(report["failure_kind"], "replay_pause_timeout")
        self.assertEqual(manifest["next_step_to_send"], 1)
        self.assertEqual(len(manifest["action_prefix"]), 1)
        self.assertEqual(manifest["action_prefix"][0]["strict_action"]["kind"], "end_turn")

    def test_pause_manifest_for_pre_state_mismatch_resumes_before_unsent_action(self):
        raw_expected_pre = self._minimal_combat_raw_state()
        expected_pre = normalize_verbose_state_for_strict(raw_expected_pre, phase_hint="COMBAT")
        post = normalize_verbose_state_for_strict(self._minimal_combat_raw_state(turn=1), phase_hint="COMBAT")
        raw_bad_pre = self._minimal_combat_raw_state(hp=79)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            trace_path = tmp / "strict_trace.json"
            manifest_path = tmp / "pause.json"
            report_path = tmp / "report.json"
            self._write_minimal_strict_trace(trace_path, pre_state=expected_pre, post_state=post)
            _FakeCoordinator.raw_states = [raw_bad_pre]
            with patch("spirecomm.ai.strict_recorded_run_replay.Coordinator", _FakeCoordinator):
                with patch("spirecomm.ai.strict_recorded_run_replay._start_recorded_game", return_value=None):
                    with patch("spirecomm.ai.strict_recorded_run_replay.STRICT_REPLAY_PAUSE_POLL_INTERVAL_SECONDS", 0.01):
                        with patch.dict("os.environ", {"SPIRECOMM_STRICT_REPLAY_ACTION_TIMEOUT_SECONDS": "0.01"}, clear=False):
                            report = replay_recorded_run_strict(
                                trace_path=trace_path,
                                pause_on_divergence=True,
                                pause_manifest_path=manifest_path,
                                resume_request_path=tmp / "resume.json",
                                resume_result_path=tmp / "resume_result.json",
                                pause_report_path=report_path,
                                pause_wait_timeout_seconds=0.01,
                            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(report["failure_kind"], "replay_pause_timeout")
        self.assertEqual(manifest["next_step_to_send"], 0)
        self.assertEqual(manifest["action_prefix"], [])

    def test_send_strict_command_uses_journal_audit_and_normal_send_in_explicit_journal_mode(self):
        class _Coordinator:
            def __init__(self):
                self.sent: list[str] = []
                self.immediate: list[str] = []
                self.game_is_ready = True

            def send_message(self, message):
                self.sent.append(message)

            def send_message_immediate(self, message):
                self.immediate.append(message)

        with tempfile.TemporaryDirectory() as tmpdir:
            journal_path = Path(tmpdir) / "strict_commands.log"
            queue_dir = Path(tmpdir) / "strict_command_queue"
            coordinator = _Coordinator()
            with patch.dict(
                "os.environ",
                {
                    "SPIRECOMM_STRICT_COMMAND_TRANSPORT": "journal",
                    "SPIRECOMM_STRICT_COMMAND_JOURNAL_PATH": str(journal_path),
                    "SPIRECOMM_STRICT_COMMAND_QUEUE_DIR": str(queue_dir),
                },
                clear=False,
            ):
                _send_strict_command(coordinator, "choose 0")
                _send_strict_command(coordinator, "state")
                _send_strict_command(coordinator, "ready")

            self.assertEqual(coordinator.sent, ["state", "ready"])
            self.assertEqual(coordinator.immediate, [])
            self.assertFalse(coordinator.game_is_ready)
            self.assertEqual(
                journal_path.read_text(encoding="utf-8").splitlines(),
                ["choose 0"],
            )
            self.assertEqual(list(queue_dir.glob("*.cmd")), [])

    def test_send_strict_command_can_use_explicit_file_queue_transport(self):
        class _Coordinator:
            def __init__(self):
                self.sent: list[str] = []
                self.game_is_ready = True

            def send_message(self, message):
                self.sent.append(message)

        with tempfile.TemporaryDirectory() as tmpdir:
            journal_path = Path(tmpdir) / "strict_commands.log"
            queue_dir = Path(tmpdir) / "strict_command_queue"
            coordinator = _Coordinator()
            with patch.dict(
                "os.environ",
                {
                    "SPIRECOMM_STRICT_COMMAND_TRANSPORT": "file_queue",
                    "SPIRECOMM_STRICT_COMMAND_JOURNAL_PATH": str(journal_path),
                    "SPIRECOMM_STRICT_COMMAND_QUEUE_DIR": str(queue_dir),
                },
                clear=False,
            ):
                _send_strict_command(coordinator, "choose 0")
                _send_strict_command(coordinator, "state")

            self.assertEqual(coordinator.sent, ["state"])
            self.assertFalse(coordinator.game_is_ready)
            self.assertFalse(journal_path.exists())
            self.assertEqual(
                [path.read_text(encoding="utf-8").strip() for path in sorted(queue_dir.glob("*.cmd"))],
                ["choose 0"],
            )

    def test_execute_strict_shop_leave_sends_leave_command(self):
        class _Coordinator:
            def __init__(self):
                self.sent: list[str] = []
                self.game_is_ready = True

            def send_message(self, message):
                self.sent.append(message)

        coordinator = _Coordinator()
        with patch.dict("os.environ", {"SPIRECOMM_STRICT_COMMAND_TRANSPORT": ""}, clear=False):
            kind, commands = _execute_strict_action(
                coordinator,
                live_pre_state={"phase": "SHOP"},
                strict_action={"kind": "leave", "name": "Leave", "item_kind": "leave"},
            )

        self.assertEqual(kind, "leave")
        self.assertEqual(commands, ["leave"])
        self.assertEqual(coordinator.sent, ["leave"])

    def test_execute_legacy_strict_shop_leave_sends_leave_command(self):
        class _Coordinator:
            def __init__(self):
                self.sent: list[str] = []
                self.game_is_ready = True

            def send_message(self, message):
                self.sent.append(message)

        coordinator = _Coordinator()
        with patch.dict("os.environ", {"SPIRECOMM_STRICT_COMMAND_TRANSPORT": ""}, clear=False):
            kind, commands = _execute_strict_action(
                coordinator,
                live_pre_state={"phase": "SHOP"},
                strict_action={"kind": "choose_by_name", "name": "Leave", "item_kind": "leave"},
            )

        self.assertEqual(kind, "choose_by_name")
        self.assertEqual(commands, ["leave"])
        self.assertEqual(coordinator.sent, ["leave"])

    def test_execute_strict_normal_treasure_choice_sends_choose_command(self):
        class _Coordinator:
            def __init__(self):
                self.sent: list[str] = []
                self.game_is_ready = True

            def send_message(self, message):
                self.sent.append(message)

        coordinator = _Coordinator()
        with patch.dict("os.environ", {"SPIRECOMM_STRICT_COMMAND_TRANSPORT": ""}, clear=False):
            kind, commands = _execute_strict_action(
                coordinator,
                live_pre_state={"phase": "TREASURE", "screen_state": {"chest_type": "MediumChest", "chest_open": False}},
                strict_action={"kind": "choose_by_index", "choice_index": 0},
            )

        self.assertEqual(kind, "choose_by_index")
        self.assertEqual(commands, ["choose 0"])
        self.assertEqual(coordinator.sent, ["choose 0"])

    def test_execute_strict_open_boss_chest_sends_proceed_command(self):
        class _Coordinator:
            def __init__(self):
                self.sent: list[str] = []
                self.game_is_ready = True

            def send_message(self, message):
                self.sent.append(message)

        coordinator = _Coordinator()
        with patch.dict("os.environ", {"SPIRECOMM_STRICT_COMMAND_TRANSPORT": ""}, clear=False):
            kind, commands = _execute_strict_action(
                coordinator,
                live_pre_state={"phase": "TREASURE", "screen_state": {"chest_type": "BossChest", "chest_open": True}},
                strict_action={"kind": "choose_by_index", "choice_index": 0},
            )

        self.assertEqual(kind, "choose_by_index")
        self.assertEqual(commands, ["proceed"])
        self.assertEqual(coordinator.sent, ["proceed"])

    def test_queue_strict_gameplay_command_can_prefer_immediate_send(self):
        class _Coordinator:
            def __init__(self):
                self.sent: list[str] = []
                self.immediate: list[str] = []
                self.game_is_ready = True

            def send_message(self, message):
                self.sent.append(message)

            def send_message_immediate(self, message):
                self.immediate.append(message)

        with tempfile.TemporaryDirectory() as tmpdir:
            journal_path = Path(tmpdir) / "strict_commands.log"
            coordinator = _Coordinator()
            with patch.dict(
                "os.environ",
                {
                    "SPIRECOMM_STRICT_COMMAND_JOURNAL_PATH": str(journal_path),
                },
                clear=False,
            ):
                _queue_strict_gameplay_command(
                    coordinator,
                    "choose 0",
                    prefer_immediate=True,
                )

        self.assertEqual(coordinator.sent, [])
        self.assertEqual(coordinator.immediate, ["choose 0"])
        self.assertFalse(coordinator.game_is_ready)
        self.assertFalse(journal_path.exists())

    def test_queue_strict_gameplay_command_uses_immediate_send_for_card_reward(self):
        class _Coordinator:
            def __init__(self):
                self.sent: list[str] = []
                self.immediate: list[str] = []
                self.game_is_ready = True

            def send_message(self, message):
                self.sent.append(message)

            def send_message_immediate(self, message):
                self.immediate.append(message)

        with tempfile.TemporaryDirectory() as tmpdir:
            journal_path = Path(tmpdir) / "strict_commands.log"
            coordinator = _Coordinator()
            with patch.dict(
                "os.environ",
                {
                    "SPIRECOMM_STRICT_COMMAND_TRANSPORT": "stdout",
                    "SPIRECOMM_STRICT_COMMAND_JOURNAL_PATH": str(journal_path),
                },
                clear=False,
            ):
                _queue_strict_gameplay_command(
                    coordinator,
                    "choose 0",
                    live_pre_state={"phase": "CARD_REWARD"},
                )
            journal_lines = journal_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(coordinator.sent, [])
        self.assertEqual(coordinator.immediate, [])
        self.assertFalse(coordinator.game_is_ready)
        self.assertEqual(journal_lines, ["choose 0"])

    def test_queue_strict_gameplay_command_uses_immediate_send_for_card_reward_even_in_file_queue_mode(self):
        class _Coordinator:
            def __init__(self):
                self.sent: list[str] = []
                self.immediate: list[str] = []
                self.game_is_ready = True

            def send_message(self, message):
                self.sent.append(message)

            def send_message_immediate(self, message):
                self.immediate.append(message)

        with tempfile.TemporaryDirectory() as tmpdir:
            journal_path = Path(tmpdir) / "strict_commands.log"
            queue_dir = Path(tmpdir) / "strict_command_queue"
            coordinator = _Coordinator()
            with patch.dict(
                "os.environ",
                {
                    "SPIRECOMM_STRICT_COMMAND_TRANSPORT": "file_queue",
                    "SPIRECOMM_STRICT_COMMAND_JOURNAL_PATH": str(journal_path),
                    "SPIRECOMM_STRICT_COMMAND_QUEUE_DIR": str(queue_dir),
                },
                clear=False,
            ):
                _queue_strict_gameplay_command(
                    coordinator,
                    "choose 0",
                    live_pre_state={"phase": "CARD_REWARD"},
                )
            journal_lines = journal_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(coordinator.sent, [])
        self.assertEqual(coordinator.immediate, [])
        self.assertFalse(coordinator.game_is_ready)
        self.assertEqual(journal_lines, ["choose 0"])

    def test_queue_strict_gameplay_command_does_not_use_card_reward_queue_dir_when_immediate_send_is_available(self):
        class _Coordinator:
            def __init__(self):
                self.sent: list[str] = []
                self.immediate: list[str] = []
                self.game_is_ready = True

            def send_message(self, message):
                self.sent.append(message)

            def send_message_immediate(self, message):
                self.immediate.append(message)

        with tempfile.TemporaryDirectory() as tmpdir:
            journal_path = Path(tmpdir) / "strict_commands.log"
            queue_dir = Path(tmpdir) / "strict_command_queue"
            coordinator = _Coordinator()
            with patch.dict(
                "os.environ",
                {
                    "SPIRECOMM_STRICT_COMMAND_TRANSPORT": "stdout",
                    "SPIRECOMM_STRICT_COMMAND_JOURNAL_PATH": str(journal_path),
                    "SPIRECOMM_STRICT_COMMAND_QUEUE_DIR": str(queue_dir),
                },
                clear=False,
            ):
                _queue_strict_gameplay_command(
                    coordinator,
                    "choose 0",
                    live_pre_state={"phase": "CARD_REWARD"},
                )
            journal_lines = journal_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(coordinator.sent, [])
        self.assertEqual(coordinator.immediate, [])
        self.assertFalse(coordinator.game_is_ready)
        self.assertEqual(journal_lines, ["choose 0"])
        self.assertEqual(sorted(queue_dir.glob("*.cmd")), [])

    def test_state_tap_follower_buffers_partial_json_lines(self):
        class _TapCoordinator:
            def __init__(self):
                self.ingested = []
                self.raw_message_sequence = 0

            def ingest_communication_state(self, payload, *, raw_message=None, source="tap"):
                self.raw_message_sequence += 1
                self.ingested.append((payload, raw_message, source))

        with tempfile.TemporaryDirectory() as tmpdir:
            tap_path = Path(tmpdir) / "state_tap.jsonl"
            follower = _StateTapFollower(path=tap_path)
            coordinator = _TapCoordinator()
            first_half = '{"source":"state_tap","communication_state":{"ready_for_command":false'
            second_half = ',"in_game":true}}\n'
            tap_path.write_text(first_half, encoding="utf-8")
            follower.ingest_updates(coordinator)
            self.assertEqual(coordinator.ingested, [])
            with tap_path.open("a", encoding="utf-8") as handle:
                handle.write(second_half)
            ingested = follower.ingest_updates(coordinator)

        self.assertEqual(len(coordinator.ingested), 1)
        self.assertEqual(len(ingested), 1)
        payload, _raw_message, source = coordinator.ingested[0]
        self.assertEqual(source, "state_tap")
        self.assertEqual(payload["ready_for_command"], False)
        self.assertEqual(payload["in_game"], True)

    def test_sync_known_pre_state_uses_passive_tap_bootstrap_for_hidden_neow_talk(self):
        live_talk_state = normalize_verbose_state_for_strict(
            {
                "phase": "EVENT",
                "screen_type": "EVENT",
                "room_type": "NeowRoom",
                "floor": 0,
                "act": 1,
                "current_hp": 80,
                "max_hp": 80,
                "gold": 99,
                "choice_list": ["talk"],
                "screen_state": {
                    "event_id": "Neow Event",
                    "event_name": "Neow",
                    "options": [{"choice_index": 0, "label": "Talk", "text": "[Talk]"}],
                },
                "deck": [],
                "relics": [],
                "potions": [],
            },
            phase_hint="NEOW",
        )
        multi_option_state = normalize_verbose_state_for_strict(
            {
                "phase": "EVENT",
                "screen_type": "EVENT",
                "room_type": "NeowRoom",
                "floor": 0,
                "act": 1,
                "current_hp": 80,
                "max_hp": 80,
                "gold": 99,
                "choice_list": [
                    {"kind": "neow", "choice_index": 0, "label": "OPTION_0", "text": "Choose 1 of 3 random colorless cards.", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
                    {"kind": "neow", "choice_index": 1, "label": "OPTION_1", "text": "Obtain 3 random potions.", "bonus": "THREE_SMALL_POTIONS", "drawback": "NONE"},
                ],
                "screen_state": {
                    "event_id": "Neow Event",
                    "event_name": "Neow",
                    "options": [
                        {"choice_index": 0, "label": "OPTION_0", "text": "Choose 1 of 3 random colorless cards.", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
                        {"choice_index": 1, "label": "OPTION_1", "text": "Obtain 3 random potions.", "bonus": "THREE_SMALL_POTIONS", "drawback": "NONE"},
                    ],
                },
                "deck": [],
                "relics": [],
                "potions": [],
            },
            phase_hint="NEOW",
        )
        step = StrictRecordedTraceStep(
            step=0,
            phase="NEOW",
            floor=0,
            action={},
            strict_action={},
            strict_pre_state=multi_option_state,
            strict_post_state={},
            post_phase="CARD_REWARD",
        )

        class _Coordinator:
            def __init__(self):
                self.raw_message_sequence = 17
                self.game_is_ready = False
                self.commands = []

            def send_message(self, message):
                self.commands.append(message)
                self.game_is_ready = False

        coordinator = _Coordinator()
        with patch(
            "spirecomm.ai.strict_recorded_run_replay._wait_for_expected_post_state",
            return_value=(True, True, live_talk_state),
        ) as wait_post_mock, patch(
            "spirecomm.ai.strict_recorded_run_replay._wait_for_fresh_matching_state",
            return_value=(multi_option_state, multi_option_state),
        ) as wait_match_mock:
            transitioned_state, commands = _sync_known_pre_state(
                coordinator,
                object(),
                step=step,
                live_pre=live_talk_state,
                timeout_seconds=5.0,
            )

        self.assertEqual(transitioned_state, multi_option_state)
        self.assertEqual(commands, ["choose 0"])
        self.assertEqual(coordinator.commands, ["choose 0"])
        self.assertFalse(wait_post_mock.call_args.kwargs["allow_state_probe"])
        self.assertFalse(wait_post_mock.call_args.kwargs["allow_wait_probe"])
        self.assertFalse(wait_match_mock.call_args.kwargs["allow_state_probe"])
        self.assertFalse(wait_match_mock.call_args.kwargs["allow_wait_probe"])

    def test_start_recorded_game_uses_passive_tap_for_initial_in_game_wait(self):
        class _Coordinator:
            def __init__(self):
                self.in_game = False
                self.last_error = None
                self.game_is_ready = False
                self.signaled = False
                self.sent = []

            def signal_ready(self):
                self.signaled = True

            def send_message(self, message):
                self.sent.append(message)

        coordinator = _Coordinator()
        with patch(
            "spirecomm.ai.strict_recorded_run_replay._wait_for_ready_state",
            return_value=True,
        ) as wait_ready_mock, patch(
            "spirecomm.ai.strict_recorded_run_replay._wait_for_fresh_state",
            return_value=True,
        ) as wait_fresh_mock:
            _start_recorded_game(
                coordinator,
                object(),
                player_class=PlayerClass.IRONCLAD,
                ascension_level=0,
                seed="SEED123",
            )

        self.assertTrue(coordinator.signaled)
        self.assertEqual(coordinator.sent, ["start IRONCLAD 0 SEED123"])
        self.assertTrue(wait_ready_mock.called)
        self.assertFalse(wait_fresh_mock.call_args.kwargs["allow_state_probe"])

    def test_stabilize_pre_state_after_bootstrap_uses_ready_state_probe_only(self):
        expected_state = {
            "phase": "NEOW",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "screen_state": {
                "choices": [
                    {"choice_index": 0, "kind": "neow", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
                    {"choice_index": 1, "kind": "neow", "bonus": "THREE_SMALL_POTIONS", "drawback": "NONE"},
                ]
            },
            "deck": [],
            "relics": [],
            "potions": [],
            "combat_state": None,
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
        }
        step = StrictRecordedTraceStep(
            step=0,
            phase="NEOW",
            floor=0,
            action={},
            strict_action={"kind": "choose_by_index", "choice_index": 0},
            strict_pre_state=expected_state,
            strict_post_state={},
            post_phase="CARD_REWARD",
        )

        class _Coordinator:
            raw_message_sequence = 11

        with patch(
            "spirecomm.ai.strict_recorded_run_replay._drain_pending_updates",
        ) as drain_mock, patch(
            "spirecomm.ai.strict_recorded_run_replay._wait_for_expected_post_state",
            return_value=(True, True, expected_state),
        ) as wait_post_mock:
            matched, saw_fresh, actual = _stabilize_pre_state_after_bootstrap(
                _Coordinator(),
                object(),
                step=step,
                timeout_seconds=5.0,
            )

        self.assertTrue(matched)
        self.assertTrue(saw_fresh)
        self.assertEqual(actual, expected_state)
        self.assertTrue(drain_mock.called)
        self.assertEqual(wait_post_mock.call_args.kwargs["expected_state"], expected_state)
        self.assertTrue(wait_post_mock.call_args.kwargs["allow_state_probe"])
        self.assertFalse(wait_post_mock.call_args.kwargs["allow_wait_probe"])
        self.assertTrue(wait_post_mock.call_args.kwargs["require_ready"])

    def test_wait_for_expected_post_state_matches_intermediate_state_tap_frame(self):
        expected_state = {
            "phase": "NEOW",
            "room_type": "NeowRoom",
            "screen_type": "EVENT",
            "screen_state": {
                "event_id": None,
                "event_name": None,
                "card_select_context": None,
                "choices": [
                    {"choice_index": 0, "kind": "neow", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
                    {"choice_index": 1, "kind": "neow", "bonus": "THREE_SMALL_POTIONS", "drawback": "NONE"},
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
            "combat_state": None,
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
        }
        stale_talk_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["talk"],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [{"choice_index": 0, "label": "Talk", "text": "[Talk]"}],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        expected_live_state = {
            **stale_talk_state,
            "choice_list": [
                "choose a colorless card to obtain",
                "obtain 3 random potions",
            ],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {"choice_index": 0, "label": "Choose a colorless Card to obtain", "text": "[ Choose a colorless Card to obtain ]"},
                    {"choice_index": 1, "label": "Obtain 3 random Potions", "text": "[ Obtain 3 random Potions ]"},
                ],
            },
        }

        class _TapSequence:
            def __init__(self):
                self._done = False

            def ingest_updates(self, coordinator):
                if self._done:
                    return []
                self._done = True
                events = []
                for source, communication_state in [
                    ("fresh_reward", {"ready_for_command": True, "in_game": True, "available_commands": ["choose", "state"], "game_state": expected_live_state}),
                    ("stale_talk", {"ready_for_command": True, "in_game": True, "available_commands": ["choose", "state"], "game_state": stale_talk_state}),
                ]:
                    coordinator.ingest_communication_state(communication_state, raw_message="{}", source=source)
                    events.append(
                        {
                            "sequence": coordinator.raw_message_sequence,
                            "raw_message": "{}",
                            "communication_state": communication_state,
                            "source": source,
                        }
                    )
                return events

        _FakeCoordinator.raw_states = [
            (
                {
                    "phase": "EVENT",
                    "screen_type": "EVENT",
                    "room_type": "NeowRoom",
                    "floor": 0,
                    "act": 1,
                    "current_hp": 80,
                    "max_hp": 80,
                    "gold": 99,
                    "choice_list": ["talk"],
                    "screen_state": {"event_id": "Neow Event", "event_name": "Neow", "options": [{"choice_index": 0, "label": "Talk", "text": "[Talk]"}]},
                    "deck": [],
                    "relics": [],
                    "potions": [],
                },
                True,
            )
        ]
        coordinator = _FakeCoordinator()

        matched, saw_fresh, actual = _wait_for_expected_post_state(
            coordinator,
            _TapSequence(),
            after_sequence=coordinator.raw_message_sequence,
            expected_state=expected_state,
            phase_hint="NEOW",
            timeout_seconds=0.5,
            context="unit test matching intermediate state tap frame",
            require_ready=True,
        )

        self.assertTrue(matched)
        self.assertTrue(saw_fresh)
        self.assertEqual(actual["screen_state"]["choices"], expected_state["screen_state"]["choices"])
        self.assertEqual(
            normalize_verbose_state_for_strict(coordinator.last_raw_game_state, phase_hint="NEOW"),
            expected_state,
        )

    def test_stabilize_pre_state_accepts_fresh_match_arriving_during_drain(self):
        live_talk_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["talk"],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [{"choice_index": 0, "label": "Talk", "text": "[Talk]"}],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        visible_neow_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": [
                {
                    "kind": "neow",
                    "choice_index": 0,
                    "label": "OPTION_0",
                    "text": "Choose 1 of 3 random colorless cards.",
                    "bonus": "RANDOM_COLORLESS",
                    "drawback": "NONE",
                }
            ],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {
                        "choice_index": 0,
                        "label": "OPTION_0",
                        "text": "Choose 1 of 3 random colorless cards.",
                        "bonus": "RANDOM_COLORLESS",
                        "drawback": "NONE",
                    }
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        expected_pre = normalize_verbose_state_for_strict(visible_neow_state, phase_hint="NEOW")
        step = StrictRecordedTraceStep(
            step=0,
            phase="NEOW",
            floor=0,
            action={},
            strict_action={},
            strict_pre_state=expected_pre,
            strict_post_state={},
            post_phase="CARD_REWARD",
        )

        _FakeCoordinator.raw_states = [(live_talk_state, True)]
        coordinator = _FakeCoordinator()

        def _fake_drain(*args, **kwargs):
            del args, kwargs
            coordinator.last_raw_game_state = visible_neow_state
            coordinator.last_communication_state = {
                "ready_for_command": True,
                "in_game": True,
                "error": None,
                "available_commands": ["choose", "state", "wait"],
                "game_state": visible_neow_state,
            }
            coordinator.raw_message_sequence += 2
            coordinator.game_is_ready = True

        with patch("spirecomm.ai.strict_recorded_run_replay._drain_pending_updates", side_effect=_fake_drain):
            with patch(
                "spirecomm.ai.strict_recorded_run_replay._wait_for_expected_post_state",
                side_effect=AssertionError("fresh drain match should avoid extra wait"),
            ):
                matched, saw_fresh, actual = _stabilize_pre_state_after_bootstrap(
                    coordinator,
                    None,
                    step=step,
                    timeout_seconds=1.0,
                )

        self.assertTrue(matched)
        self.assertTrue(saw_fresh)
        self.assertEqual(actual, expected_pre)

    def test_normalize_strict_state_canonicalizes_neow_live_surface(self):
        live_neow_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "event_id": "Neow Event",
            "screen_name": "Neow",
            "choice_list": [
                "choose a colorless card to obtain"
            ],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {
                        "choice_index": 0,
                        "label": "Choose a colorless Card to obtain",
                        "text": "[ Choose a colorless Card to obtain ]",
                    }
                ],
            },
            "deck": [{"id": "Strike_R", "name": "Strike", "type": "ATTACK", "rarity": "BASIC", "cost": 1}],
            "relics": [{"id": "Burning Blood", "name": "Burning Blood", "counter": -1}],
            "potions": [{"id": "Potion Slot", "name": "Potion Slot"}],
        }

        normalized = normalize_verbose_state_for_strict(live_neow_state, phase_hint="NEOW")

        self.assertEqual(normalized["phase"], "NEOW")
        self.assertEqual(normalized["room_type"], "NeowRoom")
        self.assertEqual(normalized["screen_type"], "EVENT")
        self.assertIsNone(normalized["screen_state"]["event_id"])
        self.assertIsNone(normalized["screen_state"]["event_name"])
        self.assertEqual(normalized["deck"][0]["card_id"], "Strike_R")
        self.assertEqual(normalized["deck"][0]["name"], "Strike_R")
        self.assertEqual(
            normalized["screen_state"]["choices"],
            [{"choice_index": 0, "kind": "neow", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"}],
        )
        self.assertEqual(normalized["relics"], [{"relic_id": "Burning Blood", "name": "Burning Blood"}])
        self.assertEqual(normalized["potions"], [])

    def test_normalize_strict_state_preserves_hidden_neow_talk_surface(self):
        hidden_talk_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["talk"],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {
                        "choice_index": 0,
                        "label": "Talk",
                        "text": "[Talk]",
                    }
                ],
            },
        }

        normalized = normalize_verbose_state_for_strict(hidden_talk_state, phase_hint="NEOW")

        self.assertEqual(
            normalized["screen_state"]["choices"],
            [
                {
                    "choice_index": 0,
                    "kind": "neow",
                    "bonus": None,
                    "drawback": "NONE",
                    "label": "Talk",
                    "text": "[Talk]",
                }
            ],
        )

    def test_normalize_strict_state_strips_neow_color_markup_into_semantic_choices(self):
        reward_options_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": [
                "#gchoose #ga #gcolorless #gcard #gto #gobtain",
                "#gobtain #g3 #grandom #gpotions",
                "#rlose #r8 #rmax #rhp #gchoose #ga #grare #gcard #gto #gobtain",
                "#rlose #ryour #rstarting #rrelic #gobtain #ga #grandom #gboss #grelic",
            ],
            "screen_state": {
                "options": [
                    {"choice_index": 0, "label": "#gChoose #ga #gcolorless #gCard #gto #gobtain", "text": "[ #gChoose #ga #gcolorless #gCard #gto #gobtain ]"},
                    {"choice_index": 1, "label": "#gObtain #g3 #grandom #gPotions", "text": "[ #gObtain #g3 #grandom #gPotions ]"},
                    {"choice_index": 2, "label": "#rLose #r8 #rMax #rHP #gChoose #ga #grare #gCard #gto #gobtain", "text": "[ #rLose #r8 #rMax #rHP #gChoose #ga #grare #gCard #gto #gobtain ]"},
                    {"choice_index": 3, "label": "#rLose #ryour #rstarting #rRelic #gObtain #ga #grandom #gboss #gRelic", "text": "[ #rLose #ryour #rstarting #rRelic #gObtain #ga #grandom #gboss #gRelic ]"},
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }

        normalized = normalize_verbose_state_for_strict(reward_options_state, phase_hint="NEOW")

        self.assertEqual(
            normalized["screen_state"]["choices"],
            [
                {"choice_index": 0, "kind": "neow", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
                {"choice_index": 1, "kind": "neow", "bonus": "THREE_SMALL_POTIONS", "drawback": "NONE"},
                {"choice_index": 2, "kind": "neow", "bonus": "THREE_RARE_CARDS", "drawback": "TEN_PERCENT_HP_LOSS"},
                {"choice_index": 3, "kind": "neow", "bonus": "BOSS_RELIC", "drawback": "NONE"},
            ],
        )

    def test_normalize_strict_state_parses_neow_take_damage_drawback(self):
        reward_options_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "screen_state": {
                "options": [
                    {"choice_index": 2, "label": "Take 24 damage Gain 250 Gold", "text": "[ Take 24 damage Gain 250 Gold ]"},
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }

        normalized = normalize_verbose_state_for_strict(reward_options_state, phase_hint="NEOW")

        self.assertEqual(
            normalized["screen_state"]["choices"],
            [{"choice_index": 2, "kind": "neow", "bonus": "TWO_FIFTY_GOLD", "drawback": "PERCENT_DAMAGE"}],
        )

    def test_normalize_strict_state_parses_neow_three_enemy_kill_wording(self):
        reward_options_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "screen_state": {
                "options": [
                    {
                        "choice_index": 1,
                        "label": "Enemies in your next three combats have 1 HP",
                        "text": "[ Enemies in your next three combats have 1 HP ]",
                    },
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }

        normalized = normalize_verbose_state_for_strict(reward_options_state, phase_hint="NEOW")

        self.assertEqual(
            normalized["screen_state"]["choices"],
            [{"choice_index": 1, "kind": "neow", "bonus": "THREE_ENEMY_KILL", "drawback": "NONE"}],
        )

    def test_normalize_strict_state_distinguishes_wheel_event_stages(self):
        def wheel_choice(label: str, text: str | None = None):
            state = {
                "phase": "EVENT",
                "screen_type": "EVENT",
                "room_type": "EventRoom",
                "floor": 5,
                "act": 1,
                "current_hp": 71,
                "max_hp": 80,
                "gold": 139,
                "deck": [],
                "relics": [],
                "potions": [],
                "screen_state": {
                    "event_id": "Wheel of Change",
                    "options": [{"choice_index": 0, "label": label, "text": text or label}],
                },
            }
            return normalize_verbose_state_for_strict(state, phase_hint="EVENT")["screen_state"]["choices"][0]

        self.assertEqual(wheel_choice("Play")["event_stage"], "PLAY")
        self.assertEqual(wheel_choice("spin")["event_stage"], "SPIN")
        self.assertEqual(wheel_choice("Prize?", "[Prize?] Curse - Decay.")["event_stage"], "RESULT")
        self.assertEqual(wheel_choice("Leave")["event_stage"], "LEAVE")

    def test_normalize_strict_state_defaults_card_reward_screen_flags(self):
        reward_state = {
            "phase": "CARD_REWARD",
            "screen_type": "CARD_REWARD",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["trip", "panacea", "panic button"],
            "screen_state": {
                "cards": [
                    {"id": "Trip", "name": "Trip", "type": "SKILL", "rarity": "UNCOMMON", "upgrades": 0, "cost": 0, "has_target": True},
                    {"id": "Panacea", "name": "Panacea", "type": "SKILL", "rarity": "UNCOMMON", "upgrades": 0, "cost": 0, "has_target": False, "exhausts": True},
                    {"id": "PanicButton", "name": "Panic Button", "type": "SKILL", "rarity": "UNCOMMON", "upgrades": 0, "cost": 0, "has_target": False, "exhausts": True},
                ]
            },
        }

        normalized = normalize_verbose_state_for_strict(reward_state, phase_hint="CARD_REWARD")

        self.assertTrue(normalized["screen_state"]["skip_available"])
        self.assertFalse(normalized["screen_state"]["bowl_available"])

    def test_normalize_strict_state_canonicalizes_ethereal_exhaust_pile_exhausts_flag(self):
        state = self._minimal_combat_raw_state()
        state["combat_state"]["exhaust_pile"] = [
            {
                "id": "Dazed",
                "name": "Dazed",
                "type": "STATUS",
                "rarity": "COMMON",
                "upgrades": 0,
                "cost": -2,
                "has_target": False,
                "exhausts": False,
                "ethereal": True,
            },
            {
                "id": "Strike_R",
                "name": "Strike_R",
                "type": "ATTACK",
                "rarity": "BASIC",
                "upgrades": 0,
                "cost": 1,
                "has_target": True,
                "exhausts": False,
                "ethereal": False,
            },
        ]

        normalized = normalize_verbose_state_for_strict(state, phase_hint="COMBAT")
        by_id = {card["card_id"]: card for card in normalized["combat_state"]["exhaust_pile"]}

        self.assertTrue(by_id["Dazed"]["exhausts"])
        self.assertFalse(by_id["Strike_R"]["exhausts"])

    def test_load_strict_trace_recanonicalizes_raw_states_when_available(self):
        raw_pre = self._minimal_combat_raw_state()
        raw_post = self._minimal_combat_raw_state(turn=1)
        raw_post["combat_state"]["exhaust_pile"] = [
            {
                "id": "Dazed",
                "name": "Dazed",
                "type": "STATUS",
                "rarity": "COMMON",
                "upgrades": 0,
                "cost": -2,
                "has_target": False,
                "exhausts": False,
                "ethereal": True,
            }
        ]
        stale_strict_post = normalize_verbose_state_for_strict(raw_post, phase_hint="COMBAT")
        stale_strict_post["combat_state"]["exhaust_pile"][0]["exhausts"] = False
        payload = {
            "trace_schema": STRICT_TRACE_SCHEMA,
            "seed_long": 62,
            "seed_str": "1S",
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "COMBAT",
                    "floor": 12,
                    "action": {"kind": "end"},
                    "pre_state": raw_pre,
                    "post_state": raw_post,
                    "pre_snapshot": {"phase": "COMBAT"},
                    "strict_action": {"kind": "end_turn", "phase": "COMBAT", "raw_kind": "end"},
                    "strict_pre_state": normalize_verbose_state_for_strict(raw_pre, phase_hint="COMBAT"),
                    "strict_post_state": stale_strict_post,
                    "post_phase": "COMBAT",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            trace = load_strict_recorded_trace(path)

        exhaust_pile = trace.steps[0].strict_post_state["combat_state"]["exhaust_pile"]
        self.assertTrue(exhaust_pile[0]["exhausts"])

    def test_strict_replay_rejects_legacy_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_trace.json"
            path.write_text(json.dumps({"steps": []}), encoding="utf-8")

            report = replay_recorded_run_strict(trace_path=path)

        self.assertFalse(report["success"])
        self.assertEqual(report["failure_kind"], "unsupported_legacy_trace")

    def test_strict_replay_fails_on_pre_state_mismatch(self):
        pre_state = {
            "phase": "CARD_REWARD",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "room_type": "NeowRoom",
            "choice_list": [
                {"kind": "card_reward", "choice_index": 0, "name": "Trip", "card_id": "Trip"},
                {"kind": "card_reward", "choice_index": 1, "name": "Panacea", "card_id": "Panacea"},
                {"kind": "card_reward", "choice_index": 2, "name": "Panic Button", "card_id": "PanicButton"},
            ],
            "deck": [],
            "relics": [],
            "potions": [],
        }
        post_state = {
            **pre_state,
            "phase": "NEOW",
            "choice_list": [{"kind": "neow", "choice_index": 0, "name": "OPTION_0", "label": "OPTION_0"}],
            "deck": [{"card_id": "Trip", "name": "Trip", "upgrades": 0, "type": "SKILL", "rarity": "UNCOMMON", "cost": 1}],
        }
        strict_step = build_strict_step_payload(
            action={"kind": "card_reward", "choice_index": 0, "name": "Trip", "card_id": "Trip"},
            pre_state=pre_state,
            post_state=post_state,
            phase="CARD_REWARD",
            post_phase="NEOW",
        )
        bad_live_pre_state = {
            **pre_state,
            "screen_type": "CARD_REWARD",
            "screen_state": {
                "cards": [
                    {"id": "Madness", "name": "Madness", "type": "SKILL", "rarity": "UNCOMMON", "upgrades": 0, "cost": 1, "has_target": False},
                    {"id": "Good Instincts", "name": "Good Instincts", "type": "SKILL", "rarity": "UNCOMMON", "upgrades": 0, "cost": 0, "has_target": False},
                    {"id": "Swift Strike", "name": "Swift Strike", "type": "ATTACK", "rarity": "UNCOMMON", "upgrades": 0, "cost": 0, "has_target": True},
                ]
            },
            "choice_list": ["madness", "good instincts", "swift strike"],
        }
        payload = {
            "trace_schema": STRICT_TRACE_SCHEMA,
            "seed_long": 1,
            "seed_str": "1",
            "ascension": 0,
            "steps": [
                {
                    "step": 1,
                    "phase": "CARD_REWARD",
                    "floor": 0,
                    "action": {"kind": "card_reward", "choice_index": 0, "name": "Trip", "card_id": "Trip"},
                    "strict_action": strict_step.strict_action,
                    "strict_pre_state": strict_step.strict_pre_state,
                    "strict_post_state": strict_step.strict_post_state,
                    "post_phase": "NEOW",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "strict_trace.json"
            trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _FakeCoordinator.raw_states = [bad_live_pre_state]
            with patch("spirecomm.ai.strict_recorded_run_replay.Coordinator", _FakeCoordinator):
                with patch("spirecomm.ai.strict_recorded_run_replay._start_recorded_game", return_value=None):
                    report = replay_recorded_run_strict(trace_path=trace_path)

        self.assertFalse(report["success"])
        self.assertEqual(report["failure_kind"], "pre_state_mismatch")
        actual_choices = report["actual_visible"]["screen_state"]["choices"]
        self.assertEqual([item["card"]["card_id"] for item in actual_choices], ["Madness", "Good Instincts", "Swift Strike"])

    def test_strict_replay_bootstraps_neow_talk_before_first_real_neow_choice(self):
        step_payload = {
            "trace_schema": STRICT_TRACE_SCHEMA,
            "seed_long": 1,
            "seed_str": "1",
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "NEOW",
                    "floor": 0,
                    "action": {"kind": "neow", "choice_index": 0, "name": "OPTION_0", "label": "OPTION_0"},
                    "strict_action": {
                        "phase": "NEOW",
                        "kind": "choose_by_index",
                        "raw_kind": "neow",
                        "choice_index": 0,
                        "choice": {
                            "choice_index": 0,
                            "kind": "neow",
                            "label": "OPTION_0",
                            "text": "Choose 1 of 3 random colorless cards.",
                            "event_id": None,
                            "bonus": "RANDOM_COLORLESS",
                            "drawback": "NONE",
                        },
                    },
                    "strict_pre_state": None,
                    "strict_post_state": None,
                    "post_phase": "CARD_REWARD",
                }
            ],
        }
        live_talk_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["talk"],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [{"choice_index": 0, "label": "Talk", "text": "[Talk]"}],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_options_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": [
                {
                    "kind": "neow",
                    "choice_index": 0,
                    "label": "OPTION_0",
                    "text": "Choose 1 of 3 random colorless cards.",
                    "bonus": "RANDOM_COLORLESS",
                    "drawback": "NONE",
                },
                {
                    "kind": "neow",
                    "choice_index": 1,
                    "label": "OPTION_1",
                    "text": "Obtain 3 random potions.",
                    "bonus": "THREE_SMALL_POTIONS",
                    "drawback": "NONE",
                },
            ],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {
                        "choice_index": 0,
                        "label": "OPTION_0",
                        "text": "Choose 1 of 3 random colorless cards.",
                        "bonus": "RANDOM_COLORLESS",
                        "drawback": "NONE",
                    },
                    {
                        "choice_index": 1,
                        "label": "OPTION_1",
                        "text": "Obtain 3 random potions.",
                        "bonus": "THREE_SMALL_POTIONS",
                        "drawback": "NONE",
                    },
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_talk_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["talk"],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {
                        "choice_index": 0,
                        "label": "Talk",
                        "text": "[Talk]",
                    }
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_talk_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["talk"],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {
                        "choice_index": 0,
                        "label": "Talk",
                        "text": "[Talk]",
                    }
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_talk_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["talk"],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {
                        "choice_index": 0,
                        "label": "Talk",
                        "text": "[Talk]",
                    }
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_talk_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["talk"],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {
                        "choice_index": 0,
                        "label": "Talk",
                        "text": "[Talk]",
                    }
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_talk_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["talk"],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {
                        "choice_index": 0,
                        "label": "Talk",
                        "text": "[Talk]",
                    }
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_talk_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["talk"],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {
                        "choice_index": 0,
                        "label": "Talk",
                        "text": "[Talk]",
                    }
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_reward_state = {
            "phase": "CARD_REWARD",
            "screen_type": "CARD_REWARD",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["trip"],
            "screen_state": {
                "cards": [
                    {
                        "id": "Trip",
                        "name": "Trip",
                        "type": "SKILL",
                        "rarity": "UNCOMMON",
                        "upgrades": 0,
                        "cost": 0,
                        "has_target": True,
                        "exhausts": False,
                        "is_playable": False,
                    }
                ],
                "skip_available": True,
                "bowl_available": False,
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        expected_pre = normalize_verbose_state_for_strict(live_options_state, phase_hint="NEOW")
        expected_post = normalize_verbose_state_for_strict(live_reward_state, phase_hint="CARD_REWARD")
        step_payload["steps"][0]["strict_pre_state"] = expected_pre
        step_payload["steps"][0]["strict_post_state"] = expected_post

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "strict_trace.json"
            trace_path.write_text(json.dumps(step_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _FakeCoordinator.raw_states = [live_talk_state, live_options_state, live_options_state, live_reward_state]
            with patch("spirecomm.ai.strict_recorded_run_replay.Coordinator", _FakeCoordinator):
                with patch("spirecomm.ai.strict_recorded_run_replay._start_recorded_game", return_value=None):
                    with patch(
                        "spirecomm.ai.strict_recorded_run_replay._stabilize_pre_state_after_bootstrap",
                        return_value=(True, True, expected_pre),
                    ):
                        report = replay_recorded_run_strict(trace_path=trace_path)

        self.assertTrue(report["success"])
        self.assertEqual(report["steps_replayed"], 1)
        self.assertEqual(report["results"][0]["pre_sync_commands"], ["choose 0"])
        self.assertEqual(report["results"][0]["commands"], ["choose 0"])

    def test_strict_replay_requires_fresh_bootstrap_stabilization_even_after_restored_ready_pre_state(self):
        step_payload = {
            "trace_schema": STRICT_TRACE_SCHEMA,
            "seed_long": 1,
            "seed_str": "1",
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "NEOW",
                    "floor": 0,
                    "action": {"kind": "neow", "choice_index": 0, "name": "OPTION_0", "label": "OPTION_0"},
                    "strict_action": {
                        "phase": "NEOW",
                        "kind": "choose_by_index",
                        "raw_kind": "neow",
                        "choice_index": 0,
                        "choice": {
                            "choice_index": 0,
                            "kind": "neow",
                            "label": "OPTION_0",
                            "text": "Choose 1 of 3 random colorless cards.",
                            "event_id": None,
                            "bonus": "RANDOM_COLORLESS",
                            "drawback": "NONE",
                        },
                    },
                    "strict_pre_state": None,
                    "strict_post_state": None,
                    "post_phase": "CARD_REWARD",
                }
            ],
        }
        live_talk_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["talk"],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [{"choice_index": 0, "label": "Talk", "text": "[Talk]"}],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_options_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": [
                {
                    "kind": "neow",
                    "choice_index": 0,
                    "label": "OPTION_0",
                    "text": "Choose 1 of 3 random colorless cards.",
                    "bonus": "RANDOM_COLORLESS",
                    "drawback": "NONE",
                }
            ],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {
                        "choice_index": 0,
                        "label": "OPTION_0",
                        "text": "Choose 1 of 3 random colorless cards.",
                        "bonus": "RANDOM_COLORLESS",
                        "drawback": "NONE",
                    }
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_reward_state = {
            "phase": "CARD_REWARD",
            "screen_type": "CARD_REWARD",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["trip"],
            "screen_state": {
                "cards": [
                    {
                        "id": "Trip",
                        "name": "Trip",
                        "type": "SKILL",
                        "rarity": "UNCOMMON",
                        "upgrades": 0,
                        "cost": 0,
                        "has_target": True,
                        "exhausts": False,
                        "is_playable": False,
                    }
                ],
                "skip_available": True,
                "bowl_available": False,
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        step_payload["steps"][0]["strict_pre_state"] = normalize_verbose_state_for_strict(live_options_state, phase_hint="NEOW")
        step_payload["steps"][0]["strict_post_state"] = normalize_verbose_state_for_strict(live_reward_state, phase_hint="CARD_REWARD")

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "strict_trace.json"
            trace_path.write_text(json.dumps(step_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _FakeCoordinator.raw_states = [live_talk_state, live_options_state, live_reward_state]
            with patch("spirecomm.ai.strict_recorded_run_replay.Coordinator", _FakeCoordinator):
                with patch("spirecomm.ai.strict_recorded_run_replay._start_recorded_game", return_value=None):
                    with patch(
                        "spirecomm.ai.strict_recorded_run_replay._stabilize_pre_state_after_bootstrap",
                        return_value=(True, True, step_payload["steps"][0]["strict_pre_state"]),
                    ) as stabilize_mock:
                        report = replay_recorded_run_strict(trace_path=trace_path)

        self.assertTrue(report["success"])
        self.assertEqual(report["steps_replayed"], 1)
        self.assertEqual(report["results"][0]["pre_sync_commands"], ["choose 0"])
        self.assertEqual(report["results"][0]["commands"], ["choose 0"])
        stabilize_mock.assert_called_once()

    def test_strict_replay_bootstraps_empty_neow_intro_before_first_real_neow_choice(self):
        step_payload = {
            "trace_schema": STRICT_TRACE_SCHEMA,
            "seed_long": 1,
            "seed_str": "1",
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "NEOW",
                    "floor": 0,
                    "action": {"kind": "neow", "choice_index": 0, "name": "OPTION_0", "label": "OPTION_0"},
                    "strict_action": {
                        "phase": "NEOW",
                        "kind": "choose_by_index",
                        "raw_kind": "neow",
                        "choice_index": 0,
                        "choice": {
                            "choice_index": 0,
                            "kind": "neow",
                            "label": "OPTION_0",
                            "text": "Choose 1 of 3 random colorless cards.",
                            "event_id": None,
                            "bonus": "RANDOM_COLORLESS",
                            "drawback": "NONE",
                        },
                    },
                    "strict_pre_state": None,
                    "strict_post_state": None,
                    "post_phase": "CARD_REWARD",
                }
            ],
        }
        live_empty_intro_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": [],
            "screen_name": "NONE",
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_options_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": [
                {
                    "kind": "neow",
                    "choice_index": 0,
                    "label": "OPTION_0",
                    "text": "Choose 1 of 3 random colorless cards.",
                    "bonus": "RANDOM_COLORLESS",
                    "drawback": "NONE",
                }
            ],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {
                        "choice_index": 0,
                        "label": "OPTION_0",
                        "text": "Choose 1 of 3 random colorless cards.",
                        "bonus": "RANDOM_COLORLESS",
                        "drawback": "NONE",
                    }
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_reward_state = {
            "phase": "CARD_REWARD",
            "screen_type": "CARD_REWARD",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["trip"],
            "screen_state": {
                "cards": [
                    {
                        "id": "Trip",
                        "name": "Trip",
                        "type": "SKILL",
                        "rarity": "UNCOMMON",
                        "upgrades": 0,
                        "cost": 0,
                        "has_target": True,
                        "exhausts": False,
                    }
                ],
                "skip_available": True,
                "bowl_available": False,
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        step_payload["steps"][0]["strict_pre_state"] = normalize_verbose_state_for_strict(live_options_state, phase_hint="NEOW")
        step_payload["steps"][0]["strict_post_state"] = normalize_verbose_state_for_strict(live_reward_state, phase_hint="CARD_REWARD")
        live_talk_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["talk"],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {
                        "choice_index": 0,
                        "label": "Talk",
                        "text": "[Talk]",
                    }
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "strict_trace.json"
            trace_path.write_text(json.dumps(step_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _FakeCoordinator.raw_states = [
                live_empty_intro_state,
                live_empty_intro_state,
                live_talk_state,
                live_options_state,
                live_options_state,
                live_reward_state,
            ]
            with patch("spirecomm.ai.strict_recorded_run_replay.Coordinator", _FakeCoordinator):
                with patch("spirecomm.ai.strict_recorded_run_replay._start_recorded_game", return_value=None):
                    with patch(
                        "spirecomm.ai.strict_recorded_run_replay._stabilize_pre_state_after_bootstrap",
                        return_value=(True, True, step_payload["steps"][0]["strict_pre_state"]),
                    ):
                        report = replay_recorded_run_strict(trace_path=trace_path)

        self.assertTrue(report["success"])
        self.assertEqual(report["steps_replayed"], 1)
        self.assertEqual(report["results"][0]["pre_sync_commands"], ["choose 0"])
        self.assertEqual(report["results"][0]["commands"], ["choose 0"])

    def test_strict_replay_waits_through_transient_post_state_until_expected_visible_state(self):
        pre_state = {
            "phase": "NEOW",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": [
                {
                    "kind": "neow",
                    "choice_index": 0,
                    "label": "OPTION_0",
                    "text": "Choose 1 of 3 random colorless cards.",
                    "bonus": "RANDOM_COLORLESS",
                    "drawback": "NONE",
                }
            ],
            "screen_state": {
                "options": [
                    {
                        "choice_index": 0,
                        "label": "OPTION_0",
                        "text": "Choose 1 of 3 random colorless cards.",
                        "bonus": "RANDOM_COLORLESS",
                        "drawback": "NONE",
                    }
                ]
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        transient_post_state = {
            "phase": "CARD_REWARD",
            "screen_type": "CARD_REWARD",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": [
                "choose a colorless card to obtain",
                "obtain 3 random potions",
            ],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        final_post_state = {
            "phase": "CARD_REWARD",
            "screen_type": "CARD_REWARD",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["trip", "panacea", "panic button"],
            "screen_state": {
                "cards": [
                    {"id": "Trip", "name": "Trip", "type": "SKILL", "rarity": "UNCOMMON", "upgrades": 0, "cost": 0, "has_target": True, "exhausts": False},
                    {"id": "Panacea", "name": "Panacea", "type": "SKILL", "rarity": "UNCOMMON", "upgrades": 0, "cost": 0, "has_target": False, "exhausts": True},
                    {"id": "PanicButton", "name": "Panic Button", "type": "SKILL", "rarity": "UNCOMMON", "upgrades": 0, "cost": 0, "has_target": False, "exhausts": True},
                ]
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        strict_step = build_strict_step_payload(
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0", "label": "OPTION_0", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
            pre_state=pre_state,
            post_state=final_post_state,
            phase="NEOW",
            post_phase="CARD_REWARD",
        )
        payload = {
            "trace_schema": STRICT_TRACE_SCHEMA,
            "seed_long": 1,
            "seed_str": "1",
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "NEOW",
                    "floor": 0,
                    "action": {"kind": "neow", "choice_index": 0, "name": "OPTION_0", "label": "OPTION_0", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
                    "strict_action": strict_step.strict_action,
                    "strict_pre_state": strict_step.strict_pre_state,
                    "strict_post_state": strict_step.strict_post_state,
                    "post_phase": "CARD_REWARD",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "strict_trace.json"
            trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _FakeCoordinator.raw_states = [pre_state, transient_post_state, final_post_state]
            with patch("spirecomm.ai.strict_recorded_run_replay.Coordinator", _FakeCoordinator):
                with patch("spirecomm.ai.strict_recorded_run_replay._start_recorded_game", return_value=None):
                    report = replay_recorded_run_strict(trace_path=trace_path)

        self.assertTrue(report["success"])
        self.assertEqual(report["steps_replayed"], 1)
        self.assertEqual(report["results"][0]["commands"], ["choose 0"])

    def test_strict_replay_waits_for_ready_after_neow_bootstrap_before_real_action(self):
        step_payload = {
            "trace_schema": STRICT_TRACE_SCHEMA,
            "seed_long": 1,
            "seed_str": "1",
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "NEOW",
                    "floor": 0,
                    "action": {"kind": "neow", "choice_index": 0, "name": "OPTION_0", "label": "OPTION_0"},
                    "strict_action": {
                        "phase": "NEOW",
                        "kind": "choose_by_index",
                        "raw_kind": "neow",
                        "choice_index": 0,
                        "choice": {
                            "choice_index": 0,
                            "kind": "neow",
                            "label": "OPTION_0",
                            "text": "Choose 1 of 3 random colorless cards.",
                            "bonus": "RANDOM_COLORLESS",
                            "drawback": "NONE",
                        },
                    },
                    "strict_pre_state": None,
                    "strict_post_state": None,
                    "post_phase": "CARD_REWARD",
                }
            ],
        }
        live_talk_state = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["talk"],
            "screen_state": {"event_id": "Neow Event", "event_name": "Neow", "options": [{"choice_index": 0, "label": "Talk", "text": "[Talk]"}]},
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_options_not_ready = {
            "phase": "EVENT",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": [
                {"kind": "neow", "choice_index": 0, "label": "OPTION_0", "text": "Choose 1 of 3 random colorless cards.", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
                {"kind": "neow", "choice_index": 1, "label": "OPTION_1", "text": "Obtain 3 random potions.", "bonus": "THREE_SMALL_POTIONS", "drawback": "NONE"},
            ],
            "screen_state": {
                "event_id": "Neow Event",
                "event_name": "Neow",
                "options": [
                    {"choice_index": 0, "label": "OPTION_0", "text": "Choose 1 of 3 random colorless cards.", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
                    {"choice_index": 1, "label": "OPTION_1", "text": "Obtain 3 random potions.", "bonus": "THREE_SMALL_POTIONS", "drawback": "NONE"},
                ],
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        live_options_ready = dict(live_options_not_ready)
        live_reward_state = {
            "phase": "CARD_REWARD",
            "screen_type": "CARD_REWARD",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["trip"],
            "screen_state": {
                "cards": [{"id": "Trip", "name": "Trip", "type": "SKILL", "rarity": "UNCOMMON", "upgrades": 0, "cost": 0, "has_target": True, "exhausts": False}],
                "skip_available": True,
                "bowl_available": False,
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        step_payload["steps"][0]["strict_pre_state"] = normalize_verbose_state_for_strict(live_options_ready, phase_hint="NEOW")
        step_payload["steps"][0]["strict_post_state"] = normalize_verbose_state_for_strict(live_reward_state, phase_hint="CARD_REWARD")

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "strict_trace.json"
            trace_path.write_text(json.dumps(step_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _FakeCoordinator.raw_states = [
                (live_talk_state, True),
                (live_options_not_ready, False),
                (live_options_ready, True),
                (live_options_ready, True),
                (live_reward_state, True),
            ]
            with patch("spirecomm.ai.strict_recorded_run_replay.Coordinator", _FakeCoordinator):
                with patch("spirecomm.ai.strict_recorded_run_replay._start_recorded_game", return_value=None):
                    with patch(
                        "spirecomm.ai.strict_recorded_run_replay._stabilize_pre_state_after_bootstrap",
                        return_value=(True, True, step_payload["steps"][0]["strict_pre_state"]),
                    ):
                        report = replay_recorded_run_strict(trace_path=trace_path)

        self.assertTrue(report["success"])
        self.assertEqual(report["steps_replayed"], 1)
        self.assertEqual(report["results"][0]["pre_sync_commands"], ["choose 0"])
        self.assertEqual(report["results"][0]["commands"], ["choose 0"])

    def test_strict_replay_requires_fresh_pre_state_after_restored_tap_frame_when_current_state_differs(self):
        pre_state = {
            "phase": "NEOW",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": [
                {
                    "kind": "neow",
                    "choice_index": 0,
                    "label": "OPTION_0",
                    "text": "Choose 1 of 3 random colorless cards.",
                    "bonus": "RANDOM_COLORLESS",
                    "drawback": "NONE",
                }
            ],
            "screen_state": {
                "options": [
                    {
                        "choice_index": 0,
                        "label": "OPTION_0",
                        "text": "Choose 1 of 3 random colorless cards.",
                        "bonus": "RANDOM_COLORLESS",
                        "drawback": "NONE",
                    }
                ]
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        post_state = {
            "phase": "CARD_REWARD",
            "screen_type": "CARD_REWARD",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["trip"],
            "screen_state": {
                "cards": [
                    {
                        "id": "Trip",
                        "name": "Trip",
                        "type": "SKILL",
                        "rarity": "UNCOMMON",
                        "upgrades": 0,
                        "cost": 0,
                        "has_target": True,
                        "exhausts": False,
                    }
                ],
                "skip_available": True,
                "bowl_available": False,
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        strict_step = build_strict_step_payload(
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0", "label": "OPTION_0", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
            pre_state=pre_state,
            post_state=post_state,
            phase="NEOW",
            post_phase="CARD_REWARD",
        )
        payload = {
            "trace_schema": STRICT_TRACE_SCHEMA,
            "seed_long": 1,
            "seed_str": "1",
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "NEOW",
                    "floor": 0,
                    "action": {"kind": "neow", "choice_index": 0, "name": "OPTION_0", "label": "OPTION_0", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
                    "strict_action": strict_step.strict_action,
                    "strict_pre_state": strict_step.strict_pre_state,
                    "strict_post_state": strict_step.strict_post_state,
                    "post_phase": "CARD_REWARD",
                }
            ],
        }

        class _PendingCoordinator(_FakeCoordinator):
            last_instance = None

            def __init__(self):
                super().__init__()
                self.strict_restored_state_pending = True
                type(self).last_instance = self

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "strict_trace.json"
            trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _PendingCoordinator.raw_states = [post_state]
            with patch("spirecomm.ai.strict_recorded_run_replay.Coordinator", _PendingCoordinator):
                with patch("spirecomm.ai.strict_recorded_run_replay._start_recorded_game", return_value=None):
                    with patch(
                        "spirecomm.ai.strict_recorded_run_replay._wait_for_expected_post_state",
                        return_value=(False, False, None),
                    ) as wait_mock:
                        report = replay_recorded_run_strict(trace_path=trace_path)

        self.assertFalse(report["success"])
        self.assertEqual(report["failure_kind"], "missing_fresh_live_state")
        self.assertEqual(report["steps_replayed"], 0)
        self.assertEqual(_PendingCoordinator.last_instance.commands, [])
        self.assertIn(
            "fresh pre-state after restored tap frame",
            wait_mock.call_args.kwargs["context"],
        )

    def test_strict_replay_reuses_current_ready_pre_state_after_restored_frame_when_sequence_advanced(self):
        pre_state = {
            "phase": "NEOW",
            "screen_type": "EVENT",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": [
                {
                    "kind": "neow",
                    "choice_index": 0,
                    "label": "OPTION_0",
                    "text": "Choose 1 of 3 random colorless cards.",
                    "bonus": "RANDOM_COLORLESS",
                    "drawback": "NONE",
                }
            ],
            "screen_state": {
                "options": [
                    {
                        "choice_index": 0,
                        "label": "OPTION_0",
                        "text": "Choose 1 of 3 random colorless cards.",
                        "bonus": "RANDOM_COLORLESS",
                        "drawback": "NONE",
                    }
                ]
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        post_state = {
            "phase": "CARD_REWARD",
            "screen_type": "CARD_REWARD",
            "room_type": "NeowRoom",
            "floor": 0,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "choice_list": ["trip"],
            "screen_state": {
                "cards": [
                    {
                        "id": "Trip",
                        "name": "Trip",
                        "type": "SKILL",
                        "rarity": "UNCOMMON",
                        "upgrades": 0,
                        "cost": 0,
                        "has_target": True,
                        "exhausts": False,
                    }
                ],
                "skip_available": True,
                "bowl_available": False,
            },
            "deck": [],
            "relics": [],
            "potions": [],
        }
        strict_step = build_strict_step_payload(
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0", "label": "OPTION_0", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
            pre_state=pre_state,
            post_state=post_state,
            phase="NEOW",
            post_phase="CARD_REWARD",
        )
        payload = {
            "trace_schema": STRICT_TRACE_SCHEMA,
            "seed_long": 1,
            "seed_str": "1",
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "NEOW",
                    "floor": 0,
                    "action": {"kind": "neow", "choice_index": 0, "name": "OPTION_0", "label": "OPTION_0", "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
                    "strict_action": strict_step.strict_action,
                    "strict_pre_state": strict_step.strict_pre_state,
                    "strict_post_state": strict_step.strict_post_state,
                    "post_phase": "CARD_REWARD",
                }
            ],
        }

        class _PendingCoordinator(_FakeCoordinator):
            last_instance = None

            def __init__(self):
                super().__init__()
                self.strict_restored_state_pending = True
                self.strict_last_action_sequence = 0
                type(self).last_instance = self

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "strict_trace.json"
            trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _PendingCoordinator.raw_states = [pre_state]
            with patch("spirecomm.ai.strict_recorded_run_replay.Coordinator", _PendingCoordinator):
                with patch("spirecomm.ai.strict_recorded_run_replay._start_recorded_game", return_value=None):
                    with patch(
                        "spirecomm.ai.strict_recorded_run_replay._wait_for_expected_post_state",
                        side_effect=[
                            (True, True, strict_step.strict_post_state),
                        ],
                    ) as wait_mock:
                        report = replay_recorded_run_strict(trace_path=trace_path)

        self.assertTrue(report["success"])
        self.assertEqual(report["steps_replayed"], 1)
        self.assertEqual(_PendingCoordinator.last_instance.commands, ["choose 0"])
        self.assertFalse(getattr(_PendingCoordinator.last_instance, "strict_restored_state_pending", False))
        self.assertEqual(len(wait_mock.call_args_list), 1)
        self.assertIn("post-state after step 0", wait_mock.call_args_list[0].kwargs["context"])


if __name__ == "__main__":
    unittest.main()
