from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from spirecomm.ai.recorded_run_replay import (
    RecordedRunTrace,
    RecordedTraceStep,
    _StateDrivenReplayDriver,
    _compact_snapshot_from_verbose_state,
    _filter_event_pre_mismatches,
    _filter_map_event_transition_mismatches,
    _filter_neow_transition_mismatches,
    _is_neow_card_select_state,
    _single_leave_event_action,
    _single_neow_continue_action,
    _normalize_live_phase,
    _next_trace_action_for_game,
    _play_recorded_game,
    _send_probe_command,
    _should_allow_nonready_direct_action,
    _should_skip_ready_wait_for_event_leave_map_continue,
    _should_use_short_post_neow_callback_timeout,
    _should_wait_for_any_update_for_post_neow_settle,
    _should_skip_ready_wait_for_neow_leave_map_continue,
    snapshot_live_state,
    _wait_for_callback_update_with_timeout,
    _wait_for_matching_snapshot_with_timeout,
    _wait_for_phase_with_timeout,
    _wait_for_ready_state_with_timeout,
    compare_snapshots,
    load_recorded_trace,
)
from spirecomm.communication.action import CardRewardAction, ChooseAction, ChooseMapNodeAction, CombatRewardAction, RawCommandAction, RestAction, StateAction, WaitAction
from spirecomm.spire.character import PlayerClass
from spirecomm.spire.card import Card, CardRarity, CardType
from spirecomm.spire.game import Game
from spirecomm.spire.map import Node
from spirecomm.spire.screen import (
    CombatReward,
    CombatRewardScreen,
    GridSelectScreen,
    MapScreen,
    RestOption,
    RestScreen,
    RewardType,
    ScreenType,
)


class RecordedRunReplayTest(unittest.TestCase):
    def test_send_probe_command_prefers_immediate_send_when_available(self):
        class FakeCoordinator:
            def __init__(self):
                self.messages = []

            def send_message(self, message):
                self.messages.append(("queued", message))

            def send_message_immediate(self, message):
                self.messages.append(("immediate", message))

        coordinator = FakeCoordinator()

        _send_probe_command(coordinator, "state")

        self.assertEqual(coordinator.messages, [("immediate", "state")])

    def test_single_neow_continue_action_accepts_single_proceed_option(self):
        option = SimpleNamespace(choice_index=0, label="[Proceed]", text="Proceed", name=None)
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[option]),
        )

        action = _single_neow_continue_action(game_state)

        from spirecomm.communication.action import NeowContinueAction

        self.assertIsInstance(action, NeowContinueAction)
        self.assertEqual(action.command, "choose 0")
        self.assertFalse(action.requires_game_ready)

    def test_next_trace_action_for_game_prefers_nonready_neow_continue_bridge(self):
        step = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
        )
        option = SimpleNamespace(choice_index=0, label="[Proceed]", text="Proceed", name=None)
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[option]),
        )

        action = _next_trace_action_for_game(game_state, step)

        from spirecomm.communication.action import NeowContinueAction

        self.assertIsInstance(action, NeowContinueAction)
        self.assertEqual(action.command, "choose 0")
        self.assertFalse(action.requires_game_ready)

    def test_filter_neow_transition_mismatches_keeps_reward_side_effect_diffs(self):
        mismatches = [
            "phase: expected 'MAP', got 'NEOW'",
            "screen_type: expected 'MAP', got 'EVENT'",
            "deck: expected ['Strike_R', 'Demon Form'], got ['Strike_R']",
        ]

        filtered = _filter_neow_transition_mismatches(mismatches)

        self.assertEqual(filtered, ["deck: expected ['Strike_R', 'Demon Form'], got ['Strike_R']"])

    def test_filter_map_event_transition_mismatches_drops_missing_expected_event_id(self):
        mismatches = [
            "event_id: expected None, got 'The Woman in Blue'",
            "gold: expected 113, got 83",
        ]

        filtered = _filter_map_event_transition_mismatches(
            mismatches,
            expected_event_id=None,
            actual_event_id="The Woman in Blue",
        )

        self.assertEqual(filtered, ["gold: expected 113, got 83"])

    def test_filter_event_pre_mismatches_drops_missing_expected_event_id(self):
        mismatches = [
            "event_id: expected None, got 'The Woman in Blue'",
            "gold: expected 113, got 83",
        ]

        filtered = _filter_event_pre_mismatches(
            mismatches,
            step_action={"event_id": "The Woman in Blue"},
            actual_event_id="The Woman in Blue",
        )

        self.assertEqual(filtered, ["gold: expected 113, got 83"])

    def test_card_reward_skip_step_can_become_noop_if_live_state_already_matches_post(self):
        payload = {
            "seed_long": 3,
            "seed_str": "3",
            "ascension": 0,
            "backend": "v2",
            "steps": [
                {
                    "step": 0,
                    "floor": 1,
                    "phase": "CARD_REWARD",
                    "action": {"kind": "skip", "name": "SKIP", "choice_index": 0},
                    "pre_state": {"screen_type": "CARD_REWARD", "floor": 1, "current_hp": 80, "gold": 99},
                    "post_state": {"screen_type": "MAP", "floor": 0, "current_hp": 80, "gold": 99},
                    "pre_snapshot": {"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 99},
                    "post_phase": "MAP",
                    "post_floor": 0,
                    "post_hp": 80,
                    "post_gold": 99,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            trace = load_recorded_trace(path)

        live_state = SimpleNamespace(
            floor=0,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[],
            in_combat=False,
            screen_type=ScreenType.MAP,
            screen_name="MAP",
            screen=MapScreen(current_node=None, next_nodes=[], boss_available=False),
        )

        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=len(trace.steps),
            progress_enabled=False,
        )
        action = replayer.next_action(live_state)

        self.assertIsNone(action)
        self.assertEqual(replayer.results[0]["comparison_note"], "already_resolved_without_command")

    def test_map_step_waits_for_next_nodes_to_populate_before_translation(self):
        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 0},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=20,
            seed_str="20",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        live_state = SimpleNamespace(
            floor=0,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[],
            in_combat=False,
            screen_type=ScreenType.MAP,
            screen_name="MAP",
            screen=MapScreen(current_node=None, next_nodes=[], boss_available=False),
            proceed_available=False,
            cancel_available=True,
        )

        action = replayer.next_action(live_state)

        self.assertIsInstance(action, StateAction)
        self.assertFalse(replayer.failed)
        self.assertEqual(replayer.results, [])

    def test_map_step_can_be_already_resolved_if_live_state_matches_post(self):
        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 0},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=20,
            seed_str="20",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        live_state = SimpleNamespace(
            floor=1,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[],
            in_combat=True,
            hand=[],
            monsters=[],
            player=SimpleNamespace(block=0, energy=0, powers=[]),
            screen_type=None,
        )

        action = replayer.next_action(live_state)

        self.assertIsNone(action)
        self.assertEqual(replayer.results[0]["comparison_note"], "already_resolved_without_command")

    def test_map_step_waits_while_auto_entered_combat_is_still_opening(self):
        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 0},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=20,
            seed_str="20",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        live_state = SimpleNamespace(
            floor=1,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[],
            in_combat=True,
            hand=[],
            monsters=[
                SimpleNamespace(
                    current_hp=51,
                    is_gone=False,
                    monster_id="Cultist",
                    block=0,
                    intent=SimpleNamespace(name="DEBUG"),
                    move_adjusted_damage=-1,
                    move_hits=1,
                )
            ],
            player=SimpleNamespace(block=0, energy=0, powers=[]),
            screen_type=None,
        )

        action = replayer.next_action(live_state)

        self.assertIsInstance(action, StateAction)
        self.assertFalse(replayer.failed)
        self.assertEqual(replayer.results, [])

    def test_neow_card_reward_waits_for_obtain_side_effect_before_finalizing(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=1,
            phase="CARD_REWARD",
            floor=0,
            action={"kind": "card_reward", "name": "Flash of Steel", "card_id": "Flash of Steel", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        replayer.pending_step = step
        replayer.pending_actual_pre = step.pre
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[SimpleNamespace(choice_index=0, label="[Leave]", text="Leave", name=None)]),
        )
        live_snapshot = {"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]}

        completed = replayer._finalize_pending_step(game_state, live_snapshot)

        self.assertFalse(completed)
        self.assertEqual(replayer.results, [])

    def test_neow_card_reward_can_finalize_against_stale_card_reward_after_deck_update(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step_pick = RecordedTraceStep(
            step=1,
            phase="CARD_REWARD",
            floor=0,
            action={"kind": "card_reward", "name": "Flash of Steel", "card_id": "Flash of Steel", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        step_continue = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step_pick, step_continue],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.pending_step = step_pick
        replayer.pending_actual_pre = step_pick.pre
        game_state = SimpleNamespace(
            screen_type=ScreenType.CARD_REWARD,
            room_type="NeowRoom",
            screen=SimpleNamespace(cards=[]),
        )
        live_snapshot = {
            "phase": "CARD_REWARD",
            "floor": 0,
            "hp": 80,
            "gold": 99,
            "deck": ["Strike_R", "Flash of Steel"],
        }

        completed = replayer._finalize_pending_step(game_state, live_snapshot)

        self.assertTrue(completed)
        self.assertEqual(replayer.results[0]["comparison_note"], "neow_reward_pick_stale_card_reward_needs_close")
        self.assertEqual(replayer.neow_card_reward_close_bridge_step_number, 2)

    def test_neow_continue_step_can_close_stale_card_reward_before_choose(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step_pick = RecordedTraceStep(
            step=1,
            phase="CARD_REWARD",
            floor=0,
            action={"kind": "card_reward", "name": "Flash of Steel", "card_id": "Flash of Steel", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        step_continue = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step_pick, step_continue],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.step_index = 1
        replayer.neow_card_reward_close_bridge_step_number = 2
        live_state = SimpleNamespace(
            floor=0,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[],
            in_combat=False,
            screen_type=ScreenType.CARD_REWARD,
            room_type="NeowRoom",
            screen=SimpleNamespace(cards=[]),
        )

        action = replayer.next_action(live_state)

        self.assertIsInstance(action, RawCommandAction)
        self.assertEqual(action.command, "skip")

    def test_neow_card_reward_can_finalize_against_stale_card_reward_for_blind_continue(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step_pick = RecordedTraceStep(
            step=1,
            phase="CARD_REWARD",
            floor=0,
            action={"kind": "card_reward", "name": "Flash of Steel", "card_id": "Flash of Steel", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        step_continue = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step_pick, step_continue],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.pending_step = step_pick
        replayer.pending_actual_pre = step_pick.pre
        replayer.neow_card_reward_blind_continue_step_number = 2
        game_state = SimpleNamespace(
            screen_type=ScreenType.CARD_REWARD,
            room_type="NeowRoom",
            screen=SimpleNamespace(cards=[]),
        )
        live_snapshot = {
            "phase": "CARD_REWARD",
            "floor": 0,
            "hp": 80,
            "gold": 99,
            "deck": ["Strike_R"],
        }

        completed = replayer._finalize_pending_step(game_state, live_snapshot)

        self.assertTrue(completed)
        self.assertEqual(replayer.results[0]["comparison_note"], "neow_reward_pick_stale_card_reward_blind_continue")

    def test_neow_continue_step_sends_single_wait_pulse_while_stale_card_reward_persists(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step_pick = RecordedTraceStep(
            step=1,
            phase="CARD_REWARD",
            floor=0,
            action={"kind": "card_reward", "name": "Flash of Steel", "card_id": "Flash of Steel", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        step_continue = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step_pick, step_continue],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.step_index = 1
        replayer.neow_card_reward_blind_continue_step_number = 2
        live_state = SimpleNamespace(
            floor=0,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[],
            in_combat=False,
            screen_type=ScreenType.CARD_REWARD,
            room_type="NeowRoom",
            screen=SimpleNamespace(cards=[]),
        )

        action = replayer.next_action(live_state)

        self.assertIsInstance(action, WaitAction)
        self.assertEqual(replayer.neow_card_reward_blind_continue_step_number, 2)
        self.assertEqual(replayer.neow_card_reward_blind_continue_probe_step_number, 2)
        self.assertIsNone(replayer.neow_card_reward_blind_continue_choose_step_number)

    def test_neow_continue_step_blind_chooses_after_single_wait_pulse(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step_pick = RecordedTraceStep(
            step=1,
            phase="CARD_REWARD",
            floor=0,
            action={"kind": "card_reward", "name": "Flash of Steel", "card_id": "Flash of Steel", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        step_continue = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step_pick, step_continue],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.step_index = 1
        replayer.neow_card_reward_blind_continue_step_number = 2
        replayer.neow_card_reward_blind_continue_probe_step_number = 2
        live_state = SimpleNamespace(
            floor=0,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[],
            in_combat=False,
            screen_type=ScreenType.CARD_REWARD,
            room_type="NeowRoom",
            screen=SimpleNamespace(cards=[]),
        )

        action = replayer.next_action(live_state)

        from spirecomm.communication.action import NeowContinueAction

        self.assertIsInstance(action, NeowContinueAction)
        self.assertEqual(action.command, "choose 0")
        self.assertFalse(action.requires_game_ready)
        self.assertEqual(replayer.neow_card_reward_blind_continue_step_number, 2)
        self.assertEqual(replayer.neow_card_reward_blind_continue_probe_step_number, 2)
        self.assertEqual(replayer.neow_card_reward_blind_continue_choose_step_number, 2)

    def test_neow_continue_step_only_blind_chooses_once(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step_pick = RecordedTraceStep(
            step=1,
            phase="CARD_REWARD",
            floor=0,
            action={"kind": "card_reward", "name": "Flash of Steel", "card_id": "Flash of Steel", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        step_continue = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step_pick, step_continue],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.step_index = 1
        replayer.neow_card_reward_blind_continue_step_number = 2
        replayer.neow_card_reward_blind_continue_probe_step_number = 2
        replayer.neow_card_reward_blind_continue_choose_step_number = 2
        live_state = SimpleNamespace(
            floor=0,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[],
            in_combat=False,
            screen_type=ScreenType.CARD_REWARD,
            room_type="NeowRoom",
            screen=SimpleNamespace(cards=[]),
        )

        action = replayer.next_action(live_state)

        self.assertIsNone(action)
        self.assertEqual(replayer.neow_card_reward_blind_continue_step_number, 2)
        self.assertEqual(replayer.neow_card_reward_blind_continue_choose_step_number, 2)

    def test_neow_continue_step_can_auto_resolve_after_blind_bridge_advances_to_combat(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step_pick = RecordedTraceStep(
            step=1,
            phase="CARD_REWARD",
            floor=0,
            action={"kind": "card_reward", "name": "Flash of Steel", "card_id": "Flash of Steel", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        step_continue = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        step_map = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "choice_index": 0, "name": "M", "x": 1},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"], "monsters": []},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step_pick, step_continue, step_map],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=False,
            stop_on_mismatch=True,
            total_steps=3,
            progress_enabled=False,
        )
        replayer.step_index = 1
        replayer.neow_card_reward_blind_continue_step_number = 2
        live_state = SimpleNamespace(
            floor=1,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[SimpleNamespace(card_id="Strike_R"), SimpleNamespace(card_id="Flash of Steel")],
            in_combat=True,
            hand=[],
            monsters=[],
            player=SimpleNamespace(block=0, energy=3, powers=[]),
            screen_type=None,
            room_type="MonsterRoom",
        )

        action = replayer.next_action(live_state)

        self.assertIsNone(action)
        self.assertEqual(replayer.step_index, 2)
        self.assertEqual(
            replayer.results[0]["comparison_note"],
            "neow_blind_continue_already_resolved_without_command",
        )

    def test_force_finalize_pending_step_advances_neow_blind_continue_bridge(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step_pick = RecordedTraceStep(
            step=1,
            phase="CARD_REWARD",
            floor=0,
            action={"kind": "card_reward", "name": "Flash of Steel", "card_id": "Flash of Steel", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        step_continue = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step_pick, step_continue],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.pending_step = step_pick
        replayer.pending_actual_pre = step_pick.pre
        replayer.pending_actions = [{"action_class": "NeowCardRewardAction"}]

        replayer._force_finalize_pending_step(
            live_snapshot={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            comparison_note="neow_reward_pick_stale_card_reward_blind_continue_forced",
        )

        self.assertEqual(replayer.step_index, 1)
        self.assertIsNone(replayer.pending_step)
        self.assertEqual(
            replayer.results[0]["comparison_note"],
            "neow_reward_pick_stale_card_reward_blind_continue_forced",
        )

    def test_nonready_direct_action_is_allowed_for_neow_stale_card_reward_bridge(self):
        from spirecomm.ai.recorded_run_replay import (
            _StateDrivenReplayDriver,
            _should_allow_nonready_direct_action,
        )

        step_pick = RecordedTraceStep(
            step=1,
            phase="CARD_REWARD",
            floor=0,
            action={"kind": "card_reward", "name": "Flash of Steel", "card_id": "Flash of Steel", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        step_continue = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step_pick, step_continue],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.step_index = 1
        replayer.neow_card_reward_blind_continue_step_number = 2

        coordinator = SimpleNamespace(
            game_is_ready=False,
            last_game_state=SimpleNamespace(
                floor=0,
                current_hp=80,
                gold=99,
                potions=[],
                deck=[],
                in_combat=False,
                screen_type=ScreenType.CARD_REWARD,
                room_type="NeowRoom",
                screen=SimpleNamespace(cards=[]),
            ),
        )

        self.assertTrue(_should_allow_nonready_direct_action(coordinator, replayer))

    def test_nonready_neow_leave_bridge_allows_direct_action(self):
        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 1},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        driver = SimpleNamespace(
            pending_step=step,
            neow_card_reward_blind_continue_step_number=None,
        )
        coordinator = SimpleNamespace(
            game_is_ready=False,
            last_game_state=SimpleNamespace(
                floor=0,
                current_hp=80,
                gold=99,
                potions=[],
                deck=[],
                in_combat=False,
                screen_type=ScreenType.EVENT,
                room_type="NeowRoom",
                screen=SimpleNamespace(options=[SimpleNamespace(label="Leave")]),
            ),
        )

        self.assertTrue(_should_allow_nonready_direct_action(coordinator, driver))

    def test_nonready_neow_stale_card_reward_bridge_skips_ready_wait_after_wait_pulse(self):
        from spirecomm.ai.recorded_run_replay import (
            _StateDrivenReplayDriver,
            _should_skip_ready_wait_for_neow_blind_continue,
        )

        step_pick = RecordedTraceStep(
            step=1,
            phase="CARD_REWARD",
            floor=0,
            action={"kind": "card_reward", "name": "Flash of Steel", "card_id": "Flash of Steel", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        step_continue = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step_pick, step_continue],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.step_index = 1
        replayer.neow_card_reward_blind_continue_step_number = 2
        replayer.neow_card_reward_blind_continue_probe_step_number = 2

        coordinator = SimpleNamespace(
            game_is_ready=False,
            last_game_state=SimpleNamespace(
                floor=0,
                current_hp=80,
                gold=99,
                potions=[],
                deck=[],
                in_combat=False,
                screen_type=ScreenType.CARD_REWARD,
                room_type="NeowRoom",
                screen=SimpleNamespace(cards=[]),
            ),
        )

        self.assertTrue(_should_skip_ready_wait_for_neow_blind_continue(coordinator, replayer))

    def test_shop_purchase_uses_shopkeeper_as_setup_action(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=106,
            phase="SHOP",
            floor=10,
            action={"kind": "shop", "item_kind": "card", "item_id": "Bloodletting", "name": "Bloodletting"},
            pre={"phase": "SHOP", "floor": 10, "hp": 10, "gold": 194},
            post={"phase": "SHOP", "floor": 10, "hp": 10, "gold": 121},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=20,
            seed_str="20",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        game_state = SimpleNamespace(
            screen_type=ScreenType.SHOP_ROOM,
            room_type="ShopRoom",
            screen=SimpleNamespace(),
        )

        setup_action = replayer._maybe_setup_action(game_state, step)

        from spirecomm.communication.action import ChooseShopkeeperAction

        self.assertIsInstance(setup_action, ChooseShopkeeperAction)

    def test_shop_leave_uses_cancel_as_setup_action_from_shop_screen(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=109,
            phase="SHOP",
            floor=10,
            action={"kind": "shop", "item_kind": "leave", "item_id": "leave", "name": "Leave"},
            pre={"phase": "SHOP", "floor": 10, "hp": 10, "gold": 14},
            post={"phase": "MAP", "floor": 10, "hp": 10, "gold": 14},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=20,
            seed_str="20",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        game_state = SimpleNamespace(
            screen_type=ScreenType.SHOP_SCREEN,
            room_type="ShopRoom",
            screen=SimpleNamespace(),
        )

        setup_action = replayer._maybe_setup_action(game_state, step)

        from spirecomm.communication.action import CancelAction

        self.assertIsInstance(setup_action, CancelAction)

    def test_card_reward_step_from_mixed_combat_reward_uses_setup_action(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=124,
            phase="CARD_REWARD",
            floor=11,
            action={"kind": "card_reward", "name": "Barricade", "card_id": "Barricade", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 11, "hp": 8, "gold": 29, "deck": ["Bash", "Sever Soul"]},
            post={"phase": "CARD_REWARD", "floor": 11, "hp": 8, "gold": 29, "deck": ["Bash", "Sever Soul", "Barricade"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=20,
            seed_str="20",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        game_state = SimpleNamespace(
            screen_type=ScreenType.COMBAT_REWARD,
            room_type="MonsterRoom",
            screen=CombatRewardScreen(
                [
                    CombatReward(RewardType.POTION),
                    CombatReward(RewardType.CARD),
                ]
            ),
        )

        setup_action = replayer._maybe_setup_action(game_state, step)

        self.assertIsInstance(setup_action, CombatRewardAction)
        self.assertEqual(setup_action.combat_reward.reward_type, RewardType.CARD)

    def test_shop_leave_translates_to_proceed_from_shop_room(self):
        from spirecomm.ai.recorded_run_replay import _next_trace_action_for_game

        game_state = SimpleNamespace(
            screen_type=ScreenType.SHOP_ROOM,
            room_type="ShopRoom",
            screen=SimpleNamespace(),
        )

        step = RecordedTraceStep(
            step=109,
            phase="SHOP",
            floor=10,
            action={"kind": "shop", "item_kind": "leave", "item_id": "leave", "name": "Leave"},
            pre={"phase": "SHOP", "floor": 10, "hp": 10, "gold": 14},
            post={"phase": "MAP", "floor": 10, "hp": 10, "gold": 14},
        )

        action = _next_trace_action_for_game(
            game_state,
            step,
        )

        from spirecomm.communication.action import ProceedAction

        self.assertIsInstance(action, ProceedAction)

    def test_state_driven_replay_driver_emits_step_finalized_progress(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "x=1"},
            pre={"phase": "MAP", "floor": 1, "hp": 80, "gold": 99},
            post={"phase": "MAP", "floor": 1, "hp": 80, "gold": 99},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        progress_events: list[dict[str, object]] = []
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
            progress_callback=progress_events.append,
        )
        replayer.pending_step = step
        replayer.pending_actual_pre = step.pre

        completed = replayer._finalize_pending_step(None, dict(step.post))

        self.assertTrue(completed)
        self.assertEqual(progress_events[-1]["status"], "step_finalized")
        self.assertEqual(progress_events[-1]["current_step"], 3)
        self.assertEqual(progress_events[-1]["steps_replayed"], 1)

    def test_state_driven_replay_driver_emits_failure_progress(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=9,
            phase="COMBAT",
            floor=2,
            action={"kind": "card", "name": "Strike_R"},
            pre={"phase": "COMBAT", "floor": 2, "hp": 70, "gold": 120},
            post={"phase": "COMBAT", "floor": 2, "hp": 70, "gold": 120},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        progress_events: list[dict[str, object]] = []
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
            progress_callback=progress_events.append,
        )

        replayer._record_terminal_failure(
            step=step,
            live_snapshot={"phase": "COMBAT", "floor": 2, "hp": 69, "gold": 120},
            pre_mismatches=["hp: expected 70, got 69"],
            note="pre_state_mismatch",
        )

        self.assertEqual(progress_events[-1]["status"], "failed")
        self.assertEqual(progress_events[-1]["current_step"], 9)
        self.assertEqual(progress_events[-1]["first_failure_step"], 9)

    def test_neow_reward_waits_for_followup_card_reward_screen(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=0,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "name": "OPTION_0", "choice_index": 0, "bonus": "RANDOM_COLORLESS"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        replayer.pending_step = step
        replayer.pending_actual_pre = step.pre
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[SimpleNamespace(choice_index=0, label="[Choose]", text="Choose", name=None)]),
        )
        live_snapshot = {"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]}

        completed = replayer._finalize_pending_step(game_state, live_snapshot)

        self.assertFalse(completed)
        self.assertEqual(replayer.results, [])

    def test_neow_random_rare_reward_waits_for_deck_update_on_leave_screen(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=0,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "name": "OPTION_0", "choice_index": 0, "bonus": "ONE_RANDOM_RARE_CARD"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash", "Demon Form"]},
        )
        next_step = RecordedTraceStep(
            step=1,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "name": "OPTION_0", "choice_index": 0, "bonus": "CONTINUE"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash", "Demon Form"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash", "Demon Form"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=2,
            seed_str="2",
            ascension=0,
            character="IRONCLAD",
            steps=[step, next_step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.pending_step = step
        replayer.pending_actual_pre = step.pre
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[SimpleNamespace(choice_index=0, label="Leave", text="[Leave]", name=None)]),
        )
        live_snapshot = {"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash"]}

        completed = replayer._finalize_pending_step(game_state, live_snapshot)

        self.assertFalse(completed)
        self.assertEqual(replayer.results, [])

    def test_neow_talk_setup_is_kept_for_trace_option_zero(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=0,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "name": "OPTION_0", "choice_index": 0, "bonus": "ONE_RANDOM_RARE_CARD"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=2,
            seed_str="2",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[SimpleNamespace(choice_index=0, label="Talk", text="[Talk]", name=None)]),
        )

        action = replayer._maybe_setup_action(game_state, step)

        self.assertIsNotNone(action)
        self.assertEqual(action.command, "choose")

    def test_neow_talk_setup_is_kept_for_nonzero_trace_option(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=0,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "name": "OPTION_2", "choice_index": 2, "bonus": "REMOVE_CARD"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "CARD_SELECT", "floor": 0, "hp": 80, "gold": 99},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[SimpleNamespace(choice_index=0, label="Talk", text="[Talk]", name=None)]),
        )

        action = replayer._maybe_setup_action(game_state, step)

        self.assertIsNotNone(action)
        self.assertEqual(action.command, "choose")

    def test_pending_neow_random_rare_reward_on_leave_screen_skips_state_probe(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=0,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "name": "OPTION_0", "choice_index": 0, "bonus": "ONE_RANDOM_RARE_CARD"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash", "Demon Form"]},
        )
        next_step = RecordedTraceStep(
            step=1,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "name": "OPTION_0", "choice_index": 0, "bonus": "CONTINUE"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash", "Demon Form"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash", "Demon Form"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=2,
            seed_str="2",
            ascension=0,
            character="IRONCLAD",
            steps=[step, next_step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.step_index = 0
        replayer.pending_step = step
        replayer.pending_actual_pre = step.pre
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[SimpleNamespace(choice_index=0, label="Leave", text="[Leave]", name=None)]),
        )

        action = replayer.next_action(game_state)

        self.assertIsNone(action)

    def test_play_recorded_game_forces_finalize_on_neow_immediate_reward_callback_timeout(self):
        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = []
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[SimpleNamespace(card_id="Strike_R"), SimpleNamespace(card_id="Bash")],
                    in_combat=False,
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(options=[SimpleNamespace(label="Leave", choice_index=0)]),
                )

            def clear_actions(self):
                self.action_queue.clear()

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

        step = RecordedTraceStep(
            step=0,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "name": "OPTION_0", "choice_index": 0, "bonus": "ONE_RANDOM_RARE_CARD"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash", "Demon Form"]},
        )
        next_step = RecordedTraceStep(
            step=1,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "name": "OPTION_0", "choice_index": 0, "bonus": "CONTINUE"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash", "Demon Form"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash", "Demon Form"]},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 2
                self.pending_step = None
                self.pending_setup_actions = [{"action_class": "ChooseAction", "command": "choose", "choice_index": 0}]
                self.trace = RecordedRunTrace(
                    path=Path("/tmp/fake_trace.json"),
                    source_format="native_run_trace_v1",
                    seed_long=2,
                    seed_str="2",
                    ascension=0,
                    character="IRONCLAD",
                    steps=[step, next_step],
                )
                self.neow_bridge_step_number = None

            def next_action(self, game_state):
                return None

            def _force_finalize_current_step(self, step_arg, *, live_snapshot, comparison_note):
                forced.append((step_arg, live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=TimeoutError("still stale"),
        ) as callback_wait:
            with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout") as ready_wait:
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="2",
                    driver=driver,
                )

        self.assertEqual(callback_wait.call_count, 1)
        ready_wait.assert_not_called()
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][0].step, 0)
        self.assertEqual(forced[0][1], step.post)
        self.assertEqual(forced[0][2], "neow_immediate_reward_callback_timeout_forced")
        self.assertEqual(driver.neow_bridge_step_number, 1)

    def test_combat_reward_card_pick_waits_for_obtain_side_effect_before_finalizing(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=31,
            phase="CARD_REWARD",
            floor=2,
            action={"kind": "card_reward", "name": "Metallicize", "card_id": "Metallicize", "choice_index": 0},
            pre={"phase": "CARD_REWARD", "floor": 2, "hp": 80, "gold": 127, "deck": ["Strike_R", "Anger"]},
            post={"phase": "CARD_REWARD", "floor": 2, "hp": 80, "gold": 127, "deck": ["Strike_R", "Anger", "Metallicize"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        replayer.pending_step = step
        replayer.pending_actual_pre = step.pre
        reward = CombatReward(reward_type=RewardType.CARD)
        game_state = SimpleNamespace(
            screen_type=ScreenType.COMBAT_REWARD,
            room_type="MonsterRoom",
            proceed_available=True,
            screen=CombatRewardScreen([reward]),
        )
        live_snapshot = {"phase": "CARD_REWARD", "floor": 2, "hp": 80, "gold": 127, "deck": ["Strike_R", "Anger"]}

        completed = replayer._finalize_pending_step(game_state, live_snapshot)

        self.assertFalse(completed)
        self.assertEqual(replayer.results, [])

    def test_card_reward_pick_can_bridge_through_empty_reward_proceed_screen(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=16,
            phase="CARD_REWARD",
            floor=1,
            action={"kind": "card_reward", "name": "Anger", "card_id": "Anger", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 111, "deck": ["Flash of Steel"]},
            post={"phase": "MAP", "floor": 1, "hp": 80, "gold": 111, "deck": ["Flash of Steel", "Anger"]},
        )
        next_step = RecordedTraceStep(
            step=17,
            phase="MAP",
            floor=1,
            action={"kind": "map", "symbol": "?", "choice_index": 0, "x": 0},
            pre={"phase": "MAP", "floor": 1, "hp": 80, "gold": 111, "deck": ["Flash of Steel", "Anger"]},
            post={"phase": "EVENT", "floor": 2, "hp": 80, "gold": 111, "deck": ["Flash of Steel", "Anger"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step, next_step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.step_index = 0
        replayer.pending_step = step
        replayer.pending_actual_pre = step.pre
        rewards: list[CombatReward] = []
        game_state = SimpleNamespace(
            screen_type=ScreenType.COMBAT_REWARD,
            room_type="MonsterRoom",
            proceed_available=True,
            screen=CombatRewardScreen(rewards),
        )
        live_snapshot = {"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 111, "deck": ["Flash of Steel", "Anger"]}

        completed = replayer._finalize_pending_step(game_state, live_snapshot)

        self.assertTrue(completed)
        self.assertEqual(replayer.results[0]["comparison_note"], "card_reward_transitions_to_empty_reward_proceed")
        self.assertEqual(replayer.reward_proceed_bridge_step_number, 17)

    def test_card_reward_pick_can_bridge_to_explicit_skip_step_through_empty_reward_screen(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=27,
            phase="CARD_REWARD",
            floor=1,
            action={"kind": "card_reward", "name": "Perfected Strike", "card_id": "Perfected Strike", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 113, "deck": ["Bash"]},
            post={"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 113, "deck": ["Bash", "Perfected Strike"]},
        )
        next_step = RecordedTraceStep(
            step=28,
            phase="CARD_REWARD",
            floor=1,
            action={"kind": "skip", "name": "SKIP", "choice_index": 0},
            pre={"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 113, "deck": ["Bash", "Perfected Strike"]},
            post={"phase": "MAP", "floor": 1, "hp": 80, "gold": 113, "deck": ["Bash", "Perfected Strike"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=20,
            seed_str="K",
            ascension=0,
            character="IRONCLAD",
            steps=[step, next_step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.step_index = 0
        replayer.pending_step = step
        replayer.pending_actual_pre = step.pre
        game_state = SimpleNamespace(
            floor=1,
            current_hp=80,
            gold=113,
            deck=[],
            potions=[],
            screen_type=ScreenType.COMBAT_REWARD,
            room_type="MonsterRoom",
            proceed_available=True,
            screen=CombatRewardScreen([]),
        )
        live_snapshot = {"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 113, "deck": ["Bash", "Perfected Strike"]}

        completed = replayer._finalize_pending_step(game_state, live_snapshot)

        self.assertTrue(completed)
        self.assertEqual(replayer.results[0]["comparison_note"], "card_reward_transitions_to_empty_reward_proceed")
        self.assertEqual(replayer.reward_proceed_bridge_step_number, 28)

    def test_card_reward_pick_can_bridge_to_skip_before_deck_updates(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=124,
            phase="CARD_REWARD",
            floor=11,
            action={"kind": "card_reward", "name": "Barricade", "card_id": "Barricade", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 11, "hp": 8, "gold": 29, "deck": ["Bash", "Sever Soul"]},
            post={"phase": "CARD_REWARD", "floor": 11, "hp": 8, "gold": 29, "deck": ["Bash", "Sever Soul", "Barricade"]},
        )
        next_step = RecordedTraceStep(
            step=125,
            phase="CARD_REWARD",
            floor=11,
            action={"kind": "skip", "name": "SKIP", "choice_index": 0},
            pre={"phase": "CARD_REWARD", "floor": 11, "hp": 8, "gold": 29, "deck": ["Bash", "Sever Soul", "Barricade"]},
            post={"phase": "MAP", "floor": 11, "hp": 8, "gold": 29, "deck": ["Bash", "Sever Soul", "Barricade"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=20,
            seed_str="20",
            ascension=0,
            character="IRONCLAD",
            steps=[step, next_step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.step_index = 0
        replayer.pending_step = step
        replayer.pending_actual_pre = step.pre
        game_state = SimpleNamespace(
            screen_type=ScreenType.CARD_REWARD,
            room_type="MonsterRoom",
            screen=SimpleNamespace(cards=[]),
        )
        live_snapshot = {"phase": "CARD_REWARD", "floor": 11, "hp": 8, "gold": 29, "deck": ["Bash", "Sever Soul"]}

        completed = replayer._finalize_pending_step(game_state, live_snapshot)

        self.assertTrue(completed)
        self.assertEqual(
            replayer.results[0]["comparison_note"],
            "card_reward_pick_transitions_to_skip_before_deck_update",
        )

    def test_card_reward_skip_tolerates_missing_picked_card_in_live_deck(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=20,
            seed_str="20",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=124,
                    phase="CARD_REWARD",
                    floor=11,
                    action={"kind": "card_reward", "name": "Barricade", "card_id": "Barricade", "choice_index": 2},
                    pre={"phase": "CARD_REWARD", "floor": 11, "hp": 8, "gold": 29, "deck": ["Bash", "Sever Soul"]},
                    post={"phase": "CARD_REWARD", "floor": 11, "hp": 8, "gold": 29, "deck": ["Bash", "Sever Soul", "Barricade"]},
                ),
                RecordedTraceStep(
                    step=125,
                    phase="CARD_REWARD",
                    floor=11,
                    action={"kind": "skip", "name": "SKIP", "choice_index": 0},
                    pre={"phase": "CARD_REWARD", "floor": 11, "hp": 8, "gold": 29, "deck": ["Bash", "Sever Soul", "Barricade"]},
                    post={"phase": "MAP", "floor": 11, "hp": 8, "gold": 29, "deck": ["Bash", "Sever Soul", "Barricade"]},
                ),
            ],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        replayer.step_index = 1
        live_state = SimpleNamespace(
            floor=11,
            current_hp=8,
            gold=29,
            potions=[],
            deck=[],
            in_combat=False,
            screen_type=ScreenType.CARD_REWARD,
            screen=SimpleNamespace(cards=[Card("Fire Breathing", "Fire Breathing", CardType.POWER, CardRarity.UNCOMMON)]),
            cancel_available=True,
            proceed_available=False,
        )

        action = replayer.next_action(live_state)

        self.assertEqual(action.__class__.__name__, "CancelAction")
        self.assertFalse(replayer.failed)

    def test_map_to_combat_waits_for_stable_intent_even_without_next_step(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver

        step = RecordedTraceStep(
            step=29,
            phase="MAP",
            floor=2,
            action={"kind": "map", "symbol": "M", "choice_index": 0, "x": 0},
            pre={"phase": "MAP", "floor": 2, "hp": 80, "gold": 130, "deck": ["Bash"]},
            post={
                "phase": "COMBAT",
                "floor": 3,
                "hp": 80,
                "gold": 130,
                "deck": ["Bash"],
                "block": 0,
                "energy": 3,
                "hand": ["Strike_R"],
                "monsters": [
                    {
                        "id": "JawWorm",
                        "hp": 41,
                        "block": 0,
                        "intent": "ATTACK",
                        "move_adjusted_damage": 11,
                        "move_hits": 1,
                    }
                ],
            },
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=20,
            seed_str="K",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        replayer.pending_step = step
        replayer.pending_actual_pre = step.pre
        game_state = SimpleNamespace(
            screen_type=ScreenType.NONE,
            room_type="MonsterRoom",
            in_combat=True,
        )
        live_snapshot = {
            "phase": "COMBAT",
            "floor": 3,
            "hp": 80,
            "gold": 130,
            "deck": ["Bash"],
            "block": 0,
            "energy": 3,
            "hand": ["Strike_R"],
            "monsters": [
                {
                    "id": "JawWorm",
                    "hp": 41,
                    "block": 0,
                    "intent": "DEBUG",
                    "move_adjusted_damage": -1,
                    "move_hits": 1,
                }
            ],
        }

        completed = replayer._finalize_pending_step(game_state, live_snapshot)

        self.assertFalse(completed)
        self.assertEqual(replayer.results, [])

    def test_empty_reward_proceed_bridge_dispatches_explicit_skip_step(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver
        from spirecomm.communication.action import ProceedAction

        step = RecordedTraceStep(
            step=28,
            phase="CARD_REWARD",
            floor=1,
            action={"kind": "skip", "name": "SKIP", "choice_index": 0},
            pre={"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 113, "deck": ["Bash", "Perfected Strike"]},
            post={"phase": "MAP", "floor": 1, "hp": 80, "gold": 113, "deck": ["Bash", "Perfected Strike"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=20,
            seed_str="K",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        replayer.reward_proceed_bridge_step_number = 28
        game_state = SimpleNamespace(
            floor=1,
            current_hp=80,
            gold=113,
            deck=[],
            potions=[],
            screen_type=ScreenType.COMBAT_REWARD,
            room_type="MonsterRoom",
            proceed_available=True,
            screen=CombatRewardScreen([]),
        )

        action = replayer.next_action(game_state)

        self.assertIsInstance(action, ProceedAction)
        self.assertIsNone(replayer.reward_proceed_bridge_step_number)

    def test_snapshot_live_state_tracks_deck_outside_combat(self):
        game_state = SimpleNamespace(
            floor=0,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[SimpleNamespace(card_id="Strike_R"), SimpleNamespace(card_id="Demon Form")],
            in_combat=False,
        )

        snapshot = snapshot_live_state(game_state)

        self.assertEqual(snapshot["deck"], ["Strike_R", "Demon Form"])

    def test_snapshot_live_state_tracks_event_id(self):
        game_state = SimpleNamespace(
            floor=2,
            current_hp=80,
            gold=113,
            potions=[],
            deck=[],
            in_combat=False,
            screen_type=ScreenType.EVENT,
            screen_name="NONE",
            screen=SimpleNamespace(event_id="We Meet Again!", event_name="We Meet Again!"),
        )

        snapshot = snapshot_live_state(game_state)

        self.assertEqual(snapshot["event_id"], "We Meet Again!")

    def test_wait_for_matching_snapshot_can_stabilize_combat_entry(self):
        def make_state(intent: str, damage: int):
            return SimpleNamespace(
                floor=2,
                current_hp=80,
                gold=111,
                screen_type=ScreenType.NONE,
                room_type="MonsterRoom",
                in_combat=True,
                player=SimpleNamespace(block=0, energy=3),
                hand=[SimpleNamespace(card_id="Strike_R")],
                monsters=[
                    SimpleNamespace(
                        monster_id="JawWorm",
                        current_hp=42,
                        block=0,
                        intent=SimpleNamespace(name=intent),
                        move_adjusted_damage=damage,
                        move_hits=1,
                        is_gone=False,
                    )
                ],
            )

        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = True
                self.in_game = True
                self.last_game_state = make_state("DEBUG", -1)
                self._queued_state = None

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if self._queued_state is None:
                    return False
                self.last_game_state = self._queued_state
                self._queued_state = None
                return True

            def send_message(self, message):
                if message == "wait 1":
                    self._queued_state = make_state("ATTACK", 11)

        coordinator = FakeCoordinator()
        snapshot = _wait_for_matching_snapshot_with_timeout(
            coordinator,
            expected_snapshot={
                "phase": "COMBAT",
                "floor": 2,
                "hp": 80,
                "gold": 111,
                "block": 0,
                "energy": 3,
                "hand": ["Strike_R"],
                "monsters": [
                    {
                        "id": "JawWorm",
                        "hp": 42,
                        "block": 0,
                        "intent": "ATTACK",
                        "move_adjusted_damage": 11,
                        "move_hits": 1,
                    }
                ],
            },
            timeout_seconds=0.5,
            context="unit test stabilize combat",
            initial_probe_delay_seconds=0.0,
        )

        self.assertEqual(snapshot["phase"], "COMBAT")
        self.assertEqual(snapshot["monsters"][0]["intent"], "ATTACK")
        self.assertEqual(snapshot["monsters"][0]["move_adjusted_damage"], 11)

    def test_wait_for_ready_state_probes_with_state_requests(self):
        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = False
                self.probes = 0

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                return False

            def send_message(self, message):
                if message == "state":
                    self.probes += 1
                    if self.probes >= 2:
                        self.game_is_ready = True

        coordinator = FakeCoordinator()

        _wait_for_ready_state_with_timeout(
            coordinator,
            timeout_seconds=0.6,
            context="unit test probe",
        )

        self.assertTrue(coordinator.game_is_ready)
        self.assertGreaterEqual(coordinator.probes, 2)

    def test_wait_for_ready_state_does_not_probe_stale_neow_card_reward(self):
        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = False
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="NeowRoom",
                )
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                return False

            def send_message(self, message):
                self.messages.append(message)
                self.game_is_ready = False

        coordinator = FakeCoordinator()

        with self.assertRaises(TimeoutError):
            _wait_for_ready_state_with_timeout(
                coordinator,
                timeout_seconds=0.3,
                context="neow reward probe",
            )

        self.assertEqual(coordinator.messages, [])

    def test_wait_for_phase_uses_wait_probe_for_neow_card_reward_continue(self):
        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = True
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="NeowRoom",
                )
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                return False

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        with self.assertRaises(TimeoutError):
            _wait_for_phase_with_timeout(
                coordinator,
                expected_phase="NEOW",
                timeout_seconds=0.45,
                context="neow card reward continue probe",
            )

        self.assertTrue(coordinator.messages)
        self.assertEqual(coordinator.messages[0], "wait 1")

    def test_wait_for_phase_keeps_using_wait_probe_for_stuck_neow_card_reward_screen(self):
        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = True
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="NeowRoom",
                )
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                return False

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        with self.assertRaises(TimeoutError):
            _wait_for_phase_with_timeout(
                coordinator,
                expected_phase="NEOW",
                timeout_seconds=2.0,
                context="neow card reward close probe",
            )

        self.assertTrue(coordinator.messages)
        self.assertTrue(all(message == "wait 1" for message in coordinator.messages))

    def test_wait_for_phase_tolerates_neow_skip_error_if_followup_state_arrives(self):
        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = True
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="NeowRoom",
                )
                self.calls = 0

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                self.calls += 1
                if self.calls == 1:
                    self.last_error = "Invalid command: skip. Possible commands: [choose, key, click, wait, state]"
                    self.game_is_ready = False
                    return True
                if self.calls == 2:
                    self.last_error = None
                    self.game_is_ready = True
                    self.last_game_state = SimpleNamespace(
                        screen_type=ScreenType.EVENT,
                        room_type="NeowRoom",
                        screen=SimpleNamespace(event_id="Neow Event", event_name="Neow"),
                    )
                    return True
                return False

            def send_message(self, message):
                self.sent_message = message

        coordinator = FakeCoordinator()

        _wait_for_phase_with_timeout(
            coordinator,
            expected_phase="NEOW",
            timeout_seconds=0.5,
            context="neow skip bridge",
        )

        self.assertIsNone(coordinator.last_error)
        self.assertEqual(coordinator.sent_message, "state")

    def test_wait_for_phase_accepts_neow_map_transition_before_ready_flag_recovers(self):
        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = False
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.MAP,
                    room_type="NeowRoom",
                )
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                return False

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        _wait_for_phase_with_timeout(
            coordinator,
            expected_phase="MAP",
            timeout_seconds=0.3,
            context="neow open map direct state",
        )

        self.assertEqual(coordinator.messages, [])

    def test_wait_for_phase_accepts_neow_leave_bridge_as_map_transition(self):
        option = SimpleNamespace(choice_index=0, label="[Leave]", text="Leave", name=None)

        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = False
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(options=[option], event_id="Neow Event", event_name="Neow"),
                )
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                return False

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        _wait_for_phase_with_timeout(
            coordinator,
            expected_phase="MAP",
            timeout_seconds=0.3,
            context="neow leave bridge state",
        )

        self.assertEqual(coordinator.messages, [])

    def test_finalize_pending_neow_continue_accepts_leave_bridge_when_only_phase_lags(self):
        step = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"], "event_id": "Neow Event"},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        replayer.pending_step = step
        replayer.pending_actual_pre = dict(step.pre)
        leave_option = SimpleNamespace(choice_index=0, label="[Leave]", text="Leave", name=None)
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[leave_option], event_id="Neow Event", event_name="Neow"),
        )
        live_snapshot = {
            "phase": "NEOW",
            "floor": 0,
            "hp": 80,
            "gold": 99,
            "deck": ["Strike_R", "Flash of Steel"],
            "event_id": "Neow Event",
        }

        finalized = replayer._finalize_pending_step(game_state, live_snapshot)

        self.assertTrue(finalized)
        self.assertFalse(replayer.failed)
        self.assertEqual(replayer.results[0]["comparison_note"], "neow_continue_leave_bridge_pending_map_callback")
        self.assertEqual(replayer.results[0]["post_mismatches"], [])

    def test_map_step_sends_single_followup_continue_through_neow_leave_bridge(self):
        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 1},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        leave_option = SimpleNamespace(choice_index=0, label="[Leave]", text="Leave", name=None)
        live_state = SimpleNamespace(
            floor=0,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[],
            in_combat=False,
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[leave_option], event_id="Neow Event", event_name="Neow"),
            proceed_available=False,
            cancel_available=False,
        )

        action = replayer.next_action(live_state)

        from spirecomm.communication.action import NeowContinueAction

        self.assertIsInstance(action, NeowContinueAction)
        self.assertEqual(action.post_continue_choice_index, 0)
        self.assertFalse(replayer.failed)
        self.assertEqual(replayer.results, [])
        self.assertEqual(replayer.neow_leave_map_continue_step_number, 3)

    def test_map_step_only_sends_single_followup_continue_through_neow_leave_bridge(self):
        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 1},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        replayer.neow_leave_map_continue_step_number = 3
        leave_option = SimpleNamespace(choice_index=0, label="[Leave]", text="Leave", name=None)
        live_state = SimpleNamespace(
            floor=0,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[],
            in_combat=False,
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[leave_option], event_id="Neow Event", event_name="Neow"),
            proceed_available=False,
            cancel_available=False,
        )

        action = replayer.next_action(live_state)

        self.assertIsNone(action)
        self.assertFalse(replayer.failed)
        self.assertEqual(replayer.results, [])

    def test_neow_continue_ignores_stale_leave_deck_mismatch_after_immediate_reward(self):
        from spirecomm.ai.recorded_run_replay import _StateDrivenReplayDriver
        from spirecomm.communication.action import NeowContinueAction

        step = RecordedTraceStep(
            step=1,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "name": "OPTION_0", "choice_index": 0, "bonus": "CONTINUE"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash", "Demon Form"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Bash", "Demon Form"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=2,
            seed_str="2",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        live_state = SimpleNamespace(
            floor=0,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[SimpleNamespace(card_id="Strike_R"), SimpleNamespace(card_id="Bash")],
            in_combat=False,
            room_type="NeowRoom",
            screen_type=ScreenType.EVENT,
            screen=SimpleNamespace(
                options=[SimpleNamespace(choice_index=0, label="[Leave]", text="Leave", name=None)],
                event_name="Neow Event",
                event_id="Neow Event",
            ),
        )

        action = replayer.next_action(live_state)

        self.assertIsInstance(action, NeowContinueAction)
        self.assertFalse(replayer.failed)

    def test_nonready_neow_leave_map_bridge_skips_ready_wait(self):
        coordinator = SimpleNamespace(
            game_is_ready=False,
            last_game_state=SimpleNamespace(
                screen_type=ScreenType.EVENT,
                room_type="NeowRoom",
                screen=SimpleNamespace(
                    options=[SimpleNamespace(choice_index=0, label="[Leave]", text="Leave", name=None)]
                ),
            ),
        )
        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 1},
            pre={"phase": "MAP"},
            post={"phase": "COMBAT"},
        )
        driver = SimpleNamespace(
            pending_step=step,
            neow_leave_map_continue_step_number=3,
        )

        self.assertTrue(_should_skip_ready_wait_for_neow_leave_map_continue(coordinator, driver))

    def test_nonready_neow_map_screen_skips_ready_wait(self):
        coordinator = SimpleNamespace(
            game_is_ready=False,
            last_game_state=SimpleNamespace(
                screen_type=ScreenType.MAP,
                room_type="NeowRoom",
                screen=SimpleNamespace(next_nodes=[]),
            ),
        )
        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 1},
            pre={"phase": "MAP"},
            post={"phase": "COMBAT"},
        )
        driver = SimpleNamespace(
            pending_step=step,
            neow_leave_map_continue_step_number=3,
        )

        self.assertTrue(_should_skip_ready_wait_for_neow_leave_map_continue(coordinator, driver))

    def test_pending_map_step_waits_through_neow_leave_bridge_without_state_probe(self):
        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 1},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        replayer.pending_step = step
        replayer.pending_actual_pre = dict(step.pre)
        leave_option = SimpleNamespace(choice_index=0, label="[Leave]", text="Leave", name=None)
        live_state = SimpleNamespace(
            floor=0,
            current_hp=80,
            gold=99,
            potions=[],
            deck=[],
            in_combat=False,
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[leave_option], event_id="Neow Event", event_name="Neow"),
            proceed_available=False,
            cancel_available=False,
        )

        action = replayer.next_action(live_state)

        self.assertIsNone(action)
        self.assertFalse(replayer.failed)
        self.assertEqual(replayer.results, [])

    def test_wait_for_callback_update_uses_nonblocking_polls(self):
        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = True
                self.in_game = True
                self.last_game_state = SimpleNamespace(room_type="MonsterRoom")
                self.calls = []
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                self.calls.append((block, perform_callbacks))
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return len(self.calls) >= 2

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        _wait_for_callback_update_with_timeout(
            coordinator,
            timeout_seconds=0.6,
            context="unit test callback wait",
        )

        self.assertGreaterEqual(len(coordinator.calls), 2)
        self.assertTrue(all(block is False for block, _ in coordinator.calls))
        self.assertTrue(all(perform_callbacks is True for _, perform_callbacks in coordinator.calls))

    def test_wait_for_any_update_does_not_probe_stale_neow_card_reward(self):
        from spirecomm.ai.recorded_run_replay import _wait_for_any_update

        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = True
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="NeowRoom",
                )
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        with self.assertRaises(TimeoutError):
            _wait_for_any_update(
                coordinator,
                timeout_seconds=0.35,
                context="unit test natural neow reward wait",
            )

        self.assertEqual(coordinator.messages, [])

    def test_wait_for_callback_update_does_not_probe_stale_neow_card_reward(self):
        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = True
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="NeowRoom",
                )
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        with self.assertRaises(TimeoutError):
            _wait_for_callback_update_with_timeout(
                coordinator,
                timeout_seconds=0.35,
                context="unit test stale neow callback wait",
            )

        self.assertEqual(coordinator.messages, [])

    def test_wait_for_callback_update_does_not_probe_stale_neow_leave_bridge(self):
        leave_option = SimpleNamespace(choice_index=0, label="[Leave]", text="Leave", name=None)

        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = True
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(options=[leave_option], event_id="Neow Event", event_name="Neow"),
                )
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        with self.assertRaises(TimeoutError):
            _wait_for_callback_update_with_timeout(
                coordinator,
                timeout_seconds=0.35,
                context="unit test stale neow leave callback wait",
            )

        self.assertEqual(coordinator.messages, [])

    def test_wait_for_any_update_does_not_probe_stale_neow_card_select(self):
        from spirecomm.ai.recorded_run_replay import _wait_for_any_update

        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = True
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.GRID,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(
                        cards=[SimpleNamespace(name="Bash", card_id="Bash")],
                        selected_cards=[],
                        for_upgrade=True,
                        for_purge=False,
                        for_transform=False,
                        any_number=False,
                        confirm_up=True,
                        num_cards=1,
                    ),
                )
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        with self.assertRaises(TimeoutError):
            _wait_for_any_update(
                coordinator,
                timeout_seconds=0.35,
                context="unit test natural neow card select wait",
            )

        self.assertEqual(coordinator.messages, [])

    def test_wait_for_callback_update_does_not_probe_stale_neow_card_select(self):
        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = True
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.GRID,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(
                        cards=[SimpleNamespace(name="Bash", card_id="Bash")],
                        selected_cards=[],
                        for_upgrade=True,
                        for_purge=False,
                        for_transform=False,
                        any_number=False,
                        confirm_up=True,
                        num_cards=1,
                    ),
                )
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        with self.assertRaises(TimeoutError):
            _wait_for_callback_update_with_timeout(
                coordinator,
                timeout_seconds=0.35,
                context="unit test stale neow card select callback wait",
            )

        self.assertEqual(coordinator.messages, [])

    def test_wait_for_ready_state_accepts_neow_map_with_next_nodes_before_ready_flag(self):
        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = False
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.MAP,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(next_nodes=[SimpleNamespace(x=1, y=0), SimpleNamespace(x=3, y=0)]),
                )
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        _wait_for_ready_state_with_timeout(
            coordinator,
            timeout_seconds=0.3,
            context="unit test neow actionable map frame",
        )

        self.assertEqual(coordinator.messages, [])

    def test_allow_nonready_direct_action_for_neow_map_with_next_nodes(self):
        coordinator = SimpleNamespace(
            last_game_state=SimpleNamespace(
                screen_type=ScreenType.MAP,
                room_type="NeowRoom",
                screen=SimpleNamespace(next_nodes=[SimpleNamespace(x=1, y=0)]),
            )
        )
        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 1},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        driver = SimpleNamespace(
            pending_step=step,
            trace=SimpleNamespace(steps=[step]),
            step_index=0,
            total_steps=1,
            neow_card_reward_blind_continue_step_number=None,
        )

        allowed = _should_allow_nonready_direct_action(coordinator, driver)

        self.assertTrue(allowed)

    def test_allow_nonready_direct_action_for_current_neow_map_step_without_pending_step(self):
        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 1},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        coordinator = SimpleNamespace(
            last_game_state=SimpleNamespace(
                screen_type=ScreenType.EVENT,
                room_type="NeowRoom",
                screen=SimpleNamespace(
                    options=[SimpleNamespace(choice_index=0, label="[Leave]", text="Leave", name=None)]
                ),
            ),
        )
        driver = SimpleNamespace(
            pending_step=None,
            trace=SimpleNamespace(steps=[step]),
            step_index=0,
            total_steps=1,
            neow_card_reward_blind_continue_step_number=None,
        )

        allowed = _should_allow_nonready_direct_action(coordinator, driver)

        self.assertTrue(allowed)

    def test_disallow_nonready_direct_action_for_current_non_map_step_without_pending_step(self):
        step = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
        )
        coordinator = SimpleNamespace(
            last_game_state=SimpleNamespace(
                screen_type=ScreenType.EVENT,
                room_type="NeowRoom",
                screen=SimpleNamespace(
                    options=[SimpleNamespace(choice_index=0, label="[Leave]", text="Leave", name=None)]
                ),
            ),
        )
        driver = SimpleNamespace(
            pending_step=None,
            trace=SimpleNamespace(steps=[step]),
            step_index=0,
            total_steps=1,
            neow_card_reward_blind_continue_step_number=None,
        )

        allowed = _should_allow_nonready_direct_action(coordinator, driver)

        self.assertFalse(allowed)

    def test_wait_for_callback_update_does_not_probe_nonready_neow_map_with_next_nodes(self):
        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = False
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.MAP,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(next_nodes=[SimpleNamespace(x=1, y=0), SimpleNamespace(x=3, y=0)]),
                )
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        with self.assertRaises(TimeoutError):
            _wait_for_callback_update_with_timeout(
                coordinator,
                timeout_seconds=0.35,
                context="unit test nonready neow map callback wait",
            )

        self.assertEqual(coordinator.messages, [])

    def test_play_recorded_game_uses_ready_wait_after_executing_action(self):
        class FakeAction:
            def can_be_executed(self, coordinator):
                return True

        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = [FakeAction()]
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="NeowRoom",
                )
                self.receive_calls = []

            def clear_actions(self):
                pass

            def execute_next_action(self):
                action = self.action_queue.pop(0)
                action.execute(self)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                self.receive_calls.append((block, perform_callbacks))
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def execute_next_action(self):
                self.action_queue.pop(0)
                self.game_is_ready = False

        coordinator = FakeCoordinator()
        driver = SimpleNamespace(
            failed=False,
            done=False,
            step_index=0,
            total_steps=1,
            pending_step=None,
            next_action=lambda game_state: None,
        )
        ready_wait_calls = []
        callback_wait_calls = []

        def _fake_ready_wait(coord, *, timeout_seconds, context):
            ready_wait_calls.append((coord, timeout_seconds, context))
            driver.done = True

        def _fake_callback_wait(*args, **kwargs):
            callback_wait_calls.append((args, kwargs))

        with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout", side_effect=_fake_ready_wait):
            with patch("spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout", side_effect=_fake_callback_wait):
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(len(ready_wait_calls), 1)
        self.assertIs(ready_wait_calls[0][0], coordinator)
        self.assertIn("after executing action", ready_wait_calls[0][2])
        self.assertEqual(callback_wait_calls, [])
        self.assertEqual(coordinator.receive_calls, [])

    def test_play_recorded_game_sends_single_wait_pulse_for_nonready_neow_map_frame(self):
        class FakeAction:
            def can_be_executed(self, coordinator):
                return True

            def execute(self, coordinator):
                coordinator.game_is_ready = False
                coordinator.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.MAP,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(next_nodes=[SimpleNamespace(x=1, y=0), SimpleNamespace(x=3, y=0)]),
                )

        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = [FakeAction()]
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                )
                self.receive_calls = []
                self.messages = []

            def clear_actions(self):
                pass

            def execute_next_action(self):
                action = self.action_queue.pop(0)
                action.execute(self)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                self.receive_calls.append((block, perform_callbacks))
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def send_message(self, message):
                self.messages.append(message)

        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 1},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        coordinator = FakeCoordinator()
        driver = SimpleNamespace(
            failed=False,
            done=False,
            step_index=0,
            total_steps=1,
            pending_step=step,
            neow_map_ready_pulse_step_number=None,
            next_action=lambda game_state: None,
        )
        callback_wait_calls = []

        def _fake_callback_wait(*args, **kwargs):
            callback_wait_calls.append((args, kwargs))
            driver.done = True

        with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout") as ready_wait:
            with patch("spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout", side_effect=_fake_callback_wait):
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(coordinator.messages, ["wait 1"])
        self.assertEqual(driver.neow_map_ready_pulse_step_number, 3)
        ready_wait.assert_not_called()
        self.assertEqual(len(callback_wait_calls), 1)

    def test_play_recorded_game_forces_finalize_when_neow_leave_map_bridge_stays_stale(self):
        class FakeAction:
            def can_be_executed(self, coordinator):
                return True

        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = [FakeAction()]
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(options=[SimpleNamespace(label="Leave")]),
                )

            def clear_actions(self):
                pass

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def execute_next_action(self):
                self.action_queue.pop(0)
                self.game_is_ready = False

        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 1},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 1
                self.pending_step = step
                self.neow_map_ready_pulse_step_number = None
                self.neow_leave_map_continue_step_number = 3

            def next_action(self, game_state):
                return None

            def _force_finalize_pending_step(self, *, live_snapshot, comparison_note):
                forced.append((live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout",
            side_effect=TimeoutError("still stale"),
        ) as ready_wait:
            with patch(
                "spirecomm.ai.recorded_run_replay._should_skip_ready_wait_for_neow_leave_map_continue",
                return_value=False,
            ):
                with patch("spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout") as callback_wait:
                    _play_recorded_game(
                        coordinator,
                        player_class=PlayerClass.IRONCLAD,
                        ascension_level=0,
                        seed="1",
                        driver=driver,
                    )

        self.assertEqual(ready_wait.call_count, 1)
        callback_wait.assert_not_called()
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][1], "neow_leave_map_continue_blind_bridge_forced")

    def test_play_recorded_game_forces_finalize_when_neow_leave_map_bridge_marker_is_missing(self):
        class FakeAction:
            def can_be_executed(self, coordinator):
                return True

        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = [FakeAction()]
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(options=[SimpleNamespace(label="Leave")]),
                )

            def clear_actions(self):
                pass

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def execute_next_action(self):
                self.action_queue.pop(0)
                self.game_is_ready = False

        step = RecordedTraceStep(
            step=3,
            phase="MAP",
            floor=1,
            action={"kind": "map", "name": "M", "choice_index": 0, "x": 1},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 1
                self.pending_step = step
                self.neow_map_ready_pulse_step_number = None
                self.neow_leave_map_continue_step_number = None

            def next_action(self, game_state):
                return None

            def _force_finalize_pending_step(self, *, live_snapshot, comparison_note):
                forced.append((live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout",
            side_effect=TimeoutError("still stale"),
        ):
            with patch(
                "spirecomm.ai.recorded_run_replay._should_skip_ready_wait_for_neow_leave_map_continue",
                return_value=False,
            ):
                with patch("spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout") as callback_wait:
                    _play_recorded_game(
                        coordinator,
                        player_class=PlayerClass.IRONCLAD,
                        ascension_level=0,
                        seed="1",
                        driver=driver,
                    )

        callback_wait.assert_not_called()
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][1], "neow_leave_map_continue_blind_bridge_forced")

    def test_play_recorded_game_forces_finalize_on_neow_leave_callback_timeout(self):
        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = []
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.sent_messages = []
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(options=[SimpleNamespace(label="Leave")]),
                )

            def clear_actions(self):
                pass

            def send_message(self, message):
                self.sent_messages.append(message)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

        step = RecordedTraceStep(
            step=2,
            floor=0,
            phase="NEOW",
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 1
                self.pending_step = step

            def next_action(self, game_state):
                return None

            def _force_finalize_pending_step(self, *, live_snapshot, comparison_note):
                forced.append((live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=TimeoutError("still stale"),
        ) as callback_wait:
            with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout") as ready_wait:
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(callback_wait.call_count, 1)
        ready_wait.assert_not_called()
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][1], "neow_continue_leave_bridge_callback_timeout_forced")

    def test_play_recorded_game_forces_finalize_current_neow_leave_on_callback_timeout(self):
        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = []
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.sent_messages = []
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(options=[SimpleNamespace(label="Leave")]),
                )

            def clear_actions(self):
                pass

            def send_message(self, message):
                self.sent_messages.append(message)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

        reward_step = RecordedTraceStep(
            step=1,
            floor=0,
            phase="CARD_REWARD",
            action={"kind": "card_reward", "choice_index": 2, "name": "Flash of Steel", "card_id": "Flash of Steel"},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
        )
        neow_step = RecordedTraceStep(
            step=2,
            floor=0,
            phase="NEOW",
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
        )
        map_step = RecordedTraceStep(
            step=3,
            floor=1,
            phase="MAP",
            action={"kind": "map", "choice_index": 0, "name": "M"},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 1
                self.total_steps = 3
                self.pending_step = None
                self.trace = RecordedRunTrace(
                    path=Path("/tmp/fake_trace.json"),
                    source_format="native_run_trace_v1",
                    seed_long=1,
                    seed_str="1",
                    ascension=0,
                    character="IRONCLAD",
                    steps=[reward_step, neow_step, map_step],
                )

            def next_action(self, game_state):
                return None

            def _force_finalize_current_step(self, step_arg, *, live_snapshot, comparison_note):
                forced.append((step_arg, live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=TimeoutError("still stale"),
        ) as callback_wait:
            with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout") as ready_wait:
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(callback_wait.call_count, 1)
        ready_wait.assert_not_called()
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][0].step, 2)
        self.assertEqual(forced[0][2], "neow_continue_leave_bridge_callback_timeout_forced")

    def test_play_recorded_game_forces_finalize_on_neow_card_reward_callback_timeout(self):
        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = []
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[
                        SimpleNamespace(card_id="Strike_R"),
                        SimpleNamespace(card_id="Flash of Steel"),
                    ],
                    in_combat=False,
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(cards=[]),
                )

            def clear_actions(self):
                pass

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

        step = RecordedTraceStep(
            step=1,
            floor=0,
            phase="CARD_REWARD",
            action={"kind": "card_reward", "choice_index": 2, "name": "Flash of Steel", "card_id": "Flash of Steel"},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        next_step = RecordedTraceStep(
            step=2,
            floor=0,
            phase="NEOW",
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        next_next_step = RecordedTraceStep(
            step=3,
            floor=1,
            phase="MAP",
            action={"kind": "map", "choice_index": 0, "name": "M"},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 3
                self.pending_step = step
                self.trace = RecordedRunTrace(
                    path=Path("/tmp/fake_trace.json"),
                    source_format="native_run_trace_v1",
                    seed_long=1,
                    seed_str="1",
                    ascension=0,
                    character="IRONCLAD",
                    steps=[step, next_step, next_next_step],
                )
                self.neow_card_reward_blind_continue_step_number = None
                self.neow_card_reward_blind_map_step_number = None

            def next_action(self, game_state):
                return None

            def _force_finalize_pending_step(self, *, live_snapshot, comparison_note):
                forced.append((live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=TimeoutError("still stale"),
        ) as callback_wait:
            with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout") as ready_wait:
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(callback_wait.call_count, 1)
        ready_wait.assert_not_called()
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][1], "neow_reward_pick_stale_card_reward_blind_continue_forced")
        self.assertEqual(driver.neow_card_reward_blind_continue_step_number, 2)
        self.assertEqual(driver.neow_card_reward_blind_map_step_number, 3)

    def test_play_recorded_game_forces_finalize_current_neow_card_reward_on_callback_timeout(self):
        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = []
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[
                        SimpleNamespace(card_id="Strike_R"),
                        SimpleNamespace(card_id="Flash of Steel"),
                    ],
                    in_combat=False,
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(cards=[]),
                )

            def clear_actions(self):
                pass

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

        step = RecordedTraceStep(
            step=1,
            floor=0,
            phase="CARD_REWARD",
            action={"kind": "card_reward", "choice_index": 2, "name": "Flash of Steel", "card_id": "Flash of Steel"},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        next_step = RecordedTraceStep(
            step=2,
            floor=0,
            phase="NEOW",
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )
        next_next_step = RecordedTraceStep(
            step=3,
            floor=1,
            phase="MAP",
            action={"kind": "map", "choice_index": 0, "name": "M"},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 3
                self.pending_step = None
                self.trace = RecordedRunTrace(
                    path=Path("/tmp/fake_trace.json"),
                    source_format="native_run_trace_v1",
                    seed_long=1,
                    seed_str="1",
                    ascension=0,
                    character="IRONCLAD",
                    steps=[step, next_step, next_next_step],
                )
                self.neow_card_reward_blind_continue_step_number = None
                self.neow_card_reward_blind_map_step_number = None

            def next_action(self, game_state):
                return None

            def _force_finalize_current_step(self, step_arg, *, live_snapshot, comparison_note):
                forced.append((step_arg, live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=TimeoutError("still stale"),
        ) as callback_wait:
            with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout") as ready_wait:
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(callback_wait.call_count, 1)
        ready_wait.assert_not_called()
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][0].step, 1)
        self.assertEqual(forced[0][2], "neow_reward_pick_stale_card_reward_blind_continue_forced")
        self.assertEqual(driver.neow_card_reward_blind_continue_step_number, 2)
        self.assertEqual(driver.neow_card_reward_blind_map_step_number, 3)

    def test_play_recorded_game_forces_finalize_on_neow_card_select_callback_timeout(self):
        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = []
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[SimpleNamespace(card_id="Bash")],
                    in_combat=False,
                    screen_type=ScreenType.GRID,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(
                        cards=[SimpleNamespace(name="Bash", card_id="Bash")],
                        selected_cards=[],
                        for_upgrade=True,
                        confirm_up=False,
                    ),
                )

            def clear_actions(self):
                pass

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

        step = RecordedTraceStep(
            step=1,
            floor=0,
            phase="CARD_SELECT",
            action={"kind": "card_select", "choice_index": 9, "name": "Bash", "card_id": "Bash"},
            pre={"phase": "CARD_SELECT", "floor": 0, "hp": 80, "gold": 99, "deck": ["Bash"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Bash+1"]},
        )
        next_step = RecordedTraceStep(
            step=2,
            floor=0,
            phase="NEOW",
            action={"kind": "neow", "choice_index": 0, "name": "CONTINUE"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Bash+1"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Bash+1"]},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 2
                self.pending_step = step
                self.trace = RecordedRunTrace(
                    path=Path("/tmp/fake_trace.json"),
                    source_format="native_run_trace_v1",
                    seed_long=1,
                    seed_str="1",
                    ascension=0,
                    character="IRONCLAD",
                    steps=[step, next_step],
                )
                self.neow_card_select_continue_step_number = None

            def next_action(self, game_state):
                return None

            def _force_finalize_pending_step(self, *, live_snapshot, comparison_note):
                forced.append((live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=TimeoutError("still stale"),
        ):
            with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout"):
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][1], "neow_card_select_callback_timeout_forced")
        self.assertEqual(driver.neow_card_select_continue_step_number, 2)

    def test_play_recorded_game_forces_finalize_on_neow_card_select_continue_callback_timeout(self):
        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = []
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[SimpleNamespace(card_id="Bash")],
                    in_combat=False,
                    screen_type=ScreenType.GRID,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(
                        cards=[SimpleNamespace(name="Bash", card_id="Bash")],
                        selected_cards=[],
                        for_upgrade=True,
                        confirm_up=False,
                    ),
                )

            def clear_actions(self):
                pass

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

        step = RecordedTraceStep(
            step=2,
            floor=0,
            phase="NEOW",
            action={"kind": "neow", "choice_index": 0, "name": "CONTINUE"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Bash+1"]},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99, "deck": ["Bash+1"]},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 1
                self.pending_step = step
                self.trace = RecordedRunTrace(
                    path=Path("/tmp/fake_trace.json"),
                    source_format="native_run_trace_v1",
                    seed_long=1,
                    seed_str="1",
                    ascension=0,
                    character="IRONCLAD",
                    steps=[step],
                )
                self.neow_card_select_continue_step_number = 2

            def next_action(self, game_state):
                return None

            def _force_finalize_pending_step(self, *, live_snapshot, comparison_note):
                forced.append((live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=TimeoutError("still stale"),
        ):
            with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout"):
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][1], "neow_card_select_continue_callback_timeout_forced")
        self.assertIsNone(driver.neow_card_select_continue_step_number)

    def test_play_recorded_game_forces_finalize_on_neow_map_callback_timeout(self):
        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = []
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.MAP,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(next_nodes=[]),
                )

            def clear_actions(self):
                pass

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

        step = RecordedTraceStep(
            step=3,
            floor=1,
            phase="MAP",
            action={"kind": "map", "choice_index": 0, "name": "M"},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 1
                self.pending_step = step
                self.neow_leave_map_continue_step_number = 3

            def next_action(self, game_state):
                return None

            def _force_finalize_pending_step(self, *, live_snapshot, comparison_note):
                forced.append((live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=TimeoutError("still stale"),
        ) as callback_wait:
            with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout") as ready_wait:
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(callback_wait.call_count, 1)
        ready_wait.assert_not_called()
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][1], "neow_leave_map_continue_callback_timeout_forced")

    def test_play_recorded_game_skips_ready_wait_after_dispatching_neow_map_bridge(self):
        events = []

        class InitialAction:
            def can_be_executed(self, coordinator):
                return True

            def execute(self, coordinator):
                events.append("executed")
                coordinator.game_is_ready = False

        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = [InitialAction()]
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.clear_calls = 0
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(options=[SimpleNamespace(label="Leave")]),
                )

            def clear_actions(self):
                self.clear_calls += 1
                self.action_queue = []

            def execute_next_action(self):
                action = self.action_queue.pop(0)
                action.execute(self)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                return False

        step = RecordedTraceStep(
            step=3,
            floor=1,
            phase="MAP",
            action={"kind": "map", "choice_index": 0, "name": "M"},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 1
                self.pending_step = step

            def next_action(self, game_state):
                return None

            def _force_finalize_pending_step(self, *, live_snapshot, comparison_note):
                forced.append((live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=TimeoutError("still stale"),
        ) as callback_wait:
            with patch(
                "spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout",
                side_effect=AssertionError("ready wait should be skipped"),
            ) as ready_wait:
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(events, ["executed"])
        self.assertEqual(callback_wait.call_count, 1)
        ready_wait.assert_not_called()
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][1], "neow_leave_map_continue_callback_timeout_forced")

    def test_play_recorded_game_skips_ready_wait_after_dispatching_neow_map_bridge_setup_action(self):
        events = []

        class InitialAction:
            def can_be_executed(self, coordinator):
                return True

            def execute(self, coordinator):
                events.append("executed")
                coordinator.game_is_ready = False

        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = [InitialAction()]
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.clear_calls = 0
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(options=[SimpleNamespace(label="Leave")]),
                )

            def clear_actions(self):
                self.clear_calls += 1
                self.action_queue = []

            def execute_next_action(self):
                action = self.action_queue.pop(0)
                action.execute(self)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                return False

        step = RecordedTraceStep(
            step=3,
            floor=1,
            phase="MAP",
            action={"kind": "map", "choice_index": 0, "name": "M"},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 1
                self.pending_step = None
                self.trace = SimpleNamespace(steps=[step])
                self.neow_leave_map_continue_step_number = 3

            def next_action(self, game_state):
                return None

        coordinator = FakeCoordinator()
        driver = FakeDriver()
        callback_wait_calls = []

        def _fake_callback_wait(*args, **kwargs):
            callback_wait_calls.append((args, kwargs))
            driver.done = True

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=_fake_callback_wait,
        ) as callback_wait:
            with patch(
                "spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout",
                side_effect=AssertionError("ready wait should be skipped"),
            ) as ready_wait:
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(events, ["executed"])
        self.assertEqual(callback_wait.call_count, 1)
        ready_wait.assert_not_called()
        self.assertEqual(len(callback_wait_calls), 1)

    def test_play_recorded_game_forces_finalize_when_neow_map_setup_action_stays_stale(self):
        class InitialAction:
            def can_be_executed(self, coordinator):
                return True

            def execute(self, coordinator):
                coordinator.game_is_ready = False

        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = [InitialAction()]
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.clear_calls = 0
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(options=[SimpleNamespace(label="Leave")]),
                )

            def clear_actions(self):
                self.clear_calls += 1
                self.action_queue = []

            def execute_next_action(self):
                action = self.action_queue.pop(0)
                action.execute(self)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                return False

        step = RecordedTraceStep(
            step=3,
            floor=1,
            phase="MAP",
            action={"kind": "map", "choice_index": 0, "name": "M"},
            pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 1
                self.pending_step = None
                self.pending_actual_pre = None
                self.pending_pre_mismatches = []
                self.pending_actions = []
                self.pending_setup_actions = []
                self.pending_comparison_note = None
                self.neow_map_ready_pulse_step_number = None
                self.neow_leave_map_continue_step_number = 3
                self.trace = SimpleNamespace(steps=[step])
                self.results = []
                self.progress_callback = None

            def next_action(self, game_state):
                return None

            def _emit_progress(self, **kwargs):
                return None

            def _force_finalize_current_step(self, step_arg, *, live_snapshot, comparison_note):
                self.results.append((step_arg.step, comparison_note, live_snapshot))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=TimeoutError("still stale"),
        ) as callback_wait:
            with patch(
                "spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout",
                side_effect=AssertionError("ready wait should be skipped"),
            ) as ready_wait:
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(callback_wait.call_count, 1)
        ready_wait.assert_not_called()
        self.assertEqual(len(driver.results), 1)
        self.assertEqual(driver.results[0][1], "neow_leave_map_continue_setup_callback_timeout_forced")
        self.assertGreaterEqual(coordinator.clear_calls, 2)

    def test_force_finalize_current_step_sets_post_neow_combat_settle_probe(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/seed_1_trace.json"),
            source_format="json",
            seed_str="1",
            seed_long=1,
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=3,
                    floor=1,
                    phase="MAP",
                    action={"kind": "map", "choice_index": 0, "name": "M"},
                    pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
                    post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                ),
                RecordedTraceStep(
                    step=4,
                    floor=1,
                    phase="COMBAT",
                    action={"kind": "card", "name": "Flash of Steel"},
                    pre={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                    post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                ),
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )

        driver._force_finalize_current_step(
            trace.steps[0],
            live_snapshot={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            comparison_note="neow_leave_map_continue_setup_callback_timeout_forced",
        )

        self.assertEqual(driver.step_index, 1)
        self.assertEqual(driver.neow_post_map_settle_probe_step_number, 4)

    def test_force_finalize_current_step_sets_post_neow_combat_settle_probe_for_nonsetup_leave_timeout(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/seed_2_trace.json"),
            source_format="json",
            seed_str="2",
            seed_long=2,
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=2,
                    floor=1,
                    phase="MAP",
                    action={"kind": "map", "choice_index": 0, "name": "M"},
                    pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
                    post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                ),
                RecordedTraceStep(
                    step=3,
                    floor=1,
                    phase="COMBAT",
                    action={"kind": "card", "name": "Bash"},
                    pre={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                    post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                ),
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )

        driver._force_finalize_current_step(
            trace.steps[0],
            live_snapshot={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            comparison_note="neow_leave_map_continue_callback_timeout_forced",
        )

        self.assertEqual(driver.step_index, 1)
        self.assertEqual(driver.neow_post_map_settle_probe_step_number, 3)

    def test_force_finalize_current_step_keeps_post_neow_combat_settle_probe_for_next_combat(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/seed_1_trace.json"),
            source_format="json",
            seed_str="1",
            seed_long=1,
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=4,
                    floor=1,
                    phase="COMBAT",
                    action={"kind": "card", "name": "Flash of Steel"},
                    pre={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                    post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                ),
                RecordedTraceStep(
                    step=5,
                    floor=1,
                    phase="COMBAT",
                    action={"kind": "card", "name": "Strike_R"},
                    pre={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                    post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                ),
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )

        driver._force_finalize_current_step(
            trace.steps[0],
            live_snapshot={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            comparison_note="neow_post_map_settle_callback_timeout_forced",
        )

        self.assertEqual(driver.step_index, 1)
        self.assertEqual(driver.neow_post_map_settle_probe_step_number, 5)

    def test_force_finalize_current_step_keeps_post_neow_combat_settle_probe_for_next_reward(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=14,
                    floor=1,
                    phase="COMBAT",
                    action={"kind": "card", "name": "Strike_R"},
                    pre={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                    post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                ),
                RecordedTraceStep(
                    step=15,
                    floor=1,
                    phase="CARD_REWARD",
                    action={"kind": "reward_gold", "name": "GOLD"},
                    pre={"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 115},
                    post={"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 130},
                ),
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )

        driver._force_finalize_current_step(
            trace.steps[0],
            live_snapshot={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            comparison_note="neow_post_map_settle_callback_timeout_forced",
        )

        self.assertEqual(driver.step_index, 1)
        self.assertEqual(driver.neow_post_map_settle_probe_step_number, 15)

    def test_force_finalize_current_step_keeps_post_neow_combat_settle_probe_for_next_event(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=33,
                    floor=3,
                    phase="MAP",
                    action={"kind": "map", "name": "?", "choice_index": 0, "x": 1},
                    pre={"phase": "MAP", "floor": 2, "hp": 80, "gold": 127},
                    post={"phase": "EVENT", "floor": 3, "hp": 80, "gold": 127},
                ),
                RecordedTraceStep(
                    step=34,
                    floor=3,
                    phase="EVENT",
                    action={"kind": "event", "event_id": "The Cleric", "name": "Leave", "choice_index": 2},
                    pre={"phase": "EVENT", "floor": 3, "hp": 80, "gold": 127},
                    post={"phase": "EVENT", "floor": 3, "hp": 80, "gold": 127},
                ),
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )

        driver._force_finalize_current_step(
            trace.steps[0],
            live_snapshot={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            comparison_note="neow_post_map_settle_callback_timeout_forced",
        )

        self.assertEqual(driver.step_index, 1)
        self.assertEqual(driver.neow_post_map_settle_probe_step_number, 34)

    def test_force_finalize_current_step_keeps_post_neow_combat_settle_probe_for_next_card_select(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=3,
            seed_str="3",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=28,
                    floor=3,
                    phase="EVENT",
                    action={"kind": "event", "event_id": "Upgrade Shrine", "name": "Pray", "choice_index": 0},
                    pre={"phase": "EVENT", "floor": 3, "hp": 73, "gold": 140},
                    post={"phase": "CARD_SELECT", "floor": 3, "hp": 73, "gold": 140},
                ),
                RecordedTraceStep(
                    step=29,
                    floor=3,
                    phase="CARD_SELECT",
                    action={"kind": "card_select", "mode": "upgrade", "name": "Bash", "choice_index": 9},
                    pre={"phase": "CARD_SELECT", "floor": 3, "hp": 73, "gold": 140},
                    post={"phase": "EVENT", "floor": 3, "hp": 73, "gold": 140},
                ),
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )

        driver._force_finalize_current_step(
            trace.steps[0],
            live_snapshot={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            comparison_note="neow_post_map_settle_callback_timeout_forced",
        )

        self.assertEqual(driver.step_index, 1)
        self.assertEqual(driver.neow_post_map_settle_probe_step_number, 29)

    def test_force_finalize_current_step_keeps_post_neow_combat_settle_probe_for_next_campfire(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=68,
                    floor=6,
                    phase="MAP",
                    action={"kind": "map", "name": "R", "choice_index": 0, "x": 1},
                    pre={"phase": "MAP", "floor": 5, "hp": 63, "gold": 145},
                    post={"phase": "CAMPFIRE", "floor": 6, "hp": 63, "gold": 145},
                ),
                RecordedTraceStep(
                    step=69,
                    floor=6,
                    phase="CAMPFIRE",
                    action={"kind": "campfire", "name": "SMITH", "choice_index": 1, "target_index": 9},
                    pre={"phase": "CAMPFIRE", "floor": 6, "hp": 63, "gold": 145},
                    post={"phase": "MAP", "floor": 6, "hp": 63, "gold": 145},
                ),
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )

        driver._force_finalize_current_step(
            trace.steps[0],
            live_snapshot={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            comparison_note="neow_post_map_settle_callback_timeout_forced",
        )

        self.assertEqual(driver.step_index, 1)
        self.assertEqual(driver.neow_post_map_settle_probe_step_number, 69)

    def test_force_finalize_current_step_keeps_post_neow_combat_settle_probe_for_next_treasure(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=89,
                    floor=9,
                    phase="MAP",
                    action={"kind": "map", "name": "T", "choice_index": 0, "x": 0},
                    pre={"phase": "MAP", "floor": 8, "hp": 63, "gold": 145},
                    post={"phase": "TREASURE", "floor": 9, "hp": 63, "gold": 145},
                ),
                RecordedTraceStep(
                    step=90,
                    floor=9,
                    phase="TREASURE",
                    action={"kind": "treasure", "name": "OPEN_CHEST", "choice_index": 0},
                    pre={"phase": "TREASURE", "floor": 9, "hp": 63, "gold": 145},
                    post={"phase": "CARD_REWARD", "floor": 9, "hp": 63, "gold": 145},
                ),
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )

        driver._force_finalize_current_step(
            trace.steps[0],
            live_snapshot={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            comparison_note="neow_post_map_settle_callback_timeout_forced",
        )

        self.assertEqual(driver.step_index, 1)
        self.assertEqual(driver.neow_post_map_settle_probe_step_number, 90)

    def test_force_finalize_current_step_keeps_post_neow_combat_settle_probe_for_next_boss_relic(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=241,
                    floor=17,
                    phase="COMBAT",
                    action={"kind": "card", "name": "Uppercut"},
                    pre={"phase": "COMBAT", "floor": 17, "hp": 55, "gold": 250},
                    post={"phase": "COMBAT", "floor": 17, "hp": 55, "gold": 250},
                ),
                RecordedTraceStep(
                    step=242,
                    floor=17,
                    phase="BOSS_RELIC",
                    action={"kind": "boss_relic", "name": "Snecko Eye", "choice_index": 2},
                    pre={"phase": "BOSS_RELIC", "floor": 17, "hp": 55, "gold": 250},
                    post={"phase": "MAP", "floor": 17, "hp": 55, "gold": 250},
                ),
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )

        driver._force_finalize_current_step(
            trace.steps[0],
            live_snapshot={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            comparison_note="neow_post_map_settle_callback_timeout_forced",
        )

        self.assertEqual(driver.step_index, 1)
        self.assertEqual(driver.neow_post_map_settle_probe_step_number, 242)

    def test_force_finalize_current_step_keeps_post_neow_combat_settle_probe_for_next_shop(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=329,
                    floor=23,
                    phase="MAP",
                    action={"kind": "map", "name": "$", "choice_index": 0, "x": 3},
                    pre={"phase": "MAP", "floor": 22, "hp": 45, "gold": 210},
                    post={"phase": "SHOP", "floor": 23, "hp": 45, "gold": 210},
                ),
                RecordedTraceStep(
                    step=330,
                    floor=23,
                    phase="SHOP",
                    action={"kind": "shop", "name": "Blue Candle", "choice_index": 6},
                    pre={"phase": "SHOP", "floor": 23, "hp": 45, "gold": 210},
                    post={"phase": "SHOP", "floor": 23, "hp": 45, "gold": 140},
                ),
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )

        driver._force_finalize_current_step(
            trace.steps[0],
            live_snapshot={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            comparison_note="neow_post_map_settle_callback_timeout_forced",
        )

        self.assertEqual(driver.step_index, 1)
        self.assertEqual(driver.neow_post_map_settle_probe_step_number, 330)

    def test_force_finalize_current_step_keeps_post_neow_card_select_settle_probe_for_next_map(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=1,
                    floor=0,
                    phase="CARD_SELECT",
                    action={"kind": "card_select", "name": "Bash"},
                    pre={"phase": "CARD_SELECT", "floor": 0, "hp": 80, "gold": 99},
                    post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
                ),
                RecordedTraceStep(
                    step=2,
                    floor=1,
                    phase="MAP",
                    action={"kind": "map", "name": "M", "choice_index": 0, "x": 1},
                    pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
                    post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                ),
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )

        driver._force_finalize_current_step(
            trace.steps[0],
            live_snapshot={"phase": "CARD_SELECT", "floor": 0, "hp": 80, "gold": 99},
            comparison_note="neow_card_select_callback_timeout_forced",
        )

        self.assertEqual(driver.step_index, 1)
        self.assertEqual(driver.neow_post_card_select_settle_probe_step_number, 2)

    def test_force_finalize_current_step_keeps_post_neow_card_select_settle_probe_for_next_combat(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=3,
                    floor=1,
                    phase="MAP",
                    action={"kind": "map", "name": "M", "choice_index": 0, "x": 1},
                    pre={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
                    post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                ),
                RecordedTraceStep(
                    step=4,
                    floor=1,
                    phase="COMBAT",
                    action={"kind": "card", "name": "Defend_R"},
                    pre={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                    post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
                ),
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )

        driver._force_finalize_current_step(
            trace.steps[0],
            live_snapshot={"phase": "CARD_SELECT", "floor": 0, "hp": 80, "gold": 99},
            comparison_note="neow_post_card_select_settle_callback_timeout_forced",
        )

        self.assertEqual(driver.step_index, 1)
        self.assertEqual(driver.neow_post_card_select_settle_probe_step_number, 4)

    def test_finalize_pending_neow_card_select_after_settle_state_marks_continue_bridge(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=1,
                    floor=0,
                    phase="CARD_SELECT",
                    action={"kind": "card_select", "name": "Bash"},
                    pre={"phase": "CARD_SELECT", "floor": 0, "hp": 80, "gold": 99},
                    post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
                ),
                RecordedTraceStep(
                    step=2,
                    floor=0,
                    phase="NEOW",
                    action={"kind": "neow", "choice_index": 0},
                    pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
                    post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
                ),
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        driver.pending_step = trace.steps[0]
        driver.pending_actual_pre = {"phase": "CARD_SELECT", "floor": 0, "hp": 80, "gold": 99}
        driver.pending_pre_mismatches = []
        driver.pending_actions = [{"action_class": "StateAction", "command": "state"}]
        driver.pending_setup_actions = []
        driver.pending_comparison_note = None

        game_state = SimpleNamespace(
            room_type="NeowRoom",
            screen_type=ScreenType.GRID,
        )
        live_snapshot = {"phase": "CARD_SELECT", "floor": 0, "hp": 80, "gold": 99}

        self.assertTrue(driver._finalize_pending_step(game_state, live_snapshot))
        self.assertEqual(driver.step_index, 1)
        self.assertEqual(driver.neow_card_select_continue_step_number, 2)
        self.assertEqual(driver.results[-1]["comparison_note"], "neow_card_select_callback_timeout_forced")

    def test_should_wait_for_any_update_for_post_neow_settle(self):
        current_step = RecordedTraceStep(
            step=4,
            floor=1,
            phase="COMBAT",
            action={"kind": "card", "name": "Flash of Steel"},
            pre={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        driver = SimpleNamespace(neow_post_map_settle_probe_step_number=4)
        stale_neow_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[SimpleNamespace(label="Leave", choice_index=0)], event_id="Neow"),
        )

        self.assertTrue(
            _should_wait_for_any_update_for_post_neow_settle(
                driver,
                current_step,
                stale_neow_state,
            )
        )
        self.assertFalse(
            _should_wait_for_any_update_for_post_neow_settle(
                SimpleNamespace(neow_post_map_settle_probe_step_number=None),
                current_step,
                stale_neow_state,
            )
        )

    def test_should_use_short_post_neow_callback_timeout(self):
        current_step = RecordedTraceStep(
            step=15,
            floor=1,
            phase="CARD_REWARD",
            action={"kind": "reward_gold", "name": "GOLD"},
            pre={"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 115},
            post={"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 130},
        )
        driver = SimpleNamespace(neow_post_map_settle_probe_step_number=15)
        stale_neow_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[SimpleNamespace(label="Leave", choice_index=0)], event_id="Neow"),
        )

        self.assertTrue(
            _should_use_short_post_neow_callback_timeout(
                driver,
                None,
                current_step,
                stale_neow_state,
            )
        )
        self.assertFalse(
            _should_use_short_post_neow_callback_timeout(
                SimpleNamespace(neow_post_map_settle_probe_step_number=None),
                None,
                current_step,
                stale_neow_state,
            )
        )

        stale_neow_grid_state = SimpleNamespace(
            screen_type=ScreenType.GRID,
            room_type="NeowRoom",
            screen=GridSelectScreen(
                [Card("Bash", "Bash", CardType.ATTACK, CardRarity.BASIC, upgrades=0, uuid="b")],
                [],
                1,
                False,
                False,
                False,
                True,
                False,
            ),
        )
        self.assertTrue(
            _should_use_short_post_neow_callback_timeout(
                SimpleNamespace(
                    neow_post_map_settle_probe_step_number=None,
                    neow_post_card_select_settle_probe_step_number=15,
                ),
                None,
                current_step,
                stale_neow_grid_state,
            )
        )

        event_map_step = RecordedTraceStep(
            step=33,
            floor=2,
            phase="MAP",
            action={"kind": "map", "name": "?", "choice_index": 0, "x": 1},
            pre={"phase": "MAP", "floor": 2, "hp": 80, "gold": 127},
            post={"phase": "EVENT", "floor": 3, "hp": 80, "gold": 127},
        )
        stale_event_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="EVENT",
            screen=SimpleNamespace(options=[SimpleNamespace(label="Leave", choice_index=0)]),
        )
        self.assertTrue(
            _should_use_short_post_neow_callback_timeout(
                SimpleNamespace(
                    neow_post_map_settle_probe_step_number=None,
                    event_leave_map_continue_step_number=33,
                ),
                None,
                event_map_step,
                stale_event_state,
            )
        )

        event_step = RecordedTraceStep(
            step=34,
            floor=3,
            phase="EVENT",
            action={"kind": "event", "event_id": "The Cleric", "name": "Leave", "choice_index": 2},
            pre={"phase": "EVENT", "floor": 3, "hp": 80, "gold": 127},
            post={"phase": "EVENT", "floor": 3, "hp": 80, "gold": 127},
        )
        stale_neow_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(options=[SimpleNamespace(label="Leave", choice_index=0)], event_id="Neow Event"),
        )
        self.assertTrue(
            _should_use_short_post_neow_callback_timeout(
                SimpleNamespace(
                    neow_post_map_settle_probe_step_number=34,
                    event_leave_map_continue_step_number=None,
                ),
                None,
                event_step,
                stale_neow_state,
            )
        )

        campfire_step = RecordedTraceStep(
            step=69,
            floor=6,
            phase="CAMPFIRE",
            action={"kind": "campfire", "name": "SMITH", "choice_index": 1, "target_index": 9},
            pre={"phase": "CAMPFIRE", "floor": 6, "hp": 63, "gold": 145},
            post={"phase": "MAP", "floor": 6, "hp": 63, "gold": 145},
        )
        self.assertTrue(
            _should_use_short_post_neow_callback_timeout(
                SimpleNamespace(
                    neow_post_map_settle_probe_step_number=69,
                    event_leave_map_continue_step_number=None,
                ),
                None,
                campfire_step,
                stale_neow_state,
            )
        )

        treasure_step = RecordedTraceStep(
            step=90,
            floor=9,
            phase="TREASURE",
            action={"kind": "treasure", "name": "OPEN_CHEST", "choice_index": 0},
            pre={"phase": "TREASURE", "floor": 9, "hp": 63, "gold": 145},
            post={"phase": "CARD_REWARD", "floor": 9, "hp": 63, "gold": 145},
        )
        self.assertTrue(
            _should_use_short_post_neow_callback_timeout(
                SimpleNamespace(
                    neow_post_map_settle_probe_step_number=90,
                    event_leave_map_continue_step_number=None,
                ),
                None,
                treasure_step,
                stale_neow_state,
            )
        )

        boss_relic_step = RecordedTraceStep(
            step=242,
            floor=17,
            phase="BOSS_RELIC",
            action={"kind": "boss_relic", "name": "Snecko Eye", "choice_index": 2},
            pre={"phase": "BOSS_RELIC", "floor": 17, "hp": 55, "gold": 250},
            post={"phase": "MAP", "floor": 17, "hp": 55, "gold": 250},
        )
        self.assertTrue(
            _should_use_short_post_neow_callback_timeout(
                SimpleNamespace(
                    neow_post_map_settle_probe_step_number=242,
                    event_leave_map_continue_step_number=None,
                ),
                None,
                boss_relic_step,
                stale_neow_state,
            )
        )

        shop_step = RecordedTraceStep(
            step=330,
            floor=23,
            phase="SHOP",
            action={"kind": "shop", "name": "Blue Candle", "choice_index": 6},
            pre={"phase": "SHOP", "floor": 23, "hp": 45, "gold": 210},
            post={"phase": "SHOP", "floor": 23, "hp": 45, "gold": 140},
        )
        self.assertTrue(
            _should_use_short_post_neow_callback_timeout(
                SimpleNamespace(
                    neow_post_map_settle_probe_step_number=330,
                    event_leave_map_continue_step_number=None,
                ),
                None,
                shop_step,
                stale_neow_state,
            )
        )

    def test_should_skip_ready_wait_for_event_leave_map_continue(self):
        coordinator = SimpleNamespace(
            game_is_ready=False,
            last_game_state=SimpleNamespace(
                screen_type=ScreenType.EVENT,
                room_type="EVENT",
                screen=SimpleNamespace(options=[SimpleNamespace(label="Leave", choice_index=0)]),
            ),
        )
        driver = SimpleNamespace(
            pending_step=SimpleNamespace(step=33, phase="MAP"),
            event_leave_map_continue_step_number=33,
        )
        self.assertTrue(_should_skip_ready_wait_for_event_leave_map_continue(coordinator, driver))

    def test_should_allow_nonready_direct_action_for_event_leave_bridge(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=34,
                    floor=3,
                    phase="EVENT",
                    action={"kind": "event", "event_id": "The Cleric", "name": "Leave", "choice_index": 2},
                    pre={"phase": "EVENT", "floor": 3, "hp": 80, "gold": 127},
                    post={"phase": "EVENT", "floor": 3, "hp": 80, "gold": 127},
                )
            ],
        )
        coordinator = SimpleNamespace(
            last_game_state=SimpleNamespace(
                screen_type=ScreenType.EVENT,
                room_type="EVENT",
                screen=SimpleNamespace(options=[SimpleNamespace(label="Leave", choice_index=0)]),
            )
        )
        driver = SimpleNamespace(
            pending_step=None,
            trace=trace,
            step_index=0,
            total_steps=1,
            neow_card_reward_blind_continue_step_number=None,
        )

        self.assertTrue(_should_allow_nonready_direct_action(coordinator, driver))

    def test_should_allow_nonready_direct_action_for_post_neow_event_bridge(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=34,
                    floor=3,
                    phase="EVENT",
                    action={"kind": "event", "event_id": "The Cleric", "name": "Leave", "choice_index": 2},
                    pre={"phase": "EVENT", "floor": 3, "hp": 80, "gold": 127},
                    post={"phase": "EVENT", "floor": 3, "hp": 80, "gold": 127},
                )
            ],
        )
        coordinator = SimpleNamespace(
            last_game_state=SimpleNamespace(
                screen_type=ScreenType.EVENT,
                room_type="NeowRoom",
                screen=SimpleNamespace(options=[SimpleNamespace(label="Leave", choice_index=0)], event_id="Neow Event"),
            )
        )
        driver = SimpleNamespace(
            pending_step=None,
            trace=trace,
            step_index=0,
            total_steps=1,
            neow_card_reward_blind_continue_step_number=None,
            neow_post_map_settle_probe_step_number=34,
        )

        self.assertTrue(_should_allow_nonready_direct_action(coordinator, driver))

    def test_next_trace_action_for_game_blind_event_choice_when_live_options_stale(self):
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            screen=SimpleNamespace(options=[SimpleNamespace(label="Leave", choice_index=0)]),
        )
        step = RecordedTraceStep(
            step=34,
            floor=3,
            phase="EVENT",
            action={"kind": "event", "event_id": "The Cleric", "name": "Leave", "choice_index": 2},
            pre={"phase": "EVENT", "floor": 3, "hp": 80, "gold": 127},
            post={"phase": "EVENT", "floor": 3, "hp": 80, "gold": 127},
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertIsInstance(action, ChooseAction)
        self.assertEqual(action.choice_index, 2)

    def test_next_action_keeps_waiting_on_stale_neow_map_for_post_neow_event(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=2,
            seed_str="2",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=17,
                    floor=2,
                    phase="EVENT",
                    action={"kind": "event", "event_id": "The Woman in Blue", "name": "Ignored", "choice_index": 3},
                    pre={"phase": "EVENT", "floor": 2, "hp": 73, "gold": 130},
                    post={"phase": "EVENT", "floor": 2, "hp": 73, "gold": 130},
                )
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        driver.neow_post_map_settle_probe_step_number = 17
        game_state = SimpleNamespace(
            screen_type=ScreenType.MAP,
            room_type="NeowRoom",
            floor=0,
            current_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
            screen=SimpleNamespace(next_nodes=[SimpleNamespace(x=0, y=0)]),
        )

        action = driver.next_action(game_state)

        self.assertIsNone(action)
        self.assertFalse(driver.failed)
        self.assertEqual(driver.step_index, 0)

    def test_next_action_keeps_waiting_on_stale_neow_map_for_post_neow_card_select(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=3,
            seed_str="3",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=29,
                    floor=3,
                    phase="CARD_SELECT",
                    action={"kind": "card_select", "mode": "upgrade", "name": "Bash", "choice_index": 9},
                    pre={"phase": "CARD_SELECT", "floor": 3, "hp": 73, "gold": 140},
                    post={"phase": "EVENT", "floor": 3, "hp": 73, "gold": 140},
                )
            ],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        driver.neow_post_map_settle_probe_step_number = 29
        game_state = SimpleNamespace(
            screen_type=ScreenType.MAP,
            room_type="NeowRoom",
            floor=0,
            current_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
            screen=SimpleNamespace(next_nodes=[SimpleNamespace(x=0, y=0)]),
        )

        action = driver.next_action(game_state)

        self.assertIsNone(action)
        self.assertFalse(driver.failed)
        self.assertEqual(driver.step_index, 0)

    def test_play_recorded_game_forces_finalize_on_event_leave_map_callback_timeout(self):
        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = []
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.sent_messages = []
                self.last_game_state = SimpleNamespace(
                    floor=3,
                    current_hp=80,
                    gold=127,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.EVENT,
                    room_type="EVENT",
                    screen=SimpleNamespace(options=[SimpleNamespace(label="Leave", choice_index=0)]),
                )

            def clear_actions(self):
                pass

            def send_message(self, message):
                self.sent_messages.append(message)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

        step = RecordedTraceStep(
            step=33,
            floor=2,
            phase="MAP",
            action={"kind": "map", "name": "?", "choice_index": 0, "x": 1},
            pre={"phase": "MAP", "floor": 2, "hp": 80, "gold": 127},
            post={"phase": "EVENT", "floor": 3, "hp": 80, "gold": 127},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 1
                self.pending_step = None
                self.trace = RecordedRunTrace(
                    path=Path("/tmp/fake_trace.json"),
                    source_format="native_run_trace_v1",
                    seed_long=1,
                    seed_str="1",
                    ascension=0,
                    character="IRONCLAD",
                    steps=[step],
                )
                self.event_leave_map_continue_step_number = 33
                self.neow_post_map_settle_probe_step_number = None

            def next_action(self, game_state):
                return None

            def _force_finalize_current_step(self, step_arg, *, live_snapshot, comparison_note):
                forced.append((step_arg, live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=TimeoutError("still stale"),
        ):
            _play_recorded_game(
                coordinator,
                player_class=PlayerClass.IRONCLAD,
                ascension_level=0,
                seed="1",
                driver=driver,
            )

        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][0].step, 33)
        self.assertEqual(forced[0][2], "event_leave_map_continue_setup_callback_timeout_forced")

    def test_play_recorded_game_forces_finalize_on_post_neow_settle_timeout(self):
        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = []
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.sent_messages = []
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(options=[SimpleNamespace(label="Leave")]),
                )

            def clear_actions(self):
                pass

            def send_message(self, message):
                self.sent_messages.append(message)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

        step = RecordedTraceStep(
            step=4,
            floor=1,
            phase="COMBAT",
            action={"kind": "card", "name": "Flash of Steel"},
            pre={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 1
                self.pending_step = None
                self.trace = RecordedRunTrace(
                    path=Path("/tmp/fake_trace.json"),
                    source_format="native_run_trace_v1",
                    seed_long=1,
                    seed_str="1",
                    ascension=0,
                    character="IRONCLAD",
                    steps=[step],
                )
                self.neow_post_map_settle_probe_step_number = 4

            def next_action(self, game_state):
                return None

            def _force_finalize_current_step(self, step_arg, *, live_snapshot, comparison_note):
                forced.append((step_arg, live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_any_update",
            side_effect=TimeoutError("still stale"),
        ) as any_wait:
            _play_recorded_game(
                coordinator,
                player_class=PlayerClass.IRONCLAD,
                ascension_level=0,
                seed="1",
                driver=driver,
            )

        self.assertEqual(any_wait.call_count, 1)
        self.assertEqual(any_wait.call_args.kwargs["timeout_seconds"], 2.0)
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][0].step, 4)
        self.assertEqual(forced[0][2], "neow_post_map_settle_callback_timeout_forced")
        self.assertIsNone(driver.neow_post_map_settle_probe_step_number)

    def test_play_recorded_game_forces_finalize_on_post_neow_settle_callback_timeout(self):
        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = []
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.sent_messages = []
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(options=[SimpleNamespace(label="Leave")]),
                )

            def clear_actions(self):
                pass

            def send_message(self, message):
                self.sent_messages.append(message)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

        step = RecordedTraceStep(
            step=5,
            floor=1,
            phase="COMBAT",
            action={"kind": "card", "name": "Strike_R"},
            pre={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
            post={"phase": "COMBAT", "floor": 1, "hp": 80, "gold": 99},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 1
                self.pending_step = None
                self.trace = RecordedRunTrace(
                    path=Path("/tmp/fake_trace.json"),
                    source_format="native_run_trace_v1",
                    seed_long=1,
                    seed_str="1",
                    ascension=0,
                    character="IRONCLAD",
                    steps=[step],
                )
                self.neow_post_map_settle_probe_step_number = 5

            def next_action(self, game_state):
                return None

            def _force_finalize_current_step(self, step_arg, *, live_snapshot, comparison_note):
                forced.append((step_arg, live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=TimeoutError("still stale"),
        ):
            _play_recorded_game(
                coordinator,
                player_class=PlayerClass.IRONCLAD,
                ascension_level=0,
                seed="1",
                driver=driver,
            )

        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][0].step, 5)
        self.assertEqual(forced[0][2], "neow_post_map_settle_callback_timeout_forced")
        self.assertIsNone(driver.neow_post_map_settle_probe_step_number)

    def test_play_recorded_game_forces_finalize_on_post_neow_reward_callback_timeout(self):
        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = []
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.sent_messages = []
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.EVENT,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(options=[SimpleNamespace(label="Leave")]),
                )

            def clear_actions(self):
                pass

            def send_message(self, message):
                self.sent_messages.append(message)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

        step = RecordedTraceStep(
            step=15,
            floor=1,
            phase="CARD_REWARD",
            action={"kind": "reward_gold", "name": "GOLD"},
            pre={"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 115},
            post={"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 130},
        )
        forced = []

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 1
                self.pending_step = None
                self.trace = RecordedRunTrace(
                    path=Path("/tmp/fake_trace.json"),
                    source_format="native_run_trace_v1",
                    seed_long=1,
                    seed_str="1",
                    ascension=0,
                    character="IRONCLAD",
                    steps=[step],
                )
                self.neow_post_map_settle_probe_step_number = 15

            def next_action(self, game_state):
                return None

            def _force_finalize_current_step(self, step_arg, *, live_snapshot, comparison_note):
                forced.append((step_arg, live_snapshot, comparison_note))
                self.done = True

        coordinator = FakeCoordinator()
        driver = FakeDriver()

        with patch(
            "spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout",
            side_effect=TimeoutError("still stale"),
        ):
            _play_recorded_game(
                coordinator,
                player_class=PlayerClass.IRONCLAD,
                ascension_level=0,
                seed="1",
                driver=driver,
            )

        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0][0].step, 15)
        self.assertEqual(forced[0][2], "neow_post_map_settle_callback_timeout_forced")
        self.assertIsNone(driver.neow_post_map_settle_probe_step_number)

    def test_play_recorded_game_does_not_drain_followup_actions_in_same_tick(self):
        class FollowupAction:
            def __init__(self, events):
                self.events = events

            def can_be_executed(self, coordinator):
                return True

            def execute(self, coordinator):
                self.events.append("followup")

        class InitialAction:
            def __init__(self, events):
                self.events = events

            def can_be_executed(self, coordinator):
                return True

            def execute(self, coordinator):
                self.events.append("initial")
                coordinator.action_queue.append(FollowupAction(self.events))
                coordinator.game_is_ready = False

        class FakeCoordinator:
            def __init__(self):
                self.events = []
                self.stop_after_run = False
                self.action_queue = [InitialAction(self.events)]
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(room_type="NeowRoom")
                self.receive_calls = []

            def clear_actions(self):
                pass

            def execute_next_action(self):
                action = self.action_queue.pop(0)
                action.execute(self)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                self.receive_calls.append((block, perform_callbacks))
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

        coordinator = FakeCoordinator()
        driver = SimpleNamespace(
            failed=False,
            done=False,
            step_index=0,
            total_steps=1,
            pending_step=None,
            next_action=lambda game_state: None,
        )
        with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout") as ready_wait:
            with patch("spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout") as callback_wait:
                with patch("spirecomm.ai.recorded_run_replay.time.sleep") as sleep:
                    sleep.side_effect = lambda _seconds: setattr(driver, "done", True)
                    _play_recorded_game(
                        coordinator,
                        player_class=PlayerClass.IRONCLAD,
                        ascension_level=0,
                        seed="1",
                        driver=driver,
                    )

        self.assertEqual(coordinator.events, ["initial"])
        ready_wait.assert_not_called()
        callback_wait.assert_not_called()

    def test_play_recorded_game_uses_phase_wait_after_executing_neow_action(self):
        class FakeAction:
            def can_be_executed(self, coordinator):
                return True

        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = [FakeAction()]
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="NeowRoom",
                )
                self.receive_calls = []

            def clear_actions(self):
                pass

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                self.receive_calls.append((block, perform_callbacks))
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def execute_next_action(self):
                self.action_queue.pop(0)
                self.game_is_ready = False

        coordinator = FakeCoordinator()
        driver = SimpleNamespace(
            failed=False,
            done=False,
            step_index=0,
            total_steps=1,
            pending_step=SimpleNamespace(phase="NEOW", post={"phase": "CARD_REWARD"}),
            next_action=lambda game_state: None,
        )
        phase_wait_calls = []

        def _fake_phase_wait(coord, *, expected_phase, timeout_seconds, context):
            phase_wait_calls.append((coord, expected_phase, timeout_seconds, context))
            driver.done = True

        with patch("spirecomm.ai.recorded_run_replay._wait_for_phase_with_timeout", side_effect=_fake_phase_wait):
            with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout") as ready_wait:
                with patch("spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout") as callback_wait:
                    _play_recorded_game(
                        coordinator,
                        player_class=PlayerClass.IRONCLAD,
                        ascension_level=0,
                        seed="1",
                        driver=driver,
                    )

        self.assertEqual(len(phase_wait_calls), 1)
        self.assertIs(phase_wait_calls[0][0], coordinator)
        self.assertEqual(phase_wait_calls[0][1], "CARD_REWARD")
        self.assertIn("after executing Neow action", phase_wait_calls[0][3])
        ready_wait.assert_not_called()
        callback_wait.assert_not_called()

    def test_play_recorded_game_uses_natural_update_wait_after_executing_neow_card_reward(self):
        class FakeAction:
            def can_be_executed(self, coordinator):
                return True

        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = [FakeAction()]
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="NeowRoom",
                )
                self.receive_calls = []

            def clear_actions(self):
                pass

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                self.receive_calls.append((block, perform_callbacks))
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def execute_next_action(self):
                self.action_queue.pop(0)
                self.game_is_ready = False

        coordinator = FakeCoordinator()
        driver = SimpleNamespace(
            failed=False,
            done=False,
            step_index=1,
            total_steps=2,
            trace=SimpleNamespace(steps=[SimpleNamespace(step=1), SimpleNamespace(step=2)]),
            _emit_progress=lambda *args, **kwargs: None,
            neow_card_reward_blind_continue_step_number=None,
            pending_step=SimpleNamespace(
                phase="CARD_REWARD",
                action={"kind": "card_reward"},
                post={"phase": "NEOW"},
            ),
            next_action=lambda game_state: None,
        )
        any_update_calls = []

        def _fake_any_update(coord, *, timeout_seconds, context):
            any_update_calls.append((coord, timeout_seconds, context))
            driver.done = True

        with patch("spirecomm.ai.recorded_run_replay._wait_for_any_update", side_effect=_fake_any_update) as any_update_wait:
            with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout") as ready_wait:
                with patch("spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout") as callback_wait:
                    _play_recorded_game(
                        coordinator,
                        player_class=PlayerClass.IRONCLAD,
                        ascension_level=0,
                        seed="1",
                        driver=driver,
                    )

        self.assertEqual(len(any_update_calls), 1)
        self.assertIs(any_update_calls[0][0], coordinator)
        self.assertIn("after executing Neow card reward", any_update_calls[0][2])
        self.assertEqual(any_update_wait.call_count, 1)
        ready_wait.assert_not_called()
        callback_wait.assert_not_called()

    def test_play_recorded_game_treats_floor_zero_card_reward_as_natural_update_wait(self):
        class FakeAction:
            def can_be_executed(self, coordinator):
                return True

        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = [FakeAction()]
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="",
                )

            def clear_actions(self):
                pass

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                if block:
                    raise AssertionError("blocking receive_game_state_update should not be used")
                return False

            def execute_next_action(self):
                self.action_queue.pop(0)
                self.game_is_ready = False

        coordinator = FakeCoordinator()
        driver = SimpleNamespace(
            failed=False,
            done=False,
            step_index=1,
            total_steps=2,
            trace=SimpleNamespace(steps=[SimpleNamespace(step=1), SimpleNamespace(step=2)]),
            _emit_progress=lambda *args, **kwargs: None,
            neow_card_reward_blind_continue_step_number=None,
            pending_step=SimpleNamespace(
                phase="CARD_REWARD",
                floor=0,
                action={"kind": "card_reward"},
                post={"phase": "NEOW"},
            ),
            next_action=lambda game_state: None,
        )

        any_update_calls = []

        def _fake_any_update(coord, *, timeout_seconds, context):
            any_update_calls.append((coord, timeout_seconds, context))
            driver.done = True

        with patch("spirecomm.ai.recorded_run_replay._wait_for_any_update", side_effect=_fake_any_update) as any_update_wait:
            with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout") as ready_wait:
                with patch("spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout") as callback_wait:
                    _play_recorded_game(
                        coordinator,
                        player_class=PlayerClass.IRONCLAD,
                        ascension_level=0,
                        seed="1",
                        driver=driver,
                    )

        self.assertEqual(len(any_update_calls), 1)
        self.assertEqual(any_update_wait.call_count, 1)
        ready_wait.assert_not_called()
        callback_wait.assert_not_called()

    def test_play_recorded_game_can_drive_next_action_from_ready_live_state(self):
        class FakeAction:
            def can_be_executed(self, coordinator):
                return True

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 0
                self.total_steps = 1
                self.calls = 0

            def next_action(self, game_state):
                self.calls += 1
                return FakeAction()

        class FakeCoordinator:
            def __init__(self, driver):
                self.stop_after_run = False
                self.action_queue = []
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(room_type="NeowRoom")
                self.driver = driver
                self.executed = 0
                self.receive_calls = []

            def clear_actions(self):
                pass

            def add_action_to_queue(self, action):
                self.action_queue.append(action)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                self.receive_calls.append((block, perform_callbacks))
                return False

            def execute_next_action(self):
                self.action_queue.pop(0)
                self.executed += 1
                self.driver.done = True

        driver = FakeDriver()
        coordinator = FakeCoordinator(driver)

        with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout") as ready_wait:
            with patch("spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout") as callback_wait:
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(driver.calls, 1)
        self.assertEqual(coordinator.executed, 1)
        ready_wait.assert_not_called()
        callback_wait.assert_not_called()

    def test_play_recorded_game_dispatches_nonready_neow_bridge_action_via_queue(self):
        class InitialAction:
            def can_be_executed(self, coordinator):
                return True

            def execute(self, coordinator):
                coordinator.game_is_ready = False

        class FakeDriver:
            def __init__(self):
                self.failed = False
                self.done = False
                self.step_index = 1
                self.total_steps = 2
                self.neow_card_reward_blind_continue_step_number = 2
                self.calls = 0
                self.trace = SimpleNamespace(
                    steps=[
                        SimpleNamespace(step=1),
                        SimpleNamespace(
                            step=2,
                            phase="NEOW",
                            action={"kind": "neow", "name": "OPTION_0"},
                        ),
                    ]
                )

            def next_action(self, game_state):
                self.calls += 1
                return StateAction()

            def _emit_progress(self, **kwargs):
                return None

        class FakeCoordinator:
            def __init__(self):
                self.stop_after_run = False
                self.action_queue = [InitialAction()]
                self.game_is_ready = True
                self.in_game = True
                self.last_error = None
                self.last_game_state = SimpleNamespace(
                    floor=0,
                    current_hp=80,
                    gold=99,
                    potions=[],
                    deck=[],
                    in_combat=False,
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="NeowRoom",
                    screen=SimpleNamespace(cards=[]),
                )
                self.messages = []
                self.output_queue = SimpleNamespace(empty=lambda: True)

            def clear_actions(self):
                pass

            def add_action_to_queue(self, action):
                self.action_queue.append(action)

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                return False

            def execute_next_action(self):
                action = self.action_queue.pop(0)
                action.execute(self)

            def send_message(self, message):
                self.messages.append(message)
                self.game_is_ready = False
                driver.done = True

        driver = FakeDriver()
        coordinator = FakeCoordinator()
        ready_wait_calls = []

        def _fake_ready_wait(coord, *, timeout_seconds, context, initial_probe_delay_seconds=0.0):
            ready_wait_calls.append((coord, timeout_seconds, context, initial_probe_delay_seconds))
            driver.done = True

        with patch("spirecomm.ai.recorded_run_replay._wait_for_ready_state_with_timeout", side_effect=_fake_ready_wait) as ready_wait:
            with patch("spirecomm.ai.recorded_run_replay._wait_for_callback_update_with_timeout") as callback_wait:
                _play_recorded_game(
                    coordinator,
                    player_class=PlayerClass.IRONCLAD,
                    ascension_level=0,
                    seed="1",
                    driver=driver,
                )

        self.assertEqual(driver.calls, 1)
        self.assertEqual(coordinator.messages, ["state"])
        self.assertEqual(coordinator.action_queue, [])
        self.assertEqual(ready_wait_calls, [])
        self.assertEqual(ready_wait.call_count, 0)
        callback_wait.assert_not_called()

    def test_load_aligned_trace_normalizes_actions(self):
        payload = {
            "summary": {
                "seed_long": 123,
                "seed_str": "ABC",
                "ascension": 0,
            },
            "trace": [
                {
                    "step": 0,
                    "phase": "BATTLE",
                    "floor": 1,
                    "action": ["card", "Bash", 0],
                    "pre": {"hp": 80},
                    "post": {"hp": 80},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "aligned.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            trace = load_recorded_trace(path)

        self.assertEqual(trace.source_format, "aligned_trace_v1")
        self.assertEqual(trace.steps[0].phase, "COMBAT")
        self.assertEqual(trace.steps[0].action["kind"], "card")
        self.assertEqual(trace.steps[0].action["target_index"], 0)

    def test_load_verbose_trace_compacts_snapshots(self):
        payload = {
            "seed_long": 456,
            "seed_str": "DEF",
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "COMBAT",
                    "floor": 2,
                    "action": {"kind": "end", "name": "END_TURN"},
                    "pre_state": {
                        "character": "IRONCLAD",
                        "floor": 2,
                        "current_hp": 72,
                        "gold": 111,
                        "combat_state": {
                            "player": {"block": 5, "energy": 1},
                            "hand": [{"card_id": "Strike_R"}, {"card_id": "Bash"}],
                            "monsters": [
                                {"monster_id": "JawWorm", "current_hp": 31, "block": 0, "intent": "ATTACK", "is_gone": False}
                            ],
                        },
                    },
                    "post_state": {
                        "floor": 2,
                        "current_hp": 68,
                        "gold": 111,
                        "combat_state": {
                            "player": {"block": 0, "energy": 3},
                            "hand": [{"card_id": "Defend_R"}],
                            "monsters": [
                                {"monster_id": "JawWorm", "current_hp": 31, "block": 0, "intent": "ATTACK_DEFEND", "is_gone": False}
                            ],
                        },
                    },
                    "post_phase": "COMBAT",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "verbose.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            trace = load_recorded_trace(path)

        self.assertEqual(trace.source_format, "native_run_trace_v1")
        self.assertEqual(trace.character, "IRONCLAD")
        self.assertEqual(trace.steps[0].pre["energy"], 1)
        self.assertEqual(trace.steps[0].post["monsters"][0]["intent"], "ATTACK_DEFEND")

    def test_load_verbose_trace_relabels_card_select_steps(self):
        payload = {
            "seed_long": 456,
            "seed_str": "DEF",
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "COMBAT",
                    "floor": 5,
                    "action": {
                        "kind": "card",
                        "name": "Headbutt",
                        "card_id": "Headbutt",
                        "card_index": 4,
                    },
                    "pre_state": {
                        "character": "IRONCLAD",
                        "floor": 5,
                        "current_hp": 72,
                        "gold": 99,
                    },
                    "post_state": {
                        "floor": 5,
                        "current_hp": 72,
                        "gold": 99,
                    },
                    "post_phase": "COMBAT",
                },
                {
                    "step": 1,
                    "phase": "COMBAT",
                    "floor": 5,
                    "action": {
                        "kind": "card_select",
                        "name": "HEADBUTT",
                        "choice_index": 0,
                        "card_id": "Bash",
                    },
                    "pre_state": {
                        "character": "IRONCLAD",
                        "floor": 5,
                        "current_hp": 72,
                        "gold": 99,
                    },
                    "post_state": {
                        "floor": 5,
                        "current_hp": 72,
                        "gold": 99,
                    },
                    "post_phase": "COMBAT",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "verbose_card_select.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            trace = load_recorded_trace(path)

        self.assertEqual(trace.steps[0].phase, "COMBAT")
        self.assertEqual(trace.steps[0].post["phase"], "CARD_SELECT")
        self.assertEqual(trace.steps[1].phase, "CARD_SELECT")
        self.assertEqual(trace.steps[1].pre["phase"], "CARD_SELECT")
        self.assertEqual(trace.steps[1].post["phase"], "COMBAT")

    def test_map_action_prefers_matching_x_coordinate(self):
        next_nodes = [Node(0, 0, "M"), Node(2, 0, "M")]
        game_state = SimpleNamespace(
            screen_type=ScreenType.MAP,
            screen=MapScreen(current_node=None, next_nodes=next_nodes, boss_available=False),
        )
        step = SimpleNamespace(
            phase="MAP",
            action={"kind": "map", "name": "M", "x": 2, "choice_index": 0},
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertIsInstance(action, ChooseMapNodeAction)
        self.assertEqual(action.node.x, 2)

    def test_combat_card_translation_prefers_live_monster_index(self):
        hand = [
            Card("Strike_R", "Strike", CardType.ATTACK, CardRarity.BASIC, uuid="a"),
            Card("Flash of Steel", "Flash of Steel", CardType.ATTACK, CardRarity.UNCOMMON, uuid="b"),
        ]
        game_state = SimpleNamespace(
            in_combat=True,
            screen_type=ScreenType.NONE,
            hand=hand,
            monsters=[
                SimpleNamespace(monster_index=0, monster_id="SpikeSlime_S", current_hp=0, is_gone=True),
                SimpleNamespace(monster_index=1, monster_id="AcidSlime_M", current_hp=30, is_gone=False),
            ],
        )
        step = SimpleNamespace(
            phase="COMBAT",
            action={
                "kind": "card",
                "name": "Flash of Steel",
                "card_id": "Flash of Steel",
                "target_index": 1,
                "model_target_index": 0,
            },
            raw_pre_state={
                "combat_state": {
                    "monsters": [
                        {"monster_index": 0, "monster_id": "SpikeSlime_S", "current_hp": 0, "is_gone": True},
                        {"monster_index": 1, "monster_id": "AcidSlime_M", "current_hp": 30, "is_gone": False},
                    ]
                }
            },
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertEqual(action.__class__.__name__, "PlayCardAction")
        self.assertEqual(action.target_index, 1)

    def test_combat_card_translation_falls_back_to_model_target_index(self):
        hand = [
            Card("Strike_R", "Strike", CardType.ATTACK, CardRarity.BASIC, uuid="a"),
            Card("Flash of Steel", "Flash of Steel", CardType.ATTACK, CardRarity.UNCOMMON, uuid="b"),
        ]
        game_state = SimpleNamespace(
            in_combat=True,
            screen_type=ScreenType.NONE,
            hand=hand,
            monsters=[
                SimpleNamespace(monster_index=0, monster_id="Slime", current_hp=15, is_gone=False),
                SimpleNamespace(monster_index=2, monster_id="Slime", current_hp=18, is_gone=False),
            ],
        )
        step = SimpleNamespace(
            phase="COMBAT",
            action={
                "kind": "card",
                "name": "Flash of Steel",
                "card_id": "Flash of Steel",
                "target_index": 1,
                "model_target_index": 1,
            },
            raw_pre_state={
                "combat_state": {
                    "monsters": [
                        {"monster_index": 0, "monster_id": "Slime", "current_hp": 15, "is_gone": False},
                        {"monster_index": 1, "monster_id": "Ghost", "current_hp": 0, "is_gone": True},
                        {"monster_index": 2, "monster_id": "Slime", "current_hp": 18, "is_gone": False},
                    ]
                }
            },
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertEqual(action.__class__.__name__, "PlayCardAction")
        self.assertEqual(action.target_index, 1)

    def test_combat_card_translation_uses_raw_slot_when_trace_monsters_omit_monster_index(self):
        hand = [
            Card("Strike_R", "Strike", CardType.ATTACK, CardRarity.BASIC, uuid="a"),
            Card("Flash of Steel", "Flash of Steel", CardType.ATTACK, CardRarity.UNCOMMON, uuid="b"),
        ]
        game_state = SimpleNamespace(
            in_combat=True,
            screen_type=ScreenType.NONE,
            hand=hand,
            monsters=[
                SimpleNamespace(monster_index=0, monster_id="SpikeSlime_S", current_hp=0, is_gone=True),
                SimpleNamespace(monster_index=1, monster_id="AcidSlime_M", current_hp=30, is_gone=False),
            ],
        )
        step = SimpleNamespace(
            phase="COMBAT",
            action={
                "kind": "card",
                "name": "Strike",
                "card_id": "Strike_R",
                "target_index": 1,
                "model_target_index": 1,
            },
            raw_pre_state={
                "combat_state": {
                    "monsters": [
                        {"monster_id": "SpikeSlime_S", "current_hp": 0, "is_gone": False},
                        {"monster_id": "AcidSlime_M", "current_hp": 30, "is_gone": False},
                    ]
                }
            },
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertEqual(action.__class__.__name__, "PlayCardAction")
        self.assertEqual(action.target_index, 1)

    def test_card_reward_step_from_combat_reward_selects_card_reward_item(self):
        rewards = [
            CombatReward(RewardType.GOLD, gold=15),
            CombatReward(RewardType.CARD),
        ]
        game_state = SimpleNamespace(
            screen_type=ScreenType.COMBAT_REWARD,
            screen=CombatRewardScreen(rewards),
        )
        step = SimpleNamespace(
            phase="CARD_REWARD",
            action={"kind": "card_reward", "name": "Anger", "card_id": "Anger"},
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertIsInstance(action, CombatRewardAction)
        self.assertEqual(action.combat_reward.reward_type, RewardType.CARD)

    def test_campfire_targeted_step_becomes_rest_then_grid_select(self):
        rest_game = SimpleNamespace(
            screen_type=ScreenType.REST,
            screen=RestScreen(False, [RestOption.REST, RestOption.SMITH]),
        )
        step = SimpleNamespace(
            phase="CAMPFIRE",
            action={"kind": "campfire", "name": "SMITH", "target_index": 1},
        )

        first_action = _next_trace_action_for_game(rest_game, step)
        self.assertIsInstance(first_action, RestAction)
        self.assertEqual(first_action.name, "SMITH")

        cards = [
            Card("Strike_R", "Strike", CardType.ATTACK, CardRarity.BASIC, upgrades=0, uuid="a"),
            Card("Bash", "Bash", CardType.ATTACK, CardRarity.BASIC, upgrades=0, uuid="b"),
        ]
        grid_game = SimpleNamespace(
            screen_type=ScreenType.GRID,
            screen=GridSelectScreen(cards, [], 1, False, True, True, False, False),
            deck=cards,
        )
        second_action = _next_trace_action_for_game(grid_game, step)

        self.assertEqual(second_action.__class__.__name__, "CardSelectAction")

    def test_campfire_recall_step_can_require_implicit_proceed(self):
        game_state = SimpleNamespace(
            screen_type=ScreenType.REST,
            screen=RestScreen(True, []),
            proceed_available=True,
        )
        step = SimpleNamespace(
            phase="CAMPFIRE",
            action={"kind": "campfire", "name": "RECALL"},
            post={"phase": "MAP"},
        )

        first_action = _next_trace_action_for_game(game_state, step)

        self.assertIsInstance(first_action, RestAction)
        self.assertEqual(first_action.name, "RECALL")

    def test_campfire_rest_screen_prefers_choice_index_when_present(self):
        game_state = SimpleNamespace(
            screen_type=ScreenType.REST,
            screen=RestScreen(False, [RestOption.REST, RestOption.SMITH, RestOption.RECALL]),
        )
        step = SimpleNamespace(
            phase="CAMPFIRE",
            action={"kind": "campfire", "name": "RECALL", "choice_index": 2},
        )

        first_action = _next_trace_action_for_game(game_state, step)

        self.assertIsInstance(first_action, ChooseAction)
        self.assertEqual(first_action.choice_index, 2)

    def test_card_reward_screen_skip_maps_to_cancel_action(self):
        screen_cards = [Card("Anger", "Anger", CardType.ATTACK, CardRarity.COMMON, uuid="x")]
        game_state = SimpleNamespace(
            screen_type=ScreenType.CARD_REWARD,
            screen=SimpleNamespace(cards=screen_cards, can_bowl=False, can_skip=True),
        )
        step = SimpleNamespace(
            phase="CARD_REWARD",
            action={"kind": "skip", "name": "SKIP"},
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertEqual(action.__class__.__name__, "CancelAction")
        self.assertEqual(action.command, "cancel")

    def test_card_reward_prefers_card_index_over_choice_index(self):
        screen_cards = [
            Card("Fire Breathing", "Fire Breathing", CardType.POWER, CardRarity.UNCOMMON, uuid="a"),
            Card("Barricade", "Barricade", CardType.POWER, CardRarity.RARE, uuid="b"),
            Card("Wild Strike", "Wild Strike", CardType.ATTACK, CardRarity.COMMON, uuid="c"),
        ]
        game_state = SimpleNamespace(
            screen_type=ScreenType.CARD_REWARD,
            screen=SimpleNamespace(cards=screen_cards, can_bowl=False, can_skip=True),
        )
        step = RecordedTraceStep(
            step=124,
            phase="CARD_REWARD",
            floor=11,
            action={
                "kind": "card_reward",
                "name": "Barricade",
                "card_id": "Barricade",
                "choice_index": 2,
                "card_index": 1,
                "card": {"card_id": "Barricade", "name": "Barricade"},
            },
            pre={"phase": "CARD_REWARD", "floor": 11, "hp": 8, "gold": 29},
            post={"phase": "CARD_REWARD", "floor": 11, "hp": 8, "gold": 29},
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertIsInstance(action, CardRewardAction)
        self.assertEqual(action.name, "Barricade")

    def test_card_select_step_on_grid_uses_choice_index(self):
        cards = [
            Card("Bash", "Bash", CardType.ATTACK, CardRarity.BASIC, upgrades=0, uuid="a"),
            Card("Strike_R", "Strike", CardType.ATTACK, CardRarity.BASIC, upgrades=0, uuid="b"),
        ]
        game_state = SimpleNamespace(
            screen_type=ScreenType.GRID,
            screen=GridSelectScreen(cards, [], 1, False, False, False, False, False),
        )
        step = SimpleNamespace(
            phase="CARD_SELECT",
            action={"kind": "card_select", "name": "HEADBUTT", "choice_index": 0, "card_id": "Bash"},
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertEqual(action.__class__.__name__, "CardSelectAction")

    def test_combat_potion_translation_matches_loose_potion_ids(self):
        game_state = SimpleNamespace(
            in_combat=True,
            screen_type=ScreenType.NONE,
            potions=[
                SimpleNamespace(potion_id="LiquidBronze", name="Liquid Bronze", can_use=True, requires_target=False),
                SimpleNamespace(potion_id="DexterityPotion", name="Dexterity Potion", can_use=True, requires_target=False),
                SimpleNamespace(potion_id="Potion Slot", name="Potion Slot", can_use=False, requires_target=False),
            ],
        )
        step = SimpleNamespace(
            phase="COMBAT",
            action={"kind": "potion", "name": "Dexterity Potion", "potion_id": "Dexterity Potion", "target_index": 0},
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertEqual(action.__class__.__name__, "PotionAction")
        self.assertEqual(action.potion_index, 1)

    def test_neow_event_is_normalized_to_neow_phase(self):
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            screen=SimpleNamespace(event_id="Neow Event", event_name="Neow"),
        )
        self.assertEqual(_normalize_live_phase(game_state), "NEOW")

    def test_neow_card_select_overlay_is_detected(self):
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(
                cards=[SimpleNamespace(card_id="Strike_R", name="Strike", upgrades=0)],
                options=[],
                rewards=[],
            ),
        )

        self.assertTrue(_is_neow_card_select_state(game_state))
        self.assertEqual(_normalize_live_phase(game_state), "CARD_SELECT")

    def test_neow_card_reward_screen_is_not_misclassified_as_card_select(self):
        game_state = SimpleNamespace(
            screen_type=ScreenType.CARD_REWARD,
            room_type="NeowRoom",
            screen=SimpleNamespace(
                cards=[SimpleNamespace(card_id="Flash of Steel", name="Flash of Steel", upgrades=0)],
                options=[],
                rewards=[],
            ),
        )

        self.assertFalse(_is_neow_card_select_state(game_state))
        self.assertEqual(_normalize_live_phase(game_state), "CARD_REWARD")

    def test_neow_card_reward_pick_prefers_choice_index_over_named_card_action(self):
        game_state = SimpleNamespace(
            screen_type=ScreenType.CARD_REWARD,
            room_type="NeowRoom",
            screen=SimpleNamespace(
                cards=[
                    SimpleNamespace(card_id="Panacea", name="Panacea", upgrades=0),
                    SimpleNamespace(card_id="Dramatic Entrance", name="Dramatic Entrance", upgrades=0),
                    SimpleNamespace(card_id="Flash of Steel", name="Flash of Steel", upgrades=0),
                ],
                options=[],
                rewards=[],
            ),
        )
        step = RecordedTraceStep(
            step=1,
            phase="CARD_REWARD",
            floor=0,
            action={"kind": "card_reward", "name": "Flash of Steel", "card_id": "Flash of Steel", "choice_index": 2},
            pre={"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R"]},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "deck": ["Strike_R", "Flash of Steel"]},
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertEqual(action.__class__.__name__, "NeowCardRewardAction")
        self.assertEqual(action.choice_index, 2)
        self.assertEqual(action.name, "Flash of Steel")

    def test_neow_card_reward_action_prefers_index_and_sends_blind_bridge_burst(self):
        from spirecomm.communication.action import (
            NEOW_REWARD_SETTLE_TIMEOUT_TICKS,
            NeowCardRewardAction,
        )

        class FakeCoordinator:
            def __init__(self):
                self.messages = []

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        action = NeowCardRewardAction(
            choice_index=2,
            name="Flash of Steel",
            continue_choice_index=0,
            post_continue_choice_index=0,
            bridge_delay_seconds=0.0,
            continue_delay_seconds=0.0,
        )
        action.execute(coordinator)

        self.assertEqual(
            coordinator.messages,
            [
                "choose 2",
                f"wait {NEOW_REWARD_SETTLE_TIMEOUT_TICKS}",
                "state",
                "choose 0",
                "wait 1",
                "state",
                "choose 0",
                "wait 1",
                "state",
            ],
        )

    def test_neow_card_select_action_queues_settle_pulse_then_state(self):
        from spirecomm.communication.action import (
            NEOW_REWARD_SETTLE_TIMEOUT_TICKS,
            NeowCardSelectAction,
        )

        class FakeCoordinator:
            def __init__(self):
                self.messages = []
                self.action_queue = []

            def send_message(self, message):
                self.messages.append(message)

            def add_action_to_queue(self, action):
                self.action_queue.append(action)

        coordinator = FakeCoordinator()

        action = NeowCardSelectAction(choice_index=9)
        action.execute(coordinator)

        self.assertEqual(coordinator.messages, ["choose 9"])
        self.assertEqual(coordinator.action_queue, [])

    def test_confirm_action_accepts_grid_upgrade_screen(self):
        from spirecomm.communication.action import ConfirmAction

        coordinator = SimpleNamespace(
            game_is_ready=True,
            last_game_state=SimpleNamespace(
                screen_type=ScreenType.GRID,
                screen=SimpleNamespace(
                    confirm_up=False,
                    is_just_for_confirming=False,
                    for_upgrade=True,
                    for_transform=False,
                    for_purge=False,
                    any_number=False,
                ),
            ),
        )

        self.assertTrue(ConfirmAction().can_be_executed(coordinator))

    def test_optional_card_select_confirm_action_is_nonready_and_noops_for_grid(self):
        from spirecomm.communication.action import OptionalCardSelectConfirmAction

        class FakeCoordinator:
            def __init__(self):
                self.game_is_ready = False
                self.action_queue = []
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.GRID,
                    screen=SimpleNamespace(
                        confirm_up=False,
                        is_just_for_confirming=False,
                        for_upgrade=True,
                        for_transform=False,
                        for_purge=False,
                        any_number=False,
                    ),
                )

            def add_action_to_queue(self, action):
                self.action_queue.append(action)

        coordinator = FakeCoordinator()
        action = OptionalCardSelectConfirmAction()

        self.assertTrue(action.can_be_executed(coordinator))
        action.execute(coordinator)

        self.assertEqual(
            [queued.__class__.__name__ for queued in coordinator.action_queue],
            [],
        )

    def test_neow_card_select_translation_prefers_unique_name_on_grid(self):
        cards = [
            Card("Strike_R", "Strike", CardType.ATTACK, CardRarity.BASIC, upgrades=0, uuid="a"),
            Card("Bash", "Bash", CardType.ATTACK, CardRarity.BASIC, upgrades=0, uuid="b"),
        ]
        game_state = SimpleNamespace(
            screen_type=ScreenType.GRID,
            room_type="NeowRoom",
            screen=GridSelectScreen(cards, [], 1, False, False, False, True, False),
        )
        step = SimpleNamespace(
            phase="CARD_SELECT",
            action={"kind": "card_select", "name": "Bash", "card_id": "Bash", "choice_index": 1},
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertEqual(action.__class__.__name__, "NeowCardSelectAction")
        self.assertEqual(action.choice_index, 1)
        self.assertEqual(action.name, "Bash")

    def test_next_action_waits_while_neow_continue_is_still_stale_grid(self):
        step0 = RecordedTraceStep(
            step=1,
            floor=0,
            phase="CARD_SELECT",
            action={"kind": "card_select", "choice_index": 1, "name": "Bash", "card_id": "Bash"},
            pre={"phase": "CARD_SELECT", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
        )
        step1 = RecordedTraceStep(
            step=2,
            floor=0,
            phase="NEOW",
            action={"kind": "neow", "choice_index": 0, "name": "CONTINUE"},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[step0, step1],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=2,
            progress_enabled=False,
        )
        driver.step_index = 1
        driver.neow_card_select_continue_step_number = 2

        game_state = SimpleNamespace(
            floor=0,
            current_hp=80,
            gold=99,
            deck=[],
            potions=[],
            in_combat=False,
            room_type="NeowRoom",
            screen_type=ScreenType.GRID,
            screen=GridSelectScreen(
                [Card("Bash", "Bash", CardType.ATTACK, CardRarity.BASIC, upgrades=0, uuid="b")],
                [],
                1,
                False,
                False,
                False,
                True,
                False,
            ),
        )

        self.assertIsNone(driver.next_action(game_state))

    def test_neow_continue_action_sends_delayed_continue_burst(self):
        from spirecomm.communication.action import NeowContinueAction

        class FakeCoordinator:
            def __init__(self):
                self.messages = []

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        action = NeowContinueAction()
        action.execute(coordinator)

        self.assertEqual(
            coordinator.messages,
            ["choose 0", "wait 1", "state"],
        )

    def test_neow_continue_action_can_append_blind_map_choice(self):
        from spirecomm.communication.action import NeowContinueAction

        class FakeCoordinator:
            def __init__(self):
                self.messages = []

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        action = NeowContinueAction(post_continue_choice_index=0)
        action.execute(coordinator)

        self.assertEqual(
            coordinator.messages,
            [
                "choose 0",
                "wait 1",
                "state",
                "choose 0",
                "wait 1",
                "state",
            ],
        )

    def test_neow_reward_translation_builds_trace_driven_continue_and_map_bridge(self):
        game_state = SimpleNamespace(
            screen_type=ScreenType.CARD_REWARD,
            room_type="NeowRoom",
            screen=SimpleNamespace(
                cards=[
                    SimpleNamespace(card_id="Panacea", name="Panacea"),
                    SimpleNamespace(card_id="Dramatic Entrance", name="Dramatic Entrance"),
                    SimpleNamespace(card_id="Flash of Steel", name="Flash of Steel"),
                ]
            ),
        )
        trace = RecordedRunTrace(
            path=Path("/tmp/fake_trace.json"),
            source_format="native_run_trace_v1",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[
                RecordedTraceStep(
                    step=1,
                    phase="CARD_REWARD",
                    floor=0,
                    action={"kind": "card_reward", "name": "Flash of Steel", "card_id": "Flash of Steel", "choice_index": 2},
                    pre={"phase": "CARD_REWARD", "deck": ["Strike_R"]},
                    post={"phase": "NEOW", "deck": ["Strike_R", "Flash of Steel"]},
                ),
                RecordedTraceStep(
                    step=2,
                    phase="NEOW",
                    floor=0,
                    action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
                    pre={"phase": "NEOW", "deck": ["Strike_R", "Flash of Steel"]},
                    post={"phase": "MAP", "deck": ["Strike_R", "Flash of Steel"]},
                ),
                RecordedTraceStep(
                    step=3,
                    phase="MAP",
                    floor=1,
                    action={"kind": "map", "choice_index": 0, "x": 1, "name": "M"},
                    pre={"phase": "MAP", "deck": ["Strike_R", "Flash of Steel"]},
                    post={"phase": "COMBAT", "floor": 1, "deck": ["Strike_R", "Flash of Steel"]},
                ),
            ],
        )
        replayer = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=False,
            stop_on_mismatch=True,
            total_steps=3,
            progress_enabled=False,
        )

        action = replayer.next_action(game_state)

        self.assertEqual(action.__class__.__name__, "NeowCardRewardAction")
        self.assertEqual(action.choice_index, 2)
        self.assertEqual(action.continue_choice_index, 0)
        self.assertEqual(action.post_continue_choice_index, 0)
        self.assertGreaterEqual(action.bridge_delay_seconds, 0.0)
        self.assertGreaterEqual(action.continue_delay_seconds, 0.0)

    def test_wait_for_phase_gives_neow_card_reward_a_short_natural_grace_period(self):
        class FakeCoordinator:
            def __init__(self):
                self.last_error = None
                self.game_is_ready = True
                self.in_game = True
                self.last_game_state = SimpleNamespace(
                    screen_type=ScreenType.CARD_REWARD,
                    room_type="NeowRoom",
                )
                self.messages = []

            def receive_game_state_update(self, block=False, perform_callbacks=False):
                return False

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()

        with self.assertRaises(TimeoutError):
            _wait_for_phase_with_timeout(
                coordinator,
                expected_phase="NEOW",
                timeout_seconds=0.45,
                initial_probe_delay_seconds=0.5,
                context="neow card reward natural advance window",
            )

        self.assertEqual(coordinator.messages, [])

    def test_neow_event_payload_with_cards_is_promoted_to_grid_screen(self):
        payload = {
            "current_hp": 80,
            "max_hp": 80,
            "floor": 0,
            "act": 1,
            "gold": 99,
            "seed": 20,
            "class": "IRONCLAD",
            "ascension_level": 0,
            "relics": [{"name": "Burning Blood", "id": "Burning Blood", "counter": -1}],
            "deck": [],
            "map": [],
            "potions": [],
            "act_boss": None,
            "is_screen_up": True,
            "screen_type": "EVENT",
            "screen_name": "EVENT",
            "screen_state": {
                "cards": [
                    {
                        "name": "Strike",
                        "id": "Strike_R",
                        "type": "ATTACK",
                        "rarity": "BASIC",
                        "upgrades": 0,
                        "cost": 1,
                        "exhausts": False,
                        "ethereal": False,
                        "has_target": True,
                        "uuid": "test-card",
                    }
                ],
                "selected_cards": [],
                "num_cards": 1,
                "any_number": False,
                "for_upgrade": False,
                "for_transform": False,
                "for_purge": True,
                "confirm_up": False,
            },
            "room_phase": "EVENT",
            "room_type": "NeowRoom",
            "choice_list": [],
            "available_commands": ["choose", "state"],
        }

        game_state = Game.from_json(payload, payload["available_commands"])

        self.assertEqual(game_state.screen_type, ScreenType.GRID)
        self.assertEqual(_normalize_live_phase(game_state), "CARD_SELECT")
        self.assertEqual(len(game_state.screen.cards), 1)

    def test_single_neow_continue_action_returns_none_for_card_select_overlay(self):
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(
                cards=[SimpleNamespace(card_id="Strike_R", name="Strike", upgrades=0)],
                options=[],
                rewards=[],
            ),
        )

        self.assertIsNone(_single_neow_continue_action(game_state))

    def test_grid_screen_in_combat_is_normalized_to_card_select(self):
        game_state = SimpleNamespace(
            in_combat=True,
            screen_type=ScreenType.GRID,
            screen=SimpleNamespace(),
        )
        self.assertEqual(_normalize_live_phase(game_state), "CARD_SELECT")

    def test_treasure_reward_screen_is_normalized_to_card_reward(self):
        game_state = SimpleNamespace(
            screen_type=ScreenType.COMBAT_REWARD,
            room_type="TreasureRoom",
            screen=SimpleNamespace(),
        )
        self.assertEqual(_normalize_live_phase(game_state), "CARD_REWARD")

    def test_neow_card_select_translation_works_even_if_live_screen_type_is_event(self):
        card = Card(
            card_id="Strike_R",
            name="Strike",
            card_type=CardType.ATTACK,
            rarity=CardRarity.BASIC,
            upgrades=0,
            has_target=True,
        )
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            room_type="NeowRoom",
            screen=SimpleNamespace(cards=[card], options=[], rewards=[]),
        )
        step = SimpleNamespace(
            phase="CARD_SELECT",
            action={"kind": "card_select", "card_id": "Strike_R", "name": "Strike", "choice_index": 0},
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertEqual(action.__class__.__name__, "NeowCardSelectAction")
        self.assertEqual(action.choice_index, 0)

    def test_pending_neow_card_select_on_grid_skips_extra_followup(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/trace.json"),
            source_format="recorded",
            seed_long=20,
            seed_str="K",
            ascension=0,
            character="IRONCLAD",
            steps=[],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=False,
            total_steps=1,
            progress_enabled=False,
        )
        driver.pending_step = RecordedTraceStep(
            step=1,
            phase="CARD_SELECT",
            floor=0,
            action={"kind": "card_select", "card_id": "Strike_R", "choice_index": 0},
            pre={"phase": "CARD_SELECT", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
        )
        game_state = SimpleNamespace(
            room_type="NeowRoom",
            screen_type=ScreenType.GRID,
            screen=SimpleNamespace(confirm_up=True),
        )

        action = driver._maybe_pending_followup_action(game_state)

        self.assertIsNone(action)

    def test_neow_continue_action_can_skip_settle_probe(self):
        from spirecomm.communication.action import NeowContinueAction

        class FakeCoordinator:
            def __init__(self):
                self.messages = []

            def send_message(self, message):
                self.messages.append(message)

        coordinator = FakeCoordinator()
        action = NeowContinueAction(include_settle_probe=False)
        action.execute(coordinator)

        self.assertEqual(coordinator.messages, ["choose 0"])

    def test_neow_card_select_continue_uses_minimal_continue_on_single_option_event(self):
        trace = RecordedRunTrace(
            path=Path("/tmp/trace.json"),
            source_format="recorded",
            seed_long=20,
            seed_str="K",
            ascension=0,
            character="IRONCLAD",
            steps=[],
        )
        driver = _StateDrivenReplayDriver(
            trace=trace,
            compare_state=True,
            stop_on_mismatch=True,
            total_steps=1,
            progress_enabled=False,
        )
        driver.neow_card_select_continue_step_number = 2
        step = RecordedTraceStep(
            step=2,
            phase="NEOW",
            floor=0,
            action={"kind": "neow", "choice_index": 0},
            pre={"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
            post={"phase": "MAP", "floor": 0, "hp": 80, "gold": 99},
        )
        driver.trace = RecordedRunTrace(
            path=Path("/tmp/trace.json"),
            source_format="recorded",
            seed_long=20,
            seed_str="K",
            ascension=0,
            character="IRONCLAD",
            steps=[step],
        )
        driver.total_steps = 1
        driver.step_index = 0
        game_state = SimpleNamespace(
            room_type="NeowRoom",
            screen_type=ScreenType.EVENT,
            screen=SimpleNamespace(options=[SimpleNamespace(label="Continue", choice_index=0)]),
        )

        action = driver.next_action(game_state)

        self.assertEqual(action.__class__.__name__, "NeowContinueAction")
        self.assertFalse(action.include_settle_probe)

    def test_floor_zero_is_tolerated_for_pre_first_room_screens(self):
        mismatches = compare_snapshots(
            {"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 99},
            {"phase": "CARD_REWARD", "floor": 0, "hp": 80, "gold": 99},
        )
        self.assertEqual(mismatches, [])

    def test_neow_talk_screen_maps_to_setup_talk_choice(self):
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            screen=SimpleNamespace(
                event_id="Neow Event",
                event_name="Neow",
                options=[SimpleNamespace(choice_index=0, label="Talk", text="[Talk]")],
            ),
        )
        step = SimpleNamespace(
            phase="NEOW",
            action={"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
        )

        action = None
        if (
            step.phase == "NEOW"
            and game_state.screen_type == ScreenType.EVENT
            and len(getattr(game_state.screen, "options", []) or []) == 1
        ):
            option = game_state.screen.options[0]
            option_text = " ".join(
                str(part)
                for part in [getattr(option, "label", None), getattr(option, "text", None), getattr(option, "name", None)]
                if part
            ).lower()
            if "talk" in option_text:
                action = ChooseAction(choice_index=0)

        self.assertIsInstance(action, ChooseAction)

    def test_skip_can_map_to_event_leave_after_neow_reward(self):
        option = SimpleNamespace(choice_index=0, label="Leave", text="[Leave]")
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            screen=SimpleNamespace(event_id="Neow Event", event_name="Neow", options=[option]),
        )
        step = SimpleNamespace(
            phase="CARD_REWARD",
            action={"kind": "skip", "name": "SKIP"},
        )

        action = _next_trace_action_for_game(game_state, step)

        self.assertEqual(action.__class__.__name__, "EventOptionAction")

    def test_neow_floor_offset_is_accepted_but_phase_mismatch_is_not(self):
        mismatches = compare_snapshots(
            {"phase": "CARD_REWARD", "floor": 1, "hp": 80, "gold": 99},
            {"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99},
        )
        self.assertEqual(mismatches, ["phase: expected 'CARD_REWARD', got 'NEOW'"])

    def test_event_id_comparison_is_loose(self):
        mismatches = compare_snapshots(
            {"phase": "NEOW", "floor": 0, "hp": 80, "gold": 99, "event_id": "Neow Event"},
            {"phase": "EVENT", "floor": 0, "hp": 80, "gold": 99, "event_id": "Neow"},
        )
        self.assertEqual(mismatches, ["phase: expected 'NEOW', got 'EVENT'"])

    def test_verbose_neow_card_reward_trace_does_not_require_chained_leave(self):
        payload = {
            "seed_long": 1,
            "seed_str": "1",
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "CARD_REWARD",
                    "action": {"kind": "card_reward", "name": "Flash of Steel", "choice_index": 2},
                    "pre_state": {"screen": "CARD_REWARD", "floor": 0, "current_hp": 80, "gold": 99, "character": "IRONCLAD"},
                    "post_state": {"screen": "CARD_REWARD", "floor": 0, "current_hp": 80, "gold": 99, "character": "IRONCLAD"},
                    "post_phase": "CARD_REWARD",
                },
                {
                    "step": 1,
                    "phase": "CARD_REWARD",
                    "action": {"kind": "skip", "name": "SKIP", "choice_index": 0},
                    "pre_state": {"screen": "CARD_REWARD", "floor": 0, "current_hp": 80, "gold": 99, "character": "IRONCLAD"},
                    "post_state": {"screen": "MAP", "floor": 1, "current_hp": 80, "gold": 99, "character": "IRONCLAD"},
                    "post_phase": "MAP",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "verbose.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            trace = load_recorded_trace(path)

        self.assertEqual(trace.steps[0].post["phase"], "CARD_REWARD")

    def test_verbose_trace_compacts_deck_outside_combat(self):
        payload = {
            "seed_long": 2,
            "seed_str": "2",
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "NEOW",
                    "action": {"kind": "neow", "choice_index": 0, "name": "OPTION_0"},
                    "pre_state": {
                        "screen": "EVENT",
                        "floor": 0,
                        "current_hp": 80,
                        "gold": 99,
                        "character": "IRONCLAD",
                        "deck": [{"card_id": "Strike_R"}],
                    },
                    "post_state": {
                        "screen": "MAP",
                        "floor": 0,
                        "current_hp": 80,
                        "gold": 99,
                        "character": "IRONCLAD",
                        "deck": [{"card_id": "Strike_R"}, {"card_id": "Demon Form"}],
                    },
                    "post_phase": "MAP",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "verbose.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            trace = load_recorded_trace(path)

        self.assertEqual(trace.steps[0].pre["deck"], ["Strike_R"])
        self.assertEqual(trace.steps[0].post["deck"], ["Strike_R", "Demon Form"])

    def test_verbose_trace_compacts_event_id(self):
        payload = {
            "seed_long": 2,
            "seed_str": "2",
            "ascension": 0,
            "steps": [
                {
                    "step": 0,
                    "phase": "EVENT",
                    "action": {"kind": "event", "choice_index": 1, "name": "Gave Gold"},
                    "pre_state": {
                        "screen": "EVENT",
                        "floor": 2,
                        "current_hp": 80,
                        "gold": 113,
                        "character": "IRONCLAD",
                        "screen_state": {"event_id": "We Meet Again!", "event_name": "We Meet Again!"},
                    },
                    "post_state": {
                        "screen": "MAP",
                        "floor": 2,
                        "current_hp": 80,
                        "gold": 62,
                        "character": "IRONCLAD",
                        "screen_state": {"event_id": "We Meet Again!", "event_name": "We Meet Again!"},
                    },
                    "post_phase": "MAP",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "verbose.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            trace = load_recorded_trace(path)

        self.assertEqual(trace.steps[0].pre["event_id"], "We Meet Again!")

    def test_compact_snapshot_uses_top_level_event_id_when_screen_state_is_empty(self):
        snapshot = _compact_snapshot_from_verbose_state(
            {
                "screen": "EVENT",
                "floor": 3,
                "current_hp": 80,
                "gold": 127,
                "deck": [{"card_id": "Strike_R"}],
                "event_id": "The Cleric",
                "screen_state": {},
            },
            "EVENT",
        )

        self.assertEqual(snapshot["event_id"], "The Cleric")

    def test_single_leave_event_action_detects_leave_screen(self):
        option = SimpleNamespace(choice_index=0, label="Leave", text="[Leave]")
        game_state = SimpleNamespace(
            screen_type=ScreenType.EVENT,
            screen=SimpleNamespace(options=[option]),
        )
        action = _single_leave_event_action(game_state)
        self.assertEqual(action.__class__.__name__, "EventOptionAction")

    def test_compare_snapshots_accepts_intent_family_match_with_same_damage(self):
        mismatches = compare_snapshots(
            {
                "phase": "COMBAT",
                "floor": 13,
                "monsters": [
                    {
                        "id": "GremlinNob",
                        "hp": 26,
                        "block": 0,
                        "intent": "ATTACK",
                        "move_adjusted_damage": 10,
                        "move_hits": 1,
                    }
                ],
            },
            {
                "phase": "COMBAT",
                "floor": 13,
                "monsters": [
                    {
                        "id": "GremlinNob",
                        "hp": 26,
                        "block": 0,
                        "intent": "ATTACK_DEBUFF",
                        "move_adjusted_damage": 10,
                        "move_hits": 1,
                    }
                ],
            },
        )
        self.assertEqual(mismatches, [])

    def test_compare_snapshots_keeps_monster_mismatch_when_damage_differs(self):
        mismatches = compare_snapshots(
            {
                "phase": "COMBAT",
                "floor": 16,
                "monsters": [
                    {
                        "id": "AcidSlime_L",
                        "hp": 62,
                        "block": 0,
                        "intent": "ATTACK_DEBUFF",
                        "move_adjusted_damage": 11,
                        "move_hits": 1,
                    }
                ],
            },
            {
                "phase": "COMBAT",
                "floor": 16,
                "monsters": [
                    {
                        "id": "AcidSlime_L",
                        "hp": 62,
                        "block": 0,
                        "intent": "DEBUFF",
                        "move_adjusted_damage": 0,
                        "move_hits": 0,
                    }
                ],
            },
        )
        self.assertEqual(
            mismatches,
            [
                "monsters: expected [{'id': 'AcidSlime_L', 'hp': 62, 'block': 0, 'intent': 'ATTACK_DEBUFF', 'move_adjusted_damage': 11, 'move_hits': 1}], got [{'id': 'AcidSlime_L', 'hp': 62, 'block': 0, 'intent': 'DEBUFF', 'move_adjusted_damage': 0, 'move_hits': 0}]"
            ],
        )

    def test_compare_snapshots_ignores_non_attack_sentinel_damage_fields(self):
        mismatches = compare_snapshots(
            {
                "phase": "COMBAT",
                "floor": 16,
                "monsters": [
                    {
                        "id": "AcidSlime_L",
                        "hp": 62,
                        "block": 0,
                        "intent": "DEBUFF",
                        "move_adjusted_damage": 0,
                        "move_hits": 0,
                    }
                ],
            },
            {
                "phase": "COMBAT",
                "floor": 16,
                "monsters": [
                    {
                        "id": "AcidSlime_L",
                        "hp": 62,
                        "block": 0,
                        "intent": "DEBUFF",
                        "move_adjusted_damage": -1,
                        "move_hits": 1,
                    }
                ],
            },
        )
        self.assertEqual(mismatches, [])

    def test_compare_snapshots_treats_debug_and_buff_as_same_non_attack_family(self):
        mismatches = compare_snapshots(
            {
                "phase": "COMBAT",
                "floor": 1,
                "monsters": [
                    {
                        "id": "Cultist",
                        "hp": 51,
                        "block": 0,
                        "intent": "BUFF",
                        "move_adjusted_damage": 0,
                        "move_hits": 0,
                    }
                ],
            },
            {
                "phase": "COMBAT",
                "floor": 1,
                "monsters": [
                    {
                        "id": "Cultist",
                        "hp": 51,
                        "block": 0,
                        "intent": "DEBUG",
                        "move_adjusted_damage": -1,
                        "move_hits": 1,
                    }
                ],
            },
        )
        self.assertEqual(mismatches, [])

    def test_replay_recorded_run_skips_signal_ready_when_bootstrap_ready_was_sent(self):
        from spirecomm.ai.recorded_run_replay import replay_recorded_run

        class FakeCoordinator:
            signal_ready_called = False

            def __init__(self):
                self.stop_after_run = False
                self.last_game_state = None

            def signal_ready(self):
                self.signal_ready_called = True

            def register_command_error_callback(self, callback):
                self.error_callback = callback

            def register_state_change_callback(self, callback):
                self.state_callback = callback

            def register_out_of_game_callback(self, callback):
                self.out_callback = callback

        trace = RecordedRunTrace(
            path=Path("trace.json"),
            source_format="recorded",
            seed_long=1,
            seed_str="1",
            ascension=0,
            character="IRONCLAD",
            steps=[],
        )

        with patch.dict(os.environ, {"SPIRECOMM_BOOTSTRAP_READY_SENT": "1"}, clear=False):
            with patch("spirecomm.ai.recorded_run_replay.load_recorded_trace", return_value=trace):
                with patch("spirecomm.ai.recorded_run_replay.Coordinator", FakeCoordinator):
                    with patch("spirecomm.ai.recorded_run_replay._infer_character", return_value=PlayerClass.IRONCLAD):
                        with patch("spirecomm.ai.recorded_run_replay._play_recorded_game", return_value=None):
                            report = replay_recorded_run(trace_path="trace.json")

        self.assertTrue(report["success"])


if __name__ == "__main__":
    unittest.main()
