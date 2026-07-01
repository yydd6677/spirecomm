from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from spirecomm.ai.run_choice_model import CHOICE_DIM, choice_vector, option_token
from unittest.mock import patch

from spirecomm.ai.runtime_decision import (
    ModelRequiredDecisionError,
    _event_rollout_allows_candidate_action,
    _event_rollout_model_score_drop_limit,
    choose_model_required_action,
    choose_modeled_action,
    choose_reward_screen_action,
    source_is_allowed_for_model_required,
    validate_model_required_selectors,
)
from spirecomm.native_sim_v2 import NativeRunEnv


class _DummyCardSelector:
    def __init__(self, choice_index: int = 0, scores: list[float] | None = None):
        self.available = True
        self.choice_index = choice_index
        self.scores = list(scores or [0.9, 0.1])

    def choose(self, _state, _reward_cards, can_skip=True, **_kwargs):
        return {"choice_index": self.choice_index, "scores": list(self.scores), "can_skip": can_skip}


class _DummyChoiceSelector:
    def __init__(self, choice_index: int | None = 0, scores: list[float] | None = None, available: bool = True):
        self.available = available
        self.choice_index = choice_index
        self.scores = list(scores or [0.1, 0.9])
        self.checkpoint_path = Path("/tmp/dummy.pt")

    def choose(self, _state, _candidates, **_kwargs):
        if self.choice_index is None:
            return None
        return {"choice_index": self.choice_index, "scores": list(self.scores)}


class _DummyCombatSelector:
    def __init__(self, chosen=None, scores: list[float] | None = None, available: bool = True):
        self.available = available
        self.chosen = chosen
        self.scores = list(scores or [0.0, 0.0])
        self.checkpoint_path = Path("/tmp/combat.pt")

    def choose(self, _state, _actions):
        return self.chosen, list(self.scores)


class _DummyEnv:
    def __init__(self, *, actions, potions=None, state=None, phase="CARD_REWARD", current_card_select=None):
        self._actions = list(actions)
        self.potions = list(potions or [])
        self._state = dict(state or {"floor": 1, "screen": phase})
        self.phase = phase
        self.current_card_select = current_card_select

    def legal_actions(self):
        return list(self._actions)

    def state(self):
        return dict(self._state)


def _linear_map(symbols: list[str]):
    rows = []
    for y, symbol in enumerate(symbols):
        rows.append([SimpleNamespace(x=0, y=y, room_symbol=symbol, has_emerald_key=False, edges=[])])
    for y in range(len(rows) - 1):
        rows[y][0].edges = [SimpleNamespace(dst_x=0, dst_y=y + 1)]
    return rows


class RuntimeDecisionTest(unittest.TestCase):
    def test_event_rollout_model_score_drop_supports_per_event_limits(self):
        with patch.dict(
            "os.environ",
            {
                "SPIRECOMM_EVENT_ROLLOUT_MAX_MODEL_SCORE_DROP": "0.5",
                "SPIRECOMM_EVENT_ROLLOUT_MAX_MODEL_SCORE_DROP_EVENTS": "The Library",
                "SPIRECOMM_EVENT_ROLLOUT_MAX_MODEL_SCORE_DROP_BY_EVENT": (
                    "WeMeetAgain=1.0; Vampires:1.5, invalid=not-a-number"
                ),
            },
            clear=True,
        ):
            self.assertEqual(_event_rollout_model_score_drop_limit({"wemeetagain"}), 1.0)
            self.assertEqual(_event_rollout_model_score_drop_limit({"vampires"}), 1.5)
            self.assertEqual(_event_rollout_model_score_drop_limit({"thelibrary"}), 0.5)
            self.assertEqual(_event_rollout_model_score_drop_limit({"bigfish"}), float("inf"))

    def test_event_rollout_wemeetagain_blocks_give_card_override(self):
        chosen_gold = {"kind": "event", "event_id": "WeMeetAgain", "name": "Pay 135 Gold"}
        chosen_potion = {"kind": "event", "event_id": "WeMeetAgain", "name": "Give Potion"}
        chosen_card = {"kind": "event", "event_id": "WeMeetAgain", "name": "Give Card"}
        give_card = {"kind": "event", "event_id": "WeMeetAgain", "name": "Give Card"}
        pay_gold = {"kind": "event", "event_id": "WeMeetAgain", "name": "Pay 135 Gold"}

        with patch.dict(
            "os.environ",
            {"SPIRECOMM_EVENT_ROLLOUT_WEMEETAGAIN_BLOCK_GIVE_CARD_OVERRIDE": "1"},
            clear=True,
        ):
            self.assertFalse(_event_rollout_allows_candidate_action(chosen_gold, give_card))
            self.assertFalse(_event_rollout_allows_candidate_action(chosen_potion, give_card))
            self.assertTrue(_event_rollout_allows_candidate_action(chosen_card, give_card))
            self.assertTrue(_event_rollout_allows_candidate_action(chosen_card, pay_gold))

        with patch.dict(
            "os.environ",
            {"SPIRECOMM_EVENT_ROLLOUT_WEMEETAGAIN_BLOCK_PAY_GOLD_TO_GIVE_CARD_OVERRIDE": "1"},
            clear=True,
        ):
            self.assertFalse(_event_rollout_allows_candidate_action(chosen_gold, give_card))
            self.assertTrue(_event_rollout_allows_candidate_action(chosen_potion, give_card))
            self.assertTrue(_event_rollout_allows_candidate_action(chosen_card, give_card))
            self.assertTrue(_event_rollout_allows_candidate_action(chosen_card, pay_gold))

        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(_event_rollout_allows_candidate_action(chosen_gold, give_card))

    def test_reward_policy_collects_gold_before_other_rewards(self):
        env = _DummyEnv(
            actions=[
                {"kind": "reward_gold", "name": "GOLD", "amount": 18},
                {"kind": "reward_relic", "name": "Mercury Hourglass", "relic": {"relic_id": "Mercury Hourglass"}},
                {"kind": "card_reward", "name": "Pommel Strike", "card": {"card_id": "Pommel Strike"}},
                {"kind": "skip", "name": "SKIP"},
            ],
            potions=[SimpleNamespace(can_use=True)],
        )

        action, scores, source = choose_reward_screen_action(env, _DummyCardSelector())

        self.assertEqual(action["kind"], "reward_gold")
        self.assertEqual(scores, [])
        self.assertEqual(source, "reward_policy_collect_gold")

    def test_reward_policy_prefers_relic_over_key_tradeoff(self):
        env = _DummyEnv(
            actions=[
                {"kind": "reward_relic", "name": "Mercury Hourglass", "relic": {"relic_id": "Mercury Hourglass"}},
                {"kind": "reward_key", "name": "EMERALD_KEY", "key": "emerald"},
                {"kind": "skip", "name": "SKIP"},
            ],
            potions=[SimpleNamespace(can_use=True)],
        )

        action, _, source = choose_reward_screen_action(env, None)

        self.assertEqual(action["kind"], "reward_relic")
        self.assertEqual(source, "reward_policy_relic_over_key")

    def test_reward_policy_collects_available_potion_before_key(self):
        env = _DummyEnv(
            actions=[
                {"kind": "reward_key", "name": "EMERALD_KEY", "key": "emerald"},
                {"kind": "reward_potion", "name": "Fire Potion", "potion_id": "Fire Potion"},
                {"kind": "proceed", "name": "PROCEED"},
            ],
            potions=[
                {"potion_id": "Potion Slot"},
                {"potion_id": "Potion Slot"},
                {"potion_id": "Potion Slot"},
            ],
        )

        action, _, source = choose_reward_screen_action(env, None)

        self.assertEqual(action["kind"], "reward_potion")
        self.assertEqual(action["potion_id"], "Fire Potion")
        self.assertEqual(source, "reward_policy_collect_potion")

    def test_reward_policy_takes_key_before_unavailable_potion(self):
        env = _DummyEnv(
            actions=[
                {"kind": "reward_key", "name": "EMERALD_KEY", "key": "emerald"},
                {"kind": "reward_potion", "name": "Fire Potion", "potion_id": "Fire Potion"},
                {"kind": "proceed", "name": "PROCEED"},
            ],
            potions=[
                {"potion_id": "Strength Potion"},
                {"potion_id": "Dexterity Potion"},
                {"potion_id": "Fire Potion"},
            ],
        )

        action, _, source = choose_reward_screen_action(env, None)

        self.assertEqual(action["kind"], "reward_key")
        self.assertEqual(source, "reward_policy_take_key")

    def test_reward_policy_uses_card_selector_once_non_card_rewards_are_consumed(self):
        env = _DummyEnv(
            actions=[
                {"kind": "card_reward", "name": "Anger", "card": {"card_id": "Anger"}},
                {"kind": "card_reward", "name": "Flex", "card": {"card_id": "Flex"}},
                {"kind": "skip", "name": "SKIP"},
            ],
            potions=[SimpleNamespace(can_use=True)],
        )
        selector = _DummyCardSelector(choice_index=1, scores=[0.2, 0.8])

        action, scores, source = choose_reward_screen_action(env, selector)

        self.assertEqual(action["name"], "Flex")
        self.assertEqual(scores, [0.2, 0.8])
        self.assertEqual(source, "card_reward")

    def test_model_required_card_reward_masks_skip_when_skip_is_not_legal(self):
        class Selector(_DummyCardSelector):
            def __init__(self):
                super().__init__(choice_index=1, scores=[0.2, 0.8])
                self.can_skip_seen = None

            def choose(self, _state, _reward_cards, can_skip=True, **_kwargs):
                self.can_skip_seen = can_skip
                return super().choose(_state, _reward_cards, can_skip=can_skip)

        env = _DummyEnv(
            actions=[
                {"kind": "card_reward", "name": "Anger", "card": {"card_id": "Anger"}},
                {"kind": "card_reward", "name": "Flex", "card": {"card_id": "Flex"}},
            ],
            potions=[SimpleNamespace(can_use=True)],
        )
        selector = Selector()

        action, scores, source = choose_model_required_action(env, {"card_reward": selector})

        self.assertFalse(selector.can_skip_seen)
        self.assertEqual(action["name"], "Flex")
        self.assertEqual(scores, [0.2, 0.8])
        self.assertEqual(source, "card_reward")

    def test_reward_policy_falls_back_to_first_card_when_no_selector_is_available(self):
        env = _DummyEnv(
            actions=[
                {"kind": "card_reward", "name": "Anger", "card": {"card_id": "Anger"}},
                {"kind": "card_reward", "name": "Flex", "card": {"card_id": "Flex"}},
                {"kind": "skip", "name": "SKIP"},
            ],
            potions=[SimpleNamespace(can_use=True)],
        )

        action, scores, source = choose_reward_screen_action(env, None)

        self.assertEqual(action["name"], "Anger")
        self.assertEqual(scores, [])
        self.assertEqual(source, "card_reward_fallback_first")

    def test_reward_policy_skips_full_potion_when_no_other_reward_remains(self):
        env = _DummyEnv(
            actions=[
                {"kind": "reward_potion", "name": "Flex Potion", "potion_id": "FlexPotion"},
                {"kind": "skip", "name": "SKIP"},
            ],
            potions=[SimpleNamespace(can_use=True), SimpleNamespace(can_use=True), SimpleNamespace(can_use=True)],
        )

        with patch.dict("os.environ", {"SPIRECOMM_REWARD_POTION_FULL_REPLACE": "1"}):
            action, _, source = choose_reward_screen_action(env, None)

        self.assertEqual(action["kind"], "skip")
        self.assertEqual(source, "reward_policy_skip_potion_full")

    def test_reward_policy_can_disable_full_potion_replacement(self):
        env = _DummyEnv(
            actions=[
                {"kind": "reward_potion", "name": "Strength Potion", "potion_id": "Strength Potion"},
                {"kind": "skip", "name": "SKIP"},
            ],
            potions=[
                {"potion_id": "Fire Potion"},
                {"potion_id": "Explosive Potion"},
                {"potion_id": "SteroidPotion"},
            ],
        )

        with patch.dict("os.environ", {"SPIRECOMM_REWARD_POTION_FULL_REPLACE": "0"}):
            action, _, source = choose_reward_screen_action(env, None)

        self.assertEqual(action["kind"], "skip")
        self.assertEqual(source, "reward_policy_skip_potion_full")

    def test_reward_policy_replaces_low_priority_full_potion(self):
        env = _DummyEnv(
            actions=[
                {"kind": "reward_potion", "name": "Strength Potion", "potion_id": "Strength Potion"},
                {"kind": "skip", "name": "SKIP"},
            ],
            potions=[
                {"potion_id": "Fire Potion"},
                {"potion_id": "Explosive Potion"},
                {"potion_id": "SteroidPotion"},
            ],
        )

        action, _, source = choose_reward_screen_action(env, None)

        self.assertEqual(action["kind"], "reward_potion")
        self.assertEqual(action["potion_id"], "Strength Potion")
        self.assertEqual(source, "reward_policy_replace_potion_full")

    def test_reward_policy_does_not_reopen_declined_card_reward(self):
        env = _DummyEnv(
            actions=[
                {"kind": "reward_potion", "name": "Speed Potion", "potion_id": "SpeedPotion"},
                {"kind": "raw", "name": "CARD", "label": "CARD", "choice_index": 1},
                {"kind": "proceed", "name": "PROCEED"},
            ],
            potions=[
                {"potion_id": "BlessingOfTheForge"},
                {"potion_id": "Strength Potion"},
                {"potion_id": "Dexterity Potion"},
            ],
        )
        env.reward_card_reward_declined = True

        action, _, source = choose_reward_screen_action(env, _DummyCardSelector())

        self.assertEqual(action["kind"], "proceed")
        self.assertEqual(source, "reward_policy_skip_potion_full")

    def test_choose_modeled_action_surfaces_reward_policy_source(self):
        env = _DummyEnv(
            actions=[
                {"kind": "reward_gold", "name": "GOLD", "amount": 15},
                {"kind": "card_reward", "name": "Anger", "card": {"card_id": "Anger"}},
                {"kind": "skip", "name": "SKIP"},
            ],
            potions=[SimpleNamespace(can_use=True)],
        )

        action, _, source = choose_modeled_action(env, {"card_reward": _DummyCardSelector()})

        self.assertEqual(action["kind"], "reward_gold")
        self.assertEqual(source, "reward_policy_collect_gold")

    def test_model_required_setup_fails_without_torch(self):
        with patch("spirecomm.ai.runtime_decision.torch", None):
            with self.assertRaises(ModelRequiredDecisionError) as cm:
                validate_model_required_selectors({})

        self.assertEqual(cm.exception.phase, "SETUP")
        self.assertEqual(cm.exception.reason, "torch_unavailable_use_spirecomm_rl_python")

    def test_model_required_map_multichoice_uses_dynamic_programming(self):
        env = _DummyEnv(
            phase="MAP",
            actions=[
                {"kind": "map", "choice_index": 0, "symbol": "M"},
                {"kind": "map", "choice_index": 1, "symbol": "?"},
            ],
        )

        action, scores, source = choose_model_required_action(env, {"map": _DummyChoiceSelector(available=False)})

        self.assertEqual(action["symbol"], "M")
        self.assertEqual(scores, [0.0, 0.0])
        self.assertEqual(source, "map_dp")

    def test_model_required_allows_single_forced_action(self):
        env = _DummyEnv(phase="MAP", actions=[{"kind": "map", "choice_index": 0, "symbol": "M"}])

        action, scores, source = choose_model_required_action(env, {})

        self.assertEqual(action["symbol"], "M")
        self.assertEqual(scores, [0.0])
        self.assertEqual(source, "map_dp")

    def test_model_required_neow_uses_weighted_source(self):
        env = _DummyEnv(
            phase="NEOW",
            actions=[
                {"kind": "neow", "choice_index": 0, "bonus": "ONE_RANDOM_RARE_CARD", "drawback": "NONE"},
                {"kind": "neow", "choice_index": 1, "bonus": "RANDOM_COMMON_RELIC", "drawback": "NONE"},
            ],
        )

        with patch("spirecomm.ai.runtime_decision._stable_neow_policy_random", return_value=0.99):
            action, scores, source = choose_model_required_action(env, {})

        self.assertEqual(action["choice_index"], 1)
        self.assertAlmostEqual(sum(scores), 1.0)
        self.assertGreater(scores[1], scores[0])
        self.assertEqual(source, "neow_weighted")

    def test_model_required_neow_continue_is_forced_single(self):
        env = _DummyEnv(
            phase="NEOW",
            actions=[
                {"kind": "neow", "choice_index": 0, "bonus": "CONTINUE", "drawback": "NONE"},
            ],
        )

        action, scores, source = choose_model_required_action(env, {})

        self.assertEqual(action["choice_index"], 0)
        self.assertEqual(scores, [])
        self.assertEqual(source, "forced_single")

    def test_model_required_neow_default_pool_excludes_choice_two_and_three(self):
        env = _DummyEnv(
            phase="NEOW",
            actions=[
                {"kind": "neow", "choice_index": 0, "bonus": "RANDOM_COLORLESS", "drawback": "NONE"},
                {"kind": "neow", "choice_index": 1, "bonus": "THREE_CARDS", "drawback": "NONE"},
                {"kind": "neow", "choice_index": 2, "bonus": "THREE_ENEMY_KILL", "drawback": "NONE"},
                {"kind": "neow", "choice_index": 3, "bonus": "BOSS_RELIC", "drawback": "NONE"},
            ],
        )

        with patch("spirecomm.ai.runtime_decision._stable_neow_policy_random", return_value=0.99):
            action, scores, source = choose_model_required_action(env, {})

        self.assertEqual(action["choice_index"], 1)
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(sum(scores), 1.0)
        self.assertEqual(source, "neow_weighted")

    def test_model_required_card_reward_without_selector_fails(self):
        env = _DummyEnv(
            actions=[
                {"kind": "card_reward", "name": "Anger", "card": {"card_id": "Anger"}},
                {"kind": "card_reward", "name": "Flex", "card": {"card_id": "Flex"}},
                {"kind": "skip", "name": "SKIP"},
            ],
            potions=[SimpleNamespace(can_use=True)],
        )

        with self.assertRaises(ModelRequiredDecisionError) as cm:
            choose_model_required_action(env, {"card_reward": None})

        self.assertEqual(cm.exception.phase, "CARD_REWARD")
        self.assertEqual(cm.exception.selector_name, "card_reward")

    def test_model_required_reward_relic_key_tradeoff_prefers_relic(self):
        env = _DummyEnv(
            actions=[
                {"kind": "reward_relic", "name": "Vajra", "relic": {"relic_id": "Vajra"}},
                {"kind": "reward_key", "name": "SAPPHIRE_KEY", "key": "sapphire"},
                {"kind": "proceed", "name": "PROCEED"},
            ],
            potions=[SimpleNamespace(can_use=True)],
        )

        action, scores, source = choose_model_required_action(
            env,
            {"card_reward": _DummyCardSelector(), "boss_relic": _DummyChoiceSelector(choice_index=1, scores=[0.2, 0.8])},
        )

        self.assertEqual(action["kind"], "reward_relic")
        self.assertEqual(scores, [])
        self.assertEqual(source, "reward_policy_relic_over_key")

    def test_model_required_combat_selector_no_choice_fails(self):
        env = _DummyEnv(
            phase="COMBAT",
            actions=[
                {"kind": "card", "name": "Strike"},
                {"kind": "end", "name": "END_TURN"},
            ],
        )

        with self.assertRaises(ModelRequiredDecisionError) as cm:
            choose_model_required_action(env, {"combat": _DummyCombatSelector(chosen=None)})

        self.assertEqual(cm.exception.phase, "COMBAT")
        self.assertEqual(cm.exception.selector_name, "combat")
        self.assertEqual(cm.exception.reason, "combat_selector_returned_no_choice")

    def test_model_required_card_select_uses_mode_from_legal_action(self):
        env = _DummyEnv(
            phase="CARD_SELECT",
            current_card_select={"mode": "HEADBUTT"},
            actions=[
                {"kind": "card_select", "mode": "HEADBUTT", "name": "Strike_R", "card_id": "Strike_R"},
                {"kind": "card_select", "mode": "HEADBUTT", "name": "Pommel Strike", "card_id": "Pommel Strike"},
            ],
        )

        action, scores, source = choose_model_required_action(
            env,
            {"card_reward": _DummyCardSelector(choice_index=1, scores=[0.1, 0.9])},
        )

        self.assertEqual(action["name"], "Pommel Strike")
        self.assertEqual(scores, [0.1, 0.9])
        self.assertEqual(source, "headbutt_target")

    def test_model_required_card_select_supports_true_grit_mode(self):
        env = _DummyEnv(
            phase="CARD_SELECT",
            actions=[
                {"kind": "card_select", "mode": "TRUE_GRIT", "name": "Strike_R", "card_id": "Strike_R"},
                {"kind": "card_select", "mode": "TRUE_GRIT", "name": "Defend_R", "card_id": "Defend_R"},
            ],
        )

        action, scores, source = choose_model_required_action(
            env,
            {"purge_target": _DummyChoiceSelector(choice_index=1, scores=[0.25, 0.75])},
        )

        self.assertEqual(action["name"], "Defend_R")
        self.assertEqual(scores, [0.25, 0.75])
        self.assertEqual(source, "true_grit_target")

    def test_model_required_card_select_supports_warcry_mode(self):
        env = _DummyEnv(
            phase="CARD_SELECT",
            actions=[
                {"kind": "card_select", "mode": "WARCRY", "name": "Strike_R", "card_id": "Strike_R"},
                {"kind": "card_select", "mode": "WARCRY", "name": "Defend_R", "card_id": "Defend_R"},
            ],
        )

        action, scores, source = choose_model_required_action(
            env,
            {"purge_target": _DummyChoiceSelector(choice_index=1, scores=[0.25, 0.75])},
        )

        self.assertEqual(action["name"], "Defend_R")
        self.assertEqual(scores, [0.25, 0.75])
        self.assertEqual(source, "warcry_target")

    def test_model_required_card_select_supports_prepared_mode(self):
        env = _DummyEnv(
            phase="CARD_SELECT",
            actions=[
                {"kind": "card_select", "mode": "PREPARED", "name": "Strike_G", "card_id": "Strike_G"},
                {"kind": "card_select", "mode": "PREPARED", "name": "Defend_G", "card_id": "Defend_G"},
            ],
        )

        action, scores, source = choose_model_required_action(
            env,
            {"purge_target": _DummyChoiceSelector(choice_index=1, scores=[0.25, 0.75])},
        )

        self.assertEqual(action["name"], "Defend_G")
        self.assertEqual(scores, [0.25, 0.75])
        self.assertEqual(source, "prepared_target")

    def test_model_required_gambling_chip_can_select_discard_before_confirm(self):
        env = _DummyEnv(
            phase="CARD_SELECT",
            actions=[
                {"kind": "card_select", "mode": "GAMBLING_CHIP", "name": "Strike_G", "card_id": "Strike_G"},
                {"kind": "card_select", "mode": "GAMBLING_CHIP", "name": "Defend_G", "card_id": "Defend_G"},
                {"kind": "confirm", "mode": "GAMBLING_CHIP", "name": "CONFIRM"},
            ],
        )

        action, scores, source = choose_model_required_action(
            env,
            {"purge_target": _DummyChoiceSelector(choice_index=1, scores=[0.1, 0.9, 0.0])},
        )

        self.assertEqual(action["kind"], "card_select")
        self.assertEqual(action["name"], "Defend_G")
        self.assertEqual(scores, [0.1, 0.9, 0.0])
        self.assertEqual(source, "gambling_chip_target")

    def test_model_required_gambling_chip_can_choose_confirm_when_selector_prefers_it(self):
        env = _DummyEnv(
            phase="CARD_SELECT",
            actions=[
                {"kind": "card_select", "mode": "GAMBLING_CHIP", "name": "Strike_G", "card_id": "Strike_G"},
                {"kind": "card_select", "mode": "GAMBLING_CHIP", "name": "Defend_G", "card_id": "Defend_G"},
                {"kind": "confirm", "mode": "GAMBLING_CHIP", "name": "CONFIRM"},
            ],
        )

        action, scores, source = choose_model_required_action(
            env,
            {"purge_target": _DummyChoiceSelector(choice_index=2, scores=[0.1, 0.0, 0.9])},
        )

        self.assertEqual(action["kind"], "confirm")
        self.assertEqual(scores, [0.1, 0.0, 0.9])
        self.assertEqual(source, "gambling_chip_target")

    def test_model_required_elixir_can_select_exhaust_before_confirm(self):
        env = _DummyEnv(
            phase="CARD_SELECT",
            actions=[
                {"kind": "card_select", "mode": "ELIXIR", "name": "Strike_R", "card_id": "Strike_R"},
                {"kind": "card_select", "mode": "ELIXIR", "name": "Defend_R", "card_id": "Defend_R"},
                {"kind": "confirm", "mode": "ELIXIR", "name": "CONFIRM"},
            ],
        )

        action, scores, source = choose_model_required_action(
            env,
            {"purge_target": _DummyChoiceSelector(choice_index=1, scores=[0.1, 0.9, 0.0])},
        )

        self.assertEqual(action["kind"], "card_select")
        self.assertEqual(action["name"], "Defend_R")
        self.assertEqual(scores, [0.1, 0.9, 0.0])
        self.assertEqual(source, "elixir_target")

    def test_model_required_elixir_can_choose_confirm_when_selector_prefers_it(self):
        env = _DummyEnv(
            phase="CARD_SELECT",
            actions=[
                {"kind": "card_select", "mode": "ELIXIR", "name": "Strike_R", "card_id": "Strike_R"},
                {"kind": "card_select", "mode": "ELIXIR", "name": "Defend_R", "card_id": "Defend_R"},
                {"kind": "confirm", "mode": "ELIXIR", "name": "CONFIRM"},
            ],
        )

        action, scores, source = choose_model_required_action(
            env,
            {"purge_target": _DummyChoiceSelector(choice_index=2, scores=[0.1, 0.0, 0.9])},
        )

        self.assertEqual(action["kind"], "confirm")
        self.assertEqual(scores, [0.1, 0.0, 0.9])
        self.assertEqual(source, "elixir_target")

    def test_model_required_card_select_supports_put_on_deck_mode(self):
        env = _DummyEnv(
            phase="CARD_SELECT",
            actions=[
                {"kind": "card_select", "mode": "PUT_ON_DECK", "name": "Strike_R", "card_id": "Strike_R"},
                {"kind": "card_select", "mode": "PUT_ON_DECK", "name": "Defend_R", "card_id": "Defend_R"},
            ],
        )

        action, scores, source = choose_model_required_action(
            env,
            {"purge_target": _DummyChoiceSelector(choice_index=1, scores=[0.25, 0.75])},
        )

        self.assertEqual(action["name"], "Defend_R")
        self.assertEqual(scores, [0.25, 0.75])
        self.assertEqual(source, "put_on_deck_target")

    def test_model_required_library_card_select_uses_card_reward_selector(self):
        env = _DummyEnv(
            phase="CARD_SELECT",
            actions=[
                {"kind": "card_select", "mode": "library", "name": "Armaments", "card_id": "Armaments"},
                {"kind": "card_select", "mode": "library", "name": "Disarm", "card_id": "Disarm"},
            ],
        )

        selector = _DummyCardSelector(choice_index=1, scores=[0.2, 0.8])
        action, scores, source = choose_model_required_action(env, {"card_reward": selector})

        self.assertEqual(action["name"], "Disarm")
        self.assertEqual(scores, [0.2, 0.8])
        self.assertEqual(source, "library_card_select")

    def test_model_required_potion_discovery_card_select_adds_sqrt_cost_bias(self):
        env = _DummyEnv(
            phase="CARD_SELECT",
            current_card_select={"mode": "DISCOVERY", "potion_id": "AttackPotion"},
            actions=[
                {"kind": "card_reward", "mode": "DISCOVERY", "name": "Anger", "card_id": "Anger", "cost": 0},
                {"kind": "card_reward", "mode": "DISCOVERY", "name": "Carnage", "card_id": "Carnage", "cost": 2},
            ],
        )

        selector = _DummyCardSelector(choice_index=0, scores=[1.0, 0.0])
        action, scores, source = choose_model_required_action(env, {"card_reward": selector})

        self.assertEqual(action["name"], "Carnage")
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 2 ** 0.5)
        self.assertEqual(source, "discovery_target")

    def test_model_required_discovery_card_select_adds_sqrt_cost_bias_even_without_potion_id(self):
        env = _DummyEnv(
            phase="CARD_SELECT",
            current_card_select={"mode": "DISCOVERY", "source": "card"},
            actions=[
                {"kind": "card_reward", "mode": "DISCOVERY", "name": "Anger", "card_id": "Anger", "cost": 0},
                {"kind": "card_reward", "mode": "DISCOVERY", "name": "Carnage", "card_id": "Carnage", "cost": 2},
            ],
        )

        selector = _DummyCardSelector(choice_index=0, scores=[1.0, 0.0])
        action, scores, source = choose_model_required_action(env, {"card_reward": selector})

        self.assertEqual(action["name"], "Carnage")
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 2 ** 0.5)
        self.assertEqual(source, "discovery_target")

    def test_model_required_liquid_memories_card_select_adds_sqrt_cost_bias(self):
        env = _DummyEnv(
            phase="CARD_SELECT",
            current_card_select={"mode": "LIQUID_MEMORIES", "potion_id": "LiquidMemories"},
            actions=[
                {"kind": "card_select", "mode": "LIQUID_MEMORIES", "name": "Strike_R", "card_id": "Strike_R", "cost": 1},
                {"kind": "card_select", "mode": "LIQUID_MEMORIES", "name": "Carnage", "card_id": "Carnage", "cost": 2},
            ],
        )

        selector = _DummyCardSelector(choice_index=0, scores=[0.0, 0.0])
        action, scores, source = choose_model_required_action(env, {"card_reward": selector})

        self.assertEqual(action["name"], "Carnage")
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 2 ** 0.5)
        self.assertEqual(source, "liquid_memories_target")

    def test_model_required_audit_allows_modeled_card_select_sources(self):
        self.assertTrue(source_is_allowed_for_model_required("library_card_select"))
        self.assertTrue(source_is_allowed_for_model_required("prepared_target"))
        self.assertTrue(source_is_allowed_for_model_required("true_grit_target"))
        self.assertTrue(source_is_allowed_for_model_required("warcry_target"))
        self.assertTrue(source_is_allowed_for_model_required("put_on_deck_target"))
        self.assertTrue(source_is_allowed_for_model_required("gambling_chip_target"))
        self.assertTrue(source_is_allowed_for_model_required("elixir_target"))

    def test_model_required_campfire_smith_does_not_inline_upgrade_target(self):
        env = _DummyEnv(
            phase="CAMPFIRE",
            actions=[
                {"kind": "campfire", "name": "rest", "choice_index": 0},
                {"kind": "campfire", "name": "smith", "choice_index": 1},
            ],
        )

        action, scores, source = choose_model_required_action(
            env,
            {
                "campfire": _DummyChoiceSelector(choice_index=1, scores=[0.2, 0.8]),
                "upgrade_target": _DummyChoiceSelector(choice_index=0),
            },
        )

        self.assertEqual(action["name"], "smith")
        self.assertNotIn("target_index", action)
        self.assertEqual(scores, [0.2, 0.8])
        self.assertEqual(source, "campfire")

    def test_shop_future_reserve_danger_gate_can_release_gold_for_current_relic(self):
        env = _DummyEnv(
            phase="SHOP",
            actions=[
                {"kind": "shop", "item_kind": "relic", "item_id": "Bag of Marbles", "name": "Bag of Marbles", "price": 120},
                {"kind": "shop", "item_kind": "leave", "item_id": "leave", "name": "Leave", "price": 0},
            ],
        )
        env.gold = 200
        env.deck = []
        env.map = _linear_map(["$", "E", "$"])
        env.current_map_node = (0, 0)

        with patch.dict(
            "os.environ",
            {
                "SPIRECOMM_SHOP_POLICY": "value",
                "SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE_DANGER_GATE": "1",
                "SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE_DANGER_HORIZON": "3",
                "SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE_DANGER_MULTIPLIER": "0",
            },
        ):
            action, scores, source = choose_model_required_action(
                env,
                {"shop": _DummyChoiceSelector(choice_index=0, scores=[7.0, 0.0])},
            )

        self.assertEqual(action["item_kind"], "relic")
        self.assertEqual(action["item_id"], "Bag of Marbles")
        self.assertEqual(source, "shop_value")
        self.assertGreater(scores[0], scores[1])

    def test_shop_forced_danger_item_bias_can_target_specific_potion(self):
        env = _DummyEnv(
            phase="SHOP",
            actions=[
                {
                    "kind": "shop",
                    "item_kind": "potion",
                    "item_id": "Fire Potion",
                    "potion_id": "Fire Potion",
                    "name": "Fire Potion",
                    "price": 50,
                },
                {"kind": "shop", "item_kind": "leave", "item_id": "leave", "name": "Leave", "price": 0},
            ],
        )
        env.gold = 50
        env.deck = []
        env.map = _linear_map(["$", "E"])
        env.current_map_node = (0, 0)

        with patch.dict(
            "os.environ",
            {
                "SPIRECOMM_SHOP_POLICY": "value",
                "SPIRECOMM_SHOP_VALUE_FORCED_DANGER_ITEM_BIAS_JSON": '{"potion:firepotion": 5.0}',
                "SPIRECOMM_SHOP_VALUE_FORCED_DANGER_ITEM_BIAS_HORIZON": "2",
            },
        ):
            action, scores, source = choose_model_required_action(env, {})

        self.assertEqual(action["item_kind"], "potion")
        self.assertEqual(action["potion_id"], "Fire Potion")
        self.assertEqual(source, "shop_value")
        self.assertGreater(scores[0], scores[1])

    def test_map_choice_vector_keeps_symbol_token_but_changes_extra_features(self):
        left = {"kind": "map", "symbol": "M", "x": 0, "floor": 7, "next_symbols": ["M", "?"], "child_count": 2}
        right = {"kind": "map", "symbol": "M", "x": 3, "floor": 7, "next_symbols": ["$", "R", "E"], "child_count": 3}

        left_vector = choice_vector("map", left)
        right_vector = choice_vector("map", right)

        self.assertEqual(len(left_vector), CHOICE_DIM)
        self.assertEqual(len(right_vector), CHOICE_DIM)
        self.assertEqual(left_vector[:-8], right_vector[:-8], "same-symbol map nodes should keep the same token encoding")
        self.assertNotEqual(left_vector[-8:], right_vector[-8:], "map extra features should distinguish path structure")

    def test_event_choice_token_uses_training_label_aliases(self):
        self.assertEqual(
            option_token(
                "event",
                {"kind": "event", "event_id": "Golden Shrine", "label": "Pray (100 Gold)", "choice_index": 0},
            ),
            option_token(
                "event",
                {"kind": "event", "event_id": "Golden Shrine", "label": "Pray", "choice_index": 0},
            ),
        )
        self.assertEqual(
            option_token(
                "event",
                {"kind": "event", "event_id": "Golden Shrine", "label": "Desecrate (275 Gold + Regret)", "choice_index": 1},
            ),
            option_token(
                "event",
                {"kind": "event", "event_id": "Golden Shrine", "label": "Desecrate", "choice_index": 1},
            ),
        )
        self.assertEqual(
            option_token(
                "event",
                {"kind": "event", "event_id": "Golden Shrine", "label": "Leave", "choice_index": 2},
            ),
            option_token(
                "event",
                {"kind": "event", "event_id": "Golden Shrine", "label": "Ignored", "choice_index": 2},
            ),
        )

    def test_shop_choice_token_matches_training_namespace_for_potions_and_upgraded_cards(self):
        self.assertEqual(
            option_token("shop", {"kind": "shop", "item_kind": "potion", "potion_id": "PowerPotion", "name": "Power Potion"}),
            option_token("shop", {"kind": "shop", "item_kind": "item", "item_id": "PowerPotion", "name": "PowerPotion"}),
        )
        self.assertEqual(
            option_token(
                "shop",
                {
                    "kind": "shop",
                    "item_kind": "card",
                    "item_id": "Shrug It Off",
                    "card_id": "Shrug It Off",
                    "name": "Shrug It Off+",
                    "upgrades": 1,
                },
            ),
            option_token("shop", {"kind": "shop", "item_kind": "item", "item_id": "Shrug It Off+1"}),
        )

    def test_transition_to_next_act_skips_missing_floor_and_builds_act_two_map(self):
        env = NativeRunEnv(seed=8, ascension_level=0, start_on_map=True)
        env.floor = 16
        env.act = 1
        env.phase = "BOSS_RELIC"
        env.current_map_node_id = "boss-floor-16"
        env.current_node_symbol = "BOSS"

        env._transition_to_next_act()

        self.assertEqual(env.act, 2)
        self.assertEqual(env.floor, 17)
        self.assertEqual(env.phase, "MAP")
        actions = env.legal_actions()
        self.assertTrue(actions)
        self.assertTrue(all(int(action["floor"]) == 18 for action in actions))


if __name__ == "__main__":
    unittest.main()
