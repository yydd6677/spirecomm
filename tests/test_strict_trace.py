from __future__ import annotations

import unittest

from spirecomm.ai.strict_trace import build_strict_action
from spirecomm.ai.strict_trace import normalize_verbose_state_for_strict


class StrictTraceTest(unittest.TestCase):
    def test_build_strict_action_card_reward_prefers_card_index_over_choice_index(self):
        pre_state = {
            "phase": "CARD_REWARD",
            "screen_type": "CARD_REWARD",
            "screen_state": {
                "cards": [
                    {"id": "Intimidate", "name": "Intimidate", "type": "SKILL", "rarity": "UNCOMMON", "upgrades": 0, "cost": 0},
                    {"id": "Battle Trance", "name": "Battle Trance", "type": "SKILL", "rarity": "UNCOMMON", "upgrades": 0, "cost": 0},
                    {"id": "Offering", "name": "Offering", "type": "SKILL", "rarity": "RARE", "upgrades": 0, "cost": 0},
                ]
            },
        }

        strict_action = build_strict_action(
            {
                "kind": "card_reward",
                "choice_index": 1,
                "card_index": 0,
                "card": {"card_id": "Intimidate", "name": "Intimidate"},
            },
            pre_state=pre_state,
            phase="CARD_REWARD",
        )

        self.assertEqual(strict_action["kind"], "choose_by_index")
        self.assertEqual(strict_action["choice_index"], 0)
        self.assertEqual(strict_action["choice"]["card"]["card_id"], "Intimidate")

    def test_build_strict_action_raw_card_reward_matches_visible_card_entry(self):
        pre_state = {
            "phase": "CARD_REWARD",
            "screen_type": "CARD_REWARD",
            "screen_state": {
                "rewards": [
                    {
                        "reward_type": "POTION",
                        "potion": {"id": "FearPotion", "name": "Fear Potion", "requires_target": True},
                    },
                    {"reward_type": "CARD"},
                ]
            },
        }

        strict_action = build_strict_action(
            {
                "kind": "raw",
                "name": "CARD",
                "label": "CARD",
                "choice_index": 4,
                "reward_index": 0,
            },
            pre_state=pre_state,
            phase="CARD_REWARD",
        )

        self.assertEqual(strict_action["kind"], "choose_by_index")
        self.assertEqual(strict_action["choice_index"], 1)
        self.assertEqual(strict_action["choice"]["label"], "CARD")

    def test_normalize_neow_regular_card_reward_text_as_three_cards(self):
        normalized = normalize_verbose_state_for_strict(
            {
                "phase": "NEOW",
                "screen_type": "EVENT",
                "room_type": "NeowRoom",
                "current_hp": 80,
                "max_hp": 80,
                "gold": 99,
                "screen_state": {
                    "options": [
                        {
                            "choice_index": 0,
                            "label": "Choose a Card to obtain",
                            "text": "[ Choose a Card to obtain ]",
                        }
                    ]
                },
            },
            phase_hint="NEOW",
        )

        self.assertEqual(normalized["screen_state"]["choices"][0]["bonus"], "THREE_CARDS")

    def test_normalize_omits_zero_relic_counter(self):
        normalized = normalize_verbose_state_for_strict(
            {
                "phase": "CARD_REWARD",
                "screen_type": "CARD_REWARD",
                "current_hp": 53,
                "max_hp": 80,
                "gold": 32,
                "relics": [
                    {"relic_id": "Ornamental Fan", "name": "Ornamental Fan", "counter": 0},
                    {"relic_id": "Happy Flower", "name": "Happy Flower", "counter": 2},
                ],
            },
            phase_hint="CARD_REWARD",
        )

        self.assertEqual(normalized["relics"][0], {"relic_id": "Ornamental Fan", "name": "Ornamental Fan"})
        self.assertEqual(normalized["relics"][1]["counter"], 2)

    def test_hand_select_is_confirm_up_when_required_selection_is_full(self):
        normalized = normalize_verbose_state_for_strict(
            {
                "phase": "CARD_SELECT",
                "screen_type": "HAND_SELECT",
                "screen_state": {
                    "max_cards": 1,
                    "selected": [{"id": "Strike_R", "name": "Strike", "type": "ATTACK"}],
                    "hand": [{"id": "Strike_R", "name": "Strike", "type": "ATTACK"}],
                },
            },
            phase_hint="CARD_SELECT",
        )

        self.assertTrue(normalized["screen_state"]["confirm_up"])
        self.assertEqual(normalized["screen_state"]["num_cards"], 1)


if __name__ == "__main__":
    unittest.main()
