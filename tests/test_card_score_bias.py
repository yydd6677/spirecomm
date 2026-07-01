from __future__ import annotations

import unittest
from unittest.mock import patch

from spirecomm.ai.card_reward_model import card_score_biases_for_cards


class CardScoreBiasTest(unittest.TestCase):
    def test_default_card_score_bias_is_enabled(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                card_score_biases_for_cards(
                    [
                        {"card_id": "Barricade"},
                        {"card_id": "True Grit"},
                        {"card_id": "Demon Form"},
                        {"card_id": "Pommel Strike"},
                    ]
                ),
                [-1.0, -0.6, 0.6, 0.0],
            )

    def test_card_score_bias_preset_has_two_weak_tiers_and_boost_tier(self) -> None:
        with patch.dict("os.environ", {"SPIRECOMM_CARD_SCORE_BIAS_ENABLED": "1"}, clear=True):
            self.assertEqual(
                card_score_biases_for_cards(
                    [
                        {"card_id": "Barricade"},
                        {"card_id": "True Grit"},
                        {"card_id": "Body Slam"},
                        {"card_id": "Flash of Steel"},
                        {"card_id": "Fire Breathing"},
                        {"card_id": "Combust"},
                        {"card_id": "Berserk"},
                        {"card_id": "Demon Form"},
                        {"card_id": "Metallicize"},
                        {"card_id": "Feel No Pain"},
                        {"card_id": "Brutality"},
                        {"card_id": "Pommel Strike"},
                    ]
                ),
                [-1.0, -0.6, -0.6, -0.6, 0.0, -0.6, -1.0, 0.6, 0.6, 0.6, 0.6, 0.0],
            )

    def test_card_score_bias_json_override(self) -> None:
        with patch.dict(
            "os.environ",
            {"SPIRECOMM_CARD_SCORE_BIAS_JSON": '{"Barricade": -0.25, "True Grit": -0.75}'},
            clear=True,
        ):
            self.assertEqual(
                card_score_biases_for_cards(
                    [
                        {"card_id": "Barricade"},
                        {"card_id": "True Grit"},
                        {"card_id": "Flash of Steel"},
                    ]
                ),
                [-0.25, -0.75, 0.0],
            )

    def test_block_archetype_bias_uses_body_slam_and_barricade_as_half_anchors(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SPIRECOMM_CARD_SCORE_BIAS_ENABLED": "0",
                "SPIRECOMM_CARD_ARCHETYPE_BIAS_ENABLED": "1",
                "SPIRECOMM_CARD_ARCHETYPE_BLOCK_BIAS": "0.6",
            },
            clear=True,
        ):
            candidates = [{"card_id": "Shrug It Off"}, {"card_id": "Impervious"}]
            self.assertEqual(
                card_score_biases_for_cards(candidates, {"deck": [{"card_id": "Body Slam"}]}),
                [0.3, 0.3],
            )
            self.assertEqual(
                card_score_biases_for_cards(
                    candidates,
                    {"deck": [{"card_id": "Body Slam"}, {"card_id": "Barricade"}]},
                ),
                [0.6, 0.6],
            )

    def test_aoe_archetype_bias_only_when_deck_has_at_most_one_aoe(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SPIRECOMM_CARD_SCORE_BIAS_ENABLED": "0",
                "SPIRECOMM_CARD_ARCHETYPE_BIAS_ENABLED": "1",
                "SPIRECOMM_CARD_ARCHETYPE_AOE_BIAS": "0.55",
            },
            clear=True,
        ):
            candidates = [{"card_id": "Cleave"}, {"card_id": "Immolate"}]
            self.assertEqual(card_score_biases_for_cards(candidates, {"deck": []}), [0.55, 0.55])
            self.assertEqual(
                card_score_biases_for_cards(
                    candidates,
                    {"deck": [{"card_id": "Cleave"}, {"card_id": "Whirlwind"}]},
                ),
                [0.0, 0.0],
            )

    def test_exhaust_archetype_bias_uses_dark_embrace_and_feel_no_pain_as_half_anchors(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SPIRECOMM_CARD_SCORE_BIAS_ENABLED": "0",
                "SPIRECOMM_CARD_ARCHETYPE_BIAS_ENABLED": "1",
                "SPIRECOMM_CARD_ARCHETYPE_EXHAUST_BIAS": "0.6",
            },
            clear=True,
        ):
            candidates = [{"card_id": "Burning Pact"}, {"card_id": "Second Wind"}]
            self.assertEqual(
                card_score_biases_for_cards(candidates, {"deck": [{"card_id": "Dark Embrace"}]}),
                [0.3, 0.3],
            )
            self.assertEqual(
                card_score_biases_for_cards(
                    candidates,
                    {"deck": [{"card_id": "Dark Embrace"}, {"card_id": "Feel No Pain"}]},
                ),
                [0.6, 0.6],
            )

    def test_strength_archetype_bias_uses_two_and_three_card_thresholds(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SPIRECOMM_CARD_SCORE_BIAS_ENABLED": "0",
                "SPIRECOMM_CARD_ARCHETYPE_BIAS_ENABLED": "1",
                "SPIRECOMM_CARD_ARCHETYPE_STRENGTH_BIAS": "0.35",
            },
            clear=True,
        ):
            candidates = [{"card_id": "Heavy Blade"}, {"card_id": "Limit Break"}]
            self.assertEqual(
                card_score_biases_for_cards(
                    candidates,
                    {"deck": [{"card_id": "Inflame"}, {"card_id": "Spot Weakness"}]},
                ),
                [0.175, 0.175],
            )
            self.assertEqual(
                card_score_biases_for_cards(
                    candidates,
                    {"deck": [{"card_id": "Inflame"}, {"card_id": "Spot Weakness"}, {"card_id": "Flex"}]},
                ),
                [0.35, 0.35],
            )


if __name__ == "__main__":
    unittest.main()
