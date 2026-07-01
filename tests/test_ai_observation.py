import unittest

from spirecomm.ai.card_reward_model import STATE_DIM, build_state_vector
from spirecomm.ai.lightspeed_combat_model import SerializedCombatSelector, V2CombatSelector
from spirecomm.ai.observation import canonicalize_serialized_state
from spirecomm.ai.recording import serialize_card
from spirecomm.ai.rl import MONSTER_FEATURE_DIM, MONSTER_POWER_SPECS, build_extended_combat_tensors, build_state_tensors
from spirecomm.ai.run_choice_model import CHOICE_DIM, choice_vector
from spirecomm.native_sim.cards import make_card
from spirecomm.native_sim_v2 import NativeRunEnv
from spirecomm.spire.card import Card, CardRarity, CardType


class ObservationBridgeTest(unittest.TestCase):
    def test_recording_serializer_emits_canonical_cost_fields(self):
        card = Card(
            card_id="Entrench",
            name="Entrench+",
            card_type=CardType.SKILL,
            rarity=CardRarity.UNCOMMON,
            upgrades=1,
            has_target=False,
            cost=1,
            uuid="entrench-real",
            misc=0,
            is_playable=True,
            exhausts=False,
        )

        serialized = serialize_card(card)

        self.assertEqual(serialized["cost"], 1)
        self.assertEqual(serialized["base_cost"], 1)
        self.assertEqual(serialized["cost_for_turn"], 1)
        self.assertIsNone(serialized["cost_for_combat"])
        self.assertFalse(serialized["free_to_play_once"])

    def test_v2_canonical_state_uses_effective_hand_cost(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        bash = make_card("Bash", uuid="bash")
        env.combat.hand = [bash]
        bash.cost_for_turn = 0

        canonical = canonicalize_serialized_state(env.combat.to_spirecomm_state())
        hand_card = canonical["combat_state"]["hand"][0]

        self.assertEqual(hand_card["cost"], 0)
        self.assertEqual(hand_card["base_cost"], bash.card_def.upgraded_cost if bash.upgrades > 0 and bash.card_def.upgraded_cost is not None else bash.card_def.cost)
        self.assertEqual(hand_card["cost_for_turn"], 0)

    def test_legacy_combat_tensors_keep_angry_alias_and_dexterity(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        monster = env.combat.monsters[0]
        monster.add_power("Dexterity", 2)
        monster.add_power("Angry", 3)

        tensors = build_state_tensors(env.combat.to_spirecomm_state())
        monster_features = tensors["monster_features"][0]
        base_offset = MONSTER_FEATURE_DIM - len(MONSTER_POWER_SPECS)
        spec_index = {name: index for index, (name, _, _) in enumerate(MONSTER_POWER_SPECS)}

        self.assertGreater(monster_features[base_offset + spec_index["Dexterity"]], 0.0)
        self.assertGreater(monster_features[base_offset + spec_index["Angry"]], 0.0)
        self.assertGreater(monster_features[base_offset + spec_index["Anger"]], 0.0)

    def test_extended_combat_tensors_preserve_legacy_payload(self):
        env = NativeRunEnv(seed=2, ascension_level=0)

        tensors = build_extended_combat_tensors(env.combat.to_spirecomm_state())

        self.assertIn("global_features", tensors)
        self.assertIn("hand_features", tensors)
        self.assertIn("extended_hand_features", tensors)
        self.assertEqual(len(tensors["extended_hand_features"]), 10)

    def test_v2_run_state_and_candidates_are_model_compatible(self):
        env = NativeRunEnv(seed=3, ascension_level=0)
        env.combat.outcome = "PLAYER_VICTORY"
        env.step({"kind": "end", "name": "RESOLVE_COMBAT"})
        env.step({"kind": "skip", "name": "SKIP", "choice_index": len(env.reward_cards)})
        state = env.state()
        action = env.legal_actions()[0]

        state_vector = build_state_vector(state)
        map_choice_vector = choice_vector("map", action, state)

        self.assertEqual(len(state_vector), STATE_DIM)
        self.assertEqual(len(map_choice_vector), CHOICE_DIM)

    def test_serialized_combat_selector_aliases_share_generic_base(self):
        self.assertTrue(issubclass(V2CombatSelector, SerializedCombatSelector))


if __name__ == "__main__":
    unittest.main()
