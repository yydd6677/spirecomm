from __future__ import annotations

import copy
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from generate_v3_combat_teacher_dataset import collect_curated_potion_roots, curated_envs
from spirecomm.ai.torch_compat import torch
from spirecomm.ai.v3_combat_dataset import (
    collate_labeled_roots,
    make_root_sample,
    root_gap_q_loss_padded,
    segment_softmax,
    total_candidate_loss,
)
from spirecomm.ai.v3_combat_features import (
    COMBAT_ROOM_TYPES,
    action_key,
    card_identity_ids,
    encode_state_summary,
    encode_candidate,
    potion_identity_ids,
    root_combat_actions,
    schema,
    step_branch,
)
from spirecomm.ai.v3_combat_model import V3CombatCandidateScorer
from spirecomm.ai.v3_combat_teacher import (
    ContinuationResult,
    TeacherConfig,
    _can_reuse_unblocked_continuation,
    _root_potion_continuation_value,
    candidate_rank_by_card,
    continuation_search,
    label_env,
    transition_reward,
)
from spirecomm.ai.v3_combat_transformer import (
    ROOT_TOKEN_SCHEMA_VERSION_V2,
    TOKEN_TYPES,
    V3CombatRootTransformerTokenSpec,
    V3CombatTransformerCandidateScorer,
    collate_transformer_labeled_roots,
    encode_root_transformer_actions,
    encode_transformer_candidate,
    load_v3_combat_transformer_checkpoint,
    root_token_spec,
    save_v3_combat_transformer_checkpoint,
    token_spec,
)
from spirecomm.native_sim_v3.combat.engine import SUPPORTED_COMBAT_POTION_IDS
from spirecomm.native_sim_v3.content.potions import make_potion


class V3CombatCandidateScorerTest(unittest.TestCase):
    class _PotionSearchEnv:
        def __init__(self, *, potion_count: int = 2, phase: str = "COMBAT", turn: int = 0):
            self.phase = phase
            self.outcome = ""
            self.potion_count = potion_count
            self.turn = turn
            self.used_action: dict | None = None

        def state(self):
            if self.phase != "COMBAT":
                return {"phase": self.phase, "current_hp": 80, "max_hp": 80}
            return {
                "phase": "COMBAT",
                "room_type": "MonsterRoomBoss",
                "current_hp": 80,
                "max_hp": 80,
                "choice_available": False,
                "potions": [
                    {"potion_id": "Fire Potion", "id": "Fire Potion", "name": "Fire Potion"}
                    for _ in range(self.potion_count)
                ],
                "combat_state": {
                    "turn": self.turn,
                    "player": {"current_hp": 80, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
                    "monsters": [
                        {
                            "current_hp": 200,
                            "max_hp": 200,
                            "intent": "ATTACK",
                            "move_adjusted_damage": 0,
                            "move_hits": 1,
                            "powers": [],
                        }
                    ],
                    "hand": [],
                },
            }

        def legal_actions(self):
            actions = [
                {
                    "kind": "potion",
                    "name": "Fire Potion",
                    "potion_id": "Fire Potion",
                    "potion_index": index,
                }
                for index in range(self.potion_count)
            ]
            actions.append({"kind": "end", "name": "END_TURN"})
            return actions

        def step(self, action):
            self.used_action = dict(action)
            if action.get("kind") == "potion":
                self.phase = "CARD_REWARD"
            elif action.get("kind") == "end":
                self.turn += 1

    def _curated_env(self, name: str):
        for case_name, env in curated_envs():
            if case_name == name:
                return env
        raise AssertionError(f"missing curated env {name}")

    def test_feature_schema_matches_candidate_vector(self):
        env = self._curated_env("curated_rage_multi_attack")
        before = env.serialize()
        action = root_combat_actions(env)[0]
        branch = step_branch(env, action)
        after = branch.serialize()
        features = encode_candidate(before, action, after)
        self.assertEqual(len(features), schema().candidate_dim)
        self.assertGreater(schema().state_dim, 0)
        self.assertGreater(schema().action_dim, len(card_identity_ids()) + len(potion_identity_ids()))
        self.assertGreater(schema().delta_dim, 0)

    def test_card_identity_distinguishes_selected_cards(self):
        env = self._curated_env("curated_inflame_flex_attacks")
        before = env.serialize()
        actions = root_combat_actions(env)
        identity_count = len(card_identity_ids())
        identity_start = schema().state_dim + schema().action_dim - identity_count - len(potion_identity_ids())
        identity_end = identity_start + identity_count
        encoded_by_card = {}
        for action in actions:
            branch = step_branch(env, action)
            features = encode_candidate(before, action, branch.serialize())
            card_id = str(action.get("card_id") or action.get("name") or "")
            encoded_by_card.setdefault(card_id, features[identity_start:identity_end])
        self.assertIn("Inflame", encoded_by_card)
        self.assertIn("Flex", encoded_by_card)
        self.assertNotEqual(encoded_by_card["Inflame"], encoded_by_card["Flex"])

    def test_root_actions_include_supported_potions(self):
        env = self._curated_env("curated_rage_multi_attack")
        env.potions = [make_potion("Fire Potion")]
        env.engine.potions = list(env.potions)
        before = env.serialize()
        actions = root_combat_actions(env)
        potion_actions = [action for action in actions if action.get("kind") == "potion"]
        self.assertTrue(potion_actions)
        branch = step_branch(env, potion_actions[0])
        self.assertEqual(before, env.serialize())
        self.assertEqual(branch.potions[0]["potion_id"], "Potion Slot")

    def test_room_type_flags_are_encoded(self):
        env = self._curated_env("curated_rage_multi_attack")
        for expected_index, room_type in enumerate(COMBAT_ROOM_TYPES):
            env.room_type = room_type
            env.engine.state.room_type = room_type
            encoded = encode_state_summary(env.serialize())
            flags = encoded[3:6]
            self.assertEqual(flags, [1.0 if index == expected_index else 0.0 for index in range(3)])

    def test_fire_potion_reward_scales_by_room_type(self):
        config = TeacherConfig(beam_width=2, node_budget_per_root=8, max_depth=3)
        rewards = {}
        for room_type in COMBAT_ROOM_TYPES:
            env = self._curated_env("curated_rage_multi_attack")
            env.room_type = room_type
            env.engine.state.room_type = room_type
            env.potions = [make_potion("Fire Potion")]
            env.engine.potions = list(env.potions)
            action = next(action for action in root_combat_actions(env) if action.get("kind") == "potion")
            before = env.serialize()
            branch = step_branch(env, action)
            rewards[room_type] = transition_reward(env, before, action, branch, branch.serialize(), config)
        self.assertLess(rewards["MonsterRoom"], rewards["MonsterRoomElite"])
        self.assertLess(rewards["MonsterRoomElite"], rewards["MonsterRoomBoss"])

    def test_effective_block_reward_is_deferred_until_end_turn(self):
        config = TeacherConfig(raw_incoming_damage_reduction_weight=0.0)
        root_state = {
            "phase": "COMBAT",
            "current_hp": 80,
            "max_hp": 80,
            "combat_state": {
                "player": {"current_hp": 80, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
                "monsters": [
                    {
                        "current_hp": 20,
                        "max_hp": 20,
                        "intent": "ATTACK",
                        "move_adjusted_damage": 10,
                        "move_hits": 1,
                        "powers": [],
                    }
                ],
                "hand": [],
            },
        }
        defended_state = copy.deepcopy(root_state)
        defended_state["combat_state"]["player"]["block"] = 5

        immediate = transition_reward(
            SimpleNamespace(phase="COMBAT", outcome="", state=lambda: root_state),
            root_state,
            {"kind": "card", "name": "Defend_R"},
            SimpleNamespace(phase="COMBAT", outcome="", state=lambda: defended_state),
            defended_state,
            config,
            block_reward_base_state=root_state,
        )
        self.assertEqual(immediate, 0.0)

        end_reward = transition_reward(
            SimpleNamespace(phase="COMBAT", outcome="", state=lambda: defended_state),
            defended_state,
            {"kind": "end", "name": "END_TURN"},
            SimpleNamespace(phase="COMBAT", outcome="", state=lambda: defended_state),
            defended_state,
            config,
            block_reward_base_state=root_state,
        )
        self.assertEqual(end_reward, 5.0)

        split_state = copy.deepcopy(defended_state)
        split_state["combat_state"]["monsters"][0]["intent"] = "UNKNOWN"
        split_state["combat_state"]["monsters"][0]["move_adjusted_damage"] = -1
        split_state["combat_state"]["monsters"][0]["move_hits"] = 1
        split_reward = transition_reward(
            SimpleNamespace(phase="COMBAT", outcome="", state=lambda: split_state),
            split_state,
            {"kind": "end", "name": "END_TURN"},
            SimpleNamespace(phase="COMBAT", outcome="", state=lambda: split_state),
            split_state,
            config,
            block_reward_base_state=root_state,
        )
        self.assertEqual(split_reward, 0.0)

    def test_root_potion_continuation_scales_by_room_type(self):
        config = TeacherConfig()
        action = {"kind": "potion", "potion_id": "Energy Potion"}
        self.assertAlmostEqual(_root_potion_continuation_value(action, {"room_type": "MonsterRoom"}, 0.0, 10.0, config), 3.0)
        self.assertAlmostEqual(
            _root_potion_continuation_value(action, {"room_type": "MonsterRoomElite"}, 0.0, 10.0, config),
            12.0,
        )
        self.assertAlmostEqual(
            _root_potion_continuation_value(action, {"room_type": "MonsterRoomBoss"}, 0.0, 10.0, config),
            20.0,
        )
        self.assertAlmostEqual(
            _root_potion_continuation_value(
                action,
                {"room_type": "MonsterRoomBoss"},
                -5.0,
                10.0,
                config,
                non_potion_baseline=5.0,
            ),
            10.0,
        )

    def test_continuation_search_blocks_only_same_potion_bottle(self):
        config = TeacherConfig(beam_width=3, node_budget_per_root=8, max_depth=2)
        result = continuation_search(
            self._PotionSearchEnv(potion_count=2),
            root_turn=0,
            config=config,
            blocked_potion_action={"kind": "potion", "potion_id": "Fire Potion", "potion_index": 0},
        )
        self.assertEqual(result.debug_best_line[0]["kind"], "potion")
        self.assertEqual(result.debug_best_line[0]["potion_index"], 1)

        blocked_only = continuation_search(
            self._PotionSearchEnv(potion_count=1),
            root_turn=0,
            config=config,
            blocked_potion_action={"kind": "potion", "potion_id": "Fire Potion", "potion_index": 0},
        )
        self.assertNotEqual(blocked_only.debug_best_line[0]["kind"], "potion")

    def test_blocked_baseline_reuse_requires_full_unpruned_search(self):
        blocked = {"kind": "potion", "potion_id": "Fire Potion", "potion_index": 0}
        card_only = ContinuationResult(
            value=1.0,
            depth=1,
            nodes=1,
            terminal_kind="NEXT_TURN",
            debug_best_line=[{"kind": "card", "name": "Strike_R"}],
            fully_explored=True,
        )
        self.assertTrue(_can_reuse_unblocked_continuation(card_only, blocked))

        used_blocked = ContinuationResult(
            value=1.0,
            depth=1,
            nodes=1,
            terminal_kind="VICTORY",
            debug_best_line=[{"kind": "potion", "potion_id": "Fire Potion", "potion_index": 0}],
            fully_explored=True,
        )
        self.assertFalse(_can_reuse_unblocked_continuation(used_blocked, blocked))

        pruned = ContinuationResult(
            value=1.0,
            depth=1,
            nodes=1,
            terminal_kind="NEXT_TURN",
            debug_best_line=[{"kind": "card", "name": "Strike_R"}],
            fully_explored=False,
        )
        self.assertFalse(_can_reuse_unblocked_continuation(pruned, blocked))

    def test_curated_potion_roots_cover_supported_potions(self):
        roots = collect_curated_potion_roots(limit=None)
        seen = {
            str(action.get("potion_id") or "")
            for root in roots
            for action in root.actions
            if action.get("kind") == "potion"
        }
        self.assertTrue(set(SUPPORTED_COMBAT_POTION_IDS).issubset(seen))

    def test_root_action_keys_are_unique_and_snapshot_roundtrips(self):
        env = self._curated_env("curated_rage_multi_attack")
        root = make_root_sample(env, root_id="test", source="unit")
        self.assertIsNotNone(root)
        assert root is not None
        self.assertEqual(len(root.action_keys), len(set(root.action_keys)))
        restored = pickle.loads(root.env_blob)
        self.assertEqual(restored.serialize(), env.serialize())
        self.assertEqual([action_key(action, env.serialize()) for action in root.actions], root.action_keys)

    def test_deepcopy_branch_step_does_not_mutate_original(self):
        env = self._curated_env("curated_rage_multi_attack")
        before = copy.deepcopy(env.serialize())
        action = root_combat_actions(env)[0]
        branch = step_branch(env, action)
        self.assertEqual(env.serialize(), before)
        self.assertNotEqual(branch.serialize(), before)

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_segment_softmax_sums_within_roots(self):
        values = torch.tensor([1.0, 2.0, 3.0, -1.0, 1.0])
        sample_ids = torch.tensor([0, 0, 0, 1, 1])
        probs = segment_softmax(values, sample_ids)
        self.assertAlmostEqual(float(probs[sample_ids == 0].sum().item()), 1.0, places=6)
        self.assertAlmostEqual(float(probs[sample_ids == 1].sum().item()), 1.0, places=6)

    def test_teacher_ranks_setup_cards_in_curated_cases(self):
        config = TeacherConfig(beam_width=4, node_budget_per_root=32, max_depth=8)
        rage = label_env(self._curated_env("curated_rage_multi_attack"), root_id="rage", source="unit", config=config)
        inflame = label_env(self._curated_env("curated_inflame_flex_attacks"), root_id="inflame", source="unit", config=config)
        corruption = label_env(self._curated_env("curated_corruption_skills"), root_id="corruption", source="unit", config=config)
        assert rage is not None and inflame is not None and corruption is not None
        self.assertEqual(candidate_rank_by_card(rage, "Rage"), 0)
        self.assertIn(candidate_rank_by_card(inflame, "Inflame"), {0, 1})
        self.assertEqual(candidate_rank_by_card(corruption, "Corruption"), 0)

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_model_forward_and_loss_on_labeled_roots(self):
        config = TeacherConfig(beam_width=3, node_budget_per_root=24, max_depth=6)
        roots = []
        for case_name in ("curated_rage_multi_attack", "curated_corruption_skills"):
            labeled = label_env(self._curated_env(case_name), root_id=case_name, source="unit", config=config)
            assert labeled is not None
            roots.append(labeled)
        batch = collate_labeled_roots(roots)
        model = V3CombatCandidateScorer()
        pred_q = model(batch["features"])
        self.assertEqual(tuple(pred_q.shape), (batch["features"].shape[0],))
        loss, metrics = total_candidate_loss(pred_q, batch)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn("rank_loss", metrics)

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_root_gap_q_loss_weights_large_teacher_negatives(self):
        teacher_q = torch.tensor([45.6, -23.9238, 45.6, 45.6, 45.6, -24.4])
        pred_q = torch.tensor([4.8269, 5.0359, 4.9368, 4.8115, 3.6294, -6.3876])
        counts = torch.tensor([6], dtype=torch.long)

        loss = root_gap_q_loss_padded(
            pred_q,
            teacher_q,
            counts,
            transform="sqrt",
            loss="l1",
            hard_negative_gap_threshold=10.0,
            hard_negative_weight=5.0,
        )
        teacher_gap = teacher_q - teacher_q[0]
        pred_gap = pred_q - pred_q[0]
        teacher_gap = teacher_gap / (teacher_gap.abs() + 1.0e-2).sqrt()
        pred_gap = pred_gap / (pred_gap.abs() + 1.0e-2).sqrt()
        per_candidate = (pred_gap - teacher_gap).abs()
        weights = torch.tensor([1.0, 6.0, 1.0, 1.0, 1.0, 6.0])
        expected = (per_candidate * weights).sum() / weights.sum()
        self.assertAlmostEqual(float(loss.item()), float(expected.item()), places=6)
        self.assertGreater(float(loss.item()), 5.0)
        differentiable_pred = pred_q.clone().requires_grad_(True)
        differentiable_loss = root_gap_q_loss_padded(
            differentiable_pred,
            teacher_q,
            counts,
            transform="sqrt",
            loss="l1",
            hard_negative_gap_threshold=10.0,
            hard_negative_weight=5.0,
        )
        differentiable_loss.backward()
        self.assertTrue(torch.isfinite(differentiable_pred.grad).all().item())

    def test_transformer_tokens_keep_slots_for_indexed_entities_but_not_relics(self):
        env = self._curated_env("curated_rage_multi_attack")
        env.potions = [make_potion("Fruit Juice"), make_potion("Fruit Juice")]
        env.engine.potions = list(env.potions)
        before = env.serialize()
        before["relics"] = [{"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "tier": "STARTER"}]
        action = {"kind": "potion", "name": "Fruit Juice", "potion_id": "Fruit Juice", "potion_index": 1}
        record = encode_transformer_candidate(before, action, before)
        token_types = record["token_type_ids"]
        potion_slots = [
            slot
            for token_type, slot in zip(token_types, record["slot_ids"])
            if token_type == TOKEN_TYPES["POTION"]
        ]
        relic_slots = [
            slot
            for token_type, slot in zip(token_types, record["slot_ids"])
            if token_type == TOKEN_TYPES["RELIC"]
        ]
        self.assertEqual(potion_slots[:2], [1, 2])
        self.assertTrue(relic_slots)
        self.assertTrue(all(slot == 0 for slot in relic_slots))

    def test_root_transformer_action_blocks_include_selected_entities(self):
        env = self._curated_env("curated_rage_multi_attack")
        before = env.serialize()
        actions = root_combat_actions(env)
        bash_index = next(
            index
            for index, action in enumerate(actions)
            if action.get("kind") == "card" and action.get("card_id") == "Bash"
        )
        record = encode_root_transformer_actions(before, actions, [before for _ in actions], spec=root_token_spec())
        action_position = record["action_token_positions"][bash_index]
        selected_card_position = action_position + 1
        selected_target_position = action_position + 2
        selected_potion_position = action_position + 3
        after_position = action_position + 4
        delta_position = action_position + 5
        self.assertEqual(record["token_type_ids"][selected_card_position], TOKEN_TYPES["SELECTED_CARD"])
        self.assertEqual(record["token_type_ids"][selected_target_position], TOKEN_TYPES["SELECTED_TARGET"])
        self.assertEqual(record["token_type_ids"][selected_potion_position], TOKEN_TYPES["SELECTED_POTION"])
        self.assertEqual(record["token_type_ids"][after_position], TOKEN_TYPES["AFTER_SUMMARY"])
        self.assertEqual(record["token_type_ids"][delta_position], TOKEN_TYPES["DELTA"])
        self.assertEqual(record["slot_ids"][selected_card_position], bash_index + 1)
        self.assertEqual(record["slot_ids"][selected_target_position], bash_index + 1)
        self.assertGreater(record["entity_ids"][selected_card_position], 0)
        self.assertGreater(record["entity_ids"][selected_target_position], 0)
        self.assertEqual(record["entity_ids"][selected_potion_position], 0)
        self.assertEqual(record["token_scalar_features"][selected_card_position][12], 1.0)
        self.assertEqual(record["token_scalar_features"][selected_target_position][10], 1.0)
        active_token_types = [
            token_type
            for token_type, active in zip(record["token_type_ids"], record["attention_mask"])
            if active
        ]
        self.assertNotIn(TOKEN_TYPES["LEGACY"], active_token_types)

        v2_spec = V3CombatRootTransformerTokenSpec(version=ROOT_TOKEN_SCHEMA_VERSION_V2)
        v2_record = encode_root_transformer_actions(before, actions, [before for _ in actions], spec=v2_spec)
        v2_action_position = v2_record["action_token_positions"][bash_index]
        self.assertEqual(v2_record["legacy_token_positions"][bash_index], v2_action_position + 6)
        self.assertEqual(v2_record["token_type_ids"][v2_action_position + 6], TOKEN_TYPES["LEGACY"])

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_transformer_model_forward_loss_and_checkpoint_roundtrip(self):
        config = TeacherConfig(beam_width=2, node_budget_per_root=12, max_depth=4)
        labeled = label_env(self._curated_env("curated_rage_multi_attack"), root_id="transformer", source="unit", config=config)
        assert labeled is not None
        batch = collate_transformer_labeled_roots([labeled])
        spec = token_spec()
        self.assertEqual(tuple(batch["token_scalar_features"].shape[1:]), (spec.max_sequence_length, spec.scalar_dim))
        self.assertEqual(tuple(batch["token_type_ids"].shape), tuple(batch["attention_mask"].shape))

        model = V3CombatTransformerCandidateScorer(d_model=48, num_layers=1, num_heads=4, ffn_dim=96, dropout=0.0)
        pred_q = model(batch)
        self.assertEqual(tuple(pred_q.shape), (batch["features"].shape[0],))
        loss, metrics = total_candidate_loss(pred_q, batch)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn("rank_loss", metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transformer.pt"
            save_v3_combat_transformer_checkpoint(path, model, training_args={"unit": True})
            loaded, checkpoint = load_v3_combat_transformer_checkpoint(path)
            self.assertEqual(checkpoint.get("checkpoint_version"), "v3_combat_transformer_candidate_scorer_v1")
            loaded_pred = loaded(batch)
            self.assertEqual(tuple(loaded_pred.shape), tuple(pred_q.shape))


if __name__ == "__main__":
    unittest.main()
