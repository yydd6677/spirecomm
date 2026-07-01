from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import compare_native_to_lightspeed_run
import maintain_alignment_failure_corpus
import run_native_sim
import verify_model_integration
from run_native_run import _native_env_cls as run_native_run_env_cls
from spirecomm.ai.runtime_decision import build_runtime_selectors
from spirecomm.native_sim_v2 import NativeRunEnv as V2NativeRunEnv
from spirecomm.native_sim_v3 import NativeCombatEnv, NativeRunEnv
from spirecomm.native_sim_v3.combat.engine import _canonical_power_id
from spirecomm.native_sim_v3.combat import engine as combat_engine_mod
from spirecomm.native_sim_v3.combat.engine import CombatEngine
from spirecomm.native_sim_v3.serialize import adapter as serialize_adapter
from spirecomm.native_sim_v3.content.act_chances import act_chances
from spirecomm.native_sim_v3.content.act_progression import (
    act_for_dungeon_id,
    act_progression,
    dungeon_id_for_act,
    next_dungeon_id,
)
from spirecomm.native_sim_v3.content.campfire_rules import campfire_rules, regal_pillow_bonus
from spirecomm.native_sim_v3.content.chests import chest_catalog, chest_def
from spirecomm.native_sim_v3.content.characters import starting_profile
from spirecomm.native_sim_v3.content.cards import (
    card_catalog,
    card_pools,
    initialize_runtime_card_pools,
    initialize_source_card_pools,
    make_card,
)
from spirecomm.native_sim_v3.content.encounters import act_encounter_def, encounter_catalog
from spirecomm.native_sim_v3.content.ending_rules import ending_rules
from spirecomm.native_sim_v3.content.elite_rules import emerald_elite_rules
from spirecomm.native_sim_v3.content.events import event_catalog
from spirecomm.native_sim_v3.content.events import abstract_dungeon_event_gate_rules, abstract_dungeon_shrine_gate_rules
from spirecomm.native_sim_v3.content.events import note_for_yourself_availability
from spirecomm.native_sim_v3.content.events import note_for_yourself_defaults
from spirecomm.native_sim_v3.content.events import note_for_yourself_preference
from spirecomm.native_sim_v3.content.monsters import monster_catalog
from spirecomm.native_sim_v3.content.map_rules import map_rules
from spirecomm.native_sim_v3.content.potions import draw_random_potion, make_potion, potion_pool, roll_random_potion
from spirecomm.native_sim_v3.content.pricing import (
    card_price_by_rarity,
    potion_price_by_rarity,
    relic_price_by_tier,
    reward_rarity_rules,
)
from spirecomm.native_sim_v3.content.room_reward_rules import room_reward_rules
from spirecomm.native_sim_v3.content.reward_rules import card_blizz_rules, post_combat_potion_rules, potion_roll_rules
from spirecomm.native_sim_v3.content.relics import make_relic, relic_catalog, relic_pools, starter_relics
from spirecomm.native_sim_v3.content.relics import (
    can_relic_spawn,
    draw_random_relic_end,
    initialize_relic_pools,
    non_campfire_excluded_relic_ids,
    pop_random_relic_end_from_pools,
    pop_random_relic_from_pools,
    pop_random_screenless_relic_from_pools,
    screenless_excluded_relic_ids,
)
from spirecomm.native_sim_v3.content.shop import shop_rules
from spirecomm.native_sim_v3.core.randoms import NativeRandomSet, java_shuffle_in_place
from spirecomm.native_sim_v3.core.state import MonsterState, PlayerState
from spirecomm.native_sim_v3.reference import CORE_SOURCE_MAP, REQUIRED_SOURCE_CLASSES, required_source_paths_exist
from spirecomm.native_sim_v3.run.events import EventState, resolve_event_choice
from spirecomm.native_sim_v3.run.events import (
    BASE_ELITE_CHANCE,
    BASE_MONSTER_CHANCE,
    BASE_SHOP_CHANCE,
    BASE_TREASURE_CHANCE,
    _abstract_dungeon_shrine_chance,
    _event_helper_probability_constants,
    _initialize_event_state,
    _is_special_event_available,
    _pick_random_upgradable_indexes,
    _purgeable_candidate_indexes,
    _random_face_relic,
    _upgradable_candidate_indexes,
    generate_event_for_act,
    initialize_event_pools_for_act,
    initialize_event_pools_for_dungeon,
    initialize_special_one_time_event_list,
    RESET_ELITE_CHANCE,
    RESET_MONSTER_CHANCE,
    RESET_SHOP_CHANCE,
    RESET_TREASURE_CHANCE,
    roll_question_room_result,
)
from spirecomm.native_sim_v3.run.campfire import CampfireState, rest_amount
from spirecomm.native_sim_v3.run.boss import (
    apply_boss_relic_choice,
    draw_boss_relic_choices,
    initialize_boss_relic_pool,
    starter_relic_upgrade_mapping,
)
from spirecomm.native_sim_v3.run.encounters import (
    generate_exordium_monster_lists,
    generate_monster_lists_for_act,
    generate_monster_lists_for_dungeon,
)
from spirecomm.native_sim_v3.run.map import MapEdge, MapNode, _generate_room_types, generate_act_map
from spirecomm.native_sim_v3.run.neow import (
    _neow_drawback_order,
    _neow_mini_reward_types,
    _neow_reward_option_specs,
    NeowDrawback,
    NeowOption,
    NeowRewardType,
    generate_blessing_options,
    transform_card,
)
from spirecomm.native_sim_v3.run.rewards import (
    apply_reward_preview_relics,
    generate_card_reward,
    generate_card_reward_groups_with_state,
    generate_elite_relic_rewards,
    generate_monster_room_rewards,
    generate_card_reward_with_state,
)
from spirecomm.native_sim_v3.run.shop import (
    ShopState,
    generate_shop,
    generate_shop_replacement_card,
    generate_shop_replacement_potion,
    generate_shop_replacement_relic,
)
from spirecomm.native_sim_v3.run.treasure import TreasureState, generate_treasure, open_treasure
from spirecomm.native_sim_v3.serialize.adapter import _serialize_combat_state
from spirecomm.seed_helper import seed_long_to_string, seed_string_to_long


def _power_amount(creature: PlayerState, power_id: str) -> int:
    canonical = _canonical_power_id(power_id)
    for power in creature.powers:
        if _canonical_power_id(str(power.get("power_id") or power.get("id"))) == canonical:
            return int(power.get("amount") or 0)
    return 0


class NativeSimV3IndependenceTest(unittest.TestCase):
    def test_seed_helper_matches_known_real_game_strings(self):
        self.assertEqual(seed_long_to_string(1), "1")
        self.assertEqual(seed_long_to_string(2710959347947821323), "T2R7DM3ZMYM3")
        self.assertEqual(seed_long_to_string(7133506393411724536), "23VYRWDQ5VAUL")
        self.assertEqual(seed_string_to_long("T2R7DM3ZMYM3"), 2710959347947821323)
        self.assertEqual(seed_string_to_long("TESTSEED1"), 64004406012856)
        self.assertEqual(seed_string_to_long("64004406012856"), 3730494130536020832)

    def test_transient_fading_one_dies_before_attacking(self):
        combat = NativeCombatEnv(seed=1, ascension_level=0, encounter_name="Transient")
        engine = combat.engine
        transient = engine.state.monsters[0]
        engine.player.current_hp = 41
        engine.player.block = 10
        transient.current_hp = 725
        combat_engine_mod._set_power_amount(transient, "Fading", 1)

        engine._transient_take_turn(transient)

        self.assertEqual(engine.player.current_hp, 41)
        self.assertEqual(transient.current_hp, 0)
        self.assertEqual(combat_engine_mod._get_power_amount(transient, "Fading"), 0)

    def test_map_rng_matches_real_game_act_seed_offsets(self):
        act1 = NativeRandomSet(seed=1, act=1)
        act2 = NativeRandomSet(seed=1, act=2)
        act3 = NativeRandomSet(seed=1, act=3)
        self.assertEqual(act1.stream("map").seed, 2)
        self.assertEqual(act2.stream("map").seed, 201)
        self.assertEqual(act3.stream("map").seed, 601)

    def test_floor_transition_reseeds_floor_based_ai_stream(self):
        randoms = NativeRandomSet(seed=1, floor=0)
        first = randoms.stream("ai").random(99)
        self.assertEqual(randoms.stream("ai").counter, 1)

        randoms.reset_floor_streams(4)
        self.assertEqual(randoms.floor, 4)
        self.assertEqual(randoms.stream("ai").counter, 0)

        second = randoms.stream("ai").random(99)
        control = NativeRandomSet(seed=1, floor=4)
        self.assertEqual(control.stream("ai").random(99), second)
        self.assertNotEqual(first, second)

    def test_sts_random_set_counter_uses_source_boolean_advances(self):
        card_rng = NativeRandomSet(seed=12).stream("card")
        card_rng.random(99)
        card_rng.set_counter(4)

        control = NativeRandomSet(seed=12).stream("card")
        control.random(99)
        for _ in range(3):
            control.random_boolean()

        self.assertEqual(card_rng.counter, 4)
        self.assertEqual([call.method for call in card_rng.calls], ["random", "random_boolean", "random_boolean", "random_boolean"])
        self.assertEqual(card_rng.random(99), control.random(99))

    def test_act_transition_advances_card_rng_counter_bucket_and_resets_potion_blizz(self):
        env = NativeRunEnv(seed=12, enable_neow=False, start_on_map=True)
        env.card_blizz_randomizer = -7
        env.blizzard_potion_mod = 30
        for _ in range(82):
            env.randoms.stream("card").random(999)

        env._advance_to_next_act()

        self.assertEqual(env.act, 2)
        self.assertEqual(env.randoms.stream("card").counter, 250)
        self.assertEqual(env.card_blizz_randomizer, -7)
        self.assertEqual(env.blizzard_potion_mod, 0)
        self.assertTrue(all(call.method == "random_boolean" for call in env.randoms.stream("card").calls[82:]))

    def test_act_progression_is_parsed_from_treasure_room_boss(self):
        progression = act_progression()
        self.assertTrue(progression.source_path.endswith("TreasureRoomBoss.java"))
        self.assertEqual(dungeon_id_for_act(1), "Exordium")
        self.assertEqual(dungeon_id_for_act("2"), "TheCity")
        self.assertEqual(act_for_dungeon_id("TheBeyond"), 3)
        self.assertEqual(next_dungeon_id("Exordium"), "TheCity")
        self.assertEqual(next_dungeon_id("TheCity"), "TheBeyond")
        self.assertIsNone(next_dungeon_id("TheBeyond"))
        self.assertEqual(next_dungeon_id("TheBeyond", endless=True), "Exordium")

    def test_room_type_generation_uses_java_rounding(self):
        room_list = _generate_room_types(10, act=1, ascension_level=0)
        symbols = [token.symbol for token in room_list]
        self.assertEqual(symbols.count("$"), 1)
        self.assertEqual(symbols.count("R"), 1)
        self.assertEqual(symbols.count("E"), 1)
        self.assertEqual(symbols.count("?"), 2)
        self.assertEqual(symbols.count("M"), 0)
        self.assertEqual(len(symbols), 5)

    def test_map_rules_match_decompiled_generate_map_structure(self):
        rules = map_rules()
        self.assertEqual(rules.map_height, 15)
        self.assertEqual(rules.map_width, 7)
        self.assertEqual(rules.map_path_density, 6)
        self.assertEqual(rules.first_row_room_class, "MonsterRoom")
        self.assertEqual(rules.last_row_room_class, "RestRoom")
        self.assertEqual(rules.special_row_index, 8)
        self.assertEqual(rules.special_row_room_class, "TreasureRoom")
        self.assertEqual(rules.endless_mimic_row_index, 8)
        self.assertEqual(rules.endless_mimic_room_class, "MonsterRoomElite")

    def test_room_type_generation_uses_act_specific_room_chances(self):
        exordium = [token.symbol for token in _generate_room_types(50, act="Exordium", ascension_level=0)]
        city = [token.symbol for token in _generate_room_types(50, act="TheCity", ascension_level=0)]
        beyond = [token.symbol for token in _generate_room_types(50, act="TheBeyond", ascension_level=0)]
        self.assertEqual((exordium.count("$"), exordium.count("R"), exordium.count("E"), exordium.count("?"), exordium.count("T")), (3, 6, 4, 11, 0))
        self.assertEqual((city.count("$"), city.count("R"), city.count("E"), city.count("?"), city.count("T")), (3, 6, 4, 11, 0))
        self.assertEqual((beyond.count("$"), beyond.count("R"), beyond.count("E"), beyond.count("?"), beyond.count("T")), (3, 6, 4, 11, 0))

    def test_generate_act_map_uses_source_special_row_assignment(self):
        normal_map = generate_act_map(NativeRandomSet(seed=2), act=1, ascension_level=0)
        mimic_map = generate_act_map(
            NativeRandomSet(seed=2),
            act=1,
            ascension_level=0,
            endless=True,
            blight_ids={"MimicInfestation"},
        )
        self.assertTrue(all(node.room_symbol == "T" for node in normal_map[8] if node.room_symbol is not None))
        self.assertTrue(all(node.room_symbol == "E" for node in mimic_map[8] if node.room_symbol is not None))

    def test_generate_act_map_places_emerald_elite_from_source_logic(self):
        generated_map = generate_act_map(
            NativeRandomSet(seed=2),
            act=1,
            ascension_level=0,
            final_act_available=True,
            has_emerald_key=False,
        )
        emerald_nodes = [node for row in generated_map for node in row if node.has_emerald_key]
        self.assertEqual(len(emerald_nodes), 1)
        self.assertEqual(emerald_nodes[0].room_symbol, "E")

    def test_run_state_serializes_all_three_keys(self):
        env = NativeRunEnv(
            seed=2,
            enable_neow=False,
            start_on_map=True,
            final_act_available=True,
            has_ruby_key=True,
            has_emerald_key=True,
            has_sapphire_key=True,
        )
        state = env.state()
        self.assertTrue(state["has_ruby_key"])
        self.assertTrue(state["has_emerald_key"])
        self.assertTrue(state["has_sapphire_key"])
        self.assertEqual(state["dungeon_id"], "Exordium")

    def test_the_ending_rules_are_parsed_from_source(self):
        rules = ending_rules()
        room_by_name = {room.node_name: room for room in rules.rooms}
        self.assertEqual(room_by_name["restNode"].room_class, "RestRoom")
        self.assertEqual((room_by_name["restNode"].x, room_by_name["restNode"].y), (3, 0))
        self.assertEqual(room_by_name["shopNode"].room_class, "ShopRoom")
        self.assertEqual(room_by_name["enemyNode"].room_class, "MonsterRoomElite")
        self.assertEqual(room_by_name["bossNode"].room_class, "MonsterRoomBoss")
        self.assertEqual(room_by_name["victoryNode"].room_class, "TrueVictoryRoom")
        self.assertIn("Shield and Spear", rules.elite_encounters)
        self.assertIn("The Heart", rules.boss_encounters)

    def test_generate_act4_map_uses_the_ending_special_map(self):
        generated_map = generate_act_map(NativeRandomSet(seed=2), act=4, ascension_level=0)
        self.assertEqual(len(generated_map), 5)
        self.assertEqual(generated_map[0][3].room_symbol, "R")
        self.assertEqual(generated_map[1][3].room_symbol, "$")
        self.assertEqual(generated_map[2][3].room_symbol, "E")
        self.assertEqual(generated_map[3][3].room_symbol, "BOSS")
        self.assertEqual(generated_map[4][3].room_symbol, "VICTORY")
        self.assertEqual([(edge.dst_x, edge.dst_y) for edge in generated_map[0][3].edges], [(3, 1)])
        self.assertEqual(generated_map[3][3].edges, [])

    def test_act3_advance_goes_to_the_ending_when_all_keys_are_owned(self):
        env = NativeRunEnv(
            seed=2,
            enable_neow=False,
            start_on_map=True,
            final_act_available=True,
            has_ruby_key=True,
            has_emerald_key=True,
            has_sapphire_key=True,
        )
        env.act = 3
        env._advance_to_next_act()

        self.assertEqual(env.act, 4)
        self.assertEqual(env.dungeon_id, "TheEnding")
        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.map[0][3].room_symbol, "R")
        self.assertEqual(env.act_boss, "The Heart")

    def test_act3_advance_without_all_keys_still_ends_run(self):
        env = NativeRunEnv(
            seed=2,
            enable_neow=False,
            start_on_map=True,
            final_act_available=True,
            has_ruby_key=True,
            has_emerald_key=True,
            has_sapphire_key=False,
        )
        env.act = 3
        env._advance_to_next_act()

        self.assertEqual(env.phase, "VICTORY")

    def test_act_advance_uses_source_dungeon_progression(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True)
        self.assertEqual(env.act, 1)
        self.assertEqual(env.dungeon_id, "Exordium")
        env._advance_to_next_act()
        self.assertEqual(env.act, 2)
        self.assertEqual(env.dungeon_id, "TheCity")
        env._advance_to_next_act()
        self.assertEqual(env.act, 3)
        self.assertEqual(env.dungeon_id, "TheBeyond")

    def test_the_heart_victory_enters_true_victory_room_directly(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.act = 4
        env.phase = "COMBAT"
        env.current_room_type = "MonsterRoomBoss"

        class _HeartVictoryCombat:
            def __init__(self, player):
                self.player = player
                self.encounter_name = "The Heart"
                self.engine = type("Engine", (), {"gold": 0, "bonus_reward_gold": 0})()

            def step(self, _action):
                return "VICTORY"

        env.combat = _HeartVictoryCombat(env.player)
        env._step_combat({"kind": "end"})

        self.assertEqual(env.phase, "VICTORY")
        self.assertEqual(env.current_room_type, "TrueVictoryRoom")
        self.assertEqual(env.player.powers, [])
        self.assertEqual(env.player.block, 0)

    def test_act3_boss_relic_choice_opens_spire_heart_victory_room(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.act = 3
        env.phase = "BOSS_RELIC"
        env.current_room_type = "TreasureRoomBoss"
        env.boss_relic_options = [make_relic("Astrolabe"), make_relic("Black Star"), make_relic("Calling Bell")]

        env._step_boss_relic({"kind": "skip", "choice_index": 3})

        self.assertEqual(env.phase, "EVENT")
        self.assertEqual(env.current_room_type, "VictoryRoom")
        self.assertIsNotNone(env.current_event)
        self.assertEqual(env.current_event.event_id, "Spire Heart")
        self.assertEqual(env.current_event.screen, "INTRO")

    def test_spire_heart_event_advances_to_act4_with_all_keys(self):
        env = NativeRunEnv(
            seed=2,
            ascension_level=0,
            enable_neow=False,
            start_on_map=True,
            final_act_available=True,
            has_ruby_key=True,
            has_emerald_key=True,
            has_sapphire_key=True,
        )
        env.act = 3
        env.phase = "EVENT"
        env.current_room_type = "VictoryRoom"
        env.current_event = EventState("Spire Heart", screen="MIDDLE_2")

        env.step({"kind": "event", "choice_index": 0})
        self.assertEqual(env.phase, "EVENT")
        self.assertEqual(env.current_event.screen, "GO_TO_ENDING")

        env.step({"kind": "event", "choice_index": 0})

        self.assertEqual(env.act, 4)
        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.current_room_type, "Map")
        self.assertEqual(env.map[0][3].room_symbol, "R")
        self.assertEqual(env.act_boss, "The Heart")

    def test_spire_heart_event_without_keys_ends_in_victory(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.act = 3
        env.phase = "EVENT"
        env.current_room_type = "VictoryRoom"
        env.current_event = EventState("Spire Heart", screen="MIDDLE_2")

        env.step({"kind": "event", "choice_index": 0})
        self.assertEqual(env.current_event.screen, "DEATH")

        env.step({"kind": "event", "choice_index": 0})

        self.assertEqual(env.phase, "VICTORY")
        self.assertEqual(env.current_room_type, "VictoryRoom")
        self.assertIsNone(env.current_event)

    def test_emerald_elite_rules_match_decompiled_source(self):
        rules = emerald_elite_rules()
        self.assertEqual(rules.strength_amount(1), 2)
        self.assertEqual(rules.max_hp_bonus_ratio, 0.25)
        self.assertEqual(rules.metallicize_amount(1), 4)
        self.assertEqual(rules.regenerate_amount(1), 3)

    def test_run_native_entrypoint_can_still_select_v2_explicitly(self):
        self.assertIs(run_native_run_env_cls("v2"), V2NativeRunEnv)

    def test_runtime_entrypoints_default_to_v3_backend(self):
        repo_root = Path("/home/yydd/spirecomm")
        expected_snippets = {
            repo_root / "run_native_run.py": 'default="v3"',
            repo_root / "run_native_sim.py": 'default="v3"',
            repo_root / "export_model_run_checklist.py": 'default="v3"',
            repo_root / "maintain_alignment_failure_corpus.py": 'default="v3"',
            repo_root / "compare_native_to_lightspeed_run.py": 'default="v3"',
            repo_root / "verify_model_integration.py": 'default="lightspeed,v3"',
        }
        for path, snippet in expected_snippets.items():
            self.assertIn(snippet, path.read_text(encoding="utf-8"), msg=f"expected v3 default in {path}")

    def test_runtime_entrypoint_help_text_mentions_v3_default(self):
        repo_root = Path("/home/yydd/spirecomm")
        expected_snippets = {
            repo_root / "run_native_run.py": "defaults to backend v3",
            repo_root / "run_native_sim.py": "defaults to backend v3",
            repo_root / "validate_real_game_first.py": "defaults to native backend v3",
            repo_root / "export_model_run_checklist.py": "defaults to backend v3",
            repo_root / "compare_native_to_lightspeed_run.py": "defaults to backend v3",
            repo_root / "maintain_alignment_failure_corpus.py": "defaults to backend v3",
        }
        for path, snippet in expected_snippets.items():
            self.assertIn(snippet, path.read_text(encoding="utf-8"), msg=f"expected v3 help text in {path}")

    def test_runtime_selector_builder_falls_back_cleanly_without_torch(self):
        with patch("spirecomm.ai.runtime_decision.torch", None):
            selectors = build_runtime_selectors(repo_root=Path("/home/yydd/spirecomm"))
        self.assertEqual(
            selectors,
            {
                "combat": None,
                "card_reward": None,
                "boss_relic": None,
                "map": None,
                "campfire": None,
                "shop": None,
                "event": None,
                "potion": None,
                "upgrade_target": None,
                "purge_target": None,
            },
        )

    def test_run_native_sim_falls_back_to_heuristic_actions_without_torch(self):
        with patch(
            "run_native_sim.SerializedCombatSelector",
            side_effect=ModuleNotFoundError("torch is required for model/training operations in spirecomm.ai"),
        ):
            result = run_native_sim.run_one(
                seed=1,
                ascension=0,
                max_steps=1,
                model_path=Path("/home/yydd/spirecomm/models/combat.pt"),
                device="cpu",
                verbose=False,
                backend="v3",
            )
        self.assertEqual(result["seed"], 1)
        self.assertEqual(result["steps"], 1)

    def test_compare_native_to_lightspeed_reports_missing_lightspeed_dependency_cleanly(self):
        with patch(
            "compare_native_to_lightspeed_run._load_lightspeed_runtime",
            side_effect=ModuleNotFoundError(
                "slaythespire is required for lightspeed/native comparison workflows. "
                "Install the lightspeed Python package or use native-only entrypoints."
            ),
        ):
            with self.assertRaisesRegex(ModuleNotFoundError, "slaythespire is required for lightspeed/native comparison workflows"):
                compare_native_to_lightspeed_run.compare_seed(seed=1, ascension=0, max_steps=1, backend="v3")

    def test_verify_model_integration_dependency_messages_are_friendly(self):
        torch_message = verify_model_integration._dependency_cli_message(
            ModuleNotFoundError("torch is required for model/training operations in spirecomm.ai")
        )
        policy_torch_message = verify_model_integration._dependency_cli_message(
            ImportError("CheckpointCombatPolicy requires torch. Use the spirecomm-rl conda env.")
        )
        lightspeed_message = verify_model_integration._dependency_cli_message(
            ModuleNotFoundError("slaythespire is required for lightspeed-backed model integration checks.")
        )
        self.assertIn("torch is required for verify_model_integration.py", torch_message)
        self.assertIn("native-only entrypoints", torch_message)
        self.assertIn("torch is required for verify_model_integration.py", policy_torch_message)
        self.assertIn("spirecomm-rl environment", policy_torch_message)
        self.assertIn("slaythespire is required for lightspeed-backed model integration checks", lightspeed_message)

    def test_alignment_failure_corpus_dependency_message_is_friendly(self):
        message = maintain_alignment_failure_corpus._dependency_cli_message(
            ModuleNotFoundError("slaythespire is required for lightspeed/native comparison workflows.")
        )
        self.assertIn("slaythespire is required for maintain_alignment_failure_corpus.py", message)
        self.assertIn("native-only entrypoints", message)

    def test_native_sim_v3_namespace_docs_no_longer_claim_v2_is_default(self):
        text = Path("/home/yydd/spirecomm/spirecomm/native_sim_v3/__init__.py").read_text(encoding="utf-8")
        self.assertIn("entrypoints now default to `v3`", text)
        self.assertNotIn("default runtime backend", text)

    def test_native_sim_v3_status_doc_tracks_v3_primary_backend_claim(self):
        text = Path("/home/yydd/spirecomm/spirecomm/native_sim_v3/STATUS.md").read_text(encoding="utf-8")
        self.assertIn("primary repo CLI entrypoints now default to `v3`", text)
        self.assertIn("## Release Gate", text)
        self.assertIn("## Lightweight Environment Notes", text)
        self.assertIn("## Final Close-Out", text)
        self.assertIn("For normal repo use, prefer `v3`.", text)
        self.assertIn("this `v3` rollout is complete", text)
        self.assertIn("use `v3` by default", text)
        self.assertIn("validate_real_game_first.py --mode native --native-backend v3", text)
        self.assertIn("compare_native_to_lightspeed_run.py ...` requires `slaythespire`", text)
        self.assertIn("seed=20 / 135-step", text)
        self.assertIn("seed=6027341539762311745 / complete 78-step", text)
        self.assertIn("python run_native_run.py", text)
        self.assertIn("python run_native_run.py --backend v2", text)

    def test_repo_readmes_point_to_v3_as_current_primary_backend(self):
        root_readme = Path("/home/yydd/spirecomm/README.md").read_text(encoding="utf-8")
        legacy_readme = Path("/home/yydd/spirecomm/spirecomm/native_sim/README.md").read_text(encoding="utf-8")
        v3_readme = Path("/home/yydd/spirecomm/spirecomm/native_sim_v3/README.md").read_text(encoding="utf-8")
        self.assertIn("primary repo CLI entrypoints now default to `v3`", root_readme)
        self.assertIn("native_sim_v3/STATUS.md", root_readme)
        self.assertIn("python run_native_run.py", root_readme)
        self.assertIn("lighter environments without `torch`", root_readme)
        self.assertIn("fail with short actionable messages", root_readme)
        self.assertIn("current", legacy_readme)
        self.assertIn("primary backend status", legacy_readme)
        self.assertIn("native_sim_v3/STATUS.md", legacy_readme)
        self.assertIn("primary native backend", v3_readme)
        self.assertIn("final close-out", v3_readme)
        self.assertIn("lighter environments without", v3_readme)

    def test_ironclad_starting_profile_is_parsed_from_source(self):
        profile = starting_profile("IRONCLAD")

        self.assertEqual(profile.max_hp, 80)
        self.assertEqual(profile.current_hp, 80)
        self.assertEqual(profile.gold, 99)
        self.assertEqual(profile.card_draw, 5)
        self.assertEqual(profile.base_energy, 3)
        self.assertEqual(profile.starter_relic_ids, ("Burning Blood",))
        self.assertEqual(
            profile.starter_deck_ids,
            ("Strike_R", "Strike_R", "Strike_R", "Strike_R", "Strike_R", "Defend_R", "Defend_R", "Defend_R", "Defend_R", "Bash"),
        )

    def test_other_character_starting_profiles_are_parsed_from_source(self):
        silent = starting_profile("THE_SILENT")
        defect = starting_profile("DEFECT")
        watcher = starting_profile("WATCHER")

        self.assertEqual((silent.max_hp, silent.current_hp, silent.orb_slots, silent.gold, silent.card_draw, silent.base_energy), (70, 70, 0, 99, 5, 3))
        self.assertEqual((defect.max_hp, defect.current_hp, defect.orb_slots, defect.gold, defect.card_draw, defect.base_energy), (75, 75, 3, 99, 5, 3))
        self.assertEqual((watcher.max_hp, watcher.current_hp, watcher.orb_slots, watcher.gold, watcher.card_draw, watcher.base_energy), (72, 72, 0, 99, 5, 3))
        self.assertEqual(silent.starter_relic_ids, ("Ring of the Snake",))
        self.assertEqual(defect.starter_relic_ids, ("Cracked Core",))
        self.assertEqual(watcher.starter_relic_ids, ("PureWater",))
        self.assertEqual(silent.starter_deck_ids[-2:], ("Survivor", "Neutralize"))
        self.assertEqual(defect.starter_deck_ids[-2:], ("Zap", "Dualcast"))
        self.assertEqual(watcher.starter_deck_ids[-2:], ("Eruption", "Vigilance"))

    def test_act_level_specific_chances_are_parsed_from_source(self):
        exordium = act_chances("Exordium")
        city = act_chances("TheCity")
        beyond = act_chances("TheBeyond")
        ending = act_chances("TheEnding")

        self.assertEqual(
            (
                exordium.shop_room_chance,
                exordium.rest_room_chance,
                exordium.treasure_room_chance,
                exordium.event_room_chance,
                exordium.elite_room_chance,
                exordium.small_chest_chance,
                exordium.medium_chest_chance,
                exordium.large_chest_chance,
                exordium.common_relic_chance,
                exordium.uncommon_relic_chance,
                exordium.rare_relic_chance,
                exordium.colorless_rare_chance,
                exordium.card_upgraded_chance(0),
                exordium.card_upgraded_chance(12),
            ),
            (0.05, 0.12, 0.0, 0.22, 0.08, 50, 33, 17, 50, 33, 17, 0.3, 0.0, 0.0),
        )
        self.assertEqual(
            (
                city.shop_room_chance,
                city.rest_room_chance,
                city.treasure_room_chance,
                city.event_room_chance,
                city.elite_room_chance,
                city.small_chest_chance,
                city.medium_chest_chance,
                city.large_chest_chance,
                city.common_relic_chance,
                city.uncommon_relic_chance,
                city.rare_relic_chance,
                city.colorless_rare_chance,
                city.card_upgraded_chance(0),
                city.card_upgraded_chance(12),
            ),
            (0.05, 0.12, 0.0, 0.22, 0.08, 50, 33, 17, 50, 33, 17, 0.3, 0.25, 0.125),
        )
        self.assertEqual(
            (
                beyond.shop_room_chance,
                beyond.rest_room_chance,
                beyond.treasure_room_chance,
                beyond.event_room_chance,
                beyond.elite_room_chance,
                beyond.small_chest_chance,
                beyond.medium_chest_chance,
                beyond.large_chest_chance,
                beyond.common_relic_chance,
                beyond.uncommon_relic_chance,
                beyond.rare_relic_chance,
                beyond.colorless_rare_chance,
                beyond.card_upgraded_chance(0),
                beyond.card_upgraded_chance(12),
            ),
            (0.05, 0.12, 0.0, 0.22, 0.08, 50, 33, 17, 50, 33, 17, 0.3, 0.5, 0.25),
        )
        self.assertEqual(
            (
                ending.shop_room_chance,
                ending.rest_room_chance,
                ending.treasure_room_chance,
                ending.event_room_chance,
                ending.elite_room_chance,
                ending.small_chest_chance,
                ending.medium_chest_chance,
                ending.large_chest_chance,
                ending.common_relic_chance,
                ending.uncommon_relic_chance,
                ending.rare_relic_chance,
                ending.colorless_rare_chance,
                ending.card_upgraded_chance(0),
                ending.card_upgraded_chance(12),
            ),
            (0.05, 0.12, 0.0, 0.22, 0.08, 0, 100, 0, 0, 100, 0, 0.3, 0.5, 0.25),
        )

    def test_chest_definitions_are_parsed_from_source(self):
        catalog = chest_catalog()
        self.assertEqual(
            (
                catalog["SmallChest"].common_chance,
                catalog["SmallChest"].uncommon_chance,
                catalog["SmallChest"].rare_chance,
                catalog["SmallChest"].gold_chance,
                catalog["SmallChest"].gold_amount,
            ),
            (75, 25, 0, 50, 25),
        )
        self.assertEqual(
            (
                catalog["MediumChest"].common_chance,
                catalog["MediumChest"].uncommon_chance,
                catalog["MediumChest"].rare_chance,
                catalog["MediumChest"].gold_chance,
                catalog["MediumChest"].gold_amount,
            ),
            (35, 50, 15, 35, 50),
        )
        self.assertEqual(
            (
                catalog["LargeChest"].common_chance,
                catalog["LargeChest"].uncommon_chance,
                catalog["LargeChest"].rare_chance,
                catalog["LargeChest"].gold_chance,
                catalog["LargeChest"].gold_amount,
            ),
            (0, 75, 25, 50, 75),
        )

    def test_v3_exposes_independent_surface_without_running_v2_logic(self):
        env = NativeRunEnv(seed=1, ascension_level=0, enable_neow=False)

        self.assertEqual(env.sim_backend, "v3")
        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.state()["backend"], "v3")
        self.assertTrue(env.legal_actions())
        self.assertEqual(env.player.current_hp, 80)
        self.assertEqual(env.player.max_hp, 80)
        self.assertEqual(env.gold, 99)
        self.assertEqual(env.player.base_energy, 3)
        self.assertEqual(env.player.draw_per_turn, 5)

    def test_card_catalog_extracts_real_card_metadata(self):
        catalog = card_catalog()
        pools = card_pools()

        self.assertIn("Bash", catalog)
        self.assertEqual(catalog["Bash"].cost, 2)
        self.assertEqual(catalog["Bash"].rarity, "BASIC")
        self.assertEqual(catalog["Burn"].base_magic, 2)
        self.assertEqual(catalog["Burn"].upgraded_magic, 4)
        self.assertIn("Demon Form", pools["RED_RARE"])

    def test_card_pools_follow_initialize_card_pools_hashmap_order(self):
        pools = card_pools()

        self.assertEqual(pools["RED_COMMON"][:5], ["Anger", "Cleave", "Warcry", "Flex", "Iron Wave"])
        self.assertEqual(pools["RED_UNCOMMON"][:4], ["Spot Weakness", "Inflame", "Power Through", "Dual Wield"])
        self.assertEqual(pools["RED_RARE"][:4], ["Immolate", "Offering", "Exhume", "Reaper"])
        self.assertEqual(pools["COLORLESS_UNCOMMON"][:4], ["Dark Shackles", "PanicButton", "Trip", "Dramatic Entrance"])
        self.assertEqual(pools["COLORLESS_RARE"][:4], ["Sadistic Nature", "The Bomb", "Secret Technique", "Violence"])
        self.assertEqual(pools["CURSE"][:4], ["Regret", "Writhe", "AscendersBane", "Decay"])
        self.assertEqual(pools["STATUS"][:4], ["Wound", "Dazed", "Slimed", "Void"])

    def test_source_colorless_pool_preserves_runtime_cross_rarity_order(self):
        source_pools = initialize_source_card_pools()
        non_healing = [card_id for card_id in source_pools["SRC_COLORLESS"] if card_id != "Bandage Up"]

        self.assertEqual(non_healing[:4], ["Madness", "Thinking Ahead", "Mind Blast", "Metamorphosis"])

    def test_screenless_and_non_campfire_relic_exclusions_are_parsed_from_source(self):
        self.assertEqual(
            screenless_excluded_relic_ids(),
            ("Bottled Flame", "Bottled Lightning", "Bottled Tornado", "Whetstone"),
        )
        self.assertEqual(
            non_campfire_excluded_relic_ids(),
            ("Peace Pipe", "Shovel", "Girya"),
        )

    def test_can_relic_spawn_uses_source_rules(self):
        self.assertFalse(can_relic_spawn("MawBank", floor_num=10, current_room_type="ShopRoom"))
        self.assertFalse(can_relic_spawn("Ancient Tea Set", floor_num=49))
        self.assertFalse(can_relic_spawn("Black Blood", owned_relic_ids=set()))
        self.assertTrue(can_relic_spawn("Black Blood", owned_relic_ids={"Burning Blood"}))
        self.assertFalse(can_relic_spawn("Bottled Flame", deck=[make_card("Strike_R")]))
        self.assertTrue(can_relic_spawn("Bottled Flame", deck=[make_card("Anger")]))

    def test_pop_random_relic_from_pools_respects_spawn_and_screenless_rules(self):
        relic_pools_state = {"COMMON": ["MawBank", "Anchor"], "UNCOMMON": [], "RARE": [], "SHOP": [], "BOSS": []}
        relic = pop_random_relic_from_pools(relic_pools_state, "COMMON", floor_num=10, current_room_type="ShopRoom")
        self.assertEqual(relic["relic_id"], "Anchor")

        screenless_pools_state = {"COMMON": ["Bottled Flame", "Anchor"], "UNCOMMON": [], "RARE": [], "SHOP": [], "BOSS": []}
        relic = pop_random_screenless_relic_from_pools(screenless_pools_state, "COMMON")
        self.assertEqual(relic["relic_id"], "Anchor")

    def test_pop_random_relic_end_from_pools_uses_pool_tail(self):
        relic_pools_state = {"COMMON": ["Anchor", "Bag of Marbles"], "UNCOMMON": [], "RARE": [], "SHOP": [], "BOSS": []}
        relic = pop_random_relic_end_from_pools(relic_pools_state, "COMMON")
        self.assertEqual(relic["relic_id"], "Bag of Marbles")
        self.assertEqual(relic_pools_state["COMMON"], ["Anchor"])

    def test_draw_random_relic_end_uses_initialized_pool_tail(self):
        randoms = NativeRandomSet(seed=11)
        expected_pools = initialize_relic_pools(
            NativeRandomSet(seed=11),
            character="IRONCLAD",
        )
        expected = expected_pools["COMMON"][-1]
        relic = draw_random_relic_end(randoms, "COMMON", character="IRONCLAD")
        self.assertEqual(relic["relic_id"], expected)

    def test_card_pools_parse_other_character_library_sections(self):
        silent = card_pools("THE_SILENT")
        defect = card_pools("DEFECT")
        watcher = card_pools("WATCHER")

        self.assertIn("Backflip", silent["GREEN_COMMON"])
        self.assertIn("Adrenaline", silent["GREEN_RARE"])
        self.assertIn("Ball Lightning", defect["BLUE_COMMON"])
        self.assertIn("Biased Cognition", defect["BLUE_RARE"])
        self.assertIn("Evaluate", watcher["PURPLE_COMMON"])
        self.assertIn("Scrawl", watcher["PURPLE_RARE"])

    def test_card_pool_runtime_alias_tracks_selected_character_class_pool(self):
        silent = card_pools("THE_SILENT")

        self.assertEqual(silent["CLASS_COMMON"], silent["GREEN_COMMON"])
        self.assertEqual(silent["CLASS_UNCOMMON"], silent["GREEN_UNCOMMON"])
        self.assertEqual(silent["CLASS_RARE"], silent["GREEN_RARE"])
        self.assertEqual(silent["RED_COMMON"], silent["GREEN_COMMON"])

    def test_neow_reward_structure_is_parsed_from_source(self):
        reward_specs = _neow_reward_option_specs()

        self.assertEqual(
            [reward for reward, _ in reward_specs[0]],
            [
                NeowRewardType.THREE_CARDS,
                NeowRewardType.ONE_RANDOM_RARE_CARD,
                NeowRewardType.REMOVE_CARD,
                NeowRewardType.UPGRADE_CARD,
                NeowRewardType.TRANSFORM_CARD,
                NeowRewardType.RANDOM_COLORLESS,
            ],
        )
        self.assertEqual(
            [reward for reward, _ in reward_specs[2]],
            [
                NeowRewardType.RANDOM_COLORLESS_2,
                NeowRewardType.REMOVE_TWO,
                NeowRewardType.ONE_RARE_RELIC,
                NeowRewardType.THREE_RARE_CARDS,
                NeowRewardType.TWO_FIFTY_GOLD,
                NeowRewardType.TRANSFORM_TWO_CARDS,
                NeowRewardType.TWENTY_PERCENT_HP_BONUS,
            ],
        )
        self.assertEqual(_neow_drawback_order(), (
            NeowDrawback.TEN_PERCENT_HP_LOSS,
            NeowDrawback.NO_GOLD,
            NeowDrawback.CURSE,
            NeowDrawback.PERCENT_DAMAGE,
        ))
        exclusions = dict(reward_specs[2])
        self.assertEqual(exclusions[NeowRewardType.REMOVE_TWO], NeowDrawback.CURSE)
        self.assertEqual(exclusions[NeowRewardType.TWO_FIFTY_GOLD], NeowDrawback.NO_GOLD)
        self.assertEqual(exclusions[NeowRewardType.TWENTY_PERCENT_HP_BONUS], NeowDrawback.TEN_PERCENT_HP_LOSS)

    def test_neow_mini_blessing_is_parsed_from_source(self):
        self.assertEqual(
            _neow_mini_reward_types(),
            (NeowRewardType.THREE_ENEMY_KILL, NeowRewardType.TEN_PERCENT_HP_BONUS),
        )

        options = generate_blessing_options(NativeRandomSet(seed=5), current_hp=80, max_hp=80, mini_blessing=True)
        self.assertEqual([option.bonus for option in options], [NeowRewardType.THREE_ENEMY_KILL, NeowRewardType.TEN_PERCENT_HP_BONUS])
        self.assertTrue(all(option.drawback == NeowDrawback.NONE for option in options))

    def test_env_supports_neow_mini_blessing_flow(self):
        env = NativeRunEnv(seed=5, ascension_level=0, enable_neow=True, neow_mini_blessing=True)

        actions = env.legal_actions()
        self.assertEqual(len(actions), 2)
        self.assertEqual([action["bonus"] for action in actions], ["THREE_ENEMY_KILL", "TEN_PERCENT_HP_BONUS"])

    def test_registry_catalogs_import_real_game_content_in_bulk(self):
        self.assertIn("Burning Blood", relic_catalog())
        self.assertIn("Exordium", encounter_catalog())
        self.assertIn("Golden Idol", event_catalog())
        self.assertIn("Cultist", monster_catalog())
        self.assertIn("COMMON", relic_pools())
        self.assertGreater(len(relic_catalog()), 50)
        self.assertGreater(len(event_catalog()), 20)
        self.assertGreater(len(monster_catalog()), 20)

    def test_wing_boots_starts_with_three_charges(self):
        self.assertEqual(make_relic("WingedGreaves")["counter"], 3)

    def test_matryoshka_reward_starts_with_two_charges(self):
        self.assertEqual(make_relic("Matryoshka")["counter"], 2)

    def test_omamori_reward_starts_with_two_charges(self):
        self.assertEqual(make_relic("Omamori")["counter"], 2)

    def test_nunchaku_reward_starts_with_zero_counter(self):
        self.assertEqual(make_relic("Nunchaku")["counter"], 0)

    def test_sundial_reward_starts_with_zero_counter(self):
        self.assertEqual(make_relic("Sundial")["counter"], 0)

    def test_pen_nib_reward_starts_with_zero_counter(self):
        self.assertEqual(make_relic("Pen Nib")["counter"], 0)

    def test_ink_bottle_reward_starts_with_zero_counter(self):
        self.assertEqual(make_relic("InkBottle")["counter"], 0)

    def test_wing_boots_exposes_next_row_and_spends_charge_on_off_path_choice(self):
        env = NativeRunEnv(seed=71, character="IRONCLAD", ascension_level=0, enable_neow=False, start_on_map=True)
        nodes = [[MapNode(x=x, y=y) for x in range(5)] for y in range(3)]
        nodes[0][1].room_symbol = "M"
        nodes[0][1].add_edge(MapEdge(src_x=1, src_y=0, dst_x=1, dst_y=1))
        for x, symbol in [(1, "R"), (2, "R"), (3, "E"), (4, "?")]:
            nodes[1][x].room_symbol = symbol
            nodes[1][x].add_edge(MapEdge(src_x=x, src_y=1, dst_x=x, dst_y=2))
            nodes[2][x].room_symbol = "M"
        env.map = nodes
        env.phase = "MAP"
        env.current_map_node = (1, 0)
        env.first_room_chosen = True
        env.floor = 1
        env.relics.append(make_relic("WingedGreaves"))

        actions = env.legal_actions()
        self.assertEqual([(action["symbol"], action["x"]) for action in actions], [("R", 1), ("R", 2), ("E", 3), ("?", 4)])

        off_path_rest = next(action for action in actions if action["symbol"] == "R" and action["x"] == 2)
        env._step_map(off_path_rest)
        wing_boots = next(relic for relic in env.relics if relic["relic_id"] == "WingedGreaves")
        self.assertEqual(wing_boots["counter"], 2)

    def test_potion_pool_filters_to_ironclad_plus_shared_potions(self):
        ironclad = set(potion_pool("IRONCLAD"))
        silent = set(potion_pool("THE_SILENT"))
        defect = set(potion_pool("DEFECT"))
        watcher = set(potion_pool("WATCHER"))

        self.assertIn("BloodPotion", ironclad)
        self.assertIn("ElixirPotion", ironclad)
        self.assertIn("HeartOfIron", ironclad)
        self.assertNotIn("Poison Potion", ironclad)
        self.assertNotIn("FocusPotion", ironclad)
        self.assertNotIn("BottledMiracle", ironclad)
        self.assertIn("Poison Potion", silent)
        self.assertIn("FocusPotion", defect)
        self.assertIn("BottledMiracle", watcher)
        self.assertIn("Block Potion", ironclad)
        self.assertIn("Block Potion", silent)
        self.assertIn("Block Potion", defect)
        self.assertIn("Block Potion", watcher)

    def test_starter_relics_default_to_ironclad_starter_only(self):
        relic_ids = [str(relic.get("relic_id")) for relic in starter_relics()]
        self.assertEqual(relic_ids, ["Burning Blood"])

    def test_relic_pools_filter_to_shared_and_ironclad_relics(self):
        pools = relic_pools()
        pooled_ids = {relic_id for pool in pools.values() for relic_id in pool}

        self.assertIn("Burning Blood", pools["STARTER"])
        self.assertIn("Paper Frog", pooled_ids)
        self.assertIn("Snecko Eye", pools["BOSS"])
        self.assertNotIn("Paper Crane", pooled_ids)
        self.assertNotIn("Cracked Core", pooled_ids)
        self.assertNotIn("HolyWater", pooled_ids)
        self.assertNotIn("Ring of the Serpent", pooled_ids)

    def test_initialize_boss_relic_pool_excludes_non_ironclad_boss_relics(self):
        pool = initialize_boss_relic_pool(NativeRandomSet(seed=1))

        self.assertIn("Black Blood", pool)
        self.assertIn("Snecko Eye", pool)
        self.assertNotIn("FrozenCore", pool)
        self.assertNotIn("HolyWater", pool)
        self.assertNotIn("Ring of the Serpent", pool)

    def test_boss_relic_starter_upgrade_mapping_is_parsed_from_source(self):
        self.assertEqual(
            starter_relic_upgrade_mapping(),
            {
                "Black Blood": "Burning Blood",
                "Ring of the Serpent": "Ring of the Snake",
                "FrozenCore": "Cracked Core",
                "HolyWater": "PureWater",
            },
        )

    def test_event_pool_initialization_uses_decompiled_source_order(self):
        exordium_events, exordium_shrines = initialize_event_pools_for_act(1)
        city_events, city_shrines = initialize_event_pools_for_act(2)
        ending_events, ending_shrines = initialize_event_pools_for_act(4)
        special = initialize_special_one_time_event_list(ascension_level=0)

        self.assertEqual(
            exordium_events,
            [
                "Big Fish",
                "The Cleric",
                "Dead Adventurer",
                "Golden Idol",
                "Golden Wing",
                "World of Goop",
                "Liars Game",
                "Living Wall",
                "Mushrooms",
                "Scrap Ooze",
                "Shining Light",
            ],
        )
        self.assertEqual(
            exordium_shrines,
            [
                "Match and Keep!",
                "Golden Shrine",
                "Transmorgrifier",
                "Purifier",
                "Upgrade Shrine",
                "Wheel of Change",
            ],
        )
        self.assertEqual(city_events[:4], ["Addict", "Back to Basics", "Beggar", "Colosseum"])
        self.assertEqual(city_shrines[:3], ["Match and Keep!", "Wheel of Change", "Golden Shrine"])
        self.assertEqual(ending_events, [])
        self.assertEqual(ending_shrines, [])
        self.assertEqual(
            special[:5],
            [
                "Accursed Blacksmith",
                "Bonfire Elementals",
                "Designer",
                "Duplicator",
                "FaceTrader",
            ],
        )

    def test_event_pool_initialization_accepts_explicit_dungeon_id(self):
        ending_events, ending_shrines = initialize_event_pools_for_dungeon("TheEnding")
        city_events, city_shrines = initialize_event_pools_for_dungeon("TheCity")
        self.assertEqual(ending_events, [])
        self.assertEqual(ending_shrines, [])
        self.assertEqual(city_events[:4], ["Addict", "Back to Basics", "Beggar", "Colosseum"])
        self.assertEqual(city_shrines[:3], ["Match and Keep!", "Wheel of Change", "Golden Shrine"])

    def test_abstract_dungeon_event_gate_rules_are_parsed_from_source(self):
        rules = abstract_dungeon_event_gate_rules()

        self.assertEqual(rules["Dead Adventurer"].floor_gt, 6)
        self.assertEqual(rules["Mushrooms"].floor_gt, 6)
        self.assertEqual(rules["The Cleric"].gold_ge, 35)
        self.assertEqual(rules["Beggar"].gold_ge, 75)
        self.assertTrue(rules["Colosseum"].current_node_y_gt_half)
        self.assertEqual(rules["The Moai Head"].required_relic_id, "Golden Idol")
        self.assertEqual(rules["The Moai Head"].hp_ratio_le, 0.5)

    def test_abstract_dungeon_shrine_gate_rules_are_parsed_from_source(self):
        rules = abstract_dungeon_shrine_gate_rules()

        self.assertTrue(rules["Fountain of Cleansing"].require_curse)
        self.assertEqual(rules["Designer"].dungeon_ids, ("TheCity", "TheBeyond"))
        self.assertEqual(rules["Designer"].gold_ge, 75)
        self.assertEqual(rules["FaceTrader"].dungeon_ids, ("TheCity", "Exordium"))
        self.assertEqual(rules["Knowing Skull"].dungeon_ids, ("TheCity",))
        self.assertEqual(rules["Knowing Skull"].current_hp_gt, 12)
        self.assertEqual(rules["N'loth"].dungeon_ids, ("TheCity",))
        self.assertEqual(rules["N'loth"].relic_count_ge, 2)
        self.assertEqual(rules["SecretPortal"].dungeon_ids, ("TheBeyond",))
        self.assertEqual(rules["SecretPortal"].playtime_seconds_ge, 800.0)

    def test_fountain_availability_ignores_protected_curses(self):
        common_kwargs = {
            "event_id": "Fountain of Cleansing",
            "dungeon_id": "Exordium",
            "gold": 99,
            "relic_count": 0,
            "current_hp": 80,
            "playtime_seconds": 0.0,
        }

        self.assertFalse(
            _is_special_event_available(deck=[make_card("Necronomicurse", uuid="necro")], **common_kwargs)
        )
        self.assertFalse(
            _is_special_event_available(deck=[make_card("AscendersBane", uuid="bane")], **common_kwargs)
        )
        self.assertFalse(
            _is_special_event_available(deck=[make_card("CurseOfTheBell", uuid="bell")], **common_kwargs)
        )
        self.assertTrue(
            _is_special_event_available(deck=[make_card("Injury", uuid="injury")], **common_kwargs)
        )

    def test_note_for_yourself_defaults_are_parsed_from_source(self):
        defaults = note_for_yourself_defaults()

        self.assertEqual(defaults.card_pref_key, "NOTE_CARD")
        self.assertEqual(defaults.upgrade_pref_key, "NOTE_UPGRADE")
        self.assertEqual(defaults.default_card_id, "Iron Wave")
        self.assertEqual(defaults.default_upgrades, 0)

    def test_note_for_yourself_preference_reads_active_save_slot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            preferences_dir = Path(tmp_dir)
            (preferences_dir / "STSSaveSlots").write_text('{"DEFAULT_SLOT":"1"}', encoding="utf-8")
            (preferences_dir / "STSPlayer").write_text(
                '{"NOTE_CARD":"Strike_R","NOTE_UPGRADE":"0"}',
                encoding="utf-8",
            )
            (preferences_dir / "1_STSPlayer").write_text(
                '{"NOTE_CARD":"Berserk","NOTE_UPGRADE":"1"}',
                encoding="utf-8",
            )

            preference = note_for_yourself_preference(preferences_dir)
            env = NativeRunEnv(
                seed=2,
                ascension_level=0,
                enable_neow=False,
                note_for_yourself_preferences_dir=str(preferences_dir),
            )

        self.assertEqual(preference.card_id, "Berserk")
        self.assertEqual(preference.upgrades, 1)
        self.assertEqual(env.note_for_yourself_card_id, "Berserk")
        self.assertEqual(env.note_for_yourself_upgrades, 1)

    def test_note_for_yourself_availability_is_parsed_from_source(self):
        availability = note_for_yourself_availability()

        self.assertTrue(availability.daily_run_disables)
        self.assertEqual(availability.disabled_at_ascension_ge, 15)
        self.assertEqual(availability.enabled_at_ascension_eq, 0)
        self.assertEqual(availability.unlocked_ascension_pref_key, "ASCENSION_LEVEL")

    def test_env_threads_highest_unlocked_ascension_into_special_event_pool(self):
        env = NativeRunEnv(
            seed=2,
            ascension_level=3,
            highest_unlocked_ascension=5,
            enable_neow=False,
        )

        self.assertIn("NoteForYourself", env.special_one_time_event_list)

    def test_event_helper_probability_constants_are_parsed_from_source(self):
        parsed = _event_helper_probability_constants()

        self.assertEqual(
            parsed,
            {
                "BASE_ELITE_CHANCE": 0.1,
                "BASE_MONSTER_CHANCE": 0.1,
                "BASE_SHOP_CHANCE": 0.03,
                "BASE_TREASURE_CHANCE": 0.02,
                "RAMP_ELITE_CHANCE": 0.1,
                "RAMP_MONSTER_CHANCE": 0.1,
                "RAMP_SHOP_CHANCE": 0.03,
                "RAMP_TREASURE_CHANCE": 0.02,
                "RESET_ELITE_CHANCE": 0.0,
                "RESET_MONSTER_CHANCE": 0.1,
                "RESET_SHOP_CHANCE": 0.03,
                "RESET_TREASURE_CHANCE": 0.02,
            },
        )
        self.assertEqual(_abstract_dungeon_shrine_chance(), 0.25)
        self.assertEqual(BASE_ELITE_CHANCE, parsed["BASE_ELITE_CHANCE"])
        self.assertEqual(BASE_MONSTER_CHANCE, parsed["BASE_MONSTER_CHANCE"])
        self.assertEqual(BASE_SHOP_CHANCE, parsed["BASE_SHOP_CHANCE"])
        self.assertEqual(BASE_TREASURE_CHANCE, parsed["BASE_TREASURE_CHANCE"])

    def test_seed1_floor2_question_room_generates_the_cleric_from_source_pools(self):
        randoms = NativeRandomSet(seed=1)
        event_list, shrine_list = initialize_event_pools_for_act(1)
        special = initialize_special_one_time_event_list(ascension_level=0)

        room_result, _ = roll_question_room_result(
            randoms,
            floor=2,
            current_room_type="MonsterRoom",
            relics=[],
            elite_chance=BASE_ELITE_CHANCE,
            monster_chance=BASE_MONSTER_CHANCE,
            shop_chance=BASE_SHOP_CHANCE,
            treasure_chance=BASE_TREASURE_CHANCE,
        )

        self.assertEqual(room_result, "EVENT")
        self.assertEqual(randoms.stream("event").counter, 1)

        event = generate_event_for_act(
            randoms,
            ascension_level=0,
            act=1,
            floor=2,
            gold=99,
            relics=[],
            deck=[],
            potions=[],
            current_hp=80,
            max_hp=80,
            current_node_y=2,
            map_height=15,
            event_list=event_list,
            shrine_list=shrine_list,
            special_one_time_event_list=special,
        )

        self.assertEqual(event.event_id, "The Cleric")
        self.assertEqual(randoms.stream("event").counter, 1)
        self.assertNotIn("The Cleric", event_list)

    def test_beyond_shrine_pool_excludes_nloth_so_seed_floor37_selects_lab(self):
        randoms = NativeRandomSet(seed=-3393152369649078577)
        randoms.stream("event").set_counter(10)
        event_list, shrine_list = initialize_event_pools_for_dungeon("TheBeyond")
        special = initialize_special_one_time_event_list(ascension_level=0)

        room_result, _ = roll_question_room_result(
            randoms,
            floor=37,
            current_room_type="MonsterRoom",
            relics=[
                {"relic_id": "Burning Blood"},
                {"relic_id": "Blood Vial"},
                {"relic_id": "Letter Opener"},
                {"relic_id": "Fusion Hammer"},
                {"relic_id": "PreservedInsect"},
                {"relic_id": "Boot"},
                {"relic_id": "Bag of Preparation"},
                {"relic_id": "Cursed Key"},
                {"relic_id": "Torii"},
            ],
            elite_chance=BASE_ELITE_CHANCE,
            monster_chance=BASE_MONSTER_CHANCE,
            shop_chance=BASE_SHOP_CHANCE,
            treasure_chance=BASE_TREASURE_CHANCE,
        )

        self.assertEqual(room_result, "EVENT")
        self.assertFalse(
            _is_special_event_available(
                "N'loth",
                dungeon_id="TheBeyond",
                gold=337,
                deck=[],
                relic_count=9,
                current_hp=15,
                playtime_seconds=0,
            )
        )

        event = generate_event_for_act(
            randoms,
            ascension_level=0,
            act=3,
            floor=37,
            gold=337,
            relics=[{"relic_id": f"R{i}"} for i in range(9)],
            deck=[],
            potions=[],
            current_hp=15,
            max_hp=83,
            current_node_y=2,
            map_height=15,
            event_list=event_list,
            shrine_list=shrine_list,
            special_one_time_event_list=special,
        )

        self.assertEqual(event.event_id, "Lab")

    def test_we_meet_again_random_potion_consumes_misc_before_gold_amount(self):
        randoms = NativeRandomSet(seed=6)
        randoms.reset_floor_streams(8)
        relic_draw_calls = []

        def screenless_relic_drawer(tier=None, exclude=None):
            relic_draw_calls.append((tier, exclude))
            return make_relic("HornCleat")

        event = _initialize_event_state(
            EventState("WeMeetAgain"),
            randoms=randoms,
            ascension_level=0,
            floor=8,
            gold=176,
            deck=[make_card("Strike_R"), make_card("Rampage", uuid="rampage-test")],
            relics=[],
            potions=[{"potion_id": "Energy Potion", "id": "Energy Potion", "name": "Energy Potion"}],
            max_hp=80,
            screenless_relic_drawer=screenless_relic_drawer,
        )

        self.assertEqual(event.data["potion_index"], 0)
        self.assertEqual(event.data["gold_amount"], 127)
        self.assertNotIn("given_relic", event.data)
        self.assertEqual(relic_draw_calls, [])
        self.assertEqual([call.method for call in randoms.stream("misc").calls], ["random_long", "random", "random_long"])

    def test_we_meet_again_screen_state_preserves_disabled_potion_slot(self):
        env = NativeRunEnv(seed=6, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState(
            "WeMeetAgain",
            data={"potion_index": None, "gold_amount": 113, "card_uuid": "whirlwind-test"},
        )

        options = env._event_screen_state()["options"]
        actions = env.legal_actions()

        self.assertTrue(options[0]["disabled"])
        self.assertNotIn("choice_index", options[0])
        self.assertEqual([option.get("choice_index", index) for index, option in enumerate(options)], [0, 0, 1, 2])
        self.assertEqual([action["choice_index"] for action in actions], [0, 1, 2])
        self.assertEqual([action["button_index"] for action in actions], [1, 2, 3])

    def test_we_meet_again_uses_button_index_for_locked_slot_effects(self):
        env = NativeRunEnv(seed=6, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.gold = 137
        env.current_event = EventState(
            "WeMeetAgain",
            data={
                "potion_index": None,
                "gold_amount": 113,
                "card_uuid": "whirlwind-test",
            },
        )
        relic_draw_calls = []

        def screenless_relic_drawer(tier=None, exclude=None):
            relic_draw_calls.append((tier, exclude))
            return make_relic("Orichalcum")

        env._pop_screenless_relic_from_pool = screenless_relic_drawer
        action = env.legal_actions()[0]

        env.step(action)

        self.assertEqual(len(relic_draw_calls), 1)
        self.assertEqual(env.gold, 24)
        self.assertIn("Orichalcum", {relic["relic_id"] for relic in env.relics})

    def test_we_meet_again_leave_does_not_draw_reward_relic(self):
        env = NativeRunEnv(seed=6, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.gold = 137
        env.current_event = EventState(
            "WeMeetAgain",
            data={
                "potion_index": 0,
                "gold_amount": 113,
                "card_uuid": "whirlwind-test",
            },
        )
        env.potions = [{"potion_id": "Energy Potion", "id": "Energy Potion", "name": "Energy Potion"}]
        env.deck = [make_card("Whirlwind", uuid="whirlwind-test")]
        relic_draw_calls = []

        def screenless_relic_drawer(tier=None, exclude=None):
            relic_draw_calls.append((tier, exclude))
            return make_relic("Orichalcum")

        env._pop_screenless_relic_from_pool = screenless_relic_drawer
        leave_action = next(action for action in env.legal_actions() if action.get("button_index") == 3)

        env.step(leave_action)

        self.assertEqual(relic_draw_calls, [])
        self.assertEqual(env.gold, 137)
        self.assertEqual(env.potions[0]["potion_id"], "Energy Potion")
        self.assertTrue(any(card.get("uuid") == "whirlwind-test" for card in env.deck))
        self.assertNotIn("Orichalcum", {relic["relic_id"] for relic in env.relics})

    def test_we_meet_again_paid_branches_draw_reward_relic_at_resolution(self):
        cases = [
            ("potion", {"potion_index": 0, "gold_amount": 0, "card_uuid": None}, 0),
            ("gold", {"potion_index": None, "gold_amount": 113, "card_uuid": None}, 1),
            ("card", {"potion_index": None, "gold_amount": 0, "card_uuid": "whirlwind-test"}, 2),
        ]
        for branch, data, button_index in cases:
            with self.subTest(branch=branch):
                env = NativeRunEnv(seed=6, enable_neow=False, start_on_map=True)
                env.phase = "EVENT"
                env.current_room_type = "EventRoom"
                env.gold = 137
                env.potions = [{"potion_id": "Energy Potion", "id": "Energy Potion", "name": "Energy Potion"}]
                env.deck = [make_card("Whirlwind", uuid="whirlwind-test")]
                env.current_event = EventState("WeMeetAgain", data=dict(data))
                relic_draw_calls = []

                def screenless_relic_drawer(tier=None, exclude=None):
                    relic_draw_calls.append((tier, exclude))
                    return make_relic("Orichalcum")

                env._pop_screenless_relic_from_pool = screenless_relic_drawer
                action = next(action for action in env.legal_actions() if action.get("button_index") == button_index)

                env.step(action)

                self.assertEqual(len(relic_draw_calls), 1)
                self.assertIn("Orichalcum", {relic["relic_id"] for relic in env.relics})
                if branch == "potion":
                    self.assertEqual(env.potions[0]["potion_id"], "Potion Slot")
                elif branch == "gold":
                    self.assertEqual(env.gold, 24)
                else:
                    self.assertFalse(any(card.get("uuid") == "whirlwind-test" for card in env.deck))

    def test_beggar_card_removal_returns_directly_to_map_after_confirm(self):
        env = NativeRunEnv(seed=50, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.gold = 100
        env.deck = [
            make_card("Strike_R", uuid="beggar-strike"),
            make_card("Defend_R", uuid="beggar-defend"),
        ]
        env.current_event = EventState("Beggar")

        env.step({"kind": "event", "event_id": "Beggar", "choice_index": 0})
        self.assertEqual(env.gold, 25)
        self.assertEqual(env.current_event.screen, "GAVE_MONEY")

        env.step({"kind": "event", "event_id": "Beggar", "choice_index": 0})
        self.assertEqual(env.phase, "CARD_SELECT")

        env.step({"kind": "card_select", "mode": "purge", "target_index": 1, "choice_index": 1})
        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertTrue(env.current_card_select["confirm_up"])

        env.step({"kind": "confirm", "name": "CONFIRM", "choice_index": 0})
        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.current_room_type, "Map")
        self.assertIsNone(env.current_event)
        self.assertEqual([card["card_id"] for card in env.deck], ["Strike_R"])

    def test_scrap_ooze_tracks_damage_and_relic_chance_progression(self):
        event = EventState("Scrap Ooze")
        actions = event.actions(ascension_level=0, max_hp=80)
        self.assertIn("25%", actions[0]["name"])

        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            potions=[],
        )
        self.assertEqual(result["hp"], 77)
        self.assertEqual(event.data["relic_chance"], 35)
        self.assertEqual(event.data["dmg"], 4)

    def test_shining_light_upgrades_two_random_upgradable_cards(self):
        deck = [make_card("Strike_R"), make_card("Defend_R"), make_card("Bash")]
        event = EventState("Shining Light")

        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=deck,
            potions=[],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertEqual(result["hp"], 64)
        self.assertEqual(len(result["upgrade_indexes"]), 2)

    def test_shining_light_low_damage_uses_torii_before_tungsten_rod(self):
        deck = [make_card("Strike_R"), make_card("Defend_R"), make_card("Bash")]
        cases = [
            ([make_relic("Torii")], 19),
            ([make_relic("Torii"), make_relic("TungstenRod")], 20),
        ]
        for relics, expected_hp in cases:
            with self.subTest(relics=[relic["relic_id"] for relic in relics]):
                event = EventState("Shining Light")
                result = resolve_event_choice(
                    event,
                    action_index=0,
                    randoms=NativeRandomSet(seed=2),
                    ascension_level=0,
                    current_hp=20,
                    max_hp=20,
                    gold=99,
                    deck=deck,
                    relics=relics,
                    potions=[],
                )

                self.assertEqual(event.screen, "COMPLETE")
                self.assertEqual(result["hp"], expected_hp)
                self.assertEqual(len(result["upgrade_indexes"]), 2)

    def test_shining_light_high_damage_ignores_torii_but_keeps_tungsten_rod(self):
        deck = [make_card("Strike_R"), make_card("Defend_R"), make_card("Bash")]
        event = EventState("Shining Light")

        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=deck,
            relics=[make_relic("Torii"), make_relic("TungstenRod")],
            potions=[],
        )

        self.assertEqual(result["hp"], 65)
        self.assertEqual(len(result["upgrade_indexes"]), 2)

    def test_shining_light_can_upgrade_already_upgraded_searing_blow(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("Shining Light")
        env.deck = [make_card("Searing Blow", upgrades=1, uuid="searing")]

        actions = env.legal_actions()
        self.assertEqual([action["choice_index"] for action in actions], [0, 1])
        env.step(actions[0])

        self.assertEqual(env.deck[0]["card_id"], "Searing Blow")
        self.assertEqual(env.deck[0]["upgrades"], 2)

    def test_event_percentage_values_use_mathutils_rounding(self):
        moai = _initialize_event_state(
            EventState("The Moai Head"),
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            floor=40,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
            max_hp=20,
        )
        self.assertEqual(moai.data["hp_amt"], 3)

        winding = _initialize_event_state(
            EventState("Winding Halls"),
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            floor=40,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
            max_hp=20,
        )
        self.assertEqual(winding.data["hp_amt"], 3)

        forgotten = _initialize_event_state(
            EventState("Forgotten Altar"),
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            floor=40,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
            max_hp=50,
        )
        self.assertEqual(forgotten.data["hp_loss"], 13)

        library = _initialize_event_state(
            EventState("The Library"),
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            floor=40,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
            max_hp=80,
        )
        self.assertEqual(library.data["heal_amt"], 26)

    def test_shining_light_high_asc_rounding_matches_label_and_execution(self):
        deck = [make_card("Strike_R", uuid="strike")]
        event = EventState("Shining Light")
        actions = event.actions(ascension_level=15, max_hp=75, deck=deck)
        self.assertEqual(actions[0]["name"], "Enter (23 HP)")

        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=15,
            current_hp=75,
            max_hp=75,
            gold=99,
            deck=deck,
            potions=[],
        )

        self.assertEqual(result["hp"], 52)
        self.assertEqual(result["upgrade_indexes"], [0])

    def test_upgradable_candidate_indexes_use_can_upgrade_card_semantics(self):
        cannot_upgrade = make_card("Strike_R", uuid="cannot")
        cannot_upgrade["can_upgrade"] = False
        deck = [
            make_card("Searing Blow", upgrades=1, uuid="searing"),
            cannot_upgrade,
            make_card("Injury", uuid="injury"),
            make_card("Wound", uuid="wound"),
            make_card("Defend_R", uuid="defend"),
        ]

        self.assertEqual(_upgradable_candidate_indexes(deck), [0, 4])

    def test_pick_random_upgradable_indexes_uses_java_shuffle_semantics(self):
        deck = [make_card("Strike_R"), make_card("Defend_R"), make_card("Bash")]
        randoms = NativeRandomSet(seed=17)
        expected_rng = randoms.duplicate_stream("misc")
        expected = [0, 1, 2]
        java_shuffle_in_place(expected, int(expected_rng.random_long()))

        actual = _pick_random_upgradable_indexes(randoms, deck, count=2)

        self.assertEqual(actual, expected[:2])

    def test_golden_wing_damage_branch_opens_followup_purge(self):
        event = EventState("Golden Wing")
        deck = [make_card("Strike_R"), make_card("Bash")]

        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=deck,
            potions=[],
        )
        self.assertEqual(result["hp"], 73)
        self.assertEqual(event.screen, "PURGE")

        purge = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=73,
            max_hp=80,
            gold=99,
            deck=deck,
            potions=[],
        )
        self.assertTrue(purge["open_card_select"])
        self.assertEqual(purge["card_select_mode"], "purge")

    def test_face_trader_trade_grants_face_relic(self):
        event = EventState("FaceTrader")
        first = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "MAIN")
        self.assertEqual(first["gold"], 99)

        second = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "RESULT")
        self.assertIn(second["add_relics"][0]["relic_id"], {"CultistMask", "FaceOfCleric", "GremlinMask", "NlothsMask", "SsserpentHead", "Circlet"})

    def test_world_of_goop_initializes_gold_loss_and_leave_branch(self):
        event = generate_event_for_act(
            NativeRandomSet(seed=2),
            ascension_level=0,
            act=1,
            floor=3,
            gold=15,
            relics=[],
            deck=[],
            potions=[],
            current_hp=80,
            max_hp=80,
            current_node_y=3,
            map_height=15,
            event_list=["World of Goop"],
            shrine_list=[],
            special_one_time_event_list=[],
        )
        self.assertEqual(event.event_id, "World of Goop")
        self.assertLessEqual(int(event.data["gold_loss"]), 15)
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=15,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "RESULT")
        self.assertEqual(result["gold"], 15 - int(event.data["gold_loss"]))

    def test_mushrooms_fight_branch_opens_event_combat_with_rewards(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("Mushrooms")

        env.step({"kind": "event", "event_id": "Mushrooms", "choice_index": 0})
        self.assertEqual(env.current_event.screen, "FIGHT")

        env.step({"kind": "event", "event_id": "Mushrooms", "choice_index": 0})
        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual(env.current_room_type, "EventRoom")
        self.assertEqual(env.combat.encounter_name, "The Mushroom Lair")
        self.assertEqual(env.pending_event_rewards["relics"][0]["relic_id"], "Odd Mushroom")
        self.assertGreaterEqual(int(env.pending_event_rewards["gold"]), 20)
        self.assertLessEqual(int(env.pending_event_rewards["gold"]), 30)

    def test_mushrooms_heal_parasite_uses_mark_of_the_bloom_heal_block(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("Mushrooms")
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.relics = [make_relic("Mark of the Bloom")]

        env.step({"kind": "event", "event_id": "Mushrooms", "choice_index": 1})

        self.assertEqual(env.player.current_hp, 40)
        self.assertTrue(any(card["card_id"] == "Parasite" for card in env.deck))

    def test_mushrooms_heal_parasite_does_not_apply_magic_flower_outside_combat(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("Mushrooms")
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.relics = [make_relic("Magic Flower")]

        env.step({"kind": "event", "event_id": "Mushrooms", "choice_index": 1})

        self.assertEqual(env.player.current_hp, 60)
        self.assertTrue(any(card["card_id"] == "Parasite" for card in env.deck))

    def test_event_combat_victory_adds_normal_card_reward(self):
        class VictoryCombat:
            encounter_name = "The Mushroom Lair"

            def __init__(self, deck: list[dict[str, object]]) -> None:
                self.player = PlayerState(current_hp=30, max_hp=80)
                self.master_deck = list(deck)
                self.gold = 205
                self.potions = []

            def step(self, _action: dict[str, object]) -> str:
                return "VICTORY"

        env = NativeRunEnv(seed=8795815203514330698, ascension_level=0, enable_neow=False)
        env.phase = "COMBAT"
        env.current_room_type = "EventRoom"
        env.floor = 10
        env.gold = 205
        env.player.current_hp = 30
        env.player.max_hp = 80
        env.pending_event_rewards = {
            "gold": 21,
            "relics": [make_relic("Odd Mushroom")],
            "potions": [],
            "cards": [],
            "card_groups": [],
        }
        env.combat = VictoryCombat(env.deck)

        env._step_combat({"kind": "card"})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(env.player.current_hp, 36)
        self.assertEqual(env.reward_gold, 21)
        self.assertEqual(env.reward_relics[0]["relic_id"], "Odd Mushroom")
        self.assertEqual(len(env.reward_cards), 3)
        state = env.state()
        reward_types = [reward["reward_type"] for reward in state["screen_state"]["rewards"]]
        self.assertIn("CARD", reward_types)

    def test_event_combat_gold_reward_applies_golden_idol_bonus(self):
        class VictoryCombat:
            encounter_name = "The Mushroom Lair"

            def __init__(self, deck: list[dict[str, object]]) -> None:
                self.player = PlayerState(current_hp=30, max_hp=80)
                self.master_deck = list(deck)
                self.gold = 205
                self.potions = []

            def step(self, _action: dict[str, object]) -> str:
                return "VICTORY"

        env = NativeRunEnv(seed=67, ascension_level=0, enable_neow=False)
        env.phase = "COMBAT"
        env.current_room_type = "EventRoom"
        env.floor = 10
        env.gold = 205
        env.relics = [make_relic("Golden Idol")]
        env.pending_event_rewards = {
            "gold": 20,
            "relics": [make_relic("Odd Mushroom")],
            "potions": [],
            "cards": [],
            "card_groups": [],
        }
        env.combat = VictoryCombat(env.deck)

        env._step_combat({"kind": "card"})

        self.assertEqual(env.reward_gold, 25)

    def test_golden_shrine_desecrate_grants_gold_and_regret(self):
        event = EventState("Golden Shrine")
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertEqual(result["gold"], 374)
        self.assertEqual(result["add_cards"][0]["card_id"], "Regret")

    def test_match_and_keep_builds_source_board_and_exposes_card_positions(self):
        randoms = NativeRandomSet(seed=2)
        event = generate_event_for_act(
            randoms,
            ascension_level=0,
            act=1,
            floor=8,
            gold=99,
            relics=[],
            deck=[],
            potions=[],
            current_hp=80,
            max_hp=80,
            current_node_y=8,
            map_height=15,
            event_list=[],
            shrine_list=["Match and Keep!"],
            special_one_time_event_list=[],
        )
        self.assertEqual(event.event_id, "Match and Keep!")
        self.assertEqual(len(event.data["cards"]), 12)
        self.assertEqual(randoms.stream("card").counter, 4)
        self.assertEqual(randoms.stream("shuffle").counter, 1)
        self.assertEqual(randoms.stream("misc").counter, 1)

        step1 = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "RULE_EXPLANATION")
        self.assertEqual(step1["gold"], 99)

        step2 = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "PLAY")
        self.assertEqual(step2["gold"], 99)

        actions = event.actions(ascension_level=0, max_hp=80)
        self.assertEqual(len(actions), 12)
        self.assertEqual([action["label"] for action in actions], [f"card{index}" for index in range(12)])

        step3 = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "PLAY")
        self.assertEqual(step3["gold"], 99)
        self.assertEqual(event.data["first_card_index"], 0)
        self.assertEqual(len(event.actions(ascension_level=0, max_hp=80)), 11)

    def test_match_and_keep_uses_character_start_card_pair(self):
        expected = {
            "IRONCLAD": "Bash",
            "THE_SILENT": "Neutralize",
            "DEFECT": "Zap",
            "WATCHER": "Eruption",
        }
        for character, start_card_id in expected.items():
            with self.subTest(character=character):
                event = generate_event_for_act(
                    NativeRandomSet(seed=2),
                    ascension_level=0,
                    act=1,
                    floor=8,
                    gold=99,
                    relics=[],
                    deck=[],
                    potions=[],
                    current_hp=80,
                    max_hp=80,
                    current_node_y=8,
                    map_height=15,
                    event_list=[],
                    shrine_list=["Match and Keep!"],
                    special_one_time_event_list=[],
                    runtime_card_pools=initialize_runtime_card_pools(character),
                    player_class=character,
                )
                card_ids = [card["card_id"] for card in event.data["cards"]]
                self.assertEqual(card_ids.count(start_card_id), 2)

    def test_match_and_keep_preview_relics_apply_before_duplicate_pairs(self):
        event = generate_event_for_act(
            NativeRandomSet(seed=2),
            ascension_level=0,
            act=1,
            floor=8,
            gold=99,
            relics=[make_relic("Molten Egg 2")],
            deck=[],
            potions=[],
            current_hp=80,
            max_hp=80,
            current_node_y=8,
            map_height=15,
            event_list=[],
            shrine_list=["Match and Keep!"],
            special_one_time_event_list=[],
        )
        bash_cards = [card for card in event.data["cards"] if card["card_id"] == "Bash"]

        self.assertEqual(len(bash_cards), 2)
        self.assertEqual([card["upgrades"] for card in bash_cards], [1, 1])
        self.assertEqual([card["base_damage"] for card in bash_cards], [10, 10])

    def test_match_and_keep_duplicate_pairs_preserve_previewed_stat_fields(self):
        def fake_preview(cards, *, owned_relic_ids=None):
            previewed = [dict(card) for card in cards]
            for card in previewed:
                if card["card_id"] == "Bash":
                    card["misc"] = 77
                    card["damage"] = 44
                    card["cost_for_turn"] = 0
            return previewed

        with patch("spirecomm.native_sim_v3.run.events.apply_reward_preview_relics", fake_preview):
            event = generate_event_for_act(
                NativeRandomSet(seed=2),
                ascension_level=0,
                act=1,
                floor=8,
                gold=99,
                relics=[],
                deck=[],
                potions=[],
                current_hp=80,
                max_hp=80,
                current_node_y=8,
                map_height=15,
                event_list=[],
                shrine_list=["Match and Keep!"],
                special_one_time_event_list=[],
            )
        bash_cards = [card for card in event.data["cards"] if card["card_id"] == "Bash"]

        self.assertEqual(len(bash_cards), 2)
        self.assertEqual([card["misc"] for card in bash_cards], [77, 77])
        self.assertEqual([card["damage"] for card in bash_cards], [44, 44])
        self.assertEqual([card["cost_for_turn"] for card in bash_cards], [0, 0])
        self.assertNotEqual(bash_cards[0]["uuid"], bash_cards[1]["uuid"])

    def test_match_and_keep_resolves_pairs_and_remembers_revealed_cards(self):
        event = EventState(
            "Match and Keep!",
            screen="PLAY",
            data={
                "cards": [
                    make_card("Bash", uuid="match-0"),
                    make_card("Strike_R", uuid="match-1"),
                    make_card("Defend_R", uuid="match-2"),
                    make_card("Anger", uuid="match-3"),
                    make_card("Uppercut", uuid="match-4"),
                    make_card("Cleave", uuid="match-5"),
                    make_card("Strike_R", uuid="match-6"),
                    make_card("Defend_R", uuid="match-7"),
                    make_card("Anger", uuid="match-8"),
                    make_card("Bash", uuid="match-9"),
                    make_card("Uppercut", uuid="match-10"),
                    make_card("Cleave", uuid="match-11"),
                ],
                "attempt_count": 5,
                "matched_cards": [],
            },
        )

        first = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertNotIn("add_cards", first)
        self.assertEqual(event.data["first_card_index"], 0)
        actions_after_first = event.actions(ascension_level=0, max_hp=80)
        self.assertEqual(actions_after_first[0]["match_card_index"], 9)

        second = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual([card["card_id"] for card in second["add_cards"]], ["Bash"])
        self.assertEqual(event.data["attempt_count"], 4)
        self.assertEqual(set(event.data["removed_card_indexes"]), {0, 9})
        self.assertEqual(event.data["matched_cards"], ["Bash"])

    def test_dead_adventurer_search_rewards_then_fight_event_combat(self):
        event = generate_event_for_act(
            NativeRandomSet(seed=2),
            ascension_level=0,
            act=1,
            floor=8,
            gold=99,
            relics=[],
            deck=[],
            potions=[],
            current_hp=80,
            max_hp=80,
            current_node_y=8,
            map_height=15,
            event_list=["Dead Adventurer"],
            shrine_list=[],
            special_one_time_event_list=[],
        )
        self.assertEqual(event.event_id, "Dead Adventurer")
        self.assertEqual(len(event.data["rewards"]), 3)

        search = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertIn(event.screen, {"INTRO", "FAIL", "SUCCESS"})
        if event.screen == "FAIL":
            fight = resolve_event_choice(
                event,
                action_index=0,
                randoms=NativeRandomSet(seed=2),
                ascension_level=0,
                current_hp=80,
                max_hp=80,
                gold=99,
                deck=[],
                relics=[],
                potions=[],
            )
            self.assertTrue(fight["open_combat"])
            self.assertIn(fight["encounter_name"], {"3 Sentries", "Gremlin Nob", "Lagavulin Event"})

    def test_dead_adventurer_fail_prebuilds_monsters_before_fight(self):
        randoms = NativeRandomSet(seed=-6356770840560860096)
        randoms.reset_floor_streams(8)
        event = EventState(
            "Dead Adventurer",
            data={"rewards": ["NOTHING"], "encounter_chance": 100, "num_rewards": 0, "enemy": 0},
        )

        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=randoms,
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )

        self.assertFalse(result["open_rewards"])
        self.assertEqual(event.screen, "FAIL")
        self.assertEqual([monster.current_hp for monster in event.data["prebuilt_monsters"]], [39, 39, 40])
        self.assertEqual(randoms.stream("monster_hp").counter, 3)

        fight = resolve_event_choice(
            event,
            action_index=0,
            randoms=randoms,
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )

        self.assertTrue(fight["open_combat"])
        self.assertTrue(fight["elite_trigger"])
        self.assertEqual([monster.current_hp for monster in fight["prebuilt_monsters"]], [39, 39, 40])
        self.assertEqual(randoms.stream("monster_hp").counter, 3)

    def test_dead_adventurer_success_relic_uses_screenless_drawer(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.deck = [make_card("Strike_R")]
        env.relic_pools["COMMON"] = ["Whetstone", "Anchor"]
        event = EventState(
            "Dead Adventurer",
            data={"rewards": ["RELIC"], "encounter_chance": -1, "num_rewards": 0, "enemy": 0},
        )
        normal_draw_calls = []
        screenless_tiers = []

        def normal_relic_drawer(tier):
            normal_draw_calls.append(tier)
            return make_relic("Bottled Flame")

        def screenless_relic_drawer(tier=None, exclude=None):
            screenless_tiers.append((tier, exclude))
            return env._pop_screenless_relic_from_pool("COMMON")

        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
            relic_drawer=normal_relic_drawer,
            screenless_relic_drawer=screenless_relic_drawer,
        )

        self.assertEqual([relic["relic_id"] for relic in result["add_relics"]], ["Anchor"])
        self.assertEqual(normal_draw_calls, [])
        self.assertEqual(len(screenless_tiers), 1)
        self.assertEqual(env.relic_pools["COMMON"], [])

    def test_dead_adventurer_fail_relic_reward_keeps_normal_drawer(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.deck = [make_card("Strike_R")]
        env.relic_pools["COMMON"] = ["Whetstone", "Anchor"]
        event = EventState(
            "Dead Adventurer",
            screen="FAIL",
            data={"rewards": ["RELIC"], "encounter_chance": 25, "num_rewards": 0, "enemy": 0},
        )
        screenless_draw_calls = []

        def screenless_relic_drawer(tier=None, exclude=None):
            screenless_draw_calls.append((tier, exclude))
            return env._pop_screenless_relic_from_pool("COMMON")

        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
            relic_drawer=lambda tier: env._pop_relic_from_pool("COMMON"),
            screenless_relic_drawer=screenless_relic_drawer,
        )

        self.assertTrue(result["open_combat"])
        self.assertEqual([relic["relic_id"] for relic in result["event_rewards"]["relics"]], ["Whetstone"])
        self.assertEqual(screenless_draw_calls, [])
        self.assertEqual(env.relic_pools["COMMON"], ["Anchor"])

    def test_back_to_basics_simplicity_upgrades_starter_strikes_and_defends(self):
        cannot_upgrade = make_card("Strike_G", uuid="cannot")
        cannot_upgrade["can_upgrade"] = False
        deck = [
            make_card("Strike_R", uuid="strike-r"),
            make_card("Defend_R", uuid="defend-r"),
            make_card("Strike_G", uuid="strike-g"),
            make_card("Defend_G", uuid="defend-g"),
            make_card("Strike_B", uuid="strike-b"),
            make_card("Defend_B", uuid="defend-b"),
            make_card("Strike_P", uuid="strike-p"),
            make_card("Defend_P", uuid="defend-p"),
            make_card("Bash"),
            make_card("Anger"),
            cannot_upgrade,
            make_card("Injury", uuid="injury"),
            make_card("Wound", uuid="wound"),
            make_card("Defend_P", upgrades=1, uuid="upgraded-defend-p"),
        ]
        event = EventState("Back to Basics")
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=deck,
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertEqual(result["upgrade_indexes"], list(range(8)))

    def test_duplicator_opens_duplicate_card_select(self):
        event = EventState("Duplicator")
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[make_card("Strike_R"), make_card("Bash")],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertTrue(result["open_card_select"])
        self.assertEqual(result["card_select_mode"], "duplicate")

    def test_duplicate_card_select_preserves_stat_fields_and_clears_bottle_flags(self):
        selected = make_card("RitualDagger", uuid="ritual-original")
        selected.update(
            {
                "misc": 17,
                "base_damage": 44,
                "damage": 44,
                "cost": 1,
                "cost_for_turn": 0,
                "cost_for_combat": 0,
                "free_to_play_once": True,
                "bottled": True,
                "in_bottle_flame": True,
                "in_bottle_lightning": True,
                "in_bottle_tornado": True,
            }
        )
        original_snapshot = dict(selected)
        env = NativeRunEnv(seed=2, start_on_map=True, enable_neow=False)
        env.deck = [selected]
        env.phase = "EVENT"
        env.current_event = EventState("Duplicator")

        env.step({"kind": "event", "event_id": "Duplicator", "choice_index": 0})
        self.assertEqual(env.phase, "CARD_SELECT")
        action = env.legal_actions()[0]
        env.step(action)

        self.assertEqual(env.deck[0], original_snapshot)
        self.assertEqual(len(env.deck), 2)
        duplicate = env.deck[1]
        self.assertEqual(duplicate["card_id"], "RitualDagger")
        self.assertEqual(duplicate["misc"], 17)
        self.assertEqual(duplicate["base_damage"], 44)
        self.assertEqual(duplicate["damage"], 44)
        self.assertEqual(duplicate["cost"], 1)
        self.assertEqual(duplicate["cost_for_turn"], 0)
        self.assertEqual(duplicate["cost_for_combat"], 0)
        self.assertTrue(duplicate["free_to_play_once"])
        self.assertNotEqual(duplicate["uuid"], selected["uuid"])
        self.assertFalse(duplicate["bottled"])
        self.assertFalse(duplicate["in_bottle_flame"])
        self.assertFalse(duplicate["in_bottle_lightning"])
        self.assertFalse(duplicate["in_bottle_tornado"])

    def test_dollys_mirror_duplicate_uses_stat_equivalent_copy_semantics(self):
        selected = make_card("Genetic Algorithm", uuid="genetic-original")
        selected.update({"misc": 31, "base_block": 31, "block": 31, "in_bottle_lightning": True})
        env = NativeRunEnv(seed=2, start_on_map=True, enable_neow=False)
        env.deck = [selected]
        env.phase = "RELIC_REWARD"

        env._obtain_relic(make_relic("DollysMirror"), source="reward_relic")
        self.assertEqual(env.phase, "CARD_SELECT")
        env.step(env.legal_actions()[0])

        self.assertEqual(len(env.deck), 2)
        duplicate = env.deck[1]
        self.assertEqual(duplicate["card_id"], "Genetic Algorithm")
        self.assertEqual(duplicate["misc"], 31)
        self.assertEqual(duplicate["base_block"], 31)
        self.assertEqual(duplicate["block"], 31)
        self.assertNotEqual(duplicate["uuid"], selected["uuid"])
        self.assertFalse(duplicate.get("bottled", False))
        self.assertFalse(duplicate["in_bottle_lightning"])

    def test_beggar_gave_money_opens_purge(self):
        event = EventState("Beggar")
        first = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[make_card("Strike_R"), make_card("Bash")],
            relics=[],
            potions=[],
        )
        self.assertEqual(first["gold"], 24)
        self.assertEqual(event.screen, "GAVE_MONEY")
        second = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=24,
            deck=[make_card("Strike_R"), make_card("Bash")],
            relics=[],
            potions=[],
        )
        self.assertTrue(second["open_card_select"])
        self.assertEqual(second["card_select_mode"], "purge")

    def test_forgotten_altar_actions_offer_idol_when_golden_idol_is_owned(self):
        event = EventState("Forgotten Altar", data={"hp_loss": 20})
        actions = event.actions(
            ascension_level=0,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[make_relic("Golden Idol")],
            potions=[],
        )
        self.assertEqual(actions[0]["name"], "Offer Idol")

    def test_forgotten_altar_offer_removes_golden_idol_when_obtaining_bloody_idol(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("Forgotten Altar", data={"hp_loss": 20})
        env.relics = [make_relic("Golden Idol")]

        env._step_event({"kind": "event", "event_id": "Forgotten Altar", "choice_index": 0})

        relic_ids = [str(relic.get("relic_id") or relic.get("id")) for relic in env.relics]
        self.assertNotIn("Golden Idol", relic_ids)
        self.assertIn("Bloody Idol", relic_ids)

    def test_forgotten_altar_offer_keeps_golden_idol_when_bloody_idol_owned(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("Forgotten Altar", data={"hp_loss": 20})
        env.relics = [make_relic("Golden Idol"), make_relic("Bloody Idol")]

        env._step_event({"kind": "event", "event_id": "Forgotten Altar", "choice_index": 0})

        relic_ids = [str(relic.get("relic_id") or relic.get("id")) for relic in env.relics]
        self.assertIn("Golden Idol", relic_ids)
        self.assertIn("Bloody Idol", relic_ids)
        circlets = [relic for relic in env.relics if str(relic.get("relic_id") or relic.get("id")) == "Circlet"]
        self.assertEqual(len(circlets), 1)
        self.assertEqual(circlets[0]["counter"], 1)

    def test_forgotten_altar_offer_increments_existing_circlet(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("Forgotten Altar", data={"hp_loss": 20})
        env.relics = [make_relic("Golden Idol"), make_relic("Bloody Idol"), make_relic("Circlet", counter=2)]

        env._step_event({"kind": "event", "event_id": "Forgotten Altar", "choice_index": 0})

        relic_ids = [str(relic.get("relic_id") or relic.get("id")) for relic in env.relics]
        self.assertIn("Golden Idol", relic_ids)
        circlets = [relic for relic in env.relics if str(relic.get("relic_id") or relic.get("id")) == "Circlet"]
        self.assertEqual(len(circlets), 1)
        self.assertEqual(circlets[0]["counter"], 3)

    def test_forgotten_altar_shed_blood_heals_from_max_hp_gain_before_damage(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.player.current_hp = 93
        env.player.max_hp = 93
        env.current_event = EventState("Forgotten Altar", data={"hp_loss": 23})

        env._step_event({"kind": "event", "event_id": "Forgotten Altar", "choice_index": 1})

        self.assertEqual(env.player.max_hp, 98)
        self.assertEqual(env.player.current_hp, 75)

    def test_addict_shame_branch_adds_curse_and_relic(self):
        event = EventState("Addict")
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertEqual([card["card_id"] for card in result["add_cards"]], ["Shame"])
        self.assertTrue(result["add_relics"])

    def test_addict_low_gold_action_preserves_sts_button_index(self):
        event = EventState("Addict")
        actions = event.actions(ascension_level=0, max_hp=80, gold=46)

        self.assertEqual([action["name"] for action in actions], ["Take Shame", "Leave"])
        self.assertEqual([action["choice_index"] for action in actions], [0, 1])
        self.assertEqual([action["button_index"] for action in actions], [1, 2])

    def test_addict_low_gold_screen_state_preserves_disabled_locked_button(self):
        env = NativeRunEnv(seed=22, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.gold = 52
        env.current_event = EventState("Addict")

        options = env._event_screen_state()["options"]

        self.assertEqual([option["label"] for option in options], ["Locked", "Rob", "Leave"])
        self.assertEqual([bool(option.get("disabled")) for option in options], [True, False, False])
        self.assertEqual([option.get("choice_index") for option in options], [None, 0, 1])

    def test_addict_screenless_relic_branch_uses_injected_relic_pool(self):
        env = NativeRunEnv(seed=21, ascension_level=0, enable_neow=False)
        env.relic_pools["COMMON"] = ["Bottled Flame", "Anchor"]
        event = EventState("Addict")
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=env.randoms,
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
            relic_drawer=env._pop_relic_from_pool,
            screenless_relic_drawer=lambda tier=None, exclude=None: env._pop_screenless_relic_from_pool("COMMON"),
        )
        self.assertEqual(result["add_relics"][0]["relic_id"], "Anchor")
        self.assertEqual(env.relic_pools["COMMON"], [])

    def test_drug_dealer_transform_opens_two_pick_transform_select(self):
        event = EventState("Drug Dealer")
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[make_card("Strike_R"), make_card("Defend_R"), make_card("Bash")],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertTrue(result["open_card_select"])
        self.assertEqual(result["card_select_mode"], "transform")
        self.assertEqual(result["remaining_picks"], 2)

    def test_drug_dealer_test_subject_counts_bottled_purgeable_cards(self):
        bottled_strike = make_card("Strike_R", uuid="bottled-strike")
        bottled_strike["in_bottle_flame"] = True
        bottled_defend = make_card("Defend_R", uuid="bottled-defend")
        bottled_defend["in_bottle_lightning"] = True
        event = EventState("Drug Dealer")

        actions = event.actions(
            ascension_level=0,
            max_hp=80,
            gold=99,
            deck=[bottled_strike, bottled_defend],
        )

        self.assertIn(1, [action["choice_index"] for action in actions])

    def test_drug_dealer_transform_candidates_include_bottled_purgeable_cards(self):
        bottled_strike = make_card("Strike_R", uuid="bottled-strike")
        bottled_strike["in_bottle_flame"] = True
        bottled_defend = make_card("Defend_R", uuid="bottled-defend")
        bottled_defend["in_bottle_lightning"] = True
        deck = [
            bottled_strike,
            make_card("Necronomicurse", uuid="necro"),
            bottled_defend,
            make_card("CurseOfTheBell", uuid="bell"),
            make_card("AscendersBane", uuid="bane"),
        ]
        event = EventState("Drug Dealer")

        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=deck,
            relics=[],
            potions=[],
        )

        self.assertEqual(result["candidate_indexes"], [0, 2])

    def test_ghosts_accept_loses_half_max_hp_and_adds_apparitions(self):
        event = EventState("Ghosts", data={"hp_loss": 40})
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertEqual(result["max_hp"], 40)
        self.assertEqual(result["hp"], 40)
        self.assertEqual(len(result["add_cards"]), 5)
        self.assertTrue(all(card["card_id"] == "Ghostly" for card in result["add_cards"]))

    def test_vampires_vial_option_removes_blood_vial_and_replaces_strikes(self):
        event = EventState("Vampires", data={"max_hp_loss": 24, "has_vial": True})
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[make_relic("Blood Vial")],
            potions=[],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertEqual(result["remove_relic_id"], "Blood Vial")
        self.assertTrue(result["remove_starter_strikes"])
        self.assertEqual(len(result["add_cards"]), 5)
        self.assertTrue(all(card["card_id"] == "Bite" for card in result["add_cards"]))

    def test_the_library_read_opens_grid_card_select(self):
        event = EventState("The Library", data={"heal_amt": 16, "card_blizz_randomizer": 5})
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=50,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertTrue(result["open_card_select"])
        self.assertEqual(result["card_select_mode"], "library")
        self.assertEqual(len(result["card_select_cards"]), 20)
        self.assertEqual(result["card_blizz_randomizer"], 5)
        self.assertFalse(result["requires_confirm"])

    def test_the_library_card_select_returns_to_event_with_immediate_obtain(self):
        env = NativeRunEnv(seed=12, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.card_blizz_randomizer = 2
        env.current_event = EventState("The Library", data={"heal_amt": 16})
        initial_deck = [dict(card) for card in env.deck]

        env._step_event({"kind": "event", "event_id": "The Library", "choice_index": 0})

        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertEqual(env.current_card_select["mode"], "library")
        self.assertFalse(env.current_card_select["requires_confirm"])
        self.assertEqual(env.card_blizz_randomizer, 2)
        self.assertEqual(len(env._card_select_screen_state()["cards"]), 20)
        action = env.legal_actions()[0]
        env._step_card_select(action)

        self.assertEqual(env.phase, "EVENT")
        self.assertEqual(env.current_event.event_id, "The Library")
        self.assertEqual(len(env.deck), len(initial_deck) + 1)
        self.assertEqual(env.deck[-1]["card_id"], action["card_id"])

    def test_the_library_card_select_replay_choice_index_returns_to_event(self):
        env = NativeRunEnv(seed=12, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("The Library", data={"heal_amt": 16})
        initial_deck = [dict(card) for card in env.deck]

        env._step_event({"kind": "event", "event_id": "The Library", "choice_index": 0})
        selected = dict(env.current_card_select["cards"][3])
        env._step_card_select(
            {
                "kind": "card_select",
                "mode": "library",
                "choice_index": 3,
                "card_index": 3,
                "card_id": selected["card_id"],
            }
        )

        self.assertEqual(env.phase, "EVENT")
        self.assertEqual(env.current_event.event_id, "The Library")
        self.assertEqual(len(env.deck), len(initial_deck) + 1)
        self.assertEqual(env.deck[-1]["card_id"], selected["card_id"])

    def test_the_library_sleep_heal_obeys_mark_and_ignores_magic_flower(self):
        for relic_id, expected_hp in [("Mark of the Bloom", 40), ("Magic Flower", 56)]:
            with self.subTest(relic_id=relic_id):
                env = NativeRunEnv(seed=12, enable_neow=False, start_on_map=True)
                env.phase = "EVENT"
                env.current_room_type = "EventRoom"
                env.player.current_hp = 40
                env.player.max_hp = 80
                env.relics = [make_relic(relic_id)]
                env.current_event = EventState("The Library", data={"heal_amt": 16})

                env._step_event({"kind": "event", "event_id": "The Library", "choice_index": 1})

                self.assertEqual(env.current_event.screen, "COMPLETE")
                self.assertEqual(env.player.current_hp, expected_hp)

    def test_masked_bandits_fight_opens_event_combat_with_red_mask_reward(self):
        event = EventState("Masked Bandits")
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertTrue(result["open_combat"])
        self.assertEqual(result["encounter_name"], "Masked Bandits")
        self.assertEqual(result["event_rewards"]["relics"][0]["relic_id"], "Red Mask")

    def test_masked_bandits_encounter_opens_with_three_monsters(self):
        combat = NativeCombatEnv(seed=2, ascension_level=0, encounter_name="Masked Bandits")
        monster_ids = [monster.monster_id for monster in combat.engine.state.monsters]
        self.assertEqual(monster_ids, ["BanditChild", "BanditLeader", "BanditBear"])

    def test_masked_bandits_pointy_attack_uses_monster_weak_for_each_hit(self):
        combat = NativeCombatEnv(seed=2, ascension_level=0, encounter_name="Masked Bandits")
        engine = combat.engine
        pointy = engine.state.monsters[0]
        combat_engine_mod._add_power(pointy, "Weakened", 2)
        engine.player.current_hp = 76
        engine.player.block = 3

        engine._bandit_pointy_take_turn(pointy)

        self.assertEqual(engine.player.current_hp, 73)
        self.assertEqual(engine.player.block, 0)
        self.assertEqual(_power_amount(pointy, "Weakened"), 1)

    def test_nest_ritual_dagger_branch_costs_hp_and_grants_card(self):
        event = EventState("Nest", data={"gold_gain": 99})
        resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertEqual(result["hp"], 74)
        self.assertEqual([card["card_id"] for card in result["add_cards"]], ["RitualDagger"])

    def test_nloth_trade_swaps_selected_relic_for_gift(self):
        event = EventState("N'loth", data={"choice1_id": "Burning Blood", "choice1_name": "Burning Blood"})
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[make_relic("Burning Blood"), make_relic("Anchor")],
            potions=[],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertEqual(result["remove_relic_id"], "Burning Blood")
        self.assertEqual(result["add_relics"][0]["relic_id"], "Nloth's Gift")

    def test_nloth_existing_gift_keeps_selected_relic_and_obtains_circlet(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("N'loth", data={"choice1_id": "Anchor", "choice1_name": "Anchor"})
        env.relics = [make_relic("Anchor"), make_relic("Nloth's Gift")]

        env._step_event({"kind": "event", "event_id": "N'loth", "choice_index": 0})

        relic_ids = [str(relic.get("relic_id") or relic.get("id")) for relic in env.relics]
        self.assertIn("Anchor", relic_ids)
        self.assertIn("Nloth's Gift", relic_ids)
        circlets = [relic for relic in env.relics if str(relic.get("relic_id") or relic.get("id")) == "Circlet"]
        self.assertEqual(len(circlets), 1)
        self.assertEqual(circlets[0]["counter"], 1)

    def test_nloth_existing_gift_increments_existing_circlet(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("N'loth", data={"choice1_id": "Anchor", "choice1_name": "Anchor"})
        env.relics = [make_relic("Anchor"), make_relic("Nloth's Gift"), make_relic("Circlet", counter=2)]

        env._step_event({"kind": "event", "event_id": "N'loth", "choice_index": 0})

        relic_ids = [str(relic.get("relic_id") or relic.get("id")) for relic in env.relics]
        self.assertIn("Anchor", relic_ids)
        circlets = [relic for relic in env.relics if str(relic.get("relic_id") or relic.get("id")) == "Circlet"]
        self.assertEqual(len(circlets), 1)
        self.assertEqual(circlets[0]["counter"], 3)

    def test_nloth_removing_necronomicon_removes_one_necronomicurse(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("N'loth", data={"choice1_id": "Necronomicon", "choice1_name": "Necronomicon"})
        env.relics = [make_relic("Necronomicon"), make_relic("Anchor")]
        env.deck = [
            make_card("Strike_R", uuid="strike"),
            make_card("Necronomicurse", uuid="necro-1"),
            make_card("Necronomicurse", uuid="necro-2"),
            make_card("Defend_R", uuid="defend"),
        ]

        env._step_event({"kind": "event", "event_id": "N'loth", "choice_index": 0})

        relic_ids = [str(relic.get("relic_id") or relic.get("id")) for relic in env.relics]
        self.assertNotIn("Necronomicon", relic_ids)
        self.assertIn("Nloth's Gift", relic_ids)
        self.assertEqual(sum(1 for card in env.deck if str(card.get("card_id") or "") == "Necronomicurse"), 1)

    def test_the_joust_tracks_bet_and_resolves_gold(self):
        event = EventState("The Joust", screen="EXPLANATION")
        explanation = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "PRE_JOUST")
        self.assertEqual(explanation["gold"], 49)
        event.screen = "JOUST"
        event.data["bet_for"] = False
        event.data["owner_wins"] = False
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=49,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(result["gold"], 149)

    def test_moai_head_golden_idol_branch_exchanges_relic_for_gold(self):
        event = EventState("The Moai Head", data={"hp_amt": 10})
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=60,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[make_relic("Golden Idol")],
            potions=[],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertEqual(result["gold"], 432)
        self.assertEqual(result["remove_relic_id"], "Golden Idol")

    def test_moai_head_sacrifice_heal_hook_obeys_mark_and_clamps_hp(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.relics = [make_relic("Mark of the Bloom")]
        env.current_event = EventState("The Moai Head", data={"hp_amt": 10})

        env._step_event({"kind": "event", "event_id": "The Moai Head", "choice_index": 0})

        self.assertEqual(env.player.max_hp, 70)
        self.assertEqual(env.player.current_hp, 40)

        clamp_env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        clamp_env.phase = "EVENT"
        clamp_env.current_room_type = "EventRoom"
        clamp_env.player.current_hp = 78
        clamp_env.player.max_hp = 80
        clamp_env.relics = [make_relic("Mark of the Bloom")]
        clamp_env.current_event = EventState("The Moai Head", data={"hp_amt": 10})

        clamp_env._step_event({"kind": "event", "event_id": "The Moai Head", "choice_index": 0})

        self.assertEqual(clamp_env.player.max_hp, 70)
        self.assertEqual(clamp_env.player.current_hp, 70)

    def test_tomb_red_mask_purchase_trades_all_gold_for_red_mask(self):
        event = EventState("Tomb of Lord Red Mask")
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=123,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "RESULT")
        self.assertEqual(result["gold"], 0)
        self.assertEqual(result["add_relics"][0]["relic_id"], "Red Mask")

    def test_tomb_red_mask_screen_state_preserves_locked_mask_slot(self):
        env = NativeRunEnv(seed=50, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.gold = 303
        env.current_event = EventState("Tomb of Lord Red Mask")

        options = env._event_screen_state()["options"]
        actions = env.legal_actions()

        self.assertTrue(options[0]["disabled"])
        self.assertNotIn("choice_index", options[0])
        self.assertEqual([option.get("choice_index", index) for index, option in enumerate(options)], [0, 0, 1])
        self.assertEqual([action["choice_index"] for action in actions], [0, 1])

    def test_tomb_red_mask_leave_returns_directly_to_map(self):
        env = NativeRunEnv(seed=50, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.gold = 303
        env.current_event = EventState("Tomb of Lord Red Mask")

        env.step({"kind": "event", "event_id": "Tomb of Lord Red Mask", "choice_index": 1})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.current_room_type, "Map")
        self.assertEqual(env.gold, 303)

    def test_winding_halls_writhe_branch_heals_and_adds_curse(self):
        event = EventState("Winding Halls", data={"hp_amt": 10, "heal_amt": 20, "max_hp_amt": 4})
        resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=40,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=40,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertEqual(result["hp"], 40)
        self.assertEqual(result["heal_player"], 20)
        self.assertEqual([card["card_id"] for card in result["add_cards"]], ["Writhe"])

    def test_winding_halls_writhe_heal_hook_obeys_mark_and_ignores_magic_flower(self):
        for relic_id, expected_hp in [("Mark of the Bloom", 40), ("Magic Flower", 60)]:
            with self.subTest(relic_id=relic_id):
                env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
                env.phase = "EVENT"
                env.current_room_type = "EventRoom"
                env.player.current_hp = 40
                env.player.max_hp = 80
                env.relics = [make_relic(relic_id)]
                env.current_event = EventState(
                    "Winding Halls",
                    screen="CHOICE",
                    data={"hp_amt": 10, "heal_amt": 20, "max_hp_amt": 4},
                )

                env._step_event({"kind": "event", "event_id": "Winding Halls", "choice_index": 1})

                self.assertEqual(env.current_event.screen, "COMPLETE")
                self.assertEqual(env.player.current_hp, expected_hp)
                self.assertIn("Writhe", [card["card_id"] for card in env.deck])

    def test_mindbloom_upgrade_branch_upgrades_all_and_adds_mark(self):
        event = EventState("MindBloom", data={"late_branch": False})
        deck = [make_card("Strike_R"), make_card("Inflame"), make_card("Doubt")]
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=deck,
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "LEAVE")
        self.assertEqual(result["upgrade_indexes"], [0, 1])
        self.assertEqual(result["add_relics"][0]["relic_id"], "Mark of the Bloom")

    def test_mindbloom_fight_opens_boss_combat_with_rare_relic_reward(self):
        event = EventState("MindBloom", data={"late_branch": False})
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "FIGHT")
        self.assertTrue(result["open_combat"])
        self.assertIn(result["encounter_name"], {"The Guardian", "Hexaghost", "Slime Boss"})
        self.assertEqual(len(result["event_rewards"]["relics"]), 1)

    def test_mindbloom_late_heal_obeys_mark_and_adds_doubt(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.relics = [make_relic("Mark of the Bloom")]
        env.current_event = EventState("MindBloom", data={"late_branch": True})

        env._step_event({"kind": "event", "event_id": "MindBloom", "choice_index": 2})

        self.assertEqual(env.current_event.screen, "LEAVE")
        self.assertEqual(env.player.current_hp, 40)
        self.assertIn("Doubt", [card["card_id"] for card in env.deck])

    def test_falling_removes_preselected_skill_card(self):
        event = EventState("Falling", data={"skill_uuid": "skill-1", "skill_name": "Shrug It Off"})
        resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "RESULT")
        self.assertEqual(result["remove_card_uuid"], "skill-1")

    def test_falling_setup_ignores_bottled_cards_for_type_options(self):
        bottled_skill = make_card("Shrug It Off", uuid="bottled-skill")
        bottled_skill["in_bottle_lightning"] = True
        event = _initialize_event_state(
            EventState("Falling"),
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            floor=40,
            gold=99,
            deck=[make_card("Strike_R", uuid="attack"), bottled_skill],
            relics=[],
            potions=[],
            max_hp=80,
        )

        self.assertEqual(event.data.get("attack_uuid"), "attack")
        self.assertNotIn("skill_uuid", event.data)
        event.screen = "CHOICE"
        self.assertEqual([action["choice_index"] for action in event.actions(ascension_level=0, max_hp=80)], [2])

    def test_falling_setup_selects_only_non_bottled_cards_of_type(self):
        bottled_skill = make_card("Shrug It Off", uuid="bottled-skill")
        bottled_skill["in_bottle_lightning"] = True
        event = _initialize_event_state(
            EventState("Falling"),
            randoms=NativeRandomSet(seed=3),
            ascension_level=0,
            floor=40,
            gold=99,
            deck=[bottled_skill, make_card("Defend_R", uuid="normal-skill")],
            relics=[],
            potions=[],
            max_hp=80,
        )

        self.assertEqual(event.data.get("skill_uuid"), "normal-skill")

    def test_falling_setup_consumes_rng_only_for_non_empty_non_bottled_types(self):
        bottled_skill = make_card("Shrug It Off", uuid="bottled-skill")
        bottled_skill["in_bottle_lightning"] = True
        randoms = NativeRandomSet(seed=4)

        event = _initialize_event_state(
            EventState("Falling"),
            randoms=randoms,
            ascension_level=0,
            floor=40,
            gold=99,
            deck=[
                make_card("Strike_R", uuid="attack"),
                bottled_skill,
                make_card("Inflame", uuid="power"),
            ],
            relics=[],
            potions=[],
            max_hp=80,
        )

        self.assertEqual(event.data.get("attack_uuid"), "attack")
        self.assertNotIn("skill_uuid", event.data)
        self.assertEqual(event.data.get("power_uuid"), "power")
        self.assertEqual(randoms.stream("misc").counter, 2)

    def test_fountain_of_cleansing_removes_normal_curses_but_keeps_special_ones(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("Fountain of Cleansing")
        env.deck = [
            make_card("Regret", uuid="regret"),
            make_card("AscendersBane", uuid="bane"),
            make_card("Necronomicurse", uuid="necro"),
            make_card("Strike_R", uuid="strike"),
        ]

        env.step({"kind": "event", "event_id": "Fountain of Cleansing", "choice_index": 0})

        remaining = {card["card_id"] for card in env.deck}
        self.assertNotIn("Regret", remaining)
        self.assertIn("AscendersBane", remaining)
        self.assertIn("Necronomicurse", remaining)

    def test_purgeable_candidate_indexes_exclude_protected_curses_but_keep_normal_curses(self):
        bottled_bash = make_card("Bash", uuid="bottled-bash")
        bottled_bash["in_bottle_flame"] = True
        deck = [
            make_card("Strike_R", uuid="strike"),
            make_card("Necronomicurse", uuid="necro"),
            make_card("CurseOfTheBell", uuid="bell"),
            make_card("AscendersBane", uuid="bane"),
            make_card("Injury", uuid="injury"),
            make_card("Wound", uuid="wound"),
            bottled_bash,
        ]

        self.assertEqual(_purgeable_candidate_indexes(deck), [0, 4])

    def test_knowing_skull_gold_option_costs_hp_and_grants_gold(self):
        event = EventState("Knowing Skull", screen="ASK", data={"potion_cost": 6, "gold_cost": 6, "card_cost": 6, "leave_cost": 6})
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(result["gold"], 189)
        self.assertEqual(result["hp"], 74)
        self.assertEqual(event.data["gold_cost"], 7)
        self.assertTrue(result["stay_event"])

    def test_knowing_skull_initializes_to_intro_before_ask_screen(self):
        event = _initialize_event_state(
            EventState("Knowing Skull"),
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            floor=22,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
            max_hp=80,
        )

        actions = event.actions(ascension_level=0, max_hp=80)

        self.assertEqual(event.screen, "INTRO_1")
        self.assertEqual(actions[0]["name"], "Approach")

    def test_knowing_skull_reward_choice_keeps_event_open(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState(
            "Knowing Skull",
            screen="ASK",
            data={"potion_cost": 6, "gold_cost": 6, "card_cost": 6, "leave_cost": 6},
        )
        env.gold = 99
        env.player.current_hp = 80

        env._step_event({"kind": "event", "event_id": "Knowing Skull", "choice_index": 1})

        self.assertEqual(env.phase, "EVENT")
        self.assertEqual(env.current_room_type, "EventRoom")
        self.assertIsNotNone(env.current_event)
        self.assertEqual(env.current_event.screen, "ASK")
        self.assertEqual(env.current_event.data["gold_cost"], 7)
        self.assertEqual(env.gold, 189)
        self.assertEqual(env.player.current_hp, 74)

    def test_knowing_skull_card_option_grants_colorless_card(self):
        event = EventState("Knowing Skull", data={"potion_cost": 6, "gold_cost": 6, "card_cost": 6, "leave_cost": 6}, screen="ASK")
        result = resolve_event_choice(
            event,
            action_index=2,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(len(result["add_cards"]), 1)
        self.assertEqual(str(result["add_cards"][0]["color"]), "COLORLESS")

    def test_cursed_tome_last_page_can_open_reward_relic(self):
        event = EventState("Cursed Tome", screen="LAST_PAGE", data={"final_dmg": 10, "damage_taken": 6})
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "END")
        self.assertTrue(result["open_rewards"])
        self.assertTrue(result["relic_rewards"])

    def test_forgotten_altar_shed_blood_increases_max_hp(self):
        event = EventState("Forgotten Altar", data={"hp_loss": 20})
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(result["max_hp"], 85)
        self.assertEqual(result["hp"], 65)

    def test_note_for_yourself_adds_saved_card_then_opens_purge(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("NoteForYourself", data={"obtain_card": make_card("Iron Wave")})
        start_deck_size = len(env.deck)

        env.step({"kind": "event", "event_id": "NoteForYourself", "choice_index": 0})
        self.assertEqual(env.current_event.screen, "CHOOSE")
        env.step({"kind": "event", "event_id": "NoteForYourself", "choice_index": 0})
        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertFalse(env.state()["screen_state"]["for_purge"])
        self.assertEqual(len(env.deck), start_deck_size + 1)
        self.assertTrue(any(card["card_id"] == "Iron Wave" for card in env.deck))
        self.assertTrue(env.current_card_select["candidate_indexes"])

    def test_note_for_yourself_purge_updates_saved_note_card_state(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.note_for_yourself_card_id = "Iron Wave"
        env.note_for_yourself_upgrades = 0
        env.current_event = EventState("NoteForYourself", screen="CHOOSE", data={"obtain_card": make_card("Ghostly", upgrades=1)})

        env.step({"kind": "event", "event_id": "NoteForYourself", "choice_index": 0})
        target_index = next(index for index, card in enumerate(env.deck) if card["card_id"] == "Strike_R")
        env.step({"kind": "card_select", "target_index": target_index})

        self.assertEqual(env.note_for_yourself_card_id, "Strike_R")
        self.assertEqual(env.note_for_yourself_upgrades, 0)

    def test_generate_event_for_act_uses_saved_note_card_state(self):
        event = generate_event_for_act(
            NativeRandomSet(seed=2),
            ascension_level=0,
            act=1,
            floor=10,
            gold=99,
            relics=[],
            deck=[],
            potions=[],
            current_hp=80,
            max_hp=80,
            current_node_y=None,
            map_height=15,
            event_list=[],
            shrine_list=[],
            special_one_time_event_list=["NoteForYourself"],
            note_for_yourself_card_id="Ghostly",
            note_for_yourself_upgrades=1,
        )

        self.assertEqual(event.event_id, "NoteForYourself")
        self.assertEqual(event.data["obtain_card"]["card_id"], "Ghostly")
        self.assertEqual(event.data["obtain_card"]["upgrades"], 1)

    def test_liars_game_agree_grants_gold_and_doubt(self):
        event = EventState("Liars Game")
        first = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            potions=[],
        )
        self.assertEqual(event.screen, "AGREE")
        self.assertEqual(first["gold"], 99)

        second = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            potions=[],
        )
        self.assertEqual(second["gold"], 274)
        self.assertEqual(second["add_cards"][0]["card_id"], "Doubt")

    def test_we_meet_again_initializes_random_offerings_and_relic(self):
        event = generate_event_for_act(
            NativeRandomSet(seed=4),
            ascension_level=0,
            act=1,
            floor=10,
            gold=120,
            relics=[],
            deck=[make_card("Strike_R"), make_card("Clothesline"), make_card("Defend_R")],
            potions=[roll_random_potion(NativeRandomSet(seed=3), player_class="IRONCLAD")],
            current_hp=80,
            max_hp=80,
            current_node_y=7,
            map_height=15,
            event_list=[],
            shrine_list=[],
            special_one_time_event_list=["WeMeetAgain"],
        )
        self.assertEqual(event.event_id, "WeMeetAgain")
        self.assertIsNotNone(event.data.get("potion_index"))
        self.assertGreater(int(event.data.get("gold_amount") or 0), 0)
        self.assertIsNotNone(event.data.get("card_uuid"))
        self.assertNotIn("given_relic", event.data)

    def test_we_meet_again_initialization_does_not_use_injected_screenless_relic_pool(self):
        env = NativeRunEnv(seed=22, ascension_level=0, enable_neow=False)
        env.relic_pools["COMMON"] = ["Bottled Flame", "Anchor"]
        relic_draw_calls = []

        def screenless_relic_drawer(tier=None, exclude=None):
            relic_draw_calls.append((tier, exclude))
            return env._pop_screenless_relic_from_pool("COMMON")

        event = generate_event_for_act(
            env.randoms,
            ascension_level=0,
            act=2,
            floor=10,
            gold=120,
            relics=[],
            deck=[make_card("Strike_R"), make_card("Clothesline"), make_card("Defend_R")],
            potions=[roll_random_potion(NativeRandomSet(seed=3), player_class="IRONCLAD")],
            current_hp=80,
            max_hp=80,
            current_node_y=7,
            map_height=15,
            event_list=[],
            shrine_list=[],
            special_one_time_event_list=["WeMeetAgain"],
            relic_drawer=env._pop_relic_from_pool,
            screenless_relic_drawer=screenless_relic_drawer,
        )
        self.assertEqual(event.event_id, "WeMeetAgain")
        self.assertNotIn("given_relic", event.data)
        self.assertEqual(relic_draw_calls, [])
        self.assertEqual(env.relic_pools["COMMON"], ["Bottled Flame", "Anchor"])

    def test_random_face_relic_uses_java_shuffle_semantics(self):
        randoms = NativeRandomSet(seed=29)
        expected_rng = randoms.duplicate_stream("misc")
        ordered = ["CultistMask", "FaceOfCleric", "GremlinMask", "NlothsMask", "SsserpentHead"]
        java_shuffle_in_place(ordered, int(expected_rng.random_long()))

        actual = _random_face_relic(randoms, relic_ids=set())

        self.assertEqual(actual["relic_id"], ordered[0])

    def test_lab_opens_reward_screen_with_potions_only(self):
        event = EventState("Lab")
        randoms = NativeRandomSet(seed=2)
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=randoms,
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            potions=[],
        )
        self.assertTrue(result["open_rewards"])
        self.assertTrue(result["leave_event_after_rewards"])
        self.assertEqual(len(result["potions"]), 3)
        expected_randoms = NativeRandomSet(seed=2)
        expected = [draw_random_potion(expected_randoms, player_class="IRONCLAD")["potion_id"] for _ in range(3)]
        self.assertEqual([p["potion_id"] for p in result["potions"]], expected)

    def test_lab_reward_screen_returns_to_map_after_potion_rewards(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("Lab")
        env.potions = [
            {"id": "PowerPotion", "potion_id": "PowerPotion", "name": "Power Potion", "requires_target": False},
            {"id": "Explosive Potion", "potion_id": "Explosive Potion", "name": "Explosive Potion", "requires_target": True},
            {"id": "GamblersBrew", "potion_id": "GamblersBrew", "name": "Gambler's Brew", "requires_target": False},
        ]

        env.step(env.legal_actions()[0])

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertIsNone(env.current_event)
        self.assertTrue(env.reward_potions)

        env.step({"kind": "proceed", "name": "PROCEED"})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.current_room_type, "Map")

    def test_bonfire_rare_offer_grants_max_hp_and_full_heal(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.current_event = EventState("Bonfire Elementals", screen="COMPLETE")
        env.current_card_select = {
            "mode": "purge",
            "candidate_indexes": [0],
            "return_phase": "EVENT",
            "source": "event",
            "clear_event_on_finish": False,
            "selection_effect": "bonfire",
        }
        env.phase = "CARD_SELECT"
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.deck[0] = make_card("Offering")

        env.step({"kind": "card_select", "target_index": 0})

        self.assertEqual(env.player.max_hp, 90)
        self.assertEqual(env.player.current_hp, 90)

    def test_removing_parasite_from_deck_decreases_max_hp_and_clamps_current_hp(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.deck = [make_card("Strike_R", uuid="strike"), make_card("Parasite", uuid="parasite")]
        env.player.current_hp = 80
        env.player.max_hp = 80

        removed = env._remove_deck_card_by_uuid("parasite")

        self.assertIsNotNone(removed)
        self.assertEqual(env.player.max_hp, 77)
        self.assertEqual(env.player.current_hp, 77)
        self.assertNotIn("Parasite", [card["card_id"] for card in env.deck])

    def test_shop_purge_parasite_applies_remove_from_master_deck_side_effect(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.deck = [make_card("Parasite", uuid="parasite"), make_card("Strike_R", uuid="strike")]
        env.gold = 100
        env.player.current_hp = 50
        env.player.max_hp = 80
        env.current_shop = ShopState(purge_cost=75, purge_available=True)

        env._step_shop({"kind": "shop", "item_kind": "purge", "target_index": 0})

        self.assertEqual(env.player.max_hp, 77)
        self.assertEqual(env.player.current_hp, 50)
        self.assertEqual(env.gold, 25)
        self.assertNotIn("Parasite", [card["card_id"] for card in env.deck])

    def test_card_select_transform_parasite_applies_remove_from_master_deck_side_effect(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.deck = [make_card("Parasite", uuid="parasite")]
        env.player.current_hp = 80
        env.player.max_hp = 80
        env.current_card_select = {
            "mode": "transform",
            "candidate_indexes": [0],
            "return_phase": "MAP",
            "source": "event",
            "remaining_picks": 1,
            "total_picks": 1,
        }
        env.phase = "CARD_SELECT"

        env.step({"kind": "card_select", "target_index": 0})

        self.assertEqual(env.player.max_hp, 77)
        self.assertEqual(env.player.current_hp, 77)
        self.assertEqual(len(env.deck), 1)
        self.assertNotEqual(env.deck[0]["card_id"], "Parasite")

    def test_bonfire_curse_offer_uses_circlet_counter_for_duplicate_spirit_poop_reward(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        circlet = make_relic("Circlet")
        circlet["counter"] = 2
        env.relics = [make_relic("Spirit Poop"), circlet]

        env._apply_bonfire_reward(make_card("Injury"))

        circlets = [relic for relic in env.relics if relic["relic_id"] == "Circlet"]
        self.assertEqual(len(circlets), 1)
        self.assertEqual(circlets[0]["counter"], 3)

    def test_bonfire_common_offer_heal_respects_mark_of_the_bloom(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.relics = [make_relic("Mark of the Bloom")]
        env.player.current_hp = 40
        env.player.max_hp = 80

        env._apply_bonfire_reward(make_card("Anger"))

        self.assertEqual(env.player.current_hp, 40)
        self.assertEqual(env.player.max_hp, 80)

    def test_bonfire_common_offer_does_not_apply_magic_flower_event_heal_multiplier(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.relics = [make_relic("Magic Flower")]
        env.player.current_hp = 40
        env.player.max_hp = 80

        env._apply_bonfire_reward(make_card("Anger"))

        self.assertEqual(env.player.current_hp, 45)

    def test_bonfire_common_offer_with_magic_flower_still_respects_mark_of_the_bloom(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.relics = [make_relic("Magic Flower"), make_relic("Mark of the Bloom")]
        env.player.current_hp = 40
        env.player.max_hp = 80

        env._apply_bonfire_reward(make_card("Anger"))

        self.assertEqual(env.player.current_hp, 40)

    def test_bonfire_uncommon_offer_full_heal_respects_mark_of_the_bloom(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.relics = [make_relic("Mark of the Bloom")]
        env.player.current_hp = 40
        env.player.max_hp = 80

        env._apply_bonfire_reward(make_card("Inflame"))

        self.assertEqual(env.player.current_hp, 40)
        self.assertEqual(env.player.max_hp, 80)

    def test_bonfire_rare_offer_max_hp_gain_respects_mark_of_the_bloom_heal_block(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.relics = [make_relic("Mark of the Bloom")]
        env.player.current_hp = 40
        env.player.max_hp = 80

        env._apply_bonfire_reward(make_card("Offering"))

        self.assertEqual(env.player.max_hp, 90)
        self.assertEqual(env.player.current_hp, 40)

    def test_run_state_serializes_live_event_id(self):
        env = NativeRunEnv(seed=1, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_event = EventState("The Cleric")

        state = env.state()

        self.assertEqual(state["event_id"], "The Cleric")

    def test_encounter_catalog_parses_real_exordium_pools(self):
        exordium = act_encounter_def("Exordium")
        self.assertTrue(any(item.encounter_name == "Cultist" for item in exordium.weak))
        self.assertTrue(any(item.encounter_name == "Blue Slaver" for item in exordium.strong))
        self.assertTrue(any(item.encounter_name == "Lagavulin" for item in exordium.elite))
        self.assertEqual((exordium.weak_count, exordium.strong_count, exordium.elite_count), (3, 12, 10))
        self.assertEqual(
            {rule.trigger_name: rule.excluded_names for rule in exordium.strong_exclusions},
            {
                "Looter": ("Exordium Thugs",),
                "Blue Slaver": ("Red Slaver", "Exordium Thugs"),
                "2 Louse": ("3 Louse",),
                "Small Slimes": ("Large Slime", "Lots of Slimes"),
            },
        )
        self.assertEqual(set(exordium.bosses), {"The Guardian", "Hexaghost", "Slime Boss"})
        city = act_encounter_def("TheCity")
        self.assertTrue(any(item.encounter_name == "Chosen" for item in city.weak))
        self.assertTrue(any(item.encounter_name == "Book of Stabbing" for item in city.elite))
        self.assertEqual((city.weak_count, city.strong_count, city.elite_count), (2, 12, 10))
        self.assertEqual(
            {rule.trigger_name: rule.excluded_names for rule in city.strong_exclusions},
            {
                "Spheric Guardian": ("Sentry and Sphere",),
                "3 Byrds": ("Chosen and Byrds",),
                "Chosen": ("Chosen and Byrds", "Cultist and Chosen"),
            },
        )
        beyond = act_encounter_def("TheBeyond")
        self.assertEqual((beyond.weak_count, beyond.strong_count, beyond.elite_count), (2, 12, 10))
        self.assertEqual(
            {rule.trigger_name: rule.excluded_names for rule in beyond.strong_exclusions},
            {
                "3 Darklings": ("3 Darklings",),
                "Orb Walker": ("Orb Walker",),
                "3 Shapes": ("4 Shapes",),
            },
        )

    def test_encounter_generation_tracks_normal_elite_and_boss_lists(self):
        monsters, elites, bosses = generate_exordium_monster_lists(NativeRandomSet(seed=2))
        self.assertEqual(len(monsters), 16)
        self.assertEqual(len(elites), 10)
        self.assertEqual(len(bosses), 3)
        self.assertTrue(set(bosses).issubset({"The Guardian", "Hexaghost", "Slime Boss"}))
        city_monsters, city_elites, city_bosses = generate_monster_lists_for_act(NativeRandomSet(seed=2), 2)
        self.assertEqual(len(city_monsters), 15)
        self.assertEqual(len(city_elites), 10)
        self.assertTrue(set(city_bosses).issubset({"Automaton", "Collector", "Champ"}))
        beyond_monsters, beyond_elites, beyond_bosses = generate_monster_lists_for_act(NativeRandomSet(seed=2), 3)
        self.assertEqual(len(beyond_monsters), 15)
        self.assertEqual(len(beyond_elites), 10)
        self.assertTrue(set(beyond_bosses).issubset({"Awakened One", "Time Eater", "Donu and Deca"}))

    def test_encounter_generation_accepts_explicit_dungeon_id(self):
        elites, mirrored_elites, bosses = generate_monster_lists_for_dungeon(NativeRandomSet(seed=2), "TheEnding")
        self.assertEqual(elites, mirrored_elites)
        self.assertEqual(elites, ["Shield and Spear"])
        self.assertEqual(bosses, ["The Heart"])

    def test_bronze_automaton_starts_with_artifact(self):
        combat = NativeCombatEnv(seed=3, ascension_level=0, encounter_name="Automaton")
        automaton = combat.engine.state.monsters[0]

        self.assertEqual(automaton.monster_id, "BronzeAutomaton")
        self.assertEqual(_power_amount(automaton, "Artifact"), 3)

    def test_bronze_orb_stasis_takes_cards_and_rolls_followup_in_action_order(self):
        engine = CombatEngine(
            encounter_name="Automaton",
            randoms=NativeRandomSet(seed=36),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.draw_pile = [
            make_card(card_id, uuid=f"stasis-{index}")
            for index, card_id in enumerate(
                ["Double Tap", "Strike_R", "Spot Weakness", "Seeing Red", "Strike_R", "Havoc", "Shrug It Off", "Defend_R"]
            )
        ]
        engine._automaton_take_turn(engine.state.monsters[0])
        engine._update_monster_intents()

        self.assertEqual([monster.next_move for monster in engine.state.monsters], ["STASIS", "FLAIL", "STASIS"])
        engine._take_monster_turns()

        self.assertEqual([card["card_id"] for card in engine.state.draw_pile], ["Strike_R", "Seeing Red", "Strike_R", "Havoc", "Shrug It Off", "Defend_R"])
        self.assertEqual([monster.next_move for monster in engine.state.monsters], ["BEAM", "BOOST", "SUPPORT"])
        self.assertEqual([monster.intent for monster in engine.state.monsters], ["ATTACK", "DEFEND_BUFF", "DEFEND"])
        self.assertEqual(
            [
                [power.get("card", {}).get("card_id") for power in monster.powers if power.get("power_id") == "Stasis"]
                for monster in (engine.state.monsters[0], engine.state.monsters[2])
            ],
            [["Double Tap"], ["Spot Weakness"]],
        )

    def test_bronze_orb_and_automaton_rolls_consume_action_order_ai(self):
        engine = CombatEngine(
            encounter_name="Automaton",
            randoms=NativeRandomSet(seed=44, floor=33),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.draw_pile = [make_card("Immolate", uuid="stasis-rare"), make_card("Feed", uuid="stasis-feed")]
        engine._automaton_take_turn(engine.state.monsters[0])

        self.assertEqual([entry.result for entry in engine.randoms.stream("ai").calls], [80, 55, 93])
        self.assertEqual([monster.next_move for monster in engine.state.monsters], ["STASIS", "FLAIL", "STASIS"])

        engine._take_monster_turns()

        self.assertEqual([entry.result for entry in engine.randoms.stream("ai").calls], [80, 55, 93, 68, 47, 10, 97])
        self.assertEqual([monster.next_move for monster in engine.state.monsters], ["BEAM", "BOOST", "SUPPORT"])
        self.assertEqual([monster.intent for monster in engine.state.monsters], ["ATTACK", "DEFEND_BUFF", "DEFEND"])

        engine._take_monster_turns()

        self.assertEqual([entry.result for entry in engine.randoms.stream("ai").calls], [80, 55, 93, 68, 47, 10, 97, 97, 63, 42])
        self.assertEqual([monster.next_move for monster in engine.state.monsters], ["SUPPORT", "FLAIL", "BEAM"])
        self.assertEqual([monster.intent for monster in engine.state.monsters], ["DEFEND", "ATTACK", "ATTACK"])

    def test_bronze_automaton_flail_damage_uses_strength_power(self):
        engine = CombatEngine(
            encounter_name="Automaton",
            randoms=NativeRandomSet(seed=36),
            ascension_level=0,
            player=PlayerState(current_hp=27, max_hp=80),
            master_deck=[],
        )
        engine.player.block = 16
        automaton = engine.state.monsters[0]
        automaton.next_move = "FLAIL"
        automaton.powers.append({"power_id": "Strength", "id": "Strength", "name": "Strength", "amount": 3, "misc": 3})

        engine._automaton_take_turn(automaton)

        self.assertEqual(engine.player.current_hp, 23)
        self.assertEqual(engine.player.block, 0)

    def test_bronze_automaton_hyper_beam_followup_does_not_increment_num_turns(self):
        engine = CombatEngine(
            encounter_name="Automaton",
            randoms=NativeRandomSet(seed=36),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        automaton = engine.state.monsters[0]
        automaton.next_move = "HYPER_BEAM"
        automaton.meta["num_turns"] = 0
        automaton.meta["first_turn"] = False

        engine._automaton_take_turn(automaton)

        self.assertEqual(automaton.next_move, "STUNNED")
        self.assertEqual(automaton.meta["num_turns"], 0)

    def test_neow_generates_real_options_and_card_reward_chain(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=True)

        self.assertEqual(env.phase, "NEOW")
        actions = env.legal_actions()
        self.assertEqual(len(actions), 4)
        self.assertEqual(actions[0]["kind"], "neow")

        reward_choice = next(action for action in actions if action["bonus"] == "THREE_CARDS")
        env.step(reward_choice)
        self.assertEqual(env.phase, "CARD_REWARD")
        reward_actions = env.legal_actions()
        self.assertEqual(sum(1 for action in reward_actions if action["kind"] == "card_reward"), 3)
        self.assertIn("skip", {action["kind"] for action in reward_actions})

        env.step(next(action for action in reward_actions if action["kind"] == "card_reward"))
        self.assertEqual(env.phase, "NEOW")
        env.step(env.legal_actions()[0])
        self.assertEqual(env.phase, "MAP")

    def test_neow_card_reward_skip_returns_to_continue_without_card(self):
        for bonus in ("THREE_CARDS", "RANDOM_COLORLESS", "THREE_RARE_CARDS"):
            with self.subTest(bonus=bonus):
                env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=True)
                starting_deck = [card["card_id"] for card in env.deck]
                env.step({"kind": "neow", "bonus": bonus, "drawback": "NONE"})
                self.assertEqual(env.phase, "CARD_REWARD")
                reward_actions = env.legal_actions()
                self.assertIn("skip", {action["kind"] for action in reward_actions})

                env.step(next(action for action in reward_actions if action["kind"] == "skip"))

                self.assertEqual([card["card_id"] for card in env.deck], starting_deck)
                self.assertEqual(env.phase, "NEOW")
                self.assertTrue(env.neow_pending_continue)
                env.step(env.legal_actions()[0])
                self.assertEqual(env.phase, "MAP")

    def test_seed1_neow_colorless_reward_uses_real_game_rng_split(self):
        env = NativeRunEnv(seed=1, ascension_level=0, enable_neow=True)
        reward_choice = next(action for action in env.legal_actions() if action["bonus"] == "RANDOM_COLORLESS")
        env.step(reward_choice)
        self.assertEqual(env.phase, "CARD_REWARD")
        reward_cards = [action["card_id"] for action in env.legal_actions() if action["kind"] == "card_reward"]
        self.assertEqual(len(reward_cards), 3)
        self.assertEqual(reward_cards[2], "Flash of Steel")
        self.assertIn("skip", {action["kind"] for action in env.legal_actions()})

    def test_seed3960_neow_colorless_reward_matches_live_game_cards(self):
        env = NativeRunEnv(seed=3960482443532127989, ascension_level=0, enable_neow=True)
        reward_choice = next(action for action in env.legal_actions() if action["bonus"] == "RANDOM_COLORLESS")

        env.step(reward_choice)

        self.assertEqual(env.phase, "CARD_REWARD")
        reward_cards = [action["card_id"] for action in env.legal_actions() if action["kind"] == "card_reward"]
        self.assertEqual(reward_cards, ["Trip", "Panacea", "PanicButton"])
        self.assertNotIn("Madness", reward_cards)
        self.assertIn("skip", {action["kind"] for action in env.legal_actions()})

    def test_seed12_neow_three_cards_duplicate_redraw_keeps_rarity(self):
        env = NativeRunEnv(seed=12, ascension_level=0, enable_neow=True)
        reward_choice = next(action for action in env.legal_actions() if action["bonus"] == "THREE_CARDS")

        env.step(reward_choice)

        reward_cards = [action["card_id"] for action in env.legal_actions() if action["kind"] == "card_reward"]
        self.assertEqual(reward_cards, ["Uppercut", "Anger", "Heavy Blade"])
        neow_calls = env.randoms.stream("neow").calls
        self.assertEqual(neow_calls[9].method, "random_boolean")
        self.assertEqual(neow_calls[10].method, "random")
        self.assertEqual(neow_calls[10].result, 0)
        self.assertEqual(neow_calls[11].method, "random")
        self.assertEqual(neow_calls[11].result, 17)

    def test_neow_rare_card_reward_uses_runtime_card_pool(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=True)
        env.card_pools["CLASS_RARE"] = ["Feed"]
        env.card_pools["RED_RARE"] = ["Barricade"]
        env.neow_options = [
            NeowOption(
                choice_index=0,
                bonus=NeowRewardType.ONE_RANDOM_RARE_CARD,
                drawback=NeowDrawback.NONE,
                bonus_text="Obtain a random rare card.",
            )
        ]
        env.step(env.legal_actions()[0])
        self.assertTrue(any(card["card_id"] == "Feed" for card in env.deck))

    def test_neow_transform_card_opens_card_select_and_returns_to_map(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=True)
        env.neow_options = [
            NeowOption(
                choice_index=0,
                bonus=NeowRewardType.TRANSFORM_CARD,
                drawback=NeowDrawback.NONE,
                bonus_text="Transform a card.",
            )
        ]
        action = env.legal_actions()[0]
        old_ids = [card["card_id"] for card in env.deck]
        env.step(action)
        self.assertEqual(env.phase, "CARD_SELECT")
        select_action = env.legal_actions()[0]
        transformed_index = int(select_action["target_index"])
        old_card_id = env.deck[transformed_index]["card_id"]
        old_count = sum(1 for card in env.deck if card["card_id"] == old_card_id)
        env.step(select_action)
        self.assertEqual(env.phase, "NEOW")
        env.step(env.legal_actions()[0])
        self.assertEqual(env.phase, "MAP")
        self.assertEqual(len(env.deck), len(old_ids))
        self.assertEqual(sum(1 for card in env.deck if card["card_id"] == old_card_id), old_count - 1)
        self.assertNotEqual(env.deck[-1]["card_id"], old_card_id)

    def test_neow_transform_uses_runtime_common_and_source_uncommon_pools(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=True)
        env.source_card_pools = initialize_source_card_pools()
        env.card_pools["CLASS_COMMON"] = []
        env.card_pools["RED_COMMON"] = []
        env.source_card_pools["SRC_COMMON"] = ["Clash"]
        env.source_card_pools["SRC_UNCOMMON"] = ["Dropkick"]
        env.source_card_pools["SRC_RARE"] = []
        env.neow_options = [
            NeowOption(
                choice_index=0,
                bonus=NeowRewardType.TRANSFORM_CARD,
                drawback=NeowDrawback.NONE,
                bonus_text="Transform a card.",
            )
        ]
        env.step(env.legal_actions()[0])
        self.assertEqual(env.phase, "CARD_SELECT")
        env.step(env.legal_actions()[0])
        self.assertTrue(any(card["card_id"] == "Dropkick" for card in env.deck))

    def test_neow_transform_seed10_defend_matches_source_available_pool_order(self):
        env = NativeRunEnv(seed=10, ascension_level=0, enable_neow=True)
        transform_action = next(action for action in env.legal_actions() if action.get("bonus") == NeowRewardType.TRANSFORM_CARD)
        env.step(transform_action)
        select_action = next(action for action in env.legal_actions() if int(action.get("target_index") or -1) == 5)
        env.step(select_action)
        self.assertEqual([card["card_id"] for card in env.deck], [
            "Strike_R",
            "Strike_R",
            "Strike_R",
            "Strike_R",
            "Strike_R",
            "Defend_R",
            "Defend_R",
            "Defend_R",
            "Bash",
            "True Grit",
        ])

    def test_neow_transform_curse_fallback_uses_runtime_card_pools(self):
        randoms = NativeRandomSet(seed=31)
        runtime_pools = initialize_runtime_card_pools()
        runtime_pools["CURSE"] = ["Doubt"]
        source_pools = initialize_source_card_pools(runtime_pools)
        source_pools["SRC_CURSE"] = []

        transformed = transform_card(
            randoms,
            make_card("Injury"),
            runtime_pools=runtime_pools,
            source_pools=source_pools,
        )

        self.assertEqual(transformed["card_id"], "Doubt")

    def test_neow_three_small_potions_opens_reward_screen_with_only_potions(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=True)
        env.neow_options = [
            NeowOption(
                choice_index=0,
                bonus=NeowRewardType.THREE_SMALL_POTIONS,
                drawback=NeowDrawback.NONE,
                bonus_text="Obtain 3 random potions.",
            )
        ]

        env.step(env.legal_actions()[0])

        self.assertEqual(env.phase, "CARD_REWARD")
        actions = env.legal_actions()
        self.assertEqual(sum(1 for action in actions if action["kind"] == "reward_potion"), 3)
        self.assertEqual(sum(1 for action in actions if action["kind"] == "card_reward"), 0)
        self.assertEqual(env.reward_card_groups, [])
        expected_randoms = NativeRandomSet(seed=2)
        expected = [draw_random_potion(expected_randoms, player_class="IRONCLAD")["potion_id"] for _ in range(3)]
        self.assertEqual([p["potion_id"] for p in env.reward_potions], expected)

    def test_neow_one_random_rare_card_immediately_enters_deck(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=True)
        env.neow_options = [
            NeowOption(
                choice_index=0,
                bonus=NeowRewardType.ONE_RANDOM_RARE_CARD,
                drawback=NeowDrawback.NONE,
                bonus_text="Obtain a random rare card.",
            )
        ]

        original_size = len(env.deck)
        env.step(env.legal_actions()[0])

        self.assertEqual(env.phase, "NEOW")
        self.assertEqual(len(env.deck), original_size + 1)
        self.assertTrue(env.neow_pending_continue)

    def test_neow_drawback_and_boss_relic_apply_side_effects(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=True)
        env.neow_options = [
            NeowOption(
                choice_index=0,
                bonus=NeowRewardType.BOSS_RELIC,
                drawback=NeowDrawback.NO_GOLD,
                bonus_text="Lose your starter relic. Obtain a random boss relic.",
                drawback_text="Lose all gold.",
            )
        ]

        env.step(env.legal_actions()[0])

        self.assertEqual(env.gold, 0)
        self.assertEqual(env.phase, "NEOW")
        self.assertTrue(env.neow_pending_continue)
        relic_ids = [relic["relic_id"] for relic in env.relics]
        self.assertNotIn("Burning Blood", relic_ids)
        self.assertEqual(len(relic_ids), 1)

    def test_neow_curse_drawback_adds_random_curse_to_deck(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=True)
        env.neow_options = [
            NeowOption(
                choice_index=0,
                bonus=NeowRewardType.HUNDRED_GOLD,
                drawback=NeowDrawback.CURSE,
                bonus_text="Gain 100 gold.",
                drawback_text="Obtain a curse.",
            )
        ]

        original_size = len(env.deck)
        env.step(env.legal_actions()[0])

        self.assertEqual(len(env.deck), original_size + 1)
        self.assertEqual(env.deck[-1]["type"], "CURSE")
        self.assertEqual(env.gold, 199)

    def test_neows_blessing_sets_first_three_combats_to_one_hp(self):
        relic = make_relic("NeowsBlessing", counter=3)

        combat1 = NativeCombatEnv(
            seed=1,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[relic],
            master_deck=[],
        )
        self.assertTrue(all(monster.current_hp == 1 for monster in combat1.state.monsters))
        self.assertEqual(relic["counter"], 2)

        combat2 = NativeCombatEnv(
            seed=1,
            ascension_level=0,
            encounter_name="Jaw Worm",
            relics=[relic],
            master_deck=[],
        )
        self.assertTrue(all(monster.current_hp == 1 for monster in combat2.state.monsters))
        self.assertEqual(relic["counter"], 1)

        combat3 = NativeCombatEnv(
            seed=1,
            ascension_level=0,
            encounter_name="2 Louse",
            relics=[relic],
            master_deck=[],
        )
        self.assertTrue(all(monster.current_hp == 1 for monster in combat3.state.monsters))
        self.assertEqual(relic["counter"], -2)
        self.assertTrue(relic.get("used_up"))

        combat4 = NativeCombatEnv(
            seed=1,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[relic],
            master_deck=[],
        )
        self.assertGreater(combat4.state.monsters[0].current_hp, 1)

    def test_map_generation_exposes_real_node_actions(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        actions = env.legal_actions()

        self.assertTrue(actions)
        self.assertTrue(all(action["kind"] == "map" for action in actions))
        chosen = actions[0]
        env.step(chosen)
        self.assertEqual(env.floor, chosen["floor"])
        self.assertIn(env.phase, {"COMBAT", "EVENT", "CAMPFIRE", "SHOP", "TREASURE", "MAP"})
        self.assertIn(env.state().get("act_boss"), {"The Guardian", "Hexaghost", "Slime Boss"})

    def test_seed1_map_geometry_matches_decompiled_map_generator(self):
        env = NativeRunEnv(seed=1, ascension_level=0, enable_neow=False)
        by_row = {}
        for row in env.map:
            for node in row:
                if not node.has_edges():
                    continue
                by_row.setdefault(node.y, []).append(node.x)
        self.assertEqual(
            {row: sorted(xs) for row, xs in by_row.items()},
            {
                0: [1, 3],
                1: [0, 1, 4],
                2: [1, 3, 4],
                3: [1, 2, 3, 5],
                4: [0, 1, 2, 4, 6],
                5: [1, 2, 5],
                6: [0, 2, 3, 5, 6],
                7: [0, 1, 3, 4, 6],
                8: [0, 2, 3, 5],
                9: [1, 2, 3, 6],
                10: [2, 3, 5],
                11: [1, 2, 4],
                12: [1, 2, 3],
                13: [1, 2, 3],
                14: [2, 4],
            },
        )

    def test_seed1_room_symbols_match_java_room_type_assigner_oracle(self):
        env = NativeRunEnv(seed=1, ascension_level=0, enable_neow=False)
        by_row = {}
        for row in env.map:
            entries = []
            for node in row:
                if not node.has_edges():
                    continue
                entries.append(f"{node.x}={node.room_symbol}")
            by_row[row[0].y] = entries
        self.assertEqual(
            by_row,
            {
                0: ["1=M", "3=M"],
                1: ["0=M", "1=?", "4=M"],
                2: ["1=?", "3=M", "4=$"],
                3: ["1=M", "2=M", "3=?", "5=M"],
                4: ["0=M", "1=?", "2=M", "4=?", "6=M"],
                5: ["1=R", "2=R", "5=R"],
                6: ["0=M", "2=E", "3=M", "5=?", "6=M"],
                7: ["0=R", "1=$", "3=M", "4=?", "6=M"],
                8: ["0=T", "2=T", "3=T", "5=T"],
                9: ["1=M", "2=M", "3=?", "6=M"],
                10: ["2=M", "3=R", "5=?"],
                11: ["1=R", "2=M", "4=?"],
                12: ["1=E", "2=M", "3=?"],
                13: ["1=M", "2=M", "3=E"],
                14: ["2=R", "4=R"],
            },
        )

    def test_question_room_probabilities_start_from_event_helper_base_values(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        self.assertEqual(env.question_elite_chance, BASE_ELITE_CHANCE)
        self.assertEqual(env.question_monster_chance, BASE_MONSTER_CHANCE)
        self.assertEqual(env.question_shop_chance, BASE_SHOP_CHANCE)
        self.assertEqual(env.question_treasure_chance, BASE_TREASURE_CHANCE)

    def test_question_room_probabilities_reset_on_act_transition(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.question_elite_chance = 0.4
        env.question_monster_chance = 0.5
        env.question_shop_chance = 0.12
        env.question_treasure_chance = 0.08
        env._advance_to_next_act()
        self.assertEqual(env.question_elite_chance, RESET_ELITE_CHANCE)
        self.assertEqual(env.question_monster_chance, RESET_MONSTER_CHANCE)
        self.assertEqual(env.question_shop_chance, RESET_SHOP_CHANCE)
        self.assertEqual(env.question_treasure_chance, RESET_TREASURE_CHANCE)

    def test_tiny_chest_forces_treasure_on_fourth_question_room(self):
        randoms = NativeRandomSet(seed=1)
        relics = [make_relic("Tiny Chest", counter=3)]

        room_result, next_chances = roll_question_room_result(
            randoms,
            floor=10,
            current_room_type="MonsterRoom",
            relics=relics,
            elite_chance=BASE_ELITE_CHANCE,
            monster_chance=BASE_MONSTER_CHANCE,
            shop_chance=BASE_SHOP_CHANCE,
            treasure_chance=BASE_TREASURE_CHANCE,
        )

        self.assertEqual(room_result, "TREASURE")
        self.assertEqual(relics[0]["counter"], 0)
        self.assertEqual(next_chances["treasure"], RESET_TREASURE_CHANCE)

    def test_map_exposes_synthetic_boss_node_in_top_row(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        boss_parent = next(
            (
                node
                for row in env.map
                for node in row
                if any(edge.dst_y >= 15 for edge in node.edges)
            ),
            None,
        )
        self.assertIsNotNone(boss_parent)
        env.current_map_node = (boss_parent.x, boss_parent.y)
        env.first_room_chosen = True
        actions = env.legal_actions()
        self.assertTrue(any(action["symbol"] == "BOSS" for action in actions))

    def test_map_screen_exposes_source_like_first_room_anchor(self):
        env = NativeRunEnv(seed=50, ascension_level=0, enable_neow=False, start_on_map=True)
        state = env.state()
        self.assertEqual(state["screen_state"]["current_node"], {"x": 0, "y": -1})

        env.act = 3
        env.floor = env._act_floor_offset()
        env.current_map_node = None
        env.first_room_chosen = False
        state = env.state()
        self.assertEqual(state["screen_state"]["current_node"], {"x": -1, "y": 15})

    def test_elite_rooms_draw_from_elite_list_not_normal_list(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, final_act_available=True)
        env.elite_monster_list = ["Cultist"]
        elite_name = env.elite_monster_list[0]
        normal_name = env.monster_list[0]
        self.assertNotEqual(len(env.elite_monster_list), 0)
        emerald_node = next(node for row in env.map for node in row if node.has_emerald_key)
        env._step_map(
            {
                "node_id": f"a1-r{emerald_node.y}-x{emerald_node.x}",
                "symbol": "E_GREEN",
                "floor": emerald_node.y + 1,
            }
        )
        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual(env.combat.engine.encounter_name, elite_name)
        self.assertEqual(env.monster_list[0], normal_name)

    def test_emerald_elite_strength_buff_uses_source_formula(self):
        rules = emerald_elite_rules()
        with patch.object(CombatEngine, "_roll_emerald_elite_buff", return_value=0):
            combat = NativeCombatEnv(
                seed=220,
                act=1,
                ascension_level=0,
                encounter_name="Gremlin Nob",
                room_type="MonsterRoomElite",
                has_emerald_key=True,
            )

        self.assertEqual(_power_amount(combat.state.monsters[0], "Strength"), rules.strength_amount(1))

    def test_preserved_insect_applies_after_emerald_hp_buff(self):
        rules = emerald_elite_rules()
        baseline = NativeCombatEnv(
            seed=219,
            act=1,
            ascension_level=0,
            encounter_name="3 Sentries",
            room_type="MonsterRoomElite",
        )
        with patch.object(CombatEngine, "_roll_emerald_elite_buff", return_value=1):
            combat = NativeCombatEnv(
                seed=219,
                act=1,
                ascension_level=0,
                encounter_name="3 Sentries",
                room_type="MonsterRoomElite",
                relics=[make_relic("PreservedInsect")],
                has_emerald_key=True,
            )

        for buffed, control in zip(combat.state.monsters, baseline.state.monsters):
            expected_max = control.max_hp + int(float(control.max_hp) * rules.max_hp_bonus_ratio + 0.5)
            self.assertEqual(buffed.max_hp, expected_max)
            self.assertEqual(buffed.current_hp, max(1, int(expected_max * 0.75)))

    def test_preserved_insect_applies_to_event_elite_trigger(self):
        baseline = NativeCombatEnv(
            seed=219,
            act=1,
            ascension_level=0,
            encounter_name="3 Sentries",
            room_type="EventRoom",
        )
        combat = NativeCombatEnv(
            seed=219,
            act=1,
            ascension_level=0,
            encounter_name="3 Sentries",
            room_type="EventRoom",
            relics=[make_relic("PreservedInsect")],
            elite_trigger=True,
        )

        for reduced, control in zip(combat.state.monsters, baseline.state.monsters):
            self.assertEqual(reduced.max_hp, control.max_hp)
            self.assertEqual(reduced.current_hp, max(1, int(control.max_hp * 0.75)))

    def test_green_elite_victory_offers_and_claims_emerald_key_reward(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, final_act_available=True, has_emerald_key=False)
        emerald_node = next(node for row in env.map for node in row if node.has_emerald_key)
        env.current_map_node = (emerald_node.x, emerald_node.y)
        env.current_room_type = "MonsterRoomElite"

        class _VictoryCombat:
            def __init__(self, player, gold):
                self.player = player
                self.gold = gold
                self.encounter_name = "Cultist"
                self.engine = type("Engine", (), {"gold": gold, "bonus_reward_gold": 0})()

            def step(self, _action):
                return "VICTORY"

        env.combat = _VictoryCombat(env.player, env.gold)
        env._step_combat({"kind": "end"})

        key_action = next(action for action in env.legal_actions() if action["kind"] == "reward_key")
        self.assertEqual(key_action["key"], "emerald")
        env.step(key_action)

        self.assertTrue(env.has_emerald_key)
        self.assertFalse(env.reward_emerald_key)

    def test_v3_first_map_choice_enters_combat_with_real_actions(self):
        env = NativeRunEnv(seed=1, ascension_level=0, enable_neow=True)
        env.step(env.legal_actions()[0])
        env.step(env.legal_actions()[1])
        env.step(env.legal_actions()[-1])
        env.step(env.legal_actions()[0])
        self.assertEqual(env.phase, "COMBAT")
        actions = env.legal_actions()
        self.assertTrue(any(action["kind"] == "card" for action in actions))
        self.assertTrue(any(action["kind"] == "end" for action in actions))

    def test_v3_cultist_vertical_slice_reaches_reward_screen(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.step(env.legal_actions()[0])
        for _ in range(20):
            if env.phase != "COMBAT":
                break
            actions = env.legal_actions()
            attack_actions = [candidate for candidate in actions if candidate.get("kind") == "card" and candidate.get("card_id") in {"Strike_R", "Bash"}]
            if attack_actions:
                action = attack_actions[0]
            else:
                action = next(candidate for candidate in actions if candidate["kind"] == "end")
            env.step(action)
        self.assertEqual(env.phase, "CARD_REWARD")
        kinds = {action["kind"] for action in env.legal_actions()}
        self.assertIn("reward_gold", kinds)
        self.assertIn("raw", kinds)

    def test_combat_reward_card_pick_keeps_reward_screen_open_until_skip(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.step(env.legal_actions()[0])
        for _ in range(20):
            if env.phase != "COMBAT":
                break
            actions = env.legal_actions()
            attack_actions = [candidate for candidate in actions if candidate.get("kind") == "card" and candidate.get("card_id") in {"Strike_R", "Bash"}]
            env.step(attack_actions[0] if attack_actions else next(candidate for candidate in actions if candidate["kind"] == "end"))
        self.assertEqual(env.phase, "CARD_REWARD")
        while env.phase == "CARD_REWARD":
            reward_action = next(
                (
                    action
                    for action in env.legal_actions()
                    if action["kind"] in {"reward_gold", "reward_potion", "reward_relic"}
                ),
                None,
            )
            if reward_action is None:
                break
            env.step(reward_action)
        open_action = next(
            action
            for action in env.legal_actions()
            if action["kind"] == "raw" and action["label"] == "CARD"
        )
        env.step(open_action)
        card_action = next(action for action in env.legal_actions() if action["kind"] == "card_reward")
        chosen_card_id = card_action["card_id"]
        env.step(card_action)
        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual([action["kind"] for action in env.legal_actions()], ["proceed"])
        self.assertEqual([card["card_id"] for card in env.deck][-1], chosen_card_id)
        env.step(env.legal_actions()[0])
        self.assertEqual(env.phase, "MAP")

    def test_combat_reward_card_skip_returns_to_reward_overview_with_card_entry(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "CARD_REWARD"
        env.reward_gold = None
        env.reward_stolen_gold = None
        env.reward_relics = []
        env.reward_potions = [{"potion_id": "SpeedPotion", "name": "Speed Potion", "requires_target": False}]
        env.reward_cards = [make_card("Sentinel"), make_card("Infernal Blade"), make_card("Sever Soul")]
        env.reward_card_groups = []
        env.reward_card_screen_open = True

        env.step({"kind": "skip", "name": "SKIP"})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertFalse(env.reward_card_screen_open)
        self.assertTrue(env.reward_card_reward_declined)
        self.assertEqual([card["card_id"] for card in env.reward_cards], ["Sentinel", "Infernal Blade", "Sever Soul"])
        actions = env.legal_actions()
        self.assertIn("reward_potion", {action["kind"] for action in actions})
        self.assertTrue(any(action["kind"] == "raw" and action["label"] == "CARD" for action in actions))

    def test_reward_potion_full_belt_replaces_lowest_priority_potion(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "CARD_REWARD"
        env.reward_gold = None
        env.reward_relics = []
        env.reward_cards = []
        env.reward_potions = [make_potion("FairyPotion")]
        env.potions = [make_potion("Block Potion"), make_potion("Fire Potion"), make_potion("Weak Potion")]

        env.step({"kind": "reward_potion", "potion_id": "FairyPotion"})

        self.assertFalse(env.reward_potions)
        self.assertIn("FairyPotion", [potion["potion_id"] for potion in env.potions])
        self.assertEqual(len(env.potions), env.max_potion_slots)

    def test_reward_potion_full_belt_discards_low_priority_reward(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "CARD_REWARD"
        env.reward_gold = None
        env.reward_relics = []
        env.reward_cards = []
        env.reward_potions = [make_potion("Block Potion")]
        env.potions = [make_potion("FairyPotion"), make_potion("DuplicationPotion"), make_potion("PowerPotion")]
        before = [potion["potion_id"] for potion in env.potions]

        env.step({"kind": "reward_potion", "potion_id": "Block Potion"})

        self.assertFalse(env.reward_potions)
        self.assertEqual([potion["potion_id"] for potion in env.potions], before)

    def test_shop_potions_hidden_when_belt_full_or_sozu_owned(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "SHOP"
        env.current_shop = ShopState(potions=[make_potion("Fire Potion")])
        env.gold = 999
        env.potions = [make_potion("Block Potion"), make_potion("Weak Potion"), make_potion("SpeedPotion")]
        self.assertNotIn("potion", {action.get("item_kind") for action in env.legal_actions()})

        env.potions = []
        env.relics = [make_relic("Sozu")]
        self.assertNotIn("potion", {action.get("item_kind") for action in env.legal_actions()})

    def test_stale_event_reward_context_without_event_returns_to_map(self):
        env = NativeRunEnv(seed=56, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "CARD_REWARD"
        env.current_room_type = "CARD_REWARD"
        env.current_event = None
        env.event_reward_context = True
        env.reward_cards = [make_card("Sentinel")]
        env.reward_card_screen_open = True

        env.step({"kind": "card_reward", "card": dict(env.reward_cards[0])})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.current_room_type, "Map")
        self.assertIsNone(env.current_event)
        self.assertFalse(env.event_reward_context)

    def test_madness_with_no_positive_cost_targets_does_not_loop(self):
        env = NativeCombatEnv(seed=20, encounter_name="Cultist")
        env.state.hand = [make_card("Burn"), make_card("Burn"), make_card("Madness")]
        env.engine.state.hand = env.state.hand

        env.step({"kind": "card", "card_id": "Madness", "name": "Madness", "card_index": 2, "source_index": 2})

        self.assertEqual([card["card_id"] for card in env.state.hand], ["Burn", "Burn"])

    def test_question_card_increases_post_combat_reward_card_count(self):
        rewards = generate_monster_room_rewards(
            NativeRandomSet(seed=2),
            act=1,
            ascension_level=0,
            card_blizz_randomizer=5,
            blizzard_potion_mod=0,
            owned_relic_ids={"Burning Blood", "Question Card"},
            reward_count=1,
            player_class="IRONCLAD",
        )
        self.assertEqual(len(rewards["cards"]), 4)
        self.assertEqual(len(rewards["card_groups"][0]), 4)

    def test_busted_crown_reduces_post_combat_reward_card_count(self):
        rewards = generate_monster_room_rewards(
            NativeRandomSet(seed=2),
            act=1,
            ascension_level=0,
            card_blizz_randomizer=5,
            blizzard_potion_mod=0,
            owned_relic_ids={"Burning Blood", "Busted Crown"},
            reward_count=1,
            player_class="IRONCLAD",
        )
        self.assertEqual(len(rewards["cards"]), 1)
        self.assertEqual(len(rewards["card_groups"][0]), 1)

    def test_prayer_wheel_generates_second_reward_card_group(self):
        rewards = generate_monster_room_rewards(
            NativeRandomSet(seed=2),
            act=1,
            ascension_level=0,
            card_blizz_randomizer=5,
            blizzard_potion_mod=0,
            owned_relic_ids={"Burning Blood", "Prayer Wheel"},
            reward_count=1,
            player_class="IRONCLAD",
            prayer_wheel=True,
        )
        self.assertEqual(len(rewards["card_groups"]), 2)
        self.assertEqual(len(rewards["cards"]), 3)

    def test_reward_screen_exposes_all_pending_card_groups(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "CARD_REWARD"
        env.reward_gold = 15
        env.reward_relics = []
        env.reward_potions = []
        env.reward_cards = [make_card("Anger"), make_card("Cleave"), make_card("Warcry")]
        env.reward_card_groups = [[make_card("Flex"), make_card("Headbutt"), make_card("Thunderclap")]]
        env.reward_card_screen_open = False

        screen = env._reward_card_screen_state()
        self.assertEqual(
            [reward["reward_type"] for reward in screen["rewards"]],
            ["GOLD", "CARD", "CARD"],
        )
        raw_actions = [
            action
            for action in env.legal_actions()
            if action["kind"] == "raw" and action["label"] == "CARD"
        ]
        self.assertEqual([action["choice_index"] for action in raw_actions], [1, 2])
        self.assertEqual([action["reward_index"] for action in raw_actions], [0, 1])

    def test_reward_screen_can_open_later_card_group_without_losing_earlier_group(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "CARD_REWARD"
        env.reward_gold = None
        env.reward_relics = []
        env.reward_potions = []
        env.reward_cards = [make_card("Anger"), make_card("Cleave"), make_card("Warcry")]
        env.reward_card_groups = [[make_card("Flex"), make_card("Headbutt"), make_card("Thunderclap")]]
        env.reward_card_screen_open = False

        second_group_action = next(
            action
            for action in env.legal_actions()
            if action["kind"] == "raw" and action["reward_index"] == 1
        )
        env.step(second_group_action)

        self.assertTrue(env.reward_card_screen_open)
        self.assertEqual([card["card_id"] for card in env.reward_cards], ["Flex", "Headbutt", "Thunderclap"])
        self.assertEqual(
            [[card["card_id"] for card in group] for group in env.reward_card_groups],
            [["Anger", "Cleave", "Warcry"]],
        )

    def test_reward_card_skip_returns_to_overview_with_all_pending_card_groups(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "CARD_REWARD"
        env.reward_gold = None
        env.reward_relics = []
        env.reward_potions = []
        env.reward_cards = [make_card("Anger"), make_card("Cleave"), make_card("Warcry")]
        env.reward_card_groups = [[make_card("Flex"), make_card("Headbutt"), make_card("Thunderclap")]]
        env.reward_card_screen_open = False

        first_group_action = next(
            action
            for action in env.legal_actions()
            if action["kind"] == "raw" and action["reward_index"] == 0
        )
        env.step(first_group_action)
        env.step({"kind": "skip", "name": "SKIP", "choice_index": 0})

        self.assertFalse(env.reward_card_screen_open)
        self.assertTrue(env.reward_card_reward_declined)
        self.assertEqual([card["card_id"] for card in env.reward_cards], ["Anger", "Cleave", "Warcry"])
        self.assertEqual(
            [[card["card_id"] for card in group] for group in env.reward_card_groups],
            [["Flex", "Headbutt", "Thunderclap"]],
        )
        raw_actions = [
            action
            for action in env.legal_actions()
            if action["kind"] == "raw" and action["label"] == "CARD"
        ]
        self.assertEqual([action["reward_index"] for action in raw_actions], [0, 1])

    def test_reward_screen_promotes_second_prayer_wheel_card_group_after_pick(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.phase = "CARD_REWARD"
        env.reward_gold = None
        env.reward_relics = []
        env.reward_potions = []
        env.reward_cards = [make_card("Anger"), make_card("Cleave"), make_card("Warcry")]
        env.reward_card_groups = [[make_card("Flex"), make_card("Headbutt"), make_card("Thunderclap")]]
        env.reward_card_screen_open = True

        first_pick = next(action for action in env.legal_actions() if action["kind"] == "card_reward")
        env.step(first_pick)

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(len(env.reward_cards), 3)
        self.assertEqual([card["card_id"] for card in env.reward_cards], ["Flex", "Headbutt", "Thunderclap"])
        self.assertEqual(env.reward_card_groups, [])

    def test_escaped_looter_suppresses_normal_monster_room_gold_reward(self):
        env = NativeRunEnv(seed=107, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "COMBAT"
        env.current_room_type = "MonsterRoom"
        env.gold = 0
        escaped_looter = MonsterState(
            monster_id="Looter",
            name="Looter",
            current_hp=8,
            max_hp=48,
            block=0,
            next_move="ESCAPE",
            intent="ESCAPE",
            move_adjusted_damage=-1,
            move_hits=1,
            meta={"escaped": True},
        )
        fake_engine = SimpleNamespace(
            gold=0,
            potions=list(env.potions),
            bonus_reward_gold=0,
            state=SimpleNamespace(monsters=[escaped_looter]),
        )
        env.combat = SimpleNamespace(
            step=lambda _action: "VICTORY",
            player=env.player,
            master_deck=list(env.deck),
            engine=fake_engine,
            encounter_name="Looter",
            room_type="MonsterRoom",
        )

        env._step_combat({"kind": "end", "name": "END_TURN"})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertIsNone(env.reward_gold)
        self.assertTrue(env.reward_cards)
        self.assertEqual(env.randoms.stream("treasure").counter, 0)

    def test_singing_bowl_action_grants_max_hp_and_consumes_current_group(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.relics.append(make_relic("Singing Bowl"))
        env.phase = "CARD_REWARD"
        env.reward_gold = None
        env.reward_relics = []
        env.reward_potions = []
        env.reward_cards = [make_card("Anger"), make_card("Cleave"), make_card("Warcry")]
        env.reward_card_groups = []
        env.reward_card_screen_open = True
        start_hp = env.player.current_hp
        start_max_hp = env.player.max_hp

        bowl_action = next(action for action in env.legal_actions() if action["kind"] == "singing_bowl")
        env.step(bowl_action)

        self.assertEqual(env.player.max_hp, start_max_hp + 2)
        self.assertEqual(env.player.current_hp, start_hp + 2)
        self.assertEqual(env.phase, "MAP")

    def test_singing_bowl_max_hp_heal_obeys_mark_of_the_bloom(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.relics = [make_relic("Singing Bowl"), make_relic("Mark of the Bloom")]
        env.phase = "CARD_REWARD"
        env.reward_gold = None
        env.reward_relics = []
        env.reward_potions = []
        env.reward_cards = [make_card("Anger"), make_card("Cleave"), make_card("Warcry")]
        env.reward_card_groups = []
        env.reward_card_screen_open = True
        env.player.current_hp = 40
        env.player.max_hp = 80

        bowl_action = next(action for action in env.legal_actions() if action["kind"] == "singing_bowl")
        env.step(bowl_action)

        self.assertEqual(env.player.max_hp, 82)
        self.assertEqual(env.player.current_hp, 40)
        self.assertEqual(env.phase, "MAP")

    def test_reward_preview_eggs_upgrade_matching_card_types(self):
        cards = [
            make_card("Anger"),
            make_card("Shrug It Off"),
            make_card("Inflame"),
            make_card("Writhe"),
        ]

        upgraded = apply_reward_preview_relics(
            cards,
            owned_relic_ids={"Molten Egg 2", "Toxic Egg 2", "Frozen Egg 2"},
        )

        by_id = {card["card_id"]: card for card in upgraded}
        self.assertEqual(by_id["Anger"]["upgrades"], 1)
        self.assertEqual(by_id["Shrug It Off"]["upgrades"], 1)
        self.assertEqual(by_id["Inflame"]["upgrades"], 1)
        self.assertEqual(by_id["Writhe"]["upgrades"], 0)

    def test_nloths_gift_triples_rare_reward_chance(self):
        baseline, _ = generate_card_reward_with_state(
            NativeRandomSet(seed=13),
            count=1,
            card_blizz_randomizer=5,
            owned_relic_ids={"Burning Blood"},
        )
        boosted, _ = generate_card_reward_with_state(
            NativeRandomSet(seed=13),
            count=1,
            card_blizz_randomizer=5,
            owned_relic_ids={"Burning Blood", "Nloth's Gift"},
        )

        self.assertEqual(baseline[0]["card_id"], "Blood for Blood")
        self.assertEqual(baseline[0]["rarity"], "UNCOMMON")
        self.assertEqual(boosted[0]["card_id"], "Immolate")
        self.assertEqual(boosted[0]["rarity"], "RARE")

    def test_generate_card_reward_matches_stateful_reward_helper(self):
        legacy = generate_card_reward(NativeRandomSet(seed=41), count=3)
        stateful, _ = generate_card_reward_with_state(
            NativeRandomSet(seed=41),
            count=3,
            card_blizz_randomizer=5,
        )

        self.assertEqual(
            [card["card_id"] for card in legacy],
            [card["card_id"] for card in stateful],
        )

    def test_generate_card_reward_uses_class_runtime_pools(self):
        runtime_card_pools = initialize_runtime_card_pools("THE_SILENT")
        runtime_card_pools["CLASS_COMMON"] = ["Backflip"]
        runtime_card_pools["CLASS_UNCOMMON"] = []
        runtime_card_pools["CLASS_RARE"] = []
        runtime_card_pools["RED_COMMON"] = ["Anger"]
        runtime_card_pools["RED_UNCOMMON"] = ["Shrug It Off"]
        runtime_card_pools["RED_RARE"] = ["Barricade"]

        reward, _ = generate_card_reward_with_state(
            NativeRandomSet(seed=41),
            count=1,
            card_blizz_randomizer=5,
            runtime_card_pools=runtime_card_pools,
        )

        self.assertEqual(reward[0]["card_id"], "Backflip")

    def test_generate_card_reward_retries_duplicate_in_same_rarity_bucket(self):
        randoms = NativeRandomSet(seed=1984457689828272421)
        blizz = 5
        rewards = []
        for _ in range(5):
            cards, blizz = generate_card_reward_with_state(
                randoms,
                count=3,
                card_blizz_randomizer=blizz,
                card_upgraded_chance=0.0,
            )
            rewards.append([card["card_id"] for card in cards])

        self.assertEqual(rewards[0], ["True Grit", "Flex", "Dropkick"])
        self.assertEqual(rewards[1], ["Heavy Blade", "Rampage", "Clothesline"])
        self.assertEqual(rewards[2], ["Heavy Blade", "Disarm", "Shrug It Off"])
        self.assertEqual(rewards[3], ["Sword Boomerang", "Pummel", "Fire Breathing"])
        self.assertEqual(rewards[4], ["Intimidate", "Battle Trance", "Offering"])
        self.assertEqual(blizz, 5)

    def test_real_game_potion_rng_seed1_second_combat_rolls_liquid_bronze(self):
        randoms = NativeRandomSet(seed=1)
        first_reward_roll = int(randoms.stream("potion").random(0, 99))
        self.assertEqual(first_reward_roll, 55)
        second_reward_roll = int(randoms.stream("potion").random(0, 99))
        self.assertEqual(second_reward_roll, 5)
        potion = roll_random_potion(randoms, player_class="IRONCLAD")
        self.assertEqual(potion["potion_id"], "LiquidBronze")
        self.assertEqual(randoms.stream("potion").counter, 7)

    def test_seed1_second_reward_screen_uses_liquid_bronze(self):
        import json

        env = NativeRunEnv(seed=1, ascension_level=0, enable_neow=True)
        trace_path = Path("/home/yydd/spirecomm/_cache/real_game_first/traces_mid/seed_1_trace.json")
        trace = json.loads(trace_path.read_text())
        for step in trace["steps"][:30]:
            env.step(step["action"])
        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertIsNone(env.reward_gold)
        self.assertEqual([p["potion_id"] for p in env.reward_potions], ["LiquidBronze"])

    def test_seed20_first_reward_cards_match_real_replay(self):
        import json

        env = NativeRunEnv(seed=20, ascension_level=0, enable_neow=True)
        trace_path = Path("/home/yydd/spirecomm/_cache/real_game_first/traces/seed_20_trace_50.json")
        trace = json.loads(trace_path.read_text())
        for step in trace["steps"][:12]:
            env.step(step["action"])
        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(env.randoms.stream("card").counter, 9)
        self.assertEqual(env.card_blizz_randomizer, 3)
        self.assertEqual([card["card_id"] for card in env.reward_cards], ["Sword Boomerang", "Entrench", "Perfected Strike"])

    def test_monster_room_reward_helper_tracks_blizzard_potion_mod(self):
        randoms = NativeRandomSet(seed=1)
        first_rewards = generate_monster_room_rewards(
            randoms,
            act=1,
            ascension_level=0,
            card_blizz_randomizer=5,
            blizzard_potion_mod=0,
            owned_relic_ids={"Burning Blood"},
            reward_count=1,
            player_class="IRONCLAD",
        )
        self.assertEqual(first_rewards["gold"], 12)
        self.assertIsNone(first_rewards["potion"])
        self.assertEqual(first_rewards["blizzard_potion_mod"], 10)
        rewards = generate_monster_room_rewards(
            randoms,
            act=1,
            ascension_level=0,
            card_blizz_randomizer=int(first_rewards["card_blizz_randomizer"]),
            blizzard_potion_mod=int(first_rewards["blizzard_potion_mod"]),
            owned_relic_ids={"Burning Blood"},
            reward_count=1,
            player_class="IRONCLAD",
        )
        self.assertEqual(rewards["gold"], 16)
        self.assertEqual(rewards["potion"]["potion_id"], "LiquidBronze")
        self.assertEqual(rewards["blizzard_potion_mod"], 0)
        self.assertEqual(len(rewards["cards"]), 3)

    def test_monster_room_reward_helper_uses_potion_rules_without_capacity_check(self):
        randoms = NativeRandomSet(seed=12)
        randoms.stream("potion").set_counter(23)
        randoms.stream("potion").calls.clear()

        rewards = generate_monster_room_rewards(
            randoms,
            act=2,
            room_type="MonsterRoom",
            ascension_level=0,
            card_blizz_randomizer=2,
            blizzard_potion_mod=10,
            owned_relic_ids={"Burning Blood"},
            reward_count=1,
            player_class="IRONCLAD",
        )

        self.assertIsNone(rewards["potion"])
        self.assertEqual(rewards["blizzard_potion_mod"], 20)
        self.assertEqual(
            [(call.method, call.args, call.result) for call in randoms.stream("potion").calls],
            [("random", (0, 99), 60)],
        )

    def test_room_reward_rules_match_decompiled_source(self):
        rules = room_reward_rules()

        self.assertEqual(rules.normal_rare_chance, 3)
        self.assertEqual(rules.normal_uncommon_chance, 37)
        self.assertEqual(rules.elite_rare_chance, 10)
        self.assertEqual(rules.elite_uncommon_chance, 40)
        self.assertEqual(rules.normal_relic_common_cutoff, 50)
        self.assertEqual(rules.normal_relic_rare_gt, 85)
        self.assertEqual(rules.elite_relic_common_cutoff, 50)
        self.assertEqual(rules.elite_relic_rare_gt, 82)
        self.assertEqual(rules.normal_gold_min, 10)
        self.assertEqual(rules.normal_gold_max, 20)
        self.assertEqual(rules.elite_gold_min, 25)
        self.assertEqual(rules.elite_gold_max, 35)

    def test_monster_room_elite_rewards_use_room_specific_gold_and_card_rarity_rules(self):
        rules = room_reward_rules()
        seed = 17
        elite_gold_expected = int(NativeRandomSet(seed=seed).stream("treasure").random(rules.elite_gold_min, rules.elite_gold_max))
        expected_groups, expected_blizz = generate_card_reward_groups_with_state(
            NativeRandomSet(seed=seed),
            group_count=1,
            card_blizz_randomizer=5,
            card_upgraded_chance=act_chances(1).card_upgraded_chance(0),
            rare_chance=rules.elite_rare_chance,
            uncommon_chance=rules.elite_uncommon_chance,
            owned_relic_ids={"Burning Blood"},
        )

        rewards = generate_monster_room_rewards(
            NativeRandomSet(seed=seed),
            act=1,
            room_type="MonsterRoomElite",
            ascension_level=0,
            card_blizz_randomizer=5,
            blizzard_potion_mod=0,
            owned_relic_ids={"Burning Blood"},
            reward_count=1,
            player_class="IRONCLAD",
        )

        self.assertEqual(rewards["gold"], elite_gold_expected)
        self.assertEqual(
            [card["card_id"] for card in rewards["cards"]],
            [card["card_id"] for card in expected_groups[0]],
        )
        self.assertEqual(rewards["card_blizz_randomizer"], expected_blizz)

    def test_elite_relic_rewards_use_elite_room_tier_thresholds(self):
        rules = room_reward_rules()
        seed = 17
        roll = int(NativeRandomSet(seed=seed).stream("relic").random(99))
        if roll < rules.elite_relic_common_cutoff:
            expected_tier = "COMMON"
        elif roll > rules.elite_relic_rare_gt:
            expected_tier = "RARE"
        else:
            expected_tier = "UNCOMMON"

        rewards = generate_elite_relic_rewards(
            NativeRandomSet(seed=seed),
            owned_relic_ids={"Burning Blood"},
            relic_drawer=lambda tier: {"relic_id": tier},
        )

        self.assertEqual(rewards[0]["relic_id"], expected_tier)

    def test_white_beast_statue_forces_potion_when_reward_slots_available(self):
        rewards = generate_monster_room_rewards(
            NativeRandomSet(seed=9),
            act=1,
            ascension_level=0,
            card_blizz_randomizer=5,
            blizzard_potion_mod=0,
            owned_relic_ids={"Burning Blood", "White Beast Statue"},
            reward_count=1,
            player_class="IRONCLAD",
        )

        self.assertIsNotNone(rewards["potion"])
        self.assertEqual(rewards["blizzard_potion_mod"], -10)

    def test_white_beast_statue_does_not_override_full_reward_cap(self):
        rewards = generate_monster_room_rewards(
            NativeRandomSet(seed=9),
            act=1,
            ascension_level=0,
            card_blizz_randomizer=5,
            blizzard_potion_mod=0,
            owned_relic_ids={"Burning Blood", "White Beast Statue"},
            reward_count=4,
            player_class="IRONCLAD",
        )

        self.assertIsNone(rewards["potion"])
        self.assertEqual(rewards["blizzard_potion_mod"], 10)

    def test_elite_rewards_add_relic_and_black_star_adds_second(self):
        rewards = generate_elite_relic_rewards(
            NativeRandomSet(seed=2),
            owned_relic_ids={"Burning Blood"},
            black_star=False,
        )
        black_star_rewards = generate_elite_relic_rewards(
            NativeRandomSet(seed=2),
            owned_relic_ids={"Burning Blood", "Black Star"},
            black_star=True,
        )

        self.assertEqual(len(rewards), 1)
        self.assertEqual(len(black_star_rewards), 2)
        self.assertEqual(len({relic["relic_id"] for relic in black_star_rewards}), 2)

    def test_sozu_consumes_reward_potion_without_adding_to_belt(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.relics.append(make_relic("Sozu"))
        env.phase = "CARD_REWARD"
        env.reward_cards = []
        env.reward_gold = None
        env.reward_relics = []
        env.reward_potions = [roll_random_potion(NativeRandomSet(seed=3), player_class="IRONCLAD")]

        potion_action = next(action for action in env.legal_actions() if action["kind"] == "reward_potion")
        env.step(potion_action)

        self.assertEqual(env.potions, [])
        self.assertEqual(env.reward_potions, [])

    def test_full_potion_belt_keeps_reward_potion_unclaimed(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.phase = "CARD_REWARD"
        env.reward_cards = []
        env.reward_gold = None
        env.reward_relics = []
        env.potions = [
            roll_random_potion(NativeRandomSet(seed=11), player_class="IRONCLAD"),
            roll_random_potion(NativeRandomSet(seed=12), player_class="IRONCLAD"),
            roll_random_potion(NativeRandomSet(seed=13), player_class="IRONCLAD"),
        ]
        env.reward_potions = [roll_random_potion(NativeRandomSet(seed=14), player_class="IRONCLAD")]

        potion_action = next(action for action in env.legal_actions() if action["kind"] == "reward_potion")
        reward_potion_id = env.reward_potions[0]["potion_id"]
        env.step(potion_action)

        self.assertEqual(len(env.potions), 3)
        self.assertEqual([p["potion_id"] for p in env.reward_potions], [reward_potion_id])

    def test_full_potion_belt_keeps_generated_combat_potion_reward_visible(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.phase = "COMBAT"
        env.current_room_type = "MonsterRoom"
        env.potions = [
            roll_random_potion(NativeRandomSet(seed=11), player_class="IRONCLAD"),
            roll_random_potion(NativeRandomSet(seed=12), player_class="IRONCLAD"),
            roll_random_potion(NativeRandomSet(seed=13), player_class="IRONCLAD"),
        ]

        class _VictoryCombat:
            encounter_name = "Cultist"
            engine = None

            def __init__(self, player, deck, gold):
                self.player = player
                self.master_deck = deck
                self.gold = gold

            def step(self, action):
                return "VICTORY"

        env.combat = _VictoryCombat(env.player, env.deck, env.gold)
        env._step_combat({"kind": "end", "name": "END_TURN"})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(len(env.reward_potions), 1)
        self.assertEqual(env.blizzard_potion_mod, -10)

    def test_golden_idol_event_flow_matches_real_screen_structure(self):
        event = EventState("Golden Idol")
        intro = event.actions(ascension_level=0, max_hp=80)
        self.assertEqual([action["name"] for action in intro], ["Take Golden Idol", "Ignore"])

        take_idol = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=65,
            max_hp=80,
            gold=113,
        )
        self.assertEqual(event.screen, "BOULDER")
        self.assertEqual(take_idol["add_relics"][0]["relic_id"], "Golden Idol")

        boulder = event.actions(ascension_level=0, max_hp=80)
        self.assertEqual(len(boulder), 3)
        self.assertEqual(boulder[0]["name"], "Take Injury")
        self.assertIn("Damage", boulder[1]["name"])
        self.assertIn("Max HP", boulder[2]["name"])

        wound = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=65,
            max_hp=80,
            gold=113,
        )
        self.assertEqual(event.screen, "RESULT")
        self.assertEqual(wound["add_cards"][0]["card_id"], "Injury")

        leave = event.actions(ascension_level=0, max_hp=80)
        self.assertEqual(leave[0]["name"], "Leave")

    def test_accursed_blacksmith_rummage_grants_pain_and_warped_tongs(self):
        event = EventState("Accursed Blacksmith")
        intro = event.actions(ascension_level=0, max_hp=80, deck=[make_card("Strike_R")])
        self.assertEqual([action["name"] for action in intro], ["Forge", "Rummage", "Leave"])

        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[make_card("Strike_R")],
        )
        self.assertEqual(event.screen, "RESULT")
        self.assertEqual(result["add_cards"][0]["card_id"], "Pain")
        self.assertEqual(result["add_relics"][0]["relic_id"], "WarpedTongs")
        self.assertEqual(result["add_relics"][0]["name"], "Warped Tongs")

    def test_purifier_opens_purge_card_select_and_leaves_after_completion(self):
        event = EventState("Purifier")
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[make_card("Strike_R"), make_card("Defend_R")],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertTrue(result["open_card_select"])
        self.assertEqual(result["card_select_mode"], "purge")

    def test_upgrade_shrine_opens_upgrade_card_select(self):
        event = EventState("Upgrade Shrine")
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[make_card("Strike_R"), make_card("Defend_R")],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertTrue(result["open_card_select"])
        self.assertEqual(result["card_select_mode"], "upgrade")

    def test_transmorgrifier_opens_transform_card_select(self):
        event = EventState("Transmorgrifier")
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[make_card("Strike_R"), make_card("Defend_R")],
        )
        self.assertEqual(event.screen, "COMPLETE")
        self.assertTrue(result["open_card_select"])
        self.assertEqual(result["card_select_mode"], "transform")

    def test_transmorgrifier_still_excludes_bottled_transform_candidates(self):
        bottled_strike = make_card("Strike_R", uuid="bottled-strike")
        bottled_strike["in_bottle_flame"] = True
        event = EventState("Transmorgrifier")

        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[bottled_strike, make_card("Defend_R", uuid="defend")],
        )

        self.assertEqual(result["candidate_indexes"], [1])

    def test_big_fish_box_grants_regret_and_screenless_relic(self):
        event = EventState("Big Fish")
        intro = event.actions(ascension_level=0, max_hp=80)
        self.assertEqual([action["choice_index"] for action in intro], [0, 1, 2])

        result = resolve_event_choice(
            event,
            action_index=2,
            randoms=NativeRandomSet(seed=7),
            ascension_level=0,
            current_hp=64,
            max_hp=80,
            gold=99,
            deck=[],
        )
        self.assertEqual(event.screen, "RESULT")
        self.assertEqual(result["add_cards"][0]["card_id"], "Regret")
        self.assertIn(result["add_relics"][0]["tier"], {"COMMON", "UNCOMMON", "RARE"})

    def test_big_fish_banana_uses_mark_of_the_bloom_heal_block(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("Big Fish")
        env.player.current_hp = 40
        env.player.max_hp = 90
        env.relics = [make_relic("Mark of the Bloom")]

        env.step({"kind": "event", "event_id": "Big Fish", "choice_index": 0})

        self.assertEqual(env.player.current_hp, 40)
        self.assertEqual(env.player.max_hp, 90)

    def test_big_fish_banana_does_not_apply_magic_flower_outside_combat(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("Big Fish")
        env.player.current_hp = 40
        env.player.max_hp = 90
        env.relics = [make_relic("Magic Flower")]

        env.step({"kind": "event", "event_id": "Big Fish", "choice_index": 0})

        self.assertEqual(env.player.current_hp, 70)

    def test_big_fish_donut_max_hp_heal_obeys_mark_of_the_bloom(self):
        for relic_ids, expected_hp in [([], 45), (["Mark of the Bloom"], 40)]:
            with self.subTest(relic_ids=relic_ids):
                env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
                env.phase = "EVENT"
                env.current_room_type = "EventRoom"
                env.current_event = EventState("Big Fish")
                env.player.current_hp = 40
                env.player.max_hp = 90
                env.relics = [make_relic(relic_id) for relic_id in relic_ids]

                env.step({"kind": "event", "event_id": "Big Fish", "choice_index": 1})

                self.assertEqual(env.player.max_hp, 95)
                self.assertEqual(env.player.current_hp, expected_hp)

    def test_cleric_event_can_open_purge_and_return_to_event_result(self):
        env = NativeRunEnv(seed=11, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("The Cleric")
        env.gold = 99

        actions = env.legal_actions()
        self.assertIn("Purify", {action["name"] for action in actions})
        env.step(next(action for action in actions if action["name"] == "Purify"))
        self.assertEqual(env.phase, "CARD_SELECT")

        env.step(env.legal_actions()[0])
        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertTrue(env.state()["screen_state"]["confirm_up"])

        env.step(env.legal_actions()[0])
        self.assertEqual(env.phase, "EVENT")
        self.assertIsNotNone(env.current_event)
        self.assertEqual(env.current_event.screen, "RESULT")
        self.assertEqual(env.gold, 49)

        env.step(env.legal_actions()[0])
        self.assertEqual(env.phase, "MAP")

    def test_cleric_screen_state_preserves_locked_purify_slot(self):
        env = NativeRunEnv(seed=11, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("The Cleric")
        env.gold = 40
        env.player.max_hp = 80

        options = env._event_screen_state()["options"]
        actions = env.legal_actions()

        self.assertEqual([option["label"] for option in options], ["Heal", "Locked", "Leave"])
        self.assertTrue(options[1]["disabled"])
        self.assertNotIn("choice_index", options[1])
        self.assertEqual([option.get("choice_index", index) for index, option in enumerate(options)], [0, 1, 1])
        self.assertEqual([action["choice_index"] for action in actions], [0, 2])

    def test_cleric_heal_uses_mark_of_the_bloom_heal_block(self):
        env = NativeRunEnv(seed=11, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("The Cleric")
        env.gold = 99
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.relics = [make_relic("Mark of the Bloom")]

        env.step(next(action for action in env.legal_actions() if action["choice_index"] == 0))

        self.assertEqual(env.gold, 64)
        self.assertEqual(env.player.current_hp, 40)

    def test_cleric_heal_does_not_apply_magic_flower_outside_combat(self):
        env = NativeRunEnv(seed=11, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("The Cleric")
        env.gold = 99
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.relics = [make_relic("Magic Flower")]

        env.step(next(action for action in env.legal_actions() if action["choice_index"] == 0))

        self.assertEqual(env.gold, 64)
        self.assertEqual(env.player.current_hp, 60)

    def test_living_wall_grow_uses_event_card_select_return_path(self):
        env = NativeRunEnv(seed=13, ascension_level=0, enable_neow=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("Living Wall")

        actions = env.legal_actions()
        self.assertIn("Grow", {action["name"] for action in actions})
        strike_index = next(index for index, card in enumerate(env.deck) if card["card_id"] == "Strike_R")
        env.step(next(action for action in actions if action["name"] == "Grow"))
        self.assertEqual(env.phase, "CARD_SELECT")

        grow_action = next(action for action in env.legal_actions() if action["target_index"] == strike_index)
        env.step(grow_action)
        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertTrue(env.state()["screen_state"]["confirm_up"])

        env.step(env.legal_actions()[0])
        self.assertEqual(env.phase, "EVENT")
        self.assertEqual(env.current_event.screen, "RESULT")
        self.assertEqual(env.deck[strike_index]["upgrades"], 1)

    def test_event_upgrade_actions_expose_upgraded_searing_blow(self):
        deck = [make_card("Searing Blow", upgrades=1, uuid="searing")]

        self.assertIn(
            0,
            [action["choice_index"] for action in EventState("Shining Light").actions(ascension_level=0, max_hp=80, deck=deck)],
        )
        self.assertIn(
            2,
            [action["choice_index"] for action in EventState("Living Wall").actions(ascension_level=0, max_hp=80, deck=deck)],
        )
        self.assertIn(
            0,
            [action["choice_index"] for action in EventState("Accursed Blacksmith").actions(ascension_level=0, max_hp=80, deck=deck)],
        )
        self.assertIn(
            0,
            [action["choice_index"] for action in EventState("Upgrade Shrine").actions(ascension_level=0, max_hp=80, deck=deck)],
        )

        designer = EventState(
            "Designer",
            screen="MAIN",
            data={"adjustment_upgrades_one": True, "clean_up_removes_cards": True, "adjust_cost": 40, "clean_cost": 60, "full_cost": 90, "hp_loss": 3},
        )
        self.assertIn(0, [action["choice_index"] for action in designer.actions(ascension_level=0, max_hp=80, gold=120, deck=deck)])

    def test_woman_in_blue_can_appear_even_when_gold_is_low(self):
        event = EventState("The Woman in Blue")
        actions = event.actions(ascension_level=0, max_hp=80, gold=0, deck=[])
        self.assertEqual([action["choice_index"] for action in actions], [0, 1, 2, 3])
        self.assertEqual(actions[-1]["name"], "Ignored")

    def test_woman_in_blue_low_gold_purchase_clamps_gold_and_keeps_potion_rng(self):
        event = EventState("The Woman in Blue")
        randoms = NativeRandomSet(seed=7)
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=randoms,
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=10,
            deck=[],
            potions=[],
        )
        expected_randoms = NativeRandomSet(seed=7)
        expected = [draw_random_potion(expected_randoms, player_class="IRONCLAD")["potion_id"] for _ in range(2)]

        self.assertEqual(result["gold"], 0)
        self.assertTrue(result["open_rewards"])
        self.assertEqual([potion["potion_id"] for potion in result["potions"]], expected)

    def test_woman_in_blue_a15_punch_hp_loss_remains_unchanged(self):
        event = EventState("The Woman in Blue")
        actions = event.actions(ascension_level=15, max_hp=80, gold=0, deck=[])
        self.assertEqual([action["choice_index"] for action in actions], [0, 1, 2, 3])
        self.assertEqual(actions[-1]["name"], "Punch (4 HP)")

        result = resolve_event_choice(
            event,
            action_index=3,
            randoms=NativeRandomSet(seed=7),
            ascension_level=15,
            current_hp=80,
            max_hp=80,
            gold=0,
            deck=[],
            potions=[],
        )

        self.assertEqual(result["hp"], 76)
        self.assertEqual(result["gold"], 0)

    def test_designer_main_excludes_paid_options_when_gold_is_insufficient(self):
        event = EventState(
            "Designer",
            screen="MAIN",
            data={"adjustment_upgrades_one": True, "clean_up_removes_cards": True, "adjust_cost": 40, "clean_cost": 60, "full_cost": 90, "hp_loss": 3},
        )

        actions = event.actions(
            ascension_level=0,
            max_hp=80,
            gold=0,
            deck=[make_card("Strike_R", uuid="strike"), make_card("Defend_R", uuid="defend")],
        )

        self.assertEqual([action["choice_index"] for action in actions], [3])
        self.assertEqual(actions[0]["name"], "Punch (3 HP)")

    def test_designer_main_excludes_adjust_when_no_upgradable_cards_exist(self):
        event = EventState(
            "Designer",
            screen="MAIN",
            data={"adjustment_upgrades_one": True, "clean_up_removes_cards": True, "adjust_cost": 40, "clean_cost": 60, "full_cost": 90, "hp_loss": 3},
        )

        actions = event.actions(
            ascension_level=0,
            max_hp=80,
            gold=120,
            deck=[make_card("Strike_R", upgrades=1, uuid="strike"), make_card("Defend_R", upgrades=1, uuid="defend")],
        )

        self.assertEqual([action["choice_index"] for action in actions], [1, 2, 3])

    def test_designer_main_respects_non_bottled_card_count_gates(self):
        bottled_strike = make_card("Strike_R", uuid="bottled-strike")
        bottled_strike["in_bottle_flame"] = True
        remove_event = EventState(
            "Designer",
            screen="MAIN",
            data={"adjustment_upgrades_one": True, "clean_up_removes_cards": True, "adjust_cost": 40, "clean_cost": 60, "full_cost": 90, "hp_loss": 3},
        )

        bottled_only_actions = remove_event.actions(
            ascension_level=0,
            max_hp=80,
            gold=120,
            deck=[bottled_strike],
        )

        self.assertEqual([action["choice_index"] for action in bottled_only_actions], [0, 3])

        transform_event = EventState(
            "Designer",
            screen="MAIN",
            data={"adjustment_upgrades_one": True, "clean_up_removes_cards": False, "adjust_cost": 40, "clean_cost": 60, "full_cost": 90, "hp_loss": 3},
        )
        one_card_actions = transform_event.actions(
            ascension_level=0,
            max_hp=80,
            gold=120,
            deck=[make_card("Strike_R", uuid="strike")],
        )

        self.assertEqual([action["choice_index"] for action in one_card_actions], [0, 2, 3])

    def test_woman_in_blue_uses_potion_helper_rolls(self):
        event = EventState("The Woman in Blue")
        randoms = NativeRandomSet(seed=7)
        result = resolve_event_choice(
            event,
            action_index=1,
            randoms=randoms,
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            potions=[],
        )
        self.assertTrue(result["open_rewards"])
        expected_randoms = NativeRandomSet(seed=7)
        expected = [draw_random_potion(expected_randoms, player_class="IRONCLAD")["potion_id"] for _ in range(2)]
        self.assertEqual([p["potion_id"] for p in result["potions"]], expected)

    def test_knowing_skull_potion_option_uses_uniform_potion_draw(self):
        event = EventState("Knowing Skull", screen="ASK", data={"potion_cost": 6, "gold_cost": 6, "card_cost": 6, "leave_cost": 6})
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=9),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        expected_randoms = NativeRandomSet(seed=9)
        expected = draw_random_potion(expected_randoms, player_class="IRONCLAD")["potion_id"]
        self.assertFalse(result["open_rewards"])
        self.assertEqual([p["potion_id"] for p in result["add_potions"]], [expected])

    def test_jaw_worm_vertical_slice_has_real_opening_intent(self):
        randoms = NativeRandomSet(seed=2, floor=2)
        engine = CombatEngine(
            encounter_name="Jaw Worm",
            randoms=randoms,
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        self.assertEqual(monster.monster_id, "JawWorm")
        self.assertEqual(monster.next_move, "CHOMP")
        self.assertEqual(monster.intent, "ATTACK")
        self.assertGreater(monster.move_adjusted_damage, 0)
        self.assertEqual(randoms.stream("ai").counter, 1)

    def test_jaw_worm_re_rolls_after_opening_chomp(self):
        randoms = NativeRandomSet(seed=1, floor=2)
        engine = CombatEngine(
            encounter_name="Jaw Worm",
            randoms=randoms,
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.step({"kind": "end", "name": "END_TURN"})
        monster = engine.state.monsters[0]
        self.assertNotEqual(monster.next_move, "CHOMP")
        self.assertEqual(monster.intent, "DEFEND_BUFF")

    def test_jaw_worm_bellow_re_rolls_into_thrash_for_seed1_floor2(self):
        engine = CombatEngine(
            encounter_name="Jaw Worm",
            randoms=NativeRandomSet(seed=1, floor=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.step({"kind": "end", "name": "END_TURN"})
        monster = engine.state.monsters[0]
        self.assertEqual(monster.next_move, "BELLOW")
        engine.step({"kind": "end", "name": "END_TURN"})
        self.assertEqual(monster.next_move, "THRASH")
        self.assertEqual(monster.intent, "ATTACK_DEFEND")
        self.assertEqual(monster.move_adjusted_damage, 10)
        self.assertEqual(monster.block, 6)

    def test_jaw_worm_horde_starts_in_hard_mode_with_prebattle_buff_and_rolled_moves(self):
        engine = CombatEngine(
            encounter_name="Jaw Worm Horde",
            randoms=NativeRandomSet(seed=50, floor=42),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=76, max_hp=76),
            master_deck=[],
        )

        self.assertEqual([monster.monster_id for monster in engine.state.monsters], ["JawWorm", "JawWorm", "JawWorm"])
        self.assertEqual([monster.block for monster in engine.state.monsters], [6, 6, 6])
        self.assertEqual([_power_amount(monster, "Strength") for monster in engine.state.monsters], [3, 3, 3])
        self.assertEqual([monster.next_move for monster in engine.state.monsters], ["BELLOW", "THRASH", "THRASH"])
        self.assertEqual([monster.intent for monster in engine.state.monsters], ["DEFEND_BUFF", "ATTACK_DEFEND", "ATTACK_DEFEND"])
        self.assertEqual([monster.move_adjusted_damage for monster in engine.state.monsters], [0, 10, 10])

    def test_monster_death_clears_block(self):
        engine = CombatEngine(
            encounter_name="Jaw Worm",
            randoms=NativeRandomSet(seed=2, floor=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.block = 5

        engine._kill_monster(monster)

        self.assertEqual(monster.current_hp, 0)
        self.assertEqual(monster.block, 0)

    def test_monster_thorns_retaliates_even_when_lethal_attack_kills_it(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2, floor=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        spiker = combat_engine_mod._spawn_spiker(NativeRandomSet(seed=3), 0)
        spiker.current_hp = 1
        spiker.block = 0
        combat_engine_mod._add_power(spiker, "Thorns", 3)
        engine.state.monsters = [spiker]
        engine.player.block = 7

        dealt = engine._player_deal_damage(spiker, 99, attack_card=True)

        self.assertEqual(dealt, 1)
        self.assertEqual(spiker.current_hp, 0)
        self.assertEqual(engine.player.current_hp, 80)
        self.assertEqual(engine.player.block, 4)

    def test_jaw_worm_thrash_does_not_gain_block_after_dying_to_thorns(self):
        engine = CombatEngine(
            encounter_name="Jaw Worm",
            randoms=NativeRandomSet(seed=2, floor=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.current_hp = 3
        monster.block = 0
        monster.next_move = "THRASH"
        monster.meta["thrash_block"] = 5
        combat_engine_mod._append_power(engine.player, "Thorns", 3, misc=3)
        ai_counter_before = engine.randoms.stream("ai").counter

        engine._jaw_worm_take_turn(monster)

        self.assertEqual(monster.current_hp, 0)
        self.assertEqual(monster.block, 0)
        self.assertEqual(monster.powers, [])
        self.assertGreater(engine.randoms.stream("ai").counter, ai_counter_before)

    def test_combust_applies_player_power(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[make_card("Combust", uuid="combust-test")],
        )
        engine.state.hand = [dict(engine.master_deck[0])]
        engine.state.draw_pile = [
            make_card("Strike_R", uuid="gamblers-draw-order-0"),
            make_card("Defend_R", uuid="gamblers-draw-order-1"),
            make_card("Bash", uuid="gamblers-draw-order-2"),
        ]
        engine.player.energy = 3
        action = engine.legal_actions()[0]
        engine.step(action)
        self.assertTrue(any(power["power_id"] == "Combust" for power in engine.player.powers))

    def test_demon_form_uses_real_power_id_with_space(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[make_card("Demon Form", uuid="demon-form-test")],
        )
        engine.state.hand = [dict(engine.master_deck[0])]
        engine.state.draw_pile = []
        engine.player.energy = 3
        action = engine.legal_actions()[0]
        engine.step(action)
        self.assertTrue(any(power["power_id"] == "Demon Form" for power in engine.player.powers))
        self.assertEqual(combat_engine_mod._get_power_amount(engine.player, "DemonForm"), 2)

    def test_live_intangible_player_alias_reduces_damage(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=19, max_hp=40),
            master_deck=[],
        )
        engine.player.powers = [
            {"power_id": "IntangiblePlayer", "id": "IntangiblePlayer", "amount": 3, "name": "IntangiblePlayer"}
        ]
        self.assertEqual(combat_engine_mod._get_power_amount(engine.player, "Intangible"), 3)
        self.assertTrue(combat_engine_mod._has_power(engine.player, "Intangible"))
        dealt = engine._damage_player(24, apply_player_vulnerable=False)
        self.assertEqual(dealt, 1)
        self.assertEqual(engine.player.current_hp, 18)

    def test_upgraded_apparition_loses_ethereal(self):
        self.assertTrue(make_card("Ghostly")["ethereal"])
        self.assertFalse(make_card("Ghostly", upgrades=1)["ethereal"])

    def test_upgraded_apparition_discards_at_end_turn(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        base_apparition = make_card("Ghostly", uuid="apparition-base")
        upgraded_apparition = make_card("Ghostly", upgrades=1, uuid="apparition-plus")
        engine.state.hand = [base_apparition, upgraded_apparition]
        engine.state.draw_pile = [make_card("Strike_R", uuid=f"draw-{index}") for index in range(5)]
        engine.state.discard_pile = []
        engine.state.exhaust_pile = []
        engine.player.energy = 0

        end_action = next(action for action in engine.legal_actions() if action.get("kind") == "end")
        engine.step(end_action)

        self.assertIn("apparition-base", [card["uuid"] for card in engine.state.exhaust_pile])
        self.assertIn("apparition-plus", [card["uuid"] for card in engine.state.discard_pile])

    def test_runic_dome_hidden_intent_matches_comm_mod_surface(self):
        monster = MonsterState(
            monster_id="SphericGuardian",
            name="Spheric Guardian",
            current_hp=20,
            max_hp=20,
            block=40,
            intent="ATTACK_DEFEND",
            move_adjusted_damage=10,
            move_hits=1,
        )

        payload = serialize_adapter._serialize_monster(monster, hide_intent=True)

        self.assertEqual(payload["intent"], "NONE")
        self.assertIsNone(payload["move_adjusted_damage"])
        self.assertIsNone(payload["move_hits"])

    def test_library_card_select_obtains_selected_card_immediately(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "CARD_SELECT"
        env.current_card_select = {
            "mode": "library",
            "cards": [make_card("Dropkick", uuid="library-dropkick")],
            "return_phase": "EVENT",
            "source": "event",
            "remaining_picks": 1,
            "total_picks": 1,
            "selected_target_indexes": [],
            "selected_cards": [],
            "requires_confirm": False,
            "clear_event_on_finish": False,
        }
        deck_len_before = len(env.deck)

        env._step_card_select({"kind": "card_select", "choice_index": 0})

        self.assertEqual(len(env.deck), deck_len_before + 1)
        self.assertEqual(env.deck[-1]["card_id"], "Dropkick")
        self.assertEqual(env.phase, "EVENT")

    def test_inflame_grants_strength(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Inflame", uuid="inflame")]
        engine.state.draw_pile = []
        engine.player.energy = 3
        inflame = engine.state.hand[0]
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Inflame")
        engine.step(action)
        self.assertEqual(next(power["amount"] for power in engine.player.powers if power["power_id"] == "Strength"), int(inflame.get("base_magic") or 0))

    def test_true_grit_grants_block_and_exhausts_hand_card(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("True Grit", uuid="true-grit"),
            make_card("Strike_R", uuid="exhaust-target"),
        ]
        engine.state.draw_pile = []
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "True Grit")
        engine.step(action)
        self.assertGreater(engine.player.block, 0)
        self.assertEqual(len(engine.state.exhaust_pile), 1)
        self.assertEqual(engine.state.exhaust_pile[0]["card_id"], "Strike_R")

    def test_upgraded_true_grit_opens_card_select(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("True Grit", upgrades=1, uuid="true-grit-plus"),
            make_card("Strike_R", uuid="exhaust-target-1"),
            make_card("Defend_R", uuid="exhaust-target-2"),
        ]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "True Grit")

        engine.step(action)

        self.assertGreater(engine.player.block, 0)
        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.pending_card_select["mode"], "TRUE_GRIT")
        self.assertEqual([choice["card_id"] for choice in engine.legal_actions()], ["Strike_R", "Defend_R"])
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], [])

        engine.step(engine.legal_actions()[1])
        self.assertEqual([action["kind"] for action in engine.legal_actions()], ["confirm"])
        engine.step(engine.legal_actions()[0])

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Strike_R"])
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Defend_R"])
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["True Grit"])

    def test_upgraded_true_grit_card_select_flushes_dark_embrace_draw(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        combat_engine_mod._append_power(engine.player, "Dark Embrace", 1)
        engine.state.hand = [
            make_card("True Grit", upgrades=1, uuid="true-grit-plus"),
            make_card("Strike_R", uuid="exhaust-target-1"),
            make_card("Defend_R", uuid="exhaust-target-2"),
        ]
        engine.state.draw_pile = [make_card("Bash", uuid="dark-embrace-draw")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "True Grit")

        engine.step(action)
        engine.step(engine.legal_actions()[1])
        engine.step(engine.legal_actions()[0])

        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Strike_R", "Bash"])
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Defend_R"])

    def test_upgraded_true_grit_auto_exhausts_when_only_one_card_can_be_chosen(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("True Grit", upgrades=1, uuid="true-grit-plus"),
            make_card("Intimidate", uuid="single-target"),
        ]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "True Grit")

        engine.step(action)

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Intimidate"])
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["True Grit"])

    def test_toy_ornithopter_heal_waits_for_skill_potion_discovery_choice(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=70, max_hp=80),
            master_deck=[],
            relics=[make_relic("Toy Ornithopter")],
            potions=[make_potion("SkillPotion")],
        )
        potion_action = next(candidate for candidate in engine.legal_actions() if candidate.get("potion_id") == "SkillPotion")

        engine.step(potion_action)

        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.player.current_hp, 70)

        choice_action = next(candidate for candidate in engine.legal_actions() if candidate.get("kind") == "card_reward")
        engine.step(choice_action)

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual(engine.player.current_hp, 75)

    def test_double_tap_replays_next_attack_and_expires(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Double Tap", uuid="double-tap"),
            make_card("Strike_R", uuid="strike"),
        ]
        engine.state.draw_pile = []
        engine.player.energy = 3
        target = engine.state.monsters[0]
        before_hp = target.current_hp
        tap_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Double Tap")
        engine.step(tap_action)
        self.assertEqual(_power_amount(engine.player, "Double Tap"), 1)
        strike_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Strike_R")
        engine.step(strike_action)
        self.assertEqual(target.current_hp, before_hp - 12)
        self.assertEqual(_power_amount(engine.player, "Double Tap"), 0)

    def test_double_tap_replay_counts_as_real_attack_play_for_relics(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            relics=[make_relic("Nunchaku")],
        )
        engine.state.hand = [
            make_card("Double Tap", uuid="double-tap"),
            make_card("Strike_R", uuid="strike"),
        ]
        engine.state.draw_pile = []
        engine.player.energy = 3
        nunchaku = next(relic for relic in engine.relics if relic["relic_id"] == "Nunchaku")
        nunchaku["counter"] = 5

        tap_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Double Tap")
        engine.step(tap_action)
        strike_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Strike_R")
        engine.step(strike_action)

        self.assertEqual(nunchaku["counter"], 7)
        self.assertEqual(engine.cards_played_this_turn, 3)
        self.assertEqual(_power_amount(engine.player, "Double Tap"), 0)

    def test_double_tap_replay_does_not_reuse_active_pen_nib(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            relics=[make_relic("Pen Nib", counter=9)],
        )
        target = engine.state.monsters[0]
        target.current_hp = 200
        target.max_hp = 200
        target.powers.append({"power_id": "Vulnerable", "id": "Vulnerable", "amount": 2, "misc": 2})
        engine.player.powers.append({"power_id": "Double Tap", "id": "Double Tap", "amount": 2, "misc": 2})
        engine.state.hand = [make_card("Pummel", uuid="pummel")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Pummel")
        before_hp = target.current_hp
        engine.step(action)

        self.assertEqual(before_hp - target.current_hp, 36)
        self.assertEqual(_power_amount(engine.player, "Pen Nib"), 0)
        self.assertEqual(_power_amount(engine.player, "Double Tap"), 1)

    def test_double_tap_consumes_power_but_skips_replay_when_target_dies(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            relics=[make_relic("Nunchaku")],
        )
        target = engine.state.monsters[0]
        target.current_hp = 6
        engine.state.hand = [
            make_card("Double Tap", uuid="double-tap"),
            make_card("Strike_R", uuid="strike"),
        ]
        engine.state.draw_pile = []
        engine.player.energy = 3
        nunchaku = next(relic for relic in engine.relics if relic["relic_id"] == "Nunchaku")
        nunchaku["counter"] = 2

        tap_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Double Tap")
        engine.step(tap_action)
        strike_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Strike_R")
        engine.step(strike_action)

        self.assertEqual(nunchaku["counter"], 3)
        self.assertEqual(engine.cards_played_this_turn, 2)
        self.assertEqual(_power_amount(engine.player, "Double Tap"), 0)

    def test_double_tap_consumes_before_headbutt_card_select_resolves(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        target.current_hp = 100
        target.max_hp = 100
        engine.player.powers.append({"power_id": "Double Tap", "id": "Double Tap", "amount": 1, "misc": 1})
        engine.state.hand = [make_card("Headbutt", uuid="headbutt")]
        engine.state.discard_pile = [
            make_card("Double Tap", uuid="discard-double-tap"),
            make_card("Thunderclap", uuid="discard-thunderclap"),
        ]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Headbutt")
        before_hp = target.current_hp
        engine.step(action)

        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.pending_card_select["mode"], "HEADBUTT")
        self.assertEqual(_power_amount(engine.player, "Double Tap"), 0)
        self.assertIn("deferred_double_tap_replay", engine.pending_card_select)
        self.assertEqual(target.current_hp, before_hp - 9)

    def test_necronomicon_replays_first_eligible_attack_once_per_turn(self):
        necronomicon = make_relic("Necronomicon")
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            relics=[necronomicon],
        )
        target = engine.state.monsters[0]
        target.current_hp = 100
        target.max_hp = 100
        engine.state.hand = [
            make_card("Bash", uuid="bash-1"),
            make_card("Bash", uuid="bash-2"),
        ]
        engine.state.draw_pile = []
        engine.player.energy = 10

        first_bash = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Bash")
        engine.step(first_bash)

        self.assertEqual(target.current_hp, 80)
        self.assertEqual(_power_amount(target, "Vulnerable"), 4)
        self.assertEqual(engine.cards_played_this_turn, 2)
        self.assertEqual(necronomicon["counter"], -1)

        second_bash = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Bash")
        engine.step(second_bash)

        self.assertEqual(target.current_hp, 68)
        self.assertEqual(_power_amount(target, "Vulnerable"), 6)
        self.assertEqual(engine.cards_played_this_turn, 3)
        self.assertEqual(necronomicon["counter"], -1)

    def test_upgraded_blood_for_blood_refreshes_from_upgraded_base_cost(self):
        card = make_card("Blood for Blood", upgrades=1, uuid="blood-plus")

        combat_engine_mod._refresh_card_flags([card], PlayerState(current_hp=80, max_hp=80))

        self.assertEqual(card["base_cost"], 3)
        self.assertEqual(card["cost"], 3)
        self.assertEqual(card["cost_for_combat"], 3)
        self.assertEqual(card["cost_for_turn"], 3)

    def test_slime_boss_split_replaces_boss_with_large_slimes(self):
        engine = CombatEngine(
            encounter_name="Slime Boss",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        boss = engine.state.monsters[0]
        boss.current_hp = boss.max_hp // 2
        engine._player_deal_damage(boss, 1)
        self.assertEqual(boss.next_move, "SPLIT")
        engine._slime_boss_take_turn(boss)
        self.assertEqual([monster.monster_id for monster in engine.state.monsters], ["SpikeSlime_L", "SlimeBoss", "AcidSlime_L"])
        self.assertEqual(boss.current_hp, 0)
        self.assertEqual(engine.outcome, "UNDECIDED")

    def test_large_slime_split_uses_spawn_monster_smart_positioning(self):
        engine = CombatEngine(
            encounter_name="Slime Boss",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        boss = engine.state.monsters[0]
        boss.current_hp = boss.max_hp // 2
        boss.next_move = "SPLIT"
        engine._slime_boss_take_turn(boss)

        acid_l = engine.state.monsters[2]
        acid_l.current_hp = 22
        acid_l.next_move = "SPLIT"
        engine._acid_slime_l_take_turn(acid_l)

        self.assertEqual(
            [monster.monster_id for monster in engine.state.monsters],
            ["SpikeSlime_L", "AcidSlime_M", "SlimeBoss", "AcidSlime_L", "AcidSlime_M"],
        )
        self.assertEqual([monster.current_hp for monster in engine.state.monsters], [70, 22, 0, 0, 22])
        self.assertEqual(engine.outcome, "UNDECIDED")

    def test_guardian_mode_shift_triggers_defensive_close_up(self):
        engine = CombatEngine(
            encounter_name="The Guardian",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        guardian = engine.state.monsters[0]
        threshold = int(guardian.meta["mode_shift_threshold"])
        engine._player_deal_damage(guardian, threshold)
        self.assertFalse(bool(guardian.meta["is_open"]))
        self.assertEqual(guardian.next_move, "CLOSE_UP")
        self.assertGreaterEqual(guardian.block, 20)

    def test_guardian_mode_shift_waits_until_multi_hit_card_finishes(self):
        engine = CombatEngine(
            encounter_name="The Guardian",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        guardian = engine.state.monsters[0]
        guardian.current_hp = 215
        guardian.block = 0
        guardian.powers = []
        guardian.meta["dmg_taken"] = 25
        guardian.meta["is_open"] = True
        guardian.meta["close_up_triggered"] = False
        combat_engine_mod._append_power(guardian, "Mode Shift", 5, misc=30)
        combat_engine_mod._append_power(guardian, "Vulnerable", 1, misc=1)
        combat_engine_mod._append_power(guardian, "Strength", -3, misc=-3)
        combat_engine_mod._append_power(engine.player, "Strength", 2, misc=2)
        engine.state.hand = [make_card("Sword Boomerang", uuid="boomerang")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Sword Boomerang")
        engine.step(action)

        self.assertEqual(guardian.current_hp, 194)
        self.assertEqual(guardian.block, 20)
        self.assertEqual(_power_amount(guardian, "Mode Shift"), 0)
        self.assertFalse(bool(guardian.meta["is_open"]))
        self.assertEqual(guardian.next_move, "CLOSE_UP")
        self.assertEqual(guardian.intent, "BUFF")

    def test_guardian_mode_shift_block_from_end_turn_damage_survives_close_up(self):
        engine = CombatEngine(
            encounter_name="The Guardian",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        guardian = engine.state.monsters[0]
        guardian.current_hp = 216
        guardian.block = 0
        guardian.meta["dmg_taken"] = 24
        guardian.meta["is_open"] = True
        guardian.meta["close_up_triggered"] = False
        combat_engine_mod._set_power_amount(guardian, "Mode Shift", 6)
        combat_engine_mod._append_power(engine.player, "Combust", 7, misc=1)
        engine.state.hand = []
        engine.player.energy = 0

        engine.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(guardian.current_hp, 209)
        self.assertEqual(guardian.block, 20)
        self.assertEqual(_power_amount(guardian, "Mode Shift"), 0)
        self.assertEqual(_power_amount(guardian, "Sharp Hide"), 3)
        self.assertEqual(guardian.next_move, "ROLL_ATTACK")

    def test_guardian_mode_shift_block_from_player_card_clears_before_close_up(self):
        engine = CombatEngine(
            encounter_name="The Guardian",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        guardian = engine.state.monsters[0]
        guardian.current_hp = 213
        guardian.block = 0
        guardian.meta["dmg_taken"] = 27
        guardian.meta["is_open"] = True
        guardian.meta["close_up_triggered"] = False
        combat_engine_mod._set_power_amount(guardian, "Mode Shift", 3)
        engine.state.hand = [make_card("Headbutt", uuid="headbutt")]
        engine.state.discard_pile = [
            make_card("Strike_R", uuid="discard-strike"),
            make_card("Defend_R", uuid="discard-defend"),
        ]
        engine.player.energy = 1

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Headbutt")
        engine.step(action)
        engine.step({"kind": "card_select", "mode": "HEADBUTT", "choice_index": 0, "card_index": 0})
        engine.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(guardian.block, 0)
        self.assertEqual(_power_amount(guardian, "Sharp Hide"), 3)
        self.assertEqual(guardian.next_move, "ROLL_ATTACK")

    def test_guardian_mode_shift_waits_until_headbutt_card_select_resolves(self):
        engine = CombatEngine(
            encounter_name="The Guardian",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        guardian = engine.state.monsters[0]
        guardian.current_hp = 54
        guardian.block = 0
        guardian.meta["dmg_taken"] = 25
        guardian.meta["is_open"] = True
        guardian.meta["close_up_triggered"] = False
        combat_engine_mod._set_power_amount(guardian, "Mode Shift", 5)
        combat_engine_mod._append_power(guardian, "Vulnerable", 3, misc=3)
        engine.state.hand = [make_card("Headbutt", uuid="headbutt")]
        engine.state.discard_pile = [
            make_card("Strike_R", uuid="discard-strike"),
            make_card("Defend_R", uuid="discard-defend"),
            make_card("Flex", uuid="discard-flex"),
        ]
        engine.player.energy = 1

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Headbutt")
        engine.step(action)

        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(guardian.block, 0)
        self.assertEqual(_power_amount(guardian, "Mode Shift"), -8)
        self.assertTrue(bool(guardian.meta["is_open"]))

        engine.step({"kind": "card_select", "mode": "HEADBUTT", "choice_index": 0, "card_index": 0})

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual(guardian.block, 20)
        self.assertEqual(_power_amount(guardian, "Mode Shift"), 0)
        self.assertFalse(bool(guardian.meta["is_open"]))
        self.assertEqual(guardian.next_move, "CLOSE_UP")

    def test_flame_barrier_retaliation_counts_toward_guardian_mode_shift(self):
        engine = CombatEngine(
            encounter_name="The Guardian",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        guardian = engine.state.monsters[0]
        guardian.current_hp = 174
        guardian.block = 0
        guardian.powers = []
        guardian.meta["dmg_taken"] = 24
        guardian.meta["is_open"] = True
        guardian.meta["close_up_triggered"] = False
        guardian.next_move = "WHIRLWIND"
        combat_engine_mod._append_power(guardian, "Mode Shift", 16, misc=40)
        combat_engine_mod._append_power(engine.player, "FlameBarrier", 4, misc=4)
        engine.player.block = 12

        engine._guardian_take_turn(guardian)

        self.assertLessEqual(_power_amount(guardian, "Mode Shift"), 0)
        self.assertFalse(bool(guardian.meta["is_open"]))
        self.assertEqual(guardian.block, 20)
        self.assertEqual(guardian.intent, "BUFF")

    def test_guardian_twin_slam_applies_new_mode_shift_after_flame_barrier_retaliation(self):
        engine = CombatEngine(
            encounter_name="The Guardian",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        guardian = engine.state.monsters[0]
        guardian.current_hp = 65
        guardian.block = 0
        guardian.powers = []
        guardian.meta["is_open"] = False
        guardian.meta["close_up_triggered"] = True
        guardian.meta["mode_shift_threshold"] = 50
        guardian.meta["dmg_taken"] = 0
        guardian.next_move = "TWIN_SLAM"
        combat_engine_mod._append_power(guardian, "Sharp Hide", 3, misc=3)
        combat_engine_mod._append_power(engine.player, "FlameBarrier", 4, misc=4)
        engine.player.block = 12

        engine._guardian_take_turn(guardian)

        self.assertEqual(guardian.current_hp, 57)
        self.assertEqual(_power_amount(guardian, "Mode Shift"), 50)
        self.assertEqual(_power_amount(guardian, "Sharp Hide"), 0)
        self.assertEqual(guardian.meta["dmg_taken"], 0)
        self.assertEqual(guardian.next_move, "WHIRLWIND")

    def test_guardian_mode_shift_counter_decreases_on_hp_damage(self):
        engine = CombatEngine(
            encounter_name="The Guardian",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        guardian = engine.state.monsters[0]
        threshold = int(guardian.meta["mode_shift_threshold"])
        engine._player_deal_damage(guardian, 10)
        self.assertTrue(bool(guardian.meta["is_open"]))
        self.assertEqual(_power_amount(guardian, "Mode Shift"), threshold - 10)

    def test_guardian_offensive_mode_resets_accumulated_mode_shift_damage(self):
        engine = CombatEngine(
            encounter_name="The Guardian",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        guardian = engine.state.monsters[0]
        guardian.meta["is_open"] = False
        guardian.meta["close_up_triggered"] = True
        guardian.meta["mode_shift_threshold"] = 40
        guardian.meta["dmg_taken"] = 31
        guardian.next_move = "TWIN_SLAM"
        guardian.powers = []
        combat_engine_mod._append_power(guardian, "Sharp Hide", 3, misc=3)

        engine._guardian_take_turn(guardian)

        self.assertTrue(bool(guardian.meta["is_open"]))
        self.assertFalse(bool(guardian.meta["close_up_triggered"]))
        self.assertEqual(guardian.meta["dmg_taken"], 0)
        self.assertEqual(_power_amount(guardian, "Mode Shift"), 40)

    def test_guardian_sharp_hide_retaliates_once_on_attack_card_use(self):
        engine = CombatEngine(
            encounter_name="The Guardian",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.block = 5
        guardian = engine.state.monsters[0]
        combat_engine_mod._append_power(guardian, "Sharp Hide", 3, misc=3)
        engine.state.hand = [make_card("Strike_R", uuid="strike")]
        engine.player.energy = 1
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Strike_R")
        engine.step(action)

        self.assertEqual(engine.player.current_hp, 80)
        self.assertEqual(engine.player.block, 2)
        self.assertEqual(guardian.current_hp, guardian.max_hp - 6)

    def test_guardian_sharp_hide_retaliates_once_for_sword_boomerang(self):
        engine = CombatEngine(
            encounter_name="The Guardian",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=4, max_hp=80),
            master_deck=[],
        )
        guardian = engine.state.monsters[0]
        guardian.current_hp = 62
        guardian.block = 0
        guardian.powers = []
        combat_engine_mod._append_power(guardian, "Sharp Hide", 3, misc=3)
        combat_engine_mod._append_power(guardian, "Vulnerable", 3, misc=3)
        combat_engine_mod._append_power(guardian, "Strength", -5, misc=-5)
        combat_engine_mod._append_power(engine.player, "Strength", 2, misc=2)
        engine.state.hand = [make_card("Sword Boomerang", uuid="boomerang")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Sword Boomerang")
        engine.step(action)

        self.assertEqual(engine.player.current_hp, 1)
        self.assertEqual(engine.outcome, "UNDECIDED")
        self.assertEqual(guardian.current_hp, 41)

    def test_guardian_whirlwind_applies_monster_weak_when_dealing_damage(self):
        engine = CombatEngine(
            encounter_name="The Guardian",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=15, max_hp=80),
            master_deck=[],
        )
        guardian = engine.state.monsters[0]
        guardian.next_move = "WHIRLWIND"
        combat_engine_mod._append_power(guardian, "Weak", 1, misc=1)
        engine._guardian_take_turn(guardian)
        self.assertEqual(engine.player.current_hp, 3)
        self.assertEqual(engine.outcome, "UNDECIDED")

    def test_boss_victory_enters_card_reward_then_boss_relic_and_advances_act(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.floor = 16
        env.phase = "COMBAT"
        env.current_room_type = "MonsterRoomBoss"
        class _BossVictoryCombat:
            def __init__(self, player):
                self.player = player
                self.encounter_name = "Slime Boss"

            def step(self, _action):
                return "VICTORY"

        env.combat = _BossVictoryCombat(env.player)
        original_pool = list(env.boss_relic_pool)
        env._step_combat({"kind": "end"})
        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertTrue(env.boss_reward_pending_boss_relic)
        self.assertIsNotNone(env.reward_gold)
        self.assertEqual(len(env.reward_cards), 3)
        self.assertEqual(len(env.boss_relic_options), 0)
        self.assertEqual(env.boss_relic_pool, original_pool)

        env.reward_gold = None
        env.reward_cards = []
        env._step_card_reward({"kind": "proceed"})

        self.assertEqual(env.phase, "TREASURE")
        self.assertEqual(env.floor, 17)
        self.assertEqual(env.current_room_type, "TreasureRoomBoss")
        env._step_treasure({"kind": "treasure", "choice_index": 0})

        self.assertEqual(env.phase, "BOSS_RELIC")
        self.assertEqual(len(env.boss_relic_options), 3)
        self.assertEqual(len(env.boss_relic_pool), len(original_pool) - 3)
        env._step_boss_relic({"kind": "boss_relic", "choice_index": 0})
        self.assertEqual(env.phase, "TREASURE")
        self.assertTrue(env.boss_relic_pending_act_advance)
        self.assertTrue(env.current_treasure.opened)
        env._step_treasure({"kind": "treasure", "choice_index": 0})
        self.assertEqual(env.act, 2)
        self.assertEqual(env.phase, "MAP")
        self.assertNotEqual(env.boss_relic_pool, original_pool)

    def test_act3_boss_victory_enters_boss_relic_instead_of_looping_back_to_map(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.act = 3
        env.dungeon_id = "TheBeyond"
        env.floor = 49
        env.phase = "COMBAT"
        env.current_room_type = "MonsterRoomBoss"

        class _Act3BossVictoryCombat:
            def __init__(self, player):
                self.player = player
                self.encounter_name = "Time Eater"

            def step(self, _action):
                return "VICTORY"

        env.combat = _Act3BossVictoryCombat(env.player)
        env._step_combat({"kind": "end"})

        self.assertEqual(env.phase, "CARD_REWARD")
        env.reward_gold = None
        env.reward_cards = []
        env._step_card_reward({"kind": "proceed"})

        self.assertEqual(env.phase, "TREASURE")
        self.assertEqual(env.floor, 51)
        env._step_treasure({"kind": "treasure", "choice_index": 0})

        self.assertEqual(env.phase, "BOSS_RELIC")
        self.assertEqual(env.current_room_type, "TreasureRoomBoss")
        self.assertEqual(len(env.boss_relic_options), 3)
        env._step_boss_relic({"kind": "skip", "choice_index": 3})
        self.assertEqual(env.phase, "TREASURE")
        self.assertEqual(env.current_room_type, "TreasureRoomBoss")

    def test_native_combat_env_clears_player_combat_state_on_entry(self):
        player = PlayerState(
            current_hp=73,
            max_hp=80,
            block=9,
            energy=0,
            base_energy=3,
            powers=[{"power_id": "Weak", "id": "Weak", "name": "Weak", "amount": 3, "misc": 3}],
        )
        combat = NativeCombatEnv(
            seed=20,
            ascension_level=0,
            encounter_name="Jaw Worm",
            player=player,
            master_deck=[],
        )

        self.assertEqual(combat.player.current_hp, 73)
        self.assertEqual(combat.player.max_hp, 80)
        self.assertEqual(combat.player.block, 0)
        self.assertEqual(combat.player.energy, 3)
        self.assertEqual(combat.player.powers, [])

    def test_native_combat_env_syncs_gold_after_looter_steals(self):
        combat = NativeCombatEnv(
            seed=20,
            ascension_level=0,
            encounter_name="Looter",
            gold=99,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )

        combat.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(combat.engine.gold, 84)
        self.assertEqual(combat.gold, 84)
        self.assertEqual(combat.serialize()["gold"], 84)

    def test_looter_first_mug_consumes_opening_ai_bark_roll(self):
        engine = CombatEngine(
            encounter_name="Looter",
            randoms=NativeRandomSet(seed=20),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            gold=99,
        )
        monster = engine.state.monsters[0]
        ai_counter_before = engine.randoms.stream("ai").counter

        engine._looter_take_turn(monster)

        self.assertEqual(engine.randoms.stream("ai").counter, ai_counter_before + 1)
        self.assertEqual(monster.next_move, "MUG")
        self.assertEqual(engine.gold, 84)

    def test_looter_spawn_consumes_initial_roll_move_ai_call(self):
        engine = CombatEngine(
            encounter_name="Looter",
            randoms=NativeRandomSet(seed=20),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            gold=99,
        )

        self.assertEqual(engine.randoms.stream("ai").counter, 1)
        self.assertEqual(engine.state.monsters[0].next_move, "MUG")

    def test_cultist_deterministic_roll_move_still_consumes_ai_rng(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=20),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        cultist = engine.state.monsters[0]
        self.assertEqual(engine.randoms.stream("ai").counter, 1)

        ai_counter_before = engine.randoms.stream("ai").counter
        engine._cultist_take_turn(cultist)

        self.assertEqual(engine.randoms.stream("ai").counter, ai_counter_before + 1)
        self.assertEqual(cultist.next_move, "DARK_STRIKE")

        ai_counter_before = engine.randoms.stream("ai").counter
        engine._cultist_take_turn(cultist)

        self.assertEqual(engine.randoms.stream("ai").counter, ai_counter_before + 1)
        self.assertEqual(cultist.next_move, "DARK_STRIKE")

    def test_two_thieves_encounter_spawns_real_looter_and_mugger(self):
        engine = CombatEngine(
            encounter_name="2 Thieves",
            randoms=NativeRandomSet(seed=12, act=2, floor=18),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            gold=316,
        )

        monsters = engine.state.monsters
        self.assertEqual([monster.monster_id for monster in monsters], ["Looter", "Mugger"])
        self.assertEqual([monster.current_hp for monster in monsters], [47, 50])
        self.assertEqual([monster.intent for monster in monsters], ["ATTACK", "ATTACK"])
        self.assertEqual([monster.move_adjusted_damage for monster in monsters], [10, 10])
        self.assertEqual([_power_amount(monster, "Thievery") for monster in monsters], [15, 15])
        self.assertEqual(engine.randoms.stream("ai").counter, 2)

    def test_mugger_first_and_second_mug_consume_real_ai_rolls(self):
        engine = CombatEngine(
            encounter_name="2 Thieves",
            randoms=NativeRandomSet(seed=20),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            gold=99,
        )
        mugger = engine.state.monsters[1]
        ai_counter_before = engine.randoms.stream("ai").counter

        engine._looter_take_turn(mugger)

        self.assertEqual(engine.randoms.stream("ai").counter, ai_counter_before + 1)
        self.assertEqual(mugger.next_move, "MUG")
        self.assertEqual(engine.gold, 84)

        ai_counter_before = engine.randoms.stream("ai").counter
        engine._looter_take_turn(mugger)

        self.assertEqual(engine.randoms.stream("ai").counter, ai_counter_before + 3)
        self.assertIn(mugger.next_move, {"SMOKE_BOMB", "BIGSWIPE"})
        self.assertEqual(engine.gold, 69)

    def test_mugger_death_consumes_ai_death_sfx_and_returns_stolen_gold(self):
        engine = CombatEngine(
            encounter_name="2 Thieves",
            randoms=NativeRandomSet(seed=20),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            gold=99,
        )
        mugger = engine.state.monsters[1]
        mugger.meta["stolen_gold"] = 15
        ai_counter_before = engine.randoms.stream("ai").counter

        engine._kill_monster(mugger)

        self.assertEqual(engine.randoms.stream("ai").counter, ai_counter_before + 1)
        self.assertEqual(engine.bonus_reward_gold, 15)

    def test_last_thief_escape_ends_combat_as_victory(self):
        engine = CombatEngine(
            encounter_name="2 Thieves",
            randoms=NativeRandomSet(seed=20),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            gold=99,
        )
        looter, mugger = engine.state.monsters
        engine._kill_monster(looter, escaped=True)
        mugger.next_move = "ESCAPE"
        engine.state.hand = []

        result = engine.step({"kind": "end"})

        self.assertEqual(result.outcome, "VICTORY")
        self.assertEqual(engine.outcome, "VICTORY")
        self.assertFalse(any(combat_engine_mod._alive(monster) for monster in engine.state.monsters))

    def test_escaped_looter_preserves_visible_hp_and_powers_but_is_gone(self):
        engine = CombatEngine(
            encounter_name="Looter",
            randoms=NativeRandomSet(seed=20),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            gold=99,
        )
        looter = engine.state.monsters[0]
        looter.current_hp = 6
        looter.block = 6
        looter.next_move = "ESCAPE"
        combat_engine_mod._append_power(looter, "Vulnerable", 1, misc=1)

        engine._looter_take_turn(looter)
        visible = _serialize_combat_state(engine.state)["monsters"][0]

        self.assertEqual(looter.current_hp, 6)
        self.assertEqual(looter.block, 0)
        self.assertGreater(_power_amount(looter, "Thievery"), 0)
        self.assertGreater(_power_amount(looter, "Vulnerable"), 0)
        self.assertFalse(combat_engine_mod._alive(looter))
        self.assertTrue(visible["is_gone"])
        self.assertEqual(visible["current_hp"], 6)

    def test_spawn_red_slaver_consumes_opening_ai_roll_at_battle_start(self):
        randoms = NativeRandomSet(seed=20)

        monster = combat_engine_mod._spawn_red_slaver(randoms, 0)

        self.assertEqual(randoms.stream("ai").counter, 0)
        self.assertEqual(monster.next_move, "STAB")
        self.assertTrue(monster.meta.get("opening_ai_roll"))

        engine = combat_engine_mod.CombatEngine(
            encounter_name="Red Slaver",
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            randoms=randoms,
            ascension_level=0,
            room_type="MonsterRoom",
        )

        self.assertEqual(randoms.stream("ai").counter, 1)
        self.assertEqual(engine.state.monsters[0].next_move, "STAB")
        self.assertFalse(engine.state.monsters[0].meta.get("opening_ai_roll", True))

    def test_bottom_get_strong_humanoid_consumes_misc_for_slaver_preview_and_pick(self):
        randoms = NativeRandomSet(seed=20)

        monster = combat_engine_mod._bottom_get_strong_humanoid(randoms, 0)

        self.assertEqual(randoms.stream("misc").counter, 2)
        self.assertIn(monster.monster_id, {"Cultist", "SlaverRed", "SlaverBlue", "Looter"})

    def test_spawn_acid_slime_m_defers_opening_move_roll_until_battle_start(self):
        randoms = NativeRandomSet(seed=20)

        monster = combat_engine_mod._spawn_acid_slime_m(randoms, 0)

        self.assertEqual(randoms.stream("ai").counter, 0)
        self.assertEqual(monster.meta.get("opening_move_source"), "AcidSlime_M")

    def test_three_shapes_encounter_is_implemented_and_can_take_a_turn(self):
        engine = CombatEngine(
            encounter_name="3 Shapes",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster_ids = {monster.monster_id for monster in engine.state.monsters}
        self.assertEqual(len(engine.state.monsters), 3)
        self.assertTrue(monster_ids.issubset({"Repulsor", "Spiker", "Exploder"}))
        end_action = next(action for action in engine.legal_actions() if action.get("kind") == "end")
        engine.step(end_action)
        self.assertIn(engine.outcome, {"UNDECIDED", "VICTORY", "DEFEAT"})

    def test_fixed_hp_exploder_still_consumes_monster_hp_rng(self):
        randoms = NativeRandomSet(seed=50)
        randoms.reset_floor_streams(35)

        monsters = combat_engine_mod._spawn_shapes(randoms, ascension_level=0, weak=True)

        self.assertEqual(
            [(monster.monster_id, monster.current_hp) for monster in monsters],
            [("Exploder", 30), ("Repulsor", 31), ("Spiker", 55)],
        )
        self.assertEqual(
            [call["args"] for call in randoms.debug_trace()["monster_hp"]],
            [[30, 30], [29, 35], [42, 56]],
        )
        self.assertEqual(
            [call["result"] for call in randoms.debug_trace()["ai"]],
            [67, 20, 3],
        )
        self.assertEqual(
            [monster.next_move for monster in monsters],
            ["ATTACK", "DAZE", "ATTACK"],
        )

    def test_repulsor_shuffles_dazed_into_random_draw_pile_positions(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=7),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        repulsor = MonsterState(
            monster_id="Repulsor",
            name="Repulsor",
            current_hp=31,
            max_hp=31,
            intent="DEBUFF",
            next_move="DAZE",
            meta={"dazed_amount": 2, "attack_damage": 11},
        )
        engine.state.monsters = [repulsor]
        engine.state.draw_pile = [
            make_card("Strike_R", uuid="draw-0"),
            make_card("Defend_R", uuid="draw-1"),
            make_card("Bash", uuid="draw-2"),
            make_card("Strike_R", uuid="draw-3"),
            make_card("Defend_R", uuid="draw-4"),
        ]

        engine._repulsor_take_turn(repulsor)

        draw_ids = [card["card_id"] for card in engine.state.draw_pile]
        self.assertEqual(draw_ids.count("Dazed"), 2)
        self.assertEqual(engine.state.draw_pile[-1]["uuid"], "draw-4")
        self.assertEqual(
            [call["args"] for call in engine.randoms.debug_trace()["card_random"]],
            [[4], [5]],
        )

    def test_exploder_attack_respects_weak_and_consumes_roll_move_rng(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=8),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.block = 11
        ai_calls_before = len(engine.randoms.debug_trace().get("ai", []))
        exploder = MonsterState(
            monster_id="Exploder",
            name="Exploder",
            current_hp=30,
            max_hp=30,
            intent="ATTACK",
            next_move="ATTACK",
            powers=[
                {"power_id": "Explosive", "amount": 3, "misc": 3},
                {"power_id": "Weakened", "amount": 2, "misc": 2},
            ],
            meta={"turn_count": 0, "attack_damage": 9, "explosive_damage": 30},
        )
        engine.state.monsters = [exploder]

        engine._exploder_take_turn(exploder)

        self.assertEqual(engine.player.current_hp, 80)
        self.assertEqual(engine.player.block, 5)
        self.assertEqual(
            [call["args"] for call in engine.randoms.debug_trace()["ai"][ai_calls_before:]],
            [[99]],
        )

    def test_exploder_third_turn_explodes_and_dies(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        exploder = MonsterState(
            monster_id="Exploder",
            name="Exploder",
            current_hp=30,
            max_hp=30,
            intent="UNKNOWN",
            next_move="BLOCK",
            powers=[{"power_id": "Explosive", "amount": 1, "misc": 1}],
            meta={"turn_count": 2, "explosive_damage": 30},
        )
        engine.state.monsters = [exploder]

        engine._exploder_take_turn(exploder)

        self.assertEqual(exploder.current_hp, 0)
        self.assertEqual(engine.player.current_hp, 50)

    def test_orb_walker_single_encounter_is_implemented(self):
        engine = CombatEngine(
            encounter_name="Orb Walker",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(len(engine.state.monsters), 1)
        self.assertEqual(engine.state.monsters[0].monster_id, "OrbWalker")

    def test_orb_walker_laser_adds_burn_to_random_draw_spot(self):
        engine = CombatEngine(
            encounter_name="Orb Walker",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.next_move = "LASER"
        engine.state.draw_pile = [make_card("Strike_R", uuid=f"strike-{index}") for index in range(5)]
        engine.state.discard_pile = []

        engine._orb_walker_take_turn(monster)

        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Burn"])
        self.assertEqual([card["card_id"] for card in engine.state.draw_pile], ["Strike_R", "Strike_R", "Strike_R", "Strike_R", "Burn", "Strike_R"])

    def test_orb_walker_attack_damage_uses_strength_and_weak(self):
        engine = CombatEngine(
            encounter_name="Orb Walker",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.next_move = "LASER"
        combat_engine_mod._append_power(monster, "Strength", 3)
        combat_engine_mod._append_power(monster, "Weakened", 2)

        engine._orb_walker_take_turn(monster)

        self.assertEqual(engine.player.current_hp, 71)

    def test_giant_head_encounter_is_implemented_and_starts_with_slow(self):
        engine = CombatEngine(
            encounter_name="Giant Head",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(len(engine.state.monsters), 1)
        self.assertEqual(engine.state.monsters[0].monster_id, "GiantHead")
        self.assertTrue(
            any(str(power.get("power_id")) == "Slow" for power in engine.state.monsters[0].powers)
        )

    def test_giant_head_slow_reset_preserves_zero_amount_power(self):
        engine = CombatEngine(
            encounter_name="Giant Head",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        combat_engine_mod._set_power_amount(monster, "Slow", 4)

        combat_engine_mod._set_power_amount(monster, "Slow", 0)

        slow_power = next(power for power in monster.powers if str(power.get("power_id")) == "Slow")
        self.assertEqual(slow_power["amount"], 0)

    def test_nemesis_encounter_is_implemented_and_gains_intangible_after_turn(self):
        engine = CombatEngine(
            encounter_name="Nemesis",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(len(engine.state.monsters), 1)
        monster = engine.state.monsters[0]
        self.assertEqual(monster.monster_id, "Nemesis")
        start_action = next(action for action in engine.legal_actions() if action.get("kind") == "end")
        engine.step(start_action)
        self.assertGreaterEqual(
            next(power for power in monster.powers if str(power.get("power_id")) == "Intangible")["amount"],
            1,
        )

    def test_nemesis_self_applied_intangible_persists_to_cap_next_player_hit(self):
        engine = CombatEngine(
            encounter_name="Nemesis",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.current_hp = 100
        monster.next_move = "TRI_BURN"

        engine._nemesis_take_turn(monster)

        self.assertEqual(_power_amount(monster, "Intangible"), 1)
        engine.state.hand = [make_card("Strike_R", uuid="strike")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Strike_R")
        engine.step(action)

        self.assertEqual(monster.current_hp, 99)

    def test_awakened_one_encounter_is_implemented_with_cultists(self):
        engine = CombatEngine(
            encounter_name="Awakened One",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[make_card("Inflame", uuid="inflame")],
        )
        ids = [monster.monster_id for monster in engine.state.monsters]
        self.assertEqual(ids, ["Cultist", "Cultist", "AwakenedOne"])
        self.assertEqual(ids.count("AwakenedOne"), 1)
        self.assertEqual(ids.count("Cultist"), 2)
        awakened = next(monster for monster in engine.state.monsters if monster.monster_id == "AwakenedOne")
        self.assertEqual(_power_amount(awakened, "Curiosity"), 1)
        engine.state.hand = [make_card("Inflame", uuid="inflame-play")]
        engine.player.energy = 3
        power_action = next(action for action in engine.legal_actions() if action.get("card_id") == "Inflame")
        engine.step(power_action)
        self.assertEqual(_power_amount(awakened, "Strength"), 1)

    def test_awakened_one_rebirth_triggers_on_first_death(self):
        engine = CombatEngine(
            encounter_name="Awakened One",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        awakened = next(monster for monster in engine.state.monsters if monster.monster_id == "AwakenedOne")
        awakened.current_hp = 5
        engine._player_deal_damage(awakened, 99)
        self.assertTrue(bool(awakened.meta.get("half_dead", False)))
        self.assertEqual(awakened.next_move, "REBIRTH")
        self.assertEqual(_power_amount(awakened, "Curiosity"), 0)
        engine._take_monster_turns()
        self.assertFalse(bool(awakened.meta.get("half_dead", False)))
        self.assertEqual(awakened.current_hp, awakened.max_hp)

    def test_awakened_one_attacks_respect_weak(self):
        engine = CombatEngine(
            encounter_name="Awakened One",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        awakened = next(monster for monster in engine.state.monsters if monster.monster_id == "AwakenedOne")
        awakened.next_move = "SLASH"
        awakened.powers.append({"power_id": "Weakened", "id": "Weakened", "amount": 2, "misc": 2})

        before_hp = engine.player.current_hp
        engine._awakened_one_take_turn(awakened)

        self.assertEqual(before_hp - engine.player.current_hp, 15)

    def test_three_darklings_encounter_is_implemented(self):
        engine = CombatEngine(
            encounter_name="3 Darklings",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(len(engine.state.monsters), 3)
        self.assertTrue(all(monster.monster_id == "Darkling" for monster in engine.state.monsters))

    def test_darkling_half_dead_counts_then_reincarnates(self):
        engine = CombatEngine(
            encounter_name="3 Darklings",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        darkling = engine.state.monsters[0]
        darkling.current_hp = 1
        engine._player_deal_damage(darkling, 99)
        self.assertTrue(bool(darkling.meta.get("half_dead", False)))
        self.assertEqual(darkling.next_move, "COUNT")
        self.assertEqual(engine.outcome, "UNDECIDED")
        visible = _serialize_combat_state(engine.state)["monsters"][0]
        self.assertEqual(visible["half_dead"], False)
        self.assertEqual(visible["is_gone"], True)
        self.assertNotIn("intent", visible)
        engine._take_monster_turns()
        self.assertTrue(bool(darkling.meta.get("half_dead", False)))
        self.assertEqual(darkling.next_move, "REINCARNATE")
        engine._take_monster_turns()
        self.assertFalse(bool(darkling.meta.get("half_dead", False)))
        self.assertGreater(darkling.current_hp, 0)
        self.assertEqual(_power_amount(darkling, "Life Link"), -1)

    def test_darkling_half_dead_triggers_gremlin_horn_when_others_alive(self):
        engine = CombatEngine(
            encounter_name="3 Darklings",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80, energy=0),
            master_deck=[],
            relics=[make_relic("Gremlin Horn")],
        )
        engine.state.draw_pile = [make_card("Strike_R", uuid="horn-draw")]
        darkling = engine.state.monsters[0]
        darkling.current_hp = 1
        before_energy = engine.player.energy

        engine._player_deal_damage(darkling, 99)

        self.assertTrue(bool(darkling.meta.get("half_dead", False)))
        self.assertEqual(engine.player.energy, before_energy + 1)
        self.assertEqual([card["uuid"] for card in engine.state.hand], ["horn-draw"])

    def test_darkling_half_dead_roll_move_still_consumes_ai_rng(self):
        randoms = NativeRandomSet(seed=50)
        randoms.reset_floor_streams(36)
        darkling = MonsterState(
            monster_id="Darkling",
            name="Darkling",
            current_hp=0,
            max_hp=49,
            next_move="COUNT",
            meta={"half_dead": True},
        )
        before = randoms.stream("ai").counter
        combat_engine_mod._darkling_roll_next_move(darkling, randoms, [darkling], ascension_level=0)
        self.assertEqual(randoms.stream("ai").counter, before + 1)
        self.assertEqual(darkling.next_move, "REINCARNATE")

    def test_darkling_attacks_respect_weak(self):
        engine = CombatEngine(
            encounter_name="3 Darklings",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        darkling = engine.state.monsters[0]
        darkling.next_move = "NIP"
        combat_engine_mod._add_power(darkling, "Weak", 2)

        engine._darkling_take_turn(darkling)

        self.assertEqual(engine.player.current_hp, 75)

    def test_darkling_chomp_respects_weak_per_hit(self):
        engine = CombatEngine(
            encounter_name="3 Darklings",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        darkling = engine.state.monsters[0]
        darkling.next_move = "CHOMP"
        combat_engine_mod._add_power(darkling, "Weak", 2)

        engine._darkling_take_turn(darkling)

        self.assertEqual(engine.player.current_hp, 68)

    def test_reptomancer_encounter_is_implemented_with_snake_daggers(self):
        engine = CombatEngine(
            encounter_name="Reptomancer",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        ids = [monster.monster_id for monster in engine.state.monsters]
        self.assertEqual(ids.count("Reptomancer"), 1)
        self.assertEqual(ids.count("Dagger"), 2)
        reptomancer = next(monster for monster in engine.state.monsters if monster.monster_id == "Reptomancer")
        self.assertEqual(reptomancer.next_move, "SPAWN_DAGGER")
        engine._reptomancer_take_turn(reptomancer)
        self.assertGreaterEqual([monster.monster_id for monster in engine.state.monsters].count("Dagger"), 3)

    def test_reptomancer_respawns_dagger_in_original_slot_before_corpse(self):
        engine = CombatEngine(
            encounter_name="Reptomancer",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        reptomancer = next(monster for monster in engine.state.monsters if monster.monster_id == "Reptomancer")
        right = next(monster for monster in engine.state.monsters if int(monster.meta.get("reptomancer_slot", -1)) == 0)
        engine._kill_monster(right)
        reptomancer.next_move = "SPAWN_DAGGER"

        engine._reptomancer_take_turn(reptomancer)

        slots = [
            (monster.monster_id, int(monster.meta.get("reptomancer_slot", -1)), int(monster.current_hp))
            for monster in engine.state.monsters
        ]
        self.assertEqual(slots[0][1], 1)
        self.assertEqual(slots[1][0], "Reptomancer")
        self.assertEqual(slots[2][1], 0)
        self.assertGreater(slots[2][2], 0)
        self.assertEqual(slots[3][1], 0)
        self.assertEqual(slots[3][2], 0)

    def test_reptomancer_death_kills_remaining_daggers(self):
        engine = CombatEngine(
            encounter_name="Reptomancer",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        reptomancer = next(monster for monster in engine.state.monsters if monster.monster_id == "Reptomancer")
        engine._kill_monster(reptomancer)
        self.assertTrue(all(monster.current_hp <= 0 for monster in engine.state.monsters))

    def test_time_eater_encounter_is_implemented_and_starts_with_time_warp(self):
        engine = CombatEngine(
            encounter_name="Time Eater",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        self.assertEqual(monster.monster_id, "TimeEater")
        self.assertTrue(any(str(power.get("power_id")) == "Time Warp" for power in monster.powers))

    def test_time_warp_triggers_after_twelve_cards_and_buffs_monsters(self):
        engine = CombatEngine(
            encounter_name="Time Eater",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        for power in monster.powers:
            if str(power.get("power_id")) == "Time Warp":
                power["amount"] = 11
                break
        self.assertTrue(engine._trigger_time_warp_after_card())
        self.assertEqual(_power_amount(monster, "Strength"), 2)
        self.assertEqual(_power_amount(monster, "Time Warp"), 0)

    def test_time_warp_increments_after_normal_hand_card_play(self):
        engine = CombatEngine(
            encounter_name="Time Eater",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[make_card("Spot Weakness", upgrades=1, uuid="spot-weakness")],
        )
        monster = engine.state.monsters[0]
        for power in monster.powers:
            if str(power.get("power_id")) == "Time Warp":
                power["amount"] = 0
                break
        monster.next_move = "RIPPLE"
        engine._update_monster_intents()
        engine.state.hand = [make_card("Spot Weakness", upgrades=1, uuid="spot-weakness-hand")]
        engine.player.energy = 3

        engine.step({"kind": "card", "card_index": 0, "target_index": 0})

        self.assertEqual(_power_amount(monster, "Time Warp"), 1)
        self.assertEqual(engine.player.energy, 2)

    def test_time_eater_reverberate_uses_current_scaled_multi_hit_damage(self):
        engine = CombatEngine(
            encounter_name="Time Eater",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.block = 21
        monster = engine.state.monsters[0]
        monster.next_move = "REVERBERATE"
        combat_engine_mod._append_power(monster, "Weakened", 2, misc=2)
        combat_engine_mod._append_power(engine.player, "Vulnerable", 1, misc=1)

        engine._time_eater_take_turn(monster)

        self.assertEqual(engine.player.current_hp, 80)
        self.assertEqual(engine.player.block, 0)

    def test_donu_and_deca_encounter_is_implemented(self):
        engine = CombatEngine(
            encounter_name="Donu and Deca",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        ids = {monster.monster_id for monster in engine.state.monsters}
        self.assertEqual(ids, {"Donu", "Deca"})
        self.assertTrue(all(_power_amount(monster, "Artifact") == 2 for monster in engine.state.monsters))

    def test_donu_and_deca_team_buffs_are_implemented(self):
        engine = CombatEngine(
            encounter_name="Donu and Deca",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        donu = next(monster for monster in engine.state.monsters if monster.monster_id == "Donu")
        deca = next(monster for monster in engine.state.monsters if monster.monster_id == "Deca")
        donu.next_move = "CIRCLE"
        engine._donu_take_turn(donu)
        self.assertTrue(all(_power_amount(monster, "Strength") >= 3 for monster in engine.state.monsters))
        deca.next_move = "SQUARE"
        engine._deca_take_turn(deca)
        self.assertTrue(all(monster.block >= 16 for monster in engine.state.monsters))

    def test_transient_encounter_is_implemented_and_attack_scales(self):
        engine = CombatEngine(
            encounter_name="Transient",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        self.assertEqual(monster.monster_id, "Transient")
        self.assertEqual(_power_amount(monster, "Fading"), 5)
        start_action = next(action for action in engine.legal_actions() if action.get("kind") == "end")
        engine.step(start_action)
        self.assertEqual(monster.meta.get("attack_index"), 1)
        self.assertEqual(_power_amount(monster, "Fading"), 4)
        self.assertLess(engine.player.current_hp, 80)

    def test_transient_attack_uses_adjusted_intent_damage(self):
        engine = CombatEngine(
            encounter_name="Transient",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=30, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.move_adjusted_damage = 18

        engine.step(next(action for action in engine.legal_actions() if action.get("kind") == "end"))

        self.assertEqual(engine.player.current_hp, 12)
        self.assertEqual(monster.meta.get("attack_index"), 1)

    def test_maw_encounter_is_implemented_and_roar_applies_debuffs(self):
        engine = CombatEngine(
            encounter_name="Maw",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        self.assertEqual(monster.monster_id, "Maw")
        self.assertEqual(monster.name, "The Maw")
        self.assertEqual(monster.next_move, "ROAR")
        self.assertEqual(engine.randoms.stream("ai").counter, 1)
        start_action = next(action for action in engine.legal_actions() if action.get("kind") == "end")
        engine.step(start_action)
        self.assertGreater(_power_amount(engine.player, "Weakened"), 0)
        self.assertGreater(_power_amount(engine.player, "Frail"), 0)

    def test_maw_attacks_respect_strength(self):
        engine = CombatEngine(
            encounter_name="Maw",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.next_move = "SLAM"
        combat_engine_mod._add_power(monster, "Strength", 3)

        engine._maw_take_turn(monster)

        self.assertEqual(engine.player.current_hp, 52)

    def test_maw_nom_attacks_respect_strength_per_hit(self):
        engine = CombatEngine(
            encounter_name="Maw",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.next_move = "NOM"
        monster.meta["turn_count"] = 6
        combat_engine_mod._add_power(monster, "Strength", 3)

        engine._maw_take_turn(monster)

        self.assertEqual(engine.player.current_hp, 56)

    def test_spire_growth_encounter_is_implemented_and_constrict_ticks(self):
        engine = CombatEngine(
            encounter_name="Spire Growth",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        self.assertEqual(monster.monster_id, "Serpent")
        self.assertIn(monster.next_move, {"QUICK_TACKLE", "CONSTRICT", "SMASH"})
        monster.next_move = "CONSTRICT"
        engine._spire_growth_take_turn(monster)
        self.assertEqual(_power_amount(engine.player, "Constricted"), 10)
        hp_before = engine.player.current_hp
        engine._apply_player_end_of_turn_powers()
        self.assertEqual(engine.player.current_hp, hp_before - 10)

    def test_writhing_mass_encounter_is_implemented_and_megadebuff_adds_parasite(self):
        combat = NativeCombatEnv(
            seed=2,
            character="IRONCLAD",
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[make_card("Strike_R", uuid="strike")],
            encounter_name="Writhing Mass",
        )
        monster = combat.engine.state.monsters[0]
        self.assertEqual(monster.monster_id, "WrithingMass")
        self.assertEqual(
            [(power["power_id"], power["amount"]) for power in monster.powers],
            [("Malleable", 3), ("Compulsive", -1)],
        )
        combat_engine_mod._add_power(monster, "Strength", -3)
        self.assertEqual(
            [(power["power_id"], power["amount"]) for power in monster.powers],
            [("Malleable", 3), ("Strength", -3), ("Compulsive", -1)],
        )
        monster.next_move = "MEGA_DEBUFF"
        monster.meta["used_mega_debuff"] = False
        combat.step({"kind": "end"})
        self.assertTrue(any(card.get("card_id") == "Parasite" for card in combat.master_deck))

    def test_writhing_mass_attacks_respect_negative_strength(self):
        engine = CombatEngine(
            encounter_name="Writhing Mass",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=50, max_hp=80, block=6),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        engine.player.block = 6
        monster.next_move = "ATTACK_DEBUFF"
        combat_engine_mod._set_power_amount(monster, "Strength", -3)

        engine._writhing_mass_take_turn(monster)

        self.assertEqual(engine.player.current_hp, 49)
        self.assertEqual(engine.player.block, 0)

    def test_writhing_mass_compulsive_reroll_uses_current_selected_move(self):
        class FixedAiRandoms:
            def stream(self, name):
                if name != "ai":
                    raise AssertionError(f"unexpected stream {name!r}")
                return self

            def random(self, *args):
                if args != (99,):
                    raise AssertionError(f"unexpected random args {args!r}")
                return 33

            def random_boolean(self, chance=None):
                raise AssertionError("reroll branch should not be reached")

        monster = MonsterState(
            monster_id="WrithingMass",
            name="Writhing Mass",
            current_hp=93,
            max_hp=175,
            next_move="ATTACK_BLOCK",
            move_history=["MULTI_HIT", "MULTI_HIT", "ATTACK_BLOCK", "ATTACK_DEBUFF"],
            meta={"first_move": False, "used_mega_debuff": False},
        )

        combat_engine_mod._writhing_mass_roll_next_move(monster, FixedAiRandoms(), include_current_move=True)

        self.assertEqual(monster.next_move, "ATTACK_DEBUFF")

    def test_writhing_mass_does_not_repeat_multi_hit(self):
        class FixedAiRandoms:
            def stream(self, name):
                if name != "ai":
                    raise AssertionError(f"unexpected stream {name!r}")
                return self

            def random(self, *args):
                if args != (99,):
                    raise AssertionError(f"unexpected random args {args!r}")
                return 50

            def random_boolean(self, chance=None):
                self.last_chance = chance
                return True

        randoms = FixedAiRandoms()
        monster = MonsterState(
            monster_id="WrithingMass",
            name="Writhing Mass",
            current_hp=93,
            max_hp=175,
            next_move="MULTI_HIT",
            move_history=["ATTACK_BLOCK", "MULTI_HIT"],
            meta={"first_move": False, "used_mega_debuff": False},
        )

        combat_engine_mod._writhing_mass_roll_next_move(monster, randoms)

        self.assertEqual(monster.next_move, "ATTACK_BLOCK")
        self.assertEqual(randoms.last_chance, 0.3)

    def test_slow_power_scales_next_player_attack_and_increments_after_card_use(self):
        engine = CombatEngine(
            encounter_name="Giant Head",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[make_card("Strike_R", uuid="slow-strike-1")],
        )
        monster = engine.state.monsters[0]
        first = next(action for action in engine.legal_actions() if "card_index" in action)
        engine.step(first)
        self.assertEqual(next(power for power in monster.powers if str(power.get("power_id")) == "Slow")["amount"], 1)
        hp_after_first = monster.current_hp
        damage = engine._scale_player_attack_damage(10, monster, attacker_weak=False)
        engine._player_deal_damage(monster, damage)
        self.assertEqual(hp_after_first - monster.current_hp, 11)

    def test_slow_and_player_weak_share_final_damage_rounding(self):
        engine = CombatEngine(
            encounter_name="Giant Head",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        combat_engine_mod._set_power_amount(monster, "Slow", 2)

        damage = engine._scale_player_attack_damage(6, monster, attacker_weak=True)
        dealt = engine._player_deal_damage(monster, damage)

        self.assertEqual(damage, 5)
        self.assertEqual(dealt, 5)

    def test_giant_head_attacks_respect_negative_strength(self):
        engine = CombatEngine(
            encounter_name="Giant Head",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80, block=17),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        engine.player.block = 17
        monster.next_move = "COUNT"
        combat_engine_mod._set_power_amount(monster, "Strength", -3)

        engine._giant_head_take_turn(monster)

        self.assertEqual(engine.player.block, 7)
        self.assertEqual(engine.player.current_hp, 80)

    def test_initialize_boss_relic_pool_shuffles_then_excludes_owned(self):
        randoms = NativeRandomSet(seed=2)
        pool = initialize_boss_relic_pool(
            randoms,
            owned_relic_ids={"Black Blood", "Philosopher's Stone"},
        )
        self.assertNotIn("Black Blood", pool)
        self.assertNotIn("Philosopher's Stone", pool)
        self.assertGreater(len(pool), 3)

    def test_draw_boss_relic_choices_consumes_pool_front(self):
        pool = ["Astrolabe", "Black Star", "Calling Bell", "Coffee Dripper"]
        choices = draw_boss_relic_choices(pool)
        self.assertEqual([choice["relic_id"] for choice in choices], ["Astrolabe", "Black Star", "Calling Bell"])
        self.assertEqual(pool, ["Coffee Dripper"])

    def test_black_blood_replaces_burning_blood_on_boss_pick(self):
        relics = [make_relic("Burning Blood"), make_relic("Anchor")]
        updated = apply_boss_relic_choice(relics, make_relic("Black Blood"))
        self.assertEqual(
            [str(relic["relic_id"]) for relic in updated],
            ["Black Blood", "Anchor"],
        )

    def test_run_env_syncs_deck_from_combat_before_reward_resolution(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "COMBAT"
        env.current_room_type = "MonsterRoom"

        class _VictoryCombat:
            def __init__(self, player, deck, gold):
                self.player = player
                self.master_deck = list(deck)
                self.gold = gold
                self.encounter_name = "Cultist"
                self.engine = None

            def step(self, _action):
                return "VICTORY"

        env.combat = _VictoryCombat(
            env.player,
            [make_card("Strike_R", uuid="strike"), make_card("Parasite", uuid="parasite")],
            env.gold,
        )
        env._step_combat({"kind": "end"})
        self.assertTrue(any(card.get("card_id") == "Parasite" for card in env.deck))

    def test_boss_relic_skip_still_advances_act(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "BOSS_RELIC"
        env.act = 1
        env.boss_relic_options = [make_relic("Astrolabe"), make_relic("Black Star"), make_relic("Calling Bell")]
        env._step_boss_relic({"kind": "skip", "choice_index": 3})
        self.assertEqual(env.act, 2)
        self.assertEqual(env.phase, "MAP")

    def test_tiny_house_boss_pick_opens_reward_screen_and_advances_after_skip(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "BOSS_RELIC"
        env.act = 1
        env.current_room_type = "TreasureRoomBoss"
        env.deck = [make_card("Strike_R", uuid="strike"), make_card("Defend_R", uuid="defend")]
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.boss_relic_options = [make_relic("Tiny House"), make_relic("Black Star"), make_relic("Calling Bell")]

        env._step_boss_relic({"kind": "boss_relic", "choice_index": 0})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(env.reward_gold, 50)
        self.assertEqual(len(env.reward_potions), 1)
        self.assertEqual(len(env.reward_cards), 3)
        self.assertEqual(env.player.max_hp, 85)
        self.assertEqual(env.player.current_hp, 45)
        self.assertTrue(any(int(card.get("upgrades") or 0) == 1 for card in env.deck))

        env.step({"kind": "reward_gold", "choice_index": 0})
        env.step({"kind": "reward_potion", "potion_id": env.reward_potions[0]["potion_id"], "choice_index": 0})
        env.step({"kind": "card_reward", "card": dict(env.reward_cards[0]), "card_index": 0, "choice_index": 0})
        self.assertEqual(env.phase, "CARD_REWARD")
        env.step({"kind": "proceed", "choice_index": 0})
        self.assertEqual(env.act, 2)
        self.assertEqual(env.phase, "MAP")
        self.assertIsNone(env.current_map_node)

    def test_tiny_house_max_hp_heal_obeys_mark_and_preserves_rewards(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "BOSS_RELIC"
        env.act = 1
        env.current_room_type = "TreasureRoomBoss"
        env.deck = [make_card("Strike_R", uuid="strike"), make_card("Defend_R", uuid="defend")]
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.relics = [make_relic("Mark of the Bloom")]
        env.boss_relic_options = [make_relic("Tiny House"), make_relic("Black Star"), make_relic("Calling Bell")]

        env._step_boss_relic({"kind": "boss_relic", "choice_index": 0})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(env.reward_gold, 50)
        self.assertEqual(len(env.reward_potions), 1)
        self.assertEqual(len(env.reward_cards), 3)
        self.assertEqual(env.player.max_hp, 85)
        self.assertEqual(env.player.current_hp, 40)
        self.assertTrue(any(int(card.get("upgrades") or 0) == 1 for card in env.deck))

    def test_neow_boss_relic_tiny_house_returns_to_neow_continue_after_rewards(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=True, start_on_map=False)
        env.phase = "NEOW"
        env.current_room_type = "NeowRoom"
        env.deck = [make_card("Strike_R", uuid="strike"), make_card("Defend_R", uuid="defend")]
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.relic_pools["BOSS"] = ["Tiny House"]

        env._step_neow(
            {
                "kind": "neow",
                "choice_index": 0,
                "bonus": "BOSS_RELIC",
                "drawback": "NONE",
            }
        )

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(env.current_room_type, "NeowRoom")
        self.assertEqual(env.reward_gold, 50)
        self.assertEqual(len(env.reward_potions), 1)
        self.assertEqual(env.player.max_hp, 85)
        self.assertEqual(env.player.current_hp, 45)

        env.step({"kind": "reward_gold", "choice_index": 0})
        env.step({"kind": "reward_potion", "potion_id": env.reward_potions[0]["potion_id"], "choice_index": 0})
        env.step({"kind": "skip", "choice_index": 0})

        self.assertEqual(env.phase, "NEOW")
        self.assertTrue(env.neow_pending_continue)
        env.step({"kind": "neow", "choice_index": 0})
        self.assertEqual(env.phase, "MAP")

    def test_astrolabe_boss_pick_opens_transform_select_and_advances_act(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "BOSS_RELIC"
        env.act = 1
        env.current_room_type = "TreasureRoomBoss"
        env.deck = [make_card("Strike_R", uuid="s1"), make_card("Defend_R", uuid="d1"), make_card("Bash", uuid="b1"), make_card("Anger", uuid="a1")]
        env.boss_relic_options = [make_relic("Astrolabe")]

        env._step_boss_relic({"kind": "boss_relic", "choice_index": 0})

        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertEqual(env.current_card_select["mode"], "transform")
        self.assertEqual(env.current_card_select["remaining_picks"], 3)

        while env.phase == "CARD_SELECT":
            env.step(env.legal_actions()[0])

        self.assertTrue(env.boss_relic_pending_act_advance)
        self.assertEqual(env.phase, "TREASURE")
        env.step(env.legal_actions()[0])

        self.assertEqual(env.act, 2)
        self.assertEqual(env.phase, "MAP")

    def test_empty_cage_boss_pick_opens_remove_select_and_advances_act(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "BOSS_RELIC"
        env.act = 1
        env.current_room_type = "TreasureRoomBoss"
        env.deck = [make_card("Strike_R", uuid="s1"), make_card("Defend_R", uuid="d1"), make_card("Bash", uuid="b1"), make_card("Anger", uuid="a1")]
        env.boss_relic_options = [make_relic("Empty Cage")]

        env._step_boss_relic({"kind": "boss_relic", "choice_index": 0})

        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertEqual(env.current_card_select["mode"], "remove")
        self.assertEqual(env.current_card_select["remaining_picks"], 2)

        while env.phase == "CARD_SELECT":
            env.step(env.legal_actions()[0])

        self.assertTrue(env.boss_relic_pending_act_advance)
        self.assertEqual(env.phase, "TREASURE")
        env.step(env.legal_actions()[0])

        self.assertEqual(env.act, 2)
        self.assertEqual(env.phase, "MAP")
        self.assertEqual(len(env.deck), 2)

    def test_calling_bell_boss_pick_adds_curse_and_opens_three_relic_rewards(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "BOSS_RELIC"
        env.act = 1
        env.current_room_type = "TreasureRoomBoss"
        env.boss_relic_options = [make_relic("Calling Bell")]
        env.relic_pools["COMMON"] = ["Anchor"]
        env.relic_pools["UNCOMMON"] = ["HornCleat"]
        env.relic_pools["RARE"] = ["Calipers"]

        env._step_boss_relic({"kind": "boss_relic", "choice_index": 0})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertIn("CurseOfTheBell", [card["card_id"] for card in env.deck])
        self.assertEqual([relic["relic_id"] for relic in env.reward_relics], ["Anchor", "HornCleat", "Calipers"])

        for relic in list(env.reward_relics):
            env.step({"kind": "reward_relic", "relic_id": relic["relic_id"], "choice_index": 0})
        env.step({"kind": "skip", "choice_index": 0})
        self.assertEqual(env.act, 2)
        self.assertEqual(env.phase, "MAP")

    def test_orrery_purchase_opens_four_card_reward_groups_and_returns_to_shop(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "SHOP"
        env.current_room_type = "ShopRoom"
        env.gold = 999
        env.current_shop = generate_shop(NativeRandomSet(seed=3))
        env.current_shop.relics = [{**make_relic("Orrery"), "price": 150}]

        env.step({"kind": "shop", "item_kind": "relic", "shop_index": 0})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertTrue(env.reward_cards)
        self.assertEqual(len(env.reward_card_groups), 3)

        while env.phase == "CARD_REWARD":
            if env.reward_cards:
                env.step({"kind": "skip", "choice_index": 0})
            else:
                break

        self.assertEqual(env.phase, "SHOP")
        self.assertEqual(env.current_room_type, "ShopRoom")

    def test_pandoras_box_boss_pick_replaces_starter_strikes_and_defends(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "BOSS_RELIC"
        env.act = 1
        env.current_room_type = "TreasureRoomBoss"
        env.deck = [
            make_card("Strike_R", uuid="s1"),
            make_card("Strike_R", uuid="s2"),
            make_card("Defend_R", uuid="d1"),
            make_card("Defend_R", uuid="d2"),
            make_card("Bash", uuid="b1"),
        ]
        env.boss_relic_options = [make_relic("Pandora's Box")]

        env._step_boss_relic({"kind": "boss_relic", "choice_index": 0})

        self.assertEqual(env.act, 1)
        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertEqual(env.current_card_select["mode"], "pandora_confirm")
        self.assertTrue(env._card_select_screen_state()["confirm_up"])
        self.assertEqual(sum(1 for card in env.deck if card["card_id"] in {"Strike_R", "Defend_R"}), 0)
        self.assertEqual(len(env.deck), 1)

        env.step(env.legal_actions()[0])

        self.assertTrue(env.boss_relic_pending_act_advance)
        self.assertEqual(env.phase, "TREASURE")
        env.step(env.legal_actions()[0])

        self.assertEqual(env.act, 2)
        self.assertEqual(env.phase, "MAP")
        self.assertEqual(sum(1 for card in env.deck if card["card_id"] in {"Strike_R", "Defend_R"}), 0)
        self.assertEqual(len(env.deck), 5)

    def test_cauldron_shop_purchase_opens_potion_reward_and_returns_to_map(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "SHOP"
        env.current_room_type = "ShopRoom"
        env.gold = 999
        env.current_shop = generate_shop(NativeRandomSet(seed=3))
        env.current_shop.relics = [
            {
                **make_relic("Cauldron"),
                "price": 150,
            }
        ]

        env.step({"kind": "shop", "item_kind": "relic", "shop_index": 0})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(len(env.reward_potions), 5)
        self.assertTrue(any(str(relic.get("relic_id")) == "Cauldron" for relic in env.relics))

        env.step({"kind": "skip", "choice_index": 0})
        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.current_room_type, "ShopRoom")

    def test_immediate_hp_gain_relics_increase_max_and_current_hp(self):
        for relic_id, amount in [("Strawberry", 7), ("Pear", 10), ("Mango", 14)]:
            with self.subTest(relic_id=relic_id):
                env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
                env.player.current_hp = 40
                env.player.max_hp = 80

                env._obtain_relic(make_relic(relic_id), source="reward_relic")

                self.assertEqual(env.player.max_hp, 80 + amount)
                self.assertEqual(env.player.current_hp, 40 + amount)

    def test_immediate_hp_gain_relics_max_hp_heal_obeys_mark_of_the_bloom(self):
        for relic_id, amount in [("Strawberry", 7), ("Pear", 10), ("Mango", 14)]:
            with self.subTest(relic_id=relic_id):
                env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
                env.player.current_hp = 40
                env.player.max_hp = 80
                env.relics = [make_relic("Mark of the Bloom")]

                env._obtain_relic(make_relic(relic_id), source="reward_relic")

                self.assertEqual(env.player.max_hp, 80 + amount)
                self.assertEqual(env.player.current_hp, 40)

    def test_waffle_heals_to_full_but_respects_mark_of_the_bloom(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.player.current_hp = 17
        env.player.max_hp = 80
        env._obtain_relic(make_relic("Lee's Waffle"), source="shop")
        self.assertEqual(env.player.max_hp, 87)
        self.assertEqual(env.player.current_hp, 87)

        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.player.current_hp = 17
        env.player.max_hp = 80
        env.relics.append(make_relic("Mark of the Bloom"))
        env._obtain_relic(make_relic("Lee's Waffle"), source="shop")
        self.assertEqual(env.player.max_hp, 87)
        self.assertEqual(env.player.current_hp, 17)

    def test_old_coin_and_potion_belt_apply_immediate_run_side_effects(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.gold = 99
        env.max_potion_slots = 3

        env._obtain_relic(make_relic("Old Coin"), source="reward_relic")
        env._obtain_relic(make_relic("Potion Belt"), source="reward_relic")

        self.assertEqual(env.gold, 399)
        self.assertEqual(env.max_potion_slots, 5)

    def test_necronomicon_obtain_adds_necronomicurse_and_respects_omamori(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        initial_deck_size = len(env.deck)

        env._obtain_relic(make_relic("Necronomicon"), source="reward_relic")

        self.assertEqual(len(env.deck), initial_deck_size + 1)
        self.assertTrue(any(card.get("card_id") == "Necronomicurse" for card in env.deck))

        blocked_env = NativeRunEnv(seed=4, ascension_level=0, enable_neow=False, start_on_map=True)
        blocked_env.relics.append(make_relic("Omamori", counter=1))
        blocked_initial_deck_size = len(blocked_env.deck)

        blocked_env._obtain_relic(make_relic("Necronomicon"), source="reward_relic")

        self.assertEqual(len(blocked_env.deck), blocked_initial_deck_size)
        self.assertFalse(any(card.get("card_id") == "Necronomicurse" for card in blocked_env.deck))
        omamori = next(relic for relic in blocked_env.relics if str(relic.get("relic_id") or relic.get("id")) == "Omamori")
        self.assertEqual(int(omamori.get("counter") or 0), 0)
        self.assertTrue(bool(omamori.get("used_up")))

    def test_whetstone_and_war_paint_upgrade_two_cards_of_matching_type(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.deck = [
            make_card("Strike_R", uuid="a1"),
            make_card("Twin Strike", uuid="a2"),
            make_card("Defend_R", uuid="s1"),
            make_card("Shrug It Off", uuid="s2"),
            make_card("Bash", uuid="a3"),
        ]

        env._obtain_relic(make_relic("Whetstone"), source="reward_relic")
        attack_upgrades = [int(card.get("upgrades") or 0) for card in env.deck if str(card.get("type")) == "ATTACK"]
        self.assertEqual(sum(1 for amount in attack_upgrades if amount > 0), 2)

        env._obtain_relic(make_relic("War Paint"), source="reward_relic")
        skill_upgrades = [int(card.get("upgrades") or 0) for card in env.deck if str(card.get("type")) == "SKILL"]
        self.assertEqual(sum(1 for amount in skill_upgrades if amount > 0), 2)

    def test_du_vu_doll_tracks_curse_count_from_deck(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.deck = [make_card("Strike_R"), make_card("Regret"), make_card("Shame")]

        env._obtain_relic(make_relic("Du-Vu Doll"), source="reward_relic")
        self.assertEqual(next(relic for relic in env.relics if str(relic.get("relic_id")) == "Du-Vu Doll")["counter"], 2)

        env.deck.pop()
        env._sync_state()
        self.assertEqual(next(relic for relic in env.relics if str(relic.get("relic_id")) == "Du-Vu Doll")["counter"], 1)

    def test_energy_boss_relics_raise_base_combat_energy(self):
        energy_relic_ids = [
            "Busted Crown",
            "Coffee Dripper",
            "Cursed Key",
            "Ectoplasm",
            "Fusion Hammer",
            "Philosopher's Stone",
            "Runic Dome",
            "Sozu",
            "Velvet Choker",
        ]
        for relic_id in energy_relic_ids:
            with self.subTest(relic_id=relic_id):
                engine = CombatEngine(
                    encounter_name="Cultist",
                    randoms=NativeRandomSet(seed=2),
                    ascension_level=0,
                    player=PlayerState(current_hp=80, max_hp=80),
                    master_deck=[make_card("Strike_R", uuid="strike") for _ in range(10)],
                    relics=[make_relic(relic_id)],
                )
                self.assertEqual(engine.player.energy, 4)
                engine._start_player_turn()
                self.assertEqual(engine.player.energy, 4)

        stacked_engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[make_card("Strike_R", uuid="stacked-strike") for _ in range(10)],
            relics=[make_relic("Sozu"), make_relic("Coffee Dripper")],
        )
        self.assertEqual(stacked_engine.player.energy, 5)
        stacked_engine._start_player_turn()
        self.assertEqual(stacked_engine.player.energy, 5)

    def test_snecko_eye_applies_confusion_and_draws_seven_cards(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
            relics=[make_relic("Snecko Eye")],
        )
        self.assertEqual(_power_amount(engine.player, "Confusion"), -1)
        self.assertEqual(engine.player.energy, 3)
        self.assertEqual(len(engine.state.hand), 7)
        self.assertTrue(all(0 <= int(card.get("cost_for_turn", -1)) <= 3 for card in engine.state.hand))

    def test_philosophers_stone_adds_strength_to_existing_and_spawned_monsters(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            relics=[make_relic("Philosopher's Stone")],
        )
        self.assertTrue(all(_power_amount(monster, "Strength") == 1 for monster in engine.state.monsters))

        engine = CombatEngine(
            encounter_name="Gremlin Leader",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            relics=[make_relic("Philosopher's Stone")],
        )
        leader = next(monster for monster in engine.state.monsters if monster.monster_id == "GremlinLeader")
        for monster in engine.state.monsters:
            if monster is not leader:
                monster.current_hp = 0
        leader.next_move = "RALLY"
        engine._gremlin_leader_take_turn(leader)
        spawned_minions = [monster for monster in engine.state.monsters if monster is not leader and monster.current_hp > 0]
        self.assertTrue(spawned_minions)
        self.assertTrue(all(_power_amount(monster, "Strength") >= 1 for monster in spawned_minions))

    def test_louse_opening_moves_are_rolled_from_ai_rng(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )

        self.assertEqual(
            [(monster.monster_id, monster.current_hp, monster.next_move) for monster in engine.state.monsters],
            [
                ("FuzzyLouseNormal", 11, "STRENGTHEN"),
                ("FuzzyLouseDefensive", 15, "BITE"),
            ],
        )
        self.assertTrue(all(_power_amount(monster, "CurlUp") > 0 for monster in engine.state.monsters))

    def test_louse_defensive_weaken_intent_keeps_single_hit_metadata(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        defensive = next(monster for monster in engine.state.monsters if monster.monster_id == "FuzzyLouseDefensive")
        defensive.next_move = "WEAKEN"

        engine._update_monster_intents()

        self.assertEqual(defensive.intent, "DEBUFF")
        self.assertEqual(defensive.move_adjusted_damage, -1)
        self.assertEqual(defensive.move_hits, 1)

    def test_louse_normal_bite_uses_current_strength_scaled_damage(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        normal = next(monster for monster in engine.state.monsters if monster.monster_id == "FuzzyLouseNormal")
        normal.next_move = "BITE"
        normal.move_history = []
        normal.powers = [{"power_id": "Strength", "id": "Strength", "name": "Strength", "amount": 3, "misc": 3}]

        engine._update_monster_intents()
        expected_damage = int(normal.move_adjusted_damage or 0)
        engine._louse_normal_take_turn(normal)

        self.assertEqual(engine.player.current_hp, 80 - expected_damage)

    def test_louse_bite_does_not_apply_player_vulnerable_twice(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        normal = next(monster for monster in engine.state.monsters if monster.monster_id == "FuzzyLouseNormal")
        normal.next_move = "BITE"
        engine.player.block = 3
        engine.player.powers.append({"power_id": "Vulnerable", "id": "Vulnerable", "name": "Vulnerable", "amount": 2, "misc": 2})

        engine._update_monster_intents()
        expected_damage = int(normal.move_adjusted_damage or 0)
        engine._louse_normal_take_turn(normal)

        self.assertEqual(engine.player.current_hp, 80 - max(0, expected_damage - 3))
        self.assertEqual(engine.player.block, 0)

    def test_curl_up_triggers_on_nonlethal_attack_and_is_removed(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        defensive = next(monster for monster in engine.state.monsters if monster.monster_id == "FuzzyLouseDefensive")
        curl_up = _power_amount(defensive, "Curl Up")

        dealt = engine._player_deal_damage(defensive, 8)

        self.assertEqual(dealt, 8)
        self.assertEqual(defensive.current_hp, 7)
        self.assertEqual(defensive.block, curl_up)
        self.assertEqual(_power_amount(defensive, "Curl Up"), 0)
        self.assertEqual(defensive.meta.get("state"), "CLOSED")

    def test_velvet_choker_locks_card_plays_after_six_cards(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            relics=[make_relic("Velvet Choker")],
        )
        engine.state.hand = [make_card("Strike_R", uuid=f"strike-{i}") for i in range(3)]
        engine.player.energy = 10
        engine.cards_played_this_turn = 6
        actions = engine.legal_actions()
        self.assertEqual(actions, [{"kind": "end", "name": "END_TURN", "action_index": 0}])

    def test_velvet_choker_tracks_card_counter_and_resets_each_turn(self):
        relic = make_relic("Velvet Choker")
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            relics=[relic],
        )
        engine.state.hand = [make_card("Strike_R", uuid="strike")]
        engine.player.energy = 10

        engine.step({"kind": "card", "card_index": 0, "target_index": 0})
        self.assertEqual(relic["counter"], 1)

        relic["counter"] = 6
        engine.state.hand = [make_card("Strike_R", uuid="locked")]
        actions = engine.legal_actions()
        self.assertEqual(actions, [{"kind": "end", "name": "END_TURN", "action_index": 0}])

        engine._start_player_turn()
        self.assertEqual(relic["counter"], 0)

    def test_cursed_key_adds_curse_when_opening_non_boss_treasure(self):
        env = NativeRunEnv(seed=4, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "TREASURE"
        env.current_room_type = "TreasureRoom"
        env.current_treasure = generate_treasure(env.randoms)
        env.relics.append(make_relic("Cursed Key"))
        before_deck = len(env.deck)

        env._step_treasure({"kind": "treasure", "name": "OPEN_CHEST"})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(len(env.deck), before_deck)
        self.assertIsNotNone(env.pending_cursed_key_chest_curse)

        if env.reward_gold is not None:
            env._step_card_reward({"kind": "reward_gold", "choice_index": 0})
        else:
            env._step_card_reward({"kind": "reward_relic", "choice_index": 0, "relic_id": env.reward_relics[0]["relic_id"]})

        self.assertEqual(len(env.deck), before_deck + 1)
        self.assertIn(env.deck[-1]["card_id"], {"Clumsy", "Decay", "Doubt", "Injury", "Normality", "Pain", "Parasite", "Regret", "Shame", "Writhe"})
        self.assertIsNone(env.pending_cursed_key_chest_curse)

    def test_cursed_key_chest_curse_consumes_omamori_on_open(self):
        env = NativeRunEnv(seed=4, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "TREASURE"
        env.current_room_type = "TreasureRoom"
        env.current_treasure = generate_treasure(env.randoms)
        env.relics.extend([make_relic("Cursed Key"), make_relic("Omamori")])
        before_deck = len(env.deck)

        env._step_treasure({"kind": "treasure", "name": "OPEN_CHEST"})

        omamori = next(relic for relic in env.relics if str(relic.get("relic_id") or relic.get("id")) == "Omamori")
        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(len(env.deck), before_deck)
        self.assertEqual(omamori.get("counter"), 1)
        self.assertIsNone(env.pending_cursed_key_chest_curse)

    def test_runic_dome_hides_serialized_monster_intents(self):
        combat = NativeCombatEnv(
            seed=2,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Runic Dome")],
        )
        payload = combat.serialize()
        for monster in payload["combat_state"]["monsters"]:
            self.assertEqual(monster["intent"], "UNKNOWN")
            self.assertEqual(monster["move_adjusted_damage"], 0)
            self.assertEqual(monster["move_hits"], 0)

    def test_slavers_collar_grants_extra_energy_only_in_elite_or_boss_rooms(self):
        elite_combat = NativeCombatEnv(
            seed=2,
            ascension_level=0,
            encounter_name="Gremlin Nob",
            room_type="MonsterRoomElite",
            relics=[make_relic("SlaversCollar")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        self.assertEqual(elite_combat.player.energy, 4)
        elite_combat.engine._start_player_turn()
        self.assertEqual(elite_combat.player.energy, 4)

        normal_combat = NativeCombatEnv(
            seed=2,
            ascension_level=0,
            encounter_name="Cultist",
            room_type="MonsterRoom",
            relics=[make_relic("SlaversCollar")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        self.assertEqual(normal_combat.player.energy, 3)
        normal_combat.engine._start_player_turn()
        self.assertEqual(normal_combat.player.energy, 3)

        event_elite_combat = NativeCombatEnv(
            seed=2,
            ascension_level=0,
            encounter_name="Gremlin Nob",
            room_type="EventRoom",
            relics=[make_relic("SlaversCollar")],
            elite_trigger=True,
            master_deck=[make_card("Strike_R", uuid=f"event-elite-strike-{i}") for i in range(10)],
        )
        self.assertEqual(event_elite_combat.player.energy, 4)
        event_elite_combat.engine._start_player_turn()
        self.assertEqual(event_elite_combat.player.energy, 4)

        event_normal_combat = NativeCombatEnv(
            seed=2,
            ascension_level=0,
            encounter_name="Cultist",
            room_type="EventRoom",
            relics=[make_relic("SlaversCollar")],
            master_deck=[make_card("Strike_R", uuid=f"event-normal-strike-{i}") for i in range(10)],
        )
        self.assertEqual(event_normal_combat.player.energy, 3)

    def test_gremlin_nob_anger_strength_sorts_before_existing_weak(self):
        combat = NativeCombatEnv(
            seed=2,
            ascension_level=0,
            encounter_name="Gremlin Nob",
            room_type="MonsterRoomElite",
            master_deck=[make_card("Defend_R", uuid="defend"), make_card("Strike_R", uuid="strike")],
        )
        nob = combat.engine.state.monsters[0]
        combat_engine_mod._append_power(nob, "Anger", 2, misc=2)
        combat_engine_mod._add_power(nob, "Weak", 1)
        combat.engine.state.hand = [make_card("Defend_R", uuid="defend")]
        combat.engine.player.energy = 3

        action = next(candidate for candidate in combat.engine.legal_actions() if candidate.get("card_id") == "Defend_R")
        combat.engine.step(action)

        self.assertEqual(
            [(power["power_id"], power["amount"]) for power in nob.powers],
            [("Anger", 2), ("Strength", 2), ("Weakened", 1)],
        )

    def test_mark_of_pain_adds_two_wounds_to_combat_piles(self):
        combat = NativeCombatEnv(
            seed=2,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Mark of Pain")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(5)],
        )
        self.assertEqual(combat.player.energy, 4)
        self.assertEqual(sum(1 for card in combat.engine.state.hand if card["card_id"] == "Wound"), 0)
        self.assertEqual(sum(1 for card in combat.engine.state.draw_pile if card["card_id"] == "Wound"), 2)

    def test_happy_flower_and_incense_burner_turn_counters_apply_effects(self):
        combat = NativeCombatEnv(
            seed=2,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Happy Flower"), make_relic("Incense Burner")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        happy = next(relic for relic in combat.engine.relics if str(relic.get("relic_id")) == "Happy Flower")
        burner = next(relic for relic in combat.engine.relics if str(relic.get("relic_id")) == "Incense Burner")
        self.assertEqual(happy.get("counter"), 1)
        self.assertEqual(burner.get("counter"), 1)

        combat.engine._start_player_turn()
        self.assertEqual(combat.engine.player.energy, 3)
        self.assertEqual(happy.get("counter"), 2)

        combat.engine._start_player_turn()
        self.assertEqual(combat.engine.player.energy, 4)
        self.assertEqual(happy.get("counter"), 0)

        for _ in range(2):
            combat.engine._start_player_turn()
        self.assertEqual(_power_amount(combat.engine.player, "Intangible"), 0)
        combat.engine._start_player_turn()
        self.assertEqual(_power_amount(combat.engine.player, "Intangible"), 1)
        self.assertEqual(burner.get("counter"), 0)
        self.assertEqual(burner.get("counter"), 0)

    def test_ectoplasm_blocks_gold_gain_from_relic_reward_and_event(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True)
        env.gold = 99
        env.relics.append(make_relic("Ectoplasm"))

        env._obtain_relic(make_relic("Old Coin"), source="reward_relic")
        self.assertEqual(env.gold, 99)

        env.phase = "CARD_REWARD"
        env.current_room_type = "MonsterRoom"
        env.reward_gold = 50
        env.step({"kind": "reward_gold", "choice_index": 0})
        self.assertEqual(env.gold, 99)

        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("Golden Shrine")
        env.step({"kind": "event", "event_id": "Golden Shrine", "choice_index": 1})
        self.assertEqual(env.gold, 99)
        self.assertIn("Regret", [card["card_id"] for card in env.deck])

    def test_basic_battle_start_relics_apply_source_semantics(self):
        combat = NativeCombatEnv(
            seed=4,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[
                make_relic("Anchor"),
                make_relic("Bag of Preparation"),
                make_relic("Lantern"),
                make_relic("Bag of Marbles"),
                make_relic("Vajra"),
                make_relic("Oddly Smooth Stone"),
            ],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        self.assertEqual(combat.engine.player.block, 10)
        self.assertEqual(combat.engine.player.energy, 4)
        self.assertEqual(len(combat.engine.state.hand), 7)
        self.assertEqual(_power_amount(combat.engine.player, "Strength"), 1)
        self.assertEqual(_power_amount(combat.engine.player, "Dexterity"), 1)
        self.assertEqual(_power_amount(combat.engine.state.monsters[0], "Vulnerable"), 1)

    def test_bag_of_marbles_respects_monster_artifact(self):
        combat = NativeCombatEnv(
            seed=41,
            ascension_level=0,
            encounter_name="Spheric Guardian",
            relics=[make_relic("Bag of Marbles")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        self.assertEqual(_power_amount(monster, "Vulnerable"), 0)
        self.assertEqual(_power_amount(monster, "Artifact"), 2)

    def test_spheric_guardian_single_hit_attacks_use_strength_and_weak(self):
        for move in ("FRAIL_ATTACK", "BLOCK_ATTACK"):
            with self.subTest(move=move):
                engine = CombatEngine(
                    encounter_name="Spheric Guardian",
                    randoms=NativeRandomSet(seed=2),
                    ascension_level=0,
                    player=PlayerState(current_hp=80, max_hp=80),
                    master_deck=[],
                )
                monster = engine.state.monsters[0]
                monster.next_move = move
                monster.block = 0
                monster.powers = []
                combat_engine_mod._append_power(monster, "Strength", 3)
                combat_engine_mod._append_power(monster, "Weakened", 2)

                engine._spheric_guardian_take_turn(monster)

                self.assertEqual(engine.player.current_hp, 71)

    def test_bag_of_marbles_applies_after_louse_curl_up(self):
        combat = NativeCombatEnv(
            seed=42,
            ascension_level=0,
            encounter_name="2 Louse",
            relics=[make_relic("Bag of Marbles")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        for monster in combat.engine.state.monsters:
            power_ids = [power.get("power_id") or power.get("id") for power in monster.powers]
            self.assertEqual(power_ids[:2], ["Curl Up", "Vulnerable"])

    def test_clockwork_souvenir_grants_artifact_at_battle_start(self):
        combat = NativeCombatEnv(
            seed=42,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("ClockworkSouvenir")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        self.assertEqual(_power_amount(combat.engine.player, "Artifact"), 1)

    def test_mutagenic_strength_grants_temporary_strength_at_battle_start(self):
        combat = NativeCombatEnv(
            seed=43,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("MutagenicStrength")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        self.assertEqual(_power_amount(combat.engine.player, "Strength"), 3)
        self.assertEqual(_power_amount(combat.engine.player, "Flex"), 3)
        self.assertEqual([power["power_id"] for power in combat.engine.player.powers], ["Flex", "Strength"])
        combat.engine._apply_player_end_of_turn_powers()
        self.assertEqual(_power_amount(combat.engine.player, "Strength"), 0)
        self.assertEqual(_power_amount(combat.engine.player, "Flex"), 0)

    def test_red_mask_applies_weak_to_all_enemies_at_battle_start(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="3 Cultists",
            relics=[make_relic("Red Mask")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        self.assertTrue(all(_power_amount(monster, "Weak") == 1 for monster in combat.engine.state.monsters))

    def test_gremlin_horn_draw_from_combust_end_turn_survives_discard(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="3 Cultists",
            relics=[make_relic("Gremlin Horn")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        engine = combat.engine
        engine.state.hand = [make_card("Wound", uuid="wound"), make_card("Strike_R", uuid="strike")]
        engine.state.draw_pile = [make_card("Defend_R", uuid=f"draw-{i}") for i in range(5)]
        engine.state.draw_pile.append(make_card("Shrug It Off", uuid="gremlin-horn-draw"))
        engine.state.discard_pile = []
        engine.state.exhaust_pile = []
        engine.player.block = 99
        engine.player.energy = 0
        combat_engine_mod._append_power(engine.player, "Combust", 5, misc=1)
        for index, monster in enumerate(engine.state.monsters):
            monster.current_hp = 1 if index == 0 else 50

        engine.step({"kind": "end"})

        self.assertEqual(engine.state.hand[0]["card_id"], "Shrug It Off")
        self.assertEqual(len(engine.state.hand), 6)
        self.assertNotIn("Shrug It Off", [card["card_id"] for card in engine.state.discard_pile])

    def test_gambling_chip_opens_opening_hand_select_and_confirm_can_pick_zero(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Gambling Chip")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(5)]
            + [make_card("Defend_R", uuid=f"defend-{i}") for i in range(5)],
        )
        engine = combat.engine

        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.pending_card_select["mode"], "GAMBLING_CHIP")
        self.assertFalse(engine.pending_card_select["any_number"])
        self.assertTrue(engine.pending_card_select["can_pick_zero"])

        actions = engine.legal_actions()
        self.assertTrue(any(action["kind"] == "card_select" for action in actions))
        self.assertTrue(any(action["kind"] == "confirm" for action in actions))
        hand_before = [card["uuid"] for card in engine.state.hand]
        draw_before = [card["uuid"] for card in engine.state.draw_pile]

        engine.step(next(action for action in actions if action["kind"] == "confirm"))

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["uuid"] for card in engine.state.hand], hand_before)
        self.assertEqual([card["uuid"] for card in engine.state.draw_pile], draw_before)

    def test_gambling_chip_selected_cards_discard_then_redraw(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Gambling Chip")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(5)]
            + [make_card("Defend_R", uuid=f"defend-{i}") for i in range(6)],
        )
        engine = combat.engine
        selected_uuid = engine.state.hand[0]["uuid"]

        engine.step(next(action for action in engine.legal_actions() if action["kind"] == "card_select"))
        engine.step(next(action for action in engine.legal_actions() if action["kind"] == "confirm"))

        self.assertIsNone(engine.pending_card_select)
        self.assertIn(selected_uuid, [card["uuid"] for card in engine.state.discard_pile])
        self.assertEqual(len(engine.state.hand), 5)

    def test_gamblers_brew_exposes_non_targeted_combat_potion_action(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="Cultist",
            potions=[make_potion("GamblersBrew")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        engine = combat.engine

        actions = engine.legal_actions()
        potion_action = next(action for action in actions if action.get("potion_id") == "GamblersBrew")

        self.assertEqual(potion_action["kind"], "potion")
        self.assertEqual(potion_action["potion_index"], 0)
        self.assertFalse(potion_action["requires_target"])
        self.assertNotIn("target_index", potion_action)

    def test_gamblers_brew_consumes_potion_and_opens_gambling_chip_select(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="Cultist",
            potions=[make_potion("GamblersBrew")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        engine = combat.engine

        engine.step(next(action for action in engine.legal_actions() if action.get("potion_id") == "GamblersBrew"))

        self.assertEqual(engine.potions[0]["potion_id"], "Potion Slot")
        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.pending_card_select["mode"], "GAMBLING_CHIP")
        self.assertEqual(engine.pending_card_select["potion_id"], "GamblersBrew")
        self.assertFalse(engine.pending_card_select["any_number"])
        self.assertTrue(engine.pending_card_select["can_pick_zero"])
        self.assertTrue(any(action["kind"] == "card_select" for action in engine.legal_actions()))
        self.assertTrue(any(action["kind"] == "confirm" for action in engine.legal_actions()))

    def test_gamblers_brew_selected_cards_discard_then_redraw(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="Cultist",
            potions=[make_potion("GamblersBrew")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        engine = combat.engine
        engine.state.hand = [
            make_card("Strike_R", uuid="gamblers-discard-0"),
            make_card("Defend_R", uuid="gamblers-discard-1"),
            make_card("Bash", uuid="gamblers-keep"),
        ]
        engine.state.draw_pile = [
            make_card("Strike_R", uuid="gamblers-draw-0"),
            make_card("Defend_R", uuid="gamblers-draw-1"),
        ]
        engine.state.discard_pile = []

        engine.step(next(action for action in engine.legal_actions() if action.get("potion_id") == "GamblersBrew"))
        engine.step(next(action for action in engine.legal_actions() if action.get("target_index") == 0))
        engine.step(next(action for action in engine.legal_actions() if action.get("target_index") == 1))
        engine.step(next(action for action in engine.legal_actions() if action["kind"] == "confirm"))

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual(engine.potions[0]["potion_id"], "Potion Slot")
        self.assertEqual(len(engine.state.hand), 3)
        self.assertEqual(
            {card["uuid"] for card in engine.state.discard_pile},
            {"gamblers-discard-0", "gamblers-discard-1"},
        )
        self.assertEqual(engine.state.cards_discarded_this_turn, 2)
        self.assertIn("gamblers-keep", {card["uuid"] for card in engine.state.hand})
        self.assertIn("gamblers-draw-0", {card["uuid"] for card in engine.state.hand})
        self.assertIn("gamblers-draw-1", {card["uuid"] for card in engine.state.hand})

    def test_gamblers_brew_discards_in_selection_order_and_resets_temporary_costs(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="Cultist",
            potions=[make_potion("GamblersBrew")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        engine = combat.engine
        temp_demon_form = make_card("Demon Form", uuid="gamblers-temp-cost")
        temp_demon_form["cost_for_turn"] = 0
        temp_demon_form["_reset_cost_for_turn_after_play"] = True
        engine.state.hand = [
            make_card("Defend_R", uuid="gamblers-order-0"),
            make_card("Brutality", uuid="gamblers-order-1"),
            temp_demon_form,
        ]
        engine.state.draw_pile = [
            make_card("Strike_R", uuid="gamblers-draw-order-0"),
            make_card("Defend_R", uuid="gamblers-draw-order-1"),
            make_card("Bash", uuid="gamblers-draw-order-2"),
        ]
        engine.state.discard_pile = []

        engine.step(next(action for action in engine.legal_actions() if action.get("potion_id") == "GamblersBrew"))
        engine.step(next(action for action in engine.legal_actions() if action.get("target_index") == 0))
        engine.step(next(action for action in engine.legal_actions() if action.get("target_index") == 1))
        engine.step(next(action for action in engine.legal_actions() if action.get("target_index") == 2))
        engine.step(next(action for action in engine.legal_actions() if action["kind"] == "confirm"))

        self.assertEqual(
            [card["uuid"] for card in engine.state.discard_pile],
            ["gamblers-order-0", "gamblers-order-1", "gamblers-temp-cost"],
        )
        self.assertEqual(engine.state.discard_pile[-1]["cost_for_turn"], 3)
        self.assertNotIn("_reset_cost_for_turn_after_play", engine.state.discard_pile[-1])

    def test_gamblers_brew_selection_removes_staged_cards_from_visible_hand_select(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="Cultist",
            potions=[make_potion("GamblersBrew")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        engine = combat.engine
        engine.state.hand = [
            make_card("Defend_R", uuid="gamblers-visible-0"),
            make_card("Brutality", uuid="gamblers-visible-1"),
            make_card("Strike_R", uuid="gamblers-visible-2"),
        ]

        engine.step(next(action for action in engine.legal_actions() if action.get("potion_id") == "GamblersBrew"))
        engine.step(next(action for action in engine.legal_actions() if action.get("choice_index") == 0))

        visible_state = serialize_adapter.combat_state(combat, include_commands=True)
        screen_state = visible_state["screen_state"]
        card_select_actions = [action for action in engine.legal_actions() if action["kind"] == "card_select"]

        self.assertFalse(screen_state["confirm_up"])
        self.assertEqual([card["uuid"] for card in screen_state["cards"]], ["gamblers-visible-1", "gamblers-visible-2"])
        self.assertEqual([choice["kind"] for choice in visible_state["choice_list"]], ["card_select", "card_select"])
        self.assertEqual(card_select_actions[0]["choice_index"], 0)
        self.assertEqual(card_select_actions[0]["target_index"], 1)

        engine.step(card_select_actions[0])
        engine.step(next(action for action in engine.legal_actions() if action.get("kind") == "card_select"))
        visible_state = serialize_adapter.combat_state(combat, include_commands=True)

        self.assertEqual(visible_state["screen_state"]["cards"], [])
        self.assertEqual(visible_state["choice_list"], [])
        self.assertTrue(any(action["kind"] == "confirm" for action in engine.legal_actions()))

    def test_gamblers_brew_empty_hand_consumes_potion_without_select_or_draw(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="Cultist",
            potions=[make_potion("GamblersBrew")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        engine = combat.engine
        engine.state.hand = []
        engine.state.draw_pile = [make_card("Strike_R", uuid="gamblers-stays-draw")]

        engine.step(next(action for action in engine.legal_actions() if action.get("potion_id") == "GamblersBrew"))

        self.assertEqual(engine.potions[0]["potion_id"], "Potion Slot")
        self.assertIsNone(engine.pending_card_select)
        self.assertEqual(engine.state.hand, [])
        self.assertEqual([card["uuid"] for card in engine.state.draw_pile], ["gamblers-stays-draw"])

    def test_elixir_exposes_non_targeted_combat_potion_action(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="Cultist",
            potions=[make_potion("ElixirPotion")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        engine = combat.engine

        actions = engine.legal_actions()
        potion_action = next(action for action in actions if action.get("potion_id") == "ElixirPotion")

        self.assertEqual(potion_action["kind"], "potion")
        self.assertEqual(potion_action["potion_index"], 0)
        self.assertFalse(potion_action["requires_target"])
        self.assertNotIn("target_index", potion_action)

    def test_elixir_consumes_potion_and_opens_any_number_hand_exhaust_select(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="Cultist",
            potions=[make_potion("ElixirPotion")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        engine = combat.engine

        engine.step(next(action for action in engine.legal_actions() if action.get("potion_id") == "ElixirPotion"))

        self.assertEqual(engine.potions[0]["potion_id"], "Potion Slot")
        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.pending_card_select["mode"], "ELIXIR")
        self.assertEqual(engine.pending_card_select["potion_id"], "ElixirPotion")
        self.assertTrue(engine.pending_card_select["any_number"])
        self.assertTrue(engine.pending_card_select["can_pick_zero"])
        self.assertTrue(any(action["kind"] == "card_select" for action in engine.legal_actions()))
        self.assertTrue(any(action["kind"] == "confirm" for action in engine.legal_actions()))

    def test_elixir_confirm_selected_cards_exhausts_and_applies_exhaust_side_effects(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="Cultist",
            potions=[make_potion("ElixirPotion")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        engine = combat.engine
        engine.state.hand = [
            make_card("Strike_R", uuid="elixir-exhaust-0"),
            make_card("Defend_R", uuid="elixir-exhaust-1"),
            make_card("Bash", uuid="elixir-keep"),
        ]
        engine.state.exhaust_pile = []
        engine.player.block = 0
        combat_engine_mod._add_power(engine.player, "Feel No Pain", 3)

        engine.step(next(action for action in engine.legal_actions() if action.get("potion_id") == "ElixirPotion"))
        engine.step(next(action for action in engine.legal_actions() if action.get("target_index") == 0))
        engine.step(next(action for action in engine.legal_actions() if action.get("target_index") == 1))
        engine.step(next(action for action in engine.legal_actions() if action["kind"] == "confirm"))

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual(engine.potions[0]["potion_id"], "Potion Slot")
        self.assertEqual({card["uuid"] for card in engine.state.exhaust_pile}, {"elixir-exhaust-0", "elixir-exhaust-1"})
        self.assertEqual([card["uuid"] for card in engine.state.hand], ["elixir-keep"])
        self.assertEqual(engine.player.block, 6)

    def test_elixir_confirm_zero_consumes_potion_and_leaves_hand_and_exhaust_unchanged(self):
        combat = NativeCombatEnv(
            seed=44,
            ascension_level=0,
            encounter_name="Cultist",
            potions=[make_potion("ElixirPotion")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        engine = combat.engine
        engine.state.hand = [make_card("Strike_R", uuid="elixir-keep")]
        engine.state.exhaust_pile = []

        engine.step(next(action for action in engine.legal_actions() if action.get("potion_id") == "ElixirPotion"))
        engine.step(next(action for action in engine.legal_actions() if action["kind"] == "confirm"))

        self.assertEqual(engine.potions[0]["potion_id"], "Potion Slot")
        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["uuid"] for card in engine.state.hand], ["elixir-keep"])
        self.assertEqual(engine.state.exhaust_pile, [])

    def test_ninja_scroll_adds_three_shivs_before_opening_draw(self):
        combat = NativeCombatEnv(
            seed=45,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Ninja Scroll")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        shivs = [card for card in combat.engine.state.hand if card.get("card_id") == "Shiv"]
        self.assertEqual(len(shivs), 3)
        self.assertEqual(len(combat.engine.state.hand), 8)

    def test_innate_cards_are_in_opening_hand_after_shuffle(self):
        combat = NativeCombatEnv(
            seed=4759939624316417564,
            ascension_level=0,
            encounter_name="Cultist",
            master_deck=[
                make_card("Strike_R", uuid=f"strike-{idx}") for idx in range(5)
            ]
            + [
                make_card("Defend_R", uuid=f"defend-{idx}") for idx in range(4)
            ]
            + [
                make_card("Bash", uuid="bash"),
                make_card("Dramatic Entrance", uuid="dramatic-entrance"),
                make_card("True Grit", uuid="true-grit"),
                make_card("Carnage", uuid="carnage"),
                make_card("Shrug It Off", uuid="shrug"),
            ],
        )

        self.assertIn("Dramatic Entrance", [card["card_id"] for card in combat.engine.state.hand])
        self.assertNotIn("Dramatic Entrance", [card["card_id"] for card in combat.engine.state.draw_pile])

    def test_hand_drill_applies_vulnerable_when_attack_breaks_block(self):
        combat = NativeCombatEnv(
            seed=46,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("HandDrill")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        monster.block = 6
        combat.engine._player_deal_damage(monster, 6)
        self.assertEqual(monster.block, 0)
        self.assertEqual(_power_amount(monster, "Vulnerable"), 2)

    def test_blue_candle_allows_curse_play_and_exhausts_with_hp_loss(self):
        combat = NativeCombatEnv(
            seed=47,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Blue Candle")],
            master_deck=[],
        )
        combat.engine.state.hand = [make_card("Doubt", uuid="doubt-blue-candle")]
        combat.engine.player.energy = 3
        actions = [candidate for candidate in combat.engine.legal_actions() if candidate.get("card_id") == "Doubt"]
        self.assertEqual(len(actions), 1)
        before_hp = combat.engine.player.current_hp
        combat.engine.step(actions[0])
        self.assertEqual(before_hp - combat.engine.player.current_hp, 1)
        self.assertTrue(any(card.get("card_id") == "Doubt" for card in combat.engine.state.exhaust_pile))

    def test_medical_kit_allows_status_play_and_exhausts(self):
        combat = NativeCombatEnv(
            seed=48,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Medical Kit")],
            master_deck=[],
        )
        combat.engine.state.hand = [make_card("Dazed", uuid="dazed-medical-kit")]
        combat.engine.player.energy = 3
        actions = [candidate for candidate in combat.engine.legal_actions() if candidate.get("card_id") == "Dazed"]
        self.assertEqual(len(actions), 1)
        combat.engine.step(actions[0])
        self.assertTrue(any(card.get("card_id") == "Dazed" for card in combat.engine.state.exhaust_pile))

    def test_warped_tongs_upgrades_random_card_in_hand_after_draw(self):
        combat = NativeCombatEnv(
            seed=49,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("WarpedTongs")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        self.assertTrue(any(int(card.get("upgrades") or 0) > 0 for card in combat.engine.state.hand))
        combat.engine.state.hand = []
        combat.engine.state.draw_pile = [make_card("Strike_R", uuid=f"draw-{i}") for i in range(10)]
        combat.engine._start_player_turn()
        self.assertTrue(any(int(card.get("upgrades") or 0) > 0 for card in combat.engine.state.hand))

    def test_runic_pyramid_retains_non_ethereal_hand_at_end_of_turn(self):
        combat = NativeCombatEnv(
            seed=50,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Runic Pyramid")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        combat.engine.state.hand = [
            make_card("Strike_R", uuid="retained-strike"),
            make_card("Ghostly Armor", uuid="ethereal-ghostly"),
        ]
        combat.engine._end_turn()
        self.assertTrue(any(card.get("uuid") == "retained-strike" for card in combat.engine.state.hand))
        self.assertFalse(any(card.get("uuid") == "retained-strike" for card in combat.engine.state.discard_pile))
        self.assertTrue(any(card.get("uuid") == "ethereal-ghostly" for card in combat.engine.state.exhaust_pile))

    def test_ice_cream_retains_energy_between_turns(self):
        combat = NativeCombatEnv(
            seed=51,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Ice Cream")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        combat.engine.player.energy = 2
        combat.engine._end_turn()
        self.assertEqual(combat.engine.player.energy, 5)

    def test_calipers_caps_block_loss_at_start_of_turn(self):
        combat = NativeCombatEnv(
            seed=52,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Calipers")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        combat.engine.player.block = 27
        combat.engine._end_turn()
        self.assertEqual(combat.engine.player.block, 15)

    def test_unceasing_top_draws_when_hand_becomes_empty(self):
        combat = NativeCombatEnv(
            seed=53,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Unceasing Top")],
            master_deck=[],
        )
        combat.engine.state.hand = [make_card("Strike_R", uuid="top-strike")]
        combat.engine.state.draw_pile = [make_card("Defend_R", uuid="top-defend")]
        combat.engine.state.discard_pile = []
        combat.engine.player.energy = 3
        combat.engine.state.monsters[0].current_hp = 99
        action = next(candidate for candidate in combat.engine.legal_actions() if candidate.get("card_id") == "Strike_R")
        combat.engine.step(action)
        self.assertTrue(any(card.get("card_id") == "Defend_R" for card in combat.engine.state.hand))

    def test_preserved_insect_reduces_only_elite_opening_hp(self):
        elite_combat = NativeCombatEnv(
            seed=5,
            ascension_level=0,
            encounter_name="Gremlin Nob",
            room_type="MonsterRoomElite",
            relics=[make_relic("PreservedInsect")],
            master_deck=[make_card("Strike_R", uuid=f"elite-strike-{i}") for i in range(10)],
        )
        normal_combat = NativeCombatEnv(
            seed=5,
            ascension_level=0,
            encounter_name="Gremlin Nob",
            room_type="MonsterRoom",
            relics=[make_relic("PreservedInsect")],
            master_deck=[make_card("Strike_R", uuid=f"normal-strike-{i}") for i in range(10)],
        )
        self.assertEqual(elite_combat.engine.state.monsters[0].current_hp, int(normal_combat.engine.state.monsters[0].max_hp * 0.75))
        self.assertEqual(elite_combat.engine.state.monsters[0].max_hp, normal_combat.engine.state.monsters[0].max_hp)
        self.assertEqual(normal_combat.engine.state.monsters[0].current_hp, normal_combat.engine.state.monsters[0].max_hp)

    def test_blood_vial_heals_player_at_battle_start(self):
        player = PlayerState(current_hp=40, max_hp=80)
        combat = NativeCombatEnv(
            seed=6,
            ascension_level=0,
            encounter_name="Cultist",
            player=player,
            relics=[make_relic("Blood Vial")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        self.assertEqual(combat.engine.player.current_hp, 42)

    def test_magic_flower_scales_battle_start_and_card_healing_in_combat(self):
        player = PlayerState(current_hp=40, max_hp=80)
        combat = NativeCombatEnv(
            seed=54,
            ascension_level=0,
            encounter_name="Cultist",
            player=player,
            relics=[make_relic("Blood Vial"), make_relic("Magic Flower")],
            master_deck=[],
        )
        self.assertEqual(combat.engine.player.current_hp, 43)
        combat.engine.state.hand = [make_card("Bandage Up", uuid="magic-bandage")]
        combat.engine.player.energy = 3
        action = next(candidate for candidate in combat.engine.legal_actions() if candidate.get("card_id") == "Bandage Up")
        before_hp = combat.engine.player.current_hp
        combat.engine.step(action)
        self.assertEqual(combat.engine.player.current_hp - before_hp, 6)

    def test_twisted_funnel_applies_poison_to_all_monsters_at_battle_start(self):
        combat = NativeCombatEnv(
            seed=55,
            ascension_level=0,
            encounter_name="3 Cultists",
            relics=[make_relic("TwistedFunnel")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        self.assertTrue(all(_power_amount(monster, "Poison") == 4 for monster in combat.engine.state.monsters))

    def test_pantograph_heals_in_boss_room_only(self):
        boss_player = PlayerState(current_hp=20, max_hp=80)
        boss_combat = NativeCombatEnv(
            seed=56,
            ascension_level=0,
            encounter_name="The Guardian",
            room_type="MonsterRoomBoss",
            player=boss_player,
            relics=[make_relic("Pantograph")],
            master_deck=[make_card("Strike_R", uuid=f"boss-strike-{i}") for i in range(10)],
        )
        self.assertEqual(boss_combat.engine.player.current_hp, 45)
        normal_player = PlayerState(current_hp=20, max_hp=80)
        normal_combat = NativeCombatEnv(
            seed=57,
            ascension_level=0,
            encounter_name="Cultist",
            room_type="MonsterRoom",
            player=normal_player,
            relics=[make_relic("Pantograph")],
            master_deck=[make_card("Strike_R", uuid=f"normal-strike-{i}") for i in range(10)],
        )
        self.assertEqual(normal_combat.engine.player.current_hp, 20)

    def test_abacus_grants_block_when_draw_pile_is_shuffled(self):
        combat = NativeCombatEnv(
            seed=58,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("TheAbacus")],
            master_deck=[],
        )
        combat.engine.state.hand = []
        combat.engine.state.draw_pile = []
        combat.engine.state.discard_pile = [make_card("Strike_R", uuid="abacus-strike")]
        combat.engine.player.block = 0
        combat.engine.draw_cards(1)
        self.assertEqual(combat.engine.player.block, 6)

    def test_cloak_clasp_grants_block_based_on_hand_size_at_end_of_turn(self):
        combat = NativeCombatEnv(
            seed=59,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("CloakClasp")],
            master_deck=[],
        )
        combat.engine.state.hand = [
            make_card("Strike_R", uuid="clasp-strike"),
            make_card("Defend_R", uuid="clasp-defend"),
            make_card("Bash", uuid="clasp-bash"),
        ]
        combat.engine.player.block = 0
        combat.engine._apply_player_end_of_turn_powers()
        self.assertEqual(combat.engine.player.block, 3)

    def test_bloody_idol_heals_when_gold_is_gained(self):
        env = NativeRunEnv(seed=60, character="IRONCLAD", ascension_level=0)
        env.relics.append(make_relic("Bloody Idol"))
        env.player.current_hp = 40
        env.player.max_hp = 80
        gained = env._gain_gold(25)
        self.assertEqual(gained, 25)
        self.assertEqual(env.gold, 124)
        self.assertEqual(env.player.current_hp, 45)

    def test_ceramic_fish_grants_gold_when_obtaining_card(self):
        env = NativeRunEnv(seed=65, character="IRONCLAD", ascension_level=0)
        env.relics.append(make_relic("CeramicFish"))
        starting_gold = env.gold
        env._obtain_card(make_card("Strike_R", uuid="ceramic-strike"))
        self.assertEqual(env.gold, starting_gold + 9)
        self.assertTrue(any(card.get("uuid") == "ceramic-strike" for card in env.deck))

    def test_obtain_card_completes_compact_reward_card_metadata(self):
        env = NativeRunEnv(seed=65, character="IRONCLAD", ascension_level=0)

        obtained = env._obtain_card(
            {
                "card_id": "Limit Break",
                "name": "Limit Break",
                "type": "SKILL",
                "rarity": "RARE",
                "cost": 1,
                "upgrades": 0,
                "uuid": "compact-limit-break",
            }
        )

        self.assertTrue(obtained["exhausts"])
        self.assertEqual(obtained["target"], "SELF")
        deck_card = next(card for card in env.deck if card.get("uuid") == "compact-limit-break")
        self.assertTrue(deck_card["exhausts"])

    def test_obtain_card_keeps_upgraded_limit_break_non_exhausting(self):
        env = NativeRunEnv(seed=65, character="IRONCLAD", ascension_level=0)

        obtained = env._obtain_card(
            {
                "card_id": "Limit Break",
                "name": "Limit Break+",
                "type": "SKILL",
                "rarity": "RARE",
                "cost": 1,
                "upgrades": 1,
                "uuid": "compact-limit-break-plus",
            }
        )

        self.assertFalse(obtained["exhausts"])

    def test_darkstone_periapt_increases_max_hp_when_obtaining_curse(self):
        env = NativeRunEnv(seed=66, character="IRONCLAD", ascension_level=0)
        env.relics.append(make_relic("Darkstone Periapt"))
        env.player.current_hp = 50
        env.player.max_hp = 80
        env._obtain_card(make_card("Doubt", uuid="darkstone-doubt"))
        self.assertEqual(env.player.max_hp, 86)
        self.assertEqual(env.player.current_hp, 56)

    def test_darkstone_periapt_max_hp_heal_obeys_mark_of_the_bloom(self):
        env = NativeRunEnv(seed=66, character="IRONCLAD", ascension_level=0)
        env.relics = [make_relic("Darkstone Periapt"), make_relic("Mark of the Bloom")]
        env.player.current_hp = 50
        env.player.max_hp = 80

        env._obtain_card(make_card("Doubt", uuid="darkstone-mark-doubt"))

        self.assertEqual(env.player.max_hp, 86)
        self.assertEqual(env.player.current_hp, 50)
        self.assertTrue(any(card.get("uuid") == "darkstone-mark-doubt" for card in env.deck))

    def test_omamori_blocks_first_two_curses_on_obtain(self):
        env = NativeRunEnv(seed=66, character="IRONCLAD", ascension_level=0)
        env.relics.append(make_relic("Omamori", counter=2))
        initial_deck_size = len(env.deck)
        env._obtain_card(make_card("Doubt", uuid="omamori-doubt-1"))
        self.assertEqual(len(env.deck), initial_deck_size)
        omamori = next(relic for relic in env.relics if str(relic.get("relic_id") or relic.get("id")) == "Omamori")
        self.assertEqual(int(omamori.get("counter") or 0), 1)
        env._obtain_card(make_card("Shame", uuid="omamori-shame-2"))
        self.assertEqual(len(env.deck), initial_deck_size)
        self.assertEqual(int(omamori.get("counter") or 0), 0)
        self.assertTrue(bool(omamori.get("used_up")))
        env._obtain_card(make_card("Regret", uuid="omamori-regret-3"))
        self.assertEqual(len(env.deck), initial_deck_size + 1)

    def test_reward_omamori_payload_without_counter_uses_default_counter(self):
        env = NativeRunEnv(seed=66, character="IRONCLAD", ascension_level=0)
        reward_omamori = make_relic("Omamori")
        reward_omamori["counter"] = None
        env.phase = "CARD_REWARD"
        env.reward_relics = [reward_omamori]

        env.step({"kind": "reward_relic", "relic_id": "Omamori"})

        omamori = next(relic for relic in env.relics if str(relic.get("relic_id") or relic.get("id")) == "Omamori")
        self.assertEqual(omamori.get("counter"), 2)
        self.assertFalse(env.reward_relics)

    def test_egg_relics_upgrade_obtained_matching_cards(self):
        attack_env = NativeRunEnv(seed=67, character="IRONCLAD", ascension_level=0)
        attack_env.relics.append(make_relic("Molten Egg 2"))
        attack = attack_env._obtain_card(make_card("Strike_R", uuid="egg-strike"))
        self.assertEqual(int(attack.get("upgrades") or 0), 1)

        skill_env = NativeRunEnv(seed=68, character="IRONCLAD", ascension_level=0)
        skill_env.relics.append(make_relic("Toxic Egg 2"))
        skill = skill_env._obtain_card(make_card("Defend_R", uuid="egg-defend"))
        self.assertEqual(int(skill.get("upgrades") or 0), 1)

        power_env = NativeRunEnv(seed=69, character="IRONCLAD", ascension_level=0)
        power_env.relics.append(make_relic("Frozen Egg 2"))
        power = power_env._obtain_card(make_card("Inflame", uuid="egg-inflame"))
        self.assertEqual(int(power.get("upgrades") or 0), 1)

    def test_egg_relic_on_equip_upgrades_pending_reward_cards(self):
        env = NativeRunEnv(seed=70, character="IRONCLAD", ascension_level=0)
        env.reward_cards = [
            make_card("Shockwave", uuid="pending-shockwave"),
            make_card("Shrug It Off", uuid="pending-shrug"),
            make_card("Searing Blow", uuid="pending-searing"),
        ]
        env.reward_card_groups = [[make_card("True Grit", uuid="pending-true-grit")]]

        env._obtain_relic(make_relic("Toxic Egg 2"), source="reward_relic")

        self.assertEqual([int(card.get("upgrades") or 0) for card in env.reward_cards], [1, 1, 0])
        self.assertEqual(int(env.reward_card_groups[0][0].get("upgrades") or 0), 1)

    def test_face_of_cleric_grants_max_hp_on_post_combat_victory(self):
        env = NativeRunEnv(seed=67, character="IRONCLAD", ascension_level=0)
        env.relics = [make_relic("FaceOfCleric")]
        env.player.current_hp = 40
        env.player.max_hp = 80
        env._apply_post_combat_relic_effects()
        self.assertEqual(env.player.max_hp, 81)
        self.assertEqual(env.player.current_hp, 41)

    def test_face_of_cleric_max_hp_heal_obeys_mark_of_the_bloom(self):
        env = NativeRunEnv(seed=67, character="IRONCLAD", ascension_level=0)
        env.relics = [make_relic("FaceOfCleric"), make_relic("Mark of the Bloom")]
        env.player.current_hp = 40
        env.player.max_hp = 80

        env._apply_post_combat_relic_effects()

        self.assertEqual(env.player.max_hp, 81)
        self.assertEqual(env.player.current_hp, 40)

    def test_meal_ticket_heals_when_entering_shop_room(self):
        env = NativeRunEnv(seed=68, character="IRONCLAD", ascension_level=0)
        env.relics.append(make_relic("MealTicket"))
        env.player.current_hp = 40
        env.player.max_hp = 80
        env._handle_room_entry_relics("ShopRoom")
        self.assertEqual(env.player.current_hp, 55)

    def test_ssserpent_head_grants_gold_when_entering_event_room(self):
        env = NativeRunEnv(seed=68, character="IRONCLAD", ascension_level=0)
        env.relics.append(make_relic("SsserpentHead"))
        starting_gold = env.gold
        env._handle_room_entry_relics("EventRoom")
        self.assertEqual(env.gold, starting_gold + 50)

    def test_maw_bank_gains_gold_on_room_entry_and_is_consumed_on_spend(self):
        env = NativeRunEnv(seed=69, character="IRONCLAD", ascension_level=0)
        env.relics.append(make_relic("MawBank"))
        starting_gold = env.gold
        env._handle_room_entry_relics("MonsterRoom")
        self.assertEqual(env.gold, starting_gold + 12)
        spent = env._spend_gold(5)
        self.assertEqual(spent, 5)
        maw_bank = next(relic for relic in env.relics if str(relic.get("relic_id") or relic.get("id")) == "MawBank")
        self.assertTrue(bool(maw_bank.get("used_up")))
        after_spend_gold = env.gold
        env._handle_room_entry_relics("EventRoom")
        self.assertEqual(env.gold, after_spend_gold)

    def test_maw_bank_triggers_when_entering_direct_map_monster_room(self):
        env = NativeRunEnv(seed=69, character="IRONCLAD", ascension_level=0)
        env.relics.append(make_relic("MawBank"))
        env.gold = 35
        env.monster_list = ["Cultist"]
        env.current_map_node = None
        env.map = [[MapNode(x=0, y=0, room_symbol="M")]]

        env._step_map({"kind": "map", "node_id": "a1-r0-x0", "symbol": "M", "floor": 1})

        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual(env.gold, 47)

    def test_ancient_tea_set_arms_in_rest_room_and_grants_opening_energy(self):
        env = NativeRunEnv(seed=70, character="IRONCLAD", ascension_level=0)
        env.relics.append(make_relic("Ancient Tea Set"))
        env._handle_room_entry_relics("RestRoom")
        tea_set = next(relic for relic in env.relics if str(relic.get("relic_id") or relic.get("id")) == "Ancient Tea Set")
        self.assertEqual(int(tea_set.get("counter") or 0), -2)
        combat = NativeCombatEnv(
            seed=70,
            ascension_level=0,
            encounter_name="Cultist",
            relics=env.relics,
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        self.assertEqual(combat.engine.player.energy, 5)
        tea_set_combat = next(relic for relic in combat.engine.relics if str(relic.get("relic_id") or relic.get("id")) == "Ancient Tea Set")
        self.assertEqual(int(tea_set_combat.get("counter") or 0), -1)

    def test_eternal_feather_heals_on_rest_room_entry_based_on_deck_size(self):
        env = NativeRunEnv(seed=71, character="IRONCLAD", ascension_level=0)
        env.relics.append(make_relic("Eternal Feather"))
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.deck = [make_card("Strike_R", uuid=f"feather-{i}") for i in range(17)]
        env._handle_room_entry_relics("RestRoom")
        self.assertEqual(env.player.current_hp, 49)

    def test_sling_grants_strength_only_in_elite_rooms(self):
        elite = NativeCombatEnv(
            seed=61,
            ascension_level=0,
            encounter_name="Gremlin Nob",
            room_type="MonsterRoomElite",
            relics=[make_relic("Sling")],
            master_deck=[make_card("Strike_R", uuid=f"elite-strike-{i}") for i in range(10)],
        )
        normal = NativeCombatEnv(
            seed=62,
            ascension_level=0,
            encounter_name="Cultist",
            room_type="MonsterRoom",
            relics=[make_relic("Sling")],
            master_deck=[make_card("Strike_R", uuid=f"normal-strike-{i}") for i in range(10)],
        )
        event_elite = NativeCombatEnv(
            seed=63,
            ascension_level=0,
            encounter_name="Gremlin Nob",
            room_type="EventRoom",
            relics=[make_relic("Sling")],
            elite_trigger=True,
            master_deck=[make_card("Strike_R", uuid=f"event-elite-strike-{i}") for i in range(10)],
        )
        event_normal = NativeCombatEnv(
            seed=64,
            ascension_level=0,
            encounter_name="Cultist",
            room_type="EventRoom",
            relics=[make_relic("Sling")],
            master_deck=[make_card("Strike_R", uuid=f"event-normal-strike-{i}") for i in range(10)],
        )
        self.assertEqual(_power_amount(elite.engine.player, "Strength"), 2)
        self.assertEqual(_power_amount(normal.engine.player, "Strength"), 0)
        self.assertEqual(_power_amount(event_elite.engine.player, "Strength"), 2)
        self.assertEqual(_power_amount(event_normal.engine.player, "Strength"), 0)

    def test_gremlin_mask_applies_player_weak_at_battle_start(self):
        combat = NativeCombatEnv(
            seed=63,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("GremlinMask")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        self.assertEqual(_power_amount(combat.engine.player, "Weakened"), 1)

    def test_the_specimen_transfers_poison_on_monster_death(self):
        combat = NativeCombatEnv(
            seed=64,
            ascension_level=0,
            encounter_name="2 Louse",
            relics=[make_relic("The Specimen")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        dying = combat.engine.state.monsters[0]
        survivor = combat.engine.state.monsters[1]
        combat.engine._apply_monster_debuff(dying, "Poison", 5)
        combat.engine._kill_monster(dying)
        self.assertEqual(_power_amount(survivor, "Poison"), 5)

    def test_horn_cleat_grants_block_on_second_turn_start(self):
        combat = NativeCombatEnv(
            seed=7,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("HornCleat")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        horn = next(r for r in combat.engine.relics if r["relic_id"] == "HornCleat")
        self.assertEqual(combat.engine.player.block, 0)
        self.assertEqual(horn["counter"], 1)
        combat.engine._start_player_turn()
        self.assertEqual(combat.engine.player.block, 14)
        self.assertEqual(horn["counter"], -1)
        self.assertTrue(horn["grayscale"])

    def test_horn_cleat_block_ignores_panic_button_no_block_power(self):
        combat = NativeCombatEnv(
            seed=7,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("HornCleat")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        combat.engine.player.block = 38
        combat_engine_mod._add_power(combat.engine.player, "NoBlockPower", 1)
        combat.engine._start_player_turn()

        self.assertEqual(combat.engine.player.block, 14)

    def test_orichalcum_grants_end_of_turn_block_only_when_none(self):
        combat = NativeCombatEnv(
            seed=8,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Orichalcum")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        combat.engine.player.block = 0
        combat.engine._apply_player_end_of_turn_powers()
        self.assertEqual(combat.engine.player.block, 6)

        combat.engine.player.block = 2
        combat.engine._apply_player_end_of_turn_powers()
        self.assertEqual(combat.engine.player.block, 2)

    def test_fossilized_helix_grants_buffer_and_negates_next_hit(self):
        combat = NativeCombatEnv(
            seed=81,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("FossilizedHelix")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        starting_hp = combat.engine.player.current_hp
        self.assertEqual(_power_amount(combat.engine.player, "Buffer"), 1)
        dealt = combat.engine._damage_player(9, attacker=monster)
        self.assertEqual(dealt, 0)
        self.assertEqual(combat.engine.player.current_hp, starting_hp)
        self.assertEqual(_power_amount(combat.engine.player, "Buffer"), 0)

    def test_bronze_scales_grants_thorns_and_retaliates_on_hit(self):
        combat = NativeCombatEnv(
            seed=82,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Bronze Scales")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        starting_monster_hp = monster.current_hp
        starting_player_hp = combat.engine.player.current_hp
        combat.engine._damage_player(5, attacker=monster)
        self.assertEqual(starting_monster_hp - monster.current_hp, 3)
        self.assertEqual(starting_player_hp - combat.engine.player.current_hp, 5)
        self.assertEqual(_power_amount(combat.engine.player, "Thorns"), 3)

    def test_bronze_scales_retaliates_even_when_attack_is_fully_blocked(self):
        combat = NativeCombatEnv(
            seed=82,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Bronze Scales")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        combat.engine.player.block = 10
        starting_monster_hp = monster.current_hp
        starting_player_hp = combat.engine.player.current_hp

        dealt = combat.engine._damage_player(5, attacker=monster)

        self.assertEqual(dealt, 0)
        self.assertEqual(starting_monster_hp - monster.current_hp, 3)
        self.assertEqual(combat.engine.player.current_hp, starting_player_hp)

    def test_weakened_zero_damage_multi_hit_still_triggers_thorns(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=5, floor=19),
            ascension_level=0,
            act=2,
            player=PlayerState(current_hp=62, max_hp=80, block=15),
            master_deck=[],
        )
        byrd = engine.state.monsters[0]
        byrd.current_hp = 10
        byrd.next_move = "PECK"
        byrd.meta["peck_damage"] = 1
        byrd.meta["peck_count"] = 5
        combat_engine_mod._append_power(engine.player, "Thorns", 3, misc=3)
        combat_engine_mod._add_power(byrd, "Weakened", 1)

        engine._byrd_take_turn(byrd)

        self.assertEqual(engine.player.current_hp, 62)
        self.assertEqual(byrd.current_hp, 0)
        self.assertEqual(byrd.block, 0)
        self.assertEqual(byrd.powers, [])

    def test_torii_reduces_small_post_block_hit_to_one(self):
        combat = NativeCombatEnv(
            seed=83,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Torii")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        starting_hp = combat.engine.player.current_hp
        dealt = combat.engine._damage_player(5, attacker=monster)
        self.assertEqual(dealt, 1)
        self.assertEqual(starting_hp - combat.engine.player.current_hp, 1)

    def test_tungsten_rod_reduces_explicit_hp_loss_by_one(self):
        combat = NativeCombatEnv(
            seed=84,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("TungstenRod")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        starting_hp = combat.engine.player.current_hp
        combat.engine._lose_player_hp(2)
        self.assertEqual(starting_hp - combat.engine.player.current_hp, 1)

    def test_lizard_tail_revives_player_on_lethal_damage(self):
        player = PlayerState(current_hp=4, max_hp=80)
        combat = NativeCombatEnv(
            seed=85,
            ascension_level=0,
            encounter_name="Cultist",
            player=player,
            relics=[make_relic("Lizard Tail")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        relic = next(r for r in combat.engine.relics if r["relic_id"] == "Lizard Tail")
        dealt = combat.engine._damage_player(9, attacker=monster)
        self.assertEqual(dealt, 9)
        self.assertEqual(combat.engine.player.current_hp, 40)
        self.assertEqual(relic["counter"], -2)
        self.assertTrue(relic.get("used_up"))
        self.assertEqual(combat.engine.outcome, "UNDECIDED")

    def test_used_lizard_tail_does_not_revive_player_again(self):
        player = PlayerState(current_hp=4, max_hp=80)
        combat = NativeCombatEnv(
            seed=85,
            ascension_level=0,
            encounter_name="Cultist",
            player=player,
            relics=[{"relic_id": "Lizard Tail", "id": "Lizard Tail", "name": "Lizard Tail", "counter": -2, "used_up": True}],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]

        combat.engine._damage_player(9, attacker=monster)

        self.assertEqual(combat.engine.player.current_hp, 0)
        self.assertEqual(combat.engine.outcome, "DEFEAT")

    def test_fairy_potion_revives_player_on_lethal_damage_and_is_consumed(self):
        player = PlayerState(current_hp=4, max_hp=80)
        combat = NativeCombatEnv(
            seed=85,
            ascension_level=0,
            encounter_name="Cultist",
            player=player,
            potions=[make_potion("Block Potion"), make_potion("SpeedPotion"), make_potion("FairyPotion")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        dealt = combat.engine._damage_player(9, attacker=monster)
        self.assertEqual(dealt, 9)
        self.assertEqual(combat.engine.player.current_hp, 24)
        self.assertEqual([p["potion_id"] for p in combat.engine.potions], ["Block Potion", "SpeedPotion"])
        self.assertEqual(combat.engine.outcome, "UNDECIDED")

    def test_fairy_potion_revives_before_lizard_tail(self):
        player = PlayerState(current_hp=4, max_hp=80)
        combat = NativeCombatEnv(
            seed=85,
            ascension_level=0,
            encounter_name="Cultist",
            player=player,
            relics=[make_relic("Lizard Tail")],
            potions=[make_potion("FairyPotion")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        relic = next(r for r in combat.engine.relics if r["relic_id"] == "Lizard Tail")
        combat.engine._damage_player(9, attacker=monster)
        self.assertEqual(combat.engine.player.current_hp, 24)
        self.assertEqual(combat.engine.potions, [])
        self.assertEqual(int(relic.get("counter", -1) or -1), -1)
        self.assertFalse(relic.get("used_up", False))

    def test_mark_of_the_bloom_blocks_fairy_potion_revive(self):
        player = PlayerState(current_hp=4, max_hp=80)
        combat = NativeCombatEnv(
            seed=85,
            ascension_level=0,
            encounter_name="Cultist",
            player=player,
            relics=[make_relic("Mark of the Bloom")],
            potions=[make_potion("FairyPotion")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        combat.engine._damage_player(9, attacker=monster)
        self.assertEqual(combat.engine.player.current_hp, 0)
        self.assertEqual([p["potion_id"] for p in combat.engine.potions], ["FairyPotion"])
        self.assertEqual(combat.engine.outcome, "DEFEAT")

    def test_flame_barrier_retaliates_and_clears_at_turn_start(self):
        combat = NativeCombatEnv(
            seed=86,
            ascension_level=0,
            encounter_name="Cultist",
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        combat.engine.player.powers.append(
            {
                "power_id": "FlameBarrier",
                "id": "FlameBarrier",
                "name": "FlameBarrier",
                "amount": 4,
                "misc": 4,
            }
        )
        starting_hp = monster.current_hp
        combat.engine._damage_player(6, attacker=monster)
        self.assertEqual(starting_hp - monster.current_hp, 4)
        combat.engine._start_player_turn()
        self.assertEqual(_power_amount(combat.engine.player, "FlameBarrier"), 0)

    def test_flame_barrier_retaliates_even_when_attack_is_fully_blocked(self):
        combat = NativeCombatEnv(
            seed=86,
            ascension_level=0,
            encounter_name="Cultist",
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        combat.engine.player.block = 10
        combat.engine.player.powers.append(
            {
                "power_id": "FlameBarrier",
                "id": "FlameBarrier",
                "name": "FlameBarrier",
                "amount": 4,
                "misc": 4,
            }
        )
        starting_hp = monster.current_hp

        dealt = combat.engine._damage_player(6, attacker=monster)

        self.assertEqual(dealt, 0)
        self.assertEqual(starting_hp - monster.current_hp, 4)

    def test_flame_barrier_stacks_into_single_power(self):
        combat = NativeCombatEnv(
            seed=86,
            ascension_level=0,
            encounter_name="Cultist",
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        combat.engine.player.energy = 10
        combat.engine.state.hand = [make_card("Flame Barrier", uuid="flame-barrier-0")]
        combat.engine.step({"kind": "card", "card_index": 0})
        combat.engine.state.hand = [make_card("Flame Barrier", upgrades=1, uuid="flame-barrier-1")]

        combat.engine.step({"kind": "card", "card_index": 0})

        flame_barriers = [
            power
            for power in combat.engine.player.powers
            if str(power.get("power_id") or power.get("id")) == "FlameBarrier"
        ]
        self.assertEqual(len(flame_barriers), 1)
        self.assertEqual(flame_barriers[0]["amount"], 10)

    def test_flame_barrier_serializes_with_comm_mod_power_id(self):
        combat = NativeCombatEnv(
            seed=86,
            ascension_level=0,
            encounter_name="Cultist",
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        combat.engine.player.powers.append(
            {
                "power_id": "FlameBarrier",
                "id": "FlameBarrier",
                "name": "FlameBarrier",
                "amount": 4,
                "misc": 4,
            }
        )

        payload = combat.serialize()
        powers = payload["combat_state"]["player"]["powers"]

        self.assertEqual(powers[0]["power_id"], "Flame Barrier")
        self.assertEqual(powers[0]["id"], "Flame Barrier")
        self.assertEqual(powers[0]["name"], "Flame Barrier")

    def test_thread_and_needle_grants_plated_armor_at_battle_start(self):
        combat = NativeCombatEnv(
            seed=861,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Thread and Needle")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        self.assertEqual(_power_amount(combat.engine.player, "Plated Armor"), 4)

    def test_centennial_puzzle_draws_three_only_on_first_hp_loss(self):
        combat = NativeCombatEnv(
            seed=862,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Centennial Puzzle")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        relic = next(r for r in combat.engine.relics if r["relic_id"] == "Centennial Puzzle")
        starting_hand = len(combat.engine.state.hand)
        monster = combat.engine.state.monsters[0]
        combat.engine._damage_player(6, attacker=monster)
        self.assertEqual(len(combat.engine.state.hand), starting_hand + 3)
        self.assertTrue(relic.get("used_this_combat"))
        hand_after_first = len(combat.engine.state.hand)
        combat.engine._damage_player(6, attacker=monster)
        self.assertEqual(len(combat.engine.state.hand), hand_after_first)

    def test_self_forming_clay_grants_next_turn_block_after_hp_loss(self):
        combat = NativeCombatEnv(
            seed=863,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Self Forming Clay")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        combat.engine._damage_player(6, attacker=monster)
        self.assertEqual(_power_amount(combat.engine.player, "NextTurnBlock"), 3)
        combat.engine._start_player_turn()
        self.assertEqual(_power_amount(combat.engine.player, "NextTurnBlock"), 0)
        self.assertEqual(combat.engine.player.block, 3)

    def test_red_skull_toggles_strength_when_player_becomes_bloodied(self):
        player = PlayerState(current_hp=30, max_hp=80)
        combat = NativeCombatEnv(
            seed=864,
            ascension_level=0,
            encounter_name="Cultist",
            player=player,
            relics=[make_relic("Red Skull")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        relic = next(r for r in combat.engine.relics if r["relic_id"] == "Red Skull")
        self.assertEqual(_power_amount(combat.engine.player, "Strength"), 3)
        self.assertTrue(relic.get("is_active"))
        combat.engine.player.current_hp = 50
        combat.engine._update_bloodied_relics()
        self.assertEqual(_power_amount(combat.engine.player, "Strength"), 0)
        self.assertFalse(relic.get("is_active"))

    def test_red_skull_battle_start_direct_power_preserves_source_order(self):
        combat = NativeCombatEnv(
            seed=866,
            ascension_level=0,
            encounter_name="Cultist",
            player=PlayerState(current_hp=16, max_hp=80),
            relics=[make_relic("GremlinMask"), make_relic("Akabeko"), make_relic("Red Skull")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        relic = next(r for r in combat.engine.relics if r["relic_id"] == "Red Skull")

        self.assertEqual(
            [power["power_id"] for power in combat.engine.player.powers],
            ["Vigor", "Weakened", "Strength"],
        )
        self.assertTrue(relic.get("is_active"))

    def test_red_skull_triggers_immediately_on_hp_loss_before_later_debuffs(self):
        combat = NativeCombatEnv(
            seed=864,
            ascension_level=0,
            encounter_name="Cultist",
            player=PlayerState(current_hp=44, max_hp=80),
            relics=[make_relic("Red Skull")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        combat_engine_mod._add_power(combat.engine.player, "Evolve", 1)

        combat.engine._damage_player(12, attacker=combat.engine.state.monsters[0], apply_player_vulnerable=False)
        combat.engine._apply_player_debuff("Hex", 1)

        self.assertEqual(
            [power["power_id"] for power in combat.engine.player.powers],
            ["Evolve", "Strength", "Hex"],
        )

    def test_red_skull_strength_is_removed_after_reaper_heals_out_of_bloodied(self):
        combat = NativeCombatEnv(
            seed=865,
            ascension_level=0,
            encounter_name="Cultist",
            player=PlayerState(current_hp=18, max_hp=80),
            relics=[make_relic("Red Skull")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        engine = combat.engine
        combat_engine_mod._add_power(engine.player, "Strength", 25)
        engine.state.hand = [make_card("Reaper", uuid="reaper")]
        engine.player.energy = 3

        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Reaper"))

        self.assertGreater(engine.player.current_hp, engine.player.max_hp // 2)
        self.assertEqual(_power_amount(engine.player, "Strength"), 25)
        relic = next(r for r in engine.relics if r["relic_id"] == "Red Skull")
        self.assertFalse(relic.get("is_active"))
        engine._damage_player(2, damage_type="HP_LOSS")
        self.assertEqual(_power_amount(engine.player, "Strength"), 25)

    def test_orange_pellets_clears_debuffs_immediately_after_power_card_completes_set(self):
        combat = NativeCombatEnv(
            seed=867,
            ascension_level=0,
            encounter_name="Cultist",
            player=PlayerState(current_hp=60, max_hp=80),
            relics=[make_relic("OrangePellets")],
            master_deck=[
                make_card("Strike_R", uuid="strike"),
                make_card("Defend_R", uuid="defend"),
                make_card("Brutality", uuid="brutality"),
            ],
        )
        engine = combat.engine
        combat_engine_mod._add_power(engine.player, "Hex", 1)
        engine.state.hand = [
            make_card("Strike_R", uuid="strike"),
            make_card("Defend_R", uuid="defend"),
            make_card("Brutality", uuid="brutality"),
        ]
        engine.player.energy = 10

        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Strike_R"))
        self.assertEqual(_power_amount(engine.player, "Hex"), 1)
        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Defend_R"))
        self.assertEqual(_power_amount(engine.player, "Hex"), 1)
        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Brutality"))

        self.assertEqual(_power_amount(engine.player, "Hex"), 0)
        relic = next(relic for relic in engine.relics if relic["relic_id"] == "OrangePellets")
        self.assertFalse(relic.get("attack_played"))
        self.assertFalse(relic.get("skill_played"))
        self.assertFalse(relic.get("power_played"))

    def test_orange_pellets_clears_flex_loss_debuff_but_keeps_strength(self):
        combat = NativeCombatEnv(
            seed=868,
            ascension_level=0,
            encounter_name="Cultist",
            player=PlayerState(current_hp=60, max_hp=80),
            relics=[make_relic("OrangePellets")],
            master_deck=[
                make_card("Flex", uuid="flex"),
                make_card("Strike_R", uuid="strike"),
                make_card("Brutality", uuid="brutality"),
            ],
        )
        engine = combat.engine
        engine.state.hand = [
            make_card("Flex", uuid="flex"),
            make_card("Strike_R", uuid="strike"),
            make_card("Brutality", uuid="brutality"),
        ]
        engine.player.energy = 10

        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Flex"))
        self.assertEqual(_power_amount(engine.player, "Strength"), 2)
        self.assertEqual(_power_amount(engine.player, "Flex"), 2)
        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Strike_R"))
        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Brutality"))

        self.assertEqual(_power_amount(engine.player, "Strength"), 2)
        self.assertEqual(_power_amount(engine.player, "Flex"), 0)

    def test_end_turn_flex_loss_consumes_artifact_before_reducing_strength(self):
        player = PlayerState(current_hp=60, max_hp=80)
        combat_engine_mod._add_power(player, "Strength", 4)
        combat_engine_mod._add_power(player, "Flex", 4)
        combat_engine_mod._add_power(player, "Artifact", 1)

        combat_engine_mod._apply_end_of_turn_temporary_powers(player)

        self.assertEqual(_power_amount(player, "Strength"), 4)
        self.assertEqual(_power_amount(player, "Artifact"), 0)
        self.assertEqual(_power_amount(player, "Flex"), 0)

    def test_spot_weakness_places_strength_before_existing_player_weak(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine._apply_player_debuff("Weak", 3)
        target = engine.state.monsters[0]
        target.next_move = "DARK_STRIKE"
        target.move_adjusted_damage = 6
        engine.state.hand = [make_card("Spot Weakness", uuid="spot")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Spot Weakness")
        engine.step(action)

        self.assertEqual([power["power_id"] for power in engine.player.powers], ["Strength", "Weakened"])

    def test_go_for_the_eyes_deals_damage_and_weakens_attacking_target(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        target.intent = "ATTACK"
        before_hp = target.current_hp
        engine.state.hand = [make_card("Go for the Eyes", upgrades=1, uuid="go-eyes-plus")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Go for the Eyes")
        engine.step(action)

        self.assertEqual(target.current_hp, before_hp - 4)
        self.assertEqual(_power_amount(target, "Weakened"), 2)

    def test_go_for_the_eyes_does_not_weaken_non_attacking_target(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        target.intent = "BUFF"
        before_hp = target.current_hp
        engine.state.hand = [make_card("Go for the Eyes", uuid="go-eyes")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Go for the Eyes")
        engine.step(action)

        self.assertEqual(target.current_hp, before_hp - 3)
        self.assertEqual(_power_amount(target, "Weakened"), 0)

    def test_bane_deals_one_hit_to_unpoisoned_target(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        before_hp = target.current_hp
        engine.state.hand = [make_card("Bane", uuid="bane")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Bane")
        engine.step(action)

        self.assertEqual(target.current_hp, before_hp - 7)

    def test_bane_deals_second_hit_to_poisoned_surviving_target(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        combat_engine_mod._add_power(target, "Poison", 2)
        before_hp = target.current_hp
        engine.state.hand = [make_card("Bane", uuid="bane")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Bane")
        engine.step(action)

        self.assertEqual(target.current_hp, before_hp - 14)

    def test_bane_does_not_second_hit_when_first_hit_kills_poisoned_target(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        target.current_hp = 5
        combat_engine_mod._add_power(target, "Poison", 2)
        engine.state.hand = [make_card("Bane", uuid="bane")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Bane")
        engine.step(action)

        self.assertEqual(target.current_hp, 0)
        self.assertFalse(combat_engine_mod._alive(target))

    def test_upgraded_bane_deals_ten_per_hit_to_poisoned_target(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        combat_engine_mod._add_power(target, "Poison", 2)
        before_hp = target.current_hp
        engine.state.hand = [make_card("Bane", upgrades=1, uuid="bane-plus")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Bane")
        engine.step(action)

        self.assertEqual(target.current_hp, before_hp - 20)

    def test_strength_sorts_before_player_frail_and_weak(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        combat_engine_mod._add_power(engine.player, "Demon Form", 2)
        engine._apply_player_debuff("Frail", 2)
        engine._apply_player_debuff("Weak", 2)

        combat_engine_mod._add_power(engine.player, "Strength", 2)

        self.assertEqual(
            [power["power_id"] for power in engine.player.powers],
            ["Demon Form", "Strength", "Frail", "Weakened"],
        )

    def test_player_frail_sorts_before_weak(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        combat_engine_mod._add_power(engine.player, "Metallicize", 3)
        engine._apply_player_debuff("Weak", 1)
        engine._apply_player_debuff("Frail", 1)

        self.assertEqual(
            [power["power_id"] for power in engine.player.powers],
            ["Metallicize", "Frail", "Weakened"],
        )

    def test_vajra_sorts_before_oddly_smooth_stone_power(self):
        combat = NativeCombatEnv(
            seed=867,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Oddly Smooth Stone"), make_relic("Vajra")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )

        self.assertEqual(
            [power["power_id"] for power in combat.engine.player.powers],
            ["Strength", "Dexterity"],
        )

    def test_vajra_start_power_resolves_before_akabeko_vigor(self):
        combat = NativeCombatEnv(
            seed=869,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Akabeko"), make_relic("Vajra")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )

        self.assertEqual(
            [power["power_id"] for power in combat.engine.player.powers],
            ["Strength", "Vigor"],
        )

    def test_player_vulnerable_sorts_after_dexterity(self):
        combat = NativeCombatEnv(
            seed=868,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Oddly Smooth Stone"), make_relic("Vajra")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )

        combat.engine._apply_player_debuff("Vulnerable", 1)

        self.assertEqual(
            [power["power_id"] for power in combat.engine.player.powers],
            ["Strength", "Dexterity", "Vulnerable"],
        )

    def test_player_power_card_preserves_append_order_after_vulnerable(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine._apply_player_debuff("Hex", 1)
        engine._apply_player_debuff("Vulnerable", 1)
        engine.state.hand = [make_card("Demon Form", uuid="demon-form")]
        engine.player.energy = 4

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Demon Form")
        engine.step(action)

        self.assertEqual(
            [power["power_id"] for power in engine.player.powers],
            ["Hex", "Vulnerable", "Demon Form"],
        )

    def test_demon_form_sorts_before_existing_player_weak(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine._apply_player_debuff("Hex", 1)
        engine._apply_player_debuff("Weak", 3)
        engine.state.hand = [make_card("Demon Form", uuid="demon-form")]
        engine.player.energy = 4

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Demon Form")
        engine.step(action)

        self.assertEqual(
            [power["power_id"] for power in engine.player.powers],
            ["Hex", "Demon Form", "Weakened"],
        )

    def test_combust_sorts_before_existing_player_weak(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine._apply_player_debuff("Weak", 3)
        engine.state.hand = [make_card("Combust", upgrades=1, uuid="combust")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Combust")
        engine.step(action)

        self.assertEqual(
            [power["power_id"] for power in engine.player.powers],
            ["Combust", "Weakened"],
        )

    def test_havoc_does_not_exhaust_power_card_it_plays(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Havoc", upgrades=1, uuid="havoc")]
        engine.state.draw_pile = [make_card("Inflame", uuid="havoc-inflame")]
        engine.state.discard_pile = []
        engine.state.exhaust_pile = []
        engine.player.energy = 1

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Havoc")
        engine.step(action)

        self.assertEqual(_power_amount(engine.player, "Strength"), 2)
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], [])
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Havoc"])

    def test_hex_adds_dazed_to_random_draw_pile_spot_for_non_attack(self):
        probe = NativeRandomSet(seed=44)
        draw_pile = [make_card("Strike_R", uuid=f"draw-{index}") for index in range(5)]
        expected_index = int(probe.stream("card_random").random(len(draw_pile) - 1))
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=44),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Defend_R", uuid="hex-defend")]
        engine.state.draw_pile = list(draw_pile)
        engine.player.energy = 3
        combat_engine_mod._add_power(engine.player, "Hex", 1)

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Defend_R")
        engine.step(action)

        self.assertEqual(engine.state.draw_pile[expected_index]["card_id"], "Dazed")
        self.assertEqual(sum(1 for card in engine.state.draw_pile if card["card_id"] == "Dazed"), 1)

    def test_meat_on_the_bone_heals_on_victory_when_bloodied(self):
        player = PlayerState(current_hp=30, max_hp=80)
        combat = NativeCombatEnv(
            seed=865,
            ascension_level=0,
            encounter_name="Cultist",
            player=player,
            relics=[make_relic("Meat on the Bone")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        combat.engine.outcome = "VICTORY"
        combat.engine._handle_victory_relics()
        self.assertEqual(combat.engine.player.current_hp, 42)

    def test_gremlin_horn_grants_energy_and_draw_on_nonfinal_kill(self):
        combat = NativeCombatEnv(
            seed=866,
            ascension_level=0,
            encounter_name="3 Cultists",
            relics=[make_relic("Gremlin Horn")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        combat.engine.player.energy = 0
        combat.engine.state.hand = []
        target = combat.engine.state.monsters[0]
        combat.engine._kill_monster(target)
        self.assertEqual(combat.engine.player.energy, 1)
        self.assertEqual(len(combat.engine.state.hand), 1)

    def test_pen_nib_applies_to_next_attack_and_resets_counter(self):
        combat = NativeCombatEnv(
            seed=867,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Pen Nib", counter=8)],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        relic = next(r for r in combat.engine.relics if r["relic_id"] == "Pen Nib")
        monster = combat.engine.state.monsters[0]
        combat.engine._resolve_generated_card_play(make_card("Strike_R", uuid="pen-1"))
        self.assertEqual(relic["counter"], 9)
        self.assertEqual(_power_amount(combat.engine.player, "Pen Nib"), 1)
        before = monster.current_hp
        combat.engine._resolve_generated_card_play(make_card("Strike_R", uuid="pen-2"))
        self.assertEqual(before - monster.current_hp, 12)
        self.assertEqual(relic["counter"], 0)
        self.assertEqual(_power_amount(combat.engine.player, "Pen Nib"), 0)

    def test_dead_branch_adds_random_card_to_hand_on_exhaust(self):
        combat = NativeCombatEnv(
            seed=868,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Dead Branch")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        starting_hand = len(combat.engine.state.hand)
        combat.engine._exhaust_card(make_card("Strike_R", uuid="dead-branch-exhaust"))
        self.assertEqual(len(combat.engine.state.hand), starting_hand + 1)

    def test_dead_branch_full_hand_sends_generated_card_to_discard(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            relics=[make_relic("Dead Branch")],
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": ["Clash"],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": [],
                "SRC_CURSE": [],
            },
        )
        engine.state.hand = [make_card("Strike_R", uuid=f"full-hand-{index}") for index in range(10)]
        engine.state.discard_pile = []

        engine._exhaust_card(make_card("Dazed", uuid="dead-branch-overflow-source"))

        self.assertEqual(len(engine.state.hand), 10)
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Clash"])
        self.assertTrue(engine.state.discard_pile[0]["uuid"].startswith("dead-branch-"))
        self.assertEqual([card["uuid"] for card in engine.state.exhaust_pile], ["dead-branch-overflow-source"])

    def test_dead_branch_deferred_burning_pact_overflow_happens_after_draw(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80, energy=3, base_energy=3),
            relics=[make_relic("Dead Branch")],
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": ["Clash"],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": [],
                "SRC_CURSE": [],
            },
        )
        engine.state.hand = [
            make_card("Burning Pact", uuid="burning-pact"),
            make_card("Strike_R", uuid="burning-target"),
            *[make_card("Defend_R", uuid=f"filler-{index}") for index in range(8)],
        ]
        engine.state.draw_pile = [
            make_card("Strike_R", uuid="burning-draw-0"),
            make_card("Defend_R", uuid="burning-draw-1"),
        ]
        engine.state.discard_pile = []
        engine.player.energy = 3

        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Burning Pact"))
        engine.step(next(action for action in engine.legal_actions() if action.get("target_index") == 0))

        hand_uuids = {card["uuid"] for card in engine.state.hand}
        discard_ids = [card["card_id"] for card in engine.state.discard_pile]
        self.assertEqual(len(engine.state.hand), 10)
        self.assertIn("burning-draw-0", hand_uuids)
        self.assertIn("burning-draw-1", hand_uuids)
        self.assertIn("Clash", discard_ids)
        self.assertNotIn("burning-target", hand_uuids)
        self.assertIn("burning-target", [card["uuid"] for card in engine.state.exhaust_pile])

    def test_dead_branch_triggers_for_ethereal_cards_exhausted_at_end_turn(self):
        combat = NativeCombatEnv(
            seed=9917423850887664223,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Dead Branch")],
            master_deck=[],
        )
        combat.engine.state.hand = [
            make_card("Dazed", uuid="ethereal-dazed-0"),
            make_card("Dazed", uuid="ethereal-dazed-1"),
            make_card("Dazed", uuid="ethereal-dazed-2"),
        ]
        combat.engine.state.draw_pile = []
        combat.engine.state.discard_pile = []

        combat.engine._end_turn()

        self.assertEqual([card["card_id"] for card in combat.engine.state.exhaust_pile], ["Dazed", "Dazed", "Dazed"])
        self.assertEqual(len(combat.engine.state.hand), 3)
        self.assertTrue(all(card["card_id"] != "Dazed" for card in combat.engine.state.hand))

    def test_dead_branch_uses_colored_combat_pool_not_colorless_pool(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            relics=[make_relic("Dead Branch")],
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": ["Clash"],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": ["Swift Strike"],
                "SRC_CURSE": [],
            },
        )

        engine._exhaust_card(make_card("Dazed", uuid="dead-branch-source"))

        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Clash"])

    def test_dead_branch_charon_low_hp_direct_exhaust_generates_before_ashes_kill(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            relics=[make_relic("Dead Branch"), make_relic("Charon's Ashes")],
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": ["Clash"],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": [],
                "SRC_CURSE": [],
            },
        )
        engine.state.monsters[0].current_hp = 3
        engine.state.hand = []

        engine._exhaust_card(make_card("Dazed", uuid="dead-branch-charon-source"))

        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Clash"])
        self.assertEqual(engine.state.monsters[0].current_hp, 0)
        self.assertEqual(engine.outcome, "VICTORY")

    def test_dead_branch_charon_low_hp_deferred_burning_pact_generates_before_ashes_kill(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80, energy=3, base_energy=3),
            relics=[make_relic("Dead Branch"), make_relic("Charon's Ashes")],
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": ["Clash"],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": [],
                "SRC_CURSE": [],
            },
        )
        engine.state.monsters[0].current_hp = 3
        engine.state.hand = [
            make_card("Burning Pact", uuid="burning-pact"),
            make_card("Strike_R", uuid="burning-target"),
            make_card("Defend_R", uuid="burning-keep"),
        ]
        engine.state.draw_pile = []
        engine.player.energy = 3

        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Burning Pact"))
        engine.step(next(action for action in engine.legal_actions() if action.get("target_index") == 0))

        self.assertIn("Clash", [card["card_id"] for card in engine.state.hand])
        self.assertEqual(engine.state.monsters[0].current_hp, 0)
        self.assertEqual(engine.outcome, "VICTORY")

    def test_charons_ashes_damages_all_enemies_on_exhaust(self):
        combat = NativeCombatEnv(
            seed=869,
            ascension_level=0,
            encounter_name="3 Cultists",
            relics=[make_relic("Charon's Ashes")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        before = [monster.current_hp for monster in combat.engine.state.monsters]
        combat.engine._exhaust_card(make_card("Strike_R", uuid="ashes-exhaust"))
        after = [monster.current_hp for monster in combat.engine.state.monsters]
        self.assertEqual([b - a for b, a in zip(before, after)], [3, 3, 3])

    def test_magnetism_hand_overflow_sends_generated_card_to_discard(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": [],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": ["Panacea"],
                "SRC_CURSE": [],
            },
        )
        combat_engine_mod._add_power(engine.player, "Magnetism", 1)
        engine.state.hand = [make_card("Defend_R", uuid=f"magnetism-filler-{index}") for index in range(10)]
        engine.state.draw_pile = []
        engine.state.discard_pile = []

        engine._start_player_turn()

        self.assertEqual(len(engine.state.hand), 10)
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Panacea"])

    def test_stacked_magnetism_hand_overflow_fills_slots_then_discards_excess(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": [],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": ["Panacea"],
                "SRC_CURSE": [],
            },
        )
        combat_engine_mod._add_power(engine.player, "Magnetism", 3)
        engine.state.hand = [make_card("Defend_R", uuid=f"magnetism-filler-{index}") for index in range(8)]
        engine.state.draw_pile = []
        engine.state.discard_pile = []

        engine._start_player_turn()

        self.assertEqual(len(engine.state.hand), 10)
        self.assertEqual(sum(1 for card in engine.state.hand if card["card_id"] == "Panacea"), 2)
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Panacea"])

    def test_bird_faced_urn_heals_on_power_play(self):
        player = PlayerState(current_hp=50, max_hp=80)
        combat = NativeCombatEnv(
            seed=870,
            ascension_level=0,
            encounter_name="Cultist",
            player=player,
            relics=[make_relic("Bird Faced Urn")],
            master_deck=[make_card("Inflame", uuid="inflame")],
        )
        combat.engine.state.hand = [make_card("Inflame", uuid="urn-power")]
        combat.engine.player.energy = 3
        action = next(candidate for candidate in combat.engine.legal_actions() if candidate.get("card_id") == "Inflame")
        combat.engine.step(action)
        self.assertEqual(combat.engine.player.current_hp, 52)

    def test_mummified_hand_sets_random_hand_card_cost_to_zero_after_power(self):
        combat = NativeCombatEnv(
            seed=871,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Mummified Hand")],
            master_deck=[make_card("Inflame", uuid="inflame"), make_card("Bash", uuid="bash"), make_card("Strike_R", uuid="strike")],
        )
        combat.engine.state.hand = [
            make_card("Inflame", uuid="mh-power"),
            make_card("Bash", uuid="mh-bash"),
            make_card("Strike_R", uuid="mh-strike"),
        ]
        combat.engine.player.energy = 3
        action = next(candidate for candidate in combat.engine.legal_actions() if candidate.get("card_id") == "Inflame")
        combat.engine.step(action)
        eligible_costs = {card["card_id"]: int(card.get("cost_for_turn", card.get("cost", 0)) or 0) for card in combat.engine.state.hand}
        self.assertTrue(any(cost == 0 for cost in eligible_costs.values()))

    def test_mummified_hand_discount_resets_when_played_card_enters_discard(self):
        combat = NativeCombatEnv(
            seed=872,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Mummified Hand")],
            master_deck=[make_card("Inflame", uuid="inflame"), make_card("Defend_R", uuid="defend")],
        )
        combat.engine.state.hand = [make_card("Inflame", uuid="mh-power"), make_card("Defend_R", uuid="mh-defend")]
        combat.engine.player.energy = 3

        power_action = next(candidate for candidate in combat.engine.legal_actions() if candidate.get("card_id") == "Inflame")
        combat.engine.step(power_action)
        discounted = next(card for card in combat.engine.state.hand if card.get("card_id") == "Defend_R")
        self.assertEqual(int(discounted.get("cost_for_turn", -1)), 0)

        defend_action = next(candidate for candidate in combat.engine.legal_actions() if candidate.get("card_id") == "Defend_R")
        combat.engine.step(defend_action)

        self.assertEqual(len(combat.engine.state.discard_pile), 1)
        self.assertEqual(combat.engine.state.discard_pile[0]["card_id"], "Defend_R")
        self.assertEqual(int(combat.engine.state.discard_pile[0].get("cost_for_turn", -1)), 1)

    def test_boot_raises_small_attack_damage_to_five(self):
        combat = NativeCombatEnv(
            seed=872,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Boot")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        monster = combat.engine.state.monsters[0]
        before = monster.current_hp
        dealt = combat.engine._player_deal_damage(monster, 4)
        self.assertEqual(dealt, 5)
        self.assertEqual(before - monster.current_hp, 5)

    def test_boot_raises_small_unblocked_attack_damage_after_block(self):
        combat = NativeCombatEnv(
            seed=872,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Boot")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        monster = combat.engine.state.monsters[0]
        monster.block = 7
        before = monster.current_hp

        dealt = combat.engine._player_deal_damage(monster, 9)

        self.assertEqual(monster.block, 0)
        self.assertEqual(dealt, 5)
        self.assertEqual(before - monster.current_hp, 5)

    def test_strike_dummy_adds_three_damage_to_strike_cards(self):
        combat = NativeCombatEnv(
            seed=872,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("StrikeDummy")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        combat.engine.state.hand = [make_card("Strike_R", uuid="strike")]
        combat.engine.player.energy = 3
        monster = combat.engine.state.monsters[0]
        before = monster.current_hp

        action = next(candidate for candidate in combat.engine.legal_actions() if candidate.get("card_id") == "Strike_R")
        combat.engine.step(action)

        self.assertEqual(before - monster.current_hp, 9)

    def test_paper_frog_increases_vulnerable_damage_against_monsters(self):
        combat = NativeCombatEnv(
            seed=873,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Paper Frog")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        monster = combat.engine.state.monsters[0]
        monster.powers.append({"power_id": "Vulnerable", "id": "Vulnerable", "name": "Vulnerable", "amount": 1, "misc": 1})
        before = monster.current_hp
        combat.engine._resolve_generated_card_play(make_card("Strike_R", uuid="paper-frog"))
        self.assertEqual(before - monster.current_hp, 10)

    def test_weak_and_vulnerable_damage_floors_after_combined_multiplier(self):
        combat = NativeCombatEnv(
            seed=8731,
            ascension_level=0,
            encounter_name="Cultist",
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        monster = combat.engine.state.monsters[0]
        monster.powers.append({"power_id": "Vulnerable", "id": "Vulnerable", "name": "Vulnerable", "amount": 1, "misc": 1})
        damage = combat.engine._scale_player_attack_damage(9, monster, attacker_weak=True)
        self.assertEqual(damage, 10)

    def test_pen_nib_applies_before_weak_and_target_vulnerable(self):
        combat = NativeCombatEnv(
            seed=8731,
            ascension_level=0,
            encounter_name="Cultist",
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        monster = combat.engine.state.monsters[0]
        monster.powers.append({"power_id": "Vulnerable", "id": "Vulnerable", "name": "Vulnerable", "amount": 1, "misc": 1})
        combat_engine_mod._append_power(combat.engine.player, "Pen Nib", 1, misc=1)
        combat.engine._double_attack_damage = True

        damage = combat.engine._scale_player_attack_damage(13, monster, attacker_weak=True)

        self.assertEqual(damage, 29)

    def test_paper_crane_reduces_enemy_weak_attack_to_sixty_percent(self):
        combat = NativeCombatEnv(
            seed=874,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Paper Crane")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        monster = combat.engine.state.monsters[0]
        monster.powers.append({"power_id": "Weak", "id": "Weak", "name": "Weak", "amount": 1, "misc": 1})
        monster.next_move = "DARK_STRIKE"
        combat.engine._update_monster_intents()
        self.assertEqual(monster.move_adjusted_damage, 3)

    def test_odd_mushroom_reduces_player_vulnerable_damage_multiplier(self):
        combat = NativeCombatEnv(
            seed=8741,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Odd Mushroom")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        monster = combat.engine.state.monsters[0]
        combat.engine.player.powers.append({"power_id": "Vulnerable", "id": "Vulnerable", "name": "Vulnerable", "amount": 1, "misc": 1})
        before = combat.engine.player.current_hp
        dealt = combat.engine._damage_player(10, attacker=monster)
        self.assertEqual(dealt, 12)
        self.assertEqual(before - combat.engine.player.current_hp, 12)

    def test_monster_intent_preview_includes_player_vulnerable_multiplier(self):
        combat = NativeCombatEnv(
            seed=8742,
            ascension_level=0,
            encounter_name="Red Slaver",
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        red_slaver = next(monster for monster in combat.engine.state.monsters if monster.monster_id == "SlaverRed")
        red_slaver.next_move = "STAB"
        combat.engine.player.powers.append({"power_id": "Vulnerable", "id": "Vulnerable", "name": "Vulnerable", "amount": 1, "misc": 1})

        combat.engine._update_monster_intents()

        self.assertEqual(red_slaver.move_adjusted_damage, 19)

    def test_expired_player_vulnerable_is_removed_before_next_monster_intent_preview(self):
        combat = NativeCombatEnv(
            seed=8748,
            ascension_level=0,
            encounter_name="Red Slaver",
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        red_slaver = next(monster for monster in combat.engine.state.monsters if monster.monster_id == "SlaverRed")
        red_slaver.next_move = "STAB"
        combat.engine.player.current_hp = 23
        combat.engine.player.energy = 0
        combat.engine.player.powers.append({"power_id": "Vulnerable", "id": "Vulnerable", "name": "Vulnerable", "amount": 1, "misc": 1})
        combat.engine.state.hand = []

        combat.engine.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(combat.engine.player.current_hp, 4)
        self.assertEqual(_power_amount(combat.engine.player, "Vulnerable"), 0)
        red_slaver.next_move = "SCRAPE"
        combat.engine._update_monster_intents()
        self.assertEqual(red_slaver.move_adjusted_damage, 8)

    def test_player_vulnerable_persists_through_enemy_turn_then_ticks_down(self):
        combat = NativeCombatEnv(
            seed=8744,
            ascension_level=0,
            encounter_name="Red Slaver",
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        red_slaver = next(monster for monster in combat.engine.state.monsters if monster.monster_id == "SlaverRed")
        red_slaver.next_move = "STAB"
        combat.engine.player.current_hp = 23
        combat.engine.player.energy = 0
        combat.engine.player.powers.append({"power_id": "Vulnerable", "id": "Vulnerable", "name": "Vulnerable", "amount": 1, "misc": 1})
        combat.engine.state.hand = []
        combat.engine._update_monster_intents()

        combat.engine.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(combat.engine.player.current_hp, 4)
        self.assertEqual(_power_amount(combat.engine.player, "Vulnerable"), 0)

    def test_monster_applied_weak_persists_into_next_player_turn(self):
        combat = NativeCombatEnv(
            seed=8745,
            ascension_level=0,
            encounter_name="Cultist",
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        combat.engine.player.energy = 0
        combat.engine.state.hand = []
        combat.engine.player.powers.append(
            {"power_id": "Weakened", "id": "Weakened", "name": "Weakened", "amount": 1, "misc": 1, "just_applied": True}
        )

        combat.engine.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(_power_amount(combat.engine.player, "Weakened"), 1)

        combat.engine.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(_power_amount(combat.engine.player, "Weakened"), 0)

    def test_acid_slime_large_weak_lick_applies_two_weak_next_turn(self):
        engine = CombatEngine(
            encounter_name="Large Slime",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        slime = next(monster for monster in engine.state.monsters if monster.monster_id == "AcidSlime_L")
        slime.next_move = "WEAK_LICK"
        slime.intent = "DEBUFF"
        engine.state.hand = []
        engine.player.energy = 0
        engine.step({"kind": "end", "name": "END_TURN"})
        self.assertEqual(_power_amount(engine.player, "Weakened"), 2)

    def test_entangle_clears_at_end_of_player_turn(self):
        combat = NativeCombatEnv(
            seed=8746,
            ascension_level=0,
            encounter_name="Cultist",
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        combat.engine.player.energy = 0
        combat.engine.state.hand = []
        combat.engine.player.powers.append(
            {"power_id": "Entangled", "id": "Entangled", "name": "Entangled", "amount": 1, "misc": 1}
        )

        combat.engine.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(_power_amount(combat.engine.player, "Entangled"), 0)

    def test_monster_applied_entangle_persists_into_next_player_turn(self):
        combat = NativeCombatEnv(
            seed=8747,
            ascension_level=0,
            encounter_name="Red Slaver",
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        red_slaver = next(monster for monster in combat.engine.state.monsters if monster.monster_id == "SlaverRed")
        red_slaver.next_move = "ENTANGLE"
        combat.engine.player.energy = 0
        combat.engine.state.hand = []

        combat.engine.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(_power_amount(combat.engine.player, "Entangled"), 1)
        actions = combat.engine.legal_actions()
        self.assertFalse(any(candidate.get("card_id") == "Bash" for candidate in actions))
        self.assertFalse(any(candidate.get("card_id") == "Strike_R" for candidate in actions))

    def test_ginger_and_turnip_block_player_debuffs(self):
        ginger = NativeCombatEnv(
            seed=8742,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Ginger")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        ginger.engine._apply_player_debuff("Weak", 2)
        ginger.engine._apply_player_debuff("Weakened", 1)
        self.assertEqual(_power_amount(ginger.engine.player, "Weak"), 0)
        self.assertEqual(_power_amount(ginger.engine.player, "Weakened"), 0)

        turnip = NativeCombatEnv(
            seed=8743,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Turnip")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        turnip.engine._apply_player_debuff("Frail", 2)
        self.assertEqual(_power_amount(turnip.engine.player, "Frail"), 0)

    def test_player_artifact_blocks_debuff_and_berserk_self_vulnerable(self):
        combat = NativeCombatEnv(
            seed=8744,
            ascension_level=0,
            encounter_name="Cultist",
            master_deck=[make_card("Berserk", uuid="berserk")],
        )
        combat.engine.player.powers.append({"power_id": "Artifact", "id": "Artifact", "name": "Artifact", "amount": 1, "misc": 1})
        combat.engine._apply_player_debuff("Vulnerable", 2)
        self.assertEqual(_power_amount(combat.engine.player, "Artifact"), 0)
        self.assertEqual(_power_amount(combat.engine.player, "Vulnerable"), 0)

        combat.engine.player.powers.append({"power_id": "Artifact", "id": "Artifact", "name": "Artifact", "amount": 1, "misc": 1})
        combat.engine.state.hand = [make_card("Berserk", uuid="berserk-self")]
        combat.engine.player.energy = 3
        action = next(candidate for candidate in combat.engine.legal_actions() if candidate.get("card_id") == "Berserk")
        combat.engine.step(action)
        self.assertEqual(_power_amount(combat.engine.player, "Artifact"), 0)
        self.assertEqual(_power_amount(combat.engine.player, "Vulnerable"), 0)
        self.assertEqual(_power_amount(combat.engine.player, "Berserk"), 1)

    def test_akabeko_grants_vigor_to_first_attack_only(self):
        combat = NativeCombatEnv(
            seed=9,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Akabeko")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        self.assertEqual(_power_amount(combat.engine.player, "Vigor"), 8)
        first_hp = monster.current_hp
        combat.engine._resolve_generated_card_play(make_card("Strike_R", uuid="akabeko-1"))
        self.assertEqual(first_hp - monster.current_hp, 14)
        self.assertEqual(_power_amount(combat.engine.player, "Vigor"), 0)
        second_hp = monster.current_hp
        combat.engine._resolve_generated_card_play(make_card("Strike_R", uuid="akabeko-2"))
        self.assertEqual(second_hp - monster.current_hp, 6)

    def test_mercury_hourglass_damages_enemies_on_opening_and_turn_start(self):
        combat = NativeCombatEnv(
            seed=10,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Mercury Hourglass")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        opening_hp = monster.current_hp
        self.assertEqual(monster.max_hp - opening_hp, 3)
        combat.engine._start_player_turn()
        self.assertEqual(opening_hp - monster.current_hp, 3)

    def test_attack_counter_relics_follow_source_thresholds(self):
        relics = [
            make_relic("Nunchaku"),
            make_relic("Kunai"),
            make_relic("Shuriken"),
            make_relic("Ornamental Fan"),
        ]
        combat = NativeCombatEnv(
            seed=11,
            ascension_level=0,
            encounter_name="Cultist",
            relics=relics,
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        combat.engine.player.energy = 0
        for idx in range(3):
            combat.engine._resolve_generated_card_play(make_card("Strike_R", uuid=f"counter-{idx}"))
        self.assertEqual(_power_amount(combat.engine.player, "Dexterity"), 1)
        self.assertEqual(_power_amount(combat.engine.player, "Strength"), 1)
        self.assertEqual(combat.engine.player.block, 4)
        self.assertEqual(next(r for r in combat.engine.relics if r["relic_id"] == "Kunai")["counter"], 0)
        self.assertEqual(next(r for r in combat.engine.relics if r["relic_id"] == "Shuriken")["counter"], 0)
        self.assertEqual(next(r for r in combat.engine.relics if r["relic_id"] == "Ornamental Fan")["counter"], 0)
        for idx in range(3, 10):
            combat.engine._resolve_generated_card_play(make_card("Strike_R", uuid=f"counter-{idx}"))
        self.assertEqual(combat.engine.player.energy, 1)
        self.assertEqual(next(r for r in combat.engine.relics if r["relic_id"] == "Nunchaku")["counter"], 0)

    def test_letter_opener_damages_all_enemies_every_third_skill(self):
        combat = NativeCombatEnv(
            seed=12,
            ascension_level=0,
            encounter_name="3 Cultists",
            relics=[make_relic("Letter Opener")],
            master_deck=[make_card("Defend_R", uuid=f"defend-{i}") for i in range(10)],
        )
        starting_hps = [monster.current_hp for monster in combat.engine.state.monsters]
        for idx in range(3):
            combat.engine._resolve_generated_card_play(make_card("Defend_R", uuid=f"skill-{idx}"))
        ending_hps = [monster.current_hp for monster in combat.engine.state.monsters]
        self.assertEqual([before - after for before, after in zip(starting_hps, ending_hps)], [5, 5, 5])
        self.assertEqual(next(r for r in combat.engine.relics if r["relic_id"] == "Letter Opener")["counter"], 0)

    def test_ink_bottle_draws_on_tenth_card_played(self):
        combat = NativeCombatEnv(
            seed=13,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("InkBottle")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        starting_hand = len(combat.engine.state.hand)
        for idx in range(10):
            combat.engine._resolve_generated_card_play(make_card("Strike_R", uuid=f"ink-{idx}"))
        self.assertEqual(len(combat.engine.state.hand), starting_hand + 1)
        self.assertEqual(next(r for r in combat.engine.relics if r["relic_id"] == "InkBottle")["counter"], 0)

    def test_ink_bottle_draw_waits_for_discovery_choice(self):
        ink_bottle = make_relic("InkBottle")
        ink_bottle["counter"] = 9
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=16863661471043244345),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            relics=[ink_bottle],
            master_deck=[],
        )
        engine.state.hand = [make_card("Discovery", uuid="discovery"), make_card("Defend_R", uuid="defend")]
        engine.state.draw_pile = [make_card("Bash", uuid="draw-bash")]
        engine.state.discard_pile = []
        engine.player.energy = 3

        engine.step({"kind": "card", "card_index": 0, "name": "Discovery", "card_id": "Discovery"})

        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Defend_R"])
        self.assertEqual([card["card_id"] for card in engine.state.draw_pile], ["Bash"])
        self.assertEqual(next(r for r in engine.relics if r["relic_id"] == "InkBottle")["counter"], 0)

        selected_card_id = engine.pending_card_select["cards"][0]["card_id"]
        engine.step({"kind": "card_reward", "choice_index": 0})

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Defend_R", selected_card_id, "Bash"])
        self.assertEqual(next(r for r in engine.relics if r["relic_id"] == "InkBottle")["counter"], 0)

    def test_discovery_choice_resets_temporary_zero_cost_after_play(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=38),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = []
        engine.pending_card_select = {
            "mode": "DISCOVERY",
            "cards": [make_card("Uppercut", uuid="discovery-uppercut")],
            "num_cards": 1,
            "copies": 1,
        }
        engine.player.energy = 3

        engine.step({"kind": "card_reward", "choice_index": 0})

        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Uppercut"])
        self.assertEqual(engine.state.hand[0]["cost"], 2)
        self.assertEqual(engine.state.hand[0]["cost_for_turn"], 0)

        engine.step({"kind": "card", "card_index": 0, "target_index": 0, "card_id": "Uppercut"})

        discarded = next(card for card in engine.state.discard_pile if card["card_id"] == "Uppercut")
        self.assertEqual(discarded["cost"], 2)
        self.assertEqual(discarded["cost_for_turn"], 2)

    def test_discovery_choice_keeps_cost_reset_marker_through_armaments_upgrade(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=38),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Armaments", upgrades=1, uuid="armaments-plus")]
        engine.pending_card_select = {
            "mode": "DISCOVERY",
            "cards": [make_card("Fiend Fire", uuid="discovery-fiend-fire")],
            "num_cards": 1,
            "copies": 1,
        }
        engine.player.energy = 3

        engine.step({"kind": "card_reward", "choice_index": 0})
        fiend_fire = next(card for card in engine.state.hand if card["card_id"] == "Fiend Fire")
        self.assertEqual(fiend_fire["cost_for_turn"], 0)

        engine.step({"kind": "card", "card_index": 0, "card_id": "Armaments"})
        upgraded_fiend_fire = next(card for card in engine.state.hand if card["card_id"] == "Fiend Fire")
        self.assertEqual(upgraded_fiend_fire["upgrades"], 1)
        self.assertEqual(upgraded_fiend_fire["cost_for_turn"], 0)

        engine.step({"kind": "card", "card_index": 0, "target_index": 0, "card_id": "Fiend Fire"})

        exhausted = next(card for card in engine.state.exhaust_pile if card["card_id"] == "Fiend Fire")
        self.assertEqual(exhausted["cost"], 2)
        self.assertEqual(exhausted["cost_for_turn"], 2)

    def test_nilrys_codex_opens_card_reward_before_monster_turn(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=38),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            relics=[make_relic("Nilry's Codex")],
            master_deck=[],
        )
        engine.state.hand = [make_card("Strike_R", uuid="held-strike")]
        engine.player.energy = 0

        engine.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(engine.player.current_hp, 80)
        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.pending_card_select["mode"], "NILRYS_CODEX")
        actions = engine.legal_actions()
        self.assertEqual([action["kind"] for action in actions[:-1]], ["card_reward", "card_reward", "card_reward"])
        self.assertEqual(actions[-1]["kind"], "skip")

    def test_nilrys_codex_choice_adds_card_to_random_draw_pile_spot(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=38),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.draw_pile = [make_card("Strike_R", uuid="draw-strike")]
        engine.pending_card_select = {
            "mode": "NILRYS_CODEX",
            "cards": [make_card("Uppercut", uuid="codex-uppercut")],
            "num_cards": 1,
            "can_skip": True,
        }

        engine.step({"kind": "card_reward", "choice_index": 0})

        self.assertIsNone(engine.pending_card_select)
        self.assertIn("Uppercut", [card["card_id"] for card in engine.state.draw_pile])

    def test_pocketwatch_draws_three_post_draw_after_low_card_turn(self):
        combat = NativeCombatEnv(
            seed=14,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Pocketwatch")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        pocketwatch = next(r for r in combat.engine.relics if r["relic_id"] == "Pocketwatch")
        combat.engine._start_player_turn()
        self.assertFalse(bool(pocketwatch.get("first_turn", False)))
        combat.engine.state.hand = []
        combat.engine.state.draw_pile = [make_card("Strike_R", uuid=f"pw-draw-{i}") for i in range(20)]
        pocketwatch["counter"] = 3
        combat.engine._start_player_turn()
        self.assertEqual(len(combat.engine.state.hand), 8)
        self.assertEqual(pocketwatch["counter"], 0)

    def test_art_of_war_grants_energy_if_previous_turn_had_no_attacks(self):
        combat = NativeCombatEnv(
            seed=15,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Art of War")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        art = next(r for r in combat.engine.relics if r["relic_id"] == "Art of War")
        combat.engine._start_player_turn()
        self.assertFalse(bool(art.get("first_turn", False)))
        combat.engine._start_player_turn()
        self.assertEqual(combat.engine.player.energy, 4)
        combat.engine._resolve_generated_card_play(make_card("Strike_R", uuid="art-attack"))
        self.assertFalse(bool(art.get("gain_energy_next", True)))

    def test_captains_wheel_grants_block_on_third_turn_start(self):
        combat = NativeCombatEnv(
            seed=16,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("CaptainsWheel")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )
        wheel = next(r for r in combat.engine.relics if r["relic_id"] == "CaptainsWheel")
        self.assertEqual(wheel["counter"], 1)
        combat.engine._start_player_turn()
        self.assertEqual(combat.engine.player.block, 0)
        self.assertEqual(wheel["counter"], 2)
        combat.engine._start_player_turn()
        self.assertEqual(combat.engine.player.block, 18)
        self.assertEqual(wheel["counter"], -1)
        self.assertTrue(wheel["grayscale"])

    def test_sundial_grants_energy_every_third_shuffle(self):
        combat = NativeCombatEnv(
            seed=17,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Sundial")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        sundial = next(r for r in combat.engine.relics if r["relic_id"] == "Sundial")
        sundial["counter"] = 2
        combat.engine.player.energy = 0
        combat.engine.state.hand = []
        combat.engine.state.draw_pile = []
        combat.engine.state.discard_pile = [make_card("Strike_R", uuid=f"sundial-{i}") for i in range(5)]
        combat.engine.draw_cards(1)
        self.assertEqual(combat.engine.player.energy, 2)
        self.assertEqual(sundial["counter"], 0)

    def test_sundial_does_not_count_opening_shuffle(self):
        combat = NativeCombatEnv(
            seed=170,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Sundial")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        sundial = next(r for r in combat.engine.relics if r["relic_id"] == "Sundial")
        self.assertEqual(sundial["counter"], 0)

    def test_stone_calendar_damages_all_enemies_on_seventh_turn_end(self):
        combat = NativeCombatEnv(
            seed=18,
            ascension_level=0,
            encounter_name="3 Cultists",
            relics=[make_relic("StoneCalendar")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        stone = next(r for r in combat.engine.relics if r["relic_id"] == "StoneCalendar")
        self.assertEqual(stone["counter"], 1)
        stone["counter"] = 7
        combat.engine._apply_player_end_of_turn_powers()
        ending_hps = [monster.current_hp for monster in combat.engine.state.monsters]
        self.assertEqual(ending_hps, [0, 0, 0])
        self.assertEqual(stone["counter"], -1)

    def test_stone_calendar_resets_counter_on_victory_without_triggering(self):
        combat = NativeCombatEnv(
            seed=181,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("StoneCalendar")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        stone = next(r for r in combat.engine.relics if r["relic_id"] == "StoneCalendar")
        stone["counter"] = 3

        combat.engine._handle_victory_relics()

        self.assertEqual(stone["counter"], -1)

    def test_stone_calendar_advances_once_per_player_turn(self):
        combat = NativeCombatEnv(
            seed=182,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("StoneCalendar")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        stone = next(r for r in combat.engine.relics if r["relic_id"] == "StoneCalendar")

        combat.engine._start_player_turn()

        self.assertEqual(stone["counter"], 2)

    def test_orange_pellets_clears_player_debuffs_after_attack_skill_power(self):
        combat = NativeCombatEnv(
            seed=19,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("OrangePellets")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        combat.engine.player.powers.append({"power_id": "Weak", "id": "Weak", "name": "Weak", "amount": 2, "misc": 2})
        combat.engine.player.powers.append({"power_id": "Vulnerable", "id": "Vulnerable", "name": "Vulnerable", "amount": 2, "misc": 2})
        combat.engine._resolve_generated_card_play(make_card("Strike_R", uuid="pellets-attack"))
        combat.engine._resolve_generated_card_play(make_card("Defend_R", uuid="pellets-skill"))
        combat.engine._resolve_generated_card_play(make_card("Inflame", uuid="pellets-power"))
        self.assertEqual(_power_amount(combat.engine.player, "Weak"), 0)
        self.assertEqual(_power_amount(combat.engine.player, "Vulnerable"), 0)

    def test_champion_belt_applies_weak_when_vulnerable_lands(self):
        combat = NativeCombatEnv(
            seed=20,
            ascension_level=0,
            encounter_name="Cultist",
            relics=[make_relic("Champion Belt")],
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(10)],
        )
        monster = combat.engine.state.monsters[0]
        combat.engine._apply_monster_debuff(monster, "Vulnerable", 2)
        self.assertEqual(_power_amount(monster, "Vulnerable"), 2)
        self.assertEqual(_power_amount(monster, "Weak"), 1)

    def test_treasure_room_consumes_relic_pool_front(self):
        env = NativeRunEnv(seed=4, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "TREASURE"
        env.current_treasure = generate_treasure(env.randoms)
        env.current_treasure.relic_tier = "COMMON"
        env.relic_pools["COMMON"] = ["Anchor", "Bag of Marbles"]
        env._step_treasure({"kind": "treasure", "name": "OPEN_CHEST"})
        self.assertEqual([relic["relic_id"] for relic in env.reward_relics], ["Anchor"])
        self.assertEqual(env.relic_pools["COMMON"], ["Bag of Marbles"])

    def test_elite_reward_consumes_relic_pool_front(self):
        env = NativeRunEnv(seed=5, ascension_level=0, enable_neow=False, start_on_map=True)
        env.floor = 6
        env.phase = "COMBAT"
        env.current_room_type = "MonsterRoomElite"
        env.relic_pools["COMMON"] = ["Anchor", "Bag of Preparation"]
        env.relic_pools["UNCOMMON"] = ["HornCleat"]
        env.relic_pools["RARE"] = ["Calipers"]

        class _EliteVictoryCombat:
            def __init__(self, player):
                self.player = player
                self.encounter_name = "Gremlin Nob"

            def step(self, _action):
                return "VICTORY"

        env.combat = _EliteVictoryCombat(env.player)
        env._step_combat({"kind": "end"})
        self.assertTrue(env.reward_relics)
        self.assertNotEqual(env.reward_relics[0]["relic_id"], "Circlet")

    def test_shop_generation_consumes_shop_pool(self):
        env = NativeRunEnv(seed=6, ascension_level=0, enable_neow=False, start_on_map=True)
        env.relic_pools["SHOP"] = ["ClockworkSouvenir"]
        env.relic_pools["COMMON"] = ["Anchor", "Bag of Marbles"]
        env.relic_pools["UNCOMMON"] = ["HornCleat", "Letter Opener"]
        env.relic_pools["RARE"] = ["Calipers"]
        env.current_shop = generate_shop(
            env.randoms,
            ascension_level=env.ascension_level,
            owned_relic_ids={str(relic.get("relic_id") or relic.get("id")) for relic in env.relics},
            relic_drawer=env._pop_relic_from_pool,
        )
        relic_ids = [str(relic["relic_id"]) for relic in env.current_shop.relics]
        self.assertIn("ClockworkSouvenir", relic_ids)
        self.assertNotIn("ClockworkSouvenir", env.relic_pools["SHOP"])

    def test_shop_generation_consumes_common_relic_pool_from_end(self):
        env = NativeRunEnv(seed=6, ascension_level=0, enable_neow=False, start_on_map=True)
        env.relic_pools["SHOP"] = ["ClockworkSouvenir"]
        env.relic_pools["COMMON"] = ["Anchor", "Bag of Marbles"]
        env.relic_pools["UNCOMMON"] = ["HornCleat", "Letter Opener"]
        env.relic_pools["RARE"] = ["Calipers"]
        env.current_shop = generate_shop(
            env.randoms,
            act=env.act,
            floor_num=env.floor,
            ascension_level=env.ascension_level,
            owned_relic_ids={str(relic.get("relic_id") or relic.get("id")) for relic in env.relics},
            relic_drawer=env._pop_relic_end_from_pool,
        )
        relic_ids = [str(relic["relic_id"]) for relic in env.current_shop.relics]
        self.assertEqual(relic_ids[:2], ["Bag of Marbles", "Anchor"])
        self.assertEqual(env.relic_pools["COMMON"], [])

    def test_post_combat_burning_blood_heals_before_reward_screen(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.floor = 1
        env.phase = "COMBAT"
        env.player.current_hp = 74

        class _VictoryCombat:
            def __init__(self, player):
                self.player = player
                self.encounter_name = "Cultist"

            def step(self, _action):
                return "VICTORY"

        env.combat = _VictoryCombat(env.player)
        env._step_combat({"kind": "end"})
        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(env.player.current_hp, 80)

    def test_pommel_strike_deals_damage_and_draws(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Pommel Strike", uuid="pommel")]
        engine.state.draw_pile = [make_card("Strike_R", uuid="draw-target")]
        engine.player.energy = 3
        before_hp = engine.state.monsters[0].current_hp
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Pommel Strike")
        engine.step(action)
        self.assertLess(engine.state.monsters[0].current_hp, before_hp)
        self.assertTrue(any(card["card_id"] == "Strike_R" for card in engine.state.hand))

    def test_disarm_applies_negative_strength(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Disarm", uuid="disarm")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Disarm")
        engine.step(action)
        self.assertLessEqual(_power_amount(engine.state.monsters[0], "Strength"), -2)

    def test_looter_attack_uses_current_negative_strength(self):
        engine = CombatEngine(
            encounter_name="Looter",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=37, max_hp=80),
            master_deck=[],
        )
        looter = engine.state.monsters[0]
        looter.powers.append({"power_id": "Strength", "id": "Strength", "name": "Strength", "amount": -2})
        looter.next_move = "MUG"
        looter.meta["swipe_damage"] = 10
        engine.player.block = 6
        engine.state.hand = []
        engine.step({"kind": "end"})

        self.assertEqual(engine.player.current_hp, 35)

    def test_fiend_fire_exhausts_hand_by_card_random_rng(self):
        engine = CombatEngine(
            encounter_name="2 Thieves",
            randoms=NativeRandomSet(seed=12, act=2, floor=18),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80, energy=4, base_energy=4),
            master_deck=[],
            gold=316,
        )
        engine.state.hand = [
            make_card("Fiend Fire", uuid="fiend"),
            make_card("Clothesline", uuid="clothesline"),
            make_card("Demon Form", uuid="demon-form"),
            make_card("Intimidate", uuid="intimidate"),
            make_card("Evolve", uuid="evolve"),
        ]
        engine.player.energy = 4

        engine.step({"kind": "card", "card_index": 0, "target_index": 0})

        self.assertEqual(
            [card["card_id"] for card in engine.state.exhaust_pile],
            ["Clothesline", "Evolve", "Demon Form", "Intimidate", "Fiend Fire"],
        )
        self.assertEqual(engine.state.monsters[0].current_hp, 19)

    def test_fiend_fire_does_not_exhaust_deferred_dead_branch_cards(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80, energy=4, base_energy=4),
            master_deck=[],
            relics=[make_relic("Dead Branch")],
            source_card_pools={
                "SRC_COMMON": ["Clash"],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": [],
                "SRC_CURSE": [],
            },
        )
        engine.state.hand = [
            make_card("Fiend Fire", uuid="fiend"),
            make_card("Strike_R", uuid="strike"),
            make_card("Defend_R", uuid="defend"),
        ]
        engine.player.energy = 4

        engine.step({"kind": "card", "card_index": 0, "target_index": 0})

        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Clash", "Clash", "Clash"])
        self.assertCountEqual(
            [card["card_id"] for card in engine.state.exhaust_pile],
            ["Strike_R", "Defend_R", "Fiend Fire"],
        )

    def test_hexaghost_sear_uses_current_weak(self):
        engine = CombatEngine(
            encounter_name="Hexaghost",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=43, max_hp=80),
            master_deck=[],
        )
        hexaghost = engine.state.monsters[0]
        hexaghost.next_move = "SEAR"
        hexaghost.meta["sear_damage"] = 6
        hexaghost.powers.append({"power_id": "Weakened", "id": "Weakened", "name": "Weakened", "amount": 3})
        engine.state.hand = []
        engine.step({"kind": "end"})

        self.assertEqual(engine.player.current_hp, 39)

    def test_shrug_it_off_grants_block_and_draws(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Shrug It Off", uuid="shrug")]
        engine.state.draw_pile = [make_card("Strike_R", uuid="draw-target")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Shrug It Off")
        engine.step(action)
        self.assertGreater(engine.player.block, 0)
        self.assertTrue(any(card["card_id"] == "Strike_R" for card in engine.state.hand))

    def test_cleave_hits_all_enemies(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Cleave", uuid="cleave")]
        engine.state.draw_pile = []
        engine.player.energy = 3
        before = [monster.current_hp for monster in engine.state.monsters]
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Cleave")
        engine.step(action)
        after = [monster.current_hp for monster in engine.state.monsters]
        self.assertTrue(all(a < b for a, b in zip(after, before)))

    def test_immolate_hits_all_enemies_and_adds_burn(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Immolate", uuid="immolate")]
        engine.player.energy = 3
        before = [monster.current_hp for monster in engine.state.monsters]
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Immolate")
        engine.step(action)
        after = [monster.current_hp for monster in engine.state.monsters]
        self.assertTrue(all(a < b for a, b in zip(after, before)))
        self.assertTrue(any(card["card_id"] == "Burn" for card in engine.state.discard_pile))

    def test_sword_boomerang_performs_multiple_random_hits(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Sword Boomerang", uuid="boomerang")]
        engine.state.draw_pile = []
        engine.player.energy = 3
        before = sum(monster.current_hp for monster in engine.state.monsters)
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Sword Boomerang")
        engine.step(action)
        after = sum(monster.current_hp for monster in engine.state.monsters)
        self.assertLess(after, before)

    def test_twin_strike_hits_target_twice(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Twin Strike", uuid="twin")]
        engine.state.draw_pile = []
        engine.player.energy = 3
        target = engine.state.monsters[0]
        before_hp = target.current_hp
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Twin Strike")
        engine.step(action)
        self.assertLessEqual(target.current_hp, before_hp - 10)

    def test_twin_strike_delays_louse_curl_up_until_both_hits_resolve(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Twin Strike", uuid="twin")]
        engine.state.draw_pile = []
        engine.player.energy = 3
        target_index = next(
            index
            for index, monster in enumerate(engine.state.monsters)
            if monster.monster_id == "FuzzyLouseNormal"
        )
        target = engine.state.monsters[target_index]
        target.current_hp = 14
        target.max_hp = 14
        target.block = 0
        target.powers = [{"power_id": "Curl Up", "id": "Curl Up", "name": "Curl Up", "amount": 6, "misc": 6}]

        action = next(
            candidate
            for candidate in engine.legal_actions()
            if candidate.get("card_id") == "Twin Strike" and candidate.get("target_index") == target_index
        )
        engine.step(action)

        self.assertEqual(target.current_hp, 4)
        self.assertEqual(target.block, 6)
        self.assertEqual(_power_amount(target, "Curl Up"), 0)

    def test_twin_strike_delays_malleable_block_until_both_hits_resolve(self):
        engine = CombatEngine(
            encounter_name="Snake Plant",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Twin Strike", uuid="twin")]
        engine.state.draw_pile = []
        engine.player.energy = 3
        snake = engine.state.monsters[0]
        snake.current_hp = 33
        snake.block = 3
        for power in snake.powers:
            if power.get("power_id") == "Malleable":
                power["amount"] = 4
                power["misc"] = 3
        engine._apply_monster_debuff(snake, "Vulnerable", 3)
        combat_engine_mod._add_power(engine.player, "Strength", 6)
        combat_engine_mod._add_power(engine.player, "Weakened", 2)

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Twin Strike")
        engine.step(action)

        self.assertEqual(snake.current_hp, 12)
        self.assertEqual(snake.block, 9)
        self.assertEqual(_power_amount(snake, "Malleable"), 6)

    def test_decay_deals_end_of_turn_damage_while_in_hand(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=20, max_hp=20),
            master_deck=[],
        )
        engine.state.hand = [make_card("Decay", uuid="decay-test")]
        engine.state.draw_pile = []
        engine.player.energy = 3
        engine.step({"kind": "end", "name": "END_TURN"})
        self.assertEqual(engine.player.current_hp, 18)

    def test_decay_end_turn_damage_is_blocked_before_hp_loss(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=20, max_hp=20),
            master_deck=[],
        )
        engine.state.hand = [make_card("Decay", uuid="decay-test")]
        engine.player.block = 7
        engine._apply_player_end_of_turn_powers()
        self.assertEqual(engine.player.current_hp, 20)
        self.assertEqual(engine.player.block, 5)

    def test_burn_end_turn_damage_is_blocked_by_metallicize(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=20, max_hp=20),
            master_deck=[],
        )
        engine.state.hand = [make_card("Burn", uuid="burn-test")]
        engine.player.powers = [{"power_id": "Metallicize", "amount": 3, "misc": 3}]
        engine._apply_player_end_of_turn_powers()
        self.assertEqual(engine.player.current_hp, 20)
        self.assertEqual(engine.player.block, 1)

    def test_monster_regenerate_heals_after_monster_turn(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=20, max_hp=20),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.current_hp = int(monster.max_hp) - 5
        monster.next_move = "INCANTATION"
        monster.intent = "BUFF"
        monster.powers = [{"power_id": "Regenerate", "amount": 3, "misc": 3}]
        engine._take_monster_turns()
        self.assertEqual(monster.current_hp, int(monster.max_hp) - 2)

    def test_shop_generation_exposes_model_choice_candidates(self):
        shop = generate_shop(NativeRandomSet(seed=2))
        actions = shop.actions()
        self.assertTrue(any(action["item_kind"] == "card" for action in actions))
        self.assertTrue(any(action["item_kind"] == "relic" for action in actions))
        self.assertTrue(any(action["item_kind"] == "potion" for action in actions))
        self.assertTrue(any(action["item_kind"] == "purge" for action in actions))
        self.assertTrue(any(action["item_kind"] == "leave" for action in actions))
        self.assertTrue(all(str(action.get("item_id")) in relic_catalog() for action in actions if action.get("item_kind") == "relic"))

    def test_shop_screen_state_uses_visible_upgraded_card_names(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True)
        card = make_card("Fire Breathing", uuid="shop-fire-breathing")
        card["upgrades"] = 1
        card["price"] = 36
        env.phase = "SHOP"
        env.current_room_type = "ShopRoom"
        env.current_shop = ShopState(cards=[card], relics=[], potions=[], purge_available=False)

        self.assertEqual(env.current_shop.actions()[0]["name"], "Fire Breathing+")
        self.assertEqual(env.current_shop.actions()[0]["label"], "Fire Breathing+")
        self.assertEqual(env._shop_screen_state()["cards"][0]["name"], "Fire Breathing+")

    def test_shop_potions_use_potion_helper_rolls(self):
        shop = generate_shop(NativeRandomSet(seed=2))
        expected_randoms = NativeRandomSet(seed=2)
        expected = [roll_random_potion(expected_randoms, player_class="IRONCLAD")["potion_id"] for _ in range(3)]
        self.assertEqual([p["potion_id"] for p in shop.potions], expected)

    def test_shop_generation_matches_merchant_shape_and_sale_card(self):
        shop = generate_shop(NativeRandomSet(seed=2))
        self.assertEqual(len(shop.cards), 7)
        self.assertEqual(sum(1 for card in shop.cards if str(card.get("color")) == "COLORLESS"), 2)
        self.assertEqual(sum(1 for card in shop.cards if bool(card.get("on_sale"))), 1)

    def test_shop_rules_are_parsed_from_merchant_and_shop_screen(self):
        rules = shop_rules()
        self.assertEqual(rules.colored_card_types, ("ATTACK", "ATTACK", "SKILL", "SKILL", "POWER"))
        self.assertEqual(rules.colorless_card_rarities, ("UNCOMMON", "RARE"))
        self.assertEqual((rules.card_rare_chance, rules.card_uncommon_chance), (9, 37))
        self.assertEqual((rules.relic_common_cutoff, rules.relic_uncommon_cutoff), (48, 82))
        self.assertEqual(
            (
                rules.card_price_jitter,
                rules.relic_price_jitter,
                rules.potion_price_jitter,
                rules.colorless_price_bump,
                rules.purge_cost,
                rules.purge_cost_ramp,
            ),
            (0.1, 0.05, 0.05, 1.2, 75, 25),
        )

    def test_prices_and_reward_rarity_are_parsed_from_source(self):
        self.assertEqual(
            (
                card_price_by_rarity()["COMMON"],
                card_price_by_rarity()["UNCOMMON"],
                card_price_by_rarity()["RARE"],
            ),
            (50, 75, 150),
        )
        self.assertEqual(
            (
                relic_price_by_tier()["COMMON"],
                relic_price_by_tier()["UNCOMMON"],
                relic_price_by_tier()["RARE"],
                relic_price_by_tier()["SHOP"],
                relic_price_by_tier()["BOSS"],
            ),
            (150, 250, 300, 150, 999),
        )
        self.assertEqual(
            (
                potion_price_by_rarity()["COMMON"],
                potion_price_by_rarity()["UNCOMMON"],
                potion_price_by_rarity()["RARE"],
            ),
            (50, 75, 100),
        )
        rarity_rules = reward_rarity_rules()
        self.assertEqual(
            (
                rarity_rules.rare_roll_threshold,
                rarity_rules.uncommon_roll_threshold,
                rarity_rules.rare_chance,
                rarity_rules.uncommon_chance,
            ),
            (3, 40, 3, 37),
        )

    def test_potion_roll_and_card_blizz_rules_are_parsed_from_source(self):
        potion_rules = potion_roll_rules()
        self.assertEqual(
            (
                potion_rules.common_chance,
                potion_rules.uncommon_chance,
                potion_rules.rare_chance,
            ),
            (65, 25, 10),
        )
        blizz_rules = card_blizz_rules()
        self.assertEqual(
            (
                blizz_rules.start_offset,
                blizz_rules.growth,
                blizz_rules.max_offset,
            ),
            (5, 1, -40),
        )
        post_combat_rules = post_combat_potion_rules()
        self.assertEqual(
            (
                post_combat_rules.base_chance,
                post_combat_rules.white_beast_chance,
                post_combat_rules.reward_cap,
                post_combat_rules.blizzard_mod_amount,
            ),
            (40, 100, 4, 10),
        )

    def test_make_potion_uses_source_price_for_rarity(self):
        potion = draw_random_potion(NativeRandomSet(seed=2), player_class="IRONCLAD")
        self.assertEqual(int(potion["price"]), potion_price_by_rarity()[str(potion["rarity"])])

    def test_shop_generation_uses_runtime_card_pools(self):
        runtime_card_pools = initialize_runtime_card_pools()
        runtime_card_pools["RED_COMMON"] = ["Clash"]
        runtime_card_pools["RED_UNCOMMON"] = []
        runtime_card_pools["RED_RARE"] = []
        runtime_card_pools["COLORLESS_UNCOMMON"] = ["Panacea"]
        runtime_card_pools["COLORLESS_RARE"] = []
        shop = generate_shop(NativeRandomSet(seed=2), runtime_card_pools=runtime_card_pools)
        self.assertTrue(all(card["card_id"] in {"Clash", "Panacea"} for card in shop.cards))

    def test_shop_applies_membership_courier_and_smiling_mask_pricing(self):
        shop = generate_shop(
            NativeRandomSet(seed=2),
            ascension_level=16,
            owned_relic_ids={"Membership Card", "The Courier", "Smiling Mask"},
        )
        self.assertEqual(shop.purge_cost, 50)
        self.assertTrue(all(int(card["price"]) > 0 for card in shop.cards))
        self.assertTrue(all(int(relic["price"]) > 0 for relic in shop.relics))
        self.assertTrue(all(int(potion["price"]) > 0 for potion in shop.potions))

    def test_shop_sale_card_is_discounted_before_courier_discount(self):
        baseline = generate_shop(NativeRandomSet(seed=10))
        courier = generate_shop(NativeRandomSet(seed=10), owned_relic_ids={"The Courier"})
        sale_indexes = [index for index, card in enumerate(baseline.cards) if bool(card.get("on_sale"))]
        self.assertEqual(len(sale_indexes), 1)
        sale_index = sale_indexes[0]
        self.assertEqual([card["card_id"] for card in courier.cards], [card["card_id"] for card in baseline.cards])
        self.assertEqual(int(baseline.cards[sale_index]["price"]), 38)
        self.assertEqual(int(courier.cards[sale_index]["price"]), 30)

    def test_shop_purge_cost_ramps_across_shops(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True)
        env.gold = 500
        env.phase = "SHOP"
        env.current_room_type = "ShopRoom"
        env.current_shop = generate_shop(env.randoms, purge_base_cost=env.shop_purge_base_cost)

        env.step({"kind": "shop", "item_kind": "purge", "target_index": 0})

        self.assertEqual(env.shop_purge_base_cost, 100)
        next_shop = generate_shop(env.randoms, purge_base_cost=env.shop_purge_base_cost)
        self.assertEqual(next_shop.purge_base_cost, 100)
        self.assertEqual(next_shop.purge_cost, 100)

    def test_shop_potion_purchase_is_blocked_by_sozu_without_consuming_offer(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True)
        env.phase = "SHOP"
        env.current_room_type = "ShopRoom"
        env.current_shop = generate_shop(env.randoms)
        env.relics.append(make_relic("Sozu"))
        before_gold = env.gold
        before_shop_potions = [dict(potion) for potion in env.current_shop.potions]
        potion_action = next(action for action in env.current_shop.actions() if action["item_kind"] == "potion")
        env.step(potion_action)
        self.assertEqual(env.gold, before_gold)
        self.assertEqual(len(env.potions), 0)
        self.assertEqual(env.current_shop.potions, before_shop_potions)

    def test_shop_card_purchase_refills_with_courier(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True)
        env.phase = "SHOP"
        env.current_room_type = "ShopRoom"
        env.current_shop = generate_shop(env.randoms)
        env.relics.append(make_relic("The Courier"))
        starting_count = len(env.current_shop.cards)
        target = next(card for card in env.current_shop.cards if env.gold >= int(card["price"]))
        env.step({"kind": "shop", "item_kind": "card", "shop_index": env.current_shop.cards.index(target)})
        self.assertEqual(len(env.current_shop.cards), starting_count)

    def test_full_potion_reward_requires_discard_before_collect(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True)
        env.phase = "CARD_REWARD"
        env.current_room_type = "CARD_REWARD"
        env.reward_potions = [make_potion("SneckoOil")]
        env.potions = [
            make_potion("SteroidPotion"),
            make_potion("LiquidMemories"),
            make_potion("Dexterity Potion"),
        ]

        actions = env.legal_actions()
        self.assertTrue(any(action["kind"] == "reward_potion" for action in actions))
        discard_actions = [action for action in actions if action["kind"] == "discard_potion"]
        self.assertEqual([action["potion_index"] for action in discard_actions], [0, 1, 2])

        env.step({"kind": "reward_potion", "potion_id": "SneckoOil"})
        self.assertEqual([potion["potion_id"] for potion in env.potions], ["SteroidPotion", "LiquidMemories", "Dexterity Potion"])
        self.assertEqual([potion["potion_id"] for potion in env.reward_potions], ["SneckoOil"])

        env.step({"kind": "discard_potion", "potion_index": 2})
        env.step({"kind": "reward_potion", "potion_id": "SneckoOil"})
        self.assertEqual([potion["potion_id"] for potion in env.potions], ["SteroidPotion", "LiquidMemories", "SneckoOil"])
        self.assertEqual(env.reward_potions, [])

    def test_shop_replacement_helpers_use_runtime_pools_and_shop_price_rules(self):
        runtime_card_pools = initialize_runtime_card_pools()
        runtime_card_pools["RED_COMMON"] = ["Clash"]
        runtime_card_pools["RED_UNCOMMON"] = []
        runtime_card_pools["RED_RARE"] = []
        runtime_card_pools["COLORLESS_UNCOMMON"] = ["Panacea"]
        runtime_card_pools["COLORLESS_RARE"] = []
        randoms = NativeRandomSet(seed=11)

        replacement_card = generate_shop_replacement_card(
            randoms,
            purchased_card=make_card("Strike_R"),
            existing_cards=[make_card("Clash")],
            ascension_level=16,
            owned_relic_ids={"Membership Card", "The Courier"},
            runtime_card_pools=runtime_card_pools,
        )
        self.assertEqual(replacement_card["card_id"], "Clash")
        self.assertGreater(int(replacement_card["price"]), 0)
        self.assertEqual(randoms.stream("card").counter, 1)
        self.assertEqual(randoms.stream("misc").counter, 1)

        replacement_potion = generate_shop_replacement_potion(
            randoms,
            shop_index=1,
            ascension_level=16,
            owned_relic_ids={"Membership Card", "The Courier"},
        )
        self.assertEqual(int(replacement_potion["shop_index"]), 1)
        self.assertGreater(int(replacement_potion["price"]), 0)

    def test_shop_colorless_replacement_uses_act_colorless_rare_chance(self):
        runtime_card_pools = initialize_runtime_card_pools()
        runtime_card_pools["COLORLESS_UNCOMMON"] = ["Panacea"]
        runtime_card_pools["COLORLESS_RARE"] = ["HandOfGreed"]
        with patch("spirecomm.native_sim_v3.run.shop.act_chances") as mock_act_chances:
            mock_act_chances.return_value.colorless_rare_chance = 1.0
            replacement = generate_shop_replacement_card(
                NativeRandomSet(seed=19),
                purchased_card=make_card("Panacea"),
                existing_cards=[],
                act=2,
                runtime_card_pools=runtime_card_pools,
            )
        self.assertEqual(replacement["card_id"], "HandOfGreed")

    def test_shop_replacement_relic_uses_injected_relic_drawer(self):
        randoms = NativeRandomSet(seed=12)
        replacement = generate_shop_replacement_relic(
            randoms,
            ascension_level=0,
            owned_relic_ids={"The Courier"},
            relic_drawer=lambda tier: {"relic_id": "Anchor", "name": "Anchor", "tier": tier},
        )
        self.assertEqual(replacement["relic_id"], "Anchor")
        self.assertGreater(int(replacement["price"]), 0)

    def test_shop_replacement_relic_fallback_passes_player_class_to_end_draw(self):
        with patch("spirecomm.native_sim_v3.run.shop.draw_random_relic_end") as mock_draw:
            mock_draw.return_value = {"relic_id": "Anchor", "name": "Anchor", "tier": "COMMON"}
            replacement = generate_shop_replacement_relic(
                NativeRandomSet(seed=12),
                player_class="THE_SILENT",
            )
        self.assertEqual(replacement["relic_id"], "Anchor")
        self.assertEqual(mock_draw.call_args.kwargs["character"], "THE_SILENT")

    def test_treasure_room_opens_to_reward_relics(self):
        treasure = generate_treasure(NativeRandomSet(seed=2))
        rewards = open_treasure(treasure, NativeRandomSet(seed=2))
        self.assertTrue(treasure.opened)
        self.assertTrue(rewards["relics"])

    def test_treasure_room_adds_sapphire_key_linked_to_last_relic_when_final_act_enabled(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True, final_act_available=True, has_sapphire_key=False)
        env.current_room_type = "TreasureRoom"
        env.current_treasure = generate_treasure(env.randoms)
        env.current_treasure.relic_tier = "COMMON"

        env._step_treasure({"kind": "treasure", "name": "OPEN_CHEST"})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertTrue(env.reward_relics)
        self.assertEqual(env.reward_sapphire_key_relic_id, env.reward_relics[-1]["relic_id"])
        sapphire_action = next(action for action in env.legal_actions() if action.get("key") == "sapphire")
        self.assertEqual(sapphire_action["linked_relic_id"], env.reward_relics[-1]["relic_id"])

    def test_combat_relic_reward_adds_sapphire_key_link_when_final_act_enabled(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True, final_act_available=True, has_sapphire_key=False)
        env.phase = "CARD_REWARD"
        env.current_room_type = "MonsterRoom"
        env.reward_relics = [make_relic("Gremlin Horn")]

        env._offer_sapphire_key_for_last_reward_relic()

        self.assertEqual(env.reward_sapphire_key_relic_id, "Gremlin Horn")
        sapphire_action = next(action for action in env.legal_actions() if action.get("key") == "sapphire")
        self.assertEqual(sapphire_action["linked_relic_id"], "Gremlin Horn")
        self.assertEqual(
            [reward["reward_type"] for reward in env.state()["screen_state"]["rewards"]],
            ["RELIC", "SAPPHIRE_KEY"],
        )

    def test_matryoshka_treasure_rewards_keep_real_order_and_key_link(self):
        env = NativeRunEnv(seed=16714268258855711424, enable_neow=False, start_on_map=True, final_act_available=True, has_sapphire_key=False)
        env.relics.append(make_relic("Matryoshka", counter=2))
        env.relic_pools["COMMON"] = ["Anchor", "Strawberry"]
        env.current_room_type = "TreasureRoom"
        env.current_treasure = TreasureState(chest_type="SmallChest", gold_amount=24, relic_tier="COMMON")

        env._step_treasure({"kind": "treasure", "name": "OPEN_CHEST"})

        self.assertEqual([relic["relic_id"] for relic in env.reward_relics], ["Anchor", "Strawberry"])
        self.assertEqual(next(relic for relic in env.relics if relic["relic_id"] == "Matryoshka")["counter"], 1)
        self.assertEqual(env.reward_sapphire_key_relic_id, "Strawberry")
        self.assertEqual(
            [(action["kind"], action.get("relic_id"), action.get("amount"), action.get("key")) for action in env.legal_actions()[:4]],
            [
                ("reward_relic", "Anchor", None, None),
                ("reward_gold", None, 24, None),
                ("reward_relic", "Strawberry", None, None),
                ("reward_key", "Strawberry", None, "sapphire"),
            ],
        )
        self.assertEqual(
            [reward["reward_type"] for reward in env.state()["screen_state"]["rewards"]],
            ["RELIC", "GOLD", "RELIC", "SAPPHIRE_KEY"],
        )

    def test_claiming_linked_treasure_relic_clears_sapphire_key_offer(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True, final_act_available=True, has_sapphire_key=False)
        env.current_room_type = "TreasureRoom"
        env.current_treasure = generate_treasure(env.randoms)
        env.current_treasure.relic_tier = "COMMON"

        env._step_treasure({"kind": "treasure", "name": "OPEN_CHEST"})
        linked_relic_id = env.reward_relics[-1]["relic_id"]

        env.step({"kind": "reward_relic", "relic_id": linked_relic_id})

        self.assertIsNone(env.reward_sapphire_key_relic_id)
        self.assertFalse(any(action.get("key") == "sapphire" for action in env.legal_actions()))

    def test_claiming_sapphire_key_removes_linked_relic_and_sets_state(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True, final_act_available=True, has_sapphire_key=False)
        env.current_room_type = "TreasureRoom"
        env.current_treasure = generate_treasure(env.randoms)
        env.current_treasure.relic_tier = "COMMON"

        env._step_treasure({"kind": "treasure", "name": "OPEN_CHEST"})
        linked_relic_id = env.reward_relics[-1]["relic_id"]
        sapphire_action = next(action for action in env.legal_actions() if action.get("key") == "sapphire")

        env.step(sapphire_action)

        self.assertTrue(env.has_sapphire_key)
        self.assertIsNone(env.reward_sapphire_key_relic_id)
        self.assertFalse(any(str(relic.get("relic_id")) == linked_relic_id for relic in env.reward_relics))

    def test_generate_treasure_uses_chest_specific_reward_roll_after_chest_selection(self):
        randoms = NativeRandomSet(seed=1)
        treasure = generate_treasure(randoms, act=1)
        self.assertEqual(treasure.chest_type, "MediumChest")
        self.assertEqual(chest_def(treasure.chest_type).gold_amount, 50)
        self.assertEqual(treasure.gold_amount, 0)
        self.assertTrue(treasure.gold_reward)
        self.assertEqual(treasure.relic_tier, "COMMON")
        self.assertEqual(randoms.stream("treasure").counter, 2)

        rewards = open_treasure(treasure, randoms)

        self.assertEqual(rewards["gold"], 53)
        self.assertEqual(randoms.stream("treasure").counter, 3)

    def test_campfire_exposes_rest_and_smith_actions(self):
        state = CampfireState(can_recall=False)
        actions = state.actions(deck=[make_card("Strike_R", uuid="strike"), make_card("Wound", uuid="wound")])
        self.assertTrue(any(action["name"] == "REST" for action in actions))
        self.assertTrue(any(action["name"] == "SMITH" for action in actions))
        self.assertEqual(rest_amount(80), 24)

    def test_campfire_rules_match_decompiled_sources(self):
        rules = campfire_rules()
        self.assertAlmostEqual(rules.base_rest_heal_fraction, 0.3)
        self.assertAlmostEqual(rules.night_terrors_heal_fraction, 1.0)
        self.assertEqual(rules.regal_pillow_bonus, 15)
        self.assertEqual(regal_pillow_bonus(), 15)
        self.assertEqual(rest_amount(80, night_terrors=True), 80)
        self.assertEqual(rest_amount(80, endless_full_belly=True), 12)

    def test_campfire_exposes_relic_options_and_respects_hammer_dripper(self):
        state = CampfireState(can_recall=False)
        actions = state.actions(
            deck=[make_card("Strike_R", uuid="strike"), make_card("Defend_R", uuid="defend")],
            relics=[
                make_relic("Peace Pipe"),
                make_relic("Shovel"),
                make_relic("Girya", counter=2),
                make_relic("Coffee Dripper"),
                make_relic("Fusion Hammer"),
            ],
        )
        names = {action["name"] for action in actions}
        self.assertNotIn("REST", names)
        self.assertNotIn("SMITH", names)
        self.assertIn("TOKE", names)
        self.assertIn("DIG", names)
        self.assertIn("LIFT", names)

    def test_dream_catcher_rest_opens_card_reward(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True)
        env.phase = "CAMPFIRE"
        env.current_room_type = "RestRoom"
        env.current_campfire = CampfireState(can_recall=False)
        env.relics.append(make_relic("Dream Catcher"))
        env.step({"kind": "campfire", "name": "REST"})
        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertTrue(env.reward_cards)

    def test_regal_pillow_rest_uses_source_bonus(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True)
        env.phase = "CAMPFIRE"
        env.current_room_type = "RestRoom"
        env.current_campfire = CampfireState(can_recall=False)
        env.player.current_hp = 40
        env.relics.append(make_relic("Regal Pillow"))
        env.step({"kind": "campfire", "name": "REST"})
        self.assertEqual(env.player.current_hp, 40 + rest_amount(env.player.max_hp) + regal_pillow_bonus())

    def test_campfire_smith_rebuilds_upgraded_card_stats_from_source(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True)
        env.phase = "CAMPFIRE"
        env.current_room_type = "RestRoom"
        env.current_campfire = CampfireState(can_recall=False)
        env.deck = [make_card("Bash", uuid="bash")]

        env.step({"kind": "campfire", "name": "SMITH"})

        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertEqual(env.current_card_select["mode"], "upgrade")
        self.assertEqual(env.current_card_select["return_phase"], "MAP")
        self.assertEqual(env.deck[0]["upgrades"], 0)

        env.step({"kind": "card_select", "target_index": 0})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.deck[0]["upgrades"], 1)
        self.assertEqual(env.deck[0]["base_damage"], 10)
        self.assertEqual(env.deck[0]["base_magic"], 3)

    def test_campfire_lift_increments_girya_counter(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True)
        env.phase = "CAMPFIRE"
        env.current_room_type = "RestRoom"
        env.current_campfire = CampfireState(can_recall=False)
        env.relics.append(make_relic("Girya", counter=1))
        env.step({"kind": "campfire", "name": "LIFT"})
        girya = next(relic for relic in env.relics if str(relic.get("relic_id")) == "Girya")
        self.assertEqual(int(girya.get("counter") or 0), 2)

    def test_campfire_dig_opens_relic_reward(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True)
        env.phase = "CAMPFIRE"
        env.current_room_type = "RestRoom"
        env.current_campfire = CampfireState(can_recall=False)
        env.relics.append(make_relic("Shovel"))
        env.step({"kind": "campfire", "name": "DIG"})
        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertTrue(env.reward_relics)

    def test_rest_room_map_entry_exposes_recall_when_final_act_available_and_ruby_missing(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True, final_act_available=True, has_ruby_key=False)
        rest_node = next(node for row in env.map for node in row if node.room_symbol == "R")
        env._step_map(
            {
                "kind": "map",
                "symbol": "R",
                "node_id": f"a1-r{rest_node.y}-x{rest_node.x}",
                "floor": rest_node.y + 1,
            }
        )

        self.assertEqual(env.phase, "CAMPFIRE")
        self.assertIn("RECALL", {action["name"] for action in env.legal_actions()})

    def test_campfire_recall_claims_ruby_key(self):
        env = NativeRunEnv(seed=2, enable_neow=False, start_on_map=True, final_act_available=True, has_ruby_key=False)
        env.phase = "CAMPFIRE"
        env.current_room_type = "RestRoom"
        env.current_campfire = CampfireState(can_recall=True)

        env.step({"kind": "campfire", "name": "RECALL"})

        self.assertTrue(env.has_ruby_key)
        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.current_room_type, "Map")

    def test_small_slimes_encounter_generates_two_real_monsters(self):
        engine = CombatEngine(
            encounter_name="Small Slimes",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(len(engine.state.monsters), 2)
        self.assertTrue(all(monster.intent != "UNKNOWN" for monster in engine.state.monsters))

    def test_two_louse_encounter_generates_two_real_monsters(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(len(engine.state.monsters), 2)
        self.assertTrue(all(monster.intent != "UNKNOWN" for monster in engine.state.monsters))

    def test_strong_encounter_generation_uses_real_group_helpers(self):
        engine = CombatEngine(
            encounter_name="Exordium Wildlife",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            gold=99,
            master_deck=[],
        )
        self.assertEqual(len(engine.state.monsters), 2)
        self.assertTrue(any(monster.monster_id in {"FungiBeast", "JawWorm"} for monster in engine.state.monsters))
        self.assertTrue(any(monster.monster_id in {"FuzzyLouseNormal", "FuzzyLouseDefensive", "SpikeSlime_M", "AcidSlime_M"} for monster in engine.state.monsters))

    def test_looter_tracks_stolen_gold_for_run_boundary(self):
        engine = CombatEngine(
            encounter_name="Looter",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            gold=99,
            master_deck=[],
        )
        start_gold = engine.gold
        engine._take_monster_turns()
        self.assertLess(engine.gold, start_gold)

    def test_card_reward_exposes_stolen_gold_as_separate_reward_item(self):
        env = NativeRunEnv(seed=20, ascension_level=0)
        env.phase = "CARD_REWARD"
        env.reward_stolen_gold = 30
        env.reward_gold = 16

        gold_actions = [action for action in env.legal_actions() if action["kind"] == "reward_gold"]

        self.assertEqual(
            [(action["name"], action["amount"]) for action in gold_actions],
            [("STOLEN_GOLD", 30), ("GOLD", 16)],
        )

        start_gold = env.gold
        env.step(gold_actions[0])
        self.assertEqual(env.gold, start_gold + 30)
        env.step(gold_actions[1])
        self.assertEqual(env.gold, start_gold + 46)

    def test_elite_encounter_generation_exposes_real_intents(self):
        for encounter_name in ("Gremlin Nob", "Lagavulin", "3 Sentries"):
            engine = CombatEngine(
                encounter_name=encounter_name,
                randoms=NativeRandomSet(seed=2),
                ascension_level=0,
                player=PlayerState(current_hp=80, max_hp=80),
                master_deck=[],
            )
            self.assertTrue(engine.state.monsters)
            self.assertTrue(all(monster.intent != "UNKNOWN" for monster in engine.state.monsters))

    def test_lagavulin_wakes_only_after_hp_loss(self):
        engine = CombatEngine(
            encounter_name="Lagavulin",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        lagavulin = engine.state.monsters[0]

        self.assertEqual(engine._player_deal_damage(lagavulin, 6), 0)
        self.assertEqual(lagavulin.block, 2)
        self.assertEqual(lagavulin.next_move, "SLEEP")
        self.assertFalse(bool(lagavulin.meta.get("opened", False)))

        self.assertEqual(engine._player_deal_damage(lagavulin, 6), 4)
        self.assertEqual(lagavulin.current_hp, lagavulin.max_hp - 4)
        self.assertEqual(lagavulin.next_move, "STUN")
        self.assertTrue(bool(lagavulin.meta.get("opened", False)))
        self.assertEqual(_power_amount(lagavulin, "Metallicize"), 0)

    def test_lagavulin_attack_applies_monster_strength(self):
        engine = CombatEngine(
            encounter_name="Lagavulin Event",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=30, max_hp=80),
            master_deck=[],
        )
        lagavulin = engine.state.monsters[0]
        combat_engine_mod._add_power(lagavulin, "Strength", 2)
        lagavulin.next_move = "STRONG_ATTACK"

        engine._lagavulin_take_turn(lagavulin)

        self.assertEqual(engine.player.current_hp, 10)

    def test_gremlin_gang_encounter_generates_real_group_and_turn_cycle(self):
        engine = CombatEngine(
            encounter_name="Gremlin Gang",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(len(engine.state.monsters), 4)
        self.assertTrue(
            all(
                monster.monster_id in {"GremlinWarrior", "GremlinThief", "GremlinFat", "GremlinTsundere", "GremlinWizard"}
                for monster in engine.state.monsters
            )
        )
        engine.step({"kind": "end", "name": "END_TURN"})
        self.assertIn(engine.outcome, {"UNDECIDED", "DEFEAT", "VICTORY"})

    def test_gremlin_wizard_name_matches_comm_mod(self):
        monster = combat_engine_mod._spawn_gremlin_wizard(NativeRandomSet(seed=2), ascension_level=0)
        self.assertEqual(monster.name, "Gremlin Wizard")

    def test_gremlin_wizard_magic_uses_strength_modified_attack_damage(self):
        engine = CombatEngine(
            encounter_name="Gremlin Gang",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        wizard = next(monster for monster in engine.state.monsters if monster.monster_id == "GremlinWizard")
        wizard.next_move = "DOPE_MAGIC"
        combat_engine_mod._add_power(wizard, "Strength", -9)
        combat_engine_mod._append_power(wizard, "Shackled", 9, misc=9)

        engine._gremlin_wizard_take_turn(wizard)

        self.assertEqual(engine.player.current_hp, 64)

    def test_fat_gremlin_roll_action_advances_ai_rng_before_shield_protect(self):
        randoms = NativeRandomSet(seed=5)
        randoms.reset_floor_streams(4)
        engine = CombatEngine(
            encounter_name="Gremlin Gang",
            randoms=randoms,
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        wizard = next(monster for monster in engine.state.monsters if monster.monster_id == "GremlinWizard")
        warrior = next(monster for monster in engine.state.monsters if monster.monster_id == "GremlinWarrior")
        fat = next(monster for monster in engine.state.monsters if monster.monster_id == "GremlinFat")
        shield = next(monster for monster in engine.state.monsters if monster.monster_id == "GremlinTsundere")

        self.assertEqual(
            [monster.monster_id for monster in engine.state.monsters],
            ["GremlinWizard", "GremlinWarrior", "GremlinFat", "GremlinTsundere"],
        )
        self.assertEqual(randoms.stream("ai").counter, 4)

        engine._gremlin_fat_take_turn(fat)
        engine._gremlin_tsundere_take_turn(shield)

        self.assertEqual(wizard.block, int(shield.meta["block_amount"]))
        self.assertEqual(warrior.block, 0)
        self.assertEqual(fat.block, 0)

    def test_gremlin_warrior_attack_uses_strength_power(self):
        randoms = NativeRandomSet(seed=2)
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=randoms,
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        warrior = combat_engine_mod._spawn_gremlin_warrior(randoms, ascension_level=0)
        warrior.meta["opening_ai_roll"] = False
        warrior.powers.append({"power_id": "Strength", "amount": 3})
        engine.state.monsters = [warrior]

        engine._gremlin_warrior_take_turn(warrior)

        self.assertEqual(engine.player.current_hp, 73)

    def test_gremlin_minion_attacks_use_weak_damage_scaling(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )

        fat = combat_engine_mod._spawn_gremlin_fat(engine.randoms, ascension_level=0)
        thief = combat_engine_mod._spawn_gremlin_thief(engine.randoms, ascension_level=0)
        shield = combat_engine_mod._spawn_gremlin_tsundere(engine.randoms, ascension_level=0)
        shield.next_move = "BASH"
        for monster in (fat, thief, shield):
            monster.powers.append({"power_id": "Weakened", "amount": 1})

        engine._gremlin_fat_take_turn(fat)
        engine._gremlin_thief_take_turn(thief)
        engine._gremlin_tsundere_take_turn(shield)

        self.assertEqual(engine.player.current_hp, 67)

    def test_exhume_opens_exhaust_card_select_before_returning_card(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Exhume", uuid="played-exhume")]
        engine.state.exhaust_pile = [
            make_card("Intimidate", uuid="exhaust-intimidate"),
            make_card("Carnage", uuid="exhaust-carnage"),
        ]
        engine.player.energy = 3

        engine.step({"kind": "card", "card_index": 0, "name": "Exhume", "card_id": "Exhume"})

        self.assertEqual(engine.player.energy, 2)
        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.pending_card_select["mode"], "EXHUME")
        self.assertEqual([card["card_id"] for card in engine.pending_card_select["cards"]], ["Intimidate", "Carnage"])
        self.assertEqual(engine.state.hand, [])
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Intimidate", "Carnage"])

        engine.step({"kind": "card_select", "choice_index": 1})

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Carnage"])
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Intimidate", "Exhume"])

    def test_hemokinesis_loses_base_magic_hp(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=76, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Hemokinesis", uuid="hemo")]
        engine.player.energy = 3

        engine.step({"kind": "card", "card_index": 0, "name": "Hemokinesis", "card_id": "Hemokinesis", "target_index": 0})

        self.assertEqual(engine.player.current_hp, 74)

    def test_madness_treats_missing_cost_for_turn_as_current_cost(self):
        randoms = NativeRandomSet(seed=16600467252677589652)
        randoms.reset_floor_streams(2)
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=randoms,
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        first_defend = make_card("Defend_R", uuid="first-defend")
        first_defend["cost_for_turn"] = None
        second_defend = make_card("Defend_R", uuid="second-defend")
        second_defend["cost_for_turn"] = 1
        strike = make_card("Strike_R", uuid="strike")
        strike["cost_for_turn"] = 1
        body_slam = make_card("Body Slam", uuid="body-slam")
        body_slam["cost_for_turn"] = 1
        engine.state.hand = [make_card("Madness", uuid="madness"), first_defend, strike, body_slam, second_defend]
        engine.player.energy = 3

        engine.step({"kind": "card", "card_index": 0, "name": "Madness", "card_id": "Madness"})

        self.assertEqual(engine.state.hand[0]["card_id"], "Defend_R")
        self.assertEqual(engine.state.hand[0]["cost"], 0)
        self.assertEqual(engine.state.hand[0]["cost_for_turn"], 0)
        self.assertEqual(engine.state.hand[-1]["cost"], 1)

    def test_gremlin_gang_member_death_does_not_force_remaining_gremlins_to_escape(self):
        engine = CombatEngine(
            encounter_name="Gremlin Gang",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        first_gremlin = engine.state.monsters[0]

        engine._kill_monster(first_gremlin)

        remaining_moves = [
            monster.next_move
            for monster in engine.state.monsters
            if monster is not first_gremlin and monster.current_hp > 0
        ]
        self.assertTrue(remaining_moves)
        self.assertNotIn("ESCAPE", remaining_moves)

    def test_attack_counter_relics_reset_to_negative_one_on_victory(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            relics=[
                {"relic_id": "Kunai", "name": "Kunai", "counter": 2},
                {"relic_id": "Shuriken", "name": "Shuriken", "counter": 1},
                {"relic_id": "Ornamental Fan", "name": "Ornamental Fan", "counter": 2},
                {"relic_id": "Letter Opener", "name": "Letter Opener", "counter": 1},
            ],
        )

        engine._handle_victory_relics()

        self.assertEqual([relic["counter"] for relic in engine.relics], [-1, -1, -1, -1])

    def test_shield_gremlin_protect_uses_ai_rng_for_random_block_target(self):
        randoms = NativeRandomSet(seed=2)
        engine = CombatEngine(
            encounter_name="Gremlin Gang",
            randoms=randoms,
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        shield = next(monster for monster in engine.state.monsters if monster.monster_id == "GremlinTsundere")
        ai_counter = randoms.stream("ai").counter
        misc_counter = randoms.stream("misc").counter

        engine._gremlin_tsundere_take_turn(shield)

        self.assertEqual(randoms.stream("ai").counter, ai_counter + 1)
        self.assertEqual(randoms.stream("misc").counter, misc_counter)

    def test_monster_turn_clears_block_once_before_group_actions(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        shield = MonsterState(
            monster_id="GremlinTsundere",
            name="Shield Gremlin",
            current_hp=12,
            max_hp=12,
            next_move="PROTECT",
            meta={"block_amount": 7, "bash_damage": 6},
        )
        warrior = MonsterState(
            monster_id="GremlinWarrior",
            name="Mad Gremlin",
            current_hp=21,
            max_hp=21,
            block=5,
            next_move="SCRATCH",
            meta={"scratch_damage": 4},
        )
        engine.state.monsters = [shield, warrior]

        engine._take_monster_turns()

        self.assertEqual(warrior.block, 7)

    def test_byrd_encounter_generates_real_group_and_turn_cycle(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(len(engine.state.monsters), 3)
        self.assertTrue(all(monster.monster_id == "Byrd" for monster in engine.state.monsters))
        engine.step({"kind": "end", "name": "END_TURN"})
        self.assertIn(engine.outcome, {"UNDECIDED", "DEFEAT", "VICTORY"})

    def test_byrd_flight_reduces_damage_and_grounds_on_zero(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        byrd = engine.state.monsters[0]
        self.assertEqual(_power_amount(byrd, "Flight"), 3)
        dealt = engine._player_deal_damage(byrd, 10)
        self.assertEqual(dealt, 5)
        self.assertEqual(_power_amount(byrd, "Flight"), 2)
        engine._player_deal_damage(byrd, 10)
        engine._player_deal_damage(byrd, 10)
        self.assertFalse(bool(byrd.meta.get("is_flying", True)))
        self.assertEqual(byrd.next_move, "STUNNED")
        ai_counter = engine.randoms.stream("ai").counter
        engine._byrd_take_turn(byrd)
        self.assertEqual(engine.randoms.stream("ai").counter, ai_counter + 1)
        self.assertEqual(byrd.next_move, "HEADBUTT")

    def test_headbutt_defers_byrd_flight_reduction_until_discard_selection_resolves(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        byrd = engine.state.monsters[0]
        byrd.current_hp = 20
        engine.state.hand = [make_card("Headbutt", uuid="headbutt")]
        engine.state.discard_pile = [
            make_card("Strike_R", uuid="discard-strike"),
            make_card("Defend_R", uuid="discard-defend"),
        ]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Headbutt")
        engine.step(action)

        self.assertEqual(engine.pending_card_select["mode"], "HEADBUTT")
        self.assertEqual(_power_amount(byrd, "Flight"), 3)

        engine.step({"kind": "card_select", "choice_index": 0})

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual(_power_amount(byrd, "Flight"), 2)

    def test_headbutt_defers_rage_block_until_discard_selection_resolves(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Headbutt", uuid="headbutt")]
        engine.state.discard_pile = [
            make_card("Strike_R", uuid="discard-strike"),
            make_card("Defend_R", uuid="discard-defend"),
        ]
        engine.player.energy = 3
        engine.player.powers.append({"id": "Rage", "power_id": "Rage", "amount": 3})

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Headbutt")
        engine.step(action)

        self.assertEqual(engine.pending_card_select["mode"], "HEADBUTT")
        self.assertEqual(engine.player.block, 0)

        engine.step({"kind": "card_select", "choice_index": 0})

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual(engine.player.block, 3)

    def test_chosen_and_byrd_opening_moves_consume_source_ai_rolls(self):
        engine = CombatEngine(
            encounter_name="Chosen and Byrds",
            randoms=NativeRandomSet(seed=12, floor=22),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        byrd, chosen = engine.state.monsters
        self.assertEqual(byrd.next_move, "PECK")
        self.assertEqual(chosen.next_move, "POKE")
        self.assertEqual(engine.randoms.stream("ai").counter, 3)

        engine.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(byrd.next_move, "SWOOP")
        self.assertEqual(chosen.next_move, "HEX")

    def test_byrd_peck_uses_monster_weak_scaling(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=44, max_hp=80),
            master_deck=[],
        )
        byrd = engine.state.monsters[0]
        byrd.next_move = "PECK"
        combat_engine_mod._add_power(byrd, "Weakened", 1)

        engine._byrd_take_turn(byrd)

        self.assertEqual(engine.player.current_hp, 44)

    def test_direct_monster_powers_keep_insertion_order_after_existing_flight(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=44, max_hp=80),
            master_deck=[],
        )
        byrd = engine.state.monsters[0]

        combat_engine_mod._direct_add_power(byrd, "Strength", 1)

        self.assertEqual([power["power_id"] for power in byrd.powers], ["Flight", "Strength"])

    def test_byrd_flight_sorts_between_vulnerable_and_weak(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=44, max_hp=80),
            master_deck=[],
        )
        byrd = engine.state.monsters[0]

        engine._apply_monster_debuff(byrd, "Vulnerable", 1)
        engine._apply_monster_debuff(byrd, "Weak", 1)

        self.assertEqual([power["power_id"] for power in byrd.powers], ["Vulnerable", "Flight", "Weakened"])

    def test_byrd_caw_places_strength_before_flight(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=44, max_hp=80),
            master_deck=[],
        )
        byrd = engine.state.monsters[0]
        byrd.next_move = "CAW"

        engine._byrd_take_turn(byrd)

        self.assertEqual([power["power_id"] for power in byrd.powers], ["Strength", "Flight"])

    def test_byrd_caw_stacks_existing_strength_without_reordering(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=44, max_hp=80),
            master_deck=[],
        )
        byrd = engine.state.monsters[0]
        combat_engine_mod._direct_add_power(byrd, "Strength", 1)
        byrd.next_move = "CAW"

        engine._byrd_take_turn(byrd)

        self.assertEqual([power["power_id"] for power in byrd.powers], ["Flight", "Strength"])
        self.assertEqual(_power_amount(byrd, "Strength"), 2)

    def test_byrd_peck_stops_after_flame_barrier_kills_source(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=44, max_hp=80, block=12),
            master_deck=[],
        )
        byrd = engine.state.monsters[0]
        engine.player.block = 12
        byrd.current_hp = 12
        byrd.next_move = "PECK"
        combat_engine_mod._add_power(byrd, "Strength", 2)
        combat_engine_mod._append_power(engine.player, "FlameBarrier", 4, misc=4)

        engine._byrd_take_turn(byrd)

        self.assertEqual(engine.player.current_hp, 44)
        self.assertEqual(engine.player.block, 3)
        self.assertEqual(byrd.current_hp, 0)

    def test_chosen_poke_uses_monster_weak_scaling(self):
        engine = CombatEngine(
            encounter_name="Chosen",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=44, max_hp=80),
            master_deck=[],
        )
        chosen = engine.state.monsters[0]
        chosen.next_move = "POKE"
        engine.player.block = 9
        combat_engine_mod._add_power(chosen, "Weakened", 1)

        engine._chosen_take_turn(chosen)

        self.assertEqual(engine.player.current_hp, 44)
        self.assertEqual(engine.player.block, 3)

    def test_snecko_encounter_generates_and_turn_cycles(self):
        engine = CombatEngine(
            encounter_name="Snecko",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(len(engine.state.monsters), 1)
        self.assertEqual(engine.state.monsters[0].monster_id, "Snecko")
        engine.step({"kind": "end", "name": "END_TURN"})
        self.assertIn(engine.outcome, {"UNDECIDED", "DEFEAT", "VICTORY"})

    def test_snecko_opening_glare_consumes_initial_ai_roll(self):
        probe = NativeRandomSet(seed=12, floor=21)
        probe.stream("ai").random(99)
        second_roll = int(probe.stream("ai").random(99))
        expected_next_move = "TAIL" if second_roll < 40 else "BITE"

        engine = CombatEngine(
            encounter_name="Snecko",
            randoms=NativeRandomSet(seed=12, floor=21),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        snecko = engine.state.monsters[0]
        self.assertEqual(snecko.next_move, "GLARE")
        self.assertEqual(engine.randoms.stream("ai").counter, 1)

        engine.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(_power_amount(engine.player, "Confusion"), -1)
        self.assertEqual(engine.randoms.stream("ai").counter, 2)
        self.assertEqual(snecko.next_move, expected_next_move)

    def test_end_turn_discard_preserves_confusion_cost_for_turn(self):
        engine = CombatEngine(
            encounter_name="Snecko",
            randoms=NativeRandomSet(seed=12, floor=21),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        defend = make_card("Defend_R", upgrades=1, uuid="confused-defend")
        defend["cost_for_turn"] = 3
        engine.state.hand = [defend]
        engine.state.draw_pile = [make_card("Strike_R", uuid=f"draw-{index}") for index in range(5)]
        engine.player.energy = 0
        combat_engine_mod._add_power(engine.player, "Confusion", -1)

        engine._end_turn()

        discarded = next(card for card in engine.state.discard_pile if card.get("uuid") == "confused-defend")
        self.assertEqual(discarded["cost_for_turn"], 3)

    def test_snecko_bite_uses_monster_strength_and_weak(self):
        engine = CombatEngine(
            encounter_name="Snecko",
            randoms=NativeRandomSet(seed=12, floor=21),
            ascension_level=0,
            player=PlayerState(current_hp=52, max_hp=80),
            master_deck=[],
        )
        engine.player.block = 5
        snecko = engine.state.monsters[0]
        snecko.next_move = "BITE"
        combat_engine_mod._add_power(snecko, "Strength", -2)
        combat_engine_mod._add_power(snecko, "Weakened", 1)

        engine._snecko_take_turn(snecko)

        self.assertEqual(engine.player.current_hp, 48)
        self.assertEqual(engine.player.block, 0)

    def test_three_cultists_encounter_generates_real_group(self):
        engine = CombatEngine(
            encounter_name="3 Cultists",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(len(engine.state.monsters), 3)
        self.assertTrue(all(monster.monster_id == "Cultist" for monster in engine.state.monsters))

    def test_slavers_encounter_generates_real_group(self):
        engine = CombatEngine(
            encounter_name="Slavers",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(len(engine.state.monsters), 3)
        self.assertEqual({monster.monster_id for monster in engine.state.monsters}, {"SlaverBlue", "SlaverRed", "SlaverBoss"})

    def test_fiend_fire_exhausts_hand(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Fiend Fire", uuid="fiend-fire"),
            make_card("Strike_R", uuid="ff-strike-1"),
            make_card("Strike_R", uuid="ff-strike-2"),
        ]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Fiend Fire")
        engine.step(action)
        self.assertEqual(len(engine.state.hand), 0)
        self.assertGreaterEqual(len(engine.state.exhaust_pile), 3)

    def test_spot_weakness_grants_strength_against_attacking_target(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Spot Weakness", uuid="spot-weakness")]
        engine.player.energy = 3
        target = engine.state.monsters[0]
        target.next_move = "DARK_STRIKE"
        engine._update_monster_intents()
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Spot Weakness")
        engine.step(action)
        self.assertEqual(next(power["amount"] for power in engine.player.powers if power["power_id"] == "Strength"), 3)

    def test_battle_trance_applies_no_draw_until_end_of_turn(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Battle Trance", uuid="battle-trance")]
        engine.state.draw_pile = [make_card("Strike_R", uuid=f"draw-{i}") for i in range(8)]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Battle Trance")
        engine.step(action)
        self.assertEqual(_power_amount(engine.player, "No Draw"), -1)
        size_after_trance = len(engine.state.hand)
        engine.draw_cards(2)
        self.assertEqual(len(engine.state.hand), size_after_trance)
        engine.step({"kind": "end", "name": "END_TURN"})
        self.assertEqual(_power_amount(engine.player, "No Draw"), 0)

    def test_good_instincts_gains_block_without_drawing(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Good Instincts", uuid="good-instincts")]
        engine.state.draw_pile = [make_card("Strike_R", uuid=f"gi-{i}") for i in range(3)]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Good Instincts")
        engine.step(action)
        self.assertEqual(engine.player.block, 6)
        self.assertEqual([card["card_id"] for card in engine.state.draw_pile], ["Strike_R", "Strike_R", "Strike_R"])

    def test_bandage_up_heals_player(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=20, max_hp=30),
            master_deck=[],
        )
        engine.state.hand = [make_card("Bandage Up", uuid="bandage-up")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Bandage Up")
        engine.step(action)
        self.assertEqual(engine.player.current_hp, 24)

    def test_deep_breath_shuffles_discard_into_draw_and_draws(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Deep Breath", uuid="deep-breath")]
        engine.state.draw_pile = [make_card("Strike_R", uuid="draw-card")]
        engine.state.discard_pile = [make_card("Defend_R", uuid="discard-card")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Deep Breath")
        engine.step(action)
        self.assertEqual(sum(1 for card in engine.state.discard_pile if card["card_id"] == "Defend_R"), 0)
        self.assertTrue(any(card["card_id"] == "Defend_R" for card in engine.state.hand + engine.state.draw_pile))

    def test_hand_of_greed_grants_bonus_reward_gold_on_kill(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        target.current_hp = 1
        engine.state.hand = [make_card("HandOfGreed", uuid="hand-of-greed")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "HandOfGreed")
        engine.step(action)
        self.assertEqual(engine.bonus_reward_gold, 20)

    def test_offering_loses_hp_gains_energy_and_draws(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=40, max_hp=40),
            master_deck=[],
        )
        engine.state.hand = [make_card("Offering", uuid="offering")]
        engine.state.draw_pile = [make_card("Strike_R", uuid=f"offering-draw-{i}") for i in range(5)]
        engine.player.energy = 0
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Offering")
        engine.step(action)
        self.assertEqual(engine.player.current_hp, 34)
        self.assertEqual(engine.player.energy, 2)
        self.assertGreaterEqual(sum(1 for card in engine.state.hand if card["card_id"] == "Strike_R"), 3)

    def test_hex_adds_dazed_after_offering_draws(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=40, max_hp=40),
            master_deck=[],
        )
        engine.state.hand = [make_card("Offering", uuid="offering")]
        engine.state.draw_pile = [
            make_card("Barricade", uuid="draw-barricade"),
            make_card("Strike_R", uuid="draw-strike-0"),
            make_card("Strike_R", uuid="draw-strike-1"),
            make_card("Strike_R", uuid="draw-strike-2"),
        ]
        engine.player.energy = 0
        combat_engine_mod._add_power(engine.player, "Hex", 1)

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Offering")
        engine.step(action)

        self.assertEqual(sum(1 for card in engine.state.hand if card["card_id"] == "Dazed"), 0)
        self.assertEqual(sum(1 for card in engine.state.draw_pile if card["card_id"] == "Dazed"), 1)
        self.assertEqual(sum(1 for card in engine.state.hand if card["card_id"] == "Strike_R"), 3)

    def test_dark_embrace_draws_when_cards_are_exhausted(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Dark Embrace", uuid="dark-embrace"),
            make_card("Second Wind", uuid="second-wind"),
            make_card("Defend_R", uuid="exhaust-target"),
            make_card("Strike_R", uuid="keep-attack"),
        ]
        engine.state.draw_pile = [make_card("Strike_R", uuid="dark-embrace-draw")]
        engine.player.energy = 5
        dark_embrace_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Dark Embrace")
        engine.step(dark_embrace_action)
        second_wind_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Second Wind")
        engine.step(second_wind_action)
        self.assertEqual(_power_amount(engine.player, "Dark Embrace"), 1)
        self.assertTrue(any(card["uuid"] == "dark-embrace-draw" for card in engine.state.hand))

    def test_panache_triggers_after_five_cards(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Panache", uuid="panache")] + [make_card("Defend_R", uuid=f"panache-defend-{i}") for i in range(5)]
        engine.player.energy = 10
        before = engine.state.monsters[0].current_hp
        while any(card["card_id"] == "Panache" for card in engine.state.hand):
            action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Panache")
            engine.step(action)
        for _ in range(5):
            action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Defend_R")
            engine.step(action)
        self.assertLess(engine.state.monsters[0].current_hp, before)

    def test_havoc_plays_top_card_of_draw_pile(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Havoc", upgrades=1, uuid="havoc")]
        engine.state.draw_pile = [make_card("Strike_R", uuid="havoc-top")]
        engine.player.energy = 3
        before = engine.state.monsters[0].current_hp
        card_random_counter = engine.randoms.stream("card_random").counter
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Havoc")
        engine.step(action)
        self.assertLess(engine.state.monsters[0].current_hp, before)
        self.assertFalse(engine.state.draw_pile)
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Havoc"])
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Strike_R"])
        self.assertEqual(engine.state.exhaust_pile[0].get("cost_for_turn") or engine.state.exhaust_pile[0].get("cost"), 1)
        self.assertEqual(engine.randoms.stream("card_random").counter, card_random_counter + 1)

    def test_havoc_plays_free_top_card_after_spending_last_energy(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Havoc", uuid="havoc")]
        engine.state.draw_pile = [make_card("Strike_R", uuid="havoc-top")]
        engine.player.energy = 1
        before = engine.state.monsters[0].current_hp

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Havoc"))

        self.assertEqual(engine.player.energy, 0)
        self.assertLess(engine.state.monsters[0].current_hp, before)
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Strike_R"])

    def test_havoc_triggers_slow_before_playing_top_card(self):
        engine = CombatEngine(
            encounter_name="Giant Head",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        combat_engine_mod._set_power_amount(monster, "Slow", 4)
        engine.state.hand = [make_card("Havoc", upgrades=1, uuid="havoc")]
        engine.state.draw_pile = [make_card("Strike_R", uuid="havoc-strike")]
        engine.player.energy = 3
        before = monster.current_hp

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Havoc"))

        self.assertEqual(before - monster.current_hp, 9)
        self.assertEqual(_power_amount(monster, "Slow"), 6)

    def test_havoc_after_use_marker_does_not_persist_when_replayed(self):
        engine = CombatEngine(
            encounter_name="Giant Head",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            act=3,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        engine.state.hand = [make_card("Havoc", upgrades=1, uuid="havoc")]
        engine.state.draw_pile = [make_card("Strike_R", uuid="havoc-strike")]
        engine.player.energy = 3
        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Havoc"))
        replayed_havoc = engine.state.discard_pile.pop(0)
        self.assertNotIn("_monster_after_use_already_triggered", replayed_havoc)

        combat_engine_mod._set_power_amount(monster, "Slow", 0)
        combat_engine_mod._set_power_amount(engine.player, "Strength", 7)
        engine.state.hand = [replayed_havoc]
        engine.state.draw_pile = [make_card("Limit Break", upgrades=1, uuid="havoc-limit-break")]
        engine.player.energy = 3
        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Havoc"))

        self.assertEqual(_power_amount(monster, "Slow"), 2)
        self.assertEqual(_power_amount(engine.player, "Strength"), 14)

    def test_havoc_hex_shuffle_happens_before_generated_card_draw(self):
        randoms = NativeRandomSet(seed=47, floor=21)
        for args in ((0, 3), (0,), (6,), (6,), (7,), (7,)):
            randoms.stream("card_random").random(*args)
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=randoms,
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        combat_engine_mod._add_power(engine.player, "Hex", 1)
        engine.state.hand = [make_card("Havoc", uuid="havoc")]
        engine.state.draw_pile = [
            make_card("Armaments", uuid="draw-armaments"),
            make_card("Bash", uuid="draw-bash"),
            make_card("Defend_R", uuid="draw-defend"),
            make_card("Strike_R", uuid="draw-strike-0"),
            make_card("Strike_R", uuid="draw-strike-1"),
            make_card("Strike_R", uuid="draw-strike-2"),
            make_card("Shrug It Off", uuid="draw-shrug"),
        ]
        engine.player.energy = 3

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Havoc"))

        self.assertEqual(
            [card["card_id"] for card in engine.state.draw_pile],
            ["Armaments", "Dazed", "Bash", "Defend_R", "Strike_R", "Dazed", "Strike_R"],
        )
        self.assertEqual([call.args for call in engine.randoms.stream("card_random").calls[-3:]], [(0, 0), (5,), (5,)])

    def test_havoc_uses_current_energy_for_free_x_cost_card(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=70),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.current_hp = 20
        engine.state.hand = [make_card("Havoc", upgrades=1, uuid="havoc")]
        engine.state.draw_pile = [make_card("Whirlwind", uuid="havoc-whirlwind")]
        engine.player.energy = 3
        before = monster.current_hp

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Havoc")
        engine.step(action)

        self.assertEqual(monster.current_hp, before - 15)
        self.assertEqual(engine.player.energy, 3)
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Havoc"])
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Whirlwind"])

    def test_havoc_under_corruption_exhausts_before_played_top_card(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.powers.append({"power_id": "Corruption", "amount": -1})
        engine.state.hand = [make_card("Havoc", uuid="havoc")]
        engine.state.draw_pile = [make_card("Defend_R", uuid="havoc-defend")]
        engine.player.energy = 3
        engine._refresh_all_card_flags()

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Havoc")
        engine.step(action)

        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Havoc", "Defend_R"])

    def test_havoc_exhausts_warcry_after_pending_card_select_resolves(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Havoc", uuid="havoc"), make_card("Strike_R", uuid="strike")]
        engine.state.draw_pile = [make_card("Defend_R", uuid="draw-defend"), make_card("Warcry", uuid="havoc-warcry")]
        engine.player.energy = 3

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Havoc"))

        self.assertEqual(engine.pending_card_select["mode"], "WARCRY")
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], [])
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Havoc"])

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Strike_R"))

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Warcry"])
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Havoc"])
        self.assertEqual(engine.state.draw_pile[-1]["card_id"], "Strike_R")

    def test_havoc_is_in_discard_before_warcry_draw_shuffle(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Havoc", upgrades=1, uuid="havoc")]
        engine.state.draw_pile = [make_card("Warcry", upgrades=1, uuid="havoc-warcry")]
        engine.state.discard_pile = [make_card("Strike_R", uuid="discard-strike")]
        engine.player.energy = 3

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Havoc"))

        self.assertEqual(engine.pending_card_select["mode"], "WARCRY")
        self.assertCountEqual([card["card_id"] for card in engine.state.hand], ["Havoc", "Strike_R"])
        self.assertEqual(engine.state.discard_pile, [])

    def test_havoc_exhausts_anger_original_after_anger_adds_discard_copy(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Havoc", uuid="havoc")]
        engine.state.draw_pile = [make_card("Anger", uuid="havoc-anger")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Havoc")
        engine.step(action)

        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Havoc", "Anger"])
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Anger"])

    def test_havoc_preselects_target_rng_before_playing_no_target_top_card(self):
        randoms = NativeRandomSet(seed=1126486811419295697)
        randoms.reset_floor_streams(16)
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=randoms,
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Havoc", uuid="havoc"),
            make_card("True Grit", uuid="true-grit"),
            make_card("Shrug It Off", uuid="shrug"),
            make_card("Armaments", uuid="armaments"),
            make_card("Defend_R", uuid="defend"),
        ]
        engine.state.draw_pile = [make_card("Defend_R", uuid="havoc-top-defend")]
        engine.player.energy = 3

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Havoc"))
        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "True Grit"))

        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Shrug It Off", "Defend_R"])
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Defend_R", "Armaments"])

    def test_thinking_ahead_opens_put_on_deck_card_select_after_draw(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Thinking Ahead", uuid="thinking-ahead"),
            make_card("Strike_R", uuid="strike"),
        ]
        engine.state.draw_pile = [
            make_card("Defend_R", uuid="draw-defend"),
            make_card("Headbutt", uuid="draw-headbutt"),
        ]
        engine.player.energy = 3

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Thinking Ahead"))

        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.pending_card_select["mode"], "PUT_ON_DECK")
        self.assertEqual([choice["card_id"] for choice in engine.legal_actions()], ["Strike_R", "Headbutt", "Defend_R"])

        engine.step(engine.legal_actions()[1])

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Strike_R", "Defend_R"])
        self.assertEqual([card["card_id"] for card in engine.state.draw_pile], ["Headbutt"])
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Thinking Ahead"])

    def test_entrench_doubles_current_block(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.block = 7
        engine.state.hand = [make_card("Entrench", uuid="entrench")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Entrench")
        engine.step(action)
        self.assertEqual(engine.player.block, 14)

    def test_body_slam_uses_current_block_as_damage(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.block = 11
        engine.state.hand = [make_card("Body Slam", uuid="body-slam")]
        engine.player.energy = 3
        before = engine.state.monsters[0].current_hp
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Body Slam")
        engine.step(action)
        self.assertEqual(engine.state.monsters[0].current_hp, before - 11)

    def test_dark_shackles_reduces_strength_then_restores_at_end_of_turn(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        target.powers.append(
            {
                "power_id": "Strength",
                "id": "Strength",
                "name": "Strength",
                "amount": 5,
                "card": None,
                "damage": 0,
                "just_applied": False,
                "misc": 5,
            }
        )
        engine.state.hand = [make_card("Dark Shackles", uuid="dark-shackles")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Dark Shackles")
        engine.step(action)
        self.assertEqual(_power_amount(target, "Strength"), -4)
        engine.step({"kind": "end", "name": "END_TURN"})
        self.assertGreaterEqual(_power_amount(target, "Strength"), 5)

    def test_sadistic_nature_triggers_on_applied_debuffs(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        engine.state.hand = [make_card("Sadistic Nature", uuid="sadistic"), make_card("Trip", uuid="trip")]
        engine.player.energy = 3

        sadistic_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Sadistic Nature")
        engine.step(sadistic_action)
        hp_before = int(target.current_hp)

        trip_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Trip")
        engine.step(trip_action)

        self.assertGreater(_power_amount(target, "Vulnerable"), 0)
        self.assertLess(int(target.current_hp), hp_before)

    def test_reckless_charge_adds_dazed_to_draw_pile(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.draw_pile = [make_card("Strike_R", uuid="draw-0"), make_card("Defend_R", uuid="draw-1"), make_card("Bash", uuid="draw-2")]
        expected_ids = [card["card_id"] for card in engine.state.draw_pile]
        expected_index = int(engine.randoms.duplicate_stream("card_random").random(len(engine.state.draw_pile) - 1))
        expected_ids.insert(expected_index, "Dazed")
        shuffle_counter_before = engine.randoms.stream("shuffle").counter
        engine.state.hand = [make_card("Reckless Charge", uuid="reckless-charge")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Reckless Charge")
        engine.step(action)
        self.assertEqual([card["card_id"] for card in engine.state.draw_pile], expected_ids)
        self.assertEqual(engine.randoms.stream("shuffle").counter, shuffle_counter_before)

    def test_wild_strike_adds_wound_to_random_draw_pile_spot(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=14381028785943436429),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.randoms.reset_floor_streams(6)
        engine.state.draw_pile = [
            make_card("Disarm", uuid="draw-0"),
            make_card("Headbutt", uuid="draw-1"),
            make_card("Defend_R", uuid="draw-2"),
            make_card("Heavy Blade", uuid="draw-3"),
            make_card("Defend_R", uuid="draw-4"),
            make_card("Defend_R", uuid="draw-5"),
            make_card("Defend_R", uuid="draw-6"),
            make_card("Strike_R", uuid="draw-7"),
            make_card("Rage", uuid="draw-8"),
            make_card("Strike_R", uuid="draw-9"),
            make_card("Bash", uuid="draw-10"),
            make_card("Strike_R", uuid="draw-11"),
            make_card("Strike_R", uuid="draw-12"),
        ]
        expected_ids = [card["card_id"] for card in engine.state.draw_pile]
        expected_index = int(engine.randoms.duplicate_stream("card_random").random(len(engine.state.draw_pile) - 1))
        expected_ids.insert(expected_index, "Wound")
        shuffle_counter_before = engine.randoms.stream("shuffle").counter
        engine.state.hand = [make_card("Wild Strike", uuid="wild-strike")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Wild Strike")
        engine.step(action)

        self.assertEqual([card["card_id"] for card in engine.state.draw_pile], expected_ids)
        self.assertEqual(engine.randoms.stream("shuffle").counter, shuffle_counter_before)

    def test_wild_strike_advances_card_random_when_draw_pile_is_empty(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.draw_pile = []
        engine.state.hand = [make_card("Wild Strike", uuid="wild-strike")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Wild Strike")
        engine.step(action)

        self.assertEqual([card["card_id"] for card in engine.state.draw_pile], ["Wound"])
        self.assertEqual(engine.randoms.stream("card_random").counter, 1)

    def test_reaper_hits_all_and_heals_for_damage_dealt(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=50, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Reaper", uuid="reaper")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Reaper")
        engine.step(action)
        self.assertGreater(engine.player.current_hp, 50)

    def test_exhume_returns_card_from_exhaust_to_hand(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Exhume", uuid="exhume")]
        engine.state.exhaust_pile = [make_card("Strike_R", uuid="exhausted-strike")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Exhume")
        engine.step(action)
        self.assertTrue(any(card["card_id"] == "Strike_R" for card in engine.state.hand))

    def test_limit_break_doubles_strength(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.powers.append({"power_id": "Strength", "id": "Strength", "amount": 3})
        engine.state.hand = [make_card("Limit Break", uuid="limit-break")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Limit Break")
        engine.step(action)
        self.assertEqual(_power_amount(engine.player, "Strength"), 6)

    def test_evolve_draws_when_status_is_drawn(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Evolve", uuid="evolve")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Evolve")
        engine.step(action)
        engine.state.hand = []
        engine.state.draw_pile = [make_card("Strike_R", uuid="evolve-draw"), make_card("Wound", uuid="evolve-status")]
        engine.draw_cards(1)
        self.assertTrue(any(card["card_id"] == "Wound" for card in engine.state.hand))
        self.assertTrue(any(card["card_id"] == "Strike_R" for card in engine.state.hand))

    def test_evolve_extra_draw_respects_max_hand_size(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.powers.append({"power_id": "Evolve", "id": "Evolve", "amount": 1})
        engine.state.hand = [make_card("Defend_R", uuid=f"hand-{i}") for i in range(9)]
        engine.state.draw_pile = [
            make_card("Strike_R", uuid="blocked-evolve-draw"),
            make_card("Dazed", uuid="status-draw"),
        ]

        engine.draw_cards(1)

        self.assertEqual(len(engine.state.hand), 10)
        self.assertEqual(engine.state.hand[-1]["card_id"], "Dazed")
        self.assertEqual([card["card_id"] for card in engine.state.draw_pile], ["Strike_R"])

    def test_combat_serialize_preserves_zero_gold(self):
        combat = NativeCombatEnv(
            seed=874,
            ascension_level=0,
            encounter_name="Cultist",
            gold=0,
            master_deck=[make_card("Strike_R", uuid=f"strike-{i}") for i in range(20)],
        )

        self.assertEqual(combat.serialize()["gold"], 0)

    def test_headbutt_opens_multi_discard_card_select_before_topdecking(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Headbutt", uuid="headbutt")]
        engine.state.discard_pile = [
            make_card("Bash", uuid="discard-bash"),
            make_card("Strike_R", uuid="discard-strike"),
        ]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Headbutt")
        engine.step(action)

        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual([action["card_id"] for action in engine.legal_actions()], ["Bash", "Strike_R"])
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Bash", "Strike_R"])
        engine.step(engine.legal_actions()[0])

        self.assertEqual(engine.state.draw_pile[-1]["card_id"], "Bash")
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Strike_R", "Headbutt"])

    def test_headbutt_defers_sharp_hide_until_after_card_select(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        cultist = engine.state.monsters[0]
        combat_engine_mod._append_power(cultist, "Sharp Hide", 3)
        engine.state.hand = [make_card("Headbutt", uuid="headbutt")]
        engine.state.discard_pile = [
            make_card("Bash", uuid="discard-bash"),
            make_card("Strike_R", uuid="discard-strike"),
        ]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Headbutt")
        engine.step(action)

        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.player.current_hp, 80)
        engine.step(engine.legal_actions()[0])

        self.assertEqual(engine.player.current_hp, 77)

    def test_headbutt_auto_topdecks_single_discard_without_card_select(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Headbutt", uuid="headbutt")]
        engine.state.discard_pile = [make_card("Bash", uuid="discard-bash")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Headbutt")
        engine.step(action)

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual(engine.state.draw_pile[-1]["card_id"], "Bash")
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Headbutt"])

    def test_headbutt_does_not_open_card_select_after_lethal_final_hit(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        target.current_hp = 1
        engine.state.hand = [make_card("Headbutt", uuid="headbutt")]
        engine.state.discard_pile = [
            make_card("Bash", uuid="discard-bash"),
            make_card("Strike_R", uuid="discard-strike"),
        ]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Headbutt")
        engine.step(action)

        self.assertEqual(engine.outcome, "VICTORY")
        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["card_id"] for card in engine.state.draw_pile], [])
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Bash", "Strike_R", "Headbutt"])

    def test_headbutt_topdecks_before_gremlin_horn_draws_on_kill(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            relics=[make_relic("Gremlin Horn")],
            master_deck=[],
        )
        engine.state.hand = [make_card("Headbutt", uuid="headbutt")]
        engine.state.draw_pile = [make_card("Defend_R", uuid="draw-defend")]
        engine.state.discard_pile = [make_card("Bash", uuid="discard-bash")]
        target = engine.state.monsters[0]
        target.current_hp = 1
        target.block = 0
        target.powers = []
        engine.player.energy = 1

        action = next(
            candidate
            for candidate in engine.legal_actions()
            if candidate.get("card_id") == "Headbutt" and candidate.get("target_index") == 0
        )
        engine.step(action)

        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Bash"])
        self.assertEqual([card["card_id"] for card in engine.state.draw_pile], ["Defend_R"])
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Headbutt"])
        self.assertEqual(engine.player.energy, 1)

    def test_dropkick_refunds_when_target_is_vulnerable(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        target.powers.append({"power_id": "Vulnerable", "id": "Vulnerable", "amount": 2})
        engine.state.hand = [make_card("Dropkick", uuid="dropkick")]
        engine.state.draw_pile = [make_card("Strike_R", uuid="dropkick-draw")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Dropkick")
        engine.step(action)
        self.assertEqual(engine.player.energy, 3)
        self.assertTrue(any(card["card_id"] == "Strike_R" for card in engine.state.hand))

    def test_dropkick_refunds_when_lethal_target_was_vulnerable(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        target.current_hp = 1
        target.block = 0
        target.powers.append({"power_id": "Vulnerable", "id": "Vulnerable", "amount": 2})
        engine.state.hand = [make_card("Dropkick", uuid="dropkick")]
        engine.state.draw_pile = [make_card("Strike_R", uuid="dropkick-draw")]
        engine.player.energy = 1
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Dropkick")

        engine.step(action)

        self.assertEqual(engine.player.energy, 1)
        self.assertEqual(target.current_hp, 0)
        self.assertFalse(combat_engine_mod._alive(target))
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Strike_R"])

    def test_pummel_hits_multiple_times(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Pummel", uuid="pummel")]
        engine.player.energy = 3
        target = engine.state.monsters[0]
        before = target.current_hp
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Pummel")
        engine.step(action)
        self.assertLessEqual(target.current_hp, before - 8)

    def test_iron_wave_grants_block_and_damage(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Iron Wave", uuid="iron-wave")]
        engine.player.energy = 3
        before_hp = engine.state.monsters[0].current_hp
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Iron Wave")
        engine.step(action)
        self.assertGreater(engine.player.block, 0)
        self.assertLess(engine.state.monsters[0].current_hp, before_hp)

    def test_upgraded_cloak_and_dagger_gains_block_and_adds_two_shivs(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Cloak And Dagger", upgrades=1, uuid="cloak-plus")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Cloak And Dagger")

        self.assertFalse(action.get("requires_target"))

        engine.step(action)

        self.assertEqual(engine.player.energy, 2)
        self.assertEqual(engine.player.block, 6)
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Shiv", "Shiv"])
        self.assertTrue(all(card["cost"] == 0 and card["exhausts"] for card in engine.state.hand))
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Cloak And Dagger"])

    def test_accuracy_applies_power_and_updates_existing_shivs_in_all_piles(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        hand_shiv = combat_engine_mod._make_shiv("hand-shiv")
        draw_shiv = combat_engine_mod._make_shiv("draw-shiv", upgrades=1)
        discard_shiv = combat_engine_mod._make_shiv("discard-shiv")
        exhaust_shiv = combat_engine_mod._make_shiv("exhaust-shiv", upgrades=1)
        engine.state.hand = [make_card("Accuracy", uuid="accuracy"), hand_shiv]
        engine.state.draw_pile = [draw_shiv]
        engine.state.discard_pile = [discard_shiv]
        engine.state.exhaust_pile = [exhaust_shiv]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Accuracy")
        engine.step(action)

        self.assertEqual(combat_engine_mod._get_power_amount(engine.player, "Accuracy"), 4)
        self.assertEqual(hand_shiv["base_damage"], 8)
        self.assertEqual(draw_shiv["base_damage"], 10)
        self.assertEqual(discard_shiv["base_damage"], 8)
        self.assertEqual(exhaust_shiv["base_damage"], 10)

    def test_stacking_accuracy_refreshes_existing_shiv_damage(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        existing_shiv = combat_engine_mod._make_shiv("existing-shiv", accuracy_amount=4)
        engine.player.powers = [
            {
                "power_id": "Accuracy",
                "id": "Accuracy",
                "name": "Accuracy",
                "amount": 4,
                "card": None,
                "damage": 0,
                "misc": 4,
                "just_applied": False,
            }
        ]
        engine.state.hand = [make_card("Accuracy", upgrades=1, uuid="accuracy-plus"), existing_shiv]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Accuracy")
        engine.step(action)

        self.assertEqual(combat_engine_mod._get_power_amount(engine.player, "Accuracy"), 10)
        self.assertEqual(existing_shiv["base_damage"], 14)

    def test_upgraded_cloak_and_dagger_generates_accuracy_scaled_shivs(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.powers = [
            {
                "power_id": "Accuracy",
                "id": "Accuracy",
                "name": "Accuracy",
                "amount": 6,
                "card": None,
                "damage": 0,
                "misc": 6,
                "just_applied": False,
            }
        ]
        engine.state.hand = [make_card("Cloak And Dagger", upgrades=1, uuid="cloak-plus")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Cloak And Dagger")
        engine.step(action)

        shivs = [card for card in engine.state.hand if card["card_id"] == "Shiv"]
        self.assertEqual(len(shivs), 2)
        self.assertEqual([card["base_damage"] for card in shivs], [10, 10])

    def test_dodge_and_roll_grants_block_and_next_turn_block(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Dodge and Roll", uuid="dodge")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Dodge and Roll")
        engine.step(action)

        self.assertEqual(engine.player.block, 4)
        self.assertEqual(_power_amount(engine.player, "NextTurnBlock"), 4)

    def test_upgraded_dodge_and_roll_grants_six_block_and_next_turn_block(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Dodge and Roll", upgrades=1, uuid="dodge-plus")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Dodge and Roll")
        engine.step(action)

        self.assertEqual(engine.player.block, 6)
        self.assertEqual(_power_amount(engine.player, "NextTurnBlock"), 6)

    def test_dodge_and_roll_next_turn_block_resolves_and_removes_power(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Dodge and Roll", uuid="dodge")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Dodge and Roll")
        engine.step(action)
        engine._start_player_turn()

        self.assertEqual(_power_amount(engine.player, "NextTurnBlock"), 0)
        self.assertEqual(engine.player.block, 4)

    def test_leap_grants_card_scaled_block_with_dexterity(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.powers = [
            {
                "power_id": "Dexterity",
                "id": "Dexterity",
                "name": "Dexterity",
                "amount": 2,
                "card": None,
                "damage": 0,
                "misc": 2,
                "just_applied": False,
            }
        ]
        engine.state.hand = [make_card("Leap", uuid="leap")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Leap")
        engine.step(action)

        self.assertEqual(engine.player.block, 11)

    def test_upgraded_leap_uses_twelve_base_block_before_frail(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.powers = [
            {
                "power_id": "Frail",
                "id": "Frail",
                "name": "Frail",
                "amount": 1,
                "card": None,
                "damage": 0,
                "misc": 1,
                "just_applied": False,
            }
        ]
        engine.state.hand = [make_card("Leap", upgrades=1, uuid="leap-plus")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Leap")
        engine.step(action)

        self.assertEqual(engine.player.block, 9)

    def test_prepared_draws_then_auto_discards_single_post_draw_card(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Prepared", uuid="prepared")]
        engine.state.draw_pile = [make_card("Strike_G", uuid="drawn-strike")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Prepared")
        engine.step(action)

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["card_id"] for card in engine.state.hand], [])
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Strike_G", "Prepared"])

    def test_prepared_opens_discard_select_after_draw_when_multiple_choices_exist(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Prepared", uuid="prepared"), make_card("Strike_G", uuid="strike")]
        engine.state.draw_pile = [make_card("Defend_G", uuid="drawn-defend")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Prepared")
        engine.step(action)

        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.pending_card_select["mode"], "PREPARED")
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Strike_G", "Defend_G"])

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Defend_G"))

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Strike_G"])
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Defend_G", "Prepared"])

    def test_upgraded_prepared_discards_two_cards_through_sequential_select(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Prepared", upgrades=1, uuid="prepared-plus"), make_card("Strike_G", uuid="strike")]
        engine.state.draw_pile = [
            make_card("Defend_G", uuid="drawn-defend"),
            make_card("Survivor", uuid="drawn-survivor"),
        ]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Prepared")
        engine.step(action)
        self.assertEqual((engine.pending_card_select or {}).get("remaining"), 2)

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Strike_G"))
        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.pending_card_select["remaining"], 1)

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Survivor"))

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Defend_G"])
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Strike_G", "Survivor", "Prepared"])

    def test_seeing_red_refunds_energy(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Seeing Red", uuid="seeing-red")]
        engine.player.energy = 1
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Seeing Red")
        engine.step(action)
        self.assertGreaterEqual(engine.player.energy, 2)

    def test_anger_adds_copy_to_discard(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Anger", uuid="anger")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Anger")
        engine.step(action)
        self.assertTrue(any(card["card_id"] == "Anger" for card in engine.state.discard_pile))

    def test_anger_copy_preserves_modified_cost(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        anger = make_card("Anger", uuid="anger")
        anger["cost"] = 2
        anger["cost_for_turn"] = 2
        anger["cost_for_combat"] = 2
        engine.state.hand = [anger]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Anger")
        engine.step(action)

        anger_discards = [card for card in engine.state.discard_pile if card["card_id"] == "Anger"]
        self.assertEqual([int(card.get("cost") or 0) for card in anger_discards[:2]], [2, 2])

    def test_thunderclap_hits_all_and_applies_vulnerable(self):
        engine = CombatEngine(
            encounter_name="2 Louse",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Thunderclap", uuid="thunderclap")]
        engine.player.energy = 3
        before = [monster.current_hp for monster in engine.state.monsters]
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Thunderclap")
        engine.step(action)
        after = [monster.current_hp for monster in engine.state.monsters]
        self.assertTrue(all(a < b for a, b in zip(after, before)))
        self.assertTrue(all(_power_amount(monster, "Vulnerable") > 0 for monster in engine.state.monsters))

    def test_metallicize_grants_end_of_turn_block(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Metallicize", uuid="metallicize")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Metallicize")
        engine.step(action)
        engine._apply_player_end_of_turn_powers()
        self.assertGreaterEqual(engine.player.block, 3)

    def test_armaments_opens_card_select_before_upgrading_one_card(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Armaments", uuid="armaments"),
            make_card("Strike_R", uuid="strike"),
            make_card("Defend_R", uuid="defend"),
        ]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Armaments")
        engine.step(action)
        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.pending_card_select["mode"], "ARMAMENTS")
        self.assertEqual([candidate["card_id"] for candidate in engine.legal_actions()], ["Strike_R", "Defend_R"])
        self.assertNotIn("Armaments", [card["card_id"] for card in engine.state.discard_pile])

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Strike_R"))
        self.assertIsNone(engine.pending_card_select)
        self.assertIn("Armaments", [card["card_id"] for card in engine.state.discard_pile])
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Defend_R", "Strike_R"])
        self.assertEqual([card["upgrades"] for card in engine.state.hand], [0, 1])

    def test_armaments_temporarily_removes_unupgradable_cards_and_restores_order(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Dazed", uuid="dazed-0"),
            make_card("Defend_R", uuid="defend"),
            make_card("Strike_R", uuid="strike"),
            make_card("Armaments", uuid="armaments"),
            make_card("Dazed", uuid="dazed-1"),
        ]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Armaments")
        engine.step(action)
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Defend_R", "Strike_R"])
        self.assertEqual([candidate["card_id"] for candidate in engine.legal_actions()], ["Defend_R", "Strike_R"])

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Strike_R"))

        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Defend_R", "Strike_R", "Dazed", "Dazed"])
        self.assertEqual([card["upgrades"] for card in engine.state.hand], [0, 1, 0, 0])

    def test_armaments_auto_upgrades_single_target_without_reordering_hand(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Dazed", uuid="dazed-0"),
            make_card("Strike_R", uuid="strike"),
            make_card("Armaments", uuid="armaments"),
            make_card("Dazed", uuid="dazed-1"),
        ]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Armaments")
        engine.step(action)

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Dazed", "Strike_R", "Dazed"])
        self.assertEqual([card["upgrades"] for card in engine.state.hand], [0, 1, 0])

    def test_upgraded_armaments_does_not_double_upgrade_regular_cards(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Defend_R", uuid="defend-0"),
            make_card("Strike_R", uuid="strike-0"),
            make_card("Defend_R", upgrades=1, uuid="defend-1"),
            make_card("Armaments", upgrades=1, uuid="armaments-plus"),
        ]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Armaments")
        engine.step(action)

        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Defend_R", "Strike_R", "Defend_R"])
        self.assertEqual([card["upgrades"] for card in engine.state.hand], [1, 1, 1])

    def test_upgraded_armaments_preserves_confusion_cost_for_turn(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        defend = make_card("Defend_R", uuid="confused-defend")
        defend["cost_for_turn"] = 0
        defend["cost_for_combat"] = 0
        engine.state.hand = [
            make_card("Armaments", upgrades=1, uuid="armaments-plus"),
            defend,
        ]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Armaments")
        engine.step(action)

        upgraded_defend = next(card for card in engine.state.hand if card["card_id"] == "Defend_R")
        self.assertEqual(upgraded_defend["upgrades"], 1)
        self.assertEqual(upgraded_defend["cost_for_turn"], 0)
        self.assertEqual(upgraded_defend["cost"], 1)

    def test_played_cards_preserve_confusion_cost_in_discard_pile(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        spot = make_card("Spot Weakness", upgrades=1, uuid="confused-spot")
        spot["cost_for_turn"] = 2
        engine.state.hand = [spot]
        engine.state.monsters[0].intent = "ATTACK"
        engine.state.monsters[0].move_adjusted_damage = 6
        engine.player.energy = 4

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Spot Weakness")
        engine.step(action)

        discarded = next(card for card in engine.state.discard_pile if card["card_id"] == "Spot Weakness")
        self.assertEqual(discarded["cost_for_turn"], 2)
        self.assertEqual(discarded["cost"], 1)

    def test_exhausted_cards_preserve_confusion_cost_in_exhaust_pile(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        seeing_red = make_card("Seeing Red", upgrades=1, uuid="confused-seeing-red")
        seeing_red["cost_for_turn"] = 1
        engine.state.hand = [seeing_red]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Seeing Red")
        engine.step(action)

        exhausted = next(card for card in engine.state.exhaust_pile if card["card_id"] == "Seeing Red")
        self.assertEqual(exhausted["cost_for_turn"], 1)
        self.assertEqual(exhausted["cost"], 0)

    def test_exhume_auto_returns_single_non_exhume_card(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        seeing_red = make_card("Seeing Red", upgrades=1, uuid="exhausted-seeing-red")
        engine.state.hand = [make_card("Exhume", upgrades=1, uuid="exhume-plus")]
        engine.state.exhaust_pile = [seeing_red]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Exhume")
        engine.step(action)

        self.assertIsNone(engine.pending_card_select)
        self.assertTrue(any(card["card_id"] == "Seeing Red" for card in engine.state.hand))
        self.assertTrue(any(card["card_id"] == "Exhume" for card in engine.state.exhaust_pile))

    def test_metallicize_end_turn_block_is_not_double_counted(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.block = 8
        engine.player.powers = [{"power_id": "Metallicize", "amount": 3, "misc": 3}]
        engine._apply_player_end_of_turn_powers()
        self.assertEqual(engine.player.block, 11)

    def test_barricade_keeps_player_block_into_next_turn(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Barricade", uuid="barricade")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Barricade")
        engine.step(action)
        self.assertEqual(_power_amount(engine.player, "Barricade"), -1)
        engine.player.block = 12
        engine._start_player_turn()
        self.assertEqual(engine.player.block, 12)

    def test_impervious_grants_large_block(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Impervious", uuid="impervious")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Impervious")
        engine.step(action)
        self.assertGreaterEqual(engine.player.block, 30)

    def test_juggernaut_triggers_when_block_is_gained(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Juggernaut", uuid="juggernaut"), make_card("Defend_R", uuid="defend")]
        engine.player.energy = 3
        juggernaut_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Juggernaut")
        engine.step(juggernaut_action)
        before = engine.state.monsters[0].current_hp
        defend_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Defend_R")
        engine.step(defend_action)
        self.assertLess(engine.state.monsters[0].current_hp, before)

    def test_juggernaut_damage_does_not_decrement_monster_plated_armor(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.powers = [{"power_id": "Plated Armor", "amount": 2}]
        engine.state.hand = [make_card("Juggernaut", uuid="juggernaut"), make_card("Defend_R", uuid="defend")]
        engine.player.energy = 3

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Juggernaut"))
        before = monster.current_hp
        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Defend_R"))

        self.assertLess(monster.current_hp, before)
        self.assertEqual(_power_amount(monster, "Plated Armor"), 2)

    def test_iron_wave_juggernaut_damage_resolves_before_malleable_block(self):
        engine = CombatEngine(
            encounter_name="Snake Plant",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        snake = engine.state.monsters[0]
        snake.current_hp = 32
        snake.max_hp = 76
        snake.block = 0
        snake.powers = [{"power_id": "Malleable", "amount": 3, "misc": 3}]
        engine.state.hand = [make_card("Iron Wave", uuid="iron-wave")]
        engine.player.energy = 3
        combat_engine_mod._add_power(engine.player, "Juggernaut", 5)

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Iron Wave"))

        self.assertEqual(snake.current_hp, 22)
        self.assertEqual(snake.block, 3)
        self.assertEqual(_power_amount(snake, "Malleable"), 4)

    def test_flex_grants_temporary_strength(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Flex", uuid="flex")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Flex")
        engine.step(action)
        self.assertEqual(_power_amount(engine.player, "Strength"), 2)
        engine.step({"kind": "end", "name": "END_TURN"})
        self.assertEqual(_power_amount(engine.player, "Strength"), 0)

    def test_brutality_loses_hp_and_draws_on_turn_start(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Brutality", uuid="brutality")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Brutality")
        engine.step(action)
        engine.state.hand = []
        engine.state.draw_pile = [make_card("Strike_R", uuid=f"draw-{i}") for i in range(6)]
        before_hp = engine.player.current_hp
        engine._start_player_turn()
        self.assertEqual(engine.player.current_hp, before_hp - 1)
        self.assertEqual(len(engine.state.hand), 6)

    def test_brutality_stacks_and_uses_stacked_amount_on_turn_start(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Brutality", uuid="brutality-1"),
            make_card("Brutality", uuid="brutality-2"),
        ]
        engine.player.energy = 3
        for _ in range(2):
            action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Brutality")
            engine.step(action)
        self.assertEqual(_power_amount(engine.player, "Brutality"), 2)
        self.assertEqual(
            [power["power_id"] for power in engine.player.powers if power.get("power_id") == "Brutality"],
            ["Brutality"],
        )

        engine.state.hand = []
        engine.state.draw_pile = [make_card("Strike_R", uuid=f"draw-{i}") for i in range(8)]
        before_hp = engine.player.current_hp
        engine._start_player_turn()

        self.assertEqual(engine.player.current_hp, before_hp - 2)
        self.assertEqual(len(engine.state.hand), 7)

    def test_brutality_stacked_turn_start_hp_loss_triggers_amount_based_side_effects(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.powers = [
            {"power_id": "Brutality", "amount": 2, "misc": 2},
            {"power_id": "Rupture", "amount": 1, "misc": 1},
        ]
        engine.state.hand = []
        engine.state.draw_pile = [make_card("Strike_R", uuid=f"draw-{i}") for i in range(8)]
        engine.state.discard_pile = [make_card("Blood for Blood", uuid="blood-for-blood")]
        before_hp = engine.player.current_hp

        engine._start_player_turn()

        self.assertEqual(engine.player.current_hp, before_hp - 2)
        self.assertEqual(_power_amount(engine.player, "Strength"), 1)
        self.assertEqual(len(engine.state.hand), 7)
        self.assertEqual(engine.state.discard_pile[0]["cost_for_combat"], 3)

    def test_end_turn_discards_hand_from_top_to_bottom(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Defend_R", uuid="hand-0"),
            make_card("Strike_R", uuid="hand-1"),
            make_card("Bash", uuid="hand-2"),
        ]
        engine.state.discard_pile = [make_card("Flash of Steel", uuid="discard-0")]
        engine.state.draw_pile = []
        original_monster_turns = engine._take_monster_turns
        original_start_turn = engine._start_player_turn
        engine._take_monster_turns = lambda: None
        engine._start_player_turn = lambda: None
        try:
            engine._end_turn()
        finally:
            engine._take_monster_turns = original_monster_turns
            engine._start_player_turn = original_start_turn

        self.assertEqual(
            [card["card_id"] for card in engine.state.discard_pile],
            ["Flash of Steel", "Bash", "Strike_R", "Defend_R"],
        )

    def test_infernal_blade_adds_free_attack_to_hand(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": [],
                "SRC_UNCOMMON": [],
                "SRC_RARE": ["Immolate"],
                "SRC_COLORLESS": [],
                "SRC_CURSE": [],
            },
        )
        engine.state.hand = [make_card("Infernal Blade", uuid="infernal-blade")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Infernal Blade")
        engine.step(action)
        generated = next(card for card in engine.state.hand if card["card_id"] != "Infernal Blade")
        visible_generated = next(card for card in _serialize_combat_state(engine.state)["hand"] if card["card_id"] != "Infernal Blade")
        self.assertEqual(generated["type"], "ATTACK")
        self.assertEqual(generated.get("cost"), 2)
        self.assertEqual(generated.get("cost_for_turn"), 0)
        self.assertFalse(generated.get("free_to_play_once"))
        self.assertEqual(visible_generated.get("cost"), 0)
        generated_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Immolate")
        engine.step(generated_action)
        discarded = next(card for card in engine.state.discard_pile if card["card_id"] == "Immolate")
        visible_discarded = next(card for card in _serialize_combat_state(engine.state)["discard_pile"] if card["card_id"] == "Immolate")
        self.assertEqual(discarded.get("cost_for_turn"), 2)
        self.assertFalse(discarded.get("free_to_play_once"))
        self.assertEqual(visible_discarded.get("cost"), 2)

    def test_infernal_blade_uses_injected_source_card_pool(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": ["Clash"],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": [],
                "SRC_CURSE": [],
            },
        )
        engine.state.hand = [make_card("Infernal Blade", uuid="infernal-blade")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Infernal Blade")
        engine.step(action)
        generated = next(card for card in engine.state.hand if card["card_id"] != "Infernal Blade")
        self.assertEqual(generated["card_id"], "Clash")

    def test_infernal_blade_keeps_x_cost_generated_attacks_as_x(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": [],
                "SRC_UNCOMMON": ["Whirlwind"],
                "SRC_RARE": [],
                "SRC_COLORLESS": [],
                "SRC_CURSE": [],
            },
        )
        engine.state.hand = [make_card("Infernal Blade", uuid="infernal-blade")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Infernal Blade")
        engine.step(action)

        generated = next(card for card in engine.state.hand if card["card_id"] == "Whirlwind")
        visible_generated = next(card for card in _serialize_combat_state(engine.state)["hand"] if card["card_id"] == "Whirlwind")
        self.assertIsNone(generated.get("cost_for_turn"))
        self.assertFalse(generated.get("free_to_play_once"))
        self.assertEqual(visible_generated.get("cost"), -1)

        whirlwind_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Whirlwind")
        engine.player.energy = 2
        engine.step(whirlwind_action)
        self.assertEqual(engine.player.energy, 0)

    def test_infernal_blade_hand_overflow_generated_attack_goes_to_discard_free_this_turn(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": [],
                "SRC_UNCOMMON": [],
                "SRC_RARE": ["Immolate"],
                "SRC_COLORLESS": [],
                "SRC_CURSE": [],
            },
        )
        engine.state.hand = [
            make_card("Infernal Blade", uuid="infernal-blade"),
            *[make_card("Defend_R", uuid=f"infernal-filler-{index}") for index in range(10)],
        ]
        engine.state.discard_pile = []
        engine.player.energy = 3

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Infernal Blade"))

        generated = next(card for card in engine.state.discard_pile if card["card_id"] == "Immolate")
        self.assertEqual(len(engine.state.hand), 10)
        self.assertEqual(generated.get("cost"), 2)
        self.assertEqual(generated.get("cost_for_turn"), 0)
        self.assertFalse(generated.get("free_to_play_once"))

    def test_burning_pact_exhausts_card_and_draws(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Burning Pact", uuid="burning-pact"),
            make_card("Strike_R", uuid="burning-target"),
        ]
        engine.state.draw_pile = [make_card("Defend_R", uuid=f"draw-{i}") for i in range(3)]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Burning Pact")
        engine.step(action)
        self.assertIsNotNone(engine.pending_card_select)
        choices = engine.legal_actions()
        self.assertEqual([choice["card_id"] for choice in choices], ["Strike_R"])
        engine.step(choices[0])
        self.assertIsNone(engine.pending_card_select)
        self.assertTrue(any(card["card_id"] == "Strike_R" for card in engine.state.exhaust_pile))
        self.assertGreaterEqual(len(engine.state.hand), 2)

    def test_burning_pact_shuffle_excludes_currently_played_card(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Burning Pact", uuid="burning-pact"),
            make_card("Defend_R", uuid="burning-target"),
        ]
        engine.state.draw_pile = []
        engine.state.discard_pile = [
            make_card("Strike_R", uuid="discard-strike"),
            make_card("Bash", uuid="discard-bash"),
        ]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Burning Pact")
        engine.step(action)
        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Defend_R"))
        self.assertTrue(any(card.get("uuid") == "burning-pact" for card in engine.state.discard_pile))
        self.assertFalse(any(card.get("uuid") == "burning-pact" for card in engine.state.draw_pile))

    def test_dual_wield_copies_attack_or_power(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Dual Wield", uuid="dual-wield"),
            make_card("Strike_R", uuid="dual-target"),
        ]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Dual Wield")
        engine.step(action)
        self.assertGreaterEqual(sum(1 for card in engine.state.hand if card["card_id"] == "Strike_R"), 2)

    def test_dual_wield_opens_card_select_when_multiple_targets_exist(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Dual Wield", uuid="dual-wield"),
            make_card("Strike_R", uuid="dual-attack"),
            make_card("Rage", uuid="dual-skill"),
            make_card("Inflame", uuid="dual-power"),
            make_card("Defend_R", uuid="dual-skill"),
        ]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Dual Wield")
        engine.step(action)

        self.assertIsNotNone(engine.pending_card_select)
        choices = engine.legal_actions()
        self.assertEqual([choice["card_id"] for choice in choices], ["Strike_R", "Inflame"])
        engine.step(choices[1])
        self.assertIsNone(engine.pending_card_select)
        self.assertEqual(sum(1 for card in engine.state.hand if card["card_id"] == "Inflame"), 2)
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Strike_R", "Rage", "Defend_R", "Inflame", "Inflame"])

    def test_dual_wield_single_target_hand_overflow_copy_goes_to_discard_preserving_payload(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80, energy=3, base_energy=3),
            master_deck=[],
        )
        target = make_card("Strike_R", upgrades=1, uuid="dual-target")
        target["misc"] = 7
        target["cost_for_turn"] = 0
        target["free_to_play_once"] = True
        engine.state.hand = [
            make_card("Dual Wield", upgrades=1, uuid="dual-wield"),
            target,
            *[make_card("Defend_R", uuid=f"dual-filler-{index}") for index in range(8)],
        ]
        engine.state.discard_pile = []
        engine.player.energy = 3

        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Dual Wield"))

        self.assertEqual(len(engine.state.hand), 10)
        overflow = next(card for card in engine.state.discard_pile if card["uuid"] == "dual-wield-copy-1-Strike_R")
        self.assertEqual(overflow["upgrades"], 1)
        self.assertEqual(overflow["misc"], 7)
        self.assertEqual(overflow["cost_for_turn"], 0)
        self.assertTrue(overflow["free_to_play_once"])

    def test_dual_wield_card_select_hand_overflow_copy_goes_to_discard_preserving_payload(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80, energy=3, base_energy=3),
            master_deck=[],
        )
        target = make_card("Strike_R", upgrades=1, uuid="dual-target")
        target["misc"] = 7
        target["cost_for_turn"] = 0
        target["free_to_play_once"] = True
        engine.state.hand = [
            make_card("Dual Wield", upgrades=1, uuid="dual-wield"),
            target,
            make_card("Inflame", uuid="dual-other-target"),
            *[make_card("Defend_R", uuid=f"dual-filler-{index}") for index in range(7)],
        ]
        engine.state.discard_pile = []
        engine.player.energy = 3

        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Dual Wield"))
        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Strike_R"))

        self.assertEqual(len(engine.state.hand), 10)
        overflow = next(card for card in engine.state.discard_pile if card["uuid"] == "dual-wield-copy-2-Strike_R")
        self.assertEqual(overflow["upgrades"], 1)
        self.assertEqual(overflow["misc"], 7)
        self.assertEqual(overflow["cost_for_turn"], 0)
        self.assertTrue(overflow["free_to_play_once"])
        self.assertEqual(sum(1 for card in engine.state.hand if card["card_id"] == "Strike_R"), 2)

    def test_power_through_adds_wounds_and_block(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Power Through", uuid="power-through")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Power Through")
        engine.step(action)
        self.assertGreaterEqual(engine.player.block, 15)
        self.assertGreaterEqual(sum(1 for card in engine.state.hand if card["card_id"] == "Wound"), 2)

    def test_impatience_draws_only_without_attack_in_hand(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Impatience", uuid="impatience")]
        engine.state.draw_pile = [make_card("Defend_R", uuid=f"draw-{i}") for i in range(3)]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Impatience")
        engine.step(action)
        self.assertGreaterEqual(len(engine.state.hand), 2)

    def test_snake_plant_encounter_generates_and_turn_cycles(self):
        engine = CombatEngine(
            encounter_name="Snake Plant",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(engine.state.monsters[0].monster_id, "SnakePlant")
        engine.step({"kind": "end", "name": "END_TURN"})
        self.assertIn(engine.state.monsters[0].next_move, {"CHOMPY_CHOMPS", "SPORES"})

    def test_snake_plant_attacks_use_monster_weak_scaling(self):
        engine = CombatEngine(
            encounter_name="Snake Plant",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=18, max_hp=80, block=5),
            master_deck=[],
        )
        snake = engine.state.monsters[0]
        engine.player.block = 5
        snake.next_move = "CHOMPY_CHOMPS"
        combat_engine_mod._add_power(snake, "Weakened", 2)

        engine._snake_plant_take_turn(snake)

        self.assertEqual(engine.player.current_hp, 8)
        self.assertEqual(engine.player.block, 0)

    def test_malleable_current_amount_does_not_update_base_amount(self):
        engine = CombatEngine(
            encounter_name="Snake Plant",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        snake = engine.state.monsters[0]

        engine._player_deal_damage(snake, 10)
        malleable = next(power for power in snake.powers if power.get("power_id") == "Malleable")
        self.assertEqual(malleable["amount"], 4)
        self.assertEqual(malleable["misc"], 3)

        combat_engine_mod._apply_end_of_turn_temporary_powers(snake)

        self.assertEqual(_power_amount(snake, "Malleable"), 3)

    def test_monster_powers_keep_insertion_order_for_debuffs_and_buffs(self):
        engine = CombatEngine(
            encounter_name="Snake Plant",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        snake = engine.state.monsters[0]

        combat_engine_mod._add_power(snake, "Strength", 2)
        engine._apply_monster_debuff(snake, "Vulnerable", 2)

        self.assertEqual(
            [power["power_id"] for power in snake.powers],
            ["Malleable", "Strength", "Vulnerable"],
        )

    def test_shelled_parasite_and_fungi_encounter_generates(self):
        engine = CombatEngine(
            encounter_name="Shelled Parasite and Fungi",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        ids = {monster.monster_id for monster in engine.state.monsters}
        self.assertEqual(ids, {"ShelledParasite", "FungiBeast"})

    def test_shelled_parasite_rolls_real_opening_move(self):
        engine = CombatEngine(
            encounter_name="Shell Parasite",
            randoms=NativeRandomSet(seed=12, floor=20),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        parasite = engine.state.monsters[0]

        self.assertEqual(engine.randoms.stream("ai").counter, 2)
        self.assertEqual(parasite.next_move, "LIFE_SUCK")
        self.assertEqual(parasite.intent, "ATTACK_BUFF")
        self.assertEqual(parasite.move_adjusted_damage, 10)
        self.assertEqual(parasite.move_hits, 1)

        env = NativeCombatEnv(
            seed=12,
            floor=20,
            encounter_name="Shell Parasite",
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(env.serialize()["combat_state"]["monsters"][0]["monster_id"], "Shelled Parasite")

    def test_shelled_parasite_reroll_after_repeated_fell_uses_reroll_bucket(self):
        engine = CombatEngine(
            encounter_name="Shell Parasite",
            randoms=NativeRandomSet(seed=1),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        parasite = engine.state.monsters[0]
        parasite.move_history = ["FELL"]

        class FixedAi:
            def __init__(self):
                self.rolls = [10, 75]

            def random(self, *args):
                return self.rolls.pop(0)

        engine.randoms.streams["ai"] = FixedAi()

        engine._shelled_parasite_roll_next_move(parasite)

        self.assertEqual(parasite.next_move, "LIFE_SUCK")

    def test_monster_plated_armor_decrements_after_hp_damage_and_breaks(self):
        engine = CombatEngine(
            encounter_name="Shell Parasite",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        parasite = engine.state.monsters[0]
        parasite.block = 0
        parasite.powers = [{"power_id": "Plated Armor", "amount": 2}]

        engine._player_deal_damage(parasite, 1)
        self.assertEqual(_power_amount(parasite, "Plated Armor"), 1)

        engine._player_deal_damage(parasite, 1)
        self.assertEqual(_power_amount(parasite, "Plated Armor"), 0)
        self.assertEqual(parasite.next_move, "STUNNED")
        self.assertEqual(parasite.intent, "STUN")

    def test_shelled_parasite_attacks_use_monster_weak_scaling(self):
        engine = CombatEngine(
            encounter_name="Shell Parasite",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=56, max_hp=80, block=3),
            master_deck=[],
        )
        parasite = engine.state.monsters[0]
        engine.player.block = 3
        parasite.next_move = "FELL"
        parasite.powers.append({"power_id": "Weakened", "amount": 2})

        engine._shelled_parasite_take_turn(parasite)

        self.assertEqual(engine.player.current_hp, 46)
        self.assertEqual(_power_amount(engine.player, "Frail"), 2)

    def test_shelled_parasite_life_suck_heals_before_player_thorns(self):
        engine = CombatEngine(
            encounter_name="Shell Parasite",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=50, max_hp=80),
            master_deck=[],
        )
        parasite = engine.state.monsters[0]
        parasite.current_hp = 66
        parasite.max_hp = 72
        parasite.block = 0
        parasite.next_move = "LIFE_SUCK"
        parasite.meta["suck_damage"] = 10
        combat_engine_mod._add_power(engine.player, "Thorns", 3)

        engine._shelled_parasite_take_turn(parasite)

        self.assertEqual(engine.player.current_hp, 40)
        self.assertEqual(parasite.current_hp, 69)

    def test_centurion_and_healer_encounter_generates(self):
        engine = CombatEngine(
            encounter_name="Centurion and Healer",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        ids = {monster.monster_id for monster in engine.state.monsters}
        self.assertEqual(ids, {"Centurion", "Healer"})

    def test_rage_grants_block_when_attack_is_played(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Rage", uuid="rage"), make_card("Strike_R", uuid="rage-strike")]
        engine.player.energy = 3
        rage_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Rage")
        engine.step(rage_action)
        strike_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Strike_R")
        engine.step(strike_action)
        self.assertGreater(engine.player.block, 0)

    def test_rage_is_removed_at_player_end_of_turn(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Rage", uuid="rage")]
        engine.player.energy = 3
        rage_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Rage")
        engine.step(rage_action)
        self.assertGreater(_power_amount(engine.player, "Rage"), 0)

        engine.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(_power_amount(engine.player, "Rage"), 0)

    def test_second_wind_exhausts_non_attacks_and_grants_block(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Second Wind", uuid="second-wind"),
            make_card("Defend_R", uuid="sw-defend"),
            make_card("Inflame", uuid="sw-inflame"),
            make_card("Strike_R", uuid="sw-strike"),
        ]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Second Wind")
        engine.step(action)
        self.assertTrue(all(card.get("type") == "ATTACK" for card in engine.state.hand))
        self.assertGreaterEqual(len(engine.state.exhaust_pile), 2)
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Inflame", "Defend_R"])
        self.assertGreater(engine.player.block, 0)

    def test_second_wind_triggers_juggernaut_for_each_exhausted_card(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.monsters = [
            MonsterState(monster_id="Cultist", name="Cultist", current_hp=15, max_hp=50)
        ]
        engine.state.hand = [
            make_card("Second Wind", uuid="second-wind"),
            make_card("Defend_R", uuid="sw-defend"),
            make_card("Inflame", uuid="sw-inflame"),
            make_card("Strike_R", uuid="sw-strike"),
        ]
        engine.player.energy = 3
        combat_engine_mod._add_power(engine.player, "Juggernaut", 7)

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Second Wind")
        engine.step(action)

        self.assertEqual(engine.player.block, 10)
        self.assertEqual(engine.state.monsters[0].current_hp, 1)

    def test_sever_soul_exhausts_non_attacks_in_action_stack_order(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Sever Soul", uuid="sever-soul"),
            make_card("Defend_R", uuid="sever-defend"),
            make_card("Inflame", uuid="sever-inflame"),
            make_card("Strike_R", uuid="sever-strike"),
        ]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Sever Soul")
        engine.step(action)
        self.assertTrue(all(card.get("type") == "ATTACK" for card in engine.state.hand))
        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile], ["Inflame", "Defend_R"])

    def test_hemokinesis_loses_hp_and_deals_damage(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=30, max_hp=30),
            master_deck=[],
        )
        engine.state.hand = [make_card("Hemokinesis", uuid="hemokinesis")]
        engine.player.energy = 3
        target = engine.state.monsters[0]
        before_hp = target.current_hp
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Hemokinesis")
        engine.step(action)
        self.assertEqual(engine.player.current_hp, 27)
        self.assertLess(target.current_hp, before_hp)

    def test_carnage_exhausts_if_left_in_hand_at_end_of_turn(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Carnage", uuid="carnage")]
        engine._apply_player_end_of_turn_powers()
        ethereal_cards = [card for card in engine.state.hand if bool(card.get("ethereal"))]
        non_ethereal_cards = [card for card in engine.state.hand if not bool(card.get("ethereal"))]
        engine.state.exhaust_pile.extend(ethereal_cards)
        engine.state.discard_pile.extend(non_ethereal_cards)
        engine.state.hand = []
        self.assertTrue(any(card["card_id"] == "Carnage" for card in engine.state.exhaust_pile))

    def test_end_turn_ethereal_exhaust_uses_top_card_order(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Carnage", uuid="ethereal-carnage"),
            make_card("Dazed", uuid="ethereal-dazed"),
        ]
        engine.state.draw_pile = []
        engine.state.discard_pile = []

        engine._end_turn()

        self.assertEqual([card["card_id"] for card in engine.state.exhaust_pile[:2]], ["Dazed", "Carnage"])

    def test_uppercut_applies_weak_and_vulnerable(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Uppercut", uuid="uppercut")]
        engine.player.energy = 3
        target = engine.state.monsters[0]
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Uppercut")
        engine.step(action)
        self.assertGreater(_power_amount(target, "Weakened"), 0)
        self.assertGreater(_power_amount(target, "Vulnerable"), 0)
        self.assertEqual([power["power_id"] for power in target.powers], ["Vulnerable", "Weakened"])

    def test_uppercut_places_vulnerable_before_existing_monster_weak(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        combat_engine_mod._add_power(target, "Weak", 1)
        engine.state.hand = [make_card("Uppercut", uuid="uppercut")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Uppercut")
        engine.step(action)

        self.assertEqual(
            [(power["power_id"], power["amount"]) for power in target.powers],
            [("Vulnerable", 1), ("Weakened", 2)],
        )

    def test_uppercut_artifact_blocks_weak_before_vulnerable(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        target = engine.state.monsters[0]
        combat_engine_mod._add_power(target, "Artifact", 1)
        engine.state.hand = [make_card("Uppercut", uuid="uppercut")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Uppercut")
        engine.step(action)

        self.assertEqual(_power_amount(target, "Artifact"), 0)
        self.assertEqual(_power_amount(target, "Weakened"), 0)
        self.assertEqual(_power_amount(target, "Vulnerable"), 1)

    def test_feed_increases_max_hp_on_kill(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=40, max_hp=40),
            master_deck=[],
        )
        engine.state.hand = [make_card("Feed", uuid="feed")]
        engine.player.energy = 3
        target = engine.state.monsters[0]
        target.current_hp = 1
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Feed")
        engine.step(action)
        self.assertEqual(engine.player.max_hp, 43)
        self.assertEqual(engine.player.current_hp, 43)

    def test_upgraded_feed_increases_max_hp_by_upgraded_magic_on_kill(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=40, max_hp=40),
            master_deck=[],
        )
        engine.state.hand = [make_card("Feed", upgrades=1, uuid="feed-plus")]
        engine.player.energy = 3
        target = engine.state.monsters[0]
        target.current_hp = 1
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Feed")
        engine.step(action)
        self.assertEqual(engine.player.max_hp, 44)
        self.assertEqual(engine.player.current_hp, 44)

    def test_feed_does_not_increase_max_hp_on_minion_kill(self):
        engine = CombatEngine(
            encounter_name="Gremlin Leader",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=40, max_hp=40),
            master_deck=[],
        )
        engine.state.hand = [make_card("Feed", uuid="feed")]
        engine.player.energy = 3
        target = next(monster for monster in engine.state.monsters if _power_amount(monster, "Minion") != 0)
        target.current_hp = 1

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Feed")
        engine.step(action)

        self.assertEqual(engine.player.max_hp, 40)
        self.assertEqual(engine.player.current_hp, 40)

    def test_city_event_dispatch_can_generate_supported_event(self):
        event = EventState("The Mausoleum")
        actions = event.actions(ascension_level=0, max_hp=80)
        self.assertEqual([action["name"] for action in actions], ["Opened", "Ignored"])

    def test_mausoleum_open_grants_relic_and_optional_writhe(self):
        event = EventState("The Mausoleum")
        calls: list[str] = []

        def normal_relic_drawer(tier: str) -> dict[str, object]:
            raise AssertionError(f"Mausoleum should use screenless relic drawer, got normal tier {tier}")

        def screenless_relic_drawer(tier: str | None = None, exclude: set[str] | None = None) -> dict[str, object]:
            del exclude
            calls.append(str(tier))
            return make_relic("Lantern")

        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=15,
            current_hp=72,
            max_hp=80,
            gold=99,
            relic_drawer=normal_relic_drawer,
            screenless_relic_drawer=screenless_relic_drawer,
        )
        self.assertEqual(event.screen, "RESULT")
        self.assertEqual([relic["relic_id"] for relic in result["add_relics"]], ["Lantern"])
        self.assertEqual(len(calls), 1)
        self.assertIn(calls[0], {"COMMON", "UNCOMMON", "RARE"})
        self.assertEqual(result["add_cards"][0]["card_id"], "Writhe")

    def test_wheel_of_change_actions_expose_real_stages(self):
        event = EventState("Wheel of Change")
        self.assertEqual(event.actions(ascension_level=0, max_hp=80)[0]["label"], "Play")
        event.screen = "SPIN"
        self.assertEqual(event.actions(ascension_level=0, max_hp=80)[0]["label"], "Spin")
        event.screen = "RESULT"
        self.assertEqual(event.actions(ascension_level=0, max_hp=80)[0]["label"], "Prize?")
        event.screen = "LEAVE"
        self.assertEqual(event.actions(ascension_level=0, max_hp=80)[0]["label"], "Leave")

    def test_wheel_of_change_relic_result_opens_relic_reward(self):
        event = EventState("Wheel of Change")
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=7),
            ascension_level=0,
            current_hp=72,
            max_hp=80,
            gold=99,
        )
        self.assertEqual(event.screen, "SPIN")
        self.assertFalse(result.get("open_rewards", False))
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=7),
            ascension_level=0,
            current_hp=72,
            max_hp=80,
            gold=99,
        )
        self.assertEqual(event.screen, "RESULT")
        if event.result == 1:
            self.assertFalse(result.get("open_rewards", False))
            result = resolve_event_choice(
                event,
                action_index=0,
                randoms=NativeRandomSet(seed=7),
                ascension_level=0,
                current_hp=72,
                max_hp=80,
                gold=99,
            )
            self.assertEqual(event.screen, "LEAVE")
            self.assertTrue(result["open_rewards"])
            self.assertTrue(result["relic_rewards"])
        elif event.result == 4:
            self.assertFalse(result.get("open_card_select", False))

    def test_wheel_of_change_purge_result_opens_card_select(self):
        event = EventState("Wheel of Change")
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=13),
            ascension_level=0,
            current_hp=72,
            max_hp=80,
            gold=99,
            deck=[make_card("Strike_R", uuid="purge-0"), make_card("Defend_R", uuid="purge-1")],
        )
        self.assertEqual(event.screen, "SPIN")
        self.assertFalse(result.get("open_card_select", False))
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=13),
            ascension_level=0,
            current_hp=72,
            max_hp=80,
            gold=99,
            deck=[make_card("Strike_R", uuid="purge-0"), make_card("Defend_R", uuid="purge-1")],
        )
        self.assertEqual(event.screen, "RESULT")
        if event.result == 4:
            self.assertFalse(result.get("open_card_select", False))
            result = resolve_event_choice(
                event,
                action_index=0,
                randoms=NativeRandomSet(seed=13),
                ascension_level=0,
                current_hp=72,
                max_hp=80,
                gold=99,
                deck=[make_card("Strike_R", uuid="purge-0"), make_card("Defend_R", uuid="purge-1")],
            )
            self.assertEqual(event.screen, "LEAVE")
            self.assertTrue(result["open_card_select"])
            self.assertEqual(result["card_select_mode"], "purge")
            self.assertEqual(result["candidate_indexes"], [0, 1])
            self.assertEqual(result["return_phase"], "EVENT")
            self.assertFalse(result["clear_event_on_finish"])

    def test_wheel_of_change_purge_result_uses_native_purge_filter(self):
        bottled_bash = make_card("Bash", uuid="bottled-bash")
        bottled_bash["in_bottle_flame"] = True
        event = EventState("Wheel of Change", screen="RESULT", result=4)
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=72,
            max_hp=80,
            gold=99,
            deck=[
                make_card("Strike_R", uuid="strike"),
                make_card("Necronomicurse", uuid="necro"),
                bottled_bash,
                make_card("Injury", uuid="injury"),
            ],
        )

        self.assertEqual(event.screen, "LEAVE")
        self.assertTrue(result["open_card_select"])
        self.assertEqual(result["candidate_indexes"], [0, 3])
        self.assertEqual(result["card_select_mode"], "purge")

    def test_wheel_of_change_curse_result_applies_on_result_click_not_spin(self):
        event = EventState("Wheel of Change", screen="RESULT", result=3)
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=72,
            max_hp=80,
            gold=99,
        )

        self.assertEqual(event.screen, "LEAVE")
        self.assertEqual(result["add_cards"][0]["card_id"], "Decay")

    def test_wheel_of_change_heal_result_obeys_mark_of_the_bloom(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=True)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.relics = [make_relic("Mark of the Bloom")]
        env.current_event = EventState("Wheel of Change", screen="RESULT", result=2)

        env._step_event({"kind": "event", "event_id": "Wheel of Change", "choice_index": 0})

        self.assertEqual(env.current_event.screen, "LEAVE")
        self.assertEqual(env.player.current_hp, 40)

    def test_hexaghost_vertical_slice_advances_to_divider(self):
        engine = CombatEngine(
            encounter_name="Hexaghost",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        self.assertEqual(monster.next_move, "ACTIVATE")
        engine._hexaghost_take_turn(monster)
        self.assertEqual(monster.next_move, "DIVIDER")
        self.assertEqual(monster.meta["divider_damage"], 7)

    def test_hexaghost_divider_intent_applies_weak(self):
        engine = CombatEngine(
            encounter_name="Hexaghost",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=20, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        engine._hexaghost_take_turn(monster)
        engine._update_monster_intents()
        self.assertEqual(monster.next_move, "DIVIDER")
        self.assertEqual(monster.move_adjusted_damage, 2)

        engine.state.hand = [make_card("Intimidate", uuid="intimidate")]
        engine._play_card({"kind": "card", "card_index": 0})

        self.assertEqual(monster.move_adjusted_damage, 1)
        self.assertEqual(monster.move_hits, 6)

    def test_hexaghost_inferno_upgrades_and_adds_burns(self):
        engine = CombatEngine(
            encounter_name="Hexaghost",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.next_move = "INFERNO"
        monster.meta["inferno_damage"] = 0
        engine.state.draw_pile = [make_card("Burn", uuid="draw-burn")]
        engine.state.discard_pile = [make_card("Burn", uuid="discard-burn")]

        engine._hexaghost_take_turn(monster)

        self.assertEqual([card["upgrades"] for card in engine.state.draw_pile if card["card_id"] == "Burn"], [1])
        self.assertEqual([card["upgrades"] for card in engine.state.discard_pile if card["card_id"] == "Burn"], [1, 1, 1, 1])

    def test_hexaghost_sear_adds_upgraded_burn_after_inferno(self):
        engine = CombatEngine(
            encounter_name="Hexaghost",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.next_move = "SEAR"
        monster.meta["sear_damage"] = 0
        monster.meta["burn_upgraded"] = True

        engine._hexaghost_take_turn(monster)

        burns = [card for card in engine.state.discard_pile if card["card_id"] == "Burn"]
        self.assertEqual([card["upgrades"] for card in burns], [1])
        self.assertEqual([card["base_magic"] for card in burns], [4])

    def test_perfected_strike_scales_with_strike_cards(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Perfected Strike", uuid="perfected"), make_card("Strike_R", uuid="s1"), make_card("Wild Strike", uuid="s2")]
        engine.state.draw_pile = [make_card("Twin Strike", uuid="s3")]
        engine.player.energy = 3
        target = engine.state.monsters[0]
        before_hp = target.current_hp
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Perfected Strike")
        engine.step(action)
        self.assertLessEqual(target.current_hp, before_hp - 12)

    def test_perfected_strike_counts_played_card_before_leaving_hand(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Perfected Strike", uuid="perfected")]
        engine.state.draw_pile = []
        engine.state.discard_pile = []
        engine.player.energy = 3
        target = engine.state.monsters[0]
        before_hp = target.current_hp
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Perfected Strike")
        engine.step(action)
        self.assertEqual(target.current_hp, before_hp - 8)

    def test_city_encounters_expose_real_monsters(self):
        chosen_engine = CombatEngine(
            encounter_name="Chosen",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(chosen_engine.state.monsters[0].monster_id, "Chosen")
        sphere_engine = CombatEngine(
            encounter_name="Sentry and Sphere",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertCountEqual([monster.monster_id for monster in sphere_engine.state.monsters], ["Sentry", "SphericGuardian"])
        sentry = next(monster for monster in sphere_engine.state.monsters if monster.monster_id == "Sentry")
        self.assertEqual(sentry.next_move, "BOLT")
        self.assertEqual(sentry.intent, "DEBUFF")
        sphere = next(monster for monster in sphere_engine.state.monsters if monster.monster_id == "SphericGuardian")
        self.assertEqual(_power_amount(sphere, "Barricade"), -1)
        book_engine = CombatEngine(
            encounter_name="Book of Stabbing",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(book_engine.state.monsters[0].monster_id, "BookOfStabbing")

    def test_sentry_beam_applies_monster_strength_when_dealing_damage(self):
        engine = CombatEngine(
            encounter_name="3 Sentries",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=73, max_hp=80),
            master_deck=[],
        )
        sentry = engine.state.monsters[1]
        sentry.next_move = "BEAM"
        sentry.meta["beam_damage"] = 9
        combat_engine_mod._append_power(sentry, "Strength", 2, misc=2)
        engine.player.block = 10

        engine._sentry_take_turn(sentry)

        self.assertEqual(engine.player.current_hp, 72)
        self.assertEqual(engine.player.block, 0)

    def test_spheric_guardian_big_attack_applies_monster_strength_to_each_hit(self):
        engine = CombatEngine(
            encounter_name="Spheric Guardian",
            randoms=NativeRandomSet(seed=2),
            ascension_level=2,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        sphere = engine.state.monsters[0]
        sphere.next_move = "BIG_ATTACK"
        sphere.meta["attack_damage"] = 11
        combat_engine_mod._append_power(sphere, "Strength", 1, misc=1)
        engine.player.block = 9

        engine._spheric_guardian_take_turn(sphere)

        self.assertEqual(engine.player.current_hp, 65)
        self.assertEqual(engine.player.block, 0)
        self.assertEqual(sphere.move_history[-1], "BIG_ATTACK")

    def test_slavers_taskmaster_consumes_overwritten_constructor_hp_roll(self):
        randoms = NativeRandomSet(seed=12)
        randoms.reset_floor_streams(25)
        engine = CombatEngine(
            encounter_name="Slavers",
            randoms=randoms,
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )

        self.assertEqual(
            [(monster.monster_id, monster.current_hp) for monster in engine.state.monsters],
            [("SlaverBlue", 46), ("SlaverBoss", 54), ("SlaverRed", 49)],
        )

    def test_book_of_stabbing_opening_move_uses_ai_roll_and_painful_stabs_amount(self):
        randoms = NativeRandomSet(seed=12)
        randoms.reset_floor_streams(23)
        engine = CombatEngine(
            encounter_name="Book of Stabbing",
            randoms=randoms,
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )

        book = engine.state.monsters[0]
        self.assertEqual(randoms.stream("ai").calls[0].result, 48)
        self.assertEqual(book.next_move, "STAB")
        self.assertEqual(book.move_adjusted_damage, 6)
        self.assertEqual(book.move_hits, 2)
        self.assertEqual(book.powers[0]["power_id"], "Painful Stabs")
        self.assertEqual(book.powers[0]["amount"], -1)

    def test_book_of_stabbing_big_stab_opening_when_ai_roll_is_low(self):
        randoms = NativeRandomSet(seed=1)
        randoms.reset_floor_streams(1)
        engine = CombatEngine(
            encounter_name="Book of Stabbing",
            randoms=randoms,
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )

        book = engine.state.monsters[0]
        self.assertEqual(randoms.stream("ai").calls[0].result, 9)
        self.assertEqual(book.next_move, "BIG_STAB")
        self.assertEqual(book.move_adjusted_damage, 21)
        self.assertEqual(book.move_hits, 1)

    def test_painful_stabs_adds_wound_for_each_unblocked_hit(self):
        randoms = NativeRandomSet(seed=12)
        randoms.reset_floor_streams(23)
        engine = CombatEngine(
            encounter_name="Book of Stabbing",
            randoms=randoms,
            ascension_level=0,
            player=PlayerState(current_hp=19, max_hp=80),
            master_deck=[],
        )
        engine.player.block = 5
        engine.state.hand = []
        engine.state.draw_pile = []
        engine.state.discard_pile = []
        book = engine.state.monsters[0]

        engine._monster_attack_player(book, 6, hits=2)

        self.assertEqual(engine.player.current_hp, 12)
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Wound", "Wound"])

    def test_whirlwind_consumes_all_energy_for_aoe_hits(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Whirlwind", uuid="whirlwind")]
        engine.player.energy = 3
        before_hps = [monster.current_hp for monster in engine.state.monsters]
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Whirlwind")
        engine.step(action)
        self.assertEqual(engine.player.energy, 0)
        self.assertTrue(all(monster.current_hp < before for monster, before in zip(engine.state.monsters, before_hps)))

    def test_whirlwind_chemical_x_adds_two_hits(self):
        engine = CombatEngine(
            encounter_name="3 Cultists",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            relics=[make_relic("Chemical X")],
        )
        engine.state.hand = [make_card("Whirlwind", uuid="whirlwind")]
        engine.player.energy = 1
        before_hps = [monster.current_hp for monster in engine.state.monsters]

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Whirlwind")
        engine.step(action)

        self.assertEqual(engine.player.energy, 0)
        self.assertEqual([before - monster.current_hp for monster, before in zip(engine.state.monsters, before_hps)], [15, 15, 15])

    def test_jack_of_all_trades_adds_colorless_cards_to_hand(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Jack Of All Trades", uuid="jack")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Jack Of All Trades")
        engine.step(action)
        generated = [card for card in engine.state.hand if card.get("color") == "COLORLESS"]
        self.assertGreaterEqual(len(generated), 1)

    def test_jack_of_all_trades_uses_injected_colorless_source_pool(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": [],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": ["Panacea"],
                "SRC_CURSE": [],
            },
        )
        engine.state.hand = [make_card("Jack Of All Trades", uuid="jack")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Jack Of All Trades")
        engine.step(action)
        generated = [card for card in engine.state.hand if card.get("card_id") == "Panacea"]
        self.assertEqual(len(generated), 1)

    def test_upgraded_jack_of_all_trades_hand_overflow_sends_second_generated_card_to_discard(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": [],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": ["Panacea"],
                "SRC_CURSE": [],
            },
        )
        engine.state.hand = [
            make_card("Jack Of All Trades", upgrades=1, uuid="jack"),
            *[make_card("Defend_R", uuid=f"jack-filler-{index}") for index in range(9)],
        ]
        engine.state.discard_pile = []
        engine.player.energy = 3

        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Jack Of All Trades"))

        self.assertEqual(len(engine.state.hand), 10)
        self.assertEqual(sum(1 for card in engine.state.hand if card.get("card_id") == "Panacea"), 1)
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Panacea"])

    def test_jack_of_all_trades_hand_overflow_sends_generated_card_to_discard_when_hand_already_full(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": [],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": ["Panacea"],
                "SRC_CURSE": [],
            },
        )
        engine.state.hand = [
            make_card("Jack Of All Trades", uuid="jack"),
            *[make_card("Defend_R", uuid=f"jack-filler-{index}") for index in range(10)],
        ]
        engine.state.discard_pile = []
        engine.player.energy = 3

        engine.step(next(action for action in engine.legal_actions() if action.get("card_id") == "Jack Of All Trades"))

        self.assertEqual(len(engine.state.hand), 10)
        self.assertFalse(any(card.get("card_id") == "Panacea" for card in engine.state.hand))
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Panacea"])

    def test_ghostly_applies_intangible(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=50, max_hp=50),
            master_deck=[],
        )
        engine.state.hand = [make_card("Ghostly", uuid="ghostly")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Ghostly")
        engine.step(action)
        self.assertEqual(_power_amount(engine.player, "Intangible"), 1)
        engine.state.monsters[0].next_move = "DARK_STRIKE"
        engine._update_monster_intents()
        before_hp = engine.player.current_hp
        engine._cultist_take_turn(engine.state.monsters[0])
        self.assertEqual(before_hp - engine.player.current_hp, 1)

    def test_intangible_power_stacks_without_duplicate_rollover(self):
        player = PlayerState(current_hp=50, max_hp=50)
        combat_engine_mod._append_power(player, "IntangiblePlayer", 1, misc=1)
        combat_engine_mod._add_power(player, "Intangible", 1)
        combat_engine_mod._add_power(player, "IntangiblePlayer", 1)

        intangible_powers = [
            power
            for power in player.powers
            if combat_engine_mod._canonical_power_id(str(power.get("power_id") or power.get("id") or "")) == "Intangible"
        ]
        self.assertEqual(len(intangible_powers), 1)
        self.assertEqual(_power_amount(player, "Intangible"), 3)

        combat_engine_mod._decrement_turn_powers(player)
        self.assertEqual(_power_amount(player, "Intangible"), 2)
        combat_engine_mod._decrement_turn_powers(player)
        combat_engine_mod._decrement_turn_powers(player)
        self.assertEqual(_power_amount(player, "Intangible"), 0)

    def test_corruption_sets_skill_costs_to_zero_and_exhausts_skill(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Corruption", uuid="corruption"), make_card("Defend_R", uuid="defend")]
        engine.state.draw_pile = [make_card("Defend_R", uuid="draw-defend"), make_card("Strike_R", uuid="draw-strike")]
        engine.player.energy = 3
        corruption_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Corruption")
        engine.step(corruption_action)
        corruption = next(power for power in engine.player.powers if power.get("power_id") == "Corruption")
        self.assertEqual(corruption["amount"], -1)
        defend = next(card for card in engine.state.hand if card["card_id"] == "Defend_R")
        self.assertEqual(defend["cost_for_turn"], 0)
        draw_defend = next(card for card in engine.state.draw_pile if card["card_id"] == "Defend_R")
        self.assertEqual(draw_defend["cost_for_turn"], 0)
        defend_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Defend_R")
        engine.step(defend_action)
        self.assertTrue(any(card["card_id"] == "Defend_R" for card in engine.state.exhaust_pile))

    def test_corruption_zeroes_positive_exhaust_pile_skill_costs_but_not_x_costs(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Corruption", uuid="corruption")]
        engine.state.draw_pile = [make_card("Defend_R", uuid="draw-defend")]
        engine.state.exhaust_pile = [
            make_card("Defend_R", uuid="exhaust-defend"),
            make_card("Transmutation", uuid="exhaust-transmutation"),
        ]
        engine.player.energy = 3

        corruption_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Corruption")
        engine.step(corruption_action)

        self.assertEqual(engine.state.draw_pile[0]["cost_for_turn"], 0)
        self.assertEqual(engine.state.exhaust_pile[0]["cost_for_turn"], 0)
        self.assertEqual(engine.state.exhaust_pile[0]["cost"], 0)
        self.assertIsNone(engine.state.exhaust_pile[1].get("cost_for_turn"))
        self.assertEqual(engine.state.exhaust_pile[1]["cost"], -1)

    def test_feel_no_pain_grants_block_when_cards_exhaust(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Feel No Pain", uuid="fnp"), make_card("Burning Pact", uuid="bp"), make_card("Strike_R", uuid="strike")]
        engine.player.energy = 3
        fnp_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Feel No Pain")
        engine.step(fnp_action)
        burn_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Burning Pact")
        engine.step(burn_action)
        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Strike_R"))
        self.assertGreaterEqual(engine.player.block, 3)

    def test_apotheosis_upgrades_all_combat_piles(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Apotheosis", uuid="apo"), make_card("Strike_R", uuid="hand-strike")]
        engine.state.draw_pile = [make_card("Defend_R", uuid="draw-defend")]
        engine.state.discard_pile = [make_card("Bash", uuid="discard-bash")]
        engine.state.exhaust_pile = [make_card("Ghostly", uuid="exhaust-ghostly")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Apotheosis")
        engine.step(action)
        for pile in (engine.state.hand, engine.state.draw_pile, engine.state.discard_pile, engine.state.exhaust_pile):
            upgraded_cards = [card for card in pile if card.get("card_id") != "Apotheosis"]
            self.assertTrue(all(int(card.get("upgrades") or 0) >= 1 for card in upgraded_cards))

    def test_finesse_gains_block_and_draws(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Finesse", uuid="finesse")]
        engine.state.draw_pile = [make_card("Strike_R", uuid="drawn")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Finesse")
        engine.step(action)
        self.assertGreaterEqual(engine.player.block, 2)
        self.assertTrue(any(card["card_id"] == "Strike_R" for card in engine.state.hand))

    def test_transmutation_uses_x_cost_to_generate_free_colorless_cards(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Transmutation", uuid="transmutation")]
        engine.player.energy = 2
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Transmutation")
        engine.step(action)
        generated = [card for card in engine.state.hand if card.get("color") == "COLORLESS"]
        self.assertGreaterEqual(len(generated), 2)
        self.assertTrue(all(int(card.get("cost_for_turn") or 0) == 0 for card in generated))

    def test_transmutation_hand_overflow_splits_generated_cards_between_hand_and_discard(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80, energy=2, base_energy=2),
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": [],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": ["Panacea"],
                "SRC_CURSE": [],
            },
        )
        engine.state.hand = [
            make_card("Transmutation", uuid="transmutation"),
            *[make_card("Defend_R", uuid=f"transmutation-filler-{index}") for index in range(9)],
        ]
        engine.state.discard_pile = []
        engine.player.energy = 2

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Transmutation"))

        self.assertEqual(len(engine.state.hand), 10)
        self.assertEqual(sum(1 for card in engine.state.hand if card.get("card_id") == "Panacea"), 1)
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Panacea"])
        self.assertEqual(engine.state.discard_pile[0].get("cost_for_turn"), 0)

    def test_transmutation_high_energy_hand_overflow_sends_all_excess_generated_cards_to_discard(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80, energy=4, base_energy=4),
            master_deck=[],
            source_card_pools={
                "SRC_COMMON": [],
                "SRC_UNCOMMON": [],
                "SRC_RARE": [],
                "SRC_COLORLESS": ["Panacea"],
                "SRC_CURSE": [],
            },
        )
        engine.state.hand = [
            make_card("Transmutation", uuid="transmutation"),
            *[make_card("Defend_R", uuid=f"transmutation-filler-{index}") for index in range(10)],
        ]
        engine.state.discard_pile = []
        engine.player.energy = 4

        engine.step(next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Transmutation"))

        self.assertEqual(len(engine.state.hand), 10)
        self.assertFalse(any(card.get("card_id") == "Panacea" for card in engine.state.hand))
        self.assertEqual([card["card_id"] for card in engine.state.discard_pile], ["Panacea", "Panacea", "Panacea", "Panacea"])
        self.assertTrue(all(card.get("cost_for_turn") == 0 for card in engine.state.discard_pile))

    def test_metamorphosis_generates_zero_cost_attacks_in_draw_pile(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.draw_pile = [make_card("Dazed", uuid=f"draw-{index}") for index in range(5)]
        original_draw_uuids = [card["uuid"] for card in engine.state.draw_pile]
        shuffle_counter_before = engine.randoms.stream("shuffle").counter
        engine.state.hand = [make_card("Metamorphosis", uuid="metamorphosis")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Metamorphosis")
        engine.step(action)
        generated = [card for card in engine.state.draw_pile if not str(card.get("uuid") or "").startswith("draw-")]
        self.assertGreaterEqual(len(generated), 3)
        self.assertTrue(all(card.get("type") == "ATTACK" for card in generated))
        self.assertTrue(all(int(card.get("cost_for_turn") or 0) == 0 for card in generated if int(card.get("cost") or 0) == 0))
        self.assertEqual([card["uuid"] for card in engine.state.draw_pile if str(card.get("uuid") or "").startswith("draw-")], original_draw_uuids)
        self.assertEqual(engine.randoms.stream("shuffle").counter, shuffle_counter_before)

    def test_warcry_draws_then_places_a_card_on_top_of_draw_pile(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Warcry", uuid="warcry")]
        engine.state.draw_pile = [make_card("Strike_R", uuid="draw-one")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Warcry")
        engine.step(action)
        self.assertEqual(engine.state.draw_pile[-1]["card_id"], "Strike_R")
        self.assertIsNone(engine.pending_card_select)

    def test_warcry_opens_card_select_when_multiple_hand_cards_after_draw(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Warcry", uuid="warcry"), make_card("Defend_R", uuid="defend")]
        engine.state.draw_pile = [make_card("Strike_R", uuid="draw-one")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Warcry")
        engine.step(action)
        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.pending_card_select["mode"], "WARCRY")
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Defend_R", "Strike_R"])
        self.assertEqual(engine.state.exhaust_pile, [])

        select_action = next(candidate for candidate in engine.legal_actions() if candidate.get("choice_index") == 1)
        engine.step(select_action)
        self.assertIsNone(engine.pending_card_select)
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Defend_R"])
        self.assertEqual(engine.state.draw_pile[-1]["card_id"], "Strike_R")
        self.assertEqual(engine.state.exhaust_pile[-1]["card_id"], "Warcry")

    def test_warcry_defers_fire_breathing_damage_until_put_on_deck_resolves(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        monster = engine.state.monsters[0]
        monster.current_hp = 54
        monster.max_hp = 54
        engine.player.powers.append({"power_id": "Fire Breathing", "id": "Fire Breathing", "amount": 10, "misc": 10})
        engine.state.hand = [make_card("Warcry", uuid="warcry"), make_card("Defend_R", uuid="defend")]
        engine.state.draw_pile = [make_card("Burn", uuid="burn")]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Warcry")
        engine.step(action)

        self.assertIsNotNone(engine.pending_card_select)
        self.assertEqual(engine.pending_card_select["mode"], "WARCRY")
        self.assertEqual(monster.current_hp, 54)
        self.assertEqual([card["card_id"] for card in engine.state.hand], ["Defend_R", "Burn"])

        select_action = next(candidate for candidate in engine.legal_actions() if candidate.get("choice_index") == 1)
        engine.step(select_action)

        self.assertIsNone(engine.pending_card_select)
        self.assertEqual(monster.current_hp, 44)
        self.assertEqual(engine.state.draw_pile[-1]["card_id"], "Burn")

    def test_trip_applies_vulnerable_and_trip_plus_hits_all_enemies(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Trip", uuid="trip")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Trip")
        target = engine.state.monsters[int(action["target_index"])]
        engine.step(action)
        self.assertGreater(_power_amount(target, "Vulnerable"), 0)

        upgraded = make_card("Trip", uuid="trip-plus")
        upgraded["upgrades"] = 1
        engine.state.hand = [upgraded]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Trip")
        engine.step(action)
        self.assertTrue(all(_power_amount(monster, "Vulnerable") > 0 for monster in engine.state.monsters))

    def test_blind_and_panacea_apply_expected_powers(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Blind", uuid="blind"), make_card("Panacea", uuid="panacea")]
        engine.player.energy = 3
        blind_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Blind")
        target = engine.state.monsters[int(blind_action["target_index"])]
        engine.step(blind_action)
        self.assertGreater(_power_amount(target, "Weak"), 0)
        panacea_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Panacea")
        engine.step(panacea_action)
        self.assertGreater(_power_amount(engine.player, "Artifact"), 0)

    def test_rupture_and_blood_for_blood_track_player_hp_loss(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        bite = make_card("Blood for Blood", uuid="bfb")
        bite["cost_for_turn"] = int(bite.get("base_cost") or bite.get("cost") or 4)
        engine.state.hand = [make_card("Rupture", uuid="rupture"), bite, make_card("Offering", uuid="offering")]
        engine.player.energy = 3

        rupture_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Rupture")
        engine.step(rupture_action)
        offering_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Offering")
        engine.step(offering_action)

        self.assertEqual(engine.player_damage_taken_this_combat, 6)
        self.assertEqual(_power_amount(engine.player, "Strength"), 1)
        blood_for_blood = next(card for card in engine.state.hand if card.get("card_id") == "Blood for Blood")
        self.assertEqual(int(blood_for_blood.get("cost") or 0), 3)
        self.assertEqual(int(blood_for_blood.get("cost_for_turn") or 0), 3)
        self.assertTrue(blood_for_blood.get("is_playable"))

    def test_normality_blocks_fourth_card_play_when_in_hand(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Strike_R", uuid="s1"),
            make_card("Strike_R", uuid="s2"),
            make_card("Strike_R", uuid="s3"),
            make_card("Strike_R", uuid="s4"),
            make_card("Normality", uuid="norm"),
        ]
        engine.player.energy = 10
        for _ in range(3):
            action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Strike_R")
            engine.step(action)
        self.assertEqual(engine.cards_played_this_turn, 3)
        legal = engine.legal_actions()
        self.assertFalse(any(candidate.get("kind") == "card" for candidate in legal))
        self.assertTrue(any(candidate.get("kind") == "end" for candidate in legal))

    def test_pain_triggers_when_other_card_is_played(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Pain", uuid="pain"), make_card("Strike_R", uuid="strike")]
        engine.player.energy = 3
        before_hp = int(engine.player.current_hp)
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Strike_R")
        engine.step(action)
        self.assertEqual(before_hp - int(engine.player.current_hp), 1)

    def test_end_of_turn_curse_and_status_triggers_apply(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        burn = make_card("Burn", uuid="burn")
        burn["base_magic"] = 2
        burn["magic"] = 2
        engine.state.hand = [burn, make_card("Doubt", uuid="doubt"), make_card("Shame", uuid="shame"), make_card("Regret", uuid="regret")]
        before_hp = int(engine.player.current_hp)
        engine._apply_player_end_of_turn_powers()
        self.assertEqual(before_hp - int(engine.player.current_hp), 6)
        self.assertEqual(_power_amount(engine.player, "Weak"), 1)
        self.assertEqual(_power_amount(engine.player, "Frail"), 1)

    def test_pride_adds_copy_to_draw_pile_at_end_of_turn(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            relics=[],
        )
        engine.state.hand = [make_card("Pride", uuid="pride")]
        engine._apply_player_end_of_turn_powers()
        self.assertTrue(any(card.get("card_id") == "Pride" for card in engine.state.draw_pile))

    def test_necronomicurse_returns_to_hand_when_exhausted_with_necronomicon(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
            relics=[make_relic("Necronomicon")],
        )
        curse = make_card("Necronomicurse", uuid="necro")
        engine._exhaust_card(curse)
        self.assertTrue(any(card.get("card_id") == "Necronomicurse" for card in engine.state.hand))

    def test_berserk_and_sentinel_apply_real_game_energy_hooks(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Berserk", uuid="berserk")]
        engine.player.energy = 3
        berserk_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Berserk")
        engine.step(berserk_action)
        self.assertGreater(_power_amount(engine.player, "Berserk"), 0)
        self.assertGreater(_power_amount(engine.player, "Vulnerable"), 0)

        sentinel = make_card("Sentinel", uuid="sentinel")
        engine.state.hand = [sentinel]
        engine.player.energy = 0
        engine._exhaust_card(engine.state.hand.pop())
        self.assertEqual(engine.player.energy, 2)

        engine.state.hand = []
        engine.state.draw_pile = []
        engine.state.discard_pile = []
        engine._start_player_turn()
        self.assertEqual(engine.player.energy, 4)

    def test_bite_sever_soul_shockwave_and_shell_parasite_alias(self):
        engine = CombatEngine(
            encounter_name="Shell Parasite",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=40, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(engine.state.monsters[0].monster_id, "ShelledParasite")
        engine.state.monsters[0].block = 0

        bite = make_card("Bite", uuid="bite")
        shockwave = make_card("Shockwave", uuid="shockwave")
        sever_soul = make_card("Sever Soul", uuid="sever")
        engine.state.hand = [bite, shockwave, sever_soul, make_card("Defend_R", uuid="defend")]
        engine.player.energy = 10

        bite_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Bite")
        target = engine.state.monsters[int(bite_action["target_index"])]
        hp_before = int(engine.player.current_hp)
        engine.step(bite_action)
        self.assertGreater(int(engine.player.current_hp), hp_before)
        self.assertLess(int(target.current_hp), int(target.max_hp))

        shockwave_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Shockwave")
        engine.step(shockwave_action)
        self.assertTrue(all(_power_amount(monster, "Weak") > 0 and _power_amount(monster, "Vulnerable") > 0 for monster in engine.state.monsters if monster.current_hp > 0))

        sever_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Sever Soul")
        engine.step(sever_action)
        self.assertFalse(any(card.get("card_id") == "Defend_R" for card in engine.state.hand))
        self.assertTrue(any(card.get("card_id") == "Defend_R" for card in engine.state.exhaust_pile))

    def test_sentinel_forethought_and_panic_button_behave_as_expected(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Sentinel", uuid="sentinel"), make_card("Forethought", uuid="forethought"), make_card("Strike_R", uuid="strike")]
        engine.player.energy = 3

        sentinel_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Sentinel")
        engine.step(sentinel_action)
        self.assertGreaterEqual(engine.player.block, 5)

        forethought_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Forethought")
        engine.step(forethought_action)
        self.assertEqual(engine.state.draw_pile[-1]["card_id"], "Strike_R")
        self.assertEqual(int(engine.state.draw_pile[-1].get("cost_for_turn") or 0), 0)

        engine.state.hand = [make_card("PanicButton", uuid="panic")]
        engine.player.energy = 3
        panic_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "PanicButton")
        engine.step(panic_action)
        block_after_panic = int(engine.player.block)
        self.assertGreater(block_after_panic, 0)
        self.assertGreater(_power_amount(engine.player, "NoBlockPower"), 0)
        engine._gain_player_card_block(99)
        self.assertEqual(int(engine.player.block), block_after_panic)

    def test_mayhem_fire_breathing_and_ritual_dagger_update_state(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        mayhem = make_card("Mayhem", uuid="mayhem")
        fire_breathing = make_card("Fire Breathing", uuid="fire-breathing")
        ritual = make_card("RitualDagger", uuid="ritual")
        ritual["misc"] = 15
        ritual["base_damage"] = 15
        engine.master_deck = [ritual]
        engine.state.hand = [mayhem, fire_breathing, ritual]
        engine.state.draw_pile = [make_card("Strike_R", uuid="draw-attack")]
        engine.player.energy = 10

        mayhem_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Mayhem")
        engine.step(mayhem_action)
        fire_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Fire Breathing")
        engine.step(fire_action)
        self.assertGreater(_power_amount(engine.player, "Mayhem"), 0)
        self.assertGreater(_power_amount(engine.player, "Fire Breathing"), 0)

        engine.state.draw_pile = [make_card("Dazed", uuid="dazed")]
        hp_before = int(engine.state.monsters[0].current_hp)
        engine._start_player_turn()
        self.assertEqual(int(engine.state.monsters[0].current_hp), hp_before)

        target = engine.state.monsters[0]
        target.current_hp = 10
        engine.state.hand = [ritual]
        engine.player.energy = 3
        ritual_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "RitualDagger")
        engine.step(ritual_action)
        self.assertGreaterEqual(int(ritual.get("misc") or 0), 18)
        self.assertGreaterEqual(int(engine.master_deck[0].get("misc") or 0), 18)

    def test_mayhem_plays_top_card_before_normal_turn_draw_in_v3(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=91),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = []
        engine.state.draw_pile = [
            make_card("Strike_R", uuid="draw-bottom"),
            make_card("Flex", uuid="draw-flex"),
            make_card("Bash", uuid="draw-bash"),
            make_card("Defend_R", uuid="draw-defend-a"),
            make_card("Defend_R", uuid="draw-defend-b"),
            make_card("Disarm", uuid="draw-disarm"),
            make_card("Strike_R", uuid="mayhem-top"),
        ]
        combat_engine_mod._append_power(engine.player, "Mayhem", 1, misc=1)

        engine._start_player_turn()

        self.assertEqual([card["uuid"] for card in engine.state.hand], [
            "draw-disarm",
            "draw-defend-b",
            "draw-defend-a",
            "draw-bash",
            "draw-flex",
        ])
        self.assertNotIn("mayhem-top", [card["uuid"] for card in engine.state.hand])
        self.assertIn("mayhem-top", [card["uuid"] for card in engine.state.discard_pile])

    def test_ritual_dagger_does_not_grow_on_minion_kill(self):
        engine = CombatEngine(
            encounter_name="Gremlin Leader",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        ritual = make_card("RitualDagger", uuid="ritual")
        ritual["misc"] = 15
        ritual["base_damage"] = 15
        engine.master_deck = [ritual]
        engine.state.hand = [ritual]
        engine.player.energy = 3
        target = next(monster for monster in engine.state.monsters if _power_amount(monster, "Minion") != 0)
        target.current_hp = 1

        ritual_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "RitualDagger")
        engine.step(ritual_action)

        self.assertEqual(int(ritual.get("misc") or 0), 15)
        self.assertEqual(int(ritual.get("base_damage") or 0), 15)
        self.assertEqual(int(engine.master_deck[0].get("misc") or 0), 15)

    def test_void_loses_energy_when_drawn(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.player.energy = 3
        engine.state.draw_pile = [make_card("Void", uuid="void-draw")]

        engine.draw_cards(1)

        self.assertEqual(engine.player.energy, 2)
        self.assertEqual(engine.state.hand[-1]["card_id"], "Void")

    def test_dazed_is_safe_noop_when_forced_to_play(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        dazed = make_card("Dazed", uuid="forced-dazed")
        hp_before = int(engine.player.current_hp)
        monster_hp_before = int(engine.state.monsters[0].current_hp)

        engine._apply_card_effect(dazed, None)

        self.assertEqual(engine.player.current_hp, hp_before)
        self.assertEqual(engine.state.monsters[0].current_hp, monster_hp_before)

    def test_mind_blast_and_jax_follow_real_game_scaling(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=40, max_hp=80),
            master_deck=[],
        )
        engine.state.draw_pile = [make_card("Strike_R", uuid=f"draw-{i}") for i in range(4)]
        engine.state.hand = [make_card("Mind Blast", uuid="mindblast"), make_card("J.A.X.", uuid="jax")]
        engine.player.energy = 10

        mind_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Mind Blast")
        target = engine.state.monsters[int(mind_action["target_index"])]
        hp_before = int(target.current_hp)
        engine.step(mind_action)
        self.assertEqual(hp_before - int(target.current_hp), 4)

        jax_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "J.A.X.")
        hp_before = int(engine.player.current_hp)
        engine.step(jax_action)
        self.assertEqual(hp_before - int(engine.player.current_hp), 3)
        self.assertGreaterEqual(_power_amount(engine.player, "Strength"), 2)

    def test_discovery_generates_free_card_and_writhe_generated_play_is_safe(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Discovery", uuid="discovery")]
        engine.player.energy = 3
        discovery_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Discovery")
        engine.step(discovery_action)
        self.assertEqual((engine.pending_card_select or {}).get("mode"), "DISCOVERY")
        discovery_choices = engine.legal_actions()
        self.assertEqual(len(discovery_choices), 3)
        self.assertTrue(all(action.get("kind") == "card_reward" for action in discovery_choices))
        engine.step(discovery_choices[0])
        self.assertIsNone(engine.pending_card_select)
        discovered = next(card for card in engine.state.hand if card.get("card_id") != "Discovery")
        self.assertEqual(discovered.get("cost_for_turn"), 0)
        self.assertFalse(discovered.get("free_to_play_once"))
        self.assertTrue(any(card.get("card_id") == "Discovery" for card in engine.state.exhaust_pile))

        curse = make_card("Writhe", uuid="writhe")
        engine._resolve_generated_card_play(curse)
        self.assertTrue(any(card.get("card_id") == "Writhe" for card in engine.state.discard_pile + engine.state.exhaust_pile))

        engine.state.hand = [make_card("PanicButton", uuid="panic")]
        engine.player.energy = 3
        panic_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "PanicButton")
        engine.step(panic_action)
        self.assertGreater(_power_amount(engine.player, "NoBlockPower"), 0)
        _decrement_before = int(_power_amount(engine.player, "NoBlockPower"))
        engine._end_turn()
        self.assertLess(_power_amount(engine.player, "NoBlockPower"), _decrement_before)

    def test_discovery_in_run_env_uses_card_reward_screen_before_returning_to_combat(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False)
        env.phase = "COMBAT"
        env.current_room_type = "MonsterRoom"
        env.combat = NativeCombatEnv(
            seed=2,
            ascension_level=0,
            player=env.player,
            master_deck=env.deck,
            relics=env.relics,
            potions=env.potions,
            gold=env.gold,
            encounter_name="Cultist",
            randoms=env.randoms,
        )
        env.combat.engine.state.hand = [make_card("Discovery", uuid="discovery")]
        env.combat.engine.player.energy = 3

        env.step(next(action for action in env.legal_actions() if action.get("card_id") == "Discovery"))

        self.assertEqual(env.phase, "CARD_REWARD")
        state = env.state()
        self.assertEqual(state["phase"], "CARD_REWARD")
        self.assertEqual(state["screen_type"], "CARD_REWARD")
        self.assertFalse(state["screen_state"]["skip_available"])
        choices = env.legal_actions()
        self.assertEqual(len(choices), 3)
        self.assertTrue(all(action.get("kind") == "card_reward" for action in choices))

        env.step(choices[0])

        self.assertEqual(env.phase, "COMBAT")
        discovered = next(card for card in env.combat.engine.state.hand if card.get("card_id") != "Discovery")
        self.assertEqual(discovered.get("cost_for_turn"), 0)
        self.assertFalse(discovered.get("free_to_play_once"))

    def test_enlightenment_reduces_hand_costs_and_plus_persists_for_combat(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        bash = make_card("Bash", uuid="bash")
        bludgeon = make_card("Bludgeon", uuid="bludgeon")
        engine.state.hand = [make_card("Enlightenment", uuid="enlightenment"), bash, bludgeon]
        engine.player.energy = 3

        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Enlightenment")
        engine.step(action)
        self.assertEqual(int(bash.get("cost_for_turn") or 0), 1)
        self.assertEqual(int(bludgeon.get("cost_for_turn") or 0), 1)
        self.assertNotEqual(int(bash.get("cost") or bash.get("base_cost") or 0), 1)

        enlighten_plus = make_card("Enlightenment", uuid="enlightenment-plus")
        enlighten_plus["upgrades"] = 1
        bash2 = make_card("Bash", uuid="bash2")
        engine.state.hand = [enlighten_plus, bash2]
        engine.player.energy = 3
        plus_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Enlightenment")
        engine.step(plus_action)
        self.assertEqual(int(bash2.get("cost_for_turn") or 0), 1)
        self.assertEqual(int(bash2.get("cost") or 0), 1)
        self.assertEqual(int(bash2.get("cost_for_combat") or 0), 1)

    def test_the_bomb_counts_down_at_end_of_turn_and_hits_all_enemies(self):
        engine = CombatEngine(
            encounter_name="3 Byrds",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("The Bomb", uuid="the-bomb")]
        engine.player.energy = 3
        action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "The Bomb")
        engine.step(action)
        self.assertEqual(_power_amount(engine.player, "TheBomb"), 3)
        hp_before = [int(monster.current_hp) for monster in engine.state.monsters]
        engine._apply_player_end_of_turn_powers()
        self.assertEqual(_power_amount(engine.player, "TheBomb"), 2)
        self.assertEqual([int(monster.current_hp) for monster in engine.state.monsters], hp_before)
        engine._apply_player_end_of_turn_powers()
        self.assertEqual(_power_amount(engine.player, "TheBomb"), 1)
        engine._apply_player_end_of_turn_powers()
        self.assertEqual(_power_amount(engine.player, "TheBomb"), 0)
        self.assertTrue(all(int(monster.current_hp) < before for monster, before in zip(engine.state.monsters, hp_before)))

    def test_clash_requires_attack_only_hand(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Clash", uuid="clash"), make_card("Defend_R", uuid="defend")]
        engine.player.energy = 3
        self.assertFalse(any(candidate.get("card_id") == "Clash" for candidate in engine.legal_actions()))
        engine.state.hand = [make_card("Clash", uuid="clash2"), make_card("Strike_R", uuid="strike")]
        self.assertTrue(any(candidate.get("card_id") == "Clash" for candidate in engine.legal_actions()))

    def test_entangle_blocks_attack_cards_from_legal_actions(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80, powers=[{"power_id": "Entangled", "amount": 1}]),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Bash", uuid="bash"),
            make_card("Strike_R", uuid="strike"),
            make_card("Defend_R", uuid="defend"),
        ]
        engine.player.energy = 3

        actions = engine.legal_actions()

        self.assertFalse(any(candidate.get("card_id") == "Bash" for candidate in actions))
        self.assertFalse(any(candidate.get("card_id") == "Strike_R" for candidate in actions))
        self.assertTrue(any(candidate.get("card_id") == "Defend_R" for candidate in actions))

    def test_make_card_applies_upgrade_damage_and_magic_from_source(self):
        bash = make_card("Bash", upgrades=1, uuid="bash-upgraded")

        self.assertEqual(bash["base_damage"], 10)
        self.assertEqual(bash["base_magic"], 3)

    def test_violence_secret_technique_and_secret_weapon_pull_from_draw_pile(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [
            make_card("Violence", uuid="violence"),
            make_card("Secret Technique", uuid="secret-technique"),
            make_card("Secret Weapon", uuid="secret-weapon"),
        ]
        engine.state.draw_pile = [
            make_card("Strike_R", uuid="attack1"),
            make_card("Defend_R", uuid="skill1"),
            make_card("Bash", uuid="attack2"),
            make_card("Spot Weakness", uuid="skill2"),
        ]
        engine.player.energy = 3
        violence_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Violence")
        engine.step(violence_action)
        self.assertTrue(any(card.get("type") == "ATTACK" for card in engine.state.hand))
        secret_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Secret Technique")
        engine.step(secret_action)
        self.assertTrue(any(action["kind"] == "card_select" for action in engine.legal_actions()))
        engine.step(next(action for action in engine.legal_actions() if action["kind"] == "card_select"))
        self.assertTrue(any(card.get("type") == "SKILL" for card in engine.state.hand if card.get("card_id") != "Secret Technique"))
        engine.state.draw_pile = [make_card("Cleave", uuid="attack3"), make_card("Defend_R", uuid="skill3")]
        weapon_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Secret Weapon")
        engine.step(weapon_action)
        self.assertTrue(any(card.get("type") == "ATTACK" for card in engine.state.hand if card.get("card_id") not in {"Violence", "Secret Weapon"}))

    def test_violence_uses_sts_random_draw_pile_to_hand_selection(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=5152791245673712276),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        engine.state.hand = [make_card("Violence", uuid="violence")]
        engine.state.draw_pile = [
            make_card("Defend_R", uuid="d1"),
            make_card("Strike_R", uuid="s1"),
            make_card("Uppercut", upgrades=1, uuid="u1"),
            make_card("Defend_R", uuid="d2"),
            make_card("Bash", uuid="b1"),
            make_card("Defend_R", uuid="d3"),
            make_card("Strike_R", uuid="s2"),
            make_card("Strike_R", uuid="s3"),
        ]
        engine.player.energy = 3

        violence_action = next(candidate for candidate in engine.legal_actions() if candidate.get("card_id") == "Violence")
        engine.step(violence_action)

        self.assertEqual([card.get("uuid") for card in engine.state.hand], ["s2", "u1", "b1"])
        self.assertEqual([card.get("uuid") for card in engine.state.draw_pile], ["d1", "s1", "d2", "d3", "s3"])

    def test_gremlin_leader_encounter_generates_and_leader_death_escapes_minions(self):
        engine = CombatEngine(
            encounter_name="Gremlin Leader",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(engine.state.monsters[0].monster_id, "GremlinLeader")
        self.assertGreaterEqual(len(engine.state.monsters), 3)
        leader = engine.state.monsters[0]
        engine._kill_monster(leader)
        self.assertTrue(all(monster.current_hp == 0 for monster in engine.state.monsters))

    def test_gremlin_leader_rally_chooses_both_minions_before_opening_ai_rolls(self):
        engine = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=1),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        randoms = NativeRandomSet(seed=3)
        randoms.reset_floor_streams(24)
        for _ in range(4):
            randoms.stream("ai").random(99)
        engine.randoms = randoms
        leader = MonsterState(
            monster_id="GremlinLeader",
            name="Gremlin Leader",
            current_hp=108,
            max_hp=146,
            intent="UNKNOWN",
            next_move="RALLY",
            meta={"strength_amount": 3, "block_amount": 6, "stab_damage": 6, "stab_hits": 3},
        )
        dead_a = MonsterState(
            monster_id="GremlinWarrior",
            name="Mad Gremlin",
            current_hp=0,
            max_hp=21,
            intent="NONE",
            next_move="SCRATCH",
            meta={"gremlin_leader_slot": 0},
        )
        dead_b = MonsterState(
            monster_id="GremlinWarrior",
            name="Mad Gremlin",
            current_hp=0,
            max_hp=22,
            intent="NONE",
            next_move="SCRATCH",
            meta={"gremlin_leader_slot": 1},
        )
        engine.state.monsters = [dead_a, dead_b, leader]

        before = len(randoms.stream("ai").calls)
        engine._gremlin_leader_take_turn(leader)

        alive_minions_by_slot = {
            int(monster.meta["gremlin_leader_slot"]): monster.monster_id
            for monster in engine.state.monsters
            if monster is not leader and monster.current_hp > 0
        }
        self.assertEqual(alive_minions_by_slot, {0: "GremlinFat", 1: "GremlinTsundere"})
        new_ai_args = [tuple(call.args) for call in randoms.stream("ai").calls[before:]]
        self.assertEqual(new_ai_args[:5], [(0, 7), (0, 7), (99,), (99,), (99,)])

    def test_city_boss_encounters_generate_vertical_slices(self):
        champ = CombatEngine(
            encounter_name="Champ",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(champ.state.monsters[0].monster_id, "Champ")
        self.assertIn(champ.state.monsters[0].next_move, {"HEAVY_SLASH", "DEFENSIVE_STANCE", "FACE_SLAP", "GLOAT", "TAUNT", "ANGER", "EXECUTE"})

        champ_randoms = NativeRandomSet(seed=50)
        champ_randoms.reset_floor_streams(33)
        seeded_champ = combat_engine_mod._spawn_champ(champ_randoms, ascension_level=0)
        self.assertEqual(seeded_champ.next_move, "HEAVY_SLASH")
        self.assertEqual(seeded_champ.meta["num_turns"], 1)
        self.assertEqual(champ_randoms.debug_trace()["ai"][0]["result"], 87)
        champ_randoms.stream("ai").random(99)
        champ_randoms.stream("ai").random(99)
        seeded_champ.next_move = "TAUNT"
        seeded_champ.move_history = ["HEAVY_SLASH", "FACE_SLAP", "DEFENSIVE_STANCE"]
        seeded_champ.meta["num_turns"] = 0
        champ_for_dialog = CombatEngine(
            encounter_name="Cultist",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        champ_for_dialog.randoms = champ_randoms
        champ_for_dialog.state.monsters = [seeded_champ]
        champ_for_dialog._champ_take_turn(seeded_champ)
        ai_calls = champ_randoms.debug_trace()["ai"]
        self.assertEqual([call["args"] for call in ai_calls[:5]], [[99], [99], [99], [3], [99]])
        self.assertEqual(ai_calls[4]["result"], 53)
        self.assertEqual(seeded_champ.next_move, "FACE_SLAP")

        champ_damage = CombatEngine(
            encounter_name="Champ",
            randoms=NativeRandomSet(seed=3),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        champ_monster = champ_damage.state.monsters[0]
        champ_monster.next_move = "FACE_SLAP"
        combat_engine_mod._add_power(champ_monster, "Strength", -3)
        champ_damage._champ_take_turn(champ_monster)
        self.assertEqual(champ_damage.player.current_hp, 71)

        champ_anger = CombatEngine(
            encounter_name="Champ",
            randoms=NativeRandomSet(seed=4),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        angry_champ = champ_anger.state.monsters[0]
        angry_champ.next_move = "ANGER"
        combat_engine_mod._add_power(angry_champ, "Strength", -3)
        combat_engine_mod._add_power(angry_champ, "Vulnerable", 3)
        combat_engine_mod._add_power(angry_champ, "Weak", 3)
        combat_engine_mod._add_power(angry_champ, "Shackled", 5)
        combat_engine_mod._add_power(angry_champ, "Metallicize", 5)
        champ_anger._champ_take_turn(angry_champ)
        champ_power_amounts = {
            power["power_id"]: int(power.get("amount") or 0)
            for power in angry_champ.powers
        }
        self.assertEqual(champ_power_amounts, {"Strength": 6, "Metallicize": 5})

        collector = CombatEngine(
            encounter_name="Collector",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(collector.state.monsters[0].monster_id, "TheCollector")
        collector._collector_take_turn(collector.state.monsters[0])
        self.assertGreaterEqual(sum(1 for monster in collector.state.monsters if monster.monster_id == "TorchHead"), 2)

        collector_death = CombatEngine(
            encounter_name="Collector",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        collector_death._collector_take_turn(collector_death.state.monsters[0])
        boss = next(monster for monster in collector_death.state.monsters if monster.monster_id == "TheCollector")
        collector_death._kill_monster(boss)
        self.assertEqual(collector_death.outcome, "VICTORY")
        self.assertTrue(all(monster.current_hp == 0 for monster in collector_death.state.monsters))

        automaton = CombatEngine(
            encounter_name="Automaton",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual(automaton.state.monsters[0].monster_id, "BronzeAutomaton")
        automaton._automaton_take_turn(automaton.state.monsters[0])
        self.assertEqual([monster.monster_id for monster in automaton.state.monsters], ["BronzeOrb", "BronzeAutomaton", "BronzeOrb"])
        self.assertEqual([monster.name for monster in automaton.state.monsters], ["Orb", "Bronze Automaton", "Orb"])
        self.assertEqual([(monster.current_hp, monster.max_hp) for monster in automaton.state.monsters if monster.monster_id == "BronzeOrb"], [(56, 56), (53, 53)])
        self.assertTrue(all(_power_amount(monster, "Minion") == -1 for monster in automaton.state.monsters if monster.monster_id == "BronzeOrb"))

    def test_colosseum_first_fight_reopens_event_and_second_fight_grants_rewards(self):
        event = EventState("Colosseum")

        intro = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertEqual(event.screen, "FIGHT")
        self.assertFalse(intro["open_rewards"])

        fight = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertTrue(fight["open_combat"])
        self.assertEqual(fight["encounter_name"], "Colosseum Slavers")
        self.assertEqual(fight["event_rewards"]["reopen_event"].screen, "POST_COMBAT")

        second = resolve_event_choice(
            EventState("Colosseum", screen="POST_COMBAT"),
            action_index=1,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertTrue(second["open_combat"])
        self.assertEqual(second["encounter_name"], "Colosseum Nobs")
        self.assertTrue(second["elite_trigger"])
        self.assertEqual(int(second["event_rewards"]["gold"]), 100)
        self.assertEqual(len(second["event_rewards"]["relics"]), 2)

    def test_mysterious_sphere_opens_two_orb_walkers_with_gold_and_relic_rewards(self):
        event = EventState("Mysterious Sphere", screen="PRE_COMBAT")
        screenless_tiers: list[str | None] = []
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
            screenless_relic_drawer=lambda tier=None, exclude=None: (
                screenless_tiers.append(tier) or make_relic("Thread and Needle")
            ),
        )
        self.assertTrue(result["open_combat"])
        self.assertEqual(result["encounter_name"], "2 Orb Walkers")
        self.assertEqual(len(result["event_rewards"]["relics"]), 1)
        self.assertEqual(result["event_rewards"]["relics"][0]["relic_id"], "Thread and Needle")
        self.assertEqual(screenless_tiers, ["RARE"])
        self.assertGreaterEqual(int(result["event_rewards"]["gold"]), 45)
        self.assertLessEqual(int(result["event_rewards"]["gold"]), 55)

    def test_secret_portal_warps_directly_to_boss_combat(self):
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=False, start_on_map=False)
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("SecretPortal", screen="ACCEPT")
        env.act = 1
        env.floor = 10
        env.boss_list = ["Hexaghost"]
        env.act_boss = "Hexaghost"

        env.step({"kind": "event", "event_id": "SecretPortal", "choice_index": 0})

        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual(env.current_room_type, "MonsterRoomBoss")
        self.assertEqual(env.floor, 16)
        self.assertIsNotNone(env.combat)
        self.assertEqual(env.combat.encounter_name, "Hexaghost")

    def test_sensory_stone_opens_multiple_colorless_reward_groups(self):
        event = EventState("SensoryStone", screen="INTRO_2")
        randoms = NativeRandomSet(seed=2)
        result = resolve_event_choice(
            event,
            action_index=2,
            randoms=randoms,
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )
        self.assertTrue(result["open_event_card_reward"])
        self.assertEqual(event.screen, "LEAVE")
        self.assertEqual(result["hp"], 70)
        self.assertEqual(len(result["reward_cards"]), 3)
        self.assertEqual(len(result["reward_card_groups"]), 2)
        self.assertEqual(randoms.stream("misc").counter, 1)

    def test_sensory_stone_memory_choices_consume_one_misc_random_long(self):
        cases = [
            (0, 80, 1),
            (1, 75, 2),
            (2, 70, 3),
        ]
        for action_index, expected_hp, expected_groups in cases:
            with self.subTest(action_index=action_index):
                event = EventState("SensoryStone", screen="INTRO_2")
                randoms = NativeRandomSet(seed=2)
                result = resolve_event_choice(
                    event,
                    action_index=action_index,
                    randoms=randoms,
                    ascension_level=0,
                    current_hp=80,
                    max_hp=80,
                    gold=99,
                    deck=[],
                    relics=[],
                    potions=[],
                )
                reward_group_count = int(bool(result["reward_cards"])) + len(result["reward_card_groups"])

                self.assertEqual(randoms.stream("misc").counter, 1)
                self.assertEqual(result["hp"], expected_hp)
                self.assertEqual(reward_group_count, expected_groups)

    def test_sensory_stone_touch_consumes_no_misc_rng(self):
        event = EventState("SensoryStone", screen="INTRO")
        randoms = NativeRandomSet(seed=2)
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=randoms,
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
        )

        self.assertEqual(event.screen, "INTRO_2")
        self.assertFalse(result["open_rewards"])
        self.assertEqual(randoms.stream("misc").counter, 0)

    def test_sensory_stone_uses_runtime_colorless_card_pool(self):
        event = EventState("SensoryStone", screen="INTRO_2")
        runtime_card_pools = initialize_runtime_card_pools()
        runtime_card_pools["COLORLESS_UNCOMMON"] = ["Panacea"]
        runtime_card_pools["COLORLESS_RARE"] = []
        result = resolve_event_choice(
            event,
            action_index=0,
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            current_hp=80,
            max_hp=80,
            gold=99,
            deck=[],
            relics=[],
            potions=[],
            runtime_card_pools=runtime_card_pools,
        )
        self.assertTrue(result["open_event_card_reward"])
        self.assertTrue(all(card["card_id"] == "Panacea" for card in result["reward_cards"]))

    def test_orb_walker_prebattle_strength_and_encounter_aliases(self):
        engine = CombatEngine(
            encounter_name="2 Orb Walkers",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual([monster.monster_id for monster in engine.state.monsters], ["OrbWalker", "OrbWalker"])
        self.assertTrue(all(_power_amount(monster, "Strength") == 3 for monster in engine.state.monsters))
        self.assertTrue(all(monster.next_move in {"LASER", "CLAW"} for monster in engine.state.monsters))

        colosseum = CombatEngine(
            encounter_name="Colosseum Nobs",
            randoms=NativeRandomSet(seed=2),
            ascension_level=0,
            player=PlayerState(current_hp=80, max_hp=80),
            master_deck=[],
        )
        self.assertEqual([monster.monster_id for monster in colosseum.state.monsters], ["SlaverBoss", "GremlinNob"])

    def test_reference_map_tracks_required_real_game_sources(self):
        self.assertTrue(REQUIRED_SOURCE_CLASSES)
        self.assertTrue(required_source_paths_exist())
        run_sources = CORE_SOURCE_MAP["run"]["source_classes"]
        combat_sources = CORE_SOURCE_MAP["combat"]["source_classes"]
        for required_name in REQUIRED_SOURCE_CLASSES:
            self.assertTrue(
                required_name in run_sources or required_name in combat_sources,
                msg=f"missing real-game source mapping for {required_name}",
            )

    def test_v3_tree_contains_no_v2_or_lightspeed_imports(self):
        root = Path("/home/yydd/spirecomm/spirecomm/native_sim_v3")
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("native_sim_v2", text, msg=f"unexpected v2 coupling in {path}")
            self.assertNotIn("sts_lightspeed", text, msg=f"unexpected lightspeed coupling in {path}")
            self.assertNotIn("native_sim import", text, msg=f"unexpected legacy simulator import in {path}")

    def test_env_threads_player_class_into_shop_generation(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True, character="IRONCLAD")
        env.player_class = "IRONCLAD_SENTINEL"
        env.dungeon_id = "TheCity"

        with patch("spirecomm.native_sim_v3.env.generate_shop", return_value={"cards": [], "relics": [], "potions": [], "purge_cost": 75}) as mock_generate_shop:
            env._step_map({"kind": "map", "node_id": "m-r0-x0", "symbol": "$", "floor": 1})

        self.assertEqual(env.phase, "SHOP")
        self.assertEqual(mock_generate_shop.call_args.kwargs["player_class"], "IRONCLAD_SENTINEL")
        self.assertEqual(mock_generate_shop.call_args.kwargs["dungeon_id"], "TheCity")

    def test_env_threads_dungeon_id_into_treasure_generation(self):
        env = NativeRunEnv(seed=3, ascension_level=0, enable_neow=False, start_on_map=True, character="IRONCLAD")
        env.dungeon_id = "TheBeyond"

        with patch(
            "spirecomm.native_sim_v3.env.generate_treasure",
            return_value={"chest_type": "SmallChest", "gold_amount": 0, "relic_tier": "COMMON", "opened": False},
        ) as mock_generate_treasure:
            env._step_map({"kind": "map", "node_id": "m-r0-x0", "symbol": "T", "floor": 1})

        self.assertEqual(env.phase, "TREASURE")
        self.assertEqual(mock_generate_treasure.call_args.kwargs["dungeon_id"], "TheBeyond")

    def test_env_threads_player_class_into_event_resolution(self):
        env = NativeRunEnv(seed=4, ascension_level=0, enable_neow=False, character="IRONCLAD")
        env.player_class = "IRONCLAD_SENTINEL"
        env.phase = "EVENT"
        env.current_room_type = "EventRoom"
        env.current_event = EventState("GoldenIdol", screen="INTRO")

        with patch(
            "spirecomm.native_sim_v3.env.resolve_event_choice",
            return_value={"gold": env.gold, "hp": env.player.current_hp, "max_hp": env.player.max_hp, "potions": []},
        ) as mock_resolve:
            env._step_event({"kind": "event", "event_id": "GoldenIdol", "choice_index": 0})

        self.assertEqual(mock_resolve.call_args.kwargs["player_class"], "IRONCLAD_SENTINEL")

    def test_advance_to_next_act_threads_player_class_into_boss_relic_pool(self):
        env = NativeRunEnv(seed=4, ascension_level=0, enable_neow=False, start_on_map=True, character="THE_SILENT")
        env.player_class = "THE_SILENT_SENTINEL"
        env.act = 1

        with patch("spirecomm.native_sim_v3.env.initialize_boss_relic_pool", wraps=initialize_boss_relic_pool) as mock_init:
            env._advance_to_next_act()

        self.assertEqual(mock_init.call_args.kwargs["character"], "THE_SILENT_SENTINEL")

    def test_native_combat_env_threads_character_into_default_source_pools(self):
        with patch("spirecomm.native_sim_v3.env.initialize_source_card_pools", wraps=initialize_source_card_pools) as mock_init:
            NativeCombatEnv(seed=4, ascension_level=0, character="THE_SILENT")

        self.assertEqual(mock_init.call_args.kwargs["character"], "THE_SILENT")

    def test_combat_engine_threads_character_into_default_source_pools(self):
        with patch("spirecomm.native_sim_v3.combat.engine.initialize_source_card_pools", wraps=initialize_source_card_pools) as mock_init:
            CombatEngine(
                encounter_name="Cultist",
                randoms=NativeRandomSet(seed=4),
                ascension_level=0,
                character="THE_SILENT",
                player=PlayerState(current_hp=70, max_hp=70),
                master_deck=[],
            )

        self.assertEqual(mock_init.call_args.kwargs["character"], "THE_SILENT")

    def test_run_resolves_opening_relic_combat_victory(self):
        env = NativeRunEnv(seed=545, ascension_level=0, enable_neow=False, start_on_map=True)
        lament = make_relic("NeowsBlessing")
        lament["counter"] = 1
        env.relics = [make_relic("Burning Blood"), lament, make_relic("Mercury Hourglass")]
        env.monster_list = ["Small Slimes"]

        env.step({"kind": "map", "node_id": "a1-r0-x0", "symbol": "M", "floor": 1})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertIsNone(env.combat)
        self.assertTrue(env.legal_actions())

    def test_strange_spoon_can_discard_corruption_exhausted_skill(self):
        env = NativeCombatEnv(
            seed=1,
            ascension_level=0,
            player=PlayerState(current_hp=50, max_hp=80, energy=3),
            relics=[make_relic("Strange Spoon")],
        )
        env.player.powers.append({"id": "Corruption", "power_id": "Corruption", "amount": -1})
        env.state.hand = [make_card("Defend_R")]
        env.state.draw_pile = []
        env.state.discard_pile = []
        env.state.exhaust_pile = []

        env.step({"kind": "card", "card_index": 0})

        self.assertEqual([card["card_id"] for card in env.state.discard_pile], ["Defend_R"])
        self.assertEqual(env.state.exhaust_pile, [])
        self.assertEqual(env.randoms.stream("card_random").counter, 1)

    def test_corruption_exhausts_skill_without_strange_spoon(self):
        env = NativeCombatEnv(
            seed=1,
            ascension_level=0,
            player=PlayerState(current_hp=50, max_hp=80, energy=3),
            relics=[],
        )
        env.player.powers.append({"id": "Corruption", "power_id": "Corruption", "amount": -1})
        env.state.hand = [make_card("Defend_R")]
        env.state.draw_pile = []
        env.state.discard_pile = []
        env.state.exhaust_pile = []

        env.step({"kind": "card", "card_index": 0})

        self.assertEqual(env.state.discard_pile, [])
        self.assertEqual([card["card_id"] for card in env.state.exhaust_pile], ["Defend_R"])
        self.assertEqual(env.randoms.stream("card_random").counter, 0)


if __name__ == "__main__":
    unittest.main()
