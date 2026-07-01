from copy import deepcopy
from dataclasses import replace
import json
import unittest
from pathlib import Path

import slaythespire as sts
from compare_native_to_lightspeed_run import (
    _attach_failure_metadata,
    _choice_list_signature,
    _pick_action_by_signature,
    _summarize_clusters,
    compare_seed,
)
from spirecomm.native_sim.cards import CARD_LIBRARY, ironclad_type_rarity_card_pool, make_card
from spirecomm.native_sim.potions import make_potion
from spirecomm.native_sim.randoms import StsRandom
from spirecomm.native_sim.relics import make_relic
from spirecomm.native_sim.relics import relic_can_spawn
from spirecomm.native_sim.schema import MonsterState, PlayerState
import spirecomm.native_sim_v2.helpers_run as helpers_run_module
from spirecomm.native_sim_v2 import NativeRunEnv
from spirecomm.native_sim_v2.env import NativeCombatEnv
from spirecomm.native_sim_v2.helpers_cards import COLORLESS_CARD_IDS
from spirecomm.native_sim_v2.randoms import NativeRandomStreams, StsRandom as V2StsRandom
from spirecomm.native_sim_v2.monsters import choose_next_move, encounter_to_monsters, encounter_to_monster_ids, make_monster
from spirecomm.native_sim_v2.run_core import _event_percent_hp_loss
from spirecomm.native_sim_v2.helpers_common import COMBAT_CARD_POOL_IRONCLAD, _combat_strike_count


class NativeSimV2SmokeTest(unittest.TestCase):
    def test_v2_source_tree_has_no_v1_runtime_bridge_symbols(self):
        root = Path("/home/yydd/spirecomm/spirecomm/native_sim_v2")
        forbidden = (
            "NativeCombatEnvV1",
            "NativeRunEnvV1",
            "_bind_v1_members",
            "_play_card_v1",
            "_end_turn_v1",
            "_replay_attack_card_effect_v1",
            "native_env_module",
            "native_monsters_module",
            "from spirecomm.native_sim.env import",
            "from spirecomm.native_sim.monsters import",
        )
        for path in root.glob("*.py"):
            text = path.read_text()
            for needle in forbidden:
                self.assertNotIn(needle, text, f"{needle} still present in {path}")

    def test_v2_state_is_spirecomm_compatible(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        state = env.state()
        self.assertEqual(state["character"], "IRONCLAD")
        self.assertIn("combat_state", state)
        self.assertIn("monsters", state["combat_state"])

    def test_v2_run_env_uses_v2_random_streams(self):
        env = NativeRunEnv(seed=2, ascension_level=0)
        self.assertIsInstance(env.randoms, NativeRandomStreams)
        self.assertEqual(env.rng.__class__.__module__, "spirecomm.native_sim_v2.randoms")

    def test_v2_env_instances_do_not_expose_v1_bridge_methods(self):
        run_env = NativeRunEnv(seed=4, ascension_level=0, enable_neow=True)
        combat_env = NativeCombatEnv(seed=4, ascension_level=0)
        for attr in ("_bind_v1_members", "_play_card_v1", "_end_turn_v1", "_replay_attack_card_effect_v1"):
            self.assertFalse(hasattr(run_env, attr))
            self.assertFalse(hasattr(combat_env, attr))

    def test_centennial_puzzle_refreshes_each_combat_in_v2(self):
        combat = NativeCombatEnv(
            seed=4,
            ascension_level=0,
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Centennial Puzzle", "id": "Centennial Puzzle", "name": "Centennial Puzzle", "counter": 1, "price": 0, "tier": "COMMON"},
            ],
        )
        relic = next(relic for relic in combat.relics if relic.get("relic_id") == "Centennial Puzzle")
        self.assertEqual(relic.get("counter"), 0)

    def test_v2_noncombat_state_uses_serializer_bridge(self):
        env = NativeRunEnv(seed=3, ascension_level=0, start_on_map=True)
        state = env.state()
        self.assertFalse(state["in_combat"])
        self.assertEqual(state["screen"], env.phase)
        self.assertEqual(state["choice_list"], env._choice_list())

    def test_alignment_cluster_metadata_marks_power_only_monster_diff_in_v2(self):
        result = {
            "seed": 900001,
            "match": False,
            "reason": "battle_state_mismatch",
            "step": 224,
            "trace_tail": [
                {"step": 223, "phase": "BATTLE", "choice": ("card", "anger", 0, 2), "floor": 18},
            ],
            "lightspeed": {
                "player_hp": 70,
                "player_block": 0,
                "energy": 3,
                "player_powers": (),
                "hand": ["striker"],
                "draw_pile": [],
                "discard_pile": [],
                "exhaust_pile": [],
                "monsters": [
                    ("GremlinWarrior", 20, 0, "MAD_GREMLIN_SCRATCH", "ATTACK", 4, 1, ()),
                ],
            },
            "native": {
                "player_hp": 70,
                "player_block": 0,
                "energy": 3,
                "player_powers": (),
                "hand": ["striker"],
                "draw_pile": [],
                "discard_pile": [],
                "exhaust_pile": [],
                "monsters": [
                    ("GremlinWarrior", 20, 0, "MAD_GREMLIN_SCRATCH", "ATTACK", 4, 1, (("Angry", 1, False),)),
                ],
            },
        }

        _attach_failure_metadata(result, backend="v2", source_random_seed=7)

        self.assertEqual(result["category"], "monster AI / intent")
        self.assertEqual(result["encounter_sig"], ["GremlinWarrior"])
        self.assertEqual(result["cluster_features"]["diff_kind"], "power_only")
        self.assertEqual(result["cluster_features"]["focus_monster_id"], "GremlinWarrior")
        self.assertIn("GremlinWarrior", result["cluster_key"])

    def test_alignment_cluster_metadata_marks_double_tap_draw_drift_as_after_use_in_v2(self):
        result = {
            "seed": 900002,
            "match": False,
            "reason": "battle_state_mismatch",
            "step": 259,
            "trace_tail": [
                {"step": 256, "phase": "CARD_REWARD", "choice": ("skip", "SKIP"), "floor": 18},
                {"step": 257, "phase": "MAP", "choice": ("map", "M", 2), "floor": 18},
                {"step": 258, "phase": "BATTLE", "choice": ("card", "doubletap", 1, 0), "floor": 19},
                {"step": 259, "phase": "BATTLE", "choice": ("card", "striker", 1, 0), "floor": 19},
            ],
            "lightspeed": {
                "player_hp": 66,
                "player_block": 0,
                "energy": 4,
                "player_powers": (),
                "hand": ["burningpact", "cleave", "striker", "anger"],
                "draw_pile": ["defendr", "bash", "thunderclap"],
                "discard_pile": ["anger", "anger"],
                "exhaust_pile": [],
                "monsters": [
                    ("Chosen", 89, 0, "CHOSEN_POKE", "ATTACK", 5, 2, ()),
                ],
            },
            "native": {
                "player_hp": 66,
                "player_block": 0,
                "energy": 4,
                "player_powers": (),
                "hand": ["burningpact", "cleave", "striker", "anger", "defendr"],
                "draw_pile": ["bash", "thunderclap"],
                "discard_pile": ["anger", "anger"],
                "exhaust_pile": [],
                "monsters": [
                    ("Chosen", 89, 0, "CHOSEN_POKE", "ATTACK", 5, 2, ()),
                ],
            },
        }

        _attach_failure_metadata(result, backend="v2", source_random_seed=8)

        self.assertEqual(result["category"], "autoplay/after_use")
        self.assertEqual(result["cluster_features"]["delta_kind"], "hand")
        self.assertIn(["card", "doubletap"], result["trigger_window_sig"])
        self.assertIn("doubletap", result["cluster_key"])

    def test_alignment_cluster_summary_groups_exemplars_by_cluster_key_in_v2(self):
        base_result = {
            "seed": 900100,
            "match": False,
            "reason": "battle_state_mismatch",
            "step": 259,
            "trace_tail": [
                {"step": 257, "phase": "MAP", "choice": ("map", "M", 2), "floor": 18},
                {"step": 258, "phase": "BATTLE", "choice": ("card", "doubletap", 1, 0), "floor": 19},
                {"step": 259, "phase": "BATTLE", "choice": ("card", "striker", 1, 0), "floor": 19},
            ],
            "lightspeed": {
                "player_hp": 66,
                "player_block": 0,
                "energy": 4,
                "player_powers": (),
                "hand": ["burningpact", "cleave", "striker", "anger"],
                "draw_pile": ["defendr", "bash", "thunderclap"],
                "discard_pile": ["anger", "anger"],
                "exhaust_pile": [],
                "monsters": [
                    ("Chosen", 89, 0, "CHOSEN_POKE", "ATTACK", 5, 2, ()),
                ],
            },
            "native": {
                "player_hp": 66,
                "player_block": 0,
                "energy": 4,
                "player_powers": (),
                "hand": ["burningpact", "cleave", "striker", "anger", "defendr"],
                "draw_pile": ["bash", "thunderclap"],
                "discard_pile": ["anger", "anger"],
                "exhaust_pile": [],
                "monsters": [
                    ("Chosen", 89, 0, "CHOSEN_POKE", "ATTACK", 5, 2, ()),
                ],
            },
        }

        rows = []
        for seed, step, source_random_seed in (
            (900100, 259, 8),
            (900101, 233, 8),
            (900102, 271, 11),
        ):
            row = deepcopy(base_result)
            row["seed"] = seed
            row["step"] = step
            row["trace_tail"][-1]["step"] = step
            _attach_failure_metadata(row, backend="v2", source_random_seed=source_random_seed)
            rows.append(row)

        summary = _summarize_clusters(rows, top_n=5)

        self.assertEqual(summary["cluster_count"], 1)
        only_cluster_key = next(iter(summary["cluster_counts"]))
        self.assertEqual(summary["cluster_counts"][only_cluster_key], 3)
        self.assertEqual(summary["top_clusters"][0]["source_coverage"], 2)
        self.assertEqual(summary["top_clusters"][0]["count"], 3)
        labels = {entry["label"] for entry in summary["top_clusters"][0]["exemplars"]}
        self.assertIn("prototype", labels)
        self.assertIn("earliest_clean", labels)
        self.assertIn("variant", labels)

    def test_taskmaster_hp_init_uses_lightspeed_range(self):
        rng = StsRandom(7)
        monster = make_monster("Taskmaster", rng, ascension=0)
        self.assertGreaterEqual(monster.current_hp, 54)
        self.assertLessEqual(monster.current_hp, 60)
        self.assertEqual(rng.counter, 2)

    def test_collector_starts_with_invalid_slots_and_spawn_move_in_v2(self):
        combat = NativeCombatEnv(seed=31, ascension_level=0, scheduled_encounter="TheCollector", floor=33, act=2)

        self.assertEqual([monster.monster_id for monster in combat.monsters], ["INVALID = 0", "INVALID = 0", "TheCollector"])
        self.assertEqual(combat.monsters[2].move, "THE_COLLECTOR_SPAWN")
        self.assertEqual(combat.monsters[2].intent, "MAGIC")
        self.assertEqual(combat.ai_rng.counter, 1)

    def test_bronze_orb_stasis_uses_combat_sort_order_in_v2(self):
        combat = NativeCombatEnv(seed=3110, ascension_level=0, player=PlayerState())
        left = make_monster("BronzeOrb", StsRandom(3111), ascension=0)
        center = make_monster("BronzeAutomaton", StsRandom(3112), ascension=0)
        right = make_monster("BronzeOrb", StsRandom(3113), ascension=0)
        left.move = "BRONZE_ORB_STASIS"
        right.move = "BRONZE_ORB_STASIS"
        combat.monsters = [left, center, right]
        combat.draw_pile = [
            make_card("Brutality", uuid="stasis-brutality"),
            make_card("Exhume", uuid="stasis-exhume"),
        ]
        combat.discard_pile = []
        combat.hand = []
        combat.card_random_rng = StsRandom(3)

        combat._monster_take_turn(left, 0)
        combat._monster_take_turn(right, 2)

        self.assertEqual(left.ai_state["stasis_card"].card_id, "Brutality")
        self.assertEqual(right.ai_state["stasis_card"].card_id, "Exhume")

    def test_exhume_card_select_uses_exhaust_pile_order_in_v2(self):
        combat = NativeCombatEnv(seed=31105, ascension_level=0, player=PlayerState())
        older = make_card("Apparition", uuid="older-apparition")
        filler = make_card("Carnage", uuid="filler")
        recent = make_card("Apparition", uuid="recent-apparition")
        recent.cost_for_combat = 0
        recent.cost_for_turn = 0
        combat.exhaust_pile = [older, filler, recent]
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []

        combat._open_exhaust_card_select("EXHUME", [0, 1, 2])
        combat._resolve_card_select({"kind": "card_select", "select_index": 0})

        self.assertEqual([(card.card_id, card.uuid) for card in combat.hand], [("Apparition", "older-apparition")])
        self.assertIsNone(combat.hand[0].cost_for_combat)
        self.assertEqual([card.uuid for card in combat.exhaust_pile], ["filler", "recent-apparition"])

    def test_exhume_card_select_uses_actual_exhaust_indexes_in_v2(self):
        combat = NativeCombatEnv(seed=31106, ascension_level=0, player=PlayerState())
        first = make_card("Defend_R", uuid="exhume-first")
        skipped = make_card("Exhume", uuid="exhume-self")
        last = make_card("Warcry", uuid="exhume-last")
        combat.exhaust_pile = [first, skipped, last]
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []

        combat._open_exhaust_card_select("EXHUME", [0, 2])

        self.assertEqual(
            [(option["choice_index"], option["select_index"], option["card_id"]) for option in combat.card_select_options],
            [(0, 0, "Defend_R"), (1, 2, "Warcry")],
        )

        combat._resolve_card_select({"kind": "card_select", "select_index": 2})

        self.assertEqual([(card.card_id, card.uuid) for card in combat.hand], [("Warcry", "exhume-last")])
        self.assertEqual([card.uuid for card in combat.exhaust_pile], ["exhume-first", "exhume-self"])


    def test_forethought_puts_selected_card_on_top_of_draw_pile_in_v2(self):
        combat = NativeCombatEnv(seed=3111, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Strike_R", uuid="ft-strike-a"),
            make_card("Strike_R", uuid="ft-strike-b"),
        ]
        combat.draw_pile = [
            make_card("Defend_R", uuid="ft-draw-defend"),
            make_card("Bash", uuid="ft-draw-bash"),
        ]
        combat.card_select_context = "FORETHOUGHT"
        combat.pending_resolve_card = make_card("Forethought", uuid="ft-card")

        from spirecomm.native_sim_v2.combat_core import resolve_card_select

        resolve_card_select(combat, {"kind": "card_select", "select_index": 0})

        self.assertEqual(combat.draw_pile[0].uuid, "ft-strike-a")
        self.assertTrue(combat.draw_pile[0].free_to_play_once)

    def test_madness_uses_effective_turn_cost_when_cost_for_turn_is_none_in_v2(self):
        combat = NativeCombatEnv(seed=3902, ascension_level=0, player=PlayerState(energy=1))
        combat.monsters = [make_monster("Lagavulin", StsRandom(3902), ascension=0)]
        combat.hand = [
            make_card("Strike_R", uuid="md-strike"),
            make_card("Blood for Blood", uuid="md-bfb"),
            make_card("Madness", uuid="md-madness"),
            make_card("Dark Embrace", uuid="md-dark-embrace"),
        ]
        combat.card_random_rng.seed0 = 13933345782742587896
        combat.card_random_rng.seed1 = 4613865217662875590
        combat.card_random_rng.counter = 0

        combat.play_card_impl(2, 0)

        hand_by_uuid = {card.uuid: card for card in combat.hand}
        self.assertEqual(hand_by_uuid["md-dark-embrace"].cost_for_turn, 0)
        self.assertEqual(hand_by_uuid["md-bfb"].cost, 4)
        self.assertIsNone(hand_by_uuid["md-bfb"].cost_for_turn)

    def test_v2_act_map_uses_lightspeed_room_types(self):
        env = NativeRunEnv(seed=7276044427155637262, ascension_level=0, enable_neow=True)
        env._generate_act_map(1)
        lightspeed_map = sts.SpireMap(7276044427155637262, 0, 1, False)

        for row in range(4):
            floor = row + 1
            for node_id in env.map_layers[floor]:
                node = env.map_graph[node_id]
                room = str(lightspeed_map.get_room_type(node["x"], row)).split(".")[-1]
                expected = {
                    "SHOP": "$",
                    "REST": "R",
                    "EVENT": "?",
                    "ELITE": "E",
                    "MONSTER": "M",
                    "TREASURE": "T",
                    "BOSS": "BOSS",
                    "BOSS_TREASURE": "BOSS",
                }.get(room)
                self.assertEqual(node["symbol"], expected)

    def test_perfected_strike_count_counts_current_card_once_without_exhaust_in_v2(self):
        combat = NativeCombatEnv(seed=500, ascension_level=0, player=PlayerState())
        perfected = make_card("Perfected Strike", uuid="perfected")
        combat.hand = [make_card("Strike_R", uuid="hand-strike"), perfected]
        combat.draw_pile = [make_card("Strike_R", uuid="draw-strike-a"), make_card("Strike_R", uuid="draw-strike-b")]
        combat.discard_pile = [make_card("Bash", uuid="discard-bash")]
        combat.exhaust_pile = [make_card("Strike_R", uuid="exhaust-strike")]

        self.assertEqual(_combat_strike_count(combat, perfected), 4)

    def test_perfected_strike_count_does_not_double_count_current_card_in_discard_in_v2(self):
        combat = NativeCombatEnv(seed=501, ascension_level=0, player=PlayerState())
        perfected = make_card("Perfected Strike", uuid="perfected")
        combat.hand = [make_card("Strike_R", uuid="hand-strike")]
        combat.draw_pile = [make_card("Strike_R", uuid="draw-strike-a"), make_card("Strike_R", uuid="draw-strike-b")]
        combat.discard_pile = [make_card("Double Tap", uuid="discard-double"), perfected]

        self.assertEqual(_combat_strike_count(combat, perfected), 4)

    def test_perfected_strike_replay_does_not_count_itself_after_leaving_hand_in_v2(self):
        combat = NativeCombatEnv(seed=502, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        target = combat.monsters[0]
        perfected = make_card("Perfected Strike", uuid="perfected")
        combat.player.powers["Strength"] = -2
        target.current_hp = 50
        target.max_hp = 50
        target.powers["Vulnerable"] = 1
        combat.hand = [make_card("Strike_R", uuid="hand-strike-a"), make_card("Strike_R", uuid="hand-strike-b")]
        combat.draw_pile = [make_card("Strike_R", uuid="draw-strike-a"), make_card("Strike_R", uuid="draw-strike-b")]
        combat.discard_pile = [make_card("Double Tap", uuid="discard-double"), make_card("Havoc", uuid="discard-havoc")]
        combat.exhaust_pile = [make_card("Forethought", uuid="exhaust-forethought"), make_card("Defend_R", uuid="exhaust-defend")]

        combat._replay_attack_card_effect(perfected, target, 0)

        self.assertEqual(target.current_hp, 32)

    def test_add_curse_to_deck_triggers_ceramic_fish_in_v2(self):
        env = NativeRunEnv(seed=71, ascension_level=0, enable_neow=True)
        env.gold = 10
        env.relics.append({"relic_id": "Ceramic Fish", "id": "Ceramic Fish", "name": "Ceramic Fish", "counter": -1, "price": 0, "tier": "COMMON"})

        added = env._add_curse_to_deck("Doubt", uuid="curse-doubt")

        self.assertTrue(added)
        self.assertEqual(env.gold, 19)
        self.assertEqual(env.deck[-1].card_id, "Doubt")

    def test_obtain_sundial_starts_counter_at_zero_in_v2(self):
        env = NativeRunEnv(seed=71, ascension_level=0, enable_neow=True)

        env._obtain_relic(make_relic("Sundial"))

        sundial = next(relic for relic in env.relics if relic["relic_id"] == "Sundial")
        self.assertEqual(sundial["counter"], 0)

    def test_golden_idol_outrun_adds_injury_and_triggers_ceramic_fish_in_v2(self):
        env = NativeRunEnv(seed=72, ascension_level=0, enable_neow=True)
        env.phase = "EVENT"
        env.current_event_id = "Golden Idol"
        env.floor = 4
        env.gold = 33
        env.relics.append({"relic_id": "Ceramic Fish", "id": "Ceramic Fish", "name": "Ceramic Fish", "counter": -1, "price": 0, "tier": "COMMON"})

        from spirecomm.native_sim_v2.run_core import step

        state = step(env, {"kind": "event", "event_id": "Golden Idol", "name": "Outrun", "choice_index": 2})

        self.assertEqual(env.gold, 42)
        self.assertTrue(any(card.card_id == "Injury" for card in env.deck))
        self.assertEqual(state["screen"], "MAP")

    def test_golden_idol_outrun_consumes_omamori_and_skips_injury_in_v2(self):
        env = NativeRunEnv(seed=72, ascension_level=0, enable_neow=True)
        env.phase = "EVENT"
        env.current_event_id = "Golden Idol"
        env.floor = 4
        env.relics.append({"relic_id": "Omamori", "id": "Omamori", "name": "Omamori", "counter": 2, "price": 0, "tier": "COMMON"})

        from spirecomm.native_sim_v2.run_core import step

        state = step(env, {"kind": "event", "event_id": "Golden Idol", "name": "Outrun", "choice_index": 2})

        omamori = next(relic for relic in env.relics if relic["relic_id"] == "Omamori")
        self.assertEqual(omamori["counter"], 1)
        self.assertFalse(any(card.card_id == "Injury" for card in env.deck))
        self.assertEqual(state["screen"], "MAP")

    def test_feed_kill_heals_with_magic_flower_in_v2(self):
        combat = NativeCombatEnv(seed=73, ascension_level=0, player=PlayerState(current_hp=2, max_hp=80))
        combat.relics.append({"relic_id": "Magic Flower", "id": "Magic Flower", "name": "Magic Flower", "counter": -1, "price": 0, "tier": "RARE"})
        target = make_monster("Cultist", StsRandom(73), ascension=0)
        target.current_hp = 10
        combat.monsters = [target]
        combat.hand = [make_card("Feed", uuid="feed-test")]

        combat.play_card_impl(0, 0)

        self.assertEqual(combat.player.max_hp, 83)
        self.assertEqual(combat.player.current_hp, 6)

    def test_kill_rewards_skip_gremlin_leader_minions_in_v2(self):
        for card_id in ("Feed", "Hand of Greed", "Ritual Dagger"):
            with self.subTest(card_id=card_id):
                combat = NativeCombatEnv(seed=74, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
                leader = make_monster("GremlinLeader", StsRandom(74), ascension=0)
                minion = make_monster("GremlinThief", StsRandom(75), ascension=0)
                minion.ai_state["leader_minion"] = 1
                minion.current_hp = 10
                combat.monsters = [leader, minion]
                card = make_card(card_id, uuid=f"{card_id}-minion-test")
                if card_id == "Ritual Dagger":
                    card.misc = 7
                combat.hand = [card]

                combat.play_card_impl(0, 1)

                self.assertEqual(minion.current_hp, 0)
                self.assertEqual(combat.player.max_hp, 80)
                self.assertEqual(combat.player.current_hp, 40)
                if card_id == "Hand of Greed":
                    self.assertEqual(combat.gold_gain, 0)
                if card_id == "Ritual Dagger":
                    self.assertEqual(card.misc, 7)

    def test_flex_strength_down_is_blocked_by_artifact_before_monster_weak_in_v2(self):
        combat = NativeCombatEnv(seed=73, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        slime = make_monster("AcidSlime_M", StsRandom(73), ascension=0)
        slime.move = "ACID_SLIME_M_LICK"
        slime.intent = "DEBUFF"
        slime.move_base_damage = 0
        slime.move_hits = 0
        combat.monsters = [slime]
        combat.player.powers["Strength"] = 2
        combat.player.powers["Flex Strength Down"] = 2
        combat.player.powers["Artifact"] = 1
        combat.hand = [make_card("Strike_R", uuid="end-turn-card")]

        combat.end_turn()

        self.assertEqual(combat.player.power("Strength"), 2)
        self.assertEqual(combat.player.power("Artifact"), 0)
        self.assertEqual(combat.player.power("Weakened"), 1)

    def test_horn_cleat_ignores_frail_and_triggers_juggernaut_in_v2(self):
        combat = NativeCombatEnv(seed=731, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        combat.relics.append({"relic_id": "Horn Cleat", "id": "Horn Cleat", "name": "Horn Cleat", "counter": -1, "price": 0, "tier": "UNCOMMON"})
        target = make_monster("Cultist", StsRandom(732), ascension=0)
        target.current_hp = 5
        combat.monsters = [target]
        combat.turn = 0
        combat.player.block = 0
        combat.player.energy = 0
        combat.player.powers["Frail"] = 2
        combat.player.add_power("Juggernaut", 5)

        combat.start_player_turn()

        self.assertEqual(combat.player.block, 14)
        self.assertEqual(target.current_hp, 0)

    def test_hand_of_greed_triggers_sharp_hide_counter_damage_in_v2(self):
        combat = NativeCombatEnv(seed=7311, ascension_level=0, scheduled_encounter=["TheGuardian"], player=PlayerState(current_hp=40, max_hp=80))
        guardian = combat.monsters[0]
        guardian.move = "THE_GUARDIAN_TWIN_SLAM"
        guardian.intent = "ATTACK"
        guardian.move_base_damage = 8
        guardian.move_hits = 2
        guardian.powers["Sharp Hide"] = 3
        combat.player.block = 5
        combat.player.add_power("Strength", 2)
        combat.player.add_power("Weakened", 1)
        combat.hand = [make_card("HandOfGreed", uuid="hog")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)

        self.assertEqual(combat.player.block, 2)

    def test_hand_of_greed_does_not_apply_curl_up_to_dead_louse_in_v2(self):
        combat = NativeCombatEnv(seed=7312, ascension_level=0, scheduled_encounter=["RedLouse", "RedLouse"], player=PlayerState(current_hp=40, max_hp=80))
        first, second = combat.monsters
        first.current_hp = 5
        first.max_hp = 12
        first.powers["Curl Up"] = 5
        second.current_hp = 20
        second.max_hp = 20
        combat.hand = [make_card("HandOfGreed", uuid="hog")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)

        self.assertEqual(first.current_hp, 0)
        self.assertEqual(first.block, 0)
        self.assertEqual(first.power("Curl Up"), 5)

    def test_captains_wheel_ignores_frail_and_triggers_juggernaut_in_v2(self):
        combat = NativeCombatEnv(seed=733, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        combat.relics.append({"relic_id": "Captain's Wheel", "id": "Captain's Wheel", "name": "Captain's Wheel", "counter": -1, "price": 0, "tier": "RARE"})
        target = make_monster("Cultist", StsRandom(734), ascension=0)
        target.current_hp = 5
        combat.monsters = [target]
        combat.turn = 1
        combat.player.block = 0
        combat.player.energy = 0
        combat.player.powers["Frail"] = 2
        combat.player.add_power("Juggernaut", 5)

        combat.start_player_turn()

        self.assertEqual(combat.player.block, 18)
        self.assertEqual(target.current_hp, 0)

    def test_entrench_triggers_juggernaut_on_gained_block_in_v2(self):
        combat = NativeCombatEnv(seed=74, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80, block=5))
        target = make_monster("Cultist", StsRandom(74), ascension=0)
        target.current_hp = 5
        combat.monsters = [target]
        combat.player.add_power("Juggernaut", 5)
        combat.hand = [make_card("Entrench", uuid="entrench-test")]

        combat.play_card_impl(0, 0)

        self.assertEqual(combat.player.block, 10)
        self.assertEqual(target.current_hp, 0)

    def test_byrd_flight_halves_attack_damage_and_then_decrements_in_v2(self):
        combat = NativeCombatEnv(seed=741, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        byrd = make_monster("Byrd", StsRandom(742), ascension=0)
        byrd.current_hp = 27
        byrd.max_hp = 27
        byrd.powers["Flight"] = 3
        combat.monsters = [byrd]
        combat.hand = [make_card("Bash", uuid="bash-byrd-flight")]

        combat.play_card_impl(0, 0)

        self.assertEqual(byrd.current_hp, 23)
        self.assertEqual(byrd.power("Flight"), 2)
        self.assertEqual(byrd.power("Vulnerable"), 2)

    def test_blood_for_blood_draws_with_misc_reduced_cost_in_v2(self):
        combat = NativeCombatEnv(seed=75, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80, energy=1))
        combat.monsters = [make_monster("Cultist", StsRandom(75), ascension=0)]
        card = make_card("Blood for Blood", uuid="bfb-draw")
        card.misc = 4
        combat.draw_pile = [card]
        combat.discard_pile = []
        combat.hand = []

        combat.draw_cards(1)

        self.assertEqual(len(combat.hand), 1)
        self.assertEqual(combat.hand[0].cost_for_turn, 0)
        self.assertTrue(combat.playable(combat.hand[0]))

    def test_blood_for_blood_clears_turn_only_free_cost_at_end_turn_in_v2(self):
        combat = NativeCombatEnv(seed=76, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        card = make_card("Blood for Blood", uuid="bfb-temp")
        card.cost_for_turn = 0
        combat.discard_pile = [card]

        combat._clear_temporary_cost_state()

        self.assertEqual(combat.discard_pile[0].cost_for_turn, 4)

    def test_blood_for_blood_keeps_damage_reduction_across_end_turn_in_v2(self):
        combat = NativeCombatEnv(seed=77, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        card = make_card("Blood for Blood", uuid="bfb-retain")
        card.misc = 2
        card.cost_for_turn = 2
        combat.discard_pile = [card]

        combat._clear_temporary_cost_state()

        self.assertEqual(combat.discard_pile[0].cost_for_turn, 2)

    def test_blood_for_blood_keeps_madness_zero_cost_across_end_turn_in_v2(self):
        combat = NativeCombatEnv(seed=78, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        card = make_card("Blood for Blood", uuid="bfb-madness")
        card.card_def = replace(card.card_def, cost=0, upgraded_cost=0)
        card.cost_for_turn = 0
        combat.discard_pile = [card]

        combat._clear_temporary_cost_state()

        self.assertEqual(combat.discard_pile[0].cost_for_turn, 0)

    def test_confusion_rerolls_blood_for_blood_at_three_cost_in_v2(self):
        combat = NativeCombatEnv(seed=781, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        card = make_card("Blood for Blood", uuid="bfb-confusion")
        card.misc = 1
        card.cost_for_turn = 3
        combat.draw_pile = [card]
        combat.discard_pile = []
        combat.hand = []
        combat.player.powers["Confusion"] = 1

        class _FixedConfusionRng:
            def randint(self, start, end):
                return 0

        combat.card_random_rng = _FixedConfusionRng()
        combat.draw_cards(1)

        self.assertEqual(combat.hand[0].cost_for_turn, 0)
        self.assertIsNone(combat.hand[0].cost_for_combat)

    def test_confusion_rerolls_blood_for_blood_after_damage_reduction_in_v2(self):
        combat = NativeCombatEnv(seed=782, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        card = make_card("Blood for Blood", upgrades=1, uuid="bfb-confusion-upgraded")
        card.misc = 1
        card.cost_for_turn = 2
        combat.draw_pile = [card]
        combat.discard_pile = []
        combat.hand = []
        combat.player.powers["Confusion"] = 1

        class _FixedConfusionRng:
            def randint(self, start, end):
                return 0

        combat.card_random_rng = _FixedConfusionRng()
        combat.draw_cards(1)

        self.assertEqual(combat.hand[0].cost_for_turn, 0)
        self.assertIsNone(combat.hand[0].cost_for_combat)

    def test_confusion_rerolls_unreduced_blood_for_blood_in_v2(self):
        combat = NativeCombatEnv(seed=783, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        card = make_card("Blood for Blood", uuid="bfb-confusion-base")
        card.cost_for_turn = 4
        combat.draw_pile = [card]
        combat.discard_pile = []
        combat.hand = []
        combat.player.powers["Confusion"] = 1

        class _FixedConfusionRng:
            def randint(self, start, end):
                return 0

        combat.card_random_rng = _FixedConfusionRng()
        combat.draw_cards(1)

        self.assertEqual(combat.hand[0].cost_for_turn, 0)
        self.assertIsNone(combat.hand[0].cost_for_combat)

    def test_mayhem_queues_top_card_before_draw_and_resolves_after_draw_in_v2(self):
        combat = NativeCombatEnv(seed=91, ascension_level=0, player=PlayerState())
        target = make_monster("Cultist", StsRandom(91), ascension=0)
        initial_hp = target.current_hp
        combat.monsters = [target]
        combat.hand = []
        combat.draw_pile = [make_card("Strike_R", uuid="mayhem-strike")]
        combat.discard_pile = []
        combat.player.powers["Mayhem"] = 1

        combat.start_player_turn()

        self.assertEqual(combat.pending_autoplay_cards, [])
        self.assertLess(target.current_hp, initial_hp)
        self.assertEqual([card.card_id for card in combat.hand], [])
        self.assertEqual([card.card_id for card in combat.exhaust_pile], [])

    def test_havoc_playing_power_does_not_force_exhaust_in_v2(self):
        combat = NativeCombatEnv(seed=93, ascension_level=0, player=PlayerState())
        combat.monsters = [make_monster("Cultist", StsRandom(93), ascension=0)]
        havoc = make_card("Havoc", uuid="havoc")
        combat.hand = [havoc]
        combat.draw_pile = [make_card("Corruption", uuid="corruption-top")]
        combat.discard_pile = []

        combat.play_card_impl(0, 0)

        self.assertFalse(any(card.card_id == "Corruption" for card in combat.exhaust_pile))
        self.assertNotIn("Corruption", [card.card_id for card in combat.hand])

    def test_mayhem_headbutt_card_select_resumes_start_turn_draw_in_v2(self):
        combat = NativeCombatEnv(seed=92, ascension_level=0, player=PlayerState())
        target = make_monster("Cultist", StsRandom(92), ascension=0)
        combat.monsters = [target]
        combat.hand = []
        combat.draw_pile = [
            make_card("Strike_R", uuid="draw-0"),
            make_card("Defend_R", uuid="draw-1"),
            make_card("Bash", uuid="draw-2"),
            make_card("Strike_R", uuid="draw-3"),
            make_card("Defend_R", uuid="draw-4"),
            make_card("Headbutt", uuid="top-headbutt"),
        ]
        combat.discard_pile = [
            make_card("Anger", uuid="discard-anger"),
            make_card("Flex", uuid="discard-flex"),
        ]
        combat.player.powers["Mayhem"] = 1

        combat.start_player_turn()

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertTrue(combat.pending_start_turn_resume)
        self.assertEqual(len(combat.hand), 5)

        from spirecomm.native_sim_v2.combat_core import resolve_card_select
        resolve_card_select(combat, {"kind": "card_select", "select_type": "HEADBUTT", "choice_index": 0, "deck_index": 0})

        self.assertFalse(combat.pending_start_turn_resume)
        self.assertEqual(combat.card_select_context, None)
        self.assertEqual(len(combat.hand), 5)

    def test_mayhem_true_grit_exhausts_post_draw_hand_card_in_v2(self):
        combat = NativeCombatEnv(seed=293, ascension_level=0, player=PlayerState())
        combat.monsters = [make_monster("Cultist", StsRandom(293), ascension=0)]
        combat.hand = []
        combat.draw_pile = [
            make_card("Defend_R", uuid="draw-defend"),
            make_card("True Grit", uuid="top-true-grit"),
        ]
        combat.discard_pile = []
        combat.player.powers["Mayhem"] = 1

        combat.start_player_turn()

        self.assertEqual([card.card_id for card in combat.hand], [])
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Defend_R"])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["True Grit"])

    def test_collector_torch_head_death_keeps_corpse_slot_in_v2(self):
        combat = NativeCombatEnv(seed=32, ascension_level=0, player=PlayerState())
        collector = make_monster("TheCollector", StsRandom(32), ascension=0)
        torch = make_monster("TorchHead", StsRandom(33), ascension=0)
        combat.monsters = [make_monster("INVALID = 0", StsRandom(34), ascension=0), torch, collector]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)
        torch.add_power("Regenerate", 5)
        torch.block = 7
        torch.current_hp = 0

        combat._on_monster_defeated(torch)

        self.assertEqual(combat.monsters[1].monster_id, "TorchHead")
        self.assertEqual(combat.monsters[1].current_hp, 0)
        self.assertEqual(combat.monsters[1].block, 7)
        self.assertEqual(combat.monsters[1].power("Regenerate"), 5)

    def test_gremlin_leader_slot_zero_minion_death_keeps_corpse_in_v2(self):
        combat = NativeCombatEnv(seed=33, ascension_level=0, player=PlayerState())
        leader = make_monster("GremlinLeader", StsRandom(33), ascension=0)
        gremlin = make_monster("GremlinWizard", StsRandom(34), ascension=0)
        combat.monsters = [gremlin, make_monster("GremlinFat", StsRandom(36), ascension=0), make_monster("GremlinThief", StsRandom(37), ascension=0), leader]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)
        gremlin.current_hp = 0

        combat._on_monster_defeated(gremlin)

        self.assertEqual(combat.monsters[0].monster_id, "GremlinWizard")
        self.assertEqual(combat.monsters[0].current_hp, 0)

    def test_gremlin_leader_non_slot_zero_minion_death_keeps_corpse_in_v2(self):
        combat = NativeCombatEnv(seed=34, ascension_level=0, player=PlayerState())
        leader = make_monster("GremlinLeader", StsRandom(34), ascension=0)
        gremlin = make_monster("GremlinFat", StsRandom(35), ascension=0)
        combat.monsters = [make_monster("INVALID = 0", StsRandom(36), ascension=0), gremlin, make_monster("GremlinThief", StsRandom(37), ascension=0), leader]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)
        gremlin.current_hp = 0

        combat._on_monster_defeated(gremlin)

        self.assertEqual(combat.monsters[1].monster_id, "GremlinFat")

    def test_exordium_thugs_consumes_all_candidate_construct_rolls(self):
        hp_rng = StsRandom(11)
        misc_rng = StsRandom(11)
        monsters = encounter_to_monsters("ExordiumThugs", hp_rng, misc_rng, ascension=0)
        self.assertEqual(len(monsters), 2)
        self.assertEqual([monster.monster_id for monster in monsters], ["GreenLouse", "Looter"])
        self.assertEqual(hp_rng.counter, 7)

    def test_discovery_plus_moves_to_discard_in_v2(self):
        combat = NativeCombatEnv(seed=3, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Discovery", upgrades=1, uuid="disc-plus")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3

        combat.play_card(0, 0)
        combat._resolve_card_select({"kind": "card_select", "select_index": 0})

        self.assertIn("Discovery", [card.card_id for card in combat.discard_pile])
        self.assertNotIn("Discovery", [card.card_id for card in combat.exhaust_pile])

    def test_discovery_respects_corruption_cost_marker_in_v2(self):
        combat = NativeCombatEnv(seed=5, ascension_level=0, player=PlayerState())
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.powers["Corruption"] = 1
        combat.card_select_context = "DISCOVERY"
        combat.card_select_generated_cards = [make_card("Warcry", uuid="generated-warcry")]
        combat.pending_resolve_card = make_card("Discovery", uuid="disc")

        combat._resolve_card_select({"kind": "card_select", "select_index": 0})

        self.assertEqual(len(combat.hand), 1)
        self.assertEqual(combat.hand[0].card_id, "Warcry")
        self.assertEqual(combat.hand[0].cost_for_turn, -9)

    def test_played_discovery_generated_skill_loses_turn_only_zero_cost_in_v2(self):
        combat = NativeCombatEnv(seed=5005, ascension_level=0, player=PlayerState())
        combat.monsters = [make_monster("Lagavulin", StsRandom(5005), ascension=0)]
        combat.hand = [make_card("Strike_R", uuid="discovery-strike")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.card_select_context = "DISCOVERY"
        combat.card_select_generated_cards = [make_card("Rage", uuid="Discovery-Rage-test")]
        combat.pending_resolve_card = make_card("Discovery", uuid="disc")
        combat.player.energy = 2

        combat._resolve_card_select({"kind": "card_select", "select_index": 0})

        self.assertEqual(combat.hand[1].card_id, "Rage")
        self.assertEqual(combat.hand[1].cost_for_turn, 0)
        self.assertIsNone(combat.hand[1].cost_for_combat)

        combat.play_card(1, 0)

        combat._clear_temporary_cost_state()

        self.assertEqual(combat.discard_pile[0].card_id, "Rage")
        self.assertIsNone(combat.discard_pile[0].cost_for_turn)
        self.assertIsNone(combat.discard_pile[0].cost_for_combat)
        self.assertEqual(combat.discard_pile[0].cost, 0)

    def test_nilrys_codex_opens_end_turn_card_select_and_resumes_turn_in_v2(self):
        combat = NativeCombatEnv(
            seed=7,
            ascension_level=0,
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Nilry's Codex", "id": "Nilry's Codex", "name": "Nilry's Codex", "counter": -1, "price": 0, "tier": "EVENT"},
            ],
        )
        combat.turn = 0
        combat.player.energy = 0
        combat.hand = [make_card("Strike_R", uuid="strike"), make_card("Defend_R", uuid="defend")]
        combat.draw_pile = [make_card("Bash", uuid="bash"), make_card("Armaments", uuid="arm")]
        combat.discard_pile = []

        combat.end_turn()

        self.assertEqual(combat.card_select_context, "CODEX")
        self.assertTrue(combat.pending_end_turn_resume)
        self.assertEqual([card.card_id for card in combat.hand], ["Strike_R", "Defend_R"])
        self.assertEqual(len(combat.card_select_generated_cards), 3)
        self.assertEqual(len(combat.card_select_options), 4)

        selected_id = combat.card_select_generated_cards[0].card_id
        combat._resolve_card_select({"kind": "single_card_select", "select_index": 0})

        self.assertIsNone(combat.card_select_context)
        self.assertFalse(combat.pending_end_turn_resume)
        self.assertEqual(combat.turn, 1)
        self.assertEqual(combat.player.energy, 3)
        reachable_cards = [card.card_id for card in combat.hand + combat.draw_pile + combat.discard_pile]
        self.assertIn(selected_id, reachable_cards)

    def test_opening_turn_starts_at_zero_in_v2(self):
        combat = NativeCombatEnv(seed=701, ascension_level=0, player=PlayerState())
        self.assertEqual(combat.turn, 0)

    def test_stone_calendar_triggers_on_seventh_turn_in_v2(self):
        relics = [
            {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
            {"relic_id": "Stone Calendar", "id": "Stone Calendar", "name": "Stone Calendar", "counter": -1, "price": 0, "tier": "RARE"},
        ]

        combat = NativeCombatEnv(seed=702, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState(), relics=[dict(r) for r in relics])
        cultist = combat.monsters[0]
        cultist.current_hp = 60
        combat.turn = 5
        combat.start_player_turn()
        self.assertEqual(cultist.current_hp, 60)

        combat = NativeCombatEnv(seed=703, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState(), relics=[dict(r) for r in relics])
        cultist = combat.monsters[0]
        cultist.current_hp = 60
        combat.turn = 6
        combat.end_turn()
        self.assertEqual(cultist.current_hp, 8)

    def test_attack_damage_respects_intangible_before_block_in_v2(self):
        combat = NativeCombatEnv(seed=704, ascension_level=0, scheduled_encounter=["FungiBeast"], player=PlayerState())
        fungi = combat.monsters[0]
        fungi.move = "FUNGI_BEAST_BITE"
        fungi.intent = "ATTACK"
        fungi.move_base_damage = 6
        fungi.move_hits = 1
        combat.player.current_hp = 20
        combat.player.block = 5
        combat.player.powers["Intangible"] = 1

        combat._monster_take_turn(fungi, 0)

        self.assertEqual(combat.player.current_hp, 20)
        self.assertEqual(combat.player.block, 4)

    def test_sharp_hide_counter_damage_respects_intangible_before_block_in_v2(self):
        combat = NativeCombatEnv(seed=705, ascension_level=0, scheduled_encounter=["TheGuardian"], player=PlayerState())
        guardian = combat.monsters[0]
        guardian.powers["Sharp Hide"] = 3
        combat.player.current_hp = 20
        combat.player.block = 10
        combat.player.powers["Intangible"] = 1
        combat.player.energy = 1
        combat.hand = [make_card("Strike_R", uuid="strike")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 20)
        self.assertEqual(combat.player.block, 9)

    def test_sharp_hide_counter_damage_reduces_blood_for_blood_once_in_v2(self):
        combat = NativeCombatEnv(seed=706, ascension_level=0, scheduled_encounter=["TheGuardian"], player=PlayerState())
        guardian = combat.monsters[0]
        guardian.powers["Sharp Hide"] = 3
        combat.player.current_hp = 40
        combat.player.energy = 2
        blood_for_blood = make_card("Blood for Blood", uuid="bfb-sharp-hide")
        blood_for_blood.misc = 1
        blood_for_blood.cost_for_turn = 3
        combat.hand = [make_card("Bash", uuid="bash"), blood_for_blood]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertEqual(combat.hand[0].card_id, "Blood for Blood")
        self.assertEqual(combat.hand[0].misc, 2)
        self.assertEqual(combat.hand[0].cost_for_turn, 2)

    def test_event_percent_hp_loss_defaults_to_floor_in_v2(self):
        env = NativeRunEnv(seed=5, ascension_level=0, enable_neow=True)
        env.player.max_hp = 86
        self.assertEqual(_event_percent_hp_loss(env, 0.25), 21)
        self.assertEqual(_event_percent_hp_loss(env, 0.10), 8)

    def test_shining_light_hp_loss_uses_rounding_in_v2(self):
        env = NativeRunEnv(seed=6, ascension_level=0, enable_neow=True)
        env.player.max_hp = 86
        self.assertEqual(_event_percent_hp_loss(env, 0.20, mode="round"), 17)

    def test_brutality_start_turn_does_not_trigger_rupture_in_v2(self):
        combat = NativeCombatEnv(seed=8, ascension_level=0, player=PlayerState())
        combat.turn = 2
        combat.player.current_hp = 20
        combat.player.max_hp = 20
        combat.player.powers["Brutality"] = 1
        combat.player.powers["Rupture"] = 1
        combat.hand = []
        combat.draw_pile = [make_card("Strike_R", uuid="s1"), make_card("Defend_R", uuid="d1")]
        combat.discard_pile = []

        combat.start_player_turn()

        self.assertEqual(combat.player.current_hp, 19)
        self.assertEqual(combat.player.power("Strength"), 0)
        self.assertEqual(len(combat.hand), 2)

    def test_brutality_start_turn_can_immediately_set_player_loss_in_v2(self):
        combat = NativeCombatEnv(seed=9, ascension_level=0, player=PlayerState())
        combat.turn = 2
        combat.player.current_hp = 1
        combat.player.max_hp = 20
        combat.player.powers["Brutality"] = 1
        combat.hand = []
        combat.draw_pile = [make_card("Strike_R", uuid="s1")]
        combat.discard_pile = []

        combat.start_player_turn()

        self.assertEqual(combat.player.current_hp, 0)
        self.assertEqual(combat.outcome, "PLAYER_LOSS")

    def test_mercury_hourglass_victory_still_applies_brutality_hp_loss_in_v2(self):
        combat = NativeCombatEnv(seed=10, ascension_level=0, player=PlayerState())
        combat.turn = 2
        combat.player.current_hp = 20
        combat.player.max_hp = 20
        combat.player.powers["Brutality"] = 1
        combat.relics.append({"relic_id": "Mercury Hourglass", "name": "Mercury Hourglass", "tier": "UNCOMMON", "counter": -1})
        cultist = make_monster("Cultist", StsRandom(10), ascension=0)
        cultist.current_hp = 3
        cultist.max_hp = 48
        combat.monsters = [cultist]

        combat.start_player_turn()

        self.assertEqual(combat.outcome, "PLAYER_VICTORY")
        self.assertEqual(combat.player.current_hp, 19)

    def test_empty_piles_without_passive_damage_sets_player_loss_in_v2(self):
        combat = NativeCombatEnv(seed=11, ascension_level=0, scheduled_encounter=["Lagavulin"], player=PlayerState(current_hp=19, max_hp=80))
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = [make_card("Strike_R", uuid="exhaust-strike")]

        combat._check_outcome()

        self.assertEqual(combat.outcome, "PLAYER_LOSS")

    def test_empty_piles_with_the_bomb_does_not_set_player_loss_in_v2(self):
        combat = NativeCombatEnv(seed=12, ascension_level=0, scheduled_encounter=["Lagavulin"], player=PlayerState(current_hp=19, max_hp=80))
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = [make_card("Strike_R", uuid="exhaust-strike")]
        combat.player.powers["The Bomb"] = 1
        combat.player.powers["The Bomb Damage"] = 40

        combat._check_outcome()

        self.assertEqual(combat.outcome, "UNDECIDED")

    def test_fire_breathing_start_turn_victory_still_applies_brutality_hp_loss_in_v2(self):
        combat = NativeCombatEnv(seed=10001, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        cultist = combat.monsters[0]
        cultist.current_hp = 10
        cultist.max_hp = 48
        combat.turn = 2
        combat.player.current_hp = 20
        combat.player.max_hp = 20
        combat.player.powers["Brutality"] = 1
        combat.player.powers["Fire Breathing"] = 10
        combat.hand = []
        combat.draw_pile = [
            make_card("Strike_R", uuid="strike-1"),
            make_card("Strike_R", uuid="strike-2"),
            make_card("Strike_R", uuid="strike-3"),
            make_card("Strike_R", uuid="strike-4"),
            make_card("Dazed", uuid="dazed"),
        ]
        combat.discard_pile = []

        combat.start_player_turn()

        self.assertEqual(combat.outcome, "PLAYER_VICTORY")
        self.assertEqual(combat.player.current_hp, 19)

    def test_fire_breathing_start_turn_draw_finishes_sundial_shuffle_before_victory_in_v2(self):
        combat = NativeCombatEnv(seed=100011, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        cultist = combat.monsters[0]
        cultist.current_hp = 6
        cultist.max_hp = 48
        combat.turn = 2
        combat.player.energy = 3
        combat.player.powers["Fire Breathing"] = 6
        combat.relics.append({"relic_id": "Sundial", "id": "Sundial", "name": "Sundial", "counter": 2, "price": 0, "tier": "UNCOMMON"})
        combat.hand = []
        combat.draw_pile = [
            make_card("Strike_R", uuid="strike-1"),
            make_card("Strike_R", uuid="strike-2"),
            make_card("Dazed", uuid="dazed"),
        ]
        combat.discard_pile = [
            make_card("Defend_R", uuid="discard-defend-1"),
            make_card("Defend_R", uuid="discard-defend-2"),
        ]

        combat.start_player_turn()

        self.assertEqual(combat.outcome, "PLAYER_VICTORY")
        self.assertEqual(combat.player.energy, 5)
        self.assertEqual(next(relic["counter"] for relic in combat.relics if relic["relic_id"] == "Sundial"), 0)

    def test_end_turn_feel_no_pain_juggernaut_hits_after_monster_block_clears_in_v2(self):
        combat = NativeCombatEnv(seed=10002, ascension_level=0, scheduled_encounter=["Sentry", "Sentry", "Sentry"], player=PlayerState())
        combat.player.powers["Feel No Pain"] = 3
        combat.player.powers["Juggernaut"] = 5
        combat.hand = [
            make_card("Clothesline", uuid="clothesline"),
            make_card("Dazed", uuid="dazed-1"),
            make_card("Dazed", uuid="dazed-2"),
            make_card("Dazed", uuid="dazed-3"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        for monster in combat.monsters:
            monster.block = 4
        combat.monsters[0].current_hp = 24
        combat.monsters[1].current_hp = 19
        combat.monsters[2].current_hp = 34

        class _SequenceRng:
            def __init__(self):
                self.values = [2, 1, 1]

            def random(self, upper):
                return self.values.pop(0) if self.values else 0

        combat.card_random_rng = _SequenceRng()

        combat.end_turn()

        self.assertEqual(
            [monster.current_hp for monster in combat.monsters],
            [24, 9, 29],
        )

    def test_secret_technique_opens_draw_pile_selection_when_multiple_skills_exist_in_v2(self):
        combat = NativeCombatEnv(seed=6, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Secret Technique", uuid="secret-technique")]
        combat.draw_pile = [
            make_card("Strike_R", uuid="strike"),
            make_card("Defend_R", uuid="defend"),
            make_card("Shrug It Off", uuid="shrug"),
            make_card("Bash", uuid="bash"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        class CountingRandom:
            def __init__(self):
                self.calls = []
            def random(self, upper):
                self.calls.append(upper)
                return 0

        combat.card_random_rng = CountingRandom()

        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "SECRET_TECHNIQUE")
        self.assertEqual([option["deck_index"] for option in combat.card_select_options], [1, 2])
        self.assertEqual([card.card_id for card in combat.exhaust_pile], [])
        self.assertEqual(combat.card_random_rng.calls, [0])

        combat._resolve_card_select({"kind": "card_select", "deck_index": 2})

        self.assertEqual([card.card_id for card in combat.hand], ["Shrug It Off"])
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Secret Technique"])

    def test_secret_weapon_fetches_single_attack_without_opening_selection_in_v2(self):
        combat = NativeCombatEnv(seed=8, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Secret Weapon", uuid="secret-weapon")]
        combat.draw_pile = [
            make_card("Defend_R", uuid="defend"),
            make_card("Bash", uuid="bash"),
            make_card("Inflame", uuid="inflame"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertIsNone(combat.card_select_context)
        self.assertEqual([card.card_id for card in combat.hand], ["Bash"])
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Secret Weapon"])

    def test_havoc_does_not_exhaust_unplayable_status_in_v2(self):
        combat = NativeCombatEnv(seed=7, ascension_level=0, player=PlayerState())
        combat.hand = []
        combat.draw_pile = [make_card("Injury", uuid="injury")]
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat._play_random_top_card()

        self.assertEqual(combat.draw_pile, [])
        self.assertEqual(combat.discard_pile, [])
        self.assertEqual(combat.exhaust_pile, [])

    def test_mark_of_pain_does_not_pollute_deck_in_v2(self):
        env = NativeRunEnv(seed=9, ascension_level=0)
        initial_wounds = sum(1 for card in env.deck if card.card_id == "Wound")

        env._obtain_relic({"relic_id": "Mark of Pain", "name": "Mark of Pain", "tier": "BOSS"})

        self.assertEqual(sum(1 for card in env.deck if card.card_id == "Wound"), initial_wounds)

    def test_ink_bottle_starts_tracking_from_zero_when_obtained_in_v2(self):
        env = NativeRunEnv(seed=9, ascension_level=0)

        env._obtain_relic({"relic_id": "Ink Bottle", "id": "Ink Bottle", "name": "Ink Bottle", "counter": -1, "tier": "UNCOMMON"})

        self.assertEqual(env._relic("Ink Bottle")["counter"], 0)

    def test_red_skull_grants_strength_on_bloodied_combat_start_in_v2(self):
        combat = NativeCombatEnv(seed=10, ascension_level=0, player=PlayerState(current_hp=30, max_hp=80))
        combat.relics.append({"relic_id": "Red Skull", "name": "Red Skull", "tier": "COMMON", "counter": -1})

        combat.start_combat()

        self.assertEqual(combat.player.power("Strength"), 3)

    def test_red_skull_updates_when_crossing_bloodied_threshold_in_v2(self):
        combat = NativeCombatEnv(seed=10, ascension_level=0, player=PlayerState(current_hp=50, max_hp=80))
        combat.relics.append({"relic_id": "Red Skull", "name": "Red Skull", "tier": "COMMON", "counter": -1})
        combat.start_combat()

        combat._lose_hp(10, from_attack=True)
        self.assertEqual(combat.player.power("Strength"), 3)

        combat._heal(20)
        self.assertEqual(combat.player.power("Strength"), 6)

    def test_red_skull_and_pantograph_stack_at_boss_combat_start_in_v2(self):
        combat = NativeCombatEnv(
            seed=10,
            ascension_level=0,
            floor=16,
            act=1,
            player=PlayerState(current_hp=33, max_hp=80),
            scheduled_encounter=["TheGuardian"],
        )
        combat.relics.append({"relic_id": "Red Skull", "name": "Red Skull", "tier": "COMMON", "counter": -1})
        combat.relics.append({"relic_id": "Pantograph", "name": "Pantograph", "tier": "UNCOMMON", "counter": -1})

        combat.start_combat()

        self.assertEqual(combat.player.current_hp, 58)
        self.assertEqual(combat.player.power("Strength"), 6)

    def test_bandage_up_heal_stacks_red_skull_like_lightspeed_in_v2(self):
        combat = NativeCombatEnv(seed=10, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        combat.relics.append({"relic_id": "Red Skull", "name": "Red Skull", "tier": "COMMON", "counter": -1})
        combat.start_combat()
        combat.hand = [make_card("Bandage Up", uuid="bandage-up")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 44)
        self.assertEqual(combat.player.power("Strength"), 6)

    def test_rupture_only_triggers_on_self_damage_in_v2(self):
        combat = NativeCombatEnv(seed=10, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        combat.player.powers["Rupture"] = 1

        combat._lose_hp(3, from_attack=True)
        self.assertEqual(combat.player.power("Strength"), 0)

        combat._lose_hp(3, self_damage=True)
        self.assertEqual(combat.player.power("Strength"), 1)

    def test_end_turn_burn_triggers_rupture_in_v2(self):
        combat = NativeCombatEnv(seed=10, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        combat.player.powers["Rupture"] = 2
        combat.hand = [make_card("Burn", uuid="burn")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.monsters = [make_monster("JawWorm", StsRandom(11), ascension=0)]

        combat.end_turn()

        self.assertEqual(combat.player.power("Strength"), 2)

    def test_taskmaster_scouring_whip_adds_wounds_in_v2(self):
        combat = NativeCombatEnv(seed=11, ascension_level=0, scheduled_encounter=["SlaverBlue", "Taskmaster", "SlaverRed"], player=PlayerState())
        combat.discard_pile = []
        taskmaster = next(monster for monster in combat.monsters if monster.monster_id == "Taskmaster")
        taskmaster.move = "TASKMASTER_SCOURING_WHIP"
        taskmaster.intent = "ATTACK_DEBUFF"
        combat.monsters = [taskmaster]
        combat.player.current_hp = 80
        combat.player.block = 0

        combat.end_turn()

        self.assertIn("Wound", [card.card_id for card in combat.discard_pile])

    def test_impatience_matches_lightspeed_current_behavior_in_v2(self):
        combat = NativeCombatEnv(seed=13, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Strike_R", uuid="strike"), make_card("Impatience", uuid="impatience")]
        combat.draw_pile = [make_card("Bash", uuid="bash"), make_card("Strike_R", uuid="strike-2")]
        combat.discard_pile = []
        combat.player.energy = 1

        combat.play_card(1, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Strike_R", "Strike_R", "Bash"])

    def test_impatience_moves_to_discard_before_evolve_followup_shuffle_in_v2(self):
        combat = NativeCombatEnv(seed=241202561815090497, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Strike_R", uuid="hand-strike"),
            make_card("Dazed", uuid="hand-dazed-left"),
            make_card("Impatience", uuid="impatience"),
            make_card("Dazed", uuid="hand-dazed-right"),
            make_card("Defend_R", uuid="hand-defend"),
        ]
        combat.draw_pile = [
            make_card("Strike_R", uuid="draw-strike"),
            make_card("Defend_R", uuid="draw-defend"),
            make_card("Dazed", uuid="draw-dazed"),
        ]
        combat.discard_pile = [
            make_card("Armaments", uuid="discard-armaments"),
            make_card("Cleave", uuid="discard-cleave"),
            make_card("Defend_R", uuid="discard-defend-left"),
            make_card("Strike_R", uuid="discard-strike-left"),
            make_card("Strike_R", uuid="discard-strike-right"),
            make_card("Dazed", uuid="discard-dazed-0"),
            make_card("Dazed", uuid="discard-dazed-1"),
            make_card("Dazed", uuid="discard-dazed-2"),
            make_card("Dazed", uuid="discard-dazed-3"),
            make_card("Bash", uuid="discard-bash"),
            make_card("Defend_R", uuid="discard-defend-right"),
        ]
        combat.exhaust_pile = [
            make_card("Bandage Up", uuid="exhaust-bandage"),
            make_card("Dazed", uuid="exhaust-dazed-0"),
            make_card("Dazed", uuid="exhaust-dazed-1"),
            make_card("Dazed", uuid="exhaust-dazed-2"),
            make_card("Dazed", uuid="exhaust-dazed-3"),
            make_card("Dazed", uuid="exhaust-dazed-4"),
            make_card("Dazed", uuid="exhaust-dazed-5"),
            make_card("Dazed", uuid="exhaust-dazed-6"),
            make_card("Dazed", uuid="exhaust-dazed-7"),
            make_card("Dazed", uuid="exhaust-dazed-8"),
        ]
        combat.player.powers["Evolve"] = 1
        combat.player.energy = 1
        saw_impatience_in_discard_during_draw = False
        original_draw_cards = combat.draw_cards

        def _wrapped_draw_cards(*args, **kwargs):
            nonlocal saw_impatience_in_discard_during_draw
            saw_impatience_in_discard_during_draw = any(card.card_id == "Impatience" for card in combat.discard_pile)
            return original_draw_cards(*args, **kwargs)

        combat.draw_cards = _wrapped_draw_cards

        combat.play_card(2, 0)

        self.assertTrue(saw_impatience_in_discard_during_draw)

    def test_run_hp_loss_uses_fairy_potion_before_game_over(self):
        env = NativeRunEnv(seed=15, ascension_level=0)
        env.player.current_hp = 14
        env.player.max_hp = 80
        env.potions = [make_potion("FairyPotion"), make_potion("Distilled Chaos"), make_potion("Fire Potion")]

        env._lose_run_hp(16)

        self.assertNotEqual(env.phase, "GAME_OVER")
        self.assertEqual(env.player.current_hp, 24)
        self.assertEqual(env.potions[0].potion_id, "Potion Slot")

    def test_combat_fairy_potion_heal_uses_magic_flower_in_v2(self):
        combat = NativeCombatEnv(seed=15, ascension_level=0, player=PlayerState(current_hp=1, max_hp=80))
        combat.relics.append({"relic_id": "Magic Flower", "id": "Magic Flower", "name": "Magic Flower", "counter": -1, "price": 0, "tier": "RARE"})
        combat.potions[0] = make_potion("FairyPotion")

        combat._lose_hp(5, from_attack=True)

        self.assertEqual(combat.player.current_hp, 36)
        self.assertEqual(combat.potions[0].potion_id, "Potion Slot")

    def test_combat_fairy_potion_heal_uses_sacred_bark_in_v2(self):
        combat = NativeCombatEnv(seed=16, ascension_level=0, player=PlayerState(current_hp=1, max_hp=80))
        combat.relics.append({"relic_id": "Sacred Bark", "id": "Sacred Bark", "name": "Sacred Bark", "counter": -1, "price": 0, "tier": "BOSS"})
        combat.potions[0] = make_potion("FairyPotion")

        combat._lose_hp(5, from_attack=True)

        self.assertEqual(combat.player.current_hp, 48)
        self.assertEqual(combat.potions[0].potion_id, "Potion Slot")

    def test_counter_damage_fairy_potion_heal_uses_sacred_bark_in_v2(self):
        combat = NativeCombatEnv(seed=17, ascension_level=0, player=PlayerState(current_hp=1, max_hp=80))
        combat.relics.append({"relic_id": "Sacred Bark", "id": "Sacred Bark", "name": "Sacred Bark", "counter": -1, "price": 0, "tier": "BOSS"})
        combat.potions[0] = make_potion("FairyPotion")

        combat._take_counter_damage(5)

        self.assertEqual(combat.player.current_hp, 48)
        self.assertEqual(combat.potions[0].potion_id, "Potion Slot")

    def test_deferred_juggernaut_single_target_waits_until_flush_in_v2(self):
        combat = NativeCombatEnv(seed=18, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState(current_hp=40, max_hp=80))
        combat.player.powers["Juggernaut"] = 5
        before = combat.card_random_rng.counter

        combat._gain_block(3, defer_juggernaut=True, apply_block_modifiers=False)

        self.assertEqual(combat.card_random_rng.counter, before)
        self.assertEqual(combat.pending_juggernaut_damage, 5)
        self.assertEqual(combat.monsters[0].current_hp, combat.monsters[0].max_hp)

        combat._resolve_pending_juggernaut_damage()

        self.assertEqual(combat.card_random_rng.counter, before + 1)
        self.assertEqual(combat.pending_juggernaut_damage, 0)
        self.assertEqual(combat.monsters[0].current_hp, combat.monsters[0].max_hp - 5)

    def test_skill_block_hex_inserts_dazed_before_deferred_juggernaut_rng_in_v2(self):
        combat = NativeCombatEnv(seed=19, ascension_level=0, scheduled_encounter=["Chosen"], player=PlayerState(current_hp=50, max_hp=80))
        combat.player.powers["Juggernaut"] = 5
        combat.player.powers["Hex"] = 1
        combat.hand = [make_card("Defend_R", uuid="defend")]
        combat.draw_pile = [
            make_card("Defend_R", uuid="draw-0"),
            make_card("Defend_R", uuid="draw-1"),
            make_card("Cleave", uuid="draw-2"),
            make_card("Defend_R", uuid="draw-3"),
            make_card("Hemokinesis", uuid="draw-4"),
        ]
        combat.discard_pile = []

        combat.play_card(0, 0)

        self.assertEqual(
            [card.card_id for card in combat.draw_pile],
            ["Dazed", "Defend_R", "Defend_R", "Cleave", "Defend_R", "Hemokinesis"],
        )

    def test_combat_lizard_tail_does_not_revive_from_attack_damage_in_v2(self):
        combat = NativeCombatEnv(
            seed=15,
            ascension_level=0,
            player=PlayerState(current_hp=7, max_hp=80),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Lizard Tail", "id": "Lizard Tail", "name": "Lizard Tail", "counter": -1, "price": 0, "tier": "RARE"},
            ],
        )

        combat._lose_hp(18, from_attack=True)

        self.assertEqual(combat.player.current_hp, 0)
        self.assertEqual(combat.outcome, "PLAYER_LOSS")
        relic = next(relic for relic in combat.relics if relic.get("relic_id") == "Lizard Tail")
        self.assertEqual(relic.get("counter"), -1)

    def test_combat_lizard_tail_does_not_revive_from_non_attack_damage_in_v2(self):
        combat = NativeCombatEnv(
            seed=16,
            ascension_level=0,
            player=PlayerState(current_hp=7, max_hp=80),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Lizard Tail", "id": "Lizard Tail", "name": "Lizard Tail", "counter": -1, "price": 0, "tier": "RARE"},
            ],
        )

        combat._take_non_attack_damage(18)

        self.assertEqual(combat.player.current_hp, 0)
        self.assertEqual(combat.outcome, "PLAYER_LOSS")
        relic = next(relic for relic in combat.relics if relic.get("relic_id") == "Lizard Tail")
        self.assertEqual(relic.get("counter"), -1)

    def test_shop_potions_remain_buyable_when_potion_slots_are_full(self):
        env = NativeRunEnv(seed=17, ascension_level=0)
        env.phase = "SHOP"
        env.gold = 100
        env.potions = [make_potion("Fire Potion"), make_potion("Block Potion"), make_potion("Weak Potion")]
        env.shop_items = [
            {"kind": "shop", "name": "Power Potion", "item_kind": "potion", "item_id": "Power Potion", "potion_id": "Power Potion", "price": 51, "choice_index": 0},
            {"kind": "shop", "name": "LEAVE", "item_kind": "leave", "price": 0, "choice_index": 1},
        ]

        actions = env.legal_actions()

        self.assertEqual([action["item_kind"] for action in actions], ["potion", "leave"])

        env.step(actions[0])

        self.assertEqual(env.gold, 49)
        self.assertEqual([potion.potion_id for potion in env.potions], ["Fire Potion", "Block Potion", "Weak Potion"])
        self.assertEqual([item["item_kind"] for item in env.shop_items], ["leave"])

    def test_run_hp_loss_respects_tungsten_rod_in_v2(self):
        env = NativeRunEnv(seed=18, ascension_level=0)
        env.player.current_hp = 18
        env.relics.append({"relic_id": "Tungsten Rod", "name": "Tungsten Rod", "tier": "RARE", "counter": -1})

        env._lose_run_hp(8)

        self.assertEqual(env.player.current_hp, 11)

    def test_shining_light_hp_loss_rounds_up_in_v2(self):
        env = NativeRunEnv(seed=18, ascension_level=0)
        env.phase = "EVENT"
        env.player.max_hp = 94
        env.player.current_hp = 54
        env.event_options = [
            {
                "kind": "event",
                "event_id": "Shining Light",
                "name": "Entered Light",
                "label": "Entered Light",
                "choice_index": 0,
            }
        ]
        env.deck = [make_card("Strike_R", uuid=f"strike-{i}") for i in range(5)]

        env.step(env.event_options[0])

        self.assertEqual(env.player.current_hp, 35)

    def test_obtained_lizard_tail_starts_spent_in_v2(self):
        env = NativeRunEnv(seed=19, ascension_level=0)

        env._obtain_relic(make_relic("Lizard Tail"))

        self.assertEqual(env._relic("Lizard Tail").get("counter"), 0)

    def test_face_trader_touched_rounds_up_hp_loss_and_can_end_run_in_v2(self):
        env = NativeRunEnv(seed=18, ascension_level=0)
        env.phase = "EVENT"
        env.player.max_hp = 80
        env.player.current_hp = 8
        env.gold = 86
        env.event_options = [
            {
                "kind": "event",
                "event_id": "Face Trader",
                "name": "Touched",
                "label": "Touched",
                "choice_index": 0,
            }
        ]

        env.step(env.event_options[0])

        self.assertEqual(env.phase, "GAME_OVER")
        self.assertEqual(env.player.current_hp, 0)
        self.assertEqual(env.gold, 161)

    def test_wing_statue_card_removal_does_not_open_select_after_death_in_v2(self):
        env = NativeRunEnv(seed=181, ascension_level=0)
        env.phase = "EVENT"
        env.player.current_hp = 7
        env.deck = [make_card("Strike_R", uuid=f"strike-{i}") for i in range(5)]
        env.event_options = [
            {
                "kind": "event",
                "event_id": "Wing Statue",
                "name": "Card Removal",
                "label": "Card Removal",
                "choice_index": 0,
            }
        ]

        env.step(env.event_options[0])

        self.assertEqual(env.phase, "GAME_OVER")
        self.assertEqual(env.player.current_hp, 0)
        self.assertIsNone(env.card_select_context)

    def test_wing_statue_offers_gold_option_for_dramatic_entrance_in_v2(self):
        env = NativeRunEnv(seed=181, ascension_level=0)
        env._draw_event_id = lambda: "Wing Statue"
        env.deck = [make_card("Dramatic Entrance", upgrades=1, uuid="dramatic-0")]

        env._enter_event()

        self.assertEqual(
            [(item["choice_index"], item["name"]) for item in env.event_options],
            [(0, "Card Removal"), (1, "Gained Gold"), (2, "Ignored")],
        )

    def test_library_preview_applies_molten_egg_to_attack_options_in_v2(self):
        env = NativeRunEnv(seed=182, ascension_level=0)
        env.relics.append({"relic_id": "Molten Egg", "id": "Molten Egg", "name": "Molten Egg", "counter": -1, "price": 0, "tier": "UNCOMMON"})

        env._open_library_card_select()

        attack_cards = [card for card in env.card_select_generated_cards if card.card_def.card_type == "ATTACK"]
        self.assertTrue(attack_cards)
        self.assertTrue(all(card.upgrades >= 1 for card in attack_cards))

    def test_library_generation_does_not_change_card_rarity_factor_in_v2(self):
        env = NativeRunEnv(seed=183, ascension_level=0)
        env.card_rarity_factor = 4

        rarities = ["COMMON", "UNCOMMON", "RARE"] + ["COMMON"] * 40
        env._roll_card_rarity = lambda room=None: rarities.pop(0)

        common_pool = [card for card in ironclad_type_rarity_card_pool("ATTACK", "COMMON")] + [
            make_card("Armaments"),
            make_card("Warcry"),
            make_card("Flex"),
            make_card("Shrug It Off"),
            make_card("True Grit"),
            make_card("Havoc"),
        ]
        uncommon_pool = [CARD_LIBRARY["Battle Trance"]]
        rare_pool = [CARD_LIBRARY["Feed"]]

        def _pool(*, rarity=None, card_type=None, exclude_ids=None):
            if rarity == "COMMON":
                return list(common_pool)
            if rarity == "UNCOMMON":
                return list(uncommon_pool)
            if rarity == "RARE":
                return list(rare_pool)
            return list(common_pool + uncommon_pool + rare_pool)

        env._ironclad_card_pool = _pool

        env._open_library_card_select()

        self.assertEqual(env.card_rarity_factor, 4)

    def test_library_duplicate_rerolls_rarity_in_v2(self):
        env = NativeRunEnv(seed=184, ascension_level=0)

        rarity_rolls = ["COMMON"] * 20 + ["UNCOMMON"]
        env._roll_card_rarity = lambda room=None: rarity_rolls.pop(0)

        choice_cards = [
            CARD_LIBRARY["Flex"],
            CARD_LIBRARY["Headbutt"],
            CARD_LIBRARY["Wild Strike"],
            CARD_LIBRARY["Shockwave"],
            CARD_LIBRARY["Hemokinesis"],
            CARD_LIBRARY["Iron Wave"],
            CARD_LIBRARY["Twin Strike"],
            CARD_LIBRARY["Thunderclap"],
            CARD_LIBRARY["Rampage"],
            CARD_LIBRARY["Pommel Strike"],
            CARD_LIBRARY["Fire Breathing"],
            CARD_LIBRARY["Dark Embrace"],
            CARD_LIBRARY["Clash"],
            CARD_LIBRARY["Shrug It Off"],
            CARD_LIBRARY["Cleave"],
            CARD_LIBRARY["Dropkick"],
            CARD_LIBRARY["Body Slam"],
            CARD_LIBRARY["True Grit"],
            CARD_LIBRARY["Warcry"],
            CARD_LIBRARY["Warcry"],
            CARD_LIBRARY["Battle Trance"],
        ]

        env.randoms.card.choice = lambda pool: choice_cards.pop(0)
        env._ironclad_card_pool = lambda *, rarity=None, card_type=None, exclude_ids=None: [CARD_LIBRARY["Warcry"], CARD_LIBRARY["Battle Trance"]]

        env._open_library_card_select()

        self.assertEqual(env.card_select_generated_cards[0].card_id, "Battle Trance")

    def test_event_combat_gold_stays_in_reward_screen_until_claimed(self):
        env = NativeRunEnv(seed=19, ascension_level=0)
        env.phase = "COMBAT"
        env.current_node_symbol = "EVENT_COMBAT"
        env.gold = 130
        env.pending_event_gold = 25
        env.pending_event_relic_id = "Odd Mushroom"
        combat = NativeCombatEnv(seed=19, ascension_level=0, player=env.player)
        combat.outcome = "PLAYER_VICTORY"
        combat.gold = 130
        combat.gold_gain = 0
        combat.reward_gold_bonus = 0
        env.combat = combat

        env.step({})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(env.gold, 130)
        self.assertEqual(env.reward_gold_piles, [25])

    def test_the_joust_uses_misc_rng_and_murderer_pays_100_in_v2(self):
        env = NativeRunEnv(seed=2205, ascension_level=0)
        env.phase = "EVENT"
        env.current_event_id = "The Joust"
        env.event_options = [
            {"kind": "event", "event_id": "The Joust", "name": "Murderer", "choice_index": 0},
            {"kind": "event", "event_id": "The Joust", "name": "Owner", "choice_index": 1},
        ]
        env.gold = 100
        original_event_random = env.randoms.event.random
        original_misc_random_boolean = env.randoms.misc.random_boolean
        try:
            env.randoms.event.random = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("event rng should not be used for The Joust"))
            env.randoms.misc.random_boolean = lambda chance: False
            env.step(env.event_options[0])
        finally:
            env.randoms.event.random = original_event_random
            env.randoms.misc.random_boolean = original_misc_random_boolean

        self.assertEqual(env.gold, 150)
        self.assertEqual(env.phase, "MAP")

    def test_campfire_dig_opens_reward_screen(self):
        env = NativeRunEnv(seed=21, ascension_level=0)
        env.phase = "CAMPFIRE"
        env.campfire_options = [{"kind": "campfire", "name": "DIG", "choice_index": 0}]
        env.relics.append({"relic_id": "Shovel", "name": "Shovel", "tier": "RARE", "counter": -1})

        env.step(env.campfire_options[0])

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(env.reward_context, "REST")
        self.assertEqual(len(env.reward_relics), 1)

    def test_campfire_peace_pipe_uses_toke_name(self):
        env = NativeRunEnv(seed=22, ascension_level=0)
        env.relics.append({"relic_id": "Peace Pipe", "name": "Peace Pipe", "tier": "SHOP", "counter": -1})

        env._enter_campfire()

        self.assertIn("TOKE", [option["name"] for option in env.campfire_options])

    def test_eternal_feather_heals_on_entering_campfire_in_v2(self):
        env = NativeRunEnv(seed=2201, ascension_level=0)
        env.player.max_hp = 80
        env.player.current_hp = 40
        env.deck = [make_card("Strike_R", uuid=f"deck-{i}") for i in range(10)]
        env.relics.append({"relic_id": "Eternal Feather", "name": "Eternal Feather", "tier": "UNCOMMON", "counter": -1})

        env._enter_campfire()

        self.assertEqual(env.player.current_hp, 46)

    def test_eternal_feather_does_not_double_heal_on_rest_in_v2(self):
        env = NativeRunEnv(seed=2202, ascension_level=0)
        env.player.max_hp = 80
        env.player.current_hp = 40
        env.deck = [make_card("Strike_R", uuid=f"deck-{i}") for i in range(10)]
        env.relics.append({"relic_id": "Eternal Feather", "name": "Eternal Feather", "tier": "UNCOMMON", "counter": -1})

        env._enter_campfire()
        env.step({"kind": "campfire", "name": "REST", "choice_index": 0})

        self.assertEqual(env.player.current_hp, 70)

    def test_card_reward_does_not_offer_potion_with_sozu_in_v2(self):
        env = NativeRunEnv(seed=2203, ascension_level=0)
        env.current_node_symbol = "M"
        env.floor = 13
        env.potion_chance_counter = 0
        env.potions = [
            make_potion("Essence of Steel"),
            make_potion("Blood Potion"),
            make_potion("Power Potion"),
        ]
        env.relics.append({"relic_id": "Sozu", "name": "Sozu", "tier": "BOSS", "counter": -1})

        original_roll_potion = helpers_run_module.roll_potion
        original_random = env.randoms.potion.random
        try:
            helpers_run_module.roll_potion = lambda rng: make_potion("Ancient Potion")
            env.randoms.potion.random = lambda upper=99: 0
            env._enter_card_reward(include_base_gold=False)
        finally:
            helpers_run_module.roll_potion = original_roll_potion
            env.randoms.potion.random = original_random

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(env.reward_potions, [])
        self.assertFalse(any(action.get("kind") == "reward_potion" for action in env.legal_actions()))

    def test_card_reward_still_offers_potion_when_slots_full_in_v2(self):
        env = NativeRunEnv(seed=2203, ascension_level=0)
        env.current_node_symbol = "M"
        env.floor = 13
        env.potion_chance_counter = 0
        env.potions = [
            make_potion("Essence of Steel"),
            make_potion("Blood Potion"),
            make_potion("Power Potion"),
        ]

        original_roll_potion = helpers_run_module.roll_potion
        original_random = env.randoms.potion.random
        try:
            helpers_run_module.roll_potion = lambda rng: make_potion("Ancient Potion")
            env.randoms.potion.random = lambda upper=99: 0
            env._enter_card_reward(include_base_gold=False)
        finally:
            helpers_run_module.roll_potion = original_roll_potion
            env.randoms.potion.random = original_random

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual([p.potion_id for p in env.reward_potions], ["Ancient Potion"])

        before_owned = [p.potion_id for p in env.potions]
        potion_action = next(action for action in env.legal_actions() if action.get("kind") == "reward_potion")
        env.step(potion_action)

        self.assertEqual([p.potion_id for p in env.potions], before_owned)
        self.assertEqual(env.reward_potions, [])

    def test_card_reward_counts_bonus_gold_as_single_reward_entry_for_potion_rolls_in_v2(self):
        env = NativeRunEnv(seed=2204, ascension_level=0)
        env.current_node_symbol = "M"
        env.floor = 13
        env.potion_chance_counter = 0
        env.relics.append({"relic_id": "Prayer Wheel", "name": "Prayer Wheel", "tier": "RARE", "counter": -1})

        original_roll_potion = helpers_run_module.roll_potion
        original_random = env.randoms.potion.random
        original_randint = env.randoms.treasure.randint
        try:
            helpers_run_module.roll_potion = lambda rng: make_potion("Ancient Potion")
            env.randoms.potion.random = lambda upper=99: 0
            env.randoms.treasure.randint = lambda low, high: 11
            env._enter_card_reward(extra_gold_rewards=[30], include_base_gold=True)
        finally:
            helpers_run_module.roll_potion = original_roll_potion
            env.randoms.potion.random = original_random
            env.randoms.treasure.randint = original_randint

        self.assertEqual(env.reward_gold_piles, [11, 30])
        self.assertEqual(env.reward_gold, 41)
        self.assertEqual(len(env.reward_card_bundles), 2)
        self.assertEqual(env.reward_potions, [])

    def test_card_reward_does_not_roll_potion_with_many_reward_entries_in_v2(self):
        env = NativeRunEnv(seed=2205, ascension_level=0)
        env.current_node_symbol = "E"
        env.floor = 24
        env.potion_chance_counter = 0
        env.relics.append({"relic_id": "Black Star", "name": "Black Star", "tier": "BOSS", "counter": -1})

        original_roll_potion = helpers_run_module.roll_potion
        original_random = env.randoms.potion.random
        try:
            helpers_run_module.roll_potion = lambda rng: make_potion("Block Potion")
            env.randoms.potion.random = lambda upper=99: 0
            env._enter_card_reward(include_base_gold=True)
        finally:
            helpers_run_module.roll_potion = original_roll_potion
            env.randoms.potion.random = original_random

        self.assertEqual(len(env.reward_card_bundles), 1)
        self.assertEqual(len(env.reward_relics), 2)
        self.assertEqual(len(env.reward_gold_piles), 1)
        self.assertEqual(env.reward_potions, [])

    def test_elite_card_reward_does_not_roll_potion_when_gold_relic_and_card_exist_in_v2(self):
        env = NativeRunEnv(seed=2206, ascension_level=0)
        env.current_node_symbol = "E"
        env.floor = 13
        env.potion_chance_counter = 0

        original_roll_relic = helpers_run_module.RunHelpersMixin._roll_relic
        original_roll_potion = helpers_run_module.roll_potion
        original_random = env.randoms.potion.random
        original_randint = env.randoms.treasure.randint
        try:
            helpers_run_module.RunHelpersMixin._roll_relic = lambda self, elite=False, **kwargs: {
                "relic_id": "Toy Ornithopter",
                "name": "Toy Ornithopter",
                "tier": "COMMON",
                "counter": -1,
            }
            helpers_run_module.roll_potion = lambda rng: make_potion("Liquid Bronze")
            env.randoms.potion.random = lambda upper=99: 0
            env.randoms.treasure.randint = lambda low, high: 31
            env._enter_card_reward(include_base_gold=True)
        finally:
            helpers_run_module.RunHelpersMixin._roll_relic = original_roll_relic
            helpers_run_module.roll_potion = original_roll_potion
            env.randoms.potion.random = original_random
            env.randoms.treasure.randint = original_randint

        self.assertEqual(len(env.reward_gold_piles), 1)
        self.assertEqual(len(env.reward_card_bundles), 1)
        self.assertEqual([relic["relic_id"] for relic in env.reward_relics], ["Toy Ornithopter"])
        self.assertEqual(env.reward_potions, [])

    def test_shop_type_rarity_pool_matches_lightspeed_order(self):
        common_attacks = [card.card_id for card in ironclad_type_rarity_card_pool("ATTACK", "COMMON")]
        common_skills = [card.card_id for card in ironclad_type_rarity_card_pool("SKILL", "COMMON")]
        uncommon_powers = [card.card_id for card in ironclad_type_rarity_card_pool("POWER", "UNCOMMON")]

        self.assertEqual(common_attacks[:5], ["Anger", "Body Slam", "Clash", "Cleave", "Clothesline"])
        self.assertEqual(common_attacks[-3:], ["Thunderclap", "Twin Strike", "Wild Strike"])
        self.assertEqual(common_skills, ["Armaments", "Flex", "Havoc", "Shrug It Off", "True Grit", "Warcry"])
        self.assertEqual(uncommon_powers, ["Combust", "Dark Embrace", "Evolve", "Feel No Pain", "Fire Breathing", "Inflame", "Metallicize", "Rupture"])

    def test_card_reward_preview_applies_egg_upgrades_like_lightspeed_in_v2(self):
        seed = 19
        ls_env = sts.ModelDrivenEnv(seed, 0)
        native = NativeRunEnv(seed=seed, ascension_level=0, enable_neow=True)

        for _ in range(400):
            ls_in_battle = bool(ls_env.in_battle)
            if not ls_in_battle and native.phase == "CARD_REWARD" and native.floor == 8:
                ls_actions = [dict(action) for action in sts.get_external_actions(ls_env.game_context)]
                native_actions = [dict(action) for action in native.legal_actions()]
                ls_rewards = sorted(
                    (action["card_id"], int(action.get("upgrades", 0) or 0))
                    for action in ls_actions
                    if action.get("kind") == "card_reward"
                )
                native_rewards = sorted(
                    (action["card_id"], int(action["card"].get("upgrades", 0) or 0))
                    for action in native_actions
                    if action.get("kind") == "card_reward"
                )
                self.assertEqual(
                    native_rewards,
                    ls_rewards,
                    "native card reward previews should reflect egg relic upgrades like lightspeed",
                )
                return

            if ls_in_battle:
                ls_state = sts.get_battle_state(ls_env)
                ls_actions = [dict(action) for action in sts.get_battle_actions(ls_env)]
                choice_sig = _choice_list_signature(ls_actions, ls_state)[0]
                ls_chosen = _pick_action_by_signature(ls_actions, choice_sig, ls_state)
                native_chosen = _pick_action_by_signature(
                    [dict(action) for action in native.legal_actions()],
                    choice_sig,
                    native.state(),
                )
                sts.execute_battle_action_bits(ls_env, int(ls_chosen["bits"]))
                native.step(native_chosen)
            else:
                ls_actions = [dict(action) for action in sts.get_external_actions(ls_env.game_context)]
                choice_sig = _choice_list_signature(ls_actions, None)[0]
                ls_chosen = _pick_action_by_signature(ls_actions, choice_sig, None)
                native_chosen = _pick_action_by_signature(
                    [dict(action) for action in native.legal_actions()],
                    choice_sig,
                    None,
                )
                sts.execute_action_bits(ls_env, int(ls_chosen["bits"]))
                native.step(native_chosen)

        self.fail("did not reach the floor 8 card reward for the egg preview regression seed")

    def test_searing_blow_reward_upgrade_matches_lightspeed_current_bug_in_v2(self):
        seed = 5468525756682406891
        ls_env = sts.ModelDrivenEnv(seed, 0)
        native = NativeRunEnv(seed=seed, ascension_level=0, enable_neow=True)

        for _ in range(400):
            ls_in_battle = bool(ls_env.in_battle)
            if not ls_in_battle and native.phase == "CARD_REWARD" and native.floor == 20:
                ls_actions = [dict(action) for action in sts.get_external_actions(ls_env.game_context)]
                native_actions = [dict(action) for action in native.legal_actions()]
                ls_rewards = sorted(
                    (action["card_id"], int(action.get("upgrades", 0) or 0))
                    for action in ls_actions
                    if action.get("kind") == "card_reward"
                )
                native_rewards = sorted(
                    (action["card_id"], int(action["card"].get("upgrades", 0) or 0))
                    for action in native_actions
                    if action.get("kind") == "card_reward"
                )
                self.assertEqual(
                    native_rewards,
                    ls_rewards,
                    "native Searing Blow reward preview should match lightspeed's current upgrade quirk",
                )
                return

            if ls_in_battle:
                ls_state = sts.get_battle_state(ls_env)
                ls_actions = [dict(action) for action in sts.get_battle_actions(ls_env)]
                choice_sig = _choice_list_signature(ls_actions, ls_state)[0]
                ls_chosen = _pick_action_by_signature(ls_actions, choice_sig, ls_state)
                native_chosen = _pick_action_by_signature(
                    [dict(action) for action in native.legal_actions()],
                    choice_sig,
                    native.state(),
                )
                sts.execute_battle_action_bits(ls_env, int(ls_chosen["bits"]))
                native.step(native_chosen)
            else:
                ls_actions = [dict(action) for action in sts.get_external_actions(ls_env.game_context)]
                choice_sig = _choice_list_signature(ls_actions, None)[0]
                ls_chosen = _pick_action_by_signature(ls_actions, choice_sig, None)
                native_chosen = _pick_action_by_signature(
                    [dict(action) for action in native.legal_actions()],
                    choice_sig,
                    None,
                )
                sts.execute_action_bits(ls_env, int(ls_chosen["bits"]))
                native.step(native_chosen)

        self.fail("did not reach the floor 20 card reward for the Searing Blow reward regression seed")

    def test_make_deck_card_preserves_searing_blow_upgrade_counter_in_v2(self):
        env = NativeRunEnv(seed=1, ascension_level=0, enable_neow=True)
        card = env._make_deck_card("Searing Blow", upgrades=2, uuid="searing-blow-test")

        self.assertEqual(card.upgrades, 2)
        self.assertEqual(card.misc, 2)

    def test_courier_restock_treats_hand_of_greed_as_colorless_in_v2(self):
        env = NativeRunEnv(seed=1718, ascension_level=0, start_on_map=True)
        env.relics.append(make_relic("The Courier"))

        replacement = env._restock_shop_item(
            "card",
            {
                "kind": "shop",
                "name": "Hand of Greed",
                "item_kind": "card",
                "item_id": "HandOfGreed",
                "price": 130,
                "card": {"card_id": "HandOfGreed"},
            },
        )

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement["item_kind"], "card")
        colorless_ids = {card_id.lower().replace(" ", "") for card_id in COLORLESS_CARD_IDS}
        self.assertIn(str(replacement["item_id"]).lower().replace(" ", ""), colorless_ids)

    def test_blood_for_blood_upgrade_resets_misc_but_later_damage_reduces_combat_cost_in_v2(self):
        combat = NativeCombatEnv(seed=1719, ascension_level=0, player=PlayerState())
        bfb = make_card("Blood for Blood", uuid="bfb-upgrade")
        bfb.misc = 1
        bfb.cost_for_turn = 3
        combat.hand = [bfb]

        combat._upgrade_combat_card(bfb)
        bfb.cost_for_combat = 1
        bfb.cost_for_turn = 1
        combat._on_player_took_damage_cards()
        combat._on_player_took_damage_cards()

        self.assertEqual(bfb.upgrades, 1)
        self.assertEqual(bfb.misc, 2)
        self.assertEqual(bfb.cost_for_turn, 0)
        self.assertEqual(bfb.cost_for_combat, 0)

    def test_shining_light_armaments_searing_blow_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3349817218646675354

        result = compare_seed(seed, 0, 120, backend="v2")

        self.assertTrue(result.get("match"), result)

    def test_shop_potion_prices_match_lightspeed_floor10_regression_seed_in_v2(self):
        seed = 8866187513371018371
        ls_env = sts.ModelDrivenEnv(seed, 0)
        native = NativeRunEnv(seed=seed, ascension_level=0, enable_neow=True)

        for _ in range(250):
            ls_in_battle = bool(ls_env.in_battle)
            if not ls_in_battle and native.phase == "SHOP" and native.floor == 10:
                ls_actions = [dict(action) for action in sts.get_external_actions(ls_env.game_context)]
                native_actions = [dict(action) for action in native.legal_actions()]
                ls_potions = sorted(
                    (str(action["item_id"]), int(action["price"]))
                    for action in ls_actions
                    if action.get("kind") == "shop" and action.get("item_kind") == "potion"
                )
                native_potions = sorted(
                    (str(action["potion_id"]), int(action["price"]))
                    for action in native_actions
                    if action.get("kind") == "shop" and action.get("item_kind") == "potion"
                )
                self.assertEqual(
                    native_potions,
                    ls_potions,
                    "native shop potion prices should match lightspeed for the floor 10 regression seed",
                )
                return

            if ls_in_battle:
                ls_state = sts.get_battle_state(ls_env)
                ls_actions = [dict(action) for action in sts.get_battle_actions(ls_env)]
                choice_sig = _choice_list_signature(ls_actions, ls_state)[0]
                ls_chosen = _pick_action_by_signature(ls_actions, choice_sig, ls_state)
                native_chosen = _pick_action_by_signature(
                    [dict(action) for action in native.legal_actions()],
                    choice_sig,
                    native.state(),
                )
                sts.execute_battle_action_bits(ls_env, int(ls_chosen["bits"]))
                native.step(native_chosen)
            else:
                ls_actions = [dict(action) for action in sts.get_external_actions(ls_env.game_context)]
                choice_sig = _choice_list_signature(ls_actions, None)[0]
                ls_chosen = _pick_action_by_signature(ls_actions, choice_sig, None)
                native_chosen = _pick_action_by_signature(
                    [dict(action) for action in native.legal_actions()],
                    choice_sig,
                    None,
                )
                sts.execute_action_bits(ls_env, int(ls_chosen["bits"]))
                native.step(native_chosen)

        self.fail("did not reach the floor 10 shop for the potion price regression seed")

    def test_shop_colorless_card_price_regression_seed_matches_lightspeed_in_v2(self):
        seed = 4492233816403006015
        result = compare_seed(seed, 0, 260, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_burning_blood_heals_six_after_combat_in_v2(self):
        env = NativeRunEnv(seed=1625, ascension_level=0, start_on_map=True)
        env.current_node_symbol = "M"
        env.player.current_hp = 40
        env.player.max_hp = 80
        env._enter_card_reward()

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(env.player.current_hp, 46)

    def test_meat_on_the_bone_checks_preheal_threshold_in_v2(self):
        env = NativeRunEnv(seed=1626, ascension_level=0, start_on_map=True)
        env.current_node_symbol = "M"
        env.relics = [make_relic("Black Blood"), make_relic("Meat on the Bone")]
        env.player.current_hp = 34
        env.player.max_hp = 80

        env._enter_card_reward()

        self.assertEqual(env.player.current_hp, 58)

    def test_meat_on_the_bone_with_burning_blood_checks_postheal_threshold_in_v2(self):
        env = NativeRunEnv(seed=16265, ascension_level=0, start_on_map=True)
        env.current_node_symbol = "M"
        env.relics = [make_relic("Burning Blood"), make_relic("Meat on the Bone")]
        env.player.current_hp = 35
        env.player.max_hp = 80

        env._enter_card_reward()

        self.assertEqual(env.player.current_hp, 41)

    def test_meat_on_the_bone_does_not_trigger_after_looter_escape_in_v2(self):
        env = NativeRunEnv(seed=1627, ascension_level=0, start_on_map=True)
        env.current_node_symbol = "M"
        env.relics = [make_relic("Burning Blood"), make_relic("Meat on the Bone")]
        env.player.current_hp = 36
        env.player.max_hp = 80

        env._enter_card_reward(allow_meat_on_the_bone=False)

        self.assertEqual(env.player.current_hp, 42)

    def test_meat_on_the_bone_can_trigger_after_looter_escape_when_postheal_hp_stays_low_in_v2(self):
        env = NativeRunEnv(seed=16271, ascension_level=0)
        env.phase = "COMBAT"
        env.current_node_symbol = "M"
        env.relics = [make_relic("Burning Blood"), make_relic("Meat on the Bone")]
        env.player.current_hp = 23
        env.player.max_hp = 80
        combat = NativeCombatEnv(seed=16271, ascension_level=0, player=env.player, relics=list(env.relics))
        combat.outcome = "PLAYER_VICTORY"
        combat.gold = env.gold
        combat.gold_gain = 0
        combat.reward_gold_bonus = 0
        looter = make_monster("Looter", StsRandom(16271), ascension=0)
        looter.ai_state["escaping"] = True
        combat.monsters = [looter]
        env.combat = combat

        env.step({})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(env.player.current_hp, 41)

    def test_maw_bank_triggers_when_entering_boss_relic_room_in_v2(self):
        env = NativeRunEnv(seed=16251, ascension_level=0, start_on_map=True)
        env.floor = 16
        env.gold = 100
        env.relics.append({"relic_id": "Maw Bank", "name": "Maw Bank", "tier": "COMMON", "counter": 1})

        env._enter_boss_treasure_room()

        self.assertEqual(env.floor, 17)
        self.assertEqual(env.gold, 112)
        self.assertEqual(env.phase, "BOSS_RELIC")

    def test_random_colorless_card_uses_ordered_combat_pool(self):
        combat = NativeCombatEnv(seed=23, ascension_level=0, player=PlayerState())
        class FakeRng:
            def random(self, upper):
                return 29
        combat.card_random_rng = FakeRng()

        card_id = combat._random_card_id(colorless=True)

        self.assertEqual(card_id, "Dramatic Entrance")

    def test_magnetism_is_currently_noop_at_start_of_turn_in_v2(self):
        combat = NativeCombatEnv(seed=23, ascension_level=0, player=PlayerState())
        combat.hand = []
        combat.draw_pile = [make_card("Strike_R", uuid="draw-1") for _ in range(5)]
        combat.discard_pile = []
        combat.player.powers["Magnetism"] = 1

        combat.start_player_turn()

        self.assertEqual([card.card_id for card in combat.hand], ["Strike_R"] * 5)

    def test_dead_branch_uses_card_random_combat_pool_in_v2(self):
        combat = NativeCombatEnv(seed=23, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Dead Branch", "name": "Dead Branch", "tier": "RARE", "counter": -1})
        combat.hand = []

        class FakeCardRandom:
            def random(self, upper):
                return 0

        combat.card_random_rng = FakeCardRandom()
        combat._exhaust_card(make_card("Dazed", uuid="dead-branch-dazed"))

        self.assertEqual([card.card_id for card in combat.hand], ["Sword Boomerang"])

    def test_trip_card_definition_matches_lightspeed(self):
        trip = make_card("Trip")

        self.assertEqual(trip.cost, 0)
        self.assertFalse(trip.card_def.exhausts)
        self.assertTrue(trip.card_def.has_target)

    def test_rage_card_definition_matches_lightspeed(self):
        rage = make_card("Rage")

        self.assertEqual(rage.cost, 0)
        self.assertFalse(rage.card_def.exhausts)

    def test_infernal_blade_card_definition_matches_lightspeed(self):
        infernal_blade = make_card("Infernal Blade")
        infernal_blade_plus = make_card("Infernal Blade", upgrades=1)

        self.assertEqual(infernal_blade.cost, 1)
        self.assertEqual(infernal_blade_plus.cost, 0)
        self.assertTrue(infernal_blade.card_def.exhausts)

    def test_seeing_red_card_definition_matches_lightspeed(self):
        seeing_red = make_card("Seeing Red")
        seeing_red_plus = make_card("Seeing Red", upgrades=1)

        self.assertEqual(seeing_red.cost, 1)
        self.assertEqual(seeing_red_plus.cost, 0)
        self.assertTrue(seeing_red.card_def.exhausts)

    def test_seeing_red_self_exhausts_in_v2(self):
        combat = NativeCombatEnv(seed=2430, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Seeing Red", uuid="seeing-red")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.player.energy, 2)
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Seeing Red"])
        self.assertEqual(combat.discard_pile, [])

    def test_lightspeed_marked_card_constants_match_updated_truth(self):
        cards_h = Path("/home/yydd/sts_lightspeed/include/constants/Cards.h").read_text()

        self.assertIn("case CardId::RAGE:", cards_h)
        self.assertIn("case CardId::TRIP:", cards_h)
        self.assertIn("case CardId::INFERNAL_BLADE:", cards_h)
        self.assertIn("case CardId::SEEING_RED:", cards_h)
        self.assertIn("case CardId::RAGE:\n            case CardId::RECKLESS_CHARGE:", cards_h)
        self.assertIn("case CardId::SWIFT_STRIKE:\n            case CardId::THINKING_AHEAD:\n            case CardId::TRIP:", cards_h)
        self.assertIn("case CardId::HAVOC:\n            case CardId::INFERNAL_BLADE:\n            case CardId::SEEING_RED:", cards_h)
        self.assertIn("case CardId::INFERNAL_BLADE:\n            case CardId::SEEING_RED:", cards_h)

    def test_trip_unupgraded_targets_only_selected_enemy_in_v2(self):
        combat = NativeCombatEnv(seed=24, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Trip", uuid="trip")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.monsters = [
            make_monster("JawWorm", StsRandom(1), 0),
            make_monster("Cultist", StsRandom(2), 0),
        ]

        combat.play_card(0, 1)

        self.assertEqual(combat.monsters[0].power("Vulnerable"), 0)
        self.assertEqual(combat.monsters[1].power("Vulnerable"), 2)

    def test_trip_plus_is_nontargeted_and_hits_all_enemies_in_v2(self):
        combat = NativeCombatEnv(seed=25, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Trip", upgrades=1, uuid="trip+")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.monsters = [
            make_monster("JawWorm", StsRandom(1), 0),
            make_monster("Cultist", StsRandom(2), 0),
        ]

        actions = combat.legal_actions()

        self.assertEqual(
            [(action["kind"], action["card_id"], action["target_index"]) for action in actions if action["kind"] == "card"],
            [("card", "Trip", 0)],
        )

        combat.play_card(0, 0)

        self.assertEqual(combat.monsters[0].power("Vulnerable"), 2)
        self.assertEqual(combat.monsters[1].power("Vulnerable"), 2)

    def test_trip_plus_slime_boss_regression_seed_matches_lightspeed_in_v2(self):
        result = compare_seed(438223609083836701, 0, 206, backend="v2")

        self.assertTrue(result["match"], result)

    def test_headbutt_card_select_respects_force_exhaust_in_v2(self):
        combat = NativeCombatEnv(seed=25, ascension_level=0, player=PlayerState())
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = [make_card("Battle Trance", uuid="discarded-bt")]
        combat.exhaust_pile = []
        combat.player.powers["Feel No Pain"] = 3
        combat.card_select_context = "HEADBUTT"
        combat.pending_resolve_card = make_card("Headbutt", uuid="pending-headbutt")
        combat.pending_resolve_force_exhaust = True

        combat._resolve_card_select({"kind": "card_select", "select_index": 0})

        self.assertEqual([card.card_id for card in combat.draw_pile], ["Battle Trance"])
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Headbutt"])
        self.assertEqual(combat.player.block, 3)

    def test_force_exhaust_overrides_non_exhaust_upgrade_in_v2(self):
        combat = NativeCombatEnv(seed=26, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Limit Break", upgrades=1, uuid="limit-break-plus")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0, force_exhaust=True, free_to_play=True)

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Limit Break"])
        self.assertEqual(combat.discard_pile, [])

    def test_snake_plant_enfeebling_spores_only_applies_weakened_in_v2(self):
        combat = NativeCombatEnv(seed=263, ascension_level=0, player=PlayerState())
        snake_plant = make_monster("SnakePlant", combat.monster_hp_rng, 0)
        snake_plant.move = "SNAKE_PLANT_ENFEEBLING_SPORES"
        snake_plant.intent = "DEBUFF"
        combat.monsters = [snake_plant]
        combat.player.powers["Artifact"] = 1

        combat._monster_take_turn(snake_plant)

        self.assertEqual(combat.player.power("Weakened"), 2)
        self.assertEqual(combat.player.power("Frail"), 0)
        self.assertEqual(combat.player.power("Artifact"), 0)

    def test_limit_break_doubles_negative_strength_in_v2(self):
        combat = NativeCombatEnv(seed=26, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Limit Break", uuid="limit-break")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Strength"] = -1

        combat.play_card(0, 0)

        self.assertEqual(combat.player.power("Strength"), -2)

    def test_temporary_upgraded_limit_break_discards_in_v2(self):
        combat = NativeCombatEnv(seed=261, ascension_level=0, player=PlayerState())
        limit_break = make_card("Limit Break", upgrades=1, uuid="limit-break-temp")
        limit_break._temporary_upgrade = True
        combat.hand = [limit_break]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Strength"] = 2

        combat.play_card(0, 0)

        self.assertEqual(combat.player.power("Strength"), 4)
        self.assertEqual(combat.exhaust_pile, [])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Limit Break"])

    def test_upgraded_limit_break_still_exhausts_under_corruption_in_v2(self):
        combat = NativeCombatEnv(seed=262, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Limit Break", upgrades=1, uuid="limit-break-plus")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Strength"] = 2
        combat.player.powers["Corruption"] = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.player.power("Strength"), 4)
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Limit Break"])
        self.assertEqual(combat.discard_pile, [])

    def test_armaments_limit_break_discards_regression_seed_matches_lightspeed_in_v2(self):
        result = compare_seed(4448915233998711785, 0, 200, backend="v2")
        self.assertTrue(result["match"], msg=result)

    def test_x_cost_cards_remain_playable_at_zero_energy_in_v2(self):
        combat = NativeCombatEnv(seed=27, ascension_level=0, player=PlayerState())
        transmutation = make_card("Transmutation", uuid="x-cost")
        combat.hand = [transmutation]
        combat.player.energy = 0

        self.assertTrue(combat.playable(transmutation))

    def test_x_cost_cards_keep_x_marker_for_turn_only_override_in_v2(self):
        combat = NativeCombatEnv(seed=28, ascension_level=0, player=PlayerState())
        whirlwind = make_card("Whirlwind", uuid="whirlwind")
        whirlwind.cost_for_turn = 0
        combat.hand = [whirlwind]

        state = combat.to_spirecomm_state()
        self.assertEqual(state["combat_state"]["hand"][0]["cost_for_turn"], -1)

    def test_generated_x_cost_cards_keep_explicit_zero_cost_in_state_in_v2(self):
        combat = NativeCombatEnv(seed=281, ascension_level=0, player=PlayerState())
        whirlwind = make_card("Whirlwind", uuid="whirlwind-generated")
        whirlwind.cost_for_combat = 0
        combat.hand = [whirlwind]

        state = combat.to_spirecomm_state()

        self.assertEqual(state["combat_state"]["hand"][0]["cost_for_turn"], 0)

    def test_transmutation_at_zero_energy_generates_no_cards_in_v2(self):
        combat = NativeCombatEnv(seed=29, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Transmutation", uuid="x-cost")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 0

        combat.play_card(0, 0)

        self.assertEqual(combat.hand, [])
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Transmutation"])

    def test_madness_sets_permanent_zero_cost_on_selected_card_in_v2(self):
        combat = NativeCombatEnv(seed=30, ascension_level=0, player=PlayerState())
        target = make_card("Strike_R", uuid="strike")
        combat.hand = [make_card("Madness", uuid="madness"), target]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        class FixedRandom:
            def random(self, upper):
                return 0 if upper == 0 else 1

        combat.card_random_rng = FixedRandom()

        combat.play_card(0, 0)

        self.assertEqual(target.cost, 0)
        self.assertEqual(target.cost_for_turn, 0)

    def test_madness_can_target_discovery_turn_only_zero_cost_card_in_v2(self):
        combat = NativeCombatEnv(seed=3031, ascension_level=0, player=PlayerState())
        target = make_card("Rage", uuid="Discovery-Rage-test")
        target.cost_for_turn = 0
        combat.hand = [make_card("Madness", uuid="madness"), target]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        class FixedRandom:
            def random(self, upper):
                return 0 if upper == 0 else 1

        combat.card_random_rng = FixedRandom()

        combat.play_card(0, 0)

        self.assertEqual(target.card_def.cost, 0)
        self.assertEqual(target.cost_for_turn, 0)
        combat._clear_temporary_cost_state()
        self.assertIsNone(target.cost_for_turn)
        self.assertEqual(target.cost, 0)

    def test_body_slam_uses_pre_rage_block_snapshot_in_v2(self):
        combat = NativeCombatEnv(seed=3032, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80, block=10))
        target = make_monster("Cultist", StsRandom(3032), ascension=0)
        target.current_hp = 20
        combat.monsters = [target]
        combat.player.powers["Rage"] = 3
        combat.hand = [make_card("Body Slam", uuid="body-slam")]

        combat.play_card(0, 0)

        self.assertEqual(target.current_hp, 10)
        self.assertEqual(combat.player.block, 13)

    def test_spike_slime_flame_tackle_adds_slimed_after_centennial_puzzle_draw_in_v2(self):
        combat = NativeCombatEnv(seed=3033, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80))
        combat.relics.append({"relic_id": "Centennial Puzzle", "id": "Centennial Puzzle", "name": "Centennial Puzzle", "counter": 0, "price": 0, "tier": "UNCOMMON"})
        slime = make_monster("SpikeSlime_M", StsRandom(3033), ascension=0)
        slime.move = "SPIKE_SLIME_M_FLAME_TACKLE"
        slime.intent = "ATTACK_DEBUFF"
        slime.move_base_damage = 8
        slime.move_hits = 1
        combat.monsters = [slime]
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = [
            make_card("Bash", uuid="discard-bash"),
            make_card("Defend_R", uuid="discard-defend"),
            make_card("Strike_R", uuid="discard-strike"),
        ]

        combat._monster_take_turn(slime, 0)

        self.assertEqual(len(combat.hand), 3)
        self.assertNotIn("Slimed", [card.card_id for card in combat.hand])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Slimed"])

    def test_centennial_puzzle_end_turn_draw_excludes_regular_end_turn_discards_from_shuffle_in_v2(self):
        seed = 3301439061689089639
        ls_env = sts.ModelDrivenEnv(seed, 0)
        native = NativeRunEnv(seed=seed, ascension_level=0, enable_neow=True)

        for _ in range(78):
            ls_in_battle = bool(ls_env.in_battle)
            if ls_in_battle:
                ls_actions = [dict(action) for action in sts.get_battle_actions(ls_env)]
                ls_state = sts.get_battle_state(ls_env)
                native_state = native.state()
            else:
                ls_actions = [dict(action) for action in sts.get_external_actions(ls_env.game_context)]
                ls_state = None
                native_state = None
            native_actions = [dict(action) for action in native.legal_actions()]
            chosen_sig = _choice_list_signature(ls_actions, ls_state)[0]
            ls_chosen = _pick_action_by_signature(ls_actions, chosen_sig, ls_state)
            native_chosen = _pick_action_by_signature(native_actions, chosen_sig, native_state)
            if ls_in_battle:
                sts.execute_battle_action_bits(ls_env, int(ls_chosen["bits"]))
            else:
                sts.execute_action_bits(ls_env, int(ls_chosen["bits"]))
            native.step(native_chosen)

        native.combat.end_turn()

        self.assertEqual(
            [card.card_id for card in native.combat.hand],
            ["Defend_R", "Strike_R", "Defend_R", "Heavy Blade", "Defend_R"],
        )
        self.assertEqual(
            [card.card_id for card in native.combat.draw_pile],
            ["Cleave", "Strike_R", "Strike_R", "Clothesline"],
        )
        self.assertEqual(
            [card.card_id for card in native.combat.discard_pile],
            ["Bash", "Defend_R", "Strike_R", "Dropkick", "Combust", "Havoc"],
        )

    def test_madness_resamples_full_hand_until_valid_candidate_in_v2(self):
        combat = NativeCombatEnv(seed=3030, ascension_level=0, player=PlayerState())
        invalid = make_card("Strike_R", uuid="strike-invalid")
        invalid.cost_for_turn = 0
        valid = make_card("Shockwave", uuid="shockwave-valid")
        valid.cost_for_turn = 2
        combat.hand = [make_card("Madness", uuid="madness"), invalid, valid]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        class _FilteredRng:
            def __init__(self):
                self.calls = []

            def random(self, upper):
                self.calls.append(upper)
                return 0 if len(self.calls) == 1 else 1

        combat.card_random_rng = _FilteredRng()

        combat.play_card(0, 0)

        self.assertEqual(combat.card_random_rng.calls, [1, 1])
        self.assertEqual(valid.cost, 0)
        self.assertEqual(valid.cost_for_turn, 0)
        self.assertEqual(valid.cost_for_turn, 0)
        self.assertEqual(invalid.cost_for_turn, 0)

    def test_hand_of_greed_uses_lightspeed_card_id_in_v2(self):
        hand_of_greed = make_card("Hand of Greed", uuid="hog")

        self.assertEqual(hand_of_greed.card_id, "HandOfGreed")

    def test_panic_button_exhausts_in_v2(self):
        panic_button = make_card("Panic Button", uuid="panic")

        self.assertTrue(panic_button.card_def.exhausts)

    def test_feel_no_pain_block_is_not_blocked_by_no_block_in_v2(self):
        combat = NativeCombatEnv(seed=31, ascension_level=0, player=PlayerState())
        combat.player.powers["Feel No Pain"] = 3
        combat.player.powers["No Block"] = 2
        combat._exhaust_card(make_card("Panic Button", uuid="panic"))

        self.assertEqual(combat.player.block, 3)

    def test_combust_tracks_stacked_hp_loss_in_v2(self):
        combat = NativeCombatEnv(seed=33, ascension_level=0, player=PlayerState())
        combat.player.current_hp = 30
        combat.player.powers["Combust"] = 10
        combat.combust_hp_loss = 2
        combat.monsters = [make_monster("FungiBeast", StsRandom(1), ascension=0)]
        combat._monster_take_turn = lambda *args, **kwargs: None
        combat._monster_prepare_next_turn = lambda *args, **kwargs: None

        combat.end_turn()

        self.assertEqual(combat.player.current_hp, 28)

    def test_headbutt_defers_curl_up_block_until_card_select_resolves_in_v2(self):
        combat = NativeCombatEnv(seed=35, ascension_level=0, player=PlayerState())
        target = make_monster("GreenLouse", StsRandom(5), ascension=0)
        target.current_hp = 11
        target.max_hp = 11
        target.powers = {"Curl Up": 3}
        combat.monsters = [target]
        combat.hand = [make_card("Headbutt", uuid="headbutt")]
        combat.discard_pile = [make_card("Bash", uuid="bash"), make_card("Strike_R", uuid="strike")]
        combat.draw_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat.monsters[0].block, 0)
        self.assertEqual(combat.monsters[0].powers, {})

        combat.step({"kind": "card_select", "select_index": 0})

        self.assertEqual(combat.monsters[0].block, 3)

    def test_headbutt_dead_louse_keeps_curl_up_block_after_card_select_in_v2(self):
        combat = NativeCombatEnv(seed=3501, ascension_level=0, scheduled_encounter=["GreenLouse", "RedLouse"], player=PlayerState())
        first, second = combat.monsters[:2]
        first.current_hp = 9
        first.max_hp = 11
        first.powers = {"Curl Up": 5}
        second.current_hp = 14
        second.max_hp = 14
        combat.hand = [make_card("Flex", uuid="flex"), make_card("Headbutt", uuid="headbutt")]
        combat.discard_pile = [make_card("Bash", uuid="bash"), make_card("Strike_R", uuid="strike")]
        combat.draw_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.step({"kind": "card", "card_index": 0})
        combat.step({"kind": "card", "card_index": 0, "target_index": 0})

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat.monsters[0].current_hp, 0)
        self.assertEqual(combat.monsters[0].block, 0)

        combat.step({"kind": "card_select", "select_index": 0})

        self.assertEqual(combat.monsters[0].current_hp, 0)
        self.assertEqual(combat.monsters[0].block, 5)

    def test_bash_sadistic_nature_resolves_curl_up_before_delayed_damage_in_v2(self):
        combat = NativeCombatEnv(seed=36, ascension_level=0, player=PlayerState())
        target = make_monster("GreenLouse", StsRandom(6), ascension=0)
        target.current_hp = 14
        target.max_hp = 14
        target.powers = {"Curl Up": 7}
        combat.monsters = [target]
        combat.hand = [make_card("Bash", uuid="bash")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2
        combat.player.powers["Weakened"] = 2
        combat.player.powers["Strength"] = 2
        combat.player.powers["Sadistic Nature"] = 5

        combat.step({"kind": "card", "card_index": 0, "target_index": 0})

        self.assertEqual(combat.monsters[0].current_hp, 7)
        self.assertEqual(combat.monsters[0].block, 2)
        self.assertEqual(combat.monsters[0].power("Vulnerable"), 2)

    def test_bash_sadistic_nature_updates_victory_after_delayed_kill_in_v2(self):
        combat = NativeCombatEnv(seed=3601, ascension_level=0, player=PlayerState(energy=2))
        target = make_monster("Cultist", StsRandom(3601), ascension=0)
        target.current_hp = 11
        target.max_hp = 11
        combat.monsters = [target]
        combat.hand = [make_card("Bash", uuid="bash")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.powers["Sadistic Nature"] = 5

        combat.step({"kind": "card", "card_index": 0, "target_index": 0})

        self.assertEqual(combat.outcome, "PLAYER_VICTORY")
        self.assertEqual(combat.monsters[0].current_hp, 0)

    def test_hand_of_greed_does_not_trigger_curl_up_in_v2(self):
        combat = NativeCombatEnv(seed=361, ascension_level=0, player=PlayerState())
        target = make_monster("RedLouse", StsRandom(61), ascension=0)
        target.current_hp = 10
        target.max_hp = 10
        target.powers = {"Curl Up": 4}
        combat.monsters = [target]
        combat.hand = [make_card("Hand of Greed", uuid="hog")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.step({"kind": "card", "card_index": 0, "target_index": 0})

        self.assertEqual(combat.monsters[0].current_hp, 0)
        self.assertEqual(combat.monsters[0].block, 0)
        self.assertEqual(combat.monsters[0].power("Curl Up"), 4)

    def test_headbutt_replay_does_not_apply_curl_up_block_to_dead_louse_in_v2(self):
        combat = NativeCombatEnv(seed=362, ascension_level=0, player=PlayerState())
        target = make_monster("GreenLouse", StsRandom(62), ascension=0)
        target.current_hp = 11
        target.max_hp = 11
        target.powers = {"Curl Up": 5}
        combat.monsters = [target]
        combat.hand = [make_card("Double Tap", uuid="double-tap"), make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = []
        combat.discard_pile = [make_card("Defend_R", uuid="defend")]
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.step({"kind": "card", "card_index": 0, "target_index": 0})
        combat.step({"kind": "card", "card_index": 0, "target_index": 0})
        combat.step({"kind": "card_select", "select_index": 0})

        self.assertEqual(combat.monsters[0].current_hp, 0)
        self.assertEqual(combat.monsters[0].block, 0)

    def test_double_tap_replay_resolves_curl_up_between_hits_in_v2(self):
        combat = NativeCombatEnv(seed=37, ascension_level=0, player=PlayerState())
        target = make_monster("GreenLouse", StsRandom(7), ascension=0)
        target.current_hp = 12
        target.max_hp = 12
        target.powers = {"Curl Up": 4}
        combat.monsters = [target]
        combat.hand = [make_card("Strike_R", uuid="strike")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Double Tap"] = 1
        combat.player.powers["Weakened"] = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.monsters[0].current_hp, 8)
        self.assertEqual(combat.monsters[0].block, 0)

    def test_double_tap_hemokinesis_replay_repeats_self_damage_in_v2(self):
        combat = NativeCombatEnv(seed=38, ascension_level=0, player=PlayerState())
        target = make_monster("Cultist", StsRandom(8), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.hand = [make_card("Hemokinesis", uuid="hemo")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.current_hp = 50
        combat.player.max_hp = 50
        combat.player.powers["Double Tap"] = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 46)
        self.assertEqual(combat.monsters[0].current_hp, 10)

    def test_hemokinesis_uses_pre_self_damage_strength_snapshot_in_v2(self):
        combat = NativeCombatEnv(seed=3801, ascension_level=0, player=PlayerState())
        target = make_monster("Cultist", StsRandom(801), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.hand = [make_card("Hemokinesis", uuid="hemo")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.current_hp = 50
        combat.player.max_hp = 50
        combat.player.powers["Rupture"] = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 48)
        self.assertEqual(combat.monsters[0].current_hp, 25)
        self.assertEqual(combat.player.power("Strength"), 1)

    def test_double_tap_targeted_replay_skips_when_original_target_dies_in_v2(self):
        combat = NativeCombatEnv(seed=38, ascension_level=0, player=PlayerState())
        target = make_monster("SpikeSlime_S", StsRandom(8), ascension=0)
        target.current_hp = 5
        target.max_hp = 5
        other = make_monster("AcidSlime_M", StsRandom(9), ascension=0)
        other.current_hp = 21
        other.max_hp = 21
        combat.monsters = [target, other]
        combat.hand = [make_card("Hemokinesis", uuid="hemo")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.current_hp = 50
        combat.player.max_hp = 50
        combat.player.powers["Double Tap"] = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 48)
        self.assertEqual(combat.monsters[0].current_hp, 0)
        self.assertEqual(combat.monsters[1].current_hp, 21)

    def test_v2_step_flushes_delayed_reactions_after_normal_card_play(self):
        combat = NativeCombatEnv(seed=39, ascension_level=0, player=PlayerState())
        target = make_monster("GreenLouse", StsRandom(9), ascension=0)
        target.current_hp = 11
        target.max_hp = 11
        target.powers = {"Curl Up": 3}
        combat.monsters = [target]
        combat.hand = [make_card("Strike_R", uuid="strike")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.step({"kind": "card", "card_index": 0, "target_index": 0})

        self.assertEqual(combat.monsters[0].block, 3)

    def test_true_grit_uses_card_random_rng_for_unupgraded_exhaust_choice(self):
        combat = NativeCombatEnv(seed=41, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("True Grit", uuid="true-grit"),
            make_card("Strike_R", uuid="strike"),
            make_card("Flame Barrier", uuid="flame-barrier"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        class FixedChoiceRng:
            def choice(self, sequence):
                return sequence[-1]

        combat.card_random_rng = FixedChoiceRng()

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Strike_R"])
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Flame Barrier"])

    def test_true_grit_moves_to_discard_before_ink_bottle_shuffle_draw_in_v2(self):
        combat = NativeCombatEnv(seed=999, ascension_level=0, player=PlayerState())
        combat.relics.append(make_relic("Ink Bottle", counter=9))
        combat.hand = [
            make_card("True Grit", uuid="true-grit"),
            make_card("Burn", uuid="burn"),
        ]
        combat.draw_pile = []
        combat.discard_pile = [
            make_card("Anger", uuid="anger"),
            make_card("Strike_R", uuid="strike"),
        ]
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.monsters = [make_monster("Cultist", StsRandom(999), ascension=0)]

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Anger"])
        self.assertEqual([card.card_id for card in combat.draw_pile], ["True Grit", "Strike_R"])
        self.assertEqual([card.card_id for card in combat.discard_pile], [])

    def test_true_grit_moves_to_discard_before_dark_embrace_shuffle_draw_in_v2(self):
        combat = NativeCombatEnv(seed=1000, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("True Grit", uuid="true-grit"),
            make_card("Burn", uuid="burn"),
        ]
        combat.draw_pile = []
        combat.discard_pile = [
            make_card("Anger", uuid="anger"),
            make_card("Strike_R", uuid="strike"),
        ]
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Dark Embrace"] = 1
        combat.monsters = [make_monster("Cultist", StsRandom(1000), ascension=0)]

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Strike_R"])
        self.assertEqual([card.card_id for card in combat.draw_pile], ["True Grit", "Anger"])
        self.assertEqual([card.card_id for card in combat.discard_pile], [])

    def test_armaments_card_select_draws_before_after_use_move_in_v2(self):
        combat = NativeCombatEnv(seed=1001, ascension_level=0, player=PlayerState())
        combat.relics.append(make_relic("Ink Bottle", counter=9))
        combat.hand = [
            make_card("Armaments", uuid="armaments"),
            make_card("Strike_R", uuid="strike"),
            make_card("Defend_R", uuid="defend"),
        ]
        combat.draw_pile = []
        combat.discard_pile = [
            make_card("Anger", uuid="anger"),
            make_card("Bash", uuid="bash"),
        ]
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.monsters = [make_monster("Cultist", StsRandom(1001), ascension=0)]

        combat.play_card(0, 0)
        combat.step({"kind": "card_select", "select_index": 0})

        self.assertEqual([card.card_id for card in combat.hand], ["Defend_R", "Strike_R", "Anger"])
        self.assertEqual([card.card_id for card in combat.draw_pile], ["Bash"])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Armaments"])

    def test_armaments_upgraded_apparition_does_not_exhaust_at_end_turn_in_v2(self):
        combat = NativeCombatEnv(seed=10011, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Armaments", uuid="armaments+"),
            make_card("Apparition", uuid="apparition"),
        ]
        combat.hand[0].upgrades = 1
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.monsters = [make_monster("Cultist", StsRandom(10011), ascension=0)]

        combat.play_card(0, 0)
        combat.step({"kind": "card_select", "select_index": 0})
        combat.end_turn()

        self.assertEqual([card.card_id for card in combat.exhaust_pile], [])
        self.assertIn("Apparition", [card.card_id for card in combat.hand])

    def test_fiend_fire_uses_original_hand_count_for_hits_in_v2(self):
        combat = NativeCombatEnv(seed=42, ascension_level=0, player=PlayerState())
        target = make_monster("JawWorm", StsRandom(5), ascension=0)
        target.current_hp = 50
        target.max_hp = 50
        combat.monsters = [target]
        combat.hand = [
            make_card("Fiend Fire", uuid="fiend-fire"),
            make_card("Strike_R", uuid="strike"),
            make_card("Heavy Blade", uuid="heavy-blade"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        class ZeroRandom:
            def random(self, upper):
                return 0

        combat.card_random_rng = ZeroRandom()

        combat.play_card(0, 0)

        self.assertEqual(combat.monsters[0].current_hp, 36)
        self.assertEqual(sorted(card.card_id for card in combat.exhaust_pile), ["Fiend Fire", "Heavy Blade", "Strike_R"])

    def test_fiend_fire_does_not_chain_exhaust_dead_branch_generated_cards_in_v2(self):
        combat = NativeCombatEnv(seed=43, ascension_level=0, player=PlayerState())
        target = make_monster("JawWorm", StsRandom(7), ascension=0)
        combat.monsters = [target]
        combat.hand = [
            make_card("Fiend Fire", uuid="fiend-fire"),
            make_card("Strike_R", uuid="strike"),
            make_card("Heavy Blade", uuid="heavy-blade"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.relics.append({"relic_id": "Dead Branch", "name": "Dead Branch", "tier": "RARE", "counter": -1})
        combat.player.energy = 2

        class ZeroRandom:
            def random(self, upper):
                return 0

        combat.card_random_rng = ZeroRandom()

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Sword Boomerang", "Sword Boomerang", "Sword Boomerang"])
        self.assertEqual(sorted(card.card_id for card in combat.exhaust_pile), ["Fiend Fire", "Heavy Blade", "Strike_R"])

    def test_fiend_fire_replay_with_empty_hand_does_not_deal_extra_hit_in_v2(self):
        combat = NativeCombatEnv(seed=43015, ascension_level=0, player=PlayerState())
        target = make_monster("Byrd", StsRandom(17), ascension=0)
        target.current_hp = 31
        target.max_hp = 31
        target.move = "BYRD_CAW"
        target.intent = "BUFF"
        target.powers["Flight"] = 3
        combat.monsters = [target]
        combat.hand = [
            make_card("Fiend Fire", upgrades=1, uuid="fiend-fire-plus"),
            make_card("Heavy Blade", uuid="heavy-blade"),
            make_card("Strike_R", uuid="strike"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2
        combat.player.powers["Double Tap"] = 1

        class ZeroRandom:
            def random(self, upper):
                return 0

        combat.card_random_rng = ZeroRandom()

        combat.play_card(0, 0)

        self.assertEqual(combat.monsters[0].current_hp, 21)
        self.assertEqual(combat.monsters[0].move, "BYRD_CAW")
        self.assertEqual(combat.monsters[0].power("Flight"), 1)
        self.assertEqual(sorted(card.card_id for card in combat.exhaust_pile), ["Fiend Fire", "Heavy Blade", "Strike_R"])

    def test_fiend_fire_with_empty_hand_still_triggers_sharp_hide_once_in_v2(self):
        combat = NativeCombatEnv(seed=430151, ascension_level=0, scheduled_encounter=["TheGuardian"], player=PlayerState(block=5))
        guardian = combat.monsters[0]
        guardian.powers["Sharp Hide"] = 3
        combat.hand = [make_card("Fiend Fire", uuid="fiend-fire")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)

        self.assertEqual(combat.player.block, 2)
        self.assertEqual(guardian.current_hp, guardian.max_hp)

    def test_fiend_fire_delays_attack_relic_proc_until_all_hits_in_v2(self):
        combat = NativeCombatEnv(seed=43016, ascension_level=0, player=PlayerState())
        looter = make_monster("Looter", StsRandom(18), ascension=0)
        looter.current_hp = 47
        looter.max_hp = 47
        combat.monsters = [looter]
        combat.hand = [
            make_card("Fiend Fire", uuid="fiend-fire"),
            make_card("Strike_R", uuid="strike-0"),
            make_card("Strike_R", upgrades=1, uuid="strike-1"),
            make_card("Clash", uuid="clash"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2
        combat.relics.append({"relic_id": "Shuriken", "name": "Shuriken", "tier": "COMMON", "counter": -1})
        combat.attack_played_this_turn = 2

        combat.play_card(0, 0)

        self.assertEqual(looter.current_hp, 26)
        self.assertEqual(combat.player.power("Strength"), 1)

    def test_burning_pact_draws_before_dead_branch_generated_card_in_v2(self):
        combat = NativeCombatEnv(seed=4301, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Burning Pact", uuid="burning-pact"),
            make_card("Dazed", uuid="burning-pact-dazed"),
        ]
        combat.draw_pile = [
            make_card("Strike_R", uuid="burning-pact-draw-1"),
            make_card("Defend_R", uuid="burning-pact-draw-2"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.relics.append({"relic_id": "Dead Branch", "name": "Dead Branch", "tier": "RARE", "counter": -1})
        combat.player.energy = 1
        combat._random_combat_card_id = lambda: "Impervious"

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Defend_R", "Strike_R", "Impervious"])
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Dazed"])

    def test_burning_pact_card_select_resolves_selected_dead_branch_before_self_in_v2(self):
        combat = NativeCombatEnv(seed=43011, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Burning Pact", uuid="burning-pact"),
            make_card("Impatience", uuid="burning-pact-impatience"),
        ]
        combat.draw_pile = [
            make_card("Strike_R", uuid="burning-pact-draw-1"),
            make_card("Fiend Fire", uuid="burning-pact-draw-2"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.relics.append({"relic_id": "Dead Branch", "name": "Dead Branch", "tier": "RARE", "counter": -1})
        combat.player.energy = 1

        generated = iter(["Armaments", "Seeing Red"])
        combat._random_combat_card_id = lambda: next(generated)

        combat.play_card(0, 0)
        combat.step({"kind": "card_select", "name": "EXHAUST_ONE", "select_type": "EXHAUST_ONE", "deck_index": 0})

        self.assertEqual([card.card_id for card in combat.hand], ["Fiend Fire", "Strike_R", "Armaments"])

    def test_burning_pact_card_select_resolves_ink_bottle_draw_before_self_move_in_v2(self):
        combat = NativeCombatEnv(seed=43012, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Ink Bottle", "name": "Ink Bottle", "tier": "COMMON", "counter": 9})
        combat.hand = [
            make_card("Burning Pact", uuid="burning-pact"),
            make_card("Dazed", uuid="burning-pact-dazed"),
            make_card("Havoc", uuid="burning-pact-havoc"),
        ]
        combat.draw_pile = [
            make_card("Strike_R", uuid="draw-strike"),
            make_card("Bash", uuid="draw-bash"),
        ]
        combat.discard_pile = [
            make_card("Defend_R", uuid="discard-defend-0"),
            make_card("Strike_R", uuid="discard-strike-0"),
            make_card("Defend_R", uuid="discard-defend-1"),
            make_card("Clothesline", uuid="discard-clothesline"),
            make_card("Strike_R", uuid="discard-strike-1"),
        ]
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)
        combat.step({"kind": "card_select", "name": "EXHAUST_ONE", "select_type": "EXHAUST_ONE", "deck_index": 0})

        self.assertEqual(combat.card_select_context, None)
        self.assertNotIn("Burning Pact", [card.card_id for card in combat.hand])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Burning Pact"])

    def test_burning_pact_card_select_resolves_pending_spore_cloud_in_v2(self):
        combat = NativeCombatEnv(seed=43013, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Burning Pact", uuid="burning-pact"),
            make_card("Strike_R", uuid="burning-pact-strike"),
        ]
        combat.draw_pile = [
            make_card("Defend_R", uuid="draw-defend-0"),
            make_card("Bash", uuid="draw-bash-0"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)
        combat.pending_spore_cloud_player_turn_triggers = 1
        combat.step({"kind": "card_select", "name": "EXHAUST_ONE", "select_type": "EXHAUST_ONE", "deck_index": 0})

        self.assertEqual(combat.player.power("Vulnerable"), 2)
        self.assertEqual(combat.card_select_context, None)

    def test_sever_soul_exhausts_itself_before_non_attacks_in_v2(self):
        combat = NativeCombatEnv(seed=4302, ascension_level=0, player=PlayerState())
        combat.monsters = [make_monster("JawWorm", StsRandom(11), ascension=0)]
        combat.hand = [
            make_card("Sever Soul", uuid="sever-soul"),
            make_card("Injury", uuid="sever-soul-injury"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0, force_exhaust=True, free_to_play=True)

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Sever Soul", "Injury"])

    def test_fiend_fire_resolves_other_dead_branch_cards_before_its_own_in_v2(self):
        combat = NativeCombatEnv(seed=43021, ascension_level=0, player=PlayerState())
        combat.monsters = [make_monster("JawWorm", StsRandom(13), ascension=0)]
        combat.hand = [
            make_card("Strike_R", uuid="fiend-fire-strike-1"),
            make_card("Strike_R", uuid="fiend-fire-strike-2"),
            make_card("Warcry", uuid="fiend-fire-warcry"),
            make_card("Fiend Fire", uuid="fiend-fire"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.relics.append({"relic_id": "Dead Branch", "name": "Dead Branch", "tier": "RARE", "counter": -1})
        combat.player.energy = 2

        class ZeroRandom:
            def random(self, upper):
                return 0

        generated = iter(["Dual Wield", "Fiend Fire", "Seeing Red", "Pummel"])
        combat.card_random_rng = ZeroRandom()
        combat._random_combat_card_id = lambda: next(generated)

        combat.play_card(3, 0)

        self.assertEqual(
            [card.card_id for card in combat.hand],
            ["Dual Wield", "Fiend Fire", "Seeing Red", "Pummel"],
        )


    def test_byrd_first_turn_uses_lightspeed_opening_roll_in_v2(self):
        byrd = make_monster("Byrd", StsRandom(43), ascension=0)

        rng = StsRandom(1)
        choose_next_move(byrd, rng)
        self.assertEqual(byrd.move, "BYRD_PECK")

        rng = StsRandom(2)
        byrd = make_monster("Byrd", StsRandom(43), ascension=0)
        choose_next_move(byrd, rng)
        self.assertEqual(byrd.move, "BYRD_CAW")

    def test_byrd_stunned_advances_to_headbutt_in_v2(self):
        byrd = make_monster("Byrd", StsRandom(45), ascension=0)
        byrd.move = "BYRD_STUNNED"

        choose_next_move(byrd, StsRandom(3))

        self.assertEqual(byrd.move, "BYRD_HEADBUTT")

    def test_byrd_move_data_matches_lightspeed_in_v2(self):
        byrd = make_monster("Byrd", StsRandom(47), ascension=0)
        choose_next_move(byrd, StsRandom(1))
        self.assertEqual((byrd.move, byrd.move_base_damage, byrd.move_hits), ("BYRD_PECK", 1, 5))

        from spirecomm.native_sim.monsters import _set_move

        _set_move(byrd, "BYRD_HEADBUTT")
        self.assertEqual((byrd.move_base_damage, byrd.move_hits), (3, 1))

    def test_byrd_flight_resets_at_monster_turn_start_in_v2(self):
        combat = NativeCombatEnv(seed=49, ascension_level=0, player=PlayerState())
        byrd = make_monster("Byrd", StsRandom(51), ascension=0)
        byrd.move = "BYRD_PECK"
        byrd.move_history = ["BYRD_PECK"]
        byrd.powers["Flight"] = 2
        combat.monsters = [byrd]
        combat.player.current_hp = 80
        combat.player.block = 0

        combat._monster_take_turn(byrd, 0)

        self.assertEqual(byrd.power("Flight"), 3)

    def test_byrd_zero_and_negative_flight_persist_until_monster_turn_start_in_v2(self):
        combat = NativeCombatEnv(seed=53, ascension_level=0, player=PlayerState())
        byrd = make_monster("Byrd", StsRandom(55), ascension=0)
        byrd.current_hp = 12
        byrd.max_hp = 12
        byrd.powers["Flight"] = 1
        combat.monsters = [byrd]
        combat.hand = [make_card("Strike_R", uuid="strike-1"), make_card("Strike_R", uuid="strike-2")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)

        self.assertEqual(byrd.move, "BYRD_STUNNED")
        self.assertIn("Flight", byrd.powers)
        self.assertEqual(byrd.power("Flight"), 0)
        self.assertEqual(byrd.current_hp, 9)
        state = combat.to_spirecomm_state()
        powers = {
            power["power_id"]: power["amount"]
            for power in state["combat_state"]["monsters"][0]["powers"]
        }
        self.assertNotIn("Flight", powers)

        combat.play_card(0, 0)

        self.assertEqual(byrd.power("Flight"), -1)
        self.assertEqual(byrd.current_hp, 6)

        combat._monster_take_turn(byrd, 0)

        self.assertEqual(byrd.move, "BYRD_HEADBUTT")
        self.assertEqual(byrd.power("Flight"), 3)

        combat._monster_take_turn(byrd, 0)
        self.assertEqual(byrd.move, "BYRD_FLY")
        self.assertEqual(byrd.power("Flight"), 3)

        combat._monster_take_turn(byrd, 0)
        self.assertEqual(byrd.power("Flight"), 6)

    def test_negative_byrd_flight_remains_visible_in_spirecomm_state_in_v2(self):
        combat = NativeCombatEnv(seed=56, ascension_level=0, player=PlayerState())
        byrd = make_monster("Byrd", StsRandom(57), ascension=0)
        byrd.powers["Flight"] = -1
        combat.monsters = [byrd]

        state = combat.to_spirecomm_state()
        powers = {
            power["power_id"]: power["amount"]
            for power in state["combat_state"]["monsters"][0]["powers"]
        }

        self.assertEqual(powers.get("Flight"), -1)

    def test_byrd_fly_adds_flight_in_v2(self):
        combat = NativeCombatEnv(seed=57, ascension_level=0, player=PlayerState())
        byrd = make_monster("Byrd", StsRandom(59), ascension=0)
        byrd.move = "BYRD_FLY"
        byrd.powers["Flight"] = 3
        combat.monsters = [byrd]

        combat._monster_take_turn(byrd, 0)

        self.assertEqual(byrd.power("Flight"), 6)

    def test_byrd_killed_by_thorns_still_rolls_next_move_in_v2(self):
        from spirecomm.native_sim.monsters import _set_move

        combat = NativeCombatEnv(seed=58, ascension_level=0, player=PlayerState())
        byrd = make_monster("Byrd", StsRandom(61), ascension=0)
        _set_move(byrd, "BYRD_PECK")
        byrd.move_history = ["BYRD_PECK"]
        byrd.current_hp = 1
        byrd.max_hp = 29
        combat.monsters = [byrd]
        combat.player.current_hp = 80
        combat.player.block = 0
        combat.player.powers["Thorns"] = 3
        combat.ai_rng = StsRandom(3)

        def _forced_retaliation(_amount: int, monster):
            monster.current_hp = 0
            monster.is_gone = True

        combat._deal_retaliatory_damage_to_monster = _forced_retaliation

        combat._monster_take_turn(byrd, 0)
        skip_end_round_roll = byrd.ai_state.pop("skip_end_round_roll", False)
        roll_move_if_gone = byrd.ai_state.pop("roll_move_if_gone", False)
        if (not byrd.is_gone or roll_move_if_gone) and not skip_end_round_roll:
            choose_next_move(byrd, combat.ai_rng)

        self.assertTrue(byrd.is_gone)
        self.assertEqual(byrd.current_hp, 0)
        self.assertEqual(byrd.move, "BYRD_CAW")

    def test_weakened_zero_damage_multi_hit_still_triggers_thorns_in_v2(self):
        combat = NativeCombatEnv(seed=581, ascension_level=0, scheduled_encounter=["Hexaghost"], player=PlayerState())
        ghost = combat.monsters[0]
        ghost.move = "HEXAGHOST_DIVIDER"
        ghost.intent = "ATTACK"
        ghost.move_base_damage = 1
        ghost.move_hits = 6
        ghost.powers["Weakened"] = 3
        ghost.current_hp = 30
        ghost.max_hp = 264
        combat.player.current_hp = 20
        combat.player.max_hp = 80
        combat.player.powers["Thorns"] = 3
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.end_turn()

        self.assertEqual(combat.player.current_hp, 20)
        self.assertEqual(combat.monsters[0].current_hp, 12)

    def test_shield_gremlin_switches_to_shield_bash_when_last_alive(self):
        combat = NativeCombatEnv(seed=71, ascension_level=0, player=PlayerState())
        gremlin = make_monster("GremlinTsundere", StsRandom(73), ascension=0)
        gremlin.move = "SHIELD_GREMLIN_PROTECT"
        gremlin.intent = "DEFEND"
        combat.monsters = [gremlin]

        combat._monster_take_turn(gremlin, 0)

        self.assertEqual(gremlin.move, "SHIELD_GREMLIN_SHIELD_BASH")
        self.assertEqual(gremlin.intent, "ATTACK")

    def test_shield_gremlin_shield_bash_still_attacks_in_v2(self):
        combat = NativeCombatEnv(seed=72, ascension_level=0, player=PlayerState(current_hp=20, max_hp=80))
        gremlin = make_monster("GremlinTsundere", StsRandom(74), ascension=0)
        gremlin.move = "SHIELD_GREMLIN_SHIELD_BASH"
        gremlin.intent = "ATTACK"
        gremlin.move_base_damage = 6
        gremlin.move_hits = 1
        combat.monsters = [gremlin]
        combat.player.block = 5

        combat._monster_take_turn(gremlin, 0)

        self.assertEqual(combat.player.current_hp, 19)
        self.assertEqual(gremlin.move, "SHIELD_GREMLIN_SHIELD_BASH")

    def test_book_of_stabbing_opening_roll_can_choose_single_stab(self):
        book = make_monster("BookOfStabbing", StsRandom(75), ascension=0)

        class FixedRollRng:
            def random(self, upper):
                return 10

        choose_next_move(book, FixedRollRng())

        self.assertEqual(book.move, "BOOK_OF_STABBING_SINGLE_STAB")
        self.assertEqual(book.move_base_damage, 21)
        self.assertEqual(book.move_hits, 1)

    def test_book_of_stabbing_multistab_tracks_stab_count(self):
        book = make_monster("BookOfStabbing", StsRandom(77), ascension=0)

        class FixedRollRng:
            def random(self, upper):
                return 50

        choose_next_move(book, FixedRollRng())
        self.assertEqual(book.move, "BOOK_OF_STABBING_MULTI_STAB")
        self.assertEqual(book.move_hits, 2)

        choose_next_move(book, FixedRollRng())
        self.assertEqual(book.move, "BOOK_OF_STABBING_MULTI_STAB")
        self.assertEqual(book.move_hits, 3)

    def test_book_of_stabbing_unblocked_hits_add_wounds_to_discard_in_v2(self):
        combat = NativeCombatEnv(seed=79, ascension_level=0, player=PlayerState(current_hp=80, max_hp=80))
        book = make_monster("BookOfStabbing", StsRandom(81), ascension=0)
        book.move = "BOOK_OF_STABBING_MULTI_STAB"
        book.intent = "ATTACK"
        book.move_base_damage = 6
        book.move_hits = 3
        book.ai_state["painful_stabs"] = 1
        combat.monsters = [book]
        combat.player.block = 7
        combat.discard_pile = []

        combat._monster_take_turn(book, 0)

        self.assertEqual([card.card_id for card in combat.discard_pile], ["Wound", "Wound"])

    def test_unceasing_top_draws_when_hand_empties_at_zero_energy_in_v2(self):
        combat = NativeCombatEnv(seed=83, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Unceasing Top", "name": "Unceasing Top", "tier": "UNCOMMON", "counter": -1})
        combat.hand = [make_card("Strike_R", uuid="strike-last")]
        combat.draw_pile = [make_card("Pummel", uuid="pummel-draw")]
        combat.discard_pile = []
        combat.player.energy = 1
        target = make_monster("Cultist", StsRandom(85), ascension=0)
        target.current_hp = 1
        combat.monsters = [target]

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Pummel"])

    def test_unceasing_top_does_not_reshuffle_discard_for_same_trigger_in_v2(self):
        combat = NativeCombatEnv(seed=84, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Unceasing Top", "name": "Unceasing Top", "tier": "UNCOMMON", "counter": -1})
        combat.hand = [make_card("Strike_R", uuid="strike-last")]
        combat.draw_pile = []
        combat.discard_pile = [make_card("Heavy Blade", uuid="heavy-discard")]
        combat.player.energy = 1
        target = make_monster("Cultist", StsRandom(86), ascension=0)
        target.current_hp = 20
        combat.monsters = [target]

        combat.play_card(0, 0)

        self.assertEqual(combat.hand, [])
        self.assertEqual([card.card_id for card in combat.draw_pile], [])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Heavy Blade", "Strike_R"])
        self.assertTrue(combat.pending_unceasing_top_draw)

    def test_headbutt_card_select_triggers_unceasing_top_draw_in_v2(self):
        combat = NativeCombatEnv(seed=85, ascension_level=0, player=PlayerState(energy=1))
        combat.relics.append({"relic_id": "Unceasing Top", "name": "Unceasing Top", "tier": "UNCOMMON", "counter": -1})
        target = make_monster("Cultist", StsRandom(87), ascension=0)
        target.current_hp = 20
        combat.monsters = [target]
        combat.hand = [make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = [make_card("Strike_R", uuid="draw-strike")]
        combat.discard_pile = [make_card("Dropkick", uuid="discard-dropkick"), make_card("Defend_R", uuid="discard-defend")]
        combat.exhaust_pile = []

        combat.play_card(0, 0)
        combat.step({"kind": "card_select", "select_index": 0})

        self.assertEqual([card.card_id for card in combat.hand], ["Dropkick"])
        self.assertEqual([card.card_id for card in combat.draw_pile], ["Strike_R"])

    def test_unceasing_top_status_draw_does_not_chain_evolve_in_v2(self):
        combat = NativeCombatEnv(seed=86, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Unceasing Top", "name": "Unceasing Top", "tier": "UNCOMMON", "counter": -1})
        combat.monsters = [make_monster("SpikeSlime_M", StsRandom(88), ascension=0)]
        combat.hand = [make_card("Strike_R", uuid="strike-last")]
        combat.draw_pile = [make_card("Headbutt", uuid="draw-headbutt"), make_card("Slimed", uuid="draw-slimed")]
        combat.discard_pile = []
        combat.player.energy = 1
        combat.player.powers["Evolve"] = 1

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Slimed"])
        self.assertEqual([card.card_id for card in combat.draw_pile], ["Headbutt"])

    def test_gremlin_leader_encounter_spawns_two_adds_in_v2(self):
        monsters = encounter_to_monsters("GremlinLeader", StsRandom(201), StsRandom(201), ascension=0)

        self.assertEqual(len(monsters), 4)
        self.assertEqual(monsters[0].monster_id, "INVALID = 0")
        self.assertEqual(monsters[3].monster_id, "GremlinLeader")
        self.assertTrue(all(monster.monster_id != "GremlinLeader" for monster in monsters[1:3]))
        self.assertTrue(all(monster.ai_state.get("leader_minion") == 1 for monster in monsters[1:3]))
        self.assertTrue(all(monster.power("Angry") > 0 for monster in monsters[1:3] if monster.monster_id == "GremlinWarrior"))

    def test_gremlin_leader_opens_with_encourage_when_two_allies_alive_in_v2(self):
        leader = make_monster("GremlinLeader", StsRandom(203), ascension=0)
        ally_a = make_monster("GremlinFat", StsRandom(205), ascension=0)
        ally_b = make_monster("GremlinWizard", StsRandom(207), ascension=0)
        placeholder = make_monster("INVALID = 0", StsRandom(209), ascension=0)
        group = [placeholder, ally_a, ally_b, leader]
        for monster in group:
            setattr(monster, "_group_ref", group)

        class ZeroRollRng:
            def random(self, *args):
                return 0

        choose_next_move(leader, ZeroRollRng())

        self.assertEqual(leader.move, "GREMLIN_LEADER_ENCOURAGE")
        self.assertEqual(leader.intent, "DEFEND_BUFF")

    def test_gremlin_leader_rally_uses_magic_intent_in_v2(self):
        leader = make_monster("GremlinLeader", StsRandom(211), ascension=0)

        from spirecomm.native_sim_v2.monster_support import _set_move

        _set_move(leader, "GREMLIN_LEADER_RALLY")

        self.assertEqual(leader.intent, "MAGIC")

    def test_gremlin_leader_encourage_buffs_adds_and_self_in_v2(self):
        combat = NativeCombatEnv(seed=213, ascension_level=0, player=PlayerState())
        placeholder = make_monster("INVALID = 0", StsRandom(215), ascension=0)
        ally_a = make_monster("GremlinFat", StsRandom(217), ascension=0)
        ally_b = make_monster("GremlinWizard", StsRandom(219), ascension=0)
        leader = make_monster("GremlinLeader", StsRandom(221), ascension=0)
        leader.move = "GREMLIN_LEADER_ENCOURAGE"
        leader.intent = "DEFEND_BUFF"
        combat.monsters = [placeholder, ally_a, ally_b, leader]
        for monster in combat.monsters:
            setattr(monster, "_group_ref", combat.monsters)

        combat._monster_take_turn(leader, 3)

        self.assertEqual(ally_a.power("Strength"), 3)
        self.assertEqual(ally_b.power("Strength"), 3)
        self.assertEqual(leader.power("Strength"), 3)
        self.assertEqual(ally_a.block, 6)
        self.assertEqual(ally_b.block, 6)


    def test_gremlin_leader_rally_uses_ai_rng_and_prefers_slots_1_2_then_0_in_v2(self):
        combat = NativeCombatEnv(seed=214, ascension_level=0, player=PlayerState())
        slot0 = make_monster("INVALID = 0", StsRandom(2140), ascension=0)
        slot1 = make_monster("INVALID = 0", StsRandom(2141), ascension=0)
        slot2 = make_monster("INVALID = 0", StsRandom(2142), ascension=0)
        leader = make_monster("GremlinLeader", StsRandom(2143), ascension=0)
        leader.move = "GREMLIN_LEADER_RALLY"
        leader.intent = "MAGIC"
        combat.monsters = [slot0, slot1, slot2, leader]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)

        class FixedAiRng:
            def __init__(self):
                self.values = [0, 6, 0, 0]

            def random(self, upper):
                return self.values.pop(0)

        combat.ai_rng = FixedAiRng()
        combat._monster_take_turn(leader, 3)

        self.assertEqual(combat.monsters[1].monster_id, "GremlinWarrior")
        self.assertEqual(combat.monsters[2].monster_id, "GremlinTsundere")
        self.assertEqual(combat.monsters[0].monster_id, "INVALID = 0")
        self.assertEqual(combat.monsters[1].ai_state.get("leader_minion"), 1)
        self.assertEqual(combat.monsters[2].ai_state.get("leader_minion"), 1)
        self.assertEqual(combat.monsters[1].ai_state.get("leader_summoned"), 1)
        self.assertEqual(combat.monsters[1].power("Angry"), 0)

    def test_gremlin_leader_summoned_warrior_hides_angry_in_serialization(self):
        combat = NativeCombatEnv(seed=2144, ascension_level=0, player=PlayerState())
        slot0 = make_monster("INVALID = 0", StsRandom(2145), ascension=0)
        warrior = make_monster("GremlinWarrior", StsRandom(2146), ascension=0)
        warrior.ai_state["leader_minion"] = 1
        warrior.ai_state["leader_summoned"] = 1
        wizard = make_monster("GremlinWizard", StsRandom(2147), ascension=0)
        wizard.ai_state["leader_minion"] = 1
        leader = make_monster("GremlinLeader", StsRandom(2148), ascension=0)
        combat.monsters = [slot0, warrior, wizard, leader]

        state = combat.to_spirecomm_state()
        warrior_powers = state["combat_state"]["monsters"][1]["powers"]

        self.assertGreater(warrior.power("Angry"), 0)
        self.assertFalse(any(power["id"] == "Angry" for power in warrior_powers))

    def test_gremlin_leader_opening_warrior_keeps_angry_in_serialization(self):
        combat = NativeCombatEnv(seed=21441, ascension_level=0, player=PlayerState())
        slot0 = make_monster("INVALID = 0", StsRandom(21442), ascension=0)
        warrior = make_monster("GremlinWarrior", StsRandom(21443), ascension=0)
        warrior.ai_state["leader_minion"] = 1
        wizard = make_monster("GremlinWizard", StsRandom(21444), ascension=0)
        wizard.ai_state["leader_minion"] = 1
        leader = make_monster("GremlinLeader", StsRandom(21445), ascension=0)
        combat.monsters = [slot0, warrior, wizard, leader]

        state = combat.to_spirecomm_state()
        warrior_powers = state["combat_state"]["monsters"][1]["powers"]

        self.assertGreater(warrior.power("Angry"), 0)
        self.assertTrue(any(power["id"] == "Angry" for power in warrior_powers))

    def test_invalid_slot_hides_non_shared_powers_in_serialization(self):
        combat = NativeCombatEnv(seed=2149, ascension_level=0, player=PlayerState())
        invalid = make_monster("INVALID = 0", StsRandom(2150), ascension=0)
        invalid.powers["Vulnerable"] = 1
        collector = make_monster("TheCollector", StsRandom(2151), ascension=0)
        combat.monsters = [invalid, make_monster("INVALID = 0", StsRandom(2152), ascension=0), collector]

        state = combat.to_spirecomm_state()
        invalid_powers = state["combat_state"]["monsters"][0]["powers"]

        self.assertEqual(invalid_powers, [])

    def test_invalid_slot_keeps_strength_and_shared_buffs_in_serialization(self):
        combat = NativeCombatEnv(seed=2153, ascension_level=0, player=PlayerState())
        invalid = make_monster("INVALID = 0", StsRandom(2154), ascension=0)
        invalid.powers["Strength"] = 1
        invalid.powers["Metallicize"] = 6
        invalid.powers["Regenerate"] = 5
        invalid.powers["Vulnerable"] = 2
        collector = make_monster("TheCollector", StsRandom(2155), ascension=0)
        combat.monsters = [invalid, make_monster("INVALID = 0", StsRandom(2156), ascension=0), collector]

        state = combat.to_spirecomm_state()
        invalid_powers = state["combat_state"]["monsters"][0]["powers"]

        self.assertEqual(
            [(power["id"], power["amount"]) for power in invalid_powers],
            [("Strength", 1), ("Metallicize", 6), ("Regenerate", 5)],
        )

    def test_the_champ_opening_move_roll_thresholds_match_sts_in_v2(self):
        class FixedRollRng:
            def __init__(self, roll):
                self.roll = roll

            def random(self, *args):
                return self.roll

        openings = [
            (10, "CHAMP_DEFENSIVE_STANCE", "DEFEND_BUFF"),
            (20, "CHAMP_GLOAT", "BUFF"),
            (45, "CHAMP_FACE_SLAP", "ATTACK_DEBUFF"),
            (80, "CHAMP_HEAVY_SLASH", "ATTACK"),
        ]

        for roll, expected_move, expected_intent in openings:
            champ = make_monster("TheChamp", StsRandom(2157 + roll), ascension=0)
            choose_next_move(champ, FixedRollRng(roll))
            self.assertEqual((champ.move, champ.intent), (expected_move, expected_intent))

    def test_the_champ_serializes_lightspeed_id_and_move_name_in_v2(self):
        combat = NativeCombatEnv(seed=21570, ascension_level=0, player=PlayerState())
        champ = make_monster("TheChamp", StsRandom(21571), ascension=0)

        from spirecomm.native_sim_v2.monster_support import _set_move

        _set_move(champ, "CHAMP_HEAVY_SLASH")
        combat.monsters = [champ]

        monster = combat.to_spirecomm_state()["combat_state"]["monsters"][0]

        self.assertEqual(monster["monster_id"], "Champ")
        self.assertEqual(monster["move_name"], "THE_CHAMP_HEAVY_SLASH")

    def test_the_champ_defensive_stance_grants_metallicize_not_strength_in_v2(self):
        combat = NativeCombatEnv(seed=21572, ascension_level=0, player=PlayerState())
        champ = make_monster("TheChamp", StsRandom(21573), ascension=0)
        champ.move = "CHAMP_DEFENSIVE_STANCE"
        champ.intent = "DEFEND_BUFF"
        combat.monsters = [champ]

        combat._monster_take_turn(champ, 0)

        self.assertEqual(champ.block, 15)
        self.assertEqual(champ.power("Metallicize"), 5)
        self.assertEqual(champ.power("Strength"), 0)

    def test_the_champ_gloat_uses_boss_diff_strength_amount_in_v2(self):
        combat = NativeCombatEnv(seed=215731, ascension_level=0, player=PlayerState())
        champ = make_monster("TheChamp", StsRandom(215732), ascension=0)
        champ.move = "CHAMP_GLOAT"
        champ.intent = "BUFF"
        combat.monsters = [champ]

        combat._monster_take_turn(champ, 0)

        self.assertEqual(champ.power("Strength"), 3)

    def test_the_champ_face_slap_applies_frail_and_vulnerable_in_v2(self):
        combat = NativeCombatEnv(seed=215741, ascension_level=0, player=PlayerState(current_hp=80, max_hp=80))
        champ = make_monster("TheChamp", StsRandom(215742), ascension=0)
        champ.move = "CHAMP_FACE_SLAP"
        champ.intent = "ATTACK_DEBUFF"
        champ.move_base_damage = 12
        champ.move_hits = 1
        combat.monsters = [champ]

        combat._monster_take_turn(champ, 0)

        self.assertEqual(combat.player.current_hp, 68)
        self.assertEqual(combat.player.power("Frail"), 2)
        self.assertEqual(combat.player.power("Vulnerable"), 2)

    def test_the_champ_taunt_and_anger_use_sts_semantics_in_v2(self):
        combat = NativeCombatEnv(seed=21574, ascension_level=0, player=PlayerState(current_hp=80, max_hp=80))
        champ = make_monster("TheChamp", StsRandom(21575), ascension=0)
        combat.monsters = [champ]

        champ.move = "CHAMP_TAUNT"
        champ.intent = "DEBUFF"
        combat._monster_take_turn(champ, 0)

        self.assertEqual(combat.player.current_hp, 80)
        self.assertEqual(combat.player.power("Weakened"), 2)
        self.assertEqual(combat.player.power("Vulnerable"), 2)

        champ.powers["Vulnerable"] = 1
        champ.powers["Weakened"] = 1
        champ.powers["Poison"] = 3
        champ.powers["Shackled"] = 4
        champ.powers["Strength"] = -2
        champ.move = "CHAMP_ANGER"
        champ.intent = "BUFF"

        combat._monster_take_turn(champ, 0)

        self.assertEqual(champ.power("Strength"), 6)
        self.assertEqual(champ.power("Vulnerable"), 0)
        self.assertEqual(champ.power("Weakened"), 0)
        self.assertEqual(champ.power("Poison"), 0)
        self.assertEqual(champ.power("Shackled"), 0)

    def test_dead_monster_does_not_restore_shackled_strength_at_end_of_round_in_v2(self):
        combat = NativeCombatEnv(seed=231, ascension_level=0, player=PlayerState())
        dead = make_monster("Sentry", StsRandom(231), ascension=0)
        ally = make_monster("Sentry", StsRandom(232), ascension=0)
        dead.current_hp = 0
        dead.powers["Shackled"] = 9
        ally.move = "SENTRY_BOLT"
        ally.intent = "DEBUFF"
        combat.monsters = [dead, ally]
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.end_turn_impl()

        self.assertEqual(dead.power("Strength"), 0)
        self.assertEqual(dead.power("Shackled"), 9)

    def test_true_grit_resolves_juggernaut_after_random_exhaust_in_v2(self):
        combat = NativeCombatEnv(seed=2312, ascension_level=0, player=PlayerState())
        left = make_monster("Sentry", StsRandom(2312), ascension=0)
        right = make_monster("Sentry", StsRandom(2313), ascension=0)
        combat.monsters = [left, right]
        combat.player.powers["Juggernaut"] = 5
        combat.hand = [
            make_card("True Grit", uuid="true-grit"),
            make_card("Dazed", uuid="dazed-a"),
            make_card("Dazed", uuid="dazed-b"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        class SequencedRandom:
            def __init__(self):
                self.values = [0, 1]

            def random(self, upper):
                return self.values.pop(0)

            def choice(self, sequence):
                return sequence[self.random(len(sequence) - 1)]

        combat.card_random_rng = SequencedRandom()

        combat.play_card(0, 0)

        self.assertEqual(left.current_hp, left.max_hp)
        self.assertEqual(right.current_hp, right.max_hp - 5)
        self.assertEqual(len(combat.exhaust_pile), 1)

    def test_end_turn_ethereal_feel_no_pain_juggernaut_resolves_after_combust_in_v2(self):
        combat = NativeCombatEnv(seed=2313, ascension_level=0, player=PlayerState(current_hp=20, max_hp=20))
        left = make_monster("Sentry", StsRandom(2400), ascension=0)
        right = make_monster("Sentry", StsRandom(2401), ascension=0)
        left.current_hp = 5
        left.max_hp = max(left.max_hp, 5)
        right.current_hp = 18
        right.max_hp = max(right.max_hp, 18)
        combat.monsters = [left, right]
        combat.hand = [make_card("Dazed", uuid="end-turn-dazed")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.powers["Combust"] = 5
        combat.combust_hp_loss = 1
        combat.player.powers["Feel No Pain"] = 3
        combat.player.powers["Juggernaut"] = 5

        class FixedRandom:
            def random(self, upper):
                return 0

        combat.card_random_rng = FixedRandom()

        combat.end_turn_impl()

        self.assertEqual(left.current_hp, 0)
        self.assertEqual(right.current_hp, 8)

    def test_collector_fourth_turn_is_mega_debuff_in_v2(self):
        collector = make_monster("TheCollector", StsRandom(230), ascension=0)

        class FixedRollRng:
            def random(self, *args):
                return 0

        rng = FixedRollRng()
        choose_next_move(collector, rng)
        choose_next_move(collector, rng)
        choose_next_move(collector, rng)
        choose_next_move(collector, rng)

        self.assertEqual(collector.move, "COLLECTOR_MEGA_DEBUFF")
        self.assertEqual(collector.intent, "STRONG_DEBUFF")

    def test_collector_buff_gives_self_block_and_does_not_spawn_in_v2(self):
        combat = NativeCombatEnv(seed=231, ascension_level=0, player=PlayerState())
        torch_a = make_monster("TorchHead", StsRandom(2310), ascension=0)
        torch_b = make_monster("TorchHead", StsRandom(2311), ascension=0)
        collector = make_monster("TheCollector", StsRandom(2312), ascension=0)
        collector.move = "COLLECTOR_BUFF"
        collector.intent = "BUFF"
        combat.monsters = [torch_a, torch_b, collector]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)

        combat._monster_take_turn(collector, 2)

        self.assertEqual(len(combat.monsters), 3)
        self.assertEqual(torch_a.power("Strength"), 3)
        self.assertEqual(torch_b.power("Strength"), 3)
        self.assertEqual(collector.power("Strength"), 3)
        self.assertEqual(collector.block, 15)

    def test_collector_mega_debuff_applies_all_three_debuffs_in_v2(self):
        combat = NativeCombatEnv(seed=23121, ascension_level=0, player=PlayerState())
        collector = make_monster("TheCollector", StsRandom(23122), ascension=0)
        collector.move = "COLLECTOR_MEGA_DEBUFF"
        collector.intent = "STRONG_DEBUFF"
        combat.monsters = [make_monster("INVALID = 0", StsRandom(23123), ascension=0), make_monster("INVALID = 0", StsRandom(23124), ascension=0), collector]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)

        combat._monster_take_turn(collector, 2)

        self.assertEqual(combat.player.power("Weakened"), 3)
        self.assertEqual(combat.player.power("Vulnerable"), 3)
        self.assertEqual(combat.player.power("Frail"), 3)

    def test_collector_opening_spawn_rolls_next_move_immediately_in_v2(self):
        combat = NativeCombatEnv(seed=2313, ascension_level=0, player=PlayerState())
        collector = make_monster("TheCollector", StsRandom(23130), ascension=0)
        combat.monsters = [make_monster("INVALID = 0", StsRandom(23131), ascension=0), make_monster("INVALID = 0", StsRandom(23132), ascension=0), collector]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)

        class FixedRollRng:
            def random(self, *args):
                return 0

        combat.ai_rng = FixedRollRng()
        combat._monster_take_turn(collector, 2)

        self.assertEqual([monster.monster_id for monster in combat.monsters], ["TorchHead", "TorchHead", "TheCollector"])
        self.assertEqual(collector.move, "COLLECTOR_FIREBALL")
        self.assertEqual(collector.intent, "ATTACK")
        self.assertEqual(collector.ai_state.get("collector_turn"), 2)
        self.assertEqual(collector.move_history[:2], ["COLLECTOR_FIREBALL", "THE_COLLECTOR_SPAWN"])
        self.assertTrue(collector.ai_state.get("skip_end_round_roll"))

    def test_summoned_monsters_gain_philosophers_stone_strength_in_v2(self):
        combat = NativeCombatEnv(seed=2314, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Philosopher's Stone", "id": "Philosopher's Stone", "name": "Philosopher's Stone", "counter": -1, "price": 0, "tier": "BOSS"})
        collector = make_monster("TheCollector", StsRandom(23140), ascension=0)
        combat.monsters = [make_monster("INVALID = 0", StsRandom(23141), ascension=0), make_monster("INVALID = 0", StsRandom(23142), ascension=0), collector]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)

        class FixedRollRng:
            def random(self, *args):
                return 0

        combat.ai_rng = FixedRollRng()
        combat._monster_take_turn(collector, 2)

        self.assertEqual(combat.monsters[0].power("Strength"), 1)
        self.assertEqual(combat.monsters[1].power("Strength"), 1)

    def test_collector_spawn_torch_heads_use_second_hp_roll_in_v2(self):
        combat = NativeCombatEnv(seed=2315, ascension_level=0, player=PlayerState())
        collector = make_monster("TheCollector", StsRandom(23150), ascension=0)
        combat.monsters = [make_monster("INVALID = 0", StsRandom(23151), ascension=0), make_monster("INVALID = 0", StsRandom(23152), ascension=0), collector]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)

        class FixedHpRng:
            def __init__(self, values):
                self.values = list(values)
                self.counter = 0

            def copy(self):
                clone = FixedHpRng(self.values)
                clone.counter = self.counter
                return clone

            def random(self, start, end):
                self.counter += 1
                return self.values.pop(0)

            def randint(self, start, end):
                return self.random(start, end)

        class FixedRollRng:
            def random(self, *args):
                return 0

        combat.monster_hp_rng = FixedHpRng([40, 40, 40, 38])
        combat.ai_rng = FixedRollRng()

        combat._monster_take_turn(collector, 2)

        self.assertEqual([monster.monster_id for monster in combat.monsters], ["TorchHead", "TorchHead", "TheCollector"])
        self.assertEqual([monster.current_hp for monster in combat.monsters[:2]], [38, 40])
        self.assertEqual(combat.monster_hp_rng.counter, 4)

    def test_torch_head_does_not_roll_next_move_after_tackle_in_v2(self):
        combat = NativeCombatEnv(seed=2316, ascension_level=0, player=PlayerState())
        torch = make_monster("TorchHead", StsRandom(23160), ascension=0)
        combat.monsters = [torch]
        torch.move = "TORCH_HEAD_TACKLE"
        torch.intent = "ATTACK"
        starting_ai_counter = combat.ai_rng.counter

        combat.end_turn()

        self.assertEqual(combat.ai_rng.counter, starting_ai_counter)
        self.assertEqual(combat.monsters[0].move, "TORCH_HEAD_TACKLE")

    def test_large_slime_split_inserts_second_child_without_overwriting_later_monsters_in_v2(self):
        combat = NativeCombatEnv(seed=2311, ascension_level=0, player=PlayerState())
        slime = make_monster("AcidSlime_L", StsRandom(23110), ascension=0)
        slime.move = "ACID_SLIME_L_SPLIT"
        slime.intent = "UNKNOWN"
        slime.current_hp = 11
        slime.max_hp = 11
        later = make_monster("FungiBeast", StsRandom(23111), ascension=0)
        combat.monsters = [slime, later]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)

        combat._monster_take_turn(slime, 0)

        self.assertEqual([monster.monster_id for monster in combat.monsters[:3]], ["AcidSlime_M", "AcidSlime_M", "FungiBeast"])

    def test_gremlin_leader_rally_rolls_moves_after_both_summons_in_v2(self):
        combat = NativeCombatEnv(seed=232, ascension_level=0, player=PlayerState())
        slot0 = make_monster("INVALID = 0", StsRandom(2320), ascension=0)
        slot1 = make_monster("INVALID = 0", StsRandom(2321), ascension=0)
        slot2 = make_monster("INVALID = 0", StsRandom(2322), ascension=0)
        leader = make_monster("GremlinLeader", StsRandom(2323), ascension=0)
        leader.move = "GREMLIN_LEADER_RALLY"
        leader.intent = "MAGIC"
        combat.monsters = [slot0, slot1, slot2, leader]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)

        class ScriptedAiRng:
            def __init__(self):
                self.values = [2, 7, 99, 0]

            def random(self, upper):
                return self.values.pop(0)

        combat.ai_rng = ScriptedAiRng()
        combat._monster_take_turn(leader, 3)

        self.assertEqual(combat.monsters[1].monster_id, "GremlinThief")
        self.assertEqual(combat.monsters[1].move, "SNEAKY_GREMLIN_PUNCTURE")
        self.assertEqual(combat.monsters[2].monster_id, "GremlinWizard")
        self.assertEqual(combat.monsters[2].move, "GREMLIN_WIZARD_CHARGING")

    def test_chosen_and_byrds_encounter_uses_single_byrd_in_v2(self):
        monster_ids = encounter_to_monster_ids("ChosenAndByrds", StsRandom(223))
        self.assertEqual(monster_ids, ["Byrd", "Chosen"])

    def test_bronze_automaton_encounter_starts_with_empty_side_slots_in_v2(self):
        monster_ids = encounter_to_monster_ids("BronzeAutomaton", StsRandom(224))
        self.assertEqual(monster_ids, ["INVALID = 0", "BronzeAutomaton", "INVALID = 0"])

    def test_large_slime_split_inserts_child_without_overwriting_following_slots_in_v2(self):
        combat = NativeCombatEnv(seed=225, ascension_level=0, player=PlayerState())
        acid = make_monster("AcidSlime_L", StsRandom(227), ascension=0)
        acid.current_hp = 20
        acid.move = "ACID_SLIME_L_SPLIT"
        acid.intent = "UNKNOWN"
        ally = make_monster("SpikeSlime_M", StsRandom(229), ascension=0)
        placeholder = make_monster("INVALID = 0", StsRandom(231), ascension=0)
        combat.monsters = [ally, acid, placeholder]

        combat._monster_take_turn(acid, 1)

        self.assertEqual(
            [monster.monster_id for monster in combat.monsters],
            ["SpikeSlime_M", "AcidSlime_M", "AcidSlime_M", "INVALID = 0"],
        )

    def test_panache_counter_resets_at_start_of_turn_in_v2(self):
        combat = NativeCombatEnv(seed=226, ascension_level=0, player=PlayerState())
        combat.player.add_power("Panache", 10)
        combat.panache_counter = 1
        combat.hand = [make_card("Defend_R", uuid="defend")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.monsters = [make_monster("Cultist", StsRandom(2261), ascension=0)]
        combat.monsters[0].current_hp = 40
        combat.monsters[0].max_hp = 40

        combat.end_turn()

        self.assertEqual(combat.turn, 1)
        self.assertEqual(combat.panache_counter, 5)
        self.assertEqual(combat.monsters[0].current_hp, 40)

    def test_panache_gained_mid_turn_triggers_on_next_card_in_v2(self):
        combat = NativeCombatEnv(seed=2261, ascension_level=0, player=PlayerState())
        target = make_monster("Cultist", StsRandom(2262), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.hand = [
            make_card("Panache", uuid="panache"),
            make_card("Strike_R", uuid="strike"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.player.power("Panache"), 10)
        self.assertEqual(combat.panache_counter, 0)
        self.assertEqual(combat.monsters[0].current_hp, 40)

        combat.play_card(0, 0)

        self.assertEqual(combat.panache_counter, -1)
        self.assertEqual(combat.monsters[0].current_hp, 24)

    def test_spike_slime_s_noop_roll_move_still_advances_ai_rng_when_killed_by_flame_barrier_in_v2(self):
        combat = NativeCombatEnv(seed=226, ascension_level=0, player=PlayerState())
        spike = make_monster("SpikeSlime_S", StsRandom(228), ascension=0)
        spike.current_hp = 4
        spike.max_hp = 4
        spike.move = "SPIKE_SLIME_S_TACKLE"
        spike.intent = "ATTACK"
        spike.move_base_damage = 5
        spike.move_hits = 1
        spike.move_history = ["SPIKE_SLIME_S_TACKLE", "SPIKE_SLIME_S_TACKLE"]
        spike.powers["Vulnerable"] = 1

        acid = make_monster("AcidSlime_M", StsRandom(229), ascension=0)
        acid.move = "ACID_SLIME_M_LICK"
        acid.intent = "DEBUFF"
        acid.move_base_damage = 0
        acid.move_hits = 0
        acid.move_history = ["ACID_SLIME_M_LICK", "ACID_SLIME_M_TACKLE"]

        combat.monsters = [spike, acid]
        combat.player.current_hp = 50
        combat.player.max_hp = 50
        combat.player.block = 0
        combat.player.powers["Flame Barrier"] = 4
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.ai_rng.counter = 5
        combat.ai_rng.seed0 = 18220029330099313165
        combat.ai_rng.seed1 = 12818765274207239473

        combat.end_turn()

        self.assertEqual(combat.ai_rng.counter, 7)
        self.assertEqual(combat.monsters[0].current_hp, 0)
        self.assertEqual(combat.monsters[1].move, "ACID_SLIME_M_TACKLE")

    def test_bronze_automaton_spawn_orbs_fills_side_slots_and_sets_flail_in_v2(self):
        combat = NativeCombatEnv(seed=227, ascension_level=0, player=PlayerState())
        automaton = make_monster("BronzeAutomaton", StsRandom(232), ascension=0)
        combat.monsters = [
            make_monster("INVALID = 0", StsRandom(233), ascension=0),
            automaton,
            make_monster("INVALID = 0", StsRandom(234), ascension=0),
        ]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["ascension_level"] = 0
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)

        combat._monster_take_turn(automaton, 1)

        self.assertEqual([monster.monster_id for monster in combat.monsters], ["BronzeOrb", "BronzeAutomaton", "BronzeOrb"])
        self.assertEqual(combat.monsters[1].move, "BRONZE_AUTOMATON_FLAIL")
        self.assertIn(combat.monsters[0].move, {"BRONZE_ORB_BEAM", "BRONZE_ORB_STASIS", "BRONZE_ORB_SUPPORT_BEAM"})
        self.assertIn(combat.monsters[2].move, {"BRONZE_ORB_BEAM", "BRONZE_ORB_STASIS", "BRONZE_ORB_SUPPORT_BEAM"})

    def test_bronze_automaton_spawn_orbs_preserves_slot_strength_and_adds_philosophers_stone_bonus_in_v2(self):
        combat = NativeCombatEnv(
            seed=228,
            ascension_level=0,
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Philosopher's Stone", "id": "Philosopher's Stone", "name": "Philosopher's Stone", "counter": -1, "price": 0, "tier": "BOSS"},
            ],
        )
        automaton = make_monster("BronzeAutomaton", StsRandom(235), ascension=0)
        left = make_monster("INVALID = 0", StsRandom(236), ascension=0)
        right = make_monster("INVALID = 0", StsRandom(237), ascension=0)
        left.add_power("Strength", 1)
        right.add_power("Strength", 1)
        combat.monsters = [left, automaton, right]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["ascension_level"] = 0
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)

        combat._monster_take_turn(automaton, 1)

        self.assertEqual(combat.monsters[0].power("Strength"), 2)
        self.assertEqual(combat.monsters[2].power("Strength"), 2)

    def test_bronze_automaton_spawn_orbs_skips_new_orb_turns_until_next_round_in_v2(self):
        combat = NativeCombatEnv(seed=229, ascension_level=0, player=PlayerState())
        automaton = make_monster("BronzeAutomaton", StsRandom(239), ascension=0)
        combat.monsters = [
            make_monster("INVALID = 0", StsRandom(240), ascension=0),
            automaton,
            make_monster("INVALID = 0", StsRandom(241), ascension=0),
        ]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["ascension_level"] = 0
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)
        combat.draw_pile = [make_card("Headbutt", uuid="headbutt"), make_card("Bludgeon", uuid="bludgeon")]

        extra_roll_index = combat._monster_take_turn(automaton, 1)

        self.assertEqual(extra_roll_index, 2)
        self.assertNotIn("stasis_card", combat.monsters[0].ai_state)
        self.assertNotIn("stasis_card", combat.monsters[2].ai_state)

    def test_bronze_orb_stasis_rolls_new_move_immediately_and_marks_stasis_used_in_v2(self):
        orb = make_monster("BronzeOrb", StsRandom(242), ascension=0)
        orb.move = "BRONZE_ORB_STASIS"
        orb.move_history = ["BRONZE_ORB_STASIS"]
        orb.ai_state["ascension_level"] = 0
        combat = NativeCombatEnv(seed=230, ascension_level=0, player=PlayerState())
        combat.monsters = [orb]
        combat.draw_pile = [make_card("Strike_R", uuid="stolen-strike")]

        combat._monster_take_turn(orb, 0)

        self.assertEqual(orb.ai_state.get("bronze_orb_used_stasis"), 1)
        self.assertIn(orb.move, {"BRONZE_ORB_BEAM", "BRONZE_ORB_SUPPORT_BEAM"})
        self.assertTrue(orb.ai_state.get("skip_end_round_roll"))

    def test_bronze_orb_choose_next_move_consumes_one_ai_roll_in_v2(self):
        orb = make_monster("BronzeOrb", StsRandom(243), ascension=0)
        orb.move = "BRONZE_ORB_BEAM"
        orb.move_history = ["BRONZE_ORB_BEAM", "BRONZE_ORB_STASIS"]
        orb.ai_state["ascension_level"] = 0
        orb.ai_state["bronze_orb_used_stasis"] = 1
        rng = StsRandom(244)
        counter_before = rng.counter

        choose_next_move(orb, rng)

        self.assertEqual(rng.counter, counter_before + 1)

    def test_mind_blast_uses_draw_pile_size_not_deck_size_in_v2(self):
        combat = NativeCombatEnv(seed=226, ascension_level=0, player=PlayerState())
        target = make_monster("SpikeSlime_S", StsRandom(228), ascension=0)
        target.current_hp = 12
        target.max_hp = 12
        combat.monsters = [target]
        combat.draw_pile = [make_card("Strike_R") for _ in range(6)]
        combat.deck = [make_card("Strike_R") for _ in range(11)]
        card = make_card("Mind Blast")

        combat.replay_attack_card_effect_impl(card, target, 0)

        self.assertEqual(target.current_hp, 6)

    def test_the_boot_applies_after_monster_block_in_v2(self):
        combat = NativeCombatEnv(seed=227, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "The Boot", "name": "The Boot", "tier": "COMMON", "counter": -1})
        target = make_monster("JawWorm", StsRandom(229), ascension=0)
        target.current_hp = 22
        target.max_hp = 22
        target.block = 6
        target.add_power("Vulnerable", 1)
        combat.monsters = [target]

        combat._deal_attack_damage(6, target)

        self.assertEqual(target.block, 0)
        self.assertEqual(target.current_hp, 17)

    def test_runic_dome_does_not_hide_monster_move_metadata_in_v2(self):
        combat = NativeCombatEnv(seed=215, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Runic Dome", "name": "Runic Dome", "tier": "BOSS", "counter": -1})
        state = combat.to_spirecomm_state()
        monster = state["combat_state"]["monsters"][0]

        self.assertIsNotNone(monster["move_name"])
        self.assertIsNotNone(monster["move_id"])
        self.assertNotEqual(monster["intent"], "UNKNOWN")

    def test_dead_adventurer_event_combat_skips_preserved_insect_hp_reduction_in_v2(self):
        env = NativeRunEnv(seed=217, ascension_level=0, start_on_map=True)
        env.relics.append({"relic_id": "Preserved Insect", "name": "Preserved Insect", "tier": "COMMON", "counter": -1})

        env._start_event_combat(["Sentry", "Sentry", "Sentry"], elite=False)

        insect_hp = [monster.max_hp for monster in env.combat.monsters]

        control = NativeRunEnv(seed=217, ascension_level=0, start_on_map=True)
        control._start_event_combat(["Sentry", "Sentry", "Sentry"], elite=False)
        control_hp = [monster.max_hp for monster in control.combat.monsters]

        self.assertEqual(insect_hp, control_hp)

    def test_preserved_insect_keeps_elite_max_hp_intact_in_v2(self):
        relics = [
            {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
            {"relic_id": "Preserved Insect", "id": "Preserved Insect", "name": "Preserved Insect", "counter": -1, "price": 0, "tier": "COMMON"},
        ]
        combat = NativeCombatEnv(
            seed=218,
            ascension_level=0,
            scheduled_encounter=["Lagavulin"],
            elite=True,
            player=PlayerState(),
            relics=relics,
        )
        control = NativeCombatEnv(
            seed=218,
            ascension_level=0,
            scheduled_encounter=["Lagavulin"],
            elite=True,
            player=PlayerState(),
        )

        self.assertEqual(combat.monsters[0].max_hp, control.monsters[0].max_hp)
        self.assertEqual(combat.monsters[0].current_hp, max(1, int(control.monsters[0].current_hp * 0.75)))

    def test_preserved_insect_applies_after_green_elite_hp_buff_in_v2(self):
        env = NativeRunEnv(seed=219, ascension_level=0, start_on_map=True)
        env.relics.append({"relic_id": "Preserved Insect", "name": "Preserved Insect", "tier": "COMMON", "counter": -1})
        env.current_node_symbol = "E_GREEN"
        env.current_map_node_id = "elite-test"
        env.map_graph = {"elite-test": {"burning_elite_buff": 1}}
        baseline = NativeCombatEnv(
            seed=219,
            ascension_level=0,
            scheduled_encounter=["Sentry", "Sentry", "Sentry"],
            elite=True,
            player=PlayerState(),
            relics=[{"relic_id": "Burning Blood", "name": "Burning Blood", "tier": "STARTER", "counter": -1}],
        )
        env.combat = NativeCombatEnv(
            seed=219,
            ascension_level=0,
            scheduled_encounter=["Sentry", "Sentry", "Sentry"],
            elite=True,
            player=PlayerState(),
            relics=list(env.relics),
        )

        env._apply_burning_elite_buff()

        for buffed, control in zip(env.combat.monsters, baseline.monsters):
            expected_max = control.max_hp + int(float(control.max_hp) * 0.25 + 0.5)
            self.assertEqual(buffed.max_hp, expected_max)
            self.assertEqual(buffed.current_hp, max(1, int(buffed.max_hp * 0.75)))

    def test_burning_elite_strength_buff_is_act_plus_one_in_v2(self):
        env = NativeRunEnv(seed=220, ascension_level=0, start_on_map=True)
        env.current_node_symbol = "E_GREEN"
        env.current_map_node_id = "elite-test"
        env.map_graph = {"elite-test": {"burning_elite_buff": 0}}
        env.combat = NativeCombatEnv(
            seed=220,
            ascension_level=0,
            scheduled_encounter=["GremlinNob"],
            elite=True,
            player=PlayerState(),
        )

        env._apply_burning_elite_buff()

        self.assertEqual(env.combat.monsters[0].power("Strength"), 2)

    def test_thunderclap_does_not_apply_vulnerable_to_dead_monsters_in_v2(self):
        combat = NativeCombatEnv(seed=223, ascension_level=0, player=PlayerState())
        dead = make_monster("RedLouse", StsRandom(225), ascension=0)
        alive = make_monster("RedLouse", StsRandom(227), ascension=0)
        dead.current_hp = 4
        dead.max_hp = 4
        alive.current_hp = 10
        alive.max_hp = 10
        combat.monsters = [dead, alive]
        combat.hand = [make_card("Thunderclap", uuid="tc")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertFalse(dead.alive)
        self.assertEqual(dead.power("Vulnerable"), 0)
        self.assertEqual(alive.power("Vulnerable"), 1)

    def test_slime_boss_slam_keeps_split_when_flame_barrier_pushes_below_half(self):
        combat = NativeCombatEnv(seed=79, ascension_level=0, player=PlayerState())
        boss = make_monster("SlimeBoss", StsRandom(81), ascension=0)
        boss.current_hp = 71
        boss.max_hp = 140
        boss.move = "SLIME_BOSS_SLAM"
        boss.intent = "ATTACK"
        boss.move_base_damage = 35
        boss.move_hits = 1
        combat.monsters = [boss]
        combat.player.current_hp = 80
        combat.player.block = 0
        combat.player.powers["Flame Barrier"] = 4

        combat._monster_take_turn(boss, 0)

        self.assertEqual(boss.move, "SLIME_BOSS_SPLIT")

    def test_spike_slime_large_split_rewrites_move_history_head_in_v2(self):
        combat = NativeCombatEnv(seed=80, ascension_level=0, player=PlayerState())
        slime = make_monster("SpikeSlime_L", StsRandom(82), ascension=0)
        slime.current_hp = 36
        slime.max_hp = 64
        slime.move = "SPIKE_SLIME_L_FLAME_TACKLE"
        slime.intent = "ATTACK_DEBUFF"
        slime.move_base_damage = 16
        slime.move_hits = 1
        slime.move_history = ["SPIKE_SLIME_L_FLAME_TACKLE", "SPIKE_SLIME_L_FLAME_TACKLE"]
        combat.monsters = [slime]
        combat.player.current_hp = 80
        combat.player.block = 0
        combat.player.powers["Flame Barrier"] = 4

        combat._monster_take_turn(slime, 0)

        self.assertEqual(slime.move, "SPIKE_SLIME_L_SPLIT")
        self.assertEqual(slime.move_history[0], "SPIKE_SLIME_L_SPLIT")
        self.assertEqual(slime.move_history[1], "SPIKE_SLIME_L_FLAME_TACKLE")

    def test_spike_slime_l_roll_move_does_not_force_split_below_half_in_v2(self):
        slime = make_monster("SpikeSlime_L", StsRandom(245), ascension=0)
        slime.current_hp = 32
        slime.max_hp = 68
        slime.move = "SPIKE_SLIME_L_SPLIT"
        slime.move_history = ["SPIKE_SLIME_L_FLAME_TACKLE", "SPIKE_SLIME_L_LICK"]
        slime.ai_state["ascension_level"] = 0
        choose_next_move(slime, StsRandom(246))

        self.assertIn(slime.move, {"SPIKE_SLIME_L_FLAME_TACKLE", "SPIKE_SLIME_L_LICK"})

    def test_acid_slime_l_can_repeat_tackle_at_split_threshold_in_v2(self):
        slime = make_monster("AcidSlime_L", StsRandom(2451), ascension=0)
        slime.current_hp = 32
        slime.max_hp = 65
        slime.move = "ACID_SLIME_L_TACKLE"
        slime.intent = "ATTACK"
        slime.move_base_damage = 16
        slime.move_hits = 1
        slime.move_history = ["ACID_SLIME_L_TACKLE", "ACID_SLIME_L_LICK"]
        slime.ai_state["ascension_level"] = 0

        class FixedRollRng:
            def random(self, upper):
                return 50

            def random_boolean(self, chance=None):
                return False

        choose_next_move(slime, FixedRollRng())

        self.assertEqual(slime.move, "ACID_SLIME_L_TACKLE")

    def test_spike_slime_l_split_keeps_trailing_acid_slime_l_ai_roll_order_in_v2(self):
        combat = NativeCombatEnv(seed=2452, ascension_level=0, player=PlayerState())
        spike = make_monster("SpikeSlime_L", StsRandom(24520), ascension=0)
        spike.current_hp = 23
        spike.max_hp = 65
        spike.move = "SPIKE_SLIME_L_SPLIT"
        spike.intent = "UNKNOWN"

        acid = make_monster("AcidSlime_L", StsRandom(24521), ascension=0)
        acid.current_hp = 62
        acid.max_hp = 65
        acid.move = "ACID_SLIME_L_TACKLE"
        acid.intent = "ATTACK"
        acid.move_base_damage = 16
        acid.move_hits = 1
        acid.move_history = ["ACID_SLIME_L_TACKLE", "ACID_SLIME_L_LICK"]
        acid.ai_state["ascension_level"] = 0

        combat.monsters = [spike, acid]

        class ScriptedAiRng:
            def __init__(self):
                self.rolls = [80, 0, 99, 50]

            def random(self, upper):
                return self.rolls.pop(0)

            def random_boolean(self, chance=None):
                return False

        combat.ai_rng = ScriptedAiRng()
        combat.end_turn()

        self.assertEqual(
            [monster.monster_id for monster in combat.monsters[:3]],
            ["SpikeSlime_M", "SpikeSlime_M", "AcidSlime_L"],
        )
        self.assertEqual(combat.monsters[0].move, "SPIKE_SLIME_M_LICK")
        self.assertEqual(combat.monsters[1].move, "SPIKE_SLIME_M_FLAME_TACKLE")
        self.assertEqual(combat.monsters[2].move, "ACID_SLIME_L_LICK")

    def test_slime_boss_roll_move_does_not_force_split_below_half_in_v2(self):
        boss = make_monster("SlimeBoss", StsRandom(247), ascension=0)
        boss.current_hp = 70
        boss.max_hp = 140
        boss.move = "SLIME_BOSS_SPLIT"
        boss.move_history = ["SLIME_BOSS_SLAM", "SLIME_BOSS_PREPARING"]
        boss.ai_state["ascension_level"] = 0

        choose_next_move(boss, StsRandom(248))

        self.assertEqual(boss.move, "SLIME_BOSS_GOOP_SPRAY")

    def test_direct_damage_does_not_consume_plated_armor_in_v2(self):
        combat = NativeCombatEnv(seed=57, ascension_level=0, player=PlayerState())
        parasite = make_monster("ShelledParasite", StsRandom(59), ascension=0)
        parasite.current_hp = 69
        parasite.max_hp = 69
        parasite.block = 1
        parasite.powers["Plated Armor"] = 14
        combat.monsters = [parasite]

        dealt = combat._deal_direct_damage_to_monster(5, parasite)

        self.assertEqual(dealt, 4)
        self.assertEqual(parasite.current_hp, 65)
        self.assertEqual(parasite.block, 0)
        self.assertEqual(parasite.power("Plated Armor"), 14)

    def test_attack_damage_consumes_monster_plated_armor_in_v2(self):
        combat = NativeCombatEnv(seed=5701, ascension_level=0, player=PlayerState())
        parasite = make_monster("ShelledParasite", StsRandom(5901), ascension=0)
        parasite.current_hp = 69
        parasite.max_hp = 69
        parasite.block = 1
        parasite.powers["Plated Armor"] = 14
        combat.monsters = [parasite]
        combat.hand = [make_card("Strike_R", uuid="strike")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(parasite.current_hp, 64)
        self.assertEqual(parasite.block, 0)
        self.assertEqual(parasite.power("Plated Armor"), 13)

    def test_thorns_damage_to_monster_respects_block_in_v2(self):
        combat = NativeCombatEnv(seed=5902, ascension_level=0, player=PlayerState())
        parasite = make_monster("ShelledParasite", StsRandom(5903), ascension=0)
        parasite.current_hp = 59
        parasite.max_hp = 69
        parasite.move = "SHELLED_SUCK"
        parasite.intent = "ATTACK"
        parasite.move_base_damage = 10
        parasite.move_hits = 1
        combat.monsters = [parasite]
        combat.player.current_hp = 80
        combat.player.block = 0
        combat.player.powers["Thorns"] = 3

        combat._monster_take_turn(parasite, 0)

        self.assertEqual(parasite.current_hp, 69)
        self.assertEqual(parasite.block, 11)

    def test_spheric_guardian_attack_debuff_thorns_hits_block_before_hp_in_v2(self):
        combat = NativeCombatEnv(seed=5903, ascension_level=0, player=PlayerState())
        guardian = make_monster("SphericGuardian", StsRandom(5904), ascension=0)
        guardian.block = 65
        guardian.move = "SPHERIC_GUARDIAN_ATTACK_DEBUFF"
        guardian.intent = "ATTACK_DEBUFF"
        guardian.move_base_damage = 10
        guardian.move_hits = 1
        guardian.move_history = ["SPHERIC_GUARDIAN_ACTIVATE"]
        combat.monsters = [guardian]
        combat.player.current_hp = 61
        combat.player.max_hp = 80
        combat.player.block = 10
        combat.player.powers["Thorns"] = 3

        combat._monster_take_turn(guardian, 0)

        self.assertEqual(guardian.current_hp, 20)
        self.assertEqual(guardian.block, 62)
        self.assertEqual(guardian.move, "SPHERIC_GUARDIAN_SLAM")
        self.assertEqual(guardian.intent, "ATTACK")

    def test_spheric_guardian_harden_gains_block_before_thorns_in_v2(self):
        combat = NativeCombatEnv(seed=5904, ascension_level=0, player=PlayerState())
        guardian = make_monster("SphericGuardian", StsRandom(5905), ascension=0)
        guardian.block = 0
        guardian.current_hp = 18
        guardian.max_hp = 20
        guardian.move = "SPHERIC_GUARDIAN_HARDEN"
        guardian.intent = "ATTACK_DEFEND"
        guardian.move_base_damage = 10
        guardian.move_hits = 1
        guardian.move_history = ["SPHERIC_GUARDIAN_ACTIVATE", "SPHERIC_GUARDIAN_SLAM"]
        combat.monsters = [guardian]
        combat.player.current_hp = 13
        combat.player.max_hp = 80
        combat.player.block = 25
        combat.player.powers["Thorns"] = 3

        combat._monster_take_turn(guardian, 0)

        self.assertEqual(guardian.current_hp, 18)
        self.assertEqual(guardian.block, 12)
        self.assertEqual(guardian.move, "SPHERIC_GUARDIAN_SLAM")
        self.assertEqual(guardian.intent, "ATTACK")

    def test_shelled_suck_heals_by_actual_hp_loss_after_torii_in_v2(self):
        combat = NativeCombatEnv(seed=5904, ascension_level=0, player=PlayerState())
        parasite = make_monster("ShelledParasite", StsRandom(5905), ascension=0)
        parasite.current_hp = 60
        parasite.max_hp = 69
        parasite.move = "SHELLED_SUCK"
        parasite.intent = "ATTACK"
        parasite.move_base_damage = 10
        parasite.move_hits = 1
        combat.monsters = [parasite]
        combat.player.current_hp = 80
        combat.player.block = 5
        combat.relics.append({"relic_id": "Torii", "name": "Torii", "tier": "RARE"})

        combat._monster_take_turn(parasite, 0)

        self.assertEqual(combat.player.current_hp, 79)
        self.assertEqual(combat.player.block, 0)
        self.assertEqual(parasite.current_hp, 61)

    def test_shelled_suck_heals_before_thorns_retaliation_in_v2(self):
        combat = NativeCombatEnv(seed=5906, ascension_level=0, player=PlayerState())
        parasite = make_monster("ShelledParasite", StsRandom(5907), ascension=0)
        parasite.current_hp = 67
        parasite.max_hp = 69
        parasite.move = "SHELLED_SUCK"
        parasite.intent = "ATTACK"
        parasite.move_base_damage = 10
        parasite.move_hits = 1
        combat.monsters = [parasite]
        combat.player.current_hp = 80
        combat.player.block = 5
        combat.player.powers["Thorns"] = 3

        combat._monster_take_turn(parasite, 0)

        self.assertEqual(combat.player.current_hp, 75)
        self.assertEqual(combat.player.block, 0)
        self.assertEqual(parasite.current_hp, 69)
        self.assertEqual(parasite.block, 11)

    def test_dead_shelled_parasite_does_not_roll_next_move_after_suck_in_v2(self):
        combat = NativeCombatEnv(seed=5908, ascension_level=0, scheduled_encounter=["ShelledParasite", "FungiBeast"], player=PlayerState())
        parasite = combat.monsters[0]
        fungi = combat.monsters[1]
        parasite.current_hp = 1
        parasite.max_hp = 69
        parasite.move = "SHELLED_SUCK"
        parasite.intent = "ATTACK"
        parasite.move_base_damage = 10
        parasite.move_hits = 1
        fungi.move = "FUNGI_BEAST_BITE"
        fungi.intent = "ATTACK"
        fungi.move_base_damage = 0
        fungi.move_hits = 1
        combat.player.current_hp = 80
        combat.player.block = 5
        combat.player.powers["Thorns"] = 3
        combat.relics.append({"relic_id": "Torii", "name": "Torii", "tier": "RARE"})

        combat.end_turn()

        self.assertEqual(parasite.current_hp, 0)
        self.assertEqual(parasite.move, "SHELLED_SUCK")

    def test_purity_enters_exhaust_many_confirm_state_in_v2(self):
        combat = NativeCombatEnv(seed=61, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Purity", uuid="purity"),
            make_card("Strike_R", uuid="strike-1"),
            make_card("Strike_R", uuid="strike-2"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 0

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Strike_R", "Strike_R"])
        self.assertEqual(combat.card_select_context, "EXHAUST_MANY")
        self.assertEqual(combat.exhaust_pile, [])
        self.assertEqual(len(combat.card_select_options), 1)
        self.assertEqual(combat.card_select_options[0]["kind"], "multi_card_select")

        combat.step(combat.card_select_options[0])

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Purity"])

    def test_perfected_strike_ignores_exhaust_pile_strikes_in_v2(self):
        combat = NativeCombatEnv(seed=63, ascension_level=0, player=PlayerState())
        target = make_monster("SlaverRed", StsRandom(65), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.hand = [
            make_card("Strike_R", uuid="strike"),
            make_card("Perfected Strike", uuid="perfected"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = [make_card("Strike_R", uuid="exhausted-strike")]
        combat.player.energy = 2
        combat.player.powers["Strength"] = 2
        combat.deck = []

        combat.play_card(1, 0)

        self.assertEqual(combat.monsters[0].current_hp, 28)

    def test_uppercut_uses_lightspeed_damage_and_debuff_order_in_v2(self):
        combat = NativeCombatEnv(seed=104, ascension_level=0, player=PlayerState())
        target = make_monster("SlaverRed", StsRandom(106), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        target.powers["Artifact"] = 1
        combat.monsters = [target]
        combat.hand = [make_card("Uppercut", uuid="uppercut")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)

        self.assertEqual(combat.monsters[0].current_hp, 27)
        self.assertEqual(combat.monsters[0].power("Artifact"), 0)
        self.assertEqual(combat.monsters[0].power("Weakened"), 0)
        self.assertEqual(combat.monsters[0].power("Vulnerable"), 1)

    def test_letter_opener_damage_waits_until_armaments_selection_resolves_in_v2(self):
        combat = NativeCombatEnv(seed=105, ascension_level=0, player=PlayerState())
        target = make_monster("SlaverRed", StsRandom(107), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.relics = [
            {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
            {"relic_id": "Letter Opener", "id": "Letter Opener", "name": "Letter Opener", "counter": -1, "price": 0, "tier": "UNCOMMON"},
        ]
        combat.skills_played_this_turn = 2
        combat.cards_played_this_turn = 2
        combat.hand = [
            make_card("Armaments", uuid="armaments"),
            make_card("Strike_R", uuid="strike"),
            make_card("Defend_R", uuid="defend"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "ARMAMENTS")
        self.assertEqual(combat.monsters[0].current_hp, 40)

        combat.step(combat.card_select_options[0])

        self.assertEqual(combat.monsters[0].current_hp, 35)

    def test_letter_opener_triggers_in_single_card_burning_pact_branch_in_v2(self):
        combat = NativeCombatEnv(seed=106, ascension_level=0, player=PlayerState())
        monsters = [
            make_monster("SpikeSlime_M", StsRandom(108), ascension=0),
            make_monster("SpikeSlime_M", StsRandom(109), ascension=0),
            make_monster("AcidSlime_L", StsRandom(110), ascension=0),
        ]
        for monster in monsters:
            monster.current_hp = 30
            monster.max_hp = 30
        combat.monsters = monsters
        combat.relics = [
            {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
            {"relic_id": "Letter Opener", "id": "Letter Opener", "name": "Letter Opener", "counter": -1, "price": 0, "tier": "UNCOMMON"},
        ]
        combat.skills_played_this_turn = 2
        combat.cards_played_this_turn = 2
        combat.hand = [
            make_card("Burning Pact", uuid="burning-pact"),
            make_card("Strike_R", uuid="only-target"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual([monster.current_hp for monster in combat.monsters], [25, 25, 25])

    def test_dead_blue_slaver_still_consumes_roll_before_red_slaver_in_v2(self):
        from spirecomm.native_sim_v2.monster_support import _set_move

        combat = NativeCombatEnv(seed=181, ascension_level=0, player=PlayerState())
        blue = make_monster("SlaverBlue", V2StsRandom(1), ascension=0)
        taskmaster = make_monster("Taskmaster", V2StsRandom(2), ascension=0)
        red = make_monster("SlaverRed", V2StsRandom(3), ascension=0)
        combat.monsters = [blue, taskmaster, red]
        for index, monster in enumerate(combat.monsters):
            monster.ai_state["spawn_index"] = index
            setattr(monster, "_group_ref", combat.monsters)

        _set_move(blue, "BLUE_SLAVER_STAB")
        blue.move_history = ["BLUE_SLAVER_STAB", "BLUE_SLAVER_RAKE"]
        blue.current_hp = 3
        blue.max_hp = 47

        _set_move(taskmaster, "TASKMASTER_SCOURING_WHIP")
        taskmaster.move_history = ["TASKMASTER_SCOURING_WHIP"]
        taskmaster.current_hp = 52
        taskmaster.max_hp = 52

        _set_move(red, "RED_SLAVER_SCRAPE")
        red.move_history = ["RED_SLAVER_SCRAPE", "RED_SLAVER_STAB"]
        red.current_hp = 50
        red.max_hp = 50

        combat.player.current_hp = 40
        combat.player.max_hp = 80
        combat.player.powers["Thorns"] = 3
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.ai_rng = V2StsRandom(0)
        combat.ai_rng.counter = 6
        combat.ai_rng.seed0 = 6485881627134535450
        combat.ai_rng.seed1 = 409589232796021388

        combat.end_turn()

        self.assertFalse(blue.alive)
        self.assertEqual(red.move, "RED_SLAVER_SCRAPE")

    def test_panache_damage_waits_until_armaments_selection_resolves_in_v2(self):
        combat = NativeCombatEnv(seed=106, ascension_level=0, player=PlayerState())
        target = make_monster("SlaverRed", StsRandom(108), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.player.powers["Panache"] = 10
        combat.panache_counter = 1
        combat.hand = [
            make_card("Armaments", uuid="armaments"),
            make_card("Strike_R", uuid="strike"),
            make_card("Defend_R", uuid="defend"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "ARMAMENTS")
        self.assertEqual(combat.monsters[0].current_hp, 40)

        combat.step(combat.card_select_options[0])

        self.assertEqual(combat.monsters[0].current_hp, 30)

    def test_panache_uses_explicit_counter_in_v2(self):
        combat = NativeCombatEnv(seed=1061, ascension_level=0, player=PlayerState())
        target = make_monster("SlaverRed", StsRandom(1062), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.player.powers["Panache"] = 10
        combat.panache_counter = 1
        combat.hand = [make_card("Defend_R", uuid="defend")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.monsters[0].current_hp, 30)

    def test_ink_bottle_draw_waits_until_armaments_selection_resolves_in_v2(self):
        combat = NativeCombatEnv(seed=107, ascension_level=0, player=PlayerState())
        combat.relics = [
            {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
            {"relic_id": "Ink Bottle", "id": "Ink Bottle", "name": "Ink Bottle", "counter": 9, "price": 0, "tier": "UNCOMMON"},
        ]
        combat.hand = [
            make_card("Armaments", uuid="armaments"),
            make_card("Defend_R", uuid="defend-a"),
            make_card("Strike_R", uuid="strike-a"),
            make_card("Defend_R", uuid="defend-b"),
        ]
        combat.draw_pile = [make_card("Defend_R", uuid="draw-defend")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "ARMAMENTS")
        self.assertEqual([card.uuid for card in combat.hand], ["defend-a", "strike-a", "defend-b"])
        self.assertEqual([card.uuid for card in combat.draw_pile], ["draw-defend"])

        combat.step(combat.card_select_options[0])

        self.assertEqual([card.uuid for card in combat.hand], ["strike-a", "defend-b", "defend-a", "draw-defend"])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Armaments"])
        self.assertEqual(combat.draw_pile, [])

    def test_armaments_upgraded_true_grit_opens_exhaust_select_in_v2(self):
        combat = NativeCombatEnv(seed=1071, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Armaments", uuid="armaments"),
            make_card("True Grit", uuid="true-grit"),
            make_card("Dazed", uuid="dazed-a"),
            make_card("Strike_R", uuid="strike"),
            make_card("Dazed", uuid="dazed-b"),
        ]
        combat.draw_pile = [make_card("Defend_R", uuid="draw-defend")]
        combat.discard_pile = []
        combat.exhaust_pile = [make_card("Dazed", uuid="exhausted-dazed")]
        combat.player.energy = 3

        combat.play_card(0, 0)
        combat.step(combat.card_select_options[0])  # upgrade True Grit
        combat.play_card(0, 0)  # play Strike

        self.assertEqual(combat.hand[0].card_id, "True Grit")
        self.assertEqual(combat.hand[0].upgrades, 1)

        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "EXHAUST_ONE")
        self.assertEqual([card.card_id for card in combat.hand], ["Dazed", "Dazed"])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Armaments", "Strike_R"])
        self.assertEqual(len(combat.exhaust_pile), 1)

        combat.step(combat.card_select_options[0])

        self.assertEqual([card.card_id for card in combat.hand], ["Dazed"])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Armaments", "Strike_R", "True Grit"])
        self.assertEqual(len(combat.exhaust_pile), 2)

    def test_pommel_strike_moves_to_discard_before_evolve_shuffle_in_v2(self):
        combat = NativeCombatEnv(seed=1072, ascension_level=0, player=PlayerState())
        target = make_monster("JawWorm", StsRandom(249), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.player.powers["Evolve"] = 1
        combat.hand = [make_card("Pommel Strike", upgrades=1, uuid="pommel")]
        combat.draw_pile = [make_card("Dazed", uuid="draw-dazed"), make_card("Dazed", uuid="draw-dazed-2")]
        combat.discard_pile = [make_card("Strike_R", uuid="discard-strike")]
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        all_locations = (
            [card.uuid for card in combat.hand]
            + [card.uuid for card in combat.draw_pile]
            + [card.uuid for card in combat.discard_pile]
        )
        self.assertIn("pommel", all_locations)

    def test_finesse_moves_to_discard_before_evolve_shuffle_in_v2(self):
        combat = NativeCombatEnv(seed=10721, ascension_level=0, player=PlayerState())
        target = make_monster("JawWorm", StsRandom(250), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.player.powers["Evolve"] = 1
        combat.hand = [make_card("Finesse", uuid="finesse")]
        combat.draw_pile = [make_card("Dazed", uuid="draw-dazed")]
        combat.discard_pile = [make_card("Strike_R", uuid="discard-strike")]
        combat.exhaust_pile = []
        combat.player.energy = 0

        combat.play_card(0, 0)

        all_locations = (
            [card.uuid for card in combat.hand]
            + [card.uuid for card in combat.draw_pile]
            + [card.uuid for card in combat.discard_pile]
        )
        self.assertIn("finesse", all_locations)

    def test_finesse_under_hex_and_evolve_only_inserts_one_dazed_in_v2(self):
        combat = NativeCombatEnv(seed=10723, ascension_level=0, player=PlayerState())
        target = make_monster("Chosen", StsRandom(252), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.player.powers["Hex"] = 1
        combat.player.powers["Evolve"] = 1
        combat.hand = [
            make_card("Iron Wave", uuid="iron-wave"),
            make_card("Dazed", uuid="hand-dazed"),
            make_card("Finesse", uuid="finesse"),
            make_card("Doubt", uuid="doubt"),
            make_card("Strike_R", uuid="strike"),
        ]
        combat.draw_pile = [
            make_card("Bash", uuid="draw-bash"),
            make_card("Strike_R", uuid="draw-strike"),
            make_card("Dazed", uuid="draw-dazed-0"),
            make_card("Dazed", uuid="draw-dazed-1"),
            make_card("Dazed", uuid="draw-dazed-2"),
            make_card("Dazed", uuid="draw-dazed-3"),
            make_card("Dazed", uuid="draw-dazed-4"),
        ]
        combat.discard_pile = [
            make_card("Defend_R", uuid="discard-defend-0"),
            make_card("Defend_R", uuid="discard-defend-1"),
        ]
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(2, 0)

        self.assertEqual(sum(1 for card in combat.draw_pile if card.card_id == "Dazed"), 0)

    def test_finesse_under_hex_and_evolve_places_generated_dazed_before_evolve_extra_draws_in_v2(self):
        combat = NativeCombatEnv(seed=107231, ascension_level=0, player=PlayerState())
        target = make_monster("Chosen", StsRandom(253), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.player.powers["Hex"] = 1
        combat.player.powers["Evolve"] = 2
        combat.hand = [
            make_card("Thunderclap", uuid="thunderclap"),
            make_card("Dazed", uuid="hand-dazed-0"),
            make_card("Dazed", uuid="hand-dazed-1"),
            make_card("Dazed", uuid="hand-dazed-2"),
            make_card("Finesse", uuid="finesse"),
            make_card("Strike_R", uuid="strike-0"),
            make_card("Strike_R", uuid="strike-1"),
            make_card("Strike_R", uuid="strike-2"),
        ]
        combat.draw_pile = [
            make_card("Defend_R", uuid="defend-0"),
            make_card("Bloodletting", uuid="bloodletting"),
            make_card("Defend_R", uuid="defend-1"),
            make_card("Blood for Blood", uuid="b4b"),
            make_card("Strike_R", uuid="draw-strike-0"),
            make_card("Defend_R", uuid="defend-2"),
            make_card("Defend_R", uuid="defend-3"),
            make_card("Bash", uuid="bash"),
            make_card("Perfected Strike", uuid="perfected"),
            make_card("Strike_R", uuid="draw-strike-1"),
            make_card("Dazed", uuid="draw-dazed"),
        ]
        combat.discard_pile = [make_card("Body Slam", uuid="bodyslam"), make_card("Dropkick", uuid="dropkick")]
        combat.exhaust_pile = [make_card("Carnage", uuid="carnage")]
        combat.player.energy = 0

        combat.play_card(4, 0)

        self.assertEqual(
            [card.card_id for card in combat.hand],
            ["Thunderclap", "Dazed", "Dazed", "Dazed", "Strike_R", "Strike_R", "Strike_R", "Dazed", "Strike_R", "Dazed"],
        )
        self.assertEqual(
            [card.card_id for card in combat.draw_pile],
            ["Defend_R", "Bloodletting", "Defend_R", "Blood for Blood", "Strike_R", "Defend_R", "Defend_R", "Bash", "Perfected Strike"],
        )

    def test_flash_of_steel_moves_to_discard_before_evolve_shuffle_in_v2(self):
        combat = NativeCombatEnv(seed=10722, ascension_level=0, player=PlayerState())
        target = make_monster("JawWorm", StsRandom(251), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.player.powers["Evolve"] = 1
        combat.hand = [make_card("Flash of Steel", uuid="flashofsteel")]
        combat.draw_pile = [make_card("Dazed", uuid="draw-dazed")]
        combat.discard_pile = [make_card("Strike_R", uuid="discard-strike")]
        combat.exhaust_pile = []
        combat.player.energy = 0

        combat.play_card(0, 0)

        all_locations = (
            [card.uuid for card in combat.hand]
            + [card.uuid for card in combat.draw_pile]
            + [card.uuid for card in combat.discard_pile]
        )
        self.assertIn("flashofsteel", all_locations)

    def test_guardian_mode_shift_decrements_from_flame_barrier_in_v2(self):
        combat = NativeCombatEnv(seed=67, ascension_level=0, scheduled_encounter=["TheGuardian"], player=PlayerState())
        guardian = combat.monsters[0]
        guardian.move = "THE_GUARDIAN_WHIRLWIND"
        guardian.intent = "ATTACK"
        guardian.move_base_damage = 5
        guardian.move_hits = 4
        guardian.powers["Mode Shift"] = 50
        combat.player.current_hp = 80
        combat.player.block = 0
        combat.player.powers["Flame Barrier"] = 4

        combat._monster_take_turn(guardian)

        self.assertEqual(guardian.current_hp, guardian.max_hp - 16)
        self.assertEqual(guardian.power("Mode Shift"), 34)

    def test_flame_barrier_wakes_lagavulin_and_clears_metallicize_in_v2(self):
        combat = NativeCombatEnv(seed=671, ascension_level=0, scheduled_encounter=["Lagavulin"], player=PlayerState(current_hp=80, max_hp=80))
        lagavulin = combat.monsters[0]
        lagavulin.move = "LAGAVULIN_ATTACK"
        lagavulin.intent = "ATTACK"
        lagavulin.move_base_damage = 18
        lagavulin.move_hits = 1
        lagavulin.ai_state["asleep"] = 1
        lagavulin.ai_state["latent_awake"] = 1
        lagavulin.block = 8
        combat.player.powers["Flame Barrier"] = 4
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.end_turn()

        self.assertEqual(lagavulin.current_hp, lagavulin.max_hp - 4)
        self.assertEqual(lagavulin.block, 0)
        self.assertEqual(lagavulin.power("Metallicize"), 0)
        self.assertEqual(lagavulin.ai_state.get("asleep", 0), 0)

    def test_direct_attack_wakes_lagavulin_into_stun_intent_in_v2(self):
        combat = NativeCombatEnv(seed=672, ascension_level=0, scheduled_encounter=["Lagavulin"], player=PlayerState(current_hp=80, max_hp=80))
        lagavulin = combat.monsters[0]
        lagavulin.block = 0
        lagavulin.powers["Metallicize"] = 8
        lagavulin.ai_state["asleep"] = 1
        lagavulin.ai_state["latent_awake"] = 0

        dealt = combat._deal_direct_damage_to_monster(8, lagavulin)

        self.assertEqual(dealt, 8)
        self.assertEqual(lagavulin.current_hp, lagavulin.max_hp - 8)
        self.assertEqual(lagavulin.block, 0)
        self.assertEqual(lagavulin.power("Metallicize"), 0)
        self.assertEqual(lagavulin.ai_state.get("asleep", 0), 0)
        self.assertEqual(lagavulin.move, "LAGAVULIN_STUN")
        self.assertEqual(lagavulin.intent, "STUN")

    def test_guardian_mode_shift_decrements_from_juggernaut_block_damage_in_v2(self):
        combat = NativeCombatEnv(seed=6701, ascension_level=0, player=PlayerState(current_hp=60, max_hp=80, energy=1))
        guardian = make_monster("TheGuardian", StsRandom(6702), ascension=0)
        guardian.current_hp = 240
        guardian.max_hp = 250
        guardian.powers["Mode Shift"] = 50
        guardian.move = "THE_GUARDIAN_WHIRLWIND"
        guardian.intent = "ATTACK"
        guardian.move_base_damage = 5
        guardian.move_hits = 4
        combat.monsters = [guardian]
        combat.player.powers["Juggernaut"] = 5
        combat.hand = [make_card("Defend_R", uuid="guardian-defend")]

        combat.play_card(0, 0)

        self.assertEqual(guardian.current_hp, 235)
        self.assertEqual(guardian.power("Mode Shift"), 45)

    def test_combust_killing_fungi_applies_spore_cloud_on_player_turn_timing_in_v2(self):
        combat = NativeCombatEnv(seed=69, ascension_level=0, player=PlayerState())
        left = make_monster("FungiBeast", StsRandom(230), ascension=0)
        right = make_monster("FungiBeast", StsRandom(231), ascension=0)
        left.current_hp = 2
        left.max_hp = max(left.max_hp, 2)
        right.current_hp = 23
        right.max_hp = max(right.max_hp, 23)
        for monster in (left, right):
            monster.move = "FUNGI_BEAST_BITE"
            monster.intent = "ATTACK"
            monster.move_base_damage = 6
            monster.move_hits = 1
            monster.powers["Spore Cloud"] = 2
            monster.powers["Strength"] = 3
        combat.monsters = [left, right]
        combat.player.current_hp = 40
        combat.player.max_hp = 40
        combat.player.energy = 0
        combat.player.powers["Combust"] = 5
        combat.combust_hp_loss = 1
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.end_turn()

        self.assertEqual(combat.monsters[0].current_hp, 0)
        self.assertEqual(combat.monsters[1].current_hp, 18)
        self.assertEqual(combat.player.power("Vulnerable"), 1)

    def test_lagavulin_siphon_soul_consumes_artifact_on_dexterity_before_strength_in_v2(self):
        combat = NativeCombatEnv(seed=70, ascension_level=0, player=PlayerState())
        lagavulin = make_monster("Lagavulin", StsRandom(235), ascension=0)
        lagavulin.move = "LAGAVULIN_SIPHON_SOUL"
        lagavulin.intent = "STRONG_DEBUFF"
        lagavulin.move_base_damage = 0
        lagavulin.move_hits = 0
        combat.monsters = [lagavulin]
        combat.player.powers["Artifact"] = 1

        combat._monster_take_turn(lagavulin, 0)

        self.assertEqual(combat.player.power("Artifact"), 0)
        self.assertEqual(combat.player.power("Dexterity"), 0)
        self.assertEqual(combat.player.power("Strength"), -1)

    def test_whirlwind_at_zero_energy_does_not_wake_lagavulin_in_v2(self):
        combat = NativeCombatEnv(seed=68, ascension_level=0, scheduled_encounter=["Lagavulin"], player=PlayerState())
        lagavulin = combat.monsters[0]
        combat.hand = [make_card("Whirlwind", uuid="whirlwind")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 0
        before_hp = lagavulin.current_hp
        before_block = lagavulin.block

        combat.play_card(0, 0)

        self.assertEqual(lagavulin.current_hp, before_hp)
        self.assertEqual(lagavulin.block, before_block)
        self.assertEqual(lagavulin.power("Metallicize"), 8)

    def test_juggernaut_block_damage_wakes_latent_lagavulin_and_clears_metallicize_in_v2(self):
        combat = NativeCombatEnv(seed=681, ascension_level=0, scheduled_encounter=["Lagavulin"], player=PlayerState(current_hp=40, max_hp=80, energy=1))
        lagavulin = combat.monsters[0]
        lagavulin.move = "LAGAVULIN_ATTACK"
        lagavulin.intent = "ATTACK"
        lagavulin.move_base_damage = 18
        lagavulin.move_hits = 1
        lagavulin.block = 0
        lagavulin.ai_state["asleep"] = 1
        lagavulin.ai_state["latent_awake"] = 1
        combat.player.powers["Juggernaut"] = 5
        combat.hand = [make_card("Defend_R", uuid="defend")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertEqual(lagavulin.current_hp, lagavulin.max_hp - 5)
        self.assertEqual(lagavulin.power("Metallicize"), 0)
        self.assertEqual(lagavulin.ai_state.get("asleep", 0), 0)

    def test_corruption_defend_dead_branch_happens_before_juggernaut_rng_in_v2(self):
        combat = NativeCombatEnv(seed=2022962662194087066, ascension_level=0, player=PlayerState(current_hp=21, max_hp=80, energy=0))
        combat.relics.append({"relic_id": "Dead Branch", "name": "Dead Branch", "tier": "RARE", "counter": -1})
        combat.player.powers["Corruption"] = 1
        combat.player.powers["Juggernaut"] = 5
        combat.hand = [make_card("Defend_R", uuid="defend")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        left = make_monster("GreenLouse", StsRandom(1), ascension=0)
        middle = make_monster("GreenLouse", StsRandom(2), ascension=0)
        right = make_monster("GreenLouse", StsRandom(3), ascension=0)
        left.current_hp = 13
        middle.current_hp = 8
        right.current_hp = 14
        left.block = middle.block = right.block = 0
        combat.monsters = [left, middle, right]
        combat._apply_corruption_to_existing_cards()
        expected_rng = combat.card_random_rng.copy()
        expected_dead_branch = COMBAT_CARD_POOL_IRONCLAD[int(expected_rng.random(len(COMBAT_CARD_POOL_IRONCLAD) - 1))]
        expected_target_index = int(expected_rng.random(len(combat.monsters) - 1))
        expected_hp = [13, 8, 14]
        expected_hp[expected_target_index] -= 5

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Defend_R"])
        self.assertEqual([card.card_id for card in combat.hand], [expected_dead_branch])
        self.assertEqual([monster.current_hp for monster in combat.monsters], expected_hp)

    def test_whirlwind_at_zero_energy_triggers_sharp_hide_in_v2(self):
        combat = NativeCombatEnv(seed=68, ascension_level=0, scheduled_encounter=["TheGuardian"], player=PlayerState(block=5))
        guardian = combat.monsters[0]
        guardian.powers["Sharp Hide"] = 3
        combat.hand = [make_card("Whirlwind", uuid="whirlwind")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 0

        combat.play_card(0, 0)

        self.assertEqual(combat.player.block, 2)
        self.assertEqual(guardian.current_hp, guardian.max_hp)

    def test_self_forming_clay_block_triggers_juggernaut_at_start_of_turn_in_v2(self):
        combat = NativeCombatEnv(seed=69, ascension_level=0, player=PlayerState())
        target = make_monster("Cultist", StsRandom(69), ascension=0)
        target.current_hp = 12
        combat.monsters = [target]
        combat.player.powers["Juggernaut"] = 5
        combat.player.powers["Self-Forming Clay Block"] = 3

        combat.start_player_turn()

        self.assertEqual(combat.player.block, 3)
        self.assertEqual(target.current_hp, 7)

    def test_havoc_uses_current_energy_for_free_x_cost_card_in_v2(self):
        combat = NativeCombatEnv(seed=70, ascension_level=0, scheduled_encounter=["FungiBeast"], player=PlayerState())
        fungi = combat.monsters[0]
        before_hp = fungi.current_hp
        havoc = make_card("Havoc", uuid="havoc")
        havoc.cost_for_turn = 0
        combat.hand = [havoc]
        combat.draw_pile = [make_card("Whirlwind", uuid="whirlwind")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(fungi.current_hp, before_hp - 5)
        self.assertEqual(combat.player.energy, 1)
        self.assertIn("Havoc", [card.card_id for card in combat.discard_pile])
        self.assertIn("Whirlwind", [card.card_id for card in combat.exhaust_pile])

    def test_havoc_autoplay_untargeted_power_consumes_target_rng_in_v2(self):
        combat = NativeCombatEnv(seed=701, ascension_level=0, scheduled_encounter=["FungiBeast"], player=PlayerState())
        combat.relics.append({"relic_id": "Mummified Hand", "name": "Mummified Hand", "tier": "RARE", "counter": -1})
        havoc = make_card("Havoc", uuid="havoc")
        strike = make_card("Strike_R", uuid="strike")
        combat.hand = [havoc, strike]
        combat.draw_pile = [make_card("Metallicize", uuid="metallicize")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        class _SequenceRng:
            def __init__(self):
                self.calls = []

            def random(self, upper):
                self.calls.append(upper)
                return 0

        combat.card_random_rng = _SequenceRng()

        combat.play_card(0, 0)

        self.assertEqual(strike.cost_for_turn, 0)
        self.assertEqual(combat.card_random_rng.calls, [0, 0])

    def test_havoc_autoplay_untargeted_skill_consumes_target_rng_in_v2(self):
        combat = NativeCombatEnv(seed=702, ascension_level=0, scheduled_encounter=["FungiBeast", "JawWorm"], player=PlayerState())
        havoc = make_card("Havoc", uuid="havoc")
        combat.hand = [havoc]
        combat.draw_pile = [make_card("Defend_R", uuid="top-defend")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        class _SequenceRng:
            def __init__(self):
                self.calls = []

            def random(self, upper):
                self.calls.append(upper)
                return 1 if upper >= 1 else 0

        combat.card_random_rng = _SequenceRng()

        combat.play_card(0, 0)

        self.assertEqual(combat.card_random_rng.calls, [1])
        self.assertIn("Havoc", [card.card_id for card in combat.discard_pile])
        self.assertIn("Defend_R", [card.card_id for card in combat.exhaust_pile])

    def test_havoc_targeted_autoplay_chooses_target_before_letter_opener_damage_in_v2(self):
        combat = NativeCombatEnv(seed=703, ascension_level=0, scheduled_encounter=["Cultist", "Cultist"], player=PlayerState())
        havoc = make_card("Havoc", uuid="havoc")
        combat.hand = [havoc]
        combat.draw_pile = [make_card("Strike_R", uuid="top-strike")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.skills_played_this_turn = 2
        combat.relics.append({"relic_id": "Letter Opener", "name": "Letter Opener", "tier": "UNCOMMON", "counter": -1})
        combat.monsters[0].current_hp = 5
        combat.monsters[0].max_hp = 20
        combat.monsters[1].current_hp = 14
        combat.monsters[1].max_hp = 20

        class _SequenceRng:
            def __init__(self):
                self.calls = []

            def random(self, upper):
                self.calls.append(upper)
                return 0

        combat.card_random_rng = _SequenceRng()

        combat.play_card(0, 0)

        self.assertEqual(combat.card_random_rng.calls, [1])
        self.assertEqual([monster.current_hp for monster in combat.monsters], [0, 9])
        self.assertNotIn("Strike_R", [card.card_id for card in combat.exhaust_pile])
        self.assertNotIn("Strike_R", [card.card_id for card in combat.discard_pile])
        self.assertNotIn("Strike_R", [card.card_id for card in combat.draw_pile])

    def test_letter_opener_resolves_before_juggernaut_on_third_skill_block_card_in_v2(self):
        combat = NativeCombatEnv(
            seed=7031,
            ascension_level=0,
            scheduled_encounter=["GremlinWizard", "GremlinThief", "GremlinThief", "GremlinWarrior"],
            player=PlayerState(),
        )
        combat.relics.append({"relic_id": "Letter Opener", "name": "Letter Opener", "tier": "UNCOMMON", "counter": -1})
        combat.player.add_power("Juggernaut", 5)
        combat.skills_played_this_turn = 2
        combat.cards_played_this_turn = 2
        combat.hand = [make_card("Defend_R", upgrades=1, uuid="defend+")]
        combat.player.energy = 1
        combat.monsters[0].current_hp = 11
        combat.monsters[1].current_hp = 7
        combat.monsters[2].current_hp = 6
        combat.monsters[3].current_hp = 5

        class _SequenceRng:
            def random(self, upper):
                if upper == 3:
                    return 0
                if upper == 2:
                    return 2
                return 0

        combat.card_random_rng = _SequenceRng()

        combat.play_card(0, 0)

        self.assertEqual([monster.current_hp for monster in combat.monsters], [6, 2, 0, 0])

    def test_sword_boomerang_consumes_target_rng_for_each_hit_against_single_target_in_v2(self):
        combat = NativeCombatEnv(seed=7021, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        combat.hand = [make_card("Sword Boomerang", uuid="boomerang")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        class _SequenceRng:
            def __init__(self):
                self.calls = []

            def random(self, upper):
                self.calls.append(upper)
                return 0

        cultist = combat.monsters[0]
        cultist.current_hp = 50
        cultist.max_hp = 50
        combat.card_random_rng = _SequenceRng()

        combat.play_card(0, 0)

        self.assertEqual(combat.card_random_rng.calls, [0, 0, 0])
        self.assertEqual(cultist.current_hp, 41)

    def test_centurion_defend_targets_mystic_in_v2(self):
        combat = NativeCombatEnv(seed=69, ascension_level=0, scheduled_encounter=["Centurion", "Mystic"], player=PlayerState())
        centurion, mystic = combat.monsters
        centurion.move = "CENTURION_DEFEND"
        centurion.intent = "DEFEND"
        centurion.move_base_damage = 0
        centurion.move_hits = 0
        centurion.block = 0
        mystic.block = 0

        combat._monster_take_turn(centurion, 0)

        self.assertEqual(centurion.block, 0)
        self.assertEqual(mystic.block, 15)

    def test_mystic_buff_targets_centurion_and_self_in_v2(self):
        combat = NativeCombatEnv(seed=71, ascension_level=0, scheduled_encounter=["Centurion", "Mystic"], player=PlayerState())
        centurion, mystic = combat.monsters
        mystic.move = "MYSTIC_BUFF"
        mystic.intent = "BUFF"
        mystic.move_base_damage = 0
        mystic.move_hits = 0

        combat._monster_take_turn(mystic, 1)

        self.assertEqual(centurion.power("Strength"), 2)
        self.assertEqual(mystic.power("Strength"), 2)

    def test_mystic_heal_uses_magic_intent_in_v2(self):
        mystic = make_monster("Mystic", StsRandom(73), ascension=0)

        self.assertEqual(mystic.move, "MYSTIC_HEAL")
        self.assertEqual(mystic.intent, "MAGIC")

    def test_snecko_bite_uses_lightspeed_damage_table_in_v2(self):
        snecko = make_monster("Snecko", StsRandom(78), ascension=0)
        snecko.move_history = ["SNECKO_PERPLEXING_GLARE"]
        snecko.move = "SNECKO_PERPLEXING_GLARE"
        choose_next_move(snecko, StsRandom(80))
        self.assertEqual(snecko.move, "SNECKO_BITE")
        self.assertEqual(snecko.move_base_damage, 15)

    def test_snecko_never_rolls_back_into_glare_after_first_turn_in_v2(self):
        snecko = make_monster("Snecko", StsRandom(81), ascension=0)
        snecko.move_history = ["SNECKO_PERPLEXING_GLARE"]
        snecko.move = "SNECKO_PERPLEXING_GLARE"

        for seed in range(82, 92):
            probe = make_monster("Snecko", StsRandom(81), ascension=0)
            probe.move_history = ["SNECKO_PERPLEXING_GLARE"]
            probe.move = "SNECKO_PERPLEXING_GLARE"
            choose_next_move(probe, StsRandom(seed))
            self.assertIn(probe.move, {"SNECKO_BITE", "SNECKO_TAIL_WHIP"})

    def test_snecko_tail_whip_uses_attack_debuff_intent_in_v2(self):
        snecko = make_monster("Snecko", StsRandom(92), ascension=0)
        snecko.move = "SNECKO_TAIL_WHIP"
        snecko.move_history = ["SNECKO_PERPLEXING_GLARE"]

        from spirecomm.native_sim.monsters import _set_move
        _set_move(snecko, "SNECKO_TAIL_WHIP")

        self.assertEqual(snecko.intent, "ATTACK_DEBUFF")
        self.assertEqual(snecko.move_base_damage, 8)

    def test_snecko_tail_whip_applies_vulnerable_after_damage_in_v2(self):
        combat = NativeCombatEnv(seed=93, ascension_level=0, scheduled_encounter=["Snecko"], player=PlayerState())
        snecko = combat.monsters[0]
        snecko.move = "SNECKO_TAIL_WHIP"
        snecko.intent = "ATTACK_DEBUFF"
        snecko.move_base_damage = 8
        snecko.move_hits = 1
        snecko.powers["Strength"] = 1
        combat.player.current_hp = 30
        combat.player.block = 10
        combat.relics = [
            {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
            {"relic_id": "Odd Mushroom", "id": "Odd Mushroom", "name": "Odd Mushroom", "counter": -1, "price": 0, "tier": "EVENT"},
        ]

        combat._monster_take_turn(snecko, 0)

        self.assertEqual(combat.player.current_hp, 30)
        self.assertEqual(combat.player.power("Vulnerable"), 2)

    def test_snecko_glare_does_not_reroll_retained_hand_in_v2(self):
        combat = NativeCombatEnv(seed=94, ascension_level=0, scheduled_encounter=["Snecko"], player=PlayerState())
        combat.relics = [
            {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
            {"relic_id": "Runic Pyramid", "id": "Runic Pyramid", "name": "Runic Pyramid", "counter": -1, "price": 0, "tier": "BOSS"},
        ]
        bash = make_card("Bash", uuid="bash")
        strike = make_card("Strike_R", uuid="strike")
        injury = make_card("Injury", uuid="injury")
        blind = make_card("Blind", uuid="blind")
        flex = make_card("Flex", uuid="flex")
        combat.hand = [bash, strike, injury]
        combat.draw_pile = [flex, blind]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3

        class _FixedConfusionRng:
            def __init__(self):
                self.values = [1, 3]

            def randint(self, start, end):
                return self.values.pop(0)

        combat.card_random_rng = _FixedConfusionRng()
        combat.end_turn()

        self.assertEqual(combat.player.power("Confusion"), 1)
        self.assertEqual(
            [(card.card_id, card.cost_for_turn, card.cost_for_combat) for card in combat.hand],
            [
                ("Bash", None, None),
                ("Strike_R", None, None),
                ("Injury", None, None),
                ("Blind", 1, 1),
                ("Flex", 3, 3),
            ],
        )

    def test_chosen_debilitate_attacks_and_applies_vulnerable_in_v2(self):
        combat = NativeCombatEnv(seed=9302, ascension_level=0, scheduled_encounter=["Chosen"], player=PlayerState(current_hp=40, max_hp=40))
        chosen = combat.monsters[0]
        chosen.move = "CHOSEN_DEBILITATE"
        chosen.intent = "ATTACK_DEBUFF"
        chosen.move_base_damage = 6
        chosen.move_hits = 1

        combat._monster_take_turn(chosen, 0)

        self.assertEqual(combat.player.current_hp, 34)
        self.assertEqual(combat.player.power("Vulnerable"), 2)

    def test_hypnotizing_colored_mushrooms_ignore_gains_gold_not_parasite_in_v2(self):
        env = NativeRunEnv(seed=9301, ascension_level=0, start_on_map=True)
        env.phase = "EVENT"
        env.event_id = "Hypnotizing Colored Mushrooms"
        env.gold = 12
        env.player.current_hp = 30
        env.player.max_hp = 80

        env.step({"kind": "event", "event_id": "Hypnotizing Colored Mushrooms", "name": "Ignored", "choice_index": 1})

        self.assertEqual(env.gold, 111)
        self.assertEqual(env.player.current_hp, 30)
        self.assertNotIn("Parasite", [card.card_id for card in env.deck])

    def test_masked_bandits_fight_costs_5_hp_before_event_combat_in_v2(self):
        env = NativeRunEnv(seed=9303, ascension_level=0, start_on_map=True)
        env.phase = "EVENT"
        env.event_id = "Masked Bandits"
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.event_options = [
            {"kind": "event", "event_id": "Masked Bandits", "name": "Paid Fearfully", "choice_index": 0},
            {"kind": "event", "event_id": "Masked Bandits", "name": "Fought Bandits", "choice_index": 1},
        ]

        env.step({"kind": "event", "event_id": "Masked Bandits", "name": "Fought Bandits", "choice_index": 1})

        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual(env.player.current_hp, 35)

    def test_masked_bandits_pay_fearfully_zeroes_gold_without_hp_loss_in_v2(self):
        env = NativeRunEnv(seed=9304, ascension_level=0, start_on_map=True)
        env.phase = "EVENT"
        env.event_id = "Masked Bandits"
        env.player.current_hp = 40
        env.player.max_hp = 80
        env.gold = 99
        env.event_options = [
            {"kind": "event", "event_id": "Masked Bandits", "name": "Paid Fearfully", "choice_index": 0},
            {"kind": "event", "event_id": "Masked Bandits", "name": "Fought Bandits", "choice_index": 1},
        ]

        env.step({"kind": "event", "event_id": "Masked Bandits", "name": "Paid Fearfully", "choice_index": 0})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.player.current_hp, 40)
        self.assertEqual(env.gold, 0)


    def test_snake_plant_roll_logic_matches_lightspeed_in_v2(self):
        snake = make_monster("SnakePlant", StsRandom(94), ascension=0)
        snake.move_history = ["SNAKE_PLANT_CHOMP", "SNAKE_PLANT_CHOMP"]
        snake.move = "SNAKE_PLANT_CHOMP"

        choose_next_move(snake, StsRandom(95))

        self.assertEqual(snake.move, "SNAKE_PLANT_ENFEEBLING_SPORES")

    def test_headbutt_defers_malleable_block_until_card_select_resolves_in_v2(self):
        combat = NativeCombatEnv(seed=946, ascension_level=0, scheduled_encounter=["SnakePlant"], player=PlayerState())
        snake = combat.monsters[0]
        snake.current_hp = 64
        snake.block = 4
        snake.powers["Malleable"] = 5
        snake.powers["Vulnerable"] = 2
        snake.powers["Weakened"] = 1
        snake.powers["Strength"] = 1
        combat.hand = [make_card("Headbutt", uuid="hb")]
        combat.discard_pile = [make_card("Strike_R", uuid="d0"), make_card("Defend_R", uuid="d1")]
        combat.draw_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(snake.current_hp, 55)
        self.assertEqual(snake.block, 0)
        self.assertEqual(snake.power("Malleable"), 6)

        combat.step({"kind": "card_select", "select_type": "HEADBUTT", "choice_index": 0})

        self.assertEqual(combat.card_select_context, None)
        self.assertEqual(snake.block, 5)

    def test_sword_boomerang_defers_malleable_block_until_attack_finishes_in_v2(self):
        combat = NativeCombatEnv(seed=947, ascension_level=0, scheduled_encounter=["SnakePlant"], player=PlayerState())
        snake = combat.monsters[0]
        snake.current_hp = 50
        snake.max_hp = 50
        snake.block = 0
        snake.powers["Malleable"] = 4
        combat.hand = [make_card("Sword Boomerang", uuid="boomerang")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(snake.current_hp, 41)
        self.assertEqual(snake.block, 15)
        self.assertEqual(snake.power("Malleable"), 7)

    def test_dramatic_entrance_charons_ashes_resolves_before_malleable_block_in_v2(self):
        combat = NativeCombatEnv(seed=948, ascension_level=0, scheduled_encounter=["SnakePlant"], player=PlayerState())
        snake = combat.monsters[0]
        snake.current_hp = 60
        snake.max_hp = 78
        snake.block = 4
        snake.powers["Malleable"] = 5
        snake.powers["Vulnerable"] = 1
        combat.relics.append({"relic_id": "Charon's Ashes", "name": "Charon's Ashes", "tier": "RARE"})
        combat.hand = [make_card("Dramatic Entrance", uuid="dramatic")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(snake.current_hp, 49)
        self.assertEqual(snake.block, 5)
        self.assertEqual(snake.power("Malleable"), 6)

    def test_looter_escape_keeps_hp_but_marks_gone_in_v2(self):
        combat = NativeCombatEnv(seed=74, ascension_level=0, scheduled_encounter=["Looter"], player=PlayerState())
        looter = combat.monsters[0]
        looter.current_hp = 45
        looter.move = "LOOTER_ESCAPE"
        looter.intent = "ESCAPE"

        combat._monster_take_turn(looter, 0)

        self.assertEqual(looter.current_hp, 45)
        self.assertTrue(looter.is_gone)
        self.assertTrue(looter.ai_state.get("escaping"))

    def test_looter_escape_preserves_debuff_duration_in_v2(self):
        combat = NativeCombatEnv(seed=741, ascension_level=0, scheduled_encounter=["FungiBeast"], player=PlayerState())
        looter = make_monster("Looter", StsRandom(741), ascension=0)
        fungi = combat.monsters[0]
        combat.monsters = [looter, fungi]
        looter.current_hp = 45
        looter.powers["Vulnerable"] = 2
        looter.move = "LOOTER_ESCAPE"
        looter.intent = "ESCAPE"
        fungi.move = "FUNGI_BEAST_GROW"
        fungi.intent = "BUFF"
        fungi.move_base_damage = 0
        fungi.move_hits = 0

        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.end_turn()

        self.assertEqual(looter.power("Vulnerable"), 2)

    def test_final_looter_kill_still_returns_stolen_gold_in_v2(self):
        combat = NativeCombatEnv(seed=740, ascension_level=0, scheduled_encounter=["Looter"], player=PlayerState())
        looter = combat.monsters[0]
        looter.current_hp = 6
        looter.max_hp = 48
        looter.ai_state["stolen_gold"] = 45
        combat.hand = [make_card("Strike_R", uuid="strike")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.outcome, "PLAYER_VICTORY")
        self.assertEqual(combat.reward_gold_bonus, 45)

    def test_buffer_does_not_prevent_bloodletting_hp_loss_in_v2(self):
        combat = NativeCombatEnv(seed=75, ascension_level=0, player=PlayerState())
        combat.player.current_hp = 20
        combat.player.max_hp = 80
        combat.player.energy = 2
        combat.player.powers["Buffer"] = 1
        combat.hand = [make_card("Bloodletting", uuid="bloodletting")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 17)
        self.assertEqual(combat.player.energy, 4)
        self.assertEqual(combat.player.power("Buffer"), 1)

    def test_buffer_still_prevents_attack_damage_in_v2(self):
        combat = NativeCombatEnv(seed=76, ascension_level=0, scheduled_encounter=["FungiBeast"], player=PlayerState())
        fungi = combat.monsters[0]
        fungi.move = "FUNGI_BEAST_BITE"
        fungi.intent = "ATTACK"
        fungi.move_base_damage = 6
        fungi.move_hits = 1
        combat.player.current_hp = 20
        combat.player.powers["Buffer"] = 1

        combat._monster_take_turn(fungi, 0)

        self.assertEqual(combat.player.current_hp, 20)
        self.assertEqual(combat.player.power("Buffer"), 0)

    def test_buffer_prevents_sharp_hide_counter_damage_in_v2(self):
        combat = NativeCombatEnv(seed=77, ascension_level=0, scheduled_encounter=["TheGuardian"], player=PlayerState())
        guardian = combat.monsters[0]
        guardian.powers["Sharp Hide"] = 3
        combat.player.current_hp = 20
        combat.player.energy = 2
        combat.player.powers["Buffer"] = 1
        combat.hand = [make_card("Clothesline", uuid="clothesline")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 20)
        self.assertEqual(combat.player.power("Buffer"), 0)

    def test_ornamental_fan_block_applies_before_sharp_hide_counter_damage_in_v2(self):
        combat = NativeCombatEnv(seed=77005, ascension_level=0, scheduled_encounter=["JawWorm"], player=PlayerState())
        combat.relics.append({"relic_id": "Ornamental Fan", "name": "Ornamental Fan", "tier": "COMMON", "counter": -1})
        jaw_worm = combat.monsters[0]
        jaw_worm.powers["Sharp Hide"] = 3
        combat.player.current_hp = 20
        combat.player.energy = 3
        combat.hand = [
            make_card("Anger", uuid="anger"),
            make_card("Bash", uuid="bash"),
            make_card("Strike_R", uuid="strike"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)
        combat.play_card(0, 0)
        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 14)
        self.assertEqual(combat.player.block, 1)

    def test_ornamental_fan_block_triggers_juggernaut_in_v2(self):
        combat = NativeCombatEnv(seed=77006, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        combat.relics.append({"relic_id": "Ornamental Fan", "name": "Ornamental Fan", "tier": "COMMON", "counter": -1})
        cultist = combat.monsters[0]
        cultist.current_hp = 30
        cultist.max_hp = 30
        combat.player.add_power("Juggernaut", 5)
        combat.player.energy = 3
        combat.hand = [
            make_card("Anger", uuid="anger"),
            make_card("Bash", uuid="bash"),
            make_card("Strike_R", uuid="strike"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)
        combat.play_card(0, 0)
        combat.play_card(0, 0)

        self.assertEqual(combat.player.block, 4)
        self.assertEqual(cultist.current_hp, 2)

    def test_ornamental_fan_juggernaut_waits_until_malleable_block_resolves_in_v2(self):
        combat = NativeCombatEnv(seed=770061, ascension_level=0, scheduled_encounter=["SnakePlant"], player=PlayerState())
        snake = combat.monsters[0]
        snake.current_hp = 18
        snake.max_hp = 78
        snake.block = 4
        snake.powers["Malleable"] = 5
        snake.powers["Vulnerable"] = 1
        combat.relics.append({"relic_id": "Ornamental Fan", "name": "Ornamental Fan", "tier": "COMMON", "counter": -1})
        combat.player.powers["Juggernaut"] = 5
        combat.player.powers["Weakened"] = 1
        combat.player.energy = 1
        combat.attack_played_this_turn = 2
        combat.hand = [make_card("Anger", uuid="anger")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertEqual(combat.player.block, 4)
        self.assertEqual((snake.current_hp, snake.block), (16, 0))
        self.assertEqual(snake.power("Malleable"), 6)

    def test_dramatic_entrance_delays_attack_relic_proc_until_all_targets_in_v2(self):
        combat = NativeCombatEnv(seed=77007, ascension_level=0, scheduled_encounter=["Cultist", "Cultist"], player=PlayerState())
        combat.relics.append({"relic_id": "Shuriken", "name": "Shuriken", "tier": "COMMON", "counter": -1})
        for monster in combat.monsters:
            monster.current_hp = 20
            monster.max_hp = 20
        combat.player.energy = 1
        combat.player.powers["Strength"] = 1
        combat.attack_played_this_turn = 2
        combat.hand = [make_card("Dramatic Entrance", uuid="dramatic")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertEqual([monster.current_hp for monster in combat.monsters], [11, 11])
        self.assertEqual(combat.player.power("Strength"), 2)

    def test_pommel_strike_fire_breathing_hits_guardian_after_mode_shift_block_in_v2(self):
        combat = NativeCombatEnv(seed=770071, ascension_level=0, scheduled_encounter=["TheGuardian"], player=PlayerState())
        guardian = combat.monsters[0]
        guardian.current_hp = 213
        guardian.max_hp = 250
        guardian.block = 0
        guardian.move = "THE_GUARDIAN_VENT_STEAM"
        guardian.intent = "DEBUFF"
        guardian.move_base_damage = 0
        guardian.move_hits = 0
        guardian.powers["Mode Shift"] = 3
        combat.player.energy = 1
        combat.player.powers["Fire Breathing"] = 6
        combat.hand = [make_card("Pommel Strike", uuid="pommel")]
        combat.draw_pile = [make_card("Doubt", uuid="doubt")]
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertEqual((guardian.current_hp, guardian.block), (204, 14))
        self.assertEqual(guardian.move, "THE_GUARDIAN_DEFENSIVE_MODE")

    def test_torii_does_not_reduce_bloodletting_hp_loss_in_v2(self):
        combat = NativeCombatEnv(seed=7701, ascension_level=0, player=PlayerState())
        combat.player.current_hp = 20
        combat.player.max_hp = 80
        combat.player.energy = 2
        combat.relics.append({"relic_id": "Torii", "name": "Torii", "tier": "RARE"})
        combat.hand = [make_card("Bloodletting", uuid="bloodletting")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 17)
        self.assertEqual(combat.player.energy, 4)

    def test_upgrade_combat_card_updates_upgraded_cost_when_unmodified_in_v2(self):
        combat = NativeCombatEnv(seed=7702, ascension_level=0, player=PlayerState())
        card = make_card("Corruption", uuid="corruption")
        self.assertEqual(card.cost, 3)

        combat._upgrade_combat_card(card)

        self.assertEqual(card.upgrades, 1)
        self.assertEqual(card.cost, 2)

    def test_upgrade_combat_card_replaces_stale_explicit_cost_in_v2(self):
        combat = NativeCombatEnv(seed=7703, ascension_level=0, player=PlayerState())
        card = make_card("Corruption", uuid="corruption")
        card.cost_for_turn = 3

        combat._upgrade_combat_card(card)

        self.assertEqual(card.upgrades, 1)
        self.assertEqual(card.cost, 2)

    def test_upgrade_combat_card_restores_base_upgraded_cost_after_madness_zeroes_card_def_in_v2(self):
        combat = NativeCombatEnv(seed=77031, ascension_level=0, player=PlayerState())
        card = make_card("Entrench", uuid="entrench")
        card.card_def = replace(card.card_def, cost=0, upgraded_cost=0)

        combat._upgrade_combat_card(card)

        self.assertEqual(card.upgrades, 1)
        self.assertEqual(card.cost_for_turn, 1)
        self.assertEqual(card.cost, 1)

    def test_chrysalis_cost_reduction_persists_across_discard_in_v2(self):
        combat = NativeCombatEnv(seed=78, ascension_level=0, player=PlayerState())
        combat.draw_pile = []
        combat.discard_pile = []
        combat.hand = []

        combat._put_random_cards_in_draw_pile(card_type="SKILL", count=1, cost_for_turn=0)
        card = combat.draw_pile.pop()

        self.assertEqual(card.cost_for_combat, 0)
        self.assertEqual(card.cost, 0)

        combat._move_card_to_discard(card)

        self.assertEqual(combat.discard_pile[0].cost_for_turn, 0)
        self.assertEqual(combat.discard_pile[0].cost_for_combat, 0)
        self.assertEqual(combat.discard_pile[0].cost, 0)

    def test_fiend_fire_uses_lightspeed_random_exhaust_order_in_v2(self):
        combat = NativeCombatEnv(seed=97, ascension_level=0, scheduled_encounter=["FungiBeast"], player=PlayerState())
        combat.hand = [
            make_card("Strike_R", uuid="strike"),
            make_card("Sword Boomerang", uuid="boomerang"),
            make_card("Fiend Fire", uuid="fiend"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(2, 0)

        self.assertEqual(
            [card.card_id for card in combat.exhaust_pile],
            ["Sword Boomerang", "Strike_R", "Fiend Fire"],
        )

    def test_guardian_mode_shift_defers_block_until_after_multi_hit_attack_in_v2(self):
        combat = NativeCombatEnv(seed=98, ascension_level=0, player=PlayerState())
        guardian = make_monster("TheGuardian", StsRandom(9), ascension=0)
        guardian.current_hp = 200
        guardian.max_hp = 250
        guardian.powers["Mode Shift"] = 5
        guardian.move = "THE_GUARDIAN_CHARGING_UP"
        guardian.intent = "BUFF"
        guardian.move_base_damage = 0
        guardian.move_hits = 0
        combat.monsters = [guardian]
        combat.hand = [make_card("Twin Strike", uuid="twin-strike")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(guardian.current_hp, 190)
        self.assertEqual(guardian.block, 20)
        self.assertEqual(guardian.move, "THE_GUARDIAN_DEFENSIVE_MODE")
        self.assertEqual(guardian.power("Mode Shift"), 0)

    def test_guardian_mode_shift_from_combust_does_not_keep_block_into_next_player_turn_in_v2(self):
        combat = NativeCombatEnv(seed=99, ascension_level=0, player=PlayerState())
        guardian = make_monster("TheGuardian", StsRandom(10), ascension=0)
        guardian.current_hp = 200
        guardian.max_hp = 250
        guardian.powers["Mode Shift"] = 5
        guardian.move = "THE_GUARDIAN_DEFENSIVE_MODE"
        guardian.intent = "BUFF"
        guardian.move_base_damage = 0
        guardian.move_hits = 0
        combat.monsters = [guardian]
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.current_hp = 40
        combat.player.max_hp = 40
        combat.player.powers["Combust"] = 5
        combat.combust_hp_loss = 1

        combat.end_turn()

        self.assertEqual(guardian.move, "THE_GUARDIAN_ROLL_ATTACK")
        self.assertEqual(guardian.block, 0)
        self.assertEqual(guardian.power("Sharp Hide"), 3)

    def test_guardian_mode_shift_during_fierce_bash_stays_in_defensive_mode_in_v2(self):
        combat = NativeCombatEnv(seed=100, ascension_level=0, player=PlayerState())
        guardian = make_monster("TheGuardian", StsRandom(11), ascension=0)
        guardian.current_hp = 200
        guardian.max_hp = 250
        guardian.powers["Mode Shift"] = 5
        guardian.ai_state["mode_shift_amount"] = 40
        guardian.move = "THE_GUARDIAN_VENT_STEAM"
        guardian.intent = "DEBUFF"
        guardian.move_base_damage = 0
        guardian.move_hits = 0
        combat.monsters = [guardian]
        combat.hand = [make_card("Clash", uuid="clash")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 0

        combat.play_card(0, 0)

        self.assertEqual(guardian.move, "THE_GUARDIAN_DEFENSIVE_MODE")
        self.assertEqual(guardian.intent, "BUFF")
        self.assertEqual(guardian.block, 20)

    def test_guardian_fierce_bash_shifted_by_thorns_stays_defensive_in_v2(self):
        combat = NativeCombatEnv(seed=1001, ascension_level=0, player=PlayerState())
        guardian = make_monster("TheGuardian", StsRandom(111), ascension=0)
        guardian.current_hp = 211
        guardian.max_hp = 250
        guardian.powers["Mode Shift"] = 1
        guardian.ai_state["mode_shift_amount"] = 30
        guardian.move = "THE_GUARDIAN_FIERCE_BASH"
        guardian.move_history = ["THE_GUARDIAN_FIERCE_BASH", "THE_GUARDIAN_CHARGING_UP"]
        guardian.intent = "ATTACK"
        guardian.move_base_damage = 32
        guardian.move_hits = 1
        combat.monsters = [guardian]
        combat.player.current_hp = 24
        combat.player.max_hp = 80
        combat.player.block = 0
        combat.player.powers["Thorns"] = 3
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.end_turn()

        self.assertEqual(guardian.current_hp, 208)
        self.assertEqual(guardian.move, "THE_GUARDIAN_DEFENSIVE_MODE")
        self.assertEqual(guardian.intent, "BUFF")
        self.assertEqual(guardian.block, 20)
        self.assertEqual(guardian.power("Mode Shift"), 0)

    def test_guardian_mode_shift_overkill_does_not_reduce_next_cycle_amount_in_v2(self):
        combat = NativeCombatEnv(seed=101, ascension_level=0, player=PlayerState())
        guardian = make_monster("TheGuardian", StsRandom(12), ascension=0)
        guardian.current_hp = 200
        guardian.max_hp = 250
        guardian.powers["Mode Shift"] = 5
        guardian.ai_state["mode_shift_amount"] = 40
        guardian.move = "THE_GUARDIAN_CHARGING_UP"
        guardian.intent = "BUFF"
        guardian.move_base_damage = 0
        guardian.move_hits = 0
        combat.monsters = [guardian]
        combat.hand = [make_card("Strike_R", uuid="strike")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.current_hp = 50
        combat.player.max_hp = 50

        combat.play_card(0, 0)

        guardian.move = "THE_GUARDIAN_TWIN_SLAM"
        guardian.intent = "ATTACK"
        guardian.move_base_damage = 8
        guardian.move_hits = 2
        guardian.block = 20
        guardian.powers["Sharp Hide"] = 3
        combat.player.energy = 0

        combat.end_turn()

        self.assertEqual(guardian.move, "THE_GUARDIAN_WHIRLWIND")
        self.assertEqual(guardian.power("Mode Shift"), 50)

    def test_player_plated_armor_decrements_on_unblocked_attack_in_v2(self):
        combat = NativeCombatEnv(seed=98, ascension_level=0, scheduled_encounter=["RedLouse"], player=PlayerState())
        louse = combat.monsters[0]
        louse.move = "RED_LOUSE_BITE"
        louse.intent = "ATTACK"
        louse.move_base_damage = 6
        louse.move_hits = 1
        combat.player.current_hp = 30
        combat.player.block = 0
        combat.player.powers["Plated Armor"] = 4

        combat._monster_take_turn(louse, 0)

        self.assertEqual(combat.player.power("Plated Armor"), 3)

    def test_hex_dazed_is_inserted_after_offering_draws_in_v2(self):
        combat = NativeCombatEnv(seed=103, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Offering", uuid="offering")]
        combat.draw_pile = [
            make_card("Anger", uuid="anger"),
            make_card("Dazed", uuid="dazed-1"),
            make_card("Dazed", uuid="dazed-2"),
            make_card("Bash", uuid="bash"),
            make_card("Dazed", uuid="dazed-3"),
            make_card("Strike_R", uuid="strike"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.current_hp = 50
        combat.player.max_hp = 80
        combat.player.powers["Hex"] = 1
        combat.player.powers["Evolve"] = 1

        class InsertNearTopRandom:
            def random(self, upper):
                return upper

        combat.card_random_rng = InsertNearTopRandom()

        combat.play_card(0, 0)

        hand_ids = [card.card_id for card in combat.hand]
        self.assertEqual(combat.draw_pile, [])
        self.assertEqual(hand_ids, ["Strike_R", "Dazed", "Bash", "Dazed", "Dazed", "Dazed", "Anger"])

    def test_offering_dead_branch_waits_until_evolve_draws_finish_in_v2(self):
        combat = NativeCombatEnv(seed=1032, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Offering", uuid="offering"),
            make_card("Havoc", uuid="havoc"),
            make_card("Dazed", uuid="hand-dazed-a"),
            make_card("Dazed", uuid="hand-dazed-b"),
        ]
        combat.draw_pile = [
            make_card("Strike_R", uuid="draw-strike"),
            make_card("Dazed", uuid="draw-dazed-b"),
            make_card("Defend_R", uuid="draw-defend"),
            make_card("Dazed", uuid="draw-dazed-a"),
            make_card("Flex", uuid="draw-flex"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.current_hp = 40
        combat.player.max_hp = 80
        combat.player.powers["Evolve"] = 2
        combat.relics.append({"relic_id": "Dead Branch", "name": "Dead Branch", "tier": "RARE", "counter": -1})
        combat._random_combat_card_id = lambda: "Dropkick"

        combat.play_card(0, 0)

        self.assertEqual(
            [card.card_id for card in combat.hand],
            ["Havoc", "Dazed", "Dazed", "Flex", "Dazed", "Defend_R", "Dazed", "Strike_R", "Dropkick"],
        )

    def test_reaper_heals_from_actual_hp_lost_not_overkill_in_v2(self):
        combat = NativeCombatEnv(
            seed=104,
            ascension_level=0,
            scheduled_encounter=["RedLouse", "GreenLouse"],
            player=PlayerState(),
        )
        combat.hand = [make_card("Reaper", uuid="reaper")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.current_hp = 23
        combat.player.max_hp = 80
        combat.player.energy = 2

        red, green = combat.monsters
        red.current_hp = 5
        red.powers["Vulnerable"] = 1
        green.current_hp = 15

        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 32)

    def test_reaper_heal_uses_magic_flower_in_v2(self):
        combat = NativeCombatEnv(
            seed=104,
            ascension_level=0,
            scheduled_encounter=["JawWorm"],
            player=PlayerState(),
        )
        combat.relics.append({"relic_id": "Magic Flower", "name": "Magic Flower", "tier": "RARE", "counter": -1})
        combat.hand = [make_card("Reaper", uuid="reaper")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.current_hp = 32
        combat.player.max_hp = 80
        combat.player.energy = 2
        combat.monsters[0].current_hp = 42

        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 38)

    def test_whirlwind_triggers_the_guardian_sharp_hide_once_in_v2(self):
        combat = NativeCombatEnv(seed=1041, ascension_level=0, scheduled_encounter=["TheGuardian"], player=PlayerState())
        guardian = combat.monsters[0]
        guardian.move = "THE_GUARDIAN_TWIN_SLAM"
        guardian.intent = "ATTACK"
        guardian.move_base_damage = 8
        guardian.move_hits = 2
        guardian.powers["Sharp Hide"] = 3
        combat.hand = [make_card("Whirlwind", uuid="whirlwind")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.current_hp = 30
        combat.player.max_hp = 80
        combat.player.block = 5
        combat.player.energy = 3

        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 30)
        self.assertEqual(combat.player.block, 2)

    def test_thunderclap_third_attack_shuriken_applies_after_all_targets_in_v2(self):
        combat = NativeCombatEnv(
            seed=105,
            ascension_level=0,
            scheduled_encounter=["Cultist", "Cultist"],
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Shuriken", "id": "Shuriken", "name": "Shuriken", "counter": -1, "price": 0, "tier": "COMMON"},
            ],
        )
        combat.hand = [make_card("Thunderclap", uuid="thunderclap")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Strength"] = 1
        combat.attack_played_this_turn = 2
        for monster in combat.monsters:
            monster.current_hp = 20
            monster.max_hp = 20

        combat.play_card(0, 0)

        self.assertEqual([monster.current_hp for monster in combat.monsters], [15, 15])
        self.assertEqual(combat.player.power("Strength"), 2)

    def test_cleave_third_attack_shuriken_applies_after_all_targets_in_v2(self):
        combat = NativeCombatEnv(
            seed=106,
            ascension_level=0,
            scheduled_encounter=["Cultist", "Cultist"],
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Shuriken", "id": "Shuriken", "name": "Shuriken", "counter": -1, "price": 0, "tier": "COMMON"},
            ],
        )
        combat.hand = [make_card("Cleave", uuid="cleave")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Strength"] = 1
        combat.attack_played_this_turn = 2
        for monster in combat.monsters:
            monster.current_hp = 20
            monster.max_hp = 20

        combat.play_card(0, 0)

        self.assertEqual([monster.current_hp for monster in combat.monsters], [11, 11])
        self.assertEqual(combat.player.power("Strength"), 2)

    def test_dropkick_does_not_receive_akabeko_bonus_in_v2(self):
        combat = NativeCombatEnv(seed=105, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        combat.hand = [make_card("Dropkick", uuid="dropkick")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Akabeko"] = 8
        cultist = combat.monsters[0]
        cultist.current_hp = 50
        cultist.max_hp = 50
        cultist.powers["Vulnerable"] = 1

        combat.play_card(0, 0)

        self.assertEqual(cultist.current_hp, 43)
        self.assertEqual(combat.player.power("Akabeko"), 0)
        self.assertEqual(combat.player.energy, 1)

    def test_dropkick_evolve_shuffle_includes_current_card_in_v2(self):
        combat = NativeCombatEnv(seed=106, ascension_level=0, player=PlayerState(energy=1))
        cultist = make_monster("Cultist", StsRandom(106), ascension=0)
        cultist.current_hp = 30
        cultist.max_hp = 30
        cultist.powers["Vulnerable"] = 1
        combat.monsters = [cultist]
        combat.player.powers["Evolve"] = 1
        combat.hand = [make_card("Dropkick", uuid="dropkick")]
        combat.draw_pile = [make_card("Dazed", uuid="draw-dazed")]
        combat.discard_pile = [make_card("Defend_R", uuid="discard-defend")]
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertIn("Dazed", [card.card_id for card in combat.hand])
        self.assertIn("Dropkick", [card.card_id for card in combat.hand + combat.draw_pile])
        self.assertEqual([card.card_id for card in combat.discard_pile], [])

    def test_cleave_uses_shared_akabeko_snapshot_for_all_targets_in_v2(self):
        combat = NativeCombatEnv(
            seed=106,
            ascension_level=0,
            scheduled_encounter=["RedLouse", "GreenLouse"],
            player=PlayerState(),
        )
        combat.hand = [make_card("Cleave", uuid="cleave")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Akabeko"] = 8
        for monster in combat.monsters:
            monster.current_hp = 40
            monster.max_hp = 40

        combat.play_card(0, 0)

        self.assertEqual([monster.current_hp for monster in combat.monsters], [24, 24])
        self.assertEqual(combat.player.power("Akabeko"), 0)

    def test_sword_boomerang_uses_shared_akabeko_snapshot_for_each_hit_in_v2(self):
        combat = NativeCombatEnv(seed=107, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        combat.hand = [make_card("Sword Boomerang", uuid="boomerang")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Akabeko"] = 8
        cultist = combat.monsters[0]
        cultist.current_hp = 50
        cultist.max_hp = 50

        combat.play_card(0, 0)

        self.assertEqual(cultist.current_hp, 17)
        self.assertEqual(combat.player.power("Akabeko"), 0)

    def test_cleave_uses_pen_nib_for_all_targets_in_v2(self):
        combat = NativeCombatEnv(
            seed=1071,
            ascension_level=0,
            scheduled_encounter=["RedLouse", "GreenLouse"],
            player=PlayerState(),
        )
        combat.hand = [make_card("Cleave", uuid="cleave")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Pen Nib"] = 1
        for monster in combat.monsters:
            monster.current_hp = 20
            monster.max_hp = 20
            monster.block = 0
            monster.powers.clear()

        combat.play_card(0, 0)

        self.assertEqual([monster.current_hp for monster in combat.monsters], [4, 4])
        self.assertEqual(combat.player.power("Pen Nib"), 0)

    def test_sword_boomerang_uses_pen_nib_for_all_hits_in_v2(self):
        combat = NativeCombatEnv(seed=1072, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        combat.hand = [make_card("Sword Boomerang", uuid="boomerang")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Pen Nib"] = 1
        cultist = combat.monsters[0]
        cultist.current_hp = 50
        cultist.max_hp = 50

        combat.play_card(0, 0)

        self.assertEqual(cultist.current_hp, 32)
        self.assertEqual(combat.player.power("Pen Nib"), 0)

    def test_jax_upgrade_grants_three_strength_in_v2(self):
        combat = NativeCombatEnv(seed=108, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        combat.hand = [make_card("J.A.X.", upgrades=1, uuid="jax")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.current_hp = 40
        combat.player.max_hp = 40

        combat.play_card(0, 0)

        self.assertEqual(combat.player.power("Strength"), 3)
        self.assertEqual(combat.player.current_hp, 37)

    def test_headbutt_replay_preserves_original_sharp_hide_counter_damage_in_v2(self):
        combat = NativeCombatEnv(seed=109, ascension_level=0, scheduled_encounter=["TheGuardian"], player=PlayerState())
        guardian = combat.monsters[0]
        guardian.current_hp = 200
        guardian.max_hp = 250
        guardian.powers["Sharp Hide"] = 3
        combat.hand = [make_card("Double Tap", uuid="double-tap"), make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = []
        combat.discard_pile = [make_card("Bash", uuid="bash")]
        combat.exhaust_pile = []
        combat.player.energy = 2
        combat.player.block = 10

        combat.play_card(0, 0)
        combat.play_card(0, 0)
        self.assertEqual(combat.card_select_context, "HEADBUTT")

        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})
        self.assertEqual(combat.card_select_context, "HEADBUTT")
        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})

        self.assertEqual(combat.player.block, 4)
        self.assertEqual([card.card_id for card in combat.draw_pile], ["Bash", "Double Tap"])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Headbutt"])

    def test_headbutt_replay_stops_second_select_when_first_hit_kills_target_in_v2(self):
        combat = NativeCombatEnv(seed=1090, ascension_level=0, scheduled_encounter=["RedLouse", "RedLouse"], player=PlayerState())
        first, second = combat.monsters[:2]
        first.current_hp = 6
        first.max_hp = 14
        first.powers["Vulnerable"] = 1
        first.powers["Strength"] = 3
        second.current_hp = 14
        second.max_hp = 14
        second.powers["Curl Up"] = 7
        combat.hand = [make_card("Double Tap", uuid="double-tap"), make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = [make_card("Bash", uuid="bash")]
        combat.discard_pile = [make_card("Defend_R", uuid="defend-r")]
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)
        combat.play_card(0, 0)
        self.assertEqual(combat.card_select_context, "HEADBUTT")

        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})

        self.assertIsNone(combat.card_select_context)
        self.assertEqual([card.card_id for card in combat.draw_pile], ["Bash", "Defend_R"])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Double Tap", "Headbutt"])

    def test_headbutt_replay_keeps_second_select_when_replay_kills_target_in_v2(self):
        combat = NativeCombatEnv(seed=10901, ascension_level=0, scheduled_encounter=["GreenLouse", "RedLouse"], player=PlayerState())
        first, second = combat.monsters[:2]
        first.current_hp = 11
        first.max_hp = 11
        first.powers["Curl Up"] = 5
        second.current_hp = 14
        second.max_hp = 14
        second.powers["Curl Up"] = 6
        combat.hand = [make_card("Double Tap", uuid="double-tap"), make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = [make_card("Bash", uuid="bash")]
        combat.discard_pile = [
            make_card("Defend_R", uuid="d0"),
            make_card("Strike_R", uuid="d1"),
        ]
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)
        combat.play_card(0, 0)
        self.assertEqual(combat.card_select_context, "HEADBUTT")

        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat.monsters[0].current_hp, 0)
        self.assertEqual(combat.monsters[0].block, 0)

    def test_headbutt_replay_applies_surviving_louse_curl_up_block_before_second_hit_in_v2(self):
        combat = NativeCombatEnv(seed=1090110, ascension_level=0, scheduled_encounter=["GreenLouse", "RedLouse"], player=PlayerState())
        first, second = combat.monsters[:2]
        first.current_hp = 15
        first.max_hp = 15
        first.move = "GREEN_LOUSE_BITE"
        first.intent = "ATTACK"
        first.move_base_damage = 5
        first.move_hits = 1
        first.powers["Curl Up"] = 5
        second.current_hp = 14
        second.max_hp = 14
        second.powers["Curl Up"] = 6
        combat.hand = [make_card("Double Tap", uuid="double-tap"), make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = [make_card("Bash", uuid="bash")]
        combat.discard_pile = [
            make_card("Defend_R", uuid="d0"),
            make_card("Strike_R", uuid="d1"),
        ]
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)
        combat.play_card(0, 0)
        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat.monsters[0].current_hp, 6)

        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat.monsters[0].current_hp, 2)
        self.assertEqual(combat.monsters[0].block, 0)

    def test_headbutt_replay_advances_ink_bottle_between_selects_in_v2(self):
        combat = NativeCombatEnv(seed=109011, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        cultist = combat.monsters[0]
        cultist.current_hp = 22
        cultist.max_hp = 22
        combat.relics.append({
            "relic_id": "Ink Bottle",
            "id": "Ink Bottle",
            "name": "Ink Bottle",
            "counter": 8,
            "price": 0,
            "tier": "UNCOMMON",
        })
        combat.hand = [make_card("Double Tap", uuid="double-tap"), make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = [make_card("Strike_R", uuid="draw-strike")]
        combat.discard_pile = [
            make_card("Defend_R", uuid="d0"),
            make_card("Bash", uuid="d1"),
        ]
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)
        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat._relic("Ink Bottle")["counter"], 0)

        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat._relic("Ink Bottle")["counter"], 1)

    def test_headbutt_replay_resolves_spore_cloud_before_second_select_in_v2(self):
        combat = NativeCombatEnv(seed=109012, ascension_level=0, scheduled_encounter=["FungiBeast", "FungiBeast"], player=PlayerState())
        first, second = combat.monsters[:2]
        first.current_hp = 11
        first.max_hp = 11
        second.current_hp = 24
        second.max_hp = 24
        combat.hand = [make_card("Double Tap", uuid="double-tap"), make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = [make_card("Bash", uuid="bash")]
        combat.discard_pile = [
            make_card("Defend_R", uuid="d0"),
            make_card("Strike_R", uuid="d1"),
        ]
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)
        combat.play_card(0, 0)
        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat.player.power("Vulnerable"), 0)

        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat.player.power("Vulnerable"), 2)

    def test_headbutt_replay_dead_louse_keeps_zero_curl_up_block_after_final_select_in_v2(self):
        combat = NativeCombatEnv(seed=10902, ascension_level=0, scheduled_encounter=["GreenLouse", "RedLouse"], player=PlayerState())
        first, second = combat.monsters[:2]
        first.current_hp = 11
        first.max_hp = 11
        first.powers["Curl Up"] = 5
        second.current_hp = 14
        second.max_hp = 14
        second.powers["Curl Up"] = 6
        combat.hand = [make_card("Double Tap", uuid="double-tap"), make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = [make_card("Bash", uuid="bash")]
        combat.discard_pile = [
            make_card("Defend_R", uuid="d0"),
            make_card("Strike_R", uuid="d1"),
        ]
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)
        combat.play_card(0, 0)
        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})
        self.assertEqual(combat.card_select_context, "HEADBUTT")

        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})

        self.assertIsNone(combat.card_select_context)
        self.assertEqual(combat.monsters[0].current_hp, 0)
        self.assertEqual(combat.monsters[0].block, 0)

    def test_double_tap_immolate_replay_inserts_burn_around_original_card_move_in_v2(self):
        combat = NativeCombatEnv(seed=1091, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        cultist = combat.monsters[0]
        cultist.current_hp = 200
        cultist.max_hp = 200
        combat.hand = [make_card("Double Tap", uuid="double-tap"), make_card("Immolate", uuid="immolate")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3

        combat.play_card(0, 0)
        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.discard_pile], ["Double Tap", "Burn", "Immolate", "Burn"])

    def test_double_tap_replay_counts_for_shuriken_in_v2(self):
        combat = NativeCombatEnv(
            seed=1092,
            ascension_level=0,
            scheduled_encounter=["Cultist"],
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Shuriken", "id": "Shuriken", "name": "Shuriken", "counter": -1, "price": 0, "tier": "COMMON"},
            ],
        )
        cultist = combat.monsters[0]
        cultist.current_hp = 80
        cultist.max_hp = 80
        combat.hand = [make_card("Double Tap", uuid="double-tap"), make_card("Strike_R", uuid="strike-r")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2
        combat.attack_played_this_turn = 1

        combat.play_card(0, 0)
        combat.play_card(0, 0)

        self.assertEqual(combat.player.power("Strength"), 1)

    def test_double_tap_replay_advances_ink_bottle_in_v2(self):
        combat = NativeCombatEnv(
            seed=1093,
            ascension_level=0,
            scheduled_encounter=["Cultist"],
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Ink Bottle", "id": "Ink Bottle", "name": "Ink Bottle", "counter": 7, "price": 0, "tier": "UNCOMMON"},
            ],
        )
        cultist = combat.monsters[0]
        cultist.current_hp = 80
        cultist.max_hp = 80
        combat.hand = [make_card("Double Tap", uuid="double-tap"), make_card("Strike_R", uuid="strike-r")]
        combat.draw_pile = [make_card("Defend_R", uuid="draw-defend")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)
        combat.play_card(0, 0)

        self.assertEqual(combat._relic("Ink Bottle")["counter"], 0)
        self.assertEqual([card.card_id for card in combat.hand], ["Defend_R"])

    def test_double_tap_replay_counts_for_pocketwatch_threshold_in_v2(self):
        combat = NativeCombatEnv(
            seed=1094,
            ascension_level=0,
            scheduled_encounter=["Cultist"],
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Pocketwatch", "id": "Pocketwatch", "name": "Pocketwatch", "counter": -1, "price": 0, "tier": "BOSS"},
            ],
        )
        cultist = combat.monsters[0]
        cultist.current_hp = 200
        cultist.max_hp = 200
        combat.hand = [
            make_card("Double Tap", uuid="double-tap"),
            make_card("Strike_R", uuid="strike-r"),
            make_card("Defend_R", uuid="defend-r"),
        ]
        combat.draw_pile = [make_card("Bash", uuid=f"draw-bash-{idx}") for idx in range(8)]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3
        combat.cards_played_this_turn = 1
        combat.skills_played_this_turn = 1

        combat.play_card(0, 0)
        combat.play_card(0, 0)
        combat.end_turn()

        self.assertNotIn("Pocketwatch Draw", combat.player.powers)

    def test_double_tap_headbutt_replay_counts_for_pocketwatch_threshold_in_v2(self):
        combat = NativeCombatEnv(
            seed=10941,
            ascension_level=0,
            scheduled_encounter=["Cultist"],
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Pocketwatch", "id": "Pocketwatch", "name": "Pocketwatch", "counter": -1, "price": 0, "tier": "BOSS"},
            ],
        )
        cultist = combat.monsters[0]
        cultist.current_hp = 200
        cultist.max_hp = 200
        combat.hand = [
            make_card("Double Tap", uuid="double-tap"),
            make_card("Headbutt", uuid="headbutt"),
            make_card("Defend_R", uuid="defend-r"),
        ]
        combat.discard_pile = [
            make_card("Strike_R", uuid="discard-strike"),
            make_card("Bash", uuid="discard-bash"),
        ]
        combat.draw_pile = [make_card("Anger", uuid=f"draw-anger-{idx}") for idx in range(8)]
        combat.exhaust_pile = []
        combat.player.energy = 3
        combat.cards_played_this_turn = 1
        combat.skills_played_this_turn = 1

        combat.play_card(0, 0)
        combat.play_card(0, 0)
        combat.step({"kind": "card_select", "select_type": "HEADBUTT", "choice_index": 0})
        combat.step({"kind": "card_select", "select_type": "HEADBUTT", "choice_index": 0})
        combat.end_turn()

        self.assertNotIn("Pocketwatch Draw", combat.player.powers)

    def test_double_tap_headbutt_pocketwatch_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8221505361144150152
        result = compare_seed(seed, 0, 280, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_necronomicon_replays_confused_high_cost_anger_in_v2(self):
        combat = NativeCombatEnv(
            seed=1093,
            ascension_level=0,
            scheduled_encounter=["Cultist"],
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Necronomicon", "id": "Necronomicon", "name": "Necronomicon", "counter": -1, "price": 0, "tier": "EVENT"},
            ],
        )
        cultist = combat.monsters[0]
        cultist.current_hp = 40
        cultist.max_hp = 40
        anger = make_card("Anger", uuid="anger")
        anger.cost_for_turn = 2
        combat.hand = [anger]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3
        combat.player.powers["Confusion"] = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.monsters[0].current_hp, 28)
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Anger", "Anger", "Anger"])

    def test_necronomicon_stale_counter_does_not_block_new_combat_replay_in_v2(self):
        combat = NativeCombatEnv(
            seed=1093,
            ascension_level=0,
            scheduled_encounter=["Cultist"],
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Necronomicon", "id": "Necronomicon", "name": "Necronomicon", "counter": 2, "price": 0, "tier": "EVENT"},
            ],
        )
        cultist = combat.monsters[0]
        cultist.current_hp = 40
        cultist.max_hp = 40
        anger = make_card("Anger", uuid="anger")
        anger.cost_for_turn = 2
        combat.hand = [anger]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3
        combat.player.powers["Confusion"] = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.monsters[0].current_hp, 28)
        self.assertEqual(combat._relic("Necronomicon")["counter"], combat.turn)

    def test_deep_breath_advances_sundial_on_manual_shuffle_in_v2(self):
        combat = NativeCombatEnv(
            seed=1094,
            ascension_level=0,
            scheduled_encounter=["Cultist"],
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Sundial", "id": "Sundial", "name": "Sundial", "counter": 2, "price": 0, "tier": "UNCOMMON"},
            ],
        )
        cultist = combat.monsters[0]
        cultist.current_hp = 80
        cultist.max_hp = 80
        combat.hand = [make_card("Deep Breath", uuid="deep-breath")]
        combat.draw_pile = []
        combat.discard_pile = [make_card("Strike_R", uuid="discard-strike")]
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.player.energy, 3)
        self.assertEqual(combat._relic("Sundial")["counter"], 0)
        self.assertEqual([card.card_id for card in combat.hand], ["Strike_R"])

    def test_headbutt_delays_kunai_proc_until_card_select_resolution_in_v2(self):
        combat = NativeCombatEnv(
            seed=110,
            ascension_level=0,
            scheduled_encounter=["Cultist"],
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Kunai", "id": "Kunai", "name": "Kunai", "counter": -1, "price": 0, "tier": "UNCOMMON"},
            ],
        )
        cultist = combat.monsters[0]
        cultist.current_hp = 60
        cultist.max_hp = 60
        combat.hand = [make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = []
        combat.discard_pile = [make_card("Strike_R", uuid="strike-a"), make_card("Strike_R", uuid="strike-b")]
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.attack_played_this_turn = 2

        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat.player.power("Dexterity"), 0)

        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})

        self.assertEqual(combat.player.power("Dexterity"), 1)

    def test_headbutt_delays_ornamental_fan_proc_until_card_select_resolution_in_v2(self):
        combat = NativeCombatEnv(
            seed=111,
            ascension_level=0,
            scheduled_encounter=["Cultist"],
            player=PlayerState(),
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Ornamental Fan", "id": "Ornamental Fan", "name": "Ornamental Fan", "counter": -1, "price": 0, "tier": "COMMON"},
            ],
        )
        cultist = combat.monsters[0]
        cultist.current_hp = 60
        cultist.max_hp = 60
        combat.hand = [make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = []
        combat.discard_pile = [make_card("Strike_R", uuid="strike-a"), make_card("Strike_R", uuid="strike-b")]
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.block = 0
        combat.attack_played_this_turn = 2

        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat.player.block, 0)

        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})

        self.assertEqual(combat.player.block, 4)

    def test_headbutt_delays_rage_block_until_card_select_resolution_in_v2(self):
        combat = NativeCombatEnv(
            seed=112,
            ascension_level=0,
            scheduled_encounter=["Cultist"],
            player=PlayerState(),
        )
        cultist = combat.monsters[0]
        cultist.current_hp = 60
        cultist.max_hp = 60
        combat.hand = [make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = []
        combat.discard_pile = [make_card("Strike_R", uuid="strike-a"), make_card("Strike_R", uuid="strike-b")]
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Rage"] = 3

        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat.player.block, 0)

        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})

        self.assertEqual(combat.player.block, 3)

    def test_headbutt_resolves_pending_spore_cloud_after_card_select_in_v2(self):
        combat = NativeCombatEnv(seed=1111, ascension_level=0, scheduled_encounter=["FungiBeast", "AcidSlime_M"], player=PlayerState())
        fungi = combat.monsters[0]
        slime = combat.monsters[1]
        fungi.current_hp = 11
        fungi.max_hp = 11
        slime.current_hp = 28
        slime.max_hp = 28
        combat.hand = [make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = []
        combat.discard_pile = [
            make_card("Heavy Blade", uuid="heavy-blade"),
            make_card("Strike_R", uuid="strike-r"),
        ]
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.add_power("Strength", 2)

        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat.player.power("Vulnerable"), 2)

        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})

        self.assertEqual(combat.player.power("Vulnerable"), 2)
        self.assertEqual([card.card_id for card in combat.draw_pile], ["Heavy Blade"])

    def test_reckless_charge_defers_gremlin_horn_until_after_dazed_insert_in_v2(self):
        combat = NativeCombatEnv(seed=112, ascension_level=0, scheduled_encounter=["AcidSlime_M"], player=PlayerState())
        combat.relics.append({"relic_id": "Gremlin Horn", "id": "Gremlin Horn", "name": "Gremlin Horn", "counter": -1, "price": 0, "tier": "UNCOMMON"})
        combat.hand = [make_card("Reckless Charge", uuid="reckless-charge")]
        combat.draw_pile = [make_card("Strike_R", uuid=f"strike-{i}") for i in range(14)] + [make_card("Defend_R", uuid="top-defend")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3
        slime = combat.monsters[0]
        slime.current_hp = 7
        slime.max_hp = 7
        slime.block = 0

        class FixedCardRandom:
            def random(self, upper):
                if upper == 14:
                    return 0
                if upper == 13:
                    return 3
                return 0

            def randint(self, low, high):
                return low

        combat.card_random_rng = FixedCardRandom()

        combat.play_card(0, 0)

        self.assertEqual(combat.player.energy, 3)
        self.assertEqual([card.card_id for card in combat.hand], [])
        self.assertEqual(combat.draw_pile[0].card_id, "Dazed")
        self.assertEqual(sum(1 for card in combat.draw_pile if card.card_id == "Dazed"), 1)

    def test_final_fungi_kill_skips_spore_cloud_and_gremlin_horn_in_v2(self):
        combat = NativeCombatEnv(seed=1112, ascension_level=0, scheduled_encounter=["FungiBeast"], player=PlayerState())
        combat.relics.append({"relic_id": "Gremlin Horn", "id": "Gremlin Horn", "name": "Gremlin Horn", "counter": -1, "price": 0, "tier": "UNCOMMON"})
        combat.hand = [make_card("Strike_R", uuid="strike")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        fungi = combat.monsters[0]
        fungi.current_hp = 6
        fungi.max_hp = 6

        combat.play_card(0, 0)

        self.assertEqual(combat.outcome, "PLAYER_VICTORY")
        self.assertEqual(combat.player.energy, 0)
        self.assertEqual(combat.player.power("Vulnerable"), 0)

    def test_nonfinal_fungi_kill_defers_spore_cloud_and_gremlin_horn_in_v2(self):
        combat = NativeCombatEnv(seed=1113, ascension_level=0, scheduled_encounter=["FungiBeast", "JawWorm"], player=PlayerState())
        combat.relics.append({"relic_id": "Gremlin Horn", "id": "Gremlin Horn", "name": "Gremlin Horn", "counter": -1, "price": 0, "tier": "UNCOMMON"})
        combat.hand = [make_card("Strike_R", uuid="strike")]
        combat.draw_pile = [make_card("Defend_R", uuid="defend-top")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        fungi = combat.monsters[0]
        fungi.current_hp = 6
        fungi.max_hp = 6

        combat.play_card(0, 0)

        self.assertEqual(combat.outcome, "UNDECIDED")
        self.assertEqual(combat.player.energy, 1)
        self.assertEqual(combat.player.power("Vulnerable"), 2)
        self.assertEqual([card.card_id for card in combat.hand], ["Defend_R"])

    def test_headbutt_kill_defers_gremlin_horn_until_after_select_in_v2(self):
        combat = NativeCombatEnv(seed=11131, ascension_level=0, scheduled_encounter=["FungiBeast", "JawWorm"], player=PlayerState())
        combat.relics.append({"relic_id": "Gremlin Horn", "id": "Gremlin Horn", "name": "Gremlin Horn", "counter": -1, "price": 0, "tier": "UNCOMMON"})
        combat.hand = [make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = [make_card("Bash", uuid="top-bash")]
        combat.discard_pile = [make_card("Defend_R", uuid="discard-defend"), make_card("Flex", uuid="discard-flex")]
        combat.exhaust_pile = []
        combat.player.energy = 1
        fungi = combat.monsters[0]
        fungi.current_hp = 9
        fungi.max_hp = 9

        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat.player.energy, 0)
        self.assertEqual([card.card_id for card in combat.hand], [])
        self.assertEqual(combat.pending_monster_kill_triggers, 1)

    def test_pommel_strike_plus_keeps_played_card_out_of_shuffle_until_draw_completes_in_v2(self):
        combat = NativeCombatEnv(seed=11132, ascension_level=0, player=PlayerState())
        combat.monsters = [
            make_monster("GreenLouse", StsRandom(1), ascension=0),
        ]
        combat.monsters[0].current_hp = 10
        combat.monsters[0].max_hp = 10
        combat.hand = [make_card("Strike_R", uuid="strike"), make_card("Injury", uuid="injury"), make_card("Pommel Strike", upgrades=1, uuid="pommel")]
        combat.draw_pile = [make_card("Defend_R", uuid="draw-defend")]
        combat.discard_pile = [make_card("Flex", uuid="discard-flex")]
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Evolve"] = 2

        combat.play_card(2, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Strike_R", "Injury", "Defend_R", "Flex"])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Pommel Strike"])

    def test_shrug_it_off_moves_to_discard_before_evolve_followup_shuffle_in_v2(self):
        combat = NativeCombatEnv(seed=881472999803563354, ascension_level=0, player=PlayerState())
        combat.monsters = [
            make_monster("Sentry", StsRandom(1), ascension=0),
            make_monster("Sentry", StsRandom(2), ascension=0),
            make_monster("Sentry", StsRandom(3), ascension=0),
        ]
        combat.monsters[0].current_hp = 0
        combat.monsters[0].is_gone = True
        combat.monsters[0].powers["Vulnerable"] = 1
        combat.monsters[1].current_hp = 0
        combat.monsters[1].is_gone = True
        combat.monsters[2].current_hp = 33
        combat.monsters[2].max_hp = 33
        combat.hand = [
            make_card("Strike_R", uuid="hand-strike-0"),
            make_card("Dazed", uuid="hand-dazed-0"),
            make_card("Strike_R", uuid="hand-strike-1"),
            make_card("Dazed", uuid="hand-dazed-1"),
            make_card("Strike_R", uuid="hand-strike-2"),
            make_card("Dazed", uuid="hand-dazed-2"),
            make_card("Shrug It Off", uuid="shrug"),
        ]
        combat.draw_pile = [make_card("Defend_R", uuid="draw-defend")]
        combat.discard_pile = [
            make_card("Dazed", uuid="discard-dazed-0"),
            make_card("Defend_R", uuid="discard-defend-0"),
            make_card("Defend_R", uuid="discard-defend-1"),
            make_card("Bash", uuid="discard-bash"),
            make_card("Pommel Strike", uuid="discard-pommel"),
            make_card("Shrug It Off", uuid="discard-shrug"),
            make_card("Dazed", uuid="discard-dazed-1"),
            make_card("Strike_R", uuid="discard-strike-3"),
            make_card("Dazed", uuid="discard-dazed-2"),
            make_card("Armaments", uuid="discard-armaments"),
            make_card("Dazed", uuid="discard-dazed-3"),
            make_card("Strike_R", uuid="discard-strike-4"),
            make_card("Defend_R", uuid="discard-defend-2"),
            make_card("Reckless Charge", uuid="discard-reckless"),
            make_card("Heavy Blade", uuid="discard-heavy"),
        ]
        combat.exhaust_pile = [make_card("Disarm", uuid="exhaust-disarm")] + [make_card("Dazed", uuid=f"exhaust-dazed-{i}") for i in range(17)]
        combat.player.energy = 1
        combat.player.powers["Evolve"] = 1

        combat.play_card(6, 0)

        self.assertEqual(
            [card.card_id for card in combat.hand],
            ["Strike_R", "Dazed", "Strike_R", "Dazed", "Strike_R", "Dazed", "Defend_R"],
        )
        self.assertEqual(combat.discard_pile[-1].card_id, "Shrug It Off")

    def test_shrug_it_off_does_not_move_before_ink_bottle_shuffle_when_first_draw_is_not_status_in_v2(self):
        combat = NativeCombatEnv(seed=881472999803563355, ascension_level=0, player=PlayerState())
        combat.monsters = [make_monster("JawWorm", StsRandom(1), ascension=0)]
        combat.relics.append(make_relic("Ink Bottle", counter=9))
        combat.hand = [
            make_card("Shrug It Off", uuid="shrug"),
            make_card("Strike_R", uuid="hand-strike-0"),
            make_card("Strike_R", uuid="hand-strike-1"),
        ]
        combat.draw_pile = [make_card("Sword Boomerang", uuid="draw-sword-boomerang")]
        combat.discard_pile = [
            make_card("Defend_R", uuid="discard-defend-0"),
            make_card("Defend_R", uuid="discard-defend-1"),
            make_card("Flex", uuid="discard-flex"),
            make_card("Strike_R", uuid="discard-strike-0"),
            make_card("Armaments", uuid="discard-armaments-0"),
            make_card("Armaments", uuid="discard-armaments-1"),
            make_card("Shrug It Off", uuid="discard-shrug"),
            make_card("Bash", uuid="discard-bash"),
            make_card("Defend_R", uuid="discard-defend-2"),
            make_card("Defend_R", uuid="discard-defend-3"),
        ]
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Evolve"] = 1

        combat.play_card(0, 0)

        self.assertEqual(
            [card.card_id for card in combat.hand[:3]],
            ["Strike_R", "Strike_R", "Sword Boomerang"],
        )
        self.assertEqual(combat.hand[3].card_def.card_type, "SKILL")
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Shrug It Off"])
        self.assertEqual(sum(1 for card in combat.draw_pile if card.card_id == "Shrug It Off"), 1)


    def test_double_tap_replay_flushes_spore_cloud_in_v2(self):
        combat = NativeCombatEnv(seed=11131, ascension_level=0, player=PlayerState())
        combat.monsters = [
            make_monster("FungiBeast", StsRandom(1), ascension=0),
            make_monster("FungiBeast", StsRandom(2), ascension=0),
            make_monster("FungiBeast", StsRandom(3), ascension=0),
        ]
        combat.monsters[0].current_hp = 11
        combat.monsters[0].max_hp = 11
        combat.monsters[0].powers["Weakened"] = 2
        combat.monsters[1].current_hp = 24
        combat.monsters[1].max_hp = 24
        combat.monsters[2].current_hp = 23
        combat.monsters[2].max_hp = 23
        combat.hand = [make_card("Dramatic Entrance", uuid="dramatic")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Double Tap"] = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.player.power("Vulnerable"), 2)
        self.assertEqual([monster.current_hp for monster in combat.monsters], [0, 8, 7])

    def test_player_turn_spore_cloud_decrements_at_end_of_round_in_v2(self):
        combat = NativeCombatEnv(seed=1114, ascension_level=0, scheduled_encounter=["FungiBeast", "JawWorm"], player=PlayerState())
        combat.hand = [make_card("Strike_R", uuid="strike")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        fungi = combat.monsters[0]
        fungi.current_hp = 6
        fungi.max_hp = 6
        jaw = combat.monsters[1]
        jaw.move = "JAW_WORM_CHOMP"
        jaw.intent = "ATTACK"
        jaw.move_base_damage = 0
        jaw.move_hits = 1

        combat.play_card(0, 0)
        self.assertEqual(combat.player.power("Vulnerable"), 2)

        combat.end_turn()

        self.assertEqual(combat.player.power("Vulnerable"), 1)

    def test_dead_louse_still_rolls_next_move_after_thorns_kill_in_v2(self):
        combat = NativeCombatEnv(seed=1116, ascension_level=0, scheduled_encounter=["RedLouse", "JawWorm"], player=PlayerState())
        louse = combat.monsters[0]
        jaw = combat.monsters[1]
        louse.current_hp = 1
        louse.max_hp = 11
        louse.move = "RED_LOUSE_BITE"
        louse.intent = "ATTACK"
        louse.move_base_damage = 5
        louse.move_hits = 1
        louse.move_history = ["RED_LOUSE_BITE"]
        jaw.move = "JAW_WORM_CHOMP"
        jaw.intent = "ATTACK"
        jaw.move_base_damage = 0
        jaw.move_hits = 1
        jaw.move_history = ["JAW_WORM_CHOMP"]
        combat.player.current_hp = 80
        combat.player.block = 0
        combat.player.add_power("Thorns", 3)

        class FixedAiRng:
            def random(self, upper):
                return 10

            def random_boolean(self, chance=None):
                return False

        combat.ai_rng = FixedAiRng()

        combat.end_turn()

        self.assertEqual(louse.current_hp, 0)
        self.assertEqual(louse.move, "RED_LOUSE_GROW")

    def test_dead_spike_slime_still_rolls_next_move_after_thorns_kill_in_v2(self):
        combat = NativeCombatEnv(seed=1117, ascension_level=0, scheduled_encounter=["SpikeSlime_M", "JawWorm"], player=PlayerState())
        slime = combat.monsters[0]
        jaw = combat.monsters[1]
        slime.current_hp = 1
        slime.max_hp = 28
        slime.move = "SPIKE_SLIME_M_FLAME_TACKLE"
        slime.intent = "ATTACK_DEBUFF"
        slime.move_base_damage = 8
        slime.move_hits = 1
        slime.move_history = ["SPIKE_SLIME_M_FLAME_TACKLE", "SPIKE_SLIME_M_FLAME_TACKLE"]
        jaw.move = "JAW_WORM_CHOMP"
        jaw.intent = "ATTACK"
        jaw.move_base_damage = 0
        jaw.move_hits = 1
        jaw.move_history = ["JAW_WORM_CHOMP"]
        combat.player.current_hp = 80
        combat.player.block = 0
        combat.player.add_power("Thorns", 3)

        class FixedAiRng:
            def random(self, upper):
                return 10

            def random_boolean(self, chance=None):
                return False

        combat.ai_rng = FixedAiRng()

        combat.end_turn()

        self.assertEqual(slime.current_hp, 0)
        self.assertEqual(slime.move, "SPIKE_SLIME_M_LICK")

    def test_monster_turn_spore_cloud_stays_just_applied_in_v2(self):
        combat = NativeCombatEnv(seed=1115, ascension_level=0, scheduled_encounter=["FungiBeast", "JawWorm"], player=PlayerState())
        combat.player.powers["Thorns"] = 3
        fungi = combat.monsters[0]
        fungi.current_hp = 2
        fungi.max_hp = 6
        fungi.move = "FUNGI_BEAST_BITE"
        fungi.intent = "ATTACK"
        fungi.move_base_damage = 6
        fungi.move_hits = 1
        jaw = combat.monsters[1]
        jaw.move = "JAW_WORM_CHOMP"
        jaw.intent = "ATTACK"
        jaw.move_base_damage = 0
        jaw.move_hits = 1

        combat.end_turn()

        self.assertEqual(combat.player.power("Vulnerable"), 2)

    def test_end_turn_combust_kill_resolves_gremlin_horn_before_next_draw_in_v2(self):
        combat = NativeCombatEnv(seed=2115, ascension_level=0, scheduled_encounter=["AcidSlime_M", "JawWorm"], player=PlayerState())
        combat.relics.append({"relic_id": "Gremlin Horn", "id": "Gremlin Horn", "name": "Gremlin Horn", "counter": -1, "price": 0, "tier": "UNCOMMON"})
        combat.hand = []
        combat.draw_pile = [
            make_card("Strike_R", uuid="draw-1"),
            make_card("Strike_R", uuid="draw-2"),
            make_card("Strike_R", uuid="draw-3"),
            make_card("Strike_R", uuid="draw-4"),
            make_card("Strike_R", uuid="draw-5"),
            make_card("Iron Wave", uuid="bonus-top"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 0
        combat.player.powers["Combust"] = 5
        combat.combust_hp_loss = 1
        slime = combat.monsters[0]
        slime.current_hp = 5
        slime.max_hp = 28
        jaw = combat.monsters[1]
        jaw.move = "JAW_WORM_CHOMP"
        jaw.intent = "ATTACK"
        jaw.move_base_damage = 0
        jaw.move_hits = 1

        combat.end_turn()

        self.assertEqual(combat.player.energy, 3)
        self.assertEqual(len(combat.hand), 6)
        self.assertIn("Iron Wave", [card.name for card in combat.hand])
        self.assertEqual(combat.pending_monster_kill_triggers, 0)

    def test_monster_turn_thorns_kill_resolves_gremlin_horn_before_next_draw_in_v2(self):
        combat = NativeCombatEnv(seed=2116, ascension_level=0, scheduled_encounter=["RedLouse", "JawWorm"], player=PlayerState())
        combat.relics.append({"relic_id": "Gremlin Horn", "id": "Gremlin Horn", "name": "Gremlin Horn", "counter": -1, "price": 0, "tier": "UNCOMMON"})
        combat.hand = []
        combat.draw_pile = [
            make_card("Strike_R", uuid="draw-a"),
            make_card("Strike_R", uuid="draw-b"),
            make_card("Strike_R", uuid="draw-c"),
            make_card("Strike_R", uuid="draw-d"),
            make_card("Strike_R", uuid="draw-e"),
            make_card("Iron Wave", uuid="bonus-top"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 0
        combat.player.add_power("Thorns", 3)
        louse = combat.monsters[0]
        louse.current_hp = 1
        louse.max_hp = 11
        louse.move = "RED_LOUSE_BITE"
        louse.intent = "ATTACK"
        louse.move_base_damage = 5
        louse.move_hits = 1
        jaw = combat.monsters[1]
        jaw.move = "JAW_WORM_CHOMP"
        jaw.intent = "ATTACK"
        jaw.move_base_damage = 0
        jaw.move_hits = 1

        class FixedAiRng:
            def random(self, upper):
                return 10

            def random_boolean(self, chance=None):
                return False

        combat.ai_rng = FixedAiRng()

        combat.end_turn()

        self.assertEqual(combat.player.energy, 3)
        self.assertEqual(len(combat.hand), 6)
        self.assertIn("Iron Wave", [card.name for card in combat.hand])
        self.assertEqual(combat.pending_monster_kill_triggers, 0)

    def test_fire_breathing_draw_kill_resolves_gremlin_horn_immediately_in_v2(self):
        combat = NativeCombatEnv(seed=2117, ascension_level=0, scheduled_encounter=["AcidSlime_M", "JawWorm"], player=PlayerState())
        combat.relics.append({"relic_id": "Gremlin Horn", "id": "Gremlin Horn", "name": "Gremlin Horn", "counter": -1, "price": 0, "tier": "UNCOMMON"})
        combat.hand = []
        combat.draw_pile = [
            make_card("Strike_R", uuid="draw-1"),
            make_card("Strike_R", uuid="draw-2"),
            make_card("Strike_R", uuid="draw-3"),
            make_card("Strike_R", uuid="draw-4"),
            make_card("Strike_R", uuid="draw-5"),
            make_card("Injury", uuid="injury-top"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3
        combat.player.powers["Fire Breathing"] = 6
        slime = combat.monsters[0]
        slime.current_hp = 6
        slime.max_hp = 28

        combat.draw_cards(1)

        self.assertEqual(combat.player.energy, 4)
        self.assertEqual(combat.pending_monster_kill_triggers, 0)
        self.assertEqual([card.card_id for card in combat.hand], ["Injury", "Strike_R"])

    def test_unceasing_top_status_draw_defers_fire_breathing_but_not_evolve_in_v2(self):
        combat = NativeCombatEnv(seed=21171, ascension_level=0, scheduled_encounter=["ShelledParasite"], player=PlayerState())
        combat.relics.append({"relic_id": "Unceasing Top", "id": "Unceasing Top", "name": "Unceasing Top", "counter": -1, "price": 0, "tier": "UNCOMMON"})
        combat.player.powers["Fire Breathing"] = 6
        combat.player.powers["Evolve"] = 1
        combat.hand = [make_card("Blind", uuid="blind")]
        combat.draw_pile = [
            make_card("Inflame", uuid="inflame"),
            make_card("Injury", uuid="injury"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        parasite = combat.monsters[0]
        parasite.current_hp = 21
        parasite.max_hp = 21
        parasite.block = 4

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Injury"])
        self.assertEqual(combat.pending_start_turn_fire_breathing_damage, [6])
        self.assertEqual(parasite.current_hp, 21)
        self.assertEqual(parasite.block, 4)

    def test_flame_barrier_damage_hits_monster_block_before_hp_in_v2(self):
        combat = NativeCombatEnv(seed=2118, ascension_level=0, scheduled_encounter=["RedLouse"], player=PlayerState())
        louse = combat.monsters[0]
        louse.current_hp = 10
        louse.max_hp = 10
        louse.block = 5
        louse.move = "RED_LOUSE_BITE"
        louse.intent = "ATTACK"
        louse.move_base_damage = 5
        louse.move_hits = 1
        combat.player.powers["Flame Barrier"] = 3

        combat._monster_take_turn(louse)

        self.assertEqual(louse.current_hp, 10)
        self.assertEqual(louse.block, 2)

    def test_pummel_triggers_sharp_hide_once_per_card_in_v2(self):
        combat = NativeCombatEnv(seed=1109, ascension_level=0, scheduled_encounter=["TheGuardian"], player=PlayerState())
        guardian = combat.monsters[0]
        guardian.current_hp = 200
        guardian.max_hp = 250
        guardian.powers["Sharp Hide"] = 3
        combat.player.current_hp = 40
        combat.player.block = 0
        combat.player.energy = 1
        combat.hand = [make_card("Pummel", uuid="pummel")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 37)

    def test_pummel_third_attack_shuriken_applies_after_all_hits_in_v2(self):
        combat = NativeCombatEnv(seed=1110, ascension_level=0, scheduled_encounter=["JawWorm"], player=PlayerState())
        jaw_worm = combat.monsters[0]
        jaw_worm.current_hp = 40
        jaw_worm.max_hp = 40
        combat.relics.append({"relic_id": "Shuriken", "name": "Shuriken", "tier": "COMMON", "counter": -1})
        combat.player.energy = 1
        combat.attack_played_this_turn = 2
        combat.hand = [make_card("Pummel", uuid="pummel")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertEqual(jaw_worm.current_hp, 32)
        self.assertEqual(combat.player.power("Strength"), 1)

    def test_sword_boomerang_triggers_sharp_hide_once_against_single_target_in_v2(self):
        combat = NativeCombatEnv(seed=1110, ascension_level=0, scheduled_encounter=["TheGuardian"], player=PlayerState())
        guardian = combat.monsters[0]
        guardian.current_hp = 200
        guardian.max_hp = 250
        guardian.powers["Sharp Hide"] = 3
        combat.player.current_hp = 40
        combat.player.block = 10
        combat.player.energy = 1
        combat.hand = [make_card("Sword Boomerang", uuid="boomerang")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertEqual(combat.player.block, 7)

    def test_bottled_flame_requires_non_basic_attack_in_v2(self):
        self.assertFalse(relic_can_spawn("Bottled Flame", deck=[make_card("Strike_R", uuid="strike")]))
        self.assertTrue(relic_can_spawn("Bottled Flame", deck=[make_card("Carnage", uuid="carnage")]))

    def test_bottled_lightning_requires_non_basic_skill_in_v2(self):
        self.assertFalse(relic_can_spawn("Bottled Lightning", deck=[make_card("Defend_R", uuid="defend")]))
        self.assertTrue(relic_can_spawn("Bottled Lightning", deck=[make_card("Armaments", uuid="armaments")]))

    def test_tiny_chest_treasure_still_consumes_event_rng_in_v2(self):
        env = NativeRunEnv(seed=1110, ascension_level=0)
        env.relics.append({"relic_id": "Tiny Chest", "name": "Tiny Chest", "counter": 3})
        before = env.randoms.event.counter

        outcome = env._question_room_outcome(last_room_was_shop=False)

        self.assertEqual(outcome, "T")
        self.assertEqual(env.randoms.event.counter, before + 1)
        self.assertEqual(env._relic("Tiny Chest")["counter"], 0)
        self.assertAlmostEqual(env.monster_chance, 0.20, places=6)
        self.assertAlmostEqual(env.shop_chance, 0.06, places=6)
        self.assertAlmostEqual(env.treasure_chance, 0.02, places=6)

    def test_juzu_bracelet_question_monster_still_resets_monster_chance_in_v2(self):
        env = NativeRunEnv(seed=1110, ascension_level=0)
        env.relics.append({"relic_id": "Juzu Bracelet", "name": "Juzu Bracelet", "counter": -1})
        env.monster_chance = 0.30
        env.shop_chance = 0.09
        env.treasure_chance = 0.06

        class FixedEventRng:
            counter = 2

            def random(self):
                return 0.15

        env.randoms.event = FixedEventRng()

        outcome = env._question_room_outcome(last_room_was_shop=False)

        self.assertEqual(outcome, "?")
        self.assertAlmostEqual(env.monster_chance, 0.10, places=6)
        self.assertAlmostEqual(env.shop_chance, 0.12, places=6)
        self.assertAlmostEqual(env.treasure_chance, 0.08, places=6)

    def test_colosseum_is_disabled_in_event_pool_in_v2(self):
        env = NativeRunEnv(seed=1112, ascension_level=0)
        env.act = 2
        env.floor = 26
        self.assertFalse(env._can_add_event("Colosseum"))

    def test_dead_adventurer_event_combat_can_roll_bottled_relic_in_v2(self):
        env = NativeRunEnv(seed=1111, ascension_level=0)
        env.phase = "EVENT"
        env.current_event_id = "Dead Adventurer"
        env.event_state = {"phase": 0, "rewards": [2], "encounter": "Gremlin Nob"}
        env._roll_relic_tier_for_act = lambda act: "UNCOMMON"
        env._add_card_to_deck("Carnage", uuid="dead-adventurer-carnage")
        env.relic_pools["UNCOMMON"] = ["Bottled Flame", "Eternal Feather", "Strike Dummy"]

        class FixedMisc:
            def copy(self):
                return self

            def random(self, *args):
                if len(args) == 1:
                    return 0
                if len(args) == 2:
                    return args[0]
                return 0.0

        env.randoms.misc = FixedMisc()
        env.step({"kind": "event", "event_id": "Dead Adventurer", "name": "Searched", "choice_index": 0})

        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual(env.pending_event_relic_id, "Bottled Flame")

    def test_ancient_tea_set_persists_through_question_room_before_event_combat_in_v2(self):
        env = NativeRunEnv(seed=1113, ascension_level=0, start_on_map=True)
        env.relics.append({"relic_id": "Ancient Tea Set", "name": "Ancient Tea Set", "tier": "COMMON", "counter": 1})
        env.current_node_symbol = "R"
        env._question_room_outcome = lambda last_room_was_shop=False: "?"
        env._draw_event_id = lambda: "Dead Adventurer"

        env._advance_to_node("?")

        self.assertEqual(env.phase, "EVENT")
        self.assertEqual(env._relic("Ancient Tea Set")["counter"], 1)

    def test_dead_adventurer_ancient_tea_set_reward_does_not_apply_to_followup_event_combat_in_v2(self):
        env = NativeRunEnv(seed=1114, ascension_level=0)
        env.phase = "EVENT"
        env.current_event_id = "Dead Adventurer"
        env.event_state = {"phase": 0, "rewards": [2, 1], "encounter": "Gremlin Nob"}
        env._roll_relic_tier_for_act = lambda act: "COMMON"
        env._roll_screenless_relic_of_tier = lambda tier: make_relic("Ancient Tea Set")

        class FixedMisc:
            def __init__(self):
                self.calls = 0

            def random(self, *args):
                self.calls += 1
                if len(args) == 1:
                    return 99 if self.calls == 1 else 0
                if len(args) == 2:
                    return args[0]
                return 0.0

            def copy(self):
                return self

        env.randoms.misc = FixedMisc()
        env.step({"kind": "event", "event_id": "Dead Adventurer", "name": "Searched", "choice_index": 0})
        self.assertEqual(env._relic("Ancient Tea Set")["counter"], -1)

        env.step({"kind": "event", "event_id": "Dead Adventurer", "name": "Searched", "choice_index": 0})

        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual(env.combat.player.energy, 3)
        self.assertEqual(env.combat._relic("Ancient Tea Set")["counter"], -1)

    def test_ancient_tea_set_clears_when_entering_shop_in_v2(self):
        env = NativeRunEnv(seed=1115, ascension_level=0, start_on_map=True)
        env.relics.append({"relic_id": "Ancient Tea Set", "name": "Ancient Tea Set", "tier": "COMMON", "counter": 1})
        env.current_node_symbol = "R"

        env._advance_to_node("$")

        self.assertEqual(env.phase, "SHOP")
        self.assertEqual(env._relic("Ancient Tea Set")["counter"], 0)

    def test_ancient_tea_set_clears_when_entering_treasure_room_in_v2(self):
        env = NativeRunEnv(seed=1116, ascension_level=0, start_on_map=True)
        env.relics.append({"relic_id": "Ancient Tea Set", "name": "Ancient Tea Set", "tier": "COMMON", "counter": 1})
        env.current_node_symbol = "R"

        env._advance_to_node("T")

        self.assertEqual(env.phase, "TREASURE")
        self.assertEqual(env._relic("Ancient Tea Set")["counter"], 0)

    def test_lightspeed_burning_elite_overlay_maps_by_elite_ordinal_in_v2(self):
        env = NativeRunEnv(seed=2880820615560772120, ascension_level=0, enable_neow=True)

        floor7_nodes = [env.map_graph[node_id] for node_id in env.map_layers.get(7, [])]
        floor7_by_x = {node["x"]: node for node in floor7_nodes}

        self.assertEqual(floor7_by_x[3]["symbol"], "E_GREEN")
        self.assertEqual(floor7_by_x[3].get("burning_elite_buff"), 0)
        self.assertEqual(floor7_by_x[4]["symbol"], "M")

    def test_ancient_tea_set_applies_to_dead_adventurer_event_combat_in_v2(self):
        env = NativeRunEnv(seed=1114, ascension_level=0)
        env.relics.append({"relic_id": "Ancient Tea Set", "name": "Ancient Tea Set", "tier": "COMMON", "counter": 1})
        env.phase = "EVENT"
        env.current_event_id = "Dead Adventurer"
        env.event_state = {"phase": 0, "rewards": [1], "encounter": "Gremlin Nob"}

        class FixedMisc:
            def copy(self):
                return self

            def random(self, *args):
                if len(args) == 1:
                    return 0
                if len(args) == 2:
                    return args[0]
                return 0.0

            def choice(self, items):
                return items[0]

        env.randoms.misc = FixedMisc()

        env.step({"kind": "event", "event_id": "Dead Adventurer", "name": "Searched", "choice_index": 0})

        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual(env.combat.player.energy, 5)
        self.assertEqual(env.combat._relic("Ancient Tea Set")["counter"], 0)

    def test_ancient_tea_set_clears_after_regular_question_event_in_v2(self):
        env = NativeRunEnv(seed=11145, ascension_level=0, start_on_map=True)
        env.relics.append({"relic_id": "Ancient Tea Set", "name": "Ancient Tea Set", "tier": "COMMON", "counter": 1})
        env.phase = "EVENT"
        env.current_node_symbol = "?"
        env.current_event_id = "Big Fish"
        env.event_options = [
            {"kind": "event", "event_id": "Big Fish", "name": "Banana", "choice_index": 0},
            {"kind": "event", "event_id": "Big Fish", "name": "Donut", "choice_index": 1},
            {"kind": "event", "event_id": "Big Fish", "name": "Box", "choice_index": 2},
        ]

        env.step({"kind": "event", "event_id": "Big Fish", "name": "Banana", "choice_index": 0})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env._relic("Ancient Tea Set")["counter"], 0)

    def test_ancient_tea_set_clears_after_golden_idol_advance_floor_in_v2(self):
        env = NativeRunEnv(seed=11146, ascension_level=0, start_on_map=True)
        env.relics.append({"relic_id": "Ancient Tea Set", "name": "Ancient Tea Set", "tier": "COMMON", "counter": 1})
        env.phase = "EVENT"
        env.current_node_symbol = "?"
        env.current_event_id = "Golden Idol"
        env.event_options = [
            {"kind": "event", "event_id": "Golden Idol", "name": "Outrun", "choice_index": 2},
        ]

        env.step({"kind": "event", "event_id": "Golden Idol", "name": "Outrun", "choice_index": 2})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env._relic("Ancient Tea Set")["counter"], 0)

    def test_ancient_tea_set_clears_after_question_event_remove_card_select_in_v2(self):
        env = NativeRunEnv(seed=11147, ascension_level=0, start_on_map=True)
        env.relics.append({"relic_id": "Ancient Tea Set", "name": "Ancient Tea Set", "tier": "COMMON", "counter": 1})
        env.phase = "CARD_SELECT"
        env.current_node_symbol = "?"
        env.card_select_context = "EVENT_REMOVE"
        env.card_select_count = 1
        env.card_select_available_indexes = [0]
        env.card_select_options = [{"kind": "card_select", "name": "REMOVE", "choice_index": 0, "target_index": 0}]
        env.deck = [make_card("Strike_R", uuid="event-remove-strike")]

        env.step({"kind": "card_select", "choice_index": 0})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env._relic("Ancient Tea Set")["counter"], 0)

    def test_ancient_tea_set_arms_after_campfire_smith_card_select_in_v2(self):
        env = NativeRunEnv(seed=11148, ascension_level=0, start_on_map=True)
        env.relics.append({"relic_id": "Ancient Tea Set", "name": "Ancient Tea Set", "tier": "COMMON", "counter": 0})
        env.phase = "CARD_SELECT"
        env.current_node_symbol = "R"
        env.card_select_context = "CAMPFIRE_SMITH"
        env.card_select_completion = "CAMPFIRE_LEAVE"
        env.card_select_count = 1
        env.card_select_available_indexes = [0]
        env.card_select_options = [{"kind": "card_select", "name": "UPGRADE", "choice_index": 0, "target_index": 0}]
        env.deck = [make_card("Strike_R", uuid="campfire-smith-strike")]

        env.step({"kind": "card_select", "choice_index": 0})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env._relic("Ancient Tea Set")["counter"], 1)

    def test_ancient_tea_set_from_rest_reward_applies_to_next_monster_in_v2(self):
        env = NativeRunEnv(seed=1117, ascension_level=0, start_on_map=True)
        env.floor = 6
        env.phase = "CARD_REWARD"
        env.reward_context = "REST"
        env.reward_relics = [make_relic("Ancient Tea Set")]
        env.reward_cards = []
        env.reward_card_bundles = []
        env.reward_gold = 0
        env.reward_gold_piles = []
        env.reward_potions = []
        env.reward_emerald_key = False
        env.current_node_symbol = "R"

        env.step({"kind": "reward_relic", "reward_index": 0, "relic_id": "Ancient Tea Set"})
        env.step({"kind": "skip", "name": "SKIP"})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env._relic("Ancient Tea Set")["counter"], 1)

        env.current_node_symbol = "R"
        env._advance_to_node("M")

        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual(env.combat.player.energy, 5)
        self.assertEqual(env.combat._relic("Ancient Tea Set")["counter"], 0)

    def test_end_turn_charons_ashes_kill_still_applies_combust_self_damage_in_v2(self):
        combat = NativeCombatEnv(seed=1118, ascension_level=0, player=PlayerState())
        combat.relics.append(make_relic("Charon's Ashes"))
        looter = make_monster("Looter", StsRandom(1118), ascension=0)
        looter.current_hp = 2
        looter.max_hp = 48
        combat.monsters = [looter]
        combat.hand = [make_card("Ghostly Armor", uuid="ghostly-armor"), make_card("Strike_R", uuid="strike"), make_card("Defend_R", uuid="defend")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.current_hp = 53
        combat.player.max_hp = 80
        combat.player.powers["Combust"] = 5
        combat.combust_hp_loss = 1

        combat.end_turn()

        self.assertEqual(combat.outcome, "PLAYER_VICTORY")
        self.assertEqual(combat.player.current_hp, 52)

    def test_stone_calendar_kill_does_not_force_combust_self_damage_in_v2(self):
        seed = 1114838182510209823
        result = compare_seed(seed, 0, 210, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_black_blood_meat_on_the_bone_reward_heal_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8114211754523484236
        result = compare_seed(seed, 0, 353, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_burning_blood_meat_on_the_bone_reward_heal_regression_seed_matches_lightspeed_in_v2(self):
        seed = 6418863941812289433
        result = compare_seed(seed, 0, 180, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_mercury_hourglass_brutality_reward_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1405789270657645553
        result = compare_seed(seed, 0, 180, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_sacred_bark_fairy_potion_red_skull_regression_seed_matches_lightspeed_in_v2(self):
        seed = 4082055817911646691
        result = compare_seed(seed, 0, 340, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_nloth_event_exposes_one_choice_per_relic_in_v2(self):
        env = NativeRunEnv(seed=1112, ascension_level=0)
        env.relics = [
            {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood"},
            {"relic_id": "Kunai", "id": "Kunai", "name": "Kunai"},
            {"relic_id": "Anchor", "id": "Anchor", "name": "Anchor"},
            {"relic_id": "Coffee Dripper", "id": "Coffee Dripper", "name": "Coffee Dripper"},
        ]
        env._draw_event_id = lambda: "N'loth"
        env.randoms.misc = StsRandom(0)
        env._enter_event()

        self.assertEqual(len(env.event_options), 3)
        self.assertEqual([option["name"] for option in env.event_options], ["Gave Relic", "Gave Relic", "Ignored"])

    def test_nloth_can_offer_starting_relic_in_v2(self):
        env = NativeRunEnv(seed=1113, ascension_level=0)
        env.relics = [
            {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood"},
            {"relic_id": "Kunai", "id": "Kunai", "name": "Kunai"},
            {"relic_id": "Anchor", "id": "Anchor", "name": "Anchor"},
        ]
        choices = env._nloth_relic_choices()
        self.assertEqual([index for index, _ in choices], [0, 1, 2])

    def test_nloth_two_relic_choices_still_shuffle_order_in_v2(self):
        env = NativeRunEnv(seed=1114, ascension_level=0)
        env.relics = [
            {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood"},
            {"relic_id": "Busted Crown", "id": "Busted Crown", "name": "Busted Crown"},
        ]
        env._draw_event_id = lambda: "N'loth"
        env.randoms.misc = StsRandom(0)

        env._enter_event()

        self.assertEqual(
            [option["label"] for option in env.event_options],
            ["Busted Crown", "Burning Blood", "Ignored"],
        )

    def test_the_mausoleum_opened_grants_relic_reward_then_applies_writhe_in_v2(self):
        env = NativeRunEnv(seed=11121, ascension_level=0)
        env.phase = "EVENT"
        env.current_event_id = "The Mausoleum"
        env._roll_screenless_relic_of_tier = lambda tier: make_relic("Bag of Marbles")

        class FixedMisc:
            def random_boolean(self):
                return True

        env.randoms.misc = FixedMisc()
        env.step({"kind": "event", "event_id": "The Mausoleum", "name": "Opened", "choice_index": 0})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual([relic.get("relic_id") for relic in env.reward_relics], ["Bag of Marbles"])
        self.assertIn("Writhe", [card.card_id for card in env.deck])

    def test_the_woman_in_blue_spends_gold_before_opening_potion_reward_in_v2(self):
        env = NativeRunEnv(seed=11122, ascension_level=0, start_on_map=True)
        env.phase = "EVENT"
        env.current_event_id = "The Woman in Blue"
        env.gold = 99

        env.step({"kind": "event", "event_id": "The Woman in Blue", "name": "Bought 2 Potions", "choice_index": 1})

        self.assertEqual(env.gold, 69)
        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(len(env.reward_potions), 2)

    def test_seed2_floor2_question_room_rolls_the_woman_in_blue_in_v2(self):
        trace_path = Path("/home/yydd/spirecomm/_cache/real_game_first/traces/seed_2_trace.json")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        env = NativeRunEnv(seed=2, ascension_level=0, enable_neow=True)

        def _find_action(recorded_action: dict[str, object]) -> dict[str, object]:
            live_actions = env.legal_actions()
            kind = recorded_action.get("kind")
            name = recorded_action.get("name")
            choice_index = recorded_action.get("choice_index")
            target_index = recorded_action.get("target_index")
            for action in live_actions:
                if action.get("kind") != kind:
                    continue
                if name is not None and action.get("name") != name:
                    continue
                if choice_index is not None and action.get("choice_index") != choice_index:
                    continue
                if target_index is not None and action.get("target_index") != target_index:
                    continue
                return action
            for action in live_actions:
                if action.get("kind") == kind and (choice_index is None or action.get("choice_index") == choice_index):
                    return action
            self.fail(f"could not match recorded action {recorded_action!r} against {live_actions!r}")

        for step in trace["steps"][:16]:
            env.step(_find_action(step["action"]))

        self.assertEqual(env.phase, "EVENT")
        self.assertEqual(env.floor, 2)
        self.assertEqual(env.current_event_id, "The Woman in Blue")

    def test_falling_excludes_bottled_cards_from_removed_options_in_v2(self):
        env = NativeRunEnv(seed=11123, ascension_level=0, start_on_map=True)
        bottled_attack = make_card("Headbutt", uuid="bottled-attack")
        free_attack = make_card("Pommel Strike", uuid="free-attack")
        bottled_skill = make_card("Armaments", uuid="bottled-skill")
        bottled_power = make_card("Inflame", uuid="bottled-power")
        env.deck = [bottled_attack, free_attack, bottled_skill, bottled_power]
        env.relics.extend([
            {"relic_id": "Bottled Flame", "name": "Bottled Flame", "tier": "UNCOMMON", "card_uuid": "bottled-attack"},
            {"relic_id": "Bottled Lightning", "name": "Bottled Lightning", "tier": "UNCOMMON", "card_uuid": "bottled-skill"},
            {"relic_id": "Bottled Tornado", "name": "Bottled Tornado", "tier": "UNCOMMON", "card_uuid": "bottled-power"},
        ])
        env._draw_event_id = lambda: "Falling"
        env.randoms.misc = StsRandom(0)

        env._enter_event()

        self.assertEqual([option["name"] for option in env.event_options], ["Removed Attack"])

        env.step({"kind": "event", "event_id": "Falling", "name": "Removed Attack", "choice_index": 0})

        self.assertNotIn("Pommel Strike", [card.card_id for card in env.deck])
        self.assertIn("Headbutt", [card.card_id for card in env.deck])

    def test_event_curse_obtain_does_not_trigger_darkstone_periapt_in_v2(self):
        env = NativeRunEnv(seed=11122, ascension_level=0)
        env.relics.append({"relic_id": "Darkstone Periapt", "name": "Darkstone Periapt", "tier": "UNCOMMON", "counter": -1})
        env.player.current_hp = 27
        env.player.max_hp = 94

        added = env._add_curse_to_deck("Doubt", uuid="event-doubt")

        self.assertTrue(added)
        self.assertEqual(env.player.current_hp, 27)
        self.assertEqual(env.player.max_hp, 94)
        self.assertIn("Doubt", [card.card_id for card in env.deck])

    def test_jack_of_all_trades_plus_adds_generated_cards_in_lightspeed_order_in_v2(self):
        combat = NativeCombatEnv(seed=1200, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Jack Of All Trades", upgrades=1, uuid="jack-plus")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 0

        class FixedCardRandom:
            def __init__(self):
                self.calls = []

            def random(self, upper):
                self.calls.append(upper)
                return 0 if len(self.calls) == 1 else 1

        combat.card_random_rng = FixedCardRandom()
        first_id = combat._random_card_id(colorless=True)
        second_id = combat._random_card_id(colorless=True)
        combat.card_random_rng = FixedCardRandom()

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.hand], [second_id, first_id])
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Jack Of All Trades"])

    def test_violence_keeps_last_attack_in_draw_due_to_lightspeed_bug_in_v2(self):
        combat = NativeCombatEnv(seed=12001, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Violence", uuid="violence"),
            make_card("Defend_R", uuid="defend"),
        ]
        lone_attack = make_card("Strike_R", uuid="draw-strike")
        combat.draw_pile = [lone_attack]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 0

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Defend_R", "Strike_R"])
        self.assertEqual([card.card_id for card in combat.draw_pile], ["Strike_R"])
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Violence"])
        self.assertIsNot(combat.hand[1], combat.draw_pile[0])
        self.assertEqual(combat.hand[1].uuid, combat.draw_pile[0].uuid)

    def test_violence_end_turn_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1262108514257337231

        result = compare_seed(seed, 0, 120, backend="v2")

        self.assertTrue(result.get("match"), result)

    def test_violence_copy_uuid_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1406653002557143251

        result = compare_seed(seed, 0, 80, backend="v2")

        self.assertTrue(result.get("match"), result)

    def test_add_card_to_deck_disambiguates_duplicate_reward_uuids_in_v2(self):
        env = NativeRunEnv(seed=12005, ascension_level=0, enable_neow=False)
        env.deck = []
        env.phase = "MAP"
        env.floor = 10

        first = env._add_card_to_deck("Cleave", uuid="deck-10-Cleave")
        second = env._add_card_to_deck("Cleave", uuid="deck-10-Cleave")

        self.assertEqual(first.uuid, "deck-10-Cleave")
        self.assertEqual(second.uuid, "deck-10-Cleave-1")

    def test_duplicate_reward_cleave_uuid_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8356352752729052758

        result = compare_seed(seed, 0, 160, backend="v2")

        self.assertTrue(result.get("match"), result)

    def test_duplicate_reward_evolve_uuid_regression_seed_matches_lightspeed_in_v2(self):
        seed = 4612844904161361310

        result = compare_seed(seed, 0, 180, backend="v2")

        self.assertTrue(result.get("match"), result)

    def test_gremlin_horn_shuffle_includes_played_clash_regression_seed_matches_lightspeed_in_v2(self):
        seed = 6847992400009869132

        result = compare_seed(seed, 0, 140, backend="v2")

        self.assertTrue(result.get("match"), result)

    def test_gremlin_horn_shuffle_includes_played_clash_variant_regression_seed_matches_lightspeed_in_v2(self):
        seed = 7984343519339637789

        result = compare_seed(seed, 0, 160, backend="v2")

        self.assertTrue(result.get("match"), result)

    def test_headbutt_gremlin_horn_defers_until_after_select_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8619318509780947714

        result = compare_seed(seed, 0, 140, backend="v2")

        self.assertTrue(result.get("match"), result)

    def test_dual_wield_copies_preserve_selected_card_turn_cost_in_v2(self):
        combat = NativeCombatEnv(seed=12003, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Dual Wield", uuid="dual-wield"),
            make_card("Strike_R", uuid="strike-a"),
            make_card("Strike_R", uuid="strike-b"),
        ]
        combat.hand[1].cost_for_turn = 2
        combat.hand[2].cost_for_turn = 1
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)
        self.assertEqual(combat.card_select_context, "DUAL_WIELD")

        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})

        copied_strikes = [card for card in combat.hand if card.card_id == "Strike_R"]
        self.assertEqual([card.cost_for_turn for card in copied_strikes], [1, 2, 2])
        self.assertEqual(len({card.uuid for card in copied_strikes}), 3)

    def test_dual_wield_single_selectable_copy_preserves_turn_cost_in_v2(self):
        combat = NativeCombatEnv(seed=12004, ascension_level=0, player=PlayerState())
        strike = make_card("Strike_R", uuid="strike")
        strike.cost_for_turn = 0
        combat.hand = [
            make_card("Dual Wield", uuid="dual-wield"),
            make_card("Defend_R", uuid="defend"),
            strike,
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        copied_strikes = [card for card in combat.hand if card.card_id == "Strike_R"]
        self.assertEqual([card.cost_for_turn for card in copied_strikes], [0, 0])
        self.assertEqual(len({card.uuid for card in copied_strikes}), 2)

    def test_dual_wield_ink_bottle_draw_resolves_before_current_card_moves_in_v2(self):
        combat = NativeCombatEnv(seed=12005, ascension_level=0, player=PlayerState(energy=1))
        combat.relics.append({"relic_id": "Ink Bottle", "name": "Ink Bottle", "tier": "UNCOMMON", "counter": 9})
        combat.player.powers["Evolve"] = 1
        combat.hand = [
            make_card("Dual Wield", uuid="dual-wield"),
            make_card("Strike_R", uuid="strike-a"),
            make_card("Anger", uuid="anger"),
        ]
        combat.draw_pile = []
        combat.discard_pile = [
            make_card("Defend_R", uuid="discard-defend"),
            make_card("Dazed", uuid="discard-dazed"),
        ]
        combat.exhaust_pile = []

        combat.play_card(0, 0)
        self.assertEqual(combat.card_select_context, "DUAL_WIELD")

        combat._resolve_card_select({"kind": "card_select", "choice_index": 0})

        self.assertEqual([card.card_id for card in combat.discard_pile], ["Dual Wield"])
        self.assertNotIn("Dual Wield", [card.card_id for card in combat.draw_pile])
        self.assertEqual([card.card_id for card in combat.hand], ["Anger", "Strike_R", "Strike_R", "Defend_R"])

    def test_secret_weapon_is_unplayable_without_attack_in_draw_pile_in_v2(self):
        combat = NativeCombatEnv(seed=12001, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Secret Weapon", uuid="secret-weapon")]
        combat.draw_pile = [make_card("Defend_R", uuid="defend"), make_card("Flex", uuid="flex")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        self.assertFalse(combat.playable(combat.hand[0]))
        self.assertFalse(any(action.get("card_id") == "Secret Weapon" for action in combat.legal_actions()))

    def test_secret_technique_is_unplayable_without_skill_in_draw_pile_in_v2(self):
        combat = NativeCombatEnv(seed=12002, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Secret Technique", uuid="secret-technique")]
        combat.draw_pile = [make_card("Strike_R", uuid="strike"), make_card("Bash", uuid="bash")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        self.assertFalse(combat.playable(combat.hand[0]))
        self.assertFalse(any(action.get("card_id") == "Secret Technique" for action in combat.legal_actions()))

    def test_ghosts_event_uses_council_of_ghosts_action_id_in_v2(self):
        env = NativeRunEnv(seed=1113, ascension_level=0)
        env._draw_event_id = lambda: "Ghosts"
        env._enter_event()

        self.assertEqual(env.event_options[0]["event_id"], "Council of Ghosts")
        self.assertEqual(env.event_options[1]["event_id"], "Council of Ghosts")

    def test_necronomicon_does_not_add_necronomicurse_to_run_deck_in_v2(self):
        env = NativeRunEnv(seed=1114, ascension_level=0)
        original_size = len(env.deck)

        env._obtain_relic({"relic_id": "Necronomicon", "id": "Necronomicon", "name": "Necronomicon", "counter": -1, "price": 0, "tier": "EVENT"})

        self.assertTrue(any(relic.get("relic_id") == "Necronomicon" for relic in env.relics))
        self.assertFalse(any(card.card_id == "Necronomicurse" for card in env.deck))
        self.assertEqual(len(env.deck), original_size)

    def test_cursed_tome_hp_loss_bypasses_torii_in_v2(self):
        env = NativeRunEnv(seed=11141, ascension_level=0)
        env.phase = "EVENT"
        env.current_event_id = "Cursed Tome"
        env.event_state = {"phase": 2}
        env.player.current_hp = 30
        env.relics.append(make_relic("Torii"))

        env.step({"kind": "event", "event_id": "Cursed Tome", "name": "Continue", "choice_index": 3})

        self.assertEqual(env.player.current_hp, 28)
        self.assertEqual(env.event_state.get("phase"), 3)

    def test_armaments_plus_can_upgrade_burn_in_v2(self):
        combat = NativeCombatEnv(seed=1115, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        combat.hand = [make_card("Armaments", upgrades=1, uuid="armaments+"), make_card("Burn", uuid="burn")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        burn = next(card for card in combat.hand if card.card_id == "Burn")
        self.assertEqual(burn.upgrades, 1)

    def test_end_turn_burn_is_blocked_by_metallicize_before_damage_in_v2(self):
        combat = NativeCombatEnv(seed=1116, ascension_level=0, scheduled_encounter=["Hexaghost"], player=PlayerState())
        ghost = combat.monsters[0]
        ghost.move = "HEXAGHOST_INFLAME"
        ghost.intent = "DEFEND_BUFF"
        ghost.move_base_damage = 0
        ghost.move_hits = 0
        combat.player.current_hp = 34
        combat.player.max_hp = 80
        combat.player.powers["Metallicize"] = 3
        combat.hand = [make_card("Burn", uuid="burn")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.end_turn()
        self.assertEqual(combat.player.current_hp, 34)
        self.assertEqual(combat.outcome, "UNDECIDED")

    def test_end_turn_decay_uses_pre_cleanup_block_in_v2(self):
        combat = NativeCombatEnv(seed=11161, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState(current_hp=15, max_hp=80))
        cultist = combat.monsters[0]
        cultist.move = "CULTIST_INCANTATION"
        cultist.intent = "BUFF"
        cultist.move_base_damage = 0
        cultist.move_hits = 0
        combat.player.block = 12
        combat.player.powers["Rupture"] = 1
        combat.hand = [make_card("Decay", uuid="decay")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.end_turn()

        self.assertEqual(combat.player.current_hp, 15)
        self.assertEqual(combat.player.block, 0)
        self.assertEqual(combat.player.power("Strength"), 0)

    def test_end_turn_decay_happens_before_ethereal_feel_no_pain_block_in_v2(self):
        combat = NativeCombatEnv(seed=11162, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState(current_hp=15, max_hp=80))
        cultist = combat.monsters[0]
        cultist.move = "CULTIST_INCANTATION"
        cultist.intent = "BUFF"
        cultist.move_base_damage = 0
        cultist.move_hits = 0
        combat.player.powers["Feel No Pain"] = 3
        combat.player.powers["Rupture"] = 1
        combat.hand = [
            make_card("Decay", uuid="decay"),
            make_card("Dazed", uuid="dazed-1"),
            make_card("Dazed", uuid="dazed-2"),
            make_card("Dazed", uuid="dazed-3"),
            make_card("Dazed", uuid="dazed-4"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.end_turn()

        self.assertEqual(combat.player.current_hp, 13)
        self.assertEqual(combat.player.block, 0)
        self.assertEqual(combat.player.power("Strength"), 1)

    def test_end_turn_burn_happens_before_ethereal_feel_no_pain_block_in_v2(self):
        combat = NativeCombatEnv(seed=111621, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState(current_hp=15, max_hp=80))
        cultist = combat.monsters[0]
        cultist.move = "CULTIST_INCANTATION"
        cultist.intent = "BUFF"
        cultist.move_base_damage = 0
        cultist.move_hits = 0
        combat.player.powers["Feel No Pain"] = 3
        combat.hand = [
            make_card("Burn", uuid="burn"),
            make_card("Dazed", uuid="dazed-1"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.end_turn()

        self.assertEqual(combat.player.current_hp, 13)
        self.assertEqual(combat.player.block, 0)

    def test_end_turn_charons_ashes_spore_cloud_applies_before_monster_attack_in_v2(self):
        combat = NativeCombatEnv(
            seed=11163,
            ascension_level=0,
            scheduled_encounter=["FungiBeast", "FungiBeast"],
            player=PlayerState(current_hp=24, max_hp=80),
        )
        combat.relics.append({"relic_id": "Charon's Ashes", "id": "Charon's Ashes", "name": "Charon's Ashes", "counter": -1, "price": 0, "tier": "RARE"})
        first, second = combat.monsters
        first.current_hp = 1
        first.max_hp = 24
        first.move = "FUNGI_BEAST_BITE"
        first.intent = "ATTACK"
        first.move_base_damage = 6
        first.move_hits = 1
        second.current_hp = 25
        second.max_hp = 25
        second.move = "FUNGI_BEAST_BITE"
        second.intent = "ATTACK"
        second.move_base_damage = 6
        second.move_hits = 1
        combat.player.block = 5
        combat.hand = [make_card("Ghostly Armor", uuid="ghostly")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.end_turn()

        self.assertEqual(combat.player.current_hp, 20)
        self.assertEqual(combat.player.power("Vulnerable"), 1)


    def test_end_turn_orichalcum_stacks_with_ethereal_feel_no_pain_block_in_v2(self):
        combat = NativeCombatEnv(seed=1117, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        cultist = combat.monsters[0]
        cultist.move = "CULTIST_INCANTATION"
        cultist.intent = "BUFF"
        cultist.move_base_damage = 0
        cultist.move_hits = 0
        combat.player.powers["Barricade"] = 1
        combat.player.powers["Feel No Pain"] = 3
        combat.relics.append(
            {
                "relic_id": "Orichalcum",
                "id": "Orichalcum",
                "name": "Orichalcum",
                "counter": -1,
                "price": 0,
                "tier": "COMMON",
            }
        )
        combat.hand = [
            make_card("Dazed", uuid="dazed-left"),
            make_card("Strike_R", uuid="strike"),
            make_card("Dazed", uuid="dazed-right"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.end_turn()

        self.assertEqual(combat.player.block, 12)
        self.assertEqual([card.card_id for card in combat.hand], ["Strike_R"])
        self.assertEqual([card.card_id for card in combat.discard_pile], [])
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Dazed", "Dazed"])

    def test_end_turn_orichalcum_triggers_juggernaut_in_v2(self):
        combat = NativeCombatEnv(seed=11171, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        cultist = combat.monsters[0]
        initial_hp = cultist.current_hp
        cultist.move = "CULTIST_INCANTATION"
        cultist.intent = "BUFF"
        cultist.move_base_damage = 0
        cultist.move_hits = 0
        combat.player.powers["Juggernaut"] = 5
        combat.relics.append(
            {
                "relic_id": "Orichalcum",
                "id": "Orichalcum",
                "name": "Orichalcum",
                "counter": -1,
                "price": 0,
                "tier": "COMMON",
            }
        )
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.end_turn()

        self.assertEqual(cultist.current_hp, initial_hp - 5)
        self.assertEqual(combat.player.block, 0)

    def test_end_turn_burn_death_sets_player_loss_before_monster_turn_in_v2(self):
        combat = NativeCombatEnv(seed=1118, ascension_level=0, scheduled_encounter=["Hexaghost"], player=PlayerState())
        ghost = combat.monsters[0]
        ghost.move = "HEXAGHOST_INFLAME"
        ghost.intent = "DEFEND_BUFF"
        ghost.move_base_damage = 0
        ghost.move_hits = 0
        combat.player.current_hp = 2
        combat.player.max_hp = 80
        combat.hand = [make_card("Burn", uuid="burn")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.end_turn()
        self.assertEqual(combat.player.current_hp, 0)
        self.assertEqual(combat.outcome, "PLAYER_LOSS")

    def test_hexaghost_sear_generates_upgraded_burn_after_turn_eight_in_v2(self):
        combat = NativeCombatEnv(seed=1118, ascension_level=0, scheduled_encounter=["Hexaghost"], player=PlayerState())
        ghost = combat.monsters[0]
        ghost.move = "HEXAGHOST_SEAR"
        ghost.intent = "ATTACK"
        ghost.move_base_damage = 6
        ghost.move_hits = 1
        combat.turn = 9
        combat.player.current_hp = 80
        combat.player.max_hp = 80
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat._monster_take_turn(ghost, 0)
        self.assertTrue(any(card.card_id == "Burn" and card.upgrades == 1 for card in combat.discard_pile))

    def test_battle_trance_consumes_artifact_instead_of_applying_no_draw_in_v2(self):
        combat = NativeCombatEnv(seed=99, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Battle Trance", uuid="battle-trance")]
        combat.draw_pile = [
            make_card("Strike_R", uuid="strike-1"),
            make_card("Strike_R", uuid="strike-2"),
            make_card("Strike_R", uuid="strike-3"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3
        combat.player.powers["Artifact"] = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.player.power("Artifact"), 0)
        self.assertEqual(combat.player.power("No Draw"), 0)

    def test_battle_trance_blocks_evolve_followup_draws_after_no_draw_in_v2(self):
        combat = NativeCombatEnv(seed=99, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Battle Trance", uuid="battle-trance")]
        combat.draw_pile = [
            make_card("Defend_R", uuid="defend"),
            make_card("Dazed", uuid="dazed-2"),
            make_card("Dazed", uuid="dazed-1"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3
        combat.player.powers["Evolve"] = 1

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Dazed", "Dazed", "Defend_R"])
        self.assertEqual(combat.draw_pile, [])
        self.assertEqual(combat.player.power("No Draw"), 1)

    def test_second_wind_exhausts_non_attacks_right_to_left_in_v2(self):
        combat = NativeCombatEnv(seed=200, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Second Wind", uuid="second-wind"),
            make_card("Slimed", uuid="slimed-left"),
            make_card("Injury", uuid="injury"),
            make_card("Strike_R", uuid="strike"),
            make_card("Slimed", uuid="slimed-right"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Slimed", "Injury", "Slimed"])

    def test_second_wind_applies_frail_per_exhausted_card_in_v2(self):
        combat = NativeCombatEnv(seed=2001, ascension_level=0, player=PlayerState())
        combat.player.block = 3
        combat.player.powers["Frail"] = 1
        combat.hand = [
            make_card("Second Wind", uuid="second-wind"),
            make_card("Slimed", uuid="slimed-left"),
            make_card("Strike_R", uuid="strike"),
            make_card("Slimed", uuid="slimed-right"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.player.block, 9)
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Slimed", "Slimed"])

    def test_second_wind_dead_branch_resolves_exhausted_cards_before_played_card_in_v2(self):
        combat = NativeCombatEnv(
            seed=2002,
            ascension_level=0,
            player=PlayerState(),
            relics=[make_relic("Dead Branch")],
        )
        combat.player.powers["Corruption"] = 1
        combat.hand = [
            make_card("Second Wind", uuid="second-wind"),
            make_card("True Grit", uuid="true-grit"),
            make_card("Sentinel", uuid="sentinel"),
            make_card("Warcry", uuid="warcry"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        generated = iter(["Clothesline", "Hemokinesis", "Flex", "True Grit"])
        combat._random_combat_card_id = lambda: next(generated)

        combat.play_card(0, 0)

        self.assertEqual(
            [card.card_id for card in combat.hand],
            ["Clothesline", "Hemokinesis", "Flex", "True Grit"],
        )
        self.assertEqual(
            [card.card_id for card in combat.exhaust_pile],
            ["Warcry", "Sentinel", "True Grit", "Second Wind"],
        )

    def test_second_wind_corruption_dead_branch_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1271883222551082461
        result = compare_seed(seed, 0, 281, backend="v2")
        self.assertTrue(result["match"], result)

    def test_dark_embrace_end_turn_ethereal_draw_happens_after_no_draw_clears_in_v2(self):
        combat = NativeCombatEnv(seed=100, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Ghostly Armor", uuid="ghostly")]
        combat.draw_pile = [make_card("Flame Barrier", uuid="flame-barrier")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.powers["Dark Embrace"] = 1
        combat.player.powers["No Draw"] = 1

        combat.end_turn()

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Ghostly Armor"])
        self.assertEqual([card.card_id for card in combat.hand], ["Flame Barrier"])
        self.assertEqual(combat.player.power("No Draw"), 0)

    def test_dark_embrace_end_turn_draw_keeps_confusion_cost_for_next_turn_in_v2(self):
        combat = NativeCombatEnv(seed=100, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Ghostly Armor", uuid="ghostly")]
        combat.draw_pile = [make_card("Body Slam", uuid="body-slam")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.powers["Dark Embrace"] = 1
        combat.player.powers["Confusion"] = 1

        combat.end_turn()

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Ghostly Armor"])
        self.assertEqual([card.card_id for card in combat.hand], ["Body Slam"])
        self.assertIsNotNone(combat.hand[0].cost_for_turn)

    def test_dark_embrace_end_turn_last_ethereal_draw_is_deferred_past_no_draw_in_v2(self):
        combat = NativeCombatEnv(seed=102, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Strike_R", uuid="strike"),
            make_card("Dazed", uuid="dazed"),
        ]
        combat.draw_pile = [
            make_card("Defend_R", uuid="draw-0"),
            make_card("Strike_R", uuid="draw-1"),
            make_card("Defend_R", uuid="draw-2"),
            make_card("Strike_R", uuid="draw-3"),
            make_card("Defend_R", uuid="draw-4"),
            make_card("Strike_R", uuid="draw-5"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.powers["Dark Embrace"] = 1
        combat.player.powers["No Draw"] = 1

        combat.end_turn()

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Dazed"])
        self.assertEqual(
            [card.card_id for card in combat.hand],
            ["Strike_R", "Defend_R", "Strike_R", "Defend_R", "Strike_R", "Defend_R"],
        )
        self.assertEqual(combat.player.power("No Draw"), 0)

    def test_dark_embrace_status_end_turn_draw_can_shuffle_current_regular_discards_after_no_draw_in_v2(self):
        combat = NativeCombatEnv(seed=103, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Dazed", uuid="dazed-left"),
            make_card("Strike_R", uuid="strike-left"),
            make_card("Clash", uuid="clash"),
            make_card("Strike_R", uuid="strike-right"),
            make_card("Dazed", uuid="dazed-right"),
        ]
        combat.draw_pile = []
        combat.discard_pile = [
            make_card("Armaments", uuid="discard-armaments"),
            make_card("Defend_R", uuid="discard-defend"),
            make_card("Bash", uuid="discard-bash"),
        ]
        combat.exhaust_pile = []
        combat.player.powers["Dark Embrace"] = 1
        combat.player.powers["No Draw"] = 1

        combat.end_turn()

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Dazed", "Dazed"])
        self.assertEqual(
            [card.card_id for card in combat.hand],
            ["Bash", "Defend_R", "Armaments", "Strike_R", "Strike_R", "Clash"],
        )
        self.assertEqual(combat.discard_pile, [])
        self.assertEqual(combat.draw_pile, [])
        self.assertEqual(combat.player.power("No Draw"), 0)

    def test_dark_embrace_dead_branch_status_end_turn_order_waits_until_no_draw_clears_in_v2(self):
        combat = NativeCombatEnv(seed=1031, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Defend_R", uuid="hand-defend"),
            make_card("Dazed", uuid="hand-dazed-left"),
            make_card("Dazed", uuid="hand-dazed-right"),
        ]
        combat.draw_pile = [
            make_card("Shockwave", uuid="draw-shockwave"),
            make_card("Headbutt", uuid="draw-headbutt"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.powers["Dark Embrace"] = 1
        combat.player.powers["No Draw"] = 1
        combat.relics.append({"relic_id": "Dead Branch", "name": "Dead Branch", "tier": "RARE", "counter": -1})
        dead_branch_cards = iter(["Rupture", "Brutality"])
        combat._random_combat_card_id = lambda: next(dead_branch_cards)

        combat.end_turn()

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Dazed", "Dazed"])
        self.assertEqual([card.card_id for card in combat.hand[:4]], ["Rupture", "Headbutt", "Brutality", "Shockwave"])
        self.assertIn("Defend_R", [card.card_id for card in combat.hand])
        self.assertEqual(combat.player.power("No Draw"), 0)

    def test_dark_embrace_end_turn_non_status_ethereal_draw_waits_for_remaining_discards_in_v2(self):
        combat = NativeCombatEnv(seed=101, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Sword Boomerang", uuid="sword-boomerang"),
            make_card("Carnage", uuid="carnage"),
            make_card("Strike_R", uuid="strike"),
        ]
        combat.draw_pile = [make_card("Double Tap", uuid="double-tap")]
        combat.discard_pile = [
            make_card("Defend_R", uuid="discard-defend-0"),
            make_card("Strike_R", uuid="discard-strike-0"),
            make_card("Bash", uuid="discard-bash"),
        ]
        combat.exhaust_pile = []
        combat.player.powers["Dark Embrace"] = 1

        combat.end_turn()

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Carnage"])
        self.assertIn("Sword Boomerang", [card.card_id for card in combat.hand])
        self.assertNotIn("Sword Boomerang", [card.card_id for card in combat.discard_pile])

    def test_dark_embrace_non_status_end_turn_draw_after_no_draw_can_shuffle_regular_discards_in_v2(self):
        combat = NativeCombatEnv(seed=4446613289986002436, ascension_level=0, scheduled_encounter=["Cultist"], player=PlayerState())
        cultist = combat.monsters[0]
        cultist.move = "CULTIST_INCANTATION"
        cultist.intent = "BUFF"
        cultist.move_base_damage = 0
        cultist.move_hits = 0
        combat.hand = [
            make_card("Defend_R", uuid="hand-defend-0"),
            make_card("Defend_R", uuid="hand-defend-1"),
            make_card("Carnage", uuid="hand-carnage"),
            make_card("Defend_R", uuid="hand-defend-2"),
            make_card("Defend_R", uuid="hand-defend-3"),
        ]
        combat.draw_pile = []
        combat.discard_pile = [
            make_card("Strike_R", uuid="discard-strike-0"),
            make_card("Strike_R", uuid="discard-strike-1"),
            make_card("Strike_R", uuid="discard-strike-2"),
            make_card("Bash", uuid="discard-bash"),
            make_card("Battle Trance", uuid="discard-battle-trance"),
            make_card("Body Slam", uuid="discard-body-slam"),
        ]
        combat.exhaust_pile = []
        combat.player.powers["Dark Embrace"] = 1
        combat.player.powers["No Draw"] = 1

        combat.end_turn()

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Carnage"])
        self.assertEqual(len(combat.hand), 6)
        self.assertEqual(len(combat.draw_pile), 4)
        self.assertEqual(combat.discard_pile, [])
        self.assertEqual(combat.player.power("No Draw"), 0)

    def test_confusion_randomized_cost_persists_after_end_turn_reset_in_v2(self):
        combat = NativeCombatEnv(seed=1001, ascension_level=0, player=PlayerState())
        bash = make_card("Bash", uuid="bash")
        combat.hand = [bash]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.powers["Confusion"] = 1

        class _ZeroCostRng:
            def randint(self, start, end):
                return 0

        combat.card_random_rng = _ZeroCostRng()
        combat._roll_confusion_cost(bash)
        combat._clear_temporary_cost_state()

        self.assertEqual(bash.cost_for_turn, None)
        self.assertEqual(bash.cost_for_combat, 0)
        state = combat.to_spirecomm_state()
        self.assertEqual(state["combat_state"]["hand"][0]["cost_for_turn"], 0)

    def test_confusion_immediately_randomizes_current_hand_in_v2(self):
        combat = NativeCombatEnv(seed=1002, ascension_level=0, player=PlayerState())
        bash = make_card("Bash", uuid="bash")
        flex = make_card("Flex", uuid="flex")
        combat.hand = [bash, flex]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        class _FixedConfusionRng:
            def __init__(self):
                self.values = [0, 1]

            def randint(self, start, end):
                return self.values.pop(0)

        combat.card_random_rng = _FixedConfusionRng()
        combat._apply_player_power("Confusion", 1)

        self.assertEqual([(card.card_id, card.cost_for_turn, card.cost_for_combat) for card in combat.hand], [
            ("Bash", 0, 0),
            ("Flex", 1, 1),
        ])

    def test_panic_button_consumes_artifact_instead_of_applying_no_block_in_v2(self):
        combat = NativeCombatEnv(seed=100, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Panic Button", uuid="panic-button")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.player.powers["Artifact"] = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.player.power("Artifact"), 0)
        self.assertEqual(combat.player.power("No Block"), 0)

    def test_panic_button_no_block_ticks_down_at_end_of_current_turn_in_v2(self):
        combat = NativeCombatEnv(seed=102, ascension_level=0, scheduled_encounter=["FungiBeast"], player=PlayerState())
        combat.hand = [make_card("Panic Button", uuid="panic-button")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)
        self.assertEqual(combat.player.power("No Block"), 2)

        combat.end_turn()

        self.assertEqual(combat.player.power("No Block"), 1)

    def test_snecko_eye_confusion_is_blocked_by_artifact_in_v2(self):
        player = PlayerState()
        player.powers["Artifact"] = 1
        combat = NativeCombatEnv(
            seed=101,
            ascension_level=0,
            player=player,
            relics=[
                {"relic_id": "Burning Blood", "id": "Burning Blood", "name": "Burning Blood", "counter": -1, "price": 0, "tier": "STARTER"},
                {"relic_id": "Snecko Eye", "id": "Snecko Eye", "name": "Snecko Eye", "counter": -1, "price": 0, "tier": "BOSS"},
            ],
        )

        self.assertEqual(combat.player.power("Artifact"), 0)
        self.assertEqual(combat.player.power("Confusion"), 0)

    def test_the_nest_event_options_match_lightspeed_in_v2(self):
        env = NativeRunEnv(seed=301, ascension_level=0, start_on_map=True)
        env._draw_event_id = lambda: "The Nest"

        env._enter_event()

        self.assertEqual([option["name"] for option in env.event_options], ["Stole From Cult", "Stay in Line"])

    def test_augmenter_event_option_order_matches_lightspeed_in_v2(self):
        env = NativeRunEnv(seed=302, ascension_level=0, start_on_map=True)
        env._draw_event_id = lambda: "Augmenter"

        env._enter_event()

        self.assertEqual([option["name"] for option in env.event_options], ["JAX", "Transform", "Mutagenic Strength"])

    def test_knowing_skull_tracks_progressive_hp_costs_in_v2(self):
        env = NativeRunEnv(seed=303, ascension_level=0, start_on_map=True)
        env.phase = "EVENT"
        env.current_event_id = "Knowing Skull"
        env.event_state = {"hp_amount_0": 6, "hp_amount_1": 6, "hp_amount_2": 6}
        env.player.current_hp = 40
        env.gold = 99

        env.step({"kind": "event", "event_id": "Knowing Skull", "name": "Riches", "choice_index": 0})

        self.assertEqual(env.phase, "EVENT")
        self.assertEqual(env.gold, 189)
        self.assertEqual(env.player.current_hp, 34)
        self.assertEqual(env.event_state["hp_amount_0"], 7)

    def test_old_beggar_opens_remove_screen_in_v2(self):
        env = NativeRunEnv(seed=304, ascension_level=0, start_on_map=True)
        env.phase = "EVENT"
        env.current_event_id = "Old Beggar"
        env.gold = 120

        env.step({"kind": "event", "event_id": "Old Beggar", "name": "Gave Gold", "choice_index": 0})

        self.assertEqual(env.gold, 45)
        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertEqual(env.card_select_context, "EVENT_REMOVE")

    def test_the_nest_stole_from_cult_grants_gold_in_v2(self):
        env = NativeRunEnv(seed=305, ascension_level=0, start_on_map=True)
        env.phase = "EVENT"
        env.current_event_id = "The Nest"
        env.gold = 100

        env.step({"kind": "event", "event_id": "The Nest", "name": "Stole From Cult", "choice_index": 0})

        self.assertEqual(env.gold, 199)
        self.assertEqual(env.phase, "MAP")

    def test_the_nest_stay_in_line_grants_ritual_dagger_in_v2(self):
        env = NativeRunEnv(seed=306, ascension_level=0, start_on_map=True)
        env.phase = "EVENT"
        env.current_event_id = "The Nest"
        env.player.current_hp = 40

        env.step({"kind": "event", "event_id": "The Nest", "name": "Stay in Line", "choice_index": 1})

        self.assertEqual(env.player.current_hp, 34)
        self.assertIn("Ritual Dagger", [card.card_id for card in env.deck])
        self.assertEqual(env.phase, "MAP")

    def test_augmenter_transform_opens_two_card_transform_in_v2(self):
        env = NativeRunEnv(seed=307, ascension_level=0, start_on_map=True)
        env.phase = "EVENT"
        env.current_event_id = "Augmenter"

        env.step({"kind": "event", "event_id": "Augmenter", "name": "Transform", "choice_index": 1})

        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertEqual(env.card_select_context, "EVENT_TRANSFORM")
        self.assertEqual(env.card_select_count, 2)

    def test_secret_portal_jumps_to_final_campfire_in_v2(self):
        env = NativeRunEnv(seed=308, ascension_level=0, start_on_map=True)
        env.act = 3
        env.floor = 41
        env.phase = "EVENT"
        env.current_event_id = "Secret Portal"

        env.step({"kind": "event", "event_id": "Secret Portal", "name": "Entered Portal", "choice_index": 0})

        self.assertEqual(env.floor, 49)
        self.assertEqual(env.phase, "CAMPFIRE")

    def test_mark_of_pain_inserts_wounds_into_draw_pile_without_full_shuffle_in_v2(self):
        combat = NativeCombatEnv(seed=401, ascension_level=0, scheduled_encounter=["FungiBeast"])
        combat.relics.append({
            "relic_id": "Mark of Pain",
            "id": "Mark of Pain",
            "name": "Mark of Pain",
            "counter": -1,
            "price": 0,
            "tier": "BOSS",
        })
        combat.draw_pile = [
            make_card("Strike_R", uuid="draw-0"),
            make_card("Defend_R", uuid="draw-1"),
            make_card("Bash", uuid="draw-2"),
        ]
        expected_rng = combat.card_random_rng.copy()
        expected = [card.card_id for card in combat.draw_pile]
        for _ in range(2):
            insert_index = 0 if not expected else int(expected_rng.random(len(expected) - 1))
            expected.insert(insert_index, "Wound")

        combat._apply_opening_post_draw_relics()

        self.assertEqual([card.card_id for card in combat.draw_pile], expected)

    def test_empty_cage_opens_two_remove_selections_and_then_transitions_act_in_v2(self):
        env = NativeRunEnv(seed=402, ascension_level=0, start_on_map=True)
        env.phase = "BOSS_RELIC"
        env.floor = 17
        env.act = 1
        env.deck = [
            make_card("Strike_R", uuid="strike-0"),
            make_card("AscendersBane", uuid="bane"),
            make_card("Defend_R", uuid="defend-0"),
        ]
        env.boss_relic_options = [
            {
                "kind": "boss_relic",
                "name": "Empty Cage",
                "relic_id": "Empty Cage",
                "choice_index": 0,
            }
        ]

        env.step({"kind": "boss_relic", "name": "Empty Cage", "relic_id": "Empty Cage", "choice_index": 0})

        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertEqual(env.card_select_context, "EVENT_REMOVE")
        self.assertEqual(env.card_select_count, 2)
        self.assertEqual(env.card_select_completion, "TRANSITION_NEXT_ACT")
        self.assertEqual(env.card_select_options[0]["choice_index"], 0)
        self.assertEqual(
            [option["choice_index"] for option in env.card_select_options],
            [0, 2],
        )

        initial_deck_size = len(env.deck)
        first_index = env.card_select_options[0]["choice_index"]
        env.step({"kind": "card_select", "select_type": "EVENT_REMOVE", "choice_index": first_index})
        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertEqual(len(env.deck), initial_deck_size)
        self.assertNotEqual(env.card_select_options[0]["choice_index"], 0)

        second_index = env.card_select_options[0]["choice_index"]
        env.step({"kind": "card_select", "select_type": "EVENT_REMOVE", "choice_index": second_index})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.act, 2)
        self.assertEqual(len(env.deck), initial_deck_size - 2)

    def test_calling_bell_opens_reward_screen_and_transitions_after_skip_in_v2(self):
        env = NativeRunEnv(seed=403, ascension_level=0, start_on_map=True)
        env.phase = "BOSS_RELIC"
        env.floor = 17
        env.act = 1
        env.boss_relic_options = [
            {
                "kind": "boss_relic",
                "name": "Calling Bell",
                "relic_id": "Calling Bell",
                "choice_index": 0,
            }
        ]

        env.step({"kind": "boss_relic", "name": "Calling Bell", "relic_id": "Calling Bell", "choice_index": 0})

        self.assertEqual(env.phase, "CARD_REWARD")
        self.assertEqual(env.reward_context, "BOSS_RELIC")
        self.assertEqual(len(env.reward_relics), 3)
        self.assertIn("CurseOfTheBell", [card.card_id for card in env.deck])
        self.assertEqual(
            [relic["tier"] for relic in env.reward_relics],
            ["COMMON", "UNCOMMON", "RARE"],
        )
        self.assertNotIn(
            "Whetstone",
            [relic["relic_id"] for relic in env.reward_relics],
        )

        env.step({"kind": "skip", "name": "SKIP", "choice_index": 0})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.act, 2)

    def test_astrolabe_only_targets_transformable_cards_in_v2(self):
        env = NativeRunEnv(seed=404, ascension_level=0, start_on_map=True)
        env.phase = "BOSS_RELIC"
        env.floor = 17
        env.act = 1
        env.deck = [
            make_card("Strike_R", uuid="strike-0"),
            make_card("AscendersBane", uuid="bane"),
            make_card("Defend_R", uuid="defend-0"),
            make_card("CurseOfTheBell", uuid="bell-curse"),
        ]
        env.boss_relic_options = [
            {
                "kind": "boss_relic",
                "name": "Astrolabe",
                "relic_id": "Astrolabe",
                "choice_index": 0,
            }
        ]

        env.step({"kind": "boss_relic", "name": "Astrolabe", "relic_id": "Astrolabe", "choice_index": 0})

        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertEqual(env.card_select_context, "TRANSFORM_UPGRADE")
        self.assertEqual([option["choice_index"] for option in env.card_select_options], [0, 2])

    def test_astrolabe_searing_blow_transform_stays_unupgraded_in_v2(self):
        env = NativeRunEnv(seed=4041, ascension_level=0, start_on_map=True)
        env.phase = "BOSS_RELIC"
        env.floor = 17
        env.act = 1
        env.deck = [
            make_card("Strike_R", uuid="strike-0"),
            make_card("Defend_R", uuid="defend-0"),
            make_card("Bash", uuid="bash-0"),
        ]
        env.boss_relic_options = [
            {
                "kind": "boss_relic",
                "name": "Astrolabe",
                "relic_id": "Astrolabe",
                "choice_index": 0,
            }
        ]
        transformed_ids = iter(["Searing Blow", "Pommel Strike", "Shrug It Off"])
        env._transformed_card_from_rng = lambda rng, exclude_card_id: make_card(next(transformed_ids), uuid=f"transform-{exclude_card_id}")

        env.step({"kind": "boss_relic", "name": "Astrolabe", "relic_id": "Astrolabe", "choice_index": 0})
        env.step({"kind": "card_select", "choice_index": 0, "deck_index": 0})
        env.step({"kind": "card_select", "choice_index": 1, "deck_index": 1})
        env.step({"kind": "card_select", "choice_index": 2, "deck_index": 2})

        searing = next(card for card in env.deck if card.card_id == "Searing Blow")
        pommel = next(card for card in env.deck if card.card_id == "Pommel Strike")
        shrug = next(card for card in env.deck if card.card_id == "Shrug It Off")

        self.assertEqual((searing.upgrades, searing.misc), (0, 0))
        self.assertEqual((pommel.upgrades, pommel.misc), (1, 0))
        self.assertEqual((shrug.upgrades, shrug.misc), (1, 0))

    def test_roll_card_reward_allows_duplicates_when_rarity_pool_is_exhausted_in_v2(self):
        env = NativeRunEnv(seed=4042, ascension_level=0, enable_neow=True)
        env._roll_card_rarity = lambda room=None: "COMMON"
        env._ironclad_card_pool = lambda **kwargs: [CARD_LIBRARY["Pommel Strike"]]

        cards = env._roll_card_reward(count=3)

        self.assertEqual([card.card_id for card in cards], ["Pommel Strike", "Pommel Strike", "Pommel Strike"])

    def test_pandoras_box_transforms_each_basic_card_individually_in_v2(self):
        env = NativeRunEnv(seed=4043, ascension_level=0, start_on_map=True)
        env.phase = "BOSS_RELIC"
        env.floor = 17
        env.act = 1
        env.deck = [
            make_card("Strike_R", uuid="strike-0"),
            make_card("Defend_R", uuid="defend-0"),
            make_card("Bash", uuid="bash-0"),
            make_card("Strike_R", uuid="strike-1"),
        ]
        transformed_ids = iter(["Searing Blow", "Pommel Strike", "Shrug It Off"])
        env._pandora_transformed_card_from_rng = lambda rng, exclude_card_id: make_card(next(transformed_ids), uuid=f"transform-{exclude_card_id}")

        env._obtain_relic({"relic_id": "Pandora's Box", "name": "Pandora's Box", "tier": "BOSS"})

        self.assertEqual([card.card_id for card in env.deck], ["Bash", "Shrug It Off", "Pommel Strike", "Searing Blow"])
        searing = env.deck[-1]
        self.assertEqual((searing.upgrades, searing.misc), (0, 0))

    def test_pandoras_box_transforms_trigger_ceramic_fish_for_each_added_card_in_v2(self):
        env = NativeRunEnv(seed=40431, ascension_level=0, start_on_map=True)
        env.phase = "BOSS_RELIC"
        env.floor = 17
        env.act = 1
        env.gold = 100
        env.relics.append(make_relic("Ceramic Fish"))
        env.deck = [
            make_card("Strike_R", uuid="strike-0"),
            make_card("Defend_R", uuid="defend-0"),
            make_card("Bash", uuid="bash-0"),
            make_card("Strike_R", uuid="strike-1"),
        ]
        transformed_ids = iter(["Searing Blow", "Pommel Strike", "Shrug It Off"])
        env._pandora_transformed_card_from_rng = lambda rng, exclude_card_id: make_card(next(transformed_ids), uuid=f"transform-{exclude_card_id}")

        env._obtain_relic({"relic_id": "Pandora's Box", "name": "Pandora's Box", "tier": "BOSS"})

        self.assertEqual(env.gold, 127)

    def test_pandora_transform_rng_uses_combat_pool_order_like_lightspeed_in_v2(self):
        env = NativeRunEnv(seed=4044, ascension_level=0, start_on_map=True)
        env.floor = 17

        class _FixedRng:
            def __init__(self, values):
                self._values = iter(values)

            def random(self, upper):
                value = next(self._values)
                if value > upper:
                    raise AssertionError(f"fixed RNG value {value} exceeds upper bound {upper}")
                return value

        rng = _FixedRng([14, 9, 16])

        self.assertEqual(env._pandora_transformed_card_from_rng(rng, "Strike_R").card_id, "Body Slam")
        self.assertEqual(env._pandora_transformed_card_from_rng(rng, "Strike_R").card_id, "Pommel Strike")
        self.assertEqual(env._pandora_transformed_card_from_rng(rng, "Defend_R").card_id, "Flex")

    def test_pandora_transform_rng_preserves_lightspeed_tail_order_in_v2(self):
        env = NativeRunEnv(seed=4045, ascension_level=0, start_on_map=True)
        env.floor = 17

        class _FixedRng:
            def __init__(self, values):
                self._values = iter(values)

            def random(self, upper):
                value = next(self._values)
                if value > upper:
                    raise AssertionError(f"fixed RNG value {value} exceeds upper bound {upper}")
                return value

        rng = _FixedRng([69, 70, 71])

        self.assertEqual(env._pandora_transformed_card_from_rng(rng, "Strike_R").card_id, "Exhume")
        self.assertEqual(env._pandora_transformed_card_from_rng(rng, "Strike_R").card_id, "Offering")
        self.assertEqual(env._pandora_transformed_card_from_rng(rng, "Defend_R").card_id, "Immolate")

    def test_old_coin_matches_lightspeed_no_immediate_gold_gain_in_v2(self):
        env = NativeRunEnv(seed=405, ascension_level=0, start_on_map=True)
        env.gold = 60

        env._obtain_relic({"relic_id": "Old Coin", "name": "Old Coin", "tier": "RARE"})

        self.assertEqual(env.gold, 60)

    def test_enchiridion_adds_zero_cost_power_at_combat_start_in_v2(self):
        combat = NativeCombatEnv(seed=406, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Enchiridion", "name": "Enchiridion", "tier": "EVENT", "counter": -1})
        combat.start_combat()

        zero_cost_powers = [
            card for card in combat.hand
            if card.card_def.card_type == "POWER" and card.cost_for_turn == 0
        ]
        self.assertTrue(zero_cost_powers)

    def test_enchiridion_uses_combat_power_pool_order_in_v2(self):
        combat = NativeCombatEnv(seed=4061, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Enchiridion", "name": "Enchiridion", "tier": "EVENT", "counter": -1})
        combat.locked_card_ids.update({"Evolve", "Fire Breathing", "Rupture"})

        class _ZeroRng:
            def random(self, *args):
                return 0

        combat.card_random_rng = _ZeroRng()
        combat.start_combat()

        zero_cost_powers = [
            card for card in combat.hand
            if card.card_def.card_type == "POWER" and card.cost_for_turn == 0
        ]
        self.assertEqual(len(zero_cost_powers), 1)
        self.assertEqual(zero_cost_powers[0].card_id, "Feel No Pain")

    def test_brutality_plus_is_opening_innate_in_v2(self):
        combat = NativeCombatEnv(seed=4062, ascension_level=0, player=PlayerState())
        combat.deck = [
            make_card("Strike_R", uuid="strike-0"),
            make_card("Defend_R", uuid="defend-0"),
            make_card("Brutality", upgrades=1, uuid="brutality-plus"),
            make_card("Bash", uuid="bash-0"),
            make_card("Strike_R", uuid="strike-1"),
            make_card("Defend_R", uuid="defend-1"),
        ]

        combat.start_combat()

        self.assertTrue(
            any(card.card_id == "Brutality" and card.upgrades == 1 for card in combat.hand)
        )

    def test_strike_dummy_applies_to_pommel_strike_in_v2(self):
        combat = NativeCombatEnv(seed=4063, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Strike Dummy", "name": "Strike Dummy", "tier": "UNCOMMON", "counter": -1})
        target = MonsterState("Looter", "Looter", 60, 60, "MUG", "ATTACK")
        combat.monsters = [target]
        combat.hand = [make_card("Pommel Strike", uuid="pommel")]
        combat.draw_pile = [make_card("Defend_R", uuid="draw-0")]
        combat.play_card(0, 0)

        self.assertEqual(target.current_hp, 48)

    def test_strike_dummy_applies_to_swift_strike_in_v2(self):
        combat = NativeCombatEnv(seed=40631, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Strike Dummy", "name": "Strike Dummy", "tier": "UNCOMMON", "counter": -1})
        target = MonsterState("Looter", "Looter", 60, 60, "MUG", "ATTACK")
        combat.monsters = [target]
        combat.hand = [make_card("Swift Strike", uuid="swift")]
        combat.draw_pile = []
        combat.play_card(0, 0)

        self.assertEqual(target.current_hp, 50)

    def test_swift_strike_strike_dummy_byrd_regression_seed_matches_lightspeed_in_v2(self):
        result = compare_seed(411871189319029492, 0, 278, backend="v2")

        self.assertTrue(result["match"], result)

    def test_panache_triggers_on_double_tap_replay_in_v2(self):
        combat = NativeCombatEnv(seed=40632, ascension_level=0, player=PlayerState())
        combat.player.add_power("Panache", 10)
        combat.panache_counter = 2
        combat.player.add_power("Double Tap", 1)
        target = MonsterState("Cultist", "Cultist", 80, 80, "INCANTATION", "BUFF")
        combat.monsters = [target]
        combat.hand = [make_card("Clash", uuid="clash")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.play_card(0, 0)

        self.assertEqual(combat.monsters[0].current_hp, 42)
        self.assertEqual(combat.panache_counter, 0)

    def test_double_tap_clash_panache_regression_seed_matches_lightspeed_in_v2(self):
        result = compare_seed(630499485759612886, 0, 225, backend="v2")

        self.assertTrue(result["match"], result)

    def test_mummified_hand_reduces_random_remaining_hand_card_in_v2(self):
        combat = NativeCombatEnv(seed=40615, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Mummified Hand", "name": "Mummified Hand", "tier": "RARE", "counter": -1})
        combat.hand = [
            make_card("Dark Embrace", uuid="power"),
            make_card("Strike_R", uuid="strike"),
            make_card("Defend_R", uuid="defend"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3

        class _PickLastRng:
            def random(self, upper):
                return upper

        combat.card_random_rng = _PickLastRng()
        combat.play_card(0, 0)

        strike = next(card for card in combat.hand if card.card_id == "Strike_R")
        defend = next(card for card in combat.hand if card.card_id == "Defend_R")
        self.assertIsNone(strike.cost_for_turn)
        self.assertEqual(defend.cost_for_turn, 0)

    def test_blue_candle_curse_cost_marker_matches_lightspeed_in_v2(self):
        combat = NativeCombatEnv(seed=40616, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Blue Candle", "name": "Blue Candle", "tier": "UNCOMMON", "counter": -1})
        combat.hand = [make_card("Doubt", uuid="doubt")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3

        state = combat.to_spirecomm_state()
        self.assertEqual(state["combat_state"]["hand"][0]["cost_for_turn"], -3)
        self.assertEqual(
            [action["card_id"] for action in combat.legal_actions() if action.get("kind") == "card"],
            ["Doubt"],
        )

    def test_blue_candle_self_damage_triggers_rupture_in_v2(self):
        combat = NativeCombatEnv(seed=40617, ascension_level=0, player=PlayerState(current_hp=40, max_hp=80, energy=1))
        combat.relics.append({"relic_id": "Blue Candle", "name": "Blue Candle", "tier": "UNCOMMON", "counter": -1})
        combat.player.powers["Rupture"] = 1
        combat.hand = [make_card("Injury", uuid="injury")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.monsters = [make_monster("Cultist", StsRandom(40617), ascension=0)]

        combat.play_card(0, 0)

        self.assertEqual(combat.player.current_hp, 39)
        self.assertEqual(combat.player.power("Strength"), 1)
        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Injury"])

    def test_blue_candle_curse_triggers_hex_in_v2(self):
        combat = NativeCombatEnv(seed=40618, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Blue Candle", "name": "Blue Candle", "tier": "UNCOMMON", "counter": -1})
        combat.player.powers["Hex"] = 1
        combat.hand = [make_card("CurseOfTheBell", uuid="bell")]
        combat.draw_pile = [make_card("Inflame", uuid="inflame")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.monsters = [make_monster("Chosen", StsRandom(40618), ascension=0)]

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["CurseOfTheBell"])
        self.assertEqual(sorted(card.card_id for card in combat.draw_pile), ["Dazed", "Inflame"])

    def test_forethought_marks_card_free_to_play_once_in_v2(self):
        combat = NativeCombatEnv(seed=40617, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Forethought", uuid="forethought"),
            make_card("Strike_R", uuid="strike"),
            make_card("Defend_R", uuid="defend"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)
        combat._resolve_card_select({"kind": "card_select", "select_index": 0})

        self.assertEqual([card.card_id for card in combat.draw_pile], ["Strike_R"])
        self.assertTrue(combat.draw_pile[0].free_to_play_once)
        self.assertIsNone(combat.draw_pile[0].cost_for_turn)

        combat.hand = [combat.draw_pile.pop(0)]
        combat.player.energy = 0
        self.assertTrue(combat.playable(combat.hand[0]))

    def test_forethought_marks_only_selected_duplicate_instance_free_to_play_once_in_v2(self):
        combat = NativeCombatEnv(seed=406170, ascension_level=0, player=PlayerState())
        combat.hand = [
            make_card("Forethought", uuid="forethought"),
            make_card("Strike_R", uuid="strike-a"),
            make_card("Strike_R", uuid="strike-b"),
        ]
        combat.draw_pile = [make_card("Defend_R", uuid="draw-defend")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)
        combat._resolve_card_select({"kind": "card_select", "select_index": 1})

        self.assertEqual(combat.draw_pile[0].uuid, "strike-b")
        self.assertTrue(combat.draw_pile[0].free_to_play_once)
        self.assertFalse(combat.hand[0].free_to_play_once)
        self.assertEqual(combat.hand[0].uuid, "strike-a")

    def test_forethought_free_to_play_once_is_consumed_after_play_in_v2(self):
        combat = NativeCombatEnv(seed=406171, ascension_level=0, player=PlayerState())
        target = make_monster("JawWorm", StsRandom(406172), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.hand = [
            make_card("Forethought", uuid="forethought"),
            make_card("Strike_R", uuid="strike"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)
        combat._resolve_card_select({"kind": "card_select", "select_index": 0})

        combat.hand = [combat.draw_pile.pop(0)]
        combat.player.energy = 0
        self.assertTrue(combat.playable(combat.hand[0]))

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.discard_pile], ["Forethought", "Strike_R"])
        self.assertFalse(combat.discard_pile[-1].free_to_play_once)

    def test_headbutt_consumes_free_to_play_once_after_card_select_resolution_in_v2(self):
        combat = NativeCombatEnv(seed=4061711, ascension_level=0, player=PlayerState())
        target = make_monster("JawWorm", StsRandom(4061712), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        headbutt = make_card("Headbutt", uuid="headbutt")
        headbutt.free_to_play_once = True
        combat.hand = [headbutt]
        combat.draw_pile = []
        combat.discard_pile = [
            make_card("Strike_R", uuid="discard-strike"),
            make_card("Defend_R", uuid="discard-defend"),
        ]
        combat.exhaust_pile = []
        combat.player.energy = 0

        combat.play_card(0, 0)
        self.assertEqual(combat.card_select_context, "HEADBUTT")

        combat._resolve_card_select({"kind": "card_select", "select_index": 0})

        self.assertEqual([card.card_id for card in combat.draw_pile], ["Strike_R"])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Defend_R", "Headbutt"])
        self.assertFalse(combat.discard_pile[-1].free_to_play_once)

    def test_headbutt_defers_nunchaku_energy_until_card_select_resolves_in_v2(self):
        combat = NativeCombatEnv(seed=4061712, ascension_level=0, player=PlayerState())
        target = make_monster("JawWorm", StsRandom(4061713), ascension=0)
        target.current_hp = 40
        target.max_hp = 40
        combat.monsters = [target]
        combat.relics.append(make_relic("Nunchaku"))
        combat._relic("Nunchaku")["counter"] = 9
        combat.hand = [make_card("Headbutt", uuid="headbutt")]
        combat.draw_pile = []
        combat.discard_pile = [
            make_card("Strike_R", uuid="discard-strike"),
            make_card("Defend_R", uuid="discard-defend"),
        ]
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(combat.card_select_context, "HEADBUTT")
        self.assertEqual(combat.player.energy, 0)

        combat._resolve_card_select({"kind": "card_select", "select_index": 0})

        self.assertEqual(combat.player.energy, 1)
        self.assertEqual([card.card_id for card in combat.draw_pile], ["Strike_R"])
        self.assertEqual([card.card_id for card in combat.discard_pile], ["Defend_R", "Headbutt"])

    def test_apotheosis_does_not_upgrade_itself_in_v2(self):
        combat = NativeCombatEnv(seed=406172, ascension_level=0, player=PlayerState())
        combat.hand = [make_card("Apotheosis", uuid="apotheosis")]
        combat.draw_pile = [make_card("Strike_R", uuid="strike")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Apotheosis"])
        self.assertEqual(combat.exhaust_pile[0].upgrades, 0)
        self.assertEqual(combat.exhaust_pile[0].cost, 2)
        self.assertEqual(combat.draw_pile[0].upgrades, 1)

    def test_apotheosis_only_upgrades_shared_searing_blow_once_in_v2(self):
        combat = NativeCombatEnv(seed=406172, ascension_level=0, player=PlayerState())
        apotheosis = make_card("Apotheosis", uuid="apotheosis")
        searing_blow = make_card("Searing Blow", uuid="searing-blow")
        strike = make_card("Strike_R", uuid="strike")
        combat.deck = [apotheosis, searing_blow, strike]
        combat.hand = [apotheosis, searing_blow]
        combat.draw_pile = [strike]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 2

        combat.play_card(0, 0)

        self.assertEqual((searing_blow.upgrades, searing_blow.misc), (1, 1))
        self.assertEqual((combat.hand[0].upgrades, combat.hand[0].misc), (1, 1))
        self.assertEqual((combat.deck[1].upgrades, combat.deck[1].misc), (1, 1))

    def test_enlightenment_base_only_reduces_cost_for_turn_in_v2(self):
        combat = NativeCombatEnv(seed=406173, ascension_level=0, player=PlayerState())
        bash = make_card("Bash", uuid="bash")
        combat.hand = [make_card("Enlightenment", uuid="enlightenment"), bash]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(bash.cost_for_turn, 1)
        self.assertIsNone(bash.cost_for_combat)

        combat.hand = [bash]
        combat._move_card_to_discard(combat.hand.pop(0))

        bash_in_discard = next(card for card in combat.discard_pile if card.card_id == "Bash")
        self.assertEqual(bash_in_discard.cost_for_turn, 1)
        self.assertIsNone(bash_in_discard.cost_for_combat)
        self.assertEqual(bash_in_discard.cost, 1)

    def test_enlightenment_plus_reduces_cost_for_combat_in_v2(self):
        combat = NativeCombatEnv(seed=406175, ascension_level=0, player=PlayerState())
        bash = make_card("Bash", uuid="bash")
        combat.hand = [make_card("Enlightenment", upgrades=1, uuid="enlightenment+"), bash]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1

        combat.play_card(0, 0)

        self.assertEqual(bash.cost_for_turn, 1)
        self.assertEqual(bash.cost_for_combat, 1)

    def test_temporary_cost_state_clears_at_end_of_turn_in_v2(self):
        combat = NativeCombatEnv(seed=4061751, ascension_level=0, player=PlayerState())
        defend = make_card("Defend_R", uuid="defend")
        defend.cost_for_turn = 0
        combat.hand = [defend]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.monsters = []

        combat.end_turn()

        self.assertEqual(len(combat.discard_pile), 1)
        self.assertIsNone(combat.discard_pile[0].cost_for_turn)
        self.assertFalse(combat.discard_pile[0].free_to_play_once)

    def test_free_to_play_once_persists_across_end_of_turn_in_v2(self):
        combat = NativeCombatEnv(seed=4061752, ascension_level=0, player=PlayerState())
        strike = make_card("Strike_R", uuid="strike")
        strike.free_to_play_once = True
        combat.hand = [strike]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.monsters = []

        combat.end_turn()

        self.assertEqual(len(combat.discard_pile), 1)
        self.assertTrue(combat.discard_pile[0].free_to_play_once)

    def test_dead_jaw_worm_still_rolls_next_move_after_flame_barrier_in_v2(self):
        env = NativeRunEnv(seed=4061753, ascension_level=0)
        env.phase = "COMBAT"
        env.current_node_symbol = "M"
        env.combat = NativeCombatEnv(
            seed=4061753,
            ascension_level=0,
            scheduled_encounter=["JawWorm", "SpikeSlime_M"],
            player=PlayerState(current_hp=80, max_hp=80),
        )
        combat = env.combat
        jaw, slime = combat.monsters
        jaw.current_hp = 1
        jaw.block = 6
        jaw.powers = {"Strength": 6, "Vulnerable": 1, "Weakened": 2}
        jaw.move = "JAW_WORM_CHOMP"
        jaw.intent = "ATTACK"
        jaw.move_base_damage = 11
        jaw.move_hits = 1
        jaw.move_history = ["JAW_WORM_CHOMP", "JAW_WORM_BELLOW"]
        slime.move = "SPIKE_SLIME_M_LICK"
        slime.intent = "DEBUFF"
        slime.move_base_damage = 0
        slime.move_hits = 0
        slime.move_history = ["SPIKE_SLIME_M_LICK", "SPIKE_SLIME_M_LICK"]
        combat.player.powers["Flame Barrier"] = 4

        class _JawWormRollRng:
            def random(self, upper):
                return 10

            def random_boolean(self, chance=None):
                return False

        combat.ai_rng = _JawWormRollRng()
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        env.step({"kind": "end", "name": "END_TURN"})

        self.assertEqual(jaw.current_hp, 0)
        self.assertEqual(jaw.move, "JAW_WORM_THRASH")
        self.assertEqual(jaw.intent, "ATTACK_DEFEND")
        self.assertEqual(jaw.move_base_damage, 7)

    def test_start_combat_clears_cost_for_combat_and_free_to_play_once_in_v2(self):
        bash = make_card("Bash", uuid="bash")
        bash.cost_for_combat = 1
        bash.free_to_play_once = True
        strike = make_card("Strike_R", uuid="strike")
        strike.cost_for_combat = 0
        strike.free_to_play_once = True

        combat = NativeCombatEnv(seed=406174, ascension_level=0, player=PlayerState())
        combat.deck = [bash, strike]
        combat.start_combat()

        refreshed = {card.uuid: card for card in combat.draw_pile + combat.hand + combat.discard_pile}
        self.assertIsNone(refreshed["bash"].cost_for_combat)
        self.assertFalse(refreshed["bash"].free_to_play_once)
        self.assertIsNone(refreshed["strike"].cost_for_combat)
        self.assertFalse(refreshed["strike"].free_to_play_once)

    def test_duplicator_opens_card_select_and_duplicates_selected_card_in_v2(self):
        env = NativeRunEnv(seed=406176, ascension_level=0, enable_neow=True)
        env.phase = "EVENT"
        env.current_event_id = "Duplicator"
        env.deck = [
            make_card("Strike_R", uuid="strike"),
            make_card("Bash", uuid="bash"),
        ]
        env.event_options = [
            {"kind": "event", "event_id": "Duplicator", "name": "Duplicated", "choice_index": 0},
            {"kind": "event", "event_id": "Duplicator", "name": "Ignored", "choice_index": 1},
        ]

        state = env.step({"kind": "event", "event_id": "Duplicator", "name": "Duplicated", "choice_index": 0})

        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertEqual(env.card_select_context, "DUPLICATE")
        self.assertEqual(len(env.card_select_options), 2)
        self.assertEqual(state["screen"], "CARD_SELECT")

        env.step({"kind": "card_select", "select_type": "DUPLICATE", "target_index": 1, "choice_index": 1})

        self.assertEqual([card.card_id for card in env.deck], ["Strike_R", "Bash", "Bash"])

    def test_designer_in_spire_uses_lightspeed_choice_indexes_in_v2(self):
        env = NativeRunEnv(seed=4061761, ascension_level=0, enable_neow=True)
        env.gold = 200
        env.deck = [
            make_card("Strike_R", uuid="strike"),
            make_card("Bash", uuid="bash"),
            make_card("Defend_R", uuid="defend"),
        ]
        bools = iter([False, False])
        env.randoms.misc.random_boolean = lambda chance=None: next(bools)
        env._draw_event_id = lambda: "Designer In-Spire"

        env._enter_event()

        self.assertEqual(env.phase, "EVENT")
        self.assertEqual(
            [option["choice_index"] for option in env.event_options],
            [1, 3, 4, 5],
        )

    def test_designer_in_spire_adjusted_random_upgrade_path_in_v2(self):
        env = NativeRunEnv(seed=4061762, ascension_level=0, enable_neow=True)
        env.phase = "EVENT"
        env.current_event_id = "Designer In-Spire"
        env.event_state = {
            "designer_upgrade_one": False,
            "designer_cleanup_is_remove": False,
        }
        env.gold = 200
        env.deck = [
            make_card("Strike_R", uuid="strike"),
            make_card("Bash", uuid="bash"),
            make_card("Defend_R", uuid="defend"),
        ]
        env.randoms.misc.random_long = lambda: 0

        env.step({"kind": "event", "event_id": "Designer In-Spire", "name": "Adjusted", "choice_index": 1})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.gold, 200)
        self.assertEqual(sum(card.upgrades for card in env.deck), 2)

    def test_designer_in_spire_full_service_removes_then_upgrades_in_v2(self):
        env = NativeRunEnv(seed=4061763, ascension_level=0, enable_neow=True)
        env.phase = "EVENT"
        env.current_event_id = "Designer In-Spire"
        env.event_state = {
            "designer_upgrade_one": True,
            "designer_cleanup_is_remove": True,
        }
        env.gold = 200
        env.deck = [
            make_card("Strike_R", uuid="strike"),
            make_card("Bash", uuid="bash"),
        ]
        env.randoms.misc.random_long = lambda: 0

        env.step({"kind": "event", "event_id": "Designer In-Spire", "name": "Full Service", "choice_index": 4})

        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertEqual(env.card_select_context, "EVENT_REMOVE")
        self.assertEqual(env.card_select_completion, "DESIGNER_FULL_SERVICE")

        env.step({"kind": "card_select", "select_type": "EVENT_REMOVE", "choice_index": 0})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.gold, 110)
        self.assertEqual(len(env.deck), 1)
        self.assertEqual(env.deck[0].card_id, "Bash")
        self.assertEqual(env.deck[0].upgrades, 1)

    def test_mummified_hand_ignores_free_to_play_once_cards_in_v2(self):
        combat = NativeCombatEnv(seed=40618, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Mummified Hand", "name": "Mummified Hand", "tier": "RARE", "counter": -1})
        free_strike = make_card("Strike_R", uuid="strike")
        free_strike.free_to_play_once = True
        defend = make_card("Defend_R", uuid="defend")
        combat.hand = [
            make_card("Dark Embrace", uuid="power"),
            free_strike,
            defend,
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3

        class _ZeroRng:
            def random(self, upper):
                return upper

        combat.card_random_rng = _ZeroRng()
        combat.play_card(0, 0)

        self.assertTrue(free_strike.free_to_play_once)
        self.assertIsNone(free_strike.cost_for_turn)
        self.assertEqual(defend.cost_for_turn, 0)

    def test_mummified_hand_preserves_pre_removal_power_slot_order_in_v2(self):
        combat = NativeCombatEnv(seed=40619, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Mummified Hand", "name": "Mummified Hand", "tier": "RARE", "counter": -1})
        strike = make_card("Strike_R", uuid="strike")
        power = make_card("Dark Embrace", uuid="power")
        defend = make_card("Defend_R", uuid="defend")
        combat.hand = [strike, power, defend]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3

        class _PickMiddleRng:
            def random(self, upper):
                return 1

        combat.card_random_rng = _PickMiddleRng()
        combat.play_card(1, 0)

        self.assertIsNone(strike.cost_for_turn)
        self.assertIsNone(defend.cost_for_turn)

    def test_mummified_hand_can_discount_remaining_duplicate_power_in_v2(self):
        combat = NativeCombatEnv(seed=40620, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Mummified Hand", "name": "Mummified Hand", "tier": "RARE", "counter": -1})
        metallicize = make_card("Metallicize", uuid="metal")
        remaining_power = make_card("Dark Embrace", uuid="remaining")
        played_power = make_card("Dark Embrace", uuid="played")
        strike = make_card("Strike_R", uuid="strike")
        defend = make_card("Defend_R", uuid="defend")
        combat.hand = [metallicize, remaining_power, strike, played_power, defend]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 3

        class _PickSecondCandidateRng:
            def random(self, upper):
                return 1

        combat.card_random_rng = _PickSecondCandidateRng()
        combat.play_card(3, 0)

        self.assertEqual(remaining_power.cost_for_turn, 0)
        self.assertIsNone(metallicize.cost_for_turn)
        self.assertIsNone(strike.cost_for_turn)
        self.assertIsNone(defend.cost_for_turn)

    def test_forethought_dark_embrace_chain_does_not_create_extra_zero_cost_strike_in_v2(self):
        combat = NativeCombatEnv(seed=406190, ascension_level=0, player=PlayerState(energy=3))
        combat.relics.append({"relic_id": "Mummified Hand", "name": "Mummified Hand", "tier": "RARE", "counter": -1})
        combat.monsters = [make_monster("JawWorm", StsRandom(406191), ascension=0)]
        free_strike = make_card("Strike_R", uuid="forethought-free")
        free_strike.free_to_play_once = True
        combat.hand = [
            make_card("Clash", uuid="clash"),
            make_card("Strike_R", uuid="normal-strike"),
            make_card("Defend_R", uuid="defend"),
            make_card("Dark Embrace", uuid="dark-embrace"),
            free_strike,
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        class _PickMiddleOfThreeRng:
            def random(self, upper):
                return 1

        combat.card_random_rng = _PickMiddleOfThreeRng()
        combat.play_card(3, 0)

        state = combat.to_spirecomm_state()
        playable_cards = [
            (card["card_id"], card["cost_for_turn"])
            for card in state["combat_state"]["hand"]
            if card.get("is_playable")
        ]
        self.assertEqual(
            sorted(playable_cards),
            [("Defend_R", 0), ("Strike_R", 1), ("Strike_R", 1)],
        )

    def test_corruption_apotheosis_keeps_upgraded_skill_cost_visible_in_v2(self):
        combat = NativeCombatEnv(seed=406192, ascension_level=0, player=PlayerState(energy=5))
        combat.monsters = [make_monster("JawWorm", StsRandom(406193), ascension=0)]
        combat.hand = [
            make_card("Corruption", uuid="corruption"),
            make_card("Apotheosis", uuid="apotheosis"),
            make_card("Entrench", uuid="entrench"),
        ]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)

        self.assertEqual(combat.hand[0].card_id, "Apotheosis")
        self.assertEqual(combat.hand[0].cost_for_turn, 0)
        self.assertEqual(combat.hand[0].cost_for_combat, 0)
        self.assertEqual(combat.hand[1].card_id, "Entrench")
        self.assertEqual(combat.hand[1].cost_for_turn, 0)
        self.assertEqual(combat.hand[1].cost_for_combat, 0)

        combat.play_card(0, 0)

        self.assertEqual(len(combat.hand), 1)
        self.assertEqual(combat.hand[0].card_id, "Entrench")
        self.assertEqual(combat.hand[0].upgrades, 1)
        self.assertEqual(combat.hand[0].cost_for_turn, 1)
        self.assertEqual(combat.hand[0].cost_for_combat, 0)
        self.assertEqual(combat._card_energy_cost(combat.hand[0]), 0)
        state = combat.to_spirecomm_state()
        self.assertEqual(state["combat_state"]["hand"][0]["cost_for_turn"], 1)

    def test_corruption_draws_apotheosis_upgraded_skill_as_zero_cost_in_v2(self):
        combat = NativeCombatEnv(seed=406193, ascension_level=0, player=PlayerState(energy=5))
        combat.monsters = [make_monster("JawWorm", StsRandom(406194), ascension=0)]
        combat.hand = [
            make_card("Corruption", uuid="corruption"),
            make_card("Apotheosis", uuid="apotheosis"),
        ]
        combat.draw_pile = [make_card("Entrench", uuid="entrench")]
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.play_card(0, 0)
        combat.play_card(0, 0)

        self.assertEqual(combat.draw_pile[0].card_id, "Entrench")
        self.assertEqual(combat.draw_pile[0].upgrades, 1)
        self.assertEqual(combat.draw_pile[0].cost_for_turn, 1)
        self.assertEqual(combat.draw_pile[0].cost_for_combat, 0)

        combat.end_turn()

        self.assertEqual(len(combat.hand), 1)
        self.assertEqual(combat.hand[0].card_id, "Entrench")
        self.assertEqual(combat.hand[0].upgrades, 1)
        self.assertEqual(combat.hand[0].cost_for_combat, 0)
        state = combat.to_spirecomm_state()
        self.assertEqual(state["combat_state"]["hand"][0]["cost_for_turn"], 0)
        entrench_action = next(action for action in combat.legal_actions() if action.get("card_id") == "Entrench")
        self.assertEqual(_choice_list_signature([entrench_action], state), [("card", "entrench", 0, 0)])

    def test_corruption_updates_exhausted_skills_in_v2(self):
        combat = NativeCombatEnv(seed=406194, ascension_level=0, player=PlayerState(energy=3))
        exhausted_skill = make_card("Apparition", uuid="corruption-exhausted-apparition")
        combat.hand = [make_card("Corruption", uuid="corruption")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = [exhausted_skill]

        combat.play_card(0, 0)

        self.assertEqual(combat.exhaust_pile[0].card_id, "Apparition")
        self.assertEqual(combat.exhaust_pile[0].cost_for_combat, 0)
        self.assertEqual(combat.exhaust_pile[0].cost_for_turn, 0)

    def test_mummified_hand_triggers_from_havoc_played_power_on_remaining_hand_in_v2(self):
        combat = NativeCombatEnv(seed=406191, ascension_level=0, player=PlayerState())
        combat.relics.append({"relic_id": "Mummified Hand", "name": "Mummified Hand", "tier": "RARE", "counter": -1})
        strike = make_card("Strike_R", uuid="strike")
        combat.hand = [make_card("Havoc", uuid="havoc"), strike]
        combat.draw_pile = [make_card("Evolve", uuid="evolve")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.player.energy = 1
        combat.monsters = [make_monster("JawWorm", StsRandom(406192), ascension=0)]

        class _ZeroRng:
            def random(self, upper):
                return 0

        combat.card_random_rng = _ZeroRng()
        combat.play_card(0, 0)

        self.assertEqual([card.card_id for card in combat.hand], ["Strike_R"])
        self.assertEqual(combat.hand[0].cost_for_turn, 0)

    def test_forgotten_altar_without_golden_idol_uses_shed_blood_at_choice_one_in_v2(self):
        env = NativeRunEnv(seed=4062, ascension_level=0, start_on_map=True)
        env._draw_event_id = lambda: "Forgotten Altar"
        env._enter_event()

        self.assertEqual(
            [(item["choice_index"], item["name"]) for item in env.event_options],
            [(1, "Shed Blood"), (2, "Ignored")],
        )

    def test_forgotten_altar_with_golden_idol_uses_smashed_altar_at_choice_zero_in_v2(self):
        env = NativeRunEnv(seed=4066, ascension_level=0, start_on_map=True)
        env.relics.append({"relic_id": "Golden Idol", "name": "Golden Idol", "tier": "SPECIAL"})
        env._draw_event_id = lambda: "Forgotten Altar"
        env._enter_event()

        self.assertEqual(
            [(item["choice_index"], item["name"]) for item in env.event_options],
            [(0, "Smashed Altar"), (1, "Shed Blood"), (2, "Ignored")],
        )

    def test_forgotten_altar_shed_blood_applies_max_hp_gain_before_hp_loss_in_v2(self):
        env = NativeRunEnv(seed=4067, ascension_level=0, start_on_map=True)
        env._draw_event_id = lambda: "Forgotten Altar"
        env.player.current_hp = 18
        env.player.max_hp = 80
        env._enter_event()

        env.step({"kind": "event", "event_id": "Forgotten Altar", "name": "Shed Blood", "choice_index": 1})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.player.max_hp, 85)
        self.assertEqual(env.player.current_hp, 3)

    def test_forgotten_altar_shed_blood_uses_sts_rounding_in_v2(self):
        env = NativeRunEnv(seed=40675, ascension_level=0, start_on_map=True)
        env._draw_event_id = lambda: "Forgotten Altar"
        env.player.current_hp = 51
        env.player.max_hp = 83
        env._enter_event()

        env.step({"kind": "event", "event_id": "Forgotten Altar", "name": "Shed Blood", "choice_index": 1})

        self.assertEqual(env.player.max_hp, 88)
        self.assertEqual(env.player.current_hp, 35)

    def test_forgotten_altar_smashed_altar_trades_golden_idol_for_bloody_idol_in_v2(self):
        env = NativeRunEnv(seed=4068, ascension_level=0, start_on_map=True)
        env.relics.append({"relic_id": "Golden Idol", "name": "Golden Idol", "tier": "SPECIAL"})
        env._draw_event_id = lambda: "Forgotten Altar"
        env._enter_event()

        env.step({"kind": "event", "event_id": "Forgotten Altar", "name": "Smashed Altar", "choice_index": 0})

        relic_ids = [relic["relic_id"] for relic in env.relics]
        self.assertNotIn("Golden Idol", relic_ids)
        self.assertIn("Bloody Idol", relic_ids)

    def test_scrap_ooze_event_damage_respects_torii_in_v2(self):
        env = NativeRunEnv(seed=4069, ascension_level=0, start_on_map=True)
        env.relics.append({"relic_id": "Torii", "name": "Torii", "tier": "RARE"})
        env._draw_event_id = lambda: "Scrap Ooze"
        env.player.current_hp = 14
        env.player.max_hp = 80
        env._enter_event()

        env.step({"kind": "event", "event_id": "Scrap Ooze", "name": "Success", "choice_index": 0})

        self.assertEqual(env.player.current_hp, 13)
        self.assertEqual(env.phase, "EVENT")

    def test_pleading_vagrant_omits_gold_option_when_player_cannot_pay_in_v2(self):
        env = NativeRunEnv(seed=4063, ascension_level=0, start_on_map=True)
        env._draw_event_id = lambda: "Pleading Vagrant"
        env.gold = 84
        env._enter_event()

        self.assertEqual(
            [(item["choice_index"], item["name"]) for item in env.event_options],
            [(1, "Robbed"), (2, "Ignored")],
        )

    def test_vampires_without_blood_vial_uses_choice_indexes_one_and_two_in_v2(self):
        env = NativeRunEnv(seed=4064, ascension_level=0, start_on_map=True)
        env._draw_event_id = lambda: "Vampires"
        env.relics = [relic for relic in env.relics if relic.get("relic_id") != "Blood Vial"]
        env._enter_event()

        self.assertEqual(
            [(item["choice_index"], item["name"]) for item in env.event_options],
            [(1, "Accepted"), (2, "Refused")],
        )

    def test_vampires_with_blood_vial_offers_three_choices_in_v2(self):
        env = NativeRunEnv(seed=4064, ascension_level=0, start_on_map=True)
        env._draw_event_id = lambda: "Vampires"
        env.relics.append({"relic_id": "Blood Vial", "name": "Blood Vial", "tier": "COMMON"})
        env._enter_event()

        self.assertEqual(
            [(item["choice_index"], item["name"]) for item in env.event_options],
            [(0, "Offered"), (1, "Accepted"), (2, "Refused")],
        )

    def test_vampires_offered_removes_blood_vial_without_max_hp_loss_in_v2(self):
        env = NativeRunEnv(seed=4064, ascension_level=0, start_on_map=True)
        env.phase = "EVENT"
        env.event_id = "Vampires"
        env.player.max_hp = 80
        env.player.current_hp = 80
        env.relics = [{"relic_id": "Blood Vial", "name": "Blood Vial", "tier": "COMMON"}]
        env.deck = [
            make_card("Strike_R", uuid="strike-0"),
            make_card("Strike_R", uuid="strike-1"),
            make_card("Defend_R", uuid="defend-0"),
        ]

        env.step({"kind": "event", "event_id": "Vampires", "name": "Offered", "choice_index": 0})

        self.assertEqual(env.player.max_hp, 80)
        self.assertEqual(env.player.current_hp, 80)
        self.assertNotIn("Blood Vial", [relic.get("relic_id") for relic in env.relics])
        self.assertNotIn("Strike_R", [card.card_id for card in env.deck])
        self.assertEqual(sum(1 for card in env.deck if card.card_id == "Bite"), 5)

    def test_divine_fountain_ignores_curse_of_the_bell_for_availability_in_v2(self):
        env = NativeRunEnv(seed=4064, ascension_level=0, start_on_map=True)
        env.deck = [make_card("CurseOfTheBell", uuid="bell-0")]

        self.assertFalse(env._can_add_one_time_event("The Divine Fountain"))

    def test_the_joust_only_offers_two_choices_in_v2(self):
        env = NativeRunEnv(seed=4064, ascension_level=0, start_on_map=True)
        env._draw_event_id = lambda: "The Joust"
        env.gold = 100
        env.act = 2
        env._enter_event()

        self.assertEqual(
            [(item["choice_index"], item["name"]) for item in env.event_options],
            [(0, "Murderer"), (1, "Owner")],
        )

    def test_pleading_vagrant_robbed_adds_shame_in_v2(self):
        env = NativeRunEnv(seed=4065, ascension_level=0, start_on_map=True)
        env.phase = "EVENT"
        env.event_id = "Pleading Vagrant"

        env.step({"kind": "event", "event_id": "Pleading Vagrant", "name": "Robbed", "choice_index": 1})

        self.assertIn("Shame", [card.card_id for card in env.deck])

    def test_pleading_vagrant_robbed_new_omamori_blocks_shame_in_v2(self):
        env = NativeRunEnv(seed=4065, ascension_level=0, start_on_map=True)
        env.phase = "EVENT"
        env.event_id = "Pleading Vagrant"
        env._roll_screenless_relic_of_tier = lambda tier: make_relic("Omamori")

        env.step({"kind": "event", "event_id": "Pleading Vagrant", "name": "Robbed", "choice_index": 1})

        self.assertNotIn("Shame", [card.card_id for card in env.deck])
        omamori = next(relic for relic in env.relics if relic["relic_id"] == "Omamori")
        self.assertEqual(omamori["counter"], 1)

    def test_pleading_vagrant_robbed_omamori_regression_seed_matches_lightspeed_in_v2(self):
        seed = 5431397250825282544
        result = compare_seed(seed, 0, 260, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_pleading_vagrant_gold_uses_screenless_relic_roll_regression_seed_matches_lightspeed_in_v2(self):
        seed = 774217106722755449
        result = compare_seed(seed, 0, 360, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_campfire_smith_opens_card_select_when_multiple_upgrades_exist_in_v2(self):
        env = NativeRunEnv(seed=4067, ascension_level=0, start_on_map=True)
        env.phase = "CAMPFIRE"
        env.deck = [
            make_card("Strike_R", uuid="strike-0"),
            make_card("Defend_R", uuid="defend-0"),
            make_card("Bash", uuid="bash-0"),
        ]

        state = env.step({"kind": "campfire", "name": "SMITH", "choice_index": 0})

        self.assertEqual(env.phase, "CARD_SELECT")
        self.assertIsInstance(state, dict)
        self.assertEqual(env.card_select_context, "CAMPFIRE_SMITH")
        self.assertEqual(env.card_select_completion, "CAMPFIRE_LEAVE")

    def test_centennial_puzzle_does_not_trigger_from_combust_self_damage_in_v2(self):
        combat = NativeCombatEnv(seed=407, ascension_level=0, player=PlayerState(current_hp=40, max_hp=40))
        combat.relics.append({"relic_id": "Centennial Puzzle", "name": "Centennial Puzzle", "tier": "COMMON", "counter": 0})
        combat.player.powers["Combust"] = 5
        combat.combust_hp_loss = 1
        combat.hand = [make_card("Strike_R", uuid="strike"), make_card("Wild Strike", uuid="wild")]
        combat.draw_pile = [
            make_card("Strike_R", uuid="draw-0"),
            make_card("Armaments", uuid="draw-1"),
            make_card("Dropkick", uuid="draw-2"),
            make_card("Defend_R", uuid="draw-3"),
            make_card("Bash", uuid="draw-4"),
        ]
        combat.discard_pile = []
        combat.monsters = [make_monster("SlimeBoss", StsRandom(407), ascension=0)]
        combat.monsters[0].current_hp = 135
        combat.monsters[0].move = "SLIME_BOSS_PREPARING"
        combat.monsters[0].intent = "UNKNOWN"
        combat.monsters[0].move_base_damage = 0
        combat.monsters[0].move_hits = 0

        combat.end_turn()

        self.assertEqual(combat.player.current_hp, 39)
        self.assertEqual(combat._relic("Centennial Puzzle").get("counter"), 1)
        self.assertEqual(len(combat.hand), 5)

    def test_centennial_puzzle_combust_self_damage_does_not_keep_reshuffled_cards_in_hand_in_v2(self):
        combat = NativeCombatEnv(seed=4071, ascension_level=0, player=PlayerState(current_hp=40, max_hp=40))
        combat.relics.append({"relic_id": "Centennial Puzzle", "name": "Centennial Puzzle", "tier": "COMMON", "counter": 0})
        combat.player.powers["Combust"] = 5
        combat.combust_hp_loss = 1
        combat.hand = [make_card("Strike_R", uuid="strike"), make_card("Thunderclap", uuid="thunderclap")]
        combat.draw_pile = []
        combat.discard_pile = [
            make_card("Bash", uuid="discard-0"),
            make_card("Defend_R", uuid="discard-1"),
            make_card("Strike_R", uuid="discard-2"),
            make_card("Defend_R", uuid="discard-3"),
            make_card("Injury", uuid="discard-4"),
        ]
        combat.monsters = [make_monster("Lagavulin", StsRandom(4071), ascension=0)]
        combat.monsters[0].current_hp = 108
        combat.monsters[0].max_hp = 108
        combat.monsters[0].move = "LAGAVULIN_ATTACK"
        combat.monsters[0].intent = "ATTACK"
        combat.monsters[0].move_base_damage = 18
        combat.monsters[0].move_hits = 1

        combat.end_turn()

        self.assertEqual(combat._relic("Centennial Puzzle").get("counter"), 1)
        self.assertEqual([card.card_id for card in combat.hand], ["Defend_R", "Defend_R", "Thunderclap", "Bash", "Injury"])
        self.assertEqual([card.card_id for card in combat.draw_pile], ["Strike_R", "Strike_R"])
        self.assertEqual([card.card_id for card in combat.discard_pile], [])

    def test_runic_cube_end_turn_drawn_carnage_exhausts_in_v2(self):
        combat = NativeCombatEnv(seed=408, ascension_level=0, player=PlayerState(current_hp=40, max_hp=40))
        combat.relics.append({"relic_id": "Runic Cube", "name": "Runic Cube", "tier": "BOSS"})
        combat.player.powers["Combust"] = 5
        combat.combust_hp_loss = 1
        combat.hand = [make_card("Strike_R", uuid="strike")]
        combat.draw_pile = [make_card("Carnage", uuid="draw-carnage")]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.monsters = [make_monster("Cultist", StsRandom(408), ascension=0)]

        combat.end_turn()

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Carnage"])
        self.assertNotIn("Carnage", [card.card_id for card in combat.discard_pile])

    def test_runic_cube_end_turn_drawn_carnage_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2659878691801223075
        result = compare_seed(seed, 0, 140, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_runic_cube_combust_end_turn_draw_dazed_does_not_trigger_evolve_immediately_in_v2(self):
        combat = NativeCombatEnv(seed=4082, ascension_level=0, player=PlayerState(current_hp=40, max_hp=40))
        combat.relics.append({"relic_id": "Runic Cube", "name": "Runic Cube", "tier": "BOSS"})
        combat.player.powers["Combust"] = 5
        combat.player.powers["Evolve"] = 1
        combat.combust_hp_loss = 1
        combat.hand = [make_card("Strike_R", uuid="hand-strike")]
        combat.draw_pile = [
            make_card("Strike_R", uuid="draw-strike"),
            make_card("Dazed", uuid="draw-dazed"),
        ]
        combat.discard_pile = []
        combat.exhaust_pile = []
        combat.monsters = [make_monster("Cultist", StsRandom(4082), ascension=0)]

        combat.end_turn()

        self.assertEqual([card.card_id for card in combat.exhaust_pile], ["Dazed"])
        self.assertEqual([card.card_id for card in combat.hand], ["Strike_R", "Strike_R"])
        self.assertEqual(combat.pending_start_turn_evolve_draws, 0)

    def test_runic_cube_combust_end_turn_dazed_regression_seed_matches_lightspeed_in_v2(self):
        seed = 4921959201585324293
        result = compare_seed(seed, 0, 240, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_the_bomb_explodes_before_monster_attack_at_end_turn_in_v2(self):
        combat = NativeCombatEnv(
            seed=4081,
            ascension_level=0,
            scheduled_encounter=["Sentry"],
            player=PlayerState(current_hp=20, max_hp=80),
        )
        combat.player.block = 5
        combat.player.powers["The Bomb"] = 1
        combat.player.powers["The Bomb Damage"] = 40
        combat.hand = [make_card("Dazed", uuid="hand-dazed-0"), make_card("Dazed", uuid="hand-dazed-1")]
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []
        sentry = combat.monsters[0]
        sentry.current_hp = 19
        sentry.max_hp = 39
        sentry.move = "SENTRY_BEAM"
        sentry.intent = "ATTACK"
        sentry.move_base_damage = 9
        sentry.move_hits = 1

        combat.end_turn()

        self.assertEqual(combat.outcome, "PLAYER_VICTORY")
        self.assertEqual(combat.player.current_hp, 20)

    def test_shelled_parasite_stunned_rolls_next_move_like_lightspeed_in_v2(self):
        seed = 5372018147044285222
        result = compare_seed(seed, 0, 270, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_shelled_parasite_stunned_rolls_next_move_variant_like_lightspeed_in_v2(self):
        seed = 5628613801529873643
        result = compare_seed(seed, 0, 315, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_zero_damage_body_slam_does_not_force_acid_slime_large_split_in_v2(self):
        seed = 8718688681030755219
        result = compare_seed(seed, 0, 130, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_perfected_strike_replay_does_not_overcount_self_from_discard_in_v2(self):
        seed = 6460944748906581622
        result = compare_seed(seed, 0, 175, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_havoc_replayed_perfected_strike_does_not_overcount_itself_in_v2(self):
        seed = 3855018187708896467
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_double_tap_havoc_whirlwind_replay_consumes_x_energy_regression_in_v2(self):
        seed = 1064099234444427506
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_ornamental_fan_juggernaut_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2830086076756393441
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_dramatic_entrance_aoe_relic_proc_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3008558597215218977
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_charons_ashes_spore_cloud_monster_turn_regression_seed_matches_lightspeed_in_v2(self):
        seed = 6356902814485274786
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_feed_on_gremlin_leader_minion_does_not_grant_hp_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8184069814800694377
        result = compare_seed(seed, 0, 430, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_body_slam_rage_snapshot_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3735545940371517021
        result = compare_seed(seed, 0, 120, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_spike_slime_flame_tackle_centennial_puzzle_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3602103758866957703
        result = compare_seed(seed, 0, 80, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_meat_on_the_bone_looter_escape_regression_seed_matches_lightspeed_in_v2(self):
        seed = 336190697703819741
        result = compare_seed(seed, 0, 170, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_lagavulin_centennial_puzzle_end_turn_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3301439061689089639
        result = compare_seed(seed, 0, 90, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_lagavulin_centennial_puzzle_empty_draw_pile_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8091456412596798918
        result = compare_seed(seed, 0, 100, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_cultist_centennial_puzzle_dark_embrace_end_turn_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1464080177070146593
        result = compare_seed(seed, 0, 100, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_finesse_evolve_shuffle_regression_seed_matches_lightspeed_in_v2(self):
        seed = 4876621465605962529
        result = compare_seed(seed, 0, 140, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_flash_of_steel_evolve_shuffle_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3043401873656461322
        result = compare_seed(seed, 0, 110, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_headbutt_unceasing_top_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2088451173468271302
        result = compare_seed(seed, 0, 175, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_blue_candle_rupture_regression_seed_matches_lightspeed_in_v2(self):
        seed = 5305511643868348657
        result = compare_seed(seed, 0, 70, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_bash_sadistic_nature_victory_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8530996140800604356
        result = compare_seed(seed, 0, 90, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_burning_pact_ink_bottle_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8282773350555957144
        result = compare_seed(seed, 0, 120, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_headbutt_replay_spore_cloud_fungibeast_triple_regression_seed_matches_lightspeed_in_v2(self):
        seed = 9178993164219457251
        result = compare_seed(seed, 0, 190, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_headbutt_replay_spore_cloud_fungibeast_pair_regression_seed_matches_lightspeed_in_v2(self):
        seed = 4064325335677521710
        result = compare_seed(seed, 0, 190, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_finesse_hex_evolve_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2141377328575795556
        result = compare_seed(seed, 0, 340, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_astrolabe_searing_blow_shelled_parasite_regression_seed_matches_lightspeed_in_v2(self):
        seed = 4842728808691365766
        result = compare_seed(seed, 0, 290, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_astrolabe_searing_blow_chosen_regression_seed_matches_lightspeed_in_v2(self):
        seed = 5129868176984913808
        result = compare_seed(seed, 0, 350, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_apotheosis_searing_blow_sentry_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1009898553184381870
        result = compare_seed(seed, 0, 120, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_pandoras_box_reward_roll_timeout_regression_seed_matches_lightspeed_in_v2(self):
        seed = 6030437330670576993
        result = compare_seed(seed, 0, 320, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_pandoras_box_transform_order_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1777187355622723266
        result = compare_seed(seed, 0, 180, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_pandoras_box_transform_variant_regression_seed_matches_lightspeed_in_v2(self):
        seed = 161491087696271874
        result = compare_seed(seed, 0, 285, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_blood_for_blood_fungibeast_regression_seed_matches_lightspeed_in_v2(self):
        seed = 5462830306545858817
        result = compare_seed(seed, 0, 120, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_shop_good_instincts_courier_regression_seed_matches_lightspeed_in_v2(self):
        seed = 5634853390220523375
        result = compare_seed(seed, 0, 130, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_shop_bird_faced_urn_visibility_regression_seed_matches_lightspeed_in_v2(self):
        seed = 7264005430981265320
        result = compare_seed(seed, 0, 260, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_blood_for_blood_spike_slime_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8475787393379572336
        result = compare_seed(seed, 0, 420, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_golden_idol_ancient_tea_set_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1123816951451513685
        result = compare_seed(seed, 0, 90, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_sentry_burning_elite_strength_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2880820615560772120
        result = compare_seed(seed, 0, 90, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_double_tap_infernal_blade_uppercut_regression_seed_matches_lightspeed_in_v2(self):
        seed = 558790650080141164
        result = compare_seed(seed, 0, 60, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_whirlwind_end_turn_dark_embrace_dead_branch_order_regression_seed_matches_lightspeed_in_v2(self):
        seed = 4505971042505850327
        result = compare_seed(seed, 0, 90, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_the_guardian_sharp_hide_intangible_regression_seed_matches_lightspeed_in_v2(self):
        seed = 314683764399153928
        result = compare_seed(seed, 0, 230, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_the_guardian_sharp_hide_intangible_variant_matches_lightspeed_in_v2(self):
        seed = 3523482283509041288
        result = compare_seed(seed, 0, 170, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_fiend_fire_empty_hand_sharp_hide_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8686722833710722816
        result = compare_seed(seed, 0, 260, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_armaments_madness_entrench_cost_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3723006110372850514
        result = compare_seed(seed, 0, 320, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_double_tap_headbutt_replay_preserves_ink_bottle_across_battles_in_v2(self):
        seed = 5724473262074921944
        result = compare_seed(seed, 0, 95, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_double_tap_headbutt_replay_advances_ink_bottle_for_second_defend_in_v2(self):
        seed = 6339477635611240296
        result = compare_seed(seed, 0, 140, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_snecko_blood_for_blood_uses_damage_reduction_cost_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3989297786924186112
        result = compare_seed(seed, 0, 340, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_blood_for_blood_sharp_hide_counter_damage_regression_seed_matches_lightspeed_in_v2(self):
        seed = 5362227471853472608
        result = compare_seed(seed, 0, 250, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_blood_for_blood_monster_turn_damage_regression_seed_matches_lightspeed_in_v2(self):
        seed = 7800527860349531815
        result = compare_seed(seed, 0, 120, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_cursed_tome_necronomicon_hp_loss_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3186585226306213111
        result = compare_seed(seed, 0, 285, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_dead_adventurer_ancient_tea_set_sentries_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1008004744070626160
        result = compare_seed(seed, 0, 110, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_dead_adventurer_ancient_tea_set_lagavulin_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8647786733796371168
        result = compare_seed(seed, 0, 145, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_corruption_drawn_apotheosis_upgraded_entrench_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2871947797449883100
        result = compare_seed(seed, 0, 320, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_lagavulin_dark_embrace_carnage_end_turn_regression_seed_matches_lightspeed_in_v2(self):
        seed = 4995787630853733854
        result = compare_seed(seed, 0, 100, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_finesse_hex_evolve_draw_order_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3809961811556771397
        result = compare_seed(seed, 0, 260, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_sentry_dark_embrace_deferred_ethereal_exhaust_order_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1688338378649432294
        result = compare_seed(seed, 0, 140, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_bottled_lightning_reward_closes_remaining_relics_regression_seed_matches_lightspeed_in_v2(self):
        seed = 7471708822804843731
        result = compare_seed(seed, 0, 500, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_burning_pact_base_draw_reshuffle_overflow_regression_seed_matches_lightspeed_in_v2(self):
        seed = 5101317382594045103
        result = compare_seed(seed, 0, 120, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_burning_pact_hex_evolve_draw_order_regression_seed_matches_lightspeed_in_v2(self):
        seed = 161491087696271874
        result = compare_seed(seed, 0, 400, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_snecko_necronomicon_anger_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2957407203887308767
        result = compare_seed(seed, 0, 390, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_snecko_necronomicon_anger_second_regression_seed_matches_lightspeed_in_v2(self):
        seed = 5339294800862891969
        result = compare_seed(seed, 0, 420, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_chosen_blue_candle_hex_curse_regression_seed_matches_lightspeed_in_v2(self):
        seed = 912887060817094847
        result = compare_seed(seed, 0, 280, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_chosen_blue_candle_hex_curse_second_regression_seed_matches_lightspeed_in_v2(self):
        seed = 7605545578724682123
        result = compare_seed(seed, 0, 380, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_bronze_orb_stasis_draw_order_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8641874880895185309
        result = compare_seed(seed, 0, 700, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_bronze_orb_stasis_rare_sort_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1026256006531929103
        result = compare_seed(seed, 0, 520, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_snakeplant_ornamental_fan_juggernaut_malleable_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1122483788265248191
        result = compare_seed(seed, 0, 500, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_living_wall_remove_clears_ancient_tea_set_regression_seed_matches_lightspeed_in_v2(self):
        seed = 357627622572846691
        result = compare_seed(seed, 0, 300, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_campfire_smith_arms_ancient_tea_set_regression_seed_matches_lightspeed_in_v2(self):
        seed = 6651781634082037411
        result = compare_seed(seed, 0, 500, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_sentry_battle_trance_dark_embrace_end_turn_draw_regression_seed_matches_lightspeed_in_v2(self):
        seed = 602353918093405543
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_jaw_worm_battle_trance_dark_embrace_end_turn_draw_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1441502644376229977
        result = compare_seed(seed, 0, 240, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_fuzzy_louse_double_tap_headbutt_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3940415016561543311
        result = compare_seed(seed, 0, 120, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_sentry_havoc_letter_opener_target_lock_regression_seed_matches_lightspeed_in_v2(self):
        seed = 529735052817431060
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_fuzzy_louse_pummel_shuriken_timing_regression_seed_matches_lightspeed_in_v2(self):
        seed = 986344161167740523
        result = compare_seed(seed, 0, 130, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_lagavulin_shop_searing_blow_double_upgrade_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1844461198983084346
        result = compare_seed(seed, 0, 140, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_sentry_dark_embrace_fire_breathing_cleanup_timing_regression_seed_matches_lightspeed_in_v2(self):
        seed = 6573887515320924554
        result = compare_seed(seed, 0, 140, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_sentry_dark_embrace_fire_breathing_prevents_end_turn_beam_regression_seed_matches_lightspeed_in_v2(self):
        seed = 6514690704412421820
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_looter_mugger_start_turn_fire_breathing_sundial_regression_seed_matches_lightspeed_in_v2(self):
        seed = 4307483191949508836
        result = compare_seed(seed, 0, 260, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_havoc_empty_piles_cant_win_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3106528707515613951
        result = compare_seed(seed, 0, 200, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_fuzzy_louse_slaver_cleave_shuriken_timing_regression_seed_matches_lightspeed_in_v2(self):
        seed = 6321069398540135409
        result = compare_seed(seed, 0, 130, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_slaver_havoc_headbutt_rage_timing_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2110113912458741132
        result = compare_seed(seed, 0, 130, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_spike_slime_dual_wield_ink_bottle_regression_seed_matches_lightspeed_in_v2(self):
        seed = 287165333831333479
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_sentry_dropkick_evolve_shuffle_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8891796826793370411
        result = compare_seed(seed, 0, 140, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_sentry_burning_pact_evolve_shuffle_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2185054612018301625
        result = compare_seed(seed, 0, 140, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_shining_light_lizard_tail_run_death_regression_seed_matches_lightspeed_in_v2(self):
        seed = 6143104480870334253
        result = compare_seed(seed, 0, 140, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_shining_light_fairy_potion_survival_regression_seed_matches_lightspeed_in_v2(self):
        seed = 7195347014358151499
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_spike_slime_large_move_history_split_regression_seed_matches_lightspeed_in_v2(self):
        seed = 9140430691451525301
        result = compare_seed(seed, 0, 180, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_lab_reward_clears_ancient_tea_set_before_next_event_room_regression_seed_matches_lightspeed_in_v2(self):
        seed = 4065646309702048324
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_armaments_upgraded_limit_break_still_exhausts_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3381509250774527929
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_armaments_limit_break_exhaust_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2244611209453146389
        result = compare_seed(seed, 0, 335, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_armaments_limit_break_exhaust_shelled_parasite_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2989983138484222445
        result = compare_seed(seed, 0, 305, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_snake_plant_enfeebling_spores_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2521157635577792591
        result = compare_seed(seed, 0, 378, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_fiend_fire_shuriken_timing_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8158915933587575513
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_shining_light_blood_vial_sentry_the_bomb_regression_seed_matches_lightspeed_in_v2(self):
        seed = 5795041445776309026
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_hexaghost_divider_thorns_regression_seed_matches_lightspeed_in_v2(self):
        seed = 4685557471079173135
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_weakened_zero_damage_multi_hit_still_triggers_thorns_in_v2(self):
        combat = NativeCombatEnv(seed=111649, ascension_level=0, player=PlayerState())
        byrd = make_monster("Byrd", StsRandom(111650), ascension=0)
        byrd.current_hp = 10
        byrd.max_hp = 31
        byrd.move = "BYRD_PECK"
        byrd.intent = "ATTACK"
        byrd.move_base_damage = 1
        byrd.move_hits = 5
        byrd.powers["Weakened"] = 1
        combat.monsters = [byrd]
        combat.player.powers["Thorns"] = 3

        combat._monster_take_turn(byrd, 0)

        self.assertEqual(byrd.current_hp, 0)

    def test_byrd_weakened_peck_thorns_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1688779343593914916
        result = compare_seed(seed, 0, 300, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_sentry_offering_dead_branch_evolve_order_regression_seed_matches_lightspeed_in_v2(self):
        seed = 3448207236104408935
        result = compare_seed(seed, 0, 260, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_stone_calendar_guardian_shift_suppresses_followup_combust_in_v2(self):
        combat = NativeCombatEnv(seed=111641, ascension_level=0, scheduled_encounter=["TheGuardian"], player=PlayerState(current_hp=20, max_hp=80))
        combat.relics.append({"relic_id": "Stone Calendar", "name": "Stone Calendar", "tier": "RARE", "counter": -1})
        guardian = combat.monsters[0]
        guardian.current_hp = 158
        guardian.max_hp = 250
        guardian.move = "THE_GUARDIAN_WHIRLWIND"
        guardian.intent = "ATTACK"
        guardian.move_base_damage = 5
        guardian.move_hits = 4
        guardian.powers["Mode Shift"] = 32
        guardian.powers["Vulnerable"] = 2
        combat.player.powers["Combust"] = 5
        combat.combust_hp_loss = 1
        combat.turn = 6
        combat.hand = []
        combat.draw_pile = []
        combat.discard_pile = []
        combat.exhaust_pile = []

        combat.end_turn()

        self.assertEqual(guardian.current_hp, 106)

    def test_whirlwind_the_guardian_sharp_hide_regression_seed_matches_lightspeed_in_v2(self):
        seed = 1775779181486539754
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_stone_calendar_combust_guardian_regression_seed_matches_lightspeed_in_v2(self):
        seed = 6026904106236140508
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_hand_of_greed_sharp_hide_regression_seed_matches_lightspeed_in_v2(self):
        seed = 5227011572645708121
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_unceasing_top_evolve_status_draw_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2777667323528479993
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_sentry_dual_wield_dead_branch_dark_embrace_order_regression_seed_matches_lightspeed_in_v2(self):
        seed = 6091332802294016973
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_snecko_armaments_upgraded_apparition_retains_regression_seed_matches_lightspeed_in_v2(self):
        seed = 821696850342151935
        result = compare_seed(seed, 0, 500, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_shrug_it_off_ink_bottle_evolve_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8259191592959035651
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_letter_opener_juggernaut_gremlin_pack_regression_seed_matches_lightspeed_in_v2(self):
        seed = 5377879752004883528
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_nloth_choice_order_regression_seed_matches_lightspeed_in_v2(self):
        seed = 189734299680565632
        result = compare_seed(seed, 0, 320, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_anger_sharp_hide_fairy_potion_regression_seed_matches_lightspeed_in_v2(self):
        seed = 8902546551360297163
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_snecko_confusion_juggernaut_self_forming_clay_regression_seed_matches_lightspeed_in_v2(self):
        seed = 356258230518758363
        result = compare_seed(seed, 0, 430, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_unceasing_top_fire_breathing_end_turn_regression_seed_matches_lightspeed_in_v2(self):
        seed = 526779264429346930
        result = compare_seed(seed, 0, 320, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_sentry_reward_brutality_fire_breathing_regression_seed_matches_lightspeed_in_v2(self):
        seed = 2736691288915730697
        result = compare_seed(seed, 0, 180, backend="v2")

        self.assertTrue(result["match"], msg=str(result))

    def test_sentry_juggernaut_block_clear_regression_seed_matches_lightspeed_in_v2(self):
        seed = 6470814891605189691
        result = compare_seed(seed, 0, 220, backend="v2")

        self.assertTrue(result["match"], msg=str(result))
