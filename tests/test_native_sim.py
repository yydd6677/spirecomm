import unittest
import random

from spirecomm.ai.rl import ACTION_END_TURN, build_action_mask, build_state_tensors
from spirecomm.native_sim.cards import CARD_LIBRARY, COLORLESS_CARD_IDS, ironclad_reward_pool, make_card
from spirecomm.native_sim import NativeRunEnv
from spirecomm.native_sim.monsters import choose_next_move, make_monster, monster_adjusted_damage, roll_act1_encounter
from spirecomm.native_sim.potions import POTION_LIBRARY, make_potion
from spirecomm.native_sim.relics import make_relic
from spirecomm.native_sim.schema import MonsterState, PlayerState


class NativeSimSmokeTest(unittest.TestCase):
    def test_state_is_spirecomm_model_compatible(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        state = env.state()

        tensors = build_state_tensors(state)

        self.assertIn("global_features", tensors)
        self.assertIn("hand_features", tensors)
        self.assertIn("monster_features", tensors)
        self.assertGreater(len(tensors["global_features"]), 0)
        self.assertGreater(len(tensors["hand_features"]), 0)
        self.assertGreater(len(tensors["monster_features"]), 0)

    def test_runic_dome_hides_monster_move_fields(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.combat.relics.append(make_relic("Runic Dome"))
        state = env.combat.to_spirecomm_state()
        monster = state["combat_state"]["monsters"][0]

        self.assertEqual(monster["intent"], "UNKNOWN")
        self.assertEqual(monster["move_adjusted_damage"], 0)
        self.assertEqual(monster["move_base_damage"], 0)
        self.assertEqual(monster["move_hits"], 0)
        self.assertIsNone(monster["move_id"])

    def test_ironclad_reward_pool_excludes_colorless_and_has_spot_weakness(self):
        reward_ids = {card.card_id for card in ironclad_reward_pool()}

        self.assertIn("Spot Weakness", reward_ids)
        self.assertTrue(reward_ids.isdisjoint(COLORLESS_CARD_IDS))
        self.assertNotIn("Strike_R", reward_ids)
        self.assertNotIn("Defend_R", reward_ids)
        self.assertNotIn("Bash", reward_ids)

    def test_incomplete_ironclad_unlocks_are_excluded_from_rewards(self):
        env = NativeRunEnv(seed=3573713328584081974, ironclad_unlock_level=0)
        reward = [card.card_id for card in env._roll_card_reward()]

        self.assertEqual(set(reward), {"Thunderclap", "Clash", "Clothesline"})
        locked = {
            "Heavy Blade",
            "Spot Weakness",
            "Limit Break",
            "Wild Strike",
            "Evolve",
            "Immolate",
            "Havoc",
            "Sentinel",
            "Exhume",
        }
        for _ in range(20):
            for card in env._roll_card_reward():
                self.assertNotIn(card.card_id, locked)

    def test_end_turn_is_masked_while_cards_are_playable(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        state = env.state()

        self.assertTrue(any(card.get("is_playable") for card in state["combat_state"]["hand"]))
        action_mask = build_action_mask(state)

        self.assertFalse(action_mask[ACTION_END_TURN])

    def test_can_step_a_legal_action(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        before = env.state()
        action = env.legal_actions()[0]
        after = env.step(action)

        self.assertEqual(after["character"], "IRONCLAD")
        self.assertEqual(after["seed"], before["seed"])
        self.assertIn(after["combat_state"]["turn"], {before["combat_state"]["turn"], before["combat_state"]["turn"] + 1})

    def test_all_ironclad_playable_cards_have_effect_paths(self):
        skipped_types = {"STATUS", "CURSE"}
        for card_id, card_def in CARD_LIBRARY.items():
            if card_def.card_type in skipped_types:
                continue
            with self.subTest(card_id=card_id):
                env = NativeRunEnv(seed=11, ascension_level=0)
                env.combat.hand = [make_card(card_id, uuid=f"test-{card_id}")]
                env.combat.player.energy = 99
                env.combat.play_card(0, 0)

    def test_run_state_machine_can_enter_non_combat_rooms(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.phase = "MAP"
        env.map_options = []

        for symbol, expected_phase in [
            ("R", "CAMPFIRE"),
            ("$", "SHOP"),
        ]:
            with self.subTest(symbol=symbol):
                env.phase = "MAP"
                env.step({"kind": "map", "symbol": symbol, "name": symbol})
                self.assertEqual(env.phase, expected_phase)
        env.phase = "MAP"
        env.step({"kind": "map", "symbol": "?", "name": "?"})
        self.assertIn(env.phase, {"EVENT", "COMBAT", "SHOP", "CHEST"})

    def test_map_choices_follow_persistent_graph_edges(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.combat.outcome = "PLAYER_VICTORY"
        env.step({"kind": "end", "name": "RESOLVE_COMBAT"})
        env.step({"kind": "skip", "name": "SKIP", "choice_index": len(env.reward_cards)})

        first_options = env.legal_actions()
        self.assertGreaterEqual(len(first_options), 2)
        first_choice = first_options[0]
        env.step(first_choice)
        chosen_node = first_choice["node_id"]
        env.combat.outcome = "PLAYER_VICTORY"
        env.step({"kind": "end", "name": "RESOLVE_COMBAT"})
        env.step({"kind": "skip", "name": "SKIP", "choice_index": len(env.reward_cards)})

        next_node_ids = {action["node_id"] for action in env.legal_actions()}
        self.assertTrue(next_node_ids)
        self.assertTrue(next_node_ids.issubset(set(env.map_graph[chosen_node]["children"])))

    def test_hand_size_is_capped_to_model_slots(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.combat.hand = [make_card("Strike_R", uuid=f"hand-{index}") for index in range(10)]
        env.combat.draw_pile = [make_card("Strike_R", uuid="extra")]

        env.combat.draw_cards(1)

        self.assertEqual(len(env.combat.hand), 10)
        self.assertTrue(all(action.get("card_index", 0) < 10 for action in env.combat.legal_actions() if action.get("kind") == "card"))

    def test_clash_and_blood_for_blood_legal_costs_are_modeled(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        clash = make_card("Clash", uuid="clash")
        defend = make_card("Defend_R", uuid="defend")
        env.combat.hand = [clash, defend]
        self.assertFalse(env.combat.playable(clash))

        blood = make_card("Blood for Blood", uuid="blood")
        env.combat.hand = [blood]
        env.combat.hp_lost_this_combat = 4
        env.combat.player.energy = 0
        self.assertTrue(env.combat.playable(blood))
        self.assertEqual(env.combat.to_spirecomm_state()["combat_state"]["hand"][0]["cost_for_turn"], 0)

    def test_slimed_is_playable_and_void_loses_energy_on_draw(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        slimed = make_card("Slimed", uuid="slimed")
        env.combat.hand = [slimed]
        env.combat.player.energy = 1
        self.assertTrue(env.combat.playable(slimed))
        env.combat.play_card(0, 0)
        self.assertIn(slimed, env.combat.exhaust_pile)

        env.combat.draw_pile = [make_card("Void", uuid="void")]
        env.combat.hand = []
        env.combat.player.energy = 2
        env.combat.draw_cards(1)
        self.assertEqual(env.combat.player.energy, 1)

    def test_ascension_modifiers_change_starting_hp_and_monsters(self):
        low = NativeRunEnv(seed=1, ascension_level=0)
        high = NativeRunEnv(seed=1, ascension_level=20)

        self.assertLess(high.player.current_hp, high.player.max_hp)
        self.assertGreaterEqual(high.combat.monsters[0].max_hp, low.combat.monsters[0].max_hp)
        self.assertTrue(any(card.card_id == "AscendersBane" for card in high.deck))

    def test_boss_card_reward_flows_into_boss_relic(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.phase = "CARD_REWARD"
        env.floor = 16
        env.reward_cards = [make_card("Inflame", uuid="reward")]

        env.step({"kind": "skip", "name": "SKIP", "choice_index": 1})

        self.assertEqual(env.phase, "BOSS_RELIC")
        self.assertEqual(len(env.legal_actions()), 3)

    def test_neow_has_costed_options_and_can_swap_relic(self):
        env = NativeRunEnv(seed=1, ascension_level=0, enable_neow=True)

        self.assertEqual(env.phase, "NEOW")
        self.assertTrue(any(action.get("name") == "BOSS_RELIC_SWAP" for action in env.legal_actions()))
        env.step(next(action for action in env.legal_actions() if action.get("name") == "BOSS_RELIC_SWAP"))

        self.assertEqual(env.phase, "COMBAT")
        self.assertFalse(any(relic.get("relic_id") == "Burning Blood" for relic in env.relics))
        self.assertTrue(any(relic.get("tier") == "BOSS" for relic in env.relics))

    def test_innate_cards_are_drawn_on_first_turn(self):
        combat = NativeRunEnv(seed=1, ascension_level=0).combat.__class__(
            seed=2,
            player=PlayerState(),
            deck=[
                make_card("Strike_R", uuid="strike-1"),
                make_card("Strike_R", uuid="strike-2"),
                make_card("Defend_R", uuid="defend-1"),
                make_card("AscendersBane", uuid="asc-bane"),
                make_card("Writhe", uuid="writhe"),
            ],
        )

        self.assertIn("AscendersBane", {card.card_id for card in combat.hand})
        self.assertIn("Writhe", {card.card_id for card in combat.hand})

    def test_supported_monster_moves_are_damage_serializable(self):
        monster_ids = [
            "Cultist", "JawWorm", "AcidSlime_M", "SpikeSlime_M", "RedLouse", "GreenLouse",
            "FungiBeast", "SlaverBlue", "SlaverRed", "GremlinFat", "GremlinWizard", "GremlinThief",
            "Looter", "Mugger", "Bear", "Pointy", "Romeo", "GremlinTsundere", "GremlinWarrior",
            "GremlinNob", "Lagavulin", "Sentry",
            "SlimeBoss", "AcidSlime_L", "SpikeSlime_L",
            "Hexaghost", "TheGuardian", "Byrd", "Chosen", "SphericGuardian", "SnakePlant",
            "Snecko", "ShelledParasite", "Centurion", "Mystic", "BookOfStabbing",
            "GremlinLeader", "Taskmaster", "TheChamp", "TheCollector", "TorchHead", "BronzeAutomaton", "BronzeOrb",
            "Darkling", "OrbWalker", "TheMaw", "Transient", "WrithingMass", "Spiker",
            "Exploder", "Repulsor", "Nemesis", "GiantHead", "Reptomancer", "SnakeDagger", "AwakenedOne",
            "TimeEater", "Donu", "Deca", "SpireShield", "SpireSpear", "CorruptHeart",
        ]
        rng = random.Random(1)
        player = PlayerState()
        for monster_id in monster_ids:
            with self.subTest(monster_id=monster_id):
                monster = make_monster(monster_id, rng)
                for _ in range(8):
                    choose_next_move(monster, rng)
                    self.assertGreaterEqual(monster_adjusted_damage(monster, player), 0)

    def test_key_elite_and_boss_move_sequences_are_stateful(self):
        rng = random.Random(1)

        nob = make_monster("GremlinNob", rng)
        self.assertEqual(nob.move, "GREMLIN_NOB_BELLOW")
        choose_next_move(nob, rng)
        self.assertEqual(nob.move, "GREMLIN_NOB_RUSH")
        choose_next_move(nob, rng)
        choose_next_move(nob, rng)
        self.assertEqual(nob.move, "GREMLIN_NOB_SKULL_BASH")

        slime = make_monster("SlimeBoss", rng)
        choose_next_move(slime, rng)
        choose_next_move(slime, rng)
        self.assertEqual(slime.move, "SLIME_BOSS_PREPARING")
        choose_next_move(slime, rng)
        self.assertEqual(slime.move, "SLIME_BOSS_SLAM")

        hexa = make_monster("Hexaghost", rng)
        choose_next_move(hexa, rng)
        self.assertEqual(hexa.move, "HEXAGHOST_DIVIDER")
        choose_next_move(hexa, rng)
        self.assertEqual(hexa.move, "HEXAGHOST_SEAR")

        book = make_monster("BookOfStabbing", rng)
        choose_next_move(book, rng)
        first_hits = book.move_hits
        choose_next_move(book, rng)
        self.assertGreaterEqual(book.move_hits, first_hits)

    def test_lagavulin_sleep_requires_wakeup(self):
        rng = random.Random(1)
        lagavulin = make_monster("Lagavulin", rng)
        choose_next_move(lagavulin, rng)
        self.assertEqual(lagavulin.move, "LAGAVULIN_SLEEP")

        env = NativeRunEnv(seed=1, ascension_level=0)
        env.combat.monsters = [lagavulin]
        env.combat.hand = [make_card("Strike_R", uuid="wake")]
        env.combat.player.energy = 3
        env.combat.play_card(0, 0)

        self.assertEqual(lagavulin.ai_state.get("awoken"), 1)
        self.assertNotEqual(lagavulin.move, "LAGAVULIN_SLEEP")

    def test_countdown_monsters_use_countdown_moves(self):
        rng = random.Random(1)

        wizard = make_monster("GremlinWizard", rng)
        choose_next_move(wizard, rng)
        choose_next_move(wizard, rng)
        choose_next_move(wizard, rng)
        self.assertEqual(wizard.move, "WIZARD_ULTIMATE_BLAST")

        exploder = make_monster("Exploder", rng)
        choose_next_move(exploder, rng)
        choose_next_move(exploder, rng)
        choose_next_move(exploder, rng)
        self.assertEqual(exploder.move, "EXPLODER_EXPLODE")

        dagger = make_monster("SnakeDagger", rng)
        choose_next_move(dagger, rng)
        choose_next_move(dagger, rng)
        self.assertEqual(dagger.move, "DAGGER_EXPLODE")

        transient = make_monster("Transient", rng)
        choose_next_move(transient, rng)
        first_damage = transient.move_base_damage
        choose_next_move(transient, rng)
        self.assertGreater(transient.move_base_damage, first_damage)

    def test_structural_minion_encounters_are_seeded_with_minions(self):
        rng = random.Random(1)
        collector = [monster.monster_id for monster in roll_act1_encounter(rng, floor=33, act=2, act_boss="The Collector")]
        self.assertEqual(collector.count("TorchHead"), 2)

        rng = random.Random(1)
        automaton = [monster.monster_id for monster in roll_act1_encounter(rng, floor=33, act=2, act_boss="Bronze Automaton")]
        self.assertEqual(automaton.count("BronzeOrb"), 2)

        rng = random.Random(2)
        reptomancer = [monster.monster_id for monster in roll_act1_encounter(rng, floor=40, act=3, elite=True)]
        if "Reptomancer" in reptomancer:
            self.assertGreaterEqual(reptomancer.count("SnakeDagger"), 2)

    def test_red_slaver_entangle_masks_attacks_for_a_turn(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        red_slaver = make_monster("SlaverRed", random.Random(1))
        red_slaver.move = "RED_SLAVER_ENTANGLE"
        env.combat.monsters = [red_slaver]
        env.combat.hand = [make_card("Strike_R", uuid="strike"), make_card("Defend_R", uuid="defend")]
        env.combat.player.energy = 3

        env.combat._monster_take_turn(red_slaver)

        self.assertFalse(env.combat.playable(env.combat.hand[0]))
        self.assertTrue(env.combat.playable(env.combat.hand[1]))

    def test_looter_steals_gold_and_returns_it_if_defeated(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        looter = make_monster("Looter", random.Random(1))
        looter.move = "LOOTER_MUG"
        env.combat.monsters = [looter]
        env.combat.gold = 99

        env.combat._monster_take_turn(looter)

        self.assertEqual(env.combat.gold, 84)
        self.assertEqual(looter.ai_state["stolen_gold"], 15)

        looter.current_hp = 1
        env.combat.hand = [make_card("Strike_R", uuid="kill-looter")]
        env.combat.player.energy = 3
        env.combat.play_card(0, 0)

        self.assertGreaterEqual(env.combat.gold_gain, 15)

    def test_artifact_blocks_player_debuffs_from_monsters(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.combat.player.powers["Artifact"] = 1
        env.combat._apply_player_power("Hex", 1)

        self.assertEqual(env.combat.player.power("Artifact"), 0)
        self.assertEqual(env.combat.player.power("Hex"), 0)

    def test_confusion_randomizes_newly_drawn_card_costs(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.combat.rng = random.Random(1)
        env.combat.hand = []
        env.combat.discard_pile = []
        env.combat.draw_pile = [make_card("Bash", uuid=f"bash-{index}") for index in range(5)]
        env.combat.player.powers["Confusion"] = 1

        env.combat.draw_cards(5)

        self.assertEqual([card.cost_for_turn for card in env.combat.hand], [1, 0, 2, 0, 3])

    def test_bronze_orb_stasis_returns_stolen_card_on_death(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        orb = make_monster("BronzeOrb", random.Random(1))
        orb.move = "BRONZE_ORB_STASIS"
        env.combat.monsters = [orb]
        env.combat.draw_pile = [make_card("Bash", uuid="stolen")]
        env.combat.hand = []

        env.combat._monster_take_turn(orb)

        self.assertEqual(env.combat.draw_pile, [])
        self.assertEqual(orb.ai_state["stasis_card"].card_id, "Bash")
        orb.current_hp = 1
        env.combat.hand = [make_card("Strike_R", uuid="kill-orb")]
        env.combat.player.energy = 3
        env.combat.play_card(0, 0)
        self.assertTrue(any(card.card_id == "Bash" for card in env.combat.hand + env.combat.discard_pile))

    def test_native_targets_keep_model_slots_compressed_after_summons(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        dead_monsters = [
            MonsterState(f"Dead{index}", f"Dead {index}", 0, 10, "WAIT", "UNKNOWN")
            for index in range(7)
        ]
        for monster in dead_monsters:
            monster.is_gone = True
        alive = MonsterState("Alive", "Alive", 10, 10, "WAIT", "UNKNOWN")
        env.combat.monsters = dead_monsters + [alive]
        env.combat.hand = [make_card("Strike_R", uuid="target-compress")]
        env.combat.player.energy = 3

        action = next(action for action in env.combat.legal_actions() if action.get("kind") == "card")
        state = env.combat.to_spirecomm_state()

        self.assertEqual(action["target_index"], 7)
        self.assertEqual(action["model_target_index"], 0)
        self.assertEqual(state["combat_state"]["monsters"][0]["monster_id"], "Alive")

    def test_random_move_picker_avoids_immediate_repeats_when_possible(self):
        rng = random.Random(1)
        jaw = make_monster("JawWorm", rng)
        jaw.move = "JAW_WORM_CHOMP"
        jaw.move_history = ["JAW_WORM_CHOMP"]

        for _ in range(20):
            choose_next_move(jaw, rng)
            self.assertNotEqual(jaw.move, "JAW_WORM_CHOMP")
            jaw.move_history = ["JAW_WORM_CHOMP"]

    def test_byrd_flight_can_be_broken_into_stun(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        byrd = make_monster("Byrd", random.Random(1))
        env.combat.monsters = [byrd]

        env.combat._deal_attack_damage(6, byrd, hits=3)

        self.assertEqual(byrd.move, "BYRD_STUNNED")
        env.combat._monster_take_turn(byrd)
        self.assertEqual(byrd.power("Flight"), 3)

    def test_nemesis_intangible_and_giant_head_slow_are_visible_powers(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        nemesis = make_monster("Nemesis", random.Random(1))
        choose_next_move(nemesis, random.Random(1))
        env.combat.monsters = [nemesis]
        env.combat.start_player_turn()

        self.assertEqual(nemesis.power("Intangible"), 1)
        before = nemesis.current_hp
        env.combat._apply_damage_to_monster(30, nemesis)
        self.assertEqual(before - nemesis.current_hp, 1)

        env = NativeRunEnv(seed=1, ascension_level=0)
        giant = make_monster("GiantHead", random.Random(1))
        env.combat.monsters = [giant]
        env.combat.hand = [make_card("Strike_R", uuid="slow")]
        env.combat.player.energy = 3
        env.combat.play_card(0, 0)

        self.assertGreaterEqual(giant.power("Slow"), 1)

    def test_spiker_grow_increases_thorns_not_strength(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        spiker = make_monster("Spiker", random.Random(1))
        spiker.move = "SPIKER_GROW"
        before_thorns = spiker.power("Thorns")

        env.combat.monsters = [spiker]
        env.combat._monster_take_turn(spiker)

        self.assertGreater(spiker.power("Thorns"), before_thorns)
        self.assertEqual(spiker.power("Strength"), 0)

    def test_monster_block_resets_before_monster_turn_unless_barricaded(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        cultist = make_monster("Cultist", random.Random(1))
        cultist.move = "CULTIST_INCANTATION"
        cultist.block = 12
        env.combat.monsters = [cultist]
        env.combat.hand = []
        env.combat.player.current_hp = 80

        env.combat.end_turn()

        self.assertEqual(cultist.block, 0)

        env = NativeRunEnv(seed=2, ascension_level=0)
        spheric = make_monster("SphericGuardian", random.Random(1))
        spheric.move = "SPHERIC_ACTIVATE"
        spheric.block = 40
        env.combat.monsters = [spheric]
        env.combat.hand = []
        env.combat.end_turn()

        self.assertGreaterEqual(spheric.block, 40)

    def test_enemy_malleable_and_plated_armor_affect_block(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        plant = make_monster("SnakePlant", random.Random(1))
        env.combat.monsters = [plant]
        before_block = plant.block
        before_malleable = plant.power("Malleable")

        env.combat._deal_attack_damage(6, plant)

        self.assertGreater(plant.block, before_block)
        self.assertGreater(plant.power("Malleable"), before_malleable)

        env = NativeRunEnv(seed=2, ascension_level=0)
        parasite = make_monster("ShelledParasite", random.Random(1))
        parasite.block = 20
        parasite.move = "SHELLED_SUCK"
        env.combat.monsters = [parasite]
        env.combat.hand = []
        env.combat.player.current_hp = 80

        env.combat.end_turn()

        self.assertEqual(parasite.block, parasite.power("Plated Armor"))

    def test_added_events_have_resource_effects(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.phase = "EVENT"
        before_potions = sum(1 for potion in env.potions if potion.can_use)
        env.step({"kind": "event", "event_id": "Lab", "name": "Obtained Potions", "choice_index": 0})
        self.assertGreaterEqual(sum(1 for potion in env.potions if potion.can_use), before_potions)

        env = NativeRunEnv(seed=1, ascension_level=0)
        env.phase = "EVENT"
        env.deck = [make_card("Strike_R", uuid="offered")]
        env.player.current_hp = 1
        env.step({"kind": "event", "event_id": "Bonfire Spirits", "name": "Card Offered", "choice_index": 0})
        self.assertEqual(len(env.deck), 0)
        self.assertGreater(env.player.current_hp, 1)

    def test_event_pool_is_act_scoped(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.act = 1
        act1_pool = set(env._event_pool_for_act())
        self.assertIn("Golden Idol", act1_pool)
        self.assertNotIn("MindBloom", act1_pool)
        self.assertNotIn("Masked Bandits", act1_pool)

        env.act = 2
        act2_pool = set(env._event_pool_for_act())
        self.assertIn("Masked Bandits", act2_pool)
        self.assertIn("Colosseum", act2_pool)
        self.assertNotIn("MindBloom", act2_pool)

        env.act = 3
        act3_pool = set(env._event_pool_for_act())
        self.assertIn("MindBloom", act3_pool)
        self.assertIn("Falling", act3_pool)
        self.assertNotIn("Golden Idol", act3_pool)

    def test_question_room_outcome_updates_dynamic_chances(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.monster_chance = 0.0
        env.shop_chance = 0.0
        env.treasure_chance = 0.0

        outcome = env._question_room_outcome()

        self.assertEqual(outcome, "?")
        self.assertAlmostEqual(env.monster_chance, 0.10)
        self.assertAlmostEqual(env.shop_chance, 0.03)
        self.assertAlmostEqual(env.treasure_chance, 0.02)

    def test_event_draw_removes_from_pool_without_advancing_event_rng(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        before_counter = env.randoms.event.counter

        event_id = env._draw_event_id()

        self.assertNotIn(event_id, env.event_list + env.shrine_list + env.special_one_time_event_list)
        self.assertEqual(env.randoms.event.counter, before_counter)

    def test_mind_bloom_fight_enters_event_boss_combat(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.phase = "EVENT"

        env.step({"kind": "event", "event_id": "MindBloom", "name": "Fight", "choice_index": 0})

        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual(env.combat.floor, 16)
        self.assertTrue(env.combat.monsters)

    def test_masked_bandits_enters_combat_and_rewards_red_mask(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.phase = "EVENT"
        before_gold = env.gold

        env.step({"kind": "event", "event_id": "Masked Bandits", "name": "Fought Bandits", "choice_index": 0})

        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual([monster.monster_id for monster in env.combat.monsters], ["Bear", "Pointy", "Romeo"])
        self.assertEqual(env.current_node_symbol, "EVENT_COMBAT")

        env.combat.outcome = "PLAYER_VICTORY"
        env.step({"kind": "end", "name": "RESOLVE_COMBAT"})

        self.assertGreaterEqual(env.gold, before_gold + 222)
        self.assertTrue(any(relic.get("relic_id") == "Red Mask" for relic in env.relics))

    def test_high_risk_events_enter_native_event_combats(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.phase = "EVENT"
        env.step({"kind": "event", "event_id": "Mysterious Sphere", "name": "Fought Orb Walkers", "choice_index": 0})
        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual([monster.monster_id for monster in env.combat.monsters], ["OrbWalker", "OrbWalker"])

        env = NativeRunEnv(seed=2, ascension_level=0)
        env.phase = "EVENT"
        env.step({"kind": "event", "event_id": "Dead Adventurer", "name": "Searched", "choice_index": 0})
        self.assertEqual(env.phase, "COMBAT")
        self.assertTrue(env.combat.elite)
        self.assertIn(env.combat.monsters[0].monster_id, {"GremlinNob", "Lagavulin", "Sentry"})

        env = NativeRunEnv(seed=3, ascension_level=0)
        env.phase = "EVENT"
        env.step({"kind": "event", "event_id": "Colosseum", "name": "Fought", "choice_index": 0})
        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual([monster.monster_id for monster in env.combat.monsters], ["SlaverBlue", "SlaverRed", "Taskmaster"])

    def test_bear_and_romeo_apply_native_bandit_debuffs(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        bear = make_monster("Bear", random.Random(1))
        bear.move = "BEAR_BEAR_HUG"
        env.combat._monster_take_turn(bear)
        self.assertLess(env.combat.player.power("Dexterity"), 0)

        env = NativeRunEnv(seed=2, ascension_level=0)
        romeo = make_monster("Romeo", random.Random(1))
        romeo.move = "ROMEO_MOCK"
        env.combat._monster_take_turn(romeo)
        self.assertGreater(env.combat.player.power("Weakened"), 0)
        self.assertGreater(env.combat.player.power("Frail"), 0)

    def test_common_status_debuff_monster_moves_mutate_deck_state(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        snecko = make_monster("Snecko", random.Random(1))
        snecko.move = "SNECKO_TAIL_WHIP"
        env.combat._monster_take_turn(snecko)
        self.assertGreater(env.combat.player.power("Vulnerable"), 0)

        env = NativeRunEnv(seed=2, ascension_level=0)
        shelled = make_monster("ShelledParasite", random.Random(1))
        shelled.move = "SHELLED_SUCK"
        env.combat._monster_take_turn(shelled)
        self.assertGreater(env.combat.player.power("Frail"), 0)

        env = NativeRunEnv(seed=3, ascension_level=0)
        repulsor = make_monster("Repulsor", random.Random(1))
        repulsor.move = "REPULSOR_REPULSE"
        env.combat._monster_take_turn(repulsor)
        self.assertGreaterEqual(sum(card.card_id == "Dazed" for card in env.combat.discard_pile), 2)

        env = NativeRunEnv(seed=4, ascension_level=0)
        nemesis = make_monster("Nemesis", random.Random(1))
        nemesis.move = "NEMESIS_DEBUFF"
        env.combat._monster_take_turn(nemesis)
        self.assertGreaterEqual(sum(card.card_id == "Burn" for card in env.combat.discard_pile), 3)

    def test_time_eater_time_warp_forces_turn_end(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        time_eater = make_monster("TimeEater", random.Random(1))
        time_eater.powers["Time Warp"] = 11
        env.combat.monsters = [time_eater]
        env.combat.hand = [make_card("Strike_R", uuid="warp")]
        env.combat.player.energy = 3
        before_turn = env.combat.turn

        env.combat.play_card(0, 0)

        self.assertGreater(env.combat.turn, before_turn)
        self.assertEqual(time_eater.power("Time Warp"), 0)

    def test_auto_play_top_card_skips_unplayable_special_conditions(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.combat.draw_pile = [make_card("Clash", uuid="clash")]
        env.combat.hand = [make_card("Defend_R", uuid="defend")]

        env.combat._play_random_top_card()

        self.assertTrue(any(card.card_id == "Clash" for card in env.combat.discard_pile))

    def test_rampage_growth_and_sentinel_exhaust_energy(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        monster = MonsterState("Dummy", "Dummy", 100, 100, "WAIT", "UNKNOWN")
        rampage = make_card("Rampage", uuid="rampage")
        rampage.misc = 10
        env.combat.monsters = [monster]
        env.combat.hand = [rampage]
        env.combat.player.energy = 3

        env.combat.play_card(0, 0)

        self.assertEqual(monster.current_hp, 82)
        self.assertEqual(rampage.misc, 15)

        env = NativeRunEnv(seed=1, ascension_level=0)
        sentinel = make_card("Sentinel", uuid="sentinel")
        env.combat.hand = []
        env.combat.player.energy = 0
        env.combat._exhaust_card(sentinel)

        self.assertEqual(env.combat.player.energy, 2)

    def test_rupture_triggers_on_card_hp_loss_not_enemy_attack(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.combat.player.add_power("Rupture", 1)
        env.combat.monsters = [MonsterState("Dummy", "Dummy", 100, 100, "WAIT", "UNKNOWN")]
        env.combat.hand = [make_card("Hemokinesis", uuid="hemo")]
        env.combat.player.energy = 3

        env.combat.play_card(0, 0)

        self.assertEqual(env.combat.player.power("Strength"), 1)

        env = NativeRunEnv(seed=2, ascension_level=0)
        env.combat.player.add_power("Rupture", 1)
        attacker = MonsterState("Dummy", "Dummy", 100, 100, "HIT", "ATTACK", move_base_damage=5, move_hits=1)
        env.combat._monster_take_turn(attacker)

        self.assertEqual(env.combat.player.power("Strength"), 0)

    def test_chest_can_take_sapphire_key(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env._enter_chest()

        key_action = next(action for action in env.legal_actions() if action.get("item_kind") == "sapphire_key")
        env.step(key_action)

        self.assertIn("sapphire", env.keys)
        self.assertEqual(env.phase, "MAP")

    def test_campfire_can_recall_ruby_key(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env._enter_campfire()

        recall_action = next(action for action in env.legal_actions() if action.get("name") == "RECALL")
        env.step(recall_action)

        self.assertIn("ruby", env.keys)
        self.assertEqual(env.phase, "MAP")

    def test_first_elite_can_grant_emerald_key(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.phase = "MAP"
        env._advance_to_node("E")
        env.phase = "CARD_REWARD"

        env._enter_card_reward()

        self.assertIn("emerald", env.keys)

    def test_act4_unlocks_after_act3_boss_when_keys_are_complete(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.floor = 50
        env.phase = "CARD_REWARD"
        env.reward_cards = []
        env.keys.update({"ruby", "sapphire", "emerald"})

        env.step({"kind": "skip", "name": "SKIP", "choice_index": 0})

        self.assertEqual(env.phase, "MAP")
        self.assertEqual(env.legal_actions()[0]["symbol"], "R")

    def test_act3_boss_ends_run_without_keys(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.floor = 50
        env.phase = "CARD_REWARD"
        env.reward_cards = []

        env.step({"kind": "skip", "name": "SKIP", "choice_index": 0})

        self.assertEqual(env.phase, "COMPLETE")

    def test_a20_act3_boss_queues_second_boss(self):
        env = NativeRunEnv(seed=1, ascension_level=20)
        env.floor = 50
        env.act = 3
        env.act_boss = "Awakened One"
        env.phase = "COMBAT"
        env.current_node_symbol = "BOSS"
        env.combat.floor = 50
        env.combat.act = 3
        env.combat.act_boss = "Awakened One"
        env.combat.monsters = [MonsterState("TestBoss", "Test Boss", 1, 1, "TEST", "ATTACK", move_base_damage=1, move_hits=1)]
        env.combat.hand = [make_card("Strike_R", uuid="strike")]
        env.combat.player.energy = 3

        env.step({"kind": "card", "card_index": 0, "target_index": 0})

        self.assertEqual(env.phase, "COMBAT")
        self.assertTrue(env.a20_second_boss_done)
        self.assertEqual(env.floor, 50)
        self.assertNotEqual(env.act_boss, "Awakened One")

    def test_act4_route_reaches_heart_after_keys(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.floor = 50
        env.phase = "CARD_REWARD"
        env.reward_cards = []
        env.keys.update({"ruby", "sapphire", "emerald"})
        env.step({"kind": "skip", "name": "SKIP", "choice_index": 0})

        env.step({"kind": "map", "name": "R", "symbol": "R"})
        env.step({"kind": "campfire", "name": "REST"})
        env.step({"kind": "map", "name": "$", "symbol": "$"})
        env.step({"kind": "shop", "name": "LEAVE", "item_kind": "leave"})
        env.step({"kind": "map", "name": "ACT4_ELITE", "symbol": "ACT4_ELITE"})
        env.combat.outcome = "PLAYER_VICTORY"
        env.floor = 53
        env.phase = "COMBAT"
        env.step({"kind": "end", "name": "END_TURN"})
        env.step({"kind": "map", "name": "HEART", "symbol": "HEART"})

        self.assertEqual(env.phase, "COMBAT")
        self.assertEqual(env.floor, 54)
        self.assertEqual(env.combat.monsters[0].monster_id, "CorruptHeart")

    def test_all_native_potions_have_effect_paths(self):
        for potion_id in POTION_LIBRARY:
            with self.subTest(potion_id=potion_id):
                env = NativeRunEnv(seed=1, ascension_level=0)
                env.combat.potions[0] = make_potion(potion_id)
                env.combat.use_potion(0, 0)
                self.assertTrue(env.combat.outcome in {"UNDECIDED", "PLAYER_VICTORY", "PLAYER_LOSS"})

    def test_shop_has_full_card_relic_potion_layout(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env._enter_shop()

        kinds = [item.get("item_kind") for item in env.shop_items]

        self.assertEqual(kinds.count("card"), 7)
        self.assertEqual(kinds.count("relic"), 3)
        self.assertEqual(kinds.count("potion"), 3)
        self.assertEqual(kinds.count("purge"), 1)
        self.assertEqual(kinds.count("leave"), 1)

    def test_all_colorless_cards_have_effect_paths(self):
        for card_id in sorted(COLORLESS_CARD_IDS | {"Apparition", "Bite", "Ritual Dagger"}):
            with self.subTest(card_id=card_id):
                env = NativeRunEnv(seed=13, ascension_level=0)
                env.combat.hand = [make_card(card_id, uuid=f"colorless-{card_id}")]
                env.combat.player.energy = 99
                env.combat.play_card(0, 0)
                self.assertTrue(env.combat.outcome in {"UNDECIDED", "PLAYER_VICTORY", "PLAYER_LOSS"})

    def test_artifact_blocks_monster_debuff_and_sadistic_nature_triggers(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        monster = env.combat.monsters[0]
        monster.add_power("Artifact", 1)
        env.combat._apply_monster_power(monster, "Vulnerable", 2)
        self.assertEqual(monster.power("Vulnerable"), 0)
        self.assertEqual(monster.power("Artifact"), 0)

        hp_before = monster.current_hp
        env.combat.player.add_power("Sadistic Nature", 5)
        env.combat._apply_monster_power(monster, "Weakened", 1)
        self.assertEqual(monster.power("Weakened"), 1)
        self.assertLess(monster.current_hp, hp_before)

    def test_singing_bowl_and_matryoshka_have_reward_effects(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.relics.append(make_relic("Singing Bowl"))
        env.phase = "CARD_REWARD"
        env.reward_cards = [make_card("Inflame", uuid="reward")]
        before_max_hp = env.player.max_hp
        env.step({"kind": "skip", "name": "SKIP", "choice_index": 1})
        self.assertEqual(env.player.max_hp, before_max_hp + 2)

        env = NativeRunEnv(seed=2, ascension_level=0)
        env.relics.append(make_relic("Matryoshka", counter=2))
        env._enter_chest()
        relic_count = len(env.relics)
        env.step(next(action for action in env.legal_actions() if action.get("item_kind") == "relic"))
        self.assertGreaterEqual(len(env.relics), relic_count + 2)

    def test_summons_split_and_rebirth_special_monsters(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.combat.monsters = [make_monster("GremlinLeader", random.Random(1))]
        leader = env.combat.monsters[0]
        leader.move = "GREMLIN_LEADER_RALLY"
        env.combat._monster_take_turn(leader)
        self.assertGreater(len(env.combat.monsters), 1)

        env.combat.monsters = [make_monster("SlimeBoss", random.Random(2))]
        slime = env.combat.monsters[0]
        slime.current_hp = 80
        env.combat.player.energy = 99
        env.combat.hand = [make_card("Bludgeon", uuid="splitter")]
        env.combat.play_card(0, 0)
        self.assertTrue(any(monster.monster_id.endswith("Slime_L") for monster in env.combat.monsters))

        env.combat.monsters = [make_monster("AwakenedOne", random.Random(3))]
        awakened = env.combat.monsters[0]
        awakened.current_hp = 1
        env.combat.player.energy = 99
        env.combat.hand = [make_card("Strike_R", uuid="rebirth")]
        env.combat.play_card(0, 0)
        self.assertTrue(awakened.alive)
        self.assertEqual(awakened.power("Awakened Reborn"), 1)

    def test_guardian_time_eater_and_darkling_special_powers(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.combat.monsters = [make_monster("TheGuardian", random.Random(1))]
        guardian = env.combat.monsters[0]
        guardian.powers["Mode Shift"] = 1
        env.combat.player.energy = 99
        env.combat.hand = [make_card("Strike_R", uuid="guardian")]
        env.combat.play_card(0, 0)
        self.assertGreater(guardian.power("Sharp Hide"), 0)

        env.combat.monsters = [make_monster("TimeEater", random.Random(2))]
        eater = env.combat.monsters[0]
        env.combat.hand = [make_card("Anger", uuid=f"anger-{index}") for index in range(12)]
        env.combat.player.energy = 99
        for _ in range(12):
            env.combat.play_card(0, 0)
        self.assertEqual(eater.power("Time Warp"), 0)
        self.assertGreaterEqual(eater.power("Strength"), 2)

        env.combat.monsters = [make_monster("Darkling", random.Random(3)), make_monster("Darkling", random.Random(4))]
        first = env.combat.monsters[0]
        first.current_hp = 1
        env.combat.hand = [make_card("Strike_R", uuid="darkling")]
        env.combat.player.energy = 99
        env.combat.play_card(0, 0)
        self.assertTrue(first.half_dead)
        env.combat.end_turn()
        self.assertFalse(first.half_dead)
        self.assertGreater(first.current_hp, 0)

    def test_corrupt_heart_invincible_caps_hp_damage_per_turn(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.combat.monsters = [make_monster("CorruptHeart", random.Random(1))]
        heart = env.combat.monsters[0]
        heart.current_hp = 750
        env.combat._apply_damage_to_monster(500, heart)
        self.assertEqual(heart.current_hp, 450)
        env.combat._apply_damage_to_monster(500, heart)
        self.assertEqual(heart.current_hp, 450)
        env.combat.start_player_turn()
        env.combat._apply_damage_to_monster(1, heart)
        self.assertEqual(heart.current_hp, 449)

    def test_omamori_and_darkstone_handle_curse_gain(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.relics.append(make_relic("Omamori", counter=2))
        env.relics.append(make_relic("Darkstone Periapt"))
        deck_size = len(env.deck)
        max_hp = env.player.max_hp

        self.assertFalse(env._add_curse_to_deck("Injury", uuid="blocked-1"))
        self.assertFalse(env._add_curse_to_deck("Doubt", uuid="blocked-2"))
        self.assertEqual(len(env.deck), deck_size)
        self.assertEqual(env.player.max_hp, max_hp)
        self.assertTrue(env._add_curse_to_deck("Shame", uuid="real-curse"))
        self.assertEqual(len(env.deck), deck_size + 1)
        self.assertEqual(env.player.max_hp, max_hp + 6)

    def test_necronomicurse_returns_when_exhausted(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        curse = make_card("Necronomicurse", uuid="necro")

        env.combat._exhaust_card(curse)

        self.assertTrue(any(card.card_id == "Necronomicurse" for card in env.combat.hand))
        self.assertFalse(any(card.card_id == "Necronomicurse" for card in env.combat.exhaust_pile))

    def test_shop_resource_relics_apply_prices_heal_and_maw_bank_break(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.player.current_hp = 10
        env.relics.extend([
            make_relic("Meal Ticket"),
            make_relic("Membership Card"),
            make_relic("Smiling Mask"),
            make_relic("Maw Bank", counter=1),
        ])

        env._enter_shop()

        self.assertEqual(env.player.current_hp, 25)
        self.assertEqual(next(item for item in env.shop_items if item.get("item_kind") == "purge")["price"], 25)
        buyable = next(item for item in env.shop_items if item.get("item_kind") in {"card", "relic", "potion"} and env.gold >= item.get("price", 0))
        env.step(buyable)
        self.assertEqual(next(relic for relic in env.relics if relic.get("relic_id") == "Maw Bank")["counter"], 0)

        a16_env = NativeRunEnv(seed=1, ascension_level=16)
        a16_env._enter_shop()
        self.assertEqual(next(item for item in a16_env.shop_items if item.get("item_kind") == "purge")["price"], 75)

    def test_campfire_relic_actions_and_rest_effects(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.relics.extend([
            make_relic("Peace Pipe"),
            make_relic("Shovel"),
            make_relic("Girya", counter=0),
            make_relic("Dream Catcher"),
            make_relic("Regal Pillow"),
        ])
        env.player.current_hp = 20
        deck_size = len(env.deck)
        env._enter_campfire()
        option_names = {action["name"] for action in env.legal_actions()}

        self.assertTrue({"PURGE", "DIG", "LIFT"}.issubset(option_names))
        env.step(next(action for action in env.legal_actions() if action.get("name") == "REST"))
        self.assertGreater(env.player.current_hp, 20)
        self.assertEqual(len(env.deck), deck_size + 1)

        env._enter_campfire()
        env.step(next(action for action in env.legal_actions() if action.get("name") == "LIFT"))
        self.assertEqual(next(relic for relic in env.relics if relic.get("relic_id") == "Girya")["counter"], 1)

    def test_white_beast_and_bottled_cards_have_effects(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.relics.append(make_relic("White Beast Statue"))
        env.current_node_symbol = "M"
        env._enter_card_reward()
        self.assertTrue(any(potion.can_use for potion in env.potions))

        bottled = make_relic("Bottled Flame")
        bottled["card_id"] = "Bash"
        combat = env.combat.__class__(
            seed=2,
            player=PlayerState(),
            deck=[make_card("Defend_R", uuid="defend"), make_card("Bash", uuid="bash")],
            relics=[make_relic("Burning Blood"), bottled],
        )
        self.assertTrue(any(card.card_id == "Bash" for card in combat.hand))

    def test_tiny_chest_and_wing_boots_change_map_behavior(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        tiny = make_relic("Tiny Chest", counter=3)
        env.relics.append(tiny)
        env.phase = "MAP"
        env.floor = 1
        env._advance_to_node("?")
        self.assertEqual(env.phase, "CHEST")
        self.assertEqual(tiny["counter"], 0)

        env = NativeRunEnv(seed=2, ascension_level=0)
        env.relics.append(make_relic("Wing Boots", counter=3))
        env.combat.outcome = "PLAYER_VICTORY"
        env.step({"kind": "end", "name": "RESOLVE_COMBAT"})
        env.step({"kind": "skip", "name": "SKIP", "choice_index": len(env.reward_cards)})
        first_choice = next(action for action in env.legal_actions() if action["symbol"] == "M")
        env.step(first_choice)
        env.combat.outcome = "PLAYER_VICTORY"
        env.step({"kind": "end", "name": "RESOLVE_COMBAT"})
        env.step({"kind": "skip", "name": "SKIP", "choice_index": len(env.reward_cards)})
        all_next_floor_nodes = {
            node_id for node_id in env.map_layers[env.floor + 1]
            if env.map_graph.get(node_id, {}).get("symbol")
        }
        offered_nodes = {action["node_id"] for action in env.legal_actions()}
        self.assertEqual(offered_nodes, all_next_floor_nodes)

    def test_shop_and_ironclad_relic_effects_are_not_noops(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.relics.append(make_relic("Medical Kit"))
        env.combat.relics = env.relics
        env.combat.hand = [make_card("Dazed", uuid="dazed")]
        env.combat.player.energy = 0
        self.assertTrue(env.combat.playable(env.combat.hand[0]))
        env.combat.play_card(0, 0)
        self.assertEqual(env.combat.exhaust_pile[-1].card_id, "Dazed")

        env = NativeRunEnv(seed=2, ascension_level=0)
        env.combat.relics.append(make_relic("Chemical X"))
        env.combat.monsters = [MonsterState("Dummy", "Dummy", 100, 100, "WAIT", "UNKNOWN")]
        env.combat.hand = [make_card("Whirlwind", uuid="ww")]
        env.combat.player.energy = 1
        env.combat.play_card(0, 0)
        self.assertLessEqual(env.combat.monsters[0].current_hp, 85)

        env = NativeRunEnv(seed=3, ascension_level=0)
        env.combat.relics.append(make_relic("Orange Pellets"))
        env.combat.player.powers.update({"Vulnerable": 2, "Weakened": 1})
        env.combat.hand = [make_card("Anger", uuid="attack"), make_card("Flex", uuid="skill"), make_card("Inflame", uuid="power")]
        env.combat.player.energy = 99
        env.combat.play_card(0, 0)
        env.combat.play_card(0, 0)
        env.combat.play_card(0, 0)
        self.assertEqual(env.combat.player.power("Vulnerable"), 0)
        self.assertEqual(env.combat.player.power("Weakened"), 0)

        env = NativeRunEnv(seed=4, ascension_level=0)
        env.relics.append(make_relic("Magic Flower"))
        env.player.current_hp = 10
        env.combat.relics = env.relics
        env.combat.player = env.player
        env.combat._heal(10)
        self.assertEqual(env.player.current_hp, 25)

    def test_on_obtain_shop_relics_modify_run_state(self):
        env = NativeRunEnv(seed=1, ascension_level=0)
        env.relics.append(make_relic("Ceramic Fish"))
        gold = env.gold
        env._add_card_to_deck("Pommel Strike", uuid="ceramic")
        self.assertEqual(env.gold, gold + 9)

        env = NativeRunEnv(seed=2, ascension_level=0)
        env.player.current_hp = 1
        env._obtain_relic(make_relic("Lee's Waffle"))
        self.assertEqual(env.player.current_hp, env.player.max_hp)

        env = NativeRunEnv(seed=3, ascension_level=0)
        deck_size = len(env.deck)
        env._obtain_relic(make_relic("Dolly's Mirror"))
        self.assertEqual(len(env.deck), deck_size + 1)

        env = NativeRunEnv(seed=4, ascension_level=0)
        deck_size = len(env.deck)
        env._obtain_relic(make_relic("Orrery"))
        self.assertEqual(len(env.deck), deck_size + 5)

        env = NativeRunEnv(seed=5, ascension_level=0)
        env.potions = [make_potion("Fire Potion"), make_potion("Weak Potion"), make_potion("Strength Potion")]
        env._obtain_relic(make_relic("Potion Belt"))
        env._obtain_relic(make_relic("Cauldron"))
        self.assertTrue(any(potion.can_use for potion in env.potions[3:]))


if __name__ == "__main__":
    unittest.main()
