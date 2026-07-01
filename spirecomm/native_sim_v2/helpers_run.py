from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from spirecomm.native_sim_v2.helpers_cards import CARD_LIBRARY, COLORLESS_CARD_ID_ORDER, COLORLESS_CARD_IDS, card_to_spirecomm, clone_card, ironclad_card_pool, ironclad_locked_card_ids, ironclad_type_rarity_card_pool, make_card, roll_colorless_card, starter_deck
from spirecomm.native_sim.mapgen import MAP_HEIGHT, MAP_WIDTH, generate_act_map as generate_native_act_map
from spirecomm.native_sim.potions import empty_potion_slots, get_random_potion, make_potion, potions_to_spirecomm, roll_potion
from spirecomm.native_sim.randoms import NativeRandomStreams, StsRandom, java_collections_shuffle
from spirecomm.native_sim_v2.helpers_relics import draw_relic_from_pool, init_ironclad_relic_pools, ironclad_locked_relic_ids, make_relic
from spirecomm.native_sim.schema import CardInstance, MonsterState, PlayerState, PotionInstance
from spirecomm.native_sim_v2.helpers_common import *
from spirecomm.native_sim_v2.monsters import encounter_to_monsters, generate_elite_schedule, generate_monster_schedules, generate_strong_monster_schedule, make_monster, roll_act1_encounter


class RunHelpersMixin:
    def _generate_act_map_from_lightspeed(self, act: int) -> tuple[dict[str, dict[str, Any]], dict[int, list[str]]]:
            import slaythespire as sts

            start_floor, boss_floor = self._act_floor_range(act)
            map_obj = sts.SpireMap(self.seed, self.ascension_level, act, self.enable_act4_keys and "emerald" not in self.keys)

            room_symbol = {
                "SHOP": "$",
                "REST": "R",
                "EVENT": "?",
                "ELITE": "E",
                "MONSTER": "M",
                "TREASURE": "T",
                "BOSS": "BOSS",
                "BOSS_TREASURE": "BOSS",
                "NONE": "",
                "INVALID": "",
            }

            graph: dict[str, dict[str, Any]] = {}
            layers: dict[int, list[str]] = {}
            parent_x_by_node: dict[str, list[int]] = {}

            for row in range(MAP_HEIGHT):
                floor = start_floor + row
                layers[floor] = []
                for x in range(MAP_WIDTH):
                    room = str(map_obj.get_room_type(x, row)).split(".")[-1]
                    symbol = room_symbol.get(room, "")
                    if not symbol:
                        continue
                    children_x = [
                        child_x
                        for child_x in range(MAP_WIDTH)
                        if map_obj.has_edge(x, row, child_x)
                    ]
                    node_id = f"a{act}-r{row}-x{x}"
                    graph[node_id] = {
                        "id": node_id,
                        "act": act,
                        "floor": floor,
                        "x": x,
                        "row": row,
                        "symbol": symbol,
                        "children_x": children_x,
                        "parents_x": [],
                        "children": [],
                    }
                    if children_x or row == MAP_HEIGHT - 1:
                        layers[floor].append(node_id)

            for row in range(MAP_HEIGHT - 1):
                next_floor = start_floor + row + 1
                next_by_x = {graph[node_id]["x"]: node_id for node_id in layers.get(next_floor, [])}
                for node_id in layers.get(start_floor + row, []):
                    children = [
                        next_by_x[child_x]
                        for child_x in graph[node_id].get("children_x", [])
                        if child_x in next_by_x
                    ]
                    graph[node_id]["children"] = children
                    for child_id in children:
                        parent_x_by_node.setdefault(child_id, []).append(int(graph[node_id]["x"]))

            boss_id = f"a{act}-boss"
            graph[boss_id] = {
                "id": boss_id,
                "act": act,
                "floor": boss_floor,
                "x": 3,
                "row": MAP_HEIGHT,
                "symbol": "BOSS",
                "children": [],
                "children_x": [],
                "parents_x": [],
            }
            layers[boss_floor] = [boss_id]
            for node_id in layers.get(start_floor + MAP_HEIGHT - 1, []):
                if 3 in graph[node_id].get("children_x", []):
                    graph[node_id]["children"] = [boss_id]
                    graph[boss_id]["parents_x"].append(int(graph[node_id]["x"]))

            for node_id, parents_x in parent_x_by_node.items():
                if node_id in graph:
                    graph[node_id]["parents_x"] = parents_x

            if self.enable_act4_keys and "emerald" not in self.keys:
                fallback_graph, _ = generate_native_act_map(
                    seed=self.seed,
                    ascension_level=self.ascension_level,
                    act=act,
                    start_floor=start_floor,
                    set_burning=True,
                )
                fallback_elites = sorted(
                    (node for node in fallback_graph.values() if node.get("symbol") in {"E", "E_GREEN"}),
                    key=lambda node: (int(node.get("floor", 0)), int(node.get("x", 0))),
                )
                graph_elites = sorted(
                    (node for node in graph.values() if node.get("symbol") == "E"),
                    key=lambda node: (int(node.get("floor", 0)), int(node.get("x", 0))),
                )
                burning_index = next(
                    (index for index, node in enumerate(fallback_elites) if node.get("symbol") == "E_GREEN"),
                    None,
                )
                if burning_index is not None and burning_index < len(graph_elites):
                    burning_node = graph_elites[burning_index]
                    burning_node["symbol"] = "E_GREEN"
                    fallback_burning = fallback_elites[burning_index]
                    if "burning_elite_buff" in fallback_burning:
                        burning_node["burning_elite_buff"] = fallback_burning["burning_elite_buff"]

            return graph, layers

    def _add_card_to_deck(self, card_id: str, *, upgrades: int = 0, uuid: str = "") -> CardInstance:
            base_uuid = uuid or f"deck-{self.floor}-{card_id}"
            candidate_uuid = base_uuid
            existing_uuids = {card.uuid for card in self.deck}
            suffix = 1
            while candidate_uuid in existing_uuids:
                candidate_uuid = f"{base_uuid}-{suffix}"
                suffix += 1
            card = self._make_deck_card(card_id, upgrades=upgrades, uuid=candidate_uuid)
            self.deck.append(card)
            if self._has_relic("Ceramic Fish"):
                self._gain_gold(9)
            return card

    def _add_colorless_cards_to_deck(self, count: int, *, prefix: str) -> None:
            for index in range(max(0, int(count))):
                reward = roll_colorless_card(self.randoms.card)
                self._add_card_to_deck(reward.card_id, upgrades=reward.upgrades, uuid=f"{prefix}-{self.floor}-{index}")

    def _add_curse_to_deck(self, card_id: str | None = None, *, uuid: str | None = None) -> bool:
            card_id = card_id or self._random_curse_id()
            omamori = self._relic("Omamori")
            if omamori is not None:
                counter = int(omamori.get("counter", -1))
                counter = 2 if counter < 0 else counter
                if counter > 0:
                    omamori["counter"] = counter - 1
                    return False
            self._add_card_to_deck(card_id, uuid=uuid or f"curse-{self.floor}-{card_id}")
            return True

    def _add_potion_if_space(self, potion: PotionInstance) -> bool:
            if any(relic.get("relic_id") == "Sozu" for relic in self.relics):
                return False
            for index, current in enumerate(self.potions):
                if not current.can_use:
                    self.potions[index] = potion
                    return True
            return False

    def _add_random_potion_reward(self, *, count: int = 1) -> int:
            added = 0
            for _ in range(max(0, int(count))):
                if self._add_potion_if_space(roll_potion(self.randoms.potion)):
                    added += 1
            return added

    def _apply_burning_elite_buff(self) -> None:
            if self.current_node_symbol != "E_GREEN" or not self.current_map_node_id or self.combat is None:
                return
            node = self.map_graph.get(self.current_map_node_id, {})
            buff_type = node.get("burning_elite_buff")
            if buff_type is None:
                return
            for monster in self.combat.monsters:
                if buff_type == 0:
                    monster.add_power("Strength", self.act + 1)
                elif buff_type == 1:
                    increase = _sts_round(float(monster.max_hp) * 0.25)
                    monster.max_hp += increase
                    monster.current_hp += increase
                    if self._has_relic("Preserved Insect"):
                        # Lightspeed applies the green-elite HP buff before
                        # Preserved Insect trims opening elite HP.
                        monster.current_hp = max(1, int(monster.max_hp * 0.75))
                elif buff_type == 2:
                    monster.add_power("Metallicize", self.act * 2 + 2)
                elif buff_type == 3:
                    monster.add_power("Regenerate", self.act * 2 + 1)

    def _apply_neow_bonus(self, bonus: str) -> dict[str, Any]:
            if bonus == "THREE_CARDS":
                self._open_neow_card_reward(rare_only=False, colorless=False)
                return self.state()
            if bonus == "ONE_RANDOM_RARE_CARD":
                chosen = self._random_class_card_of_rarity_from_rng(self.randoms.neow, "RARE")
                self._add_card_to_deck(chosen.card_id, uuid=f"neow-rare-{chosen.card_id}")
                return self._complete_neow()
            if bonus == "REMOVE_CARD":
                self._open_neow_card_select("NEOW_REMOVE", 1)
                return self.state()
            if bonus == "UPGRADE_CARD":
                self._open_neow_card_select("NEOW_UPGRADE", 1)
                return self.state()
            if bonus == "TRANSFORM_CARD":
                self._open_neow_card_select("NEOW_TRANSFORM", 1)
                return self.state()
            if bonus == "RANDOM_COLORLESS":
                self._open_neow_card_reward(rare_only=False, colorless=True)
                return self.state()
            if bonus == "THREE_SMALL_POTIONS":
                self._add_random_potion_reward(count=3)
                return self._complete_neow()
            if bonus == "RANDOM_COMMON_RELIC":
                self._obtain_relic(self._roll_relic_of_tier("COMMON"))
                return self._complete_neow()
            if bonus == "TEN_PERCENT_HP_BONUS":
                hp_gain = int(self.player.max_hp * 0.1)
                self.player.max_hp += hp_gain
                self.player.current_hp += hp_gain
                return self._complete_neow()
            if bonus == "THREE_ENEMY_KILL":
                self._obtain_relic(make_relic("Neow's Lament"))
                return self._complete_neow()
            if bonus == "HUNDRED_GOLD":
                self._gain_gold(100)
                return self._complete_neow()
            if bonus == "RANDOM_COLORLESS_2":
                self._open_neow_card_reward(rare_only=True, colorless=True)
                return self.state()
            if bonus == "REMOVE_TWO":
                self._open_neow_card_select("NEOW_REMOVE", 2)
                return self.state()
            if bonus == "ONE_RARE_RELIC":
                self._obtain_relic(self._roll_relic_of_tier("RARE"))
                return self._complete_neow()
            if bonus == "THREE_RARE_CARDS":
                self._open_neow_card_reward(rare_only=True, colorless=False)
                return self.state()
            if bonus == "TWO_FIFTY_GOLD":
                self._gain_gold(250)
                return self._complete_neow()
            if bonus == "TRANSFORM_TWO_CARDS":
                self._open_neow_card_select("NEOW_TRANSFORM", 2)
                return self.state()
            if bonus == "TWENTY_PERCENT_HP_BONUS":
                hp_gain = int(self.player.max_hp * 0.2)
                self.player.max_hp += hp_gain
                self.player.current_hp += hp_gain
                return self._complete_neow()
            if bonus == "BOSS_RELIC":
                self.relics = [relic for relic in self.relics if relic.get("relic_id") != "Burning Blood"]
                self._obtain_relic(self._roll_boss_relics(count=1)[0])
                return self._complete_neow()
            return self._complete_neow()

    def _apply_neow_drawback(self, drawback: str) -> None:
            if drawback == "TEN_PERCENT_HP_LOSS":
                self.player.max_hp = max(1, int(self.player.max_hp * 0.9))
                self.player.current_hp = min(self.player.current_hp, self.player.max_hp)
            elif drawback == "NO_GOLD":
                self.gold = 0
            elif drawback == "PERCENT_DAMAGE":
                self.player.current_hp = max(1, (self.player.current_hp // 10) * 7)
            elif drawback == "LOSE_STARTER_RELIC":
                self.relics = [relic for relic in self.relics if relic.get("relic_id") != "Burning Blood"]

    def _apply_shop_discount(self, price: int, *, include_ascension: bool = True) -> int:
            discounted = int(price)
            if include_ascension and self.ascension_level >= 16:
                discounted = max(0, round(discounted * 0.8))
            if self._has_relic("The Courier"):
                discounted = max(0, round(discounted * 0.8))
            if self._has_relic("Membership Card"):
                discounted = max(0, round(discounted * 0.5))
            return discounted

    def _bottled_deck_indexes(self) -> set[int]:
            bottled_uuids = {
                str(relic.get("card_uuid"))
                for relic in self.relics
                if relic.get("relic_id") in {"Bottled Flame", "Bottled Lightning", "Bottled Tornado"} and relic.get("card_uuid")
            }
            if bottled_uuids:
                return {index for index, card in enumerate(self.deck) if card.uuid in bottled_uuids}
            bottled_ids = {
                str(relic.get("card_id"))
                for relic in self.relics
                if relic.get("relic_id") in {"Bottled Flame", "Bottled Lightning", "Bottled Tornado"} and relic.get("card_id")
            }
            if not bottled_ids:
                return set()
            return {index for index, card in enumerate(self.deck) if card.card_id in bottled_ids}

    def _roll_falling_choice_indexes(self) -> dict[str, int]:
            bottled_indexes = self._bottled_deck_indexes()
            choices: dict[str, int] = {}
            for card_type, key in (("ATTACK", "attack"), ("SKILL", "skill"), ("POWER", "power")):
                eligible = [
                    index
                    for index, card in enumerate(self.deck)
                    if card.card_def.card_type == card_type and index not in bottled_indexes
                ]
                if not eligible:
                    choices[key] = -1
                    continue
                selected_idx = int(self.randoms.misc.random(len(eligible) - 1))
                choices[key] = eligible[selected_idx]
            return choices

    def _nloth_relic_choices(self) -> list[tuple[int, dict[str, Any]]]:
            choices: list[tuple[int, dict[str, Any]]] = []
            for index, relic in enumerate(self.relics):
                relic_id = str(relic.get("relic_id") or "")
                if relic_id in {"N'loth's Gift"}:
                    continue
                choices.append((index, relic))
            return choices

    def _roll_nloth_relic_choices(self) -> list[int]:
            choices = self._nloth_relic_choices()
            choice_indexes = list(range(len(choices)))
            java_collections_shuffle(choice_indexes, self.randoms.misc.random_long())
            selected = choice_indexes[: min(2, len(choice_indexes))]
            return [choices[idx][0] for idx in selected]

    def _can_add_event(self, event_id: str) -> bool:
            if event_id in {"Dead Adventurer", "Hypnotizing Colored Mushrooms"}:
                return self.floor > 6
            if event_id == "The Moai Head":
                return self.player.current_hp <= self.player.max_hp // 2 or self._has_relic("Golden Idol")
            if event_id == "The Cleric":
                return self.gold >= 35
            if event_id == "Old Beggar":
                return self.gold >= 75
            if event_id == "Colosseum":
                return False
            return True

    def _can_add_one_time_event(self, event_id: str) -> bool:
            if event_id == "The Divine Fountain":
                return any(
                    card.card_def.card_type == "CURSE" and _card_can_transform(card)
                    for card in self.deck
                )
            if event_id == "Designer In-Spire":
                return self.act in {2, 3} and self.gold >= 75
            if event_id == "Duplicator":
                return self.act in {2, 3}
            if event_id == "Face Trader":
                return self.act in {1, 2}
            if event_id == "Knowing Skull":
                return self.act == 2 and self.player.current_hp > 12
            if event_id == "N'loth":
                return self.act == 2 and len(self.relics) >= 2
            if event_id == "The Joust":
                return self.act == 2 and self.gold >= 50
            if event_id == "The Woman in Blue":
                return self.gold >= 50
            if event_id == "Secret Portal":
                return self.act == 3
            return True

    def _complete_neow(self) -> dict[str, Any]:
            self.neow_options = []
            self.floor = 0
            self.phase = "MAP"
            self._enter_map()
            return self.state()

    def _consume_match_and_keep_rng(self) -> list[CardInstance]:
            cards = [
                self._random_class_card_of_rarity("RARE"),
                self._random_class_card_of_rarity("UNCOMMON"),
                self._random_class_card_of_rarity("COMMON"),
            ]
            # returnColorlessCard(UNCOMMON) shuffles the colorless pool using the
            # shuffle RNG, not cardRng.
            self.randoms.shuffle.random_long()
            colorless_pool = [
                CARD_LIBRARY[card_id] for card_id in COLORLESS_CARD_ID_ORDER
                if CARD_LIBRARY[card_id].rarity == "UNCOMMON"
            ]
            if colorless_pool:
                cards.append(make_card(colorless_pool[0].card_id, uuid=f"match-colorless-{self.floor}"))
            curse_pool = [
                card_id for card_id, card_def in CARD_LIBRARY.items()
                if card_def.card_type == "CURSE" and card_id not in {"AscendersBane", "CurseOfTheBell"}
            ]
            if curse_pool:
                self.randoms.card.random(len(curse_pool) - 1)
            cards.append(make_card("Strike_R", uuid=f"match-starter-{self.floor}"))
            self.randoms.misc.random_long()
            return cards

    def _current_map_row(self) -> int:
            if not self.current_map_node_id:
                return 0
            return int(self.map_graph.get(self.current_map_node_id, {}).get("row", 0))

    def _draw_event_id(self) -> str:
            if not (self.event_list or self.shrine_list or self.special_one_time_event_list):
                self._reset_act_random_room_state()
            # StS/lightspeed select the concrete event from a copy of eventRng:
            # question-room outcome consumes eventRng, but event selection itself
            # only mutates event pools.
            event_rng = self.randoms.event.copy()
            use_shrine = event_rng.random(1.0) < 0.25
            available_shrines = list(self.shrine_list) + [
                event_id for event_id in self.special_one_time_event_list
                if self._can_add_one_time_event(event_id)
            ]
            available_events = [event_id for event_id in self.event_list if self._can_add_event(event_id)]
            if use_shrine:
                if not available_shrines and available_events:
                    event_id = event_rng.choice(available_events)
                    self.event_list.remove(event_id)
                    return event_id
                combined = available_shrines or available_events
                event_id = event_rng.choice(combined)
                if event_id in self.shrine_list:
                    self.shrine_list.remove(event_id)
                elif event_id in self.special_one_time_event_list:
                    self.special_one_time_event_list.remove(event_id)
                return event_id
            pool = available_events or available_shrines
            event_id = event_rng.choice(pool)
            if event_id in self.event_list:
                self.event_list.remove(event_id)
            elif event_id in self.shrine_list:
                self.shrine_list.remove(event_id)
            elif event_id in self.special_one_time_event_list:
                self.special_one_time_event_list.remove(event_id)
            return event_id

    def _enter_boss_relic(self) -> None:
            self.phase = "BOSS_RELIC"
            self.boss_relic_options = self._roll_boss_relics(count=3)
            for index, relic in enumerate(self.boss_relic_options):
                relic["kind"] = "boss_relic"
                relic["choice_index"] = index

    def _enter_boss_treasure_room(self) -> None:
            self.floor += 1
            maw_bank = self._relic("Maw Bank")
            if maw_bank is not None and int(maw_bank.get("counter", -1)) != 0:
                self._gain_gold(12)
            floor_rng = StsRandom(self.seed + self.floor)
            self.randoms.misc = floor_rng.copy()
            self.randoms.shuffle = floor_rng.copy()
            self.randoms.card_random = floor_rng.copy()
            self.rng = self.randoms.misc
            self._enter_boss_relic()

    def _enter_campfire(self) -> None:
            self.phase = "CAMPFIRE"
            if self._has_relic("Eternal Feather"):
                self._heal_run(3 * (len(self.deck) // 5))
            self.campfire_options = []
            if not any(relic.get("relic_id") == "Coffee Dripper" for relic in self.relics):
                self.campfire_options.append({"kind": "campfire", "name": "REST", "choice_index": len(self.campfire_options)})
            if not any(relic.get("relic_id") == "Fusion Hammer" for relic in self.relics):
                self.campfire_options.append({"kind": "campfire", "name": "SMITH", "choice_index": len(self.campfire_options)})
            if self._has_relic("Peace Pipe"):
                self.campfire_options.append({"kind": "campfire", "name": "TOKE", "choice_index": len(self.campfire_options)})
            if self._has_relic("Shovel"):
                self.campfire_options.append({"kind": "campfire", "name": "DIG", "choice_index": len(self.campfire_options)})
            girya = self._relic("Girya")
            if girya is not None and int(girya.get("counter", 0)) < 3:
                self.campfire_options.append({"kind": "campfire", "name": "LIFT", "choice_index": len(self.campfire_options)})
            if self.enable_act4_keys and "ruby" not in self.keys:
                self.campfire_options.append({"kind": "campfire", "name": "RECALL", "choice_index": len(self.campfire_options)})
            if not self.campfire_options:
                self.campfire_options.append({"kind": "campfire", "name": "LEAVE", "choice_index": 0})

    def _enter_card_reward(
            self,
            *,
            extra_gold_rewards: list[int] | None = None,
            include_base_gold: bool = True,
            allow_meat_on_the_bone: bool = True,
        ) -> None:
            self.phase = "CARD_REWARD"
            self.reward_close_required = False
            self.reward_card_bundles = []
            self.reward_gold = 0
            self.reward_gold_piles = []
            self.reward_emerald_key = False
            self.reward_relics = []
            self.reward_potions = []
            reward_count = 3 + (1 if any(relic.get("relic_id") == "Question Card" for relic in self.relics) else 0)
            reward_bundle_count = 2 if self.current_node_symbol == "M" and any(relic.get("relic_id") == "Prayer Wheel" for relic in self.relics) else 1
            if any(relic.get("relic_id") == "Busted Crown" for relic in self.relics):
                reward_count = max(1, reward_count - 2)
            for _ in range(reward_bundle_count):
                self.reward_card_bundles.append(self._roll_card_reward(count=reward_count))
            self._refresh_reward_cards()
            if include_base_gold:
                if self.current_node_symbol in {"E", "E_GREEN", "ACT4_ELITE"}:
                    gold_gain = self.randoms.treasure.randint(25, 35)
                elif self.current_node_symbol == "BOSS":
                    gold_gain = 100 + self.randoms.misc.randint(-5, 5)
                    if self.ascension_level >= 13:
                        gold_gain = int(round(gold_gain * 0.75))
                else:
                    gold_gain = self.randoms.treasure.randint(10, 20)
                if any(relic.get("relic_id") == "Golden Idol" for relic in self.relics):
                    gold_gain += _sts_round(float(gold_gain) * 0.25)
                self.reward_gold_piles.append(gold_gain)
                self.reward_gold += gold_gain
            if extra_gold_rewards:
                for amount in extra_gold_rewards:
                    amount = int(amount)
                    if amount > 0:
                        self.reward_gold_piles.append(amount)
                        self.reward_gold += amount
            if self.current_node_symbol in {"E", "E_GREEN", "ACT4_ELITE"}:
                self.reward_relics.append(self._roll_relic(elite=True))
                if any(relic.get("relic_id") == "Black Star" for relic in self.relics):
                    self.reward_relics.append(self._roll_relic(elite=True))
                if self.enable_act4_keys and self.current_node_symbol == "E_GREEN" and "emerald" not in self.keys:
                    self.reward_emerald_key = True
            self.player.block = 0
            self.player.energy = 0
            self.player.powers = {}
            has_meat_on_the_bone = any(relic.get("relic_id") == "Meat on the Bone" for relic in self.relics)
            meat_on_the_bone_active = (
                allow_meat_on_the_bone
                and has_meat_on_the_bone
                and self.player.current_hp <= self.player.max_hp // 2
            )
            if self._has_relic("Black Blood"):
                self._heal_run(12)
            elif self._has_relic("Burning Blood"):
                self._heal_run(6)
                meat_on_the_bone_active = (
                    allow_meat_on_the_bone
                    and has_meat_on_the_bone
                    and self.player.current_hp <= self.player.max_hp // 2
                )
            if meat_on_the_bone_active:
                self._heal_run(12)
            if any(relic.get("relic_id") == "Face Of Cleric" for relic in self.relics):
                self.player.max_hp += 1
                self.player.current_hp += 1
            chance = 100 if self._has_relic("White Beast Statue") else 40 + self.potion_chance_counter
            rewards_size = (
                len(self.reward_card_bundles)
                + len(self.reward_relics)
                + len(self.reward_potions)
                + len(self.reward_gold_piles)
            )
            # Real game reward screens stop adding potion rewards once the
            # standard gold/card/relic trio is already present.
            if rewards_size >= 3:
                chance = 0
            if not self._has_relic("Sozu"):
                if int(self.randoms.potion.random(99)) >= chance:
                    self.potion_chance_counter += 10
                else:
                    self.reward_potions.append(roll_potion(self.randoms.potion))
                    self.potion_chance_counter -= 10

    def _enter_chest(self) -> None:
            self.phase = "CHEST"
            gold_amounts = {"SMALL": 25, "MEDIUM": 50, "LARGE": 75}
            if self.chest_have_gold:
                base_gold = gold_amounts.get(self.chest_size, 25)
                self.chest_gold_amount = int(round(self.randoms.treasure.random(base_gold * 0.9, base_gold * 1.1)))
            else:
                self.chest_gold_amount = 0

            relic = self._roll_relic_of_tier(self.chest_relic_tier)
            relic["kind"] = "chest"
            relic["name"] = relic.get("name", "Relic")
            relic["item_kind"] = "relic"
            relic["choice_index"] = 0
            relic["gold_amount"] = self.chest_gold_amount
            relic["chest_size"] = self.chest_size
            self.chest_options = [relic]
            if self.enable_act4_keys and "sapphire" not in self.keys:
                self.chest_options.append({
                    "kind": "chest",
                    "name": "SAPPHIRE_KEY",
                    "item_kind": "sapphire_key",
                    "choice_index": 1,
                    "gold_amount": self.chest_gold_amount,
                    "chest_size": self.chest_size,
                })

    def _enter_event(self) -> None:
            self.phase = "EVENT"
            event_id = self._draw_event_id()
            self.current_event_id = event_id
            self.event_state = {}
            if event_id == "Bonfire Spirits":
                self._open_card_select("BONFIRE_SPIRITS", 1)
                return
            if event_id == "Lab":
                self._open_potion_reward_screen(count=2 if self.ascension_level >= 15 else 3, context="EVENT")
                return
            if event_id == "Scrap Ooze":
                self.event_state["counter"] = 0
            elif event_id == "Dead Adventurer":
                rewards = [0, 1, 2]
                java_collections_shuffle(rewards, self.randoms.misc.random_long())
                self.event_state["phase"] = 0
                self.event_state["rewards"] = rewards
                self.event_state["encounter"] = self.randoms.misc.choice(["Three Sentries", "Gremlin Nob", "Lagavulin"])
            elif event_id == "Cursed Tome":
                self.event_state["phase"] = 0
            elif event_id == "Knowing Skull":
                self.event_state["hp_amount_0"] = 6
                self.event_state["hp_amount_1"] = 6
                self.event_state["hp_amount_2"] = 6
            elif event_id in {"We Meet Again!", "WeMeetAgain"}:
                potion_indexes = [idx for idx, potion in enumerate(self.potions) if potion.can_use]
                if potion_indexes:
                    java_collections_shuffle(potion_indexes, self.randoms.misc.next_long())
                    self.event_state["potion_idx"] = potion_indexes[0]
                else:
                    self.event_state["potion_idx"] = None
                self.event_state["gold"] = -1 if self.gold < 50 else int(self.randoms.misc.random(50, min(150, self.gold)))
                card_indexes = [
                    idx
                    for idx, card in enumerate(self.deck)
                    if card.card_def.rarity != "BASIC" and card.card_def.card_type != "CURSE"
                ]
                if card_indexes:
                    java_collections_shuffle(card_indexes, self.randoms.misc.random_long())
                    selected_idx = card_indexes[0]
                    for idx, card in enumerate(self.deck[:selected_idx]):
                        if card.card_id == self.deck[selected_idx].card_id:
                            selected_idx = idx
                            break
                    self.event_state["card_idx"] = selected_idx
                else:
                    self.event_state["card_idx"] = None
            elif event_id == "World of Goop":
                low, high = (35, 75) if self.ascension_level >= 15 else (20, 50)
                self.event_state["gold_loss"] = min(self.gold, int(self.randoms.misc.random(low, high)))
            elif event_id == "N'loth":
                self.event_state["relic_indexes"] = self._roll_nloth_relic_choices()
            elif event_id == "Designer In-Spire":
                self.event_state["designer_upgrade_one"] = bool(self.randoms.misc.random_boolean())
                self.event_state["designer_cleanup_is_remove"] = bool(self.randoms.misc.random_boolean())
            elif event_id == "Falling":
                self.event_state.update(self._roll_falling_choice_indexes())
            option_labels = {
                "Big Fish": ["Banana", "Donut", "Box"],
                "Golden Idol": ["Take", "Ignored"],
                "Golden Shrine": ["Prayed", "Desecrated", "Ignored"],
                "Shining Light": ["Entered Light", "Ignored"],
                "The Cleric": ["Healed", "Card Removal", "Leave"],
                "Living Wall": ["Forget", "Change", "Grow"],
                "World of Goop": ["Gather Gold", "Left Gold"],
                "Scrap Ooze": ["Success", "Fled"],
                "The Ssssserpent": ["Agreed", "Ignored"],
                "Hypnotizing Colored Mushrooms": ["Fought Mushrooms", "Ignored"],
                "Mindbloom": ["Fight", "Gold", "Heal", "Upgrade"],
                "Falling": ["Removed Attack", "Removed Skill", "Removed Power"],
                "Winding Halls": ["Embrace Madness", "Max HP", "Writhe"],
                "Wing Statue": ["Card Removal", "Gained Gold", "Ignored"],
                "Drug Dealer": ["Got JAX", "Inject Mutagens", "Ignored"],
                "Ancient Writing": ["Elegance", "Simplicity"],
            "Old Beggar": ["Gave Gold", "Ignored"],
            "Cursed Tome": ["Read", "Ignored"],
            "Augmenter": ["JAX", "Transform", "Mutagenic Strength"],
            "The Nest": ["Stole From Cult", "Stay in Line"],
            "The Library": ["Read", "Heal"],
            "Accursed Blacksmith": ["Forge", "Rummage", "Ignored"],
            "The Mausoleum": ["Opened", "Ignored"],
            "Tomb of Lord Red Mask": ["Got Gold", "Paid", "Ignored"],
                "Masked Bandits": ["Paid Fearfully", "Fought Bandits"],
                "Vampires": ["Accepted", "Refused"],
                "Ghosts": ["Accepted", "Refused"],
                "Duplicator": ["Duplicated", "Ignored"],
                "N'loth": ["Gave Relic", "Ignored"],
                "The Joust": ["Murderer", "Owner"],
                "The Divine Fountain": ["Drank", "Ignored"],
            "Knowing Skull": ["Riches", "Success", "A Pick Me Up", "Leave"],
            "Note For Yourself": ["Took Card", "Ignored"],
            "Secret Portal": ["Entered Portal", "Ignored"],
                "We Meet Again!": ["Gave Potion", "Gave Gold", "Gave Card", "Ignored"],
                "Bonfire Spirits": ["Card Offered", "Ignored"],
                "Designer In-Spire": ["Adjusted", "Cleaned Up", "Full Service", "Ignored"],
                "Face Trader": ["Touched", "Traded", "Ignored"],
                "Forgotten Altar": ["Shed Blood", "Smashed Altar", "Ignored"],
                "Lab": ["Obtained Potions"],
                "Match and Keep": ["Played"],
                "The Moai Head": ["Jumped", "Offered Golden Idol", "Ignored"],
                "Purifier": ["Purged", "Ignored"],
                "Sensory Stone": ["Recall", "Remember", "Live Forever", "Ignored"],
                "The Woman in Blue": ["Bought 1 Potion", "Bought 2 Potions", "Bought 3 Potions", "Ignored"],
                "Transmogrifier": ["Transformed", "Ignored"],
                "Upgrade Shrine": ["Upgraded", "Ignored"],
                "Wheel of Change": ["Spun"],
                "Pleading Vagrant": ["Gave Gold", "Robbed", "Ignored"],
                "Dead Adventurer": ["Searched", "Escaped"],
                "Ominous Forge": ["Forge", "Rummage", "Ignored"],
                "Mysterious Sphere": ["Fought Orb Walkers", "Ignored"],
                "Colosseum": ["Fought", "Cowardice"],
            }.get(event_id, ["Ignored"])
            if event_id == "Golden Idol":
                if any(relic.get("relic_id") == "Golden Idol" for relic in self.relics):
                    self.event_options = [
                        {"kind": "event", "event_id": event_id, "name": "Outrun", "label": "Outrun", "choice_index": 2},
                        {"kind": "event", "event_id": event_id, "name": "Smash", "label": "Smash", "choice_index": 3},
                        {"kind": "event", "event_id": event_id, "name": "Hide", "label": "Hide", "choice_index": 4},
                    ]
                else:
                    self.event_options = [
                        {"kind": "event", "event_id": event_id, "name": "Take", "label": "Take", "choice_index": 0},
                        {"kind": "event", "event_id": event_id, "name": "Ignored", "label": "Ignored", "choice_index": 1},
                    ]
            elif event_id == "The Cleric":
                self.event_options = [
                    {"kind": "event", "event_id": event_id, "name": "Healed", "label": "Healed", "choice_index": 0},
                ]
                purge_cost = 75 if self.ascension_level >= 15 else 50
                if self.gold >= purge_cost:
                    self.event_options.append(
                        {"kind": "event", "event_id": event_id, "name": "Card Removal", "label": "Card Removal", "choice_index": 1}
                    )
                self.event_options.append(
                    {"kind": "event", "event_id": event_id, "name": "Leave", "label": "Leave", "choice_index": 2}
                )
            elif event_id == "Wing Statue":
                self.event_options = [
                    {"kind": "event", "event_id": event_id, "name": "Card Removal", "label": "Card Removal", "choice_index": 0},
                ]
                if any(_base_damage_for_card(card) >= 10 for card in self.deck):
                    self.event_options.append(
                        {"kind": "event", "event_id": event_id, "name": "Gained Gold", "label": "Gained Gold", "choice_index": 1}
                    )
                self.event_options.append(
                    {"kind": "event", "event_id": event_id, "name": "Ignored", "label": "Ignored", "choice_index": 2}
                )
            elif event_id in {"We Meet Again!", "WeMeetAgain"}:
                self.event_options = []
                if self.event_state.get("potion_idx") is not None:
                    self.event_options.append({"kind": "event", "event_id": event_id, "name": "Gave Potion", "label": "Gave Potion", "choice_index": 0})
                if int(self.event_state.get("gold", -1)) >= 0:
                    self.event_options.append({"kind": "event", "event_id": event_id, "name": "Gave Gold", "label": "Gave Gold", "choice_index": 1})
                if self.event_state.get("card_idx") is not None:
                    self.event_options.append({"kind": "event", "event_id": event_id, "name": "Gave Card", "label": "Gave Card", "choice_index": 2})
                self.event_options.append({"kind": "event", "event_id": event_id, "name": "Ignored", "label": "Ignored", "choice_index": 3})
            elif event_id == "N'loth":
                self.event_options = []
                relic_indexes = [int(index) for index in self.event_state.get("relic_indexes", [])]
                for choice_index, relic_index in enumerate(relic_indexes):
                    if not (0 <= relic_index < len(self.relics)):
                        continue
                    relic = self.relics[relic_index]
                    relic_name = str(relic.get("name") or relic.get("relic_id") or "Relic")
                    self.event_options.append(
                        {
                            "kind": "event",
                            "event_id": event_id,
                            "name": "Gave Relic",
                            "label": relic_name,
                            "choice_index": choice_index,
                        }
                    )
                self.event_options.append(
                    {
                        "kind": "event",
                        "event_id": event_id,
                        "name": "Ignored",
                        "label": "Ignored",
                        "choice_index": len(self.event_options),
                    }
                )
            elif event_id == "Falling":
                self.event_options = []
                if int(self.event_state.get("attack", -1)) >= 0:
                    self.event_options.append(
                        {"kind": "event", "event_id": event_id, "name": "Removed Attack", "label": "Removed Attack", "choice_index": 0}
                    )
                if int(self.event_state.get("skill", -1)) >= 0:
                    self.event_options.append(
                        {"kind": "event", "event_id": event_id, "name": "Removed Skill", "label": "Removed Skill", "choice_index": 1}
                    )
                if int(self.event_state.get("power", -1)) >= 0:
                    self.event_options.append(
                        {"kind": "event", "event_id": event_id, "name": "Removed Power", "label": "Removed Power", "choice_index": 2}
                    )
                if not self.event_options:
                    self.event_options.append(
                        {"kind": "event", "event_id": event_id, "name": "Ignored", "label": "Ignored", "choice_index": 3}
                    )
            elif event_id == "Cursed Tome":
                self._set_cursed_tome_options()
            elif event_id == "Forgotten Altar":
                self.event_options = []
                if self._has_relic("Golden Idol"):
                    self.event_options.append(
                        {"kind": "event", "event_id": event_id, "name": "Smashed Altar", "label": "Smashed Altar", "choice_index": 0}
                    )
                    self.event_options.append(
                        {"kind": "event", "event_id": event_id, "name": "Shed Blood", "label": "Shed Blood", "choice_index": 1}
                    )
                else:
                    self.event_options.append(
                        {"kind": "event", "event_id": event_id, "name": "Shed Blood", "label": "Shed Blood", "choice_index": 1}
                    )
                self.event_options.append(
                    {"kind": "event", "event_id": event_id, "name": "Ignored", "label": "Ignored", "choice_index": 2}
                )
            elif event_id == "Pleading Vagrant":
                self.event_options = []
                if self.gold >= 85:
                    self.event_options.append(
                        {"kind": "event", "event_id": event_id, "name": "Gave Gold", "label": "Gave Gold", "choice_index": 0}
                    )
                self.event_options.append(
                    {"kind": "event", "event_id": event_id, "name": "Robbed", "label": "Robbed", "choice_index": 1}
                )
                self.event_options.append(
                    {"kind": "event", "event_id": event_id, "name": "Ignored", "label": "Ignored", "choice_index": 2}
                )
            elif event_id == "Vampires":
                if self._has_relic("Blood Vial"):
                    self.event_options = [
                        {"kind": "event", "event_id": event_id, "name": "Offered", "label": "Offered", "choice_index": 0},
                        {"kind": "event", "event_id": event_id, "name": "Accepted", "label": "Accepted", "choice_index": 1},
                        {"kind": "event", "event_id": event_id, "name": "Refused", "label": "Refused", "choice_index": 2},
                    ]
                else:
                    self.event_options = [
                        {"kind": "event", "event_id": event_id, "name": "Accepted", "label": "Accepted", "choice_index": 1},
                        {"kind": "event", "event_id": event_id, "name": "Refused", "label": "Refused", "choice_index": 2},
                    ]
            elif event_id == "Designer In-Spire":
                self.event_options = []
                gold_cost0, gold_cost1, gold_cost2 = self._designer_in_spire_costs()
                upgrade_one = bool(self.event_state.get("designer_upgrade_one", True))
                clean_up_is_remove = bool(self.event_state.get("designer_cleanup_is_remove", True))
                if self.gold >= gold_cost0 and any(_card_can_upgrade(card) for card in self.deck):
                    self.event_options.append(
                        {
                            "kind": "event",
                            "event_id": event_id,
                            "name": "Adjusted",
                            "label": "Adjusted",
                            "choice_index": 0 if upgrade_one else 1,
                        }
                    )
                if self.gold >= gold_cost1:
                    if clean_up_is_remove:
                        if any(_card_can_transform(card) for card in self.deck):
                            self.event_options.append(
                                {
                                    "kind": "event",
                                    "event_id": event_id,
                                    "name": "Cleaned Up",
                                    "label": "Cleaned Up",
                                    "choice_index": 2,
                                }
                            )
                    else:
                        if sum(1 for card in self.deck if _card_can_transform(card)) >= 2:
                            self.event_options.append(
                                {
                                    "kind": "event",
                                    "event_id": event_id,
                                    "name": "Cleaned Up",
                                    "label": "Cleaned Up",
                                    "choice_index": 3,
                                }
                            )
                if self.gold >= gold_cost2 and any(_card_can_transform(card) for card in self.deck):
                    self.event_options.append(
                        {
                            "kind": "event",
                            "event_id": event_id,
                            "name": "Full Service",
                            "label": "Full Service",
                            "choice_index": 4,
                        }
                    )
                self.event_options.append(
                    {
                        "kind": "event",
                        "event_id": event_id,
                        "name": "Ignored",
                        "label": "Ignored",
                        "choice_index": 5,
                    }
                )
            else:
                option_event_id = "Council of Ghosts" if event_id == "Ghosts" else event_id
                self.event_options = [
                    {"kind": "event", "event_id": option_event_id, "name": label, "label": label, "choice_index": index}
                    for index, label in enumerate(option_labels)
                ]

    def _enter_neow(self) -> None:
            self.phase = "NEOW"
            self.neow_options = list(self.neow_options)

    def _enter_shop(self) -> None:
            self.phase = "SHOP"
            tea_set = self._relic("Ancient Tea Set")
            if tea_set is not None and int(tea_set.get("counter", 0)) > 0:
                # Align with lightspeed's current room-transition behavior:
                # Ancient Tea Set survives into event combats from '?', but a
                # shop visit clears the pending +2 energy before the next fight.
                tea_set["counter"] = 0
            if self._has_relic("Meal Ticket"):
                self._heal_run(15)
            cards, colorless_cards = self._roll_shop_cards()
            card_items: list[dict[str, Any]] = []
            for index, (card, rarity) in enumerate(cards):
                card_items.append({
                    "kind": "shop",
                    "name": card.name,
                    "item_kind": "card",
                    "item_id": card.card_id,
                    "price": self._shop_card_price_base(card, rarity_override=rarity, colorless=False),
                    "card": card_to_spirecomm(card),
                    "choice_index": index,
                })
            for colorless, rarity in colorless_cards:
                card_items.append({
                    "kind": "shop",
                    "name": colorless.name,
                    "item_kind": "card",
                    "item_id": colorless.card_id,
                    "price": self._shop_card_price_base(colorless, colorless=True, rarity_override=rarity),
                    "card": card_to_spirecomm(colorless),
                    "choice_index": len(card_items),
                })
            sale_idx = int(self.randoms.merchant.random(4))
            if 0 <= sale_idx < len(card_items):
                card_items[sale_idx]["price"] //= 2
            for item in card_items:
                item["price"] = self._apply_shop_discount(int(item["price"]))
            self.shop_items = list(card_items)
            shop_relics: list[dict[str, Any]] = []
            for _ in range(2):
                tier_roll = int(self.randoms.merchant.random(99))
                tier = "COMMON" if tier_roll < 48 else "UNCOMMON" if tier_roll < 82 else "RARE"
                relic = self._roll_relic_of_tier(tier, shop_room=True, from_front=False)
                relic["price"] = self._shop_relic_price_base(relic)
                shop_relics.append(relic)
            shop_relic = self._roll_relic_of_tier("SHOP", shop_room=True, from_front=False)
            shop_relic["price"] = self._shop_relic_price_base(shop_relic, rarity_override="SHOP")
            shop_relics.append(shop_relic)
            for relic in shop_relics:
                relic["price"] = self._apply_shop_discount(int(relic["price"]))
            for relic in shop_relics:
                self.shop_items.append({
                    "kind": "shop",
                    "name": relic["name"],
                    "item_kind": "relic",
                    "item_id": relic["relic_id"],
                    "price": relic["price"],
                    "relic": relic,
                    "choice_index": len(self.shop_items),
                })
            if not any(item.get("relic_id") == "Sozu" for item in self.relics):
                for _ in range(3):
                    potion = roll_potion(self.randoms.potion)
                    potion.price = self._shop_potion_price_base(potion)
                    potion.price = self._apply_shop_discount(int(potion.price))
                    self.shop_items.append({
                        "kind": "shop",
                        "name": potion.name,
                        "item_kind": "potion",
                        "item_id": potion.potion_id,
                        "price": potion.price,
                        "potion_id": potion.potion_id,
                        "choice_index": len(self.shop_items),
                    })
            purge_base = 50 if self._has_relic("Smiling Mask") else 75 + 25 * self.shop_remove_count
            purge_price = self._apply_shop_discount(purge_base, include_ascension=False)
            self.shop_items.append({"kind": "shop", "name": "PURGE", "item_kind": "purge", "price": purge_price, "choice_index": len(self.shop_items)})
            self.shop_items.append({"kind": "shop", "name": "LEAVE", "item_kind": "leave", "price": 0, "choice_index": len(self.shop_items)})

    def _event_pool_for_act(self) -> list[str]:
            events, shrines = self._event_pools_for_act()
            return events + shrines + list(self.special_one_time_event_list or self._one_time_event_pool())

    def _event_pools_for_act(self) -> tuple[list[str], list[str]]:
            shrines_common = ["Match and Keep", "Wheel of Change", "Golden Shrine", "Transmogrifier", "Purifier", "Upgrade Shrine"]
            if self.act <= 1:
                return (
                    [
                        "Big Fish", "The Cleric", "Dead Adventurer", "Golden Idol",
                        "Wing Statue", "World of Goop", "The Ssssserpent", "Living Wall",
                        "Hypnotizing Colored Mushrooms", "Scrap Ooze", "Shining Light",
                    ],
                    ["Match and Keep", "Golden Shrine", "Transmogrifier", "Purifier", "Upgrade Shrine", "Wheel of Change"],
                )
            if self.act == 2:
                return (
                    [
                        "Pleading Vagrant", "Ancient Writing", "Old Beggar", "Colosseum",
                        "Cursed Tome", "Augmenter", "Forgotten Altar", "Ghosts",
                        "Masked Bandits", "The Nest", "The Library", "The Mausoleum", "Vampires",
                    ],
                    list(shrines_common),
                )
            return (
                [
                    "Falling", "Mindbloom", "The Moai Head", "Mysterious Sphere",
                    "Sensory Stone", "Tomb of Lord Red Mask", "Winding Halls",
                ],
                list(shrines_common),
            )

    def _first_purge_index(self) -> int | None:
            priority = ["AscendersBane", "Wound", "Burn", "Dazed", "Strike_R", "Defend_R"]
            for card_id in priority:
                for index, card in enumerate(self.deck):
                    if card.card_id == card_id:
                        return index
            return 0 if self.deck else None

    def _first_upgradable_index(self) -> int | None:
            for index, card in enumerate(self.deck):
                if card.upgrades <= 0 and card.card_def.card_type not in {"STATUS", "CURSE"}:
                    return index
            return None

    def _designer_in_spire_is_unfavorable(self) -> bool:
            return self.ascension_level >= 15

    def _designer_in_spire_costs(self) -> tuple[int, int, int]:
            if self._designer_in_spire_is_unfavorable():
                return 50, 75, 110
            return 40, 60, 90

    def _random_upgradeable_indexes_from_rng(self, rng: StsRandom, count: int) -> list[int]:
            indexes = [index for index, card in enumerate(self.deck) if _card_can_upgrade(card)]
            if not indexes:
                return []
            java_collections_shuffle(indexes, rng.random_long())
            return indexes[:max(0, int(count))]

    def _upgrade_random_cards_from_rng(self, rng: StsRandom, count: int) -> None:
            for deck_index in self._random_upgradeable_indexes_from_rng(rng, count):
                if 0 <= deck_index < len(self.deck):
                    _increment_card_upgrade(self.deck[deck_index])

    def _gain_gold(self, amount: int) -> None:
            amount = max(0, int(amount))
            if amount <= 0 or any(relic.get("relic_id") == "Ectoplasm" for relic in self.relics):
                return
            self.gold += amount
            if any(relic.get("relic_id") == "Bloody Idol" for relic in self.relics):
                self._heal_run(5)

    def _generate_act_map(self, act: int) -> None:
            if act >= 4:
                return
            start_floor, boss_floor = self._act_floor_range(act)
            try:
                graph, layers = self._generate_act_map_from_lightspeed(act)
            except Exception:
                graph, layers = generate_native_act_map(
                    seed=self.seed,
                    ascension_level=self.ascension_level,
                    act=act,
                    start_floor=start_floor,
                    set_burning=self.enable_act4_keys and "emerald" not in self.keys,
                )
            # The native simulator marks the burning elite as E_GREEN for model
            # visibility. Real run-history path symbols still record it as E.
            self.map_graph = graph
            self.map_layers = layers
            self.current_map_node_id = None

    def _generate_monster_schedules_for_act(self, act: int) -> None:
            if act == 4:
                self.monster_list = []
                self.monster_list_offset = 0
                self.elite_monster_list = ["ShieldAndSpear"]
                self.elite_monster_list_offset = 0
                self._act_bosses[4] = "CorruptHeart"
                return
            monsters, elites, boss, second_boss = generate_monster_schedules(self.randoms.monster, act, self.ascension_level)
            self.monster_list = monsters
            self.monster_list_offset = 0
            self.elite_monster_list = elites
            self.elite_monster_list_offset = 0
            boss_name = {
                "TheGuardian": "The Guardian",
                "Hexaghost": "Hexaghost",
                "SlimeBoss": "Slime Boss",
                "BronzeAutomaton": "Bronze Automaton",
                "TheCollector": "The Collector",
                "TheChamp": "The Champ",
                "AwakenedOne": "Awakened One",
                "TimeEater": "Time Eater",
                "DonuDeca": "Donu and Deca",
            }.get(boss, boss)
            self._act_bosses[act] = boss_name
            if second_boss:
                self.second_act_boss = {
                    "AwakenedOne": "Awakened One",
                    "TimeEater": "Time Eater",
                    "DonuDeca": "Donu and Deca",
                }.get(second_boss, second_boss)

    def _generate_neow_options(self) -> None:
            rng = self.randoms.neow
            options: list[dict[str, Any]] = []

            first_bonus = [
                "THREE_CARDS",
                "ONE_RANDOM_RARE_CARD",
                "REMOVE_CARD",
                "UPGRADE_CARD",
                "TRANSFORM_CARD",
                "RANDOM_COLORLESS",
            ][int(rng.random(0, 5))]
            options.append(self._make_neow_option(0, first_bonus, "NONE"))

            second_bonus = [
                "THREE_SMALL_POTIONS",
                "RANDOM_COMMON_RELIC",
                "TEN_PERCENT_HP_BONUS",
                "THREE_ENEMY_KILL",
                "HUNDRED_GOLD",
            ][int(rng.random(0, 4))]
            options.append(self._make_neow_option(1, second_bonus, "NONE"))

            drawback = [
                "TEN_PERCENT_HP_LOSS",
                "NO_GOLD",
                "CURSE",
                "PERCENT_DAMAGE",
            ][int(rng.random(0, 3))]
            if drawback in NEOW_MID_TIER_BY_DRAWBACK:
                third_bonus = NEOW_MID_TIER_BY_DRAWBACK[drawback][int(rng.random(0, 5))]
            else:
                third_bonus = [
                    "RANDOM_COLORLESS_2",
                    "REMOVE_TWO",
                    "ONE_RARE_RELIC",
                    "THREE_RARE_CARDS",
                    "TWO_FIFTY_GOLD",
                    "TRANSFORM_TWO_CARDS",
                    "TWENTY_PERCENT_HP_BONUS",
                ][int(rng.random(0, 6))]
            options.append(self._make_neow_option(2, third_bonus, drawback))
            options.append(self._make_neow_option(3, "BOSS_RELIC", "LOSE_STARTER_RELIC"))
            rng.random(0, 0)
            self.neow_options = options

    def _has_relic(self, relic_id: str) -> bool:
            return any(relic.get("relic_id") == relic_id for relic in self.relics)

    def _heal_run(self, amount: int) -> None:
            amount = max(0, int(amount))
            if self._has_relic("Mark of the Bloom"):
                return
            self.player.current_hp = min(self.player.max_hp, self.player.current_hp + amount)

    def _ironclad_card_pool(self, *, card_type: str | None = None, rarity: str | None = None):
            return ironclad_card_pool(card_type=card_type, rarity=rarity, exclude_ids=self.locked_card_ids)

    def _lose_run_hp(self, amount: int) -> None:
            amount = max(0, int(amount))
            if amount <= 0:
                return
            if self._has_relic("Torii") and 1 < amount <= 5:
                amount = 1
            if self._has_relic("Tungsten Rod"):
                amount = max(0, amount - 1)
                if amount <= 0:
                    return
            self.player.current_hp = max(0, self.player.current_hp - amount)
            if self.player.current_hp <= 0:
                self._restore_from_run_death()

    def _lose_run_hp_raw(self, amount: int) -> None:
            amount = max(0, int(amount))
            if amount <= 0:
                return
            self.player.current_hp = max(0, self.player.current_hp - amount)
            if self.player.current_hp <= 0:
                self._restore_from_run_death()


    def _make_deck_card(self, card_id: str, *, upgrades: int = 0, uuid: str = "") -> CardInstance:
            card = make_card(card_id, upgrades=upgrades, uuid=uuid)
            if card.card_id == "Searing Blow" and upgrades > 0:
                card.misc = max(card.misc, upgrades)
            if card.card_def.card_type == "ATTACK" and any(relic.get("relic_id") == "Molten Egg" for relic in self.relics):
                if card.card_id == "Searing Blow":
                    _increment_card_upgrade(card)
                else:
                    _ensure_card_upgraded(card)
            if card.card_def.card_type == "SKILL" and any(relic.get("relic_id") == "Toxic Egg" for relic in self.relics):
                _ensure_card_upgraded(card)
            if card.card_def.card_type == "POWER" and any(relic.get("relic_id") == "Frozen Egg" for relic in self.relics):
                _ensure_card_upgraded(card)
            return card

    def _make_neow_option(self, index: int, bonus: str, drawback: str) -> dict[str, Any]:
            return {
                "kind": "neow",
                "name": f"OPTION_{index}",
                "label": f"OPTION_{index}",
                "choice_index": index,
                "bonus": bonus,
                "drawback": drawback,
                "bonus_text": NEOW_BONUS_LABELS[bonus],
                "drawback_text": NEOW_DRAWBACK_LABELS[drawback],
            }

    def _neow_card_reward(self, *, rare_only: bool = False) -> list[CardInstance]:
            reward: list[CardInstance] = []
            seen: set[str] = set()
            for _ in range(3):
                rarity = "UNCOMMON" if self.randoms.neow.random_boolean(0.33) else "COMMON"
                if rare_only:
                    rarity = "RARE"
                card = self._random_class_card_of_rarity_from_rng(self.randoms.neow, rarity)
                while card.card_id in seen:
                    card = self._random_class_card_of_rarity_from_rng(self.randoms.neow, rarity)
                seen.add(card.card_id)
                card.uuid = f"neow-reward-{card.card_id}-{len(reward)}"
                reward.append(card)
            return reward

    def _neow_colorless_card_reward(self, *, rare_only: bool = False) -> list[CardInstance]:
            reward: list[CardInstance] = []
            seen: set[str] = set()
            for _ in range(3):
                rarity = "UNCOMMON" if self.randoms.neow.random_boolean(0.33) else "COMMON"
                if rare_only:
                    rarity = "RARE"
                elif rarity == "COMMON":
                    rarity = "UNCOMMON"
                pool = [
                    CARD_LIBRARY[card_id] for card_id in COLORLESS_CARD_ID_ORDER
                    if CARD_LIBRARY[card_id].rarity == rarity
                ]
                chosen = pool[int(self.randoms.card.random(len(pool) - 1))]
                while chosen.card_id in seen:
                    chosen = pool[int(self.randoms.card.random(len(pool) - 1))]
                seen.add(chosen.card_id)
                reward.append(CardInstance(chosen, uuid=f"neow-colorless-{chosen.card_id}-{len(reward)}"))
            return reward

    def _next_elite_encounter(self) -> str:
            if self.elite_monster_list_offset >= len(self.elite_monster_list):
                self.elite_monster_list = generate_elite_schedule(self.randoms.monster, self.act)
                self.elite_monster_list_offset = 0
            encounter = self.elite_monster_list[self.elite_monster_list_offset]
            self.elite_monster_list_offset += 1
            return encounter

    def _next_monster_encounter(self) -> str:
            if self.monster_list_offset >= len(self.monster_list):
                self.monster_list = generate_strong_monster_schedule(self.randoms.monster, self.act)
                self.monster_list_offset = 0
            encounter = self.monster_list[self.monster_list_offset]
            self.monster_list_offset += 1
            return encounter

    def _one_time_event_pool(self) -> list[str]:
            pool = [
                "Ominous Forge", "Bonfire Spirits", "Designer In-Spire", "Duplicator",
                "Face Trader", "The Divine Fountain", "Knowing Skull", "Lab", "N'loth",
                "Secret Portal", "The Joust", "We Meet Again!", "The Woman in Blue",
            ]
            if self.ascension_level < 15:
                pool.insert(9, "Note For Yourself")
            return pool

    def _open_bottle_card_select(self, relic_id: str) -> bool:
            type_by_relic = {
                "Bottled Flame": "ATTACK",
                "Bottled Lightning": "SKILL",
                "Bottled Tornado": "POWER",
            }
            card_type = type_by_relic.get(relic_id)
            if card_type is None:
                return False
            self.phase = "CARD_SELECT"
            self.card_select_context = "BOTTLE_REWARD"
            self.pending_bottle_relic_id = relic_id
            self.card_select_count = 1
            self.card_select_options = []
            for index, card in enumerate(self.deck):
                if card.card_def.card_type != card_type:
                    continue
                self.card_select_options.append({
                    "kind": "card_select",
                    "name": "BOTTLE",
                    "select_type": "BOTTLE",
                    "label": card.name,
                    "choice_index": index,
                    "target_index": index,
                    "deck_index": index,
                    "card": card_to_spirecomm(card),
                })
            return bool(self.card_select_options)

    def _open_card_select(self, context: str, count: int) -> None:
            self.phase = "CARD_SELECT"
            self.card_select_context = context
            self.card_select_count = count
            bottled_blocked_contexts = {
                "NEOW_REMOVE",
                "NEOW_TRANSFORM",
                "EVENT_REMOVE",
                "EVENT_TRANSFORM",
                "BONFIRE_SPIRITS",
            }
            bottled_indexes = self._bottled_deck_indexes() if context in bottled_blocked_contexts else set()
            upgrade_contexts = {"NEOW_UPGRADE", "EVENT_UPGRADE", "CAMPFIRE_SMITH"}
            transform_contexts = {"NEOW_TRANSFORM", "EVENT_TRANSFORM", "TRANSFORM_UPGRADE", "BONFIRE_SPIRITS"}
            remove_contexts = {"NEOW_REMOVE", "EVENT_REMOVE", "CAMPFIRE_TOKE"}
            multi_step_contexts = {"TRANSFORM_UPGRADE", "EVENT_REMOVE", "EVENT_TRANSFORM"}
            if context in multi_step_contexts and self.card_select_available_indexes:
                selectable_indexes = [
                    index
                    for index in self.card_select_available_indexes
                    if 0 <= index < len(self.deck)
                ]
            else:
                selectable_indexes = [
                    index
                    for index, card in enumerate(self.deck)
                    if index not in bottled_indexes
                    and (context not in upgrade_contexts or _card_can_upgrade(card))
                    and (context not in transform_contexts | remove_contexts or _card_can_transform(card))
                ]
                self.card_select_available_indexes = list(selectable_indexes)
                self.card_select_selected_indexes = []
            self.card_select_options = [
                {
                    "kind": "card_select",
                    "name": self.deck[index].name,
                    "select_type": context,
                    "label": self.deck[index].name,
                    "choice_index": index,
                    "target_index": index,
                    "card": card_to_spirecomm(self.deck[index]),
                }
                for index in selectable_indexes
            ]

    def _open_library_card_select(self) -> None:
            self.phase = "CARD_SELECT"
            self.card_select_context = "LIBRARY_OBTAIN"
            self.card_select_count = 1
            seen: set[str] = set()
            cards: list[CardInstance | None] = [None] * 20
            for index in range(19, -1, -1):
                while True:
                    rarity = self._roll_card_rarity(room="?")
                    pool = self._ironclad_card_pool(rarity=rarity) or self._ironclad_card_pool()
                    chosen = self.randoms.card.choice(pool)
                    if chosen.card_id not in seen:
                        break
                seen.add(chosen.card_id)
                cards[index] = self._make_deck_card(
                    chosen.card_id,
                    upgrades=int(getattr(chosen, "upgrades", 0) or 0),
                    uuid=f"library-{self.floor}-{index}-{chosen.card_id}",
                )
            self.card_select_generated_cards = [card for card in cards if card is not None]
            self.card_select_options = [
                {
                    "kind": "card_select",
                    "name": card.name,
                    "select_type": "OBTAIN",
                    "label": card.name,
                    "choice_index": index,
                    "target_index": -1,
                    "deck_index": -1,
                    "card": card_to_spirecomm(card),
                }
                for index, card in enumerate(self.card_select_generated_cards)
            ]

    def _open_neow_card_reward(self, *, rare_only: bool = False, colorless: bool = False) -> None:
            self.phase = "CARD_REWARD"
            self.reward_context = "NEOW"
            self.reward_close_required = False
            self.reward_card_bundles = [self._neow_colorless_card_reward(rare_only=rare_only) if colorless else self._neow_card_reward(rare_only=rare_only)]
            self._refresh_reward_cards()

    def _open_neow_card_select(self, context: str, count: int) -> None:
            self._open_card_select(context, count)

    def _open_potion_reward_screen(self, *, count: int = 1, context: str = "EVENT") -> None:
            self.phase = "CARD_REWARD"
            self.reward_context = context
            self.reward_close_required = False
            self.reward_card_bundles = []
            self.reward_cards = []
            self.reward_gold = 0
            self.reward_gold_piles = []
            self.reward_emerald_key = False
            self.reward_relics = []
            self.reward_potions = []
            for _ in range(max(0, int(count))):
                self.reward_potions.append(get_random_potion(self.randoms.potion, "IRONCLAD"))

    def _open_relic_reward_screen(self, relic: dict[str, Any], *, context: str = "EVENT") -> None:
            self.phase = "CARD_REWARD"
            self.reward_context = context
            self.reward_close_required = False
            self.reward_card_bundles = []
            self.reward_cards = []
            self.reward_gold = 0
            self.reward_gold_piles = []
            self.reward_emerald_key = False
            self.reward_relics = [relic]
            self.reward_potions = []

    def _owned_relic_ids(self) -> set[str]:
            return {str(item.get("relic_id")) for item in self.relics}

    def _question_room_outcome(self, *, last_room_was_shop: bool = False) -> str:
            roll = self.randoms.event.random()
            force_treasure = False
            tiny_chest = self._relic("Tiny Chest")
            if tiny_chest is not None:
                counter = int(tiny_chest.get("counter", 0))
                counter += 1
                if counter >= 4:
                    tiny_chest["counter"] = 0
                    force_treasure = True
                else:
                    tiny_chest["counter"] = counter
            monster_size = int(_f32(self.monster_chance * _f32(100.0)))
            shop_size = monster_size + (0 if last_room_was_shop else int(_f32(self.shop_chance * _f32(100.0))))
            treasure_size = shop_size + int(_f32(self.treasure_chance * _f32(100.0)))
            index = int(roll * 100.0)
            if index < monster_size:
                outcome = "M"
            elif index < shop_size:
                outcome = "$"
            elif index < treasure_size:
                outcome = "T"
            else:
                outcome = "?"
            if force_treasure:
                outcome = "T"
            rolled_outcome = outcome
            if outcome == "M" and self._has_relic("Juzu Bracelet"):
                outcome = "?"
            self.monster_chance = _f32(0.10) if rolled_outcome == "M" else _f32(self.monster_chance + _f32(0.10))
            self.shop_chance = _f32(0.03) if outcome == "$" else _f32(self.shop_chance + _f32(0.03))
            self.treasure_chance = _f32(0.02) if outcome == "T" else _f32(self.treasure_chance + _f32(0.02))
            return outcome

    def _random_class_card_of_rarity(self, rarity: str) -> CardInstance:
            pool = self._ironclad_card_pool(rarity=rarity)
            if not pool:
                pool = self._ironclad_card_pool()
            chosen = pool[int(self.randoms.card.random(len(pool) - 1))]
            return make_card(chosen.card_id, uuid=f"rarity-{self.floor}-{chosen.card_id}")

    def _random_class_card_of_rarity_from_rng(self, rng: StsRandom, rarity: str) -> CardInstance:
            pool = self._ironclad_card_pool(rarity=rarity) or self._ironclad_card_pool()
            chosen = pool[int(rng.random(len(pool) - 1))]
            return CardInstance(chosen, uuid=f"random-{rarity.lower()}-{chosen.card_id}-{self.floor}")

    def _random_curse_id(self) -> str:
            return self.randoms.card.choice([
                "Regret",
                "Injury",
                "Shame",
                "Parasite",
                "Normality",
                "Doubt",
                "Writhe",
                "Pain",
                "Decay",
                "Clumsy",
            ])

    def _refresh_reward_cards(self) -> None:
            self.reward_cards = [card for bundle in self.reward_card_bundles for card in bundle]

    def _relic(self, relic_id: str) -> dict[str, Any] | None:
            return next((relic for relic in self.relics if relic.get("relic_id") == relic_id), None)

    def _remove_shop_item(self, action: dict[str, Any]) -> None:
            item_kind = str(action.get("item_kind") or "")
            action_id = action.get("item_id")
            action_name = action.get("name")
            action_price = int(action.get("price", 0) or 0)
            kept: list[dict[str, Any]] = []
            removed = False
            for item in self.shop_items:
                same_item = (
                    item.get("item_kind") == item_kind
                    and item.get("item_id") == action_id
                    and str(item.get("name") or "") == str(action_name or "")
                    and int(item.get("price", 0) or 0) == action_price
                )
                if not removed and same_item:
                    removed = True
                    continue
                kept.append(item)
            self.shop_items = kept
            replacement = self._restock_shop_item(item_kind, action)
            if replacement is not None:
                replacement["choice_index"] = len(self.shop_items)
                self.shop_items.insert(max(0, len(self.shop_items) - 1), replacement)

    def _reset_act_random_room_state(self) -> None:
            self.monster_chance = _f32(0.10)
            self.shop_chance = _f32(0.03)
            self.treasure_chance = _f32(0.02)
            self.potion_chance_counter = 0
            self.event_list, self.shrine_list = self._event_pools_for_act()

    def _restock_shop_item(self, item_kind: str, removed_item: dict[str, Any] | None = None) -> dict[str, Any] | None:
            if not self._has_relic("The Courier") or item_kind not in {"card", "relic", "potion"}:
                return None
            if item_kind == "card":
                removed_card_id = str((removed_item or {}).get("item_id") or "")
                normalized_removed_card_id = removed_card_id.lower().replace(" ", "")
                colorless_normalized_ids = {card_id.lower().replace(" ", "") for card_id in COLORLESS_CARD_IDS}
                if normalized_removed_card_id in colorless_normalized_ids:
                    rarity = "RARE" if self.randoms.merchant.random() < 0.30 else "UNCOMMON"
                    colorless_pool = [CARD_LIBRARY[card_id] for card_id in COLORLESS_CARD_ID_ORDER if CARD_LIBRARY[card_id].rarity == rarity]
                    chosen = colorless_pool[int(self.randoms.card.random(len(colorless_pool) - 1))]
                    card = self._make_deck_card(chosen.card_id, uuid=f"shop-colorless-{chosen.card_id}-{self.floor}")
                    price = self._shop_courier_restock_card_price(rarity, colorless=True)
                else:
                    rarity = self._roll_card_rarity(room="$")
                    card = self._random_class_card_of_rarity_from_rng(self.randoms.math_util, rarity)
                    price = self._shop_courier_restock_card_price(rarity, colorless=False)
                return {
                    "kind": "shop",
                    "name": card.name,
                    "item_kind": "card",
                    "item_id": card.card_id,
                    "price": price,
                    "card": card_to_spirecomm(card),
                }
            if item_kind == "relic":
                tier_roll = int(self.randoms.merchant.random(99))
                tier = "COMMON" if tier_roll < 48 else "UNCOMMON" if tier_roll < 82 else "RARE"
                relic = self._roll_relic_of_tier(tier, shop_room=True, from_front=False)
                relic["price"] = self._shop_courier_restock_misc_price()
                return {
                    "kind": "shop",
                    "name": relic["name"],
                    "item_kind": "relic",
                    "item_id": relic["relic_id"],
                    "price": relic["price"],
                    "relic": relic,
                }
            potion = roll_potion(self.randoms.potion)
            potion.price = self._shop_courier_restock_misc_price()
            return {
                "kind": "shop",
                "name": potion.name,
                "item_kind": "potion",
                "item_id": potion.potion_id,
                "price": potion.price,
                "potion_id": potion.potion_id,
            }

    def _restore_from_run_death(self) -> bool:
            if self.player.current_hp > 0:
                return False
            if self._has_relic("Mark of the Bloom"):
                self.phase = "GAME_OVER"
                return True
            bark_multiplier = 2 if self._has_relic("Sacred Bark") else 1
            for index, potion in enumerate(self.potions):
                if potion.potion_id != "FairyPotion":
                    continue
                self.potions[index] = PotionInstance()
                self.player.current_hp = max(1, int(self.player.max_hp * (0.3 * bark_multiplier)))
                return True
            if (relic := self._relic("Lizard Tail")) is not None and int(relic.get("counter", -1)) != 0:
                relic["counter"] = 0
                self.player.current_hp = max(1, self.player.max_hp // 2)
                return True
            self.phase = "GAME_OVER"
            return True

    def _roll_boss_relics(self, count: int = 3) -> list[dict[str, Any]]:
            return [self._roll_relic_of_tier("BOSS") for _ in range(count)]

    def _roll_card_rarity(self, room: str | None = None) -> str:
            room = room or self.current_node_symbol
            # The real game still consumes the card RNG rarity roll for boss
            # rewards before forcing the outcome to rare. Keeping that counter
            # aligned matters for every later card reward in the run.
            roll = int(self.randoms.card.random(99)) + self.card_rarity_factor
            if room == "BOSS":
                return "RARE"
            rare_chance = 10 if room in {"E", "E_GREEN", "ACT4_ELITE"} else 3
            uncommon_chance = 40 if room in {"E", "E_GREEN", "ACT4_ELITE"} else 37
            if room != "R" and self._has_relic("N'loth's Gift"):
                rare_chance *= 3
            if roll < rare_chance:
                return "RARE"
            if roll < rare_chance + uncommon_chance:
                return "UNCOMMON"
            return "COMMON"

    def _advance_card_rarity_factor_after_roll(self, rarity: str) -> None:
            if rarity == "COMMON":
                self.card_rarity_factor = max(self.card_rarity_factor - 1, -40)
            elif rarity == "RARE":
                self.card_rarity_factor = 5

    def _roll_card_reward(self, count: int = 3, *, room: str | None = None) -> list[CardInstance]:
            options: list[CardInstance] = []
            seen: set[str] = set()
            rarities: list[str] = []
            for _ in range(max(0, int(count))):
                rarity = self._roll_card_rarity(room)
                rarities.append(rarity)
                self._advance_card_rarity_factor_after_roll(rarity)
                rarity_pool = self._ironclad_card_pool(rarity=rarity) or self._ironclad_card_pool()
                chosen = self.randoms.card.choice(rarity_pool)
                if chosen.card_id in seen:
                    unseen_exists = any(card.card_id not in seen for card in rarity_pool)
                    while unseen_exists and chosen.card_id in seen:
                        chosen = self.randoms.card.choice(rarity_pool)
                seen.add(chosen.card_id)
                options.append(CardInstance(chosen, uuid=f"reward-{chosen.card_id}-{len(options)}"))
            upgraded_chance = self._upgraded_card_chance()
            for card, rarity in zip(options, rarities):
                if rarity != "RARE" and self.randoms.card.random_boolean(upgraded_chance):
                    # Lightspeed's reward-upgrade branch constructs Searing Blow
                    # with the generic upgraded flag instead of incrementing its
                    # special upgrade counter, so the card still serializes and
                    # behaves like an unupgraded reward there.
                    if card.card_id != "Searing Blow":
                        _ensure_card_upgraded(card)
                    continue
                if card.card_def.card_type == "ATTACK" and self._has_relic("Molten Egg"):
                    _ensure_card_upgraded(card)
                elif card.card_def.card_type == "SKILL" and self._has_relic("Toxic Egg"):
                    _ensure_card_upgraded(card)
                elif card.card_def.card_type == "POWER" and self._has_relic("Frozen Egg"):
                    _ensure_card_upgraded(card)
            return options

    def _roll_map_symbol(self, act: int, floor: int, start_floor: int, boss_floor: int, rng: StsRandom) -> str:
            if floor == boss_floor:
                return "BOSS"
            if floor == boss_floor - 1:
                return "R"
            relative = floor - start_floor
            if relative <= 1:
                return rng.choices(["M", "?", "$"], weights=[0.72, 0.22, 0.06], k=1)[0]
            if relative <= 3:
                return rng.choices(["M", "?", "$", "T"], weights=[0.54, 0.24, 0.12, 0.10], k=1)[0]
            return rng.choices(["M", "?", "$", "R", "T", "E"], weights=[0.42, 0.22, 0.11, 0.12, 0.04, 0.09], k=1)[0]

    def _roll_relic(self, *, shop_room: bool = False, from_front: bool = True, elite: bool = False) -> dict[str, Any]:
            if elite:
                roll = int(self.randoms.relic.random(99))
                tier = "COMMON" if roll < 50 else "RARE" if roll > 82 else "UNCOMMON"
            else:
                tier = self._roll_relic_tier_for_act(self.act)
            return self._roll_relic_of_tier(tier, shop_room=shop_room, from_front=from_front)

    def _roll_relic_of_tier(self, tier: str, *, shop_room: bool = False, from_front: bool = True) -> dict[str, Any]:
            return draw_relic_from_pool(
                self.relic_pools,
                tier,
                owned=self._owned_relic_ids(),
                floor=self.floor,
                shop_room=shop_room,
                from_front=from_front,
                deck=self.deck,
            )

    def _roll_relic_tier_for_act(self, act: int) -> str:
            common_chance = 0 if act == 4 else 50
            uncommon_chance = 100 if act == 4 else 33
            roll = int(self.randoms.relic.random(99))
            if roll < common_chance:
                return "COMMON"
            if roll < common_chance + uncommon_chance:
                return "UNCOMMON"
            return "RARE"

    def _roll_screenless_relic_of_tier(self, tier: str, *, shop_room: bool = False, from_front: bool = True) -> dict[str, Any]:
            blocked = {"Bottled Flame", "Bottled Lightning", "Bottled Tornado", "Whetstone"}
            while True:
                relic = self._roll_relic_of_tier(tier, shop_room=shop_room, from_front=from_front)
                if str(relic.get("relic_id") or "") not in blocked:
                    return relic

    def _roll_shop_card_rarity(self) -> str:
            roll = int(self.randoms.card.random(99)) + self.card_rarity_factor
            if roll < 9:
                return "RARE"
            if roll >= 46:
                return "COMMON"
            return "UNCOMMON"

    def _roll_shop_cards(self) -> tuple[list[tuple[CardInstance, str]], list[tuple[CardInstance, str]]]:
            def get_card_from_pool(card_type: str, rarity: str) -> CardInstance:
                pool = ironclad_type_rarity_card_pool(card_type, rarity, exclude_ids=self.locked_card_ids)
                if not pool:
                    pool = self._ironclad_card_pool(card_type=card_type, rarity=rarity)
                chosen = pool[int(self.randoms.card.random(len(pool) - 1))]
                return self._make_deck_card(chosen.card_id, uuid=f"shop-{chosen.card_id}-{self.floor}")

            def assign_random_card_excluding(card_type: str, exclude_id: str) -> tuple[CardInstance, str]:
                while True:
                    rarity = self._roll_shop_card_rarity()
                    card = get_card_from_pool(card_type, rarity)
                    if card.card_id != exclude_id:
                        return card, rarity

            attack_a_rarity = self._roll_shop_card_rarity()
            attack_a = get_card_from_pool("ATTACK", attack_a_rarity)
            attack_b, attack_b_rarity = assign_random_card_excluding("ATTACK", attack_a.card_id)
            skill_a_rarity = self._roll_shop_card_rarity()
            skill_a = get_card_from_pool("SKILL", skill_a_rarity)
            skill_b, skill_b_rarity = assign_random_card_excluding("SKILL", skill_a.card_id)
            power_rarity = self._roll_shop_card_rarity()
            if power_rarity == "COMMON":
                power_rarity = "UNCOMMON"
            power_card = get_card_from_pool("POWER", power_rarity)

            colorless_cards: list[tuple[CardInstance, str]] = []
            for rarity in ("UNCOMMON", "RARE"):
                colorless_pool = [
                    CARD_LIBRARY[card_id]
                    for card_id in COLORLESS_CARD_ID_ORDER
                    if CARD_LIBRARY[card_id].rarity == rarity
                ]
                if not colorless_pool and rarity == "UNCOMMON":
                    colorless_pool = [
                        CARD_LIBRARY[card_id]
                        for card_id in COLORLESS_CARD_ID_ORDER
                        if CARD_LIBRARY[card_id].rarity == "COMMON"
                    ]
                chosen = self.randoms.card.choice(colorless_pool)
                colorless_cards.append((self._make_deck_card(chosen.card_id, uuid=f"shop-colorless-{chosen.card_id}"), rarity))
            return [
                (attack_a, attack_a_rarity),
                (attack_b, attack_b_rarity),
                (skill_a, skill_a_rarity),
                (skill_b, skill_b_rarity),
                (power_card, power_rarity),
            ], colorless_cards

    def _set_cursed_tome_options(self) -> None:
            phase = int(self.event_state.get("phase", 0))
            if phase == 0:
                options = [(0, "Read"), (1, "Ignored")]
            elif phase in {1, 2, 3}:
                options = [(phase + 1, "Continue")]
            elif phase == 4:
                options = [(5, "Take Book"), (6, "Leave")]
            else:
                options = []
            self.event_options = [
                {"kind": "event", "event_id": "Cursed Tome", "name": label, "label": label, "choice_index": idx}
                for idx, label in options
            ]

    def _shop_card_price_base(self, card: CardInstance, *, colorless: bool = False, rarity_override: str | None = None) -> int:
            rarity = rarity_override or card.card_def.rarity
            base = {
                "COMMON": 50,
                "UNCOMMON": 75,
                "RARE": 150,
            }.get(rarity, 75)
            raw_price = _f32(float(base) * _f32(self.randoms.merchant.random(0.9, 1.1)))
            if colorless:
                raw_price = _f32(raw_price * _f32(1.2))
            return max(0, int(raw_price))

    def _shop_card_price(self, card: CardInstance, *, colorless: bool = False, rarity_override: str | None = None, sale: bool = False) -> int:
            price = self._shop_card_price_base(card, colorless=colorless, rarity_override=rarity_override)
            if sale:
                price //= 2
            return self._apply_shop_discount(price)

    def _shop_courier_restock_card_price(self, rarity: str, *, colorless: bool = False) -> int:
            base = {
                "COMMON": 50,
                "UNCOMMON": 75,
                "RARE": 150,
            }.get(rarity, 75)
            price = float(base) * float(self.randoms.merchant.random(0.9, 1.1))
            if colorless:
                price *= 1.2
            # Mirror lightspeed's current getNewCardPrice implementation, which
            # mistakenly checks The Courier twice instead of Membership Card.
            if self._has_relic("The Courier"):
                price *= 0.8
            if self._has_relic("The Courier"):
                price *= 0.5
            return int(price)

    def _shop_courier_restock_misc_price(self) -> int:
            # Mirror lightspeed's current getNewPrice implementation, which drops
            # the provided base price and discounts entirely.
            return int(round(self.randoms.merchant.random(0.95, 1.05)))

    def _shop_potion_price_base(self, potion: PotionInstance) -> int:
            base = {
                "COMMON": 50,
                "UNCOMMON": 75,
                "RARE": 100,
            }.get(potion.potion_def.rarity if potion.potion_def else "COMMON", 50)
            price_factor = _f32(self.randoms.merchant.random(_f32(0.95), _f32(1.05)))
            return max(0, _sts_round(_f32(float(base) * price_factor)))

    def _shop_potion_price(self, potion: PotionInstance) -> int:
            return self._apply_shop_discount(self._shop_potion_price_base(potion))

    def _shop_relic_price_base(self, relic: dict[str, Any], *, rarity_override: str | None = None) -> int:
            rarity = rarity_override or str(relic.get("tier") or relic.get("rarity") or "COMMON")
            base = {
                "COMMON": 150,
                "UNCOMMON": 250,
                "IRONCLAD": 250,
                "RARE": 300,
                "SHOP": 150,
                "EVENT": 250,
            }.get(rarity, 200)
            return max(0, _sts_round(_f32(float(base) * _f32(self.randoms.merchant.random(0.95, 1.05)))))

    def _shop_relic_price(self, relic: dict[str, Any], *, rarity_override: str | None = None) -> int:
            return self._apply_shop_discount(self._shop_relic_price_base(relic, rarity_override=rarity_override))

    def _start_event_combat(
            self,
            monster_ids: list[str],
            *,
            relic_id: str | None = None,
            gold_gain: int = 0,
            elite: bool = False,
        ) -> None:
            from spirecomm.native_sim_v2.env import NativeCombatEnv

            self.phase = "COMBAT"
            self.current_node_symbol = "EVENT_COMBAT"
            self.pending_event_relic_id = relic_id
            self.pending_event_gold = max(0, int(gold_gain))
            self.combat = NativeCombatEnv(
                seed=self.seed,
                ascension_level=self.ascension_level,
                floor=self.floor,
                act=self.act,
                act_boss=self.act_boss,
                elite=elite,
                external_misc_rng=self.randoms.misc,
                external_potion_rng=self.randoms.potion,
                scheduled_encounter=list(monster_ids),
                player=self.player,
                deck=[clone_card(card) for card in self.deck],
                relics=list(self.relics),
                potions=list(self.potions),
                gold=self.gold,
                locked_card_ids=set(self.locked_card_ids),
            )

    def _sync_card_rng_for_act_transition(self) -> None:
            target_counter = {2: 250, 3: 500, 4: 750}.get(self.act)
            if target_counter is not None and self.randoms.card.counter < target_counter:
                self.randoms.card.set_counter(target_counter)

    def _transformed_card_from_rng(self, rng: StsRandom, exclude_card_id: str) -> CardInstance:
            transform_pool = TRANSFORM_CARD_POOL_IRONCLAD
            exclude_in_pool = (
                exclude_card_id in transform_pool
                and CARD_LIBRARY[exclude_card_id].rarity != "BASIC"
            )
            if exclude_in_pool:
                index = int(rng.random(len(transform_pool) - 2))
                chosen = transform_pool[index + 1] if transform_pool[index] == exclude_card_id else transform_pool[index]
            else:
                chosen = transform_pool[int(rng.random(len(transform_pool) - 1))]
            return make_card(chosen, uuid=f"transform-{chosen}-{self.floor}")

    def _pandora_transformed_card_from_rng(self, rng: StsRandom, exclude_card_id: str) -> CardInstance:
            # Pandora's Box currently follows a lightspeed-specific hybrid pool:
            # the first 20 slots use combat-pool ordering, while the tail keeps
            # the normal transform ordering.
            transform_pool = COMBAT_CARD_POOL_IRONCLAD[:20] + TRANSFORM_CARD_POOL_IRONCLAD[20:]
            exclude_in_pool = (
                exclude_card_id in transform_pool
                and CARD_LIBRARY[exclude_card_id].rarity != "BASIC"
            )
            if exclude_in_pool:
                index = int(rng.random(len(transform_pool) - 2))
                chosen = transform_pool[index + 1] if transform_pool[index] == exclude_card_id else transform_pool[index]
            else:
                chosen = transform_pool[int(rng.random(len(transform_pool) - 1))]
            return make_card(chosen, uuid=f"transform-{chosen}-{self.floor}")

    def _transition_to_next_act(self) -> None:
            next_act = min(self.act + 1, 4)
            if next_act != self.act:
                self.act = next_act
                if next_act in {2, 3} and self.floor in {16, 33}:
                    # Preserve the real game's off-map transition floors so the
                    # next map screen targets floor 18/35 instead of a missing 17/34.
                    self.floor += 1
                self._sync_card_rng_for_act_transition()
                self._reset_act_random_room_state()
                self._generate_monster_schedules_for_act(self.act)
                if hasattr(self, "_act_bosses"):
                    self.act_boss = self._act_bosses.get(self.act, self.act_boss)
                self._generate_act_map(self.act)
                self.current_map_node_id = None
                missing_hp = max(0, self.player.max_hp - self.player.current_hp)
                if self.ascension_level >= 5:
                    self._heal_run(_sts_round(missing_hp * 0.75))
                else:
                    self._heal_run(self.player.max_hp)
            self._enter_map()

    def _upgraded_card_chance(self) -> float:
            if self.act < 2:
                return 0.0
            if self.act == 2:
                return 0.25 if self.ascension_level < 12 else 0.125
            return 0.50 if self.ascension_level < 12 else 0.25
