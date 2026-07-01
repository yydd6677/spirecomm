import time
import random

from spirecomm.spire.game import Game
from spirecomm.spire.character import Intent, PlayerClass
import spirecomm.spire.card
from spirecomm.spire.screen import RestOption, ScreenType
from spirecomm.communication.action import *
from spirecomm.ai.priorities import *
from spirecomm.ai.card_reward_model import CardRewardSelector, PurgeTargetSelector, UpgradeTargetSelector
from spirecomm.ai.run_choice_model import (
    BossRelicSelector,
    EventChoiceSelector,
    CampfireChoiceSelector,
    MapChoiceSelector,
    ShopChoiceSelector,
    PotionUseSelector,
)
from spirecomm.ai.policy import RuleBasedCombatPolicy


class SimpleAgent:

    def __init__(self, chosen_class=PlayerClass.THE_SILENT):
        self.game = Game()
        self.errors = 0
        self.choose_good_card = False
        self.skipped_cards = False
        self.visited_shop = False
        self.map_route = []
        self.chosen_class = chosen_class
        self.priorities = Priority()
        self.card_reward_selector = CardRewardSelector()
        self.boss_relic_selector = BossRelicSelector()
        self.event_choice_selector = EventChoiceSelector()
        self.campfire_choice_selector = CampfireChoiceSelector()
        self.map_choice_selector = MapChoiceSelector()
        self.shop_choice_selector = ShopChoiceSelector()
        self.potion_use_selector = PotionUseSelector()
        self.upgrade_target_selector = UpgradeTargetSelector()
        self.purge_target_selector = PurgeTargetSelector()
        self.change_class(chosen_class)

    def change_class(self, new_class):
        self.chosen_class = new_class
        if self.chosen_class == PlayerClass.THE_SILENT:
            self.priorities = SilentPriority()
        elif self.chosen_class == PlayerClass.IRONCLAD:
            self.priorities = IroncladPriority()
        elif self.chosen_class == PlayerClass.DEFECT:
            self.priorities = DefectPowerPriority()
        else:
            self.priorities = random.choice(list(PlayerClass))

    def handle_error(self, error):
        if error is not None and "Invalid command:" in error:
            return StateAction()
        raise Exception(error)

    def get_next_action_in_game(self, game_state):
        self.game = game_state
        #time.sleep(0.07)
        if self.is_ftue_overlay():
            return RawCommandAction("key confirm 100", requires_game_ready=False)
        if self.game.choice_available:
            return self.handle_screen()
        if self.game.proceed_available:
            return ProceedAction()
        if self.game.potion_available and not self.game.choice_available:
            potion_action = self.choose_potion_action()
            if potion_action is not None:
                return potion_action
        if self.game.play_available:
            return self.get_play_card_action()
        if self.game.end_available:
            return EndTurnAction()
        if self.game.cancel_available:
            return CancelAction()

    def is_ftue_overlay(self):
        return self.game.in_combat and getattr(self.game, "screen_name", None) == "FTUE"

    def get_next_action_out_of_game(self):
        return StartGameAction(self.chosen_class)

    def is_monster_attacking(self):
        for monster in self.game.monsters:
            if monster.intent.is_attack() or monster.intent == Intent.NONE:
                return True
        return False

    def get_incoming_damage(self):
        incoming_damage = 0
        for monster in self.game.monsters:
            if not monster.is_gone and not monster.half_dead:
                if monster.move_adjusted_damage is not None:
                    incoming_damage += monster.move_adjusted_damage * monster.move_hits
                elif monster.intent == Intent.NONE:
                    incoming_damage += 5 * self.game.act
        return incoming_damage

    def get_low_hp_target(self):
        available_monsters = [monster for monster in self.game.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        best_monster = min(available_monsters, key=lambda x: x.current_hp)
        return best_monster

    def get_high_hp_target(self):
        available_monsters = [monster for monster in self.game.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        best_monster = max(available_monsters, key=lambda x: x.current_hp)
        return best_monster

    def many_monsters_alive(self):
        available_monsters = [monster for monster in self.game.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        return len(available_monsters) > 1

    def get_play_card_action(self):
        playable_cards = [card for card in self.game.hand if card.is_playable]
        zero_cost_cards = [card for card in playable_cards if card.cost == 0]
        zero_cost_attacks = [card for card in zero_cost_cards if card.type == spirecomm.spire.card.CardType.ATTACK]
        zero_cost_non_attacks = [card for card in zero_cost_cards if card.type != spirecomm.spire.card.CardType.ATTACK]
        nonzero_cost_cards = [card for card in playable_cards if card.cost != 0]
        aoe_cards = [card for card in playable_cards if self.priorities.is_card_aoe(card)]
        if self.game.player.block > self.get_incoming_damage() - (self.game.act + 4):
            offensive_cards = [card for card in nonzero_cost_cards if not self.priorities.is_card_defensive(card)]
            if len(offensive_cards) > 0:
                nonzero_cost_cards = offensive_cards
            else:
                nonzero_cost_cards = [card for card in nonzero_cost_cards if not card.exhausts]
        if len(playable_cards) == 0:
            return EndTurnAction()
        if len(zero_cost_non_attacks) > 0:
            card_to_play = self.priorities.get_best_card_to_play(zero_cost_non_attacks)
        elif len(nonzero_cost_cards) > 0:
            card_to_play = self.priorities.get_best_card_to_play(nonzero_cost_cards)
            if len(aoe_cards) > 0 and self.many_monsters_alive() and card_to_play.type == spirecomm.spire.card.CardType.ATTACK:
                card_to_play = self.priorities.get_best_card_to_play(aoe_cards)
        elif len(zero_cost_attacks) > 0:
            card_to_play = self.priorities.get_best_card_to_play(zero_cost_attacks)
        else:
            # This shouldn't happen!
            return EndTurnAction()
        if card_to_play.has_target:
            available_monsters = [monster for monster in self.game.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
            if len(available_monsters) == 0:
                return EndTurnAction()
            if card_to_play.type == spirecomm.spire.card.CardType.ATTACK:
                target = self.get_low_hp_target()
            else:
                target = self.get_high_hp_target()
            return PlayCardAction(card=card_to_play, target_monster=target)
        else:
            return PlayCardAction(card=card_to_play)

    def use_next_potion(self):
        for potion in self.game.get_real_potions():
            if potion.can_use:
                if potion.requires_target:
                    return PotionAction(True, potion=potion, target_monster=self.get_low_hp_target())
                else:
                    return PotionAction(True, potion=potion)

    def choose_potion_action(self):
        usable_potions = [potion for potion in self.game.get_real_potions() if potion.can_use]
        if not usable_potions:
            return None
        if (
            self.chosen_class == PlayerClass.IRONCLAD
            and self.potion_use_selector is not None
            and self.potion_use_selector.available
        ):
            candidates = []
            for potion in usable_potions:
                candidates.append(
                    {
                        "name": "USE_" + potion.potion_id,
                        "action": "use",
                        "potion_id": potion.potion_id,
                        "item_id": potion.potion_id,
                    }
                )
            candidates.append({"name": "HOLD", "action": "hold", "potion_id": "hold", "item_id": "hold"})
            selection = self.potion_use_selector.choose(self.game, candidates)
            if selection is not None:
                choice_index = selection.get("choice_index")
                if choice_index is not None and 0 <= choice_index < len(usable_potions):
                    potion = usable_potions[choice_index]
                    if potion.requires_target:
                        available_monsters = [
                            monster for monster in self.game.monsters
                            if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone
                        ]
                        if not available_monsters:
                            return None
                        return PotionAction(True, potion=potion, target_monster=self.get_low_hp_target())
                    return PotionAction(True, potion=potion)
                return None
        if self.game.room_type == "MonsterRoomBoss":
            return self.use_next_potion()
        return None

    def handle_screen(self):
        if self.game.screen_type == ScreenType.EVENT:
            is_neow_event = getattr(self.game.screen, "event_id", "") == "Neow" or getattr(self.game.screen, "event_name", "") == "Neow"
            if is_neow_event and len(self.game.screen.options) == 2:
                return ChooseAction(1)
            learned_choice = self.choose_event_option()
            if learned_choice is not None:
                return learned_choice
            if self.game.screen.event_id in ["Vampires", "Masked Bandits", "Knowing Skull", "Ghosts", "Liars Game", "Golden Idol", "Drug Dealer", "The Library"]:
                return ChooseAction(len(self.game.screen.options) - 1)
            else:
                return ChooseAction(0)
        elif self.game.screen_type == ScreenType.CHEST:
            return OpenChestAction()
        elif self.game.screen_type == ScreenType.SHOP_ROOM:
            if not self.visited_shop:
                self.visited_shop = True
                return ChooseShopkeeperAction()
            else:
                self.visited_shop = False
                return ProceedAction()
        elif self.game.screen_type == ScreenType.REST:
            return self.choose_rest_option()
        elif self.game.screen_type == ScreenType.CARD_REWARD:
            return self.choose_card_reward()
        elif self.game.screen_type == ScreenType.COMBAT_REWARD:
            for reward_item in self.game.screen.rewards:
                if reward_item.reward_type == RewardType.POTION and self.game.are_potions_full():
                    continue
                elif reward_item.reward_type == RewardType.CARD and self.skipped_cards:
                    continue
                else:
                    return CombatRewardAction(reward_item)
            self.skipped_cards = False
            return ProceedAction()
        elif self.game.screen_type == ScreenType.MAP:
            return self.make_map_choice()
        elif self.game.screen_type == ScreenType.BOSS_REWARD:
            relics = self.game.screen.relics
            learned_choice = self.choose_boss_relic(relics)
            if learned_choice is not None:
                return learned_choice
            best_boss_relic = self.priorities.get_best_boss_relic(relics)
            return BossRewardAction(best_boss_relic)
        elif self.game.screen_type == ScreenType.SHOP_SCREEN:
            learned_choice = self.choose_shop_option()
            if learned_choice is not None:
                return learned_choice
            if self.game.screen.purge_available and self.game.gold >= self.game.screen.purge_cost:
                return ChooseAction(name="purge")
            for card in self.game.screen.cards:
                if self.game.gold >= card.price and not self.priorities.should_skip(card):
                    return BuyCardAction(card)
            for relic in self.game.screen.relics:
                if self.game.gold >= relic.price:
                    return BuyRelicAction(relic)
            return CancelAction()
        elif self.game.screen_type == ScreenType.GRID:
            return self.choose_grid_cards()
        elif self.game.screen_type == ScreenType.HAND_SELECT:
            if not self.game.choice_available:
                return ProceedAction()
            # Usually, we don't want to choose the whole hand for a hand select. 3 seems like a good compromise.
            num_cards = min(self.game.screen.num_cards, 3)
            return CardSelectAction(self.priorities.get_cards_for_action(self.game.current_action, self.game.screen.cards, num_cards))
        else:
            return ProceedAction()

    def choose_rest_option(self):
        rest_options = self.game.screen.rest_options
        if len(rest_options) > 0 and not self.game.screen.has_rested:
            learned_choice = self.choose_campfire_option(rest_options)
            if learned_choice is not None:
                return learned_choice
            if RestOption.REST in rest_options and self.game.current_hp < self.game.max_hp / 2:
                return RestAction(RestOption.REST)
            elif RestOption.REST in rest_options and self.game.act != 1 and self.game.floor % 17 == 15 and self.game.current_hp < self.game.max_hp * 0.9:
                return RestAction(RestOption.REST)
            elif RestOption.SMITH in rest_options:
                return RestAction(RestOption.SMITH)
            elif RestOption.LIFT in rest_options:
                return RestAction(RestOption.LIFT)
            elif RestOption.DIG in rest_options:
                return RestAction(RestOption.DIG)
            elif RestOption.REST in rest_options and self.game.current_hp < self.game.max_hp:
                return RestAction(RestOption.REST)
            else:
                return ChooseAction(0)
        else:
            return ProceedAction()

    def choose_event_option(self):
        if (
            self.chosen_class != PlayerClass.IRONCLAD
            or self.event_choice_selector is None
            or not self.event_choice_selector.available
        ):
            return None
        candidates = []
        for option_index, option in enumerate(self.game.screen.options):
            if getattr(option, "disabled", False):
                continue
            choice_index = option.choice_index if option.choice_index is not None else option_index
            candidates.append(
                {
                    "event_id": getattr(self.game.screen, "event_id", "") or getattr(self.game.screen, "event_name", ""),
                    "event_name": getattr(self.game.screen, "event_name", "") or getattr(self.game.screen, "event_id", ""),
                    "label": option.label or option.text or str(choice_index),
                    "text": option.text or option.label or str(choice_index),
                    "choice_index": choice_index,
                }
            )
        if not candidates:
            return None
        selection = self.event_choice_selector.choose(self.game, candidates)
        if selection is None:
            return None
        choice_index = selection.get("choice_index")
        if choice_index is None or choice_index < 0 or choice_index >= len(candidates):
            return None
        return ChooseAction(candidates[choice_index]["choice_index"])

    def choose_boss_relic(self, relics):
        if (
            self.chosen_class != PlayerClass.IRONCLAD
            or self.boss_relic_selector is None
            or not self.boss_relic_selector.available
            or not relics
        ):
            return None
        selection = self.boss_relic_selector.choose(self.game, relics)
        if selection is None:
            return None
        choice_index = selection.get("choice_index")
        if choice_index is None or choice_index < 0 or choice_index >= len(relics):
            return None
        return BossRewardAction(relics[choice_index])

    def choose_campfire_option(self, rest_options):
        if (
            self.chosen_class != PlayerClass.IRONCLAD
            or self.campfire_choice_selector is None
            or not self.campfire_choice_selector.available
            or not rest_options
        ):
            return None
        selection = self.campfire_choice_selector.choose(self.game, rest_options)
        if selection is None:
            return None
        choice_index = selection.get("choice_index")
        if choice_index is None or choice_index < 0 or choice_index >= len(rest_options):
            return None
        return RestAction(rest_options[choice_index])

    def choose_shop_option(self):
        if (
            self.chosen_class != PlayerClass.IRONCLAD
            or self.shop_choice_selector is None
            or not self.shop_choice_selector.available
        ):
            return None
        candidates = []
        actions = []
        for card in self.game.screen.cards:
            if self.game.gold >= card.price:
                candidates.append(
                    {
                        "name": "BUY_" + card.card_id,
                        "action": "buy",
                        "item_kind": "card",
                        "item_id": card.card_id,
                    }
                )
                actions.append(BuyCardAction(card))
        for relic in self.game.screen.relics:
            if self.game.gold >= relic.price:
                candidates.append(
                    {
                        "name": "BUY_" + relic.relic_id,
                        "action": "buy",
                        "item_kind": "relic",
                        "item_id": relic.relic_id,
                    }
                )
                actions.append(BuyRelicAction(relic))
        for potion in self.game.screen.potions:
            if self.game.gold >= potion.price and not self.game.are_potions_full():
                candidates.append(
                    {
                        "name": "BUY_" + potion.name,
                        "action": "buy",
                        "item_kind": "potion",
                        "item_id": potion.name,
                    }
                )
                actions.append(BuyPotionAction(potion))
        if self.game.screen.purge_available and self.game.gold >= self.game.screen.purge_cost:
            candidates.append({"name": "PURGE", "action": "purge", "item_kind": "purge", "item_id": "purge"})
            actions.append(ChooseAction(name="purge"))
        candidates.append({"name": "LEAVE", "action": "leave", "item_kind": "leave", "item_id": "leave"})
        actions.append(CancelAction())

        selection = self.shop_choice_selector.choose(self.game, candidates)
        if selection is None:
            return None
        choice_index = selection.get("choice_index")
        if choice_index is None or choice_index < 0 or choice_index >= len(actions):
            return None
        return actions[choice_index]

    def choose_grid_cards(self):
        if not self.game.choice_available:
            return ProceedAction()
        screen = self.game.screen
        cards = list(screen.cards or [])
        if not cards:
            return ProceedAction()
        if self.chosen_class == PlayerClass.IRONCLAD:
            selector = None
            if screen.for_upgrade:
                selector = self.upgrade_target_selector
            elif screen.for_purge:
                selector = self.purge_target_selector
            if selector is not None and selector.available:
                selection = selector.choose(self.game, cards)
                if selection is not None:
                    scores = selection.get("scores") or []
                    if len(scores) == len(cards):
                        num_cards = max(1, int(screen.num_cards or 1))
                        ranked_indices = sorted(range(len(cards)), key=lambda index: scores[index], reverse=True)
                        return CardSelectAction([cards[index] for index in ranked_indices[:num_cards]])
                    choice_index = selection.get("choice_index")
                    if choice_index is not None and 0 <= choice_index < len(cards):
                        return CardSelectAction([cards[choice_index]])
        if screen.for_upgrade or self.choose_good_card:
            available_cards = self.priorities.get_sorted_cards(cards)
        else:
            available_cards = self.priorities.get_sorted_cards(cards, reverse=True)
        num_cards = self.game.screen.num_cards
        return CardSelectAction(available_cards[:num_cards])

    def count_copies_in_deck(self, card):
        count = 0
        for deck_card in self.game.deck:
            if deck_card.card_id == card.card_id:
                count += 1
        return count

    def choose_card_reward(self):
        reward_cards = self.game.screen.cards
        if (
            self.chosen_class == PlayerClass.IRONCLAD
            and self.card_reward_selector is not None
            and self.card_reward_selector.available
            and len(reward_cards) > 0
        ):
            selection = self.card_reward_selector.choose(
                self.game,
                reward_cards,
                can_skip=self.game.screen.can_skip,
            )
            if selection is not None:
                choice_index = selection.get("choice_index")
                if choice_index == 3 and self.game.screen.can_bowl:
                    return CardRewardAction(bowl=True)
                if choice_index == 3 and self.game.screen.can_skip:
                    self.skipped_cards = True
                    return CancelAction()
                if choice_index is not None and 0 <= choice_index < len(reward_cards):
                    return CardRewardAction(reward_cards[choice_index])
        if self.game.screen.can_skip and not self.game.in_combat:
            pickable_cards = [card for card in reward_cards if self.priorities.needs_more_copies(card, self.count_copies_in_deck(card))]
        else:
            pickable_cards = reward_cards
        if len(pickable_cards) > 0:
            potential_pick = self.priorities.get_best_card(pickable_cards)
            return CardRewardAction(potential_pick)
        elif self.game.screen.can_bowl:
            return CardRewardAction(bowl=True)
        else:
            self.skipped_cards = True
            return CancelAction()

    def generate_map_route(self):
        node_rewards = self.priorities.MAP_NODE_PRIORITIES.get(self.game.act)
        best_rewards = {0: {node.x: node_rewards[node.symbol] for node in self.game.map.nodes[0].values()}}
        best_parents = {0: {node.x: 0 for node in self.game.map.nodes[0].values()}}
        min_reward = min(node_rewards.values())
        map_height = max(self.game.map.nodes.keys())
        for y in range(0, map_height):
            best_rewards[y+1] = {node.x: min_reward * 20 for node in self.game.map.nodes[y+1].values()}
            best_parents[y+1] = {node.x: -1 for node in self.game.map.nodes[y+1].values()}
            for x in best_rewards[y]:
                node = self.game.map.get_node(x, y)
                best_node_reward = best_rewards[y][x]
                for child in node.children:
                    test_child_reward = best_node_reward + node_rewards[child.symbol]
                    if test_child_reward > best_rewards[y+1][child.x]:
                        best_rewards[y+1][child.x] = test_child_reward
                        best_parents[y+1][child.x] = node.x
        best_path = [0] * (map_height + 1)
        best_path[map_height] = max(best_rewards[map_height].keys(), key=lambda x: best_rewards[map_height][x])
        for y in range(map_height, 0, -1):
            best_path[y - 1] = best_parents[y][best_path[y]]
        self.map_route = best_path

    def make_map_choice(self):
        if (
            self.chosen_class == PlayerClass.IRONCLAD
            and self.map_choice_selector is not None
            and self.map_choice_selector.available
            and len(self.game.screen.next_nodes) > 0
        ):
            if self.game.screen.boss_available:
                return ChooseMapBossAction()
            selection = self.map_choice_selector.choose(self.game, self.game.screen.next_nodes)
            if selection is not None:
                choice_index = selection.get("choice_index")
                if choice_index is not None and 0 <= choice_index < len(self.game.screen.next_nodes):
                    return ChooseMapNodeAction(self.game.screen.next_nodes[choice_index])
        if len(self.game.screen.next_nodes) > 0 and self.game.screen.next_nodes[0].y == 0:
            self.generate_map_route()
            self.game.screen.current_node.y = -1
        if self.game.screen.boss_available:
            return ChooseMapBossAction()
        chosen_x = self.map_route[self.game.screen.current_node.y + 1]
        for choice in self.game.screen.next_nodes:
            if choice.x == chosen_x:
                return ChooseMapNodeAction(choice)
        # This should never happen
        return ChooseAction(0)


class HybridAgent:

    def __init__(self, chosen_class=PlayerClass.THE_SILENT, combat_policy=None, recorder=None):
        self.fallback_agent = SimpleAgent(chosen_class=chosen_class)
        self.fallback_agent.recorder = recorder
        self.combat_policy = combat_policy or RuleBasedCombatPolicy()
        self.recorder = recorder
        self.coordinator = None

    def change_class(self, new_class):
        self.fallback_agent.change_class(new_class)

    def handle_error(self, error):
        return self.fallback_agent.handle_error(error)

    def get_next_action_out_of_game(self):
        return self.fallback_agent.get_next_action_out_of_game()

    def set_coordinator(self, coordinator):
        self.coordinator = coordinator

    def get_next_action_in_game(self, game_state):
        self.fallback_agent.game = game_state
        if self.recorder is not None:
            self.recorder.on_state(game_state)
        source = "fallback"

        if game_state.potion_available and not game_state.choice_available:
            potion_action = self.fallback_agent.choose_potion_action()
            if potion_action is not None:
                action = potion_action
                source = "PotionUseSelector"
                if self.recorder is not None:
                    self.recorder.record_step(game_state, action, source)
                return action

        if game_state.play_available and not game_state.choice_available:
            action = self.combat_policy.choose_action(
                game_state,
                self.fallback_agent,
                coordinator=self.coordinator,
            )
            source = getattr(self.combat_policy, "source_name", self.combat_policy.__class__.__name__)
        else:
            action = self.fallback_agent.get_next_action_in_game(game_state)

        if action is None:
            action = StateAction()
            source = "fallback_state_refresh"

        if self.recorder is not None:
            self.recorder.record_step(game_state, action, source)
        return action

    def on_game_start(self, player_class, ascension_level=0, seed=None):
        if self.recorder is not None:
            self.recorder.start_run(player_class, ascension_level=ascension_level, seed=seed)

    def on_game_end(self, victory):
        if self.recorder is not None:
            self.recorder.end_run(victory)

    def reload_combat_policy(self):
        if hasattr(self.combat_policy, "reload"):
            self.combat_policy.reload()
