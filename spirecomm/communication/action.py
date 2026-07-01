import time

from spirecomm.spire.screen import ScreenType, RewardType


NEOW_REWARD_SETTLE_TIMEOUT_TICKS = 200


class Action:
    """A base class for an action to take in Slay the Spire"""

    def __init__(self, command="state", requires_game_ready=True):
        self.command = command
        self.requires_game_ready = requires_game_ready

    def can_be_executed(self, coordinator):
        """Indicates whether the given action can currently be executed, given the coordinator's state

        :param coordinator: The coordinator which will be used to execute the action
        :return: True if the action can currently be executed
        ":rtype: boolean
        """
        if self.requires_game_ready:
            return coordinator.game_is_ready
        else:
            return True

    def execute(self, coordinator):
        """Given the coordinator's current state, execute the given action

        :param coordinator: The coordinator which will be used to execute the action
        :return: None
        """
        coordinator.send_message(self.command)


class CommittedAction(Action):
    """Represents an action that has already been executed during planning.

    Execution becomes a no-op so the coordinator can continue from the committed
    branch state without sending the same game command twice.
    """

    def __init__(self, original_action):
        super().__init__(getattr(original_action, "command", "state"), requires_game_ready=False)
        self.original_action = original_action
        for attribute in [
            "card",
            "card_index",
            "target_monster",
            "target_index",
            "choice_index",
            "name",
            "potion",
            "potion_index",
            "use",
            "combat_reward",
        ]:
            if hasattr(original_action, attribute):
                setattr(self, attribute, getattr(original_action, attribute))

    def execute(self, coordinator):
        # The branch action has already been executed during planning. We still
        # need a fresh state push so the main loop can continue from the new
        # post-action screen (for example combat reward / map / game over).
        coordinator.send_message("state")


class RawCommandAction(Action):
    """Send an arbitrary command string to Communication Mod."""

    def __init__(self, command, requires_game_ready=True):
        super().__init__(command, requires_game_ready=requires_game_ready)


class StateAction(Action):
    """Request an immediate state push from Communication Mod."""

    def __init__(self):
        super().__init__("state", requires_game_ready=False)


class WaitAction(Action):
    """Ask Communication Mod to wait for a number of update ticks."""

    def __init__(self, timeout):
        super().__init__("wait {}".format(int(timeout)), requires_game_ready=False)


class ClickAction(Action):
    """Click a fixed UI coordinate through Communication Mod."""

    def __init__(self, button, x, y, timeout=100):
        command = "click {} {} {} {}".format(button, float(x), float(y), int(timeout))
        super().__init__(command, requires_game_ready=False)
        self.button = button
        self.x = float(x)
        self.y = float(y)
        self.timeout = int(timeout)


class PlayCardAction(Action):
    """An action to play a specified card from your hand"""

    def __init__(self, card=None, card_index=-1, target_monster=None, target_index=None):
        super().__init__("play")
        self.card = card
        self.card_index = card_index
        self.target_index = target_index
        self.target_monster = target_monster

    def can_be_executed(self, coordinator):
        if not super().can_be_executed(coordinator):
            return False
        game_state = coordinator.last_game_state
        if game_state is None or not game_state.play_available:
            return False

        if self.card is not None:
            if self.card not in game_state.hand:
                return False
            card = self.card
        else:
            if self.card_index < 0 or self.card_index >= len(game_state.hand):
                return False
            card = game_state.hand[self.card_index]

        return card.is_playable

    def execute(self, coordinator):
        if self.card is not None:
            self.card_index = coordinator.last_game_state.hand.index(self.card)
        if self.card_index == -1:
            raise Exception("Specified card for CardAction is not in hand")
        hand_card_index = self.card_index + 1
        if self.target_monster is not None:
            self.target_index = self.target_monster.monster_index
        if self.target_index is None:
            coordinator.send_message("{} {}".format(self.command, hand_card_index))
        else:
            coordinator.send_message("{} {} {}".format(self.command, hand_card_index, self.target_index))


class PotionAction(Action):
    """An action to use or discard a selected potion"""

    def __init__(self, use, potion=None, potion_index=-1, target_monster=None, target_index=None):
        super().__init__("potion")
        self.use = use
        self.potion = potion
        self.potion_index = potion_index
        self.target_monster = target_monster
        self.target_index = target_index

    def can_be_executed(self, coordinator):
        if not super().can_be_executed(coordinator):
            return False
        game_state = coordinator.last_game_state
        return game_state is not None and game_state.potion_available

    def execute(self, coordinator):
        if self.potion is not None:
            self.potion_index = coordinator.last_game_state.potions.index(self.potion)
        if self.potion_index == -1:
            raise Exception("Specified potion for PotionAction is not available")
        arguments = [self.command]
        if self.use:
            arguments.append("use")
        else:
            arguments.append("discard")
        arguments.append(str(self.potion_index))
        if self.target_monster is not None:
            self.target_index = self.target_monster.monster_index
        if self.target_index is not None:
            arguments.append(str(self.target_index))
        coordinator.send_message(" ".join(arguments))


class EndTurnAction(Action):
    """An action to end your turn"""

    def __init__(self):
        super().__init__("end")

    def can_be_executed(self, coordinator):
        if not super().can_be_executed(coordinator):
            return False
        game_state = coordinator.last_game_state
        if game_state is None or not game_state.end_available:
            return False
        # Hard-disable ending the turn while any playable card remains.
        if getattr(game_state, "in_combat", False):
            hand = getattr(game_state, "hand", None) or []
            if any(getattr(card, "is_playable", False) for card in hand):
                return False
        return True


class ProceedAction(Action):
    """An action to use the CommunicationMod 'Proceed' command"""

    def __init__(self):
        super().__init__("proceed")

    def can_be_executed(self, coordinator):
        if not super().can_be_executed(coordinator):
            return False
        game_state = coordinator.last_game_state
        return game_state is not None and game_state.proceed_available


class ConfirmAction(Action):
    """An action to confirm a card/grid selection through Communication Mod."""

    def __init__(self):
        # Confirmation often needs to fire while Communication Mod is still
        # marked non-ready from the preceding grid/hand select choose command.
        super().__init__("confirm", requires_game_ready=False)

    def can_be_executed(self, coordinator):
        if not super().can_be_executed(coordinator):
            return False
        game_state = coordinator.last_game_state
        if game_state is None:
            return False
        if game_state.screen_type == ScreenType.HAND_SELECT:
            return True
        if game_state.screen_type == ScreenType.GRID:
            screen = getattr(game_state, "screen", None)
            return bool(
                getattr(screen, "confirm_up", False)
                or getattr(screen, "is_just_for_confirming", False)
                or getattr(screen, "for_upgrade", False)
                or getattr(screen, "for_transform", False)
                or getattr(screen, "for_purge", False)
                or getattr(screen, "any_number", False)
            )
        return False


class CancelAction(Action):
    """An action to use the CommunicationMod 'Cancel' command"""

    def __init__(self):
        super().__init__("cancel")

    def can_be_executed(self, coordinator):
        if not super().can_be_executed(coordinator):
            return False
        game_state = coordinator.last_game_state
        return game_state is not None and game_state.cancel_available


class ChooseAction(Action):
    """An action to use the CommunicationMod 'Choose' command"""

    def __init__(self, choice_index=0, name=None):
        super().__init__("choose")
        self.choice_index = choice_index
        self.name = name

    def execute(self, coordinator):
        if self.name is not None:
            coordinator.send_message("{} {}".format(self.command, self.name))
        else:
            coordinator.send_message("{} {}".format(self.command, self.choice_index))


class NeowCardSelectAction(ChooseAction):
    """Choose a Neow card-selection card and force a fresh state push.

    Communication Mod surfaces Neow's remove/transform UI as an EVENT screen
    with cards, and the choose command does not always trigger an immediate
    serialized state update on its own. Use the same settle pulse as the Neow
    card reward bridge so replay waits out the event timer before requesting a
    fresh state.
    """

    def __init__(self, choice_index=0, name=None):
        super().__init__(choice_index=choice_index, name=name)

    def execute(self, coordinator):
        super().execute(coordinator)
        # Neow upgrade/remove grids do not expose a stable follow-up command
        # surface. Let replay detect the stale GRID frame and bridge it via the
        # dedicated timeout/finalize path instead of enqueueing extra actions
        # that keep the step alive without changing state.


class NeowCardRewardAction(ChooseAction):
    """Choose a Neow card reward.

    Keep this action scoped to the visible reward pick. The trace contains
    explicit follow-up Neow continue / map steps, and queueing them here can
    send `choose` into a transient map frame before nodes are selectable.
    """

    def __init__(
        self,
        choice_index=0,
        name=None,
        continue_choice_index=None,
        post_continue_choice_index=None,
        bridge_delay_seconds=0.0,
        continue_delay_seconds=0.0,
    ):
        super().__init__(choice_index=choice_index, name=name)
        self.continue_choice_index = None if continue_choice_index is None else int(continue_choice_index)
        self.post_continue_choice_index = (
            None if post_continue_choice_index is None else int(post_continue_choice_index)
        )
        self.bridge_delay_seconds = max(0.0, float(bridge_delay_seconds))
        self.continue_delay_seconds = max(0.0, float(continue_delay_seconds))

    def execute(self, coordinator):
        coordinator.send_message(f"{self.command} {int(self.choice_index)}")
        if self.bridge_delay_seconds > 0.0:
            time.sleep(self.bridge_delay_seconds)
        coordinator.send_message(f"wait {NEOW_REWARD_SETTLE_TIMEOUT_TICKS}")
        coordinator.send_message("state")
        if self.continue_choice_index is not None:
            if self.continue_delay_seconds > 0.0:
                time.sleep(self.continue_delay_seconds)
            coordinator.send_message(f"choose {self.continue_choice_index}")
            coordinator.send_message("wait 1")
            coordinator.send_message("state")
        if self.post_continue_choice_index is not None:
            coordinator.send_message(f"choose {self.post_continue_choice_index}")
            coordinator.send_message("wait 1")
            coordinator.send_message("state")


class NeowContinueAction(Action):
    """Advance a single-option Neow continue screen and request a fresh state.

    This bridge is used after the replay has already yielded one natural pulse
    to let Neow transitions advance. Keep the follow-up nudge small so we do
    not regress earlier Neow frames that are already stable.
    """

    def __init__(self, post_continue_choice_index=None, include_settle_probe=True):
        super().__init__("choose 0", requires_game_ready=False)
        self.post_continue_choice_index = (
            None if post_continue_choice_index is None else int(post_continue_choice_index)
        )
        self.include_settle_probe = bool(include_settle_probe)

    def execute(self, coordinator):
        coordinator.send_message("choose 0")
        if self.include_settle_probe:
            coordinator.send_message("wait 1")
            coordinator.send_message("state")
        if self.post_continue_choice_index is not None:
            coordinator.send_message(f"choose {self.post_continue_choice_index}")
            coordinator.send_message("wait 1")
            coordinator.send_message("state")


class ChooseShopkeeperAction(ChooseAction):
    """An action to open the shop on a shop screen"""

    def __init__(self):
        super().__init__(name="shop")


class OpenChestAction(ChooseAction):
    """An action to open a chest on a chest screen"""

    def __init__(self):
        super().__init__(name="open")


class BuyCardAction(ChooseAction):
    """An action to buy a card in a shop"""

    def __init__(self, card):
        super().__init__(name=card.name)


class BuyPotionAction(ChooseAction):
    """An action to buy a potion in a shop. Currently, buys the first available potion of the same name."""

    def __init__(self, potion):
        super().__init__(name=potion.name)

    def execute(self, coordinator):
        game_state = getattr(coordinator, "last_game_state", None)
        if game_state is not None and game_state.are_potions_full():
            coordinator.send_message("state")
            return
        super().execute(coordinator)


class BuyRelicAction(ChooseAction):
    """An action to buy a relic in a shop"""

    def __init__(self, relic):
        super().__init__(name=relic.name)


class BuyPurgeAction(Action):
    """An action to buy a card removal at a shop"""

    def __init__(self, card_to_purge=None):
        super().__init__()
        self.card_to_purge = card_to_purge

    def execute(self, coordinator):
        if coordinator.last_game_state.screen_type != ScreenType.SHOP_SCREEN:
            raise Exception("BuyPurgeAction is only available on a Shop Screen")
        coordinator.add_action_to_queue(ChooseAction(name="purge"))
        if self.card_to_purge is not None:
            coordinator.add_action_to_queue(CardSelectAction([self.card_to_purge]))


class EventOptionAction(ChooseAction):
    """An action to choose an event option"""

    def __init__(self, option):
        super().__init__(choice_index=option.choice_index)


class RestAction(ChooseAction):
    """An action to choose a rest option at a rest site"""

    def __init__(self, rest_option):
        super().__init__(name=rest_option.name)


class CardRewardAction(ChooseAction):
    """An action to choose a card reward, or use Singing Bowl"""

    def __init__(self, card=None, bowl=False):
        if bowl:
            name = "bowl"
        elif card is not None:
            name = card.name
        else:
            raise Exception("Must provide a card for CardRewardAction if not choosing the Singing Bowl")
        super().__init__(name=name)


class CombatRewardAction(ChooseAction):
    """An action to choose a combat reward"""

    def __init__(self, combat_reward):
        self.combat_reward = combat_reward
        super().__init__()

    def execute(self, coordinator):
        if coordinator.last_game_state.screen_type != ScreenType.COMBAT_REWARD:
            raise Exception("CombatRewardAction is only available on a Combat Reward Screen.")
        reward_list = coordinator.last_game_state.screen.rewards
        if self.combat_reward not in reward_list:
            raise Exception("Reward is not available: {}".format(self.combat_reward.reward_type))
        if self.combat_reward.reward_type == RewardType.POTION and coordinator.last_game_state.are_potions_full():
            raise Exception("Cannot choose potion reward with full potion slots.")
        self.choice_index = reward_list.index(self.combat_reward)
        super().execute(coordinator)


class BossRewardAction(ChooseAction):
    """An action to choose a boss relic"""

    def __init__(self, relic):
        super().__init__(name=relic.name)


class OptionalCardSelectConfirmAction(Action):
    """An action to click confirm on a hand or grid select screen, only if available"""

    def __init__(self):
        # Grid/hand-select follow-up bridges often run while Communication Mod
        # is still in a non-ready transition frame right after the initial
        # choose command. Keep this wrapper itself non-ready so it can decide
        # whether to enqueue a real confirm or just request fresh state.
        super().__init__(requires_game_ready=False)

    def execute(self, coordinator):
        screen_type = coordinator.last_game_state.screen_type
        if screen_type == ScreenType.HAND_SELECT:
            coordinator.add_action_to_queue(ConfirmAction())
        elif screen_type == ScreenType.GRID:
            # Communication Mod does not need an explicit confirm on Neow/grid
            # selects. Let the caller's queued settle wait/state drive the next
            # transition instead of injecting another bridge action here.
            return
        else:
            coordinator.add_action_to_queue(StateAction())


class CardSelectAction(Action):
    """An action to choose the selected cards on a hand or grid select screen"""

    def __init__(self, cards):
        self.cards = cards
        super().__init__()

    def execute(self, coordinator):
        screen_type = coordinator.last_game_state.screen_type
        screen = coordinator.last_game_state.screen
        if screen_type not in [ScreenType.HAND_SELECT, ScreenType.GRID]:
            raise Exception("CardSelectAction is only available on a Hand Select or Grid Select Screen.")
        num_selected_cards = len(screen.selected_cards)
        num_remaining_cards = screen.num_cards - num_selected_cards
        available_cards = screen.cards
        if screen_type == ScreenType.GRID and not screen.any_number and len(self.cards) != num_remaining_cards:
            raise Exception("Wrong number of cards selected for CardSelectAction (provided {}, need {})".format(len(self.cards), num_remaining_cards))
        elif len(self.cards) > num_remaining_cards:
            raise Exception("Too many cards selected for CardSelectAction (provided {}, max {})".format(len(self.cards), num_remaining_cards))
        chosen_indices = []
        for card in self.cards:
            if card not in available_cards:
                raise Exception("Card {} is not available in the Hand Select Screen".format(card.name))
            else:
                chosen_indices.append(available_cards.index(card))
        chosen_indices.sort(reverse=True)
        for index in chosen_indices:
            coordinator.add_action_to_queue(ChooseAction(choice_index=index))
        coordinator.add_action_to_queue(OptionalCardSelectConfirmAction())


class ChooseMapNodeAction(ChooseAction):
    """An action to choose a map node, other than the boss"""

    def __init__(self, node):
        self.node = node
        super().__init__()

    def execute(self, coordinator):
        if coordinator.last_game_state.screen_type != ScreenType.MAP:
            raise Exception("MapChoiceAction is only available on a Map Screen")
        next_nodes = coordinator.last_game_state.screen.next_nodes
        if self.node not in next_nodes:
            raise Exception("Node {} is not available to choose.".format(self.node))
        self.choice_index = next_nodes.index(self.node)
        super().execute(coordinator)


class ChooseMapBossAction(ChooseAction):
    """An action to choose the boss map node"""

    def __init__(self):
        super().__init__()

    def execute(self, coordinator):
        if coordinator.last_game_state.screen_type != ScreenType.MAP:
            raise Exception("ChooseMapBossAction is only available on a Map Screen")
        if not coordinator.last_game_state.screen.boss_available:
            raise Exception("The boss is not available to choose.")
        self.name = "boss"
        super().execute(coordinator)


class StartGameAction(Action):
    """An action to start a new game, if not already in a game"""

    def __init__(self, player_class, ascension_level=0, seed=None):
        super().__init__("start")
        self.player_class = player_class
        self.ascension_level = ascension_level
        self.seed = seed

    def execute(self, coordinator):
        arguments = [self.command, self.player_class.name, str(self.ascension_level)]
        if self.seed is not None:
            arguments.append(str(self.seed))
        coordinator.send_message(" ".join(arguments))


class StateAction(Action):
    """An action to use the CommunicationMod 'State' command"""

    def __init__(self, requires_game_ready=False):
        super().__init__(command="state", requires_game_ready=False)
