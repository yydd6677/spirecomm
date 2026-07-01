from __future__ import annotations

import math
from typing import Any

from spirecomm.native_sim.potions import get_random_potion, make_potion
from spirecomm.native_sim.randoms import java_collections_shuffle
from spirecomm.native_sim.schema import CardInstance, PotionInstance
from spirecomm.native_sim_v2.helpers_cards import CARD_LIBRARY, COLORLESS_CARD_ID_ORDER, card_to_spirecomm, clone_card, make_card
from spirecomm.native_sim_v2.helpers_relics import make_relic
from spirecomm.native_sim_v2.helpers_common import (
    _card_can_transform,
    _card_can_upgrade,
    _ensure_card_upgraded,
    _increment_card_upgrade,
)
from spirecomm.native_sim_v2.randoms import StsRandom
from spirecomm.native_sim_v2.serialize import run_state


def _event_percent_hp_loss(env, ratio: float, *, mode: str = "floor") -> int:
    amount = env.player.max_hp * ratio
    if mode == "round":
        return max(1, math.floor(amount + 0.5))
    if mode == "ceil":
        return max(1, math.ceil(amount))
    return max(1, int(amount))


def state(env: Any) -> dict[str, Any]:
    return run_state(env)


def legal_actions(env: Any) -> list[dict[str, Any]]:
    if env.phase == "COMBAT":
        if env.combat.outcome != "UNDECIDED":
            return [{"kind": "end", "name": "RESOLVE_COMBAT", "action_index": 0, "bits": 0}]
        return env.combat.legal_actions()
    if env.phase == "NEOW":
        return list(env.neow_options)
    if env.phase == "CARD_REWARD":
        actions: list[dict[str, Any]] = []
        flat_index = 0
        for reward_index, bundle in enumerate(env.reward_card_bundles):
            for card_index, card in enumerate(bundle):
                actions.append({
                    "kind": "card_reward",
                    "name": card.name,
                    "card_id": card.card_id,
                    "choice_index": flat_index,
                    "reward_index": reward_index,
                    "card_index": card_index,
                    "card": card_to_spirecomm(card),
                })
                flat_index += 1
        for index, relic in enumerate(env.reward_relics):
            relic_id = relic.get("relic_id")
            action_relic_id = "Paper Phrog" if relic_id == "Paper Frog" else relic_id
            actions.append({
                "kind": "reward_relic",
                "name": str(relic.get("name") or relic.get("relic_id") or "RELIC"),
                "relic_id": action_relic_id,
                "choice_index": len(actions),
                "reward_index": index,
            })
        for index, potion in enumerate(env.reward_potions):
            if potion.can_use:
                actions.append({
                    "kind": "reward_potion",
                    "name": potion.name,
                    "potion_id": potion.potion_id,
                    "choice_index": len(actions),
                    "reward_index": index,
                })
        for index, amount in enumerate(env.reward_gold_piles):
            actions.append({
                "kind": "reward_gold",
                "name": "GOLD",
                "choice_index": len(actions),
                "amount": amount,
                "reward_index": index,
            })
        if env.reward_emerald_key:
            actions.append({
                "kind": "reward_key",
                "name": "KEY",
                "choice_index": len(actions),
                "key": "emerald",
            })
        actions.append({"kind": "skip", "name": "SKIP", "choice_index": len(actions)})
        return actions
    if env.phase == "CARD_SELECT":
        return list(env.card_select_options)
    if env.phase == "BOSS_RELIC":
        actions = list(env.boss_relic_options)
        actions.append({
            "kind": "boss_relic",
            "name": "SKIP",
            "relic_id": "SKIP",
            "choice_index": len(actions),
        })
        return actions
    if env.phase == "MAP":
        return list(env.map_options)
    if env.phase == "CAMPFIRE":
        return list(env.campfire_options)
    if env.phase == "SHOP":
        return [item for item in env.shop_items if item.get("item_kind") == "leave" or env.gold >= int(item.get("price", 0))]
    if env.phase == "EVENT":
        return list(env.event_options)
    if env.phase == "TREASURE":
        return list(env.treasure_options)
    if env.phase == "CHEST":
        return list(env.chest_options)
    return []


def start_combat(env: Any, *, elite: bool = False) -> None:
    from spirecomm.native_sim_v2.env import NativeCombatEnv

    env.phase = "COMBAT"
    if elite and env.current_node_symbol not in {"E_GREEN", "ACT4_ELITE"}:
        env.current_node_symbol = "E"
    scheduled_encounter: list[str] | str | None = None
    if env.floor == 53:
        scheduled_encounter = ["SpireShield", "SpireSpear"]
    elif env.floor == 54:
        scheduled_encounter = ["CorruptHeart"]
    elif env.floor in {16, 33, 50}:
        scheduled_encounter = {
            "Hexaghost": "Hexaghost",
            "Slime Boss": "SlimeBoss",
            "The Guardian": "TheGuardian",
            "The Champ": "TheChamp",
            "The Collector": "TheCollector",
            "Bronze Automaton": "BronzeAutomaton",
            "Awakened One": "AwakenedOne",
            "Time Eater": "TimeEater",
            "Donu and Deca": "DonuDeca",
        }.get(env.act_boss)
    elif elite:
        scheduled_encounter = env._next_elite_encounter()
    else:
        scheduled_encounter = env._next_monster_encounter()
    env.player.powers = {}
    env.player.block = 0
    env.combat = NativeCombatEnv(
        seed=env.seed,
        ascension_level=env.ascension_level,
        floor=env.floor,
        act=env.act,
        act_boss=env.act_boss,
        elite=elite,
        external_misc_rng=env.randoms.misc,
        external_potion_rng=env.randoms.potion,
        scheduled_encounter=scheduled_encounter,
        player=env.player,
        deck=[clone_card(card) for card in env.deck],
        relics=list(env.relics),
        potions=list(env.potions),
        gold=env.gold,
        locked_card_ids=set(env.locked_card_ids),
    )
    env._apply_burning_elite_buff()


def start_event_boss_combat(env: Any, *, act_boss: str | None = None) -> None:
    from spirecomm.native_sim_v2.env import NativeCombatEnv

    env.phase = "COMBAT"
    env.current_node_symbol = "E"
    boss = act_boss or env.randoms.monster.choice(["Hexaghost", "Slime Boss", "The Guardian"])
    env.combat = NativeCombatEnv(
        seed=env.seed,
        ascension_level=env.ascension_level,
        floor=16,
        act=1,
        act_boss=boss,
        elite=False,
        external_misc_rng=env.randoms.misc,
        external_potion_rng=env.randoms.potion,
        player=env.player,
        deck=[clone_card(card) for card in env.deck],
        relics=list(env.relics),
        potions=list(env.potions),
        gold=env.gold,
        locked_card_ids=set(env.locked_card_ids),
    )


def step(env: Any, action: dict[str, Any]) -> dict[str, Any]:
    if env.phase == "COMBAT":
        state_payload = env.combat.to_spirecomm_state() if env.combat.outcome != "UNDECIDED" else env.combat.step(action)
        env.player = env.combat.player
        env.relics = list(env.combat.relics)
        if env.combat.outcome == "PLAYER_VICTORY":
            env.player.powers = {}
            env.player.block = 0
            env.potions = env.combat.potions
            env.gold = env.combat.gold
            env._gain_gold(env.combat.gold_gain)
            if env.floor == 50 and env.ascension_level >= 20 and not env.a20_second_boss_done:
                env.a20_second_boss_done = True
                act3_bosses = ["Awakened One", "Time Eater", "Donu and Deca"]
                candidates = [boss for boss in act3_bosses if boss != env.act_boss] or act3_bosses
                env.act_boss = env.randoms.monster.choice(candidates)
                env.current_node_symbol = "BOSS"
                env._start_combat(elite=False)
                return env.state()
            if env.floor == 53:
                env._enter_map()
                return env.state()
            if env.floor >= 54:
                env.phase = "COMPLETE"
                return env.state()
            extra_gold_rewards: list[int] = []
            if env.combat.reward_gold_bonus > 0:
                extra_gold_rewards.append(env.combat.reward_gold_bonus)
            if env.pending_event_gold > 0:
                extra_gold_rewards.append(env.pending_event_gold)
                env.pending_event_gold = 0
            env._enter_card_reward(
                extra_gold_rewards=extra_gold_rewards,
                include_base_gold=env.current_node_symbol != "EVENT_COMBAT",
            )
            if env.pending_event_relic_id:
                if not env._has_relic(env.pending_event_relic_id):
                    env.reward_relics.append(make_relic(env.pending_event_relic_id))
                env.pending_event_relic_id = None
            return env.state()
        if env.combat.outcome == "PLAYER_LOSS":
            env.phase = "GAME_OVER"
        return state_payload
    if env.phase == "NEOW":
        option_index = int(action.get("choice_index", 0))
        if 0 <= option_index < len(env.neow_options):
            option = env.neow_options[option_index]
            drawback = str(option.get("drawback") or "NONE")
            env._apply_neow_drawback(drawback)
            if drawback == "CURSE":
                env._add_curse_to_deck(uuid=f"neow-curse-{env.floor}")
            return env._apply_neow_bonus(str(option.get("bonus") or "THREE_CARDS"))
        return env.state()
    if env.phase == "CARD_REWARD":
        if action.get("kind") == "card_reward":
            reward_index = action.get("reward_index")
            card_index = action.get("card_index")
            chosen: CardInstance | None = None
            if reward_index is not None and card_index is not None:
                reward_index = int(reward_index)
                card_index = int(card_index)
                if 0 <= reward_index < len(env.reward_card_bundles) and 0 <= card_index < len(env.reward_card_bundles[reward_index]):
                    chosen = env.reward_card_bundles[reward_index][card_index]
                    env.reward_card_bundles.pop(reward_index)
            else:
                index = int(action.get("choice_index", 0))
                flat_cards = [
                    (bundle_index, bundle_card_index, card)
                    for bundle_index, bundle in enumerate(env.reward_card_bundles)
                    for bundle_card_index, card in enumerate(bundle)
                ]
                if 0 <= index < len(flat_cards):
                    reward_index, _, chosen = flat_cards[index]
                    env.reward_card_bundles.pop(reward_index)
            if chosen is not None:
                env._add_card_to_deck(chosen.card_id, upgrades=chosen.upgrades, uuid=f"deck-{env.floor}-{chosen.card_id}")
                env._refresh_reward_cards()
            return env.state()
        if action.get("kind") == "reward_gold" and env.reward_gold_piles:
            reward_index = int(action.get("reward_index", 0))
            if 0 <= reward_index < len(env.reward_gold_piles):
                amount = env.reward_gold_piles.pop(reward_index)
                if not any(relic.get("relic_id") == "Ectoplasm" for relic in env.relics):
                    env._gain_gold(amount)
                env.reward_gold = sum(env.reward_gold_piles)
            return env.state()
        if action.get("kind") == "reward_relic":
            reward_index = int(action.get("reward_index", 0))
            if 0 <= reward_index < len(env.reward_relics):
                relic = env.reward_relics.pop(reward_index)
                env._obtain_relic(relic)
                if env.reward_emerald_key:
                    env.reward_emerald_key = False
                if env._open_bottle_card_select(str(relic.get("relic_id") or "")):
                    return env.state()
            return env.state()
        if action.get("kind") == "reward_potion":
            reward_index = int(action.get("reward_index", 0))
            if 0 <= reward_index < len(env.reward_potions):
                potion = env.reward_potions.pop(reward_index)
                env._add_potion_if_space(potion)
            return env.state()
        if action.get("kind") == "reward_key" and env.reward_emerald_key:
            env.keys.add("emerald")
            env.reward_emerald_key = False
            env.reward_relics = []
            return env.state()
        if action.get("kind") == "skip" and any(relic.get("relic_id") == "Singing Bowl" for relic in env.relics) and env.reward_card_bundles:
            env.player.max_hp += 2
            env.player.current_hp += 2
        env.reward_card_bundles = []
        env.reward_cards = []
        env.reward_gold = 0
        env.reward_gold_piles = []
        env.reward_emerald_key = False
        env.reward_relics = []
        env.reward_potions = []
        env.reward_close_required = False
        if env.reward_context == "NEOW":
            env.reward_context = None
            return env._complete_neow()
        if env.reward_context == "REST":
            env.reward_context = None
            tea_set = env._relic("Ancient Tea Set")
            if tea_set is not None:
                tea_set["counter"] = 1
            env._enter_map()
            return env.state()
        if env.reward_context == "EVENT":
            env.reward_context = None
            if env.current_node_symbol == "?":
                tea_set = env._relic("Ancient Tea Set")
                if tea_set is not None and int(tea_set.get("counter", 0)) > 0:
                    tea_set["counter"] = 0
        if env.reward_context == "BOSS_RELIC":
            env.reward_context = None
            env._transition_to_next_act()
            return env.state()
        if env.floor in {16, 33}:
            env._enter_boss_treasure_room()
            return env.state()
        if env.floor == 50:
            if env.enable_act4_keys and {"ruby", "sapphire", "emerald"}.issubset(env.keys):
                env._advance_floor()
            else:
                env.phase = "COMPLETE"
            return env.state()
        env._advance_floor()
        return env.state()
    if env.phase == "CARD_SELECT":
        def _clear_tea_set_after_question_event_card_select() -> None:
            if env.current_node_symbol != "?":
                return
            tea_set = env._relic("Ancient Tea Set")
            if tea_set is not None and int(tea_set.get("counter", 0)) > 0:
                tea_set["counter"] = 0

        def _arm_tea_set_after_rest_site_card_select() -> None:
            if env.current_node_symbol != "R":
                return
            tea_set = env._relic("Ancient Tea Set")
            if tea_set is not None:
                tea_set["counter"] = 1

        index = int(action.get("target_index", action.get("choice_index", 0)) or 0)
        handled_multi_context = False
        if env.card_select_context == "LIBRARY_OBTAIN":
            choice_index = int(action.get("choice_index", 0) or 0)
            if 0 <= choice_index < len(env.card_select_generated_cards):
                selected = env.card_select_generated_cards[choice_index]
                env._add_card_to_deck(selected.card_id, upgrades=selected.upgrades, uuid=f"library-{env.floor}-{selected.card_id}")
            env.card_select_options = []
            env.card_select_generated_cards = []
            env.card_select_context = None
            env.card_select_count = 0
            env.card_select_available_indexes = []
            env.card_select_selected_indexes = []
            env._advance_floor()
            return env.state()
        if env.card_select_context == "TRANSFORM_UPGRADE":
            if index in env.card_select_available_indexes and 0 <= index < len(env.deck):
                env.card_select_selected_indexes.append(index)
                env.card_select_available_indexes = [
                    candidate for candidate in env.card_select_available_indexes if candidate != index
                ]
                env.card_select_count -= 1
            if env.card_select_count > 0:
                env._open_card_select("TRANSFORM_UPGRADE", env.card_select_count)
                return env.state()
            selected_entries = [
                (deck_index, env.deck[deck_index].card_id)
                for deck_index in env.card_select_selected_indexes
                if 0 <= deck_index < len(env.deck)
            ]
            for deck_index, _ in sorted(selected_entries, key=lambda item: item[0], reverse=True):
                env.deck.pop(deck_index)
            for selection_order, (_, removed_card_id) in enumerate(selected_entries):
                transformed = env._transformed_card_from_rng(env.randoms.misc, removed_card_id)
                transformed_upgrades = int(transformed.upgrades)
                if transformed.card_id != "Searing Blow":
                    transformed_upgrades = max(1, transformed_upgrades)
                env._add_card_to_deck(
                    transformed.card_id,
                    upgrades=transformed_upgrades,
                    uuid=f"astrolabe-{env.floor}-{selection_order}-{removed_card_id}",
                )
            env.card_select_options = []
            env.card_select_context = None
            env.card_select_count = 0
            env.card_select_available_indexes = []
            env.card_select_selected_indexes = []
            env._transition_to_next_act()
            return env.state()
        if env.card_select_context in {"EVENT_REMOVE", "EVENT_TRANSFORM"} and (
            env.card_select_count > 1 or env.card_select_selected_indexes
        ):
            handled_multi_context = True
            if index in env.card_select_available_indexes and 0 <= index < len(env.deck):
                env.card_select_selected_indexes.append(index)
                env.card_select_available_indexes = [
                    candidate for candidate in env.card_select_available_indexes if candidate != index
                ]
                env.card_select_count -= 1
            if env.card_select_count > 0:
                env._open_card_select(str(env.card_select_context or ""), env.card_select_count)
                return env.state()
            selected_indexes = sorted(env.card_select_selected_indexes, reverse=True)
            if env.card_select_context == "EVENT_REMOVE":
                for deck_index in selected_indexes:
                    if 0 <= deck_index < len(env.deck):
                        env.deck.pop(deck_index)
            else:
                removed_ids: list[str] = []
                for deck_index in selected_indexes:
                    if 0 <= deck_index < len(env.deck):
                        removed_ids.append(env.deck[deck_index].card_id)
                        env.deck.pop(deck_index)
                for selection_order, removed_card_id in enumerate(reversed(removed_ids)):
                    card = env._transformed_card_from_rng(env.randoms.misc, removed_card_id)
                    env._add_card_to_deck(
                        card.card_id,
                        upgrades=card.upgrades,
                        uuid=f"event-transform-{env.floor}-multi-{selection_order}",
                    )
        if not handled_multi_context and 0 <= index < len(env.deck):
            if env.card_select_context == "NEOW_REMOVE":
                env.deck.pop(index)
            elif env.card_select_context == "NEOW_UPGRADE":
                _increment_card_upgrade(env.deck[index])
            elif env.card_select_context == "NEOW_TRANSFORM":
                removed = env.deck.pop(index)
                card = env._transformed_card_from_rng(env.randoms.neow, removed.card_id)
                env._add_card_to_deck(card.card_id, upgrades=card.upgrades, uuid=f"neow-transform-{env.floor}-{index}")
            elif env.card_select_context == "EVENT_REMOVE":
                env.deck.pop(index)
            elif env.card_select_context in {"EVENT_UPGRADE", "CAMPFIRE_SMITH"}:
                _increment_card_upgrade(env.deck[index])
            elif env.card_select_context == "EVENT_TRANSFORM":
                removed = env.deck.pop(index)
                card = env._transformed_card_from_rng(env.randoms.misc, removed.card_id)
                env._add_card_to_deck(card.card_id, upgrades=card.upgrades, uuid=f"event-transform-{env.floor}-{index}")
            elif env.card_select_context == "CAMPFIRE_TOKE":
                env.deck.pop(index)
            elif env.card_select_context == "DUPLICATE":
                selected = env.deck[index]
                env._add_card_to_deck(
                    selected.card_id,
                    upgrades=selected.upgrades,
                    uuid=f"duplicator-{env.floor}-{selected.card_id}",
                )
            elif env.card_select_context == "BONFIRE_SPIRITS":
                offered = env.deck.pop(index)
                rarity = offered.card_def.rarity
                if rarity == "CURSE":
                    if not env._has_relic("Spirit Poop"):
                        env._obtain_relic(make_relic("Spirit Poop"))
                elif rarity == "BASIC":
                    pass
                elif rarity in {"COMMON", "SPECIAL"}:
                    env._heal_run(5)
                elif rarity == "UNCOMMON":
                    env._heal_run(10)
                elif rarity == "RARE":
                    env.player.max_hp += 10
                    env._heal_run(env.player.max_hp)
            elif env.card_select_context == "BOTTLE_REWARD":
                relic = env._relic(str(env.pending_bottle_relic_id or ""))
                if relic is not None:
                    relic["card_id"] = env.deck[index].card_id
                    relic["card_uuid"] = env.deck[index].uuid
            env.card_select_count -= 1
        if env.card_select_count > 0:
            env._open_card_select(str(env.card_select_context or ""), env.card_select_count)
            return env.state()
        env.card_select_options = []
        context = env.card_select_context
        completion = env.card_select_completion
        env.card_select_context = None
        env.card_select_completion = None
        env.pending_bottle_relic_id = None
        env.card_select_count = 0
        env.card_select_available_indexes = []
        env.card_select_selected_indexes = []
        if context and str(context).startswith("NEOW_"):
            return env._complete_neow()
        if context == "BOTTLE_REWARD":
            remaining_non_relic_rewards = (
                bool(env.reward_card_bundles)
                or bool(env.reward_potions)
                or bool(env.reward_gold_piles)
                or bool(env.reward_emerald_key)
            )
            if env.reward_context is None and env.reward_relics and not remaining_non_relic_rewards:
                env.reward_relics = []
            if (
                env.reward_card_bundles
                or env.reward_relics
                or env.reward_potions
                or env.reward_gold_piles
                or env.reward_emerald_key
            ):
                env.phase = "CARD_REWARD"
                return env.state()
            if env.reward_context == "BOSS_RELIC":
                env.reward_context = None
                env._transition_to_next_act()
                return env.state()
            env._advance_floor()
            return env.state()
        if completion == "TRANSITION_NEXT_ACT":
            env._transition_to_next_act()
            return env.state()
        if completion == "DESIGNER_FULL_SERVICE":
            env._upgrade_random_cards_from_rng(env.randoms.misc, 1)
            env._advance_floor()
            return env.state()
        if completion == "CAMPFIRE_LEAVE":
            if context in {"CAMPFIRE_SMITH", "CAMPFIRE_TOKE"}:
                _arm_tea_set_after_rest_site_card_select()
            env._enter_map()
            return env.state()
        if context in {"EVENT_REMOVE", "EVENT_UPGRADE", "EVENT_TRANSFORM"}:
            _clear_tea_set_after_question_event_card_select()
        env._advance_floor()
        return env.state()
    if env.phase == "BOSS_RELIC":
        index = int(action.get("choice_index", 0))
        if 0 <= index < len(env.boss_relic_options):
            env._obtain_relic(dict(env.boss_relic_options[index]))
        env.boss_relic_options = []
        if env.phase != "BOSS_RELIC":
            return env.state()
        env._transition_to_next_act()
        return env.state()
    if env.phase == "MAP":
        env._advance_to_node(str(action.get("node_id") or action.get("symbol") or action.get("name") or "M"))
        return env.state()
    if env.phase == "CAMPFIRE":
        if action.get("name") == "REST":
            env._heal_run(max(1, int(env.player.max_hp * 0.3)))
            if env._has_relic("Regal Pillow"):
                env._heal_run(15)
            if env._has_relic("Dream Catcher"):
                reward_count = 3 + (1 if env._has_relic("Question Card") else 0)
                if env._has_relic("Busted Crown"):
                    reward_count = max(1, reward_count - 2)
                env.phase = "CARD_REWARD"
                env.reward_context = "REST"
                env.reward_close_required = False
                env.reward_card_bundles = [env._roll_card_reward(count=reward_count, room="R")]
                env._refresh_reward_cards()
                env.reward_gold = 0
                env.reward_gold_piles = []
                env.reward_emerald_key = False
                env.reward_relics = []
                env.reward_potions = []
                return env.state()
        elif action.get("name") == "SMITH":
            if action.get("target_index") is not None:
                index = int(action["target_index"])
                if 0 <= index < len(env.deck) and _card_can_upgrade(env.deck[index]):
                    _increment_card_upgrade(env.deck[index])
            else:
                upgradable = [index for index, card in enumerate(env.deck) if _card_can_upgrade(card)]
                if len(upgradable) > 1:
                    env._open_card_select("CAMPFIRE_SMITH", 1)
                    env.card_select_completion = "CAMPFIRE_LEAVE"
                    return env.state()
                if upgradable:
                    _increment_card_upgrade(env.deck[upgradable[0]])
        elif action.get("name") in {"PURGE", "TOKE"}:
            if action.get("target_index") is not None:
                index = int(action["target_index"])
                if 0 <= index < len(env.deck):
                    env.deck.pop(index)
            else:
                removable = [index for index, card in enumerate(env.deck) if _card_can_transform(card)]
                if len(removable) > 1:
                    env._open_card_select("CAMPFIRE_TOKE", 1)
                    env.card_select_completion = "CAMPFIRE_LEAVE"
                    return env.state()
                if removable:
                    env.deck.pop(removable[0])
        elif action.get("name") == "DIG":
            tea_set = next((relic for relic in env.relics if relic.get("relic_id") == "Ancient Tea Set"), None)
            if tea_set is not None:
                tea_set["counter"] = 1
            env.phase = "CARD_REWARD"
            env.reward_context = "REST"
            env.reward_close_required = False
            env.reward_card_bundles = []
            env.reward_cards = []
            env.reward_gold = 0
            env.reward_gold_piles = []
            env.reward_emerald_key = False
            env.reward_relics = [env._roll_relic()]
            env.reward_potions = []
            return env.state()
        elif action.get("name") == "LIFT":
            girya = env._relic("Girya")
            if girya is not None:
                girya["counter"] = min(3, max(0, int(girya.get("counter", 0))) + 1)
        elif action.get("name") == "RECALL":
            env.keys.add("ruby")
        tea_set = next((relic for relic in env.relics if relic.get("relic_id") == "Ancient Tea Set"), None)
        if tea_set is not None:
            tea_set["counter"] = 1
        env._enter_map()
        return env.state()
    if env.phase == "SHOP":
        item_kind = action.get("item_kind")
        price = int(action.get("price", 0) or 0)
        if item_kind == "leave":
            env._enter_map()
            return env.state()
        if env.gold >= price:
            env.gold -= price
            if price > 0 and (maw_bank := env._relic("Maw Bank")) is not None:
                maw_bank["counter"] = 0
            if item_kind == "card" and action.get("card"):
                env._add_card_to_deck(action["card"]["card_id"], upgrades=action["card"].get("upgrades", 0), uuid=f"shop-{env.floor}-{action['card']['card_id']}")
                env._remove_shop_item(action)
            elif item_kind == "relic" and action.get("relic"):
                env._obtain_relic(dict(action["relic"]))
                env._remove_shop_item(action)
            elif item_kind == "potion" and action.get("potion_id"):
                env._add_potion_if_space(make_potion(str(action["potion_id"]), price=0))
                env._remove_shop_item(action)
            elif item_kind == "purge":
                index = int(action["target_index"]) if action.get("target_index") is not None else env._first_purge_index()
                if index is not None:
                    env.deck.pop(index)
                env.shop_remove_count += 1
                env.shop_items = [item for item in env.shop_items if item.get("item_kind") != "purge"]
        return env.state()
    if env.phase == "TREASURE":
        if action.get("name") == "OPEN_CHEST":
            env.treasure_options = []
            env._enter_chest()
            return env.state()
        env.treasure_options = []
        env._enter_map()
        return env.state()
    if env.phase == "EVENT":
        event_id = str(action.get("event_id") or "")
        if event_id == "Council of Ghosts":
            event_id = "Ghosts"
        name = str(action.get("name") or action.get("label") or "")
        if event_id == "Big Fish" and name in {"Banana", "BANANA"}:
            env._heal_run(max(1, env.player.max_hp // 3))
        elif event_id == "Big Fish" and name in {"Donut", "DONUT"}:
            env.player.max_hp += 5
            env.player.current_hp += 5
        elif event_id == "Big Fish" and name in {"Box", "BOX"}:
            env._obtain_relic(env._roll_screenless_relic_of_tier(env._roll_relic_tier_for_act(env.act)))
            env._add_curse_to_deck("Regret", uuid=f"big-fish-box-{env.floor}")
        elif event_id == "Golden Idol" and name in {"Take", "Take Damage", "TAKE_GOLDEN_IDOL"}:
            if not any(relic.get("relic_id") == "Golden Idol" for relic in env.relics):
                env._obtain_relic(make_relic("Golden Idol"))
            env.event_options = [
                {"kind": "event", "event_id": event_id, "name": "Outrun", "label": "Outrun", "choice_index": 2},
                {"kind": "event", "event_id": event_id, "name": "Smash", "label": "Smash", "choice_index": 3},
                {"kind": "event", "event_id": event_id, "name": "Hide", "label": "Hide", "choice_index": 4},
            ]
            return env.state()
        elif event_id == "Golden Idol" and name in {"Outrun", "Take Wound"}:
            env._add_curse_to_deck("Injury", uuid=f"golden-idol-injury-{env.floor}")
            env._advance_floor()
            return env.state()
        elif event_id == "Golden Idol" and name in {"Smash", "Lose Max HP"}:
            env._lose_run_hp(_event_percent_hp_loss(env, 0.35 if env.ascension_level >= 15 else 0.25))
            env._advance_floor()
            return env.state()
        elif event_id == "Golden Idol" and name in {"Hide"}:
            env.player.max_hp = max(
                1,
                env.player.max_hp - _event_percent_hp_loss(env, 0.10 if env.ascension_level >= 15 else 0.08),
            )
            env.player.current_hp = min(env.player.current_hp, env.player.max_hp)
            env._advance_floor()
            return env.state()
        elif event_id == "Shining Light" and name in {"Entered Light", "UPGRADE_TWO"}:
            env._lose_run_hp(_event_percent_hp_loss(env, 0.2, mode="round"))
            if env.phase == "GAME_OVER":
                return env.state()
            upgradeable = [
                idx for idx, card in enumerate(env.deck)
                if _card_can_upgrade(card)
            ]
            java_collections_shuffle(upgradeable, env.randoms.misc.random_long())
            for index in upgradeable[:2]:
                _increment_card_upgrade(env.deck[index])
        elif event_id == "Golden Shrine" and name in {"Prayed", "GAIN_GOLD"}:
            env._gain_gold(50 if env.ascension_level >= 15 else 100)
        elif event_id == "Golden Shrine" and name in {"Desecrated", "GAIN_GOLD_CURSE"}:
            env._gain_gold(275)
            env._add_curse_to_deck("Regret", uuid=f"golden-shrine-regret-{env.floor}")
        elif event_id == "The Divine Fountain" and name in {"Drank", "DRANK"}:
            env.deck = [card for card in env.deck if card.card_def.card_type != "CURSE"]
        elif event_id == "The Cleric" and name in {"Healed", "HEAL"}:
            env.gold = max(0, env.gold - 35)
            env._heal_run(int(env.player.max_hp * 0.25))
        elif event_id == "The Cleric" and name in {"Card Removal", "PURGE"}:
            env.gold = max(0, env.gold - (75 if env.ascension_level >= 15 else 50))
            env._open_card_select("EVENT_REMOVE", 1)
            return env.state()
        elif event_id == "Living Wall" and name in {"Forget", "REMOVE"}:
            env._open_card_select("EVENT_REMOVE", 1)
            return env.state()
        elif event_id == "Living Wall" and name in {"Change", "TRANSFORM"}:
            env._open_card_select("EVENT_TRANSFORM", 1)
            return env.state()
        elif event_id == "Living Wall" and name in {"Grow", "UPGRADE"}:
            env._open_card_select("EVENT_UPGRADE", 1)
            return env.state()
        elif event_id == "Ancient Writing" and name == "Elegance":
            env._open_card_select("EVENT_REMOVE", 1)
            return env.state()
        elif event_id == "Ancient Writing" and name == "Simplicity":
            for card in env.deck:
                if card.card_id in {"Strike_R", "Defend_R"}:
                    _ensure_card_upgraded(card)
        elif event_id == "Old Beggar" and name == "Gave Gold":
            env.gold = max(0, env.gold - 75)
            env._open_card_select("EVENT_REMOVE", 1)
            return env.state()
        elif event_id == "Cursed Tome":
            choice_index = int(action.get("choice_index", 0))
            phase = int(env.event_state.get("phase", 0))
            if choice_index == 0:
                env.event_state["phase"] = phase + 1
                env._set_cursed_tome_options()
                return env.state()
            if choice_index == 1:
                env._advance_floor()
                return env.state()
            if choice_index in {2, 3, 4}:
                env._lose_run_hp_raw(max(0, phase))
                if env.phase == "GAME_OVER":
                    return env.state()
                env.event_state["phase"] = phase + 1
                env._set_cursed_tome_options()
                return env.state()
            if choice_index == 5:
                env._lose_run_hp_raw(15 if env.ascension_level >= 15 else 10)
                if env.phase == "GAME_OVER":
                    return env.state()
                roll = int(env.randoms.misc.random(2))
                relic_id = ["Necronomicon", "Enchiridion", "Nilry's Codex"][roll]
                env.phase = "CARD_REWARD"
                env.reward_card_bundles = []
                env.reward_cards = []
                env.reward_gold = 0
                env.reward_gold_piles = []
                env.reward_potions = []
                env.reward_relics = [make_relic(relic_id)]
                env.reward_emerald_key = False
                return env.state()
            if choice_index == 6:
                env._lose_run_hp_raw(3)
                if env.phase == "GAME_OVER":
                    return env.state()
                env._advance_floor()
                return env.state()
        elif event_id == "World of Goop" and name in {"Left Gold", "LOSE_GOLD"}:
            gold_loss = int(env.event_state.get("gold_loss", 0))
            env.gold = max(0, env.gold - max(0, gold_loss))
        elif event_id == "World of Goop" and name in {"Gather Gold", "LOSE_HP_GAIN_GOLD"}:
            env._lose_run_hp(11)
            env._gain_gold(75)
            if env.phase == "GAME_OVER":
                return env.state()
        elif event_id == "Hypnotizing Colored Mushrooms" and name == "Fought Mushrooms":
            gold_amt = int(env.randoms.misc.random(20, 30))
            env._start_event_combat(["FungiBeast", "FungiBeast", "FungiBeast"], relic_id="Odd Mushroom", gold_gain=gold_amt)
            return env.state()
        elif event_id == "Hypnotizing Colored Mushrooms" and name == "Ignored":
            env._gain_gold(50 if env.ascension_level >= 15 else 99)
        elif event_id == "Scrap Ooze" and name in {"Success", "LOSE_HP_GAIN_RELIC"}:
            env._lose_run_hp(5 if env.ascension_level >= 15 else 3)
            if env.phase == "GAME_OVER":
                return env.state()
            attempts = int(env.event_state.get("counter", 0))
            roll = int(env.randoms.misc.random(99))
            relic_chance = attempts * 10 + 25
            if roll >= 99 - relic_chance:
                env._obtain_relic(env._roll_screenless_relic_of_tier(env._roll_relic_tier_for_act(1)))
            else:
                env.event_state["counter"] = attempts + 1
                return env.state()
        elif event_id == "Mindbloom" and name == "Fight":
            env._start_event_boss_combat()
            return env.state()
        elif event_id == "Mindbloom" and name == "Gold":
            env._gain_gold(999)
        elif event_id == "Mindbloom" and name == "Heal":
            env.player.current_hp = env.player.max_hp
        elif event_id == "Mindbloom" and name == "Upgrade":
            for card in env.deck:
                if card.card_def.card_type not in {"STATUS", "CURSE"}:
                    _ensure_card_upgraded(card)
        elif event_id == "Falling":
            index_key = "attack" if name == "Removed Attack" else "skill" if name == "Removed Skill" else "power"
            deck_index = int(env.event_state.get(index_key, -1))
            if 0 <= deck_index < len(env.deck):
                env.deck.pop(deck_index)
        elif event_id == "Winding Halls" and name == "Max HP":
            env.player.max_hp = max(1, env.player.max_hp - 5)
            env.player.current_hp = min(env.player.current_hp, env.player.max_hp)
        elif event_id == "Winding Halls" and name == "Writhe":
            env._add_curse_to_deck("Writhe", uuid=f"winding-halls-{env.floor}")
        elif event_id == "Winding Halls" and name == "Embrace Madness":
            env._lose_run_hp(12)
            env._add_card_to_deck("Madness", uuid=f"winding-halls-madness-{env.floor}-0")
            env._add_card_to_deck("Madness", uuid=f"winding-halls-madness-{env.floor}-1")
        elif event_id == "Wing Statue" and name == "Card Removal":
            env._lose_run_hp(7)
            if env.phase == "GAME_OVER":
                return env.state()
            env._open_card_select("EVENT_REMOVE", 1)
            return env.state()
        elif event_id == "Wing Statue" and name == "Gained Gold":
            env._gain_gold(int(env.randoms.misc.random(50, 80)))
        elif event_id == "Drug Dealer" and name in {"Got JAX", "Obtain J.A.X."}:
            env._add_card_to_deck("J.A.X.", uuid=f"jax-{env.floor}")
        elif event_id == "Drug Dealer" and name in {"Inject Mutagens", "Became Test Subject"}:
            env._obtain_relic(make_relic("Mutagenic Strength"))
        elif event_id == "Augmenter" and name == "JAX":
            env._add_card_to_deck("J.A.X.", uuid=f"augmenter-jax-{env.floor}")
        elif event_id == "Augmenter" and name == "Transform":
            env._open_card_select("EVENT_TRANSFORM", 2)
            return env.state()
        elif event_id == "Augmenter" and name == "Mutagenic Strength":
            env._obtain_relic(make_relic("Mutagenic Strength"))
        elif event_id == "The Library" and name == "Heal":
            env._heal_run(max(1, env.player.max_hp // 3))
        elif event_id == "The Library" and name == "Read":
            env._open_library_card_select()
            return env.state()
        elif event_id == "The Nest" and name == "Stole From Cult":
            env._gain_gold(50 if env.ascension_level >= 15 else 99)
        elif event_id == "The Nest" and name in {"Stay in Line", "Ritual Dagger"}:
            env._lose_run_hp(6)
            if env.phase == "GAME_OVER":
                return env.state()
            env._add_card_to_deck("Ritual Dagger", uuid=f"the-nest-{env.floor}")
        elif event_id == "Accursed Blacksmith" and name == "Forge":
            for card in env.deck:
                if card.card_def.card_type not in {"STATUS", "CURSE"}:
                    _ensure_card_upgraded(card)
        elif event_id == "Accursed Blacksmith" and name == "Rummage":
            if not env._has_relic("Warped Tongs"):
                env._obtain_relic(make_relic("Warped Tongs"))
            env._add_curse_to_deck("Pain", uuid=f"blacksmith-{env.floor}")
        elif event_id == "The Mausoleum" and name == "Opened":
            relic = env._roll_screenless_relic_of_tier("RARE")
            env._open_relic_reward_screen(relic, context="EVENT")
            if env.ascension_level >= 15 or env.randoms.misc.random_boolean():
                env._add_curse_to_deck("Writhe", uuid=f"mausoleum-{env.floor}")
            return env.state()
        elif event_id == "Masked Bandits" and name == "Paid Fearfully":
            env.gold = 0
        elif event_id == "Masked Bandits" and name == "Fought Bandits":
            env._lose_run_hp(5)
            if env.phase == "GAME_OVER":
                return env.state()
            env._start_event_combat(["Bear", "Pointy", "Romeo"], relic_id="Red Mask", gold_gain=222)
            return env.state()
        elif event_id == "Vampires" and name in {"Offered", "Accepted"}:
            if name == "Offered":
                env.relics = [relic for relic in env.relics if relic.get("relic_id") != "Blood Vial"]
            else:
                env.player.max_hp = max(1, int(env.player.max_hp * 0.7))
                env.player.current_hp = min(env.player.current_hp, env.player.max_hp)
            env.deck = [card for card in env.deck if card.card_id != "Strike_R"]
            for index in range(5):
                env._add_card_to_deck("Bite", uuid=f"vampires-{env.floor}-{index}")
        elif event_id == "Ghosts" and name == "Accepted":
            env.player.max_hp = max(1, env.player.max_hp // 2)
            env.player.current_hp = min(env.player.current_hp, env.player.max_hp)
            for index in range(5):
                env._add_card_to_deck("Apparition", uuid=f"apparition-{env.floor}-{index}")
        elif event_id == "Duplicator" and name == "Duplicated" and env.deck:
            if len(env.deck) > 1:
                env._open_card_select("DUPLICATE", 1)
                return env.state()
            source = env.deck[0]
            env._add_card_to_deck(source.card_id, upgrades=source.upgrades, uuid=f"duplicator-{env.floor}-{source.card_id}")
        elif event_id == "N'loth" and name == "Gave Relic":
            relic_indexes = [int(index) for index in env.event_state.get("relic_indexes", [])]
            choice_index = int(action.get("choice_index", 0) or 0)
            if 0 <= choice_index < len(relic_indexes):
                relic_index = relic_indexes[choice_index]
                if not (0 <= relic_index < len(env.relics)):
                    relic_index = -1
                if relic_index >= 0:
                    env.relics.pop(relic_index)
                    env._obtain_relic(make_relic("N'loth's Gift"))
        elif event_id == "Note For Yourself" and name == "Took Card":
            env._add_card_to_deck(env.note_for_yourself_card_id, uuid=f"note-for-yourself-{env.floor}")
            env._open_card_select("EVENT_REMOVE", 1)
            return env.state()
        elif event_id == "The Joust" and name == "Murderer":
            env.gold = max(0, env.gold - 50)
            owner_wins = bool(env.randoms.misc.random_boolean(0.3))
            if not owner_wins:
                env._gain_gold(100)
        elif event_id == "The Joust" and name == "Owner":
            env.gold = max(0, env.gold - 50)
            owner_wins = bool(env.randoms.misc.random_boolean(0.3))
            if owner_wins:
                env._gain_gold(250)
        elif event_id == "Designer In-Spire":
            unfavorable = env.ascension_level >= 15
            gold_cost0 = 50 if unfavorable else 40
            gold_cost1 = 75 if unfavorable else 60
            gold_cost2 = 110 if unfavorable else 90
            choice_index = int(action.get("choice_index", 0) or 0)
            upgrade_one = bool(env.event_state.get("designer_upgrade_one", True))
            clean_up_is_remove = bool(env.event_state.get("designer_cleanup_is_remove", True))
            if choice_index == (0 if upgrade_one else 1):
                if upgrade_one:
                    env.gold = max(0, env.gold - gold_cost0)
                    env._open_card_select("EVENT_UPGRADE", 1)
                    return env.state()
                env._upgrade_random_cards_from_rng(env.randoms.misc, 2)
            elif choice_index == (2 if clean_up_is_remove else 3):
                env.gold = max(0, env.gold - gold_cost1)
                if clean_up_is_remove:
                    env._open_card_select("EVENT_REMOVE", 1)
                    return env.state()
                transformable_indexes = [
                    index for index, card in enumerate(env.deck)
                    if _card_can_transform(card)
                ]
                if len(transformable_indexes) >= 2:
                    java_collections_shuffle(transformable_indexes, env.randoms.misc.random_long())
                    selected = sorted(transformable_indexes[:2], reverse=True)
                    removed_ids: list[str] = []
                    for deck_index in selected:
                        removed_ids.append(env.deck[deck_index].card_id)
                        env.deck.pop(deck_index)
                    for selection_order, removed_card_id in enumerate(reversed(removed_ids)):
                        card = env._transformed_card_from_rng(env.randoms.misc, removed_card_id)
                        env._add_card_to_deck(
                            card.card_id,
                            upgrades=card.upgrades,
                            uuid=f"designer-transform-{env.floor}-{selection_order}",
                        )
            elif choice_index == 4:
                env.gold = max(0, env.gold - gold_cost2)
                env._open_card_select("EVENT_REMOVE", 1)
                env.card_select_completion = "DESIGNER_FULL_SERVICE"
                return env.state()
            elif choice_index == 5:
                env._lose_run_hp(5 if unfavorable else 3)
                if env.phase == "GAME_OVER":
                    return env.state()
        elif event_id == "Face Trader" and name == "Touched":
            env._gain_gold(75)
            env._lose_run_hp(_event_percent_hp_loss(env, 0.10))
        elif event_id == "Face Trader" and name == "Traded":
            env._obtain_relic(env._roll_relic())
        elif event_id == "Forgotten Altar" and name == "Shed Blood":
            hp_loss = _event_percent_hp_loss(env, 0.25, mode="round")
            env.player.max_hp += 5
            env.player.current_hp = min(env.player.max_hp, env.player.current_hp + 5)
            env._lose_run_hp(hp_loss)
            if env.phase == "GAME_OVER":
                return env.state()
        elif event_id == "Forgotten Altar" and name == "Smashed Altar":
            if any(item.get("relic_id") == "Golden Idol" for item in env.relics):
                env.relics = [item for item in env.relics if item.get("relic_id") != "Golden Idol"]
                if not env._has_relic("Bloody Idol"):
                    env._obtain_relic(make_relic("Bloody Idol"))
            else:
                env._add_curse_to_deck("Decay", uuid=f"forgotten-altar-{env.floor}")
        elif event_id == "Lab" and name == "Obtained Potions":
            env._add_random_potion_reward(count=3)
        elif event_id == "Match and Keep" and name == "Played":
            cards = env._consume_match_and_keep_rng()
            for index, reward in enumerate(cards[:2]):
                env._add_card_to_deck(reward.card_id, upgrades=reward.upgrades, uuid=f"match-and-keep-{env.floor}-{index}")
        elif event_id == "Purifier" and name in {"Purged", "REMOVE"}:
            env._open_card_select("EVENT_REMOVE", 1)
            return env.state()
        elif event_id == "The Moai Head" and name == "Jumped":
            env._heal_run(env.player.max_hp)
            env.player.max_hp = max(1, env.player.max_hp - 12)
            env.player.current_hp = min(env.player.current_hp, env.player.max_hp)
        elif event_id == "The Moai Head" and name == "Offered Golden Idol":
            if any(item.get("relic_id") == "Golden Idol" for item in env.relics):
                env.relics = [item for item in env.relics if item.get("relic_id") != "Golden Idol"]
                env.player.max_hp += 10
                env.player.current_hp += 10
            else:
                env._heal_run(20)
        elif event_id == "Sensory Stone" and name == "Recall":
            env._add_colorless_cards_to_deck(1, prefix="sensory-stone-recall")
        elif event_id == "Sensory Stone" and name == "Remember":
            env._add_colorless_cards_to_deck(2, prefix="sensory-stone-remember")
            env._lose_run_hp(5)
        elif event_id == "Sensory Stone" and name == "Live Forever":
            env._add_colorless_cards_to_deck(3, prefix="sensory-stone-live-forever")
            env._lose_run_hp(10)
        elif event_id == "The Woman in Blue" and name.startswith("Bought"):
            amount = 1 if "1" in name else 2 if "2" in name else 3
            env.gold = max(0, env.gold - {1: 20, 2: 30, 3: 40}[amount])
            env._open_potion_reward_screen(count=amount, context="EVENT")
            return env.state()
        elif event_id == "Transmogrifier" and name == "Transformed":
            env._open_card_select("EVENT_TRANSFORM", 1)
            return env.state()
        elif event_id == "Upgrade Shrine" and name == "Upgraded":
            env._open_card_select("EVENT_UPGRADE", 1)
            return env.state()
        elif event_id == "Wheel of Change" and name == "Spun":
            result = int(env.randoms.misc.random(5))
            if result == 0:
                env._gain_gold(env.act * 100)
            elif result == 1:
                relic = env._roll_screenless_relic_of_tier(env._roll_relic_tier_for_act(env.act))
                env._open_relic_reward_screen(relic, context="EVENT")
                return env.state()
            elif result == 2:
                env._heal_run(env.player.max_hp)
            elif result == 3:
                env._add_curse_to_deck("Decay", uuid=f"wheel-{env.floor}")
            elif result == 4:
                env._open_card_select("EVENT_REMOVE", 1)
                return env.state()
            else:
                env._lose_run_hp(_event_percent_hp_loss(env, 0.15 if env.ascension_level >= 15 else 0.10))
                if env.phase == "GAME_OVER":
                    return env.state()
        elif event_id == "Pleading Vagrant" and name == "Gave Gold":
            env.gold = max(0, env.gold - 85)
            env._obtain_relic(env._roll_screenless_relic_of_tier(env._roll_relic_tier_for_act(env.act)))
        elif event_id == "Pleading Vagrant" and name == "Robbed":
            env._obtain_relic(env._roll_screenless_relic_of_tier(env._roll_relic_tier_for_act(env.act)))
            env._add_curse_to_deck("Shame", uuid=f"pleading-vagrant-{env.floor}")
        elif event_id == "Dead Adventurer" and name == "Searched":
            phase = int(env.event_state.get("phase", 0))
            rewards = list(env.event_state.get("rewards", [0, 1, 2]))
            encounter_chance = phase * 25 + (35 if env.ascension_level >= 15 else 25)
            did_encounter = int(env.randoms.misc.random(99)) < encounter_chance
            if did_encounter:
                gold_gain = int(env.randoms.misc.random(25, 35))
                relic_id = None
                for reward in rewards[phase:]:
                    if reward == 0:
                        gold_gain += 30
                    elif reward == 2 and relic_id is None:
                        relic_id = env._roll_relic_of_tier(env._roll_relic_tier_for_act(1)).get("relic_id")
                encounter = str(env.event_state.get("encounter") or "Gremlin Nob")
                if encounter == "Three Sentries":
                    monster_ids = ["Sentry", "Sentry", "Sentry"]
                elif encounter == "Lagavulin":
                    monster_ids = ["LagavulinEvent"]
                else:
                    monster_ids = ["GremlinNob"]
                env._start_event_combat(monster_ids, relic_id=relic_id, gold_gain=gold_gain, elite=False)
                return env.state()
            reward = rewards[phase] if phase < len(rewards) else 1
            if reward == 0:
                env._gain_gold(30)
            elif reward == 2:
                relic = env._roll_screenless_relic_of_tier(env._roll_relic_tier_for_act(1))
                env._obtain_relic(relic)
            env.event_state["phase"] = phase + 1
            return env.state()
        elif event_id == "Ominous Forge" and name == "Forge":
            env._open_card_select("EVENT_UPGRADE", 1)
            return env.state()
        elif event_id == "Ominous Forge" and name == "Rummage":
            env._add_curse_to_deck("Pain", uuid=f"ominous-forge-{env.floor}")
            if not env._has_relic("Warped Tongs"):
                env._obtain_relic(make_relic("Warped Tongs"))
        elif event_id == "The Ssssserpent" and name == "Agreed":
            env._add_curse_to_deck("Doubt", uuid=f"liars-game-{env.floor}")
            env._gain_gold(150 if env.ascension_level >= 15 else 175)
        elif event_id == "Mysterious Sphere" and name == "Fought Orb Walkers":
            relic_id = env._roll_relic().get("relic_id")
            env._start_event_combat(["OrbWalker", "OrbWalker"], relic_id=relic_id)
            return env.state()
        elif event_id == "Colosseum" and name == "Fought":
            relic_id = env._roll_relic().get("relic_id")
            env._start_event_combat(["SlaverBlue", "SlaverRed", "Taskmaster"], relic_id=relic_id, gold_gain=100)
            return env.state()
        elif event_id in {"We Meet Again!", "WeMeetAgain"} and name in {"Gave Potion", "Gave Gold", "Gave Card"}:
            if name == "Gave Potion":
                potion_idx = env.event_state.get("potion_idx")
                if potion_idx is not None and 0 <= int(potion_idx) < len(env.potions):
                    env.potions[int(potion_idx)] = PotionInstance()
            elif name == "Gave Gold":
                gold_loss = int(env.event_state.get("gold", -1))
                if gold_loss > 0:
                    env.gold = max(0, env.gold - gold_loss)
            elif name == "Gave Card":
                card_idx = env.event_state.get("card_idx")
                if card_idx is not None and 0 <= int(card_idx) < len(env.deck):
                    env.deck.pop(int(card_idx))
            env._obtain_relic(env._roll_screenless_relic_of_tier(env._roll_relic_tier_for_act(env.act)))
        elif event_id == "Tomb of Lord Red Mask" and name == "Got Gold":
            env._gain_gold(222)
        elif event_id == "Tomb of Lord Red Mask" and name == "Paid":
            env.gold = 0
            env._obtain_relic(make_relic("Red Mask"))
        elif event_id == "Knowing Skull":
            if name == "Riches":
                hp_amount = int(env.event_state.get("hp_amount_0", 6))
                env._lose_run_hp(hp_amount)
                if env.phase == "GAME_OVER":
                    return env.state()
                env.event_state["hp_amount_0"] = hp_amount + 1
                env._gain_gold(90)
                return env.state()
            if name == "Success":
                hp_amount = int(env.event_state.get("hp_amount_1", 6))
                env._lose_run_hp(hp_amount)
                if env.phase == "GAME_OVER":
                    return env.state()
                env.event_state["hp_amount_1"] = hp_amount + 1
                colorless_pool = [
                    CARD_LIBRARY[card_id]
                    for card_id in COLORLESS_CARD_ID_ORDER
                    if CARD_LIBRARY[card_id].rarity == "UNCOMMON"
                ]
                if colorless_pool:
                    chosen = env.randoms.card.choice(colorless_pool)
                    env._add_card_to_deck(chosen.card_id, uuid=f"knowing-skull-card-{env.floor}-{chosen.card_id}")
                return env.state()
            if name == "A Pick Me Up":
                hp_amount = int(env.event_state.get("hp_amount_2", 6))
                env._lose_run_hp(hp_amount)
                if env.phase == "GAME_OVER":
                    return env.state()
                env.event_state["hp_amount_2"] = hp_amount + 1
                env._add_potion_if_space(get_random_potion(env.randoms.potion, "IRONCLAD"))
                return env.state()
            if name == "Leave":
                env._lose_run_hp(6)
                if env.phase == "GAME_OVER":
                    return env.state()
        elif event_id == "Secret Portal" and name == "Entered Portal":
            _, boss_floor = env._act_floor_range(env.act)
            env.floor = boss_floor - 1
            env.current_map_node_id = None
            env.current_node_symbol = "R"
            floor_rng = StsRandom(env.seed + env.floor)
            env.randoms.misc = floor_rng.copy()
            env.randoms.shuffle = floor_rng.copy()
            env.randoms.card_random = floor_rng.copy()
            env.rng = env.randoms.misc
            env.current_event_id = None
            env.event_state = {}
            env._enter_campfire()
            return env.state()
        if env.phase == "GAME_OVER":
            return env.state()
        env.current_event_id = None
        env.event_state = {}
        tea_set = env._relic("Ancient Tea Set")
        if tea_set is not None and int(tea_set.get("counter", 0)) > 0:
            # Ancient Tea Set survives long enough to apply to event combats
            # reached from '?', but once a regular event resolves back to the
            # map lightspeed clears the pending +2 energy before the next room.
            tea_set["counter"] = 0
        env._enter_map()
        return env.state()
    if env.phase == "CHEST":
        gold_amount = int(action.get("gold_amount", env.chest_gold_amount) or 0)
        if gold_amount > 0:
            env._gain_gold(gold_amount)
        if action.get("item_kind") == "sapphire_key":
            env.keys.add("sapphire")
        elif action.get("item_kind") == "relic":
            if any(relic.get("relic_id") == "Cursed Key" for relic in env.relics):
                env._add_curse_to_deck(uuid=f"cursed-key-{env.floor}")
            env._obtain_relic(dict(action))
            matryoshka = next((relic for relic in env.relics if relic.get("relic_id") == "Matryoshka" and int(relic.get("counter", 0)) > 0), None)
            if matryoshka is not None:
                matryoshka["counter"] = int(matryoshka.get("counter", 0)) - 1
                env._obtain_relic(env._roll_relic())
        env.chest_options = []
        env._enter_map()
        return env.state()
    return env.state()


__all__ = ["legal_actions", "start_combat", "start_event_boss_combat", "state", "step"]
