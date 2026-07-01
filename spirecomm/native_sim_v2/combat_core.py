from __future__ import annotations

from spirecomm.native_sim.schema import PotionInstance
from spirecomm.native_sim.schema import CardInstance
from spirecomm.native_sim_v2.helpers_cards import clone_card
from spirecomm.native_sim_v2.helpers_common import _card_can_upgrade


UPGRADED_NON_EXHAUST_CARD_IDS = {
    "Calculated Gamble",
    "Discovery",
    "Hologram",
    "Limit Break",
    "Rainbow",
    "Secret Technique",
    "Secret Weapon",
    "Thinking Ahead",
}


def card_exhausts_on_use(card: CardInstance) -> bool:
    if card.upgrades > 0 and card.card_id in UPGRADED_NON_EXHAUST_CARD_IDS:
        return False
    return bool(card.card_def.exhausts)


def resolve_after_use_card_move(
    env,
    card: CardInstance,
    *,
    force_exhaust: bool = False,
    defer_dark_embrace_draws: list[int] | None = None,
    add_hex_dazed: int = 0,
) -> None:
    for _ in range(max(0, int(add_hex_dazed))):
        env._insert_temp_card_into_draw_pile("Dazed")
    if env.pending_resolve_card is card and env.pending_resolve_used_free_to_play_once:
        card.free_to_play_once = False
        env.pending_resolve_used_free_to_play_once = False
    if force_exhaust or card_exhausts_on_use(card) or (env.player.power("Corruption") > 0 and card.card_def.card_type == "SKILL"):
        env._exhaust_card(card, defer_dark_embrace_draws=defer_dark_embrace_draws)
    elif card.card_def.card_type != "POWER":
        env._move_card_to_discard(card)


def resolve_pending_delayed_reactions(env) -> None:
    if env.card_select_context is None:
        env._resolve_pending_monster_block_gains()
        env._resolve_pending_monster_direct_damage()
        env._resolve_pending_spore_cloud_triggers()
        env._check_outcome()


def step(env, action: dict[str, object]):
    kind = action.get("kind")
    if kind in {"card_select", "single_card_select", "multi_card_select"}:
        env._resolve_card_select(action)
        resolve_pending_delayed_reactions(env)
        return env.to_spirecomm_state()
    if kind == "end":
        env.end_turn()
    elif kind == "card":
        env.play_card(int(action.get("card_index", 0)), int(action.get("target_index", 0) or 0))
    elif kind == "potion":
        potion_index = int(action.get("potion_index", 0))
        if action.get("action") == "discard":
            env.potions[potion_index] = PotionInstance()
        else:
            env.use_potion(potion_index, int(action.get("target_index", 0) or 0))
    else:
        raise ValueError(f"unsupported native combat action: {action}")
    resolve_pending_delayed_reactions(env)
    return env.to_spirecomm_state()


def play_card(
    env,
    hand_index: int,
    target_index: int,
    *,
    force_exhaust: bool,
    free_to_play: bool,
    energy_on_use: int | None = None,
) -> None:
    previous_processing_player_action = env.processing_player_action
    env.processing_player_action = True
    try:
        env.play_card_impl(
            hand_index,
            target_index,
            force_exhaust=force_exhaust,
            free_to_play=free_to_play,
            energy_on_use=energy_on_use,
        )
        resolve_pending_delayed_reactions(env)
    finally:
        env.processing_player_action = previous_processing_player_action


def end_turn(env) -> None:
    env.end_turn_impl()
    resolve_pending_delayed_reactions(env)


def replay_attack_card_effect(env, card: CardInstance, target, energy_spent: int) -> None:
    env._replay_attack_card_effect(card, target, energy_spent)


def resolve_card_select(env, action: dict[str, object]) -> bool:
    context = env.card_select_context
    pending_status_effects = list(env.pending_resolve_status_effects)
    env.pending_resolve_status_effects = []
    hand_index = action.get("select_index")
    if hand_index is None:
        hand_index = action.get("deck_index")
    if hand_index is None:
        hand_index = action.get("choice_index")
    if hand_index is None:
        raise ValueError("card_select action missing index")
    hand_index = int(hand_index)

    if context == "ARMAMENTS":
        if 0 <= hand_index < len(env.hand):
            upgraded_card = env.hand[hand_index]
            env._upgrade_combat_card(upgraded_card)
            valid_cards: list[CardInstance] = []
            invalid_cards: list[CardInstance] = []
            for index, card in enumerate(env.hand):
                if index == hand_index:
                    continue
                if _card_can_upgrade(card):
                    valid_cards.append(card)
                else:
                    invalid_cards.append(card)
            env.hand = valid_cards + [upgraded_card] + invalid_cards
        if env.pending_resolve_card is not None:
            if pending_status_effects:
                env._resolve_deferred_status_draw_effects(pending_status_effects)
                pending_status_effects = []
            resolve_after_use_card_move(
                env,
                env.pending_resolve_card,
                force_exhaust=env.pending_resolve_force_exhaust,
                add_hex_dazed=env.pending_resolve_hex_dazed,
            )
    elif context in {"BURNING_PACT", "EXHAUST_ONE"}:
        deferred_dark_embrace_draws: list[int] = []
        deferred_after_exhaust_actions: list[tuple[str, int | str]] = []
        pending_hex_dazed = env.pending_resolve_hex_dazed
        if 0 <= hand_index < len(env.hand):
            exhausted_card = env.hand.pop(hand_index)
            pending_card = env.pending_resolve_card
            if pending_card is not None and pending_card.card_id == "Burning Pact":
                env._exhaust_card(exhausted_card, deferred_after_exhaust_actions=deferred_after_exhaust_actions)
            else:
                env._exhaust_card(exhausted_card, defer_dark_embrace_draws=deferred_dark_embrace_draws)
            pending_card = env.pending_resolve_card
        if pending_card is not None and pending_card.card_id == "True Grit":
            resolve_after_use_card_move(
                env,
                pending_card,
                force_exhaust=env.pending_resolve_force_exhaust,
                defer_dark_embrace_draws=deferred_dark_embrace_draws,
                add_hex_dazed=pending_hex_dazed,
            )
            if pending_status_effects:
                env._resolve_deferred_status_draw_effects(pending_status_effects)
                pending_status_effects = []
        else:
            draw_count = 2
            if pending_card is not None and pending_card.upgrades > 0:
                draw_count = 3
            pending_card_moved_before_deferred_draws = False
            env.draw_cards(draw_count, deferred_status_effects=pending_status_effects)
            if pending_card is not None and pending_card.card_id == "Burning Pact" and deferred_after_exhaust_actions:
                pre_move_actions = [
                    action for action in deferred_after_exhaust_actions
                    if action[0] == "add_to_hand"
                ]
                if pre_move_actions:
                    if len(env.hand) >= 10 and not pending_card_moved_before_deferred_draws:
                        resolve_after_use_card_move(
                            env,
                            pending_card,
                            force_exhaust=env.pending_resolve_force_exhaust,
                            defer_dark_embrace_draws=deferred_dark_embrace_draws,
                            add_hex_dazed=pending_hex_dazed,
                        )
                        pending_card_moved_before_deferred_draws = True
                        pending_hex_dazed = 0
                    env._resolve_deferred_after_exhaust_actions(pre_move_actions)
                    deferred_after_exhaust_actions = [
                        action for action in deferred_after_exhaust_actions
                        if action[0] != "add_to_hand"
                    ]
            deferred_draw_count = sum(
                amount
                for effect_kind, amount in pending_status_effects
                if effect_kind == "draw" and amount > 0
            )
            evolve_status_draw_chain_pending = (
                deferred_draw_count > 0
                and env.player.power("Evolve") > 0
                and any(draw.card_def.card_type == "STATUS" for draw in env.draw_pile)
            )
            ink_bottle_only_pending_draw = (
                pending_status_effects == [("draw", 1)]
                and any(relic.get("relic_id") == "Ink Bottle" and relic.get("counter") == 0 for relic in env.relics)
            )
            if (
                pending_card is not None
                and pending_card.card_id == "Burning Pact"
                and (deferred_draw_count > len(env.draw_pile) or evolve_status_draw_chain_pending)
                and not ink_bottle_only_pending_draw
            ):
                resolve_after_use_card_move(
                    env,
                    pending_card,
                    force_exhaust=env.pending_resolve_force_exhaust,
                    defer_dark_embrace_draws=deferred_dark_embrace_draws,
                    add_hex_dazed=pending_hex_dazed,
                )
                pending_card_moved_before_deferred_draws = True
                pending_hex_dazed = 0
            if pending_card is not None and pending_card.card_id == "Burning Pact" and pending_status_effects:
                if pending_hex_dazed > 0:
                    for _ in range(pending_hex_dazed):
                        env._insert_temp_card_into_draw_pile("Dazed")
                    pending_hex_dazed = 0
                env._resolve_deferred_status_draw_effects(pending_status_effects)
                pending_status_effects = []
            if pending_card is not None and not pending_card_moved_before_deferred_draws:
                resolve_after_use_card_move(
                    env,
                    pending_card,
                    force_exhaust=env.pending_resolve_force_exhaust,
                    defer_dark_embrace_draws=deferred_dark_embrace_draws,
                    add_hex_dazed=pending_hex_dazed,
                )
            if pending_status_effects:
                env._resolve_deferred_status_draw_effects(pending_status_effects)
                pending_status_effects = []
        if deferred_after_exhaust_actions:
            env._resolve_deferred_after_exhaust_actions(deferred_after_exhaust_actions)
        for deferred_draw_count in deferred_dark_embrace_draws:
            env.draw_cards(deferred_draw_count)
    elif context == "DUAL_WIELD":
        if 0 <= hand_index < len(env.hand):
            selected_card = clone_card(env.hand[hand_index], reset_cost_for_turn=False)
            selected_card.uuid = env._new_uuid(f"dual-wield-{selected_card.card_id}")
            valid_cards: list[CardInstance] = []
            invalid_cards: list[CardInstance] = []
            for index, card in enumerate(env.hand):
                if index == hand_index:
                    continue
                if card.card_def.card_type in {"ATTACK", "POWER"}:
                    valid_cards.append(card)
                else:
                    invalid_cards.append(card)
            env.hand = valid_cards + invalid_cards + [selected_card]
            copy_count = 1
            if env.pending_resolve_card is not None and env.pending_resolve_card.upgrades > 0:
                copy_count = 2
            for _ in range(copy_count):
                copied_card = clone_card(selected_card, reset_cost_for_turn=False)
                copied_card.uuid = env._new_uuid(f"dual-wield-{selected_card.card_id}")
                if len(env.hand) < 10:
                    env.hand.append(copied_card)
                else:
                    env.discard_pile.append(copied_card)
        if pending_status_effects:
            env._resolve_deferred_status_draw_effects(pending_status_effects)
            pending_status_effects = []
        if env.pending_resolve_card is not None:
            resolve_after_use_card_move(
                env,
                env.pending_resolve_card,
                force_exhaust=env.pending_resolve_force_exhaust,
                add_hex_dazed=env.pending_resolve_hex_dazed,
            )
    elif context == "HEADBUTT":
        original_counter_damage = env.pending_counter_damage
        env.pending_counter_damage = 0
        if 0 <= hand_index < len(env.discard_pile):
            selected_card = env.discard_pile.pop(hand_index)
            env.draw_pile.append(selected_card)
        pending_card = env.pending_resolve_card
        pending_replays = env.pending_attack_replays
        pending_target_index = env.pending_attack_target_index
        if pending_card is not None:
            resolve_after_use_card_move(
                env,
                pending_card,
                force_exhaust=env.pending_resolve_force_exhaust,
                add_hex_dazed=env.pending_resolve_hex_dazed,
            )
        if pending_status_effects:
            env._resolve_deferred_status_draw_effects(pending_status_effects)
            pending_status_effects = []
        env._resolve_pending_monster_kill_triggers()
        env._resolve_pending_attack_relic_proc()
        if env.pending_after_use_energy_gain > 0:
            env.player.energy += env.pending_after_use_energy_gain
            env.pending_after_use_energy_gain = 0
        if original_counter_damage > 0:
            env._take_counter_damage(original_counter_damage)
        if pending_replays > 0 and pending_card is not None:
            env._resolve_pending_monster_block_gains()
            target = None
            if pending_target_index is not None and 0 <= pending_target_index < len(env.monsters):
                target = env.monsters[pending_target_index]
            for _ in range(pending_replays):
                if env._resolve_headbutt_replay(pending_card, target):
                    env.pending_resolve_card = None
                    return True
                if env.outcome != "UNDECIDED":
                    break
        if env.pending_counter_damage > 0:
            counter_damage = env.pending_counter_damage
            env.pending_counter_damage = 0
            env._take_counter_damage(counter_damage)
        if env._has_relic("Unceasing Top") and not env.hand:
            if env.draw_pile:
                env.draw_cards(1)
            elif env.discard_pile:
                env.pending_unceasing_top_draw = True
    elif context == "WARCRY":
        if 0 <= hand_index < len(env.hand):
            selected_card = env.hand.pop(hand_index)
            env.draw_pile.append(selected_card)
        if env.pending_resolve_card is not None:
            resolve_after_use_card_move(
                env,
                env.pending_resolve_card,
                force_exhaust=env.pending_resolve_force_exhaust,
                add_hex_dazed=env.pending_resolve_hex_dazed,
            )
        if pending_status_effects:
            env._resolve_deferred_status_draw_effects(pending_status_effects)
            pending_status_effects = []
    elif context == "FORETHOUGHT":
        if 0 <= hand_index < len(env.hand):
            selected_card = env.hand.pop(hand_index)
            if selected_card.cost > 0:
                selected_card.free_to_play_once = True
            env.draw_pile.insert(0, selected_card)
        if env.pending_resolve_card is not None:
            resolve_after_use_card_move(
                env,
                env.pending_resolve_card,
                force_exhaust=env.pending_resolve_force_exhaust,
                add_hex_dazed=env.pending_resolve_hex_dazed,
            )
    elif context == "EXHUME":
        exhaust_index = action.get("deck_index")
        if exhaust_index is None:
            exhaust_index = action.get("select_index")
        if exhaust_index is None:
            exhaust_index = action.get("choice_index")
        if exhaust_index is not None:
            exhaust_index = int(exhaust_index)
        else:
            exhaust_index = -1
        if 0 <= exhaust_index < len(env.exhaust_pile):
            selected_card = env.exhaust_pile.pop(exhaust_index)
            if len(env.hand) < 10:
                env.hand.append(selected_card)
            else:
                env.discard_pile.append(selected_card)
        if env.pending_resolve_card is not None:
            resolve_after_use_card_move(
                env,
                env.pending_resolve_card,
                force_exhaust=True,
                add_hex_dazed=env.pending_resolve_hex_dazed,
            )
    elif context == "GAMBLE":
        pass
    elif context == "EXHAUST_MANY":
        if env.pending_resolve_card is not None:
            resolve_after_use_card_move(
                env,
                env.pending_resolve_card,
                force_exhaust=env.pending_resolve_force_exhaust,
                add_hex_dazed=env.pending_resolve_hex_dazed,
            )
    elif context == "DISCOVERY":
        if 0 <= hand_index < len(env.card_select_generated_cards):
            selected_card = clone_card(env.card_select_generated_cards[hand_index])
            selected_card.cost_for_turn = 0
            if env.player.power("Corruption") > 0 and selected_card.card_def.card_type == "SKILL":
                selected_card.cost_for_turn = -9
            if len(env.hand) >= 10:
                env.discard_pile.append(selected_card)
            else:
                env.hand.append(selected_card)
        if env.pending_resolve_card is not None:
            resolve_after_use_card_move(
                env,
                env.pending_resolve_card,
                force_exhaust=env.pending_resolve_force_exhaust,
                add_hex_dazed=env.pending_resolve_hex_dazed,
            )
        env._resolve_pending_juggernaut_damage()
    elif context == "CODEX":
        if 0 <= hand_index < len(env.card_select_generated_cards):
            selected_card = clone_card(env.card_select_generated_cards[hand_index])
            insert_index = 0 if not env.draw_pile else int(env.card_random_rng.random(len(env.draw_pile) - 1))
            env.draw_pile.insert(insert_index, selected_card)
        env.card_select_context = None
        env.card_select_generated_cards = []
        env.card_select_source_indexes = []
        env.card_select_options = []
        if env.pending_end_turn_resume:
            env.end_turn_impl()
            return True
    elif context in {"SECRET_TECHNIQUE", "SECRET_WEAPON"}:
        env._choose_draw_pile_card_to_hand(hand_index)
        if env.pending_resolve_card is not None:
            resolve_after_use_card_move(
                env,
                env.pending_resolve_card,
                force_exhaust=env.pending_resolve_force_exhaust,
                add_hex_dazed=env.pending_resolve_hex_dazed,
            )
    else:
        return False

    if pending_status_effects:
        env._resolve_deferred_status_draw_effects(pending_status_effects)
    env._resolve_pending_juggernaut_damage()
    if env.pending_after_use_direct_damage_all > 0:
        deferred_damage = env.pending_after_use_direct_damage_all
        env.pending_after_use_direct_damage_all = 0
        env._deal_direct_damage_all(deferred_damage)
    env.pending_resolve_card = None
    env.pending_resolve_force_exhaust = False
    env.pending_resolve_hex_dazed = 0
    env.pending_resolve_used_free_to_play_once = False
    env.pending_resolve_status_effects = []
    env.pending_counter_damage = 0
    env.pending_attack_replays = 0
    env.pending_attack_target_index = None
    env.pending_attack_relic_proc = False
    env.pending_after_use_energy_gain = 0
    env.card_select_context = None
    env.card_select_generated_cards = []
    env.card_select_source_indexes = []
    env.card_select_options = []
    env._check_outcome()
    if env.pending_start_turn_resume and env.card_select_context is None and env.outcome == "UNDECIDED":
        env._resume_start_player_turn_after_autoplay()
    return True


__all__ = [
    "UPGRADED_NON_EXHAUST_CARD_IDS",
    "card_exhausts_on_use",
    "end_turn",
    "play_card",
    "replay_attack_card_effect",
    "resolve_after_use_card_move",
    "resolve_card_select",
    "resolve_pending_delayed_reactions",
    "step",
]
