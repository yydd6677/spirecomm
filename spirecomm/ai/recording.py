import collections
import datetime
import json
import os
import uuid

from spirecomm.ai.serialization import to_primitive
from spirecomm.ai.observation import CANONICAL_STATE_VERSION
from spirecomm.native_sim.cards import CARD_LIBRARY
from spirecomm.spire.screen import ScreenType, RewardType


SCHEMA_NAME = "spirecomm.combat_trajectory"
SCHEMA_VERSION = 1


def serialize_card(card):
    card_def = CARD_LIBRARY.get(card.card_id)
    base_cost = card.cost
    if card_def is not None:
        if card.upgrades > 0 and card_def.upgraded_cost is not None:
            base_cost = card_def.upgraded_cost
        else:
            base_cost = card_def.cost
    return {
        "card_id": card.card_id,
        "name": card.name,
        "uuid": card.uuid,
        "type": card.type.name,
        "rarity": card.rarity.name,
        "cost": card.cost,
        "base_cost": base_cost,
        "cost_for_turn": card.cost,
        "cost_for_combat": None,
        "free_to_play_once": False,
        "upgrades": card.upgrades,
        "misc": card.misc,
        "has_target": card.has_target,
        "is_playable": card.is_playable,
        "exhausts": card.exhausts,
    }


def serialize_potion(potion):
    return {
        "potion_id": potion.potion_id,
        "name": potion.name,
        "can_use": potion.can_use,
        "can_discard": potion.can_discard,
        "requires_target": potion.requires_target,
        "price": potion.price,
    }


def serialize_relic(relic):
    return {
        "relic_id": relic.relic_id,
        "name": relic.name,
        "counter": relic.counter,
        "price": relic.price,
    }


def serialize_power(power):
    return {
        "power_id": power.power_id,
        "name": power.power_name,
        "amount": power.amount,
        "damage": power.damage,
        "misc": power.misc,
        "just_applied": power.just_applied,
        "card": serialize_card(power.card) if power.card is not None else None,
    }


def serialize_orb(orb):
    return {
        "name": orb.name,
        "orb_id": orb.orb_id,
        "evoke_amount": orb.evoke_amount,
        "passive_amount": orb.passive_amount,
    }


def serialize_monster(monster):
    return {
        "monster_index": monster.monster_index,
        "name": monster.name,
        "monster_id": monster.monster_id,
        "current_hp": monster.current_hp,
        "max_hp": monster.max_hp,
        "block": monster.block,
        "intent": monster.intent.name,
        "half_dead": monster.half_dead,
        "is_gone": monster.is_gone,
        "move_id": monster.move_id,
        "last_move_id": monster.last_move_id,
        "second_last_move_id": monster.second_last_move_id,
        "move_base_damage": monster.move_base_damage,
        "move_adjusted_damage": monster.move_adjusted_damage,
        "move_hits": monster.move_hits,
        "powers": [serialize_power(power) for power in monster.powers],
    }


def serialize_player(player):
    return {
        "current_hp": player.current_hp,
        "max_hp": player.max_hp,
        "block": player.block,
        "energy": player.energy,
        "powers": [serialize_power(power) for power in player.powers],
        "orbs": [serialize_orb(orb) for orb in player.orbs],
    }


def serialize_combat_reward(reward):
    payload = {
        "reward_type": reward.reward_type.name,
        "gold": reward.gold,
        "relic": serialize_relic(reward.relic) if reward.relic is not None else None,
        "potion": serialize_potion(reward.potion) if reward.potion is not None else None,
        "link": serialize_relic(reward.link) if reward.link is not None else None,
    }
    return payload


def serialize_screen(screen_type, screen):
    payload = {
        "screen_type": screen_type.name,
        "raw": to_primitive(screen),
    }

    if screen_type == ScreenType.COMBAT_REWARD:
        payload["combat_rewards"] = [serialize_combat_reward(reward) for reward in screen.rewards]
    return payload


def serialize_game_state(game_state):
    payload = {
        "observation_version": CANONICAL_STATE_VERSION,
        "floor": game_state.floor,
        "act": game_state.act,
        "seed": game_state.seed,
        "character": game_state.character.name,
        "ascension_level": game_state.ascension_level,
        "act_boss": game_state.act_boss,
        "gold": game_state.gold,
        "current_hp": game_state.current_hp,
        "max_hp": game_state.max_hp,
        "in_combat": game_state.in_combat,
        "screen": serialize_screen(game_state.screen_type, game_state.screen),
        "screen_up": game_state.screen_up,
        "room_phase": game_state.room_phase.name,
        "room_type": game_state.room_type,
        "choice_available": game_state.choice_available,
        "choice_list": list(game_state.choice_list),
        "commands": {
            "end": game_state.end_available,
            "potion": game_state.potion_available,
            "play": game_state.play_available,
            "proceed": game_state.proceed_available,
            "cancel": game_state.cancel_available,
        },
        "deck": [serialize_card(card) for card in game_state.deck],
        "relics": [serialize_relic(relic) for relic in game_state.relics],
        "potions": [serialize_potion(potion) for potion in game_state.potions],
    }

    if game_state.in_combat:
        payload["combat_state"] = {
            "turn": game_state.turn,
            "cards_discarded_this_turn": game_state.cards_discarded_this_turn,
            "player": serialize_player(game_state.player),
            "monsters": [serialize_monster(monster) for monster in game_state.monsters],
            "hand": [serialize_card(card) for card in game_state.hand],
            "draw_pile": [serialize_card(card) for card in game_state.draw_pile],
            "discard_pile": [serialize_card(card) for card in game_state.discard_pile],
            "exhaust_pile": [serialize_card(card) for card in game_state.exhaust_pile],
            "limbo": [serialize_card(card) for card in game_state.limbo],
            "card_in_play": serialize_card(game_state.card_in_play) if game_state.card_in_play is not None else None,
        }
    else:
        payload["combat_state"] = None

    return payload


def serialize_action(action, game_state):
    payload = {
        "action_class": action.__class__.__name__,
        "command": getattr(action, "command", None),
    }

    for attribute in ["card_index", "target_index", "choice_index", "name", "potion_index", "use"]:
        if hasattr(action, attribute):
            payload[attribute] = getattr(action, attribute)

    if hasattr(action, "card") and action.card is not None:
        payload["card"] = {
            "card_id": action.card.card_id,
            "name": action.card.name,
            "uuid": action.card.uuid,
        }
        if payload.get("card_index", -1) == -1 and action.card in game_state.hand:
            payload["card_index"] = game_state.hand.index(action.card)

    if hasattr(action, "target_monster") and action.target_monster is not None:
        payload["target_monster"] = {
            "monster_index": action.target_monster.monster_index,
            "name": action.target_monster.name,
        }
        if payload.get("target_index") is None:
            payload["target_index"] = action.target_monster.monster_index

    if hasattr(action, "combat_reward") and action.combat_reward is not None:
        payload["combat_reward"] = serialize_combat_reward(action.combat_reward)

    return payload


def inventory_counter(items, key_fn):
    counts = collections.Counter()
    for item in items:
        counts[key_fn(item)] += 1
    return counts


def counter_delta(previous_items, current_items, key_fn):
    previous = inventory_counter(previous_items, key_fn)
    current = inventory_counter(current_items, key_fn)
    gained = []
    lost = []
    for key, count in (current - previous).items():
        gained.extend([key] * count)
    for key, count in (previous - current).items():
        lost.extend([key] * count)
    return gained, lost


def get_real_potions(game_state):
    return [potion for potion in game_state.potions if potion.potion_id != "Potion Slot"]


def living_monsters(game_state):
    if not game_state.in_combat:
        return []
    return [
        monster for monster in game_state.monsters
        if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone
    ]


def total_monster_hp(game_state):
    return sum(monster.current_hp for monster in living_monsters(game_state))


def total_incoming_damage(game_state):
    if not game_state.in_combat:
        return 0
    damage = 0
    for monster in living_monsters(game_state):
        if monster.move_adjusted_damage is not None:
            damage += monster.move_adjusted_damage * monster.move_hits
    return damage


def reward_screen_summary(game_state):
    if game_state.screen_type != ScreenType.COMBAT_REWARD:
        return []
    return [serialize_combat_reward(reward) for reward in game_state.screen.rewards]


def build_transition_delta(previous_state, current_state):
    previous_player = previous_state.player if previous_state.in_combat else None
    current_player = current_state.player if current_state.in_combat else None

    previous_alive = living_monsters(previous_state)
    current_alive = living_monsters(current_state)
    previous_alive_keys = {
        (monster.monster_index, monster.name) for monster in previous_alive
    }
    current_alive_keys = {
        (monster.monster_index, monster.name) for monster in current_alive
    }

    deck_gained, deck_lost = counter_delta(
        previous_state.deck,
        current_state.deck,
        lambda card: "{}+{}".format(card.card_id, card.upgrades),
    )
    relic_gained, relic_lost = counter_delta(
        previous_state.relics,
        current_state.relics,
        lambda relic: relic.relic_id,
    )
    potion_gained, potion_lost = counter_delta(
        get_real_potions(previous_state),
        get_real_potions(current_state),
        lambda potion: potion.potion_id,
    )

    return {
        "current_hp_delta": current_state.current_hp - previous_state.current_hp,
        "max_hp_delta": current_state.max_hp - previous_state.max_hp,
        "gold_delta": current_state.gold - previous_state.gold,
        "floor_delta": current_state.floor - previous_state.floor,
        "deck_size_delta": len(current_state.deck) - len(previous_state.deck),
        "relic_count_delta": len(current_state.relics) - len(previous_state.relics),
        "potion_count_delta": len(get_real_potions(current_state)) - len(get_real_potions(previous_state)),
        "gained_cards": deck_gained,
        "lost_cards": deck_lost,
        "gained_relics": relic_gained,
        "lost_relics": relic_lost,
        "gained_potions": potion_gained,
        "lost_potions": potion_lost,
        "player_block_delta": (
            current_player.block - previous_player.block
            if previous_player is not None and current_player is not None else None
        ),
        "player_energy_delta": (
            current_player.energy - previous_player.energy
            if previous_player is not None and current_player is not None else None
        ),
        "hand_count_delta": (
            len(current_state.hand) - len(previous_state.hand)
            if previous_state.in_combat and current_state.in_combat else None
        ),
        "draw_pile_count_delta": (
            len(current_state.draw_pile) - len(previous_state.draw_pile)
            if previous_state.in_combat and current_state.in_combat else None
        ),
        "discard_pile_count_delta": (
            len(current_state.discard_pile) - len(previous_state.discard_pile)
            if previous_state.in_combat and current_state.in_combat else None
        ),
        "exhaust_pile_count_delta": (
            len(current_state.exhaust_pile) - len(previous_state.exhaust_pile)
            if previous_state.in_combat and current_state.in_combat else None
        ),
        "monster_total_hp_delta": total_monster_hp(current_state) - total_monster_hp(previous_state),
        "monsters_alive_delta": len(current_alive) - len(previous_alive),
        "monsters_killed": [
            {"monster_index": index, "name": name}
            for index, name in sorted(previous_alive_keys - current_alive_keys)
        ],
        "incoming_damage_delta": total_incoming_damage(current_state) - total_incoming_damage(previous_state),
        "screen_type_changed": current_state.screen_type.name != previous_state.screen_type.name,
        "room_phase_changed": current_state.room_phase.name != previous_state.room_phase.name,
        "combat_finished": previous_state.in_combat and not current_state.in_combat,
        "reward_screen_seen": current_state.screen_type == ScreenType.COMBAT_REWARD,
        "reward_screen_summary": reward_screen_summary(current_state),
    }


class TrajectoryRecorder:

    def __init__(self, output_directory, record_mode="combat"):
        self.output_directory = output_directory
        self.record_mode = record_mode
        self.current_run = None
        self.current_combat = None
        self.pending_step = None
        self.combat_index = 0
        self.last_run_id = None
        os.makedirs(self.output_directory, exist_ok=True)

    def start_run(self, player_class, ascension_level=0, seed=None):
        run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.current_run = {
            "run_id": run_id,
            "player_class": player_class.name,
            "ascension_level": ascension_level,
            "seed": seed,
            "started_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        self.current_combat = None
        self.pending_step = None
        self.combat_index = 0

    def should_record_game_state(self, game_state):
        if self.record_mode == "all":
            return True
        return game_state.in_combat or self.current_combat is not None

    def on_state(self, game_state):
        if self.current_run is None or not self.should_record_game_state(game_state):
            return

        if game_state.in_combat and self.current_combat is None:
            self._start_combat(game_state)

        if self.current_combat is None:
            return

        if self.pending_step is not None:
            self._finalize_pending_step(game_state)

        if not game_state.in_combat:
            self._finish_combat(game_state)

    def record_step(self, game_state, action, source):
        if self.current_combat is None or not game_state.in_combat:
            return

        self.pending_step = {
            "step_index": self.current_combat["next_step_index"],
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "source": source,
            "action": serialize_action(action, game_state),
            "state_before": serialize_game_state(game_state),
            "raw_state_before": game_state,
        }
        self.current_combat["next_step_index"] += 1

    def record_preference(self, game_state, source, chosen_action, chosen_score, candidate_scores):
        if self.current_combat is None or not game_state.in_combat:
            return
        if chosen_action is None or not candidate_scores or len(candidate_scores) < 2:
            return

        serialized_candidates = []
        seen_keys = set()
        for action, score in candidate_scores:
            if action is None:
                continue
            payload = serialize_action(action, game_state)
            payload_key = json.dumps(payload, sort_keys=True)
            if payload_key in seen_keys:
                continue
            seen_keys.add(payload_key)
            serialized_candidates.append({
                "action": payload,
                "score": float(score),
                "preferred": payload == serialize_action(chosen_action, game_state),
            })

        if len(serialized_candidates) < 2:
            return

        self._write_record({
            "record_type": "preference",
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "run_id": self.current_run["run_id"],
            "combat_id": self.current_combat["combat_id"],
            "combat_index": self.current_combat["combat_index"],
            "step_index": self.current_combat["next_step_index"],
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "source": source,
            "state_before": serialize_game_state(game_state),
            "chosen_action": serialize_action(chosen_action, game_state),
            "chosen_score": float(chosen_score),
            "candidates": serialized_candidates,
        })

    def end_run(self, victory):
        if self.current_run is not None:
            self.last_run_id = self.current_run["run_id"]
        if self.current_combat is not None:
            if self.pending_step is not None:
                self._write_record({
                    "record_type": "transition",
                    "schema_name": SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.current_run["run_id"],
                    "combat_id": self.current_combat["combat_id"],
                    "combat_index": self.current_combat["combat_index"],
                    "step_index": self.pending_step["step_index"],
                    "timestamp": self.pending_step["timestamp"],
                    "source": self.pending_step["source"],
                    "action": self.pending_step["action"],
                    "state_before": self.pending_step["state_before"],
                    "state_after": None,
                    "delta": None,
                    "terminal": True,
                    "truncated": True,
                })
                self.pending_step = None
            self._write_summary(final_state=None, victory=victory, truncated=True)
            self.current_combat["file"].close()
            self.current_combat = None

        self.current_run = None

    def _start_combat(self, game_state):
        combat_id = "{}_combat_{:03d}".format(self.current_run["run_id"], self.combat_index)
        path = os.path.join(self.output_directory, "{}.jsonl".format(combat_id))
        combat_file = open(path, "a")
        self.current_combat = {
            "combat_id": combat_id,
            "combat_index": self.combat_index,
            "path": path,
            "file": combat_file,
            "started_at": datetime.datetime.utcnow().isoformat() + "Z",
            "start_state": game_state,
            "next_step_index": 0,
        }
        self.combat_index += 1

        self._write_record({
            "record_type": "meta",
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "run": self.current_run,
            "combat": {
                "combat_id": combat_id,
                "combat_index": self.current_combat["combat_index"],
                "started_at": self.current_combat["started_at"],
                "act": game_state.act,
                "floor": game_state.floor,
                "room_type": game_state.room_type,
                "act_boss": game_state.act_boss,
                "character": game_state.character.name,
                "ascension_level": game_state.ascension_level,
                "initial_state": serialize_game_state(game_state),
            },
        })

    def _finalize_pending_step(self, game_state):
        previous_state = self.pending_step["raw_state_before"]
        self._write_record({
            "record_type": "transition",
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "run_id": self.current_run["run_id"],
            "combat_id": self.current_combat["combat_id"],
            "combat_index": self.current_combat["combat_index"],
            "step_index": self.pending_step["step_index"],
            "timestamp": self.pending_step["timestamp"],
            "source": self.pending_step["source"],
            "action": self.pending_step["action"],
            "state_before": self.pending_step["state_before"],
            "state_after": serialize_game_state(game_state),
            "delta": build_transition_delta(previous_state, game_state),
            "terminal": not game_state.in_combat,
            "truncated": False,
        })
        self.pending_step = None

    def _finish_combat(self, game_state):
        self._write_summary(final_state=game_state, victory=None, truncated=False)
        self.current_combat["file"].close()
        self.current_combat = None

    def _write_summary(self, final_state, victory, truncated):
        if final_state is not None:
            summary = {
                "finished_at": datetime.datetime.utcnow().isoformat() + "Z",
                "steps": self.current_combat["next_step_index"],
                "truncated": truncated,
                "run_victory": victory,
                "player_hp": final_state.current_hp,
                "player_max_hp": final_state.max_hp,
                "gold": final_state.gold,
                "floor": final_state.floor,
                "screen_type": final_state.screen_type.name,
                "room_phase": final_state.room_phase.name,
                "reward_screen_summary": reward_screen_summary(final_state),
                "state_after_combat": serialize_game_state(final_state),
            }
        else:
            summary = {
                "finished_at": datetime.datetime.utcnow().isoformat() + "Z",
                "steps": self.current_combat["next_step_index"],
                "truncated": truncated,
                "run_victory": victory,
                "state_after_combat": None,
            }

        self._write_record({
            "record_type": "summary",
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "run_id": self.current_run["run_id"],
            "combat_id": self.current_combat["combat_id"],
            "combat_index": self.current_combat["combat_index"],
            "summary": summary,
        })

    def _write_record(self, payload):
        self.current_combat["file"].write(json.dumps(payload, sort_keys=True) + "\n")
        self.current_combat["file"].flush()
