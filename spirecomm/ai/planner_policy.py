import os
import sys

from spirecomm.ai.learned_policy import CheckpointCombatPolicy
from spirecomm.ai.recording import build_transition_delta, serialize_game_state
from spirecomm.ai.rl import BLOCK_CARD_IDS, compute_reward
from spirecomm.communication.action import CommittedAction, EndTurnAction, PlayCardAction
from spirecomm.spire.card import CardType
from spirecomm.spire.game import RoomPhase
from spirecomm.spire.screen import ScreenType


BASE_UNDO_CLICK_X = 145.0
BASE_UNDO_CLICK_Y = 762.0
UNDO_VISIBLE_SCREEN_TYPES = {ScreenType.HAND_SELECT, ScreenType.GRID, ScreenType.CARD_REWARD}
KNOWN_ATTACK_DAMAGE = {
    "Strike": 6,
    "Strike_R": 6,
    "Bash": 8,
}


def living_monsters(game_state):
    return [
        monster
        for monster in game_state.monsters
        if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone
    ]


def total_monster_hp(game_state):
    return sum(monster.current_hp + monster.block for monster in living_monsters(game_state))


def total_incoming_damage(game_state):
    total = 0
    for monster in living_monsters(game_state):
        if monster.move_adjusted_damage is not None and monster.move_adjusted_damage > 0:
            total += monster.move_adjusted_damage * monster.move_hits
    return total


def is_defensive_card(card):
    if card is None:
        return False
    return getattr(card, "card_id", None) in BLOCK_CARD_IDS or getattr(card, "name", None) in BLOCK_CARD_IDS


def only_defensive_playables(game_state):
    playable_cards = [
        card for card in game_state.hand
        if card is not None and getattr(card, "is_playable", False)
    ]
    if not playable_cards:
        return False
    return all(is_defensive_card(card) for card in playable_cards)


def action_energy_cost(game_state, action):
    if not isinstance(action, PlayCardAction):
        return 0.0
    card = getattr(action, "card", None)
    if card is None:
        return 0.0
    cost = getattr(card, "cost", 0)
    if cost is None:
        return 0.0
    if cost < 0:
        player = getattr(game_state, "player", None)
        return float(getattr(player, "energy", 0) or 0.0)
    return float(max(cost, 0))


def reward_efficiency_denominator(energy_spent):
    if energy_spent <= 0.0:
        return 0.5
    return energy_spent


class UndoPlannerCombatPolicy:

    source_name = "UndoPlannerCombatPolicy"

    def __init__(self):
        self.base_policy = CheckpointCombatPolicy()
        self.top_k = int(os.environ.get("SPIRECOMM_PLANNER_TOP_K", "3"))
        self.max_updates = int(os.environ.get("SPIRECOMM_PLANNER_MAX_UPDATES", "500"))
        self.click_timeout = int(os.environ.get("SPIRECOMM_PLANNER_CLICK_TIMEOUT", "50"))
        self.state_refreshes = int(os.environ.get("SPIRECOMM_PLANNER_STATE_REFRESHES", "3"))
        self.log_path = os.environ.get("SPIRECOMM_PLANNER_LOG", "/home/yydd/spirecomm/planner.log")

    def reload(self):
        self.base_policy.reload()

    def choose_action(self, game_state, fallback_agent, coordinator=None):
        live_state = coordinator.last_game_state if coordinator is not None and coordinator.last_game_state is not None else game_state
        recorder = getattr(fallback_agent, "recorder", None)
        if coordinator is None or not self._can_plan(live_state):
            self._log("skip plan: can_plan=false floor={} turn={} play={} choice={} screen_up={}".format(
                live_state.floor,
                live_state.turn,
                live_state.play_available,
                live_state.choice_available,
                live_state.screen_up,
            ))
            return self.base_policy.choose_action(live_state, fallback_agent, coordinator=coordinator)

        serialized_state = serialize_game_state(live_state)
        scoring = self.base_policy.score_state(serialized_state)
        candidates = self._build_candidates(live_state, fallback_agent, scoring)
        if len(candidates) < 2:
            self._log("skip plan: candidate_count={} floor={} turn={}".format(
                len(candidates),
                live_state.floor,
                live_state.turn,
            ))
            return self.base_policy.choose_action(live_state, fallback_agent, coordinator=coordinator)

        for candidate in candidates:
            rebound_action = self._rebind_action(candidate, live_state, fallback_agent)
            if rebound_action is None or not rebound_action.can_be_executed(coordinator):
                continue
            if self._is_finishing_attack_candidate(live_state, rebound_action):
                self._log("selected immediate finishing attack: {}".format(
                    self._describe_action(rebound_action),
                ))
                return rebound_action

        original_signature = self._state_signature(live_state)
        best_action = None
        best_score = None
        restore_failed = False
        tried_count = 0
        branch_results = []

        for candidate in candidates:
            tried_count += 1

            branch_state = self._execute_candidate(coordinator, candidate)
            if branch_state is None:
                self._log("branch execute failed: {}".format(self._describe_action(candidate)))
                continue

            if not self._undo_available(branch_state):
                if self._is_committed_terminal_state(branch_state):
                    self._log("committing terminal branch without undo: {}".format(self._describe_action(candidate)))
                    return CommittedAction(candidate)
                if not self._restore_original_state(coordinator, original_signature):
                    restore_failed = True
                    self._log("restore failed after branch without undo visibility: {}".format(self._describe_action(candidate)))
                    break
                continue

            branch_score = self._score_branch(live_state, branch_state, candidate)
            branch_results.append((candidate, branch_score))
            self._log("branch score {} => {:.4f}".format(self._describe_action(candidate), branch_score))
            if not self._restore_original_state(coordinator, original_signature):
                restore_failed = True
                self._log("restore failed after scoring branch: {}".format(self._describe_action(candidate)))
                break

            if best_score is None or branch_score > best_score:
                best_score = branch_score
                best_action = candidate

        current_state = coordinator.last_game_state if coordinator is not None and coordinator.last_game_state is not None else live_state
        if restore_failed or self._state_signature(current_state) != original_signature:
            if coordinator is not None:
                coordinator.send_and_wait("state", max_updates=self.max_updates)
                if coordinator.last_game_state is not None:
                    current_state = coordinator.last_game_state
            self._log("fallback base policy after restore mismatch/failure floor={} turn={} tried_candidates={}".format(
                current_state.floor,
                current_state.turn,
                tried_count,
            ))
            return self._fallback_action(current_state, fallback_agent, coordinator)

        if best_action is not None:
            rebound_action = self._rebind_action(best_action, current_state, fallback_agent)
            if rebound_action is not None and rebound_action.can_be_executed(coordinator):
                rebound_candidates = []
                for candidate, score in branch_results:
                    rebound_candidate = self._rebind_action(candidate, current_state, fallback_agent)
                    if rebound_candidate is not None and rebound_candidate.can_be_executed(coordinator):
                        rebound_candidates.append((rebound_candidate, score))
                if recorder is not None and len(rebound_candidates) >= 2:
                    recorder.record_preference(
                        current_state,
                        self.source_name,
                        rebound_action,
                        best_score,
                        rebound_candidates,
                    )
                self._log("selected planned action: {} score={:.4f}".format(
                    self._describe_action(rebound_action),
                    best_score,
                ))
                return rebound_action

        self._log("fallback base policy after planning floor={} turn={} tried_candidates={}".format(
            current_state.floor,
            current_state.turn,
            tried_count,
        ))
        return self._fallback_action(current_state, fallback_agent, coordinator)

    def _can_plan(self, game_state):
        return game_state.in_combat and game_state.play_available and not game_state.choice_available and self._undo_available(game_state)

    def _is_committed_terminal_state(self, game_state):
        if game_state is None:
            return False
        return (not game_state.in_combat) or game_state.room_phase != RoomPhase.COMBAT

    def _undo_available(self, game_state):
        if not game_state.in_combat:
            return False
        if game_state.screen_up:
            return game_state.screen_type in UNDO_VISIBLE_SCREEN_TYPES
        return True

    def _build_candidates(self, game_state, fallback_agent, scoring):
        action_scores = scoring["action_logits"][0]
        target_scores = scoring["target_logits"][0]
        action_mask = scoring["action_mask"][0]

        legal_indices = [index for index, allowed in enumerate(action_mask.tolist()) if allowed]
        ranked_indices = sorted(legal_indices, key=lambda index: float(action_scores[index]), reverse=True)

        candidates = []
        seen = set()
        for action_index in ranked_indices[: max(self.top_k, 1)]:
            target_index = int(target_scores.argmax(dim=-1).item())
            action = self.base_policy.decode_action_from_prediction(
                game_state,
                fallback_agent,
                action_index,
                target_index,
            )
            action_key = self._action_key(action)
            if action is None or action_key in seen:
                continue
            seen.add(action_key)
            candidates.append(action)
        return candidates

    def _action_key(self, action):
        if isinstance(action, EndTurnAction):
            return ("end",)
        if isinstance(action, PlayCardAction):
            return ("play", getattr(action.card, "uuid", None), getattr(action, "target_index", None))
        return (action.__class__.__name__, getattr(action, "command", None))

    def _execute_candidate(self, coordinator, action):
        action.execute(coordinator)
        if not coordinator.wait_for_command_state(block=True, max_updates=self.max_updates):
            return None
        if coordinator.last_error is not None or not coordinator.in_game or coordinator.last_game_state is None:
            return None
        return coordinator.last_game_state

    def _restore_original_state(self, coordinator, original_signature):
        self._log("sending undo command")
        if not coordinator.send_and_wait("undo", max_updates=self.max_updates):
            return False
        if self._state_signature(coordinator.last_game_state) == original_signature:
            self._log("restore success after undo")
            return True

        for _ in range(self.state_refreshes):
            if not coordinator.send_and_wait("state", max_updates=self.max_updates):
                return False
            if self._state_signature(coordinator.last_game_state) == original_signature:
                self._log("restore success after state refresh")
                return True
        self._log("restore failed after state refreshes")
        return False

    def _score_branch(self, original_state, branch_state, action):
        state_before = serialize_game_state(original_state)
        state_after = serialize_game_state(branch_state)
        record = {
            "state_before": state_before,
            "state_after": state_after,
            "action": self._action_payload(action),
            "delta": build_transition_delta(original_state, branch_state),
        }

        terminal_summary = None
        if record["delta"].get("combat_finished"):
            terminal_summary = {
                "final_screen_type": branch_state.screen_type.name,
                "run_victory": getattr(getattr(branch_state, "screen", None), "victory", False)
                if branch_state.screen_type == ScreenType.GAME_OVER else True,
            }

        immediate_reward = compute_reward(record, terminal_summary=terminal_summary)

        tail_value = 0.0
        if branch_state.in_combat and branch_state.room_phase == RoomPhase.COMBAT:
            serialized_branch = serialize_game_state(branch_state)
            scoring = self.base_policy.score_state(serialized_branch)
            tail_value = float(scoring["value"][0].item())

        return immediate_reward + tail_value

    def _action_payload(self, action):
        payload = {
            "command": getattr(action, "command", "state"),
            "action_class": action.__class__.__name__,
        }
        if isinstance(action, PlayCardAction):
            payload["card_index"] = getattr(action, "card_index", None)
            payload["target_index"] = getattr(action, "target_index", None)
        return payload

    def _state_signature(self, game_state):
        if game_state is None:
            return None

        player = game_state.player
        monsters = tuple(
            (
                monster.monster_id,
                monster.current_hp,
                monster.block,
                monster.intent.name,
                monster.move_adjusted_damage,
                monster.move_hits,
            )
            for monster in game_state.monsters
        )
        hand = tuple(card.uuid for card in game_state.hand)
        powers = tuple((power.power_id, power.amount) for power in (player.powers if player is not None else []))
        return (
            game_state.floor,
            game_state.room_type,
            game_state.screen_type.name,
            game_state.screen_up,
            game_state.turn,
            game_state.current_hp,
            player.block if player is not None else 0,
            player.energy if player is not None else 0,
            hand,
            len(game_state.draw_pile),
            len(game_state.discard_pile),
            len(game_state.exhaust_pile),
            monsters,
            powers,
        )

    def _rebind_action(self, action, game_state, fallback_agent):
        if isinstance(action, EndTurnAction):
            return EndTurnAction()

        if not isinstance(action, PlayCardAction):
            return None

        card_index = getattr(action, "card_index", -1)
        if card_index is not None and card_index >= 0 and card_index < len(game_state.hand):
            card = game_state.hand[card_index]
            if card.is_playable:
                return PlayCardAction(card_index=card_index, target_index=getattr(action, "target_index", None))

        action_card = getattr(action, "card", None)
        action_uuid = getattr(action_card, "uuid", None)
        if action_uuid is not None:
            for index, card in enumerate(game_state.hand):
                if card.uuid == action_uuid and card.is_playable:
                    return PlayCardAction(card_index=index, target_index=getattr(action, "target_index", None))

        return self.base_policy.choose_action(game_state, fallback_agent)

    def _describe_action(self, action):
        if isinstance(action, EndTurnAction):
            return "end"
        if isinstance(action, PlayCardAction):
            card = getattr(action, "card", None)
            card_name = getattr(card, "name", None)
            return "play {} idx={} target={}".format(card_name, getattr(action, "card_index", None), getattr(action, "target_index", None))
        return action.__class__.__name__

    def _log(self, message):
        line = "UndoPlanner: {}\n".format(message)
        try:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass
        print(line, file=sys.stderr, end="")

    def _fallback_action(self, game_state, fallback_agent, coordinator):
        if game_state is not None and game_state.in_combat and game_state.play_available and not game_state.choice_available:
            return self.base_policy.choose_action(game_state, fallback_agent, coordinator=coordinator)
        return fallback_agent.get_next_action_in_game(game_state)

    def _find_target_by_index(self, game_state, target_index):
        for monster in living_monsters(game_state):
            if monster.monster_index == target_index:
                return monster
        return None

    def _estimate_attack_damage(self, card):
        base_damage = KNOWN_ATTACK_DAMAGE.get(card.card_id, KNOWN_ATTACK_DAMAGE.get(card.name, 10))
        if card.card_id in ["Strike_R", "Strike"] or card.name == "Strike":
            return base_damage + 3 * card.upgrades
        if card.card_id == "Bash" or card.name == "Bash":
            return base_damage + 2 * card.upgrades
        return base_damage + 3 * max(0, card.upgrades)

    def _is_finishing_attack_candidate(self, game_state, action):
        if not isinstance(action, PlayCardAction):
            return False

        card = action.card
        if card is None or getattr(card, "type", None) != CardType.ATTACK:
            return False

        monsters = living_monsters(game_state)
        if not monsters:
            return False

        estimated_damage = self._estimate_attack_damage(card)
        if card.has_target:
            target = self._find_target_by_index(game_state, getattr(action, "target_index", None))
            if target is None:
                return False
            if len(monsters) == 1:
                return (target.current_hp + target.block) <= estimated_damage
            return False

        return total_monster_hp(game_state) <= estimated_damage
