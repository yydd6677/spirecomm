from __future__ import annotations

import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from spirecomm.ai.agent import HybridAgent
from spirecomm.ai.learned_policy import CheckpointCombatPolicy
from spirecomm.ai.recording import TrajectoryRecorder
from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass


def _normalized_phase_name(game_state) -> str:
    screen_type = getattr(game_state, "screen_type", None)
    screen_name = getattr(screen_type, "name", "UNKNOWN")
    if screen_name == "EVENT":
        event_id = getattr(game_state.screen, "event_id", "") or getattr(game_state.screen, "event_name", "")
        if "neow" in str(event_id).lower():
            return "NEOW"
        return "EVENT"
    if screen_name in {"CARD_REWARD", "COMBAT_REWARD"}:
        return "CARD_REWARD"
    if screen_name == "SHOP_ROOM":
        return "SHOP"
    if screen_name == "SHOP_SCREEN":
        return "SHOP"
    if screen_name == "REST":
        return "CAMPFIRE"
    if screen_name == "BOSS_REWARD":
        return "BOSS_RELIC"
    if screen_name == "MAP":
        return "MAP"
    if screen_name in {"GRID", "HAND_SELECT"}:
        return "CARD_SELECT"
    if screen_name == "CHEST":
        return "TREASURE"
    if getattr(game_state, "in_combat", False):
        return "COMBAT"
    return screen_name


class TrackingHybridAgent(HybridAgent):
    def __init__(self, *, chosen_class, combat_policy, recorder=None):
        super().__init__(chosen_class=chosen_class, combat_policy=combat_policy, recorder=recorder)
        self.phase_counts: Counter[str] = Counter()
        self.screen_counts: Counter[str] = Counter()
        self.source_counts: Counter[str] = Counter()

    def get_next_action_in_game(self, game_state):
        self.fallback_agent.game = game_state
        phase_name = _normalized_phase_name(game_state)
        self.phase_counts[phase_name] += 1
        self.screen_counts[getattr(game_state.screen_type, "name", "UNKNOWN")] += 1
        if self.recorder is not None:
            self.recorder.on_state(game_state)
        source = "fallback"

        if game_state.potion_available and not game_state.choice_available:
            potion_action = self.fallback_agent.choose_potion_action()
            if potion_action is not None:
                action = potion_action
                source = "PotionUseSelector"
                self.source_counts[source] += 1
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
            from spirecomm.communication.action import StateAction

            action = StateAction()
            source = "fallback_state_refresh"

        self.source_counts[source] += 1
        if self.recorder is not None:
            self.recorder.record_step(game_state, action, source)
        return action


@contextmanager
def _temporary_env(overrides: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_seeded_real_game(
    *,
    seed: str,
    player_class: str = "IRONCLAD",
    ascension: int = 0,
    combat_model: str | Path | None = None,
    device: str = "cpu",
    observation_version: str | None = None,
    trajectory_dir: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    combat_model_path = str(combat_model or (repo_root / "models" / "combat.pt"))
    recorder = TrajectoryRecorder(str(trajectory_dir), record_mode="combat") if trajectory_dir else None

    env_overrides = {
        "SPIRECOMM_MODEL_DEVICE": device,
        "SPIRECOMM_MODEL_PATH": combat_model_path,
        "SPIRECOMM_COMBAT_OBSERVATION_VERSION": observation_version,
    }

    with _temporary_env(env_overrides):
        combat_policy = CheckpointCombatPolicy(
            checkpoint_path=combat_model_path,
            device=device,
            observation_version=observation_version,
        )
        chosen_class = PlayerClass[player_class]
        agent = TrackingHybridAgent(
            chosen_class=chosen_class,
            combat_policy=combat_policy,
            recorder=recorder,
        )
        coordinator = Coordinator()
        agent.set_coordinator(coordinator)
        coordinator.signal_ready()
        coordinator.register_command_error_callback(agent.handle_error)
        coordinator.register_state_change_callback(agent.get_next_action_in_game)
        coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)

        agent.on_game_start(chosen_class, ascension_level=ascension, seed=seed)
        victory = coordinator.play_one_game(chosen_class, ascension_level=ascension, seed=seed)
        agent.on_game_end(victory)

    final_state = coordinator.last_game_state
    if final_state is None:
        raise RuntimeError("Real-game run finished without a final game state.")

    return {
        "seed": seed,
        "player_class": chosen_class.name,
        "ascension": ascension,
        "victory": bool(victory),
        "floor": int(getattr(final_state, "floor", 0) or 0),
        "act": int(getattr(final_state, "act", 0) or 0),
        "current_hp": int(getattr(final_state, "current_hp", 0) or 0),
        "max_hp": int(getattr(final_state, "max_hp", 0) or 0),
        "gold": int(getattr(final_state, "gold", 0) or 0),
        "screen_type": getattr(getattr(final_state, "screen_type", None), "name", "UNKNOWN"),
        "room_phase": getattr(getattr(final_state, "room_phase", None), "name", "UNKNOWN"),
        "phase_counts": dict(agent.phase_counts),
        "screen_counts": dict(agent.screen_counts),
        "source_counts": dict(agent.source_counts),
        "trajectory_dir": str(trajectory_dir) if trajectory_dir else None,
        "last_run_id": getattr(recorder, "last_run_id", None) if recorder is not None else None,
    }
