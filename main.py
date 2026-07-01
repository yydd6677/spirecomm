import itertools
import datetime
import os
import sys

from spirecomm.communication.coordinator import Coordinator
from spirecomm.ai.agent import HybridAgent, SimpleAgent
from spirecomm.ai.auto_train import build_auto_trainer
from spirecomm.ai.policy import load_policy
from spirecomm.ai.recording import TrajectoryRecorder
from spirecomm.spire.character import PlayerClass


def build_agent():
    chosen_class = PlayerClass[os.environ.get("SPIRECOMM_STARTER_CLASS", PlayerClass.THE_SILENT.name)]
    trajectory_directory = os.environ.get("SPIRECOMM_TRAJECTORY_DIR")
    record_mode = os.environ.get("SPIRECOMM_RECORD_MODE", "combat")
    policy_spec = os.environ.get("SPIRECOMM_POLICY_CLASS")

    recorder = None
    if trajectory_directory:
        recorder = TrajectoryRecorder(trajectory_directory, record_mode=record_mode)

    if policy_spec is None and recorder is None:
        return SimpleAgent(chosen_class=chosen_class)

    combat_policy = None
    if policy_spec is not None:
        combat_policy = load_policy(policy_spec)

    return HybridAgent(chosen_class=chosen_class, combat_policy=combat_policy, recorder=recorder)


def build_class_cycle():
    only_class = os.environ.get("SPIRECOMM_ONLY_CLASS")
    if only_class:
        chosen_class = PlayerClass[only_class]
        return itertools.repeat(chosen_class)
    return itertools.cycle(PlayerClass)


if __name__ == "__main__":
    agent = build_agent()
    auto_trainer = build_auto_trainer()
    coordinator = Coordinator()
    if hasattr(agent, "set_coordinator"):
        agent.set_coordinator(coordinator)
    coordinator.signal_ready()
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)

    # Play games forever, optionally pinning to a single class.
    for chosen_class in build_class_cycle():
        agent.change_class(chosen_class)
        if hasattr(agent, "on_game_start"):
            agent.on_game_start(chosen_class)
        result = coordinator.play_one_game(chosen_class)
        if hasattr(agent, "on_game_end"):
            agent.on_game_end(result)
        if auto_trainer is not None and hasattr(agent, "recorder"):
            training_result = auto_trainer.train_latest_run(agent.recorder)
            if training_result is not None and hasattr(agent, "reload_combat_policy"):
                agent.reload_combat_policy()
