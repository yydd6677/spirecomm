import os
import sys

import torch

from spirecomm.ai.observation import LEGACY_COMBAT_OBSERVATION_VERSION
from spirecomm.ai.rl import (
    CombatPolicyNetwork,
    flatten_episodes,
    load_checkpoint,
    load_preference_examples,
    load_trajectory_episodes,
    run_epoch,
    save_checkpoint,
)


class AutoTrainManager:

    def __init__(
        self,
        trajectory_directory,
        checkpoint_path,
        source_filter="CheckpointCombatPolicy",
        device="cpu",
        epochs=2,
        batch_size=128,
        learning_rate=5e-3,
        value_weight=0.5,
        entropy_weight=0.05,
        behavior_cloning_weight=0.05,
        min_examples=32,
        mode="preference",
        observation_version=LEGACY_COMBAT_OBSERVATION_VERSION,
    ):
        self.trajectory_directory = trajectory_directory
        self.checkpoint_path = checkpoint_path
        self.source_filter = source_filter
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.behavior_cloning_weight = behavior_cloning_weight
        self.min_examples = min_examples
        self.last_trained_run_id = None
        self.mode = mode
        self.observation_version = observation_version

    def train_latest_run(self, recorder):
        if recorder is None or recorder.last_run_id is None:
            return None
        if recorder.last_run_id == self.last_trained_run_id:
            return None
        if not os.path.exists(self.checkpoint_path):
            print(
                "AutoTrain: checkpoint does not exist yet, skipping {}".format(self.checkpoint_path),
                file=sys.stderr,
                flush=True,
            )
            return None

        if self.mode == "preference":
            examples, stats = load_preference_examples(
                self.trajectory_directory,
                source_filter=[self.source_filter],
                run_id=recorder.last_run_id,
                observation_version=self.observation_version,
            )
        else:
            episodes, stats = load_trajectory_episodes(
                self.trajectory_directory,
                source_filter=[self.source_filter],
                run_id=recorder.last_run_id,
                observation_version=self.observation_version,
            )
            examples = flatten_episodes(episodes)
        if stats["examples_loaded"] < self.min_examples:
            print(
                "AutoTrain: skipping run {} because only {} usable {} were found.".format(
                    recorder.last_run_id,
                    stats["examples_loaded"],
                    "preferences" if self.mode == "preference" else "transitions",
                ),
                file=sys.stderr,
                flush=True,
            )
            self.last_trained_run_id = recorder.last_run_id
            return None

        model = CombatPolicyNetwork().to(self.device)
        checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)

        final_metrics = {}
        for epoch in range(1, self.epochs + 1):
            final_metrics = run_epoch(
                model,
                examples,
                optimizer=optimizer,
                device=self.device,
                batch_size=self.batch_size,
                mode=self.mode,
                seed=epoch,
                value_weight=self.value_weight,
                entropy_weight=self.entropy_weight,
                behavior_cloning_weight=self.behavior_cloning_weight,
            )
            print(
                "AutoTrain run {} epoch {:02d}: {}".format(
                    recorder.last_run_id,
                    epoch,
                    ", ".join("{}={:.4f}".format(key, final_metrics[key]) for key in sorted(final_metrics.keys())),
                ),
                file=sys.stderr,
                flush=True,
            )

        save_checkpoint(
            self.checkpoint_path,
            model,
            training_args={
                "mode": self.mode,
                "checkpoint": self.checkpoint_path,
                "source_filter": self.source_filter,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "run_id": recorder.last_run_id,
                "auto_train": True,
                "entropy_weight": self.entropy_weight,
                "value_weight": self.value_weight,
                "behavior_cloning_weight": self.behavior_cloning_weight,
                "combat_observation_version": self.observation_version,
            },
            dataset_stats=stats,
        )
        self.last_trained_run_id = recorder.last_run_id
        return {
            "run_id": recorder.last_run_id,
            "metrics": final_metrics,
            "stats": stats,
            "checkpoint_path": self.checkpoint_path,
        }


def build_auto_trainer():
    if os.environ.get("SPIRECOMM_AUTO_TRAIN", "0") != "1":
        return None

    trajectory_directory = os.environ.get("SPIRECOMM_TRAJECTORY_DIR")
    checkpoint_path = os.environ.get("SPIRECOMM_MODEL_PATH")
    if not trajectory_directory or not checkpoint_path:
        print(
            "AutoTrain: SPIRECOMM_TRAJECTORY_DIR and SPIRECOMM_MODEL_PATH are required.",
            file=sys.stderr,
            flush=True,
        )
        return None

    return AutoTrainManager(
        trajectory_directory=trajectory_directory,
        checkpoint_path=checkpoint_path,
        source_filter=os.environ.get("SPIRECOMM_AUTO_TRAIN_SOURCE_FILTER", "CheckpointCombatPolicy"),
        device=os.environ.get("SPIRECOMM_MODEL_DEVICE", "cpu"),
        epochs=int(os.environ.get("SPIRECOMM_AUTO_TRAIN_EPOCHS", "2")),
        batch_size=int(os.environ.get("SPIRECOMM_AUTO_TRAIN_BATCH_SIZE", "128")),
        learning_rate=float(os.environ.get("SPIRECOMM_AUTO_TRAIN_LR", "1e-4")),
        value_weight=float(os.environ.get("SPIRECOMM_AUTO_TRAIN_VALUE_WEIGHT", "0.5")),
        entropy_weight=float(os.environ.get("SPIRECOMM_AUTO_TRAIN_ENTROPY_WEIGHT", "0.05")),
        behavior_cloning_weight=float(os.environ.get("SPIRECOMM_AUTO_TRAIN_BC_WEIGHT", "0.01")),
        min_examples=int(os.environ.get("SPIRECOMM_AUTO_TRAIN_MIN_EXAMPLES", "32")),
        mode=os.environ.get("SPIRECOMM_AUTO_TRAIN_MODE", "preference"),
        observation_version=os.environ.get("SPIRECOMM_COMBAT_OBSERVATION_VERSION", LEGACY_COMBAT_OBSERVATION_VERSION),
    )
