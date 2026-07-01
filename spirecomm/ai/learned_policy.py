import os

from spirecomm.ai.observation import LEGACY_COMBAT_OBSERVATION_VERSION


class CheckpointCombatPolicy:

    source_name = "CheckpointCombatPolicy"

    def __init__(self, checkpoint_path=None, device=None, sample_actions=False, observation_version=LEGACY_COMBAT_OBSERVATION_VERSION):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("CheckpointCombatPolicy requires torch. Use the spirecomm-rl conda env.") from exc

        from spirecomm.ai.recording import serialize_game_state
        from spirecomm.ai.rl import (
            CombatPolicyNetwork,
            build_action_mask,
            build_state_tensors,
            build_target_mask,
            decode_action_from_prediction,
            load_checkpoint,
            masked_logits,
        )

        self.torch = torch
        self.serialize_game_state = serialize_game_state
        self.build_action_mask = build_action_mask
        self.build_state_tensors = build_state_tensors
        self.build_target_mask = build_target_mask
        self.decode_action_from_prediction = decode_action_from_prediction
        self.load_checkpoint = load_checkpoint
        self.masked_logits = masked_logits
        self.checkpoint_path = checkpoint_path or os.environ.get("SPIRECOMM_MODEL_PATH")
        self.device = device or os.environ.get("SPIRECOMM_MODEL_DEVICE", "cpu")
        self.sample_actions = sample_actions or os.environ.get("SPIRECOMM_SAMPLE_ACTIONS", "0") == "1"
        self.observation_version = observation_version or os.environ.get(
            "SPIRECOMM_COMBAT_OBSERVATION_VERSION",
            LEGACY_COMBAT_OBSERVATION_VERSION,
        )

        if not self.checkpoint_path:
            raise ValueError("CheckpointCombatPolicy needs SPIRECOMM_MODEL_PATH or checkpoint_path.")

        self.model = CombatPolicyNetwork()
        self.model.to(self.device)
        self.reload()

    def _batch_from_state(self, serialized_state):
        state = self.build_state_tensors(serialized_state, observation_version=self.observation_version)
        return {
            "global_features": self.torch.tensor(
                [state["global_features"] + [state["potion_count"]]],
                dtype=self.torch.float32,
                device=self.device,
            ),
            "deck_features": self.torch.tensor([state["deck_features"]], dtype=self.torch.float32, device=self.device),
            "hand_card_ids": self.torch.tensor([state["hand_card_ids"]], dtype=self.torch.long, device=self.device),
            "hand_features": self.torch.tensor([state["hand_features"]], dtype=self.torch.float32, device=self.device),
            "monster_ids": self.torch.tensor([state["monster_ids"]], dtype=self.torch.long, device=self.device),
            "monster_intents": self.torch.tensor([state["monster_intents"]], dtype=self.torch.long, device=self.device),
            "monster_features": self.torch.tensor(
                [state["monster_features"]],
                dtype=self.torch.float32,
                device=self.device,
            ),
            "relic_ids": self.torch.tensor([state["relic_ids"]], dtype=self.torch.long, device=self.device),
        }

    def choose_action(self, game_state, fallback_agent, coordinator=None):
        serialized_state = self.serialize_game_state(game_state)
        scoring = self.score_state(serialized_state)
        action_logits = scoring["action_logits"]
        target_logits = scoring["target_logits"]
        with self.torch.no_grad():
            if self.sample_actions:
                action_probs = self.torch.softmax(action_logits, dim=-1)
                action_index = int(self.torch.multinomial(action_probs[0], 1).item())
            else:
                action_index = int(action_logits.argmax(dim=-1).item())
            target_index = int(target_logits.argmax(dim=-1).item())

        return self.decode_action_from_prediction(game_state, fallback_agent, action_index, target_index)

    def score_state(self, serialized_state):
        action_mask = self.torch.tensor(
            [self.build_action_mask(serialized_state)],
            dtype=self.torch.bool,
            device=self.device,
        )
        target_mask = self.torch.tensor(
            [self.build_target_mask(serialized_state)],
            dtype=self.torch.bool,
            device=self.device,
        )
        batch = self._batch_from_state(serialized_state)
        with self.torch.no_grad():
            outputs = self.model(batch)
            action_logits = self.masked_logits(outputs["action_logits"], action_mask)
            target_logits = self.masked_logits(outputs["target_logits"], target_mask)
        return {
            "action_mask": action_mask,
            "target_mask": target_mask,
            "action_logits": action_logits,
            "target_logits": target_logits,
            "value": outputs["value"],
        }

    def reload(self):
        checkpoint = self.load_checkpoint(self.checkpoint_path, self.device)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint {} is incompatible with the current model feature schema. "
                "You likely need to retrain the model after changing state features.".format(self.checkpoint_path)
            ) from exc
        self.model.to(self.device)
        self.model.eval()
