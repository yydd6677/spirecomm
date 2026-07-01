import os
from pathlib import Path

from spirecomm.ai.observation import LEGACY_COMBAT_OBSERVATION_VERSION
from spirecomm.ai.rl import (
    ACTION_END_TURN,
    ACTION_PLAY_OFFSET,
    CombatPolicyNetwork,
    MAX_HAND_SIZE,
    build_action_mask,
    build_state_tensors,
    build_target_mask,
    load_checkpoint,
    masked_logits,
)
from spirecomm.ai.torch_compat import torch


class SerializedCombatSelector:
    """Use the spirecomm combat checkpoint on a serialized battle state.

    Any backend that can provide a spirecomm-style serialized combat state plus
    legal actions can reuse this selector, including lightspeed and native v2.
    """

    def __init__(self, checkpoint_path=None, device=None, observation_version=LEGACY_COMBAT_OBSERVATION_VERSION):
        repo_root = Path(__file__).resolve().parents[2]
        self.checkpoint_path = Path(
            checkpoint_path
            or os.environ.get("SPIRECOMM_MODEL_PATH")
            or repo_root / "models" / "combat.pt"
        )
        self.device = device or os.environ.get("SPIRECOMM_MODEL_DEVICE", "cpu")
        self.observation_version = observation_version or os.environ.get(
            "SPIRECOMM_COMBAT_OBSERVATION_VERSION",
            LEGACY_COMBAT_OBSERVATION_VERSION,
        )
        self.model = None
        if self.checkpoint_path.exists():
            self.model = CombatPolicyNetwork().to(self.device)
            checkpoint = load_checkpoint(str(self.checkpoint_path), self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

    @property
    def available(self):
        return self.model is not None

    def _batch_from_state(self, serialized_state):
        state = build_state_tensors(serialized_state, observation_version=self.observation_version)
        return {
            "global_features": torch.tensor(
                [state["global_features"] + [state["potion_count"]]],
                dtype=torch.float32,
                device=self.device,
            ),
            "deck_features": torch.tensor([state["deck_features"]], dtype=torch.float32, device=self.device),
            "hand_card_ids": torch.tensor([state["hand_card_ids"]], dtype=torch.long, device=self.device),
            "hand_features": torch.tensor([state["hand_features"]], dtype=torch.float32, device=self.device),
            "monster_ids": torch.tensor([state["monster_ids"]], dtype=torch.long, device=self.device),
            "monster_intents": torch.tensor([state["monster_intents"]], dtype=torch.long, device=self.device),
            "monster_features": torch.tensor([state["monster_features"]], dtype=torch.float32, device=self.device),
            "relic_ids": torch.tensor([state["relic_ids"]], dtype=torch.long, device=self.device),
        }

    def _score_state(self, serialized_state):
        action_mask = torch.tensor([build_action_mask(serialized_state)], dtype=torch.bool, device=self.device)
        target_mask = torch.tensor([build_target_mask(serialized_state)], dtype=torch.bool, device=self.device)
        batch = self._batch_from_state(serialized_state)
        with torch.no_grad():
            outputs = self.model(batch)
            action_logits = masked_logits(outputs["action_logits"], action_mask)[0]
            target_logits = masked_logits(outputs["target_logits"], target_mask)[0]
        return action_logits, target_logits

    @staticmethod
    def _model_action_index(action):
        kind = action.get("kind")
        if kind == "end":
            return ACTION_END_TURN
        if kind == "card":
            slot = int(action.get("card_index", action.get("source_index", 0)) or 0)
            if slot < 0 or slot >= MAX_HAND_SIZE:
                return None
            return ACTION_PLAY_OFFSET + slot
        return None

    def choose(self, serialized_state, legal_actions):
        if not self.available:
            return None, []

        model_actions = [
            action
            for action in legal_actions
            if action.get("kind") in {"card", "end"} and self._model_action_index(action) is not None
        ]
        if not model_actions:
            return None, []

        action_logits, target_logits = self._score_state(serialized_state)

        action_indices = sorted(
            {self._model_action_index(action) for action in model_actions},
            key=lambda index: float(action_logits[index].item()) if index is not None else float("-inf"),
            reverse=True,
        )
        for action_index in action_indices:
            if action_index is None or float(action_logits[action_index].item()) <= -1e8:
                continue
            candidates = [
                action
                for action in model_actions
                if self._model_action_index(action) == action_index
            ]
            if action_index == ACTION_END_TURN:
                return candidates[0], [float(value) for value in action_logits.detach().cpu().tolist()]
            if not candidates:
                continue
            target_candidates = [action for action in candidates if action.get("requires_target")]
            if target_candidates:
                def target_score(action):
                    target_index = int(action.get("model_target_index", action.get("target_index", 0)) or 0)
                    if target_index < 0 or target_index >= len(target_logits):
                        return float("-inf")
                    return float(target_logits[target_index].item())

                chosen = max(
                    target_candidates,
                    key=target_score,
                )
                return chosen, [float(value) for value in action_logits.detach().cpu().tolist()]
            return candidates[0], [float(value) for value in action_logits.detach().cpu().tolist()]

        return None, [float(value) for value in action_logits.detach().cpu().tolist()]


class LightspeedCombatSelector(SerializedCombatSelector):
    pass


class V2CombatSelector(SerializedCombatSelector):
    pass
