from __future__ import annotations

from pathlib import Path
from typing import Any

from spirecomm.ai.checkpoint_compat import torch_load_portable_path
from spirecomm.ai.torch_compat import nn, require_torch, torch
from spirecomm.ai.v3_combat_features import FEATURE_SCHEMA_VERSION, schema


CHECKPOINT_VERSION = "v3_combat_candidate_scorer_v1"


class V3CombatCandidateScorer(nn.Module):
    def __init__(self, input_dim: int | None = None, hidden_dims: tuple[int, ...] = (512, 256, 128), dropout: float = 0.05) -> None:
        super().__init__()
        feature_schema = schema()
        self.input_dim = int(input_dim or feature_schema.candidate_dim)
        layers: list[Any] = []
        current_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Any) -> Any:
        return self.network(features).squeeze(-1)


def save_v3_combat_checkpoint(
    path: str | Path,
    model: V3CombatCandidateScorer,
    *,
    training_args: dict[str, Any] | None = None,
    dataset_metadata: dict[str, Any] | None = None,
) -> None:
    require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema": schema().__dict__,
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": model.input_dim,
            },
            "training_args": dict(training_args or {}),
            "dataset_metadata": dict(dataset_metadata or {}),
        },
        target,
    )


def load_v3_combat_checkpoint(path: str | Path, device: str = "cpu") -> tuple[V3CombatCandidateScorer, dict[str, Any]]:
    require_torch()
    checkpoint = torch_load_portable_path(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported v3 combat checkpoint version: {checkpoint.get('checkpoint_version')}")
    if checkpoint.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "v3 combat checkpoint feature schema mismatch: "
            f"{checkpoint.get('feature_schema_version')} != {FEATURE_SCHEMA_VERSION}"
        )
    config = dict(checkpoint.get("model_config") or {})
    model = V3CombatCandidateScorer(input_dim=int(config.get("input_dim") or schema().candidate_dim))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
