#!/usr/bin/env python3
from __future__ import annotations
# Allow this CLI to run directly from its workflow subdirectory.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any

from spirecomm.ai.run_value import load_run_value_checkpoint
from spirecomm.ai.torch_compat import require_torch, torch
from scripts.run_value.train_run_value_model import _eval_model, load_manifest


def _metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    value = checkpoint.get("metadata")
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a run-value checkpoint on a tensor cache.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    require_torch()
    model, checkpoint = load_run_value_checkpoint(args.checkpoint, device=str(args.device))
    metadata = _metadata(checkpoint)
    manifest = load_manifest(args.cache_dir)
    chunks = list(manifest.get("chunks") or [])
    if not chunks:
        raise SystemExit(f"cache has no chunks: {args.cache_dir}")
    metrics = _eval_model(
        model,
        chunks,
        batch_size=int(args.batch_size),
        device=str(args.device),
        survival_bins=int(metadata.get("survival_bins") or getattr(model, "survival_bins", 0) or 0),
        use_survival=bool(metadata.get("use_survival_for_mae") or False),
        survival_weight=float(metadata.get("survival_weight") or 0.0),
        survival_value_weight=float(metadata.get("survival_value_weight") or 0.0),
        final_floor_bins=int(metadata.get("final_floor_bins") or getattr(model, "final_floor_bins", 0) or 0),
        final_floor_readout=str(metadata.get("final_floor_readout") or "none"),
        final_floor_weight=float(metadata.get("final_floor_weight") or 0.0),
        final_floor_value_weight=float(metadata.get("final_floor_value_weight") or 0.0),
        residual_floor_baseline=metadata.get("residual_floor_baseline"),
        regression_loss=str(metadata.get("regression_loss") or "smooth_l1"),
        smooth_l1_beta=float(metadata.get("smooth_l1_beta") or 1.0),
        final_loss_weight=float(metadata.get("final_loss_weight") or 0.25),
        act_bce_weight=float(metadata.get("act_bce_weight") or (1.0 / 3.0)),
        death_bce_weight=float(metadata.get("death_bce_weight") or (1.0 / 6.0)),
        value_calibration=metadata.get("value_calibration"),
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "manifest": {
            key: manifest.get(key)
            for key in (
                "count",
                "train_count",
                "validation_count",
                "state_feature_dim",
                "sample_mode",
                "feature_variant",
                "record_weight_mode",
            )
        },
        "metadata": {
            key: metadata.get(key)
            for key in (
                "best_epoch",
                "best_validation_remaining_mae",
                "survival_bins",
                "use_survival_for_mae",
                "final_floor_bins",
                "final_floor_readout",
                "final_loss_weight",
                "regression_loss",
                "row_weight_mode",
                "residual_baseline_key",
            )
        },
        "metrics": metrics,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
