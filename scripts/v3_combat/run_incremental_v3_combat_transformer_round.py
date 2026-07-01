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
import subprocess
from pathlib import Path
from typing import Any

from spirecomm.ai.torch_compat import require_torch


def _checkpoint_epoch(path: Path) -> int:
    if not path.exists():
        return 0
    torch = require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = dict(payload.get("training_state") or {})
    return int(state.get("current_epoch") or 0)


def _run(command: list[str], *, cwd: Path) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache newly completed shards and run one incremental transformer train round.")
    parser.add_argument("--source-dir", action="append", type=Path, required=True)
    parser.add_argument("--active-source-dir", action="append", type=Path, default=[])
    parser.add_argument("--stable-lag-shards", type=int, default=4)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--validation-source-shards-file", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--round-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-shards", type=int, default=0)
    parser.add_argument("--cache-save-every-shards", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--min-mem-available-gb", type=float, default=1.0)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    python = "/home/yydd/miniforge3/envs/spirecomm-rl/bin/python"

    cache_cmd = [
        python,
        "scripts/v3_combat/cache_incremental_v3_combat_transformer_dataset.py",
        "--output",
        str(args.tensor_output),
        "--dtype",
        "float16",
        "--stable-lag-shards",
        str(max(0, int(args.stable_lag_shards))),
        "--save-every-shards",
        str(max(1, int(args.cache_save_every_shards))),
        "--validation-source-shards-file",
        str(args.validation_source_shards_file),
        "--create-validation-if-missing",
        "--validation-fraction",
        str(args.validation_fraction),
        "--seed",
        str(args.seed),
    ]
    for source_dir in args.source_dir:
        cache_cmd.extend(["--source-dir", str(source_dir)])
    for active_source_dir in args.active_source_dir:
        cache_cmd.extend(["--active-source-dir", str(active_source_dir)])
    if args.max_new_shards > 0:
        cache_cmd.extend(["--max-new-shards", str(args.max_new_shards)])
    _run(cache_cmd, cwd=repo_root)

    latest = args.model_output.with_suffix(args.model_output.suffix + ".epochs") / "latest.pt"
    current_epoch = _checkpoint_epoch(latest)
    total_epochs = current_epoch + max(1, int(args.round_epochs))
    train_cmd = [
        python,
        "scripts/v3_combat/train_v3_combat_transformer_scorer.py",
        "--tensor-dataset",
        str(args.tensor_output),
        "--output",
        str(args.model_output),
        "--epochs",
        str(total_epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--min-learning-rate",
        str(args.min_learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--device",
        str(args.device),
        "--validation-source-shards-file",
        str(args.validation_source_shards_file),
        "--min-mem-available-gb",
        str(args.min_mem_available_gb),
        "--memory-log",
        str(args.model_output.with_suffix(args.model_output.suffix + ".memory_log.jsonl")),
    ]
    if latest.exists():
        train_cmd.extend(["--resume-checkpoint", str(latest)])
    _run(train_cmd, cwd=repo_root)

    summary: dict[str, Any] = {
        "tensor_output": str(args.tensor_output),
        "model_output": str(args.model_output),
        "previous_epoch": current_epoch,
        "target_epoch": total_epochs,
        "validation_source_shards_file": str(args.validation_source_shards_file),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
