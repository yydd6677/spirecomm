import argparse
import os

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
    split_episodes,
)


def format_metrics(metrics):
    if not metrics:
        return "no-data"
    pieces = []
    for key in sorted(metrics.keys()):
        pieces.append("{}={:.4f}".format(key, metrics[key]))
    return ", ".join(pieces)


def main():
    parser = argparse.ArgumentParser(description="Train a first-pass Slay the Spire combat model from spirecomm trajectories.")
    parser.add_argument("--trajectory-dir", required=True, help="Directory containing *_combat_*.jsonl files.")
    parser.add_argument("--output", required=True, help="Where to save the model checkpoint.")
    parser.add_argument("--mode", choices=["bc", "reinforce", "preference"], default="bc")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--source-filter", default=None, help="Comma-separated recorder source names to keep.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to warm-start from.")
    parser.add_argument("--run-id", default=None, help="Optional run id prefix to restrict training to one recorded game.")
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--bc-weight", type=float, default=0.01)
    parser.add_argument("--observation-version", default=os.environ.get("SPIRECOMM_COMBAT_OBSERVATION_VERSION", LEGACY_COMBAT_OBSERVATION_VERSION))
    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = 12 if args.mode == "bc" else 2
    if args.mode == "reinforce" and not args.checkpoint:
        raise SystemExit("Reinforce mode requires --checkpoint so training starts from an existing policy.")

    source_filter = None
    if args.source_filter:
        source_filter = [piece.strip() for piece in args.source_filter.split(",") if piece.strip()]

    if args.mode == "preference":
        train_examples, stats = load_preference_examples(
            args.trajectory_dir,
            source_filter=source_filter,
            limit_files=args.limit_files,
            run_id=args.run_id,
            observation_version=args.observation_version,
        )
        if not train_examples:
            raise SystemExit("No usable preference trajectories were found.")
        validation_examples = []
    else:
        episodes, stats = load_trajectory_episodes(
            args.trajectory_dir,
            source_filter=source_filter,
            limit_files=args.limit_files,
            run_id=args.run_id,
            observation_version=args.observation_version,
        )
        if not episodes:
            raise SystemExit("No usable combat trajectories were found.")

        train_episodes, validation_episodes = split_episodes(
            episodes,
            validation_fraction=args.validation_fraction if args.mode == "bc" else 0.0,
            seed=args.seed,
        )
        train_examples = flatten_episodes(train_episodes)
        validation_examples = flatten_episodes(validation_episodes)

    model = CombatPolicyNetwork().to(args.device)
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, args.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print("Loaded {} episodes / {} transitions".format(stats["episodes_loaded"], stats["examples_loaded"]))
    print("Skipped {} unsupported actions".format(stats["skipped_actions"]))
    print("Recorder sources:", stats["sources"])
    print("Training mode:", args.mode)
    print("Training transitions:", len(train_examples))
    print("Validation transitions:", len(validation_examples))

    best_validation = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_examples,
            optimizer=optimizer,
            device=args.device,
            batch_size=args.batch_size,
            mode=args.mode,
            seed=args.seed + epoch,
            value_weight=args.value_weight,
            entropy_weight=args.entropy_weight,
            behavior_cloning_weight=args.bc_weight,
        )
        validation_metrics = {}
        if validation_examples:
            validation_metrics = run_epoch(
                model,
                validation_examples,
                optimizer=None,
                device=args.device,
                batch_size=args.batch_size,
                mode="bc",
                seed=args.seed,
            )

        print("epoch {:02d} train {}".format(epoch, format_metrics(train_metrics)))
        if validation_metrics:
            print("epoch {:02d} valid {}".format(epoch, format_metrics(validation_metrics)))
            current_validation = validation_metrics.get("loss", 0.0)
            if best_validation is None or current_validation < best_validation:
                best_validation = current_validation

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_checkpoint(
        args.output,
        model,
        training_args=vars(args),
        dataset_stats=stats,
    )

    print("")
    print("Saved checkpoint to {}".format(args.output))
    print("To run the model in spirecomm, set:")
    print("  SPIRECOMM_POLICY_CLASS=spirecomm.ai.learned_policy:CheckpointCombatPolicy")
    print("  SPIRECOMM_MODEL_PATH={}".format(os.path.abspath(args.output)))


if __name__ == "__main__":
    main()
