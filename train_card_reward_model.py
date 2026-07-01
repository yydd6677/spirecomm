#!/usr/bin/env python3
import argparse
import os

import torch

from spirecomm.ai.card_reward_model import (
    evaluate_card_reward_model,
    load_card_reward_checkpoint,
    load_expert_card_reward_examples,
    save_card_reward_checkpoint,
    split_examples,
    train_card_reward_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Ironclad card reward model from expert labels.")
    parser.add_argument("--samples", default="/home/yydd/sts/_tmp/card_reward_expert_samples.jsonl")
    parser.add_argument("--output", default="/home/yydd/spirecomm/models/card_reward.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    examples = load_expert_card_reward_examples(args.samples)
    if not examples:
        raise SystemExit("No usable expert card reward samples found in {}".format(args.samples))

    model, summary = train_card_reward_model(
        examples,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_card_reward_checkpoint(model, args.output, training_summary=summary)

    train_examples, valid_examples = split_examples(examples, valid_fraction=args.valid_fraction, seed=args.seed)
    loaded_model, _ = load_card_reward_checkpoint(args.output, device=args.device)
    train_metrics = evaluate_card_reward_model(loaded_model, train_examples, args.device)
    valid_metrics = evaluate_card_reward_model(loaded_model, valid_examples, args.device)

    print("Loaded {} expert samples from {}".format(len(examples), args.samples))
    print("Train examples: {}  Valid examples: {}".format(len(train_examples), len(valid_examples)))
    print("Train loss: {:.4f}  accuracy: {:.4f}".format(train_metrics["loss"], train_metrics["accuracy"]))
    print("Valid loss: {:.4f}  accuracy: {:.4f}".format(valid_metrics["loss"], valid_metrics["accuracy"]))
    print("Saved card reward model to {}".format(os.path.abspath(args.output)))
    print("Suggested runtime env:")
    print("  SPIRECOMM_CARD_REWARD_MODEL_PATH={}".format(os.path.abspath(args.output)))


if __name__ == "__main__":
    main()
