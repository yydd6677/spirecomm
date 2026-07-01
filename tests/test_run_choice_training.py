import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from spirecomm.ai.run_choice_model import EventChoiceSelector, RunChoicePolicyNetwork, option_token, save_run_choice_checkpoint
from train_run_choice_models import build_pairs_from_run, campfire_candidates


class RunChoiceTrainingTest(unittest.TestCase):
    def test_campfire_candidates_include_recall_as_negative_before_ruby_key(self):
        pairs = list(
            build_pairs_from_run(
                {
                    "floor_reached": 16,
                    "campfire_choices": [
                        {"floor": 8, "key": "SMITH"},
                        {"floor": 15, "key": "RECALL"},
                    ],
                },
                "campfire",
            )
        )

        self.assertIn(
            ("SMITH", "RECALL"),
            {(pair["pos_choice"], pair["neg_choice"]) for pair in pairs},
        )
        self.assertIn(
            ("RECALL", "SMITH"),
            {(pair["pos_choice"], pair["neg_choice"]) for pair in pairs},
        )

    def test_campfire_peace_pipe_uses_training_purge_label(self):
        self.assertIn("PURGE", campfire_candidates("SMITH", ["Peace Pipe"], can_recall=True))
        self.assertNotIn("TOKE", campfire_candidates("SMITH", ["Peace Pipe"], can_recall=True))
        self.assertEqual(
            option_token("campfire", {"kind": "campfire", "name": "toke"}),
            option_token("campfire", "PURGE"),
        )

    def test_event_pairs_canonicalize_labels_for_training_tokens(self):
        pairs = list(
            build_pairs_from_run(
                {
                    "floor_reached": 1,
                    "event_choices": [
                        {"floor": 1, "event_id": "Golden Shrine", "player_choice": "Pray (100 Gold)"}
                    ],
                },
                "event",
                event_metadata={"vocab": {"goldenshrine": ["Pray", "Ignored"]}},
            )
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(
            option_token("event", pairs[0]["pos_choice"]),
            option_token("event", {"event_id": "Golden Shrine", "label": "Pray"}),
        )

    def test_event_prior_delta_selector_adds_checkpoint_prior_to_model_delta(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "event_prior_delta.pt"
            model = RunChoicePolicyNetwork()
            for parameter in model.parameters():
                torch.nn.init.constant_(parameter, 0.0)
            entered = {"event_id": "Shining Light", "label": "Entered Light"}
            ignored = {"event_id": "Shining Light", "label": "Ignored"}
            save_run_choice_checkpoint(
                model,
                str(path),
                training_summary={
                    "task": "event",
                    "score_mode": "event_prior_delta",
                    "event_prior_weight": 1.0,
                    "event_option_log_prior_by_token": {
                        option_token("event", entered): -0.1,
                        option_token("event", ignored): -2.0,
                    },
                },
            )

            selector = EventChoiceSelector(checkpoint_path=path, device="cpu")
            result = selector.choose({}, [ignored, entered])

        self.assertEqual(result["choice_index"], 1)
        self.assertAlmostEqual(result["scores"][0], -2.0)
        self.assertAlmostEqual(result["scores"][1], -0.1)


if __name__ == "__main__":
    unittest.main()
