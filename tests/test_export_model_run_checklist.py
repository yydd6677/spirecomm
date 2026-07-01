from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.native.export_model_run_checklist import _capture_run
from spirecomm.ai.strict_trace import STRICT_TRACE_SCHEMA


class _DummySelector:
    available = True
    checkpoint_path = Path("/tmp/dummy.pt")


class _VictoryEnv:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.phase = "MAP"
        self.floor = 10
        self.gold = 99
        self.deck = []
        self.player = type("Player", (), {"current_hp": 42})()

    def state(self):
        return {"phase": self.phase, "floor": self.floor, "choice_list": []}

    def step(self, action):
        del action
        self.phase = "VICTORY"


class ExportModelRunChecklistTest(unittest.TestCase):
    def test_capture_run_stops_when_env_reaches_victory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            with patch("scripts.native.export_model_run_checklist.build_runtime_selectors", return_value={}):
                with patch("scripts.native.export_model_run_checklist._native_env_cls", return_value=_VictoryEnv):
                    with patch("scripts.native.export_model_run_checklist.choose_modeled_action", return_value=({"kind": "map"}, [1.0], "map")):
                        trace = _capture_run(
                            seed=1,
                            ascension=0,
                            backend="v3",
                            max_steps=50,
                            repo_root=repo_root,
                            device="cpu",
                            combat_device=None,
                            observation_version=None,
                            trace_policy="legacy-fallback",
                        )

        self.assertEqual(trace["result"]["final_phase"], "VICTORY")
        self.assertEqual(trace["result"]["steps"], 1)
        self.assertEqual(len(trace["steps"]), 1)

    def test_capture_run_defaults_to_strict_trace_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            with patch("scripts.native.export_model_run_checklist.build_runtime_selectors", return_value={}):
                with patch("scripts.native.export_model_run_checklist._native_env_cls", return_value=_VictoryEnv):
                    with patch("scripts.native.export_model_run_checklist.choose_modeled_action", return_value=({"kind": "map", "choice_index": 0, "symbol": "M"}, [1.0], "map")):
                        trace = _capture_run(
                            seed=1,
                            ascension=0,
                            backend="v3",
                            max_steps=50,
                            repo_root=repo_root,
                            device="cpu",
                            combat_device=None,
                            observation_version=None,
                            trace_policy="legacy-fallback",
                        )

        self.assertEqual(trace["trace_schema"], STRICT_TRACE_SCHEMA)
        self.assertIn("strict_action", trace["steps"][0])
        self.assertIn("strict_pre_state", trace["steps"][0])
        self.assertIn("strict_post_state", trace["steps"][0])
        self.assertEqual(trace["steps"][0]["strict_action"]["kind"], "choose_by_index")

    def test_capture_run_model_required_records_policy_and_model_status(self):
        selectors = {name: _DummySelector() for name in [
            "combat",
            "card_reward",
            "boss_relic",
            "map",
            "campfire",
            "shop",
            "event",
            "potion",
            "upgrade_target",
            "purge_target",
        ]}
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            with patch("scripts.native.export_model_run_checklist.build_runtime_selectors", return_value=selectors):
                with patch("scripts.native.export_model_run_checklist.validate_model_required_selectors", return_value=None):
                    with patch("scripts.native.export_model_run_checklist._native_env_cls", return_value=_VictoryEnv):
                        with patch("scripts.native.export_model_run_checklist.choose_model_required_action", return_value=({"kind": "map", "choice_index": 0, "symbol": "M"}, [1.0], "map")):
                            trace = _capture_run(
                                seed=1,
                                ascension=0,
                                backend="v3",
                                max_steps=50,
                                repo_root=repo_root,
                                device="cpu",
                                combat_device=None,
                                observation_version=None,
                                trace_policy="model-required",
                            )

        self.assertEqual(trace["trace_policy"], "model_required")
        self.assertTrue(trace["model_required"])
        self.assertTrue(trace["model_status"]["selectors"]["combat"]["available"])


if __name__ == "__main__":
    unittest.main()
