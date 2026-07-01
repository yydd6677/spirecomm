from __future__ import annotations

import unittest
from pathlib import Path

from spirecomm.ai.integration_verifier import (
    DEFAULT_LIGHTSPEED_BUILD,
    compare_seed_with_model,
    validate_combat_model_loads,
    validate_selector_phase_fixtures,
)
from spirecomm.ai.runtime_decision import build_runtime_selectors


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


LIGHTSPEED_BUILD_AVAILABLE = DEFAULT_LIGHTSPEED_BUILD.exists()
TORCH_AVAILABLE = _has_torch()


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for model integration tests")
class ModelIntegrationTest(unittest.TestCase):
    def test_combat_models_load_with_legacy_observation_schema(self):
        results = validate_combat_model_loads(device="cpu", include_alternates=True)

        self.assertEqual([result["model_name"] for result in results], ["combat", "combat_bc", "combat_pref"])
        for result in results:
            with self.subTest(model=result["model_name"]):
                self.assertTrue(result["selector_available"], msg=str(result))
                self.assertTrue(result["policy_loaded"], msg=str(result))
                self.assertEqual(result["policy_shapes"]["action_logits"], [1, 11])
                self.assertEqual(result["policy_shapes"]["target_logits"], [1, 7])

    def test_default_runtime_selectors_consume_real_actions_on_phase_fixtures(self):
        selectors = build_runtime_selectors(repo_root=Path("/home/yydd/spirecomm"), device="cpu")
        results = validate_selector_phase_fixtures(selectors)

        expected = {
            "combat",
            "card_reward",
            "boss_relic",
            "map",
            "campfire",
            "event",
            "shop",
            "potion",
            "upgrade_target",
            "purge_target",
        }
        self.assertEqual({result["selector"] for result in results}, expected)
        for result in results:
            with self.subTest(selector=result["selector"]):
                self.assertTrue(result["consumed"], msg=str(result))

    @unittest.skipUnless(LIGHTSPEED_BUILD_AVAILABLE, "lightspeed build is required for parity checks")
    def test_model_guided_targeted_parity_cases_match_lightspeed(self):
        selectors = build_runtime_selectors(repo_root=Path("/home/yydd/spirecomm"), device="cpu")
        cases = [
            {"seed": 1, "max_steps": 40},
            {"seed": 1, "target_phase": "EVENT", "target_floor": 3, "max_steps": 200},
            {"seed": 8866187513371018371, "target_phase": "SHOP", "target_floor": 10, "max_steps": 260},
            {"seed": 12, "target_phase": "BOSS_RELIC", "target_floor": 17, "max_steps": 600},
        ]

        for case in cases:
            with self.subTest(case=case):
                result = compare_seed_with_model(
                    selectors=selectors,
                    backend_pair="lightspeed,v3",
                    lightspeed_build=DEFAULT_LIGHTSPEED_BUILD,
                    **case,
                )
                self.assertTrue(result["match"], msg=str(result))
                self.assertTrue(result.get("model_action_executed", True), msg=str(result))

    @unittest.skipUnless(LIGHTSPEED_BUILD_AVAILABLE, "lightspeed build is required for parity checks")
    def test_alternate_combat_models_match_lightspeed_on_short_combat_prefix(self):
        repo_root = Path("/home/yydd/spirecomm")
        for model_name in ("combat_bc.pt", "combat_pref.pt"):
            with self.subTest(model=model_name):
                selectors = build_runtime_selectors(
                    repo_root=repo_root,
                    device="cpu",
                    combat_model=repo_root / "models" / model_name,
                )
                result = compare_seed_with_model(
                    seed=1,
                    selectors=selectors,
                    max_steps=40,
                    backend_pair="lightspeed,v3",
                    lightspeed_build=DEFAULT_LIGHTSPEED_BUILD,
                )
                self.assertTrue(result["match"], msg=str(result))
