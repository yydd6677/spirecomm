from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spirecomm.ai.real_game_first_validation import (
    DEFAULT_FULL_REPLAY_SESSION_TIMEOUT_SECONDS,
    _compute_replay_session_timeout_seconds,
    REPLAY_LAUNCH_GRACE_SECONDS,
    _persist_launcher_logs,
    _run_native_seed,
    build_seed_corpus,
    compare_metric_summaries,
    render_real_game_first_summary,
    run_replay_validation,
    summarize_seed_results,
)


class RealGameFirstValidationTest(unittest.TestCase):
    def test_compute_replay_session_timeout_seconds_scales_with_max_steps(self):
        self.assertEqual(
            _compute_replay_session_timeout_seconds(60, max_steps=None),
            60 + REPLAY_LAUNCH_GRACE_SECONDS,
        )
        self.assertEqual(
            _compute_replay_session_timeout_seconds(60, max_steps=64),
            60 + 256,
        )

    def test_persist_launcher_logs_writes_stdout_and_stderr_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "seed_1_real_replay_report.json"
            stdout_path, stderr_path = _persist_launcher_logs(report_path, "hello stdout", "hello stderr")

            self.assertEqual(stdout_path, report_path.with_suffix(".stdout.log"))
            self.assertEqual(stderr_path, report_path.with_suffix(".stderr.log"))
            self.assertEqual(stdout_path.read_text(encoding="utf-8"), "hello stdout")
            self.assertEqual(stderr_path.read_text(encoding="utf-8"), "hello stderr")

    def test_run_native_seed_treats_victory_as_terminal_success(self):
        class VictoryEnv:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.phase = "MAP"
                self.floor = 12
                self.act = 3
                self.gold = 99
                self.player = type("Player", (), {"current_hp": 77, "max_hp": 80})()

            def step(self, action):
                del action
                self.phase = "VICTORY"

        with patch("spirecomm.ai.real_game_first_validation._native_env_cls_for_backend", return_value=VictoryEnv):
            with patch("spirecomm.ai.real_game_first_validation.choose_modeled_action", return_value=({"kind": "map"}, [1.0], "map")):
                result = _run_native_seed(
                    seed=1,
                    selectors={},
                    ascension=0,
                    max_steps=50,
                    max_floor=60,
                    backend="v3",
                )

        self.assertEqual(result["phase"], "VICTORY")
        self.assertTrue(result["victory"])
        self.assertEqual(result["steps"], 1)

    def test_build_seed_corpus_reads_seed_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "seeds.txt"
            path.write_text("# comment\n1\n\n2\n3\n", encoding="utf-8")
            self.assertEqual(build_seed_corpus(seed_file=path), [1, 2, 3])

    def test_build_seed_corpus_is_deterministic_for_random_mode(self):
        first = build_seed_corpus(count=5, random_seed=7)
        second = build_seed_corpus(count=5, random_seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_summarize_seed_results_uses_act_or_floor_for_reach_rates(self):
        summary = summarize_seed_results(
            [
                {"floor": 16, "act": 1, "victory": False, "phase_counts": {"COMBAT": 3}, "source_counts": {"combat": 3}},
                {"floor": 17, "act": 2, "victory": False, "phase_counts": {"MAP": 1}, "source_counts": {"map": 1}},
                {"floor": 35, "act": 3, "victory": True, "phase_counts": {"SHOP": 2}, "source_counts": {"shop": 2}},
            ]
        )

        self.assertAlmostEqual(summary["mean_floor"], (16 + 17 + 35) / 3)
        self.assertAlmostEqual(summary["act2_reach_rate"], 200.0 / 3.0)
        self.assertAlmostEqual(summary["act3_reach_rate"], 100.0 / 3.0)
        self.assertAlmostEqual(summary["win_rate"], 100.0 / 3.0)
        self.assertEqual(summary["phase_coverage"]["COMBAT"], 3)
        self.assertEqual(summary["source_coverage"]["shop"], 2)

    def test_compare_metric_summaries_applies_blocking_thresholds(self):
        native = {"mean_floor": 10.0, "act2_reach_rate": 20.0, "act3_reach_rate": 5.0, "win_rate": 1.0}
        real = {"mean_floor": 10.3, "act2_reach_rate": 21.0, "act3_reach_rate": 7.5, "win_rate": 2.5}

        comparison = compare_metric_summaries(native, real)

        self.assertTrue(comparison["ok"])
        self.assertAlmostEqual(comparison["deltas"]["mean_floor_delta"], 0.3)
        self.assertAlmostEqual(comparison["deltas"]["act3_reach_rate_delta"], 2.5)

    def test_render_real_game_first_summary_includes_blocking_sections(self):
        text = render_real_game_first_summary(
            {
                "seed_corpus": [1, 2],
                "curated_replay_seeds": [1],
                "real_game_blocking": {
                    "ok": True,
                    "native_rollout": {"summary": {"count": 2, "mean_floor": 10.0, "act2_reach_rate": 50.0, "act3_reach_rate": 0.0, "win_rate": 0.0}},
                    "metric_delta": {"ok": True, "deltas": {"mean_floor_delta": 0.1}, "thresholds": {"mean_floor_delta": 0.5}},
                },
                "lightspeed_reference": {"status": "skipped", "note": "reference only"},
            }
        )

        self.assertIn("Real-Game-First Validation Summary", text)
        self.assertIn("real_game_blocking.ok: True", text)
        self.assertIn("native_rollout: count=2", text)
        self.assertIn("mean_floor_delta: 0.1", text)

    def test_run_replay_validation_uses_absolute_trace_and_report_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            trace_dir = repo_root / "relative_traces"
            report_dir = repo_root / "relative_reports"

            (trace_dir).mkdir(parents=True, exist_ok=True)

            fake_trace = trace_dir / "seed_1_trace.json"
            fake_trace.write_text("{}", encoding="utf-8")

            class FakeProcess:
                def __init__(self):
                    self.returncode = 0
                    self.pid = 12345

                def poll(self):
                    return 0

                def communicate(self, timeout=None):
                    del timeout
                    return ("", "")

            with patch("spirecomm.ai.real_game_first_validation.export_native_trace_for_seed", return_value=fake_trace) as export_trace:
                with patch("spirecomm.ai.real_game_first_validation.subprocess.Popen", return_value=FakeProcess()) as popen:
                    with patch("spirecomm.ai.real_game_first_validation._terminate_process_tree", return_value=None):
                        run_replay_validation(
                            [1],
                            backend="v3",
                            trace_dir=trace_dir.relative_to(repo_root),
                            report_dir=report_dir.relative_to(repo_root),
                            launch_align=True,
                            replay_timeout_seconds=1,
                        )

            args, kwargs = popen.call_args
            launched_args = args[0]
            launched_env = kwargs["env"]
            self.assertTrue(Path(launched_args[1]).is_absolute())
            self.assertTrue(Path(launched_env["SPIRECOMM_REPLAY_REPORT"]).is_absolute())
            self.assertEqual(launched_env["SPIRECOMM_REPLAY_MODE"], "strict")
            self.assertEqual(
                launched_env["SPIRECOMM_REPLAY_SESSION_TIMEOUT_SECONDS"],
                str(DEFAULT_FULL_REPLAY_SESSION_TIMEOUT_SECONDS),
            )
            self.assertEqual(export_trace.call_args.kwargs["trace_policy"], "model_required")

    def test_run_replay_validation_rejects_unexpected_model_required_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace_dir = root / "traces"
            report_dir = root / "reports"
            trace_dir.mkdir(parents=True, exist_ok=True)
            fake_trace = trace_dir / "seed_1_trace.json"
            fake_trace.write_text(
                json.dumps({"steps": [{"action_source": "fallback"}]}),
                encoding="utf-8",
            )

            with patch("spirecomm.ai.real_game_first_validation.export_native_trace_for_seed", return_value=fake_trace):
                report = run_replay_validation(
                    [1],
                    backend="v3",
                    trace_dir=trace_dir,
                    report_dir=report_dir,
                    launch_align=False,
                    trace_policy="model-required",
                )

        result = report["results"][0]
        self.assertFalse(result["success"])
        self.assertEqual(result["non_model_sources"], ["fallback"])
        self.assertIn("unexpected non-model", result["runner_error"])

    def test_run_replay_validation_leaves_paused_process_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace_dir = root / "traces"
            report_dir = root / "reports"
            trace_dir.mkdir(parents=True, exist_ok=True)
            fake_trace = trace_dir / "seed_1_trace.json"
            fake_trace.write_text("{}", encoding="utf-8")

            class FakeProcess:
                def __init__(self, *args, **kwargs):
                    self.pid = 43210
                    self.returncode = None
                    report_path = Path(kwargs["env"]["SPIRECOMM_REPLAY_REPORT"])
                    report_path.write_text(
                        json.dumps({"success": False, "paused": True, "first_failure_step": 3}),
                        encoding="utf-8",
                    )

                def poll(self):
                    return None

            with patch("spirecomm.ai.real_game_first_validation.export_native_trace_for_seed", return_value=fake_trace):
                with patch("spirecomm.ai.real_game_first_validation.subprocess.Popen", side_effect=FakeProcess):
                    with patch("spirecomm.ai.real_game_first_validation._terminate_process_tree", return_value=None) as terminate:
                        report = run_replay_validation(
                            [1],
                            backend="v3",
                            trace_dir=trace_dir,
                            report_dir=report_dir,
                            launch_align=True,
                            replay_timeout_seconds=1,
                            pause_on_divergence=True,
                        )

            result = report["results"][0]
            self.assertTrue(result["paused"])
            self.assertTrue(result["paused_process_left_running"])
            self.assertEqual(result["launcher_pid"], 43210)
            terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
