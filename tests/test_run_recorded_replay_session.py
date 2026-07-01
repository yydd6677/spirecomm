from __future__ import annotations

import io
import os
import importlib
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


class RunRecordedReplaySessionTest(unittest.TestCase):
    def test_bootstrap_ready_if_needed_sends_ready_once_when_trace_path_exists(self):
        stdout = io.StringIO()
        env = {
            "SPIRECOMM_TRACE_PATH": "/tmp/fake_trace.json",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("SPIRECOMM_BOOTSTRAP_READY_SENT", None)
            with patch("sys.stdout", stdout):
                module = importlib.import_module("scripts.native.run_recorded_replay_session")
                module._bootstrap_ready_if_needed()
            self.assertEqual(os.environ.get("SPIRECOMM_BOOTSTRAP_READY_SENT"), "1")

        self.assertEqual(stdout.getvalue(), "ready\n")

    def test_bootstrap_ready_if_needed_is_noop_after_first_send(self):
        stdout = io.StringIO()
        with patch.dict(
            os.environ,
            {
                "SPIRECOMM_TRACE_PATH": "/tmp/fake_trace.json",
                "SPIRECOMM_BOOTSTRAP_READY_SENT": "1",
            },
            clear=False,
        ):
            with patch("sys.stdout", stdout):
                module = importlib.import_module("scripts.native.run_recorded_replay_session")
                module._bootstrap_ready_if_needed()

        self.assertEqual(stdout.getvalue(), "")

    def test_main_dispatches_strict_replay_by_default(self):
        module = importlib.import_module("scripts.native.run_recorded_replay_session")
        with patch.dict(
            os.environ,
            {
                "SPIRECOMM_TRACE_PATH": "/tmp/fake_trace.json",
                "SPIRECOMM_REPLAY_REPORT": "/tmp/fake_report.json",
                "SPIRECOMM_BOOTSTRAP_READY_SENT": "1",
            },
            clear=False,
        ):
            with patch.object(module, "replay_recorded_run_strict", return_value={"success": True, "steps_replayed": 0, "steps_total": 0}) as strict_replay:
                with patch.object(module, "render_strict_replay_report_summary", return_value="ok\n"):
                    exit_code = module.main()

        self.assertEqual(exit_code, 0)
        self.assertTrue(strict_replay.called)
        self.assertEqual(strict_replay.call_args.kwargs["raw_state_log_path"], Path("/tmp/fake_report.raw_state_log.jsonl"))

    def test_main_clears_stale_replay_artifacts_before_start(self):
        module = importlib.import_module("scripts.native.run_recorded_replay_session")
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "fake_report.json"
            stale_paths = [
                report_path.with_suffix(".progress.json"),
                report_path.with_suffix(".coordinator.log"),
                report_path.with_suffix(".raw_state_log.jsonl"),
                report_path.with_suffix(".raw_state_log.debug.jsonl"),
                report_path.with_suffix(".pause.json"),
                report_path.with_suffix(".resume.json"),
                report_path.with_suffix(".resume_result.json"),
            ]
            for path in stale_paths:
                path.write_text("stale", encoding="utf-8")

            def _fake_replay(**kwargs):
                for path in stale_paths:
                    self.assertFalse(path.exists(), path)
                return {"success": True, "steps_replayed": 0, "steps_total": 0}

            with patch.dict(
                os.environ,
                {
                    "SPIRECOMM_TRACE_PATH": "/tmp/fake_trace.json",
                    "SPIRECOMM_REPLAY_REPORT": str(report_path),
                    "SPIRECOMM_BOOTSTRAP_READY_SENT": "1",
                },
                clear=False,
            ):
                with patch.object(module, "replay_recorded_run_strict", side_effect=_fake_replay):
                    with patch.object(module, "render_strict_replay_report_summary", return_value="ok\n"):
                        exit_code = module.main()

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
