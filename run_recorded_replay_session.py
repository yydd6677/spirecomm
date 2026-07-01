#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

def _bootstrap_ready_if_needed() -> None:
    if not os.environ.get("SPIRECOMM_TRACE_PATH"):
        return
    if os.environ.get("SPIRECOMM_BOOTSTRAP_READY_SENT") == "1":
        return
    print("ready", file=sys.stdout, flush=True)
    os.environ["SPIRECOMM_BOOTSTRAP_READY_SENT"] = "1"


_bootstrap_ready_if_needed()

from spirecomm.ai import recorded_run_replay as recorded_run_replay_module
from spirecomm.ai.recorded_run_replay import render_replay_report_summary, replay_recorded_run
from spirecomm.ai.strict_recorded_run_replay import (
    render_strict_replay_report_summary,
    replay_recorded_run_strict,
)


class _ReplaySessionTimeout(RuntimeError):
    pass


def _install_session_timeout(timeout_seconds: float):
    if timeout_seconds <= 0:
        return lambda: None

    def _handle_timeout(signum, frame):  # pragma: no cover - integration-level timeout path
        raise _ReplaySessionTimeout(
            f"recorded replay session exceeded {timeout_seconds:.1f}s without finishing"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)

    def _cleanup():
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)

    return _cleanup


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def main() -> int:
    print(
        f"[replay-session] script={__file__} module={recorded_run_replay_module.__file__}",
        file=sys.stderr,
        flush=True,
    )
    trace_path = os.environ.get("SPIRECOMM_TRACE_PATH")
    if not trace_path:
        print("SPIRECOMM_TRACE_PATH is required for recorded replay sessions.", file=sys.stderr, flush=True)
        return 2

    report_path = os.environ.get("SPIRECOMM_REPLAY_REPORT")
    if report_path is None:
        report_path = str(Path(trace_path).with_suffix(".real_replay_report.json"))
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    progress_path = report_file.with_suffix(".progress.json")
    coordinator_log_path = report_file.with_suffix(".coordinator.log")
    os.environ.setdefault("SPIRECOMM_COORDINATOR_LOG", str(coordinator_log_path))

    replay_mode = os.environ.get("SPIRECOMM_REPLAY_MODE", "strict").strip().lower()
    pause_on_divergence = _env_flag("SPIRECOMM_STRICT_PAUSE_ON_DIVERGENCE", False)
    timeout_seconds = float(os.environ.get("SPIRECOMM_REPLAY_SESSION_TIMEOUT_SECONDS", "900"))
    cleanup_timeout = _install_session_timeout(0.0 if pause_on_divergence else timeout_seconds)
    raw_state_log_path = report_file.with_suffix(".raw_state_log.jsonl")
    pause_manifest_path = report_file.with_suffix(".pause.json")
    resume_request_path = report_file.with_suffix(".resume.json")
    resume_result_path = report_file.with_suffix(".resume_result.json")
    raw_state_debug_log_path = raw_state_log_path.with_name(raw_state_log_path.stem + ".debug.jsonl")
    for stale_path in (
        progress_path,
        coordinator_log_path,
        raw_state_log_path,
        raw_state_debug_log_path,
        pause_manifest_path,
        resume_request_path,
        resume_result_path,
    ):
        _unlink_if_exists(stale_path)
    os.environ.setdefault("SPIRECOMM_STRICT_PAUSE_MANIFEST_PATH", str(pause_manifest_path))
    os.environ.setdefault("SPIRECOMM_STRICT_RESUME_REQUEST_PATH", str(resume_request_path))
    os.environ.setdefault("SPIRECOMM_STRICT_RESUME_RESULT_PATH", str(resume_result_path))

    try:
        if replay_mode == "bridge":
            report = replay_recorded_run(
                trace_path=trace_path,
                character=os.environ.get("SPIRECOMM_REPLAY_CHARACTER"),
                compare_state=_env_flag("SPIRECOMM_REPLAY_COMPARE", True),
                stop_on_mismatch=not _env_flag("SPIRECOMM_REPLAY_KEEP_GOING", False),
                max_steps=int(os.environ["SPIRECOMM_REPLAY_MAX_STEPS"]) if os.environ.get("SPIRECOMM_REPLAY_MAX_STEPS") else None,
                progress_path=progress_path,
            )
            summary_renderer = render_replay_report_summary
        else:
            report = replay_recorded_run_strict(
                trace_path=trace_path,
                character=os.environ.get("SPIRECOMM_REPLAY_CHARACTER"),
                max_steps=int(os.environ["SPIRECOMM_REPLAY_MAX_STEPS"]) if os.environ.get("SPIRECOMM_REPLAY_MAX_STEPS") else None,
                raw_state_log_path=raw_state_log_path,
                pause_on_divergence=pause_on_divergence,
                pause_manifest_path=pause_manifest_path,
                resume_request_path=resume_request_path,
                resume_result_path=resume_result_path,
                pause_report_path=report_path,
            )
            summary_renderer = render_strict_replay_report_summary
    except Exception as exc:  # pragma: no cover - communication loop failures are integration-level
        failure_report = {
            "trace_path": trace_path,
            "success": False,
            "runner_error": repr(exc),
            "replay_mode": replay_mode,
        }
        if progress_path.exists():
            try:
                failure_report["last_progress"] = json.loads(progress_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                failure_report["progress_path"] = str(progress_path)
        if coordinator_log_path.exists():
            failure_report["coordinator_log_path"] = str(coordinator_log_path)
        report_file.write_text(json.dumps(failure_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"recorded replay failed: {exc!r}", file=sys.stderr, flush=True)
        return 1
    finally:
        cleanup_timeout()

    report["coordinator_log_path"] = str(coordinator_log_path)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_file.with_suffix(".txt").write_text(summary_renderer(report), encoding="utf-8")
    print(
        f"recorded replay finished: success={report['success']} steps={report['steps_replayed']}/{report['steps_total']} report={report_path} progress={progress_path}",
        file=sys.stderr,
        flush=True,
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
