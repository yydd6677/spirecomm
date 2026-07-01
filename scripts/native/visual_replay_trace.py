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
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts.native.export_model_run_checklist import _capture_run, _format_action_with_source, _render_checklist
from spirecomm.ai.real_game_first_validation import _terminate_process_tree


DEFAULT_REPO_ROOT = Path("/home/yydd/spirecomm")
DEFAULT_ALIGN_LAUNCHER = Path("/home/yydd/sts_instances/align/launch_recorded_replay.sh")
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "_cache" / "visual_replay_traces"
DEFAULT_REPORT_DIR = DEFAULT_REPO_ROOT / "_cache" / "visual_replay_reports"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_trace_stem(trace_path: Path) -> str:
    return trace_path.stem.replace(" ", "_")


def _repo_relative_path(path: Path | None, repo_root: Path) -> Path | None:
    if path is None:
        return None
    return (path if path.is_absolute() else repo_root / path).expanduser().resolve()


def _validate_replayable_trace(trace_path: Path, *, mode: str) -> dict[str, Any]:
    payload = _load_json(trace_path)
    steps = list(payload.get("steps") or [])
    if not steps:
        raise ValueError(f"trace has no steps: {trace_path}")
    first = dict(steps[0])
    if "pre_state" not in first or "post_state" not in first:
        raise ValueError(
            "visual replay needs a full native trace with pre_state/post_state. "
            "Compact rollout traces cannot drive the real game; pass --seed to generate a full trace."
        )
    if mode == "strict" and (
        "strict_action" not in first or "strict_pre_state" not in first or "strict_post_state" not in first
    ):
        raise ValueError("strict visual replay needs strict_action/strict_pre_state/strict_post_state in the trace.")
    if payload.get("seed_long") is None and payload.get("seed_str") is None:
        raise ValueError("trace is missing seed_long/seed_str; the real game cannot be started deterministically.")
    return payload


def _generate_trace(
    *,
    seed: int,
    ascension: int,
    backend: str,
    max_steps: int | None,
    repo_root: Path,
    device: str,
    combat_device: str | None,
    combat_model: Path | None,
    combat_selector: str,
    v3_combat_model: Path,
    observation_version: str | None,
    trace_policy: str,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = _capture_run(
        seed=seed,
        ascension=ascension,
        backend=backend,
        max_steps=max_steps,
        repo_root=repo_root,
        device=device,
        combat_device=combat_device,
        combat_model=combat_model,
        combat_selector=combat_selector,
        v3_combat_model=v3_combat_model,
        observation_version=observation_version,
        trace_schema_mode="strict",
        trace_policy=trace_policy,
    )
    prefix = f"seed_{seed}"
    trace_path = output_dir / f"{prefix}_trace.json"
    checklist_path = output_dir / f"{prefix}_checklist.txt"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    checklist_path.write_text(_render_checklist(trace), encoding="utf-8")
    return trace_path, checklist_path, trace


def _write_checklist_for_existing_trace(trace_path: Path, output_dir: Path) -> Path | None:
    output_dir = output_dir.expanduser().resolve()
    payload = _load_json(trace_path)
    if not payload.get("steps") or "pre_state" not in dict(payload["steps"][0]):
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    checklist_path = output_dir / f"{_safe_trace_stem(trace_path)}_checklist.txt"
    checklist_path.write_text(_render_checklist(payload), encoding="utf-8")
    return checklist_path


def _action_lookup(trace: dict[str, Any]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for raw_step in trace.get("steps") or []:
        step = dict(raw_step or {})
        if "pre_state" not in step or "action" not in step:
            continue
        try:
            lookup[int(step["step"])] = _format_action_with_source(step)
        except Exception:
            lookup[int(step.get("step", len(lookup)))] = json.dumps(step.get("action") or {}, ensure_ascii=False)
    return lookup


def _read_json_if_ready(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except json.JSONDecodeError:
        return None


def _print_progress_if_changed(
    *,
    progress_path: Path,
    action_by_step: dict[int, str],
    last_key: tuple[Any, ...] | None,
) -> tuple[Any, ...] | None:
    progress = _read_json_if_ready(progress_path)
    if not progress:
        return last_key
    key = (
        progress.get("status"),
        progress.get("current_step"),
        progress.get("steps_replayed"),
        progress.get("live_phase"),
        progress.get("live_floor"),
    )
    if key == last_key:
        return last_key
    step = progress.get("current_step")
    action_text = ""
    if step is not None:
        try:
            action_text = action_by_step.get(int(step), "")
        except (TypeError, ValueError):
            action_text = ""
    bits = [
        f"status={progress.get('status')}",
        f"step={step}/{progress.get('steps_total')}",
        f"phase={progress.get('current_phase') or progress.get('live_phase')}",
        f"floor={progress.get('live_floor')}",
        f"hp={progress.get('live_hp')}",
    ]
    if action_text:
        bits.append(f"action={action_text}")
    print("[visual-replay] " + " ".join(bits), file=sys.stderr, flush=True)
    return key


def _launch_visual_replay(
    *,
    trace_path: Path,
    trace_payload: dict[str, Any],
    launcher: Path,
    report_dir: Path,
    mode: str,
    character: str | None,
    max_steps: int | None,
    timeout_seconds: float,
    use_xvfb: bool,
    close_on_finish: bool,
    close_on_timeout: bool,
    dry_run: bool,
) -> dict[str, Any]:
    trace_path = trace_path.expanduser().resolve()
    launcher = launcher.expanduser().resolve()
    report_dir = report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{_safe_trace_stem(trace_path)}.visual_replay_report.json"
    progress_path = report_path.with_suffix(".progress.json")
    stdout_path = report_path.with_suffix(".stdout.log")
    stderr_path = report_path.with_suffix(".stderr.log")
    for path in (report_path, progress_path, stdout_path, stderr_path, report_path.with_suffix(".txt")):
        if path.exists():
            path.unlink()

    replay_mode = "strict" if mode == "strict" else "bridge"
    env = {
        **os.environ,
        "STS_USE_XVFB": "1" if use_xvfb else "0",
        "SPIRECOMM_TRACE_PATH": str(trace_path),
        "SPIRECOMM_REPLAY_REPORT": str(report_path),
        "SPIRECOMM_REPLAY_MODE": replay_mode,
        "SPIRECOMM_REPLAY_SESSION_TIMEOUT_SECONDS": str(int(timeout_seconds)),
        "SPIRECOMM_STRICT_REPLAY_READY_TIMEOUT_SECONDS": os.environ.get(
            "SPIRECOMM_STRICT_REPLAY_READY_TIMEOUT_SECONDS",
            "120",
        ),
        "SPIRECOMM_REPLAY_VERBOSE": "1",
    }
    launcher_args = [str(launcher), str(trace_path)]
    if mode == "visual":
        env["SPIRECOMM_REPLAY_COMPARE"] = "0"
        env["SPIRECOMM_REPLAY_KEEP_GOING"] = "1"
        launcher_args.extend(["--no-compare", "--keep-going"])
    if character:
        launcher_args.extend(["--character", character])
    if max_steps is not None:
        launcher_args.extend(["--max-steps", str(max_steps)])

    result: dict[str, Any] = {
        "trace_path": str(trace_path),
        "report_path": str(report_path),
        "progress_path": str(progress_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "mode": mode,
        "replay_mode": replay_mode,
        "use_xvfb": use_xvfb,
        "command": launcher_args,
    }
    if dry_run:
        result["dry_run"] = True
        return result

    if not use_xvfb and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("visual replay requested STS_USE_XVFB=0, but DISPLAY/WAYLAND_DISPLAY is not set.")

    action_by_step = _action_lookup(trace_payload)
    start_time = time.time()
    last_progress_key: tuple[Any, ...] | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            launcher_args,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        result["launcher_pid"] = process.pid
        timed_out = False
        try:
            while True:
                report = _read_json_if_ready(report_path)
                if report and (
                    "success" in report or "runner_error" in report or report.get("paused") is True
                ):
                    result["report"] = report
                    break
                if process.poll() is not None:
                    report = _read_json_if_ready(report_path)
                    if report:
                        result["report"] = report
                    else:
                        result["runner_error"] = f"launcher exited before report was written, returncode={process.returncode}"
                    break
                if timeout_seconds > 0 and time.time() - start_time > timeout_seconds:
                    timed_out = True
                    result["timed_out"] = True
                    result["runner_error"] = f"visual replay timed out after {timeout_seconds:.1f}s"
                    break
                last_progress_key = _print_progress_if_changed(
                    progress_path=progress_path,
                    action_by_step=action_by_step,
                    last_key=last_progress_key,
                )
                time.sleep(0.5)
        finally:
            should_close = close_on_finish if not timed_out else close_on_timeout
            if should_close and process.poll() is None:
                _terminate_process_tree(process.pid)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _terminate_process_tree(process.pid, grace_seconds=0.0)
                    process.wait(timeout=10)
            result["launcher_returncode"] = process.poll()
            result["process_left_running"] = process.poll() is None

    report = result.get("report")
    if isinstance(report, dict):
        result["success"] = bool(report.get("success"))
        result["steps_replayed"] = report.get("steps_replayed")
        result["steps_total"] = report.get("steps_total")
        result["first_failure_step"] = report.get("first_failure_step")
        result["runner_error"] = report.get("runner_error")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or load a full v3 trace, launch a visible Slay the Spire instance, "
            "and auto-play the run through Communication Mod."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path, help="Existing full trace JSON with pre_state/post_state.")
    source.add_argument("--seed", type=int, help="Generate a full trace from this seed before launching.")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--backend", choices=["v1", "v2", "v3"], default="v3")
    parser.add_argument("--trace-policy", choices=["model-required", "legacy-fallback"], default="model-required")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default=None)
    parser.add_argument("--combat-selector", choices=["legacy-slot", "v3-candidate"], default="v3-candidate")
    parser.add_argument("--combat-model", type=Path, default=None, help="Legacy slot combat checkpoint.")
    parser.add_argument("--v3-combat-model", type=Path, default=Path("models/v3_combat_scorer.pt"))
    parser.add_argument("--observation-version", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_ALIGN_LAUNCHER)
    parser.add_argument("--character", default=None)
    parser.add_argument(
        "--mode",
        choices=["visual", "strict"],
        default="strict",
        help=(
            "strict uses the audited strict-action replay path and is the default. "
            "visual is the legacy bridge/no-compare path and is only kept for debugging."
        ),
    )
    parser.add_argument("--xvfb", action="store_true", help="Run headless under Xvfb instead of opening a visible window.")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--close-on-finish", action="store_true", help="Close the game process after replay report is written.")
    parser.add_argument("--close-on-timeout", action="store_true", help="Close the game process if the monitor times out.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare paths and print launch command without starting the game.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = DEFAULT_REPO_ROOT
    checklist_path: Path | None = None
    if args.seed is not None:
        trace_path, checklist_path, trace_payload = _generate_trace(
            seed=args.seed,
            ascension=args.ascension,
            backend=args.backend,
            max_steps=args.max_steps,
            repo_root=repo_root,
            device=args.device,
            combat_device=args.combat_device,
            combat_model=_repo_relative_path(args.combat_model, repo_root),
            combat_selector=args.combat_selector,
            v3_combat_model=_repo_relative_path(args.v3_combat_model, repo_root) or args.v3_combat_model,
            observation_version=args.observation_version,
            trace_policy=args.trace_policy,
            output_dir=args.output_dir,
        )
    else:
        trace_path = args.trace.resolve()
        trace_payload = _validate_replayable_trace(trace_path, mode=args.mode)
        checklist_path = _write_checklist_for_existing_trace(trace_path, args.output_dir)

    trace_payload = _validate_replayable_trace(trace_path, mode=args.mode)
    result = _launch_visual_replay(
        trace_path=trace_path,
        trace_payload=trace_payload,
        launcher=args.launcher,
        report_dir=args.report_dir,
        mode=args.mode,
        character=args.character,
        max_steps=args.max_steps,
        timeout_seconds=args.timeout_seconds,
        use_xvfb=args.xvfb,
        close_on_finish=args.close_on_finish,
        close_on_timeout=args.close_on_timeout,
        dry_run=args.dry_run,
    )
    result["checklist_path"] = str(checklist_path) if checklist_path is not None else None

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(
        json.dumps(
            {
                "trace": result["trace_path"],
                "checklist": result.get("checklist_path"),
                "report": result["report_path"],
                "success": result.get("success"),
                "steps_replayed": result.get("steps_replayed"),
                "steps_total": result.get("steps_total"),
                "first_failure_step": result.get("first_failure_step"),
                "process_left_running": result.get("process_left_running"),
                "launcher_pid": result.get("launcher_pid"),
                "runner_error": result.get("runner_error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"visual replay failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
