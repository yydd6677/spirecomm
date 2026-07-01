from __future__ import annotations

import json
import sys
import os
import signal
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

from export_model_run_checklist import _capture_run
from spirecomm.ai.real_game_runner import run_seeded_real_game
from spirecomm.ai.runtime_decision import (
    TRACE_POLICY_MODEL_REQUIRED,
    ModelRequiredDecisionError,
    build_runtime_selectors,
    choose_modeled_action,
    normalize_trace_policy,
    source_is_allowed_for_model_required,
)
from spirecomm.native_sim_v2 import NativeRunEnv as V2NativeRunEnv
from spirecomm.native_sim_v3 import NativeRunEnv as V3NativeRunEnv
from spirecomm.seed_helper import canonical_seed_string


DEFAULT_ALIGN_LAUNCHER = Path("/home/yydd/sts_instances/align/launch_recorded_replay.sh")
DEFAULT_TRACE_DIR = Path("/home/yydd/spirecomm/_cache/real_game_first/traces")
DEFAULT_REPLAY_REPORT_DIR = Path("/home/yydd/spirecomm/_cache/real_game_first/replay_reports")
DEFAULT_OUTPUT_PATH = Path("/home/yydd/spirecomm/_cache/real_game_first/report.json")
DEFAULT_REPLAY_TIMEOUT_SECONDS = 600
DEFAULT_FULL_REPLAY_SESSION_TIMEOUT_SECONDS = 7200
TRACE_SOURCE_SCAN_MAX_BYTES = 20 * 1024 * 1024
REPLAY_LAUNCH_GRACE_SECONDS = 120
MAX_REPLAY_ADAPTIVE_GRACE_SECONDS = 600
DEFAULT_BLOCKING_THRESHOLDS = {
    "mean_floor_delta": 0.5,
    "act2_reach_rate_delta": 3.0,
    "act3_reach_rate_delta": 3.0,
    "win_rate_delta": 2.0,
}
DEFAULT_CURATED_REPLAY_SEEDS = [
    1,
    12,
    7133506393411724536,
    8866187513371018371,
]


def _truncate_text(text: str, *, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    removed = len(text) - len(head) - len(tail)
    return f"{head}\n...[truncated {removed} chars]...\n{tail}"


def _persist_launcher_logs(report_path: Path, stdout: str, stderr: str) -> tuple[Path, Path]:
    stdout_path = report_path.with_suffix(".stdout.log")
    stderr_path = report_path.with_suffix(".stderr.log")
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return stdout_path, stderr_path


def _compute_replay_session_timeout_seconds(
    replay_timeout_seconds: int,
    *,
    max_steps: int | None,
) -> int:
    adaptive_grace_seconds = REPLAY_LAUNCH_GRACE_SECONDS
    if max_steps is not None:
        adaptive_grace_seconds = max(
            REPLAY_LAUNCH_GRACE_SECONDS,
            min(int(max_steps) * 4, MAX_REPLAY_ADAPTIVE_GRACE_SECONDS),
        )
    return int(replay_timeout_seconds) + adaptive_grace_seconds


def _collect_descendant_pids(root_pid: int) -> set[int]:
    try:
        ps_output = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
    except Exception:
        return set()
    children_by_parent: dict[int, list[int]] = {}
    for raw_line in ps_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            pid_str, ppid_str = line.split()
            pid = int(pid_str)
            ppid = int(ppid_str)
        except ValueError:
            continue
        children_by_parent.setdefault(ppid, []).append(pid)
    collected: set[int] = set()
    stack = [root_pid]
    while stack:
        current = stack.pop()
        for child_pid in children_by_parent.get(current, []):
            if child_pid in collected:
                continue
            collected.add(child_pid)
            stack.append(child_pid)
    return collected


def _terminate_process_tree(root_pid: int, *, grace_seconds: float = 3.0) -> None:
    tracked_pids = {root_pid, *_collect_descendant_pids(root_pid)}
    try:
        os.killpg(root_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        alive = []
        for pid in sorted(tracked_pids):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue
            alive.append(pid)
        if not alive:
            return
        time.sleep(0.1)
    for pid in sorted(tracked_pids, reverse=True):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def build_seed_corpus(
    *,
    seed_file: Path | None = None,
    count: int = 200,
    random_seed: int = 63,
    sequential: bool = False,
    start_seed: int = 1,
) -> list[int]:
    if seed_file is not None:
        values: list[int] = []
        for raw_line in seed_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values.append(int(line))
        return values
    if sequential:
        return [start_seed + offset for offset in range(count)]
    rng = Random(random_seed)
    return [rng.getrandbits(64) for _ in range(count)]


def _act_reach_flags(result: dict[str, Any]) -> tuple[bool, bool]:
    act = int(result.get("act", 0) or 0)
    floor = int(result.get("floor", 0) or 0)
    reached_act2 = act >= 2 or floor >= 17
    reached_act3 = act >= 3 or floor >= 34
    return reached_act2, reached_act3


def summarize_seed_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "count": 0,
            "mean_floor": 0.0,
            "act2_reach_rate": 0.0,
            "act3_reach_rate": 0.0,
            "win_rate": 0.0,
            "phase_coverage": {},
            "source_coverage": {},
            "max_floor": 0,
            "min_floor": 0,
        }
    phase_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    act2_hits = 0
    act3_hits = 0
    win_hits = 0
    for result in results:
        reached_act2, reached_act3 = _act_reach_flags(result)
        act2_hits += int(reached_act2)
        act3_hits += int(reached_act3)
        win_hits += int(bool(result.get("victory")))
        phase_counts.update(result.get("phase_counts") or {})
        source_counts.update(result.get("source_counts") or {})
    floors = [int(result.get("floor", 0) or 0) for result in results]
    return {
        "count": len(results),
        "mean_floor": float(mean(floors)),
        "act2_reach_rate": 100.0 * act2_hits / len(results),
        "act3_reach_rate": 100.0 * act3_hits / len(results),
        "win_rate": 100.0 * win_hits / len(results),
        "phase_coverage": dict(sorted(phase_counts.items())),
        "source_coverage": dict(sorted(source_counts.items())),
        "max_floor": max(floors),
        "min_floor": min(floors),
    }


def compare_metric_summaries(
    native_summary: dict[str, Any],
    real_summary: dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    active_thresholds = dict(DEFAULT_BLOCKING_THRESHOLDS)
    if thresholds:
        active_thresholds.update(thresholds)
    deltas = {
        "mean_floor_delta": abs(float(native_summary["mean_floor"]) - float(real_summary["mean_floor"])),
        "act2_reach_rate_delta": abs(float(native_summary["act2_reach_rate"]) - float(real_summary["act2_reach_rate"])),
        "act3_reach_rate_delta": abs(float(native_summary["act3_reach_rate"]) - float(real_summary["act3_reach_rate"])),
        "win_rate_delta": abs(float(native_summary["win_rate"]) - float(real_summary["win_rate"])),
    }
    return {
        "thresholds": active_thresholds,
        "deltas": deltas,
        "ok": all(deltas[key] <= active_thresholds[key] for key in deltas),
    }


def _native_env_cls_for_backend(backend: str):
    if backend == "v2":
        return V2NativeRunEnv
    if backend == "v3":
        return V3NativeRunEnv
    raise ValueError(f"Unsupported native backend: {backend}")


def _run_native_seed(
    *,
    seed: int,
    selectors: dict[str, Any],
    ascension: int,
    max_steps: int | None,
    max_floor: int,
    backend: str,
) -> dict[str, Any]:
    env_cls = _native_env_cls_for_backend(backend)
    env = env_cls(seed=seed, ascension_level=ascension, enable_neow=False)
    phase_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    steps = 0
    max_act_seen = int(getattr(env, "act", 1) or 1)
    max_floor_seen = int(getattr(env, "floor", 0) or 0)
    while (
        env.phase not in {"GAME_OVER", "COMPLETE", "VICTORY"}
        and (max_steps is None or steps < max_steps)
        and env.floor <= max_floor
    ):
        phase_counts[env.phase] += 1
        action, _, source = choose_modeled_action(env, selectors)
        source_counts[source] += 1
        env.step(action)
        steps += 1
        max_act_seen = max(max_act_seen, int(getattr(env, "act", 0) or 0))
        max_floor_seen = max(max_floor_seen, int(getattr(env, "floor", 0) or 0))

    return {
        "seed": seed,
        "backend": backend,
        "phase": env.phase,
        "floor": int(env.floor),
        "act": max_act_seen,
        "max_floor_seen": max_floor_seen,
        "steps": steps,
        "victory": env.phase in {"COMPLETE", "VICTORY"},
        "current_hp": int(env.player.current_hp),
        "max_hp": int(env.player.max_hp),
        "gold": int(env.gold),
        "phase_counts": dict(phase_counts),
        "source_counts": dict(source_counts),
    }


def run_native_seed_corpus(
    seeds: list[int],
    *,
    backend: str = "v3",
    repo_root: Path | None = None,
    device: str = "cpu",
    combat_device: str | None = None,
    combat_selector: str | None = None,
    v3_combat_model: Path | None = None,
    observation_version: str | None = None,
    ascension: int = 0,
    max_steps: int | None = None,
    max_floor: int = 60,
) -> dict[str, Any]:
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    selectors = build_runtime_selectors(
        repo_root=root,
        device=device,
        combat_device=combat_device,
        combat_selector=combat_selector,
        v3_combat_model=v3_combat_model,
        observation_version=observation_version,
    )
    selectors["enable_neow"] = False
    results = [
        _run_native_seed(
            seed=seed,
            selectors=selectors,
            ascension=ascension,
            max_steps=max_steps,
            max_floor=max_floor,
            backend=backend,
        )
        for seed in seeds
    ]
    return {
        "results": results,
        "summary": summarize_seed_results(results),
    }


def run_real_game_seed_corpus(
    seeds: list[int],
    *,
    player_class: str = "IRONCLAD",
    ascension: int = 0,
    combat_model: str | Path | None = None,
    device: str = "cpu",
    observation_version: str | None = None,
    trajectory_dir: str | Path | None = None,
) -> dict[str, Any]:
    results = [
        run_seeded_real_game(
            seed=canonical_seed_string(seed) or str(seed),
            player_class=player_class,
            ascension=ascension,
            combat_model=combat_model,
            device=device,
            observation_version=observation_version,
            trajectory_dir=trajectory_dir,
        )
        for seed in seeds
    ]
    return {
        "results": results,
        "summary": summarize_seed_results(results),
    }


def export_native_trace_for_seed(
    seed: int,
    *,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    repo_root: Path | None = None,
    backend: str = "v3",
    ascension: int = 0,
    max_steps: int | None = None,
    device: str = "cpu",
    combat_device: str | None = None,
    combat_selector: str | None = None,
    v3_combat_model: Path | None = None,
    observation_version: str | None = None,
    trace_schema_mode: str = "strict",
    trace_policy: str = TRACE_POLICY_MODEL_REQUIRED,
) -> Path:
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace = _capture_run(
        seed=seed,
        ascension=ascension,
        backend=backend,
        max_steps=max_steps,
        repo_root=root,
        device=device,
        combat_device=combat_device,
        combat_selector=combat_selector,
        v3_combat_model=v3_combat_model,
        observation_version=observation_version,
        trace_schema_mode=trace_schema_mode,
        trace_policy=trace_policy,
    )
    trace_path = trace_dir / f"seed_{seed}_trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return trace_path


def run_v2_seed_corpus(
    seeds: list[int],
    *,
    repo_root: Path | None = None,
    device: str = "cpu",
    combat_device: str | None = None,
    combat_selector: str | None = None,
    v3_combat_model: Path | None = None,
    observation_version: str | None = None,
    ascension: int = 0,
    max_steps: int | None = None,
    max_floor: int = 60,
) -> dict[str, Any]:
    return run_native_seed_corpus(
        seeds,
        backend="v2",
        repo_root=repo_root,
        device=device,
        combat_device=combat_device,
        combat_selector=combat_selector,
        v3_combat_model=v3_combat_model,
        observation_version=observation_version,
        ascension=ascension,
        max_steps=max_steps,
        max_floor=max_floor,
    )


def export_v2_trace_for_seed(
    seed: int,
    *,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    repo_root: Path | None = None,
    backend: str = "v2",
    ascension: int = 0,
    max_steps: int | None = None,
    device: str = "cpu",
    combat_device: str | None = None,
    combat_selector: str | None = None,
    v3_combat_model: Path | None = None,
    observation_version: str | None = None,
    trace_policy: str = TRACE_POLICY_MODEL_REQUIRED,
) -> Path:
    return export_native_trace_for_seed(
        seed,
        trace_dir=trace_dir,
        repo_root=repo_root,
        backend=backend,
        ascension=ascension,
        max_steps=max_steps,
        device=device,
        combat_device=combat_device,
        combat_selector=combat_selector,
        v3_combat_model=v3_combat_model,
        observation_version=observation_version,
        trace_policy=trace_policy,
    )


def run_replay_validation(
    seeds: list[int],
    *,
    backend: str = "v3",
    trace_dir: Path = DEFAULT_TRACE_DIR,
    report_dir: Path = DEFAULT_REPLAY_REPORT_DIR,
    align_launcher: Path = DEFAULT_ALIGN_LAUNCHER,
    launch_align: bool = False,
    keep_going: bool = False,
    use_xvfb: bool = True,
    replay_timeout_seconds: int = DEFAULT_REPLAY_TIMEOUT_SECONDS,
    repo_root: Path | None = None,
    ascension: int = 0,
    max_steps: int | None = None,
    device: str = "cpu",
    combat_device: str | None = None,
    combat_selector: str | None = None,
    v3_combat_model: Path | None = None,
    observation_version: str | None = None,
    replay_mode: str = "strict",
    pause_on_divergence: bool = False,
    trace_policy: str = TRACE_POLICY_MODEL_REQUIRED,
) -> dict[str, Any]:
    trace_dir = trace_dir.resolve()
    report_dir = report_dir.resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    first_failure_histogram: Counter[str] = Counter()
    normalized_trace_policy = normalize_trace_policy(trace_policy)
    for seed in seeds:
        print(f"[replay] exporting trace for seed={seed}", file=sys.stderr, flush=True)
        try:
            trace_path = export_native_trace_for_seed(
                seed,
                trace_dir=trace_dir,
                repo_root=repo_root,
                backend=backend,
                ascension=ascension,
                max_steps=max_steps,
                device=device,
                combat_device=combat_device,
                combat_selector=combat_selector,
                v3_combat_model=v3_combat_model,
                observation_version=observation_version,
                trace_schema_mode="strict",
                trace_policy=normalized_trace_policy,
            )
        except ModelRequiredDecisionError as exc:
            result = {
                "seed": seed,
                "trace_path": None,
                "report_path": str(report_dir / f"seed_{seed}_real_replay_report.json"),
                "trace_policy": normalized_trace_policy,
                "success": False,
                "runner_error": str(exc),
                "trace_error": exc.to_dict(),
                "non_model_sources": ["trace_generation_failed"],
            }
            results.append(result)
            print(
                f"[replay] seed={seed} trace export failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if not keep_going:
                break
            continue
        report_path = report_dir / f"seed_{seed}_real_replay_report.json"
        trace_size_bytes = trace_path.stat().st_size
        non_model_sources: list[str]
        if normalized_trace_policy == TRACE_POLICY_MODEL_REQUIRED or trace_size_bytes <= TRACE_SOURCE_SCAN_MAX_BYTES:
            trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
            non_model_sources = sorted(
                {
                    str(step.get("action_source") or "")
                    for step in trace_payload.get("steps", [])
                    if not source_is_allowed_for_model_required(step.get("action_source"))
                }
            )
        else:
            non_model_sources = ["trace_source_scan_skipped_large_file"]
        result: dict[str, Any] = {
            "seed": seed,
            "trace_path": str(trace_path),
            "trace_policy": normalized_trace_policy,
            "trace_size_bytes": trace_size_bytes,
            "report_path": str(report_path),
            "non_model_sources": non_model_sources,
        }
        if normalized_trace_policy == TRACE_POLICY_MODEL_REQUIRED and non_model_sources:
            result["success"] = False
            result["runner_error"] = (
                "model-required trace contained unexpected non-model action sources: "
                + ", ".join(non_model_sources)
            )
            results.append(result)
            print(
                f"[replay] seed={seed} model-required source audit failed: {non_model_sources}",
                file=sys.stderr,
                flush=True,
            )
            if not keep_going:
                break
            continue
        if launch_align:
            session_timeout_seconds = (
                _compute_replay_session_timeout_seconds(
                    replay_timeout_seconds,
                    max_steps=max_steps,
                )
                if max_steps is not None
                else DEFAULT_FULL_REPLAY_SESSION_TIMEOUT_SECONDS
            )
            if report_path.exists():
                report_path.unlink()
            summary_path = report_path.with_suffix(".txt")
            if summary_path.exists():
                summary_path.unlink()
            progress_path = report_path.with_suffix(".progress.json")
            if progress_path.exists():
                progress_path.unlink()
            strict_state_tap_path = report_path.with_suffix(".state_tap.jsonl")
            if strict_state_tap_path.exists():
                strict_state_tap_path.unlink()
            strict_command_journal_path = report_path.with_suffix(".command_journal.log")
            if strict_command_journal_path.exists():
                strict_command_journal_path.unlink()
            strict_command_queue_dir = report_path.with_suffix(".command_queue.d")
            if strict_command_queue_dir.exists():
                shutil.rmtree(strict_command_queue_dir)
            strict_pause_manifest_path = report_path.with_suffix(".pause.json")
            if strict_pause_manifest_path.exists():
                strict_pause_manifest_path.unlink()
            strict_resume_request_path = report_path.with_suffix(".resume.json")
            if strict_resume_request_path.exists():
                strict_resume_request_path.unlink()
            strict_resume_result_path = report_path.with_suffix(".resume_result.json")
            if strict_resume_result_path.exists():
                strict_resume_result_path.unlink()
            strict_command_transport = os.environ.get(
                "SPIRECOMM_STRICT_COMMAND_TRANSPORT",
                "stdout",
            ).strip() or "stdout"
            env = {
                **dict(os.environ),
                "STS_USE_XVFB": "1" if use_xvfb else "0",
                "SPIRECOMM_REPLAY_REPORT": str(report_path),
                "SPIRECOMM_REPLAY_SESSION_TIMEOUT_SECONDS": str(session_timeout_seconds),
                "SPIRECOMM_REPLAY_MODE": replay_mode,
                "SPIRECOMM_STRICT_STATE_TAP_PATH": str(strict_state_tap_path),
                "SPIRECOMM_STRICT_COMMAND_TRANSPORT": strict_command_transport,
                "SPIRECOMM_STRICT_COMMAND_JOURNAL_PATH": str(strict_command_journal_path),
                "SPIRECOMM_STRICT_COMMAND_QUEUE_DIR": str(strict_command_queue_dir),
                "SPIRECOMM_STRICT_PAUSE_ON_DIVERGENCE": "1" if pause_on_divergence else "0",
                "SPIRECOMM_STRICT_PAUSE_MANIFEST_PATH": str(strict_pause_manifest_path),
                "SPIRECOMM_STRICT_RESUME_REQUEST_PATH": str(strict_resume_request_path),
                "SPIRECOMM_STRICT_RESUME_RESULT_PATH": str(strict_resume_result_path),
            }
            args = [str(align_launcher), str(trace_path)]
            if keep_going:
                args.append("--keep-going")
            print(
                f"[replay] launching align for seed={seed} trace={trace_path} report={report_path}",
                file=sys.stderr,
                flush=True,
            )
            stdout_path = report_path.with_suffix(".stdout.log")
            stderr_path = report_path.with_suffix(".stderr.log")
            if stdout_path.exists():
                stdout_path.unlink()
            if stderr_path.exists():
                stderr_path.unlink()
            replay_report: dict[str, Any] | None = None
            timed_out = False
            paused = False
            leave_process_running = False
            stdout = ""
            stderr = ""
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
                process = subprocess.Popen(
                    args,
                    env=env,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    start_new_session=True,
                )
                try:
                    deadline = time.time() + session_timeout_seconds
                    next_progress_at = time.time() + 10.0
                    while time.time() < deadline:
                        if report_path.exists():
                            try:
                                replay_report = json.loads(report_path.read_text(encoding="utf-8"))
                            except json.JSONDecodeError:
                                time.sleep(0.25)
                                continue
                            if isinstance(replay_report, dict) and (
                                "success" in replay_report
                                or "runner_error" in replay_report
                                or replay_report.get("paused") is True
                            ):
                                if replay_report.get("paused") is True:
                                    paused = True
                                    leave_process_running = process.poll() is None
                                break
                        if process.poll() is not None:
                            break
                        now = time.time()
                        if now >= next_progress_at:
                            elapsed = int(now - (deadline - session_timeout_seconds))
                            progress_suffix = ""
                            if progress_path.exists():
                                try:
                                    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
                                except json.JSONDecodeError:
                                    progress_payload = None
                                if isinstance(progress_payload, dict):
                                    progress_suffix = (
                                        f", progress_step={progress_payload.get('current_step')}"
                                        f", progress_phase={progress_payload.get('current_phase')}"
                                        f", live_floor={progress_payload.get('live_floor')}"
                                        f", status={progress_payload.get('status')}"
                                    )
                            print(
                                f"[replay] seed={seed} still running "
                                f"({elapsed}s elapsed, timeout={session_timeout_seconds}s, "
                                f"replay_timeout={replay_timeout_seconds}s{progress_suffix})",
                                file=sys.stderr,
                                flush=True,
                            )
                            next_progress_at = now + 10.0
                        time.sleep(0.25)
                    else:
                        timed_out = True
                finally:
                    if not leave_process_running:
                        _terminate_process_tree(process.pid)
                        wait = getattr(process, "wait", None)
                        if callable(wait):
                            try:
                                wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                _terminate_process_tree(process.pid, grace_seconds=0.0)
                                wait(timeout=10)

            stdout = stdout_path.read_text(encoding="utf-8", errors="ignore")
            stderr = stderr_path.read_text(encoding="utf-8", errors="ignore")

            result["launcher_pid"] = process.pid
            result["launcher_returncode"] = process.poll()
            result["launcher_stdout_bytes"] = len(stdout.encode("utf-8", errors="ignore"))
            result["launcher_stderr_bytes"] = len(stderr.encode("utf-8", errors="ignore"))
            result["launcher_stdout_tail"] = _truncate_text(stdout)
            result["launcher_stderr_tail"] = _truncate_text(stderr)
            result["launcher_stdout_path"] = str(stdout_path)
            result["launcher_stderr_path"] = str(stderr_path)
            result["timeout_seconds"] = replay_timeout_seconds
            result["session_timeout_seconds"] = session_timeout_seconds
            result["timed_out"] = timed_out
            result["strict_state_tap_path"] = str(strict_state_tap_path)
            result["paused"] = paused
            result["paused_process_left_running"] = leave_process_running
            result["pause_manifest_path"] = str(strict_pause_manifest_path)
            result["resume_request_path"] = str(strict_resume_request_path)
            result["resume_result_path"] = str(strict_resume_result_path)
            if replay_report is None and report_path.exists():
                replay_report = json.loads(report_path.read_text(encoding="utf-8"))
            if replay_report is None:
                replay_report = {
                    "trace_path": str(trace_path),
                    "success": False,
                    "runner_error": (
                        f"replay report missing after align exit "
                        f"(returncode={process.returncode}, timed_out={timed_out})"
                    ),
                }
                report_path.write_text(
                    json.dumps(replay_report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            if progress_path.exists():
                try:
                    result["last_progress"] = json.loads(progress_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    result["progress_path"] = str(progress_path)
            if replay_report is not None:
                result["success"] = bool(replay_report.get("success"))
                result["first_failure_step"] = replay_report.get("first_failure_step")
                result["first_failure_phase"] = replay_report.get("first_failure_phase")
                result["state_diff_summary"] = replay_report.get("first_failure_mismatches") or []
                if replay_report.get("first_failure_phase"):
                    first_failure_histogram[str(replay_report["first_failure_phase"])] += 1
                if replay_report.get("runner_error") is not None:
                    result["runner_error"] = replay_report.get("runner_error")
                print(
                    f"[replay] seed={seed} finished success={result['success']} "
                    f"first_failure_step={result.get('first_failure_step')}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                result["success"] = False
                result["runner_error"] = "replay report was not created"
                print(
                    f"[replay] seed={seed} finished without a replay report",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            result["success"] = None
        results.append(result)

    replayed = [item for item in results if item.get("success") is not None]
    all_green = bool(replayed) and all(bool(item.get("success")) for item in replayed)
    return {
        "results": results,
        "summary": {
            "count": len(results),
            "replayed": len(replayed),
            "all_green": all_green if replayed else None,
            "first_failure_histogram": dict(sorted(first_failure_histogram.items())),
            "replay_mode": replay_mode,
        },
    }


def build_real_game_first_report(
    *,
    seed_corpus: list[int],
    curated_replay_seeds: list[int] | None = None,
    native_backend: str = "v3",
    repo_root: Path | None = None,
    device: str = "cpu",
    combat_device: str | None = None,
    combat_selector: str | None = None,
    v3_combat_model: Path | None = None,
    observation_version: str | None = None,
    ascension: int = 0,
    max_steps: int | None = None,
    max_floor: int = 60,
    run_native: bool = True,
    run_real: bool = False,
    run_replay: bool = False,
    launch_align: bool = False,
    keep_going: bool = False,
    use_xvfb: bool = True,
    trajectory_dir: str | Path | None = None,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    replay_report_dir: Path = DEFAULT_REPLAY_REPORT_DIR,
    thresholds: dict[str, float] | None = None,
    replay_mode: str = "strict",
    pause_on_divergence: bool = False,
    trace_policy: str = TRACE_POLICY_MODEL_REQUIRED,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "seed_corpus": [int(seed) for seed in seed_corpus],
        "curated_replay_seeds": [int(seed) for seed in (curated_replay_seeds or DEFAULT_CURATED_REPLAY_SEEDS)],
        "trace_policy": normalize_trace_policy(trace_policy),
        "real_game_blocking": {},
        "lightspeed_reference": {
            "status": "skipped",
            "note": "lightspeed is retained only as a historical reference and no longer blocks validation.",
        },
    }

    native_block = run_native_seed_corpus(
        seed_corpus,
        backend=native_backend,
        repo_root=repo_root,
        device=device,
        combat_device=combat_device,
        combat_selector=combat_selector,
        v3_combat_model=v3_combat_model,
        observation_version=observation_version,
        ascension=ascension,
        max_steps=max_steps,
        max_floor=max_floor,
    ) if run_native else None
    real_block = run_real_game_seed_corpus(
        seed_corpus,
        player_class="IRONCLAD",
        ascension=ascension,
        device=device,
        observation_version=observation_version,
        trajectory_dir=trajectory_dir,
    ) if run_real else None
    replay_block = run_replay_validation(
        curated_replay_seeds or DEFAULT_CURATED_REPLAY_SEEDS,
        backend=native_backend,
        trace_dir=trace_dir,
        report_dir=replay_report_dir,
        launch_align=launch_align,
        keep_going=keep_going,
        use_xvfb=use_xvfb,
        repo_root=repo_root,
        ascension=ascension,
        max_steps=max_steps,
        device=device,
        combat_device=combat_device,
        combat_selector=combat_selector,
        v3_combat_model=v3_combat_model,
        observation_version=observation_version,
        replay_mode=replay_mode,
        pause_on_divergence=pause_on_divergence,
        trace_policy=trace_policy,
    ) if run_replay else None

    report["real_game_blocking"]["native_backend"] = native_backend
    report["real_game_blocking"]["native_rollout"] = native_block
    report["real_game_blocking"]["real_game_rollout"] = real_block
    report["real_game_blocking"]["replay"] = replay_block
    metric_delta = compare_metric_summaries(native_block["summary"], real_block["summary"], thresholds=thresholds) if native_block and real_block else None
    report["real_game_blocking"]["metric_delta"] = metric_delta

    replay_ok = replay_block["summary"]["all_green"] if replay_block is not None else None
    metric_ok = metric_delta["ok"] if metric_delta is not None else None
    partial_checks = [value for value in (metric_ok, replay_ok) if value is not None]
    report["real_game_blocking"]["ok"] = all(partial_checks) if partial_checks else None
    return report


def render_real_game_first_summary(report: dict[str, Any]) -> str:
    lines = ["Real-Game-First Validation Summary"]
    lines.append(f"seed_corpus_count: {len(report.get('seed_corpus') or [])}")
    lines.append(f"curated_replay_seeds: {', '.join(str(seed) for seed in report.get('curated_replay_seeds') or [])}")
    blocking = dict(report.get("real_game_blocking") or {})
    lines.append(f"real_game_blocking.ok: {blocking.get('ok')}")

    def _append_rollout(prefix: str, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        summary = payload.get("summary") or {}
        lines.append(
            f"{prefix}: count={summary.get('count')} mean_floor={summary.get('mean_floor')} "
            f"act2={summary.get('act2_reach_rate')}% act3={summary.get('act3_reach_rate')}% win={summary.get('win_rate')}%"
        )

    native_backend = blocking.get("native_backend") or "native"
    _append_rollout(f"{native_backend}_rollout", blocking.get("native_rollout"))
    _append_rollout("real_game_rollout", blocking.get("real_game_rollout"))

    metric_delta = blocking.get("metric_delta")
    if metric_delta:
        lines.append(f"metric_delta.ok: {metric_delta.get('ok')}")
        for key, value in (metric_delta.get("deltas") or {}).items():
            threshold = (metric_delta.get("thresholds") or {}).get(key)
            lines.append(f"  - {key}: {value} (threshold={threshold})")

    replay = blocking.get("replay")
    if replay:
        replay_summary = replay.get("summary") or {}
        lines.append(
            f"replay: count={replay_summary.get('count')} replayed={replay_summary.get('replayed')} "
            f"all_green={replay_summary.get('all_green')}"
        )
        histogram = replay_summary.get("first_failure_histogram") or {}
        if histogram:
            lines.append("  first_failure_histogram:")
            for phase, count in sorted(histogram.items()):
                lines.append(f"    - {phase}: {count}")

    reference = report.get("lightspeed_reference") or {}
    lines.append(f"lightspeed_reference: {reference.get('status')} ({reference.get('note')})")
    return "\n".join(lines) + "\n"
