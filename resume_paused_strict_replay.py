#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spirecomm.ai.real_game_first_validation import export_native_trace_for_seed
from spirecomm.ai.strict_recorded_run_replay import (
    load_strict_recorded_trace,
    validate_strict_pause_resume_trace,
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"pause manifest is not a JSON object: {path}")
    return payload


def _generate_trace(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    repo_root: Path | None,
    trace_dir: Path | None,
    backend: str,
    device: str,
    combat_device: str | None,
    combat_selector: str | None,
    v3_combat_model: Path | None,
    observation_version: str | None,
    trace_policy: str,
) -> Path:
    seed = int(manifest["seed_long"])
    target_dir = trace_dir or manifest_path.parent / "resume_traces"
    return export_native_trace_for_seed(
        seed,
        trace_dir=target_dir,
        repo_root=repo_root,
        backend=backend,
        ascension=int(manifest.get("ascension", 0) or 0),
        max_steps=None,
        device=device,
        combat_device=combat_device,
        combat_selector=combat_selector,
        v3_combat_model=v3_combat_model,
        observation_version=observation_version,
        trace_schema_mode="strict",
        trace_policy=trace_policy,
    )


def _infer_combat_options_from_trace(manifest: dict[str, Any]) -> tuple[str | None, Path | None]:
    trace_path_raw = manifest.get("trace_path")
    if not trace_path_raw:
        return None, None
    trace_path = Path(str(trace_path_raw))
    if not trace_path.exists():
        return None, None
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    combat_status = (
        ((payload.get("model_status") or {}).get("selectors") or {}).get("combat") or {}
    )
    selector_type = str(combat_status.get("type") or "")
    checkpoint_path = combat_status.get("checkpoint_path")
    if selector_type == "V3CandidateCombatSelector":
        return "v3-candidate", Path(str(checkpoint_path)) if checkpoint_path else None
    if selector_type == "SerializedCombatSelector":
        return "legacy-slot", None
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a regenerated v3 trace against a paused strict replay and request resume."
    )
    parser.add_argument("pause_manifest", type=Path)
    parser.add_argument("--trace-path", type=Path, default=None, help="Use an already regenerated strict trace.")
    parser.add_argument("--trace-dir", type=Path, default=None, help="Directory for an auto-generated resume trace.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--backend", choices=["v2", "v3"], default="v3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default=None)
    parser.add_argument("--combat-selector", choices=["legacy-slot", "v3-candidate"], default=None)
    parser.add_argument("--v3-combat-model", type=Path, default=None)
    parser.add_argument("--observation-version", default=None)
    parser.add_argument(
        "--trace-policy",
        choices=["model-required", "legacy-fallback"],
        default=None,
        help="Policy for auto-generated resume traces; defaults to the paused trace policy, then model-required.",
    )
    parser.add_argument("--abort", default=None, help="Write an abort request instead of a resume request.")
    args = parser.parse_args()

    manifest_path = args.pause_manifest.resolve()
    manifest = _load_manifest(manifest_path)
    request_path = Path(manifest["resume_request_path"])
    result_path = Path(manifest["resume_result_path"])
    pause_id = str(manifest["pause_id"])
    trace_policy = args.trace_policy or str(manifest.get("trace_policy") or "model-required")
    inferred_combat_selector, inferred_v3_combat_model = (
        (None, None) if args.trace_path is not None else _infer_combat_options_from_trace(manifest)
    )
    combat_selector = args.combat_selector or inferred_combat_selector
    v3_combat_model = args.v3_combat_model or inferred_v3_combat_model

    if args.abort is not None:
        request = {
            "command": "abort",
            "pause_id": pause_id,
            "reason": args.abort,
        }
        _write_json_atomic(request_path, request)
        print(json.dumps({"ok": True, "request_path": str(request_path), "command": "abort"}, ensure_ascii=False, indent=2))
        return 0

    trace_path = args.trace_path.resolve() if args.trace_path is not None else _generate_trace(
        manifest=manifest,
        manifest_path=manifest_path,
        repo_root=args.repo_root,
        trace_dir=args.trace_dir,
        backend=args.backend,
        device=args.device,
        combat_device=args.combat_device,
        combat_selector=combat_selector,
        v3_combat_model=v3_combat_model,
        observation_version=args.observation_version,
        trace_policy=trace_policy,
    ).resolve()
    trace = load_strict_recorded_trace(trace_path)
    validation = validate_strict_pause_resume_trace(manifest, trace)
    helper_result = {
        "accepted": bool(validation.get("ok")),
        "source": "resume_paused_strict_replay.py",
        "pause_id": pause_id,
        "trace_path": str(trace_path),
        "validation": validation,
    }
    if not validation.get("ok"):
        _write_json_atomic(result_path, helper_result)
        print(json.dumps(helper_result, ensure_ascii=False, indent=2))
        return 1

    request = {
        "command": "resume",
        "pause_id": pause_id,
        "trace_path": str(trace_path),
        "next_step_to_send": int(manifest["next_step_to_send"]),
    }
    _write_json_atomic(request_path, request)
    helper_result["request_path"] = str(request_path)
    print(json.dumps(helper_result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
