#!/usr/bin/env python3
from __future__ import annotations
# Allow this CLI to run directly from its workflow subdirectory.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("models/v3_combat_ppo_actionset.pt.ppo_work/ppo_metrics.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid jsonl row: {exc}") from exc
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _latest_progress(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    rows = _read_jsonl(path)
    return rows[-1] if rows else None


def _get(row: dict[str, Any], path: str, default: float = 0.0) -> float:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def _fmt(value: float) -> str:
    if abs(value) >= 1000.0:
        return f"{value:.0f}"
    if abs(value) >= 100.0:
        return f"{value:.1f}"
    if abs(value) >= 10.0:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _series_points(rows: list[dict[str, Any]], metric: str) -> list[tuple[float, float]]:
    return [(float(index + 1), _get(row, metric)) for index, row in enumerate(rows)]


def _scale(values: list[float], lower: float | None = None, upper: float | None = None) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(values) if lower is None else float(lower)
    hi = max(values) if upper is None else float(upper)
    if lo == hi:
        pad = max(1.0, abs(lo) * 0.1)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    if lower is not None:
        lo = float(lower)
    else:
        lo -= pad
    if upper is not None:
        hi = float(upper)
    else:
        hi += pad
    return lo, hi


def _polyline(
    points: list[tuple[float, float]],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    if not points:
        return ""
    denom_x = max(1.0e-9, x_max - x_min)
    denom_y = max(1.0e-9, y_max - y_min)
    coords = []
    for x_value, y_value in points:
        x = left + (x_value - x_min) / denom_x * width
        y = top + height - (y_value - y_min) / denom_y * height
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def _draw_panel(
    *,
    title: str,
    rows: list[dict[str, Any]],
    metrics: list[tuple[str, str, str]],
    x: float,
    y: float,
    width: float,
    height: float,
    fixed_lower: float | None = None,
    fixed_upper: float | None = None,
) -> str:
    plot_left = x + 52
    plot_top = y + 34
    plot_width = width - 70
    plot_height = height - 62
    x_min = 1.0
    x_max = max(1.0, float(len(rows)))
    all_values: list[float] = []
    metric_points: list[tuple[str, str, list[tuple[float, float]]]] = []
    for metric, label, color in metrics:
        points = _series_points(rows, metric)
        metric_points.append((label, color, points))
        all_values.extend(value for _update, value in points)
    y_min, y_max = _scale(all_values, fixed_lower, fixed_upper)
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="#10151f" stroke="#253248"/>',
        f'<text x="{x + 16}" y="{y + 22}" fill="#edf2ff" font-size="15" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="#40506a"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#40506a"/>',
        f'<text x="{plot_left - 8}" y="{plot_top + 5}" text-anchor="end" fill="#a9b7d0" font-size="11">{html.escape(_fmt(y_max))}</text>',
        f'<text x="{plot_left - 8}" y="{plot_top + plot_height}" text-anchor="end" fill="#a9b7d0" font-size="11">{html.escape(_fmt(y_min))}</text>',
        f'<text x="{plot_left}" y="{plot_top + plot_height + 20}" fill="#a9b7d0" font-size="11">1</text>',
        f'<text x="{plot_left + plot_width}" y="{plot_top + plot_height + 20}" text-anchor="end" fill="#a9b7d0" font-size="11">{len(rows)}</text>',
    ]
    legend_x = x + width - 18
    legend_y = y + 20
    for index, (label, color, points) in enumerate(metric_points):
        polyline = _polyline(
            points,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            left=plot_left,
            top=plot_top,
            width=plot_width,
            height=plot_height,
        )
        if polyline:
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
            last_x, last_y = polyline.split()[-1].split(",")
            parts.append(f'<circle cx="{last_x}" cy="{last_y}" r="3.2" fill="{color}"/>')
        parts.append(f'<text x="{legend_x}" y="{legend_y + index * 15}" text-anchor="end" fill="{color}" font-size="11">{html.escape(label)}</text>')
    return "\n".join(parts)


def _empty_svg(message: str) -> str:
    escaped = html.escape(message)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="360" viewBox="0 0 1100 360">
<rect width="1100" height="360" fill="#0b0f16"/>
<rect x="48" y="64" width="1004" height="210" rx="18" fill="#10151f" stroke="#253248"/>
<text x="80" y="126" fill="#edf2ff" font-size="24" font-weight="700">PPO Training Progress</text>
<text x="80" y="170" fill="#a9b7d0" font-size="16">{escaped}</text>
<text x="80" y="210" fill="#6f7f99" font-size="13">Run scripts/v3_combat/train_v3_combat_ppo.py until at least one update writes ppo_metrics.jsonl, then rerun this visualizer.</text>
</svg>
'''


def _build_summary(
    rows: list[dict[str, Any]],
    input_path: Path,
    output_path: Path,
    latest_progress: dict[str, Any] | None,
) -> dict[str, Any]:
    if not rows:
        return {
            "input": str(input_path),
            "output": str(output_path),
            "update_count": 0,
            "latest_progress": latest_progress or {},
        }
    best_index = max(range(len(rows)), key=lambda index: _get(rows[index], "rollout.mean_floor"))
    last = rows[-1]
    return {
        "input": str(input_path),
        "output": str(output_path),
        "update_count": len(rows),
        "latest_update": int(last.get("update") or len(rows)),
        "latest": {
            "mean_floor": _get(last, "rollout.mean_floor"),
            "root_count": _get(last, "rollout.root_count"),
            "truncated_count": _get(last, "rollout.truncated_count"),
            "loss": _get(last, "train.loss"),
            "approx_kl": _get(last, "train.approx_kl"),
            "kl_to_reference": _get(last, "train.kl_to_reference"),
            "entropy": _get(last, "train.entropy"),
            "clip_fraction": _get(last, "train.clip_fraction"),
            "explained_variance": _get(last, "train.explained_variance"),
        },
        "best_rollout_mean_floor": {
            "update": int(rows[best_index].get("update") or best_index + 1),
            "mean_floor": _get(rows[best_index], "rollout.mean_floor"),
        },
        "latest_progress": latest_progress or {},
    }


def _build_svg(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if not rows:
        progress = summary.get("latest_progress") or {}
        if progress:
            message = (
                "No completed PPO updates found yet. "
                f"latest={progress.get('event')} update={progress.get('update')} "
                f"phase={progress.get('phase', '')} "
                f"completed={progress.get('completed', progress.get('batch', 0))}/"
                f"{progress.get('total', '')}"
            )
        else:
            message = "No completed PPO updates found."
        return _empty_svg(message)
    width = 1280
    height = 920
    latest = summary["latest"]
    subtitle = (
        f"updates={summary['update_count']}  latest_floor={_fmt(latest['mean_floor'])}  "
        f"roots={_fmt(latest['root_count'])}  approx_kl={_fmt(latest['approx_kl'])}  "
        f"ref_kl={_fmt(latest['kl_to_reference'])}"
    )
    progress = summary.get("latest_progress") or {}
    if progress:
        progress_bits = [
            f"progress={progress.get('event')}",
            f"u={progress.get('update')}",
        ]
        if progress.get("phase"):
            progress_bits.append(f"phase={progress.get('phase')}")
        if progress.get("completed") is not None:
            progress_bits.append(f"seeds={progress.get('completed')}/{progress.get('total')}")
        if progress.get("batch") is not None:
            progress_bits.append(f"batch={progress.get('batch')}")
        subtitle = f"{subtitle}  |  {' '.join(str(bit) for bit in progress_bits)}"
    panels = [
        _draw_panel(
            title="Rollout Outcome",
            rows=rows,
            metrics=[
                ("rollout.mean_floor", "mean floor", "#6ee7b7"),
                ("rollout.win_count", "wins", "#facc15"),
                ("rollout.death_count", "deaths", "#fb7185"),
            ],
            x=34,
            y=118,
            width=594,
            height=230,
            fixed_lower=0.0,
        ),
        _draw_panel(
            title="Collected Roots",
            rows=rows,
            metrics=[
                ("rollout.root_count", "root count", "#93c5fd"),
                ("rollout.truncated_count", "truncated seeds", "#f97316"),
            ],
            x=652,
            y=118,
            width=594,
            height=230,
            fixed_lower=0.0,
        ),
        _draw_panel(
            title="PPO Loss",
            rows=rows,
            metrics=[
                ("train.loss", "total", "#c084fc"),
                ("train.policy_loss", "policy", "#60a5fa"),
                ("train.value_loss", "value", "#f472b6"),
            ],
            x=34,
            y=372,
            width=594,
            height=230,
        ),
        _draw_panel(
            title="Policy Drift",
            rows=rows,
            metrics=[
                ("train.approx_kl", "approx KL", "#fbbf24"),
                ("train.kl_to_reference", "KL to old", "#fb7185"),
                ("train.clip_fraction", "clip frac", "#38bdf8"),
            ],
            x=652,
            y=372,
            width=594,
            height=230,
            fixed_lower=0.0,
        ),
        _draw_panel(
            title="Exploration And Value",
            rows=rows,
            metrics=[
                ("train.entropy", "entropy", "#34d399"),
                ("train.explained_variance", "value EV", "#a78bfa"),
                ("train.mean_return", "return", "#f59e0b"),
            ],
            x=34,
            y=626,
            width=1212,
            height=230,
        ),
    ]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#0b0f16"/>
<text x="34" y="48" fill="#edf2ff" font-size="30" font-weight="800">V3 Combat PPO Training Progress</text>
<text x="34" y="78" fill="#a9b7d0" font-size="15">{html.escape(subtitle)}</text>
<text x="34" y="894" fill="#6f7f99" font-size="12">Note: rollout mean floor is sampled training rollout, not deterministic seed1-300 evaluation.</text>
{chr(10).join(panels)}
</svg>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Render v3 combat PPO training metrics to a zero-dependency SVG.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--progress-input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--print-summary", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_name("training_progress.svg")
    summary_path = args.summary or output_path.with_suffix(".summary.json")
    rows = _read_jsonl(input_path)
    progress_path = args.progress_input
    if progress_path is None:
        candidate = input_path.with_name("ppo_progress.jsonl")
        progress_path = candidate if candidate.exists() else None
    latest_progress = _latest_progress(progress_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(rows, input_path, output_path, latest_progress)
    output_path.write_text(_build_svg(rows, summary), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.print_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
