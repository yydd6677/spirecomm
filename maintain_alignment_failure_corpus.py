#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from compare_native_to_lightspeed_run import _attach_failure_metadata, _compare_seed_with_timeout, _summarize_clusters


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_list(random_seed: int, count: int) -> list[int]:
    rng = random.Random(random_seed)
    return [rng.randint(0, (1 << 63) - 1) for _ in range(count)]


def _emit_progress(prefix: str, completed: int, total: int, failures: int, start_time: float) -> None:
    elapsed = max(0.0, time.time() - start_time)
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else float("inf")
    eta_text = f"{eta:.1f}s" if eta != float("inf") else "unknown"
    print(
        f"[{prefix}] completed={completed}/{total} failures={failures} elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta_text}",
        flush=True,
    )


def _run_seed_batch(
    seeds: list[int],
    *,
    ascension: int,
    max_steps: int,
    backend: str,
    jobs: int,
    seed_timeout: int,
    progress_every: int,
    progress_prefix: str,
    source_random_seed: int | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    completed = 0
    failures = 0
    start = time.time()

    def _handle(result: dict[str, Any]) -> None:
        nonlocal completed, failures
        _attach_failure_metadata(result, backend=backend, source_random_seed=source_random_seed)
        results.append(result)
        completed += 1
        if not result.get("match"):
            failures += 1
        if progress_every > 0 and (completed % progress_every == 0 or completed == len(seeds)):
            _emit_progress(progress_prefix, completed, len(seeds), failures, start)

    if jobs <= 1:
        for seed in seeds:
            _handle(_compare_seed_with_timeout(seed, ascension, max_steps, backend, seed_timeout))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_map = {
                executor.submit(_compare_seed_with_timeout, seed, ascension, max_steps, backend, seed_timeout): seed
                for seed in seeds
            }
            for future in as_completed(future_map):
                _handle(future.result())

    return results


def _summarize_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [row for row in rows if not row.get("match", False)]
    category_counts = Counter(row.get("category", "unknown") for row in failed)
    reason_counts = Counter(row.get("reason", "unknown") for row in failed)
    summary = {
        "count": len(rows),
        "failed": len(failed),
        "category_counts": dict(sorted(category_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    summary.update(_summarize_clusters(rows))
    return summary


def _seed_source_index(sources: list[dict[str, Any]]) -> dict[int, int]:
    source_by_seed: dict[int, int] = {}
    for source in sources:
        baseline_seed = source.get("random_seed")
        checkpoint = source.get("checkpoint")
        if baseline_seed is None or not checkpoint:
            continue
        checkpoint_path = Path(str(checkpoint))
        if not checkpoint_path.exists():
            continue
        for row in _load_jsonl(checkpoint_path):
            if row.get("match", False):
                continue
            try:
                seed = int(row["seed"])
            except (KeyError, TypeError, ValueError):
                continue
            source_by_seed.setdefault(seed, int(baseline_seed))
    return source_by_seed


def _dependency_cli_message(exc: BaseException) -> str | None:
    text = str(exc)
    if "slaythespire is required for lightspeed/native comparison workflows" in text:
        return (
            "slaythespire is required for maintain_alignment_failure_corpus.py because it "
            "replays lightspeed/native comparison batches. Install the lightspeed Python "
            "package/build, or use native-only entrypoints such as run_native_run.py, "
            "run_native_sim.py, or export_model_run_checklist.py."
        )
    return None


def _persist_snapshot(
    *,
    corpus_path: Path,
    state_path: Path,
    corpus_by_seed: dict[int, dict[str, Any]],
    backend: str,
    ascension: int,
    max_steps: int,
    count_per_baseline: int,
    target_failures: int,
    refill_threshold: int,
    next_random_seed: int,
    sources: list[dict[str, Any]],
) -> None:
    final_rows = [corpus_by_seed[seed] for seed in sorted(corpus_by_seed)]
    _write_jsonl(corpus_path, final_rows)
    _write_state(
        state_path,
        {
            "backend": backend,
            "ascension": ascension,
            "max_steps": max_steps,
            "count_per_baseline": count_per_baseline,
            "target_failures": target_failures,
            "refill_threshold": refill_threshold,
            "next_random_seed": next_random_seed,
            "sources": sources,
            "corpus_path": str(corpus_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Maintain a rolling corpus of alignment failures across changing random_seed baselines (defaults to backend v3).")
    parser.add_argument("--backend", choices=["v1", "v2", "v3"], default="v3", help="Native backend to use; defaults to v3, with v2 kept for comparison.")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--count-per-baseline", type=int, default=1000)
    parser.add_argument("--start-random-seed", type=int, default=2)
    parser.add_argument("--target-failures", type=int, default=200)
    parser.add_argument("--refill-threshold", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed-timeout", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--baseline-cache-dir", type=Path, required=True)
    parser.add_argument("--skip-revalidate", action="store_true")
    args = parser.parse_args()

    try:
        state = _load_state(args.state_path)
        next_random_seed = int(state.get("next_random_seed", args.start_random_seed))
        sources: list[dict[str, Any]] = list(state.get("sources", []))
        source_by_seed = _seed_source_index(sources)

        corpus_rows = _load_jsonl(args.corpus_path)
        corpus_by_seed: dict[int, dict[str, Any]] = {}
        for row in corpus_rows:
            if row.get("match", False):
                continue
            seed = int(row["seed"])
            source_random_seed = row.get("source_random_seed", source_by_seed.get(seed))
            _attach_failure_metadata(row, backend=args.backend, source_random_seed=source_random_seed)
            corpus_by_seed[seed] = row

        if corpus_by_seed and not args.skip_revalidate:
            previous_corpus_by_seed = dict(corpus_by_seed)
            revalidated = _run_seed_batch(
                list(sorted(corpus_by_seed)),
                ascension=args.ascension,
                max_steps=args.max_steps,
                backend=args.backend,
                jobs=args.jobs,
                seed_timeout=args.seed_timeout,
                progress_every=args.progress_every,
                progress_prefix="revalidate",
            )
            corpus_by_seed = {}
            for row in revalidated:
                if row.get("match", False):
                    continue
                seed = int(row["seed"])
                old_row = previous_corpus_by_seed.get(seed, {})
                source_random_seed = old_row.get("source_random_seed", source_by_seed.get(seed))
                _attach_failure_metadata(row, backend=args.backend, source_random_seed=source_random_seed)
                corpus_by_seed[seed] = row
            _persist_snapshot(
                corpus_path=args.corpus_path,
                state_path=args.state_path,
                corpus_by_seed=corpus_by_seed,
                backend=args.backend,
                ascension=args.ascension,
                max_steps=args.max_steps,
                count_per_baseline=args.count_per_baseline,
                target_failures=args.target_failures,
                refill_threshold=args.refill_threshold,
                next_random_seed=next_random_seed,
                sources=sources,
            )

        while len(corpus_by_seed) < args.target_failures or len(corpus_by_seed) < args.refill_threshold:
            baseline_seed = next_random_seed
            next_random_seed += 1
            seeds = _seed_list(baseline_seed, args.count_per_baseline)
            baseline_path = args.baseline_cache_dir / f"baseline_seed{baseline_seed}_{args.count_per_baseline}x{args.max_steps}.jsonl"
            if baseline_path.exists():
                baseline_results = _load_jsonl(baseline_path)
                for row in baseline_results:
                    _attach_failure_metadata(row, backend=args.backend, source_random_seed=baseline_seed)
                print(f"[resume-baseline] random_seed={baseline_seed} loaded={len(baseline_results)} from={baseline_path}", flush=True)
            else:
                baseline_results = _run_seed_batch(
                    seeds,
                    ascension=args.ascension,
                    max_steps=args.max_steps,
                    backend=args.backend,
                    jobs=args.jobs,
                    seed_timeout=args.seed_timeout,
                    progress_every=args.progress_every,
                    progress_prefix=f"baseline:{baseline_seed}",
                    source_random_seed=baseline_seed,
                )
                _write_jsonl(baseline_path, baseline_results)
            summary = _summarize_failures(baseline_results)
            sources.append(
                {
                    "random_seed": baseline_seed,
                    "count": args.count_per_baseline,
                    "max_steps": args.max_steps,
                    "failed": summary["failed"],
                    "category_counts": summary["category_counts"],
                    "checkpoint": str(baseline_path),
                }
            )
            for row in baseline_results:
                if row.get("match", False):
                    continue
                _attach_failure_metadata(row, backend=args.backend, source_random_seed=baseline_seed)
                corpus_by_seed.setdefault(int(row["seed"]), row)
            _persist_snapshot(
                corpus_path=args.corpus_path,
                state_path=args.state_path,
                corpus_by_seed=corpus_by_seed,
                backend=args.backend,
                ascension=args.ascension,
                max_steps=args.max_steps,
                count_per_baseline=args.count_per_baseline,
                target_failures=args.target_failures,
                refill_threshold=args.refill_threshold,
                next_random_seed=next_random_seed,
                sources=sources,
            )
            if len(corpus_by_seed) >= args.target_failures:
                break

        final_rows = [corpus_by_seed[seed] for seed in sorted(corpus_by_seed)]
        _persist_snapshot(
            corpus_path=args.corpus_path,
            state_path=args.state_path,
            corpus_by_seed=corpus_by_seed,
            backend=args.backend,
            ascension=args.ascension,
            max_steps=args.max_steps,
            count_per_baseline=args.count_per_baseline,
            target_failures=args.target_failures,
            refill_threshold=args.refill_threshold,
            next_random_seed=next_random_seed,
            sources=sources,
        )

        summary = _summarize_failures(final_rows)
        summary["corpus_size"] = len(final_rows)
        summary["next_random_seed"] = next_random_seed
        summary["sources_used"] = [source["random_seed"] for source in sources]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except ModuleNotFoundError as exc:
        message = _dependency_cli_message(exc)
        if message is None:
            raise
        print(message, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
