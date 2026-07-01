# native_sim_v3 status

## Scope

This status file tracks the current completion claim for `native_sim_v3` as it
exists in this repo, not an abstract future target.

The practical validation target so far is:

- Ironclad A0 run generation
- source-driven simulator rules
- real-game replay validation through Communication Mod

## Current Claim

`native_sim_v3` is effectively the primary native backend in this repo.

That claim is based on two things:

- primary repo CLI entrypoints now default to `v3`
- replay validation is green across multiple curated seeds and many random
  seeds, including several full current traces

## 2026-05-06 New-Model Alignment Checkpoint

The latest `models/v3_combat_scorer.pt` strict replay alignment was paused while
validating seeds `1..100` against the real game.

Confirmed in this pass:

- `seed=1` strict replay passed: `288/288`
- `seed=2` strict replay passed
- `seed=3` was advanced from the original floor 24 Gremlin Leader divergence to
  a later floor 30 combat divergence

Fixes applied before pausing:

- Gremlin Leader `RALLY` now chooses both summoned gremlin types before consuming
  spawned minion opening AI rolls.
- Confusion/Snecko random costs no longer overwrite card base cost/combat cost.
- Upgrading cards preserves non-base temporary `cost_for_turn`.
- Played and exhausted cards preserve temporary visible `cost_for_turn` in
  discard/exhaust piles.
- `Exhume` auto-returns the single valid non-Exhume exhausted card instead of
  opening `CARD_SELECT`.

Current stop point:

- report:
  `_cache/real_game_first/new_model_align_seed3_fix5_reports/seed_3_real_replay_report.json`
- trace:
  `_cache/real_game_first/new_model_align_seed3_fix5_traces/seed_3_trace.json`
- failure:
  `seed=3`, `post_state_mismatch`, `COMBAT`, `first_failure_step=492`,
  `steps_replayed=492/572`
- current action:
  play `Twin Strike` targeting monster `0`
- visible location:
  `floor=30`, `current_hp=52`, `room_type=MonsterRoom`

Continue from this report if v3 alignment is resumed.

## Release Gate

The practical release gate for the current repo state is satisfied for the
Ironclad A0 target:

- source-driven simulator mainline is in place
- primary repo CLI entrypoints now default to `v3`
- curated replay validation is green
- multiple random-seed replay samples are green
- multiple full current random traces are green

This does **not** mean every future validation target is complete. It means the
repo can now treat `v3` as the primary native backend for the currently claimed
scope.

## Strong Replay Evidence

Curated / named coverage already replayed green:

- `seed=20 / 135-step`
- `seed=1 / 32-step`
- `seed=1 / 64-step`
- `seed=1 / 128-step`
- `seed=1 / 256-step`
- `seed=1 / 512-step`
- `seed=12 / 256-step`
- `seed=7133506393411724536 / complete 88-step`
- `seed=2 / 128-step`
- `seed=3 / 128-step`
- `seed=8866187513371018371 / 128-step`

Random-seed corpus replayed green at `128-step`:

- `8194582523602576612`
- `17485029721327973432`
- `7283207964119141687`
- `890727360438182992`
- `15149836622520594227`
- `1736392818365009963`
- `10750541312280087032`
- `16781078052021535861`
- `3960482443532127989`
- `1585446675937841368`
- `7713914763314685786`

Random seeds whose full current traces have also been replayed green:

- `seed=17485029721327973432 / complete 133-step`
- `seed=10750541312280087032 / complete 181-step`
- `seed=8194582523602576612 / complete 131-step`
- `seed=7283207964119141687 / complete 116-step`
- `seed=890727360438182992 / complete 70-step`
- `seed=15149836622520594227 / complete 107-step`
- `seed=1736392818365009963 / complete 138-step`
- `seed=16781078052021535861 / complete 112-step`
- `seed=3960482443532127989 / complete 148-step`
- `seed=1585446675937841368 / complete 89-step`
- `seed=7713914763314685786 / complete 85-step`
- `seed=6027341539762311745 / complete 78-step`

## What Is Still Residual Risk

The remaining risk is mostly not new gameplay semantics. It is:

- replay harness robustness on very long sessions
- narrow stale-state bridge logic in `spirecomm.ai.recorded_run_replay`
- broader character/ascension coverage that has not yet been claimed

So the current interpretation should be:

- simulator mainline is in place
- `v3` is ready to be the primary native backend
- residual uncertainty lives mostly in validation tooling at longer horizons

## Current Recommendation

For normal repo use, prefer `v3`.

Use explicit `v2` only when you are:

- comparing regressions against the older backend
- bisecting a suspected replay/tooling issue
- validating legacy assumptions that were written specifically for `v2`

## Recommended Usage

The practical repo-default path is now:

- `python run_native_run.py`
- `python run_native_sim.py`
- `python export_model_run_checklist.py --seed <seed>`
- `python validate_real_game_first.py --native-backend v3 ...`

Explicit `v2` selection still exists for comparison and rollback:

- `python run_native_run.py --backend v2`
- `python run_native_sim.py --backend v2`
- `python export_model_run_checklist.py --backend v2 --seed <seed>`

## Lightweight Environment Notes

The current repo-default `v3` path is usable even in a lighter environment that
does not have every optional dependency installed.

These native-only entrypoints now start directly without `torch` or
lightspeed-specific Python modules:

- `python run_native_run.py`
- `python run_native_sim.py`
- `python export_model_run_checklist.py --seed <seed>`
- `python validate_real_game_first.py --mode native --native-backend v3 ...`

These comparison/model-validation entrypoints still require extra dependencies,
but they now fail with short actionable messages instead of raw stack traces:

- `python compare_native_to_lightspeed_run.py ...` requires `slaythespire`
- `python maintain_alignment_failure_corpus.py ...` requires `slaythespire`
- `python verify_model_integration.py ...` requires `torch` and, for
  lightspeed-backed checks, `slaythespire`

## Final Close-Out

For the currently claimed target, this `v3` rollout is complete:

- Ironclad A0 run generation is source-driven and in place
- real-game replay evidence is strong enough that the repo now treats `v3` as
  the primary native backend
- the remaining work is expansion and tooling hardening, not reopening the
  current gameplay-completeness claim

In practical terms:

- use `v3` by default for normal native simulator work in this repo
- keep explicit `v2` only for comparison, bisecting, or legacy parity checks
- treat future replay/tooling work as residual polish unless it exposes a new
  gameplay-semantic mismatch
  

## Repository Interpretation

If this file becomes outdated, update it together with:

- `spirecomm/native_sim_v3/README.md`
- root `README.md`
- any CLI default that changes the practical default backend claim
