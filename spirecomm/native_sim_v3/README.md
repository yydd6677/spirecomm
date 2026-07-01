# native_sim_v3

`native_sim_v3` is the source-driven Slay the Spire simulator backend in this
repo. Its gameplay rules are derived from the local decompiled game sources and
validated against real-game replay through Communication Mod.

For the repo's current completion claim and replay evidence, see
[`STATUS.md`](./STATUS.md).

That status file now also records the final close-out for the current Ironclad
A0 claim: `v3` is the primary native backend, and the remaining work is mostly
tooling hardening or broader scope expansion.

## Current Status

`native_sim_v3` is now the primary native backend for Ironclad A0 work in this
repo. The simulator mainline is in place, and the remaining validation work is
mainly about replay/tooling robustness on very long real-game sessions rather
than new simulator rule gaps.

In practice this means:

- run/combat/event/shop/reward/map/final-act logic is wired up
- source-driven constants and progression rules are in place
- real-game replay is green across multiple curated and random seeds
- the current replay bridges live in tooling, not in simulator rule logic
- primary repo CLI entrypoints now default to `v3`
- explicit `v2` fallback still exists when comparison is needed
- native-only `v3` entrypoints also run cleanly in lighter environments without
  `torch`, while comparison/model-validation tools now fail with short
  dependency-specific messages

## Real-Game Replay Coverage

The following validations have been replayed green against the real game:

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

Random seed corpus replayed green at `128-step`:

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

## What Is Still Not "Done Done"

The main remaining uncertainty is not core simulator semantics. It is replay
session robustness under very long real-game runs:

- very long sessions can still be bounded by replay session timeout
- some replay success still relies on narrow stale-state bridge handling in
  `spirecomm.ai.recorded_run_replay`
- multiplayer-scale coverage across many more characters/ascension settings has
  not been claimed yet

So the current claim is:

- `native_sim_v3` is ready for serious use and validation work
- it has strong real-game replay evidence
- the remaining polish is mostly replay harness stability, not missing gameplay
  subsystems
