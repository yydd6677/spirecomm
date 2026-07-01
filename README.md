# spirecomm
A package for using Communication Mod with Slay the Spire, plus a simple AI

## Communication Mod

Communication Mod is a mod that allows communication between Slay the Spire and an external process. It can be found here:

https://github.com/ForgottenArbiter/CommunicationMod

The spirecomm package facilitates communicating with Slay the Spire through Communication Mod and accessing the state of the game.

## Requirements:

- Python 3.5+
- kivy, only for the example GUI for Communication Mod, found in utilities

## Running the AI:

To run a simple Slay the Spire AI, configure Communication Mod to run main.py

## Hybrid / data collection mode

The project now supports a hybrid mode where combat can be delegated to a pluggable policy while non-combat
decisions continue to use the existing rule-based agent. This is intended as the bridge toward behavior cloning
and later reinforcement learning.

Useful environment variables:

- `SPIRECOMM_TRAJECTORY_DIR`: if set, writes one JSONL trajectory file per run
- `SPIRECOMM_RECORD_MODE`: `combat` (default) or `all`
- `SPIRECOMM_POLICY_CLASS`: optional import path in the form `package.module:ClassName`
- `SPIRECOMM_STARTER_CLASS`: optional starting class name, defaults to `THE_SILENT`

If no environment variables are set, the old `SimpleAgent` behavior is preserved.

When `SPIRECOMM_RECORD_MODE=combat`, the recorder writes one JSONL file per combat.

Each combat trajectory file contains:

- one `meta` record with run metadata and the initial combat state
- one `transition` record per combat action, including `state_before`, `action`, `state_after`, and an explicit `delta`
- one `summary` record with the post-combat state and visible rewards

The `delta` payload is intended to preserve reward-relevant changes explicitly, such as:

- HP / max HP changes
- gold changes
- relic / potion / deck changes
- monster HP and kill count changes
- pile-size changes
- whether combat ended and whether a combat reward screen appeared

The intended workflow is:

1. Run the current hybrid/rule bot while recording trajectories
2. Train a combat policy on the recorded state/action pairs
3. Expose that policy via `SPIRECOMM_POLICY_CLASS`
4. Keep out-of-combat decisions rule-based until the combat model is stable

## First-pass reinforcement learning workflow

The project now includes a deliberately small combat RL scaffold aimed at getting a new ML practitioner moving quickly.

Design choices in the first version:

- only combat decisions are learned
- the action space is intentionally small: `end turn` or `play hand slot 0-9`
- targeted cards use a separate target head over monster slots `0-6`
- out-of-combat choices still use the original rule-based logic
- reward shaping favors winning fights, dealing damage efficiently, preserving HP, and reducing incoming damage

The training entrypoint is:

`python3 scripts/model_training/train_combat_model.py --trajectory-dir trajectories --output models/combat_bc.pt --mode bc`

Recommended first run:

1. Activate the dedicated ML environment
2. Train with `--mode bc` on the rule bot trajectories
3. Run that checkpoint in-game through `CheckpointCombatPolicy`
4. Record new trajectories from the learned policy
5. Fine-tune with `--mode reinforce` on those self-play trajectories

Runtime environment variables for the learned policy:

- `SPIRECOMM_POLICY_CLASS=spirecomm.ai.learned_policy:CheckpointCombatPolicy`
- `SPIRECOMM_MODEL_PATH=/absolute/path/to/model.pt`
- `SPIRECOMM_SAMPLE_ACTIONS=1` if you want stochastic exploration instead of greedy play

For the first iteration, this is enough to produce a model that can auto-battle and then keep improving from newly recorded results.

## Installing spirecomm:

Run `python3 setup.py install` from the distribution root directory

## native_sim_v3

The repo also includes a source-driven simulator backend at
[`spirecomm/native_sim_v3`](./spirecomm/native_sim_v3/README.md).

Current completion details and replay evidence live in:

- [`spirecomm/native_sim_v3/STATUS.md`](./spirecomm/native_sim_v3/STATUS.md)

Current status:

- simulator mainline is in place for Ironclad A0 validation work
- real-game replay is green across multiple curated and random seeds
- primary repo CLI entrypoints now default to `v3`
- native-only `v3` entrypoints run in lighter environments without `torch` or lightspeed-specific Python packages
- remaining work is mainly replay/tooling robustness on very long sessions

Recommended entrypoints:

- `python3 scripts/native/run_native_run.py`
- `python3 scripts/native/run_native_sim.py`
- `python3 scripts/native/validate_real_game_first.py --native-backend v3 ...`

Comparison/model-validation helpers that still depend on `torch` or
`slaythespire` now fail with short actionable messages instead of raw Python
stack traces.
