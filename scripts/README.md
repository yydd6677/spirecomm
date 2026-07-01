# Script Entry Points

Root-level Python and shell entrypoints are grouped by workflow:

- `native/`: native simulator runs, real-game validation, recorded replay, strict trace alignment, and model integration checks.
- `v3_combat/`: v3 combat teacher data, rollout evaluation, sweep launchers, transformer/PPO training, shard relabeling, and diagnostics.
- `run_value/`: run-value trajectory collection, value model training/evaluation, non-combat branch labeling, and action-policy iteration.
- `model_training/`: general card reward, card target, run-choice, upgrade-prior, and legacy combat model training.

Run scripts from the repository root with `python3`, for example:

```bash
python3 scripts/native/run_native_run.py --help
python3 scripts/v3_combat/evaluate_v3_rollout_batch.py --help
python3 scripts/run_value/train_run_value_model.py --help
python3 scripts/model_training/train_combat_model.py --help
```
