# Codex Assessment: V2 Exact Recommendations

## Decision

I should not claim full confidence that the next implementation will reach global row-level `remaining_floor MAE < 3`.

The external answer itself says `<3` should not be treated as an immediate engineering promise. It recommends grouped-oracle and KNN-oracle diagnostics first. That is the correct gate: if oracle lower bounds under current observable state are still far above 3 on floors `0-34`, then no amount of ordinary MLP tuning can honestly promise `<3` from single-rollout terminal labels.

## What Is Strong Enough To Implement Now

The answer gives enough detail to implement a V2 diagnostic and ablation pipeline:

```text
1. grouped median oracle lower bounds
2. same-phase/floor KNN oracle lower bounds
3. no-seed/no-raw-RNG feature ablations
4. before + chosen_after training samples
5. per-record total weight preservation with before:after = 0.40:0.60
6. run-normalized phase/floor/source weighting
7. auxiliary-only survival/final-bin heads
8. post-hoc floor/phase/sample_kind additive calibrator
9. hard branch validation before any runtime value override
```

These are valid next steps even if `<3` is not guaranteed.

## Why I Do Not Trust Immediate MAE < 3

Current best local value model:

```text
model                              value_current2336_5k_residual_floor_mlp_v1
best_validation_remaining_mae       6.7792459098474875
floor_mean_baseline_mae             7.821386447080963
train rows                          1699185
validation rows                     192591
state feature dim                   2336
```

The model still has large mid-run underprediction:

```text
floor 20-24: pred_mean=10.8846 true_mean=13.3877
floor 25-29: pred_mean=8.5535  true_mean=11.0948
floor 30-34: pred_mean=6.2936  true_mean=8.4192
```

The best result is only about `1.04` MAE better than the floor-mean baseline. Reaching `<3` requires another `3.78+` MAE reduction. Without proving that the conditional noise lower bound is below 3, this is not a defensible promise.

## Implementation Direction If Proceeding

The next code work should be ordered as follows:

1. Implement `analyze_run_value_dataset.py`.
2. Add diagnostics for grouped-oracle MAE, KNN-oracle MAE, seed/run/floor/phase/source balance, and before-vs-after distribution.
3. Extend `train_run_value_model.py` cache records to include `sample_kind`, `source`, and optional chosen `state_after`.
4. Add row-weight modes for run-normalized phase/floor/source weighting.
5. Add no-seed/no-raw-RNG encoder mode before retraining.
6. Reproduce current best on A0, then run A1/A2/A3 ablations.
7. Only after diagnostics show meaningful headroom, add the top-12 explicit path/readiness/resource features.

## Remaining Technical Uncertainty

The V2 answer is specific, but several implementation details still need to be nailed down before I would run a large training sweep:

```text
1. How to define combat_room_index / reward_screen_id robustly from current records.
2. Whether source strings are reliable enough for forced_single and phase caps.
3. Whether no-seed/no-RNG should change STATE_FEATURE_DIM or zero out fields behind a feature flag.
4. Whether coarse survival/final bins should be added to the existing dense-bin trainer or implemented as separate explicit thresholds/classes.
5. Whether after-state validation should use only chosen after-states or also branch after-states from shadow logs.
6. What exact KNN approximation is acceptable at 1.9M rows without creating a slow diagnostic bottleneck.
```

## Practical Acceptance Criteria

The next implementation is worthwhile if it achieves one of these:

```text
global row-level MAE <= 6.50
seed-balanced MAE improves by >= 0.30
floor 20-34 calibration bias drops below 1.0 floor
after-state validation MAE is not worse than before-state MAE by >0.10
hard branch accepted true_delta is positive with lower CI >= 0
```

If none of these move, the bottleneck is probably not the training view alone.

