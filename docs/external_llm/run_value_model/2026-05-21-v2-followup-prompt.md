# Follow-Up Prompt: Remaining Blockers Before Implementing Run Value V2

Your V2 answer is much more actionable. I agree with the main conclusion that MAE `<3` needs lower-bound diagnostics first. I now need narrower implementation guidance on a few unresolved details.

## Current Concrete State

Current best model:

```text
value_current2336_5k_residual_floor_mlp_v1
best_validation_remaining_mae = 6.7792459098474875
floor_mean_baseline_mae = 7.821386447080963
rows = 1,891,776
validation rows = 192,591
state_feature_dim = 2336
```

Current records include:

```text
state_before
state_after
source
phase
floor
room_type
terminal final_floor
rerank info
branch_error
```

Current trainer uses only `state_before`.

## Questions

1. For combat row capping, if current records do not have a reliable `combat_room_index`, is `(seed, floor, phase, room_type, source)` a good fallback, or will that over-collapse distinct combats/reward screens?
2. For reward/card/shop/event screen capping, should the group key use `(seed, floor, phase, source)` or should it hash the visible `choice_list` / `screen_state` so repeated screens are grouped but distinct screens are not?
3. For no-seed/no-raw-RNG ablation, is it better to change the encoder dimension/schema or keep dimension fixed and zero out risky positions behind flags? The project has many cached tensors and checkpoints, so schema churn has cost.
4. You recommend coarse survival thresholds and final-floor bins. The current trainer supports dense `survival_bins` and `final_floor_bins` appended after the existing 8 outputs. Should I adapt this mechanism to explicit coarse thresholds/classes, or is dense 50-bin support acceptable if the auxiliary weight is tiny?
5. For the grouped median oracle, G4 says `quantized_encode_state_without_seed_rng_hashes`. At 1.9M rows, what exact dimensionality-reduction or hashing method would you use to avoid almost every group being unique?
6. For KNN oracle, exact KNN over 1.9M x 2336 may be slow. What approximation is acceptable for a diagnostic? Random projection to 64/128 dims? PCA? FAISS? Same floor/phase prefilter plus reservoir sample?
7. Should `state_after` samples be validated separately from `state_before` with separate metrics and calibration tables, or should they be mixed in validation using the same weights as training?
8. If A1 before+after improves after-state MAE but slightly worsens row-level global MAE, should it still be considered a better value model for rerank?

Please answer only these implementation details. Avoid broad restatement of the full plan.

