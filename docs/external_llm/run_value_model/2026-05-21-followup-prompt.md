# Follow-Up Prompt For External LLM: Run Value Model V2

I received your previous answer and compared it against the current local implementation. Please revise your recommendations with the following concrete facts in mind.

## Current Best Result

The current best value model is:

```text
model: value_current2336_5k_residual_floor_mlp_v1
state_feature_dim: 2336
train rows: 1,699,185
validation rows: 192,591
validation split: seed % 10 == 0
best_validation_remaining_mae: 6.7792459098474875
floor-mean validation baseline: 7.821386447080963
best epoch: 10
```

The current rollout data has about 5k seeds/runs:

```text
mean_floor = 29.8822
win_count = 289 / 5000
death_count = 4678 / 5000
truncated_count = 32
error_count = 1
```

Phase distribution is very imbalanced:

```text
COMBAT      1,198,427
CARD_REWARD   331,085
MAP           144,200
CARD_SELECT    93,245
EVENT          67,078
SHOP           39,643
CAMPFIRE       30,184
TREASURE       21,331
NEOW            9,397
BOSS_RELIC      5,218
```

Current value calibration has a strong mid-run underprediction:

```text
floor 20-24: pred_mean=10.8846 true_mean=13.3877
floor 25-29: pred_mean=8.5535  true_mean=11.0948
floor 30-34: pred_mean=6.2936  true_mean=8.4192
```

## Important Current Implementation Details

The encoder already contains more than a raw hash bag:

```text
STATE_FEATURE_DIM = 2336
numeric summary dim = 224
deck/readiness summary via _deck_readiness()
choice numeric and choice hash bag
future card/relic/boss relic pool hash bags
map_state aggregate features
map edge/reachable map hash bags
rng_trace and rng_state features
hand/draw/discard/exhaust zone bags
player/monster power bags
```

The model code already supports:

```text
residual_floor_baseline
survival_bins
final_floor_bins
late_fusion architecture
seed-balanced row weighting
```

Previous survival/final-floor experiments did not beat the scalar residual model:

```text
value_maprng_v3_2k_survival              best MAE 7.3899
value_maprng_v3_2k_finalfloor_cls        best MAE 7.2492
value_maprng_v3_2k_survival_finalcls     best MAE 7.2115
value_current2336_5k_residual_floor_mlp  best MAE 6.7792
```

Current training uses `state_before` only. The trajectory records do contain `state_after`, and runtime value rerank scores candidate after-states. So there is likely a training/runtime distribution mismatch.

The first value-rerank pilot was unsafe:

```text
baseline mean_floor       = 32.07
shadow mean_floor         = 32.07
value rerank mean_floor   = 28.70
policy blended mean_floor = 29.72
```

## What I Need From You Now

Please do not give another broad plan. I need exact technical recommendations for the next local iteration.

1. Given that survival/final-floor heads already underperformed as primary readouts, should they be auxiliary-only? If yes, what exact loss weights and readout should be used?
2. How should `state_before` and chosen `state_after` samples be mixed in training? Should they share the same terminal target? How do we avoid double-counting long runs or over-weighting combat microstates?
3. What lower-bound diagnostics should I run to determine whether global row-level MAE `<3` is impossible with single-rollout terminal labels? Please specify grouping keys and interpretation.
4. Since shallow path/deck/RNG/map features already exist, which 8-12 explicit features have the highest marginal value now? Please avoid broad lists; prioritize what to implement first.
5. Should seed hash / RNG hash features be removed, retained, or ablated? How should validation be split to test whether they cause pseudo-generalization?
6. What exact conservative value-gate should be fit from hard branch validation? Please specify threshold variables, initial values, and accept/reject criteria.
7. If I train an ensemble for uncertainty, is 3 models enough or is 5 materially better? How should ensemble std or quantile interval enter the gate?
8. Would you recommend a phase-specific head now, or should phase balancing and after-state training come first?

Please answer in terms of this project's constraints:

```text
Ironclad, Ascension 0
run-level outcome target: final_floor / remaining_floor
no new hand-written reward function
value model should eventually support safe after-state rerank / q_env improvement
CPU rollout data is expensive; GPU training is cheap
```

