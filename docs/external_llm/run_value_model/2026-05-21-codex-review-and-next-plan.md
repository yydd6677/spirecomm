# Codex Review And Next Plan: Run Value Model

## What The External Answer Gets Right

The answer's main warning is correct: the current `remaining_floor` MAE should not be treated as a small hyperparameter problem. The current best local model is `value_current2336_5k_residual_floor_mlp_v1` with:

```text
best_validation_remaining_mae = 6.7792459098474875
floor_mean_baseline_mae       = 7.821386447080963
validation_rows               = 192591
train_rows                    = 1699185
total_rows                    = 1891776
state_feature_dim             = 2336
```

The validation calibration already shows the key failure mode: early and mid floors are noisy, and floors `20-34` are systematically underestimated. Examples from the best model:

```text
floor 20-24: pred_mean=10.8846 true_mean=13.3877
floor 25-29: pred_mean=8.5535  true_mean=11.0948
floor 30-34: pred_mean=6.2936  true_mean=8.4192
```

The answer is also right that row-level `V^pi(s)` and action reranking `Q(s,a)` are different targets. The first value-rerank pilot already showed this mismatch:

```text
baseline mean_floor      = 32.07
shadow mean_floor        = 32.07
value rerank mean_floor  = 28.70
policy blended mean_floor= 29.72
```

So the next useful system is not an unconstrained value takeover. It must start with diagnostics, calibration, uncertainty, and branch validation.

## Corrections To The External Answer

Some suggestions are already partly implemented locally, so they should not be treated as fresh silver bullets.

Current `spirecomm.ai.run_value` already includes:

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
survival_bins support
final_floor_bins support
late_fusion architecture support
residual_floor_baseline support
seed-balanced row weighting support
```

Previous experiments also tried several versions of survival/final-floor heads, and they did not beat the current residual scalar MLP:

```text
value_maprng_v3_2k_survival              best MAE 7.3899
value_maprng_v3_2k_finalfloor_cls        best MAE 7.2492
value_maprng_v3_2k_survival_finalcls     best MAE 7.2115
value_current2336_5k_residual_floor_mlp  best MAE 6.7792
```

Therefore, the next version should not simply switch the runtime readout to survival expectation. If distributional heads are added, they should initially be auxiliary calibration and uncertainty heads while the scalar residual head remains the primary value readout.

The external answer also underplays a current implementation issue: value training uses only `state_before`, even though trajectory records contain `state_after`. Since runtime value rerank scores candidate after-states, this is a real target-distribution mismatch and should be fixed before more ambitious architecture work.

## Highest-Value Next Design

The next implementation should be `run_value_v2_diagnostics_and_weighted_scalar`, not a large new value Transformer.

### 1. Add A Diagnostic Dashboard First

Add a script/report that computes:

```text
row-level MAE
seed-balanced MAE
run-start MAE
floor bucket MAE and calibration
phase-specific MAE and calibration
row count by seed/floor/phase/source
COMBAT dominance and forced_single share
before-state vs chosen-after-state MAE
floor/context median lower-bound MAE
```

The lower-bound check is critical. If a simple grouped median predictor by `floor bucket + phase + hp bucket + act + boss` is still far above `3`, then chasing global `<3` on single-rollout terminal labels is not a valid short-term target.

### 2. Rebuild The Training View

Keep the existing 2336-dim encoder first, but change the training view:

```text
include state_before samples
include chosen state_after samples
tag sample_kind = before / chosen_after
remove or heavily downweight forced_single
cap rows per seed/floor/phase/source
add phase/floor/run-balanced sampling or explicit row weights
use a contiguous held-out seed range in addition to seed % 10 validation
```

This is lower risk than changing the model structure and directly attacks two known issues: COMBAT/long-run row dominance and after-state distribution mismatch.

### 3. Train Scalar Residual Baselines Before New Heads

Train these in order:

```text
A. reproduce value_current2336_5k_residual_floor_mlp_v1 on the new training view
B. residual MLP + run/floor/phase/source weighting
C. B + chosen_after rows
D. C + auxiliary final-floor/survival heads, scalar residual readout unchanged
```

Only if `C` beats `B` and `D` improves calibration without hurting MAE should distributional heads become part of the main value checkpoint.

### 4. Add Only High-Marginal Explicit Features

The current feature encoder already has shallow deck/map/RNG features. The next additions should be targeted and measurable:

```text
path DP:
  min floors to elite/shop/rest/boss
  forced elite within 3/5 floors
  reachable path elite/rest/shop counts
  safe vs greedy path summaries

encounter readiness:
  Nob / Lagavulin / Sentries readiness
  Act boss readiness
  Act 2 multi-enemy and burst-risk readiness

resource pressure:
  hp margin to next elite/boss
  effective Ironclad sustain after Burning Blood
  strong potion available for elite/boss
  potion slot pressure
  gold to next shop and removal affordability
```

These features should remain state covariates. They should not be manually added to action scores.

### 5. Delay Runtime Rerank Until Branch Validation Exists

Do not re-enable unconstrained `baseline_score + alpha * value_score`. The next safe sequence is:

```text
train calibrated scalar V
run shadow with candidate after-state value logging
select disagreement/high-gap/high-uncertainty roots
perform hard branch continuation on selected roots
fit a conservative gate in floor units
only then run paired rollout
```

Initial gate constraints should be strict:

```text
candidate must be in baseline top-2 or top-3
predicted floor gap >= 1.5 to 2.0 for combat
predicted floor gap >= 2.0 to 3.0 for card/shop/boss relic
uncertainty must be low
override rate capped at 1% to 5%
phase whitelist starts narrow
```

## Implementation Order

Recommended next concrete work:

1. Create `analyze_run_value_dataset.py` for lower-bound, phase/floor/source/seed balance, and before-vs-after diagnostics.
2. Extend `train_run_value_model.py` cache to support `state_after` samples and `sample_kind`.
3. Add phase/floor/source row weighting or a balanced sampler.
4. Re-train the residual scalar model on the new view and compare against `6.7792`.
5. Add quantile/calibration heads only after the scalar training view is stable.
6. Build hard branch validation from shadow logs before any runtime value rerank.

## Questions Still Worth Sending Back To The External LLM

The next prompt should not ask for another broad plan. It should ask for decisions on unresolved technical points:

```text
1. Given that survival/final-floor heads already underperformed, should they be auxiliary-only? What weights/readout would you use?
2. How exactly should state_before and chosen state_after be mixed without double-counting one run's terminal label?
3. What grouped-median lower-bound diagnostics are sufficient to decide whether global MAE <3 is impossible?
4. Which 8-12 explicit path/readiness/resource features have the highest marginal value beyond the current shallow features?
5. How should a conservative value gate be calibrated from hard branch validation?
6. Should seed/rng hash features be removed, retained, or split into ablations?
```

