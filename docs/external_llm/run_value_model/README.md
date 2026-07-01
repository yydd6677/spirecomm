# Run Value Model External LLM Discussions

This folder keeps the run-level value-model prompts, external LLM answers, and Codex follow-up reviews in one place.

## 2026-05-21 Round

- `2026-05-21-context-prompt.md`: the project context and question sent to the external LLM.
- `2026-05-21-external-llm-answer.md`: the returned plan from the external LLM.
- `2026-05-21-codex-review-and-next-plan.md`: repo-grounded review of the answer and the proposed next implementation direction.
- `2026-05-21-followup-prompt.md`: a corrected follow-up prompt that highlights local implementation details the external LLM may have missed.
- `2026-05-21-v2-exact-recommendations.md`: the external LLM's follow-up answer with exact V2 recommendations.
- `2026-05-21-v2-codex-assessment.md`: Codex assessment of whether the V2 answer is enough to chase MAE `<3`.
- `2026-05-21-v2-followup-prompt.md`: a narrower follow-up prompt focused on the remaining blockers before implementation.

## Current Local Facts

- Current best run-value model: `value_current2336_5k_residual_floor_mlp_v1`.
- Best validation remaining-floor MAE: `6.7792459098474875`.
- Floor-mean baseline MAE on the same validation split: `7.821386447080963`.
- Training cache rows: `1,891,776`; validation rows: `192,591`.
- Current state feature dimension: `2336`.
- Current train split is `seed % 10 == 0` validation, not a contiguous held-out seed range.
- Current training uses `state_before` only; chosen `state_after` is recorded but not part of value training.
- Current engineering conclusion: do not claim MAE `<3` is guaranteed. First run lower-bound diagnostics and after-state/weighting ablations.
