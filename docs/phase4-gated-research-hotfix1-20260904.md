# RazorTrust Phase 4 gated research hotfix 1

This hotfix adds the full Phase-4 development gate without changing production authority.

Evaluation order:

1. Frozen 13-feature tabular baseline.
2. 13 features plus point-in-time engineered temporal features.
3. LSTM and small Transformer challengers only if the temporal ablation gate passes.

Fixed development gate:
- PR-AUC must improve by at least 0.01.
- Expected cost must improve.
- Risk recall must not decrease.
- Fixed costs: false RELEASE = 100, false HOLD = 25.

Safety:
- merchant-grouped development folds;
- only transactions with timestamp < hold.triggered_at;
- SEALED TEST untouched;
- stress set untouched;
- production_action_eligible = false;
- serving_change_authorized = false;
- automatic promotion = false;
- champion remains xgb-if-settlement@2.
