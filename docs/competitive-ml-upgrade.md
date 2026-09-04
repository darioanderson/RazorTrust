# RazorTrust competitive ML validation upgrade

## Why this exists

RazorTrust's settlement-hold verifier is intentionally conservative and currently uses synthetic research data for its domain-specific model work. Buildathon judges can reasonably ask for a second, externally recognizable held-out benchmark with explicit precision and recall.

This upgrade adds a public-fraud validation harness without changing the serving model, `HUMAN_ONLY`, thresholds, Razorpay state, SEALED research data, or production authority.

## Public benchmark

The runner supports:

- a local CSV such as the ULB/Worldline `creditcard.csv` dataset; or
- OpenML dataset id `1597` through scikit-learn's `fetch_openml`.

The evaluation contract is fixed:

1. sort chronologically when a `Time` column exists;
2. first 70% = model training;
3. next 15% = probability calibration and operating-point selection;
4. final 15% = untouched held-out test;
5. test rows are never used for training, calibration choice, or threshold selection.

The harness compares uncalibrated, sigmoid, and (when sufficiently large) isotonic calibration on the calibration partition only. It then freezes two operating points:

- `HIGH_PRECISION`: maximize recall under calibration FPR <= 0.15%;
- `BALANCED`: maximize F1 under calibration FPR <= 1%.

Held-out reporting includes:

- PR-AUC;
- ROC-AUC;
- precision;
- recall;
- F1;
- false-positive rate;
- Brier score;
- stratified bootstrap 95% confidence intervals for PR-AUC, precision and recall.

## Claim discipline

Public-benchmark metrics are **external benchmark results**, not Razorpay production results and not settlement-hold production performance. Domain-specific RazorTrust metrics remain synthetic/research-only until real labelled settlement data exists.

## Why not force GraphSAGE/LSTM

Phase 3 graph statistics and GraphSAGE failed the locked ablation gate. Phase 4 temporal features also failed the primary gate and are undergoing a diagnostic confirmation. Those negative results are retained rather than hidden. Model complexity is accepted only when it earns measurable value.
