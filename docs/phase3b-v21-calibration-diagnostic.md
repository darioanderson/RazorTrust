# Phase 3B V2.1 — Calibration, policy and embedding-drift diagnostic

This is an additive, research-only diagnostic. It does **not** replace or rewrite the
Phase 3B V2 decision `PHASE3B_V2_REJECT_HETERO_GRAPHSAGE_COMPLEXITY`.

## Purpose

V2 showed better ranking from heterogeneous GraphSAGE fusion but a worse selected
operating point. V2.1 separates the remaining stages chronologically:

1. representation training — fit heterogeneous normalization and GraphSAGE;
2. classifier training — fit tabular, graph-statistics, and fusion XGBoost models;
3. calibration — fit a sigmoid score calibrator only;
4. policy — choose diagnostic thresholds only;
5. test — evaluate frozen choices only.

The Fraud Detection Handbook dataset is sampled case-control for this pilot. Therefore
calibrated values are **sample calibration diagnostics**, not population fraud
probabilities and not Razorpay settlement-hold probabilities.

## Calibration

V2.1 uses only sigmoid/Platt-style calibration on the independent calibration partition.
It intentionally does not use isotonic regression because the bounded diagnostic
calibration sample is small and a flexible isotonic map can overfit small samples.

Reported probability diagnostics include PR-AUC, Brier loss, log loss, quantile
reliability bins, and positive/negative score summaries.

## Policy diagnostics

POLICY selects four operating points on a fixed threshold grid:

- the original V2 max-F1 rule with high-threshold tie breaking;
- max-F1 with low-threshold tie breaking;
- a recall-floor diagnostic;
- a sample-FPR-cap diagnostic.

TEST applies those frozen POLICY thresholds. A separate TEST threshold sweep is written
for post-hoc diagnosis only and is explicitly marked selection-ineligible.

## Embedding drift

The report measures target-transaction embedding drift across classifier training,
calibration, policy, and test partitions using mean-vector distance, standardized mean
shift, and embedding L2-norm distributions.

## Safety invariants

- canonical 13-feature production contract is unchanged;
- `xgb-if-settlement@2` remains champion;
- `human-only@1` remains the enforcement runtime;
- automatic RELEASE remains disabled;
- no SEALED TEST or stress data is accessed;
- no automatic promotion, serving change, or money movement is enabled.
