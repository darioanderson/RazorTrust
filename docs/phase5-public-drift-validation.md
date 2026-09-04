# Phase 5 — Public Drift Validation V1

Status: research-only monitoring validation.

This phase does not replace the RazorTrust production model, change the exact
13-feature production contract, enable automatic retraining, or authorize money
movement.

## Purpose

The existing monitoring foundation already supports:

- batch two-sample KS + Jensen-Shannon distance,
- Benjamini-Hochberg multiple-testing correction,
- effect-size gating,
- Evidently drift reports,
- online ADWIN and KSWIN change detection.

The original smoke evaluation shifted every synthetic feature by a fixed amount.
That proves obvious-change detection but does not measure natural chronological
drift, score drift, delayed-label error drift, or controlled partial-feature
detection power.

V1 therefore uses the local ULB/OpenML 1597 credit-card dataset and a
chronological experiment:

- 55% model train
- 15% calibration
- 15% policy/reference
- 15% future monitor, split into three chronological windows

The public dataset is transaction-fraud data, not Razorpay settlement-hold
ground truth.

## Streams and meanings

Feature batch drift is label-free. Score drift uses the calibrated fraud
probability stream and is label-free. Delayed error drift uses a 0/1
classification-error stream and is only available after labels arrive.

Drift never means fraud. Drift never triggers automatic retraining, automatic
promotion, RELEASE, or money movement.

## Controlled injection

The default injection features are V1, V2, and V3. They are anonymized PCA
components; injected shifts are detector-power probes only. Strengths are
specified in units of each feature's policy-reference standard deviation.

## Severity

GREEN: no material distribution or performance signal.

AMBER: a distribution signal or material performance degradation.

RED: both a distribution signal and material performance degradation.

This severity is research-only and must not be treated as a production
governance rule without a separate approval process.
