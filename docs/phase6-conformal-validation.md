# Phase 6 — Conformal Uncertainty Validation V1

Status: research-only.

This phase does not modify the production model, production uncertainty module,
the exact 13-feature contract, serving policy, automatic RELEASE behavior, or
the canonical SEALED TEST.

## Why this validation exists

RazorTrust currently forms APS conformity scores from cross-fitted policy
probabilities. The final ModelBundle, however, uses a probability calibrator
refit on the entire development calibration partition.

That is a useful development technique for calibration and threshold selection,
but strict split conformal prediction expects the same fixed predictor to
produce probabilities on the conformalization set and on future test samples.

MAPIE also provides a separate CrossConformalClassifier for true cross-conformal
workflows. This V1 therefore does not rewrite production code. It compares:

1. strict fixed-predictor split conformal:
   a final frozen ModelBundle predicts an independent conformalization dataset
   and an independent research-test dataset;

2. the current legacy-style cross-fitted-probability approach as a diagnostic.

## Data isolation

Three independent synthetic generator seeds are used:

- development seed: model fitting, probability calibration, cost-policy selection;
- conformal seed: APS conformalization only;
- research-test seed: evaluation only.

The canonical SEALED TEST and stress set are never accessed.

## Metrics

The report includes marginal empirical coverage, Wilson 95% interval, average
set size, singleton and ambiguity rates, class-specific coverage diagnostics,
and conservative routing metrics.

Class-specific coverage is diagnostic only because standard APS targets
marginal rather than class-conditional coverage.

## Controlled shift

The fresh research-test feature matrix is shifted on selected canonical
features by fixed multiples of their conformal-reference standard deviations.
Coverage guarantees are explicitly not claimed after detected shift.

## Safety

RELEASE remains a system recommendation only. The active enforcement runtime
remains human-only@1. Ambiguous, empty, and full prediction sets must never
route to RELEASE.
