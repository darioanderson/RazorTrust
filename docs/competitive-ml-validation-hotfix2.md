# Competitive ML Validation Hotfix 2a

This patch hardens the public external-validity benchmark only. It does not modify the serving
champion, enforcement runtime, Razorpay credentials, OPA, database state, model artifacts, SEALED
TEST, stress data, or automatic money movement.

## Validation changes

- Replaces the old 70/15/15 layout with four independent roles:
  - 55% model training
  - 15% calibration
  - 15% policy / threshold selection
  - 15% untouched test
- Splits the calibration partition again into disjoint calibrator-fit and calibrator-selection halves.
- Chooses the calibration method on the selection half, then refits only the chosen method on the
  complete calibration partition. Policy and test remain unseen.
- Fits missing-value medians from TRAIN only and applies those medians to later partitions.
- Makes chronology claims conditional on an actual usable time column. OpenML 1597 does not expose
  `Time` through its default frame, so the default benchmark uses a deterministic stratified random
  split and says so explicitly.
- Adds confusion counts, log loss, false positives per 10,000 legitimate transactions, false
  negatives per 100 frauds, and predicted-positive/review burden.
- Does not invent a monetary false-positive cost. Public data has no defensible settlement-review
  unit cost, so the report labels the burden metric as non-monetary.
- Adds source content hashing and more explicit safety/provenance metadata.

## Claims

Every result remains labelled `PUBLIC HELD-OUT BENCHMARK`. This dataset is transaction-fraud data,
not Razorpay settlement-hold ground truth and not Razorpay production performance.

The frozen serving champion remains `xgb-if-settlement@2`; HUMAN_ONLY and automatic RELEASE safety
remain unchanged.
