# RazorTrust - Verified Results

## Public Benchmark

Dataset:

ULB / Worldline Credit Card Fraud Detection dataset

Rows:

284,807

Fraud-labelled rows:

492

Evaluation method:

Chronological 60 / 20 / 20 split.

---

## Frozen Evaluation Results

| Metric | Result |
|---|---:|
| Average Precision | 0.7924385 |
| ROC-AUC | 0.9784368 |
| Precision | 0.8636364 |
| Recall | 0.7600000 |
| F1 | 0.8085106 |
| Brier Score | 0.000402506 |
| True Positives | 57 |
| False Positives | 9 |
| True Negatives | 56,878 |
| False Negatives | 18 |
| False Positive Rate | 0.000158208 |
| False Negative Rate | 0.24 |
| Frozen Threshold | 0.25 |
| Cost Units | 369 |
| Cost Units / 1,000 | 6.4780 |

---

## Confusion Matrix

```text
                      Predicted

                   Legit     Fraud
Actual Legit      56,878         9
Actual Fraud          18        57
```

---

## False-Positive Cost

The system produced 9 false positives across 56,887 legitimate frozen-evaluation cases, for an FPR of approximately 0.0158%.

The evaluation cost function uses:

```text
false negative cost = 20
false positive cost = 1
```

The threshold is selected on calibration data, not on the frozen evaluation labels.

---

## Integrity Checks

Verified:

- raw dataset SHA256;
- normalized dataset SHA256;
- expected schema;
- expected row count;
- expected fraud count;
- chronological split;
- held-out truth excluded from recommendation;
- held-out truth excluded from scoring;
- post-recommendation truth reveal;
- calibration-legitimate-only Isolation Forest reference;
- full test suite;
- blind runtime execution;
- OPA block;
- HUMAN_ONLY final authority.

---

## Claim Boundaries

These results do NOT prove:

- every fraud can be detected;
- zero false negatives;
- production readiness for autonomous settlement action;
- Razorpay-specific fraud ground truth;
- safe autonomous money release.

They demonstrate a measurable, defense-only research benchmark and a financial safety architecture designed to fail safely.