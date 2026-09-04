# RazorTrust - Five-Minute Demo Script

## 0:00 - 0:25 - Problem

Say:

"Fraud systems usually focus on one question: is this transaction fraudulent?

RazorTrust asks a harder question: what happens when the model is uncertain or wrong, but money is about to move?

RazorTrust combines ML risk detection with uncertainty, explainability, deterministic policy controls, provenance, and mandatory human authorization."

Show:

- RazorTrust UI
- HUMAN_ONLY
- AUTO RELEASE OFF

---

## 0:25 - 0:50 - Architecture

Show the architecture diagram.

Say:

"The flow starts with source provenance and authoritative history. Bad or insufficient history is blocked before ML.

Valid data goes through point-in-time features, XGBoost, calibration, Isolation Forest, SHAP and conformal uncertainty.

The recommendation then reaches an OPA safety boundary. The AI never receives autonomous release authority."

---

## 0:50 - 1:25 - Real Evaluation

Show benchmark status.

Say:

"I evaluated the system on the public real-world ULB / Worldline fraud dataset.

It contains 284,807 transactions and 492 fraud labels.

The split is chronological: 60 percent train, 20 percent calibration, 20 percent frozen evaluation."

Show:

- dataset hashes
- source verified
- split information

---

## 1:25 - 2:05 - Metrics

Show metrics.

Say:

"On 56,962 frozen evaluation transactions:

Average Precision is 0.7924.
ROC-AUC is 0.9784.
Precision is 86.36 percent.
Recall is 76 percent.
F1 is 80.85 percent.

The confusion matrix is:

57 true positives,
9 false positives,
56,878 true negatives,
and 18 false negatives.

I am deliberately showing the false negatives because risk systems should report their failures honestly."

---

## 2:05 - 2:35 - False Positive Cost

Say:

"The frozen evaluation produced only 9 false positives across 56,887 legitimate cases, an FPR of about 0.0158 percent.

For policy selection I model false-negative cost as 20 and false-positive cost as 1.

That policy was selected from calibration data, not from the frozen evaluation labels."

---

## 2:35 - 3:30 - Blind Execution

Open one blind benchmark case.

Show:

```text
source_label = null
label_revealed = false
```

Run the case.

Walk through stages:

```text
SOURCE_PROVENANCE
DATASET_INTEGRITY
CHRONOLOGICAL_SPLIT
BENCHMARK_XGBOOST
PROBABILITY_CALIBRATION
ISOLATION_FOREST
SHAP
CONFORMAL_UNCERTAINTY
BENCHMARK_RECOMMENDATION
OPA
SOURCE_GROUND_TRUTH
HUMAN_ONLY
```

Say:

"The ground-truth label is deliberately unavailable while the recommendation is formed.

Only after scoring and recommendation does RazorTrust read the source label."

Show:

```text
truth_access_mode = POST_RECOMMENDATION_ONLY
used_for_model_scoring = false
used_for_calibration_of_this_case = false
held_out_labels_used_for_recommendation = false
```

---

## 3:30 - 4:10 - Failure Safety

Show OPA stage.

Say:

"Even after the AI makes a recommendation, this public benchmark is blocked from the production settlement policy.

OPA reports BLOCKED.

The policy input is not submitted.

The system then ends in WAITING_FOR_HUMAN."

Show:

```text
automatic_release_enabled = false
production_action_eligible = false
human_authorization_required = true
```

---

## 4:10 - 4:35 - Provider Cases

Say:

"RazorTrust preserves an important distinction: a payment failure is not automatically fraud.

Provider-backed cases can be blocked because the system does not have enough trustworthy history.

Instead of inventing confidence, RazorTrust requests evidence or human review."

Do NOT claim failed payments are confirmed fraud.

---

## 4:35 - 5:00 - Close

Say:

"RazorTrust is not trying to build a model that claims it catches every fraud.

It is building a risk system that knows when to score, when to abstain, how to explain itself, and when it must stop and ask a human.

The result is measurable, auditable, defense-only, and designed so an AI recommendation can never silently become an irreversible money action."