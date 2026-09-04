# Phase 3B V2 — Heterogeneous GraphSAGE Fusion

Research-only architecture correction. It does not alter the canonical RazorTrust production path,
the exact 13 production features, `xgb-if-settlement@2`, `human-only@1`, OPA, SEALED TEST, or money movement.

## Why V2 exists

V1 compared `tabular + graph statistics` against a standalone GraphSAGE model. That was not a true
incremental test because the GNN challenger discarded information available to the graph-stat baseline.
V2 keeps the baseline intact and asks whether a learned heterogeneous graph embedding adds value beyond it.

## Graph schema

Node types:
- `customer`
- `transaction`
- `terminal`

Relations:
- customer `makes` transaction
- transaction `made_by` customer
- transaction `at` terminal
- terminal `hosts` transaction

Historical transaction events remain transaction nodes instead of being collapsed into one customer-terminal
edge. Each transaction carries amount, cyclical time features, weekend/night flags, relative age, and a
current-transaction marker. Customer and terminal nodes carry point-in-time aggregate activity and delayed
known-fraud rates.

## Leakage rules

For target transaction at time `T`:
- historical topology requires `timestamp < T`;
- current `TX_FRAUD` and `TX_FRAUD_SCENARIO` are target/evaluation only;
- future events are forbidden;
- fraud-derived history is available only after the configured delay (default 7 days);
- node-feature normalization is fit on representation-training snapshots only.

## Fair fusion evaluation

Training is chronologically split inside the TRAIN partition:
1. earlier representation-training cases fit GraphSAGE and normalization;
2. later classifier-training cases fit all XGBoost comparators;
3. POLICY selects thresholds;
4. TEST remains untouched until final evaluation.

Comparators:
- A: tabular XGBoost;
- B: tabular + point-in-time graph statistics;
- C: B + frozen heterogeneous GraphSAGE embedding.

This prevents the fusion classifier from being trained on embeddings whose target labels were used to train
the GNN for those same cases.

## Pre-registered research gate

Graph-stat value gate vs tabular:
- PR-AUC delta >= +0.01;
- F1 not worse;
- recall >= baseline - 0.05;
- sample FPR <= baseline + 0.02.

Heterogeneous fusion incremental gate vs graph statistics:
- PR-AUC delta >= +0.01;
- F1 not worse;
- recall >= graph stats - 0.03;
- sample FPR <= graph stats + 0.02;
- Scenario 2 or Scenario 3 recall must improve.

Pilot case-control metrics are comparative research metrics and are not population operating rates.
No result from this experiment is production-action eligible or automatically promotable.
