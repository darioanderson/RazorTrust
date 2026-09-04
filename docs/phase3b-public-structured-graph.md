# Phase 3B — Public Structured Graph Pilot

Research-only benchmark using the Fraud Detection Handbook simulated transaction dataset.

- Uses `CUSTOMER_ID` and `TERMINAL_ID` as real simulator entities; it does not fabricate identities from ULB PCA features.
- Historical graph topology uses transactions strictly before the scored transaction.
- Fraud-derived historical node statistics are delayed by 7 days.
- The current transaction is added as an unlabelled target node connected to its known customer and terminal.
- `TX_FRAUD` and `TX_FRAUD_SCENARIO` are targets/evaluation labels only.
- The pilot is chronological and case-control sampled to keep GraphSAGE CPU cost bounded. Its FPR and positive-rate metrics are sample metrics, not population operating rates.
- Original RazorTrust Phase 3 artifacts and decisions remain unchanged.
- No SEALED TEST, stress data, serving state, champion alias, OPA policy, automatic RELEASE, or money movement is modified.

Gate order:

1. Tabular XGBoost vs tabular + graph statistics.
2. GraphSAGE is considered only if graph statistics clear the first gate.
3. GraphSAGE must add at least +0.01 PR-AUC over graph statistics, preserve recall, avoid material sample-FPR regression, and improve scenario 2 or 3 recall.
