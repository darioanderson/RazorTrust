# RazorTrust 0.5.0 — Phase 3/4, automation and operator visibility

This candidate extends the verified 0.4.1 human-gated build without changing the fixed safety rule.

## Model governance

- Registry champion: `xgb-if-settlement@2`
- Signed champion release: `artifacts/tier0/model-release`
- Champion public key: `artifacts/tier0/model-release-public-key.txt`
- Human-only enforcement runtime: `human-only@1`
- Automatic RELEASE in human-only mode: **forbidden**
- Challenger auto-promotion: **forbidden**

A registry champion and an enforcement runtime are deliberately separate concepts. The champion remains
available for research/shadow comparison while `human-only@1` is the active enforcement adapter.

## Phase 3 — Graph/GNN

Existing leakage-safe graph statistics remain the mandatory baseline. The candidate adds an actual
GraphSAGE research model using PyTorch Geometric. Graph snapshots:

- contain only transactions with `timestamp < cutoff`;
- never use `ring_id` as a feature;
- map raw identifiers to ephemeral integer nodes;
- use merchant/device/customer/geo node types plus non-label activity features;
- remain research-only and cannot become the serving model without the fixed promotion gates.

Entrypoint: `scripts/train_graphsage_challenger.py`.

## Phase 4 — Sequence neural model

The existing engineered sequence baseline remains. The candidate adds a small LSTM research challenger
using strictly pre-cutoff event sequences. Inputs include log amount, log time gap, authorization failure,
device/geo transitions, and only refund/chargeback timestamps already observable at the cutoff.

Entrypoint: `scripts/train_sequence_lstm_challenger.py`.

## Phase 5 and Phase 6

These were already implemented as foundations before 0.5.0:

- batch KS + Jensen-Shannon drift checks;
- ADWIN and KSWIN online drift monitors;
- EVT tail severity;
- MAPIE split-conformal APS prediction sets and conservative abstention policy.

They remain support/safety layers and do not authorize money movement.

## Safe automation

`HoldAutomationOrchestrator` can periodically evaluate existing OPEN cases **only when a persisted
evaluation input already exists**. It never fabricates transaction history or baselines. Missing inputs are
skipped. In `human_only`, all automatic evaluations route through the mandatory human gate.

Staging enables:

- `RAZORTRUST_AUTOMATION_ENABLED=true`
- `RAZORTRUST_AUTOMATION_INTERVAL_SECONDS=60`

This is decision-workflow automation, not autonomous settlement execution.

## Operator UI

The overview now exposes:

- registry champion and signed artifact location;
- active enforcement runtime;
- automatic RELEASE status;
- GraphSAGE and LSTM challenger states;
- automation pipeline status;
- all seven phase implementation states.
