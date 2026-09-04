# Phase 4 temporal diagnostic hotfix 1

Diagnostic-only audit before accepting the Phase 4 negative result.

Checks:
- point-in-time and rolling-window invariants;
- NaN/Inf, zero variance, no-history rate, and feature distributions;
- 1h/6h/24h/72h windows;
- add-one and drop-one ablation for each temporal feature;
- history-availability and event-count encodings;
- repeated generator seeds using identical grouped CV logic;
- fixed false RELEASE / false HOLD costs of 100 / 25.

Safety: no SEALED or stress evaluation, no production action, no model promotion, and champion remains xgb-if-settlement@2.
