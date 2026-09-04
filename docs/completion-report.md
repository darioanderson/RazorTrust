# RazorTrust completion report

This report separates implemented behavior from empirical promotion gates and from controls that
require a real deployment environment. It intentionally makes no production-performance claim.

## Final v3A/v3B stopping decision - 2026-09-03

The v3A error audit used only merchant-isolated development OOF predictions. At its locked operating
point it measured 2.50% false RELEASE and 22.14% true RELEASE recall. False releases were concentrated
in refund-abuse-style (16) and auth-provider-incident (9) windows; 109 of 140 baseline legitimate
windows were missed. A predeclared boundary-AUC rule justified exactly three derived v3B signals:
legitimate stability, identity dispersion gap, and authorization/novelty pressure.

The one final v3B Optuna experiment ran 15 seeded XGBoost trials on the RTX 5060 (`cuda:0`). The
untouched gate measured 3.00% false RELEASE and 10.00% true RELEASE recall. It therefore failed the
unchanged target (<= 5% false RELEASE and >= 20% true RELEASE recall). The stopping decision is
`STOP_ML`; no sealed, stress, future-time, unseen-merchant, or unknown-family validation was opened.
The candidate, audit, HPO history, manifests, hashes, and signature are frozen as research-only and
non-promotable.

The live product workflow has accordingly shifted away from model experimentation. Staging can now
run explicitly with `RAZORTRUST_DECISION_MODE=human_only`: readiness does not require a model release,
but every evaluation deterministically routes to `ESCALATE` / `HUMAN_REVIEW` with no automated
RELEASE path. In `model` mode outside development, a signed verified release remains mandatory and
readiness fails closed when it is absent or invalid. A named reviewer must approve, request evidence,
escalate, or explicitly override the recommendation. Overrides require a replacement decision, reason,
and rationale.
The audit payload retains human and AI identity, agent session, delegated permission, transaction,
authorized amount/item, provider metadata, timestamp, and the hash-chain proof. Authorized reviewers
can download a case-specific PDF dossier from the API or console.

## Implemented application path

- Point-in-time 13-feature snapshots with merchant-isolated data splits.
- Weighted XGBoost plus legitimate-only Isolation Forest, group out-of-fold calibration, a shared
  versioned cost matrix, constrained threshold selection, and class-specific SHAP explanations.
- FastAPI orchestration with idempotency, one evidence round, role/merchant authorization, OPA
  guardrails, fail-closed behavior, trace propagation, and optional OpenTelemetry/Sentry exporters.
- PostgreSQL decision, RFC 8785 audit-chain, and outbox writes in one transaction. Local JSONL is
  explicitly non-authoritative when database mode is enabled.
- Ed25519-signed model releases using native XGBoost and skops artifacts. Loading verifies the
  signature, every file hash, feature schema, cost artifact, and release metadata before deserialization.
- Optuna grouped HPO on development merchants only; bootstrap intervals, six explicit baselines,
  fixed-threshold cost sensitivity, MAPIE APS conformal abstention, River change detectors, KS/JS
  drift with multiple-testing correction, Evidently reports, and EVT tail severity.
- Leave-one-attack-family-out novelty evaluation, point-in-time ecosystem graph statistics, a grouped
  cold-subgraph ablation gate, and an engineered temporal baseline.

## Reproducible validation snapshot

The full-scale mechanics run used synthetic seed 20260830, 40 merchants per scenario family, 80
transactions per merchant, 300 estimators, and 5 grouped HPO trials. Its sealed merchant test set
contained 84 rows: 13 `RELEASE`, 32 `EVIDENCE_NEEDED`, and 39 `ESCALATE` cases. Temperature calibration
was selected. The full Tier 0 threshold policy reported 4.70238095 expected cost units, zero observed
false releases, escalation recall 1.0, log loss 0.03230722, multiclass Brier score 0.01362032, and
top-label ECE 0.02036235. Merchant-bootstrap 95% intervals were [3.98809524, 5.71428571] for expected
cost, [0.0, 0.0] for false-release rate, and [1.0, 1.0] for escalation recall. The APS conformal layer
reported 1.0 empirical coverage, 1.07142857 average set size, a 0.94047619 singleton rate, and a
0.05952381 ambiguous-set rate at 90% nominal confidence.

All three reports contain top-level `source_provenance` and `source_tree_sha256` fields with value
`d6135535fa07fa259055952374978d1769d90883829a5a66c08de48edc6bf3b2`. The synthetic dataset SHA-256
is `5aa8550a6713238e5d66c3eee3218bf6df3e92217ce13c261a8b210032a616da`, and the sealed evaluation
content SHA-256 is `3f3a3b5526cd3496d37970cb2a56848966762d7473f0d7403cb597b5c3061811`.

Coarse decision metrics (false-release rate, escalation recall, conformal coverage) reproduce exactly
across environments given a matching source hash; fine-grained probability metrics (log loss, ECE,
conformal set size) may vary slightly due to floating-point non-determinism in gradient-boosted tree
construction across hardware, which is a known limitation independent of seeding correctness.

These numbers are mechanics checks on a small synthetic sample, not real-world performance claims.
The immutable run artifacts are deliberately excluded from the source release ZIP; rerun
`scripts/train_model.py` with the documented parameters to reproduce them.

The independent seed-20260830 research rerun used 40 merchants per scenario family. Graph statistics
did not clear their gate: graph PR-AUC was 0.99663726 versus 0.99667613 for tabular features, graph
expected cost was 2.05357143 versus 1.83035714, and both ring recalls were 0.99166667. GraphSAGE and
temporal GNNs therefore remain blocked. A matched-scale PyTorch LOAFO rerun used 80 transactions per
merchant and evaluated 40 held-out cases per family. Autoencoder recall was 1.0 for account-takeover
shift, card-testing style, distributed device rings, refund-abuse style, slow-low rings, and synthetic
volume spikes, and 0.725 for mixed evasion. Isolation Forest recall on the same families was 1.0, 1.0,
0.1, 0.125, 0.725, 0.475, and 1.0 respectively (the last value is mixed evasion). The autoencoder
therefore adds substantial measured value on four families, ties on two, and trails Isolation Forest
on mixed evasion. The existing harness does not emit an automatic autoencoder promotion decision, so
this closes the missing evaluation but does not by itself place the optional tier in the live release
path. Sequence neural models remain blocked until they beat the engineered temporal baseline. The
system retains exactly one evidence round.

The two weakest LOAFO families have a specific, understood representation gap. For distributed device
rings, the novelty detector receives only the locked 13-feature `model_matrix`; merchant, customer,
device, and ring identities are forbidden, and cross-merchant reuse statistics such as shared-device
count and component growth exist only in the separately gated graph ablation. At seed 20260830 and the
documented scale, the distributed-ring median volume z-score was 0.0, GMV z-score was 0.256368, and
hourly volume slope was -0.006957. Its six reused devices produced a local entropy of 1.791139, but the
detector cannot see that those devices connect multiple merchants. Mahalanobis distance flagged every
case, while Isolation Forest recall was only 0.1. The autoencoder also flagged every case, but averaging
all three percentile signals before applying the unchanged 0.95 threshold produced combined recall of
0.45. The cross-merchant relationship remains absent from every novelty input despite the stronger
reconstruction-error signal.

Slow-low rings share the same missing cross-merchant context and are deliberately shaped to avoid a
large single-window spike. Their median 24-hour volume z-score was -0.666667, GMV z-score was
-0.214773, and hourly slope was 0.0. Isolation Forest recall was 0.725, while autoencoder and combined
recall were both 1.0 at this scale. The autoencoder closes this synthetic held-out test, but transactions
are still spread evenly inside one 24-hour hold window and the locked feature vector has no multi-day
trajectory; slow accumulation across windows remains structurally unobservable. These are documented
limitations, not evidence that either family is safe, and no feature, detector, or promotion threshold
was changed.

## External production controls

The repository supplies interfaces and local infrastructure, but a deployment owner must still provide
real processor data, an independent immutable audit checkpoint, production key custody/rotation,
signed OPA bundle distribution, vulnerability and container signing in CI, alert routing, retention and
privacy policy, human-review operations, and a shadow/canary period with rollback authority. These are
deployment controls, not facts that can be truthfully simulated inside a source package.
