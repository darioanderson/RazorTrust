# RazorTrust implementation report

Date: 2026-09-03

## Outcome

RazorTrust completed the v3A audit and the single permitted v3B GPU experiment under the predeclared gate. v3B passed the false-RELEASE constraint but failed true-RELEASE recall, so the binding decision is `STOP_ML`. Sealed, stress, future-time, unseen-merchant, and unknown-family validation data were not accessed.

The implementation therefore shifted to the product path: evidence cases, explicit human authorization, attribution, immutable audit records, a PDF dossier, operator metrics, fail-closed behavior, and buildathon presentation assets. A new explicit `human_only` deployment mode keeps the API operational without a production model while forcing every risk evaluation into human review; it is not a fallback model and cannot auto-RELEASE.

## Research results

| Stage | False RELEASE | True RELEASE recall | Result |
|---|---:|---:|---|
| v3A development OOF audit | 2.5% | 22.14% | Audit only |
| v3B fixed gate | 3.0% | 10.0% | Failed recall gate |
| Locked target | <= 5.0% | >= 20.0% | Not changed |

The v3A false-RELEASE errors were concentrated in refund-abuse-style (16), authorization-provider-incident (9), flash-sale (4), synthetic-volume-spike (2), mixed-evasion (2), influencer-campaign (1), and product-launch (1) families. Baseline cases accounted for 109 missed legitimate cases, or 78% of that family.

Only three v3B feature additions were allowed by the audit rule: `legitimate_stability_score`, `identity_dispersion_gap`, and `auth_novelty_pressure`.

## Frozen research candidate

The v3B research candidate contains the model, calibrator, feature set, policy, dataset and partition hashes, reproducibility report, audit reports, canonical manifest, Ed25519 signature, and public verification key. It declares no auto-promotion and requires human approval. Signature verification passed for the manifest and all 12 referenced files.

This is a research artifact, not a production release.

## Product workflow implemented

- AI RELEASE and ESCALATE outputs now enter `HUMAN_REVIEW`; they no longer resolve a hold automatically.
- Human actions support approve RELEASE, request evidence, escalate, and explicit AI override.
- Overrides require an authorized replacement decision, reason, and rationale.
- Audit attribution retains human identity, AI agent identity/session, delegated permissions, amount/item scope, provider metadata, transaction identity, and timestamp.
- The Evidence Case API and UI show payment identity, risk probabilities, main signals, requested evidence, device/geo status, recent transaction timeline, supporting-document hashes, attribution, and the audit chain.
- The operator summary exposes attention counts, decision counts, research metrics, risk-family counts, recent decisions, pending reviews, evidence completeness, and readiness.
- The one-click PDF dossier is generated from recorded case facts and includes an explicit human-approval warning.
- A runnable failure demo proves that a model outage becomes `ESCALATE` plus `HUMAN_REVIEW`, never RELEASE.

## Verification completed

- Changed-file Python lint: passed (with the repository's pre-existing FastAPI `B008` rule excluded).
- JavaScript syntax check: passed.
- Focused regression suite: 32 passed.
- Broad installed-dependency suite: 115 passed, 1 skipped.
- Five suites requiring optional `torch`, `networkx`, `mapie`, or `river` extras were excluded from the broad run; no failures were hidden.
- v3B candidate signature and 12 manifest file hashes: valid.
- Evidence dossier: 2-page PDF, rendered and visually inspected.
- Buildathon deck: 9 slides, all rendered and individually inspected; overflow test passed.
- Live browser exercise: create hold, evaluate, submit evidence, rescore, display evidence facts, and open human-authorization control.

## Run the demonstrations

```powershell
$env:RAZORTRUST_POLICY_MODE = "local"
$env:RAZORTRUST_AUTHORIZATION_REQUIRED = "false"
uvicorn razortrust.api:app --reload
```

Use the web interface at `http://127.0.0.1:8000`, or run:

```powershell
python scripts/run_demo.py
python scripts/run_failure_demo.py
```

The timed five-minute presentation script is in `docs/buildathon-demo.md`.

## Remaining boundary

The software can now run safely in explicit human-only mode without pretending that the failed v3B candidate is production-approved. A real deployment still requires a live evidence-attestation provider, production KMS/HSM/Vault custody, external immutable audit checkpoint storage, external OTLP/alert routing, operator validation, privacy/retention review, and live-environment load/restore/rollback drills. A future model-backed mode still requires a separately validated signed production release.
