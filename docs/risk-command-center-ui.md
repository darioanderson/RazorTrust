# RazorTrust Risk Command Center + Evaluation Lab

Status: final product/UI polish. No serving-model or financial-enforcement change.

## Design intent

The command center is deliberately closer to modern fraud-operations products than to a generic analytics dashboard:

- Stripe Radar: centralized fraud operations, reviews, risk controls and insights.
- Sardine-style investigation workflow: case-first evidence and analyst action.
- WCAG 2.2: visible keyboard focus and minimum target sizing.

The interface keeps RazorTrust's unique governance story more prominent than the visual references:

- `HUMAN_ONLY` is persistent.
- Automatic RELEASE is shown as OFF.
- Model, evidence, policy boundary and human authorization are visually separated.
- Rejected GraphSAGE and temporal experiments stay visible.
- Public benchmark evidence is clearly separated from synthetic research.
- Drift and conformal validation are displayed as research-only safety evidence.

## Frozen research evidence displayed

- Phase 3: graph statistics retained; static GraphSAGE rejected.
- Phase 4: temporal features closed for no incremental value; LSTM/Transformer not advanced.
- Phase 5: drift monitoring retained and frozen.
- Phase 6 V1.1: conformal uncertainty retained and frozen; 99.46% fresh marginal coverage at a 90% target; zero false RELEASE recommendations on the fresh research test.
- Public ULB benchmark: frozen public held-out transaction-fraud benchmark, explicitly not Razorpay production performance.

## Product safety

This UI patch does not modify `api.py`, risk scoring, OPA, model artifacts, feature schemas, model promotion, `human-only@1`, or automatic RELEASE settings.
