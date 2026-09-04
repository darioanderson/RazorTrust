# Human-only production mode

`RAZORTRUST_DECISION_MODE=human_only` is the safe operating mode when no model candidate has passed
the safe-RELEASE gate. It is intentionally separate from model failure fallback.

## Invariants

- No model release is loaded or required.
- Configuring `RAZORTRUST_MODEL_RELEASE_PATH` at the same time is rejected.
- Every evaluation returns an `ESCALATE` recommendation with `reason_code=HUMAN_ONLY`.
- The case moves to `HUMAN_REVIEW`; automated RELEASE is impossible.
- Evidence verification, OPA authorization, database health, audit-chain validation, and Razorpay
  readiness checks remain enforced.
- Human authorization remains required to resolve a case.

## Configuration

```text
RAZORTRUST_ENV=staging
RAZORTRUST_DECISION_MODE=human_only
RAZORTRUST_AUTHORIZATION_REQUIRED=true
```

Use `RAZORTRUST_DECISION_MODE=model` only after a signed, independently validated production model
release exists. Outside development, model mode remains fail closed when that release is unavailable
or fails verification.
