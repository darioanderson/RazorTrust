# Post-audit hardening — 2026-09-03

This patch starts implementing the remaining code-side items found during the production ZIP audit.
It does not claim to provision external infrastructure such as KMS/HSM/Vault, a live evidence
provider, public TLS/DNS, WORM storage, or an OTLP/alert backend.

## Implemented now

### Explicit human-only operating mode

- Added `RAZORTRUST_DECISION_MODE=model|human_only`.
- `human_only` is explicit, never an implicit model fallback.
- In `human_only`, readiness does not require a model release.
- Every evaluation deterministically recommends `ESCALATE`, records `reason_code=HUMAN_ONLY`, and
  enters `HUMAN_REVIEW`.
- Automated RELEASE is impossible in this mode.
- A model release path is rejected when `human_only` is configured, preventing accidental model
  activation.
- `model` mode keeps the previous fail-closed behavior outside development.
- Readiness now reports `decision_mode` and `risk_runtime`.
- Hardened staging Compose explicitly selects `human_only` while the ML stopping decision remains
  `FINAL_ML_STOP_NO_SAFE_RELEASE_CANDIDATE`.
- Docker E2E now includes a separate human-only staging service and verifies its readiness/runtime.

### Evidence documentation correction

- Corrected the stale statement that the evidence verifier is universally mocked.
- Development may use the legacy compatibility verifier.
- Staging/production still require Ed25519-signed provider attestations and fail closed without a
  configured verification key.

### Real backup/restore drill

- Replaced the old catalog-only backup check with a real fresh-database restore drill.
- The drill creates a PostgreSQL custom-format dump, copies it to the requested host path, creates a
  disposable restore database, restores with `pg_restore --exit-on-error`, verifies public tables and
  `alembic_version`, and removes the temporary database unless explicitly retained.
- The live `razortrust` database name is refused as a restore target.

## External items still intentionally not fabricated

- Live evidence-attestation provider and its key custody/rotation.
- KMS/HSM/Vault-backed signing and secret custody.
- Public DNS and publicly trusted TLS ingress.
- External WORM/object-lock audit checkpoint destination.
- Real OTLP backend and alert routing.
- Multi-replica distributed rate limiting.
- Live-environment load, restore, rollback and operator drills.
- Processor/product-specific money-moving adapter (still deliberately out of scope).
- Any new production model; the ML stopping decision is unchanged.
