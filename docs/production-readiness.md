# Production deployment and custody

`docker-compose.staging.yml` is the production-like deployment contract. It differs intentionally
from the local developer stack. Because no model candidate has passed the safe-RELEASE gate, staging
sets `RAZORTRUST_DECISION_MODE=human_only`; this keeps readiness operational while forcing every
evaluation into named human review and forbidding automated RELEASE.


- authorization is mandatory and evaluated by OPA;
- API principals, database credentials, Razorpay credentials, and webhook secrets are mounted as
  Docker secrets and read through `*_FILE` settings;
- only the Caddy gateway is published, with internal TLS, request-size limits, and security headers;
- the application enforces trusted hosts, a fail-closed per-process rate limit, HSTS, CSP, and
  no-store responses;
- the API container runs read-only, without Linux capabilities or privilege escalation;
- OPA downloads an ES256-signed policy bundle and receives only the public verification key;
- OpenTelemetry must export to a real configured collector destination instead of `debug`.

Before deployment, build `deploy/policy/bundle.tar.gz` with `scripts/build_signed_opa_bundle.py`.
Keep its private key in KMS/HSM-backed CI custody. Supply all secret file environment variables
required by the Compose file; startup fails if any are absent. Replace Caddy's internal certificate
authority with the organization's public/private PKI at the external ingress boundary.

The in-process limiter protects a single instance. A multi-replica deployment must additionally
enforce an identity-aware distributed limit at the gateway or service mesh. Run the k6 smoke profile
before promotion, run `scripts/backup_restore_drill.ps1` (or the Python entry point) to restore a
custom-format dump into a fresh disposable PostgreSQL database and verify its schema/Alembic state,
and exercise rollback to the previous immutable image digest and signed policy bundle.

Signed audit checkpoints are the bridge from the transactional audit chain to immutable storage.
Run `scripts/publish_audit_checkpoint.py` against an HTTPS object-lock endpoint. The publisher
verifies the checkpoint signature before upload, addresses objects by their SHA-256 digest, and uses
`If-None-Match: *` so an existing checkpoint cannot be overwritten. Configure retention/legal hold
at the storage provider; an ordinary writable bucket is not WORM storage.

Settlement execution remains deliberately out of scope. A human authorization is an auditable
decision, not a money-moving call. Add a processor/product-specific adapter only after its exact API,
idempotency, permissions, reconciliation, and rollback contract is approved.
