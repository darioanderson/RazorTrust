# Final release verification — 2026-09-03

> **Post-audit patch note:** version 0.4.1 adds explicit `human_only` operation and a real
> fresh-database backup/restore drill. In this build environment, Python syntax and 14 focused
> human-only/settings/backup tests passed; the original release's full Windows/Docker verification
> must be rerun with `scripts/verify_post_audit_hardening.ps1` before replacing the previously
> verified 0.4.0 package.

The release was verified from the previously delivered whole-project ZIP in a new extraction and a
fresh Python 3.12 environment with the `test,ml,novelty,graph,drift,uncertainty` extras. No additional
model training, tuning, sealed evaluation, or unknown-family evaluation was performed.

## Passing gates

- Full unfiltered Python regression: **136 passed** across all discovered test modules.
- Ruff formatting and lint: pass for canonical source, migrations, scripts, and tests.
- mypy: pass for all **58** `razortrust` source modules.
- OPA: **9/9** hold-policy and authorization tests pass on OPA 1.19.1.
- Hardened staging Compose: configuration parses successfully with all required secret inputs.
- Genuine Docker E2E: PostgreSQL 17.11, all Alembic migrations, FastAPI, and OPA 1.19.1 passed
  signed webhook ingestion, deduplication, reconstruction/recovery, feature-contract blocking,
  risk evaluation, signed evidence, human authorization, RBAC isolation, audit and PDF generation,
  API restart/persistence, and model/OPA fail-closed outage checks.

## Deliberate release posture

The final ML decision remains `FINAL_ML_STOP_NO_SAFE_RELEASE_CANDIDATE`. Staging uses the explicit
`human_only` decision mode: readiness may pass without a model release, but every evaluation routes to
human review and automated RELEASE is impossible. `model` mode outside development still fails closed
until a signed, verified production model release is configured. Evidence also fails closed outside
development unless a provider Ed25519 verification key is configured. A human authorization records
permission for an exact item; it does not execute settlement movement. RazorTrust contains no
universal money-moving adapter.

## Reproduction

```text
python -m ruff format --check alembic src scripts tests
python -m ruff check alembic src scripts tests
python -m mypy src/razortrust
python -m pytest
docker run --rm -v <project>/policies:/policies:ro openpolicyagent/opa:1.19.1-static test /policies -v
python scripts/run_docker_e2e.py
docker compose -f docker-compose.staging.yml config --quiet
```
