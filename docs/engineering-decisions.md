# Engineering decisions

## The primary domain is settlement holds

An earlier prototype mixed settlement risk with agent-payment authorization. They have different actors, states, and failure modes. The executable package now contains only the settlement-hold domain. The earlier direction is kept outside `src/` as historical context.

## Risk and policy are separate

The model estimates class probabilities. The cost policy chooses a candidate action. OPA can only preserve or make that action safer. This avoids embedding model logic in Rego and makes failures easy to reason about.

## All system failures escalate

Feature, model, calibration, persistence, and policy failures must never become an automatic release. The current implementation converts model and OPA failures into explicit `ESCALATE` records with reason codes.

## One evidence round is a domain invariant

The evidence round is not a UI convention. It is enforced by the repository transition logic and tested. Client-provided recency and match fields are rejected; the server derives them.

## The development scorer is labelled honestly

The runnable rule adapter exists to test the surrounding system before a trained artifact is available. Its output is uncalibrated, and its contributions are not called SHAP. Replacing it requires the merchant-isolated training and calibration gates described in the project report.

## PostgreSQL is an actual runtime path

When `RAZORTRUST_DATABASE_URL` is set, the API uses the async SQLAlchemy repository. The first
Alembic migration defines request-id uniqueness and the one-evidence-per-hold constraint, and the
Compose stack runs migrations before admitting API traffic. The in-memory repository remains a
deliberate zero-dependency development mode.

Database mode now commits the decision, RFC 8785 audit-chain record, and outbox event in one
transaction. The JSONL ledger is retained only for local development and export.
