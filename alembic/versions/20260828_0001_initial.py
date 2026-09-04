"""Create the Tier 0 settlement-hold tables.

Revision ID: 20260828_0001
Revises:
Create Date: 2026-08-28
"""

import sqlalchemy as sa

from alembic import op

revision = "20260828_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hold_cases",
        sa.Column("hold_id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("merchant_id", sa.String(length=128), nullable=False),
        sa.Column("source_event_id", sa.String(length=128), nullable=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reason_code", sa.String(length=128), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("evidence_round", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("evidence_round IN (0, 1)", name="ck_hold_evidence_round"),
        sa.PrimaryKeyConstraint("hold_id"),
        sa.UniqueConstraint("request_id", name="uq_hold_request_id"),
    )
    op.create_index("ix_hold_cases_merchant_id", "hold_cases", ["merchant_id"])

    op.create_table(
        "feature_snapshots",
        sa.Column("snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("hold_id", sa.Uuid(), nullable=False),
        sa.Column("feature_schema_version", sa.String(length=32), nullable=False),
        sa.Column("as_of_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("evaluation_input", sa.JSON(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["hold_id"], ["hold_cases.hold_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("snapshot_id"),
    )
    op.create_index("ix_feature_snapshots_hold_id", "feature_snapshots", ["hold_id"])

    op.create_table(
        "model_decisions",
        sa.Column("decision_id", sa.Uuid(), nullable=False),
        sa.Column("hold_id", sa.Uuid(), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("feature_schema_version", sa.String(length=32), nullable=False),
        sa.Column("policy_version", sa.String(length=128), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("decision_payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["hold_id"], ["hold_cases.hold_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("decision_id"),
    )
    op.create_index("ix_model_decisions_hold_id", "model_decisions", ["hold_id"])

    op.create_table(
        "evidence_submissions",
        sa.Column("evidence_id", sa.Uuid(), nullable=False),
        sa.Column("hold_id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("evidence_payload", sa.JSON(), nullable=False),
        sa.Column("recency_hours", sa.Float(), nullable=False),
        sa.Column("type_match", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["hold_id"], ["hold_cases.hold_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("evidence_id"),
        sa.UniqueConstraint("hold_id"),
        sa.UniqueConstraint("request_id", name="uq_evidence_request_id"),
    )

    op.create_table(
        "audit_events",
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("event_payload", sa.JSON(), nullable=False),
        sa.Column("previous_hash", sa.String(length=64), nullable=False),
        sa.Column("record_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint("case_id", "sequence_no", name="uq_audit_case_sequence"),
    )
    op.create_index("ix_audit_events_case_id", "audit_events", ["case_id"])

    op.create_table(
        "model_manifests",
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("signature_b64", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("model_version"),
    )
    op.create_table(
        "policy_versions",
        sa.Column("policy_version", sa.String(length=128), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("policy_version"),
    )


def downgrade() -> None:
    op.drop_table("policy_versions")
    op.drop_table("model_manifests")
    op.drop_index("ix_audit_events_case_id", table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_table("evidence_submissions")
    op.drop_index("ix_model_decisions_hold_id", table_name="model_decisions")
    op.drop_table("model_decisions")
    op.drop_index("ix_feature_snapshots_hold_id", table_name="feature_snapshots")
    op.drop_table("feature_snapshots")
    op.drop_index("ix_hold_cases_merchant_id", table_name="hold_cases")
    op.drop_table("hold_cases")
