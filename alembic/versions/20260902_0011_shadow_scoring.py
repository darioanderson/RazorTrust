"""R4D persisted shadow-scoring provenance and model outputs.

Revision ID: 20260902_0011
Revises: 20260902_0010
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260902_0011"
down_revision = "20260902_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "razorpay_shadow_score_runs",
        sa.Column("shadow_score_id", sa.Uuid(), nullable=False),
        sa.Column("source_mode", sa.String(length=48), nullable=False),
        sa.Column("source_id", sa.String(length=160), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("blockers", sa.JSON(), nullable=False),
        sa.Column("feature_schema_version", sa.String(length=32), nullable=False),
        sa.Column("feature_vector", sa.JSON(), nullable=True),
        sa.Column("feature_vector_sha256", sa.String(length=64), nullable=True),
        sa.Column("feature_provenance", sa.JSON(), nullable=True),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("calibration_method", sa.String(length=64), nullable=True),
        sa.Column("probabilities", sa.JSON(), nullable=True),
        sa.Column("anomaly_score", sa.Float(), nullable=True),
        sa.Column("novelty_override", sa.Boolean(), nullable=False),
        sa.Column("top_features", sa.JSON(), nullable=False),
        sa.Column("shadow_recommendation", sa.String(length=32), nullable=True),
        sa.Column("expected_cost_units", sa.Float(), nullable=True),
        sa.Column("policy_version", sa.String(length=128), nullable=False),
        sa.Column("shadow_policy_mode", sa.String(length=32), nullable=False),
        sa.Column("production_action_eligible", sa.Boolean(), nullable=False),
        sa.Column("production_action_blocker", sa.String(length=512), nullable=False),
        sa.Column("score_contract_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload_sha256", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("shadow_score_id"),
    )
    op.create_index("ix_shadow_score_source_mode", "razorpay_shadow_score_runs", ["source_mode"])
    op.create_index("ix_shadow_score_source_id", "razorpay_shadow_score_runs", ["source_id"])
    op.create_index("ix_shadow_score_account_id", "razorpay_shadow_score_runs", ["account_id"])
    op.create_index("ix_shadow_score_as_of", "razorpay_shadow_score_runs", ["as_of"])
    op.create_index("ix_shadow_score_status", "razorpay_shadow_score_runs", ["status"])
    op.create_index(
        "ix_shadow_score_feature_hash", "razorpay_shadow_score_runs", ["feature_vector_sha256"]
    )
    op.create_index(
        "ix_shadow_score_model_version", "razorpay_shadow_score_runs", ["model_version"]
    )
    op.create_index(
        "ix_shadow_score_recommendation", "razorpay_shadow_score_runs", ["shadow_recommendation"]
    )
    op.create_index("ix_shadow_score_created_at", "razorpay_shadow_score_runs", ["created_at"])
    op.create_index(
        "ix_shadow_score_payload_hash", "razorpay_shadow_score_runs", ["payload_sha256"]
    )


def downgrade() -> None:
    op.drop_index("ix_shadow_score_payload_hash", table_name="razorpay_shadow_score_runs")
    op.drop_index("ix_shadow_score_created_at", table_name="razorpay_shadow_score_runs")
    op.drop_index("ix_shadow_score_recommendation", table_name="razorpay_shadow_score_runs")
    op.drop_index("ix_shadow_score_model_version", table_name="razorpay_shadow_score_runs")
    op.drop_index("ix_shadow_score_feature_hash", table_name="razorpay_shadow_score_runs")
    op.drop_index("ix_shadow_score_status", table_name="razorpay_shadow_score_runs")
    op.drop_index("ix_shadow_score_as_of", table_name="razorpay_shadow_score_runs")
    op.drop_index("ix_shadow_score_account_id", table_name="razorpay_shadow_score_runs")
    op.drop_index("ix_shadow_score_source_id", table_name="razorpay_shadow_score_runs")
    op.drop_index("ix_shadow_score_source_mode", table_name="razorpay_shadow_score_runs")
    op.drop_table("razorpay_shadow_score_runs")
