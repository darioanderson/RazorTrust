"""R4D.1 novelty diagnostics and accuracy-integrity metadata.

Revision ID: 20260902_0012
Revises: 20260902_0011
Create Date: 2026-09-02
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260902_0012"
down_revision = "20260902_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "razorpay_shadow_score_runs",
        sa.Column("anomaly_raw_score", sa.Float(), nullable=True),
    )
    op.add_column(
        "razorpay_shadow_score_runs",
        sa.Column("anomaly_reference_max", sa.Float(), nullable=True),
    )
    op.add_column(
        "razorpay_shadow_score_runs",
        sa.Column("anomaly_tail_excess", sa.Float(), nullable=True),
    )
    op.add_column(
        "razorpay_shadow_score_runs",
        sa.Column("anomaly_reference_size", sa.Integer(), nullable=True),
    )
    op.add_column(
        "razorpay_shadow_score_runs",
        sa.Column("anomaly_model_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "razorpay_shadow_score_runs",
        sa.Column("anomaly_reference_mode", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("razorpay_shadow_score_runs", "anomaly_reference_mode")
    op.drop_column("razorpay_shadow_score_runs", "anomaly_model_version")
    op.drop_column("razorpay_shadow_score_runs", "anomaly_reference_size")
    op.drop_column("razorpay_shadow_score_runs", "anomaly_tail_excess")
    op.drop_column("razorpay_shadow_score_runs", "anomaly_reference_max")
    op.drop_column("razorpay_shadow_score_runs", "anomaly_raw_score")
