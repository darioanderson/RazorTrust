"""Add analyst outcomes as auditable training labels.

Revision ID: 20260829_0003
Revises: 20260828_0002
Create Date: 2026-08-29
"""

import sqlalchemy as sa

from alembic import op

revision = "20260829_0003"
down_revision = "20260828_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "analyst_reviews",
        sa.Column("review_id", sa.Uuid(), nullable=False),
        sa.Column("hold_id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("analyst_id", sa.String(length=128), nullable=False),
        sa.Column("review_payload", sa.JSON(), nullable=False),
        sa.Column("training_label", sa.String(length=32), nullable=False),
        sa.Column("audit_head_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["hold_id"], ["hold_cases.hold_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("review_id"),
        sa.UniqueConstraint("hold_id"),
        sa.UniqueConstraint("request_id", name="uq_analyst_review_request_id"),
    )


def downgrade() -> None:
    op.drop_table("analyst_reviews")
