"""Add privacy-minimised Razorpay webhook event storage.

Revision ID: 20260901_0004
Revises: 20260829_0003
Create Date: 2026-09-01
"""

import sqlalchemy as sa

from alembic import op

revision = "20260901_0004"
down_revision = "20260829_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "processor_webhook_events",
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("provider_event_id", sa.String(length=128), nullable=False),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=True),
        sa.Column("event_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload_sha256", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processing_status", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint("provider", "provider_event_id", name="uq_processor_provider_event"),
    )
    op.create_index(
        "ix_processor_webhook_events_event_type",
        "processor_webhook_events",
        ["event_type"],
    )
    op.create_index(
        "ix_processor_webhook_events_account_id",
        "processor_webhook_events",
        ["account_id"],
    )
    op.create_index(
        "ix_processor_webhook_events_received_at",
        "processor_webhook_events",
        ["received_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_processor_webhook_events_received_at", table_name="processor_webhook_events")
    op.drop_index("ix_processor_webhook_events_account_id", table_name="processor_webhook_events")
    op.drop_index("ix_processor_webhook_events_event_type", table_name="processor_webhook_events")
    op.drop_table("processor_webhook_events")
