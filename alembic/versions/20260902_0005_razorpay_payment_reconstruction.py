"""Add authoritative Razorpay payment reconstruction state.

Revision ID: 20260902_0005
Revises: 20260901_0004
Create Date: 2026-09-02
"""

import sqlalchemy as sa

from alembic import op

revision = "20260902_0005"
down_revision = "20260901_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "processor_webhook_events",
        sa.Column("processing_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "processor_webhook_events",
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "processor_webhook_events",
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "processor_webhook_events",
        sa.Column("last_error", sa.String(length=128), nullable=True),
    )
    op.create_index(
        "ix_processor_webhook_events_processing_status",
        "processor_webhook_events",
        ["processing_status"],
    )
    op.create_index(
        "ix_processor_webhook_events_processed_at",
        "processor_webhook_events",
        ["processed_at"],
    )

    op.create_table(
        "processor_payments",
        sa.Column("payment_id", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=True),
        sa.Column("order_id", sa.String(length=128), nullable=True),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("method", sa.String(length=64), nullable=True),
        sa.Column("captured", sa.Boolean(), nullable=False),
        sa.Column("amount_refunded", sa.BigInteger(), nullable=False),
        sa.Column("refund_status", sa.String(length=32), nullable=True),
        sa.Column("international", sa.Boolean(), nullable=True),
        sa.Column("provider_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_event_id", sa.String(length=128), nullable=False),
        sa.Column("source_event_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("authoritative_sha256", sa.String(length=64), nullable=False),
        sa.Column("enriched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("payment_id"),
    )
    op.create_index("ix_processor_payments_account_id", "processor_payments", ["account_id"])
    op.create_index("ix_processor_payments_order_id", "processor_payments", ["order_id"])
    op.create_index("ix_processor_payments_status", "processor_payments", ["status"])
    op.create_index(
        "ix_processor_payments_provider_created_at",
        "processor_payments",
        ["provider_created_at"],
    )

    op.create_table(
        "processor_payment_observations",
        sa.Column("observation_id", sa.Uuid(), nullable=False),
        sa.Column("payment_id", sa.String(length=128), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("method", sa.String(length=64), nullable=True),
        sa.Column("captured", sa.Boolean(), nullable=False),
        sa.Column("amount_refunded", sa.BigInteger(), nullable=False),
        sa.Column("refund_status", sa.String(length=32), nullable=True),
        sa.Column("international", sa.Boolean(), nullable=True),
        sa.Column("provider_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_event_id", sa.String(length=128), nullable=False),
        sa.Column("source_event_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("authoritative_sha256", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("observation_id"),
        sa.UniqueConstraint("source_event_id", name="uq_payment_observation_source_event"),
    )
    op.create_index(
        "ix_processor_payment_observations_payment_id",
        "processor_payment_observations",
        ["payment_id"],
    )
    op.create_index(
        "ix_processor_payment_observations_account_id",
        "processor_payment_observations",
        ["account_id"],
    )
    op.create_index(
        "ix_processor_payment_observations_status",
        "processor_payment_observations",
        ["status"],
    )
    op.create_index(
        "ix_processor_payment_observations_observed_at",
        "processor_payment_observations",
        ["observed_at"],
    )

    op.create_table(
        "merchant_processor_accounts",
        sa.Column("account_row_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("provider_account_id", sa.String(length=128), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_payment_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_event_type", sa.String(length=128), nullable=True),
        sa.PrimaryKeyConstraint("account_row_id"),
        sa.UniqueConstraint(
            "provider", "provider_account_id", name="uq_processor_merchant_account"
        ),
    )
    op.create_index(
        "ix_merchant_processor_accounts_provider_account_id",
        "merchant_processor_accounts",
        ["provider_account_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_merchant_processor_accounts_provider_account_id",
        table_name="merchant_processor_accounts",
    )
    op.drop_table("merchant_processor_accounts")
    op.drop_index(
        "ix_processor_payment_observations_observed_at",
        table_name="processor_payment_observations",
    )
    op.drop_index(
        "ix_processor_payment_observations_status",
        table_name="processor_payment_observations",
    )
    op.drop_index(
        "ix_processor_payment_observations_account_id",
        table_name="processor_payment_observations",
    )
    op.drop_index(
        "ix_processor_payment_observations_payment_id",
        table_name="processor_payment_observations",
    )
    op.drop_table("processor_payment_observations")
    op.drop_index("ix_processor_payments_provider_created_at", table_name="processor_payments")
    op.drop_index("ix_processor_payments_status", table_name="processor_payments")
    op.drop_index("ix_processor_payments_order_id", table_name="processor_payments")
    op.drop_index("ix_processor_payments_account_id", table_name="processor_payments")
    op.drop_table("processor_payments")
    op.drop_index("ix_processor_webhook_events_processed_at", table_name="processor_webhook_events")
    op.drop_index(
        "ix_processor_webhook_events_processing_status", table_name="processor_webhook_events"
    )
    op.drop_column("processor_webhook_events", "last_error")
    op.drop_column("processor_webhook_events", "processed_at")
    op.drop_column("processor_webhook_events", "processing_started_at")
    op.drop_column("processor_webhook_events", "processing_attempts")
