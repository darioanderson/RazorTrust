"""Add refunds, settlements, disputes and reconciliation state.

Revision ID: 20260902_0006
Revises: 20260902_0005
Create Date: 2026-09-02
"""

import sqlalchemy as sa

from alembic import op

revision = "20260902_0006"
down_revision = "20260902_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "merchant_processor_accounts",
        sa.Column("last_refund_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "merchant_processor_accounts",
        sa.Column("last_settlement_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "merchant_processor_accounts",
        sa.Column("last_dispute_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "processor_refunds",
        sa.Column("refund_id", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=True),
        sa.Column("payment_id", sa.String(length=128), nullable=False),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("speed_requested", sa.String(length=32), nullable=True),
        sa.Column("speed_processed", sa.String(length=32), nullable=True),
        sa.Column("provider_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_event_id", sa.String(length=128), nullable=False),
        sa.Column("source_event_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("authoritative_sha256", sa.String(length=64), nullable=False),
        sa.Column("enriched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("refund_id"),
    )
    op.create_index("ix_processor_refunds_account_id", "processor_refunds", ["account_id"])
    op.create_index("ix_processor_refunds_payment_id", "processor_refunds", ["payment_id"])
    op.create_index("ix_processor_refunds_status", "processor_refunds", ["status"])
    op.create_index(
        "ix_processor_refunds_provider_created_at",
        "processor_refunds",
        ["provider_created_at"],
    )

    op.create_table(
        "processor_refund_observations",
        sa.Column("observation_id", sa.Uuid(), nullable=False),
        sa.Column("refund_id", sa.String(length=128), nullable=False),
        sa.Column("payment_id", sa.String(length=128), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=True),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("speed_requested", sa.String(length=32), nullable=True),
        sa.Column("speed_processed", sa.String(length=32), nullable=True),
        sa.Column("provider_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_event_id", sa.String(length=128), nullable=False),
        sa.Column("source_event_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("authoritative_sha256", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("observation_id"),
        sa.UniqueConstraint("source_event_id", name="uq_refund_observation_source_event"),
    )
    op.create_index(
        "ix_processor_refund_observations_refund_id", "processor_refund_observations", ["refund_id"]
    )
    op.create_index(
        "ix_processor_refund_observations_payment_id",
        "processor_refund_observations",
        ["payment_id"],
    )
    op.create_index(
        "ix_processor_refund_observations_account_id",
        "processor_refund_observations",
        ["account_id"],
    )
    op.create_index(
        "ix_processor_refund_observations_status", "processor_refund_observations", ["status"]
    )
    op.create_index(
        "ix_processor_refund_observations_observed_at",
        "processor_refund_observations",
        ["observed_at"],
    )

    op.create_table(
        "processor_settlements",
        sa.Column("settlement_id", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=True),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("fees", sa.BigInteger(), nullable=False),
        sa.Column("tax", sa.BigInteger(), nullable=False),
        sa.Column("utr", sa.String(length=128), nullable=True),
        sa.Column("provider_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_event_id", sa.String(length=128), nullable=False),
        sa.Column("source_event_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("authoritative_sha256", sa.String(length=64), nullable=False),
        sa.Column("enriched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("settlement_id"),
    )
    op.create_index("ix_processor_settlements_account_id", "processor_settlements", ["account_id"])
    op.create_index("ix_processor_settlements_status", "processor_settlements", ["status"])
    op.create_index("ix_processor_settlements_utr", "processor_settlements", ["utr"])
    op.create_index(
        "ix_processor_settlements_provider_created_at",
        "processor_settlements",
        ["provider_created_at"],
    )

    op.create_table(
        "processor_settlement_observations",
        sa.Column("observation_id", sa.Uuid(), nullable=False),
        sa.Column("settlement_id", sa.String(length=128), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=True),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("fees", sa.BigInteger(), nullable=False),
        sa.Column("tax", sa.BigInteger(), nullable=False),
        sa.Column("utr", sa.String(length=128), nullable=True),
        sa.Column("provider_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_event_id", sa.String(length=128), nullable=False),
        sa.Column("source_event_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("authoritative_sha256", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("observation_id"),
        sa.UniqueConstraint("source_event_id", name="uq_settlement_observation_source_event"),
    )
    op.create_index(
        "ix_processor_settlement_observations_settlement_id",
        "processor_settlement_observations",
        ["settlement_id"],
    )
    op.create_index(
        "ix_processor_settlement_observations_account_id",
        "processor_settlement_observations",
        ["account_id"],
    )
    op.create_index(
        "ix_processor_settlement_observations_status",
        "processor_settlement_observations",
        ["status"],
    )
    op.create_index(
        "ix_processor_settlement_observations_observed_at",
        "processor_settlement_observations",
        ["observed_at"],
    )

    op.create_table(
        "processor_disputes",
        sa.Column("dispute_id", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=True),
        sa.Column("payment_id", sa.String(length=128), nullable=False),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("amount_deducted", sa.BigInteger(), nullable=False),
        sa.Column("reason_code", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("phase", sa.String(length=64), nullable=True),
        sa.Column("respond_by", sa.DateTime(timezone=True), nullable=True),
        sa.Column("provider_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_event_id", sa.String(length=128), nullable=False),
        sa.Column("source_event_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("authoritative_sha256", sa.String(length=64), nullable=False),
        sa.Column("enriched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("dispute_id"),
    )
    op.create_index("ix_processor_disputes_account_id", "processor_disputes", ["account_id"])
    op.create_index("ix_processor_disputes_payment_id", "processor_disputes", ["payment_id"])
    op.create_index("ix_processor_disputes_reason_code", "processor_disputes", ["reason_code"])
    op.create_index("ix_processor_disputes_status", "processor_disputes", ["status"])
    op.create_index("ix_processor_disputes_phase", "processor_disputes", ["phase"])
    op.create_index("ix_processor_disputes_respond_by", "processor_disputes", ["respond_by"])
    op.create_index(
        "ix_processor_disputes_provider_created_at", "processor_disputes", ["provider_created_at"]
    )

    op.create_table(
        "processor_dispute_observations",
        sa.Column("observation_id", sa.Uuid(), nullable=False),
        sa.Column("dispute_id", sa.String(length=128), nullable=False),
        sa.Column("payment_id", sa.String(length=128), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=True),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("amount_deducted", sa.BigInteger(), nullable=False),
        sa.Column("reason_code", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("phase", sa.String(length=64), nullable=True),
        sa.Column("respond_by", sa.DateTime(timezone=True), nullable=True),
        sa.Column("provider_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_event_id", sa.String(length=128), nullable=False),
        sa.Column("source_event_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("authoritative_sha256", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("observation_id"),
        sa.UniqueConstraint("source_event_id", name="uq_dispute_observation_source_event"),
    )
    op.create_index(
        "ix_processor_dispute_observations_dispute_id",
        "processor_dispute_observations",
        ["dispute_id"],
    )
    op.create_index(
        "ix_processor_dispute_observations_payment_id",
        "processor_dispute_observations",
        ["payment_id"],
    )
    op.create_index(
        "ix_processor_dispute_observations_account_id",
        "processor_dispute_observations",
        ["account_id"],
    )
    op.create_index(
        "ix_processor_dispute_observations_status", "processor_dispute_observations", ["status"]
    )
    op.create_index(
        "ix_processor_dispute_observations_observed_at",
        "processor_dispute_observations",
        ["observed_at"],
    )

    op.create_table(
        "processor_reconciliation_items",
        sa.Column("reconciliation_id", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("entity_id", sa.String(length=128), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("debit", sa.BigInteger(), nullable=False),
        sa.Column("credit", sa.BigInteger(), nullable=False),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("fee", sa.BigInteger(), nullable=False),
        sa.Column("tax", sa.BigInteger(), nullable=False),
        sa.Column("on_hold", sa.Boolean(), nullable=True),
        sa.Column("settled", sa.Boolean(), nullable=True),
        sa.Column("provider_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("settlement_id", sa.String(length=128), nullable=True),
        sa.Column("payment_id", sa.String(length=128), nullable=True),
        sa.Column("order_id", sa.String(length=128), nullable=True),
        sa.Column("dispute_id", sa.String(length=128), nullable=True),
        sa.Column("method", sa.String(length=64), nullable=True),
        sa.Column("settlement_utr", sa.String(length=128), nullable=True),
        sa.Column("fetched_year", sa.Integer(), nullable=False),
        sa.Column("fetched_month", sa.Integer(), nullable=False),
        sa.Column("fetched_day", sa.Integer(), nullable=True),
        sa.Column("authoritative_sha256", sa.String(length=64), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("reconciliation_id"),
    )
    for name, column in (
        ("ix_processor_reconciliation_items_entity_id", "entity_id"),
        ("ix_processor_reconciliation_items_entity_type", "entity_type"),
        ("ix_processor_reconciliation_items_provider_created_at", "provider_created_at"),
        ("ix_processor_reconciliation_items_settled_at", "settled_at"),
        ("ix_processor_reconciliation_items_settlement_id", "settlement_id"),
        ("ix_processor_reconciliation_items_payment_id", "payment_id"),
        ("ix_processor_reconciliation_items_order_id", "order_id"),
        ("ix_processor_reconciliation_items_dispute_id", "dispute_id"),
        ("ix_processor_reconciliation_items_settlement_utr", "settlement_utr"),
        ("ix_processor_reconciliation_items_fetched_at", "fetched_at"),
    ):
        op.create_index(name, "processor_reconciliation_items", [column])

    # R3A processed refund events only as payment refreshes. Re-open them so R3B
    # records the refund entity too. Events previously skipped solely because R3A
    # did not support settlements/disputes are also made eligible for reconstruction.
    op.execute(
        """
        UPDATE processor_webhook_events
        SET processing_status = 'RECEIVED',
            processing_attempts = 0,
            processing_started_at = NULL,
            processed_at = NULL,
            last_error = NULL
        WHERE provider = 'RAZORPAY'
          AND (
            event_type IN ('refund.created','refund.processed','refund.failed','refund.speed_changed')
            OR (
              processing_status = 'SKIPPED'
              AND last_error = 'unsupported_r3a_event_type'
              AND event_type IN (
                'settlement.processed',
                'payment.dispute.created',
                'payment.dispute.won',
                'payment.dispute.lost',
                'payment.dispute.closed',
                'payment.dispute.under_review',
                'payment.dispute.action_required'
              )
            )
          )
        """
    )


def downgrade() -> None:
    for name in (
        "ix_processor_reconciliation_items_fetched_at",
        "ix_processor_reconciliation_items_settlement_utr",
        "ix_processor_reconciliation_items_dispute_id",
        "ix_processor_reconciliation_items_order_id",
        "ix_processor_reconciliation_items_payment_id",
        "ix_processor_reconciliation_items_settlement_id",
        "ix_processor_reconciliation_items_settled_at",
        "ix_processor_reconciliation_items_provider_created_at",
        "ix_processor_reconciliation_items_entity_type",
        "ix_processor_reconciliation_items_entity_id",
    ):
        op.drop_index(name, table_name="processor_reconciliation_items")
    op.drop_table("processor_reconciliation_items")

    op.drop_index(
        "ix_processor_dispute_observations_observed_at", table_name="processor_dispute_observations"
    )
    op.drop_index(
        "ix_processor_dispute_observations_status", table_name="processor_dispute_observations"
    )
    op.drop_index(
        "ix_processor_dispute_observations_account_id", table_name="processor_dispute_observations"
    )
    op.drop_index(
        "ix_processor_dispute_observations_payment_id", table_name="processor_dispute_observations"
    )
    op.drop_index(
        "ix_processor_dispute_observations_dispute_id", table_name="processor_dispute_observations"
    )
    op.drop_table("processor_dispute_observations")
    for name in (
        "ix_processor_disputes_provider_created_at",
        "ix_processor_disputes_respond_by",
        "ix_processor_disputes_phase",
        "ix_processor_disputes_status",
        "ix_processor_disputes_reason_code",
        "ix_processor_disputes_payment_id",
        "ix_processor_disputes_account_id",
    ):
        op.drop_index(name, table_name="processor_disputes")
    op.drop_table("processor_disputes")

    op.drop_index(
        "ix_processor_settlement_observations_observed_at",
        table_name="processor_settlement_observations",
    )
    op.drop_index(
        "ix_processor_settlement_observations_status",
        table_name="processor_settlement_observations",
    )
    op.drop_index(
        "ix_processor_settlement_observations_account_id",
        table_name="processor_settlement_observations",
    )
    op.drop_index(
        "ix_processor_settlement_observations_settlement_id",
        table_name="processor_settlement_observations",
    )
    op.drop_table("processor_settlement_observations")
    for name in (
        "ix_processor_settlements_provider_created_at",
        "ix_processor_settlements_utr",
        "ix_processor_settlements_status",
        "ix_processor_settlements_account_id",
    ):
        op.drop_index(name, table_name="processor_settlements")
    op.drop_table("processor_settlements")

    op.drop_index(
        "ix_processor_refund_observations_observed_at", table_name="processor_refund_observations"
    )
    op.drop_index(
        "ix_processor_refund_observations_status", table_name="processor_refund_observations"
    )
    op.drop_index(
        "ix_processor_refund_observations_account_id", table_name="processor_refund_observations"
    )
    op.drop_index(
        "ix_processor_refund_observations_payment_id", table_name="processor_refund_observations"
    )
    op.drop_index(
        "ix_processor_refund_observations_refund_id", table_name="processor_refund_observations"
    )
    op.drop_table("processor_refund_observations")
    for name in (
        "ix_processor_refunds_provider_created_at",
        "ix_processor_refunds_status",
        "ix_processor_refunds_payment_id",
        "ix_processor_refunds_account_id",
    ):
        op.drop_index(name, table_name="processor_refunds")
    op.drop_table("processor_refunds")

    op.drop_column("merchant_processor_accounts", "last_dispute_at")
    op.drop_column("merchant_processor_accounts", "last_settlement_at")
    op.drop_column("merchant_processor_accounts", "last_refund_at")
