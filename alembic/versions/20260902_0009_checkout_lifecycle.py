"""R4C.1 checkout lifecycle hardening.

Revision ID: 20260902_0009
Revises: 20260902_0008
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260902_0009"
down_revision = "20260902_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "razorpay_checkout_orders",
        sa.Column("provider_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "razorpay_checkout_orders",
        sa.Column("lifecycle_status", sa.String(length=32), nullable=False, server_default="OPEN"),
    )
    op.add_column(
        "razorpay_checkout_orders",
        sa.Column("last_payment_status", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "razorpay_checkout_orders",
        sa.Column("webhook_confirmed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "razorpay_checkout_orders",
        sa.Column("last_provider_reconciled_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "razorpay_checkout_orders",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "razorpay_checkout_orders",
        sa.Column("abandoned_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Preserve already-verified R4C sessions as PAID. Other existing sessions are OPEN
    # until an authoritative order reconciliation updates them.
    op.execute(
        """
        UPDATE razorpay_checkout_orders
        SET lifecycle_status = CASE
            WHEN signature_verified_at IS NOT NULL AND authoritative_payment_status IN ('captured','refunded') THEN 'PAID'
            WHEN authoritative_payment_status = 'authorized' THEN 'AUTHORIZED'
            ELSE 'OPEN'
        END,
        provider_status = CASE
            WHEN signature_verified_at IS NOT NULL AND authoritative_payment_status IN ('captured','refunded') THEN 'paid'
            ELSE provider_status
        END,
        provider_attempts = CASE
            WHEN signature_verified_at IS NOT NULL AND authoritative_payment_status IN ('captured','refunded') THEN GREATEST(provider_attempts, 1)
            ELSE provider_attempts
        END,
        last_payment_status = authoritative_payment_status,
        expires_at = COALESCE(expires_at, created_at + interval '24 hours')
        """
    )
    op.create_index(
        "ix_checkout_lifecycle_status", "razorpay_checkout_orders", ["lifecycle_status"]
    )
    op.create_index(
        "ix_checkout_webhook_confirmed_at", "razorpay_checkout_orders", ["webhook_confirmed_at"]
    )
    op.create_index("ix_checkout_expires_at", "razorpay_checkout_orders", ["expires_at"])
    op.create_index("ix_checkout_abandoned_at", "razorpay_checkout_orders", ["abandoned_at"])

    op.alter_column("razorpay_checkout_orders", "provider_attempts", server_default=None)
    op.alter_column("razorpay_checkout_orders", "lifecycle_status", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_checkout_abandoned_at", table_name="razorpay_checkout_orders")
    op.drop_index("ix_checkout_expires_at", table_name="razorpay_checkout_orders")
    op.drop_index("ix_checkout_webhook_confirmed_at", table_name="razorpay_checkout_orders")
    op.drop_index("ix_checkout_lifecycle_status", table_name="razorpay_checkout_orders")
    op.drop_column("razorpay_checkout_orders", "abandoned_at")
    op.drop_column("razorpay_checkout_orders", "expires_at")
    op.drop_column("razorpay_checkout_orders", "last_provider_reconciled_at")
    op.drop_column("razorpay_checkout_orders", "webhook_confirmed_at")
    op.drop_column("razorpay_checkout_orders", "last_payment_status")
    op.drop_column("razorpay_checkout_orders", "lifecycle_status")
    op.drop_column("razorpay_checkout_orders", "provider_attempts")
