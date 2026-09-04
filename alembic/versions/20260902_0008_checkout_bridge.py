"""R4C Razorpay checkout and telemetry bridge.

Revision ID: 20260902_0008
Revises: 20260902_0007
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260902_0008"
down_revision = "20260902_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "razorpay_checkout_orders",
        sa.Column("checkout_session_id", sa.Uuid(), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=False),
        sa.Column("razorpay_order_id", sa.String(length=128), nullable=True),
        sa.Column("razorpay_payment_id", sa.String(length=128), nullable=True),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("receipt", sa.String(length=40), nullable=False),
        sa.Column("provider_status", sa.String(length=32), nullable=False),
        sa.Column("device_fingerprint_sha256", sa.String(length=64), nullable=False),
        sa.Column("customer_geo", sa.String(length=64), nullable=False),
        sa.Column("client_event_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("telemetry_id", sa.Uuid(), nullable=True),
        sa.Column("authoritative_payment_status", sa.String(length=32), nullable=True),
        sa.Column("signature_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("checkout_contract_version", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("checkout_session_id"),
        sa.UniqueConstraint("razorpay_order_id", name="uq_razorpay_checkout_order_id"),
        sa.UniqueConstraint("receipt", name="uq_razorpay_checkout_receipt"),
    )
    op.create_index("ix_checkout_account_id", "razorpay_checkout_orders", ["account_id"])
    op.create_index("ix_checkout_order_id", "razorpay_checkout_orders", ["razorpay_order_id"])
    op.create_index("ix_checkout_payment_id", "razorpay_checkout_orders", ["razorpay_payment_id"])
    op.create_index("ix_checkout_receipt", "razorpay_checkout_orders", ["receipt"])
    op.create_index("ix_checkout_telemetry_id", "razorpay_checkout_orders", ["telemetry_id"])
    op.create_index(
        "ix_checkout_signature_verified_at",
        "razorpay_checkout_orders",
        ["signature_verified_at"],
    )
    op.create_index("ix_checkout_created_at", "razorpay_checkout_orders", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_checkout_created_at", table_name="razorpay_checkout_orders")
    op.drop_index("ix_checkout_signature_verified_at", table_name="razorpay_checkout_orders")
    op.drop_index("ix_checkout_telemetry_id", table_name="razorpay_checkout_orders")
    op.drop_index("ix_checkout_receipt", table_name="razorpay_checkout_orders")
    op.drop_index("ix_checkout_payment_id", table_name="razorpay_checkout_orders")
    op.drop_index("ix_checkout_order_id", table_name="razorpay_checkout_orders")
    op.drop_index("ix_checkout_account_id", table_name="razorpay_checkout_orders")
    op.drop_table("razorpay_checkout_orders")
