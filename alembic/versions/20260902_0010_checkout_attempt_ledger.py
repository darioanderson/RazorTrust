"""R4C.2 payment-attempt ledger and authoritative attempt sync.

Revision ID: 20260902_0010
Revises: 20260902_0009
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260902_0010"
down_revision = "20260902_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "razorpay_checkout_payment_attempts",
        sa.Column("attempt_id", sa.Uuid(), nullable=False),
        sa.Column("checkout_session_id", sa.Uuid(), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=False),
        sa.Column("razorpay_order_id", sa.String(length=128), nullable=False),
        sa.Column("razorpay_payment_id", sa.String(length=128), nullable=False),
        sa.Column("payment_status", sa.String(length=32), nullable=False),
        sa.Column("captured", sa.Boolean(), nullable=False),
        sa.Column("last_event_type", sa.String(length=128), nullable=False),
        sa.Column("last_source_event_id", sa.String(length=128), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("attempt_id"),
        sa.UniqueConstraint("razorpay_payment_id", name="uq_checkout_attempt_payment_id"),
    )
    op.create_index(
        "ix_checkout_attempt_session_id",
        "razorpay_checkout_payment_attempts",
        ["checkout_session_id"],
    )
    op.create_index(
        "ix_checkout_attempt_account_id", "razorpay_checkout_payment_attempts", ["account_id"]
    )
    op.create_index(
        "ix_checkout_attempt_order_id", "razorpay_checkout_payment_attempts", ["razorpay_order_id"]
    )
    op.create_index(
        "ix_checkout_attempt_payment_id",
        "razorpay_checkout_payment_attempts",
        ["razorpay_payment_id"],
    )
    op.create_index(
        "ix_checkout_attempt_payment_status",
        "razorpay_checkout_payment_attempts",
        ["payment_status"],
    )
    op.create_index(
        "ix_checkout_attempt_last_seen_at", "razorpay_checkout_payment_attempts", ["last_seen_at"]
    )

    # Backfill attempts already reconstructed by R3A/R3B and bound to a checkout order.
    # A failed payment is intentionally not written into first_party_transaction_telemetry.payment_id:
    # one order can have multiple failed attempts before a later success. The attempt ledger is 1:N.
    op.execute(
        """
        INSERT INTO razorpay_checkout_payment_attempts (
            attempt_id,
            checkout_session_id,
            account_id,
            razorpay_order_id,
            razorpay_payment_id,
            payment_status,
            captured,
            last_event_type,
            last_source_event_id,
            first_seen_at,
            last_seen_at,
            created_at,
            updated_at
        )
        SELECT
            (
                substr(md5(p.payment_id || c.checkout_session_id::text), 1, 8) || '-' ||
                substr(md5(p.payment_id || c.checkout_session_id::text), 9, 4) || '-' ||
                substr(md5(p.payment_id || c.checkout_session_id::text), 13, 4) || '-' ||
                substr(md5(p.payment_id || c.checkout_session_id::text), 17, 4) || '-' ||
                substr(md5(p.payment_id || c.checkout_session_id::text), 21, 12)
            )::uuid,
            c.checkout_session_id,
            c.account_id,
            p.order_id,
            p.payment_id,
            p.status,
            p.captured,
            'BACKFILL',
            NULL,
            COALESCE(p.enriched_at, c.created_at),
            COALESCE(p.enriched_at, c.updated_at),
            COALESCE(p.enriched_at, c.created_at),
            COALESCE(p.enriched_at, c.updated_at)
        FROM processor_payments p
        JOIN razorpay_checkout_orders c
          ON c.razorpay_order_id = p.order_id
        WHERE p.order_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM razorpay_checkout_payment_attempts a
              WHERE a.razorpay_payment_id = p.payment_id
          )
        """
    )

    # Repair stale local attempt counters from the explicit 1:N attempt ledger.
    op.execute(
        """
        UPDATE razorpay_checkout_orders c
        SET provider_attempts = GREATEST(
            c.provider_attempts,
            COALESCE((
                SELECT COUNT(*)
                FROM razorpay_checkout_payment_attempts a
                WHERE a.checkout_session_id = c.checkout_session_id
            ), 0)
        )
        """
    )


def downgrade() -> None:
    op.drop_index(
        "ix_checkout_attempt_last_seen_at", table_name="razorpay_checkout_payment_attempts"
    )
    op.drop_index(
        "ix_checkout_attempt_payment_status", table_name="razorpay_checkout_payment_attempts"
    )
    op.drop_index("ix_checkout_attempt_payment_id", table_name="razorpay_checkout_payment_attempts")
    op.drop_index("ix_checkout_attempt_order_id", table_name="razorpay_checkout_payment_attempts")
    op.drop_index("ix_checkout_attempt_account_id", table_name="razorpay_checkout_payment_attempts")
    op.drop_index("ix_checkout_attempt_session_id", table_name="razorpay_checkout_payment_attempts")
    op.drop_table("razorpay_checkout_payment_attempts")
