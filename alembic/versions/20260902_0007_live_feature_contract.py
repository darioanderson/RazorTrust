"""Add first-party telemetry for the locked live feature contract.

Revision ID: 20260902_0007
Revises: 20260902_0006
Create Date: 2026-09-02
"""

import sqlalchemy as sa

from alembic import op

revision = "20260902_0007"
down_revision = "20260902_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "processor_payment_observations",
        sa.Column("order_id", sa.String(length=128), nullable=True),
    )
    op.create_index(
        "ix_processor_payment_observations_order_id",
        "processor_payment_observations",
        ["order_id"],
    )
    op.execute(
        """
        UPDATE processor_payment_observations AS o
        SET order_id = p.order_id
        FROM processor_payments AS p
        WHERE p.payment_id = o.payment_id AND o.order_id IS NULL
        """
    )

    op.create_table(
        "first_party_transaction_telemetry",
        sa.Column("telemetry_id", sa.Uuid(), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=False),
        sa.Column("order_id", sa.String(length=128), nullable=False),
        sa.Column("payment_id", sa.String(length=128), nullable=True),
        sa.Column("device_fingerprint_sha256", sa.String(length=64), nullable=False),
        sa.Column("customer_geo", sa.String(length=64), nullable=False),
        sa.Column("client_event_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_contract_version", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("telemetry_id"),
    )
    op.create_index(
        "ix_first_party_telemetry_account_id",
        "first_party_transaction_telemetry",
        ["account_id"],
    )
    op.create_index(
        "ix_first_party_telemetry_order_id",
        "first_party_transaction_telemetry",
        ["order_id"],
    )
    op.create_index(
        "ix_first_party_telemetry_payment_id",
        "first_party_transaction_telemetry",
        ["payment_id"],
    )
    op.create_index(
        "ix_first_party_telemetry_device_hash",
        "first_party_transaction_telemetry",
        ["device_fingerprint_sha256"],
    )
    op.create_index(
        "ix_first_party_telemetry_customer_geo",
        "first_party_transaction_telemetry",
        ["customer_geo"],
    )
    op.create_index(
        "ix_first_party_telemetry_observed_at",
        "first_party_transaction_telemetry",
        ["observed_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_first_party_telemetry_observed_at", table_name="first_party_transaction_telemetry"
    )
    op.drop_index(
        "ix_first_party_telemetry_customer_geo", table_name="first_party_transaction_telemetry"
    )
    op.drop_index(
        "ix_first_party_telemetry_device_hash", table_name="first_party_transaction_telemetry"
    )
    op.drop_index(
        "ix_first_party_telemetry_payment_id", table_name="first_party_transaction_telemetry"
    )
    op.drop_index(
        "ix_first_party_telemetry_order_id", table_name="first_party_transaction_telemetry"
    )
    op.drop_index(
        "ix_first_party_telemetry_account_id", table_name="first_party_transaction_telemetry"
    )
    op.drop_table("first_party_transaction_telemetry")
    op.drop_index(
        "ix_processor_payment_observations_order_id", table_name="processor_payment_observations"
    )
    op.drop_column("processor_payment_observations", "order_id")
