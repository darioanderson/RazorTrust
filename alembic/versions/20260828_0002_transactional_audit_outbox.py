"""Add canonical transactional audit and outbox fields.

Revision ID: 20260828_0002
Revises: 20260828_0001
Create Date: 2026-08-28
"""

import sqlalchemy as sa

from alembic import op

revision = "20260828_0002"
down_revision = "20260828_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("audit_events", sa.Column("canonical_payload", sa.Text(), nullable=True))
    op.add_column("audit_events", sa.Column("trace_id", sa.String(length=64), nullable=True))
    op.execute("UPDATE audit_events SET canonical_payload = '{}', trace_id = 'legacy'")
    op.alter_column("audit_events", "canonical_payload", nullable=False)
    op.alter_column("audit_events", "trace_id", nullable=False)
    op.create_index("ix_audit_events_trace_id", "audit_events", ["trace_id"])
    op.create_table(
        "outbox_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("topic", sa.String(length=128), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_outbox_events_topic", "outbox_events", ["topic"])


def downgrade() -> None:
    op.drop_index("ix_outbox_events_topic", table_name="outbox_events")
    op.drop_table("outbox_events")
    op.drop_index("ix_audit_events_trace_id", table_name="audit_events")
    op.drop_column("audit_events", "trace_id")
    op.drop_column("audit_events", "canonical_payload")
