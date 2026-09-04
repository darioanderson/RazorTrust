"""Allow an evidence request followed by a final human approval.

Revision ID: 20260903_0013
Revises: 20260902_0012
Create Date: 2026-09-03
"""

from alembic import op

revision = "20260903_0013"
down_revision = "20260902_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("analyst_reviews_hold_id_key", "analyst_reviews", type_="unique")


def downgrade() -> None:
    op.create_unique_constraint("analyst_reviews_hold_id_key", "analyst_reviews", ["hold_id"])
