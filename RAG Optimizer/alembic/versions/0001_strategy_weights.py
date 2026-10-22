"""Add strategy_weights table.

Revision ID: 0001
"""

from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS strategy_weights (
            id           SERIAL PRIMARY KEY,
            strategy     VARCHAR(50)  NOT NULL,
            query_type   VARCHAR(50)  NOT NULL,
            avg_score    FLOAT        NOT NULL DEFAULT 0.0,
            sample_count INT          NOT NULL DEFAULT 0,
            updated_at   TIMESTAMP    NOT NULL DEFAULT NOW(),
            UNIQUE (strategy, query_type)
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_strategy_weights_query_type
        ON strategy_weights (query_type);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS strategy_weights;")
