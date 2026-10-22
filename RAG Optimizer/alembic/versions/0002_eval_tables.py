"""Add ragas_scores and judge_scores tables.

Revision ID: 0002
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS ragas_scores (
            id                SERIAL       PRIMARY KEY,
            query             TEXT         NOT NULL,
            strategy          VARCHAR(50)  NOT NULL,
            faithfulness      FLOAT,
            answer_relevancy  FLOAT,
            context_precision FLOAT,
            composite         FLOAT,
            was_exploration   BOOLEAN      NOT NULL DEFAULT FALSE,
            created_at        TIMESTAMP    NOT NULL DEFAULT NOW()
        );
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS judge_scores (
            id              SERIAL       PRIMARY KEY,
            query           TEXT         NOT NULL,
            strategy        VARCHAR(50)  NOT NULL,
            relevance       INT,
            completeness    INT,
            groundedness    INT,
            clarity         INT,
            composite       FLOAT,
            reasoning       TEXT,
            was_exploration BOOLEAN      NOT NULL DEFAULT FALSE,
            created_at      TIMESTAMP    NOT NULL DEFAULT NOW()
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_ragas_strategy  ON ragas_scores  (strategy);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_judge_strategy  ON judge_scores   (strategy);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ragas_scores;")
    op.execute("DROP TABLE IF EXISTS judge_scores;")
