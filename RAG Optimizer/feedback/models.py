"""SQLAlchemy ORM models for the feedback loop.

These serve as the schema source of truth and type reference.
The updater uses raw asyncpg for bulk queries; ORM is available for
ad-hoc exploration and future Phase 6 API endpoints.
"""

from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class FeedbackEvent(Base):
    """Explicit (thumbs up/down) or implicit (behavioural) user signal."""

    __tablename__ = "feedback_events"

    id              = Column(Integer, primary_key=True)
    query           = Column(Text, nullable=False)
    strategy        = Column(String(50), nullable=False)
    query_type      = Column(String(50), nullable=False)
    signal_type     = Column(String(20), nullable=False)   # "explicit" | "implicit"
    signal_value    = Column(Float, nullable=False)         # 1.0=positive  0.0=negative  0.5=neutral
    was_exploration = Column(Boolean, nullable=False, default=False)
    session_id      = Column(String(100), nullable=True)
    created_at      = Column(DateTime, server_default=func.now())


class RagasScore(Base):
    """RAGAS evaluation scores produced by benchmarks/ragas_scorer.py."""

    __tablename__ = "ragas_scores"

    id                = Column(Integer, primary_key=True)
    query             = Column(Text, nullable=False)
    strategy          = Column(String(50), nullable=False)
    query_type        = Column(String(50), nullable=False, default="unknown")
    faithfulness      = Column(Float)
    answer_relevancy  = Column(Float)
    context_precision = Column(Float)
    composite         = Column(Float)
    was_exploration   = Column(Boolean, nullable=False, default=False)
    created_at        = Column(DateTime, server_default=func.now())


class JudgeScore(Base):
    """LLM judge scores produced by benchmarks/llm_judge.py."""

    __tablename__ = "judge_scores"

    id              = Column(Integer, primary_key=True)
    query           = Column(Text, nullable=False)
    strategy        = Column(String(50), nullable=False)
    query_type      = Column(String(50), nullable=False, default="unknown")
    relevance       = Column(Integer)
    completeness    = Column(Integer)
    groundedness    = Column(Integer)
    clarity         = Column(Integer)
    composite       = Column(Float)
    reasoning       = Column(Text)
    was_exploration = Column(Boolean, nullable=False, default=False)
    created_at      = Column(DateTime, server_default=func.now())
