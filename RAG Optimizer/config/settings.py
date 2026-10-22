"""Application settings — single source of truth for all connection strings and tuning params."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Infrastructure ──────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    elasticsearch_url: str = "http://localhost:9200"
    # Plain postgresql:// used by asyncpg; Alembic prepends +asyncpg automatically
    postgres_dsn: str = "postgresql://rag:rag@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379/0"

    # ── API Keys ────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    cohere_api_key: str = ""

    # ── Reranker selection ───────────────────────────────────────────────
    # auto: cohere when key present, otherwise local BGE
    # cohere: managed Cohere API (requires COHERE_API_KEY)
    # bge: local CrossEncoder model
    # none: disable reranking
    reranker_provider: str = "auto"
    bge_reranker_model: str = "BAAI/bge-reranker-base"

    # ── Bandit ─────────────────────────────────────────────────────────
    bandit_epsilon: float = 0.15
    bandit_min_samples: int = 10

    # ── Cache TTLs (seconds) ────────────────────────────────────────────
    classifier_cache_ttl: int = 60
    registry_cache_ttl: int = 300

    # ── Retrieval defaults ─────────────────────────────────────────────
    default_top_k: int = 10
    rerank_top_n: int = 5
    retrieval_confidence_threshold: float = 0.35
    enable_confidence_retry: bool = True

    # ── Cost estimation (USD per 1k tokens) ───────────────────────────
    # Defaults are approximate for Anthropic Sonnet-class pricing.
    input_cost_per_1k_tokens_usd: float = 0.003
    output_cost_per_1k_tokens_usd: float = 0.015

    # ── Online evaluation ───────────────────────────────────────────────
    # Share of live requests scored by RAGAS-style metrics + LLM judge (several LLM calls each).
    eval_sample_rate: float = 1.0

    # ── Observability ─────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    prometheus_port: int = 8001

    # ── Security ─────────────────────────────────────────────────
    # When non-empty, all API endpoints (except /health) require X-Api-Key header
    api_key: str = ""
    metrics_require_api_key: bool = False
    metrics_api_key: str = ""
    cors_origins: list[str] = ["*"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
