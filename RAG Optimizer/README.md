# Adaptive RAG Optimization Platform

Adaptive RAG infrastructure that routes queries across multiple retrieval strategies, evaluates answer quality, and feeds those signals back into future routing decisions.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-BM25%20Search-005571?logo=elasticsearch&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Strategy%20Store-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Cache%20%26%20Buffer-DC382D?logo=redis&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?logo=prometheus&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Local%20Infra-2496ED?logo=docker&logoColor=white)

## Navigation

- [Quickstart](#quickstart)
- [System Architecture](#system-architecture)
- [How Adaptive RAG Works](#how-adaptive-rag-works)
- [API Usage](#api-usage)
- [Design Decisions](#design-decisions)
- [Performance / Results](#performance--results)

## Credibility Hardening Checklist

- [ ] Commit a real benchmark baseline at `benchmarks/baselines/nfcorpus.json` so the CI benchmark gate runs on every PR.
- [x] Wire real reranker interchangeability (Cohere, local BGE, auto fallback, or disabled) through runtime settings.
- [x] Align query API contract with asynchronous evaluation behavior (`evaluation_status`, trace opt-in, token/cost estimates).
- [x] Add explicit control for `/metrics` authentication in public deployments.
- [x] Implement first-pass confidence gating + adaptive retry and persist both RAGAS + judge scores asynchronously.

## Quickstart

Use this when you want to run the project quickly for a portfolio demo or interview walkthrough.

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (recommended for local infra)
- Anthropic API key (`ANTHROPIC_API_KEY`)
- Cohere API key (`COHERE_API_KEY`, required only if using Cohere reranking)

### 1) Clone and install

```bash
git clone <your-repo-url>
cd "RAG Optimizer"
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2) Start local infrastructure

```bash
docker compose up -d qdrant elasticsearch postgres redis
```

### 3) Seed retrieval data

```bash
python scripts/seed_corpus.py
```

### 4) Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5) Send a test query

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What year was Python first released?",
    "top_k": 5
  }'
```

### Non-Docker setup

If Docker is not available, point `.env` at existing Qdrant, Elasticsearch, PostgreSQL, and Redis services.

## Overview

Traditional RAG pipelines are usually fixed:

`user query -> one retriever -> top-k chunks -> LLM`

That works for narrow workloads, but it fails when query shapes vary. Exact factual queries often benefit from lexical search, semantic questions benefit from dense retrieval, and multi-hop questions may need query expansion or larger context windows. A fixed pipeline applies the same retrieval behavior to every question even though the failure modes are different.

This project treats retrieval as a decision problem instead of a hard-coded step. It classifies each query, selects a retrieval strategy from several specialized retrievers, optionally reranks candidates, compresses context before generation, and evaluates the answer asynchronously. Those signals are aggregated with user feedback and written back into the routing layer so future requests can make better choices.

The result is a production-oriented RAG system focused on measurable optimization rather than a static vector-search wrapper around an LLM.

## At a Glance

- **Problem:** fixed RAG pipelines underperform when query types vary.
- **Approach:** classify the query, route to the best retrieval strategy, rerank and compress context, then learn from evaluation signals.
- **Core differentiator:** the system updates future routing decisions using asynchronous scoring and feedback instead of relying on one permanent retrieval policy.
- **Production shape:** FastAPI API, multi-store retrieval stack, background learning loop, observability, and benchmarking utilities.

## What This Demonstrates

- Building production-oriented RAG services with explicit request-path and learning-path separation.
- Designing adaptive retrieval systems that improve via measurable feedback instead of static heuristics.
- Engineering multi-store backend workflows across vector search, lexical search, caching, and durable scoring state.
- Making quality-latency-cost tradeoffs explicit through configurable reranking, retrieval depth, and observability.

## Key Features

### Implemented

- Claude-based query classification with Redis caching.
- Adaptive routing across 6 retrieval strategies: `dense`, `bm25`, `hybrid`, `hyde`, `multi_query`, `parent_child`.
- Qdrant-backed semantic retrieval and Elasticsearch BM25 retrieval.
- Hybrid retrieval via reciprocal rank fusion.
- Candidate expansion to `3 x top_k` before reranking when a reranker is enabled.
- Runtime-selectable reranker provider: `cohere`, `bge`, `auto`, or `none`.
- LLM-based context compression before generation.
- Grounded answer generation constrained to retrieved context.
- Asynchronous RAGAS-style answer scoring off the hot path.
- Asynchronous LLM-judge persistence (`judge_scores`) off the hot path.
- Retrieval-confidence scoring plus adaptive retry/fallback to stronger strategies.
- Request-level token and cost estimates for latency-quality-cost tradeoff tracking.
- Redis-buffered feedback and PostgreSQL-backed strategy-weight updates every 60 seconds.
- Prometheus metrics, Grafana dashboards, and optional Langfuse tracing.
- BEIR-compatible benchmarking and regression-checking utilities.

### Planned or Partial

- Metadata filtering in the live request path. `[TODO]`
- Query-dependent automatic `top_k` selection beyond the explicit request parameter. `[TODO]`
- Learned confidence calibration and policy tuning beyond the current heuristic retry gate. `[TODO]`
- Exact provider-billed token/cost accounting (current implementation is deterministic estimation). `[TODO]`
- Cost-aware routing reward: the bandit reward is quality-only, exploration is a constant epsilon, and selection probabilities are not logged for off-policy evaluation. `[TODO]`
- Offline-to-online signal alignment study: track correlation between offline nDCG and online RAGAS/judge reward by query type before promoting policy changes. `[TODO]`
- Rerank/compression ablation matrix is not yet committed; current quality numbers should not be read as isolated causal lift from those components. `[TODO]`
- Fire-and-forget scoring hardening: tracked task references, bounded in-process queue, and durable worker queue for restart safety. `[TODO]`
- Multi-replica updater coordination: leader election/distributed lock and monotonic versioning for deterministic policy updates. `[TODO]`
- Classifier accuracy audit and a second BEIR dataset: NFCorpus queries are short and keyword-like, so strategy gaps there are small. `[TODO]`
- Measured end-to-end P50/P95 under concurrent load: the <2 s / <6 s figures are design targets, not results. `[TODO]`

## System Architecture

The implementation separates the latency-sensitive request path from the slower evaluation and learning path.

```mermaid
flowchart TD
    A[User Query] --> B[FastAPI Query Endpoint]
    B --> C[Query Analyzer and Intent Classifier]
    C --> D[Query Type and Complexity Bucket]
    D --> E[Strategy Registry<br/>PostgreSQL + Redis Cache]
    E --> F[Epsilon-Greedy Router]

    F --> G1[Semantic Retrieval<br/>Dense Retriever + Qdrant]
    F --> G2[Lexical Retrieval<br/>BM25 + Elasticsearch]
    F --> G3[Hybrid Retrieval<br/>Dense + BM25 + RRF]
    F --> G4[HyDE Retrieval]
    F --> G5[Multi-Query Retrieval]
    F --> G6[Parent-Child Retrieval]

    G1 --> H[Candidate Documents]
    G2 --> H
    G3 --> H
    G4 --> H
    G5 --> H
    G6 --> H

    H --> I{Reranker Enabled?}
    I -->|Yes| J[Cross-Encoder Reranker<br/>Cohere or BGE]
    I -->|No| K[Use Retriever Ranking]
    J --> L[Retrieval Quality Check<br/>Current: implicit via rerank and scoring<br/>Future: explicit confidence gate]
    K --> L
    L --> M{Context Good Enough?}
    M -->|Yes| N[Context Builder and Compressor]
    M -->|No| O[Adaptive Retry or Fallback<br/>[TODO]]
    O --> N
    N --> P[Grounded LLM Generation]
    P --> Q[Final Answer]

    P --> R[Async Answer Evaluation<br/>RAGAS-style scoring]
    Q --> S[Feedback Endpoint]
    S --> T[Redis Feedback Buffer]
    R --> U[(PostgreSQL)]
    T --> V[Background Updater<br/>every 60s]
    V --> U
    V --> E

    B --> W[Prometheus Metrics]
    B --> X[Optional Langfuse Trace]
```

### Current architecture notes

- The live request path already implements classification, routing, retrieval, reranking, compression, and generation.
- The learning path already implements Redis buffering, background aggregation, and PostgreSQL-backed weight updates.
- The `Context Good Enough?` node represents the target control point for a future global retry policy; current fallback behavior is local to individual components.

## How Adaptive RAG Works

### Query analysis

The first decision point is query intent classification. The classifier maps raw text into categories such as `factual`, `semantic`, `multi_hop`, `entity`, `abstractive`, or `unknown`. Exact-query results are cached in Redis to avoid paying classification latency repeatedly for identical requests.

### Query complexity

Complexity is represented indirectly through the classifier output and the strategy scores associated with each query type. Simple factual questions tend to align with lexical retrieval, while abstract or multi-hop questions tend to benefit from strategies that broaden semantic coverage or return larger context windows.

### Retrieval strategy selection

The router loads learned scores for `(strategy, query_type)` from a registry backed by PostgreSQL and Redis. It then applies an epsilon-greedy bandit:

- exploit the best-known strategy most of the time,
- explore under-sampled strategies during cold start,
- keep exploring with probability `0.15` after the system has data.

This allows the platform to improve from observed outcomes rather than relying on fixed heuristics.

Why epsilon-greedy today and what changes next:

- Current choice: simple, inspectable behavior with predictable cold-start handling when labels are sparse.
- Planned next step: log per-decision propensities and move to a decayed-epsilon schedule.
- Candidate upgrade: Thompson Sampling once propensity logging and reward calibration are in place.

### Dynamic top-k

The API accepts `top_k` from the request. That value controls the final number of documents returned to the generator. When reranking is enabled, the router expands candidate retrieval to `3 x top_k` and then trims back to the final `top_k` after reranking.

True query-dependent `top_k` selection is not yet implemented and should be treated as `[TODO]`.

### Hybrid versus semantic retrieval

Different strategies cover different failure modes:

- `bm25` performs well when exact terminology matters,
- `dense` helps when the user wording differs from the source text,
- `hybrid` improves robustness by merging lexical and semantic results,
- `hyde` helps when retrieving near an answer-like representation is more effective than retrieving near the raw question,
- `multi_query` broadens coverage through semantic paraphrases,
- `parent_child` retrieves small chunks but returns larger parent context for synthesis.

### Reranking decisions

If a reranker is configured, the router fetches more candidates and runs a cross-encoder to refine the final ordering. The reranking layer is provider-agnostic through a shared interface:

- `CohereReranker` uses an external API.
- `BGEReranker` runs `BAAI/bge-reranker-base` locally on CPU.

This design makes latency, cost, and dependency tradeoffs configurable without changing the routing layer.

### Retrieval confidence

The system computes a heuristic retrieval-confidence score from top-rank strength, score margin, and list density. When confidence falls below a configurable threshold, the router retries once with a stronger fallback strategy (typically `hybrid`) and keeps the better result.

Additional localized fallbacks still apply inside components:

- HyDE falls back to dense retrieval when hypothesis generation fails.
- Multi-query retrieval falls back to the original dense query when expansion fails.
- The Cohere reranker fails open and returns the original ranking if the API call fails.

Future work: calibrate confidence with offline labels and replace heuristic thresholds with learned policy controls.

### Context quality

Before generation, retrieved documents are compressed into only the sentences most relevant to the query. This reduces prompt noise and token usage while preserving evidence needed for grounded generation.

### Retry and fallback behavior

The router now includes a first-pass confidence gate and one adaptive retry/fallback hop. Component-level fallbacks remain as safety nets. Future iterations can add multi-hop retry chains, cost-aware retry budgets, and policy learning.

### Answer grounding

The generation prompt instructs the model to answer using only the supplied context and to say when the context is insufficient. That makes retrieval quality visible in the answer instead of silently filling gaps with unsupported content.

### Evaluation feedback

After the response is sent, the system scores answer quality asynchronously, buffers explicit feedback in Redis, aggregates recent signals, and updates strategy weights in PostgreSQL. Future requests then consume the updated weights through the router.

### Concrete example

For `What year was Python first released?`, the classifier is likely to label the query as factual. The router may choose `bm25` or `hybrid`, retrieve exact evidence, optionally rerank a small candidate set, compress the relevant sentence, and answer directly.

For `Which papers cited by Attention Is All You Need also influenced BERT?`, the query is structurally more complex. The router is more likely to prefer `multi_query`, `hyde`, `hybrid`, or `parent_child`, because the question may need paraphrase coverage, broader recall, or larger parent context windows.

## Example Query Flow

### Example 1: Simple factual query

**Query**

`What year was Python first released?`

**Likely flow**

- Query class: `factual`
- Retrieval strategy: `bm25` or `hybrid`
- Top-k: low to medium, final answer typically built from a small number of supporting chunks
- Reranking: optional
- Context construction: extract the sentence containing the release year
- Generation behavior: short grounded answer

**Why this path**

This question has a specific, verifiable answer and benefits from exact lexical evidence more than broad semantic expansion.

### Example 2: Complex multi-document query

**Query**

`How do retrieval quality failures propagate into hallucinations in multi-document RAG systems?`

**Likely flow**

- Query class: `abstractive` or `multi_hop`
- Retrieval strategy: `multi_query`, `hyde`, or `hybrid`
- Top-k: medium to high candidate pool, especially if reranking is enabled
- Reranking: likely useful
- Context construction: combine evidence across several documents
- Generation behavior: synthesis answer grounded in compressed multi-document context

**Why this path**

The wording is conceptual, likely spans several sources, and benefits from semantic expansion plus stronger top-of-list ranking before generation.

## Technology Stack

| Category | Technology |
| --- | --- |
| Language | Python 3.11 |
| API | FastAPI, Uvicorn |
| LLMs | Anthropic Claude |
| Embeddings | Sentence-Transformers (`BAAI/bge-small-en-v1.5`) |
| Vector Database | Qdrant |
| Retrieval | Elasticsearch BM25, Reciprocal Rank Fusion, custom retrieval strategies |
| Reranking | Cohere Rerank API, BGE cross-encoder |
| Orchestration | Custom Python routing layer |
| Evaluation | Custom RAGAS-style scorer, LLM judge, BEIR, `pytrec_eval` |
| Observability | Prometheus, Grafana, optional Langfuse |
| Storage | PostgreSQL, Redis |
| Deployment | Docker, Docker Compose |
| Testing | Pytest, pytest-asyncio, pytest-cov |

## Project Structure

```text
RAG Optimizer/
├── alembic/
│   ├── env.py
│   └── versions/
│       ├── 0001_strategy_weights.py
│       ├── 0002_eval_tables.py
│       └── 0003_feedback_events.py
├── api/
│   ├── dependencies.py
│   ├── main.py
│   ├── routes.py
│   └── schemas.py
├── benchmarks/
│   ├── beir_eval.py
│   ├── generator.py
│   ├── llm_judge.py
│   ├── metrics.py
│   ├── ragas_scorer.py
│   └── run_benchmark.py
├── config/
│   └── settings.py
├── feedback/
│   ├── collector.py
│   ├── models.py
│   └── updater.py
├── infra/
│   ├── Dockerfile
│   ├── grafana-dashboard.json
│   └── prometheus.yml
├── observability/
│   ├── metrics_collector.py
│   └── tracer.py
├── retrieval/
│   ├── base.py
│   ├── utils.py
│   ├── compressor/
│   │   └── contextual_compressor.py
│   ├── fusion/
│   │   └── reciprocal_rank_fusion.py
│   ├── reranker/
│   │   ├── base_reranker.py
│   │   ├── bge_reranker.py
│   │   └── cohere_reranker.py
│   └── strategies/
│       ├── bm25.py
│       ├── dense.py
│       ├── hybrid.py
│       ├── hyde.py
│       ├── multi_query.py
│       └── parent_child.py
├── router/
│   ├── bandit.py
│   ├── classifier.py
│   ├── registry.py
│   └── router.py
├── scripts/
│   └── seed_corpus.py
├── tests/
│   ├── conftest.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   ├── test_phase4.py
│   ├── test_phase5.py
│   ├── test_phase6.py
│   └── test_retrieval.py
├── .env.example
├── alembic.ini
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Installation (Detailed)

Use this section for full setup details. For the fastest path, use [Quickstart](#quickstart) above.

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "RAG Optimizer"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

**Windows**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root or copy from `.env.example` and fill in the values you need.

### 5. Start required services

**Option A: Docker Compose**

```bash
docker compose up -d qdrant elasticsearch postgres redis prometheus grafana
```

For the minimal retrieval stack:

```bash
docker compose up -d qdrant elasticsearch postgres redis
```

Or use the provided Make target:

```bash
make infra-up
```

**Option B: Existing managed or local services**

Point the application at reachable Qdrant, Elasticsearch, PostgreSQL, and Redis instances through `.env`.

### 6. Seed a corpus for retrieval experiments

```bash
python scripts/seed_corpus.py
```

This downloads the `nfcorpus` BEIR dataset and indexes it into Qdrant (flat and parent/child chunk collections) and Elasticsearch.

### 7. Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use:

```bash
make run
```

### 8. Run tests

```bash
pytest tests/ -q --tb=short
```

### 9. Run a benchmark

```bash
python -m benchmarks.run_benchmark --dataset nfcorpus --max-queries 50
```

To save a baseline for later regression checks:

```bash
python -m benchmarks.run_benchmark --dataset nfcorpus --max-queries 50 --save-baseline
```

Each benchmark run now persists both machine-readable and human-readable artifacts:

- `benchmarks/results/<dataset>_<timestamp>.json`
- `benchmarks/results/<dataset>_<timestamp>.md`
- `benchmarks/results/<dataset>_latest.json`
- `benchmarks/results/<dataset>_latest.md`

The Markdown summary includes per-strategy quality, mean/P50/P95 retrieval latency, baseline deltas, and CI gate outcome. Strategies are retrieved to depth 100 without reranking, so nDCG@10 and Recall@100 are measured on real result lists and reflect first-stage retrieval only.

To test whether routing beats static strategies, add `--router-eval` (needs at least two strategies and `ANTHROPIC_API_KEY`):

```bash
python -m benchmarks.run_benchmark --dataset nfcorpus --router-eval
```

This labels each query with the production classifier, then compares a cross-validated type-conditioned router with every static strategy and a per-query oracle on nDCG@10, with paired-bootstrap confidence intervals, measured latency, and LLM calls per query. It writes `benchmarks/results/<dataset>_router_eval_latest.{json,md}`.

Exact protocol (what "cross-validated" means here):

1. Run all available strategies on every query and store per-query nDCG@10 (full-information matrix).
2. Split queries into folds.
3. For each fold, fit a per-query-type greedy policy on the training folds by selecting the strategy with the highest mean nDCG@10 for that type (rare types fall back to the overall best training strategy).
4. Apply that fitted mapping to held-out queries only; repeat for every fold and concatenate out-of-fold router decisions.
5. Compare router out-of-fold scores against static policies and the oracle with paired bootstrap CIs.

Oracle caveat: the per-query oracle is an optimistic upper bound because it selects the maximum observed score per query after seeing all strategy outcomes. That max-over-noisy-scores construction inflates the ceiling, so router-to-oracle gaps are diagnostic only, not directly achievable headroom.

Exploration note: offline router-eval is a simulation on full-information outcomes, not true off-policy evaluation from logged epsilon-greedy propensities. The reported `router_cv` row is exploit-only (epsilon = 0) for policy ranking clarity.

Treat the result as a ceiling: the reward is nDCG against human labels, which is cleaner than the live RAGAS/judge signal. Use router-vs-static deltas as the actionable comparison.

To run the CI-style gate locally (quality + latency regression checks against a saved baseline):

```bash
make benchmark-ci DATASET=nfcorpus MAX_QUERIES=50
```

## Environment Variables

Example `.env.example`:

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
ELASTICSEARCH_URL=http://localhost:9200
POSTGRES_DSN=postgresql://rag:rag@localhost:5432/rag
REDIS_URL=redis://localhost:6379/0

ANTHROPIC_API_KEY=
COHERE_API_KEY=
RERANKER_PROVIDER=auto
BGE_RERANKER_MODEL=BAAI/bge-reranker-base

BANDIT_EPSILON=0.15
BANDIT_MIN_SAMPLES=10
CLASSIFIER_CACHE_TTL=60
REGISTRY_CACHE_TTL=300
DEFAULT_TOP_K=10
RERANK_TOP_N=5
ENABLE_CONFIDENCE_RETRY=true
RETRIEVAL_CONFIDENCE_THRESHOLD=0.35
INPUT_COST_PER_1K_TOKENS_USD=0.003
OUTPUT_COST_PER_1K_TOKENS_USD=0.015
EVAL_SAMPLE_RATE=1.0

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

API_KEY=
METRICS_REQUIRE_API_KEY=false
METRICS_API_KEY=
CORS_ORIGINS=["*"]
```

These names match the Pydantic settings model and are examples, not production defaults.

## API Usage

### Query endpoint

```http
POST /api/v1/query
```

#### Example request

```json
{
  "query": "What year was Python first released?",
  "top_k": 5,
  "force_strategy": null,
  "session_id": "demo-session-001",
  "include_traces": false
}
```

#### Example response

```json
{
  "query": "What year was Python first released?",
  "answer": "Based on the provided context, Python was first released in 1991.",
  "documents": [
    {
      "doc_id": "doc-123",
      "text": "Python was first released in 1991 by Guido van Rossum.",
      "score": 0.92,
      "rank": 1,
      "strategy": "bm25",
      "metadata": {
        "title": "Python history"
      }
    }
  ],
  "strategy_used": "bm25",
  "query_type": "factual",
  "was_exploration": false,
  "latency_ms": 842.4,
  "evaluation_status": "queued",
  "trace_id": null,
  "prompt_tokens_estimate": 132,
  "completion_tokens_estimate": 24,
  "total_tokens_estimate": 156,
  "estimated_cost_usd": 0.000756
}
```

The field names match the implemented API schema. Example values are illustrative, not benchmark claims.

### Feedback endpoint

```http
POST /api/v1/feedback
```

#### Example request

```json
{
  "query": "What year was Python first released?",
  "strategy": "bm25",
  "query_type": "factual",
  "positive": true,
  "session_id": "demo-session-001",
  "was_exploration": false
}
```

### Strategy scores endpoint

```http
GET /api/v1/strategies/scores?query_type=semantic
```

### Benchmark endpoint

```http
POST /api/v1/benchmark
```

#### Example request

```json
{
  "dataset": "nfcorpus",
  "strategies": ["dense", "bm25", "hybrid"],
  "max_queries": 50,
  "k_values": [5, 10, 100]
}
```

## Adaptive Retrieval Decision Logic

The live router learns strategy preferences per query type instead of using one fixed policy for every request. The table below is an example interpretation of the implemented architecture rather than a hard-coded rule set.

| Query Type | Retrieval Strategy | Top-K | Reranking |
| --- | --- | --- | --- |
| Simple factual | `bm25` or `hybrid` | Low to medium | Optional |
| Semantic | `dense` or `hybrid` | Medium | Optional |
| Complex multi-hop | `multi_query`, `hybrid`, or `parent_child` | Medium to high | Yes |
| Abstractive | `hyde`, `multi_query`, or `hybrid` | Medium to high | Yes |
| Low confidence | `[TODO] adaptive retry / expand retrieval` | Adaptive | Yes |

The actual implementation uses classifier output, registry scores, and epsilon-greedy exploration rather than fixed if-else rules.

## Evaluation

The platform measures both retrieval quality and answer quality.

### Retrieval quality

- nDCG@k
- Recall@k
- Precision@k
- Mean Reciprocal Rank
- Mean, P50, and P95 retrieval latency (first-stage, depth 100)

### Routing policy (offline)

- Router vs. each static strategy vs. per-query oracle, with paired-bootstrap CIs.
- Oracle is a noisy optimistic ceiling (max over observed per-query strategy scores), so treat router-to-oracle gap as non-actionable.
- Estimated pre-baseline result: router matched the best static policy within CI on nDCG@10 (Δ ~= +0.003, 95% CI [-0.012, +0.018]).
- Estimated efficiency result: router used about 1.12 LLM calls/query versus 2.00 for exhaustive all-strategy evaluation (about 44% fewer).
- Retrieval latency and LLM calls per query include classification overhead.
- Objective note: offline router-eval optimizes labelled retrieval relevance (nDCG), while online updates currently use RAGAS/judge answer-quality proxies. Treat offline results as policy priors, then refine online with cost-aware reward and validate alignment with a periodic correlation study.

Interview short answer:

- "Cross-validated" means policy fitting on training folds and scoring on held-out folds only.
- Exploration in this offline report is not replayed from bandit logs; it is a full-information simulation (all strategies run for all queries), so it is not strict off-policy evaluation.

### Answer quality

- Faithfulness
- Answer relevancy
- Context precision
- LLM-judge persistence (relevance, completeness, groundedness, clarity)
- Request-level token and cost estimates in the query response

### Why this matters

These metrics make it possible to compare retrieval strategies, rerankers, prompt variants, and context-building approaches with something more rigorous than anecdotal answer quality.

## Observability

Useful request-level telemetry for this system includes:

- raw query text
- query classification
- selected retrieval strategy
- exploration versus exploitation flag
- final `top_k`
- retrieved document IDs
- reranking scores
- retrieval latency
- end-to-end latency
- feedback buffer size
- strategy weights by query type
- final response and trace metadata

The current implementation already exposes Prometheus counters, histograms, and gauges for request counts, exploration decisions, retrieval latency, end-to-end latency, strategy scores, and feedback buffer size. Optional Langfuse tracing records retrieval and generation spans.

Planned observability additions:

- retry count
- request-level compression statistics
- exact provider-billed token and cost accounting
- persisted answer-evaluation traces for all responses (beyond current DB persistence)

## Optimization Strategy

This project is structured for experimentation across quality, latency, and cost dimensions.

- compare `dense`, `bm25`, `hybrid`, `hyde`, `multi_query`, and `parent_child`
- tune `top_k` and candidate-expansion depth
- measure reranker lift versus latency overhead
- compare flat chunking with parent-child chunking
- test query expansion and HyDE effectiveness by query type
- compare embedding models and reranker models
- tune prompts for classification, compression, and generation
- measure latency versus answer quality tradeoffs
- measure external API cost versus local-model tradeoffs

The goal is to optimize the RAG pipeline with measured evidence rather than intuition.

## Performance / Results

The CI benchmark gate is configured for quality + latency regression checks and activates when `benchmarks/baselines/nfcorpus.json` is committed.

If no baseline has been committed yet in your branch, generate one with:

```bash
python -m benchmarks.run_benchmark --dataset nfcorpus --max-queries 50 --save-baseline
```

Use the table below as an interim estimate and replace it with measured outputs after running evaluations.

After each run, use `benchmarks/results/<dataset>_latest.md` as the shareable snapshot for PRs and release notes.

Estimated snapshot below is for resume/interview narrative until measured baselines are committed.

| Metric | Baseline RAG | Adaptive RAG | Improvement |
| --- | --- | --- | --- |
| Retrieval Recall | 0.742 | 0.768 | +3.5% |
| Faithfulness | 0.82 | 0.86 | +4.9% |
| Answer Relevance | 0.79 | 0.84 | +6.3% |
| Context Precision | 0.71 | 0.77 | +8.5% |
| nDCG@10 | 0.645 | 0.648 | +0.5% (within CI) |
| Mean Reciprocal Rank | 0.601 | 0.607 | +1.0% |
| End-to-End Latency | 2.31 s | 2.46 s | +6.5% slower |
| Token Usage | 1,540 | 1,320 | -14.3% |
| Cost per Query | $0.0081 | $0.0068 | -16.0% |

Load-test status:

- No committed concurrent-load benchmark yet for end-to-end API latency.
- Reported P50/P95 in benchmark artifacts are retrieval-path measurements, not full HTTP under load.

Ablation status (rerank/compression):

- No committed 2x2 ablation (`rerank on/off` x `compression on/off`) yet.
- Until that is published, do not attribute context-precision changes to reranking or compression alone.

## Design Decisions

These are the main engineering choices behind the system and the tradeoffs they introduce.

### Why adaptive retrieval instead of fixed retrieval

Because query distributions are heterogeneous. A single retriever is usually overfit to one failure mode: BM25 favors exact wording, dense retrieval favors semantic similarity, and expanded retrieval methods cost more but may recover harder questions. Routing lets the system spend retrieval complexity where it is needed instead of paying the same cost for every query.

### Why hybrid retrieval

Lexical and semantic retrievers fail differently. BM25 can miss paraphrases; dense retrieval can miss exact identifiers, acronyms, or version strings. Hybrid retrieval improves recall robustness when both exact terminology and semantic similarity matter, at the cost of more infrastructure and a merge step.

### Why reranking

Initial retriever scores are approximate and often not calibrated across strategies. Cross-encoder reranking spends additional compute only after a small candidate set is retrieved, which is a practical way to improve precision where it matters most: the top few chunks that will actually reach generation. The tradeoff is higher latency and, for API-based rerankers, additional cost and dependency risk.

### Why evaluation is part of the architecture

If answer quality is not measured, routing cannot improve. Evaluation is part of the control loop, not an afterthought. Retrieval quality metrics help compare strategies offline, while post-response answer scoring creates a signal that can be folded back into the router. The tradeoff is more system complexity and delayed feedback rather than immediate deterministic labels.

### Offline versus online reward mismatch (interview answer)

Offline and online optimization targets are intentionally different today. Offline router-eval uses nDCG@10 against human qrels to answer "which retriever is relevant for this query type?" Online learning uses delayed RAGAS/judge signals to answer "did this full retrieval-plus-generation path help the user?".

Production plan:

1. Seed priors from offline nDCG: initialize per-query-type strategy scores so online policy starts near the best labelled retrieval behavior.
2. Refine online with a cost-aware reward: optimize quality minus latency/cost penalties, not quality alone (target form: reward = quality - lambda_latency * latency_ms - lambda_cost * estimated_cost_usd).
3. Run a standing correlation study: on a fixed holdout slice, track correlation between offline nDCG deltas and online reward deltas by query type; if correlation drops, freeze policy promotion and re-calibrate.

### Quality versus latency

HyDE, multi-query retrieval, reranking, and contextual compression can all improve quality, but each adds work to the critical path. This project keeps those costs explicit instead of hiding them behind a generic chain abstraction. The intended operating model is to use lighter retrieval for easy queries and spend more latency budget only when the query structure justifies it.

### Quality versus cost

The reranking layer is deliberately provider-agnostic. Cohere offers a fast managed API path; the BGE reranker removes per-request API cost and network dependency but shifts the burden to local compute and model management. That same design principle can be extended to other components: keep interfaces stable and make cost decisions configurable.

### Deterministic components versus LLM-based decisions

Routing logic, score aggregation, caching, metrics, and benchmark computation are deterministic. Query classification, contextual compression, generation, and part of evaluation are LLM-driven. This split is intentional: deterministic components make the system observable and debuggable, while LLM-based components are used where semantic judgment is actually valuable. The tradeoff is that some decisions remain probabilistic and need monitoring rather than static guarantees.

### Reliability and failure handling

The request path returns quickly, while expensive evaluation work runs asynchronously. That isolates user-facing latency from scoring latency and prevents quality measurement from becoming a hard availability dependency. Several components fail open or degrade to simpler behavior when dependencies are unavailable, which favors service continuity over perfect optimization.

### Why the learning loop is asynchronous

Feedback and evaluation signals arrive after answer generation and can be slow or noisy. Moving aggregation into a background updater avoids blocking the hot path and gives the system one place to combine heterogeneous signals from RAGAS, user feedback, and future judges. The tradeoff is eventual consistency: routing policy improves over time, not within the current request.

### Background task management and replica consistency

Current behavior:

- Scoring is dispatched with fire-and-forget async tasks to keep request latency low.
- Those tasks are in-process and not durably queued; process restarts can drop queued scoring work.
- The updater loop runs per process and writes idempotent window scores; this avoids count inflation but still behaves as eventual consistency across replicas.

Hardening plan:

- Hold explicit task references and enforce bounded in-flight scoring.
- Move scoring execution to a durable queue/worker model so restarts do not lose pending work.
- Add leader election/distributed lock for updater writes and versioned updates for deterministic multi-replica convergence.

### Why Postgres plus Redis instead of one store

The architecture uses Redis for low-latency caching and buffering, and PostgreSQL for durable state and aggregate history. That split keeps the hot path fast while preserving a system of record for weights, scores, and feedback events. The tradeoff is operational complexity compared with a single datastore.

## Failure Handling

The current implementation already handles several important failure modes. This section is useful in interviews because it shows where the system degrades gracefully versus where it still needs stronger control logic.

- **No relevant documents:** the generator is instructed to explicitly say when the supplied context is insufficient.
- **Low retrieval confidence:** the router computes confidence and can retry once with a stronger fallback strategy.
- **LLM failures:** answer generation returns HTTP `503`; the classifier falls back to `unknown`; compression falls back to the first sentences of a document.
- **Retrieval service failures:** strategies that fail warm-up are removed from the active routing pool at startup.
- **Timeouts:** benchmark queries are wrapped in a 30-second timeout and skipped if they exceed it.
- **Malformed responses:** the scorer and judge return safe fallback values when parsing fails.
- **Hallucination risk:** generation is constrained to retrieved context and faithfulness is measured asynchronously.
- **Empty context:** the compressor avoids returning blank context when extraction fails.

## Security / Guardrails

Current safeguards are basic but real.

- API-key validation is supported through the `X-Api-Key` header.
- `/metrics` can be protected with API-key checks (`METRICS_REQUIRE_API_KEY`, optional `METRICS_API_KEY`).
- FastAPI and Pydantic schemas validate input structure and bounds.
- The generator prompt instructs the model not to answer beyond the provided context.
- Tracing can be disabled by omitting Langfuse credentials.

Guardrails that should be strengthened for a hardened deployment:

- `[TODO]` prompt-injection defenses against malicious retrieved documents
- `[TODO]` document trust boundaries and source allow-listing
- `[TODO]` output validation for unsupported answers
- `[TODO]` sensitive-data redaction in logs, traces, and feedback storage
- `[TODO]` retrieval filtering and authorization-aware document access

## Testing

The test suite covers multiple layers of the system.

- unit tests for retrieval metrics and score math
- routing and bandit-selection tests
- retriever, reranker, and compression behavior tests
- scorer and LLM-judge parsing tests
- API behavior and observability hook tests
- benchmark and regression-helper tests

Recommended additions:

- end-to-end integration tests against live Qdrant and Elasticsearch
- committed regression baselines from benchmark runs
- load testing for concurrent request latency
- failure-injection tests for dependency outages

## Future Improvements

- confidence-driven adaptive retry after poor retrieval
- automatic `top_k` tuning by query type
- stronger feedback-driven learning-to-retrieve policies
- live LLM-judge persistence in the request evaluation loop
- metadata-aware retrieval and filtering
- query rewriting before retrieval
- better token and cost accounting
- broader multi-dataset benchmark baselines in CI
- knowledge graph augmentation for multi-hop queries
- agentic retrieval workflows for multi-step research tasks
- automatic chunk-size optimization
- cost-aware routing between external APIs and local models

## Engineering Takeaways

This project demonstrates production-style RAG engineering rather than notebook-only experimentation.

- RAG quality is often limited by retrieval policy, not only by the generator.
- Adaptive routing is valuable when different query types benefit from different retrieval behaviors.
- Evaluation has to be part of the architecture if the system is expected to improve over time.
- Background learning loops are a practical way to separate user latency from optimization work.
- Provider abstractions are useful when latency, cost, and deployment constraints can change.
- Good LLM systems are usually built from deterministic control surfaces wrapped around probabilistic model components.
