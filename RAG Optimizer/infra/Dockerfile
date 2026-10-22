# ── Build stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download the BGE embedding model (~130 MB) so containers start in seconds, not minutes.
# Without this, the first cold start downloads from HuggingFace and Cloud Run's 10 s
# health-check deadline expires before the app is ready.
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('BAAI/bge-small-en-v1.5')"

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim

RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser/app

COPY --from=builder /root/.local /home/appuser/.local
# Model files — baked at build time, no HuggingFace download at runtime
COPY --from=builder /root/.cache/huggingface /home/appuser/.cache/huggingface

COPY --chown=appuser:appuser . .

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/home/appuser/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Azure Container Apps injects PORT; fall back to 8000 for local docker run
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
