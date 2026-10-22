#!/usr/bin/env bash
# Deploy the Adaptive RAG Platform to Google Cloud Run.
#
# Prerequisites:
#   gcloud auth login && gcloud auth configure-docker
#   All required services must be running (see INFRASTRUCTURE below).
#
# Usage:
#   export GCP_PROJECT_ID=my-project
#   export ANTHROPIC_API_KEY=sk-ant-...
#   export QDRANT_URL=https://your-cluster.cloud.qdrant.io
#   ... (see required env vars below)
#   bash scripts/deploy-cloudrun.sh

set -euo pipefail

# ── Required env vars ────────────────────────────────────────────────────────
: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}"
: "${QDRANT_URL:?Set QDRANT_URL (e.g. https://your-cluster.cloud.qdrant.io)}"
: "${QDRANT_API_KEY:?Set QDRANT_API_KEY}"
: "${ELASTICSEARCH_URL:?Set ELASTICSEARCH_URL (Elastic Cloud endpoint)}"
: "${POSTGRES_DSN:?Set POSTGRES_DSN (postgresql://user:pass@host/db)}"
: "${REDIS_URL:?Set REDIS_URL (redis://user:pass@host:port)}"
: "${API_KEY:?Set API_KEY — all /api/v1/* requests must send X-Api-Key: this-value}"

# ── Optional env vars (have sensible defaults) ────────────────────────────────
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${CLOUD_RUN_SERVICE:-adaptive-rag}"
IMAGE="gcr.io/${GCP_PROJECT_ID}/${SERVICE_NAME}"
COHERE_API_KEY="${COHERE_API_KEY:-}"
LANGFUSE_PUBLIC_KEY="${LANGFUSE_PUBLIC_KEY:-}"
LANGFUSE_SECRET_KEY="${LANGFUSE_SECRET_KEY:-}"
CORS_ORIGINS="${CORS_ORIGINS:-*}"

echo "==> Building image ${IMAGE}:latest"
docker build -f infra/Dockerfile -t "${IMAGE}:latest" .

echo "==> Pushing to Container Registry"
docker push "${IMAGE}:latest"

echo "==> Deploying to Cloud Run (region: ${REGION})"
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}:latest" \
  --region "${REGION}" \
  --platform managed \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10 \
  --port 8000 \
  --allow-unauthenticated \
  --timeout 300 \
  --set-env-vars "\
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY},\
COHERE_API_KEY=${COHERE_API_KEY},\
QDRANT_URL=${QDRANT_URL},\
QDRANT_API_KEY=${QDRANT_API_KEY},\
ELASTICSEARCH_URL=${ELASTICSEARCH_URL},\
POSTGRES_DSN=${POSTGRES_DSN},\
REDIS_URL=${REDIS_URL},\
LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY},\
LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY},\
API_KEY=${API_KEY},\
CORS_ORIGINS=${CORS_ORIGINS}"

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --format="value(status.url)")

echo ""
echo "==> Deployment complete!"
echo "    URL:    ${SERVICE_URL}"
echo "    Docs:   ${SERVICE_URL}/docs"
echo "    Health: ${SERVICE_URL}/api/v1/health"
echo ""
echo "==> Smoke test"
curl -sf "${SERVICE_URL}/api/v1/health" | python -m json.tool

# ── INFRASTRUCTURE checklist ──────────────────────────────────────────────────
# Before deploying, ensure these managed services exist:
#
#   Qdrant Cloud   https://cloud.qdrant.io   (free tier: 1 GB)
#   Elastic Cloud  https://cloud.elastic.co  (free trial: 14 days)
#   Cloud SQL      gcloud sql instances create rag-db --database-version=POSTGRES_16
#   Upstash Redis  https://upstash.com       (free tier: 10K req/day)
#
# After first deploy, seed the corpus (run once):
#   gcloud run jobs create seed-corpus \
#     --image "${IMAGE}:latest" \
#     --command python --args scripts/seed_corpus.py \
#     --region "${REGION}" \
#     --set-env-vars "..."
#   gcloud run jobs execute seed-corpus --region "${REGION}"
