#!/usr/bin/env bash
# Deploy the Adaptive RAG Platform to Azure Container Apps.
#
# Prerequisites:
#   az login
#   az extension add --name containerapp
#   All managed services must be provisioned (see INFRASTRUCTURE below).
#
# Usage:
#   export AZURE_SUBSCRIPTION_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
#   export AZURE_RESOURCE_GROUP=adaptive-rag-rg
#   export AZURE_LOCATION=eastus
#   export ANTHROPIC_API_KEY=sk-ant-...
#   ... (see required env vars below)
#   bash scripts/deploy-azure.sh

set -euo pipefail

# ── Required env vars ────────────────────────────────────────────────────────
: "${AZURE_SUBSCRIPTION_ID:?Set AZURE_SUBSCRIPTION_ID}"
: "${AZURE_RESOURCE_GROUP:?Set AZURE_RESOURCE_GROUP}"
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}"
: "${QDRANT_URL:?Set QDRANT_URL (e.g. https://your-cluster.cloud.qdrant.io)}"
: "${QDRANT_API_KEY:?Set QDRANT_API_KEY}"
: "${ELASTICSEARCH_URL:?Set ELASTICSEARCH_URL (Azure AI Search or Elastic Cloud endpoint)}"
: "${POSTGRES_DSN:?Set POSTGRES_DSN (postgresql://user:pass@host.postgres.database.azure.com/db)}"
: "${REDIS_URL:?Set REDIS_URL (rediss://user:pass@host.redis.cache.windows.net:6380)}"
: "${API_KEY:?Set API_KEY — all /api/v1/* requests must send X-Api-Key: this-value}"

# ── Optional env vars ────────────────────────────────────────────────────────
LOCATION="${AZURE_LOCATION:-eastus}"
APP_NAME="${CONTAINER_APP_NAME:-adaptive-rag}"
ACR_NAME="${AZURE_ACR_NAME:-adaptiveragacr}"
IMAGE="${ACR_NAME}.azurecr.io/${APP_NAME}"
COHERE_API_KEY="${COHERE_API_KEY:-}"
LANGFUSE_PUBLIC_KEY="${LANGFUSE_PUBLIC_KEY:-}"
LANGFUSE_SECRET_KEY="${LANGFUSE_SECRET_KEY:-}"
CORS_ORIGINS="${CORS_ORIGINS:-*}"

echo "==> Setting Azure subscription"
az account set --subscription "${AZURE_SUBSCRIPTION_ID}"

echo "==> Ensuring resource group exists"
az group create --name "${AZURE_RESOURCE_GROUP}" --location "${LOCATION}" --output none

echo "==> Ensuring Azure Container Registry exists"
az acr create \
  --resource-group "${AZURE_RESOURCE_GROUP}" \
  --name "${ACR_NAME}" \
  --sku Basic \
  --output none 2>/dev/null || true

echo "==> Building and pushing image to ACR"
az acr build \
  --registry "${ACR_NAME}" \
  --image "${APP_NAME}:latest" \
  --file infra/Dockerfile \
  .

echo "==> Ensuring Container Apps environment exists"
az containerapp env create \
  --name "${APP_NAME}-env" \
  --resource-group "${AZURE_RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --output none 2>/dev/null || true

echo "==> Deploying Container App"
az containerapp create \
  --name "${APP_NAME}" \
  --resource-group "${AZURE_RESOURCE_GROUP}" \
  --environment "${APP_NAME}-env" \
  --image "${IMAGE}:latest" \
  --registry-server "${ACR_NAME}.azurecr.io" \
  --cpu 2 \
  --memory 4Gi \
  --min-replicas 0 \
  --max-replicas 10 \
  --ingress external \
  --target-port 8000 \
  --env-vars \
    "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}" \
    "COHERE_API_KEY=${COHERE_API_KEY}" \
    "QDRANT_URL=${QDRANT_URL}" \
    "QDRANT_API_KEY=${QDRANT_API_KEY}" \
    "ELASTICSEARCH_URL=${ELASTICSEARCH_URL}" \
    "POSTGRES_DSN=${POSTGRES_DSN}" \
    "REDIS_URL=${REDIS_URL}" \
    "LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}" \
    "LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}" \
    "API_KEY=${API_KEY}" \
    "CORS_ORIGINS=${CORS_ORIGINS}" \
  --output table

APP_URL=$(az containerapp show \
  --name "${APP_NAME}" \
  --resource-group "${AZURE_RESOURCE_GROUP}" \
  --query "properties.configuration.ingress.fqdn" \
  --output tsv)

echo ""
echo "==> Deployment complete!"
echo "    URL:    https://${APP_URL}"
echo "    Docs:   https://${APP_URL}/docs"
echo "    Health: https://${APP_URL}/api/v1/health"
echo ""
echo "==> Smoke test"
curl -sf "https://${APP_URL}/api/v1/health" | python -m json.tool

# ── INFRASTRUCTURE checklist ──────────────────────────────────────────────────
# Provision these Azure services before running this script:
#
#   Azure Database for PostgreSQL (Flexible Server)
#     az postgres flexible-server create --name rag-db --resource-group ${AZURE_RESOURCE_GROUP} \
#       --location ${LOCATION} --admin-user raguser --admin-password <pass> --sku-name Standard_B1ms
#
#   Azure Cache for Redis
#     az redis create --name adaptive-rag-cache --resource-group ${AZURE_RESOURCE_GROUP} \
#       --location ${LOCATION} --sku Basic --vm-size c0
#
#   Qdrant Cloud (cloud-agnostic)   https://cloud.qdrant.io   (free tier: 1 GB)
#   Elastic Cloud (cloud-agnostic)  https://cloud.elastic.co  (free trial: 14 days)
#   -- OR --
#   Azure AI Search (replaces Elasticsearch for keyword retrieval)
#
# Seed the corpus after first deploy (run as a Container Apps Job):
#   az containerapp job create --name seed-corpus \
#     --resource-group ${AZURE_RESOURCE_GROUP} \
#     --environment ${APP_NAME}-env \
#     --image ${IMAGE}:latest \
#     --replica-timeout 3600 \
#     --command "python" "scripts/seed_corpus.py"
#   az containerapp job start --name seed-corpus --resource-group ${AZURE_RESOURCE_GROUP}

