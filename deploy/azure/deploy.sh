#!/usr/bin/env bash
# Deploy the ColdSight service to Azure Container Apps (image built in Azure Container Registry).
# Usage: RTSP_URL='rtsp://user:pass@camera/stream' ./deploy/azure/deploy.sh
set -euo pipefail
RG=${RG:-coldsight-rg}; LOC=${LOC:-centralindia}; ACR=${ACR:-coldsightacr$RANDOM}
ENV_NAME=${ENV_NAME:-coldsight-env}; APP=${APP:-coldsight-dock1}
: "${RTSP_URL:?set RTSP_URL to the dock camera stream}"

az group create -n "$RG" -l "$LOC" -o none
az acr create -g "$RG" -n "$ACR" --sku Basic --admin-enabled true -o none
az acr build -r "$ACR" -t coldsight:v2 .                       # builds the Dockerfile in the cloud
az containerapp env create -g "$RG" -n "$ENV_NAME" -l "$LOC" -o none
az containerapp create -g "$RG" -n "$APP" --environment "$ENV_NAME" \
  --image "$ACR.azurecr.io/coldsight:v2" --registry-server "$ACR.azurecr.io" \
  --cpu 2 --memory 4Gi --min-replicas 1 --max-replicas 1 \
  --ingress external --target-port 8000 \
  --secrets rtsp-url="$RTSP_URL" --env-vars COLDSIGHT_SOURCE=secretref:rtsp-url COLDSIGHT_LINE_X=640 -o none
echo "Health: https://$(az containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv)/health"
