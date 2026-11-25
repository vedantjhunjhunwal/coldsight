# Azure deployment

`deploy.sh` builds the runtime image in Azure Container Registry and runs it on Azure Container Apps
with one always-on replica per dock camera. The camera URL is stored as a Container Apps secret.
Prometheus metrics are served at `/metrics`; point Azure Monitor managed Prometheus at it for
dashboards and alerts (for example, `coldsight_input_luma_psi > 0.25` for 10 minutes).

Not run from this repo's CI: it needs an Azure subscription (`az login`). The image itself is
built and smoke-tested locally with `docker build` / `docker run`.

New model versions: export with `scripts/export_quantize.py` (records SHA-256 and metrics in
`models/registry.json`), rebuild the image with a new tag, then
`az containerapp update -g $RG -n $APP --image $ACR.azurecr.io/coldsight:<tag>`; Container Apps
keeps the previous revision for rollback.
