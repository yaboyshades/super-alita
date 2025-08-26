#!/usr/bin/env bash
set -euo pipefail
echo "[Alita Setup] Initializing local environment"

mkdir -p adapters training_data logs data/chroma data/redis

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected"
else
  echo "GPU NOT detected (CPU mode)"
fi

echo "Building core service images"
docker compose build app || docker-compose build app

echo "(Optional) build auxiliary services"
docker compose build context-server swarm finetune || true

echo "Done. Launch with: docker compose up -d app"