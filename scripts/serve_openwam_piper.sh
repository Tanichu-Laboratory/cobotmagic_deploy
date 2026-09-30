#!/usr/bin/env bash
set -euo pipefail
DEPLOY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OPENWAM_PYTHON="${OPENWAM_PYTHON:-/workspace/project/OpenWAM/.venv-piper/bin/python}"
cd "$DEPLOY_ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
exec "$OPENWAM_PYTHON" -m cobotmagic_deployment.servers.policy_server_openwam_piper "$@"
