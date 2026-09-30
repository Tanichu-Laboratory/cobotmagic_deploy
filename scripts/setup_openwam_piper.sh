#!/usr/bin/env bash
set -euo pipefail
DEPLOY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PIPER_VENV=/workspace/project/OpenWAM/.venv-piper
export UV_PYTHON_INSTALL_DIR=/workspace/.python
export UV_CACHE_DIR=/workspace/.uv-cache
export UV_LINK_MODE=copy
export DS_BUILD_OPS=0
if [[ ! -x "$PIPER_VENV/bin/python" ]]; then
  uv venv --python 3.11 "$PIPER_VENV"
fi
# Frozen closure includes opencv-python-headless in place of GUI OpenCV.
# --no-deps preserves that intentional substitution in this headless container.
uv pip install --python "$PIPER_VENV/bin/python" --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -r "$DEPLOY_ROOT/scripts/openwam_piper_requirements.lock"
cd "$DEPLOY_ROOT"
HF_HUB_OFFLINE=0 "$PIPER_VENV/bin/python" -m cobotmagic_deployment.tools.download_openwam_piper
