#!/usr/bin/env bash
set -e

echo "=== WAN 2.2 B200 Template — starting ==="

COMFYUI_IMAGE="/comfyui"
COMFYUI_VOLUME="/workspace/ComfyUI"

# ── STEP 1: First boot — copy ComfyUI from image to volume ──────
# /workspace is the RunPod persistent Network Volume.
# On first boot /workspace/ComfyUI does not exist yet.
if [ ! -d "${COMFYUI_VOLUME}" ]; then
    echo "[1/3] First boot — copying ComfyUI to /workspace..."
    cp -r "${COMFYUI_IMAGE}" "${COMFYUI_VOLUME}"
    echo "      OK — ComfyUI copied to ${COMFYUI_VOLUME}"
else
    echo "[1/3] ComfyUI already present in ${COMFYUI_VOLUME} — skipping"
fi

# Models are NOT downloaded automatically at startup.
# Use the /admin/models page in the web UI to download them on demand.

# ── STEP 2: Start ComfyUI ────────────────────────────────────────
echo "[2/3] Starting ComfyUI on port 8188..."
cd "${COMFYUI_VOLUME}"
python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --disable-auto-launch \
    --highvram \
    > /var/log/comfyui.log 2>&1 &

COMFY_PID=$!
echo "      ComfyUI PID: ${COMFY_PID}"

# Wait for ComfyUI to be ready
echo "      Waiting for ComfyUI..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo "      ComfyUI ready after $((i * 2)) seconds."
        break
    fi
    sleep 2
done

# ── STEP 3: Start FastAPI wrapper ────────────────────────────────
echo "[3/3] Starting FastAPI wrapper on port 8000..."
cd /app
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    > /var/log/api.log 2>&1 &

API_PID=$!
echo "      FastAPI PID: ${API_PID}"

echo "=== Services started ==="
echo "    ComfyUI: https://PODID-8188.proxy.runpod.net"
echo "    API:     https://PODID-8000.proxy.runpod.net"
echo "    Web UI:  https://PODID-8000.proxy.runpod.net/app"
echo "    Models:  https://PODID-8000.proxy.runpod.net/admin/models"

# Keep the container alive — exit if ComfyUI dies
wait ${COMFY_PID}
