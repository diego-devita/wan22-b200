#!/usr/bin/env bash
set -e

echo "=== WAN 2.2 B200 Template — starting ==="

COMFYUI_IMAGE="/comfyui"
COMFYUI_VOLUME="/workspace/ComfyUI"
MODELS_DIR="${COMFYUI_VOLUME}/models"

# ── STEP 1: First boot — copy ComfyUI from image to volume ──────
# /workspace is the RunPod persistent Network Volume.
# On first boot /workspace/ComfyUI does not exist yet.
if [ ! -d "${COMFYUI_VOLUME}" ]; then
    echo "[1/4] First boot — copying ComfyUI to /workspace..."
    cp -r "${COMFYUI_IMAGE}" "${COMFYUI_VOLUME}"
    echo "      OK — ComfyUI copied to ${COMFYUI_VOLUME}"
else
    echo "[1/4] ComfyUI already present in ${COMFYUI_VOLUME} — skipping"
fi

# ── STEP 2: Download models (skipped if already present) ────────
echo "[2/4] Checking models..."

download_model() {
    local dest_dir=$1
    local repo=$2
    local filename=$3

    mkdir -p "${dest_dir}"

    if [ ! -f "${dest_dir}/${filename}" ]; then
        echo "      Downloading: ${filename}..."
        HF_REPO="${repo}" HF_FILE="${filename}" HF_DIR="${dest_dir}" \
        python -c "
import os
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id=os.environ['HF_REPO'],
    filename=os.environ['HF_FILE'],
    local_dir=os.environ['HF_DIR'],
    local_dir_use_symlinks=False
)
print('      OK:', path)
"
    else
        echo "      Already present: ${filename}"
    fi
}

# Diffusion models
download_model "${MODELS_DIR}/diffusion_models" \
    "lightx2v/Wan2.2-Official-Models" \
    "wan2.2_i2v_A14b_high_noise_lightx2v.safetensors"

download_model "${MODELS_DIR}/diffusion_models" \
    "lightx2v/Wan2.2-Official-Models" \
    "wan2.2_i2v_A14b_low_noise_lightx2v.safetensors"

# Text encoder
download_model "${MODELS_DIR}/text_encoders" \
    "LS110824/text_encoders" \
    "wan21UMT5XxlFP32_fp32.safetensors"

# VAE
download_model "${MODELS_DIR}/vae" \
    "Kijai/WanVideo_comfy" \
    "Wan2_1_VAE_fp32.safetensors"

# Upscaler
download_model "${MODELS_DIR}/upscale_models" \
    "FacehugmanIII/4x_foolhardy_Remacri" \
    "4x_foolhardy_Remacri.pth"

# rife49.pth is managed by ComfyUI-Frame-Interpolation on first use

echo "[2/4] Models OK"

# ── STEP 3: Start ComfyUI ────────────────────────────────────────
echo "[3/4] Starting ComfyUI on port 8188..."
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

# ── STEP 4: Start FastAPI wrapper ────────────────────────────────
echo "[4/4] Starting FastAPI wrapper on port 8000..."
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

# Keep the container alive — exit if ComfyUI dies
wait ${COMFY_PID}
