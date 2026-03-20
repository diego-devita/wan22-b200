# WAN 2.2 i2v — RunPod B200 Template

ComfyUI + FastAPI wrapper for WAN 2.2 image-to-video generation on RunPod B200.

## Stack

- **CUDA 12.8.1** + **PyTorch cu128** — native Blackwell (B200) support
- **ComfyUI** with all custom nodes required by the workflow
- **FastAPI** wrapper exposing 3 HTTP endpoints
- **Automatic model download** on first boot via HuggingFace Hub

## Architecture

```
Image build (Dockerfile)
  ├── ComfyUI installed in /comfyui
  ├── Custom nodes installed in /comfyui/custom_nodes
  └── FastAPI wrapper installed in /app/main.py

Runtime (start.sh)
  ├── STEP 1 — first boot: copy /comfyui → /workspace/ComfyUI (Network Volume)
  ├── STEP 2 — download models to /workspace/ComfyUI/models/ if not present
  ├── STEP 3 — start ComfyUI on port 8188
  └── STEP 4 — start FastAPI on port 8000
```

`/workspace` is the RunPod Network Volume — persistent across pod restarts.  
`/comfyui` inside the image is only used as a source for the first-boot copy.

## Custom Nodes

| Node | Purpose |
|------|---------|
| ComfyUI-Manager | node management |
| ComfyUI_essentials | SimpleMath+, GetImageSize+, ImageFromBatch+ |
| ComfyUI-VideoHelperSuite | VHS_VideoCombine |
| ComfyUI-Frame-Interpolation | RIFE VFI (60fps interpolation) |
| ComfyUI-Impact-Pack | ImageListToImageBatch |
| cg-use-everywhere | Anything Everywhere, Prompts Everywhere |
| ComfyLiterals | Float node |

## Models Downloaded on First Boot

| File | Source | Size |
|------|--------|------|
| wan2.2_i2v_A14b_high_noise_lightx2v.safetensors | lightx2v/Wan2.2-Official-Models | 28.6 GB |
| wan2.2_i2v_A14b_low_noise_lightx2v.safetensors | lightx2v/Wan2.2-Official-Models | 28.6 GB |
| wan21UMT5XxlFP32_fp32.safetensors | LS110824/text_encoders | ~11 GB |
| Wan2_1_VAE_fp32.safetensors | Kijai/WanVideo_comfy | ~500 MB |
| 4x_foolhardy_Remacri.pth | FacehugmanIII/4x_foolhardy_Remacri | 67 MB |
| rife49.pth | auto-downloaded by ComfyUI-Frame-Interpolation | 21 MB |

**Total: ~70 GB** — Network Volume must be at least **100 GB**.

## RunPod Template Setup

| Field | Value |
|-------|-------|
| Container Image | `ghcr.io/YOURUSERNAME/wan22-b200:latest` |
| Expose HTTP Ports | `8188,8000` |
| Container Disk | `20 GB` |
| Volume Mount Path | `/workspace` |

## API Endpoints

### `POST /upload`
Upload an input image to ComfyUI.

```bash
curl -X POST https://PODID-8000.proxy.runpod.net/upload \
  -F "image=@photo.jpg"
# → {"filename": "abc123.jpg"}
```

### `POST /queue`
Queue a video generation job.

```bash
curl -X POST https://PODID-8000.proxy.runpod.net/queue \
  -F "filename=abc123.jpg" \
  -F "positive_prompt=A woman walking on the beach" \
  -F "duration_frames=81"
# → {"prompt_id": "xyz-..."}
# duration_frames: 81=5s | 161=10s | 241=15s | 321=20s
```

### `GET /result/{prompt_id}`
Poll for result. Returns `{"status": "pending"}` while processing, or the MP4 file when done.

```bash
curl https://PODID-8000.proxy.runpod.net/result/xyz-... \
  --output video.mp4
```

### `GET /health`
Check that both services are up.

```bash
curl https://PODID-8000.proxy.runpod.net/health
# → {"status": "ok", "comfyui": "reachable"}
```

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Image build — installs everything in /comfyui and /app |
| `start.sh` | Runtime entrypoint — copies ComfyUI, downloads models, starts services |
| `main.py` | FastAPI wrapper with /upload, /queue, /result, /health endpoints |
