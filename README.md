# WAN 2.2 i2v — RunPod B200 Template

ComfyUI + FastAPI wrapper for WAN 2.2 image-to-video generation on RunPod B200.

## Stack

- **CUDA 12.8.1** + **PyTorch cu128** — native Blackwell (B200) support
- **ComfyUI** with all custom nodes required by the workflow
- **FastAPI** HTTP wrapper with Basic Auth
- **Web UI** served directly by FastAPI (no CORS, no separate host)
- **On-demand model downloads** via the `/admin/models` page

## Architecture

```
Image build (Dockerfile)
  ├── ComfyUI installed in /comfyui
  ├── Custom nodes installed in /comfyui/custom_nodes
  └── app/ copied to /app  (main.py, models.json, www/)

Runtime (start.sh)
  ├── STEP 1 — first boot: copy /comfyui → /workspace/ComfyUI (Network Volume)
  ├── STEP 2 — start ComfyUI on port 8188
  └── STEP 3 — start FastAPI on port 8000
```

`/workspace` is the RunPod Network Volume — persistent across pod restarts.  
Models are **not** downloaded at startup. Use the `/admin/models` page instead.

## Web UI

| URL | Description |
|-----|-------------|
| `https://PODID-8000.proxy.runpod.net/app` | Video generation client |
| `https://PODID-8000.proxy.runpod.net/admin/models` | Model manager |

Both pages are served by FastAPI and protected by HTTP Basic Auth.  
**Username:** anything — **Password:** value of the `API_KEY` env var.

## RunPod Template Setup

| Field | Value |
|-------|-------|
| Container Image | `ghcr.io/YOURUSERNAME/wan22-b200:latest` |
| Expose HTTP Ports | `8188,8000` |
| Container Disk | `20 GB` |
| Volume Mount Path | `/workspace` |
| Environment Variable | `API_KEY=your_secret_key` |

Network Volume must be at least **100 GB** to hold all models (~70 GB total).

## Models

Downloaded on demand from the `/admin/models` page. All stored in `/workspace/ComfyUI/models/`.

| File | Repo | Size | Dest |
|------|------|------|------|
| wan2.2_i2v_A14b_high_noise_lightx2v.safetensors | lightx2v/Wan2.2-Official-Models | 28.6 GB | diffusion_models |
| wan2.2_i2v_A14b_low_noise_lightx2v.safetensors | lightx2v/Wan2.2-Official-Models | 28.6 GB | diffusion_models |
| wan21UMT5XxlFP32_fp32.safetensors | LS110824/text_encoders | 11 GB | text_encoders |
| Wan2_1_VAE_fp32.safetensors | Kijai/WanVideo_comfy | 0.5 GB | vae |
| 4x_foolhardy_Remacri.pth | FacehugmanIII/4x_foolhardy_Remacri | 0.07 GB | upscale_models |
| rife49.pth | auto — ComfyUI-Frame-Interpolation | 21 MB | — |

## Custom Nodes

| Node | Provides |
|------|----------|
| ComfyUI-Manager | node management |
| ComfyUI_essentials | SimpleMath+, GetImageSize+, ImageFromBatch+ |
| ComfyUI-VideoHelperSuite | VHS_VideoCombine |
| ComfyUI-Frame-Interpolation | RIFE VFI (60fps interpolation) |
| ComfyUI-Impact-Pack | ImageListToImageBatch |
| cg-use-everywhere | Anything Everywhere, Prompts Everywhere |
| ComfyLiterals | Float node |

## API Endpoints

All endpoints except `/api/health` require HTTP Basic Auth.

### `GET /api/health`
```bash
curl https://PODID-8000.proxy.runpod.net/api/health
# → {"status": "ok", "comfyui": "reachable"}
```

### `POST /api/upload`
```bash
curl -u :your_secret_key -X POST \
  https://PODID-8000.proxy.runpod.net/api/upload \
  -F "image=@photo.jpg"
# → {"filename": "abc123.jpg"}
```

### `POST /api/queue`
```bash
curl -u :your_secret_key -X POST \
  https://PODID-8000.proxy.runpod.net/api/queue \
  -F "filename=abc123.jpg" \
  -F "positive_prompt=A woman walking on the beach" \
  -F "duration_frames=81"
# → {"prompt_id": "xyz-..."}
# duration_frames: 81=5s | 161=10s | 241=15s | 321=20s
```

### `GET /api/result/{prompt_id}`
Returns `{"status": "pending"}` while processing, or the MP4 file when done.
```bash
curl -u :your_secret_key \
  https://PODID-8000.proxy.runpod.net/api/result/xyz-... \
  --output video.mp4
```

### `GET /api/admin/models`
```bash
curl -u :your_secret_key \
  https://PODID-8000.proxy.runpod.net/api/admin/models
```

### `POST /api/admin/models/download/{filename}`
```bash
curl -u :your_secret_key -X POST \
  https://PODID-8000.proxy.runpod.net/api/admin/models/download/Wan2_1_VAE_fp32.safetensors
```

## Repo Structure

```
wan22-b200/
├── Dockerfile
├── start.sh
├── README.md
└── app/
    ├── main.py
    ├── models.json
    └── www/
        ├── index.html      ← /app
        └── models.html     ← /admin/models
```

## Customizing Without Rebuild

Files in `/workspace` override the defaults baked into the image:

| File | Override path |
|------|---------------|
| `app/main.py` | — (requires rebuild) |
| `app/models.json` | `/workspace/models.json` |
| `app/www/index.html` | `/workspace/www/index.html` |
| `app/www/models.html` | `/workspace/www/models.html` |
