# WAN 2.2 i2v — RunPod B200 Template

ComfyUI + FastAPI wrapper for WAN 2.2 image-to-video generation on RunPod B200.

## Stack

- **CUDA 12.8.1** + **PyTorch cu128** — native Blackwell (B200) support
- **ComfyUI** with all custom nodes required by the workflow
- **FastAPI** HTTP wrapper with Basic Auth
- **Web UI** served directly by FastAPI — no CORS, no separate host to configure
- **On-demand model downloads** via the `/admin/models` page, with real-time progress

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

## RunPod Template Setup

| Field | Value |
|-------|-------|
| Container Image | `ghcr.io/YOURUSERNAME/wan22-b200:latest` |
| Expose HTTP Ports | `8000` |
| Container Disk | `20 GB` |
| Volume Mount Path | `/workspace` |

> ⚠️ **Do NOT expose port 8188.** ComfyUI has no authentication and must only be
> reachable internally. FastAPI proxies all ComfyUI calls over `127.0.0.1:8188`.

### Required environment variable

You **must** set `API_KEY` in the RunPod template environment variables before
deploying. Without it the server falls back to the default value `changeme`,
leaving the API open to anyone who knows your pod URL.

| Variable | Example value |
|----------|---------------|
| `API_KEY` | `w2v-xK9mPqR7nTz4bLjF` |

Use any random string of 20+ characters without spaces.  
To generate one in PowerShell: `[System.Web.Security.Membership]::GeneratePassword(24, 4)`

## Web UI

| URL | Description |
|-----|-------------|
| `https://PODID-8000.proxy.runpod.net/app` | Video generation client |
| `https://PODID-8000.proxy.runpod.net/admin/models` | Model manager |

Both pages require HTTP Basic Auth. **Username:** anything — **Password:** your `API_KEY` value.  
The browser remembers the credentials after the first login prompt.

## Models

Downloaded on demand from the `/admin/models` page.  
All files are stored in `/workspace/ComfyUI/models/` on the Network Volume — persistent across restarts.  
Network Volume must be at least **100 GB** (models total ~70 GB).

| File | URL | Size | Dest |
|------|-----|------|------|
| wan2.2_i2v_A14b_high_noise_lightx2v.safetensors | huggingface.co/lightx2v/Wan2.2-Official-Models | 28.6 GB | diffusion_models |
| wan2.2_i2v_A14b_low_noise_lightx2v.safetensors | huggingface.co/lightx2v/Wan2.2-Official-Models | 28.6 GB | diffusion_models |
| wan21UMT5XxlFP32_fp32.safetensors | huggingface.co/LS110824/text_encoders | 11 GB | text_encoders |
| Wan2_1_VAE_fp32.safetensors | huggingface.co/Kijai/WanVideo_comfy | 0.5 GB | vae |
| 4x_foolhardy_Remacri.pth | huggingface.co/FacehugmanIII/4x_foolhardy_Remacri | 0.07 GB | upscale_models |
| rife49.pth | auto — ComfyUI-Frame-Interpolation on first use | 21 MB | — |

Download progress is tracked in real time using the `Content-Length` response header.
If a server does not provide `Content-Length`, the UI shows bytes downloaded without a percentage.

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

All endpoints except `/api/health` require HTTP Basic Auth (`-u :API_KEY`).

### `GET /api/health`
```bash
curl https://PODID-8000.proxy.runpod.net/api/health
# → {"status": "ok", "comfyui": "reachable"}
```

### `POST /api/upload`
```bash
curl -u :your_api_key -X POST \
  https://PODID-8000.proxy.runpod.net/api/upload \
  -F "image=@photo.jpg"
# → {"filename": "abc123.jpg"}
```

### `POST /api/queue`
```bash
curl -u :your_api_key -X POST \
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
curl -u :your_api_key \
  https://PODID-8000.proxy.runpod.net/api/result/xyz-... \
  --output video.mp4
```

### `GET /api/admin/models`
```bash
curl -u :your_api_key \
  https://PODID-8000.proxy.runpod.net/api/admin/models
```

### `POST /api/admin/models/download/{filename}`
```bash
curl -u :your_api_key -X POST \
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
        ├── index.html      ← served at /app
        └── models.html     ← served at /admin/models
```

## Customizing Without Rebuild

Files placed on the Network Volume override the defaults baked into the image.
Edit them directly via SSH on the pod — no rebuild required.

| File | Default (in image) | Override (on volume) |
|------|--------------------|----------------------|
| models.json | `/app/models.json` | `/workspace/models.json` |
| index.html | `/app/www/index.html` | `/workspace/www/index.html` |
| models.html | `/app/www/models.html` | `/workspace/www/models.html` |

`main.py` has no override path — changes to it require a rebuild and redeploy.
