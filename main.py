import copy, uuid
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="WAN 2.2 i2v API")

COMFY_URL = "http://127.0.0.1:8188"

# Node IDs from the workflow
LOAD_IMAGE_NODE = "218"
POSITIVE_PROMPT_NODE = "243"
DURATION_NODE = "319"

# Full workflow definition
WORKFLOW = {
    "80": {"inputs": {"frame_rate": 16, "loop_count": 0, "filename_prefix": "teacache", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True, "trim_to_audio": False, "pingpong": False, "save_output": True, "images": ["290", 0]}, "class_type": "VHS_VideoCombine", "_meta": {"title": "Initial 16FPS Video"}},
    "94": {"inputs": {"frame_rate": 60, "loop_count": 0, "filename_prefix": "Hunyuan/videos/30/vid", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True, "trim_to_audio": False, "pingpong": False, "save_output": True, "images": ["303", 0]}, "class_type": "VHS_VideoCombine", "_meta": {"title": "Final 60 FPS Video"}},
    "95": {"inputs": {"frame_rate": 16, "loop_count": 0, "filename_prefix": "Hunyuan/videos/24/vid", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True, "trim_to_audio": False, "pingpong": False, "save_output": True, "images": ["98", 0]}, "class_type": "VHS_VideoCombine", "_meta": {"title": "Upscaled 16 FPS video"}},
    "98": {"inputs": {"upscale_method": "lanczos", "width": ["220", 0], "height": ["219", 0], "crop": "center", "image": ["290", 0]}, "class_type": "ImageScale"},
    "154": {"inputs": {"model_name": "4x_foolhardy_Remacri.pth"}, "class_type": "UpscaleModelLoader"},
    "155": {"inputs": {"UPSCALE_MODEL": ["154", 0]}, "class_type": "Anything Everywhere"},
    "218": {"inputs": {"image": "PLACEHOLDER.jpg"}, "class_type": "LoadImage", "_meta": {"title": "Input Image"}},
    "219": {"inputs": {"value": "a*b", "a": ["222", 1], "b": ["305", 0]}, "class_type": "SimpleMath+", "_meta": {"title": "height"}},
    "220": {"inputs": {"value": "a*b", "a": ["222", 0], "b": ["305", 0]}, "class_type": "SimpleMath+", "_meta": {"title": "width"}},
    "222": {"inputs": {"image": ["223", 0]}, "class_type": "GetImageSize+"},
    "223": {"inputs": {"start": 0, "length": 1, "image": ["224", 0]}, "class_type": "ImageFromBatch+"},
    "224": {"inputs": {"images": ["290", 0]}, "class_type": "ImageListToImageBatch"},
    "229": {"inputs": {"unet_name": "wan2.2_i2v_A14b_high_noise_lightx2v.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "231": {"inputs": {"clip_name": "wan21UMT5XxlFP32_fp32.safetensors", "type": "wan", "device": "default"}, "class_type": "CLIPLoader"},
    "232": {"inputs": {"vae_name": "Wan2_1_VAE_fp32.safetensors"}, "class_type": "VAELoader"},
    "243": {"inputs": {"text": "A beautiful woman walking", "clip": ["231", 0]}, "class_type": "CLIPTextEncode", "_meta": {"title": "Positive Prompt"}},
    "244": {"inputs": {"text": "Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down", "clip": ["231", 0]}, "class_type": "CLIPTextEncode", "_meta": {"title": "Negative Prompt"}},
    "245": {"inputs": {"CONDITIONING": ["244", 0]}, "class_type": "Prompts Everywhere"},
    "260": {"inputs": {"shift": 11, "model": ["229", 0]}, "class_type": "ModelSamplingSD3"},
    "281": {"inputs": {"VAE": ["232", 0]}, "class_type": "Anything Everywhere"},
    "282": {"inputs": {"CLIP": ["231", 0]}, "class_type": "Anything Everywhere"},
    "290": {"inputs": {"samples": ["311", 0], "vae": ["232", 0]}, "class_type": "VAEDecode"},
    "303": {"inputs": {"ckpt_name": "rife49.pth", "clear_cache_after_n_frames": 5, "multiplier": 4, "fast_mode": False, "ensemble": True, "scale_factor": 1, "dtype": "float32", "torch_compile": False, "batch_size": 1, "frames": ["98", 0]}, "class_type": "RIFE VFI"},
    "304": {"inputs": {"add_noise": "enable", "noise_seed": 83899493743336, "steps": ["315", 0], "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0, "end_at_step": ["316", 0], "return_with_leftover_noise": "enable", "model": ["260", 0], "positive": ["314", 0], "negative": ["314", 1], "latent_image": ["314", 2]}, "class_type": "KSamplerAdvanced"},
    "305": {"inputs": {"Number": "1.5"}, "class_type": "Float"},
    "306": {"inputs": {"unet_name": "wan2.2_i2v_A14b_low_noise_lightx2v.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "307": {"inputs": {"shift": 11, "model": ["306", 0]}, "class_type": "ModelSamplingSD3"},
    "311": {"inputs": {"add_noise": "disable", "noise_seed": 0, "steps": ["315", 0], "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple", "start_at_step": ["316", 0], "end_at_step": 10000, "return_with_leftover_noise": "disable", "model": ["307", 0], "positive": ["314", 0], "negative": ["314", 1], "latent_image": ["304", 0]}, "class_type": "KSamplerAdvanced"},
    "314": {"inputs": {"width": 240, "height": 416, "length": ["319", 0], "batch_size": 1, "positive": ["243", 0], "negative": ["244", 0], "vae": ["232", 0], "start_image": ["218", 0]}, "class_type": "WanImageToVideo"},
    "315": {"inputs": {"value": 20}, "class_type": "PrimitiveInt"},
    "316": {"inputs": {"value": "a / 2", "a": ["315", 0]}, "class_type": "SimpleMath+"},
    "319": {"inputs": {"value": 81}, "class_type": "PrimitiveInt", "_meta": {"title": "Video Duration (81=5s 161=10s 241=15s 321=20s)"}},
}


# ─────────────────────────────────────────────────────────────
# ENDPOINT 1 — /upload
# Upload input image to ComfyUI input folder.
# Returns the registered filename to use in /queue.
# ─────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(image: UploadFile = File(...)):
    """
    Upload an image to ComfyUI.
    Returns: { "filename": "abc123.jpg" }
    Use the returned filename in the /queue call.
    """
    image_bytes = await image.read()

    # Unique name to avoid collisions between parallel requests
    ext = image.filename.rsplit(".", 1)[-1] if "." in image.filename else "jpg"
    unique_name = f"{uuid.uuid4().hex}.{ext}"

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{COMFY_URL}/upload/image",
            files={"image": (unique_name, image_bytes, image.content_type or "image/jpeg")},
            data={"overwrite": "true"},
        )
        r.raise_for_status()
        registered_name = r.json()["name"]

    return JSONResponse({"filename": registered_name})


# ─────────────────────────────────────────────────────────────
# ENDPOINT 2 — /queue
# Queue the workflow with the given parameters.
# Returns the prompt_id to use in /result/{prompt_id}.
# ─────────────────────────────────────────────────────────────
@app.post("/queue")
async def queue(
    filename: str = Form(...),
    positive_prompt: str = Form(...),
    duration_frames: int = Form(81),  # 81=5s | 161=10s | 241=15s | 321=20s
):
    """
    Queue a video generation job.
    - filename: name returned by /upload
    - positive_prompt: description of the video to generate
    - duration_frames: 81=5s, 161=10s, 241=15s, 321=20s (default 81)
    Returns: { "prompt_id": "..." }
    """
    workflow = copy.deepcopy(WORKFLOW)
    workflow[LOAD_IMAGE_NODE]["inputs"]["image"] = filename
    workflow[POSITIVE_PROMPT_NODE]["inputs"]["text"] = positive_prompt
    workflow[DURATION_NODE]["inputs"]["value"] = duration_frames

    client_id = str(uuid.uuid4())

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{COMFY_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        r.raise_for_status()
        data = r.json()

    if "error" in data:
        raise HTTPException(status_code=400, detail=str(data["error"]))

    return JSONResponse({"prompt_id": data["prompt_id"]})


# ─────────────────────────────────────────────────────────────
# ENDPOINT 3 — /result/{prompt_id}
# Poll job status.
# Returns the MP4 file directly when complete,
# or a status object if still processing.
# ─────────────────────────────────────────────────────────────
@app.get("/result/{prompt_id}")
async def result(prompt_id: str):
    """
    Check job status and download the video when ready.
    Possible responses:
    - { "status": "pending" }  → job still queued or running
    - { "status": "error", "detail": "..." }  → job failed
    - MP4 file directly  → job completed
    """
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{COMFY_URL}/history/{prompt_id}")
        r.raise_for_status()
        history = r.json()

    # Job not yet in history — still queued or running
    if prompt_id not in history:
        return JSONResponse({"status": "pending"})

    job = history[prompt_id]

    # Check for errors
    status_str = job.get("status", {}).get("status_str", "")
    if status_str == "error":
        messages = job.get("status", {}).get("messages", [])
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(messages)},
        )

    outputs = job.get("outputs", {})

    # Look for video in outputs — priority: node 94 (60fps) → 95 (upscaled 16fps) → 80 (16fps raw)
    video_info = None
    for node_id in ["94", "95", "80"]:
        if node_id in outputs:
            node_out = outputs[node_id]
            # VHS_VideoCombine uses "gifs" as key even for MP4
            for key in ("gifs", "videos"):
                if key in node_out and node_out[key]:
                    video_info = node_out[key][0]
                    break
        if video_info:
            break

    if not video_info:
        # Job completed but no video found — return debug info
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": "No video found in outputs",
                "output_nodes": list(outputs.keys()),
            },
        )

    # Download the video from ComfyUI and stream it directly to the client
    filename = video_info["filename"]
    subfolder = video_info.get("subfolder", "")

    params = {"filename": filename, "type": "output"}
    if subfolder:
        params["subfolder"] = subfolder

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.get(f"{COMFY_URL}/view", params=params)
        r.raise_for_status()
        video_bytes = r.content

    return StreamingResponse(
        iter([video_bytes]),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ─────────────────────────────────────────────────────────────
# ENDPOINT EXTRA — /health
# Check that ComfyUI is reachable.
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{COMFY_URL}/system_stats")
            r.raise_for_status()
        return {"status": "ok", "comfyui": "reachable"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "comfyui": "unreachable", "detail": str(e)},
        )
