import asyncio
import copy
import json
import uuid

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="WAN 2.2 i2v API")

COMFY_URL = "http://127.0.0.1:8188"

# Node ID del LoadImage nel tuo workflow
LOAD_IMAGE_NODE = "218"
POSITIVE_PROMPT_NODE = "243"
DURATION_NODE = "319"

# Il tuo workflow completo
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
    "319": {"inputs": {"value": 81}, "class_type": "PrimitiveInt", "_meta": {"title": "Durata Video (81=5s 161=10s 241=15s 321=20s)"}},
}


# ─────────────────────────────────────────────────────────────
# ENDPOINT 1 — /upload
# Carica l'immagine di input nella cartella input/ di ComfyUI
# Restituisce il nome file registrato da usare in /queue
# ─────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(image: UploadFile = File(...)):
    """
    Carica un'immagine su ComfyUI.
    Restituisce: { "filename": "abc123.jpg" }
    Usa il filename restituito nella chiamata a /queue.
    """
    image_bytes = await image.read()

    # Nome univoco per evitare collisioni tra richieste parallele
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
# Mette in coda il workflow con i parametri ricevuti
# Restituisce il prompt_id da usare in /result/{prompt_id}
# ─────────────────────────────────────────────────────────────
@app.post("/queue")
async def queue(
    filename: str = Form(...),
    positive_prompt: str = Form(...),
    duration_frames: int = Form(81),  # 81=5s | 161=10s | 241=15s | 321=20s
):
    """
    Mette in coda la generazione del video.
    - filename: nome restituito da /upload
    - positive_prompt: descrizione del video da generare
    - duration_frames: 81=5s, 161=10s, 241=15s, 321=20s (default 81)
    Restituisce: { "prompt_id": "..." }
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
# Polling sullo stato del job.
# Se completato, restituisce direttamente il video MP4.
# Se ancora in corso, restituisce lo stato.
# ─────────────────────────────────────────────────────────────
@app.get("/result/{prompt_id}")
async def result(prompt_id: str):
    """
    Controlla lo stato del job e scarica il video se pronto.
    Risposte possibili:
    - { "status": "pending" }  → job ancora in coda o in esecuzione
    - { "status": "error", "detail": "..." }  → job fallito
    - video MP4 direttamente come file  → job completato
    """
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{COMFY_URL}/history/{prompt_id}")
        r.raise_for_status()
        history = r.json()

    # Job non ancora in history → ancora in coda o in esecuzione
    if prompt_id not in history:
        return JSONResponse({"status": "pending"})

    job = history[prompt_id]

    # Controlla se c'è stato un errore
    status_str = job.get("status", {}).get("status_str", "")
    if status_str == "error":
        messages = job.get("status", {}).get("messages", [])
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(messages)},
        )

    outputs = job.get("outputs", {})

    # Cerca il video negli output — priorità: nodo 94 (60fps) → 95 (upscaled 16fps) → 80 (16fps raw)
    video_info = None
    for node_id in ["94", "95", "80"]:
        if node_id in outputs:
            node_out = outputs[node_id]
            # VHS_VideoCombine usa "gifs" come chiave anche per MP4
            for key in ("gifs", "videos"):
                if key in node_out and node_out[key]:
                    video_info = node_out[key][0]
                    break
        if video_info:
            break

    if not video_info:
        # Job completato ma nessun video trovato — restituisce debug
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": "Nessun video trovato negli output",
                "output_nodes": list(outputs.keys()),
            },
        )

    # Scarica il video da ComfyUI e lo restituisce direttamente al client
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
# Verifica che ComfyUI sia raggiungibile
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
