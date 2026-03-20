# ============================================================
# WAN 2.2 i2v — RunPod B200 Template
# CUDA 12.8.1 | PyTorch cu128 | ComfyUI | FastAPI wrapper
# Porte: 8188 (ComfyUI) | 8000 (API)
# ============================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# ── variabili d'ambiente ─────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_PREFER_BINARY=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# ── dipendenze di sistema ────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# ── venv ─────────────────────────────────────────────────────
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── PyTorch cu128 (stabile, compatibile B200) ────────────────
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 && \
    rm -rf /root/.cache/pip

# ── ComfyUI ──────────────────────────────────────────────────
WORKDIR /workspace
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

# ── custom nodes ─────────────────────────────────────────────
WORKDIR /workspace/ComfyUI/custom_nodes

# ComfyUI-Manager
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

# ComfyUI_essentials — SimpleMath+, GetImageSize+, ImageFromBatch+
RUN git clone https://github.com/cubiq/ComfyUI_essentials.git && \
    cd ComfyUI_essentials && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

# ComfyUI-VideoHelperSuite — VHS_VideoCombine
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

# ComfyUI-Frame-Interpolation — RIFE VFI
RUN git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    cd ComfyUI-Frame-Interpolation && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

# ComfyUI-Impact-Pack — ImageListToImageBatch
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    cd ComfyUI-Impact-Pack && \
    pip install -r requirements.txt && \
    python install.py && \
    rm -rf /root/.cache/pip

# cg-use-everywhere — Anything Everywhere, Prompts Everywhere
RUN git clone https://github.com/chrisgoringe/cg-use-everywhere.git
# nessun requirements.txt

# ComfyLiterals — nodo Float
RUN git clone https://github.com/M1kep/ComfyLiterals.git
# nessun requirements.txt

# ── dipendenze FastAPI wrapper ────────────────────────────────
RUN pip install \
    fastapi \
    uvicorn[standard] \
    httpx \
    python-multipart && \
    rm -rf /root/.cache/pip

# ── copia il wrapper API ──────────────────────────────────────
WORKDIR /workspace
COPY main.py /workspace/main.py

# ── startup script ────────────────────────────────────────────
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8188 8000

CMD ["/start.sh"]
