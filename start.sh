#!/usr/bin/env bash
set -e

echo "=== WAN 2.2 B200 Template — avvio ==="

# Symlink modelli dal Network Volume se montato
# Il Network Volume va montato su /runpod-volume
# I modelli devono stare in /runpod-volume/models/
if [ -d "/runpod-volume/models" ]; then
    echo "Network Volume trovato — collegamento modelli..."
    rm -rf /workspace/ComfyUI/models
    ln -sf /runpod-volume/models /workspace/ComfyUI/models
    echo "Modelli collegati da /runpod-volume/models"
else
    echo "ATTENZIONE: Network Volume non trovato in /runpod-volume"
    echo "I modelli devono essere presenti in /workspace/ComfyUI/models"
fi

# Avvia ComfyUI in background sulla porta 8188
echo "Avvio ComfyUI su porta 8188..."
cd /workspace/ComfyUI
python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --disable-auto-launch \
    --highvram \
    > /var/log/comfyui.log 2>&1 &

COMFY_PID=$!
echo "ComfyUI PID: $COMFY_PID"

# Aspetta che ComfyUI sia pronto (polling su /system_stats)
echo "Attesa avvio ComfyUI..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI pronto dopo ${i} secondi."
        break
    fi
    sleep 2
done

# Avvia il wrapper FastAPI sulla porta 8000
echo "Avvio FastAPI wrapper su porta 8000..."
cd /workspace
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    > /var/log/api.log 2>&1 &

API_PID=$!
echo "FastAPI PID: $API_PID"

echo "=== Servizi avviati ==="
echo "ComfyUI: http://0.0.0.0:8188"
echo "API:     http://0.0.0.0:8000"

# Mantieni il container vivo, muori se ComfyUI muore
wait $COMFY_PID
