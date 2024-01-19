#!/bin/bash
set -e

if [ "$FORCE_CPU" ]; then
    COQUI_CUDA="false"
else
    COQUI_CUDA="true"
fi

export COQUI_TOS_AGREED=1

if [ -r "/xtts/main.py" ]; then
    echo "Starting coqui xtts"
    cd /xtts
    uvicorn main:app --host 0.0.0.0 --port 5002
else
    # Fix/suppress cudnn warning to not confuse people
    ln -sf /usr/local/lib/python3.10/dist-packages/torch/lib/libnvrtc-*.so.11.2 \
        /usr/local/lib/python3.10/dist-packages/torch/lib/libnvrtc.so

    if [ "$TTS_MODEL_NAME" ]; then
        echo "Using coqui model $TTS_MODEL_NAME"
        python3 TTS/server/server.py --model_name "$TTS_MODEL_NAME" --use_cuda "$COQUI_CUDA"
    else
        echo "Using default coqui model"
        python3 TTS/server/server.py --use_cuda "$COQUI_CUDA"
    fi
fi