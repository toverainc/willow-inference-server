#!/bin/bash
set -e

if [ "$TTS_MODEL_NAME" ]; then
    echo "Using coqui model $TTS_MODEL_NAME"
    python3 TTS/server/server.py --model_name "$TTS_MODEL_NAME" --use_cuda true
else
    echo "Using default coqui model"
    python3 TTS/server/server.py --use_cuda true
fi