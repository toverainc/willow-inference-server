#!/bin/bash
set -e

# Log level - acceptable values are debug, info, warning, error, critical. Suggest info or debug.
LOG_LEVEL=${LOG_LEVEL:-info}

set +a

# Temp for hacky willow config
mkdir -p nginx/static/audio

if [ ! -d models ]; then
    echo "Models not found. Downloading, please wait..."
    ./download_models.sh
fi

# Make sure we have it just in case
mkdir -p custom_speakers

gunicorn main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:19000 \
    --graceful-timeout 10 --forwarded-allow-ips "*" --log-level "$LOG_LEVEL" -t 0 --keep-alive 3600