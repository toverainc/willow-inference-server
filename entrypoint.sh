#!/usr/bin/env bash
set -e

# Log level - acceptable values are debug, info, warning, error, critical. Suggest info or debug.
LOG_LEVEL=${LOG_LEVEL:-debug}

FORWARDED_ALLOW_IPS=${FORWARDED_ALLOW_IPS:-127.0.0.1}

export TOKENIZERS_PARALLELISM=true

set +a

# Temp for hacky willow config
mkdir -p nginx/static/audio

# Make sure we have it just in case
mkdir -p custom_speakers

gunicorn main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:19000 \
    --graceful-timeout 10 --forwarded-allow-ips "$FORWARDED_ALLOW_IPS" --log-level "$LOG_LEVEL" \
    -t 0 --keep-alive 3600