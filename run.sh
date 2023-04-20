#!/bin/bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

# Test for local environment file and use any overrides
if [ -r .env ]; then
    echo "Startup - using configuration overrides from .env file"
    . .env
else
    echo "Startup - using default configuration values"
    touch .env
fi

#Import source the .env file
set -a
source .env
set +a

# Which docker image to run
IMAGE=${IMAGE:-air-infer-api}

# Listen port
LISTEN_PORT=${LISTEN_PORT:-19000}

# Log level - acceptable values are debug, info, warning, error, critical. Suggest info or debug.
LOG_LEVEL=${LOG_LEVEL:-info}

# Media port range
# WebRTC dynamically negotiates UDP ports for each session
# You should keep this as small as possible for expected WebRTC connections
MEDIA_PORT_RANGE=${MEDIA_PORT_RANGE:-10000-10300}

# Listen IP
LISTEN_IP=${LISTEN_IP:-0.0.0.0}

# GPUS
GPUS=${GPUS:-"all"}

# Allow forwarded IPs. This is a list of hosts to allow parsing of X-Forwarded headers from
FORWARDED_ALLOW_IPS=${FORWARDED_ALLOW_IP:-127.0.0.1}

# Shared memory size for docker
SHM_SIZE=${SHM_SIZE:-1gb}

TAG=${TAG:-latest}

# Temp for hacky sallow config
mkdir -p audio

if [ ! -d models ]; then
    echo "Models not found. Downloading, please wait..."
    ./download_models.sh
fi

# Make sure we have it just in case
mkdir -p custom_speakers

docker run --rm -it --gpus "$GPUS" --shm-size="$SHM_SIZE" --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $PWD:/app -v $PWD/cache:/root/.cache  --env-file .env \
    --name air-infer-api \
    -p "$LISTEN_IP":"$LISTEN_PORT":"$LISTEN_PORT" -p "$MEDIA_PORT_RANGE":"$MEDIA_PORT_RANGE"/udp "$IMAGE":"$TAG" \
    gunicorn main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:"$LISTEN_PORT" --graceful-timeout 10 --forwarded-allow-ips "$FORWARDED_ALLOW_IPS" --log-level "$LOG_LEVEL" -t 0
