#!/bin/bash

# Which docker image to run
IMAGE=${IMAGE:-air-infer-api}

# Listen port
LISTEN_PORT="19000"

# Log level
LOG_LEVEL="info"

# Media port range
# WebRTC dynamically negotiates UDP ports to for each session
# You should keep this as small as possible for WebRTC connections
MEDIA_PORT_RANGE="10000-10300"

# Listen IP
LISTEN_IP="0.0.0.0"

# GPUS
GPUS="all"

# Allow forwarded IPs. This is a list of hosts to allow parsing of X-Forwarded headers from
FORWARDED_ALLOW_IPS="127.0.0.1"

# API Key - if defined all requests will require the X-Api-Key header with the configured value
if [ -r .api_key ]; then
    API_KEY=$(cat .api_key)
fi

if [ -r .env ]; then
    echo "Startup - using configuration overrides from .env file"
    . .env
else
    echo "Startup - using default configuration values"
    touch .env
fi

# TODO: Improve cmdline args
if [ "$1" ]; then
    TAG="$1"
else
    TAG="latest"
fi

docker run --rm -it --gpus "$GPUS" --shm-size=64g --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $PWD:/app -v $PWD/cache:/root/.cache  --env-file .env \
    --name air-infer-api \
    -p "$LISTEN_IP":"$LISTEN_PORT":"$LISTEN_PORT" -p "$MEDIA_PORT_RANGE":"$MEDIA_PORT_RANGE"/udp "$IMAGE":"$TAG" \
    gunicorn main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:"$LISTEN_PORT" --graceful-timeout 10 --forwarded-allow-ips "$FORWARDED_ALLOW_IPS" --log-level "$LOG_LEVEL" -t 0
