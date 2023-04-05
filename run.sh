#!/bin/bash

# Which docker image to run
IMAGE=${IMAGE:-air-infer-api}

# Listen port
LISTEN_PORT="19000"

# Media port range
# WebRTC dynamically negotiates UDP ports to for each session
# You should keep this as small as possible for WebRTC connections
MEDIA_PORT_RANGE="10000-10300"

# Listen IP
LISTEN_IP="0.0.0.0"

# GPUS
GPUS="all"

# API Key - if defined all requests will require the X-Api-Key header with the configured value
if [ -r .api_key ]; then
    API_KEY=$(cat .api_key)
fi

if [ -r .env ]; then
    echo "Using environment variables from env file"
    . .env
else
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
    gunicorn main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:"$LISTEN_PORT" --graceful-timeout 10 --forwarded-allow-ips '*' --log-level debug -t 0
