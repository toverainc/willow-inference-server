#!/bin/bash

# Listen port
PORT="19000"

# Try to get local CF/private IP
IP=$(ip addr list | grep '10.2' | cut -d' ' -f6 | cut -d '/' -f1)

if [ -z "$IP" ]; then
    echo "Couldn't determine local IP - exiting"
    exit 1
else
    echo "Using listen IP $IP"
fi

# Performance/load params
# Gunicorn workers - unfortunately loads multiple copies of models
export WEB_CONCURRENCY="1"

# See inter_threads: https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html
export MODEL_THREADS="10"

# TODO: Improve cmdline args
if [ "$1" ]; then
    TAG="$1"
else
    TAG="latest"
fi

docker run --rm -it --gpus all --shm-size=64g --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $PWD:/app -v $PWD/cache:/root/.cache -e WEB_CONCURRENCY -e MODEL_THREADS \
    --name air-infer-api \
    -p "$IP":"$PORT":"$PORT" -p 10000-10300:10000-10300/udp air-infer-api:"$TAG" \
    gunicorn main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:"$PORT" --graceful-timeout 10 --forwarded-allow-ips '*' --log-level debug -t 0
