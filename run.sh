#!/bin/bash
PORT="19000"

# Try to get local CF/private IP
IP=$(ip addr list | grep '10.2' | cut -d' ' -f6 | cut -d '/' -f1)

if [ -z "$IP" ]; then
    echo "Couldn't determine local IP - exiting"
    exit 1
else
    echo "Using listen IP $IP"
fi

# Currently loads four copies of all models... Hopefully there's a better way.
export WEB_CONCURRENCY="2"

# TODO: Improve cmdline args
if [ "$1" ]; then
    TAG="$1"
else
    TAG="latest"
fi

if [ "$2" ]; then
    export CUDA_VISIBLE_DEVICES="$1"
else
    export CUDA_VISIBLE_DEVICES="0"
fi

#     -p "$PORT":8000 -p 60000-60100:60000-60100/udp
# --ssl-keyfile="/app/key.pem" --ssl-certfile="/app/cert.pem"
# --loop uvloop --http httptools
docker run --rm -it --gpus all --shm-size=1g --ipc=host \
    -v $PWD:/app -v $PWD/cache:/root/.cache -e WEB_CONCURRENCY \
    --name air-infer-api \
    -p "$IP":"$PORT":"$PORT" -p 10000-10300:10000-10300/udp air-infer-api:"$TAG" \
    uvicorn main:app --host 0.0.0.0 --port "$PORT" --ws websockets --proxy-headers --forwarded-allow-ips '*'
