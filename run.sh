#!/bin/bash
PORT="19000"
IP="10.200.0.202"

# Currently loads four copies of all models... Hopefully there's a better way.
export WEB_CONCURRENCY="4"

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
docker run --rm -it --gpus all --shm-size=1g --ipc=host \
    -v $PWD:/app -v $PWD/cache:/root/.cache -e CUDA_VISIBLE_DEVICES -e WEB_CONCURRENCY \
    --name air-infer-api \
    -p "$IP":"$PORT":"$PORT" -p 10000-10100:10000-10100/udp air-infer-api:"$TAG" \
    uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload --ssl-keyfile="/app/key.pem" --ssl-certfile="/app/cert.pem" --loop uvloop --http httptools --ws websockets
