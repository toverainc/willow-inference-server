#!/bin/bash

if [ "$1" ]; then
    export CUDA_VISIBLE_DEVICES="$1"
else
    export CUDA_VISIBLE_DEVICES="0"
fi

#     -p 19000:8000 -p 8081:8080 
docker run --rm -it --gpus all --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $PWD:/app -v $PWD/cache:/root/.cache -e CUDA_VISIBLE_DEVICES --name air-infer-api \
    --net host air-infer-api:latest \
    uvicorn main:app --host 0.0.0.0 --port 19000 --reload --ssl-keyfile="/app/key.pem" --ssl-certfile="/app/cert.pem"
