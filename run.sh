#!/bin/bash

if [ "$1" ]; then
    export CUDA_VISIBLE_DEVICES="$1"
else
    export CUDA_VISIBLE_DEVICES="0,1"
fi

docker run --rm -it --gpus all --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $PWD:/app -v $PWD/cache:/root/.cache -e CUDA_VISIBLE_DEVICES \
    -p 19000:8000 air-infer-api:latest \
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
