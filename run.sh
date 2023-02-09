#!/bin/bash

docker run --rm -it --gpus all --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $PWD:/app -v $PWD/cache:/root/.cache \
    -p 19000:8000 air-infer-api:latest \
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
