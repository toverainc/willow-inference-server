#!/bin/bash

docker run --rm -it --gpus all --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $PWD:/app -v $PWD/cache:/root/.cache -v "/data":"/data" air-infer-api:latest \
    /bin/bash