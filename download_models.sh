#!/bin/bash

build_one () {
    docker run --rm --gpus all --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $PWD:/app -v $PWD/cache:/root/.cache air-infer-api:latest \
        /app/whisper.sh $1
}

build_one openai/whisper-base
build_one openai/whisper-tiny
build_one openai/whisper-medium
build_one openai/whisper-large-v2
