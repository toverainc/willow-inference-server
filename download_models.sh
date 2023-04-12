#!/bin/bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"
build_one () {
    docker run --rm --gpus all --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $PWD:/app -v $PWD/cache:/root/.cache air-infer-api:latest \
        /app/whisper.sh $1
}

build_t5 () {
    docker run --rm --gpus all --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $PWD:/app -v $PWD/cache:/root/.cache air-infer-api:latest \
        /app/speecht5.sh
}

build_toucan () {
    docker run --rm --gpus all --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $PWD:/app -v $PWD/cache:/root/.cache air-infer-api:latest \
        python deps/toucan/run_model_downloader.py
}

build_one openai/whisper-tiny
build_one openai/whisper-base
build_one openai/whisper-medium
build_one openai/whisper-large-v2
build_t5
