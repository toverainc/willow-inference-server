#!/bin/bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

# Test for local environment file and use any overrides
if [ -r .env ]; then
    echo "Using configuration overrides from .env file"
    . .env
else
    echo "Using default configuration values"
    touch .env
fi

#Import source the .env file
set -a
source .env

CHATBOT_PARAMS=${CHATBOT_PARAMS:-13B}

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

build_chatbot () {
    docker run --rm --gpus all --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $PWD:/app -v $PWD/cache:/root/.cache air-infer-api:latest \
        /app/chatbot/utils.sh install $CHATBOT_PARAMS
}

build_one openai/whisper-tiny
build_one openai/whisper-base
build_one openai/whisper-medium
build_one openai/whisper-large-v2
build_t5

if [ -d "chatbot/llama" ] || [ -r "chatbot/vicuna.tar.zstd" ]; then
    build_chatbot
fi
