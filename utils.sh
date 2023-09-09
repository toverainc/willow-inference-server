#!/usr/bin/env bash
set -e
WIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$WIS_DIR"

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

# Which docker image to run
IMAGE=${IMAGE:-willow-inference-server}

# HTTPS Listen port
LISTEN_PORT_HTTPS=${LISTEN_PORT_HTTPS:-19000}

# Listen port
LISTEN_PORT=${LISTEN_PORT:-19001}

# Log level - acceptable values are debug, info, warning, error, critical. Suggest info or debug.
LOG_LEVEL=${LOG_LEVEL:-debug}

# Media port range
# WebRTC dynamically negotiates UDP ports for each session
# You should keep this as small as possible for expected WebRTC connections
MEDIA_PORT_RANGE=${MEDIA_PORT_RANGE:-10000-10050}

# Listen IP
LISTEN_IP=${LISTEN_IP:-0.0.0.0}

# GPUS - WIP for docker compose
GPUS=${GPUS:-"all"}

# Detect GPU support
if command -v nvidia-smi &> /dev/null; then
    DOCKER_GPUS="--gpus $GPUS"
    DOCKER_COMPOSE_FILE="docker-compose.yml"
else
    echo "NVIDIA GPU Support not detected - using CPU"
    DOCKER_GPUS=""
    DOCKER_COMPOSE_FILE="docker-compose-cpu.yml"
fi

# Clean this up
if [ "$FORCE_CPU" ]; then
    echo "Forcing CPU per configuration"
    DOCKER_GPUS=""
    DOCKER_COMPOSE_FILE="docker-compose-cpu.yml"
fi

# Allow forwarded IPs. This is a list of hosts to allow parsing of X-Forwarded headers from
FORWARDED_ALLOW_IPS=${FORWARDED_ALLOW_IPS:-127.0.0.1}

# Shared memory size for docker
SHM_SIZE=${SHM_SIZE:-1gb}

TAG=${TAG:-latest}
NAME=${NAME:wis}

# c2translate config options
export CT2_VERBOSE=1
export QUANT="float16"

set +a

# Container or host?
# podman sets container var to podman, make docker act like that
if [ -f /.dockerenv ]; then
    export container="docker"
fi

check_container(){
    if [ "$container" ]; then
        return
    fi

    echo "You need to run this command inside of the container - you are on the host"
    exit 1
}

check_host(){
    if [ ! "$container" ]; then
        return
    fi

    echo "You need to run this command from the host - you are in the container"
    exit 1
}

whisper_model() {
    echo "Setting up WIS model $1..."
    MODEL="$1"
    MODEL_OUT=`echo $MODEL | sed -e 's,/,-,g'`

    #ct2-transformers-converter --force --model "$MODEL" --quantization "$QUANT" --output_dir models/"$MODEL_OUT"
    #python -c 'import transformers; processor=transformers.WhisperProcessor.from_pretrained("'$MODEL'"); processor.save_pretrained("./models/'$MODEL_OUT'")'
    git clone https://huggingface.co/"$MODEL" models/"$MODEL_OUT"
    rm -rf "$MODEL_OUT"/.git
}

t5_model() {
    echo "Setting up T5 model..."
    python -c 'import transformers; processor=transformers.SpeechT5Processor.from_pretrained("microsoft/speecht5_tts"); processor.save_pretrained("./models/microsoft-speecht5_tts")'
    python -c 'import transformers; model=transformers.SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts"); model.save_pretrained("./models/microsoft-speecht5_tts")'
    python -c 'import transformers; vocoder=transformers.SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan"); vocoder.save_pretrained("./models/microsoft-speecht5_hifigan")'
}

sv_model() {
    echo "Setting up SV model..."
    python -c 'import transformers; feature_extractor=transformers.AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"); feature_extractor.save_pretrained("./models/microsoft-wavlm-base-plus-sv")'
    python -c 'import transformers; model=transformers.AutoModelForAudioXVector.from_pretrained("microsoft/wavlm-base-plus-sv"); model.save_pretrained("./models/microsoft-wavlm-base-plus-sv")'
}

build_one_whisper () {
    docker run --rm $DOCKER_GPUS --shm-size="$SHM_SIZE" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $WIS_DIR:/app -v $WIS_DIR/cache:/root/.cache "$IMAGE":"$TAG" \
        /app/utils.sh whisper-model $1
}

build_t5 () {
    docker run --rm $DOCKER_GPUS --shm-size="$SHM_SIZE" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $WIS_DIR:/app -v $WIS_DIR/cache:/root/.cache "$IMAGE":"$TAG" \
        /app/utils.sh t5-model
}

build_sv () {
    docker run --rm $DOCKER_GPUS --shm-size="$SHM_SIZE" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $WIS_DIR:/app -v $WIS_DIR/cache:/root/.cache "$IMAGE":"$TAG" \
        /app/utils.sh sv-model
}

build_chatbot () {
    docker run --rm $DOCKER_GPUS --shm-size="$SHM_SIZE" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $WIS_DIR:/app -v $WIS_DIR/cache:/root/.cache "$IMAGE":"$TAG" \
        /app/chatbot/utils.sh install $CHATBOT_PARAMS
}

dep_check() {
    # Temp for hacky willow config
    mkdir -p nginx/static/audio

    if [ ! -d models ]; then
        echo "Models not found. You need to run ./utils.sh download-models - exiting"
        exit 1
    fi

    # Make sure we have it just in case
    mkdir -p speakers/custom_tts speakers/voice_auth nginx/cache cache

    # Check for new certs
    if [ ! -r nginx/cert.pem ] || [ ! -r nginx/key.pem ]; then
        echo "No SSL cert found - you need to run ./utils.sh gen-cert"
        exit 1
    fi

    # For unprivileged docker
    chmod 0666 nginx/key.pem nginx/cert.pem
}

gunicorn_direct() {
    docker run --rm -it $DOCKER_GPUS --shm-size="$SHM_SIZE" --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $WIS_DIR:/app -v $WIS_DIR/cache:/root/.cache  --env-file .env \
    --name "$NAME" \
    -p "$LISTEN_IP":"$LISTEN_PORT":"$LISTEN_PORT" -p "$MEDIA_PORT_RANGE":"$MEDIA_PORT_RANGE"/udp \
    "$IMAGE":"$TAG" \
    gunicorn main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:"$LISTEN_PORT" \
    --graceful-timeout 10 --forwarded-allow-ips "$FORWARDED_ALLOW_IPS" --log-level "$LOG_LEVEL" -t 0 \
    --keyfile nginx/key.pem --certfile nginx/cert.pem --ssl-version TLSv1_2
}

gen_cert() {
    if [ -z "$1" ]; then
        echo "You need to provide your domain/common name"
        exit 1
    fi

    # Remove old wis certs if present
    if [ -r cert.pem ] || [ -r key.pem ]; then
        echo "Removing old WIS certificate - enter password when prompted"
        sudo rm -f key.pem cert.pem
    fi

    openssl req -x509 -newkey rsa:2048 -keyout nginx/key.pem -out nginx/cert.pem -sha256 -days 3650 \
        -nodes -subj "/CN=$1"

    chmod 0666 nginx/key.pem nginx/cert.pem
}

freeze_requirements() {
    if [ ! -f /.dockerenv ]; then
        echo "This script is meant to be run inside the container - exiting"
        exit 1
    fi

    # Freeze
    pip freeze > requirements.txt

    # When using Nvidia docker images they include a bunch of invalid local refs - remove them
    sed -i '/file:/d' requirements.txt

    # When using Nvidia docker images they include polygraphy - remove it
    sed -i '/polygraphy/d' requirements.txt

    # Torch needs to be installed with the current CUDA version in the Docker image - remove them
    sed -i '/torch/d' requirements.txt

    # Remove auto-gptq because we install manually
    sed -i '/auto-gptq/d' requirements.txt
}

build_docker() {
    docker build -t "$IMAGE":"$TAG" .
}

shell() {
    docker run --rm -it $DOCKER_GPUS --shm-size="$SHM_SIZE" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $WIS_DIR:/app -v $WIS_DIR/cache:/root/.cache "$IMAGE":"$TAG" \
        /usr/bin/env bash
}

download_models() {
        CHATBOT_PARAMS=${CHATBOT_PARAMS:-13B}

    build_one_whisper tovera/wis-whisper-tiny
    build_one_whisper tovera/wis-whisper-base
    build_one_whisper tovera/wis-whisper-small
    build_one_whisper tovera/wis-whisper-medium
    build_one_whisper tovera/wis-whisper-large-v2
    build_t5
    build_sv

    if [ -d "chatbot/llama" ] || [ -r "chatbot/vicuna.tar.zstd" ]; then
        build_chatbot
    fi
}

clean_cache() {
    sudo rm -rf nginx/cache cache/huggingface
}

clean_models() {
    sudo rm -rf models/*
}

case $1 in

download-models)
    sudo rm -rf models
    download_models
;;

build-docker|build)
    check_host
    build_docker
;;

clean-cache)
    clean_cache
;;

gen-cert)
    check_host
    gen_cert $2
;;

freeze-requirements)
    check_container
    freeze_requirements
;;

whisper-model)
    whisper_model $2
;;

t5-model)
    t5_model
;;

sv-model)
    sv_model
;;

gunicorn)
    dep_check
    check_host
    gunicorn_direct
;;

install)
    check_host
    build_docker
    clean_models
    download_models
    clean_cache
    echo "Install complete - you can now start with ./utils.sh run"
;;

start|run|up)
    dep_check
    check_host
    shift
    docker compose -f "$DOCKER_COMPOSE_FILE" up --remove-orphans "$@"
;;

stop|down)
    dep_check
    check_host
    shift
    docker compose -f "$DOCKER_COMPOSE_FILE" down "$@"
;;

shell|docker)
    check_host
    shell
;;

*)
    dep_check
    check_host
    echo "Passing unknown argument directly to docker compose"
    docker compose -f "$DOCKER_COMPOSE_FILE" "$@"
;;

esac
