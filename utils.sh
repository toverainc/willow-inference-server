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
LISTEN_IP=${LISTEN_IP:+${LISTEN_IP}:}

# GPUS - WIP for docker compose
GPUS=${GPUS:-"all"}

# Minimum WIS Nvidia driver version
WIS_MIN_NVIDIA_VER=525

# Recommended WIS Nvidia driver version
WIS_REC_NVIDIA_VER=535

# Allow forwarded IPs. This is a list of hosts to allow parsing of X-Forwarded headers from
FORWARDED_ALLOW_IPS=${FORWARDED_ALLOW_IPS:-127.0.0.1}

# Shared memory size for docker
SHM_SIZE=${SHM_SIZE:-1gb}

TAG=${TAG:-latest}
NAME=${NAME:wis}

COQUI_IMAGE=${COQUI_IMAGE:-ghcr.io/coqui-ai/tts}
COQUI_TAG=${COQUI_TAG:-v0.22.0}
NGINX_TAG=${NGINX_TAG:-1.25.4}

WIS_NGINX_IMAGE=${WIS_NGINX_IMAGE:-willow-inference-server-nginx}
WIS_NGINX_TAG=${NGINX_TAG}

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

sv_model() {
    echo "Setting up SV model..."
    python -c 'import transformers; feature_extractor=transformers.AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"); feature_extractor.save_pretrained("./models/microsoft-wavlm-base-plus-sv")'
    python -c 'import transformers; model=transformers.AutoModelForAudioXVector.from_pretrained("microsoft/wavlm-base-plus-sv"); model.save_pretrained("./models/microsoft-wavlm-base-plus-sv")'
}

build_one_whisper () {
    docker run --rm $DOCKER_GPUS --shm-size="$SHM_SIZE" --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $WIS_DIR:/app -v $WIS_DIR/cache:/root/.cache "$IMAGE":"$TAG" \
        /app/utils.sh whisper-model $1
}

build_sv () {
    docker run --rm $DOCKER_GPUS --shm-size="$SHM_SIZE" --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $WIS_DIR:/app -v $WIS_DIR/cache:/root/.cache "$IMAGE":"$TAG" \
        /app/utils.sh sv-model
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
    if [ ! -r nginx/cert.pem ]; then
        echo "No SSL cert found - you need to run ./utils.sh gen-cert your_hostname.your_domain"
        exit 1
    fi
}

gunicorn_direct() {
    docker run --rm -it $DOCKER_GPUS --shm-size="$SHM_SIZE" \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $WIS_DIR:/app -v $WIS_DIR/cache:/root/.cache  --env-file .env \
    --name "$NAME" \
    -p "$LISTEN_IP":"$LISTEN_PORT":"$LISTEN_PORT" -p "$MEDIA_PORT_RANGE":"$MEDIA_PORT_RANGE"/udp \
    "$IMAGE":"$TAG" \
    gunicorn main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:"$LISTEN_PORT" \
    --graceful-timeout 10 --forwarded-allow-ips "$FORWARDED_ALLOW_IPS" --log-level "$LOG_LEVEL" -t 0 \
    --keyfile nginx/key.pem --certfile nginx/cert.pem --ssl-version TLSv1_2
}

run_nginx_command() {
    # We need to make dir world writeable temporarily
    chmod 777 "$WIS_DIR/nginx"
    docker run --rm -it --user nginx --entrypoint /nginx/wrap_command.sh -v "$WIS_DIR/nginx":/nginx "$WIS_NGINX_IMAGE":"$WIS_NGINX_TAG" "$@"
    chmod 755  "$WIS_DIR/nginx"
}

run_nginx_command_root() {
    docker run --rm -it --user root --entrypoint /nginx/wrap_command.sh -v "$WIS_DIR/nginx":/nginx "$WIS_NGINX_IMAGE":"$WIS_NGINX_TAG" "$@"
}

gen_ec_key() {
    if [ ! -f nginx/x25519.pem ]; then
        run_nginx_command openssl genpkey -algorithm x25519 -out nginx/x25519.pem
    fi
}

gen_dh_param() {
    if [ ! -f nginx/dhparam.pem ]; then
        echo "Generating secure DH parameters, this can take a while..."
        run_nginx_command openssl dhparam -out nginx/dhparam.pem 2048
    fi
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

    run_nginx_command openssl req -x509 -newkey rsa:2048 -keyout nginx/key.pem -out nginx/cert.pem -sha256 -days 3650 \
        -nodes -subj "/CN=$1"

    gen_ec_key
    gen_dh_param
}

gen_nginx_auth() {
    # WIS Key Auth
    cp $WIS_DIR/nginx/auth.conf.template $WIS_DIR/nginx/auth.conf
    if [ "$WIS_API_KEY" ]; then
        KEY_LENGTH=${#WIS_API_KEY}
        if [ "$KEY_LENGTH" -lt 8 ]; then
            echo "You defined a WIS API Key but it's $KEY_LENGTH characters"
            echo "We will not start until your key is at least 8 characters - and it should be longer!"
            echo "You can generate a high quality random key with ./utils.sh gen-key"
            exit 1
        else
            echo "Generating WIS API Key authentication..."
            sed -i "s/%%DEFAULT%%/0/" "$WIS_DIR/nginx/auth.conf"
            sed -i "s/%%WIS_API_KEY%%/$WIS_API_KEY/" "$WIS_DIR/nginx/auth.conf"
        fi
    else
        sed -i "s/%%DEFAULT%%/1/" "$WIS_DIR/nginx/auth.conf"
        sed -i "s/%%WIS_API_KEY%%/unused/" "$WIS_DIR/nginx/auth.conf"
    fi

    # WIS Basic Auth
    run_nginx_command touch /nginx/htpasswd
    cp $WIS_DIR/nginx/auth-basic.conf.template $WIS_DIR/nginx/auth-basic.conf
    if [ -s "$WIS_DIR/nginx/htpasswd" ]; then
        echo "Enabling WIS HTTP Basic Auth..."
        sed -i "s/%%AUTH_BASIC%%/'Authentication'/" "$WIS_DIR/nginx/auth-basic.conf"
    else
        sed -i "s/%%AUTH_BASIC%%/off/" "$WIS_DIR/nginx/auth-basic.conf"
    fi
    run_nginx_command chmod 0600 /nginx/htpasswd
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
}

build_docker() {
    docker build -t "$IMAGE":"$TAG" .
    docker build . --build-arg NGINX_TAG="$NGINX_TAG" -f Dockerfile.nginx -t "$WIS_NGINX_IMAGE":"$WIS_NGINX_TAG"
}

shell() {
    docker run --rm -it $DOCKER_GPUS --shm-size="$SHM_SIZE" --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $WIS_DIR:/app -v $WIS_DIR/cache:/root/.cache "$IMAGE":"$TAG" \
        /usr/bin/env bash
}

download_models() {
    build_one_whisper tovera/wis-whisper-tiny
    build_one_whisper tovera/wis-whisper-base
    build_one_whisper tovera/wis-whisper-small
    build_one_whisper tovera/wis-whisper-medium
    build_one_whisper tovera/wis-whisper-large-v2
    build_sv
}

clean_cache() {
    sudo rm -rf nginx/cache cache/huggingface
}

clean_models() {
    sudo rm -rf models/*
}

detect_compute() {
    if command -v nvidia-smi &> /dev/null; then
        NVIDIA_DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0 | cut -d'.' -f1)
        if [ $NVIDIA_DRIVER_VER -ge $WIS_MIN_NVIDIA_VER ]; then
            DOCKER_GPUS="--gpus $GPUS"
            DOCKER_COMPOSE_FILE="docker-compose.yml"
        else
            echo "Nvidia driver version $NVIDIA_DRIVER_VER is not compatible with WIS"
            echo "You will need to upgrade to at least Nvidia driver version $WIS_MIN_NVIDIA_VER"
            echo "Willow recommends Nvidia driver version $WIS_REC_NVIDIA_VER or later"
            echo "We will continue in CPU mode in five seconds"
            sleep 5
            FORCE_CPU=1
        fi
    else
        echo "Nvidia driver not detected"
        FORCE_CPU=1
    fi

    # If FORCE_CPU is set by configuration or auto-detection above it wins
    if [ "$FORCE_CPU" ]; then
        echo "Using CPU for WIS"
        DOCKER_GPUS=""
        DOCKER_COMPOSE_FILE="docker-compose-cpu.yml"
        export FORCE_CPU
    fi
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

build-xtts)
    check_host
    docker build -t xtts:latest . -f Dockerfile.xtts
    echo "To use Coqui XTTS add COQUI_IMAGE=xtts and COQUI_TAG=latest to .env"
;;

clean-cache)
    clean_cache
;;

gen-auth)
    gen_nginx_auth
;;

gen-key)
    KEY=$(openssl rand -base64 32)
    echo "Set this in .env and use WAS to configure your clients with it:"
    echo "WIS_API_KEY=$KEY"
;;

gen-cert)
    check_host
    gen_cert $2
;;

htpasswd)
    shift
    run_nginx_command htpasswd "$@"
;;

useradd)
    shift
    run_nginx_command touch /nginx/htpasswd
    echo "Please type and confirm password for user $1"
    run_nginx_command htpasswd -B /nginx/htpasswd "$@"
;;

userdel)
    shift
    run_nginx_command htpasswd /nginx/htpasswd -D "$@"
;;

userlist)
    echo "Current users for basic authentication:"
    run_nginx_command cut -d':' -f1 /nginx/htpasswd
;;

freeze-requirements)
    check_container
    freeze_requirements
;;

whisper-model)
    whisper_model $2
;;

sv-model)
    sv_model
;;

gunicorn)
    dep_check
    check_host
    detect_compute
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
    detect_compute
    gen_ec_key
    gen_dh_param
    gen_nginx_auth
    # Always ensure nginx cache is writable
    mkdir -p nginx/cache
    run_nginx_command_root chown -R nginx /nginx/cache
    shift
    docker compose -f "$DOCKER_COMPOSE_FILE" up --remove-orphans "$@"
;;

stop|down)
    dep_check
    check_host
    detect_compute
    shift
    docker compose -f "$DOCKER_COMPOSE_FILE" down "$@"
;;

shell|docker)
    check_host
    detect_compute
    shell
;;

*)
    dep_check
    check_host
    detect_compute
    gen_ec_key
    gen_dh_param
    # We need to regen auth because users can bring up services here too
    gen_nginx_auth
    echo "Passing unknown argument directly to docker compose"
    docker compose -f "$DOCKER_COMPOSE_FILE" "$@"
;;

esac
