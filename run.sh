#!/bin/bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

# Test for local environment file and use any overrides
if [ -r .env ]; then
    echo "Startup - using configuration overrides from .env file"
    . .env
else
    echo "Startup - using default configuration values"
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
LOG_LEVEL=${LOG_LEVEL:-info}

# Media port range
# WebRTC dynamically negotiates UDP ports for each session
# You should keep this as small as possible for expected WebRTC connections
MEDIA_PORT_RANGE=${MEDIA_PORT_RANGE:-10000-10050}

# Listen IP
LISTEN_IP=${LISTEN_IP:-0.0.0.0}

# GPUS - WIP for docker compose
GPUS=${GPUS:-"all"}

# Allow forwarded IPs. This is a list of hosts to allow parsing of X-Forwarded headers from
FORWARDED_ALLOW_IPS=${FORWARDED_ALLOW_IP:-127.0.0.1}

# Shared memory size for docker
SHM_SIZE=${SHM_SIZE:-1gb}

TAG=${TAG:-latest}
NAME=${NAME:wis}

set +a

# Temp for hacky willow config
mkdir -p nginx/static/audio

if [ ! -d models ]; then
    echo "Models not found. Downloading, please wait..."
    ./download_models.sh
fi

# Make sure we have it just in case
mkdir -p custom_speakers

# Migrate existing certs
for i in cert key; do
    PEM="$i".pem
    if [ -r "$PEM" ]; then
        mv "$PEM" nginx/
    fi
done

# Check for new certs
if [ ! -r nginx/cert.pem ] || [ ! -r nginx/key.pem ]; then
    echo "No SSL cert found - you need to run ./gen_cert.sh"
    exit 1
fi

# Compatibility with former docs and practice
if [ $1 ]; then
    docker compose "$@"
else
    docker compose up
fi
