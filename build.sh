#!/bin/bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

if [ "$1" ]; then
    TAG="$1"
else
    TAG="latest"
fi

DOCKER_BUILDKIT=1 docker build -t willow-inference-server:"$TAG" .
