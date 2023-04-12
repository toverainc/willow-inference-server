#!/bin/bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

if [ "$1" ]; then
    TAG="$1"
else
    TAG="latest"
fi

rm -rf deps/toucan
mkdir -p deps
cd deps && git clone https://github.com/DigitalPhonetics/IMS-Toucan.git toucan && cd toucan && git checkout 5f1dce3
cd "$SCRIPT_DIR"

DOCKER_BUILDKIT=1 docker build -t air-infer-api:"$TAG" .
