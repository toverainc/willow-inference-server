#!/bin/bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

if [ "$1" ]; then
    TAG="$1"
else
    TAG="latest"
fi

docker build -t air-infer-api:"$TAG" .
