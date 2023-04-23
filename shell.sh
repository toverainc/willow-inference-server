#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

if [ "$1" ]; then
    TAG="$1"
else
    TAG="latest"
fi

docker run --rm -it --gpus all --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $PWD:/app -v $PWD/cache:/root/.cache air-infer-api:"$TAG" \
    /bin/bash