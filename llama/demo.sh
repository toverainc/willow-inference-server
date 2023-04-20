#!/bin/bash

if [ "$1" ]; then
    SIZE="$1"
else
    SIZE="13B"
fi

python3 -m fastchat.serve.cli --model-path /data/ml/vicuna-"$SIZE" --load-8bit