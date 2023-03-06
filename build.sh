#!/bin/bash

if [ "$1" ]; then
    TAG="$1"
else
    TAG="latest"
fi

docker build -t air-infer-api:"$TAG" .
