#!/bin/bash

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

if [ -r "$1" ]; then
  IMAGE="$1"
else
  echo "Need image"
  exit 1
fi

if [ "$2" ]; then
  URL="$2"
else
  URL="http://127.0.0.6:58000/api/infer"
fi

MIME_TYPE=$(file -b --mime-type "$IMAGE")

MODEL="medvit-trt-fp32"

curl "$URL?model=$MODEL" -H "accept: application/json" \
-H "Content-Type: multipart/form-data" -F "file=@$IMAGE"
