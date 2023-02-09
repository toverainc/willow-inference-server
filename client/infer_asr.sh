#!/bin/bash

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

if [ -r "$1" ]; then
  FILE="$1"
else
  echo "Need file"
  exit 1
fi

if [ "$2" ]; then
  URL="$2"
else
  URL="http://127.0.0.6:58000/api/asr"
fi

curl "$URL" -H "accept: application/json" \
-H "Content-Type: multipart/form-data" -F "file=@$FILE"
