#!/bin/bash

if [ -z "$1" ]; then
    echo "You need to provide your domain/common name"
    exit 1
fi

openssl req -x509 -newkey rsa:2048 -keyout nginx/key.pem -out nginx/cert.pem -sha256 -days 3650 -nodes -subj "/CN=$1"