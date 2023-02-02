#!/bin/bash

docker run --rm -it -v $PWD:/app -p 127.0.0.6:58000:8000 air-infer-api:latest \
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload