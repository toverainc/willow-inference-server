#!/bin/bash

# We need to pass docker socket so it can launch containers for model-navigator
docker run --rm -it -v $PWD:/app -p 127.0.0.5:8000:8000 airapi:latest
