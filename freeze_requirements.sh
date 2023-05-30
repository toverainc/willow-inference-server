#!/usr/bin/env bash

if [ ! -f /.dockerenv ]; then
    echo "This script is meant to be run inside the container - exiting"
    exit 1
fi

# Freeze
pip freeze > requirements.txt

# When using Nvidia docker images they include a bunch of invalid local refs - remove them
sed -i '/file:/d' requirements.txt

# Torch needs to be installed with the current CUDA version in the Docker image - remove them
sed -i '/torch/d' requirements.txt

# Remove auto-gptq because we install manually
sed -i '/auto-gptq/d' requirements.txt