# air-infer-api

Getting started:
```bash
# Ensure you have nvidia-container-toolkit and not nvidia-docker
# On Arch Linux:
yay -S libnvidia-container-tools libnvidia-container nvidia-container-toolkit

# Build docker container
./build.sh

# Download and quantize the models
./download_models.sh

# Make a directory?
mkdir audio

# Define your .env file with API keys?
# TODO!

# Run as a developer
./dev.sh

# OR, run in prod
./run.sh
```

You can view API documentation at http://[your host]/docs
