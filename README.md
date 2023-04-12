# air-infer-api

Getting started:
```bash
# Ensure you have nvidia-container-toolkit and not nvidia-docker
# On Arch Linux:
yay -S libnvidia-container-tools libnvidia-container nvidia-container-toolkit

# Ubuntu/Debian/Fedora/etc: TO-DO

# Build docker container
./build.sh

# Download and quantize the models
./download_models.sh

# Define your .env file with API keys?
# TODO!

# Run
./run.sh
```

You can view API documentation at http://[your host]:19000/docs
