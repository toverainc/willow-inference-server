# Willow Inference Server

Willow Inference Server (WIS) is an opinionated and targeted inference server instance. Our goal is to "automagically" enable self-hosting with open SOTA/best of breed models to enable speech and language tasks:

- Primarily targeting CUDA with support for low-end (old and cheap) devices such as the Tesla P4, GTX 1060, and up. Don't worry - it screams on an RTX 4090 too! [(See benchmarks)](#benchmarks)
- ASR. Heavy emphasis - Whisper optimized for very high quality as-close-to-real-time-as-possible speech recognition via a variety of means (WebRTC, POST a file, integration with devices and client applications, etc).
- TTS. Emerging support, primarily provided for assistant tasks and visually impaired users.
- LLM(s). Optionally pass input through a provided/configured LLM for question answering, chatbot, and assistant tasks (supports stacking with ASR and/or TTS).

With the goal of enabling democratization of this functionality WIS will detect available CUDA VRAM, compute platform support, etc and optimize and/or disable functionality automatically (currently in order - ASR, TTS, LLM). For all supported Whisper models (large-v2, medium, and base) current minimum supported hardware is GTX 1060 3GB (6GB for TTS). User applications are able to programatically select and configure Whisper models and parameters (beam size, language detection/translation, etc) on a per-request basis depending on the needs of the application to optimize response times.

Note that we are primarily targeting CUDA - the performance, cost, and power usage of cheap GPUs like the Tesla P4 and GTX 1060 is too good to ignore. We'll make our best effort to support CPU wherever possible for current and future functionality but our emphasis is on performant latency-sensitive tasks. In the event ROCm, Apple Neural, etc advances (compatible with our goals) we're open to supporting them as well.

## Getting started
```bash
# Ensure you have nvidia-container-toolkit and not nvidia-docker
# On Arch Linux:
yay -S libnvidia-container-tools libnvidia-container nvidia-container-toolkit

# Ubuntu/Debian/Fedora/etc: TO-DO

# Build docker container
./build.sh

# Download and quantize the models
./download_models.sh

# Run
./run.sh
```

You can view API documentation at http://[your host]:19000/docs

## Benchmarks

| Device   | Model    | Beam Size | Speech Duration (ms) | Inference Time (ms) | Realtime Multiple |
|----------|----------|-----------|----------------------|---------------------|-------------------|
| RTX 4090 | large-v2 | 5         | 3840                 | 140                 | 27x               |
| H100     | large-v2 | 5         | 3840                 | 294                 | 12x               |
| H100     | large-v2 | 5         | 10688                | 519                 | 20x               |
| H100     | large-v2 | 5         | 29248                | 1223                | 23x               |
| GTX 1060 | large-v2 | 5         | 3840                 | 1114                | 3x                |
| Tesla P4 | large-v2 | 5         | 3840                 | 1099                | 3x                |
| RTX 4090 | medium   | 1         | 3840                 | 84                  | 45x               |
| GTX 1060 | medium   | 1         | 3840                 | 588                 | 6x                |
| Tesla P4 | medium   | 1         | 3840                 | 586                 | 6x                |
| RTX 4090 | medium   | 1         | 29248                | 377                 | 77x               |
| GTX 1060 | medium   | 1         | 29248                | 1612                | 18x               |
| Tesla P4 | medium   | 1         | 29248                | 1730                | 16x               |
| RTX 4090 | base     | 1         | 180000               | 277                 | 648x (not a typo) |

When using WebRTC end-to-end latency in the browser and supported applications is the numbers above plus network latency - with the advantage being you can skip the "upload" portion as audio is streamed in realtime!

## Comparison Benchmarks

Raspberry Pi Benchmarks run on Raspberry Pi 4 4GB Debian 11.7 aarch64 with faster-whisper 0.5.1. Canakit 3 AMP USB-C power adapter and fan. All models int8 with ```OMP_NUM_THREADS=4``` set and language set as en. Same methodology as timing above with model load time excluded (WIS keeps models loaded). All inference time numbers rounded down. Max temperatures as reported by ```vcgencmd measure_temp``` were 57.9 C.

| Device   | Model    | Beam Size | Speech Duration (ms) | Inference Time (ms) | Realtime Multiple |
|----------|----------|-----------|----------------------|---------------------|-------------------|
| Pi       | tiny     | 1         | 3840                 | 3333                | 1.15x             |
| Pi       | base     | 1         | 3840                 | 6207                | 0.62x             |
| Pi       | medium   | 1         | 3840                 | 50807               | 0.08x             |
| Pi       | large-v2 | 1         | 3840                 | 91036               | 0.04x             |

