# Willow Inference Server

[Willow](https://github.com/toverainc/willow) Inference Server (WIS) is a focused, highly optimized language inference server implementation. Our goal is to "automagically" enable performant, cost-effective self-hosting with open SOTA/best of breed models to enable speech and language tasks:

- Primarily targeting CUDA with support for low-end (cheap) devices such as the Tesla P4, GTX 1060, and up. Don't worry - it screams on an RTX 4090 too! [(See benchmarks)](#benchmarks)
- Memory optimized - all three default Whisper (base, medium, large-v2) models and TTS support sessions inside of 6GB VRAM. LLM support defaults to int4 quantization (conversion scripts included). ASR/STT + TTS + Vicuna 13B require roughly 18GB VRAM. Less for 7B, of course!
- ASR. Heavy emphasis - Whisper optimized for very high quality as-close-to-real-time-as-possible speech recognition via a variety of means (Willow, WebRTC, POST a file, integration with devices and client applications, etc). Results in hundreds of milliseconds or less for most intended speech tasks.
- TTS. Emerging support, primarily provided for assistant tasks (like Willow!) and visually impaired users.
- LLM. Optionally pass input through a provided/configured LLM for question answering, chatbot, and assistant tasks. Currently supports LLaMA deriviates with preference for Vicuna (the author likes 13B). Built in support for quantization to int4 to conserve GPU memory.
- Support for a variety of transports. REST, WebRTC, Web Sockets (primarily for LLM).
- Performance and memory optimized. Leverages [CTranslate2](https://github.com/OpenNMT/CTranslate2) for Whisper support and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) for LLMs.
- [Willow](https://github.com/toverainc/willow) support.
- Support for WebRTC - stream audio in real-time from browsers or WebRTC applications to optimize quality and response time. Heavily optimized for long-running sessions using WebRTC audio track management. Leave your connection open for days at a time and have self-hosted ASR transcription within hundreds of milliseconds while conserving network bandwidth and CPU.

With the goal of enabling democratization of this functionality WIS will detect available CUDA VRAM, compute platform support, etc and optimize and/or disable functionality automatically (currently in order - ASR, TTS, LLM). For all supported Whisper models (large-v2, medium, and base) current minimum supported hardware is GTX 1060 3GB (6GB for ASR and TTS). User applications across all supported transports are able to programatically select and configure Whisper models and parameters (model size, beam, language detection/translation, etc) on a per-request basis depending on the needs of the application to optimize response times.

Note that we are primarily targeting CUDA - the performance, cost, and power usage of cheap GPUs like the Tesla P4 and GTX 1060 is too good to ignore. We'll make our best effort to support CPU wherever possible for current and future functionality but our emphasis is on performant latency-sensitive tasks even with low-end GPUs like the GTX 1060/Tesla P4 (as of this writing roughly $100 USD on the used market - and plenty of stock!).

## Getting started
```bash
# Clone this repo:
git clone https://github.com/toverainc/willow-inference-server.git && cd willow-inference-server

# Ensure you have nvidia-container-toolkit and not nvidia-docker
# On Arch Linux:
yay -S libnvidia-container-tools libnvidia-container nvidia-container-toolkit

# Ubuntu:
./deps/ubuntu.sh

# Build docker container
./build.sh

# Download and quantize the models
./download_models.sh

# Generate TLS cert (or place a "real" one at key.pem and cert.pem - Let's Encrypt/Certbot/ACME coming soon)
./gen_cert.sh

# Run
./run.sh
```

## Links and Resources

Willow: Configure Willow to use ```https://[your host]:19000/api/willow``` then build and flash
WebRTC demo client: ```https://[your host]:19000/rtc```
API documentation for REST interface: ```https://[your host]:19000/docs```

## Configuration
System runtime can be configured by placing a ```.env``` file in the WIS root to override any variables set by ```run.sh```. You can also change more WIS specific parameters by copying ```settings.py``` to ```custom_settings.py```.

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

As you can see the realtime multiple increases dramatically with longer speech segments.

When using WebRTC or Willow end-to-end latency in the browser/Willow and supported applications is the numbers above plus network latency for response - with the advantage being you can skip the "upload" portion as audio is streamed in realtime!

## Comparison Benchmarks

Raspberry Pi Benchmarks run on Raspberry Pi 4 4GB Debian 11.7 aarch64 with faster-whisper 0.5.1. Canakit 3 AMP USB-C power adapter and fan. All models int8 with ```OMP_NUM_THREADS=4``` set and language set as en. Same methodology as timing above with model load time excluded (WIS keeps models loaded). All inference time numbers rounded down. Max temperatures as reported by ```vcgencmd measure_temp``` were 57.9 C.

| Device   | Model    | Beam Size | Speech Duration (ms) | Inference Time (ms) | Realtime Multiple |
|----------|----------|-----------|----------------------|---------------------|-------------------|
| Pi       | tiny     | 1         | 3840                 | 3333                | 1.15x             |
| Pi       | base     | 1         | 3840                 | 6207                | 0.62x             |
| Pi       | medium   | 1         | 3840                 | 50807               | 0.08x             |
| Pi       | large-v2 | 1         | 3840                 | 91036               | 0.04x             |

More coming soon!

## WebRTC Tricks
The author has a long background of work in VoIP, WebRTC, etc. We deploy some fairly unique "tricks" to support long-running WebRTC sessions while conserving bandwidth and CPU. In between start/stop of audio record we pause (and then resume) the WebRTC audio track to bring bandwidth down to 5 kbps at 5 packets per second at idle while keeping response times low. This is done to keep ICE active and any NAT/firewall pinholes open while minimizing bandwidth and CPU usage. Did I mention it's optimized?

Start/stop of sessions and return of results uses WebRTC data channels.

## Fun Ideas

- Integrate WebRTC with Home Assistant dashboard to support streaming audio directly from the HA dashboard on desktop or mobile.
- Desktop/mobile transcription apps (look out for a future announcement on this!).
- Desktop/mobile assistant apps - Willow everywhere!
