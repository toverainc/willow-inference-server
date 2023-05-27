# Willow Inference Server

Watch the [WIS WebRTC Demo](https://www.youtube.com/watch?v=PxCO5eONqSQ)

[Willow](https://github.com/toverainc/willow) Inference Server (WIS) is a focused and highly optimized language inference server implementation. Our goal is to "automagically" enable performant, cost-effective self-hosting of released state of the art/best of breed models to enable speech and language tasks:

- Primarily targeting CUDA with support for low-end (cheap) devices such as the Tesla P4, GTX 1060, and up. Don't worry - it screams on an RTX 4090 too! [(See benchmarks)](#benchmarks). Can also run CPU-only.
- Memory optimized - all three default Whisper (base, medium, large-v2) models loaded simultaneously with TTS support inside of 6GB VRAM. LLM support defaults to int4 quantization (conversion scripts included). ASR/STT + TTS + Vicuna 13B require roughly 18GB VRAM. Less for 7B, of course!
- ASR. Heavy emphasis - Whisper optimized for very high quality as-close-to-real-time-as-possible speech recognition via a variety of means (Willow, WebRTC, POST a file, integration with devices and client applications, etc). Results in hundreds of milliseconds or less for most intended speech tasks.
- TTS. Primarily provided for assistant tasks (like Willow!) and visually impaired users.
- LLM. Optionally pass input through a provided/configured LLM for question answering, chatbot, and assistant tasks. Currently supports LLaMA deriviates with strong preference for Vicuna (the author likes 13B). Built in support for quantization to int4 to conserve GPU memory.
- Support for a variety of transports. REST, WebRTC, Web Sockets (primarily for LLM).
- Performance and memory optimized. Leverages [CTranslate2](https://github.com/OpenNMT/CTranslate2) for Whisper support and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) for LLMs.
- [Willow](https://github.com/toverainc/willow) support. WIS powers the Tovera hosted best-effort example server Willow users enjoy.
- Support for WebRTC - stream audio in real-time from browsers or WebRTC applications to optimize quality and response time. Heavily optimized for long-running sessions using WebRTC audio track management. Leave your session open for days at a time and have self-hosted ASR transcription within hundreds of milliseconds while conserving network bandwidth and CPU!
- Support for custom TTS voices. With relatively small audio recordings WIS can create and manage custom TTS voices. See API documentation for more information.

With the goal of enabling democratization of this functionality WIS will detect available CUDA VRAM, compute platform support, etc and optimize and/or disable functionality automatically (currently in order - ASR, TTS, LLM). For all supported Whisper models (large-v2, medium, and base) loaded simultaneously current minimum supported hardware is GTX 1060 3GB (6GB for ASR and TTS). User applications across all supported transports are able to programatically select and configure Whisper models and parameters (model size, beam, language detection/translation, etc) and TTS voices on a per-request basis depending on the needs of the application to balance speed/quality.

Note that we are primarily targeting CUDA - the performance, cost, and power usage of cheap GPUs like the Tesla P4 and GTX 1060 is too good to ignore. We'll make our best effort to support CPU wherever possible for current and future functionality but our emphasis is on performant latency-sensitive tasks even with low-end GPUs like the GTX 1060/Tesla P4 (as of this writing roughly $100 USD on the used market - and plenty of stock!).

## Getting started

For CUDA support you will need to have the NVIDIA drivers for your supported hardware installed.

```bash
# Clone this repo:
git clone https://github.com/toverainc/willow-inference-server.git && cd willow-inference-server

# Ensure you have nvidia-container-toolkit and not nvidia-docker
# On Arch Linux:
yay -S libnvidia-container-tools libnvidia-container nvidia-container-toolkit docker-buildx

# Ubuntu:
./deps/ubuntu.sh

# Build docker container
./utils.sh build-docker

# Download and quantize models
./utils.sh download-models

# Generate self-signed TLS cert (or place a "real" one at nginx/key.pem and nginx/cert.pem)
# Let's Encrypt/Certbot/ACME coming soon
./utils.sh gen-cert [your hostname]

# Run
./utils.sh run
```

Note that (like Willow) Willow Inference Server is very early and advancing rapidly! Users are encouraged to contribute (hence the build requirement). For the 1.0 release of WIS we will provide ready to deploy Docker containers.

## Links and Resources

Willow: Configure Willow to use ```https://[your host]:19000/api/willow``` then build and flash.

WebRTC demo client: ```https://[your host]:19000/rtc```

API documentation for REST interface: ```https://[your host]:19000/api/docs```

## Configuration

System runtime can be configured by placing a ```.env``` file in the WIS root to override any variables set by ```utils.sh```. You can also change more WIS specific parameters by copying ```settings.py``` to ```custom_settings.py```.

## Windows Support

WIS has been successfully tested on Windows with WSL (Windows Subsystem for Linux). With ASR and STT only requiring a total of 6GB VRAM WIS can be run concurrently with standard Windows desktop tasks on GPUs with 8GB VRAM.

## Benchmarks

| Device   | Model    | Beam Size | Speech Duration (ms) | Inference Time (ms) | Realtime Multiple |
|----------|----------|-----------|----------------------|---------------------|-------------------|
| RTX 4090 | large-v2 | 5         | 3840                 | 140                 | 27x               |
| RTX 3090 | large-v2 | 5         | 3840                 | 255                 | 15x               |
| H100     | large-v2 | 5         | 3840                 | 294                 | 12x               |
| H100     | large-v2 | 5         | 10688                | 519                 | 20x               |
| H100     | large-v2 | 5         | 29248                | 1223                | 23x               |
| GTX 1060 | large-v2 | 5         | 3840                 | 1114                | 3x                |
| Tesla P4 | large-v2 | 5         | 3840                 | 1099                | 3x                |
| RTX 4090 | medium   | 1         | 3840                 | 84                  | 45x               |
| RTX 3090 | medium   | 1         | 3840                 | 170                 | 22x               |
| GTX 1060 | medium   | 1         | 3840                 | 588                 | 6x                |
| Tesla P4 | medium   | 1         | 3840                 | 586                 | 6x                |
| RTX 4090 | medium   | 1         | 29248                | 377                 | 77x               |
| RTX 3090 | medium   | 1         | 29248                | 656                 | 43x               |
| GTX 1060 | medium   | 1         | 29248                | 1612                | 18x               |
| Tesla P4 | medium   | 1         | 29248                | 1730                | 16x               |
| RTX 4090 | base     | 1         | 180000               | 277                 | 648x (not a typo) |
| RTX 3090 | base     | 1         | 180000               | 594                 | 303x (not a typo) |

As you can see the realtime multiple increases dramatically with longer speech segments. Note that these numbers will also vary slightly depending on broader system configuration - CPU, RAM, etc.

When using WebRTC or Willow end-to-end latency in the browser/Willow and supported applications is the numbers above plus network latency for response - with the advantage being you can skip the "upload" portion as audio is streamed in realtime!

We are very interested in working with the community to optimize WIS for CPU. We haven't focused on it because we consider medium beam 1 to be the minimum configuration for intended tasks and CPUs cannot currently meet our latency targets.

## Comparison Benchmarks

Raspberry Pi Benchmarks run on Raspberry Pi 4 4GB Debian 11.7 aarch64 with [faster-whisper](https://github.com/guillaumekln/faster-whisper) version 0.5.1. Canakit 3 AMP USB-C power adapter and fan. All models int8 with ```OMP_NUM_THREADS=4``` and language set as en. Same methodology as timing above with model load time excluded (WIS keeps models loaded). All inference time numbers rounded down. Max temperatures as reported by ```vcgencmd measure_temp``` were 57.9 C.

| Device   | Model    | Beam Size | Speech Duration (ms) | Inference Time (ms) | Realtime Multiple |
|----------|----------|-----------|----------------------|---------------------|-------------------|
| Pi       | tiny     | 1         | 3840                 | 3333                | 1.15x             |
| Pi       | base     | 1         | 3840                 | 6207                | 0.62x             |
| Pi       | medium   | 1         | 3840                 | 50807               | 0.08x             |
| Pi       | large-v2 | 1         | 3840                 | 91036               | 0.04x             |

More coming soon!

## CUDA

We understand the focus and emphasis on CUDA may be troubling or limiting for some users. We will provide additional CPU vs GPU benchmarks but spoiler alert: a $100 used GPU from eBay will beat the fastest CPUs on the market while consuming less power at SIGNIFICANTLY lower cost. GPUs are very fundamentally different architectually and while there is admirable work being done with CPU optimized projects such as [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and CTranslate2 we believe that GPUs will maintain drastic speed, cost, and power advantages for the forseeable future. That said, we are interested in getting feedback (and PRs!) from WIS users to make full use of CTranslate2 to optimize for CPU.

### GPU Sweet Spot - May 2023

Perusing eBay and other used marketplaces the GTX 1070 seems to be the best performance/price ratio for ASR/STT and TTS while leaving VRAM room for the future. The author ordered an EVGA GTX 1070 FTW ACX3.0 for $120 USD with shipping and tax on 5/19/2023.

To support LLM/Vicuna an RTX 3090/4090 is suggested. RTX 3090 being sold for approximately $800 as of this writing (5/23/2023).

## LLM
For LLM/LLaMA/Vicuna support you will need to obtain the original Meta LLaMA models. Place the original Meta LLaMA model(s) in ```chatbot/``` and:

```bash

# Start shell docker container
./utils shell

# Go to chatbot directory
cd chatbot

# Convert to Hugging Face format, apply Vicuna 1.1 delta, quantize to int4, and install
./utils.sh install 13B
```

Restart/start WIS and Vicuna should be detected and loaded. See API documentation at ```https://[your host]:19000/api/docs```

Should support any model size but most heavily tested with 13B.

## WebRTC Tricks

The author has a long background with VoIP, WebRTC, etc. We deploy some fairly unique "tricks" to support long-running WebRTC sessions while conserving bandwidth and CPU. In between start/stop of audio record we pause (and then resume) the WebRTC audio track to bring bandwidth down to 5 kbps at 5 packets per second at idle while keeping response times low. This is done to keep ICE active and any NAT/firewall pinholes open while minimizing bandwidth and CPU usage. Did I mention it's optimized?

Start/stop of sessions and return of results uses WebRTC data channels.

## WebRTC Client Library

See the [Willow TypeScript Client repo](https://github.com/toverainc/willow-ts-client) to integrate WIS WebRTC support into your own frontend.

## Fun Ideas

- Integrate WebRTC with Home Assistant dashboard to support streaming audio directly from the HA dashboard on desktop or mobile.
- Desktop/mobile transcription apps (look out for a future announcement on this!).
- Desktop/mobile assistant apps - Willow everywhere!

## The Future (in no particular order)

### Better TTS

We're looking for feedback from the community on preferred implementations, voices, etc. See the [open issue](https://github.com/toverainc/willow-inference-server/issues/60).

### TTS Caching

Why do it again when you're saying the same thing? Support on-disk caching of TTS output for lightning fast TTS response times.

### Support for more languages

Meta released MMS on 5/22/2023, supporting over 1,000 languages across speech to text and text to speech!

### Code refactoring and modularization

WIS is very early and we will refactor, modularize, and improve documentation well before the 1.0 release.

### Chaining of functions (apps)

We may support user-defined modules to chain together any number of supported tasks within one request, enabling such things as:

Speech in -> LLM -> Speech out

Speech in -> Arbitrary API -> Speech out

...and more, directly in WIS!
