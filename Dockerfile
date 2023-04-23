FROM nvcr.io/nvidia/tensorrt:22.12-py3

WORKDIR /app

# Install zstd for model compression and distribution
RUN apt-get update && apt-get install -y zstd  && rm -rf /var/lib/apt/lists/*

# Install our torch ver matching cuda
RUN --mount=type=cache,target=/root/.cache pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
# Run pip install with cache so we speedup subsequent rebuilds
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

# Install auto-gptq (triton)
RUN --mount=type=cache,target=/root/.cache pip install git+https://github.com/qwopqwop200/AutoGPTQ-triton.git@9e5df6c034aacaeedf8f32abbc05a88bfe1a9e87

# Install auto-gptq (cuda)
#ARG TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9,9.0+PTX"
#RUN --mount=type=cache,target=/root/.cache pip install git+https://github.com/PanQiWei/AutoGPTQ.git@4f84d2168424b3cec3441a3378ad5366f68354b3

COPY . .

CMD /bin/bash
EXPOSE 19000
