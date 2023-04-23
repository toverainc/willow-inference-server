FROM nvcr.io/nvidia/tensorrt:22.12-py3

WORKDIR /app

# Install zstd for model compression and distribution
RUN apt-get update && apt-get install -y zstd  && rm -rf /var/lib/apt/lists/*

# Install our torch ver matching cuda
RUN --mount=type=cache,target=/root/.cache pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
# Run pip install with cache so we speedup subsequent rebuilds
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

# Install auto-gptq
RUN --mount=type=cache,target=/root/.cache pip install git+https://github.com/qwopqwop200/AutoGPTQ-triton.git@9e5df6c034aacaeedf8f32abbc05a88bfe1a9e87

COPY . .

CMD /bin/bash
EXPOSE 19000
