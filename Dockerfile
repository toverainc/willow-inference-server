FROM nvcr.io/nvidia/tensorrt:23.08-py3

WORKDIR /app

# Set in environment in case we need to build any extensions
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9;9.0+PTX"

# Install zstd and git-lfs for model compression and distribution
RUN apt-get update && apt-get install -y zstd  git-lfs && rm -rf /var/lib/apt/lists/*

# Install our torch ver matching cuda
RUN --mount=type=cache,target=/root/.cache pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
# Run pip install with cache so we speedup subsequent rebuilds
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

# Install auto-gptq
RUN --mount=type=cache,target=/root/.cache pip install auto-gptq==0.4.2+cu118 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118

COPY . .

CMD ./entrypoint.sh
EXPOSE 19000
EXPOSE 19001
