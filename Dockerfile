FROM nvcr.io/nvidia/tensorrt:23.08-py3

WORKDIR /app

# Set in environment in case we need to build any extensions
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9;9.0+PTX"

# Install zstd and git-lfs for model compression and distribution
RUN apt-get update && apt-get install -y \
  git-lfs \
  pkg-config \
  zstd \
  && rm -rf /var/lib/apt/lists/*

# Install our torch ver matching cuda
RUN --mount=type=cache,target=/root/.cache pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

COPY requirements.txt .
# Run pip install with cache so we speedup subsequent rebuilds
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

COPY . .

CMD ./entrypoint.sh
EXPOSE 19000
EXPOSE 19001
