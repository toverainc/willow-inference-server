# A lot of software, a lot of versions

ARG CLANG_VER=14
ARG CTRANSLATE2_VER=cd26b3e
ARG NVIDIA_VER=23.05
ARG ONEAPI_VER=2023.1.0
ARG ONEDNN_VER=3.1.1
ARG TORCH_VER=2.1.0
ARG TORCH_AUDIO_VER=2.1.0

# Misc
ARG CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ARG CTRANSLATE2_ROOT="/opt/ctranslate2"
ARG CTRANSLATE2_URL="https://github.com/chiiyeh/CTranslate2.git"

# Builder
FROM nvcr.io/nvidia/tensorrt:${NVIDIA_VER}-py3 as builder

ARG CLANG_VER
ARG CTRANSLATE2_ROOT
ENV CTRANSLATE2_ROOT=${CTRANSLATE2_ROOT}
ARG CTRANSLATE2_URL
ARG CTRANSLATE2_VER
ARG CUDA_ARCH_LIST
ARG ONEAPI_VER
ARG ONEDNN_VER
ARG TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        wget \
        lsb-release \
        software-properties-common \
        gnupg

WORKDIR /root

# Install clang
RUN wget -O llvm.sh https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
RUN ./llvm.sh ${CLANG_VER}

RUN apt-get update && apt-get install -y --no-install-recommends libomp-${CLANG_VER}-dev

# Use clang
ENV CC=/usr/bin/clang-${CLANG_VER}
ENV CXX=/usr/bin/clang++-${CLANG_VER}

# Install oneAPI
RUN wget -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    apt-key add *.PUB && \
    rm *.PUB && \
    echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        intel-oneapi-mkl-devel-${ONEAPI_VER} \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install cmake
RUN --mount=type=cache,target=/root/.cache pip install cmake==3.22.*

# Install oneDNN
RUN wget -q https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${ONEDNN_VER}.tar.gz && \
    tar xf *.tar.gz && \
    rm *.tar.gz && \
    cd oneDNN-* && \
    cmake -DCMAKE_BUILD_TYPE=Release -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_EXAMPLES=OFF \
        -DONEDNN_BUILD_TESTS=OFF -DONEDNN_ENABLE_WORKLOAD=INFERENCE \
        -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" -DONEDNN_BUILD_GRAPH=OFF . && \
    make -j$(nproc) install && \
    cd .. && \
    rm -r oneDNN-*

# Build CTranslate2
RUN git clone --recursive ${CTRANSLATE2_URL}

WORKDIR /root/CTranslate2

RUN git checkout ${CTRANSLATE2_VER}

ARG CXX_FLAGS
ENV CXX_FLAGS=${CXX_FLAGS:-"-msse4.1"}
ARG CUDA_NVCC_FLAGS
ENV CUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS:-"-Xfatbin=-compress-all"}
ENV CUDA_ARCH_LIST=${CUDA_ARCH_LIST:-"Common"}

RUN mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${CTRANSLATE2_ROOT} \
          -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=ON -DWITH_DNNL=ON -DOPENMP_RUNTIME=COMP \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
          -DCUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS}" -DCUDA_ARCH_LIST="${CUDA_ARCH_LIST}" .. && \
    VERBOSE=1 make -j$(nproc) install

ENV LANG=en_US.UTF-8
COPY README.md .

RUN --mount=type=cache,target=/root/.cache cd python && \
    pip --no-cache-dir install -r install_requirements.txt && \
    python3 setup.py bdist_wheel --dist-dir ${CTRANSLATE2_ROOT}

# Runtime
FROM nvcr.io/nvidia/tensorrt:${NVIDIA_VER}-py3

ARG CLANG_VER
ARG CTRANSLATE2_ROOT
ENV CTRANSLATE2_ROOT=${CTRANSLATE2_ROOT}
ARG TORCH_VER
ARG TORCH_AUDIO_VER

WORKDIR /app

# Install deps
RUN apt-get update && apt-get install -y zstd git-lfs libsox3

# Install llvm repo
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${CLANG_VER} main" > /etc/apt/sources.list.d/llvm.list

# Install clang omp runtime
RUN apt-get update && apt-get install -y libomp5-${CLANG_VER} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install our torch ver matching cuda
RUN --mount=type=cache,target=/root/.cache pip install torch==${TORCH_VER} torchaudio==${TORCH_AUDIO_VER}

COPY requirements.txt .
# Run pip install with cache so we speedup subsequent rebuilds
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

# Dynamic library fixes
# Workaround for speaker verification with torchaudio
RUN ln -sf /usr/lib/x86_64-linux-gnu/libsox.so.3 /usr/lib/x86_64-linux-gnu/libsox.so
RUN ldconfig

# Set CTranslate2 path
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CTRANSLATE2_ROOT/lib

# Install CTranslate2 from builder
COPY --from=builder $CTRANSLATE2_ROOT $CTRANSLATE2_ROOT
RUN python3 -m pip --no-cache-dir install $CTRANSLATE2_ROOT/*.whl && \
    rm $CTRANSLATE2_ROOT/*.whl

COPY . .

CMD ./entrypoint.sh
EXPOSE 19000
EXPOSE 19001
