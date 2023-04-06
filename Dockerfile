FROM nvcr.io/nvidia/tensorrt:22.12-py3

WORKDIR /app

COPY . .

# Run pip install with cache so we speedup subsequent rebuilds
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

CMD /bin/bash
EXPOSE 19000
