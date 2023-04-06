FROM nvcr.io/nvidia/tensorrt:22.12-py3

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN pip cache purge

CMD uvicorn main:app --host 0.0.0.0 --port 8000 --log-level critical --loop uvloop --http httptools --ws websockets --proxy-headers --forwarded-allow-ips '*'
EXPOSE 8000
