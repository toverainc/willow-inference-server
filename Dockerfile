FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /app

COPY . .

#RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt
RUN pip install ctranslate2 librosa transformers
RUN pip install --upgrade numba
RUN pip install aiortc

CMD uvicorn main:app --host 0.0.0.0 --port 8000 --log-level critical
EXPOSE 8000
