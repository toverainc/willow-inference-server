FROM nvcr.io/nvidia/pytorch:23.03-py3

WORKDIR /app

COPY . .

#RUN pip install --upgrade pip setuptools
#RUN pip install -r requirements.txt
RUN pip install git+https://github.com/huggingface/transformers.git
RUN pip install sentencepiece
#RUN pip install torchaudio
RUN pip install ctranslate2 librosa
#RUN pip install --upgrade numba
RUN pip install aiortc
RUN pip install pyston_lite_autoload
RUN pip install uvloop httptools
RUN pip install "fastapi[all]" "uvicorn[standard]" gunicorn
RUN pip install colorlog
RUN pip install "tritonclient[all]"

CMD uvicorn main:app --host 0.0.0.0 --port 8000 --log-level critical --loop uvloop --http httptools --ws websockets --proxy-headers --forwarded-allow-ips '*'
EXPOSE 8000
