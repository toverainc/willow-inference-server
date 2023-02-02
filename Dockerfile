FROM python:3.11.1-slim-bullseye

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD uvicorn main:app --host 0.0.0.0 --port 8000 --log-level critical
EXPOSE 8000