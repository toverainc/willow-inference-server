FROM python:3.10-slim-bullseye

WORKDIR /app

COPY . .

RUN pip install fastapi[all]
RUN pip install -r requirements.txt
RUN pip install python-multipart

CMD uvicorn main:app --host 0.0.0.0 --port 8000 --log-level critical
EXPOSE 8000