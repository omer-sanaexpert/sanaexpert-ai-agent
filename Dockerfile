# syntax=docker/dockerfile:1

FROM python:3.11

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

EXPOSE ${PORT:-8000}

CMD ["gunicorn", \
     "--bind", "0.0.0.0:${PORT:-8000}", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--log-file", "-", \
     "main:app"]