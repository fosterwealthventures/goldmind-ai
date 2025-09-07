# syntax=docker/dockerfile:1.7

# -------- Base image --------
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# OS deps: curl for healthcheck, CA certs for HTTPS, netbase for /etc/services
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      netbase \
      build-essential \
   && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------- Python deps (cache-friendly) --------
# 1) copy only requirements to leverage Docker layer cache
COPY requirements.txt /app/requirements.txt

# 2) install pins TensorFlow 2.15 expects on Python 3.10, then TF itself, then the rest
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip wheel setuptools \
 && pip install --no-cache-dir \
      numpy==1.26.4 \
      typing-extensions==4.7.1 \
      gast==0.5.4 \
 && pip install --no-cache-dir \
      tensorflow-cpu==2.15.1 \
      tensorflow-estimator==2.15.0 \
      keras==2.15.0 \
      tensorboard==2.15.1 \
 && pip install --no-cache-dir -r /app/requirements.txt

# -------- App code --------
# Keep same layout: your server imports from app/ and engine/, and models from /app/lstm_models
COPY app/ /app/app/
COPY engine/ /app/engine/
COPY app/lstm_models/ /app/lstm_models/

# -------- Runtime tuning --------
# (Tune workers/threads/timeouts to match CPU & expected compute time)
ENV PORT=8080 \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=8 \
    GUNICORN_TIMEOUT=240 \
    # Some libs (e.g., requests) respect this for threadpools
    UV_THREADPOOL_SIZE=64

EXPOSE 8080

# Healthcheck hits your server's /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# -------- Security: run as non-root --------
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser

# -------- Entrypoint --------
# server.py is at /app/app/server.py and defines "app"
# We keep working dir /app so 'engine' imports resolve, and point Gunicorn at app.server:app
CMD ["bash", "-lc", "exec gunicorn -w ${GUNICORN_WORKERS:-1} -k gthread -t ${GUNICORN_TIMEOUT:-240} --threads ${GUNICORN_THREADS:-8} app.server:app --bind 0.0.0.0:${PORT}"]
