# syntax=docker/dockerfile:1
FROM python:3.10-slim

# ---- Base env ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ---- OS deps (minimal) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
 && rm -rf /var/lib/apt/lists/*

# Everything below is relative to /app
WORKDIR /app

# ---- Python deps (cache-friendly) ----
# 1) copy only requirements first to leverage Docker layer cache
COPY requirements.txt /app/requirements.txt

# 2) Install pins TF 2.15 expects on Py3.10, then TF itself, then the rest
RUN python -m pip install --upgrade pip wheel setuptools \
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

# ---- App code (after deps to keep cache stable) ----
# Your app code lives in compute/app/, copy it into /app/app
COPY app/ /app/app/
# If your server imports from compute/engine, include it
COPY engine/ /app/engine/

# ---- Model artifacts (baked into the image) ----
# Your env uses MODEL_PATH like /app/lstm_models/..., so place models exactly there
COPY app/lstm_models/ /app/lstm_models/

# ---- Runtime env / ports ----
ENV PORT=8080 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=8 \
    GUNICORN_TIMEOUT=240
EXPOSE 8080

# ---- Healthcheck ----
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# ---- Start (Gunicorn) ----
# server.py is at /app/app/server.py and defines "app"
# We keep working dir /app so 'engine' imports resolve, and point Gunicorn at app.server:app
CMD ["bash","-lc","exec gunicorn -w ${GUNICORN_WORKERS:-1} -k gthread -t ${GUNICORN_TIMEOUT:-30} --threads ${GUNICORN_THREADS:-2} app.server:app"]