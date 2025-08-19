FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NOWARNINGS=yes \
    TZ=Etc/UTC

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates apt-utils build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel build pip-tools

COPY requirements.in /tmp/requirements.in
RUN pip-compile -q /tmp/requirements.in -o /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --prefer-binary -r /tmp/requirements.txt
RUN mkdir -p /app/requirements && cp /tmp/requirements.txt /app/requirements/requirements.txt

COPY app/ /app

CMD exec gunicorn -k gthread --workers 2 --threads 8 --timeout 120 \
  --bind :${PORT:-8080} --chdir /app server:app
