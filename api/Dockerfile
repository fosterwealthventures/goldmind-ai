FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt lives in the api/ context
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt -i https://pypi.org/simple

# copy the rest of the api app
COPY . /app

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
