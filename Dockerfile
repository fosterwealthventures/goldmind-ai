FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tini && rm -rf /var/lib/apt/lists/*

# App
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app

# Env
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    APP_VERSION=0.1.0

# Health endpoint for Cloud Run's "health check" probe
# (Cloud Run pings / for container start; we keep our /health for smoke tests)
EXPOSE 8080

# Use Tini as init for proper signal handling
ENTRYPOINT ["/usr/bin/tini","--"]

# Start server
CMD ["python","-m","app.server"]
