# ---------- Base image ----------
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps minimally needed for common scientific/HTTP libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Copy source ----------
# Keep structure: /app/api and /app/shared so imports like "from shared import ..." work
COPY shared/ ./shared/
COPY api/ ./api/

# ---------- Python dependencies ----------
# If api/requirements.txt exists we use it; otherwise install a lean default set
RUN python -m pip install --upgrade pip && \
    ( \
      if [ -f ./api/requirements.txt ]; then \
        pip install -r ./api/requirements.txt; \
      else \
        pip install \
          fastapi \
          uvicorn[standard] \
          httpx \
          requests \
          pydantic \
          numpy \
          pandas \
          python-dotenv; \
      fi \
    )

# Ensure Python can import from /app
ENV PYTHONPATH=/app

# ---------- Security: non-root user ----------
RUN addgroup --system api && adduser --system --ingroup api api
USER api

# Cloud Run listens on 8080 by default
ENV PORT=8080
EXPOSE 8080

# ---------- Start server ----------
# Module path assumes your ASGI app is in api/server.py with "app = FastAPI(...)"
CMD ["uvicorn", "api.app.server:app", "--host", "0.0.0.0", "--port", "8000"]

