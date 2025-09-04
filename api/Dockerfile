# api/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build tools only if you need to compile wheels; remove if not needed
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code from the api/ context
COPY . .

# IMPORTANT: start the ASGI server and use Cloud Run's PORT
# Replace "app.server:app" with the actual module:path of your FastAPI app object
CMD ["sh", "-c", "python -m uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
