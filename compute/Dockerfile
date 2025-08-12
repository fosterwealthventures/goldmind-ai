FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libatlas-base-dev libopenblas-dev \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app + engine + assets
COPY app ./app
COPY engine ./engine
COPY data ./data
COPY lstm_models ./lstm_models
COPY config.json ./config.json

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "4", "--timeout", "300", "-b", ":8080", "app.server:app"]
