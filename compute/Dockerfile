FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NOWARNINGS=yes \
    TZ=Etc/UTC

WORKDIR /app

# System deps (add others only if you truly need them)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
# Use whichever file you maintain in compute/: requirements.txt (or requirements.in)
COPY requirements.txt ./
# If you use requirements.in instead, swap the two lines:
# COPY requirements.in ./requirements.in
# RUN python -m pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.in
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gunicorn

# ---- App code ----
COPY . /app

# Cloud Run will provide $PORT
ENV PORT=8080
# Our app lives in compute/app/server.py with "app = create_app()"
# --chdir app puts us inside compute/app so module is just "server:app"
CMD exec gunicorn --bind :${PORT} --workers 2 --threads 8 --timeout 120 --chdir app server:app
