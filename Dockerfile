# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install runtime deps first (better layer caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app
COPY pyproject.toml .
COPY src ./src
COPY app.py ./
ENV PYTHONPATH=/app/src

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8080/healthz || exit 1

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8080"]
