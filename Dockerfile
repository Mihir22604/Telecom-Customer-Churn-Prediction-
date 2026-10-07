# ==============================================================================
# Multi-Stage Dockerfile for Telecom Customer Churn Prediction System
# Default target: api (FastAPI Backend)
# Alternative target: ui (Streamlit Frontend)
# ==============================================================================

FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# Target: API (FastAPI Backend Microservice)
# ------------------------------------------------------------------------------
FROM base AS api

COPY src/ /app/src/
COPY api/ /app/api/
COPY models/ /app/models/
COPY .env.example /app/.env.example

RUN mkdir -p /app/logs && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------------------------------------------------------
# Target: UI (Streamlit Web Dashboard)
# ------------------------------------------------------------------------------
FROM base AS ui

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    FASTAPI_URL=http://api:8000

COPY app/ /app/app/
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/processed/ /app/data/processed/

RUN mkdir -p /app/logs && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
