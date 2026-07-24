# PBT Trading Platform — Docker Image
# Single-container deployment with all dependencies
# Build: docker build -t pbt-trading .
# Run:   docker run -p 8000:8000 -e APCA_API_KEY_ID=xxx pbt-trading

FROM python:3.11-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Install Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.115.5 \
    uvicorn[standard]==0.32.1 \
    websockets==13.1 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    pandas==2.2.3 \
    httpx==0.27.2 \
    prometheus-client==0.21.0

# Install Alpaca integration by default for deployments that use the Alpaca broker
ARG ALPACA_INTEGRATION=1
RUN if [ "$ALPACA_INTEGRATION" = "1" ]; then pip install --no-cache-dir alpaca-py==0.34.0; fi

# Optional: install ML tracking
ARG ML_TRACKING=0
RUN if [ "$ML_TRACKING" = "1" ]; then pip install --no-cache-dir mlflow==2.17.2; fi

# ---- Copy application code ----
COPY . .

# ---- Create runtime directories ----
RUN mkdir -p checkpoints logs reports experiments/runs

# ---- Environment defaults ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PBT_HOST=0.0.0.0 \
    PBT_PORT=8000 \
    PBT_POPULATION=12 \
    PBT_GENERATIONS=100 \
    PBT_BROKER=paper \
    PBT_CAPITAL=100000

# ---- Healthcheck ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PBT_PORT}/api/health || exit 1

EXPOSE 8000

# Default: unified engine + dashboard (shared live state)
# Render injects $PORT; fall back to PBT_PORT / 8000 for local Docker runs.
CMD ["sh", "-c", "\
    python main.py run \
        --host ${PBT_HOST} \
        --port ${PORT:-${PBT_PORT:-8000}} \
        --population ${PBT_POPULATION} \
        --generations ${PBT_GENERATIONS} \
        --broker ${PBT_BROKER} \
        --capital ${PBT_CAPITAL} \
    "]

# ---- Multi-stage: production image ----
FROM base AS production

# Add non-root user
RUN useradd -m -u 1000 pbt && chown -R pbt:pbt /app
USER pbt

LABEL maintainer="PBT Trading Platform" \
      version="1.1.0" \
      description="Population-Based Training algorithmic trading system"
