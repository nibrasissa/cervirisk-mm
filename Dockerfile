# Multi-stage Dockerfile for CerviRisk-MM.
#
# Stage 1 (builder)  : installs Python dependencies in an isolated layer
#                      so that source-code changes don't invalidate the heavy
#                      scikit-learn / xgboost / pandas wheel cache.
# Stage 2 (runtime)  : minimal slim image with just the venv + the project
#                      source + the trained model artifact.
#
# Build:
#     docker build -t cervirisk-mm:0.1.0 .
#
# Run (model must already be trained on the host and present in ./models):
#     docker run -p 8000:8000 -v $(pwd)/models:/app/models cervirisk-mm:0.1.0
#
# Or use docker-compose.yml for a one-command boot.

# ---- Stage 1: builder -----------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# System build deps (only present in the builder; not copied into runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies in an isolated venv so the final stage can copy it whole.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# ---- Stage 2: runtime -----------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CERVIRISK_HOST=0.0.0.0 \
    CERVIRISK_PORT=8000

# Run as non-root (least-privilege)
RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app

# Pull venv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Project sources (small, last so changes invalidate only this layer)
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY docs/ ./docs/
COPY README.md .

# File ownership for the non-root user
RUN chown -R app:app /app
USER app

EXPOSE 8000

# Healthcheck — Docker pings /health every 30s; container marked unhealthy
# if three consecutive checks fail.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); \
        sys.exit(0 if r.status == 200 else 1)"

# uvicorn, not gunicorn — single-process is fine for this prototype.
# For production scale-out, swap to `gunicorn -k uvicorn.workers.UvicornWorker`.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
