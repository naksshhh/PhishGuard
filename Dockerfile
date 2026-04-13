# ============================================================
# PhishGuard++ — Cloud Run Dockerfile
# Models are downloaded from GCS at startup via Python GCS client.
# ============================================================

FROM python:3.11-slim-bullseye

# Install lean system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ──────────────────────────────
COPY requirements.txt .

# CPU-only PyTorch (no GPU on standard Cloud Run)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir google-cloud-storage

# ── Copy application code ────────────────────────────────────
COPY . .

# Copy and prepare entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create empty models dir (populated from GCS at startup)
RUN mkdir -p /app/models

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV GCS_BUCKET=project-5d926acd-a531-48ca-bef-phishguard-models

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:${PORT}/docs || exit 1

CMD ["/entrypoint.sh"]
