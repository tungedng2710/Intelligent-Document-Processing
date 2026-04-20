# ── IDP Pipeline API ─────────────────────────────────────────────────────────
# Base: Python 3.12 on Ubuntu 24.04
# Services exposed:
#   7871 – OCR API  (document → markdown)
#   7872 – LLM API  (markdown → JSON)
#   7873 – Wrapper API (combined)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---------- system dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OpenCV / PaddleOCR runtime libs
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        # misc utilities
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- PyTorch CPU (for marker-pdf) ----------
RUN pip install \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cpu

# ---------- application Python dependencies ----------
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# ---------- application source code ----------
COPY pipelines/ ./pipelines/
COPY modules/   ./modules/

# ---------- entrypoint ----------
ADD docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 7871 7872 7873

ENTRYPOINT ["docker-entrypoint.sh"]
# default: start all three services
CMD ["--service", "all"]
