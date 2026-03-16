# ── IDP Pipeline API ─────────────────────────────────────────────────────────
# Base: NVIDIA CUDA 12.6 + cuDNN runtime on Ubuntu 24.04 (ships Python 3.12)
# Services exposed:
#   7871 – OCR API  (document → markdown)
#   7872 – LLM API  (markdown → JSON)
#   7873 – Wrapper API (combined)
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ---------- system dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        # OpenCV / PaddleOCR runtime libs
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        # misc utilities
        curl \
    && rm -rf /var/lib/apt/lists/* \
    # create isolated venv (avoids PEP 668 "externally managed" error on 24.04)
    && python3.12 -m venv "$VIRTUAL_ENV"

WORKDIR /app

# ---------- PyTorch with CUDA 12.6 (install first – largest layer) ----------
RUN pip install \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu126

# ---------- application Python dependencies ----------
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# ---------- application source code ----------
COPY pipelines/ ./pipelines/
COPY modules/   ./modules/

# ---------- entrypoint ----------
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 7871 7872 7873

ENTRYPOINT ["docker-entrypoint.sh"]
# default: start all three services
CMD ["--service", "all"]
