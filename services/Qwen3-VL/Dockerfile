FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=Qwen/Qwen2-VL-7B-Instruct
ENV PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    poppler-utils \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements_api.txt .
RUN pip3 install --no-cache-dir -r requirements_api.txt

# Copy application files
COPY api_vllm.py .
COPY start_api.sh .
RUN chmod +x start_api.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/')"

# Run the API
CMD ["python3", "api_vllm.py", "--host", "0.0.0.0", "--port", "8000"]
