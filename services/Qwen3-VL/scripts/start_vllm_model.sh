#!/bin/bash
# Deploy merged Qwen3-VL model with vLLM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-$SCRIPT_DIR/merged_model}"
GPU_MEMORY="${GPU_MEMORY:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "========================================"
echo "Deploying Merged Qwen3-VL Model"
echo "========================================"
echo "Model Path: $MODEL_PATH"
echo "GPU Memory Utilization: $GPU_MEMORY"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Host: $HOST"
echo "Port: $PORT"
echo "========================================"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found at $MODEL_PATH"
    exit 1
fi

# Check for required files
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Error: config.json not found in model directory"
    exit 1
fi

echo ""
echo "Starting vLLM server..."
echo ""

# Start vLLM server
vllm serve "$MODEL_PATH" \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEMORY" \
    --max-model-len "$MAX_MODEL_LEN" \
    --host "$HOST" \
    --port "$PORT" \
    --async-scheduling \
    --mm-processor-cache-gb 0

echo ""
echo "vLLM server stopped."
