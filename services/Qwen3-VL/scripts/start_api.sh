#!/bin/bash

# Qwen3-VL API Startup Script
# This script starts the Qwen3-VL document parser API using vLLM

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Default values - now using merged model
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
LORA_PATH="${LORA_PATH:-$PROJECT_ROOT/models/checkpoints/sft-tagged/checkpoint-300}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9890}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

echo "================================================"
echo "Starting Qwen3-VL Document Parser API"
echo "================================================"
echo "Model: $MODEL_PATH"
echo "LoRA Path: $LORA_PATH"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "================================================"

# Check if model path exists (skip for HuggingFace model IDs like "org/model")
if [[ ! "$MODEL_PATH" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$ ]] && [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path '$MODEL_PATH' does not exist."
    echo "Please ensure the merged model is available or set MODEL_PATH environment variable."
    exit 1
fi

# Check if poppler-utils is installed (required for pdf2image)
if ! command -v pdftoppm &> /dev/null; then
    echo "Warning: poppler-utils not found. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y poppler-utils
    elif command -v yum &> /dev/null; then
        sudo yum install -y poppler-utils
    else
        echo "Please install poppler-utils manually for PDF support"
    fi
fi

# Start the API
cd "$SCRIPT_DIR/.." && python -m src.api.api_vllm \
    --model-path "$MODEL_PATH" \
    --lora-path "$LORA_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"