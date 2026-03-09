#!/usr/bin/env bash
# Launch both document processing API services
#   - Document Classification (Qwen3-VL) on port 7871
#   - Bank Report Extraction  (Qwen3-VL) on port 7872
#
# Usage:
#   bash launch_apis.sh          # start both (installs ollama + model if needed)
#   bash launch_apis.sh stop     # kill both
#   bash launch_apis.sh restart  # restart both

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT_CLASSIFY=7871
PORT_EXTRACT=7872

APP_CLASSIFY="app.app_document_classification_vlm.main:app"
APP_EXTRACT="app.app_bank_report_extraction.main:app"

REQUIRED_MODEL="qwen3-vl:8b"

PID_DIR="$SCRIPT_DIR/.pids"
mkdir -p "$PID_DIR"
mkdir -p "$SCRIPT_DIR/logs"

# ---------------------------------------------------------------------------
# Ollama bootstrap
# ---------------------------------------------------------------------------

ensure_ollama() {
    if ! command -v ollama &>/dev/null; then
        echo "=== ollama not found, installing... ==="
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "=== ollama found: $(ollama --version 2>&1 | head -1) ==="
    fi
}

ensure_ollama_running() {
    # Normalize OLLAMA_HOST: add http:// if no scheme present
    local raw_host="${OLLAMA_HOST:-localhost:11434}"
    local ollama_host
    if [[ "$raw_host" != http://* && "$raw_host" != https://* ]]; then
        ollama_host="http://$raw_host"
    else
        ollama_host="$raw_host"
    fi
    if curl -sf "$ollama_host/api/version" &>/dev/null; then
        echo "=== ollama server already running at $ollama_host ==="
        return
    fi

    echo "=== Starting ollama server... ==="
    nohup ollama serve > "$SCRIPT_DIR/logs/ollama.log" 2>&1 &
    echo $! > "$PID_DIR/ollama.pid"

    # Wait up to 30 s for server to be ready
    local attempts=0
    while ! curl -sf "$ollama_host/api/version" &>/dev/null; do
        sleep 1
        attempts=$((attempts + 1))
        if [[ $attempts -ge 30 ]]; then
            echo "ERROR: ollama server did not start within 30 seconds." >&2
            exit 1
        fi
    done
    echo "  ollama server ready (PID $(cat "$PID_DIR/ollama.pid"))"
}

ensure_model() {
    local model="$1"
    local raw_host="${OLLAMA_HOST:-localhost:11434}"
    local ollama_host
    if [[ "$raw_host" != http://* && "$raw_host" != https://* ]]; then
        ollama_host="http://$raw_host"
    else
        ollama_host="$raw_host"
    fi

    # Ask the running server whether the model exists
    if curl -sf "$ollama_host/api/tags" | grep -q "\"$model\""; then
        echo "=== Model '$model' already available ==="
    else
        echo "=== Pulling model '$model' (this may take a while) ==="
        ollama pull "$model"
    fi
}

# ---------------------------------------------------------------------------
# Service control
# ---------------------------------------------------------------------------

stop_services() {
    for svc in classify extract; do
        pid_file="$PID_DIR/$svc.pid"
        if [[ -f "$pid_file" ]]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "Stopping $svc (PID $pid) ..."
                kill "$pid"
            fi
            rm -f "$pid_file"
        fi
    done
    echo "All services stopped."
}

start_services() {
    ensure_ollama
    ensure_ollama_running
    ensure_model "$REQUIRED_MODEL"

    echo ""
    echo "=== Starting Document Classification API on port $PORT_CLASSIFY ==="
    nohup python3 -m uvicorn "$APP_CLASSIFY" \
        --host 0.0.0.0 --port "$PORT_CLASSIFY" \
        > "$SCRIPT_DIR/logs/classify.log" 2>&1 &
    echo $! > "$PID_DIR/classify.pid"
    echo "  PID: $!"

    echo "=== Starting Bank Report Extraction API on port $PORT_EXTRACT ==="
    nohup python3 -m uvicorn "$APP_EXTRACT" \
        --host 0.0.0.0 --port "$PORT_EXTRACT" \
        > "$SCRIPT_DIR/logs/extract.log" 2>&1 &
    echo $! > "$PID_DIR/extract.pid"
    echo "  PID: $!"

    echo ""
    echo "Services started:"
    echo "  Classification : http://0.0.0.0:$PORT_CLASSIFY"
    echo "  Extraction     : http://0.0.0.0:$PORT_EXTRACT"
    echo ""
    echo "Logs:"
    echo "  $SCRIPT_DIR/logs/classify.log"
    echo "  $SCRIPT_DIR/logs/extract.log"
    echo ""
    echo "To stop: bash $0 stop"
}

case "${1:-start}" in
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 1
        start_services
        ;;
    start|*)
        stop_services 2>/dev/null || true
        start_services
        ;;
esac
