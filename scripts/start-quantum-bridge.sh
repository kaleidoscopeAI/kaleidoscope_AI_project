#!/bin/bash
# Kaleidoscope AI - Quantum Bridge Startup Script
# ==============================================
# This script sets up and launches the Quantum Bridge visualization system
# with proper environment variables and error handling

# Stop on errors
set -e

# Configuration
HOST="0.0.0.0"
PORT="8000"
OLLAMA_HOST="127.0.0.1"
OLLAMA_PORT="11434"
OLLAMA_MODEL="llama3:8b"
KAI_SECRET="change-me"  # IMPORTANT: Change this in production!

# Create directories
mkdir -p logs
mkdir -p static

# Output header
echo "=============================================="
echo "  Kaleidoscope AI - Quantum Bridge"
echo "=============================================="
echo "Starting Quantum Bridge on $HOST:$PORT"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not installed!"
    exit 1
fi

# Export environment variables
export OLLAMA_HOST="$OLLAMA_HOST"
export OLLAMA_PORT="$OLLAMA_PORT"
export OLLAMA_MODEL="$OLLAMA_MODEL"
export KAI_SECRET="$KAI_SECRET"

# Check if Ollama is running
if command -v curl &> /dev/null; then
    if ! curl -s "http://$OLLAMA_HOST:$OLLAMA_PORT/api/tags" &> /dev/null; then
        echo "WARNING: Ollama does not appear to be running at $OLLAMA_HOST:$OLLAMA_PORT"
        echo "LLM features will be unavailable until Ollama is started"
    else
        echo "Ollama detected and running"
    fi
fi

# Start the Quantum Bridge
echo "Launching Quantum Bridge..."
python3 quantum_bridge.py --host "$HOST" --port "$PORT" 2>&1 | tee -a logs/quantum_bridge_$(date +%Y%m%d).log
