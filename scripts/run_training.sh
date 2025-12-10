#!/bin/bash

set -e

echo "================================"
echo "GCS Training Launch"
echo "================================"

# Set directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="/workspaces/av/basic_gcs/configs/training_config.yaml"

# Check config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "✓ Config file: $CONFIG_FILE"

# Start MeshCat server (optional)
if command -v meshcat-server &> /dev/null; then
    echo "✓ Starting MeshCat server..."
    meshcat-server &
    MESHCAT_PID=$!
    sleep 2
    echo "  Open http://localhost:6000 in your browser"
else
    echo "⚠️  meshcat-server not found. Visualization will be disabled."
fi

# Run training
echo ""
echo "Starting training..."
python -m src.training.agent --config "$CONFIG_FILE" --epochs 1000

# Cleanup
if [ ! -z "$MESHCAT_PID" ]; then
    kill $MESHCAT_PID 2>/dev/null || true
fi

echo ""
echo "✓ Training completed"
echo "================================"
