#!/bin/bash
# Start Mangle integration for Super Alita

# Display banner
echo "========================================"
echo "  Super Alita + Mangle Integration"
echo "========================================"
echo ""

# Check if Mangle is installed
if ! command -v mangle &> /dev/null; then
    echo "❌ Mangle not found. Please install Mangle first:"
    echo "   go install github.com/google/mangle/cmd/mangle@latest"
    exit 1
fi

echo "✅ Found Mangle installation"
MANGLE_PATH=$(which mangle)
echo "   Path: $MANGLE_PATH"

# Set up environment variables
export MANGLE_BIN_PATH=$MANGLE_PATH
echo "✅ Set MANGLE_BIN_PATH environment variable"

# Check if .env file exists, create if not
if [ ! -f .env ]; then
    echo "⚠️ No .env file found, creating from .env.example"
    cp .env.example .env
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create .env file"
        exit 1
    fi
    echo "✅ Created .env file"
else
    echo "✅ Found existing .env file"
fi

# Start Super Alita server
echo ""
echo "🚀 Starting Super Alita server with Mangle integration..."
echo ""
uvicorn app:app --reload --port 8080
