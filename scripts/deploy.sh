#!/bin/bash
# Deployment script for Super Alita v4.0

set -e

ENVIRONMENT=${1:-development}
echo "🚀 Deploying Super Alita v4.0 to $ENVIRONMENT"
echo "============================================="

# Load environment-specific configuration
if [ -f ".env.$ENVIRONMENT" ]; then
    echo "📋 Loading configuration for $ENVIRONMENT"
    export $(cat .env.$ENVIRONMENT | grep -v '^#' | xargs)
else
    echo "⚠️ No environment config found for $ENVIRONMENT"
    echo "   Using .env or system environment variables"
fi

# Validate required dependencies
echo "🔍 Validating dependencies..."
python -c "import fastapi, uvicorn, httpx; print('✅ Core dependencies OK')"

# Run pre-deployment health check
echo "🏥 Running pre-deployment health check..."
python src/main.py --no-chat

if [ $? -ne 0 ]; then
    echo "❌ Health check failed - aborting deployment"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs data/chromadb data/backups

# Start the application based on environment
case $ENVIRONMENT in
    "production")
        echo "🏭 Starting production server..."
        uvicorn src.main:app --host 0.0.0.0 --port 8080 --workers 4
        ;;
    "development")
        echo "🔧 Starting development server with auto-reload..."
        uvicorn src.main:app --host 127.0.0.1 --port 8080 --reload
        ;;
    "test")
        echo "🧪 Running in test mode..."
        python src/main.py --host 127.0.0.1 --port 8081 --no-chat
        echo "✅ Test deployment successful"
        ;;
    *)
        echo "❓ Unknown environment: $ENVIRONMENT"
        echo "   Available: production, development, test"
        exit 1
        ;;
esac