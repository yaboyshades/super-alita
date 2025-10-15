#!/bin/bash
# Test runner script for Super Alita v4.0

set -e

echo "🧪 Running Super Alita v4.0 Test Suite"
echo "====================================="

# Set test environment
export ALITA_PROFILE=test
export LOG_LEVEL=ERROR
export ALITA_REQUIRE_API_KEY=false

# Create test directories
mkdir -p tmp/test_data
export CHROMADB_PATH=tmp/test_data/chromadb
export EVENT_BACKUP_PATH=tmp/test_data/events.jsonl

echo "🔧 Installing test dependencies..."
pip install -e . pytest pytest-asyncio pytest-cov httpx

echo "🏗️ Running unit tests..."
pytest tests/services/ -v --tb=short || echo "⚠️ Service tests had issues"

echo "🌐 Running router tests..."
pytest tests/routers/ -v --tb=short || echo "⚠️ Router tests had issues"

echo "📊 Running performance tests..."
pytest tests/performance/ -v --tb=short || echo "⚠️ Performance tests had issues"

echo "📋 Running full test suite with coverage..."
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

echo "✅ Test suite completed!"
echo "📊 Coverage report generated in htmlcov/index.html"

# Cleanup
rm -rf tmp/test_data

echo "🎉 Test suite completed successfully!"