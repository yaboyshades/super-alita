#!/usr/bin/env bash
# Cross-platform development server startup script
# Works on Windows (Git Bash), macOS, and Linux

set -e  # Exit on any error

echo "🚀 Starting Super Alita Development Environment"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found, creating from .env.example..."
    cp .env.example .env
    print_success "Created .env file"
fi

# Check Python environment
if [ ! -d ".venv" ]; then
    print_error "Virtual environment not found!"
    print_status "Please run: python -m venv .venv"
    exit 1
fi

# Activate virtual environment (cross-platform)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash)
    source .venv/Scripts/activate
else
    # macOS/Linux
    source .venv/bin/activate
fi

print_success "Virtual environment activated"

# Install dependencies if needed
print_status "Checking dependencies..."
pip install -r requirements.txt -r requirements-test.txt > /dev/null 2>&1

# Start the development server
print_status "Starting Super Alita server on http://127.0.0.1:8080"
print_warning "Use Ctrl+C to stop the server"

exec uvicorn app:app --reload --host 127.0.0.1 --port 8080 --log-level info