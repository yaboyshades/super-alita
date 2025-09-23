#!/bin/bash
# CDIL: Contract-Driven Interface Locks - Symbol Extraction
# Extracts symbol graphs from code and verifies signatures

set -e

# Default values
VERIFY=false
OUTPUT_FILE=".contracts/signature.lock.json"
SOURCE_DIRS="src"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --verify)
      VERIFY=true
      shift
      ;;
    --output|-o)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --sources|-s)
      SOURCE_DIRS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Extract symbols and generate signature lock
echo "Extracting symbols from: $SOURCE_DIRS"
python3 -m tools.cdil.main --sources $SOURCE_DIRS --output "$OUTPUT_FILE" extract

if [ "$VERIFY" = true ]; then
  echo "Verifying signature lock..."
  python3 -m tools.cdil.main --sources $SOURCE_DIRS --lock-file "$OUTPUT_FILE" verify
fi

echo "CDIL extraction completed. Lock file: $OUTPUT_FILE"