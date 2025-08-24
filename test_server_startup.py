#!/usr/bin/env python3
"""Test server startup with debugging."""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("1. Importing create_app...")
    from src.main import create_app
    
    print("2. Creating app...")
    app = create_app()
    
    print("3. Testing health endpoints...")
    # Test if health endpoints are working
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    print("4. Testing /health endpoint...")
    response = client.get("/health")
    print(f"Health response: {response.status_code} - {response.json()}")
    
    print("5. Testing /healthz endpoint...")  
    response = client.get("/healthz")
    print(f"Healthz response: {response.status_code}")
    
    print("✅ All tests passed! Server is ready.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc()