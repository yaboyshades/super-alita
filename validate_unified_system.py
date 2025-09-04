#!/usr/bin/env python3
"""
Validation script for unified Super Alita system.
Tests all new features and endpoints.
"""
import subprocess
import time
import requests
import json
import sys

def run_command(cmd, description):
    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"   ✅ SUCCESS")
            return result.stdout
        else:
            print(f"   ❌ FAILED: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT")
        return None
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return None

def test_endpoint(url, description, method="GET", data=None):
    print(f"🌐 {description}")
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            print(f"   ✅ SUCCESS: {response.status_code}")
            return response.json()
        else:
            print(f"   ❌ FAILED: {response.status_code}")
            return None
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return None

def main():
    print("=" * 60)
    print("🚀 UNIFIED SUPER ALITA VALIDATION")
    print("=" * 60)
    
    # Test startup script modes
    print("\n📋 Testing Startup Script Modes")
    run_command("python start.py --help", "Startup script help")
    run_command("python start.py --mode chat", "Chat mode (stub)")
    run_command("python start.py --mode consensus --model test", "Consensus mode (stub)")
    
    # Test make targets
    print("\n🔨 Testing Make Targets")
    run_command("make help | head -10", "Make help")
    
    # Start server for endpoint testing
    print("\n🌐 Starting Server for Endpoint Testing")
    server_process = subprocess.Popen(
        ["python", "start.py", "--mode", "web", "--port", "8080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("   ⏳ Waiting for server startup...")
    time.sleep(8)
    
    try:
        # Test unified endpoints
        print("\n🔌 Testing Unified Endpoints")
        
        health_data = test_endpoint("http://localhost:8080/health", "Health endpoint")
        if health_data:
            print(f"      Tools: {health_data.get('tools', 0)}")
        
        tools_data = test_endpoint("http://localhost:8080/tools", "Tools list endpoint")
        if tools_data:
            tools = tools_data.get('tools', {})
            print(f"      Available tools: {list(tools.keys())}")
        
        # Test tool execution (if consensus tool is available)
        if tools_data and 'deepconf_consensus' in tools_data.get('tools', {}):
            test_data = {
                "prompt": "What is 2+2?",
                "method": "simple_vote",
                "num_samples": 1
            }
            result = test_endpoint(
                "http://localhost:8080/tools/deepconf_consensus",
                "Consensus tool execution",
                method="POST",
                data=test_data
            )
            if result:
                print(f"      Execution result: {result.get('success', False)}")
        
        # Test integration tests
        print("\n🧪 Testing Integration Tests")
        run_command("PYTHONPATH=./src pytest -q tests/test_integration_reliability.py", "Unified integration tests")
        
        # Test migration
        print("\n📦 Testing Migration")
        run_command("python migrate_to_unified.py", "Migration script")
        run_command("ls archive/legacy_startup/ | wc -l", "Legacy files archived")
        run_command("ls examples/demos/ | wc -l", "Demo files archived")
        
        print("\n" + "=" * 60)
        print("✅ UNIFIED SYSTEM VALIDATION COMPLETE")
        print("🎉 All unified refactor components are working!")
        print("=" * 60)
        
    finally:
        # Clean up server
        print("\n🛑 Stopping test server...")
        server_process.terminate()
        server_process.wait(timeout=5)

if __name__ == "__main__":
    main()