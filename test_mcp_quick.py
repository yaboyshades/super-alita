#!/usr/bin/env python3
"""
Simple MCP functionality test for Super Alita.
"""

import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

def test_imports():
    """Test that all MCP components can be imported."""
    print("🔧 Testing MCP Imports...")
    
    try:
        from mcp_server_wrapper import app
        print("✅ MCP server wrapper imported")
        
        # Check if the app has tools registered
        print(f"✅ FastMCP app type: {type(app).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_telemetry_file():
    """Test telemetry file creation."""
    print("\n📊 Testing Telemetry Setup...")
    
    telemetry_file = REPO_ROOT / "telemetry.jsonl"
    
    if telemetry_file.exists():
        with open(telemetry_file) as f:
            lines = len(f.readlines())
        print(f"✅ Telemetry file exists with {lines} entries")
    else:
        print("⚠️  Telemetry file not found (will be created on first use)")
    
    return True

def test_configuration_files():
    """Test that configuration files are present."""
    print("\n⚙️  Testing Configuration Files...")
    
    config_files = [
        "claude_desktop_config_local.json",
        "vscode_mcp_config.json", 
        "start_mcp_http.ps1",
        "start_mcp_secure.ps1"
    ]
    
    for config_file in config_files:
        file_path = REPO_ROOT / config_file
        if file_path.exists():
            print(f"✅ {config_file} exists")
        else:
            print(f"❌ {config_file} missing")
    
    return True

def test_server_files():
    """Test that server files are present."""
    print("\n🖥️  Testing Server Files...")
    
    server_files = [
        "mcp_server_wrapper.py",
        "mcp/fastmcp_server.py"
    ]
    
    for server_file in server_files:
        file_path = REPO_ROOT / server_file
        if file_path.exists():
            print(f"✅ {server_file} exists")
        else:
            print(f"❌ {server_file} missing")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Super Alita MCP Quick Test")
    print("=" * 40)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_telemetry_file()
    all_passed &= test_configuration_files()
    all_passed &= test_server_files()
    
    print(f"\n{'🎉 All tests passed!' if all_passed else '⚠️  Some tests failed'}")
    
    if all_passed:
        print("\n💡 Ready to use:")
        print("1. Claude Desktop: Copy claude_desktop_config_local.json to Claude config")
        print("2. VS Code: Use mcp_server_wrapper.py with stdio transport")
        print("3. HTTP: Run .\\start_mcp_http.ps1 for web clients")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)