#!/usr/bin/env python3
"""
Fix streaming timeout issues following Super Alita patterns.
Apply configuration fixes to resolve connection drops.
"""

import json
from pathlib import Path


def fix_uvicorn_config():
    """Apply uvicorn configuration fixes for streaming."""
    print("🔧 Applying Uvicorn Streaming Fixes...")

    # Check if there's a uvicorn config file
    config_files = [
        "uvicorn.json",
        "server.json",
        ".uvicorn.json"
    ]

    config_found = False
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ Found config: {config_file}")
            config_found = True
            break

    if not config_found:
        print("📝 Creating uvicorn configuration...")

        # Create optimized uvicorn config
        config = {
            "host": "127.0.0.1",
            "port": 8080,
            "reload": True,
            "timeout_keep_alive": 120,
            "timeout_graceful_shutdown": 30,
            "limit_concurrency": 100,
            "limit_max_requests": 1000,
            "workers": 1,
            "access_log": True,
            "use_colors": True
        }

        with open("uvicorn.json", "w") as f:
            json.dump(config, f, indent=2)

        print("✅ Created uvicorn.json with optimized streaming settings")

    # Print recommended startup command
    print("\n🚀 Recommended server startup:")
    print("uvicorn app:app --config uvicorn.json")
    print("\nOR with direct parameters:")
    print("uvicorn app:app --host 127.0.0.1 --port 8080 --reload --timeout-keep-alive 120")


def check_app_config():
    """Check app.py for streaming configuration."""
    print("\n🔍 Checking App Configuration...")

    app_py = Path("app.py")
    if app_py.exists():
        content = app_py.read_text()

        # Check for timeout configurations
        streaming_configs = [
            "timeout",
            "keep_alive",
            "lifespan",
            "chunk"
        ]

        found_configs = []
        for config in streaming_configs:
            if config in content.lower():
                found_configs.append(config)

        if found_configs:
            print(f"✅ Found streaming configs: {found_configs}")
        else:
            print("⚠️  No explicit streaming configs found")

        # Check for CORS configuration
        if "cors" in content.lower():
            print("✅ CORS configuration found")
        else:
            print("⚠️  Consider adding CORS for streaming")
    else:
        print("❌ app.py not found")


def create_streaming_test_server():
    """Create a simple test server to validate streaming."""
    print("\n🧪 Creating Streaming Test Server...")

    test_server_code = '''#!/usr/bin/env python3
"""
Simple streaming test server to validate connection handling.
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json
import time

app = FastAPI(title="Streaming Test Server")

@app.get("/test/stream")
async def test_stream():
    """Test streaming endpoint."""
    
    async def generate_stream():
        for i in range(5):
            data = {
                "chunk": i,
                "timestamp": time.time(),
                "message": f"Test chunk {i}"
            }
            yield f"data: {json.dumps(data)}\\n\\n"
            await asyncio.sleep(1)
        
        # Final chunk
        final_data = {"finished": True, "total_chunks": 5}
        yield f"data: {json.dumps(final_data)}\\n\\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.get("/test/health")
async def test_health():
    """Simple health check."""
    return {"status": "ok", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8081,
        timeout_keep_alive=120,
        access_log=True
    )
'''

    test_file = Path("test_streaming_server.py")
    test_file.write_text(test_server_code)

    print(f"✅ Created {test_file}")
    print("🚀 Run with: python test_streaming_server.py")
    print("🧪 Test with: curl http://127.0.0.1:8081/test/stream")


def main():
    """Main fix application following development instructions."""
    print("🔧 Streaming Connection Fix Tool")
    print("Following Super Alita Development Instructions")
    print("=" * 50)

    fix_uvicorn_config()
    check_app_config()
    create_streaming_test_server()

    print("\n" + "=" * 50)
    print("✅ Streaming fixes applied!")
    print("\n🎯 Next Steps:")
    print("1. Restart server with: uvicorn app:app --config uvicorn.json")
    print("2. Run: python debug_reug_streaming.py")
    print("3. If issues persist, test with: python test_streaming_server.py")
    print("4. Validate with: python test_consensus_integration.py")


if __name__ == "__main__":
    main()
