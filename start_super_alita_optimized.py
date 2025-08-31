#!/usr/bin/env python3
"""
Optimized Super Alita startup for 20B model with proper timeout handling
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Set up environment early
os.environ.update({
    "LLM_MODEL": "ollama:gpt-oss:20b",
    "OLLAMA_HOST": "http://127.0.0.1:11434",
    "PYTHONPATH": "./src",
    # Increase timeouts for large models
    "REUG_MODEL_STREAM_TIMEOUT_S": "120.0",  # 2 minutes for model responses
    "STARTUP_TIMEOUT": "180",  # 3 minutes for startup
    # Reduce memory pressure during startup
    "REUG_MAX_TOOL_CALLS": "3",
    "OLLAMA_KEEP_ALIVE": "5m",  # Keep model in memory longer
})

def start_server_with_timeout():
    """Start the server with proper timeout handling"""
    print("🚀 Starting Super Alita with optimized 20B configuration...")
    
    # Create a modified startup script that handles timeouts better
    startup_script = """
import asyncio
import os
import sys
import uvicorn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def start_app():
    try:
        # Import the main app
        from main import app
        
        # Configure uvicorn with larger timeouts
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=8080,
            timeout_keep_alive=60,
            timeout_graceful_shutdown=30,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        
        print("🌐 Starting server on http://127.0.0.1:8080")
        await server.serve()
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(start_app()))
"""
    
    # Write the optimized startup script
    script_path = Path("start_optimized.py")
    script_path.write_text(startup_script)
    
    try:
        # Start with timeout
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print("⏱️ Waiting for server to start (max 3 minutes)...")
        
        # Monitor output for 3 minutes
        start_time = time.time()
        timeout = 180  # 3 minutes
        
        while process.poll() is None:
            if time.time() - start_time > timeout:
                print("⏰ Startup timeout - terminating process")
                process.terminate()
                process.wait(timeout=10)
                return False
                
            # Read output
            try:
                line = process.stdout.readline()
                if line:
                    print(line.rstrip())
                    if "Application startup complete" in line or "Uvicorn running" in line:
                        print("✅ Server started successfully!")
                        break
            except:
                pass
                
            time.sleep(0.1)
        
        if process.poll() is None:
            # Server is running, wait for it
            print("🎉 Server is running! Press Ctrl+C to stop.")
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                process.terminate()
                process.wait(timeout=10)
            return True
        else:
            print(f"❌ Server process exited with code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False
    finally:
        # Clean up
        if script_path.exists():
            script_path.unlink()

def verify_prerequisites():
    """Verify all prerequisites are met"""
    print("🔍 Verifying prerequisites...")
    
    # Check Ollama
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10)
        if "gpt-oss:20b" not in result.stdout:
            print("⚠️ GPT-OSS 20B not loaded, loading now...")
            subprocess.run(["ollama", "run", "gpt-oss:20b", "ready"], timeout=60)
        print("✅ Ollama and 20B model ready")
    except Exception as e:
        print(f"❌ Ollama check failed: {e}")
        return False
    
    # Check Python environment
    if not Path("src/main.py").exists():
        print("❌ main.py not found")
        return False
    
    print("✅ All prerequisites met")
    return True

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            🤖 Super Alita Optimized for 20B Model           ║")
    print("║              Enhanced Timeout & Memory Handling             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    if not verify_prerequisites():
        return 1
    
    print("🔧 Environment configured:")
    print(f"  • Model: {os.environ['LLM_MODEL']}")
    print(f"  • Timeout: {os.environ['REUG_MODEL_STREAM_TIMEOUT_S']}s")
    print(f"  • Host: {os.environ['OLLAMA_HOST']}")
    print()
    
    success = start_server_with_timeout()
    
    if success:
        print("\n✅ Super Alita completed successfully")
        return 0
    else:
        print("\n❌ Super Alita failed to start")
        print("\n💡 Suggestions:")
        print("  • Check if port 8080 is available")
        print("  • Verify Ollama is running: ollama ps")
        print("  • Try with a smaller model first")
        return 1

if __name__ == "__main__":
    sys.exit(main())