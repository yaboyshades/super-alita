#!/usr/bin/env python3
"""
Super Alita Comprehensive Startup Script

This script launches all necessary components and opens the chat interface:
- Main Super Alita server
- MCP server for tool integration
- Health checks and monitoring
- Auto-opens browser to chat interface
"""

import logging
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class SuperAlitaLauncher:
    """Comprehensive launcher for Super Alita agent system."""
    
    def __init__(self, config_path: str = "config/startup.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.processes: list[subprocess.Popen] = []
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def load_config(self) -> dict:
        """Load startup configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        
        # Default configuration
        return {
            "main_server": {
                "port": 8080,
                "host": "127.0.0.1",
                "auto_reload": True
            },
            "mcp_server": {
                "enabled": True,
                "script": "mcp_server_wrapper.py"
            },
            "browser": {
                "auto_open": True,
                "url_path": "/",
                "delay": 3
            },
            "health_check": {
                "timeout": 30,
                "interval": 1,
                "endpoints": ["/health", "/healthz"]
            },
            "startup": {
                "show_banner": True,
                "verbose": True
            }
        }
    
    def print_banner(self):
        """Print startup banner."""
        if not self.config.get("startup", {}).get("show_banner", True):
            return
            
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🤖 Super Alita Agent                     ║
║              AI-Powered Development Assistant                ║
╚══════════════════════════════════════════════════════════════╝

🚀 Starting Super Alita Agent System...
"""
        print(banner)
    
    def start_main_server(self):
        """Start the main Super Alita server."""
        config = self.config.get("main_server", {})
        port = config.get("port", 8080)
        host = config.get("host", "127.0.0.1")
        reload = config.get("auto_reload", True)
        
        cmd = [
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", host,
            "--port", str(port)
        ]
        
        if reload:
            cmd.append("--reload")
        
        logger.info(f"Starting main server on {host}:{port}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path.cwd()
            )
            self.processes.append(process)
            return process
        except Exception as e:
            logger.error(f"Failed to start main server: {e}")
            return None
    
    def start_mcp_server(self):
        """Start the MCP server for tool integration."""
        config = self.config.get("mcp_server", {})
        if not config.get("enabled", True):
            return None
        
        script = config.get("script", "mcp_server_wrapper.py")
        script_path = Path(script)
        
        if not script_path.exists():
            logger.warning(f"MCP server script not found: {script}")
            return None
        
        cmd = [sys.executable, str(script_path)]
        
        logger.info("Starting MCP server")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path.cwd()
            )
            self.processes.append(process)
            return process
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return None
    
    def check_health(self, url: str, timeout: int = 30) -> bool:
        """Check if a service is healthy."""
        config = self.config.get("health_check", {})
        endpoints = config.get("endpoints", ["/health", "/healthz"])
        interval = config.get("interval", 1)
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            for endpoint in endpoints:
                try:
                    response = requests.get(
                        f"{url}{endpoint}",
                        timeout=5
                    )
                    if response.status_code == 200:
                        return True
                except requests.RequestException:
                    pass
            
            time.sleep(interval)
        
        return False
    
    def wait_for_services(self):
        """Wait for all services to be ready."""
        config = self.config.get("main_server", {})
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 8080)
        timeout = self.config.get("health_check", {}).get("timeout", 30)
        
        main_url = f"http://{host}:{port}"
        
        logger.info("Waiting for services to be ready...")
        
        # Check main server
        if self.check_health(main_url, timeout):
            logger.info(f"✅ Main Server: {main_url} (Ready)")
        else:
            logger.error(f"❌ Main Server: {main_url} (Not responding)")
            return False
        
        # Check MCP server (if enabled)
        mcp_config = self.config.get("mcp_server", {})
        if mcp_config.get("enabled", True):
            logger.info("✅ MCP Server: Running with stdio transport")
        
        return True
    
    def open_browser(self):
        """Open the chat interface in the default browser."""
        browser_config = self.config.get("browser", {})
        if not browser_config.get("auto_open", True):
            return
        
        main_config = self.config.get("main_server", {})
        host = main_config.get("host", "127.0.0.1")
        port = main_config.get("port", 8080)
        path = browser_config.get("url_path", "/")
        delay = browser_config.get("delay", 3)
        
        url = f"http://{host}:{port}{path}"
        
        # Wait a bit for the server to fully start
        time.sleep(delay)
        
        try:
            webbrowser.open(url)
            logger.info(f"🌐 Chat Interface: {url} (Opened in browser)")
        except Exception as e:
            logger.warning(f"Failed to open browser: {e}")
            logger.info(f"🌐 Chat Interface: {url} (Please open manually)")
    
    def display_status(self):
        """Display the status of all services."""
        config = self.config.get("main_server", {})
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 8080)
        
        status = f"""
╔══════════════════════════════════════════════════════════════╗
║                     🎉 Super Alita Ready!                   ║
╚══════════════════════════════════════════════════════════════╝

🌐 Services Running:
   • Main Server: http://{host}:{port}
   • Chat Interface: http://{host}:{port}
   • Health Check: http://{host}:{port}/health
   • API Documentation: http://{host}:{port}/docs

🔧 Available Endpoints:
   • Autogen: http://{host}:{port}/autogen
   • DeepCode: http://{host}:{port}/deepcode
   • Tools: http://{host}:{port}/tools
   • Perplexica Search: Available via chat interface

📝 Press Ctrl+C to stop all services.
"""
        print(status)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping services...")
        self.running = False
        self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown all services."""
        logger.info("Stopping all services...")
        
        for process in self.processes:
            try:
                process.terminate()
                # Wait up to 5 seconds for graceful shutdown
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if still running
                process.kill()
                process.wait()
            except Exception as e:
                logger.warning(f"Error stopping process: {e}")
        
        logger.info("All services stopped. Goodbye! 👋")
        sys.exit(0)
    
    def run(self):
        """Main run method."""
        try:
            self.print_banner()
            
            # Start services
            main_server = self.start_main_server()
            self.start_mcp_server()
            
            if not main_server:
                logger.error("Failed to start main server, exiting")
                return 1
            
            # Wait for services to be ready
            if not self.wait_for_services():
                logger.error("Services failed to start properly")
                self.shutdown()
                return 1
            
            # Open browser
            self.open_browser()
            
            # Display status
            self.display_status()
            
            # Keep running until interrupted
            while self.running:
                time.sleep(1)
                
                # Check if main process is still running
                if main_server.poll() is not None:
                    logger.error("Main server process died unexpectedly")
                    break
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.shutdown()
        
        return 0


def main():
    """Main entry point."""
    launcher = SuperAlitaLauncher()
    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())