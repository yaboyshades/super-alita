#!/usr/bin/env python3
"""
Unified launcher for Super Alita v3.0.

Consolidates multiple entry points into a single, configurable launcher
with feature flags and graceful degradation.

Adapted from GitHub pattern: arthurcolle/openai-mcp - unified CLI with mode selection
Reference: https://github.com/arthurcolle/openai-mcp/blob/main/cli.py#L2200
"""

import argparse
import logging
import os
import sys
from pathlib import Path

try:
    import uvicorn
    from rich.console import Console
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback console
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    
    class Panel:
        @staticmethod
        def fit(text, title="", border_style=""):
            return f"=== {title} ===\n{text}"
    
    console = Console()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

if RICH_AVAILABLE:
    console = Console()
else:
    console = Console()

class UnifiedLauncher:
    """Unified launcher with feature flags and mode selection."""
    
    def __init__(self):
        self.console = console
        self.modes = {
            "web": self._launch_web_server,
            "cli": self._launch_cli_mode,
            "dev": self._launch_dev_mode,
            "mcp": self._launch_mcp_server,
            "test": self._launch_test_mode,
            "research": self._launch_research_mode,
        }
        
    def _get_feature_flags(self) -> dict[str, bool]:
        """Get feature flags from environment variables."""
        return {
            "github_demo": os.getenv("ENABLE_GITHUB_DEMO", "false").lower() == "true",
            "perplexica_demo": os.getenv("ENABLE_PERPLEXICA_DEMO", "false").lower() == "true", 
            "autogen_demo": os.getenv("ENABLE_AUTOGEN_DEMO", "false").lower() == "true",
            "consensus_enhanced": os.getenv("ENABLE_ENHANCED_CONSENSUS", "true").lower() == "true",
            "dev_mode": os.getenv("SUPER_ALITA_DEV", "false").lower() == "true",
            "research_enabled": os.getenv("RESEARCH_ENABLED", "false").lower() == "true",
        }
    
    def _setup_logging(self, verbose: bool = False):
        """Configure logging based on verbosity."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    def _check_prerequisites(self) -> dict[str, bool]:
        """Check system prerequisites and dependencies."""
        checks = {}
        
        # Check Python version
        checks["python_version"] = sys.version_info >= (3, 9)
        
        # Check environment file
        checks["env_file"] = Path(".env").exists()
        
        # Check critical imports
        try:
            import fastapi
            import uvicorn
            checks["fastapi"] = True
        except ImportError:
            checks["fastapi"] = False
            
        try:
            from src.main import create_app
            checks["main_app"] = True
        except ImportError as e:
            checks["main_app"] = False
            print(f"Main app import error: {e}")
            
        return checks
    
    def _display_startup_info(self, mode: str, **kwargs):
        """Display startup information panel."""
        flags = self._get_feature_flags()
        enabled_features = [k for k, v in flags.items() if v]
        
        info_text = "Super Alita v3.0 Unified Launcher\n"
        info_text += f"Mode: {mode}\n"
        info_text += f"Features: {', '.join(enabled_features) or 'None'}\n"
        
        if kwargs:
            for key, value in kwargs.items():
                info_text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        if RICH_AVAILABLE:
            self.console.print(Panel.fit(
                info_text.strip(),
                title="Starting Super Alita",
                border_style="green"
            ))
        else:
            print(f"=== Starting Super Alita ===\n{info_text}")
    
    def _launch_web_server(self, host: str = "127.0.0.1", port: int = 8080, **kwargs):
        """Launch main web server mode."""
        try:
            from src.main import create_app
            
            self._display_startup_info("web", host=host, port=port)
            
            app = create_app()
            
            if not app:
                self.console.print("❌ Failed to create FastAPI app")
                return False
            
            # Uvicorn configuration
            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="info",
                access_log=True,
                reload=kwargs.get("reload", False),
                workers=kwargs.get("workers", 1),
            )
            
            server = uvicorn.Server(config)
            server.run()
            
            return True
            
        except ImportError as e:
            self.console.print(f"❌ Error: Failed to import main app - {e}")
            return False
        except Exception as e:
            self.console.print(f"❌ Error starting web server: {e}")
            return False
    
    def _launch_cli_mode(self, **kwargs):
        """Launch CLI interface mode."""
        try:
            from src.vscode_integration.agent_mcp_server import main as cli_main
            
            self._display_startup_info("cli")
            self.console.print("Starting CLI mode...")
            
            return cli_main()
            
        except ImportError:
            self.console.print("❌ Error: CLI mode not available")
            return False
        except Exception as e:
            self.console.print(f"❌ Error in CLI mode: {e}")
            return False
    
    def _launch_dev_mode(self, **kwargs):
        """Launch development mode with auto-reload."""
        self._display_startup_info("dev", reload=True, debug=True)
        
        # Enable debug features
        os.environ["SUPER_ALITA_DEV"] = "true"
        os.environ["ENABLE_GITHUB_DEMO"] = "true" 
        os.environ["ENABLE_PERPLEXICA_DEMO"] = "true"
        
        # Launch with reload
        return self._launch_web_server(reload=True, **kwargs)
    
    def _launch_mcp_server(self, **kwargs):
        """Launch MCP server mode."""
        try:
            from mcp_server.server import main as mcp_main
            
            self._display_startup_info("mcp")
            return mcp_main()
            
        except ImportError:
            self.console.print("❌ Error: MCP server not available")
            return False
        except Exception as e:
            self.console.print(f"❌ Error starting MCP server: {e}")
            return False
    
    def _launch_test_mode(self, **kwargs):
        """Launch test mode for validation."""
        self._display_startup_info("test")
        
        try:
            import subprocess
            
            # Run deployment validation
            self.console.print("Running deployment validation...")
            result = subprocess.run([sys.executable, "validate_deployment.py"], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.console.print("✅ All validation tests passed!")
                return True
            else:
                self.console.print(f"❌ Validation failed:\n{result.stdout}\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.console.print("❌ Validation timed out")
            return False
        except Exception as e:
            self.console.print(f"❌ Error running validation: {e}")
            return False
    
    def _launch_research_mode(self, **kwargs):
        """Launch research mode with advanced capabilities."""
        self._display_startup_info("research", experimental=True)
        
        # Check if research dependencies are available
        try:
            import torch
            import transformers
            self.console.print("✅ Research dependencies available")
        except ImportError as e:
            self.console.print(f"❌ Missing research dependencies: {e}")
            self.console.print("Install with: pip install -r requirements-research.txt")
            return False
        
        # Check if research components exist
        research_path = Path("src/research")
        if not research_path.exists():
            self.console.print("❌ Research components not found")
            self.console.print("Merge PR #330 to get research capabilities")
            return False
        
        try:
            # Set environment for research mode
            os.environ["RESEARCH_ENABLED"] = "true"
            os.environ["ALITA_ENABLE_Z3"] = "true"
            
            # Import and run research application
            from src.main_research import main as research_main
            
            self.console.print("🔬 Starting Super Alita Research Edition...")
            import asyncio
            asyncio.run(research_main())
            return True
            
        except ImportError as e:
            self.console.print(f"❌ Research mode import failed: {e}")
            self.console.print("Research components may not be installed")
            return False
        except KeyboardInterrupt:
            self.console.print("\n⚠️ Research demo interrupted")
            return True
        except Exception as e:
            self.console.print(f"❌ Research mode failed: {e}")
            return False
    
    def run(self, mode: str, **kwargs) -> bool:
        """Run the launcher in specified mode."""
        # Setup logging
        self._setup_logging(kwargs.get("verbose", False))
        
        # Check prerequisites
        checks = self._check_prerequisites()
        failed_checks = [k for k, v in checks.items() if not v]
        
        if failed_checks:
            self.console.print(f"❌ Prerequisites failed: {', '.join(failed_checks)}")
            
            # Provide helpful suggestions
            if "python_version" in failed_checks:
                self.console.print("Please upgrade to Python 3.9+")
            if "env_file" in failed_checks:
                self.console.print("Hint: Copy .env.example to .env")
            if "fastapi" in failed_checks:
                self.console.print("Hint: Run 'pip install -r requirements.txt'")
            if "main_app" in failed_checks:
                self.console.print("Hint: Check src/main.py imports")
                
            return False
        
        # Validate mode
        if mode not in self.modes:
            self.console.print(f"❌ Unknown mode: {mode}")
            self.console.print(f"Available modes: {', '.join(self.modes.keys())}")
            return False
        
        # Run mode
        return self.modes[mode](**kwargs)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Super Alita v3.0 Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python start.py --mode web                    # Start web server
  python start.py --mode web --port 8080       # Custom port
  python start.py --mode dev                   # Development mode  
  python start.py --mode cli                   # CLI interface
  python start.py --mode mcp                   # MCP server
  python start.py --mode test                  # Run validation tests
  python start.py --mode research              # Research edition

Environment Variables:
  ENABLE_GITHUB_DEMO=true                      # Enable GitHub integration demo
  ENABLE_PERPLEXICA_DEMO=true                  # Enable Perplexica search demo
  ENABLE_AUTOGEN_DEMO=true                     # Enable AutoGen pipeline demo
  SUPER_ALITA_DEV=true                         # Enable development features
  RESEARCH_ENABLED=true                        # Enable research capabilities
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["web", "cli", "dev", "mcp", "test", "research"],
        default="web",
        help="Launch mode (default: web)"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1", 
        help="Host address (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number (default: 8080)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create launcher and run
    launcher = UnifiedLauncher()
    
    try:
        success = launcher.run(
            mode=args.mode,
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            verbose=args.verbose
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        console.print("\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()