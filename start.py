#!/usr/bin/env python3
"""
Unified launcher for Super Alita v3.0.

Consolidates multiple entry points into a single, configurable launcher
with feature flags and graceful degradation.

Adapted from GitHub pattern: arthurcolle/openai-mcp - unified CLI with mode selection
Reference: https://github.com/arthurcolle/openai-mcp/blob/main/cli.py#L2200
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from rich.console import Console
from rich.panel import Panel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

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
        }
        
    def _get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags from environment variables."""
        return {
            "github_demo": os.getenv("ENABLE_GITHUB_DEMO", "false").lower() == "true",
            "perplexica_demo": os.getenv("ENABLE_PERPLEXICA_DEMO", "false").lower() == "true", 
            "autogen_demo": os.getenv("ENABLE_AUTOGEN_DEMO", "false").lower() == "true",
            "consensus_enhanced": os.getenv("ENABLE_ENHANCED_CONSENSUS", "true").lower() == "true",
            "dev_mode": os.getenv("SUPER_ALITA_DEV", "false").lower() == "true",
        }
    
    def _setup_logging(self, verbose: bool = False):
        """Configure logging based on verbosity."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    def _check_prerequisites(self) -> Dict[str, bool]:
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
        except ImportError:
            checks["main_app"] = False
            
        return checks
    
    def _display_startup_info(self, mode: str, **kwargs):
        """Display startup information panel."""
        flags = self._get_feature_flags()
        enabled_features = [k for k, v in flags.items() if v]
        
        info_text = f"[bold green]Super Alita v3.0 Unified Launcher[/bold green]\n"
        info_text += f"Mode: {mode}\n"
        info_text += f"Features: {', '.join(enabled_features) or 'None'}\n"
        
        if kwargs:
            for key, value in kwargs.items():
                info_text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        self.console.print(Panel.fit(
            info_text.strip(),
            title="Starting Super Alita",
            border_style="green"
        ))
    
    def _launch_web_server(self, host: str = "127.0.0.1", port: int = 8080, **kwargs):
        """Launch main web server mode."""
        try:
            from src.main import create_app
            
            self._display_startup_info("web", host=host, port=port)
            
            # Import and run with timeout/retry patterns from GitHub examples
            app = create_app()
            
            # Uvicorn configuration adapted from production patterns
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
            
        except ImportError as e:
            self.console.print(f"[red]Error: Failed to import main app - {e}[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]Error starting web server: {e}[/red]")
            return False
        
        return True
    
    def _launch_cli_mode(self, **kwargs):
        """Launch CLI interface mode."""
        try:
            # Import CLI dependencies
            from src.vscode_integration.agent_mcp_server import main as cli_main
            
            self._display_startup_info("cli")
            self.console.print("[yellow]Starting CLI mode...[/yellow]")
            
            # Run CLI with feature flag support
            return cli_main()
            
        except ImportError:
            self.console.print("[red]Error: CLI mode not available[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]Error in CLI mode: {e}[/red]")
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
            self.console.print("[red]Error: MCP server not available[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]Error starting MCP server: {e}[/red]")
            return False
    
    def _launch_test_mode(self, **kwargs):
        """Launch test mode for validation."""
        self._display_startup_info("test")
        
        try:
            import subprocess
            
            # Run deployment validation
            self.console.print("[yellow]Running deployment validation...[/yellow]")
            result = subprocess.run([sys.executable, "validate_deployment.py"], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.console.print("[green]✅ All validation tests passed![/green]")
                return True
            else:
                self.console.print(f"[red]❌ Validation failed:\n{result.stdout}\n{result.stderr}[/red]")
                return False
                
        except subprocess.TimeoutExpired:
            self.console.print("[red]❌ Validation timed out[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]❌ Error running validation: {e}[/red]")
            return False
    
    def run(self, mode: str, **kwargs) -> bool:
        """Run the launcher in specified mode."""
        # Setup logging
        self._setup_logging(kwargs.get("verbose", False))
        
        # Check prerequisites
        checks = self._check_prerequisites()
        failed_checks = [k for k, v in checks.items() if not v]
        
        if failed_checks:
            self.console.print(f"[red]❌ Prerequisites failed: {', '.join(failed_checks)}[/red]")
            
            # Provide helpful suggestions
            if "python_version" in failed_checks:
                self.console.print("[yellow]Please upgrade to Python 3.9+[/yellow]")
            if "env_file" in failed_checks:
                self.console.print("[yellow]Hint: Copy .env.example to .env[/yellow]")
            if "fastapi" in failed_checks:
                self.console.print("[yellow]Hint: Run 'pip install -r requirements.txt'[/yellow]")
                
            return False
        
        # Validate mode
        if mode not in self.modes:
            self.console.print(f"[red]❌ Unknown mode: {mode}[/red]")
            self.console.print(f"Available modes: {', '.join(self.modes.keys())}")
            return False
        
        # Run mode
        return self.modes[mode](**kwargs)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Super Alita v3.0 Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py --mode web                    # Start web server
  python start.py --mode web --port 8080       # Custom port
  python start.py --mode dev                   # Development mode  
  python start.py --mode cli                   # CLI interface
  python start.py --mode mcp                   # MCP server
  python start.py --mode test                  # Run validation tests

Environment Variables:
  ENABLE_GITHUB_DEMO=true                      # Enable GitHub integration demo
  ENABLE_PERPLEXICA_DEMO=true                  # Enable Perplexica search demo
  ENABLE_AUTOGEN_DEMO=true                     # Enable AutoGen pipeline demo
  SUPER_ALITA_DEV=true                         # Enable development features
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["web", "cli", "dev", "mcp", "test"],
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
        console.print("\n[yellow]⚠️ Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()