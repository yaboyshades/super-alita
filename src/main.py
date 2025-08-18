"""
Super Alita: Production-Grade Autonomous Cognitive Agent
Main orchestrator and plugin manager.
"""

import asyncio
import sys
import signal
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Core imports
from .core.event_bus import EventBus, get_global_bus
from .core.neural_atom import NeuralStore
from .core.genealogy import GenealogyTracer, get_global_tracer
from .core.plugin_interface import PluginInterface

# Plugin imports
from .plugins.event_bus_plugin import EventBusPlugin
from .plugins.semantic_memory_plugin import SemanticMemoryPlugin
from .plugins.semantic_fsm_plugin import SemanticFSMPlugin
from .plugins.skill_discovery_plugin import SkillDiscoveryPlugin
from .plugins.ladder_aog_plugin import LADDERAOGPlugin


class SuperAlita:
    """
    Super Alita autonomous cognitive agent.
    
    Orchestrates all plugins and manages the agent lifecycle through
    an event-driven architecture with genealogy tracking.
    """
    
    def __init__(self, config_path: str = "src/config/agent.yaml"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.event_bus: Optional[EventBus] = None
        self.neural_store = NeuralStore()
        self.genealogy_tracer: Optional[GenealogyTracer] = None
        self.plugins: Dict[str, PluginInterface] = {}
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.shutdown_handlers: List[asyncio.Task] = []
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Basic logging setup (will be enhanced after config load)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/super_alita.log')
            ]
        )
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            self.config = self._get_default_config()
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing configuration: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is missing."""
        
        return {
            "agent": {"name": "Super Alita", "version": "1.0.0"},
            "event_bus": {"max_history": 1000, "max_workers": 4},
            "neural_store": {"max_atoms": 10000},
            "plugins": {
                "semantic_memory": {"enabled": True},
                "semantic_fsm": {"enabled": True},
                "skill_discovery": {"enabled": True}
            },
            "genealogy": {"export_interval_hours": 24},
            "logging": {"level": "INFO"}
        }
    
    def _configure_logging(self) -> None:
        """Configure logging based on loaded config."""
        
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        
        # Update log level
        logging.getLogger().setLevel(level)
        
        self.logger.info(f"Logging configured at level: {log_config.get('level', 'INFO')}")
    
    def _register_plugins(self) -> None:
        """Register all available plugins."""
        
        plugin_config = self.config.get("plugins", {})
        
        # EventBus Plugin (core communication)
        self.plugins["event_bus"] = EventBusPlugin()
        
        # Semantic Memory Plugin
        if plugin_config.get("semantic_memory", {}).get("enabled", True):
            self.plugins["semantic_memory"] = SemanticMemoryPlugin()
        
        # Semantic FSM Plugin
        if plugin_config.get("semantic_fsm", {}).get("enabled", True):
            self.plugins["semantic_fsm"] = SemanticFSMPlugin()
        
        # Skill Discovery Plugin
        if plugin_config.get("skill_discovery", {}).get("enabled", True):
            self.plugins["skill_discovery"] = SkillDiscoveryPlugin()
        
        # LADDER-AOG Reasoning Plugin
        if plugin_config.get("ladder_aog", {}).get("enabled", True):
            self.plugins["ladder_aog"] = LADDERAOGPlugin()
        
        self.logger.info(f"Registered {len(self.plugins)} plugins: {list(self.plugins.keys())}")
    
    async def initialize(self) -> None:
        """Initialize the agent and all components."""
        
        self.logger.info("Initializing Super Alita agent...")
        
        # Load configuration
        self.load_config()
        self._configure_logging()
        
        # Initialize event bus
        self.event_bus = await get_global_bus()
        self.logger.info("Event bus initialized")
        
        # Initialize genealogy tracer
        self.genealogy_tracer = get_global_tracer()
        self.logger.info("Genealogy tracer initialized")
        
        # Register plugins
        self._register_plugins()
        
        # Initialize plugins
        await self._initialize_plugins()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        self.logger.info("Super Alita agent initialized successfully")
    
    async def _initialize_plugins(self) -> None:
        """Initialize all registered plugins."""
        
        # Create a copy of the plugins dictionary to avoid iteration issues
        plugins_to_initialize = list(self.plugins.items())
        
        for plugin_name, plugin in plugins_to_initialize:
            try:
                plugin_config = self.config.get("plugins", {}).get(plugin_name, {})
                
                self.logger.info(f"Initializing plugin: {plugin_name}")
                await plugin.setup(self.event_bus, self.neural_store, plugin_config)
                
                self.logger.info(f"Plugin {plugin_name} setup complete")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
                # Remove failed plugin
                if plugin_name in self.plugins:
                    del self.plugins[plugin_name]
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self) -> None:
        """Start the agent and all plugins."""
        
        if self.is_running:
            self.logger.warning("Agent is already running")
            return
        
        self.logger.info("Starting Super Alita agent...")
        self.start_time = datetime.utcnow()
        self.is_running = True
        
        # Start plugins
        for plugin_name, plugin in self.plugins.items():
            try:
                self.logger.info(f"Starting plugin: {plugin_name}")
                await plugin.start()
                self.logger.info(f"Plugin {plugin_name} started successfully")
            except Exception as e:
                self.logger.error(f"Failed to start plugin {plugin_name}: {e}")
        
        # Emit agent start event
        await self.event_bus.emit(
            "system",
            source_plugin="super_alita",
            level="info",
            message="agent_started",
            component="main",
            metadata={
                "agent_name": self.config.get("agent", {}).get("name", "Super Alita"),
                "version": self.config.get("agent", {}).get("version", "1.0.0"),
                "plugins": list(self.plugins.keys()),
                "start_time": self.start_time.isoformat()
            }
        )
        
        # Start background tasks
        await self._start_background_tasks()
        
        self.logger.info("Super Alita agent started successfully")
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitor())
        self.shutdown_handlers.append(health_task)
        
        # Genealogy export task
        export_task = asyncio.create_task(self._periodic_genealogy_export())
        self.shutdown_handlers.append(export_task)
        
        # Stats reporting task
        stats_task = asyncio.create_task(self._periodic_stats_report())
        self.shutdown_handlers.append(stats_task)
    
    async def _health_monitor(self) -> None:
        """Monitor agent and plugin health."""
        
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check plugin health
                for plugin_name, plugin in self.plugins.items():
                    try:
                        health = await plugin.health_check()
                        
                        if health.get("status") != "healthy":
                            self.logger.warning(f"Plugin {plugin_name} health issue: {health}")
                            
                            # Emit health event
                            await self.event_bus.emit(
                                "health_check",
                                source_plugin="super_alita",
                                component=plugin_name,
                                status=health.get("status", "unknown"),
                                metrics=health.get("metrics", {}),
                                recommendations=health.get("issues", [])
                            )
                    
                    except Exception as e:
                        self.logger.error(f"Health check failed for {plugin_name}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
    
    async def _periodic_genealogy_export(self) -> None:
        """Periodically export genealogy data."""
        
        export_interval = self.config.get("genealogy", {}).get("export_interval_hours", 24) * 3600
        
        while self.is_running:
            try:
                await asyncio.sleep(export_interval)
                
                if self.genealogy_tracer:
                    # Create export directory
                    export_dir = Path(self.config.get("genealogy", {}).get("export_directory", "data/genealogy"))
                    export_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Export genealogy
                    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    graphml_path = export_dir / f"genealogy_{timestamp}.graphml"
                    json_path = export_dir / f"genealogy_{timestamp}.json"
                    
                    self.genealogy_tracer.export_to_graphml(str(graphml_path))
                    self.genealogy_tracer.export_to_json(str(json_path))
                    
                    self.logger.info(f"Genealogy exported to {graphml_path} and {json_path}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in genealogy export: {e}")
    
    async def _periodic_stats_report(self) -> None:
        """Periodically report agent statistics."""
        
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes
                
                stats = await self.get_agent_stats()
                
                # Emit stats event
                await self.event_bus.emit(
                    "system",
                    source_plugin="super_alita",
                    level="info",
                    message="stats_report",
                    component="main",
                    stats=stats
                )
                
                self.logger.info(f"Agent stats: {stats}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in stats report: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the agent gracefully."""
        
        if not self.is_running:
            return
        
        self.logger.info("Shutting down Super Alita agent...")
        self.is_running = False
        
        # Cancel background tasks
        for task in self.shutdown_handlers:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop plugins
        for plugin_name, plugin in self.plugins.items():
            try:
                self.logger.info(f"Stopping plugin: {plugin_name}")
                await plugin.stop()
                self.logger.info(f"Plugin {plugin_name} stopped")
            except Exception as e:
                self.logger.error(f"Error stopping plugin {plugin_name}: {e}")
        
        # Stop event bus
        if self.event_bus:
            await self.event_bus.stop()
        
        # Final genealogy export
        if self.genealogy_tracer:
            try:
                export_dir = Path("data/genealogy")
                export_dir.mkdir(parents=True, exist_ok=True)
                
                final_export = export_dir / f"final_genealogy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.graphml"
                self.genealogy_tracer.export_to_graphml(str(final_export))
                self.logger.info(f"Final genealogy exported to {final_export}")
            except Exception as e:
                self.logger.error(f"Error in final genealogy export: {e}")
        
        self.logger.info("Super Alita agent shutdown complete")
    
    async def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        
        runtime_seconds = 0
        if self.start_time:
            runtime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Get neural store stats
        store_stats = self.neural_store.get_stats()
        
        # Get event bus stats
        bus_stats = await self.event_bus.get_stats() if self.event_bus else {}
        
        # Get genealogy stats
        genealogy_stats = {}
        if self.genealogy_tracer:
            genealogy_stats = self.genealogy_tracer.get_statistics()
        
        # Get plugin stats
        plugin_stats = {}
        for plugin_name, plugin in self.plugins.items():
            try:
                health = await plugin.health_check()
                plugin_stats[plugin_name] = {
                    "status": health.get("status", "unknown"),
                    "version": plugin.version
                }
            except:
                plugin_stats[plugin_name] = {"status": "error"}
        
        return {
            "agent": {
                "name": self.config.get("agent", {}).get("name", "Super Alita"),
                "version": self.config.get("agent", {}).get("version", "1.0.0"),
                "runtime_seconds": runtime_seconds,
                "is_running": self.is_running
            },
            "plugins": plugin_stats,
            "neural_store": store_stats,
            "event_bus": bus_stats,
            "genealogy": genealogy_stats
        }
    
    async def process_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """Process a command and return results."""
        
        command_lower = command.lower()
        
        if command_lower == "status":
            return await self.get_agent_stats()
        
        elif command_lower == "health":
            health_reports = {}
            for plugin_name, plugin in self.plugins.items():
                health_reports[plugin_name] = await plugin.health_check()
            return {"health_reports": health_reports}
        
        elif command_lower == "export_genealogy":
            if self.genealogy_tracer:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                export_path = f"data/genealogy_export_{timestamp}.graphml"
                
                Path("data").mkdir(exist_ok=True)
                self.genealogy_tracer.export_to_graphml(export_path)
                
                return {"message": f"Genealogy exported to {export_path}"}
            return {"error": "Genealogy tracer not available"}
        
        elif command_lower.startswith("emit_event"):
            # Parse event emission command
            parts = command.split(" ", 2)
            if len(parts) >= 2:
                event_type = parts[1]
                data = kwargs
                
                await self.event_bus.emit(
                    event_type,
                    source_plugin="command_interface",
                    **data
                )
                
                return {"message": f"Event '{event_type}' emitted"}
        
        elif command_lower == "trigger_skill_discovery":
            await self.event_bus.emit(
                "system",
                source_plugin="command_interface",
                message="skill_discovery_requested"
            )
            return {"message": "Skill discovery triggered"}
        
        elif command_lower == "trigger_evolution":
            await self.event_bus.emit(
                "system",
                source_plugin="command_interface", 
                message="evolution_requested"
            )
            return {"message": "Evolution triggered"}
        
        else:
            return {"error": f"Unknown command: {command}"}
    
    async def run_interactive(self) -> None:
        """Run agent in interactive mode."""
        
        print("Super Alita Interactive Mode")
        print("Commands: status, health, export_genealogy, trigger_skill_discovery, trigger_evolution, quit")
        
        while self.is_running:
            try:
                command = input("\n> ").strip()
                
                if command.lower() in ["quit", "exit", "q"]:
                    break
                
                if command:
                    result = await self.process_command(command)
                    print(f"Result: {result}")
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        await self.shutdown()


async def main():
    """Main entry point."""
    
    # Create agent
    agent = SuperAlita()
    
    try:
        # Initialize agent
        await agent.initialize()
        
        # Start agent
        await agent.start()
        
        # Check if running interactively
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            await agent.run_interactive()
        else:
            # Run until interrupted
            print("Super Alita is running. Press Ctrl+C to stop.")
            try:
                while agent.is_running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                pass
    
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        return 1
    
    finally:
        await agent.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
