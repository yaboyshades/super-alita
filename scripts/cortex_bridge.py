#!/usr/bin/env python3
"""
Cortex Bridge: VS Code → Cortex Event Ingestion
Tails JSONL telemetry and feeds events to Cortex orchestrator
"""

import asyncio
import json
import time
import pathlib
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

# Import Cortex components (adjust paths as needed)
try:
    from src.core.events import Event, create_event
    from src.core.orchestrator import Orchestrator
    from src.core.event_bus import EventBus
except ImportError:
    # Fallback for development
    print("Warning: Cortex components not found, running in mock mode")
    
    @dataclass
    class Event:
        id: str
        kind: str
        ts: float
        actor: str
        payload: Dict[str, Any]
        schema_version: str = "v1"
    
    class Orchestrator:
        def add_event(self, event: Event):
            print(f"[MOCK] Added event: {event.kind} from {event.actor}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CortexBridge:
    """Bridge between VS Code telemetry and Cortex orchestrator"""
    
    def __init__(self, telemetry_path: Optional[pathlib.Path] = None):
        self.telemetry_path = telemetry_path or pathlib.Path.home() / ".super-alita" / "telemetry.jsonl"
        self.orchestrator: Optional[Orchestrator] = None
        self.running = False
        
        # Ensure telemetry directory exists
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Touch the file if it doesn't exist
        if not self.telemetry_path.exists():
            self.telemetry_path.touch()
    
    async def initialize_cortex(self):
        """Initialize Cortex orchestrator (adapt to your setup)"""
        try:
            # TODO: Initialize your actual Cortex components here
            # self.orchestrator = await setup_cortex_orchestrator()
            self.orchestrator = Orchestrator()  # Mock for now
            logger.info("Cortex orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Cortex: {e}")
            self.orchestrator = Orchestrator()  # Fallback mock
    
    def transform_vscode_event(self, vscode_event: Dict[str, Any]) -> Event:
        """Transform VS Code event to Cortex Event format"""
        try:
            return Event(
                id=vscode_event.get('id', ''),
                kind=vscode_event.get('kind', 'UNKNOWN'),
                ts=vscode_event.get('ts', time.time()),
                actor=vscode_event.get('actor', 'unknown'),
                payload=vscode_event.get('payload', {}),
                schema_version=vscode_event.get('schema_version', 'v1')
            )
        except Exception as e:
            logger.error(f"Failed to transform event: {e}")
            return None
    
    async def tail_jsonl(self):
        """Tail the JSONL file and feed events to Cortex"""
        logger.info(f"Starting to tail {self.telemetry_path}")
        
        with open(self.telemetry_path, 'r', encoding='utf-8') as fp:
            # Seek to end of file
            fp.seek(0, 2)
            
            while self.running:
                line = fp.readline()
                if not line:
                    await asyncio.sleep(0.2)
                    continue
                
                try:
                    # Parse JSON event
                    vscode_event = json.loads(line.strip())
                    
                    # Transform to Cortex format
                    cortex_event = self.transform_vscode_event(vscode_event)
                    if cortex_event and self.orchestrator:
                        self.orchestrator.add_event(cortex_event)
                        logger.info(f"Processed event: {cortex_event.kind} from {cortex_event.actor}")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in telemetry: {line.strip()}")
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
    
    async def start(self):
        """Start the bridge"""
        logger.info("Starting Cortex Bridge...")
        self.running = True
        
        await self.initialize_cortex()
        await self.tail_jsonl()
    
    def stop(self):
        """Stop the bridge"""
        logger.info("Stopping Cortex Bridge...")
        self.running = False

class CortexAtomEmitter:
    """Emit structured Atom/Bond events for KG consistency"""
    
    @staticmethod
    def emit_guardian_atoms(findings: list, file_path: str) -> list:
        """Convert guardian findings to deterministic atoms"""
        atoms = []
        
        for finding in findings:
            # Generate deterministic UUID v5
            content = f"GuardianFinding|{finding.get('rule', 'unknown')}|{finding.get('message', '')}"
            atom_id = f"atom_{hash(content) % 1000000}"  # Simplified for demo
            
            atom = {
                "id": atom_id,
                "type": "GuardianFinding",
                "title": finding.get('rule', 'Unknown Rule'),
                "content": finding.get('message', ''),
                "metadata": {
                    "file_path": file_path,
                    "severity": finding.get('severity', 'info'),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "super_alita_guardian"
                }
            }
            atoms.append(atom)
        
        return atoms

async def main():
    """Main entry point"""
    bridge = CortexBridge()
    
    try:
        await bridge.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
        bridge.stop()
    except Exception as e:
        logger.error(f"Bridge error: {e}")
        bridge.stop()

if __name__ == "__main__":
    asyncio.run(main())