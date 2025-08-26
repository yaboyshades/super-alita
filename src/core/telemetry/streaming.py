"""
WebSocket streaming for real-time telemetry data
"""

import asyncio
import json
import logging
from typing import Set, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect

from .collector import TelemetryEvent, TelemetryCollector

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for telemetry streaming"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, client_info: Optional[Dict[str, Any]] = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_metadata[websocket] = client_info or {}
        logger.info(f"New telemetry client connected. Total: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_metadata.pop(websocket, None)
            logger.info(f"Telemetry client disconnected. Total: {len(self.active_connections)}")
    
    async def send_to_all(self, data: Dict[str, Any]):
        """Send data to all connected clients"""
        if not self.active_connections:
            return
            
        message = json.dumps(data)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to connection: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def send_to_client(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send data to a specific client"""
        try:
            await websocket.send_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"Failed to send to specific client: {e}")
            self.disconnect(websocket)


class WebSocketStreamer:
    """
    Streams telemetry data to connected WebSocket clients
    """
    
    def __init__(self, telemetry_collector: TelemetryCollector):
        self.collector = telemetry_collector
        self.connection_manager = ConnectionManager()
        self.is_streaming = False
        
        # Subscribe to telemetry events
        self.collector.subscribe(self._on_telemetry_event)
    
    def _on_telemetry_event(self, event: TelemetryEvent):
        """Handle new telemetry events"""
        if self.is_streaming and self.connection_manager.active_connections:
            # Schedule the send operation
            asyncio.create_task(self._broadcast_event(event))
    
    async def _broadcast_event(self, event: TelemetryEvent):
        """Broadcast telemetry event to all connected clients"""
        try:
            message = {
                "type": "telemetry_event",
                "data": event.to_dict(),
                "timestamp": event.timestamp
            }
            await self.connection_manager.send_to_all(message)
        except Exception as e:
            logger.error(f"Error broadcasting telemetry event: {e}")
    
    async def handle_websocket(self, websocket: WebSocket, client_id: Optional[str] = None):
        """Handle a WebSocket connection for telemetry streaming"""
        client_info = {"client_id": client_id} if client_id else {}
        await self.connection_manager.connect(websocket, client_info)
        
        try:
            # Send initial state
            await self._send_initial_state(websocket)
            
            # Handle incoming messages
            while True:
                data = await websocket.receive_text()
                await self._handle_client_message(websocket, json.loads(data))
                
        except WebSocketDisconnect:
            logger.info("Client disconnected normally")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.connection_manager.disconnect(websocket)
    
    async def _send_initial_state(self, websocket: WebSocket):
        """Send initial telemetry state to new client"""
        try:
            # Send current metrics
            metrics_message = {
                "type": "metrics_update",
                "data": self.collector.get_metrics().to_dict(),
                "timestamp": asyncio.get_event_loop().time()
            }
            await self.connection_manager.send_to_client(websocket, metrics_message)
            
            # Send recent events
            recent_events = self.collector.get_recent_events(limit=50)
            for event in recent_events:
                event_message = {
                    "type": "telemetry_event",
                    "data": event.to_dict(),
                    "timestamp": event.timestamp
                }
                await self.connection_manager.send_to_client(websocket, event_message)
                
        except Exception as e:
            logger.error(f"Error sending initial state: {e}")
    
    async def _handle_client_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle incoming message from client"""
        try:
            msg_type = message.get("type")
            
            if msg_type == "get_metrics":
                # Send current metrics
                response = {
                    "type": "metrics_update",
                    "data": self.collector.get_metrics().to_dict(),
                    "timestamp": asyncio.get_event_loop().time()
                }
                await self.connection_manager.send_to_client(websocket, response)
                
            elif msg_type == "get_cycle_events":
                # Send events for specific cycle
                cycle_id = message.get("cycle_id")
                if cycle_id:
                    events = self.collector.get_events_by_cycle(cycle_id)
                    response = {
                        "type": "cycle_events",
                        "cycle_id": cycle_id,
                        "events": [event.to_dict() for event in events],
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await self.connection_manager.send_to_client(websocket, response)
                    
            elif msg_type == "get_phase_stats":
                # Send phase statistics
                phase = message.get("phase")
                if phase:
                    stats = self.collector.get_phase_statistics(phase)
                    response = {
                        "type": "phase_statistics",
                        "phase": phase,
                        "statistics": stats,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await self.connection_manager.send_to_client(websocket, response)
                    
            elif msg_type == "start_streaming":
                self.is_streaming = True
                response = {
                    "type": "streaming_status",
                    "streaming": True,
                    "timestamp": asyncio.get_event_loop().time()
                }
                await self.connection_manager.send_to_client(websocket, response)
                
            elif msg_type == "stop_streaming":
                self.is_streaming = False
                response = {
                    "type": "streaming_status", 
                    "streaming": False,
                    "timestamp": asyncio.get_event_loop().time()
                }
                await self.connection_manager.send_to_client(websocket, response)
                
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def start_metrics_broadcast(self, interval_seconds: float = 5.0):
        """Start periodic metrics broadcasting"""
        while True:
            try:
                if self.connection_manager.active_connections:
                    metrics_message = {
                        "type": "metrics_update",
                        "data": self.collector.get_metrics().to_dict(),
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await self.connection_manager.send_to_all(metrics_message)
                
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in metrics broadcast: {e}")
                await asyncio.sleep(interval_seconds)
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.connection_manager.active_connections)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about active connections"""
        return {
            "active_connections": self.get_connection_count(),
            "clients": [
                metadata for metadata in self.connection_manager.connection_metadata.values()
            ]
        }