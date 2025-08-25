"""
FastAPI telemetry dashboard server
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .collector import TelemetryCollector, TelemetryMetrics, TelemetryEvent
from .streaming import WebSocketStreamer


class TelemetryQuery(BaseModel):
    """Query parameters for telemetry data"""
    cycle_id: Optional[str] = None
    phase: Optional[str] = None
    limit: int = 100
    event_type: Optional[str] = None


class TelemetryDashboard:
    """
    FastAPI-based telemetry dashboard
    """
    
    def __init__(self, telemetry_collector: TelemetryCollector, host: str = "0.0.0.0", port: int = 8001):
        self.collector = telemetry_collector
        self.streamer = WebSocketStreamer(telemetry_collector)
        self.host = host
        self.port = port
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Super Alita Telemetry Dashboard",
            description="Real-time monitoring and analytics for Cortex runtime",
            version="1.0.0"
        )
        
        self._setup_routes()
        self._setup_static_files()
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve the main dashboard HTML"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/metrics", response_model=dict)
        async def get_metrics() -> Dict[str, Any]:
            """Get current telemetry metrics"""
            return self.collector.get_metrics().to_dict()
        
        @self.app.get("/api/events", response_model=List[dict])
        async def get_events(
            limit: int = 100,
            cycle_id: Optional[str] = None,
            phase: Optional[str] = None,
            event_type: Optional[str] = None
        ) -> List[Dict[str, Any]]:
            """Get telemetry events with optional filtering"""
            events = self.collector.get_recent_events(limit)
            
            # Apply filters
            if cycle_id:
                events = [e for e in events if e.cycle_id == cycle_id]
            if phase:
                events = [e for e in events if e.phase == phase]
            if event_type:
                events = [e for e in events if e.event_type == event_type]
                
            return [event.to_dict() for event in events]
        
        @self.app.get("/api/cycles/{cycle_id}/events", response_model=List[dict])
        async def get_cycle_events(cycle_id: str) -> List[Dict[str, Any]]:
            """Get all events for a specific cycle"""
            events = self.collector.get_events_by_cycle(cycle_id)
            return [event.to_dict() for event in events]
        
        @self.app.get("/api/phases/{phase}/stats", response_model=dict)
        async def get_phase_stats(phase: str) -> Dict[str, Any]:
            """Get statistics for a specific phase"""
            stats = self.collector.get_phase_statistics(phase)
            if not stats:
                raise HTTPException(status_code=404, detail=f"No statistics found for phase: {phase}")
            return stats
        
        @self.app.get("/api/health")
        async def health_check() -> Dict[str, Any]:
            """Health check endpoint"""
            return {
                "status": "healthy",
                "total_events": self.collector.metrics.total_events,
                "active_connections": self.streamer.get_connection_count(),
                "uptime": "unknown"  # Would need startup time tracking
            }
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """WebSocket endpoint for real-time streaming"""
            await self.streamer.handle_websocket(websocket, client_id)
        
        @self.app.websocket("/ws")
        async def websocket_endpoint_anonymous(websocket: WebSocket):
            """Anonymous WebSocket endpoint"""
            await self.streamer.handle_websocket(websocket)
        
        @self.app.post("/api/clear_events")
        async def clear_old_events(background_tasks: BackgroundTasks, keep_last: int = 1000):
            """Clear old events to prevent memory growth"""
            background_tasks.add_task(self.collector.clear_old_events, keep_last)
            return {"message": f"Scheduled clearing of old events, keeping {keep_last} most recent"}
    
    def _setup_static_files(self):
        """Setup static file serving"""
        # In a real deployment, you'd serve static files from a directory
        # For now, we'll serve the dashboard HTML inline
        pass
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Super Alita Telemetry Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #e0e0e0;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #2d2d2d;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #3d3d3d;
        }
        .metric-title {
            font-size: 14px;
            color: #b0b0b0;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .events-section {
            background: #2d2d2d;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #3d3d3d;
        }
        .event-item {
            background: #1e1e1e;
            margin: 10px 0;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #4CAF50;
        }
        .event-meta {
            font-size: 12px;
            color: #888;
            margin-bottom: 5px;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-connected { background: #4CAF50; }
        .status-disconnected { background: #f44336; }
        .btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }
        .btn:hover { background: #45a049; }
        .btn:disabled { background: #666; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Super Alita Telemetry Dashboard</h1>
        <p>Real-time monitoring for Cortex runtime</p>
        <div>
            <span id="connection-status" class="status-indicator status-disconnected"></span>
            <span id="connection-text">Disconnected</span>
        </div>
        <div style="margin-top: 10px;">
            <button id="connect-btn" class="btn" onclick="connectWebSocket()">Connect</button>
            <button id="disconnect-btn" class="btn" onclick="disconnectWebSocket()" disabled>Disconnect</button>
        </div>
    </div>

    <div class="metrics-grid" id="metrics-grid">
        <!-- Metrics will be populated by JavaScript -->
    </div>

    <div class="events-section">
        <h3>Recent Events</h3>
        <div id="events-container">
            <!-- Events will be populated by JavaScript -->
        </div>
    </div>

    <script>
        let socket = null;
        let isConnected = false;

        function updateConnectionStatus(connected) {
            isConnected = connected;
            const indicator = document.getElementById('connection-status');
            const text = document.getElementById('connection-text');
            const connectBtn = document.getElementById('connect-btn');
            const disconnectBtn = document.getElementById('disconnect-btn');

            if (connected) {
                indicator.className = 'status-indicator status-connected';
                text.textContent = 'Connected';
                connectBtn.disabled = true;
                disconnectBtn.disabled = false;
            } else {
                indicator.className = 'status-indicator status-disconnected';
                text.textContent = 'Disconnected';
                connectBtn.disabled = false;
                disconnectBtn.disabled = true;
            }
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            socket = new WebSocket(wsUrl);

            socket.onopen = function() {
                updateConnectionStatus(true);
                console.log('WebSocket connected');
                
                // Request initial metrics
                socket.send(JSON.stringify({type: 'get_metrics'}));
                
                // Start streaming
                socket.send(JSON.stringify({type: 'start_streaming'}));
            };

            socket.onmessage = function(event) {
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            };

            socket.onclose = function() {
                updateConnectionStatus(false);
                console.log('WebSocket disconnected');
            };

            socket.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateConnectionStatus(false);
            };
        }

        function disconnectWebSocket() {
            if (socket) {
                socket.close();
                socket = null;
            }
        }

        function handleWebSocketMessage(message) {
            switch (message.type) {
                case 'metrics_update':
                    updateMetrics(message.data);
                    break;
                case 'telemetry_event':
                    addEvent(message.data);
                    break;
                default:
                    console.log('Unknown message type:', message.type);
            }
        }

        function updateMetrics(metrics) {
            const grid = document.getElementById('metrics-grid');
            grid.innerHTML = `
                <div class="metric-card">
                    <div class="metric-title">Total Cycles</div>
                    <div class="metric-value">${metrics.total_cycles}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Total Events</div>
                    <div class="metric-value">${metrics.total_events}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Avg Cycle Duration</div>
                    <div class="metric-value">${metrics.avg_cycle_duration_ms.toFixed(1)}ms</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Success Rate</div>
                    <div class="metric-value">${(metrics.success_rate * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Active Cycles</div>
                    <div class="metric-value">${metrics.active_cycles}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Error Count</div>
                    <div class="metric-value">${metrics.error_count}</div>
                </div>
            `;
        }

        function addEvent(event) {
            const container = document.getElementById('events-container');
            const eventDiv = document.createElement('div');
            eventDiv.className = 'event-item';
            
            const timestamp = new Date(event.timestamp * 1000).toLocaleTimeString();
            
            eventDiv.innerHTML = `
                <div class="event-meta">
                    ${timestamp} | ${event.source} | ${event.event_type}
                    ${event.cycle_id ? '| Cycle: ' + event.cycle_id : ''}
                    ${event.phase ? '| Phase: ' + event.phase : ''}
                </div>
                <div>
                    ${event.duration_ms ? 'Duration: ' + event.duration_ms.toFixed(1) + 'ms' : ''}
                    ${event.metadata ? '| ' + JSON.stringify(event.metadata) : ''}
                </div>
            `;
            
            container.insertBefore(eventDiv, container.firstChild);
            
            // Keep only last 50 events
            while (container.children.length > 50) {
                container.removeChild(container.lastChild);
            }
        }

        // Auto-connect on page load
        document.addEventListener('DOMContentLoaded', function() {
            connectWebSocket();
        });
    </script>
</body>
</html>
        """
    
    async def start_server(self):
        """Start the telemetry dashboard server"""
        import uvicorn
        
        # Start background tasks
        asyncio.create_task(self.streamer.start_metrics_broadcast())
        
        # Start the server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI app instance"""
        return self.app