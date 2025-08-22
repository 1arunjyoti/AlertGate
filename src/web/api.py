from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import cv2
import json
import asyncio
import time
from typing import List, Dict, Any, Optional
import logging

class WebDashboard:
    def __init__(self, preview_fps: int = 2):
        self.app = FastAPI(title="EdgeSentinel Dashboard")
        self.preview_fps = preview_fps
        self.connected_clients: List[WebSocket] = []
        self.latest_frame = None
        self.latest_jpeg: Optional[bytes] = None
        self.jpeg_quality: int = 75
        self.latest_stats = {}
        self.event_history = []
        self.loop = None  # To store the event loop of the dashboard's thread
        
        # Setup static files and templates[59][72]
        self.app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
        self.templates = Jinja2Templates(directory="src/web/templates")
        
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Get the running event loop on startup."""
            self.loop = asyncio.get_running_loop()
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            return self.templates.TemplateResponse("dashboard.html", {"request": request})
        
        @self.app.get("/video_feed")
        def video_feed():
            return StreamingResponse(self.generate_frames(), 
                                   media_type="multipart/x-mixed-replace; boundary=frame")
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.connected_clients.append(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)
        
        @self.app.get("/api/stats")
        async def get_stats():
            return self.latest_stats
        
        @self.app.get("/api/events")
        async def get_events():
            return {"events": self.event_history[-50:]}  # Last 50 events
    
    def generate_frames(self):
        """Generate MJPEG stream for video preview with minimal overhead."""
        interval = max(1.0 / max(self.preview_fps, 1), 0.01)
        while True:
            if self.latest_jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + self.latest_jpeg + b'\r\n')
            time.sleep(interval)
    
    def update_frame(self, frame):
        """Update latest frame for streaming by pre-encoding to JPEG."""
        # Resize once here to reduce CPU load in generator
        try:
            h, w = frame.shape[:2]
            target_w, target_h = 640, 480
            if (w, h) != (target_w, target_h):
                frame = cv2.resize(frame, (target_w, target_h))
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ret:
                self.latest_jpeg = buffer.tobytes()
        except Exception as e:
            logging.error(f"Failed to encode frame: {e}")
    
    def update_stats(self, stats: Dict[str, Any]):
        """Update dashboard statistics."""
        self.latest_stats = stats
        # Broadcast to connected WebSocket clients
        if self.connected_clients and self.loop:
            asyncio.run_coroutine_threadsafe(self.broadcast_stats(stats), self.loop)
    
    async def broadcast_stats(self, stats: Dict[str, Any]):
        """Broadcast stats to all connected WebSocket clients."""
        message = json.dumps({"type": "stats", "data": stats})
        disconnected = []
        
        for client in self.connected_clients:
            try:
                await client.send_text(message)
            except:
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            if client in self.connected_clients:
                self.connected_clients.remove(client)
    
    def add_event(self, event_data: Dict[str, Any]):
        """Add event to history."""
        self.event_history.append(event_data)
        if len(self.event_history) > 100:  # Keep last 100 events
            self.event_history.pop(0)
        
        # Broadcast event to clients
        if self.connected_clients and self.loop:
            asyncio.run_coroutine_threadsafe(self.broadcast_event(event_data), self.loop)
    
    async def broadcast_event(self, event_data: Dict[str, Any]):
        """Broadcast new event to all connected clients."""
        message = json.dumps({"type": "event", "data": event_data})
        disconnected = []
        
        for client in self.connected_clients:
            try:
                await client.send_text(message)
            except:
                disconnected.append(client)
        
        for client in disconnected:
            if client in self.connected_clients:
                self.connected_clients.remove(client)
