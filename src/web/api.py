from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import cv2
import numpy as np
import json
import asyncio
import time
from typing import List, Dict, Any, Optional
import logging

#TurboJPEG import for faster JPEG encoding
try:
    from turbojpeg import TurboJPEG, TJPF_BGR
except Exception:
    TurboJPEG = None
    TJPF_BGR = None

class WebDashboard:
    def __init__(self, preview_fps: int = 2, event_store=None, events_limit: int = 50):
        self.app = FastAPI(title="EdgeSentinel Dashboard")
        self.preview_fps = preview_fps
        self.connected_clients: List[WebSocket] = []
        self.latest_frame = None
        self.latest_jpeg: Optional[bytes] = None
        self.jpeg_quality: int = 75
        self.latest_stats = {}
        # Keep small in-memory tail for immediate UI feedback; DB remains source of truth
        self.event_history = []
        self.event_store = event_store
        self.events_limit = events_limit
        self.loop = None  # To store the event loop of the dashboard's thread
        # TurboJPEG encoder instance
        self.jpeg = None
        
        # Initialize TurboJPEG if available
        if TurboJPEG is not None:
            try:
                self.jpeg = TurboJPEG()
                logging.info("TurboJPEG enabled for JPEG encoding")
            except Exception as e:
                logging.warning(f"TurboJPEG unavailable: {e}; falling back to cv2.imencode")
        
        # Setup static files and templates[59][72]
        self.app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
        self.templates = Jinja2Templates(directory="src/web/templates")
        
        self.setup_routes()
        
    # Setup FastAPI routes
    def setup_routes(self):
        """Setup FastAPI routes."""
        
        # Get the event loop on startup
        @self.app.on_event("startup")
        async def startup_event():
            """Get the running event loop on startup."""
            self.loop = asyncio.get_running_loop()
        
        # Dashboard page
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            return self.templates.TemplateResponse("dashboard.html", {"request": request})
        
        # MJPEG video feed
        @self.app.get("/video_feed")
        def video_feed():
            return StreamingResponse(self.generate_frames(), 
                                   media_type="multipart/x-mixed-replace; boundary=frame")
        
        # WebSocket endpoint for real-time stats and events
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
        
        # API endpoint for latest stats
        @self.app.get("/api/stats")
        async def get_stats():
            return self.latest_stats
        
        # API endpoint for recent events
        @self.app.get("/api/events")
        async def get_events():
            limit = self.events_limit or 50
            if self.event_store is not None:
                try:
                    events = self.event_store.get_recent_events(limit)
                    return {"events": events}
                except Exception as e:
                    logging.error(f"Failed to fetch events from store: {e}")
            # Fallback to in-memory
            return {"events": self.event_history[-limit:]}
    
    # MJPEG stream generator
    def generate_frames(self):
        """Generate MJPEG stream for video preview with minimal overhead."""
        interval = max(1.0 / max(self.preview_fps, 1), 0.01)
        while True:
            if self.latest_jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + self.latest_jpeg + b'\r\n')
            time.sleep(interval)
    
    # Update latest frame
    def update_frame(self, frame):
        """Update latest frame for streaming by pre-encoding to JPEG."""
        # Resize once here to reduce CPU load in generator
        try:
            h, w = frame.shape[:2]
            target_w, target_h = 640, 480
            if (w, h) != (target_w, target_h):
                frame = cv2.resize(frame, (target_w, target_h))
            if self.jpeg is not None and TJPF_BGR is not None:
                # Ensure contiguous memory and BGR pixel format for TurboJPEG
                frame = np.ascontiguousarray(frame)
                self.latest_jpeg = self.jpeg.encode(frame, quality=self.jpeg_quality, pixel_format=TJPF_BGR)
            else:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if ret:
                    self.latest_jpeg = buffer.tobytes()
        except Exception as e:
            logging.error(f"Failed to encode frame: {e}")
    
    # Update dashboard statistics
    def update_stats(self, stats: Dict[str, Any]):
        """Update dashboard statistics."""
        self.latest_stats = stats
        # Broadcast to connected WebSocket clients
        if self.connected_clients and self.loop:
            asyncio.run_coroutine_threadsafe(self.broadcast_stats(stats), self.loop)
    
    # Broadcast stats to WebSocket clients
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
    
    # Add event to history and broadcast
    def add_event(self, event_data: Dict[str, Any]):
        """Add event to history."""
        self.event_history.append(event_data)
        if len(self.event_history) > 100:  # Keep last 100 events
            self.event_history.pop(0)
        
        # Broadcast event to clients
        if self.connected_clients and self.loop:
            asyncio.run_coroutine_threadsafe(self.broadcast_event(event_data), self.loop)
    
    # Broadcast event to WebSocket clients
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
