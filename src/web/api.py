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
import yaml
from pydantic import BaseModel

class ROIPayload(BaseModel):
    type: str # 'include' or 'exclude'
    points: List[List[float]]

class ClassesPayload(BaseModel):
    classes: List[str]

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
        self.active_streams = 0
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
        async def video_feed():
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

        # API endpoint to update ROI
        @self.app.post("/api/config/roi")
        async def update_roi(payload: ROIPayload):
            try:
                config_path = "config/config.yaml"
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                if 'roi' not in config_data:
                    config_data['roi'] = {'enabled': True, 'include_zones': {}, 'exclude_zones': {}}
                
                # Format points to fewer decimal places
                formatted_points = [[round(p[0], 3), round(p[1], 3)] for p in payload.points]
                
                # Update either include or exclude zones. For simplicity, we overwrite a zone named 'custom_ui'
                zone_name = "custom_ui_" + payload.type
                target_dict = f"{payload.type}_zones"
                
                # We clear out any existing zones of this type to replace them with the new one
                config_data['roi'][target_dict] = {
                    zone_name: {
                        'points': formatted_points,
                        'classes': ["person", "cat", "dog"]
                    }
                }
                
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                    
                return {"status": "success", "message": "ROI updated successfully in config.yaml"}
            except Exception as e:
                logging.error(f"Failed to update ROI: {e}")
                return {"status": "error", "detail": str(e)}

        @self.app.get("/api/config/classes")
        async def get_classes():
            try:
                with open("config/config.yaml", 'r') as f:
                    config_data = yaml.safe_load(f)
                return {"classes": config_data.get('detection', {}).get('target_classes', [])}
            except Exception as e:
                return {"status": "error", "detail": str(e)}

        @self.app.post("/api/config/classes")
        async def update_classes(payload: ClassesPayload):
            try:
                config_path = "config/config.yaml"
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                if 'detection' not in config_data:
                    config_data['detection'] = {}
                    
                config_data['detection']['target_classes'] = payload.classes
                
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                    
                return {"status": "success", "message": "Target classes updated successfully"}
            except Exception as e:
                logging.error(f"Failed to update target classes: {e}")
                return {"status": "error", "detail": str(e)}

    # MJPEG stream generator
    async def generate_frames(self):
        """Generate MJPEG stream for video preview with minimal overhead."""
        self.active_streams += 1
        try:
            interval = max(1.0 / max(self.preview_fps, 1), 0.01)
            while True:
                if self.latest_jpeg is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + self.latest_jpeg + b'\r\n')
                await asyncio.sleep(interval)
        finally:
            self.active_streams -= 1
        
    # Update latest frame
    def update_frame(self, frame):
        """Update latest frame for streaming by pre-encoding to JPEG."""
        if getattr(self, 'active_streams', 0) <= 0:
            return  # Skip expensive resizing and JPEG encoding if no one is watching
        
        # Resize once here to reduce CPU load in generator
        try:
            h, w = frame.shape[:2]
            target_w, target_h = 640, 480
            if (w, h) != (target_w, target_h):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
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
