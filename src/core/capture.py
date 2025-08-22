import cv2
import time
import logging
from typing import Optional, Tuple
import numpy as np

class RTSPCapture:
    def __init__(self, url: str, buffer_size: int = 1, transport: str = "tcp"):
        self.url = url
        self.buffer_size = buffer_size
        self.transport = transport
        self.cap: Optional[cv2.VideoCapture] = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """Connect to RTSP stream."""
        try:
            if self.transport == "tcp":
                import os
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
                
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            if not self.cap.isOpened():
                raise Exception("Failed to open capture")
                
            self.logger.info(f"✅ Connected to RTSP stream: {self.url}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ RTSP connection failed: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from stream."""
        if not self.cap or not self.cap.isOpened():
            return False, None
            
        ret, frame = self.cap.read()
        return ret, frame
    
    def reconnect(self) -> bool:
        """Reconnect to stream."""
        self.logger.warning("🔄 Attempting to reconnect...")
        self.release()
        time.sleep(2)
        return self.connect()
    
    def release(self):
        """Release capture resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
