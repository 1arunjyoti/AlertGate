import cv2
import time
import logging
from typing import Optional, Tuple
import numpy as np
import os
import threading

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
            # Configure FFmpeg options for lower latency
            opts = [f"rtsp_transport;{self.transport}",
                    "fflags;nobuffer",
                    "reorder_queue_size;0",
                    "max_delay;0",
                    "stimeout;2000000"]  # 2s IO timeout
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "|".join(opts)
            
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


class LatestFrameBuffer:
    """Single-slot latest-frame buffer with versioning for lockless-style consumption.
    Consumers can poll and skip if the version hasn't changed.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq: int = 0
        self._timestamp: float = 0.0

    def set(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame
            self._seq += 1
            self._timestamp = time.time()

    def get(self) -> Tuple[Optional[np.ndarray], int, float]:
        with self._lock:
            return self._frame, self._seq, self._timestamp


class CaptureWorker(threading.Thread):
    """Background capture thread that continuously updates LatestFrameBuffer."""
    def __init__(self, capture: RTSPCapture, buffer: LatestFrameBuffer, stop_event: threading.Event, read_sleep: float = 0.0):
        super().__init__(daemon=True, name="capture-worker")
        self.capture = capture
        self.buffer = buffer
        self.stop_event = stop_event
        self.read_sleep = read_sleep
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Ensure connection
        if not self.capture.connect():
            self.logger.error("❌ Failed to connect to camera")
            return
        while not self.stop_event.is_set():
            ret, frame = self.capture.read()
            if not ret or frame is None:
                self.logger.warning("⚠️ Frame read failed in worker, attempting reconnect...")
                if not self.capture.reconnect():
                    time.sleep(1.0)
                continue
            self.buffer.set(frame)
            if self.read_sleep > 0:
                time.sleep(self.read_sleep)

    def close(self):
        self.stop_event.set()
