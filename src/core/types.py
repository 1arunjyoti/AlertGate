from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Event:
    detections: List[Detection]
    frame_number: int
    zone: str
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MotionInfo:
    motion_detected: bool
    motion_area: int
    contour_count: int

@dataclass
class FrameData:
    frame: np.ndarray
    frame_number: int
    motion_info: MotionInfo
    detections: List[Detection] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
