import cv2
import numpy as np
from typing import Optional
from .types import MotionInfo

class MotionGate:
    def __init__(self, threshold: int = 25, min_contour_area: int = 500, 
                 learning_rate: float = 0.01):
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.learning_rate = learning_rate
        # Process at reduced resolution for speed
        self.scale = 0.5  # 50% size
        
        # Background subtractor[47][50]
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=True
        )
        
    def detect_motion(self, frame: np.ndarray) -> MotionInfo:
        """Detect motion in frame using background subtraction."""
        # Downscale for speed
        if self.scale != 1.0:
            small = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        else:
            small = frame
        
        # Apply background subtraction on downscaled frame
        fg_mask_small = self.bg_subtractor.apply(small, learningRate=self.learning_rate)
        
        # Remove shadows (they appear as gray in mask)
        fg_mask_small[fg_mask_small != 255] = 0
        
        # Apply morphological operations to reduce noise[47]
        k = max(int(5 * self.scale), 3)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fg_mask_small = cv2.morphologyEx(fg_mask_small, cv2.MORPH_OPEN, kernel)
        fg_mask_small = cv2.morphologyEx(fg_mask_small, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        area_scale = (self.scale * self.scale)
        scaled_min_area = max(1, int(self.min_contour_area * area_scale))
        valid_contours = [c for c in contours if cv2.contourArea(c) > scaled_min_area]
        
        # Calculate total motion area
        total_area = sum(cv2.contourArea(c) for c in valid_contours)
        
        motion_detected = len(valid_contours) > 0
        
        return MotionInfo(
            motion_detected=motion_detected,
            motion_area=int(total_area),
            contour_count=len(valid_contours)
        )
