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
        
        # Background subtractor[47][50]
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=True
        )
        
    def detect_motion(self, frame: np.ndarray) -> MotionInfo:
        """Detect motion in frame using background subtraction."""
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Remove shadows (they appear as gray in mask)
        fg_mask[fg_mask != 255] = 0
        
        # Apply morphological operations to reduce noise[47]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        
        # Calculate total motion area
        total_area = sum(cv2.contourArea(c) for c in valid_contours)
        
        motion_detected = len(valid_contours) > 0
        
        return MotionInfo(
            motion_detected=motion_detected,
            motion_area=int(total_area),
            contour_count=len(valid_contours)
        )
