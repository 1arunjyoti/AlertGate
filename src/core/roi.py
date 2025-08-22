import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from .types import Detection

class ROIManager:
    def __init__(self, roi_config: Dict[str, Any]):
        self.config = roi_config
        self.include_masks: Dict[str, np.ndarray] = {}
        self.exclude_masks: Dict[str, np.ndarray] = {}
        self.include_contours: Dict[str, list] = {}
        self.exclude_contours: Dict[str, list] = {}
        
    def create_masks(self, frame_shape: Tuple[int, int]):
        """Create polygon masks from normalized coordinates."""
        height, width = frame_shape[:2]
        
        # Create include zone masks[46][49]
        for zone_name, zone_config in self.config.get('include_zones', {}).items():
            points = np.array(zone_config['points'], dtype=np.float32)
            # Convert normalized coordinates to pixel coordinates
            points[:, 0] *= width
            points[:, 1] *= height
            points = points.astype(np.int32)
            
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            self.include_masks[zone_name] = mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.include_contours[zone_name] = contours
            
        # Create exclude zone masks
        for zone_name, zone_config in self.config.get('exclude_zones', {}).items():
            points = np.array(zone_config['points'], dtype=np.float32)
            points[:, 0] *= width
            points[:, 1] *= height
            points = points.astype(np.int32)
            
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            self.exclude_masks[zone_name] = mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.exclude_contours[zone_name] = contours
    
    def filter_detections(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections based on ROI zones."""
        if not self.config.get('enabled', False):
            return detections
            
        filtered_detections = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Check include zones
            in_include_zone = False
            for zone_name, mask in self.include_masks.items():
                zone_config = self.config['include_zones'][zone_name]
                allowed_classes = zone_config.get('classes', [])
                
                if (detection.class_name in allowed_classes and 
                    mask[center_y, center_x] > 0):
                    in_include_zone = True
                    break
            
            # Check exclude zones
            in_exclude_zone = False
            for zone_name, mask in self.exclude_masks.items():
                zone_config = self.config['exclude_zones'][zone_name]
                excluded_classes = zone_config.get('classes', [])
                
                if (detection.class_name in excluded_classes and 
                    mask[center_y, center_x] > 0):
                    in_exclude_zone = True
                    break
            
            # Include detection if it's in include zone and not in exclude zone
            if in_include_zone and not in_exclude_zone:
                filtered_detections.append(detection)
                
        return filtered_detections
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROI zones on frame for visualization."""
        overlay = frame.copy()
        
        # Draw include zones in green
        for zone_name, contours in self.include_contours.items():
            if contours:
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                point = contours[0][0][0]
                cv2.putText(overlay, f"INCLUDE: {zone_name}", 
                           (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw exclude zones in red
        for zone_name, contours in self.exclude_contours.items():
            if contours:
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
                point = contours[0][0][0]
                cv2.putText(overlay, f"EXCLUDE: {zone_name}", 
                           (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
