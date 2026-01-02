from collections import deque, defaultdict
from typing import List, Dict, Any
from .types import Detection

class TemporalFilter:
    # Initialize with configuration for temporal filtering
    def __init__(self, temporal_config: Dict[str, Any]):
        self.config = temporal_config
        # Store detection history for each class[1][4]
        self.detection_history: Dict[str, deque] = defaultdict(deque)
        
    # Methods for adding detections, checking votes, and history
    def add_detections(self, detections: List[Detection], frame_number: int) -> Dict[str, bool]:
        """Add detections and return which classes should trigger alerts."""
        triggers = {}
        
        # Update history for each configured class
        for class_name in self.config.keys():
            class_config = self.config[class_name]
            window_size = class_config['window_size']
            votes_required = class_config['votes_required']
            
            # Check if class was detected in current frame
            detected = any(d.class_name == class_name for d in detections)
            
            # Add to history
            history = self.detection_history[class_name]
            history.append(detected)
            
            # Keep only window_size items
            if len(history) > window_size:
                history.popleft()
            
            # Check if we have enough positive votes
            if len(history) >= votes_required:
                positive_votes = sum(history)
                triggers[class_name] = positive_votes >= votes_required
            else:
                triggers[class_name] = False
                
        return triggers
    
    # Reset history for a class after alert
    def reset_class_history(self, class_name: str):
        """Reset detection history for a specific class (after alert sent)."""
        if class_name in self.detection_history:
            self.detection_history[class_name].clear()
    
    # Get current voting status for web dashboard
    def get_voting_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current voting status for web dashboard."""
        status = {}
        
        # Iterate through each configured class
        for class_name in self.config.keys():
            class_config = self.config[class_name]
            history = list(self.detection_history[class_name])
            
            status[class_name] = {
                'votes_required': class_config['votes_required'],
                'window_size': class_config['window_size'],
                'current_votes': sum(history),
                'history_length': len(history),
                'recent_detections': history[-5:] if history else []
            }
            
        return status
