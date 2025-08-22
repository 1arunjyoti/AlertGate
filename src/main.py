import asyncio
import yaml
import logging
import cv2
import time
from datetime import datetime
from typing import Dict, Any, List
import threading
import uvicorn
from dotenv import load_dotenv
import os

# Import our modules
from core.capture import RTSPCapture
from core.motion_gate import MotionGate
from core.roi import ROIManager
from core.temporal_filter import TemporalFilter
from core.types import Detection, Event, FrameData
from core.inference import YOLODetector
from services.telegram_service import TelegramNotifier
from web.api import WebDashboard

load_dotenv()

class AlertGate:
    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            print(f"✅ Configuration loaded from {config_path}")
        except FileNotFoundError:
            print(f"❌ Config file {config_path} not found!")
            raise
            
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['logging']['file'], encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.config['camera']['url'] = os.getenv('CAMERA_URL', self.config['camera']['url'])
        self.capture = RTSPCapture(
            self.config['camera']['url'],
            self.config['camera']['buffer_size'],
            self.config['camera']['rtsp_transport']
        )
        
        self.motion_gate = MotionGate(
            threshold=self.config['motion']['threshold'],
            min_contour_area=self.config['motion']['min_contour_area'],
            learning_rate=self.config['motion']['background_learning_rate']
        )
        
        self.roi_manager = ROIManager(self.config['roi'])
        self.temporal_filter = TemporalFilter(self.config['temporal'])
        self.detector = YOLODetector(self.config['detection'])
        self.notifier = TelegramNotifier()
        
        # Web dashboard
        if self.config['web']['enabled']:
            self.dashboard = WebDashboard(self.config['web']['preview_fps'])
        
        # State tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_alert_times = {}
        self.stats = {
            'total_detections': 0,
            'alerts_sent': 0,
            'fps': 0,
            'frame_number': 0,
            'uptime': 0
        }
        
    def run(self):
        """Main detection loop."""
        self.logger.info("🚀 Starting AlertGate...")
        
        # Start web dashboard in separate thread
        if self.config['web']['enabled']:
            dashboard_thread = threading.Thread(
                target=self.start_dashboard,
                daemon=True
            )
            dashboard_thread.start()
            self.logger.info(f"🌐 Web dashboard: http://localhost:{self.config['web']['port']}")
        
        # Connect to camera
        if not self.capture.connect():
            self.logger.error("❌ Failed to connect to camera")
            return
        
        # Main processing loop
        last_fps_update = time.time()
        fps_frame_count = 0
        
        try:
            while True:
                ret, frame = self.capture.read()
                
                if not ret:
                    self.logger.warning("⚠️ Frame read failed, attempting reconnect...")
                    if not self.capture.reconnect():
                        break
                    continue
                
                self.frame_count += 1
                fps_frame_count += 1
                
                # Initialize ROI masks on first frame
                if self.frame_count == 1:
                    self.roi_manager.create_masks(frame.shape)
                
                # Process frame
                frame_data = self.process_frame(frame)
                
                # Update web dashboard
                if self.config['web']['enabled']:
                    # Add ROI visualization for dashboard
                    dashboard_frame = self.roi_manager.draw_zones(frame)
                    self.dashboard.update_frame(dashboard_frame)
                
                # Update FPS calculation
                current_time = time.time()
                if current_time - last_fps_update >= 1.0:
                    self.stats['fps'] = fps_frame_count / (current_time - last_fps_update)
                    fps_frame_count = 0
                    last_fps_update = current_time
                
                # Update stats
                self.stats.update({
                    'frame_number': self.frame_count,
                    'uptime': int(current_time - self.start_time),
                    'temporal_voting': self.temporal_filter.get_voting_status(),
                    'motion': {
                        'detected': frame_data.motion_info.motion_detected,
                        'area': frame_data.motion_info.motion_area,
                        'contours': frame_data.motion_info.contour_count
                    }
                })
                
                if self.config['web']['enabled']:
                    self.dashboard.update_stats(self.stats)
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Stopping AlertGate...")
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def process_frame(self, frame) -> FrameData:
        """Process a single frame through the detection pipeline."""
        
        # Step 1: Motion detection
        motion_info = self.motion_gate.detect_motion(frame)
        
        detections = []
        
        # Step 2: Run inference only if motion detected (or if disabled)
        if (not self.config['motion']['skip_frames_no_motion'] or 
            motion_info.motion_detected or 
            self.frame_count % 30 == 0):  # Periodic check even without motion
            
            detections = self.detector.detect(frame)
            
            # Step 3: Apply ROI filtering
            detections = self.roi_manager.filter_detections(detections)
            
            self.stats['total_detections'] += len(detections)
        
        # Step 4: Temporal filtering
        triggers = self.temporal_filter.add_detections(detections, self.frame_count)
        
        # Step 5: Send alerts for triggered classes
        for class_name, should_alert in triggers.items():
            if should_alert and self.should_send_alert(class_name):
                self.send_alert(class_name, detections, frame)
        
        return FrameData(
            frame=frame,
            frame_number=self.frame_count,
            motion_info=motion_info,
            detections=detections
        )
    
    def should_send_alert(self, class_name: str) -> bool:
        """Check if enough time has passed since last alert for this class."""
        current_time = time.time()
        cooldown = self.config['alerts']['cooldown_seconds'].get(class_name, 60)
        
        last_alert = self.last_alert_times.get(class_name, 0)
        return (current_time - last_alert) >= cooldown
    
    def send_alert(self, class_name: str, detections: List[Detection], frame):
        """Send alert notification."""
        try:
            # Find best detection for this class
            class_detections = [d for d in detections if d.class_name == class_name]
            if not class_detections:
                return
            
            best_detection = max(class_detections, key=lambda d: d.confidence)
            
            # Save snapshot (implement in a separate service)
            snapshot_path = self.save_snapshot(frame, detections)
            
            # Send Telegram notification
            self.notifier.send_detection_alert(best_detection, snapshot_path)
            
            # Update state
            self.last_alert_times[class_name] = time.time()
            self.temporal_filter.reset_class_history(class_name)
            self.stats['alerts_sent'] += 1
            
            # Add to web dashboard
            if self.config['web']['enabled']:
                event_data = {
                    'class_name': best_detection.class_name,
                    'confidence': best_detection.confidence,
                    'timestamp': best_detection.timestamp.isoformat(),
                    'frame_number': self.frame_count,
                    'zone': 'backyard'  # Could be enhanced to detect actual zone
                }
                self.dashboard.add_event(event_data)
            
            self.logger.info(f"🚨 Alert sent: {class_name} ({best_detection.confidence:.2f})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send alert: {e}")
    
    def save_snapshot(self, frame, detections: List[Detection]) -> str:
        """Save annotated snapshot."""
        # Check if sending images is enabled in alerts
        if not self.config.get('alerts', {}).get('send_image', False):
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cat_detection_{timestamp}_frame_{self.frame_count}.jpg"
        
        snapshot_config = self.config.get('recording')
        if not snapshot_config:
            self.logger.warning("❌ 'recording' section not found in config. Cannot save snapshot.")
            return None
            
        directory = snapshot_config.get('snapshots_dir')
        if not directory:
            self.logger.error("❌ Snapshot directory ('snapshots_dir') not configured under 'recording'.")
            return None

        filepath = os.path.join(directory, filename)

        # Draw detections on the frame
        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            label = f"{det.class_name} {det.confidence:.2f}"
            
            # Bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Save the frame
        cv2.imwrite(filepath, annotated_frame)
        self.logger.info(f"📸 Snapshot saved to {filepath}")
        return filepath
    
    def start_dashboard(self):
        """Start web dashboard server."""
        uvicorn.run(
            self.dashboard.app,
            host=self.config['web']['host'],
            port=self.config['web']['port'],
            log_level="warning"
        )
    
    def cleanup(self):
        """Clean up resources."""
        self.capture.release()
        self.logger.info("👋 AlertGate stopped")

if __name__ == "__main__":
    sentinel = AlertGate()
    sentinel.run()
