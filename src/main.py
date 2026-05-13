import yaml
import logging
import numpy as np
import time
import threading
import uvicorn
from dotenv import load_dotenv
import os


from alerts.alert_manager import AlertManager
from alerts.snapshot_writer import SnapshotWriter
from core.capture import RTSPCapture, LatestFrameBuffer, CaptureWorker
from core.motion_gate import MotionGate
from core.roi import ROIManager
from core.temporal_filter import TemporalFilter
from core.types import Detection, FrameData
from core.inference import YOLODetector
from pipeline.roi_crop import get_include_crop_rect
from services.telegram_service import TelegramNotifier
from services.event_store import EventStore
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
        # Detection cadence
        self.detect_every_n = int(self.config.get('detection', {}).get('process_every_n_frames', 3))
        if self.detect_every_n < 1:
            self.detect_every_n = 1
        
        # Event store (SQLite)
        db_cfg = self.config.get('database', {})
        db_path = os.getenv('DATABASE_PATH', db_cfg.get('path', './media/alertgate.db'))
        self.event_store = EventStore(db_path)
        try:
            retention_days = int(db_cfg.get('retention_days', 30))
            self.event_store.prune_older_than(retention_days)
        except Exception:
            pass
        
        # Web dashboard
        self.dashboard = None
        if self.config['web']['enabled']:
            events_limit = int(self.config['web'].get('max_events_history', 100))
            self.dashboard = WebDashboard(self.config['web']['preview_fps'], event_store=self.event_store, events_limit=events_limit)
        
        # State tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.stats = {
            'total_detections': 0,
            'alerts_sent': 0,
            'fps': 0,
            'frame_number': 0,
            'uptime': 0
        }

        self.snapshot_writer = SnapshotWriter(self.config, self.logger)
        self.alert_manager = AlertManager(
            alerts_config=self.config['alerts'],
            web_config=self.config['web'],
            notifier=self.notifier,
            snapshot_writer=self.snapshot_writer,
            event_store=self.event_store,
            dashboard=self.dashboard,
            temporal_filter=self.temporal_filter,
            stats=self.stats,
            logger=self.logger,
            frame_number_getter=lambda: self.frame_count,
        )
        
        # Latest-frame buffer & capture worker control
        self.frame_buffer = LatestFrameBuffer()
        self.stop_event = threading.Event()
        self.capture_worker = None
        
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
        
        # Start capture worker thread (it handles connecting/reconnecting)
        self.capture_worker = CaptureWorker(self.capture, self.frame_buffer, self.stop_event)
        self.capture_worker.start()
        
        # Main processing loop
        last_fps_update = time.time()
        fps_frame_count = 0
        last_seq = -1
        
        try:
            while True:
                frame, seq, ts = self.frame_buffer.get()
                if frame is None or seq == last_seq:
                    # No new frame yet; yield CPU briefly
                    time.sleep(0.005)
                    continue
                last_seq = seq
                
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
                
                # Small delay to prevent CPU overload (5ms)
                time.sleep(0.005)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Stopping AlertGate...")
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def process_frame(self, frame) -> FrameData:
        """Process a single frame through the detection pipeline."""
        
        # Motion detection
        motion_info = self.motion_gate.detect_motion(frame)
        
        detections = []
        
        # Run inference only if motion detected (or if disabled)
        if (not self.config['motion']['skip_frames_no_motion'] or 
            motion_info.motion_detected or 
            self.frame_count % self.detect_every_n == 0):  # Configurable periodic check
            
            # Optionally crop to include-zone bounding box for faster inference
            crop_rect = get_include_crop_rect(self.roi_manager, frame.shape)
            if crop_rect is not None:
                x1, y1, x2, y2 = crop_rect
                detect_frame = frame[y1:y2, x1:x2]
            else:
                x1 = y1 = 0
                detect_frame = frame

            # Ensure memory layout contiguous to avoid extra copies in OpenCV/Torch
            detect_frame = np.ascontiguousarray(detect_frame)

            detections = self.detector.detect(detect_frame)
            
            # Translate detection boxes back to full-frame coords if cropped
            if crop_rect is not None:
                translated = []
                for d in detections:
                    bx1, by1, bx2, by2 = d.bbox
                    translated.append(Detection(
                        class_id=d.class_id,
                        class_name=d.class_name,
                        confidence=d.confidence,
                        bbox=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                        timestamp=d.timestamp
                    ))
                detections = translated
            
            # Apply ROI filtering
            detections = self.roi_manager.filter_detections(detections)
            
            self.stats['total_detections'] += len(detections)
        
        # Temporal filtering
        triggers = self.temporal_filter.add_detections(detections, self.frame_count)
        
        # Send alerts for triggered classes
        for class_name, should_alert in triggers.items():
            if should_alert and self.alert_manager.should_send_alert(class_name):
                self.alert_manager.send_alert(class_name, detections, frame)
        
        return FrameData(
            frame=frame,
            frame_number=self.frame_count,
            motion_info=motion_info,
            detections=detections
        )
    
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
        try:
            if hasattr(self, 'stop_event'):
                self.stop_event.set()
            if hasattr(self, 'capture_worker') and self.capture_worker is not None:
                self.capture_worker.join(timeout=2.0)
        except Exception:
            pass
        self.capture.release()
        try:
            if hasattr(self, 'alert_manager'):
                self.alert_manager.shutdown()
        except Exception:
            pass
        try:
            if hasattr(self, 'event_store') and self.event_store is not None:
                self.event_store.close()
        except Exception:
            pass
        self.logger.info("👋 AlertGate stopped")

if __name__ == "__main__":
    sentinel = AlertGate()
    sentinel.run()
