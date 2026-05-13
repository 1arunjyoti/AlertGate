import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List

from core.types import Detection


class AlertManager:
    def __init__(
        self,
        alerts_config: Dict,
        web_config: Dict,
        notifier,
        snapshot_writer,
        event_store,
        dashboard,
        temporal_filter,
        stats: Dict,
        logger,
        frame_number_getter: Callable[[], int],
    ):
        self.alerts_config = alerts_config
        self.web_config = web_config
        self.notifier = notifier
        self.snapshot_writer = snapshot_writer
        self.event_store = event_store
        self.dashboard = dashboard
        self.temporal_filter = temporal_filter
        self.stats = stats
        self.logger = logger
        self.frame_number_getter = frame_number_getter
        self.last_alert_times: Dict[str, float] = {}
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='alerts')

    def should_send_alert(self, class_name: str) -> bool:
        """Check if enough time has passed since the last alert for this class."""
        current_time = time.time()
        cooldown = self.alerts_config.get('cooldown_seconds', {}).get(class_name, 60)

        last_alert = self.last_alert_times.get(class_name, 0)
        return (current_time - last_alert) >= cooldown

    def send_alert(self, class_name: str, detections: List[Detection], frame):
        """Send an alert notification without blocking the main loop."""
        class_detections = [d for d in detections if d.class_name == class_name]
        if not class_detections:
            return

        best_detection = max(class_detections, key=lambda d: d.confidence)
        frame_number = self.frame_number_getter()

        self.last_alert_times[class_name] = time.time()
        self.temporal_filter.reset_class_history(class_name)

        try:
            self.executor.submit(
                self._send_alert_task,
                class_name,
                best_detection,
                list(detections),
                frame,
                frame_number,
            )
        except Exception:
            self._send_alert_task(class_name, best_detection, list(detections), frame, frame_number)

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _send_alert_task(
        self,
        class_name: str,
        best_detection: Detection,
        detections: List[Detection],
        frame,
        frame_number: int,
    ):
        try:
            snapshot_path = self.snapshot_writer.save(frame, detections, frame_number)
            self.notifier.send_detection_alert(best_detection, snapshot_path)
            self.stats['alerts_sent'] += 1

            event_data = {
                'class_name': best_detection.class_name,
                'confidence': best_detection.confidence,
                'timestamp': best_detection.timestamp.isoformat(),
                'frame_number': frame_number,
                'zone': 'backyard',
                'snapshot_path': snapshot_path,
            }

            self._persist_event(event_data)

            if self.web_config.get('enabled') and self.dashboard is not None:
                self.dashboard.add_event(event_data)

            self.logger.info(f"🚨 Alert sent: {class_name} ({best_detection.confidence:.2f})")
        except Exception as e:
            self.logger.error(f"❌ Failed to send alert: {e}")

    def _persist_event(self, event_data: Dict):
        try:
            if self.event_store is None:
                return

            insert_id = self.event_store.add_event(event_data)
            if isinstance(insert_id, int) and insert_id > 0:
                event_data['id'] = insert_id
        except Exception:
            pass
