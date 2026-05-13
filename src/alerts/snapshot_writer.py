import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2

from core.types import Detection


class SnapshotWriter:
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger

    def save(self, frame, detections: List[Detection], frame_number: int) -> Optional[str]:
        """Save an annotated snapshot if alert images are enabled."""
        if not self.config.get('alerts', {}).get('send_image', False):
            return None

        snapshot_config = self.config.get('recording')
        if not snapshot_config:
            self.logger.warning("❌ 'recording' section not found in config. Cannot save snapshot.")
            return None

        directory = snapshot_config.get('snapshots_dir')
        if not directory:
            self.logger.error("❌ Snapshot directory ('snapshots_dir') not configured under 'recording'.")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cat_detection_{timestamp}_frame_{frame_number}.jpg"
        filepath = os.path.join(directory, filename)

        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            label = f"{det.class_name} {det.confidence:.2f}"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        os.makedirs(directory, exist_ok=True)
        cv2.imwrite(filepath, annotated_frame)
        self.logger.info(f"📸 Snapshot saved to {filepath}")
        return filepath
