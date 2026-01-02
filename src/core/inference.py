import cv2
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO
from .types import Detection
import torch

class YOLODetector:
    def __init__(self, det_cfg: Dict[str, Any]):
        """
        det_cfg keys:
          - model: str ("yolo11n.pt" or local path)
          - confidence: float
          - target_classes: List[str]
          - inference_size: int (e.g., 640)
        """
        self.model_name = det_cfg.get("model", "yolo11n.pt")
        self.conf_threshold = float(det_cfg.get("confidence", 0.55))
        self.imgsz = int(det_cfg.get("inference_size", 640))
        self.iou = float(det_cfg.get("iou", 0.45))
        self.max_det = int(det_cfg.get("max_det", 50))
        self.target_classes = [c.lower() for c in det_cfg.get("target_classes", ["cat"])]

        # Load model
        self.model = YOLO(self.model_name)
        self.class_names = self.model.names
        # Build map name->id for filtering
        self.name_to_id = {name.lower(): cid for cid, name in self.class_names.items()}
        self.target_ids = {self.name_to_id[c] for c in self.target_classes if c in self.name_to_id}

        # Select device and enable half precision on CUDA for speed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model.to(self.device)
            if self.device == "cuda":
                self.model.half()
                # Enable cuDNN autotuner for fixed-size inputs to accelerate convs
                torch.backends.cudnn.benchmark = True
        except Exception:
            # Fallback if model doesn't support .to()/.half()
            pass

        # Fuse Conv+BN where supported to reduce inference latency
        try:
            self.model.fuse()
        except Exception:
            pass

        # Ensure eval mode
        try:
            self.model.eval()
        except Exception:
            pass

        # Lightweight warmup to stabilize autotuner/JIT paths
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            n_warmup = 2
            with torch.inference_mode():
                for _ in range(n_warmup):
                    _ = self.model(dummy, imgsz=self.imgsz, verbose=False, conf=self.conf_threshold, iou=self.iou, max_det=self.max_det)
        except Exception:
            pass

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Runs inference on a BGR frame and returns a list of Detection.
        """
        # Filter at inference time to reduce overhead
        classes = list(self.target_ids) if self.target_ids else None
        with torch.inference_mode():
            results = self.model(
                frame_bgr,
                imgsz=self.imgsz,
                verbose=False,
                conf=self.conf_threshold,
                iou=self.iou,
                max_det=self.max_det,
                classes=classes,
            )
        detections: List[Detection] = []

        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf)
                # Additional safeguard filtering (should already be applied by model call)
                if conf < self.conf_threshold:
                    continue
                if self.target_ids and cls_id not in self.target_ids:
                    continue
                
                # Bounding box in xyxy format
                xyxy = b.xyxy.cpu().numpy().tolist()[0]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, xyxy)
                class_name = self.class_names.get(cls_id, str(cls_id))

                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2)
                    )
                )

        return detections

    @staticmethod
    def draw_detections(frame_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Utility to draw detection boxes and labels on a copy of the frame.
        """
        out = frame_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{d.class_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(out, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out
