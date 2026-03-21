from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

from backend.app.config import get_settings


class YoloDetector:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = YOLO(settings.YOLO_WEIGHTS_PATH)
        self.class_names = self.model.names

    def detect(self, image_bgr: np.ndarray, conf_threshold: float = 0.25) -> List[dict]:
        """
        Run YOLO on a BGR image and return a list of detections.
        Each detection: {category, confidence, bbox: (x1,y1,x2,y2)}.
        """
        # Ultralytics expects RGB or path; we convert to RGB array.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb, verbose=False)

        detections: List[dict] = []
        if not results:
            return detections

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf.item())
                if conf < conf_threshold:
                    continue
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                category = (
                    self.class_names[cls_id]
                    if isinstance(self.class_names, dict)
                    else str(cls_id)
                )
                detections.append(
                    {
                        "category": category,
                        "confidence": conf,
                        "bbox": (
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                        ),
                    }
                )
        return detections


_yolo_detector: YoloDetector | None = None


def get_yolo_detector() -> YoloDetector:
    global _yolo_detector
    if _yolo_detector is None:
        _yolo_detector = YoloDetector()
    return _yolo_detector

