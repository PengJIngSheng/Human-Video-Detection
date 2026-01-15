import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not found. Please install it first:")
    print("pip install ultralytics")
    import sys

    sys.exit(1)


class StaffDetector:
    def __init__(self,
                 weights_path: str,
                 device: str = 'cuda',
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 img_size: int = 640,
                 half: bool = True):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device

        print(f"Loading model from {weights_path}...")
        self.model = YOLO(weights_path)

        self.model.to(device)

        self.names = self.model.names

        print(f"Model loaded successfully on {device}")
        print(f"Model classes: {self.names}")

        self._warmup()

    def _warmup(self):
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy_img, verbose=False)
        print("Model warmup complete")

    def detect(self, img: np.ndarray, target_class: Optional[int] = None) -> List[Dict]:

        results = self.model.predict(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False,
            device=self.device
        )

        detections = []

        for result in results:
            boxes = result.boxes

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())

                if target_class is not None and cls != target_class:
                    continue

                detection = {
                    'bbox': [float(x) for x in xyxy],  # [x1, y1, x2, y2]
                    'confidence': conf,
                    'class': cls,
                    'class_name': self.names[cls]
                }
                detections.append(detection)

        return detections

    def detect_staff(self, img: np.ndarray) -> List[Dict]:

        staff_detections = self.detect(img)

        return staff_detections

    def batch_detect(self, frames: List[np.ndarray]) -> List[List[Dict]]:

        results = []
        for frame in frames:
            detections = self.detect_staff(frame)
            results.append(detections)
        return results
