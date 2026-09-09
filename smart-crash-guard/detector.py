# detector.py
from pathlib import Path

from ultralytics import YOLO
 
class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.25):
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent / model_path
        self.model = YOLO(str(model_path))
        self.conf = conf
        vehicle_names = {"car", "truck", "bus", "motorcycle"}
        self.vehicle_ids = [
            class_id for class_id, name in self.model.names.items()
            if name in vehicle_names
        ]
 
    def detect(self, frame):
        """Return detected vehicles with bounding boxes, labels, and confidence."""
        result = self.model(frame, classes=self.vehicle_ids,
                             conf=self.conf, verbose=False)[0]
        detections = []
        for box, conf, class_id in zip(
            result.boxes.xyxy, result.boxes.conf, result.boxes.cls
        ):
            x1, y1, x2, y2 = box.tolist()
            class_id = int(class_id)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(conf),
                "label": self.model.names[class_id],
            })
        return detections