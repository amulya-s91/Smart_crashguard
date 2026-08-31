from ultralytics import YOLO
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 
class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf
        self.vehicle_ids = [k for k, v in self.model.names.items()
                             if v in ["car", "truck", "bus", "motorcycle"]]
 
    def detect(self, frame):
        """Returns list of dicts: [{'bbox': (x1,y1,x2,y2), 'conf': float}, ...]"""
        result = self.model(frame, classes=self.vehicle_ids,
                             conf=self.conf, verbose=False)[0]
        detections = []
        for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
            x1, y1, x2, y2 = box.tolist()
            detections.append({"bbox": (x1, y1, x2, y2), "conf": float(conf)})
        return detections