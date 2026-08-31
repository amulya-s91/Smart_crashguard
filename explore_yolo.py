import os

os.environ.setdefault("YOLO_OFFLINE", "true")

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image_path = "smart-crash-guard/phase0_yolo_basics/test_image.jpg"
prediction_options = dict(imgsz=320, verbose=False, device="cpu")
results = model(image_path, **prediction_options)
result = results[0]
print(type(result))
print(result.boxes)
boxes = result.boxes
print("Number of detections:", len(boxes)) 
print("Bounding boxes (xyxy format = x1,y1,x2,y2 corners):")
print(boxes.xyxy)
print("Confidence scores:")
print(boxes.conf)
print("Class IDs:")
print(boxes.cls)
print("Class names dictionary:",model.names)
for conf_level in [0.1, 0.5, 0.9]:
    r = model(image_path, conf=conf_level, **prediction_options)[0]
    print(f"Confidence threshold {conf_level}: {len(r.boxes)} detections")
    r.save(filename=f"output_conf_{conf_level}.jpg")
r_normal = model(image_path, iou=0.45, **prediction_options)[0]   # default-ish NMS behavior
r_loose  = model(image_path, iou=0.95, **prediction_options)[0]    # NMS barely removes anything
 
print("Normal NMS:", len(r_normal.boxes), "boxes")
print("Loose NMS:", len(r_loose.boxes), "boxes")
r_loose.save(filename="output_loose_nms.jpg")

vehicle_classes = ["car", "truck", "bus", "motorcycle"]
vehicle_ids = [k for k, v in model.names.items() if v in vehicle_classes]
print("Vehicle class IDs:", vehicle_ids)
 
r = model(image_path, classes=vehicle_ids, **prediction_options)[0]
print("Vehicle-only detections:", len(r.boxes))
r.save(filename="output_vehicles_only.jpg")
