import cv2
from ultralytics import YOLO
 
model = YOLO("yolov8n.pt")
vehicle_ids = [k for k, v in model.names.items() if v in ["car","truck","bus","motorcycle"]]
 
cap = cv2.VideoCapture("../data/test_clips/normal_driving_1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
out = cv2.VideoWriter("../outputs/day1_no_tracking.mp4",
                       cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
 
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    result = model(frame, classes=vehicle_ids, verbose=False)[0]
    annotated = result.plot()   # draws boxes with built-in colors/labels
    out.write(annotated)
    frame_num += 1
 
cap.release()
out.release()
print(f"Processed {frame_num} frames -> outputs/day1_no_tracking.mp4")