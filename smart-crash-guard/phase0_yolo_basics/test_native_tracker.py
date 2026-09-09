# phase0_yolo_basics/test_naive_tracker.py  (or project root, your choice)
from pathlib import Path 
from collections import defaultdict
import argparse
import cv2, sys
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detector import VehicleDetector
from tracker import NaiveTracker
from kinematics import VelocityTracker
 
det = VehicleDetector()
trk = NaiveTracker()
velocity_tracker = VelocityTracker()
velocity_log = defaultdict(list)

parser = argparse.ArgumentParser(description="Track vehicles and plot raw velocity.")
parser.add_argument(
    "input_video",
    nargs="?",
    type=Path,
    default=PROJECT_ROOT / "data" / "test_clips" / "video.mp4",
    help="Path to a video file or image directory",
)
parser.add_argument(
    "--max-frames",
    type=int,
    default=None,
    help="Stop after processing this many frames (useful for quick CPU tests)",
)
parser.add_argument(
    "--fps",
    type=float,
    default=30.0,
    help="FPS for image-directory input; ignored for video files (default: 30)",
)
args = parser.parse_args()
input_path = args.input_video if args.input_video.is_absolute() else Path.cwd() / args.input_video

image_paths = None
cap = None
if input_path.is_dir():
    image_paths = sorted(
        path for path in input_path.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No JPG, JPEG, or PNG frames found in: {input_path}")
    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise OSError(f"Could not read image: {image_paths[0]}")
    fps = args.fps
    h, w = first_frame.shape[:2]
else:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open input video: {input_path}\n"
            "Provide a video path or an image directory."
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    w, h = int(cap.get(3)), int(cap.get(4))

out = cv2.VideoWriter(str(PROJECT_ROOT / "outputs" / "day2_naive_tracker.mp4"),
                       cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
if not out.isOpened():
    raise OSError("Could not open the output video writer")

frame_index = 0
while True:
    if args.max_frames is not None and frame_index >= args.max_frames:
        break
    if image_paths is not None:
        if frame_index >= len(image_paths):
            break
        frame = cv2.imread(str(image_paths[frame_index]))
        frame_index += 1
        ret = frame is not None
    else:
        ret, frame = cap.read()
        frame_index += 1
    if not ret:
        break
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h))
    detections = det.detect(frame)
    tracked = trk.update(detections)
    velocities = velocity_tracker.update(tracked, fps)
    for tid, velocity in velocities.items():
        velocity_log[tid].append(velocity)
    for t in tracked:
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {t['id']}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    out.write(frame)
    if frame_index % 10 == 0:
        print(f"Processed {frame_index} frames...", flush=True)
 
if cap is not None:
    cap.release()
out.release()
print("Done -> outputs/day2_naive_tracker.mp4")

if velocity_log:
    chosen_id = max(velocity_log, key=lambda tid: len(velocity_log[tid]))
    plt.plot(velocity_log[chosen_id])
    plt.xlabel("Frame")
    plt.ylabel("Velocity (px/sec)")
    plt.title(f"Raw velocity, track {chosen_id}")
    plt.savefig(PROJECT_ROOT / "outputs" / "day3_raw_velocity.png")
    plt.close()
    print("Done -> outputs/day3_raw_velocity.png")