import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2, csv
from pathlib import Path
from detector import VehicleDetector
from tracker import GatedTracker, VelocityTracker, is_sudden_deceleration

PROJECT_DIR = Path(__file__).resolve().parent

def run_pipeline(video_path, k=2.5, window=15, min_history=20):
    det = VehicleDetector()
    trk = GatedTracker()
    vel = VelocityTracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return [], 30

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    flagged_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = det.detect(frame)
        tracked = trk.update(detections)
        velocities = vel.update(tracked, fps)

        for tid, v in velocities.items():
            history = vel.velocity_history.get(tid, [])
            if len(history) >= min_history and is_sudden_deceleration(history, k, window):
                flagged_frames.append(frame_idx)

        frame_idx += 1

    cap.release()
    return flagged_frames, fps

def generate_overlay_video(video_path, output_path, k=2.5, window=15, min_history=20):
    det = VehicleDetector()
    trk = GatedTracker()
    vel = VelocityTracker()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = det.detect(frame)
        tracked = trk.update(detections)
        velocities = vel.update(tracked, fps)

        for t in tracked:
            tid = t["id"]
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
            v = velocities.get(tid, 0.0)
            history = vel.velocity_history.get(tid, [])
            flagged = len(history) >= min_history and is_sudden_deceleration(history, k, window)

            color = (0, 0, 255) if flagged else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {tid} | {v:.0f}px/s"
            if flagged:
                label += "!!ACCIDENT!!"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Overlay video saved -> {output_path}")

def evaluate_all_clips(labels_path=None, clips_dir=None):
    labels_path = Path(labels_path) if labels_path else PROJECT_DIR / "labels.csv"
    clips_dir = Path(clips_dir) if clips_dir else PROJECT_DIR / "data" / "test_clips"
    results = []
    with labels_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = clips_dir / row["filename"]
            flagged, fps = run_pipeline(str(path))
            predicted_accident = len(flagged) > 0
            actual_accident = row["has_accident"] == "1"
            entry = {
                "filename": row["filename"],
                "actual": actual_accident,
                "predicted": predicted_accident,
                "correct": actual_accident == predicted_accident,
            }
            if actual_accident and predicted_accident:
                pred_time = flagged[0] / fps
                true_time = float(row["accident_time_sec"])
                entry["localization_error_sec"] = abs(pred_time - true_time)
            results.append(entry)
    return results

if __name__ == "__main__":
    results = evaluate_all_clips()
    for r in results:
        print(r)
    correct = sum(1 for r in results if r["correct"])
    print(f"\nAccuracy: {correct}/{len(results)}")