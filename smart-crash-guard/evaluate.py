from pathlib import Path

import cv2

from detector import VehicleDetector
from kinematics import VelocityTracker
from tracker import GatedTracker


def _intersection_over_union(first_bbox, second_bbox):
	"""Return the overlap ratio of two bounding boxes."""
	first_x1, first_y1, first_x2, first_y2 = first_bbox
	second_x1, second_y1, second_x2, second_y2 = second_bbox
	intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
	intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
	intersection = intersection_width * intersection_height
	first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
	second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
	union = first_area + second_area - intersection
	return intersection / union if union else 0.0


def _new_vehicle_overlap(tracked, previous_boxes, minimum_iou=0.10):
	"""Detect two vehicles beginning to occupy the same image area."""
	for index, first in enumerate(tracked):
		for second in tracked[index + 1:]:
			current_iou = _intersection_over_union(first["bbox"], second["bbox"])
			if current_iou < minimum_iou:
				continue
			previous_iou = _intersection_over_union(
				previous_boxes.get(first["id"], first["bbox"]),
				previous_boxes.get(second["id"], second["bbox"]),
			)
			if previous_iou < minimum_iou:
				return True
	return False


def run_pipeline(video_path, velocity_threshold=800.0, min_consecutive_frames=2,
				 return_details=False):
	"""Detect likely events and optionally return details for the first event."""
	input_path = Path(video_path)
	capture = cv2.VideoCapture(str(input_path))
	if not capture.isOpened():
		raise FileNotFoundError(f"Could not open input video: {input_path}")

	fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
	detector = VehicleDetector()
	tracker = GatedTracker()
	velocity_tracker = VelocityTracker()
	flagged = []
	consecutive = 0
	frame_index = 0
	previous_boxes = {}
	max_vehicle_count = 0
	max_confidence = 0.0
	event_trigger = None

	try:
		while True:
			success, frame = capture.read()
			if not success:
				break

			detections = detector.detect(frame)
			tracked = tracker.update(detections)
			max_vehicle_count = max(max_vehicle_count, len(tracked))
			if detections:
				max_confidence = max(max_confidence, max(item["conf"] for item in detections))
			velocities = velocity_tracker.update(tracked, fps)
			is_fast = any(value >= velocity_threshold for value in velocities.values())
			is_collision = _new_vehicle_overlap(tracked, previous_boxes)
			consecutive = consecutive + 1 if is_fast else 0
			if is_collision or consecutive >= min_consecutive_frames:
				flagged.append(frame_index)
				if event_trigger is None:
					event_trigger = "vehicle collision/overlap" if is_collision else "abnormal vehicle motion"
			previous_boxes = {vehicle["id"]: vehicle["bbox"] for vehicle in tracked}
			frame_index += 1
	finally:
		capture.release()

	if not return_details:
		return flagged, fps
	first_event = flagged[0] if flagged else None
	details = {
		"detected": first_event is not None,
		"frame": first_event,
		"time_seconds": first_event / fps if first_event is not None else None,
		"trigger": event_trigger,
		"vehicle_count": max_vehicle_count,
		"confidence": max_confidence,
	}
	return flagged, fps, details


def generate_overlay_video(video_path, output_path, velocity_threshold=800.0):
	"""Write a video with tracked vehicle boxes, IDs, and velocities."""
	capture = cv2.VideoCapture(str(video_path))
	if not capture.isOpened():
		raise FileNotFoundError(f"Could not open input video: {video_path}")

	fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
	width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	output = cv2.VideoWriter(
		str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
	)
	if not output.isOpened():
		capture.release()
		raise OSError(f"Could not open output video: {output_path}")

	detector = VehicleDetector()
	tracker = GatedTracker()
	velocity_tracker = VelocityTracker()

	try:
		while True:
			success, frame = capture.read()
			if not success:
				break
			tracked = tracker.update(detector.detect(frame))
			velocities = velocity_tracker.update(tracked, fps)
			for vehicle in tracked:
				x1, y1, x2, y2 = map(int, vehicle["bbox"])
				velocity = velocities.get(vehicle["id"], 0.0)
				color = (0, 0, 255) if velocity >= velocity_threshold else (0, 255, 0)
				cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
				cv2.putText(
					frame,
					f"ID {vehicle['id']} {velocity:.0f}px/s",
					(x1, max(y1 - 8, 18)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.55,
					color,
					2,
				)
			output.write(frame)
	finally:
		capture.release()
		output.release()


def plot_velocity(velocity_log, chosen_id=None, output_path="../outputs/day3_raw_velocity.png"):
	try:
		import matplotlib
		matplotlib.use("Agg", force=True)
		import matplotlib.pyplot as plt
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			"matplotlib is required for plot_velocity(). Install it with: pip install matplotlib"
		) from exc

	if not velocity_log:
		print("No velocity data available yet; skipping plot generation.")
		return

	if chosen_id is None:
		chosen_id = max(velocity_log, key=lambda tid: len(velocity_log[tid]))

	plt.plot(velocity_log[chosen_id])
	plt.xlabel("Frame")
	plt.ylabel("Velocity (px/sec)")
	plt.title(f"Raw velocity, track {chosen_id}")
	plt.savefig(output_path)
	plt.close()
	print(f"Saved velocity plot to {output_path}")
