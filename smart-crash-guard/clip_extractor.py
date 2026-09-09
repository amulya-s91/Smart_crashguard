from pathlib import Path

import cv2


def extract_clip(video_path, event_time, output_path="outputs/crash_clip.mp4", padding=5.0):
	"""Extract a clip around an event time and return its path and bounds."""
	capture = cv2.VideoCapture(str(video_path))
	if not capture.isOpened():
		raise FileNotFoundError(f"Could not open input video: {video_path}")

	fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
	frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
	duration = frame_count / fps if frame_count else 0.0
	start = max(0.0, event_time - padding)
	end = min(duration, event_time + padding) if duration else event_time + padding
	width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	output_file = Path(output_path)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	writer = cv2.VideoWriter(
		str(output_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
	)
	if not writer.isOpened():
		capture.release()
		raise OSError(f"Could not open output video: {output_file}")

	capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
	try:
		while capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 < end:
			success, frame = capture.read()
			if not success:
				break
			writer.write(frame)
	finally:
		capture.release()
		writer.release()

	return str(output_file), start, end
