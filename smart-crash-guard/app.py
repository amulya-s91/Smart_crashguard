
import streamlit as st
import tempfile, os
from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import run_pipeline, generate_overlay_video
from clip_extractor import extract_clip
 
st.set_page_config(page_title="Smart Crash Guard", layout="centered")
st.title("Smart Crash Guard")
st.write("smart crashguard detects the accident and extract the footage")
 
uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
 
if uploaded_file is not None:
    # Save upload to a temp file — YOLO/OpenCV need a real file path, not an in-memory buffer
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.getvalue())
    tfile.close()
    video_path = tfile.name
 
    st.video(video_path)
 
    if st.button("Run detection"):
        try:
            with st.spinner("video processing... "):
                flagged, fps, details = run_pipeline(video_path, return_details=True)
 
            os.makedirs("outputs", exist_ok=True)
            with st.spinner("processing video..."):
                overlay_path = "outputs/overlay.mp4"
                generate_overlay_video(video_path, overlay_path)
        except (ConnectionResetError, BrokenPipeError) as exc:
            st.error(f"The connection closed while processing the video: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            st.stop()

        if not flagged:
            st.warning("Vehicles were processed, but no accident event was detected in this video.")
        else:
            event_time = details["time_seconds"]
            detected_at = datetime.now().astimezone()
            details.update({
                "date": detected_at.strftime("%Y-%m-%d"),
                "time": detected_at.strftime("%I:%M:%S %p"),
                "timezone": detected_at.tzname() or "local time",
            })
            st.success(f"Event detected at {event_time:.2f} seconds.")
            st.subheader("Accident details")
            detail_row = st.columns(3)
            detail_row[0].metric("Date", details["date"])
            detail_row[1].metric("Time", f'{details["time"]} ({details["timezone"]})')
            detail_row[2].metric("Video position", f'{details["time_seconds"]:.2f} sec')
            with st.spinner("Extracting clip..."):
                clip_path, start, end = extract_clip(video_path, event_time,
                                                       output_path="outputs/crash_clip.mp4")
            st.write(f"Extracted clip: {start:.1f}s to {end:.1f}s")
            st.video(clip_path)

            with open(clip_path, "rb") as clip_file:
                clip_bytes = clip_file.read()
            st.download_button(
                "Download extracted clip",
                data=clip_bytes,
                file_name="crash_clip.mp4",
                mime="video/mp4",
            )
