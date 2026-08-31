from collections import defaultdict


def plot_velocity(velocity_log, chosen_id=None, output_path="../outputs/day3_raw_velocity.png"):
    try:
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


import streamlit as st
import tempfile, os
from evaluate import run_pipeline, generate_overlay_video
from clip_extractor import extract_clip
 
st.set_page_config(page_title="Smart Crash Guard", layout="centered")
st.title("Smart Crash Guard")
st.write("Upload a driving video. The system will detect a likely accident event and extract the surrounding footage.")
 
uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
 
if uploaded_file is not None:
    # Save upload to a temp file — YOLO/OpenCV need a real file path, not an in-memory buffer
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
 
    st.video(video_path)
 
    if st.button("Run detection"):
        with st.spinner("Running detection and tracking — this can take a while on CPU..."):
            flagged, fps = run_pipeline(video_path)
 
        if not flagged:
            st.warning("No accident event detected in this video.")
        else:
            event_time = flagged[0] / fps
            st.success(f"Event detected at {event_time:.2f} seconds.")
 
            os.makedirs("outputs", exist_ok=True)
 
            with st.spinner("Extracting clip..."):
                clip_path, start, end = extract_clip(video_path, event_time,
                                                       output_path="outputs/crash_clip.mp4")
            st.write(f"Extracted clip: {start:.1f}s to {end:.1f}s")
            st.video(clip_path)
 
            with open(clip_path, "rb") as f:
                st.download_button("Download extracted clip", f, file_name="crash_clip.mp4")
 
            with st.spinner("Generating debug overlay (bboxes + velocity)..."):
                overlay_path = "outputs/overlay.mp4"
                generate_overlay_video(video_path, overlay_path)
            st.write("Debug overlay (see why this frame was flagged):")
            st.video(overlay_path)
