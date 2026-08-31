from moviepy import VideoFileClip
 
def extract_clip(video_path, event_time_sec, before_sec=120, after_sec=120, output_path="outputs/crash_clip.mp4"):
    clip = VideoFileClip(video_path)
    duration = clip.duration
 
    start = max(0, event_time_sec - before_sec)
    end = min(duration, event_time_sec + after_sec)
 
    if start >= end:
        raise ValueError("Invalid clip window — check event_time_sec against video duration.")
 
    subclip = clip.subclipped(start, end)
    subclip.write_videofile(output_path, codec="libx264", audio=False)
    clip.close()
    return output_path, start, end