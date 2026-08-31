import math
 
def euclidean(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
 
class VelocityTracker:
    """
    Maintains a history of centroids per track ID and computes
    RAW (unsmoothed) per-frame velocity in pixels/frame.
    """
    def __init__(self):
        self.history = {}   # id -> list of centroids
 
    def update(self, tracked_objects, fps):
        velocities = {}
        for obj in tracked_objects:
            tid, c = obj["id"], obj["centroid"]
            self.history.setdefault(tid, []).append(c)
            hist = self.history[tid]
            if len(hist) >= 2:
                dist_px = euclidean(hist[-2], hist[-1])
                velocities[tid] = dist_px * fps   # px/frame * frames/sec = px/sec
            else:
                velocities[tid] = 0.0
        return velocities
