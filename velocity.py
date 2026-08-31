import math
import statistics


def distance(a, b):
    """Euclidean distance between two points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


class VelocityTracker:
    def __init__(self, smoothing_window=5):
        self.history = {}
        self.velocity_history = {}
        self.smoothing_window = smoothing_window

    def update(self, tracked_objects, fps):
        smoothed = {}
        for obj in tracked_objects:
            tid, c = obj["id"], obj["centroid"]
            self.history.setdefault(tid, []).append(c)
            hist = self.history[tid]

            if len(hist) >= 2:
                raw_v = distance(hist[-2], hist[-1]) * fps
            else:
                raw_v = 0.0

            self.velocity_history.setdefault(tid, []).append(raw_v)
            recent = self.velocity_history[tid][-self.smoothing_window:]
            smoothed[tid] = sum(recent) / len(recent)

        return smoothed


def rolling_stats(velocity_list, window=15):
    """Returns (mean, stdev) of the last N velocity readings."""
    recent = velocity_list[-window:]
    if len(recent) < 2:
        return None, None
    return statistics.mean(recent), statistics.stdev(recent)


def is_sudden_deceleration(velocity_list, k=2.5, window=15):
    """
    Flags a sudden drop if the latest velocity is more than
    k standard deviations BELOW the recent rolling mean.
    """
    mean, stdev = rolling_stats(velocity_list[:-1], window)
    if mean is None or stdev == 0:
        return False
    current = velocity_list[-1]
    return current < (mean - k * stdev)


__all__ = ["VelocityTracker", "is_sudden_deceleration"]
