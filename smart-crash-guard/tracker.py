import math
 
def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)
 
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
 
class NaiveTracker:
    """
    Version 1 — deliberately naive: matches every detection to the
    nearest previous track, no matter how far away it is. Used to
    demonstrate WHY a distance gate is needed (Day 2 checkpoint).
    """
    def __init__(self):
        self.tracks = {}      # id -> last known centroid
        self.next_id = 0
 
    def update(self, detections):
        assigned = []
        used_ids = set()
        for det in detections:
            c = centroid(det["bbox"])
            if not self.tracks:
                new_id = self._new_track(c)
            else:
                # nearest existing track, no gating
                nearest_id = min(self.tracks, key=lambda tid: distance(self.tracks[tid], c))
                if nearest_id in used_ids:
                    new_id = self._new_track(c)   # already claimed this frame
                else:
                    new_id = nearest_id
                    self.tracks[new_id] = c
            used_ids.add(new_id)
            assigned.append({**det, "id": new_id, "centroid": c})
        return assigned
 
    def _new_track(self, c):
        tid = self.next_id
        self.tracks[tid] = c
        self.next_id += 1
        return tid
class GatedTracker:
    """
    Version 2 — adds a max_distance gate. If the nearest existing track
    is farther than this, treat the detection as a NEW object rather
    than force-matching it to a distant, unrelated track.
    """
    def __init__(self, max_distance=80, max_missed_frames=10):
        self.tracks = {}          # id -> centroid
        self.missed = {}          # id -> frames since last matched
        self.next_id = 0
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
 
    def update(self, detections):
        assigned = []
        used_ids = set()
 
        for det in detections:
            c = centroid(det["bbox"])
            candidates = {tid: distance(pos, c) for tid, pos in self.tracks.items()
                          if tid not in used_ids}
            if candidates:
                nearest_id = min(candidates, key=candidates.get)
                if candidates[nearest_id] <= self.max_distance:
                    tid = nearest_id
                    self.tracks[tid] = c
                    self.missed[tid] = 0
                else:
                    tid = self._new_track(c)
            else:
                tid = self._new_track(c)
            used_ids.add(tid)
            assigned.append({**det, "id": tid, "centroid": c})
 
        # age out tracks that had no matching detection this frame
        for tid in list(self.tracks):
            if tid not in used_ids:
                self.missed[tid] = self.missed.get(tid, 0) + 1
                if self.missed[tid] > self.max_missed_frames:
                    del self.tracks[tid]
                    del self.missed[tid]
 
        return assigned
 
    def _new_track(self, c):
        tid = self.next_id
        self.tracks[tid] = c
        self.missed[tid] = 0
        self.next_id += 1
        return tid  