"""Stage 1: flag candidate accidents from trajectory cues.

Cues seen on a track within a short window add up to a candidate score in [0, 1]:
abrupt deceleration, a sudden stop from speed, two boxes overlapping while converging,
an abrupt heading change, and a vehicle left stationary while its approach keeps moving.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from traffic_ai.vision.features import TrackState


@dataclass
class Candidate:
    t: float
    track_ids: list[int]
    approach: str | None
    score: float
    cues: list[str] = field(default_factory=list)
    box: np.ndarray | None = None


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _heading_change(st: TrackState, window: float = 1.5) -> float:
    pts = [(h[0], h[1], h[2]) for h in st.hist if st.last_seen - h[0] <= 2 * window]
    if len(pts) < 6:
        return 0.0
    pts = np.array(pts)
    mid = len(pts) // 2
    v1, v2 = pts[mid, 1:] - pts[0, 1:], pts[-1, 1:] - pts[mid, 1:]
    if np.linalg.norm(v1) < 1.0 or np.linalg.norm(v2) < 0.3:
        return 0.0
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _converging(a: TrackState, b: TrackState) -> bool:
    if len(a.hist) < 3 or len(b.hist) < 3:
        return False
    pa0, pa1 = np.array(a.hist[-3][1:3]), np.array(a.hist[-1][1:3])
    pb0, pb1 = np.array(b.hist[-3][1:3]), np.array(b.hist[-1][1:3])
    return np.linalg.norm(pa1 - pb1) < np.linalg.norm(pa0 - pb0) - 0.1


class CandidateDetector:
    WEIGHTS = {"abrupt_decel": 0.35, "sudden_stop": 0.3, "overlap_converging": 0.45,
               "heading_change": 0.2, "stopped_in_flow": 0.3}

    def __init__(self, decel_threshold: float = 4.0, stop_seconds: float = 8.0, overlap_iou: float = 0.25,
                 score_threshold: float = 0.6, window_s: float = 3.0, cooldown_s: float = 20.0):
        self.decel_thr, self.stop_s, self.iou_thr = decel_threshold, stop_seconds, overlap_iou
        self.score_thr, self.window, self.cooldown = score_threshold, window_s, cooldown_s
        self.events: dict[int, dict[str, float]] = defaultdict(dict)  # track -> cue -> last time seen
        self.partners: dict[int, set[int]] = defaultdict(set)
        self.stopped_since: dict[int, float] = {}
        self.last_alert: dict[int, float] = {}

    def update(self, tracks: dict[int, TrackState], t: float, stream_speed: dict[str, float]) -> list[Candidate]:
        live = {i: s for i, s in tracks.items() if s.last_seen >= t - 0.5}
        for i, st in live.items():
            ev = self.events[i]
            if st.accel < -self.decel_thr:
                ev["abrupt_decel"] = t
            if st.speed < 0.5:
                self.stopped_since.setdefault(i, t)
                if st.max_speed_since(t - 1.5) > 6.0:
                    ev["sudden_stop"] = t
                if t - self.stopped_since[i] > self.stop_s and stream_speed.get(st.hist[-1][3], 0.0) > 3.0:
                    ev["stopped_in_flow"] = t
            else:
                self.stopped_since.pop(i, None)
            if _heading_change(st) > 60:
                ev["heading_change"] = t
        ids = list(live)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                sa, sb = live[ids[a]], live[ids[b]]
                if iou(sa.hist[-1][4], sb.hist[-1][4]) > self.iou_thr and _converging(sa, sb):
                    self.events[ids[a]]["overlap_converging"] = t
                    self.events[ids[b]]["overlap_converging"] = t
                    self.partners[ids[a]].add(ids[b])
                    self.partners[ids[b]].add(ids[a])
        out = []
        for i in ids:
            cues = sorted(k for k, ts in self.events[i].items() if t - ts <= self.window)
            score = min(1.0, sum(self.WEIGHTS[k] for k in cues))
            if score >= self.score_thr and t - self.last_alert.get(i, -1e9) > self.cooldown:
                group = [i] + sorted(j for j in self.partners[i] if j in live)
                for j in group:
                    self.last_alert[j] = t
                out.append(Candidate(t, group, live[i].hist[-1][3], score, cues, live[i].hist[-1][4]))
        for k in [k for k in self.events if k not in tracks]:
            self.events.pop(k)
            self.partners.pop(k, None)
        return out
