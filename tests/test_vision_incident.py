"""Synthetic-track tests for traffic-state estimation and the accident pipeline (no model needed)."""
import numpy as np

from traffic_ai.incident.alert import AlertManager
from traffic_ai.incident.confirm import HeuristicConfirmer
from traffic_ai.incident.rules import CandidateDetector
from traffic_ai.vision.detector import Track
from traffic_ai.vision.features import TrafficStateEstimator
from traffic_ai.vision.geometry import Approach, CameraConfig

FPS = 10


def _camera():
    # One approach: a 40 m x 7 m strip, 20 px per metre, stop line at y = 0 (top of the image)
    poly = np.array([[0, 0], [140, 0], [140, 800], [0, 800]], dtype=np.float32)
    line = np.array([[0, 0], [140, 0]], dtype=np.float32)
    return CameraConfig([Approach("N", poly, line, length_m=40, lanes=2)], metres_per_pixel=0.05)


def _box(x, y, w=36, h=80):
    return np.array([x - w / 2, y - h, x + w / 2, y])


def test_queue_and_speed():
    est = TrafficStateEstimator(_camera(), interval_s=10)
    for k in range(10 * FPS + 1):
        t = k / FPS
        moving_y = 780 - 10 * 20 * t  # 10 m/s towards the stop line
        tracks = [Track(1, "Car", 0.9, _box(40, 100)), Track(2, "Car", 0.9, _box(40, 200)),
                  Track(3, "Two-wheeler", 0.9, _box(100, max(moving_y, 300)))]
        s = est.update(tracks, t)
    assert s["N"]["queue_veh"] >= 2
    assert est.rows, "an interval row should have been emitted"
    row = est.rows[0]
    assert row["count_N"] == 3
    assert 4.0 < max(est.tracks[3].speed for _ in [0]) or row["speed_N"] > 0


def test_collision_raises_incident(tmp_path):
    cam = _camera()
    est = TrafficStateEstimator(cam, interval_s=60)
    det = CandidateDetector(decel_threshold=4.0, stop_seconds=8, overlap_iou=0.2, score_threshold=0.6)
    alerts = AlertManager(tmp_path, "Test junction")
    confirm = HeuristicConfirmer(est)
    found = []
    for k in range(8 * FPS):
        t = k / FPS
        if t < 3:  # a car closes in at 8 m/s on a two-wheeler moving at 2 m/s
            ya, yb = 700 - 160 * t, 420 - 40 * t
            xa, xb = 50, 60
        else:      # impact: both stop with their boxes overlapping
            ya, yb, xa, xb = 700 - 160 * 3 + 40, 300, 50, 60
        tracks = [Track(1, "Car", 0.9, _box(xa, ya)), Track(2, "Two-wheeler", 0.9, _box(xb, yb))]
        est.update(tracks, t)
        found += det.update(est.tracks, t, {"N": 5.0})
    assert found, "collision should be flagged"
    c = found[0]
    assert "overlap_converging" in c.cues or "abrupt_decel" in c.cues
    prob = confirm([], c)
    inc = alerts.raise_incident(c, prob, prob >= 0.5, [{"track_id": 1, "class": "Car"}], [], FPS)
    assert inc.status == "confirmed"
    assert (tmp_path / f"{inc.incident_id}.json").exists()


def test_normal_flow_has_no_alerts():
    est = TrafficStateEstimator(_camera(), interval_s=60)
    det = CandidateDetector()
    found = []
    for k in range(10 * FPS):
        t = k / FPS
        tracks = [Track(i, "Car", 0.9, _box(30 + 40 * (i % 2), 780 - 8 * 20 * t - 150 * i)) for i in range(4)]
        tracks = [tr for tr in tracks if tr.box[3] > 0]
        est.update(tracks, t)
        found += det.update(est.tracks, t, {"N": 8.0})
    assert not found
