"""Live-control logic without cameras: road-width density, adaptive timing, severity and alert delivery."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from traffic_ai.incident.alert import AlertManager
from traffic_ai.incident.notify import send_alert
from traffic_ai.incident.rules import Candidate
from traffic_ai.incident.severity import assess
from traffic_ai.live.setup import DEFAULT, lanes_of
from traffic_ai.live.signal import AdaptiveSignal
from traffic_ai.vision.detector import Track
from traffic_ai.vision.features import TrackState, TrafficStateEstimator
from traffic_ai.vision.geometry import Approach, CameraConfig


def _cam(width_m):
    poly = np.array([[0, 0], [140, 0], [140, 800], [0, 800]], dtype=np.float32)
    line = np.array([[0, 800], [140, 800]], dtype=np.float32)
    return CameraConfig([Approach("N", poly, line, length_m=40, lanes=2, width_m=width_m)], metres_per_pixel=0.05)


def test_wider_road_means_lower_density():
    tracks = [Track(i, "Car", 0.9, np.array([10, 100 * i, 50, 100 * i + 80])) for i in range(1, 7)]
    narrow = TrafficStateEstimator(_cam(7.0)).update(tracks, 0.0)["N"]
    wide = TrafficStateEstimator(_cam(14.0)).update(tracks, 0.0)["N"]
    assert narrow["occupancy_pct"] > 0 and abs(narrow["occupancy_pct"] - 2 * wide["occupancy_pct"]) < 1e-6
    assert abs(narrow["density_pcu_km_lane"] - 2 * wide["density_pcu_km_lane"]) < 1e-6
    assert lanes_of({"road_width_m": 10.5, "lanes": 0}) == 3


def _signal(**over):
    cfg = dict(DEFAULT["signal"], **over)
    return AdaptiveSignal(["N", "E", "S", "W"], cfg, {a: 2 for a in "NESW"})


def test_green_time_follows_density():
    sig = _signal()
    for a, q in zip("NESW", (20, 2, 8, 0)):
        sig.update_demand(a, {"queue_pcu": q, "pcu": q, "occupancy_pct": 10}, True, alpha=1.0)
    g = {a: sig.green_for(i)[0] for i, a in enumerate("NESW")}
    assert g["N"] > g["S"] > g["E"] >= sig.cfg["min_green"]
    assert g["N"] <= sig.cfg["max_green"]
    assert sig._empty(3) and not sig._empty(0)


def test_offline_camera_falls_back_to_fixed_green():
    sig = _signal()
    sig.update_demand("N", None, False)
    g, why = sig.green_for(0)
    assert g == sig.cfg["fixed_green"] and "offline" in why


def test_cycle_runs_green_yellow_allred_and_skips_empty():
    sig = _signal(min_green=5, max_green=20)
    for a, q in zip("NESW", (6, 0, 6, 0)):
        sig.update_demand(a, {"queue_pcu": q, "pcu": q, "occupancy_pct": 5}, True, alpha=1.0)
    seen = []
    t = 0.0
    while t < 200:
        st = sig.tick(t)
        if not seen or seen[-1] != (st["phase"], st["stage"]):
            seen.append((st["phase"], st["stage"]))
        if st["stage"] == "green":  # the queue clears once served
            for a in st["phase"].split("+"):
                sig.update_demand(a, {"queue_pcu": 0, "pcu": 0, "occupancy_pct": 0}, True, alpha=1.0)
            for a in "NS":
                if a not in st["phase"]:
                    sig.update_demand(a, {"queue_pcu": 6, "pcu": 6, "occupancy_pct": 5}, True, alpha=1.0)
        t += 0.5
    phases = [p for p, s in seen if s == "green"]
    assert "E" not in phases and "W" not in phases, "empty approaches are skipped"
    assert ("N", "yellow") in seen and ("N", "all_red") in seen
    assert "S" in phases


def _state(cls, speeds, t0=0.0, dt=0.1):
    st = TrackState()
    for k, v in enumerate(speeds):
        st.speeds.append((t0 + k * dt, v))
    st.speed, st.cls = speeds[-1], cls
    return st


def test_severity_levels():
    fast = [12.0] * 15 + [0.0] * 5   # ~43 km/h to a stop
    slow = [1.5] * 15 + [0.0] * 5
    t = 1.9
    severe = assess(Candidate(t, [1, 2], "N", 0.9, ["abrupt_decel"]), {1: _state("Bus", fast), 2: _state("Two-wheeler", fast)})
    minor = assess(Candidate(t, [1, 2], "N", 0.9, ["overlap_converging"]),
                   {1: _state("Sedan", [*slow[:-1], 0.8]), 2: _state("Hatchback", [*slow[:-1], 0.8])})
    assert severe.level == "severe" and severe.impact_speed_kmh > 35
    assert minor.level == "minor"


def test_severe_alert_reaches_webhook(tmp_path):
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    setup = json.loads(json.dumps(DEFAULT))
    setup["junction"].update(name="Test Chowk", lat=20.29, lon=85.82)
    setup["alerts"]["webhook"] = f"http://127.0.0.1:{srv.server_port}/"
    fast = [12.0] * 15 + [0.0] * 5
    cand = Candidate(1.9, [1, 2], "N", 0.9, ["abrupt_decel"])
    sev = assess(cand, {1: _state("Truck", fast), 2: _state("Two-wheeler", fast)})
    inc = AlertManager(tmp_path, "Test Chowk").raise_incident(
        cand, 0.9, True, [{"class": "Truck", "speed_kmh": 0}], [], 10, severity=sev, camera="North camera",
        maps_url="https://maps.google.com/?q=20.29,85.82", location="Test Chowk, North approach")
    results = []
    send_alert(inc, setup, on_result=lambda ch, ok, msg: results.append((ch, ok, msg)), background=False)
    srv.shutdown()
    assert results == [("webhook", True, "")]
    assert got[0]["severity"] == "severe" and "Test Chowk" in got[0]["message"] and "maps.google" in got[0]["message"]
