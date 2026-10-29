"""Stage 3: incident summary and notification for an authorised operator or responder."""
from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2

log = logging.getLogger("traffic_ai.incident")


@dataclass
class Incident:
    incident_id: str
    location: str
    approach: str | None
    video_time_s: float
    wall_time: str
    candidate_score: float
    confirm_probability: float
    status: str               # "confirmed" or "needs_review"
    cues: list[str]
    vehicles: list[dict]
    clip_path: str | None
    snapshot_path: str | None
    summary: str
    severity: str | None = None           # minor | moderate | severe
    severity_score: float | None = None
    severity_reasons: list[str] | None = None
    impact_speed_kmh: float | None = None
    camera: str | None = None
    maps_url: str | None = None


class AlertManager:
    def __init__(self, out_dir: Path, location: str, start_time: datetime | None = None,
                 webhook_url: str | None = None):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.location, self.webhook = location, webhook_url
        self.start = start_time or datetime.now()
        self.incidents: list[Incident] = []

    def raise_incident(self, cand, prob: float, confirmed: bool, vehicles: list[dict], frames_bgr: list,
                       fps: float, severity=None, camera: str | None = None, maps_url: str | None = None,
                       when: datetime | None = None, location: str | None = None) -> Incident:
        n = len(self.incidents) + 1
        when = when or self.start + timedelta(seconds=cand.t)
        iid = f"INC-{when:%Y%m%d-%H%M%S}-{n:03d}"
        where = location or self.location
        clip = snap = None
        if frames_bgr:
            h, w = frames_bgr[0].shape[:2]
            clip = str(self.out / f"{iid}.mp4")
            vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for f in frames_bgr:
                vw.write(f)
            vw.release()
            snap = str(self.out / f"{iid}.jpg")
            img = frames_bgr[-1].copy()
            if cand.box is not None:
                x1, y1, x2, y2 = map(int, cand.box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.imwrite(snap, img)
        kinds = ", ".join(sorted({v["class"] for v in vehicles})) or "unknown vehicle"
        status = "confirmed" if confirmed else "needs_review"
        sev = f"{severity.level.upper()} " if severity is not None else ""
        app = f", {cand.approach} approach" if cand.approach and not location else ""
        summary = (f"{sev}{'collision' if confirmed else 'possible collision'} at {where}{app} "
                   f"at {when:%H:%M:%S on %d %b %Y}. Involves {len(vehicles)} vehicle(s): {kinds}. "
                   + (f"Severity factors: {'; '.join(severity.reasons)}. " if severity is not None else "")
                   + f"Cues: {', '.join(cand.cues)}. Confidence {prob:.0%}. Verify on camera before dispatch.")
        summary = summary[0].upper() + summary[1:]
        inc = Incident(iid, where, cand.approach, round(cand.t, 2), when.isoformat(timespec="seconds"),
                       round(cand.score, 3), round(prob, 3), status, cand.cues, vehicles, clip, snap, summary,
                       severity.level if severity is not None else None,
                       severity.score if severity is not None else None,
                       severity.reasons if severity is not None else None,
                       severity.impact_speed_kmh if severity is not None else None, camera, maps_url)
        self.incidents.append(inc)
        (self.out / f"{iid}.json").write_text(json.dumps(asdict(inc), indent=2))
        log.warning("ALERT %s", summary)
        if self.webhook and confirmed:
            self._post(asdict(inc))
        return inc

    def _post(self, payload: dict):
        try:
            req = urllib.request.Request(self.webhook, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:  # never let alert delivery stop the pipeline
            log.error("webhook failed: %s", e)

    def save_index(self):
        (self.out / "incidents.json").write_text(json.dumps([asdict(i) for i in self.incidents], indent=2))
