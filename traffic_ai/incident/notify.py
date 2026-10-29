"""Send collision alerts to the traffic controller: webhook (JSON), e-mail (SMTP) and Telegram.

Each channel is optional and configured in the app (Camera setup > Alerts), stored in configs/junction.json.
Delivery runs on a background thread and never blocks the cameras; results are reported back per channel.
"""
from __future__ import annotations

import json
import logging
import smtplib
import ssl
import threading
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict
from email.message import EmailMessage
from pathlib import Path

log = logging.getLogger("traffic_ai.notify")


def alert_text(inc, setup: dict) -> str:
    j = setup["junction"]
    lines = [f"🚨 {(inc.severity or 'unrated').upper()} COLLISION: {j.get('name') or 'Junction'}",
             f"Where: {inc.location}" + (f" (camera: {inc.camera})" if inc.camera else ""),
             f"When: {inc.wall_time.replace('T', ' ')}"]
    if j.get("lat") not in (None, "") and j.get("lon") not in (None, ""):
        lines.append(f"GPS: {float(j['lat']):.6f}, {float(j['lon']):.6f}")
    if inc.maps_url:
        lines.append(f"Map: {inc.maps_url}")
    veh = ", ".join(f"{v['class']} ({v.get('speed_kmh', '?')} km/h now)" for v in inc.vehicles) or "not identified"
    lines += [f"Vehicles: {veh}"]
    if inc.severity_reasons:
        lines.append("Why: " + "; ".join(inc.severity_reasons))
    lines += [f"Detection confidence: {inc.confirm_probability:.0%}  ·  ID {inc.incident_id}",
              "Automatic alert from the Traffic AI camera system. Verify on camera before dispatch."]
    return "\n".join(lines)


def _post_json(url: str, payload: dict):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10).read()


def _email(cfg: dict, subject: str, body: str, image: str | None):
    msg = EmailMessage()
    msg["Subject"], msg["From"] = subject, cfg.get("from") or cfg["username"]
    msg["To"] = cfg["to"]
    msg.set_content(body)
    if image and Path(image).exists():
        msg.add_attachment(Path(image).read_bytes(), maintype="image", subtype="jpeg", filename=Path(image).name)
    port = int(cfg.get("smtp_port") or 587)
    ctx = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(cfg["smtp_host"], port, context=ctx, timeout=20) as s:
            s.login(cfg["username"], cfg["password"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(cfg["smtp_host"], port, timeout=20) as s:
            s.starttls(context=ctx)
            if cfg.get("username"):
                s.login(cfg["username"], cfg["password"])
            s.send_message(msg)


def _telegram(cfg: dict, text: str, image: str | None):
    base = f"https://api.telegram.org/bot{cfg['bot_token']}"
    if image and Path(image).exists():
        boundary = uuid.uuid4().hex
        parts = []
        for k, v in (("chat_id", str(cfg["chat_id"])), ("caption", text[:1024])):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="incident.jpg"\r\n'
                     f"Content-Type: image/jpeg\r\n\r\n".encode() + Path(image).read_bytes() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(f"{base}/sendPhoto", data=b"".join(parts),
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        urllib.request.urlopen(req, timeout=20).read()
        if len(text) <= 1024:
            return
    data = urllib.parse.urlencode({"chat_id": cfg["chat_id"], "text": text}).encode()
    urllib.request.urlopen(f"{base}/sendMessage", data=data, timeout=20).read()


def channels(setup: dict) -> list[str]:
    a = setup["alerts"]
    out = []
    if a.get("webhook"):
        out.append("webhook")
    if a["email"].get("enabled") and a["email"].get("to"):
        out.append("e-mail")
    if a["telegram"].get("enabled") and a["telegram"].get("bot_token") and a["telegram"].get("chat_id"):
        out.append("Telegram")
    return out


def send_alert(inc, setup: dict, on_result=None, background: bool = True) -> None:
    """Deliver an incident on every configured channel. on_result(channel, ok, message) per channel."""
    a = setup["alerts"]
    text = alert_text(inc, setup)
    subject = f"[{(inc.severity or 'alert').upper()}] Collision at {setup['junction'].get('name') or 'junction'}"

    def report(ch, ok, msg=""):
        (log.info if ok else log.error)("alert via %s: %s %s", ch, "sent" if ok else "FAILED", msg)
        if on_result:
            on_result(ch, ok, msg)

    def run():
        for ch in channels(setup):
            try:
                if ch == "webhook":
                    _post_json(a["webhook"], {**asdict(inc), "message": text})
                elif ch == "e-mail":
                    _email(a["email"], subject, text, inc.snapshot_path)
                else:
                    _telegram(a["telegram"], text, inc.snapshot_path)
                report(ch, True)
            except Exception as e:  # never let a failed channel stop the others
                report(ch, False, str(e))

    if background:
        threading.Thread(target=run, daemon=True).start()
    else:
        run()
