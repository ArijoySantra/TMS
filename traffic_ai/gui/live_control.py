"""Live control page: camera feeds, density per approach, the adaptive signal plan, and collision alerts."""
from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout,
                               QHeaderView, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from traffic_ai.gui.common import APPROACH_COLORS, OUTPUT_DIR, bgr_to_qimage, hint, open_folder, page_header, panel
from traffic_ai.gui.detect import Preview
from traffic_ai.live.setup import APPROACH_NAMES, load_setup

LAMP = {"G": "#1fbf5c", "Y": "#f5b800", "R": "#e5484d"}
SEV_COLOR = {"severe": "#d92d20", "moderate": "#e88a00", "minor": "#5f6b7a"}
LEVEL_COLOR = {"Low": "#1baf7a", "Medium": "#eda100", "High": "#eb6834", "Jammed": "#d92d20"}


class SignalBoard(QWidget):
    """The junction from above with a signal head on each approach, the countdown and density level."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.status, self.approaches = {}, {}
        self.blink = False

    def set(self, status, approaches):
        self.status, self.approaches = status or {}, approaches or {}
        self.blink = not self.blink
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        s = min(w, h)
        cx, cy = w / 2, h / 2
        road = s * 0.22
        p.fillRect(self.rect(), QColor("#e9efe4"))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#5a5f68"))
        p.drawRect(QRectF(cx - road / 2, 0, road, h))
        p.drawRect(QRectF(0, cy - road / 2, w, road))
        lights = self.status.get("lights", {})
        stage = self.status.get("stage", "")
        rem = self.status.get("remaining_s", 0.0)
        f = QFont(self.font())
        hw, hh, gap = s * 0.075, s * 0.19, 8
        # one head per corner, next to the road its traffic arrives on
        pos = {"N": (cx - road / 2 - hw - gap, cy - road / 2 - hh - 40),
               "E": (cx + road / 2 + gap, cy - road / 2 - hh - 40),
               "S": (cx + road / 2 + gap, cy + road / 2 + gap),
               "W": (cx - road / 2 - hw - gap, cy + road / 2 + gap)}
        for a, (x, y) in pos.items():
            col = lights.get(a)
            p.setBrush(QColor("#1d2433"))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(x, y, hw, hh), 6, 6)
            for k, c in enumerate("RYG"):
                on = col == c and not (stage == "starting" and self.blink)
                p.setBrush(QColor(LAMP[c]) if on else QColor("#3a4150"))
                r = hw * 0.34
                p.drawEllipse(QRectF(x + hw / 2 - r, y + hh * (0.18 + 0.32 * k) - r, 2 * r, 2 * r))
            f.setBold(True)
            f.setPointSize(9)
            p.setFont(f)
            p.setPen(QColor(APPROACH_COLORS[a]))
            p.drawText(QRectF(x - 40, y + hh + 2, hw + 80, 16), Qt.AlignCenter, APPROACH_NAMES[a])
            d = self.approaches.get(a)
            f.setBold(False)
            f.setPointSize(8)
            p.setFont(f)
            if d:
                p.setPen(QColor(LEVEL_COLOR.get(d.get("level"), "#1d2433")))
                p.drawText(QRectF(x - 40, y + hh + 17, hw + 80, 16), Qt.AlignCenter,
                           f"{d.get('level', '')} · {d.get('occupancy_pct', 0):.0f}%")
            elif a in self.status.get("next_plan", {}):
                p.setPen(QColor("#8a94a3"))
                p.drawText(QRectF(x - 40, y + hh + 17, hw + 80, 16), Qt.AlignCenter, "no camera data")
        if stage in ("green", "yellow"):  # countdown in the middle of the junction
            col = LAMP["G" if stage == "green" else "Y"]
            f.setBold(True)
            f.setPointSize(max(10, int(road / 4.5)))
            p.setFont(f)
            p.setPen(QColor(col))
            p.drawText(QRectF(cx - road / 2, cy - road / 2, road, road), Qt.AlignCenter, f"{rem:.0f}")


class CameraTile(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 8)
        self.title = QLabel("Camera")
        self.title.setObjectName("sectionTitle")
        lay.addWidget(self.title)
        self.view = Preview("Waiting for picture…")
        self.view.setMinimumSize(240, 150)
        lay.addWidget(self.view, 1)


def _table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    return t


class LiveControlPage(QWidget):
    def __init__(self, status, go=None):
        super().__init__()
        self.setObjectName("page")
        self.status, self.go = status, go
        self.engine = None
        self.tiles: dict[int, CameraTile] = {}
        self.unacked = []
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        head = QHBoxLayout()
        head.addWidget(page_header(
            "Live control",
            "Cameras measure traffic density on every approach, the signal plan adapts green times to it, and "
            "collisions are detected, rated by severity and reported to the traffic controller."), 1)
        self.mode = QComboBox()
        self.mode.addItem("Adaptive timing", "adaptive")
        self.mode.addItem("Fixed-time", "fixed")
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        b_setup = QPushButton("Camera setup")
        b_logs = QPushButton("Logs and incidents")
        for wdg in (self.mode, self.start_btn, self.stop_btn, b_setup, b_logs):
            head.addWidget(wdg, 0, Qt.AlignBottom)
        root.addLayout(head)

        self.banner = QFrame()
        self.banner.setStyleSheet("background:#d92d20; border-radius:8px;")
        bl = QHBoxLayout(self.banner)
        self.banner_text = QLabel()
        self.banner_text.setStyleSheet("color:white; font-weight:600; font-size:11pt;")
        self.banner_text.setWordWrap(True)
        ack = QPushButton("Acknowledge")
        ack.clicked.connect(self.acknowledge)
        bl.addWidget(self.banner_text, 1)
        bl.addWidget(ack)
        self.banner.hide()
        root.addWidget(self.banner)

        body = QHBoxLayout()
        body.setSpacing(14)
        self.grid_holder = QWidget()
        self.grid = QGridLayout(self.grid_holder)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(10)
        self.empty = QFrame()
        self.empty.setObjectName("panel")
        el = QVBoxLayout(self.empty)
        el.addStretch(1)
        lab = QLabel("No cameras connected yet")
        lab.setObjectName("pageTitle")
        lab.setAlignment(Qt.AlignCenter)
        el.addWidget(lab)
        h = hint("Open Camera setup, add one camera per approach (IP/RTSP camera, CCTV recorder channel or USB "
                 "webcam), enter the road width it covers, save, then press Start.")
        h.setAlignment(Qt.AlignCenter)
        el.addWidget(h)
        b = QPushButton("Open camera setup")
        b.clicked.connect(lambda: self.go and self.go("Camera setup"))
        el.addWidget(b, 0, Qt.AlignCenter)
        el.addStretch(1)
        self.grid.addWidget(self.empty, 0, 0)
        body.addWidget(self.grid_holder, 6)

        side = QVBoxLayout()
        sp, sl = panel("Signals")
        self.board = SignalBoard()
        self.board.setMinimumHeight(260)
        sl.addWidget(self.board, 1)
        self.sig_text = QLabel("Stopped")
        self.sig_text.setWordWrap(True)
        self.sig_text.setMinimumHeight(36)
        sl.addWidget(self.sig_text)
        side.addWidget(sp, 4)
        dp, dl = panel("Density and next green times")
        self.table = _table(["Approach", "Vehicles", "PCU", "Queue", "Road used", "Level", "Next green"])
        self.table.setMinimumHeight(150)
        dl.addWidget(self.table)
        self.link = hint("")
        dl.addWidget(self.link)
        side.addWidget(dp, 2)
        ip, il = panel("Collisions")
        self.inc_list = QListWidget()
        self.inc_list.setWordWrap(True)
        il.addWidget(self.inc_list)
        side.addWidget(ip, 2)
        body.addLayout(side, 5)
        root.addLayout(body, 1)

        self.timer = QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.refresh)
        self.beeper = QTimer(self)
        self.beeper.setInterval(1500)
        self.beeper.timeout.connect(QApplication.beep)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.mode.currentIndexChanged.connect(self.change_mode)
        b_setup.clicked.connect(lambda: self.go and self.go("Camera setup"))
        b_logs.clicked.connect(lambda: open_folder(Path(self.engine.out) if self.engine else OUTPUT_DIR / "live"))
        self.tick = 0

    # ------------------------------------------------------------------ lifecycle
    def _build_tiles(self, cams):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w and w is not self.empty:
                w.deleteLater()
        self.empty.hide()
        self.tiles = {}
        n = len(cams)
        cols = 1 if n == 1 else 2
        for i in range(n):
            t = CameraTile()
            self.grid.addWidget(t, i // cols, i % cols)
            self.tiles[i] = t

    def start(self):
        setup = load_setup()
        setup["signal"]["control"] = self.mode.currentData()
        cams = [c for c in setup["cameras"] if c.get("enabled", True)]
        if not cams:
            QMessageBox.information(self, "Live control", "Add at least one camera in Camera setup first.")
            if self.go:
                self.go("Camera setup")
            return
        try:
            from traffic_ai.live.engine import LiveEngine
            self.engine = LiveEngine(setup)
            self.engine.start()
        except Exception as e:
            QMessageBox.critical(self, "Live control", f"{e}\n\n{traceback.format_exc()}"[:3000])
            self.engine = None
            return
        self.setup = setup
        self._build_tiles(cams)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start()
        self.status.showMessage(f"Live control started with {len(cams)} camera(s). Loading the detector…")

    def stop(self):
        if self.engine:
            eng, self.engine = self.engine, None
            self.timer.stop()
            eng.stop()
            self.status.showMessage(f"Stopped. Logs: {eng.out}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.sig_text.setText("Stopped")
        self.board.set({}, {})

    def change_mode(self):
        if self.engine:
            self.engine.signal.cfg["control"] = self.mode.currentData()
            self.status.showMessage(f"Signal control switched to {self.mode.currentText().lower()}; "
                                    "it applies from the next green.")

    def shutdown(self):
        self.stop()

    # ------------------------------------------------------------------ refresh
    def refresh(self):
        e = self.engine
        if e is None:
            return
        self.tick += 1
        snap = e.snapshot()
        for i, c in snap["cameras"].items():
            t = self.tiles.get(i)
            if t is None:
                continue
            state = {"live": "● live", "connecting": "connecting…", "offline": "✕ offline"}.get(c["status"], c["status"])
            extra = f"  ·  analysing {c['proc_fps']:.1f} fps" if c["proc_fps"] else ""
            err = f"  ·  {c['error'][:60]}" if c["error"] else ""
            watch = "whole junction" if c["approach"] == "multi" else APPROACH_NAMES.get(c["approach"], "")
            t.title.setText(f"{c['name']} ({watch})  ·  {state}{extra}{err}")
            t.title.setStyleSheet(f"color: {'#1d2433' if c['status'] == 'live' else '#d92d20'};")
            if c["frame"] is not None and self.tick % 2 == 0:
                t.view.set_image(bgr_to_qimage(c["frame"]))
        sig, apps = snap["signal"], snap["approaches"]
        if self.tick % 2 == 0:
            self.board.set(sig, apps)
        if sig:
            if sig.get("stage") == "starting":
                self.sig_text.setText(sig["reason"])
            else:
                stage = {"green": "green", "yellow": "yellow", "all_red": "all red"}[sig["stage"]]
                names = " + ".join(APPROACH_NAMES[a] for a in sig["phase"].split("+"))
                self.sig_text.setText(f"<b>{names}</b>: {stage}, {sig['remaining_s']:.0f} s left · {sig['reason']}"
                                      f"<br>{'Adaptive' if sig['control'] == 'adaptive' else 'Fixed-time'} cycle "
                                      f"≈ {sig['cycle_s']:.0f} s")
            rows = []
            for a in "NESW":
                if a not in sig.get("next_plan", {}) and a not in apps:
                    continue
                d = apps.get(a)
                g = sig.get("next_plan", {}).get(a)
                if d:
                    rows.append((APPROACH_NAMES[a], str(d["vehicles"]), f"{d['pcu']:.1f}",
                                 f"{d['queue_veh']}", f"{d['occupancy_pct']:.0f}%",
                                 d["level"], "–" if g is None else f"{g:.0f} s"))
                else:
                    rows.append((APPROACH_NAMES[a], "–", "–", "–", "–", "no data", "–" if g is None else f"{g:.0f} s"))
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for k, v in enumerate(row):
                    it = QTableWidgetItem(v)
                    if k == 5:
                        it.setForeground(QColor(LEVEL_COLOR.get(v, "#8a94a3")))
                    self.table.setItem(r, k, it)
        out, err = snap["output"]
        self.link.setText(f"Signal controller link: {out}" + (f" ({err})" if err else ""))
        for ev in e.events():
            self.on_event(ev)

    def on_event(self, ev):
        if ev[0] == "incident":
            inc, notify = ev[1], ev[2]
            sev = inc.severity or "minor"
            icon = "🚨" if sev == "severe" else "⚠️" if sev == "moderate" else "•"
            it = QListWidgetItem(f"{icon} {sev.upper()}  {inc.wall_time[11:19]}  {inc.location}\n{inc.summary}")
            it.setForeground(QColor(SEV_COLOR[sev]))
            it.setData(Qt.UserRole, inc.incident_id)
            self.inc_list.insertItem(0, it)
            if notify:
                self.unacked.append(inc)
                self.banner_text.setText(f"{sev.upper()} COLLISION · {inc.location} · {inc.wall_time[11:19]} · "
                                         f"camera {inc.camera}. Notifying the traffic controller…")
                self.banner.show()
                if self.setup["alerts"].get("sound", True):
                    QApplication.beep()
                    self.beeper.start()
        elif ev[0] == "delivery":
            _, iid, ch, ok, msg = ev
            self.inc_list.insertItem(0, QListWidgetItem(f"   ↳ {iid}: alert {'sent' if ok else 'FAILED'} via {ch}"
                                                        + ("" if ok else f" ({msg[:80]})")))
            if self.banner.isVisible():
                self.banner_text.setText(self.banner_text.text().replace("Notifying the traffic controller…", "")
                                         + f" {ch}: {'sent ✓' if ok else 'failed ✕'}")

    def acknowledge(self):
        self.unacked.clear()
        self.banner.hide()
        self.beeper.stop()
