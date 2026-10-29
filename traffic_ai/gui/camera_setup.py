"""Camera setup page: junction location, cameras per approach (with road width), signal timing,
the link to the signal controller, and who gets collision alerts. Saved to configs/junction.json."""
from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPointF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

from traffic_ai.config import CONFIG_DIR
from traffic_ai.gui.common import ModelPicker, bgr_to_qimage, hint, page_header, panel
from traffic_ai.live.setup import APPROACH_NAMES, CAMERA_DEFAULT, lanes_of, load_setup, save_setup

APPROACH_CHOICES = [("N", "North approach"), ("E", "East approach"), ("S", "South approach"),
                    ("W", "West approach"), ("multi", "Whole junction (camera file with N/E/S/W areas)")]


class GrabWorker(QThread):
    got = Signal(object)

    def __init__(self, source):
        super().__init__()
        self.source = source

    def run(self):
        from traffic_ai.live.engine import grab_one
        self.got.emit(grab_one(self.source))


class AreaCanvas(QLabel):
    """Shows a camera frame; left-click adds polygon corners, right-click removes the last one."""

    def __init__(self, frame, points):
        super().__init__()
        self.img = bgr_to_qimage(frame)
        self.points = [tuple(p) for p in (points or [])]
        self.setMinimumSize(800, 450)
        self.setCursor(Qt.CrossCursor)

    def _geom(self):
        s = min(self.width() / self.img.width(), self.height() / self.img.height())
        ox, oy = (self.width() - self.img.width() * s) / 2, (self.height() - self.img.height() * s) / 2
        return s, ox, oy

    def mousePressEvent(self, e):
        s, ox, oy = self._geom()
        x, y = (e.position().x() - ox) / s, (e.position().y() - oy) / s
        if e.button() == Qt.RightButton:
            if self.points:
                self.points.pop()
        elif 0 <= x <= self.img.width() and 0 <= y <= self.img.height():
            self.points.append((round(x), round(y)))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s, ox, oy = self._geom()
        p.fillRect(self.rect(), QColor("#1b2230"))
        p.drawPixmap(int(ox), int(oy), QPixmap.fromImage(self.img).scaled(int(self.img.width() * s),
                                                                         int(self.img.height() * s)))
        pts = [QPointF(ox + x * s, oy + y * s) for x, y in self.points]
        if pts:
            p.setPen(QPen(QColor("#00e5ff"), 2))
            p.setBrush(QColor(0, 229, 255, 50))
            p.drawPolygon(QPolygonF(pts))
            for q in pts:
                p.drawEllipse(q, 4, 4)
            if len(pts) >= 3:  # stop line = bottom edge of the area
                ys = [y for _, y in self.points]
                xs = [x for x, _ in self.points]
                p.setPen(QPen(QColor("#ff3b30"), 3))
                p.drawLine(QPointF(ox + min(xs) * s, oy + max(ys) * s), QPointF(ox + max(xs) * s, oy + max(ys) * s))


class AreaDialog(QDialog):
    def __init__(self, frame, points, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mark the approach area")
        lay = QVBoxLayout(self)
        lay.addWidget(hint("Click the corners of the road area this camera should count, covering all lanes up to "
                           "the stop line. Right-click removes the last corner. The red line (bottom edge) is used as "
                           "the stop line. The road length and width you enter must match this area."))
        self.canvas = AreaCanvas(frame, points)
        lay.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: (self.canvas.points.clear(), self.canvas.update()))
        whole = QPushButton("Use the whole frame")
        whole.clicked.connect(lambda: (self.canvas.points.clear(), self.accept()))
        row.addWidget(clear)
        row.addWidget(whole)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        row.addWidget(bb)
        lay.addLayout(row)
        self.resize(1000, 640)

    def points(self):
        return [list(p) for p in self.canvas.points] if len(self.canvas.points) >= 3 else None


class CameraDialog(QDialog):
    def __init__(self, cam: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera")
        self.cam = copy.deepcopy(cam)
        self.frame = None
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(cam["name"])
        self.approach = QComboBox()
        for k, v in APPROACH_CHOICES:
            self.approach.addItem(v, k)
        self.approach.setCurrentIndex(max(0, self.approach.findData(cam["approach"])))
        src_row = QHBoxLayout()
        self.source = QLineEdit(str(cam["source"]))
        self.source.setPlaceholderText("rtsp://user:password@192.168.1.64:554/Streaming/Channels/101")
        b_web, b_file = QPushButton("USB webcam"), QPushButton("Video file (testing)…")
        b_web.clicked.connect(lambda: self.source.setText("0"))
        b_file.clicked.connect(self.pick_file)
        src_row.addWidget(self.source, 1)
        src_row.addWidget(b_web)
        src_row.addWidget(b_file)
        self.width_m = QDoubleSpinBox()
        self.width_m.setRange(2.5, 60)
        self.width_m.setSingleStep(0.5)
        self.width_m.setSuffix(" m")
        self.width_m.setValue(float(cam.get("road_width_m", 7.0)))
        self.length_m = QDoubleSpinBox()
        self.length_m.setRange(5, 300)
        self.length_m.setSuffix(" m")
        self.length_m.setValue(float(cam.get("length_m", 40)))
        self.lanes = QSpinBox()
        self.lanes.setRange(0, 12)
        self.lanes.setSpecialValueText("Estimate from width")
        self.lanes.setValue(int(cam.get("lanes") or 0))
        self.lane_hint = hint("")
        self.cam_file = QLineEdit(cam.get("camera_file", ""))
        b_cf = QPushButton("Browse…")
        b_cf.clicked.connect(self.pick_camfile)
        cf_row = QHBoxLayout()
        cf_row.addWidget(self.cam_file, 1)
        cf_row.addWidget(b_cf)
        self.enabled = QCheckBox("Use this camera")
        self.enabled.setChecked(cam.get("enabled", True))
        form.addRow("Name", self.name)
        form.addRow("Watches", self.approach)
        form.addRow("Camera address", src_row)
        form.addRow("", hint("IP / CCTV camera: its RTSP address (from the camera's manual or app). "
                             "USB webcam: 0, 1, … A video file can stand in for a camera while testing."))
        form.addRow("Road width", self.width_m)
        form.addRow("Road length in view", self.length_m)
        form.addRow("Lanes", self.lanes)
        form.addRow("", self.lane_hint)
        form.addRow("Camera file", cf_row)
        form.addRow("", self.enabled)
        lay.addLayout(form)
        row = QHBoxLayout()
        self.test_btn = QPushButton("Test connection")
        self.area_btn = QPushButton("Mark road area…")
        self.preview = QLabel("No test yet")
        self.preview.setFixedSize(320, 180)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#1b2230; color:#9aa6b8; border-radius:6px;")
        self.area_lab = hint("")
        col = QVBoxLayout()
        col.addWidget(self.test_btn)
        col.addWidget(self.area_btn)
        col.addWidget(self.area_lab)
        col.addStretch(1)
        row.addLayout(col)
        row.addWidget(self.preview)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.test_btn.clicked.connect(self.test)
        self.area_btn.clicked.connect(self.mark_area)
        self.approach.currentIndexChanged.connect(self._refresh)
        self.width_m.valueChanged.connect(self._refresh)
        self.lanes.valueChanged.connect(self._refresh)
        self._refresh()

    def _refresh(self):
        multi = self.approach.currentData() == "multi"
        self.cam_file.setEnabled(multi)
        self.area_btn.setEnabled(not multi)
        n = self.lanes.value() or max(1, round(self.width_m.value() / 3.5))
        self.lane_hint.setText(f"Density is measured over {self.width_m.value():.1f} m of road ({n} lane(s) "
                               f"of capacity): PCU per km per lane and the share of road area occupied.")
        roi = self.cam.get("roi")
        self.area_lab.setText(f"Road area: {len(roi)} corners" if roi else "Road area: whole frame")

    def pick_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Video file", "", "Videos (*.mp4 *.avi *.mov *.mkv);;All (*)")
        if f:
            self.source.setText(f)

    def pick_camfile(self):
        f, _ = QFileDialog.getOpenFileName(self, "Camera file", str(CONFIG_DIR), "Camera file (*.json)")
        if f:
            self.cam_file.setText(f)

    def test(self, then=None):
        self.test_btn.setEnabled(False)
        self.preview.setText("Connecting…")
        self._grab = GrabWorker(self.source.text())

        def done(frame):
            self.test_btn.setEnabled(True)
            self.frame = frame
            if frame is None:
                self.preview.setText("No picture. Check the address,\nnetwork and camera password.")
                return
            self.preview.setPixmap(QPixmap.fromImage(bgr_to_qimage(frame)).scaled(
                320, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if then:
                then()
        self._grab.got.connect(done)
        self._grab.start()

    def mark_area(self):
        if self.frame is None:
            return self.test(then=self.mark_area)
        d = AreaDialog(self.frame, self.cam.get("roi"), self)
        if d.exec():
            self.cam["roi"] = d.points()
            self.cam["stop_line"] = None
            self._refresh()

    def result_cam(self) -> dict:
        self.cam.update(name=self.name.text().strip() or "Camera", approach=self.approach.currentData(),
                        source=self.source.text().strip(), road_width_m=self.width_m.value(),
                        length_m=self.length_m.value(), lanes=self.lanes.value(),
                        camera_file=self.cam_file.text().strip(), enabled=self.enabled.isChecked())
        return self.cam


def _spin(lo, hi, val, suffix=" s", step=1):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(int(val))
    w.setSuffix(suffix)
    w.setSingleStep(step)
    return w


class CameraSetupPage(QWidget):
    def __init__(self, status):
        super().__init__()
        self.setObjectName("page")
        self.status = status
        self.setup = load_setup()
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        head = QHBoxLayout()
        head.addWidget(page_header("Camera setup",
                                   "Connect the junction's cameras, set road widths and signal timing, and choose "
                                   "who is alerted about collisions."), 1)
        self.save_btn = QPushButton("Save setup")
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self.save)
        head.addWidget(self.save_btn, 0, Qt.AlignBottom)
        root.addLayout(head)
        tabs = QTabWidget()
        tabs.addTab(self._junction_tab(), "Junction and cameras")
        tabs.addTab(self._signal_tab(), "Signal timing")
        tabs.addTab(self._output_tab(), "Signal controller link")
        tabs.addTab(self._alert_tab(), "Collision alerts")
        root.addWidget(tabs, 1)

    # ------------------------------------------------------------------ junction + cameras
    def _junction_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        jp, jl = panel("Junction location (sent with every alert)")
        form = QFormLayout()
        j = self.setup["junction"]
        self.j_name = QLineEdit(j.get("name", ""))
        self.j_addr = QLineEdit(j.get("address", ""))
        self.j_addr.setPlaceholderText("e.g. Master Canteen Square, Bhubaneswar, Odisha")
        self.j_lat, self.j_lon = QLineEdit(str(j.get("lat") or "")), QLineEdit(str(j.get("lon") or ""))
        self.j_lat.setPlaceholderText("20.2961")
        self.j_lon.setPlaceholderText("85.8245")
        ll = QHBoxLayout()
        ll.addWidget(QLabel("Latitude"))
        ll.addWidget(self.j_lat)
        ll.addWidget(QLabel("Longitude"))
        ll.addWidget(self.j_lon)
        ll.addWidget(hint("Right-click the junction in Google Maps to copy these."))
        form.addRow("Junction name", self.j_name)
        form.addRow("Address", self.j_addr)
        form.addRow("GPS", ll)
        jl.addLayout(form)
        lay.addWidget(jp)
        cp, cl = panel("Cameras")
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Name", "Watches", "Camera address", "Road width", "Area", "In use"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.doubleClicked.connect(self.edit_cam)
        cl.addWidget(self.table)
        row = QHBoxLayout()
        for text, fn in (("Add camera", self.add_cam), ("Edit", self.edit_cam), ("Remove", self.remove_cam)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        cl.addLayout(row)
        cl.addWidget(hint("Usual setup: one camera per approach, mounted on the signal pole facing the incoming "
                          "traffic. One high overhead camera can cover the whole junction instead (Watches: whole "
                          "junction, with a camera file such as configs\\camera_example.json)."))
        lay.addWidget(cp, 1)
        ap, al = panel("Detection")
        f2 = QFormLayout()
        self.model = ModelPicker()
        cur = self.setup["analysis"].get("model")
        if cur:
            i = self.model.findData(cur)
            if i < 0:
                self.model.insertItem(0, f"Custom: {Path(cur).name}", cur)
                i = 0
            self.model.setCurrentIndex(i)
        self.a_fps = _spin(1, 15, self.setup["analysis"]["analysis_fps"], " frames/s per camera")
        self.a_imgsz = QComboBox()
        for s in (480, 640, 960):
            self.a_imgsz.addItem(f"{s} px", s)
        self.a_imgsz.setCurrentIndex(max(0, self.a_imgsz.findData(int(self.setup["analysis"]["imgsz"]))))
        f2.addRow("Model", self.model)
        f2.addRow("Analyse", self.a_fps)
        f2.addRow("Detector size", self.a_imgsz)
        al.addLayout(f2)
        al.addWidget(hint("On a laptop without a GPU, 4 cameras at 3-5 frames/s and 640 px is a good balance."))
        lay.addWidget(ap)
        self._fill_table()
        return w

    def _fill_table(self):
        cams = self.setup["cameras"]
        self.table.setRowCount(len(cams))
        for r, c in enumerate(cams):
            watch = "Whole junction" if c["approach"] == "multi" else APPROACH_NAMES[c["approach"]]
            vals = [c["name"], watch, str(c["source"]), f"{c['road_width_m']:.1f} m ({lanes_of(c)} lanes)",
                    "marked" if c.get("roi") else "whole frame", "yes" if c.get("enabled", True) else "no"]
            for k, v in enumerate(vals):
                self.table.setItem(r, k, QTableWidgetItem(v))

    def add_cam(self):
        used = {c["approach"] for c in self.setup["cameras"]}
        free = next((a for a in "NESW" if a not in used), "N")
        cam = dict(copy.deepcopy(CAMERA_DEFAULT), approach=free, name=f"{APPROACH_NAMES[free]} camera")
        d = CameraDialog(cam, self)
        if d.exec():
            self.setup["cameras"].append(d.result_cam())
            self._fill_table()

    def edit_cam(self, *_):
        r = self.table.currentRow()
        if r < 0:
            return
        d = CameraDialog(self.setup["cameras"][r], self)
        if d.exec():
            self.setup["cameras"][r] = d.result_cam()
            self._fill_table()

    def remove_cam(self):
        r = self.table.currentRow()
        if r >= 0 and QMessageBox.question(self, "Remove camera",
                                           f"Remove {self.setup['cameras'][r]['name']}?") == QMessageBox.Yes:
            self.setup["cameras"].pop(r)
            self._fill_table()

    # ------------------------------------------------------------------ signal timing
    def _signal_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        s = self.setup["signal"]
        p, pl = panel("How green time is decided")
        form = QFormLayout()
        self.s_control = QComboBox()
        self.s_control.addItem("Adaptive: green time from live traffic density", "adaptive")
        self.s_control.addItem("Fixed-time (for comparison)", "fixed")
        self.s_control.setCurrentIndex(self.s_control.findData(s["control"]))
        self.s_phasing = QComboBox()
        self.s_phasing.addItem("One approach at a time (N → E → S → W)", "split")
        self.s_phasing.addItem("Opposite approaches together (N+S, then E+W)", "ns_ew")
        self.s_phasing.setCurrentIndex(self.s_phasing.findData(s["phasing"]))
        self.s_min, self.s_max = _spin(5, 60, s["min_green"]), _spin(15, 180, s["max_green"])
        self.s_fixed = _spin(10, 120, s["fixed_green"])
        self.s_yellow, self.s_red = _spin(2, 6, s["yellow"]), _spin(0, 6, s["all_red"])
        self.s_head = QDoubleSpinBox()
        self.s_head.setRange(1.0, 4.0)
        self.s_head.setSingleStep(0.1)
        self.s_head.setValue(float(s["sat_headway_s"]))
        self.s_head.setSuffix(" s per PCU per lane")
        self.s_ext = _spin(2, 15, s["extend_step_s"])
        self.s_skip = QCheckBox("Skip an approach when its camera sees no vehicle")
        self.s_skip.setChecked(bool(s["skip_empty"]))
        for lab, wid in (("Control", self.s_control), ("Phases", self.s_phasing), ("Minimum green", self.s_min),
                         ("Maximum green", self.s_max), ("Fixed green (fixed-time / camera offline)", self.s_fixed),
                         ("Yellow", self.s_yellow), ("All-red clearance", self.s_red),
                         ("Saturation headway", self.s_head), ("Extend green by", self.s_ext), ("", self.s_skip)):
            form.addRow(lab, wid)
        pl.addLayout(form)
        pl.addWidget(hint("Adaptive green = 4 s start-up + headway × (queued PCU ÷ lanes) + half a headway per moving "
                          "PCU, kept between minimum and maximum green, and extended while the queue is still "
                          "discharging. Lanes come from each camera's road width. If a camera goes offline, its "
                          "approach falls back to the fixed green."))
        lay.addWidget(p)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------ controller link
    def _output_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        o = self.setup["controller_output"]
        p, pl = panel("Send signal states and density to the junction controller")
        form = QFormLayout()
        self.o_type = QComboBox()
        for k, v in (("none", "Only show in this app"), ("serial", "ESP32 / Arduino signal model over USB (serial)"),
                     ("http", "HTTP: POST JSON to a controller or traffic-management server"),
                     ("udp", "UDP: JSON datagram to a controller on the network")):
            self.o_type.addItem(v, k)
        self.o_type.setCurrentIndex(self.o_type.findData(o["type"]))
        self.o_port = QLineEdit(o["serial_port"])
        self.o_url = QLineEdit(o["url"])
        self.o_url.setPlaceholderText("http://192.168.1.50:8080/signal")
        self.o_host = QLineEdit(o["udp_host"])
        self.o_udp = _spin(1, 65535, o["udp_port"], "")
        for lab, wid in (("Send to", self.o_type), ("Serial port", self.o_port), ("URL", self.o_url),
                         ("UDP host", self.o_host), ("UDP port", self.o_udp)):
            form.addRow(lab, wid)
        pl.addLayout(form)
        pl.addWidget(hint('Serial sends "GRRR" style lines (N E S W, G/Y/R) that hardware\\esp32_signal reads. HTTP and '
                          "UDP send the lights, remaining time, next green times and per-approach density as JSON, "
                          "on every change and once a second. Research prototype: connect it only to a demo model "
                          "or a test controller, never to a live junction without the road authority's approval."))
        lay.addWidget(p)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------ alerts
    def _alert_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        a = self.setup["alerts"]
        p, pl = panel("When a collision is detected")
        form = QFormLayout()
        self.al_min = QComboBox()
        self.al_min.addItem("Severe collisions only", "severe")
        self.al_min.addItem("Moderate and severe collisions", "moderate")
        self.al_min.addItem("Every collision", "minor")
        self.al_min.setCurrentIndex(self.al_min.findData(a["min_severity"]))
        self.al_sound = QCheckBox("Sound an alarm in the app")
        self.al_sound.setChecked(a["sound"])
        self.al_hook = QLineEdit(a["webhook"])
        self.al_hook.setPlaceholderText("https://control-room.example/api/incidents (optional)")
        form.addRow("Notify the traffic controller for", self.al_min)
        form.addRow("", self.al_sound)
        form.addRow("Webhook", self.al_hook)
        pl.addLayout(form)
        lay.addWidget(p)
        e = a["email"]
        ep, el = panel("E-mail")
        f2 = QFormLayout()
        self.em_on = QCheckBox("Send e-mail alerts with the snapshot attached")
        self.em_on.setChecked(e["enabled"])
        self.em_to = QLineEdit(e["to"])
        self.em_to.setPlaceholderText("controlroom@city.gov.in, officer@example.com")
        self.em_host, self.em_port = QLineEdit(e["smtp_host"]), _spin(1, 65535, e["smtp_port"], "")
        self.em_user, self.em_pass = QLineEdit(e["username"]), QLineEdit(e["password"])
        self.em_pass.setEchoMode(QLineEdit.Password)
        self.em_from = QLineEdit(e["from"])
        for lab, wid in (("", self.em_on), ("Send to", self.em_to), ("SMTP server", self.em_host),
                         ("Port", self.em_port), ("Username", self.em_user), ("Password", self.em_pass),
                         ("From", self.em_from)):
            f2.addRow(lab, wid)
        el.addLayout(f2)
        el.addWidget(hint("Gmail: smtp.gmail.com, port 587, your address as username and an App Password "
                          "(Google account → Security → App passwords) as password. Stored in configs\\junction.json "
                          "on this PC."))
        lay.addWidget(ep)
        t = a["telegram"]
        tp, tl = panel("Telegram")
        f3 = QFormLayout()
        self.tg_on = QCheckBox("Send Telegram alerts with the snapshot")
        self.tg_on.setChecked(t["enabled"])
        self.tg_token, self.tg_chat = QLineEdit(t["bot_token"]), QLineEdit(str(t["chat_id"]))
        self.tg_token.setEchoMode(QLineEdit.Password)
        for lab, wid in (("", self.tg_on), ("Bot token", self.tg_token), ("Chat ID", self.tg_chat)):
            f3.addRow(lab, wid)
        tl.addLayout(f3)
        tl.addWidget(hint("Create a bot with @BotFather to get the token; add it to the control-room group and use "
                          "the group's chat ID."))
        lay.addWidget(tp)
        row = QHBoxLayout()
        b = QPushButton("Send a test alert")
        b.clicked.connect(self.test_alert)
        row.addWidget(b)
        self.test_lab = hint("")
        row.addWidget(self.test_lab, 1)
        lay.addLayout(row)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------ save / test
    def collect(self) -> dict:
        s = self.setup

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None
        s["junction"].update(name=self.j_name.text().strip(), address=self.j_addr.text().strip(),
                             lat=num(self.j_lat.text()), lon=num(self.j_lon.text()))
        s["analysis"].update(model=self.model.weights(), analysis_fps=self.a_fps.value(),
                             imgsz=self.a_imgsz.currentData())
        s["signal"].update(control=self.s_control.currentData(), phasing=self.s_phasing.currentData(),
                           min_green=self.s_min.value(), max_green=max(self.s_max.value(), self.s_min.value() + 5),
                           fixed_green=self.s_fixed.value(), yellow=self.s_yellow.value(),
                           all_red=self.s_red.value(), sat_headway_s=self.s_head.value(),
                           extend_step_s=self.s_ext.value(), skip_empty=self.s_skip.isChecked())
        s["controller_output"].update(type=self.o_type.currentData(), serial_port=self.o_port.text().strip(),
                                      url=self.o_url.text().strip(), udp_host=self.o_host.text().strip(),
                                      udp_port=self.o_udp.value())
        s["alerts"].update(min_severity=self.al_min.currentData(), sound=self.al_sound.isChecked(),
                           webhook=self.al_hook.text().strip())
        s["alerts"]["email"].update(enabled=self.em_on.isChecked(), to=self.em_to.text().strip(),
                                    smtp_host=self.em_host.text().strip(), smtp_port=self.em_port.value(),
                                    username=self.em_user.text().strip(), password=self.em_pass.text(),
                                    **{"from": self.em_from.text().strip()})
        s["alerts"]["telegram"].update(enabled=self.tg_on.isChecked(), bot_token=self.tg_token.text().strip(),
                                       chat_id=self.tg_chat.text().strip())
        return s

    def save(self):
        save_setup(self.collect())
        self.status.showMessage("Setup saved. Open Live control and press Start.")

    def test_alert(self):
        from traffic_ai.incident.alert import Incident
        from traffic_ai.incident.notify import channels, send_alert
        from traffic_ai.live.setup import location_text, maps_url
        s = self.collect()
        chans = channels(s)
        if not chans:
            self.test_lab.setText("No alert channel is set up yet.")
            return
        inc = Incident("TEST-ALERT", location_text(s, "N"), "N", 0.0, datetime.now().isoformat(timespec="seconds"),
                       0.9, 0.9, "confirmed", ["test"], [{"class": "Car", "speed_kmh": 0.0}], None, None,
                       "Test alert", "severe", 0.9, ["this is a test message"], 40.0, "Test camera", maps_url(s))
        self.test_lab.setText(f"Sending via {', '.join(chans)}…")
        results = []

        def done(ch, ok, msg):
            results.append(f"{ch}: {'sent ✓' if ok else 'failed: ' + msg}")
            self._test_text = "   ".join(results)
        self._test_text = ""
        send_alert(inc, s, on_result=done)
        from PySide6.QtCore import QTimer

        def poll(n=0):
            if self._test_text:
                self.test_lab.setText(self._test_text)
            if len(results) < len(chans) and n < 60:
                QTimer.singleShot(500, lambda: poll(n + 1))
        poll()
