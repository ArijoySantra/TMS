"""Vehicle detection on images, and the full camera pipeline on video / webcam."""
from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from traffic_ai.config import CONFIG_DIR, load_config
from traffic_ai.gui.common import (OUTPUT_DIR, Card, ModelPicker, bgr_to_qimage, hint,
                                   open_folder, page_header, panel, sample_image_dirs)

IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.mpg *.wmv);;All files (*)"


class Preview(QLabel):
    """Keeps the aspect ratio of the image it shows when the window is resized."""

    def __init__(self, placeholder: str):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #1b2230; color: #9aa6b8; border-radius: 6px;")
        self._pix = None

    def set_image(self, img):
        self._pix = QPixmap.fromImage(img)
        self._rescale()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


def _table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    return t


def _fill(table, rows):
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            table.setItem(r, c, QTableWidgetItem(str(v)))


# --------------------------------------------------------------------------------------------- images
class ImageWorker(QThread):
    result = Signal(str, object, dict)  # path, QImage, counts
    done = Signal(int)
    failed = Signal(str)

    def __init__(self, paths, weights, conf, imgsz):
        super().__init__()
        self.paths, self.weights, self.conf, self.imgsz = paths, weights, conf, imgsz
        self._stop = False
        self.n = 0

    def run(self):
        try:
            from traffic_ai.vision.pipeline import run_images
            cfg = load_config()
            cfg["vision"].update(model=self.weights, conf=self.conf, imgsz=self.imgsz)

            def on_image(p, img, counts):
                self.n += 1
                self.result.emit(str(p), bgr_to_qimage(img), {k: int(v) for k, v in counts.items()})
                return not self._stop

            run_images(self.paths, cfg, out_dir=str(OUTPUT_DIR / "images"), on_image=on_image)
            self.done.emit(self.n)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class ImagePage(QWidget):
    def __init__(self, status):
        super().__init__()
        self.setObjectName("page")
        self.status = status
        self.worker = None
        self.paths: list[Path] = []
        self.results: dict[str, tuple] = {}
        self.totals: dict[str, int] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.addWidget(page_header(
            "Photo detection",
            "Detect and count road users in photos. The default model is the YOLO26s fine-tuned on your UVH-26 "
            "images, which knows Indian classes such as two-wheelers and auto-rickshaws."))
        row = QHBoxLayout()
        b_files, b_folder = QPushButton("Open images…"), QPushButton("Open folder…")
        self.b_sample = QPushButton("Use sample images")
        self.b_sample.setEnabled(bool(sample_image_dirs()))
        self.model = ModelPicker()
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.05, 0.95)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.30)
        self.conf.setPrefix("Confidence ")
        self.imgsz = QComboBox()
        for s in (640, 960, 1280):
            self.imgsz.addItem(f"{s} px", s)
        self.run_btn = QPushButton("Detect")
        self.run_btn.setObjectName("primary")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        b_out = QPushButton("Open output folder")
        for w in (b_files, b_folder, self.b_sample):
            row.addWidget(w)
        row.addSpacing(10)
        row.addWidget(QLabel("Model"))
        row.addWidget(self.model, 1)
        for w in (self.conf, self.imgsz, self.run_btn, self.stop_btn, b_out):
            row.addWidget(w)
        root.addLayout(row)
        self.sel_lab = hint("No images chosen yet.")
        root.addWidget(self.sel_lab)

        body = QHBoxLayout()
        lp, ll = panel("Results")
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ListMode)
        self.list.setIconSize(QSize(64, 40))
        self.list.setMinimumWidth(220)
        ll.addWidget(self.list)
        body.addWidget(lp, 2)
        pp, pl = panel()
        self.preview = Preview("Choose images, then press Detect")
        pl.addWidget(self.preview)
        body.addWidget(pp, 6)
        side = QVBoxLayout()
        self.c_img = Card("Road users in this image")
        self.c_all = Card("Total over all images")
        side.addWidget(self.c_img)
        side.addWidget(self.c_all)
        tp, tl = panel("Counts by class")
        self.table = _table(["Class", "This image", "All images"])
        tl.addWidget(self.table)
        side.addWidget(tp, 1)
        body.addLayout(side, 3)
        root.addLayout(body, 1)

        b_files.clicked.connect(self.pick_files)
        b_folder.clicked.connect(self.pick_folder)
        self.b_sample.clicked.connect(self.pick_sample)
        self.run_btn.clicked.connect(self.run)
        self.stop_btn.clicked.connect(lambda: self.worker and setattr(self.worker, "_stop", True))
        b_out.clicked.connect(lambda: open_folder(OUTPUT_DIR / "images"))
        self.list.currentItemChanged.connect(self.show_item)

    def _set_paths(self, paths):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths = sorted(Path(p) for p in paths if Path(p).suffix.lower() in exts)
        self.sel_lab.setText(f"{len(self.paths)} image(s) selected" +
                             (f" from {self.paths[0].parent}" if self.paths else ""))

    def pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose images", "", IMAGE_FILTER)
        if files:
            self._set_paths(files)

    def pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose a folder of images")
        if d:
            self._set_paths(Path(d).iterdir())

    def pick_sample(self):
        d = sample_image_dirs()[0]
        self._set_paths(sorted(d.glob("*.jpg"))[:40])

    def run(self):
        if not self.paths:
            self.pick_files()
            if not self.paths:
                return
        self.list.clear()
        self.results, self.totals = {}, {}
        self.worker = ImageWorker([str(p) for p in self.paths], self.model.weights(), self.conf.value(),
                                  self.imgsz.currentData())
        self.worker.result.connect(self.on_result)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self._finished)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.showMessage("Loading the detector (the first run takes a little longer)…")
        self.worker.start()

    def _finished(self):
        self.worker = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_result(self, path, img, counts):
        self.results[path] = (img, counts)
        for k, v in counts.items():
            self.totals[k] = self.totals.get(k, 0) + v
        it = QListWidgetItem(QIcon(QPixmap.fromImage(img).scaled(64, 40, Qt.KeepAspectRatio)),
                             f"{Path(path).name}  ·  {sum(counts.values())}")
        it.setData(Qt.UserRole, path)
        self.list.addItem(it)
        self.list.setCurrentItem(it)
        self.status.showMessage(f"Detected {len(self.results)} / {len(self.paths)} images…")

    def show_item(self, item):
        if item is None:
            return
        img, counts = self.results[item.data(Qt.UserRole)]
        self.preview.set_image(img)
        self.c_img.set(str(sum(counts.values())))
        self.c_all.set(str(sum(self.totals.values())))
        classes = sorted(self.totals, key=lambda k: -self.totals[k])
        _fill(self.table, [(k, counts.get(k, 0), self.totals[k]) for k in classes])

    def on_done(self, n):
        self.status.showMessage(f"Done: {n} images. Annotated copies and detections.csv are in "
                                f"{OUTPUT_DIR / 'images'}")

    def on_failed(self, msg):
        self.status.showMessage("Detection failed.")
        QMessageBox.critical(self, "Detection failed", msg[:3000])

    def shutdown(self):
        if self.worker:
            self.worker._stop = True
            self.worker.wait(5000)


# --------------------------------------------------------------------------------------------- video
class VideoWorker(QThread):
    frame = Signal(object, dict, dict)  # QImage, state, info
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, source, camera, weights, stride, imgsz, max_seconds, save_video, rain):
        super().__init__()
        self.args = (source, camera, weights, stride, imgsz, max_seconds, save_video, rain)
        self._stop = False

    def run(self):
        source, camera, weights, stride, imgsz, max_seconds, save_video, rain = self.args
        try:
            from traffic_ai.vision.pipeline import run_video
            cfg = load_config()
            cfg["vision"].update(model=weights, frame_stride=stride, imgsz=imgsz)

            def on_frame(img, state, info):
                inc = info.pop("last_incident")
                if inc is not None:
                    info["incident"] = {"id": inc.incident_id, "status": inc.status, "summary": inc.summary}
                self.frame.emit(bgr_to_qimage(img), {k: dict(v) for k, v in state.items()}, info)
                return not self._stop

            res = run_video(source, cfg, camera, save_video=save_video, max_seconds=max_seconds or None,
                            rain=rain, on_frame=on_frame)
            self.done.emit(res)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class VideoPage(QWidget):
    def __init__(self, status):
        super().__init__()
        self.setObjectName("page")
        self.status = status
        self.worker = None
        self.out_dir = OUTPUT_DIR / "video"
        self.seen_incidents = set()
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.addWidget(page_header(
            "Recorded video",
            "Analyse a recorded video file with the full SEE → PREDICT → DECIDE pipeline: tracking, queue and speed per "
            "approach, 5–15 minute forecasts, accident-risk scores, a recommended green and accident alerts."))
        r1 = QHBoxLayout()
        self.source = QLineEdit()
        self.source.setPlaceholderText("Video file, RTSP address, or 0 for the webcam")
        b_vid, b_cam = QPushButton("Open video…"), QPushButton("Use webcam")
        self.camera = QComboBox()
        self.camera.addItem("Whole frame (no calibration)", None)
        for p in sorted(CONFIG_DIR.glob("camera*.json")):
            self.camera.addItem(f"Camera file: {p.name}", str(p))
        self.camera.addItem("Other camera file…", "__browse__")
        r1.addWidget(QLabel("Source"))
        r1.addWidget(self.source, 3)
        r1.addWidget(b_vid)
        r1.addWidget(b_cam)
        r1.addSpacing(10)
        r1.addWidget(QLabel("Approaches"))
        r1.addWidget(self.camera, 2)
        root.addLayout(r1)
        r2 = QHBoxLayout()
        self.model = ModelPicker()
        self.stride = QSpinBox()
        self.stride.setRange(1, 10)
        self.stride.setValue(2)
        self.stride.setPrefix("Every ")
        self.stride.setSuffix(" frame(s)")
        self.imgsz = QComboBox()
        for s in (640, 960):
            self.imgsz.addItem(f"{s} px", s)
        self.max_s = QSpinBox()
        self.max_s.setRange(0, 36000)
        self.max_s.setSpecialValueText("Whole video")
        self.max_s.setSuffix(" s")
        self.max_s.setPrefix("Limit ")
        self.rain = QCheckBox("Raining")
        self.save = QCheckBox("Save annotated video")
        self.save.setChecked(True)
        self.run_btn = QPushButton("Start")
        self.run_btn.setObjectName("primary")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        b_out = QPushButton("Open output folder")
        r2.addWidget(QLabel("Model"))
        r2.addWidget(self.model, 2)
        for w in (self.stride, self.imgsz, self.max_s, self.rain, self.save, self.run_btn, self.stop_btn, b_out):
            r2.addWidget(w)
        root.addLayout(r2)

        body = QHBoxLayout()
        pp, pl = panel()
        self.preview = Preview("Choose a video or the webcam, then press Start")
        pl.addWidget(self.preview)
        body.addWidget(pp, 6)
        side = QVBoxLayout()
        cards = QHBoxLayout()
        self.c_time, self.c_green, self.c_inc = Card("Video time"), Card("Recommended green"), Card("Incidents")
        for c in (self.c_time, self.c_green, self.c_inc):
            cards.addWidget(c)
        side.addLayout(cards)
        sp, sl = panel("Traffic state by approach")
        self.state = _table(["Approach", "Vehicles", "Queue", "Queue length", "Speed", "Risk"])
        sl.addWidget(self.state)
        sl.addWidget(hint("Risk is the XGBoost accident-risk score for the last full minute. With a camera file "
                          "that defines N, E, S and W, forecasts are added and the green is recommended per "
                          "movement."))
        side.addWidget(sp, 1)
        ip, il = panel("Incident alerts")
        self.incidents = QListWidget()
        self.incidents.setWordWrap(True)
        il.addWidget(self.incidents)
        side.addWidget(ip, 1)
        body.addLayout(side, 4)
        root.addLayout(body, 1)

        b_vid.clicked.connect(self.pick_video)
        b_cam.clicked.connect(lambda: self.source.setText("0"))
        self.camera.activated.connect(self.pick_camera)
        self.run_btn.clicked.connect(self.run)
        self.stop_btn.clicked.connect(lambda: self.worker and setattr(self.worker, "_stop", True))
        b_out.clicked.connect(lambda: open_folder(self.out_dir))

    def pick_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "Choose a traffic video", "", VIDEO_FILTER)
        if f:
            self.source.setText(f)

    def pick_camera(self, i):
        if self.camera.itemData(i) != "__browse__":
            return
        f, _ = QFileDialog.getOpenFileName(self, "Choose a camera file", str(CONFIG_DIR), "Camera file (*.json)")
        if f:
            self.camera.insertItem(1, f"Camera file: {Path(f).name}", f)
            self.camera.setCurrentIndex(1)
        else:
            self.camera.setCurrentIndex(0)

    def run(self):
        src = self.source.text().strip()
        if not src:
            self.pick_video()
            src = self.source.text().strip()
            if not src:
                return
        if not src.isdigit() and "://" not in src and not Path(src).exists():
            QMessageBox.warning(self, "Video analysis", f"File not found:\n{src}")
            return
        self.incidents.clear()
        self.seen_incidents = set()
        self.worker = VideoWorker(src, self.camera.currentData(), self.model.weights(), self.stride.value(),
                                  self.imgsz.currentData(), self.max_s.value(), self.save.isChecked(),
                                  int(self.rain.isChecked()))
        self.worker.frame.connect(self.on_frame)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self._finished)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.showMessage("Loading the detector and opening the video…")
        self.worker.start()

    def _finished(self):
        self.worker = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_frame(self, img, state, info):
        self.preview.set_image(img)
        self.out_dir = Path(info["out_dir"])
        t = info["t"]
        self.c_time.set(f"{int(t) // 60}:{int(t) % 60:02d}")
        self.c_inc.set(str(info["incidents"]), "#e5484d" if info["incidents"] else None)
        row = info.get("last_row") or {}
        if row.get("recommended_green"):
            self.c_green.set(row["recommended_green"], "#1fbf5c")
        rows = []
        for name, s in state.items():
            risk = row.get(f"risk_{name}")
            rows.append((name, s["vehicles"], f"{s['queue_veh']} veh", f"{s['queue_m']:.0f} m",
                         f"{s['mean_speed'] * 3.6:.0f} km/h", "–" if risk is None else f"{risk:.2f}"))
        _fill(self.state, rows)
        inc = info.get("incident")
        if inc and inc["id"] not in self.seen_incidents:
            self.seen_incidents.add(inc["id"])
            icon = "🚨" if inc["status"] == "confirmed" else "⚠️"
            self.incidents.insertItem(0, f"{icon} {inc['id']} ({inc['status'].replace('_', ' ')})\n{inc['summary']}")
        self.status.showMessage(f"Frame {info['frame']}  ·  {t:.1f} s of video processed")

    def on_done(self, res):
        self.out_dir = Path(res["out_dir"])
        self.status.showMessage(f"Finished: {res['frames']} frames, {res['intervals']} one-minute intervals, "
                                f"{res['incidents']} incidents. Results in {res['out_dir']}")

    def on_failed(self, msg):
        self.status.showMessage("Video analysis failed.")
        QMessageBox.critical(self, "Video analysis failed", msg[:3000])

    def shutdown(self):
        if self.worker:
            self.worker._stop = True
            self.worker.wait(5000)
