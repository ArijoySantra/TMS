"""Live junction page: runs a signal controller in SUMO and draws the junction, vehicles and signals."""
from __future__ import annotations

import math
import subprocess
import sys
import time
import traceback

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QHeaderView,
                               QLabel, QMessageBox, QProgressBar, QPushButton, QSlider, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from traffic_ai.config import load_config
from traffic_ai.gui.charts import Chart
from traffic_ai.gui.common import (APPROACH_COLORS, CONTROLLERS, MUTED, Card, cli_command, hint, page_header,
                                   panel)

VEHICLE_STYLE = {  # type id -> (colour, length m, width m)
    "motorcycle": ("#ff8a3d", 2.0, 0.8), "car": ("#4f8ef7", 4.2, 1.7), "auto": ("#2bb673", 3.0, 1.4),
    "lcv": ("#9b6bdf", 5.0, 1.9), "bus": ("#e5484d", 11.0, 2.5), "truck": ("#8a5a44", 9.0, 2.5),
    "aggressive": ("#111111", 4.2, 1.7),
}
LEGEND = [("Two-wheeler", "motorcycle"), ("Car", "car"), ("Auto", "auto"), ("LCV", "lcv"), ("Bus", "bus"),
          ("Truck", "truck")]
SPEEDS = [1, 2, 5, 10, 25, 60, 0]  # simulated seconds per real second; 0 = as fast as possible
SIGNAL_COLORS = {"G": "#1fbf5c", "g": "#1fbf5c", "y": "#f5b800", "Y": "#f5b800", "r": "#e5484d", "R": "#e5484d"}


class _Stop(Exception):
    pass


def _hide_sumo_console():
    """On Windows, start sumo.exe without flashing a console window next to the app."""
    if sys.platform != "win32":
        return
    import traci.main as tm
    if getattr(tm.subprocess, "hidden", False):
        return

    class _NoWindow:
        hidden = True

        def __getattr__(self, name):
            return getattr(subprocess, name)

        @staticmethod
        def Popen(*a, **k):  # noqa: N802
            k.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
            return subprocess.Popen(*a, **k)

    tm.subprocess = _NoWindow()


class SimWorker(QThread):
    geometry = Signal(dict)
    snapshot = Signal(dict)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, controller: str, noisy: bool, seed: int, minutes: int):
        super().__init__()
        self.controller, self.noisy, self.seed, self.minutes = controller, noisy, seed, minutes
        self.speed = 10
        self.paused = False
        self._stop = False
        self._last_emit = 0.0
        self._wall0 = 0.0
        self._sim0 = 0

    def stop(self):
        self._stop = True

    def set_speed(self, speed: int):
        self.speed = speed
        self._wall0, self._sim0 = time.perf_counter(), getattr(self, "_t", 0)

    def run(self):
        env = None
        try:
            import traci.constants as tc

            from traffic_ai.cli import build_controller
            cfg = load_config()
            _hide_sumo_console()
            # TraCI (SUMO as a separate process) rather than libsumo: works the same in the packaged app.
            env, ctrl = build_controller(cfg, self.controller, noisy=self.noisy, use_libsumo=False,
                                         episode_seconds=self.minutes * 60)
            env.reset(seed=self.seed)
            ctrl.reset(env)
            c = env.conn
            lanes = {ln: (c.lane.getShape(ln), c.lane.getWidth(ln)) for ln in c.lane.getIDList()}
            lane_links: dict[str, list[int]] = {}
            for i, (_, _, ln) in enumerate(env.link_info):
                lane_links.setdefault(ln, []).append(i)
            jid = c.trafficlight.getControlledJunctions(env.tls_id)[0] if hasattr(
                c.trafficlight, "getControlledJunctions") else "C"
            self.geometry.emit({"lanes": lanes, "junction": c.junction.getShape(jid), "lane_links": lane_links})
            c.junction.subscribeContext(jid, tc.CMD_GET_VEHICLE_VARIABLE, 1000,
                                        [tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_TYPE])
            self._tc = tc
            self._jid = jid
            self._trace = []
            self._wall0, self._sim0, self._t = time.perf_counter(), 0, 0
            env.step_callback = self._on_second
            done = False
            while not done:
                _, _, done, _, _ = env.step(ctrl.act(env))
            self._emit(env, final=True)
            m = env.episode_metrics()
            m.update(controller=self.controller, noisy=self.noisy, seed=self.seed, minutes=self.minutes,
                     trace=self._trace)
            self.done.emit(m)
        except _Stop:
            self.done.emit({})
        except Exception as e:  # shown to the user
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")
        finally:
            if env is not None:
                env.close()

    # called by the environment after every simulated second
    def _on_second(self, env):
        self._t = env.t
        if env.t % 10 == 0:
            self._trace.append((env.t / 60, float(env.stats["queue"][-1]) if env.stats["queue"] else 0.0))
        while self.paused and not self._stop:
            time.sleep(0.05)
            self._wall0, self._sim0 = time.perf_counter(), env.t
        if self._stop:
            raise _Stop()
        now = time.perf_counter()
        if self.speed:
            target = self._wall0 + (env.t - self._sim0) / self.speed
            if target > now:
                time.sleep(target - now)
            elif now - target > 1.0:  # fell behind (slow PC): do not try to catch up
                self._wall0, self._sim0 = now, env.t
        if time.perf_counter() - self._last_emit >= 1 / 30:
            self._emit(env)

    def _emit(self, env, final: bool = False):
        self._last_emit = time.perf_counter()
        c, tc = env.conn, self._tc
        res = c.junction.getContextSubscriptionResults(self._jid) or {}
        veh = [(v[tc.VAR_POSITION][0], v[tc.VAR_POSITION][1], v[tc.VAR_ANGLE], v[tc.VAR_TYPE])
               for v in res.values()]
        q_app = np.zeros(4)
        for i, ln in enumerate(env.lanes):
            q_app[env.lane_approach[i]] += env.true_meas[i, 1] if final else c.lane.getLastStepHaltingNumber(ln)
        s = env.stats
        hours = max(env.t, 1) / 3600
        self.snapshot.emit({
            "t": env.t, "total": env.episode_seconds, "vehicles": veh,
            "tls": c.trafficlight.getRedYellowGreenState(env.tls_id),
            "phase": env.phase_names[env.phase], "time_in_green": env.time_since_green,
            "queue_by_approach": q_app.tolist(), "queue": float(q_app.sum()),
            "mean_delay": float(np.mean(s["delays"])) if s["delays"] else float("nan"),
            "throughput": s["arrived"] / hours, "switches_h": s["switches"] / hours,
            "forecast": env.forecast.tolist() if env.use_forecast else None,
            "mask": env.mask.tolist() if env.noisy else None, "trace": list(self._trace),
        })


class JunctionCanvas(QWidget):
    """Top-down drawing of the SUMO junction."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 420)
        self.geom = None
        self.snap = None
        self.radius = 85.0  # metres shown around the junction centre; mouse wheel zooms
        self.setToolTip("Scroll to zoom in or out")

    def set_geometry(self, g):
        self.geom = g
        pts = np.array(g["junction"]) if g["junction"] else np.array(
            [p for shape, _ in g["lanes"].values() for p in shape])
        self.centre = pts.mean(axis=0)
        all_pts = np.array([p for shape, _ in g["lanes"].values() for p in shape])
        self.max_radius = float(np.abs(all_pts - self.centre).max())
        self._update_bounds()

    def _update_bounds(self):
        cx, cy = self.centre
        r = self.radius
        self.bounds = (cx - r, cy - r, cx + r, cy + r)
        self.update()

    def wheelEvent(self, e):
        if self.geom is None:
            return
        self.radius = float(np.clip(self.radius * (0.85 if e.angleDelta().y() > 0 else 1 / 0.85), 25,
                                    self.max_radius + 5))
        self._update_bounds()

    def set_snapshot(self, s):
        self.snap = s
        self.update()

    def _xf(self):
        x0, y0, x1, y1 = self.bounds
        pad = 16
        scale = min((self.width() - 2 * pad) / (x1 - x0), (self.height() - 2 * pad) / (y1 - y0))
        ox = (self.width() - (x1 - x0) * scale) / 2
        oy = (self.height() - (y1 - y0) * scale) / 2
        return lambda x, y: QPointF(ox + (x - x0) * scale, self.height() - (oy + (y - y0) * scale)), scale

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#cfe3c4"))  # grass
        if self.geom is None:
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "Press Start to run the junction simulation")
            return
        xf, scale = self._xf()
        p.setClipRect(self.rect())
        # roads
        for ln, (shape, width) in self.geom["lanes"].items():
            pen = QPen(QColor("#5a5f68"), width * scale + 0.6, Qt.SolidLine, Qt.FlatCap, Qt.RoundJoin)
            p.setPen(pen)
            p.drawPolyline(QPolygonF([xf(*pt) for pt in shape]))
        if self.geom["junction"]:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#5a5f68"))
            p.drawPolygon(QPolygonF([xf(*pt) for pt in self.geom["junction"]]))
        # lane separators on normal lanes
        p.setPen(QPen(QColor(255, 255, 255, 140), 1, Qt.DashLine))
        for ln, (shape, width) in self.geom["lanes"].items():
            if ln.startswith(":"):
                continue
            (ax, ay), (bx, by) = shape[0], shape[-1]
            dx, dy = bx - ax, by - ay
            n = math.hypot(dx, dy) or 1
            nx, ny = -dy / n * width / 2, dx / n * width / 2
            p.drawLine(xf(ax + nx, ay + ny), xf(bx + nx, by + ny))
        s = self.snap
        if s:
            # signal heads at each stop line
            tls = s["tls"]
            for ln, idx in self.geom["lane_links"].items():
                shape, width = self.geom["lanes"][ln]
                states = {tls[i] for i in idx if i < len(tls)}
                ch = "G" if states & {"G", "g"} else ("y" if states & {"y", "Y"} else "r")
                (ax, ay), (bx, by) = shape[-2], shape[-1]
                n = math.hypot(bx - ax, by - ay) or 1
                nx, ny = -(by - ay) / n * width / 2, (bx - ax) / n * width / 2
                p.setPen(QPen(QColor(SIGNAL_COLORS[ch]), max(3.0, 1.4 * scale), Qt.SolidLine, Qt.FlatCap))
                p.drawLine(xf(bx + nx, by + ny), xf(bx - nx, by - ny))
            # vehicles: SUMO position is the front bumper, angle is degrees clockwise from north
            p.setPen(QPen(QColor(0, 0, 0, 90), 0.6))
            for x, y, ang, vt in s["vehicles"]:
                col, length, width = VEHICLE_STYLE.get(vt, ("#4f8ef7", 4.2, 1.7))
                a = math.radians(ang)
                hx, hy = math.sin(a), math.cos(a)
                px, py = hy, -hx
                bx, by = x - hx * length, y - hy * length
                w = width / 2
                poly = QPolygonF([xf(x + px * w, y + py * w), xf(x - px * w, y - py * w),
                                  xf(bx - px * w, by - py * w), xf(bx + px * w, by + py * w)])
                p.setBrush(QColor(col))
                p.drawPolygon(poly)
            # approach labels
            x0, y0, x1, y1 = self.bounds
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            m = (x1 - x0) * 0.06
            f = QFont(self.font())
            f.setBold(True)
            f.setPointSize(11)
            p.setFont(f)
            for name, (lx, ly) in {"N": (cx - 3 * m, y1 - m), "S": (cx + 3 * m, y0 + m), "E": (x1 - m, cy + 3 * m),
                                   "W": (x0 + m, cy - 3 * m)}.items():
                pt = xf(lx, ly)
                p.setPen(QColor(APPROACH_COLORS[name]))
                p.drawText(QRectF(pt.x() - 12, pt.y() - 10, 24, 20), Qt.AlignCenter, name)
        # legend
        f = QFont(self.font())
        f.setPointSize(8)
        p.setFont(f)
        y = 10
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 215))
        p.drawRoundedRect(QRectF(8, 6, 98, 18 * len(LEGEND) + 8), 6, 6)
        for label, key in LEGEND:
            p.setBrush(QColor(VEHICLE_STYLE[key][0]))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(16, y + 4, 14, 9), 2, 2)
            p.setPen(QColor("#1d2433"))
            p.drawText(QRectF(36, y, 70, 18), Qt.AlignVCenter, label)
            y += 18


class ApproachBars(QWidget):
    """Queue per approach as four horizontal bars."""

    def __init__(self):
        super().__init__()
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.bars, self.vals = {}, {}
        for r, a in enumerate("NESW"):
            name = QLabel({"N": "North", "E": "East", "S": "South", "W": "West"}[a])
            bar = QProgressBar()
            bar.setRange(0, 60)
            bar.setTextVisible(False)
            bar.setStyleSheet(f"QProgressBar::chunk {{ background: {APPROACH_COLORS[a]}; border-radius: 4px; }}")
            val = QLabel("0")
            val.setMinimumWidth(56)
            lay.addWidget(name, r, 0)
            lay.addWidget(bar, r, 1)
            lay.addWidget(val, r, 2)
            self.bars[a], self.vals[a] = bar, val

    def set(self, q, mask=None, forecast=None):
        for i, a in enumerate("NESW"):
            self.bars[a].setValue(int(min(q[i], 60)))
            txt = f"{q[i]:.0f} veh"
            if forecast is not None:
                txt += f"  → {forecast[0][i]:.1f}/min"
            if mask is not None and mask[i]:
                txt += "  (camera lost)"
            self.vals[a].setText(txt)


class JunctionPage(QWidget):
    def __init__(self, status):
        super().__init__()
        self.setObjectName("page")
        self.status = status
        self.worker = None
        self.runs = []
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.addWidget(page_header(
            "Signal simulation",
            "Watch an AI signal controller run a 4-arm junction with Indian mixed traffic (SUMO simulation). "
            "Run the same seed with different controllers to compare them."))

        # controls
        ctl = QHBoxLayout()
        self.ctrl = QComboBox()
        for key, (label, _) in CONTROLLERS.items():
            self.ctrl.addItem(label, key)
        self.ctrl.setCurrentIndex(3)
        self.noisy = QCheckBox("Noisy / missing camera data")
        self.seed = QSpinBox()
        self.seed.setRange(1, 99999)
        self.seed.setValue(1)
        self.seed.setPrefix("Seed ")
        self.minutes = QComboBox()
        for m in (10, 20, 30, 60):
            self.minutes.addItem(f"{m} sim. minutes", m)
        self.minutes.setCurrentIndex(2)
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.speed = QSlider(Qt.Horizontal)
        self.speed.setRange(0, len(SPEEDS) - 1)
        self.speed.setValue(3)
        self.speed.setFixedWidth(110)
        self.speed_lab = QLabel()
        self._speed_text()
        self.sumo_gui = QPushButton("Open in SUMO-GUI")
        for w in (QLabel("Controller"), self.ctrl, self.noisy, self.seed, self.minutes, self.start_btn,
                  self.pause_btn, self.stop_btn):
            ctl.addWidget(w)
        ctl.addSpacing(12)
        ctl.addWidget(QLabel("Speed"))
        ctl.addWidget(self.speed)
        ctl.addWidget(self.speed_lab)
        ctl.addStretch(1)
        ctl.addWidget(self.sumo_gui)
        root.addLayout(ctl)

        body = QHBoxLayout()
        body.setSpacing(14)
        cv_panel, cv_lay = panel()
        self.canvas = JunctionCanvas()
        cv_lay.addWidget(self.canvas, 1)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        cv_lay.addWidget(self.progress)
        body.addWidget(cv_panel, 5)

        side = QVBoxLayout()
        side.setSpacing(10)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.c_time, self.c_phase = Card("Simulated time"), Card("Green phase")
        self.c_queue, self.c_delay = Card("Vehicles queued now"), Card("Mean delay (s/veh)")
        self.c_thru, self.c_sw = Card("Throughput (veh/h)"), Card("Phase switches / h")
        for i, c in enumerate((self.c_time, self.c_phase, self.c_queue, self.c_delay, self.c_thru, self.c_sw)):
            grid.addWidget(c, i // 2, i % 2)
        side.addLayout(grid)
        qp, ql = panel("Queue by approach")
        self.bars = ApproachBars()
        ql.addWidget(self.bars)
        self.fc_hint = hint("")
        ql.addWidget(self.fc_hint)
        side.addWidget(qp)
        chp, chl = panel("Total queue over time")
        self.chart = Chart(2.2)
        chl.addWidget(self.chart)
        side.addWidget(chp, 1)
        body.addLayout(side, 4)
        root.addLayout(body, 1)

        rp, rl = panel("Runs this session")
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Controller", "Camera", "Seed", "Minutes", "Mean delay (s)",
                                              "Mean queue (veh)", "Throughput (veh/h)", "Switches/h"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setFixedHeight(130)
        rl.addWidget(self.table)
        root.addWidget(rp)

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.speed.valueChanged.connect(self._speed_changed)
        self.sumo_gui.clicked.connect(self.open_sumo_gui)
        self._draw_chart(None)

    def _speed_text(self):
        v = SPEEDS[self.speed.value()]
        self.speed_lab.setText("Max" if v == 0 else f"{v}×")

    def _speed_changed(self):
        self._speed_text()
        if self.worker:
            self.worker.set_speed(SPEEDS[self.speed.value()])

    def start(self):
        if self.worker:
            return
        key = self.ctrl.currentData()
        self.worker = SimWorker(key, self.noisy.isChecked(), self.seed.value(), self.minutes.currentData())
        self.worker.set_speed(SPEEDS[self.speed.value()])
        self.worker.geometry.connect(self.canvas.set_geometry)
        self.worker.snapshot.connect(self.on_snapshot)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self._cleanup)
        self._set_running(True)
        self.status.showMessage(f"Starting SUMO with the {CONTROLLERS[key][0]} controller…")
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.stop()

    def toggle_pause(self):
        if self.worker:
            self.worker.paused = not self.worker.paused
            self.pause_btn.setText("Resume" if self.worker.paused else "Pause")

    def _set_running(self, running):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.pause_btn.setEnabled(running)
        self.pause_btn.setText("Pause")
        for w in (self.ctrl, self.noisy, self.seed, self.minutes):
            w.setEnabled(not running)

    def _cleanup(self):
        self.worker = None
        self._set_running(False)

    def on_snapshot(self, s):
        self.canvas.set_snapshot(s)
        t = s["t"]
        self.c_time.set(f"{t // 3600:d}:{t // 60 % 60:02d}:{t % 60:02d}")
        self.c_phase.set(f"{s['phase']} ({s['time_in_green']} s)")
        self.c_queue.set(f"{s['queue']:.0f}")
        self.c_delay.set("–" if math.isnan(s["mean_delay"]) else f"{s['mean_delay']:.1f}")
        if t >= 120:
            self.c_thru.set(f"{s['throughput']:.0f}")
            self.c_sw.set(f"{s['switches_h']:.0f}")
        self.bars.set(s["queue_by_approach"], s["mask"], s["forecast"])
        self.fc_hint.setText("→ value is the forecast arrivals per minute in the next 5 minutes (PatchTST)."
                             if s["forecast"] is not None else "")
        self.progress.setValue(int(100 * t / s["total"]))
        if t % 30 == 0 or not hasattr(self, "_last_chart") or time.perf_counter() - self._last_chart > 1:
            self._draw_chart(s["trace"])

    def _draw_chart(self, trace):
        self._last_chart = time.perf_counter()
        ax = self.chart.axes()
        for r in self.runs[-4:]:
            if r["trace"]:
                x, y = zip(*r["trace"])
                ax.plot(x, y, lw=1.2, alpha=0.55, color=CONTROLLERS[r["controller"]][1],
                        label=f"{CONTROLLERS[r['controller']][0]} (seed {r['seed']})")
        if trace:
            x, y = zip(*trace)
            key = self.worker.controller if self.worker else self.ctrl.currentData()
            ax.plot(x, y, lw=2.2, color=CONTROLLERS[key][1], label=f"{CONTROLLERS[key][0]} (now)")
        ax.set_xlabel("Minute")
        ax.set_ylabel("Vehicles queued")
        ax.set_ylim(bottom=0)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="upper left")
        self.chart.draw_idle()

    def on_done(self, m):
        if not m:
            self.status.showMessage("Simulation stopped.")
            return
        self.runs.append(m)
        r = self.table.rowCount()
        self.table.insertRow(r)
        vals = [CONTROLLERS[m["controller"]][0], "Noisy" if m["noisy"] else "Clean", str(m["seed"]),
                str(m["minutes"]), f"{m['mean_delay_s']:.1f}", f"{m['mean_queue_veh']:.1f}",
                f"{m['throughput_veh_h']:.0f}", f"{m['switches_per_h']:.0f}"]
        for c, v in enumerate(vals):
            it = QTableWidgetItem(v)
            if c == 0:
                it.setForeground(QColor(CONTROLLERS[m["controller"]][1]))
            self.table.setItem(r, c, it)
        self._draw_chart(None)
        self.status.showMessage(f"{vals[0]} finished: mean delay {vals[4]} s/veh, mean queue {vals[5]} vehicles.")

    def on_failed(self, msg):
        self.status.showMessage("Simulation failed.")
        QMessageBox.critical(self, "Simulation failed", msg[:3000])

    def open_sumo_gui(self):
        args = ["demo", "--controller", self.ctrl.currentData(), "--gui", "--seed", str(self.seed.value())]
        if self.noisy.isChecked():
            args.append("--noisy")
        try:
            subprocess.Popen(cli_command(*args))
            self.status.showMessage("Opening SUMO-GUI. Press the green Play button in its window.")
        except OSError as e:
            QMessageBox.warning(self, "SUMO-GUI", str(e))

    def shutdown(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(5000)
