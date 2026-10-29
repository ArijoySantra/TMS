"""Results dashboard: the same views as the Streamlit dashboard, drawn natively."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QComboBox, QHBoxLayout, QHeaderView, QLabel,
                               QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QRadioButton,
                               QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

from traffic_ai.config import OUTPUT_DIR, RESULTS_DIR
from traffic_ai.gui.charts import Chart
from traffic_ai.gui.common import ACCENT, CONTROLLERS, Card, hint, open_folder, page_header

CTRL_ORDER = [v[0] for v in CONTROLLERS.values()]
CTRL_COLOR = {v[0]: v[1] for v in CONTROLLERS.values()}
METRICS = {"mean_delay_s": "Mean delay (s/veh)", "mean_queue_veh": "Mean queue (veh)",
           "throughput_veh_h": "Throughput (veh/h)", "switches_per_h": "Phase switches per hour",
           "recovery_time_s": "Queue recovery time (s)"}
CONDITIONS = {"clean": "Clean cameras", "noisy": "Noisy / missing approaches", "heavy": "Heavy noise (2×)"}
MODEL_NAMES = {"naive": "Persistence", "gru": "GRU", "lstm": "LSTM", "patchtst": "PatchTST"}


def read_csv(name):
    p = RESULTS_DIR / name
    try:
        return pd.read_csv(p) if p.exists() else None
    except pd.errors.EmptyDataError:
        return None


def df_table(df: pd.DataFrame, fmt="{:.2f}") -> QTableWidget:
    t = QTableWidget(len(df), len(df.columns))
    t.setHorizontalHeaderLabels([str(c) for c in df.columns])
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    for r, row in enumerate(df.itertuples(index=False)):
        for c, v in enumerate(row):
            txt = fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else (
                "–" if isinstance(v, float) and pd.isna(v) else str(v))
            it = QTableWidgetItem(txt)
            it.setTextAlignment(Qt.AlignCenter if c else Qt.AlignLeft | Qt.AlignVCenter)
            t.setItem(r, c, it)
    t.setMinimumHeight(min(60 + 30 * len(df), 260))
    return t


def scroll(widget):
    s = QScrollArea()
    s.setWidgetResizable(True)
    s.setFrameShape(QScrollArea.NoFrame)
    s.setWidget(widget)
    return s


def missing(text):
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.addStretch(1)
    lab = hint(text)
    lab.setAlignment(Qt.AlignCenter)
    lay.addWidget(lab)
    lay.addStretch(1)
    return w


class ControlTab(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        self.ep = read_csv("controller_episodes.csv")
        self.tr = read_csv("controller_queue_traces.csv")
        row = QHBoxLayout()
        self.group = QButtonGroup(self)
        for i, (k, v) in enumerate(CONDITIONS.items()):
            b = QRadioButton(v)
            b.setProperty("key", k)
            b.setChecked(i == 0)
            self.group.addButton(b)
            row.addWidget(b)
        row.addSpacing(16)
        self.metric_box = QComboBox()
        for k, v in METRICS.items():
            self.metric_box.addItem(v, k)
        row.addWidget(QLabel("Metric"))
        row.addWidget(self.metric_box)
        row.addStretch(1)
        lay.addLayout(row)
        self.bars = Chart(2.8)
        self.lines = Chart(2.6)
        lay.addWidget(self.bars)
        lay.addWidget(self.lines)
        lay.addWidget(hint("Bars are the mean over seeded evaluation hours; whiskers are one standard deviation. "
                           "Lower delay and queue are better."))
        self.group.buttonClicked.connect(self.redraw)
        self.metric_box.currentIndexChanged.connect(self.redraw)
        self.redraw()

    def redraw(self, *_):
        cond = self.group.checkedButton().property("key")
        metric = self.metric_box.currentData()
        d = self.ep[self.ep.condition == cond]
        agg = d.groupby("controller")[metric].agg(["mean", "std"])
        names = [c for c in CTRL_ORDER if c in agg.index]
        ax = self.bars.axes()
        x = np.arange(len(names))
        ax.bar(x, agg.loc[names, "mean"], yerr=agg.loc[names, "std"], color=[CTRL_COLOR[n] for n in names],
               width=0.6, capsize=4, error_kw={"ecolor": "#52514e", "lw": 1})
        for xi, v in zip(x, agg.loc[names, "mean"]):
            ax.annotate(f"{v:.1f}", (xi, v), ha="left", va="bottom", fontsize=8, xytext=(4, 2),
                        textcoords="offset points")
        ax.set_xticks(x, [n.replace("Uncertainty-aware", "Uncert.-aware") for n in names])
        ax.set_ylabel(METRICS[metric])
        ax.set_title(f"{METRICS[metric]}, {CONDITIONS[cond].lower()}", loc="left")
        ax.grid(axis="x", visible=False)
        self.bars.draw_idle()
        ax = self.lines.axes()
        if self.tr is not None:
            t = self.tr[self.tr.condition == cond]
            for n in names:
                s = t[t.controller == n]
                ax.plot(s.minute, s.queue, color=CTRL_COLOR[n], lw=1.8, label=n)
            ax.legend(ncol=5, fontsize=7.5, loc="upper left")
        ax.set_title("Total queue over one evaluation hour", loc="left")
        ax.set_xlabel("Minute")
        ax.set_ylabel("Vehicles queued")
        self.lines.draw_idle()


class ForecastTab(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        self.m = read_csv("forecast_metrics.csv")
        self.tr = read_csv("forecast_trace.csv")
        row = QHBoxLayout()
        self.target = QComboBox()
        self.target.addItem("Vehicle count per minute", "count")
        self.target.addItem("Queue length", "queue")
        self.app = QComboBox()
        for a, n in zip("NESW", ["North", "East", "South", "West"]):
            self.app.addItem(n, a)
        self.hz = QComboBox()
        for h in (5, 10, 15):
            self.hz.addItem(f"{h} minutes ahead", h)
        self.hz.setCurrentIndex(2)
        for lab, w in (("Target", self.target), ("Approach", self.app), ("Horizon", self.hz)):
            row.addWidget(QLabel(lab))
            row.addWidget(w)
        row.addStretch(1)
        lay.addLayout(row)
        self.mae = Chart(2.4)
        self.trace = Chart(2.6)
        lay.addWidget(self.mae)
        lay.addWidget(self.trace)
        lay.addWidget(hint("PatchTST is the model the PPO + forecast controller uses. MAE = mean absolute error "
                           "on held-out simulation runs (lower is better)."))
        for w in (self.target, self.app, self.hz):
            w.currentIndexChanged.connect(self.redraw)
        self.redraw()

    def redraw(self, *_):
        target, a, hz = self.target.currentData(), self.app.currentData(), self.hz.currentData()
        mm = self.m[self.m.target == target]
        ax = self.mae.axes()
        models = [m for m in MODEL_NAMES if m in set(mm.model)]
        width = 0.2
        colors = ["#9aa4b2", "#2a78d6", "#1baf7a", "#eb6834"]
        for i, m in enumerate(models):
            s = mm[mm.model == m].sort_values("horizon_min")
            ax.bar(np.arange(len(s)) + (i - 1.5) * width, s.MAE, width, label=MODEL_NAMES[m], color=colors[i])
        ax.set_xticks(range(3), ["5 min", "10 min", "15 min"])
        ax.set_ylabel("MAE")
        ax.set_title("Forecast error by horizon", loc="left")
        ax.legend(ncol=4, fontsize=7.5)
        ax.grid(axis="x", visible=False)
        self.mae.draw_idle()
        ax = self.trace.axes()
        col = f"{target}_{a}"
        if self.tr is not None and f"{col}_true_h{hz}" in self.tr:
            d = self.tr.head(240)
            ax.plot(d.step, d[f"{col}_true_h{hz}"], color="#2a78d6", lw=1.8, label="Actual")
            ax.plot(d.step, d[f"{col}_pred_h{hz}"], color="#eb6834", lw=1.8, label="PatchTST forecast")
            ax.legend(fontsize=7.5, loc="upper left")
        ax.set_title(f"{self.app.currentText()} approach, {hz} minutes ahead", loc="left")
        ax.set_xlabel("Test window (minutes)")
        self.trace.draw_idle()


class RiskTab(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        rm = json.loads((RESULTS_DIR / "risk_metrics.json").read_text())
        cards = QHBoxLayout()
        for k, lab in (("AUC", "ROC AUC"), ("F1", "F1 score"), ("recall", "Recall"), ("precision", "Precision")):
            cards.addWidget(Card(lab, f"{rm[k]:.2f}"))
        lay.addLayout(cards)
        hm = read_csv("risk_heatmap.csv")
        if hm is not None:
            ch = Chart(2.2)
            ax = ch.axes()
            vals = hm.drop(columns="approach").to_numpy(float)
            im = ax.imshow(vals, aspect="auto", cmap="Blues", vmin=0, vmax=1)
            ax.set_yticks(range(len(hm)), hm.approach)
            cols = list(hm.columns[1:])
            ax.set_xticks(range(len(cols)), cols)
            ax.set_xlabel("Minute of the hour")
            ax.set_ylabel("Approach")
            ax.grid(False)
            ax.set_title("Hotspots and peak periods: mean predicted risk", loc="left")
            for (i, j), v in np.ndenumerate(vals):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else "#1d2433")
            ch.fig.colorbar(im, ax=ax, fraction=0.03)
            lay.addWidget(ch)
        fi = read_csv("risk_feature_importance.csv")
        if fi is not None:
            fi.columns = ["feature", "importance"]
            fi = fi.sort_values("importance")
            ch = Chart(2.6)
            ax = ch.axes()
            ax.barh(fi.feature, fi.importance, color=ACCENT)
            ax.set_title("What drives the risk score (XGBoost importance)", loc="left")
            ax.grid(axis="y", visible=False)
            lay.addWidget(ch)
        lay.addWidget(hint("Labels come from SUMO collisions caused by a small share of risky drivers, so treat "
                           "these as simulation results until real crash records are available."))


class DetectorTab(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        d = read_csv("uvh26_val100_detector_comparison.csv")
        if d is None:
            lay.addWidget(missing("No detector benchmark yet."))
            return
        d = d.copy()
        d["model"] = d.model.replace({"best.pt": "YOLO26s fine-tuned (UVH-26)", "yolo26s.pt": "YOLO26s (COCO)",
                                      "rtdetr-l.pt": "RT-DETR-L (COCO)"})
        keep = ["model", "ms_per_image_cpu", "mAP50 (5 groups)", "mAP50-95 (5 groups)",
                "vehicle_AP50 (class-agnostic)", "mAP50 (14 UVH-26 classes)"]
        t = d[keep].rename(columns={"ms_per_image_cpu": "CPU ms / image"})
        lay.addWidget(QLabel("Detectors on 100 held-out UVH-26 images"))
        lay.addWidget(df_table(t))
        ch = Chart(2.6)
        ax = ch.axes()
        x = np.arange(len(t))
        ax.bar(x - 0.18, t["mAP50 (5 groups)"], 0.36, label="mAP50", color=ACCENT)
        ax.bar(x + 0.18, t["mAP50-95 (5 groups)"], 0.36, label="mAP50-95", color="#1baf7a")
        ax.set_xticks(x, t.model)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis="x", visible=False)
        ax.set_title("Accuracy on five shared vehicle groups (higher is better)", loc="left")
        lay.addWidget(ch)
        pc = read_csv("uvh26_val100_detector_per_class.csv")
        if pc is not None:
            pc = pc[pc.scheme == "5 groups"].pivot_table(index="class", columns="model", values="AP50").reset_index()
            pc.columns = [{"best.pt": "Fine-tuned", "yolo26s.pt": "YOLO26s COCO", "rtdetr-l.pt": "RT-DETR-L COCO"}
                          .get(c, c) for c in pc.columns]
            lay.addWidget(QLabel("AP50 per class"))
            lay.addWidget(df_table(pc))
        lay.addStretch(1)


class IncidentTab(QWidget):
    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        self.files = sorted([*(OUTPUT_DIR / "video").glob("*/incidents/INC-*.json"),
                             *(OUTPUT_DIR / "live").glob("*/incidents/INC-*.json")], key=lambda f: f.stat().st_mtime)
        if not self.files:
            lay.addWidget(missing("No incidents yet. Analyse a traffic video on the Video analysis page; "
                                  "accident alerts appear here."))
            return
        split = QSplitter()
        self.list = QListWidget()
        for f in reversed(self.files):
            inc = json.loads(f.read_text())
            icon = "🚨" if inc["status"] == "confirmed" else "⚠️"
            sev = f"  ·  {inc['severity']}" if inc.get("severity") else ""
            it = QListWidgetItem(f"{icon} {inc['incident_id']}  ·  {inc['status'].replace('_', ' ')}{sev}")
            it.setData(Qt.UserRole, inc)
            self.list.addItem(it)
        right = QWidget()
        rl = QVBoxLayout(right)
        self.img = QLabel()
        self.img.setAlignment(Qt.AlignCenter)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        rl.addWidget(self.img, 3)
        rl.addWidget(self.text, 2)
        split.addWidget(self.list)
        split.addWidget(right)
        split.setSizes([260, 700])
        lay.addWidget(split)
        self.list.currentItemChanged.connect(self.show_incident)
        self.list.setCurrentRow(0)

    def show_incident(self, item):
        if item is None:
            return
        inc = item.data(Qt.UserRole)
        snap = inc.get("snapshot_path")
        if snap and Path(snap).exists():
            self.img.setPixmap(QPixmap(snap).scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.img.setText("No snapshot")
        details = {k: inc.get(k) for k in ("severity", "severity_reasons", "camera", "maps_url", "approach",
                                           "candidate_score", "confirm_probability", "cues", "vehicles", "clip_path")}
        self.text.setPlainText(f"{inc['summary']}\n\n{json.dumps(details, indent=2)}")


class CameraTab(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        self.runs = sorted((OUTPUT_DIR / "video").glob("*/traffic_states.csv"))
        if not self.runs:
            lay.addWidget(missing("Process a video on the Video analysis page to see per-minute traffic states."))
            return
        row = QHBoxLayout()
        self.sel = QComboBox()
        for r in self.runs:
            self.sel.addItem(r.parent.name, str(r))
        row.addWidget(QLabel("Video"))
        row.addWidget(self.sel)
        b = QPushButton("Open folder")
        b.clicked.connect(lambda: open_folder(Path(self.sel.currentData()).parent))
        row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.holder = QVBoxLayout()
        lay.addLayout(self.holder, 1)
        self.sel.currentIndexChanged.connect(self.load)
        self.load()

    def load(self, *_):
        while self.holder.count():
            w = self.holder.takeAt(0).widget()
            if w:
                w.deleteLater()
        try:
            df = pd.read_csv(self.sel.currentData())
        except pd.errors.EmptyDataError:
            df = None
        if df is None or df.empty:
            self.holder.addWidget(missing("This video was shorter than one minute, so no traffic states were "
                                          "recorded."))
        else:
            t = df_table(df)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.holder.addWidget(t)


class ResultsPage(QWidget):
    def __init__(self, status):
        super().__init__()
        self.setObjectName("page")
        self.status = status
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(20, 16, 20, 16)
        head = QHBoxLayout()
        head.addWidget(page_header(
            "Results dashboard",
            "Experiment results: adaptive signal control vs fixed-time, traffic forecasting, accident-risk "
            "hotspots, detector accuracy, and incidents from analysed videos."), 1)
        b = QPushButton("Reload")
        b.clicked.connect(self.build)
        b2 = QPushButton("Open results folder")
        b2.clicked.connect(lambda: open_folder(RESULTS_DIR))
        head.addWidget(b, 0, Qt.AlignBottom)
        head.addWidget(b2, 0, Qt.AlignBottom)
        self.root.addLayout(head)
        self.tabs = None
        self.build()

    def build(self):
        idx = self.tabs.currentIndex() if self.tabs else 0
        if self.tabs:
            self.tabs.deleteLater()
        self.tabs = QTabWidget()
        specs = [("Signal control", ControlTab, "controller_episodes.csv", "Run the experiments to compare controllers."),
                 ("Forecasting", ForecastTab, "forecast_metrics.csv", "Train the forecasting models first."),
                 ("Accident risk", RiskTab, "risk_metrics.json", "Train the risk model first."),
                 ("Detectors", DetectorTab, None, ""),
                 ("Incidents", IncidentTab, None, ""),
                 ("Camera traffic states", CameraTab, None, "")]
        for name, cls, need, msg in specs:
            if need and not (RESULTS_DIR / need).exists():
                self.tabs.addTab(missing(msg), name)
                continue
            try:
                self.tabs.addTab(scroll(cls()), name)
            except Exception as e:  # a damaged results file should not break the whole page
                self.tabs.addTab(missing(f"Could not load this view: {e}"), name)
        self.tabs.setCurrentIndex(idx)
        self.root.addWidget(self.tabs, 1)
        self.status.showMessage(f"Results loaded from {RESULTS_DIR}")
