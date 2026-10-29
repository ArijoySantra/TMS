"""About / home page: what the system does and where its files are."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from traffic_ai.config import CONFIG_DIR, OUTPUT_DIR, ROOT
from traffic_ai.gui.common import hint, open_folder, page_header, panel

STAGES = [
    ("SEE", "Detect and track road users with YOLO26 fine-tuned on UVH-26 and ByteTrack, then measure counts, "
            "density, queue length and speed per approach from live cameras, using the road width for density.", "Camera setup"),
    ("PREDICT", "Forecast vehicle counts and queues 5, 10 and 15 minutes ahead with PatchTST, and score accident "
                "risk per approach with XGBoost.", "Results dashboard"),
    ("DECIDE", "Adaptive green times from live density at the real junction (PPO agents in simulation), and "
               "collision detection with severity that alerts the traffic controller.", "Live control"),
]


class AboutPage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.setObjectName("page")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.addWidget(page_header(
            "AI-based adaptive traffic signal control, predictive analytics and accident detection",
            "A research prototype for Indian mixed traffic."))
        row = QHBoxLayout()
        for title, text, page in STAGES:
            p, pl = panel(title)
            lab = QLabel(text)
            lab.setWordWrap(True)
            pl.addWidget(lab)
            pl.addStretch(1)
            b = QPushButton(f"Open {page}")
            b.clicked.connect(lambda _=False, n=page: window.go(n))
            pl.addWidget(b, 0, Qt.AlignLeft)
            row.addWidget(p)
        lay.addLayout(row)

        p, pl = panel("How to use it")
        steps = [
            "<b>Camera setup</b>: enter the junction's name, address and GPS, add one camera per approach (RTSP / IP "
            "camera or USB webcam) with the road width it covers, and set who receives collision alerts.",
            "<b>Live control</b>: press Start. Each camera's density sets the next green time; collisions are rated "
            "minor, moderate or severe, and severe ones alert the traffic controller with the location.",
            "<b>Signal simulation</b>: compare controllers, including the PPO agents, on the SUMO junction.",
            "<b>Results dashboard</b>: charts and tables from the experiments for your report.",
            "<b>Recorded video</b> and <b>Photo detection</b>: analyse files instead of live cameras.",
        ]
        for s in steps:
            lab = QLabel("•  " + s)
            lab.setWordWrap(True)
            lab.setTextFormat(Qt.RichText)
            pl.addWidget(lab)
        lay.addWidget(p)

        p, pl = panel("Files")
        grid = QGridLayout()
        for r, (name, path) in enumerate([("Project folder", ROOT), ("Settings (default.yaml, camera files)",
                                                                       CONFIG_DIR),
                                          ("Results, models and outputs", OUTPUT_DIR)]):
            grid.addWidget(QLabel(name), r, 0)
            lab = QLabel(str(path))
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(lab, r, 1)
            b = QPushButton("Open")
            b.clicked.connect(lambda _=False, x=path: open_folder(x))
            grid.addWidget(b, r, 2)
        grid.setColumnStretch(1, 1)
        pl.addLayout(grid)
        lay.addWidget(p)
        lay.addWidget(hint(
            "All signal-control, forecasting and risk results come from SUMO simulation, not real junctions. "
            "This is a research prototype: do not connect it to a real traffic signal controller."))
        lay.addStretch(1)
