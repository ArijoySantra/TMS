"""Shared pieces for the desktop app: paths, theme, small widgets and helpers."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPalette
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget)

from traffic_ai.config import OUTPUT_DIR, ROOT

GUI_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = OUTPUT_DIR / "weights"

# Colours. Controller colours match the Streamlit dashboard so screenshots agree.
ACCENT = "#2a78d6"
INK = "#1d2433"
MUTED = "#5f6b7a"
SURFACE = "#ffffff"
CANVAS = "#f3f5f8"
BORDER = "#dfe3ea"
SIDEBAR = "#101b2d"
CONTROLLERS = {
    "fixed": ("Fixed-time", "#2a78d6"),
    "max_pressure": ("Max-Pressure", "#eb6834"),
    "ppo": ("PPO", "#1baf7a"),
    "ppo_forecast": ("PPO + forecast", "#eda100"),
    "ppo_uncertainty": ("Uncertainty-aware PPO", "#e87ba4"),
}
APPROACH_COLORS = {"N": "#ff8000", "E": "#00b400", "S": "#0080ff", "W": "#c800c8"}

STYLE = f"""
QWidget {{ font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif; font-size: 10pt; color: {INK}; }}
QMainWindow, #page {{ background: {CANVAS}; }}
#sidebar {{ background: {SIDEBAR}; }}
#sidebar QLabel {{ color: #e8edf5; }}
#brand {{ font-size: 13pt; font-weight: 600; color: white; }}
#brandSub {{ color: #8fa3bf; font-size: 9pt; }}
#nav {{ background: transparent; border: none; outline: none; }}
#nav::item {{ color: #c6d1e0; padding: 10px 14px; border-radius: 6px; margin: 2px 8px; }}
#nav::item:selected {{ background: {ACCENT}; color: white; }}
#nav::item:hover:!selected {{ background: #1c2b44; }}
#pageTitle {{ font-size: 16pt; font-weight: 600; }}
#pageSub {{ color: {MUTED}; }}
#card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; }}
#cardLabel {{ color: {MUTED}; font-size: 9pt; }}
#cardValue {{ font-size: 15pt; font-weight: 600; }}
#panel {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; }}
#sectionTitle {{ font-weight: 600; font-size: 10.5pt; }}
#hint {{ color: {MUTED}; font-size: 9pt; }}
QPushButton {{ background: {SURFACE}; border: 1px solid #c9d0db; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover {{ background: #eef2f7; }}
QPushButton:disabled {{ color: #9aa4b2; background: #f3f5f8; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border: 1px solid {ACCENT}; font-weight: 600; }}
QPushButton#primary:hover {{ background: #1f66bd; }}
QPushButton#primary:disabled {{ background: #9dbbe3; border-color: #9dbbe3; }}
QPushButton#danger {{ color: #b3261e; }}
QComboBox, QLineEdit {{ background: white; border: 1px solid #c9d0db; border-radius: 6px; padding: 4px 6px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; background: {SURFACE}; top: -1px; }}
QTabBar::tab {{ padding: 7px 14px; margin-right: 2px; border: 1px solid transparent; color: {MUTED}; }}
QTabBar::tab:selected {{ color: {INK}; border-bottom: 2px solid {ACCENT}; font-weight: 600; }}
QTableWidget, QTableView, QListWidget, QPlainTextEdit {{ background: white; border: 1px solid {BORDER}; border-radius: 6px; gridline-color: #edf0f4; }}
QHeaderView::section {{ background: #f6f8fb; border: none; border-bottom: 1px solid {BORDER}; padding: 5px; font-weight: 600; }}
QProgressBar {{ border: 1px solid {BORDER}; border-radius: 5px; background: #eef1f5; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QStatusBar {{ background: white; border-top: 1px solid {BORDER}; color: {MUTED}; }}
"""


def apply_theme(app) -> None:
    """Fusion style with a fixed light palette, so Windows dark mode cannot make text unreadable."""
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in [(QPalette.Window, CANVAS), (QPalette.WindowText, INK), (QPalette.Base, "#ffffff"),
                      (QPalette.AlternateBase, "#f6f8fb"), (QPalette.Text, INK), (QPalette.Button, "#ffffff"),
                      (QPalette.ButtonText, INK), (QPalette.Highlight, ACCENT),
                      (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, "#ffffff"),
                      (QPalette.ToolTipText, INK), (QPalette.PlaceholderText, "#8a94a3")]:
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)


class Card(QFrame):
    """A small stat tile: label above a big value."""

    def __init__(self, label: str, value: str = "–", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(0)
        self._label = QLabel(label)
        self._label.setObjectName("cardLabel")
        self._value = QLabel(value)
        self._value.setObjectName("cardValue")
        lay.addWidget(self._label)
        lay.addWidget(self._value)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, value: str, color: str | None = None):
        self._value.setText(value)
        self._value.setStyleSheet(f"color: {color};" if color else "")


def panel(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("panel")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(12, 10, 12, 12)
    if title:
        t = QLabel(title)
        t.setObjectName("sectionTitle")
        lay.addWidget(t)
    return f, lay


def page_header(title: str, subtitle: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 4)
    lay.setSpacing(2)
    t = QLabel(title)
    t.setObjectName("pageTitle")
    s = QLabel(subtitle)
    s.setObjectName("pageSub")
    s.setWordWrap(True)
    lay.addWidget(t)
    lay.addWidget(s)
    return w


def hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("hint")
    lab.setWordWrap(True)
    return lab


def bgr_to_qimage(img: np.ndarray) -> QImage:
    """OpenCV BGR array -> detached QImage (safe to send between threads)."""
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ModelPicker(QComboBox):
    """Detector weights: the fine-tuned UVH-26 models shipped in outputs/weights, COCO weights
    (downloaded on first use), or any .pt file the user browses to."""

    BROWSE = "__browse__"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.refresh()
        self.activated.connect(self._on_activated)

    def refresh(self):
        self.clear()
        for p in sorted(WEIGHTS_DIR.glob("*.pt")):
            if p.name in ("yolo26s.pt", "rtdetr-l.pt"):
                continue
            self.addItem(f"Fine-tuned on UVH-26: {p.stem}", str(p))
        self.addItem("YOLO26s, COCO (downloads ~20 MB once)", str(WEIGHTS_DIR / "yolo26s.pt"))
        self.addItem("RT-DETR-L, COCO (downloads ~65 MB once, slow on CPU)", str(WEIGHTS_DIR / "rtdetr-l.pt"))
        self.addItem("Other weights file…", self.BROWSE)

    def _on_activated(self, i):
        if self.itemData(i) != self.BROWSE:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose detector weights", str(WEIGHTS_DIR),
                                              "PyTorch weights (*.pt)")
        if path:
            self.insertItem(0, f"Custom: {Path(path).name}", path)
        self.setCurrentIndex(0)

    def weights(self) -> str:
        return self.currentData()


def open_folder(path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def cli_command(*args: str) -> list[str]:
    """Command that runs `python -m traffic_ai.cli <args>`, also from the packaged exe."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--cli", *args]
    return [sys.executable, "-m", "traffic_ai.cli", *args]


def sample_image_dirs() -> list[Path]:
    """Folders with example images: samples/ shipped with the app, and the project's Dataset folder."""
    cands = [ROOT / "samples" / "images", ROOT.parent / "Dataset" / "images" / "test",
             ROOT / "Dataset" / "images" / "test"]
    return [p for p in cands if p.is_dir() and any(p.glob("*.jpg"))]


__all__ = ["ACCENT", "APPROACH_COLORS", "BORDER", "CONTROLLERS", "Card", "GUI_DIR", "INK", "MUTED", "ModelPicker",
           "OUTPUT_DIR", "ROOT", "Qt", "WEIGHTS_DIR", "apply_theme", "bgr_to_qimage", "cli_command", "hint",
           "open_folder", "page_header", "panel", "sample_image_dirs"]
