"""Traffic AI desktop application.

Run from source:   python -m traffic_ai.gui      (or double-click Start_TrafficAI.bat on Windows)
Packaged:          TrafficAI.exe                 (TrafficAI.exe --cli <command> runs the command line tool)
"""
from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def _redirect_output():
    """A windowed exe has no console (sys.stdout is None); keep library output in a log file instead."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    from traffic_ai.config import OUTPUT_DIR
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    f = open(log_dir / "app.log", "a", buffering=1, encoding="utf-8")  # noqa: SIM115
    sys.stdout = sys.stdout or f
    sys.stderr = sys.stderr or f


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    multiprocessing.freeze_support()
    _redirect_output()
    if argv and argv[0] == "--cli":  # packaged exe used as the command line tool
        import traceback

        from traffic_ai.cli import main as cli_main
        try:
            cli_main(argv[1:])
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else 0
        except Exception:
            traceback.print_exc()  # goes to outputs/logs/app.log from the windowed exe
            return 1
        return 0

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                                   QStackedWidget, QStatusBar, QVBoxLayout, QWidget)

    from traffic_ai import __version__
    from traffic_ai.config import ROOT, ensure_dirs
    from traffic_ai.gui.common import GUI_DIR, apply_theme

    ensure_dirs()
    os.chdir(ROOT)  # relative paths in configs and Ultralytics downloads resolve inside the project folder

    if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Traffic AI")
    app.setWindowIcon(QIcon(str(GUI_DIR / "icon.png")))
    apply_theme(app)

    from traffic_ai.gui.about import AboutPage
    from traffic_ai.gui.camera_setup import CameraSetupPage
    from traffic_ai.gui.live_control import LiveControlPage
    from traffic_ai.gui.detect import ImagePage, VideoPage
    from traffic_ai.gui.junction import JunctionPage
    from traffic_ai.gui.results import ResultsPage

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Traffic AI Control Center")
            self.resize(1360, 860)
            status = QStatusBar()
            self.setStatusBar(status)
            central = QWidget()
            lay = QHBoxLayout(central)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            side = QWidget()
            side.setObjectName("sidebar")
            side.setFixedWidth(220)
            sl = QVBoxLayout(side)
            sl.setContentsMargins(0, 18, 0, 12)
            brand = QLabel("  🚦 Traffic AI")
            brand.setObjectName("brand")
            sub = QLabel("    Adaptive signals · Forecasts\n    Accident detection")
            sub.setObjectName("brandSub")
            sl.addWidget(brand)
            sl.addWidget(sub)
            sl.addSpacing(16)
            self.nav = QListWidget()
            self.nav.setObjectName("nav")
            sl.addWidget(self.nav, 1)
            ver = QLabel(f"    Version {__version__}")
            ver.setObjectName("brandSub")
            sl.addWidget(ver)
            self.stack = QStackedWidget()
            self.pages = [("Live control", LiveControlPage(status, self.go)),
                          ("Camera setup", CameraSetupPage(status)),
                          ("Signal simulation", JunctionPage(status)),
                          ("Results dashboard", ResultsPage(status)),
                          ("Recorded video", VideoPage(status)), ("Photo detection", ImagePage(status)),
                          ("About", AboutPage(self))]
            for name, page in self.pages:
                self.nav.addItem(QListWidgetItem(name))
                self.stack.addWidget(page)
            self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
            self.nav.setCurrentRow(0)
            lay.addWidget(side)
            lay.addWidget(self.stack, 1)
            self.setCentralWidget(central)
            status.showMessage(f"Project folder: {ROOT}")

        def go(self, name):
            names = [n for n, _ in self.pages]
            self.nav.setCurrentRow(names.index(name))

        def closeEvent(self, e):
            for _, page in self.pages:
                if hasattr(page, "shutdown"):
                    page.shutdown()
            super().closeEvent(e)

    w = MainWindow()
    w.show()
    if os.environ.get("TRAFFIC_AI_SCREENSHOT"):  # used to check the build: save each page and quit
        from PySide6.QtCore import QTimer

        def shoot(i=0):
            if i:
                w.grab().save(os.path.join(os.environ["TRAFFIC_AI_SCREENSHOT"], f"page{i - 1}.png"))
            if i < len(w.pages):
                w.nav.setCurrentRow(i)
                QTimer.singleShot(700, lambda: shoot(i + 1))
            else:
                app.quit()
        QTimer.singleShot(500, shoot)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
