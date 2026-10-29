# PyInstaller recipe for the Windows app. Run through Build_TrafficAI_EXE.bat, not directly.
# Output: dist/TrafficAI/TrafficAI.exe  (one folder; configs/, outputs/ and samples/ are copied next to it)
import glob
import importlib.util
import os

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
datas = [(os.path.join(ROOT, "traffic_ai", "gui", "icon.png"), os.path.join("traffic_ai", "gui"))]
binaries = []
hiddenimports = collect_submodules("traffic_ai") + ["serial", "serial.tools.list_ports", "smtplib", "email.message"]

# Packages that ship data files, executables or plugins PyInstaller would otherwise miss.
for pkg in ["ultralytics", "sumo", "sumolib", "traci", "libsumo", "stable_baselines3", "lap", "gymnasium"]:
    try:
        d, b, h = collect_all(pkg)
    except BaseException as e:  # noqa: BLE001
        print(f"WARNING: could not collect {pkg}: {e}")
        continue
    datas += d
    binaries += b
    hiddenimports += h

# xgboost (its test helpers break collect_all) and torchvision (_C / image extensions loaded through
# torch.ops, invisible to the import scan): take every native library in the package folder explicitly.
datas += collect_data_files("xgboost", excludes=["testing/**"])
hiddenimports += collect_submodules("xgboost", filter=lambda n: ".testing" not in n)
for pkg in ["xgboost", "torchvision"]:
    base = os.path.dirname(importlib.util.find_spec(pkg).origin)
    for ext in ("*.dll", "*.so", "*.pyd", "*.dylib"):
        for f in glob.glob(os.path.join(base, "**", ext), recursive=True):
            binaries.append((f, os.path.join(pkg, os.path.relpath(os.path.dirname(f), base))))

# Version metadata that libraries look up at run time (importlib.metadata).
for dist in ["torch", "torchvision", "ultralytics", "ultralytics-thop", "stable_baselines3", "gymnasium",
             "numpy", "pandas", "opencv-python-headless", "opencv-python", "xgboost", "scikit-learn", "scipy",
             "matplotlib", "pillow", "pyyaml", "requests", "tqdm", "psutil", "polars", "lap", "eclipse-sumo",
             "libsumo", "traci", "sumolib", "PySide6", "joblib", "filelock", "fsspec", "sympy", "networkx"]:
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

excludes = ["tkinter", "streamlit", "altair", "IPython", "jupyter", "notebook", "pytest", "tensorboard",
            "tensorflow", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
            "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts",
            "PySide6.QtDataVisualization", "PySide6.QtGraphs", "PyQt5", "PyQt6"]

a = Analysis(
    [os.path.join(ROOT, "TrafficAI.pyw")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TrafficAI",
    console=False,
    icon=os.path.join(SPECPATH, "TrafficAI.ico"),
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="TrafficAI")
