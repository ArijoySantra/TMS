"""Project paths and YAML config loading."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import yaml


def _root() -> Path:
    """Project folder holding configs/ and outputs/. In the packaged Windows app this is the folder
    next to TrafficAI.exe (code lives in _internal/, data stays writable beside the exe)."""
    if os.environ.get("TRAFFIC_AI_HOME"):
        return Path(os.environ["TRAFFIC_AI_HOME"]).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ROOT = _root()
CONFIG_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "outputs"
SUMO_DIR = OUTPUT_DIR / "sumo"
DATA_DIR = OUTPUT_DIR / "data"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load configs/default.yaml, optionally overlaid with another YAML file."""
    with open(CONFIG_DIR / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f))
    return cfg


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, SUMO_DIR, DATA_DIR, MODEL_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
