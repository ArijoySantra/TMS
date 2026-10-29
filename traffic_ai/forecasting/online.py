"""Forecasters used online by the signal controller (and the video pipeline)."""
from __future__ import annotations

import json

import numpy as np
import torch

from traffic_ai.config import MODEL_DIR
from traffic_ai.forecasting.models import build_model
from traffic_ai.forecasting.train import TARGET_WINDOW


class NaiveForecaster:
    """Persistence: last 5-minute mean, repeated for every horizon."""

    def __init__(self, horizons):
        self.horizons = horizons

    def predict(self, history: np.ndarray) -> np.ndarray:
        if len(history) == 0:
            return np.zeros((len(self.horizons), 8))
        return np.repeat(history[-TARGET_WINDOW:].mean(0, keepdims=True), len(self.horizons), axis=0)


class TorchForecaster:
    """Loads a trained GRU / LSTM / PatchTST checkpoint. predict((T, C)) -> (H, C)."""

    def __init__(self, path=None):
        if path is None:
            best = json.loads((MODEL_DIR / "forecast_best.json").read_text())["best"]
            path = MODEL_DIR / f"forecast_{best}.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.history, self.horizons = ck["history"], ck["horizons"]
        self.columns = ck["columns"]
        self.scale = torch.tensor(ck["scale"])
        self.model = build_model(ck["name"], len(self.columns), len(self.horizons), self.history, ck["patch_len"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        torch.set_num_threads(1)

    def predict(self, history: np.ndarray) -> np.ndarray:
        h = np.asarray(history, dtype=np.float32)
        if len(h) == 0:
            return np.zeros((len(self.horizons), len(self.columns)), dtype=np.float32)
        if len(h) < self.history:  # pad the warm-up period with the earliest observation
            h = np.concatenate([np.repeat(h[:1], self.history - len(h), axis=0), h])
        x = torch.tensor(h[-self.history:]).unsqueeze(0) / self.scale
        with torch.no_grad():
            y = self.model(x)[0] * self.scale
        return np.clip(y.numpy(), 0, None)


def load_forecaster(cfg: dict):
    try:
        return TorchForecaster()
    except FileNotFoundError:
        print("  no trained forecaster found, using persistence forecast")
        return NaiveForecaster(cfg["forecast"]["horizons"])
