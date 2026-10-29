"""Train and evaluate traffic forecasters on per-minute series (from SUMO or from the video pipeline).

Targets: for each horizon h (minutes), the mean of each channel over the 5 minutes ending at t+h.
Channels: vehicle count and queue length for each approach.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from traffic_ai.config import DATA_DIR, MODEL_DIR, RESULTS_DIR, ensure_dirs
from traffic_ai.forecasting.models import build_model

TARGET_WINDOW = 5


def channel_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("count_")] + [c for c in df.columns if c.startswith("queue_")]


def make_windows(series: np.ndarray, history: int, horizons: list[int]):
    """series (T, C) -> X (N, history, C), Y (N, H, C)."""
    T = len(series)
    h_max = max(horizons)
    X, Y = [], []
    for t in range(history, T - h_max + 1):
        X.append(series[t - history:t])
        Y.append([series[t + h - TARGET_WINDOW:t + h].mean(0) for h in horizons])
    if not X:
        C = series.shape[1]
        return np.zeros((0, history, C)), np.zeros((0, len(horizons), C))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def split_runs(df: pd.DataFrame, history: int, horizons: list[int]):
    cols = channel_columns(df)
    runs = sorted(df.run.unique())
    n = len(runs)
    n_test, n_val = max(1, n // 6), max(1, n // 6)
    parts = {"train": runs[:n - n_val - n_test], "val": runs[n - n_val - n_test:n - n_test], "test": runs[n - n_test:]}
    out = {}
    for name, rs in parts.items():
        xs, ys = zip(*(make_windows(df[df.run == r].sort_values("minute")[cols].to_numpy(np.float32), history, horizons)
                       for r in rs))
        out[name] = (np.concatenate(xs), np.concatenate(ys))
    return out, cols


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mask = np.abs(y_true) >= 1.0  # MAPE only where the true value is at least one vehicle
    return {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "MAPE_%": float((np.abs(err[mask]) / np.abs(y_true[mask])).mean() * 100) if mask.any() else float("nan")}


def naive_forecast(X: np.ndarray, n_horizons: int) -> np.ndarray:
    """Persistence baseline: last 5-minute mean for every horizon."""
    return np.repeat(X[:, -TARGET_WINDOW:].mean(1, keepdims=True), n_horizons, axis=1)


def train_model(name: str, data: dict, cfg: dict, n_channels: int, verbose: bool = True):
    f = cfg["forecast"]
    torch.manual_seed(0)
    Xtr, Ytr = data["train"]
    scale = Xtr.reshape(-1, n_channels).std(0) + 1e-3
    model = build_model(name, n_channels, len(f["horizons"]), f["history"], f["patch_len"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, f["epochs"])
    loss_fn = nn.HuberLoss()
    s = torch.tensor(scale)
    tr = torch.utils.data.TensorDataset(torch.tensor(Xtr) / s, torch.tensor(Ytr) / s)
    loader = torch.utils.data.DataLoader(tr, batch_size=f["batch_size"], shuffle=True)
    Xv, Yv = (torch.tensor(a) / s for a in data["val"])
    best, best_state = float("inf"), None
    for ep in range(f["epochs"]):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), Yv).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == f["epochs"] - 1):
            print(f"    {name} epoch {ep:3d} val_loss {vl:.4f}")
    model.load_state_dict(best_state)
    return model, scale


def predict(model, scale, X: np.ndarray) -> np.ndarray:
    model.eval()
    s = torch.tensor(scale)
    with torch.no_grad():
        return (model(torch.tensor(X, dtype=torch.float32) / s) * s).numpy()


def run(cfg: dict, csv_path=None, models=("gru", "lstm", "patchtst")) -> pd.DataFrame:
    ensure_dirs()
    f = cfg["forecast"]
    df = pd.read_csv(csv_path or DATA_DIR / "traffic_timeseries.csv")
    data, cols = split_runs(df, f["history"], f["horizons"])
    C, H = len(cols), len(f["horizons"])
    print(f"  windows: train {len(data['train'][0])}, val {len(data['val'][0])}, test {len(data['test'][0])}")
    Xte, Yte = data["test"]
    rows, preds = [], {"naive": naive_forecast(Xte, H)}
    for name in models:
        model, scale = train_model(name, data, cfg, C)
        preds[name] = predict(model, scale, Xte)
        torch.save({"name": name, "state_dict": model.state_dict(), "scale": scale, "columns": cols,
                    "history": f["history"], "horizons": f["horizons"], "patch_len": f["patch_len"]},
                   MODEL_DIR / f"forecast_{name}.pt")
    for name, p in preds.items():
        for hi, h in enumerate(f["horizons"]):
            for kind in ("count", "queue"):
                idx = [i for i, c in enumerate(cols) if c.startswith(kind)]
                m = metrics(Yte[:, hi, idx], p[:, hi, idx])
                rows.append({"model": name, "horizon_min": h, "target": kind, **m})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "forecast_metrics.csv", index=False)
    # Save a test-run trace for the dashboard (count on the first approach, 15-min horizon)
    best = res[res.model != "naive"].groupby("model").MAE.mean().idxmin()
    trace = pd.DataFrame({"step": np.arange(len(Yte))})
    for hi, h in enumerate(f["horizons"]):
        for ci, c in enumerate(cols):
            trace[f"{c}_true_h{h}"] = Yte[:, hi, ci]
            trace[f"{c}_pred_h{h}"] = preds[best][:, hi, ci]
    trace.to_csv(RESULTS_DIR / "forecast_trace.csv", index=False)
    (MODEL_DIR / "forecast_best.json").write_text(json.dumps({"best": best}))
    print(res.groupby(["model", "target"])[["MAE", "RMSE", "MAPE_%"]].mean().round(3))
    return res
