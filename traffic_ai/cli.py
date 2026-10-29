"""Command line entry point.  python -m traffic_ai.cli <command> [options]

  build-net        build the SUMO junction
  collect          simulate traffic series + collision data          (--what forecast|risk|all)
  train-forecast   GRU / LSTM / PatchTST forecasting
  train-risk       XGBoost accident-risk model
  train-rl         PPO controllers                                   (--variant ppo|ppo_forecast|ppo_uncertainty|all)
  evaluate         compare all controllers, clean and noisy cameras
  sim-all          everything above in order
  demo             run one controller in SUMO (add --gui on a desktop)
  images           detect road users in a folder of images
  compare-detectors  benchmark detectors (e.g. YOLO26 vs RT-DETR) on images; mAP if labels exist
  video            full camera pipeline on a video file / stream / webcam index
  dashboard        launch the Streamlit dashboard
  app              open the desktop application (also: TrafficAI.exe / Start_TrafficAI.bat)
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from traffic_ai.config import MODEL_DIR, ROOT, ensure_dirs, load_config

CONTROLLER_CHOICES = ["fixed", "max_pressure", "ppo", "ppo_forecast", "ppo_uncertainty"]


def build_controller(cfg, name: str, gui: bool = False, noisy: bool = False, **env_kw):
    from traffic_ai.forecasting.online import load_forecaster
    from traffic_ai.rl.train import VARIANTS
    from traffic_ai.sim.controllers import FixedTimeController, MaxPressureController, PPOController
    from traffic_ai.sim.env import TrafficSignalEnv
    if name in ("fixed", "max_pressure"):
        env = TrafficSignalEnv(cfg, gui=gui, noisy=noisy, **env_kw)
        return env, FixedTimeController() if name == "fixed" else MaxPressureController()
    v = VARIANTS[name]
    env = TrafficSignalEnv(cfg, gui=gui, noisy=noisy or v["noisy"], use_forecast=v["use_forecast"],
                           uncertainty_features=v["uncertainty_features"],
                           forecaster=load_forecaster(cfg) if v["use_forecast"] else None, **env_kw)
    return env, PPOController(str(MODEL_DIR / f"{name}.zip"), name)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML overriding configs/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-net")
    c = sub.add_parser("collect")
    c.add_argument("--what", choices=["forecast", "risk", "all"], default="all")
    c.add_argument("--risk-runs", type=int, default=40)
    f = sub.add_parser("train-forecast")
    f.add_argument("--csv", help="per-minute series (default: simulated); e.g. a video traffic_states.csv")
    f.add_argument("--models", nargs="+", default=["gru", "lstm", "patchtst"])
    sub.add_parser("train-risk").add_argument("--csv")
    r = sub.add_parser("train-rl")
    r.add_argument("--variant", default="all", choices=["all", "ppo", "ppo_forecast", "ppo_uncertainty"])
    r.add_argument("--timesteps", type=int)
    e = sub.add_parser("evaluate")
    e.add_argument("--episodes", type=int)
    sub.add_parser("sim-all")
    d = sub.add_parser("demo")
    d.add_argument("--controller", choices=CONTROLLER_CHOICES, default="ppo_forecast")
    d.add_argument("--gui", action="store_true")
    d.add_argument("--noisy", action="store_true")
    d.add_argument("--seed", type=int, default=1)
    i = sub.add_parser("images")
    i.add_argument("--folder", required=True)
    i.add_argument("--model")
    i.add_argument("--limit", type=int)
    cd = sub.add_parser("compare-detectors")
    cd.add_argument("--folder", required=True, help="images/<split> folder; labels/<split> used if present")
    cd.add_argument("--models", nargs="+", default=["yolo26s.pt", "rtdetr-l.pt"])
    cd.add_argument("--conf", type=float, default=0.25)
    cd.add_argument("--limit", type=int)
    v = sub.add_parser("video")
    v.add_argument("--source", required=True, help="video file, RTSP URL or webcam index")
    v.add_argument("--camera", help="camera JSON with approach polygons (configs/camera_example.json)")
    v.add_argument("--model", help="detector weights, e.g. best.pt from the UVH-26 notebook")
    v.add_argument("--videomae", help="fine-tuned VideoMAE folder for incident confirmation")
    v.add_argument("--max-seconds", type=float)
    v.add_argument("--rain", type=int, default=0)
    v.add_argument("--webhook", help="POST confirmed incidents to this URL")
    v.add_argument("--no-video", action="store_true")
    sub.add_parser("dashboard")
    sub.add_parser("app")
    a = p.parse_args(argv)
    if a.cmd == "app":
        from traffic_ai.gui.app import main as app_main
        return app_main()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    ensure_dirs()

    if a.cmd in ("build-net", "sim-all"):
        from traffic_ai.sim.scenario import build_network
        print("network:", build_network(cfg))
    if a.cmd in ("collect", "sim-all"):
        from traffic_ai.sim.collect import collect_forecast_data, collect_risk_data
        what = getattr(a, "what", "all")
        if what in ("forecast", "all"):
            collect_forecast_data(cfg)
        if what in ("risk", "all"):
            collect_risk_data(cfg, runs=getattr(a, "risk_runs", 40))
    if a.cmd in ("train-forecast", "sim-all"):
        from traffic_ai.forecasting.train import run
        run(cfg, getattr(a, "csv", None), getattr(a, "models", ["gru", "lstm", "patchtst"]))
    if a.cmd in ("train-risk", "sim-all"):
        from traffic_ai.risk.model import train
        train(cfg, getattr(a, "csv", None))
    if a.cmd in ("train-rl", "sim-all"):
        from traffic_ai.rl.train import VARIANTS, train
        variant = getattr(a, "variant", "all")
        for name in (VARIANTS if variant == "all" else [variant]):
            train(cfg, name, getattr(a, "timesteps", None))
    if a.cmd in ("evaluate", "sim-all"):
        from traffic_ai.rl.evaluate import evaluate
        evaluate(cfg, getattr(a, "episodes", None))
    if a.cmd == "demo":
        env, ctrl = build_controller(cfg, a.controller, gui=a.gui, noisy=a.noisy)
        env.reset(seed=a.seed)
        ctrl.reset(env)
        done = False
        while not done:
            _, _, done, _, _ = env.step(ctrl.act(env))
        print({k: round(v, 2) for k, v in env.episode_metrics().items()})
        env.close()
    if a.cmd == "images":
        from traffic_ai.vision.pipeline import run_images
        if a.model:
            cfg["vision"]["model"] = a.model
        df = run_images(a.folder, cfg, limit=a.limit)
        print(df.drop(columns="image").sum().to_string())
    if a.cmd == "compare-detectors":
        import pandas as pd
        from traffic_ai.config import OUTPUT_DIR
        from traffic_ai.vision.benchmark import compare
        r = compare(a.folder, a.models, conf=a.conf, imgsz=cfg["vision"]["imgsz"], limit=a.limit,
                    sample_dir=str(OUTPUT_DIR / "detector_samples"))
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(r["summary"].to_string(index=False))
            print(r["agreement"])
            if r["labelled_images"] == 0:
                print("No label files found: mAP needs labels/<split>/*.txt next to images/<split>.")
    if a.cmd == "video":
        from traffic_ai.vision.pipeline import run_video
        if a.model:
            cfg["vision"]["model"] = a.model
        print(run_video(a.source, cfg, a.camera, videomae=a.videomae, save_video=not a.no_video,
                        max_seconds=a.max_seconds, rain=a.rain, webhook=a.webhook))
    if a.cmd == "dashboard":
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py")])


if __name__ == "__main__":
    main()
