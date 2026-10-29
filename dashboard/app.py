"""Operator dashboard: signal-control comparison, forecasts, accident-risk hotspots and incidents.

Run:  streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from traffic_ai.config import OUTPUT_DIR, RESULTS_DIR  # noqa: E402

# Fixed categorical order: colour follows the controller, never its rank.
CONTROLLERS = ["Fixed-time", "Max-Pressure", "PPO", "PPO + forecast", "Uncertainty-aware PPO"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
CTRL_SCALE = alt.Scale(domain=CONTROLLERS, range=COLORS)
METRICS = {"mean_delay_s": "Mean delay (s/veh)", "mean_queue_veh": "Mean queue (veh)",
           "throughput_veh_h": "Throughput (veh/h)", "switches_per_h": "Phase switches per hour",
           "recovery_time_s": "Queue recovery time (s)"}

st.set_page_config(page_title="Traffic AI control room", layout="wide")
st.title("AI traffic signal control, predictive analytics and accident detection")


def read_csv(path):
    return pd.read_csv(path) if Path(path).exists() else None


tab_ctrl, tab_fc, tab_risk, tab_inc, tab_cam = st.tabs(
    ["Signal control", "Forecasting", "Accident risk", "Incidents", "Camera traffic states"])

with tab_ctrl:
    ep = read_csv(RESULTS_DIR / "controller_episodes.csv")
    if ep is None:
        st.info("Run `python -m traffic_ai.cli evaluate` to compare controllers.")
    else:
        cond = st.radio("Camera observations", ["clean", "noisy", "heavy"], horizontal=True,
                        format_func={"clean": "Clean", "noisy": "Noisy / missing approaches",
                                     "heavy": "Heavy noise (2x)"}.get)
        metric = st.selectbox("Metric", list(METRICS), format_func=METRICS.get)
        d = ep[ep.condition == cond]
        agg = d.groupby("controller")[metric].agg(["mean", "std"]).reset_index()
        present = [c for c in CONTROLLERS if c in set(agg.controller)]
        bars = alt.Chart(agg).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=36).encode(
            x=alt.X("controller:N", sort=present, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("mean:Q", title=METRICS[metric]),
            color=alt.Color("controller:N", scale=CTRL_SCALE, legend=None),
            tooltip=[alt.Tooltip("controller:N"), alt.Tooltip("mean:Q", format=".1f", title="mean"),
                     alt.Tooltip("std:Q", format=".1f", title="std across seeds")])
        err = alt.Chart(agg).mark_errorbar(color="#52514e").encode(
            x=alt.X("controller:N", sort=present), y="mean:Q", yError="std:Q")
        st.altair_chart((bars + err).properties(height=320), use_container_width=True)
        tr = read_csv(RESULTS_DIR / "controller_queue_traces.csv")
        if tr is not None:
            st.subheader("Total queue over one evaluation hour")
            t = tr[tr.condition == cond]
            line = alt.Chart(t).mark_line(strokeWidth=2).encode(
                x=alt.X("minute:Q", title="Minute"), y=alt.Y("queue:Q", title="Vehicles queued"),
                color=alt.Color("controller:N", scale=CTRL_SCALE, legend=alt.Legend(orient="top", title=None)),
                tooltip=["controller", "minute", alt.Tooltip("queue:Q", format=".0f")])
            st.altair_chart(line.properties(height=300), use_container_width=True)
        st.subheader("Summary table")
        st.dataframe(d.drop(columns=["seed", "condition"]).groupby("controller").mean().round(1)
                     .reindex(present), use_container_width=True)

with tab_fc:
    m = read_csv(RESULTS_DIR / "forecast_metrics.csv")
    if m is None:
        st.info("Run `python -m traffic_ai.cli train-forecast`.")
    else:
        target = st.radio("Target", ["count", "queue"], horizontal=True,
                          format_func=lambda x: "Vehicle count per minute" if x == "count" else "Queue length")
        mm = m[m.target == target]
        st.dataframe(mm.pivot_table(index="model", columns="horizon_min", values=["MAE", "RMSE", "MAPE_%"])
                     .round(2), use_container_width=True)
        tr = read_csv(RESULTS_DIR / "forecast_trace.csv")
        if tr is not None:
            c1, c2 = st.columns(2)
            app = c1.selectbox("Approach", ["N", "E", "S", "W"])
            hz = c2.selectbox("Horizon (minutes)", [5, 10, 15], index=2)
            col = f"{target}_{app}"
            d = tr[["step", f"{col}_true_h{hz}", f"{col}_pred_h{hz}"]].head(240)
            d.columns = ["step", "Actual", "Forecast"]
            long = d.melt("step", var_name="series", value_name="value")
            ch = alt.Chart(long).mark_line(strokeWidth=2).encode(
                x=alt.X("step:Q", title="Test window (minutes)"), y=alt.Y("value:Q", title=None),
                color=alt.Color("series:N", scale=alt.Scale(domain=["Actual", "Forecast"],
                                                            range=["#2a78d6", "#eb6834"]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["series", "step", alt.Tooltip("value:Q", format=".1f")])
            st.altair_chart(ch.properties(height=300), use_container_width=True)

with tab_risk:
    p = RESULTS_DIR / "risk_metrics.json"
    if not p.exists():
        st.info("Run `python -m traffic_ai.cli train-risk`.")
    else:
        rm = json.loads(p.read_text())
        cols = st.columns(4)
        for c, k in zip(cols, ["AUC", "F1", "recall", "precision"]):
            c.metric(k, f"{rm[k]:.2f}")
        hm = read_csv(RESULTS_DIR / "risk_heatmap.csv")
        if hm is not None:
            st.subheader("Hotspots and peak periods: mean predicted risk")
            long = hm.melt("approach", var_name="minute", value_name="risk")
            long["minute"] = long.minute.astype(int)
            heat = alt.Chart(long).mark_rect(stroke="white", strokeWidth=2).encode(
                x=alt.X("minute:O", title="Minute of the hour"), y=alt.Y("approach:N", title="Approach"),
                color=alt.Color("risk:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), title="Risk"),
                tooltip=["approach", "minute", alt.Tooltip("risk:Q", format=".2f")])
            st.altair_chart(heat.properties(height=200), use_container_width=True)
        fi = read_csv(RESULTS_DIR / "risk_feature_importance.csv")
        if fi is not None:
            fi.columns = ["feature", "importance"]
            st.altair_chart(alt.Chart(fi).mark_bar(color="#2a78d6", cornerRadiusEnd=4).encode(
                y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("importance:Q", title="XGBoost importance"),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")]).properties(height=260),
                use_container_width=True)

with tab_inc:
    files = sorted((OUTPUT_DIR / "video").glob("*/incidents/INC-*.json"))
    if not files:
        st.info("No incidents yet. Run `python -m traffic_ai.cli video --source <file>`.")
    for f in reversed(files):
        inc = json.loads(f.read_text())
        icon = "🚨" if inc["status"] == "confirmed" else "⚠️"
        with st.expander(f"{icon} {inc['incident_id']}  {inc['status'].replace('_', ' ')}  {inc['wall_time']}"):
            st.write(inc["summary"])
            if inc.get("snapshot_path") and Path(inc["snapshot_path"]).exists():
                st.image(inc["snapshot_path"])
            st.json({k: inc[k] for k in ("approach", "candidate_score", "confirm_probability", "cues", "vehicles")})

with tab_cam:
    runs = sorted((OUTPUT_DIR / "video").glob("*/traffic_states.csv"))
    if not runs:
        st.info("Process a video with `python -m traffic_ai.cli video --source <file> --camera configs/camera_example.json`.")
    else:
        run = st.selectbox("Video", runs, format_func=lambda p: p.parent.name)
        try:
            df = pd.read_csv(run)
        except pd.errors.EmptyDataError:
            df = None
        if df is None or df.empty:
            st.info("This video was shorter than one aggregation interval, so no traffic states were recorded.")
        else:
            st.dataframe(df, use_container_width=True)
