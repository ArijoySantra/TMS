"""Short SUMO episodes: the environment, safety constraints and baselines behave."""
import numpy as np

from traffic_ai.config import load_config
from traffic_ai.forecasting.online import NaiveForecaster
from traffic_ai.sim.controllers import FixedTimeController, MaxPressureController
from traffic_ai.sim.env import TrafficSignalEnv


def _cfg():
    return load_config()


def test_env_spaces_and_min_green():
    cfg = _cfg()
    env = TrafficSignalEnv(cfg, use_forecast=True, forecaster=NaiveForecaster(cfg["forecast"]["horizons"]),
                           noisy=True, uncertainty_features=True, episode_seconds=300, use_libsumo=False)
    obs, _ = env.reset(seed=3)
    assert env.observation_space.contains(obs)
    # Ask to switch every step: switches must respect min-green + yellow
    done, steps = False, 0
    while not done:
        obs, r, done, _, _ = env.step(steps % env.action_space.n)
        assert env.observation_space.contains(obs)
        steps += 1
    m = env.episode_metrics()
    min_cycle = cfg["signal"]["min_green"] + cfg["signal"]["yellow_time"]
    assert m["switches_per_h"] <= 3600 / min_cycle + 1
    env.close()


def test_baselines_run_and_max_pressure_beats_fixed():
    cfg = _cfg()
    res = {}
    for ctrl in (FixedTimeController(), MaxPressureController()):
        env = TrafficSignalEnv(cfg, episode_seconds=1800, use_libsumo=False)
        env.reset(seed=7)
        ctrl.reset(env)
        done = False
        while not done:
            _, _, done, _, _ = env.step(ctrl.act(env))
        res[ctrl.name] = env.episode_metrics()
        env.close()
    assert res["Max-Pressure"]["mean_delay_s"] < res["Fixed-time"]["mean_delay_s"]
    assert all(np.isfinite(v["throughput_veh_h"]) and v["throughput_veh_h"] > 500 for v in res.values())
