"""Mirror the simulated junction's signal state onto the ESP32 traffic-light model.

    python hardware/serial_bridge.py --port /dev/ttyUSB0 --controller ppo_forecast

Without --port it prints the states instead (dry run). Needs `pip install pyserial` for hardware.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traffic_ai.sim.scenario import APPROACHES  # noqa: E402


def approach_colors(env, state: str) -> str:
    """Collapse the per-link SUMO state into one colour per approach (N E S W)."""
    out = []
    for a in APPROACHES:
        chars = [c for (o, _, _), c in zip(env.link_info, state) if o == a]
        out.append("G" if "G" in chars else "Y" if "y" in chars else "R")
    return "".join(out)


def main():
    from traffic_ai.cli import build_controller
    from traffic_ai.config import load_config
    p = argparse.ArgumentParser()
    p.add_argument("--port")
    p.add_argument("--controller", default="max_pressure")
    p.add_argument("--speed", type=float, default=1.0, help="simulation seconds per real second")
    p.add_argument("--gui", action="store_true")
    a = p.parse_args()
    cfg = load_config()
    env, ctrl = build_controller(cfg, a.controller, gui=a.gui)
    ser = None
    if a.port:
        import serial
        ser = serial.Serial(a.port, 115200, timeout=1)
        time.sleep(2)
    env.reset(seed=1)
    ctrl.reset(env)
    last = None
    orig = env._advance

    def advance(seconds):  # send the light state every simulated second, paced in real time
        for _ in range(int(seconds)):
            orig(1)
            s = approach_colors(env, env.conn.trafficlight.getRedYellowGreenState(env.tls_id))
            nonlocal last
            if s != last:
                print(f"t={env.t:5d}s  N E S W = {' '.join(s)}")
                if ser:
                    ser.write((s + "\n").encode())
                last = s
            time.sleep(1.0 / a.speed)

    env._advance = advance
    done = False
    while not done:
        _, _, done, _, _ = env.step(ctrl.act(env))


if __name__ == "__main__":
    main()
