"""Move the robot to the calibrated park pose and hold it there.

Every other tool here parks as a side effect of doing something else. This
just parks, so it can be the first thing you run after powering on, before
zeroing a joint or looking at the robot.

    ./.venv/bin/python -u scripts/park.py
    ./.venv/bin/python -u scripts/park.py --kp low     # soft, easy to handle

Ctrl-C leaves the servos holding. They release only when power is cut, so
support the robot before doing that.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration  # noqa: E402
from playground.nubzuki.hardware import ServoHardware  # noqa: E402
from playground.nubzuki.robot_runtime import park  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--kp", choices=("runtime", "low"), default="runtime",
                        help="'low' keeps the robot compliant for handling")
    args = parser.parse_args()

    calibration = NubzukiCalibration(args.calibration)
    order = list(calibration.joint_order)
    dt = 1.0 / calibration.control_frequency_hz
    runtime = calibration.data["runtime"]
    hardware = ServoHardware(calibration, args.port)

    hardware.disable_torque()
    hardware.preflight()
    start = hardware.read_positions()
    hardware.set_positions(dict(zip(order, start)))
    hardware.set_kps([int(runtime["low_kp"])] * 14)
    hardware.enable_torque()
    park(hardware, calibration, start, dt)

    if args.kp == "runtime":
        gains = [
            int(runtime["head_kp"]) if name in HEAD_JOINTS else int(runtime["leg_kp"])
            for name in order
        ]
    else:
        gains = [int(runtime["low_kp"])] * 14
    hardware.set_kps(gains)

    measured = hardware.read_positions()
    print(f"\nParked at kp {args.kp} ({gains[0]} legs).")
    print(f"{'joint':<16}{'park':>9}{'actual':>9}{'error':>9}")
    for index, name in enumerate(order):
        target = math.degrees(calibration.park_rad(name))
        actual = math.degrees(float(measured[index]))
        print(f"{name:<16}{target:>9.2f}{actual:>9.2f}{actual - target:>9.2f}")
    print("\nHolding. Ctrl-C to exit (servos keep holding).")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStill holding. Support the robot before cutting power.")


if __name__ == "__main__":
    main()
