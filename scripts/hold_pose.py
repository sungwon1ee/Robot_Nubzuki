"""Hold named joints at given angles so you can look at the robot.

A sweep proves a joint moves; it cannot prove which way. Commanding both sides
of a mirrored pair to the SAME angle does: in the model both hip pitches share
the axis "0 -1 0", so a positive command must swing both legs the same way. If
one goes forward and the other back, that joint's `direction` in the
calibration disagrees with the model, and every policy will drive it
backwards while a sweep still reports it healthy.

    ./.venv/bin/python -u scripts/hold_pose.py left_hip_pitch=10 right_hip_pitch=10
    ./.venv/bin/python -u scripts/hold_pose.py left_hip_pitch=-10 right_hip_pitch=-10

Everything not named holds its park angle. Ctrl-C parks and exits.
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
    parser.add_argument("assignments", nargs="+", metavar="joint=deg")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--calibration", default=None)
    args = parser.parse_args()

    calibration = NubzukiCalibration(args.calibration)
    order = list(calibration.joint_order)
    dt = 1.0 / calibration.control_frequency_hz
    runtime = calibration.data["runtime"]

    wanted: dict[str, float] = {}
    for item in args.assignments:
        if "=" not in item:
            raise SystemExit(f"Expected joint=deg, got {item!r}")
        name, value = item.split("=", 1)
        if name not in order:
            raise SystemExit(f"Unknown joint: {name}")
        angle = math.radians(float(value))
        lower, upper = calibration.limits_rad(name)
        if not lower <= angle <= upper:
            raise SystemExit(
                f"{name}={value} is outside its limits "
                f"({math.degrees(lower):.1f} .. {math.degrees(upper):.1f} deg)"
            )
        wanted[name] = angle

    hardware = ServoHardware(calibration, args.port)
    hardware.disable_torque()
    hardware.preflight()
    start = hardware.read_positions()
    hardware.set_positions(dict(zip(order, start)))
    hardware.set_kps([int(runtime["low_kp"])] * 14)
    hardware.enable_torque()
    park(hardware, calibration, start, dt)
    hardware.set_kps([
        int(runtime["head_kp"]) if name in HEAD_JOINTS else int(runtime["leg_kp"])
        for name in order
    ])

    pose = {name: calibration.park_rad(name) for name in order}
    pose.update(wanted)
    # Slew there rather than stepping, so nothing snaps under load.
    current = dict(zip(order, hardware.read_positions()))
    max_delta = float(runtime["max_motor_velocity_rad_s"]) * dt * 0.2
    for _ in range(int(3.0 / dt)):
        moved = False
        for name in order:
            error = pose[name] - current[name]
            if abs(error) > 1e-4:
                current[name] += max(-max_delta, min(max_delta, error))
                moved = True
        hardware.set_positions(current)
        if not moved:
            break
        time.sleep(dt)
    hardware.set_positions(pose)
    time.sleep(0.5)

    measured = hardware.read_positions()
    print(f"\n{'joint':<16}{'commanded':>11}{'measured':>10}")
    for name in wanted:
        print(f"{name:<16}{math.degrees(pose[name]):>11.2f}"
              f"{math.degrees(float(measured[order.index(name)])):>10.2f}")
    print("\nLook at the robot. Both sides of a mirrored pair given the same")
    print("angle must move the same way. Ctrl-C to park and exit.")

    try:
        while True:
            hardware.set_positions(pose)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nParking.")
    finally:
        try:
            park(hardware, calibration, hardware.read_positions(), dt)
            print("Parked and holding.")
        except Exception as error:
            print(f"Park failed: {error}")


if __name__ == "__main__":
    main()
