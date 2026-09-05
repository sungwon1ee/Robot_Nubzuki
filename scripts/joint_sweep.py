"""Move one joint at a time and report what actually moved.

Runs on the robot with no policy involved, so it separates the two reasons a
leg stays still: the policy never commanded it (a joint-order or mapping bug,
which shows up here as every joint moving fine) or the servo does not follow
(which shows up as that joint's measured travel being a fraction of what was
asked). Suspend the robot first - it will not hold a stance while a leg joint
is being swept.

    ./.venv/bin/python -u scripts/joint_sweep.py --port /dev/ttyACM0
    ./.venv/bin/python -u scripts/joint_sweep.py --joints left_knee left_ankle
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration  # noqa: E402
from playground.nubzuki.hardware import ServoHardware  # noqa: E402
from playground.nubzuki.robot_runtime import park  # noqa: E402

# A joint that follows less than this fraction of the commanded swing is not
# tracking. Servo backlash and encoder resolution eat a few percent; a dead or
# unpowered joint reads near zero.
FOLLOW_THRESHOLD = 0.35


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--joints", nargs="*", default=None)
    parser.add_argument("--amplitude-deg", type=float, default=4.0)
    parser.add_argument("--settle", type=float, default=0.6,
                        help="seconds to wait after each command")
    args = parser.parse_args()

    calibration = NubzukiCalibration(args.calibration)
    order = list(calibration.joint_order)
    targets = args.joints or order
    unknown = [name for name in targets if name not in order]
    if unknown:
        raise SystemExit(f"Unknown joints: {unknown}")

    amplitude = math.radians(args.amplitude_deg)
    dt = 1.0 / calibration.control_frequency_hz
    hardware = ServoHardware(calibration, args.port)

    print(f"Sweeping {len(targets)} joint(s) by +/-{args.amplitude_deg:g} deg.")
    print("Suspend the robot. Ctrl-C stops and parks.\n")

    results = []
    try:
        hardware.disable_torque()
        hardware.preflight()
        start = hardware.read_positions()
        hardware.set_positions(dict(zip(order, start)))
        low = int(calibration.data["runtime"]["low_kp"])
        hardware.set_kps([low] * 14)
        hardware.enable_torque()
        park(hardware, calibration, start, dt)
        park_pose = np.asarray([calibration.park_rad(name) for name in order])
        leg_kp = int(calibration.data["runtime"]["leg_kp"])
        head_kp = int(calibration.data["runtime"]["head_kp"])
        hardware.set_kps([
            head_kp if name in HEAD_JOINTS else leg_kp for name in order
        ])
        time.sleep(0.5)

        for name in targets:
            index = order.index(name)
            lower, upper = calibration.limits_rad(name)
            centre = float(park_pose[index])
            high = min(centre + amplitude, upper)
            low_target = max(centre - amplitude, lower)
            commanded = high - low_target
            if commanded < math.radians(0.5):
                results.append((name, 0.0, 0.0, "no room inside joint limits"))
                continue

            def hold(value: float) -> float:
                pose = park_pose.copy()
                pose[index] = value
                hardware.set_positions(dict(zip(order, pose)))
                time.sleep(args.settle)
                return float(hardware.read_positions()[index])

            measured_high = hold(high)
            measured_low = hold(low_target)
            hold(centre)
            measured = abs(measured_high - measured_low)
            ratio = measured / commanded if commanded else 0.0
            note = "ok" if ratio >= FOLLOW_THRESHOLD else "NOT FOLLOWING"
            results.append((name, math.degrees(commanded), math.degrees(measured), note))
            print(
                f"  {name:<16} commanded {math.degrees(commanded):5.2f} deg  "
                f"measured {math.degrees(measured):5.2f} deg  ({ratio*100:3.0f}%)  {note}"
            )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            park(hardware, calibration, hardware.read_positions(), dt)
            print("\nParked. Servos still holding; support the robot before cutting power.")
        except Exception as error:
            print(f"Park failed: {error}")

    failed = [r for r in results if r[3] == "NOT FOLLOWING"]
    if failed:
        print(f"\n{len(failed)} joint(s) did not follow: {[r[0] for r in failed]}")
        print("That is a servo, wiring or power problem, not the policy.")
    elif results:
        print("\nEvery joint followed its command. If a leg stays still while the")
        print("policy runs, the policy is not commanding it - check joint order.")


if __name__ == "__main__":
    main()
