"""Re-measure one joint's calibration offset after mechanical work.

The calibration maps servo_rad = direction * logical_rad + offset_rad. Take a
joint apart and reassemble it and the servo horn lands on a different tooth, so
the mapping is wrong by a fixed amount while everything else about the joint is
unchanged. This measures that amount and rewrites only that joint's
offset_deg, leaving the other thirteen alone.

Torque is released on the joint being measured so you can move it by hand, and
held on every other joint so the robot does not collapse while you work.

    ./.venv/bin/python -u scripts/zero_joint.py left_hip_pitch
    ./.venv/bin/python -u scripts/zero_joint.py left_hip_pitch --at-deg 0 --apply

Hold the joint at the reference angle you pass with --at-deg (0 by default,
meaning the joint's neutral) and press Enter. Compare against the opposite
joint: the tool prints both so a left/right mismatch is visible before you
commit to it.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration  # noqa: E402
from playground.nubzuki.hardware import ServoHardware  # noqa: E402
from playground.nubzuki.robot_runtime import park  # noqa: E402

MIRROR = {"left": "right", "right": "left"}


def opposite(name: str) -> str | None:
    side = name.split("_")[0]
    return name.replace(side, MIRROR[side], 1) if side in MIRROR else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("joint")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--calibration", default="config/nubzuki_calibration.json")
    parser.add_argument("--at-deg", type=float, default=0.0,
                        help="the logical angle the joint is being held at")
    parser.add_argument("--apply", action="store_true",
                        help="write the new offset; without it, only report")
    args = parser.parse_args()

    path = Path(args.calibration)
    calibration = NubzukiCalibration(str(path))
    if args.joint not in calibration.joint_order:
        raise SystemExit(f"Unknown joint: {args.joint}")

    hardware = ServoHardware(calibration, args.port)
    servo_id = calibration.servo_id(args.joint)
    direction = calibration.direction(args.joint)
    old_offset = float(calibration.joints[args.joint]["offset_deg"])

    print(f"Joint {args.joint}  (servo {servo_id}, direction {direction:+.0f})")
    print(f"Current offset: {old_offset:+.2f} deg")
    # Park first. Measuring against whatever pose the robot happened to be
    # left in gives a reference angle that means nothing, and the opposite
    # joint is only a usable comparison when it is somewhere known.
    print()
    print("Parking, then releasing this joint so you can move it by hand.")
    dt = 1.0 / calibration.control_frequency_hz
    hardware.disable_torque()
    hardware.preflight()
    start = hardware.read_positions()
    hardware.set_positions(dict(zip(calibration.joint_order, start)))
    low_kp = int(calibration.data["runtime"]["low_kp"])
    hardware.set_kps([low_kp] * 14)
    hardware.enable_torque()
    park(hardware, calibration, start, dt)
    # Stay at the low gain for the measurement. At the runtime gain the other
    # joints fight back while you are handling the robot, which moves the very
    # pose you are referencing against.
    hardware.disable_torque([args.joint])
    print(f"Parked and held at kp {low_kp}. {args.joint} is free.")

    twin = opposite(args.joint)
    print()
    print(f"Move {args.joint} to {args.at_deg:+.2f} deg and hold it there.")
    if twin:
        print(f"Match it against {twin}, which is held at its park angle of "
              f"{float(calibration.joints[twin]['park_deg']):+.2f} deg.")
    park_deg = float(calibration.joints[args.joint]["park_deg"])
    if abs(park_deg - args.at_deg) > 0.05:
        print(f"(This joint's own park angle is {park_deg:+.2f} deg, so a "
              f"symmetric stance is {park_deg:+.2f}, not {args.at_deg:+.2f}. "
              f"Pass --at-deg {park_deg:g} if you are matching the park pose.)")
    try:
        input("Press Enter when it is in position (Ctrl-C to abort): ")
    except KeyboardInterrupt:
        print("\nAborted; re-enabling torque.")
        hardware.enable_torque()
        park(hardware, calibration, hardware.read_positions(), dt)
        return

    raw = hardware.io.read_present_position([servo_id])[0]
    target = math.radians(args.at_deg)
    new_offset = math.degrees(float(raw) - direction * target)

    logical_now = calibration.servo_to_logical(args.joint, raw)
    print()
    print(f"  servo reads      {math.degrees(float(raw)):+8.2f} deg")
    print(f"  reads as logical {math.degrees(logical_now):+8.2f} deg "
          f"(should be {args.at_deg:+.2f})")
    print(f"  offset {old_offset:+.2f} -> {new_offset:+.2f} deg "
          f"(change {new_offset - old_offset:+.2f})")
    if twin:
        twin_now = hardware.read_positions()[calibration.joint_order.index(twin)]
        print(f"  {twin} is at {math.degrees(twin_now):+.2f} deg, "
              f"offset {float(calibration.joints[twin]['offset_deg']):+.2f}")

    hardware.enable_torque()
    park(hardware, calibration, hardware.read_positions(), dt)
    hardware.set_kps([
        int(calibration.data["runtime"]["head_kp"]) if name in HEAD_JOINTS
        else int(calibration.data["runtime"]["leg_kp"])
        for name in calibration.joint_order
    ])
    print("\nParked and holding at the runtime gain again.")

    if not args.apply:
        print("\nReport only. Re-run with --apply to write this offset.")
        return
    if abs(new_offset - old_offset) > 45.0:
        print(f"\nRefusing to write a {new_offset - old_offset:+.1f} deg change: "
              f"that is a wrong reference angle or the wrong joint, not a "
              f"reassembled horn.")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    backup = path.with_name(
        f"{path.stem}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak{path.suffix}"
    )
    shutil.copyfile(path, backup)
    data["joints"][args.joint]["offset_deg"] = round(new_offset, 3)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {path} (backup at {backup.name})")
    print("The calibration hash changed, so older MJX policies that pin it will")
    print("refuse to load. MJLab policies do not pin it and are unaffected.")


if __name__ == "__main__":
    main()
