"""Measure joystick neutral noise and conservative head response limits."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration
from playground.nubzuki.controller import XboxController
from playground.nubzuki.hardware import ServoHardware


def procedure() -> str:
    return (
        "Support the robot so the feet cannot bear weight. The command samples the "
        "controller at rest for 3 seconds, enables only the four head joints at low "
        "KP, and moves one head joint at a time through +/-15% of its calibrated span."
    )


def _measure_deadzone(controller: XboxController) -> dict[str, float]:
    samples = {name: [] for name in ("left_x", "left_y", "right_x", "right_y")}
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        axes, _, b_pressed = controller.read()
        if b_pressed:
            raise RuntimeError("Identification aborted with B")
        for name, value in axes.items():
            samples[name].append(abs(value))
        time.sleep(0.02)
    return {name: float(np.quantile(values, 0.99)) for name, values in samples.items()}


def _measure_joint(
    hardware: ServoHardware, calibration: NubzukiCalibration, name: str,
) -> dict[str, float]:
    low, high = calibration.limits_rad(name)
    amplitude = 0.15 * (high - low)
    traces = []
    for _ in range(5):
        for target in (amplitude, -amplitude):
            started = time.monotonic()
            hardware.set_positions({name: calibration.clip(name, target)})
            times, positions = [], []
            while time.monotonic() - started < 0.75:
                position = hardware.read_positions()[calibration.joint_order.index(name)]
                times.append(time.monotonic() - started)
                positions.append(position)
                time.sleep(0.02)
            traces.append((np.asarray(times), np.asarray(positions), target))
    peak_velocities, peak_accelerations, delays = [], [], []
    for times, positions, target in traces:
        if len(times) < 5 or not np.isfinite(positions).all():
            raise RuntimeError(f"Invalid encoder trace for {name}")
        velocity = np.gradient(positions, times)
        acceleration = np.gradient(velocity, times)
        peak_velocities.append(float(np.quantile(np.abs(velocity), 0.95)))
        peak_accelerations.append(float(np.quantile(np.abs(acceleration), 0.95)))
        initial = positions[0]
        moved = np.flatnonzero(np.abs(positions - initial) >= abs(target - initial) * 0.05)
        delays.append(float(times[moved[0]]) if moved.size else float(times[-1]))
    velocity = 0.5 * float(np.median(peak_velocities))
    acceleration = 0.5 * float(np.median(peak_accelerations))
    if velocity <= 0 or acceleration <= 0:
        raise RuntimeError(f"No measurable response for {name}")
    return {
        "response_delay_s": float(np.median(delays)),
        "velocity_limit_rad_s": velocity,
        "acceleration_limit_rad_s2": acceleration,
    }


def identify_head(
    port: str, output: Path, calibration: NubzukiCalibration, assume_yes: bool = False,
) -> dict:
    print(procedure())
    if not assume_yes and input("Type IDENTIFY to continue: ").strip() != "IDENTIFY":
        raise RuntimeError("Identification cancelled")
    controller = XboxController()
    hardware = ServoHardware(calibration, port)
    head_kp = int(calibration.data["runtime"]["low_kp"])
    try:
        hardware.preflight()
        deadzone = _measure_deadzone(controller)
        hardware.disable_torque()
        hardware.set_joint_kps(HEAD_JOINTS, [head_kp] * len(HEAD_JOINTS))
        hardware.set_positions({name: 0.0 for name in HEAD_JOINTS})
        time.sleep(1.0)
        joints = {name: _measure_joint(hardware, calibration, name) for name in HEAD_JOINTS}
        hardware.set_positions({name: 0.0 for name in HEAD_JOINTS})
        time.sleep(1.0)
    finally:
        hardware.disable_torque()
        controller.close()
    profile = {
        "schema_version": 1,
        "calibration_sha256": calibration.sha256,
        "control_frequency_hz": calibration.control_frequency_hz,
        "joystick_deadzone": deadzone,
        "joints": joints,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    return profile

