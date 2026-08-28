"""Standing policy hardware loop with fail-closed preflight."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration
from playground.nubzuki.controller import XboxController, axes_to_head_targets
from playground.nubzuki.hardware import ServoHardware
from playground.nubzuki.head_dynamics import HeadDynamicsProfile, HeadTrajectoryLimiter
from playground.nubzuki.policy import ObservationBuilder, StandingPolicy
from playground.nubzuki.sensors import FootContacts, ImuSensor


def _make_controller(control: str, host: str, web_port: int):
    if control == "phone":
        from playground.nubzuki.phone_controller import PhoneController
        controller = PhoneController(
            host=host, port=web_port, target_label="실물 로봇 제어"
        )
        print(f"\nOpen this on your phone, on the same network:\n    {controller.url}\n")
        return controller
    return XboxController()


def run_robot(policy_path: str, port: str, calibration_path: str | None,
              head_profile_path: str, imu_upside_down: bool = False,
              control: str = "joystick", host: str = "0.0.0.0",
              web_port: int = 8766) -> None:
    calibration = NubzukiCalibration(calibration_path)
    profile = HeadDynamicsProfile.load(head_profile_path, calibration)
    if not profile.measured:
        raise RuntimeError(
            "This head dynamics profile is the unmeasured simulation fallback. "
            "Run `nubzuki-standing identify-head` on the robot first."
        )
    policy = StandingPolicy(policy_path, calibration, profile)
    hardware = ServoHardware(calibration, port)
    controller = _make_controller(control, host, web_port)
    imu = ImuSensor(upside_down=imu_upside_down)
    feet = FootContacts()
    builder = ObservationBuilder()
    limiter = HeadTrajectoryLimiter(profile)
    armed = False
    previous_targets = np.zeros(14)
    a_was_pressed = False
    dt = 1.0 / calibration.control_frequency_hz
    try:
        hardware.disable_torque()
        hardware.preflight()
        print("Preflight OK. Press A to arm; press B for immediate torque off.")
        while True:
            started = time.monotonic()
            axes, a_pressed, b_pressed = controller.read()
            if b_pressed:
                print("Emergency stop")
                break
            if not armed:
                if a_pressed and not a_was_pressed:
                    low = int(calibration.data["runtime"]["low_kp"])
                    hardware.set_kps([low] * 14)
                    hardware.set_positions({name: 0.0 for name in calibration.joint_order})
                    time.sleep(1.0)
                    leg_kp = int(calibration.data["runtime"]["leg_kp"])
                    head_kp = int(calibration.data["runtime"]["head_kp"])
                    kps = [head_kp if name in HEAD_JOINTS else leg_kp for name in calibration.joint_order]
                    hardware.set_kps(kps)
                    armed = True
                    print("Policy armed")
                a_was_pressed = a_pressed
                time.sleep(max(0.0, dt - (time.monotonic() - started)))
                continue
            if not controller.fresh():
                raise RuntimeError("Joystick data is stale")
            raw_targets = axes_to_head_targets(axes, calibration, profile)
            head_targets = limiter.step(raw_targets)
            command = np.asarray([0.0, 0.0, 0.0] + [head_targets[name] for name in HEAD_JOINTS])
            imu_data = imu.read()
            qpos = hardware.read_positions()
            qvel = hardware.read_velocities()
            observation = builder.build(
                imu_data["gyro"], imu_data["accelerometer"], command,
                qpos, qvel, feet.read(),
            )
            action = policy.infer(observation)
            builder.advance(action)
            requested = action * calibration.action_scale_rad
            lowers = np.asarray([calibration.limits_rad(name)[0] for name in calibration.joint_order])
            uppers = np.asarray([calibration.limits_rad(name)[1] for name in calibration.joint_order])
            requested = np.clip(requested, lowers, uppers)
            max_delta = float(calibration.data["runtime"]["max_motor_velocity_rad_s"]) * dt
            requested = np.clip(requested, previous_targets - max_delta, previous_targets + max_delta)
            hardware.set_positions(dict(zip(calibration.joint_order, requested)))
            previous_targets = requested
            time.sleep(max(0.0, dt - (time.monotonic() - started)))
    finally:
        hardware.disable_torque()
        controller.close()
        feet.close()
