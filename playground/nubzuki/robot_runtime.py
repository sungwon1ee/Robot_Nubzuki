"""Standing policy hardware loop that never drops the robot on its own."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration
from playground.nubzuki.controller import (
    XboxController,
    axes_to_head_targets,
    forward_velocity_command,
)
from playground.nubzuki.hardware import ServoHardware
from playground.nubzuki.head_dynamics import HeadDynamicsProfile, HeadTrajectoryLimiter
from playground.nubzuki.policy import ObservationBuilder, StandingPolicy
from playground.nubzuki.sensors import FootContacts, ImuSensor


# Fraction of the servo velocity limit used while lowering onto the park pose.
# Slow enough to watch and catch, fast enough to finish in well under a second.
PARK_SPEED_FRACTION = 0.2
PARK_TOLERANCE_RAD = 1e-4


def _make_controller(control: str, host: str, web_port: int):
    if control == "phone":
        from playground.nubzuki.phone_controller import PhoneController
        controller = PhoneController(
            host=host, port=web_port, target_label="실물 로봇 제어"
        )
        print(f"\nOpen this on your phone, on the same network:\n    {controller.url}\n")
        return controller
    return XboxController()


def park(hardware: ServoHardware, calibration: NubzukiCalibration,
         from_targets, dt: float) -> None:
    """Slew onto the calibrated park pose and keep holding it, torque on.

    A standing robot must not go limp: releasing fourteen servos at once is a
    fall, not a stop. Nothing the running loop can do cuts torque - not a lost
    controller, not an exception, not the B button. The servos are released
    only by cutting power, with the robot already supported.
    """
    order = calibration.joint_order
    target = np.asarray([calibration.park_rad(name) for name in order])
    position = np.asarray(from_targets, dtype=float).copy()
    max_delta = (
        float(calibration.data["runtime"]["max_motor_velocity_rad_s"])
        * dt
        * PARK_SPEED_FRACTION
    )
    # Bounded so a servo that stops acknowledging cannot spin here forever.
    for _ in range(int(5.0 / dt)):
        error = target - position
        if float(np.max(np.abs(error))) <= PARK_TOLERANCE_RAD:
            break
        position = position + np.clip(error, -max_delta, max_delta)
        hardware.set_positions(dict(zip(order, position)))
        time.sleep(dt)
    hardware.set_positions(dict(zip(order, target)))


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
    link_lost = False
    previous_targets = np.zeros(14)
    a_was_pressed = False
    dt = 1.0 / calibration.control_frequency_hz
    try:
        hardware.disable_torque()
        hardware.preflight()
        print("Preflight OK. Press A to arm; press B to park and stop.")
        while True:
            started = time.monotonic()
            axes, a_pressed, b_pressed = controller.read()
            if b_pressed:
                print("Stop requested")
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
            # A dropped controller is not a reason to stop. The sticks only steer
            # the head, and `PhoneController.read` already zeroes a stale sample
            # so the head recentres; the standing policy keeps the legs under
            # the robot. Raising here turned a Wi-Fi hiccup into a fall.
            input_fresh = controller.fresh()
            if input_fresh:
                if link_lost:
                    print("Controller link restored")
                    link_lost = False
            elif not link_lost:
                print(
                    "Controller link lost - head recentring, standing policy "
                    "still running. Ctrl-C or B to park and stop."
                )
                link_lost = True
            mode = controller.mode()
            head_axes = axes if mode == "head" else {name: 0.0 for name in axes}
            raw_targets = axes_to_head_targets(head_axes, calibration, profile)
            head_targets = limiter.step(raw_targets)
            forward = (
                forward_velocity_command(axes, mode, policy.metadata)
                if input_fresh else 0.0
            )
            command = np.asarray(
                [forward, 0.0, 0.0]
                + [head_targets[name] for name in HEAD_JOINTS]
            )
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
        try:
            if not armed:
                # Nothing was ever energised, so this only mirrors the state
                # the loop started in. It is unreachable once the policy arms.
                hardware.disable_torque()
            else:
                park(hardware, calibration, previous_targets, dt)
                print(
                    "Parked. Servos are still holding. Support the robot, then "
                    "cut power to release them."
                )
        except Exception as error:  # Never mask whatever brought us here.
            print(f"Park failed, holding last commanded pose: {error}")
        controller.close()
        feet.close()
