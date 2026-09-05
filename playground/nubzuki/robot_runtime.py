"""Standing policy hardware loop that never drops the robot on its own."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration
from playground.nubzuki.controller import (
    XboxController,
    axes_to_head_targets,
    forward_velocity_command,
    head_axes_for_mode,
    yaw_rate_command,
)
from playground.nubzuki.hardware import ServoHardware
from playground.nubzuki.head_dynamics import HeadDynamicsProfile, HeadTrajectoryLimiter
from playground.nubzuki.mjlab_policy import (
    MjlabObservationBuilder,
    MjlabPolicy,
    head_pose_command,
    twist_command,
)
from playground.nubzuki.policy import ObservationBuilder, StandingPolicy
from playground.nubzuki.sensors import FootContacts, ImuSensor


# Fraction of the servo velocity limit used while lowering onto the park pose.
# Slow enough to watch and catch, fast enough to finish in well under a second.
PARK_SPEED_FRACTION = 0.2
PARK_TOLERANCE_RAD = 1e-4


def _is_mjlab_policy(policy_path: str) -> bool:
    """Pick the policy family from the metadata rather than from a flag.

    The two families need different observations, a different joint order and
    a different action mapping; asking the operator to remember which one a
    file is invites exactly the mistake that ends with the robot on its face.
    """
    metadata = Path(policy_path).expanduser().with_suffix(".json")
    if not metadata.exists():
        raise FileNotFoundError(f"Policy metadata missing: {metadata}")
    try:
        return json.loads(metadata.read_text(encoding="utf-8")).get("framework") == "mjlab"
    except ValueError as error:
        raise RuntimeError(f"Cannot read {metadata}: {error}") from error


def _projected_gravity(imu_data: dict) -> np.ndarray:
    """Unit gravity direction in the body frame, as MJLab defines it."""
    gravity = np.asarray(imu_data.get("gravity"), dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise RuntimeError("IMU returned no usable gravity vector")
    norm = float(np.linalg.norm(gravity))
    if norm < 1.0:
        raise RuntimeError(f"Implausible gravity magnitude: {norm:.2f} m/s^2")
    # The BNO055 reports gravity with the accelerometer's convention: at rest
    # the vector points UP, away from gravity (+g on the up axis), which read
    # [0.07, 0.05, +1.00] standing in the park pose. MJLab's projected_gravity
    # is the direction gravity acts IN, so upright is [0, 0, -1]. Negate.
    return -gravity / norm


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
              web_port: int = 8766, debug_log_path: str | None = None) -> None:
    calibration = NubzukiCalibration(calibration_path)
    profile = HeadDynamicsProfile.load(head_profile_path, calibration)
    if not profile.measured:
        raise RuntimeError(
            "This head dynamics profile is the unmeasured simulation fallback. "
            "Run `nubzuki-standing identify-head` on the robot first."
        )
    is_mjlab = _is_mjlab_policy(policy_path)
    if is_mjlab:
        policy = MjlabPolicy(policy_path, calibration)
        builder = MjlabObservationBuilder(policy)
        print(
            f"MJLab policy: {policy.observation_size}D observation, "
            f"action scale {policy.action_scale}"
        )
    else:
        policy = StandingPolicy(policy_path, calibration, profile)
        builder = ObservationBuilder()
    hardware = ServoHardware(calibration, port)
    controller = _make_controller(control, host, web_port)
    imu = ImuSensor(upside_down=imu_upside_down)
    feet = FootContacts()
    limiter = HeadTrajectoryLimiter(profile)
    armed = False
    servos_energized = False
    link_lost = False
    previous_targets = np.zeros(14)
    a_was_pressed = False
    dt = 1.0 / calibration.control_frequency_hz
    debug_file = None
    debug_writer = None
    debug_rows = 0
    debug_started = time.monotonic()
    if debug_log_path:
        path = Path(debug_log_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        debug_file = path.open("w", newline="", encoding="utf-8")
        debug_writer = csv.writer(debug_file)
        debug_writer.writerow([
            "time_s", "forward_cmd_m_s", "yaw_cmd_rad_s",
            "gyro_x", "gyro_y", "gyro_z",
            "accel_x", "accel_y", "accel_z",
            "left_contact", "right_contact",
            "left_hip_roll_actual_rad", "right_hip_roll_actual_rad",
            "left_hip_roll_policy_target_rad", "right_hip_roll_policy_target_rad",
            "left_hip_roll_sent_target_rad", "right_hip_roll_sent_target_rad",
        ])
        debug_file.flush()
        print(f"Debug CSV: {path.resolve()}")
    try:
        hardware.disable_torque()
        hardware.preflight()
        # Never leave a connected robot limp while waiting for ARM.  Enter the
        # calibrated park pose gently at low gain, then hold it at runtime gain.
        previous_targets = hardware.read_positions()
        hardware.set_positions(dict(zip(calibration.joint_order, previous_targets)))
        low = int(calibration.data["runtime"]["low_kp"])
        hardware.set_kps([low] * 14)
        hardware.enable_torque()
        servos_energized = True
        park(hardware, calibration, previous_targets, dt)
        previous_targets = np.asarray([
            calibration.park_rad(name) for name in calibration.joint_order
        ])
        leg_kp = int(calibration.data["runtime"]["leg_kp"])
        head_kp = int(calibration.data["runtime"]["head_kp"])
        runtime_kps = [
            head_kp if name in HEAD_JOINTS else leg_kp
            for name in calibration.joint_order
        ]
        hardware.set_kps(runtime_kps)
        print(
            "Preflight OK. Holding park pose. "
            "Press A to arm; press B to park and stop."
        )
        while True:
            started = time.monotonic()
            axes, a_pressed, b_pressed = controller.read()
            if b_pressed:
                print("Stop requested")
                break
            if not armed:
                if a_pressed and not a_was_pressed:
                    if is_mjlab:
                        # Standing in park, gravity must read straight down in
                        # the body frame. A remapped or upside-down IMU shows
                        # up here as a tilt the policy would spend the whole
                        # run fighting, so refuse to arm on it.
                        gravity = _projected_gravity(imu.read())
                        if gravity[2] > -0.9:
                            raise RuntimeError(
                                f"Projected gravity is {np.round(gravity, 3).tolist()}, "
                                f"expected about [0, 0, -1] while standing. Check "
                                f"--imu-upside-down and the IMU axis remap before arming."
                            )
                        print(f"IMU check OK: projected gravity {np.round(gravity, 3).tolist()}")
                    hardware.set_kps([low] * 14)
                    hardware.set_positions({name: 0.0 for name in calibration.joint_order})
                    time.sleep(1.0)
                    hardware.set_kps(runtime_kps)
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
            head_axes = head_axes_for_mode(axes, mode)
            raw_targets = axes_to_head_targets(head_axes, calibration, profile)
            head_targets = limiter.step(raw_targets)
            if not input_fresh:
                forward, yaw_rate = 0.0, 0.0
            elif is_mjlab:
                forward, yaw_rate = twist_command(axes, mode, policy)
            else:
                forward = forward_velocity_command(axes, mode, policy.metadata)
                yaw_rate = yaw_rate_command(axes, mode, policy.metadata)
            imu_data = imu.read()
            qpos = hardware.read_positions()
            qvel = hardware.read_velocities()
            contacts = feet.read()
            if is_mjlab:
                observation = builder.build(
                    imu_data["gyro"], _projected_gravity(imu_data), qpos, qvel,
                    np.asarray([forward, 0.0, yaw_rate]),
                    head_pose_command(head_targets, policy),
                )
                action = policy.infer(observation)
                builder.advance(action)
                # MJLab actions are deltas around the training default pose,
                # not absolute targets.
                policy_target = policy.joint_targets(action)
            else:
                command = np.asarray(
                    [forward, 0.0, yaw_rate]
                    + [head_targets[name] for name in HEAD_JOINTS]
                )
                observation = builder.build(
                    imu_data["gyro"], imu_data["accelerometer"], command,
                    qpos, qvel, contacts,
                )
                action = policy.infer(observation)
                builder.advance(action)
                policy_target = action * calibration.action_scale_rad
            lowers = np.asarray([calibration.limits_rad(name)[0] for name in calibration.joint_order])
            uppers = np.asarray([calibration.limits_rad(name)[1] for name in calibration.joint_order])
            policy_target = np.clip(policy_target, lowers, uppers)
            max_delta = float(calibration.data["runtime"]["max_motor_velocity_rad_s"]) * dt
            requested = np.clip(
                policy_target, previous_targets - max_delta, previous_targets + max_delta
            )
            hardware.set_positions(dict(zip(calibration.joint_order, requested)))
            if debug_writer is not None:
                left_roll, right_roll = 1, 10
                debug_writer.writerow([
                    time.monotonic() - debug_started, forward, yaw_rate,
                    *imu_data["gyro"], *imu_data["accelerometer"], *contacts,
                    qpos[left_roll], qpos[right_roll],
                    policy_target[left_roll], policy_target[right_roll],
                    requested[left_roll], requested[right_roll],
                ])
                debug_rows += 1
                if debug_rows % calibration.control_frequency_hz == 0:
                    debug_file.flush()
            previous_targets = requested
            time.sleep(max(0.0, dt - (time.monotonic() - started)))
    finally:
        try:
            if not servos_energized:
                # A preflight failure happened before a position target was
                # written, so the initial torque-off state is still safe.
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
        if debug_file is not None:
            debug_file.close()
