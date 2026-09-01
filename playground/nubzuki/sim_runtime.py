"""Native MuJoCo ONNX simulation driven by a phone or a gamepad."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from playground.nubzuki import constants
from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration
from playground.nubzuki.controller import (
    axes_to_head_targets,
    forward_velocity_command,
    head_axes_for_mode,
    yaw_rate_command,
)
from playground.nubzuki.head_dynamics import HeadDynamicsProfile, HeadTrajectoryLimiter
from playground.nubzuki.policy import ObservationBuilder, StandingPolicy


def _sensor(model, data, name):
    sensor_id = model.sensor(name).id
    address = model.sensor_adr[sensor_id]
    dimension = model.sensor_dim[sensor_id]
    return np.asarray(data.sensordata[address:address + dimension]).copy()


def _contacts(model, data) -> np.ndarray:
    floor = model.geom("floor").id
    result = []
    for name in constants.FEET_GEOMS:
        foot = model.geom(name).id
        colliding = False
        for index in range(data.ncon):
            contact = data.contact[index]
            if {int(contact.geom1), int(contact.geom2)} == {floor, foot} and contact.dist < 0:
                colliding = True
                break
        result.append(colliding)
    return np.asarray(result, dtype=float)


def _load_profile(head_profile_path: str, calibration: NubzukiCalibration):
    """Return the head profile and whether the policy hash may be checked."""
    path = Path(head_profile_path).expanduser()
    if path.exists():
        return HeadDynamicsProfile.load(path, calibration), True
    print(
        f"No head dynamics profile at {path}; using unmeasured simulation defaults "
        f"(stick response is not your robot's). Run `identify-head` before hardware."
    )
    return HeadDynamicsProfile.fallback(calibration), False


def _make_controller(control: str, host: str, port: int):
    if control == "phone":
        from playground.nubzuki.phone_controller import PhoneController

        controller = PhoneController(host=host, port=port)
        print(f"\nOpen this on your phone, on the same network:\n    {controller.url}\n")
        return controller
    if control == "joystick":
        from playground.nubzuki.controller import XboxController

        return XboxController()
    raise ValueError(f"Unknown control source: {control}")


def run_simulation(
    policy_path: str,
    calibration_path: str | None,
    head_profile_path: str,
    control: str = "phone",
    host: str = "0.0.0.0",
    port: int = 8765,
    floor_friction: float | None = None,
    action_delay: float = 0.0,
) -> None:
    calibration = NubzukiCalibration(calibration_path)
    profile, may_check_hash = _load_profile(head_profile_path, calibration)
    policy = StandingPolicy(
        policy_path, calibration, profile if may_check_hash else None,
        require_deployable=False,
    )
    model_path = constants.simulation_xml()
    print(f"MuJoCo model: {model_path}")
    if model_path == constants.FLAT_TERRAIN_XML:
        print("Detailed Nubzuki CAD scene was not found; using lightweight training geometry.")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if floor_friction is not None:
        if not np.isfinite(floor_friction) or floor_friction < 0.0:
            raise ValueError("floor friction must be a finite non-negative value")
        floor_id = model.geom("floor").id
        model.geom_friction[floor_id, 0] = float(floor_friction)
        print(f"Floor sliding friction override: {floor_friction:g}")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    qpos_addresses = [model.jnt_qposadr[model.joint(name).id] for name in calibration.joint_order]
    qvel_addresses = [model.jnt_dofadr[model.joint(name).id] for name in calibration.joint_order]
    controller = _make_controller(control, host, port)
    limiter = HeadTrajectoryLimiter(profile)
    builder = ObservationBuilder()
    dt = 1.0 / calibration.control_frequency_hz
    if not np.isfinite(action_delay) or action_delay < 0.0:
        raise ValueError("action delay must be a finite non-negative value")
    action_delay_steps = int(round(action_delay * calibration.control_frequency_hz))
    action_queue = deque(
        (np.zeros(len(calibration.joint_order)) for _ in range(action_delay_steps + 1)),
        maxlen=action_delay_steps + 1,
    )
    if action_delay_steps:
        print(
            f"Action delay override: {action_delay_steps} control steps "
            f"({action_delay_steps * dt:.3f} s)"
        )
    announced = False
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            tracked_body_id = model.body("trunk").id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = tracked_body_id
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 180.0
            viewer.cam.elevation = -15.0
            while viewer.is_running():
                started = time.monotonic()
                axes, _, b_pressed = controller.read()
                if b_pressed:
                    break
                if not announced and controller.fresh():
                    print("Controller connected.")
                    announced = True
                mode = controller.mode()
                head_axes = head_axes_for_mode(axes, mode)
                desired = axes_to_head_targets(head_axes, calibration, profile)
                head = limiter.step(desired)
                forward = forward_velocity_command(axes, mode, policy.metadata)
                yaw_rate = yaw_rate_command(axes, mode, policy.metadata)
                command = np.asarray(
                    [forward, 0.0, yaw_rate] + [head[name] for name in HEAD_JOINTS]
                )
                qpos = np.asarray(data.qpos[qpos_addresses])
                qvel = np.asarray(data.qvel[qvel_addresses])
                obs = builder.build(
                    _sensor(model, data, constants.GYRO_SENSOR),
                    _sensor(model, data, constants.ACCELEROMETER_SENSOR),
                    command, qpos, qvel, _contacts(model, data),
                )
                action = policy.infer(obs)
                builder.advance(action)
                action_queue.append(action.copy())
                delayed_action = action_queue[0]
                data.ctrl[:] = delayed_action * calibration.action_scale_rad
                for _ in range(10):
                    mujoco.mj_step(model, data)
                # Mouse panning changes the viewer's look-at point (and can
                # switch camera mode), so restore only the tracking target.
                # Rotation and zoom remain user-adjustable.
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = tracked_body_id
                viewer.cam.lookat[:] = data.xpos[tracked_body_id]
                viewer.sync()
                time.sleep(max(0.0, dt - (time.monotonic() - started)))
    finally:
        controller.close()
