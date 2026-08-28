"""Measured joystick and head dynamics used by simulation and hardware."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration


AXES = ("left_x", "left_y", "right_x", "right_y")

# Used only to watch a policy in simulation before the robot exists. These are
# not measurements of anything; `identify-head` produces the real values and
# hardware refuses a profile that is not marked as measured.
FALLBACK_DEADZONE = 0.08
FALLBACK_VELOCITY_LIMIT_RAD_S = 2.0
FALLBACK_ACCELERATION_LIMIT_RAD_S2 = 10.0


@dataclass(frozen=True)
class JointDynamics:
    response_delay_s: float
    velocity_limit_rad_s: float
    acceleration_limit_rad_s2: float


class HeadDynamicsProfile:
    def __init__(self, data: dict, calibration: NubzukiCalibration):
        self.data = data
        self.measured = bool(data.get("measured", True))
        if data.get("calibration_sha256") != calibration.sha256:
            raise ValueError("Head dynamics profile does not match calibration")
        self.deadzone = {axis: float(data["joystick_deadzone"][axis]) for axis in AXES}
        self.joints = {
            name: JointDynamics(**{key: float(value) for key, value in data["joints"][name].items()})
            for name in HEAD_JOINTS
        }

    @classmethod
    def load(cls, path: str | Path, calibration: NubzukiCalibration):
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Head dynamics profile not found: {path}. Run `nubzuki-standing identify-head`."
            )
        return cls(json.loads(path.read_text(encoding="utf-8")), calibration)

    @classmethod
    def fallback(cls, calibration: NubzukiCalibration):
        """An unmeasured profile, for simulation only."""
        return cls(
            {
                "schema_version": 1,
                "measured": False,
                "calibration_sha256": calibration.sha256,
                "control_frequency_hz": calibration.control_frequency_hz,
                "joystick_deadzone": {axis: FALLBACK_DEADZONE for axis in AXES},
                "joints": {
                    name: {
                        "response_delay_s": 0.0,
                        "velocity_limit_rad_s": FALLBACK_VELOCITY_LIMIT_RAD_S,
                        "acceleration_limit_rad_s2": FALLBACK_ACCELERATION_LIMIT_RAD_S2,
                    }
                    for name in HEAD_JOINTS
                },
            },
            calibration,
        )

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class HeadTrajectoryLimiter:
    def __init__(self, profile: HeadDynamicsProfile, dt: float = 0.02):
        self.profile = profile
        self.dt = float(dt)
        self.position = {name: 0.0 for name in HEAD_JOINTS}
        self.velocity = {name: 0.0 for name in HEAD_JOINTS}

    def reset(self) -> None:
        for name in HEAD_JOINTS:
            self.position[name] = 0.0
            self.velocity[name] = 0.0

    def step(self, targets: dict[str, float]) -> dict[str, float]:
        result = {}
        for name in HEAD_JOINTS:
            dynamics = self.profile.joints[name]
            acceleration = dynamics.acceleration_limit_rad_s2
            error = float(targets[name]) - self.position[name]
            distance = abs(error)

            # Fastest approach speed that can still brake to a stop within the
            # remaining distance, discounted by the half step the discrete
            # integrator spends at the current speed. Without this the joint
            # runs at full speed until it crosses the target and then stops in
            # a single step, which is a deceleration far above the measured
            # limit and the one thing a head trajectory must not do.
            braking = -0.5 * acceleration * self.dt + np.sqrt(
                (0.5 * acceleration * self.dt) ** 2 + 2.0 * acceleration * distance
            )
            speed = min(
                dynamics.velocity_limit_rad_s, distance / self.dt, float(braking)
            )
            requested_velocity = np.sign(error) * speed

            step_change = acceleration * self.dt
            previous_velocity = self.velocity[name]
            dv = np.clip(
                requested_velocity - previous_velocity, -step_change, step_change
            )
            velocity = float(previous_velocity + dv)
            next_position = self.position[name] + velocity * self.dt

            # Settle exactly on the target once the remaining motion is within
            # a single acceleration-limited step, so a held stick does not idle
            # a hair away from its commanded angle. The guard is on the speed
            # the joint arrives with, not the speed it would leave with:
            # zeroing the velocity is itself a deceleration and has to fit in
            # the same budget as any other step.
            if (
                abs(float(targets[name]) - next_position) <= step_change * self.dt
                and abs(previous_velocity) <= step_change
            ):
                next_position = float(targets[name])
                velocity = 0.0

            self.position[name] = float(next_position)
            self.velocity[name] = velocity
            result[name] = self.position[name]
        return result

