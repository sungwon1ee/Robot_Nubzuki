"""Single source of truth for Nubzuki simulation and hardware calibration."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


EXPECTED_JOINT_ORDER = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
    "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
    "right_ankle",
)
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_calibration_path() -> Path:
    configured = os.environ.get("NUBZUKI_CALIBRATION")
    if configured:
        return Path(configured).expanduser().resolve()
    return repository_root() / "config" / "nubzuki_calibration.json"


class NubzukiCalibration:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser().resolve() if path else default_calibration_path()
        self.data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        self.joint_order = tuple(self.data["joint_order"])
        self.command_order = tuple(self.data["command_order"])
        self.joints = self.data["joints"]
        self.validate()

    def validate(self) -> None:
        if self.joint_order != EXPECTED_JOINT_ORDER:
            raise ValueError(f"Unexpected joint order: {self.joint_order}")
        if set(self.joint_order) != set(self.joints):
            raise ValueError("Calibration joint_order and joints do not match")
        servo_ids = [self.servo_id(name) for name in self.joint_order]
        if len(servo_ids) != len(set(servo_ids)):
            raise ValueError("Calibration contains duplicate servo IDs")
        if self.observation_size != 85 or self.privileged_observation_size != 153:
            raise ValueError("Policy must use observation ABI 85/153")
        if self.action_size != 14 or self.control_frequency_hz != 50:
            raise ValueError("Standing policy must use 14 actions at 50 Hz")
        for name in self.joint_order:
            low, high = self.limits_rad(name)
            if not low < high:
                raise ValueError(f"Invalid limits for {name}: {low}, {high}")
            park = self.park_rad(name)
            if not low <= park <= high:
                raise ValueError(f"Park pose for {name} is outside its limits: {park}")

    @property
    def observation_size(self) -> int:
        return int(self.data["observation_size"])

    @property
    def privileged_observation_size(self) -> int:
        return int(self.data["privileged_observation_size"])

    @property
    def action_size(self) -> int:
        return int(self.data["action_size"])

    @property
    def control_frequency_hz(self) -> int:
        return int(self.data["control_frequency_hz"])

    @property
    def action_scale_rad(self) -> float:
        return float(self.data["policy"]["action_scale_rad"])

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def head_sha256(self) -> str:
        """Hash of only what head dynamics actually depend on.

        Re-zeroing a leg changes the whole-file hash but cannot change how the
        neck responds to a stick. Pinning the whole file made every leg
        calibration invalidate a head measurement that was still perfectly
        good.
        """
        subset = {
            "control_frequency_hz": self.data["control_frequency_hz"],
            "joints": {name: self.data["joints"][name] for name in HEAD_JOINTS},
        }
        canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def servo_id(self, name: str) -> int:
        return int(self.joints[name]["servo_id"])

    def direction(self, name: str) -> float:
        return float(self.joints[name]["direction"])

    def offset_rad(self, name: str) -> float:
        return math.radians(float(self.joints[name]["offset_deg"]))

    def limits_rad(self, name: str) -> tuple[float, float]:
        low, high = self.joints[name]["limits_deg"]
        return math.radians(float(low)), math.radians(float(high))

    def park_rad(self, name: str) -> float:
        """The supported pose the robot is lowered onto when the loop exits."""
        return math.radians(float(self.joints[name]["park_deg"]))

    def logical_to_servo(self, name: str, logical_rad: float) -> float:
        return self.direction(name) * float(logical_rad) + self.offset_rad(name)

    def servo_to_logical(self, name: str, servo_rad: float) -> float:
        return (float(servo_rad) - self.offset_rad(name)) / self.direction(name)

    def logical_velocity_from_servo(self, name: str, velocity: float) -> float:
        return float(velocity) / self.direction(name)

    def clip(self, name: str, value: float) -> float:
        low, high = self.limits_rad(name)
        return min(max(float(value), low), high)
