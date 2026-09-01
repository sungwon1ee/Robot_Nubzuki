"""Minimal calibrated Feetech hardware interface with lazy dependencies."""

from __future__ import annotations

import numpy as np

from playground.nubzuki.calibration import NubzukiCalibration


class ServoHardware:
    def __init__(self, calibration: NubzukiCalibration, port: str):
        try:
            import rustypot
        except ImportError as error:
            raise RuntimeError("Install the `robot` extra on the Raspberry Pi") from error
        self.calibration = calibration
        self.names = calibration.joint_order
        self.ids = [calibration.servo_id(name) for name in self.names]
        self.io = rustypot.feetech(port, 1_000_000)

    def set_kps(self, values) -> None:
        self.io.set_kps(self.ids, list(values))

    def set_joint_kps(self, names, values) -> None:
        ids = [self.calibration.servo_id(name) for name in names]
        self.io.set_kps(ids, list(values))

    def disable_torque(self, names=None) -> None:
        ids = self.ids if names is None else [self.calibration.servo_id(name) for name in names]
        self.io.disable_torque(ids)

    def enable_torque(self, names=None) -> None:
        ids = self.ids if names is None else [self.calibration.servo_id(name) for name in names]
        self.io.enable_torque(ids)

    def set_positions(self, positions: dict[str, float]) -> None:
        names = tuple(positions)
        ids = [self.calibration.servo_id(name) for name in names]
        logical = [self.calibration.clip(name, positions[name]) for name in names]
        servo = [self.calibration.logical_to_servo(name, value) for name, value in zip(names, logical)]
        self.io.write_goal_position(ids, servo)

    def read_positions(self) -> np.ndarray:
        values = self.io.read_present_position(self.ids)
        return np.asarray([
            self.calibration.servo_to_logical(name, value)
            for name, value in zip(self.names, values)
        ], dtype=float)

    def read_velocities(self) -> np.ndarray:
        values = self.io.read_present_velocity(self.ids)
        return np.asarray([
            self.calibration.logical_velocity_from_servo(name, value)
            for name, value in zip(self.names, values)
        ], dtype=float)

    def preflight(self) -> None:
        positions = self.read_positions()
        velocities = self.read_velocities()
        if positions.shape != (14,) or velocities.shape != (14,):
            raise RuntimeError("Servo readback shape mismatch")
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise RuntimeError("Servo readback contains non-finite values")
