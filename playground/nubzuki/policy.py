"""ONNX inference and strict standing policy contract checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from playground.nubzuki.calibration import NubzukiCalibration
from playground.nubzuki.head_dynamics import HeadDynamicsProfile


class StandingPolicy:
    def __init__(
        self, path: str | Path, calibration: NubzukiCalibration,
        head_profile: HeadDynamicsProfile | None = None,
        require_deployable: bool = True,
    ):
        import onnxruntime as ort
        self.path = Path(path).expanduser().resolve()
        metadata_path = self.path.with_suffix(".json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Policy metadata missing: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if require_deployable and self.metadata.get("deployable") is not True:
            raise RuntimeError("Smoke or unmarked policy cannot run on hardware")
        expected = {
            "observation_size": 85, "action_size": 14,
            "control_frequency_hz": 50, "calibration_sha256": calibration.sha256,
        }
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise RuntimeError(
                    f"Policy contract mismatch for {key}: {self.metadata.get(key)!r} != {value!r}"
                )
        if self.metadata.get("joint_order") != list(calibration.joint_order):
            raise RuntimeError("Policy joint order does not match calibration")
        if head_profile is not None:
            recorded = self.metadata.get("head_dynamics_sha256")
            if recorded is not None and recorded != head_profile.sha256:
                raise RuntimeError("Policy metadata and head dynamics profile do not match")
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        if input_shape[-1] != 85 or output_shape[-1] != 14:
            raise RuntimeError(f"ONNX ABI mismatch: {input_shape} -> {output_shape}")
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (85,) or not np.isfinite(observation).all():
            raise RuntimeError(f"Invalid policy observation: {observation.shape}")
        action = self.session.run(None, {self.input_name: observation[None, :]})[0][0]
        if action.shape != (14,) or not np.isfinite(action).all():
            raise RuntimeError("Policy returned invalid action")
        return np.asarray(action, dtype=float)


class ObservationBuilder:
    def __init__(self):
        self.last = np.zeros(14)
        self.last_last = np.zeros(14)
        self.last_last_last = np.zeros(14)

    def build(self, gyro, accelerometer, command, qpos, qvel, contacts) -> np.ndarray:
        obs = np.concatenate([
            np.asarray(gyro, dtype=float), np.asarray(accelerometer, dtype=float),
            np.asarray(command, dtype=float), np.asarray(qpos, dtype=float),
            np.asarray(qvel, dtype=float) * 0.05, self.last, self.last_last,
            self.last_last_last, np.asarray(contacts, dtype=float),
        ]).astype(np.float32)
        if obs.shape != (85,) or not np.isfinite(obs).all():
            raise RuntimeError(f"Standing observation is invalid: {obs.shape}")
        return obs

    def advance(self, action) -> None:
        self.last_last_last = self.last_last.copy()
        self.last_last = self.last.copy()
        self.last = np.asarray(action, dtype=float).copy()
