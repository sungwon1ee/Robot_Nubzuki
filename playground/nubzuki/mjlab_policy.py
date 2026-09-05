"""Run an MJLab/MicroDuck policy on the hardware.

The MJX policies this runtime was built for take an 85-D observation and emit
absolute scaled joint targets. An MJLab policy is a different animal and none
of it is interchangeable:

  * 61-D observation, ordered by the training environment's observation
    manager, with a 13-D command block (twist 3, head pose 4, body pose 6)
    instead of the old 7.
  * joint velocity enters unscaled, not multiplied by 0.05.
  * only ONE past action is fed back, not three.
  * actions are deltas around the training default pose, scaled by 0.25 --
    NOT absolute targets, and that default pose is not the calibrated park
    pose (its knee and ankle differ by a couple of degrees).
  * the joint order is the model's, which is left leg / right leg / head,
    while the calibration and every hardware call here use left leg / head /
    right leg. Feeding one to the other silently steers the neck with a hip
    command.

So none of this is guessed: `scripts/export_deploy.py` dumps the contract
straight out of the environment that trained the checkpoint, and this module
refuses to run on anything it cannot verify against that file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CONTRACT_KIND = "mjlab"

# Term names this module knows how to fill. A contract naming anything else is
# rejected rather than silently zero-filled.
_SUPPORTED_TERMS = {
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "head_command",
    "body_command",
}


class MjlabPolicy:
    """ONNX actor plus the deployment contract exported alongside it."""

    def __init__(self, path: str | Path, calibration):
        import onnxruntime as ort

        self.path = Path(path).expanduser().resolve()
        metadata_path = self.path.with_suffix(".json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Policy metadata missing: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if self.metadata.get("framework") != CONTRACT_KIND:
            raise RuntimeError(
                f"{metadata_path} is not an MJLab contract "
                f"(framework={self.metadata.get('framework')!r}). Use the MJX "
                f"policy path for that file."
            )
        if self.metadata.get("deployable") is not True:
            raise RuntimeError("Policy is not marked deployable")

        frequency = int(self.metadata["control_frequency_hz"])
        if frequency != calibration.control_frequency_hz:
            raise RuntimeError(
                f"Policy runs at {frequency} Hz, runtime at "
                f"{calibration.control_frequency_hz} Hz"
            )

        self.terms = [
            (str(term["name"]), int(term["dim"])) for term in self.metadata["observation"]
        ]
        unknown = {name for name, _ in self.terms} - _SUPPORTED_TERMS
        if unknown:
            raise RuntimeError(
                f"Contract has observation terms this runtime cannot build: "
                f"{sorted(unknown)}"
            )
        self.observation_size = sum(dim for _, dim in self.terms)
        if self.observation_size != int(self.metadata["observation_size"]):
            raise RuntimeError("Contract term widths do not sum to observation_size")

        self.joint_order = list(self.metadata["joint_order"])
        if sorted(self.joint_order) != sorted(calibration.joint_order):
            raise RuntimeError("Policy joint set does not match the calibration")
        self.action_scale = float(self.metadata["action_scale"])
        ranges = self.metadata.get("command_ranges", {})
        self.forward_range = tuple(ranges.get("twist_lin_vel_x", (0.0, 0.0)))
        self.yaw_rate_range = tuple(ranges.get("twist_ang_vel_z", (0.0, 0.0)))
        self.head_ranges = [tuple(r) for r in ranges.get("head_pose", [(0.0, 0.0)] * 4)]
        self.default_joint_pos = np.asarray(
            self.metadata["default_joint_pos"], dtype=float
        )
        if self.default_joint_pos.shape != (len(self.joint_order),):
            raise RuntimeError("default_joint_pos does not match joint_order")

        self.session = ort.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"]
        )
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        if input_shape[-1] != self.observation_size or output_shape[-1] != len(
            self.joint_order
        ):
            raise RuntimeError(
                f"ONNX ABI mismatch: {input_shape} -> {output_shape}, contract "
                f"says {self.observation_size} -> {len(self.joint_order)}"
            )
        self.input_name = self.session.get_inputs()[0].name

        # Runtime arrays are in calibration order; the policy speaks model
        # order. Build both permutations once.
        self.to_model = np.asarray(
            [list(calibration.joint_order).index(name) for name in self.joint_order]
        )
        self.to_runtime = np.asarray(
            [self.joint_order.index(name) for name in calibration.joint_order]
        )

    def infer(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observation_size,) or not np.isfinite(
            observation
        ).all():
            raise RuntimeError(f"Invalid policy observation: {observation.shape}")
        action = self.session.run(None, {self.input_name: observation[None, :]})[0][0]
        if action.shape != (len(self.joint_order),) or not np.isfinite(action).all():
            raise RuntimeError("Policy returned invalid action")
        return np.asarray(action, dtype=float)

    def joint_targets(self, action: np.ndarray) -> np.ndarray:
        """Model-order action -> calibration-order absolute joint targets."""
        target = self.default_joint_pos + self.action_scale * np.asarray(action)
        return target[self.to_runtime]


class MjlabObservationBuilder:
    """Assemble the 61-D vector in exactly the contract's term order."""

    def __init__(self, policy: MjlabPolicy):
        self.policy = policy
        self.last_action = np.zeros(len(policy.joint_order))

    def build(
        self,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        twist: np.ndarray,
        head_pose: np.ndarray,
    ) -> np.ndarray:
        """All joint arguments are in calibration order; commands are not."""
        policy = self.policy
        joint_pos_rel = np.asarray(qpos, dtype=float)[policy.to_model] - policy.default_joint_pos
        joint_vel = np.asarray(qvel, dtype=float)[policy.to_model]
        pieces = {
            "base_ang_vel": np.asarray(gyro, dtype=float),
            "projected_gravity": np.asarray(projected_gravity, dtype=float),
            "joint_pos": joint_pos_rel,
            "joint_vel": joint_vel,
            "actions": self.last_action,
            "command": np.asarray(twist, dtype=float),
            "head_command": np.asarray(head_pose, dtype=float),
            # The policy was trained with body pose commanded, but the vel task
            # keeps its tracking weight at zero and samples a range of zero.
            # Deployment holds it at the neutral it always saw.
            "body_command": np.zeros(6),
        }
        parts = []
        for name, dim in policy.terms:
            value = pieces[name]
            if value.shape != (dim,):
                raise RuntimeError(
                    f"Observation term {name} is {value.shape}, contract says ({dim},)"
                )
            parts.append(value)
        obs = np.concatenate(parts).astype(np.float32)
        if obs.shape != (policy.observation_size,) or not np.isfinite(obs).all():
            raise RuntimeError(f"MJLab observation is invalid: {obs.shape}")
        return obs

    def advance(self, action: np.ndarray) -> None:
        self.last_action = np.asarray(action, dtype=float).copy()


def _scale(value: float, limits: tuple[float, float]) -> float:
    low, high = float(limits[0]), float(limits[1])
    return value * (high if value >= 0 else abs(low))


def twist_command(axes: dict, mode: str, policy: MjlabPolicy) -> tuple[float, float]:
    """Stick deflection -> (forward m/s, yaw rad/s) inside the trained ranges.

    The MJX helpers in controller.py read a different metadata schema and
    return zero for an MJLab contract, which looks exactly like a robot that
    refuses to walk. This reads the ranges the checkpoint was actually
    exported with, so the command can never leave what the policy has seen.
    """
    from playground.nubzuki.controller import apply_deadzone

    if mode != "walk":
        return 0.0, 0.0
    forward = _scale(apply_deadzone(axes.get("left_y", 0.0), 0.1), policy.forward_range)
    yaw_rate = _scale(apply_deadzone(axes.get("left_x", 0.0), 0.1), policy.yaw_rate_range)
    return forward, yaw_rate


def head_pose_command(head_targets: dict[str, float], policy: MjlabPolicy) -> np.ndarray:
    """Absolute head joint targets -> deltas from the training default pose.

    The head command block is ordered (neck_pitch, head_pitch, head_yaw,
    head_roll) and is a delta, so the runtime's absolute head targets have to
    be referenced against the same default pose the actions are. The result is
    clamped to the ranges the policy was trained on: the head limiter works in
    the calibration's mechanical limits, which for an early walking checkpoint
    are far wider than anything the policy has ever been asked for.
    """
    order = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
    command = []
    for index, name in enumerate(order):
        default = float(policy.default_joint_pos[policy.joint_order.index(name)])
        low, high = policy.head_ranges[index]
        command.append(min(max(float(head_targets[name]) - default, low), high))
    return np.asarray(command)
