"""Play an MJLab Nubzuki policy with the existing phone web joystick.

The phone server itself is untouched: this reuses `PhoneController` and its
page from the `playground` package next door. Only the wiring is new — instead
of feeding an ONNX standing policy, the stick values are written straight into
the MJLab command manager's `twist` and `head_pose` terms every control step.

Run (macOS needs mjpython for the native viewer):

    cd .../Robot_Nubzuki/mjlab_nubzuki
    uv run mjpython src/mjlab_nubzuki/play_phone.py \
      Mjlab-Velocity-Flat-BAM-Nubzuki \
      --checkpoint-file checkpoints/model_270.pt \
      --num-envs 1 --viewer native
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import tyro

# The phone server lives in the sibling Robot_Nubzuki playground package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from playground.nubzuki.controller import apply_deadzone, scale_axis  # noqa: E402
from playground.nubzuki.phone_controller import PhoneController  # noqa: E402


# --------------------------------------------------------------------------- #
# Command ranges are read from the training config of the checkpoint being
# played, not hard-coded here: a policy trained with lin_vel_x in (0.04, 0.18)
# must not be handed 0.25 just because the task config has moved on since.
# These are only the fallback for a checkpoint with no params/env.yaml beside
# it, and the run says so out loud when it uses them.
FALLBACK_FORWARD_RANGE = (0.0, 0.0)
FALLBACK_YAW_RATE_RANGE = (0.0, 0.0)
FALLBACK_HEAD_RANGES = ((0.0, 0.0),) * 4
DEADZONE = 0.1


class PhoneCommandSource:
    """Turns one phone sample into (twist[3], head_pose[4])."""

    def __init__(self, controller: PhoneController, ranges: dict):
        self.controller = controller
        self.forward_range = tuple(ranges["twist_lin_vel_x"])
        self.yaw_rate_range = tuple(ranges["twist_ang_vel_z"])
        self.head_ranges = [tuple(r) for r in ranges["head_pose"]]
        self._announced = False
        self._cache: tuple[list[float], list[float]] | None = None
        self._cached_at = 0.0

    def sample(self) -> tuple[list[float], list[float]]:
        # Both command terms ask within the same control step; one read keeps
        # the twist and the head command from coming out of different samples.
        now = time.monotonic()
        if self._cache is None or now - self._cached_at > 0.005:
            self._cache = self._read()
            self._cached_at = now
        return self._cache

    def _read(self) -> tuple[list[float], list[float]]:
        axes, _a, _b = self.controller.read()
        if not self.controller.fresh():
            # Phone out of range / page closed: full stop, head back to park.
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
        if not self._announced:
            print("Phone controller connected.")
            self._announced = True

        mode = self.controller.mode()
        stick = {name: apply_deadzone(value, DEADZONE) for name, value in axes.items()}

        if mode == "head":
            # Head-only: both sticks drive the head, the legs hold still.
            twist = [0.0, 0.0, 0.0]
            head = [
                scale_axis(stick["right_y"], self.head_ranges[0]),   # neck_pitch
                scale_axis(-stick["left_y"], self.head_ranges[1]),   # head_pitch
                scale_axis(stick["left_x"], self.head_ranges[2]),    # head_yaw
                scale_axis(stick["right_x"], self.head_ranges[3]),   # head_roll
            ]
            return twist, head

        # Walk mode: left stick = locomotion, right stick = small head yaw/pitch.
        forward = scale_axis(stick["left_y"], self.forward_range)
        yaw_rate = scale_axis(stick["left_x"], self.yaw_rate_range)
        head = [
            0.0,
            scale_axis(-stick["right_y"], self.head_ranges[1]),  # head_pitch
            scale_axis(stick["right_x"], self.head_ranges[2]),   # head_yaw
            0.0,
        ]
        return [forward, 0.0, yaw_rate], head


def install_command_overrides(source: PhoneCommandSource) -> None:
    """Patch the two command terms so every step reads the phone instead of
    the training-time random sampler. `_update_command` runs after any
    resample, so whatever the sampler drew is overwritten before it is used."""
    from mjlab_microduck.tasks.mdp import (
        UniformPoseCommand,
        VelocityCommandCommandOnly,
    )

    def twist_update(self) -> None:
        twist, _head = source.sample()
        values = torch.tensor(twist, device=self.device, dtype=self.vel_command_b.dtype)
        self.vel_command_b[:] = values
        self.vel_command_w[:] = values
        # Standing envs would otherwise be zeroed by the base implementation.
        self.is_standing_env[:] = False

    def head_update(self) -> None:
        # UniformPoseCommand also backs body_pose (6D); only the 4D head
        # command is ours to drive.
        if self._command.shape[1] != len(source.head_ranges):
            return
        _twist, head = source.sample()
        self._command[:] = torch.tensor(
            head, device=self.device, dtype=self._command.dtype
        )

    VelocityCommandCommandOnly._update_command = twist_update
    UniformPoseCommand._update_command = head_update


def main() -> None:
    import mjlab.tasks  # noqa: F401  (populates the task registry)
    import mjlab_nubzuki.tasks  # noqa: F401
    from mjlab.scripts.play import PlayConfig, run_play
    from mjlab.tasks.registry import list_tasks

    import mjlab

    chosen_task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args, phone_args = _split_phone_args(remaining)
    play_cfg = tyro.cli(
        PlayConfig,
        args=args,
        default=PlayConfig(),
        prog=f"{sys.argv[0]} {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    ranges = _command_ranges(play_cfg.checkpoint_file)
    controller = PhoneController(
        host=phone_args["host"], port=phone_args["port"], target_label="MJLab 시뮬레이터"
    )
    print(f"\nOpen this on your phone, on the same network:\n    {controller.url}\n")
    install_command_overrides(PhoneCommandSource(controller, ranges))
    try:
        run_play(chosen_task, play_cfg)
    finally:
        controller.close()


def _command_ranges(checkpoint_file: str | None) -> dict:
    """What this checkpoint was trained to accept, straight from its run."""
    from mjlab_nubzuki.run_config import find_env_yaml, trained_command_ranges

    env_yaml = find_env_yaml(Path(checkpoint_file)) if checkpoint_file else None
    if env_yaml is not None:
        ranges = trained_command_ranges(env_yaml)
        print(f"Command ranges from {env_yaml}:")
        for key in ("twist_lin_vel_x", "twist_ang_vel_z"):
            print(f"  {key}: {ranges[key]}")
        return ranges
    print(
        "WARNING: no params/env.yaml beside this checkpoint, so the ranges it\n"
        "         was trained with are unknown. The sticks will command zero.\n"
        "         Copy the run's params/ directory next to the checkpoint."
    )
    return {
        "twist_lin_vel_x": list(FALLBACK_FORWARD_RANGE),
        "twist_ang_vel_z": list(FALLBACK_YAW_RATE_RANGE),
        "head_pose": [list(r) for r in FALLBACK_HEAD_RANGES],
    }


def _split_phone_args(argv: list[str]) -> tuple[list[str], dict]:
    """Pull --host/--port out before tyro sees the play options."""
    phone = {"host": "0.0.0.0", "port": 8765}
    rest: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("--host", "--port") or token.startswith(("--host=", "--port=")):
            if "=" in token:
                key, value = token.split("=", 1)
                index += 1
            else:
                key, value = token, argv[index + 1]
                index += 2
            name = key.lstrip("-")
            phone[name] = int(value) if name == "port" else value
            continue
        rest.append(token)
        index += 1
    return rest, phone


if __name__ == "__main__":
    main()
