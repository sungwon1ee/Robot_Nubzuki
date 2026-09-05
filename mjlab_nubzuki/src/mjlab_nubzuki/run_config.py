"""Read what a checkpoint was actually trained with.

MJLab writes params/env.yaml into every run directory. Reading command ranges
from today's task config instead is how a policy trained to walk at 0.18 m/s
ends up being commanded 0.25: the code moves on, the checkpoint does not. Both
the hardware export and the phone-driven simulation take their ranges from
here so the two cannot drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path


def _tuple_after(block: str, key: str, count: int = 2) -> list[float] | None:
    """Pull `key: !!python/tuple` followed by `- <number>` lines out of YAML.

    The dumped config uses python-specific tags, so yaml.safe_load refuses it
    and unsafe_load would import whatever the file names. These ranges are
    plain numbers; read them literally.
    """
    match = re.search(rf"^\s*{re.escape(key)}:\s*(?:!!python/tuple)?\s*$", block, re.M)
    if match is None:
        return None
    values = []
    for line in block[match.end():].splitlines()[1:]:
        item = re.match(r"^\s*-\s*(-?[\d.eE+]+)\s*$", line)
        if item is None:
            break
        values.append(float(item.group(1)))
        if len(values) == count:
            break
    return values if len(values) == count else None


def _section(text: str, key: str, next_keys: tuple[str, ...]) -> str:
    start = text.index(f"\n  {key}:")
    ends = [text.find(f"\n  {n}:", start + 1) for n in next_keys]
    ends = [e for e in ends if e > 0]
    return text[start : min(ends)] if ends else text[start:]


def trained_command_ranges(env_yaml: Path) -> dict:
    """The command ranges the checkpoint was actually trained with.

    Reading them from today's task config instead is how a policy trained to
    walk at 0.18 m/s ends up being commanded 0.25: the code moved on, the
    checkpoint did not.
    """
    text = "\n" + env_yaml.read_text(encoding="utf-8")
    commands = text[text.index("\ncommands:") :]
    twist = _section(commands, "twist", ("head_pose", "body_pose"))
    head = _section(commands, "head_pose", ("body_pose",))

    ranges = {}
    for key, name in (
        ("lin_vel_x", "twist_lin_vel_x"),
        ("lin_vel_y", "twist_lin_vel_y"),
        ("ang_vel_z", "twist_ang_vel_z"),
    ):
        value = _tuple_after(twist, key)
        if value is None:
            raise SystemExit(f"Could not read commands.twist.ranges.{key} from {env_yaml}")
        ranges[name] = value

    head_ranges = re.search(r"^\s*ranges:\s*(?:!!python/tuple)?\s*$", head, re.M)
    if head_ranges is None:
        raise SystemExit(f"Could not read commands.head_pose.ranges from {env_yaml}")
    numbers = [
        float(m.group(1))
        for m in re.finditer(r"^\s*-\s*(-?[\d.eE+]+)\s*$", head[head_ranges.end():], re.M)
    ][:8]
    if len(numbers) != 8:
        raise SystemExit(f"head_pose.ranges in {env_yaml} is not four pairs")
    ranges["head_pose"] = [numbers[i : i + 2] for i in range(0, 8, 2)]
    return ranges


def find_env_yaml(checkpoint: Path) -> Path | None:
    """MJLab writes params/env.yaml beside the checkpoints of every run."""
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        candidate = directory / "params" / "env.yaml"
        if candidate.exists():
            return candidate
    return None


