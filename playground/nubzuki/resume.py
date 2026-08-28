"""Resume bookkeeping for interrupted training runs.

Brax restores policy, value and normalizer parameters through `restore_params`
but leaves `TrainingState.env_steps` at zero, so a resumed run counts from zero
again. Left alone that would rewrite the checkpoint directories of the run it
continues and fold the TensorBoard curve back onto itself. Everything needed to
translate a checkpoint back into an absolute environment step lives here, free
of JAX and Brax imports so it can be tested on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
import re


CHECKPOINT_STEP_PATTERN = re.compile(r"step_(\d+)")


def checkpoint_step(path: Path) -> int:
    """Recover the absolute environment step a checkpoint was written at."""
    metadata = Path(path).parent / "policy.json"
    if metadata.exists():
        recorded = json.loads(metadata.read_text(encoding="utf-8")).get("checkpoint_step")
        if recorded is not None:
            return int(recorded)
    match = CHECKPOINT_STEP_PATTERN.search(str(path))
    if match:
        return int(match.group(1))
    raise ValueError(
        f"Cannot infer the environment step of {path}. Pass --step-offset explicitly."
    )


def resolve_restore(
    spec: str | None, output_dir: Path, step_offset: int | None = None,
) -> tuple[Path | None, int]:
    """Return the checkpoint directory to continue from and its absolute step."""
    output_dir = Path(output_dir)
    if not spec:
        return None, int(step_offset or 0)
    if spec in ("auto", "latest"):
        pointer = output_dir / "latest.json"
        if not pointer.exists():
            print(f"No previous run at {pointer}; starting from scratch.")
            return None, int(step_offset or 0)
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        path = Path(payload["checkpoint"]).expanduser()
        offset = int(payload["checkpoint_step"])
    else:
        path = Path(spec).expanduser().resolve()
        offset = checkpoint_step(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint to restore does not exist: {path}")
    if step_offset is not None:
        offset = int(step_offset)
    return path, offset


def remaining_timesteps(target: int, step_offset: int) -> int:
    """Steps still to run so the whole schedule totals `target`."""
    remaining = int(target) - int(step_offset)
    if remaining <= 0:
        raise SystemExit(
            f"Nothing to do: {step_offset} steps already trained and --num-timesteps "
            f"is {target}. Raise --num-timesteps to continue."
        )
    return remaining
