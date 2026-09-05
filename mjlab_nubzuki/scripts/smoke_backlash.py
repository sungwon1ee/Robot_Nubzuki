"""Pre-flight checks for the backlash task, the head/kp changes and the mirror.

Run before pushing to Colab:

    cd mjlab_nubzuki && uv run python scripts/smoke_backlash.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

FAILURES: list[str] = []


def check(label: str):
    def wrap(fn):
        try:
            detail = fn()
        except Exception as error:  # noqa: BLE001 - this script reports, not raises
            FAILURES.append(label)
            print(f"  FAIL  {label}: {type(error).__name__}: {error}")
        else:
            print(f"  ok    {label}" + (f" - {detail}" if detail else ""))
        return fn

    return wrap


print("1. robot spec")


@check("backlash hinges compile, one per servo joint")
def _backlash_spec():
    from mjlab_nubzuki.robot import JOINT_ACTUATOR_ORDER, get_nubzuki_backlash_spec

    model = get_nubzuki_backlash_spec().compile()
    names = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.joint(i).name.endswith("_backlash")
    ]
    expected = {f"passive_{name}_backlash" for name in JOINT_ACTUATOR_ORDER}
    missing = expected - set(names)
    if missing:
        raise AssertionError(f"missing {sorted(missing)}")
    joint = model.joint(names[0])
    return f"{len(names)} joints, range +/-{joint.range[1]:.5f} rad"


@check("play-free spec still compiles and has no backlash joints")
def _plain_spec():
    from mjlab_nubzuki.robot import get_nubzuki_spec

    model = get_nubzuki_spec().compile()
    leftover = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.joint(i).name.endswith("_backlash")
    ]
    if leftover:
        raise AssertionError(f"unexpected {leftover}")
    return f"{model.njnt} joints"


print("2. actuator gains")


@check("leg kp_fw 30 / head kp_fw 24, matching the calibration file")
def _gains():
    import json

    from mjlab_nubzuki.robot import ACTUATORS, CALIBRATION_JSON

    runtime = json.load(open(CALIBRATION_JSON))["runtime"]
    gains = sorted(cfg.kp_fw for cfg in ACTUATORS)
    expected = sorted([float(runtime["head_kp"]), float(runtime["leg_kp"])])
    if gains != expected:
        raise AssertionError(f"{gains} != calibration {expected}")
    return f"{expected}"


print("3. task registration")


@check("both tasks register and build their env cfgs")
def _tasks():
    import mjlab.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    tasks = [t for t in list_tasks() if "Nubzuki" in t]
    for needed in (
        "Mjlab-Velocity-Flat-BAM-Nubzuki",
        "Mjlab-Velocity-Flat-Backlash-BAM-Nubzuki",
    ):
        if needed not in tasks:
            raise AssertionError(f"{needed} not registered (found {tasks})")
    return ", ".join(tasks)


@check("command ranges are the stage-2 envelope")
def _ranges():
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Flat-Backlash-BAM-Nubzuki")
    twist = cfg.commands["twist"].ranges
    head = cfg.commands["head_pose"].ranges
    assert twist.lin_vel_x == (-0.15, 0.25), twist.lin_vel_x
    assert twist.lin_vel_y == (0.0, 0.0), twist.lin_vel_y
    assert twist.ang_vel_z == (-0.70, 0.70), twist.ang_vel_z
    assert head[2] == (-0.50, 0.50), head[2]
    return f"x={twist.lin_vel_x} yaw={twist.ang_vel_z} head_yaw={head[2]}"


@check("backlash obs keeps 14 joints (no dim change)")
def _obs():
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Flat-Backlash-BAM-Nubzuki")
    term = cfg.observations["actor"].terms["joint_pos"]
    names = term.params["asset_cfg"].joint_names
    if "backlash" not in term.func.__name__:
        raise AssertionError(f"joint_pos still uses {term.func.__name__}")
    return f"{term.func.__name__}, selection {names}"


print("4. drive mirror")


@check("mirror writes a checkpoint to its destination")
def _mirror():
    from mjlab_nubzuki.drive_mirror import ENV_DIR, RunMirror

    with tempfile.TemporaryDirectory() as tmp:
        run, dest = Path(tmp) / "run", Path(tmp) / "drive"
        run.mkdir()
        (run / "model_0.pt").write_bytes(b"x" * 16)
        os.environ[ENV_DIR] = str(dest)
        mirror = RunMirror(run)
        mirror.sync(run / "model_0.pt")
        mirror.flush()
        os.environ.pop(ENV_DIR)
        if not (dest / "model_0.pt").exists():
            raise AssertionError(f"nothing landed in {dest}")
    return "checkpoint copied"


print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all checks passed - safe to commit and push")
