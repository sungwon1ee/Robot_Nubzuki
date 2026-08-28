"""Validate the complete standing contract without PPO training."""

from __future__ import annotations

import functools
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np

from playground.nubzuki import randomize
from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration
from playground.nubzuki.standing import Standing, _cost_head_pos, default_config


def validate(calibration_path: str | None = None) -> dict:
    calibration = NubzukiCalibration(calibration_path)
    env = Standing(config=default_config())
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    if state.obs["state"].shape != (85,):
        raise AssertionError(f"actor observation mismatch: {state.obs['state'].shape}")
    if state.obs["privileged_state"].shape != (153,):
        raise AssertionError("privileged observation mismatch")
    if env.action_size != 14 or tuple(env.actuator_names) != calibration.joint_order:
        raise AssertionError("action size or joint order mismatch")
    state = jax.jit(env.step)(state, jp.zeros(14))
    if not np.isfinite(np.asarray(state.obs["state"])).all() or not np.isfinite(float(state.reward)):
        raise AssertionError("non-finite reset/step result")
    keys = jax.random.split(jax.random.PRNGKey(10), 512)
    commands = np.asarray(jax.vmap(env.sample_command)(keys))
    if not np.allclose(commands[:, :3], 0.0):
        raise AssertionError("velocity command is not zero")
    for index, name in enumerate(HEAD_JOINTS, start=3):
        low, high = calibration.limits_rad(name)
        if commands[:, index].min() < low or commands[:, index].max() > high:
            raise AssertionError(f"command outside calibrated range for {name}")
    if float(_cost_head_pos(jp.zeros(14), jp.array([0, 0, 0, 0.1, 0, 0, 0]))) <= 0:
        raise AssertionError("head position cost is inactive while standing")
    if default_config().noise_config.action_max_delay != 3:
        raise AssertionError("action delay must sample indices 0, 1 or 2")
    randomizer = functools.partial(
        randomize.domain_randomize, floor_geom_id=env._floor_geom_id,
        torso_body_id=env._torso_body_id,
    )
    randomized, _ = randomizer(env.mjx_model, jax.random.split(jax.random.PRNGKey(2), 2))
    if randomized.body_mass.shape[0] != 2:
        raise AssertionError("domain randomizer did not create a batch")
    # Brax reassembles its 64-bit step counter from two 32-bit halves at the end
    # of every epoch. Checking the behaviour rather than a version number costs
    # a millisecond here and saves discovering it after the first epoch.
    from brax.training.types import UInt64

    try:
        steps = int(UInt64(hi=1, lo=1024))
    except TypeError as error:
        raise AssertionError(
            "Brax cannot convert its step counter with this NumPy "
            f"({np.__version__}); install numpy>=2.1: {error}"
        ) from error
    if steps != (1 << 32) + 1024:
        raise AssertionError(f"Brax step counter round-trip is wrong: {steps}")

    tree = Path(__file__).parent
    for path in tree.glob("*.py"):
        if path.name == "validate_setup.py":
            continue
        text = path.read_text(encoding="utf-8").lower()
        if "poly_reference_motion" in text or "polynomial_coefficients" in text:
            raise AssertionError(f"imitation dependency found in {path.name}")
    return {
        "actor_observation": 85, "privileged_observation": 153,
        "actions": 14, "joint_order": list(calibration.joint_order),
        "calibration_sha256": calibration.sha256,
    }


def main() -> None:
    result = validate()
    print("Nubzuki standing validation: OK")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

