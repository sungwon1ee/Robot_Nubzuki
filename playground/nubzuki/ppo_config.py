"""Pinned PPO settings used instead of a version-dependent named preset."""

from __future__ import annotations

import json
import math
from pathlib import Path


UPSTREAM_PPO = {
    "num_timesteps": 150_000_000,
    "num_evals": 31,
    "reward_scaling": 1.0,
    "episode_length": 1000,
    "normalize_observations": True,
    "action_repeat": 1,
    "unroll_length": 20,
    "num_minibatches": 32,
    "num_updates_per_batch": 4,
    "discounting": 0.97,
    "learning_rate": 3.0e-4,
    "entropy_cost": 0.005,
    "num_envs": 8192,
    "batch_size": 256,
    "clipping_epsilon": 0.2,
    "max_grad_norm": 1.0,
}

NETWORK_CONFIG = {
    "policy_hidden_layer_sizes": (512, 256, 128),
    "value_hidden_layer_sizes": (512, 256, 128),
    "policy_obs_key": "state",
    "value_obs_key": "privileged_state",
}


def load_device_profile(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Device profile not found: {path}. Run `nubzuki-standing benchmark`."
        )
    profile = json.loads(path.read_text(encoding="utf-8"))
    if int(profile.get("num_envs", 0)) <= 0:
        raise ValueError(f"Invalid device profile: {path}")
    return profile


DEFAULT_CHECKPOINT_EVERY = 5_000_000


def eval_count(num_timesteps: int, checkpoint_every: int) -> int:
    """How many Brax evals give a checkpoint at most `checkpoint_every` apart.

    Brax writes a checkpoint at the end of each of its `num_evals - 1` epochs,
    so this is what decides how much work an interrupted run can lose. Every
    checkpoint also costs one evaluation rollout plus an ONNX export, so a
    shorter interval buys a smaller loss window with wall-clock time.
    """
    if int(checkpoint_every) <= 0:
        raise ValueError("--checkpoint-every must be positive")
    return max(2, math.ceil(int(num_timesteps) / int(checkpoint_every)) + 1)


def training_config(
    preset: str,
    num_timesteps: int,
    profile_path: Path,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    num_eval_envs: int | None = None,
    num_envs: int | None = None,
) -> dict:
    config = dict(UPSTREAM_PPO)
    config["num_timesteps"] = int(num_timesteps)
    config["num_evals"] = eval_count(num_timesteps, checkpoint_every)
    if num_eval_envs is not None:
        if int(num_eval_envs) <= 0:
            raise ValueError("--num-eval-envs must be positive")
        config["num_eval_envs"] = int(num_eval_envs)
    if preset in ("profile", "macbook"):
        config["num_envs"] = int(load_device_profile(profile_path)["num_envs"])
    elif preset == "smoke":
        config.update(
            num_timesteps=int(num_timesteps), num_envs=64, episode_length=128,
            unroll_length=16, batch_size=16, num_minibatches=4,
            num_updates_per_batch=1, num_evals=1,
        )
        config.pop("num_eval_envs", None)
    elif preset != "official":
        raise ValueError(f"Unknown training preset: {preset}")

    if num_envs is not None:
        config["num_envs"] = int(num_envs)

    # Brax collects batch_size * num_minibatches transitions per training step
    # and splits them across num_envs, so the split has to come out whole.
    per_step = config["batch_size"] * config["num_minibatches"]
    if per_step % config["num_envs"] or per_step < config["num_envs"]:
        raise ValueError(
            f"num_envs={config['num_envs']} does not divide "
            f"batch_size * num_minibatches = {per_step}; Brax requires that."
        )
    return config

