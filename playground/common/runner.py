"""Brax PPO runner with resumable checkpoint and ONNX artifacts."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from brax.training.acme import running_statistics
from flax.training import orbax_utils
import jax
from mujoco_playground import wrapper
from orbax import checkpoint as ocp
from tensorboardX import SummaryWriter

from playground.common.export_onnx import export_onnx
from playground.nubzuki.ppo_config import NETWORK_CONFIG, training_config
from playground.nubzuki.resume import remaining_timesteps, resolve_restore


class BaseRunner:
    def __init__(self, args) -> None:
        self.args = args
        self.output_dir = Path(args.output).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = self.output_dir / "tensorboard"
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.env = None
        self.eval_env = None
        self.randomizer = None
        self.action_size = None
        self.obs_size = None
        self.policy_metadata = {}
        self.step_offset = 0
        cache = Path(".tmp/jax_cache").resolve()
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache)

    def progress_callback(self, num_steps: int, metrics: dict) -> None:
        absolute = self.step_offset + int(num_steps)
        # TensorboardX keeps a background event-writer thread.  On Colab that
        # thread can stop after the initial JAX evaluation while add_scalar
        # continues without surfacing an error, leaving a valid event file that
        # contains only step 0.  Give every evaluation its own short-lived
        # writer so its events are fully committed before training continues.
        with SummaryWriter(
            log_dir=self.tensorboard_dir,
            filename_suffix=f".step-{absolute:012d}",
        ) as writer:
            for name, value in metrics.items():
                writer.add_scalar(name, value, absolute)
            writer.flush()
        print(f"step={absolute} eval_reward={metrics.get('eval/episode_reward', 'n/a')}")

    def policy_params_fn(self, current_step, make_policy, params) -> None:
        del make_policy
        absolute = self.step_offset + int(current_step)
        artifact_dir = self.output_dir / "checkpoints" / f"step_{absolute:012d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = artifact_dir / "params"
        checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        checkpointer.save(str(checkpoint_path), params, force=True, save_args=save_args)
        onnx_path = artifact_dir / "policy.onnx"
        export_onnx(params, self.action_size, self.ppo_params, self.obs_size, str(onnx_path))
        metadata = dict(self.policy_metadata)
        metadata.update(
            checkpoint_step=absolute,
            preset=self.args.preset,
            steps_this_run=int(current_step),
            resumed_from_step=self.step_offset,
        )
        metadata_path = artifact_dir / "policy.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        latest = {
            "checkpoint_step": absolute, "checkpoint": str(checkpoint_path),
            "policy": str(onnx_path), "metadata": str(metadata_path),
            "target_timesteps": int(self.args.num_timesteps),
        }
        (self.output_dir / "latest.json").write_text(
            json.dumps(latest, indent=2, sort_keys=True), encoding="utf-8"
        )

    def train(self) -> None:
        if self.env is None or self.eval_env is None:
            raise RuntimeError("Runner environment was not initialized")
        restore_path, self.step_offset = resolve_restore(
            getattr(self.args, "restore", None),
            self.output_dir,
            getattr(self.args, "step_offset", None),
        )
        target = int(self.args.num_timesteps)
        remaining = remaining_timesteps(target, self.step_offset)
        if restore_path is not None:
            print(
                f"Resuming from {restore_path} at step {self.step_offset}; "
                f"training {remaining} more steps to reach {target}."
            )
        self.ppo_params = training_config(
            self.args.preset, remaining, Path(self.args.mac_profile),
            checkpoint_every=getattr(self.args, "checkpoint_every", None) or 5_000_000,
            num_eval_envs=getattr(self.args, "num_eval_envs", None),
            num_envs=getattr(self.args, "num_envs", None),
        )
        checkpoints = max(self.ppo_params["num_evals"] - 1, 1)
        print(
            f"{checkpoints} checkpoints, about {remaining // checkpoints} steps apart; "
            "that is the most an interrupted run can lose."
        )
        network_factory = functools.partial(ppo_networks.make_ppo_networks, **NETWORK_CONFIG)
        print("PPO parameters:", self.ppo_params)
        restore_params = None
        if restore_path is not None:
            restored = ocp.PyTreeCheckpointer().restore(str(restore_path))
            normalizer = restored[0]
            if isinstance(normalizer, dict):
                normalizer = running_statistics.RunningStatisticsState(**normalizer)
            restore_params = (normalizer, restored[1], restored[2])
        train_fn = functools.partial(
            ppo.train, **self.ppo_params, network_factory=network_factory,
            randomization_fn=self.randomizer, progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn, restore_checkpoint_path=None,
            restore_params=restore_params,
        )
        train_fn(environment=self.env, eval_env=self.eval_env,
                 wrap_env_fn=wrapper.wrap_for_brax_training)
