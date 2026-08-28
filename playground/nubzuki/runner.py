"""Nubzuki standing training runner."""

from __future__ import annotations

import functools
from pathlib import Path

from playground.common.runner import BaseRunner
from playground.nubzuki import randomize
from playground.nubzuki.calibration import NubzukiCalibration
from playground.nubzuki.standing import Standing, default_config


class NubzukiStandingRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        calibration = NubzukiCalibration(args.calibration)
        config = default_config()
        self.env = Standing(task="flat_terrain", config=config)
        self.eval_env = Standing(task="flat_terrain", config=config)
        self.randomizer = functools.partial(
            randomize.domain_randomize,
            floor_geom_id=self.env._floor_geom_id,
            torso_body_id=self.env._torso_body_id,
        )
        self.action_size = self.env.action_size
        self.obs_size = int(self.env.observation_size["state"][0])
        self.policy_metadata = {
            "schema_version": 2, "robot": "nubzuki", "policy": "standing",
            "model_semantics_version": 3,
            "deployable": args.preset != "smoke",
            "upstream_commit": "ba59de88ab76163f2e0c2c95b4cd45fea5745106",
            "calibration_sha256": calibration.sha256,
            "observation_size": self.obs_size,
            "privileged_observation_size": 153,
            "action_size": self.action_size, "control_frequency_hz": 50,
            "command_order": list(calibration.command_order),
            "joint_order": list(calibration.joint_order),
            "observation_layout": [
                ["gyro", 3], ["accelerometer", 3], ["command", 7],
                ["joint_position_error", 14], ["joint_velocity_x_0.05", 14],
                ["action_history", 42], ["foot_contact", 2],
            ],
            "action_scale_rad": calibration.action_scale_rad,
            "action_scales_rad": [calibration.action_scale_rad] * 14,
            "action_delay_steps": [0, 2], "home_position_rad": [0.0] * 14,
            "no_placo": True, "no_imitation": True,
        }
        profile_path = Path("config/head_dynamics.json")
        if profile_path.exists():
            from playground.nubzuki.head_dynamics import HeadDynamicsProfile
            profile = HeadDynamicsProfile.load(profile_path, calibration)
            self.policy_metadata["head_dynamics_sha256"] = profile.sha256
