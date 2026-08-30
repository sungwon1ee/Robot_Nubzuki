"""Forward/stop walking task built on the standing policy contract."""

from __future__ import annotations

import jax
import jax.numpy as jp
from ml_collections import config_dict

from playground.common.rewards import (
    cost_ang_vel_xy,
    cost_feet_clearance,
    cost_feet_height,
    cost_feet_slip_contact,
    reward_feet_air_time_window,
    reward_forward_walking_composite,
    reward_pose_tracking,
    reward_standing_composite,
    reward_tracking_lin_vel,
    reward_upright,
    reward_variable_posture,
)
from playground.nubzuki.standing import Standing, default_config as standing_config


WALKING_STAGES = ("discovery", "refine", "control")


def default_config(stage: str = "discovery") -> config_dict.ConfigDict:
    if stage not in WALKING_STAGES:
        raise ValueError(f"Unknown walking curriculum stage: {stage}")
    config = standing_config()
    config.walking_stage = stage
    config.target_swing_height_m = 0.02
    config.upright_std = 0.05**0.5
    config.standing_leg_pose_std = [0.10, 0.05, 0.15, 0.15, 0.10] * 2
    config.walking_leg_pose_std = [0.30, 0.05, 0.40, 0.40, 0.25] * 2

    # Microduck velocity-task weights.  Keep action-rate at its gait-discovery
    # value; Microduck only ramps it toward -1 after a gait already exists.
    scales = config.reward_config.scales
    scales.orientation = 0.0
    scales.torques = 0.0
    scales.action_rate = -0.1
    scales.stand_still = 0.0
    scales.alive = 0.0
    scales.head_pos = 0.0
    scales.negative_hip_roll = 0.0
    scales.tracking_lin_vel = 0.0
    scales.upright = 0.0
    scales.walking_task = 5.0
    scales.standing_task = 2.0

    if stage == "discovery":
        config.forward_velocity_range_m_s = [0.15, 0.15]
        config.zero_command_probability = 0.05
        config.enable_head_command = False
        scales.pose = 0.0
        scales.feet_air_time = 2.0
        scales.foot_clearance = 0.0
        scales.feet_height = 0.0
        scales.foot_slip = -0.02
        scales.body_ang_vel = 0.0
        scales.action_rate = -0.01
        scales.head_pose_tracking = 0.0
    elif stage == "refine":
        config.forward_velocity_range_m_s = [0.12, 0.18]
        config.zero_command_probability = 0.10
        config.enable_head_command = False
        scales.pose = 0.1
        scales.feet_air_time = 2.5
        scales.foot_clearance = -0.1
        scales.feet_height = -0.05
        scales.foot_slip = -0.05
        scales.body_ang_vel = -0.01
        scales.action_rate = -0.05
        scales.head_pose_tracking = 0.0
    else:
        config.forward_velocity_range_m_s = [0.04, 0.18]
        config.zero_command_probability = 0.20
        config.enable_head_command = True
        scales.pose = 0.3
        scales.feet_air_time = 2.0
        scales.foot_clearance = -0.3
        scales.feet_height = -0.1
        scales.foot_slip = -0.1
        scales.body_ang_vel = -0.02
        scales.action_rate = -0.1
        scales.head_pose_tracking = 1.0
    config.reward_config.tracking_sigma = 0.01
    return config


class Walking(Standing):
    """One policy for joystick forward motion and zero-command standing."""

    def _get_reward(
        self, data, action, info, metrics, done, first_contact, contact
    ):
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact
        )
        gravity = self.get_gravity(data)
        joint_pos = self.get_actuator_joints_qpos(data.qpos)
        local_vel = self.get_local_linvel(data)
        rewards["walking_task"] = reward_forward_walking_composite(
            info["command"], local_vel, gravity,
            self._config.reward_config.tracking_sigma,
            self._config.upright_std,
        )
        rewards["standing_task"] = reward_standing_composite(
            info["command"], joint_pos, self._default_actuator, gravity,
            jp.asarray(self._config.standing_leg_pose_std),
            self._config.upright_std,
        )
        rewards["pose"] = reward_variable_posture(
            joint_pos,
            self._default_actuator,
            info["command"],
            jp.asarray(self._config.standing_leg_pose_std),
            jp.asarray(self._config.walking_leg_pose_std),
        )
        rewards["head_pose_tracking"] = reward_pose_tracking(
            joint_pos[5:9], info["command"][3:], std=0.5
        )
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        feet_pos = data.site_xpos[self._feet_site_id]
        rewards["foot_clearance"] = cost_feet_clearance(
            feet_vel,
            feet_pos,
            self._config.target_swing_height_m,
        ) * (jp.linalg.norm(info["command"][:3]) > 0.01)
        rewards["feet_height"] = cost_feet_height(
            info["swing_peak"],
            first_contact,
            self._config.target_swing_height_m,
        ) * (jp.linalg.norm(info["command"][:3]) > 0.01)
        rewards["feet_air_time"] = reward_feet_air_time_window(
            info["feet_air_time"] * (~contact),
            info["command"],
            in_air=~contact,
            threshold_min=0.125,
            threshold_max=0.30,
        )
        rewards["foot_slip"] = cost_feet_slip_contact(
            feet_vel, contact, info["command"]
        )
        rewards["body_ang_vel"] = cost_ang_vel_xy(
            self.get_global_angvel(data)
        )

        return rewards

    def sample_command(self, rng: jax.Array) -> jax.Array:
        velocity_rng, zero_rng, head_rng = jax.random.split(rng, 3)
        standing_command = super().sample_command(head_rng)
        standing_command = jp.where(
            self._config.enable_head_command,
            standing_command,
            jp.zeros_like(standing_command),
        )
        forward_velocity = jax.random.uniform(
            velocity_rng,
            minval=self._config.forward_velocity_range_m_s[0],
            maxval=self._config.forward_velocity_range_m_s[1],
        )
        command = standing_command.at[0].set(forward_velocity)
        return jp.where(
            jax.random.bernoulli(
                zero_rng, p=self._config.zero_command_probability
            ),
            jp.zeros(7),
            command,
        )
