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
    reward_pose_tracking,
    reward_tracking_lin_vel,
    reward_upright,
    reward_variable_posture,
)
from playground.nubzuki.standing import Standing, default_config as standing_config


def default_config() -> config_dict.ConfigDict:
    config = standing_config()
    config.forward_velocity_range_m_s = [0.03, 0.15]
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
    scales.tracking_lin_vel = 2.0
    scales.upright = 2.0
    scales.pose = 1.0
    scales.feet_air_time = 3.0
    scales.foot_clearance = -2.0
    scales.feet_height = -0.25
    scales.foot_slip = -0.1
    scales.body_ang_vel = -0.05
    scales.head_pose_tracking = 2.0
    config.reward_config.tracking_sigma = 0.1
    return config


class Walking(Standing):
    """One policy for joystick forward motion and zero-command standing."""

    def _get_reward(
        self, data, action, info, metrics, done, first_contact, contact
    ):
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact
        )
        rewards["tracking_lin_vel"] = reward_tracking_lin_vel(
            info["command"],
            self.get_local_linvel(data),
            self._config.reward_config.tracking_sigma,
        )
        gravity = self.get_gravity(data)
        rewards["upright"] = reward_upright(
            gravity, self._config.upright_std
        )
        joint_pos = self.get_actuator_joints_qpos(data.qpos)
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
