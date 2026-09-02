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
    cost_head_action_rate,
    cost_head_joint_velocity,
    cost_head_roll_home,
    cost_head_roll_velocity,
    cost_hip_pitch_overload,
    cost_yaw_rate,
    reward_feet_air_time_window,
    reward_forward_walking_composite,
    reward_pose_tracking,
    reward_standing_composite,
    reward_tracking_yaw_rate,
    reward_tracking_lin_vel,
    reward_upright,
    reward_variable_posture,
)
from playground.nubzuki.standing import Standing, default_config as standing_config


WALKING_STAGES = (
    "discovery", "locomotion", "sim2real_1", "sim2real_2", "sim2real_3",
    "sim2real", "torque_limit_1", "torque_limit_2", "torque_limit_3",
    "hip_relief",
    "torque_limit", "head_position_1", "head_position_2", "head_position_3",
    "refine", "control", "turning",
)


def default_config(stage: str = "discovery") -> config_dict.ConfigDict:
    if stage not in WALKING_STAGES:
        raise ValueError(f"Unknown walking curriculum stage: {stage}")
    config = standing_config()
    config.walking_stage = stage
    config.target_swing_height_m = 0.02
    config.upright_std = 0.05**0.5
    config.standing_leg_pose_std = [0.10, 0.05, 0.15, 0.15, 0.10] * 2
    config.walking_leg_pose_std = [0.30, 0.05, 0.40, 0.40, 0.25] * 2

    # Shared locomotion-task baseline.  Each explicit stage changes only the
    # commands and regularizers needed for that skill.
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
    scales.yaw_tracking = 0.0
    scales.straight_yaw_rate = 0.0
    scales.head_roll_home = 0.0
    scales.head_roll_vel = 0.0
    scales.hip_overload = 0.0
    config.yaw_rate_range_rad_s = [0.0, 0.0]
    config.min_turn_yaw_rate_rad_s = 0.0
    config.yaw_tracking_sigma = 0.1
    config.straight_command_probability = 1.0
    config.turn_in_place_probability = 0.0
    config.head_mode_probability = 0.0
    config.head_zero_probability = 1.0
    config.simultaneous_head_probability = 1.0
    config.head_yaw_pitch_only = False

    if stage in (
        "locomotion", "sim2real", "sim2real_1", "sim2real_2", "sim2real_3",
        "torque_limit_1", "torque_limit_2", "torque_limit_3", "torque_limit",
        "hip_relief",
    ):
        # Continue from the best gait checkpoint and learn only forward motion
        # plus curved left/right turns.  Head commands, reverse and in-place
        # turns are deliberately absent until locomotion is robust on video.
        config.forward_velocity_range_m_s = [0.04, 0.18]
        config.yaw_rate_range_rad_s = [-0.70, 0.70]
        config.min_turn_yaw_rate_rad_s = 0.15
        config.yaw_tracking_sigma = 0.1
        config.straight_command_probability = 0.20
        config.turn_in_place_probability = 0.0
        config.zero_command_probability = 0.20
        config.enable_head_command = False
        config.head_mode_probability = 0.0
        config.head_zero_probability = 1.0

        scales.pose = 0.3
        scales.feet_air_time = 2.5
        scales.foot_clearance = -0.1
        scales.feet_height = -0.05
        scales.foot_slip = -0.1
        scales.body_ang_vel = -0.02
        scales.yaw_rate = 0.0
        scales.yaw_tracking = 5.0
        scales.straight_yaw_rate = -0.1
        scales.action_rate = -0.3
        scales.head_pose_tracking = 0.0
        scales.head_action_rate = -0.01
        scales.head_joint_vel = 0.0
        scales.head_roll_home = 0.0
        scales.head_roll_vel = 0.0
        if stage.startswith("torque_limit") or stage == "hip_relief":
            # First isolate actuator strength from long-delay adaptation.  The
            # original 3.23 N.m model let hip pitch sit at its stall limit for
            # much of every step, which the moving physical servo cannot do.
            force_limits = {
                "torque_limit_1": 2.8,
                "torque_limit_2": 2.5,
                "torque_limit_3": 2.2,
                # Backward-compatible final stress stage.
                "torque_limit": 2.0,
                "hip_relief": 2.8,
            }
            config.leg_force_limit_nm = force_limits[stage]
            # Keep the actuator-response condition that already transferred
            # well: one fixed 4--6 tick delay per episode (mean 0.10 s).
            config.noise_config.action_min_delay = 4
            config.noise_config.action_max_delay = 7
            # Total command mix: 60% straight, 20% curved, 20% stopped.
            config.straight_command_probability = 0.75
            if stage == "hip_relief":
                # Preserve torque_limit_1 exactly and discourage only sustained
                # hip-pitch effort above 75% of the available actuator torque.
                scales.hip_overload = -2.0
        if stage.startswith("sim2real"):
            # Curriculum totals include the 20% stop commands.  The sampler
            # chooses straight versus turning only after the stop draw, so the
            # conditional probabilities below produce total distributions of
            # 60/20/20, 40/40/20 and 20/60/20 (straight/turn/stop).
            curriculum = {
                "sim2real_1": (4, 7, 0.75),
                "sim2real_2": (6, 9, 0.50),
                "sim2real_3": (8, 11, 0.25),
                # Backward-compatible alias for the former final stage.
                "sim2real": (8, 11, 0.25),
            }
            delay_min, delay_max, straight_probability = curriculum[stage]
            config.noise_config.action_min_delay = delay_min
            config.noise_config.action_max_delay = delay_max
            config.straight_command_probability = straight_probability
    elif stage.startswith("head_position_"):
        stage_index = int(stage.rsplit("_", 1)[1]) - 1
        head_probabilities = (0.20, 0.40, 0.60)
        head_range_factors = (0.20, 0.40, 0.60)
        head_tracking_weights = (0.50, 0.75, 1.00)

        # Preserve the proven locomotion distribution and overlay only the
        # two head axes exposed by WALK + HEAD: pitch and yaw.
        config.forward_velocity_range_m_s = [0.04, 0.18]
        config.yaw_rate_range_rad_s = [-0.70, 0.70]
        config.min_turn_yaw_rate_rad_s = 0.15
        config.yaw_tracking_sigma = 0.1
        config.straight_command_probability = 0.20
        config.turn_in_place_probability = 0.0
        config.zero_command_probability = 0.20
        config.enable_head_command = True
        config.head_mode_probability = 0.0
        config.head_zero_probability = 1.0
        config.simultaneous_head_probability = head_probabilities[stage_index]
        config.head_range_factor = head_range_factors[stage_index]
        config.head_yaw_pitch_only = True

        scales.pose = 0.3
        scales.feet_air_time = 2.5
        scales.foot_clearance = -0.1
        scales.feet_height = -0.05
        scales.foot_slip = -0.1
        scales.body_ang_vel = -0.02
        scales.yaw_rate = 0.0
        scales.yaw_tracking = 5.0
        scales.straight_yaw_rate = -0.1
        scales.action_rate = -0.3
        scales.head_pose_tracking = head_tracking_weights[stage_index]
        scales.head_action_rate = -0.01
        scales.head_joint_vel = 0.0
        scales.head_roll_home = 0.0
        scales.head_roll_vel = 0.0
    elif stage == "discovery":
        config.forward_velocity_range_m_s = [0.15, 0.15]
        config.zero_command_probability = 0.05
        config.enable_head_command = False
        scales.pose = 0.0
        scales.feet_air_time = 2.0
        scales.foot_clearance = 0.0
        scales.feet_height = 0.0
        scales.foot_slip = -0.02
        scales.body_ang_vel = 0.0
        scales.yaw_rate = 0.0
        scales.action_rate = -0.01
        scales.head_pose_tracking = 0.0
        scales.head_action_rate = 0.0
        scales.head_joint_vel = 0.0
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
        scales.yaw_rate = -0.05
        scales.action_rate = -0.05
        scales.head_pose_tracking = 0.0
        scales.head_action_rate = -0.01
        scales.head_joint_vel = 0.0
    elif stage == "control":
        config.forward_velocity_range_m_s = [0.04, 0.18]
        config.zero_command_probability = 0.25 * 0.25
        config.enable_head_command = True
        config.head_mode_probability = 0.25
        config.head_zero_probability = 0.25
        scales.pose = 0.3
        scales.feet_air_time = 2.0
        scales.foot_clearance = -0.3
        scales.feet_height = -0.1
        scales.foot_slip = -0.1
        scales.body_ang_vel = -0.02
        scales.yaw_rate = -0.1
        scales.action_rate = -0.1
        scales.head_pose_tracking = 1.0
        scales.head_action_rate = -0.02
        scales.head_joint_vel = -0.01
        scales.head_roll_home = -5.0
        scales.head_roll_vel = -0.1
    else:
        config.forward_velocity_range_m_s = [0.06, 0.18]
        config.yaw_rate_range_rad_s = [-0.3, 0.3]
        config.yaw_tracking_sigma = 0.04
        config.straight_command_probability = 0.50
        config.turn_in_place_probability = 0.0
        config.zero_command_probability = 0.25 * 0.25
        config.enable_head_command = True
        config.head_mode_probability = 0.30
        config.head_zero_probability = 0.25
        scales.pose = 0.3
        scales.feet_air_time = 2.0
        scales.foot_clearance = -0.3
        scales.feet_height = -0.1
        scales.foot_slip = -0.1
        scales.body_ang_vel = -0.02
        scales.yaw_rate = 0.0
        scales.yaw_tracking = 5.0
        scales.straight_yaw_rate = -0.1
        scales.action_rate = -0.1
        scales.head_pose_tracking = 1.0
        scales.head_action_rate = -0.03
        scales.head_joint_vel = -0.02
        scales.head_roll_home = -10.0
        scales.head_roll_vel = -0.2
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
        locomotion_active = jp.linalg.norm(info["command"][:3]) > 0.01
        # Walking controls yaw/pitch but leaves roll mechanically free.  Head
        # mode still tracks all four joints, including roll and neck pitch.
        if self._config.head_yaw_pitch_only:
            rewards["head_pose_tracking"] = reward_pose_tracking(
                joint_pos[6:8], info["command"][4:6], std=0.5
            )
        else:
            rewards["head_pose_tracking"] = jp.where(
                locomotion_active,
                reward_pose_tracking(
                    joint_pos[5:8], info["command"][3:6], std=0.5
                ),
                reward_pose_tracking(
                    joint_pos[5:9], info["command"][3:], std=0.5
                ),
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
        rewards["hip_overload"] = cost_hip_pitch_overload(
            data.actuator_force,
            jp.maximum(self._config.leg_force_limit_nm, 2.8),
        ) * locomotion_active
        rewards["yaw_rate"] = cost_yaw_rate(self.get_gyro(data))
        turning_command = jp.abs(info["command"][2]) > 0.05
        rewards["yaw_tracking"] = reward_tracking_yaw_rate(
            info["command"], self.get_gyro(data),
            self._config.yaw_tracking_sigma, gravity,
            self._config.upright_std,
        ) * turning_command
        straight_command = (
            (info["command"][0] > 0.01)
            & (jp.abs(info["command"][2]) < 0.01)
        )
        rewards["straight_yaw_rate"] = cost_yaw_rate(
            self.get_gyro(data)
        ) * straight_command
        rewards["head_action_rate"] = cost_head_action_rate(
            action, info["last_act"]
        ) * locomotion_active
        rewards["head_joint_vel"] = cost_head_joint_velocity(
            self.get_actuator_joints_qvel(data.qvel)
        ) * locomotion_active
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        stopped_at_home = jp.linalg.norm(info["command"]) < 0.01
        rewards["head_roll_home"] = cost_head_roll_home(joint_pos) * (
            locomotion_active | stopped_at_home
        )
        rewards["head_roll_vel"] = cost_head_roll_velocity(
            joint_vel
        ) * locomotion_active

        return rewards

    def sample_command(self, rng: jax.Array) -> jax.Array:
        (
            velocity_rng,
            yaw_rng,
            straight_rng,
            turn_in_place_rng,
            mode_rng,
            zero_rng,
            head_zero_rng,
            simultaneous_head_rng,
            neck_rng,
            pitch_rng,
            head_yaw_rng,
            roll_rng,
        ) = jax.random.split(rng, 12)
        factor = self._config.head_range_factor
        head_command = jp.array(
            [
                jax.random.uniform(
                    neck_rng,
                    minval=self._config.neck_pitch_range[0] * factor,
                    maxval=self._config.neck_pitch_range[1] * factor,
                ),
                jax.random.uniform(
                    pitch_rng,
                    minval=self._config.head_pitch_range[0] * factor,
                    maxval=self._config.head_pitch_range[1] * factor,
                ),
                jax.random.uniform(
                    head_yaw_rng,
                    minval=self._config.head_yaw_range[0] * factor,
                    maxval=self._config.head_yaw_range[1] * factor,
                ),
                jax.random.uniform(
                    roll_rng,
                    minval=self._config.head_roll_range[0] * factor,
                    maxval=self._config.head_roll_range[1] * factor,
                ),
            ]
        )
        forward_velocity = jax.random.uniform(
            velocity_rng,
            minval=self._config.forward_velocity_range_m_s[0],
            maxval=self._config.forward_velocity_range_m_s[1],
        )
        walking_head_command = jp.where(
            self._config.head_yaw_pitch_only,
            head_command.at[jp.array([0, 3])].set(0.0),
            head_command,
        )
        walking_head_command = jp.where(
            jax.random.bernoulli(
                simultaneous_head_rng,
                p=self._config.simultaneous_head_probability,
            ),
            walking_head_command,
            jp.zeros(4),
        )
        moving_command = jp.hstack(
            [forward_velocity, 0.0, 0.0, walking_head_command]
        )
        yaw_rate = jax.random.uniform(
            yaw_rng,
            minval=self._config.yaw_rate_range_rad_s[0],
            maxval=self._config.yaw_rate_range_rad_s[1],
        )
        min_turn_rate = self._config.min_turn_yaw_rate_rad_s
        yaw_rate = jp.where(
            jp.abs(yaw_rate) < min_turn_rate,
            jp.where(yaw_rate >= 0.0, min_turn_rate, -min_turn_rate),
            yaw_rate,
        )
        yaw_rate = jp.where(
            jax.random.bernoulli(
                straight_rng, p=self._config.straight_command_probability
            ),
            0.0,
            yaw_rate,
        )
        moving_command = moving_command.at[2].set(yaw_rate)
        turn_in_place = jax.random.bernoulli(
            turn_in_place_rng, p=self._config.turn_in_place_probability
        ) & (jp.abs(yaw_rate) > 0.01)
        moving_command = moving_command.at[0].set(
            jp.where(turn_in_place, 0.0, moving_command[0])
        )
        # Walk mode does not expose head-roll control.  Its zero placeholder is
        # ignored by head-pose tracking while moving, so roll remains free.
        moving_command = moving_command.at[6].set(0.0)
        head_mode_command = jp.hstack([jp.zeros(3), head_command])
        head_mode_command = jp.where(
            jax.random.bernoulli(
                head_zero_rng, p=self._config.head_zero_probability
            ),
            jp.zeros(7),
            head_mode_command,
        )
        controlled_command = jp.where(
            jax.random.bernoulli(
                mode_rng, p=self._config.head_mode_probability
            ),
            head_mode_command,
            moving_command,
        )
        if self._config.enable_head_command:
            if self._config.head_yaw_pitch_only:
                stop = jax.random.bernoulli(
                    zero_rng, p=self._config.zero_command_probability
                )
                controlled_command = controlled_command.at[:3].set(
                    jp.where(stop, jp.zeros(3), controlled_command[:3])
                )
            return controlled_command
        return jp.where(
            jax.random.bernoulli(
                zero_rng, p=self._config.zero_command_probability
            ),
            jp.zeros(7),
            moving_command.at[3:].set(0.0),
        )
