"""Open Duck standing task, adapted only to the Nubzuki MuJoCo model."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
import numpy as np

from playground.common.rewards import (
    cost_action_rate,
    cost_orientation,
    cost_stand_still,
    cost_torques,
    reward_alive,
)
from playground.nubzuki import constants
from playground.nubzuki.base import NubzukiEnv
from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration


def _geoms_colliding(data: mjx.Data, geom1: int, geom2: int) -> jax.Array:
    pair = (
        ((data.contact.geom1 == geom1) & (data.contact.geom2 == geom2))
        | ((data.contact.geom1 == geom2) & (data.contact.geom2 == geom1))
    )
    return jp.any(pair & (data.contact.dist < 0.0))


def _cost_head_pos(joints_qpos: jax.Array, command: jax.Array) -> jax.Array:
    """Head tracking cost used by the original standing_policy branch."""
    head_command = command[3:]
    head_position = joints_qpos[5:9]
    return jp.nan_to_num(jp.sum(jp.square(head_position - head_command)))


def default_config() -> config_dict.ConfigDict:
    """The upstream Open Duck standing configuration."""
    calibration = NubzukiCalibration()
    head_ranges = {name: calibration.limits_rad(name) for name in HEAD_JOINTS}
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.25,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,
            action_min_delay=0,
            action_max_delay=3,
            imu_min_delay=0,
            imu_max_delay=3,
            scales=config_dict.create(
                hip_pos=0.03,
                knee_pos=0.05,
                ankle_pos=0.08,
                joint_vel=2.5,
                gravity=0.1,
                linvel=0.1,
                gyro=0.05,
                accelerometer=0.005,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                orientation=-0.5,
                torques=-1.0e-3,
                action_rate=-0.375,
                stand_still=-0.3,
                alive=20.0,
                head_pos=-2.0,
            ),
            tracking_sigma=0.01,
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[0.5, 4.0],
            force_range_n=[3.0, 20.0],
            duration_range_s=[0.08, 0.20],
        ),
        neck_pitch_range=list(head_ranges["neck_pitch"]),
        head_pitch_range=list(head_ranges["head_pitch"]),
        head_yaw_range=list(head_ranges["head_yaw"]),
        head_roll_range=list(head_ranges["head_roll"]),
        head_range_factor=1.0,
    )


class Standing(NubzukiEnv):
    """Open Duck's standing environment with Nubzuki model indexing."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        home = self._mj_model.keyframe("home")
        self._init_q = jp.array(home.qpos)
        self._default_actuator = jp.array(home.ctrl)
        self._actuators = self._mj_model.nu

        joint_ids = np.asarray(self._mj_model.actuator_trnid[:, 0])
        lowers, uppers = self._mj_model.jnt_range[joint_ids].T
        center = (lowers + uppers) / 2
        joint_range = uppers - lowers
        factor = self._config.soft_joint_pos_limit_factor
        self._soft_lowers = jp.array(center - 0.5 * joint_range * factor)
        self._soft_uppers = jp.array(center + 0.5 * joint_range * factor)

        self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id
        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in constants.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in constants.FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in constants.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        qpos_noise_scale = np.zeros(self._actuators)
        for index, name in enumerate(self.actuator_names):
            if "_hip" in name:
                qpos_noise_scale[index] = self._config.noise_config.scales.hip_pos
            elif "_knee" in name:
                qpos_noise_scale[index] = self._config.noise_config.scales.knee_pos
            elif "_ankle" in name:
                qpos_noise_scale[index] = self._config.noise_config.scales.ankle_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)
        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[0:2].add(dxy)

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        base_qpos = base_qpos.at[3:7].set(
            math.quat_mul(base_qpos[3:7], quat)
        )
        qpos = self.set_floating_base_qpos(base_qpos, qpos)

        rng, key = jax.random.split(rng)
        joint_qpos = self.get_actuator_joints_qpos(qpos)
        # Upstream multiplies a non-zero home pose. Nubzuki's logical home is
        # zero, so bounded additive noise preserves the intended reset spread.
        joint_qpos += jax.random.uniform(
            key, (self._actuators,), minval=-1.0, maxval=1.0
        ) * self._qpos_noise_scale
        joint_qpos = jp.clip(joint_qpos, self._soft_lowers, self._soft_uppers)
        qpos = self.set_actuator_joints_qpos(joint_qpos, qpos)

        rng, key = jax.random.split(rng)
        qvel = self.set_floating_base_qvel(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5), qvel
        )
        ctrl = self.get_actuator_joints_qpos(qpos)
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)

        rng, cmd_rng = jax.random.split(rng)
        command = self.sample_command(cmd_rng)
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        info = {
            "rng": rng,
            "step": 0,
            "command": command,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            "push_remaining_steps": jp.array(0, dtype=jp.int32),
            "action_history": jp.zeros(
                self._config.noise_config.action_max_delay * self._actuators
            ),
            "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
            "imitation_i": 0,
            "current_reference_motion": jp.zeros(0),
        }

        metrics = {}
        for name, scale in self._config.reward_config.scales.items():
            if scale != 0:
                prefix = "reward" if scale > 0 else "cost"
                metrics[f"{prefix}/{name}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        contact = self._contacts(data)
        obs = self._get_obs(data, info, contact)
        return mjx_env.State(
            data, obs, jp.zeros(()), jp.zeros(()), metrics, info
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state.info["imitation_i"] = 0
        state.info["current_reference_motion"] = jp.zeros(0)
        (
            state.info["rng"],
            push1_rng,
            push2_rng,
            push_duration_rng,
            push_interval_rng,
            delay_rng,
        ) = jax.random.split(
            state.info["rng"], 6
        )

        action_history = (
            jp.roll(state.info["action_history"], self._actuators)
            .at[: self._actuators]
            .set(action)
        )
        state.info["action_history"] = action_history
        action_idx = jax.random.randint(
            delay_rng,
            (1,),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        delayed_action = action_history.reshape((-1, self._actuators))[
            action_idx[0]
        ]

        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_force_n = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.force_range_n[0],
            maxval=self._config.push_config.force_range_n[1],
        )
        push_event = (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
            == 0
        )
        push_event &= self._config.push_config.enable
        sampled_push = (
            jp.array([jp.cos(push_theta), jp.sin(push_theta)]) * push_force_n
        )
        push_duration_s = jax.random.uniform(
            push_duration_rng,
            minval=self._config.push_config.duration_range_s[0],
            maxval=self._config.push_config.duration_range_s[1],
        )
        sampled_duration_steps = jp.maximum(
            jp.round(push_duration_s / self.dt).astype(jp.int32), 1
        )
        remaining_steps = jp.where(
            push_event,
            sampled_duration_steps,
            state.info["push_remaining_steps"],
        )
        push = jp.where(push_event, sampled_push, state.info["push"])
        applied_push = jp.where(remaining_steps > 0, push, jp.zeros(2))
        wrench = jp.concatenate([applied_push, jp.zeros(4)])
        xfrc_applied = state.data.xfrc_applied.at[self._torso_body_id].set(wrench)
        state = state.replace(
            data=state.data.replace(xfrc_applied=xfrc_applied)
        )

        motor_targets = (
            self._default_actuator
            + delayed_action * self._config.action_scale
        )
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )
        state.info["motor_targets"] = motor_targets

        contact = self._contacts(data)
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        feet_height = data.site_xpos[self._feet_site_id][..., -1]
        state.info["swing_peak"] = jp.maximum(
            state.info["swing_peak"], feet_height
        )

        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)
        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards = {
            name: value * self._config.reward_config.scales[name]
            for name, value in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        remaining_steps = jp.maximum(remaining_steps - 1, 0)
        state.info["push"] = jp.where(remaining_steps > 0, push, jp.zeros(2))
        state.info["push_remaining_steps"] = remaining_steps
        state.info["step"] += 1
        state.info["push_step"] = jp.where(
            push_event, 0, state.info["push_step"] + 1
        )
        next_push_interval = jax.random.uniform(
            push_interval_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        next_push_interval_steps = jp.maximum(
            jp.round(next_push_interval / self.dt).astype(jp.int32), 1
        )
        state.info["push_interval_steps"] = jp.where(
            push_event,
            next_push_interval_steps,
            state.info["push_interval_steps"],
        )
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500), 0, state.info["step"]
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for name, value in rewards.items():
            scale = self._config.reward_config.scales[name]
            if scale != 0:
                prefix = "reward" if scale > 0 else "cost"
                state.metrics[f"{prefix}/{name}"] = value if scale > 0 else -value
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(reward.dtype),
        )

    def _contacts(self, data: mjx.Data) -> jax.Array:
        return jp.array(
            [
                _geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall = self.get_gravity(data)[-1] < 0.0
        return fall | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    def _get_obs(
        self,
        data: mjx.Data,
        info: dict[str, Any],
        contact: jax.Array,
    ) -> mjx_env.Observation:
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = gyro + (
            2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1
        ) * self._config.noise_config.level * self._config.noise_config.scales.gyro

        accelerometer = self.get_accelerometer(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_accelerometer = accelerometer + (
            2 * jax.random.uniform(noise_rng, shape=accelerometer.shape) - 1
        ) * self._config.noise_config.level * self._config.noise_config.scales.accelerometer

        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = gravity + (
            2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1
        ) * self._config.noise_config.level * self._config.noise_config.scales.gravity
        imu_history = jp.roll(info["imu_history"], 3).at[:3].set(noisy_gravity)
        info["imu_history"] = imu_history
        imu_idx = jax.random.randint(
            noise_rng,
            (1,),
            minval=self._config.noise_config.imu_min_delay,
            maxval=self._config.noise_config.imu_max_delay,
        )
        noisy_gravity = imu_history.reshape((-1, 3))[imu_idx[0]]
        del noisy_gravity

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = joint_angles + (
            2.0 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1.0
        ) * self._config.noise_config.level * self._qpos_noise_scale

        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = joint_vel + (
            2.0 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1.0
        ) * self._config.noise_config.level * self._config.noise_config.scales.joint_vel

        linvel = self.get_local_linvel(data)
        state = jp.hstack(
            [
                noisy_gyro,
                noisy_accelerometer,
                info["command"],
                noisy_joint_angles - self._default_actuator,
                noisy_joint_vel * self._config.dof_vel_scale,
                info["last_act"],
                info["last_last_act"],
                info["last_last_last_act"],
                contact,
                info["current_reference_motion"],
            ]
        )

        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        privileged_state = jp.hstack(
            [
                state,
                gyro,
                accelerometer,
                gravity,
                linvel,
                self.get_global_angvel(data),
                joint_angles - self._default_actuator,
                joint_vel,
                data.qpos[self._floating_base_qpos_addr + 2],
                data.actuator_force,
                contact,
                feet_vel,
                info["feet_air_time"],
                info["current_reference_motion"],
            ]
        )
        return {"state": state, "privileged_state": privileged_state}

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics, done, first_contact, contact
        return {
            "orientation": cost_orientation(self.get_gravity(data)),
            "torques": cost_torques(data.actuator_force),
            "action_rate": cost_action_rate(action, info["last_act"]),
            "alive": reward_alive(),
            "stand_still": cost_stand_still(
                info["command"],
                self.get_actuator_joints_qpos(data.qpos),
                self.get_actuator_joints_qvel(data.qvel),
                self._default_actuator,
                True,
            ),
            "head_pos": _cost_head_pos(
                self.get_actuator_joints_qpos(data.qpos),
                info["command"],
            ),
        }

    def sample_command(self, rng: jax.Array) -> jax.Array:
        _, _, _, zero_rng, neck_rng, pitch_rng, yaw_rng, roll_rng = (
            jax.random.split(rng, 8)
        )
        factor = self._config.head_range_factor
        neck_pitch = jax.random.uniform(
            neck_rng,
            minval=self._config.neck_pitch_range[0] * factor,
            maxval=self._config.neck_pitch_range[1] * factor,
        )
        head_pitch = jax.random.uniform(
            pitch_rng,
            minval=self._config.head_pitch_range[0] * factor,
            maxval=self._config.head_pitch_range[1] * factor,
        )
        head_yaw = jax.random.uniform(
            yaw_rng,
            minval=self._config.head_yaw_range[0] * factor,
            maxval=self._config.head_yaw_range[1] * factor,
        )
        head_roll = jax.random.uniform(
            roll_rng,
            minval=self._config.head_roll_range[0] * factor,
            maxval=self._config.head_roll_range[1] * factor,
        )
        return jp.where(
            jax.random.bernoulli(zero_rng, p=0.1),
            jp.zeros(7),
            jp.hstack(
                [0.0, 0.0, 0.0, neck_pitch, head_pitch, head_yaw, head_roll]
            ),
        )
