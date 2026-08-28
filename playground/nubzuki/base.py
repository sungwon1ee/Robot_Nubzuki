"""Nubzuki-specific MJX environment base class."""

from typing import Any, Optional

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

from playground.nubzuki import constants


class NubzukiEnv(mjx_env.MjxEnv):
    """Shared model indexing and sensor access for Nubzuki tasks."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[dict[str, Any]] = None,
    ):
        super().__init__(config, config_overrides)
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = xml_path

        free_joints = [
            index
            for index in range(self._mj_model.njnt)
            if self._mj_model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(free_joints) != 1:
            raise ValueError(f"Nubzuki requires one free joint, found {free_joints}")
        self._floating_base_id = free_joints[0]
        self._floating_base_qpos_addr = int(
            self._mj_model.jnt_qposadr[self._floating_base_id]
        )
        self._floating_base_qvel_addr = int(
            self._mj_model.jnt_dofadr[self._floating_base_id]
        )

        self.actuator_names = [
            self._mj_model.actuator(index).name
            for index in range(self._mj_model.nu)
        ]
        self._actuator_joint_ids = jp.asarray(
            self._mj_model.actuator_trnid[:, 0], dtype=jp.int32
        )
        self._actuator_qpos_addr = jp.asarray(
            self._mj_model.jnt_qposadr[self._mj_model.actuator_trnid[:, 0]],
            dtype=jp.int32,
        )
        self._actuator_qvel_addr = jp.asarray(
            self._mj_model.jnt_dofadr[self._mj_model.actuator_trnid[:, 0]],
            dtype=jp.int32,
        )

    def get_floating_base_qpos(self, qpos: jax.Array) -> jax.Array:
        start = self._floating_base_qpos_addr
        return qpos[start : start + 7]

    def get_floating_base_qvel(self, qvel: jax.Array) -> jax.Array:
        start = self._floating_base_qvel_addr
        return qvel[start : start + 6]

    def set_floating_base_qpos(
        self, value: jax.Array, qpos: jax.Array
    ) -> jax.Array:
        start = self._floating_base_qpos_addr
        return qpos.at[start : start + 7].set(value)

    def set_floating_base_qvel(
        self, value: jax.Array, qvel: jax.Array
    ) -> jax.Array:
        start = self._floating_base_qvel_addr
        return qvel.at[start : start + 6].set(value)

    def get_actuator_joints_qpos(self, qpos: jax.Array) -> jax.Array:
        return qpos[self._actuator_qpos_addr]

    def set_actuator_joints_qpos(
        self, value: jax.Array, qpos: jax.Array
    ) -> jax.Array:
        return qpos.at[self._actuator_qpos_addr].set(value)

    def get_actuator_joints_qvel(self, qvel: jax.Array) -> jax.Array:
        return qvel[self._actuator_qvel_addr]

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.GRAVITY_SENSOR
        )

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.GLOBAL_LINVEL_SENSOR
        )

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.GLOBAL_ANGVEL_SENSOR
        )

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.LOCAL_LINVEL_SENSOR
        )

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.ACCELEROMETER_SENSOR
        )

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.GYRO_SENSOR
        )

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

