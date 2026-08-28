"""Open Duck domain randomization with Nubzuki model indices."""

import jax
import jax.numpy as jp
from mujoco import mjx


def domain_randomize(
    model: mjx.Model,
    rng: jax.Array,
    *,
    floor_geom_id: int,
    torso_body_id: int,
):
    """Apply the standing_policy branch's dynamics randomization ranges."""
    dof_id = jp.array(
        [
            index
            for index, has_friction in enumerate(model.dof_hasfrictionloss)
            if has_friction
        ]
    )
    joint_id = model.dof_jntid[dof_id]
    dof_addr = jp.array(
        [address for address in model.jnt_dofadr if address in dof_id]
    )
    joint_addr = model.jnt_qposadr[joint_id]

    @jax.vmap
    def randomize_one(key):
        key, subkey = jax.random.split(key)
        geom_friction = model.geom_friction.at[floor_geom_id, 0].set(
            jax.random.uniform(subkey, minval=0.5, maxval=1.0)
        )

        key, subkey = jax.random.split(key)
        frictionloss = model.dof_frictionloss[dof_addr] * jax.random.uniform(
            subkey, shape=(model.nu,), minval=0.9, maxval=1.1
        )
        dof_frictionloss = model.dof_frictionloss.at[dof_addr].set(frictionloss)

        key, subkey = jax.random.split(key)
        armature = model.dof_armature[dof_addr] * jax.random.uniform(
            subkey, shape=(model.nu,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[dof_addr].set(armature)

        key, subkey = jax.random.split(key)
        center_of_mass_offset = jax.random.uniform(
            subkey, (3,), minval=-0.05, maxval=0.05
        )
        body_ipos = model.body_ipos.at[torso_body_id].set(
            model.body_ipos[torso_body_id] + center_of_mass_offset
        )

        key, subkey = jax.random.split(key)
        mass_scale = jax.random.uniform(
            subkey, shape=(model.nbody,), minval=0.9, maxval=1.1
        )
        body_mass = model.body_mass * mass_scale

        key, subkey = jax.random.split(key)
        torso_mass_offset = jax.random.uniform(
            subkey, minval=-0.1, maxval=0.1
        )
        body_mass = body_mass.at[torso_body_id].add(torso_mass_offset)

        key, subkey = jax.random.split(key)
        qpos0 = model.qpos0.at[joint_addr].add(
            jax.random.uniform(
                subkey, shape=(model.nu,), minval=-0.03, maxval=0.03
            )
        )

        key, subkey = jax.random.split(key)
        kp_scale = jax.random.uniform(
            subkey, shape=(model.nu,), minval=0.9, maxval=1.1
        )
        current_kp = model.actuator_gainprm[:, 0]
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(
            current_kp * kp_scale
        )
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(
            -current_kp * kp_scale
        )

        return (
            geom_friction,
            body_ipos,
            dof_frictionloss,
            dof_armature,
            body_mass,
            qpos0,
            actuator_gainprm,
            actuator_biasprm,
        )

    (
        friction,
        body_ipos,
        frictionloss,
        armature,
        body_mass,
        qpos0,
        actuator_gainprm,
        actuator_biasprm,
    ) = randomize_one(rng)

    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "body_ipos": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "body_mass": 0,
            "qpos0": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )
    randomized_model = model.tree_replace(
        {
            "geom_friction": friction,
            "body_ipos": body_ipos,
            "dof_frictionloss": frictionloss,
            "dof_armature": armature,
            "body_mass": body_mass,
            "qpos0": qpos0,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
        }
    )
    return randomized_model, in_axes

