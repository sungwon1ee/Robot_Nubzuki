"""Register the first Nubzuki BAM locomotion task."""

from copy import deepcopy
from dataclasses import dataclass

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

# MicroDuck's pinned task revision still spells the old voltage-sag option
# ``vin_drop_gain_range`` while current BAM spells it
# ``vin_drop_resistance_range``. Its own robot configs are registered as a
# dependency before ours, so accept the retired keyword while loading them.
# Nubzuki itself uses the current resistance-based API below.
import mjlab_microduck.actuator as microduck_actuator


@dataclass(kw_only=True)
class _LegacyFrictionCfg(microduck_actuator.FrictionDRBamActuatorCfg):
    vin_drop_gain_range: tuple[float, float] | None = None


@dataclass(kw_only=True)
class _LegacyBacklashCfg(microduck_actuator.BacklashEncoderBamActuatorCfg):
    vin_drop_gain_range: tuple[float, float] | None = None


microduck_actuator.FrictionDRBamActuatorCfg = _LegacyFrictionCfg
microduck_actuator.BacklashEncoderBamActuatorCfg = _LegacyBacklashCfg

from mjlab_microduck.tasks import microduck_velocity_env_cfg as velocity_module
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg

from .robot import NUBZUKI_BAM_ROBOT_CFG


class NubzukiOnPolicyRunner(VelocityOnPolicyRunner):
    """Standard MJLab velocity runner without MicroDuck-specific export hooks."""


def make_nubzuki_bam_env_cfg(play: bool = False):
    # The factory reads this module global when constructing the scene. Replace
    # only the entity; reward/observation/event code remains upstream MicroDuck.
    original_robot = velocity_module.MICRODUCK_WALK_ROBOT_CFG
    velocity_module.MICRODUCK_WALK_ROBOT_CFG = NUBZUKI_BAM_ROBOT_CFG
    try:
        cfg = velocity_module.make_microduck_velocity_env_cfg(play=play)
    finally:
        velocity_module.MICRODUCK_WALK_ROBOT_CFG = original_robot

    cfg.viewer.body_name = "trunk_base"
    cfg.sim.nconmax = 100
    cfg.sim.naconmax = 200

    # Nubzuki's neutral root height is encoded in its XML at 0.205 m.
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)

    # The clean walking policy contract: forward + curved turns. Keep
    # MicroDuck's standing curriculum: 2% initially, then 5/10/15/20/25% as
    # the gait matures instead of taxing gait discovery with 20% idle samples.
    # No reverse, lateral motion, turn-in-place or head command in this stage.
    twist = cfg.commands["twist"]
    twist.rel_turn_in_place_envs = 0.0
    twist.ranges.lin_vel_x = (0.04, 0.18)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (-0.70, 0.70)

    # Hold all four head joints at neutral. Head control gets a later task after
    # the BAM gait transfers to hardware.
    head = cfg.commands["head_pose"]
    head.ranges = ((0.0, 0.0),) * 4
    cfg.curriculum.pop("head_pose_range", None)
    cfg.curriculum.pop("head_pose_bias_weight", None)
    cfg.rewards["head_pose_bias"].weight = 1.0
    body = cfg.commands["body_pose"]
    body.ranges = ((0.0, 0.0),) * 6
    cfg.curriculum.pop("body_pose_range", None)
    # The Nubzuki XML has no subtree-angular-momentum sensor. Body angular
    # velocity remains active, so remove this tiny redundant regularizer.
    cfg.rewards.pop("angular_momentum", None)

    # Nubzuki names its head bodies differently, so remove only MicroDuck's
    # head-assembly CoM event. Torso mass/CoM, friction, armature, encoder and
    # IMU randomization remain enabled.
    cfg.events.pop("randomize_head_com", None)
    cfg.curriculum.pop("head_com_range", None)

    # Nubzuki actions are desired joint deltas in radians. Keep exploration
    # inside its much smaller mechanical joint ranges.
    cfg.actions["joint_pos"].scale = 0.25

    if play:
        cfg.scene.num_envs = min(cfg.scene.num_envs, 16)
    return cfg


NUBZUKI_BAM_RL_CFG = deepcopy(MicroduckRlCfg)
NUBZUKI_BAM_RL_CFG.wandb_project = "mjlab_nubzuki"
NUBZUKI_BAM_RL_CFG.experiment_name = "velocity_bam"
NUBZUKI_BAM_RL_CFG.run_name = "sts3215_m6_delay3_6"

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-BAM-Nubzuki",
    env_cfg=make_nubzuki_bam_env_cfg(),
    play_env_cfg=make_nubzuki_bam_env_cfg(play=True),
    rl_cfg=NUBZUKI_BAM_RL_CFG,
    runner_cls=NubzukiOnPolicyRunner,
)
