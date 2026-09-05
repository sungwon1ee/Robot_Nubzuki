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

    # reset_base samples offsets around HOME_FRAME. Keep its z offset at zero;
    # the absolute 0.205 m standing height lives in the entity initial state.
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)

    # The clean walking policy contract: forward + curved turns. Keep
    # MicroDuck's standing curriculum: 2% initially, then 5/10/15/20/25% as
    # the gait matures instead of taxing gait discovery with 20% idle samples.
    # No reverse, lateral motion or turn-in-place in this stage. Head commands
    # stay deliberately tiny, but non-zero, so those policy inputs do not die
    # before a later head-control curriculum.
    twist = cfg.commands["twist"]
    twist.rel_turn_in_place_envs = 0.0
    twist.ranges.lin_vel_x = (0.04, 0.18)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (-0.70, 0.70)

    # At 4096 envs, one PPO iteration contains 24 control steps per env. The
    # original MicroDuck schedule does not change until iteration 500, beyond
    # this run's ~306 iterations. Nubzuki's weaker BAM-driven stance needs a
    # short balance-first phase before most environments receive gait commands.
    cfg.curriculum["standing_envs"].params["standing_stages"] = [
        {"step": 0, "rel_standing_envs": 1.00},
        {"step": 50 * 24, "rel_standing_envs": 0.60},
        {"step": 100 * 24, "rel_standing_envs": 0.30},
        {"step": 150 * 24, "rel_standing_envs": 0.15},
        {"step": 220 * 24, "rel_standing_envs": 0.10},
    ]
    twist.rel_standing_envs = 1.0

    # Commands are deltas from the calibrated park pose, not absolute angles.
    # Keep 25% exact-neutral samples so the policy also learns to hold HOME.
    head = cfg.commands["head_pose"]
    head.ranges = (
        (-0.03, 0.03),  # neck_pitch: +/- 1.7 deg
        (-0.03, 0.03),  # head_pitch: +/- 1.7 deg
        (-0.05, 0.05),  # head_yaw: +/- 2.9 deg
        (-0.01, 0.01),  # head_roll: +/- 0.6 deg
    )
    head.zero_command_prob = 0.25
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
        # Playback should show the policy/home pose under deterministic nominal
        # conditions. Upstream's play config still contains training pushes and
        # domain randomization, which makes a zero-agent HOME check misleading.
        for event_name in (
            "push_robot",
            "randomize_com",
            "randomize_joint_friction",
            "randomize_armature",
            "foot_friction",
            "encoder_bias",
            "base_com",
            "randomize_mass_inertia",
        ):
            cfg.events.pop(event_name, None)
        cfg.curriculum = {}
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
