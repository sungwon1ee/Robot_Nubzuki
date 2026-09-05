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

from mjlab_microduck.tasks.backlash import make_backlash_variant

from .drive_mirror import RunMirror
from .robot import (
    NUBZUKI_BAM_DETAILED_BACKLASH_ROBOT_CFG,
    NUBZUKI_BAM_DETAILED_ROBOT_CFG,
    NUBZUKI_BAM_BACKLASH_ROBOT_CFG,
    NUBZUKI_BAM_ROBOT_CFG,
)


class NubzukiOnPolicyRunner(VelocityOnPolicyRunner):
    """Standard MJLab velocity runner without MicroDuck-specific export hooks.

    Adds one behaviour: when NUBZUKI_MIRROR_DIR is set (Colab -> Google Drive),
    every checkpoint save also copies the run directory there in the
    background, so an interrupted session never loses more than one save
    interval and TensorBoard can watch the Drive copy directly.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        self._mirror = RunMirror(log_dir) if log_dir is not None else None

    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        if self._mirror is not None:
            self._mirror.sync(path)

    def learn(self, *args, **kwargs):
        try:
            return super().learn(*args, **kwargs)
        finally:
            if self._mirror is not None and self._mirror.enabled:
                print("[INFO] Flushing final mirror copy...")
                self._mirror.flush()


def make_nubzuki_bam_env_cfg(play: bool = False):
    # The factory reads this module global when constructing the scene. Replace
    # only the entity; reward/observation/event code remains upstream MicroDuck.
    original_robot = velocity_module.MICRODUCK_WALK_ROBOT_CFG
    robot_cfg = NUBZUKI_BAM_DETAILED_ROBOT_CFG if play else NUBZUKI_BAM_ROBOT_CFG
    velocity_module.MICRODUCK_WALK_ROBOT_CFG = robot_cfg
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
    # Stage 2 command envelope. Stage 1 trained forward-only (0.04..0.18) with
    # no reverse and no turn-in-place, which is not a usable joystick
    # contract. This adds reverse and spin-on-the-spot. What it does NOT change
    # is the yaw rate: +/-0.70 was measured to be fast enough on the stage-1
    # policy, so widening it would only ask for turns the robot has no reason
    # to make. Reverse is deliberately slower than forward. Lateral stays
    # disabled: a two-stick phone page has no axis left to command it, so
    # training it would only spend capacity on a mode the robot is never asked
    # for.
    twist = cfg.commands["twist"]
    twist.rel_turn_in_place_envs = 0.10
    twist.ranges.lin_vel_x = (-0.15, 0.25)
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
    # Head commands are deltas from the calibrated park pose, and the park pose
    # is not centred in the joint range, so the reachable delta is asymmetric.
    # These are ~90% of the room left between park and the 0.9 soft limit --
    # the full usable envelope, commanded from iteration 0 rather than grown by
    # a curriculum whose MicroDuck schedule (500-2000 iters) does not fit this
    # run. Stage 1 kept these at +/-0.03 only to stop the input neurons dying.
    head = cfg.commands["head_pose"]
    head.ranges = (
        # The charger blocks downward travel, so neither pitch is ever
        # commanded below park (see NO_DOWNWARD_TRAVEL in robot.py, which moves
        # the joint limit to match -- the command range alone would not stop
        # the policy's own action from pushing into the obstruction).
        (0.00, 0.47),   # neck_pitch: park -11.9 deg, up to +26.9 deg
        (0.00, 0.33),   # head_pitch: park -8.1 deg, up to +18.9 deg
        (-0.50, 0.50),  # head_yaw: park 0 deg, room +/-31.5 deg
        (-0.17, 0.17),  # head_roll: park 0 deg, room +/-10.8 deg
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
        cfg.events["reset_base"].params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        # Detailed SDF bodies generate more candidate matches than the
        # lightweight training proxies. Keep foot-contact observations intact
        # instead of truncating them at MuJoCo Warp's default of 64.
        cfg.sim.contact_sensor_maxmatch = 128
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

# Its own experiment directory: backlash checkpoints are not interchangeable
# with play-free ones, and mixing them under one logs/ tree invites loading the
# wrong model.
NUBZUKI_BAM_BACKLASH_RL_CFG = deepcopy(NUBZUKI_BAM_RL_CFG)
NUBZUKI_BAM_BACKLASH_RL_CFG.experiment_name = "velocity_bam_backlash"
NUBZUKI_BAM_BACKLASH_RL_CFG.run_name = "sts3215_m6_backlash1deg"

def make_nubzuki_bam_backlash_env_cfg(play: bool = False):
    """Same task with +/-1 deg of gear play in series with every servo.

    The real STS3215 gearbox has play and its magnetic encoder sits on the
    output side of it, so the firmware position loop is blind while the servo
    winds through the dead zone. MicroDuck models exactly this as a separate
    task family; this is Nubzuki's. Registered alongside the play-free task
    rather than replacing it, so the two stay comparable.

    Observation and action dimensions are unchanged (still 14 joints): the
    policy sees qpos[servo] + qpos[backlash], the encoder's view.
    """
    cfg = make_nubzuki_bam_env_cfg(play=play)
    robot_cfg = (
        NUBZUKI_BAM_DETAILED_BACKLASH_ROBOT_CFG
        if play
        else NUBZUKI_BAM_BACKLASH_ROBOT_CFG
    )
    return make_backlash_variant(cfg, robot_cfg)


register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-BAM-Nubzuki",
    env_cfg=make_nubzuki_bam_env_cfg(),
    play_env_cfg=make_nubzuki_bam_env_cfg(play=True),
    rl_cfg=NUBZUKI_BAM_RL_CFG,
    runner_cls=NubzukiOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Backlash-BAM-Nubzuki",
    env_cfg=make_nubzuki_bam_backlash_env_cfg(),
    play_env_cfg=make_nubzuki_bam_backlash_env_cfg(play=True),
    rl_cfg=NUBZUKI_BAM_BACKLASH_RL_CFG,
    runner_cls=NubzukiOnPolicyRunner,
)
