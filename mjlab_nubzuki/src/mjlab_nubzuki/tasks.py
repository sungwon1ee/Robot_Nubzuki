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

    # Standing at zero command has no reward term of its own: upstream teaches
    # it purely through the fraction of environments whose command is zeroed,
    # ramping 2% -> 25% as the gait matures. Nubzuki used to override this with
    # a balance-first schedule (100% standing, down to 10% by iteration 220),
    # which left 80% of the run with only 10% standing data -- and a policy
    # that wandered off when told to hold still. Upstream's schedule is kept
    # as-is now; note it finishes at iteration 2000, so a shorter run stops
    # partway up the ramp.

    # Commands are deltas from the calibrated park pose, not absolute angles.
    # Keep 25% exact-neutral samples so the policy also learns to hold HOME.
    # Head commands are deltas from the calibrated park pose, and the park pose
    # is not centred in the joint range, so the reachable delta is asymmetric.
    # These are ~90% of the room left between park and the 0.9 soft limit --
    # the full usable envelope, commanded from iteration 0 rather than grown by
    # a curriculum whose MicroDuck schedule (500-2000 iters) does not fit this
    # run. Stage 1 kept these at +/-0.03 only to stop the input neurons dying.
    # Head command range, ramped the way upstream does rather than opened wide
    # from step 0: a posture objective competing with an unformed gait is what
    # a curriculum exists to avoid. Upstream's five stages (5/15/35/65/100% of
    # the final cap) are kept, rescaled to Nubzuki's much smaller reachable
    # deltas -- its final +/-1.10 rad pitch is several times this robot's whole
    # range. The two pitch lower bounds stay pinned at zero at every stage: the
    # charger under the head blocks travel below park, so the ramp may only
    # open upward (see NO_DOWNWARD_TRAVEL in robot.py, which moves the joint
    # limit to match).
    head = cfg.commands["head_pose"]
    head.ranges = (
        (0.00, 0.024),    # neck_pitch: park .. park +1.4 deg
        (0.00, 0.017),    # head_pitch: park .. park +1.0 deg
        (-0.025, 0.025),  # head_yaw:   +/-1.4 deg
        (-0.009, 0.009),  # head_roll:  +/-0.5 deg
    )
    head.zero_command_prob = 0.25
    cfg.curriculum["head_pose_range"].params["range_stages"] = [
        {"step": 0 * 24, "ranges": ((+0.000, +0.024), (+0.000, +0.017), (-0.025, +0.025), (-0.009, +0.009))},
        {"step": 500 * 24, "ranges": ((+0.000, +0.070), (+0.000, +0.050), (-0.075, +0.075), (-0.026, +0.026))},
        {"step": 1000 * 24, "ranges": ((+0.000, +0.164), (+0.000, +0.115), (-0.175, +0.175), (-0.059, +0.059))},
        {"step": 1500 * 24, "ranges": ((+0.000, +0.305), (+0.000, +0.215), (-0.325, +0.325), (-0.111, +0.111))},
        {"step": 2000 * 24, "ranges": ((+0.000, +0.470), (+0.000, +0.330), (-0.500, +0.500), (-0.170, +0.170))},
    ]

    # head_pose_bias prices the residual head droop. Upstream holds it at zero
    # until a gait exists and only then ramps it; starting at 1.0 taxed posture
    # precision from step 0.
    cfg.curriculum["head_pose_bias_weight"].params["weight_stages"] = [
        {"step": 0, "weight": 0.0},
        {"step": 600 * 24, "weight": 1.0},
        {"step": 1000 * 24, "weight": 2.0},
        {"step": 1500 * 24, "weight": 3.0},
    ]
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
