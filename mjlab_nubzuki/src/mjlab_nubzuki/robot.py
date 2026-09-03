"""Nubzuki articulation configured with the measured STS3215 BAM M6 model."""

from pathlib import Path

import mujoco
import torch
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab_microduck.actuator import (
    FrictionDRBamActuator,
    FrictionDRBamActuatorCfg,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
NUBZUKI_XML = REPOSITORY_ROOT / "playground/nubzuki/xmls/nubzuki_mjx.xml"
STS3215_M6_JSON = Path(__file__).resolve().parent / "params/feetech_sts3215_7_4V_m6.json"


def get_nubzuki_spec() -> mujoco.MjSpec:
    if not NUBZUKI_XML.exists():
        raise FileNotFoundError(f"Nubzuki XML not found: {NUBZUKI_XML}")
    spec = mujoco.MjSpec.from_file(str(NUBZUKI_XML))
    # MicroDuck's velocity task consistently calls the torso trunk_base.
    # Keep that contract so its battle-tested rewards and randomizers can be
    # reused without copying or forking thousands of lines of MDP code.
    spec.body("trunk").name = "trunk_base"
    # Match MJLab/MicroDuck's policy-facing IMU sensor contract. The underlying
    # site and measured quantities are unchanged.
    spec.sensor("gyro").name = "imu_ang_vel"
    spec.sensor("local_linvel").name = "imu_lin_vel"
    spec.sensor("accelerometer").name = "imu_accel"
    return spec


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={r".*": 0.0},
    joint_vel={r".*": 0.0},
)

COLLISIONS = CollisionCfg(
    geom_names_expr=(r".*_collision",),
    condim={r"^(left|right)_foot_collision$": 3, r".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# This is the real actuator path, not an ideal PD plus a delay buffer. BAM M6
# calculates the STS3215 firmware P loop, voltage saturation, back-EMF, motor
# torque and load-dependent friction on every MuJoCo Warp step.
class NubzukiSts3215BamActuator(FrictionDRBamActuator):
    """Initialize the STS3215 firmware's rate-limited target on GPU.

    The fitted M6 parameters were published after the pinned MJLab BAM branch;
    that branch's STS3215 class only initialized this state when loading a
    physical testbench log. Vectorized training has no log, so seed it at the
    neutral joint pose and reset the selected worlds to neutral each episode.
    """

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        self._bam_model.actuator.q_target_smooth = torch.zeros(
            (data.nworld, len(self._target_ids_list)),
            dtype=torch.float32,
            device=device,
        )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        target = self._bam_model.actuator.q_target_smooth
        if env_ids is None:
            target[:] = 0.0
        else:
            target[env_ids] = 0.0


class NubzukiSts3215BamActuatorCfg(FrictionDRBamActuatorCfg):
    def build(self, entity, target_ids, target_names):
        return NubzukiSts3215BamActuator(self, entity, target_ids, target_names)


ACTUATORS = NubzukiSts3215BamActuatorCfg(
    # This JSON is copied verbatim from BAM's official STS3215 7.4 V M6
    # parameter set. Pinning it here prevents a moving BAM branch from silently
    # changing the actuator fitted to this robot.
    json_path=str(STS3215_M6_JSON),
    target_names_expr=(
        r"^(left|right)_(hip_yaw|hip_roll|hip_pitch|knee|ankle)$|"
        r"^(neck_pitch|head_pitch|head_yaw|head_roll)$",
    ),
    kp_fw=30.0,
    vin_range=(7.0, 8.2),
    vin_drop_resistance_range=(0.0, 0.20),
    vin_min=6.5,
    delay_min_lag=3,
    delay_max_lag=6,
    delay_update_period=64,
    delay_per_env_phase=True,
)

NUBZUKI_BAM_ROBOT_CFG = EntityCfg(
    spec_fn=get_nubzuki_spec,
    init_state=HOME_FRAME,
    collisions=(COLLISIONS,),
    articulation=EntityArticulationInfoCfg(
        actuators=(ACTUATORS,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
