"""Nubzuki articulation configured with the measured STS3215 BAM M6 model."""

import json
import math
from pathlib import Path

import mujoco
import torch
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab_microduck.actuator import (
    FrictionDRBamActuator,
    FrictionDRBamActuatorCfg,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
NUBZUKI_XML = REPOSITORY_ROOT / "playground/nubzuki/xmls/nubzuki_mjx.xml"
NUBZUKI_DETAILED_XML = REPOSITORY_ROOT.parent / "Nubzuki/mjcf/nubzuki_v1.xml"
CALIBRATION_JSON = REPOSITORY_ROOT / "config/nubzuki_calibration.json"
STS3215_M6_JSON = Path(__file__).resolve().parent / "params/feetech_sts3215_7_4V_m6.json"
JOINT_ACTUATOR_ORDER = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
)


def _load_nubzuki_spec(xml_path: Path) -> mujoco.MjSpec:
    if not xml_path.exists():
        raise FileNotFoundError(f"Nubzuki XML not found: {xml_path}")
    spec = mujoco.MjSpec.from_file(str(xml_path))
    # MicroDuck's velocity task consistently calls the torso trunk_base.
    # Keep that contract so its battle-tested rewards and randomizers can be
    # reused without copying or forking thousands of lines of MDP code.
    spec.body("trunk").name = "trunk_base"
    # Match MJLab/MicroDuck's policy-facing IMU sensor contract. The underlying
    # site and measured quantities are unchanged.
    spec.sensor("gyro").name = "imu_ang_vel"
    spec.sensor("local_linvel").name = "imu_lin_vel"
    spec.sensor("accelerometer").name = "imu_accel"

    # BAM receives commands in articulation joint order. The original XML
    # declared head actuators between the two legs, so BAM torque column 5 was
    # applied to the neck while it represented right_hip_yaw, and subsequent
    # columns were shifted likewise. Rebuild actuators in exact joint order.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for joint_name in JOINT_ACTUATOR_ORDER:
        actuator = spec.add_actuator()
        actuator.name = joint_name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.target = joint_name
        actuator.set_to_position(13.37)
    return spec


def get_nubzuki_spec() -> mujoco.MjSpec:
    """Lightweight collision model used by vectorized training."""
    return _load_nubzuki_spec(NUBZUKI_XML)


def get_nubzuki_detailed_spec() -> mujoco.MjSpec:
    """Existing CAD visuals and SDF collision model used by local playback."""
    # The detailed CAD project is available beside this repository on the
    # development Mac. Colab only receives this repository, so retain a useful
    # playback fallback there instead of failing task registration.
    xml_path = NUBZUKI_DETAILED_XML if NUBZUKI_DETAILED_XML.exists() else NUBZUKI_XML
    spec = _load_nubzuki_spec(xml_path)
    if xml_path == NUBZUKI_XML:
        return spec

    # Collision categories: floor=1, trunk=2, left leg=4, head=8,
    # right leg=16. CAD collision geoms retain floor contact, while only the
    # two opposite legs collide with each other. Assembly-overlapping
    # trunk/leg and same-leg pairs are excluded.
    for geom in spec.geoms:
        name = geom.name
        if not name or "collision" not in name:
            continue
        if name.startswith("left_"):
            geom.contype, geom.conaffinity = 4, 1 | 16
        elif name.startswith("right_"):
            geom.contype, geom.conaffinity = 16, 1 | 4
        elif name == "trunk_collision":
            geom.contype, geom.conaffinity = 2, 1
        elif name.startswith("head"):
            geom.contype, geom.conaffinity = 8, 1

    # The raw head and trunk SDFs overlap in the assembled CAD at HOME. Keep
    # them for floor contact and use non-overlapping proxies solely for the
    # required head-to-trunk self collision.
    trunk_proxy = spec.body("trunk_base").add_geom()
    trunk_proxy.name = "trunk_head_collision_proxy"
    trunk_proxy.type = mujoco.mjtGeom.mjGEOM_BOX
    trunk_proxy.pos = [0.012, 0.0, 0.035]
    trunk_proxy.size = [0.085, 0.095, 0.075]
    trunk_proxy.contype, trunk_proxy.conaffinity = 64, 128
    trunk_proxy.rgba = [0.8, 0.2, 0.2, 0.0]

    head_proxy = spec.body("head_roll_link").add_geom()
    head_proxy.name = "head_trunk_collision_proxy"
    head_proxy.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    head_proxy.fromto = [0.079, -0.105, 0.041, 0.079, 0.105, 0.041]
    head_proxy.size = [0.105, 0.0, 0.0]
    head_proxy.contype, head_proxy.conaffinity = 128, 64
    head_proxy.rgba = [0.8, 0.2, 0.2, 0.0]
    return spec


def _load_park_pose() -> dict[str, float]:
    """Use the same calibrated neutral pose in training and on hardware."""
    with CALIBRATION_JSON.open() as stream:
        calibration = json.load(stream)
    pose = {
        name: math.radians(float(calibration["joints"][name]["park_deg"]))
        for name in calibration["joint_order"]
    }
    # A knee exactly at its 0-degree hard stop is clipped to -2.25 degrees by
    # MJLab's 0.9 soft-limit reset. Make that small bend explicit so default,
    # reset and action offsets all agree instead of starting with hidden error.
    pose["left_knee"] = math.radians(-2.25)
    pose["right_knee"] = math.radians(-2.25)
    # Train from a geometrically symmetric stance. The calibration file stores
    # per-servo hardware park offsets, but carrying its 1.31-degree ankle
    # mismatch into simulation makes the untrained robot collapse to one side.
    pose["left_ankle"] = math.radians(-3.2)
    pose["right_ankle"] = math.radians(-3.2)
    return pose


HOME_JOINT_POS = _load_park_pose()

HOME_FRAME = EntityCfg.InitialStateCfg(
    # InitialStateCfg overrides the free-joint position from the XML, so the
    # standing height must be explicit here rather than relying on base@pos.
    # The calibrated ankles plus the small knee bend lower the foot edges, so
    # 0.212 m starts just above the floor without a reset contact impulse.
    pos=(0.0, 0.0, 0.212),
    joint_pos=HOME_JOINT_POS,
    joint_vel={r".*": 0.0},
)

DETAILED_HOME_FRAME = EntityCfg.InitialStateCfg(
    # Detailed CAD feet sit about 2.5 mm above their lightweight counterparts.
    # This height gives both soles a shallow, symmetric initial floor contact.
    pos=(0.0, 0.0, 0.20945),
    joint_pos=HOME_JOINT_POS,
    joint_vel={r".*": 0.0},
)

# This is the real actuator path, not an ideal PD plus a delay buffer. BAM M6
# calculates the STS3215 firmware P loop, voltage saturation, back-EMF, motor
# torque and load-dependent friction on every MuJoCo Warp step.
class NubzukiSts3215BamActuator(FrictionDRBamActuator):
    """Initialize and reset the STS3215 firmware target from joint position.

    The fitted M6 parameters were published after the pinned MJLab BAM branch;
    that branch's STS3215 class only initialized this state when loading a
    physical testbench log. Vectorized training has no log, so defer seeding
    until compute(), where the actual post-reset joint position is available.
    """

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        self._bam_model.actuator.q_target_smooth = torch.zeros(
            (data.nworld, len(self._target_ids_list)),
            dtype=torch.float32,
            device=device,
        )
        self._reset_all_targets = True
        self._target_reset_env_ids = []

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._reset_all_targets = True
            self._target_reset_env_ids.clear()
        else:
            self._target_reset_env_ids.append(env_ids)

    def compute(self, cmd):
        target = self._bam_model.actuator.q_target_smooth
        if self._reset_all_targets:
            target.copy_(cmd.pos)
            self._reset_all_targets = False
            self._target_reset_env_ids.clear()
        else:
            for env_ids in self._target_reset_env_ids:
                target[env_ids] = cmd.pos[env_ids]
            self._target_reset_env_ids.clear()
        return super().compute(cmd)


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
    # Preserve the carefully separated collision masks from nubzuki_mjx.xml.
    # MicroDuck's FULL_COLLISION editor sets every matched geom to contype=1,
    # which re-enables Nubzuki's intentionally disabled overlapping hip-link
    # proxies and makes the robot explosively self-collide at reset.
    articulation=EntityArticulationInfoCfg(
        actuators=(ACTUATORS,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

NUBZUKI_BAM_DETAILED_ROBOT_CFG = EntityCfg(
    spec_fn=get_nubzuki_detailed_spec,
    init_state=DETAILED_HOME_FRAME,
    articulation=EntityArticulationInfoCfg(
        actuators=(ACTUATORS,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
