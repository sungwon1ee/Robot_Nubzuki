"""Names and model paths for Nubzuki."""

from etils import epath


ROOT_PATH = epath.Path(__file__).parent
FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_flat_terrain.xml"
DETAILED_SCENE_XML = ROOT_PATH.parent.parent.parent / "Nubzuki" / "mjcf" / "scene.xml"


def simulation_xml() -> epath.Path:
    """Use the detailed CAD visuals for native MuJoCo when available.

    Training remains pinned to the lightweight MJX model above.  The detailed
    model lives in the adjacent Nubzuki CAD project in the desktop workspace.
    """
    return DETAILED_SCENE_XML if DETAILED_SCENE_XML.exists() else FLAT_TERRAIN_XML


def task_to_xml(task_name: str) -> epath.Path:
    if task_name != "flat_terrain":
        raise ValueError(f"Unsupported Nubzuki task: {task_name}")
    return FLAT_TERRAIN_XML


FEET_SITES = ["left_foot", "right_foot"]
LEFT_FEET_GEOMS = ["left_foot_collision"]
RIGHT_FEET_GEOMS = ["right_foot_collision"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS
FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

HIP_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
]
KNEE_JOINT_NAMES = ["left_knee", "right_knee"]
JOINTS_ORDER_NO_HEAD = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

ROOT_BODY = "trunk"
GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
