"""Film a crowd of independently commanded Nubzuki policies in MJLab.

Each robot is one parallel MJLab environment.  The native viewer composites
those environments into one shot, so this is inexpensive and uses the exact
same actor checkpoint for every robot.  Robots are visually co-located but
remain in independent physics worlds; they do not collide with one another.

Example (macOS):

    uv run mjpython src/mjlab_nubzuki/play_crowd.py \
      --checkpoint checkpoints/walking_v2/model_3950.pt \
      --robots 36
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


TASK_ID = "Mjlab-Velocity-Flat-BAM-Nubzuki-Crowd"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/walking_v2/model_3950.pt"),
    )
    parser.add_argument("--robots", type=int, default=36)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-32.0)
    parser.add_argument(
        "--command-arrows",
        action="store_true",
        help="Show each robot's target velocity arrow.",
    )
    parser.add_argument(
        "--keep-overlay",
        action="store_true",
        help="Keep MJLab's status text in the top-left corner.",
    )
    return parser.parse_args()


# Background/lighting taken from Nubzuki/mjcf/scene.xml, the brighter standalone
# viewer scene. Only the look is copied; physics and camera stay as configured
# below.
BRIGHT_HEADLIGHT_DIFFUSE = (0.78, 0.78, 0.78)
BRIGHT_HEADLIGHT_AMBIENT = (0.42, 0.42, 0.42)
BRIGHT_HEADLIGHT_SPECULAR = (0.08, 0.08, 0.08)
BRIGHT_HAZE = (0.88, 0.90, 0.96, 1.0)
BRIGHT_SKY_TOP = (0.72, 0.78, 0.94)
BRIGHT_SKY_BOTTOM = (0.96, 0.86, 0.90)
BRIGHT_GROUND_RGB1 = (0.63, 0.67, 0.76)
BRIGHT_GROUND_RGB2 = (0.54, 0.58, 0.68)
BRIGHT_GROUND_MARK = (0.46, 0.50, 0.60)


def _apply_bright_scene(spec) -> None:
    """Repaint the crowd scene with scene.xml's palette."""
    import mujoco

    visual = spec.visual
    visual.headlight.diffuse = list(BRIGHT_HEADLIGHT_DIFFUSE)
    visual.headlight.ambient = list(BRIGHT_HEADLIGHT_AMBIENT)
    visual.headlight.specular = list(BRIGHT_HEADLIGHT_SPECULAR)
    visual.rgba.haze = list(BRIGHT_HAZE)

    # mjlab's base scene has no skybox at all, so the viewer falls back to a
    # flat dark backdrop. Add scene.xml's pastel gradient.
    if not any(
        tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX for tex in spec.textures
    ):
        sky = spec.add_texture()
        sky.name = "skybox"
        sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
        sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        sky.rgb1 = list(BRIGHT_SKY_TOP)
        sky.rgb2 = list(BRIGHT_SKY_BOTTOM)
        sky.width = 512
        sky.height = 3072

    # Lighten mjlab's near-black checker floor to scene.xml's cool grey.
    for texture in spec.textures:
        if texture.name != "groundplane":
            continue
        texture.rgb1 = list(BRIGHT_GROUND_RGB1)
        texture.rgb2 = list(BRIGHT_GROUND_RGB2)
        texture.markrgb = list(BRIGHT_GROUND_MARK)
    for material in spec.materials:
        if material.name != "groundplane":
            continue
        material.reflectance = 0.08
        material.texrepeat = [5.0, 5.0]


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if args.robots < 1:
        raise SystemExit("--robots must be at least 1")
    if args.spacing <= 0:
        raise SystemExit("--spacing must be positive")

    # Importing this module also installs the compatibility shim needed by the
    # pinned MicroDuck task before its task configs are constructed.
    import mjlab_nubzuki.tasks as nubzuki_tasks
    from mjlab.scripts import play as play_module
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.viewer import NativeMujocoViewer
    from mjlab.viewer.viewer_config import ViewerConfig
    from mjlab_nubzuki.run_config import find_env_yaml, trained_command_ranges

    env_cfg = nubzuki_tasks.make_nubzuki_bam_env_cfg(play=True)
    env_cfg.scene.num_envs = args.robots
    env_cfg.scene.env_spacing = args.spacing
    env_cfg.scene.spec_fn = _apply_bright_scene
    # Render the detailed CAD exterior, but calculate contacts with the same
    # cheap boxes/capsules used during training. This removes the crowd-scale
    # SDF bottleneck without changing what appears on camera.
    from copy import deepcopy
    from mjlab_nubzuki.robot import (
        NUBZUKI_BAM_DETAILED_VISUAL_LIGHT_COLLISION_ROBOT_CFG,
    )

    env_cfg.scene.entities["robot"] = deepcopy(
        NUBZUKI_BAM_DETAILED_VISUAL_LIGHT_COLLISION_ROBOT_CFG
    )
    env_cfg.sim.contact_sensor_maxmatch = 64

    # Keep commands faithful to the run even if today's task defaults change.
    env_yaml = find_env_yaml(checkpoint)
    if env_yaml is None:
        raise SystemExit(
            "No params/env.yaml found beside the checkpoint. Copy the training "
            "run's params directory next to it before filming."
        )
    ranges = trained_command_ranges(env_yaml)
    twist = env_cfg.commands["twist"]
    twist.ranges.lin_vel_x = tuple(ranges["twist_lin_vel_x"])
    twist.ranges.lin_vel_y = tuple(ranges["twist_lin_vel_y"])
    twist.ranges.ang_vel_z = tuple(ranges["twist_ang_vel_z"])
    twist.debug_vis = args.command_arrows
    env_cfg.commands["head_pose"].ranges = tuple(
        tuple(pair) for pair in ranges["head_pose"]
    )

    # A free world camera keeps the whole square formation in frame.
    side = math.ceil(math.sqrt(args.robots))
    formation_width = max(1.0, (side - 1) * args.spacing)
    env_cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
    env_cfg.viewer.lookat = (0.0, 0.0, 0.12)
    env_cfg.viewer.distance = args.distance or max(3.0, formation_width * 1.35)
    env_cfg.viewer.azimuth = args.azimuth
    env_cfg.viewer.elevation = args.elevation
    env_cfg.viewer.max_extra_envs = args.robots
    env_cfg.viewer.enable_reflections = False
    env_cfg.viewer.enable_shadows = False

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg,
        play_env_cfg=env_cfg,
        rl_cfg=nubzuki_tasks.NUBZUKI_BAM_RL_CFG,
        runner_cls=nubzuki_tasks.NubzukiOnPolicyRunner,
    )

    class CrowdViewer(NativeMujocoViewer):
        def __init__(self, *viewer_args, **viewer_kwargs):
            super().__init__(*viewer_args, **viewer_kwargs)
            self._show_all_envs = True

        def _set_status_overlay(self, viewer):
            if args.keep_overlay:
                super()._set_status_overlay(viewer)

    # run_play resolves this symbol from its own module.
    play_module.NativeMujocoViewer = CrowdViewer
    print(
        f"Filming {args.robots} robots from {checkpoint.name} "
        "(detailed visuals, lightweight collisions)\n"
        f"Commands: forward {tuple(ranges['twist_lin_vel_x'])}, "
        f"yaw {tuple(ranges['twist_ang_vel_z'])}\n"
        "Drag to orbit, scroll to zoom, Space to pause, Enter to reset."
    )
    play_module.run_play(
        TASK_ID,
        play_module.PlayConfig(
            checkpoint_file=str(checkpoint),
            num_envs=args.robots,
            viewer="native",
        ),
    )


if __name__ == "__main__":
    main()
