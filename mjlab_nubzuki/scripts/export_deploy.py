"""Turn an MJLab checkpoint into the two files the robot runtime needs.

Writes into an output directory (default ``deploy/<checkpoint stem>/``):

  policy.onnx   the exported actor
  policy.json   the deployment contract - observation term order and widths,
                joint order, action scale, default pose, control rate

The contract is dumped out of the environment that trained the checkpoint
rather than written by hand. An MJLab policy's observation layout is not the
85-D vector this robot's older MJX policies used, and its joint order is not
the calibration's; both are things you cannot eyeball from a .pt file, and
getting either wrong produces a robot that walks into the floor instead of an
error message.

Usage:

    cd mjlab_nubzuki
    uv run python scripts/export_deploy.py checkpoints/model_570.pt
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab_nubzuki.run_config import find_env_yaml, trained_command_ranges

DEFAULT_TASK = "Mjlab-Velocity-Flat-BAM-Nubzuki"
# Terms playground/nubzuki/mjlab_policy.py knows how to build on hardware.
SUPPORTED_TERMS = {
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "head_command",
    "body_command",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--env-yaml", type=Path, default=None,
        help="params/env.yaml from the run that produced this checkpoint; "
             "found automatically when it sits beside the checkpoint",
    )
    parser.add_argument(
        "--accept-current-ranges", action="store_true",
        help="export with today's task ranges when the training config cannot "
             "be found, accepting that they may not be what the policy learned",
    )
    args = parser.parse_args()

    checkpoint: Path = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    out_dir: Path = (
        args.out
        if args.out is not None
        else checkpoint.parent.parent / "deploy" / checkpoint.stem
    ).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    import mjlab.tasks  # noqa: F401  (populates the registry)
    import mjlab_nubzuki.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # play=True is the deployment-facing config: no pushes, no domain
    # randomization, and the same default pose the hardware will park at.
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(args.task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    runner.export_policy_to_onnx(str(out_dir), "policy.onnx")
    print(f"[export] {out_dir / 'policy.onnx'}")

    obs_manager = env.observation_manager
    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("joint_pos")

    names = obs_manager.active_terms["actor"]
    dims = obs_manager.group_obs_term_dim["actor"]
    terms = [{"name": n, "dim": int(d[0])} for n, d in zip(names, dims)]
    unknown = {t["name"] for t in terms} - SUPPORTED_TERMS
    if unknown:
        raise SystemExit(
            f"The trained policy has observation terms the hardware runtime "
            f"cannot build: {sorted(unknown)}. Teach mjlab_policy.py about "
            f"them before deploying."
        )

    # Only the joints the action term drives belong in the contract. On the
    # backlash model the entity also carries 14 unactuated passive_*_backlash
    # hinges, so robot.joint_names is 28 long while the policy's action and
    # joint observations are 14 -- exporting the full list produced a contract
    # the hardware rejected as "joint set does not match the calibration".
    action_ids = list(action_term._target_ids)
    joint_order = [robot.joint_names[i] for i in action_ids]
    default_joint_pos = robot.data.default_joint_pos[0, action_ids].tolist()

    # The observation must be built over the same joints in the same order, or
    # the runtime feeds the policy a permuted vector that still has the right
    # shape. Compare against the selection the joint_pos term actually uses.
    obs_term = obs_manager._group_obs_term_cfgs["actor"][names.index("joint_pos")]
    obs_asset = obs_term.params.get("asset_cfg")
    if obs_asset is not None and obs_asset.joint_ids not in (slice(None), None):
        obs_ids = list(obs_asset.joint_ids)
        if obs_ids != action_ids:
            raise SystemExit(
                f"joint_pos observes {[robot.joint_names[i] for i in obs_ids]} "
                f"but actions drive {joint_order}; the runtime cannot line those "
                f"up. Fix the task config before deploying."
            )

    # Command ranges decide what a full stick deflection asks for. They must
    # come from the run that trained this checkpoint, not from whatever the
    # task config says today.
    env_yaml = args.env_yaml or find_env_yaml(checkpoint)
    if env_yaml is not None:
        command_ranges = trained_command_ranges(env_yaml)
        ranges_source = str(env_yaml)
        print(f"[ranges] from the training config: {env_yaml}")
    elif args.accept_current_ranges:
        command_ranges = {
            "twist_lin_vel_x": list(env.cfg.commands["twist"].ranges.lin_vel_x),
            "twist_lin_vel_y": list(env.cfg.commands["twist"].ranges.lin_vel_y),
            "twist_ang_vel_z": list(env.cfg.commands["twist"].ranges.ang_vel_z),
            "head_pose": [list(r) for r in env.cfg.commands["head_pose"].ranges],
        }
        ranges_source = "current task config (NOT verified against the checkpoint)"
        print("[ranges] WARNING: using today's task config, not the training run")
    else:
        raise SystemExit(
            f"No params/env.yaml found next to {checkpoint}, so the command "
            f"ranges this policy was trained with are unknown. Copy the run's "
            f"params/ directory next to the checkpoint, pass --env-yaml, or "
            f"re-run with --accept-current-ranges if you are certain today's "
            f"task config still matches this checkpoint."
        )

    contract = {
        "framework": "mjlab",
        "deployable": True,
        "task": args.task,
        "checkpoint": checkpoint.name,
        "control_frequency_hz": int(round(1.0 / env.step_dt)),
        "observation_size": int(obs_manager.group_obs_dim["actor"][0]),
        "action_size": len(joint_order),
        "observation": terms,
        "joint_order": joint_order,
        "default_joint_pos": default_joint_pos,
        "action_scale": float(action_term.cfg.scale),
        "command_ranges": command_ranges,
        "command_ranges_source": ranges_source,
    }
    contract_path = out_dir / "policy.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"[export] {contract_path}")

    print()
    print(f"observation ({contract['observation_size']}D), in order:")
    for term in terms:
        print(f"  {term['dim']:>3}  {term['name']}")
    print(f"action: {contract['action_size']}, scale {contract['action_scale']}")
    print(f"command ranges ({ranges_source}):")
    for key, value in command_ranges.items():
        print(f"  {key}: {value}")
    print(f"joint order: {joint_order}")
    print()
    print("Next: scp this directory to the robot, then run the runtime with")
    print("  --policy policies/<name>/policy.onnx")

    env.close()


if __name__ == "__main__":
    main()
