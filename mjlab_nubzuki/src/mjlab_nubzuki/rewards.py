"""Reward terms specific to Nubzuki's narrow mechanical joint ranges."""

from __future__ import annotations

import torch


def commanded_joint_limit_violation_l2(env, action_name: str = "joint_pos"):
    """Penalize position targets outside the joints' physical hard limits.

    MJLab's standard ``dof_pos_limits`` term only sees the resulting joint
    position.  A policy can therefore command far through a hard stop without
    paying for it, because the constraint keeps the simulated joint in range.
    This term instead prices the impossible part of the command itself.
    """
    action = env.action_manager.get_term(action_name)
    entity = action._entity
    target = action.raw_action * action.scale + action.offset
    limits = entity.data.joint_pos_limits[:, action.target_ids]
    below = torch.clamp(limits[..., 0] - target, min=0.0)
    above = torch.clamp(target - limits[..., 1], min=0.0)
    return torch.sum(torch.square(below + above), dim=1)


def moving_knee_upper_margin_l2(
    env,
    sensor_name: str,
    asset_cfg,
    action_name: str = "joint_pos",
    command_name: str = "twist",
    command_threshold: float = 0.01,
    support_ceiling_rad: float = -0.05236,
    swing_ceiling_rad: float = -0.17453,
):
    """Keep commanded knees away from the extension stop while moving.

    Nubzuki's phase-free gait has no oscillator saying which leg should swing.
    Contact state supplies that distinction once a foot unloads.  Before
    liftoff both knees retain a small bend, preventing the policy from using
    the zero-degree hard stops as rigid virtual supports; once airborne, the
    swing knee is asked for enough flexion to create useful toe clearance.
    The term is disabled for zero command, so standing can retain its learned
    posture.
    """
    action = env.action_manager.get_term(action_name)
    target = action.raw_action * action.scale + action.offset
    # SceneEntityCfg IDs address the complete articulation.  In the backlash
    # task that includes 14 passive hinges, while raw_action contains only the
    # 14 driven joints, so translate entity IDs into action-column indices.
    driven_ids = list(action.target_ids)
    knee_action_ids = [driven_ids.index(joint_id) for joint_id in asset_cfg.joint_ids]
    knee_target = target[:, knee_action_ids]

    contact = env.scene[sensor_name].data.found
    if contact.ndim > 2:
        contact = torch.any(contact, dim=tuple(range(2, contact.ndim)))
    in_air = contact == 0
    ceiling = torch.where(
        in_air,
        torch.as_tensor(swing_ceiling_rad, device=target.device),
        torch.as_tensor(support_ceiling_rad, device=target.device),
    )
    violation = torch.clamp(knee_target - ceiling, min=0.0)

    command = env.command_manager.get_command(command_name)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    active = (linear_norm + angular_norm > command_threshold).float()
    return torch.sum(torch.square(violation), dim=1) * active
