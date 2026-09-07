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
