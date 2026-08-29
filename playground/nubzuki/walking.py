"""Forward/stop walking task built on the standing policy contract."""

from __future__ import annotations

import jax
import jax.numpy as jp
from ml_collections import config_dict

from playground.common.rewards import reward_tracking_lin_vel
from playground.nubzuki.standing import Standing, default_config as standing_config


def default_config() -> config_dict.ConfigDict:
    config = standing_config()
    config.forward_velocity_range_m_s = [0.03, 0.15]
    config.reward_config.scales.tracking_lin_vel = 3.0
    return config


class Walking(Standing):
    """One policy for joystick forward motion and zero-command standing."""

    def _get_reward(
        self, data, action, info, metrics, done, first_contact, contact
    ):
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact
        )
        rewards["tracking_lin_vel"] = reward_tracking_lin_vel(
            info["command"],
            self.get_local_linvel(data),
            self._config.reward_config.tracking_sigma,
        )

        return rewards

    def sample_command(self, rng: jax.Array) -> jax.Array:
        velocity_rng, zero_rng, head_rng = jax.random.split(rng, 3)
        standing_command = super().sample_command(head_rng)
        forward_velocity = jax.random.uniform(
            velocity_rng,
            minval=self._config.forward_velocity_range_m_s[0],
            maxval=self._config.forward_velocity_range_m_s[1],
        )
        command = standing_command.at[0].set(forward_velocity)
        return jp.where(
            jax.random.bernoulli(
                zero_rng, p=self._config.zero_command_probability
            ),
            jp.zeros(7),
            command,
        )
