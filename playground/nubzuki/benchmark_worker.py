"""One isolated, no-update Mac benchmark candidate."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time

import jax
import jax.numpy as jnp

from playground.nubzuki.standing import Standing, default_config


ROLLOUT_STEPS = 4
REPEATS = 3


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def run(num_envs: int) -> dict:
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"Mac benchmark requires CPU backend, got {jax.default_backend()}")
    env = Standing(config=default_config())
    keys = jax.random.split(jax.random.PRNGKey(7), num_envs)
    actions = jnp.zeros((num_envs, env.action_size), dtype=jnp.float32)

    @jax.jit
    def rollout(reset_keys, zero_actions):
        state = jax.vmap(env.reset)(reset_keys)
        def body(carry, _):
            carry = jax.vmap(env.step)(carry, zero_actions)
            return carry, None
        state, _ = jax.lax.scan(body, state, None, length=ROLLOUT_STEPS)
        return state.obs["state"]

    key = jax.random.PRNGKey(9)
    dims = (85, 512, 256, 128, 28)
    params = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        key, subkey = jax.random.split(key)
        params.append((jax.random.normal(subkey, (fan_in, fan_out)) / jnp.sqrt(fan_in), jnp.zeros(fan_out)))

    def ppo_actor_loss(weights, observations):
        x = observations
        for kernel, bias in weights[:-1]:
            x = jax.nn.swish(x @ kernel + bias)
        logits = x @ weights[-1][0] + weights[-1][1]
        mean, log_std = jnp.split(logits, 2, axis=-1)
        log_std = jnp.clip(log_std, -5.0, 2.0)
        log_prob = -0.5 * jnp.sum((mean / jnp.exp(log_std)) ** 2 + 2 * log_std, axis=-1)
        ratio = jnp.exp(log_prob - jax.lax.stop_gradient(log_prob))
        clipped = jnp.clip(ratio, 0.8, 1.2)
        return -jnp.mean(jnp.minimum(ratio, clipped))

    gradient = jax.jit(jax.value_and_grad(ppo_actor_loss))

    # First call includes JIT compilation, which on this model costs tens of
    # seconds and barely moves with the environment count. Timing it together
    # with the rollout reports compile time dressed up as throughput, so the
    # two are measured separately and only the steady-state rate is reported.
    started = time.perf_counter()
    observations = rollout(keys, actions)
    observations.block_until_ready()
    loss, grads = gradient(params, observations)
    jax.tree_util.tree_leaves(grads)[0].block_until_ready()
    compile_seconds = time.perf_counter() - started

    rates = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        observations = rollout(keys, actions)
        observations.block_until_ready()
        rates.append(num_envs * ROLLOUT_STEPS / (time.perf_counter() - started))
    steady_seconds = sum(num_envs * ROLLOUT_STEPS / rate for rate in rates)

    finite = bool(jnp.isfinite(observations).all() & jnp.isfinite(loss))
    return {
        "num_envs": num_envs,
        "backend": jax.default_backend(),
        "finite": finite,
        "compile_seconds": compile_seconds,
        "steady_state_seconds": steady_seconds,
        "elapsed_seconds": compile_seconds + steady_seconds,
        # Median of the repeats: one slow sample from OS scheduling should not
        # decide how long the user thinks training takes.
        "environment_steps_per_second": sorted(rates)[len(rates) // 2],
        "peak_rss_bytes": _rss_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.num_envs), sort_keys=True))


if __name__ == "__main__":
    main()

