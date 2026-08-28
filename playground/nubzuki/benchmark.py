"""Measure this device and pick an environment count, without training."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
import sys


CPU_CANDIDATES = (256, 512, 1024, 2048)
# Capped at 8192: Brax splits batch_size * num_minibatches = 8192 transitions
# across the environments each training step, so a larger count cannot be used
# without changing the PPO shape away from upstream's.
ACCELERATOR_CANDIDATES = (1024, 2048, 4096, 8192)
RSS_LIMIT_BYTES = 16 * 1024**3
# Throughput saturates well before the largest workable count: on the Mac CPU
# 512 and 2048 environments are within a percent of each other. Past the knee a
# bigger count only costs memory and compile time, so the smallest count that
# is still this close to the best measured rate wins.
SATURATION_TOLERANCE = 0.05


def detect_backend() -> str:
    """Ask a throwaway process which backend JAX would choose here."""
    completed = subprocess.run(
        [sys.executable, "-c", "import jax; print(jax.default_backend())"],
        text=True, capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Could not determine the JAX backend:\n{completed.stderr[-2000:]}")
    return completed.stdout.strip().splitlines()[-1]


def select_candidate(passing: list[dict]) -> tuple[dict, dict]:
    """Return (chosen, fastest) from the candidates that completed."""
    fastest = max(passing, key=lambda item: item["environment_steps_per_second"])
    threshold = fastest["environment_steps_per_second"] * (1.0 - SATURATION_TOLERANCE)
    chosen = min(
        (item for item in passing if item["environment_steps_per_second"] >= threshold),
        key=lambda item: item["num_envs"],
    )
    return chosen, fastest


def benchmark_device(output: Path) -> dict:
    backend = detect_backend()
    candidates = CPU_CANDIDATES if backend == "cpu" else ACCELERATOR_CANDIDATES
    results = []
    passing = []
    for candidate in candidates:
        command = [
            sys.executable, "-m", "playground.nubzuki.benchmark_worker",
            "--num-envs", str(candidate),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            # Out of device memory lands here too, which is the point: the run
            # that cannot start is the one we must not select.
            results.append({"num_envs": candidate, "ok": False, "error": completed.stderr[-2000:]})
            break
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        within_memory = backend != "cpu" or payload["peak_rss_bytes"] <= RSS_LIMIT_BYTES
        payload["ok"] = bool(payload["finite"] and within_memory)
        results.append(payload)
        if not payload["ok"]:
            break
        passing.append(payload)

    if not passing:
        raise RuntimeError(
            f"No benchmark candidate completed on the {backend} backend. "
            "See the candidate errors in the output."
        )

    selected, fastest = select_candidate(passing)
    rate = float(selected["environment_steps_per_second"])
    profile = {
        "schema_version": 3,
        "backend": backend,
        "platform": f"{platform.system()}-{platform.machine()}",
        "num_envs": selected["num_envs"],
        # Steady state only: the compile pass is excluded, because it is paid
        # once per run and does not scale with the training length.
        "environment_steps_per_second": rate,
        "compile_seconds": selected.get("compile_seconds"),
        "peak_rss_bytes": selected["peak_rss_bytes"],
        "fastest_num_envs": fastest["num_envs"],
        "fastest_environment_steps_per_second": fastest["environment_steps_per_second"],
        "hours_per_10m_steps": 10_000_000 / rate / 3600 if rate > 0 else None,
        "candidates": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    note = ""
    if selected["num_envs"] != fastest["num_envs"]:
        note = (
            f" ({fastest['num_envs']} was fastest at "
            f"{fastest['environment_steps_per_second']:,.0f}/s, within "
            f"{SATURATION_TOLERANCE:.0%}, so the smaller count is used)"
        )
    print(
        f"\nBackend {backend}. Selected {profile['num_envs']} environments at "
        f"{rate:,.0f} env steps/s{note}; compile {profile['compile_seconds']:.0f}s once.\n"
        f"Rollout only, without the PPO update: 10M steps is about "
        f"{profile['hours_per_10m_steps']:.2f} h, so expect training to be slower "
        f"than that.\n"
    )
    return profile


# Kept so the older command name keeps working.
benchmark_mac = benchmark_device
