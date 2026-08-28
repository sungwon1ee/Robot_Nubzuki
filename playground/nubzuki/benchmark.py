"""Select a safe Mac CPU environment count without applying PPO updates."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import subprocess
import sys


CANDIDATES = (256, 512, 1024, 2048)
RSS_LIMIT_BYTES = 16 * 1024**3


def benchmark_mac(output: Path) -> dict:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("benchmark-mac requires an Apple Silicon Mac")
    results = []
    selected = None
    for candidate in CANDIDATES:
        command = [
            sys.executable, "-m", "playground.nubzuki.benchmark_worker",
            "--num-envs", str(candidate),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            results.append({"num_envs": candidate, "ok": False, "error": completed.stderr[-2000:]})
            continue
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        payload["ok"] = bool(payload["finite"] and payload["peak_rss_bytes"] <= RSS_LIMIT_BYTES)
        results.append(payload)
        if payload["ok"]:
            selected = payload
        else:
            break
    if selected is None:
        raise RuntimeError("No Mac benchmark candidate stayed finite below 16 GiB RSS")
    rate = float(selected["environment_steps_per_second"])
    profile = {
        "schema_version": 2,
        "backend": "cpu",
        "num_envs": selected["num_envs"],
        # Steady state only: the compile pass is excluded, because it is paid
        # once per run and does not scale with the training length.
        "environment_steps_per_second": rate,
        "compile_seconds": selected.get("compile_seconds"),
        "peak_rss_bytes": selected["peak_rss_bytes"],
        "hours_per_10m_steps": 10_000_000 / rate / 3600 if rate > 0 else None,
        "candidates": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"\nSelected {profile['num_envs']} environments at {rate:,.0f} env steps/s "
        f"(compile {profile['compile_seconds']:.0f}s once).\n"
        f"Rollout only, without the PPO update: 10M steps is about "
        f"{profile['hours_per_10m_steps']:.1f} h, so expect training to be slower "
        f"than that.\n"
    )
    return profile

