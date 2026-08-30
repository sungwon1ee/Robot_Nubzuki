"""Command line interface for training, simulation and hardware."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

if platform.system() == "Darwin":
    # Apple GPUs have no supported JAX path and jax-metal is experimental, so
    # macOS is pinned to CPU. Elsewhere JAX picks its own backend, which is how
    # a CUDA machine gets used. JAX_PLATFORMS in the environment still wins.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nubzuki-standing")
    parser.add_argument("--calibration", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="validate model and policy ABI without training")
    for name, help_text in (
        ("benchmark", "measure this device and select a safe environment count"),
        ("benchmark-mac", argparse.SUPPRESS),
    ):
        benchmark = sub.add_parser(name, help=help_text)
        benchmark.add_argument("--output", default=".local/mac_profile.json")
    train = sub.add_parser("train", help="start PPO only when explicitly invoked")
    train.add_argument(
        "--env", choices=("standing", "walking"), default="standing",
        help="train standing or the forward/stop walking task",
    )
    train.add_argument(
        "--walk-stage",
        choices=(
            "discovery", "refine", "control", "turning",
            "microduck_0", "microduck_1", "microduck_2",
            "microduck_3", "microduck_4", "microduck_5",
            "microduck_auto",
        ),
        default="discovery",
        help="walking curriculum stage; ignored by the standing environment",
    )
    train.add_argument(
        "--preset", choices=("smoke", "profile", "macbook", "official"), default="profile",
        help="profile uses the benchmark result; macbook is an alias for it",
    )
    train.add_argument(
        "--num-envs", type=int, default=None,
        help="override the preset's environment count",
    )
    train.add_argument("--num-timesteps", type=int, default=None)
    train.add_argument("--output", default=None)
    train.add_argument(
        "--device-profile", "--mac-profile", dest="mac_profile",
        default=".local/mac_profile.json",
        help="benchmark result to take the environment count from",
    )
    train.add_argument(
        "--restore",
        default="auto",
        help="'auto' continues the latest checkpoint in --output, or pass a params directory",
    )
    train.add_argument(
        "--fresh", action="store_true", help="ignore any existing checkpoint in --output"
    )
    train.add_argument(
        "--step-offset", type=int, default=None,
        help="override the resumed absolute step (only if it cannot be inferred)",
    )
    train.add_argument(
        "--checkpoint-every", type=int, default=5_000_000,
        help="environment steps between checkpoints; this is what an interrupted run can lose",
    )
    train.add_argument(
        "--num-eval-envs", type=int, default=None,
        help="environments per evaluation rollout (Brax default 128); lower it when checkpointing often",
    )
    sim = sub.add_parser("sim", help="run an ONNX policy in native MuJoCo")
    sim.add_argument("--policy", required=True)
    sim.add_argument("--head-profile", default="config/head_dynamics.json")
    sim.add_argument(
        "--control", choices=("phone", "joystick"), default="phone",
        help="phone serves a touch page over HTTP; joystick uses a connected gamepad",
    )
    sim.add_argument("--host", default="0.0.0.0", help="phone control bind address")
    sim.add_argument("--port", type=int, default=8765, help="phone control port")
    identify = sub.add_parser("identify-head", help="measure joystick and head response")
    identify.add_argument("--port", default="/dev/ttyACM0")
    identify.add_argument("--output", default="config/head_dynamics.json")
    identify.add_argument(
        "--control", choices=("phone", "joystick"), default="joystick",
        help="phone serves the identification controller over HTTP; joystick uses Xbox",
    )
    identify.add_argument("--host", default="0.0.0.0", help="phone control bind address")
    identify.add_argument("--web-port", type=int, default=8766, help="phone control HTTP port")
    identify.add_argument("--dry-run", action="store_true")
    identify.add_argument("--yes", action="store_true")
    robot = sub.add_parser("robot", help="run the policy on physical Nubzuki")
    robot.add_argument("--policy", required=True)
    robot.add_argument("--port", default="/dev/ttyACM0")
    robot.add_argument(
        "--control", choices=("phone", "joystick"), default="joystick",
        help="phone serves the standing controller over HTTP; joystick uses Xbox",
    )
    robot.add_argument("--host", default="0.0.0.0", help="phone control bind address")
    robot.add_argument("--web-port", type=int, default=8766, help="phone control HTTP port")
    robot.add_argument("--head-profile", default="config/head_dynamics.json")
    robot.add_argument("--imu-upside-down", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.calibration:
        os.environ["NUBZUKI_CALIBRATION"] = str(Path(args.calibration).expanduser().resolve())
    if args.command == "validate":
        from playground.nubzuki.validate_setup import validate
        print(json.dumps(validate(args.calibration), indent=2))
    elif args.command in ("benchmark", "benchmark-mac"):
        from playground.nubzuki.benchmark import benchmark_device
        print(json.dumps(benchmark_device(Path(args.output)), indent=2))
    elif args.command == "train":
        from playground.nubzuki.runner import NubzukiStandingRunner
        args.num_timesteps = args.num_timesteps or (1024 if args.preset == "smoke" else 150_000_000)
        args.output = args.output or f"runs/{args.env}"
        if args.fresh:
            args.restore = None
        NubzukiStandingRunner(args).train()
    elif args.command == "sim":
        from playground.nubzuki.sim_runtime import run_simulation
        run_simulation(
            args.policy, args.calibration, args.head_profile,
            args.control, args.host, args.port,
        )
    elif args.command == "identify-head":
        from playground.nubzuki.calibration import NubzukiCalibration
        from playground.nubzuki.identify import identify_head, procedure
        if args.dry_run:
            print(procedure())
        else:
            profile = identify_head(
                args.port, Path(args.output), NubzukiCalibration(args.calibration), args.yes,
                args.control, args.host, args.web_port,
            )
            print(json.dumps(profile, indent=2))
    elif args.command == "robot":
        from playground.nubzuki.robot_runtime import run_robot
        run_robot(
            args.policy, args.port, args.calibration, args.head_profile,
            args.imu_upside_down, args.control, args.host, args.web_port,
        )


if __name__ == "__main__":
    main()
