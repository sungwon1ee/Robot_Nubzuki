"""Command line interface for training, simulation and hardware."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nubzuki-standing")
    parser.add_argument("--calibration", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="validate model and policy ABI without training")
    benchmark = sub.add_parser("benchmark-mac", help="select a safe Mac CPU env count")
    benchmark.add_argument("--output", default=".local/mac_profile.json")
    train = sub.add_parser("train", help="start PPO only when explicitly invoked")
    train.add_argument("--preset", choices=("smoke", "macbook", "official"), default="macbook")
    train.add_argument("--num-timesteps", type=int, default=None)
    train.add_argument("--output", default="runs/standing")
    train.add_argument("--mac-profile", default=".local/mac_profile.json")
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
    identify.add_argument("--dry-run", action="store_true")
    identify.add_argument("--yes", action="store_true")
    robot = sub.add_parser("robot", help="run the policy on physical Nubzuki")
    robot.add_argument("--policy", required=True)
    robot.add_argument("--port", default="/dev/ttyACM0")
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
    elif args.command == "benchmark-mac":
        from playground.nubzuki.benchmark import benchmark_mac
        print(json.dumps(benchmark_mac(Path(args.output)), indent=2))
    elif args.command == "train":
        from playground.nubzuki.runner import NubzukiStandingRunner
        args.num_timesteps = args.num_timesteps or (1024 if args.preset == "smoke" else 150_000_000)
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
                args.port, Path(args.output), NubzukiCalibration(args.calibration), args.yes
            )
            print(json.dumps(profile, indent=2))
    elif args.command == "robot":
        from playground.nubzuki.robot_runtime import run_robot
        run_robot(args.policy, args.port, args.calibration, args.head_profile, args.imu_upside_down)


if __name__ == "__main__":
    main()

