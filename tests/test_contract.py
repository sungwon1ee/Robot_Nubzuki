import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration
from playground.nubzuki.controller import apply_deadzone, axes_to_head_targets
from playground.nubzuki.cli import build_parser
from playground.nubzuki.head_dynamics import HeadDynamicsProfile, HeadTrajectoryLimiter
from playground.nubzuki.policy import ObservationBuilder
from playground.nubzuki.ppo_config import training_config
from playground.nubzuki.standing import default_config


class StandingContractTests(unittest.TestCase):
    def test_v3_uses_physical_hip_roll_axes_and_stronger_pushes(self):
        env_config = default_config()
        self.assertEqual(list(env_config.push_config.interval_range), [4.0, 8.0])
        self.assertEqual(list(env_config.push_config.magnitude_range), [0.1, 1.25])

    def test_v3_policy_semantics_are_explicit(self):
        source = Path("playground/nubzuki/runner.py").read_text()
        self.assertIn('"model_semantics_version": 3', source)
        self.assertEqual(NubzukiCalibration().data["runtime"]["head_kp"], 24)

    def test_robot_phone_control_defaults_to_port_8766(self):
        args = build_parser().parse_args(
            ["robot", "--policy", "policy.onnx", "--control", "phone"]
        )
        self.assertEqual(args.control, "phone")
        self.assertEqual(args.web_port, 8766)

    def test_identify_head_supports_phone_control(self):
        args = build_parser().parse_args(
            ["identify-head", "--control", "phone"]
        )
        self.assertEqual(args.control, "phone")
        self.assertEqual(args.web_port, 8766)

    def setUp(self):
        self.calibration = NubzukiCalibration()
        self.profile_data = {
            "schema_version": 1,
            "calibration_sha256": self.calibration.sha256,
            "control_frequency_hz": 50,
            "joystick_deadzone": {
                "left_x": 0.1, "left_y": 0.1, "right_x": 0.1, "right_y": 0.1,
            },
            "joints": {
                name: {
                    "response_delay_s": 0.02,
                    "velocity_limit_rad_s": 1.0,
                    "acceleration_limit_rad_s2": 2.0,
                }
                for name in HEAD_JOINTS
            },
        }
        self.profile = HeadDynamicsProfile(self.profile_data, self.calibration)

    def test_calibration_abi(self):
        self.assertEqual(self.calibration.observation_size, 85)
        self.assertEqual(self.calibration.privileged_observation_size, 153)
        self.assertEqual(self.calibration.action_size, 14)
        self.assertEqual(self.calibration.control_frequency_hz, 50)

    def test_servo_round_trip(self):
        for name in self.calibration.joint_order:
            for value in (-0.1, 0.0, 0.1):
                recovered = self.calibration.servo_to_logical(
                    name, self.calibration.logical_to_servo(name, value)
                )
                self.assertAlmostEqual(recovered, value)

    def test_observation_layout_is_85(self):
        builder = ObservationBuilder()
        obs = builder.build(
            np.zeros(3), np.zeros(3), np.zeros(7), np.zeros(14),
            np.zeros(14), np.zeros(2),
        )
        self.assertEqual(obs.shape, (85,))
        builder.advance(np.ones(14))
        shifted = builder.build(
            np.zeros(3), np.zeros(3), np.zeros(7), np.zeros(14),
            np.zeros(14), np.zeros(2),
        )
        np.testing.assert_allclose(shifted[41:55], 1.0)

    def test_absolute_joystick_mapping(self):
        axes = {"left_x": 1.0, "left_y": -1.0, "right_x": 1.0, "right_y": 0.0}
        targets = axes_to_head_targets(axes, self.calibration, self.profile)
        self.assertAlmostEqual(targets["head_yaw"], self.calibration.limits_rad("head_yaw")[1])
        self.assertAlmostEqual(targets["head_pitch"], self.calibration.limits_rad("head_pitch")[1])
        self.assertEqual(targets["neck_pitch"], 0.0)
        self.assertEqual(apply_deadzone(0.05, 0.1), 0.0)

    def test_trajectory_respects_measured_caps(self):
        limiter = HeadTrajectoryLimiter(self.profile, dt=0.02)
        before = limiter.position.copy()
        after = limiter.step({name: 1.0 for name in HEAD_JOINTS})
        for name in HEAD_JOINTS:
            self.assertLessEqual(abs(limiter.velocity[name]), 1.0)
            self.assertLessEqual(abs(after[name] - before[name]), 1.0 * 0.02)

    def test_mac_profile_controls_num_envs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps({"backend": "cpu", "num_envs": 512}))
            config = training_config("macbook", 150_000_000, path)
            self.assertEqual(config["num_envs"], 512)
            self.assertEqual(config["num_timesteps"], 150_000_000)


if __name__ == "__main__":
    unittest.main()
