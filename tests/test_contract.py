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
from playground.nubzuki.robot_runtime import PARK_SPEED_FRACTION, park
from playground.nubzuki.standing import default_config


class RecordingHardware:
    """Stands in for ServoHardware so the park path can be exercised offline."""

    def __init__(self):
        self.writes = []
        self.torque_disabled = False

    def set_positions(self, positions):
        self.writes.append(dict(positions))

    def disable_torque(self, names=None):
        self.torque_disabled = True


class StandingContractTests(unittest.TestCase):
    def test_v3_uses_physical_hip_roll_axes_and_random_force_pushes(self):
        env_config = default_config()
        self.assertEqual(list(env_config.push_config.interval_range), [4.0, 8.0])
        self.assertEqual(
            list(env_config.push_config.torso_force_range_n), [3.0, 10.0]
        )
        self.assertEqual(
            list(env_config.push_config.head_force_range_n), [1.0, 3.0]
        )
        self.assertEqual(
            list(env_config.push_config.duration_range_s), [0.08, 0.20]
        )

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


class SafeStopTests(unittest.TestCase):
    """A dropped controller must never become a fall."""

    def setUp(self):
        self.calibration = NubzukiCalibration()

    def test_stale_controller_does_not_abort_the_loop(self):
        source = Path("playground/nubzuki/robot_runtime.py").read_text()
        self.assertNotIn("Joystick data is stale", source)
        # The only guard allowed to raise is the unmeasured-profile check,
        # which runs before a single servo is energised. Once the loop is
        # running, nothing may throw its way out to a torque cut.
        loop = source.split("while True:", 1)[1]
        self.assertNotIn("raise", loop)

    def test_nothing_the_armed_loop_can_do_cuts_torque(self):
        source = Path("playground/nubzuki/robot_runtime.py").read_text()
        # Once armed there is no path back to a torque cut: not a lost
        # controller, not an exception, not the B button. Both remaining
        # calls sit in the pre-arm state the loop started in.
        self.assertEqual(source.count("disable_torque()"), 2)
        self.assertIn("if not armed:", source)
        self.assertNotIn("emergency", source.lower())
        after_arming = source.split("armed = True", 1)[1]
        self.assertNotIn("disable_torque()", after_arming.split("if not armed:")[0])

    def test_park_lands_on_the_calibrated_pose_without_cutting_torque(self):
        hardware = RecordingHardware()
        start = np.zeros(14)
        park(hardware, self.calibration, start, dt=0.02)
        self.assertFalse(hardware.torque_disabled)
        self.assertTrue(hardware.writes)
        for name in self.calibration.joint_order:
            self.assertAlmostEqual(
                hardware.writes[-1][name], self.calibration.park_rad(name)
            )

    def test_park_respects_the_servo_velocity_limit(self):
        hardware = RecordingHardware()
        start = np.zeros(14)
        park(hardware, self.calibration, start, dt=0.02)
        budget = (
            float(self.calibration.data["runtime"]["max_motor_velocity_rad_s"])
            * 0.02
            * PARK_SPEED_FRACTION
        )
        previous = {name: 0.0 for name in self.calibration.joint_order}
        for write in hardware.writes:
            for name, value in write.items():
                self.assertLessEqual(abs(value - previous[name]), budget + 1e-9)
                previous[name] = value

    def test_every_park_pose_is_inside_its_joint_limits(self):
        for name in self.calibration.joint_order:
            low, high = self.calibration.limits_rad(name)
            self.assertGreaterEqual(self.calibration.park_rad(name), low)
            self.assertLessEqual(self.calibration.park_rad(name), high)


if __name__ == "__main__":
    unittest.main()
