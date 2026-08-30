import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import jax

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration
from playground.nubzuki.controller import (
    apply_deadzone,
    axes_to_head_targets,
    forward_velocity_command,
)
from playground.nubzuki.cli import build_parser
from playground.nubzuki.head_dynamics import HeadDynamicsProfile, HeadTrajectoryLimiter
from playground.nubzuki.policy import ObservationBuilder
from playground.nubzuki.ppo_config import training_config
from playground.nubzuki.robot_runtime import PARK_SPEED_FRACTION, park
from playground.nubzuki.standing import _cost_negative_hip_roll, default_config
from playground.nubzuki.walking import Walking, default_config as walking_config


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
    def test_colab_notebook_trains_and_logs_the_walking_branch(self):
        notebook = json.loads(
            Path("notebooks/nubzuki_standing_colab.ipynb").read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("BRANCH = 'walking'", source)
        self.assertIn("ENV = 'walking'", source)
        self.assertIn("runs/walking_microduck", source)
        self.assertIn("--env {ENV}", source)

    def test_walking_keeps_abi_and_samples_only_forward_or_stop(self):
        config = walking_config()
        self.assertEqual(config.target_swing_height_m, 0.02)
        self.assertEqual(config.reward_config.scales.tracking_lin_vel, 2.0)
        self.assertEqual(config.reward_config.scales.upright, 2.0)
        self.assertEqual(config.reward_config.scales.pose, 1.0)
        self.assertEqual(config.reward_config.scales.feet_height, -0.25)
        self.assertEqual(config.reward_config.scales.feet_air_time, 3.0)
        self.assertEqual(config.reward_config.scales.foot_clearance, -2.0)
        self.assertEqual(config.reward_config.scales.foot_slip, -0.1)
        self.assertEqual(config.reward_config.scales.action_rate, -0.1)
        self.assertEqual(config.reward_config.scales.alive, 0.0)
        self.assertEqual(config.reward_config.scales.head_pose_tracking, 2.0)
        env = Walking(config=config)
        state = env.reset(np.array([0, 1], dtype=np.uint32))
        self.assertEqual(state.obs["state"].shape, (85,))
        commands = np.asarray(
            jax.vmap(env.sample_command)(jax.random.split(jax.random.PRNGKey(7), 256))
        )
        self.assertTrue(np.all(commands[:, 0] >= 0.0))
        np.testing.assert_allclose(commands[:, 1:3], 0.0)
        self.assertGreater(np.mean(commands[:, 0] == 0.0), 0.1)

    def test_walk_mode_maps_only_forward_stick(self):
        metadata = {
            "policy": "walking",
            "forward_velocity_range_m_s": [0.03, 0.15],
        }
        self.assertAlmostEqual(
            forward_velocity_command({"left_y": 1.0}, "walk", metadata),
            0.15,
        )
        self.assertEqual(
            forward_velocity_command({"left_y": -1.0}, "walk", metadata),
            0.0,
        )
        self.assertEqual(
            forward_velocity_command({"left_y": 1.0}, "head", metadata),
            0.0,
        )

    def test_zero_command_and_head_tracking_are_prioritized(self):
        config = default_config()
        self.assertEqual(config.zero_command_probability, 0.20)
        self.assertEqual(config.reward_config.scales.head_pos, -5.0)
        self.assertEqual(config.reward_config.scales.negative_hip_roll, -2.0)

    def test_only_negative_hip_roll_is_penalized(self):
        outward = np.zeros(14)
        outward[[1, 10]] = 0.2
        self.assertEqual(float(_cost_negative_hip_roll(outward)), 0.0)

        inward = np.zeros(14)
        inward[[1, 10]] = [-0.2, -0.1]
        self.assertAlmostEqual(
            float(_cost_negative_hip_roll(inward)), 0.05, places=6
        )

    def test_simple_home_pose_has_no_self_collision(self):
        model_path = Path(
            "playground/nubzuki/xmls/scene_flat_terrain.xml"
        ).resolve()
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
        mujoco.mj_forward(model, data)

        contacts = {
            frozenset(
                (
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
                    ),
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
                    ),
                )
            )
            for contact in data.contact
        }
        self.assertNotIn(
            frozenset(("head_collision", "trunk_collision")), contacts
        )
        self.assertNotIn(
            frozenset(("head_collision", "trunk_rear_collision")), contacts
        )
        self.assertEqual(data.ncon, 0)

    def test_simple_collision_shell_tracks_the_cad_shape(self):
        model_path = Path(
            "playground/nubzuki/xmls/scene_flat_terrain.xml"
        ).resolve()
        model = mujoco.MjModel.from_xml_path(str(model_path))
        trunk_id = model.geom("trunk_collision").id
        battery_id = model.geom("trunk_rear_collision").id
        self.assertAlmostEqual(float(model.geom_size[trunk_id, 2]), 0.075)
        np.testing.assert_allclose(
            model.geom_size[battery_id],
            [0.02727, 0.044755, 0.057815],
            atol=1e-12,
        )
        head_id = model.geom("head_collision").id
        self.assertEqual(
            int(model.geom_type[head_id]), int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        )
        self.assertAlmostEqual(float(model.geom_size[head_id, 0]), 0.105)
        self.assertAlmostEqual(float(model.geom_size[head_id, 1]), 0.105)

    def test_left_and_right_knees_and_feet_can_self_collide(self):
        model_path = Path(
            "playground/nubzuki/xmls/scene_flat_terrain.xml"
        ).resolve()
        model = mujoco.MjModel.from_xml_path(str(model_path))

        def compatible(first: str, second: str) -> bool:
            first_id = model.geom(first).id
            second_id = model.geom(second).id
            return bool(
                int(model.geom_contype[first_id])
                & int(model.geom_conaffinity[second_id])
                or int(model.geom_contype[second_id])
                & int(model.geom_conaffinity[first_id])
            )

        self.assertTrue(
            compatible("left_knee_collision", "right_knee_collision")
        )
        self.assertTrue(
            compatible("left_foot_collision", "right_foot_collision")
        )
        self.assertTrue(
            compatible("left_knee_collision", "right_foot_collision")
        )
        self.assertTrue(
            compatible("right_knee_collision", "left_foot_collision")
        )

    def test_v4_uses_open_duck_horizontal_velocity_pushes(self):
        env_config = default_config()
        self.assertEqual(list(env_config.push_config.interval_range), [5.0, 10.0])
        self.assertEqual(
            list(env_config.push_config.magnitude_range_m_s), [0.1, 1.0]
        )

    def test_v4_policy_semantics_are_explicit(self):
        source = Path("playground/nubzuki/runner.py").read_text()
        self.assertIn('"model_semantics_version": 6', source)
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
