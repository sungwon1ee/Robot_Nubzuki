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
    head_axes_for_mode,
    yaw_rate_command,
)
from playground.nubzuki.cli import build_parser
from playground.nubzuki.head_dynamics import HeadDynamicsProfile, HeadTrajectoryLimiter
from playground.nubzuki.policy import ObservationBuilder
from playground.nubzuki.ppo_config import training_config
from playground.nubzuki.robot_runtime import PARK_SPEED_FRACTION, park
from playground.nubzuki.standing import _cost_negative_hip_roll, default_config
from playground.nubzuki.walking import (
    Walking,
    default_config as walking_config,
)


class RecordingHardware:
    """Stands in for ServoHardware so the park path can be exercised offline."""

    def __init__(self):
        self.writes = []
        self.torque_disabled = False
        self.torque_enabled = False

    def set_positions(self, positions):
        self.writes.append(dict(positions))

    def disable_torque(self, names=None):
        self.torque_disabled = True

    def enable_torque(self, names=None):
        self.torque_enabled = True


class StandingContractTests(unittest.TestCase):
    def test_walking_keeps_abi_and_samples_only_forward_or_stop(self):
        config = walking_config()
        self.assertEqual(config.target_swing_height_m, 0.02)
        self.assertEqual(config.walking_stage, "discovery")
        self.assertEqual(config.forward_velocity_range_m_s, [0.15, 0.15])
        self.assertEqual(config.reward_config.scales.walking_task, 5.0)
        self.assertEqual(config.reward_config.scales.standing_task, 2.0)
        self.assertEqual(config.reward_config.scales.pose, 0.0)
        self.assertEqual(config.reward_config.scales.feet_height, 0.0)
        self.assertEqual(config.reward_config.scales.feet_air_time, 2.0)
        self.assertEqual(config.reward_config.scales.foot_clearance, 0.0)
        self.assertEqual(config.reward_config.scales.foot_slip, -0.02)
        self.assertEqual(config.reward_config.scales.action_rate, -0.01)
        self.assertEqual(config.reward_config.scales.yaw_rate, 0.0)
        self.assertEqual(config.reward_config.scales.alive, 0.0)
        self.assertEqual(config.reward_config.scales.head_pose_tracking, 0.0)
        self.assertEqual(config.reward_config.scales.head_action_rate, 0.0)
        env = Walking(config=config)
        state = env.reset(np.array([0, 1], dtype=np.uint32))
        self.assertEqual(state.obs["state"].shape, (85,))
        commands = np.asarray(
            jax.vmap(env.sample_command)(jax.random.split(jax.random.PRNGKey(7), 256))
        )
        self.assertTrue(np.all(commands[:, 0] >= 0.0))
        np.testing.assert_allclose(commands[:, 1:3], 0.0)
        self.assertGreater(np.mean(commands[:, 0] == 0.0), 0.02)

    def test_walking_curriculum_progressively_adds_control_and_regularization(self):
        discovery = walking_config("discovery")
        refine = walking_config("refine")
        control = walking_config("control")
        self.assertLess(discovery.reward_config.scales.action_rate, 0.0)
        self.assertGreater(
            abs(control.reward_config.scales.action_rate),
            abs(discovery.reward_config.scales.action_rate),
        )
        self.assertEqual(discovery.reward_config.scales.foot_clearance, 0.0)
        self.assertLess(refine.reward_config.scales.foot_clearance, 0.0)
        self.assertEqual(refine.reward_config.scales.yaw_rate, -0.05)
        self.assertEqual(control.reward_config.scales.yaw_rate, -0.1)
        self.assertEqual(refine.reward_config.scales.head_action_rate, -0.01)
        self.assertEqual(control.reward_config.scales.head_action_rate, -0.02)
        self.assertEqual(control.reward_config.scales.head_joint_vel, -0.01)
        self.assertEqual(control.reward_config.scales.head_roll_home, -5.0)
        self.assertEqual(control.reward_config.scales.head_roll_vel, -0.1)
        self.assertTrue(control.enable_head_command)
        self.assertEqual(control.head_mode_probability, 0.25)
        self.assertEqual(control.head_zero_probability, 0.25)
        self.assertGreater(control.reward_config.scales.head_pose_tracking, 0.0)
        self.assertEqual(control.zero_command_probability, 0.25 * 0.25)
        control_env = Walking(config=control)
        commands = np.asarray(
            jax.vmap(control_env.sample_command)(
                jax.random.split(jax.random.PRNGKey(19), 4096)
            )
        )
        moving = np.linalg.norm(commands[:, :3], axis=1) > 0.01
        head_mode = ~moving
        head_at_home = np.all(commands[:, 3:] == 0.0, axis=1)
        self.assertAlmostEqual(np.mean(head_mode), 0.25, delta=0.03)
        self.assertAlmostEqual(np.mean(head_at_home[head_mode]), 0.25, delta=0.04)
        np.testing.assert_allclose(commands[moving, 6], 0.0)

        turning = walking_config("turning")
        self.assertEqual(turning.yaw_rate_range_rad_s, [-0.3, 0.3])
        self.assertEqual(turning.yaw_tracking_sigma, 0.04)
        self.assertEqual(turning.straight_command_probability, 0.50)
        self.assertEqual(turning.turn_in_place_probability, 0.0)
        self.assertEqual(turning.reward_config.scales.yaw_rate, 0.0)
        self.assertEqual(turning.reward_config.scales.yaw_tracking, 5.0)
        self.assertEqual(turning.reward_config.scales.straight_yaw_rate, -0.1)
        self.assertEqual(turning.reward_config.scales.head_action_rate, -0.03)
        self.assertEqual(turning.reward_config.scales.head_joint_vel, -0.02)
        self.assertEqual(turning.reward_config.scales.head_roll_home, -10.0)
        self.assertEqual(turning.reward_config.scales.head_roll_vel, -0.2)
        self.assertEqual(turning.head_zero_probability, 0.25)
        self.assertEqual(turning.head_mode_probability, 0.30)
        turning_env = Walking(config=turning)
        turning_commands = np.asarray(
            jax.vmap(turning_env.sample_command)(
                jax.random.split(jax.random.PRNGKey(23), 4096)
            )
        )
        turn_in_place = (
            (turning_commands[:, 0] == 0.0)
            & (np.abs(turning_commands[:, 2]) > 0.01)
        )
        self.assertFalse(np.any(turn_in_place))
        self.assertTrue(np.any(turning_commands[:, 2] > 0.01))
        self.assertTrue(np.any(turning_commands[:, 2] < -0.01))

    def test_turning_maps_left_stick_to_yaw_rate(self):
        metadata = {
            "policy": "walking",
            "yaw_rate_range_rad_s": [-0.3, 0.3],
        }
        self.assertAlmostEqual(
            yaw_rate_command({"left_x": 1.0}, "walk", metadata), 0.3
        )
        self.assertAlmostEqual(
            yaw_rate_command({"left_x": -1.0}, "walk", metadata), -0.3
        )
        self.assertEqual(
            yaw_rate_command({"left_x": 1.0}, "head", metadata), 0.0
        )

    def test_walk_mode_routes_right_stick_to_head_yaw_and_pitch_only(self):
        axes = {
            "left_x": 0.8,
            "left_y": 0.7,
            "right_x": -0.4,
            "right_y": 0.3,
        }
        routed = head_axes_for_mode(axes, "walk")
        self.assertEqual(routed["left_x"], -0.4)
        self.assertEqual(routed["left_y"], 0.3)
        self.assertEqual(routed["right_x"], 0.0)
        self.assertEqual(routed["right_y"], 0.0)
        self.assertEqual(head_axes_for_mode(axes, "head"), axes)

    def test_locomotion_trains_forward_curves_without_head_commands(self):
        config = walking_config("locomotion")
        self.assertEqual(config.forward_velocity_range_m_s, [0.04, 0.18])
        self.assertEqual(config.yaw_rate_range_rad_s, [-0.7, 0.7])
        self.assertEqual(config.min_turn_yaw_rate_rad_s, 0.15)
        self.assertEqual(config.straight_command_probability, 0.20)
        self.assertEqual(config.head_mode_probability, 0.0)
        self.assertFalse(config.enable_head_command)
        self.assertEqual(config.turn_in_place_probability, 0.0)
        self.assertEqual(config.zero_command_probability, 0.20)
        self.assertEqual(config.reward_config.scales.action_rate, -0.3)
        self.assertEqual(config.reward_config.scales.head_pose_tracking, 0.0)
        self.assertEqual(config.reward_config.scales.head_roll_home, 0.0)
        self.assertEqual(config.reward_config.scales.head_roll_vel, 0.0)

        env = Walking(config=config)
        commands = np.asarray(
            jax.vmap(env.sample_command)(
                jax.random.split(jax.random.PRNGKey(29), 8192)
            )
        )
        self.assertTrue(np.any(commands[:, 0] > 0.01))
        self.assertFalse(np.any(commands[:, 0] < 0.0))
        self.assertTrue(np.any(commands[:, 2] > 0.05))
        self.assertTrue(np.any(commands[:, 2] < -0.05))
        turning = np.abs(commands[:, 2]) > 1.0e-6
        self.assertTrue(np.all(np.abs(commands[turning, 2]) >= 0.15))
        self.assertFalse(np.any(np.abs(commands[:, 3:]) > 1.0e-4))
        stopped = np.linalg.norm(commands[:, :3], axis=1) < 1.0e-6
        self.assertAlmostEqual(np.mean(stopped), 0.20, delta=0.03)
        turn_in_place = (
            (commands[:, 0] == 0.0) & (np.abs(commands[:, 2]) > 0.05)
        )
        self.assertFalse(np.any(turn_in_place))

    def test_sim2real_curriculum_preserves_locomotion_and_progresses_delay(self):
        locomotion = walking_config("locomotion")
        expected = (
            ("sim2real_1", 4, 7, 0.60, 0.20),
            ("sim2real_2", 6, 9, 0.40, 0.40),
            ("sim2real_3", 8, 11, 0.20, 0.60),
        )
        for index, (stage, delay_min, delay_max, straight_ratio, turn_ratio) in enumerate(expected):
            config = walking_config(stage)
            self.assertEqual(config.forward_velocity_range_m_s,
                             locomotion.forward_velocity_range_m_s)
            self.assertEqual(config.yaw_rate_range_rad_s,
                             locomotion.yaw_rate_range_rad_s)
            self.assertEqual(config.reward_config.scales,
                             locomotion.reward_config.scales)
            self.assertEqual(config.noise_config.action_min_delay, delay_min)
            self.assertEqual(config.noise_config.action_max_delay, delay_max)

            env = Walking(config=config)
            commands = np.asarray(jax.vmap(env.sample_command)(
                jax.random.split(jax.random.PRNGKey(80 + index), 32768)
            ))
            stopped = np.linalg.norm(commands[:, :3], axis=1) < 1.0e-6
            straight = (commands[:, 0] > 0.01) & (np.abs(commands[:, 2]) < 1.0e-6)
            turning = (commands[:, 0] > 0.01) & (np.abs(commands[:, 2]) > 0.01)
            self.assertAlmostEqual(np.mean(stopped), 0.20, delta=0.02)
            self.assertAlmostEqual(np.mean(straight), straight_ratio, delta=0.02)
            self.assertAlmostEqual(np.mean(turning), turn_ratio, delta=0.02)

        alias = walking_config("sim2real")
        final = walking_config("sim2real_3")
        self.assertEqual(alias.noise_config.action_min_delay,
                         final.noise_config.action_min_delay)
        self.assertEqual(alias.noise_config.action_max_delay,
                         final.noise_config.action_max_delay)
        self.assertEqual(alias.straight_command_probability,
                         final.straight_command_probability)

    def test_head_position_stages_overlay_only_yaw_and_pitch(self):
        expected = (
            ("head_position_1", 0.20, 0.20, 0.50),
            ("head_position_2", 0.40, 0.40, 0.75),
            ("head_position_3", 0.60, 0.60, 1.00),
        )
        for index, (stage, probability, factor, weight) in enumerate(expected):
            config = walking_config(stage)
            self.assertTrue(config.enable_head_command)
            self.assertEqual(config.head_mode_probability, 0.0)
            self.assertEqual(config.simultaneous_head_probability, probability)
            self.assertEqual(config.head_range_factor, factor)
            self.assertTrue(config.head_yaw_pitch_only)
            self.assertEqual(config.reward_config.scales.head_pose_tracking, weight)
            self.assertEqual(config.reward_config.scales.action_rate, -0.3)

            env = Walking(config=config)
            commands = np.asarray(
                jax.vmap(env.sample_command)(
                    jax.random.split(jax.random.PRNGKey(50 + index), 8192)
                )
            )
            self.assertTrue(np.allclose(commands[:, 3], 0.0))
            self.assertTrue(np.allclose(commands[:, 6], 0.0))
            head_active = np.linalg.norm(commands[:, 4:6], axis=1) > 1.0e-6
            self.assertAlmostEqual(np.mean(head_active), probability, delta=0.03)
            stopped = np.linalg.norm(commands[:, :3], axis=1) < 1.0e-6
            self.assertAlmostEqual(np.mean(stopped), 0.20, delta=0.03)

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
        signed_metadata = {
            "policy": "walking",
            "forward_velocity_range_m_s": [-0.18, 0.18],
        }
        self.assertAlmostEqual(
            forward_velocity_command({"left_y": -1.0}, "walk", signed_metadata),
            -0.18,
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

    def test_robot_accepts_debug_csv_path(self):
        args = build_parser().parse_args([
            "robot", "--policy", "policy.onnx", "--debug-log", "logs/test.csv",
        ])
        self.assertEqual(args.debug_log, "logs/test.csv")

    def test_sim_accepts_floor_friction_override(self):
        args = build_parser().parse_args(
            ["sim", "--policy", "policy.onnx", "--floor-friction", "0.2"]
        )
        self.assertEqual(args.floor_friction, 0.2)

    def test_sim_accepts_action_delay_override(self):
        args = build_parser().parse_args(
            ["sim", "--policy", "policy.onnx", "--action-delay", "0.2"]
        )
        self.assertEqual(args.action_delay, 0.2)

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
        # Once the startup park target is energised there is no path back to a
        # torque cut: not a lost controller, not an exception, not the B button.
        self.assertEqual(source.count("disable_torque()"), 2)
        self.assertIn("if not servos_energized:", source)
        self.assertNotIn("emergency", source.lower())
        after_energizing = source.split("servos_energized = True", 1)[1]
        self.assertNotIn(
            "disable_torque()",
            after_energizing.split("if not servos_energized:")[0],
        )

    def test_robot_holds_park_before_waiting_for_arm(self):
        source = Path("playground/nubzuki/robot_runtime.py").read_text()
        startup = source.split("while True:", 1)[0]
        self.assertIn("hardware.enable_torque()", startup)
        self.assertLess(
            startup.index("hardware.set_positions"),
            startup.index("hardware.enable_torque()"),
        )
        self.assertIn("park(hardware, calibration, previous_targets, dt)", startup)
        self.assertIn("Holding park pose", startup)

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
