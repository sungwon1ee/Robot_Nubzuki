import unittest

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration
from playground.nubzuki.head_dynamics import HeadDynamicsProfile, HeadTrajectoryLimiter


DT = 0.02


def make_profile(calibration, velocity_limit, acceleration_limit):
    return HeadDynamicsProfile(
        {
            "schema_version": 1,
            "calibration_sha256": calibration.sha256,
            "control_frequency_hz": 50,
            "joystick_deadzone": {
                axis: 0.08 for axis in ("left_x", "left_y", "right_x", "right_y")
            },
            "joints": {
                name: {
                    "response_delay_s": 0.02,
                    "velocity_limit_rad_s": velocity_limit,
                    "acceleration_limit_rad_s2": acceleration_limit,
                }
                for name in HEAD_JOINTS
            },
        },
        calibration,
    )


def drive(limiter, joint, target, steps=400):
    """Return peak speed, peak |acceleration| and the settled position."""
    peak_speed = 0.0
    peak_acceleration = 0.0
    previous = limiter.velocity[joint]
    for _ in range(steps):
        limiter.step({name: target for name in HEAD_JOINTS})
        velocity = limiter.velocity[joint]
        peak_speed = max(peak_speed, abs(velocity))
        peak_acceleration = max(peak_acceleration, abs(velocity - previous) / DT)
        previous = velocity
        if limiter.position[joint] == target and velocity == 0.0:
            break
    return peak_speed, peak_acceleration, limiter.position[joint]


class HeadTrajectoryLimiterTests(unittest.TestCase):
    def setUp(self):
        self.calibration = NubzukiCalibration()

    def test_full_deflection_never_exceeds_measured_limits(self):
        """A flicked stick must not be turned into a jerk at either end.

        The limiter used to run at full speed until it crossed the target and
        then zero its velocity in a single step, a deceleration many times the
        measured limit.
        """
        tolerance = 1e-9
        for velocity_limit, acceleration_limit in (
            (0.3, 1.0), (0.5, 2.0), (1.0, 5.0), (2.0, 10.0), (5.0, 50.0),
        ):
            profile = make_profile(self.calibration, velocity_limit, acceleration_limit)
            for joint in HEAD_JOINTS:
                for bound in self.calibration.limits_rad(joint):
                    limiter = HeadTrajectoryLimiter(profile, dt=DT)
                    speed, acceleration, position = drive(limiter, joint, bound)
                    with self.subTest(joint=joint, target=bound, v=velocity_limit):
                        self.assertLessEqual(speed, velocity_limit + tolerance)
                        self.assertLessEqual(acceleration, acceleration_limit + tolerance)
                        self.assertAlmostEqual(position, bound, places=12)

    def test_reversal_mid_motion_stays_within_limits(self):
        profile = make_profile(self.calibration, 2.0, 10.0)
        limiter = HeadTrajectoryLimiter(profile, dt=DT)
        low, high = self.calibration.limits_rad("head_yaw")
        for _ in range(5):
            limiter.step({name: high for name in HEAD_JOINTS})
        previous = limiter.velocity["head_yaw"]
        for _ in range(200):
            limiter.step({name: low for name in HEAD_JOINTS})
            velocity = limiter.velocity["head_yaw"]
            self.assertLessEqual(abs(velocity), 2.0 + 1e-9)
            self.assertLessEqual(abs(velocity - previous) / DT, 10.0 + 1e-9)
            previous = velocity
        self.assertAlmostEqual(limiter.position["head_yaw"], low, places=12)

    def test_slow_stick_is_tracked_without_lag(self):
        """Motion well inside the limits should pass through untouched."""
        profile = make_profile(self.calibration, 2.0, 10.0)
        limiter = HeadTrajectoryLimiter(profile, dt=DT)
        position = 0.0
        for _ in range(40):
            position += 0.002  # 0.1 rad/s, far below the 2.0 rad/s cap
            reached = limiter.step({name: position for name in HEAD_JOINTS})
            self.assertAlmostEqual(reached["head_yaw"], position, places=12)

    def test_target_held_at_zero_stays_at_zero(self):
        profile = make_profile(self.calibration, 2.0, 10.0)
        limiter = HeadTrajectoryLimiter(profile, dt=DT)
        for _ in range(50):
            reached = limiter.step({name: 0.0 for name in HEAD_JOINTS})
        for name in HEAD_JOINTS:
            self.assertEqual(reached[name], 0.0)
            self.assertEqual(limiter.velocity[name], 0.0)


if __name__ == "__main__":
    unittest.main()
