import json
import time
import unittest
import urllib.error
import urllib.request

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration
from playground.nubzuki.controller import axes_to_head_targets
from playground.nubzuki.head_dynamics import HeadDynamicsProfile
from playground.nubzuki.phone_controller import AXES, PhoneController


def post(url, payload):
    request = urllib.request.Request(
        url + "/input", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.status


class PhoneControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = PhoneController(host="127.0.0.1", port=0, timeout_s=0.3)
        self.url = f"http://127.0.0.1:{self.controller.server.server_address[1]}"
        self.calibration = NubzukiCalibration()

    def tearDown(self):
        self.controller.close()

    def test_page_is_served_and_self_contained(self):
        with urllib.request.urlopen(self.url, timeout=2) as response:
            page = response.read().decode()
        self.assertIn("<!doctype html>", page)
        self.assertNotIn("http://", page.split("<script>")[-1])
        for axis in AXES:
            self.assertIn(axis, page)
        self.assertIn("border-radius:50%", page)
        self.assertIn("MODE · HEAD", page)
        self.assertIn("MODE · WALK", page)
        self.assertIn("시뮬레이터가 실행 중인지 확인", page)
        self.assertNotIn("__TARGET_LABEL__", page)

    def test_page_names_the_physical_robot_when_requested(self):
        self.controller.close()
        self.controller = PhoneController(
            host="127.0.0.1", port=0, timeout_s=0.3,
            target_label="실물 로봇 제어",
        )
        self.url = f"http://127.0.0.1:{self.controller.server.server_address[1]}"
        with urllib.request.urlopen(self.url, timeout=2) as response:
            page = response.read().decode()
        self.assertIn("실물 로봇 제어가 실행 중인지 확인", page)

    def test_no_input_is_not_fresh(self):
        self.assertFalse(self.controller.fresh())
        self.assertTrue(self.controller.waiting())
        axes, a, b = self.controller.read()
        self.assertEqual(axes, {name: 0.0 for name in AXES})
        self.assertFalse(a or b)

    def test_input_arrives_and_is_reported_fresh(self):
        self.assertEqual(post(self.url, {"left_x": 0.5, "right_y": -0.25, "a": True}), 204)
        axes, a_pressed, b_pressed = self.controller.read()
        self.assertAlmostEqual(axes["left_x"], 0.5)
        self.assertAlmostEqual(axes["right_y"], -0.25)
        self.assertTrue(a_pressed)
        self.assertFalse(b_pressed)
        self.assertTrue(self.controller.fresh())

    def test_control_mode_is_validated(self):
        self.assertEqual(self.controller.mode(), "head")
        post(self.url, {"mode": "walk"})
        self.assertEqual(self.controller.mode(), "walk")
        post(self.url, {"mode": "walk", "left_x": 1.0, "right_y": -1.0})
        axes, _, _ = self.controller.read()
        self.assertEqual(axes["left_x"], 1.0)
        self.assertEqual(axes["right_y"], -1.0)
        post(self.url, {"mode": "not-a-mode"})
        self.assertEqual(self.controller.mode(), "head")

    def test_a_lost_connection_recentres_instead_of_holding_deflection(self):
        post(self.url, {"left_x": 1.0, "a": True})
        self.assertTrue(self.controller.fresh())
        time.sleep(0.35)
        axes, a_pressed, _ = self.controller.read()
        self.assertFalse(self.controller.fresh())
        self.assertEqual(axes["left_x"], 0.0)
        self.assertFalse(a_pressed)

    def test_b_survives_a_dropout_so_stop_is_never_lost(self):
        post(self.url, {"b": True})
        time.sleep(0.35)
        _, _, b_pressed = self.controller.read()
        self.assertTrue(b_pressed)

    def test_out_of_range_and_malformed_values_are_contained(self):
        post(self.url, {"left_x": 9.0, "left_y": -9.0, "right_x": "0.5", "right_y": float("nan")})
        axes, _, _ = self.controller.read()
        self.assertEqual(axes["left_x"], 1.0)
        self.assertEqual(axes["left_y"], -1.0)
        self.assertAlmostEqual(axes["right_x"], 0.5)
        self.assertEqual(axes["right_y"], 0.0)

    def test_garbage_body_is_rejected_without_killing_the_server(self):
        request = urllib.request.Request(
            self.url + "/input", data=b"not json",
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(request, timeout=2)
        self.assertEqual(post(self.url, {"left_x": 0.2}), 204)

    def test_phone_axes_drive_the_same_mapping_as_the_gamepad(self):
        profile = HeadDynamicsProfile.fallback(self.calibration)
        post(self.url, {"left_x": 1.0, "left_y": -1.0, "right_x": 1.0, "right_y": 0.0})
        axes, _, _ = self.controller.read()
        targets = axes_to_head_targets(axes, self.calibration, profile)
        self.assertAlmostEqual(targets["head_yaw"], self.calibration.limits_rad("head_yaw")[1])
        self.assertAlmostEqual(targets["head_pitch"], self.calibration.limits_rad("head_pitch")[1])
        self.assertEqual(targets["neck_pitch"], 0.0)
        self.assertEqual(set(targets), set(HEAD_JOINTS))


class FallbackProfileTests(unittest.TestCase):
    def test_fallback_is_marked_unmeasured(self):
        calibration = NubzukiCalibration()
        self.assertFalse(HeadDynamicsProfile.fallback(calibration).measured)

    def test_a_measured_profile_stays_measured(self):
        calibration = NubzukiCalibration()
        profile = HeadDynamicsProfile(
            {
                "schema_version": 1, "calibration_sha256": calibration.sha256,
                "control_frequency_hz": 50,
                "joystick_deadzone": {axis: 0.1 for axis in AXES},
                "joints": {
                    name: {"response_delay_s": 0.02, "velocity_limit_rad_s": 1.0,
                           "acceleration_limit_rad_s2": 2.0}
                    for name in HEAD_JOINTS
                },
            },
            calibration,
        )
        self.assertTrue(profile.measured)


if __name__ == "__main__":
    unittest.main()
