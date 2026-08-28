"""Standing-only Xbox mapping: two sticks control four head joints."""

from __future__ import annotations

import time

from playground.nubzuki.calibration import NubzukiCalibration
from playground.nubzuki.head_dynamics import HeadDynamicsProfile


def apply_deadzone(value: float, deadzone: float) -> float:
    magnitude = abs(float(value))
    if magnitude <= deadzone:
        return 0.0
    scaled = (magnitude - deadzone) / (1.0 - deadzone)
    return (1.0 if value >= 0 else -1.0) * min(scaled, 1.0)


def scale_axis(value: float, limits: tuple[float, float]) -> float:
    low, high = limits
    return float(value) * (high if value >= 0 else abs(low))


def axes_to_head_targets(
    axes: dict[str, float], calibration: NubzukiCalibration,
    profile: HeadDynamicsProfile,
) -> dict[str, float]:
    values = {
        name: apply_deadzone(axes[name], profile.deadzone[name]) for name in axes
    }
    return {
        "neck_pitch": scale_axis(values["right_y"], calibration.limits_rad("neck_pitch")),
        "head_pitch": scale_axis(values["left_y"], calibration.limits_rad("head_pitch")),
        "head_yaw": scale_axis(values["left_x"], calibration.limits_rad("head_yaw")),
        "head_roll": scale_axis(values["right_x"], calibration.limits_rad("head_roll")),
    }


class XboxController:
    def __init__(self, timeout_s: float = 0.25):
        import pygame
        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            raise RuntimeError("No joystick detected")
        self.device = pygame.joystick.Joystick(0)
        self.device.init()
        if self.device.get_numaxes() < 4:
            raise RuntimeError("Joystick needs at least four axes")
        self.timeout_s = float(timeout_s)
        self.last_read = time.monotonic()
        self.attached = True

    def read(self) -> tuple[dict[str, float], bool, bool]:
        self.pygame.event.pump()
        axes = {
            "left_x": -float(self.device.get_axis(0)),
            "left_y": -float(self.device.get_axis(1)),
            "right_x": -float(self.device.get_axis(2)),
            "right_y": -float(self.device.get_axis(3)),
        }
        # pygame keeps returning the last known axis values after a pad is
        # unplugged, so freshness has to come from the device still being
        # enumerated rather than from the poll having happened.
        self.attached = self.pygame.joystick.get_count() > 0
        if self.attached:
            self.last_read = time.monotonic()
        a_pressed = bool(self.device.get_button(0))
        b_pressed = bool(self.device.get_button(1))
        return axes, a_pressed, b_pressed

    def fresh(self) -> bool:
        return self.attached and time.monotonic() - self.last_read <= self.timeout_s

    def close(self) -> None:
        self.device.quit()
        self.pygame.quit()

