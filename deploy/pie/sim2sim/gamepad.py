"""Local pygame gamepad input for PIE sim2sim."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


GAMEPAD_AXIS = {
    "xbox": {"LX": 0, "LY": 1, "RX": 3, "RY": 4},
    "switch": {"LX": 0, "LY": 1, "RX": 2, "RY": 3},
}

GAMEPAD_BUTTON = {
    "xbox": {"A": 0, "START": 7},
    "switch": {"A": 0, "START": 11},
}


@dataclass(frozen=True)
class GamepadState:
    lx: float
    ly: float
    rx: float
    ry: float
    stop_pressed: bool
    reset_pressed: bool


def apply_deadzone(value: float, deadzone: float) -> float:
    """Apply a continuous deadzone while preserving full-scale endpoints."""
    if not math.isfinite(value):
        raise ValueError("Gamepad axis must be finite.")
    if not math.isfinite(deadzone) or not 0.0 <= deadzone < 1.0:
        raise ValueError("Gamepad deadzone must lie in [0, 1).")
    clipped = float(np.clip(value, -1.0, 1.0))
    if abs(clipped) <= deadzone:
        return 0.0
    return float(np.sign(clipped) * (abs(clipped) - deadzone) / (1.0 - deadzone))


def forward_speed_from_unit_input(
    value: float,
    *,
    max_speed: float,
    deadzone: float,
) -> float:
    if not math.isfinite(max_speed) or max_speed < 0.0:
        raise ValueError("Gamepad maximum speed must be finite and nonnegative.")
    return float(max(0.0, apply_deadzone(value, deadzone)) * max_speed)


def yaw_rate_from_unit_input(
    value: float,
    *,
    max_yaw_rate: float,
    deadzone: float,
) -> float:
    if not math.isfinite(max_yaw_rate) or max_yaw_rate < 0.0:
        raise ValueError("Gamepad maximum yaw rate must be finite and nonnegative.")
    return float(-apply_deadzone(value, deadzone) * max_yaw_rate)


class Gamepad:
    """Pygame reader with Xbox and Switch axis layouts."""

    def __init__(self, device_id: int, gamepad_type: str) -> None:
        if gamepad_type not in GAMEPAD_AXIS:
            raise ValueError(f"Unsupported gamepad type: {gamepad_type!r}.")
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "Gamepad control requires pygame. Install it with 'pip install pygame'."
            ) from exc

        self._pygame = pygame
        self._axis = GAMEPAD_AXIS[gamepad_type]
        self._button = GAMEPAD_BUTTON[gamepad_type]
        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError("No gamepad detected by pygame.")
        if not 0 <= device_id < count:
            raise ValueError(
                f"Joystick device {device_id} is unavailable; "
                f"detected {count} gamepad(s)."
            )
        self._joystick = pygame.joystick.Joystick(device_id)
        self._joystick.init()
        print(
            f"[INFO] Gamepad: id={device_id}, type={gamepad_type}, "
            f"name={self._joystick.get_name()!r}, "
            f"axes={self._joystick.get_numaxes()}, "
            f"buttons={self._joystick.get_numbuttons()}"
        )

    def _get_axis(self, name: str) -> float:
        axis_id = self._axis[name]
        if axis_id >= self._joystick.get_numaxes():
            return 0.0
        return float(np.clip(self._joystick.get_axis(axis_id), -1.0, 1.0))

    def _get_button(self, name: str) -> bool:
        button_id = self._button[name]
        if button_id >= self._joystick.get_numbuttons():
            return False
        return bool(self._joystick.get_button(button_id))

    def sample(self) -> GamepadState:
        self._pygame.event.pump()
        return GamepadState(
            lx=self._get_axis("LX"),
            ly=-self._get_axis("LY"),
            rx=self._get_axis("RX"),
            ry=-self._get_axis("RY"),
            stop_pressed=self._get_button("A"),
            reset_pressed=self._get_button("START"),
        )

    def close(self) -> None:
        self._joystick.quit()
        self._pygame.joystick.quit()


__all__ = [
    "GAMEPAD_AXIS",
    "Gamepad",
    "GamepadState",
    "apply_deadzone",
    "forward_speed_from_unit_input",
    "yaw_rate_from_unit_input",
]
