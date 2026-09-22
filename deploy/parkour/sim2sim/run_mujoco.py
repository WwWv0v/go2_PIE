#!/usr/bin/env python3
"""Self-contained AMP + V6 Real-Parkour deployment in native MuJoCo."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


os.environ.setdefault("XDG_CACHE_HOME", "/tmp/parkour_sim2sim_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/parkour_sim2sim_mpl_cache")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
if "--headless" in sys.argv and "--validate-only" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from .course import (
    CourseStage,
    build_straight_course,
    goal_direction_b,
    validate_policy_course,
)
from .policy_runtime import (
    CONTROL_DT,
    G1_KD,
    G1_KP,
    NUM_ACTIONS,
    PHYSICS_DT,
    AmpOnnxPolicy,
    DepthHistory,
    DepthSnapshot,
    RealParkourOnnxPolicy,
    StartupHeadingLock,
    UNITREE_KEY_BIT,
    apply_deadzone,
    build_amp_observation,
    build_real_observation,
    projected_gravity,
    torso_yaw_from_pelvis_and_waist,
)
from .simulator import (
    EFFORT_LIMIT,
    DepthCamera,
    DepthOverlay,
    ModelBindings,
    build_model,
    capture_reset_state,
    initialize_standing,
    open_viewer,
    restore_reset_state,
    set_goal_marker,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_AMP_POLICY = PACKAGE_ROOT / "assets/policies/amp_policy.onnx"
DEFAULT_REAL_POLICY = (
    PACKAGE_ROOT / "assets/policies/student_policy_v6_model_10000.onnx"
)
POLICY_DECIMATION = round(CONTROL_DT / PHYSICS_DT)
WAIST_INDICES = np.asarray((12, 13, 14), dtype=np.int32)


class PolicyMode(str, Enum):
    AMP = "amp"
    REAL_PARKOUR = "real_parkour"


MODE_LABEL = {
    PolicyMode.AMP: "AMP",
    PolicyMode.REAL_PARKOUR: "Real Parkour V6",
}


@dataclass(frozen=True)
class RobotPolicyState:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    quaternion_wxyz: np.ndarray
    angular_velocity: np.ndarray


@dataclass(frozen=True)
class PolicyStep:
    target: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    action: np.ndarray
    goal_direction: np.ndarray | None = None
    depth_frame: int | None = None


@dataclass(frozen=True)
class RemoteSample:
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    keys: int = 0


def _smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _key_mask(*names: str) -> int:
    return sum(1 << UNITREE_KEY_BIT[name] for name in names)


class InputRouter:
    """Edge-trigger the same LB+A/Y/B combinations used on the robot."""

    AMP_MASK = _key_mask("L1", "A")
    REAL_MASK = _key_mask("L1", "Y")
    RESET_MASK = _key_mask("L1", "B")

    def __init__(self, *, deadzone: float, command_ranges: np.ndarray) -> None:
        self.deadzone = float(deadzone)
        self.command_ranges = np.asarray(command_ranges, dtype=np.float32).reshape(3, 2)
        self._previous = {
            self.AMP_MASK: False,
            self.REAL_MASK: False,
            self.RESET_MASK: False,
        }

    def update(
        self, sample: RemoteSample
    ) -> tuple[PolicyMode | None, bool, np.ndarray]:
        active = {
            mask: sample.keys & mask == mask
            for mask in (self.AMP_MASK, self.REAL_MASK, self.RESET_MASK)
        }
        request: PolicyMode | None = None
        if active[self.AMP_MASK] and not self._previous[self.AMP_MASK]:
            request = PolicyMode.AMP
        if active[self.REAL_MASK] and not self._previous[self.REAL_MASK]:
            request = PolicyMode.REAL_PARKOUR
        reset = active[self.RESET_MASK] and not self._previous[self.RESET_MASK]
        self._previous.update(active)

        unit = np.asarray(
            (
                apply_deadzone(sample.ly, self.deadzone),
                apply_deadzone(-sample.lx, self.deadzone),
                apply_deadzone(-sample.rx, self.deadzone),
            ),
            dtype=np.float32,
        )
        command = np.empty(3, dtype=np.float32)
        for index, value in enumerate(unit):
            limit = (
                self.command_ranges[index, 1]
                if value >= 0.0
                else abs(self.command_ranges[index, 0])
            )
            command[index] = value * limit
        return request, reset, command


class Gamepad:
    AXES = {
        "xbox": {"LX": 0, "LY": 1, "RX": 3, "RY": 4, "LT": 2, "RT": 5},
        "switch": {"LX": 0, "LY": 1, "RX": 2, "RY": 3, "LT": 5, "RT": 4},
    }
    BUTTONS = {
        "xbox": {
            "A": 0, "B": 1, "X": 2, "Y": 3, "LB": 4, "RB": 5,
            "SELECT": 6, "START": 7,
        },
        "switch": {
            "A": 0, "B": 1, "X": 3, "Y": 4, "LB": 6, "RB": 7,
            "SELECT": 10, "START": 11,
        },
    }

    def __init__(self, device: int, gamepad_type: str) -> None:
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="pkg_resources is deprecated as an API.*",
                    category=UserWarning,
                )
                import pygame
                import pygame._sdl2.controller as sdl_controller
        except ImportError as exc:
            raise RuntimeError(
                "Gamepad control requires pygame; install it with 'pip install pygame'."
            ) from exc
        self._pygame = pygame
        self._axes = self.AXES[gamepad_type]
        self._buttons = self.BUTTONS[gamepad_type]
        self._controller = None
        self._standard_axis: dict[str, int] = {}
        self._standard_button: dict[str, int] = {}
        pygame.init()
        pygame.joystick.init()
        sdl_controller.init()
        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError("No gamepad detected by pygame.")
        if not 0 <= device < count:
            raise ValueError(
                f"Joystick device {device} is unavailable; detected {count} gamepad(s)."
            )
        if sdl_controller.is_controller(device):
            self._controller = sdl_controller.Controller(device)
            self._controller.init()
            self._joystick = self._controller.as_joystick()
            self._standard_axis = {
                "LX": pygame.CONTROLLER_AXIS_LEFTX,
                "LY": pygame.CONTROLLER_AXIS_LEFTY,
                "RX": pygame.CONTROLLER_AXIS_RIGHTX,
                "RY": pygame.CONTROLLER_AXIS_RIGHTY,
                "LT": pygame.CONTROLLER_AXIS_TRIGGERLEFT,
                "RT": pygame.CONTROLLER_AXIS_TRIGGERRIGHT,
            }
            self._standard_button = {
                "A": pygame.CONTROLLER_BUTTON_A,
                "B": pygame.CONTROLLER_BUTTON_B,
                "X": pygame.CONTROLLER_BUTTON_X,
                "Y": pygame.CONTROLLER_BUTTON_Y,
                "LB": pygame.CONTROLLER_BUTTON_LEFTSHOULDER,
                "RB": pygame.CONTROLLER_BUTTON_RIGHTSHOULDER,
                "SELECT": pygame.CONTROLLER_BUTTON_BACK,
                "START": pygame.CONTROLLER_BUTTON_START,
                "DPAD_UP": pygame.CONTROLLER_BUTTON_DPAD_UP,
                "DPAD_RIGHT": pygame.CONTROLLER_BUTTON_DPAD_RIGHT,
                "DPAD_DOWN": pygame.CONTROLLER_BUTTON_DPAD_DOWN,
                "DPAD_LEFT": pygame.CONTROLLER_BUTTON_DPAD_LEFT,
            }
            mapping = "SDL semantic controller"
        else:
            self._joystick = pygame.joystick.Joystick(device)
            self._joystick.init()
            mapping = f"raw {gamepad_type} fallback"
        print(
            f"[INFO] Gamepad: id={device}, type={gamepad_type}, "
            f"name={self._joystick.get_name()!r}, "
            f"axes={self._joystick.get_numaxes()}, "
            f"buttons={self._joystick.get_numbuttons()}, mapping={mapping}"
        )

    def _get_axis(self, name: str) -> float:
        if self._controller is not None:
            raw = int(self._controller.get_axis(self._standard_axis[name]))
            divisor = 32768.0 if raw < 0 else 32767.0
            return float(np.clip(raw / divisor, -1.0, 1.0))
        axis = self._axes[name]
        if axis >= self._joystick.get_numaxes():
            return 0.0
        return float(np.clip(self._joystick.get_axis(axis), -1.0, 1.0))

    def _get_button(self, name: str) -> bool:
        if self._controller is not None:
            return bool(self._controller.get_button(self._standard_button[name]))
        button = self._buttons[name]
        if button >= self._joystick.get_numbuttons():
            return False
        return bool(self._joystick.get_button(button))

    def sample(self) -> RemoteSample:
        self._pygame.event.pump()
        if self._controller is not None:
            hat = (
                int(self._get_button("DPAD_RIGHT"))
                - int(self._get_button("DPAD_LEFT")),
                int(self._get_button("DPAD_UP"))
                - int(self._get_button("DPAD_DOWN")),
            )
        else:
            hat = self._joystick.get_hat(0) if self._joystick.get_numhats() else (0, 0)
        pressed = {
            "R1": self._get_button("RB"),
            "L1": self._get_button("LB"),
            "start": self._get_button("START"),
            "select": self._get_button("SELECT"),
            "R2": self._get_axis("RT") > 0.0,
            "L2": self._get_axis("LT") > 0.0,
            "F1": False,
            "F2": False,
            "A": self._get_button("A"),
            "B": self._get_button("B"),
            "X": self._get_button("X"),
            "Y": self._get_button("Y"),
            "up": hat[1] > 0,
            "right": hat[0] > 0,
            "down": hat[1] < 0,
            "left": hat[0] < 0,
        }
        keys = sum(
            int(value) << UNITREE_KEY_BIT[name] for name, value in pressed.items()
        )
        return RemoteSample(
            lx=self._get_axis("LX"),
            ly=-self._get_axis("LY"),
            rx=self._get_axis("RX"),
            keys=keys,
        )

    def close(self) -> None:
        if self._controller is not None:
            self._controller.quit()
        else:
            self._joystick.quit()
        self._pygame.joystick.quit()


def create_gamepad(args: argparse.Namespace) -> Gamepad:
    """Connect the required deployment gamepad."""
    return Gamepad(args.joystick_device, args.gamepad_type)


class PolicySuite:
    def __init__(
        self,
        amp_path: Path,
        real_path: Path,
        provider: str,
        goal_mode: str,
        action_clip: float,
    ) -> None:
        self.amp = AmpOnnxPolicy(amp_path, provider)
        self.real = RealParkourOnnxPolicy(real_path, provider)
        self.goal_mode = goal_mode
        self.action_clip = float(action_clip)
        self.heading = StartupHeadingLock()
        self.mode = PolicyMode.AMP
        self.mode_entered_at = 0.0
        self.amp_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.real_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    def reset(self, now: float) -> None:
        self.mode = PolicyMode.AMP
        self.mode_entered_at = float(now)
        self.amp_action.fill(0.0)
        self.real_action.fill(0.0)
        self.heading.reset()

    @staticmethod
    def _torso_yaw(state: RobotPolicyState) -> float:
        return torso_yaw_from_pelvis_and_waist(
            state.quaternion_wxyz, state.joint_pos[WAIST_INDICES]
        )

    def switch(self, mode: PolicyMode, state: RobotPolicyState, now: float) -> bool:
        if mode == self.mode and mode == PolicyMode.AMP:
            return False
        self.mode = mode
        self.mode_entered_at = float(now)
        self.amp_action.fill(0.0)
        self.real_action.fill(0.0)
        if mode == PolicyMode.REAL_PARKOUR:
            self.heading.lock_yaw(self._torso_yaw(state))
        else:
            self.heading.reset()
        return True

    def infer(
        self,
        state: RobotPolicyState,
        amp_command: np.ndarray,
        depth: DepthSnapshot,
        torso_pos: np.ndarray,
        torso_quat: np.ndarray,
        active_goal: CourseStage,
    ) -> PolicyStep:
        if self.mode == PolicyMode.AMP:
            observation = build_amp_observation(
                state, amp_command, self.amp_action, self.amp.params.default_pos
            )
            self.amp_action = np.clip(
                self.amp(observation), -self.action_clip, self.action_clip
            )
            target = (
                self.amp.params.default_pos
                + self.amp.params.action_scale * self.amp_action
            )
            return PolicyStep(
                target.astype(np.float64),
                self.amp.params.kp.astype(np.float64),
                self.amp.params.kd.astype(np.float64),
                self.amp_action.copy(),
            )

        torso_yaw = self._torso_yaw(state)
        if self.goal_mode == "startup-heading":
            direction = self.heading.direction_b_from_yaw(torso_yaw)
        else:
            direction = goal_direction_b(active_goal.goal_pos_w, torso_pos, torso_quat)
        observation = build_real_observation(
            direction=direction,
            state=state,
            previous_action=self.real_action,
            default_pos=self.real.default_pos,
        )
        self.real_action = np.clip(
            self.real(observation, depth.policy_depth),
            -self.action_clip,
            self.action_clip,
        )
        return PolicyStep(
            self.real.joint_target(self.real_action).astype(np.float64),
            G1_KP.copy(),
            G1_KD.copy(),
            self.real_action.copy(),
            direction,
            depth.frame_number,
        )


def read_policy_state(data: mujoco.MjData, bindings: ModelBindings) -> RobotPolicyState:
    gyro = slice(bindings.gyro_sensor_adr, bindings.gyro_sensor_adr + 3)
    return RobotPolicyState(
        np.asarray(data.qpos[bindings.joint_qpos_adr], dtype=np.float32).copy(),
        np.asarray(data.qvel[bindings.joint_dof_adr], dtype=np.float32).copy(),
        np.asarray(data.xquat[bindings.pelvis_body_id], dtype=np.float32).copy(),
        np.asarray(data.sensordata[gyro], dtype=np.float32).copy(),
    )


def torso_pose(
    data: mujoco.MjData, bindings: ModelBindings
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(data.xpos[bindings.torso_body_id], dtype=np.float64).copy(),
        np.asarray(data.xquat[bindings.torso_body_id], dtype=np.float64).copy(),
    )


def _check_safety(
    state: RobotPolicyState, max_tilt: float, max_angular_speed: float
) -> None:
    gravity = projected_gravity(state.quaternion_wxyz)
    tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
    angular_speed = float(np.linalg.norm(state.angular_velocity))
    if tilt > max_tilt:
        raise RuntimeError(f"Safety stop: tilt {tilt:.3f}rad > {max_tilt:.3f}rad")
    if angular_speed > max_angular_speed:
        raise RuntimeError(
            f"Safety stop: angular speed {angular_speed:.3f} > {max_angular_speed:.3f}rad/s"
        )


def _synthetic_depth() -> DepthSnapshot:
    return DepthSnapshot(
        0.0,
        0,
        np.full((1, 8, 18, 32), 0.5, dtype=np.float32),
    )


def validate_only(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bindings: ModelBindings,
    policies: PolicySuite,
    active_goal: CourseStage,
) -> None:
    state = read_policy_state(data, bindings)
    position, quaternion = torso_pose(data, bindings)
    depth = _synthetic_depth()
    amp = policies.infer(
        state, np.zeros(3, dtype=np.float32), depth, position, quaternion, active_goal
    )
    policies.switch(PolicyMode.REAL_PARKOUR, state, float(data.time))
    real = policies.infer(
        state, np.zeros(3, dtype=np.float32), depth, position, quaternion, active_goal
    )
    for label, step in (("AMP", amp), ("Real Parkour", real)):
        arrays = (step.target, step.kp, step.kd, step.action)
        if any(value.shape != (NUM_ACTIONS,) for value in arrays):
            raise AssertionError(f"{label} output shape is invalid")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise FloatingPointError(f"{label} output contains non-finite values")
    if policies.goal_mode == "startup-heading" and not np.allclose(
        real.goal_direction, (1.0, 0.0), atol=1.0e-6
    ):
        raise AssertionError("Startup heading lock contract failed")
    router = InputRouter(
        deadzone=0.1,
        command_ranges=np.asarray(((-0.5, 0.8), (-0.5, 0.5), (-1.0, 1.0))),
    )
    for expected, mask in (
        (PolicyMode.AMP, router.AMP_MASK),
        (PolicyMode.REAL_PARKOUR, router.REAL_MASK),
    ):
        requested, reset, _ = router.update(RemoteSample(keys=mask))
        if requested != expected or reset:
            raise AssertionError("Gamepad FSM contract failed")
        router.update(RemoteSample())
    _, reset, _ = router.update(RemoteSample(keys=router.RESET_MASK))
    if not reset:
        raise AssertionError("Gamepad reset contract failed")
    policies.reset(float(data.time))
    print(
        "[OK] Parkour sim2sim standalone validation passed\n"
        f"     terrain=15, nq/nv/nu={model.nq}/{model.nv}/{model.nu}\n"
        "     AMP=obs[1,96]->action[1,29]\n"
        "     V6=obs[1,95]+depth[1,8,18,32]->action[1,29]"
    )


def _apply_pd_torque(
    data: mujoco.MjData,
    bindings: ModelBindings,
    target: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> None:
    position = data.qpos[bindings.joint_qpos_adr]
    velocity = data.qvel[bindings.joint_dof_adr]
    torque = kp * (target - position) - kd * velocity
    data.ctrl[bindings.actuator_ids] = np.clip(torque, -EFFORT_LIMIT, EFFORT_LIMIT)


def run(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bindings: ModelBindings,
    policies: PolicySuite,
    active_goal: CourseStage,
    args: argparse.Namespace,
) -> None:
    gamepad = create_gamepad(args)
    router = InputRouter(
        deadzone=args.deadzone,
        command_ranges=np.asarray(
            (args.cmd_x_range, args.cmd_y_range, args.cmd_yaw_range), dtype=np.float32
        ),
    )
    reset_state = capture_reset_state(data)
    initial_joint_pos = data.qpos[bindings.joint_qpos_adr].copy()
    target = initial_joint_pos.copy()
    active_kp = policies.amp.params.kp.astype(np.float64).copy()
    active_kd = policies.amp.params.kd.astype(np.float64).copy()
    blend_target = target.copy()
    blend_kp = active_kp.copy()
    blend_kd = active_kd.copy()
    blend_started: float | None = None

    camera = DepthCamera(model, bindings.camera_id)
    history = DepthHistory(maxlen=args.depth_history)
    _, latest_policy = camera.capture(data)
    depth_frame = 0
    history.push(DepthSnapshot(float(data.time), depth_frame, latest_policy))
    latest_student_depth = history.policy_input(
        float(data.time) - args.depth_latency
    ).policy_depth
    next_depth = float(data.time) + 1.0 / args.depth_rate
    viewer_context = (
        nullcontext(None)
        if args.headless
        else open_viewer(model, data, bindings, None)
    )

    start_sim = float(data.time)
    start_wall = time.monotonic()
    next_print = start_sim + args.print_every
    last_time = float(data.time)
    initialized = False
    pending_mode: PolicyMode | None = PolicyMode.AMP
    policy_steps = 0
    max_action = 0.0

    def reset(source: str) -> None:
        nonlocal start_sim, start_wall, next_print, next_depth, depth_frame
        nonlocal initialized, pending_mode, blend_started, policy_steps, max_action
        nonlocal initial_joint_pos, latest_policy, latest_student_depth
        restore_reset_state(model, data, reset_state)
        policies.reset(float(data.time))
        initial_joint_pos = data.qpos[bindings.joint_qpos_adr].copy()
        target[:] = initial_joint_pos
        active_kp[:] = policies.amp.params.kp
        active_kd[:] = policies.amp.params.kd
        history.clear()
        _, latest_policy = camera.capture(data)
        depth_frame = 0
        history.push(DepthSnapshot(float(data.time), depth_frame, latest_policy))
        latest_student_depth = history.policy_input(
            float(data.time) - args.depth_latency
        ).policy_depth
        next_depth = float(data.time) + 1.0 / args.depth_rate
        start_sim = float(data.time)
        start_wall = time.monotonic()
        next_print = start_sim + args.print_every
        initialized = False
        pending_mode = PolicyMode.AMP
        blend_started = None
        policy_steps = 0
        max_action = 0.0
        print(f"[RESET] {source}")

    print(
        "[INFO] Terrain 15 only. Gamepad: LB+A=AMP, LB+Y=Real V6, LB+B=reset."
    )
    try:
        with viewer_context as viewer:
            overlay = DepthOverlay(viewer) if viewer is not None and args.show_depth else None
            overlay_active = False
            while args.duration == 0.0 or data.time - start_sim < args.duration:
                if viewer is not None and not viewer.is_running():
                    break
                loop_wall = time.monotonic()
                pad_mode, pad_reset, amp_command = router.update(gamepad.sample())
                if pad_reset:
                    reset("gamepad LB+B")
                elif data.time + 1.0e-9 < last_time:
                    reset("viewer")
                last_time = float(data.time)
                if pad_mode is not None:
                    pending_mode = pad_mode

                elapsed = float(data.time) - start_sim
                active_at = args.stand_up_duration + args.stand_hold_duration
                state = read_policy_state(data, bindings)
                stage_label: str
                if elapsed < args.stand_up_duration:
                    blend = _smoothstep(elapsed / max(args.stand_up_duration, 1.0e-9))
                    target[:] = (1.0 - blend) * initial_joint_pos + blend * policies.amp.params.default_pos
                    stage_label = "stand-up"
                elif elapsed < active_at:
                    target[:] = policies.amp.params.default_pos
                    stage_label = "hold"
                else:
                    if not initialized:
                        policies.reset(float(data.time))
                        assert pending_mode is not None
                        if pending_mode != PolicyMode.AMP:
                            policies.switch(pending_mode, state, float(data.time))
                        pending_mode = None
                        initialized = True
                        blend_target[:] = target
                        blend_kp[:] = active_kp
                        blend_kd[:] = active_kd
                        blend_started = float(data.time)
                        print(f"[FSM] Active: {MODE_LABEL[policies.mode]}")
                    elif pending_mode is not None:
                        previous = policies.mode
                        blend_target[:] = target
                        blend_kp[:] = active_kp
                        blend_kd[:] = active_kd
                        if policies.switch(pending_mode, state, float(data.time)):
                            blend_started = float(data.time)
                            print(f"[FSM] {MODE_LABEL[previous]} -> {MODE_LABEL[pending_mode]}")
                        pending_mode = None

                    position, quaternion = torso_pose(data, bindings)
                    depth = history.policy_input(float(data.time) - args.depth_latency)
                    latest_student_depth = depth.policy_depth
                    warmup = min(
                        1.0,
                        (float(data.time) - policies.mode_entered_at)
                        / max(args.amp_command_warmup, 1.0e-9),
                    )
                    step = policies.infer(
                        state,
                        amp_command * warmup,
                        depth,
                        position,
                        quaternion,
                        active_goal,
                    )
                    if blend_started is not None:
                        ratio = (
                            1.0
                            if args.transition_duration == 0.0
                            else (float(data.time) - blend_started) / args.transition_duration
                        )
                        blend = _smoothstep(ratio)
                        target[:] = (1.0 - blend) * blend_target + blend * step.target
                        active_kp[:] = (1.0 - blend) * blend_kp + blend * step.kp
                        active_kd[:] = (1.0 - blend) * blend_kd + blend * step.kd
                        if ratio >= 1.0:
                            blend_started = None
                    else:
                        target[:] = step.target
                        active_kp[:] = step.kp
                        active_kd[:] = step.kd
                    policy_steps += 1
                    max_action = max(max_action, float(np.max(np.abs(step.action))))
                    stage_label = MODE_LABEL[policies.mode]

                depth_updated = False
                for _ in range(POLICY_DECIMATION):
                    _apply_pd_torque(data, bindings, target, active_kp, active_kd)
                    mujoco.mj_step(model, data)
                    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                        raise FloatingPointError(f"MuJoCo diverged at t={data.time:.3f}s")
                    if data.time + 1.0e-9 >= next_depth:
                        _, latest_policy = camera.capture(data)
                        depth_frame += 1
                        history.push(DepthSnapshot(float(data.time), depth_frame, latest_policy))
                        depth_updated = True
                        while next_depth <= data.time + 1.0e-9:
                            next_depth += 1.0 / args.depth_rate

                if args.safety_stop:
                    _check_safety(
                        read_policy_state(data, bindings),
                        args.max_tilt,
                        args.max_angular_speed,
                    )
                last_time = float(data.time)
                if viewer is not None:
                    show = bool(args.show_depth and initialized and policies.mode == PolicyMode.REAL_PARKOUR)
                    if show and (depth_updated or not overlay_active):
                        assert overlay is not None
                        overlay.update(latest_student_depth)
                        overlay_active = True
                    elif not show and overlay_active:
                        viewer.clear_images()
                        overlay_active = False
                    viewer.sync()

                if data.time + 1.0e-9 >= next_print:
                    position, _ = torso_pose(data, bindings)
                    print(
                        f"[STATE] t={data.time - start_sim:6.2f}s mode={stage_label:<16} "
                        f"torso=({position[0]:.2f},{position[1]:.2f},{position[2]:.2f}) "
                        f"depth={depth_frame} max_action={max_action:.2f}"
                    )
                    while next_print <= data.time + 1.0e-9:
                        next_print += args.print_every
                if args.realtime:
                    remaining = CONTROL_DT - (time.monotonic() - loop_wall)
                    if remaining > 0.0:
                        time.sleep(remaining)
    finally:
        camera.close()
        gamepad.close()
    wall = time.monotonic() - start_wall
    print(
        f"[DONE] sim={data.time - start_sim:.2f}s wall={wall:.2f}s "
        f"policy_steps={policy_steps} terrain=15"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone Unitree G1 AMP + V6 Real-Parkour MuJoCo deployment"
    )
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--terrain", choices=("15",), default="15")
    parser.add_argument(
        "--goal-mode", choices=("startup-heading", "course"), default="startup-heading"
    )
    parser.add_argument("--ccd-iterations", type=int, default=50)
    parser.add_argument(
        "--duration", type=float, default=0.0, help="0 runs until the viewer closes"
    )
    parser.add_argument(
        "--stand-up-duration",
        type=float,
        default=0.0,
        help="Optional open-loop pose interpolation; disabled by default",
    )
    parser.add_argument(
        "--stand-hold-duration",
        type=float,
        default=0.0,
        help="Optional open-loop pose hold; disabled by default",
    )
    parser.add_argument("--transition-duration", type=float, default=0.35)
    parser.add_argument("--amp-command-warmup", type=float, default=0.5)
    parser.add_argument("--cmd-x-range", type=float, nargs=2, default=(-0.5, 0.8))
    parser.add_argument("--cmd-y-range", type=float, nargs=2, default=(-0.5, 0.5))
    parser.add_argument("--cmd-yaw-range", type=float, nargs=2, default=(-1.0, 1.0))
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--joystick-device", type=int, default=0)
    parser.add_argument("--gamepad-type", choices=("xbox", "switch"), default="xbox")
    parser.add_argument("--depth-rate", type=float, default=50.0)
    parser.add_argument("--depth-latency", type=float, default=0.0)
    parser.add_argument("--depth-history", type=int, default=64)
    parser.add_argument("--action-clip", type=float, default=100.0)
    parser.add_argument("--max-tilt", type=float, default=1.2)
    parser.add_argument("--max-angular-speed", type=float, default=12.0)
    parser.add_argument("--no-safety-stop", action="store_false", dest="safety_stop")
    parser.set_defaults(safety_stop=True)
    parser.add_argument("--print-every", type=float, default=1.0)
    parser.add_argument("--no-realtime", action="store_false", dest="realtime")
    parser.set_defaults(realtime=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--show-depth", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.joystick_device < 0:
        raise ValueError("--joystick-device cannot be negative")
    positive = {
        "ccd_iterations": args.ccd_iterations,
        "depth_rate": args.depth_rate,
        "depth_history": args.depth_history,
        "action_clip": args.action_clip,
        "print_every": args.print_every,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    nonnegative = (
        "duration",
        "stand_up_duration",
        "stand_hold_duration",
        "transition_duration",
        "amp_command_warmup",
        "depth_latency",
    )
    for name in nonnegative:
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    policies = PolicySuite(
        DEFAULT_AMP_POLICY,
        DEFAULT_REAL_POLICY,
        args.provider,
        args.goal_mode,
        args.action_clip,
    )
    validate_policy_course(policies.real.metadata)
    stages = build_straight_course(args.terrain)
    model, bindings = build_model(stages, args.ccd_iterations)
    data = initialize_standing(model, bindings, policies.amp.params.default_pos)
    set_goal_marker(data, bindings, stages[0])
    mujoco.mj_forward(model, data)
    active_goal = stages[0]
    if args.validate_only:
        validate_only(model, data, bindings, policies, active_goal)
        return 0
    run(model, data, bindings, policies, active_goal, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
