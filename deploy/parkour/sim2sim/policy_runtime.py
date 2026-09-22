"""Self-contained AMP/V6 ONNX, observation, depth, and control contracts."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


PHYSICS_DT = 0.005
CONTROL_DT = 0.02
NUM_ACTIONS = 29
AMP_OBS_DIM = 96
REAL_OBS_DIM = 95
TASK_ID_V6 = "Unitree-G1-Direction-Mixed-Student-V6"
OBSERVATION_LAYOUT = (
    "goal_direction_b[2],projected_gravity[3],base_ang_vel[3],"
    "joint_pos_rel[29],joint_vel[29],previous_action[29]"
)

ACTUATED_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

CAMERA_POS = np.asarray((0.0576235, 0.01753, 0.42987), dtype=np.float64)
CAMERA_QUAT_WXYZ = np.asarray(
    (0.6592524821011074, 0.2557071857487173, -0.2557071857487173, -0.6592524821011074),
    dtype=np.float64,
)
DEPTH_RENDER_HEIGHT = 36
DEPTH_RENDER_WIDTH = 64
DEPTH_FOVY_DEG = 58.29


@dataclass(frozen=True)
class DepthContract:
    height: int = 18
    width: int = 32
    channels: int = 8
    rate_hz: float = 50.0
    min_m: float = 0.0
    max_m: float = 2.5
    frame_offsets_s: tuple[float, ...] = (
        0.7,
        0.6,
        0.5,
        0.4,
        0.3,
        0.2,
        0.1,
        0.0,
    )

    @property
    def frame_shape(self) -> tuple[int, int, int, int]:
        return (1, 1, self.height, self.width)

    @property
    def tensor_shape(self) -> tuple[int, int, int, int]:
        return (1, self.channels, self.height, self.width)


V6_DEPTH_CONTRACT = DepthContract()


@dataclass(frozen=True)
class DepthSnapshot:
    timestamp: float
    frame_number: int
    policy_depth: np.ndarray


class DepthHistory:
    """Timestamped V6 single frames assembled oldest-to-newest on demand."""

    def __init__(self, maxlen: int = 64, contract: DepthContract = V6_DEPTH_CONTRACT):
        if maxlen < 2:
            raise ValueError("Depth history must retain at least two frames.")
        self.contract = contract
        self._frames: deque[DepthSnapshot] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def push(self, snapshot: DepthSnapshot) -> None:
        depth = np.asarray(snapshot.policy_depth, dtype=np.float32)
        if depth.shape != self.contract.frame_shape:
            raise ValueError(
                f"Depth frame is {depth.shape}; expected {self.contract.frame_shape}."
            )
        if not np.all(np.isfinite(depth)) or np.any((depth < 0.0) | (depth > 1.0)):
            raise FloatingPointError("Normalized depth must be finite and in [0,1].")
        stored = np.ascontiguousarray(depth)
        stored.setflags(write=False)
        with self._lock:
            if self._frames and snapshot.timestamp < self._frames[-1].timestamp:
                raise ValueError("Depth timestamps must be nondecreasing.")
            self._frames.append(
                DepthSnapshot(float(snapshot.timestamp), int(snapshot.frame_number), stored)
            )

    def policy_input(self, observation_time: float) -> DepthSnapshot:
        if not math.isfinite(observation_time):
            raise ValueError("Observation time must be finite.")
        with self._lock:
            frames = tuple(self._frames)
        if not frames:
            raise RuntimeError("Depth history is empty.")
        selected: list[DepthSnapshot] = []
        for offset in self.contract.frame_offsets_s:
            target = observation_time - offset
            candidate = frames[0]
            for frame in frames:
                if frame.timestamp <= target + 1.0e-9:
                    candidate = frame
                else:
                    break
            selected.append(candidate)
        stacked = np.ascontiguousarray(
            np.concatenate([frame.policy_depth for frame in selected], axis=1)
        )
        if stacked.shape != self.contract.tensor_shape:
            raise AssertionError("Depth history assembled an invalid tensor.")
        newest = selected[-1]
        return DepthSnapshot(newest.timestamp, newest.frame_number, stacked)


def preprocess_v6_depth(optical_z: np.ndarray) -> np.ndarray:
    """Apply the training V6 64x36 upper crop, reflect blur, and normalization."""
    depth = np.asarray(optical_z, dtype=np.float32)
    if depth.shape != (DEPTH_RENDER_HEIGHT, DEPTH_RENDER_WIDTH):
        raise ValueError(
            f"Raw V6 depth must be {(DEPTH_RENDER_HEIGHT, DEPTH_RENDER_WIDTH)}; "
            f"got {depth.shape}."
        )
    depth = depth.copy()
    depth[~np.isfinite(depth) | (depth <= 0.0)] = V6_DEPTH_CONTRACT.max_m
    depth = depth[:18, 16:48]
    coords = np.asarray((-1.0, 0.0, 1.0), dtype=np.float32)
    kernel_1d = np.exp(-0.5 * coords**2)
    kernel_1d /= kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    padded = np.pad(depth, ((1, 1), (1, 1)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    blurred = np.sum(windows * kernel, axis=(-2, -1), dtype=np.float32)
    normalized = np.clip(blurred, 0.0, V6_DEPTH_CONTRACT.max_m)
    normalized /= V6_DEPTH_CONTRACT.max_m
    return np.ascontiguousarray(normalized[None, None], dtype=np.float32)


def quat_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12 or not np.all(np.isfinite(quat)):
        raise ValueError("Quaternion is invalid.")
    w, x, y, z = quat / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    return (quat_to_rotation(quaternion_wxyz).T @ np.asarray((0.0, 0.0, -1.0))).astype(
        np.float32
    )


def torso_yaw_from_pelvis_and_waist(
    pelvis_quat_wxyz: np.ndarray, waist_yaw_roll_pitch: np.ndarray
) -> float:
    pelvis = quat_to_rotation(pelvis_quat_wxyz)
    yaw, roll, pitch = np.asarray(waist_yaw_roll_pitch, dtype=np.float64).reshape(3)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rz = np.asarray(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = np.asarray(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    torso = pelvis @ rz @ rx @ ry
    result = float(math.atan2(torso[1, 0], torso[0, 0]))
    if not math.isfinite(result):
        raise FloatingPointError("Torso yaw is invalid.")
    return result


@dataclass
class StartupHeadingLock:
    _direction_w: np.ndarray | None = None
    _initial_yaw: float | None = None

    @property
    def locked(self) -> bool:
        return self._direction_w is not None

    @property
    def initial_yaw(self) -> float | None:
        return self._initial_yaw

    def reset(self) -> None:
        self._direction_w = None
        self._initial_yaw = None

    def lock_yaw(self, yaw: float) -> float:
        if not math.isfinite(yaw):
            raise ValueError("Yaw must be finite.")
        self._direction_w = np.asarray((math.cos(yaw), math.sin(yaw)))
        self._initial_yaw = float(yaw)
        return float(yaw)

    def direction_b_from_yaw(self, yaw: float) -> np.ndarray:
        if self._direction_w is None:
            raise RuntimeError("Startup heading has not been locked.")
        c, s = math.cos(-yaw), math.sin(-yaw)
        x, y = self._direction_w
        result = np.asarray((c * x - s * y, s * x + c * y), dtype=np.float32)
        return result / max(float(np.linalg.norm(result)), 1.0e-6)


UNITREE_KEY_BIT = {
    "R1": 0,
    "L1": 1,
    "start": 2,
    "select": 3,
    "R2": 4,
    "L2": 5,
    "F1": 6,
    "F2": 7,
    "A": 8,
    "B": 9,
    "X": 10,
    "Y": 11,
    "up": 12,
    "right": 13,
    "down": 14,
    "left": 15,
}


def apply_deadzone(value: float, deadzone: float) -> float:
    if not math.isfinite(value) or not 0.0 <= deadzone < 1.0:
        raise ValueError("Axis/deadzone is invalid.")
    clipped = float(np.clip(value, -1.0, 1.0))
    if abs(clipped) <= deadzone:
        return 0.0
    return float(np.sign(clipped) * (abs(clipped) - deadzone) / (1.0 - deadzone))


_KP_5020, _KD_5020 = 14.25062309787429, 0.907222843292423
_KP_7520_14, _KD_7520_14 = 40.17923863450712, 2.557889775413375
_KP_7520_22, _KD_7520_22 = 99.09842777666111, 6.308801853496639
_KP_4010, _KD_4010 = 16.77832748089279, 1.06814150219
G1_KP = np.asarray(
    (
        _KP_7520_14, _KP_7520_22, _KP_7520_14, _KP_7520_22,
        2 * _KP_5020, 2 * _KP_5020, _KP_7520_14, _KP_7520_22,
        _KP_7520_14, _KP_7520_22, 2 * _KP_5020, 2 * _KP_5020,
        _KP_7520_14, 2 * _KP_5020, 2 * _KP_5020, _KP_5020, _KP_5020,
        _KP_5020, _KP_5020, _KP_5020, _KP_4010, _KP_4010, _KP_5020,
        _KP_5020, _KP_5020, _KP_5020, _KP_5020, _KP_4010, _KP_4010,
    ),
    dtype=np.float64,
)
G1_KD = np.asarray(
    (
        _KD_7520_14, _KD_7520_22, _KD_7520_14, _KD_7520_22,
        2 * _KD_5020, 2 * _KD_5020, _KD_7520_14, _KD_7520_22,
        _KD_7520_14, _KD_7520_22, 2 * _KD_5020, 2 * _KD_5020,
        _KD_7520_14, 2 * _KD_5020, 2 * _KD_5020, _KD_5020, _KD_5020,
        _KD_5020, _KD_5020, _KD_5020, _KD_4010, _KD_4010, _KD_5020,
        _KD_5020, _KD_5020, _KD_5020, _KD_5020, _KD_4010, _KD_4010,
    ),
    dtype=np.float64,
)


class PolicyState(Protocol):
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    quaternion_wxyz: np.ndarray
    angular_velocity: np.ndarray


def build_amp_observation(
    state: PolicyState,
    command: np.ndarray,
    previous_action: np.ndarray,
    default_pos: np.ndarray,
) -> np.ndarray:
    result = np.concatenate(
        (
            state.angular_velocity,
            projected_gravity(state.quaternion_wxyz),
            np.asarray(command, dtype=np.float32).reshape(3),
            state.joint_pos - default_pos,
            state.joint_vel,
            previous_action,
        )
    ).astype(np.float32)
    if result.shape != (AMP_OBS_DIM,) or not np.all(np.isfinite(result)):
        raise FloatingPointError("AMP observation is invalid.")
    return result


def build_real_observation(
    *,
    direction: np.ndarray,
    state: PolicyState,
    previous_action: np.ndarray,
    default_pos: np.ndarray,
) -> np.ndarray:
    result = np.concatenate(
        (
            np.asarray(direction, dtype=np.float32).reshape(2),
            projected_gravity(state.quaternion_wxyz),
            state.angular_velocity,
            state.joint_pos - default_pos,
            state.joint_vel,
            previous_action,
        )
    ).astype(np.float32)
    if result.shape != (REAL_OBS_DIM,) or not np.all(np.isfinite(result)):
        raise FloatingPointError("Real Parkour observation is invalid.")
    return result


def _providers(provider: str) -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Policy inference requires onnxruntime.") from exc
    available = ort.get_available_providers()
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDA ONNX Runtime is unavailable: {available}.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if provider == "auto" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if provider in ("auto", "cpu") and "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    raise ValueError("provider must be auto, cpu, or cuda")


def _metadata_array(metadata: dict[str, str], key: str, size: int) -> np.ndarray:
    try:
        result = np.asarray([float(v) for v in metadata[key].split(",") if v])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"ONNX metadata {key!r} is missing or invalid.") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"ONNX metadata {key!r} must contain {size} values.")
    return result.astype(np.float32)


@dataclass(frozen=True)
class AmpParameters:
    default_pos: np.ndarray
    action_scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


class AmpOnnxPolicy:
    def __init__(self, path: str | Path, provider: str = "cpu") -> None:
        import onnxruntime as ort

        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.path), sess_options=options, providers=_providers(provider)
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or tuple(inputs[0].shape) != (1, AMP_OBS_DIM):
            raise ValueError("AMP ONNX input must be [1,96].")
        if len(outputs) != 1 or tuple(outputs[0].shape) != (1, NUM_ACTIONS):
            raise ValueError("AMP ONNX output must be [1,29].")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        joints = tuple(filter(None, self.metadata.get("joint_names", "").split(",")))
        if joints != ACTUATED_JOINT_NAMES:
            raise ValueError("AMP ONNX joint order differs from Parkour sim2sim.")
        self.params = AmpParameters(
            _metadata_array(self.metadata, "default_joint_pos", NUM_ACTIONS),
            _metadata_array(self.metadata, "action_scale", NUM_ACTIONS),
            _metadata_array(self.metadata, "joint_stiffness", NUM_ACTIONS),
            _metadata_array(self.metadata, "joint_damping", NUM_ACTIONS),
        )

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32).reshape(1, AMP_OBS_DIM)
        action = self.session.run([self.output_name], {self.input_name: value})[0]
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (1, NUM_ACTIONS) or not np.all(np.isfinite(action)):
            raise FloatingPointError("AMP ONNX returned an invalid action.")
        return action[0].copy()


class RealParkourOnnxPolicy:
    def __init__(self, path: str | Path, provider: str = "cpu") -> None:
        import onnxruntime as ort

        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.path), sess_options=options, providers=_providers(provider)
        )
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        if self.metadata.get("task") != TASK_ID_V6:
            raise ValueError("Parkour sim2sim requires the V6 Real Parkour ONNX.")
        inputs = {item.name: tuple(item.shape) for item in self.session.get_inputs()}
        expected = {
            "obs": (1, REAL_OBS_DIM),
            "student_depth": V6_DEPTH_CONTRACT.tensor_shape,
        }
        if inputs != expected:
            raise ValueError(f"Real ONNX inputs are {inputs}; expected {expected}.")
        outputs = {item.name: tuple(item.shape) for item in self.session.get_outputs()}
        if outputs != {"actions": (1, NUM_ACTIONS)}:
            raise ValueError("Real ONNX output must be actions[1,29].")
        joints = tuple(filter(None, self.metadata.get("joint_names", "").split(",")))
        if joints != ACTUATED_JOINT_NAMES:
            raise ValueError("Real ONNX joint order differs from Parkour sim2sim.")
        if self.metadata.get("observation_layout") != OBSERVATION_LAYOUT:
            raise ValueError("Real ONNX observation layout is not the V6 layout.")
        self.default_pos = _metadata_array(
            self.metadata, "default_joint_pos", NUM_ACTIONS
        )
        scale_values = [
            float(v) for v in self.metadata.get("action_scale", "").split(",") if v
        ]
        if len(scale_values) == 1:
            scale_values *= NUM_ACTIONS
        self.action_scale = np.asarray(scale_values, dtype=np.float32)
        if self.action_scale.shape != (NUM_ACTIONS,):
            raise ValueError("Real ONNX action_scale must contain 1 or 29 values.")
        camera_pos = _metadata_array(self.metadata, "camera_pos", 3)
        camera_quat = _metadata_array(self.metadata, "camera_quat_wxyz", 4)
        if not np.allclose(camera_pos, CAMERA_POS, atol=1.0e-6) or not np.allclose(
            camera_quat, CAMERA_QUAT_WXYZ, atol=1.0e-6
        ):
            raise ValueError("Real ONNX camera pose differs from Parkour sim2sim.")

    def __call__(self, observation: np.ndarray, depth: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, REAL_OBS_DIM)
        depth_value = np.asarray(depth, dtype=np.float32).reshape(
            V6_DEPTH_CONTRACT.tensor_shape
        )
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(depth_value)):
            raise FloatingPointError("Real policy input is invalid.")
        action = self.session.run(
            ["actions"], {"obs": obs, "student_depth": depth_value}
        )[0]
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (1, NUM_ACTIONS) or not np.all(np.isfinite(action)):
            raise FloatingPointError("Real ONNX returned an invalid action.")
        return action[0].copy()

    def joint_target(self, action: np.ndarray) -> np.ndarray:
        return self.default_pos + self.action_scale * np.asarray(action, dtype=np.float32)


__all__ = [
    "ACTUATED_JOINT_NAMES",
    "AMP_OBS_DIM",
    "AmpOnnxPolicy",
    "CAMERA_POS",
    "CAMERA_QUAT_WXYZ",
    "CONTROL_DT",
    "DEPTH_FOVY_DEG",
    "DEPTH_RENDER_HEIGHT",
    "DEPTH_RENDER_WIDTH",
    "DepthHistory",
    "DepthSnapshot",
    "G1_KD",
    "G1_KP",
    "NUM_ACTIONS",
    "PHYSICS_DT",
    "REAL_OBS_DIM",
    "RealParkourOnnxPolicy",
    "StartupHeadingLock",
    "UNITREE_KEY_BIT",
    "V6_DEPTH_CONTRACT",
    "apply_deadzone",
    "build_amp_observation",
    "build_real_observation",
    "preprocess_v6_depth",
    "projected_gravity",
    "quat_to_rotation",
    "torso_yaw_from_pelvis_and_waist",
]
