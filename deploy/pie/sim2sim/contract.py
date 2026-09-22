"""Runtime constants and observation preprocessing for PIE sim-to-sim."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import numpy as np


CONTROL_DT = 0.02
PHYSICS_DT = 0.005
PHYSICS_STEPS_PER_CONTROL = 4
DEPTH_UPDATE_PERIOD_STEPS = 5

ACTUATED_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
NUM_ACTIONS = len(ACTUATED_JOINT_NAMES)

DEFAULT_JOINT_POS = np.asarray(
    (
        -0.1,
        0.9,
        -1.8,
        0.1,
        0.9,
        -1.8,
        -0.1,
        0.9,
        -1.8,
        0.1,
        0.9,
        -1.8,
    ),
    dtype=np.float32,
)
ACTION_SCALE = np.full(NUM_ACTIONS, 0.25, dtype=np.float32)

JOINT_STIFFNESS = np.asarray(
    tuple(value for _ in range(4) for value in (20.0, 20.0, 40.0)),
    dtype=np.float64,
)
JOINT_DAMPING = np.asarray(
    tuple(value for _ in range(4) for value in (1.0, 1.0, 2.0)),
    dtype=np.float64,
)
JOINT_EFFORT_LIMIT = np.asarray(
    tuple(value for _ in range(4) for value in (23.5, 23.5, 45.0)),
    dtype=np.float64,
)
JOINT_ARMATURE = np.asarray(
    tuple(value for _ in range(4) for value in (0.01, 0.01, 0.02)),
    dtype=np.float64,
)

OBSERVATION_NAMES = (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
)
OBSERVATION_TERM_DIMS = (3, 3, 3, NUM_ACTIONS, NUM_ACTIONS, NUM_ACTIONS)
OBS_DIM = sum(OBSERVATION_TERM_DIMS)
PROPRIO_HISTORY_LENGTH = 10
PROPRIO_HISTORY_DIM = OBS_DIM * PROPRIO_HISTORY_LENGTH

DEPTH_RAW_WIDTH = 106
DEPTH_HEIGHT = 60
DEPTH_CROP_LEFT = 10
DEPTH_CROP_RIGHT = 10
DEPTH_WIDTH = DEPTH_RAW_WIDTH - DEPTH_CROP_LEFT - DEPTH_CROP_RIGHT
DEPTH_HISTORY_LENGTH = 2
DEPTH_HISTORY_SHAPE = (DEPTH_HISTORY_LENGTH, DEPTH_HEIGHT, DEPTH_WIDTH)
DEPTH_HORIZONTAL_FOV_DEG = 87.0
DEPTH_FOVY_DEG = 2.0 * math.degrees(
    math.atan(
        math.tan(math.radians(DEPTH_HORIZONTAL_FOV_DEG) * 0.5)
        * DEPTH_HEIGHT
        / DEPTH_RAW_WIDTH
    )
)
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 3.0
DEPTH_CAMERA_POS = (0.345, 0.0, 0.07)
DEPTH_CAMERA_QUAT = (0.5792280, 0.4055798, -0.4055798, -0.5792280)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


for _constant_array in (
    DEFAULT_JOINT_POS,
    ACTION_SCALE,
    JOINT_STIFFNESS,
    JOINT_DAMPING,
    JOINT_EFFORT_LIMIT,
    JOINT_ARMATURE,
):
    _readonly(_constant_array)


def quaternion_wxyz_to_rotation(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """Return the body-to-world rotation for a normalized wxyz quaternion."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"Invalid quaternion: {quaternion}.")
    norm = float(np.linalg.vector_norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("Quaternion norm is zero.")
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def projected_gravity(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """Project world gravity direction into the Go2 base frame."""
    rotation = quaternion_wxyz_to_rotation(quaternion_wxyz)
    return np.asarray(rotation.T @ np.asarray((0.0, 0.0, -1.0)), dtype=np.float32)


def build_proprioception(
    base_ang_vel: Sequence[float],
    quaternion_wxyz: Sequence[float],
    command: Sequence[float],
    joint_pos: Sequence[float],
    joint_vel: Sequence[float],
    last_action: Sequence[float],
) -> np.ndarray:
    """Build the exact clean 45-D PIE actor observation."""
    values = (
        np.asarray(base_ang_vel, dtype=np.float32).reshape(3),
        projected_gravity(quaternion_wxyz),
        np.asarray(command, dtype=np.float32).reshape(3),
        np.asarray(joint_pos, dtype=np.float32).reshape(NUM_ACTIONS)
        - DEFAULT_JOINT_POS,
        np.asarray(joint_vel, dtype=np.float32).reshape(NUM_ACTIONS),
        np.asarray(last_action, dtype=np.float32).reshape(NUM_ACTIONS),
    )
    observation = np.concatenate(values).astype(np.float32, copy=False)
    if observation.shape != (OBS_DIM,) or not np.all(np.isfinite(observation)):
        raise FloatingPointError(
            f"Invalid PIE proprioception with shape {observation.shape}."
        )
    return np.ascontiguousarray(observation)


class ProprioHistory:
    """PIE's term-major, oldest-to-newest 10-step observation history."""

    def __init__(self, length: int = PROPRIO_HISTORY_LENGTH) -> None:
        if length <= 0:
            raise ValueError("Proprioception history length must be positive.")
        self.length = int(length)
        self._frames: deque[np.ndarray] = deque(maxlen=self.length)

    def reset(self) -> None:
        self._frames.clear()

    def append(self, observation: Sequence[float]) -> None:
        frame = np.asarray(observation, dtype=np.float32).reshape(OBS_DIM)
        if not np.all(np.isfinite(frame)):
            raise FloatingPointError("Cannot append non-finite proprioception.")
        if not self._frames:
            for _ in range(self.length):
                self._frames.append(frame.copy())
            return
        self._frames.append(frame.copy())

    @property
    def array(self) -> np.ndarray:
        if len(self._frames) != self.length:
            raise RuntimeError("Append one observation before reading history.")
        frames = np.stack(tuple(self._frames), axis=0)
        # MJLab stores a separate CircularBuffer for each observation term and
        # concatenates those flattened buffers.  This is deliberately not the
        # simpler frames.reshape(-1) ordering.
        chunks: list[np.ndarray] = []
        start = 0
        for dim in OBSERVATION_TERM_DIMS:
            chunks.append(frames[:, start : start + dim].reshape(-1))
            start += dim
        history = np.concatenate(chunks).astype(np.float32, copy=False)
        if history.shape != (PROPRIO_HISTORY_DIM,):
            raise AssertionError("Internal proprioception history shape mismatch.")
        return np.ascontiguousarray(history)


def _depth_ray_scale() -> np.ndarray:
    focal_pixels = 0.5 * DEPTH_HEIGHT / math.tan(0.5 * math.radians(DEPTH_FOVY_DEG))
    pixel_x = (
        np.arange(DEPTH_RAW_WIDTH, dtype=np.float32) + 0.5 - 0.5 * DEPTH_RAW_WIDTH
    ) / focal_pixels
    pixel_y = (
        np.arange(DEPTH_HEIGHT, dtype=np.float32) + 0.5 - 0.5 * DEPTH_HEIGHT
    ) / focal_pixels
    return np.sqrt(1.0 + pixel_y[:, None] ** 2 + pixel_x[None, :] ** 2).astype(
        np.float32
    )


_Z_TO_RAY_SCALE = _readonly(_depth_ray_scale())
_GAUSSIAN_KERNEL = np.asarray(
    (
        (0.07511361, 0.12384140, 0.07511361),
        (0.12384140, 0.20417996, 0.12384140),
        (0.07511361, 0.12384140, 0.07511361),
    ),
    dtype=np.float32,
)
_GAUSSIAN_KERNEL /= _GAUSSIAN_KERNEL.sum()
_readonly(_GAUSSIAN_KERNEL)


def preprocess_depth_z(depth_z: np.ndarray) -> np.ndarray:
    """Convert a native MuJoCo optical-Z image to one PIE depth frame.

    MJLab's camera renderer reports Euclidean distance along each normalized
    pixel ray, while native ``mujoco.Renderer`` reports optical-Z.  The
    conversion below matches the training camera before applying its 10-pixel
    side crop, 3x3 sigma=1 Gaussian blur, clipping, and normalization.
    """
    depth = np.asarray(depth_z, dtype=np.float32)
    if depth.shape != (DEPTH_HEIGHT, DEPTH_RAW_WIDTH):
        raise ValueError(
            f"Raw depth shape is {depth.shape}; expected "
            f"({DEPTH_HEIGHT}, {DEPTH_RAW_WIDTH})."
        )
    invalid = ~np.isfinite(depth) | (depth <= 0.0)
    ray_depth = depth * _Z_TO_RAY_SCALE
    ray_depth[invalid] = DEPTH_MAX_M
    ray_depth = ray_depth[:, DEPTH_CROP_LEFT : DEPTH_RAW_WIDTH - DEPTH_CROP_RIGHT]

    padded = np.pad(ray_depth, ((1, 1), (1, 1)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    blurred = np.einsum("ijxy,xy->ij", windows, _GAUSSIAN_KERNEL, optimize=True)
    normalized = np.clip(blurred, DEPTH_MIN_M, DEPTH_MAX_M) / DEPTH_MAX_M
    result = np.ascontiguousarray(normalized[None, :, :], dtype=np.float32)
    if result.shape != (1, DEPTH_HEIGHT, DEPTH_WIDTH):
        raise AssertionError("Internal PIE depth shape mismatch.")
    return result


class DepthHistory:
    """Two 10-Hz PIE depth frames, oldest first."""

    def __init__(self, length: int = DEPTH_HISTORY_LENGTH) -> None:
        if length <= 0:
            raise ValueError("Depth history length must be positive.")
        self.length = int(length)
        self._frames: deque[np.ndarray] = deque(maxlen=self.length)

    def reset(self) -> None:
        self._frames.clear()

    def append(self, frame: np.ndarray) -> None:
        value = np.asarray(frame, dtype=np.float32).reshape(
            1, DEPTH_HEIGHT, DEPTH_WIDTH
        )
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("Cannot append non-finite depth.")
        if not self._frames:
            for _ in range(self.length):
                self._frames.append(value.copy())
            return
        self._frames.append(value.copy())

    @property
    def array(self) -> np.ndarray:
        if len(self._frames) != self.length:
            raise RuntimeError("Append one depth frame before reading history.")
        result = np.concatenate(tuple(self._frames), axis=0)
        return np.ascontiguousarray(result, dtype=np.float32)


__all__ = [
    "ACTION_SCALE",
    "ACTUATED_JOINT_NAMES",
    "CONTROL_DT",
    "DEFAULT_JOINT_POS",
    "DEPTH_CAMERA_POS",
    "DEPTH_CAMERA_QUAT",
    "DEPTH_CROP_LEFT",
    "DEPTH_CROP_RIGHT",
    "DEPTH_FOVY_DEG",
    "DEPTH_HEIGHT",
    "DEPTH_HISTORY_LENGTH",
    "DEPTH_HISTORY_SHAPE",
    "DEPTH_MAX_M",
    "DEPTH_MIN_M",
    "DEPTH_RAW_WIDTH",
    "DEPTH_UPDATE_PERIOD_STEPS",
    "DEPTH_WIDTH",
    "DepthHistory",
    "JOINT_ARMATURE",
    "JOINT_DAMPING",
    "JOINT_EFFORT_LIMIT",
    "JOINT_STIFFNESS",
    "NUM_ACTIONS",
    "OBSERVATION_NAMES",
    "OBSERVATION_TERM_DIMS",
    "OBS_DIM",
    "PHYSICS_DT",
    "PHYSICS_STEPS_PER_CONTROL",
    "PROPRIO_HISTORY_DIM",
    "PROPRIO_HISTORY_LENGTH",
    "ProprioHistory",
    "build_proprioception",
    "preprocess_depth_z",
    "projected_gravity",
    "quaternion_wxyz_to_rotation",
]
