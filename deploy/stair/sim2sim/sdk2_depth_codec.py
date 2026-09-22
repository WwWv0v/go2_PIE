"""Shared Unitree SDK2 transport contract for G1 Stair sim-to-sim."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from types import MappingProxyType
from typing import Any

import numpy as np


# Unitree SDK2 topics.  Domain 0 is intentionally forbidden below because it
# is the real-robot default domain.
TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_STAIR_DEPTH = "rt/camera/stair_depth"
DEPTH_FRAME_ID = "stair_depth_optical"


# Camera/policy contract from Unitree-G1-Stair.
DEPTH_HEIGHT = 48
DEPTH_WIDTH = 64
DEPTH_FOVY_DEG = 58.0
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 3.0

_FLOAT32_BYTES = np.dtype("<f4").itemsize
_POINT_FIELD_FLOAT32 = 7  # sensor_msgs/PointField.FLOAT32
_DEPTH_NUM_PIXELS = DEPTH_HEIGHT * DEPTH_WIDTH
_DEPTH_ROW_STEP = DEPTH_WIDTH * _FLOAT32_BYTES
_DEPTH_PAYLOAD_BYTES = _DEPTH_NUM_PIXELS * _FLOAT32_BYTES


# G1 29-DoF DDS motor order.  Tuple position is the motor_cmd/motor_state ID.
G1_DDS_JOINT_NAMES = (
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
G1_DDS_NUM_MOTORS = len(G1_DDS_JOINT_NAMES)
G1_DDS_MOTOR_INDEX = MappingProxyType(
  {name: index for index, name in enumerate(G1_DDS_JOINT_NAMES)}
)
G1_LEG_DDS_INDICES = tuple(range(12))
G1_WAIST_DDS_INDICES = (12, 13, 14)
G1_ARM_DDS_INDICES = tuple(range(15, G1_DDS_NUM_MOTORS))


# Exact gains used by the native Stair runner.  These originate from
# src/assets/robots/unitree_g1/g1_constants.py; metadata-rounded values are not
# used because they measurably change the closed-loop plant.
_KP_5020 = 14.25062309787429
_KD_5020 = 0.907222843292423
_KP_7520_14 = 40.17923863450712
_KD_7520_14 = 2.557889775413375
_KP_7520_22 = 99.09842777666111
_KD_7520_22 = 6.308801853496639
_KP_4010 = 16.77832748089279
_KD_4010 = 1.06814150219


def _readonly_array(values: Sequence[float], dtype: np.dtype[Any]) -> np.ndarray:
  array = np.asarray(values, dtype=dtype)
  array.setflags(write=False)
  return array


G1_HOME = _readonly_array(
  (
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
    0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
  ),
  np.dtype(np.float32),
)
G1_KP = _readonly_array(
  (
    _KP_7520_14, _KP_7520_22, _KP_7520_14, _KP_7520_22,
    2.0 * _KP_5020, 2.0 * _KP_5020,
    _KP_7520_14, _KP_7520_22, _KP_7520_14, _KP_7520_22,
    2.0 * _KP_5020, 2.0 * _KP_5020,
    _KP_7520_14, 2.0 * _KP_5020, 2.0 * _KP_5020,
    _KP_5020, _KP_5020, _KP_5020, _KP_5020, _KP_5020,
    _KP_4010, _KP_4010,
    _KP_5020, _KP_5020, _KP_5020, _KP_5020, _KP_5020,
    _KP_4010, _KP_4010,
  ),
  np.dtype(np.float64),
)
G1_KD = _readonly_array(
  (
    _KD_7520_14, _KD_7520_22, _KD_7520_14, _KD_7520_22,
    2.0 * _KD_5020, 2.0 * _KD_5020,
    _KD_7520_14, _KD_7520_22, _KD_7520_14, _KD_7520_22,
    2.0 * _KD_5020, 2.0 * _KD_5020,
    _KD_7520_14, 2.0 * _KD_5020, 2.0 * _KD_5020,
    _KD_5020, _KD_5020, _KD_5020, _KD_5020, _KD_5020,
    _KD_4010, _KD_4010,
    _KD_5020, _KD_5020, _KD_5020, _KD_5020, _KD_5020,
    _KD_4010, _KD_4010,
  ),
  np.dtype(np.float64),
)
G1_EFFORT_LIMIT = _readonly_array(
  (
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 50.0, 50.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
  ),
  np.dtype(np.float64),
)

# Short aliases are convenient when constructing a LowCmd message.
HOME = G1_HOME
KP = G1_KP
KD = G1_KD
EFFORT_LIMIT = G1_EFFORT_LIMIT


def validate_sim_transport(domain_id: int, interface: str) -> None:
  """Reject transport settings that could reach a physical robot.

  Unitree robots normally use DDS domain 0.  This sim-only adapter therefore
  requires a nonzero valid CycloneDDS domain and the loopback interface.
  """
  if isinstance(domain_id, bool) or not isinstance(domain_id, (int, np.integer)):
    raise TypeError("DDS domain_id must be an integer in [1, 232].")
  if domain_id == 0:
    raise ValueError(
      "DDS domain 0 is reserved for the physical robot; choose a sim domain."
    )
  if not 1 <= int(domain_id) <= 232:
    raise ValueError("DDS domain_id must be in [1, 232].")
  if interface != "lo":
    raise ValueError(
      "SDK2 sim-to-sim is restricted to the loopback interface 'lo'."
    )


def _validate_depth_z(depth_z: np.ndarray, *, source: str) -> np.ndarray:
  depth = np.asarray(depth_z)
  if depth.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
    raise ValueError(
      f"{source} depth shape is {depth.shape}; expected "
      f"({DEPTH_HEIGHT}, {DEPTH_WIDTH})."
    )
  if not np.issubdtype(depth.dtype, np.number):
    raise TypeError(f"{source} depth must be numeric, got {depth.dtype}.")
  depth = np.asarray(depth, dtype=np.float32)
  if not np.all(np.isfinite(depth)):
    raise ValueError(f"{source} depth contains NaN or infinity.")
  if np.any(depth < 0.0):
    raise ValueError(f"{source} optical-Z depth must be nonnegative (0 is miss).")
  return np.ascontiguousarray(depth)


def _sim_time_to_stamp(sim_time_s: float) -> tuple[int, int]:
  try:
    sim_time = float(sim_time_s)
  except (TypeError, ValueError) as exc:
    raise TypeError("sim_time_s must be a finite nonnegative number.") from exc
  if not math.isfinite(sim_time) or sim_time < 0.0:
    raise ValueError("sim_time_s must be finite and nonnegative.")

  seconds = math.floor(sim_time)
  nanoseconds = int(round((sim_time - seconds) * 1_000_000_000))
  if nanoseconds == 1_000_000_000:
    seconds += 1
    nanoseconds = 0
  if seconds > np.iinfo(np.int32).max:
    raise ValueError("sim_time_s exceeds the SDK2 Time_ int32 seconds range.")
  return int(seconds), nanoseconds


@lru_cache(maxsize=1)
def _sdk2_message_types() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
  try:
    from unitree_sdk2py.idl.builtin_interfaces.msg.dds_ import Time_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_, PointField_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_
  except ImportError as exc:
    raise RuntimeError(
      "encode_depth requires unitree_sdk2py in the active environment."
    ) from exc
  return PointCloud2_, PointField_, Header_, Time_


def encode_depth(
  depth_z: np.ndarray,
  sim_time_s: float,
  frame_id: str = DEPTH_FRAME_ID,
) -> Any:
  """Encode one 48x64 optical-Z frame into a strict SDK2 PointCloud2_ message.

  ``sim_time_s`` must be MuJoCo simulation time (``mj_data.time``), not wall
  time.  Miss pixels must already be represented by zero before this call.
  """
  if frame_id != DEPTH_FRAME_ID:
    raise ValueError(f"frame_id must be {DEPTH_FRAME_ID!r}, got {frame_id!r}.")
  depth = _validate_depth_z(depth_z, source="Encoded")
  seconds, nanoseconds = _sim_time_to_stamp(sim_time_s)
  PointCloud2_, PointField_, Header_, Time_ = _sdk2_message_types()

  payload = np.asarray(depth, dtype="<f4", order="C").tobytes(order="C")
  if len(payload) != _DEPTH_PAYLOAD_BYTES:  # Defensive invariant.
    raise AssertionError("Internal depth payload size mismatch.")

  return PointCloud2_(
    header=Header_(
      stamp=Time_(sec=seconds, nanosec=nanoseconds),
      frame_id=frame_id,
    ),
    height=DEPTH_HEIGHT,
    width=DEPTH_WIDTH,
    fields=[
      PointField_(
        name="z",
        offset=0,
        datatype=_POINT_FIELD_FLOAT32,
        count=1,
      )
    ],
    is_bigendian=False,
    point_step=_FLOAT32_BYTES,
    row_step=_DEPTH_ROW_STEP,
    data=list(payload),
    is_dense=False,
  )


def _message_attr(message: Any, name: str) -> Any:
  if not hasattr(message, name):
    raise ValueError(f"Depth message is missing required attribute {name!r}.")
  return getattr(message, name)


def _decode_sim_time(message: Any) -> float:
  header = _message_attr(message, "header")
  frame_id = _message_attr(header, "frame_id")
  if frame_id != DEPTH_FRAME_ID:
    raise ValueError(
      f"Depth frame_id is {frame_id!r}; expected {DEPTH_FRAME_ID!r}."
    )
  stamp = _message_attr(header, "stamp")
  seconds = _message_attr(stamp, "sec")
  nanoseconds = _message_attr(stamp, "nanosec")
  if isinstance(seconds, bool) or not isinstance(seconds, (int, np.integer)):
    raise ValueError("Depth timestamp seconds must be an integer.")
  if isinstance(nanoseconds, bool) or not isinstance(
    nanoseconds, (int, np.integer)
  ):
    raise ValueError("Depth timestamp nanoseconds must be an integer.")
  if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
    raise ValueError("Depth timestamp must be finite, normalized simulation time.")
  return float(seconds) + float(nanoseconds) * 1.0e-9


def decode_depth(message: Any) -> tuple[np.ndarray, float]:
  """Strictly decode a Stair optical-Z PointCloud2_ message.

  Returns ``(depth_z, sim_time_s)``.  The returned image is an owned,
  C-contiguous float32 array with shape ``(48, 64)``.
  """
  if _message_attr(message, "height") != DEPTH_HEIGHT:
    raise ValueError(f"Depth message height must be {DEPTH_HEIGHT}.")
  if _message_attr(message, "width") != DEPTH_WIDTH:
    raise ValueError(f"Depth message width must be {DEPTH_WIDTH}.")
  if bool(_message_attr(message, "is_bigendian")):
    raise ValueError("Depth message must be little-endian.")
  if _message_attr(message, "point_step") != _FLOAT32_BYTES:
    raise ValueError(f"Depth point_step must be {_FLOAT32_BYTES} bytes.")
  if _message_attr(message, "row_step") != _DEPTH_ROW_STEP:
    raise ValueError(f"Depth row_step must be {_DEPTH_ROW_STEP} bytes.")

  fields = _message_attr(message, "fields")
  if not isinstance(fields, Sequence) or len(fields) != 1:
    raise ValueError("Depth message must contain exactly one PointField.")
  field = fields[0]
  field_contract = {
    "name": "z",
    "offset": 0,
    "datatype": _POINT_FIELD_FLOAT32,
    "count": 1,
  }
  for attribute, expected in field_contract.items():
    actual = _message_attr(field, attribute)
    if actual != expected:
      raise ValueError(
        f"Depth field {attribute} is {actual!r}; expected {expected!r}."
      )

  try:
    payload = bytes(_message_attr(message, "data"))
  except (TypeError, ValueError) as exc:
    raise ValueError("Depth data must be a uint8 byte sequence.") from exc
  if len(payload) != _DEPTH_PAYLOAD_BYTES:
    raise ValueError(
      f"Depth payload is {len(payload)} bytes; expected {_DEPTH_PAYLOAD_BYTES}."
    )

  depth = np.frombuffer(payload, dtype="<f4", count=_DEPTH_NUM_PIXELS)
  depth = depth.reshape(DEPTH_HEIGHT, DEPTH_WIDTH).astype(np.float32, copy=True)
  depth = _validate_depth_z(depth, source="Decoded")
  sim_time_s = _decode_sim_time(message)
  return depth, sim_time_s


_FOCAL_PIXELS = 0.5 * DEPTH_HEIGHT / math.tan(
  0.5 * math.radians(DEPTH_FOVY_DEG)
)
_PIXEL_X = (
  np.arange(DEPTH_WIDTH, dtype=np.float32) + 0.5 - 0.5 * DEPTH_WIDTH
) / _FOCAL_PIXELS
_PIXEL_Y = (
  np.arange(DEPTH_HEIGHT, dtype=np.float32) + 0.5 - 0.5 * DEPTH_HEIGHT
) / _FOCAL_PIXELS
_Z_TO_RAY_SCALE = np.sqrt(
  1.0 + _PIXEL_Y[:, None] ** 2 + _PIXEL_X[None, :] ** 2
).astype(np.float32)
_Z_TO_RAY_SCALE.setflags(write=False)


def preprocess_depth_z(depth_z: np.ndarray) -> np.ndarray:
  """Convert optical-Z depth to the normalized Stair ONNX depth tensor.

  Optical-Z is converted to Euclidean normalized-ray distance using the same
  pixel-center convention as the native runner.  Zero misses become 0.05 m,
  then all values are clipped to [0.05, 3.0] m and divided by 3.0.
  """
  depth = _validate_depth_z(depth_z, source="Preprocessed")
  misses = depth == 0.0
  ray_depth = depth * _Z_TO_RAY_SCALE
  ray_depth[misses] = DEPTH_MIN_M
  normalized = np.clip(ray_depth, DEPTH_MIN_M, DEPTH_MAX_M) / DEPTH_MAX_M
  return np.ascontiguousarray(normalized[None, None, :, :], dtype=np.float32)


__all__ = [
  "DEPTH_FOVY_DEG",
  "DEPTH_FRAME_ID",
  "DEPTH_HEIGHT",
  "DEPTH_MAX_M",
  "DEPTH_MIN_M",
  "DEPTH_WIDTH",
  "EFFORT_LIMIT",
  "G1_ARM_DDS_INDICES",
  "G1_DDS_JOINT_NAMES",
  "G1_DDS_MOTOR_INDEX",
  "G1_DDS_NUM_MOTORS",
  "G1_EFFORT_LIMIT",
  "G1_HOME",
  "G1_KD",
  "G1_KP",
  "G1_LEG_DDS_INDICES",
  "G1_WAIST_DDS_INDICES",
  "HOME",
  "KD",
  "KP",
  "TOPIC_STAIR_DEPTH",
  "TOPIC_LOWCMD",
  "TOPIC_LOWSTATE",
  "decode_depth",
  "encode_depth",
  "preprocess_depth_z",
  "validate_sim_transport",
]
