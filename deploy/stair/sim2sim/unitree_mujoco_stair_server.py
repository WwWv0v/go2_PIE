#!/usr/bin/env python3
"""Unitree SDK2 MuJoCo server for the G1 Stair controller."""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep renderer caches out of potentially read-only home directories.  EGL
# must be selected before MuJoCo is imported, but validate-only intentionally
# creates no GL context at all.
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ame_mjlab_cache")
if (
  "--headless" in sys.argv
  and "--validate-only" not in sys.argv
  and "MUJOCO_GL" not in os.environ
):
  os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from deploy.stair.sim2sim.sdk2_depth_codec import (  # noqa: E402
  DEPTH_FOVY_DEG,
  DEPTH_HEIGHT,
  DEPTH_MAX_M,
  DEPTH_MIN_M,
  DEPTH_WIDTH,
  G1_DDS_JOINT_NAMES,
  G1_EFFORT_LIMIT,
  G1_HOME,
  G1_KD,
  G1_KP,
  TOPIC_STAIR_DEPTH,
  TOPIC_LOWCMD,
  TOPIC_LOWSTATE,
  encode_depth,
  validate_sim_transport,
)


PHYSICS_DT = 0.005
DEPTH_DECIMATION = 4
DEPTH_DT = PHYSICS_DT * DEPTH_DECIMATION
DEPTH_RENDER_NEAR_M = 0.001
DEPTH_BACKGROUND_FAR_FRACTION = 0.99
DEPTH_CAMERA_POS = (0.10, 0.0, 0.10)
DEPTH_CAMERA_QUAT = (0.9396926, 0.0, -0.3420201, 0.0)
DEPTH_CAMERA_VISUAL_SIZE = (0.018, 0.045, 0.012)
DEPTH_CAMERA_VISUAL_POS = (0.091644, 0.0, 0.109959)
G1_NUM_MOTORS = 29
G1_MODE_MACHINE = 5

DEFAULT_SCENE_FILE = (
  Path(__file__).resolve().parent
  / "assets/unitree_g1/scene_stair.xml"
)

# Exact sole collision geometry from src/assets/robots/unitree_g1/xmls/g1.xml.
# CollisionCfg in g1_constants.py changes the inherited condim from 6 to 3 and
# gives every foot capsule a nominal sliding friction of 0.6.
TRAINING_FOOT_CAPSULE_FROMTO = {
  "left": (
    (0.1, -0.026, -0.025, 0.05, -0.027, -0.025),
    (-0.044, -0.018, -0.025, 0.123, -0.018, -0.025),
    (-0.052, -0.01, -0.025, 0.13, -0.01, -0.025),
    (-0.054, 0.0, -0.025, 0.132, 0.0, -0.025),
    (-0.052, 0.01, -0.025, 0.13, 0.01, -0.025),
    (-0.044, 0.018, -0.025, 0.123, 0.018, -0.025),
    (0.1, 0.026, -0.025, 0.05, 0.026, -0.025),
  ),
  "right": (
    (0.1, -0.026, -0.025, 0.05, -0.026, -0.025),
    (-0.044, -0.018, -0.025, 0.123, -0.018, -0.025),
    (-0.052, -0.01, -0.025, 0.13, -0.01, -0.025),
    (-0.054, 0.0, -0.025, 0.132, 0.0, -0.025),
    (-0.052, 0.01, -0.025, 0.13, 0.01, -0.025),
    (-0.044, 0.018, -0.025, 0.123, 0.018, -0.025),
    (0.1, 0.026, -0.025, 0.05, 0.026, -0.025),
  ),
}
TRAINING_FOOT_CAPSULE_RADIUS = 0.01
TRAINING_FOOT_FRICTION = (0.6, 0.005, 0.0001)
TRAINING_FOOT_GEOM_COUNT = 14

# Actuator names in the official Unitree model.  The mapping below also checks
# every transmission target, so neither XML order nor DDS order is assumed.
G1_ACTUATOR_NAMES = (
  "left_hip_pitch",
  "left_hip_roll",
  "left_hip_yaw",
  "left_knee",
  "left_ankle_pitch",
  "left_ankle_roll",
  "right_hip_pitch",
  "right_hip_roll",
  "right_hip_yaw",
  "right_knee",
  "right_ankle_pitch",
  "right_ankle_roll",
  "waist_yaw",
  "waist_roll",
  "waist_pitch",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_shoulder_yaw",
  "left_elbow",
  "left_wrist_roll",
  "left_wrist_pitch",
  "left_wrist_yaw",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_shoulder_yaw",
  "right_elbow",
  "right_wrist_roll",
  "right_wrist_pitch",
  "right_wrist_yaw",
)

@dataclass(frozen=True)
class ModelBindings:
  dds_motor_ids: np.ndarray
  joint_ids: np.ndarray
  joint_qpos_adr: np.ndarray
  joint_dof_adr: np.ndarray
  actuator_ids: np.ndarray
  actuator_ctrl_min: np.ndarray
  actuator_ctrl_max: np.ndarray
  root_qpos_adr: int
  root_dof_adr: int
  camera_id: int
  imu_quat_adr: int
  imu_gyro_adr: int
  imu_acc_adr: int


@dataclass(frozen=True)
class CommandSnapshot:
  sequence: int
  received_at: float
  q: np.ndarray
  dq: np.ndarray
  tau: np.ndarray
  kp: np.ndarray
  kd: np.ndarray


@dataclass
class TransportStats:
  accepted_commands: int = 0
  bad_crc_commands: int = 0
  invalid_commands: int = 0
  lowstate_publish_failures: int = 0
  depth_publish_failures: int = 0


def _name_to_id(
  model: mujoco.MjModel,
  object_type: mujoco.mjtObj,
  name: str,
) -> int:
  object_id = mujoco.mj_name2id(model, object_type, name)
  if object_id < 0:
    raise ValueError(f"MuJoCo model is missing {object_type.name} {name!r}.")
  return int(object_id)


def _sensor_address(
  model: mujoco.MjModel,
  name: str,
  expected_dim: int,
) -> int:
  sensor_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
  actual_dim = int(model.sensor_dim[sensor_id])
  if actual_dim != expected_dim:
    raise ValueError(
      f"Sensor {name!r} has dimension {actual_dim}; expected {expected_dim}."
    )
  return int(model.sensor_adr[sensor_id])


def _install_training_foot_collisions(
  spec: mujoco.MjSpec,
  scene_file: Path,
) -> None:
  """Replace each scene foot's contacts with the Stair training capsules."""
  for side, segments in TRAINING_FOOT_CAPSULE_FROMTO.items():
    body_name = f"{side}_ankle_roll_link"
    foot_body = spec.body(body_name)
    if foot_body is None:
      raise ValueError(
        f"Unitree scene {scene_file} is not a compatible G1 scene: "
        f"{body_name} is missing."
      )

    # The official Unitree model has four unnamed sphere contacts here.  Remove
    # every contact-enabled geom on the sole so an already-modified scene also
    # converges to exactly one copy of the training collision representation.
    for geom in list(foot_body.geoms):
      if int(geom.contype) != 0 or int(geom.conaffinity) != 0:
        spec.delete(geom)

    for index, fromto in enumerate(segments, start=1):
      foot_body.add_geom(
        name=f"{side}_foot{index}_collision",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=fromto,
        size=(TRAINING_FOOT_CAPSULE_RADIUS, 0.0, 0.0),
        contype=1,
        conaffinity=1,
        condim=3,
        priority=1,
        friction=TRAINING_FOOT_FRICTION,
        group=3,
      )


def _validate_training_foot_collisions(model: mujoco.MjModel) -> None:
  """Guard the compiled sim2sim sole-contact contract against drift."""
  expected_names: set[str] = set()
  for side, segments in TRAINING_FOOT_CAPSULE_FROMTO.items():
    body_id = _name_to_id(
      model,
      mujoco.mjtObj.mjOBJ_BODY,
      f"{side}_ankle_roll_link",
    )
    expected_side_names = {
      f"{side}_foot{index}_collision"
      for index in range(1, len(segments) + 1)
    }
    expected_names.update(expected_side_names)
    active_geom_ids = np.flatnonzero(
      (model.geom_bodyid == body_id)
      & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    active_names = {
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
      for geom_id in active_geom_ids
    }
    if active_names != expected_side_names:
      raise ValueError(
        f"{side} foot contact geoms are {sorted(active_names)}; expected "
        f"{sorted(expected_side_names)}."
      )

  if len(expected_names) != TRAINING_FOOT_GEOM_COUNT:
    raise AssertionError("Stair training foot-collision table is not 14 geoms.")

  geom_ids = np.asarray(
    [
      _name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
      for name in sorted(expected_names)
    ],
    dtype=np.int32,
  )
  expected_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
  contract_matches = (
    np.all(model.geom_type[geom_ids] == expected_type)
    and np.allclose(
      model.geom_size[geom_ids, 0],
      TRAINING_FOOT_CAPSULE_RADIUS,
      atol=1.0e-12,
    )
    and np.all(model.geom_contype[geom_ids] == 1)
    and np.all(model.geom_conaffinity[geom_ids] == 1)
    and np.all(model.geom_condim[geom_ids] == 3)
    and np.all(model.geom_priority[geom_ids] == 1)
    and np.all(model.geom_group[geom_ids] == 3)
    and np.allclose(
      model.geom_friction[geom_ids],
      TRAINING_FOOT_FRICTION,
      atol=1.0e-12,
    )
  )
  if not contract_matches:
    raise ValueError(
      "Compiled foot collisions do not match the Stair training contract."
    )


def _install_training_depth_camera_visual(
  spec: mujoco.MjSpec,
  torso: Any,
) -> None:
  """Install the small camera shell from the Stair training robot."""
  existing_visual = spec.geom("front_depth_camera_visual")
  if existing_visual is not None:
    spec.delete(existing_visual)
  torso.add_geom(
    name="front_depth_camera_visual",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=DEPTH_CAMERA_VISUAL_SIZE,
    pos=DEPTH_CAMERA_VISUAL_POS,
    quat=DEPTH_CAMERA_QUAT,
    contype=0,
    conaffinity=0,
    group=2,
    density=0.0,
    rgba=(0.2, 0.2, 0.2, 1.0),
  )


def _validate_training_depth_camera(model: mujoco.MjModel) -> None:
  """Validate the functional camera and its non-colliding visual shell."""
  torso_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
  camera_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_depth")
  visual_id = _name_to_id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "front_depth_camera_visual",
  )
  matches = (
    int(model.geom_bodyid[visual_id]) == torso_id
    and int(model.geom_type[visual_id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
    and np.allclose(
      model.geom_size[visual_id], DEPTH_CAMERA_VISUAL_SIZE, atol=1.0e-12
    )
    and np.allclose(
      model.geom_pos[visual_id], DEPTH_CAMERA_VISUAL_POS, atol=1.0e-12
    )
    and np.allclose(
      model.geom_quat[visual_id], DEPTH_CAMERA_QUAT, atol=1.0e-7
    )
    and int(model.geom_contype[visual_id]) == 0
    and int(model.geom_conaffinity[visual_id]) == 0
    and int(model.geom_group[visual_id]) == 2
    and np.allclose(model.cam_pos[camera_id], DEPTH_CAMERA_POS, atol=1.0e-12)
    and np.allclose(model.cam_quat[camera_id], DEPTH_CAMERA_QUAT, atol=1.0e-7)
    and math.isclose(
      float(model.cam_fovy[camera_id]), DEPTH_FOVY_DEG, abs_tol=1.0e-12
    )
  )
  if not matches:
    raise ValueError(
      "Compiled depth camera or visual shell does not match Stair training."
    )


def build_model(
  scene_file: Path,
) -> tuple[mujoco.MjModel, ModelBindings, int]:
  """Compile a full-body Unitree G1 scene with the Stair camera."""
  if not scene_file.is_file():
    raise FileNotFoundError(f"Unitree MuJoCo scene XML not found: {scene_file}")
  spec = mujoco.MjSpec.from_file(str(scene_file))

  _install_training_foot_collisions(spec, scene_file)

  torso = spec.body("torso_link")
  if torso is None:
    raise ValueError(
      f"Unitree scene {scene_file} is not a compatible G1 scene: "
      "torso_link is missing."
    )
  if spec.camera("front_depth") is not None:
    raise ValueError(
      "The Unitree scene already defines a camera named 'front_depth'. "
      "Rename or remove it so the server can install the policy-aligned camera."
    )
  _install_training_depth_camera_visual(spec, torso)
  torso.add_camera(
    name="front_depth",
    pos=DEPTH_CAMERA_POS,
    quat=DEPTH_CAMERA_QUAT,
    fovy=DEPTH_FOVY_DEG,
  )
  # Preserve all scene terrain/assets plus the remaining official robot
  # dynamics, integrator, torque motors, joint limits, and gains.  Outside the
  # training-aligned feet/camera additions, only the 200 Hz timestep is changed.
  spec.option.timestep = PHYSICS_DT
  model = spec.compile()

  _validate_training_foot_collisions(model)
  _validate_training_depth_camera(model)

  expected_shape = (36, 35, G1_NUM_MOTORS)
  if (model.nq, model.nv, model.nu) != expected_shape:
    raise ValueError(
      "G1 29-DoF contract changed: "
      f"expected nq={expected_shape[0]}, nv={expected_shape[1]}, "
      f"nu={expected_shape[2]}; got {model.nq}, {model.nv}, {model.nu}."
    )
  if not math.isclose(float(model.opt.timestep), PHYSICS_DT, abs_tol=1.0e-12):
    raise ValueError(f"MuJoCo timestep is {model.opt.timestep}, expected {PHYSICS_DT}.")
  if len(G1_DDS_JOINT_NAMES) != G1_NUM_MOTORS:
    raise AssertionError("DDS joint table is not 29-DoF.")
  if G1_HOME.shape != (G1_NUM_MOTORS,):
    raise AssertionError("G1 home position contract is not 29-DoF.")
  if any(array.shape != (G1_NUM_MOTORS,) for array in (G1_KP, G1_KD)):
    raise AssertionError("G1 gain contract is not 29-DoF.")
  if G1_EFFORT_LIMIT.shape != (G1_NUM_MOTORS,):
    raise AssertionError("G1 effort contract is not 29-DoF.")

  dds_motor_ids = np.arange(G1_NUM_MOTORS, dtype=np.int32)
  num_active_motors = len(dds_motor_ids)
  joint_ids = np.empty(num_active_motors, dtype=np.int32)
  qpos_adr = np.empty(num_active_motors, dtype=np.int32)
  dof_adr = np.empty(num_active_motors, dtype=np.int32)
  actuator_ids = np.empty(num_active_motors, dtype=np.int32)
  for active_index, dds_id in enumerate(dds_motor_ids):
    joint_name = G1_DDS_JOINT_NAMES[dds_id]
    actuator_name = G1_ACTUATOR_NAMES[dds_id]
    joint_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    actuator_id = _name_to_id(
      model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
    )
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
      raise ValueError(f"DDS motor joint {joint_name!r} is not a hinge.")
    if int(model.actuator_trntype[actuator_id]) != int(
      mujoco.mjtTrn.mjTRN_JOINT
    ):
      raise ValueError(f"Actuator {actuator_name!r} is not a joint transmission.")
    transmitted_joint = int(model.actuator_trnid[actuator_id, 0])
    if transmitted_joint != joint_id:
      actual = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_JOINT, transmitted_joint
      )
      raise ValueError(
        f"Actuator {actuator_name!r} drives {actual!r}, not {joint_name!r}."
      )
    joint_ids[active_index] = joint_id
    qpos_adr[active_index] = int(model.jnt_qposadr[joint_id])
    dof_adr[active_index] = int(model.jnt_dofadr[joint_id])
    actuator_ids[active_index] = actuator_id

  if len(set(joint_ids.tolist())) != num_active_motors:
    raise ValueError("DDS joint-name mapping contains duplicate model joints.")
  if len(set(actuator_ids.tolist())) != num_active_motors:
    raise ValueError("DDS actuator-name mapping contains duplicate actuators.")
  if not np.all(model.actuator_ctrllimited[actuator_ids]):
    raise ValueError("Every official torque motor must have a control limit.")
  ctrl_ranges = np.asarray(model.actuator_ctrlrange[actuator_ids], dtype=np.float64)
  if not np.all(np.isfinite(ctrl_ranges)) or not np.all(
    ctrl_ranges[:, 0] < ctrl_ranges[:, 1]
  ):
    raise ValueError("Official actuator control ranges are invalid.")
  expected_gear = np.zeros((num_active_motors, 6), dtype=np.float64)
  expected_gear[:, 0] = 1.0
  if (
    np.any(model.actuator_dyntype[actuator_ids] != mujoco.mjtDyn.mjDYN_NONE)
    or np.any(
      model.actuator_gaintype[actuator_ids] != mujoco.mjtGain.mjGAIN_FIXED
    )
    or np.any(
      model.actuator_biastype[actuator_ids] != mujoco.mjtBias.mjBIAS_NONE
    )
    or not np.allclose(model.actuator_gear[actuator_ids], expected_gear)
    or not np.allclose(model.actuator_gainprm[actuator_ids, 0], 1.0)
  ):
    raise ValueError(
      "The G1 plant must use direct, stateless, unit-gear torque motors; "
      "only scene terrain and the training-aligned foot contacts may change."
    )

  root_joint_id = _name_to_id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
  )
  if int(model.jnt_type[root_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
    raise ValueError("floating_base_joint must be a free joint.")
  camera_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_depth")

  if not math.isfinite(float(model.stat.extent)) or model.stat.extent <= 0.0:
    raise ValueError(f"Compiled model extent is invalid: {model.stat.extent}.")
  model.vis.map.znear = DEPTH_RENDER_NEAR_M / float(model.stat.extent)

  bindings = ModelBindings(
    dds_motor_ids=dds_motor_ids,
    joint_ids=joint_ids,
    joint_qpos_adr=qpos_adr,
    joint_dof_adr=dof_adr,
    actuator_ids=actuator_ids,
    actuator_ctrl_min=ctrl_ranges[:, 0].copy(),
    actuator_ctrl_max=ctrl_ranges[:, 1].copy(),
    root_qpos_adr=int(model.jnt_qposadr[root_joint_id]),
    root_dof_adr=int(model.jnt_dofadr[root_joint_id]),
    camera_id=camera_id,
    imu_quat_adr=_sensor_address(model, "imu_quat", 4),
    imu_gyro_adr=_sensor_address(model, "imu_gyro", 3),
    imu_acc_adr=_sensor_address(model, "imu_acc", 3),
  )
  static_geoms = int(
    np.count_nonzero(model.body_weldid[model.geom_bodyid] == 0)
  )
  return model, bindings, static_geoms


def initialize_data(
  model: mujoco.MjModel,
  bindings: ModelBindings,
  spawn_position: tuple[float, float, float],
) -> mujoco.MjData:
  data = mujoco.MjData(model)
  mujoco.mj_resetData(model, data)
  root = bindings.root_qpos_adr
  data.qpos[root : root + 7] = (
    *spawn_position,
    1.0,
    0.0,
    0.0,
    0.0,
  )
  data.qpos[bindings.joint_qpos_adr] = G1_HOME[bindings.dds_motor_ids]
  data.qvel[:] = 0.0
  data.ctrl[:] = 0.0
  mujoco.mj_forward(model, data)
  if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
    raise FloatingPointError("Initial MuJoCo state is not finite.")
  return data


def _quaternion_wxyz_to_rpy(quaternion: np.ndarray) -> np.ndarray:
  w, x, y, z = (float(value) for value in quaternion)
  norm = math.sqrt(w * w + x * x + y * y + z * z)
  if not math.isfinite(norm) or norm <= 1.0e-12:
    raise FloatingPointError(f"Invalid IMU quaternion: {quaternion}.")
  w, x, y, z = w / norm, x / norm, y / norm, z / norm
  roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  sin_pitch = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
  pitch = math.asin(sin_pitch)
  yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return np.asarray((roll, pitch, yaw), dtype=np.float64)


class DdsBridge:
  """Thread-safe SDK2 transport; all SDK imports and initialization are lazy."""

  def __init__(self, domain_id: int, interface: str) -> None:
    from unitree_sdk2py.core.channel import (
      ChannelFactoryInitialize,
      ChannelPublisher,
      ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    # validate_sim_transport has already rejected domain 0 and non-loopback.
    ChannelFactoryInitialize(domain_id, interface)
    self._lowstate_factory = unitree_hg_msg_dds__LowState_
    self._crc = CRC()
    self._crc_lock = threading.Lock()
    self._command_lock = threading.Lock()
    self._stats_lock = threading.Lock()
    self._latest_command: CommandSnapshot | None = None
    self._sequence = 0
    self._stats = TransportStats()

    self._lowstate_publisher = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
    self._depth_publisher = ChannelPublisher(TOPIC_STAIR_DEPTH, PointCloud2_)
    self._lowcmd_subscriber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
    self._lowstate_publisher.Init()
    self._depth_publisher.Init()
    self._lowcmd_subscriber.Init(self._on_lowcmd, 1)

  def _bump(self, name: str) -> None:
    with self._stats_lock:
      setattr(self._stats, name, getattr(self._stats, name) + 1)

  def _on_lowcmd(self, message: Any) -> None:
    try:
      with self._crc_lock:
        expected_crc = int(self._crc.Crc(message))
      if int(message.crc) != expected_crc:
        self._bump("bad_crc_commands")
        return
      if int(message.mode_pr) != 0 or int(message.mode_machine) != G1_MODE_MACHINE:
        raise ValueError("LowCmd mode does not target the G1 simulation machine.")
      motors = message.motor_cmd[:G1_NUM_MOTORS]
      if len(motors) != G1_NUM_MOTORS or any(
        int(motor.mode) != 1 for motor in motors
      ):
        raise ValueError("The first 29 LowCmd motors must all use mode=1.")
      q = np.asarray([motor.q for motor in motors], dtype=np.float64)
      dq = np.asarray([motor.dq for motor in motors], dtype=np.float64)
      tau = np.asarray([motor.tau for motor in motors], dtype=np.float64)
      kp = np.asarray([motor.kp for motor in motors], dtype=np.float64)
      kd = np.asarray([motor.kd for motor in motors], dtype=np.float64)
      if not all(np.all(np.isfinite(values)) for values in (q, dq, tau, kp, kd)):
        raise ValueError("LowCmd contains NaN or infinity.")
      if np.any(kp < 0.0) or np.any(kd < 0.0):
        raise ValueError("LowCmd kp and kd must be nonnegative.")
      with self._command_lock:
        self._sequence += 1
        self._latest_command = CommandSnapshot(
          sequence=self._sequence,
          received_at=time.monotonic(),
          q=q.copy(),
          dq=dq.copy(),
          tau=tau.copy(),
          kp=kp.copy(),
          kd=kd.copy(),
        )
      self._bump("accepted_commands")
    except (AttributeError, IndexError, TypeError, ValueError):
      self._bump("invalid_commands")

  def latest_command(self) -> CommandSnapshot | None:
    with self._command_lock:
      command = self._latest_command
      if command is None:
        return None
      return CommandSnapshot(
        sequence=command.sequence,
        received_at=command.received_at,
        q=command.q.copy(),
        dq=command.dq.copy(),
        tau=command.tau.copy(),
        kp=command.kp.copy(),
        kd=command.kd.copy(),
      )

  def publish_lowstate(
    self,
    data: mujoco.MjData,
    bindings: ModelBindings,
    previous_dq: np.ndarray,
  ) -> np.ndarray:
    active_q = np.asarray(
      data.qpos[bindings.joint_qpos_adr], dtype=np.float64
    )
    active_dq = np.asarray(
      data.qvel[bindings.joint_dof_adr], dtype=np.float64
    )
    active_ddq = (active_dq - previous_dq) / PHYSICS_DT
    active_tau_est = np.asarray(
      data.actuator_force[bindings.actuator_ids], dtype=np.float64
    )
    quaternion = np.asarray(
      data.sensordata[bindings.imu_quat_adr : bindings.imu_quat_adr + 4],
      dtype=np.float64,
    )
    gyroscope = np.asarray(
      data.sensordata[bindings.imu_gyro_adr : bindings.imu_gyro_adr + 3],
      dtype=np.float64,
    )
    accelerometer = np.asarray(
      data.sensordata[bindings.imu_acc_adr : bindings.imu_acc_adr + 3],
      dtype=np.float64,
    )
    if not all(
      np.all(np.isfinite(values))
      for values in (
        active_q,
        active_dq,
        active_ddq,
        active_tau_est,
        quaternion,
        gyroscope,
        accelerometer,
      )
    ):
      raise FloatingPointError("Cannot publish a non-finite LowState.")

    # Scatter model values through the validated DDS mapping instead of relying
    # on XML joint or actuator ordering.
    q = np.zeros(G1_NUM_MOTORS, dtype=np.float64)
    dq = np.zeros(G1_NUM_MOTORS, dtype=np.float64)
    ddq = np.zeros(G1_NUM_MOTORS, dtype=np.float64)
    tau_est = np.zeros(G1_NUM_MOTORS, dtype=np.float64)
    q[bindings.dds_motor_ids] = active_q
    dq[bindings.dds_motor_ids] = active_dq
    ddq[bindings.dds_motor_ids] = active_ddq
    tau_est[bindings.dds_motor_ids] = active_tau_est

    state = self._lowstate_factory()
    state.mode_pr = 0
    state.mode_machine = G1_MODE_MACHINE
    state.tick = int(round(float(data.time) * 1000.0)) & 0xFFFFFFFF
    state.imu_state.quaternion = quaternion.astype(np.float32).tolist()
    state.imu_state.gyroscope = gyroscope.astype(np.float32).tolist()
    state.imu_state.accelerometer = accelerometer.astype(np.float32).tolist()
    state.imu_state.rpy = _quaternion_wxyz_to_rpy(quaternion).astype(
      np.float32
    ).tolist()
    for index in range(G1_NUM_MOTORS):
      motor = state.motor_state[index]
      motor.mode = 1
      motor.q = float(q[index])
      motor.dq = float(dq[index])
      motor.ddq = float(ddq[index])
      motor.tau_est = float(tau_est[index])
    state.crc = 0
    with self._crc_lock:
      state.crc = int(self._crc.Crc(state))
    if not self._lowstate_publisher.Write(state):
      self._bump("lowstate_publish_failures")
    return active_dq.copy()

  def publish_depth(self, depth_z: np.ndarray, sim_time: float) -> None:
    message = encode_depth(depth_z, sim_time)
    if not self._depth_publisher.Write(message):
      self._bump("depth_publish_failures")

  def stats(self) -> TransportStats:
    with self._stats_lock:
      return TransportStats(**vars(self._stats))

  def close(self) -> None:
    self._lowcmd_subscriber.Close()
    self._depth_publisher.Close()
    self._lowstate_publisher.Close()


class DepthCamera:
  def __init__(self, model: mujoco.MjModel, camera_id: int) -> None:
    self._model = model
    self._camera_id = camera_id
    self._renderer = mujoco.Renderer(
      model, height=DEPTH_HEIGHT, width=DEPTH_WIDTH
    )
    self._renderer.enable_depth_rendering()
    self._scene_option = mujoco.MjvOption()
    self._scene_option.geomgroup[:] = 0
    self._scene_option.geomgroup[:3] = 1
    self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    self._far_clip = float(model.vis.map.zfar * model.stat.extent)
    self._near_clip = float(model.vis.map.znear * model.stat.extent)
    if not 0.0 < self._near_clip < DEPTH_MIN_M:
      raise ValueError(
        "Depth near plane must lie below the Stair clamp: "
        f"near={self._near_clip}, clamp={DEPTH_MIN_M}."
      )
    if (
      not math.isfinite(self._far_clip)
      or self._far_clip * DEPTH_BACKGROUND_FAR_FRACTION <= DEPTH_MAX_M
    ):
      raise ValueError(
        "Depth background threshold must exceed the Stair cutoff: "
        f"threshold={self._far_clip * DEPTH_BACKGROUND_FAR_FRACTION}, "
        f"cutoff={DEPTH_MAX_M}."
      )

  def capture(self, data: mujoco.MjData) -> np.ndarray:
    self._renderer.update_scene(
      data,
      camera=self._camera_id,
      scene_option=self._scene_option,
    )
    depth = np.asarray(self._renderer.render(), dtype=np.float32)
    if depth.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
      raise ValueError(
        f"Rendered depth shape is {depth.shape}; expected "
        f"({DEPTH_HEIGHT}, {DEPTH_WIDTH})."
      )
    misses = (
      ~np.isfinite(depth)
      | (depth <= 0.0)
      | (depth >= self._far_clip * DEPTH_BACKGROUND_FAR_FRACTION)
    )
    depth = depth.copy()
    depth[misses] = 0.0
    if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
      raise FloatingPointError("Depth image contains invalid optical-Z values.")
    return np.ascontiguousarray(depth, dtype=np.float32)

  def close(self) -> None:
    self._renderer.close()


def _colorize_depth(depth_z: np.ndarray) -> np.ndarray:
  """Convert metric optical-Z to RGB (near=warm, far=cool, miss=black)."""
  depth = np.asarray(depth_z, dtype=np.float32)
  if depth.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
    raise ValueError(f"Cannot display depth with shape {depth.shape}.")
  valid = np.isfinite(depth) & (depth > 0.0)
  normalized = np.nan_to_num(
    depth / DEPTH_MAX_M,
    nan=1.0,
    posinf=1.0,
    neginf=0.0,
  )
  normalized = np.clip(normalized, 0.0, 1.0)
  color_stops = np.asarray(
    (
      (255, 55, 20),
      (255, 220, 45),
      (40, 210, 210),
      (40, 90, 220),
      (15, 20, 55),
    ),
    dtype=np.float32,
  )
  scaled = normalized * (len(color_stops) - 1)
  lower = np.minimum(scaled.astype(np.int32), len(color_stops) - 2)
  blend = (scaled - lower)[..., None]
  image = color_stops[lower] * (1.0 - blend) + color_stops[lower + 1] * blend
  image[~valid] = 0.0
  return image.astype(np.uint8)


class DepthOverlay:
  """Native MuJoCo viewer overlay of the exact DDS optical-Z frame."""

  _SCALE = 4
  _MARGIN = 12

  def __init__(self, viewer_handle: Any) -> None:
    self._viewer = viewer_handle

  def update(self, depth_z: np.ndarray) -> None:
    image = _colorize_depth(depth_z)
    image = np.repeat(image, self._SCALE, axis=0)
    image = np.repeat(image, self._SCALE, axis=1)
    image = self._fit_to_viewport(image)
    height, width = image.shape[:2]
    # MjrRect uses a bottom-left origin, so subtracting both dimensions from
    # the main viewport's absolute right/top edges anchors this overlay at the
    # upper-right of the 3D view, after accounting for the left control panel.
    main_viewport = self._viewer.viewport
    viewport = mujoco.MjrRect(
      max(
        0,
        int(main_viewport.left + main_viewport.width) - width - self._MARGIN,
      ),
      max(
        0,
        int(main_viewport.bottom + main_viewport.height)
        - height
        - self._MARGIN,
      ),
      width,
      height,
    )
    self._viewer.set_images((viewport, image))

  def _fit_to_viewport(self, image: np.ndarray) -> np.ndarray:
    available_width = max(
      1, int(self._viewer.viewport.width) - 2 * self._MARGIN
    )
    available_height = max(
      1, int(self._viewer.viewport.height) - 2 * self._MARGIN
    )
    height, width = image.shape[:2]
    fit_scale = min(
      1.0, available_width / width, available_height / height
    )
    if fit_scale >= 1.0:
      return image
    target_width = max(1, int(width * fit_scale))
    target_height = max(1, int(height * fit_scale))
    x_indices = np.linspace(0, width - 1, target_width).round().astype(np.int32)
    y_indices = np.linspace(0, height - 1, target_height).round().astype(np.int32)
    return image[y_indices][:, x_indices]

  def close(self) -> None:
    if self._viewer.is_running():
      self._viewer.clear_images()


def _compute_control(
  data: mujoco.MjData,
  bindings: ModelBindings,
  command: CommandSnapshot | None,
  hold_q: np.ndarray,
  watchdog_active: bool,
) -> np.ndarray:
  q = np.asarray(data.qpos[bindings.joint_qpos_adr], dtype=np.float64)
  dq = np.asarray(data.qvel[bindings.joint_dof_adr], dtype=np.float64)
  dds_ids = bindings.dds_motor_ids
  if command is None or watchdog_active:
    control = G1_KP[dds_ids] * (hold_q - q) - G1_KD[dds_ids] * dq
  else:
    control = (
      command.tau[dds_ids]
      + command.kp[dds_ids] * (command.q[dds_ids] - q)
      + command.kd[dds_ids] * (command.dq[dds_ids] - dq)
    )
  if not np.all(np.isfinite(control)):
    control = G1_KP[dds_ids] * (hold_q - q) - G1_KD[dds_ids] * dq
  if not np.all(np.isfinite(control)):
    raise FloatingPointError("Both commanded and watchdog controls are non-finite.")
  return np.clip(
    control, bindings.actuator_ctrl_min, bindings.actuator_ctrl_max
  )


def _print_status(
  data: mujoco.MjData,
  bindings: ModelBindings,
  bridge: DdsBridge,
  watchdog_active: bool,
) -> None:
  stats = bridge.stats()
  root = bindings.root_qpos_adr
  control = np.asarray(data.ctrl[bindings.actuator_ids], dtype=np.float64)
  print(
    "[STATUS] "
    f"t={data.time:7.3f}s root=({data.qpos[root]:+.2f},"
    f"{data.qpos[root + 1]:+.2f},{data.qpos[root + 2]:+.2f}) "
    f"|tau|max={np.max(np.abs(control)):.1f} "
    f"watchdog={'HOLD' if watchdog_active else 'command'} "
    f"cmd_ok={stats.accepted_commands} crc_bad={stats.bad_crc_commands} "
    f"invalid={stats.invalid_commands} pub_fail="
    f"{stats.lowstate_publish_failures}/{stats.depth_publish_failures}"
  )


def run(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  bindings: ModelBindings,
  args: argparse.Namespace,
) -> None:
  bridge: DdsBridge | None = None
  depth_camera: DepthCamera | None = None
  depth_overlay: DepthOverlay | None = None
  viewer_handle: Any | None = None
  try:
    bridge = DdsBridge(args.domain_id, args.interface)
    depth_camera = DepthCamera(model, bindings.camera_id)
    if not args.headless:
      from mujoco import viewer as mj_viewer

      viewer_handle = mj_viewer.launch_passive(model, data)
      with viewer_handle.lock():
        viewer_handle.opt.flags[int(mujoco.mjtVisFlag.mjVIS_CAMERA)] = 0
        viewer_handle.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
    if args.show_depth:
      if viewer_handle is None:  # Defensive; argument validation rejects this.
        raise RuntimeError("--show-depth requires the native MuJoCo viewer.")
      depth_overlay = DepthOverlay(viewer_handle)
      print(
        "[INFO] Showing the published front_depth optical-Z frame in the "
        "native viewer (near=warm, far=cool, renderer miss=black)."
      )

    with viewer_handle.lock() if viewer_handle is not None else nullcontext():
      previous_dq = np.asarray(
        data.qvel[bindings.joint_dof_adr], dtype=np.float64
      ).copy()
      previous_dq = bridge.publish_lowstate(data, bindings, previous_dq)
      depth_frame = depth_camera.capture(data)
      bridge.publish_depth(depth_frame, float(data.time))
    if depth_overlay is not None:
      depth_overlay.update(depth_frame)

    hold_q = np.asarray(
      G1_HOME[bindings.dds_motor_ids], dtype=np.float64
    ).copy()
    latest_command: CommandSnapshot | None = None
    last_sequence = -1
    watchdog_active = True
    last_reported_watchdog: bool | None = None
    last_print_time = -math.inf

    if args.wait_for_command:
      print(
        "[WAIT] Physics is paused at the initial pose until the first valid "
        f"{TOPIC_LOWCMD} arrives; press Ctrl-C to stop."
      )
      command_wait_started = time.monotonic()
      next_sensor_publish = time.monotonic()
      while latest_command is None:
        if viewer_handle is not None and not viewer_handle.is_running():
          return
        latest_command = bridge.latest_command()
        if latest_command is not None:
          last_sequence = latest_command.sequence
          hold_q[:] = latest_command.q[bindings.dds_motor_ids]
          watchdog_active = False
          print("[WAIT] First valid LowCmd received; starting physics.")
          break
        now = time.monotonic()
        if (
          args.command_timeout > 0.0
          and now - command_wait_started >= args.command_timeout
        ):
          raise TimeoutError(
            f"No valid LowCmd arrived within {args.command_timeout:.1f}s."
          )
        if now >= next_sensor_publish:
          with viewer_handle.lock() if viewer_handle is not None else nullcontext():
            previous_dq = bridge.publish_lowstate(data, bindings, previous_dq)
            depth_frame = depth_camera.capture(data)
            bridge.publish_depth(depth_frame, float(data.time))
          if depth_overlay is not None:
            depth_overlay.update(depth_frame)
          if viewer_handle is not None:
            viewer_handle.sync()
          next_sensor_publish = now + DEPTH_DT
        time.sleep(0.001)

    with viewer_handle.lock() if viewer_handle is not None else nullcontext():
      start_sim_time = float(data.time)
    next_wall_tick = time.perf_counter()
    physics_steps = 0

    while True:
      with viewer_handle.lock() if viewer_handle is not None else nullcontext():
        elapsed = float(data.time) - start_sim_time
      if args.duration > 0.0 and elapsed >= args.duration - 1.0e-12:
        break
      if viewer_handle is not None and not viewer_handle.is_running():
        break

      received = bridge.latest_command()
      if received is not None:
        latest_command = received
        if received.sequence != last_sequence:
          last_sequence = received.sequence
          hold_q[:] = received.q[bindings.dds_motor_ids]
      next_watchdog_active = (
        latest_command is None
        or time.monotonic() - latest_command.received_at > args.watchdog_timeout
      )
      watchdog_entered = next_watchdog_active and not watchdog_active
      watchdog_active = next_watchdog_active
      watchdog_changed = watchdog_active != last_reported_watchdog

      publish_depth = (physics_steps + 1) % DEPTH_DECIMATION == 0
      depth_frame = None
      with viewer_handle.lock() if viewer_handle is not None else nullcontext():
        if watchdog_entered:
          hold_q[:] = data.qpos[bindings.joint_qpos_adr]
        if watchdog_changed:
          state = "holding current pose" if watchdog_active else "following LowCmd"
          print(f"[WATCHDOG] {state} at t={data.time:.3f}s")
          last_reported_watchdog = watchdog_active
        control = _compute_control(
          data, bindings, latest_command, hold_q, watchdog_active
        )
        data.ctrl[bindings.actuator_ids] = control
        mujoco.mj_step(model, data)
        # Refresh camera/body poses and sensors at the same post-step state used
        # by both messages. Both messages therefore carry one exact sim time.
        mujoco.mj_forward(model, data)
        physics_steps += 1
        previous_dq = bridge.publish_lowstate(data, bindings, previous_dq)
        if publish_depth:
          depth_frame = depth_camera.capture(data)
          bridge.publish_depth(depth_frame, float(data.time))

        if float(data.time) - last_print_time >= args.print_every:
          _print_status(data, bindings, bridge, watchdog_active)
          last_print_time = float(data.time)
      if publish_depth:
        if depth_overlay is not None and depth_frame is not None:
          depth_overlay.update(depth_frame)
        if viewer_handle is not None:
          viewer_handle.sync()

      if args.realtime:
        next_wall_tick += PHYSICS_DT
        now = time.perf_counter()
        if next_wall_tick > now:
          time.sleep(next_wall_tick - now)
        elif now - next_wall_tick > 0.25:
          next_wall_tick = now
  finally:
    if depth_overlay is not None:
      depth_overlay.close()
    if viewer_handle is not None:
      viewer_handle.close()
    if depth_camera is not None:
      depth_camera.close()
    if bridge is not None:
      bridge.close()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Official Unitree MuJoCo SDK2 server for G1 Stair."
  )
  parser.add_argument(
    "--scene-file",
    default=str(DEFAULT_SCENE_FILE),
    help=(
      "Complete Unitree MuJoCo G1 scene XML. Terrain and assets are loaded "
      "unchanged; foot contacts are replaced by the Stair training capsules."
    ),
  )
  parser.add_argument("--spawn-x", type=float, default=0.0)
  parser.add_argument("--spawn-y", type=float, default=0.0)
  parser.add_argument("--spawn-z", type=float, default=0.8)
  parser.add_argument("--domain-id", type=int, default=6)
  parser.add_argument("--interface", default="lo")
  parser.add_argument("--watchdog-timeout", type=float, default=0.1)
  parser.add_argument(
    "--command-timeout",
    type=float,
    default=0.0,
    help="Wall seconds to wait for first LowCmd (0 waits until Ctrl-C).",
  )
  parser.add_argument(
    "--duration",
    type=float,
    default=20.0,
    help="MuJoCo simulation seconds after physics starts (0 runs until Ctrl-C).",
  )
  parser.add_argument("--print-every", type=float, default=1.0)
  parser.add_argument("--headless", action="store_true")
  parser.add_argument(
    "--show-depth",
    action="store_true",
    help=(
      "Overlay the exact 48x64 optical-Z frame published over DDS in the "
      "native MuJoCo viewer."
    ),
  )
  parser.add_argument(
    "--wait-for-command",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Pause physics at the initial pose until the first valid LowCmd.",
  )
  parser.add_argument(
    "--realtime", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument(
    "--validate-only",
    action="store_true",
    help="Compile and validate the plant without initializing DDS or OpenGL.",
  )
  return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
  validate_sim_transport(args.domain_id, args.interface)
  finite = {
    "duration": args.duration,
    "print-every": args.print_every,
    "watchdog-timeout": args.watchdog_timeout,
    "command-timeout": args.command_timeout,
    "spawn-x": args.spawn_x,
    "spawn-y": args.spawn_y,
    "spawn-z": args.spawn_z,
  }
  for name, value in finite.items():
    if not math.isfinite(value):
      raise ValueError(f"--{name} must be finite; got {value}.")
  if args.duration < 0.0:
    raise ValueError("--duration must be nonnegative (0 runs until stopped).")
  if args.command_timeout < 0.0:
    raise ValueError("--command-timeout must be nonnegative.")
  if args.print_every <= 0.0:
    raise ValueError("--print-every must be positive.")
  if args.watchdog_timeout <= 0.0:
    raise ValueError("--watchdog-timeout must be positive.")
  if args.spawn_z <= 0.0:
    raise ValueError("--spawn-z must be positive.")
  if args.show_depth and (args.headless or args.validate_only):
    raise ValueError(
      "--show-depth requires the native GUI; do not combine it with "
      "--headless or --validate-only."
    )
  if not args.realtime and not args.validate_only:
    raise ValueError(
      "SDK2 mode requires realtime physics because LowCmd and freshness "
      "timeouts use wall time; --no-realtime is not supported."
    )
  if (
    not args.validate_only
    and os.environ.get("MUJOCO_GL", "").lower() == "disable"
  ):
    raise RuntimeError(
      "MUJOCO_GL=disable cannot render depth; use EGL for headless execution."
    )


def main() -> None:
  args = parse_args()
  _validate_args(args)
  scene_file = Path(args.scene_file).expanduser().resolve()
  model, bindings, static_geoms = build_model(scene_file)
  spawn_position = (args.spawn_x, args.spawn_y, args.spawn_z)
  data = initialize_data(model, bindings, spawn_position)
  print(
    "[INFO] Unitree MuJoCo G1 scene: "
    f"file={scene_file}, "
    f"nq={model.nq}, nv={model.nv}, nu={model.nu}, dt={model.opt.timestep:.3f}s, "
    f"static_geoms={static_geoms}, spawn={spawn_position}, waist=controlled, "
    f"foot_contacts={TRAINING_FOOT_GEOM_COUNT}xtraining_capsule, "
    "camera_shell=training_box, "
    f"depth={DEPTH_HEIGHT}x{DEPTH_WIDTH}@50Hz, "
    f"DDS=domain{args.domain_id}/{args.interface}"
  )
  print(
    "[INFO] Torque contract: tau + kp*(q_des-q) + kd*(dq_des-dq), "
    "clipped to each official actuator ctrlrange."
  )
  mismatched_limits = [
    (
      G1_DDS_JOINT_NAMES[index],
      float(bindings.actuator_ctrl_max[active_index]),
      float(G1_EFFORT_LIMIT[index]),
    )
    for active_index, index in enumerate(bindings.dds_motor_ids)
    if not (
      math.isclose(
        float(bindings.actuator_ctrl_min[active_index]),
        -float(G1_EFFORT_LIMIT[index]),
        abs_tol=1.0e-9,
      )
      and math.isclose(
        float(bindings.actuator_ctrl_max[active_index]),
        float(G1_EFFORT_LIMIT[index]),
        abs_tol=1.0e-9,
      )
    )
  ]
  if mismatched_limits:
    details = ", ".join(
      f"{name}: official=±{official:g}, training=±{training:g} Nm"
      for name, official, training in mismatched_limits
    )
    print(f"[WARN] Official-scene effort-limit domain shift: {details}.")
  if args.validate_only:
    print(
      "[VALID] Model/name/sensor/actuator/home contracts passed; "
      "DDS and OpenGL were not initialized."
    )
    return
  run(model, data, bindings, args)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("\n[INFO] Interrupted; SDK2 MuJoCo server stopped.")
