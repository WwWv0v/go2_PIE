#!/usr/bin/env python3
"""Run the G1 Stair policy against unitree_mujoco over SDK2 DDS.

This is a simulation-only controller.  It accepts state and optical-Z depth from
the companion unitree_mujoco server, runs the recurrent ONNX policy at 50 Hz,
and continuously publishes the latest 29-DoF PD target at 200 Hz.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from deploy.stair.sim2sim.policy_runtime import (  # noqa: E402
  ACTION_SCALE,
  ACTUATED_JOINT_NAMES,
  CONTROL_DT,
  DEFAULT_JOINT_POS,
  NUM_ACTIONS,
  OBS_DIM,
  StairOnnxPolicy,
  _quat_to_rotation,
  _wrap_to_pi,
  resolve_onnx_path,
)
from deploy.stair.sim2sim.sdk2_depth_codec import (  # noqa: E402
  G1_DDS_MOTOR_INDEX,
  G1_DDS_NUM_MOTORS,
  G1_HOME,
  G1_KD,
  G1_KP,
  TOPIC_STAIR_DEPTH,
  TOPIC_LOWCMD,
  TOPIC_LOWSTATE,
  decode_depth,
  preprocess_depth_z,
  validate_sim_transport,
)


DEFAULT_DOMAIN_ID = 6
DEFAULT_INTERFACE = "lo"
DEFAULT_LOWCMD_RATE_HZ = 200.0
LOWSTATE_TICK_SECONDS = 1.0e-3
LOWSTATE_HISTORY_LENGTH = 64
DEPTH_HISTORY_LENGTH = 16
G1_MODE_MACHINE = 5

MODEL_TO_DDS_IDS = np.asarray(
  [G1_DDS_MOTOR_INDEX[name] for name in ACTUATED_JOINT_NAMES], dtype=np.int32
)


@dataclass(frozen=True)
class LowStateSnapshot:
  received_at: float
  sim_time: float
  mode_machine: int
  quaternion_wxyz: np.ndarray
  angular_velocity: np.ndarray
  joint_pos: np.ndarray
  joint_vel: np.ndarray


@dataclass(frozen=True)
class DepthSnapshot:
  received_at: float
  sim_time: float
  tensor: np.ndarray


class SensorBuffer:
  """Thread-safe, owned snapshots copied out of DDS callback messages."""

  def __init__(self) -> None:
    self.condition = threading.Condition()
    self.lowstate: LowStateSnapshot | None = None
    self.depth: DepthSnapshot | None = None
    self.lowstate_history: deque[LowStateSnapshot] = deque(
      maxlen=LOWSTATE_HISTORY_LENGTH
    )
    self.depth_history: deque[DepthSnapshot] = deque(
      maxlen=DEPTH_HISTORY_LENGTH
    )
    self.lowstate_error: str | None = None
    self.depth_error: str | None = None

  def lowstate_callback(self, message: Any) -> None:
    try:
      motor_state = message.motor_state
      if len(motor_state) < G1_DDS_NUM_MOTORS:
        raise ValueError(
          f"LowState has {len(motor_state)} motors; expected at least "
          f"{G1_DDS_NUM_MOTORS}."
        )
      joint_pos = np.asarray(
        [motor_state[index].q for index in range(G1_DDS_NUM_MOTORS)],
        dtype=np.float32,
      )
      joint_vel = np.asarray(
        [motor_state[index].dq for index in range(G1_DDS_NUM_MOTORS)],
        dtype=np.float32,
      )
      quaternion = np.asarray(message.imu_state.quaternion, dtype=np.float64)
      angular_velocity = np.asarray(message.imu_state.gyroscope, dtype=np.float32)
      if quaternion.shape != (4,) or angular_velocity.shape != (3,):
        raise ValueError("LowState IMU does not have quaternion[4] and gyro[3].")
      quaternion_norm = float(np.linalg.norm(quaternion))
      if not math.isfinite(quaternion_norm) or quaternion_norm < 1.0e-6:
        raise ValueError("LowState IMU quaternion is invalid.")
      quaternion /= quaternion_norm
      arrays = (joint_pos, joint_vel, quaternion, angular_velocity)
      if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("LowState contains NaN or infinity.")
      tick = int(message.tick)
      if tick < 0:
        raise ValueError("LowState tick must be nonnegative simulation milliseconds.")
      if int(message.mode_pr) != 0 or int(message.mode_machine) != G1_MODE_MACHINE:
        raise ValueError(
          "LowState is not from the expected G1 29-DoF simulation machine."
        )
      if any(int(motor_state[index].mode) != 1 for index in range(29)):
        raise ValueError("The first 29 LowState motors must use mode=1.")
      snapshot = LowStateSnapshot(
        received_at=time.monotonic(),
        sim_time=tick * LOWSTATE_TICK_SECONDS,
        mode_machine=int(message.mode_machine),
        quaternion_wxyz=quaternion.copy(),
        angular_velocity=angular_velocity.copy(),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
      )
    except Exception as exc:  # DDS callbacks must not kill their reader thread.
      with self.condition:
        self.lowstate_error = str(exc)
        self.condition.notify_all()
      return
    with self.condition:
      self.lowstate = snapshot
      self.lowstate_history.append(snapshot)
      self.lowstate_error = None
      self.condition.notify_all()

  def depth_callback(self, message: Any) -> None:
    try:
      depth_z, sim_time = decode_depth(message)
      tensor = preprocess_depth_z(depth_z)
      if tensor.shape != (1, 1, 48, 64) or not np.all(np.isfinite(tensor)):
        raise ValueError(f"Preprocessed depth tensor is invalid: {tensor.shape}.")
      snapshot = DepthSnapshot(
        received_at=time.monotonic(),
        sim_time=float(sim_time),
        tensor=tensor.copy(),
      )
    except Exception as exc:  # DDS callbacks must not kill their reader thread.
      with self.condition:
        self.depth_error = str(exc)
        self.condition.notify_all()
      return
    with self.condition:
      self.depth = snapshot
      self.depth_history.append(snapshot)
      self.depth_error = None
      self.condition.notify_all()

  def pair(
    self,
    *,
    state_timeout: float,
    depth_timeout: float,
    max_sim_skew: float,
  ) -> tuple[LowStateSnapshot, DepthSnapshot]:
    with self.condition:
      lowstates = tuple(self.lowstate_history)
      depths = tuple(self.depth_history)
    if not lowstates or not depths:
      raise RuntimeError(self._missing_message())
    # DDS state and camera callbacks run independently. Pair the newest depth
    # frame with the closest retained LowState instead of racing two unrelated
    # "latest" slots. The server emits an exact state sample at every depth
    # timestamp, so the normal skew is 0--5 ms even under callback jitter.
    depth = depths[-1]
    lowstate = min(
      lowstates,
      key=lambda state: (
        abs(state.sim_time - depth.sim_time),
        -state.received_at,
      ),
    )
    now = time.monotonic()
    state_age = now - lowstate.received_at
    depth_age = now - depth.received_at
    sim_skew = abs(lowstate.sim_time - depth.sim_time)
    if state_age > state_timeout:
      raise RuntimeError(
        f"LowState is stale: age={state_age:.3f}s > {state_timeout:.3f}s."
      )
    if depth_age > depth_timeout:
      raise RuntimeError(
        f"Depth is stale: age={depth_age:.3f}s > {depth_timeout:.3f}s."
      )
    if sim_skew > max_sim_skew:
      raise RuntimeError(
        "LowState/depth simulation timestamps are not synchronized: "
        f"state={lowstate.sim_time:.3f}s, depth={depth.sim_time:.3f}s, "
        f"skew={sim_skew:.3f}s > {max_sim_skew:.3f}s. "
        "The simulator must encode mj_data.time in depth header.stamp and "
        "milliseconds in LowState.tick."
      )
    return lowstate, depth

  def wait_for_pair(
    self,
    timeout: float,
    *,
    state_timeout: float,
    depth_timeout: float,
    max_sim_skew: float,
  ) -> tuple[LowStateSnapshot, DepthSnapshot]:
    deadline = time.monotonic() + timeout
    last_error = self._missing_message()
    while True:
      try:
        return self.pair(
          state_timeout=state_timeout,
          depth_timeout=depth_timeout,
          max_sim_skew=max_sim_skew,
        )
      except RuntimeError as exc:
        last_error = str(exc)
      remaining = deadline - time.monotonic()
      if remaining <= 0.0:
        raise TimeoutError(
          f"Timed out waiting for synchronized SDK2 sensors: {last_error}"
        )
      with self.condition:
        self.condition.wait(timeout=min(remaining, 0.05))

  def wait_for_next_pair(
    self,
    previous_depth: DepthSnapshot,
    *,
    state_timeout: float,
    depth_timeout: float,
    max_sim_skew: float,
  ) -> tuple[LowStateSnapshot, DepthSnapshot]:
    """Wait for a fresh depth simulation timestamp and its closest state."""
    last_error = "No newer depth simulation timestamp has arrived."
    while True:
      try:
        lowstate, depth = self.pair(
          state_timeout=state_timeout,
          depth_timeout=depth_timeout,
          max_sim_skew=max_sim_skew,
        )
        if (
          depth.received_at > previous_depth.received_at
          and not math.isclose(
            depth.sim_time,
            previous_depth.sim_time,
            rel_tol=0.0,
            abs_tol=1.0e-9,
          )
        ):
          return lowstate, depth
        last_error = (
          "Depth transport is alive but MuJoCo simulation time has not advanced: "
          f"{depth.sim_time:.9f}s."
        )
        # Repeated frames at a paused simulation time still renew transport
        # freshness. Once physics starts, the next 20 ms frame is accepted.
        freshness_deadline = min(
          lowstate.received_at + state_timeout,
          depth.received_at + depth_timeout,
        )
      except RuntimeError as exc:
        last_error = str(exc)
        with self.condition:
          lowstate = self.lowstate
          depth = self.depth
        if lowstate is None or depth is None:
          raise RuntimeError(last_error) from exc
        freshness_deadline = min(
          lowstate.received_at + state_timeout,
          depth.received_at + depth_timeout,
        )

      remaining = freshness_deadline - time.monotonic()
      if remaining <= 0.0:
        raise TimeoutError(
          f"Timed out waiting for the next synchronized depth frame: {last_error}"
        )
      with self.condition:
        self.condition.wait(timeout=min(remaining, 0.02))

  def _missing_message(self) -> str:
    missing = []
    if self.lowstate is None:
      missing.append(f"{TOPIC_LOWSTATE} ({self.lowstate_error or 'no frame'})")
    if self.depth is None:
      missing.append(f"{TOPIC_STAIR_DEPTH} ({self.depth_error or 'no frame'})")
    return "Missing valid DDS input: " + ", ".join(missing)


class CommandTarget:
  """Latest PD target shared by policy and the high-rate publisher."""

  def __init__(self) -> None:
    self.lock = threading.Lock()
    self.target = np.asarray(G1_HOME, dtype=np.float64).copy()
    self.enabled = False
    self.mode_machine = 0

  def update(self, target: np.ndarray, mode_machine: int) -> None:
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (G1_DDS_NUM_MOTORS,) or not np.all(np.isfinite(target)):
      raise ValueError("LowCmd target must be a finite 29-vector.")
    with self.lock:
      self.target[:] = target
      self.mode_machine = int(mode_machine)
      self.enabled = True

  def snapshot(self) -> tuple[np.ndarray, int, bool]:
    with self.lock:
      return self.target.copy(), self.mode_machine, self.enabled


class Sdk2Transport:
  """Lazy SDK2 setup so --validate-only never initializes DDS."""

  def __init__(
    self,
    domain_id: int,
    interface: str,
    sensors: SensorBuffer,
    command: CommandTarget,
    lowcmd_rate: float,
  ) -> None:
    try:
      from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
      )
      from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
      from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
      from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
      from unitree_sdk2py.utils.crc import CRC
    except ImportError as exc:
      raise RuntimeError(
        "Unitree SDK2 Python is required in the active conda environment."
      ) from exc

    ChannelFactoryInitialize(domain_id, interface)
    self.sensors = sensors
    self.command = command
    self.lowcmd_rate = lowcmd_rate
    self.stop_event = threading.Event()
    self.thread: threading.Thread | None = None
    self.thread_error: BaseException | None = None
    self.publish_count = 0
    self.lowcmd_factory = unitree_hg_msg_dds__LowCmd_
    self.crc = CRC()
    self.crc_lock = threading.Lock()

    self.publisher = ChannelPublisher(TOPIC_LOWCMD, LowCmd_)
    self.publisher.Init()
    self.lowstate_subscriber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
    self.lowstate_subscriber.Init(self._on_lowstate, 10)
    self.depth_subscriber = ChannelSubscriber(TOPIC_STAIR_DEPTH, PointCloud2_)
    self.depth_subscriber.Init(sensors.depth_callback, 4)

  def start_publisher(self) -> None:
    if self.thread is not None:
      return
    self.thread = threading.Thread(
      target=self._publish_loop, name="stair-lowcmd", daemon=True
    )
    self.thread.start()

  def _on_lowstate(self, message: Any) -> None:
    try:
      with self.crc_lock:
        expected_crc = int(self.crc.Crc(message))
      if int(message.crc) != expected_crc:
        raise ValueError("LowState CRC check failed.")
    except Exception as exc:
      with self.sensors.condition:
        self.sensors.lowstate_error = str(exc)
        self.sensors.condition.notify_all()
      return
    self.sensors.lowstate_callback(message)

  def _publish_loop(self) -> None:
    period = 1.0 / self.lowcmd_rate
    deadline = time.monotonic()
    message = self.lowcmd_factory()
    try:
      while not self.stop_event.is_set():
        target, mode_machine, enabled = self.command.snapshot()
        message.mode_pr = 0
        message.mode_machine = mode_machine
        for index in range(G1_DDS_NUM_MOTORS):
          motor = message.motor_cmd[index]
          motor.mode = 1 if enabled else 0
          motor.q = float(target[index]) if enabled else 0.0
          motor.dq = 0.0
          motor.tau = 0.0
          motor.kp = float(G1_KP[index]) if enabled else 0.0
          motor.kd = float(G1_KD[index]) if enabled else 0.0
        with self.crc_lock:
          message.crc = self.crc.Crc(message)
        if not self.publisher.Write(message):
          raise RuntimeError("SDK2 LowCmd publisher Write returned False.")
        self.publish_count += 1
        deadline += period
        delay = deadline - time.monotonic()
        if delay > 0.0:
          self.stop_event.wait(delay)
        elif delay < -period:
          deadline = time.monotonic()
    except BaseException as exc:  # Propagate background failures to policy loop.
      self.thread_error = exc
      self.stop_event.set()

  def check(self) -> None:
    if self.thread_error is not None:
      raise RuntimeError("LowCmd publisher failed.") from self.thread_error

  def close(self) -> None:
    if self.thread is not None:
      self.stop_event.set()
      self.thread.join(timeout=1.0)
    self.lowstate_subscriber.Close()
    self.depth_subscriber.Close()
    self.publisher.Close()


def _validate_mapping() -> None:
  expected_ids = np.arange(G1_DDS_NUM_MOTORS, dtype=np.int32)
  if not np.array_equal(MODEL_TO_DDS_IDS, expected_ids):
    raise ValueError(
      f"Stair model-to-DDS joint mapping is invalid: {MODEL_TO_DDS_IDS.tolist()}."
    )
  if not np.allclose(G1_HOME[MODEL_TO_DDS_IDS], DEFAULT_JOINT_POS, atol=1.0e-7):
    raise ValueError("SDK2 home pose does not match the Stair ONNX joint order.")
  for name, values in (("G1_KP", G1_KP), ("G1_KD", G1_KD)):
    if values.shape != (29,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
      raise ValueError(f"{name} must be a finite positive 29-vector.")


def _command(lowstate: LowStateSnapshot, args: argparse.Namespace) -> np.ndarray:
  rotation = _quat_to_rotation(lowstate.quaternion_wxyz)
  heading = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
  if args.heading_hold:
    yaw = np.clip(
      args.heading_kp * _wrap_to_pi(args.heading_target - heading), -0.8, 0.8
    )
  else:
    yaw = np.clip(args.cmd_yaw, -0.8, 0.8)
  return np.asarray((args.cmd_x, args.cmd_y, yaw), dtype=np.float32)


def _observation(
  lowstate: LowStateSnapshot,
  command: np.ndarray,
  last_action: np.ndarray,
) -> np.ndarray:
  rotation = _quat_to_rotation(lowstate.quaternion_wxyz)
  projected_gravity = rotation.T @ np.asarray((0.0, 0.0, -1.0))
  observation = np.concatenate(
    (
      lowstate.angular_velocity,
      projected_gravity,
      command,
      lowstate.joint_pos[MODEL_TO_DDS_IDS] - DEFAULT_JOINT_POS,
      lowstate.joint_vel[MODEL_TO_DDS_IDS],
      last_action,
    )
  ).astype(np.float32)
  if observation.shape != (OBS_DIM,) or not np.all(np.isfinite(observation)):
    raise FloatingPointError(f"Invalid Stair observation: {observation}.")
  return observation


def _policy_target(action: np.ndarray) -> np.ndarray:
  action = np.asarray(action, dtype=np.float32)
  if action.shape != (NUM_ACTIONS,) or not np.all(np.isfinite(action)):
    raise FloatingPointError(f"Invalid Stair action: {action}.")
  target = np.asarray(G1_HOME, dtype=np.float64).copy()
  target[MODEL_TO_DDS_IDS] = (
    DEFAULT_JOINT_POS + ACTION_SCALE.astype(np.float64) * action
  )
  return target


def _load_policy(args: argparse.Namespace) -> StairOnnxPolicy:
  onnx_path, checkpoint_path = resolve_onnx_path(args.checkpoint_file)
  if checkpoint_path is not None:
    print(f"[WARN] Using sibling ONNX for checkpoint alias: {onnx_path}")
  else:
    print(f"[INFO] Loading ONNX: {onnx_path}")
  policy = StairOnnxPolicy(onnx_path, args.provider)
  if checkpoint_path is not None:
    exported_run = policy.metadata.get("run_path")
    if exported_run != checkpoint_path.parent.name:
      raise ValueError(
        "Sibling ONNX run metadata does not match checkpoint directory: "
        f"{exported_run!r} != {checkpoint_path.parent.name!r}."
      )
  return policy


def _validate_only(policy: StairOnnxPolicy, args: argparse.Namespace) -> None:
  lowstate = LowStateSnapshot(
    received_at=time.monotonic(),
    sim_time=0.0,
    mode_machine=0,
    quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
    angular_velocity=np.zeros(3, dtype=np.float32),
    joint_pos=np.asarray(G1_HOME, dtype=np.float32).copy(),
    joint_vel=np.zeros(29, dtype=np.float32),
  )
  depth = preprocess_depth_z(np.full((48, 64), 3.0, dtype=np.float32))
  command = _command(lowstate, args)
  observation = _observation(
    lowstate, command, np.zeros(NUM_ACTIONS, dtype=np.float32)
  )
  action = policy(observation, depth)
  target = _policy_target(action)
  print(
    "[OK] SDK2 controller validation passed without DDS initialization: "
    f"obs={observation.shape}, depth={depth.shape}, action={action.shape}, "
    f"target={target.shape}."
  )


def _sensor_pair(
  sensors: SensorBuffer,
  args: argparse.Namespace,
  previous_depth: DepthSnapshot | None = None,
) -> tuple[LowStateSnapshot, DepthSnapshot]:
  if previous_depth is None:
    return sensors.pair(
      state_timeout=args.state_timeout,
      depth_timeout=args.depth_timeout,
      max_sim_skew=args.max_sim_skew,
    )
  return sensors.wait_for_next_pair(
    previous_depth,
    state_timeout=args.state_timeout,
    depth_timeout=args.depth_timeout,
    max_sim_skew=args.max_sim_skew,
  )


def _stand_warmup(
  sensors: SensorBuffer,
  command_target: CommandTarget,
  args: argparse.Namespace,
) -> None:
  lowstate, _ = _sensor_pair(sensors, args)
  start_joint_pos = lowstate.joint_pos.astype(np.float64).copy()
  start = time.monotonic()
  while True:
    lowstate, _ = _sensor_pair(sensors, args)
    elapsed = time.monotonic() - start
    ratio = 1.0 if args.stand_warmup == 0.0 else min(
      elapsed / args.stand_warmup, 1.0
    )
    smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
    target = (1.0 - smooth_ratio) * start_joint_pos + smooth_ratio * G1_HOME
    command_target.update(target, lowstate.mode_machine)
    if ratio >= 1.0:
      return
    time.sleep(min(CONTROL_DT, args.stand_warmup - elapsed))


def _run(policy: StairOnnxPolicy, args: argparse.Namespace) -> None:
  sensors = SensorBuffer()
  command_target = CommandTarget()
  transport = Sdk2Transport(
    args.domain_id,
    args.interface,
    sensors,
    command_target,
    args.lowcmd_rate,
  )
  try:
    print(
      f"[INFO] Waiting for {TOPIC_LOWSTATE} and {TOPIC_STAIR_DEPTH} on "
      f"domain={args.domain_id}, interface={args.interface} ..."
    )
    sensors.wait_for_pair(
      args.startup_timeout,
      state_timeout=args.state_timeout,
      depth_timeout=args.depth_timeout,
      max_sim_skew=args.max_sim_skew,
    )
    lowstate, _ = _sensor_pair(sensors, args)
    command_target.update(lowstate.joint_pos, lowstate.mode_machine)
    transport.start_publisher()
    _stand_warmup(sensors, command_target, args)
    policy.reset()
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy_start = time.monotonic()
    next_print = policy_start + args.print_every
    control_steps = 0
    max_abs_action = 0.0
    last_sim_time = -math.inf
    previous_depth: DepthSnapshot | None = None
    first_depth_sim_time: float | None = None
    print("[INFO] Stand warmup complete; starting Stair policy at 50 Hz.")

    while args.duration == 0.0 or time.monotonic() - policy_start < args.duration:
      transport.check()
      lowstate, depth = _sensor_pair(sensors, args, previous_depth)
      if lowstate.sim_time + 1.0e-6 < last_sim_time:
        policy.reset()
        last_action.fill(0.0)
        print("[WARN] MuJoCo time reset detected; recurrent policy state cleared.")
      last_sim_time = lowstate.sim_time
      previous_depth = depth
      if first_depth_sim_time is None:
        first_depth_sim_time = depth.sim_time
      command = _command(lowstate, args)
      observation = _observation(lowstate, command, last_action)
      action = policy(observation, depth.tensor)
      target = _policy_target(action)
      command_target.update(target, lowstate.mode_machine)
      last_action[:] = action
      max_abs_action = max(max_abs_action, float(np.max(np.abs(action))))
      control_steps += 1

      now = time.monotonic()
      if now >= next_print:
        print(
          f"[STAT] t={now - policy_start:.1f}s sim={lowstate.sim_time:.3f}s "
          f"sensor_skew={abs(lowstate.sim_time - depth.sim_time):.3f}s "
          f"cmd=({command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}) "
          f"max|action|={max_abs_action:.3f} lowcmd={transport.publish_count}"
        )
        next_print += args.print_every

    elapsed = max(time.monotonic() - policy_start, 1.0e-9)
    sim_elapsed = (
      0.0
      if previous_depth is None or first_depth_sim_time is None
      else max(previous_depth.sim_time - first_depth_sim_time, 0.0)
    )
    sim_policy_rate = (
      0.0
      if sim_elapsed <= 0.0
      else max(control_steps - 1, 0) / sim_elapsed
    )
    print(
      f"[RESULT] wall_duration={elapsed:.2f}s sim_span={sim_elapsed:.3f}s "
      f"policy_steps={control_steps} wall_policy_rate={control_steps / elapsed:.1f}Hz "
      f"sim_policy_rate={sim_policy_rate:.1f}Hz "
      f"lowcmd_rate={transport.publish_count / (elapsed + args.stand_warmup):.1f}Hz "
      f"max_abs_action={max_abs_action:.3f}"
    )
  finally:
    transport.close()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Unitree SDK2 DDS sim2sim client for G1 Stair."
  )
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument("--domain-id", type=int, default=DEFAULT_DOMAIN_ID)
  parser.add_argument("--interface", default=DEFAULT_INTERFACE)
  parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="cpu")
  parser.add_argument("--cmd-x", type=float, default=0.5)
  parser.add_argument("--cmd-y", type=float, default=0.0)
  parser.add_argument("--cmd-yaw", type=float, default=0.0)
  parser.add_argument(
    "--heading-hold", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument("--heading-target", type=float, default=0.0)
  parser.add_argument("--heading-kp", type=float, default=1.2)
  parser.add_argument("--duration", type=float, default=20.0)
  parser.add_argument(
    "--stand-warmup",
    type=float,
    default=0.0,
    help=(
      "Seconds to interpolate to the home pose before policy control. "
      "The default starts Stair immediately because its reset pose is not "
      "statically stable in the official plant."
    ),
  )
  parser.add_argument("--startup-timeout", type=float, default=10.0)
  parser.add_argument("--state-timeout", type=float, default=0.50)
  parser.add_argument("--depth-timeout", type=float, default=0.50)
  parser.add_argument("--max-sim-skew", type=float, default=0.035)
  parser.add_argument("--lowcmd-rate", type=float, default=DEFAULT_LOWCMD_RATE_HZ)
  parser.add_argument("--print-every", type=float, default=1.0)
  parser.add_argument(
    "--validate-only",
    action="store_true",
    help="Validate policy/mapping/preprocessing without initializing SDK2 DDS.",
  )
  return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
  validate_sim_transport(args.domain_id, args.interface)
  finite_values = {
    "cmd-x": args.cmd_x,
    "cmd-y": args.cmd_y,
    "cmd-yaw": args.cmd_yaw,
    "heading-target": args.heading_target,
    "heading-kp": args.heading_kp,
    "duration": args.duration,
    "stand-warmup": args.stand_warmup,
    "startup-timeout": args.startup_timeout,
    "state-timeout": args.state_timeout,
    "depth-timeout": args.depth_timeout,
    "max-sim-skew": args.max_sim_skew,
    "lowcmd-rate": args.lowcmd_rate,
    "print-every": args.print_every,
  }
  for name, value in finite_values.items():
    if not math.isfinite(value):
      raise ValueError(f"--{name} must be finite; got {value}.")
  nonnegative = ("duration", "stand-warmup")
  for name in nonnegative:
    if getattr(args, name.replace("-", "_")) < 0.0:
      raise ValueError(f"--{name} must be nonnegative.")
  positive = (
    "startup-timeout",
    "state-timeout",
    "depth-timeout",
    "max-sim-skew",
    "lowcmd-rate",
    "print-every",
  )
  for name in positive:
    if getattr(args, name.replace("-", "_")) <= 0.0:
      raise ValueError(f"--{name} must be positive.")
  if args.lowcmd_rate < 1.0 / CONTROL_DT:
    raise ValueError("--lowcmd-rate must be at least the 50 Hz policy rate.")


def main() -> None:
  args = parse_args()
  _validate_args(args)
  _validate_mapping()
  policy = _load_policy(args)
  print(
    "[INFO] SDK2 sim-only contract: policy=50Hz, "
    f"LowCmd={args.lowcmd_rate:.0f}Hz, DDS={args.domain_id}/{args.interface}, "
    f"joints={NUM_ACTIONS} full-body policy DoFs."
  )
  if args.validate_only:
    _validate_only(policy, args)
    return
  _run(policy, args)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("\n[INFO] Interrupted; stopping LowCmd and closing DDS.")
