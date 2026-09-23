#!/usr/bin/env python3
"""Run an exported Unitree-Go2-PIE policy in native MuJoCo."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Renderer backends must be selected before importing MuJoCo.  Validation does
# not create a GL context, while a headless policy run defaults to EGL.
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/parkour_mjlab_pie_cache")
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

from deploy.pie.sim2sim.contract import (  # noqa: E402
    ACTION_SCALE,
    ACTUATED_JOINT_NAMES,
    DEFAULT_JOINT_POS,
    DEPTH_CAMERA_POS,
    DEPTH_CAMERA_QUAT,
    DEPTH_FOVY_DEG,
    DEPTH_HEIGHT,
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    DEPTH_RAW_WIDTH,
    DEPTH_UPDATE_PERIOD_STEPS,
    JOINT_ARMATURE,
    JOINT_DAMPING,
    JOINT_EFFORT_LIMIT,
    JOINT_STIFFNESS,
    NUM_ACTIONS,
    OBS_DIM,
    PHYSICS_DT,
    PHYSICS_STEPS_PER_CONTROL,
    DepthHistory,
    ProprioHistory,
    build_proprioception,
    preprocess_depth_z,
)
from deploy.pie.sim2sim.gamepad import (  # noqa: E402
    GAMEPAD_AXIS,
    Gamepad,
    forward_speed_from_unit_input,
    yaw_rate_from_unit_input,
)
from deploy.pie.sim2sim.policy_runtime import (  # noqa: E402
    PieOnnxPolicy,
    resolve_onnx_path,
)


TerrainKind = Literal["flat", "stairs", "stairs-up", "stairs-down"]
GO2_XML = REPO_ROOT / "src/assets/robots/unitree_go2/xmls/go2.xml"
STAIRS_SCENE_XML = Path(__file__).resolve().parent / "assets/scene_stairs.xml"
DEPTH_CAMERA_NAME = "front_depth"
DEPTH_CAMERA_VISUAL_NAME = "front_depth_camera_visual"
# Use a 9.0 cm-wide, 3.6 cm-high, 2.4 cm-deep black camera shell. Its local X
# half-size is the longer one, and its front face is 1 mm behind the optical
# center along camera-local +Z.
DEPTH_CAMERA_VISUAL_SIZE = (0.045, 0.018, 0.012)
DEPTH_CAMERA_VISUAL_POS = (0.332784, 0.0, 0.074446)
DEPTH_CAMERA_VISUAL_RGBA = (0.0, 0.0, 0.0, 1.0)
DEPTH_CAMERA_VISUAL_GROUP = 2
DEPTH_RENDER_NEAR_M = 0.001
DEPTH_BACKGROUND_FAR_FRACTION = 0.99
REFERENCE_STEP_HEIGHT = 0.12
REFERENCE_STEP_WIDTH = 0.30
REFERENCE_NUM_STEPS = 6


@dataclass(frozen=True)
class ModelBindings:
    root_qpos_adr: int
    root_dof_adr: int
    joint_ids: np.ndarray
    joint_qpos_adr: np.ndarray
    joint_dof_adr: np.ndarray
    actuator_ids: np.ndarray
    gyro_adr: int
    camera_id: int
    spawn_ground_height: float


@dataclass
class VelocityCommand:
    lin_vel_x: float
    ang_vel_z: float
    reset_requested: bool = False

    def vector(self) -> np.ndarray:
        return np.asarray((self.lin_vel_x, 0.0, self.ang_vel_z), dtype=np.float32)

    def key_callback(self, keycode: int) -> None:
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if key == "W":
            self.lin_vel_x = min(self.lin_vel_x + 0.1, 1.5)
        elif key == "S":
            self.lin_vel_x = max(self.lin_vel_x - 0.1, 0.0)
        elif key == "A":
            self.ang_vel_z = min(self.ang_vel_z + 0.1, 1.2)
        elif key == "D":
            self.ang_vel_z = max(self.ang_vel_z - 0.1, -1.2)
        elif key == " ":
            self.lin_vel_x = 0.0
            self.ang_vel_z = 0.0
        elif key == "R":
            self.reset_requested = True
        else:
            return
        print(f"[CMD] vx={self.lin_vel_x:+.2f} m/s, wz={self.ang_vel_z:+.2f} rad/s")


class GamepadVelocityController:
    """Map a local gamepad onto the command range used during PIE training."""

    def __init__(
        self,
        gamepad: Gamepad,
        *,
        max_speed: float,
        max_yaw_rate: float,
        deadzone: float,
    ) -> None:
        # These helpers also validate all three limits before the control loop.
        forward_speed_from_unit_input(0.0, max_speed=max_speed, deadzone=deadzone)
        yaw_rate_from_unit_input(0.0, max_yaw_rate=max_yaw_rate, deadzone=deadzone)
        self._gamepad = gamepad
        self._max_speed = max_speed
        self._max_yaw_rate = max_yaw_rate
        self._deadzone = deadzone
        self._reset_was_pressed = False

    def update(self, command: VelocityCommand) -> None:
        state = self._gamepad.sample()

        if state.stop_pressed:
            command.lin_vel_x = 0.0
            command.ang_vel_z = 0.0
        else:
            # PIE was trained with lin_vel_y fixed at zero and forward
            # velocity in [0, 1.5], so LX and reverse LY are intentionally
            # ignored instead of sending out-of-distribution commands.
            command.lin_vel_x = forward_speed_from_unit_input(
                state.ly,
                max_speed=self._max_speed,
                deadzone=self._deadzone,
            )
            command.ang_vel_z = yaw_rate_from_unit_input(
                state.rx,
                max_yaw_rate=self._max_yaw_rate,
                deadzone=self._deadzone,
            )

        if state.reset_pressed and not self._reset_was_pressed:
            command.reset_requested = True
        self._reset_was_pressed = state.reset_pressed

    def close(self) -> None:
        self._gamepad.close()


def _name_to_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo model is missing {object_type.name} {name!r}.")
    return int(object_id)


def _sensor_address(model: mujoco.MjModel, name: str, expected_dim: int) -> int:
    sensor_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    dimension = int(model.sensor_dim[sensor_id])
    if dimension != expected_dim:
        raise ValueError(
            f"Sensor {name!r} has dimension {dimension}; expected {expected_dim}."
        )
    return int(model.sensor_adr[sensor_id])


def _configure_robot_spec(spec: mujoco.MjSpec) -> None:
    """Apply the same joints, contacts, and position actuators as PIE."""
    thigh_ranges = {
        "FL_thigh_joint": (-1.5708, 2.2),
        "FR_thigh_joint": (-1.5708, 2.2),
        "RL_thigh_joint": (-0.5236, 2.2),
        "RR_thigh_joint": (-0.5236, 2.2),
    }
    for joint_name, joint_range in thigh_ranges.items():
        joint = spec.joint(joint_name)
        if joint is None:
            raise ValueError(f"Go2 model is missing joint {joint_name!r}.")
        joint.range = joint_range

    foot_names = {f"{leg}_foot_collision" for leg in ("FR", "FL", "RR", "RL")}
    for geom in spec.geoms:
        name = geom.name or ""
        if name.endswith("_collision"):
            geom.contype = 1
            geom.conaffinity = 1
            geom.condim = 1
            geom.priority = 0
            if name in foot_names:
                geom.condim = 3
                geom.priority = 1
                geom.friction[0] = 0.6
                geom.solimp[:3] = (0.9, 0.95, 0.023)
        else:
            geom.contype = 0
            geom.conaffinity = 0

    for index, joint_name in enumerate(ACTUATED_JOINT_NAMES):
        joint = spec.joint(joint_name)
        if joint is None:
            raise ValueError(f"Go2 model is missing joint {joint_name!r}.")
        joint.armature = float(JOINT_ARMATURE[index])

        stiffness = float(JOINT_STIFFNESS[index])
        damping = float(JOINT_DAMPING[index])
        effort = float(JOINT_EFFORT_LIMIT[index])
        actuator = spec.add_actuator(name=joint_name, target=joint_name)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm[0] = stiffness
        actuator.biasprm[1] = -stiffness
        actuator.biasprm[2] = -damping
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        actuator.forcerange = (-effort, effort)


def _load_robot_spec() -> mujoco.MjSpec:
    if not GO2_XML.is_file():
        raise FileNotFoundError(f"Go2 training XML not found: {GO2_XML}")
    robot_spec = mujoco.MjSpec.from_file(str(GO2_XML))
    _configure_robot_spec(robot_spec)
    return robot_spec


def _load_reference_stairs_spec() -> mujoco.MjSpec:
    """Load the PIE up/landing/down scene and attach the training Go2."""
    if not STAIRS_SCENE_XML.is_file():
        raise FileNotFoundError(f"PIE stairs scene XML not found: {STAIRS_SCENE_XML}")
    scene_spec = mujoco.MjSpec.from_file(str(STAIRS_SCENE_XML))
    robot_spec = _load_robot_spec()
    anchor = scene_spec.worldbody.add_frame(name="training_go2_attachment")
    scene_spec.attach(robot_spec, prefix="", suffix="", frame=anchor)
    return scene_spec


def _add_box(
    spec: mujoco.MjSpec,
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
) -> None:
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,
        size=size,
        rgba=rgba,
        contype=1,
        conaffinity=1,
        condim=3,
        friction=(0.6, 0.005, 0.0001),
    )


def _add_terrain(
    spec: mujoco.MjSpec,
    terrain: TerrainKind,
    step_height: float,
    step_width: float,
    num_steps: int,
) -> float:
    if step_height <= 0.0 or step_width <= 0.0 or num_steps <= 0:
        raise ValueError("Stair dimensions and count must be positive.")
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0.0, 0.0, 0.0),
        size=(20.0, 20.0, 0.1),
        rgba=(0.55, 0.55, 0.52, 1.0),
        contype=1,
        conaffinity=1,
        condim=3,
        friction=(0.6, 0.005, 0.0001),
    )
    if terrain == "flat":
        return 0.0

    stair_start = 1.0
    stair_half_width = 1.0
    total_height = num_steps * step_height
    color_a = (0.20, 0.43, 0.75, 1.0)
    color_b = (0.27, 0.52, 0.85, 1.0)

    if terrain == "stairs-down":
        # Elevated approach runway.  The robot starts on this top surface and
        # traverses forward down to the global ground plane.
        approach_half_length = 3.0
        _add_box(
            spec,
            "stairs_down_approach",
            (stair_start - approach_half_length, 0.0, total_height * 0.5),
            (approach_half_length, stair_half_width, total_height * 0.5),
            color_a,
        )

    for index in range(num_steps):
        top_height = (
            (index + 1) * step_height
            if terrain == "stairs-up"
            else (num_steps - index - 1) * step_height
        )
        if top_height <= 0.0:
            continue
        _add_box(
            spec,
            f"{terrain.replace('-', '_')}_step_{index:02d}",
            (
                stair_start + (index + 0.5) * step_width,
                0.0,
                top_height * 0.5,
            ),
            (step_width * 0.5, stair_half_width, top_height * 0.5),
            color_a if index % 2 == 0 else color_b,
        )

    if terrain == "stairs-up":
        platform_start = stair_start + num_steps * step_width
        platform_half_length = 3.0
        _add_box(
            spec,
            "stairs_up_platform",
            (
                platform_start + platform_half_length,
                0.0,
                total_height * 0.5,
            ),
            (platform_half_length, stair_half_width, total_height * 0.5),
            color_a,
        )
        return 0.0
    return total_height


def build_model(
    terrain: TerrainKind = "stairs",
    *,
    step_height: float = REFERENCE_STEP_HEIGHT,
    step_width: float = REFERENCE_STEP_WIDTH,
    num_steps: int = REFERENCE_NUM_STEPS,
) -> tuple[mujoco.MjModel, ModelBindings]:
    """Build the native MuJoCo PIE plant and policy-aligned camera."""
    if terrain == "stairs":
        if (
            not math.isclose(step_height, REFERENCE_STEP_HEIGHT)
            or not math.isclose(step_width, REFERENCE_STEP_WIDTH)
            or num_steps != REFERENCE_NUM_STEPS
        ):
            raise ValueError(
                "The reference 'stairs' scene is fixed at 0.12 m rise, "
                "0.30 m tread, and 6 steps. Use stairs-up or stairs-down "
                "for custom dimensions."
            )
        spec = _load_reference_stairs_spec()
        spawn_ground_height = 0.0
    else:
        spec = _load_robot_spec()
        spawn_ground_height = _add_terrain(
            spec, terrain, step_height, step_width, num_steps
        )

    base = spec.body("base_link")
    if base is None:
        raise ValueError("Go2 model is missing body 'base_link'.")
    if spec.camera(DEPTH_CAMERA_NAME) is not None:
        raise ValueError(f"Go2 model already defines camera {DEPTH_CAMERA_NAME!r}.")
    base.add_camera(
        name=DEPTH_CAMERA_NAME,
        pos=DEPTH_CAMERA_POS,
        quat=DEPTH_CAMERA_QUAT,
        fovy=DEPTH_FOVY_DEG,
    )
    base.add_geom(
        name=DEPTH_CAMERA_VISUAL_NAME,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=DEPTH_CAMERA_VISUAL_POS,
        quat=DEPTH_CAMERA_QUAT,
        size=DEPTH_CAMERA_VISUAL_SIZE,
        rgba=DEPTH_CAMERA_VISUAL_RGBA,
        contype=0,
        conaffinity=0,
        density=0.0,
        group=DEPTH_CAMERA_VISUAL_GROUP,
    )
    spec.worldbody.add_light(
        name="sun",
        pos=(0.0, 0.0, 5.0),
        dir=(0.0, 0.0, -1.0),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=False,
    )

    spec.option.timestep = PHYSICS_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.option.ccd_iterations = 500
    model = spec.compile()

    expected_dimensions = (7 + NUM_ACTIONS, 6 + NUM_ACTIONS, NUM_ACTIONS)
    if (model.nq, model.nv, model.nu) != expected_dimensions:
        raise ValueError(
            "Go2 model dimensions changed: "
            f"expected nq/nv/nu={expected_dimensions}, "
            f"actual={(model.nq, model.nv, model.nu)}."
        )

    joint_ids = np.empty(NUM_ACTIONS, dtype=np.int32)
    joint_qpos_adr = np.empty(NUM_ACTIONS, dtype=np.int32)
    joint_dof_adr = np.empty(NUM_ACTIONS, dtype=np.int32)
    actuator_ids = np.empty(NUM_ACTIONS, dtype=np.int32)
    for index, joint_name in enumerate(ACTUATED_JOINT_NAMES):
        joint_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
            raise ValueError(
                f"Actuator {joint_name!r} is connected to the wrong joint."
            )
        joint_ids[index] = joint_id
        joint_qpos_adr[index] = int(model.jnt_qposadr[joint_id])
        joint_dof_adr[index] = int(model.jnt_dofadr[joint_id])
        actuator_ids[index] = actuator_id

    expected_force_ranges = np.column_stack((-JOINT_EFFORT_LIMIT, JOINT_EFFORT_LIMIT))
    if (
        not np.allclose(
            model.actuator_gainprm[actuator_ids, 0], JOINT_STIFFNESS, atol=1.0e-12
        )
        or not np.allclose(
            model.actuator_biasprm[actuator_ids, 1],
            -JOINT_STIFFNESS,
            atol=1.0e-12,
        )
        or not np.allclose(
            model.actuator_biasprm[actuator_ids, 2],
            -JOINT_DAMPING,
            atol=1.0e-12,
        )
        or not np.allclose(
            model.actuator_forcerange[actuator_ids],
            expected_force_ranges,
            atol=1.0e-12,
        )
        or not np.allclose(
            model.dof_armature[joint_dof_adr], JOINT_ARMATURE, atol=1.0e-12
        )
    ):
        raise ValueError("Compiled Go2 actuator contract does not match PIE.")

    root_joint_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    camera_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_CAMERA, DEPTH_CAMERA_NAME)
    camera_visual_id = _name_to_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, DEPTH_CAMERA_VISUAL_NAME
    )
    base_body_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if not np.allclose(model.cam_pos[camera_id], DEPTH_CAMERA_POS, atol=1.0e-12):
        raise ValueError("Compiled camera position does not match PIE.")
    if not np.allclose(model.cam_quat[camera_id], DEPTH_CAMERA_QUAT, atol=1.0e-7):
        raise ValueError("Compiled camera orientation does not match PIE.")
    if not math.isclose(
        float(model.cam_fovy[camera_id]), DEPTH_FOVY_DEG, abs_tol=1.0e-10
    ):
        raise ValueError("Compiled camera FOV does not match PIE.")
    camera_visual_matches = (
        int(model.geom_bodyid[camera_visual_id]) == base_body_id
        and int(model.geom_type[camera_visual_id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
        and np.allclose(
            model.geom_pos[camera_visual_id],
            DEPTH_CAMERA_VISUAL_POS,
            atol=1.0e-12,
        )
        and np.allclose(
            model.geom_quat[camera_visual_id], DEPTH_CAMERA_QUAT, atol=1.0e-7
        )
        and np.allclose(
            model.geom_size[camera_visual_id],
            DEPTH_CAMERA_VISUAL_SIZE,
            atol=1.0e-12,
        )
        and np.allclose(
            model.geom_rgba[camera_visual_id],
            DEPTH_CAMERA_VISUAL_RGBA,
            atol=1.0e-7,
        )
        and int(model.geom_contype[camera_visual_id]) == 0
        and int(model.geom_conaffinity[camera_visual_id]) == 0
        and int(model.geom_group[camera_visual_id]) == DEPTH_CAMERA_VISUAL_GROUP
    )
    if not camera_visual_matches:
        raise ValueError("Compiled camera visual does not match PIE sim2sim.")

    model.vis.map.znear = DEPTH_RENDER_NEAR_M / float(model.stat.extent)
    return model, ModelBindings(
        root_qpos_adr=int(model.jnt_qposadr[root_joint_id]),
        root_dof_adr=int(model.jnt_dofadr[root_joint_id]),
        joint_ids=joint_ids,
        joint_qpos_adr=joint_qpos_adr,
        joint_dof_adr=joint_dof_adr,
        actuator_ids=actuator_ids,
        gyro_adr=_sensor_address(model, "imu_ang_vel", 3),
        camera_id=camera_id,
        spawn_ground_height=spawn_ground_height,
    )


def initialize_data(
    model: mujoco.MjModel, bindings: ModelBindings, spawn_x: float = 0.0
) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    root = bindings.root_qpos_adr
    data.qpos[root : root + 7] = (
        spawn_x,
        0.0,
        bindings.spawn_ground_height + 0.32,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    data.qpos[bindings.joint_qpos_adr] = DEFAULT_JOINT_POS
    data.qvel[:] = 0.0
    data.ctrl[bindings.actuator_ids] = DEFAULT_JOINT_POS
    mujoco.mj_forward(model, data)
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise FloatingPointError("Initial MuJoCo state is not finite.")
    return data


def build_observation(
    data: mujoco.MjData,
    bindings: ModelBindings,
    command: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    root = bindings.root_qpos_adr
    quaternion = data.qpos[root + 3 : root + 7]
    angular_velocity = data.sensordata[bindings.gyro_adr : bindings.gyro_adr + 3]
    joint_pos = data.qpos[bindings.joint_qpos_adr]
    joint_vel = data.qvel[bindings.joint_dof_adr]
    return build_proprioception(
        angular_velocity,
        quaternion,
        command,
        joint_pos,
        joint_vel,
        last_action,
    )


class DepthCamera:
    """Native MuJoCo optical-Z renderer matching the raw PIE camera."""

    def __init__(self, model: mujoco.MjModel, camera_id: int) -> None:
        self._model = model
        self._camera_id = camera_id
        self._renderer = mujoco.Renderer(
            model, height=DEPTH_HEIGHT, width=DEPTH_RAW_WIDTH
        )
        self._renderer.enable_depth_rendering()
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[:] = 0
        self._scene_option.geomgroup[:3] = 1
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self._far_clip = float(model.vis.map.zfar * model.stat.extent)
        near_clip = float(model.vis.map.znear * model.stat.extent)
        if not 0.0 < near_clip < DEPTH_MIN_M:
            raise ValueError(
                f"Depth near plane {near_clip} must be below {DEPTH_MIN_M}."
            )
        if self._far_clip * DEPTH_BACKGROUND_FAR_FRACTION <= DEPTH_MAX_M:
            raise ValueError("Depth far plane does not exceed the PIE cutoff.")

    def capture(self, data: mujoco.MjData) -> np.ndarray:
        self._renderer.update_scene(
            data, camera=self._camera_id, scene_option=self._scene_option
        )
        depth = np.asarray(self._renderer.render(), dtype=np.float32)
        if depth.shape != (DEPTH_HEIGHT, DEPTH_RAW_WIDTH):
            raise ValueError(f"Rendered depth has invalid shape {depth.shape}.")
        misses = (
            ~np.isfinite(depth)
            | (depth <= 0.0)
            | (depth >= self._far_clip * DEPTH_BACKGROUND_FAR_FRACTION)
        )
        result = depth.copy()
        result[misses] = 0.0
        return np.ascontiguousarray(result, dtype=np.float32)

    def close(self) -> None:
        self._renderer.close()


def _colorize_depth(depth_z: np.ndarray) -> np.ndarray:
    """Convert metric optical-Z to RGB (near=warm, far=cool, miss=black)."""
    depth = np.asarray(depth_z, dtype=np.float32)
    if depth.shape != (DEPTH_HEIGHT, DEPTH_RAW_WIDTH):
        raise ValueError(f"Cannot display depth with shape {depth.shape}.")
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.clip(
        np.nan_to_num(depth / DEPTH_MAX_M, nan=1.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
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
    """Native viewer overlay of the exact raw frame entering preprocessing."""

    _SCALE = 3
    _MARGIN = 12

    def __init__(self, viewer_handle) -> None:
        self._viewer = viewer_handle

    def update(self, depth_z: np.ndarray) -> None:
        image = _colorize_depth(depth_z)
        image = np.repeat(image, self._SCALE, axis=0)
        image = np.repeat(image, self._SCALE, axis=1)
        image = self._fit_to_viewport(image)
        height, width = image.shape[:2]
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
        available_width = max(1, int(self._viewer.viewport.width) - 2 * self._MARGIN)
        available_height = max(1, int(self._viewer.viewport.height) - 2 * self._MARGIN)
        height, width = image.shape[:2]
        fit_scale = min(1.0, available_width / width, available_height / height)
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


class Sim2SimRuntime:
    def __init__(
        self,
        model: mujoco.MjModel,
        bindings: ModelBindings,
        data: mujoco.MjData,
        policy: PieOnnxPolicy,
        camera: DepthCamera,
        command: VelocityCommand,
        gamepad: GamepadVelocityController | None = None,
    ) -> None:
        self.model = model
        self.bindings = bindings
        self.data = data
        self.policy = policy
        self.camera = camera
        self.command = command
        self.gamepad = gamepad
        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.proprio_history = ProprioHistory()
        self.depth_history = DepthHistory()
        self.control_step = 0
        self.latest_raw_depth: np.ndarray | None = None

    def reset(self) -> None:
        new_data = initialize_data(self.model, self.bindings)
        self.data.qpos[:] = new_data.qpos
        self.data.qvel[:] = new_data.qvel
        self.data.ctrl[:] = new_data.ctrl
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.last_action.fill(0.0)
        self.proprio_history.reset()
        self.depth_history.reset()
        self.policy.reset()
        self.control_step = 0
        self.latest_raw_depth = None
        self.command.reset_requested = False

    def step(self) -> None:
        if self.gamepad is not None:
            self.gamepad.update(self.command)
        if self.command.reset_requested:
            self.reset()

        if self.control_step % DEPTH_UPDATE_PERIOD_STEPS == 0:
            raw_depth = self.camera.capture(self.data)
            self.latest_raw_depth = raw_depth
            self.depth_history.append(preprocess_depth_z(raw_depth))

        observation = build_observation(
            self.data,
            self.bindings,
            self.command.vector(),
            self.last_action,
        )
        self.proprio_history.append(observation)
        action = self.policy(
            observation,
            self.proprio_history.array,
            self.depth_history.array,
        )
        target = DEFAULT_JOINT_POS + ACTION_SCALE * action
        if not np.all(np.isfinite(target)):
            raise FloatingPointError("PIE target joint position is non-finite.")
        self.data.ctrl[self.bindings.actuator_ids] = target
        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            mujoco.mj_step(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(
            np.isfinite(self.data.qvel)
        ):
            raise FloatingPointError("MuJoCo state became non-finite.")
        self.last_action = action
        self.control_step += 1


def validate_runtime_contract(
    model: mujoco.MjModel, bindings: ModelBindings, data: mujoco.MjData
) -> None:
    observation = build_observation(
        data,
        bindings,
        np.zeros(3, dtype=np.float32),
        np.zeros(NUM_ACTIONS, dtype=np.float32),
    )
    proprio_history = ProprioHistory()
    proprio_history.append(observation)
    synthetic_depth = np.full(
        (DEPTH_HEIGHT, DEPTH_RAW_WIDTH), DEPTH_MAX_M, dtype=np.float32
    )
    depth_history = DepthHistory()
    depth_history.append(preprocess_depth_z(synthetic_depth))
    if observation.shape != (OBS_DIM,):
        raise AssertionError("Observation contract validation failed.")
    print(
        "[OK] PIE native sim2sim contract: "
        f"nq/nv/nu={model.nq}/{model.nv}/{model.nu}, "
        f"obs={observation.shape}, history={proprio_history.array.shape}, "
        f"depth={depth_history.array.shape}, camera_fovy={DEPTH_FOVY_DEG:.6f}deg, "
        f"spawn_ground_z={bindings.spawn_ground_height:.3f}m"
    )


def _run_headless(runtime: Sim2SimRuntime, duration: float, realtime: bool) -> None:
    end_time = float(runtime.data.time) + duration
    while float(runtime.data.time) < end_time:
        started = time.monotonic()
        runtime.step()
        if realtime:
            remaining = PHYSICS_DT * PHYSICS_STEPS_PER_CONTROL - (
                time.monotonic() - started
            )
            if remaining > 0.0:
                time.sleep(remaining)


def _run_viewer(runtime: Sim2SimRuntime, show_depth: bool) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(
        runtime.model,
        runtime.data,
        key_callback=runtime.command.key_callback,
    ) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        overlay = DepthOverlay(viewer) if show_depth else None
        try:
            while viewer.is_running():
                started = time.monotonic()
                runtime.step()
                if overlay is not None and runtime.latest_raw_depth is not None:
                    overlay.update(runtime.latest_raw_depth)
                viewer.sync()
                remaining = PHYSICS_DT * PHYSICS_STEPS_PER_CONTROL - (
                    time.monotonic() - started
                )
                if remaining > 0.0:
                    time.sleep(remaining)
        finally:
            if overlay is not None:
                overlay.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native MuJoCo sim2sim for Unitree-Go2-PIE."
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        help="Exported policy.onnx or its newest sibling model_N.pt alias.",
    )
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--terrain",
        choices=("flat", "stairs", "stairs-up", "stairs-down"),
        default="stairs",
    )
    parser.add_argument(
        "--step-height",
        type=float,
        default=REFERENCE_STEP_HEIGHT,
        help="Step rise for standalone stairs-up/stairs-down terrains.",
    )
    parser.add_argument(
        "--step-width",
        type=float,
        default=REFERENCE_STEP_WIDTH,
        help="Tread depth for standalone stairs-up/stairs-down terrains.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=REFERENCE_NUM_STEPS,
        help="Step count for standalone stairs-up/stairs-down terrains.",
    )
    parser.add_argument("--lin-vel-x", type=float, default=0.5)
    parser.add_argument("--ang-vel-z", type=float, default=0.0)
    parser.add_argument(
        "--joystick",
        action="store_true",
        help="Control forward speed and yaw from a local pygame gamepad.",
    )
    parser.add_argument(
        "--joystick-type",
        choices=tuple(GAMEPAD_AXIS),
        default="xbox",
    )
    parser.add_argument("--joystick-device", type=int, default=0)
    parser.add_argument("--joystick-deadzone", type=float, default=0.10)
    parser.add_argument("--joystick-max-speed", type=float, default=1.5)
    parser.add_argument("--joystick-max-yaw-rate", type=float, default=1.2)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Overlay the raw 60x106 optical-Z frame in the native viewer.",
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Compile and validate the plant/camera/observation contract without GL or ONNX.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.lin_vel_x <= 1.5:
        raise ValueError("--lin-vel-x must be in [0, 1.5].")
    if not -1.2 <= args.ang_vel_z <= 1.2:
        raise ValueError("--ang-vel-z must be in [-1.2, 1.2].")
    if args.joystick_device < 0:
        raise ValueError("--joystick-device must be nonnegative.")
    if not 0.0 <= args.joystick_deadzone < 1.0:
        raise ValueError("--joystick-deadzone must be in [0, 1).")
    if not 0.0 <= args.joystick_max_speed <= 1.5:
        raise ValueError("--joystick-max-speed must be in [0, 1.5].")
    if not 0.0 <= args.joystick_max_yaw_rate <= 1.2:
        raise ValueError("--joystick-max-yaw-rate must be in [0, 1.2].")
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if not args.validate_only and not args.checkpoint_file:
        raise ValueError(
            "--checkpoint-file is required unless --validate-only is used."
        )
    if args.show_depth and (args.headless or args.validate_only):
        raise ValueError(
            "--show-depth requires the native viewer; do not combine it with "
            "--headless or --validate-only."
        )
    if (
        args.headless
        and not args.validate_only
        and os.environ.get("MUJOCO_GL") == "disable"
    ):
        raise RuntimeError(
            "Headless depth rendering cannot use MUJOCO_GL=disable; use egl or unset it."
        )

    model, bindings = build_model(
        args.terrain,
        step_height=args.step_height,
        step_width=args.step_width,
        num_steps=args.num_steps,
    )
    data = initialize_data(model, bindings)
    validate_runtime_contract(model, bindings, data)
    if args.validate_only:
        return

    assert args.checkpoint_file is not None
    onnx_path, checkpoint_path = resolve_onnx_path(args.checkpoint_file)
    if checkpoint_path is not None:
        print(
            f"[WARN] Using sibling ONNX export for {checkpoint_path.name}: {onnx_path}"
        )
    policy = PieOnnxPolicy(onnx_path, provider=args.provider)
    print(f"[INFO] ONNX={onnx_path}, providers={policy.providers}")
    command = VelocityCommand(args.lin_vel_x, args.ang_vel_z)
    gamepad = None
    camera = None
    try:
        if args.joystick:
            gamepad = GamepadVelocityController(
                Gamepad(args.joystick_device, args.joystick_type),
                max_speed=args.joystick_max_speed,
                max_yaw_rate=args.joystick_max_yaw_rate,
                deadzone=args.joystick_deadzone,
            )
            print(
                "[INFO] Gamepad controls: left-stick up=forward, "
                "right-stick left/right=yaw, A=stop, Start=reset."
            )
        camera = DepthCamera(model, bindings.camera_id)
        runtime = Sim2SimRuntime(
            model, bindings, data, policy, camera, command, gamepad=gamepad
        )
        if args.headless:
            _run_headless(runtime, args.duration, args.realtime)
        else:
            print("[INFO] Keyboard: W/S speed, A/D yaw, Space stop, R reset.")
            _run_viewer(runtime, args.show_depth)
    finally:
        if camera is not None:
            camera.close()
        if gamepad is not None:
            gamepad.close()
    print(
        f"[OK] sim_time={float(data.time):.3f}s, "
        f"control_steps={runtime.control_step}, base_xyz={data.qpos[:3]}"
    )


if __name__ == "__main__":
    main()
