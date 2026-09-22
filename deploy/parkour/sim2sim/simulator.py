"""Standalone native-MuJoCo model, reset, depth camera, and viewer helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .course import SCENE_FILE, CourseStage
from .policy_runtime import (
    ACTUATED_JOINT_NAMES,
    CAMERA_POS,
    CAMERA_QUAT_WXYZ,
    DEPTH_FOVY_DEG,
    DEPTH_RENDER_HEIGHT,
    DEPTH_RENDER_WIDTH,
    G1_KP,
    NUM_ACTIONS,
    PHYSICS_DT,
    preprocess_v6_depth,
    quat_to_rotation,
)


ASSET_ROOT = Path(__file__).resolve().parent / "assets"
ROBOT_FILE = ASSET_ROOT / "unitree_g1/g1_29dof_spherehand.xml"
CAMERA_NAME = "student_depth"
CAMERA_VISUAL_NAME = "student_depth_camera_visual"
GOAL_MARKER_NAME = "parkour_sim2sim_goal"
CAMERA_VISUAL_GROUP = 4
GOAL_MARKER_GROUP = 5
ROBOT_COLLISION_GROUP = 3
EFFORT_LIMIT = np.asarray(
    (
        88, 139, 88, 139, 50, 50,
        88, 139, 88, 139, 50, 50,
        88, 50, 50,
        25, 25, 25, 25, 25, 5, 5,
        25, 25, 25, 25, 25, 5, 5,
    ),
    dtype=np.float64,
)
G1_NATURAL_FREQUENCY = 10.0 * 2.0 * math.pi
G1_ARMATURE = G1_KP / G1_NATURAL_FREQUENCY**2
FOOT_COLLISION_NAMES = frozenset(
    f"{side}_foot{index}_collision"
    for side in ("left", "right")
    for index in range(1, 8)
)
START_ROOT_QPOS = np.asarray(
    (
        0.010077368706469692,
        -0.0018540159019453384,
        0.793,
        0.9999271070658111,
        0.0,
        0.0,
        -0.012073961860045297,
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class ModelBindings:
    root_qpos_adr: int
    root_dof_adr: int
    pelvis_body_id: int
    torso_body_id: int
    camera_id: int
    gyro_sensor_adr: int
    joint_qpos_adr: np.ndarray
    joint_dof_adr: np.ndarray
    actuator_ids: np.ndarray
    goal_mocap_id: int


@dataclass(frozen=True)
class ResetState:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        raise ValueError(f"MuJoCo model is missing {object_type.name} {name!r}.")
    return int(result)


def _sensor_address(model: mujoco.MjModel, name: str, dimension: int) -> int:
    sensor = _id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if int(model.sensor_dim[sensor]) != dimension:
        raise ValueError(f"Sensor {name!r} does not have dimension {dimension}.")
    return int(model.sensor_adr[sensor])


def _configure_sphere_hand_plant(spec: mujoco.MjSpec) -> None:
    """Recreate the V6 training armature, collision, and torque plant."""
    if spec.actuators:
        raise ValueError("Standalone sphere-hand G1 must not contain actuators.")
    for joint_name, effort, armature in zip(
        ACTUATED_JOINT_NAMES, EFFORT_LIMIT, G1_ARMATURE, strict=True
    ):
        joint = spec.joint(joint_name)
        if joint is None:
            raise ValueError(f"Standalone G1 is missing joint {joint_name!r}.")
        joint.armature = float(armature)
        joint.damping = 0.0
        joint.frictionloss = 0.0
        spec.add_actuator(
            name=joint_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=joint_name,
            dyntype=mujoco.mjtDyn.mjDYN_NONE,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            biastype=mujoco.mjtBias.mjBIAS_NONE,
            biasprm=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            gear=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ctrllimited=True,
            ctrlrange=(-float(effort), float(effort)),
            forcelimited=True,
            forcerange=(-float(effort), float(effort)),
        )
    collisions: set[str] = set()
    for geom in spec.geoms:
        if not geom.name or not geom.name.endswith("_collision"):
            continue
        collisions.add(geom.name)
        geom.contype = 1
        geom.conaffinity = 1
        geom.condim = 3 if geom.name in FOOT_COLLISION_NAMES else 1
        if geom.name in FOOT_COLLISION_NAMES:
            geom.priority = 1
            geom.friction = (0.6, 0.005, 0.0001)
    missing_feet = FOOT_COLLISION_NAMES - collisions
    if missing_feet:
        raise ValueError(f"Standalone G1 is missing foot collisions: {missing_feet}.")


def build_model(
    stages: tuple[CourseStage, ...], ccd_iterations: int
) -> tuple[mujoco.MjModel, ModelBindings]:
    if len(stages) != 1 or stages[0].terrain_id != "15":
        raise ValueError("The standalone scene contains exactly terrain 15.")
    if stages[0].terrain_file.resolve() != SCENE_FILE.resolve():
        raise ValueError("Terrain 15 must resolve to the standalone scene_parkour.xml.")
    if not SCENE_FILE.is_file():
        raise FileNotFoundError(f"Standalone Parkour scene is missing: {SCENE_FILE}")
    if not ROBOT_FILE.is_file():
        raise FileNotFoundError(f"Standalone G1 MJCF is missing: {ROBOT_FILE}")
    spec = mujoco.MjSpec.from_file(str(SCENE_FILE))
    spec.modelname = "parkour_sim2sim_g1_amp_real_parkour"
    robot_spec = mujoco.MjSpec.from_file(str(ROBOT_FILE))
    _configure_sphere_hand_plant(robot_spec)
    torso = robot_spec.body("torso_link")
    if torso is None:
        raise ValueError("Standalone G1 is missing torso_link.")
    torso.add_camera(
        name=CAMERA_NAME,
        pos=tuple(CAMERA_POS),
        quat=tuple(CAMERA_QUAT_WXYZ),
        fovy=DEPTH_FOVY_DEG,
    )
    camera_rotation = quat_to_rotation(CAMERA_QUAT_WXYZ)
    visual_size = np.asarray((0.045, 0.018, 0.012))
    visual_pos = CAMERA_POS + camera_rotation[:, 2] * (visual_size[2] + 0.002)
    torso.add_geom(
        name=CAMERA_VISUAL_NAME,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=tuple(visual_size),
        pos=tuple(visual_pos),
        quat=tuple(CAMERA_QUAT_WXYZ),
        rgba=(0.04, 0.05, 0.06, 1.0),
        group=CAMERA_VISUAL_GROUP,
        contype=0,
        conaffinity=0,
        mass=0.0,
    )
    attachment = spec.worldbody.add_frame(name="standalone_g1_attachment")
    spec.attach(robot_spec, prefix="", suffix="", frame=attachment)
    marker = spec.worldbody.add_body(name=GOAL_MARKER_NAME, mocap=True)
    marker.add_geom(
        name=f"{GOAL_MARKER_NAME}_sphere",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.12, 0.0, 0.0),
        rgba=(0.1, 1.0, 0.2, 0.9),
        group=GOAL_MARKER_GROUP,
        contype=0,
        conaffinity=0,
        mass=0.0,
    )
    spec.option.timestep = PHYSICS_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 100
    spec.option.ls_iterations = 50
    spec.option.ccd_iterations = ccd_iterations
    model = spec.compile()
    if (model.nq, model.nv, model.nu) != (36, 35, NUM_ACTIONS):
        raise ValueError(
            f"Standalone G1 nq/nv/nu changed: {(model.nq, model.nv, model.nu)}."
        )
    root_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    qpos = np.empty(NUM_ACTIONS, dtype=np.int32)
    dof = np.empty(NUM_ACTIONS, dtype=np.int32)
    actuators = np.empty(NUM_ACTIONS, dtype=np.int32)
    for index, name in enumerate(ACTUATED_JOINT_NAMES):
        joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if int(model.actuator_trnid[actuator, 0]) != joint:
            raise ValueError(f"Actuator {name!r} does not drive its namesake joint.")
        qpos[index] = int(model.jnt_qposadr[joint])
        dof[index] = int(model.jnt_dofadr[joint])
        actuators[index] = actuator
    if not np.allclose(model.actuator_ctrlrange[actuators, 1], EFFORT_LIMIT):
        raise ValueError("Standalone actuator effort limits changed.")
    if not np.allclose(model.dof_armature[dof], G1_ARMATURE, atol=1.0e-12):
        raise ValueError("Standalone joint armatures differ from V6 training.")
    goal_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, GOAL_MARKER_NAME)
    goal_mocap = int(model.body_mocapid[goal_body])
    if goal_mocap < 0:
        raise ValueError("Goal marker is not a mocap body.")
    model.vis.map.znear = 0.001 / float(model.stat.extent)
    return model, ModelBindings(
        int(model.jnt_qposadr[root_joint]),
        int(model.jnt_dofadr[root_joint]),
        _id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"),
        _id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"),
        _id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME),
        _sensor_address(model, "imu_ang_vel", 3),
        qpos,
        dof,
        actuators,
        goal_mocap,
    )


def initialize_standing(
    model: mujoco.MjModel,
    bindings: ModelBindings,
    default_joint_pos: np.ndarray,
) -> mujoco.MjData:
    """Create the fixed AMP start state without loading a reference motion."""
    joint_pos = np.asarray(default_joint_pos, dtype=np.float64)
    if joint_pos.shape != (NUM_ACTIONS,) or not np.all(np.isfinite(joint_pos)):
        raise ValueError("AMP default joint position must contain 29 finite values.")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    root_q = bindings.root_qpos_adr
    data.qpos[root_q : root_q + 7] = START_ROOT_QPOS
    data.qpos[bindings.joint_qpos_adr] = joint_pos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def set_goal_marker(data: mujoco.MjData, bindings: ModelBindings, stage: CourseStage) -> None:
    data.mocap_pos[bindings.goal_mocap_id] = stage.goal_pos_w
    data.mocap_quat[bindings.goal_mocap_id] = stage.goal_quat_wxyz


def capture_reset_state(data: mujoco.MjData) -> ResetState:
    return ResetState(
        float(data.time), data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(),
        data.mocap_pos.copy(), data.mocap_quat.copy()
    )


def restore_reset_state(model: mujoco.MjModel, data: mujoco.MjData, state: ResetState) -> None:
    mujoco.mj_resetData(model, data)
    data.time = state.time
    data.qpos[:] = state.qpos
    data.qvel[:] = state.qvel
    data.ctrl[:] = state.ctrl
    data.mocap_pos[:] = state.mocap_pos
    data.mocap_quat[:] = state.mocap_quat
    mujoco.mj_forward(model, data)


class DepthCamera:
    def __init__(self, model: mujoco.MjModel, camera_id: int) -> None:
        self._renderer = mujoco.Renderer(
            model, height=DEPTH_RENDER_HEIGHT, width=DEPTH_RENDER_WIDTH
        )
        self._renderer.enable_depth_rendering()
        self._camera_id = camera_id
        self._option = mujoco.MjvOption()
        self._option.geomgroup[:] = 0
        self._option.geomgroup[0] = 1
        self._option.geomgroup[2] = 1
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self._far = float(model.vis.map.zfar * model.stat.extent)

    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        self._renderer.update_scene(
            data, camera=self._camera_id, scene_option=self._option
        )
        optical_z = np.asarray(self._renderer.render(), dtype=np.float32).copy()
        misses = ~np.isfinite(optical_z) | (optical_z <= 0.0) | (optical_z >= 0.99 * self._far)
        optical_z[misses] = 0.0
        return optical_z, preprocess_v6_depth(optical_z)

    def close(self) -> None:
        self._renderer.close()


def colorize_student_depth(student_depth: np.ndarray) -> np.ndarray:
    """Display the newest channel of the exact normalized ONNX depth tensor."""
    depth = np.asarray(student_depth, dtype=np.float32)
    if depth.ndim != 4 or depth.shape[0] != 1 or depth.shape[2:] != (18, 32):
        raise ValueError(
            f"student_depth must have shape (1,C,18,32); got {depth.shape}."
        )
    if depth.shape[1] < 1 or not np.all(np.isfinite(depth)):
        raise ValueError("student_depth must contain finite depth channels.")
    newest = (255.0 * np.clip(depth[0, -1], 0.0, 1.0)).astype(np.uint8)
    scale = max(1, 256 // newest.shape[1])
    newest = np.repeat(np.repeat(newest, scale, axis=0), scale, axis=1)
    return np.repeat(newest[..., None], 3, axis=-1)


class DepthOverlay:
    def __init__(self, viewer: Any) -> None:
        self.viewer = viewer

    def update(self, student_depth: np.ndarray) -> None:
        image = colorize_student_depth(student_depth)
        height, width = image.shape[:2]
        viewport = self.viewer.viewport
        target = mujoco.MjrRect(
            max(0, viewport.left + viewport.width - width - 12),
            max(0, viewport.bottom + viewport.height - height - 12),
            width,
            height,
        )
        self.viewer.set_images((target, image))

    def clear(self) -> None:
        if self.viewer.is_running():
            self.viewer.clear_images()


def open_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bindings: ModelBindings,
    key_callback: Any,
) -> Any:
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = bindings.pelvis_body_id
    viewer.cam.distance = 3.5
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -18.0
    with viewer.lock():
        viewer.opt.geomgroup[ROBOT_COLLISION_GROUP] = 0
        viewer.opt.geomgroup[CAMERA_VISUAL_GROUP] = 1
        viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_CAMERA)] = 0
    return viewer


__all__ = [
    "DepthCamera",
    "DepthOverlay",
    "EFFORT_LIMIT",
    "ModelBindings",
    "build_model",
    "capture_reset_state",
    "colorize_student_depth",
    "initialize_standing",
    "open_viewer",
    "restore_reset_state",
    "set_goal_marker",
]
