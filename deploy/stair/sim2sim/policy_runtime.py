"""Runtime-only ONNX contract shared by the Unitree SDK2 controller."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTROL_DT = 0.02
DEPTH_HEIGHT = 48
DEPTH_WIDTH = 64

LEG_JOINT_NAMES = (
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
)
WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
ARM_JOINT_NAMES = (
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
ACTUATED_JOINT_NAMES = LEG_JOINT_NAMES + WAIST_JOINT_NAMES + ARM_JOINT_NAMES
NUM_ACTIONS = len(ACTUATED_JOINT_NAMES)
OBS_DIM = 3 + 3 + 3 + NUM_ACTIONS * 3

DEFAULT_JOINT_POS = np.array(
    [
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.35,
        0.18,
        0.0,
        0.87,
        0.0,
        0.0,
        0.0,
        0.35,
        -0.18,
        0.0,
        0.87,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
ACTION_SCALE = np.array(
    [
        0.5475464629911068,
        0.35066146637882434,
        0.5475464629911068,
        0.35066146637882434,
        0.43857731392336724,
        0.43857731392336724,
    ]
    * 2
    + [
        0.5475464629911068,
        0.43857731392336724,
        0.43857731392336724,
    ]
    + [
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.07450087032950714,
        0.07450087032950714,
    ]
    * 2,
    dtype=np.float32,
)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (REPO_ROOT / path).resolve()


def resolve_onnx_path(checkpoint_file: str | Path) -> tuple[Path, Path | None]:
    """Resolve policy.onnx, optionally from a latest model_N.pt alias."""
    requested = _resolve_path(checkpoint_file)
    if requested.suffix.lower() == ".onnx":
        onnx_path = requested
        checkpoint_path = None
    elif requested.suffix.lower() == ".pt":
        checkpoint_path = requested
        onnx_path = requested.with_name("policy.onnx")
    else:
        raise ValueError(
            "--checkpoint-file must point to model_N.pt or policy.onnx; "
            f"got: {requested}"
        )

    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path is not None:
        try:
            requested_step = int(checkpoint_path.stem.removeprefix("model_"))
        except ValueError as exc:
            raise ValueError(
                "A .pt ONNX alias must be named model_N.pt. Pass policy.onnx explicitly "
                f"for other filenames: {checkpoint_path}"
            ) from exc
        checkpoint_steps = []
        for candidate in checkpoint_path.parent.glob("model_*.pt"):
            try:
                checkpoint_steps.append(int(candidate.stem.removeprefix("model_")))
            except ValueError:
                continue
        if checkpoint_steps and requested_step != max(checkpoint_steps):
            raise ValueError(
                f"{checkpoint_path.name} is not the newest checkpoint in its run directory. "
                "policy.onnx is overwritten on every save, so this alias cannot prove they "
                "match. Export the requested checkpoint and pass that ONNX explicitly."
            )
    if not onnx_path.is_file():
        suffix = (
            " The SDK2 controller needs the sibling policy.onnx export."
            if checkpoint_path is not None
            else ""
        )
        raise FileNotFoundError(f"ONNX policy not found: {onnx_path}.{suffix}")
    return onnx_path, checkpoint_path


def _parse_csv_floats(metadata: dict[str, str], key: str) -> np.ndarray:
    try:
        values = [float(item) for item in metadata[key].split(",") if item]
    except KeyError as exc:
        raise ValueError(f"ONNX metadata is missing required field {key!r}.") from exc
    return np.asarray(values, dtype=np.float64)


def _as_fixed_shape(value: Sequence[object], name: str) -> tuple[int, ...]:
    if not all(isinstance(dim, int) and dim > 0 for dim in value):
        raise ValueError(f"ONNX {name} must have a fixed positive shape, got {value}.")
    return tuple(int(dim) for dim in value)


class StairOnnxPolicy:
    """ONNX policy wrapper that owns the exported LSTM and GRU states."""

    INPUT_SHAPES = {
        "obs": (1, OBS_DIM),
        "depth": (1, 1, DEPTH_HEIGHT, DEPTH_WIDTH),
        "estimator_h_in": (1, 1, 128),
        "estimator_c_in": (1, 1, 128),
        "recurrent_h_in": (1, 1, 256),
    }
    OUTPUT_SHAPES = {
        "actions": (1, NUM_ACTIONS),
        "estimator_h_out": (1, 1, 128),
        "estimator_c_out": (1, 1, 128),
        "recurrent_h_out": (1, 1, 256),
    }

    def __init__(self, path: Path, provider: str) -> None:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        available = ort.get_available_providers()
        if provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "--provider cuda was requested, but onnxruntime has no CUDA provider. "
                    f"Available providers: {available}"
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "auto" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.path = path
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers
        )
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self._validate_contract()
        self.reset()

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def _validate_contract(self) -> None:
        actual_inputs = {
            item.name: _as_fixed_shape(item.shape, f"input {item.name!r}")
            for item in self.session.get_inputs()
        }
        actual_outputs = {
            item.name: _as_fixed_shape(item.shape, f"output {item.name!r}")
            for item in self.session.get_outputs()
        }
        if actual_inputs != self.INPUT_SHAPES:
            raise ValueError(
                "ONNX input contract does not match Unitree-G1-Stair: "
                f"expected={self.INPUT_SHAPES}, actual={actual_inputs}"
            )
        if actual_outputs != self.OUTPUT_SHAPES:
            raise ValueError(
                "ONNX output contract does not match Unitree-G1-Stair: "
                f"expected={self.OUTPUT_SHAPES}, actual={actual_outputs}"
            )

        joint_names = tuple(self.metadata.get("joint_names", "").split(","))
        if joint_names != ACTUATED_JOINT_NAMES:
            raise ValueError(
                "ONNX joint metadata does not match the full-body 29-DoF Stair task. "
                f"Expected {ACTUATED_JOINT_NAMES}, got {joint_names}."
            )
        observation_names = tuple(self.metadata.get("observation_names", "").split(","))
        expected_observations = (
            "base_ang_vel",
            "projected_gravity",
            "command",
            "joint_pos",
            "joint_vel",
            "actions",
        )
        if observation_names != expected_observations:
            raise ValueError(
                "Unexpected ONNX observation ordering: "
                f"expected={expected_observations}, actual={observation_names}"
            )

        metadata_default = _parse_csv_floats(self.metadata, "default_joint_pos")
        if metadata_default.shape != DEFAULT_JOINT_POS.shape or not np.allclose(
            metadata_default, DEFAULT_JOINT_POS, atol=1.0e-3
        ):
            raise ValueError(
                "ONNX default_joint_pos metadata does not match this task."
            )
        metadata_scale = _parse_csv_floats(self.metadata, "action_scale")
        if metadata_scale.shape != ACTION_SCALE.shape or not np.allclose(
            metadata_scale, ACTION_SCALE, atol=1.0e-3
        ):
            raise ValueError("ONNX action_scale metadata does not match this task.")

    def reset(self) -> None:
        self.estimator_h = np.zeros((1, 1, 128), dtype=np.float32)
        self.estimator_c = np.zeros((1, 1, 128), dtype=np.float32)
        self.recurrent_h = np.zeros((1, 1, 256), dtype=np.float32)

    def __call__(self, observation: np.ndarray, depth: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32).reshape(1, OBS_DIM)
        depth = np.asarray(depth, dtype=np.float32).reshape(
            1, 1, DEPTH_HEIGHT, DEPTH_WIDTH
        )
        outputs = self.session.run(
            [
                "actions",
                "estimator_h_out",
                "estimator_c_out",
                "recurrent_h_out",
            ],
            {
                "obs": observation,
                "depth": depth,
                "estimator_h_in": self.estimator_h,
                "estimator_c_in": self.estimator_c,
                "recurrent_h_in": self.recurrent_h,
            },
        )
        output_names = (
            "actions",
            "estimator_h_out",
            "estimator_c_out",
            "recurrent_h_out",
        )
        checked_outputs: list[np.ndarray] = []
        for name, output in zip(output_names, outputs, strict=True):
            array = np.asarray(output, dtype=np.float32)
            expected_shape = self.OUTPUT_SHAPES[name]
            if array.shape != expected_shape:
                raise ValueError(
                    f"Policy output {name!r} has shape {array.shape}; expected {expected_shape}."
                )
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(f"Policy produced non-finite output {name!r}.")
            checked_outputs.append(array)
        action, self.estimator_h, self.estimator_c, self.recurrent_h = checked_outputs
        return action.reshape(NUM_ACTIONS)


def _quat_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"Invalid quaternion: {quaternion}")
    w, x, y, z = quaternion
    return np.array(
        [
            [
                w * w + x * x - y * y - z * z,
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                w * w - x * x + y * y - z * z,
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                w * w - x * x - y * y + z * z,
            ],
        ],
        dtype=np.float64,
    )


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
