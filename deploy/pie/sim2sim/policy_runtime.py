"""Strict ONNX runtime wrapper for Unitree-Go2-PIE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .contract import (
    ACTION_SCALE,
    ACTUATED_JOINT_NAMES,
    DEFAULT_JOINT_POS,
    DEPTH_HISTORY_SHAPE,
    JOINT_DAMPING,
    JOINT_STIFFNESS,
    NUM_ACTIONS,
    OBSERVATION_NAMES,
    OBS_DIM,
    PROPRIO_HISTORY_DIM,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (REPO_ROOT / path).resolve()


def resolve_onnx_path(checkpoint_file: str | Path) -> tuple[Path, Path | None]:
    """Resolve policy.onnx, optionally from the newest model_N.pt alias."""
    requested = _resolve_path(checkpoint_file)
    checkpoint_path: Path | None
    if requested.suffix.lower() == ".onnx":
        onnx_path = requested
        checkpoint_path = None
    elif requested.suffix.lower() == ".pt":
        checkpoint_path = requested
        onnx_path = requested.with_name("policy.onnx")
    else:
        raise ValueError(
            "--checkpoint-file must point to policy.onnx or model_N.pt; "
            f"got {requested}."
        )

    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        try:
            requested_step = int(checkpoint_path.stem.removeprefix("model_"))
        except ValueError as exc:
            raise ValueError(
                "A .pt alias must be named model_N.pt; otherwise pass ONNX directly."
            ) from exc
        available_steps = []
        for candidate in checkpoint_path.parent.glob("model_*.pt"):
            try:
                available_steps.append(int(candidate.stem.removeprefix("model_")))
            except ValueError:
                continue
        if available_steps and requested_step != max(available_steps):
            raise ValueError(
                f"{checkpoint_path.name} is not the newest checkpoint in its run. "
                "policy.onnx is overwritten on save, so pass an explicitly exported "
                "ONNX for older checkpoints."
            )

    if not onnx_path.is_file():
        suffix = (
            " The sibling policy.onnx export is required." if checkpoint_path else ""
        )
        raise FileNotFoundError(f"ONNX policy not found: {onnx_path}.{suffix}")
    return onnx_path, checkpoint_path


def _fixed_shape(value: Sequence[Any], name: str) -> tuple[int, ...]:
    if not all(isinstance(dim, int) and dim > 0 for dim in value):
        raise ValueError(f"ONNX {name} must have a fixed shape, got {value}.")
    return tuple(int(dim) for dim in value)


def _metadata_floats(metadata: dict[str, str], key: str) -> np.ndarray:
    try:
        values = [float(item) for item in metadata[key].split(",") if item]
    except KeyError as exc:
        raise ValueError(f"ONNX metadata is missing {key!r}.") from exc
    return np.asarray(values, dtype=np.float32)


class PieOnnxPolicy:
    """ONNX policy wrapper that owns PIE's exported GRU state."""

    INPUT_SHAPES = {
        "proprio": (1, OBS_DIM),
        "proprio_history": (1, PROPRIO_HISTORY_DIM),
        "depth_history": (1, *DEPTH_HISTORY_SHAPE),
        "memory_h_in": (1, 1, 128),
    }
    OUTPUT_SHAPES = {
        "actions": (1, NUM_ACTIONS),
        "memory_h_out": (1, 1, 128),
    }

    def __init__(self, path: Path, provider: str = "auto") -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required to execute the exported PIE policy."
            ) from exc

        options = ort.SessionOptions()
        options.log_severity_level = 3
        available = ort.get_available_providers()
        if provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDA provider requested but unavailable; "
                    f"available providers: {available}."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "auto" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.path = Path(path)
        self.session = ort.InferenceSession(
            str(self.path), sess_options=options, providers=providers
        )
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self._validate_contract()
        self.reset()

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def _validate_contract(self) -> None:
        actual_inputs = {
            value.name: _fixed_shape(value.shape, f"input {value.name!r}")
            for value in self.session.get_inputs()
        }
        actual_outputs = {
            value.name: _fixed_shape(value.shape, f"output {value.name!r}")
            for value in self.session.get_outputs()
        }
        if actual_inputs != self.INPUT_SHAPES:
            raise ValueError(
                "ONNX input contract does not match Unitree-Go2-PIE: "
                f"expected={self.INPUT_SHAPES}, actual={actual_inputs}."
            )
        if actual_outputs != self.OUTPUT_SHAPES:
            raise ValueError(
                "ONNX output contract does not match Unitree-Go2-PIE: "
                f"expected={self.OUTPUT_SHAPES}, actual={actual_outputs}."
            )

        joint_names = tuple(self.metadata.get("joint_names", "").split(","))
        if joint_names != ACTUATED_JOINT_NAMES:
            raise ValueError(
                "ONNX joint order does not match PIE: "
                f"expected={ACTUATED_JOINT_NAMES}, actual={joint_names}."
            )
        observation_names = tuple(self.metadata.get("observation_names", "").split(","))
        if observation_names != OBSERVATION_NAMES:
            raise ValueError(
                "ONNX observation order does not match PIE: "
                f"expected={OBSERVATION_NAMES}, actual={observation_names}."
            )
        command_names = tuple(self.metadata.get("command_names", "").split(","))
        if command_names != ("twist",):
            raise ValueError(f"Unexpected ONNX command metadata: {command_names}.")

        metadata_default = _metadata_floats(self.metadata, "default_joint_pos")
        if metadata_default.shape != DEFAULT_JOINT_POS.shape or not np.allclose(
            metadata_default, DEFAULT_JOINT_POS, atol=1.0e-3
        ):
            raise ValueError("ONNX default_joint_pos does not match PIE.")
        metadata_scale = _metadata_floats(self.metadata, "action_scale")
        # MJLab serializes a scalar JointPositionAction scale as one value,
        # while a per-joint configuration is serialized as twelve values.
        if metadata_scale.shape == (1,):
            metadata_scale = np.full_like(ACTION_SCALE, metadata_scale.item())
        if metadata_scale.shape != ACTION_SCALE.shape or not np.allclose(
            metadata_scale, ACTION_SCALE, atol=1.0e-3
        ):
            raise ValueError("ONNX action_scale does not match PIE.")
        metadata_stiffness = _metadata_floats(self.metadata, "joint_stiffness")
        if metadata_stiffness.shape != JOINT_STIFFNESS.shape or not np.allclose(
            metadata_stiffness, JOINT_STIFFNESS, atol=1.0e-3
        ):
            raise ValueError("ONNX joint_stiffness does not match PIE.")
        metadata_damping = _metadata_floats(self.metadata, "joint_damping")
        if metadata_damping.shape != JOINT_DAMPING.shape or not np.allclose(
            metadata_damping, JOINT_DAMPING, atol=1.0e-3
        ):
            raise ValueError("ONNX joint_damping does not match PIE.")

    def reset(self) -> None:
        self.memory_h = np.zeros((1, 1, 128), dtype=np.float32)

    def __call__(
        self,
        proprio: np.ndarray,
        proprio_history: np.ndarray,
        depth_history: np.ndarray,
    ) -> np.ndarray:
        inputs = {
            "proprio": np.asarray(proprio, dtype=np.float32).reshape(1, OBS_DIM),
            "proprio_history": np.asarray(proprio_history, dtype=np.float32).reshape(
                1, PROPRIO_HISTORY_DIM
            ),
            "depth_history": np.asarray(depth_history, dtype=np.float32).reshape(
                1, *DEPTH_HISTORY_SHAPE
            ),
            "memory_h_in": self.memory_h,
        }
        outputs = self.session.run(["actions", "memory_h_out"], inputs)
        action = np.asarray(outputs[0], dtype=np.float32)
        memory_h = np.asarray(outputs[1], dtype=np.float32)
        if action.shape != self.OUTPUT_SHAPES["actions"]:
            raise ValueError(f"Policy actions have invalid shape {action.shape}.")
        if memory_h.shape != self.OUTPUT_SHAPES["memory_h_out"]:
            raise ValueError(f"Policy memory has invalid shape {memory_h.shape}.")
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(memory_h)):
            raise FloatingPointError("PIE policy produced NaN or infinity.")
        self.memory_h = memory_h
        return action.reshape(NUM_ACTIONS)


__all__ = ["PieOnnxPolicy", "resolve_onnx_path"]
