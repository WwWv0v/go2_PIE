"""Self-contained fixed terrain-15 course and policy goal geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


ASSET_ROOT = Path(__file__).resolve().parent / "assets"
SCENE_FILE = ASSET_ROOT / "scene_parkour.xml"
TERRAIN_IDS = ("15",)
# This identifier is embedded in the V6 ONNX metadata. It names the training
# course version; no runtime gate or success evaluator is implemented here.
COURSE_VERSION = "straight_centerline_gate_v2"

# These are the fixed terrain-15 values encoded in assets/scene_parkour.xml.
GOAL_POS_W = np.asarray(
    (8.0009363085021157, 0.0, 0.73072707653045654), dtype=np.float64
)
GOAL_QUAT_WXYZ = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)


@dataclass(frozen=True)
class CourseStage:
    terrain_id: str
    terrain_file: Path
    goal_pos_w: np.ndarray
    goal_quat_wxyz: np.ndarray


def validate_policy_course(metadata: dict[str, str]) -> None:
    if not SCENE_FILE.is_file():
        raise FileNotFoundError(f"Parkour sim2sim scene is missing: {SCENE_FILE}")
    if metadata.get("course_version") != COURSE_VERSION:
        raise ValueError("ONNX course version differs from Parkour sim2sim.")
    ids = tuple(filter(None, metadata.get("terrain_ids", "").split(",")))
    if "15" not in ids:
        raise ValueError("ONNX metadata does not contain terrain 15.")


def parse_sequence(value: str | Sequence[str]) -> tuple[str, ...]:
    source = value.split(",") if isinstance(value, str) else value
    result = tuple(item.strip().removeprefix("climb_") for item in source if item.strip())
    if result != TERRAIN_IDS:
        raise ValueError("This Parkour sim2sim package contains only terrain 15.")
    return result


def build_straight_course(sequence: str | Sequence[str]) -> tuple[CourseStage, ...]:
    parse_sequence(sequence)
    return (
        CourseStage(
            terrain_id="15",
            terrain_file=SCENE_FILE,
            goal_pos_w=GOAL_POS_W.copy(),
            goal_quat_wxyz=GOAL_QUAT_WXYZ.copy(),
        ),
    )


def _rotate_yaw(vector: np.ndarray, yaw: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    result = value.copy()
    c, s = math.cos(yaw), math.sin(yaw)
    result[..., 0] = c * value[..., 0] - s * value[..., 1]
    result[..., 1] = s * value[..., 0] + c * value[..., 1]
    return result


def goal_direction_b(
    goal: np.ndarray, torso_pos: np.ndarray, torso_quat: np.ndarray
) -> np.ndarray:
    w, x, y, z = np.asarray(torso_quat, dtype=np.float64) / np.linalg.norm(torso_quat)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    delta = _rotate_yaw(np.asarray(goal)[:2] - np.asarray(torso_pos)[:2], -yaw)
    return (delta / max(float(np.linalg.norm(delta)), 1.0e-6)).astype(np.float32)


__all__ = [
    "CourseStage",
    "SCENE_FILE",
    "TERRAIN_IDS",
    "build_straight_course",
    "goal_direction_b",
    "parse_sequence",
    "validate_policy_course",
]
