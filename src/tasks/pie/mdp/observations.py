from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mjlab.managers import ManagerTermBase
from mjlab.sensor import CameraSensor, RayCastSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def camera_depth(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    cutoff_distance: float = 3.0,
    min_depth: float = 0.05,
    crop_left: int = 0,
    crop_right: int = 0,
    gaussian_blur: tuple[int, float] | None = None,
) -> torch.Tensor:
    """Return cropped, normalized depth in ``(B, 1, H, W)`` layout."""
    sensor: CameraSensor = env.scene[sensor_name]
    depth = sensor.data.depth
    assert depth is not None, f"Camera '{sensor_name}' has no depth output."
    depth = depth.permute(0, 3, 1, 2)
    if crop_left < 0 or crop_right < 0:
        raise ValueError("Depth crop widths must be nonnegative.")
    if crop_left + crop_right >= depth.shape[-1]:
        raise ValueError(
            f"Depth crop ({crop_left}, {crop_right}) removes all "
            f"{depth.shape[-1]} image columns."
        )
    if crop_left or crop_right:
        crop_stop = depth.shape[-1] - crop_right if crop_right else None
        depth = depth[..., crop_left:crop_stop]
    # MuJoCo-Warp leaves a zero in the depth buffer when a ray misses.  Real
    # depth transports may represent the same condition as NaN or infinity.
    # Treat every non-positive/non-finite sample as "no obstacle in range".
    invalid = (depth <= 0.0) | ~torch.isfinite(depth)
    depth = torch.where(
        invalid,
        torch.full_like(depth, cutoff_distance),
        depth,
    )
    if gaussian_blur is not None:
        kernel_size, sigma = gaussian_blur
        if kernel_size <= 0 or kernel_size % 2 == 0 or sigma <= 0.0:
            raise ValueError(
                "Gaussian blur requires an odd kernel size and positive sigma."
            )
        coords = (
            torch.arange(kernel_size, device=depth.device, dtype=depth.dtype)
            - (kernel_size - 1) / 2
        )
        kernel_1d = torch.exp(-0.5 * (coords / sigma).square())
        kernel_1d /= kernel_1d.sum()
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).view(
            1, 1, kernel_size, kernel_size
        )
        padding = kernel_size // 2
        depth = F.conv2d(
            F.pad(depth, (padding,) * 4, mode="reflect"),
            kernel_2d,
        )
    return depth.clamp(min=min_depth, max=cutoff_distance) / cutoff_distance


class DepthHistory(ManagerTermBase):

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        del cfg
        super().__init__(env)
        self._history: torch.Tensor | None = None
        self._last_update_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._needs_reset = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._needs_reset[env_ids] = True
        self._last_update_step[env_ids] = -1

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        cutoff_distance: float = 3.0,
        min_depth: float = 0.05,
        crop_left: int = 0,
        crop_right: int = 0,
        gaussian_blur: tuple[int, float] | None = None,
        frame_history_length: int = 2,
        update_period_steps: int = 5,
    ) -> torch.Tensor:
        if frame_history_length <= 0:
            raise ValueError("Depth frame history length must be positive.")
        if update_period_steps <= 0:
            raise ValueError("Depth update period must be positive.")

        episode_step = env.episode_length_buf.to(device=self.device, dtype=torch.long)
        periodic_update = episode_step.remainder(update_period_steps) == 0
        update_mask = self._needs_reset | (
            periodic_update & (episode_step != self._last_update_step)
        )

        if self._history is None or bool(torch.any(update_mask)):
            frame = camera_depth(
                env,
                sensor_name=sensor_name,
                cutoff_distance=cutoff_distance,
                min_depth=min_depth,
                crop_left=crop_left,
                crop_right=crop_right,
                gaussian_blur=gaussian_blur,
            )

            if self._history is None:
                self._history = frame.unsqueeze(1).repeat(
                    1, frame_history_length, 1, 1, 1
                )
                update_mask = torch.ones_like(update_mask)
                reset_mask = update_mask
            else:
                expected_shape = (
                    self.num_envs,
                    frame_history_length,
                    *frame.shape[1:],
                )
                if tuple(self._history.shape) != expected_shape:
                    raise ValueError(
                        "Depth history shape changed from "
                        f"{tuple(self._history.shape)} to {expected_shape}."
                    )
                reset_mask = update_mask & self._needs_reset
                shift_mask = update_mask & ~reset_mask
                if bool(torch.any(shift_mask)):
                    self._history[shift_mask, :-1] = self._history[
                        shift_mask, 1:
                    ].clone()
                    self._history[shift_mask, -1] = frame[shift_mask]
                if bool(torch.any(reset_mask)):
                    self._history[reset_mask] = (
                        frame[reset_mask]
                        .unsqueeze(1)
                        .repeat(1, frame_history_length, 1, 1, 1)
                    )

            self._last_update_step[update_mask] = episode_step[update_mask]
            self._needs_reset[update_mask] = False

        assert self._history is not None
        return self._history.reshape(self.num_envs, -1)


def foot_clearance(
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    max_clearance: float = 0.6,
) -> torch.Tensor:
    clearances = []
    for sensor_name in sensor_names:
        sensor: RayCastSensor = env.scene[sensor_name]
        clearance = sensor.data.pos_w[:, 2:3] - sensor.data.hit_pos_w[..., 2]
        miss = sensor.data.distances < 0
        clearance = torch.where(
            miss, torch.full_like(clearance, max_clearance), clearance
        )
        clearances.append(clearance.amin(dim=1, keepdim=True))
    return torch.cat(clearances, dim=1).clamp_(0.0, max_clearance)
