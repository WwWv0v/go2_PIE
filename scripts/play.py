"""Replay a Unitree Go2 PIE checkpoint in MJLab."""

# ruff: noqa: E402 -- ensure this repository wins over other editable `src` packages.

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


TASK_ID = "Unitree-Go2-PIE"


def _network_depth_for_display(
    env: ManagerBasedRlEnv,
    env_idx: int,
    group_name: str,
    term_name: str,
) -> torch.Tensor:
    """Return one environment's exact network-depth tensor as ``(C, H, W)``."""
    obs_buf = env.obs_buf
    if group_name not in obs_buf:
        raise ValueError(
            f"Network-depth observation group {group_name!r} is unavailable; "
            f"available groups: {tuple(obs_buf)}."
        )
    depth = obs_buf[group_name]
    if isinstance(depth, dict):
        if term_name not in depth:
            raise ValueError(
                f"Network-depth group {group_name!r} has no term {term_name!r}."
            )
        depth = depth[term_name]
    if not isinstance(depth, torch.Tensor):
        raise TypeError(
            f"Network-depth observation must be a tensor, got {type(depth).__name__}."
        )
    if env_idx < 0 or env_idx >= depth.shape[0]:
        raise IndexError(
            f"Network-depth env index {env_idx} is outside batch size {depth.shape[0]}."
        )
    if depth.ndim == 4:
        return depth[env_idx]
    if depth.ndim != 2:
        raise ValueError(
            "Expected network depth with shape (N,C,H,W) or flattened (N,D), "
            f"got {tuple(depth.shape)}."
        )

    term_cfg = env.observation_manager.get_term_cfg(group_name, term_name)
    params = term_cfg.params
    history_length = int(
        params.get("frame_history_length", max(1, int(term_cfg.history_length)))
    )
    sensor_name = params.get("sensor_name")
    if not isinstance(sensor_name, str):
        raise ValueError(
            f"Network-depth term {group_name!r}/{term_name!r} has no sensor_name."
        )
    sensor = env.scene[sensor_name]
    height = int(sensor.cfg.height)
    width = int(sensor.cfg.width)

    crop_region = params.get("crop_region")
    if crop_region is not None:
        crop_up, crop_down, crop_left, crop_right = crop_region
        height -= int(crop_up) + int(crop_down)
        width -= int(crop_left) + int(crop_right)
    else:
        height -= int(params.get("crop_up", 0)) + int(params.get("crop_down", 0))
        width -= int(params.get("crop_left", 0)) + int(params.get("crop_right", 0))

    expected_size = history_length * height * width
    if height <= 0 or width <= 0 or depth.shape[-1] != expected_size:
        raise ValueError(
            f"Cannot reshape network depth {tuple(depth.shape)} as "
            f"({history_length}, {height}, {width}); expected {expected_size} values."
        )
    return depth[env_idx].reshape(history_length, height, width)


class NetworkDepthViserPlayViewer(ViserPlayViewer):
    """Viser playback with the newest exact policy-depth frame."""

    def __init__(
        self,
        env,
        policy,
        network_depth_group: str = "camera",
        network_depth_term: str = "front_depth",
    ) -> None:
        self._network_depth_group = network_depth_group
        self._network_depth_term = network_depth_term
        super().__init__(env, policy)

    def setup(self) -> None:
        super().setup()
        depth = self._get_network_depth()
        channels, height, width = depth.shape
        self._network_depth_display_scale = max(1, 256 // width)
        display_height = height * self._network_depth_display_scale
        display_width = width * self._network_depth_display_scale

        with self._server.gui.add_folder("Policy Network Depth"):
            self._server.gui.add_markdown(
                "Newest channel of the exact policy input after cropping, invalid "
                "sample handling, Gaussian blur, and normalization."
            )
            self._network_depth_handle = self._server.gui.add_image(
                image=np.zeros((display_height, display_width, 3), dtype=np.uint8),
                label=(
                    f"{self._network_depth_group} newest ({height}×{width}, "
                    f"channel {channels}/{channels})"
                ),
                format="png",
            )
        self._update_network_depth_image()

    def _get_network_depth(self) -> torch.Tensor:
        return _network_depth_for_display(
            self.env.unwrapped,
            int(self._scene.env_idx),
            self._network_depth_group,
            self._network_depth_term,
        )

    def _update_network_depth_image(self) -> None:
        depth = self._get_network_depth().detach().float().clamp(0.0, 1.0)
        newest = (depth[-1].cpu().numpy() * 255.0).astype(np.uint8)
        if self._network_depth_display_scale > 1:
            newest = np.repeat(
                np.repeat(newest, self._network_depth_display_scale, axis=0),
                self._network_depth_display_scale,
                axis=1,
            )
        self._network_depth_handle.image = np.repeat(newest[..., None], 3, axis=-1)

    def sync_env_to_viewer(self) -> None:
        super().sync_env_to_viewer()
        self._update_network_depth_image()


def _set_play_terrain_origin(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    terrain_level: int,
) -> None:
    """Select a fixed generated-terrain difficulty row for play resets."""
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        return

    device = terrain.terrain_levels.device
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    if terrain_level < 0 or terrain_level >= num_rows:
        raise ValueError(
            f"terrain_level={terrain_level} is out of range [0, {num_rows - 1}]"
        )
    levels = torch.full(
        (len(env_ids),), terrain_level, device=device, dtype=torch.long
    )
    types = torch.randint(0, num_cols, (len(env_ids),), device=device)
    terrain.terrain_levels[env_ids] = levels
    terrain.terrain_types[env_ids] = types
    terrain.env_origins[env_ids] = terrain.terrain_origins[levels, types]


def _configure_play_terrain_level(env_cfg, terrain_level: int | None) -> None:
    """Configure deterministic difficulty rows before creating the play scene."""
    if terrain_level is None:
        return

    terrain_cfg = env_cfg.scene.terrain
    terrain_generator = None if terrain_cfg is None else terrain_cfg.terrain_generator
    if terrain_generator is None:
        raise ValueError("--terrain-level requires a generated terrain.")

    terrain_generator.curriculum = True
    terrain_generator.num_rows = max(terrain_generator.num_rows, 10)
    if terrain_level < 0 or terrain_level >= terrain_generator.num_rows:
        raise ValueError(
            f"terrain_level={terrain_level} is out of range "
            f"[0, {terrain_generator.num_rows - 1}]"
        )

    terrain_generator.sub_terrains = {
        name: replace(sub_cfg, proportion=1.0)
        for name, sub_cfg in terrain_generator.sub_terrains.items()
    }
    terrain_generator.num_cols = len(terrain_generator.sub_terrains)

    env_cfg.events.pop("randomize_terrain", None)
    env_cfg.events = {
        "select_terrain": EventTermCfg(
            func=_set_play_terrain_origin,
            mode="reset",
            params={"terrain_level": terrain_level},
        ),
        **env_cfg.events,
    }
    print(f"[INFO] Play terrain selection: fixed level={terrain_level}")


@dataclass(frozen=True)
class PlayConfig:
    checkpoint_file: str
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = False
    terrain_level: int | None = None
    """Fixed generated-terrain difficulty row for play resets."""
    network_depth_vis: bool = False
    """Show the newest exact network-depth input in the Viser GUI."""
    network_depth_group: str = "camera"
    """Observation group containing the policy depth input."""
    network_depth_term: str = "front_depth"
    """Observation term containing the policy depth input."""


def run_play(cfg: PlayConfig) -> None:
    configure_torch_backends()
    checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    if cfg.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = cfg.num_envs
    _configure_play_terrain_level(env_cfg, cfg.terrain_level)
    if cfg.no_terminations:
        env_cfg.terminations = {}

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)

        viewer = cfg.viewer
        if viewer == "auto":
            has_display = bool(
                os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            )
            viewer = "native" if has_display else "viser"
        if cfg.network_depth_vis and viewer != "viser":
            raise ValueError("--network-depth-vis requires --viewer viser.")

        print(
            f"[INFO] task={TASK_ID}, checkpoint={checkpoint}, "
            f"device={device}, viewer={viewer}"
        )
        if viewer == "native":
            NativeMujocoViewer(wrapped, policy).run()
        elif cfg.network_depth_vis:
            print(
                "[INFO] Viser network-depth visualization enabled: "
                f"group={cfg.network_depth_group}, term={cfg.network_depth_term}"
            )
            NetworkDepthViserPlayViewer(
                wrapped,
                policy,
                network_depth_group=cfg.network_depth_group,
                network_depth_term=cfg.network_depth_term,
            ).run()
        else:
            ViserPlayViewer(wrapped, policy).run()
    finally:
        env.close()


def main() -> None:
    run_play(tyro.cli(PlayConfig))


if __name__ == "__main__":
    main()
