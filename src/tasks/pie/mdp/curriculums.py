"""PIE-specific terrain curriculum diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


def _terrain_level_moves(
    distance: torch.Tensor,
    command_speed: torch.Tensor,
    episode_length_s: float,
    terrain_length: float,
    terminated: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return terrain level changes, treating non-timeout termination as failure."""
    successful = ~terminated
    move_up = (distance > terrain_length * 0.5) & successful

    insufficient_progress = distance < command_speed * episode_length_s * 0.5
    move_down = (terminated | insufficient_progress) & ~move_up
    return move_up, move_down


def terrain_levels_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
    """Advance the curriculum and cache the mean level of each terrain type."""
    asset: Entity = env.scene[asset_cfg.name]

    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None

    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    command_speed = torch.norm(command[env_ids, :2], dim=1)
    move_up, move_down = _terrain_level_moves(
        distance=distance,
        command_speed=command_speed,
        episode_length_s=env.max_episode_length_s,
        terrain_length=terrain_generator.size[0],
        terminated=env.termination_manager.terminated[env_ids],
    )
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    terrain_columns = terrain.terrain_types
    terrain_names = list(terrain_generator.sub_terrains)
    proportions = torch.tensor(
        [cfg.proportion for cfg in terrain_generator.sub_terrains.values()],
        device=levels.device,
        dtype=torch.float,
    )
    cumulative_proportions = torch.cumsum(proportions / proportions.sum(), dim=0)
    column_fractions = (
        torch.arange(
            terrain_generator.num_cols, device=levels.device, dtype=torch.float
        )
        / terrain_generator.num_cols
        + 0.001
    )
    column_terrain_ids = torch.sum(
        column_fractions.unsqueeze(1) >= cumulative_proportions.unsqueeze(0), dim=1
    ).clamp(max=len(terrain_names) - 1)
    env_terrain_ids = column_terrain_ids[terrain_columns]

    stats: dict[str, torch.Tensor] = {}
    for terrain_id, terrain_name in enumerate(terrain_names):
        mask = env_terrain_ids == terrain_id
        stats[f"{terrain_name}_level"] = (
            levels[mask].mean()
            if torch.any(mask)
            else torch.zeros((), device=levels.device)
        )
    env._pie_terrain_curriculum_stats = stats
    return torch.mean(levels)


def terrain_curriculum_stat(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    stat_name: str,
) -> torch.Tensor:
    """Return one cached per-terrain level without changing the curriculum."""
    del env_ids
    stats = getattr(env, "_pie_terrain_curriculum_stats", None)
    if stats is None or stat_name not in stats:
        return torch.zeros((), device=env.device)
    return stats[stat_name]
