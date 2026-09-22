from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_body_planar_velocity(
    env: ManagerBasedRlEnv,
    sigma: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track the commanded body-frame planar velocity from PIE Table I."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    error = torch.sum(
        torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
    )
    return torch.exp(-error / sigma)


def track_body_yaw_velocity(
    env: ManagerBasedRlEnv,
    sigma: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track the commanded body-frame yaw rate from PIE Table I."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-error / sigma)


def track_world_forward_velocity(
    env: ManagerBasedRlEnv,
    sigma: float,
    command_name: str,
    lateral_weight: float = 2.0,
    low_speed_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track a world-frame +x command while suppressing world-frame y drift."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    actual_w = asset.data.root_link_lin_vel_w
    error_x = command[:, 0] - actual_w[:, 0]
    error_y = command[:, 1] - actual_w[:, 1]
    low_speed = (torch.norm(command[:, :2], dim=1) < low_speed_threshold).float()
    l1_error = torch.abs(error_x) + lateral_weight * torch.abs(error_y)
    l2_error_sq = torch.square(error_x) + lateral_weight * torch.square(error_y)

    env.extras["log"]["Metrics/world_vel_x"] = torch.mean(actual_w[:, 0])
    env.extras["log"]["Metrics/world_vel_y_abs"] = torch.mean(torch.abs(actual_w[:, 1]))
    return torch.exp(-(low_speed * l1_error + (1.0 - low_speed) * l2_error_sq) / sigma)


def heading_alignment_reward(
    env: ManagerBasedRlEnv,
    sigma: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward alignment with the command's fixed world-frame heading."""
    asset: Entity = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    if command_term is not None and hasattr(command_term, "heading_target"):
        target_heading = command_term.heading_target
    else:
        target_heading = torch.zeros(env.num_envs, device=env.device)
    heading_error = wrap_to_pi(target_heading - asset.data.heading_w)
    env.extras["log"]["Metrics/heading_error_abs"] = torch.mean(
        torch.abs(heading_error)
    )
    return torch.exp(-torch.square(heading_error) / sigma)


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize vertical base velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 2].square()


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize roll and pitch angular velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b[:, :2].square().sum(dim=1)


def joint_power_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mechanical power penalty used by the original PIE reward."""
    asset: Entity = env.scene[asset_cfg.name]
    torque = asset.data.actuator_force[:, asset_cfg.actuator_ids]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque) * torch.abs(velocity), dim=1)


def collision_count(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 0.1,
) -> torch.Tensor:
    """Count non-foot links that collide with terrain."""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force = torch.linalg.vector_norm(data.force_history, dim=-1)
        return (force > force_threshold).any(dim=-1).sum(dim=1).float()
    if data.force is not None:
        force = torch.linalg.vector_norm(data.force, dim=-1)
        return (force > force_threshold).sum(dim=1).float()
    assert data.found is not None
    return (data.found > 0).sum(dim=1).float()


def joint_deviation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize squared joint displacement from the nominal standing pose."""
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(error), dim=1)


def base_height_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    support_force_threshold: float = 1.0,
    base_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    foot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward base height relative to the average height of supporting feet."""
    asset: Entity = env.scene[base_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = contact_sensor.data
    assert sensor_data.force is not None

    foot_z = asset.data.site_pos_w[:, foot_cfg.site_ids, 2]
    force_z = torch.abs(sensor_data.force[..., 2])
    if force_z.shape[1] != foot_z.shape[1]:
        num_feet = min(force_z.shape[1], foot_z.shape[1])
        force_z = force_z[:, :num_feet]
        foot_z = foot_z[:, :num_feet]

    support = force_z > support_force_threshold
    support_count = torch.sum(support.float(), dim=1)
    support_z = torch.sum(foot_z * support.float(), dim=1) / torch.clamp(
        support_count, min=1.0
    )
    fallback_z = torch.min(foot_z, dim=1).values
    support_z = torch.where(support_count > 0.0, support_z, fallback_z)

    if base_cfg.body_ids:
        base_z = asset.data.body_link_pos_w[:, base_cfg.body_ids, 2].squeeze(1)
    else:
        base_z = asset.data.root_link_pos_w[:, 2]
    height_error = torch.abs((base_z - support_z) - target_height)
    return torch.exp(-10.0 * height_error)


def body_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize roll and pitch of the selected body."""
    asset: Entity = env.scene[asset_cfg.name]
    if asset_cfg.body_ids:
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
        projected_gravity_b = quat_apply_inverse(body_quat_w, asset.data.gravity_vec_w)
    else:
        projected_gravity_b = asset.data.projected_gravity_b
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)


def feet_gait(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """Reward agreement between diagonal-trot phase and measured foot contact."""
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(
        1, -1
    )
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = leg_phase < threshold
    reward = (is_stance == is_contact).float().mean(dim=1)

    command = env.command_manager.get_command(command_name)
    if command is not None:
        total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        reward *= (total_command > command_threshold).float()
    return reward


def feet_slip(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize planar velocity of feet that are in contact."""
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    active = (total_command > command_threshold).float()

    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    slip_speed = torch.norm(foot_vel_xy, dim=-1)
    cost = torch.sum(torch.square(slip_speed) * in_contact, dim=1) * active
    contact_count = torch.sum(in_contact)
    env.extras["log"]["Metrics/slip_velocity_mean"] = torch.sum(
        slip_speed * in_contact
    ) / torch.clamp(contact_count, min=1)
    return cost
