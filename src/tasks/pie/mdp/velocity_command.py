from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    wrap_to_pi,
)

if TYPE_CHECKING:
    import viser

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
    cfg: UniformVelocityCommandCfg

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError("heading_command=True but ranges.heading is set to None.")
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            raise ValueError("ranges.heading is set but heading_command=False.")

        self.robot: Entity = env.scene[cfg.entity_name]

        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.heading_error = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.is_standing_env = torch.zeros_like(self.is_heading_env)

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # Set by create_gui() when the viewer is active.
        self._joystick_enabled: viser.GuiCheckboxHandle | None = None
        self._joystick_sliders: list[viser.GuiSliderHandle] = []
        self._joystick_get_env_idx: Callable[[], int] | None = None

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self) -> None:
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2],
                dim=-1,
            )
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(
                self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
            )
            / max_command_step
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        if self.cfg.heading_command:
            assert self.cfg.ranges.heading is not None
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = (
                r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
            )
        self.is_standing_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        )

        init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
        init_vel_env_ids = env_ids[init_vel_mask]
        if len(init_vel_env_ids) > 0:
            root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
            root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
            lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
            lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
            root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
            root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
            root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
            root_state = torch.cat(
                [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
            )
            self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

    def _update_command(self) -> None:
        if self.cfg.heading_command:
            self.heading_error = wrap_to_pi(
                self.heading_target - self.robot.data.heading_w
            )
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * self.heading_error[env_ids],
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    # GUI.

    def create_gui(
        self,
        name: str,
        server: "viser.ViserServer",
        get_env_idx: Callable[[], int],
    ) -> None:
        """Create velocity joystick sliders in the Viser viewer."""
        from viser import Icon

        ranges = self.cfg.ranges

        axes = [
            ("lin_vel_x", ranges.lin_vel_x[1]),
            ("lin_vel_y", ranges.lin_vel_y[1]),
            ("ang_vel_z", ranges.ang_vel_z[1]),
        ]
        sliders: list = []

        with server.gui.add_folder(name.capitalize()):
            enabled = server.gui.add_checkbox("Enable", initial_value=False)

            for label, max_val in axes:
                max_input = server.gui.add_slider(
                    f"Max {label}",
                    initial_value=max_val,
                    step=0.1,
                    min=0.1,
                    max=10.0,
                )
                slider = server.gui.add_slider(
                    label,
                    min=-max_val,
                    max=max_val,
                    step=0.05,
                    initial_value=0.0,
                )

                @max_input.on_update
                def _(_ev, _s=slider, _m=max_input) -> None:
                    _s.min = -_m.value
                    _s.max = _m.value

                sliders.append(slider)

            zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

            @zero_btn.on_click
            def _(_) -> None:
                for s in sliders:
                    s.value = 0.0

        # Store GUI state for compute() override.
        self._joystick_enabled = enabled
        self._joystick_sliders = sliders
        self._joystick_get_env_idx = get_env_idx

    def compute(self, dt: float) -> None:
        super().compute(dt)
        if self._joystick_enabled is not None and self._joystick_enabled.value:
            assert self._joystick_get_env_idx is not None
            idx = self._joystick_get_env_idx()
            for i, s in enumerate(self._joystick_sliders):
                self.vel_command_b[idx, i] = s.value

    # Visualization.

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Draw velocity command and actual velocity arrows."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        cmds = self.command.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_w = self.robot.data.root_link_quat_w
        base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
        lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
        ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

        scale = self.cfg.viz.scale
        z_offset = self.cfg.viz.z_offset

        for batch in env_indices:
            base_pos_w = base_pos_ws[batch]
            base_mat_w = base_mat_ws[batch]
            cmd = cmds[batch]
            lin_vel_b = lin_vel_bs[batch]
            ang_vel_b = ang_vel_bs[batch]

            # Skip if robot appears uninitialized (at origin).
            if np.linalg.norm(base_pos_w) < 1e-6:
                continue

            # Helper to transform local to world coordinates.
            def local_to_world(
                vec: np.ndarray,
                pos: np.ndarray = base_pos_w,
                mat: np.ndarray = base_mat_w,
            ) -> np.ndarray:
                return pos + mat @ vec

            # Command linear velocity arrow (blue).
            cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
            cmd_lin_to = local_to_world(
                (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
            )
            visualizer.add_arrow(
                cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
            )

            # Command angular velocity arrow (green).
            cmd_ang_from = cmd_lin_from
            cmd_ang_to = local_to_world(
                (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
            )
            visualizer.add_arrow(
                cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
            )

            # Actual linear velocity arrow (cyan).
            act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
            act_lin_to = local_to_world(
                (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0]))
                * scale
            )
            visualizer.add_arrow(
                act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
            )

            # Actual angular velocity arrow (light green).
            act_ang_from = act_lin_from
            act_ang_to = local_to_world(
                (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
            )
            visualizer.add_arrow(
                act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
            )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
    entity_name: str
    heading_command: bool = False
    heading_control_stiffness: float = 1.0
    rel_standing_envs: float = 0.0
    rel_heading_envs: float = 1.0
    init_velocity_prob: float = 0.0

    @dataclass
    class Ranges:
        lin_vel_x: tuple[float, float]
        lin_vel_y: tuple[float, float]
        ang_vel_z: tuple[float, float]
        heading: tuple[float, float] | None = None

    ranges: Ranges

    @dataclass
    class VizCfg:
        z_offset: float = 0.2
        scale: float = 0.5

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
        return UniformVelocityCommand(self, env)

    def __post_init__(self):
        if self.heading_command and self.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but "
                "the `ranges.heading` parameter is set to None."
            )


YawModeProbabilities = tuple[float, float, float]


def _yaw_modes(
    flat_mask: torch.Tensor,
    mode_samples: torch.Tensor,
    flat_probabilities: YawModeProbabilities,
    obstacle_probabilities: YawModeProbabilities,
) -> torch.Tensor:
    """Select 0=straight, 1=gentle turn, or 2=full turn for each environment."""
    straight_probability = torch.where(
        flat_mask,
        flat_probabilities[0],
        obstacle_probabilities[0],
    )
    gentle_cutoff = straight_probability + torch.where(
        flat_mask,
        flat_probabilities[1],
        obstacle_probabilities[1],
    )

    modes = torch.zeros_like(mode_samples, dtype=torch.long)
    modes[mode_samples >= straight_probability] = 1
    modes[mode_samples >= gentle_cutoff] = 2
    return modes


class TerrainAwareVelocityCommand(UniformVelocityCommand):
    """Sample yaw modes according to the environment's current terrain type."""

    cfg: TerrainAwareVelocityCommandCfg

    def create_gui(
        self,
        name: str,
        server: "viser.ViserServer",
        get_env_idx: Callable[[], int],
    ) -> None:
        """Create Viser controls while preserving axes fixed by the task config."""
        from viser import Icon

        ranges = self.cfg.ranges
        axes = (
            ("lin_vel_x", ranges.lin_vel_x),
            ("lin_vel_y", ranges.lin_vel_y),
            ("ang_vel_z", ranges.ang_vel_z),
        )
        sliders = []
        mutable_sliders = []

        with server.gui.add_folder(name.capitalize()):
            enabled = server.gui.add_checkbox("Enable", initial_value=False)

            for label, value_range in axes:
                lower, upper = value_range
                fixed = lower == upper
                display_max = max(abs(lower), abs(upper), 0.1)

                if fixed:
                    # MJLab 1.2.0 creates a "Max" slider with min=0.1 and
                    # initial_value=0.0 for a disabled axis, which Viser rejects.
                    # Keep the handle in the three-axis list so the inherited
                    # compute() mapping remains x/y/yaw, but make it immutable.
                    slider = server.gui.add_slider(
                        label,
                        min=-display_max,
                        max=display_max,
                        step=0.05,
                        initial_value=lower,
                        disabled=True,
                    )
                else:
                    max_input = server.gui.add_slider(
                        f"Max {label}",
                        initial_value=display_max,
                        step=0.1,
                        min=0.1,
                        max=max(10.0, display_max),
                    )
                    slider = server.gui.add_slider(
                        label,
                        min=-display_max,
                        max=display_max,
                        step=0.05,
                        initial_value=0.0,
                    )

                    @max_input.on_update
                    def _(_ev, _slider=slider, _max_input=max_input) -> None:
                        _slider.min = -_max_input.value
                        _slider.max = _max_input.value

                    mutable_sliders.append(slider)

                sliders.append(slider)

            zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

            @zero_btn.on_click
            def _(_) -> None:
                for slider in mutable_sliders:
                    slider.value = 0.0

        self._joystick_enabled = enabled
        self._joystick_sliders = sliders
        self._joystick_get_env_idx = get_env_idx

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)

        flat_mask = self._flat_terrain_mask(env_ids)
        mode_samples = torch.rand(len(env_ids), device=self.device)
        modes = _yaw_modes(
            flat_mask,
            mode_samples,
            self.cfg.flat_yaw_probabilities,
            self.cfg.obstacle_yaw_probabilities,
        )

        obstacle_env_ids = env_ids[~flat_mask]
        if len(obstacle_env_ids) > 0:
            self.vel_command_b[obstacle_env_ids, 0] = torch.empty(
                len(obstacle_env_ids), device=self.device
            ).uniform_(*self.cfg.obstacle_lin_vel_x)

        # Straight mode has an exactly-zero yaw command. This discrete mass at
        # zero cannot be produced by sampling a continuous uniform range.
        self.vel_command_b[env_ids, 2] = 0.0

        gentle_env_ids = env_ids[modes == 1]
        if len(gentle_env_ids) > 0:
            self.vel_command_b[gentle_env_ids, 2] = torch.empty(
                len(gentle_env_ids), device=self.device
            ).uniform_(*self.cfg.gentle_ang_vel_z)

        full_env_ids = env_ids[modes == 2]
        if len(full_env_ids) > 0:
            self.vel_command_b[full_env_ids, 2] = torch.empty(
                len(full_env_ids), device=self.device
            ).uniform_(*self.cfg.ranges.ang_vel_z)

    def _flat_terrain_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return which selected environments are assigned to flat terrain."""
        terrain = self._env.scene.terrain
        if terrain is None or terrain.cfg.terrain_generator is None:
            return torch.ones(len(env_ids), dtype=torch.bool, device=self.device)

        terrain_generator = terrain.cfg.terrain_generator
        terrain_names = list(terrain_generator.sub_terrains)
        if "flat" not in terrain_names:
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        proportions = torch.tensor(
            [cfg.proportion for cfg in terrain_generator.sub_terrains.values()],
            device=self.device,
            dtype=torch.float,
        )
        cumulative_proportions = torch.cumsum(proportions / proportions.sum(), dim=0)
        column_fractions = (
            torch.arange(
                terrain_generator.num_cols,
                device=self.device,
                dtype=torch.float,
            )
            / terrain_generator.num_cols
            + 0.001
        )
        column_terrain_ids = torch.sum(
            column_fractions.unsqueeze(1) >= cumulative_proportions.unsqueeze(0),
            dim=1,
        ).clamp(max=len(terrain_names) - 1)
        env_terrain_ids = column_terrain_ids[terrain.terrain_types[env_ids]]
        return env_terrain_ids == terrain_names.index("flat")


@dataclass(kw_only=True)
class TerrainAwareVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for terrain-conditioned straight/gentle/full yaw modes."""

    # Ordered as (straight, gentle, full). Obstacles deliberately assign zero
    # probability to full turns; full-range yaw commands are exclusive to flat.
    flat_yaw_probabilities: YawModeProbabilities = (0.40, 0.30, 0.30)
    obstacle_yaw_probabilities: YawModeProbabilities = (0.80, 0.20, 0.00)
    obstacle_lin_vel_x: tuple[float, float] = (0.20, 1.50)
    gentle_ang_vel_z: tuple[float, float] = (-0.30, 0.30)

    def build(self, env: ManagerBasedRlEnv) -> TerrainAwareVelocityCommand:
        return TerrainAwareVelocityCommand(self, env)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_probabilities("flat", self.flat_yaw_probabilities)
        self._validate_probabilities("obstacle", self.obstacle_yaw_probabilities)
        if self.obstacle_lin_vel_x[0] > self.obstacle_lin_vel_x[1]:
            raise ValueError("obstacle_lin_vel_x must be an increasing range.")
        if not (
            self.ranges.lin_vel_x[0]
            <= self.obstacle_lin_vel_x[0]
            <= self.obstacle_lin_vel_x[1]
            <= self.ranges.lin_vel_x[1]
        ):
            raise ValueError("obstacle_lin_vel_x must lie inside ranges.lin_vel_x.")
        if self.gentle_ang_vel_z[0] > self.gentle_ang_vel_z[1]:
            raise ValueError("gentle_ang_vel_z must be an increasing range.")
        if not (
            self.ranges.ang_vel_z[0]
            <= self.gentle_ang_vel_z[0]
            <= self.gentle_ang_vel_z[1]
            <= self.ranges.ang_vel_z[1]
        ):
            raise ValueError("gentle_ang_vel_z must lie inside ranges.ang_vel_z.")

    @staticmethod
    def _validate_probabilities(
        terrain_name: str,
        probabilities: YawModeProbabilities,
    ) -> None:
        if any(probability < 0.0 for probability in probabilities):
            raise ValueError(f"{terrain_name} yaw probabilities must be non-negative.")
        if abs(sum(probabilities) - 1.0) > 1.0e-6:
            raise ValueError(f"{terrain_name} yaw probabilities must sum to 1.")
