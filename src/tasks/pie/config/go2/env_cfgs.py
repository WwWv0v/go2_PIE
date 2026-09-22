"""Standalone Unitree Go2 PIE stair-locomotion environment."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    CameraSensorCfg,
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.assets.robots import get_go2_robot_cfg
from src.tasks.pie import mdp
from src.tasks.pie.mdp.velocity_command import TerrainAwareVelocityCommandCfg
from src.tasks.pie.terrains import PIE_TERRAINS_CFG


_FOOT_NAMES = ("FR", "FL", "RR", "RL")
_FOOT_GEOMS = tuple(f"{name}_foot_collision" for name in _FOOT_NAMES)
_FOOT_RAYS = tuple(f"{name}_foot_ray" for name in _FOOT_NAMES)

# The source PIE renderer casts a 106 x 60 image with an 87-degree horizontal
# field of view, then removes ten columns from both sides before giving the
# resulting 86 x 60 image to the policy.
_PIE_DEPTH_RAW_WIDTH = 106
_PIE_DEPTH_HEIGHT = 60
_PIE_DEPTH_CROP_LEFT = 10
_PIE_DEPTH_CROP_RIGHT = 10
_PIE_DEPTH_HORIZONTAL_FOV_DEG = 87.0


def _horizontal_to_vertical_fov(horizontal_fov: float) -> float:
    return 2.0 * math.degrees(
        math.atan(
            math.tan(math.radians(horizontal_fov) * 0.5)
            * _PIE_DEPTH_HEIGHT
            / _PIE_DEPTH_RAW_WIDTH
        )
    )


_PIE_DEPTH_FOVY_DEG = _horizontal_to_vertical_fov(_PIE_DEPTH_HORIZONTAL_FOV_DEG)
_PIE_DEPTH_FOVY_RANDOM_RANGE = (
    _horizontal_to_vertical_fov(_PIE_DEPTH_HORIZONTAL_FOV_DEG - 1.0)
    - _PIE_DEPTH_FOVY_DEG,
    _horizontal_to_vertical_fov(_PIE_DEPTH_HORIZONTAL_FOV_DEG + 1.0)
    - _PIE_DEPTH_FOVY_DEG,
)

# The manufacturer ranges allow a thigh to rotate into a mechanically valid but
# unusable inverted-leg branch.  Keep the original lower limits and give PIE a
# generous locomotion/stair upper limit that excludes that branch.
_PIE_THIGH_JOINT_RANGES = {
    "FL_thigh_joint": (-1.5708, 2.2),
    "FR_thigh_joint": (-1.5708, 2.2),
    "RL_thigh_joint": (-0.5236, 2.2),
    "RR_thigh_joint": (-0.5236, 2.2),
}


def _get_pie_go2_spec():
    robot_cfg = get_go2_robot_cfg()
    assert robot_cfg.spec_fn is not None
    spec = robot_cfg.spec_fn()
    for joint_name, joint_range in _PIE_THIGH_JOINT_RANGES.items():
        spec.joint(joint_name).range = joint_range
    return spec


def _get_pie_go2_robot_cfg():
    return replace(get_go2_robot_cfg(), spec_fn=_get_pie_go2_spec)


def _proprio_terms(*, noisy: bool) -> dict[str, ObservationTermCfg]:
    """The 45-dimensional proprioceptive vector from the PIE paper."""
    return {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2) if noisy else None,
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05) if noisy else None,
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            noise=Unoise(n_min=-0.01, n_max=0.01) if noisy else None,
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            noise=Unoise(n_min=-1.5, n_max=1.5) if noisy else None,
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }


def unitree_go2_pie_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the self-contained, stair-focused Go2 PIE environment."""
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        # Inclusive sampling gives 18 x 11 = 198 privileged height samples.
        pattern=GridPatternCfg(size=(1.7, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0, 1, 2),
        debug_vis=False,
    )

    foot_rays = tuple(
        RayCastSensorCfg(
            name=f"{foot}_foot_ray",
            frame=ObjRef(type="site", name=foot, entity="robot"),
            ray_alignment="world",
            pattern=GridPatternCfg(size=(0.02, 0.02), resolution=0.02),
            max_distance=0.6,
            exclude_parent_body=True,
            include_geom_groups=(0, 1, 2),
            debug_vis=False,
        )
        for foot in _FOOT_NAMES
    )

    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=_FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
        history_length=4,
    )
    nonfoot_ground = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=r".*_collision\d*$",
            exclude=_FOOT_GEOMS,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    # MuJoCo cameras look along local -Z with local +Y as image-up.  This
    # quaternion points the optical axis forward and 20 degrees down while
    # keeping image-up aligned with the robot's +Z axis.  MuJoCo takes a
    # vertical FOV, so convert PIE's 87-degree horizontal FOV using its raw
    # 106 x 60 render aspect ratio.  The observation then applies PIE's
    # ten-column crop on each side.
    depth_camera = CameraSensorCfg(
        name="front_depth",
        parent_body="robot/base_link",
        # Match the source PIE front-camera transform and keep the optical
        # center just ahead of the Go2 nose geometry.
        pos=(0.345, 0.0, 0.07),
        quat=(0.5792280, 0.4055798, -0.4055798, -0.5792280),
        fovy=_PIE_DEPTH_FOVY_DEG,
        width=_PIE_DEPTH_RAW_WIDTH,
        height=_PIE_DEPTH_HEIGHT,
        data_types=("depth",),
        enabled_geom_groups=(0, 1, 2),
        use_shadows=False,
        use_textures=False,
    )

    noisy_proprio = _proprio_terms(noisy=True)
    clean_proprio = _proprio_terms(noisy=False)
    observations = {
        "actor": ObservationGroupCfg(
            terms=noisy_proprio,
            concatenate_terms=True,
            enable_corruption=not play,
            history_length=1,
        ),
        "proprio_history": ObservationGroupCfg(
            terms=copy.deepcopy(noisy_proprio),
            concatenate_terms=True,
            enable_corruption=not play,
            history_length=10,
            flatten_history_dim=True,
        ),
        "camera": ObservationGroupCfg(
            terms={
                "front_depth": ObservationTermCfg(
                    func=mdp.DepthHistory,
                    params={
                        "sensor_name": depth_camera.name,
                        "cutoff_distance": 3.0,
                        "crop_left": _PIE_DEPTH_CROP_LEFT,
                        "crop_right": _PIE_DEPTH_CROP_RIGHT,
                        "gaussian_blur": (3, 1.0),
                        "frame_history_length": 2,
                        "update_period_steps": 5,
                    },
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
            history_length=None,
            flatten_history_dim=True,
        ),
        "critic": ObservationGroupCfg(
            terms={
                **copy.deepcopy(clean_proprio),
                "base_lin_vel": ObservationTermCfg(
                    func=mdp.builtin_sensor,
                    params={"sensor_name": "robot/imu_lin_vel"},
                ),
                "height_scan": ObservationTermCfg(
                    func=envs_mdp.height_scan,
                    params={"sensor_name": terrain_scan.name},
                    scale=1.0 / terrain_scan.max_distance,
                ),
            },
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
        "velocity_target": ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(
                    func=mdp.builtin_sensor,
                    params={"sensor_name": "robot/imu_lin_vel"},
                )
            },
            enable_corruption=False,
            history_length=1,
        ),
        "height_target": ObservationGroupCfg(
            terms={
                "height_scan": ObservationTermCfg(
                    func=envs_mdp.height_scan,
                    params={"sensor_name": terrain_scan.name},
                    scale=1.0 / terrain_scan.max_distance,
                )
            },
            enable_corruption=False,
            history_length=1,
        ),
        "foot_clearance_target": ObservationGroupCfg(
            terms={
                "foot_clearance": ObservationTermCfg(
                    func=mdp.foot_clearance,
                    params={"sensor_names": _FOOT_RAYS, "max_clearance": 0.6},
                    scale=1.0 / 0.6,
                )
            },
            enable_corruption=False,
            history_length=1,
        ),
        # PiePPO replaces this current clean vector with o_(t+1) before storage.
        "successor_target": ObservationGroupCfg(
            terms=copy.deepcopy(clean_proprio),
            enable_corruption=False,
            history_length=1,
        ),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.25,
            use_default_offset=True,
        )
    }
    commands: dict[str, CommandTermCfg] = {
        "twist": TerrainAwareVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,
            heading_command=False,
            debug_vis=True,
            flat_yaw_probabilities=(0.40, 0.30, 0.30),
            obstacle_yaw_probabilities=(0.80, 0.20, 0.00),
            obstacle_lin_vel_x=(0.20, 1.50),
            gentle_ang_vel_z=(-0.30, 0.30),
            ranges=TerrainAwareVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.2, 1.2),
                heading=None,
            ),
        )
    }

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.15, 0.15),
                "velocity_range": (-0.1, 0.1),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 8.0),
            params={
                "velocity_range": {
                    "x": (-0.4, 0.4),
                    "y": (-0.4, 0.4),
                    "z": (-0.25, 0.25),
                    "roll": (-0.3, 0.3),
                    "pitch": (-0.3, 0.3),
                    "yaw": (-0.4, 0.4),
                }
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOMS),
                "operation": "abs",
                "ranges": (0.2, 1.2),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
                "operation": "add",
                "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        ),
        "base_payload": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                # PIE treats this as a point payload attached at the trunk COM.
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
                "operation": "add",
                "ranges": (-1.0, 2.0),
            },
        ),
        "pd_gains": EventTermCfg(
            mode="startup",
            func=dr.pd_gains,
            params={
                # Leave actuator ids as slice(None) so grouped Go2 actuators are
                # randomized as actuator objects rather than as twelve control ids.
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": (0.9, 1.1),
                "kd_range": (0.9, 1.1),
                "operation": "scale",
            },
        ),
        "motor_strength": EventTermCfg(
            mode="startup",
            func=dr.effort_limits,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "effort_limit_range": (0.9, 1.1),
                "operation": "scale",
            },
        ),
        "camera_position": EventTermCfg(
            mode="startup",
            func=dr.cam_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", camera_names=(depth_camera.name,)),
                "operation": "add",
                "ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)},
            },
        ),
        "camera_pitch": EventTermCfg(
            mode="startup",
            func=dr.cam_quat,
            params={
                "asset_cfg": SceneEntityCfg("robot", camera_names=(depth_camera.name,)),
                "pitch_range": (-math.radians(1.0), math.radians(1.0)),
            },
        ),
        "camera_fovy": EventTermCfg(
            mode="startup",
            func=dr.cam_fovy,
            params={
                "asset_cfg": SceneEntityCfg("robot", camera_names=(depth_camera.name,)),
                "operation": "add",
                "ranges": _PIE_DEPTH_FOVY_RANDOM_RANGE,
            },
        ),
    }

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_body_planar_velocity,
            weight=1.5,
            params={"command_name": "twist", "sigma": 0.25},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_body_yaw_velocity,
            weight=0.5,
            params={"command_name": "twist", "sigma": 0.25},
        ),
        "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0),
        "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
        "orientation": RewardTermCfg(
            func=mdp.body_orientation_l2,
            # The paper-aligned -1.0 coefficient is too weak once the gait and
            # stair-traversal rewards are active: a 30 degree lean only costs
            # 0.25 reward before weighting. Make upright posture a primary
            # objective instead of a nearly free velocity-tracking shortcut.
            weight=-5.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
        ),
        "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
        "power": RewardTermCfg(
            func=mdp.joint_power_l1,
            weight=-2.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "collision": RewardTermCfg(
            func=mdp.collision_count,
            weight=-10.0,
            params={"sensor_name": nonfoot_ground.name, "force_threshold": 0.1},
        ),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "smoothness": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.01),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            # Do not let rollouts spend time learning from a severely tilted
            # posture which is already unrecoverable for stair traversal.
            params={"limit_angle": math.radians(55.0)},
        ),
        "illegal_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_name": nonfoot_ground.name, "force_threshold": 10.0},
        ),
    }
    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "twist"},
        ),
        **{
            f"terrain_{terrain_name}_level": CurriculumTermCfg(
                func=mdp.terrain_curriculum_stat,
                params={"stat_name": f"{terrain_name}_level"},
            )
            for terrain_name in PIE_TERRAINS_CFG.sub_terrains
        },
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=replace(PIE_TERRAINS_CFG),
                max_init_terrain_level=2,
            ),
            entities={"robot": _get_pie_go2_robot_cfg()},
            sensors=(
                terrain_scan,
                *foot_rays,
                feet_ground,
                nonfoot_ground,
                depth_camera,
            ),
            num_envs=1 if play else 4096,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=1.8,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=48,
            njmax=1800,
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=500,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.events.pop("push_robot", None)
        cfg.events.pop("camera_position", None)
        cfg.events.pop("camera_pitch", None)
        cfg.events.pop("camera_fovy", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )
        assert cfg.scene.terrain is not None
        assert cfg.scene.terrain.terrain_generator is not None
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

    # Diagonal-trot contact timing and stance-foot slip suppression.
    cfg.rewards["foot_gait"] = RewardTermCfg(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.6,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "threshold": 0.56,
            "command_threshold": 0.1,
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["foot_slip"] = RewardTermCfg(
        func=mdp.feet_slip,
        weight=-0.25,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_NAMES),
        },
    )

    # Targeted posture shaping.
    cfg.rewards["joint_pos_limits"] = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["hip_deviation_l2"] = RewardTermCfg(
        func=mdp.joint_deviation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_joint",))},
    )
    cfg.rewards["base_height"] = RewardTermCfg(
        func=mdp.base_height_reward,
        weight=0.2,
        params={
            "sensor_name": "feet_ground_contact",
            "target_height": 0.27,
            "support_force_threshold": 1.0,
            "base_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
            "foot_cfg": SceneEntityCfg("robot", site_names=_FOOT_NAMES),
        },
    )

    return cfg
