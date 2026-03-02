# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
import carb
import torch

NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
"""Path to the root directory on the Nucleus Server."""

NVIDIA_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
"""Path to the root directory on the NVIDIA Nucleus Server."""

ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
"""Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg


from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: ski
from .ses_v2_5dof import SES_V2_5DOF_CFG


##
# Scene definition
##
@configclass
class SesV2LssSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene."""
    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # robot
    robot = SES_V2_5DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=5000.0),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    '''
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["lss_arm_joint_1", "lss_arm_joint_2", "lss_arm_joint_3", "lss_arm_joint_4", 
                     "lss_arm_joint_6"],
        scale=0.5,
        use_default_offset=True,
        debug_vis=True
    )
    '''
    # Set actions for the specific robot type (franka)
    arm_action: ActionTerm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["lss_arm_joint_1", "lss_arm_joint_2", "lss_arm_joint_3", "lss_arm_joint_4", 
                     "lss_arm_joint_6"],
        body_name="Wrist_Up",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["lss_arm_joint_5", "lss_arm_joint_7"],
        open_command_expr={"lss_arm_joint_5": -0.78 * 3, "lss_arm_joint_7": 0.78 * 3},
        close_command_expr={"lss_arm_joint_5": 0.0, "lss_arm_joint_7": 0.0}
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="Wrist_Up",  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.25, 0.25), pos_y=(0.2, 0.2), pos_z=(0.0, 0.0), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(-1.0, -1.0)
        ),
    )
    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.75, 1.25),
            "velocity_range": (0.0, 0.0),
        },
    )

    #reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            #"pose_range": {"x": (0.15, 0.45), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "pose_range": {"x": (0.35, 0.35), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # task terms
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.16}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


def root_x_below_threshold(env, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # Extract the asset
    asset = env.scene[asset_cfg.name]
    
    # Get World Positions (Shape: [num_envs, 3])
    world_pos = asset.data.root_pos_w

    # Get Environment Origins (Shape: [num_envs, 3])
    env_origins = env.scene.env_origins
    
    # Calculate Local Position
    local_pos = world_pos - env_origins
    
    # Check if Local X is below the threshold
    return local_pos[:, 0] < threshold


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    object_out_of_bounds_x = DoneTerm(
        func=root_x_below_threshold, 
        params={"threshold": 0.15, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )


##
# Environment configuration
##
@configclass
class SesV2LssEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: SesV2LssSceneCfg = SesV2LssSceneCfg(num_envs=2000, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = CurriculumCfg()
    
    '''
    # Set Cube as object
    scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.25, 0.0, 0.055], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            mass_props=MassPropertiesCfg(
                mass=0.05, # Set mass in kg (e.g., 100g for a small cube)
            ),
            rigid_props=RigidBodyPropertiesCfg(
                linear_damping=0.01,   # Prevents sliding forever
                angular_damping=0.05,  # Prevents spinning forever
                max_depenetration_velocity=1.0, # Reduces "explosive" bounces on collision
            ),
            rigid_material=RigidBodyMaterialCfg(
                restitution=0.0,      # 0.0 means no bounce (perfectly inelastic)
                static_friction=0.5,
                dynamic_friction=0.5,
            )
        ),
    )
    '''
    scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.25, 0.0, 0.055], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"source/assets/Collected_SES_V2_LSS_5DOF/dex_cube_instanceable_01.usd", 
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                disable_gravity=False,
                linear_damping=0.01,   # Prevents sliding forever
                angular_damping=0.05,  # Prevents spinning forever
                max_depenetration_velocity=1.0, # Reduces "explosive" bounces on collision
            ),
        ),
    )
    
    # Listens to the required transforms
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    scene.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Base_Top",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/Wrist_Up",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
        ],
    )
    
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 50.0
        
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        
        # simulation settings
        self.sim.dt = 1 / 240
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class SesV2LssEnvCfg_PLAY(SesV2LssEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.sim.dt = 1 / 240
        
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
