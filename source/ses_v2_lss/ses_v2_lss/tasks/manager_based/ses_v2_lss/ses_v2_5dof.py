# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.
Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os
import math

SES_V2_5DOF_CFG = ArticulationCfg(
    # Where is the USD file for this robot?
    spawn=sim_utils.UsdFileCfg(       
        usd_path=f"source/assets/Collected_SES_V2_LSS_5DOF/SES_V2_LSS_5DOF_NEW.usd", 
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True
        ),
    ),

    # What is its initial position of the robot, and its joints?
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.01),
        rot=(0.5, 0.5, 0.5, 0.5),
        joint_pos={
            "lss_arm_joint_1": math.radians(180),
            "lss_arm_joint_2": math.radians(25),
            "lss_arm_joint_3": math.radians(45),
            "lss_arm_joint_4": math.radians(65),
            "lss_arm_joint_6": math.radians(0),

            "lss_arm_joint_5": 0.0,
            "lss_arm_joint_7": 0.0,
        },
    ),

    # What parts of the robot move, and how stiff / damped are they?
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["lss_arm_joint_1", "lss_arm_joint_2", "lss_arm_joint_3", "lss_arm_joint_4",
                              "lss_arm_joint_6"],
            effort_limit_sim=28.16,
            velocity_limit=19.0,    
            velocity_limit_sim=19.0,
            stiffness=3200.0,
            damping=320.0,
        ),

        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["lss_arm_joint_5", "lss_arm_joint_7"],
            effort_limit_sim=5.816,
            velocity_limit=19.0,
            velocity_limit_sim=19.0,
            stiffness=1600.0,
            damping=320.0,
        ),
    }
)
