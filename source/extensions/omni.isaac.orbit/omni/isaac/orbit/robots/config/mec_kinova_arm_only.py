# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Ridgeback-Manipulation robots.

The following configurations are available:

* :obj:`RIDGEBACK_FRANKA_PANDA_CFG`: Clearpath Ridgeback base with Franka Emika arm

Reference: https://github.com/ridgeback/ridgeback_manipulation
"""


from omni.isaac.orbit.actuators.config.franka import PANDA_HAND_MIMIC_GROUP_CFG
from omni.isaac.orbit.actuators.config.robotiq import ROBOTIQ_2F85_MIMIC_GROUP_CFG
from omni.isaac.orbit.actuators.group import ActuatorControlCfg, ActuatorGroupCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from ..single_arm import SingleArmManipulatorCfg
from ..mobile_manipulator import MobileManipulatorCfg
import numpy as np


USD_PATH = f"/home/nikepupu/Desktop/mec_kinova.usd"


KINOVA_CFG = SingleArmManipulatorCfg(
    meta_info=SingleArmManipulatorCfg.MetaInfoCfg(
        usd_path=USD_PATH,
        # base_num_dof=3,
        arm_num_dof=7,
        tool_num_dof=2,
        tool_sites_names=[
            "right_inner_finger_pad",
            "left_inner_finger_pad",
            # "_f85_instanceable/robotiq_arg2f_base_link/right_inner_knuckle_joint",
            # "_f85_instanceable/robotiq_arg2f_base_link/left_inner_knuckle_joint",
            # "_f85_instanceable/robotiq_arg2f_base_link/right_outer_knuckle_joint",
        ],
    ),

    data_info = SingleArmManipulatorCfg.DataInfoCfg(
        enable_jacobian=True
    ),
    
    init_state=SingleArmManipulatorCfg.InitialStateCfg(
        dof_pos={
            #arm
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "joint_6": 0.0,
            "joint_7": 0.0,

            #hand
            "finger_joint": 0.0 , 
            ".*_inner_knuckle_joint": 0.0 , 
            ".*_inner_finger_joint": 0.0, 
            ".*right_outer_knuckle_joint":0.0
        },
        dof_vel={".*": 0.0},
    ),

    ee_info=SingleArmManipulatorCfg.EndEffectorFrameCfg(
        body_name="robotiq_arg2f_base_link", pos_offset=(0.0, 0.0, 0.1034), rot_offset=(1.0, 0.0, 0.0, 0.0)
    ),

    # rigid_props=SingleArmManipulatorCfg.RigidBodyPropertiesCfg(
    #     disable_gravity=True,
    # ),
    
    actuator_groups={
        "shoulder": ActuatorGroupCfg(
            dof_names=["joint_[1-4]"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=87.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": 800.0},
                damping={".*": 40.0},
                dof_pos_offset={
                    "joint_1": 0.0,
                    "joint_2": 0.0,
                    "joint_3": 0.0,
                    "joint_4": 0.0,
                },
            ),
        ),
        "forearm": ActuatorGroupCfg(
            dof_names=["joint_[5-7]"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=12.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": 800.0},
                damping={".*": 40.0},
                dof_pos_offset={"joint_5": 0.0, "joint_6": 0, "joint_7": 0},
            ),
        ),
        "hand": ROBOTIQ_2F85_MIMIC_GROUP_CFG
    },
)