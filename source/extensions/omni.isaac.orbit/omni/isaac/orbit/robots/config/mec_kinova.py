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
from omni.isaac.orbit.actuators.config.robotiq import CUSTOM_ROBOTIQ_2F85_MIMIC_GROUP_CFG
from ..mobile_manipulator import MobileManipulatorCfg
import numpy as np


USD_PATH = f"/home/nikepupu/Desktop/mec_kinova_with_base_flatten.usd"


MEC_KINOVA_CFG = MobileManipulatorCfg(
    meta_info=MobileManipulatorCfg.MetaInfoCfg(
        usd_path=USD_PATH,
        base_num_dof=3,
        arm_num_dof=7,
        tool_num_dof=2,
        tool_sites_names=[
            "right_inner_finger",
            "left_inner_finger",
        ],
    ),

    data_info = MobileManipulatorCfg.DataInfoCfg(
        enable_jacobian=True
    ),
    
    init_state=MobileManipulatorCfg.InitialStateCfg(
        dof_pos={
            # base
            "base_y_base_x": 0.0,
            "base_theta_base_y": 0.0,
            "base_link_base_theta": 0.0,
            #arm
            "Actuator1": 0.0,
            "Actuator2": 0.0,
            "Actuator3": 0.0,
            "Actuator4": 0.0,
            "Actuator5": 0.0,
            "Actuator6": 0.0,
            "Actuator7": 0.0,

            #hand
            "finger_joint": 0.0 , 
            ".*_inner_knuckle_joint": 0.0 , 
            ".*_inner_finger_joint": 0.0, 
            ".*right_outer_knuckle_joint":0.0
        },
        dof_vel={".*": 0.0},
    ),

    ee_info=MobileManipulatorCfg.EndEffectorFrameCfg(
        body_name="robotiq_85_base_link", pos_offset=(0.0, 0.0, 0.149), rot_offset=(1.0, 0.0, 0.0, 0.0)
    ),

    rigid_props=MobileManipulatorCfg.RigidBodyPropertiesCfg(
        disable_gravity=True,
    ),
    
    actuator_groups={
         "base": ActuatorGroupCfg(
            dof_names=["base_y_base_x", "base_theta_base_y", "base_link_base_theta"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=1000.0),
            control_cfg=ActuatorControlCfg(command_types=["v_abs"], stiffness={".*": 0.0}, damping={".*": 1e5}),
        ),

        "shoulder": ActuatorGroupCfg(
            dof_names=["Actuator[1-4]"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=87.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": 800.0},
                damping={".*": 40.0},
                dof_pos_offset={
                    "Actuator1": 0.0,
                    "Actuator2": 0.0,
                    "Actuator3": 0.0,
                    "Actuator4": 0.0,
                },
            ),
        ),
        "forearm": ActuatorGroupCfg(
            dof_names=["Actuator[5-7]"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=12.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": 800.0},
                damping={".*": 40.0},
                dof_pos_offset={"Actuator5": 0.0, "Actuator6": 0, "Actuator7": 0},
            ),
        ),
        "hand": CUSTOM_ROBOTIQ_2F85_MIMIC_GROUP_CFG
    },
)