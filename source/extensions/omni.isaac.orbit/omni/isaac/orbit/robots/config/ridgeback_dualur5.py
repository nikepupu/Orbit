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
from omni.isaac.orbit.actuators.group import ActuatorControlCfg, ActuatorGroupCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from ..mobile_manipulator import MobileManipulatorCfg
import numpy as np


_RIDGEBACK_DUAL_UR5_ARM_USD = f"/home/nikepupu/Desktop/arnoldv2/working_version5.usd"



RIDGEBACK_DUALARM_UR5_CFG = MobileManipulatorCfg(
    meta_info=MobileManipulatorCfg.MetaInfoCfg(
        usd_path=_RIDGEBACK_DUAL_UR5_ARM_USD,
        base_num_dof=3,
        arm_num_dof=0,
        tool_num_dof=14,
        tool_sites_names=[
            "left_gripper_finger_1_link_0",
            "left_gripper_finger_1_link_1",
            "left_gripper_finger_1_link_2",
            "left_gripper_finger_1_link_3",

            "left_gripper_finger_2_link_0",
            "left_gripper_finger_2_link_1",
            "left_gripper_finger_2_link_2",
            "left_gripper_finger_2_link_3",

            "left_gripper_finger_middle_link_0",
            "left_gripper_finger_middle_link_1",
            "left_gripper_finger_middle_link_2",
            "left_gripper_finger_middle_link_3",
            "left_gripper_palm",
            "left_gripper_flange"
        ],
    ),
    
    init_state=MobileManipulatorCfg.InitialStateCfg(
        dof_pos={
            # base
            "dummy_base_prismatic_y_joint": 0.0,
            "dummy_base_prismatic_x_joint": 0.0,
            "dummy_base_revolute_z_joint": 0.0,
            # ur5 arm
            # np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0
            # "left_arm_shoulder_pan_joint": np.pi/2,
            # "left_arm_shoulder_lift_joint": -np.pi/2,
            # "left_arm_elbow_joint": -np.pi/2,
            # "left_arm_wrist_1_joint":-np.pi / 2,
            # "left_arm_wrist_2_joint": np.pi / 2,
            # "left_arm_wrist_3_joint": 0.0,
            # tool
            # "left_gripper*_joint_*": 0.035,
        },
        dof_vel={".*": 0.0},
    ),
    ee_info=MobileManipulatorCfg.EndEffectorFrameCfg(
        body_name="base_link", pos_offset=(0.0, 0.0, 0.1034), rot_offset=(1.0, 0.0, 0.0, 0.0)
    ),
    actuator_groups={
        "base": ActuatorGroupCfg(
            dof_names=["dummy_base.*"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=1000.0),
            control_cfg=ActuatorControlCfg(command_types=["v_abs"], stiffness={".*": 0.0}, damping={".*": 1e5}),
        ),
        # "ur5_shoulder": ActuatorGroupCfg(
        #     dof_names=[ "left_arm_shoulder_pan_joint",
        #                "left_arm_shoulder_lift_joint", 
        #                 "left_arm_elbow_joint",
        #             ],
        #     model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=87.0),
        #     control_cfg=ActuatorControlCfg(
        #         command_types=["p_abs"],
        #         stiffness={".*": 800.0},
        #         damping={".*": 40.0},
        #         dof_pos_offset={
        #             "left_arm_shoulder_pan_joint": 0.0,
        #             "left_arm_shoulder_lift_joint": 0.0,
        #             "left_arm_elbow_joint": 0.0,
        #         },
        #     ),
        # ),
        # "ur5_forearm": ActuatorGroupCfg(
        #     dof_names=["left_arm_wrist_[1-3]_joint"],
        #     model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=12.0),
        #     control_cfg=ActuatorControlCfg(
        #         command_types=["p_abs"],
        #         stiffness={".*": 800.0},
        #         damping={".*": 40.0},
        #         dof_pos_offset={"left_arm_wrist_1_joint": 0.0, 
        #                         "left_arm_wrist_2_joint": 0.0, "left_arm_wrist_3_joint": 0.0},
        #     ),
        # ),
        # "panda_hand": PANDA_HAND_MIMIC_GROUP_CFG,
    },
)