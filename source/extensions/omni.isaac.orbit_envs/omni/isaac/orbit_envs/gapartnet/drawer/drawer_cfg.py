# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematicsCfg
# from omni.isaac.orbit.robots.config.ridgeback_body import RIDGEBACK_BODY_CFG
from omni.isaac.orbit.robots.config.mec_kinova_arm_only import KINOVA_CFG
from omni.isaac.orbit.robots.config.mec_kinova import MEC_KINOVA_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulatorCfg
from omni.isaac.orbit.robots.mobile_manipulator import MobileManipulatorCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.objects.articulated import ArticulatedObjectCfg

from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, SimCfg, ViewerCfg
from omni.isaac.orbit.objects import RigidObjectCfg
##
# Scene settings
##


# @configclass
# class DrawerCfg(ArticulatedObjectCfg):
#     """Properties for the table."""

    

#     meta_info=ArticulatedObjectCfg.MetaInfoCfg(
#         usd_path=f"/home/nikepupu/Desktop/Orbit/usd/40147/mobility_relabel_gapartnet.usd",
#         # scale=(1.0, 1.0, 1.0), # we need to use instanceable asset since it consumes less memory
#         # sites_names = []
#     ),


##
# MDP settings
##

@configclass
class MarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.1, 0.1, 0.1]  # x,y,z


@configclass
class RandomizationCfg:
    """Randomization of scene at reset."""

    @configclass
    class EndEffectorDesiredPoseCfg:
        """Randomization of end-effector pose command."""

        # category
        position_cat: str = "default"  # randomize position: "default", "uniform"
        orientation_cat: str = "default"  # randomize position: "default", "uniform"
        # randomize position
        position_default = [0.5, 0.0, 0.5]  # position default (x,y,z)
        position_uniform_min = [0.25, -0.25, 0.25]  # position (x,y,z)
        position_uniform_max = [0.5, 0.25, 0.5]  # position (x,y,z)
        # randomize orientation
        orientation_default = [1.0, 0.0, 0.0, 0.0]  # orientation default

    # initialize
    ee_desired_pose: EndEffectorDesiredPoseCfg = EndEffectorDesiredPoseCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg:
        """Observations for policy group."""

        # global group settings
        enable_corruption: bool = False
        # observation terms
        base_dof_pos_normalized = {"scale": 1.0, "noise": {"name": "uniform", "min": -0.01, "max": 0.01}}
        arm_dof_pos_normalized = {"scale": 1.0, "noise": {"name": "uniform", "min": -0.01, "max": 0.01}}
        tool_dof_pos_scaled = {"scale": 1.0}
        # -- end effector state
        tool_positions = {"scale": 1.0}
        tool_orientations = {"scale": 1.0}

        arm_dof_vel = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.1, "max": 0.1}}
        base_dof_vel = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.1, "max": 0.1}}
        # ee_position = {}
        # ee_position_command = {}
        joints_state = {"scale": 1.0}
        actions = {}

        handle_positions = {"scale": 1.0}
        # handle_rotations = {"scale": 1.0}
        

    # global observation settings
    return_dict_obs_in_group = False
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # tracking_robot_position_l2 = {"weight": 0.0}
    # tracking_robot_position_exp = {"weight": 2.5, "sigma": 0.05}  # 0.25
    # penalizing_robot_dof_velocity_l2 = {"weight": -0.02}  # -1e-4
    # penalizing_robot_dof_acceleration_l2 = {"weight": -1e-5}
    # penalizing_action_rate_l2 = {"weight": -0.1}
    custom_reward = {"weight": 1.0}


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    episode_timeout = True  # reset when episode length ended


@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # action space
    control_type = "default"  # "default", "inverse_kinematics"
    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 2

    # configuration loaded when control_type == "inverse_kinematics"
    inverse_kinematics: DifferentialInverseKinematicsCfg = DifferentialInverseKinematicsCfg(
        command_type="pose_rel",
        ik_method="dls",
        position_command_scale=(0.1, 0.1, 0.1),
        rotation_command_scale=(0.1, 0.1, 0.1),
    )


##
# Environment configuration
##


@configclass
class DrawerEnvCfg(IsaacEnvCfg):
    """Configuration for the reach environment."""

    # General Settings
    env: EnvCfg = EnvCfg(num_envs=4, env_spacing=10, episode_length_s=5.0)
    viewer: ViewerCfg = ViewerCfg(debug_vis=True, eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))
    # Physics settings
    sim: SimCfg = SimCfg(dt=1.0 / 60.0, substeps=1)

    # Scene Settings
    robot: MobileManipulatorCfg = MEC_KINOVA_CFG
    drawer: ArticulatedObjectCfg = ArticulatedObjectCfg(
        meta_info=ArticulatedObjectCfg.MetaInfoCfg(
            usd_path=f"/home/nikepupu/Desktop/Orbit/usd/40147/mobility_relabel_gapartnet.usd",
            scale=(1.0, 1.0, 1.0), # we need to use instanceable asset since it consumes less memory
            sites_names = []
        ),
        
        init_state=ArticulatedObjectCfg.InitialStateCfg(
            dof_pos={".*": 0.0},
            dof_vel={".*": 0.0}, 
            # rot = [0.5, 0.5, -0.5, -0.5]
        ),

        

    )
    

    marker: MarkerCfg = MarkerCfg()

    # MDP settings
    randomization: RandomizationCfg = RandomizationCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()  
