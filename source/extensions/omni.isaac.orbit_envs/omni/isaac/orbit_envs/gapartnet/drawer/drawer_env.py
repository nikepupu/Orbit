# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gym.spaces
import math
import torch
import torch.nn.functional as F

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematics
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import random_orientation, sample_uniform, scale_transform
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager
from omni.isaac.orbit.robots.mobile_manipulator import MobileManipulator
# from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs
from omni.isaac.orbit.objects.articulated.articulated_object import ArticulatedObject
from omni.isaac.orbit.objects import RigidObject
from .drawer_cfg import RandomizationCfg, DrawerEnvCfg
import numpy as np
from pxr import UsdGeom
from pytorch3d.transforms import quaternion_to_matrix
from omni.isaac.orbit.markers import PointMarker, StaticMarker
import omni
from omni.isaac.dynamic_control import _dynamic_control
from omni.physx.scripts import physicsUtils
# ./orbit.sh -p source/standalone/workflows/sb3/train.py --task Isaac-Gapartnet-Drawer-v0  --num_envs 6

def quat_axis(q, axis_idx):
    """Extract a specific axis from a quaternion."""
    rotm = quaternion_to_matrix(q)
    axis = rotm[:, axis_idx]

    return axis
    
from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
class CabinetView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "CabinetView",
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        # self._drawers = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/Drawer/.*/joint_1", name="cabinets_view", reset_xform_properties=False
        # )

class DrawerEnv(IsaacEnv):
    """Environment for reaching to desired pose for a single-arm manipulator."""

    def __init__(self, cfg: DrawerEnvCfg = None, **kwargs):
        # copy configuration
        self.cfg = cfg
        # parse the configuration for controller configuration
        # note: controller decides the robot control mode
        self._pre_process_cfg()
        # create classes (these are called by the function :meth:`_design_scene`
        self.robot = MobileManipulator(cfg=self.cfg.robot)
        # self.object = RigidObject(cfg=self.cfg.manipulationObject)

        # initialize the base class to setup the scene.
        super().__init__(self.cfg, **kwargs)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()

        # prepare the observation manager
        self._observation_manager = DrawerObservationManager(class_to_dict(self.cfg.observations), self, self.device)
        # prepare the reward manager
        self._reward_manager = DrawerRewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        # print information about MDP
        print("[INFO] Observation Manager:", self._observation_manager)
        print("[INFO] Reward Manager: ", self._reward_manager)

        # compute the observation space
        num_obs = self._observation_manager._group_obs_dim["policy"][0]
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        print("[INFO]: Completed setting up the environment...")
        # Take an initial step to initialize the scene.
        self.sim.step()
        # -- fill up buffers
        self.robot.update_buffers(self.dt)

        

    """
    Implementation specifics.
    """

    def _design_scene(self):
        import omni
        from omni.isaac.core.prims import XFormPrim

        # ground plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=0)
        # table
        prim = prim_utils.create_prim(self.template_env_ns + "/Drawer", usd_path=self.cfg.drawer.usd_path)
        # apply physics material
        from pxr import Usd, UsdPhysics, UsdShade, UsdGeom
        self.stage = omni.usd.get_context().get_stage()

        _physicsMaterialPath = prim.GetPath().AppendChild("physicsMaterial")
        UsdShade.Material.Define(self.stage, _physicsMaterialPath)
        material = UsdPhysics.MaterialAPI.Apply(self.stage.GetPrimAtPath(_physicsMaterialPath))
        material.CreateStaticFrictionAttr().Set(1.0)
        material.CreateDynamicFrictionAttr().Set(1.0)
        material.CreateRestitutionAttr().Set(1.0)

        physicsUtils.add_physics_material_to_prim(self.stage, prim, _physicsMaterialPath)

        
        prim_path = self.template_env_ns + "/Drawer"
        bboxes = omni.usd.get_context().compute_path_world_bounding_box(prim_path)
        min_box = np.array(bboxes[0])
        
        zmin = min_box[2]
        
        drawer = XFormPrim(prim_path=prim_path)
        position, orientation = drawer.get_world_pose()
        
        position[2] += -zmin 
        drawer.set_world_pose(position, orientation)
       
        # bboxes = prim.ComputeWorldBound(0, UsdGeom.Tokens.default_ )
        # prim_bboxes = np.array([bboxes.ComputeAlignedRange().GetMin(), bboxes.ComputeAlignedRange().GetMax()])
     
        # robot
        self.robot.spawn(self.template_env_ns + "/Robot", translation=[-1.2, 0.2, 0.0])

        self._ee_markers = StaticMarker(
                "/Visuals/ee_current", self.num_envs, usd_path=self.cfg.marker.usd_path, scale=self.cfg.marker.scale
            )

        # self._goal_markers = StaticMarker(
        #         "/Visuals/ee_goal", self.num_envs, usd_path=self.cfg.marker.usd_path, scale=self.cfg.marker.scale
        #     )

        self.drawer_link_path =  "/link_4"
        self.drawer_joint_path = "/link_2/joint_1"
        

        print('done design scene')
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: VecEnvIndices):
        # randomize the MDP
        # -- robot DOF state
        # is env do not have _cabinets
        if not hasattr(self, '_cabinets'):
            self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/Drawer", name="cabinet_view")
            self._cabinets.initialize()

        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        # -- desired end-effector pose
        # self._randomize_ee_desired_pose(env_ids, cfg=self.cfg.randomization.ee_desired_pose)

        # -- Reward logging
        # fill extras with episode information
        self.extras["episode"] = dict()
        # reset
        # -- rewards manager: fills the sums for terminated episodes
        self._reward_manager.reset_idx(env_ids, self.extras["episode"])
        # -- obs manager
        self._observation_manager.reset_idx(env_ids)
        # -- reset history
        self.previous_actions[env_ids] = 0
        # -- MDP reset
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        # controller reset
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller.reset_idx(env_ids)
        
        # print('cabinets joint: ', self._cabinets.get_joint_positions(clone=False))
        # exit()
        self._cabinets.set_joint_positions(
            torch.zeros_like(self._cabinets.get_joint_positions(clone=False)[env_ids]), indices=env_ids
        )

        self._cabinets.set_joint_velocities(
            torch.zeros_like(self._cabinets.get_joint_velocities(clone=False)[env_ids]), indices=env_ids
        )

        full_name =  f"/World/envs/env_0/Drawer" + self.drawer_joint_path 
        joint = self.stage.GetPrimAtPath(full_name)
        self.upper = joint.GetAttribute("physics:upperLimit").Get()
        self.lower = joint.GetAttribute("physics:lowerLimit").Get()

        
        # dc = _dynamic_control.acquire_dynamic_control_interface()
        # for idx in env_ids:
        #     full_name =  f"/World/envs/env_{idx}/Drawer" + self.drawer_joint_path 
            
        #     joint = self.stage.GetPrimAtPath(full_name)
        #     joint_type = joint.GetTypeName()
        #     upper = joint.GetAttribute("physics:upperLimit").Get()
        #     lower = joint.GetAttribute("physics:lowerLimit").Get()
        #     art = dc.get_articulation(full_name)
        #     dof_ptr = dc.find_articulation_dof(art, full_name)
        #     percentage = 0

        #     tmp = percentage / 100.0 *(upper-lower) + lower

        #     if joint_type == 'PhysicsPrismaticJoint':
        #         dof_pos = tmp
        #     else:
        #         dof_pos = math.radians(tmp)

            
            
        #     dc.wake_up_articulation(art)
        #     dc.set_dof_position(dof_ptr, dof_pos)
           

    def _step_impl(self, actions: torch.Tensor):
        # pre-step: set actions into buffer
        
        self.actions = actions.clone().to(device=self.device)
        
        # map the last dimension
        # < 0.5 to -1 and > 0.5 to 1
        # self.actions[:, -1] = -1
        self.actions[:, -1] = torch.where(self.actions[:, -1] < 0.0, -1, 1)
        # print('gripper: ', self.actions[:, -1])
        # action_probs = F.softmax(self.actions[:, -1], dim=0)
        # print('gripper prob: ', action_probs )

        # chosen_actions = []
        # for prob in action_probs:
        #     action = torch.tensor([-1, 1]).numpy()
        #     chosen_action = torch.multinomial(torch.tensor([prob.item(), 1 - prob.item()]), 1).item()
        #     chosen_actions.append(chosen_action)

        # self.actions[:, -1] = torch.tensor(chosen_actions)

        self._ee_markers.set_world_poses(self.robot.data.ee_state_w[:, 0:3], self.robot.data.ee_state_w[:, 3:7])

        # transform actions based on controller
        if self.cfg.control.control_type == "inverse_kinematics":
            # set the controller commands
            self._ik_controller.set_command(self.actions[:,:-1])
            # compute the joint commands
            self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                self.robot.data.ee_state_w[:, 3:7],
                self.robot.data.ee_jacobian,
                self.robot.data.arm_dof_pos,
            )
            # offset actuator command with position offsets
            # self.robot_actions[:, self.robot.arm_num_dof] -= self.robot.data.actuator_pos_offset[
            #     :, : self.robot.arm_num_dof 
            # ]
        elif self.cfg.control.control_type == "default":
            # self.robot_actions[:, : self.robot.arm_num_dof+self.robot.base_num_dof + 1] = self.actions
            self.robot_actions[:, : self.robot.arm_num_dof + 1] = self.actions
        # perform physics stepping
        for _ in range(self.cfg.control.decimation):
            # set actions into buffers
            self.robot.apply_action(self.robot_actions)
            # simulate
            self.sim.step(render=self.enable_render)
            # check that simulation is playing
            if self.sim.is_stopped():
                return
        # post-step:
        # -- compute common buffers
        self.robot.update_buffers(self.dt)
        # -- compute MDP signals
        # reward
        self.reward_buf = self._reward_manager.compute()
        # terminations
        self._check_termination()
        # -- store history
        self.previous_actions = self.actions.clone()

        # -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        # -- update USD visualization
       

    def _get_observations(self) -> VecEnvObs:
        # compute observations
        return self._observation_manager.compute()

    """
    Helper functions - Scene handling.
    """

    def _pre_process_cfg(self) -> None:
        """Pre processing of configuration parameters."""
        # set configuration for task-space controller
        if self.cfg.control.control_type == "inverse_kinematics":
            print("Using inverse kinematics controller...")
            # enable jacobian computation
            self.cfg.robot.data_info.enable_jacobian = True
            # enable gravity compensation
            self.cfg.robot.rigid_props.disable_gravity = True
            # set the end-effector offsets
            self.cfg.control.inverse_kinematics.position_offset = self.cfg.robot.ee_info.pos_offset
            self.cfg.control.inverse_kinematics.rotation_offset = self.cfg.robot.ee_info.rot_offset
        else:
            print("Using default joint controller...")

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # compute constants for environment
        self.dt = self.cfg.control.decimation * self.physics_dt  # control-dt
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)

        # convert configuration parameters to torch
        # randomization
        # -- desired pose
        config = self.cfg.randomization.ee_desired_pose
        for attr in ["position_uniform_min", "position_uniform_max", "position_default", "orientation_default"]:
            setattr(config, attr, torch.tensor(getattr(config, attr), device=self.device, requires_grad=False))

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()

        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Robot")


        # create controller
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller = DifferentialInverseKinematics(
                self.cfg.control.inverse_kinematics, self.robot.count, self.device
            )
            self.num_actions = self._ik_controller.num_actions + 1
        elif self.cfg.control.control_type == "default":
            self.num_actions = self.robot.num_actions

        # history
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # robot joint actions
        self.robot_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        # commands
        # self.ee_des_pose_w = torch.zeros((self.num_envs, 7), device=self.device)


    """
    Helper functions - MDP.
    """

    def _check_termination(self) -> None:
        # extract values from buffer
        # compute resets
        self.reset_buf[:] = 0
        # -- episode length
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)

    # def _randomize_ee_desired_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.EndEffectorDesiredPoseCfg):
    #     """Randomize the desired pose of the end-effector."""
    #     # -- desired object root position
    #     if cfg.position_cat == "default":
    #         # constant command for position
    #         self.ee_des_pose_w[env_ids, 0:3] = cfg.position_default
    #     elif cfg.position_cat == "uniform":
    #         # sample uniformly from box
    #         # note: this should be within in the workspace of the robot
    #         self.ee_des_pose_w[env_ids, 0:3] = sample_uniform(
    #             cfg.position_uniform_min, cfg.position_uniform_max, (len(env_ids), 3), device=self.device
    #         )
    #     else:
    #         raise ValueError(f"Invalid category for randomizing the desired object positions '{cfg.position_cat}'.")
    #     # -- desired object root orientation
    #     if cfg.orientation_cat == "default":
    #         # constant position of the object
    #         self.ee_des_pose_w[env_ids, 3:7] = cfg.orientation_default
    #     elif cfg.orientation_cat == "uniform":
    #         self.ee_des_pose_w[env_ids, 3:7] = random_orientation(len(env_ids), self.device)
    #     else:
    #         raise ValueError(
    #             f"Invalid category for randomizing the desired object orientation '{cfg.orientation_cat}'."
    #         )
    #     # transform command from local env to world
    #     self.ee_des_pose_w[env_ids, 0:3] += self.envs_positions[env_ids]


class DrawerObservationManager(ObservationManager):
    """Reward manager for single-arm reaching environment."""

    def arm_dof_pos_normalized(self, env: DrawerEnv):
        """DOF positions for the arm normalized to its max and min ranges."""
        # print('arm_dof_pos_normalized: ', scale_transform(
        #     env.robot.data.arm_dof_pos,
        #     env.robot.data.soft_dof_pos_limits[:, env.robot.base_num_dof:env.robot.base_num_dof+env.robot.arm_num_dof, 0],
        #     env.robot.data.soft_dof_pos_limits[:, env.robot.base_num_dof:env.robot.base_num_dof+env.robot.arm_num_dof, 1],
        # ))
        # print('dof pos: ', env.robot.data.arm_dof_pos)
        # print('dof limit lower: ', env.robot.data.soft_dof_pos_limits[:, env.robot.base_num_dof:env.robot.base_num_dof+env.robot.arm_num_dof, 0])
        # print('dof limit upper: ', env.robot.data.soft_dof_pos_limits[:, env.robot.base_num_dof:env.robot.base_num_dof+env.robot.arm_num_dof, 1])
        return scale_transform(
            env.robot.data.arm_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, env.robot.base_num_dof:env.robot.base_num_dof+env.robot.arm_num_dof, 0],
            env.robot.data.soft_dof_pos_limits[:, env.robot.base_num_dof:env.robot.base_num_dof+env.robot.arm_num_dof, 1],
        )
    
    def handle_positions(self, env: DrawerEnv):
        import omni
        stage = omni.usd.get_context().get_stage() 
        
        positions = torch.zeros(env.num_envs, 3)
        for idx in range(env.num_envs):
            link_path =  f"/World/envs/env_{idx}/Drawer" + env.drawer_link_path 
            min_box, max_box = omni.usd.get_context().compute_path_world_bounding_box(link_path)
            
            min_point = torch.tensor(np.array(min_box))- env.envs_positions[idx].cpu()
            max_point = torch.tensor(np.array(max_box))- env.envs_positions[idx].cpu()
            positions[idx] = (min_point +  max_point)/2.0

           

        # print('rewards: ', rewards)
        # print('handle_positions: ', positions)
        return positions.cuda()
    
    # def handle_rotations(self, env: DrawerEnv):
    #     return - env.envs_positions
    def base_dof_pos_normalized(self, env: DrawerEnv):
        """DOF positions for the base normalized to its max and min ranges."""
        # print('base dof pos: ', env.robot.data.base_dof_pos,  env.robot.data.soft_dof_pos_limits[:, 0:env.robot.base_num_dof, 0],
        #        env.robot.data.soft_dof_pos_limits[:, 0:env.robot.base_num_dof, 1])
        return scale_transform(
            env.robot.data.base_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, 0:env.robot.base_num_dof, 0],
            env.robot.data.soft_dof_pos_limits[:, 0:env.robot.base_num_dof, 1],
        )

    def base_dof_vel(self, env: DrawerEnv):
        """DOF velocity of the base."""
        # print('base dof vel: ', env.robot.data.base_dof_vel)
        return env.robot.data.base_dof_vel

    def arm_dof_vel(self, env: DrawerEnv):
        """DOF velocity of the arm."""

        # print('arm dof vel: ', env.robot.data.arm_dof_vel)
        return env.robot.data.arm_dof_vel
    
    def tool_dof_pos_scaled(self, env: DrawerEnv):
        """DOF positions of the tool normalized to its max and min ranges."""

        # print('tool_dof_pos_scaled', scale_transform(
        #     env.robot.data.tool_dof_pos,
        #    env.robot.data.soft_dof_pos_limits[:, env.robot.base_num_dof+env.robot.arm_num_dof:, 0],
        #     env.robot.data.soft_dof_pos_limits[:, env.robot.base_num_dof+env.robot.arm_num_dof:, 1],
        # ))

        return scale_transform(
            env.robot.data.tool_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, env.robot.base_num_dof+env.robot.arm_num_dof:, 0],
            env.robot.data.soft_dof_pos_limits[:, env.robot.base_num_dof+env.robot.arm_num_dof:, 1],
        )
        # return scale_transform(
        #     env.robot.data.tool_dof_pos,
        #     env.robot.data.soft_dof_pos_limits[:, env.robot.arm_num_dof + env.robot.base_num_dof :, 0],
        #     env.robot.data.soft_dof_pos_limits[:, env.robot.arm_num_dof + env.robot.base_num_dof :, 1],
        # )

    def tool_positions(self, env: DrawerEnv):
        """Current end-effector position of the arm."""
        # print('tool_positions: ', (env.robot.data.ee_state_w[:, :3] - env.envs_positions) )
        return env.robot.data.ee_state_w[:, :3] - env.envs_positions

    def tool_orientations(self, env: DrawerEnv):
        """Current end-effector orientation of the arm."""
        # make the first element positive
        quat_w = env.robot.data.ee_state_w[:, 3:7]
        quat_w[quat_w[:, 0] < 0] *= -1

        # print('tool_orientations: ', quat_w)
        return quat_w

    def joints_state(self, env: DrawerEnv):
        if not hasattr(env, '_cabinets'):
            env._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/Drawer", name="cabinet_view")
            env._cabinets.initialize()
        print('joint state: ', env._cabinets.get_joint_positions(clone=False))
        return env._cabinets.get_joint_positions(clone=False)

    # def ee_position(self, env: DrawerEnv):
    #     """Current end-effector position of the arm."""

    #     # print('ee_position: ', (env.robot.data.ee_state_w[:, :3] - env.envs_positions).shape )
    #     return env.robot.data.ee_state_w[:, :3] - env.envs_positions

    # def ee_position_command(self, env: DrawerEnv):
    #     """Desired end-effector position of the arm."""

    #     # print('ee_position_command', (env.ee_des_pose_w[:, :3] - env.envs_positions).shape )
    #     return env.ee_des_pose_w[:, :3] - env.envs_positions

    def actions(self, env: DrawerEnv):
        """Last actions provided to env."""
        # print('actions: ', env.actions)
        return env.actions


class DrawerRewardManager(RewardManager):
    """Reward manager for single-arm reaching environment."""

    def custom_reward(self, env: DrawerEnv):
        """Penalize tracking position error using L2-kernel."""
         # Calculate handle vectors
        # TODO need to check this
        
        import omni
        rewards = torch.zeros(env.num_envs)
        for idx in range(env.num_envs):
            link_path =  f"/World/envs/env_{idx}/Drawer" + env.drawer_link_path 
            min_box, max_box = omni.usd.get_context().compute_path_world_bounding_box(link_path)
            corners = torch.zeros((8, 3))
            
            max_pt = torch.tensor(np.array(min_box)) - env.envs_positions[idx].cpu()
            min_pt = torch.tensor(np.array(max_box)) - env.envs_positions[idx].cpu()
            # Top right back
            corners[0] = torch.tensor([max_pt[0], min_pt[1], max_pt[2]])
            # Top right front
            corners[1] = torch.tensor([min_pt[0], min_pt[1], max_pt[2]])
            # Top left front
            corners[2] = torch.tensor([min_pt[0], max_pt[1], max_pt[2]])
            # Top left back (Maximum)
            corners[3] = max_pt
            # Bottom right back
            corners[4] = torch.tensor([max_pt[0], min_pt[1], min_pt[2]])
            # Bottom right front (Minimum)
            corners[5] = min_pt
            # Bottom left front
            corners[6] = torch.tensor([min_pt[0], max_pt[1], min_pt[2]])
            # Bottom left back
            corners[7] = torch.tensor([max_pt[0], max_pt[1], min_pt[2]])
            
            from omni.debugdraw import _debugDraw
            my_debugDraw = _debugDraw.acquire_debug_draw_interface()
            
            def draw_box(maximum, minimum):
                    import carb
                    color = 4283782485
                    # minimum = carb.Float3(origin[0] - extent[0], origin[1] - extent[1], origin[2] - extent[2])
                    # maximum = carb.Float3(origin[0] + extent[0], origin[1] + extent[1], origin[2] + extent[2])
                    my_debugDraw.draw_line(carb.Float3(minimum[0], minimum[1], minimum[2]),color, carb.Float3(maximum[0], minimum[1], minimum[2]), color)
                    my_debugDraw.draw_line(carb.Float3(maximum[0], minimum[1], minimum[2]),color, carb.Float3(maximum[0], maximum[1], minimum[2]), color)
                    my_debugDraw.draw_line(carb.Float3(maximum[0], maximum[1], minimum[2]),color, carb.Float3(minimum[0], maximum[1], minimum[2]), color)
                    my_debugDraw.draw_line(carb.Float3(minimum[0], maximum[1], minimum[2]),color, carb.Float3(minimum[0], minimum[1], minimum[2]), color)
                    my_debugDraw.draw_line(carb.Float3(minimum[0], minimum[1], minimum[2]),color, carb.Float3(minimum[0], minimum[1], maximum[2]), color)
                    my_debugDraw.draw_line(carb.Float3(minimum[0], minimum[1], maximum[2]),color, carb.Float3(maximum[0], minimum[1], maximum[2]), color)
                    my_debugDraw.draw_line(carb.Float3(maximum[0], minimum[1], maximum[2]),color, carb.Float3(maximum[0], maximum[1], maximum[2]), color)
                    my_debugDraw.draw_line(carb.Float3(maximum[0], maximum[1], maximum[2]),color, carb.Float3(minimum[0], maximum[1], maximum[2]), color)
                    my_debugDraw.draw_line(carb.Float3(minimum[0], maximum[1], maximum[2]),color, carb.Float3(minimum[0], minimum[1], maximum[2]), color)
                    my_debugDraw.draw_line(carb.Float3(maximum[0], minimum[1], minimum[2]),color, carb.Float3(maximum[0], minimum[1], maximum[2]), color)
                    my_debugDraw.draw_line(carb.Float3(maximum[0], maximum[1], minimum[2]),color, carb.Float3(maximum[0], maximum[1], maximum[2]), color)
                    my_debugDraw.draw_line(carb.Float3(minimum[0], maximum[1], minimum[2]),color, carb.Float3(minimum[0], maximum[1], maximum[2]), color)
            
            
            # draw_box(max_pt, min_pt)

            handle_out = corners[0] - corners[4]
            handle_long = corners[1] - corners[0]
            handle_short = corners[3] - corners[0]

            # print('long(0): ', handle_long)
            # print('short(1): ', handle_short)
            # print('out(2): ', handle_out)

            # handle_long, handle_short, handle_out = handle_short, handle_out, handle_long
            # print('==================')
            # print('long(0): ', handle_long)
            # print('short(1): ', handle_short)
            # print('out(2): ', handle_out)
            # handle_out, handle_long = handle_long, handle_out
            
            
            handle_mid_point = (max_pt + min_pt) / 2

            handle_out_length = torch.norm(handle_out)
            handle_long_length = torch.norm(handle_long)
            handle_short_length = torch.norm(handle_short)

            # handle_shortest = torch.min(torch.min(handle_out_length, handle_long_length), handle_short_length)
            handle_out = handle_out / handle_out_length
            handle_long = handle_long / handle_long_length
            handle_short = handle_short / handle_short_length

            # print('long(0): ', handle_long)
            # print('short(1): ', handle_short)
            # print('out(2): ', handle_out)

            # exit()

            tool_positions = (env.robot.data.ee_state_w[:, :3] - env.envs_positions)[idx].cpu()
            quat_w = env.robot.data.ee_state_w[:, 3:7]
            quat_w[quat_w[:, 0] < 0] *= -1
             # change quarternion order from W, X, Y, Z to X, Y, Z, W
            quat_w = quat_w[:, [1, 2, 3, 0]]
            tool_orientations = quat_w.cpu()[idx]
           
                # reaching
            tcp_to_obj_delta = tool_positions[:3] - handle_mid_point
            # print('delta: ', tcp_to_obj_delta)
            tcp_to_obj_dist = tcp_to_obj_delta.norm()
            # print('tcp_to_obj_dist: ', tcp_to_obj_dist)
            is_reached_out = (tcp_to_obj_delta * handle_out).sum().abs() < handle_out_length/2 
            short_ltip = ((tool_positions[:3] - handle_mid_point) * handle_short).sum() 
            short_rtip = ((tool_positions[:3] - handle_mid_point) * handle_short).sum()
            is_reached_short = (short_ltip * short_rtip) < 0
            is_reached_long = (tcp_to_obj_delta * handle_long).sum().abs() < handle_long_length/2 
            is_reached = is_reached_out & is_reached_short & is_reached_long

            if is_reached:
                print('is_reached')
            # print('reached: ', is_reached_short, is_reached_long, is_reached_out)
            
            reaching_reward = - tcp_to_obj_dist + 0.1 * (is_reached_out + is_reached_short + is_reached_long)

            # # rotation reward
            hand_rot = tool_orientations
            hand_grip_dir = quat_axis(hand_rot, 2)
            hand_grip_dir_length = torch.norm(hand_grip_dir)
            hand_grip_dir  = hand_grip_dir/ hand_grip_dir_length
            
            hand_sep_dir = quat_axis(hand_rot, 1)
            hand_sep_dir_length = torch.norm(hand_sep_dir)
            hand_sep_dir = hand_sep_dir / hand_sep_dir_length

            hand_down_dir = quat_axis(hand_rot, 0)
            hand_down_dir_length = torch.norm(hand_down_dir)
            hand_down_dir = hand_down_dir / hand_down_dir_length

            
            dot1 = (-hand_grip_dir * handle_out).sum()
            dot2 = torch.max((hand_sep_dir * handle_short).sum(), (-hand_sep_dir * handle_short).sum()) 
            dot3 = torch.max((hand_down_dir * handle_long).sum(), (-hand_down_dir * handle_long).sum())

           
            rot_reward = dot1 + dot2 + dot3 - 3
            if rot_reward > 0:
                print('something wrong: ')
                print("hand_grip_dir: ", hand_grip_dir)
                print("handle_out: ", handle_out)
                print("hand_sep_dir: ", hand_sep_dir)
                print("handle_short: ", handle_short)
                print("hand_down_dir: ", hand_down_dir)
                print("handle_long: ", handle_long)
            
            def gripper_open_percentage(X, epsilon=1e-7):
                    """
                    Computes the gripper open percentage.
                    
                    X: Current state tensor.
                    O: Open state tensor.
                    C: Close state tensor.
                    epsilon: A small value to check for negligible differences.
                    
                    Returns:
                    Percentage of gripper open state.
                    """

                    # O = torch.tensor([[-1.2278e-07, -2.2230e-07, 1.8682e-07, 1.2820e-07, -8.7570e-01, 8.7570e-01]])
                    # C = torch.tensor([[7.2500e-01, 8.7570e-01, -8.7570e-01, -8.7570e-01, 6.9399e-08, -7.0420e-08]])

                    O = 8.7570e-01#torch.tensor([[-1.2278e-07, -2.2230e-07, 1.8682e-07]])
                    C = 0#torch.tensor([[7.2500e-01, 8.7570e-01, -8.7570e-01]])
                    
                    
                    # Clip values in X to be within [C, O]
                    # X = torch.where(X > O, O, X)
                    # X = torch.where(X < C, C, X)
                    

                    X = X[-1]
                    if X < 0:
                        X = 0
                    elif X > 0.8757:
                        X = 0.8757
                    
                    # Compute the normalized differences
                    diffs = (X - C) / (O - C + epsilon)
                    
                    # Check for non-negligible differences between O and C
                    # mask = torch.abs(O - C) > epsilon
                    
                    # Calculate average percentage open for the significant elements
                    percentage = diffs * 100
                    
                    # Ensure the result is between 0 and 100
                    percentage = max(0, min(100, percentage))
                    
                    return percentage/100.0

           
            gripper_length =  0.08 * gripper_open_percentage(env.robot.data.tool_dof_pos[idx].cpu())

            # close_reward = (0.1 - gripper_length) * is_reached + (gripper_length) * (~is_reached)

            close_reward = (0.1 - gripper_length) * is_reached + 0.1*(gripper_length-0.1) * (~is_reached)

           

            # print("reached: ",  is_reached_out, is_reached_short, is_reached_long, is_reached, gripper_open_percentage(env.robot.data.tool_dof_pos[idx].cpu()), close_reward)

            grasp_success = is_reached & (gripper_length < handle_short_length + 0.01) & (rot_reward > -0.2)

            # how much cabinets opened
            joint_pos = env._cabinets.get_joint_positions(clone=False)[idx]
            
            pos = joint_pos[1]

            joint_state_reward =  ((pos - env.lower)/(env.upper-env.lower))
            # if joint_state_reward > 0.05:
            #     print('joint_state_reward: ', joint_state_reward)
            reward =  reaching_reward + 0.5*rot_reward + 10*close_reward + 100*joint_state_reward 
            if joint_state_reward > 0.9:
                reward += 100
            rewards[idx] = reward

        # self._goal_markers.set_world_poses(env.ee_des_pose_w[:, :3], env.ee_des_pose_w[:, 3:7])
        
        return rewards.cuda()

    # def tracking_robot_position_exp(self, env: DrawerEnv, sigma: float):
    #     """Penalize tracking position error using exp-kernel."""
    #     # compute error
    #     error = torch.sum(torch.square(env.ee_des_pose_w[:, :3] - env.robot.data.ee_state_w[:, 0:3]), dim=1)
    #     return torch.exp(-error / sigma)

    # def penalizing_robot_dof_velocity_l2(self, env: DrawerEnv):
    #     """Penalize large movements of the robot arm."""
    #     return torch.sum(torch.square(env.robot.data.arm_dof_vel), dim=1)

    # def penalizing_robot_dof_acceleration_l2(self, env: DrawerEnv):
    #     """Penalize fast movements of the robot arm."""
    #     return torch.sum(torch.square(env.robot.data.dof_acc), dim=1)

    # def penalizing_action_rate_l2(self, env: DrawerEnv):
    #     """Penalize large variations in action commands."""
    #     return torch.sum(torch.square(env.actions - env.previous_actions), dim=1)
