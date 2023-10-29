# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the physics engine to simulate a mobile manipulator.

We currently support the following robots:

* Franka Emika Panda on a Clearpath Ridgeback Omni-drive Base

From the default configuration file for these robots, zero actions imply a default pose.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import torch
import omni
import numpy as np
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.robots.config.mec_kinova import MEC_KINOVA_CFG 
from omni.isaac.orbit.robots.config.mec_kinova_arm_only import KINOVA_CFG

from omni.isaac.orbit.robots.mobile_manipulator import MobileManipulator
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.controllers.differential_inverse_kinematics import (
    DifferentialInverseKinematics,
    DifferentialInverseKinematicsCfg,
)
from omni.isaac.orbit.markers import StaticMarker
from omni.physx.scripts import physicsUtils

"""
Helpers
"""


def design_scene():
    """Add prims to the scene."""
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane")
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )

    prim_utils.create_prim(f"/World/Drawer", usd_path="/home/nikepupu/Desktop/Orbit/usd/40147/mobility_relabel_gapartnet.usd",
                            translation=[0,0,0.6])
    
    from pxr import Usd, UsdPhysics, UsdShade, UsdGeom, PhysxSchema
    from omni.isaac.core.materials import PhysicsMaterial
    stage = omni.usd.get_context().get_stage()

    prim = stage.GetPrimAtPath("/World/Drawer")
    _physicsMaterialPath = prim.GetPath().AppendChild("physicsMaterial")
    # UsdShade.Material.Define(self.stage, _physicsMaterialPath)
    # material = UsdPhysics.MaterialAPI.Apply(self.stage.GetPrimAtPath(_physicsMaterialPath))
    # material.CreateStaticFrictionAttr().Set(1.0)
    # material.CreateDynamicFrictionAttr().Set(1.0)
    # material.CreateRestitutionAttr().Set(1.0)

    material = PhysicsMaterial(
            prim_path=_physicsMaterialPath,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        # -- enable patch-friction: yields better results!
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
    physx_material_api.CreateImprovePatchFrictionAttr().Set(True)
    # -- bind material to feet
    # for site_name in self.cfg.meta_info.tool_sites_names:
    # kit_utils.apply_nested_physics_material(f"/World/Drawer/link_4", material.prim_path)
    prim = stage.GetPrimAtPath("/World/Drawer/link_4/collisions")
    collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath())
    if not collision_api:
        collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    
    collision_api.CreateApproximationAttr().Set("convexDecomposition")
    # meshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(stage.GetPrimAtPath("/World/Drawer/link_4"))
    # meshCollision.CreateSdfResolutionAttr().Set(256)

    # physicsUtils.add_physics_material_to_prim(self.stage, prim, _physicsMaterialPath)


"""
Main
"""


def main():
    """Spawns a mobile manipulator and applies random joint position commands."""

    # Load kit helper
    sim = SimulationContext(physics_dt=1/60, rendering_dt=1/60, backend="torch")
    # Set main camera
    set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.0])
    # Spawn things into stage
    robot_cfg = KINOVA_CFG
    # robot_cfg.rigid_props.disable_gravity = True
    robot = SingleArmManipulator(cfg=robot_cfg)
    robot.spawn("/World/envs/env_0/Robot", translation=(-1.3, 0.2, 0.0))
    # robot.spawn("/World/Robot_2", translation=(0.0, 1.0, 0.0))
    design_scene()
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    ep_step_count = 0
    # Simulate physics
    # Create controller
    # the controller takes as command type: {position/pose}_{abs/rel}
    ik_control_cfg = DifferentialInverseKinematicsCfg(
        command_type="pose_abs",
        ik_method="dls",
        position_offset=robot.cfg.ee_info.pos_offset,
        rotation_offset=robot.cfg.ee_info.rot_offset,
    )
    num_envs = 1
    ik_controller = DifferentialInverseKinematics(ik_control_cfg, num_envs, sim.device)

    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    robot.initialize("/World/envs/env_.*/Robot")
     # dummy action
    actions = torch.rand(robot.count, robot.num_actions, device=robot.device)
    # actions[:, 0 : robot.base_num_dof] = 0.0
    actions[:, -1] = -1
    # Now we are ready!
    print("[INFO]: Setup complete...")

    ik_controller.initialize()
    # Reset states
    robot.reset_buffers()
    ik_controller.reset_idx()

    # Markers
    # ee_marker = StaticMarker("/Visuals/ee_current", count=1, scale=(0.1, 0.1, 0.1))
    # goal_marker = StaticMarker("/Visuals/ee_goal", count=1, scale=(0.1, 0.1, 0.1))

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Create buffers to store actions
    ik_commands = torch.zeros(robot.count, ik_controller.num_actions, device=robot.device)
    robot_actions = torch.ones(robot.count, robot.num_actions, device=robot.device) * -1
    link_path =  f"/World/Drawer/link_4"
   
    min_box, max_box = omni.usd.get_context().compute_path_world_bounding_box(link_path)
            
    min_pt = torch.tensor(np.array(min_box))
    max_pt = torch.tensor(np.array(max_box))
    center1 = (min_pt +  max_pt)/2.0
    
    # Set end effector goals
    # Define goals for the arm
    ee_goals = [
        # [0.2, 0.2, 1.4, 0.7071068,  0.0, 0.7071068, 0.0]
        # [0.5, -0.4, 0.6, -0.5, 0.5, -0.5, 0.5],
        # [center1[0]-0.10, center1[1], center1[2], 0.7071068, 0.0, 0.7071068, 0.0, -1],
        [center1[0]-0.02, center1[1], center1[2], 0.7071068, 0.0, 0.7071068, 0.0, -1],
        [center1[0]-0.02, center1[1], center1[2], 0.7071068, 0.0, 0.7071068, 0.0, 1],
        [center1[0]-0.20, center1[1], center1[2], 0.7071068, 0.0, 0.7071068, 0.0, 1],
        # [center1[0]-0.10, center1[1], center1[2], 0.7071068, 0.0, 0.7071068, 0.0, 1],
        # [center1[0]-0.15, center1[1], center1[2], 0.7071068, 0.0, 0.7071068, 0.0, 1],
        # [center1[0]-0.20, center1[1], center1[2], 0.7071068, 0.0, 0.7071068, 0.0, 1],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
     # Track the given command
    current_goal_idx = -1

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    count = 0
    
    robot.update_buffers(sim_dt)

    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue
        # reset
        def check():
            link_path =  f"/World/Drawer/link_4"

            min_box, max_box = omni.usd.get_context().compute_path_world_bounding_box(link_path)
                    
            min_pt = torch.tensor(np.array(min_box))
            max_pt = torch.tensor(np.array(max_box))
            
            corners = torch.zeros((8, 3))
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

            handle_short = corners[0] - corners[4]
            handle_out = corners[1] - corners[0]
            handle_long = corners[3] - corners[0]

            handle_short, handle_long = handle_long, handle_short

            handle_out_length = torch.norm(handle_out)
            handle_long_length = torch.norm(handle_long)
            handle_short_length = torch.norm(handle_short)
            handle_out = handle_out / handle_out_length
            handle_long = handle_long / handle_long_length
            handle_short = handle_short / handle_short_length
            
            handle_mid_point = (max_pt + min_pt) / 2

            # Note: We need to update buffers before the first step for the controller.
            tool_positions = robot.data.ee_state_w[:, :3]
            tcp_to_obj_delta = tool_positions[:3] - handle_mid_point
            # print('delta: ', tcp_to_obj_delta)
            tcp_to_obj_dist = tcp_to_obj_delta.norm()
            # print('tcp_to_obj_dist: ', tcp_to_obj_dist)
            is_reached_out = (tcp_to_obj_delta * handle_out).sum().abs() < (handle_out_length/2 )
            # short_ltip = ((tool_positions[:3] - handle_mid_point) * handle_short).sum() 
            # short_rtip = ((tool_positions[:3] - handle_mid_point) * handle_short).sum()
            # is_reached_short = (short_ltip * short_rtip) < 0
            is_reached_short = (tcp_to_obj_delta * handle_short).sum().abs() < (handle_short_length/2)
            is_reached_long = (tcp_to_obj_delta * handle_long).sum().abs() < (handle_long_length*2) 
            is_reached = is_reached_out & is_reached_short & is_reached_long
            print('handle_out: ', handle_out)
            print('handle_long: ', handle_long)
            print('handle_short: ', handle_short)
            print('handle_out_length: ', handle_out_length)
            print('handle_long_length: ', handle_long_length)
            print('handle_short_length: ', handle_short_length)
            print(tcp_to_obj_delta, is_reached_out, is_reached_short, is_reached_long)
            return is_reached
        check()
        if current_goal_idx < len(ee_goals):
            
            if count % 500 == 0:
                if current_goal_idx + 1 < len(ee_goals):
                    current_goal_idx = current_goal_idx + 1
                ik_commands[:] = ee_goals[current_goal_idx][:-1]

                
            # set the controller commands
            ik_controller.set_command(ik_commands)
            # compute the joint commands
       
            robot_actions[:, : robot.arm_num_dof] = ik_controller.compute(
                robot.data.ee_state_w[:, 0:3],
                robot.data.ee_state_w[:, 3:7],
                robot.data.ee_jacobian,
                robot.data.arm_dof_pos,
            )
            robot_actions[:, -1] = ee_goals[current_goal_idx][-1]
      
            
            robot.apply_action(robot_actions)
        else:
            exit()
            # perform step
        sim.step(render=not args_cli.headless)
            # update sim-time
        sim_time += sim_dt
        count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)
            # update marker positions
            # ee_marker.set_world_poses(robot.data.ee_state_w[:, 0:3], robot.data.ee_state_w[:, 3:7])
            # goal_marker.set_world_poses(ik_commands[:, 0:3], ik_commands[:, 3:7])

if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
