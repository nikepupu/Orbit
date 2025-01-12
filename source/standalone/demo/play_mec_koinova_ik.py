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


"""
Main
"""


def main():
    """Spawns a mobile manipulator and applies random joint position commands."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    # Set main camera
    set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.0])
    # Spawn things into stage
    robot_cfg = KINOVA_CFG
    # robot_cfg.rigid_props.disable_gravity = True
    robot = SingleArmManipulator(cfg=robot_cfg)
    robot.spawn("/World/envs/env_0/Robot", translation=(0.0, 0.0, 0.0))
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
    ee_marker = StaticMarker("/Visuals/ee_current", count=1, scale=(0.1, 0.1, 0.1))
    goal_marker = StaticMarker("/Visuals/ee_goal", count=1, scale=(0.1, 0.1, 0.1))

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Create buffers to store actions
    ik_commands = torch.zeros(robot.count, ik_controller.num_actions, device=robot.device)
    robot_actions = torch.ones(robot.count, robot.num_actions, device=robot.device) * -1

    # Set end effector goals
    # Define goals for the arm
    ee_goals = [
        # [0.2, 0.2, 1.4, 0.7071068,  0.0, 0.7071068, 0.0]
        # [0.5, -0.4, 0.6, -0.5, 0.5, -0.5, 0.5],
        [0.5, -0.4, 0.6, 0.5, 0.5, 0.5, 0.5],
        # [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    ik_commands[:] = ee_goals[current_goal_idx]

     # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    count = 0
    # Note: We need to update buffers before the first step for the controller.
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
        if count % 1000 == 0:
            # reset time
            count = 0
            sim_time = 0.0
            # reset dof state
            dof_pos, dof_vel = robot.get_default_dof_state()
            robot.set_dof_state(dof_pos, dof_vel)
            robot.reset_buffers()
            # reset controller
            ik_controller.reset_idx()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        # set the controller commands
        ik_controller.set_command(ik_commands)
        # compute the joint commands
        print('robot data: ', robot.data)
        robot_actions[:, : robot.arm_num_dof] = ik_controller.compute(
            robot.data.ee_state_w[:, 0:3],
            robot.data.ee_state_w[:, 3:7],
            robot.data.ee_jacobian,
            robot.data.arm_dof_pos,
        )
        # in some cases the zero action correspond to offset in actuators
        # so we need to subtract these over here so that they can be added later on
        # arm_command_offset = robot.data.actuator_pos_offset[:, robot.base_num_dof : robot.base_num_dof  + robot.arm_num_dof]
        # # offset actuator command with position offsets
        # # note: valid only when doing position control of the robot
        # robot_actions[:, robot.base_num_dof :  robot.base_num_dof + robot.arm_num_dof] -= arm_command_offset
        # apply actions
        
        robot.apply_action(robot_actions)
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
            ee_marker.set_world_poses(robot.data.ee_state_w[:, 0:3], robot.data.ee_state_w[:, 3:7])
            goal_marker.set_world_poses(ik_commands[:, 0:3], ik_commands[:, 3:7])

if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
