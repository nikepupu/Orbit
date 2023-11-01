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
    robot_cfg.rigid_props.disable_gravity = True
    robot = SingleArmManipulator(cfg=robot_cfg)
    robot.spawn("/World/Robot_1", translation=(0.0, -1.0, 0.0))
    # robot.spawn("/World/Robot_2", translation=(0.0, 1.0, 0.0))
    design_scene()
    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    robot.initialize("/World/Robot.*")
    # Reset states
    robot.reset_buffers()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # dummy action
    actions = torch.rand(robot.count, robot.num_actions, device=robot.device)
    # actions[:, 0 : robot.base_num_dof] = 0.0
    actions[:, -1] = -1

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    ep_step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue
        # reset
        if ep_step_count  == 0:
            sim_time = 0.0
           
            # reset dof state
            dof_pos, dof_vel = robot.get_default_dof_state()
           
            robot.set_dof_state(dof_pos, dof_vel)
            robot.reset_buffers()
            # reset command
            actions = torch.rand(robot.count, robot.num_actions, device=robot.device)
            # actions[:, 0 : robot.base_num_dof] = 0.0
            actions[:, -1] = 1
            # print(">>>>>>>> Reset! Opening gripper.")
        # change the gripper action
        if ep_step_count % 100 == 0:
            print(' env.robot.data.tool_dof_pos: ', robot.data.tool_dof_pos)
        if ep_step_count % 400 == 0 and ep_step_count  > 0:
            # flip command
            # actions[:, -1] = -actions[:, -1]
            print('flip gripper')
        # ==============================================
        # change the base action
        # if ep_step_count == 0:
        #     actions[:, : robot.base_num_dof] = 0.0
        #     actions[:, 0] = 1.0
        # if ep_step_count == 100:
        #     actions[:, : robot.base_num_dof] = 0.0
        #     actions[:, 0] = -1.0

        # if ep_step_count == 200:
        #     actions[:, : robot.base_num_dof] = 0.0
        #     actions[:, 1] = 1.0

        # if ep_step_count == 300:
        #     actions[:, : robot.base_num_dof] = 0.0
        #     actions[:, 1] = -1.0

        # if ep_step_count == 400:
        #     actions[:, : robot.base_num_dof] = 0.0
        #     actions[:, 2] = 1.0

        # if ep_step_count == 500:
        #     actions[:, : robot.base_num_dof] = 0.0
        #     actions[:, 2] = -1.0
        
        # if ep_step_count == 700:
        #     ep_step_count = 0
        # ==============================================

        # if ep_step_count == 200:
        #     actions[:, : robot.base_num_dof] = 0.0
        #     actions[:, 0] = 1.0
        # if ep_step_count == 300:
        #     actions[:, : robot.base_num_dof] = 0.0
        #     actions[:, 0] = -1.0
        # if ep_step_count == 400:
        #     actions[:, : robot.base_num_dof] = 0.0
        #     actions[:, 1] = 1.0
        # if ep_step_count == 500:
        #     actions[:, : robot.base_num_dof] = 0.0
        #     actions[:, 1] = -1.0
        # if ep_step_count == 600:
        #     actions[:, : robot.base_num_dof] = 0.0
        #     actions[:, 2] = 1.0
        # if ep_step_count == 700:
        #     actions[:, : robot.base_num_dof] = 0.0
        #     actions[:, 2] = -1.0
        # if ep_step_count == 900:
        #     actions[:, : robot.base_num_dof] = 0.0
        #     actions[:, 2] = 1.0
        # change the arm action
        # if ep_step_count % 100:
        #     # print('robot.base_num_dof: ', robot.base_num_dof)
        #     # print('robot.arm_num_dof: ', robot.arm_num_dof)
        #     # print('robot.num_actions: ', robot.num_actions)
        #     actions[:, robot.base_num_dof : -1 ] = torch.rand(robot.count, robot.arm_num_dof, device=robot.device)
        # apply action
        # print(actions)
        robot.apply_action(actions)
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        ep_step_count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)
            # read buffers
            # if ep_step_count % 20 == 0:
            #     if robot.data.tool_dof_pos[0, -1] > 0.01:
            #         print("Opened gripper.")
            #     else:
            #         print("Closed gripper.")


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()