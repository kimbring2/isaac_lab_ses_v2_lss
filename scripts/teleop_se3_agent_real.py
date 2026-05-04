"""Script to run teleoperation with Isaac Lab manipulation environments.
Supports multiple input devices (e.g., keyboard and spacemouse) and devices configured within the environment."""

"""Launch Isaac Sim Simulator first."""
import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices'."
        " Built-ins: keyboard and spacemouse. Not all tasks support all built-ins."
    ),
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=3.0, help="Sensitivity factor.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app





"""Rest everything follows."""
import logging
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg, Se3Composite, Se3CompositeCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

import ses_v2_lss.tasks  # noqa: F401

from lss_controller import LSSArmController
import time
import sys
import pyzed.sl as sl
import cv2
import argparse
import socket 


# Initialize the arm
arm = LSSArmController(port='/dev/ttyUSB0')

# Initialize the ZED Camera
ip_address = '192.168.0.191:30000'
camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1
led_on = True 
selection_rect = sl.Rect()
select_in_progress = False
origin_rect = (-1, -1)

init_parameters = sl.InitParameters()
init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL
init_parameters.sdk_verbose = 1
init_parameters.set_from_stream(ip_address.split(':')[0], int(ip_address.split(':')[1]))
cam = sl.Camera()
status = cam.open(init_parameters)
if status > sl.ERROR_CODE.SUCCESS:
    print("Camera Open : " + repr(status) + ". Exit program.")
    exit()

runtime = sl.RuntimeParameters()
win_name = "Camera Remote Control"
mat = sl.Mat()
cv2.namedWindow(win_name)


# import logger
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Run teleoperation with an Isaac Lab manipulation environment.
    Creates the environment, sets up teleoperation interfaces and callbacks, and runs the main simulation loop until the application is closed.
    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task

    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. Received environment config type: {type(env_cfg).__name__}"
        )

    # modify configuration
    env_cfg.terminations.time_out = None

    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    print("env: ", env)

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True


    # Callback handlers
    def reset_recording_instance() -> None:
        """Reset the environment to its initial state. Sets a flag to reset the environment on the next simulation step."""
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """Activate teleoperation control of the robot. Enables the application of teleoperation commands to the environment."""
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """Deactivate teleoperation control of the robot. Disables the application of teleoperation commands to the environment."""
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")


    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance, "START": start_teleoperation, "STOP": stop_teleoperation, "RESET": reset_recording_instance,
    }

    teleoperation_active = True

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks)
        else:
            logger.warning(f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default.")
            
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity))
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity))
            elif args_cli.teleop_device.lower() == "composite":
                teleop_interface = Se3Composite(Se3CompositeCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.15 * sensitivity))
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Configure the teleop device in the environment config.")
                
                env.close()
                simulation_app.close()
                
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.")

    total_step = 0

    # 1. Turn on interactive mode before the loop
    plt.ion()
    plt.figure(figsize=(10, 6))

    # simulate environment
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                # get device command
                action = teleop_interface.advance()

                # Only apply teleop commands when active
                if teleoperation_active:

                    '''
                    zed_x_camera_rgb_image = env.scene["zed_x_tiled_camera"].data.output["rgb"]
                    zed_x_camera_depth_image = env.scene["zed_x_tiled_camera"].data.output["depth"]
                    zed_x_camera_depth_image = zed_x_camera_depth_image.cpu().numpy()[0]

                    max_render_dist = 5.0
                    zed_x_camera_depth_image = np.clip(zed_x_camera_depth_image, 0, max_render_dist)
                    depth_normalized = cv2.normalize(zed_x_camera_depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                    # 2. Apply a Colormap (COLORMAP_MAGMA is equivalent to your 'magma')
                    # Note: COLORMAP_MAGMA was added in OpenCV 4.5.2. Use COLORMAP_JET for older versions.
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

                    # 3. Display the image
                    depth_colored = cv2.resize(depth_colored, (640, 480), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow('Sim Depth Camera', depth_colored)

                    err = cam.grab(runtime) # Check that a new image is successfully acquired
                    if err <= sl.ERROR_CODE.SUCCESS:
                        # cam.retrieve_image(mat, sl.VIEW.LEFT) # Retrieve left image
                        cam.retrieve_image(mat, sl.VIEW.DEPTH) # Retrieve left image

                        cvImage = mat.get_data()
                        if (not selection_rect.is_empty() and selection_rect.is_contained(sl.Rect(0, 0, cvImage.shape[1], cvImage.shape[0]))):
                            cv2.rectangle(cvImage, 
                                          (selection_rect.x, selection_rect.y), 
                                          (selection_rect.width + selection_rect.x, selection_rect.height + selection_rect.y), 
                                          (220, 180, 20), 
                                          2)
                        
                        cvImage = cv2.resize(cvImage, (640, 480), interpolation=cv2.INTER_LINEAR)
                        cv2.imshow('Real Depth Camera', cvImage)
                    else:
                        print("Error during capture : ", err)
                        break

                    key = cv2.waitKey(5)
                    '''

                    # process actions
                    actions = action.repeat(env.num_envs, 1)

                    # apply actions
                    env.step(actions)

                    # 1. Access the robot asset from the scene
                    robot = env.scene["robot"]

                    # 2. Get the joint positions
                    current_joint_pos = robot.data.joint_pos
                    
                    # 3. Move to CPU and convert to numpy
                    joint_array = current_joint_pos[0].cpu().numpy()

                    # 4. Convert radians to degrees
                    joint_degrees = np.degrees(joint_array)

                    # 5. Flatten to a list
                    joint_list = joint_degrees.tolist()
                    #print(f"Joint Positions (Degrees): {joint_list}")

                    # 6. Print with formatting
                    formatted_joints = [int(q) for q in joint_list]
                    # Formatted Joint Degrees: [-174, 67, 80, -12, 0, 75, -75]

                    #print("total_step: ", total_step)
                    if total_step % 50 == 0:
                        #print("formatted_joints[5]: ", formatted_joints[5])
                        target_positions = [formatted_joints[0] + 4, 
                                            formatted_joints[1], 
                                            -formatted_joints[2], 
                                            -formatted_joints[3], 
                                            formatted_joints[5], 
                                            formatted_joints[4]]
                        print(f"Formatted Joint Degrees: {formatted_joints}")
                        
                        arm.move_joints(target_positions, duration_ms=500)
                        #print("total_step: ", total_step)

                        time.sleep(0.5)
                        
                        # 3. Get positions
                        #for i in range(1, 7):
                        #    pos = arm.get_position(i)
                        #    print(f"Servo {i} is at: {pos} degrees")
                        
                    # Give it time to move
                    #time.sleep(1)
                    total_step += 1
                else:
                    #env.sim.render()
                    simulation_app.update()

                if should_reset_recording_instance:
                    env.reset()
                    teleop_interface.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    env.close()
    print("Environment closed")

    # 4. Always good practice to close
    arm.close()

    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    # run the main function
    main()
    
    # close sim app
    simulation_app.close()