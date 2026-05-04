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
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import functional as VF

from devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg, Se3Composite, Se3CompositeCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

import ses_v2_lss.tasks  # noqa: F401

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
    #env_cfg.terminations.time_out = None

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

    #print("env: ", env)

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

    # simulate environment
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                # get device command
                action = teleop_interface.advance()

                # Only apply teleop commands when active
                if teleoperation_active:
                    # process actions
                    actions = action.repeat(env.num_envs, 1)

                    zed_x_camera_rgb_image = env.scene["zed_x_tiled_camera"].data.output["rgb"]
                    zed_x_camera_depth_image = env.scene["zed_x_tiled_camera"].data.output["depth"]
                    zed_x_camera_depth_image = zed_x_camera_depth_image.cpu().numpy()[0]
                    max_render_dist = 5.0
                    zed_x_camera_depth_image = np.clip(zed_x_camera_depth_image, 0, max_render_dist)
                    depth_min = zed_x_camera_depth_image.min()
                    depth_max = zed_x_camera_depth_image.max()
                    depth_normalized = cv2.normalize(zed_x_camera_depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                    # 2. Apply a Colormap (COLORMAP_MAGMA is equivalent to your 'magma')
                    # Note: COLORMAP_MAGMA was added in OpenCV 4.5.2. Use COLORMAP_JET for older versions.
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

                    # 3. Display the image
                    depth_colored = cv2.resize(depth_colored, (640, 480), interpolation=cv2.INTER_LINEAR)
                    #cv2.imshow('Sim Depth Camera', depth_colored)
                    #cv2.waitKey(1)

                    '''
                    weights = torch.tensor([0.2989, 0.5870, 0.1140], device='cuda:0').view(1, 3, 1, 1)
                    zed_x_camera_rgb_image = zed_x_camera_rgb_image.permute(0, 3, 1, 2).float() / 255.0
                    zed_x_camera_depth_image = zed_x_camera_depth_image.permute(0, 3, 1, 2).float() / 255.0

                    #zed_x_camera_rgb_image = F.interpolate(zed_x_camera_rgb_image, size=(64, 64), mode='bilinear', 
                    #                                       align_corners=False)
                    zed_x_camera_rgb_image = (zed_x_camera_rgb_image * weights).sum(dim=1, keepdim=True)

                    zed_x_camera_rgb_image = zed_x_camera_rgb_image.cpu().numpy()[0]
                    zed_x_camera_depth_image = zed_x_camera_depth_image.cpu().numpy()[0]

                    zed_x_camera_rgb_image = np.transpose(zed_x_camera_rgb_image, axes=(1, 2, 0)) * 255.0
                    zed_x_camera_depth_image = np.transpose(zed_x_camera_depth_image, axes=(1, 2, 0)) * 255.0

                    zed_x_camera_rgb_image = zed_x_camera_rgb_image.astype(np.uint8) 
                    zed_x_camera_depth_image = zed_x_camera_depth_image.astype(np.uint8) 
                    print("zed_x_camera_depth_image.shape 2: ", zed_x_camera_depth_image.shape)
                    '''

                    #cv2.imshow('ZED-X Camera RGB', zed_x_camera_rgb_image)
                    #cv2.imshow('ZED-X Camera Depth', zed_x_camera_depth_image)
                    #cv2.waitKey(1)

                    # apply actions
                    env.step(actions)
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


if __name__ == "__main__":
    # run the main function
    main()
    
    # close sim app
    simulation_app.close()