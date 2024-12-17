import time 
from typing import Dict 

from numpy import choose
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.utils.input_utils import *

MAX_FR = 25 # max frame rate for running simulation

if __name__ == "__main__":
    options = {}
    print("Please choose environment for single arm robot")
    options["env_name"] = choose_environment()

    print("Currently we only support single-arm environments for openVLA testing")
    if ("TwoArm" in options["env_name"]) or ("Humanoid" in options["env_name"]):
        pass
    else: 
        options["robots"] = choose_robots(exclude_bimanual=True)

    joint_dim = 6 if options["robots"] == "UR5e" else 7

    print("Please choose only JOINT_POSITION controller...")
    controller_name = choose_controller(part_controllers=True)

    arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
    robot = options["robots"][0] if isinstance(options["robots"], list) else options["robots"]
    options["controller_configs"] = refactor_composite_controller_config(
        arm_controller_config, robot, ["right", "left"]
    )

    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    env.reset()
    env.viewer.set_camera(camera_id=0)

    # get gripper dimension
    gripper_dim = 0
    gripper_dim = env.robots[0].gripper["right"].dof

    # Neutral position 
    neutral = np.zeros(joint_dim + gripper_dim)
    
    print(f"Controlling {options['robots']} robot with {joint_dim} joints and {gripper_dim} gripper dimension")
    print(f"Total action dimension: {joint_dim + gripper_dim}")
    print("Enter space-separated joint positions (e.g., '0 0.5 0 -0.5 0 0 0' for UR5e with gripper)")


    while True:
        try:
            start = time.time()
            action = neutral.copy()

            # Get joint positions from user for now, eventually will be from VLA 
            joint_pos = input("Enter joint positions (or 'q' to quit): ")
            if joint_pos.lower() == 'q':
                break

            try: 
                joint_values = np.array([float(x) for x in joint_pos.split()])
                if len(joint_values) != (joint_dim + gripper_dim):
                    print(f"Error: Expected {joint_dim + gripper_dim} values ({joint_dim} joints + {gripper_dim} gripper), got {len(joint_values)}")
                    continue
                action = joint_values  # Set the entire action including gripper
            except ValueError:
                print("Error: Invalid input format. Please enter space-separated numbers")
                continue

            env.step(action)
            env.render()

            # limit frame rate if necessary
            elapsed = time.time() - start
            diff = 1 / MAX_FR - elapsed
            if diff > 0:
                time.sleep(diff)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            break

    env.close()