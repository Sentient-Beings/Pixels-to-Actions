"""Tele-operate robot with Joystick.

JoyStick:
    We use the joystick to control the end-effector of the robot.
    The joystick provides 6-DoF control commands.
    The commands are mapped to joint velocities through an inverse kinematics
    solver from Bullet physics.

Additionally, --pos_sensitivity and --rot_sensitivity provide relative gains for increasing / decreasing the user input
device sensitivity


***Choose controller with the --controller argument***

Choice of using either inverse kinematics controller (ik) or operational space controller (osc):
Main difference is that user inputs with ik's rotations are always taken relative to eef coordinate frame, whereas
    user inputs with osc's rotations are taken relative to global frame (i.e.: static / camera frame of reference).


***Choose environment specifics with the following arguments***

    --environment: Task to perform, e.g.: "Lift", "NutAssembly", etc. Note: No Two arm envs are supported 

    --robots: Robot(s) with which to perform the task. Can be any in
        {"Panda", "Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e", "Baxter"}. Note: Currently only "Ur5e" is supported

    --toggle-camera-on-grasp: If enabled, gripper input presses will cycle through the available camera angles

Examples:

    For normal single-arm environment:
        $ python demo_device_control.py --environment PickPlaceCan --robots Sawyer --controller osc
"""

# Basic imports
import argparse
import time
import numpy as np
import cv2
import threading 
from queue import Queue, Empty
import redis

# Robosuite imports
import robosuite as suite
from robosuite import load_composite_controller_config
import robosuite.macros as macros
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

class ImagePublisher:
    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.frame_queue = Queue(maxsize=1)
        self.running = False
        self.publish_thread = None
        self.connection_retry_delay = 1
        self.max_retries = 5

    def connect_redis(self):
        retries = 0
        while retries < self.max_retries and self.running:
            try:
                if self.redis_client is None or not self.redis_client.ping():
                    print(f"Attempting to connect to Redis (attempt {retries + 1}/{self.max_retries})...")
                    self.redis_client = redis.Redis(
                        host=self.redis_host, 
                        port=self.redis_port,
                        socket_keepalive=True,
                        socket_connect_timeout=5,
                        retry_on_timeout=True
                    )
                    self.redis_client.ping()
                    print("Successfully connected to Redis!")  
                return True
            
            except redis.ConnectionError as e:
                print(f"Redis connection failed: {e}")
                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.connection_retry_delay} seconds...")
                    time.sleep(self.connection_retry_delay)
                self.redis_client = None
                
            except Exception as e:
                print(f"Unexpected error while connecting to Redis: {e}")
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.connection_retry_delay)
                self.redis_client = None
                
        return False    

    def publish_loop(self):
        while self.running:
            try:
                if not self.connect_redis():
                    print("Failed to connect to Redis after multiple attempts")
                    time.sleep(self.connection_retry_delay)
                    continue

                # Get frame from queue with timeout
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except Empty:
                    continue

                # Convert image to JPEG format
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not success:
                    continue

                img_bytes = buffer.tobytes()

                # Add frame dimensions
                height, width = frame.shape[:2]
                metadata = f"{width},{height}|".encode()

                # Publish the image
                try:
                    self.redis_client.publish("image_channel", metadata + img_bytes)
                except redis.ConnectionError as e:
                    print(f"Lost connection to Redis: {e}")
                    # Force reconnection
                    self.redis_client = None  
                    continue
                except Exception as e:
                    print(f"Error publishing image: {e}")
                    continue
                # Small delay to prevent CPU overload
                time.sleep(0.01) 

            except Exception as e:
                print(f"Error in publish loop: {e}")
    
    def start(self):
        if not self.running:
            self.running = True
            self.publish_thread = threading.Thread(target=self.publish_loop)
            self.publish_thread.daemon = True
            self.publish_thread.start()
            print("Publisher started")

    def stop(self):
        self.running = False
        if self.publish_thread:
            self.publish_thread.join(timeout=5.0)
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass
        print("Publisher stopped")

    def publish_frame(self, frame):
        if frame is None:
            return
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait() 
                except Empty:
                    pass
            self.frame_queue.put_nowait(frame)
        except Exception as e:
            print(f"Error queuing frame: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument(
        "--controller",
        type=str,
        default='WHOLE_BODY_MINK_IK',
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples) or None to get the robot's default controller if it exists",
    )
    parser.add_argument("--device", type=str, default="joystick")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument("--camera", type=str, default="frontview", help="Name of camera to render")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--skip_frame", type=int, default=1)
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # Create environment
    env = suite.make(
        **config,
        has_renderer=False,
        control_freq=20,
        render_camera=args.camera,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # initialize device
    if args.device == "joystick":
        from robosuite.devices import Joystick

        device = Joystick(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: only 'joystick' is supported")

    # Initialize image publisher
    image_publisher = ImagePublisher()
    image_publisher.start()

    while True:
        obs = env.reset()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]
        # Loop until we get a reset from the input or the task completes
        while True:
            start = time.time()

            # Set active robot
            active_robot = env.robots[device.active_robot]

            # Get the newest action
            input_ac_dict = device.input2action()

            # If action is none, then this a reset so we should break
            if input_ac_dict is None:
                break

            from copy import deepcopy

            action_dict = deepcopy(input_ac_dict)  # {}
            # set arm actions
            for arm in active_robot.arms:
                if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                    controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                else:
                    controller_input_type = active_robot.part_controllers[arm].input_type

                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError

            # Maintain gripper state for each robot but only update the active robot with action
            env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action)
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

            obs, reward, done, info = env.step(env_action)
            frame = obs[args.camera + "_image"]
            # Ensure stable orientation
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Send images to Redis Channel
                image_publisher.publish_frame(frame)
                cv2.imshow("Robot Camera", frame)
                key = cv2.waitKey(1) 

            # limit frame rate if necessary
            if args.max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / args.max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)     
        # Cleanup
        cv2.destroyAllWindows()
        # image_publisher.stop()     
