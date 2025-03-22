from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import pygame
import threading
import time

import mink

_HERE = Path(__file__).parent.parent
_XML = _HERE / "model" / "scene.xml"

class JoystickController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.step_size = 0.002
        self.running = True
        self.mid = model.body("target").mocapid[0]
        
        # Initialize gripper state
        self.gripper_pos = 0.0
        self.gripper_step = 0.005
        self.gripper_actuator_id = model.actuator("gripper_position").id
        
        pygame.init()
        pygame.joystick.init()
        
        while pygame.joystick.get_count() == 0:
            print("No joystick detected. Waiting for joystick connection...")
            time.sleep(1)
            pygame.joystick.quit()
            pygame.joystick.init()
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Initialized joystick: {self.joystick.get_name()}")
        
        self.joystick_thread = threading.Thread(target=self.joystick_loop)
        self.joystick_thread.daemon = True
        self.joystick_thread.start()
    
    def joystick_loop(self):
        while self.running:
            pygame.event.pump()
            
            x = self.joystick.get_axis(0)
            y = -self.joystick.get_axis(1)
            z = -self.joystick.get_axis(3)
            
            # deadzone is needed to avoid drift (or noise) in the joystick
            deadzone = 0.15
            scale = 0.5
            
            def apply_deadzone_and_scale(value, deadzone, scale):
                if abs(value) < deadzone:
                    return 0.0
                # Smooth scaling after deadzone
                sign = 1 if value > 0 else -1
                scaled = ((abs(value) - deadzone) / (1 - deadzone)) * scale
                return sign * scaled
            
            x = apply_deadzone_and_scale(x, deadzone, scale)
            y = apply_deadzone_and_scale(y, deadzone, scale)
            z = apply_deadzone_and_scale(z, deadzone, scale)
            
            # Gripper position control with R1 (button 5) and L1 (button 4)
            if self.joystick.get_button(5):  # R1 - Close gripper
                self.gripper_pos = max(-1.0, self.gripper_pos - self.gripper_step)
                self.data.ctrl[self.gripper_actuator_id] = self.gripper_pos
            elif self.joystick.get_button(4):  # L1 - Open gripper
                self.gripper_pos = min(1.0, self.gripper_pos + self.gripper_step)
                self.data.ctrl[self.gripper_actuator_id] = self.gripper_pos
            
            self.data.mocap_pos[self.mid][0] += x * self.step_size
            self.data.mocap_pos[self.mid][1] += y * self.step_size
            self.data.mocap_pos[self.mid][2] += z * self.step_size
            
            time.sleep(0.01)
    
    def cleanup(self):
        self.running = False
        self.joystick_thread.join()
        pygame.quit()

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    ## =================== ##
    ## Setup IK.
    ## =================== ##

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # Enable collision avoidance between (wrist3, floor) and (wrist3, wall).
    wrist_3_geoms = mink.get_body_geom_ids(model, model.body("wrist_3_link").id)
    collision_pairs = [
        (wrist_3_geoms, ["floor", "wall"]),
    ]

    limits = [
        mink.ConfigurationLimit(model=configuration.model),
        mink.CollisionAvoidanceLimit(
            model=configuration.model,
            geom_pairs=collision_pairs,
        ),
    ]

    max_velocities = {
        "shoulder_pan": np.pi/2,
        "shoulder_lift": np.pi/2,
        "elbow": np.pi/2,
        "wrist_1": np.pi/2,
        "wrist_2": np.pi/2,
        "wrist_3": np.pi/2,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    ## =================== ##

    controller = JoystickController(model, data)
    
    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            configuration.update(data.qpos)
            mujoco.mj_forward(model, data)

            # Initialize the mocap target at the end-effector site.
            mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

            rate = RateLimiter(frequency=500.0, warn=False)
            while viewer.is_running():
                # Update task target.
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                end_effector_task.set_target(T_wt)

                # Compute velocity and integrate into the next configuration.
                for i in range(max_iters):
                    vel = mink.solve_ik(
                        configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits
                    )
                    configuration.integrate_inplace(vel, rate.dt)
                    err = end_effector_task.compute_error(configuration)
                    pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                    ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                    if pos_achieved and ori_achieved:
                        break

                data.ctrl[:6] = configuration.q[:6]
                mujoco.mj_step(model, data)

                viewer.sync()
                rate.sleep()

    finally:
        controller.cleanup()
