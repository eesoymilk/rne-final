from jetbot import Robot, Camera
import time

class RealEnv:
    def __init__(self, forward_speed: float = 0.5, turn_speed: float = 0.1):
        self.robot = Robot()
        self.actions = {
            # 0: {
            #     "name": "forward",
            #     "motor_speed": (forward_speed * 1.1, forward_speed),
            # },
            # 1: {"name": "left", "motor_speed": (-turn_speed, turn_speed)},
            # 2: {"name": "right", "motor_speed": (turn_speed, -turn_speed)},
            # 3: {
            #     "name": "backward",
            #     "motor_speed": (-0.2, -0.2),
            # },
            
            0: {
                "name": "forward",
                "motor_speed": (forward_speed, forward_speed),
            },
            1: {"name": "left", "motor_speed": (0, turn_speed)},
            2: {"name": "right", "motor_speed": (turn_speed, 0)},
            3: {
                "name": "backward",
                "motor_speed": (-0.2, -0.2),
            },
            4: {"name": "sharp_left", "motor_speed": (-0.8* turn_speed, 0.8* turn_speed)},
            5: {"name": "sharp_right", "motor_speed": (0.8* turn_speed, -0.8* turn_speed)},
            # 6: {"name": "stop", "motor_speed": (0, 0)},  # Deprecated
        }
        self.camera = Camera.instance(width=224, height=224)

    def set_motor(self, value_l, value_r):
        self.robot.set_motors(value_l, value_r)
    
    def reset(self):
        obs = self.camera.value
        return obs
    
    def step(self, action):
        try:
            self.set_motor(*self.actions[action]["motor_speed"])
            print(f'\tmove: {self.actions[action]["name"]}  :  {self.actions[action]["motor_speed"]}')
            # if action in [0, 3]: # move
            #     time.sleep(0.4)
            # else:
            #     time.sleep(0.2)
            time.sleep(0.3)
            self.set_motor(0, 0) # stop
            time.sleep(0.3)
            next_obs = self.camera.value
            return next_obs, 0, False
        except KeyError:
            raise ValueError(f"Invalid action: {action}")
        
    def close(self):
        self.robot.stop()
        self.camera.stop()
