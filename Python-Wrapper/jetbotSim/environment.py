import sys

sys.path.append('./jetbotSim')
import numpy as np
import cv2
import websocket
from websocket import create_connection
import threading
import time
import config
import json

import numpy.typing as npt


class Env:
    ACTIONS = {
        0: {"name": "forward", "motor_speed": (0.5, 0.5)},
        1: {"name": "right", "motor_speed": (0.2, 0)},
        2: {"name": "left", "motor_speed": (0, 0.2)},
        3: {"name": "backward", "motor_speed": (-0.2, -0.2)},
        4: {"name": "stop", "motor_speed": (0, 0)},
    }

    def __init__(self):
        self.ws = None
        self.wst = None
        self._connect_server(config.ip, config.actor)
        self.buffer = None
        self.on_change = False

        self._left_motor = 0
        self._right_motor = 0
        self.reset()

    def _connect_server(self, ip, actor):
        self.ws = websocket.WebSocketApp(
            "ws://%s/%s/camera/subscribe" % (ip, actor),
            on_message=self._on_message_env,
            # on_message=lambda ws, msg: self._on_message_env(ws, msg),
        )
        self.command_ws = create_connection(
            "ws://%s/%s/controller/session" % (ip, actor)
        )
        self.wst = threading.Thread(target=self.ws.run_forever)
        self.wst.daemon = True
        self.wst.start()
        time.sleep(1)

    def _on_message_env(self, ws, msg):
        self.buffer = msg
        self.on_change = True

    def read_socket(self) -> tuple[npt.NDArray[np.uint8], int, bool]:
        while True:
            if self.buffer is not None and self.on_change:
                nparr = np.fromstring(self.buffer[5:], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                reward = int.from_bytes(self.buffer[:4], 'little', signed=True)
                done = bool.from_bytes(self.buffer[4:5], 'little')
                self.on_change = False
                self.buffer = None
                return img.copy(), reward, done

    def _move_to_wheel(self, value: float) -> float:
        length = 2 * np.pi * config.wheel_rad
        angular_vel = 360 * (1000 * value / length)
        return angular_vel

    def send_command(
        self,
        left_value: float,
        right_value: float,
        flag: int,
    ) -> tuple[npt.NDArray[np.uint8], int, bool]:
        jsonStr = json.dumps(
            {'leftMotor': left_value, 'rightMotor': right_value, 'flag': flag}
        )
        self.command_ws.send(jsonStr)
        return self.read_socket()

    def set_left_motor(self, value):
        left_ang = self._move_to_wheel(value)
        self.send_command(left_ang, 0.0, 1)

    def set_right_motor(self, value):
        right_ang = self._move_to_wheel(value)
        self.send_command(0.0, right_ang, 2)

    def set_motor(self, value_l, value_r):
        left_ang = self._move_to_wheel(value_l)
        right_ang = self._move_to_wheel(value_r)
        return self.send_command(left_ang, right_ang, 4)

    def add_motor(self, value_l, value_r):
        left_ang = self._move_to_wheel(value_l)
        right_ang = self._move_to_wheel(value_r)
        self.send_command(left_ang, right_ang, 3)

    def forward(self, value):
        ang = self._move_to_wheel(value)
        self.send_command(ang, ang, 4)

    def backward(self, value):
        ang = self._move_to_wheel(value)
        self.send_command(-ang, -ang, 4)

    def left(self, value):
        ang = self._move_to_wheel(value)
        self.send_command(-ang, ang, 4)

    def right(self, value):
        ang = self._move_to_wheel(value)
        self.send_command(ang, -ang, 4)

    def step(self, action: int) -> tuple[npt.NDArray[np.uint8], int, bool]:
        try:
            return self.set_motor(*self.ACTIONS[action]["motor_speed"])
        except KeyError:
            raise ValueError(f"Invalid action: {action}")

    def stop(self) -> tuple[npt.NDArray[np.uint8], int, bool]:
        return self.send_command(0.0, 0.0, 4)

    def reset(self) -> tuple[npt.NDArray[np.uint8], int, bool]:
        return self.send_command(0.0, 0.0, 4)
