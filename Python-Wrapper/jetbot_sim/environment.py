import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

import time
import json
import threading

import cv2
import websocket
import numpy as np
import numpy.typing as npt

try:
    from jetbot_sim.config import Config
except ImportError:
    from .config import Config

SocketResponse = tuple[npt.NDArray[np.uint8], int, bool]


class Env:
    def __init__(self, forward_speed: float = 0.5, turn_speed: float = 0.1):
        self.ws = None
        self.wst = None
        self._connect_server(Config.ip, Config.actor)
        self.buffer = None
        self.on_change = False

        self._left_motor = 0
        self._right_motor = 0

        self.actions = {
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
            4: {"name": "sharp_left", "motor_speed": (-0.7* turn_speed, 0.7* turn_speed)},
            5: {"name": "sharp_right", "motor_speed": (0.7* turn_speed, -0.7* turn_speed)}
            #6: {"name": "stop", "motor_speed": (0, 0)},  # Deprecated
        }

        self.reset()

    def _connect_server(self, ip, actor):
        self.ws = websocket.WebSocketApp(
            "ws://%s/%s/camera/subscribe" % (ip, actor),
            on_message=self._on_message_env,
        )
        self.wst = threading.Thread(target=self.ws.run_forever)
        self.wst.daemon = True
        self.wst.start()
        self.command_ws = websocket.create_connection(
            "ws://%s/%s/controller/session" % (ip, actor)
        )
        time.sleep(1)

    def _on_message_env(self, ws, msg):
        self.buffer = msg
        self.on_change = True

    def read_socket(self) -> SocketResponse:
        while True:
            if self.buffer is not None and self.on_change:
                nparr = np.fromstring(self.buffer[5:], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                reward = int.from_bytes(self.buffer[:4], 'little', signed=True)
                done = bool.from_bytes(self.buffer[4:5], 'little') or reward == -5
                self.on_change = False
                self.buffer = None
                return img.copy(), reward, done

    def _move_to_wheel(self, value: float) -> float:
        length = 2 * np.pi * Config.wheel_rad
        angular_vel = 360 * (1000 * value / length)
        return angular_vel

    def send_command(
        self,
        *,
        left_value: float = 0.0,
        right_value: float = 0.0,
        reset: bool = False,
    ) -> SocketResponse:
        jsonStr = json.dumps(
            {'leftMotor': left_value, 'rightMotor': right_value, 'reset': reset, 'flag': 0 if reset else 4} 
        )
        self.command_ws.send(jsonStr)
        return self.read_socket()

    def set_motor(self, value_l, value_r) -> SocketResponse:
        left_ang = self._move_to_wheel(value_l)
        right_ang = self._move_to_wheel(value_r)
        return self.send_command(left_value=left_ang, right_value=right_ang)

    def forward(self, value) -> SocketResponse:
        ang = self._move_to_wheel(value)
        return self.send_command(left_value=ang, right_value=ang)

    def backward(self, value) -> SocketResponse:
        ang = self._move_to_wheel(value)
        return self.send_command(left_value=-ang, right_value=-ang)

    def left(self, value) -> SocketResponse:
        ang = self._move_to_wheel(value)
        return self.send_command(left_value=-ang, right_value=ang)

    def right(self, value) -> SocketResponse:
        ang = self._move_to_wheel(value)
        return self.send_command(left_value=ang, right_value=-ang)

    def stop(self) -> SocketResponse:
        return self.send_command()

    def step(self, action: int) -> SocketResponse:
        try:
            return self.set_motor(*self.actions[action]["motor_speed"])
        except KeyError:
            raise ValueError(f"Invalid action: {action}")

    def reset(self) -> SocketResponse:
        return self.send_command(reset=True)
