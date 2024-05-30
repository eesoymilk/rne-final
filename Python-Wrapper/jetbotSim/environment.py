import time
import json
import threading

import cv2
import websocket
import numpy as np
import numpy.typing as npt

from jetbotSim import config

SocketResponse = tuple[npt.NDArray[np.uint8], int, bool]


class Env:
    FORWARD_SPEED = 0.5
    TURN_SPEED = 0.1
    ACTIONS = {
        0: {
            "name": "forward",
            "motor_speed": (FORWARD_SPEED, FORWARD_SPEED),
        },
        1: {"name": "left", "motor_speed": (0, TURN_SPEED)},
        2: {"name": "right", "motor_speed": (TURN_SPEED, 0)},
        3: {
            "name": "backward",
            "motor_speed": (-FORWARD_SPEED, -FORWARD_SPEED),
        },
        4: {"name": "stop", "motor_speed": (0, 0)}, # Deprecated
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
        *,
        left_value: float = 0.0,
        right_value: float = 0.0,
        reset: bool = False,
    ) -> SocketResponse:
        jsonStr = json.dumps(
            {'leftMotor': left_value, 'rightMotor': right_value, 'reset': reset}
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
            return self.set_motor(*self.ACTIONS[action]["motor_speed"])
        except KeyError:
            raise ValueError(f"Invalid action: {action}")

    def reset(self) -> SocketResponse:
        return self.send_command(reset=True)
