import sys

sys.path.append('./jetbotSim')
import numpy as np
import cv2
import websocket
from websocket import create_connection
import threading
import time
import config

import numpy.typing as npt


class Env:
    def __init__(self):
        self.ws = None
        self.wst = None
        self._connect_server(config.ip, config.actor)
        self.buffer = None
        self.on_change = False

    def _connect_server(self, ip, actor):
        self.ws = websocket.WebSocketApp(
            "ws://%s/%s/camera/subscribe" % (ip, actor),
            on_message=self._on_message_env,
            # on_message=lambda ws, msg: self._on_message_env(ws, msg),
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
