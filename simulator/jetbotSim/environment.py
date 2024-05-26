import sys
sys.path.append('./jetbotSim')
import numpy as np
import cv2
import websocket
from websocket import create_connection
import threading
import time
import config

class Env():
    def __init__(self):    
        self.ws = None
        self.wst = None
        self._connect_server(config.ip, config.actor)
        self.buffer = None
        self.on_change = False

    def _connect_server(self, ip, actor):
        self.ws = websocket.WebSocketApp("ws://%s/%s/camera/subscribe"%(ip, actor), on_message = lambda ws,msg: self._on_message_env(ws, msg))
        self.wst = threading.Thread(target=self.ws.run_forever)
        self.wst.daemon = True
        self.wst.start()
        time.sleep(1)   #wait for connect
    
    def _on_message_env(self, ws, msg):
        self.buffer = msg
        self.on_change = True
        
    def run(self, execute):
        print("\n[Start Observation]")
        while True:
            if self.buffer is not None and self.on_change:
                nparr = np.fromstring(self.buffer[5:], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                reward = int.from_bytes(self.buffer[:4], 'little')
                done = bool.from_bytes(self.buffer[4:5], 'little')
                execute({"img":img.copy(), "reward":reward, "done":done})
                self.on_change = False
        