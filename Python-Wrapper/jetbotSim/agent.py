import sys

sys.path.append('./jetbotSim')
import cv2
import torch
import torch.nn as nn
import numpy as np


class Agent:
    def __init__(self, env, robot, device="cpu"):
        self.env = env
        self.robot = robot
        self.device = device
        self.net: nn.Module = torch.hub.load(
            'milesial/Pytorch-UNet',
            'unet_carvana',
            pretrained=True,
            scale=0.5,
        )
        self.net.to(device)
        self.frames = 0

    def step(self, action):
        if action == 0:
            self.robot.set_motor(0.5, 0.5)
        elif action == 1:
            self.robot.set_motor(0.2, 0.0)
        elif action == 2:
            self.robot.set_motor(0.0, 0.0)

    def execute(self, obs):
        self.frames += 1
        img = obs["img"]
        out = self.net(
            torch.tensor(img, device=self.device).permute(2, 0, 1).unsqueeze(0).float() / 255
        )
        print(f"    output shape: {out.shape}")
        # cv2.imwrite(f'test_img.png', img)
        reward = obs['reward']
        done = obs['done']
        if self.frames < 100:
            self.step(0)
        else:
            self.frames = 0
            self.robot.reset()
        print(f'\rframes:{self.frames}, reward:{reward}, done:{done} ', end="")

    def run(self):
        print("\n[Start Observation]")
        while True:
            if self.env.buffer is not None and self.env.on_change:
                nparr = np.fromstring(self.env.buffer[5:], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                reward = int.from_bytes(self.env.buffer[:4], 'little')
                done = bool.from_bytes(self.env.buffer[4:5], 'little')
                self.execute(
                    {"img": img.copy(), "reward": reward, "done": done}
                )
                self.env.on_change = False
