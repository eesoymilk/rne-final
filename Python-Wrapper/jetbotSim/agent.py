import sys

sys.path.append('./jetbotSim')
sys.path.append('.')

import pickle
import os
if(os.name == 'nt'):
#     import msvcrt
# else:
    from kbhit import KBHit
    msvcrt = KBHit()
import cv2
import torch
import torch.nn as nn
import numpy as np


class Agent:
    def __init__(self, env, robot, device="cpu"):
        self.env = env
        self.robot = robot
        self.device = device

        if device == "cpu":
            self.net: nn.Module = torch.hub.load(
                'milesial/Pytorch-UNet',
                'unet_carvana',
                scale=0.5,
            )
            chkpt = torch.hub.load_state_dict_from_url(
                "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
                map_location='cpu',
            )
            self.net.load_state_dict(chkpt)
        else:
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
            self.robot.set_motor(0.1, 0.0)
        elif action == 2:
            self.robot.set_motor(0.0, 0.1)
        elif action == 3:
            self.robot.set_motor(-0.2, -0.2)
        elif action == 4:
            self.robot.set_motor(-0.0, -0.0)

    def execute(self, obs):
        self.frames += 1
        img = obs["img"]
        out = self.net(
            torch.tensor(img, device=self.device).permute(2, 0, 1).unsqueeze(0).float() / 255
        )
        cv2.imwrite(f'input1.png', img)
        cv2.imwrite(f'output0.png', out[0, 0].detach().cpu().numpy())
        cv2.imwrite(f'output1.png', out[0, 1].detach().cpu().numpy())
        reward = obs['reward']
        done = obs['done']
        if self.frames < 1000 and not done:
            # self.robot.left(10 if self.frames%4 else 0)
            # self.robot.forward(0 if self.frames%4 else 5)
            dir = b'w'
            if(msvcrt.kbhit()):
                dir = msvcrt.getch()
            if(dir == b'w' or dir == 'w'):
                print("Pressed w")
                self.step(0)
            elif(dir == b'd' or dir == 'd'):
                print("Pressed d")
                self.step(1)
            elif(dir == b'a' or dir == 'a'):
                print("Pressed a")
                self.step(2)
            elif(dir == b's' or dir == 's'):
                print("Pressed s")
                self.step(3)
            elif(dir == b'p' or dir == 'p'):
                print("Pressed p")
                self.step(4)
            elif(dir == b'q' or dir == 'q'):
                exit()
            else:
                # print("none")
                self.step(0)
            # self.step(0)
        else:
            self.frames = 0
            self.robot.reset()
        print(f'\rframes:{self.frames}, reward:{reward}', end=" ")
        if(done):
            print(f'\r, done:{done}')

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
