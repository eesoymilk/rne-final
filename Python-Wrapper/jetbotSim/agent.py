import sys

sys.path.append('.')

import os

if os.name == 'nt':
    import msvcrt as getch
else:
    import getch

import abc
from random import randint
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt

from robot import Robot
from environment import Env


class BaseAgent:
    def __init__(self, env: Env, robot: Robot):
        self.env = env
        self.robot = robot
        self.frames = 0
        self.frame_text: str = ""

    @abc.abstractmethod
    def get_action(self, obs: npt.NDArray[np.uint8]) -> int:
        pass

    def step(self, action):
        if action == 0:
            self.robot.set_motor(0.5, 0.5)
        elif action == 1:
            self.robot.set_motor(0.2, 0)
        elif action == 2:
            self.robot.set_motor(0, 0.2)
        elif action == 3:
            self.robot.set_motor(-0.2, -0.2)
        elif action == 4:
            self.robot.set_motor(0, 0)

    def run(self):
        print("\n[Start Observation]")

        self.robot.reset()
        while True:
            obs, reward, done = self.env.read_socket()
            self.frame_text = f'frames:{self.frames}, reward:{reward}'
            if not done:
                self.frames += 1
                action = self.get_action(obs)
                self.frame_text = f"{self.frame_text}, action:{action}"
                self.step(action)
            else:
                self.frames = 0
                self.frame_text = f"{self.frame_text}, done:{done}"
                self.robot.reset()

            print(f"\r{self.frame_text}")


class Agent(BaseAgent):
    """This agent utilizes a pre-trained U-Net model to segment the image and determine the action to take."""

    def __init__(self, env: Env, robot: Robot, device: Optional[str] = None):
        super().__init__(env, robot)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":
            # Manually load the model and checkpoint since torch.hub.load does not work on CPU
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
        self.net.to(self.device)

    def get_action(self, obs: npt.NDArray[np.uint8]) -> int:
        segmented_obs = self.net(
            torch.tensor(obs, device=self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255
        )
        # return randint for now, will work on model later
        return randint(0, 4)


class HumanAgent(BaseAgent):

    def __init__(self, env: Env, robot: Robot):
        super().__init__(env, robot)
        self.stdscr = None

    def get_action(self, obs: npt.NDArray[np.uint8]) -> int:
        try:
            key = getch.getch().decode('utf-8')

            if key == 'w':
                return 0
            elif key == 'd':
                return 1
            elif key == 'a':
                return 2
            elif key == 's':
                return 3
            elif key == 'p':
                return 4
            elif key == 'q':
                raise KeyboardInterrupt
            else:
                raise ValueError
        except (UnicodeDecodeError, ValueError):
            return 0
