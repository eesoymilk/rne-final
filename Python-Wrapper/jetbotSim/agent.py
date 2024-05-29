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
    ACTIONS = {
        0: {"name": "forward", "motor_speed": (0.5, 0.5)},
        1: {"name": "right", "motor_speed": (0.2, 0)},
        2: {"name": "left", "motor_speed": (0, 0.2)},
        3: {"name": "backward", "motor_speed": (-0.2, -0.2)},
        4: {"name": "stop", "motor_speed": (0, 0)},
    }

    def __init__(self, env: Env):
        self.env = env
        self.frames = 0

    @abc.abstractmethod
    def learn(
        self,
        obs: npt.NDArray[np.uint8],
        action: int,
        reward: int,
        next_obs: npt.NDArray[np.uint8],
        done: bool,
    ):
        pass

    @abc.abstractmethod
    def get_action(self, obs: npt.NDArray[np.uint8]) -> int:
        pass

    def run(self, episodes: int = 100):
        print("\n[Start Observation]")

        for ep in range(episodes):
            self.frames = 0
            obs, _, _ = self.env.reset()
            while True:
                action = self.get_action(obs)
                next_obs, reward, done = self.env.step(action)
                self.learn(obs, action, reward, next_obs, done)
                frame_text = f'frames:{self.frames}, reward:{reward}'

                if done:
                    frame_text = f"{frame_text}, done:{done}"
                    print(f"\r{frame_text}")
                    break

                self.frames += 1
                obs = next_obs
                frame_text = f"{frame_text}, action:{action}"
                print(f"\r{frame_text}")

            print(f"\n[Episode {ep + 1}/{episodes}]")


class Agent(BaseAgent):
    """This agent utilizes a pre-trained U-Net model to segment the image and determine the action to take."""

    def __init__(self, env: Env, device: Optional[str] = None):
        super().__init__(env)

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

    def __init__(self, env: Env):
        super().__init__(env)
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
