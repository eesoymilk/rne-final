import os

if os.name == 'nt':
    import msvcrt as getch
else:
    import getch

import abc
from random import randint
from typing import Optional

import torch
import torchvision
import torch.nn as nn
import numpy as np
import numpy.typing as npt

from jetbotSim import Env


class BaseAgent:
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
    """This agent utilizes a pre-trained ResNet18 model to predict the action to take."""

    def __init__(self, env: Env, device: Optional[str] = None):
        super().__init__(env)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = torchvision.models.resnet18(weights='DEFAULT')
        self.net.to(self.device)

    def get_action(self, obs: npt.NDArray[np.uint8]) -> int:
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
            elif key == 'a':
                return 1
            elif key == 'd':
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
