import sys

sys.path.append('./jetbotSim')
sys.path.append('.')

import pickle
import os, cv2
from kbhit import KBHit
kb = KBHit()
import torch
import torchvision
import torch.nn as nn
import numpy as np
import numpy.typing as npt
import abc
from random import randint
from typing import Optional

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
            if(kb.kbhit()):
                dir = kb.getch()
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
        self.action = 0

    def get_action(self, obs: npt.NDArray[np.uint8]) -> int:
        if(kb.kbhit()):
            key = kb.getch()
            if key == 'w':
                self.action = 0
            elif key == 'a':
                self.action = 1
            elif key == 'd':
                self.action = 2
            elif key == 's':
                self.action = 3
            elif key == 'p':
                self.action = 4
            elif key == 'q':
                raise KeyboardInterrupt
            else:
                raise ValueError
        return self.action
