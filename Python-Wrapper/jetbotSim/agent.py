import sys

sys.path.append('.')

import abc
import numpy as np
import numpy.typing as npt

from robot import Robot
from environment import Env


class Agent:
    def __init__(self, env: Env, robot: Robot):
        self.env = env
        self.robot = robot
        self.frames = 0

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
            print(f'\rframes:{self.frames}, reward:{reward}', end=" ")
            if not done:
                self.frames += 1
                action = self.get_action(obs)
                self.step(action)
            else:
                self.frames = 0
                print(f'\r, done:{done}')
                self.robot.reset()
