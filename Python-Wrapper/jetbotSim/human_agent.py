import sys

sys.path.append('.')

import pickle
import os

if os.name == 'nt':
    import msvcrt as getch
else:
    import getch

import cv2
import torch
import numpy as np
import numpy.typing as npt
import torch.nn as nn

from agent import Agent


class HumanAgent(Agent):

    def get_action(self, obs: npt.NDArray[np.uint8]) -> int:
        dir = b'w'
        if self.frames % 2:
            dir = b'w'
            dir = getch.getch()
        if dir == b'w' or dir == 'w':
            print("Pressed w")
            self.step(0)
        elif dir == b'd' or dir == 'd':
            print("Pressed d")
            self.step(1)
        elif dir == b'a' or dir == 'a':
            print("Pressed a")
            self.step(2)
        elif dir == b's' or dir == 's':
            print("Pressed s")
            self.step(3)
        elif dir == b'p' or dir == 'p':
            print("Pressed p")
            self.step(4)
        elif dir == b'q' or dir == 'q':
            exit()
        else:
            # print("none")
            self.step(0)
        # self.step(0)
