import random

from typing import Optional
from collections import namedtuple, deque

Transition = namedtuple(
    "Transition", ("obs", "action", "next_obs", "reward", "done")
)


class ReplayBuffer:

    def __init__(self, capacity: int, batch_size: int):
        self._capacity = capacity
        self.batch_size = batch_size
        self.memory = deque([], maxlen=capacity)

    @property
    def capacity(self):
        return self._capacity

    def add(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def extend(self, transitions):
        """Save a list of transitions"""
        self.memory.extend(transitions)

    def sample(self, batch_size: Optional[int] = None):
        return random.sample(
            self.memory,
            self.batch_size if batch_size is None else batch_size,
        )

    def __len__(self):
        return len(self.memory)
