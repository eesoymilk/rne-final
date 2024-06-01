import random
import numpy as np

from typing import Optional
from collections import namedtuple, deque

Transition = namedtuple(
    "Transition", ("obs", "action", "next_obs", "reward", "done")
)


class ReplayBuffer:

    def __init__(
        self,
        obs_dim: tuple[int, int, int],
        capacity: int,
        batch_size: int,
    ):
        self._capacity = capacity
        self._count = 0
        self.batch_size = batch_size

        self.observations = np.zeros((capacity, *obs_dim), dtype=np.float32)
        self.next_observations = np.zeros(
            (capacity, *obs_dim), dtype=np.float32
        )
        self.actions = np.zeros(capacity, dtype=int)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

    @property
    def capacity(self):
        return self._capacity

    def add(self, obs, next_obs, act, reward, done):
        idx = self._count % self.capacity

        self.observations[idx] = obs
        self.next_observations[idx] = next_obs
        self.actions[idx] = act
        self.rewards[idx] = reward
        self.dones[idx] = done

        self._count += 1

    def sample(self, batch_size: Optional[int] = None):
        size = min(self._count, self.capacity)
        batch_size = self.batch_size if batch_size is None else batch_size

        indices = np.random.randint(0, size, min(batch_size, size))
        return (
            self.observations[indices],
            self.next_observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
        )

    def __len__(self):
        return min(self._count, self.capacity)
