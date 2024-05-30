import numpy as np
from typing import Optional


class ReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int = 2_000_000,
        batch_size: int = 128,
    ):
        self._capacity = capacity
        self._count = 0
        self.batch_size = batch_size

        self.observations = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros(
            (self.capacity, obs_dim), dtype=np.float32
        )
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool)

    @property
    def capacity(self):
        return self._capacity

    def add(self, obs, act, next_obs, reward, done):
        idx = self._count % self.capacity

        self.observations[idx] = obs
        self.actions[idx] = act
        self.next_observations[idx] = next_obs
        self.rewards[idx] = reward
        self.dones[idx] = done

        self._count += 1

    def sample(self, batch_size: Optional[int] = None):
        size = min(self._count, self.capacity)
        batch_size = self.batch_size if batch_size is None else batch_size

        indices = np.random.randint(0, size, min(batch_size, size))
        return (
            self.observations[indices],
            self.actions[indices],
            self.next_observations[indices],
            self.rewards[indices],
            self.dones[indices],
        )
