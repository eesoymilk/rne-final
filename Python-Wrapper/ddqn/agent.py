import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))


import cv2
import torch
from torch import Tensor
import numpy as np
import numpy.typing as npt
import abc
from typing import Optional

try:
    from utils.kbhit import KBHit
    from utils.segment import segment
    from ddqn.ddqn import JetbotDDQN
    from ddqn.replay_buffer import ReplayBuffer
    from jetbot_sim.environment import Env
except ImportError:
    # This will trigger when running the script directly
    from ..utils.kbhit import KBHit
    from ..utils.segment import segment
    from ..jetbot_sim.environment import Env
    from .ddqn import JetbotDDQN
    from .replay_buffer import ReplayBuffer


class BaseAgent:
    def __init__(self, env: Env):
        self.env = env
        self.frames = 0
        self.processed_dim = (84, 84)

    @abc.abstractmethod
    def cache(
        self,
        obs: npt.NDArray[np.float32],
        next_obs: npt.NDArray[np.float32],
        action: int,
        reward: float,
        done: bool,
    ):
        pass

    @abc.abstractmethod
    def learn(self):
        pass

    @abc.abstractmethod
    def get_action(self, obs: npt.NDArray[np.uint8]) -> int:
        pass

    def preprocess(self, obs: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        return segment(cv2.resize(obs, self.processed_dim))

    def run(self, episodes: int = 100):
        print("\n[Start Observation]")
        i = 0
        for ep in range(episodes):
            self.frames = 0
            obs, _, _ = self.env.reset()
            obs = self.preprocess(obs)
            while True:
                action = self.get_action(obs)
                next_obs, reward, done = self.env.step(action)
                next_obs = self.preprocess(next_obs)
                self.cache(obs, next_obs, action, reward, done)
                self.learn()
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

    def __init__(
        self,
        env: Env,
        obs_dim: tuple[int, int, int],
        action_dim: int,
        save_dir: Path,
        batch_size: int = 32,
        memory_size: int = 1_000_000,
        gamma: float = 0.99,
        learning_rate: float = 0.001,
        burnin: int = 32,
        sync_every: int = 10_000,
        learn_every: int = 4,
        save_every: int = 10_000,
        exploration_rate: float = 1.0,
        exploration_rate_decay: int = 0.999975,
        exploration_rate_min: float = 0.1,
        device: Optional[str] = None,
        checkpoint: Optional[Path] = None,
    ):
        super().__init__(env)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.memory = ReplayBuffer(capacity=memory_size, batch_size=batch_size)

        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min
        self.gamma = gamma

        self.curr_step = 0
        self.burnin = burnin  # min. experiences before training
        self.learn_every = learn_every
        self.sync_every = sync_every
        self.save_every = save_every
        self.save_dir = save_dir

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = JetbotDDQN(action_dim).float()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.net = self.net.to(device=self.device)

        if checkpoint is not None:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=learning_rate,
        )
        self.loss_fn = torch.nn.SmoothL1Loss()

    def get_action(self, observation: npt.NDArray[np.float32]) -> int:
        """
        Given a observation, choose an epsilon-greedy action and update value of step.

        Inputs:
        observation(NDArray[np.float32]): A single observation of the current obs, dimension is (obs_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            obs_tensor = (
                torch.FloatTensor(observation)
                .to(device=self.device)
                .unsqueeze(0)
            )
            action_values = self.net(obs_tensor, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate_min, self.exploration_rate
        )

        self.curr_step += 1
        return action_idx

    def cache(
        self,
        obs: npt.NDArray[np.float32],
        next_obs: npt.NDArray[np.float32],
        action: int,
        reward: float,
        done: bool,
    ):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        obs (NDArray[np.float32]),
        next_obs (NDArray[np.float32]),
        action (int),
        reward (float),
        done (bool)
        """
        obs = torch.FloatTensor(obs)
        next_obs = torch.FloatTensor(next_obs)
        action = torch.IntTensor([action])
        reward = torch.DoubleTensor([reward])
        done = torch.BoolTensor([done])

        self.memory.add(obs, next_obs, action, reward, done)

    def recall(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size)
        obs, next_obs, action, reward, done = map(torch.stack, zip(*batch))

        return (
            obs.to(device=self.device),
            next_obs.to(device=self.device),
            action.to(device=self.device).squeeze(),
            reward.to(device=self.device).squeeze(),
            done.to(device=self.device).squeeze(),
        )

    def td_estimate(self, obs: Tensor, action: Tensor):
        # Q_online(s,a)
        current_Q: Tensor = self.net(obs, model="online")[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward: Tensor, next_obs: Tensor, done: Tensor):
        next_obs_Q: Tensor = self.net(next_obs, model="online")
        best_action: Tensor = torch.argmax(next_obs_Q, axis=1)
        next_Q: Tensor = self.net(next_obs, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate: Tensor, td_target: Tensor):
        loss: Tensor = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.net.sync()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        obs, next_obs, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(obs, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_obs, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def train(self, n_steps: int = 50_000_001):
        episode, episode_steps, episode_reward = 0, 0, 0
        qs, losses = [], []
        obs, _, _ = self.env.reset()
        obs = self.preprocess(obs)
        for current_step in range(n_steps):
            action = self.get_action(obs)
            next_obs, reward, done = self.env.step(action)
            next_obs = self.preprocess(next_obs)

            self.cache(obs, next_obs, action, reward, done)
            q, loss = self.learn()

            episode_steps += 1
            episode_reward += reward

            if q is not None and loss is not None:
                qs.append(q)
                losses.append(loss)

            if done:
                msg = [
                    f"============ Episode {episode + 1} ============",
                    f"Steps: {current_step}",
                    f"Episode Steps: {episode_steps}",
                    f"Reward: {episode_reward}",
                ]
                if qs and losses:
                    msg.append(f"Average Q: {np.mean(qs):.6f}")
                    msg.append(f"Average Loss: {np.mean(losses):.6f}")

                print("\n".join(msg))

                episode += 1
                episode_steps, episode_reward = 0, 0
                qs, losses = [], []

                obs, _, _ = self.env.reset()
                obs = self.preprocess(obs)
            else:
                obs = next_obs

    def save(self, verbose: bool = False):
        save_path = (
            self.save_dir
            / f"ddqn_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(
                cnns=self.net.cnns.state_dict(),
                value_stream=self.net.val_stream.state_dict(),
                advantage_stream=self.net.adv_stream.state_dict(),
                exploration_rate=self.exploration_rate,
            ),
            save_path,
        )

        if verbose:
            print(f"MarioDDQN saved to {save_path} at step {self.curr_step}")

    def load(self, load_path: Path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp: dict = torch.load(load_path, map_location=self.device)
        exploration_rate = ckp.get("exploration_rate")
        cnns = ckp.get("cnns")
        val_stream = ckp.get("value_stream")
        adv_stream = ckp.get("advantage_stream")

        print(
            f"Loading model at {load_path} with exploration rate {exploration_rate}"
        )
        self.net.cnns.load_state_dict(cnns)
        self.net.val_stream.load_state_dict(val_stream)
        self.net.adv_stream.load_state_dict(adv_stream)
        self.net.sync()
        self.exploration_rate = exploration_rate


class HumanAgent(BaseAgent):

    def __init__(self, env: Env):
        super().__init__(env)
        self.kb = KBHit()
        self.action = 0

    def get_action(self, obs: npt.NDArray[np.uint8]) -> int:
        if self.kb.kbhit():
            key = self.kb.getch()
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
        return self.action