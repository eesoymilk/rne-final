from copy import deepcopy

import torch
from torch import nn


class JetbotDDQN(nn.Module):

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 392,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cnns = self.__build_cnns()
        self.val_stream = self.__build_value_stream()
        self.adv_stream = self.__build_advantage_stream(output_dim)

        self.tgt_cnns = deepcopy(self.cnns)
        self.tgt_val_stream = deepcopy(self.val_stream)
        self.tgt_adv_stream = deepcopy(self.adv_stream)

        # Q_target parameters are frozen.
        self.__freeze_target()

    def forward(self, obs: torch.Tensor, model):
        if model == "online":
            obs = self.cnns(obs)
            val: torch.Tensor = self.val_stream(obs)
            adv: torch.Tensor = self.adv_stream(obs)
        elif model == "target":
            obs = self.tgt_cnns(obs)
            val: torch.Tensor = self.tgt_val_stream(obs)
            adv: torch.Tensor = self.tgt_adv_stream(obs)
        else:
            raise ValueError(f"model: {model} not recognized")

        return val + adv - adv.mean()

    def sync(self):
        self.tgt_cnns.load_state_dict(self.cnns.state_dict())
        self.tgt_val_stream.load_state_dict(self.val_stream.state_dict())
        self.tgt_adv_stream.load_state_dict(self.adv_stream.state_dict())

    def __build_cnns(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def __build_value_stream(self):
        return nn.Sequential(
            nn.Linear(3136, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def __build_advantage_stream(self, output_dim: int):
        return nn.Sequential(
            nn.Linear(3136, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def __freeze_target(self):
        for p in self.tgt_cnns.parameters():
            p.requires_grad = False

        for p in self.tgt_val_stream.parameters():
            p.requires_grad = False

        for p in self.tgt_adv_stream.parameters():
            p.requires_grad = False
