from copy import deepcopy
from typing import Literal

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
        self.lstm = self.__build_lstm()
        self.val_stream = self.__build_value_stream()
        self.adv_stream = self.__build_advantage_stream(output_dim)

        self.tgt_cnns = deepcopy(self.cnns)
        self.tgt_lstm = deepcopy(self.lstm)
        self.tgt_val_stream = deepcopy(self.val_stream)
        self.tgt_adv_stream = deepcopy(self.adv_stream)

        # Q_target parameters are frozen.
        self.__freeze_target()

    def forward(self, obs: torch.Tensor, model: Literal["online", "target"]):
        if model == "online":
            obs = self.cnns(obs)
            # print(f"{obs.shape=}")
            lstm_out, _ = self.lstm(obs)
            # print(f"{lstm_out.shape=}")
            val: torch.Tensor = self.val_stream(lstm_out)
            adv: torch.Tensor = self.adv_stream(lstm_out)
        elif model == "target":
            obs = self.tgt_cnns(obs)
            # print(f"{obs.shape=}")
            lstm_out, _ = self.tgt_lstm(obs)
            # print(f"{lstm_out.shape=}")
            val: torch.Tensor = self.tgt_val_stream(lstm_out)
            adv: torch.Tensor = self.tgt_adv_stream(lstm_out)
        else:
            raise ValueError(f"model: {model} not recognized")

        result = val + adv - adv.mean()
        # print(f"{result.shape=}")
        return result

    def sync(self):
        self.tgt_cnns.load_state_dict(self.cnns.state_dict())
        self.tgt_lstm.load_state_dict(self.lstm.state_dict())
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

    def __build_lstm(self):
        return nn.LSTM(3136, 3136, batch_first=True)

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
