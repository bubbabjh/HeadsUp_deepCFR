from collections import deque
import random
import torch
from torch import nn
from poker_eng import PokerEnv
import numpy as np
import copy
from pathlib import Path
import datetime

class AdvantageNet(nn.Module):

    def __init__(self, num_inputs, num_outputs, load_frozen=None):
        super().__init__()

        self.sig_layers = nn.Sequential(
            nn.Linear((13 + 4) * 7 + 3, 256, num_outputs),
            nn.Sigmoid(),
            nn.Linear(num_outputs, num_outputs),
            nn.Sigmoid()
        )

        self.relu_layers = nn.Sequential(
            nn.Linear(num_outputs,num_outputs),
            nn.ReLU(),
            nn.Linear(num_outputs,num_outputs),
            nn.ReLU(),
            nn.Linear(num_outputs,num_outputs),
            nn.ReLU(),
            nn.Linear(num_outputs,num_outputs),
            nn.ReLU(),
            nn.Linear(num_outputs,num_outputs),
            nn.ReLU()
        )


        self.action_layer = nn.Linear(num_outputs,3)
        self.amnt = nn.Linear(num_outputs, 1)


        self.rl = nn.ReLU()
        self.sm = nn.Softmax()


        self.init_weights()

    def init_weights(self):
        for i in range(len(self.sig_layers)):
            if i % 2 == 0:
                torch.nn.init.xavier_uniform_(self.sig_layers[i].weight)
        for i in range(len(self.relu_layers)):
            if i % 2 == 0:
                torch.nn.init.xavier_uniform_(self.relu_layers[i].weight)

        torch.nn.init.xavier_uniform_(self.action_layer.weight)

    def forward(self, state, train=True):

        lin_out = self.relu_layers(self.sig_layers(state.float()))
        action = torch.squeeze(self.rl(self.action_layer(lin_out)))
        amnt = torch.squeeze(self.rl(self.amnt(lin_out)))

        return torch.squeeze(action), torch.squeeze(amnt)

