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
            nn.Linear(num_inputs, num_outputs),
            nn.Sigmoid(),
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

        self.rl = nn.ReLU()
        self.sm = nn.Softmax(dim=1)

        self.init_weights()
        self.lrl = nn.LeakyReLU()


    def init_weights(self):
        for i in range(len(self.sig_layers)):
            if i % 2 == 0:
                torch.nn.init.xavier_uniform_(self.sig_layers[i].weight)
        for i in range(len(self.relu_layers)):
            if i % 2 == 0:
                torch.nn.init.xavier_uniform_(self.relu_layers[i].weight)

        torch.nn.init.zeros_(self.action_layer.weight)

    def forward(self, state, train=True):

        lin_out = self.relu_layers(self.sig_layers(state.float()))
        action = torch.squeeze(self.action_layer(lin_out))

        return torch.squeeze(action)

class RaiseNet(nn.Module):

    def __init__(self, num_inputs, num_outputs, load_frozen=None):
        super().__init__()

        self.sig_layers = nn.Sequential(
            nn.Linear(num_inputs, num_outputs),
            nn.Sigmoid(),
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

        self.amnt = nn.Linear(num_outputs,2)

        self.rl = nn.ReLU()
        self.sm = nn.Softmax(dim=1)

        self.init_weights()
        self.lrl = nn.LeakyReLU(negative_slope=.9)
        self.loss = None


    def init_weights(self):
        for i in range(len(self.sig_layers)):
            if i % 2 == 0:
                torch.nn.init.xavier_uniform_(self.sig_layers[i].weight)
        for i in range(len(self.relu_layers)):
            if i % 2 == 0:
                torch.nn.init.xavier_uniform_(self.relu_layers[i].weight)

        torch.nn.init.xavier_uniform_(self.amnt.weight)

    def forward(self, state, train=True):

        lin_out = self.relu_layers(self.sig_layers(state.float()))

        amnt = torch.squeeze(self.lrl(self.amnt(lin_out)))

        return torch.squeeze(amnt)

    def backward(self, retain_variables=True):
        self.loss.backward(retain_graph=retain_variables)
        return self.loss