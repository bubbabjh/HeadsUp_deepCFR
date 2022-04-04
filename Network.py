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

        self.encoder = nn.Sequential(

            nn.Linear(num_inputs, num_outputs),
            nn.LeakyReLU(negative_slope=.1),
            nn.Linear(num_outputs, num_outputs),
            nn.LeakyReLU(negative_slope=.1),
        )

        self.relu_layers = nn.Sequential(
            nn.Linear(num_outputs, 32),
            nn.LeakyReLU(negative_slope=.1),
        )

        self.action_layer = nn.Linear(32,4, bias=False)
        #self.amnt = nn.Linear(num_outputs,2)

        self.rl = nn.ReLU()
        self.sm = nn.Softmax(dim=0)

        self.init_weights()
        self.lrl = nn.LeakyReLU()

        self.drop = torch.nn.Dropout(p=.2)

        self.encoderf = copy.deepcopy(self.encoder)
        self.no_grad_relu = copy.deepcopy(self.relu_layers)
        self.no_grad_action = copy.deepcopy(self.action_layer)
        for p in self.encoderf.parameters():
            p.requires_grad = False
        for p in self.no_grad_relu.parameters():
            p.requires_grad = False
        for p in self.no_grad_action.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def update_frozen(self):
        self.encoderf = copy.deepcopy(self.encoder)
        self.no_grad_relu = copy.deepcopy(self.relu_layers)
        self.no_grad_action = copy.deepcopy(self.action_layer)
        for p in self.encoderf.parameters():
            p.requires_grad = False
        for p in self.no_grad_relu.parameters():
            p.requires_grad = False
        for p in self.no_grad_action.parameters():
            p.requires_grad = False

    def init_weights(self):
        for i in range(len(self.encoder)):
            if i % 2 == 0:
                torch.nn.init.xavier_uniform_(self.encoder[i].weight)
        for i in range(len(self.relu_layers)):
            if i % 2 == 0:
                torch.nn.init.xavier_uniform_(self.relu_layers[i].weight)

        torch.nn.init.zeros_(self.action_layer.weight)

    def forward(self, state, train=True):
        if train:
            lin_out = self.drop(self.relu_layers(self.encoder(state.float())))
            action = self.drop(torch.squeeze(self.action_layer(lin_out)))
            #amnt = torch.squeeze(self.amnt(lin_out))

            return torch.squeeze(action, dim=-1)
        else:
            lin_out = self.no_grad_relu(self.encoderf(state.float()))
            action = torch.squeeze(self.no_grad_action(lin_out))

            return torch.squeeze(action,dim=-1)
