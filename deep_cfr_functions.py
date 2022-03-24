
import datetime
from pathlib import Path

import numpy as np
import torch
from Network import AdvantageNet
import copy
from poker_eng import PokerEnv
from collections import deque

from torch.utils.data import DataLoader, Dataset

import os
import pandas as pd
from torchvision.io import read_image

class ActionDataset(Dataset):

    def __init__(self, actions, states, ts, amnts):
        self.actions = actions
        self.states = states
        self.ts = ts
        self.amnts = amnts

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.actions[idx], self.states[idx], self.ts[idx], self.amnts[idx]


class History:
    def __init__(self):
        self.env = PokerEnv([1,.5],100)
        self.env.game.reset_game([1,.5],100)


def get_state_array(env, action_player, inaction_player):
    state_array = np.zeros((13 + 4) * 7 + 3)

    state = [action_player.card_1, action_player.card_2, env.game.board, env.game.pot_size,
     action_player.stack_size, inaction_player.stack_size]

    try:
        state_array[state[0][0]] = 1
        state_array[state[0][1] + 13] = 1
        state_array[state[1][0] + 17] = 1
        state_array[state[1][1] + 17 + 13] = 1


        for i in range(len(state[2])):
            state_array[state[2][i][0] + (17 * (i + 2))] = 1
            state_array[state[2][i][1] + (17 * (i + 2)) + 13] = 1

        state_array[-3] = state[-3]
        state_array[-2] = state[-2]
        state_array[-1] = state[-1]
    except Exception:
        state_array = np.zeros((13 + 4) * 7 + 3)

    state = torch.tensor(state_array)
    return state

def non_zero_test(network):
    env = PokerEnv((.5, 1), 100)
    # Pocket Aces should be raised.
    env.game.p1.card_1 = (12, 0)
    env.game.p1.card_2 = (12, 1)

    env.game.p2.card_1 = (4, 2)
    env.game.p2.card_2 = (8, 3)
    state_array = get_state_array(env, env.game.p1, env.game.p2)
    if torch.cuda.is_available():
        state_array = state_array.to(device="cuda")
    action, amnt = network(state_array)
    if action[0] == 0 and action[1] == 0 and action[2] == 0:
        print("All zeros")


def deep_cfr_minimization(T, N):
    save_dir1 = Path("strat") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir2 = Path("adv") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir1.mkdir(parents=True)
    save_dir2.mkdir(parents=True)
    adv0 = deque(maxlen=10000000)
    adv1 = deque(maxlen=10000000)
    strat = deque(maxlen=100000000)

    strat_net = AdvantageNet((13 + 4) * 7 + 3, 256).float()

    t0 = AdvantageNet((13 + 4) * 7 + 3, 256).float()

    t1 = AdvantageNet((13 + 4) * 7 + 3, 256).float()

    gamma = .999999999
    randomness = 0

    if torch.cuda.is_available():
        print("Using cuda")
        t0 = t0.to(device="cuda")
        t1 = t1.to(device="cuda")
        strat_net = strat_net.to(device="cuda")

    for t in range(T):
        p = t % 2
        init_history = History()
        print(f"t={t}")
        np.random.choice(range(20), 10, replace=False)
        for n in range(N):
            randomness *= gamma
            init_history.env.done = False

            init_history.env.reset([1,.5], np.random.randint(10,500))

            if p == 0:
                mr = traverse(init_history, p, t0, t1, adv0, strat, t + 1,random_val=randomness)
            else:
                mr = traverse(init_history, p, t0, t1, adv1, strat, t + 1, random_val=randomness)
        if p == 0:
            print(len(adv0))
            copy_net = copy.deepcopy(t0)
            train(adv0, t0, 0)
            t1 = copy_net
        else:
            print(len(adv1))
            copy_net = copy.deepcopy(t1)
            train(adv0, t1, 0)
            t0 = copy_net
        non_zero_test(copy_net)
        print(f"Last reward: {mr}")
        torch.save(copy_net.state_dict(), str(save_dir2) + "/" + str(t))

    train(strat, strat_net, 1)
    torch.save(strat_net.state_dict(), str(save_dir1) + "/" + str(t))
    return strat_net

def traverse(h, p, t1, t2, m_adv, m_strat, t,depth = 0, random_val=.1):
    #print(depth)
    if h.env.done:
        return h.env.reward

    state = get_state_array(h.env,h.env.action_player,h.env.inaction_player)
    if torch.cuda.is_available():
        state = state.to(device="cuda")

    if h.env.action_player.pnum == p:  # Player gets to choose
        if p == 0:
            player = t1
        else:
            player = t2

        action, amnt = player(state)  # Regret matching?
        reward = 0
        rewards = []
        rand = np.random.random()
        for i in range(len(action)):
            # if depth < 2:
            #     print(f"Depth {depth}: action {i}")
            #     print(f"len adv: {len(m_adv)}")

            temp_history = copy.deepcopy(h)
            temp_history.env.step(i, amnt)

            r = traverse(temp_history, p, t1, t2, m_adv, m_strat,t,depth + 1,random_val=random_val)
            if r is None:
                r = 0
            reward += r
            rewards.append(r)

        advantages = []
        for i in range(len(action)):
            advantages.append(np.array(rewards[i]))
        advantages = np.array(advantages)
        advantages = torch.from_numpy(advantages)
        if torch.cuda.is_available():
            advantages = advantages.to(device="cuda")
        m_adv.append(np.array([state, advantages * t, t, amnt], dtype=object))
        return np.max(rewards)

    elif h.env.action_player.pnum == 1-p:
        if p == 1:
            player = t1
        else:
            player = t2

        action,amnt = player(state)
        m_strat.append(np.array([state, action * t, t, amnt]))
        sm = torch.softmax(action.cpu(), dim=0)

        r = np.random.choice([i for i in range(len(action))],
                             1,
                             p=sm.detach().numpy())


        h.env.step(r, amnt)
        return traverse(h, p, t1, t2, m_adv, m_strat,t, depth + 1,random_val=random_val)



def lossf(d1, d2):

    d1 = torch.cat(d1)
    d2 = torch.cat(d2)
    lf = torch.nn.MSELoss()
    ret = lf(d1,d2)
    return ret

def train(data, network, indicator):
    data = np.array(data, dtype=object)
    try:
        train_dataset = ActionDataset(data[:, 1], data[:, 0], data[:, 2], data[:,3])
        train_loader = DataLoader(train_dataset,
                                  batch_size=256,
                                  pin_memory=False)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        train_iters = 1000

        for i in range(train_iters):
            ts = []
            actions = []
            data_actions = []
            pred_amnts = []
            amnts = []
            optimizer.zero_grad()
            for batch, (data_action, state, t, amnt) in enumerate(train_loader):
                data_action, state, t, amnt = data_action.to(device="cuda"), state.to(device="cuda"), t.to(device="cuda"), amnt.to(device="cuda")


                action, pred_amnt = network(state)

                if indicator:
                    action = network.sm(action)

                ts.append(t)
                actions.append(action)
                data_actions.append(data_action)
                pred_amnts.append(pred_amnt)
                amnts.append(amnt)

            loss = lossf(actions, data_actions)
            if indicator == 1:
                loss += lossf(amnts, pred_amnts)

            loss = loss.clone().detach()
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
    except Exception as e:
        print("list empty")

deep_cfr_minimization(5, 150)