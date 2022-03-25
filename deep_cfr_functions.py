import datetime
from pathlib import Path

import numpy as np
import torch
from Network import AdvantageNet, RaiseNet
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

    def copy(self):
        new_env = PokerEnv([1,.5], 100)
        new_env.game.deck = copy.deepcopy(self.env.game.deck)
        new_env.game.action_finished = copy.deepcopy(self.env.game.action_finished)
        new_env.game.blinds = copy.deepcopy(self.env.game.blinds)
        new_env.game.sb_player = copy.deepcopy(self.env.game.sb_player)
        new_env.game.board = copy.deepcopy(self.env.game.board)
        new_env.game.action_player = copy.deepcopy(self.env.game.action_player)
        new_env.game.phase = copy.deepcopy(self.env.game.phase)
        new_env.game.started_new_phase = copy.deepcopy(self.env.game.started_new_phase)
        new_env.game.all_in = copy.deepcopy(self.env.game.all_in)

        new_env.game.game_over = copy.deepcopy(self.env.game.game_over)
        new_env.game.game_num = copy.deepcopy(self.env.game.game_num)

        new_env.game.p1.pnum = 0
        new_env.game.p2.pnum = 1
        new_env.game.p1.card_1 = copy.deepcopy(self.env.game.p1.card_1)
        new_env.game.p1.card_2 = copy.deepcopy(self.env.game.p1.card_2)
        new_env.game.p2.card_1 = copy.deepcopy(self.env.game.p2.card_1)
        new_env.game.p2.card_2 = copy.deepcopy(self.env.game.p2.card_2)

        if isinstance(self.env.game.p1.stack_size, torch.Tensor):
            new_env.game.p1.stack_size = self.env.game.p1.stack_size.clone()
        else:
            new_env.game.p1.stack_size = copy.deepcopy(self.env.game.p1.stack_size)
        if isinstance(self.env.game.p2.stack_size, torch.Tensor):
            new_env.game.p2.stack_size = self.env.game.p2.stack_size.clone()
        else:
            new_env.game.p2.stack_size = copy.deepcopy(self.env.game.p2.stack_size)

        if isinstance(self.env.game.p1.starting_stack_size, torch.Tensor):
            new_env.game.p1.starting_stack_size = self.env.game.p1.starting_stack_size.clone()
        else:
            new_env.game.p1.starting_stack_size = copy.deepcopy(self.env.game.p1.starting_stack_size)
        if isinstance(self.env.game.p2.starting_stack_size, torch.Tensor):
            new_env.game.p2.starting_stack_size = self.env.game.p2.starting_stack_size.clone()
        else:
            new_env.game.p2.starting_stack_size = copy.deepcopy(self.env.game.p2.starting_stack_size)

        if isinstance(self.env.game.p1.last_win, torch.Tensor):
            new_env.game.p1.last_win = self.env.game.p1.last_win.clone()
        else:
            new_env.game.p1.last_win = copy.deepcopy(self.env.game.p1.last_win)
        if isinstance(self.env.game.p2.last_win, torch.Tensor):
            new_env.game.p2.last_win = self.env.game.p2.last_win.clone()
        else:
            new_env.game.p2.last_win = copy.deepcopy(self.env.game.p2.last_win)

        if isinstance(self.env.game.pot_size, torch.Tensor):
            new_env.game.pot_size = self.env.game.pot_size.clone()
        else:
            new_env.game.pot_size = copy.deepcopy(self.env.game.pot_size)

        if isinstance(self.env.game.to_call, torch.Tensor):
            new_env.game.to_call = self.env.game.to_call.clone()
        else:
            new_env.game.to_call = copy.deepcopy(self.env.game.to_call)

        history = History()
        history.env = new_env
        return history



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

def aces_test(network, rn):
    env = PokerEnv((.5, 1), 100)
    # Pocket Aces should be raised.
    env.game.p1.card_1 = (12, 0)
    env.game.p1.card_2 = (12, 1)

    env.game.p2.card_1 = (4, 2)
    env.game.p2.card_2 = (8, 3)
    state_array = get_state_array(env, env.game.p1, env.game.p2)
    if torch.cuda.is_available():
        state_array = state_array.to(device="cuda")
    action = network(state_array)
    amnt = rn(state_array)
    ch = torch.distributions.chi2.Chi2(amnt[0])
    amnt = ch.sample() + amnt[1]

    print(f"Pocket aces: {action} {amnt}")

def ts_test(network, rn):
    env = PokerEnv((.5, 1), 100)
    # Pocket Aces should be raised.
    env.game.p1.card_1 = (0, 0)
    env.game.p1.card_2 = (6, 1)

    env.game.p2.card_1 = (4, 2)
    env.game.p2.card_2 = (8, 3)
    state_array = get_state_array(env, env.game.p1, env.game.p2)
    if torch.cuda.is_available():
        state_array = state_array.to(device="cuda")
    action = network(state_array)
    amnt = rn(state_array)
    ch = torch.distributions.chi2.Chi2(amnt[0])
    amnt = ch.sample() + amnt[1]
    print(f"Two seven off: {action} {amnt}")


def deep_cfr_minimization(T, N):
    torch.autograd.set_detect_anomaly(True)
    save_dir1 = Path("strat") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir2 = Path("adv") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir1.mkdir(parents=True)
    save_dir2.mkdir(parents=True)
    adv0 = deque(maxlen=10000000)
    adv1 = deque(maxlen=10000000)
    rm = deque(maxlen=10000000)
    strat = deque(maxlen=100000000)

    strat_net = AdvantageNet((13 + 4) * 7 + 3, 256).float()

    t0 = AdvantageNet((13 + 4) * 7 + 3, 128).float()

    t1 = AdvantageNet((13 + 4) * 7 + 3, 128).float()

    rn = RaiseNet((13 + 4) * 7 + 3, 128).float()

    gamma = .999999999
    randomness = 0

    if torch.cuda.is_available():
        print("Using cuda")
        t0 = t0.to(device="cuda")
        t1 = t1.to(device="cuda")
        strat_net = strat_net.to(device="cuda")
        rn = rn.to(device="cuda")

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
                mr = traverse(init_history, p, t0, t1, rn, rm, adv0, strat, t + 1,random_val=randomness)
            else:
                mr = traverse(init_history, p, t0,  t1, rn, rm, adv1, strat, t + 1, random_val=randomness)
        if p == 0:
            print(len(adv0))
            train_raise(rm, rn, 100)
            t0 = train(adv0, t0, 0, 100)


            copy_net = t0
        else:
            print(len(adv1))
            train_raise(rm, rn, 100)
            t1 = train(adv1, t1, 0, 100)

            copy_net = t1

        aces_test(copy_net,rn)
        ts_test(copy_net,rn)
        print(f"Last reward: {mr}")
        torch.save(copy_net.state_dict(), str(save_dir2) + "/" + str(t))

    strat_net = train(strat, strat_net, 1,2000)
    aces_test(strat_net,rn)
    ts_test(strat_net,rn)
    torch.save(strat_net.state_dict(), str(save_dir1) + "/" + str(t))
    return strat_net

def traverse(h, p, t1, t2, rn, rm, m_adv, m_strat, t,depth = 0, random_val=.1):
    #print(depth)
    mc = False
    ai = False
    if depth > 12:
        mc = True
    if depth > 18:
        ai = True
    if depth > 25:
        rew = torch.tensor(-500)
        if torch.cuda.is_available():
            rew = rew.to(device="cuda")
        return rew
    if h.env.done and h.env.action_player.pnum == p:
        if not isinstance(h.env.reward, torch.Tensor):
            rew = torch.tensor(h.env.reward)
        else:
            rew = h.env.reward
        if torch.cuda.is_available():
            rew = rew.to(device="cuda")
        return rew
    elif h.env.done:
        if not isinstance(h.env.inaction_player.last_win, torch.Tensor):
            rew = torch.tensor(h.env.inaction_player.last_win)
        else:
            rew = h.env.inaction_player.last_win
        if torch.cuda.is_available():
            rew = rew.to(device="cuda")
        return rew

    state = get_state_array(h.env,h.env.action_player,h.env.inaction_player)
    if torch.cuda.is_available():
        state = state.to(device="cuda")

    if mc:
        mc_rewards = []
        amnt = 1 if not ai else 500
        if h.env.action_player.pnum == p:
            for i in range(3):
                temp_history = h.copy()
                temp_history.env.step(i, amnt)
                mc_rewards.append(traverse(temp_history, p, t1, t2, rn, rm, m_adv, m_strat,t, depth + 1,random_val=random_val))
            try:
                return torch.max(torch.cat(mc_rewards))
            except RuntimeError:
                return torch.tensor(0)
        else:
            for i in range(3):
                temp_history = h.copy()
                temp_history.env.step(i, amnt)
                mc_rewards.append(
                    traverse(temp_history, p, t1, t2, rn, rm, m_adv, m_strat, t, depth + 1, random_val=random_val))
            try:
                return torch.min(torch.cat(mc_rewards))
            except RuntimeError:
                return torch.tensor(0)

    if h.env.action_player.pnum == p:  # Player gets to choose
        if p == 0:
            player = t1
        else:
            player = t2

        action = player(state)  # Regret matching?
        amnt = rn(state)
        ch = torch.distributions.chi2.Chi2(amnt[0])
        amnt = ch.sample() + amnt[1]
        reward = torch.tensor(0).float()
        if torch.cuda.is_available():
            reward = reward.to(device="cuda")
        rewards = []
        rand = np.random.random()
        for i in range(len(action)):
            # if depth < 2:
            #     print(f"Depth {depth}: action {i}")
            #     print(f"len adv: {len(m_adv)}")

            temp_history = h.copy()
            temp_history.env.step(i, amnt)

            r = traverse(temp_history, p, t1, t2, rn, rm, m_adv, m_strat,t,depth + 1,random_val=random_val)

            if r is None:
                r = 0
            reward += r
            rewards.append(r)

        advantages = []
        for i in range(len(action)):

            if isinstance(rewards[i], torch.Tensor):
                advantages.append(rewards[i])
            else:
                advantages.append(torch.tensor(rewards[i], dtype=float))

        if torch.cuda.is_available():
            for i in range(len(advantages)):
                advantages[i] = advantages[i].to(device="cuda")

        if advantages[2].requires_grad:
            rm.append(np.array([state, advantages, t, amnt], dtype=object))
        m_adv.append(np.array([state, advantages, t, amnt], dtype=object))
        return torch.max(torch.tensor(rewards))

    elif h.env.action_player.pnum == 1-p:
        if p == 1:
            player = t1
        else:
            player = t2

        action = player(state)
        amnt = rn(state)
        ch = torch.distributions.chi2.Chi2(amnt[0])
        amnt = ch.sample() + amnt[1]
        m_strat.append(np.array([state, action, t, amnt]))
        sm = torch.softmax(action.cpu(), dim=0)

        r = np.random.choice([i for i in range(len(action))],
                             1,
                             p=sm.detach().numpy())


        h.env.step(r, amnt)
        return traverse(h, p, t1, t2, rn, rm, m_adv, m_strat,t, depth + 1,random_val=random_val)



def lossf(d1, d2):
    if isinstance(d1, list):
        if len(d1) > 1:
            d1 = torch.cat(d1[:-1])
        else:
            d1 = torch.cat(d1)
        d1 = d1.flatten()

    d2 = torch.cat(d2[0])


    lf = torch.nn.MSELoss()
    ret = lf(d1.flatten(),d2[:len(d1)].squeeze())
    return ret

def train(data, network, indicator, train_iters):
    data = np.array(data, dtype=object)

    train_dataset = ActionDataset(data[:, 1], data[:, 0], data[:, 2], data[:,3])
    train_loader = DataLoader(train_dataset,
                              batch_size=64,
                              pin_memory=False)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    total_loss = 0
    mu = 1
    for i in range(train_iters):
        ts = []
        actions = []
        data_actions = []

        optimizer.zero_grad()
        for batch, (data_action, state, t, amnt) in enumerate(train_loader):
            state, t, amnt =  state.to(device="cuda"), t.to(device="cuda"), amnt.to(device="cuda")
            state, t, amnt =  state.float(), t.float(), amnt.float()

            action = network(state)

            if indicator:
                action = network.sm(action)

            ts.append(t)
            actions.append(action)
            data_actions.append(data_action)

        # FIXME action reward and raise size interaction

        if indicator == 1:
            # Convert to lists to get rid of gradient
            data_action = data_action
            loss = lossf(actions, data_actions).float()


        else:
            loss = lossf(actions, data_actions).float()


        total_loss += loss.item()


        if i == train_iters - 1:
            loss.backward(retain_graph=False)
        else:
            loss.backward(retain_graph=True)
        optimizer.step()
    print(f"Average Epoch Loss: {total_loss / (i + 1)}")
    return network

def train_raise(data, network, train_iters):
    data = np.array(data, dtype=object)
    rg = False

    train_dataset = ActionDataset(data[:, 1], data[:, 0], data[:, 2], data[:,3])
    train_loader = DataLoader(train_dataset,
                              batch_size=64,
                              pin_memory=False)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    total_loss = 0
    for i in range(train_iters):

        loss = torch.tensor(0).float()

        if torch.cuda.is_available():
            loss = loss.to(device="cuda")

        for batch, (data_action, state, t, amnt) in enumerate(train_loader):

            optimizer.zero_grad()
            network.loss = -torch.mean(data_action[2])
            network.backward(True)
            optimizer.step()

        total_loss += loss.item()


    print(f"Average Epoch Loss (raise): {total_loss / (i + 1)}")
    return network

deep_cfr_minimization(100, 10)