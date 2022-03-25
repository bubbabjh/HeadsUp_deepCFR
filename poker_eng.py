import gym
import math
import random
from HU_NLH import HeadsUpPokerGame
import numpy as np

class PokerEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, blinds, stack_size):
        super(PokerEnv, self).__init__()
        self.game = HeadsUpPokerGame(blinds, stack_size)
        self.action_player = self.game.p1
        self.inaction_player = self.game.p2
        self.done = False
        self.state = np.zeros(10)
        self.reward = 0



    def reset(self, blinds, stack_size):
        self.game.reset_game(blinds, stack_size)
        return self._next_observation()



    def _next_observation(self):
        if self.game.action_player == 0:
            self.action_player = self.game.p1
            self.inaction_player = self.game.p2
        else:
            self.action_player = self.game.p2
            self.inaction_player = self.game.p1

        return [self.action_player.card_1, self.action_player.card_2, self.game.board, self.game.pot_size,
                self.action_player.stack_size, self.inaction_player.stack_size]

    def step(self, action, amnt):

        obs = self._next_observation()
        self._take_action(action, amnt)

        self.done = True if self.action_player.last_win != 0 else False

        self.reward = self.action_player.last_win
        # reward -= abs(action_player.expected_ev - action[-1])
        self.action_player.last_win = 0

        if self.game.action_player == 0:
            self.action_player = self.game.p1
            self.inaction_player = self.game.p2
        else:
            self.action_player = self.game.p2
            self.inaction_player = self.game.p1

        self.state = [self.action_player.card_1, self.action_player.card_2, self.game.board, self.game.pot_size,
                      self.action_player.stack_size, self.inaction_player.stack_size]

        return obs, self.reward, self.done, [self.action_player.card_1, self.action_player.card_2, self.game.board, self.game.pot_size,
                                   self.action_player.stack_size, self.inaction_player.stack_size]

    def _take_action(self, action,amnt):
        self.game.take_action(action,amnt)


    def render(self, mode='human'):
        print("Nothing for now")