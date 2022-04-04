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
        self.inact_reward = 0
        self.last_action = 0




    def reset(self, blinds, stack_size):
        self.game.reset_game(blinds, stack_size)
        return self._next_observation()

    def get_state(self):
        return [self.action_player.card_1, self.action_player.card_2, self.game.board, self.game.pot_size,
                self.action_player.stack_size, self.inaction_player.stack_size, self.game.to_call, self.action_player.was_last_raiser, self.last_action]

    def _next_observation(self):
        if self.game.action_player == 0:
            self.action_player = self.game.p1
            self.inaction_player = self.game.p2
        else:
            self.action_player = self.game.p2
            self.inaction_player = self.game.p1

        return [self.action_player.card_1, self.action_player.card_2, self.game.board, self.game.pot_size,
     self.action_player.stack_size, self.inaction_player.stack_size, self.game.to_call, self.action_player.was_last_raiser,
             self.last_action]

    def action_to_amnt(self, action):
        if action !=0 and action != 1:
            self.action_player.was_last_raiser = 1
            self.inaction_player.was_last_raiser = 0

        amnt = 0
        if action == 2:
            amnt = max(self.game.pot_size, 1)
        elif action == 3:
            amnt = max(self.game.pot_size * 3, 1)
        elif action == 4:
            amnt = max(self.game.pot_size * 3, 1)
        elif action == 5:
            amnt = max(self.game.pot_size * 1000, 1)
        return amnt


    def step(self, action):

        self.action_player.last_win = 0
        self.inaction_player.last_win = 0
        self._next_observation()
        amnt = self.action_to_amnt(action)
        taction = min(action, 2)
        self._take_action(taction, amnt)



        self.reward = self.action_player.last_win
        self.inact_reward = self.inaction_player.last_win
        # reward -= abs(action_player.expected_ev - action[-1])
        self.done = True if self.game.pot_size == 0 else False


        # if self.done and action != 0:
        #     winner = self.game.determine_winner()
        #     if winner == "tie":
        #         self.game.p1.stack_size += self.game.pot_size / 2
        #         self.game.p2.stack_size += self.game.pot_size / 2
        #     elif winner == "p1":
        #         self.game.p1.stack_size += self.game.pot_size
        #     elif winner == "p2":
        #         self.game.p2.stack_size += self.game.pot_size
        #     self.game.pot_size = 0
        #     return



        self.action_player.last_win = 0
        self.inaction_player.last_win = 0

        if self.game.action_player == 0: # FIXME
            self.action_player = self.game.p1
            self.inaction_player = self.game.p2
        else:
            self.action_player = self.game.p2
            self.inaction_player = self.game.p1

        self.state = [self.action_player.card_1, self.action_player.card_2, self.game.board, self.game.pot_size,
                      self.action_player.stack_size, self.inaction_player.stack_size]
        self.last_action = action
        return self.reward, self.done, [self.action_player.card_1, self.action_player.card_2, self.game.board, self.game.pot_size,
                                   self.action_player.stack_size, self.inaction_player.stack_size]

    def _take_action(self, action,amnt):
        self.game.take_action(action,amnt)


    def render(self, mode='human'):
        print("Nothing for now")