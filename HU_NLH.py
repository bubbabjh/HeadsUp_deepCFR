import random
import torch
import numpy as np


class Player:
    def __init__(self, stack_size,pnum):
        self.pnum = pnum
        self.card_1 = None
        self.card_2 = None
        self.stack_size = stack_size
        self.starting_stack_size = stack_size
        self.last_win = 0
        self.expected_ev = 0

    def set_cards_none(self):
        self.card_1 = None
        self.card_2 = None

class HeadsUpPokerGame:

    def __init__(self, blinds, init_stack):
        self.use_cuda = torch.cuda.is_available()
        self.num_players = 2
        self.deck = None
        self.action_finished = False
        self.pot_size = 0
        self.blinds = blinds
        self.p1 = Player(init_stack,0)
        self.p2 = Player(init_stack,1)
        self.sb_player = 0
        self.action_player = 0
        self.board = []
        self.phase = "pre-flop"
        self.started_new_phase = True
        self.all_in = False
        self.to_call = 0
        self.game_over = 0
        self.game_num = 0

        self.val_dict =    {
                                0: "2",
                                1: "3",
                                2: "4",
                                3: "5",
                                4: "6",
                                5: "7",
                                6: "8",
                                7: "9",
                                8: "T",
                                9: "J",
                                10: "Q",
                                11: "K",
                                12: "A",
                            }

        self.suits_dict =   {
                                0: "c",
                                1: "s",
                                2: "d",
                                3: "h",

                            }

        self.priority_dict =    {
                                    "SF": 8,
                                    "Quads": 7,
                                    "FH": 6,
                                    "Flush": 5,
                                    "Straight": 4,
                                    "Trips": 3,
                                    "TP": 2,
                                    "Pair": 1,
                                    "HC": 0
                                }
        self.all_cards()
        self.do_blinds()
        self.deal_cards()

    def set_game(self, deck, action_finished, pot_size, blinds, p1, p2, sb_player, action_player,board,phase,started_new_phase,all_in,to_call,game_over,game_num):
        self.use_cuda = torch.cuda.is_available()
        self.num_players = 2
        self.deck = deck
        self.action_finished = action_finished
        self.pot_size = pot_size
        self.blinds = blinds
        self.p1 = p1
        self.p2 = p2
        self.sb_player = sb_player
        self.action_player = action_player
        self.board = board
        self.phase = phase
        self.started_new_phase = started_new_phase
        self.all_in = all_in
        self.to_call = to_call
        self.game_over = game_over
        self.game_num = game_num

    def reset_game(self, blinds, init_stack):
        self.num_players = 2
        self.game_over = 0
        self.deck = None
        self.action_finished = False
        self.pot_size = 0
        self.blinds = blinds
        self.p1 = Player(init_stack, 0)
        self.p2 = Player(init_stack, 1)
        self.sb_player = 0
        self.action_player = 0
        self.board = []
        self.phase = "pre-flop"
        self.started_new_phase = True
        self.all_in = False
        self.all_cards()
        self.do_blinds()
        self.deal_cards()

    def do_blinds(self):

        self.p1.starting_stack_size = self.p1.stack_size
        self.p2.starting_stack_size = self.p2.stack_size
        if self.sb_player == 0:
            self.p1.stack_size -= self.blinds[0]
            self.p2.stack_size -= self.blinds[1]
        else:
            self.p2.stack_size -= self.blinds[0]
            self.p1.stack_size -= self.blinds[1]
        self.pot_size += self.blinds[0] + self.blinds[1]
        self.to_call = self.blinds[1] - self.blinds[0]
        if self.p1.stack_size <= 0 or self.p2.stack_size <= 0:
            self.game_over = 1

    def all_cards(self):
        deck = []
        for i in range(13):
            for j in range(4):
                deck.append((i, j))
        self.deck = deck

    def deal_cards(self):
        random.shuffle(self.deck)
        if self.sb_player == 0:
            self.p1.card_1 = self.deck.pop(0)
            self.p2.card_1 = self.deck.pop(0)
            self.p1.card_2 = self.deck.pop(0)
            self.p2.card_2 = self.deck.pop(0)
        else:
            self.p2.card_1 = self.deck.pop(0)
            self.p1.card_1 = self.deck.pop(0)
            self.p2.card_2 = self.deck.pop(0)
            self.p1.card_2 = self.deck.pop(0)

    def next_card(self):
        if self.phase == "pre-flop":
            self.started_new_phase = True
            self.phase = "flop"
            self.board.append(self.deck.pop(0))
            self.board.append(self.deck.pop(0))
            self.board.append(self.deck.pop(0))
            self.phase = "turn"
        elif self.phase == "flop":
            self.started_new_phase = True
            self.board.append(self.deck.pop(0))
            self.phase == "turn"
        elif self.phase == "turn":
            self.started_new_phase = True
            self.phase = "river"
            self.board.append(self.deck.pop(0))
        else:
            self.started_new_phase = True
            self.phase = "showdown"

    def set_action(self):
        # Only to be called at the start of a phase
        if self.phase == "pre-flop":
            self.action_player = self.sb_player
        else:
            self.action_player = 1 - self.sb_player

    def take_action(self, action, bet_size):
        self.p1.last_win = 0
        self.p2.last_win = 0
        try:



            if self.started_new_phase:
                self.set_action()

            acting_p = self.p1 if self.action_player == 0 else self.p2
            inacting_p = self.p2 if self.action_player == 0 else self.p1

            if acting_p.stack_size < 0:
                inacting_p.stack_size += acting_p.stack_size
                acting_p.stack_size = 0
            if inacting_p.stack_size < 0:
                acting_p.stack_size += inacting_p.stack_size
                inacting_p.stack_size = 0

            # If all in continue to next step. Doesn't matter who it is
            if (acting_p.stack_size <= 0) and self.phase != "showdown": # FIXME Go to showdown
                while self.phase != "showdown":
                    self.next_card()
                return
            elif acting_p.stack_size <= 0:
                winner = self.determine_winner()
                if winner == "tie":
                    self.p1.stack_size += self.pot_size / 2
                    self.p2.stack_size += self.pot_size / 2
                elif winner == "p1":
                    self.p1.stack_size += self.pot_size
                elif winner == "p2":
                    self.p2.stack_size += self.pot_size
                self.pot_size = 0
                self.p1.last_win = self.p1.stack_size - self.p1.starting_stack_size
                self.p2.last_win = self.p2.stack_size - self.p2.starting_stack_size
                self.to_call = 0
                self.all_cards()
                self.deal_cards()
                self.sb_player = 1 - self.sb_player
                self.do_blinds()
                return
            elif inacting_p.stack_size <= 0 and self.to_call <= 0 and self.phase != "showdown":
                self.next_card()
                return
            elif inacting_p.stack_size <= 0 and self.to_call <= 0:
                winner = self.determine_winner()
                if winner == "tie":
                    self.p1.stack_size += self.pot_size / 2
                    self.p2.stack_size += self.pot_size / 2
                if winner == "p1":
                    self.p1.stack_size += self.pot_size
                if winner == "p2":
                    self.p2.stack_size += self.pot_size
                self.pot_size = 0
                self.p1.last_win = self.p1.stack_size - self.p1.starting_stack_size
                self.p2.last_win = self.p2.stack_size - self.p2.starting_stack_size
                self.to_call = 0
                self.all_cards()
                self.deal_cards()
                self.sb_player = 1 - self.sb_player
                self.do_blinds()
                return

        # For index 0 of action, 0 is Fold/Check, 1 is check/call, 2 is bet/raise/re-raise
            # For index 1 of action, % of pot to bet if action index 0 is 2, or to call if action is 1
            folded = False
            chosen_action = action


            if bet_size < 1 and chosen_action == 2:
                bet_size = 1

            if chosen_action == 2 and inacting_p.stack_size <= 0:
                chosen_action = 1

            if chosen_action == 2 and self.to_call * 2 >= self.pot_size * bet_size:
                bet_size = 2 * self.to_call

            if bet_size >= acting_p.stack_size:
                bet_size = acting_p.stack_size

            if chosen_action == 0:
                if self.to_call != 0:
                    folded = True
                    self.to_call = 0
                else:
                    pass
            elif chosen_action == 1:
                if self.to_call == 0: # Check
                    self.to_call = 0
                    pass
                else: # Call
                    if acting_p.stack_size >= self.to_call:
                        acting_p.stack_size -= self.to_call
                        self.pot_size += self.to_call
                        self.to_call = 0
                        if (inacting_p.stack_size <= 0) and self.phase != "showdown":
                            while self.phase != "showdown":
                                self.next_card()

                    else: # All in
                        self.pot_size += acting_p.stack_size
                        acting_p.stack_size = 0
                        while self.phase != "showdown":
                            self.next_card()


            elif chosen_action == 2: # Raise
                if acting_p.stack_size >= bet_size:
                    acting_p.stack_size -= bet_size
                    self.pot_size += bet_size
                    self.to_call = bet_size - self.to_call

                else: # All in
                    self.pot_size += acting_p.stack_size
                    acting_p.stack_size = 0

            if not folded:
                self.action_player = 1 - self.action_player
                if chosen_action != 2 and not self.started_new_phase and self.phase != "showdown":
                    self.next_card()
                    return
                elif chosen_action != 2 and not self.started_new_phase:
                    winner = self.determine_winner()
                    if winner == "tie":
                        self.p1.stack_size += self.pot_size / 2
                        self.p2.stack_size += self.pot_size / 2
                    elif winner == "p1":
                        self.p1.stack_size += self.pot_size
                    elif winner == "p2":
                        self.p2.stack_size += self.pot_size
                    self.pot_size = 0
                    self.p1.last_win = self.p1.stack_size - self.p1.starting_stack_size
                    self.p2.last_win = self.p2.stack_size - self.p2.starting_stack_size
                    self.to_call = 0
                    self.all_cards()
                    self.deal_cards()
                    self.sb_player = 1 - self.sb_player
                    self.do_blinds()
                    return
                self.started_new_phase = False

            else:
                if self.action_player == 0: # p1 folds
                    self.p2.stack_size += self.pot_size
                if self.action_player == 1: # p2 folds
                    self.p1.stack_size += self.pot_size
                self.pot_size = 0
                self.p1.last_win = self.p1.stack_size - self.p1.starting_stack_size
                self.p2.last_win = self.p2.stack_size - self.p2.starting_stack_size
                self.to_call = 0
                self.all_cards()
                self.deal_cards()
                self.sb_player = 1 - self.sb_player
                self.do_blinds()
                self.started_new_phase = True
        except:
            print("Error occured")

    # Applies to most varients of poker (THM, CP, Stud, ...)
    staticmethod
    def best_hand(self, hand):
        hand.sort()
        hand.reverse()
        c_prev = None
        streak = 0
        starting_index = 0
        #Check royal/straight flush

        for j in range(4):
            temp_hand = hand.copy()
            num_removed = 0
            for i in range(len(temp_hand)):
                if temp_hand[i - num_removed][1] != j:
                    temp_hand.pop(i - num_removed)
                    num_removed += 1
            for i in range(len(temp_hand)):
                if i == 0:
                    streak = 1
                    c_prev = temp_hand[i]
                    continue
                if c_prev[0] - temp_hand[i][0] == 1:
                    streak += 1
                    c_prev = temp_hand[i]
                else:
                    streak = 1
                    c_prev = temp_hand[i]
                    starting_index = i
                if streak == 4 and temp_hand[i][0] == 0 and temp_hand[0][0] == 12:
                    return ["SF", temp_hand[starting_index]]
                if streak == 5:
                    return ["SF", temp_hand[starting_index]]

        streak = 0
        starting_index = 0
        c_prev = 0
        # Check quads
        for i in range(len(hand)):
            if i == 0:
                streak = 1
                c_prev = hand[i]
                continue
            if c_prev[0] == hand[i][0]:
                streak += 1
                c_prev = hand[i]
            else:
                starting_index = i
                streak = 1
                c_prev = hand[i]
            high_card = hand[0] if starting_index != 0 else hand[4]
            if streak == 4:
                return ["Quads", hand[i], high_card]

        streak = 0
        starting_index = 0
        c_prev = 0
        pair = None
        trips = None
        # Check full house
        for i in range(len(hand)):
            if i == 0:
                streak = 1
                c_prev = hand[i]
                continue
            if c_prev[0] == hand[i][0]:
                streak += 1
                c_prev = hand[i]
            else:
                streak = 1
                c_prev = hand[i]
            if streak == 2 and pair is None:
                pair = hand[i]
            if streak == 3:
                trips = hand[i]
                if pair[0] == hand[i][0]:
                    pair = None

            if trips is not None and pair is not None:
                return ["FH", trips, pair]

        # Check flush
        streak = 0
        starting_index = 0
        c_prev = 0
        # Check full house
        # sort by suit
        hand.sort(key=lambda x: x[1])

        for i in range(len(hand)):
            if i == 0:
                streak = 1
                c_prev = hand[i]
                continue
            if c_prev[1] == hand[i][1]:
                streak += 1
                c_prev = hand[i]
            else:
                streak = 1
                c_prev = hand[i]
                starting_index = i
            if streak == 5:
                return ["Flush", hand[starting_index]]

        # Check straight
        # sort by card val
        hand.sort()
        hand.reverse()

        for i in range(len(hand)):
            if i == 0:
                streak = 1
                c_prev = hand[i]
                continue
            if c_prev[0] - hand[i][0] == 1:
                streak += 1
                c_prev = hand[i]
            elif c_prev[0] != hand[i][0]:
                streak = 1
                c_prev = hand[i]
                starting_index = i

            if streak == 4 and hand[i][0] == 0 and hand[0][0] == 12:
                return ["Straight", hand[starting_index]]
            if streak == 5:
                return ["Straight", hand[starting_index]]

        streak = 0
        starting_index = 0
        c_prev = 0
        # Check trips
        for i in range(len(hand)):
            if i == 0:
                streak = 1
                c_prev = hand[i]
                continue
            if c_prev[0] == hand[i][0]:
                streak += 1
                c_prev = hand[i]
            else:
                starting_index = i
                streak = 1
                c_prev = hand[i]
            high_card = hand[0] if starting_index != 0 else hand[3]
            if starting_index == 0:
                second_high = hand[4]
            elif starting_index == 1:
                second_high = hand[5]
            else:
                second_high = hand[1]
            if streak == 3:
                return ["Trips", hand[i], high_card, second_high]

        streak = 0
        c_prev = 0
        starting_index = 0

        pair1 = None
        pair2 = None
        # Check two_pair
        for i in range(len(hand)):
            if i == 0:
                streak = 1
                c_prev = hand[i]
                continue
            if c_prev[0] == hand[i][0]:
                streak += 1
                c_prev = hand[i]
            else:
                streak = 1
                c_prev = hand[i]
            if streak == 2 and pair1 is None:
                pair1 = hand[i]
                starting_index = i - 1
            elif streak == 2:
                pair2 = hand[i]

            if starting_index > 0:
                high_card = 0
            elif i - 1 != 2:
                high_card = 2
            else:
                high_card = 4

            if pair1 is not None and pair2 is not None:
                return ["TP", pair1, pair2, hand[high_card]]

        streak = 0
        c_prev = 0
        starting_index = 0
        pair = None
        # Check pair
        for i in range(len(hand)):
            if i == 0:
                streak = 1
                c_prev = hand[i]
                continue
            if c_prev[0] == hand[i][0]:
                streak += 1
                c_prev = hand[i]
                pair = hand[i]
            else:
                streak = 1
                c_prev = hand[i]

            if i == 1:
                high_card = hand[2]
                second_high = hand[3]
                third_high = hand[4]
            elif i == 2:
                high_card = hand[0]
                second_high = hand[3]
                third_high = hand[4]
            elif i == 3:
                high_card = hand[0]
                second_high = hand[1]
                third_high = hand[4]
            else:
                high_card = hand[0]
                second_high = hand[1]
                third_high = hand[2]

            if pair is not None:
                return ["Pair", pair, high_card, second_high, third_high]

        return ["HC"] + hand[:5]


    def determine_winner(self):
        player1_hand = []
        player1_hand.append(self.p1.card_1)
        player1_hand.append(self.p1.card_2)
        player1_hand += self.board

        player2_hand = []
        player2_hand.append(self.p2.card_1)
        player2_hand.append(self.p2.card_2)
        player2_hand += self.board

        p1_best = self.best_hand(player1_hand)
        p2_best = self.best_hand(player2_hand)

        for i in range(len(p1_best)):
            p1_check = p1_best[i]
            p2_check = p2_best[i]

            if isinstance(p1_check,str):
                p1_check = (self.priority_dict[p1_check], 0)
                p2_check = (self.priority_dict[p2_check], 0)
            if p1_check > p2_check:
                return "p1"
            if p1_check < p2_check:
                return"p2"
        return "tie"

    def print_runnout(self):
        print(f"P1 hand: {self.val_dict[self.p1.card_1[0]]}{self.suits_dict[self.p1.card_1[1]]} {self.val_dict[self.p1.card_2[0]]}{self.suits_dict[self.p1.card_2[1]]}")
        print(f"P2 hand: {self.val_dict[self.p2.card_1[0]]}{self.suits_dict[self.p2.card_1[1]]} {self.val_dict[self.p2.card_2[0]]}{self.suits_dict[self.p2.card_2[1]]}")
        print(f"Board: {self.val_dict[self.board[0][0]]}{self.suits_dict[self.board[0][1]]} {self.val_dict[self.board[1][0]]}{self.suits_dict[self.board[1][1]]} "
              f"{self.val_dict[self.board[2][0]]}{self.suits_dict[self.board[2][1]]} {self.val_dict[self.board[3][0]]}{self.suits_dict[self.board[3][1]]} "
              f"{self.val_dict[self.board[4][0]]}{self.suits_dict[self.board[4][1]]}")



