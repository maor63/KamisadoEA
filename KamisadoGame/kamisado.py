import numpy as np
from enum import Enum


class Player(Enum):
    WHITE = 0
    BLACK = 1


class Kamisado:
    board = np.asarray([
        ["Orange", "Blue", "Purple", "Pink", "Yellow", "Red", "Green", "Brown"],
        ["Red", "Orange", "Pink", "Green", "Blue", "Yellow", "Brown", "Purple"],
        ["Green", "Pink", "Orange", "Red", "Purple", "Brown", "Yellow", "Blue"],
        ["Pink", "Purple", "Blue", "Orange", "Brown", "Green", "Red", "Yellow"],
        ["Yellow", "Red", "Green", "Brown", "Orange", "Blue", "Purple", "Pink"],
        ["Blue", "Yellow", "Brown", "Purple", "Red", "Orange", "Pink", "Green"],
        ["Purple", "Brown", "Yellow", "Blue", "Green", "Pink", "Orange", "Red"],
        ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]
    ])
    current_player = Player.WHITE
    tower_can_play = ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]

    black_player_pos = {"Orange": (0, 0), "Blue": (0, 1), "Purple": (0, 2), "Pink": (0, 3),
                        "Yellow": (0, 4), "Red": (0, 5), "Green": (0, 6), "Brown": (0, 7)}

    white_player_pos = {"Brown": (7, 0), "Green": (7, 1), "Red": (7, 2), "Yellow": (7, 3),
                        "Pink": (7, 4), "Purple": (7, 5), "Blue": (7, 6), "Orange": (7, 7)}

    def __init__(self, black_player_pos=None, white_player_pos=None, current_player=None, tower_can_play=None):
        if black_player_pos and white_player_pos:
            self.black_player_pos = black_player_pos
            self.white_player_pos = white_player_pos
        if current_player:
            self.current_player = current_player
        if tower_can_play:
            self.tower_can_play = tower_can_play

    def is_legal_move(self, pos):
        y, x = pos
        return 0 <= y <= 7 and 0 <= x <= 7

    def get_possible_moves(self):
        black_tower_pos_set = set(self.black_player_pos.values())
        white_tower_pos_set = set(self.white_player_pos.values())
        tower_pos_set = black_tower_pos_set | white_tower_pos_set
        possible_moves_dict = {}
        if self.current_player == Player.WHITE:
            for tower in self.tower_can_play:
                possible_moves = []
                tower_y, tower_x = self.white_player_pos[tower]
                possible_moves += [(i, tower_x) for i in range(8) if i < tower_y]
                sum_ = tower_y + tower_x
                # right diagonal
                possible_moves += [(i, sum_ - i) for i in range(8) if i < tower_y]
                # left diagonal
                possible_moves += [(i, 2 * tower_x - (sum_ - i)) for i in range(8) if i < tower_y]
                possible_moves = list(filter(self.is_legal_move, possible_moves))
                possible_moves = list(filter(lambda pos: pos not in tower_pos_set, possible_moves))
                possible_moves_dict[tower] = possible_moves
        else:
            for tower in self.tower_can_play:
                possible_moves = []
                tower_y, tower_x = self.white_player_pos[tower]
                possible_moves += [(i, tower_x) for i in range(8) if i > tower_y]
                sum_ = tower_y + tower_x
                # right diagonal
                possible_moves += [(i, sum_ - i) for i in range(8) if i > tower_y]
                # left diagonal
                possible_moves += [(i, 2 * tower_x - (sum_ - i)) for i in range(8) if i > tower_y]
                possible_moves = list(filter(self.is_legal_move, possible_moves))
                possible_moves = list(filter(lambda pos: pos not in tower_pos_set, possible_moves))
                possible_moves_dict[tower] = possible_moves

        return possible_moves_dict

    def move_tower(self, tower, pos):
        assert isinstance(pos, tuple)
        new_black_player_pos = dict(self.black_player_pos)
        new_white_player_pos = dict(self.white_player_pos)
        if self.current_player == Player.WHITE:
            new_white_player_pos[tower] = pos
            current_player = Player.BLACK
        else:
            new_black_player_pos[tower] = pos
            current_player = Player.WHITE
        # tower_can_play = [boa]
        return Kamisado(new_black_player_pos, new_white_player_pos, current_player)
