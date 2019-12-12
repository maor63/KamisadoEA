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

    def get_possible_moves(self):
        black_tower_pos_set = set(self.black_player_pos.values())
        white_tower_pos_set = set(self.white_player_pos.values())
        tower_pos_set = black_tower_pos_set | white_tower_pos_set
        possible_moves = []
        if self.current_player == Player.WHITE:
            for tower in self.tower_can_play:
                tower_y, tower_x = self.white_player_pos[tower]
                possible_moves += [(i, tower_x) for i in range(8) if (i, tower_x) not in tower_pos_set and i < tower_y]
                sum_ = tower_y + tower_x
                right_diagonal = [(i, sum_ - i) for i in range(8) if (i, sum_ - i) not in tower_pos_set and i < tower_y]

        return possible_moves
