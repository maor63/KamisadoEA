import random
from itertools import chain

import numpy as np
from enum import Enum


class Player(Enum):
    WHITE = 0
    BLACK = 1


class Kamisado:
    board_layout = np.asarray([
        ["Orange", "Blue", "Purple", "Pink", "Yellow", "Red", "Green", "Brown"],
        ["Red", "Orange", "Pink", "Green", "Blue", "Yellow", "Brown", "Purple"],
        ["Green", "Pink", "Orange", "Red", "Purple", "Brown", "Yellow", "Blue"],
        ["Pink", "Purple", "Blue", "Orange", "Brown", "Green", "Red", "Yellow"],
        ["Yellow", "Red", "Green", "Brown", "Orange", "Blue", "Purple", "Pink"],
        ["Blue", "Yellow", "Brown", "Purple", "Red", "Orange", "Pink", "Green"],
        ["Purple", "Brown", "Yellow", "Blue", "Green", "Pink", "Orange", "Red"],
        ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]
    ])

    def __init__(self, black_player_pos=None, white_player_pos=None, current_player=None, tower_can_play=None):
        start_towers = ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]

        start_black_player_pos = {"Orange": (0, 0), "Blue": (0, 1), "Purple": (0, 2), "Pink": (0, 3),
                                  "Yellow": (0, 4), "Red": (0, 5), "Green": (0, 6), "Brown": (0, 7)}

        start_white_player_pos = {"Brown": (7, 0), "Green": (7, 1), "Red": (7, 2), "Yellow": (7, 3),
                                  "Pink": (7, 4), "Purple": (7, 5), "Blue": (7, 6), "Orange": (7, 7)}

        self.black_player_pos = black_player_pos if black_player_pos else start_black_player_pos
        self.white_player_pos = white_player_pos if white_player_pos else start_white_player_pos
        self.players_pos = {Player.WHITE: self.white_player_pos, Player.BLACK: self.black_player_pos}
        self.current_player = current_player if current_player else Player.WHITE
        self.tower_can_play = tower_can_play if tower_can_play else start_towers
        self.tower_pos_set = set(self.black_player_pos.values()) | set(self.white_player_pos.values())
        self.possible_moves_dict = None

    def is_legal_move(self, pos):
        y, x = pos
        return 0 <= y <= 7 and 0 <= x <= 7 and pos not in self.tower_pos_set

    def get_possible_moves(self):
        if self.possible_moves_dict:
            return self.possible_moves_dict
        else:
            player_pos_dict = self.players_pos[self.current_player]
            possible_moves_dict = {}
            for tower in self.tower_can_play:
                tower_y, tower_x = player_pos_dict[tower]
                forward_moves = self.generate_forward(tower_x, tower_y)
                right_moves = self.generate_right(tower_x, tower_y)
                left_moves = self.generate_left(tower_x, tower_y)

                possible_moves = forward_moves + right_moves + left_moves

                # possible_moves = list(filter(self.is_legal_move, possible_moves))
                possible_moves = possible_moves if possible_moves else [None]
                possible_moves_dict[tower] = possible_moves

            self.possible_moves_dict = possible_moves_dict
            return possible_moves_dict

    def generate_forward(self, tower_x, tower_y):
        forward_moves = []
        for i in range(1, 8):
            if self.current_player == Player.WHITE:
                forward_move = (tower_y - i, tower_x)
            else:
                forward_move = (tower_y + i, tower_x)

            if self.is_legal_move(forward_move):
                forward_moves.append(forward_move)
            else:
                break
        return forward_moves

    def generate_right(self, tower_x, tower_y):
        right_moves = []
        for i in range(1, 8):
            if self.current_player == Player.WHITE:
                right_move = (tower_y - i, tower_x + i)
            else:
                right_move = (tower_y + i, tower_x - i)

            if self.is_legal_move(right_move):
                right_moves.append(right_move)
            else:
                break
        return right_moves

    def generate_left(self, tower_x, tower_y):
        left_moves = []
        for i in range(1, 8):
            if self.current_player == Player.WHITE:
                left_move = (tower_y - i, tower_x - i)
            else:
                left_move = (tower_y + i, tower_x + i)

            if self.is_legal_move(left_move):
                left_moves.append(left_move)
            else:
                break
        return left_moves

    def filter_moves(self, block_list, moves, tower_y):
        if self.current_player == Player.WHITE:
            return self.filer_moves_for_white(block_list, moves, tower_y)
        else:
            return self.filer_moves_for_black(block_list, moves, tower_y)

    def filer_moves_for_white(self, block_list, moves, tower_y):
        block_list = list(filter(lambda pos: pos[0] < tower_y, block_list))
        if block_list:
            min_y = max(list(zip(*block_list))[0])
        else:
            min_y = -1
        moves = list(filter(lambda pos: min_y < pos[0] < tower_y, moves))
        return moves

    def filer_moves_for_black(self, block_list, moves, tower_y):
        block_list = list(filter(lambda pos: pos[0] > tower_y, block_list))
        if block_list:
            max_y = min(list(zip(*block_list))[0])
        else:
            max_y = 8
        moves = list(filter(lambda pos: max_y > pos[0] > tower_y, moves))
        return moves

    def generate_moves_and_block_list(self, move_builder):
        block_list = []
        moves = []
        for i in range(8):
            move = move_builder(i)
            if move in self.tower_pos_set:
                block_list.append(move)
            else:
                moves.append(move)
        return block_list, moves

    def _generate_moves_for_tower(self, tower_x, tower_y, move_restriction):
        possible_moves = []
        possible_moves += [(i, tower_x) for i in range(8)]
        sum_ = tower_y + tower_x
        # right diagonal
        possible_moves += [(i, sum_ - i) for i in range(8)]
        # left diagonal
        possible_moves += [(i, 2 * tower_x - (sum_ - i)) for i in range(8)]
        possible_moves = list(filter(self.is_legal_move, possible_moves))
        possible_moves = list(filter(move_restriction, possible_moves))
        return possible_moves

    def move_tower(self, tower, pos):
        assert isinstance(pos, tuple) or pos is None
        possible_moves = self.get_possible_moves()
        if tower not in possible_moves or pos not in set(chain(*possible_moves.values())):
            tower = random.choice(list(possible_moves.keys()))
            pos = random.choice(list(possible_moves[tower]))
            return self._move_tower_to_pos(pos, tower)
        else:
            return self._move_tower_to_pos(pos, tower)

    def _move_tower_to_pos(self, pos, tower):
        new_black_player_pos = dict(self.black_player_pos)
        new_white_player_pos = dict(self.white_player_pos)
        if self.current_player == Player.WHITE:
            pos = pos if pos else new_white_player_pos[tower]
            new_white_player_pos[tower] = pos
            current_player = Player.BLACK
        else:
            pos = pos if pos else new_black_player_pos[tower]
            new_black_player_pos[tower] = pos
            current_player = Player.WHITE
        tower_can_play = [self.board_layout[pos]]
        return Kamisado(new_black_player_pos, new_white_player_pos, current_player, tower_can_play)

    def clone(self):
        return Kamisado(self.black_player_pos, self.white_player_pos, self.current_player, self.tower_can_play)

    def is_game_won(self):
        is_white_won = any([pos[0] == 0 for tower, pos in self.white_player_pos.items()])
        is_black_won = any([pos[0] == 7 for tower, pos in self.black_player_pos.items()])
        if is_white_won:
            return Player.WHITE
        elif is_black_won:
            return Player.BLACK
        else:
            return None

    def __str__(self):
        board_current_layout = np.array([[''] * 8] * 8)
        for tower, pos in self.white_player_pos.items():
            board_current_layout[pos] = f'W_{tower}'
        for tower, pos in self.black_player_pos.items():
            board_current_layout[pos] = f'B_{tower}'
        return str(board_current_layout)
