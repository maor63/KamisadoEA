import random
from itertools import chain, product
import dask
from dask.distributed import Client, progress
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

    def __init__(self, black_player_pos=None, white_player_pos=None, current_player=None, tower_can_play=None,
                 init_board=None):
        start_towers = ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]

        if init_board:
            assert len(set(init_board)) == 8
            self.board_layout = self.board_layout[init_board]
        self.init_board = init_board

        if black_player_pos:
            self.black_player_pos = black_player_pos
        else:
            self.black_player_pos = {color: (0, i) for i, color in enumerate(self.board_layout[0])}

        if white_player_pos:
            self.white_player_pos = white_player_pos
        else:
            self.white_player_pos = {color: (7, i) for i, color in enumerate(self.board_layout[7])}


        self.players_pos = {Player.WHITE: self.white_player_pos, Player.BLACK: self.black_player_pos}
        self.current_player = current_player if current_player else Player.WHITE
        self.tower_can_play = tower_can_play if tower_can_play else start_towers
        self.tower_pos_set = set(self.black_player_pos.values()) | set(self.white_player_pos.values())
        self.possible_moves_dict = None
        self.possible_moves_tuples = None

    def is_legal_move(self, pos):
        y, x = pos
        return 0 <= y <= 7 and 0 <= x <= 7 and pos not in self.tower_pos_set

    def get_possible_moves(self):
        if self.possible_moves_dict:
            return self.possible_moves_dict
        else:
            player_pos_dict = self.players_pos[self.current_player]
            # possible_moves_dict = {}
            possible_moves_tuples = []
            for tower in self.tower_can_play:
                tower_y, tower_x = player_pos_dict[tower]
                forward_moves = self.generate_forward(tower_x, tower_y)
                right_moves = self.generate_right(tower_x, tower_y)
                left_moves = self.generate_left(tower_x, tower_y)

                possible_moves = self.combine_moves(forward_moves, left_moves, right_moves)
                possible_moves_tuples.append((tower, possible_moves))
            self.possible_moves_dict = dict(possible_moves_tuples)
            return self.possible_moves_dict

    def combine_moves(self, forward_moves, left_moves, right_moves):
        possible_moves = forward_moves + right_moves + left_moves
        possible_moves = possible_moves if possible_moves else [None]
        return possible_moves

    def getPossibleMovesTuples(self):
        if not self.possible_moves_tuples:
            player_possible_moves = self.get_possible_moves()
            possible_moves = []
            for tower, moves in player_possible_moves.items():
                for move in moves:
                    possible_moves.append((tower, move))
            self.possible_moves_tuples = possible_moves
        return self.possible_moves_tuples

    def generate_forward(self, tower_x, tower_y):
        old_forward_moves = []
        for i in range(1, 8):
            if self.current_player == Player.WHITE:
                forward_move = (tower_y - i, tower_x)
            else:
                forward_move = (tower_y + i, tower_x)

            if self.is_legal_move(forward_move):
                old_forward_moves.append(forward_move)
            else:
                break
        return old_forward_moves

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

    def move_tower(self, tower, pos):
        assert isinstance(pos, tuple) or pos is None
        possible_moves = self.get_possible_moves()
        if tower in possible_moves and pos in set(possible_moves[tower]):
            return self._move_tower_to_pos(pos, tower)
        else:
            return self

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
        return Kamisado(new_black_player_pos, new_white_player_pos, current_player, tower_can_play, self.init_board)

    def clone(self):
        return Kamisado(self.black_player_pos, self.white_player_pos, self.current_player, self.tower_can_play,
                        self.init_board)

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
