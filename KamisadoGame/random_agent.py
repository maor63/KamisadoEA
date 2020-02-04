import random

from KamisadoGame.kamisado import Kamisado


class RandomAgent:
    name = 'RandomAgent'

    def play(self, board):
        assert isinstance(board, Kamisado)
        tower_moves = board.get_possible_moves()
        tower = random.choice(list(tower_moves.keys()))
        move = random.choice(list(tower_moves[tower]))
        return tower, move
