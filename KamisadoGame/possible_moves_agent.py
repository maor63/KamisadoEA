from KamisadoGame.minmax_agent import MinMaxAgent
from KamisadoGame.kamisado import Kamisado, Player
from itertools import chain


class PossibleMovesAgent(MinMaxAgent):
    def evaluate_game(self, board, max_player):
        assert isinstance(board, Kamisado)
        new_board = board.clone()
        new_board.tower_can_play = ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]
        new_board.current_player = Player.WHITE
        white_possible_sum = sum([len(possible_moves) for tower, possible_moves in board.get_possible_moves().items()])

        new_board.current_player = Player.BLACK
        black_possible_sum = sum([len(possible_moves) for tower, possible_moves in board.get_possible_moves().items()])
        if max_player == Player.WHITE:
            return white_possible_sum - black_possible_sum
        else:
            return black_possible_sum - white_possible_sum
