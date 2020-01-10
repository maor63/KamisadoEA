from KamisadoGame.minmax_agent import MinMaxAgent
from KamisadoGame.kamisado import Kamisado, Player
from itertools import chain


class StrikingPositionAgent(MinMaxAgent):
    def evaluate_game(self, board, max_player):
        assert isinstance(board, Kamisado)
        new_board = board.clone()
        new_board.tower_can_play = ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]
        new_board.current_player = Player.WHITE
        white_striking_sum = len([1 for tower, pos in new_board.getPossibleMovesTuples() if pos and pos[0] == 0])

        new_board = board.clone()
        new_board.tower_can_play = ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]
        new_board.current_player = Player.BLACK
        black_striking_sum = len([1 for tower, pos in new_board.getPossibleMovesTuples() if pos and pos[0] == 7])
        if max_player == Player.WHITE:
            return white_striking_sum - black_striking_sum
        else:
            return black_striking_sum - white_striking_sum
