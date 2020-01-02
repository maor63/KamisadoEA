from KamisadoGame.minmax_agent import MinMaxAgent
from KamisadoGame.kamisado import Kamisado, Player


class TowerProgressAgent(MinMaxAgent):
    def evaluate_game(self, board, max_player):
        assert isinstance(board, Kamisado)
        white_progress_sum = sum([7 - y for tower, (y, x) in board.white_player_pos.items()])
        black_progress_sum = sum([y - 0 for tower, (y, x) in board.black_player_pos.items()])
        if max_player == Player.WHITE:
            return white_progress_sum - black_progress_sum
        else:
            return black_progress_sum - white_progress_sum
