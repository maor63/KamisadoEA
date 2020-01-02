import random
from abc import ABC, abstractmethod
from KamisadoGame.kamisado import Kamisado
from collections import Counter


class MinMaxAgent(ABC):
    MAX, MIN = 1000, -1000

    def __init__(self, max_depth=3):
        self._max_depth = max_depth

    def play(self, board):
        assert isinstance(board, Kamisado)
        towers_possible_moves = board.get_possible_moves()
        moves_estimation_dict = Counter()
        for tower, possible_moves in towers_possible_moves.items():
            for possible_move in possible_moves:
                new_board = board.move_tower(tower, possible_move)
                minimax_val = self.minimax(0, new_board, False, self.MIN, self.MAX, self._max_depth, board.current_player)
                moves_estimation_dict[(tower, possible_move)] = minimax_val

        tower, move = moves_estimation_dict.most_common(1)[0][0]
        return tower, move

    @abstractmethod
    def evaluate_game(self, board, max_player):
        pass

    def minimax(self, depth, board, isMaximizingPlayer, alpha, beta, max_depth, max_player):
        assert isinstance(board, Kamisado)
        # Terminating condition. i.e
        # leaf node is reached
        if depth == max_depth or board.is_game_won():
            if board.is_game_won():
                return self.MAX if board.is_game_won() == max_player else self.MIN
            else:
                return self.evaluate_game(board, max_player)

        if isMaximizingPlayer:
            return self.max_step(alpha, beta, board, depth, isMaximizingPlayer, max_depth, max_player)
        else:
            return self.min_step(alpha, beta, board, depth, isMaximizingPlayer, max_depth, max_player)

    def min_step(self, alpha, beta, board, depth, isMaximizingPlayer, max_depth, max_player):
        return self.iterate_children(alpha, self.MAX, beta, board, depth, self._eval_min_alpha_beta_best,
                                     isMaximizingPlayer, max_depth, max_player)

    def max_step(self, alpha, beta, board, depth, isMaximizingPlayer, max_depth, max_player):
        return self.iterate_children(alpha, self.MIN, beta, board, depth, self._eval_max_alpha_beta_best,
                                     isMaximizingPlayer, max_depth, max_player)

    def iterate_children(self, alpha, best, beta, board, depth, eval_alpha_beta_best, isMaximizingPlayer, max_depth, max_player):
        # iterate children
        towers_possible_moves = board.get_possible_moves()
        for tower, possible_moves in towers_possible_moves.items():
            # random.shuffle(possible_moves)
            for possible_move in possible_moves:
                new_board = board.move_tower(tower, possible_move)
                val = self.minimax(depth + 1, new_board, not isMaximizingPlayer, alpha, beta, max_depth, max_player)
                alpha, best, beta = eval_alpha_beta_best(alpha, best, beta, val)
                # Alpha Beta Pruning
                if beta <= alpha:
                    break
        return best

    def _eval_min_alpha_beta_best(self, alpha, best, beta, val):
        best = min(best, val)
        alpha, beta = alpha, min(beta, best)
        return alpha, best, beta

    def _eval_max_alpha_beta_best(self, alpha, best, beta, val):
        best = max(best, val)
        alpha, beta = max(alpha, best), beta
        return alpha, best, beta
