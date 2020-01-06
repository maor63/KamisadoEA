from itertools import chain
from collections import Counter
from KamisadoGame.kamisado import Kamisado, Player
from KamisadoGame.minmax_agent import MinMaxAgent


class PossibleStrikingAgent(MinMaxAgent):
    def evaluate_game(self, board, max_player):
        assert isinstance(board, Kamisado)
        new_board = board.clone()
        new_board.tower_can_play = ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]
        new_board.current_player = Player.WHITE
        white_tower_moves = self.getPossibleMoves(new_board)

        new_board = board.clone()
        new_board.current_player = Player.BLACK
        black_tower_moves = self.getPossibleMoves(new_board)

        if max_player == Player.WHITE:
            white_tower_that_can_win = [tower for tower, move in white_tower_moves if move and move[0] == 0]
            black_possible_color_tiles = [tower for tower, move in white_tower_moves]
            black_color_tile_counter = Counter(black_possible_color_tiles)
            return sum([black_color_tile_counter[color] for color in white_tower_that_can_win])
        else:
            white_possible_color_tiles = [tower for tower, move in white_tower_moves]
            black_tower_that_can_win = [tower for tower, move in black_tower_moves if move and move[0] == 7]
            white_color_tile_counter = Counter(white_possible_color_tiles)
            return sum([white_color_tile_counter[color] for color in black_tower_that_can_win])

    def getPossibleMoves(self, board):
        assert isinstance(board, Kamisado)
        player_possible_moves = board.get_possible_moves()
        possible_moves = []
        for tower, moves in player_possible_moves.items():
            for move in moves:
                possible_moves.append((tower, move))
        return possible_moves
