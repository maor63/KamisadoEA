from unittest import TestCase

from KamisadoGame.kamisado import Kamisado


class TestKamisado(TestCase):
    def test_start_game_white_moves(self):
        board = Kamisado()
        white_moves = board.get_possible_moves()
        self.assertEqual(12, len(white_moves['Brown']))
        self.assertEqual(13, len(white_moves['Green']))

    def test_move_tower(self):
        board = Kamisado()
        white_moves = board.get_possible_moves()
        tower = 'Red'
        tower_old_pos = board.white_player_pos[tower]
        tower_new_pos = white_moves[tower][3]
        new_board = board.move_tower(tower, tower_new_pos)
        self.assertEqual(board.white_player_pos[tower], tower_old_pos)
        self.assertEqual(new_board.white_player_pos[tower], tower_new_pos)

    def test_second_move_black(self):
        board = Kamisado()
        white_moves = board.get_possible_moves()
        new_board = board.move_tower('Brown', (6, 0))
        black_moves = new_board.get_possible_moves()
        self.assertIn('Purple', black_moves)
        self.assertEqual(1, len(black_moves))
        self.assertEqual(13, len(black_moves['Purple']))
