from unittest import TestCase

from KamisadoGame.kamisado import Kamisado


class TestKamisado(TestCase):
    def test_start_game_black_has_no_moves(self):
        game = Kamisado()
        black_moves = game.get_possible_moves('black')
        self.assertEqual([], black_moves)

    def test_start_game_white_has_48_moves(self):
        game = Kamisado()
        white_moves = game.get_possible_moves('white')
        self.assertEqual([((6, 0), 'Brown')], white_moves)
