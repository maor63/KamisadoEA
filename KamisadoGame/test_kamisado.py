from unittest import TestCase

from KamisadoGame.kamisado import Kamisado


class TestKamisado(TestCase):
    def test_start_game_white_has_102_moves(self):
        game = Kamisado()
        white_moves = game.get_possible_moves()
        self.assertEqual(12, len(white_moves['Brown']))
        self.assertEqual(13, len(white_moves['Green']))

    def test_second_move_have_15_moves(self):
        game = Kamisado()
        white_moves = game.get_possible_moves()


