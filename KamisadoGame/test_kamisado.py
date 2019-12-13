from unittest import TestCase

from KamisadoGame.kamisado import Kamisado, Player


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

    def test_couple_of_moves(self):
        board = Kamisado()
        moves0 = board.get_possible_moves()
        board = board.move_tower('Red', (5, 2))
        moves1 = board.get_possible_moves()
        board = board.move_tower('Brown', (4, 3))
        moves2 = board.get_possible_moves()
        board = board.move_tower('Brown', (6, 1))
        moves3 = board.get_possible_moves()
        self.assertEqual(13, len(moves0['Red']))
        self.assertEqual(10, len(moves1['Brown']))
        self.assertEqual(7, len(moves2['Brown']))
        self.assertEqual(4, len(moves3['Brown']))

    def test_when_white_tower_blocked(self):
        board = Kamisado()
        board.black_player_pos.update({"Orange": (4, 2), "Blue": (4, 3), "Purple": (4, 4)})
        board.white_player_pos.update({"Orange": (5, 3)})
        board.tower_pos_set = set(board.black_player_pos.values()) | set(board.white_player_pos.values())
        board.current_player = Player.WHITE
        board.tower_can_play = ["Orange"]
        moves0 = board.get_possible_moves()
        board = board.move_tower("Orange", None)
        self.assertEqual(1, len(moves0['Orange']))
        self.assertEqual(None, moves0['Orange'][0])
        self.assertListEqual(board.tower_can_play, ['Purple'])

    def test_when_black_tower_blocked(self):
        board = Kamisado()
        board.white_player_pos.update({"Orange": (4, 2), "Blue": (4, 3), "Purple": (4, 4)})
        board.black_player_pos.update({"Orange": (3, 3)})
        board.tower_pos_set = set(board.black_player_pos.values()) | set(board.white_player_pos.values())
        board.current_player = Player.BLACK
        board.tower_can_play = ["Orange"]
        moves0 = board.get_possible_moves()
        board = board.move_tower("Orange", None)
        self.assertEqual(1, len(moves0['Orange']))
        self.assertEqual(None, moves0['Orange'][0])
        self.assertListEqual(board.tower_can_play, ['Orange'])

    def test_game_won(self):
        board = Kamisado()
        self.assertIsNone(board.is_game_won())
        board.white_player_pos.update({"Blue": (0, 3)})
        board.black_player_pos.update({})
        board.tower_pos_set = set(board.black_player_pos.values()) | set(board.white_player_pos.values())
        self.assertEqual(Player.WHITE, board.is_game_won())
        board.white_player_pos.update({"Blue": (7, 3)})
        board.black_player_pos.update({"Blue": (7, 4)})
        board.tower_pos_set = set(board.black_player_pos.values()) | set(board.white_player_pos.values())
        self.assertEqual(Player.BLACK, board.is_game_won())
