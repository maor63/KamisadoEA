from unittest import TestCase
from collections import defaultdict
from KamisadoGame.kamisado import Kamisado, Player


class TestKamisado(TestCase):
    # def test_fast_possible_moves_load(self):
    #     board1 = Kamisado()
    #     board1.seen_boards_dict = defaultdict(dict)
    #     pm1 = board1.get_possible_moves()
    #     self.assertEqual(1, len(board1.seen_boards_dict.keys()))
    #     board2 = Kamisado()
    #     pm2 = board2.get_possible_moves()
    #     self.assertEqual(1, len(board1.seen_boards_dict.keys()))
    #
    #     init_board = [2, 0, 1, 6, 4, 7, 5, 3]
    #     board3 = Kamisado(init_board=init_board)
    #     pm3 = board3.get_possible_moves()
    #     self.assertEqual(1, len(board1.seen_boards_dict.keys()))
    #     self.assertListEqual(pm1['Brown'], pm3['Pink'])

    def test_init_kamisado_with_random_board(self):
        board = Kamisado()
        start_black_player_pos = {"Orange": (0, 0), "Blue": (0, 1), "Purple": (0, 2), "Pink": (0, 3),
                                  "Yellow": (0, 4), "Red": (0, 5), "Green": (0, 6), "Brown": (0, 7)}
        start_white_player_pos = {"Brown": (7, 0), "Green": (7, 1), "Red": (7, 2), "Yellow": (7, 3),
                                  "Pink": (7, 4), "Purple": (7, 5), "Blue": (7, 6), "Orange": (7, 7)}
        self.assertDictEqual(start_white_player_pos, board.white_player_pos)
        self.assertDictEqual(start_black_player_pos, board.black_player_pos)

        init_board = [2, 0, 1, 6, 4, 7, 5, 3]
        board = Kamisado(init_board=init_board)
        start_black_player_pos = {"Green": (0, 0), "Pink": (0, 1), "Orange": (0, 2), "Red": (0, 3),
                                  "Purple": (0, 4), "Brown": (0, 5), "Yellow": (0, 6), "Blue": (0, 7)}
        start_white_player_pos = {"Pink": (7, 0), "Purple": (7, 1), "Blue": (7, 2), "Orange": (7, 3),
                                  "Brown": (7, 4), "Green": (7, 5), "Red": (7, 6), "Yellow": (7, 7)}
        self.assertDictEqual(start_white_player_pos, board.white_player_pos)
        self.assertDictEqual(start_black_player_pos, board.black_player_pos)

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
