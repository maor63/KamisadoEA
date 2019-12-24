import time
from unittest import TestCase

from KamisadoGame.kamisado import Kamisado
from KamisadoGame.random_agent import RandomAgent


class TestRandomAgent(TestCase):
    def test_play(self):
        board = Kamisado()
        p1 = RandomAgent()
        p2 = RandomAgent()
        players = [p1, p2]
        none_count = 0
        i = 0
        while not board.is_game_won() and none_count < 10:
            player = players[i % len(players)]
            tower, move = player.play(board)
            board = board.move_tower(tower, move)
            none_count = none_count + 1 if not move else none_count
            print(board)
            time.sleep(5)
        print(f'Player {board.is_game_won()} won')
        print(board)

    def test_play_reach_tie(self):
        board = Kamisado()
        board.black_player_pos.update({"Orange": (0, 0), "Blue": (0, 1), "Purple": (0, 2), "Pink": (0, 3),
                                       "Yellow": (0, 4), "Red": (0, 5), "Green": (0, 6), "Brown": (0, 7)})
        board.white_player_pos.update({"Orange": (1, 0), "Blue": (1, 1), "Purple": (1, 2), "Pink": (1, 3),
                                       "Yellow": (1, 4), "Red": (1, 5), "Green": (1, 6), "Brown": (1, 7)})
        board.tower_pos_set = set(board.black_player_pos.values()) | set(board.white_player_pos.values())
        p1 = RandomAgent()
        p2 = RandomAgent()
        players = [p1, p2]
        none_count = 0
        i = 0
        while not board.is_game_won() and none_count < 10:
            player = players[i % len(players)]
            tower, move = player.play(board)
            board = board.move_tower(tower, move)
            none_count = none_count + 1 if not move else none_count
        self.assertEqual(none_count, 10)
        print(f'Player {board.is_game_won()} won')
        print(board)
