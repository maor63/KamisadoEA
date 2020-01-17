import os
import time
import timeit
from multiprocessing import freeze_support

import numpy as np
from collections import defaultdict
from KamisadoGame.possible_moves_agent import PossibleMovesAgent
from KamisadoGame.possible_striking_agent import PossibleStrikingAgent
from KamisadoGame.tower_progress_agent import TowerProgressAgent
from KamisadoGame.kamisado import Kamisado, Player
from KamisadoGame.random_agent import RandomAgent
from KamisadoGame.striking_position_agent import StrikingPositionAgent

seen_board = defaultdict(dict)

p1 = StrikingPositionAgent(0)
p2 = PossibleMovesAgent(0)

players = [p1, p2]
times = []
score = [0, 0, 0]
for j in range(100):
    start = timeit.default_timer()
    pass
    none_count = 0
    i = 0
    board = Kamisado()
    while not board.is_game_won() and none_count < 10:
        if (str(board), board.current_player.value) not in seen_board:
            possible_move_dict, _ = board._calc_possible_move_dict()
            seen_board[(str(board), board.current_player.value)] = possible_move_dict
        else:
            print('load')
        board.possible_moves_dict = seen_board[(str(board), board.current_player.value)]

        player = players[i % len(players)]
        tower, move = player.play(board)
        board = board.move_tower(tower, move)
        none_count = none_count + 1 if not move else none_count
        i += 1
        # print(board)
    # print(f'Player {board.is_game_won()} won')
    if board.is_game_won() == Player.WHITE:
        score[0] += 1
    elif board.is_game_won() == Player.BLACK:
        score[2] += 1
    else:
        score[1] += 1
    end = timeit.default_timer()
    times.append(end - start)

# players.reverse()
# for j in range(10):
#     start = timeit.default_timer()
#     pass
#     none_count = 0
#     i = 0
#     board = Kamisado()
#     while not board.is_game_won() and none_count < 10:
#         player = players[i % len(players)]
#         tower, move = player.play(board)
#         board = board.move_tower(tower, move)
#         none_count = none_count + 1 if not move else none_count
#         i += 1
#         # print(board)
#     # print(f'Player {board.is_game_won()} won')
#     if board.is_game_won() == Player.WHITE:
#         score[0] += 1
#     elif board.is_game_won() == Player.BLACK:
#         score[2] += 1
#     else:
#         score[1] += 1
#     end = timeit.default_timer()
#     times.append(end - start)
print(f'Avg game time {np.mean(times)} sec')
print(f'Sum game time {np.sum(times)} sec')
print(score)


os.listdir()