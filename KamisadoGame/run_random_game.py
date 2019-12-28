import time
import timeit
import numpy as np
from KamisadoGame.kamisado import Kamisado
from KamisadoGame.random_agent import RandomAgent

p1 = RandomAgent()
p2 = RandomAgent()

players = [p1, p2]
times = []
for j in range(10000):
    start = timeit.default_timer()
    pass
    none_count = 0
    i = 0
    board = Kamisado()
    while not board.is_game_won() and none_count < 10:
        player = players[i % len(players)]
        tower, move = player.play(board)
        board = board.move_tower(tower, move)
        none_count = none_count + 1 if not move else none_count
        # print(board)
    print(f'Player {board.is_game_won()} won')
    end = timeit.default_timer()
    times.append(end-start)
print(f'Avg game time {np.mean(times)} sec')
print(f'Sum game time {np.sum(times)} sec')