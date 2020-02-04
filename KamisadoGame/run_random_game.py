import os
import random
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


def kamisado_simulator(p1_play_move, p2_play_move, max_steps_num=10000, init_board=None):
    board = Kamisado(init_board=init_board)
    players = [p1_play_move, p2_play_move]
    none_count = 0
    i = 0
    # while not board.is_game_won() and none_count < 10:
    start = timeit.default_timer()
    for i in range(max_steps_num):
        play_move = players[i % len(players)]
        move_tuple = play_move(board)
        # move_tuple = move_tuple if move_tuple != () else random_player.play(board)
        tower, move = move_tuple
        board = board.move_tower(tower, move)
        none_count = none_count + 1 if not move else 0
        if board.is_game_won() or none_count >= 3:
            break
    end = timeit.default_timer()
    # print(f'game time {end-start} sec')
    return board, i


def get_score_for_two_players(p1_move, p2_move, games_count=100, init_board=None):
    score = np.array([0, 0, 0])
    for i in range(games_count):
        board, moves_count = kamisado_simulator(p1_move, p2_move, init_board=init_board)
        res = board.is_game_won()
        if res == Player.WHITE:
            score[0] += 1
        elif res == Player.BLACK:
            score[2] += 1
        else:
            score[1] += 1

        board, moves_count = kamisado_simulator(p2_move, p1_move, init_board=init_board)
        res = board.is_game_won()
        if res == Player.WHITE:
            score[2] += 1
        elif res == Player.BLACK:
            score[0] += 1
        else:
            score[1] += 1
    return score


seen_board = defaultdict(dict)

sp = StrikingPositionAgent(0)
pm = PossibleMovesAgent(0)
tp = TowerProgressAgent(0)
rp = RandomAgent()

players = [sp, pm, tp, rp]
test_rows = []
games_count = 1
print('############################Test Data###################################')


for i, p1 in enumerate(players):
    for p2 in players[i+1:]:
        row = []
        print(f'{p1.name} VS {p2.name}')
        for j in range(100):
            init_board = list(range(8))
            random.shuffle(init_board)
            score = get_score_for_two_players(p1.play, p2.play, games_count, init_board=init_board)
            row.append(score)
        score_sum = np.array(row).sum(axis=0)
        print(f'{p1.name} VS {p2.name} score: {score_sum}, {score_sum / 200}')
