import copy
import operator
import random
import timeit
from collections import Counter, defaultdict
import numpy
from itertools import chain, combinations
from functools import partial
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.tools import selRandom

from KamisadoGame.kamisado import Kamisado, Player
from KamisadoGame.random_agent import RandomAgent


def getBoard(board):
    return board


def if_then_else(condition, out1, out2):
    return out1 if condition else out2


def getOpenEndPostionsCount(board):
    assert isinstance(board, Kamisado)
    player = board.current_player
    tower_places = np.zeros((8, 8))
    if Player.WHITE == player:
        for pos in board.black_player_pos.values():
            tower_places[pos] = 1
        open_end_position = tower_places[:1].sum()
    else:
        for pos in board.white_player_pos.values():
            tower_places[pos] = 1
        open_end_position = tower_places[-1:].sum()
    return open_end_position


def moveSrikingPositionCount(board, move_tuple):
    assert isinstance(board, Kamisado)
    if move_tuple:
        new_board = board.move_tower(*move_tuple)
        new_board.current_player = board.current_player
        possible_moves = getPossibleMoves(new_board)
        possible_moves = [(tower, move) for (tower, move) in possible_moves if move is not None]
        return len(get_win_moves(possible_moves))
    else:
        return 0


def getTotalTowerProgress(board):
    assert isinstance(board, Kamisado)
    tower_places = np.zeros((8, 8))
    player = board.current_player
    player_positions = board.players_pos[player]
    for pos in player_positions.values():
        tower_places[pos] = 1
    if Player.WHITE == player:
        tower_progress = tower_places[:4].sum()
    else:
        tower_progress = tower_places[4:].sum()
    return tower_progress


def getMoveTowerProgress(board, move_tuple):
    assert isinstance(board, Kamisado)
    tower_places = np.zeros((8, 8))
    player = board.current_player
    if move_tuple[1] is None:
        move_tuple = (move_tuple[0], board.players_pos[player][move_tuple[0]])
    tower, (tower_y, tower_x) = move_tuple
    if Player.WHITE == player:
        tower_progress_frec = (7 - tower_y) / 7
    else:
        tower_progress_frec = tower_y / 7
    return tower_progress_frec


def getPossibleMovesCount(board):
    assert isinstance(board, Kamisado)
    return len(getPossibleMoves(board))


def getPossibleMoves(board):
    assert isinstance(board, Kamisado)
    player_possible_moves = board.get_possible_moves()
    possible_moves = []
    for tower, moves in player_possible_moves.items():
        for move in moves:
            possible_moves.append((tower, move))
    return possible_moves


def getEnemyPossibleMovesCount(board, move_tuple):
    assert isinstance(board, Kamisado)
    tower, pos = move_tuple
    new_board = board.move_tower(tower, pos)
    return getPossibleMovesCount(new_board)


def isThereWinMove(moves_tuples):
    if moves_tuples and None not in set(list(zip(*moves_tuples))[1]):
        return len(get_win_moves(moves_tuples)) > 0
    else:
        return False


def get_win_moves(moves_tuples):
    if moves_tuples:
        return [(tower, (y, x)) for (tower, (y, x)) in moves_tuples if y == 0 or y == 7]
    else:
        return []


def isLostMove(board, move_tuple):
    assert isinstance(board, Kamisado)
    if move_tuple:
        tower, move = move_tuple
        new_board = board.move_tower(tower, move)
        return isThereWinMove(getPossibleMoves(new_board))
    else:
        return False


def isWinMove(move_tuple):
    if move_tuple:
        return isThereWinMove([move_tuple])
    else:
        return False


random_player = RandomAgent()


def get_gp_play_move(gp_policy):
    def gp_play_move(board):
        assert isinstance(board, Kamisado)
        moves_ranks = Counter({move_tuple: gp_policy(board, move_tuple) for move_tuple in getPossibleMoves(board)})
        selected_move = moves_ranks.most_common(1)[0][0]
        return selected_move

    return gp_play_move


def kamisado_simulator(p1_play_move, p2_play_move, max_steps_num=10000):
    board = Kamisado()
    players = [p1_play_move, p2_play_move]
    none_count = 0
    i = 0
    # while not board.is_game_won() and none_count < 10:
    for i in range(max_steps_num):
        play_move = players[i % len(players)]
        move_tuple = play_move(board)
        move_tuple = move_tuple if move_tuple != () else random_player.play(board)
        tower, move = move_tuple
        board = board.move_tower(tower, move)
        none_count = none_count + 1 if not move else none_count
        if board.is_game_won() or none_count >= 10:
            break
    return board.is_game_won(), i


def evalSolver(individual, games=5):
    start = timeit.default_timer()
    gp_policy = toolbox.compile(expr=individual)

    ea_play_move = get_gp_play_move(gp_policy)
    # print(individual)
    games_won = 0
    games_lost = 0
    games_tie = 0
    won_moves_count = []
    lose_moves_count = []
    max_steps_num = 100
    for i in range(games):
        res, moves_count = kamisado_simulator(ea_play_move, random_player.play, max_steps_num)
        if res == Player.WHITE:
            games_won += 1
            won_moves_count.append(moves_count)
        elif res == Player.BLACK:
            games_lost += 1
            lose_moves_count.append(moves_count)
        else:
            games_tie += 1

        res, moves_count = kamisado_simulator(random_player.play, ea_play_move, max_steps_num)
        if res == Player.BLACK:
            games_won += 1
            won_moves_count.append(moves_count)
        elif res == Player.WHITE:
            games_lost += 1
            lose_moves_count.append(moves_count)
        else:
            games_tie += 1

    end = timeit.default_timer()
    # print('\rEval time {0:.5} sec'.format(str(end - start)), end='')
    # return score_board[str(individual)],
    # return games_won, won_moves_count
    won_moves_avg = np.mean(won_moves_count) if won_moves_count else max_steps_num
    lose_moves_avg = np.mean(lose_moves_count) if lose_moves_count else max_steps_num
    tree_length = len(individual)
    return games_won, won_moves_avg, lose_moves_avg, tree_length, games_tie


def getMove(move_tuple):
    return move_tuple


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def selAgentTournament(individuals, k, tournsize):
    chosen = []
    for i in range(k):
        aspirants = random.sample(individuals, tournsize)
        while len(aspirants) > 1:
            random.shuffle(aspirants)
            p1, p2 = aspirants.pop(), aspirants.pop()
            p1_move = get_playe_move_from_policy(p1)
            p2_move = get_playe_move_from_policy(p2)
            res, moves_count = kamisado_simulator(p1_move, p2_move)
            if res == Player.WHITE:
                aspirants.append(p1)
            else:
                aspirants.append(p2)
        chosen += aspirants
    return chosen


def get_playe_move_from_policy(p1):
    gp_policy = toolbox.compile(expr=p1)
    p1_move = get_gp_play_move(gp_policy)
    return p1_move


def get_score_for_two_players(p1_move, p2_move):
    score = [0, 0, 0]
    for i in range(100):
        res, moves_count = kamisado_simulator(p1_move, p2_move)
        if res == Player.WHITE:
            score[0] += 1
        elif res == Player.BLACK:
            score[2] += 1
        else:
            score[1] += 1

        res, moves_count = kamisado_simulator(p2_move, p1_move)
        if res == Player.WHITE:
            score[2] += 1
        elif res == Player.BLACK:
            score[0] += 1
        else:
            score[1] += 1
    return score


pset = gp.PrimitiveSetTyped("main", [Kamisado, tuple], float)
pset.addPrimitive(getBoard, [Kamisado], Kamisado)
pset.addPrimitive(getTotalTowerProgress, [Kamisado], float)
pset.addPrimitive(getMoveTowerProgress, [Kamisado, tuple], float)
pset.addPrimitive(moveSrikingPositionCount, [Kamisado, tuple], float)
pset.addPrimitive(getPossibleMovesCount, [Kamisado], float)
pset.addPrimitive(getEnemyPossibleMovesCount, [Kamisado, tuple], float)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.le, [float, float], bool)
pset.addPrimitive(isLostMove, [Kamisado, tuple], bool)
pset.addPrimitive(isWinMove, [tuple], bool)
pset.addPrimitive(getMove, [tuple], tuple)
pset.addPrimitive(if_then_else, [bool, float, float], float)

pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.xor, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(operator.neg, [float], float)

pset.addTerminal(True, bool)
pset.addTerminal(False, bool)
pset.addTerminal([], list)
# pset.addTerminal((), tuple)
for i in range(11):
    pset.addTerminal(i / 10, float)
    pset.addTerminal(i, float)
pset.renameArguments(ARG0="Board")
pset.renameArguments(ARG1="move_tuple")
# pset.addTerminal(Kamisado(), Kamisado)

creator.create("FitnessMax", base.Fitness, weights=(2.0, -1.0, 1.0, -1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
seed = 12
max_ = 7
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSolver)
toolbox.register("select", selAgentTournament, tournsize=2)
# toolbox.register("select", tools.selTournament, tournsize=6)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=2, max_=max_)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_))

games_won = tools.Statistics(lambda ind: ind.fitness.values[0])
avg_move_to_win = tools.Statistics(lambda ind: ind.fitness.values[1])
# avg_move_to_lose = tools.Statistics(lambda ind: ind.fitness.values[2])
# tree_len = tools.Statistics(lambda ind: ind.fitness.values[3])
avg_game_tie = tools.Statistics(lambda ind: ind.fitness.values[4])
# mstats = tools.MultiStatistics(games_won=games_won, avg_move_to_win=avg_move_to_win, avg_move_to_lose=avg_move_to_lose, tree_len=tree_len)
mstats = tools.MultiStatistics(games_won=games_won, avg_move_to_win=avg_move_to_win, avg_game_tie=avg_game_tie)
# mstats = wins_stats
mstats.register("Avg", np.mean)
mstats.register("Std", np.std)
mstats.register("Median", np.median)
mstats.register("Min", np.min)
mstats.register("Max", np.max)

pop = toolbox.population(n=100)
hof = tools.HallOfFame(3)

random.seed(seed)
pop, logbook = algorithms.eaSimple(pop, toolbox, 0.7, 0.001, 100, stats=mstats,
                                   halloffame=hof, verbose=True)
print(hof[0])
p1_move = get_playe_move_from_policy(hof[0])
p2_move = get_playe_move_from_policy(hof[1])

score1 = get_score_for_two_players(p1_move, p1_move)
print(f'p1 VS p1 score: {score1}')

score2 = get_score_for_two_players(p1_move, p2_move)
print(f'p1 VS p2 score: {score2}')
