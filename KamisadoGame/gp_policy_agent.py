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


def kamisado_simulator(p1_play_move, p2_play_move):
    board = Kamisado()
    players = [p1_play_move, p2_play_move]
    none_count = 0
    i = 0
    # while not board.is_game_won() and none_count < 10:
    for i in range(100):
        play_move = players[i % len(players)]
        move_tuple = play_move(board)
        move_tuple = move_tuple if move_tuple != () else random_player.play(board)
        tower, move = move_tuple
        board = board.move_tower(tower, move)
        none_count = none_count + 1 if not move else none_count
        if board.is_game_won() and none_count >= 10:
            break
    return board.is_game_won(), i


def evalSolver(individual, games=10):
    start = timeit.default_timer()
    gp_policy = toolbox.compile(expr=individual)

    ea_play_move = get_gp_play_move(gp_policy)
    # print(individual)
    games_won = 0
    won_moves_count = 0
    lose_moves_count = 0
    for i in range(games):
        res, moves_count = kamisado_simulator(ea_play_move, random_player.play)
        if res == Player.WHITE:
            games_won += 1
            won_moves_count += moves_count
        else:
            lose_moves_count += moves_count

        res, moves_count = kamisado_simulator(random_player.play, ea_play_move)
        if res == Player.BLACK:
            games_won += 1
            won_moves_count += moves_count
        else:
            lose_moves_count += moves_count
    end = timeit.default_timer()
    # print('\rEval time {0:.5} sec'.format(str(end - start)), end='')
    # return score_board[str(individual)],
    # return games_won, won_moves_count
    return lose_moves_count / won_moves_count, games_won


def getMove(move_tuple):
    return move_tuple

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSetTyped("main", [Kamisado, tuple], float)
pset.addPrimitive(getBoard, [Kamisado], Kamisado)
pset.addPrimitive(getTotalTowerProgress, [Kamisado], float)
pset.addPrimitive(getMoveTowerProgress, [Kamisado, tuple], float)
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

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
seed = 0
max_ = 5
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSolver)
# toolbox.register("select", selAgentTournament, tournsize=5)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=2, max_=max_)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_))

wins_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
move_count_stats = tools.Statistics(lambda ind: ind.fitness.values[1])
mstats = tools.MultiStatistics(wins_stats=wins_stats, games_won=move_count_stats)
# mstats = wins_stats
mstats.register("Avg", np.mean)
mstats.register("Std", np.std)
mstats.register("Median", np.median)
mstats.register("Min", np.min)
mstats.register("Max", np.max)

pop = toolbox.population(n=1000)
hof = tools.HallOfFame(1)

random.seed(seed)
pop, logbook = algorithms.eaSimple(pop, toolbox, 0.7, 0.01, 100, stats=mstats,
                                   halloffame=hof, verbose=True)
print(hof[0])
