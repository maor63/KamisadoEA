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


def if_then_else(condition, out1, out2):
    return out1 if condition else out2


def orderByTowerProgress(board, moves_tuples):
    assert isinstance(board, Kamisado)
    player = board.current_player
    possible_moves = board.get_possible_moves()
    ordered_moves = Counter()
    if moves_tuples:
        for move_tuple in moves_tuples:
            tower, move = move_tuple
            new_board = board.move_tower(tower, move)
            tower_places = np.zeros((8, 8))
            player_positions = new_board.player_pos[player]
            for pos in player_positions.values():
                tower_places[pos] = 1
            if Player.WHITE == player:
                ordered_moves[(tower, move)] = tower_places[:4].sum()
            else:
                ordered_moves[(tower, move)] = tower_places[4:].sum()
        ordered_moves_tuples = [x[0] for x in ordered_moves.most_common(len(ordered_moves))]
        return ordered_moves_tuples
    else:
        return moves_tuples



# def orderByStrikingPostionCount(board):
#     try:
#         assert isinstance(board, Kamisado)
#         player = board.current_player
#         possible_moves = board.get_possible_moves()
#         ordered_moves = Counter()
#         for tower, moves_tuples in possible_moves.items():
#             for move in moves_tuples:
#                 new_board = board.move_tower(tower, move)
#                 new_board.current_player = player
#                 new_board.tower_can_play = ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]
#                 player_possible_moves = new_board.get_possible_moves()
#                 all_moves = list(chain(*[moves for moves in player_possible_moves.values()]))
#                 if Player.WHITE == player:
#                     ordered_moves[(tower, move)] = sum([1 for move in all_moves if move is not None and move[0] == 0])
#                 else:
#                     ordered_moves[(tower, move)] = sum([1 for move in all_moves if move is not None and move[0] == 7])
#         moves_tuples = [x[0] for x in ordered_moves.most_common(len(ordered_moves))]
#         return moves_tuples
#     except Exception as e:
#         print(e)

def orderByOpenStrikingPostions(board, moves_tuples):
    assert isinstance(board, Kamisado)
    player = board.current_player
    ordered_moves = Counter()
    if moves_tuples:
        for move_tuple in moves_tuples:
            tower, move = move_tuple
            new_board = board.move_tower(tower, move)
            tower_places = np.zeros((8, 8))
            if Player.WHITE == player:
                for pos in new_board.black_player_pos.values():
                    tower_places[pos] = 1
                ordered_moves[move_tuple] = tower_places[:1].sum()
            else:
                for pos in new_board.white_player_pos.values():
                    tower_places[pos] = 1
                ordered_moves[move_tuple] = tower_places[-1:].sum()
        ordered_moves_tuples = [x[0] for x in ordered_moves.most_common(len(ordered_moves))]
        return ordered_moves_tuples
    else:
        return moves_tuples


def orderByNumOfPossibleMoves(board):
    assert isinstance(board, Kamisado)
    player = board.current_player
    possible_moves = board.get_possible_moves()
    ordered_moves = Counter()
    for tower, moves_tuples in possible_moves.items():
        for move in moves_tuples:
            new_board = board.move_tower(tower, move)
            new_board.current_player = player
            new_board.tower_can_play = ["Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange"]
            player_possible_moves = new_board.get_possible_moves()
            all_moves = list(chain(*[moves for moves in player_possible_moves.values()]))
            ordered_moves[(tower, move)] = len(all_moves)
    moves_tuples = [x[0] for x in ordered_moves.most_common(len(ordered_moves))]
    return moves_tuples


def getPossibleMoves(board):
    assert isinstance(board, Kamisado)
    player_possible_moves = board.get_possible_moves()
    possible_moves = []
    for tower, moves in player_possible_moves.items():
        for move in moves:
            possible_moves.append((tower, move))
    return possible_moves


def getMax(moves_tuples):
    if moves_tuples:
        return moves_tuples[0]
    else:
        return ()


def getMin(moves_tuples):
    if moves_tuples:
        return moves_tuples[-1]
    else:
        return ()


def getMedian(moves_tuples):
    if moves_tuples:
        return moves_tuples[len(moves_tuples) // 2]
    else:
        return ()


def isThereWinMove(moves_tuples):
    if moves_tuples and None not in set(list(zip(*moves_tuples))[1]):
        return any(get_win_moves(moves_tuples))
    else:
        return False


def get_win_moves(moves_tuples):
    if moves_tuples:
        return [(tower, (y, x)) for (tower, (y, x)) in moves_tuples if y == 0 or y == 7]
    else:
        return []


def isLostMove(move_tuple, board):
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


def getRandomMove(moves_tuples):
    if moves_tuples:
        return random.choice(moves_tuples)
    else:
        return ()


def getWinMoveOrFirst(moves_tuples):
    try:
        if moves_tuples:
            if isThereWinMove(moves_tuples):
                return get_win_moves(moves_tuples)[0]
            else:
                return moves_tuples[0]
        else:
            return tuple()
    except Exception as e:
        print(e)


def removeLostMoves(move_tuples, board):
    try:
        new_move_tuples = [move_tuple for move_tuple in move_tuples if not isLostMove(move_tuple, board)]
        if new_move_tuples:
            return new_move_tuples
        else:
            return move_tuples
    except Exception as e:
        print(e)


score_board = defaultdict(int)
BOARD = Kamisado()
random_player = RandomAgent()


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


def evalSolver(individual):
    start = timeit.default_timer()
    ea_play_move = toolbox.compile(expr=individual)
    # print(individual)
    games_won = 0
    won_moves_count = 0
    for i in range(10):
        res, moves_count = kamisado_simulator(ea_play_move, random_player.play)
        if res == Player.WHITE:
            games_won += 1
            won_moves_count += moves_count

        res, moves_count = kamisado_simulator(random_player.play, ea_play_move)
        if res == Player.BLACK:
            games_won += 1
            won_moves_count += moves_count
    end = timeit.default_timer()
    # print('\rEval time {0:.5} sec'.format(str(end - start)), end='')
    # return score_board[str(individual)],
    return games_won, won_moves_count
    # return games_won,


def getBoard(board):
    return board


def selAgentTournament(individuals, k, tournsize):
    chosen = []
    id_to_ind = {id(ind): ind for ind in individuals}
    for i in range(k):
        lead_board = Counter()
        aspirants = selRandom(individuals, tournsize)
        for p1, p2 in combinations(aspirants, 2):
            p1_play_move = toolbox.compile(expr=p1)
            p2_play_move = toolbox.compile(expr=p2)
            res1 = kamisado_simulator(p1_play_move, p2_play_move)
            res2 = kamisado_simulator(p2_play_move, p1_play_move)
            score1 = get_score_from_result(res1)
            score2 = get_score_from_result(res2)
            lead_board[id(p1)] = score1[0] + score2[1]
            lead_board[id(p2)] = score1[0] + score2[0]
        chosen.append(id_to_ind[lead_board.most_common(1)[0][0]])
        for ind in individuals:
            score_board[str(ind)] = lead_board[id(ind)]
    return chosen


def get_score_from_result(res1):
    if res1 is None:
        score = [1, 1]
    elif res1 == Player.WHITE:
        score = [3, 0]
    else:
        score = [0, 3]
    return score


pset = gp.PrimitiveSetTyped("main", [Kamisado], tuple)
pset.addPrimitive(getBoard, [Kamisado], Kamisado)
pset.addPrimitive(orderByTowerProgress, [Kamisado, list], list)
# pset.addPrimitive(orderByStrikingPostionCount, [Kamisado], list)
pset.addPrimitive(orderByOpenStrikingPostions, [Kamisado, list], list)
# pset.addPrimitive(orderByNumOfPossibleMoves, [Kamisado], list)
pset.addPrimitive(getPossibleMoves, [Kamisado], list)
pset.addPrimitive(getMax, [list], tuple)
pset.addPrimitive(getMin, [list], tuple)
pset.addPrimitive(getMedian, [list], tuple)
pset.addPrimitive(isThereWinMove, [list], bool)
pset.addPrimitive(isLostMove, [tuple, Kamisado], bool)
pset.addPrimitive(isWinMove, [tuple], bool)
pset.addPrimitive(getRandomMove, [list], tuple)
pset.addPrimitive(getWinMoveOrFirst, [list], tuple)
pset.addPrimitive(removeLostMoves, [list, Kamisado], list)
pset.addPrimitive(if_then_else, [bool, list, list], list)
pset.addPrimitive(if_then_else, [bool, tuple, tuple], tuple)

pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.xor, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

pset.addTerminal(True, bool)
pset.addTerminal(False, bool)
pset.addTerminal([], list)
pset.addTerminal((), tuple)
pset.renameArguments(ARG0="Board")
# pset.addTerminal(Kamisado(), Kamisado)

creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
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
mstats = tools.MultiStatistics(wins_stats=wins_stats, move_count_stats=move_count_stats)
# mstats = wins_stats
mstats.register("Avg", np.mean)
mstats.register("Std", np.std)
mstats.register("Median", np.median)
mstats.register("Min", np.min)
mstats.register("Max", np.max)

pop = toolbox.population(n=10)
hof = tools.HallOfFame(1)

pop, logbook = algorithms.eaSimple(pop, toolbox, 0.7, 0.01, 100, stats=mstats,
                                   halloffame=hof, verbose=True)
print(hof[0])
